"""
Prompt ingestion on a second thread (Generator.prefill_thread, Generator._prefill_round) against
the toy model of test_stagger_decode.

No GPU or model is required. The generator's real prefill round, decode step and scheduler run
over the toy model; the toy job ingests its prompt in chunks through the same coroutine as Job,
and turns into assertions what a chunk on another thread must not meet: a checkpoint, a decode
step, a second chunk or the release of its pages while its forward passes run. A chunk's passes
can be made to last a given number of decode steps, so that the two threads really overlap.
"""

import threading
import time
from contextlib import contextmanager

import pytest
import torch

import test_stagger_decode as toy
from test_stagger_decode import GEN, PROMPTS, ToyJob, _host_torch, make_generator   # noqa: F401

MAIN = "MainThread"


class FakeLane:
    """What the generator needs from a Lane, with no CUDA device behind it."""

    made = []

    def __init__(self, index, devices):
        self.index = index
        self.devices = list(devices)
        self.entered = []
        self.synced = []
        self.passes = 0
        self.in_pass = 0
        FakeLane.made.append(self)

    def enter(self):
        self.entered.append(threading.current_thread().name)

    @contextmanager
    def passing(self):
        self.in_pass += 1
        self.passes += 1
        try:
            yield
        finally:
            self.in_pass -= 1

    def synchronize(self):
        assert self.in_pass == 1, "the lane is synchronized inside its pass"
        self.synced.append(threading.current_thread().name)


@pytest.fixture(autouse = True)
def _fake_lane(monkeypatch):
    FakeLane.made = []
    monkeypatch.setattr(GEN, "Lane", FakeLane)


class Slots:
    """The toy model's CPU expert host, with the slots of the real one."""

    def __init__(self, n = 4):
        self.n = n
        self.passes = 0
        self.split = None

    def begin_pass(self):
        self.passes += 1

    def split_lanes(self, decode_slots = 2):
        d = min(decode_slots, self.n - 1)
        if d < 1:
            return None
        self.split = (d, self.n - d)
        return self.split


def lasting(steps):
    """A chunk whose forward passes last the given number of decode steps (counted on the
    model's log of module calls), or until nobody is decoding. A test that is about to make
    the generator wait for the chunk says so first (job.hurry): the passes then end a little
    later, as kernels already issued would."""
    def on_chunk(job, a, b):
        if threading.current_thread().name == MAIN:
            return      # inline: the decode cannot advance during the chunk, that is the point
        g = job.generator
        # Nothing is decoded beside a chunk shorter than the generator's minimum, or beside a
        # decode batch larger than its maximum: those chunks are waited for
        rows = sum(1 + g.num_draft_tokens for j in list(g.active_jobs) if j.is_prefill_done())
        if b - a < g.side_min_tokens or rows > g.side_max_rows:
            return
        heads = lambda: sum(1 for idx, _ in job.model.log if idx == 7)
        start = heads()
        deadline = time.time() + 10
        while heads() < start + steps:
            if job.hurry.is_set():
                time.sleep(0.02)
                return
            if not any(j.is_prefill_done() for j in list(g.active_jobs)):
                return
            assert time.time() < deadline, "the decode does not advance beside the chunk"
            time.sleep(0.0005)
    return on_chunk


def long_prompt(seed, n):
    return [(seed * 7 + i * 13) % toy.VOCAB for i in range(n)]


def drive(thread, plan, script = None, draft = True, stagger = False, slots = 4, min_jobs = 2,
          min_tokens = 16, max_rows = 8):
    """
    Run jobs to the end. plan: {name: dict(prompt, max_new, at, chunk, steps)}: the job is
    enqueued before iteration `at`, ingests `chunk` tokens per chunk and each chunk lasts
    `steps` decode steps when it runs beside a decode. Returns (tokens per job, generator, jobs,
    the stage and job of every result in order).
    """
    g = make_generator(draft = draft, stagger = stagger, min_jobs = min_jobs)
    g.model.host = Slots(slots)
    g.model.config.moe_cpu_hosts = {"cuda:0": g.model.host}
    g.prefill_thread = thread
    g.side_min_tokens = min_tokens
    g.side_max_rows = max_rows
    jobs = {}
    events = []
    seen = {}
    n = 0
    pending = dict(plan)
    try:
        while pending or g.num_remaining_jobs():
            for name in [k for k, v in pending.items() if v.get("at", 0) <= n]:
                spec = pending.pop(name)
                job = ToyJob(g.model, name, spec["prompt"], spec.get("max_new", 40), chunk = spec.get("chunk", 0))
                job.hurry = threading.Event()
                if spec.get("steps"):
                    job.on_chunk = lasting(spec["steps"])
                jobs[name] = job
                g.enqueue(job)
            if script is not None:
                script(g, jobs, n)
            for r in g.iterate():
                events.append((r["stage"], r["job"].name))
                if r["stage"] == "streaming":
                    seen.setdefault(r["job"].name, []).append(r["token"])
            n += 1
            assert n < 20000, "the generator does not finish"
    finally:
        if g._side_pool is not None:
            g._side_pool.shutdown(wait = True)
    assert g._side is None
    for name, job in jobs.items():
        if not job.cancelled:
            assert seen.get(name, []) == job.out, f"{name}: results differ from what the job sampled"
    return {name: job.out for name, job in jobs.items()}, g, jobs, events


PLAN = {
    "a": dict(prompt = PROMPTS["a"], max_new = 400),
    "b": dict(prompt = long_prompt(3, 65), max_new = 30, at = 6, chunk = 16, steps = 5),
}


# --------------------------------------------------------------------------------------------
#   Same tokens, and the decode goes on
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("draft", [True, False])
def test_tokens_with_the_prefill_thread_are_the_regular_ones(draft):
    ref, g0, jobs0, _ = drive(False, PLAN, draft = draft)
    out, g, jobs, _ = drive(True, PLAN, draft = draft)
    assert out == ref
    assert len(out["a"]) == 400 and len(out["b"]) == 30
    # Off: the chunks ran on the caller's thread and no thread was made
    assert set(jobs0["b"].chunk_threads) == {MAIN} and g0._side_pool is None
    assert g0.side_stats["chunks"] == 0
    # On: 64 prompt tokens in chunks of 16, all of them on the prefill thread, and so is the
    # prompt of a, ingested with nobody decoding
    assert len(jobs["b"].chunk_threads) == 4
    assert all(t.startswith("exl3-prefill") for j in jobs.values() for t in j.chunk_threads)
    assert g.side_stats["chunks"] == 4 + 1 and g.side_stats["errors"] == 0


def test_the_decode_goes_on_beside_a_chunk():
    _, g, jobs, events = drive(True, PLAN)
    stats = g.side_stats
    assert stats["beside"] == 4
    # Each chunk lasted five decode steps, and the round that finds it done is not one of them
    assert stats["decode_steps"] >= 4 * 5
    # Between the first and the last record of b's prompt, a kept receiving tokens
    first = events.index(("prefill", "b"))
    last = len(events) - 1 - events[::-1].index(("prefill", "b"))
    between = events[first:last]
    assert between.count(("streaming", "a")) >= 3 * 5
    assert events.count(("prefill", "b")) == 4


def test_without_the_thread_a_chunk_stops_the_decode():
    # What the feature changes, seen on the same plan: inline, every round with a chunk in it
    # takes one decode step and no more
    _, g, jobs, events = drive(False, PLAN)
    first = events.index(("prefill", "b"))
    last = len(events) - 1 - events[::-1].index(("prefill", "b"))
    steps = [e for e in events[first:last] if e[0] == "streaming"]
    assert len({e for e in steps}) == 1 and 3 <= len(steps) <= 3 * 2      # one step per round, one or two tokens each


def test_with_nobody_decoding_the_round_waits_for_its_chunks():
    # Two jobs ingesting and none generating: a chunk each per round, as upstream, but on the
    # prefill thread (the workspaces of a prefill pass exist once)
    plan = {
        "b": dict(prompt = long_prompt(3, 41), max_new = 12, chunk = 16),
        "c": dict(prompt = long_prompt(5, 37), max_new = 12, chunk = 16),
    }
    ref, _, _, ev0 = drive(False, plan)
    out, g, jobs, ev = drive(True, plan)
    assert out == ref
    assert ev == ev0                 # the very same sequence of results as inline
    assert all(t.startswith("exl3-prefill") for j in jobs.values() for t in j.chunk_threads)
    assert g.side_stats["chunks"] == 3 + 3 and g.side_stats["beside"] == 0


# --------------------------------------------------------------------------------------------
#   What a chunk on another thread must not meet
# --------------------------------------------------------------------------------------------

def test_checkpoints_are_offered_between_chunks_never_during_one():
    # The toy job asserts that no checkpoint is offered while a chunk is planned and not
    # recorded; here: that it is offered at every chunk boundary before the next chunk starts
    _, g, jobs, _ = drive(True, PLAN)
    offered = jobs["b"].checkpoints
    for boundary in (16, 32, 48, 64):
        assert boundary in offered


def test_the_thread_has_one_lane_and_a_chunk_that_is_done_has_run():
    _, g, jobs, _ = drive(True, PLAN)
    lane = FakeLane.made[0]
    assert len(FakeLane.made) == 1 and lane.index == 1 and lane.devices == [0, 1]
    # The thread entered the lane once, by itself
    assert len(lane.entered) == 1 and lane.entered[0].startswith("exl3-prefill")
    # Every chunk is a pass of the lane, waited for on the thread before it counts as done
    # (four chunks of b's prompt and the one of a's)
    assert lane.passes == 5 and lane.in_pass == 0
    assert len(lane.synced) == 5 and all(t.startswith("exl3-prefill") for t in lane.synced)
    # The CPU expert worker's slots were split once
    assert g.model.host.split == (2, 2)


def test_a_chunk_shorter_than_a_page_is_waited_for():
    # The tail of a prompt takes the decode-sized paths of the modules, the ones the decode
    # step is on: nothing is decoded beside it. 60 prompt tokens: three chunks of 16 and one of 12
    plan = dict(PLAN, b = dict(PLAN["b"], prompt = long_prompt(3, 61)))
    ref, _, _, _ = drive(False, plan)
    out, g, jobs, events = drive(True, plan)
    assert out == ref
    assert g.side_stats["chunks"] == 4 + 1 and g.side_stats["beside"] == 3
    # The tail still ran on the prefill thread
    assert all(t.startswith("exl3-prefill") for t in jobs["b"].chunk_threads)
    # ... and its record came in the round that started it: no token of a in between
    third = [i for i, e in enumerate(events) if e == ("prefill", "b")][2]
    after = events[third + 1:]
    k = after.index(("prefill", "b"))
    assert len([e for e in after[:k] if e == ("streaming", "a")]) <= 2     # the third chunk's own round


def test_a_decode_batch_too_large_for_the_decode_paths_gets_no_chunk_beside_it():
    # Two jobs generating with a draft token each are 4 rows x tokens; with the limit at 2 the
    # chunks are waited for, as inline
    plan = dict(PLAN, c = dict(prompt = PROMPTS["c"], max_new = 400))
    ref, _, _, _ = drive(False, plan)
    out, g, jobs, _ = drive(True, plan, max_rows = 2)
    assert out == ref
    assert g.side_stats["beside"] == 0 and g.side_stats["decode_steps"] == 0
    assert all(t.startswith("exl3-prefill") for t in jobs["b"].chunk_threads)
    # With the limit where it is, the same plan overlaps
    _, g2, _, _ = drive(True, plan)
    assert g2.side_stats["beside"] == 4


def test_two_jobs_ingesting_take_turns():
    plan = {
        "a": dict(prompt = PROMPTS["a"], max_new = 600),
        "b": dict(prompt = long_prompt(3, 49), max_new = 10, at = 4, chunk = 16, steps = 3),
        "c": dict(prompt = long_prompt(5, 49), max_new = 10, at = 4, chunk = 16, steps = 3),
    }
    order = []
    def script(g, jobs, n):
        for name in ("b", "c"):
            job = jobs.get(name)
            if job is not None and job.on_chunk is not None and not hasattr(job, "_wrapped"):
                inner = job.on_chunk
                def on_chunk(job_, a, b, inner = inner, name = name):
                    order.append(name)
                    inner(job_, a, b)
                job.on_chunk = on_chunk
                job._wrapped = True
    ref, _, _, _ = drive(False, plan)
    out, g, jobs, _ = drive(True, plan, script = script)
    assert out == ref
    assert order == ["b", "c", "b", "c", "b", "c"]
    assert g.side_stats["chunks"] == 6 + 1 and g.side_stats["beside"] == 6


def test_cancel_lets_the_chunk_finish_before_the_pages_go():
    # The toy job asserts that its pages are not released while a chunk of its prompt runs
    done = {}
    def script(g, jobs, n):
        b = jobs.get("b")
        if b is not None and not b.cancelled and g._side is not None and g._side.job is b \
                and len(b.chunk_threads) == 2:
            assert b.chunk_running and not g._side.future.done()
            b.cancelled = True
            b.hurry.set()
            g.cancel(b)
            done["at"] = n
            assert g._side is None and not b.chunk_running and b.released == 1
            assert b not in g.active_jobs
    ref, _, _, _ = drive(False, {"a": PLAN["a"]})
    out, g, jobs, events = drive(True, PLAN, script = script)
    assert "at" in done
    assert out["a"] == ref["a"]                    # the other job never noticed
    assert out["b"] == []
    # The second chunk was not recorded
    assert events.count(("prefill", "b")) == 1


def test_clear_queue_lets_the_chunk_finish_too():
    g = make_generator()
    g.model.host = Slots()
    g.model.config.moe_cpu_hosts = {"cuda:0": g.model.host}
    g.prefill_thread = True
    g.side_min_tokens = 16
    a = ToyJob(g.model, "a", PROMPTS["a"], 200)
    b = ToyJob(g.model, "b", long_prompt(3, 65), 20, chunk = 16)
    b.on_chunk = lasting(4)
    a.hurry, b.hurry = threading.Event(), threading.Event()
    a.droppable = b.droppable = True
    g.enqueue(a)
    for _ in range(3):
        g.iterate()
    g.enqueue(b)
    g.iterate()
    assert g._side is not None and g._side.job is b
    deadline = time.time() + 10
    while not b.chunk_running:          # the thread has the chunk, it may not have started it
        assert time.time() < deadline
        time.sleep(0.0005)
    b.hurry.set()
    g.clear_queue()
    assert g._side is None and not b.chunk_running and b.released == 1
    assert g.num_remaining_jobs() == 0
    g._side_pool.shutdown(wait = True)


def test_a_failed_chunk_fails_its_job_only():
    def script(g, jobs, n):
        b = jobs.get("b")
        if b is not None and not hasattr(b, "_wrapped"):
            inner = b.on_chunk
            def on_chunk(job, a_, b_):
                inner(job, a_, b_)
                if a_ == 16:
                    raise torch.cuda.OutOfMemoryError("no room for the second chunk")
            b.on_chunk = on_chunk
            b._wrapped = True
    ref, _, _, _ = drive(False, {"a": PLAN["a"]})
    out, g, jobs, events = drive(True, PLAN, script = script)
    assert out["a"] == ref["a"]
    assert events.count(("error", "b")) == 1 and events.count(("prefill", "b")) == 1
    assert jobs["b"].released == 1 and jobs["b"] not in g.active_jobs
    assert g.side_stats["errors"] == 1
    # The lane was synchronized after the failure as well: what the pass had issued is done
    # before the pages go (a's prompt, b's first chunk, b's failed one)
    assert len(FakeLane.made[0].synced) == 3


def test_a_chunk_that_fails_at_the_plan_fails_its_job_only():
    def script(g, jobs, n):
        b = jobs.get("b")
        if b is not None and not hasattr(b, "_wrapped"):
            real = b._prefill_co
            def co(results):
                if b.sequences[0].kv_position == 16:
                    raise RuntimeError("no page for the second chunk")
                return (yield from real(results))
            b._prefill_co = co
            b._wrapped = True
    ref, _, _, _ = drive(False, {"a": PLAN["a"]})
    out, g, jobs, events = drive(True, PLAN, script = script)
    assert out["a"] == ref["a"]
    assert events.count(("error", "b")) == 1 and jobs["b"].released == 1


# --------------------------------------------------------------------------------------------
#   The switch
# --------------------------------------------------------------------------------------------

def test_switched_off_while_a_chunk_is_under_way():
    def script(g, jobs, n):
        b = jobs.get("b")
        if b is not None and len(b.chunk_threads) == 2 and g.prefill_thread:
            g.prefill_thread = False
            b.hurry.set()       # the next round waits for the chunk
    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, _ = drive(True, PLAN, script = script)
    assert out == ref
    threads = jobs["b"].chunk_threads
    # The chunk under way is seen through; the rest of the prompt is ingested inline
    assert [t.startswith("exl3-prefill") for t in threads] == [True, True, False, False]
    assert threads[2:] == [MAIN, MAIN]


def test_switched_on_while_a_job_is_ingesting():
    def script(g, jobs, n):
        b = jobs.get("b")
        if b is not None and len(b.chunk_threads) == 1 and not g.prefill_thread:
            g.prefill_thread = True
    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, _ = drive(False, PLAN, script = script)
    assert out == ref
    threads = jobs["b"].chunk_threads
    assert threads[0] == MAIN and all(t.startswith("exl3-prefill") for t in threads[1:])


def test_not_available_falls_back_to_inline(capsys):
    # A worker with a single slot cannot be split between two threads
    out, g, jobs, _ = drive(True, PLAN, slots = 1)
    ref, _, _, _ = drive(False, PLAN)
    assert out == ref
    assert set(jobs["b"].chunk_threads) == {MAIN} and g._side_pool is None
    lines = [l for l in capsys.readouterr().out.splitlines() if "Prefill thread" in l]
    assert lines == [" !! Prefill thread not available: the CPU expert worker has a single slot (EXL3_MOE_CPU_SLOTS)"]


def test_not_available_with_tensor_parallelism(capsys):
    g = make_generator()
    g.model.loaded_tp = True
    g.prefill_thread = True
    assert g._side_plan() is False
    assert g._side_plan() is False
    lines = [l for l in capsys.readouterr().out.splitlines() if "Prefill thread" in l]
    assert lines == [" !! Prefill thread not available: tensor-parallel model"]


def test_the_state_tells_a_server_where_ingestion_runs():
    # Off: as upstream, whatever the model
    out, g, _, _ = drive(False, PLAN)
    assert (g.prefill_thread_state, g.prefill_thread_reason) == ("off", None)

    # Asked for: nothing is set up before the first round, then the thread is at work
    g = make_generator()
    g.prefill_thread = True
    assert (g.prefill_thread_state, g.prefill_thread_reason) == ("pending", None)
    out, g, _, _ = drive(True, PLAN)
    assert (g.prefill_thread_state, g.prefill_thread_reason) == ("on", None)

    # Asked for with a model that cannot: the generator ingests inline and says why
    out, g, _, _ = drive(True, PLAN, slots = 1)
    assert g.prefill_thread_state == "unavailable"
    assert g.prefill_thread_reason == "the CPU expert worker has a single slot (EXL3_MOE_CPU_SLOTS)"

    # Turned off at run time
    g.prefill_thread = False
    assert g.prefill_thread_state == "off"


def test_the_log_says_what_ran_beside_the_decode(capsys):
    _, g, _, _ = drive(True, PLAN)
    lines = [l for l in capsys.readouterr().out.splitlines() if "Prefill thread" in l]
    assert lines[0] == " -- Prefill thread: on, CPU expert worker slots 2 for the decode and 2 for the prefill"
    assert len(lines) == 2
    # One line for the stretch: the chunks, their seconds and the decode steps beside them
    words = lines[1].split()
    assert lines[1].startswith(" -- Prefill thread: 4 chunks in ") and lines[1].endswith(" decode steps beside them")
    assert int(words[-5]) == g.side_stats["decode_steps"] >= 20


def test_with_the_staggered_decode():
    # Three jobs generating, staggered; a fourth one arrives with a long prompt. The decode goes
    # back to one batch while the prompt is ingested (beside it) and is staggered again after
    plan = {
        "a": dict(prompt = PROMPTS["a"], max_new = 300),
        "c": dict(prompt = PROMPTS["c"], max_new = 300),
        "d": dict(prompt = PROMPTS["d"], max_new = 300),
        "b": dict(prompt = long_prompt(3, 65), max_new = 60, at = 9, chunk = 16, steps = 4),
    }
    ref, _, _, _ = drive(False, plan)
    out, g, jobs, _ = drive(True, plan, stagger = True)
    assert out == ref
    assert g.side_stats["beside"] == 4
    assert g.stagger_stats["steps"] > 50 and g.stagger_stats["drains"] == 0
