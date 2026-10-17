"""
Prompt ingestion on a second thread (Generator.prefill_thread, Generator._prefill_round) against
the toy model of test_stagger_decode.

No GPU or model is required. The generator's real prefill round, decode step and scheduler run
over the toy model; the toy job ingests its prompt in chunks through the same coroutine as Job,
and turns into assertions what a chunk on another thread must not meet: a checkpoint, a decode
step, a second chunk or the release of its pages while its forward passes run. A chunk's passes
can be made to last a given number of decode steps, so that the two threads really overlap.
"""

import ast
import os
import threading
import time
import types
from contextlib import contextmanager

import pytest
import torch

import test_stagger_decode as toy
from test_stagger_decode import GEN, PROMPTS, ToyJob, _host_torch, make_generator

# (the toy's autouse fixture: the generator module sees host tensors only)
_FIXTURES = (_host_torch,)

pytestmark = pytest.mark.nogpu

MAIN = "MainThread"
PAGE = toy.PAGE


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
        rows = sum(1 + g.max_draft_tokens for j in list(g.active_jobs) if j.is_prefill_done())
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
          min_tokens = 16, max_rows = 8, setup = None):
    """
    Run jobs to the end. plan: {name: dict(prompt, max_new, at, chunk, steps)}: the job is
    enqueued before iteration `at`, ingests `chunk` tokens per chunk and each chunk lasts
    `steps` decode steps when it runs beside a decode. setup(generator) runs before the first
    iteration. Returns (tokens per job, generator, jobs, the stage and job of every result in
    order).
    """
    g = make_generator(draft = draft, stagger = stagger, min_jobs = min_jobs)
    g.model.host = Slots(slots)
    g.model.config.moe_cpu_hosts = {"cuda:0": g.model.host}
    g.prefill_thread = thread
    g.side_min_tokens = min_tokens
    g.side_max_rows = max_rows
    if setup is not None:
        setup(g)
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


# --------------------------------------------------------------------------------------------
#   The shortest chunk that may run beside a decode
# --------------------------------------------------------------------------------------------

def planned(stream_min_rows = None, floor = None):
    """A generator whose first round has set the prefill thread up, on a CPU expert host that
    takes batches below stream_min_rows as decode-size ones (None: a host that does not say)."""
    g = make_generator()
    g.model.host = Slots()
    if stream_min_rows is not None:
        g.model.host.stream_min_rows = stream_min_rows
    g.model.config.moe_cpu_hosts = {"cuda:0": g.model.host}
    g.prefill_thread = True
    g.side_min_tokens = GEN.SIDE_MIN_TOKENS if floor is None else floor
    try:
        assert g._side_plan()
    finally:
        if g._side_pool is not None:
            g._side_pool.shutdown(wait = True)
    return g


def test_the_floor_is_one_token_past_the_worker_s_decode_size_batches():
    # Below a page: the last chunk of a prompt, and a short prompt, run beside the decode too
    assert GEN.SIDE_MIN_TOKENS == 33
    assert planned(32).side_min_tokens == 33
    assert planned(None).side_min_tokens == 33
    # A host told to take larger batches as decode-size ones raises it; one told smaller does
    # not lower it
    assert planned(128).side_min_tokens == 129
    assert planned(8).side_min_tokens == 33
    # and what was asked for above it stays (EXL3_SIDE_MIN_TOKENS)
    assert planned(32, floor = 256).side_min_tokens == 256


def test_a_chunk_at_the_floor_runs_beside_and_one_below_is_waited_for():
    plan = {
        "a": dict(prompt = PROMPTS["a"], max_new = 300),
        "b": dict(prompt = long_prompt(5, 16 + 15 + 1), max_new = 10, at = 6, chunk = 16, steps = 3),
    }
    ref, _, _, _ = drive(False, plan)
    out, g, jobs, _ = drive(True, plan, min_tokens = 16)
    assert out == ref
    # 31 prompt tokens in a chunk of 16 and one of 15: only the first is long enough
    assert len(jobs["b"].chunk_threads) == 2
    assert g.side_stats["beside"] == 1



# --------------------------------------------------------------------------------------------
#   1.6.2: what else the thread is not offered beside
# --------------------------------------------------------------------------------------------

def spares(*counts):
    """The split layers of a model, each with this many spare GPU slots (promote-on-miss)"""
    return types.SimpleNamespace(moe_cpu_swap_modules = [types.SimpleNamespace(_split_spares = n) for n in counts])


def test_not_available_with_promote_on_miss(capsys):
    # 1.6.x's promote-on-miss runs a step at the first split layer of every pass, prompt chunks
    # included, over one state for every split layer: promotions into GPU slots, demotions into
    # worker slots, map flips. None of it is per lane
    ref, _, _, _ = drive(False, PLAN)

    def with_spares(g):
        g.model.config.infer_params = spares(0, 4, 0)

    out, g, jobs, _ = drive(True, PLAN, setup = with_spares)
    assert out == ref
    assert set(jobs["b"].chunk_threads) == {MAIN} and g._side_pool is None
    assert g.prefill_thread_state == "unavailable"
    assert g.prefill_thread_reason.startswith("promote-on-miss")
    assert g.model.host.split is None           # the worker's slots were left as they were
    lines = [l for l in capsys.readouterr().out.splitlines() if "Prefill thread" in l]
    assert lines == [f" !! Prefill thread not available: {g.prefill_thread_reason}"]

    # With no spare slots anywhere (EXL3_MOE_CPU_PROMOTE=0, as in production) the thread is on
    def without(g):
        g.model.config.infer_params = spares(0, 0, 0)

    out, g, jobs, _ = drive(True, PLAN, setup = without)
    assert out == ref and g.prefill_thread_state == "on"
    assert all(t.startswith("exl3-prefill") for t in jobs["b"].chunk_threads)


def test_not_available_with_a_draft_model_other_than_mtp(capsys):
    g = make_generator()
    g.mtp_draft = False                         # a draft model, and not an MTP head
    g.prefill_thread = True
    assert g._side_plan() is False and g._side_pool is None
    assert g.prefill_thread_reason == "draft model other than MTP"
    # The n-gram drafter has no model: the thread is on
    g = make_generator(draft = False)
    g.model.config.moe_cpu_hosts = {"cuda:0": Slots()}
    g.ngram_match_min = 4
    g.prefill_thread = True
    assert g._side_plan() is True
    g._side_pool.shutdown(wait = True)


def test_a_verification_pass_is_counted_at_its_widest():
    # The draft step decides the width of the verification pass after the prefill round: with
    # the hybrid draft's long window (max_draft_tokens) a single job generating can be past the
    # decode-size paths, and the chunks are waited for
    ref, _, _, _ = drive(False, PLAN)

    def hybrid(g):
        g.max_draft_tokens = 8                  # hybrid_draft_tokens = 8 over the MTP drafter's 1

    out, g, jobs, _ = drive(True, PLAN, setup = hybrid)
    assert out == ref
    assert g.side_stats["chunks"] == 4 + 1 and g.side_stats["beside"] == 0
    assert all(t.startswith("exl3-prefill") for t in jobs["b"].chunk_threads)


# --------------------------------------------------------------------------------------------
#   1.6.2: the disk tier, close(), a cache taken over, a failed decode step
# --------------------------------------------------------------------------------------------

class SideTier:
    """The CPU tier write-behind queues pages through, as toy.ToyTier, asserting what a chunk
    under way on the prefill thread must not meet: write-behind queues no page of the chunk's
    (it writes from the sequence position on) and a page queued is never written again."""

    def __init__(self, gen):
        self.gen = gen
        self.max_slots = 1 << 20
        self.stored = {}
        self.beside_chunk = 0

    def __contains__(self, h):
        return False

    def store(self, page, serial):
        assert page.kv_position == PAGE, "write-behind queued an incomplete page"
        side = self.gen._side
        if side is not None:
            seq = side.job.sequences[0]
            assert all(page is not p for p in seq.allocated_pages[seq.kv_position // PAGE:]), \
                f"write-behind queued a page {side.job.name}'s chunk under way writes"
            if any(page is p for p in seq.allocated_pages):
                self.beside_chunk += 1
        first = self.stored.setdefault(page.phash, (page, page.sequence.clone()))
        assert first[0] is page and torch.equal(first[1], page.sequence), "a queued page was written again"


def test_write_behind_beside_a_chunk_under_way():
    # 1.6.2's disk tier queues the complete pages of every active job at the end of iterate(),
    # the job whose chunk is under way included. The chunk is recorded on this thread only, so
    # the job's sequence position is the chunk's start until then, and write-behind keeps two
    # pages behind it: the pages queued are the ones earlier chunks completed
    plan = {
        "a": dict(prompt = PROMPTS["a"], max_new = 900),
        "b": dict(prompt = long_prompt(3, 5 * PAGE + 41), max_new = 20, at = 6, chunk = PAGE, steps = 3),
    }
    tiers = []

    def setup(g):
        g.disk_page_cache = toy.ToyDisk()
        g.cpu_page_cache = SideTier(g)
        g.disk_cache_write_interval = 1e-9
        tiers.append(g.cpu_page_cache)

    ref, _, _, _ = drive(False, plan, setup = setup)
    out, g, jobs, _ = drive(True, plan, setup = setup)
    assert out == ref
    assert g.side_stats["beside"] >= 5
    tier = tiers[1]
    assert tier.beside_chunk > 0
    for page, content in tier.stored.values():
        assert torch.equal(page.sequence, content), "a queued page was written again"


def under_way(g, b, iters = 40):
    """Iterate until a chunk of b's prompt is running on the prefill thread"""
    deadline = time.time() + 10
    for _ in range(iters):
        g.iterate()
        if g._side is not None and g._side.job is b:
            break
    assert g._side is not None and g._side.job is b
    while not b.chunk_running:
        assert time.time() < deadline
        time.sleep(0.0005)


def side_generator():
    g = make_generator()
    g.model.host = Slots()
    g.model.config.moe_cpu_hosts = {"cuda:0": g.model.host}
    g.prefill_thread = True
    g.side_min_tokens = 16
    a = ToyJob(g.model, "a", PROMPTS["a"], 300)
    b = ToyJob(g.model, "b", long_prompt(3, 65), 20, chunk = 16)
    b.on_chunk = lasting(50)
    a.hurry, b.hurry = threading.Event(), threading.Event()
    g.enqueue(a)
    for _ in range(3):
        g.iterate()
    g.enqueue(b)
    return g, a, b


def test_close_waits_for_the_chunk_under_way_before_it_flushes():
    # 1.6.2's close() persists what is still only in VRAM before the generator is retired. A
    # chunk under way writes into the cache's pages and states from the other thread: it is
    # seen through first, and the thread goes
    g, a, b = side_generator()
    under_way(g, b)
    flushed = []
    g.disk_page_cache = types.SimpleNamespace(write_disabled = False, close = lambda: None)
    g.disk_cache_flush_on_idle = True
    g.flush_cache_to_disk = lambda *args, **kw: flushed.append(b.chunk_running)
    b.hurry.set()
    pool = g._side_pool
    g.close()
    assert flushed == [False] and g._side is None and not b.chunk_running
    assert pool._shutdown and all(not t.is_alive() for t in pool._threads)
    assert b.released == 0                      # close() releases no job, as upstream's


def test_a_generator_whose_cache_was_taken_over_waits_for_its_chunk():
    # A newer generator owns the cache: iterate() refuses to run, drops its jobs without touching
    # the cache and, first, sees the chunk under way through (nothing more of it is issued after)
    g, a, b = side_generator()
    under_way(g, b)
    g.cache.owner_serial = 2
    g.cache_owner_serial = 1
    b.hurry.set()
    with pytest.raises(RuntimeError, match = "no longer owns its Cache"):
        g.iterate()
    assert g._side is None and not b.chunk_running
    assert b.released == 0 and not g.active_jobs
    g._side_pool.shutdown(wait = True)


def test_a_cancel_on_a_cache_taken_over_waits_for_the_chunk_too():
    g, a, b = side_generator()
    under_way(g, b)
    g.cache.owner_serial = 2
    g.cache_owner_serial = 1
    b.hurry.set()
    g.cancel(b)
    assert g._side is None and not b.chunk_running and b.released == 1
    g._side_pool.shutdown(wait = True)


def test_a_failed_decode_step_leaves_the_chunk_under_way_to_clear_queue():
    # NonFiniteLogits (1.6.2) on a job's first sampled token is fatal for the iteration, which
    # may have left a chunk under way: it stays the generator's, and clear_queue() (or close())
    # sees it through before the pages go. Here c is generating, a's one-chunk prompt goes in
    # beside it and a fails on its first token, in the round that starts b's chunk beside both
    class NonFinite(RuntimeError):
        pass

    g = make_generator()
    g.model.host = Slots()
    g.model.config.moe_cpu_hosts = {"cuda:0": g.model.host}
    g.prefill_thread = True
    g.side_min_tokens = 16
    c = ToyJob(g.model, "c", PROMPTS["c"], 300)
    a = ToyJob(g.model, "a", long_prompt(5, 17), 20, chunk = 16)
    b = ToyJob(g.model, "b", long_prompt(3, 97), 20, chunk = 16)
    b.on_chunk = lasting(50)
    for j in (a, b, c):
        j.hurry = threading.Event()
    a.fail_first = NonFinite
    g.enqueue(c)
    for _ in range(3):
        g.iterate()
    g.enqueue(a)
    g.enqueue(b)
    with pytest.raises(NonFinite):
        for _ in range(200):
            g.iterate()
    assert g._side is not None and g._side.job is b
    b.hurry.set()
    for j in (a, c):
        j.droppable = True                      # sampling when the step failed
    g.clear_queue()
    assert g._side is None and not b.chunk_running
    assert (a.released, b.released, c.released) == (1, 1, 1)
    g._side_pool.shutdown(wait = True)


# --------------------------------------------------------------------------------------------
#   1.6.2: the scan inputs a speculative pass stages for its rewind
# --------------------------------------------------------------------------------------------

class Interleave:
    """Runs a layer of the chunk under way at each rewind of a speculative pass, just before
    the rewind replays the scan inputs that pass staged in the layers' states: the window a
    chunk on the other thread can always hit (a decode step's pass and its rewind are apart by
    its sampling, and the thread runs meanwhile)."""

    def __init__(self):
        self.ready = threading.Event()
        self.turn = threading.Event()
        self.done = threading.Event()
        self.hits = 0

    def layer(self, job, idx, phase):
        if threading.current_thread().name == MAIN:
            return
        if phase == "before":
            self.ready.set()
            if self.turn.wait(0.5):
                self.turn.clear()
        else:
            self.done.set()

    def rewind(self):
        if self.ready.is_set():
            self.ready.clear()
            self.done.clear()
            self.turn.set()
            if self.done.wait(2):
                self.hits += 1


def test_a_chunk_beside_a_speculative_decode_leaves_its_staged_inputs_alone():
    # A prompt chunk's pass carries no recurrent history: on 1.6.2 it stages nothing in the
    # layers' states (GDNLayerState.spec_*), whose rows the decode's rewind replays after its
    # sampling. The chunk runs through the same layers between a decode pass and its rewind
    # (Interleave) and the tokens are the ones of inline ingestion
    plan = {
        "a": dict(prompt = PROMPTS["a"], max_new = 1500),
        "b": dict(prompt = long_prompt(3, 97), max_new = 20, at = 6, chunk = 16),
    }
    ref, _, _, _ = drive(False, plan)
    inter = Interleave()

    def setup(g):
        g.model.chunks_through_layers = True
        g.model.layer_hook = inter.layer
        g.model.rewind_hook = inter.rewind

    out, g, jobs, _ = drive(True, plan, setup = setup)
    assert out == ref
    assert inter.hits >= 6 * len(g.model.layers) // 2
    assert all(t.startswith("exl3-prefill") for t in jobs["b"].chunk_threads)


def test_were_a_chunk_to_stage_its_inputs_the_rewind_would_replay_them():
    # The same run with layers that stage a prompt chunk's inputs as they do a speculative
    # pass's: the rewind of the pass the chunk ran beside replays the chunk's
    plan = {
        "a": dict(prompt = PROMPTS["a"], max_new = 1500),
        "b": dict(prompt = long_prompt(3, 97), max_new = 20, at = 6, chunk = 16),
    }
    inter = Interleave()

    def setup(g):
        g.model.chunks_through_layers = True
        g.model.stage_prefill = True
        g.model.layer_hook = inter.layer
        g.model.rewind_hook = inter.rewind

    with pytest.raises(AssertionError, match = "replay over a pass of shape"):
        drive(True, plan, setup = setup)


def _source(*path):
    with open(os.path.join(toy.REPO, *path)) as f:
        return f.read()


def test_a_prompt_chunk_stages_nothing_for_a_rewind():
    # What the two tests above rest on, read off the library's sources (nothing imported):
    # Job's chunk passes carry no recurrent history, and the recurrent layers take their staging
    # views only on a pass with history. To run before taking a newer upstream
    job = ast.parse(_source("exllamav3", "generator", "job.py"))
    co = next(n for n in ast.walk(job) if isinstance(n, ast.FunctionDef) and n.name == "_prefill_co")
    run = next(n for n in ast.walk(co) if isinstance(n, ast.FunctionDef) and n.name == "run")
    keys = {k.value for n in ast.walk(run) if isinstance(n, ast.Dict) for k in n.keys if isinstance(k, ast.Constant)}
    assert "cache_seqlens" in keys and "recurrent_states" in keys
    assert not {"recurrent_history", "recurrent_spec_step"} & keys
    assert not any(isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
                   and n.slice.value in ("recurrent_history", "recurrent_spec_step") for n in ast.walk(run))

    for name in ("gated_delta_net.py", "mamba2.py"):
        src = _source("exllamav3", "modules", name)
        tree = ast.parse(src)
        assert 'save_history = params.get("recurrent_history", False)' in src
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        def guarded(node):
            while node in parents:
                node = parents[node]
                if isinstance(node, ast.FunctionDef) and node.name == "stage":
                    return True         # its own calls are below
                if isinstance(node, (ast.If, ast.IfExp)) and "history" in ast.unparse(node.test):
                    return True
            return False

        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and (
            (isinstance(n.func, ast.Attribute) and n.func.attr in ("staged_views", "staged_step_views")) or
            (isinstance(n.func, ast.Name) and n.func.id == "stage"))]
        assert calls, name
        for c in calls:
            assert guarded(c), f"{name}:{c.lineno}: staging views taken outside a pass with history"


def test_graph_captures_stay_off_the_streams_the_two_threads_issue_on():
    # The decode's graphed sites capture lazily, on the generator's thread, while the prefill
    # thread may be issuing on the same device. A capture is on a stream of its own, made for it
    # and non-blocking, in thread-local mode: what the other thread launches on the pass's stream
    # goes neither into the graph nor against the capture's rules. (The extension comes from
    # the published wheel, built from these sources)
    src = _source("exllamav3", "exllamav3_ext", "graph.cu")
    begin = src.split("cudaStream_t Graph::capture_begin()", 1)[1].split("\n}\n", 1)[0]
    assert "cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking)" in begin
    assert "cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal)" in begin
    ext_dir = os.path.join(toy.REPO, "exllamav3", "exllamav3_ext")
    begins = []
    for root, _, files in os.walk(ext_dir):
        for fn in files:
            if fn.endswith((".cu", ".cpp", ".cuh", ".h")):
                with open(os.path.join(root, fn), errors = "replace") as f:
                    if "cudaStreamBeginCapture(" in f.read():
                        begins.append(fn)
    assert begins == ["graph.cu"]
