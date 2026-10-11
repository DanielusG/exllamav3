"""
Prompt chunks in flight on two cards (Generator.prefill_pipeline, Generator._pipe_round) against
the toy model of test_stagger_decode, driven as in test_prefill_thread.

No GPU or model is required. The toy job hands out a chunk's passes in two halves, as Job does
with the MTP head, and turns the rules of the pipeline into assertions: a card has one chunk at a
time, a layer takes the chunks in order, a chunk is recorded after the one before it, a
checkpoint is offered on a state that is whole and never with a chunk under way, a chunk is not
planned past a position where a checkpoint is due, pages are not released while a half runs.
The last test is on Job's own coroutine: the chunks planned ahead are the chunks planned in turn.
"""

import threading
import time
import types

import pytest
import torch

import test_stagger_decode as toy
import test_tail_checkpoint as ttc
from test_stagger_decode import GEN, PROMPTS, ToyJob, _host_torch, make_generator   # noqa: F401
from test_prefill_thread import FakeLane, Slots, _fake_lane, drive, lasting, long_prompt, MAIN   # noqa: F401

FIRST = "exl3-prefill_"         # the thread of the first card's halves (the prefill thread)
SECOND = "exl3-prefill2_"       # the thread of the second card's


def pipe(plan, **kw):
    kw.setdefault("slots", 6)
    return drive(True, plan, pipeline = True, **kw)


def halves(job) -> tuple:
    names = job.chunk_threads
    return sum(1 for t in names if t.startswith(FIRST)), sum(1 for t in names if t.startswith(SECOND))


def slow(seconds = 0.01):
    """A half that takes a while, for chunks with nobody decoding beside them."""
    def on_chunk(job, a, b):
        if threading.current_thread().name != MAIN:
            time.sleep(seconds)
    return on_chunk


def wrap(jobs, name, make):
    """Put make(inner) in place of a job's on_chunk, once."""
    job = jobs.get(name)
    if job is not None and not hasattr(job, "_wrapped"):
        job.on_chunk = make(job.on_chunk or (lambda *a: None))
        job._wrapped = True
    return job


# 96 prompt tokens in chunks of 16 while a generates; each half lasts three decode steps
PLAN = {
    "a": dict(prompt = PROMPTS["a"], max_new = 900),
    "b": dict(prompt = long_prompt(3, 97), max_new = 30, at = 6, chunk = 16, steps = 3),
}


# --------------------------------------------------------------------------------------------
#   Same tokens, two chunks under way
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("stagger", [False, True])
def test_tokens_with_the_pipeline_are_the_regular_ones(stagger):
    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, events = pipe(PLAN, stagger = stagger)
    assert out == ref
    # Six chunks, each half on the thread of its card; a's prompt went the same way
    assert halves(jobs["b"]) == (6, 6) and halves(jobs["a"]) == (1, 1)
    stats = g.side_stats
    assert stats["chunks"] == 6 + 1 and stats["errors"] == 0
    # Chunks were under way together, the later one planned before the earlier was recorded
    assert stats["piped"] >= 3 and stats["ahead"] >= 3
    # and a kept receiving tokens meanwhile
    first = events.index(("prefill", "b"))
    last = len(events) - 1 - events[::-1].index(("prefill", "b"))
    assert events[first:last].count(("streaming", "a")) >= 5 * 3
    assert stats["beside"] == 6 and stats["decode_steps"] >= 6 * 3


def test_the_pipeline_takes_less_decode_steps_than_one_chunk_at_a_time():
    # The measure of the feature on the toy: a half lasts three decode steps either way, and
    # with the two halves of neighbouring chunks under way together the prompt is in sooner
    def steps_during(events):
        first = events.index(("prefill", "b"))
        last = len(events) - 1 - events[::-1].index(("prefill", "b"))
        return events[first:last].count(("streaming", "a"))
    _, g1, _, ev1 = drive(True, PLAN)
    _, g2, _, ev2 = pipe(PLAN)
    assert g1.side_stats["piped"] == 0
    assert steps_during(ev2) < steps_during(ev1)


def test_with_nobody_decoding_a_round_is_worth_a_chunk():
    plan = {"b": dict(prompt = long_prompt(3, 97), max_new = 12, chunk = 16)}
    ref, _, _, ev0 = drive(False, plan)
    seen = []
    def script(g, jobs, n):
        if wrap(jobs, "b", lambda inner: slow()) is not None:
            seen.append(jobs["b"].sequences[0].kv_position)
    out, g, jobs, ev = pipe(plan, script = script)
    assert out == ref
    assert ev == ev0                    # the very same sequence of results as inline
    # Every iterate() recorded a chunk, with the next one already on the first card
    assert seen[:6] == [0, 16, 32, 48, 64, 80]
    assert g.side_stats["ahead"] == 5 and g.side_stats["piped"] == 5
    assert g.side_stats["beside"] == 0 and g.side_stats["decode_steps"] == 0


def test_the_halves_have_a_lane_each_and_a_half_that_is_done_has_run():
    _, g, jobs, _ = pipe(PLAN)
    side, first, second = FakeLane.made
    # The prefill thread's own lane, then one object per half: same lane for the first (its
    # thread is the prefill thread), lane 2 for the second, each with its own card to wait for
    assert (side.index, first.index, second.index) == (1, 1, 2)
    assert (first.devices, second.devices) == ([0], [1])
    assert len(second.entered) == 1 and second.entered[0].startswith(SECOND)
    assert first.entered == []
    # Seven chunks: seven passes of each lane, each waited for on its thread
    assert first.passes == second.passes == 7 and first.in_pass == second.in_pass == 0
    assert len(first.synced) == 7 and all(t.startswith(FIRST) for t in first.synced)
    assert len(second.synced) == 7 and all(t.startswith(SECOND) for t in second.synced)
    # The CPU expert worker's slots: two for the decode, two for each half
    assert g.model.host.split == (2, 2, 2)


def head_on(card: int):
    """A script that puts the toy model's MTP head on a card before the first chunk."""
    def script(g, jobs, n):
        if n == 0:
            g.model.head_card = card
            g.draft_model.modules = [types.SimpleNamespace(device = f"cuda:{card}")]
    return script


def test_the_head_s_update_takes_its_turn_on_the_first_card_when_the_head_is_there(capsys):
    # The toy job asserts that a card has one prompt pass at a time, the head's update included:
    # left to the second half's thread, the update of a chunk would run on the first card while
    # the next chunk's first half is there (on the machine the memory of the two added up)
    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, _ = pipe(PLAN, script = head_on(0))
    assert out == ref
    assert g._pipe_head_first
    assert g.side_stats["chunks"] == 6 + 1 and g.side_stats["errors"] == 0 and g.side_stats["ahead"] >= 3
    # Every update on the first card's thread, after its chunk's second half
    assert len(jobs["b"].head_threads) == 6 and all(t.startswith(FIRST) for t in jobs["b"].head_threads)
    side, first, second = FakeLane.made
    assert (first.devices, second.devices) == ([0], [1])
    # (the first card's lane: a first half and a head's update per chunk)
    assert first.passes == 2 * 7 and second.passes == 7
    assert "the MTP head's update takes its turn on the first card" in capsys.readouterr().out


def test_with_the_head_on_the_second_card_its_update_follows_the_second_half():
    out, g, jobs, _ = pipe(PLAN, script = head_on(1))
    assert not g._pipe_head_first
    assert len(jobs["b"].head_threads) == 6 and all(t.startswith(SECOND) for t in jobs["b"].head_threads)
    side, first, second = FakeLane.made
    assert first.passes == second.passes == 7


def test_the_head_can_be_left_to_the_second_thread_for_comparison(monkeypatch):
    # EXL3_PREFILL_PIPELINE_HEAD=second: the update follows the second half wherever the head
    # is. With the toy's head on the second card that changes nothing; it is read at set-up
    monkeypatch.setenv("EXL3_PREFILL_PIPELINE_HEAD", "second")
    def script(g, jobs, n):
        if n == 0:
            g.draft_model.modules = [types.SimpleNamespace(device = "cuda:0")]
    out, g, jobs, _ = pipe(PLAN, script = script)
    assert not g._pipe_head_first
    assert all(t.startswith(SECOND) for t in jobs["b"].head_threads)
    assert FakeLane.made[2].devices == [0, 1]


@pytest.mark.parametrize("card", [0, 1])
def test_a_job_that_leaves_with_the_head_s_update_owed(card):
    # Cancel with two chunks under way, wherever the head is: nothing is left running
    done = {}
    def script(g, jobs, n):
        head_on(card)(g, jobs, n)
        b = jobs.get("b")
        if b is not None and not b.cancelled and under_way(g, b) == 2 and b.chunk_running \
                and b.sequences[0].kv_position >= 16:
            b.cancelled = True
            b.hurry.set()
            g.cancel(b)
            done["at"] = n
            assert not g._pipe and not b.chunk_running and b.released == 1
    ref, _, _, _ = drive(False, {"a": PLAN["a"]})
    out, g, jobs, _ = pipe(PLAN, script = script)
    assert "at" in done and out["a"] == ref["a"] and out["b"] == []


@pytest.mark.parametrize("card", [0, 1])
def test_a_failed_update_of_the_head_fails_its_job_only(card):
    def script(g, jobs, n):
        head_on(card)(g, jobs, n)
        b = jobs.get("b")
        if b is not None and not hasattr(b, "_wrapped"):
            real = b._prefill_co
            def co(results, at = None):
                step = real(results, at)
                run = next(step)
                first, rest, after = run.halves
                def bad(state):
                    if run.end == 32:
                        b.hurry.set()       # (the toy's other half would wait for decode steps)
                        raise torch.cuda.OutOfMemoryError("no room for the head")
                    after(state)
                run.halves = (first, rest, bad)
                sent = yield run
                try:
                    step.send(sent)
                except StopIteration:
                    pass
            b._prefill_co = co
            b._wrapped = True
    ref, _, _, _ = drive(False, {"a": PLAN["a"]})
    out, g, jobs, events = pipe(PLAN, script = script)
    assert out["a"] == ref["a"]
    assert events.count(("error", "b")) == 1 and events.count(("prefill", "b")) == 1
    assert jobs["b"].released == 1 and g.side_stats["errors"] == 1 and not jobs["b"].chunk_running


# --------------------------------------------------------------------------------------------
#   Checkpoints
# --------------------------------------------------------------------------------------------

def test_a_chunk_is_never_planned_past_a_checkpoint():
    # A checkpoint is due every 32 tokens. The toy job asserts that no chunk is planned ahead
    # from such a position and that a checkpoint is offered on a state every layer of which is
    # at the checkpoint's position; here: that every one of them was offered
    plan = dict(PLAN, b = dict(PLAN["b"], ckpt = 32))
    ref, _, _, _ = drive(False, plan)
    out, g, jobs, _ = pipe(plan)
    assert out == ref
    offered = jobs["b"].checkpoints
    for due in (32, 64, 96):
        assert due in offered
    # Nothing was offered with a chunk under way: the positions in between are missing
    assert not {16, 48, 80} & set(offered)
    # Only the chunks from 16, 48 and 80 can have been planned ahead
    assert 1 <= g.side_stats["ahead"] <= 3


def test_one_chunk_at_a_time_where_every_chunk_ends_at_a_checkpoint():
    plan = dict(PLAN, b = dict(PLAN["b"], ckpt = 16))
    ref, _, _, _ = drive(False, plan)
    out, g, jobs, _ = pipe(plan)
    assert out == ref
    assert g.side_stats["ahead"] == 0 and g.side_stats["piped"] == 0
    assert [p for p in jobs["b"].checkpoints if p and p <= 96][:6] == [16, 32, 48, 64, 80, 96]


# --------------------------------------------------------------------------------------------
#   What may not overlap
# --------------------------------------------------------------------------------------------

def test_two_jobs_ingesting_are_under_way_together():
    plan = {
        "a": dict(prompt = PROMPTS["a"], max_new = 900),
        "b": dict(prompt = long_prompt(3, 49), max_new = 10, at = 4, chunk = 16, steps = 3),
        "c": dict(prompt = long_prompt(5, 49), max_new = 10, at = 4, chunk = 16, steps = 3),
    }
    order = []
    def script(g, jobs, n):
        for name in ("b", "c"):
            def make(inner, name = name):
                def on_chunk(job, a, b):
                    if threading.current_thread().name.startswith(FIRST):
                        order.append(name)
                    inner(job, a, b)
                return on_chunk
            wrap(jobs, name, make)
    ref, _, _, _ = drive(False, plan)
    out, g, jobs, _ = pipe(plan, script = script)
    assert out == ref
    assert g.side_stats["chunks"] == 6 + 1 and g.side_stats["piped"] >= 4
    # Both had their first chunk before either had its second, and three each in the end
    assert sorted(order[:2]) == ["b", "c"] and order.count("b") == order.count("c") == 3


def test_a_chunk_too_short_for_a_decode_step_waits_for_the_others_and_runs_alone():
    # 92 prompt tokens: five chunks of 16 and one of 12, which takes the decode-sized paths of
    # the modules. Nothing is decoded beside it, and no other chunk is under way with it
    plan = dict(PLAN, b = dict(PLAN["b"], prompt = long_prompt(3, 93)))
    with_it = []
    def script(g, jobs, n):
        def make(inner):
            def on_chunk(job, a, b):
                if b - a < 16:
                    with_it.append(sum(1 for p in list(job.generator._pipe)
                                       if p.future is not None and p.run.tokens >= 16))
                inner(job, a, b)
            return on_chunk
        wrap(jobs, "b", make)
    ref, _, _, _ = drive(False, plan)
    out, g, jobs, _ = pipe(plan, script = script)
    assert out == ref
    assert g.side_stats["chunks"] == 6 + 1 and g.side_stats["beside"] == 5
    assert with_it == [0, 0]            # its two halves, with no other chunk under way


def test_a_decode_batch_too_large_for_the_decode_paths_gets_one_chunk_at_a_time():
    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, _ = pipe(PLAN, max_rows = 1)
    assert out == ref
    assert g.side_stats["chunks"] == 6 + 1
    assert g.side_stats["beside"] == 0 and g.side_stats["piped"] == 0 and g.side_stats["ahead"] == 0


def test_a_job_that_may_not_be_planned_ahead_has_one_chunk_under_way():
    # Embeddings move the chunks' ends; the job still goes through the two halves
    def script(g, jobs, n):
        b = jobs.get("b")
        if b is not None:
            b.embeddings = [object()]
    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, _ = pipe(PLAN, script = script)
    assert out == ref
    assert halves(jobs["b"]) == (6, 6)
    assert g.side_stats["ahead"] == 0


def test_a_state_whose_position_a_pass_reads_is_not_planned_ahead(monkeypatch):
    monkeypatch.setattr(toy.ToyState, "pass_reads_position", True)
    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, _ = pipe(PLAN)
    assert out == ref and g.side_stats["ahead"] == 0


# --------------------------------------------------------------------------------------------
#   A job that leaves with chunks under way
# --------------------------------------------------------------------------------------------

def under_way(g, job) -> int:
    return sum(1 for p in g._pipe if p.job is job and p.future is not None)


def test_cancel_stops_both_chunks_before_the_pages_go():
    # The toy job asserts that its pages are not released while a half of a chunk runs
    done = {}
    def script(g, jobs, n):
        b = jobs.get("b")
        if b is not None and not b.cancelled and under_way(g, b) == 2 and b.chunk_running \
                and b.sequences[0].kv_position >= 16:
            b.cancelled = True
            b.hurry.set()
            recorded = b.sequences[0].kv_position
            g.cancel(b)
            done["at"] = recorded
            assert not g._pipe and not b.chunk_running and b.released == 1
            assert b not in g.active_jobs
    ref, _, _, _ = drive(False, {"a": PLAN["a"]})
    out, g, jobs, events = pipe(PLAN, script = script)
    assert "at" in done
    assert out["a"] == ref["a"]                    # the other job never noticed
    assert out["b"] == []
    # Neither of the two chunks was recorded
    assert events.count(("prefill", "b")) == done["at"] // 16


def test_cancel_of_one_job_leaves_the_other_s_chunk_alone():
    plan = {
        "a": dict(prompt = PROMPTS["a"], max_new = 900),
        "b": dict(prompt = long_prompt(3, 49), max_new = 10, at = 4, chunk = 16, steps = 3),
        "c": dict(prompt = long_prompt(5, 49), max_new = 10, at = 4, chunk = 16, steps = 3),
    }
    done = {}
    def script(g, jobs, n):
        b, c = jobs.get("b"), jobs.get("c")
        if b is not None and c is not None and not b.cancelled and under_way(g, b) and under_way(g, c):
            b.cancelled = True
            b.hurry.set()
            g.cancel(b)
            done["left"] = [p.job.name for p in g._pipe]
    ref, _, _, _ = drive(False, {k: v for k, v in plan.items() if k != "b"})
    out, g, jobs, _ = pipe(plan, script = script)
    assert done["left"] == ["c"]
    assert out["a"] == ref["a"] and out["c"] == ref["c"] and out["b"] == []


def test_clear_queue_stops_the_chunks_too():
    g = make_generator()
    g.model.host = Slots(6)
    g.model.config.moe_cpu_hosts = {"cuda:0": g.model.host}
    g.prefill_thread = g.prefill_pipeline = True
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
    assert under_way(g, b) == 2
    deadline = time.time() + 10
    while not b.chunk_running:
        assert time.time() < deadline
        time.sleep(0.0005)
    b.hurry.set()
    g.clear_queue()
    assert not g._pipe and not b.chunk_running and b.released == 1
    assert g.num_remaining_jobs() == 0
    for pool in (g._side_pool, g._pipe_pool):
        pool.shutdown(wait = True)


@pytest.mark.parametrize("thread", [FIRST, SECOND])
def test_a_failed_half_fails_its_job_only(thread):
    # The chunk from 16 fails, on either card, with the chunk from 32 already on the first
    def script(g, jobs, n):
        def make(inner):
            def on_chunk(job, a, b):
                inner(job, a, b)
                if a == 16 and threading.current_thread().name.startswith(thread):
                    job.hurry.set()         # (the toy's other half would wait for decode steps)
                    raise torch.cuda.OutOfMemoryError("no room for the second chunk")
            return on_chunk
        wrap(jobs, "b", make)
    ref, _, _, _ = drive(False, {"a": PLAN["a"]})
    out, g, jobs, events = pipe(PLAN, script = script)
    assert out["a"] == ref["a"]
    assert events.count(("error", "b")) == 1 and events.count(("prefill", "b")) == 1
    assert jobs["b"].released == 1 and jobs["b"] not in g.active_jobs
    assert g.side_stats["errors"] == 1 and not jobs["b"].chunk_running


def test_a_chunk_that_fails_at_the_plan_fails_its_job_only():
    def script(g, jobs, n):
        b = jobs.get("b")
        if b is not None and not hasattr(b, "_wrapped"):
            real = b._prefill_co
            def co(results, at = None):
                if at == 32:
                    b.hurry.set()           # (the toy's chunk under way would wait for decode steps)
                    raise RuntimeError("no page for the third chunk")
                return (yield from real(results, at))
            b._prefill_co = co
            b._wrapped = True
    ref, _, _, _ = drive(False, {"a": PLAN["a"]})
    out, g, jobs, events = pipe(PLAN, script = script)
    assert out["a"] == ref["a"]
    assert events.count(("error", "b")) == 1 and jobs["b"].released == 1
    assert not jobs["b"].chunk_running


# --------------------------------------------------------------------------------------------
#   The switch
# --------------------------------------------------------------------------------------------

def test_switched_off_with_chunks_under_way():
    def script(g, jobs, n):
        b = jobs.get("b")
        if b is not None and g.prefill_pipeline and under_way(g, b) == 2 and b.sequences[0].kv_position >= 16:
            g.prefill_pipeline = False
            b.hurry.set()       # the next round waits for the chunks
    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, _ = pipe(PLAN, script = script)
    assert out == ref
    # The chunks under way were seen through in their halves; the rest of the prompt went one
    # whole chunk at a time on the prefill thread
    first, second = halves(jobs["b"])
    assert 3 <= second < 6 and first == second + (6 - second)
    assert g.prefill_pipeline_state == "off" and g.prefill_thread_state == "on"


def test_switched_on_while_a_job_is_ingesting():
    def script(g, jobs, n):
        b = jobs.get("b")
        if b is not None and len(b.chunk_threads) >= 1 and not g.prefill_pipeline:
            g.prefill_pipeline = True
    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, _ = drive(True, PLAN, script = script, slots = 6)
    assert out == ref
    first, second = halves(jobs["b"])
    assert second >= 3 and first == second + (6 - second)
    assert g.side_stats["piped"] >= 2


def lines_of(capsys) -> list:
    return [l for l in capsys.readouterr().out.splitlines() if "Prefill pipeline" in l]


def test_not_available_without_the_mtp_head(capsys):
    plan = dict(PLAN, b = dict(PLAN["b"], halves = False))
    ref, _, _, _ = drive(False, plan, draft = False)
    out, g, jobs, _ = pipe(plan, draft = False)
    assert out == ref
    # One whole chunk at a time on the prefill thread, as without the switch
    assert halves(jobs["b"]) == (6, 0) and g._pipe_pool is None
    assert g.prefill_pipeline_state == "unavailable" and g.prefill_thread_state == "on"
    assert lines_of(capsys) == [" !! Prefill pipeline not available: " + g.prefill_pipeline_reason]
    assert "MTP head" in g.prefill_pipeline_reason


def test_not_available_with_the_layers_on_one_card(capsys, monkeypatch):
    monkeypatch.setattr(GEN, "cuda_stage_bounds", lambda model: [])
    out, g, jobs, _ = pipe(PLAN)
    assert halves(jobs["b"]) == (6, 0)
    assert g.prefill_pipeline_reason == "the layers of the model are on 1 card(s), not on two"


def test_not_available_with_too_few_worker_slots(capsys):
    # Two slots are one for the decode and one for the prefill thread: none for a second lane
    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, _ = pipe(PLAN, slots = 2)
    assert out == ref
    assert halves(jobs["b"]) == (6, 0)
    assert g.prefill_pipeline_reason == "the CPU expert worker has too few slots for a lane more (EXL3_MOE_CPU_SLOTS)"


def test_not_available_when_the_prefill_thread_is_not(capsys):
    out, g, jobs, _ = pipe(PLAN, slots = 1)
    assert set(jobs["b"].chunk_threads) == {MAIN}
    assert g.prefill_pipeline_state == "unavailable"
    assert g.prefill_pipeline_reason == g.prefill_thread_reason


def test_the_state_tells_whether_chunks_are_in_flight():
    out, g, _, _ = drive(True, PLAN)
    assert (g.prefill_pipeline_state, g.prefill_pipeline_reason) == ("off", None)
    g = make_generator()
    g.prefill_thread = g.prefill_pipeline = True
    assert (g.prefill_pipeline_state, g.prefill_pipeline_reason) == ("pending", None)
    out, g, _, _ = pipe(PLAN)
    assert (g.prefill_pipeline_state, g.prefill_pipeline_reason) == ("on", None)


def test_the_log_says_what_it_set_up_and_what_ran_beside_the_decode(capsys):
    _, g, _, _ = pipe(PLAN, slots = 4)
    lines = lines_of(capsys)
    assert lines[0] == (" -- Prefill pipeline: on, a chunk changes card at module 4 of 8, CPU expert worker slots "
                        "2 for the decode and 1 + 1 for the two halves (EXL3_MOE_CPU_SLOTS=6 gives each half two)")
    assert len(lines) == 2
    assert lines[1].startswith(" -- Prefill pipeline: 6 chunks in ") and lines[1].endswith(" decode steps beside them")
    # A decode step with two chunks under way is one step
    assert int(lines[1].split()[-5]) == g.side_stats["decode_steps"]


# --------------------------------------------------------------------------------------------
#   Job's own coroutine: the chunks planned ahead are the chunks planned in turn
# --------------------------------------------------------------------------------------------

PAGE = ttc.PAGE


class Ids:
    def __init__(self, n):
        self.ids = torch.arange(n, dtype = torch.long).view(1, -1)

    def __len__(self):
        return self.ids.shape[-1]

    def torch_slice(self, a, b):
        return self.ids[:, a:b]


class State(ttc.State):
    position = 0


def real_job(prompt: int, chunk: int, interval: int, cached_pages: int = 0):
    gen = ttc.make_gen(tail = 0, interval = interval, chunk = chunk)
    gen.model = types.SimpleNamespace(caps = {})
    gen.cache = gen.draft_cache = gen.draft_model = None
    gen.mtp_draft = gen.dflash_draft = False
    job = ttc.make_job(gen, "p", prompt, cached_pages = cached_pages, resumed = {} if cached_pages else None)
    seq = job.sequences[0]
    seq.sequence_ids = Ids(prompt + 1)
    seq.prefill_complete = False
    seq.block_index_tensor = None
    seq.page_hashes = []
    seq.allocated_pages = seq.allocated_pages[:prompt // PAGE + 2]
    for p in seq.allocated_pages:
        p.sequence = torch.zeros((1, PAGE), dtype = torch.long)
        p.prev_hash = None
        p.can_revert = True
    job.recurrent_state = State(job)
    job.recurrent_state.position = cached_pages * PAGE
    job.embeddings = []
    job.time_first_prefill = None
    job.alt_rope_freqs = None
    job.cached_tokens = 0
    job.total_pages = 0
    job.serial_number = 1
    job.identifier = None
    return job


def shape(job) -> tuple:
    seq = job.sequences[0]
    return (seq.kv_position, seq.prefill_complete, [p.kv_position for p in seq.allocated_pages],
            ttc.positions(job.generator.recurrent_cache))


def ingest_real(job, ahead: bool) -> list:
    """The job's prompt through Job._prefill_co, a chunk planned after the record of the one
    before it, or (ahead) before it wherever the generator's rule allows; the generator's offer
    of a checkpoint after every record that leaves nothing under way."""
    gen, seq = job.generator, job.sequences[0]
    chunks, results = [], []
    under = []          # (coroutine, its chunk's passes) not recorded yet

    def plan(at = None):
        step = job._prefill_co(results) if at is None else job._prefill_co(results, at)
        try:
            run = next(step)
        except StopIteration:
            return False
        chunks.append((run.end - run.tokens, run.end, run.checkpoint))
        under.append((step, run))
        return True

    def may_go_ahead(run) -> bool:
        return ahead and not run.checkpoint and run.end < len(seq.sequence_ids) - 1 \
            and not job.is_checkpoint_boundary(at = run.end)

    while not job.is_prefill_done():
        if not under and not plan():
            break
        if len(under) == 1 and may_go_ahead(under[0][1]):
            plan(under[0][1].end)
        # The passes of everything under way have run by the time the oldest is recorded
        job.recurrent_state.position = under[-1][1].end
        step, run = under.pop(0)
        with pytest.raises(StopIteration):
            step.send(None)
        if not under:
            job.maybe_stash_recurrent(gen.recurrent_cache)
    return chunks


@pytest.mark.parametrize("prompt, chunk, interval, cached", [
    (20000, 4096, 4096, 0),         # production's sizes: the last two chunks end at a checkpoint
    (70000, 4096, 4096, 0),         # across the interval of prompt ingestion (32768)
    (20000, 2048, 8192, 0),
    (9000, 4096, 4096, 0),
    (5000, 4096, 4096, 0),
    (20000, 4096, 4096, 16),        # resumed from a checkpoint at page 16
    (4096 * 3, 4096, 4096, 0),      # the prompt ends on a page boundary
    (300, 4096, 4096, 0),
])
def test_job_plans_ahead_the_chunks_it_plans_in_turn(prompt, chunk, interval, cached):
    one = real_job(prompt, chunk, interval, cached)
    two = real_job(prompt, chunk, interval, cached)
    in_turn = ingest_real(one, ahead = False)
    planned_ahead = ingest_real(two, ahead = True)
    assert planned_ahead == in_turn
    assert shape(two) == shape(one)
    assert one.is_prefill_done() and in_turn[-1][1] == prompt
    # Whole chunks between checkpoints, a pass of its own up to the last page boundary
    assert all(b - a <= chunk for a, b, _ in in_turn)
    if prompt % PAGE and prompt > PAGE:
        assert [c for c in in_turn if c[2]] == [in_turn[-2]] and in_turn[-2][1] == prompt // PAGE * PAGE


def test_a_long_prompt_is_mostly_planned_ahead():
    # 70000 tokens in chunks of 4096 with production's intervals: of 19 chunks, only the ones
    # that start at a checkpoint are planned in turn (every 32768 tokens, every 4096 within two
    # chunks of the end, and the prompt's last page boundary)
    job = real_job(70000, 4096, 4096)
    seq = job.sequences[0]
    turns = []
    real = job._prefill_co
    def co(results, at = None):
        turns.append(at)
        return real(results, at)
    job._prefill_co = co
    chunks = ingest_real(job, ahead = True)
    starts_in_turn = [a for (a, b, c), at in zip(chunks, [t for t in turns]) if at is None]
    assert len(chunks) == 19 and seq.kv_position == 70000
    assert starts_in_turn == [0, 32768, 65536, 69632, 69888]
