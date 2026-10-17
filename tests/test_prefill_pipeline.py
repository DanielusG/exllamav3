"""
Prompt chunks in flight on two cards (Generator.prefill_pipeline, Generator._pipe_round) against
the toy model of test_stagger_decode, driven as in test_prefill_thread.

No GPU or model is required. The toy job hands out a chunk's passes in two halves, as Job does
with the MTP head, and turns the rules of the pipeline into assertions: a card has one chunk at a
time (the head's update included), a layer takes the chunks in order, a chunk is recorded after
the one before it and with all its pieces done, a checkpoint is offered on a state that is whole
and never with a chunk under way, a chunk is not planned past a position where a checkpoint is
due, pages are not released while a half runs.

The tests at the end run Job's own coroutine (1.6.2's): the chunks planned ahead are the chunks
planned in turn, with the generator's rule for when that is allowed (_pipe_ahead), and the record
of a chunk leaves alone a state the next one has advanced.

What 1.6.2 adds against the v1.5.4 branch: write-behind of the persistent disk tier beside two
chunks of one job, close() and a superseded cache with two chunks under way, a failed decode step
(NonFiniteLogits), promote-on-miss, the widest verification pass, a chunk between a speculative
pass and its rewind (the staged scan inputs), PairedState, and Job's record of a page that would
swap the block table.
"""

import importlib.util
import os
import sys
import threading
import time
import types

import pytest
import torch

import test_stagger_decode as toy
from test_stagger_decode import GEN, PROMPTS, ToyJob, _host_torch, make_generator   # noqa: F401
from test_prefill_thread import (FakeLane, Interleave, Slots, _fake_lane, drive, lasting,  # noqa: F401
                                 long_prompt, spares, MAIN)

pytestmark = pytest.mark.nogpu

FIRST = "exl3-prefill_"         # the thread of the first card's halves (the prefill thread)
SECOND = "exl3-prefill2_"       # the thread of the second card's
PAGE = toy.PAGE


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


def under_way(g, job) -> int:
    return sum(1 for p in g._pipe if p.job is job and p.future is not None)


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


def test_two_chunks_in_flight_take_fewer_decode_steps_than_one():
    # The measure of the feature on the toy: a half lasts three decode steps either way, and
    # with the two halves of neighbouring chunks under way together the prompt is in sooner
    # than with one chunk in flight at a time (EXL3_PREFILL_PIPELINE_DEPTH=1: the same halves,
    # one after the other). On the v1.5.4 branch this compared against the prefill thread, whose
    # whole chunk lasts what a half lasts here: a margin of a round per chunk, which a loaded
    # machine can take away
    def steps_during(events):
        first = events.index(("prefill", "b"))
        last = len(events) - 1 - events[::-1].index(("prefill", "b"))
        return events[first:last].count(("streaming", "a"))

    def one(g):
        g.pipe_depth = 1

    _, g1, jobs1, ev1 = pipe(PLAN, setup = one)
    _, g2, _, ev2 = pipe(PLAN)
    assert halves(jobs1["b"]) == (6, 6)
    assert g1.side_stats["piped"] == g1.side_stats["ahead"] == 0
    assert steps_during(ev2) < 0.75 * steps_during(ev1)


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


def test_the_chunk_before_one_planned_ahead_is_marked_for_its_record():
    # Job's record (1.6.2) takes back a state it finds past the chunk's end, unless the chunk
    # carries the mark _pipe_fill leaves on it when it plans the next one ahead: by the record
    # that one's passes may have advanced the state. (Job's side: the tests at the end)
    seen = {"pairs": 0}

    def script(g, jobs, n):
        b = jobs.get("b")
        mine = [p for p in g._pipe if b is not None and p.job is b]
        for p, q in zip(mine, mine[1:]):
            assert p.run.followed and q.ahead, "a chunk planned ahead of a record that does not know"
            seen["pairs"] += 1
        if mine:
            assert not mine[-1].run.followed

    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, _ = pipe(PLAN, script = script)
    assert out == ref and seen["pairs"] >= 3


def test_a_record_late_enough_to_find_the_next_chunk_done_leaves_the_state_alone():
    # The case the mark is for: the generator's thread gets to the record of a chunk only after
    # the chunk planned ahead of it has been through both cards (a long step of its own, say),
    # and finds the state past the chunk's end. Nobody decoding, so the halves end by themselves
    plan = {"b": dict(prompt = long_prompt(3, 193), max_new = 12, chunk = 16)}
    ref, _, _, _ = drive(False, plan)
    late = {"n": 0}

    def script(g, jobs, n):
        b = wrap(jobs, "b", lambda inner: slow(0.005))
        mine = [p for p in g._pipe if b is not None and p.job is b and p.future is not None]
        if len(mine) == 2:
            deadline = time.time() + 10
            while b.recurrent_state.target.position < mine[1].run.end:
                assert time.time() < deadline
                time.sleep(0.0005)
            late["n"] += 1

    out, g, jobs, _ = pipe(plan, script = script)
    assert out == ref
    assert late["n"] >= 3 and g.side_stats["ahead"] >= 3


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


def test_a_verification_pass_is_counted_at_its_widest():
    # 1.6.2's hybrid draft decides its long verification pass after the prefill round: the
    # decode batch is counted at 1 + max_draft_tokens per row, and with one job generating that
    # is past the decode-size paths. One chunk at a time, waited for
    ref, _, _, _ = drive(False, PLAN)

    def hybrid(g):
        g.max_draft_tokens = 8

    out, g, jobs, _ = pipe(PLAN, setup = hybrid)
    assert out == ref
    assert g.side_stats["chunks"] == 6 + 1
    assert g.side_stats["beside"] == 0 and g.side_stats["piped"] == 0 and g.side_stats["ahead"] == 0
    assert halves(jobs["b"]) == (6, 6)


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


def test_the_position_rule_is_read_off_every_side_of_a_pair():
    # 1.6.2's job states are PairedStates (target and draft); a side whose class does not say
    # counts as one whose position a pass reads, and so does a job without a recurrent state
    reads = GEN.Generator._state_reads_position
    free = types.SimpleNamespace(pass_reads_position = False)
    bound = types.SimpleNamespace(pass_reads_position = True)
    mute = types.SimpleNamespace()
    assert not reads(toy.RC.PairedState(free))
    assert not reads(toy.RC.PairedState(free, free))
    assert reads(toy.RC.PairedState(free, bound))
    assert reads(toy.RC.PairedState(mute))
    assert reads(None)
    assert not reads(free)                      # a bare state (no pair around it)
    # GDN's own: no module reads its position during a pass
    src = open(os.path.join(toy.REPO, "exllamav3", "modules", "gated_delta_net.py")).read()
    body = src.split("class GDNState:", 1)[1].split("\nclass ", 1)[0]
    assert "\n    pass_reads_position = False\n" in body


def test_a_page_that_the_record_may_swap_for_another_stops_the_plan_ahead():
    # Job.hash_completed_page gives a page its content hash at the record, and when another
    # job's referenced page already has it, takes that page instead: the block table is rebuilt
    # and the page's reference dropped, while a chunk planned ahead would still read the old
    # table. A chunk whose pages do not all carry their content hash already is recorded before
    # the next one is planned
    g = make_generator()
    job = ToyJob(g.model, "b", long_prompt(3, 2001), 20, chunk = 16)
    job.prepare_for_queue(g, 0)
    job.allocate_pages()
    seq = job.sequences[0]
    for i, page in enumerate(seq.allocated_pages[:2]):
        page.phash = b"\x02" * 8 + bytes([i]) * 8
    seq.page_hashes = [p.phash for p in seq.allocated_pages[:2]]
    step = job._prefill_co([])
    run = next(step)
    last = GEN._SidePass(job, step, object(), True, run)
    assert (run.end - run.tokens, run.end) == (0, 16)
    # (pages of 256: the first chunk of 16 completes no page, so nothing to compare yet)
    assert g._pipe_ahead(job, last) == 16
    run.tokens, run.end = 512, 512            # a chunk that completes the first two pages
    assert g._pipe_ahead(job, last) == 512
    seq.allocated_pages[1].phash = bytes(8) + b"random!!"
    assert g._pipe_ahead(job, last) is None
    step.close()


# --------------------------------------------------------------------------------------------
#   A job that leaves with chunks under way
# --------------------------------------------------------------------------------------------

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


def pipe_generator(prompt = 65, steps = 50):
    """a generating, b's prompt in chunks of 16 that last `steps` decode steps a half."""
    g = make_generator()
    g.model.host = Slots(6)
    g.model.config.moe_cpu_hosts = {"cuda:0": g.model.host}
    g.prefill_thread = g.prefill_pipeline = True
    g.side_min_tokens = 16
    a = ToyJob(g.model, "a", PROMPTS["a"], 300)
    b = ToyJob(g.model, "b", long_prompt(3, prompt), 20, chunk = 16)
    b.on_chunk = lasting(steps)
    a.hurry, b.hurry = threading.Event(), threading.Event()
    g.enqueue(a)
    for _ in range(3):
        g.iterate()
    g.enqueue(b)
    return g, a, b


def two_under_way(g, b, iters = 40):
    """Iterate until two chunks of b's prompt are under way and one of them runs."""
    deadline = time.time() + 10
    for _ in range(iters):
        g.iterate()
        if under_way(g, b) == 2:
            break
    assert under_way(g, b) == 2
    while not b.chunk_running:
        assert time.time() < deadline
        time.sleep(0.0005)


def shut(g):
    for pool in (g._side_pool, g._pipe_pool):
        if pool is not None:
            pool.shutdown(wait = True)


def test_clear_queue_stops_the_chunks_too():
    g, a, b = pipe_generator(steps = 4)
    a.droppable = b.droppable = True
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
    shut(g)


def test_close_stops_both_chunks_before_it_flushes():
    # 1.6.2's close() persists what is still only in VRAM before the generator is retired; two
    # chunks under way write into the cache's pages and states from two threads. They stop at
    # the half they are in, that half is waited for, and both threads go
    g, a, b = pipe_generator()
    two_under_way(g, b)
    flushed = []
    g.disk_page_cache = types.SimpleNamespace(write_disabled = False, close = lambda: None)
    g.disk_cache_flush_on_idle = True
    g.flush_cache_to_disk = lambda *args, **kw: flushed.append((b.chunk_running, len(g._pipe)))
    b.hurry.set()
    pools = (g._side_pool, g._pipe_pool)
    g.close()
    assert flushed == [(0, 0)] and not g._pipe and not b.chunk_running
    for pool in pools:
        assert pool._shutdown and all(not t.is_alive() for t in pool._threads)
    assert b.released == 0                      # close() releases no job, as upstream's


def test_a_generator_whose_cache_was_taken_over_stops_both_chunks():
    g, a, b = pipe_generator()
    two_under_way(g, b)
    g.cache.owner_serial = 2
    g.cache_owner_serial = 1
    b.hurry.set()
    with pytest.raises(RuntimeError, match = "no longer owns its Cache"):
        g.iterate()
    assert not g._pipe and not b.chunk_running
    assert b.released == 0 and not g.active_jobs
    shut(g)


def test_a_cancel_on_a_cache_taken_over_stops_both_chunks_too():
    g, a, b = pipe_generator()
    two_under_way(g, b)
    g.cache.owner_serial = 2
    g.cache_owner_serial = 1
    b.hurry.set()
    g.cancel(b)
    assert not g._pipe and not b.chunk_running and b.released == 1
    shut(g)


def test_a_failed_decode_step_leaves_the_chunks_under_way_to_clear_queue():
    # NonFiniteLogits (1.6.2) on a job's first sampled token is fatal for the iteration. Here c
    # is generating, a's one-chunk prompt goes in beside it, b's prompt in chunks that last; a
    # fails on its first token with two chunks of b's under way: they stay the generator's,
    # and clear_queue() stops them before the pages go
    class NonFinite(RuntimeError):
        pass

    g = make_generator()
    g.model.host = Slots(6)
    g.model.config.moe_cpu_hosts = {"cuda:0": g.model.host}
    g.prefill_thread = g.prefill_pipeline = True
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
    assert under_way(g, b) == 2
    b.hurry.set()
    for j in (a, c):
        j.droppable = True                      # sampling when the step failed
    g.clear_queue()
    assert not g._pipe and not b.chunk_running
    assert (a.released, b.released, c.released) == (1, 1, 1)
    shut(g)


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
#   1.6.2: the disk tier's write-behind, the staged scan inputs
# --------------------------------------------------------------------------------------------

class PipeTier:
    """The CPU tier write-behind queues pages through, as toy.ToyTier, asserting what chunks in
    flight must not meet: no page from the start of any chunk under way on (the job's sequence
    position is its oldest chunk's start until that one is recorded, and write-behind stays two
    pages behind it), and a page queued is never written again."""

    def __init__(self, gen):
        self.gen = gen
        self.max_slots = 1 << 20
        self.stored = {}
        self.beside = 0
        self.beside_ahead = 0

    def __contains__(self, h):
        return False

    def store(self, page, serial):
        assert page.kv_position == PAGE, "write-behind queued an incomplete page"
        for p in list(self.gen._pipe):
            seq = p.job.sequences[0]
            start = p.run.end - p.run.tokens
            assert all(page is not q for q in seq.allocated_pages[start // PAGE:]), \
                f"write-behind queued a page at or after {p.job.name}'s chunk from {start}"
            if any(page is q for q in seq.allocated_pages):
                self.beside += 1
                self.beside_ahead += p.ahead
        first = self.stored.setdefault(page.phash, (page, page.sequence.clone()))
        assert first[0] is page and torch.equal(first[1], page.sequence), "a queued page was written again"


@pytest.mark.parametrize("card", [0, 1])
def test_write_behind_beside_two_chunks_of_one_job(card):
    # 1.6.2's disk tier queues the complete pages of every active job at the end of iterate(),
    # the job with two chunks under way included, one of them planned ahead of the other's
    # record. The pages queued are the ones the recorded chunks completed, all their pieces
    # done (with the head on the first card, its update is the chunk's last piece)
    plan = {
        "a": dict(prompt = PROMPTS["a"], max_new = 1500),
        "b": dict(prompt = long_prompt(3, 8 * PAGE + 41), max_new = 20, at = 6, chunk = PAGE, steps = 3),
    }
    tiers = []

    def setup(g):
        g.disk_page_cache = toy.ToyDisk()
        g.cpu_page_cache = PipeTier(g)
        g.disk_cache_write_interval = 1e-9
        tiers.append(g.cpu_page_cache)

    ref, _, _, _ = drive(False, plan, setup = setup)
    out, g, jobs, _ = pipe(plan, setup = setup, script = head_on(card))
    assert out == ref
    assert g._pipe_head_first == (card == 0)
    assert g.side_stats["ahead"] >= 4
    tier = tiers[1]
    assert tier.beside > 0 and tier.beside_ahead > 0
    assert len(tier.stored) >= 8
    for page, content in tier.stored.values():
        assert torch.equal(page.sequence, content), "a queued page was written again"


PLAN_SPEC = {
    "a": dict(prompt = PROMPTS["a"], max_new = 1500),
    "b": dict(prompt = long_prompt(3, 97), max_new = 20, at = 6, chunk = 16),
}


def test_chunks_beside_a_speculative_decode_leave_its_staged_inputs_alone():
    # A prompt chunk's pass carries no recurrent history: on 1.6.2 it stages nothing in the
    # layers' states, whose rows the decode's rewind replays after its sampling. Here the halves
    # of two chunks of one job run through the same layers from two threads between a decode
    # pass and its rewind, and the tokens are the ones of inline ingestion
    ref, _, _, _ = drive(False, PLAN_SPEC)
    inter = Interleave()

    def setup(g):
        g.model.chunks_through_layers = True
        g.model.layer_hook = inter.layer
        g.model.rewind_hook = inter.rewind

    out, g, jobs, _ = pipe(PLAN_SPEC, setup = setup)
    assert out == ref
    assert inter.hits >= len(g.model.layers)
    assert halves(jobs["b"]) == (6, 6) and g.side_stats["ahead"] >= 1


def test_were_the_chunks_to_stage_their_inputs_the_rewind_would_replay_them():
    inter = Interleave()

    def setup(g):
        g.model.chunks_through_layers = True
        g.model.stage_prefill = True
        g.model.layer_hook = inter.layer
        g.model.rewind_hook = inter.rewind

    with pytest.raises(AssertionError, match = "replay over a pass of shape"):
        pipe(PLAN_SPEC, setup = setup)


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


def test_not_available_with_promote_on_miss(capsys):
    # 1.6.2's promote-on-miss rules the prefill thread out (test_prefill_thread), and with it
    # the pipeline: ingestion inline, as upstream, and the worker's slots as they were
    ref, _, _, _ = drive(False, PLAN)

    def with_spares(g):
        g.model.config.infer_params = spares(0, 4, 0)

    out, g, jobs, _ = pipe(PLAN, setup = with_spares)
    assert out == ref
    assert set(jobs["b"].chunk_threads) == {MAIN} and g._side_pool is None and g._pipe_pool is None
    assert g.prefill_thread_state == g.prefill_pipeline_state == "unavailable"
    assert g.prefill_pipeline_reason == g.prefill_thread_reason and g.prefill_pipeline_reason.startswith("promote-on-miss")
    assert g.model.host.split is None
    assert lines_of(capsys) == []


def test_the_state_tells_whether_chunks_are_in_flight():
    out, g, _, _ = drive(True, PLAN)
    assert (g.prefill_pipeline_state, g.prefill_pipeline_reason) == ("off", None)
    g = make_generator()
    g.prefill_thread = g.prefill_pipeline = True
    assert (g.prefill_pipeline_state, g.prefill_pipeline_reason) == ("pending", None)
    out, g, _, _ = pipe(PLAN)
    assert (g.prefill_pipeline_state, g.prefill_pipeline_reason) == ("on", None)
    # The thread switched off at run time takes the pipeline with it
    g.prefill_thread = False
    assert g.prefill_pipeline_state == "off"


def test_the_switch_turns_the_prefill_thread_on(monkeypatch):
    # (what a server relies on: it asks for the pipeline alone). Read off the constructor, which
    # needs a model to run
    import ast
    src = open(os.path.join(toy.REPO, "exllamav3", "generator", "generator.py")).read()
    init = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                and any(a.arg == "prefill_pipeline" for a in n.args.args))
    turned_on = [n for n in ast.walk(init) if isinstance(n, ast.If) and ast.unparse(n.test) == "self.prefill_pipeline"
                 and any(ast.unparse(t) == "self.prefill_thread" for stmt in n.body if isinstance(stmt, ast.Assign)
                         for t in stmt.targets)]
    assert turned_on
    stats = next(n for n in ast.walk(init) if isinstance(n, ast.Assign)
                 and any(ast.unparse(t) == "self.side_stats" for t in n.targets))
    assert {"piped", "ahead"} <= {k.value for k in stats.value.keys}


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
#   Job's own coroutine (1.6.2): the chunks planned ahead are the chunks planned in turn
# --------------------------------------------------------------------------------------------

REPO = toy.REPO


def _load_job():
    try:
        sys.path.insert(0, REPO)
        try:
            from exllamav3.generator import job as job_mod
            from exllamav3.cache import recurrent as rec_mod
            return job_mod, rec_mod
        finally:
            sys.path.remove(REPO)
    except Exception:
        pass
    pkg = "exl3_pipe_job_stub"
    src = os.path.join(REPO, "exllamav3")

    def stub(name, **attrs):
        full = f"{pkg}.{name}" if name else pkg
        m = types.ModuleType(full)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[full] = m
        return m

    def real(name):
        full = f"{pkg}.{name}"
        spec = importlib.util.spec_from_file_location(full, os.path.join(src, *name.split(".")) + ".py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        return m

    class _Any:
        def __init__(self, *a, **k):
            pass

    stub("")
    for p in ("cache", "generator", "model", "tokenizer", "util"):
        stub(p)
    stub("model.forward_cursor", ForwardCursor = _Any)
    stub("ext", exllamav3_ext = types.SimpleNamespace())
    stub("util.memory", malloc_trim = lambda: None)
    stub("util.tensor", SeqTensor = _Any)
    sys.modules[f"{pkg}.util"].profile_opt = lambda *a, **k: None
    sys.modules[f"{pkg}.tokenizer"].MMEmbedding = _Any
    stub("generator.pagetable", Sequence = _Any, tensor_hash_checksum = lambda ids, prev: b"", random_hash = lambda: b"")
    stub("generator.filter", Filter = _Any)
    stub("generator.loop_detect", LoopDetector = _Any)
    stub("generator.sampler", Sampler = _Any, DefaultSampler = _Any)
    real("constants")
    return real("generator.job"), real("cache.recurrent")


JOB, REC = _load_job()
CKPT = 100      # bytes of a pretend checkpoint


class State:
    """A job's recurrent state as Job's prefill and the checkpoint cache see it: a position the
    passes advance (set by whoever runs them), a size, a stash of the state as it stands (a
    checkpoint taken with a later chunk's passes in it shows), and the rewinds it was given
    (none is expected: a record never takes back what a chunk planned ahead has done)."""
    checkpoint_size = CKPT
    pass_reads_position = False

    def __init__(self, job, position):
        self.job = job
        self.position = position
        self.rewinds = []

    def stash(self):
        return {"position": self.position, "checkpoint_size": CKPT}

    def rewind_to(self, position):
        self.rewinds.append((self.position, position))
        self.position = position


class Ids:
    def __init__(self, n):
        self.ids = torch.arange(n, dtype = torch.long).view(1, -1)

    def __len__(self):
        return self.ids.shape[-1]

    def torch_slice(self, a, b):
        return self.ids[:, a:b]


class Page:
    def __init__(self, phash):
        self.phash = phash
        self.kv_position = 0
        self.prev_hash = None
        self.can_revert = True
        self.sequence = torch.zeros((1, PAGE), dtype = torch.long)


def real_job(prompt: int, chunk: int, interval: int, cached_pages: int = 0):
    cache = REC.RecurrentCache(types.SimpleNamespace(loaded_tp = False), max_size = 100 * CKPT)
    gen = types.SimpleNamespace(
        recurrent_cache = cache,
        recurrent_checkpoint_interval = interval,
        recurrent_checkpoint_interval_pp = 32768,
        max_chunk_size = chunk,
        model = types.SimpleNamespace(caps = {}),
        cache = None, draft_cache = None, draft_model = None,
        mtp_draft = False, dflash_draft = False,
    )
    job = JOB.Job.__new__(JOB.Job)
    job.generator = gen
    seq = types.SimpleNamespace(
        kv_position = cached_pages * PAGE,
        sequence_ids = Ids(prompt + 1),
        allocated_pages = [Page(f"p:{i}".encode()) for i in range(prompt // PAGE + 2)],
        prefill_complete = False,
        block_index_tensor = None,
        page_hashes = [],
        mtp_carry_hidden = None,
    )
    for p in seq.allocated_pages[:cached_pages]:
        p.kv_position = PAGE
    job.sequences = [seq]
    job.cached_pages = cached_pages
    job.new_tokens = 0
    job.recurrent_state = State(job, cached_pages * PAGE)
    job.last_recurrent_checkpoint_pos = cached_pages * PAGE if cached_pages else None
    job.embeddings = []
    job.time_first_prefill = None
    job.alt_rope_freqs = None
    job.cached_tokens = 0
    job.total_pages = 0
    job.serial_number = 1
    job.identifier = None
    return job


def positions(cache) -> list:
    return sorted(v["position"] for v in cache.values())


def shape(job) -> tuple:
    seq = job.sequences[0]
    return (seq.kv_position, seq.prefill_complete, [p.kv_position for p in seq.allocated_pages],
            positions(job.generator.recurrent_cache))


def ingest_real(job, ahead: bool) -> list:
    """The job's prompt through Job._prefill_co, a chunk planned after the record of the one
    before it, or (ahead) before it wherever the generator's rule allows (Generator._pipe_ahead,
    the real one, and the mark _pipe_fill leaves on the chunk under way); the generator's offer
    of a checkpoint after every record that leaves nothing under way."""
    gen = job.generator
    rule = GEN.Generator.__new__(GEN.Generator)
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

    while not job.is_prefill_done():
        if not under and not plan():
            break
        if ahead and len(under) == 1:
            at = rule._pipe_ahead(job, GEN._SidePass(job, under[0][0], object(), True, under[0][1]))
            if at is not None:
                under[0][1].followed = True
                plan(at)
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
    # No record took back what the chunk planned ahead of it had advanced
    assert one.recurrent_state.rewinds == two.recurrent_state.rewinds == []
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
    starts_in_turn = [a for (a, b, c), at in zip(chunks, turns) if at is None]
    assert len(chunks) == 19 and seq.kv_position == 70000
    assert starts_in_turn == [0, 32768, 65536, 69632, 69888]


def test_the_record_of_a_chunk_followed_by_another_leaves_the_state_alone():
    # 1.6.2's record rewinds a state found past the chunk's end (the extension an atomic
    # multimodal prefill processes twice). With the next chunk planned ahead and run, the state
    # is past it by that chunk's length, and that is not something to take back. Without the
    # mark the generator leaves on the chunk, the record would
    job = real_job(20000, 4096, 4096)
    results = []
    step = job._prefill_co(results)
    run = next(step)
    assert (run.end, run.checkpoint, run.followed) == (4096, False, False)
    step2 = job._prefill_co(results, run.end)
    run2 = next(step2)
    assert (run2.end - run2.tokens, run2.end) == (4096, 8192)
    run.followed = True
    job.recurrent_state.position = 8192          # both chunks' passes have run
    with pytest.raises(StopIteration):
        step.send(None)
    assert job.recurrent_state.rewinds == [] and job.sequences[0].kv_position == 4096
    with pytest.raises(StopIteration):
        step2.send(None)
    assert job.recurrent_state.rewinds == [] and job.sequences[0].kv_position == 8192

    # The same with the mark left off: the record of the first chunk rewinds the second's work
    job = real_job(20000, 4096, 4096)
    step = job._prefill_co([])
    run = next(step)
    step2 = job._prefill_co([], run.end)
    next(step2)
    job.recurrent_state.position = 8192
    with pytest.raises(StopIteration):
        step.send(None)
    assert job.recurrent_state.rewinds == [(8192, 4096)]
    step2.close()
