"""
A lower streaming threshold for prompt chunks beside a decode (EXL3_MOE_STREAM_T_BESIDE):
MoeCpuTuning reads it, MoeCpuHost._stream_threshold applies it to a chunk planned on a lane other
than 0 while the host's decode_beside holds, and Generator._note_beside sets decode_beside every
round of prompt ingestion.

No GPU, extension or model is required. The host module is loaded on its own with the rest of
the package stubbed, as in test_moe_host_lanes, whose model of the streams, the flags and the
worker's ring runs the real streamed path at the end; the generator's real prefill round runs
over the toy model of test_stagger_decode, as in test_prefill_thread and test_prefill_pipeline.
"""

import ast
import os
import threading
import types

import numpy as np
import pytest
import torch

import test_moe_host_lanes as lanes
from test_moe_host_lanes import HOST, LANE, Sim, _lane_zero      # noqa: F401
from test_stagger_decode import _host_torch                        # noqa: F401
from test_prefill_thread import PLAN, _fake_lane, drive, long_prompt, MAIN   # noqa: F401
from test_prefill_pipeline import PLAN as PIPE_PLAN, pipe

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAR = "EXL3_MOE_STREAM_T_BESIDE"
MIN_VAR = "EXL3_MOE_STREAM_T_BESIDE_MIN_ROWS"
STATS0 = {"normal": [0, 0, 0], "beside": [0, 0, 0], "small": [0, 0, 0]}


# --------------------------------------------------------------------------------------------
#   The variable
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (None, 0), ("", 0), ("  ", 0), ("1", 1), ("12", 12), (" 3 ", 3),
])
def test_the_variable_is_an_integer_of_at_least_one_or_off(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(VAR, raising = False)
    else:
        monkeypatch.setenv(VAR, value)
    assert HOST.MoeCpuTuning().stream_t_beside == expected


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc", "1e3", "0x10"])
def test_any_other_value_is_refused(monkeypatch, value):
    # 0 or less would stream experts that no token of the chunk selected
    monkeypatch.setenv(VAR, value)
    with pytest.raises(ValueError, match = VAR):
        HOST.MoeCpuTuning()


@pytest.mark.parametrize("value, expected", [
    (None, HOST.STREAM_T_BESIDE_MIN_ROWS), ("", HOST.STREAM_T_BESIDE_MIN_ROWS), ("1", 1), (" 512 ", 512), ("4096", 4096),
])
def test_the_smallest_chunk_is_an_integer_of_at_least_one_or_the_default(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(MIN_VAR, raising = False)
    else:
        monkeypatch.setenv(MIN_VAR, value)
    assert HOST.MoeCpuTuning().stream_t_beside_min_rows == expected
    assert HOST.STREAM_T_BESIDE_MIN_ROWS == 2048


@pytest.mark.parametrize("value", ["0", "-5", "2k", "1.5"])
def test_any_other_smallest_chunk_is_refused(monkeypatch, value):
    monkeypatch.setenv(MIN_VAR, value)
    with pytest.raises(ValueError, match = MIN_VAR):
        HOST.MoeCpuTuning()


def _tree(*parts):
    with open(os.path.join(REPO, "exllamav3", *parts)) as f:
        return ast.parse(f.read())


def _functions(node):
    return [n for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def test_the_variable_is_read_once_by_the_tuning_object():
    # Read where the other knobs of the CPU expert host are read, once, at import (MoeCpuTuning
    # on the module's TUNING); a host takes the value when it is made, and nothing reads the
    # environment while a chunk is planned
    tree = _tree("model", "moe_cpu_host.py")
    reads = []
    for fn in _functions(tree):
        for n in ast.walk(fn):
            if isinstance(n, ast.Constant) and n.value in (VAR, MIN_VAR):
                reads.append(fn.name)
    tuning = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoeCpuTuning")
    init = next(n for n in tuning.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert reads == ["__init__", "__init__"]
    assert any(isinstance(n, ast.Constant) and n.value == VAR for n in ast.walk(init))
    assert any(isinstance(n, ast.Constant) and n.value == MIN_VAR for n in ast.walk(init))
    host = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoeCpuHost")
    for name in ("submit_prefill", "_stream_threshold"):
        fn = next(n for n in host.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert "environ" not in ast.unparse(fn)
    # The generator does not read it: it asks the hosts
    gen = _tree("generator", "generator.py")
    assert not any(isinstance(n, ast.Constant) and n.value in (VAR, MIN_VAR) for n in ast.walk(gen))


def test_a_host_takes_the_value_of_the_tuning_object_when_it_is_made(monkeypatch):
    config = types.SimpleNamespace(directory = "/nonexistent", infer_params = types.SimpleNamespace())
    # The environment after the import changes nothing
    monkeypatch.setenv(VAR, "5")
    h = HOST.MoeCpuHost(config)
    assert h.stream_t_beside == HOST.TUNING.stream_t_beside
    assert h.stream_t_beside_min_rows == HOST.TUNING.stream_t_beside_min_rows
    assert h.decode_beside is False
    # A same-process sweep sets the tuning object before it makes a host (MoeCpuTuning's rule)
    monkeypatch.setattr(HOST.TUNING, "stream_t_beside", 2)
    h = HOST.MoeCpuHost(config)
    assert (h.stream_t_beside, h.decode_beside) == (2, False)
    assert h.beside_stats == STATS0
    monkeypatch.setattr(HOST.TUNING, "stream_t_beside", 0)
    h = HOST.MoeCpuHost(config)
    assert (h.stream_t_beside, h.beside_stats) == (0, None)
    # A host made without __init__ (upstream's tests of the streamed path) has it off
    bare = HOST.MoeCpuHost.__new__(HOST.MoeCpuHost)
    assert (bare.stream_t_beside, bare.decode_beside, bare.beside_stats) == (0, False, None)
    assert bare.stream_t_beside_min_rows == HOST.STREAM_T_BESIDE_MIN_ROWS


# --------------------------------------------------------------------------------------------
#   The threshold of a layer
# --------------------------------------------------------------------------------------------

E = 16
ROWS = 32
TOPK = 4
CARD_T = 23         # the card's own threshold (st["stream_t"])
# Assignments per expert of the chunk; the rest of the rows x topk slots are -1 sentinels
COUNTS = {0: 40, 1: 23, 2: 22, 3: 11, 4: 10, 5: 2, 6: 1}


def routing(counts = COUNTS, rows = ROWS, seed = 0):
    flat = [e for e, c in counts.items() for _ in range(c)]
    flat += [-1] * (rows * TOPK - len(flat))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(flat), generator = g)
    sel = torch.tensor(flat, dtype = torch.long)[perm].view(rows, TOPK)
    return torch.zeros((rows, 8), dtype = torch.half), sel, torch.ones((rows, TOPK), dtype = torch.half)


def recording_host(monkeypatch, beside_t = 0, card_t = CARD_T, min_rows = 1):
    """A host whose streamed and plain paths record what they are handed."""
    sim = Sim()
    h = lanes.make_host(monkeypatch, sim, 6)
    h.specs = [dict(num_experts = E, expert_bytes = 64, hi = 8, ho = 8, topk = TOPK)]
    h.aux = {0: {}}
    h.wslot_size = 1 << 20
    h.stream_min_rows = 8
    h.stream_t_beside = beside_t
    h.stream_t_beside_min_rows = min_rows
    h.beside_stats = {k: list(v) for k, v in STATS0.items()} if beside_t else None
    h.calls = []
    h._ensure_stream_state = lambda device: {"stream_t": card_t}
    h._submit_prefill_streamed = lambda layer, y, sel, w, spec, streamed, st, *rest: \
        h.calls.append(("streamed", list(streamed))) or "streamed"
    h.submit = lambda layer, y, sel, w: h.calls.append(("cpu", None)) or "cpu"
    return h


def plan(h, lane, decode_beside, rows = ROWS, counts = COUNTS):
    LANE._tls.lane = lane
    h.decode_beside = decode_beside
    h.calls.clear()
    h.submit_prefill(0, *routing(counts, rows))
    assert len(h.calls) == 1
    return h.calls[0]


def at(t, counts = COUNTS):
    """What submit_prefill hands on at threshold t: the streamed path with the experts that
    reach it, the plain CPU path when none does."""
    streamed = [e for e in range(E) if counts.get(e, 0) >= t]
    return ("streamed", streamed) if streamed else ("cpu", None)


@pytest.mark.parametrize("seed", range(4))
def test_off_streams_what_the_card_s_threshold_streams(monkeypatch, seed):
    # On every lane, told or not that a decode goes on beside: the experts of today, at the
    # card's threshold
    h = recording_host(monkeypatch, beside_t = 0)
    rng = np.random.default_rng(seed)
    counts = {int(e): int(c) for e, c in zip(rng.choice(E, 8, replace = False), rng.integers(1, 30, 8))}
    while sum(counts.values()) > ROWS * TOPK:
        counts.pop(next(iter(counts)))
    for lane in (0, 1, 2):
        for beside in (False, True):
            assert plan(h, lane, beside, counts = counts) == at(CARD_T, counts)
    assert h.beside_stats is None


def test_on_the_lower_threshold_is_for_a_lane_beside_a_decode_only(monkeypatch):
    h = recording_host(monkeypatch, beside_t = 1)
    normal, low = at(CARD_T), at(1)
    assert normal == ("streamed", [0, 1]) and low == ("streamed", [0, 1, 2, 3, 4, 5, 6])
    # Lane 0 is the generator's own thread: a chunk there is inline, and the decode after it
    # waits for all of it
    assert plan(h, 0, False) == normal
    assert plan(h, 0, True) == normal
    # The prefill thread's lane and the pipeline's second
    for lane in (1, 2):
        assert plan(h, lane, False) == normal
        assert plan(h, lane, True) == low
    # Any value: the experts with at least that many assignments
    h.stream_t_beside = 11
    assert plan(h, 1, True) == at(11) == ("streamed", [0, 1, 2, 3])
    # Used as it is, not scaled by the card's link, and not capped by the card's threshold
    h.stream_t_beside = 30
    assert plan(h, 1, True) == at(30) == ("streamed", [0])


def test_a_chunk_that_streams_nothing_at_the_card_s_threshold(monkeypatch):
    # Nothing as hot as the card's threshold: the plain CPU path as ever, unless the decode
    # goes on beside the chunk
    h = recording_host(monkeypatch, beside_t = 2, card_t = 64)
    assert plan(h, 1, False) == ("cpu", None)
    assert plan(h, 0, True) == ("cpu", None)
    assert plan(h, 1, True) == at(2)


def test_a_decode_size_batch_never_streams(monkeypatch):
    # Below stream_min_rows the batch takes the worker's decode-size path, whatever the threshold
    h = recording_host(monkeypatch, beside_t = 1)
    few = {0: 7, 1: 5}
    for lane in (0, 1):
        assert plan(h, lane, True, rows = 7, counts = few) == ("cpu", None)
    assert plan(h, 1, True, rows = 8, counts = few) == at(1, few)


def test_the_layers_planned_at_each_threshold_are_counted_per_lane(monkeypatch):
    h = recording_host(monkeypatch, beside_t = 1)
    plan(h, 0, True)
    plan(h, 1, False)
    plan(h, 1, True)
    plan(h, 1, True)
    plan(h, 2, True)
    assert h.beside_stats == {"normal": [1, 1, 0], "beside": [0, 2, 1], "small": [0, 0, 0]}
    # A decode-size batch is not a streamed layer
    plan(h, 1, True, rows = 7, counts = {0: 3})
    assert h.beside_stats == {"normal": [1, 1, 0], "beside": [0, 2, 1], "small": [0, 0, 0]}


def test_a_short_chunk_beside_a_decode_keeps_the_card_s_threshold(monkeypatch):
    # A chunk shorter than EXL3_MOE_STREAM_T_BESIDE_MIN_ROWS selects many experts once or twice:
    # at threshold 1 it would stream nearly every CPU-resident one it selects (a 47-token prompt
    # beside a decode: 5.0 s to its first token instead of 1.1, 2026-10-25). It keeps the card's
    # threshold, and is counted apart
    h = recording_host(monkeypatch, beside_t = 1, min_rows = ROWS + 1)
    assert plan(h, 1, True) == at(CARD_T)
    assert plan(h, 2, True) == at(CARD_T)
    assert plan(h, 1, False) == at(CARD_T)
    assert h.beside_stats == {"normal": [0, 1, 0], "beside": [0, 0, 0], "small": [0, 1, 1]}
    # At the minimum the lower threshold applies
    h.stream_t_beside_min_rows = ROWS
    assert plan(h, 1, True) == at(1)
    assert h.beside_stats["beside"] == [0, 1, 0]


def test_the_load_s_worst_case_does_not_depend_on_the_threshold():
    # The autosplit sizes the streamed path for every assignment of the chunk on CPU-resident
    # experts, all of them streamed (prefill_worst_case_parts: the outputs, the input copy and
    # the sorted lists for all of them; the larger of every row on one expert and the batched
    # tier's worst case), and allocates its persistent buffers (weight-staging ring, fused-tier
    # buffers, batched-tier statics) during the measured load. A lower threshold adds batches
    # through those buffers. If the worst case ever read the threshold, a lower one beside a
    # decode would need its own accounting
    tree = _tree("model", "moe_cpu_host.py")
    host = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoeCpuHost")
    for name in ("prefill_worst_case_parts", "_device_buffers"):
        src = ast.unparse(next(n for n in host.body if isinstance(n, ast.FunctionDef) and n.name == name))
        assert "stream_t" not in src and "_stream_threshold" not in src, name
    mlp = _tree("modules", "block_sparse_mlp.py")
    cls = next(n for n in mlp.body if isinstance(n, ast.ClassDef) and any(
        isinstance(f, ast.FunctionDef) and f.name == "_autosplit_worst_case" for f in n.body))
    fn = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "_autosplit_worst_case")
    src = ast.unparse(fn)
    assert "rows * self.num_experts_per_tok" in src and "A - a" in src and "stream_t" not in src


# The real streamed path, against the model of the streams and of the worker

STREAM_COUNTS = {0: 8, 1: 5, 2: 2, 3: 1}     # of lanes.ROWS x lanes.TOPK = 16 assignments


def streamed_chunk(monkeypatch, decode_beside, lane = 1):
    """One layer of a chunk through the real submit_prefill and streamed path on `lane`: the
    tail jobs it published to the worker's ring and the weight batches it streamed."""
    sim = Sim()
    h = lanes.streaming_host(monkeypatch, sim, pinned = True, layers = 1)
    h.specs = [dict(lanes.SPEC, topk = lanes.TOPK)]
    h.stream_min_rows = 4
    h.stream_t_beside = 1
    h.stream_t_beside_min_rows = 1
    h.beside_stats = {k: list(v) for k, v in STATS0.items()}
    h.decode_beside = decode_beside
    st = dict(lanes.stream_state(sim, 0), stream_t = 4)
    h._ensure_stream_state = lambda device: st
    sel = torch.tensor([[0, 1]] * 5 + [[0, 2]] * 2 + [[0, 3]], dtype = torch.long)
    assert {e: int((sel == e).sum()) for e in range(lanes.E)} == STREAM_COUNTS
    y = torch.zeros((lanes.ROWS, lanes.H), dtype = torch.half)
    w = torch.ones((lanes.ROWS, lanes.TOPK), dtype = torch.half)
    bg = sim.run()
    failed = []

    def chunk():
        try:
            if lane:
                LANE.Lane(lane, [0]).enter()
            h.submit_prefill(0, y, sel, w)
            sim.current().synchronize()
        except BaseException as e:
            failed.append(e)

    t = threading.Thread(target = chunk)
    t.start()
    t.join(30)
    sim.stop = True
    bg.join(5)
    assert not t.is_alive() and not failed, failed
    assert sim.errors == [] and all(not s.ops for s in sim.streams)
    tail_jobs = [int(j[5]) for j in h.v_jobs[:int(h.v_jobs_tail[0])]]
    assert set(tail_jobs) <= {0}
    return len(tail_jobs), h.wseq


def test_beside_a_decode_the_chunk_leaves_the_worker_no_tail(monkeypatch):
    # What the threshold is for: at the card's (4 here) the chunk streams experts 0 and 1 and
    # sends the rows of 2 and 3 to the worker, one job on the ring a decode's jobs wait behind;
    # at 1 it streams all four and the ring gets nothing
    assert streamed_chunk(monkeypatch, decode_beside = False) == (1, 1)
    assert streamed_chunk(monkeypatch, decode_beside = True) == (0, 2)
    # Inline, on the generator's thread, as without the variable
    assert streamed_chunk(monkeypatch, decode_beside = True, lane = 0) == (1, 1)


# --------------------------------------------------------------------------------------------
#   What the generator tells the hosts
# --------------------------------------------------------------------------------------------

def with_threshold(t = 1, left = False):
    """setup for drive(): the toy's host with the variable set; left: as an earlier generator
    over the same model may have left it, told that a decode goes on beside"""
    def setup(g):
        g.model.host.stream_t_beside = t
        g.model.host.stream_t_beside_min_rows = HOST.STREAM_T_BESIDE_MIN_ROWS
        g.model.host.decode_beside = left
    return setup


def recording(seen):
    """A script for drive(): every chunk (every half, in flight on two cards) records, as its
    passes start, whether the host was told that the decode goes on beside it."""
    def script(g, jobs, n):
        for job in jobs.values():
            if getattr(job, "_recorded", False):
                continue
            inner = job.on_chunk or (lambda *a: None)

            def on_chunk(j, a, b, inner = inner, host = g.model.host):
                seen.append((j.name, a, threading.current_thread().name, getattr(host, "decode_beside", None)))
                inner(j, a, b)

            job.on_chunk = on_chunk
            job._recorded = True
    return script


def told(seen, name):
    return [flag for job, _, _, flag in seen if job == name]


def test_nothing_is_written_without_the_variable(capsys):
    ref, _, _, _ = drive(False, PLAN)
    seen = []
    out, g, jobs, _ = drive(True, PLAN, script = recording(seen))
    assert out == ref
    assert g._beside_hosts == () and not hasattr(g.model.host, "decode_beside")
    assert told(seen, "b") == [None] * 4
    lines = [l for l in capsys.readouterr().out.splitlines() if "Prefill thread: on" in l]
    assert lines == [" -- Prefill thread: on, CPU expert worker slots 2 for the decode and 2 for the prefill"]


def test_chunks_beside_a_decode_are_told_so(capsys):
    # a generates through b's ingestion: b's four chunks run beside its decode. a's own prompt
    # went in with nobody decoding, on the same thread, waited for
    ref, _, _, _ = drive(False, PLAN)
    seen = []
    out, g, jobs, _ = drive(True, PLAN, script = recording(seen), setup = with_threshold())
    assert out == ref
    assert g._beside_hosts == (g.model.host,)
    assert told(seen, "a") == [False]
    assert told(seen, "b") == [True] * 4
    assert all(t.startswith("exl3-prefill") for _, _, t, _ in seen)
    assert g.side_stats["beside"] == 4
    lines = [l for l in capsys.readouterr().out.splitlines() if "Prefill thread: on" in l]
    assert lines == [" -- Prefill thread: on, CPU expert worker slots 2 for the decode and 2 for the prefill; "
                     "experts streamed from 1 assignments in chunks beside a decode (chunks of 2048 rows or more)"]


def test_a_generator_starts_its_hosts_from_nothing_decoding():
    # An earlier generator over the same model stopped while a decode went on beside a chunk:
    # the next one's first prompt, with nobody decoding, still gets the card's threshold
    seen = []
    _, g, _, _ = drive(True, PLAN, script = recording(seen), setup = with_threshold(left = True))
    assert told(seen, "a") == [False]
    assert told(seen, "b") == [True] * 4


def test_a_decode_batch_too_wide_to_overlap_is_not_beside_its_chunks():
    # Past side_max_rows the round waits for every chunk: nothing decodes beside them
    ref, _, _, _ = drive(False, PLAN)
    seen = []
    out, g, _, _ = drive(True, PLAN, script = recording(seen), setup = with_threshold(), max_rows = 1)
    assert out == ref
    assert g.side_stats["beside"] == 0
    assert told(seen, "b") == [False] * 4


def test_the_chunks_after_the_decode_ends_are_not_beside_it():
    # a stops generating while b's prompt is still going in: b's first chunks beside its decode,
    # the rest with nobody decoding, and never back
    plan = {
        "a": dict(prompt = PLAN["a"]["prompt"], max_new = 24),
        "b": dict(prompt = long_prompt(3, 16 * 8 + 1), max_new = 10, at = 3, chunk = 16, steps = 5),
    }
    ref, _, _, _ = drive(False, plan)
    seen = []
    out, g, _, _ = drive(True, plan, script = recording(seen), setup = with_threshold())
    assert out == ref
    flags = told(seen, "b")
    assert len(flags) == 8 and flags[0] is True and flags[-1] is False
    assert flags == sorted(flags, reverse = True)


def test_inline_the_hosts_are_never_told():
    # Without the prefill thread every chunk is on the generator's thread, and the threshold of
    # lane 0 is the card's whatever the flag: the generator does not even look for the hosts
    seen = []
    _, g, _, _ = drive(False, PLAN, script = recording(seen), setup = with_threshold())
    assert g._beside_hosts == ()
    assert {t for _, _, t, _ in seen} == {MAIN}
    assert set(told(seen, "b")) == {False}


def test_switched_off_at_run_time_the_hosts_are_told_nothing_decodes_beside():
    # (test_prefill_thread's case: the chunk under way is seen through, the rest of the prompt
    # goes in inline)
    seen = []
    rec = recording(seen)

    def script(g, jobs, n):
        rec(g, jobs, n)
        b = jobs.get("b")
        if b is not None and len(b.chunk_threads) == 2 and g.prefill_thread:
            g.prefill_thread = False
            b.hurry.set()

    ref, _, _, _ = drive(False, PLAN)
    out, g, jobs, _ = drive(True, PLAN, script = script, setup = with_threshold())
    assert out == ref
    assert jobs["b"].chunk_threads[2:] == [MAIN, MAIN]
    assert told(seen, "b") == [True, True, False, False]
    assert g.model.host.decode_beside is False


def test_in_flight_on_two_cards_both_halves_are_told():
    ref, _, _, _ = drive(False, PIPE_PLAN)
    seen = []
    out, g, jobs, _ = pipe(PIPE_PLAN, script = recording(seen), setup = with_threshold())
    assert out == ref
    halves = [(t, flag) for job, _, t, flag in seen if job == "b"]
    # Six chunks of 16, each in two halves, on the two threads
    assert len(halves) == 12
    assert {t.split("_")[0] for t, _ in halves} == {"exl3-prefill", "exl3-prefill2"}
    assert all(flag is True for _, flag in halves)
    assert told(seen, "a") == [False, False]
