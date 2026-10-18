"""
The weight of prompt ingestion's expert selections in the dynamic placement
(modules/block_sparse_mlp_cpu.py: hist_prefill_weight, _split_hist_apart, _split_hist_pf,
_split_hist_fold, and what save_expert_stats saves).

No GPU, extension or worker process: the split mixin is loaded on its own (with util/lane.py,
which it imports), the extension's map kernel is replaced by the same counting on the host, the
CPU expert host by one that records what it is handed (its fused decode issue counting into the
histogram it is given, as exl3_moe_split_issue_fused does), and the layer's swaps are recorded
instead of made.

Where a selection is counted is decided by the batch's rows, as the module decides its path:
a batch of at least the host's stream_min_rows rows (submit_prefill, the streamed path) is
prefill-size, from whichever thread; every smaller one takes a decode-size path (the fused
issue, or the copy path when that declines or is switched off) and counts in _split_hist.
"""

import importlib.util
import json
import os
import sys
import threading
import types

import pytest
import torch

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = "exl3_placement_stub"
E = 8           # experts of the layer
FIRST = 4       # physical slots [0, FIRST) are on the card, the rest on the CPU worker
ROWS = 32       # the host's stream_min_rows (EXL3_MOE_STREAM_MIN_ROWS, 32 by default)


def moe_split_map(sel, smap, hist, sel_cpu, first):
    # routing.cu: moe_split_map_kernel (one count per selection, in place translate)
    for i in range(sel.numel()):
        r = int(sel[i])
        hist[r] += 1.0
        p = int(smap[r])
        sel_cpu[i] = p - first if p >= first else -1
        sel[i] = p


def load():
    src = os.path.join(REPO, "exllamav3")

    def stub(name, **attrs):
        full = f"{PKG}.{name}" if name else PKG
        m = types.ModuleType(full)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[full] = m
        return m

    def real(name):
        full = f"{PKG}.{name}"
        spec = importlib.util.spec_from_file_location(full, os.path.join(src, *name.split(".")) + ".py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        return m

    stub("")
    for p in ("modules", "util"):
        stub(p)
    stub("ext", exllamav3_ext = types.SimpleNamespace(moe_split_map = moe_split_map))
    lane = real("util.lane")
    return real("modules.block_sparse_mlp_cpu"), lane


MOD, LANE = load()


class Host:
    """The CPU expert host as far as the split's submit goes: the fused decode issue counts every
    selection into the histogram it is handed (the extension's kernel doubles as moe_split_map),
    the copy path and the prefill path record the worker-side selection they get."""

    def __init__(self, stream_min_rows = ROWS, fused = True):
        self.stream_min_rows = stream_min_rows
        self.fused = fused
        self.calls = []

    def submit_issue_fused(self, layer_idx, y, sel, w, smap, hist, first):
        if not self.fused:
            return None                                         # declined: the copy path follows
        self.calls.append(("fused", hist))
        if hist is not None:
            for r in sel.view(-1).tolist():
                hist[r] += 1.0
        return ("fused job",)

    def submit_issue(self, layer_idx, y, sel_cpu, w):
        self.calls.append(("issue", sel_cpu.clone()))
        return ("issue job",)

    def submit_prefill(self, layer_idx, y, sel_cpu, w):
        self.calls.append(("prefill", sel_cpu.clone()))
        return torch.zeros(y.shape[0], 1)


class Layer(MOD.BlockSparseMLP_CPU):
    """A split layer as far as the placement goes: map, histograms, the submit's host, and a sweep
    that records the swaps it decides on."""

    def __init__(self, ip = None, host = None, key = "layer"):
        self.key = key
        self.num_experts = E
        self._cpu_init_state()
        self.cpu_split_first = FIRST
        self.cpu_layer_idx = 0
        self.device = None
        self.cpu_host = host or Host()
        with torch.inference_mode():                            # as at load (cpu_post_load)
            self._split_map = torch.arange(E, dtype = torch.long)
            self._split_hist = torch.zeros(E)
        self._split_selcpu_t = None
        self._swap_tick_count = 0
        self._split_step_count = 0
        self.swapped = []
        if ip is None:
            ip = types.SimpleNamespace(moe_cpu_swap_modules = [], moe_cpu_swap_pending = False)
        ip.moe_cpu_swap_modules.append(self)
        self.config = types.SimpleNamespace(infer_params = ip, directory = "/models/test")

    def _split_swap_experts(self, r_cold, r_hot, mp):
        self.swapped.append((r_cold, r_hot))
        slot = int(mp[r_cold])
        mp[r_cold], mp[r_hot] = int(mp[r_hot]), slot
        return True


def layers(n, **kw):
    ip = types.SimpleNamespace(moe_cpu_swap_modules = [], moe_cpu_swap_pending = False)
    return [Layer(ip, key = f"layer.{i}", **kw) for i in range(n)], ip


def submit(m, rows, *experts):
    """One batch of `rows` rows, each picking `experts`, through the split's submit (in inference
    mode, as the forward pass is on every thread that issues one)"""
    sel = torch.tensor([list(experts)] * rows, dtype = torch.long)
    return submit_sel(m, sel)


def submit_sel(m, sel):
    with torch.inference_mode():
        rows = sel.shape[0]
        return m.cpu_split_submit(torch.zeros(rows, 16), rows, sel, torch.ones(sel.shape))


def sweep(*ms):
    ip = ms[0].config.infer_params
    ip.moe_cpu_swap_pending = True
    MOD.run_pending_swap_sweeps(ip)


def on_lane(fn, index = 1, device = 0):
    out = {}

    def body():
        LANE.Lane(index, [device]).enter()
        try:
            out["value"] = fn()
        except BaseException as e:
            out["error"] = e

    t = threading.Thread(target = body)
    t.start()
    t.join(10)
    if "error" in out:
        raise out["error"]
    return out["value"]


def hist(m):
    return m._split_hist.tolist()


def pf(m):
    return None if m._split_hist_pf is None else m._split_hist_pf.tolist()


@pytest.fixture(autouse = True)
def _policy(monkeypatch):
    # a floor and a ratio that any clear difference passes
    monkeypatch.setenv("EXL3_MOE_CPU_SWAP_FLOOR", "0.1")
    monkeypatch.setenv("EXL3_MOE_CPU_SWAP_HYST", "1.5")
    for name in ("EXL3_MOE_HIST_PREFILL_WEIGHT", "EXL3_MOE_CPU_SWAP_MAX", "EXL3_MOE_CPU_SWAP_INTERVAL"):
        monkeypatch.delenv(name, raising = False)
    yield
    for name in ("lane", "host_waits"):
        if hasattr(LANE._tls, name):
            delattr(LANE._tls, name)


# --------------------------------------------------------------------------------------------
#   The weight
# --------------------------------------------------------------------------------------------

def test_the_weight_defaults_to_upstream_s(monkeypatch):
    assert MOD.hist_prefill_weight() == 1.0
    for said, weight in (("0", 0.0), ("0.004", 0.004), ("3", 3.0), ("-1", 0.0), ("nonsense", 1.0), ("1.0", 1.0)):
        monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", said)
        assert MOD.hist_prefill_weight() == weight


# --------------------------------------------------------------------------------------------
#   Where the selections are counted
# --------------------------------------------------------------------------------------------

def test_decode_size_batches_count_in_the_main_histogram_on_every_path(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0.004")
    # The fused issue: the histogram it is handed is the main one, counted in its kernel
    m = Layer()
    submit(m, 1, 0, 5)
    submit(m, ROWS - 1, 6, 1)
    assert [c[0] for c in m.cpu_host.calls] == ["fused", "fused"]
    assert all(c[1] is m._split_hist for c in m.cpu_host.calls)
    assert hist(m) == [1, ROWS - 1, 0, 0, 0, 1, ROWS - 1, 0] and m._split_hist_pf is None
    # The copy path, when the fused issue declines (a shape its single slot does not take)
    m = Layer(host = Host(fused = False))
    submit(m, 3, 5, 2)
    assert [c[0] for c in m.cpu_host.calls] == ["issue"]
    assert m.cpu_host.calls[0][1].tolist() == [[1, -1]] * 3
    assert hist(m) == [0, 0, 3, 0, 0, 3, 0, 0] and m._split_hist_pf is None
    # ... and when it is switched off (EXL3_MOE_SPLIT_FUSED=0)
    monkeypatch.setattr(MOD, "_split_fused", False)
    m = Layer()
    submit(m, 2, 7, 7)
    assert [c[0] for c in m.cpu_host.calls] == ["issue"]
    assert hist(m) == [0, 0, 0, 0, 0, 0, 0, 4] and m._split_hist_pf is None


def test_prompt_ingestion_is_counted_apart(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0.004")
    m = Layer()
    submit(m, 1, 0, 5)                                          # a decode-size batch
    assert hist(m) == [1, 0, 0, 0, 0, 1, 0, 0] and m._split_hist_pf is None
    submit(m, ROWS, 6, 1)                                       # a prompt chunk: prefill-size
    assert [c[0] for c in m.cpu_host.calls] == ["fused", "prefill"]
    assert m.cpu_host.calls[1][1].tolist() == [[2, -1]] * ROWS  # the worker's selection as before
    assert hist(m) == [1, 0, 0, 0, 0, 1, 0, 0]
    assert pf(m) == [0, ROWS, 0, 0, 0, 0, ROWS, 0]
    dec, pre = m.split_hits()
    assert dec is m._split_hist and pre is m._split_hist_pf


def test_the_threshold_is_the_host_s_own(monkeypatch):
    # The rows at which the module takes the prefill path decide, not a number of its own
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0")
    m = Layer(host = Host(stream_min_rows = 8))
    submit(m, 7, 5)
    submit(m, 8, 6)
    assert [c[0] for c in m.cpu_host.calls] == ["fused", "prefill"]
    assert hist(m)[5] == 7 and hist(m)[6] == 0 and pf(m)[6] == 8


def test_with_the_weight_at_one_a_prompt_counts_as_upstream_s(monkeypatch):
    for said in (None, "1", "1.0", "nonsense"):
        if said is None:
            monkeypatch.delenv("EXL3_MOE_HIST_PREFILL_WEIGHT", raising = False)
        else:
            monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", said)
        m = Layer()
        submit(m, ROWS, 6, 6)
        submit(m, 1, 5, 0)
        assert hist(m) == [1, 0, 0, 0, 0, 1, 2 * ROWS, 0]
        assert m._split_hist_pf is None                         # nothing made, nothing apart
        dec, pre = m.split_hits()
        assert dec is m._split_hist and float(pre.sum()) == 0


def test_the_weight_in_force_at_each_batch_decides(monkeypatch):
    # The weight may be set after the load (a server's flags): a batch counts by the weight of
    # its moment, and the fold by the weight of the sweep's
    m = Layer()
    submit(m, ROWS, 6, 6)
    assert m._split_hist_pf is None and hist(m)[6] == 2 * ROWS
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0")
    submit(m, ROWS, 7, 7)
    assert hist(m)[7] == 0 and pf(m)[7] == 2 * ROWS
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "1")
    submit(m, ROWS, 4, 4)
    assert hist(m)[4] == 2 * ROWS and pf(m)[4] == 0
    sweep(m)                                                    # folded at the sweep's weight, 1
    assert pf(m) == [0] * E
    assert m._split_hist[7] > 0


def test_a_static_split_counts_nothing(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0")
    m = Layer()
    m._split_map = None
    m._split_hist = None
    submit(m, ROWS, 6, 1)
    submit(m, 1, 5, 0)
    assert m._split_hist_pf is None
    assert m.cpu_host.calls[0][1].tolist() == [[2, -1]] * ROWS
    assert m.cpu_host.calls[1][1] is None                       # the fused issue gets no histogram


# --------------------------------------------------------------------------------------------
#   Lanes: the rows decide, whichever thread issues the batch
# --------------------------------------------------------------------------------------------

def test_a_chunk_on_the_prefill_thread_is_counted_apart(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0.004")
    m = Layer(host = Host(fused = False))                        # the copy path makes the module's buffer
    submit(m, 1, 0, 5)
    shared = m._split_selcpu_t
    assert shared is not None and shared.numel() == 64
    on_lane(lambda: submit(m, ROWS + 1, 6, 1))                  # beside the decode: > stream_min_rows
    assert hist(m) == [1, 0, 0, 0, 0, 1, 0, 0]
    assert pf(m) == [0, ROWS + 1, 0, 0, 0, 0, ROWS + 1, 0]
    # its selection still in a buffer of its own (66 selections: the module's 64 would have
    # been replaced by a larger one)
    assert m._split_selcpu_t is shared
    assert m.cpu_host.calls[-1][0] == "prefill" and m.cpu_host.calls[-1][1].tolist() == [[2, -1]] * (ROWS + 1)


def test_the_two_halves_of_a_chunk_are_counted_apart_on_their_cards(monkeypatch):
    # Generator.prefill_pipeline: the first card's layers on lane 1, the second's on lane 2
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0.004")
    (a, b), _ = layers(2)
    on_lane(lambda: submit(a, 40, 6, 7), index = 1, device = 0)
    on_lane(lambda: submit(b, 40, 5, 7), index = 2, device = 1)
    submit(a, 2, 4, 4)
    submit(b, 2, 4, 4)
    for m, hot in ((a, 6), (b, 5)):
        assert pf(m)[hot] == 40 and pf(m)[7] == 40 and hist(m)[hot] == 0
        assert hist(m)[4] == 4 and pf(m)[4] == 0


def test_a_short_chunk_counts_as_a_decode_size_batch_on_any_thread(monkeypatch):
    # A chunk below stream_min_rows (the last pages of a prompt, a short prompt) takes the fused
    # decode issue and costs the worker what a decode step costs: counted in full, as on the
    # v1.5.4 branch. The prefill thread waits for it anyway (side_min_tokens)
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0")
    m = Layer()
    on_lane(lambda: submit(m, ROWS - 1, 6, 6))
    assert m.cpu_host.calls[-1][0] == "fused"
    assert hist(m)[6] == 2 * (ROWS - 1) and m._split_hist_pf is None
    # and a prefill-size batch from the generator's own thread (a chunk run inline, or a
    # verification pass that wide) is counted apart
    submit(m, ROWS, 7, 7)
    assert pf(m)[7] == 2 * ROWS and hist(m)[7] == 0


# --------------------------------------------------------------------------------------------
#   The sweep
# --------------------------------------------------------------------------------------------

def traffic(m):
    """A prompt that hammers CPU-resident expert 6 and a generation that uses CPU-resident
    expert 5 and, less, the card's 0..3."""
    submit(m, 50, 6, 6, 6, 6)                                   # 200 selections of 6, prefill-size
    for _ in range(4):
        submit(m, 1, 5, 5, 0, 1)
    submit(m, 1, 2, 3)


def test_with_equal_weights_the_prompt_decides(monkeypatch):
    m = Layer()
    traffic(m)
    sweep(m)
    # upstream's ranking: 6 (200 hits) goes to the card first, then 5 (8)
    assert [hot for _, hot in m.swapped][:2] == [6, 5]
    assert m._split_hist_pf is None
    assert hist(m)[6] == 100.0                                  # 200, then the decay


def test_with_no_weight_the_generation_decides(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0")
    m = Layer()
    traffic(m)
    sweep(m)
    assert [hot for _, hot in m.swapped] == [5]                 # the prompt's expert stays where it is
    assert pf(m) == [0] * E
    assert hist(m)[6] == 0.0


def test_a_small_weight_lets_the_prompt_break_ties(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0.01")
    m = Layer()
    traffic(m)
    sweep(m)
    # 5 has 8 hits, 6 has 200 x 0.01 = 2: both clear the coldest of the card (1 hit), 5 first
    assert [hot for _, hot in m.swapped] == [5, 6]


def test_every_layer_is_folded_once_before_any_is_swept(monkeypatch):
    # A reader wrapped around a layer's sweep (the server's telemetry) must find the histogram
    # the sweep is about to use, also in the layers the swap budget does not reach
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0.5")
    monkeypatch.setenv("EXL3_MOE_CPU_SWAP_MAX", "1")
    ms, ip = layers(3)
    for m in ms:
        traffic(m)
    events, seen = [], []
    for i, m in enumerate(ms):
        fold, orig = m._split_hist_fold, m._split_sweep_layer

        def folding(w, i = i, fold = fold):
            events.append(("fold", i, w))
            return fold(w)

        def sweeping(budget, i = i, orig = orig):
            events.append(("sweep", i))
            seen.append([hist(x)[6] for x in ms])
            return orig(budget)

        m._split_hist_fold, m._split_sweep_layer = folding, sweeping
    sweep(*ms)
    assert events == [("fold", 0, 0.5), ("fold", 1, 0.5), ("fold", 2, 0.5), ("sweep", 0)]
    assert seen == [[100.0, 100.0, 100.0]]                      # 200 x 0.5 in all three, before any decay
    assert hist(ms[0])[6] == 50.0 and hist(ms[1])[6] == 100.0   # the swept layer decayed, the others not
    assert all(pf(m) == [0] * E for m in ms)
    # No second fold until the next sweep, and that one adds only what came since
    MOD.run_pending_swap_sweeps(ip)
    assert len(events) == 4
    submit(ms[1], ROWS, 6)
    sweep(*ms)
    assert [e for e in events[4:] if e[0] == "fold"] == [("fold", i, 0.5) for i in range(3)]
    assert hist(ms[1])[6] == 100.0 + ROWS * 0.5


def test_no_sweep_pending_no_fold(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0")
    m = Layer()
    traffic(m)
    MOD.run_pending_swap_sweeps(m.config.infer_params)
    assert float(m._split_hist_pf.sum()) == 200 and m.swapped == []


def test_a_sweep_held_back_while_a_chunk_is_under_way_folds_nothing(monkeypatch):
    # The inline sweep of an overdue placement is refused while a lane issues (F8's guard); the
    # counts stay apart until the sweep that does run
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0")
    monkeypatch.setenv("EXL3_MOE_CPU_SWAP_INTERVAL", "4")
    m = Layer()
    traffic(m)
    m._swap_tick_count = 100
    with LANE.Lane(1, [0]).passing():
        m._split_swap_tick()
    assert m.config.infer_params.moe_cpu_swap_pending and float(m._split_hist_pf.sum()) == 200
    m._split_swap_tick()
    assert not m.config.infer_params.moe_cpu_swap_pending
    assert float(m._split_hist_pf.sum()) == 0 and m.swapped == [(2, 5)]


def test_with_the_weight_at_one_the_sweep_is_upstream_s(monkeypatch):
    # The same selections, a prompt's as prefill-size batches in one layer and all as
    # decode-size batches in the other (upstream counts both in _split_hist): the histograms,
    # maps and swaps of every sweep identical bit for bit, and the weight never read by a sweep
    monkeypatch.setenv("EXL3_MOE_CPU_SWAP_MAX", "2")
    monkeypatch.setenv("EXL3_MOE_CPU_SWAP_INTERVAL", str(10 ** 9))   # sweeps only where the test asks
    a, b = Layer(), Layer()
    g = torch.Generator().manual_seed(7)
    weight = MOD.hist_prefill_weight

    def unread():
        raise AssertionError("the weight read by a sweep with nothing apart")

    def picks(rows, p):
        return torch.multinomial(p, rows * 2, replacement = True, generator = g).view(rows, 2)

    for rnd in range(6):
        monkeypatch.setattr(MOD, "hist_prefill_weight", weight)
        p = torch.rand(E, generator = g) ** 4 + 0.01                # a different hot set each round
        for _ in range(3):
            rows = int(torch.randint(ROWS, 3 * ROWS, (1,), generator = g))
            sel = picks(rows, p)
            submit_sel(a, sel.clone())
            for row in sel:
                submit_sel(b, row.view(1, 2).clone())
            for _ in range(5):
                sel = picks(1, p)
                submit_sel(a, sel.clone())
                submit_sel(b, sel.clone())
        assert a._split_hist_pf is None and torch.equal(a._split_hist, b._split_hist)
        assert {c[0] for c in a.cpu_host.calls} == {"prefill", "fused"}
        assert {c[0] for c in b.cpu_host.calls} == {"fused"}
        monkeypatch.setattr(MOD, "hist_prefill_weight", unread)
        sweep(a)
        sweep(b)
        assert torch.equal(a._split_hist, b._split_hist) and torch.equal(a._split_map, b._split_map)
        assert a.swapped == b.swapped
    assert a.swapped and any(float(x) != int(x) for x in a._split_hist)   # swaps made, decays seen


# --------------------------------------------------------------------------------------------
#   Promote-on-miss
# --------------------------------------------------------------------------------------------

def test_promote_on_miss_ignores_the_weight(monkeypatch):
    # Promote decays and snapshots _split_hist at every step and reads a step's routes from it;
    # it does not sweep, so nothing would fold a second histogram in: every selection counts in
    # _split_hist, as upstream
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0")
    steps = []
    monkeypatch.setattr(MOD, "_split_promote_step", lambda ip, reg, rows: steps.append(rows))
    (a, b), ip = layers(2)
    a._split_spares = b._split_spares = 2
    submit(a, ROWS, 6, 6)
    submit(b, ROWS, 6, 6)
    submit(a, 1, 5, 5)
    assert steps == [ROWS, 1]                                   # the first module steps promote
    for m in (a, b):
        assert m._split_hist_pf is None and hist(m)[6] == 2 * ROWS
    assert not ip.moe_cpu_swap_pending                          # and no sweep is asked for
    assert MOD.hist_prefill_weight() == 0.0                     # the variable still says what it says


def test_spares_on_a_later_layer_do_not_turn_the_weight_off(monkeypatch):
    # The mode is the first registered module's (_split_swap_tick): with it sweeping, every
    # layer is swept and folded
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0")
    (a, b), _ = layers(2)
    b._split_spares = 2
    submit(b, ROWS, 6, 6)
    assert pf(b)[6] == 2 * ROWS and hist(b)[6] == 0


# --------------------------------------------------------------------------------------------
#   The saved expert profile, the load and the unload
# --------------------------------------------------------------------------------------------

def saved(tmp_path, monkeypatch, ms):
    monkeypatch.setattr(MOD, "EXPERT_STATS_DIR", str(tmp_path))
    ms[0]._split_step_count = MOD._STATS_MIN_STEPS
    MOD.save_expert_stats(ms[0].config.infer_params)
    path = MOD._expert_stats_path(ms[0].config, E)
    with open(path) as f:
        return json.load(f)["layers"]


def test_the_profile_saves_what_the_sweep_ranks_by(tmp_path, monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0.25")
    (a, b), _ = layers(2)
    for m in (a, b):
        m._split_perm = None
    b._split_perm = [7, 6, 5, 4, 3, 2, 1, 0]                    # b's experts were permuted at load
    for m in (a, b):
        traffic(m)
    before = [(hist(m), pf(m)) for m in (a, b)]
    out = saved(tmp_path, monkeypatch, [a, b])
    want = [c + 0.25 * p for c, p in zip(hist(a), pf(a))]
    assert out["layer.0"] == want and want[6] == 50.0
    assert out["layer.1"] == list(reversed([c + 0.25 * p for c, p in zip(hist(b), pf(b))]))
    assert [(hist(m), pf(m)) for m in (a, b)] == before         # nothing folded by the save


def test_with_the_weight_at_one_the_profile_is_upstream_s(tmp_path, monkeypatch):
    (a,), _ = layers(1)
    a._split_perm = None
    traffic(a)
    monkeypatch.setattr(MOD, "hist_prefill_weight", lambda: pytest.fail("weight read"))
    assert saved(tmp_path, monkeypatch, [a])["layer.0"] == hist(a)


def test_the_unload_and_the_load_drop_the_counts_kept_apart(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0")
    m = Layer()
    traffic(m)
    assert m._split_hist_pf is not None
    m.cpu_host.unregister = lambda: None
    m._split_saved = ([], [], [], [], E, None, None)
    m.cpu_unload()
    assert m._split_hist_pf is None and m._split_hist is None
    assert m.config.infer_params.moe_cpu_swap_modules == []
    # a load starts with nothing apart
    m._split_hist_pf = torch.ones(E)
    m.cpu_split_first, m._split_perm, m._split_dynamic, m.device = FIRST, None, True, "cpu"
    m.cpu_post_load()
    assert m._split_hist_pf is None and hist(m) == [0] * E
