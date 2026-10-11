"""
The weight of prompt ingestion's expert selections in the dynamic placement
(modules/block_sparse_mlp_cpu.py: hist_prefill_weight, _split_hist_pf, _split_hist_fold).

No GPU, extension or worker process: the split mixin is loaded on its own, with the extension's
map kernel replaced by the same counting on the host, and a layer whose swaps are recorded
instead of made.
"""

import importlib.util
import os
import sys
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = "exl3_placement_stub"
E = 8           # experts of the layer
FIRST = 4       # physical slots [0, FIRST) are on the card, the rest on the CPU worker


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

    def moe_split_map(sel, smap, hist, sel_cpu, first):
        # routing.cu: moe_split_map_kernel
        for i in range(sel.numel()):
            r = int(sel[i])
            hist[r] += 1.0
            p = int(smap[r])
            sel_cpu[i] = p - first if p >= first else -1
            sel[i] = p

    stub("")
    for p in ("modules", "util"):
        stub(p)
    stub("ext", exllamav3_ext = types.SimpleNamespace(moe_split_map = moe_split_map))
    real("util.lane")
    return real("modules.block_sparse_mlp_cpu")


MOD = load()


class Layer(MOD.BlockSparseMLP_CPU):
    """A split layer as far as the placement goes: map, histograms, and a sweep that records
    the swaps it decides on."""

    def __init__(self):
        self.key = "layer"
        self.num_experts = E
        self.cpu_split_first = FIRST
        self.device = None
        self._split_map = torch.arange(E, dtype = torch.long)
        self._split_hist = torch.zeros(E)
        self._split_hist_pf = torch.zeros(E)
        self._split_selcpu_t = None
        self._swap_tick_count = 0
        self.swapped = []

    def _split_swap_experts(self, r_cold, r_hot, mp):
        self.swapped.append((r_cold, r_hot))
        slot = int(mp[r_cold])
        mp[r_cold], mp[r_hot] = int(mp[r_hot]), slot
        return True


def pending(layers):
    return types.SimpleNamespace(moe_cpu_swap_pending = True, moe_cpu_swap_modules = list(layers))


def picks(*experts):
    return torch.tensor([list(experts)], dtype = torch.long)


@pytest.fixture(autouse = True)
def _policy(monkeypatch):
    # a floor and a ratio that any clear difference passes
    monkeypatch.setenv("EXL3_MOE_CPU_SWAP_FLOOR", "0.1")
    monkeypatch.setenv("EXL3_MOE_CPU_SWAP_HYST", "1.5")
    monkeypatch.delenv("EXL3_MOE_HIST_PREFILL_WEIGHT", raising = False)


def test_the_weight_defaults_to_upstream_s(monkeypatch):
    assert MOD.hist_prefill_weight() == 1.0
    for said, weight in (("0", 0.0), ("0.004", 0.004), ("3", 3.0), ("-1", 0.0), ("nonsense", 1.0)):
        monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", said)
        assert MOD.hist_prefill_weight() == weight


def test_prompt_ingestion_is_counted_apart():
    m = Layer()
    sel_cpu = m._split_translate(picks(0, 5, 5))                # a decode-size batch
    assert sel_cpu.tolist() == [[-1, 1, 1]]
    assert m._split_hist.tolist() == [1, 0, 0, 0, 0, 2, 0, 0] and float(m._split_hist_pf.sum()) == 0
    m._split_translate(picks(6, 6, 6, 1), prefill = True)       # a prompt chunk
    assert m._split_hist.tolist() == [1, 0, 0, 0, 0, 2, 0, 0]
    assert m._split_hist_pf.tolist() == [0, 1, 0, 0, 0, 0, 3, 0]
    dec, pf = m.split_hits()
    assert dec is m._split_hist and pf is m._split_hist_pf


def test_a_layer_without_the_second_histogram_counts_as_before():
    m = Layer()
    m._split_hist_pf = None
    m._split_translate(picks(6, 6), prefill = True)
    assert m._split_hist.tolist() == [0, 0, 0, 0, 0, 0, 2, 0]
    m._split_hist_fold(0.0)                                     # nothing to fold, nothing lost
    assert float(m._split_hist.sum()) == 2
    assert float(m.split_hits()[1].sum()) == 0


def traffic(m):
    """A prompt that hammers CPU-resident expert 6 and a generation that uses CPU-resident
    expert 5 and, less, the card's 0..3."""
    for _ in range(50):
        m._split_translate(picks(6, 6, 6, 6), prefill = True)
    for _ in range(4):
        m._split_translate(picks(5, 5, 0, 1))
    m._split_translate(picks(2, 3))


def test_with_equal_weights_the_prompt_decides(monkeypatch):
    m = Layer()
    traffic(m)
    MOD.run_pending_swap_sweeps(pending([m]))
    # upstream's ranking: 6 (200 hits) goes to the card first, then 5 (8)
    assert [hot for _, hot in m.swapped][:2] == [6, 5]
    assert float(m._split_hist_pf.sum()) == 0                   # folded, and started over
    assert m._split_hist.tolist()[6] == 100.0                   # 200 folded in, then the decay


def test_with_no_weight_the_generation_decides(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0")
    m = Layer()
    traffic(m)
    MOD.run_pending_swap_sweeps(pending([m]))
    assert [hot for _, hot in m.swapped] == [5]                 # the prompt's expert stays where it is
    assert float(m._split_hist_pf.sum()) == 0
    assert m._split_hist.tolist()[6] == 0.0


def test_a_small_weight_lets_the_prompt_break_ties(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_HIST_PREFILL_WEIGHT", "0.01")
    m = Layer()
    traffic(m)
    MOD.run_pending_swap_sweeps(pending([m]))
    # 5 has 8 hits, 6 has 200 x 0.01 = 2: both clear the coldest of the card (1 hit), 5 first
    assert [hot for _, hot in m.swapped] == [5, 6]


def test_every_layer_is_folded_before_any_is_swept(monkeypatch):
    # A reader wrapped around a layer's sweep (a server's telemetry) must find the histogram
    # the sweep is about to use, also in the layers the swap budget does not reach
    monkeypatch.setenv("EXL3_MOE_CPU_SWAP_MAX", "1")
    layers = [Layer(), Layer(), Layer()]
    for m in layers:
        traffic(m)
    seen = []
    for m in layers:
        orig = m._split_sweep_layer

        def sweep(budget, m = m, orig = orig):
            seen.append([float(x._split_hist_pf.sum()) for x in layers])
            return orig(budget)

        m._split_sweep_layer = sweep
    MOD.run_pending_swap_sweeps(pending(layers))
    assert seen == [[0.0, 0.0, 0.0]]                            # the budget stops after the first layer
    assert all(float(m._split_hist[6]) > 0 for m in layers)     # the prompt's hits are in all three


def test_no_sweep_pending_no_fold():
    m = Layer()
    traffic(m)
    ip = pending([m])
    ip.moe_cpu_swap_pending = False
    MOD.run_pending_swap_sweeps(ip)
    assert float(m._split_hist_pf.sum()) == 200 and m.swapped == []
