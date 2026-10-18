"""
The expert split's mixin (modules/block_sparse_mlp_cpu.py) beside a second thread that issues
passes (util/lane.py): no inline placement sweep while a lane other than 0 is at work, and a
selection translate of its own for that lane.

No GPU or extension is required: the module is loaded on its own, with the rest of the package
stubbed, and the split module is a bare instance with the attributes these two paths read.
"""

import importlib.util
import os
import sys
import threading
import types

import pytest
import torch

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = "exl3_split_lanes_stub"


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
    ext = types.SimpleNamespace()
    stub("ext", exllamav3_ext = ext)
    lane = real("util.lane")
    return real("modules.block_sparse_mlp_cpu"), lane, ext


MIX, LANE, EXT = load()


@pytest.fixture(autouse = True)
def _lane_zero():
    yield
    for name in ("lane", "host_waits"):
        if hasattr(LANE._tls, name):
            delattr(LANE._tls, name)


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


# --------------------------------------------------------------------------------------------
#   The inline sweep
# --------------------------------------------------------------------------------------------

def overdue_module(monkeypatch):
    """The first registered split module, a sweep badly overdue (four intervals): the case in
    which upstream sweeps inline instead of waiting for the queue to drain."""
    swept = []
    monkeypatch.setattr(MIX, "run_pending_swap_sweeps", lambda ip: swept.append(ip))
    monkeypatch.setattr(MIX, "save_expert_stats", lambda ip: None)
    monkeypatch.setenv("EXL3_MOE_CPU_SWAP_INTERVAL", "4")
    m = MIX.BlockSparseMLP_CPU.__new__(MIX.BlockSparseMLP_CPU)
    m._split_spares = 0
    m._split_submit_rows = 1
    m._swap_tick_count = 100
    m._split_step_count = 100
    ip = types.SimpleNamespace(moe_cpu_swap_modules = [m], moe_cpu_swap_pending = False)
    m.config = types.SimpleNamespace(infer_params = ip)
    return m, ip, swept


def test_an_overdue_sweep_runs_inline_with_one_thread(monkeypatch):
    m, ip, swept = overdue_module(monkeypatch)
    m._split_swap_tick()
    assert swept == [ip] and ip.moe_cpu_swap_pending


def test_no_inline_sweep_from_the_prefill_thread(monkeypatch):
    # The sweep quiesces the cards and rewrites expert slots in place: from the prefill thread
    # the decode would go on issuing beside it. It stays pending for the queue's drain
    m, ip, swept = overdue_module(monkeypatch)
    on_lane(m._split_swap_tick)
    assert swept == [] and ip.moe_cpu_swap_pending


def test_no_inline_sweep_while_a_chunk_is_under_way_on_the_prefill_thread(monkeypatch):
    m, ip, swept = overdue_module(monkeypatch)
    lane = LANE.Lane(1, [0])
    with lane.passing():
        m._split_swap_tick()
    assert swept == [] and ip.moe_cpu_swap_pending
    m._split_swap_tick()
    assert swept == [ip]


def test_no_inline_sweep_from_the_second_half_s_thread(monkeypatch):
    # Generator.prefill_pipeline: the second half of a chunk is issued on lane 2, by a thread of
    # its own. The first registered split module is on the first card, so its tick comes from
    # lane 1 or the decode; from lane 2 (a model with no split layer on the first card) the
    # sweep stays pending as well
    m, ip, swept = overdue_module(monkeypatch)
    on_lane(m._split_swap_tick, index = 2, device = 1)
    assert swept == [] and ip.moe_cpu_swap_pending


def test_no_inline_sweep_while_either_half_of_a_chunk_is_under_way(monkeypatch):
    # The decode's tick (lane 0) while the second half is on the second card, then while the
    # first half alone is: no sweep until neither lane has a pass under way
    m, ip, swept = overdue_module(monkeypatch)
    first, second = LANE.Lane(1, [0]), LANE.Lane(2, [1])
    with second.passing():
        m._split_swap_tick()
    assert swept == []
    with first.passing():
        with second.passing():
            m._split_swap_tick()
        m._split_swap_tick()
    assert swept == [] and ip.moe_cpu_swap_pending
    m._split_swap_tick()
    assert swept == [ip]


# --------------------------------------------------------------------------------------------
#   The copy path's selection translate
# --------------------------------------------------------------------------------------------

def translating_module():
    calls = []

    def split_map(sel, mp, hist, out, first):
        # The worker-side selection: -1 for GPU-resident picks, the worker's index otherwise
        calls.append(out)
        out.copy_(torch.where(sel >= first, sel - first, torch.full_like(sel, -1)))

    EXT.moe_split_map = split_map
    m = MIX.BlockSparseMLP_CPU.__new__(MIX.BlockSparseMLP_CPU)
    m.cpu_split_first = 4
    m._split_map = torch.arange(8)
    m._split_hist = torch.zeros(8)
    m._split_selcpu_t = None
    return m, calls


def test_the_prefill_thread_translates_into_a_buffer_of_its_own():
    # The copy path (EXL3_MOE_SPLIT_FUSED=0, or a decode batch the fused issue declines) writes
    # the module's buffer; a chunk's selection read by its later kernels must not be in it
    m, calls = translating_module()
    dec = m._split_translate(torch.tensor([[1, 5]]))
    shared = m._split_selcpu_t
    assert dec.tolist() == [[-1, 1]] and dec.data_ptr() == shared.data_ptr()
    chunk = on_lane(lambda: m._split_translate(torch.tensor([[6, 2], [7, 4], [0, 5]])))
    assert chunk.tolist() == [[2, -1], [3, 0], [-1, 1]]
    assert m._split_selcpu_t is shared
    assert not (shared.data_ptr() <= chunk.data_ptr() < shared.data_ptr() + shared.numel() * shared.element_size())
    # The decode's next translate leaves the chunk's selection as it was
    m._split_translate(torch.tensor([[7, 6]]))
    assert chunk.tolist() == [[2, -1], [3, 0], [-1, 1]]


def test_the_two_halves_of_a_chunk_translate_into_buffers_of_their_own():
    # Two lanes of prompt chunks at once (Generator.prefill_pipeline): neither writes the
    # module's buffer, and each selection stays as it was while the other lane and the decode
    # translate
    m, calls = translating_module()
    m._split_translate(torch.tensor([[1, 5]]))
    shared = m._split_selcpu_t
    a = on_lane(lambda: m._split_translate(torch.tensor([[6, 2], [7, 4]])), index = 1, device = 0)
    b = on_lane(lambda: m._split_translate(torch.tensor([[5, 7], [0, 6]])), index = 2, device = 1)
    m._split_translate(torch.tensor([[7, 6]]))
    assert a.tolist() == [[2, -1], [3, 0]] and b.tolist() == [[1, 3], [-1, 2]]
    assert m._split_selcpu_t is shared
    spans = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in (shared, a, b)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert spans[i][1] <= spans[j][0] or spans[j][1] <= spans[i][0]
