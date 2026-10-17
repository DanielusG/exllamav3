"""
Staging sets of a row table, per lane (modules/row_table.py with util/lane.py): two threads that
look rows up at the same time must not share a pinned staging set, because only the thread whose
upload is reading a set knows when it may be rewritten.

No GPU or extension is required: the table is held in RAM and the module is loaded on its own.
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
PKG = "exl3_rowtable_stub"


def load():
    src = os.path.join(REPO, "exllamav3")

    def stub(name, **attrs):
        full = f"{PKG}.{name}" if name else PKG
        m = types.ModuleType(full)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[full] = m

    def real(name):
        full = f"{PKG}.{name}"
        spec = importlib.util.spec_from_file_location(full, os.path.join(src, *name.split(".")) + ".py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        return m

    stub("")
    for p in ("modules", "util", "loader"):
        stub(p)
    stub("ext", exllamav3_ext = types.SimpleNamespace())
    stub("loader.safetensors", DiskTensorHandle = type("DiskTensorHandle", (), {}))
    lane = real("util.lane")
    return real("modules.row_table"), lane


RT, LANE = load()


class _Torch:
    """torch without pinned memory (a CPU build has none)."""

    def empty(self, *a, pin_memory = False, **k):
        return torch.zeros(*a, **k)

    def __getattr__(self, name):
        return getattr(torch, name)


@pytest.fixture(autouse = True)
def _host(monkeypatch):
    monkeypatch.setattr(RT, "torch", _Torch())
    yield
    for name in ("lane", "host_waits"):
        if hasattr(LANE._tls, name):
            delattr(LANE._tls, name)


def make_table(rows = 64, width = 4):
    t = RT.RowTable("emb", ["emb.weight"])
    t._set_stores([torch.arange(rows * width, dtype = torch.float32).view(rows, width)])
    t.row_words = width
    t.row_dtype = torch.float32
    return t


def look(table, ids):
    ids = torch.tensor(ids, dtype = torch.int64)

    def resolve(ids_, pin):
        return ids_, True

    return table.lookup(ids, ids.numel(), resolve, lambda rows, pin, U: rows, torch.device("cpu"), synced = True)


def test_one_thread_keeps_its_sets_as_before():
    t = make_table()
    out = look(t, [3, 5])
    assert out.tolist() == [[12.0, 13.0, 14.0, 15.0], [20.0, 21.0, 22.0, 23.0]]
    look(t, [1])
    # One set, taken and given back by each lookup
    assert list(t._lane_pins) == [0] and len(t._lane_pins[0]) == 1
    assert not t._lane_pins[0][0].held


def test_each_lane_stages_through_its_own_sets():
    t = make_table()
    look(t, [3, 5])
    main_set = t._pins[0]
    seen = {}

    def other():
        RT_lane = LANE.Lane(1, [])
        RT_lane.enter()
        seen["out"] = look(t, [7, 8, 9]).tolist()
        seen["set"] = t._pins[0]
        seen["pending"] = t._pending

    th = threading.Thread(target = other)
    th.start()
    th.join(10)
    assert seen["out"] == [[28.0, 29.0, 30.0, 31.0], [32.0, 33.0, 34.0, 35.0], [36.0, 37.0, 38.0, 39.0]]
    # The other thread never touched the set this thread's upload may still be reading
    assert seen["set"] is not main_set
    assert main_set.packed[:2].tolist() == [[12.0, 13.0, 14.0, 15.0], [20.0, 21.0, 22.0, 23.0]]
    assert sorted(t._lane_pins) == [0, 1]
    assert t._pins[0] is main_set and t._pending is not seen["pending"]


def test_a_queued_prefetch_is_found_and_dropped_whichever_lane_queued_it():
    t = make_table()
    done = types.SimpleNamespace(cancel = lambda: True, done = lambda: True, result = lambda: None)
    pin0, pin1 = RT._PinSet(), RT._PinSet()
    pin0.held = pin1.held = True
    e0 = {"ids": torch.tensor([1]), "pin": pin0, "future": done}
    e1 = {"ids": torch.tensor([2]), "pin": pin1, "future": done}
    t._pending.append(e0)
    LANE._tls.lane = 1
    t._pending.append(e1)
    assert t._match(torch.tensor([2])) is e1 and t._match(torch.tensor([1])) is None
    LANE._tls.lane = 0
    assert t._match(torch.tensor([1])) is e0 and t._match(torch.tensor([2])) is None
    t.drain_prefetch()
    assert t._lane_pending == {0: [], 1: []}
    assert not pin0.held and not pin1.held and t.prefetch_stats["retired"] == 2


def test_close_forgets_every_lane():
    t = make_table()
    look(t, [1])
    LANE._tls.lane = 1
    look(t, [2])
    t.close()
    assert t._lane_pins == {}


def test_close_unlocks_a_table_held_in_ram_and_forgets_every_lane(monkeypatch):
    # 1.6.2 can mlock() a table held in RAM (--ngram_lock): close() still unlocks it, whatever
    # the lanes did with their staging sets
    unlocked = []
    mem = types.ModuleType(f"{PKG}.util.memory")
    mem.unlock_host_ranges = lambda ranges: unlocked.append(list(ranges))
    monkeypatch.setitem(sys.modules, mem.__name__, mem)
    t = make_table()
    t._locked = [(4096, 8192)]
    look(t, [1])
    seen = {}

    def other():
        LANE.Lane(1, []).enter()
        seen["out"] = look(t, [2, 3]).tolist()

    th = threading.Thread(target = other)
    th.start()
    th.join(10)
    assert seen["out"] == [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]
    assert sorted(t._lane_pins) == [0, 1]
    t.close()
    assert unlocked == [[(4096, 8192)]] and t._locked == [] and t._lane_pins == {}
