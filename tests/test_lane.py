"""
Lanes (util/lane.py) and what is keyed by them without a module of its own to test it in: the
workspace cache of util/tensor.py and the token count Job._prefill_co hands to the generator.
No GPU or extension is required.
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
PKG = "exl3_lane_stub"


def load():
    src = os.path.join(REPO, "exllamav3")
    for name in ("", "util"):
        full = f"{PKG}.{name}" if name else PKG
        m = types.ModuleType(full)
        m.__path__ = []
        sys.modules[full] = m
    stub = types.ModuleType(f"{PKG}.util.device_copy")
    stub.to_device = lambda t, device: t
    sys.modules[stub.__name__] = stub

    def real(name):
        full = f"{PKG}.{name}"
        spec = importlib.util.spec_from_file_location(full, os.path.join(src, *name.split(".")) + ".py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        return m

    return real("util.lane"), real("util.tensor")


LANE, TENSOR = load()


def on_lane(fn, index = 1):
    """Run fn on a thread that entered a lane; returns what it returned."""
    out = {}

    def body():
        LANE.Lane(index, [0, 1]).enter()
        out["value"] = fn()

    t = threading.Thread(target = body)
    t.start()
    t.join(10)
    return out["value"]


def test_a_thread_is_on_lane_zero_until_it_enters_one():
    assert LANE.current_lane() == 0 and not LANE.host_waits()
    assert on_lane(lambda: (LANE.current_lane(), LANE.host_waits())) == (1, True)
    assert on_lane(LANE.current_lane, index = 2) == 2
    # ... and the thread that never asked stays where it was
    assert LANE.current_lane() == 0 and not LANE.host_waits()


def test_busy_while_a_pass_is_under_way_on_a_lane():
    lane = LANE.Lane(1, [0])
    assert not LANE.lanes_busy()
    with lane.passing():
        assert LANE.lanes_busy()
        with lane.passing():
            assert LANE.lanes_busy()
        assert LANE.lanes_busy()
    assert not LANE.lanes_busy()
    try:
        with lane.passing():
            raise RuntimeError("the pass failed")
    except RuntimeError:
        pass
    assert not LANE.lanes_busy()


def test_devices_of_the_models():
    def model(*devices):
        return types.SimpleNamespace(modules = [types.SimpleNamespace(device = d) for d in devices])

    assert LANE.cuda_devices(model("cuda:1", None, "cuda:0", "cuda:1"), None, model("cpu", "cuda:1")) == [0, 1]
    assert LANE.cuda_devices(model("cpu"), None) == []
    # and of a run of modules (Generator._pipe_plan: the two halves of a pass)
    mods = model("cpu", "cuda:0", None, "cuda:0", "cuda:1").modules
    assert LANE.module_devices(mods[:4]) == [0] and LANE.module_devices(mods[4:]) == [1]
    assert LANE.module_devices([]) == []


def test_each_of_two_lanes_beside_lane_zero_has_its_own_workspaces():
    # Generator.prefill_pipeline: lanes 1 and 2, one per card, beside the decode's
    cache = TENSOR.GTensorCache()
    a = cache.get("cpu", (4, 8), torch.float, "scratch")
    b = on_lane(lambda: cache.get("cpu", (4, 8), torch.float, "scratch"), index = 1)
    c = on_lane(lambda: cache.get("cpu", (4, 8), torch.float, "scratch"), index = 2)
    assert len({a.data_ptr(), b.data_ptr(), c.data_ptr()}) == 3
    assert on_lane(lambda: cache.get("cpu", (4, 8), torch.float, "scratch"), index = 2) is c


def test_each_lane_has_its_own_workspaces():
    cache = TENSOR.GTensorCache()
    a = cache.get("cpu", (4, 8), torch.float, "scratch")
    assert cache.get("cpu", (4, 8), torch.float, "scratch") is a
    key = cache.make_key("cpu", (4, 8), torch.float, "scratch")
    assert key == "cpu/(4, 8)/torch.float32/scratch"            # lane 0: the key it always had
    b, key_b = on_lane(lambda: (cache.get("cpu", (4, 8), torch.float, "scratch"),
                                cache.make_key("cpu", (4, 8), torch.float, "scratch")))
    assert b is not a and key_b == key + "/lane1"
    assert on_lane(lambda: cache.get("cpu", (4, 8), torch.float, "scratch")) is b
    # The sized-by-bucket workspaces go through the same key
    c = cache.get_bucketed("cpu", 5, torch.float, "bucket")
    d = on_lane(lambda: cache.get_bucketed("cpu", 5, torch.float, "bucket"))
    assert c.shape == d.shape == (5,) and c.data_ptr() != d.data_ptr()


def test_a_chunk_says_how_long_it_is():
    # Generator._prefill_round leaves a chunk beside the decode only if the function that
    # runs it carries the chunk's length (Job needs the extension to be imported, so this
    # reads the source: the attribute is set on the yielded function, before the yield)
    src = open(os.path.join(REPO, "exllamav3", "generator", "job.py")).read()
    body = src.split("def _prefill_co(self, results: list):", 1)[1].split("\n    def ", 1)[0]
    assert body.count("sent = yield run") == 1
    before = body.split("sent = yield run", 1)[0]
    assert "run.tokens = prefill_ids.shape[-1]" in before.split("def run():", 1)[1]
