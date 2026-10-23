"""
Lanes of the CPU expert host (MoeCpuHost.split_lanes and the issue / collect paths): two threads
issuing jobs to the one worker at the same time.

No GPU, extension or worker process is required. The host's real issue and collect code runs
against a model of what it talks to: CUDA streams as queues of operations executed in order (flag
waits and writes, event records and waits), and a worker that serves the job ring strictly in
order, waiting for the data flag of the job at its head, as cpu/moe_handoff.cu does. The model
checks the two things the slot protocol exists for (a slot is not rewritten before its output was
read back; a readback finds its own job's output) and whether everything issued ever runs.

On 1.6.2 a decode-size job is issued by one extension call (exl3_moe_split_issue_fused) that
writes its descriptor into the ring and moves the ring's tail itself; the model's stand-in does
the same, on the same arrays as the host's Python path.

The module under test is loaded on its own, with the rest of the package stubbed.
"""

import collections
import gc
import importlib.util
import os
import sys
import threading
import time
import types
import weakref
from contextlib import contextmanager

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = "exl3_host_stub"
H = 8           # hidden width
TOPK = 2
CAP = 4         # rows per slot


# --------------------------------------------------------------------------------------------
#   The model of the cards and of the worker
# --------------------------------------------------------------------------------------------

class Stream:
    def __init__(self, sim, name):
        self.sim = sim
        self.name = name
        self.ops = collections.deque()
        self.issued = []            # (thread's lane, op kind) of everything ever enqueued
        sim.streams.append(self)

    def put(self, op):
        self.issued.append((self.sim.lane_of_caller(), op[0]))
        self.ops.append(op)

    def wait_event(self, ev):
        if isinstance(ev, VEvent):
            self.put(("vwait", ev, ev.version))
        else:
            self.put(("wait_event", ev))

    def synchronize(self):
        """A host wait for everything issued on this stream so far."""
        ev = Event(self.sim)
        self.put(("record", ev))
        ev.synchronize()


class Event:
    def __init__(self, sim):
        self.sim = sim
        self.done = threading.Event()

    def record(self, stream = None):
        (stream or self.sim.current()).put(("record", self))

    def synchronize(self):
        assert self.done.wait(self.sim.patience), "a host wait never ends"


class VEvent:
    """A CUDA event as the streamed path keeps one per weight-staging slot and device: recorded
    again for every batch, and a wait, on a stream or on the host, is for the record that was the
    most recent when the wait was made."""

    def __init__(self, sim):
        self.sim = sim
        self.version = 0
        self.done_v = 0

    def record(self, stream = None):
        self.version += 1
        (stream or self.sim.current()).put(("vrecord", self, self.version))

    def synchronize(self):
        v = self.version
        deadline = time.time() + self.sim.patience
        while self.done_v < v:
            assert time.time() < deadline, "a host wait never ends"
            time.sleep(0.0002)


class Sim:
    """Streams, flags and the worker. step() runs whatever can run and says whether it did."""

    def __init__(self, patience = 10.0):
        self.streams = []
        self.flags = collections.defaultdict(int)
        self.tls = threading.local()
        self.patience = patience
        self.lane_of_caller = lambda: 0
        self.default = Stream(self, "pass")
        self.host = None
        self.errors = []
        self.tenant = {}            # slot -> seq of the job whose inputs it holds
        self.served = []
        self.stop = False
        self.stager = None          # the worker's stager thread, for the streamed path

    def current(self):
        return getattr(self.tls, "stream", None) or self.default

    def _run_op(self, op) -> bool:
        kind = op[0]
        if kind == "wait":
            return self.flags[op[1]] >= op[2]
        if kind == "wait_event":
            return op[1].done.is_set()
        if kind == "vwait":
            return op[1].done_v >= op[2]
        if kind == "vrecord":
            op[1].done_v = max(op[1].done_v, op[2])
        elif kind in ("copy", "read"):
            op[1]()
        elif kind == "record":
            op[1].done.set()
        elif kind == "write":
            what, slot = op[1]
            if what == "ready":
                if self.flags[("consumed", slot)] < self.tenant.get(slot, 0):
                    self.errors.append(f"slot {slot} rewritten before job {self.tenant[slot]} was read back")
                self.tenant[slot] = op[2]
            elif what == "consumed":
                if self.flags[("done", slot)] != op[2]:
                    self.errors.append(f"job {op[2]} read back slot {slot} holding job {self.flags[('done', slot)]}")
            self.flags[op[1]] = op[2]
        return True

    def step(self) -> bool:
        progress = False
        for s in list(self.streams):
            while s.ops and self._run_op(s.ops[0]):
                s.ops.popleft()
                progress = True
        # The worker: the job at the head of the ring, once its data flag is up
        host = self.host
        head, tail = int(host.v_jobs_head[0]), int(host.v_jobs_tail[0])
        if head < tail:
            job = host.v_jobs[head % HOST.MOE_JOB_RING]
            seq = int(job[0]) | (int(job[HOST.MOE_JOB_SEQ_HI]) << 32)
            slot = int(job[4])
            if self.flags[("ready", slot)] >= seq:
                self.flags[("done", slot)] = seq
                self.served.append(seq)
                host.v_jobs_head[0] = head + 1
                progress = True
        if self.stager is not None and self.stager(self):
            progress = True
        return progress

    def settle(self):
        while self.step():
            pass

    def run(self):
        """In the background, for hosts that wait."""
        def loop():
            while not self.stop:
                if not self.step():
                    time.sleep(0.0002)
        t = threading.Thread(target = loop, daemon = True)
        t.start()
        return t


def fake_torch(sim):
    class Cuda:
        Stream = staticmethod(lambda device = None: Stream(sim, "handoff"))
        Event = staticmethod(lambda **kw: Event(sim))

        @staticmethod
        @contextmanager
        def stream(s):
            prev = getattr(sim.tls, "stream", None)
            sim.tls.stream = s
            try:
                yield
            finally:
                sim.tls.stream = prev

        @staticmethod
        @contextmanager
        def device(d):
            yield

        @staticmethod
        def current_stream(d = None):
            return sim.current()

    class Torch:
        cuda = Cuda

        def __getattr__(self, name):
            return getattr(torch, name)

    return Torch()


def fake_ext(sim):
    ext = types.SimpleNamespace()
    ext.exl3_moe_cpu_set_memops = lambda on: None
    ext.exl3_moe_flag_wait = lambda addr, value, abort: sim.current().put(("wait", addr, value))
    ext.moe_split_issue = lambda *a: sim.current().put(("kernel", "issue"))
    ext.moe_split_collect_add = lambda *a: sim.current().put(("kernel", "collect"))

    def flag_write(addr, value):
        if addr[0] == "ready" and not sim.host._issue_lock.locked():
            sim.errors.append("a data flag enqueued outside the issue lock")
        sim.current().put(("write", addr, value))

    ext.exl3_moe_flag_write = flag_write

    def issue_fused(shm_base, seq, layer, rows, topk, slot, prev_seq, sel, map_, hist, y, w,
                    h_sel, h_x, h_w, dev_count, hi, first_cpu, data_ready, consumed, abort):
        # As cpu/moe_handoff.cu: the descriptor at the ring's tail and the tail one higher, with no
        # lock of its own and no thread switch in between (the extension keeps the interpreter);
        # then the consumed wait, the issue kernel and the data flag on the current stream
        host = sim.host
        if not host._issue_lock.locked():
            sim.errors.append("a fused issue outside the issue lock")
        tail = int(host.v_jobs_tail[0])
        if tail - int(host.v_jobs_head[0]) >= HOST.MOE_JOB_RING - 4:
            return False
        job = host.v_jobs[tail % HOST.MOE_JOB_RING]
        job[0] = seq & 0xFFFFFFFF
        job[HOST.MOE_JOB_SEQ_HI] = seq >> 32
        job[1], job[2], job[3], job[4], job[5] = layer, rows, topk, slot, 2
        host.v_jobs_tail[0] = tail + 1
        if prev_seq:
            sim.current().put(("wait", consumed, prev_seq))
        sim.current().put(("kernel", "issue"))
        sim.current().put(("write", data_ready, seq))
        return True

    def collect_fused(final, h_out, dev_count, slot, ho, done, seq, consumed, abort):
        sim.current().put(("wait", done, seq))
        sim.current().put(("kernel", "collect"))
        sim.current().put(("write", consumed, seq))

    ext.exl3_moe_split_issue_fused = issue_fused
    ext.exl3_moe_split_collect_fused = collect_fused
    return ext


# --------------------------------------------------------------------------------------------
#   Loading the host module
# --------------------------------------------------------------------------------------------

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
    for p in ("model", "util", "modules"):
        stub(p)
    # (imported by the streamed path, used by its batched tier only)
    stub("modules.moe_batch_recon", plan_groups = None)
    stub("ext", exllamav3_ext = types.SimpleNamespace(
        exl3_moe_cpu_set_memops = lambda on: None, exl3_moe_cpu_core_order = lambda: ([], 0)))
    stub("util.device_copy", host_to_device = lambda t, device: t)
    stub("model.moe_cpu_affinity", plan_host_cpus = None, default_worker_threads = None,
         apply_process_affinity = None, process_cpus = None)
    stub("util.misc", Cleanupper = lambda: None, install_parent_death_signal = lambda: None)
    stub("util.shm", check_shm_capacity = lambda *a: None)
    stub("util.memory", check_host_memory = lambda *a, **k: None, windows_memory_status = lambda: None)
    stub("model.model_tp_cuda", cuda_host_register = None, cuda_host_unregister = None,
         cuda_host_get_device_pointer = None, CUDA_HOST_REGISTER_PORTABLE = 1, CUDA_HOST_REGISTER_MAPPED = 2)
    lane = real("util.lane")
    return real("model.moe_cpu_host"), lane


HOST, LANE = load()


def make_host(monkeypatch, sim, slots):
    monkeypatch.setattr(HOST, "torch", fake_torch(sim))
    monkeypatch.setattr(HOST, "ext", fake_ext(sim))
    h = HOST.MoeCpuHost.__new__(HOST.MoeCpuHost)
    h.num_slots = slots
    h.cap_rows = CAP
    h.seq = 0
    h.next_slot = 0
    h.slot_last_seq = [0] * HOST.MOE_MAX_SLOTS
    h._issue_lock = threading.Lock()
    h._lanes = None
    h._handoff = {}
    h._handoff_held = {}
    h.base_ptr = 0
    h.gpu_base_ptr = 0
    h.dev_count = {}
    h.specs = [dict(hi = H, ho = H, topk = TOPK)] * 64
    h.v_jobs = np.zeros((HOST.MOE_JOB_RING, HOST.MOE_JOB_BYTES // 4), dtype = np.uint32)
    h.v_jobs_tail = np.zeros(1, dtype = np.uint32)
    h.v_jobs_head = np.zeros(1, dtype = np.uint32)
    h.v_abort = np.zeros(1, dtype = np.uint32)
    h.proc = types.SimpleNamespace(is_alive = lambda: True)
    h.slots = [dict(
        x = torch.zeros((CAP, H), dtype = torch.half),
        sel = torch.zeros((CAP, TOPK), dtype = torch.int32),
        w = torch.zeros((CAP, TOPK), dtype = torch.half),
        out = torch.zeros((CAP, H)),
        x_dev = 0, sel_dev = 0, w_dev = 0, out_dev = 0,
        data_ready = ("ready", s), done = ("done", s), consumed = ("consumed", s),
    ) for s in range(slots)]
    sim.host = h
    sim.lane_of_caller = LANE.current_lane
    return h


@pytest.fixture(autouse = True)
def _lane_zero():
    yield
    for name in ("lane", "host_waits"):
        if hasattr(LANE._tls, name):
            delattr(LANE._tls, name)


def rows(n):
    return (torch.zeros((n, H), dtype = torch.half), torch.zeros((n, TOPK), dtype = torch.long),
            torch.zeros((n, TOPK), dtype = torch.half))


def decode_job(h, layer):
    """One layer of a decode step: issue, the card's own expert work, collect."""
    y, sel, w = rows(2)
    handle = h.submit_issue_fused(layer, y, sel, w, None, None, 0)
    HOST.ext.moe_split_issue()      # stands for the fused expert kernels
    h.submit_collect_fused(handle, torch.zeros((2, H)))


def chunked(h, layer, n_rows, between = None):
    """One layer of a prefill pass on the chunked path: issue, whatever the card does
    meanwhile, collect."""
    y, sel, w = rows(n_rows)
    out = torch.zeros((n_rows, H))
    jobs, rtmp = h._issue_compute(layer, y, sel, w, h.specs[layer], out, H)
    if between is not None:
        between()
    h._collect_compute(jobs, out, rtmp, H)


# --------------------------------------------------------------------------------------------
#   Slots
# --------------------------------------------------------------------------------------------

def test_split_lanes(monkeypatch):
    sim = Sim()
    assert make_host(monkeypatch, sim, 4).split_lanes() == (2, 2)
    h = make_host(monkeypatch, sim, 6)
    assert h._lane_slots() == 6
    assert h.split_lanes() == (2, 4)
    assert h.split_lanes() == (2, 4)                # once
    assert h._lane_slots() == 2
    assert make_host(monkeypatch, sim, 2).split_lanes() == (1, 1)
    h1 = make_host(monkeypatch, sim, 1)
    assert h1.split_lanes() is None and h1._lanes is None


def test_split_lanes_for_a_lane_of_prompt_chunks_per_card(monkeypatch):
    # Generator.prefill_pipeline: lane 0 for the decode, lanes 1 and 2 for the two halves of a chunk
    sim = Sim()
    assert make_host(monkeypatch, sim, 6).split_lanes(lanes = 3) == (2, 2, 2)
    assert make_host(monkeypatch, sim, 8).split_lanes(lanes = 3) == (2, 3, 3)
    assert make_host(monkeypatch, sim, 5).split_lanes(lanes = 3) == (2, 2, 1)
    assert make_host(monkeypatch, sim, 4).split_lanes(lanes = 3) == (2, 1, 1)
    assert make_host(monkeypatch, sim, 3).split_lanes(lanes = 3) == (1, 1, 1)
    h2 = make_host(monkeypatch, sim, 2)
    assert h2.split_lanes(lanes = 3) is None and h2._lanes is None
    # It replaces the split of the prefill thread alone: lane 0 keeps its slots, lane 1 the
    # bottom of its range. Asking for fewer lanes afterwards changes nothing
    h = make_host(monkeypatch, sim, 6)
    assert h.split_lanes() == (2, 4)
    assert h.split_lanes(lanes = 3) == (2, 2, 2)
    assert [h._lanes[i][:2] for i in range(3)] == [[0, 2], [2, 2], [4, 2]]
    assert h.split_lanes() == (2, 2, 2)
    # Each lane rotates over its own range
    taken = {}
    for lane in (0, 1, 2):
        LANE._tls.lane = lane
        taken[lane] = [h._take_slot() for _ in range(4)]
    assert taken == {0: [0, 1, 0, 1], 1: [2, 3, 2, 3], 2: [4, 5, 4, 5]}


def test_one_thread_alone_is_what_it_was(monkeypatch):
    # No lanes: the slots in rotation, a window as deep as the slots, everything on the
    # pass's stream
    sim = Sim()
    h = make_host(monkeypatch, sim, 4)
    chunked(h, 0, 6 * CAP)
    for layer in range(1, 4):
        decode_job(h, layer)
    assert [int(j[4]) for j in h.v_jobs[:9]] == [0, 1, 2, 3, 0, 1, 2, 3, 0]
    assert [int(j[5]) for j in h.v_jobs[:9]] == [0] * 6 + [2] * 3
    assert len(sim.streams) == 1
    sim.settle()
    assert not sim.default.ops and sim.errors == [] and sim.served == list(range(1, 10))


def interleaved(monkeypatch, split, at = 5, decodes = 1):
    """A prefill pass with more chunks than slots, and decode jobs that another thread issues
    as the pass's chunk number `at` is about to take its slot (the fifth: right after the pass
    has filled its window). Both on the pass's stream, no host waits: the slots are the only
    difference between the two runs."""
    sim = Sim()
    h = make_host(monkeypatch, sim, 4 if not split else 6)
    if split:
        assert h.split_lanes() == (2, 4)
    take = h._take_slot
    calls = {"n": 0, "busy": False}

    def take_slot():
        if not calls["busy"]:
            calls["n"] += 1
            if calls["n"] == at:
                # The chunk is about to take its slot: the other thread gets in first
                calls["busy"] = True
                LANE._tls.lane = 0
                for layer in range(decodes):
                    decode_job(h, 1 + layer)
                LANE._tls.lane = 1 if split else 0
                calls["busy"] = False
        return take()

    h._take_slot = take_slot
    LANE._tls.lane = 1 if split else 0
    chunked(h, 0, 6 * CAP)
    sim.settle()
    return sim, h


def test_two_threads_on_one_rotation_stop_the_stream(monkeypatch):
    # The hazard the lanes exist for. The decode job takes the slot the fifth chunk was going
    # to take; the chunk lands in the next one, whose tenant (the second chunk) is read back
    # further down the same stream: the stream waits for itself
    sim, h = interleaved(monkeypatch, split = False)
    stuck = sim.default.ops[0]
    assert stuck[0] == "wait" and stuck[1] == ("consumed", 1)
    assert len(sim.served) < 7


def test_with_a_range_of_slots_each_they_do_not(monkeypatch):
    sim, h = interleaved(monkeypatch, split = True)
    assert not sim.default.ops and sim.errors == []
    assert sorted(sim.served) == list(range(1, 8))
    # The decode job took a slot of lane 0, the chunks the four of lane 1
    kinds = [(int(j[5]), int(j[4])) for j in h.v_jobs[:7]]
    assert [s for k, s in kinds if k == 0] == [2, 3, 4, 5, 2, 3]
    assert [s for k, s in kinds if k == 2] == [0]


def test_the_fused_decode_issue_keeps_to_its_lane_s_slots(monkeypatch):
    # 1.6.2's fused issue takes its slot from the decode lane's range too. Three decode jobs
    # while the pass's window is still open (its fourth chunk about to issue): on one rotation
    # the third would land in the slot of the pass's first chunk, not read back yet, and wait on
    # the stream for a readback issued behind it
    sim, h = interleaved(monkeypatch, split = True, at = 4, decodes = 3)
    assert not sim.default.ops and sim.errors == []
    assert sorted(sim.served) == list(range(1, 10))
    kinds = [(int(j[5]), int(j[4])) for j in h.v_jobs[:9]]
    assert [s for k, s in kinds if k == 2] == [0, 1, 0]
    assert {s for k, s in kinds if k == 0} == {2, 3, 4, 5}


# --------------------------------------------------------------------------------------------
#   A lane that shares the pass's stream
# --------------------------------------------------------------------------------------------

def test_the_chunked_path_keeps_its_waits_off_the_shared_stream(monkeypatch):
    sim = Sim()
    h = make_host(monkeypatch, sim, 6)
    h.split_lanes()
    bg = sim.run()
    result = {}

    def prefill():
        try:
            LANE.Lane(1, [0]).enter()
            for layer in range(3):
                chunked(h, layer, 9 * CAP, between = lambda: HOST.ext.moe_split_issue())
                # On return the output is there: nothing is left to run
                result[layer] = (len(sim.served), [len(s.ops) for s in sim.streams])
        except BaseException as e:
            result["error"] = e

    t = threading.Thread(target = prefill)
    t.start()
    t.join(20)
    sim.stop = True
    bg.join(5)
    assert not t.is_alive() and "error" not in result, result.get("error")
    assert sim.errors == []
    assert [result[l][0] for l in range(3)] == [9, 18, 27]
    assert all(n == 0 for l in range(3) for n in result[l][1])
    # On the pass's stream: the event the copies wait for and the card's own work. Every flag
    # wait and write went to the handoff stream
    assert len(sim.streams) == 2
    assert {kind for _, kind in sim.default.issued} == {"record", "kernel"}
    handoff = sim.streams[1]
    assert {kind for _, kind in handoff.issued} == {"wait_event", "wait", "write", "record"}


def issue_and_drop(h, n_rows):
    """Issue a layer the way the streamed prefill does, from temporaries nobody else keeps;
    returns what is needed to collect it and weak references to the sources of its copies."""
    y, sel, w = rows(n_rows)
    out = torch.zeros((n_rows, H))
    jobs, rtmp = h._issue_compute(0, y, sel, w, h.specs[0], out, H)
    refs = [weakref.ref(y), weakref.ref(w)]
    del y, sel, w
    gc.collect()
    return jobs, out, rtmp, refs


def test_the_sources_of_the_copies_outlive_the_call_that_issued_them(monkeypatch):
    # Found on the real model (NaN after a prompt ingested on the second thread): the copies
    # of all but the first jobs run on the handoff stream when the worker gets to them, after
    # the call has returned and its temporaries are gone; the allocator had handed their
    # memory to the pass's next kernels by then
    sim = Sim()
    h = make_host(monkeypatch, sim, 6)
    h.split_lanes()
    bg = sim.run()
    result = {}

    def prefill():
        try:
            LANE.Lane(1, [0]).enter()
            jobs, out, rtmp, refs = issue_and_drop(h, 9 * CAP)
            result["held"] = [r() is not None for r in refs]
            h._collect_compute(jobs, out, rtmp, H)
            gc.collect()
            result["after"] = [r() is not None for r in refs]
            result["left"] = len(h._handoff_held)
        except BaseException as e:
            result["error"] = e

    t = threading.Thread(target = prefill)
    t.start()
    t.join(20)
    sim.stop = True
    bg.join(5)
    assert "error" not in result, result.get("error")
    assert result["held"] == [True, True]           # until the host has seen the last readback
    assert result["after"] == [False, False] and result["left"] == 0


def test_on_the_pass_s_own_stream_nothing_is_held(monkeypatch):
    # One thread, one stream: the copies are ahead, in stream order, of whatever reuses the
    # memory, and the sources go when the call returns, as upstream
    sim = Sim()
    h = make_host(monkeypatch, sim, 4)
    jobs, out, rtmp, refs = issue_and_drop(h, 6 * CAP)
    assert [r() is not None for r in refs] == [False, False] and h._handoff_held == {}
    h._collect_compute(jobs, out, rtmp, H)
    sim.settle()
    assert sim.errors == [] and not sim.default.ops


def test_a_pass_that_never_collects_does_not_pile_up_sources(monkeypatch):
    sim = Sim()
    h = make_host(monkeypatch, sim, 6)
    h.split_lanes()
    LANE.Lane(1, [0]).enter()
    kept = []
    for _ in range(80):
        y, sel, w = rows(CAP)
        kept.append(h._issue_compute(0, y, sel, w, h.specs[0], torch.zeros((CAP, H)), H))
    assert len(h._handoff_held) == 64


def test_decode_and_prefill_at_the_same_time(monkeypatch):
    # Two real threads, their publications interleaved by the scheduler; the decode thread
    # waits for its stream after every step (its sampling), the prefill thread before every
    # layer (its selection counts) and in the collect
    sim = Sim()
    h = make_host(monkeypatch, sim, 6)
    h.split_lanes()
    bg = sim.run()
    failed = []
    stop = threading.Event()
    steps = {"decode": 0, "layers": 0}

    def decode():
        try:
            while not stop.is_set():
                for layer in range(12):
                    decode_job(h, layer)
                sim.default.synchronize()
                steps["decode"] += 1
        except BaseException as e:
            failed.append(e)

    def prefill():
        try:
            LANE.Lane(1, [0]).enter()
            for layer in range(40):
                sim.default.synchronize()
                chunked(h, layer % 12, (3 + layer % 7) * CAP, between = lambda: HOST.ext.moe_split_issue())
                steps["layers"] += 1
        except BaseException as e:
            failed.append(e)
        finally:
            stop.set()

    threads = [threading.Thread(target = f) for f in (decode, prefill)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(40)
    alive = [t for t in threads if t.is_alive()]
    time.sleep(0.05)
    sim.stop = True
    bg.join(5)
    assert not alive and not failed, failed
    assert sim.errors == []
    assert steps["layers"] == 40 and steps["decode"] >= 3
    # Everything published was served, in the order it was published
    assert int(h.v_jobs_head[0]) == int(h.v_jobs_tail[0]) == len(sim.served)
    assert sim.served == sorted(sim.served)
    assert all(not s.ops for s in sim.streams)
    # The decode never issued on the handoff stream, the prefill never made the pass's wait
    handoff = sim.streams[1]
    assert {lane for lane, _ in handoff.issued} == {1}
    assert not [1 for lane, kind in sim.default.issued if lane == 1 and kind in ("wait", "wait_event")]


# --------------------------------------------------------------------------------------------
#   1.6.2: the fused decode issue, the sequence number's two halves, the streamed path
# --------------------------------------------------------------------------------------------

class _SlowRing:
    """The job ring, with lane 1's first descriptor write held up for `hold` seconds: the
    interpreter switching threads between _publish's read of the ring's tail and its write of
    the descriptor, which are bytecodes apart (the fused issue is one extension call)."""

    def __init__(self, arr, hold):
        self.arr = arr
        self.hold = hold
        self.held = threading.Event()

    def __getitem__(self, i):
        if LANE.current_lane() == 1 and not self.held.is_set():
            self.held.set()
            time.sleep(self.hold)
        return self.arr[i]


def test_a_fused_decode_issue_waits_for_a_chunk_that_is_publishing(monkeypatch):
    # 1.6.2's fused issue reads the ring's tail and writes it back one higher inside the
    # extension, with no lock of its own. A chunk's job published from Python beside it, the
    # tail read and the tail written with a thread switch in between, would hand both jobs the
    # same ring entry: one of them lost, the stream that waits for it stopped for good
    sim = Sim()
    h = make_host(monkeypatch, sim, 6)
    h.split_lanes()
    ring = h.v_jobs = _SlowRing(h.v_jobs, 0.2)
    failed = []

    def prefill():
        try:
            LANE.Lane(1, [0]).enter()
            y, sel, w = rows(CAP)
            h._issue_compute(0, y, sel, w, h.specs[0], torch.zeros((CAP, H)), H)
        except BaseException as e:
            failed.append(e)

    def decode():
        try:
            assert ring.held.wait(5)
            decode_job(h, 1)
        except BaseException as e:
            failed.append(e)

    threads = [threading.Thread(target = f) for f in (prefill, decode)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not failed, failed
    # One ring entry each, in the order the lock let them in
    assert int(h.v_jobs_tail[0]) == 2
    assert [int(ring.arr[i][0]) for i in range(2)] == [1, 2]
    assert [int(ring.arr[i][5]) for i in range(2)] == [0, 2]
    sim.settle()
    assert sim.errors == [] and sim.served == [1, 2]
    assert not sim.default.ops


def test_a_fused_issue_that_finds_the_ring_full_waits_for_the_worker_and_retries(monkeypatch):
    sim = Sim()
    h = make_host(monkeypatch, sim, 4)
    h.v_jobs_tail[0] = HOST.MOE_JOB_RING - 4         # within the safety margin of a full ring
    h.seq = HOST.MOE_JOB_RING - 4

    def drain():
        time.sleep(0.05)
        h.v_jobs_head[0] = HOST.MOE_JOB_RING - 4

    t = threading.Thread(target = drain)
    t.start()
    decode_job(h, 3)
    t.join(5)
    job = h.v_jobs[(HOST.MOE_JOB_RING - 4) % HOST.MOE_JOB_RING]
    assert int(h.v_jobs_tail[0]) == HOST.MOE_JOB_RING - 3 and int(job[0]) == HOST.MOE_JOB_RING - 3
    sim.settle()
    assert sim.errors == [] and not sim.default.ops


def test_a_descriptor_carries_both_halves_of_its_sequence_number(monkeypatch):
    # 1.6.2's descriptors and flags are 32 bits wide and compared cyclically by the worker; the
    # chunked path's descriptor goes through _set_job_seq as upstream's does
    sim = Sim()
    h = make_host(monkeypatch, sim, 4)
    h.seq = (1 << 32) + 5
    y, sel, w = rows(CAP)
    out = torch.zeros((CAP, H))
    jobs, rtmp = h._issue_compute(0, y, sel, w, h.specs[0], out, H)
    job = h.v_jobs[0]
    assert int(job[0]) == 6 and int(job[HOST.MOE_JOB_SEQ_HI]) == 1
    h._collect_compute(jobs, out, rtmp, H)
    sim.settle()
    assert sim.errors == [] and sim.served == [(1 << 32) + 6] and not sim.default.ops


def _host_function(name):
    import ast
    with open(os.path.join(REPO, "exllamav3", "model", "moe_cpu_host.py")) as f:
        tree = ast.parse(f.read())
    host = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoeCpuHost")
    return next(n for n in host.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_the_streamed_prefill_never_makes_a_shared_stream_wait():
    # A chunk on the prefill thread streams its hot experts (1.6.2: by DMA straight from the
    # pinned arena, or through the stage ring) on the copy stream. The pass's stream is the one
    # the decode's kernels run on: it must never wait for the worker or for a DMA. The worker's
    # flags are waited for on the copy stream only, and the wait for a landed batch is the
    # host's on a lane that shares the stream (host_waits)
    import ast
    fn = _host_function("_submit_prefill_streamed")
    parents = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def ancestors(node):
        while node in parents:
            node = parents[node]
            yield node

    def in_copy_stream(node):
        for a in ancestors(node):
            if isinstance(a, ast.With):
                for item in a.items:
                    call = item.context_expr
                    if isinstance(call, ast.Call) and ast.unparse(call.func) == "torch.cuda.stream" \
                            and ast.unparse(call.args[0]) == "copy_stream":
                        return True
        return False

    flag_waits = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "ext.exl3_moe_flag_wait"]
    assert flag_waits and all(in_copy_stream(n) for n in flag_waits)

    stream_waits = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("wait_event", "wait_stream")
                    and ast.unparse(n.func.value) == "torch.cuda.current_stream()"]
    assert len(stream_waits) == 1
    guard = next(a for a in ancestors(stream_waits[0]) if isinstance(a, ast.If))
    assert ast.unparse(guard.test) == "host_waits()"
    assert any(n is stream_waits[0] for stmt in guard.orelse for n in ast.walk(stmt))
    assert any(isinstance(n, ast.Call) and ast.unparse(n.func).endswith(".synchronize") for stmt in guard.body for n in ast.walk(stmt))
    # The tail goes through the lane-aware issue and collect
    calls = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert {"self._issue_compute", "self._collect_compute"} <= calls


# --------------------------------------------------------------------------------------------
#   Two lanes of prompt chunks on the streamed path (Generator.prefill_pipeline)
# --------------------------------------------------------------------------------------------

E = 4               # experts of a layer, all of them streamed
EXP_B = 96          # bytes of an expert (three projections of 32)
ROWS = 8


class FakeVram:
    """A weight-staging slot's VRAM side on one card: which expert each block holds, written by
    the copies when the copy stream gets to them."""

    def __init__(self, device, ws):
        self.device = device
        self.ws = ws
        self.blocks = {}

    def __getitem__(self, sl):
        return FakeVramSlice(self, sl.start or 0)


class FakeVramSlice:
    def __init__(self, vram, start):
        self.vram = vram
        self.start = start

    def view(self, *shape):
        return self

    def copy_(self, src, non_blocking = False):
        sim = src.sim
        vram, start = self.vram, self.start

        def run():
            if isinstance(src, ArenaSlice):
                # Pinned arena: one expert's block straight out of the arena
                vram.blocks[start // (EXP_B // 2)] = src.expert
            else:
                # Staged: whatever the worker's stager left in the slot's pinned side
                staged = sim.pinned.get(src.ws)
                if staged is None:
                    sim.errors.append(f"cuda:{vram.device} slot {vram.ws}: copied an empty pinned slot")
                    return
                for bi, e in enumerate(staged[1]):
                    vram.blocks[bi] = e

        sim.current().put(("copy", run))
        return self


class ArenaSlice:
    def __init__(self, sim, expert):
        self.sim = sim
        self.expert = expert


class FakeArena:
    def __init__(self, sim):
        self.sim = sim

    def __getitem__(self, sl):
        return ArenaSlice(self.sim, (sl.start * 2) // EXP_B)


class PinnedSlice:
    def __init__(self, sim, ws):
        self.sim = sim
        self.ws = ws


class FakePinned:
    def __init__(self, sim, ws):
        self.sim = sim
        self.ws = ws

    def __getitem__(self, sl):
        return PinnedSlice(self.sim, self.ws)


def stager(sim):
    """The worker's stager thread: the stage ring in order, a job staged once the previous
    tenant of its slot has been copied out of the slot's pinned side (pinned_free)."""
    h = sim.host
    head, tail = int(h.v_stage_head[0]), int(h.v_stage_tail[0])
    if head >= tail:
        return False
    jobs = getattr(h.v_stage_jobs, "arr", h.v_stage_jobs)
    job = jobs[head % HOST.MOE_STAGE_RING]
    seq = int(job[0]) | (int(job[HOST.MOE_JOB_SEQ_HI]) << 32)
    ws, n, prev = int(job[4]), int(job[2]), int(job[6])
    if sim.flags[("pinned_free", ws)] < prev:
        return False
    sim.pinned[ws] = (seq, [int(e) for e in job[7:7 + n]])
    sim.staged.append(seq)
    sim.flags[("stage_done", ws)] = seq
    h.v_stage_head[0] = head + 1
    return True


SPEC = dict(
    num_experts = E, hi = H, ho = H, activation = 3, act_limit = 0,
    proj_dims = {"g": (16, 16, 1), "u": (16, 16, 1), "d": (16, 16, 1)},
    proj_bytes = (32, 32, 32), expert_bytes = EXP_B,
    stream_layout = (((0, 0), (0, 0), (0, 0)), False),
)


def streaming_host(monkeypatch, sim, pinned, layers):
    """A host whose streamed path runs on two cards, its experts' compute replaced by reads that
    check, when the pass's stream gets to them, that the VRAM slot holds the expert streamed."""
    h = make_host(monkeypatch, sim, 6)
    assert h.split_lanes(lanes = 3) == (2, 2, 2)
    h._stage_lock = threading.Lock()
    h.num_wslots = 2
    h.next_wslot = 0
    h.wseq = 0
    h.wslot_prev_seq = [0] * HOST.MOE_MAX_WSLOTS
    h.wslot_size = 2 * EXP_B
    h.batch_experts = 2
    h.batch_recon = False
    h.pinned = pinned
    h.arena_registered = None
    h.layer_blocks = [{e: (0, e * EXP_B) for e in range(E)}] * layers
    h.arena_views = [FakeArena(sim)]
    h.v_stage_jobs = np.zeros((HOST.MOE_STAGE_RING, HOST.MOE_JOB_BYTES // 4), dtype = np.uint32)
    h.v_stage_tail = np.zeros(1, dtype = np.uint32)
    h.v_stage_head = np.zeros(1, dtype = np.uint32)
    h.stage_done_addr = [("stage_done", s) for s in range(HOST.MOE_MAX_WSLOTS)]
    h.pinned_free_addr = [("pinned_free", s) for s in range(HOST.MOE_MAX_WSLOTS)]
    h.wviews = [FakePinned(sim, s) for s in range(HOST.MOE_MAX_WSLOTS)]
    marks = [types.SimpleNamespace(expert = e) for e in range(E)]
    h.aux = {l: {f"{k}_{p}": marks for k in ("suh", "svh") for p in "gud"} for l in range(layers)}

    def dq(x, tv, dims, suh, svh, bias, w_scratch, out_dtype = torch.half):
        vram, bi, e = tv.vram, tv.start // (EXP_B // 2), suh.expert

        def check():
            if vram.blocks.get(bi) != e:
                sim.errors.append(f"cuda:{vram.device} slot {vram.ws} block {bi}: expert "
                                  f"{vram.blocks.get(bi)} where {e} was streamed")
        sim.current().put(("read", check))
        return torch.zeros((x.shape[0], dims[1]), dtype = out_dtype)

    h._dq_linear = dq
    h._act = lambda spec, g, u: torch.zeros_like(u)
    sim.pinned = {}
    sim.staged = []
    sim.stager = stager
    return h


def stream_state(sim, device):
    return dict(
        copy_stream = Stream(sim, f"copy{device}"),
        vram_slots = [FakeVram(device, ws) for ws in range(2)],
        native_slots = None,
        wready_ev = [VEvent(sim) for _ in range(2)],
        wconsumed_ev = [VEvent(sim) for _ in range(2)],
        wslot_used = [False] * 2,
        w_scratch = None, fused_bufs = {}, recon = {},
    )


def streamed_layers(h, sim, lane, device, layers, start = None):
    """What a lane of prompt chunks issues on its card: the streamed path, layer after layer, on
    the card's pass stream; then a host wait for that stream (the half is done)."""
    LANE.Lane(lane, [device]).enter()
    sim.tls.stream = sim.passes[device]
    st = sim.states[device]
    if start is not None:
        assert start.wait(5)
    sel = torch.tensor([[0, 1], [2, 3]] * (ROWS // 2), dtype = torch.long)
    w = torch.ones((ROWS, TOPK), dtype = torch.half)
    y = torch.zeros((ROWS, H), dtype = torch.half)
    flat = sel.reshape(-1)
    for layer in range(layers):
        h._submit_prefill_streamed(layer, y, sel, w, SPEC, list(range(E)), st, [ROWS * TOPK // E] * E,
                                   flat, flat + 1, 0)
    sim.passes[device].synchronize()


def two_lanes_streaming(monkeypatch, pinned, layers = 6, hold = 0.2, patience = 10.0):
    sim = Sim(patience = patience)
    h = streaming_host(monkeypatch, sim, pinned, layers)
    sim.passes = {d: Stream(sim, f"pass{d}") for d in (0, 1)}
    sim.states = {d: stream_state(sim, d) for d in (0, 1)}
    ring = None
    if not pinned:
        # Lane 1's first write of a stage job held up: the interpreter switching threads in the
        # middle of the slot's and the ring's bookkeeping, which are bytecodes apart
        ring = h.v_stage_jobs = _SlowRing(h.v_stage_jobs, hold)
    bg = sim.run()
    failed = []

    def lane(index, device, start = None):
        try:
            streamed_layers(h, sim, index, device, layers, start)
        except BaseException as e:
            failed.append(e)

    threads = [threading.Thread(target = lane, args = (1, 0)),
               threading.Thread(target = lane, args = (2, 1, ring.held if ring is not None else None))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(4 * patience)
    sim.stop = True
    bg.join(5)
    return sim, h, failed, threads


@pytest.mark.parametrize("pinned", [False, True])
def test_two_lanes_of_prompt_chunks_stream_at_once(monkeypatch, pinned):
    # The two halves of a chunk in flight on two cards stream their layers' hot experts at the
    # same time: one rotation of weight-staging slots, one stage ring and one set of pinned
    # slots for both (the VRAM slots, their events and the copy streams are per card). Every
    # expert's compute finds its own weights in its card's VRAM slot, and everything issued runs
    sim, h, failed, threads = two_lanes_streaming(monkeypatch, pinned)
    assert not any(t.is_alive() for t in threads) and not failed, failed
    assert sim.errors == []
    assert all(not s.ops for s in sim.streams)
    batches = 2 * 6 * E // 2
    assert h.wseq == batches
    if pinned:
        assert sim.staged == [] and int(h.v_stage_tail[0]) == 0
    else:
        # One stage job per batch, served in the order of their sequence numbers
        assert sim.staged == list(range(1, batches + 1))
        assert int(h.v_stage_head[0]) == int(h.v_stage_tail[0]) == batches


def test_the_stage_lock_covers_the_slot_the_stage_ring_and_the_copy_that_frees_it():
    # What has to be taken in one go by a lane of prompt chunks, read off the source: the slot
    # and its sequence number, the stage job and the record of the slot's tenant, and the copy
    # stream's wait for the stager, the copy and the flag that frees the slot's pinned side (a
    # stream whose waits on the stage flags came out of the order of their sequence numbers
    # would wait for a tenant staged behind it)
    import ast
    fn = _host_function("_submit_prefill_streamed")
    locks = [n for n in ast.walk(fn) if isinstance(n, ast.With)
             and any(ast.unparse(i.context_expr) == "self._stage_lock" for i in n.items)]
    assert len(locks) == 1
    inside = {id(n) for n in ast.walk(locks[0])}
    stores = [n for n in ast.walk(fn) if isinstance(n, (ast.Assign, ast.AugAssign))
              and any(ast.unparse(t).startswith(("self.next_wslot", "self.wseq", "self.v_stage_tail", "self.wslot_prev_seq"))
                      for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    assert len(stores) == 4 and all(id(n) in inside for n in stores)
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) in (
        "ext.exl3_moe_flag_wait", "ext.exl3_moe_flag_write", "_set_job_seq")]
    assert len(calls) == 3 and all(id(n) in inside for n in calls)
    reads = [n for n in ast.walk(fn) if isinstance(n, ast.Subscript) and ast.unparse(n.value) == "self.v_stage_jobs"]
    assert reads and all(id(n) in inside for n in reads)


def test_the_trim_of_held_sources_survives_the_other_lane(monkeypatch):
    # With two lanes of prompt chunks both keep the sources of their copies in _handoff_held:
    # the oldest entry may be gone by the time a lane trims it
    sim = Sim()
    h = make_host(monkeypatch, sim, 6)
    h.split_lanes(lanes = 3)

    class Gone(dict):
        def __iter__(self):
            return iter([("taken", "by the other lane")])

    h._handoff_held = Gone({i: None for i in range(64)})
    LANE.Lane(2, [1]).enter()
    y, sel, w = rows(CAP)
    h._issue_compute(0, y, sel, w, h.specs[0], torch.zeros((CAP, H)), H)
    assert len(h._handoff_held) == 65
