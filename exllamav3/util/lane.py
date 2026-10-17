from __future__ import annotations
import os
import threading
from contextlib import contextmanager
import torch

"""
Lanes: forward passes issued by two threads at the same time.

The library issues one pass at a time from one thread, and what its modules share between
passes relies on it: workspaces are looked up by device, shape and tag and are safe because a
kernel's output is consumed before the next kernel on that device's stream reuses the buffer;
the CPU expert worker's slots are handed out in rotation; a row table's staging sets belong to
the last lookup. A second thread issuing a pass beside the first breaks all of that, unless
everything of that kind exists once per thread. A lane is that: a number each such thread
carries (0 for the thread that never asked for one), which the shared structures add to their
keys.

What a lane does NOT have is CUDA streams of its own: both threads issue on the same stream of
each device, so on a card kernels still run one at a time, in the order they were issued. That
is not a choice. The compiled EXL3 kernels (GEMM, MGEMM, the fused MoE kernel) coordinate
their thread blocks through one lock buffer per device, indexed from zero by every launch
(DevCtx::get_locks, exl3_gemm_inner.cuh: barrier_acquire on locks[tile]): two of them running
at once on a card would read each other's turn counters. Until the extension keeps that state
per stream, two streams on one device are wrong sums or a spin that never ends.

What follows from one stream for two threads:

  - The gain is host time and CPU expert worker time, not card time: one thread's kernels run
    while the other thread waits for the worker or for the host. A thread that makes the
    stream itself wait for the worker (the flag memops of model/moe_cpu_host.py) stops the
    other thread's kernels too, so a lane other than 0 waits on the host instead
    (host_waits()).
  - A kernel sequence that goes through a buffer baked into a module at load time (the decode
    graphs and their statics, the fused decode kernels' scratch) is safe only against passes
    that never take the same path. Passes on different lanes must differ in kind: decode-sized
    on lane 0 (at most 8 rows x tokens), prompt chunks of at least a page on the other. Whoever
    runs the lanes keeps to that (Generator._prefill_round).
  - A lane's thread enters inference mode itself: grad mode is per thread.
"""

_tls = threading.local()
_busy_lock = threading.Lock()
_busy = 0

# EXL3_LANE_HOST_WAITS=0: a lane's waits for the CPU expert worker stay on the pass's stream, as
# lane 0's. The other lane's kernels then wait behind them (the feature gains next to nothing),
# but the handoff stream and its host waits are out of the picture: for telling whose fault a
# wrong number is
_HOST_WAITS = os.environ.get("EXL3_LANE_HOST_WAITS", "1") != "0"


def current_lane() -> int:
    """Lane of the calling thread, 0 unless it entered one."""
    return getattr(_tls, "lane", 0)


def host_waits() -> bool:
    """Whether the calling thread must keep its waits for the CPU expert worker off the
    stream (it shares the stream with another lane, whose kernels a stream wait would stop)."""
    return getattr(_tls, "host_waits", False)


def lanes_busy() -> bool:
    """Whether a pass is under way on a lane other than 0. Anything that rewrites what every
    pass reads (an expert placement sweep) has to wait for that to be false."""
    return _busy > 0


def cuda_devices(*models) -> list[int]:
    """Indices of the CUDA devices the modules of these models are on, in order."""
    found = set()
    for model in models:
        if model is None:
            continue
        for module in getattr(model, "modules", ()):
            dev = getattr(module, "device", None)
            if dev is None:
                continue
            dev = torch.device(dev)
            if dev.type == "cuda":
                found.add(dev.index or 0)
    return sorted(found)


def sync_current_streams(devices: list[int]):
    """Wait for what has been issued on the calling thread's stream of each device."""
    for d in devices:
        torch.cuda.current_stream(d).synchronize()


class Lane:

    def __init__(self, index: int, devices: list[int]):
        assert index > 0, "lane 0 is the default lane of every thread"
        self.index = index
        self.devices = list(devices)

    def enter(self):
        """Make the calling thread issue on this lane, for good (a thread serves one lane)."""
        _tls.lane = self.index
        _tls.host_waits = _HOST_WAITS

    @contextmanager
    def passing(self):
        """Around a pass issued on this lane (lanes_busy)."""
        global _busy
        with _busy_lock:
            _busy += 1
        try:
            yield
        finally:
            with _busy_lock:
                _busy -= 1

    def synchronize(self):
        """Wait for everything issued so far on the devices' streams, by any lane: after it,
        what this lane issued has run."""
        sync_current_streams(self.devices)
