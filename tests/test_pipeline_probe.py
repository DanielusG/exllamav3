"""
The order the pipeline probe runs a prompt's chunks in (experiments/pipeline_probe.py:
pipeline_order, plan_spans). No GPU or model: the stages are functions that log and sleep.
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import pipeline_probe as PP      # noqa: E402


def test_spans_are_whole_chunks_on_page_boundaries():
    assert PP.plan_spans(10000, 4096) == [(0, 4096), (4096, 8192)]
    assert PP.plan_spans(8192, 4096) == [(0, 4096), (4096, 8192)]
    assert PP.plan_spans(5000, 2100) == [(0, 2048), (2048, 4096)]
    assert PP.plan_spans(1000, 4096) == []


def run(n, a_s = 0.0, b_s = 0.0, fail = None):
    log, lock = [], threading.Lock()

    def note(what, k):
        with lock:
            log.append((what, k, threading.current_thread().name))

    def stage_a(k):
        note("a+", k)
        if fail == ("a", k):
            raise ValueError(f"a{k}")
        time.sleep(a_s)
        note("a-", k)
        return k * 10

    def stage_b(k, state):
        assert state == k * 10
        note("b+", k)
        if fail == ("b", k):
            raise ValueError(f"b{k}")
        time.sleep(b_s)
        note("b-", k)

    with ThreadPoolExecutor(max_workers = 1, thread_name_prefix = "side") as pool:
        t0 = time.perf_counter()
        times = PP.pipeline_order(n, stage_a, stage_b, pool.submit)
        return log, times, time.perf_counter() - t0


def test_each_stage_follows_the_same_stage_of_the_chunk_before():
    log, times, _ = run(7, 0.01, 0.01)
    at = {(what, k): i for i, (what, k, _) in enumerate(log)}
    for k in range(7):
        assert at[("a-", k)] < at[("b+", k)]                    # a chunk's own halves in order
        if k:
            assert at[("a-", k - 1)] < at[("a+", k)]            # first card: one chunk at a time, in order
            assert at[("b-", k - 1)] < at[("b+", k)]            # second card too
    # even chunks on the calling thread, odd ones on the other
    assert {k % 2 for what, k, name in log if name.startswith("side")} == {1}
    assert {k % 2 for what, k, name in log if not name.startswith("side")} == {0}
    assert len(times) == 7 and all(t is not None and len(t) == 3 for t in times)


def test_the_two_stages_of_neighbouring_chunks_overlap():
    n, a_s, b_s = 8, 0.05, 0.05
    log, _, dt = run(n, a_s, b_s)
    at = {(what, k): i for i, (what, k, _) in enumerate(log)}
    # the next chunk starts on the first card before this one has left the second
    assert sum(1 for k in range(n - 1) if at[("a+", k + 1)] < at[("b-", k)]) >= n - 2
    assert dt < 0.8 * n * (a_s + b_s)                          # in line it would be n * (a + b)


@pytest.mark.parametrize("fail", [("a", 0), ("a", 3), ("b", 2), ("b", 5)])
def test_a_failed_stage_stops_both_threads_and_is_raised(fail):
    with pytest.raises(ValueError, match = f"{fail[0]}{fail[1]}"):
        run(8, 0.0, 0.0, fail = fail)
