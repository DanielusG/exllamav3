"""
The MTP draft window chosen per decode step from a measured cost curve (Generator.draft_window =
"cost", generator/draft_window.py).

No GPU or model is required. The policy is checked on its own, with step times fed through its
timing points and acceptance estimates given directly; then the generator's real draft and decode
steps run over the toy model of the staggered tests (tests/test_stagger_decode.py), with a clock
that a pass advances by a synthetic cost of its shape (jobs x window), so the curve the policy
measures is known: a flat curve, a steep one and one with a cliff. The toy's target takes the
drafter's guess one time in three, independently at every position.

When the compiled extension is not installed the source files under test are loaded on their own,
with the rest of the package stubbed.
"""

import sys
import threading

import pytest
import torch

import test_stagger_decode as TOY
import test_mtp_decode as M
import test_prefill_thread as PT
from test_stagger_decode import _host_torch
from test_mtp_decode import _host_torch_job

pytestmark = pytest.mark.nogpu

# (the toy's autouse fixtures: the generator and job modules see host tensors only)
_FIXTURES = (_host_torch, _host_torch_job)

GEN = TOY.GEN
DW = sys.modules[GEN.DraftWindowPolicy.__module__]
CAL = sys.modules[DW.DraftConfidenceCalibrator.__module__]
VOCAB = TOY.VOCAB
LENGTHS = {"a": 300, "b": 260, "c": 340, "d": 280}


@pytest.fixture(autouse = True)
def _wide_layer_states(monkeypatch):
    # The toy's staging rows for windows of up to 3 drafts with 4 jobs (as the cache's max_history)
    monkeypatch.setattr(TOY.ToyLayerState, "ROWS", 4 * 4)


# --------------------------------------------------------------------------------------------
#   The policy on its own
# --------------------------------------------------------------------------------------------

class FakeCalibrator:
    """Confidences that are the acceptance estimates themselves"""

    def estimate_many(self, scores):
        return None if scores is None else list(scores)

    def rate(self):
        return 0.5

    def add_label(self, score, accepted):
        pass

    def decay_step(self):
        pass

    bins = {}
    min_count = 8


def policy_with(curve, windows = None, mode = "r", n = 1, w_max = 3):
    """A policy whose curve for (mode, n) holds the step times curve(w) in ms"""
    p = DW.DraftWindowPolicy(windows)
    p.calibrator = FakeCalibrator()
    t = 0.0
    for w in p.windows_for(w_max):
        for _ in range(DW.MIN_SAMPLES + 2):
            p.point(t, "same", (mode, n, w), True, True)
            t += curve(w) / 1000.0
        p.point(t, "same", (mode, n, w), True, True)
        p._last = None
    return p


def choose(p, accept, n = 1, w_max = 3, cap = None, mode = "r"):
    """The window a step takes with the per-position estimates accept[k] (each job the same), and
    the positions drafted for it"""
    dw = p.begin(mode, n, w_max, cap, timed = False)
    for k in range(dw.draft_limit()):
        if dw.add([accept[k]] * n):
            break
    return dw.close(), len(dw.conf)


def test_the_curve_is_the_median_of_the_clean_steps_after_the_first():
    p = DW.DraftWindowPolicy()
    key = ("r", 2, 1)
    p.point(0.0, "s", key, True, False)
    p.point(1.0, "s", key, True, False)          # the first step at a shape: dropped
    assert key not in p.curve or not p.curve[key].samples
    for t, sig, clean in ((1.030, "s", True), (1.060, "s", True), (1.100, "s", True), (1.110, "t", True),
                          (1.120, "t", False), (1.135, "t", True)):
        p.point(t, sig, key, clean, False)
    # 30, 30, 40 ms, then the signature changed (not timed), a step not clean, 15 ms
    assert list(p.curve[key].samples) == pytest.approx([30.0, 30.0, 40.0, 15.0])
    assert p.curve[key].ms == pytest.approx(30.0)
    assert p.stats["curve"]["r:2:1"] == {"samples": 4, "ms": 30.0}
    # A step without a key breaks the chain
    p.point(1.2, "t", None, True, False)
    p.point(1.25, "t", key, True, False)
    assert len(p.curve[key].samples) == 5


def test_a_flat_curve_takes_the_longest_useful_window():
    p = policy_with(lambda w: 20 + 0.2 * w)
    assert choose(p, [0.8, 0.8, 0.8]) == (3, 3)
    # A third position that is never accepted is not worth even a flat cost
    assert choose(p, [0.9, 0.9, 0.0]) == (2, 3)


def test_a_steep_curve_with_low_acceptance_takes_a_short_window():
    p = policy_with(lambda w: 20 + 10 * w)
    # 1/20 against 1.3/30: no draft verified, and the head stops after the first position
    assert choose(p, [0.3, 0.3, 0.3]) == (0, 1)
    # 1.7/30 beats 1/20 and the 2.19/40 predicted for a second position: one drafted, one verified
    assert choose(p, [0.7, 0.7, 0.7]) == (1, 1)
    # Very likely drafts pay even on a steep curve
    assert choose(p, [0.99, 0.99, 0.99]) == (3, 3)


def test_a_cliff_in_the_curve_is_read_from_the_data():
    # Three jobs: above 8 rows (3 x (1 + W) > 8: W >= 2) the step costs three times as much
    p = policy_with(lambda w: 25 + 2 * w + (60 if 3 * (1 + w) > 8 else 0), n = 3)
    assert choose(p, [0.9, 0.9, 0.9], n = 3)[0] == 1
    # One job has no cliff on that curve
    q = policy_with(lambda w: 25 + 2 * w, n = 1)
    assert choose(q, [0.9, 0.9, 0.9], n = 1)[0] == 3


def test_the_head_stops_drafting_as_soon_as_longer_windows_cannot_pay():
    p = policy_with(lambda w: 20 + 4 * w)
    # A confident first draft and a doubtful second: stop after the second, verify one
    assert choose(p, [0.95, 0.1, 0.9]) == (1, 2)
    # A doubtful first draft: stop at once
    assert choose(p, [0.1, 0.9, 0.9]) == (0, 1)


def test_jobs_of_a_batch_count_together():
    p = policy_with(lambda w: 20 + 3 * w, n = 2)
    dw = p.begin("r", 2, 3, None, timed = False)
    # One job sure of its drafts, the other hopeless: the batch's window is the sum's best
    for k in range(3):
        if dw.add([0.95, 0.0]):
            break
    assert dw.close() == 3
    assert dw.expected == pytest.approx(2 + 0.95 + 0.95 ** 2 + 0.95 ** 3)


def test_without_a_measure_the_window_is_the_longest():
    p = DW.DraftWindowPolicy()
    p.calibrator = FakeCalibrator()
    assert choose(p, [0.1, 0.1, 0.1]) == (3, 3)


def test_a_restricted_set_of_windows():
    p = policy_with(lambda w: 20 + 10 * w, windows = [1, 3])
    assert p.windows_for(3) == [1, 3] and p.windows_for(2) == [1] and p.windows_for(0) == [0]
    assert choose(p, [0.3, 0.3, 0.3])[0] == 1
    assert DW.parse_windows("0, 1,3") == [0, 1, 3] and DW.parse_windows("") is None
    assert DW.parse_windows([3, 1, 1]) == [1, 3] and DW.parse_windows(None) is None
    with pytest.raises(ValueError):
        DW.parse_windows("-1,2")


def test_a_cap_limits_the_windows_and_explores_nothing():
    p = DW.DraftWindowPolicy()
    dw = p.begin("r", 3, 3, 1, timed = True)
    assert dw.allowed == [0, 1] and dw.capped and dw.target is None
    # A cap below every window allowed leaves the narrowest
    q = DW.DraftWindowPolicy([2, 3])
    assert q.begin("r", 4, 3, 0, timed = True).allowed == [2]


def test_a_change_of_the_decode_step_s_switches_forgets_the_curve():
    p = policy_with(lambda w: 20 + w)
    p.begin("r", 1, 3, None, timed = False, switches = (True, False))
    assert p.curve
    p.begin("r", 1, 3, None, timed = False, switches = (True, False))
    assert p.curve
    p.begin("r", 1, 3, None, timed = False, switches = (True, True))
    assert not p.curve and p.stats["curve"] == {}


def simulate(p, cost, steps, n = 3, w_max = 3, accept = 0.9):
    """steps decode steps of n jobs, each timed at cost(n, W) ms; returns the windows explored"""
    t = 0.0
    explored = []
    for _ in range(steps):
        dw = p.begin("r", n, w_max, None, timed = True)
        for k in range(dw.draft_limit()):
            if dw.add([accept] * n):
                break
        w = dw.close()
        if dw.target is not None:
            explored.append((p.steps, w))
        t += cost(n, w) / 1000.0
        p.point(t, "same", dw.key(), True, dw.target is not None)
        p.finish(dw, n, 0)
    return explored


def test_exploration_times_every_window_then_rarely_what_it_does_not_take():
    p = DW.DraftWindowPolicy()
    p.calibrator = FakeCalibrator()
    cliff = lambda n, w: 25 + 2 * w + (60 if n * (1 + w) > 8 else 0)
    explored = simulate(p, cliff, 20000)
    for w in (0, 1, 2, 3):
        assert len(p.curve[("r", 3, w)].samples) >= DW.MIN_SAMPLES
    # The warm-up: every window until it has its samples (the first one at a shape dropped)
    warm = [w for step, w in explored if step <= 100]
    assert len(warm) >= (DW.MIN_SAMPLES + 1) * 4
    # Then the windows past the cliff (2, 3) and the one never worth it (0) are timed again after
    # 512, 1024, 2048, 4096, 8192, 8192... steps each, and the one taken (1) never
    late = [w for step, w in explored if step > 100]
    assert sorted(set(late)) == [0, 2, 3] and all(late.count(w) <= 6 for w in (0, 2, 3)), late
    assert p.stats["chosen"][1] > 19500


def test_estimate_many_is_estimate():
    cal = CAL.DraftConfidenceCalibrator(0.5)
    scores = [-3.5, 0.2, 1.7, 2.2, 4.9, 9.0, 12.5]
    assert cal.estimate_many(scores) == [1.0] * len(scores)
    assert cal.estimate_many(None) is None and cal.rate() == 1.0
    for i in range(400):
        s = (i * 7) % 11
        cal.add_label(s + 0.5, (i * 13) % 17 < s)
    assert cal.estimate_many(scores) == [cal.estimate(s) for s in scores]
    tested = sum(v[0] for v in cal.bins.values())
    assert cal.rate() == pytest.approx(sum(v[1] for v in cal.bins.values()) / tested)


# --------------------------------------------------------------------------------------------
#   The generator over the toy, with a clock the passes move
# --------------------------------------------------------------------------------------------

class Clock:
    """time as the generator module sees it: perf_counter is a clock the passes advance"""

    def __init__(self):
        import time as real
        self._real = real
        self.t = 0.0

    def perf_counter(self):
        return self.t

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(GEN, "time", c)
    return c


def flat(n, w):
    return 20 + 0.1 * w * n


def steep(n, w):
    return 20 + 5 * w * n


def cliff(n, w):
    return 20 + w * n + (60 if n * (1 + w) > 8 else 0)


def timed_head(g, clock, cost):
    """The target's decode passes advance the clock by cost(jobs, window) ms"""
    head = g.model.modules[-1]
    base = type(head)

    class TimedHead(base):
        def forward(self, x, params):
            rows, width = params["input_ids"].shape
            if params.get("cache_seqlens") is not None and width <= 8:
                clock.t += cost(rows, width - 1) / 1000.0
            return super().forward(x, params)

    head.__class__ = TimedHead


def with_conf(g, conf = None):
    """The drafter exports a confidence per drafted row (conf(input id) -> (draft, score), or a
    constant score), as the Qwen MTP heads do through the target's head"""
    draft = g.draft_model
    base = type(draft)

    class Conf(base):
        def sample_from_state(self, state, params):
            ids = super().sample_from_state(state, params)
            if params.get("export_draft_conf"):
                rows = ids.shape[0]
                score = torch.full((rows, 1), 5.0)
                if conf is not None:
                    # (the toy head's state is its input id)
                    src = M.host(state).reshape(rows, -1)[:, 0].long().tolist()
                    out = M.host(ids).clone()
                    for r, tok in enumerate(src):
                        out[r, 0], score[r, 0] = conf(tok, int(out[r, 0]))
                    ids = M.dev(out) if isinstance(ids, M.Dev) else out
                params["draft_conf"] = M.dev(score) if isinstance(ids, M.Dev) else score
            return ids

    draft.__class__ = Conf


def run(names, clock = None, cost = flat, policy = True, verify = True, update = False, stagger = False,
        num_draft_tokens = 3, conf = None, setup = None, **kw):
    """test_mtp_decode.run (or run_fold with update) with draft_window set, the clock and the
    drafter's confidences"""

    def setup_(g):
        with_conf(g, conf)
        if clock is not None:
            timed_head(g, clock, cost)
        if policy:
            g.draft_window = "cost"
        if setup is not None:
            setup(g)

    kw.setdefault("lengths", {n: LENGTHS[n] for n in names})
    kw.setdefault("min_jobs", 3)
    if update:
        return M.run_fold(names, update = True, verify = verify, stagger = stagger, num_draft_tokens = num_draft_tokens,
                          setup = setup_, **kw)
    return M.run(names, verify = verify, stagger = stagger, num_draft_tokens = num_draft_tokens, setup = setup_, **kw)


@pytest.fixture
def rounds(monkeypatch):
    """(jobs, window, explored, capped) of every window the policy chose"""
    seen = []
    real_close = DW.DraftWindowRound.close

    def close(self):
        w = real_close(self)
        seen.append((self.n, w, self.target is not None, self.capped))
        return w

    monkeypatch.setattr(DW.DraftWindowRound, "close", close)
    return seen


@pytest.mark.parametrize("verify,update", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("num_draft_tokens", [1, 3])
@pytest.mark.parametrize("names", ["a", "abc"])
def test_greedy_tokens_are_the_same_with_the_window_chosen(clock, names, num_draft_tokens, verify, update):
    kw = dict(num_draft_tokens = num_draft_tokens, verify = verify, update = update)
    ref, _, _ = run(names, policy = False, **kw)
    out, g, _ = run(names, clock, cost = steep if len(names) > 1 else flat, **kw)
    assert out == ref
    st = g.draft_window_stats
    assert st["steps"] > 100 and st["timed"] > 50
    # Every window was taken (0 included)
    assert set(st["chosen"]) == set(range(num_draft_tokens + 1))
    if update:
        # The head's cache held what it had to at every read, the windows of 0 folded too
        assert g.draft_model.poisoned == 0
        assert g.mtp_update_stats["fold_drafts"] > 50


def test_the_policy_off_is_the_fixed_window(clock):
    out, g, jobs = run("ab", clock, policy = False)
    assert g.draft_window == "fixed" and g.draft_window_policy is None and g.draft_window_stats is None
    # Every step verified the whole window
    assert all(w == 3 for j in jobs.values() for _, w, _ in j.draft_stats)


@pytest.mark.parametrize("names,cost,want", [
    ("a", flat, 3),         # 1 job, flat: the longest window
    ("ab", steep, 0),       # 2 jobs, 10 ms a drafted position: 1 token / 20 ms beats 1.33 / 30
    ("abc", cliff, 1),      # 3 jobs: past 8 rows the step takes 60 ms more
])
def test_the_window_follows_the_measured_curve(clock, rounds, names, cost, want):
    out, g, _ = run(names, clock, cost = cost)
    st = g.draft_window_stats
    n = len(names)
    curve = st["curve"]
    for w in range(4):
        assert curve[f"r:{n}:{w}"]["ms"] == pytest.approx(cost(n, w))
    taken = [w for n_, w, explored, _ in rounds if n_ == n and not explored]
    assert len(taken) > 150 and taken.count(want) > 0.9 * len(taken)
    # Expected tokens per step close to what the steps gave
    assert st["realised"] == pytest.approx(st["expected"], rel = 0.1)
    if names == "abc":
        # The windows past the cliff are taken only to time them: the warm-up, and no refresh
        # in so few steps beyond the first ones
        assert not {2, 3} & set(taken)
        cliffed = [w for n_, w, explored, _ in rounds if n_ == n and explored and w >= 2]
        assert len(cliffed) <= 2 * (DW.MIN_SAMPLES + 1) + 4


@pytest.mark.parametrize("update", [False, True])
def test_staggered_groups_choose_their_own_windows(clock, update):
    ref, _, _ = run("abcd", policy = False, stagger = True, update = update)
    out, g, _ = run("abcd", clock, cost = cliff, stagger = True, update = update)
    assert out == ref
    st = g.draft_window_stats
    assert g.stagger_stats["steps"] > 100
    # Groups of two, timed apart from the regular steps (4 jobs in one batch before the stagger)
    assert all(st["curve"][f"s:2:{w}"]["samples"] >= DW.MIN_SAMPLES for w in range(4))
    assert st["curve"]["s:2:1"]["ms"] == pytest.approx(cliff(2, 1))
    assert st["chosen"].get(1, 0) + st["chosen"].get(2, 0) > 0.9 * st["steps"]
    if update:
        assert g.draft_model.poisoned == 0


def test_the_head_s_confidence_decides_each_step(clock):
    """A drafter that knows when its guess is hopeless (an id the target never samples, scored 0)
    and is otherwise right one time in three (scored 10): on a curve where a drafted position
    pays at 1/3 but not at 0, the policy verifies nothing exactly at the hopeless steps"""
    seen = []
    real_close = DW.DraftWindowRound.close

    def close(self):
        w = real_close(self)
        if self.target is None and self.conf and self.conf[0] is not None:
            seen.append((self.conf[0][0], w))
        return w

    def conf(tok, draft):
        return (VOCAB + 3, 0.0) if tok % 4 == 0 else (draft, 10.0)

    mp = pytest.MonkeyPatch()
    mp.setattr(DW.DraftWindowRound, "close", close)
    try:
        cost = lambda n, w: 20 + 4 * w * n
        ref, _, _ = run("a", policy = False, num_draft_tokens = 1)
        out, g, _ = run("a", clock, cost = cost, num_draft_tokens = 1, conf = conf)
    finally:
        mp.undo()
    assert out == ref
    late = seen[100:]
    assert len(late) > 100
    assert all(w == 0 for s, w in late if s == 0.0)
    assert all(w == 1 for s, w in late if s == 10.0)
    assert {s for s, _ in late} == {0.0, 10.0}


def test_no_readback_of_its_own(clock, monkeypatch):
    """Every readback of the decode and draft steps is counted (the confidences come back with
    the drafts, the labels read host memory): with both switches, one per position drafted and
    one for the sampling, whatever the window"""
    # The toy's device tensors pass for CUDA ones where the step asks (a window of 0 takes the
    # step without drafts, which stages a CUDA token through pinned memory and one synchronize)
    monkeypatch.setattr(M.Dev, "is_cuda", property(lambda self: True), raising = False)
    counted = {}

    def setup(g):
        g.model.devices = True
        real_wait = g._host_wait

        def wait(device):
            M.Readbacks.n += 1
            return real_wait(device)

        g._host_wait = wait
        cuda = GEN.torch.cuda

        class Sync:
            def __getattr__(self, name):
                return getattr(cuda, name)

            def synchronize(self, *a, **k):
                M.Readbacks.n += 1

        GEN.torch.cuda = Sync()

    def script(g, jobs, n, add):
        if all(1 <= j.new_tokens < 250 for j in jobs.values()) and len(jobs) == 3:
            st = g.draft_window_stats
            snap = (M.Readbacks.n, dict(g.readback_stats), st["steps"], st["drafted"])
            counted.setdefault("t0", snap)
            counted["t1"] = snap

    M.Readbacks.n = 0
    out, g, _ = run("abc", clock, cost = cliff, update = True, verify = True, setup = setup, script = script)
    (r0, s0, n0, d0), (r1, s1, n1, d1) = counted["t0"], counted["t1"]
    steps = s1["steps"] - s0["steps"]
    assert steps > 100 and n1 - n0 == steps
    assert r1 - r0 == s1["readbacks"] - s0["readbacks"]
    # One readback per position drafted (the fold's first draft included), one for the sampling
    assert r1 - r0 == (d1 - d0) // 3 + steps


def test_the_window_is_capped_while_a_prompt_chunk_runs_beside(monkeypatch, rounds):
    """With three jobs generating and up to three drafts each, a verification pass is up to 12
    rows: a fixed window keeps every chunk of a prompt from running beside it. The cost window
    counts the decode at its narrowest window (3 rows), so the chunks run beside, and the
    windows of those steps stay within side_max_rows"""
    monkeypatch.setattr(GEN, "Lane", PT.FakeLane)
    plan = {
        "a": dict(prompt = TOY.PROMPTS["a"], max_new = 200),
        "c": dict(prompt = TOY.PROMPTS["c"], max_new = 200),
        "d": dict(prompt = TOY.PROMPTS["d"], max_new = 200),
        "b": dict(prompt = PT.long_prompt(3, 65), max_new = 30, at = 20, chunk = 16, steps = 4),
    }

    def setup_fixed(g):
        g.num_draft_tokens = g.max_draft_tokens = 3
        g.draft_ids_pinned = torch.zeros((4, 3), dtype = torch.long)

    def setup_cost(g):
        setup_fixed(g)
        with_conf(g)
        g.draft_window = "cost"

    ref, g0, _, _ = PT.drive(True, plan, setup = setup_fixed)
    out, g, _, _ = PT.drive(True, plan, setup = setup_cost)
    assert out == ref
    assert g0.side_stats["beside"] == 0
    assert g.side_stats["beside"] > 0
    capped = [(n, w) for n, w, _, c in rounds if c]
    assert capped and all(n * (1 + w) <= g.side_max_rows for n, w in capped)
    assert g.draft_window_stats["capped"] == len(capped)


def test_switched_on_and_off_while_jobs_run(clock):
    def script(g, jobs, n, add):
        g.draft_window = "cost" if (n // 40) % 2 else "fixed"

    ref, _, _ = run("abc", policy = False, update = True)
    out, g, _ = run("abc", clock, cost = steep, policy = False, update = True, script = script)
    assert out == ref
    assert g.draft_model.poisoned == 0
    assert 0 < g.draft_window_stats["steps"] < g.readback_stats["steps"]


def test_the_rows_counted_for_a_chunk_beside_the_decode():
    g = TOY.make_generator(num_draft_tokens = 3)
    g.max_draft_tokens = 3
    jobs = [TOY.ToyJob(g.model, n, TOY.PROMPTS[n], 10) for n in "abc"]
    for j in jobs:
        j.sequences[0].kv_position = len(j.sequences[0].sequence_ids) - 1
    g.active_jobs = jobs
    assert g._decode_rows() == 12
    g.draft_window = "cost"
    assert g._decode_rows() == 3
    g.draft_window_set = [2, 3]
    g.draft_window_policy = None
    assert g._decode_rows() == 9
    # A hybrid draft decides its window after the prefill round: counted at its widest
    g.hybrid_draft_tokens = 5
    g.max_draft_tokens = 5
    assert g._decode_rows() == 18


# --------------------------------------------------------------------------------------------
#   The test hook: a window schedule (EXL3_DRAFT_WINDOW_SCHEDULE, Generator.draft_window_schedule)
# --------------------------------------------------------------------------------------------

def take(p, schedule, accept = 1.0, n = 1, w_max = 3, cap = None, timed = True):
    """One decode step of the policy with a schedule: (window, positions drafted, scheduled)"""
    dw = p.begin("r", n, w_max, cap, timed = timed, schedule = schedule)
    for k in range(dw.draft_limit()):
        if dw.add([accept] * n):
            break
    w = dw.close()
    p.finish(dw, n, 0)
    return w, len(dw.conf), dw.scheduled


def test_parse_schedule_keeps_the_order_and_the_repeats():
    assert DW.parse_schedule("0,1,2,3") == [0, 1, 2, 3]
    assert DW.parse_schedule("3, 0,3,1") == [3, 0, 3, 1] and DW.parse_schedule([2, 2]) == [2, 2]
    assert DW.parse_schedule(None) is None and DW.parse_schedule("") is None and DW.parse_schedule(" , ") is None
    with pytest.raises(ValueError):
        DW.parse_schedule("1,-1")


def test_a_schedule_takes_its_windows_in_order_whatever_the_estimates():
    # On a steep curve with hopeless drafts the policy verifies nothing and stops after the first
    # position; a schedule drafts and verifies exactly its windows (a window of 0 its first position)
    p = policy_with(lambda w: 20 + 10 * w)
    assert choose(p, [0.05, 0.05, 0.05]) == (0, 1)
    got = [take(p, [0, 1, 2, 3], accept = 0.05) for _ in range(9)]
    assert got == [(w, max(1, w), True) for w in (0, 1, 2, 3, 0, 1, 2, 3, 0)]
    assert p.stats["scheduled"] == 9 and p.stats["explored"] == 0
    # Sure drafts on a flat curve, where the policy takes 3: the schedule still decides
    q = policy_with(lambda w: 20 + 0.1 * w)
    assert [take(q, "3,0,3,1", accept = 0.99)[0] for _ in range(6)] == [3, 0, 3, 1, 3, 0]


def test_a_schedule_is_clipped_to_what_the_step_allows():
    # The windows allowed (draft_window_set): the largest one not above the scheduled window, the
    # smallest when none is
    p = DW.DraftWindowPolicy([0, 2])
    p.calibrator = FakeCalibrator()
    assert [take(p, "3,1,0,2")[0] for _ in range(4)] == [2, 0, 0, 2]
    q = DW.DraftWindowPolicy([2, 3])
    q.calibrator = FakeCalibrator()
    assert take(q, [1])[0] == 2
    # num_draft_tokens below the schedule
    r = DW.DraftWindowPolicy()
    r.calibrator = FakeCalibrator()
    assert take(r, [3], w_max = 1)[:2] == (1, 1)
    # The cap of a prompt chunk beside the decode (Generator._dw_begin): clipped, and counted as capped
    dw = r.begin("r", 3, 3, 1, timed = False, schedule = [3])
    assert dw.capped and dw.scheduled and dw.target == 1 and dw.allowed == [0, 1]


def test_the_schedule_starts_again_when_it_changes_and_when_asked():
    p = DW.DraftWindowPolicy()
    p.calibrator = FakeCalibrator()
    sched = [0, 1, 2, 3]
    assert [take(p, sched)[0] for _ in range(3)] == [0, 1, 2]
    p.restart_schedule()
    assert [take(p, sched)[0] for _ in range(2)] == [0, 1]
    # The same schedule as another object: not a change
    assert take(p, [0, 1, 2, 3])[0] == 2
    # Another schedule: from its first entry
    assert [take(p, "3,0")[0] for _ in range(3)] == [3, 0, 3]
    # No schedule: the policy's own path again (no measure yet: it explores, then the longest)
    w, _, scheduled = take(p, None)
    assert not scheduled and p.stats["scheduled"] == 9


def test_without_a_schedule_the_hook_is_never_entered(clock, monkeypatch):
    """Unset, the cost path is the policy's: _scheduled never runs, and the windows taken are the
    ones a run takes with the hook's method gone"""
    seen = {}
    real = DW.DraftWindowRound.close

    def record(key):
        def close(self):
            w = real(self)
            seen.setdefault(key, []).append((self.n, w, self.target, self.capped, len(self.conf)))
            return w
        return close

    def boom(self, schedule, allowed):
        raise AssertionError("the schedule's path without a schedule")

    monkeypatch.setattr(DW.DraftWindowRound, "close", record("plain"))
    ref, g0, _ = run("abc", clock, cost = cliff, update = True)
    assert g0.draft_window_schedule is None and g0.draft_window_stats["scheduled"] == 0
    monkeypatch.setattr(DW.DraftWindowPolicy, "_scheduled", boom)
    monkeypatch.setattr(DW.DraftWindowRound, "close", record("guarded"))
    clock.t = 0.0
    out, g, _ = run("abc", clock, cost = cliff, update = True)
    assert out == ref and seen["guarded"] == seen["plain"]
    assert g.draft_window_stats == g0.draft_window_stats


@pytest.fixture
def scheduled_rounds(monkeypatch):
    """(jobs, window, positions drafted, scheduled, capped) of every round"""
    seen = []
    real_close = DW.DraftWindowRound.close

    def close(self):
        w = real_close(self)
        seen.append((self.n, w, len(self.conf), self.scheduled, self.capped))
        return w

    monkeypatch.setattr(DW.DraftWindowRound, "close", close)
    return seen


@pytest.mark.parametrize("schedule", ["3", "0,1,2,3", "3,0,3,1"])
@pytest.mark.parametrize("verify,update", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("names,stagger", [("a", False), ("abc", False), ("abcd", True)])
def test_greedy_tokens_with_a_window_schedule(clock, scheduled_rounds, schedule, verify, update, names, stagger):
    """Under a schedule the windows vary step by step as the schedule says: the tokens are the
    fixed window's, every step drafts exactly its window, and the recurrent states (the toy's
    replay of the accepted prefix over the pass's staged rows) and the head's cache (with the
    update folded) follow windows of varying length"""
    sched = DW.parse_schedule(schedule)
    ref, _, ref_jobs = run(names, policy = False, verify = verify, update = update, stagger = stagger)
    out, g, jobs = run(names, clock, cost = cliff, verify = verify, update = update, stagger = stagger,
                       setup = lambda g: setattr(g, "draft_window_schedule", sched))
    assert out == ref
    taken = [w for _, w, _, _, _ in scheduled_rounds]
    assert len(taken) > 50 and taken == [sched[i % len(sched)] for i in range(len(taken))]
    assert all(s and not c and d == max(1, w) for _, w, d, s, c in scheduled_rounds)
    st = g.draft_window_stats
    assert st["scheduled"] == st["steps"] == len(taken) and st["explored"] == 0
    # The windows verified are the schedule's (0 verifies none)
    assert {w for j in jobs.values() for _, w, _ in j.draft_stats} == set(sched) - {0}
    if stagger:
        assert g.stagger_stats["steps"] > 20
    if update:
        assert g.draft_model.poisoned == 0
    if verify:
        assert g.mtp_verify_stats["windows"] > 20
    # A schedule of the fixed window is the fixed window, draft for draft
    if schedule == "3":
        assert drafts_of_jobs(jobs) == drafts_of_jobs(ref_jobs)


def drafts_of_jobs(jobs):
    return {n: (j.accepted_draft_tokens, j.rejected_draft_tokens, list(j.draft_stats)) for n, j in jobs.items()}


def test_a_schedule_starts_again_with_each_stretch_of_work():
    """Generator.enqueue on a generator with no job restarts the schedule; with jobs it does not"""
    g = TOY.make_generator(num_draft_tokens = 3)
    g.draft_window = "cost"
    g.draft_window_schedule = [0, 1, 2, 3]
    g.mtp_batched_verify = g.mtp_batched_update = False
    policy = g._dw_policy()
    policy._schedule_at = 5
    a = TOY.ToyJob(g.model, "a", TOY.PROMPTS["a"], 10)
    g.enqueue(a)
    assert policy._schedule_at == 0
    policy._schedule_at = 3
    g.enqueue(TOY.ToyJob(g.model, "b", TOY.PROMPTS["b"], 10))
    assert policy._schedule_at == 3
    # Without the policy nothing is touched
    h = TOY.make_generator(num_draft_tokens = 3)
    h.enqueue(TOY.ToyJob(h.model, "a", TOY.PROMPTS["a"], 10))
    assert h.draft_window_policy is None


def test_the_schedule_beside_a_prompt_chunk_is_capped(monkeypatch, scheduled_rounds):
    """A chunk on the prefill thread beside three jobs: the schedule's 3 is clipped to the cap
    (n x (1 + W) within side_max_rows), the tokens are the fixed window's"""
    monkeypatch.setattr(GEN, "Lane", PT.FakeLane)
    plan = {
        "a": dict(prompt = TOY.PROMPTS["a"], max_new = 200),
        "c": dict(prompt = TOY.PROMPTS["c"], max_new = 200),
        "d": dict(prompt = TOY.PROMPTS["d"], max_new = 200),
        "b": dict(prompt = PT.long_prompt(3, 65), max_new = 30, at = 20, chunk = 16, steps = 4),
    }

    def setup_fixed(g):
        g.num_draft_tokens = g.max_draft_tokens = 3
        g.draft_ids_pinned = torch.zeros((4, 3), dtype = torch.long)

    def setup_sched(g):
        setup_fixed(g)
        with_conf(g)
        g.draft_window = "cost"
        g.draft_window_schedule = [3]

    ref, _, _, _ = PT.drive(True, plan, setup = setup_fixed)
    out, g, _, _ = PT.drive(True, plan, setup = setup_sched)
    assert out == ref
    capped = [(n, w) for n, w, _, _, c in scheduled_rounds if c]
    assert capped and all(n * (1 + w) <= g.side_max_rows and w == g.side_max_rows // n - 1 for n, w in capped)
    assert all(w == 3 for n, w, _, _, c in scheduled_rounds if not c)


def test_the_generator_reads_the_schedule_from_the_environment_only():
    """Generator.__init__ (not run by the toy) reads EXL3_DRAFT_WINDOW_SCHEDULE into
    draft_window_schedule with parse_schedule, and _dw_begin hands it to the policy"""
    src = open(GEN.__file__).read()
    assert 'self.draft_window_schedule = parse_schedule(os.environ.get("EXL3_DRAFT_WINDOW_SCHEDULE"))' in src
    assert "schedule = self.draft_window_schedule," in src
    assert GEN.Generator.draft_window_schedule is None
