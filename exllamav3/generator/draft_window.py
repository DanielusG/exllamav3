from __future__ import annotations
from collections import deque

from .draft_confidence import DraftConfidenceCalibrator

"""
The MTP draft window chosen per decode step from a measured cost curve (Generator(draft_window =
"cost"), see Generator._dw_begin and FORK.md).

A step that verifies W drafts for each of its n jobs is expected to give

    n + sum over jobs j, positions k <= W of  a_j1 * a_j2 * ... * a_jk

tokens, a_jk the estimated probability that job j's k-th draft is accepted once the ones before it
were (DraftConfidenceCalibrator over the head's winning logit), and to take T(n, W), the wall time
of such steps as measured. The policy takes the W with the most tokens per second. The head drafts
a position at a time, so the choice is made as it goes: after each position the windows already
drafted are scored with their estimates and the longer ones with a prediction (each job's next
positions as likely as its last one); drafting stops once no longer window is predicted to beat
the best one drafted.

T(n, W) is the median of the last RING step times of that window with that many jobs, separately
for regular and staggered steps (DraftWindowPolicy.curve). A step is timed from the end of the sampling of the
step before it to the end of its own sampling, where the host has just waited for the pass (no
wait of its own), and only when nothing else ran in between (Generator._dw_end). Every window
gets MIN_SAMPLES samples before the policy relies on it (exploration: the window with the fewest
samples is taken), and a window it no longer takes is timed again after REFRESH steps, then after
twice as many each time it is still not taken, up to MAX_BACKOFF times REFRESH: a window past a
cliff of the cost curve (more rows than the decode paths' fast kernels serve) is tried at most
once in REFRESH x MAX_BACKOFF steps once it is known.
"""

# Samples of a window before the policy relies on its median; samples kept per window
MIN_SAMPLES = 4
RING = 32
# Steps after which a window the policy does not take is timed again, the factor that grows
# each time it is still not taken, and the fewest steps between two such refreshes
REFRESH = 512
MAX_BACKOFF = 16
REFRESH_GAP = 32
# Explorations of a window during the warm-up that may give no usable sample (the jobs changed in
# the meantime) before it is left to the refreshes
MAX_ATTEMPTS = 24


def parse_windows(spec) -> list | None:
    """
    The windows the policy may choose from: a list of ints or a comma-separated string ("0,1,3").
    None or an empty string: every window from 0 to the generator's num_draft_tokens.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = [s for s in spec.replace(" ", "").split(",") if s]
        if not spec:
            return None
    out = sorted({int(w) for w in spec})
    if not out:
        return None
    if out[0] < 0:
        raise ValueError(f"draft windows must be 0 or more: {out}")
    return out


def _median(values) -> float:
    s = sorted(values)
    m = len(s)
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


class _Window:
    """Timing of one (mode, jobs, window)"""
    __slots__ = ("samples", "ms", "seen", "attempts", "due", "backoff")

    def __init__(self):
        self.samples = deque(maxlen = RING)
        self.ms = None          # median of samples, ms
        self.seen = 0           # usable samples, the first one (dropped) included
        self.attempts = 0       # warm-up explorations
        self.due = 0            # policy step from which it may be timed again
        self.backoff = 1


class DraftWindowRound:
    """
    One decode step's window, chosen as the head drafts (DraftWindowPolicy.begin). The draft step
    drafts up to draft_limit() positions, hands each position's confidences to add() (it says
    when to stop) and then close() gives the window to verify. target is a window being timed
    (exploration): the step drafts and verifies exactly that.
    """

    def __init__(self, policy, mode, n, allowed, target, capped, times):
        self.policy = policy
        self.mode = mode
        self.n = n
        self.allowed = allowed
        self.target = target
        self.capped = capped
        self.times = times
        self.limit = target if target is not None else allowed[-1]
        self.known = any(t is not None for t in times.values())
        self.gain = [0.0]                   # gain[k]: expected accepted drafts of the batch, window k
        self.reach = [1.0] * n              # each job's probability of reaching the next position
        self.last = [1.0] * n               # each job's last conditional estimate
        self.conf = []                      # host confidences per position (a list per row, or None)
        self.window = None
        self.expected = None
        self.start = 0                      # positions taken from the last step's fold
        self.passes = 0                     # head passes the draft step ran
        self.drafts = None                  # what the draft step returned (the window verified)
        self.probe = None                   # window 0: the first drafts (host), labelled by the step

    def key(self):
        return self.mode, self.n, self.window

    def draft_limit(self) -> int:
        """Positions to draft at most: the window being timed, or the longest one allowed (a
        window of 0 still drafts its first position: the head's cache at the job's position is
        written by that pass, and its confidence decides between 0 and more)"""
        return max(1, self.limit)

    def expected_passes(self) -> int:
        """Head passes a step of this window runs when nothing else is drafted"""
        return max(1, self.window or 0) - self.start

    def add(self, conf: list | None) -> bool:
        """The head's confidences of the next drafted position (None: none exported). True when
        drafting should stop here (the window is then decided)"""
        policy = self.policy
        est = policy.calibrator.estimate_many(conf)
        if est is None:
            est = [policy.calibrator.rate()] * self.n
        self.reach = [r * e for r, e in zip(self.reach, est)]
        self.last = est
        self.gain.append(self.gain[-1] + sum(self.reach))
        self.conf.append(conf)
        k = len(self.conf)
        if self.target is not None:
            if k >= self.draft_limit():
                self._decide(self.target)
                return True
            return False
        w = self._choose(k)
        if w is None:
            return False
        self._decide(w)
        return True

    def close(self) -> int:
        """The window to verify, once the draft step has drafted what it would"""
        if self.window is None:
            k = len(self.conf)
            w = self.target if self.target is not None else self._choose(k, final = True)
            self._decide(w if w is not None else self._longest(k))
        return self.window

    def _longest(self, k):
        fit = [w for w in self.allowed if w <= k]
        return fit[-1] if fit else self.allowed[0]

    def _decide(self, w):
        self.window = w
        self.expected = self.n + self.gain[min(w, len(self.gain) - 1)]

    def _choose(self, k, final = False):
        """After k drafted positions: the window to stop at, or None to draft on"""
        if not self.known:
            # Nothing measured for these jobs yet: the longest window, as without the policy
            return self._longest(k) if final or k >= self.limit else None
        n = self.n
        times = self.times
        best_w, best_r = None, -1.0
        for w in self.allowed:
            if w > k:
                break
            t = times.get(w)
            if t is not None:
                r = (n + self.gain[w]) / t
                if r > best_r:
                    best_w, best_r = w, r
        if not final and k < self.limit:
            # Longer windows, each job's next positions predicted as likely as its last one
            g = self.gain[k]
            inc = list(self.reach)
            pos = k
            for w in self.allowed:
                if w <= k:
                    continue
                if w > self.limit:
                    break
                while pos < w:
                    inc = [a * q for a, q in zip(inc, self.last)]
                    g += sum(inc)
                    pos += 1
                t = times.get(w)
                if t is not None and (n + g) / t > best_r:
                    return None
        return best_w if best_w is not None else self._longest(k)


class DraftWindowPolicy:
    """
    The cost policy's state: the step-time curve, the confidence calibrator, the exploration
    schedule and the counters (stats, a plain dict the generator publishes as draft_window_stats;
    "curve" in it holds each timed window's sample count and median).
    """

    def __init__(self, windows: list | None = None):
        self.windows = windows
        # The target is not used here (estimate() and rate() only); the bins and their decay are
        # the dynamic draft's
        self.calibrator = DraftConfidenceCalibrator(0.5)
        self.curve = {}                     # (mode, n, window) -> _Window
        self.steps = 0
        self._next_refresh = 0
        self._last = None                   # (time, signature) of the last decode step's sampling
        self._switches = None
        self._allowed = {}
        self.stats = {
            "steps": 0, "explored": 0, "capped": 0, "overridden": 0, "timed": 0,
            "chosen": {}, "drafted": 0, "verified": 0, "accepted": 0,
            "expected": 0.0, "realised": 0, "scored": 0, "labels": 0,
            "curve": {}, "calibrator": {"bins": 0, "tested": 0.0, "acceptance": None},
        }

    def windows_for(self, w_max: int) -> list:
        """The windows allowed with num_draft_tokens = w_max (never above it)"""
        allowed = self._allowed.get(w_max)
        if allowed is None:
            src = self.windows if self.windows is not None else range(w_max + 1)
            allowed = [w for w in src if w <= w_max] or [w_max]
            self._allowed[w_max] = allowed
        return allowed

    def reset_curve(self):
        """Forget the step times (the decode step changed: a switch of the MTP decode step)"""
        self.curve = {}
        self.stats["curve"] = {}
        self._last = None

    def begin(self, mode: str, n: int, w_max: int, cap: int | None, timed: bool, switches = None) -> DraftWindowRound:
        """
        A decode step of n jobs is about to draft. cap: the longest window the step may take (a
        prompt chunk runs beside it, see Generator._dw_begin), None for none. timed: whether the
        step can be timed (nothing else runs in it), the condition for exploring.
        """
        if switches != self._switches:
            if self._switches is not None:
                self.reset_curve()
            self._switches = switches
        self.steps += 1
        allowed = self.windows_for(w_max)
        capped = False
        if cap is not None and cap < allowed[-1]:
            allowed = [w for w in allowed if w <= cap] or allowed[:1]
            capped = True
        target = self._explore(mode, n, allowed) if timed and not capped and len(allowed) > 1 else None
        times = {}
        for w in allowed:
            e = self.curve.get((mode, n, w))
            times[w] = e.ms if e is not None else None
        return DraftWindowRound(self, mode, n, allowed, target, capped, times)

    def _window(self, key) -> _Window:
        e = self.curve.get(key)
        if e is None:
            e = self.curve[key] = _Window()
        return e

    def _explore(self, mode, n, allowed):
        """The window to time at this step, or None"""
        # Warm-up: a window with too few samples, the one with the fewest first
        pick = None
        for w in allowed:
            e = self._window((mode, n, w))
            if len(e.samples) < MIN_SAMPLES and e.attempts < MAX_ATTEMPTS:
                if pick is None or len(e.samples) < len(pick[1].samples):
                    pick = (w, e)
        if pick is not None:
            pick[1].attempts += 1
            return pick[0]
        # Refresh: a window not timed for a while (it is not the one the policy takes), the one
        # longest due first; then again after twice as long while it is still not taken
        if self.steps < self._next_refresh:
            return None
        for w in allowed:
            e = self.curve[(mode, n, w)]
            if self.steps >= e.due and (pick is None or e.due < pick[1].due):
                pick = (w, e)
        if pick is None:
            return None
        w, e = pick
        e.due = self.steps + REFRESH * e.backoff
        e.backoff = min(2 * e.backoff, MAX_BACKOFF)
        self._next_refresh = self.steps + REFRESH_GAP
        return w

    def point(self, t: float, signature, key, clean: bool, exploring: bool):
        """
        A decode step's sampling ended at t (seconds). The step's time is from the last such point
        to this one; it goes to key's curve when it is clean, key is not None and nothing changed
        in between (signature: the jobs, the mode, what else would have run). Every decode step
        calls this, so a step that is not timed breaks the chain.
        """
        last = self._last
        self._last = (t, signature)
        if key is None or not clean or last is None or last[1] != signature:
            return
        dt = (t - last[0]) * 1000.0
        if dt <= 0.0:
            return
        e = self._window(key)
        e.seen += 1
        if e.seen == 1:
            return      # the first step at a shape captures its graphs
        e.samples.append(dt)
        e.ms = _median(e.samples)
        if not exploring:
            e.backoff = 1
            e.due = self.steps + REFRESH
        elif len(e.samples) == MIN_SAMPLES:
            e.due = self.steps + REFRESH
        stats = self.stats
        stats["timed"] += 1
        stats["curve"][f"{key[0]}:{key[1]}:{key[2]}"] = {"samples": len(e.samples), "ms": round(e.ms, 3)}

    def label(self, score, accepted: bool):
        """A verified draft position: its confidence and whether the target sampled the draft"""
        if score is None:
            return
        self.calibrator.add_label(score, accepted)
        self.stats["labels"] += 1

    def overridden(self, dw: DraftWindowRound):
        """The step verified another window than the policy's (a hybrid round)"""
        self.stats["overridden"] += 1

    def finish(self, dw: DraftWindowRound, realised: int, accepted: int):
        """The step verified the policy's window: counters and the calibrator's decay"""
        stats = self.stats
        w = dw.window
        stats["steps"] += 1
        stats["chosen"][w] = stats["chosen"].get(w, 0) + 1
        stats["drafted"] += dw.n * len(dw.conf)
        stats["verified"] += dw.n * w
        stats["accepted"] += accepted
        if dw.target is not None:
            stats["explored"] += 1
        if dw.capped:
            stats["capped"] += 1
        if dw.target is None and not dw.capped:
            stats["expected"] += dw.expected
            stats["realised"] += realised
            stats["scored"] += 1
        cal = self.calibrator
        cal.decay_step()
        if stats["steps"] % 32 == 1:
            tested = sum(v[0] for v in cal.bins.values())
            accepted_ = sum(v[1] for v in cal.bins.values())
            stats["calibrator"] = {
                "bins": sum(1 for v in cal.bins.values() if v[0] >= cal.min_count),
                "tested": round(tested, 1),
                "acceptance": round(accepted_ / tested, 4) if tested > 0 else None,
            }
