"""
The sampling of a speculative window in one call (Generator.mtp_batched_verify:
CustomSampler.forward_window, Job.sample_window, Job._window_masks) with the server's two Qwen3.8
sampling sets:

  thinking      temperature 1.0, top-P 0.95, top-K 20, no penalty     (a fused stack only)
  non-thinking  temperature 0.7, top-P 0.8, top-K 20, presence 1.5    (the penalty in the head, per
                row, on the job's past IDs followed by the drafts before that position)

What the batched verify promises (FORK.md, "An MTP window verified with one readback"): row 0 is
sampled exactly as a lone forward() of that position with the same rand_u32 (the fused sampler keys
its noise on (rand_u32, flat index)); every other row is sampled from exactly its own distribution,
with noise independent of the other rows' (it differs from a lone call by design, so its tokens are
not compared, its distribution is). The checks:

  - row 0 against a one-row forward(), bit for bit, over many seeds (and greedy rows: each one is
    the argmax of its own row);
  - every row's empirical frequencies over N draws against the exact target probabilities,
    computed here in float64 and independently of the sampler (penalty, temperature, top-K, then
    top-P with 1.6.3's convention: the smallest top set whose mass reaches P, the token that
    crosses P included, the mass normalized over the top-K set): no token outside the kept set
    ever, each kept token within Z = 5 standard deviations of its probability (sqrt(p (1 - p) / N)),
    and a chi-square goodness of fit (bins of expected count 5 or more, the rest merged) above
    ALPHA = 1e-6;
  - rows 0 and 1 of the same logits sampled independently: the 4 x 4 contingency table of their
    tokens (the three most likely and the rest) passes a chi-square test of independence at ALPHA
    (noise shared between rows makes the two rows sample the same token every time).

With about 60 per-token tests and 6 chi-square tests per stack the family-wise false alarm is about
1e-4. The distributions are built so that the token crossing P carries 4-6% of the mass with the
cumulative mass before it at least 0.01 below P and after it at least 0.01 above
(test_the_designed_distributions_are_unambiguous): with N = 16384 a sampler that drops that token
(1.6.2's convention) is more than 10 standard deviations off.

Where it runs: on a card with the compiled extension (the real fused sampler and penalty kernel),
the tests marked gpu_*; they skip on a machine without CUDA or without the extension. On this
machine, without either, the same checks run over a stand-in of the extension in float64 (a
categorical sample through Gumbel-max over the kept set, the noise of a generator seeded with
rand_u32 indexed by the flat position): that covers the Python of forward_window (the head per
row, the tail over all rows, the past IDs and masks per row) and the checks themselves, which also
catch three broken stand-ins and wrong past IDs (test_the_checks_catch_broken_samplers). The
kernel's own distribution is covered only on the card.

Filters: LLGuidanceFilter (the server's json_schema / json_object) has a native rollback
(rollback_tokens) and no trigger token, so a job with it takes the window path; its masks per
position (Job._window_masks: the filter fed the drafts and rolled back) are checked against the
serial path's with the real llguidance matcher over a byte-level tokenizer, when llguidance is
installed (it is not part of the fork's requirements).

The GPU part on the workstation (production stopped; under two minutes):

    <env python> -m pytest --noconftest tests/test_window_sampling.py -q
"""

import importlib.util
import math
import os
import sys
import types

import pytest
import torch

import test_mtp_decode as M
from test_mtp_decode import _host_torch_job
from test_stagger_decode import _host_torch

# (the toy's autouse fixtures: the job module sees host tensors only)
_FIXTURES = (_host_torch, _host_torch_job)

REPO = M.REPO
Z = 5.0
ALPHA = 1e-6
VOCAB = 1000            # the tokenizer's real vocabulary
WIDTH = 1024            # the logits' width (padded, as the model's)
TOP_K = 20
ROWS = 4                # a window of 3 drafts

STACKS = {
    "thinking": dict(temperature = 1.0, top_p = 0.95, top_k = TOP_K, pres_p = 0.0),
    "non_thinking": dict(temperature = 0.7, top_p = 0.8, top_k = TOP_K, pres_p = 1.5),
}
# Top-20 weights of each stack's synthetic distribution (the rest of the vocabulary, 980 tokens of
# 0.005-0.015 each, outside the top 20); the non-thinking rows' past IDs and drafts
WEIGHTS = {
    "thinking": [30, 20, 15, 10, 8, 6, 4.0, 3.9, 0.5, 0.4, 0.35, 0.3, 0.27, 0.24, 0.21, 0.19, 0.17, 0.15, 0.13, 0.11],
    "non_thinking": [30, 20, 14, 10, 7, 5, 4, 3, 2.6, 2.1, 1.7, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2],
}
# The 21st token, above the tail: outside top-K, it enters it when a mask bans one of the top 20,
# and the token after it is then the tail's first, clearly below it (the tail's values are close
# together, so a top-K boundary inside it would be a near tie)
NEXT_WEIGHT = {"thinking": 0.08, "non_thinking": 0.15}
PAST = [1, 4]
DRAFTS = [0, 2, 3]
# With a mask per row (a filter's, Job._window_masks: a packed bitmask, one row per position): the
# rank in each row's own logits of the token its mask bans (None: nothing banned), a token of the
# kept set in every banned row
BANNED_RANK = {"thinking": [None, 1, 3, 6], "non_thinking": [None, 5, 6, 7]}


# --------------------------------------------------------------------------------------------
#   The reference and the statistics (float64, nothing of the sampler)
# --------------------------------------------------------------------------------------------

def target_probs(logits_row, temperature, top_k, top_p, pres_p = 0.0, past = (), banned = ()):
    """The exact distribution a row is sampled from: the mask's banned tokens out, presence
    penalty on the past IDs, the temperature, top-K, then top-P over the top-K set (the token
    crossing P kept), renormalized. Over the real vocabulary (the padding never sampled)."""
    l = logits_row[:VOCAB].double().clone()
    for t in banned:
        l[t] = float("-inf")
    if pres_p:
        for t in set(int(x) for x in past):
            l[t] -= pres_p
    l = l / temperature
    order = torch.argsort(l, descending = True, stable = True)
    keep = order[:top_k] if top_k else order
    q = torch.softmax(l[keep], 0)
    before = torch.cumsum(q, 0) - q
    kept = keep[before < top_p]
    out = torch.zeros(VOCAB, dtype = torch.float64)
    out[kept] = torch.softmax(l[kept], 0)
    return out.nan_to_num(0.0)


def _gammq(a, x):
    """The regularized upper incomplete gamma function Q(a, x) (series below a + 1, continued
    fraction above)"""
    if x <= 0.0:
        return 1.0
    lead = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        ap, term, total = a, 1.0 / a, 1.0 / a
        for _ in range(10000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(lead))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            break
    return math.exp(lead) * h


def chi2_sf(x, df):
    return _gammq(df / 2.0, x / 2.0)


def distribution_problems(samples, q, what = ""):
    """What is wrong with N samples of a row against its exact distribution q (empty: nothing)"""
    samples = torch.as_tensor(samples).reshape(-1).long()
    n = samples.numel()
    counts = torch.bincount(samples, minlength = WIDTH).double()
    out = []
    outside = [int(t) for t in torch.nonzero((counts[:VOCAB] > 0) & (q == 0)).view(-1)] + \
        [int(t) for t in torch.nonzero(counts[VOCAB:] > 0).view(-1) + VOCAB]
    if outside:
        out.append(f"{what}sampled outside the kept set: {outside[:8]}")
    support = torch.nonzero(q > 0).view(-1)
    for t in support.tolist():
        p = float(q[t])
        f = float(counts[t]) / n
        sd = math.sqrt(p * (1.0 - p) / n)
        if abs(f - p) > Z * sd + 1e-12:
            out.append(f"{what}token {t}: frequency {f:.4f} against {p:.4f} ({(f - p) / sd:+.1f} sd)")
    expected = q[support] * n
    observed = counts[support]
    big = expected >= 5.0
    e = expected[big].tolist()
    o = observed[big].tolist()
    if (~big).any():
        e.append(float(expected[~big].sum()))
        o.append(float(observed[~big].sum()))
    if len(e) >= 2:
        chi2 = sum((oi - ei) ** 2 / ei for oi, ei in zip(o, e) if ei > 0)
        p_value = chi2_sf(chi2, len(e) - 1)
        if p_value < ALPHA:
            out.append(f"{what}chi-square {chi2:.1f} on {len(e) - 1} df, p = {p_value:.2e}")
    return out


def independence_problems(a, b, cats):
    """Two rows' samples (the same draws) against independence: a contingency table of their
    tokens in the categories `cats` plus the rest"""
    a = torch.as_tensor(a).reshape(-1).tolist()
    b = torch.as_tensor(b).reshape(-1).tolist()
    idx = {t: i for i, t in enumerate(cats)}
    k = len(cats) + 1
    table = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        table[idx.get(x, k - 1)][idx.get(y, k - 1)] += 1
    n = len(a)
    rows = [sum(r) for r in table]
    cols = [sum(table[i][j] for i in range(k)) for j in range(k)]
    chi2 = 0.0
    for i in range(k):
        for j in range(k):
            e = rows[i] * cols[j] / n
            if e > 0:
                chi2 += (table[i][j] - e) ** 2 / e
    df = (sum(1 for r in rows if r) - 1) * (sum(1 for c in cols if c) - 1)
    p_value = chi2_sf(chi2, max(df, 1))
    same = sum(1 for x, y in zip(a, b) if x == y) / n
    if p_value < ALPHA:
        return [f"rows not independent: chi-square {chi2:.1f} on {df} df, p = {p_value:.2e}, same token {same:.3f}"]
    return []


def window_logits(stack, dtype = torch.float):
    """(1, ROWS, WIDTH) logits: the stack's top-20 weights over a tail outside them; rows 0 and 1
    the same (for the independence check). Thinking: rows 2 and 3 the same weights on other token
    IDs (a fixed permutation). Non-thinking: every row the same logits, the rows told apart by
    their past IDs (each position's drafts before it penalized, the drafts being top tokens). The
    padding past VOCAB holds finite values, as a model's does."""
    g = torch.Generator().manual_seed(20261024)
    base = torch.empty(WIDTH, dtype = torch.float64).uniform_(0.005, 0.015, generator = g).log()
    for i, w in enumerate(WEIGHTS[stack] + [NEXT_WEIGHT[stack]]):
        base[i] = math.log(w)
    rows = [base, base.clone()]
    for r in (2, 3):
        row = base.clone()
        if not STACKS[stack]["pres_p"]:
            row[:VOCAB] = base[:VOCAB][torch.randperm(VOCAB, generator = g)]
        rows.append(row)
    return torch.stack(rows).unsqueeze(0).to(dtype)


def past_ids(stack):
    """One set of past IDs per row (the drafts before each position appended), None without a
    penalty"""
    if not STACKS[stack]["pres_p"]:
        return None
    return [torch.tensor([PAST + DRAFTS[:r]], dtype = torch.long) for r in range(ROWS)]


def banned(stack, logits):
    """The token each row's mask bans (a list per row)"""
    out = []
    for r, rank in enumerate(BANNED_RANK[stack]):
        order = torch.argsort(logits[0, r, :VOCAB].double(), descending = True, stable = True)
        out.append([] if rank is None else [int(order[rank])])
    return out


def row_masks(stack, logits, device = "cpu"):
    """The packed bitmask per row (bit clear = banned), as Job._window_masks hands the sampler"""
    bits = torch.full((ROWS, WIDTH // 32), -1, dtype = torch.int32)
    for r, toks in enumerate(banned(stack, logits)):
        for t in toks:
            word = int(bits[r, t >> 5]) & 0xFFFFFFFF
            word &= ~(1 << (t & 31)) & 0xFFFFFFFF
            bits[r, t >> 5] = word - (1 << 32) if word >= 1 << 31 else word
    return bits.to(device)


def targets(stack, logits, masked = False):
    st = STACKS[stack]
    past = past_ids(stack)
    ban = banned(stack, logits) if masked else [[]] * ROWS
    return [target_probs(logits[0, r].double(), st["temperature"], st["top_k"], st["top_p"], st["pres_p"],
                         [] if past is None else past[r].view(-1).tolist(), ban[r]) for r in range(ROWS)]


def test_the_reference_keeps_the_token_that_crosses_p():
    l = torch.tensor([math.log(w) for w in (50, 30, 15, 5)] + [-30.0] * (WIDTH - 4), dtype = torch.float64)
    q = target_probs(l, 1.0, 0, 0.75)
    # 0.5 before the second token, 0.8 before the third: the second crosses 0.75 and is kept
    assert torch.count_nonzero(q) == 2 and q[1] == pytest.approx(30 / 80)
    q = target_probs(l, 1.0, 0, 0.85)
    assert torch.count_nonzero(q) == 3 and q[2] == pytest.approx(15 / 95)
    # top-K first, top-P on the top-K set's own mass
    assert torch.count_nonzero(target_probs(l, 1.0, 2, 0.99)) == 2
    # The presence penalty on past IDs, before the temperature
    q = target_probs(l, 0.5, 0, 1.0, pres_p = 1.0, past = [0])
    assert q[0] / q[1] == pytest.approx(math.exp((math.log(50) - 1.0 - math.log(30)) / 0.5), rel = 1e-9)


def test_the_chi_square_tail():
    assert chi2_sf(3.841, 1) == pytest.approx(0.05, abs = 2e-4)
    assert chi2_sf(16.919, 9) == pytest.approx(0.05, abs = 2e-4)
    assert chi2_sf(9.0, 9) == pytest.approx(0.4373, abs = 2e-4)
    assert chi2_sf(100.0, 9) < 1e-15 and chi2_sf(0.0, 3) == 1.0


@pytest.mark.parametrize("stack", list(STACKS))
@pytest.mark.parametrize("dtype", [torch.float, torch.half])
@pytest.mark.parametrize("masked", [False, True])
def test_the_designed_distributions_are_unambiguous(stack, dtype, masked):
    """The token crossing P carries a few percent of the mass and the cumulative mass before and
    after it is at least 0.01 from P, at every row (with the drafts' penalties, and with the
    masks' bans), in both dtypes: rounding cannot move the kept set, and dropping that token is
    far beyond Z. A banned token is one the row would keep without its mask"""
    st = STACKS[stack]
    logits = window_logits(stack, dtype)
    past = past_ids(stack)
    ban = banned(stack, logits)
    for r in range(ROWS):
        l = logits[0, r, :VOCAB].double().clone()
        if masked:
            for t in ban[r]:
                assert targets(stack, logits)[r][t] > 0.02, "the mask bans a token of the kept set"
                l[t] = float("-inf")
        if st["pres_p"]:
            for t in set(past[r].view(-1).tolist()):
                l[t] -= st["pres_p"]
        l = l / st["temperature"]
        top = torch.sort(l, descending = True).values[:TOP_K + 1]
        assert torch.isfinite(top).all() and torch.all(top[:-1] - top[1:] > 1e-3), "the top tokens are distinct"
        q = torch.softmax(top[:TOP_K], 0)
        before = torch.cumsum(q, 0) - q
        c = int((before < st["top_p"]).sum()) - 1
        assert st["top_p"] - before[c] >= 0.01 and before[c] + q[c] - st["top_p"] >= 0.0099, (r, before[c], q[c])
        assert q[c] >= 0.035


# --------------------------------------------------------------------------------------------
#   A stand-in of the extension (float64), with broken variants for the checks' own test
# --------------------------------------------------------------------------------------------

class StandIn:
    """What custom.py calls of the extension: the fused sampler (greedy, and sampling with top-K /
    top-P over the tempered logits, the crossing token kept), the presence/frequency penalty of one
    row, the bitmask copy. The noise of flat position i is the i-th draw of a generator seeded with
    rand_u32: a row's noise depends on its position in the call, row 0's is a lone call's."""
    FUSED_SAMPLER_MAX_BLOCKS = 4
    FUSED_SAMPLER_HIST_STRIDE = 8
    bug = None          # "shared" noise across rows | "drop_crossing" (1.6.2's top-P) | "row_offset"

    @classmethod
    def fused_sampler(cls, logits, mask, bits, out, workspace, dim, inv_temp, minp_log, rand, mode, filters,
                      top_k, top_p, inv_temp_filter, histogram):
        bsz, width = logits.shape[0], logits.view(logits.shape[0], -1).shape[-1]
        x = logits.view(bsz, -1)[:, :dim].double()
        if mask is not None:
            x = x + mask.view(mask.shape[0], -1)[:, :dim].double()
        if bits is not None:
            b = bits.view(bits.shape[0], -1)
            keep = torch.zeros((b.shape[0], dim), dtype = torch.bool)
            for t in range(dim):
                keep[:, t] = ((b[:, t >> 5].long() >> (t & 31)) & 1).bool()
            x = torch.where(keep, x, torch.full_like(x, float("-inf")))
        if mode == 0:
            out.copy_(torch.argmax(x, dim = -1, keepdim = True))
            return
        g = torch.Generator().manual_seed(int(rand))
        offset = width if (cls.bug == "row_offset" and bsz > 1) else 0
        u = torch.rand(offset + bsz * width, generator = g, dtype = torch.float64)[offset:].view(bsz, width)
        if cls.bug == "shared":
            u = u[:1].expand(bsz, width)
        gumbel = -torch.log(-torch.log(u[:, :dim].clamp(1e-300, 1.0 - 1e-16)))
        for r in range(bsz):
            l = x[r]
            keep = torch.isfinite(l)
            if mode == 3:
                lf = l * inv_temp_filter
                if filters & 1 and top_k:
                    kth = torch.sort(lf[keep], descending = True).values[min(top_k, int(keep.sum())) - 1]
                    keep &= lf >= kth
                if filters & 2 and top_p < 1.0:
                    idx = torch.nonzero(keep).view(-1)
                    q = torch.softmax(lf[idx], 0)
                    order = torch.argsort(q, descending = True)
                    cum = torch.cumsum(q[order], 0)
                    held = (cum <= top_p) if cls.bug == "drop_crossing" else (cum - q[order] < top_p)
                    held[0] = True
                    keep = torch.zeros_like(keep)
                    keep[idx[order[held]]] = True
            score = torch.where(keep, l * inv_temp + gumbel[r], torch.full_like(l, float("-inf")))
            out[r, 0] = torch.argmax(score)

    apply_pres_freq_pens = M.Ext.apply_pres_freq_pens
    apply_logit_bitmask = M.Ext.apply_logit_bitmask


@pytest.fixture
def stand_in(monkeypatch):
    """The real custom.py and presets.py (loaded by test_mtp_decode with the package stubbed) over
    the stand-in"""
    monkeypatch.setattr(M.CUSTOM, "ext", StandIn)
    monkeypatch.setattr(StandIn, "bug", None)
    return M.CUSTOM, M.PRESETS


def tokenizer():
    return types.SimpleNamespace(actual_vocab_size = VOCAB)


def sampler_of(presets, stack):
    st = STACKS[stack]
    return presets.ComboSampler(temperature = st["temperature"], top_p = st["top_p"], top_k = st["top_k"],
                                pres_p = st["pres_p"])


def lone(sampler, logits, past, seed, r = 0, mask = None):
    """A one-row forward() of position r, as the serial path samples it (with its mask row)"""
    return int(sampler.forward(logits[:, r:r + 1, :].clone(), None if past is None else past[r], seed,
                               tokenizer(), logit_mask = None if mask is None else mask[r:r + 1]).view(-1)[0])


def draws(sampler, logits, past, n, device = "cpu", mask = None):
    """n windows sampled at once with seeds 0 .. n-1: (n, ROWS) tokens, read back at the end"""
    out = torch.empty((n, ROWS), dtype = torch.long, device = device)
    use_past = past if sampler.window_past_ids else None
    for seed in range(n):
        out[seed] = sampler.forward_window(logits.clone(), use_past, seed, tokenizer(), logit_mask = mask).view(-1)
    return out.cpu()


def check_rows(sampler, logits, past, n, device = "cpu", masked = False):
    """Every row's distribution, rows 0 and 1 independent: the problems found"""
    mask = row_masks(sampler.stack, logits.cpu(), device) if masked else None
    got = draws(sampler, logits, past, n, device, mask)
    tg = targets(sampler.stack, logits.cpu(), masked)
    problems = []
    for r in range(ROWS):
        problems += distribution_problems(got[:, r], tg[r], f"row {r}: ")
    cats = torch.argsort(tg[0], descending = True)[:3].tolist()
    problems += independence_problems(got[:, 0], got[:, 1], cats)
    return problems


def make_sampler(presets, stack):
    s = sampler_of(presets, stack)
    s.stack = stack
    return s


# --------------------------------------------------------------------------------------------
#   On this machine: forward_window's Python over the stand-in, and the checks' power
# --------------------------------------------------------------------------------------------

@pytest.mark.nogpu
@pytest.mark.parametrize("stack", list(STACKS))
@pytest.mark.parametrize("masked", [False, True])
def test_row_0_is_a_lone_forward_over_the_stand_in(stand_in, stack, masked):
    _, presets = stand_in
    s = make_sampler(presets, stack)
    assert s.window_sampling and (s.window_past_ids == (stack == "non_thinking"))
    logits = window_logits(stack)
    past = past_ids(stack)
    use_past = past if s.window_past_ids else None
    mask = row_masks(stack, logits) if masked else None
    for seed in range(64):
        got = s.forward_window(logits.clone(), use_past, seed, tokenizer(), logit_mask = mask).view(-1).tolist()
        assert got[0] == lone(s, logits, past, seed, mask = mask), seed


@pytest.mark.nogpu
@pytest.mark.parametrize("stack", list(STACKS))
@pytest.mark.parametrize("masked", [False, True])
def test_every_row_follows_its_distribution_over_the_stand_in(stand_in, stack, masked):
    _, presets = stand_in
    s = make_sampler(presets, stack)
    assert check_rows(s, window_logits(stack), past_ids(stack), 4096, masked = masked) == []


@pytest.mark.nogpu
def test_greedy_rows_are_each_row_s_argmax_over_the_stand_in(stand_in):
    _, presets = stand_in
    s = presets.GreedySampler()
    logits = window_logits("thinking") + torch.randn((1, ROWS, WIDTH), generator = torch.Generator().manual_seed(3))
    got = s.forward_window(logits.clone(), None, 5, tokenizer()).view(-1).tolist()
    assert got == [int(torch.argmax(logits[0, r, :VOCAB])) for r in range(ROWS)]


@pytest.mark.nogpu
@pytest.mark.parametrize("bug,check", [("shared", "independence"), ("drop_crossing", "distribution"),
                                       ("row_offset", "row0")])
def test_the_checks_catch_broken_samplers(stand_in, monkeypatch, bug, check):
    """Noise shared by the rows, 1.6.2's top-P (the crossing token dropped), row 0 given another
    row's noise: each is caught by the check meant for it, at the numbers of draws used here"""
    _, presets = stand_in
    monkeypatch.setattr(StandIn, "bug", bug)
    s = make_sampler(presets, "thinking")
    logits = window_logits("thinking")
    if check == "row0":
        assert any(s.forward_window(logits.clone(), None, seed, tokenizer()).view(-1)[0] != lone(s, logits, None, seed)
                   for seed in range(64))
        return
    problems = check_rows(s, logits, None, 2048)
    kind = "not independent" if check == "independence" else "frequency"
    assert any(kind in p for p in problems), problems


@pytest.mark.nogpu
def test_the_checks_catch_a_mask_row_given_to_the_wrong_position(stand_in):
    """Every row sampled with row 0's mask (the masks of the positions after it lost): the rows
    whose mask bans a kept token are off"""
    _, presets = stand_in
    s = make_sampler(presets, "thinking")
    logits = window_logits("thinking")
    mask = row_masks("thinking", logits)
    got = draws(s, logits, None, 2048, mask = mask[:1].expand(ROWS, -1).contiguous())
    tg = targets("thinking", logits, masked = True)
    assert distribution_problems(got[:, 0], tg[0]) == []
    assert all(distribution_problems(got[:, r], tg[r]) for r in range(1, ROWS))


@pytest.mark.nogpu
def test_the_checks_catch_the_drafts_missing_from_the_past_ids(stand_in):
    """Rows 1-3 penalized as row 0 (the drafts before each position left out of its past IDs):
    their distributions are off"""
    _, presets = stand_in
    s = make_sampler(presets, "non_thinking")
    logits = window_logits("non_thinking")
    past = past_ids("non_thinking")
    wrong = [past[0]] * ROWS
    got = draws(s, logits, wrong, 2048)
    tg = targets("non_thinking", logits)
    assert distribution_problems(got[:, 0], tg[0]) == []
    assert all(distribution_problems(got[:, r], tg[r]) for r in range(1, ROWS))


# --------------------------------------------------------------------------------------------
#   On the card: the compiled fused sampler and penalty kernel
# --------------------------------------------------------------------------------------------

def _real_sampler():
    """The library's own custom.py and presets.py over the compiled extension, or None"""
    if not torch.cuda.is_available():
        return None
    try:
        sys.path.insert(0, REPO)
        from exllamav3.generator.sampler import presets, custom
        import exllamav3_ext       # noqa: F401
    except Exception:
        return None
    finally:
        if REPO in sys.path:
            sys.path.remove(REPO)
    if not os.path.abspath(custom.__file__).startswith(REPO + os.sep):
        return None
    return custom, presets


@pytest.fixture(scope = "module")
def real():
    got = _real_sampler()
    if got is None:
        pytest.skip("needs CUDA and the compiled extension (the fork's sources ahead of the wheel)")
    return got


DEVICE = "cuda"


@pytest.mark.parametrize("stack", list(STACKS))
@pytest.mark.parametrize("dtype", [torch.float, torch.half])
@pytest.mark.parametrize("masked", [False, True])
def test_gpu_row_0_is_a_lone_forward_bit_for_bit(real, stack, dtype, masked):
    custom, presets = real
    s = make_sampler(presets, stack)
    assert s.window_sampling and s.steps[-1].__class__.__name__ == "SS_Fused"
    logits = window_logits(stack, dtype).to(DEVICE)
    past = past_ids(stack)
    past = None if past is None else [p.to(DEVICE) for p in past]
    use_past = past if s.window_past_ids else None
    mask = row_masks(stack, logits.cpu(), DEVICE) if masked else None
    bad = []
    for seed in range(512):
        got = int(s.forward_window(logits.clone(), use_past, seed, tokenizer(), logit_mask = mask).view(-1)[0])
        want = lone(s, logits, past, seed, mask = mask)
        if got != want:
            bad.append((seed, got, want))
    assert not bad, bad[:8]


def test_gpu_greedy_rows_are_each_row_s_argmax(real):
    custom, presets = real
    s = presets.GreedySampler()
    logits = (window_logits("thinking") + torch.randn((1, ROWS, WIDTH), generator = torch.Generator().manual_seed(3))).to(DEVICE)
    for seed in range(8):
        got = s.forward_window(logits.clone(), None, seed, tokenizer()).view(-1).tolist()
        assert got == [int(torch.argmax(logits[0, r, :VOCAB])) for r in range(ROWS)]
        assert got[0] == lone(s, logits, None, seed)


@pytest.mark.parametrize("stack", list(STACKS))
@pytest.mark.parametrize("masked", [False, True])
def test_gpu_every_row_follows_its_distribution_and_rows_are_independent(real, stack, masked):
    custom, presets = real
    s = make_sampler(presets, stack)
    logits = window_logits(stack).to(DEVICE)
    past = past_ids(stack)
    past = None if past is None else [p.to(DEVICE) for p in past]
    assert check_rows(s, logits, past, 16384, device = DEVICE, masked = masked) == []


# --------------------------------------------------------------------------------------------
#   LLGuidanceFilter (json_schema) on the window path: masks against the serial path's
# --------------------------------------------------------------------------------------------

LLG_VOCAB = 257         # 256 bytes and an end token
LLG_WIDTH = 288         # padded to a multiple of 32


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class ByteTokenizer:
    """What LLGuidanceFilter reads of ExLlamaV3's Tokenizer: the HF tokenizer (byte-level, one token
    per byte, <eos> = 256) and the end token. Hashable, as create_ll_tokenizer's cache wants."""

    def __init__(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        b2u = _bytes_to_unicode()
        tok = Tokenizer(models.BPE(vocab = {b2u[b]: b for b in range(256)}, merges = []))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
        tok.decoder = decoders.ByteLevel()
        tok.add_special_tokens(["<eos>"])
        self.tokenizer = tok
        self.config = types.SimpleNamespace(eos_token_id_list = [256])
        self.eos_token_id = 256


def _load_llguidance_filter():
    """The library's llguidance.py next to the real Filter base class test_mtp_decode loaded (the
    package stubbed), or None without llguidance"""
    try:
        import llguidance      # noqa: F401
        import tokenizers      # noqa: F401
    except ImportError:
        return None
    full = M.FILTER.__name__.rsplit(".", 1)[0] + ".llguidance"
    spec = importlib.util.spec_from_file_location(full, os.path.join(REPO, "exllamav3", "generator", "filter", "llguidance.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[full] = m
    spec.loader.exec_module(m)
    return m


LLG = _load_llguidance_filter()
SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}


@pytest.fixture(scope = "module")
def byte_tok():
    if LLG is None:
        pytest.skip("llguidance (and tokenizers) not installed: LLGuidanceFilter's window masks not covered here")
    return ByteTokenizer()


def llg_job(tok, consumed = "", **kw):
    """A job of test_mtp_decode with a real LLGuidanceFilter over the byte-level tokenizer, the
    grammar already fed `consumed` (as the tokens sampled before)"""
    f = LLG.LLGuidanceFilter(tokenizer = tok, json_schema = SCHEMA, **kw)
    # (the generator computes its masks in the filter pool; the same masks, here in line)
    f.use_background_worker = lambda: False
    for ch in consumed:
        f.feed(ord(ch))
    job = M.make_job([f])
    job.generator.padded_vocab_size = LLG_WIDTH
    job.generator.tokenizer = types.SimpleNamespace(actual_vocab_size = LLG_VOCAB)
    return job, f


@pytest.mark.nogpu
def test_a_job_with_an_llguidance_filter_takes_the_window_path(byte_tok):
    job, f = llg_job(byte_tok)
    assert type(f).rollback_tokens is not M.FILTER.Filter.rollback_tokens and f.trigger_token is None
    assert job.window_sampling_why() is None
    job, _ = llg_job(byte_tok, trigger_token = ord("{"))
    assert job.window_sampling_why() == "filter"


@pytest.mark.nogpu
@pytest.mark.parametrize("consumed,drafts", [("{", '"a"'), ('{"a"', ":12"), ('{"a":', "7 }")])
def test_llguidance_window_masks_are_the_serial_masks(byte_tok, consumed, drafts):
    job, f = llg_job(byte_tok, consumed)
    ids = [ord(c) for c in drafts]
    ref = M.serial_masks(job, ids)
    before = f.get_next_logit_mask().clone()
    journal = list(f._journal)
    masks = M.window(job, ids)
    assert masks.dtype == torch.int32 and masks.shape == (len(ids) + 1, LLG_WIDTH // 32)
    for r in range(len(ids) + 1):
        if ref[r] is None:
            # The filter completed on the draft before: nothing masked, in either path
            assert masks[r].eq(-1).all(), f"position {r}"
        else:
            assert M.same_mask(ref[r], M.row(masks, r)), f"position {r}"
    # Fed and rolled back: the matcher is where it was
    assert torch.equal(f.get_next_logit_mask(), before) and f._journal == journal


@pytest.mark.nogpu
def test_llguidance_a_refused_draft_ends_the_masks(byte_tok):
    job, f = llg_job(byte_tok, "{")
    before = f.get_next_logit_mask().clone()
    journal = list(f._journal)
    masks = M.window(job, [ord("x"), ord('"'), ord("a")])
    # 'x' is not allowed after '{': nothing is fed, every row repeats the first (never read)
    assert all(torch.equal(masks[r], masks[0]) for r in range(4))
    assert torch.equal(f.get_next_logit_mask(), before) and f._journal == journal


@pytest.mark.nogpu
@pytest.mark.parametrize("eos_after", [False, True])
def test_llguidance_a_draft_that_completes_the_grammar(byte_tok, eos_after):
    job, f = llg_job(byte_tok, '{"a":1', eos_after_completed = eos_after)
    masks = M.window(job, [ord("}"), ord("z")])
    if eos_after:
        # Completing ends the job if accepted: the row after repeats the last, never read
        assert torch.equal(masks[2], masks[1])
    else:
        # The filter is done: nothing masked past it, as the serial path
        assert masks[2].eq(-1).all()
    assert f.is_active and not f.is_completed()
