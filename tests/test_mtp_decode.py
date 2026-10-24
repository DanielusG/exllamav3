"""
The MTP decode step with one readback (Generator.mtp_batched_verify): every position of a job's
speculative window sampled in one call, the samples compared with the drafts on the device, the
tokens and the matched counts of every job read back at once.

No GPU or model is required. The generator's real draft and decode steps run over the toy model of
the staggered tests (tests/test_stagger_decode.py), with stub jobs that sample greedily and can
sample a window at once; the real Job's window sampling (masks fed ahead and rolled back, past IDs
per position) runs over a stub generator and fake filters; the real sampler's forward_window over a
stand-in for the compiled extension, against its own forward() one row at a time.

Readbacks are counted on the toy: the logits, hidden states and drafts the toy model produces are
tensors of a subclass that stands for device memory, and every host read of one (item, tolist,
cpu, a blocking copy into host memory) counts, as does every explicit wait of the generator
(Generator._host_wait) and every torch.cuda.synchronize.

When the compiled extension is not installed the source files under test are loaded on their own,
with the rest of the package stubbed.
"""

import importlib.util
import math
import os
import random
import sys
import types

import pytest
import torch

import test_stagger_decode as TOY
import test_tail_checkpoint as TAIL
from test_stagger_decode import _host_torch

pytestmark = pytest.mark.nogpu

# (the toy's autouse fixture: the generator module sees host tensors only)
_FIXTURES = (_host_torch,)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = TOY.GEN
JOB = TAIL.JOB
VOCAB = TOY.VOCAB


@pytest.fixture(autouse = True)
def _host_torch_job(monkeypatch):
    # Job's pinned buffers as plain host tensors (no CUDA here)
    monkeypatch.setattr(JOB, "torch", TOY._Torch())


# --------------------------------------------------------------------------------------------
#   Device tensors and readbacks
# --------------------------------------------------------------------------------------------

class Readbacks:
    n = 0


class Dev(torch.Tensor):
    """A tensor the toy model has on a card: reading it on the host is a readback."""

    READS = {torch.Tensor.item, torch.Tensor.tolist, torch.Tensor.cpu, torch.Tensor.numpy,
             torch.Tensor.__bool__, torch.Tensor.__int__, torch.Tensor.__float__, torch.Tensor.__index__}

    @classmethod
    def __torch_function__(cls, func, types_, args = (), kwargs = None):
        kwargs = kwargs or {}
        host_result = False
        if func in cls.READS and args and isinstance(args[0], Dev):
            Readbacks.n += 1
            host_result = True
        elif func is torch.Tensor.copy_ and len(args) > 1 and not isinstance(args[0], Dev) and isinstance(args[1], Dev):
            # Into host memory: a readback unless non-blocking (a later wait reads it)
            if not (kwargs.get("non_blocking") or (len(args) > 2 and args[2])):
                Readbacks.n += 1
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        elif func is torch.Tensor.to and args and isinstance(args[0], Dev):
            dev = args[1] if len(args) > 1 else kwargs.get("device")
            if dev is not None and torch.device(dev).type == "cpu" and not kwargs.get("non_blocking"):
                Readbacks.n += 1
                host_result = True
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
        if host_result:
            # Host memory from here on (cpu() of a host tensor is the tensor itself)
            return ret.as_subclass(torch.Tensor) if isinstance(ret, torch.Tensor) else ret
        return torch._tensor._convert(ret, cls)


def dev(t):
    return t.as_subclass(Dev)


class DevToyHead(TOY.ToyHead):
    def forward(self, x, params):
        logits = super().forward(x, params)
        params["export_states"] = [dev(s) for s in params["export_states"]]
        return dev(logits)


class DevToyDraft(TOY.ToyDraft):
    def forward(self, batch_ids, params):
        return dev(super().forward(batch_ids, params))


def dev_model(g):
    """The toy model's head and drafter on a card."""
    model = g.model
    head = DevToyHead(model, model.modules[-1].idx, model.modules[-1].device)
    model.modules[-1] = head
    model.fwd_modules = [(m, 0, i) for i, m in enumerate(model.modules)]
    if g.draft_model is not None:
        g.draft_model = DevToyDraft()


# --------------------------------------------------------------------------------------------
#   Toy jobs that sample a window at once
# --------------------------------------------------------------------------------------------

class WinJob(TOY.ToyJob):
    """ToyJob with Job's window interface: greedy, so a window sampled at once gives what the
    serial path gives position by position. `why` makes the job decline (a fallback)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.why = None
        self.windows = 0

    def window_sampling_why(self):
        return self.why

    def sample_window(self, logits, drafts):
        assert len(drafts) == logits.shape[1] - 1
        assert self.mid_pass, f"{self.name}: window sampled outside its pass"
        self.mid_pass = False
        self.windows += 1
        return torch.argmax(logits, dim = -1).view(1, -1)

    def receive_sample(self, logits, next_token, *rest):
        # As Job.receive_sample: the token comes to the host first, the rest reads it there
        return super().receive_sample(logits, next_token.cpu(), *rest)


def run(names, verify, stagger = False, num_draft_tokens = 1, lengths = None, script = None, setup = None,
        job_cls = WinJob, min_jobs = 2, devices = False, **job_kw):
    """As test_stagger_decode.run, with the jobs of job_cls and mtp_batched_verify set"""

    def setup_(g):
        g.mtp_batched_verify = verify
        g.mtp_verify_stats = {"windows": 0, "serial": 0, "readbacks": 0, "fallbacks": {}}
        g.readback_stats = {"steps": 0, "readbacks": 0}
        if devices:
            dev_model(g)
        if setup is not None:
            setup(g)

    real = TOY.ToyJob
    TOY.ToyJob = job_cls
    try:
        return TOY.run(names, stagger = stagger, num_draft_tokens = num_draft_tokens, lengths = lengths,
                       script = script, setup = setup_, min_jobs = min_jobs, **job_kw)
    finally:
        TOY.ToyJob = real


LENGTHS = {"a": 70, "b": 55, "c": 90, "d": 40}


def drafts_of(jobs):
    return {n: (j.accepted_draft_tokens, j.rejected_draft_tokens) for n, j in jobs.items()}


# --------------------------------------------------------------------------------------------
#   The generator over the toy
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("num_draft_tokens", [1, 2, 3])
@pytest.mark.parametrize("names", ["a", "ab", "abcd"])
def test_windows_sampled_at_once_give_the_serial_tokens(names, num_draft_tokens):
    ref, _, ref_jobs = run(names, verify = False, num_draft_tokens = num_draft_tokens, lengths = LENGTHS)
    out, g, jobs = run(names, verify = True, num_draft_tokens = num_draft_tokens, lengths = LENGTHS)
    assert out == ref
    # Greedy: the same drafts are accepted and rejected, job by job
    assert drafts_of(jobs) == drafts_of(ref_jobs)
    acc = sum(a for a, _ in drafts_of(jobs).values())
    rej = sum(r for _, r in drafts_of(jobs).values())
    assert acc > 5 and rej > 5
    stats = g.mtp_verify_stats
    assert stats["windows"] > 20 and stats["serial"] == 0 and stats["fallbacks"] == {}
    assert stats["readbacks"] < stats["windows"] or len(names) == 1
    assert all(j.windows > 10 for j in jobs.values())


def test_off_the_generator_never_samples_a_window():
    _, g, jobs = run("ab", verify = False, lengths = LENGTHS)
    assert all(j.windows == 0 for j in jobs.values())
    assert g.mtp_verify_stats["windows"] == 0


@pytest.mark.parametrize("names", ["abc", "abcd"])
def test_staggered_groups_sample_their_windows_at_once(names):
    ref, _, ref_jobs = run(names, verify = False, stagger = True, lengths = LENGTHS)
    out, g, jobs = run(names, verify = True, stagger = True, lengths = LENGTHS)
    plain, _, _ = run(names, verify = False, lengths = LENGTHS)
    assert out == ref == plain
    assert drafts_of(jobs) == drafts_of(ref_jobs)
    assert g.stagger_stats["steps"] > 20 and g.stagger_stats["spec_holds"] > 20
    assert g.mtp_verify_stats["windows"] > 20


def test_checkpoint_boundaries_cut_a_window_sampled_at_once():
    """A recurrent checkpoint boundary ends the window on the host, as in the serial loop: the
    positions sampled past it are never used."""
    def script(g, jobs, n, add):
        for j in jobs.values():
            j.ckpt_every = 5

    ref, _, ref_jobs = run("abcd", verify = False, num_draft_tokens = 2, lengths = LENGTHS, script = script)
    out, g, jobs = run("abcd", verify = True, num_draft_tokens = 2, lengths = LENGTHS, script = script)
    assert out == ref
    assert drafts_of(jobs) == drafts_of(ref_jobs)
    assert {n: j.checkpoints for n, j in jobs.items()} == {n: j.checkpoints for n, j in ref_jobs.items()}
    assert sum(len(j.checkpoints) for j in jobs.values()) > 20


def test_requeues_inside_a_window_sampled_at_once():
    ref, _, ref_jobs = run("abc", verify = False, num_draft_tokens = 2, lengths = LENGTHS, rq_at = 9)
    out, g, jobs = run("abc", verify = True, num_draft_tokens = 2, lengths = LENGTHS, rq_at = 9)
    assert out == ref
    assert drafts_of(jobs) == drafts_of(ref_jobs)
    assert all(j.requeues > 2 for j in jobs.values())


def test_a_job_that_declines_takes_the_serial_path_beside_the_others():
    def script(g, jobs, n, add):
        jobs["b"].why = "probs"

    ref, _, ref_jobs = run("abc", verify = False, lengths = LENGTHS)
    out, g, jobs = run("abc", verify = True, lengths = LENGTHS, script = script)
    assert out == ref
    assert drafts_of(jobs) == drafts_of(ref_jobs)
    assert jobs["b"].windows == 0 and jobs["a"].windows > 10 and jobs["c"].windows > 10
    assert g.mtp_verify_stats["fallbacks"]["probs"] > 10


def test_a_failed_window_falls_back_to_the_serial_path():
    class Failing(WinJob):
        def sample_window(self, logits, drafts):
            if self.name == "a" and self.new_tokens % 2:
                return None
            return super().sample_window(logits, drafts)

    ref, _, _ = run("ab", verify = False, lengths = LENGTHS)
    out, g, jobs = run("ab", verify = True, lengths = LENGTHS, job_cls = Failing)
    assert out == ref
    assert g.mtp_verify_stats["fallbacks"]["filter_error"] > 5


def test_jobs_without_the_window_interface_take_the_serial_path():
    ref, _, _ = run("ab", verify = False, lengths = LENGTHS, job_cls = TOY.ToyJob)
    out, g, _ = run("ab", verify = True, lengths = LENGTHS, job_cls = TOY.ToyJob)
    assert out == ref
    assert g.mtp_verify_stats["windows"] == 0 and g.mtp_verify_stats["fallbacks"]["job"] > 10


def readbacks_per_step(names, verify, num_draft_tokens = 1, stagger = False, **kw):
    """Readbacks of the decode steps once every job has its first token: those the toy counts
    (Dev reads, explicit waits, synchronize calls), and the generator's own count, per step"""
    counted = {"steps": 0, "reads": 0}
    seen = {}

    def setup(g):
        real_wait = g._host_wait

        def wait(device):
            Readbacks.n += 1
            return real_wait(device)

        g._host_wait = wait
        cuda = GEN.torch.cuda

        class Sync:
            def __getattr__(self, name):
                return getattr(cuda, name)

            def synchronize(self, *a, **k):
                Readbacks.n += 1

        GEN.torch.cuda = Sync()

    def script(g, jobs, n, add):
        # Count once every job has its first token, until the first is near its end (all of them
        # still generating, the groups of a staggered decode as they are in the middle)
        if len(jobs) == len(names) and all(1 <= j.new_tokens < 150 for j in jobs.values()):
            if "t0" not in seen:
                seen["t0"] = (Readbacks.n, dict(g.readback_stats))
            seen["t1"] = (Readbacks.n, dict(g.readback_stats))

    Readbacks.n = 0
    lengths = {n: 200 for n in names}
    run(names, verify = verify, num_draft_tokens = num_draft_tokens, stagger = stagger, lengths = lengths,
        script = script, setup = setup, devices = True, **kw)
    (r0, s0), (r1, s1) = seen["t0"], seen["t1"]
    steps = s1["steps"] - s0["steps"]
    return (r1 - r0) / steps, (s1["readbacks"] - s0["readbacks"]) / steps


@pytest.mark.parametrize("num_draft_tokens", [1, 2])
@pytest.mark.parametrize("names", ["a", "abc"])
def test_readbacks_per_step(names, num_draft_tokens):
    """Serial: each draft step reads its drafts back twice (into the drafts and into the next
    step's inputs), and every position sampled is a readback. With the windows sampled at once,
    one readback for the step's sampling."""
    n = len(names)
    off, off_own = readbacks_per_step(names, verify = False, num_draft_tokens = num_draft_tokens)
    on, on_own = readbacks_per_step(names, verify = True, num_draft_tokens = num_draft_tokens)
    draft = 2 * num_draft_tokens
    # Serial: per job at least one position (the first), at most the whole window
    assert draft + n <= off <= draft + n * (num_draft_tokens + 1)
    assert on == pytest.approx(draft + 1)
    # The generator's own count agrees with the toy's
    assert off_own == pytest.approx(off) and on_own == pytest.approx(on)


def test_readbacks_per_step_staggered():
    off, _ = readbacks_per_step("abcd", verify = False, stagger = True, min_jobs = 3)
    on, _ = readbacks_per_step("abcd", verify = True, stagger = True, min_jobs = 3)
    # A group of two: draft 2, then 2-4 positions serially, 1 at once
    assert 4 <= off <= 6
    assert on == pytest.approx(3)


# --------------------------------------------------------------------------------------------
#   The real Job's window sampling
# --------------------------------------------------------------------------------------------

V = 96          # vocabulary of the stub generator (a multiple of 32)


class FakeFilter:
    """A filter with native rollback over a toy grammar: after token t only t + 1 .. t + 4 (mod V)
    are allowed, packed; it completes on token 0. Feeds, masks and rollbacks are logged."""

    def __init__(self, dense = False, eos_after_completed = False, fail_on = None):
        self.trigger_token = None
        self.is_active = True
        self.eos_after_completed = eos_after_completed
        self.dense = dense
        self.fail_on = fail_on
        self.state = [7]
        self._journal = []
        self.log = []
        self._bits = torch.zeros((1, V // 32), dtype = torch.int32)
        self._dense = torch.zeros((1, V), dtype = torch.half)

    # Filter's own feed / rewind, as Filter does them (journal, native rollback)
    def feed(self, token):
        if not self.is_active:
            self._journal.append(("pass", token))
            return False
        if token == self.fail_on:
            raise RuntimeError("matcher error")
        self.state.append(token)
        self.log.append(("feed", token))
        if token == 0:
            self.is_active = False
            self._journal.append(("complete", token))
            return self.eos_after_completed
        self._journal.append(("accept", token))
        return False

    def rewind(self, n):
        if n == 0:
            return
        popped = self._journal[-n:]
        del self._journal[-n:]
        k = sum(1 for e, _ in popped if e in ("accept", "complete"))
        if any(e == "complete" for e, _ in popped):
            self.is_active = True
        self.rollback_tokens(k)
        self.log.append(("rewind", n))

    def rollback_tokens(self, n):
        if n:
            del self.state[-n:]
        return True

    def allowed(self):
        t = self.state[-1]
        return [(t + k) % V for k in range(1, 5)]

    def get_next_logit_mask(self):
        # The buffer is reused, as LLGuidanceFilter's
        ok = self.allowed()
        if self.dense:
            self._dense.fill_(float("-inf"))
            self._dense[0, ok] = 0.0
            return self._dense
        self._bits.zero_()
        for t in ok:
            self._bits[0, t >> 5] |= (1 << (t & 31)) if (t & 31) < 31 else -(1 << 31)
        return self._bits

    def use_background_worker(self):
        return False


def _load_filter():
    """The real Filter base class (rewind by journal replay unless rollback_tokens is overridden)"""
    pkg = "exl3_window_filter_stub"
    for name, attrs in (("", {}), ("tokenizer", {"Tokenizer": object}), ("generator", {}), ("generator.filter", {})):
        m = types.ModuleType(f"{pkg}.{name}" if name else pkg)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[m.__name__] = m
    full = f"{pkg}.generator.filter.filter"
    spec = importlib.util.spec_from_file_location(full, os.path.join(REPO, "exllamav3", "generator", "filter", "filter.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[full] = m
    spec.loader.exec_module(m)
    return m


FILTER = _load_filter()


def make_job(filters = (), sampler = None, new_tokens = 5, prompt = (1, 2, 3, 4, 5, 6, 7)):
    job = JOB.Job.__new__(JOB.Job)
    job.generator = types.SimpleNamespace(padded_vocab_size = V, max_draft_tokens = 3,
                                          tokenizer = types.SimpleNamespace(actual_vocab_size = V))
    ids = TAIL.Ids(list(prompt))
    job.sequences = [types.SimpleNamespace(sequence_ids = ids)]
    job.new_tokens = new_tokens
    job.min_new_tokens = 0
    job.forced_ids = None
    job.return_probs = False
    job.return_top_tokens = 0
    job.banned_strings_utf32_offsets = None
    job.checkpoint = None
    job.prefix_token = None
    job.filters = list(filters)
    job.filters_suspended = False
    job.filter_futures = []
    job.logit_masks = []
    job.pinned_logit_mask = None
    job.pinned_logit_bitmask = None
    job.host_logit_mask = None
    job.pinned_window_masks = {}
    job.logits_device = torch.device("cpu")
    job.pinned_ids = None
    job.pinned_ids_valid = 0
    job.current_pinned_ids = None
    job.sampler = sampler or types.SimpleNamespace(window_sampling = True, window_past_ids = False, reqs_past_ids = False)
    job.rng = random.Random(1234)
    job.max_new_tokens = 100
    return job


def serial_masks(job, drafts):
    """The mask the serial path builds for each position, the drafts accepted one by one (as the
    generator does: prepare_logit_mask, then feed through receive_sample, then the next mask)"""
    out = []
    fed = 0
    for r in range(len(drafts) + 1):
        for f in job.filters:
            if f.is_active:
                job.logit_masks.append(f.get_next_logit_mask())
                job.filter_futures.append(None)
        job.prepare_logit_mask()
        out.append(None if job.host_logit_mask is None else job.host_logit_mask.clone())
        if r < len(drafts):
            for f in job.filters:
                f.feed(drafts[r])
            fed += 1
    for f in job.filters:
        f.rewind(fed)
    return out


def window(job, drafts):
    """prepare_logit_mask for the first position, then the window's masks"""
    for f in job.filters:
        if f.is_active:
            job.logit_masks.append(f.get_next_logit_mask())
            job.filter_futures.append(None)
    job.prepare_logit_mask()
    return job._window_masks(drafts, len(drafts) + 1)


def row(mask, r):
    return mask[r:r + 1] if mask is not None else None


def same_mask(a, b):
    if a is None or b is None:
        return a is None and b is None
    if a.dtype == torch.half:
        return torch.equal(torch.isfinite(a), torch.isfinite(b[:, :a.shape[-1]]))
    return torch.equal(a, b[:, :a.shape[-1]])


@pytest.mark.parametrize("dense", [False, True])
def test_window_masks_are_the_serial_masks(dense):
    f = FakeFilter(dense = dense)
    job = make_job([f])
    # 8 then 9 and 10 are each allowed after the one before (7 -> 8 .. 11)
    drafts = [8, 9, 10]
    ref = serial_masks(job, drafts)
    state = list(f.state)
    masks = window(job, drafts)
    assert masks.dtype == (torch.half if dense else torch.int32) and masks.shape[0] == 4
    for r in range(4):
        assert same_mask(ref[r], row(masks, r)), f"position {r}"
    # Fed and rolled back: the filter is where it was, its journal too
    assert f.state == state and f._journal == []
    assert [e for e, _ in f.log if e == "feed"] == ["feed"] * 6


def test_a_draft_the_mask_refuses_ends_the_masks_there():
    f = FakeFilter()
    job = make_job([f])
    # 30 is not allowed after 7: nothing past it is fed; the rows after repeat the last
    masks = window(job, [8, 30, 31])
    assert [t for e, t in f.log if e == "feed"] == [8]
    assert torch.equal(masks[2], masks[1]) and torch.equal(masks[3], masks[1])
    assert f.state == [7]


def test_a_filter_that_completes_unconstrains_the_rows_after_it():
    f = FakeFilter()
    f.state = [95]      # 96 % 96 = 0 is allowed next and completes the filter
    job = make_job([f])
    masks = window(job, [0, 50])
    assert masks[2].eq(-1).all()            # nothing masked once the filter is done
    assert f.is_active and f.state == [95]  # rolled back, active again


def test_a_filter_that_ends_the_job_stops_the_rows():
    f = FakeFilter(eos_after_completed = True)
    f.state = [95]
    job = make_job([f])
    masks = window(job, [0, 50])
    assert torch.equal(masks[2], masks[1])
    assert f.is_active and f.state == [95]


def test_a_filter_that_raises_is_put_back_and_the_job_goes_serial():
    good = FakeFilter()
    bad = FakeFilter(fail_on = 9)
    job = make_job([good, bad])
    for f in job.filters:
        job.logit_masks.append(f.get_next_logit_mask())
        job.filter_futures.append(None)
    job.prepare_logit_mask()
    draws = job.rng.getstate()
    assert job.sample_window(torch.zeros((1, 4, V)), [8, 9, 10]) is None
    assert good.state == [7] and bad.state == [7]
    assert good._journal == [] and bad._journal == []
    # Nothing was drawn
    assert job.rng.getstate() == draws


def test_window_past_ids_are_the_sequence_and_the_drafts_before():
    job = make_job()
    job.sampler = types.SimpleNamespace(reqs_past_ids = True)
    job.prepare_sampling_past_ids()
    past = job._window_past_ids([40, 41, 42])
    assert [p.tolist() for p in past] == [
        [[1, 2, 3, 4, 5, 6, 7]],
        [[1, 2, 3, 4, 5, 6, 7, 40]],
        [[1, 2, 3, 4, 5, 6, 7, 40, 41]],
        [[1, 2, 3, 4, 5, 6, 7, 40, 41, 42]],
    ]
    # Beyond the watermark: the next staging writes the real tokens over the drafts
    assert job.pinned_ids_valid == 7
    job.sequences[0].sequence_ids.append(99)
    job.prepare_sampling_past_ids()
    assert job.current_device_ids.tolist() == [[1, 2, 3, 4, 5, 6, 7, 99]]


def test_why_a_window_is_not_sampled_at_once():
    assert make_job().window_sampling_why() is None
    cases = {
        "first_token": dict(new_tokens = 0),
        "sampler": dict(sampler = types.SimpleNamespace(window_sampling = False)),
    }
    for why, kw in cases.items():
        assert make_job(**kw).window_sampling_why() == why
    for attr, value, why in (
        ("forced_ids", torch.zeros((1, 1)), "forced_tokens"),
        ("return_probs", True, "probs"),
        ("return_top_tokens", 3, "probs"),
        ("banned_strings_utf32_offsets", object(), "banned_strings"),
        ("checkpoint", {"offset": 1}, "banned_strings"),
        ("min_new_tokens", 10, "min_new_tokens"),
    ):
        job = make_job()
        setattr(job, attr, value)
        assert job.window_sampling_why() == why, attr
    job = make_job(new_tokens = -1)
    assert job.window_sampling_why() == "first_token"
    # Filters: a trigger token, or no native rollback
    f = FakeFilter()
    f.trigger_token = 5
    assert make_job([f]).window_sampling_why() == "filter"

    # A filter that rolls back by replaying its journal (the base class's rollback_tokens)
    class Plain(FILTER.Filter):
        def __init__(self):
            self.trigger_token = None
            self.is_active = True

    real = JOB.Filter
    JOB.Filter = FILTER.Filter
    try:
        assert make_job([Plain()]).window_sampling_why() == "filter"
        p = Plain()
        p.is_active = False     # completed, no trigger: passes everything
        assert make_job([p]).window_sampling_why() is None
        assert make_job([FakeFilter()]).window_sampling_why() is None
    finally:
        JOB.Filter = real
    # Suspended filters (forced tokens once) are not fed
    f = FakeFilter()
    f.trigger_token = 5
    job = make_job([f])
    job.filters_suspended = True
    assert job.window_sampling_why() is None


def test_sample_window_hands_the_sampler_one_draw_masks_and_past_ids():
    calls = []

    class Sampler:
        window_sampling = True
        window_past_ids = True
        reqs_past_ids = True

        def forward_window(self, logits, past_ids, rand_u32, tokenizer, logit_mask = None):
            calls.append((logits.shape, [p.shape[-1] for p in past_ids], rand_u32, logit_mask))
            return torch.zeros((1, logits.shape[1]), dtype = torch.long)

    f = FakeFilter()
    job = make_job([f], sampler = Sampler())
    job.prepare_sampling_past_ids()
    for flt in job.filters:
        job.logit_masks.append(flt.get_next_logit_mask())
        job.filter_futures.append(None)
    job.prepare_logit_mask()
    expect = random.Random(1234).randint(0, (1 << 32) - 1)
    out = job.sample_window(torch.zeros((1, 3, V)), [8, 9])
    assert out.shape == (1, 3)
    (shape, past, seed, mask), = calls
    assert shape == (1, 3, V) and past == [7, 8, 9] and seed == expect
    assert mask.shape == (3, V // 32)


# --------------------------------------------------------------------------------------------
#   The real sampler's forward_window, over a stand-in for the extension
# --------------------------------------------------------------------------------------------

def _noise(seed, n, offset = 0):
    """Deterministic stand-in for the Philox noise keyed on (seed, flat index)"""
    idx = torch.arange(offset, offset + n, dtype = torch.float64)
    return torch.frac(torch.sin(idx * 12.9898 + (seed % 1000) * 78.233) * 43758.5453)


class Ext:
    """What custom.py calls, in torch: the fused sampler (greedy and plain sampling, with the
    mask), presence/frequency penalties for one row, the bitmask copy."""
    FUSED_SAMPLER_MAX_BLOCKS = 4
    FUSED_SAMPLER_HIST_STRIDE = 8

    @staticmethod
    def fused_sampler(logits, mask, bits, out, workspace, dim, inv_temp, minp_log, rand, mode, filters,
                      top_k, top_p, inv_temp_filter, histogram):
        bsz = logits.shape[0]
        x = logits.view(bsz, -1)[:, :dim].double()
        if mask is not None:
            m = mask.view(mask.shape[0], -1)[:, :dim].double()
            x = x + m
        if bits is not None:
            b = bits.view(bits.shape[0], -1)
            keep = torch.zeros((b.shape[0], dim), dtype = torch.bool)
            for t in range(dim):
                keep[:, t] = ((b[:, t >> 5].long() >> (t & 31)) & 1).bool()
            x = torch.where(keep, x, torch.full_like(x, float("-inf")))
        if mode == 0:
            out.copy_(torch.argmax(x, dim = -1, keepdim = True))
            return
        g = -torch.log(-torch.log(_noise(rand, bsz * dim).view(bsz, dim).clamp(1e-9, 1 - 1e-9)))
        out.copy_(torch.argmax(x * inv_temp + g, dim = -1, keepdim = True))

    @staticmethod
    def apply_pres_freq_pens(in_logits, out_logits, past_ids, pres_p, freq_p, sustain, decay):
        assert in_logits.shape[0] <= 1 and past_ids.shape[0] == 1
        v = in_logits.float().clone()
        seen = torch.zeros(v.shape[-1])
        counts = torch.zeros(v.shape[-1])
        for t in past_ids.view(-1).tolist():
            seen[t] = 1.0
            counts[t] += 1.0
        out_logits.copy_(v - seen * pres_p - counts * freq_p)

    @staticmethod
    def apply_logit_bitmask(x, out, bits):
        dim = x.shape[-1]
        keep = torch.zeros(x.shape, dtype = torch.bool)
        b = bits.view(bits.shape[0], -1)
        for t in range(dim):
            keep[:, t] = ((b[:, t >> 5].long() >> (t & 31)) & 1).bool()
        out.copy_(torch.where(keep, x, torch.full_like(x, float("-inf"))))


def _load_sampler():
    pkg = "exl3_window_sampler_stub"
    src = os.path.join(REPO, "exllamav3")

    def stub(name, **attrs):
        full = f"{pkg}.{name}" if name else pkg
        m = types.ModuleType(full)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[full] = m
        return m

    def real(name):
        full = f"{pkg}.{name}"
        path = os.path.join(src, *name.split(".")) + ".py"
        spec = importlib.util.spec_from_file_location(full, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        return m

    stub("")
    stub("tokenizer", Tokenizer = object)
    stub("ext", exllamav3_ext = Ext)
    stub("util", next_power_of_2 = lambda x: 1 << (x - 1).bit_length(), profile_opt = lambda *a, **k: None)
    stub("util.tensor", buffered_arange = lambda n, device: torch.arange(n, device = device))
    stub("generator")
    stub("generator.sampler")
    real("generator.sampler.sampler")
    custom = real("generator.sampler.custom")
    presets = real("generator.sampler.presets")
    return custom, presets


try:
    sys.path.insert(0, REPO)
    from exllamav3.generator.sampler import custom as _custom, presets as _presets
    raise ImportError("the real extension is there: these tests run on the stand-in only")
except Exception:
    CUSTOM, PRESETS = _load_sampler()
finally:
    sys.path.remove(REPO)


def _tok():
    return types.SimpleNamespace(actual_vocab_size = 90)


def _per_row(sampler, logits, past, seed, mask):
    out = []
    for r in range(logits.shape[1]):
        m = None if mask is None else mask[r:r + 1]
        out.append(int(sampler.forward(logits[:, r:r + 1, :].clone(), None if past is None else past[r],
                                       seed, _tok(), logit_mask = m).view(-1)[0]))
    return out


@pytest.mark.parametrize("pres_p", [0.0, 1.5])
@pytest.mark.parametrize("masked", [None, "dense", "bits"])
def test_forward_window_greedy_is_each_row_alone(pres_p, masked):
    torch.manual_seed(0)
    sampler = PRESETS.ComboSampler(temperature = 0.0, pres_p = pres_p)
    logits = torch.randn((1, 4, V)).half()
    past = [torch.tensor([[3, 5, 7] + [11, 13, 17][:r]]) for r in range(4)]
    mask = None
    if masked == "dense":
        mask = torch.zeros((4, V), dtype = torch.half)
        mask[:, 0:20] = float("-inf")
    elif masked == "bits":
        mask = torch.full((4, V // 32), -1, dtype = torch.int32)
        mask[:, 0] = 0
    if pres_p:
        assert sampler.window_past_ids and sampler.window_head is not None
    use_past = past if sampler.window_past_ids else None
    got = sampler.forward_window(logits.clone(), use_past, 99, _tok(), logit_mask = mask).view(-1).tolist()
    assert got == _per_row(sampler, logits, use_past, 99, mask)


@pytest.mark.parametrize("width", [V + 32, V - 32])
def test_forward_window_takes_a_dense_mask_of_another_width(width):
    """A mask wider than the logits is cropped, a narrower one padded with -inf: what a single
    row's mask of that width gives"""
    torch.manual_seed(2)
    sampler = PRESETS.ComboSampler(temperature = 0.0)
    logits = torch.randn((1, 3, V)).half()
    mask = torch.zeros((3, width), dtype = torch.half)
    mask[:, :10] = float("-inf")
    got = sampler.forward_window(logits.clone(), None, 3, _tok(), logit_mask = mask).view(-1).tolist()
    assert got == _per_row(sampler, logits, None, 3, mask)


@pytest.mark.parametrize("pres_p", [0.0, 1.5])
def test_forward_window_sampling_first_row_is_the_serial_draw(pres_p):
    torch.manual_seed(1)
    sampler = PRESETS.ComboSampler(temperature = 0.8, pres_p = pres_p)
    assert sampler.window_sampling
    logits = torch.randn((1, 3, V)).half()
    past = [torch.tensor([[3, 5, 7] + [11, 13][:r]]) for r in range(3)]
    use_past = past if sampler.window_past_ids else None
    for seed in (5, 6, 7):
        got = sampler.forward_window(logits.clone(), use_past, seed, _tok()).view(-1).tolist()
        assert got[0] == _per_row(sampler, logits[:, :1], None if use_past is None else use_past[:1], seed, None)[0]
        # Reproducible
        assert got == sampler.forward_window(logits.clone(), use_past, seed, _tok()).view(-1).tolist()


def test_which_stacks_sample_a_window_at_once():
    C, P = CUSTOM, PRESETS
    # The server's sets (Qwen3.8 thinking / non-thinking) and greedy
    assert P.ComboSampler(temperature = 0.7, top_k = 20, top_p = 0.8, pres_p = 1.5).window_sampling
    assert P.ComboSampler(temperature = 1.0, top_k = 20, top_p = 0.95).window_sampling
    assert P.GreedySampler().window_sampling
    # Pres. 0 is a no-op: no past IDs needed, although reqs_past_ids is set by the step
    s = P.ComboSampler(temperature = 1.0, top_k = 20, top_p = 0.95, pres_p = 0.0)
    assert s.reqs_past_ids and not s.window_past_ids and s.fused_only
    s = P.ComboSampler(temperature = 0.7, top_k = 20, top_p = 0.8, pres_p = 1.5)
    assert s.window_past_ids and [type(x) for x in s.window_head[0]] == [C.SS_PresFreqP]
    # Adaptive-P carries state from one sample to the next
    assert not P.ComboSampler(temperature = 1.0, adaptive_target = 0.5).window_sampling
    assert not C.CustomSampler([C.SS_Sample_mn()]).window_sampling
    # Penalties outside a fused head (fused sampler off): per-row past IDs would need every step one row at a time
    old = C.fused_sampler_enable
    C.fused_sampler_enable = False
    try:
        assert not P.ComboSampler(temperature = 0.7, pres_p = 1.5).window_sampling
        assert P.ComboSampler(temperature = 0.7, top_k = 20).window_sampling
    finally:
        C.fused_sampler_enable = old


# --------------------------------------------------------------------------------------------
#   The MTP head's update in one pass (Generator.mtp_batched_update) over a paged head cache
# --------------------------------------------------------------------------------------------

POISON = 10 ** 6
POISON_TOKEN = VOCAB + 7    # a draft no toy target ever samples


def host(t):
    return t.as_subclass(torch.Tensor) if isinstance(t, Dev) else t


class CacheHead(TOY.ToyHead):
    """The toy's head, recording the hidden state the target exports at each position of the pages
    it writes (as the target's K/V cache holds what the last pass at a position wrote)"""

    def forward(self, x, params):
        logits = super().forward(x, params)
        table = params["block_table"]
        seqlens = params["cache_seqlens"]
        rows, width = params["input_ids"].shape
        for r in range(rows):
            for i in range(width):
                p = int(seqlens[r]) + i
                self.model.hidden_at[(int(table[r, p // TOY.PAGE]), p % TOY.PAGE)] = float(x[r, i])
        if self.model.devices:
            params["export_states"] = [dev(s) for s in params["export_states"]]
            return dev(logits)
        return logits


class CacheDraft(TOY.ToyDraft):
    """
    MTP-like head with a paged cache of its own, in the target's page layout. A pass writes, for
    each of its positions p, the token at p and the target hidden state it is given (that of p - 1)
    into the slot of p, then computes each position's output from the slots before it: when every
    slot it reads (the 16 positions before) holds the job's token at that position and the hidden
    state the target exported at the position before, the toy drafts the token after its
    input, as ToyDraft; else the draft is POISON_TOKEN. A slot written by a draft step past the first
    (draft_step >= 1, the head's own state for hidden) may be read only by such steps, within the
    draft round that wrote it. Slots of prompt positions (no hidden state recorded before them)
    are not checked.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.slots = {}
        self.forwards = 0
        self.poisoned = 0
        self.drafts = 0

    def _slot(self, table, r, p):
        return int(table[r, p // TOY.PAGE]), p % TOY.PAGE

    def _write(self, ids, params, spec):
        table = params["block_table"]
        seqlens = params["cache_seqlens"]
        hidden = host(params["target_hidden"])
        rows, width = ids.shape
        for r in range(rows):
            for i in range(width):
                p = int(seqlens[r]) + i
                self.slots[self._slot(table, r, p)] = ("spec" if spec else "true", int(ids[r, i]), float(hidden[r, i].reshape(-1)[0]))

    def _reads_ok(self, table, r, p, spec):
        for q in range(max(1, p - 16), p):
            prev = self._slot(table, r, q - 1)
            if prev not in self.model.hidden_at:
                continue        # q - 1 in the prompt
            key = self._slot(table, r, q)
            slot = self.slots.get(key)
            if slot is None:
                return False
            if slot[0] == "spec":
                if not spec:
                    return False
                continue
            job, logical = self.model.pages[key[0]]
            ids = job.sequences[0].sequence_ids.ids
            q_ = logical * TOY.PAGE + key[1]
            if q_ >= len(ids) or slot[1] != ids[q_] or slot[2] != self.model.hidden_at[prev]:
                return False
        return True

    def forward(self, batch_ids, params):
        assert params["target_hidden"].shape[:2] == batch_ids.shape
        self.forwards += 1
        spec = params.get("draft_step", 0) >= 1
        self._write(batch_ids, params, spec)
        table = params["block_table"]
        seqlens = params["cache_seqlens"]
        rows, width = batch_ids.shape
        out = torch.zeros((rows, width, 1))
        for r in range(rows):
            for i in range(width):
                ok = self._reads_ok(table, r, int(seqlens[r]) + i, spec)
                out[r, i, 0] = float(batch_ids[r, i]) if ok else POISON
        return dev(out) if self.model.devices else out

    def sample_from_state(self, state, params):
        state = host(state)
        poisoned = state.view(-1) >= POISON
        self.drafts += state.view(-1).shape[0]
        self.poisoned += int(poisoned.sum())
        ids = torch.where(poisoned.view(-1, 1), torch.full((state.view(-1).shape[0], 1), POISON_TOKEN),
                          (state.long().view(-1, 1) + 1) % VOCAB)
        return dev(ids) if self.model.devices else ids

    def prefill(self, ids, params):
        self.prefills += 1
        self._write(ids, params, False)


class FoldJob(WinJob):
    """WinJob whose pages the head cache can find, with the MTP carry a real prompt ingestion
    leaves (the target's hidden state before the position generation starts from)"""

    def _reset(self, ids):
        super()._reset(ids)
        seq = self.sequences[0]
        for i in range(len(seq.allocated_pages)):
            self.model.pages[self.first_page + i] = (self, i)

    def prefill(self, results):
        super().prefill(results)
        if self.is_prefill_done() and self.mtp_last_hidden is None:
            seq = self.sequences[0]
            q = seq.kv_position - 1
            h = self.model.hidden_at.get((int(seq.block_index_tensor[0, q // TOY.PAGE]), q % TOY.PAGE), 0.0)
            self.mtp_last_hidden = torch.full((1, 1, 1), h)


def run_fold(names, update, verify = False, devices = False, setup = None, **kw):
    """run() over the paged head cache, with mtp_batched_update set"""

    def setup_(g):
        g.model.pages = {}
        g.model.hidden_at = {}
        g.model.devices = devices
        head = CacheHead(g.model, g.model.modules[-1].idx, g.model.modules[-1].device)
        g.model.modules[-1] = head
        g.model.fwd_modules = [(m, 0, i) for i, m in enumerate(g.model.modules)]
        g.draft_model = CacheDraft(g.model)
        g.mtp_batched_update = update
        g.mtp_update_stats = {"passes": 0, "rows": 0, "folds": 0, "fold_drafts": 0, "fold_misses": 0}
        if setup is not None:
            setup(g)

    kw.setdefault("lengths", LENGTHS)
    kw.setdefault("job_cls", FoldJob)
    return run(names, verify = verify, setup = setup_, **kw)


def head_of(g):
    return g.draft_model


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("num_draft_tokens", [1, 2, 3])
@pytest.mark.parametrize("names", ["a", "ab", "abcd"])
def test_folded_updates_give_the_serial_tokens(names, num_draft_tokens, verify):
    ref, g0, ref_jobs = run_fold(names, update = False, num_draft_tokens = num_draft_tokens)
    out, g, jobs = run_fold(names, update = True, verify = verify, num_draft_tokens = num_draft_tokens)
    assert out == ref
    # The head's cache held what it had to at every read, in both runs, so the same drafts
    assert head_of(g0).poisoned == 0 and head_of(g).poisoned == 0
    assert drafts_of(jobs) == drafts_of(ref_jobs)
    stats = g.mtp_update_stats
    assert stats["folds"] > 20 and stats["fold_drafts"] > 20
    # One head pass a step instead of a draft pass plus an update per job that accepted
    steps = g.readback_stats["steps"]
    assert head_of(g).forwards + head_of(g).prefills < head_of(g0).forwards + head_of(g0).prefills
    assert head_of(g).prefills <= stats["passes"] - stats["fold_drafts"] + 5
    if num_draft_tokens == 1:
        assert head_of(g).forwards <= steps + stats["fold_misses"] + 5


def test_the_check_on_the_head_cache_sees_a_missing_update(monkeypatch):
    """The check on the test itself: with no update of the accepted positions the head reads slots
    that hold rejected positions' tokens, or nothing"""
    _, g0, _ = run_fold("ab", update = False)
    assert head_of(g0).poisoned == 0
    monkeypatch.setattr(GEN.Generator, "_mtp_update_pass", lambda self, *a: None)
    monkeypatch.setattr(GEN.Generator, "_mtp_fold", lambda self, *a: None)
    _, g, _ = run_fold("ab", update = True)
    assert head_of(g).poisoned > 5


def test_the_check_on_the_head_cache_sees_a_draft_taken_at_the_wrong_position(monkeypatch):
    real = GEN.Generator._mtp_fold

    def off_by_one(self, rows, *a):
        return real(self, [(job, a_, min(n + 1, self.num_draft_tokens + 1)) for job, a_, n in rows], *a)

    monkeypatch.setattr(GEN.Generator, "_mtp_fold", off_by_one)
    ref, _, _ = run_fold("ab", update = False)
    out, g, _ = run_fold("ab", update = True)
    # Exact all the same: drafts only decide what is verified
    assert out == ref
    assert head_of(g).poisoned > 5 or g.mtp_update_stats["fold_drafts"] == 0


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("names", ["abc", "abcd"])
def test_folded_updates_in_a_staggered_decode(names, verify):
    plain, _, _ = run_fold(names, update = False)
    ref, g0, ref_jobs = run_fold(names, update = False, stagger = True)
    out, g, jobs = run_fold(names, update = True, verify = verify, stagger = True)
    assert out == ref == plain
    assert head_of(g0).poisoned == 0 and head_of(g).poisoned == 0
    assert drafts_of(jobs) == drafts_of(ref_jobs)
    assert g.stagger_stats["steps"] > 20 and g.mtp_update_stats["fold_drafts"] > 20


@pytest.mark.parametrize("num_draft_tokens", [1, 2])
def test_folded_updates_across_requeues_and_checkpoint_cuts(num_draft_tokens):
    def script(g, jobs, n, add):
        for j in jobs.values():
            j.ckpt_every = 7

    kw = dict(num_draft_tokens = num_draft_tokens, rq_at = 11, script = script)
    ref, g0, ref_jobs = run_fold("abc", update = False, **kw)
    out, g, jobs = run_fold("abc", update = True, verify = True, **kw)
    assert out == ref
    assert head_of(g0).poisoned == 0 and head_of(g).poisoned == 0
    assert drafts_of(jobs) == drafts_of(ref_jobs)
    assert all(j.requeues > 2 for j in jobs.values())
    # A job that requeues is brought up to date by the batched update, not folded
    stats = g.mtp_update_stats
    assert stats["rows"] > stats["folds"] > 20


def test_a_job_that_joins_without_a_fold_makes_the_head_draft_for_all():
    def script(g, jobs, n, add):
        if n == 15:
            add("c")
        if n == 30:
            add("d")

    ref, g0, _ = run_fold("ab", update = False, script = script)
    out, g, _ = run_fold("ab", update = True, script = script)
    assert out == ref
    assert head_of(g).poisoned == 0
    assert g.mtp_update_stats["fold_misses"] >= 2


def test_a_fold_off_the_job_s_pages_is_not_made(monkeypatch):
    monkeypatch.setattr(GEN.Generator, "_fold_fits", staticmethod(lambda job, last: False))
    ref, _, _ = run_fold("ab", update = False)
    out, g, _ = run_fold("ab", update = True)
    assert out == ref and head_of(g).poisoned == 0
    assert g.mtp_update_stats["folds"] == 0 and g.mtp_update_stats["rows"] > 10


@pytest.mark.parametrize("names", ["a", "abc"])
def test_readbacks_per_step_with_the_update_folded(names):
    """With both switches, two readbacks a step whatever the number of jobs: the drafts the fold
    made, and the step's sampling"""
    def fold_setup(g):
        g.model.pages = {}
        g.model.hidden_at = {}
        g.model.devices = True
        head = CacheHead(g.model, g.model.modules[-1].idx, g.model.modules[-1].device)
        g.model.modules[-1] = head
        g.model.fwd_modules = [(m, 0, i) for i, m in enumerate(g.model.modules)]
        g.draft_model = CacheDraft(g.model)
        g.mtp_batched_update = True
        g.mtp_update_stats = {"passes": 0, "rows": 0, "folds": 0, "fold_drafts": 0, "fold_misses": 0}

    real = TOY.ToyJob
    for verify, expect in ((True, 2), (False, None)):
        TOY.ToyJob = FoldJob
        try:
            got, own = _readbacks_with(names, verify, fold_setup)
        finally:
            TOY.ToyJob = real
        assert got == pytest.approx(own)
        if expect is not None:
            assert got == pytest.approx(expect)
        else:
            # Serial sampling: 1 for the drafts, then one per position
            assert 1 + len(names) <= got <= 1 + 2 * len(names)


def _readbacks_with(names, verify, extra_setup):
    """readbacks_per_step's measure with a setup of our own and devices off in run()"""
    counted = {}

    def setup(g):
        extra_setup(g)
        real_wait = g._host_wait

        def wait(device):
            Readbacks.n += 1
            return real_wait(device)

        g._host_wait = wait

    def script(g, jobs, n, add):
        if len(jobs) == len(names) and all(1 <= j.new_tokens < 150 for j in jobs.values()):
            counted.setdefault("t0", (Readbacks.n, dict(g.readback_stats)))
            counted["t1"] = (Readbacks.n, dict(g.readback_stats))

    Readbacks.n = 0
    run(names, verify = verify, lengths = {n: 200 for n in names}, script = script, setup = setup, job_cls = FoldJob)
    (r0, s0), (r1, s1) = counted["t0"], counted["t1"]
    steps = s1["steps"] - s0["steps"]
    return (r1 - r0) / steps, (s1["readbacks"] - s0["readbacks"]) / steps
