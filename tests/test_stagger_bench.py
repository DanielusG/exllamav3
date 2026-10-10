"""
The report side of experiments/stagger_bench.py on fakes (no GPU, no model, no extension): the
bench itself can only run on the machine with the model, with the server stopped, so everything
between "a loaded model" and "the text of the report" is exercised here first.

The fakes follow the shapes the library hands out (result dicts of Generator.iterate(), the hit
histogram and placement map of a split MoE layer); what they cannot tell is whether the library
still has those shapes.
"""

import importlib.util
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_ALLOC = "PYTORCH_CUDA_ALLOC_CONF"


def _load():
    # The bench sets the CUDA allocator for its own process at import; not for this one
    before = os.environ.get(_ALLOC)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "stagger_bench.py")
    spec = importlib.util.spec_from_file_location("stagger_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if before is None:
        os.environ.pop(_ALLOC, None)
    else:
        os.environ[_ALLOC] = before
    return mod


sb = _load()


def natural_token(name: str, i: int) -> int:
    return (ord(name) * 31 + i * 7) % 1000


class FakeJob:

    def __init__(self, input_ids, max_new_tokens, sampler, identifier, return_probs = False, return_top_tokens = 0):
        self.identifier = identifier
        self.max_new_tokens = max_new_tokens
        self.return_probs = return_probs
        self.return_top_tokens = return_top_tokens
        self.new_tokens = 0
        self.accepted_draft_tokens = 0
        self.rejected_draft_tokens = 0
        self.forced = None

    def constrain_output_now(self, ids):
        assert ids.dim() == 2 and ids.shape[0] == 1
        self.forced = ids.view(-1).tolist()


class FakeLayer:
    """A split MoE layer: experts 0..3 in VRAM, 4..7 on the CPU worker; expert 6 is hot."""

    def __init__(self):
        self.cpu_split_first = 4
        self._split_map = torch.arange(8)
        self._split_hist = torch.zeros(8)

    def hit(self, n: int):
        self._split_hist[2] += 3 * n
        self._split_hist[6] += n


class FakeModel:

    def __init__(self):
        self.fwd_modules = []
        self.forwards = 0
        self.config = SimpleNamespace(
            infer_params = SimpleNamespace(moe_cpu_swap_modules = [FakeLayer(), FakeLayer()]),
            moe_cpu_hosts = {},
        )

    def forward(self, *args, **kwargs):
        self.forwards += 1

    def sweep(self):
        """Promote the hot expert once, decay the counts (what the library's sweep does)."""
        for m in self.config.infer_params.moe_cpu_swap_modules:
            if int(m._split_map[6]) >= m.cpu_split_first:
                a, b = int(m._split_map[0]), int(m._split_map[6])
                m._split_map[0], m._split_map[6] = b, a
            m._split_hist.mul_(0.5)


class FakeProf:
    """Stands in for StreamProf: the fake generator files a step's brackets itself."""

    def __init__(self):
        self.step = 0
        self.fallbacks = 0
        self.resolved = []
        self.active = False

    def begin(self):
        self.resolved, self.active = [], True

    def end(self):
        self.active = False

    def file(self):
        for dev, pass_ms, wait_ms in ((0, 5.0, 0.2), (1, 9.0, 0.4)):
            self.resolved.append(("pass", dev, self.step, pass_ms))
            for _ in range(3):
                self.resolved.append(("collect", dev, self.step, wait_ms))
                self.resolved.append(("issue", dev, self.step, 0.01))


class FakeGen:
    """
    Two tokens per job and call. Staggered, the jobs take turns in two groups. `skew` is what a
    wrong staggered computation would look like from outside: the probabilities move and, every
    tenth position, another token becomes the most likely one.
    """

    max_batch_size = 4
    num_draft_tokens = 1

    def __init__(self, model, prof = None, skew = 0.0, fail_forced_stagger = False):
        self.model = model
        self.prof = prof
        self.skew = skew
        self.fail_forced_stagger = fail_forced_stagger
        self.stagger_decode = False
        self.stagger_stats = {"steps": 0, "launched": 0, "sampled": 0, "solo": 0, "drains": 0,
                              "t_launch": 0.0, "t_issue": 0.0, "t_sample": 0.0}
        self.active = []
        self.turn = 0
        self.cleared = 0

    def enqueue(self, jobs):
        self.active += jobs

    def num_remaining_jobs(self):
        return len(self.active)

    def clear_queue(self):
        self.active = []
        self.cleared += 1

    def iterate_draftmodel_mtp_gen(self, results):
        return None

    def _stagger_issue(self, late, early, split):
        pass

    def iterate(self):
        results = []
        jobs = list(self.active)
        staggered = self.stagger_decode and len(jobs) > 1
        if staggered:
            if self.fail_forced_stagger and jobs[0].forced is not None:
                raise RuntimeError("boom")
            half = len(jobs) - len(jobs) // 2
            jobs = jobs[:half] if self.turn % 2 == 0 else jobs[half:]
            self.turn += 1
            for k in ("steps", "launched", "sampled"):
                self.stagger_stats[k] += 1
            self._stagger_issue(None, None, 2)
        else:
            self.model.forward()
        self.iterate_draftmodel_mtp_gen(results)
        if self.prof is not None and self.prof.active:
            self.prof.file()
        for job in jobs:
            n = min(2, job.max_new_tokens - job.new_tokens)
            ids, probs, kt, kp = [], [], [], []
            for _ in range(n):
                i = job.new_tokens
                natural = natural_token(job.identifier, i)
                p = 0.6 + 0.03 * (i % 10)
                first, second = natural, natural + 1
                if staggered and self.skew:
                    p -= self.skew
                    if i % 10 == 0:
                        first, second = second, first
                ids.append(job.forced[i] if job.forced is not None else natural)
                probs.append(p)
                kt.append([first, second])
                kp.append([p, 0.9 * (1.0 - p)])
                job.new_tokens += 1
            job.accepted_draft_tokens += max(0, n - 1)
            eos = job.new_tokens >= job.max_new_tokens
            r = {"job": job, "stage": "streaming", "eos": eos, "token_ids": torch.tensor([ids])}
            if job.return_probs:
                r["token_probs"] = torch.tensor([probs])
            if job.return_top_tokens:
                r["top_k_tokens"] = torch.tensor([kt])
                r["top_k_probs"] = torch.tensor([kp])
            results.append(r)
            if eos:
                self.active.remove(job)
            for m in self.model.config.infer_params.moe_cpu_swap_modules:
                m.hit(n)
        return results


def make_env(prof = None, **gen_kwargs):
    model = FakeModel()
    gen = FakeGen(model, prof = prof, **gen_kwargs)
    return SimpleNamespace(
        torch = torch,
        gen = gen,
        model = model,
        Job = FakeJob,
        sampler_cls = object,
        prompts = {name: torch.zeros((1, 8), dtype = torch.long) for name in sb.NAMES},
        max_jobs = 4,
        n_jobs_list = [2, 3, 4],
        sweep = model.sweep,
        prof = prof,
        split = 2,
        min_jobs_default = 3,
        n_modules = 5,
        version = "0.0.0",
        revision = "abc1234 2026-10-10",
        lib_path = "/nowhere",
        torch_version = torch.__version__,
        model_path = "/nowhere/model",
        devices = ["card A", "card B"],
        draft_desc = "MTP, 1 token",
        copy_stats = lambda: (12, 0),
    )


ARGS = SimpleNamespace(
    bench_adapt = 8, bench_adapt_tokens = 16, bench_warmup = 8, bench_tokens = 40,
    bench_prof_tokens = 24, bench_interleave = 1,
)


def section(text: str, title: str) -> str:
    start = text.index("## " + title)
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def test_whole_report_on_a_generator_that_computes_the_same_staggered():
    env = make_env(prof = FakeProf())
    text = sb.measure(env, ARGS)

    assert "**Attenzione**" not in text
    assert "- Banco di prova: `stagger_bench.py` al commit abc1234 2026-10-10" in section(text, "Condizioni")
    summary = section(text, "Sintesi")
    for n in (2, 3, 4):
        assert f"\n| {n} | " in summary

    # The placement settles after the sweep that promotes the hot expert
    placement = section(text, "Piazzamento degli esperti")
    assert "| 1 | 0.750 | 2 |" in placement
    assert "| 2 | 1.000 | 0 |" in placement
    assert "| 3 |" not in placement

    # Every comparison of every job: all positions, the forced token on top, nothing moved
    check = section(text, "Stesso calcolo?")
    rows = [line for line in check.splitlines() if line.startswith("| ") and ("contro" in line or "rifatto" in line)]
    assert len(rows) == 3 * (2 + 3 + 4)
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[2:] == ["40", "40", "0", "-", "-", "0 (identici)", "0.00000"], row

    # Brackets per call and card, as filed
    prof = section(text, "Dove va il tempo")
    solo = next(line for line in prof.splitlines() if line.startswith("| `a` da solo"))
    cells = [c.strip() for c in solo.strip("|").split("|")]
    assert cells[2:] == ["5.00 / 9.00", "0.60 / 1.20", "0.03 / 0.03", "3.0 / 3.0", "0.200 / 0.400", "0.200 / 0.400", "0"]
    assert len([line for line in prof.splitlines() if "sfalsato |" in line]) == 3

    # The instruments are gone and the generator is empty
    assert "forward" not in vars(env.model)
    assert "iterate_draftmodel_mtp_gen" not in vars(env.gen)
    assert env.gen.num_remaining_jobs() == 0


def test_a_different_staggered_computation_shows_in_the_check():
    env = make_env(skew = 0.25)
    text = sb.measure(env, ARGS)
    check = section(text, "Stesso calcolo?")
    for line in check.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if "sfalsato contro" in cells[1]:
            # 4 of 40 positions (every tenth) with another token on top, the lead there is the
            # reference's own: p - 0.9 (1 - p) at p = 0.6
            assert cells[3] == "36"
            assert cells[4] == "4 (primo all'indice 0)"
            assert cells[5] == "0.2400"
            assert cells[6] == "0"                  # the probabilities differ from the first position
            assert cells[7] == "0.2500"
        elif "rifatto" in cells[1] or "un batch contro" in cells[1]:
            assert cells[2:] == ["40", "40", "0", "-", "-", "0 (identici)", "0.00000"], line


def test_a_failed_phase_is_reported_and_the_rest_survives():
    env = make_env(fail_forced_stagger = True)
    text = sb.measure(env, ARGS)
    assert text.count("**Attenzione**") == 2      # the check, and the profile (no StreamProf here)
    assert "«same computation (forced tokens)» è fallita con `RuntimeError: boom`" in text
    assert env.gen.cleared == 1
    assert "\n| 4 | " in section(text, "Sintesi")
    assert "Nessun confronto completato." in section(text, "Stesso calcolo?")
    assert "Nessun giro di profilo completato." in section(text, "Dove va il tempo")


def test_no_adaptation_when_asked_or_without_a_dynamic_placement():
    args = SimpleNamespace(**{**vars(ARGS), "bench_adapt": 0})
    text = sb.measure(make_env(), args)
    assert "Adattamento disattivato" in section(text, "Piazzamento degli esperti")

    env = make_env()
    env.model.config.infer_params.moe_cpu_swap_modules = []
    text = sb.measure(env, ARGS)
    assert "niente da adattare" in section(text, "Piazzamento degli esperti")
    assert "\n| 2 | " in section(text, "Sintesi")


def test_compare_forced():
    ref = {"probs": {"a": [0.9, 0.5, 0.7]}, "top": {"a": [([1, 2], [0.9, 0.05]), ([3, 4], [0.5, 0.45]), ([5, 6], [0.7, 0.2])]}}
    run = {"probs": {"a": [0.9, 0.4, 0.7]}, "top": {"a": [([1, 2], [0.9, 0.05]), ([4, 3], [0.5, 0.4]), ([5, 6], [0.7, 0.2])]},
           "tokens": {"a": [1, 3, 5]}}
    c = sb.compare_forced(ref, run, "a")
    assert c["n"] == 3 and c["flips"] == 1 and c["first_flip"] == 1 and c["on_top"] == 2
    assert c["first_delta"] == 1
    assert c["max"] == pytest.approx(0.1) and c["mean"] == pytest.approx(0.1 / 3)
    assert c["margin"] == pytest.approx(0.05)
    assert sb.compare_forced(ref, {"probs": {}, "top": {}, "tokens": {}}, "a") is None


def test_stream_prof_digest_keeps_the_calls_of_the_window():
    resolved = [("collect", 0, step, 1.0) for step in range(10) for _ in range(2)]
    resolved += [("pass", 1, step, float(step)) for step in range(10)]
    d = sb.StreamProf.digest(resolved, 3, 7)        # calls 4, 5, 6, 7
    assert d[("collect", 0)]["ms_call"] == pytest.approx(2.0)
    assert d[("collect", 0)]["n_call"] == pytest.approx(2.0)
    assert d[("pass", 1)]["ms_call"] == pytest.approx((4 + 5 + 6 + 7) / 4)
    assert d[("pass", 1)]["med"] == 6.0 and d[("pass", 1)]["p90"] == 7.0


def test_groups_follow_the_generators_pick():
    assert sb.groups_of(2) == ["a", "b"]
    assert sb.groups_of(3) == ["ab", "c"]
    assert sb.groups_of(4) == ["ab", "cd"]


class FakeEvent:
    """A CUDA event on a clock that ticks once per record()."""

    clock = 0.0

    def __init__(self, enable_timing = False):
        self.t = None

    def record(self):
        FakeEvent.clock += 1.0
        self.t = FakeEvent.clock

    def elapsed_time(self, other):
        return other.t - self.t


class FakeCuda:

    Event = FakeEvent

    def __init__(self):
        self.entered = []
        self.synced = []

    def device(self, dev):
        cuda = self

        class Scope:
            def __enter__(self):
                cuda.entered.append(dev.index)

            def __exit__(self, *exc):
                return False

        return Scope()

    def device_count(self):
        return 2

    def synchronize(self, d):
        self.synced.append(d)


class FakeHost:
    """The two fused handoffs of MoeCpuHost, with the argument order of their call sites."""

    cap_rows = 64

    def submit_issue_fused(self, layer_idx, y, selected_experts, routing_weights, split_map, split_hist, first_cpu):
        if y.rows > self.cap_rows:
            return None
        return (layer_idx + 100, 0, y.rows, 8, 8, y.device, None)

    def submit_collect_fused(self, handle, final_2d):
        final_2d.append(handle[0])


def test_stream_prof_brackets_the_handoffs_and_the_passes_then_leaves_no_trace():
    dev0, dev1 = SimpleNamespace(index = 0), SimpleNamespace(index = 1)
    fake_torch = SimpleNamespace(cuda = FakeCuda())
    host = FakeHost()
    model = FakeModel()
    model.config.moe_cpu_hosts = {"text": host}
    gen = FakeGen(model)
    timer = sb.Timer(model, "forward")              # as Bench does: the profile wraps on top of it
    timed = vars(model)["forward"]

    prof = sb.StreamProf(fake_torch, model, gen, (dev0, dev1))
    prof.begin()
    prof.step = 7
    y = SimpleNamespace(device = dev1, rows = 4)
    handle = host.submit_issue_fused(3, y, None, None, None, None, 5)
    assert handle[0] == 103 and handle[5] is dev1
    collected = []
    host.submit_collect_fused(handle, collected)
    assert collected == [103]
    assert host.submit_issue_fused(3, SimpleNamespace(device = dev0, rows = 65), None, None, None, None, 5) is None
    model.forward()
    gen._stagger_issue(object(), None, 2)           # only a second-card half
    gen._stagger_issue(None, object(), 2)           # only a first-card half
    prof.end()

    kinds = [(kind, dev, step) for kind, dev, step, _ in prof.resolved]
    assert kinds == [
        ("issue", 1, 7), ("collect", 1, 7), ("issue", 0, 7),
        ("pass", 0, 7), ("pass", 1, 7),
        ("pass", 1, 7), ("pass", 0, 7),
    ]
    assert all(ms > 0 for _, _, _, ms in prof.resolved)
    assert prof.fallbacks == 1
    assert fake_torch.cuda.synced == [0, 1]
    assert model.forwards == 1 and timer.calls == 1

    assert "submit_issue_fused" not in vars(host) and "submit_collect_fused" not in vars(host)
    assert "_stagger_issue" not in vars(gen)
    assert vars(model)["forward"] is timed
    timer.remove()
    assert "forward" not in vars(model)


def test_stage_devices_are_the_cards_on_either_side_of_the_boundary():
    def mod(dev):
        return (SimpleNamespace(device = dev), None, None)

    model = SimpleNamespace(fwd_modules = [mod(None), mod("cuda:0"), mod("cuda:0"), mod("cuda:1"), mod("cpu")])
    d0, d1 = sb.stage_devices(torch, model, 3)
    assert (d0.type, d0.index, d1.type, d1.index) == ("cuda", 0, "cuda", 1)
    d0, d1 = sb.stage_devices(torch, model, None)
    assert d0.index == 0 and d1 is None


def test_the_bench_names_its_own_commit():
    rev = sb.bench_revision()
    assert rev and "\n" not in rev
