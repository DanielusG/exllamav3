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
import stat
import sys
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


def make_env(prof = None, host = None, **gen_kwargs):
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
        host = host,
        yield_at = 1415,
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
    bench_prof_tokens = 24, bench_interleave = 1, bench_host_tokens = 24,
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


# --- the host's time: sampled stacks

LIB = "/lib/exllamav3"
GEN = LIB + "/generator/generator.py"


def stack(*frames) -> list:
    """Frames as (function, file, line) from "function@file:line" strings."""
    out = []
    for f in frames:
        name, _, where = f.partition("@")
        file, _, line = where.rpartition(":")
        out.append((name, file, int(line)))
    return out


def raw_line(frames: list, n: int) -> str:
    return ";".join(f"{name} ({file}:{line})" for name, file, line in frames) + f" {n}"


ISSUE_MIX = stack(
    f"iterate@{GEN}:600", f"iterate_staggered@{GEN}:1270", f"_stagger_issue@{GEN}:1171",
    f"run_to@{LIB}/model/forward_cursor.py:83", f"forward@{LIB}/modules/transformer.py:151",
    f"mix@{LIB}/modules/hyperconnections.py:520", f"_mix@{LIB}/modules/hyperconnections.py:479",
    f"ws@{LIB}/modules/hyperconnections.py:476", f"get_bucketed@{LIB}/util/tensor.py:237",
)
ISSUE_ROUTING = stack(
    f"iterate@{GEN}:600", f"iterate_gen@{GEN}:1301", f"finish@{LIB}/model/forward_cursor.py:114",
    f"decorate_context@/env/site-packages/torch/utils/_contextlib.py:124", f"forward@{LIB}/model/model.py:439",
    f"forward_ls@{LIB}/model/model_ls.py:360", f"forward@{LIB}/modules/transformer.py:187",
    f"forward@{LIB}/modules/block_sparse_mlp.py:1037", f"routing_std@{LIB}/modules/block_sparse_mlp_routing.py:130",
)
DRAFT = stack(f"iterate@{GEN}:600", f"iterate_staggered@{GEN}:1259", f"iterate_draftmodel_mtp_gen@{GEN}:870",
              f"forward@{LIB}/model/model.py:439", f"forward_ls@{LIB}/model/model_ls.py:360")
INPUTS = stack(f"iterate@{GEN}:600", f"iterate_staggered@{GEN}:1262", f"_iterate_gen_co@{GEN}:1415",
               f"_begin_forward@{GEN}:1058", f"__init__@{LIB}/model/forward_cursor.py:60")
SAMPLE = stack(f"iterate@{GEN}:600", f"iterate_staggered@{GEN}:1281", f"_stagger_sample@{GEN}:1185",
               f"_iterate_gen_co@{GEN}:1560", f"receive_sample@{LIB}/generator/job.py:941")
MTP_UPDATE = stack(f"iterate@{GEN}:600", f"iterate_gen@{GEN}:1303", f"_iterate_gen_co@{GEN}:1698",
                   f"prefill@{LIB}/model/model.py:387", f"forward_ls@{LIB}/model/model_ls.py:360")
ADVANCE = stack(f"iterate@{GEN}:600", f"iterate_staggered@{GEN}:1281", f"_stagger_sample@{GEN}:1182",
                f"finish@{LIB}/model/forward_cursor.py:97", f"advance_recurrent_states@{LIB}/cache/recurrent_util.py:70")
OWN = stack(f"iterate@{GEN}:600", f"iterate_staggered@{GEN}:1276", f"hold@{LIB}/model/forward_cursor.py:91")
OUTSIDE = stack("run@/x/stagger_bench.py:430")


def test_a_tagged_call_shows_its_name_on_the_stack():
    seen = []

    def body():
        f = sys._getframe(1)
        seen.append(f.f_code.co_name)
        return 7

    assert sb.tagged(sb.HOST_TAG + "3", body) == 7
    assert seen == ["bench_host__3"]
    assert sb._trampoline.__code__.co_name == "_trampoline"        # the template is not renamed


def test_stacks_are_parsed_and_cut_by_run():
    raw = "\n".join([
        raw_line(stack("<module>@/x/b.py:1", "bench_host__0@/x/b.py:5", "run@/x/b.py:9") + ISSUE_MIX, 30),
        raw_line(stack("<module>@/x/b.py:1", "bench_host__1@/x/b.py:5") + DRAFT, 4),
        raw_line(stack("_loop@/x/b.py:103"), 500),                  # another thread: no tagged frame
        "py-spy says something that is not a stack",
        "",
    ])
    stacks = sb.parse_stacks(raw)
    assert [n for n, _ in stacks] == [30, 4, 500]
    assert stacks[0][1][1] == ("bench_host__0", "/x/b.py", 5)
    runs = sb.host_runs(stacks)
    assert set(runs) == {"bench_host__0", "bench_host__1"}
    assert runs["bench_host__0"] == [(30, stack("run@/x/b.py:9") + ISSUE_MIX)]
    assert runs["bench_host__1"] == [(4, DRAFT)]


def test_a_stack_is_assigned_to_a_phase_by_the_generator_functions_on_it(tmp_path):
    y = 1415
    assert sb.host_phase(ISSUE_MIX, y) == (sb.PHASE_ISSUE, "hyperconnections: mix")
    assert sb.host_phase(ISSUE_ROUTING, y) == (sb.PHASE_ISSUE, "MoE: instradamento")
    assert sb.host_phase(DRAFT, y) == (sb.PHASE_DRAFT, None)       # its forward is not the pass
    assert sb.host_phase(INPUTS, y) == (sb.PHASE_INPUTS, None)
    assert sb.host_phase(SAMPLE, y) == (sb.PHASE_SAMPLE, None)
    assert sb.host_phase(MTP_UPDATE, y) == (sb.PHASE_MTP_UPDATE, None)
    assert sb.host_phase(ADVANCE, y) == (sb.PHASE_ADVANCE, None)
    assert sb.host_phase(OWN, y) == ("altro in iterate(): iterate_staggered", None)
    assert sb.host_phase(OUTSIDE, y) == (sb.PHASE_OUTSIDE, None)

    # Without the line of the yield the step's own code cannot be split, and is named as such
    own = stack(f"iterate@{GEN}:600", f"iterate_gen@{GEN}:1303", f"_iterate_gen_co@{GEN}:1330")
    assert sb.host_phase(own, None) == ("passo: codice proprio (ingressi o campionamento)", None)
    assert sb.host_phase(own, y) == (sb.PHASE_INPUTS, None)

    # The wait for the cards is the line of receive_sample that reads the token back
    job = tmp_path / "job.py"
    job.write_text("def receive_sample():\n    next_token = next_token.cpu()\n    piece = lookup(next_token)\n")
    head = stack(f"iterate@{GEN}:600", f"iterate_gen@{GEN}:1303", f"_iterate_gen_co@{GEN}:1560")
    assert sb.host_phase(head + [("receive_sample", str(job), 2)], y) == (sb.PHASE_WAIT, None)
    assert sb.host_phase(head + [("receive_sample", str(job), 3)], y) == (sb.PHASE_SAMPLE, None)

    # Components: the first library file below the loop decides, whatever the block's line says
    def comp(*tail):
        return sb.issue_component(stack(f"run_to@{LIB}/model/forward_cursor.py:83", f"forward@{LIB}/modules/transformer.py:187", *tail))

    assert comp(f"apply_@{LIB}/modules/hyperconnections.py:555") == "hyperconnections: apply_"
    assert comp(f"forward@{LIB}/modules/block_sparse_mlp.py:1092", f"cpu_split_submit@{LIB}/modules/block_sparse_mlp_cpu.py:251",
                f"submit_issue_fused@{LIB}/model/moe_cpu_host.py:1217") == "MoE: consegna al worker"
    assert comp(f"forward@{LIB}/modules/block_sparse_mlp.py:1395", f"cpu_split_combine@{LIB}/modules/block_sparse_mlp_cpu.py:314") \
        == "MoE: raccolta dal worker"
    assert comp(f"forward@{LIB}/modules/block_sparse_mlp.py:1381") == "MoE: esperti residenti e resto"
    assert comp(f"forward@{LIB}/modules/gated_delta_net.py:1075") == "attenzione lineare (GDN)"
    assert comp(f"forward@{LIB}/modules/attn.py:626", f"step@{LIB}/modules/attention_fn/bc_attn.py:622") == "attenzione QSA"
    assert comp() == "blocco (codice proprio)"
    assert sb.issue_component(stack(f"run_to@{LIB}/model/forward_cursor.py:83")) == "ciclo sui moduli"


def host_col(label: str, stagger: bool, rows: list, ms_step: float = 20.0, expected: float = 100.0):
    return (label, {"window": {"ms_step": ms_step}, "stagger": stagger}, sb.host_digest(rows, 1415), expected)


def cells(text: str, first: str) -> list:
    line = next(l for l in text.splitlines() if l.startswith(f"| {first} |"))
    return [c.strip() for c in line.strip("|").split("|")]


def test_the_host_section_turns_sample_shares_into_ms_per_call():
    rows = [(50, ISSUE_MIX), (20, ISSUE_ROUTING), (10, DRAFT), (10, SAMPLE), (5, INPUTS), (5, OUTSIDE)]
    digest = sb.host_digest(rows, 1415)
    assert digest["total"] == 100 and digest["scratch"] == 50
    assert digest["phases"][sb.PHASE_ISSUE] == 70
    assert digest["lines"][("ws", LIB + "/modules/hyperconnections.py", 476)] == 0      # torch-free leaf is tensor.py
    assert digest["lines"][("get_bucketed", LIB + "/util/tensor.py", 237)] == 50

    text = sb.host_section(
        [host_col("2 job, un batch", False, rows), host_col("2 job, sfalsato", True, rows, ms_step = 10.0),
         host_col("4 job, sfalsato", True, [(40, ISSUE_MIX)], expected = 300.0)],
        ["una nota"], 400, 250,
    )
    assert "una nota" in text and "250 volte al secondo" in text
    assert cells(text, "ms per chiamata") == ["ms per chiamata", "20.00", "10.00", "20.00"]
    assert cells(text, "campioni (attesi)") == ["campioni (attesi)", "100 (100)", "100 (100)", "40 (300)"]
    assert cells(text, sb.PHASE_ISSUE)[1:] == ["14.00", "7.00", "20.00"]
    assert cells(text, sb.PHASE_DRAFT)[1:] == ["2.00", "1.00", "-"]
    assert cells(text, sb.PHASE_OUTSIDE)[1:] == ["1.00", "0.50", "-"]
    assert cells(text, "hyperconnections: mix")[1:] == ["10.00", "5.00", "20.00"]
    assert cells(text, "MoE: instradamento")[1:] == ["4.00", "2.00", "-"]
    assert cells(text, "di cui nella cache dei buffer di lavoro (`util/tensor.py`)")[1:] == ["10.00", "5.00", "20.00"]
    # Phases in call order, the bench's own time last
    order = [l.split("|")[1].strip() for l in text.split("### Fasi di una chiamata")[1].split("###")[0].splitlines() if l.startswith("| ")]
    assert order[3:] == [sb.PHASE_DRAFT, sb.PHASE_INPUTS, sb.PHASE_ISSUE, sb.PHASE_SAMPLE, sb.PHASE_OUTSIDE]
    # A run with too few samples is flagged, the others are not
    warn = next(l for l in text.splitlines() if l.startswith("**Attenzione**"))
    assert "4 job, sfalsato" in warn and "2 job" not in warn
    # Hottest lines for the first and the last staggered run
    assert "### Righe più campionate: 2 job, sfalsato" in text and "### Righe più campionate: 4 job, sfalsato" in text
    assert "| 50.0% | 5.00 | `util/tensor.py:237` | `get_bucketed` |" in text

    assert "Nessun giro campionato." in sb.host_section([], [], 400, None)


class FakeCapture:
    """Stands for py-spy: a capture whose stacks follow the tags of the runs made meanwhile."""
    rate = 250
    messages = ""

    def __init__(self, fail = False):
        self.fail = fail
        self.active = False
        self.captures = 0

    def begin(self):
        assert not self.active
        self.active = True

    def end(self) -> str:
        assert self.active
        self.active = False
        self.captures += 1
        if self.fail:
            raise RuntimeError("py-spy gave no samples (exit code 1): no permission")
        lines = []
        for i in range(7):
            head = stack("<module>@/x/b.py:1", f"bench_host__{i}@/x/b.py:5", "run@/x/b.py:9")
            lines += [raw_line(head + ISSUE_MIX, 60 + i), raw_line(head + DRAFT, 10), raw_line(head + SAMPLE, 20)]
        return "\n".join(lines) + "\n"


def test_the_host_phase_runs_tagged_inside_one_capture():
    host = FakeCapture()
    env = make_env(prof = FakeProf(), host = host)
    extra = {}
    text = sb.measure(env, ARGS, extra = extra)
    assert "**Attenzione**" not in text and host.captures == 1 and not host.active
    part = section(text, "Dove va il tempo dell'host")
    header = cells(part, "fase")
    assert header == ["fase", "`a` da solo", "2 job, un batch", "2 job, sfalsato", "3 job, un batch", "3 job, sfalsato",
                      "4 job, un batch", "4 job, sfalsato"]
    assert cells(part, "campioni (attesi)")[1].startswith("90 (") and cells(part, "campioni (attesi)")[7].startswith("96 (")
    assert len(cells(part, sb.PHASE_ISSUE)) == 8 and "-" not in cells(part, sb.PHASE_ISSUE)
    assert "- Fasi eseguite, dopo piazzamento e riscaldamento: speed, check, prof, host" in section(text, "Condizioni")
    # The stacks go out as a second Markdown text, one part per run, without the tagged frame
    stacks = extra["stacks"]
    assert stacks.startswith("# ") and stacks.count("\n## ") == 7 and "## 4 job, sfalsato" in stacks
    assert "\n66 run (x/b.py:9);iterate (generator/generator.py:600);" in stacks and "bench_host__" not in stacks


def test_without_py_spy_or_with_a_failed_capture_the_report_says_so():
    text = sb.measure(make_env(prof = FakeProf()), ARGS)
    part = section(text, "Dove va il tempo dell'host")
    assert "py-spy non trovato" in part and "Nessun giro campionato." in part and "**Attenzione**" not in text

    host = FakeCapture(fail = True)
    text = sb.measure(make_env(prof = FakeProf(), host = host), ARGS)
    assert text.count("**Attenzione**") == 1 and not host.active
    assert "«where the host's time goes» è fallita con `RuntimeError: py-spy gave no samples" in text
    assert "Nessun giro campionato." in section(text, "Dove va il tempo dell'host")


def test_phases_can_be_left_out():
    host = FakeCapture()
    env = make_env(prof = FakeProf(), host = host)
    args = SimpleNamespace(**{**vars(ARGS), "bench_phases": ["host"]})
    text = sb.measure(env, args)
    assert "**Attenzione**" not in text and host.captures == 1
    assert "Misura di velocità non richiesta" in section(text, "Sintesi")
    assert "- Fasi eseguite, dopo piazzamento e riscaldamento: host" in section(text, "Condizioni")
    for title in ("Un job alla volta", "2 job insieme", "Stesso calcolo?", "Dove va il tempo\n"):
        assert "## " + title not in text
    assert "## Piazzamento degli esperti" in text and "## Dove va il tempo dell'host" in text

    # The check needs each job's own tokens, so it brings the solo runs along but not the speed table
    args = SimpleNamespace(**{**vars(ARGS), "bench_phases": ["check"]})
    text = sb.measure(make_env(prof = FakeProf()), args)
    assert "## Un job alla volta" in text and "## Stesso calcolo?" in text and "0 (identici)" in text
    assert "## 2 job insieme" not in text and "## Dove va il tempo" not in text


FAKE_PYSPY = """#!{python}
import signal, sys, time
args = sys.argv[1:]
out = args[args.index("--output") + 1]
mode = "{mode}"
if mode == "refuse":
    sys.stderr.write("Permission Denied: try running again with elevated permissions\\n")
    sys.exit(1)
assert args[0] == "record" and "--idle" in args and "--nonblocking" in args and "--full-filenames" in args
assert args[args.index("--format") + 1] == "raw"

def stop(*_):
    if mode == "ok":
        with open(out, "w") as f:
            f.write("<module> (/x/b.py:1);bench_host__0 (/x/b.py:5);iterate (/lib/generator.py:5) " + args[args.index("--rate") + 1] + "\\n")
    sys.exit(0)

signal.signal(signal.SIGINT, stop)
sys.stderr.write("attached to " + args[args.index("--pid") + 1] + "\\n")
sys.stderr.flush()
while True:
    time.sleep(0.02)
"""


def fake_pyspy(tmp_path, mode: str) -> str:
    path = tmp_path / f"py-spy-{mode}"
    path.write_text(FAKE_PYSPY.format(python = sys.executable, mode = mode))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_host_prof_starts_py_spy_on_this_process_and_reads_its_output(tmp_path):
    host = sb.HostProf(fake_pyspy(tmp_path, "ok"), rate = 125, startup = 0.5)
    host.begin()
    out = host._out
    raw = host.end()
    assert sb.host_runs(sb.parse_stacks(raw)) == {"bench_host__0": [(125, [("iterate", "/lib/generator.py", 5)])]}
    assert host.messages == f"attached to {os.getpid()}"
    assert not os.path.exists(out)                                  # nothing is left behind

    # py-spy that cannot attach: the capture fails at once, with its own words
    host = sb.HostProf(fake_pyspy(tmp_path, "refuse"), startup = 0.5)
    with pytest.raises(RuntimeError, match = "py-spy exited at once: Permission Denied"):
        host.begin()

    # py-spy that ends without samples
    host = sb.HostProf(fake_pyspy(tmp_path, "empty"), startup = 0.5)
    host.begin()
    with pytest.raises(RuntimeError, match = "py-spy gave no samples"):
        host.end()
    with pytest.raises(RuntimeError, match = "not started"):
        host.end()


def test_the_yield_of_the_decode_step_is_found():
    class WithStep:
        def _iterate_gen_co(self):
            a = 1
            logits, results = yield a
            return logits

    class Without:
        pass

    line = sb.decode_step_yield(WithStep)
    assert line == WithStep._iterate_gen_co.__code__.co_firstlineno + 2
    assert sb.decode_step_yield(Without) is None


def test_long_prompts_are_sized_by_tokens_and_differ_from_their_first_words():
    count = lambda text: len(text.split())                  # one token per word
    text = sb.long_prompt(count, "What is a B-tree?", "a", 6000)
    assert 5500 < count(text) < 6600 and text.endswith("Now set the document aside. What is a B-tree?")
    assert sb.long_prompt(count, "What is a B-tree?", "a", 6000) == text          # same prompt at every run
    # No shared first page between two jobs: the page table would hand both the same pages
    other = sb.long_prompt(count, "What is a B-tree?", "b", 6000)
    assert text.split()[:40] != other.split()[:40]
    # A tokenizer with three tokens per word gets a third of the paragraphs
    assert 5500 < 3 * count(sb.long_prompt(lambda t: 3 * len(t.split()), "Q?", "a", 6000)) < 6600

    env = make_env()
    assert "- Prompt: brevi (la sola domanda)" in section(sb.measure(env, ARGS), "Condizioni")
    env.prompt_desc = "da 60012 a 60340 token per job"
    assert "- Prompt: da 60012 a 60340 token per job" in section(sb.measure(env, ARGS), "Condizioni")
