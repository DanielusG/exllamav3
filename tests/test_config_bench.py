"""
The bench of one configuration (experiments/config_bench.py) from a loaded model to the text of
its report, on a toy generator: no GPU, extension or worker process. The bench itself can only
run on the machine with the model, with the server stopped.

The fakes are those of test_stagger_bench.py: they follow the shapes the library hands out, and
cannot tell whether the library still has them.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

import test_stagger_bench as tsb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(REPO, "experiments"))
_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
import config_bench as CB      # noqa: E402
if _alloc is None:
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)     # the bench sets it for its own process


def test_kl_of_the_listed_tokens():
    same = CB.kl_top([1, 2], [0.7, 0.2], [1, 2], [0.7, 0.2])
    assert same == pytest.approx(0.0, abs = 1e-12)
    moved = CB.kl_top([1, 2], [0.7, 0.2], [1, 2], [0.5, 0.4])
    assert moved == pytest.approx(0.7 * torch.log(torch.tensor(0.7 / 0.5)).item()
                                  + 0.2 * torch.log(torch.tensor(0.2 / 0.4)).item(), rel = 1e-6)
    # a token the run does not list gets the most it could have there
    gone = CB.kl_top([1, 2], [0.7, 0.3], [1, 3], [0.9, 0.05])
    assert gone > moved
    assert CB.kl_top([], [], [1], [1.0]) == 0.0


def bench_env(n_jobs = (3,)):
    env = tsb.make_env()
    env.__dict__.update(
        n_jobs_list = list(n_jobs), stagger = False,
        ingest = lambda i, chunk = None: {"tokens": 8192, "seconds": 8.0 + i, "chunk": chunk or 2048},
        memory = lambda: [("cuda:0", 140), ("cuda:1", 110)],
        host_cpus = [], host_pinned = 0, worker_threads = 12, ext_path = "/nowhere/ext.so", prompt_desc = None,
        pins = [],
    )
    env.pin = env.pins.append
    return env


def bench_args(**over):
    args = SimpleNamespace(
        bench_phases = list(CB.PHASES), bench_tokens = 40, bench_rounds = 2, bench_warmup = 8, bench_adapt = 2,
        bench_adapt_tokens = 16, bench_ingest_tokens = 8192, bench_ingest_rounds = 2,
    )
    args.__dict__.update(over)
    return args


def data_rows(part: str) -> list:
    """The cells of every table row of a part of the report, headers included."""
    return [[c.strip() for c in line.split("|")[1:-1]] for line in part.splitlines()
            if line.startswith("| ") and "---" not in line]


def test_whole_report_on_a_toy_generator():
    env = bench_env()
    text = CB.measure(env, bench_args())
    assert "Attenzione" not in text, text
    assert not env.gen.num_remaining_jobs() and env.gen.cleared == 0
    for title in ("Piazzamento degli esperti", "Stessi token, giri ripetuti", "Generazione libera, un job",
                  "3 job insieme", "Ingestione di un documento"):
        assert "## " + title in text
    # one configuration: one row of figures in each table of times
    for title in ("Generazione libera, un job", "3 job insieme"):
        header, *rows = data_rows(tsb.section(text, title))
        assert header[0] == "tok/s (mediana)" and len(rows) == 1
        assert float(rows[0][0]) > 0 and rows[0][-1] == "100.0%"
    forced = tsb.section(text, "Stessi token, giri ripetuti")
    header, *rows = data_rows(forced[forced.index("### Tempi"):forced.index("### Il rumore tra due giri")])
    assert header[0] == "tok/s (mediana)" and len(rows) == 1 and float(rows[0][0]) > 0
    # the toy computes the same thing at every round: four prompts of 40 tokens, nothing moved
    header, *rows = data_rows(forced[forced.index("### Il rumore tra due giri"):])
    assert header[0] == "posizioni" and len(rows) == 1
    assert rows[0][0] == "160" and rows[0][1] == "100.00%" and float(rows[0][2]) == 0.0 and rows[0][-1] == "sì"
    assert "cuda:0 140 MiB" in text


class Drifting(tsb.FakeGen):
    """The toy generator with a computation that does not repeat: from the second forced run of
    a prompt on the probabilities move and, every tenth position, another token is the most
    likely one."""

    def __init__(self, model):
        super().__init__(model)
        self.forced_runs = {}       # prompt -> forced runs started

    def iterate(self):
        for job in self.active:
            if job.forced is not None and job.new_tokens == 0:
                self.forced_runs[job.identifier] = self.forced_runs.get(job.identifier, 0) + 1
        results = super().iterate()
        for r in results:
            if "token_probs" in r and self.forced_runs[r["job"].identifier] > 1:
                r["token_probs"] = r["token_probs"] - 0.05
                kt, kp = r["top_k_tokens"].clone(), r["top_k_probs"].clone()
                kp[..., 0] -= 0.05
                kp[..., 1] += 0.05
                n0 = r["job"].new_tokens - kt.shape[1]
                for i in range(kt.shape[1]):
                    if (n0 + i) % 10 == 0:
                        kt[0, i] = kt[0, i].flip(0)
                r["top_k_tokens"], r["top_k_probs"] = kt, kp
        return results


def test_a_run_that_does_not_repeat_shows_in_the_noise():
    env = bench_env()
    env.gen = Drifting(env.model)
    text = CB.measure(env, bench_args(bench_phases = ["forced"]))
    assert "Attenzione" not in text, text
    forced = tsb.section(text, "Stessi token, giri ripetuti")
    row = data_rows(forced[forced.index("### Il rumore tra due giri"):])[1]
    assert row[1] == "90.00%" and float(row[2]) > 0.0 and row[-1] == "no"


def test_only_the_phases_asked_for_and_no_noise_from_a_single_round():
    text = CB.measure(bench_env(), bench_args(bench_phases = ["forced", "free"], bench_rounds = 1))
    assert "Attenzione" not in text, text
    forced = tsb.section(text, "Stessi token, giri ripetuti")
    assert "### Tempi" in forced and "### Il rumore" not in forced      # no second run to compare with
    assert "## Generazione libera, un job" in text
    assert "## 3 job insieme" not in text and "## Ingestione di un documento" not in text


def test_a_failed_phase_is_reported_and_the_rest_survives():
    env = bench_env()
    real = env.gen.iterate

    def boom():
        # the forced runs fail, everything else goes through
        if any(job.forced is not None for job in env.gen.active):
            raise RuntimeError("boom")
        return real()

    env.gen.iterate = boom
    text = CB.measure(env, bench_args(bench_phases = ["forced", "free"]))
    assert "è fallita con `RuntimeError: boom`" in text
    # the job of the failed run is out of the queue, and the phase after it has its figures
    assert env.gen.cleared == 1 and not env.gen.num_remaining_jobs() and env.gen.iterate is boom
    assert len(data_rows(tsb.section(text, "Stessi token, giri ripetuti"))) == 1       # the header alone
    header, *rows = data_rows(tsb.section(text, "Generazione libera, un job"))
    assert len(rows) == 1 and float(rows[0][0]) > 0


def test_the_host_phase_compares_pinned_and_free_threads_and_leaves_them_pinned():
    env = bench_env()
    env.host_cpus = [10, 11, 22, 23]
    text = CB.measure(env, bench_args(bench_phases = ["host"]))
    assert "Attenzione" not in text, text
    part = tsb.section(text, "Thread di questo processo: liberi o fissati")
    assert "[10, 11, 22, 23]" in part
    rows = {row[0]: row[1:] for row in data_rows(part)}
    # pinned against free, run by run: four prompts, two rounds
    assert float(rows["thread liberi"][0]) > 0 and rows["thread liberi"][1:3] == ["-", "-"]
    assert rows["thread fissati"][1].endswith("%") and rows["thread fissati"][2].endswith("/8")
    assert env.pins == [False, True, True, False, True]           # alternated, and pinned again at the end
    # without CPUs to pin to there is nothing to compare
    assert "liberi o fissati" not in CB.measure(bench_env(), bench_args(bench_phases = ["host"]))


def test_the_ingestion_can_compare_chunk_sizes():
    env = bench_env()
    text = CB.measure(env, bench_args(bench_phases = ["ingest"], bench_chunks = [2048, 4096]))
    part = tsb.section(text, "Ingestione di un documento")
    rows = [line for line in part.splitlines() if line.startswith("| ") and "---" not in line][1:]
    assert [r.split("|")[2].strip() for r in rows] == ["2048", "4096", "4096", "2048"]


def test_one_prompt_ingested_again_with_other_chunk_sizes():
    env = bench_env()
    env.gen.max_chunk_size = 2048
    seen = []
    env.forget = lambda: seen.append(env.gen.max_chunk_size) or True
    sizes = []
    step = env.gen.iterate
    env.gen.iterate = lambda: (sizes.append(env.gen.max_chunk_size), step())[1]
    text = CB.measure(env, bench_args(bench_phases = ["chunkcmp"], bench_chunks = [2048, 4096]))
    assert "Attenzione" not in text, text
    part = tsb.section(text, "Stesso prompt, pezzi di prefill diversi")
    rows = [line for line in part.splitlines() if line.startswith("| ") and "---" not in line]
    there, back = rows[1:5], rows[6:]
    assert [r.split("|")[1].strip() for r in there] == [
        "2048 (riferimento)", "4096", "2048 il primo passaggio, poi 4096", "2048 (il rumore)"]
    assert [r.split("|")[1].strip() for r in back] == ["4096 (riferimento)", "2048"]
    # every run started from a forgotten prompt, and the generator has its own chunk size back
    assert len(seen) == 7 and env.gen.max_chunk_size == 2048
    assert env.gen.iterate() is not None and sizes[-1] == 2048
    # stretches of calls at one size: reference, 4096, the staged arm (one call at 2048, then 4096),
    # the noise, the other way round (4096 writes and is forced, then 2048)
    groups = [[sizes[0], 0]]
    for x in sizes:
        if x != groups[-1][0]:
            groups.append([x, 0])
        groups[-1][1] += 1
    assert [g[0] for g in groups] == [2048, 4096, 2048, 4096, 2048, 4096, 2048], groups
    assert groups[2][1] == 1, groups
    # the toy model computes the same thing whatever the chunk: identical to the reference
    assert all(r.split("|")[9].strip() == "sì" for r in there[1:] + back[1:]), rows
    # nothing to compare without sizes, or with a model that cannot forget
    assert "pezzi di prefill diversi" not in CB.measure(bench_env(), bench_args(bench_phases = ["chunkcmp"]))
    env = bench_env()
    env.gen.max_chunk_size = 2048
    env.forget = lambda: False
    assert "Attenzione" in CB.measure(env, bench_args(bench_phases = ["chunkcmp"], bench_chunks = [4096]))


def test_the_turn_phase_reports_the_cost_and_the_next_turn():
    env = bench_env()
    env.gen.max_chunk_size = 2048
    state = {"tail": 0, "stats": {"stashes": 0, "replaced": 0, "t_stash": 0.0}, "turns": []}
    step = env.gen.iterate

    def iterate():
        # a tail checkpoint every four calls while the switch is on
        if state["tail"]:
            state["n"] = state.get("n", 0) + 1
            if state["n"] % 4 == 0:
                state["stats"]["stashes"] += 1
                state["stats"]["replaced"] += 1
                state["stats"]["t_stash"] += 0.05
        return step()

    def turn(ids, max_new):
        state["turns"].append(state["tail"])
        n = int(ids.shape[-1])
        return {"prompt": n, "resumed": (n - 40 if state["tail"] else n - 1200) // 256 * 256, "ttft": 0.4 if state["tail"] else 1.9}

    env.gen.iterate = iterate
    env.__dict__.update(
        set_tail_ckpt = lambda n: state.__setitem__("tail", n), ckpt_stats = lambda: dict(state["stats"]),
        turn = turn, suffix = torch.zeros((1, 6), dtype = torch.long), page = 256, forget = lambda: True, torch = torch,
    )
    text = CB.measure(env, bench_args(bench_phases = ["turn"], bench_turn_tokens = 48))
    assert "Attenzione" not in text, text
    part = tsb.section(text, "Il checkpoint che segue la risposta")
    rows = [line for line in part.splitlines() if line.startswith("| ") and "---" not in line]
    cost = [r for r in rows if r.split("|")[1].strip() in ("senza", "con")]
    assert [r.split("|")[1].strip() for r in cost] == ["senza", "con"]
    assert cost[0].split("|")[6].strip() in ("0", "0.0") and float(cost[1].split("|")[6]) > 0      # stashes per run
    assert cost[1].split("|")[7].strip() == "50.0"                                                 # ms per stash
    nxt = [r for r in rows if r.split("|")[2].strip() in ("sì", "no")]
    assert [r.split("|")[2].strip() for r in nxt] == ["no", "sì", "sì", "no"] and state["turns"] == [0, 256, 256, 0]
    assert state["tail"] == 0                           # left off
    # a library without the feature: nothing to measure, and no failure
    assert "che segue la risposta" not in CB.measure(bench_env(), bench_args(bench_phases = ["turn"]))


def test_host_threads_can_be_pinned():
    cpus = sorted(os.sched_getaffinity(0))
    assert CB.pin_host_threads(cpus) >= 1
    assert sorted(os.sched_getaffinity(0)) == cpus
