"""
The prefill bench (experiments/prefill_bench.py) from a loaded model to the text of its report,
on the toy model of test_stagger_decode: the real generator, with and without its prefill
thread, behind the bench's own runner. No GPU, model or server is required.
"""

import os
import sys
import threading
import types

import pytest
import torch

import test_stagger_decode as toy
from test_stagger_decode import ToyJob, _host_torch, make_generator      # noqa: F401
from test_prefill_thread import Slots, _fake_lane, lasting, long_prompt   # noqa: F401

sys.path.insert(0, os.path.join(toy.REPO, "experiments"))
import prefill_bench as PB      # noqa: E402

CHUNK = 16
GIB = 1024 ** 3


class EnvTorch:
    """torch as the bench sees it: no card to ask about memory."""

    class cuda:
        reset_peak_memory_stats = staticmethod(lambda d: None)
        max_memory_allocated = staticmethod(lambda d: (3 + d) * GIB)
        mem_get_info = staticmethod(lambda d: (GIB // 2, 16 * GIB))

    def __getattr__(self, name):
        return getattr(torch, name)


def make_env(steps = 5):
    g = make_generator()
    g.model.host = Slots(4)
    g.model.config.moe_cpu_hosts = {"cuda:0": g.model.host}
    g.side_min_tokens = CHUNK
    g.max_chunk_size = CHUNK
    resets = []
    g.pagetable.reset_page_table = lambda: resets.append(len(resets))
    g.recurrent_cache = types.SimpleNamespace(prune_stranded = lambda: 0)

    class BenchJob(ToyJob):
        """Job as the bench makes it, over the toy job: forced tokens, probabilities, results
        with the keys of the real ones."""

        def __init__(self, input_ids, max_new_tokens, sampler = None, identifier = None,
                     return_probs = False, return_top_tokens = 0):
            super().__init__(g.model, identifier, input_ids.reshape(-1).tolist(), max_new_tokens, chunk = CHUNK)
            self.identifier = identifier
            self.return_probs = return_probs
            self.forced = None
            self.hurry = threading.Event()
            if steps:
                self.on_chunk = lasting(steps)

        def constrain_output_now(self, tokens):
            self.forced = tokens.reshape(-1).tolist()

        def receive_logits(self, logits):
            self.mid_pass = False
            if self.forced is not None:
                return torch.tensor([[self.forced[len(self.out)]]]), None, None, None
            return torch.argmax(logits, dim = -1).view(1, 1), None, None, None

        def receive_sample(self, logits, next_token, next_k_tokens, next_k_probs, next_prob, results):
            eos, tok, rq = super().receive_sample(logits, next_token, next_k_tokens, next_k_probs, next_prob, results)
            r = results[-1]
            r["token_ids"] = next_token.view(1, 1)
            if self.return_probs:
                # The toy head puts all the weight on one token: 1 or 0 for the forced one
                p = logits.reshape(-1).float()
                p = p / p.sum().clamp(min = 1.0)
                best = torch.topk(p, 2)
                r["token_probs"] = p[int(next_token.item())].view(1, 1)
                r["top_k_tokens"] = best.indices.view(1, 1, 2)
                r["top_k_probs"] = best.values.view(1, 1, 2)
            return eos, tok, rq

    env = types.SimpleNamespace(
        torch = EnvTorch(),
        gen = g,
        model = g.model,
        Job = BenchJob,
        sampler_cls = lambda: None,
        prompts = {
            "a": torch.tensor([toy.PROMPTS["a"]], dtype = torch.long),
            "b": torch.tensor([long_prompt(3, 8 * CHUNK + 1)], dtype = torch.long),
        },
        sweep = None,
        slots = lambda: "4: 2 per il decode e 2 per il prefill",
        device_ids = [0, 1],
        version = "0.0.0-test",
        revision = "0000000",
        lib_path = "/lib",
        torch_version = torch.__version__,
        model_path = "/models/toy",
        devices = ["card 0", "card 1"],
        draft_desc = "MTP, 1 token",
    )
    env.resets = resets
    return env


ARGS = types.SimpleNamespace(
    bench_tokens = 120, bench_decoder_tokens = 2000, bench_reader_tokens = 12, bench_lead = 6, bench_warmup = 8,
    bench_adapt = 0, bench_adapt_tokens = 20, bench_phases = ["speed", "check"],
)


def rows_of(report: str, title: str) -> list:
    """Cells of the table under a section title, header included."""
    body = report.split(title, 1)[1]
    lines = [l for l in body.split("\n## ")[0].splitlines() if l.startswith("|")]
    return [[c.strip() for c in l.strip("|").split("|")] for l in lines if not set(l) <= set("|-: ")]


@pytest.fixture
def env():
    e = make_env()
    yield e
    if e.gen._side_pool is not None:
        e.gen._side_pool.shutdown(wait = True)


# --------------------------------------------------------------------------------------------
#   Reading a run
# --------------------------------------------------------------------------------------------

def test_reading_a_run():
    rec = {
        "timeline": [(1.0, {"a": 2}), (2.0, {"a": 4}), (3.0, {"a": 5, "b": 0}), (4.0, {"a": 9, "b": 1}), (5.0, {"a": 11, "b": 3})],
        "enqueued": {"a": 0.5, "b": 2.0},
        "first_token": {"a": 1.0, "b": 4.0},
        "prompt_tokens": {"a": 5, "b": 400},
        "prefill": {"a": [], "b": [(3.0, 256, 400), (4.0, 400, 400)]},
        "tokens": {"a": list(range(11)), "b": [7, 8, 9]},
        "probs": {"a": [0.5] * 11, "b": [0.1] * 3},
        "top": {"a": [None] * 11, "b": [None] * 3},
    }
    assert PB.tokens_at(rec, "a", 2.0) == 4 and PB.tokens_at(rec, "a", 2.5) == 4 and PB.tokens_at(rec, "a", 0.1) == 0
    assert PB.tokens_before(rec, "a", 4.0) == 5 and PB.tokens_before(rec, "a", 4.5) == 9
    w = PB.ingestion(rec)
    assert (w["t0"], w["t1"], w["seconds"], w["chunks"]) == (2.0, 4.0, 2.0, 2) and w["tok_s"] == 200.0
    # Between the enqueue of b and its first token a went from 4 to 9 tokens
    assert PB.decode_rate(rec, "a", w["t0"], w["t1"]) == (2.5, 5)
    assert PB.whole_rate(rec, "a") == (11 - 2) / 4.0
    assert PB.ingestion(rec, "c") is None
    short = PB.cut(rec, "a", 3)
    assert short["tokens"]["a"] == [0, 1, 2] and len(short["probs"]["a"]) == 3 and short["tokens"]["b"] == [7, 8, 9]
    assert len(rec["tokens"]["a"]) == 11


def test_a_run_of_the_runner(env):
    run = PB.Runner(env)
    a, b = env.prompts["a"], env.prompts["b"]
    rec = run.run(True, dict(ids = a, max_new = 2000), dict(ids = b, max_new = 12), lead = 6)
    assert env.resets == [0]
    assert len(rec["tokens"]["b"]) == 12 and rec["cancelled"] == ["a"]
    assert not env.gen.num_remaining_jobs()
    # b was enqueued once a had its lead, and its prompt went in eight chunks beside a's decode
    w = PB.ingestion(rec)
    assert w["chunks"] == 8 and w["prompt_tokens"] == 8 * CHUNK + 1
    assert PB.tokens_at(rec, "a", w["t0"]) >= 6
    assert rec["side"]["beside"] == 8 and rec["side"]["decode_steps"] >= 8 * 5
    rate, n = PB.decode_rate(rec, "a", w["t0"], w["t1"])
    assert n >= 8 * 5 and rate > 0
    assert rec["peak"] == {0: 3 * GIB, 1: 4 * GIB}


def test_the_decoder_must_outlast_its_lead(env):
    run = PB.Runner(env)
    with pytest.raises(RuntimeError, match = "before its lead"):
        run.run(False, dict(ids = env.prompts["a"], max_new = 4), dict(ids = env.prompts["b"], max_new = 4), lead = 50)
    env.gen.clear_queue()


# --------------------------------------------------------------------------------------------
#   The whole measure
# --------------------------------------------------------------------------------------------

def test_the_report(env):
    report = PB.measure(env, ARGS)
    assert "**Attenzione**" not in report

    summary = rows_of(report, "## Sintesi")
    assert [r[0] for r in summary[1:]] == [
        "`a` da solo", "`b` da solo, come oggi", "`b` da solo, sul thread", "insieme, come oggi", "insieme, col thread",
    ]
    assert all(c != "-" for c in summary[4][1:]) and all(c != "-" for c in summary[5][1:])

    window = rows_of(report, "## La finestra di ingestione")
    today, thread = window[1], window[2]
    assert (today[0], thread[0]) == ("come oggi", "col thread")
    assert today[1] == thread[1] == "8"
    # As today a step of decode per chunk (one or two tokens); with the thread, the steps that
    # fit beside each chunk
    assert 8 <= int(today[2]) <= 2 * 8 + 4
    assert int(thread[2]) >= 8 * 5
    assert (today[4], today[5]) == ("0", "0")
    assert thread[4] == "8" and int(thread[5]) >= 8 * 5

    check = rows_of(report, "## Stesso calcolo")
    assert len(check) == 5
    assert all(r[-1].startswith("**identico**") for r in check[1:]), check
    assert "generate durante l'ingestione" in check[2][-1]
    assert int(check[2][1]) >= 8 * 5            # positions of a compared: those inside the window

    mem = rows_of(report, "## Memoria delle schede")
    assert mem[0] == ["giro", "picco cuda:0", "picco cuda:1", "libera cuda:0", "libera cuda:1"]
    assert mem[1][1:] == ["3.00", "4.00", "0.50", "0.50"] and len(mem) == 5

    assert "Slot del worker degli esperti: 4: 2 per il decode e 2 per il prefill" in report
    assert "almeno 16 token" in report
    # Every run started from an empty prompt cache
    assert len(env.resets) >= 12


def test_a_wrong_number_beside_the_decode_is_reported(env, monkeypatch):
    # The check is not blind: a layer that folds one wrong value into the state of a job whose
    # chunk runs on the prefill thread changes that job's tokens from there on
    real = toy.layer_step

    def layer_step(state, layer, tokens, pos0, page0):
        out = real(state, layer, tokens, pos0, page0)
        if threading.current_thread().name.startswith("exl3-prefill") and layer == 3 and pos0 == 2 * CHUNK:
            state.acc[layer] += 1
        return out

    monkeypatch.setattr(toy, "layer_step", layer_step)
    report = PB.measure(env, ARGS)
    check = rows_of(report, "## Stesso calcolo")
    assert check[1][-1].startswith("**identico**") and check[3][-1].startswith("**identico**")    # the noise floors
    assert check[2][-1].startswith("**identico**")              # a is not touched
    assert check[4][-1].startswith("**diverso**"), check[4]     # b ingested on the thread


def test_a_failed_phase_is_said_and_the_rest_goes_on(env):
    calls = {"n": 0}
    real = env.Job

    def job(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("no such job")
        return real(*a, **k)

    env.Job = job
    report = PB.measure(env, types.SimpleNamespace(**dict(vars(ARGS), bench_phases = ["speed"])))
    assert "**Attenzione**: la fase «as today (thread off)» è fallita con `RuntimeError: no such job`" in report
    assert "## Stesso calcolo" not in report
    summary = rows_of(report, "## Sintesi")
    assert summary[5][0] == "insieme, col thread" and summary[5][1] != "-"
