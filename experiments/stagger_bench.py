#!/usr/bin/env python
"""
Bench of the staggered decode (Generator.stagger_decode) on the production model.

The model is loaded exactly as the OpenAI server loads it (same flags, through the server's own
ModelContainer), then a raw Generator is driven from this process. For 2, 3 and 4 jobs
generating at the same time it measures the regular decode (one batch) and the staggered one,
and checks with greedy sampling that every job of the staggered run yields the same tokens as
its group does alone.

Run it with the server STOPPED (it needs the server's VRAM), from the server repository:

    cd ~/Desktop/exllamav3-openai-server
    python /path/to/exllamav3/experiments/stagger_bench.py --model-path ... <the server's flags>

The library is imported from the checkout this file is in, ahead of an installed one; the
compiled extension (a top-level module of its own) still comes from the installed package, so a
branch that only changes Python needs no build as long as it is based on the installed version.

Options of the bench itself start with --bench-; everything else goes to the server's argument
parser. The report is a Markdown file (prof-runs/ of the current directory when it exists).
"""

from __future__ import annotations

import os

# Before torch is imported, as the server does (golden rule 2 of its AGENTS.md)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import argparse
import statistics
import subprocess
import sys
import threading
import time

PROMPTS = [
    "Write a detailed technical essay, at least 1500 words long, on how a B-tree index works "
    "in a relational database: node layout, search, insertion with splits, deletion with "
    "merges, and why it suits disks.",
    "Write a long short story, at least 1500 words, about a lighthouse keeper who finds a "
    "message in a bottle every morning for a month. Use rich descriptions and dialogue.",
    "Write a complete Python module, with docstrings and comments, that implements an LRU "
    "cache, a trie with prefix search, and a priority queue based on a binary heap. Then "
    "explain each design choice at length.",
    "Explain at length, in at least 1500 words, the history of the steam engine from Newcomen "
    "to the locomotive, with dates, people, and the engineering problems each step solved.",
]


class GpuSampler:
    """utilization.gpu and power.draw of every card, one nvidia-smi call per sample (a looping
    nvidia-smi behind a pipe buffers its output for seconds)."""

    def __init__(self, period: float = 0.3):
        self.period = period
        self.samples = []       # (t, [(util, watts) per card])
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target = self._loop, name = "bench-gpu", daemon = True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout = 5)

    def _loop(self):
        cmd = ["nvidia-smi", "--query-gpu=index,utilization.gpu,power.draw", "--format=csv,noheader,nounits"]
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                out = subprocess.run(cmd, capture_output = True, text = True, timeout = 3).stdout
                row = []
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    row.append((float(parts[1]), float(parts[2])))
                if row:
                    self.samples.append((time.perf_counter(), row))
            except Exception as e:
                self.error = repr(e)
                return
            self._stop.wait(max(0.0, self.period - (time.perf_counter() - t0)))

    def mean(self, t_a: float, t_b: float, guard: float = 0.5):
        """Per card (mean util %, mean W, samples) between two instants, skipping the start:
        the counter is an average over the time before each sample."""
        rows = [row for t, row in self.samples if t_a + guard <= t <= t_b]
        if not rows:
            return None
        cards = len(rows[0])
        return [
            (
                statistics.fmean(r[c][0] for r in rows),
                statistics.fmean(r[c][1] for r in rows),
                len(rows),
            )
            for c in range(cards)
        ]


class Timer:
    """Wraps a bound method on an instance and adds up the time spent in it."""

    def __init__(self, obj, name: str):
        self.total = 0.0
        self.calls = 0
        self._obj = obj
        self._name = name
        self._orig = getattr(obj, name)
        timer = self

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return timer._orig(*args, **kwargs)
            finally:
                timer.total += time.perf_counter() - t0
                timer.calls += 1

        setattr(obj, name, timed)

    def take(self):
        total, calls = self.total, self.calls
        self.total, self.calls = 0.0, 0
        return total, calls

    def remove(self):
        try:
            delattr(self._obj, self._name)
        except AttributeError:
            pass


def run_once(gen, Job, sampler_cls, prompts: dict, tokens: int, stagger: bool, gpu: GpuSampler | None, timers: dict):
    """
    Generate `tokens` tokens for every prompt at the same time. Returns a dict with the tokens
    of each job and the figures of the stretch in which ALL the jobs were generating (from the
    iteration after the last job's first token to the one before the first job's last).
    """
    gen.stagger_decode = stagger
    stats0 = dict(gen.stagger_stats)
    for t in timers.values():
        t.take()

    jobs = {}
    for name, ids in prompts.items():
        job = Job(input_ids = ids, max_new_tokens = tokens, sampler = sampler_cls(), identifier = name)
        jobs[name] = job
    by_job = {id(job): name for name, job in jobs.items()}
    gen.enqueue(list(jobs.values()))

    out = {name: [] for name in jobs}
    done = set()
    timeline = []       # (t after the iteration, tokens so far per job)
    marks = []          # timer and stagger totals after each iteration
    while gen.num_remaining_jobs():
        results = gen.iterate()
        now = time.perf_counter()
        for r in results:
            name = by_job.get(id(r.get("job")))
            if r.get("stage") == "error":
                raise RuntimeError(f"job {name} failed: {r.get('error')!r}")
            if r.get("stage") != "streaming":
                continue
            ids = r.get("token_ids")
            if ids is not None:
                out[name] += ids.view(-1).tolist()
            if r.get("eos"):
                done.add(name)
        timeline.append((now, {n: j.new_tokens + getattr(j, "rq_new_tokens", 0) for n, j in jobs.items()}, len(done)))
        marks.append((
            {k: (t.total, t.calls) for k, t in timers.items()},
            dict(gen.stagger_stats),
        ))

    # The stretch with every job generating
    first = next((i for i, (_, toks, _) in enumerate(timeline) if all(v >= 1 for v in toks.values())), None)
    last = next((i for i, (_, _, n_done) in enumerate(timeline) if n_done > 0), len(timeline)) - 1
    res = {
        "tokens": out,
        "jobs": len(jobs),
        "stagger": stagger,
        "accepted": sum(j.accepted_draft_tokens for j in jobs.values()),
        "rejected": sum(j.rejected_draft_tokens for j in jobs.values()),
        "window": None,
    }
    if first is not None and last > first + 4:
        t_a, toks_a, _ = timeline[first]
        t_b, toks_b, _ = timeline[last]
        n_tok = sum(toks_b.values()) - sum(toks_a.values())
        steps = last - first
        dur = t_b - t_a
        tm_a, st_a = marks[first]
        tm_b, st_b = marks[last]
        res["window"] = {
            "seconds": dur,
            "tokens": n_tok,
            "steps": steps,
            "tok_s": n_tok / dur,
            "ms_step": 1000.0 * dur / steps,
            "tok_step": n_tok / steps,
            "per_job": {n: (toks_b[n] - toks_a[n]) / dur for n in toks_a},
            "timers_ms_step": {k: 1000.0 * (tm_b[k][0] - tm_a[k][0]) / steps for k in tm_a},
            "stagger": {k: st_b[k] - st_a[k] for k in st_b},
            "gpu": gpu.mean(t_a, t_b) if gpu is not None else None,
        }
    res["stagger_total"] = {k: gen.stagger_stats[k] - stats0[k] for k in stats0}
    return res


def first_difference(a: list, b: list):
    """None when one list is a prefix of the other (or equal), else the first index that differs."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def fmt(x, nd = 1):
    return "-" if x is None else f"{x:.{nd}f}"


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(add_help = False)
    ap.add_argument("--bench-jobs", default = "2,3,4", help = "numeri di job da provare (default 2,3,4)")
    ap.add_argument("--bench-tokens", type = int, default = 400, help = "token generati da ogni job (default 400)")
    ap.add_argument("--bench-warmup", type = int, default = 48, help = "token del giro di riscaldamento (default 48)")
    ap.add_argument("--bench-interleave", type = int, default = 1, help = "moduli per turno nell'emissione alternata (default 1)")
    ap.add_argument("--bench-out", default = None, help = "file .md del rapporto")
    ap.add_argument("--bench-no-gpu", action = "store_true", help = "non campionare nvidia-smi")
    ap.add_argument("--bench-server-repo", default = os.getcwd(), help = "repository del server (default: directory corrente)")
    ap.add_argument("--bench-help", action = "store_true")
    args, server_argv = ap.parse_known_args()
    if args.bench_help:
        ap.print_help()
        print(__doc__)
        return 0
    if args.bench_out is not None and not args.bench_out.endswith(".md"):
        raise SystemExit("--bench-out: il rapporto è un file .md")

    # The server's package first, then this checkout's library ahead of an installed one
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.abspath(args.bench_server_repo))
    import logging
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch
    import exllamav3
    from exllamav3 import Generator, Job
    from exllamav3.generator.sampler import GreedySampler
    from exllamav3.modules import block_sparse_mlp_cpu
    from exllamav3.util import device_copy
    from openai.config import parse_args
    from openai.model_container import ModelContainer

    if not hasattr(Generator, "iterate_staggered"):
        raise SystemExit(
            f"La libreria importata ({exllamav3.__file__}) non ha il decode sfalsato: "
            "installa il fork nell'ambiente (pip install -e) e riprova."
        )

    n_jobs_list = sorted({int(x) for x in args.bench_jobs.split(",") if x.strip()})
    cfg = parse_args(server_argv)
    container = ModelContainer(cfg)
    container.load_model_sync()
    model, tokenizer = container.model, container.tokenizer

    draft_model = container.draft_model
    gen_kwargs = {}
    if draft_model is not None:
        gen_kwargs.update(
            draft_model = draft_model,
            draft_cache = container.draft_cache,
            num_draft_tokens = cfg.draft_num_tokens,
        )
    if getattr(cfg, "recurrent_checkpoint_interval", None) is not None:
        gen_kwargs["recurrent_checkpoint_interval"] = cfg.recurrent_checkpoint_interval
    gen = Generator(
        model = model,
        cache = container.cache,
        tokenizer = tokenizer,
        max_batch_size = container.max_batch_size,
        max_chunk_size = cfg.max_chunk_size,
        **gen_kwargs,
    )
    gen.stagger_interleave = args.bench_interleave
    max_jobs = min(max(n_jobs_list), gen.max_batch_size, len(PROMPTS))
    n_jobs_list = [n for n in n_jobs_list if 2 <= n <= max_jobs]

    # A placement sweep between two runs moves experts between VRAM and the CPU worker, which
    # changes the numerics of the layers it touches: greedy runs would no longer be comparable
    block_sparse_mlp_cpu.run_pending_swap_sweeps = lambda infer_params: None

    def encode(text: str):
        chat = getattr(model, "default_chat_prompt", None)
        if chat is not None:
            text = chat(text)
        return tokenizer.encode(text, encode_special_tokens = True)

    names = "abcd"
    prompts = {names[i]: encode(PROMPTS[i]) for i in range(max_jobs)}

    timers = {
        "draft": Timer(gen, "iterate_draftmodel_mtp_gen"),
        "forward": Timer(model, "forward"),
    }
    gpu = None
    if not args.bench_no_gpu:
        gpu = GpuSampler()
        gpu.start()

    def run(job_names, stagger, tokens = args.bench_tokens):
        return run_once(gen, Job, GreedySampler, {n: prompts[n] for n in job_names}, tokens, stagger, gpu, timers)

    def groups_of(n: int):
        """The two groups a staggered run of the first n jobs settles into (Generator._stagger_pick)."""
        first = n - n // 2
        return [names[:first], names[first:n]]

    print(" -- bench: warm-up", flush = True)
    run(names[:1], False, args.bench_warmup)
    for n in n_jobs_list:
        for g in groups_of(n):
            run(g, False, args.bench_warmup)
        run(names[:n], False, args.bench_warmup)
        run(names[:n], True, args.bench_warmup)

    report = []
    summary = []
    solo = {}
    print(" -- bench: one job at a time", flush = True)
    for name in names[:max_jobs]:
        solo[name] = run(name, False)
    solo_again = run(names[0], False)
    repeat_diff = first_difference(solo[names[0]]["tokens"][names[0]], solo_again["tokens"][names[0]])

    rows = []
    for name, r in solo.items():
        w = r["window"]
        rows.append([name, len(r["tokens"][name]), fmt(w and w["tok_s"]), fmt(w and w["ms_step"], 2), fmt(w and w["tok_step"], 2)])
    report.append("## Un job alla volta\n\n" + table(["job", "token", "tok/s", "ms per passo", "token per passo"], rows))
    report.append(
        f"Ripetizione del job `{names[0]}` da solo: "
        + ("stessi token." if repeat_diff is None else f"**token diversi dall'indice {repeat_diff}**: "
           "due esecuzioni identiche non danno gli stessi token, quindi il confronto qui sotto non può "
           "pretendere l'uguaglianza esatta.")
    )

    refs = {}       # group string -> run
    for n in n_jobs_list:
        print(f" -- bench: {n} jobs", flush = True)
        grp = groups_of(n)
        for g in grp:
            if g not in refs:
                refs[g] = solo[g] if len(g) == 1 else run(g, False)
        batched = run(names[:n], False)
        staggered = run(names[:n], True)

        rows = []
        for label, r in (("un batch", batched), ("sfalsato", staggered)):
            w = r["window"]
            if w is None:
                rows.append([label, "finestra troppo corta"] + ["-"] * 9)
                continue
            acc = r["accepted"] + r["rejected"]
            g_ = w["gpu"]
            st = w["stagger"]
            steps = max(1, w["steps"])
            if r["stagger"]:
                host = (f"bozza+ingressi {1000 * st['t_launch'] / steps:.2f}, emissione {1000 * st['t_issue'] / steps:.2f}, "
                        f"campionamento {1000 * st['t_sample'] / steps:.2f}")
            else:
                tm = w["timers_ms_step"]
                rest = w["ms_step"] - tm["draft"] - tm["forward"]
                host = f"bozza {tm['draft']:.2f}, emissione {tm['forward']:.2f}, resto {rest:.2f}"
            rows.append([
                label,
                fmt(w["tok_s"]),
                " / ".join(fmt(v) for v in w["per_job"].values()),
                fmt(w["ms_step"], 2),
                fmt(w["tok_step"], 2),
                fmt(r["accepted"] / acc if acc else None, 3),
                " + ".join(f"{u:.0f}%" for u, _, _ in g_) if g_ else "-",
                " + ".join(f"{p:.0f} W" for _, p, _ in g_) if g_ else "-",
                g_[0][2] if g_ else 0,
                fmt(w["seconds"], 1),
                host,
            ])
        section = [f"## {n} job insieme\n", table(
            ["modo", "tok/s totali", "tok/s per job", "ms per chiamata", "token per chiamata",
             "accettazione bozza", "uso schede", "potenza", "campioni GPU", "finestra s", "tempo host per chiamata (ms)"],
            rows,
        )]
        wb, ws = batched["window"], staggered["window"]
        if wb and ws:
            gain = ws["tok_s"] / wb["tok_s"] - 1.0
            summary.append([n, fmt(wb["tok_s"]), fmt(ws["tok_s"]), f"{100 * gain:+.1f}%"])
            section.append(f"\nSfalsato rispetto a un batch: **{100 * gain:+.1f}%** di tok/s totali.")
        st = staggered["stagger_total"]
        section.append(
            f"\nContatori del decode sfalsato sull'intero giro: {st['steps']} chiamate, {st['launched']} gruppi emessi, "
            f"{st['sampled']} campionati, {st['solo']} completati da soli, {st['drains']} svuotamenti. "
            f"Gruppi attesi: {' e '.join('`' + g + '`' for g in grp)}."
        )

        # Tokens: every job of the staggered run against its group alone
        rows = []
        for g in grp:
            for name in g:
                ref = refs[g]["tokens"][name]
                d_s = first_difference(ref, staggered["tokens"][name])
                d_b = first_difference(ref, batched["tokens"][name])
                rows.append([
                    name, f"`{g}`", len(staggered["tokens"][name]),
                    "uguali" if d_s is None else f"**diversi dall'indice {d_s}**",
                    "uguali" if d_b is None else f"diversi dall'indice {d_b}",
                ])
        section.append("\n" + table(
            ["job", "gruppo di riferimento", "token", "sfalsato contro il gruppo da solo", "un batch contro il gruppo da solo"],
            rows,
        ))
        report.append("\n".join(section))

    if gpu is not None:
        gpu.stop()
    for t in timers.values():
        t.remove()

    split = gen._stagger_plan()
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    head = [
        "# Decode sfalsato: misura",
        "",
        time.strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "## Sintesi",
        "",
        table(["job insieme", "un batch tok/s", "sfalsato tok/s", "differenza"], summary) if summary else "Nessuna misura completa.",
        "",
        "La colonna «sfalsato contro il gruppo da solo» delle sezioni sotto dice se il decode sfalsato calcola "
        "le stesse cose di quello normale: con campionamento greedy ogni job deve dare gli stessi token che dà "
        "il suo gruppo da solo. La colonna «un batch» è solo informativa: comporre il batch in modo diverso può "
        "cambiare qualche arrotondamento.",
        "",
        "## Condizioni",
        "",
        f"- ExLlamaV3 {getattr(exllamav3, '__version__', '?')} da `{os.path.dirname(exllamav3.__file__)}`, torch {torch.__version__}",
        f"- Modello `{cfg.model_path}`",
        f"- Schede: {', '.join(devices)}",
        f"- Confine tra le schede: modulo {split} di {len(model.fwd_modules)}"
        + ("" if split else " (**decode sfalsato non disponibile**)"),
        f"- Bozza: {'MTP, ' + str(gen.num_draft_tokens) + ' token' if draft_model is not None else 'nessuna'}",
        f"- Token per job {args.bench_tokens}, moduli per turno {args.bench_interleave}, batch massimo {gen.max_batch_size}",
        f"- Copie tra schede: {device_copy.stats['direct']} dirette, {device_copy.stats['bounced']} passate dalla RAM"
        + (" (**una copia passata dalla RAM blocca l'host a metà passo**)" if device_copy.stats["bounced"] else ""),
        "- Sweep del piazzamento degli esperti sospesi per tutta la misura; campionamento greedy; nessuna condizione di stop",
        "- Le cifre di ogni riga valgono per il tratto in cui tutti i job generano; «chiamata» è una `iterate()`: "
        "nel modo normale un passo per tutti i job, nel modo sfalsato un passo per un gruppo",
        "- Uso e potenza delle schede: media dei campioni di `nvidia-smi` nel tratto, saltando il primo mezzo secondo"
        + (f" (campionatore fermato da un errore: {gpu.error})" if gpu is not None and gpu.error else ""),
        "",
    ]
    text = "\n".join(head) + "\n" + "\n\n".join(report) + "\n"

    out = args.bench_out
    if out is None:
        folder = "prof-runs" if os.path.isdir("prof-runs") else "."
        out = os.path.join(folder, time.strftime("stagger-bench-%Y%m%d-%H%M%S.md"))
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print(f" -- bench: report written to {out}", flush = True)

    # Orderly release of the cards and of the CPU expert worker. The expert tier is NOT closed:
    # its close() saves the placement, and the production file must not learn from a bench
    try:
        if draft_model is not None:
            draft_model.unload()
        model.unload()
    except Exception as e:
        print(f" -- bench: unload failed ({e!r}); exiting anyway", flush = True)
    return 0


if __name__ == "__main__":
    code = main()
    # No interpreter teardown: exit hooks of the server's modules (placement save among them)
    # do not belong in a bench, and the CPU expert worker ends with its parent (PDEATHSIG)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code or 0)
