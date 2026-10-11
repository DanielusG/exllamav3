#!/usr/bin/env python3
"""
Bench of one configuration on the production model: the decode of a job alone and of several
together, optionally the ingestion of a document. One process measures one configuration: for
an A/B of the server's flags or of thread placement (--bench-host-cpus), one process per arm
and the reports side by side.

One model load, the expert placement adapted to the bench's own prompts and then left where it
is for every phase:

  - forced: every job is generated again with the tokens it produced alone forced on it. Each
    run then works on the same sequence: the time of a step, the tokens a step yields (the
    drafts the main model accepts), and how far the probabilities moved from one run to the
    next (most likely token changed, KL estimated on the most likely tokens, probability given
    to the forced text). The run done twice is the noise floor of the configuration;
  - free, jobs: greedy generation, a job alone and several jobs together. This is what a client
    sees, and it moves with the text as well as with the configuration;
  - host: this process's threads on CPUs of their own (--bench-host-cpus) against the same
    threads left to the scheduler, in the same process;
  - ingest: a document nothing has seen, from the enqueue to its first token, in prefill passes
    of the server's size or of others (--bench-chunks);
  - chunkcmp: one prompt ingested again in prefill passes of other sizes (--bench-chunks) with
    its own text forced on it: where a pass ends is not something the model computes with, so
    the probabilities must stay where they were;
  - turn: the checkpoint that follows a generating job (Generator's recurrent_tail_checkpoint):
    what it costs a long answer, and what the next turn of the conversation saves with it.

Run it with the server STOPPED, from the server repository, with the server's own flags:

    cd ~/Desktop/exllamav3-openai-server
    python /path/to/exllamav3/experiments/config_bench.py --bench-out prof-runs/config.md <the server's flags>

Options of the bench start with --bench-; everything else goes to the server's argument parser.
The server's expert placement file is never written.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stagger_bench import (     # noqa: E402
    NAMES, PROMPTS, Bench, bench_revision, fmt, long_prompt, placement_counts, placement_maps, table,
)

TOP_TOKENS = 32     # most likely tokens recorded per position of a forced run
PHASES = ("forced", "free", "jobs", "host", "ingest", "chunkcmp", "turn")


# --------------------------------------------------------------------------------------------
#   Figures of a run
# --------------------------------------------------------------------------------------------

def kl_top(ref_tokens: list, ref_probs: list, run_tokens: list, run_probs: list) -> float:
    """
    KL(reference || run) of one position from the most likely tokens of each (sorted lists):
    over the reference's listed tokens plus one bucket for everything else. A token the
    reference lists and the run does not is given the most it can have there (the run's
    smallest listed probability, or what its list leaves over). An estimate from below; exact
    as the lists grow.
    """
    eps = 1e-9
    if not ref_probs or not run_probs:
        return 0.0
    q_of = dict(zip(run_tokens, run_probs))
    left_run = max(0.0, 1.0 - sum(run_probs))
    absent = sum(1 for t in ref_tokens if t not in q_of)
    q_absent = max(eps, min(min(run_probs), left_run / absent)) if absent else 0.0
    kl = 0.0
    q_sum = 0.0
    for t, p in zip(ref_tokens, ref_probs):
        q = q_of.get(t, q_absent)
        q_sum += q
        if p > 0.0:
            kl += p * math.log(p / max(q, eps))
    p_rest = max(0.0, 1.0 - sum(ref_probs))
    if p_rest > eps:
        kl += p_rest * math.log(p_rest / max(1.0 - q_sum, eps))
    return max(kl, 0.0)


def quantile(values: list, q: float):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def compare_runs(ref: dict, run: dict, name: str):
    """One job of a forced run against the same job of the reference run (same forced tokens).
    None when either recorded nothing."""
    p_r, p_x = ref["probs"].get(name, []), run["probs"].get(name, [])
    t_r, t_x = ref["top"].get(name, []), run["top"].get(name, [])
    forced = run["tokens"].get(name, [])
    n = min(len(p_r), len(p_x), len(t_r), len(t_x), len(forced))
    if n == 0:
        return None
    kl = [kl_top(t_r[i][0], t_r[i][1], t_x[i][0], t_x[i][1]) for i in range(n)]
    nll_r = -sum(math.log(max(p, 1e-12)) for p in p_r[:n]) / n
    nll_x = -sum(math.log(max(p, 1e-12)) for p in p_x[:n]) / n
    return {
        "n": n,
        "flips": sum(1 for i in range(n) if t_r[i][0][0] != t_x[i][0][0]),
        "kl_mean": statistics.fmean(kl),
        "kl_p99": quantile(kl, 0.99),
        "kl_max": max(kl),
        "ppl_ratio": math.exp(nll_x - nll_r),
        "on_top": sum(1 for i in range(n) if t_x[i][0][0] == forced[i]),
        "identical": all(a == b for a, b in zip(p_r[:n], p_x[:n])),
    }


def pool(parts: list):
    """Several jobs' compare_runs as one (positions pooled)."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    n = sum(p["n"] for p in parts)
    return {
        "n": n,
        "flips": sum(p["flips"] for p in parts),
        "kl_mean": sum(p["kl_mean"] * p["n"] for p in parts) / n,
        "kl_p99": max(p["kl_p99"] for p in parts),
        "kl_max": max(p["kl_max"] for p in parts),
        "ppl_ratio": math.exp(sum(math.log(p["ppl_ratio"]) * p["n"] for p in parts) / n),
        "on_top": sum(p["on_top"] for p in parts),
        "identical": all(p["identical"] for p in parts),
    }


def median(values: list):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def pin_host_threads(cpus: list) -> int:
    """Every thread this process has now goes to `cpus` (threads made later inherit from the
    one that makes them). The CPU expert worker is another process and keeps its own pins.
    Returns how many threads were moved."""
    moved = 0
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
            moved += 1
        except OSError:
            pass
    return moved


# --------------------------------------------------------------------------------------------
#   From a loaded model to the text of the report
# --------------------------------------------------------------------------------------------

def measure(env, args, gpu = None) -> str:
    torch, gen, model = env.torch, env.gen, env.model
    names = NAMES[:env.max_jobs]
    bench = Bench(env, gpu)
    bench.top_tokens = TOP_TOKENS
    phases = set(args.bench_phases)
    notes = []
    sections = []

    def phase(label: str, fn):
        print(f" -- bench: {label}", flush = True)
        try:
            return fn()
        except Exception as e:
            traceback.print_exc()
            notes.append(
                f"**Attenzione**: la fase «{label}» è fallita con `{type(e).__name__}: {e}`; "
                "le cifre che ne dipendono mancano e quelle delle fasi successive vanno lette con riserva."
            )
            try:
                if gen.num_remaining_jobs():
                    gen.clear_queue()
            except Exception:
                traceback.print_exc()
            return None

    # 1. Expert placement on the bench's own prompts
    adapt_rows = []
    adapt_note = None

    def adapt():
        nonlocal adapt_note
        if args.bench_adapt <= 0:
            adapt_note = "Adattamento disattivato (`--bench-adapt 0`): il piazzamento è quello caricato dal file del server."
            return
        if env.sweep is None or placement_counts(torch, model) is None:
            adapt_note = "Il modello non ha un piazzamento dinamico degli esperti: niente da adattare."
            return
        for i in range(args.bench_adapt):
            r = bench.run(names, False, args.bench_adapt_tokens)
            before = placement_maps(torch, model)
            env.sweep()
            after = placement_maps(torch, model)
            moved = sum(int((a != b).sum()) for a, b in zip(before, after)) // 2
            adapt_rows.append([i + 1, fmt(r["hit"], 3), moved])
            if moved == 0:
                break

    phase("expert placement", adapt)
    text = [
        "Quota delle selezioni di esperti che cadono su esperti residenti in VRAM. Il piazzamento caricato dal "
        f"file del server viene adattato ai prompt della misura: ogni giro genera {args.bench_adapt_tokens} "
        "token per job con tutti i job insieme, poi uno sweep della libreria scambia gli esperti. Dopo l'ultimo "
        "giro il piazzamento resta fermo: tutte le fasi lavorano sugli stessi esperti in VRAM. "
        "Il file di piazzamento del server non viene scritto.",
    ]
    if adapt_rows:
        text.append(table(["giro", "selezioni servite dalla VRAM nel giro", "esperti scambiati dallo sweep che segue"], adapt_rows))
    if adapt_note:
        text.append(adapt_note)
    sections.append("## Piazzamento degli esperti\n\n" + "\n\n".join(text))

    # 2. Warm-up of the shapes the measure uses
    def warmup():
        bench.run(names[:1], False, args.bench_warmup)
        for n in env.n_jobs_list:
            bench.run(names[:n], env.stagger, args.bench_warmup)

    phase("warm-up", warmup)

    # 3. The tokens each prompt gives alone
    solo = {}

    def reference():
        for name in names:
            solo[name] = bench.run(name, False, args.bench_tokens)

    phase("reference text of each prompt", reference)
    ref_tokens = {name: r["tokens"][name] for name, r in solo.items() if r["tokens"].get(name)}

    def speed_cells(runs: dict) -> list:
        """The figures of a set of runs under SPEED_HEADER: medians over the runs."""
        ws = [r["window"] for r in runs.values() if r["window"]]
        acc = sum(r["accepted"] for r in runs.values())
        rej = sum(r["rejected"] for r in runs.values())
        return [
            fmt(median([w["tok_s"] for w in ws])),
            fmt(median([w["ms_step"] for w in ws]), 2),
            fmt(median([w["tok_step"] for w in ws]), 2),
            f"{100 * acc / (acc + rej):.1f}%" if acc + rej else "-",
        ]

    SPEED_HEADER = ["tok/s (mediana)", "ms per passo", "token per passo", "draft accettati"]

    def speed_table(runs: dict) -> str:
        return table(SPEED_HEADER, [speed_cells(runs)] if runs else [])

    # 4. Same tokens at every round
    forced_runs = {}        # (name, round) -> run

    def forced():
        if not ref_tokens:
            raise RuntimeError("no reference tokens")
        for i in range(args.bench_rounds):
            for name in ref_tokens:
                forced_runs[(name, i)] = bench.run(name, False, 0, forced = ref_tokens)

    if "forced" in phases:
        phase("same tokens at every round", forced)
        part = [
            "Ogni prompt viene rigenerato **imponendogli** i token che ha prodotto da solo, una volta per giro "
            f"({args.bench_rounds} giri). Tutti i giri elaborano quindi la stessa sequenza. La finestra misurata va "
            "dal primo all'ultimo token.",
            "### Tempi\n\n" + speed_table(forced_runs),
            "- **draft accettati**: token proposti dalla testa MTP che coincidono col token imposto, su tutti i proposti",
        ]
        ref_runs = {name: forced_runs.get((name, 0)) for name in ref_tokens}
        if args.bench_rounds > 1 and all(ref_runs.values()):
            c = pool([compare_runs(ref_runs[n], forced_runs.get((n, 1)), n) for n in ref_tokens if forced_runs.get((n, 1))])
            q_row = ["-"] * 7 if c is None else [
                c["n"], f"{100 * (1 - c['flips'] / c['n']):.2f}%", f"{c['kl_mean']:.5f}", f"{c['kl_p99']:.4f}",
                f"{c['kl_max']:.3f}", f"x{c['ppl_ratio']:.4f}", "sì" if c["identical"] else "no",
            ]
            part.append(
                "### Il rumore tra due giri\n\n"
                "Il secondo giro di ogni prompt contro il primo (il riferimento), posizioni di tutti i prompt insieme: "
                "stessa configurazione e stessa sequenza, quindi quello che si sposta qui è rumore.\n\n"
                + table(
                    ["posizioni", "stesso token più probabile", "KL media", "KL p99", "KL massima",
                     "perplessità del testo imposto", "probabilità identiche"],
                    [q_row],
                )
                + f"\n\n- **KL**: divergenza KL(riferimento ‖ giro) stimata a ogni posizione sui {TOP_TOKENS} token più "
                "probabili del riferimento più un resto (stima per difetto)\n"
                "- **perplessità del testo imposto**: rapporto tra la perplessità che il giro dà al testo imposto e "
                "quella del riferimento\n"
                "- **probabilità identiche**: la probabilità del token imposto è la stessa, bit per bit, in ogni posizione"
            )
        unforced = sorted({
            f"`{name}`" for (name, _), r in forced_runs.items() if r["tokens"].get(name) != ref_tokens.get(name)
        })
        if unforced:
            part.append("**Attenzione**: i token usciti non sono quelli imposti per " + ", ".join(unforced) + ".")
        sections.append("## Stessi token, giri ripetuti\n\n" + "\n\n".join(part))

    # 5. Free generation, one job
    free_runs = {}          # (name, round) -> run

    def free():
        for i in range(args.bench_rounds):
            for name in names:
                free_runs[(name, i)] = bench.run(name, False, args.bench_tokens)

    if "free" in phases:
        phase("free generation, one job", free)
        sections.append(
            "## Generazione libera, un job\n\n"
            f"Greedy, {args.bench_tokens} token per prompt, {args.bench_rounds} giri: è quello che vede un client. "
            "Il testo cambia da un giro all'altro, e con lui i draft accettati: le differenze piccole tra due "
            "configurazioni qui sono rumore, la tabella a token imposti dice cosa viene dal calcolo.\n\n"
            + speed_table(free_runs)
        )

    # 6. Several jobs together
    if "jobs" in phases:
        for n in env.n_jobs_list:
            multi = {}          # (jobs, round) -> run

            def together(n = n, multi = multi):
                for i in range(args.bench_rounds):
                    multi[(n, i)] = bench.run(names[:n], env.stagger, args.bench_tokens)

            phase(f"{n} jobs together", together)
            sections.append(
                f"## {n} job insieme\n\n"
                f"Greedy, {args.bench_tokens} token per job, {'decode sfalsato' if env.stagger else 'un batch'}; tok/s "
                "sommati sui job, nella finestra in cui generano tutti.\n\n"
                + speed_table(multi)
            )

    # 7. This process's threads on CPUs of their own, or wherever the scheduler puts them
    if "host" in phases and env.host_cpus:
        modes = ["liberi", "fissati"]
        host_runs = {m: {} for m in modes}

        def host():
            try:
                for i in range(args.bench_rounds):
                    for mode in (modes if i % 2 == 0 else modes[::-1]):
                        env.pin(mode == "fissati")
                        for name in names:
                            host_runs[mode][(name, i)] = bench.run(name, False, args.bench_tokens)
            finally:
                env.pin(True)

        phase("host threads free or pinned", host)
        host_rows = []
        for mode in modes:
            runs = host_runs[mode]
            if not runs:
                continue
            # pinned against free: the pairwise ratio of tok/s, same prompt and round
            ratios = []
            if mode == "fissati":
                for key, r in runs.items():
                    b = host_runs["liberi"].get(key)
                    if r["window"] and b is not None and b["window"]:
                        ratios.append(r["window"]["tok_s"] / b["window"]["tok_s"])
            tok_s, *rest = speed_cells(runs)
            host_rows.append([
                "thread " + mode,
                tok_s,
                f"{100 * (median(ratios) - 1):+.1f}%" if ratios else "-",
                f"{sum(1 for x in ratios if x > 1)}/{len(ratios)}" if ratios else "-",
                *rest,
            ])
        sections.append(
            "## Thread di questo processo: liberi o fissati\n\n"
            f"Greedy, {args.bench_tokens} token per prompt, un job. «fissati»: tutti i thread di questo "
            f"processo (quello che emette i passi, quelli del driver CUDA) sulle CPU {env.host_cpus}; «liberi»: su "
            "tutte le CPU, dove li mette lo scheduler. I thread del worker degli esperti restano fissati dalla "
            "libreria ai loro core in entrambi i casi. Stesso processo, stesso piazzamento, giri alternati.\n\n"
            + table(["impostazione", SPEED_HEADER[0], "contro liberi", "giri più veloci", *SPEED_HEADER[1:]], host_rows)
            + "\n\n- **contro liberi**: mediana del rapporto di tok/s tra il giro coi thread fissati e quello coi thread "
            "liberi dello stesso prompt e dello stesso giro; **giri più veloci**: in quanti di quei confronti ha vinto"
        )

    # 8. Ingestion of a document
    if "ingest" in phases and args.bench_ingest_tokens > 0:
        ing_rows = []
        chunks = list(getattr(args, "bench_chunks", None) or [None])

        def ingest():
            for i in range(args.bench_ingest_rounds):
                for chunk in (chunks if i % 2 == 0 else chunks[::-1]):
                    r = env.ingest(i, chunk)
                    ing_rows.append([i + 1, r["chunk"], r["tokens"], fmt(r["seconds"], 2), fmt(r["tokens"] / r["seconds"])])

        phase("ingestion of a document", ingest)
        if ing_rows:
            sections.append(
                "## Ingestione di un documento\n\n"
                "Un job da solo con un documento mai visto (niente in cache): tempo dall'accodamento al primo token. "
                "«pezzo» è la dimensione massima di un passaggio di prefill (`max_chunk_size`).\n\n"
                + table(["giro", "pezzo", "token del prompt", "secondi", "token/s"], ing_rows)
            )

    # 9. One prompt ingested again in passes of another size, its own text forced on it
    cmp_sizes = list(getattr(args, "bench_chunks", None) or [])
    if "chunkcmp" in phases and cmp_sizes and getattr(env, "forget", None) is not None:
        base_chunk = gen.max_chunk_size
        others = [c for c in cmp_sizes if c != base_chunk]
        # an arm is (first pass, the passes after it): a prompt's first pass is the only one that
        # can stay below the sparse attention's threshold, so it is tried at the base size too
        arms_c = [(base_chunk, base_chunk)] + [(c, c) for c in others] + [(base_chunk, c) for c in others if c > base_chunk]
        cmp_name = names[0]
        cmp_runs = []       # (arm, run) in the order run: the first is the reference
        back_runs = []      # the other way round: (arm, run) on the text the first other size wrote

        def arm_name(arm) -> str:
            return f"{arm[0]}" if arm[0] == arm[1] else f"{arm[0]} il primo passaggio, poi {arm[1]}"

        def ingest_run(arm, tokens: int, forced = None) -> dict:
            """The prompt from its first token: the first pass of at most arm[0] tokens, the
            others of arm[1]."""
            if not env.forget():
                raise RuntimeError("this model resumes a prompt from its K/V pages alone: nothing to ingest again")
            had = "iterate" in vars(gen)
            orig = gen.iterate
            calls = [0]

            def staged():
                r = orig()
                calls[0] += 1
                if calls[0] == 1:
                    gen.max_chunk_size = arm[1]
                return r

            gen.max_chunk_size = arm[0]
            gen.iterate = staged
            try:
                t0 = time.perf_counter()
                r = bench.run([cmp_name], False, tokens, forced = forced)
                r["seconds"] = time.perf_counter() - t0
            finally:
                gen.max_chunk_size = base_chunk
                if had:
                    gen.iterate = orig
                else:
                    del gen.iterate
            r["free"] = env.memory() if env.memory is not None else []
            return r

        def chunkcmp():
            if cmp_name not in ref_tokens:
                raise RuntimeError("no reference tokens")
            for arm in arms_c + [arms_c[0]]:
                cmp_runs.append((arm, ingest_run(arm, 0, forced = ref_tokens)))
            if others:
                # the text the other size writes by itself, forced on both sizes: a setting that
                # only moves the numbers loses as much on the other's text as the other on its own
                alt = (others[0], others[0])
                text = ingest_run(alt, args.bench_tokens)["tokens"].get(cmp_name)
                if text:
                    for arm in (alt, arms_c[0]):
                        back_runs.append((arm, ingest_run(arm, 0, forced = {cmp_name: text})))

        phase("same prompt, another chunk size", chunkcmp)

        def cmp_table(runs: list, forced_text) -> str:
            ref = runs[0][1]
            p_ref = ref["probs"].get(cmp_name, [])
            rows_ = []
            for i, (arm, r) in enumerate(runs):
                free = ", ".join(f"{m}" for _, m in r["free"]) or "-"
                if i == 0:
                    rows_.append([f"{arm_name(arm)} (riferimento)", fmt(r["seconds"], 1), len(p_ref), "-", "-", "-", "-", "-", "-", free])
                    continue
                c = compare_runs(ref, r, cmp_name)
                if c is None:
                    rows_.append([arm_name(arm), fmt(r["seconds"], 1), "-", "-", "-", "-", "-", "-", "-", free])
                    continue
                delta = [abs(a - b) for a, b in zip(p_ref, r["probs"].get(cmp_name, []))]
                rows_.append([
                    arm_name(arm) + (" (il rumore)" if arm == runs[0][0] else ""), fmt(r["seconds"], 1), c["n"],
                    f"{100 * (1 - c['flips'] / c['n']):.2f}%", f"{c['kl_mean']:.6f}", f"{c['kl_max']:.4f}",
                    f"{statistics.fmean(delta):.5f} / {max(delta):.4f}", f"x{c['ppl_ratio']:.4f}",
                    "sì" if c["identical"] else "no", free,
                ])
                if r["tokens"].get(cmp_name) != forced_text:
                    notes.append(f"**Attenzione**: nel giro «{arm_name(arm)}» i token usciti non sono quelli imposti.")
            return table(
                ["pezzo", "secondi del giro", "posizioni", "stesso token più probabile", "KL media", "KL massima",
                 "spostamento della probabilità del token imposto (medio / massimo)", "perplessità del testo imposto",
                 "probabilità identiche", "MiB liberi sulle schede dopo il giro"],
                rows_,
            )

        if cmp_runs:
            part = [
                f"Il prompt `{cmp_name}` ({int(env.prompts[cmp_name].shape[-1])} token) viene ingerito di nuovo dal primo token "
                "(i checkpoint dello stato ricorrente vengono tolti prima di ogni giro) con passaggi di prefill di "
                "dimensione diversa, e gli si impongono i token che ha prodotto da solo. Riferimento: il primo giro. "
                "Il giro rifatto con la stessa dimensione del riferimento è il rumore.",
                cmp_table(cmp_runs, ref_tokens.get(cmp_name)),
                "- **secondi del giro**: ingestione del prompt più i token imposti\n"
                "- **MiB liberi**: memoria che il driver dà per libera su ogni scheda; l'allocatore non restituisce "
                "quello che ha preso, quindi scende al primo giro che chiede di più e resta lì",
            ]
            if back_runs:
                part += [
                    "### Il verso opposto\n\n"
                    f"Lo stesso prompt col testo che scrive quando è ingerito a pezzi da {arm_name(back_runs[0][0])}, imposto "
                    "alle due dimensioni. Il testo imposto è sempre un po' meno probabile per chi non l'ha scritto: se "
                    "nei due versi la perplessità sale di quanto sale qui, le due dimensioni spostano i numeri senza "
                    "che una sia peggiore dell'altra.",
                    cmp_table(back_runs, back_runs[0][1]["tokens"].get(cmp_name)),
                ]
            sections.append("## Stesso prompt, pezzi di prefill diversi\n\n" + "\n\n".join(part))

    # 10. The checkpoint that follows a generating job: its cost, and the next turn's gain
    if "turn" in phases and getattr(env, "set_tail_ckpt", None) is not None and getattr(env, "turn", None) is not None:
        t_name = names[0]
        t_tokens = int(getattr(args, "bench_turn_tokens", 0) or 1100)
        page = env.page
        cost_runs = {0: {}, page: {}}       # tail interval -> round -> run
        cost_stats = {0: [], page: []}
        next_rows = []

        def tail_run(tail: int, forced: dict) -> dict:
            env.set_tail_ckpt(tail)
            s0 = env.ckpt_stats()
            try:
                r = bench.run([t_name], False, 0, forced = forced)
            finally:
                env.set_tail_ckpt(0)
            s1 = env.ckpt_stats()
            r["tail_stats"] = {k: s1[k] - s0[k] for k in s1}
            return r

        def turn():
            env.set_tail_ckpt(0)
            text = bench.run([t_name], False, t_tokens)["tokens"].get(t_name)
            if not text:
                raise RuntimeError("no reference tokens")
            forced = {t_name: text}
            for i in range(args.bench_rounds):
                for tail in ((0, page) if i % 2 == 0 else (page, 0)):
                    r = tail_run(tail, forced)
                    cost_runs[tail][i] = r
                    cost_stats[tail].append(r["tail_stats"])
            # The next turn: the same prompt ingested again from its first token, the same answer
            # generated, then a request that holds prompt, answer and a few tokens more
            ids = env.torch.cat([env.prompts[t_name], env.torch.tensor([text], dtype = env.prompts[t_name].dtype), env.suffix], dim = -1)
            for i in range(args.bench_rounds):
                for tail in ((0, page) if i % 2 == 0 else (page, 0)):
                    if not env.forget():
                        raise RuntimeError("this model resumes a prompt from its K/V pages alone: no checkpoint to follow an answer")
                    tail_run(tail, forced)
                    env.set_tail_ckpt(tail)
                    try:
                        t = env.turn(ids, 16)
                    finally:
                        env.set_tail_ckpt(0)
                    next_rows.append([
                        i + 1, "sì" if tail else "no", t["prompt"], t["resumed"] if t["resumed"] is not None else "-",
                        t["prompt"] - t["resumed"] if t["resumed"] is not None else "-", fmt(t["ttft"], 2),
                    ])

        phase("the checkpoint that follows an answer", turn)
        part = [
            f"Il prompt `{t_name}` scrive {t_tokens} token, poi quel testo gli viene imposto con e senza il checkpoint "
            f"che segue il job mentre genera (uno stash dello stato ricorrente a ogni pagina di {page} token, al "
            "posto di quello della pagina prima), a giri alternati.",
        ]
        c_rows = []
        for tail in (0, page):
            runs_ = cost_runs[tail]
            if not runs_:
                continue
            ws = [r["window"] for r in runs_.values() if r["window"]]
            ratios = [r["window"]["tok_s"] / cost_runs[0][i]["window"]["tok_s"] for i, r in runs_.items()
                      if tail and r["window"] and cost_runs[0].get(i) and cost_runs[0][i]["window"]]
            st = cost_stats[tail]
            n_st = sum(x["stashes"] for x in st)
            acc = sum(r["accepted"] for r in runs_.values())
            rej = sum(r["rejected"] for r in runs_.values())
            same = "-"
            if tail and cost_runs[0].get(0) and runs_.get(0):
                c = compare_runs(cost_runs[0][0], runs_[0], t_name)
                same = "-" if c is None else ("sì" if c["identical"] else f"no (KL media {c['kl_mean']:.6f})")
            c_rows.append([
                "con" if tail else "senza", fmt(median([w["tok_s"] for w in ws])),
                f"{100 * (median(ratios) - 1):+.1f}%" if ratios else "-",
                fmt(median([w["ms_step"] for w in ws]), 2), f"{100 * acc / (acc + rej):.1f}%" if acc + rej else "-",
                fmt(n_st / len(st), 1) if st else "-", fmt(1000 * sum(x["t_stash"] for x in st) / n_st, 1) if n_st else "-",
                fmt(sum(x["replaced"] for x in st) / len(st), 1) if st else "-", same,
            ])
        if c_rows:
            part.append("### Quanto costa a chi genera\n\n" + table(
                ["checkpoint di coda", "tok/s (mediana)", "contro senza", "ms per passo", "draft accettati",
                 "stash per giro", "ms per stash", "checkpoint sostituiti per giro", "probabilità identiche a senza"],
                c_rows,
            ) + "\n\n- **ms per stash**: tempo dentro la chiamata che copia lo stato in RAM, attesa del passo in corso compresa")
        if next_rows:
            part.append(
                "### Cosa risparmia il turno dopo\n\n"
                "Prima il prompt viene ingerito da capo e la risposta rigenerata (stesso testo), con o senza il "
                "checkpoint di coda; poi arriva una richiesta che contiene prompt, risposta e pochi token nuovi, "
                "come il turno successivo di una conversazione. «ripresi» è la posizione da cui il job riparte.\n\n"
                + table(["giro", "checkpoint di coda", "token del prompt", "ripresi", "da rileggere", "secondi al primo token"], next_rows)
            )
        sections.append("## Il checkpoint che segue la risposta\n\n" + "\n\n".join(part))

    mem = env.memory() if env.memory is not None else []
    head = [
        "# Misura di una configurazione sul modello",
        "",
        f"- banco: `{os.path.basename(__file__)}` al commit {env.revision}",
        f"- libreria: exllamav3 {env.version} da `{env.lib_path}`, torch {env.torch_version}",
        f"- estensione: `{env.ext_path}`",
        f"- modello: `{env.model_path}`",
        f"- schede: {', '.join(env.devices) or 'nessuna'}",
        f"- bozze: {env.draft_desc}; thread del worker: {env.worker_threads or 'sconosciuti'}",
        f"- prompt: {env.prompt_desc or 'brevi (' + ', '.join(str(int(env.prompts[n].shape[-1])) for n in names) + ' token)'}",
    ]
    if env.host_cpus:
        head.append(f"- thread di questo processo fissati sulle CPU {env.host_cpus} dopo il caricamento ({env.host_pinned} thread)")
    if mem:
        head.append("- memoria libera sulle schede a fine misura: " + ", ".join(f"{d} {m} MiB" for d, m in mem))
    head.append("")
    if notes:
        head += notes + [""]
    return "\n".join(head) + "\n" + "\n\n".join(sections) + "\n"


# --------------------------------------------------------------------------------------------
#   The real thing
# --------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(add_help = False)
    ap.add_argument("--bench-tokens", type = int, default = 400, help = "token generati da ogni job (default 400)")
    ap.add_argument("--bench-rounds", type = int, default = 2, help = "giri di ogni misura (default 2)")
    ap.add_argument("--bench-jobs", default = "3", help = "numeri di job insieme da provare (default 3; vuoto: nessuno)")
    ap.add_argument("--bench-stagger", default = "auto",
                    help = "decode sfalsato nei giri con più job: auto (come EXL3_STAGGER_DECODE), on, off")
    ap.add_argument("--bench-warmup", type = int, default = 48, help = "token dei giri di riscaldamento (default 48)")
    ap.add_argument("--bench-adapt", type = int, default = 8,
                    help = "giri al massimo per adattare il piazzamento degli esperti ai prompt della misura (default 8, 0 = no)")
    ap.add_argument("--bench-adapt-tokens", type = int, default = 160, help = "token per job in un giro di adattamento (default 160)")
    ap.add_argument("--bench-context", type = int, default = 0,
                    help = "token di documento messi davanti a ogni domanda (default 0: prompt brevi)")
    ap.add_argument("--bench-ingest-tokens", type = int, default = 0,
                    help = "token di un documento da ingerire per misurare il prefill (default 0: non misurato)")
    ap.add_argument("--bench-ingest-rounds", type = int, default = 2, help = "documenti ingeriti per dimensione del pezzo (default 2)")
    ap.add_argument("--bench-chunks", default = "",
                    help = "dimensioni del pezzo di prefill da provare nell'ingestione e nel confronto chunkcmp, "
                           "es. 2048,4096 (default: quella del server)")
    ap.add_argument("--bench-turn-tokens", type = int, default = 1100,
                    help = "token della risposta lunga nella fase turn (default 1100)")
    ap.add_argument("--bench-prompts", type = int, default = 0,
                    help = "quanti dei prompt del banco usare (default 0: tutti)")
    ap.add_argument("--bench-host-cpus", default = "",
                    help = "CPU logiche su cui fissare i thread di questo processo dopo il caricamento, es. 10,11,22,23 "
                           "(default: nessuna); la fase host li confronta con i thread lasciati liberi")
    ap.add_argument("--bench-phases", default = ",".join(PHASES),
                    help = "fasi: forced (token imposti), free (generazione libera), jobs (più job), host, ingest, "
                           "chunkcmp (stesso prompt ingerito con pezzi diversi, serve --bench-chunks), turn (checkpoint che "
                           "segue la risposta); default tutte")
    ap.add_argument("--bench-out", default = None, help = "file .md del rapporto")
    ap.add_argument("--bench-server-repo", default = os.getcwd(), help = "repository del server (default: directory corrente)")
    ap.add_argument("--bench-help", action = "store_true")
    args, server_argv = ap.parse_known_args()
    if args.bench_help:
        ap.print_help()
        print(__doc__)
        return 0
    if args.bench_out is not None and not args.bench_out.endswith(".md"):
        raise SystemExit("--bench-out: il rapporto è un file .md")
    args.bench_phases = [p.strip() for p in args.bench_phases.split(",") if p.strip()]
    unknown = [p for p in args.bench_phases if p not in PHASES]
    if unknown:
        raise SystemExit(f"--bench-phases: fasi sconosciute {unknown}; valgono {', '.join(PHASES)}")
    host_cpus = [int(x) for x in args.bench_host_cpus.split(",") if x.strip()]
    args.bench_chunks = [int(x) for x in args.bench_chunks.split(",") if x.strip()]

    # The server's package first, then this checkout's library ahead of an installed one
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.abspath(args.bench_server_repo))
    import logging
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch
    import exllamav3
    from exllamav3 import Generator, Job
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.generator.sampler import GreedySampler
    from exllamav3.modules import block_sparse_mlp_cpu
    from exllamav3.version import __version__ as exl3_version
    from openai.config import parse_args
    from openai.model_container import ModelContainer

    revision = bench_revision()
    print(f" -- bench: {os.path.basename(__file__)} at commit {revision}; phases: {', '.join(args.bench_phases)}", flush = True)
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
    stagger = {"on": True, "off": False}.get(args.bench_stagger, bool(getattr(gen, "stagger_decode", False)))

    all_cpus = sorted(os.sched_getaffinity(0))
    host_pinned = pin_host_threads(host_cpus) if host_cpus else 0
    if host_cpus:
        print(f" -- bench: {host_pinned} threads of this process pinned to CPUs {host_cpus}", flush = True)

    max_jobs = min(len(PROMPTS), args.bench_prompts) if args.bench_prompts > 0 else len(PROMPTS)
    n_jobs_list = sorted({int(x) for x in args.bench_jobs.split(",") if x.strip()}) if "jobs" in args.bench_phases else []
    n_jobs_list = [n for n in n_jobs_list if 2 <= n <= min(gen.max_batch_size, max_jobs)]

    # The library's own sweep, past the wrapper the server's expert tier puts around it (the
    # tier saves the placement file after every sweep it sees); from here on nothing sweeps
    # unless the bench asks
    lib_sweep = block_sparse_mlp_cpu.run_pending_swap_sweeps
    lib_sweep = getattr(lib_sweep, "_exl3srv_orig", lib_sweep)
    block_sparse_mlp_cpu.run_pending_swap_sweeps = lambda infer_params: None

    def sweep():
        ip = model.config.infer_params
        ip.moe_cpu_swap_pending = True
        lib_sweep(ip)

    def encode(text: str):
        chat = getattr(model, "default_chat_prompt", None)
        if chat is not None:
            text = chat(text)
        return tokenizer.encode(text, encode_special_tokens = True)

    def count_tokens(text: str) -> int:
        return tokenizer.encode(text).shape[-1]

    def prompt_text(i: int) -> str:
        if args.bench_context <= 0:
            return PROMPTS[i]
        return long_prompt(count_tokens, PROMPTS[i], NAMES[i], args.bench_context)

    prompts = {NAMES[i]: encode(prompt_text(i)) for i in range(max_jobs)}
    sizes = [int(p.shape[-1]) for p in prompts.values()]
    prompt_desc = None
    if args.bench_context > 0:
        prompt_desc = f"da {min(sizes)} a {max(sizes)} token per job (documento davanti alla domanda, `--bench-context {args.bench_context}`)"

    def ingest(i: int, chunk = None) -> dict:
        """One job alone on a document nothing has seen: seconds from enqueue to its first
        token. `chunk`: the generator's largest prefill pass for this document."""
        doc = encode(long_prompt(count_tokens, PROMPTS[1], f"ingest-{os.getpid()}-{i}-{chunk}", args.bench_ingest_tokens))
        job = Job(input_ids = doc, max_new_tokens = 8, sampler = GreedySampler(), identifier = f"ingest-{i}-{chunk}")
        before = gen.max_chunk_size
        if chunk:
            gen.max_chunk_size = chunk
        try:
            t0 = time.perf_counter()
            gen.enqueue(job)
            t1 = None
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("stage") == "error":
                        raise RuntimeError(f"ingestion failed: {r.get('error')!r}")
                    if t1 is None and r.get("stage") == "streaming" and r.get("token_ids") is not None:
                        t1 = time.perf_counter()
        finally:
            gen.max_chunk_size = before
            if gen.num_remaining_jobs():
                gen.clear_queue()
        if t1 is None:
            raise RuntimeError("the ingestion gave no token")
        return {"tokens": int(doc.shape[-1]), "seconds": t1 - t0, "chunk": chunk or before}

    def forget() -> bool:
        """Make the next run ingest its prompts from their first token (as prefill_bench does):
        a recurrent model resumes a cached prompt only from a page with a stashed state
        checkpoint, so every checkpoint is dropped through the library's own prune, with a page
        table that calls them all stranded. False for a model without recurrent states."""
        assert not gen.num_remaining_jobs()
        rc = getattr(gen, "recurrent_cache", None)
        if rc is None:
            return False
        real = rc.pagetable
        rc.pagetable = SimpleNamespace(is_resumable = lambda key: False)
        try:
            rc.prune_stranded()
        finally:
            rc.pagetable = real
        return True

    def set_tail(tokens: int):
        gen.tail_checkpoint_interval = int(tokens) if gen.recurrent_cache is not None else 0

    def turn(ids, max_new: int) -> dict:
        """One job on the given tokens: where its state resumed from and the seconds to its
        first token."""
        job = Job(input_ids = ids, max_new_tokens = max_new, sampler = GreedySampler(), identifier = "turn")
        t0 = time.perf_counter()
        gen.enqueue(job)
        t1, resumed = None, None
        try:
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("stage") == "error":
                        raise RuntimeError(f"the turn failed: {r.get('error')!r}")
                    if r.get("stage") == "started" and resumed is None:
                        resumed = int(getattr(job, "cached_pages", 0)) * PAGE_SIZE
                    if t1 is None and r.get("stage") == "streaming" and r.get("token_ids") is not None:
                        t1 = time.perf_counter()
        finally:
            if gen.num_remaining_jobs():
                gen.clear_queue()
        if t1 is None:
            raise RuntimeError("the turn gave no token")
        return {"prompt": int(ids.shape[-1]), "resumed": resumed, "ttft": t1 - t0}

    def memory() -> list:
        out = []
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            out.append((f"cuda:{i}", free // (1 << 20)))
        return out

    hosts = list(getattr(model.config, "moe_cpu_hosts", {}).values())

    env = SimpleNamespace(
        torch = torch,
        gen = gen,
        model = model,
        Job = Job,
        sampler_cls = GreedySampler,
        prompts = prompts,
        prompt_desc = prompt_desc,
        max_jobs = max_jobs,
        n_jobs_list = n_jobs_list,
        stagger = stagger,
        sweep = sweep,
        ingest = ingest,
        forget = forget,
        set_tail_ckpt = set_tail if hasattr(gen, "tail_stats") else None,
        ckpt_stats = lambda: dict(gen.tail_stats),
        turn = turn,
        suffix = tokenizer.encode("\n\nContinue from there, in two sentences."),
        page = PAGE_SIZE,
        memory = memory if torch.cuda.is_available() else None,
        host_cpus = host_cpus,
        host_pinned = host_pinned,
        pin = lambda on: pin_host_threads(host_cpus if on else all_cpus),
        worker_threads = getattr(hosts[0], "threads", None) if hosts else None,
        version = exl3_version,
        revision = revision,
        lib_path = os.path.dirname(exllamav3.__file__),
        ext_path = getattr(ext, "__file__", "?"),
        torch_version = torch.__version__,
        model_path = cfg.model_path,
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        draft_desc = f"MTP, {gen.num_draft_tokens} token" if draft_model is not None else "nessuna",
    )

    text = measure(env, args)

    out = args.bench_out
    if out is None:
        folder = "prof-runs" if os.path.isdir("prof-runs") else "."
        out = os.path.join(folder, time.strftime("config-bench-%Y%m%d-%H%M%S.md"))
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
