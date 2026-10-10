#!/usr/bin/env python3
"""
Misura dell'ingestione dei prompt su un secondo thread (Generator.prefill_thread) sul modello vero.

Un caso solo, completo: il job `a` genera mentre il job `b` ingerisce un documento. Il banco
dice quanti token al secondo riceve `a` in quella finestra e quanto dura l'ingestione di `b`,
nei due modi (come oggi: un passo di decode per ogni pezzo di prefill; col thread: il decode
continua accanto al pezzo), contro ciascuno dei due da solo. Poi controlla che il calcolo sia
lo stesso: con token imposti, le probabilità di `a` accanto a un prefill contro `a` da solo, e
quelle di `b` dopo un prompt ingerito accanto a un decode contro lo stesso prompt ingerito come
oggi.

Si lancia a server FERMO, dalla directory del server, con gli stessi argomenti del server (il
modello viene caricato come lo carica lui):

    python /percorso/exllamav3/experiments/prefill_bench.py --bench-out prof-runs/prefill.md -- \\
        --model-dir ... (gli argomenti di exllama-server.sh)

Gli argomenti del banco iniziano per --bench-; tutto il resto va al parser del server.
Il file di piazzamento degli esperti del server non viene scritto.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stagger_bench import (     # noqa: E402
    PROMPTS, GpuSampler, bench_revision, compare_forced, fmt, hit_rate, long_prompt,
    placement_counts, placement_maps, table,
)

DECODER = "a"
READER = "b"


# --------------------------------------------------------------------------------------------
#   One run
# --------------------------------------------------------------------------------------------

class Runner:
    """A loaded model and its generator; run() is one scripted generation."""

    def __init__(self, env):
        self.env = env

    def reset_cache(self):
        """Forget every cached prompt page, so that the next run ingests its prompt again."""
        gen = self.env.gen
        assert not gen.num_remaining_jobs()
        gen.pagetable.reset_page_table()
        rc = getattr(gen, "recurrent_cache", None)
        if rc is not None:
            rc.prune_stranded()

    def run(self, thread: bool, decoder: dict | None, reader: dict | None, lead: int = 0, reset: bool = True):
        """
        decoder / reader: dict(ids, max_new, forced = None). The decoder starts at once; the
        reader is enqueued once the decoder has generated `lead` tokens (at once without a
        decoder). The run ends when the reader is done, or the decoder when there is no reader;
        a decoder still generating then is cancelled. Returns what happened, per iteration.
        """
        env = self.env
        torch, gen = env.torch, env.gen
        if reset:
            self.reset_cache()
        gen.prefill_thread = thread
        side0 = dict(gen.side_stats)
        counts0 = placement_counts(torch, env.model)
        for d in env.device_ids:
            torch.cuda.reset_peak_memory_stats(d)

        specs = {}
        if decoder is not None:
            specs[DECODER] = decoder
        if reader is not None:
            specs[READER] = reader
        jobs, by_job = {}, {}

        def make(name):
            spec = specs[name]
            kwargs = {}
            n_new = spec["max_new"]
            if spec.get("forced") is not None:
                n_new = len(spec["forced"])
                kwargs = {"return_probs": True, "return_top_tokens": 2}
            job = env.Job(
                input_ids = spec["ids"],
                max_new_tokens = n_new,
                sampler = env.sampler_cls(),
                identifier = name,
                **kwargs,
            )
            jobs[name] = job
            by_job[id(job)] = name
            gen.enqueue(job)
            if spec.get("forced") is not None:
                job.constrain_output_now(torch.tensor([spec["forced"]], dtype = torch.long))
            return time.perf_counter()

        rec = {
            "thread": thread,
            "tokens": {n: [] for n in specs},
            "probs": {n: [] for n in specs},
            "top": {n: [] for n in specs},
            "enqueued": {},
            "first_token": {},
            "prefill": {n: [] for n in specs},      # (t, tokens ingested so far, tokens to ingest)
            "timeline": [],                         # (t, tokens generated so far per job)
            "prompt_tokens": {n: int(specs[n]["ids"].shape[-1]) for n in specs},
            "cancelled": [],
        }
        if decoder is not None:
            rec["enqueued"][DECODER] = make(DECODER)
        elif reader is not None:
            rec["enqueued"][READER] = make(READER)
        done = set()
        last = READER if reader is not None else DECODER

        while gen.num_remaining_jobs():
            if reader is not None and READER not in jobs and len(rec["tokens"][DECODER]) >= lead:
                rec["enqueued"][READER] = make(READER)
            results = gen.iterate()
            now = time.perf_counter()
            for r in results:
                name = by_job.get(id(r.get("job")))
                if name is None:
                    continue
                stage = r.get("stage")
                if stage == "error":
                    raise RuntimeError(f"job {name} failed: {r.get('error')!r}")
                if stage == "prefill":
                    rec["prefill"][name].append((now, r.get("curr_progress"), r.get("max_progress")))
                if stage != "streaming":
                    continue
                ids = r.get("token_ids")
                if ids is not None and ids.numel():
                    rec["first_token"].setdefault(name, now)
                    rec["tokens"][name] += ids.reshape(-1).tolist()
                p = r.get("token_probs")
                if p is not None:
                    rec["probs"][name] += p.reshape(-1).tolist()
                kt, kp = r.get("top_k_tokens"), r.get("top_k_probs")
                if kt is not None and kp is not None:
                    rec["top"][name] += list(zip(
                        kt.reshape(-1, kt.shape[-1]).tolist(),
                        kp.reshape(-1, kp.shape[-1]).tolist(),
                    ))
                if r.get("eos"):
                    done.add(name)
            rec["timeline"].append((now, {n: len(rec["tokens"][n]) for n in specs}))
            if last in done:
                for name, job in jobs.items():
                    if name not in done:
                        gen.cancel(job)
                        rec["cancelled"].append(name)
                break
            if reader is not None and READER not in jobs and DECODER in done:
                raise RuntimeError(f"the decoder ended after {len(rec['tokens'][DECODER])} tokens, before its lead of {lead}")

        rec["side"] = {k: gen.side_stats[k] - side0[k] for k in side0}
        rec["hit"] = hit_rate(counts0, placement_counts(torch, env.model))
        rec["accepted"] = {n: j.accepted_draft_tokens for n, j in jobs.items()}
        rec["rejected"] = {n: j.rejected_draft_tokens for n, j in jobs.items()}
        rec["peak"] = {d: torch.cuda.max_memory_allocated(d) for d in env.device_ids}
        rec["free"] = {d: torch.cuda.mem_get_info(d)[0] for d in env.device_ids}
        return rec


# --------------------------------------------------------------------------------------------
#   Reading a run
# --------------------------------------------------------------------------------------------

def tokens_at(rec: dict, name: str, t: float) -> int:
    """Tokens `name` had generated by time t (after the last iteration that ended by then)."""
    n = 0
    for when, toks in rec["timeline"]:
        if when > t:
            break
        n = toks.get(name, 0)
    return n


def tokens_before(rec: dict, name: str, t: float) -> int:
    """Tokens `name` had generated before the iteration that ended at time t."""
    n = 0
    for when, toks in rec["timeline"]:
        if when >= t:
            break
        n = toks.get(name, 0)
    return n


def ingestion(rec: dict, name: str = READER):
    """The window in which `name` ingested its prompt: from its enqueue to its first token.
    Returns dict(t0, t1, seconds, prompt_tokens, tok_s, chunks) or None."""
    t0, t1 = rec["enqueued"].get(name), rec["first_token"].get(name)
    if t0 is None or t1 is None or t1 <= t0:
        return None
    n = rec["prompt_tokens"][name]
    return {
        "t0": t0, "t1": t1, "seconds": t1 - t0, "prompt_tokens": n, "tok_s": n / (t1 - t0),
        "chunks": len(rec["prefill"][name]),
    }


def decode_rate(rec: dict, name: str, t0: float, t1: float):
    """Tokens per second `name` generated between two times, and the tokens."""
    if t1 <= t0:
        return None, 0
    n = tokens_at(rec, name, t1) - tokens_at(rec, name, t0)
    return n / (t1 - t0), n


def whole_rate(rec: dict, name: str):
    """Tokens per second of `name` from its first token to its last, alone in the run."""
    stamps = [(t, toks.get(name, 0)) for t, toks in rec["timeline"] if toks.get(name, 0) > 0]
    if len(stamps) < 5:
        return None
    (t_a, n_a), (t_b, n_b) = stamps[0], stamps[-1]
    return (n_b - n_a) / max(t_b - t_a, 1e-9)


def cut(rec: dict, name: str, n: int) -> dict:
    """The run with the tokens of `name` cut to the first n positions (compare_forced)."""
    out = dict(rec)
    for key in ("tokens", "probs", "top"):
        out[key] = dict(rec[key])
        out[key][name] = rec[key][name][:n]
    return out


def gib(x):
    return "-" if x is None else f"{x / 1024 ** 3:.2f}"


def check_row(label: str, c, extra: str = "") -> list:
    if c is None:
        return [label, "-", "-", "-", "-", "non registrato" + extra]
    same = c["first_delta"] is None
    return [
        label, c["n"],
        "nessuna" if same else c["first_delta"],
        "0" if same else f"{c['max']:.2e}",
        c["flips"],
        ("**identico**" if same else "**diverso**") + extra,
    ]


# --------------------------------------------------------------------------------------------
#   The measure
# --------------------------------------------------------------------------------------------

def measure(env, args, gpu = None) -> str:
    torch, gen = env.torch, env.gen
    run = Runner(env)
    notes = []
    sections = []
    a_ids, b_ids = env.prompts[DECODER], env.prompts[READER]
    long_a = dict(ids = a_ids, max_new = args.bench_decoder_tokens)
    short_b = dict(ids = b_ids, max_new = args.bench_reader_tokens)

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

    def gpu_mean(w):
        return gpu.mean(w["t0"], w["t1"]) if gpu is not None and w is not None else None

    # 1. Expert placement on the bench's own prompts (thread off: the lanes are not made yet)
    adapt_rows = []

    def adapt():
        if args.bench_adapt <= 0 or env.sweep is None or placement_counts(torch, env.model) is None:
            return
        for i in range(args.bench_adapt):
            r = run.run(False, dict(ids = a_ids, max_new = args.bench_adapt_tokens + args.bench_lead), short_b, args.bench_lead)
            before = placement_maps(torch, env.model)
            env.sweep()
            after = placement_maps(torch, env.model)
            moved = sum(int((x != y).sum()) for x, y in zip(before, after)) // 2
            adapt_rows.append([i + 1, fmt(r["hit"], 3), moved])
            if moved == 0:
                break

    phase("expert placement", adapt)
    text = (
        "Quota delle selezioni di esperti servite dalla VRAM in un giro (decode di `a` più ingestione di `b`, "
        "come oggi), e scambi dello sweep della libreria che lo segue. Dopo l'ultimo giro il piazzamento resta "
        "fermo. Il file di piazzamento del server non viene scritto."
    )
    sections.append("## Piazzamento degli esperti\n\n" + text + "\n\n" + (
        table(["giro", "selezioni servite dalla VRAM", "esperti scambiati dallo sweep che segue"], adapt_rows)
        if adapt_rows else "Nessun adattamento (`--bench-adapt 0` o modello senza piazzamento dinamico)."
    ))

    # 2. As today, with the thread off; nothing has made the lanes yet, so the CPU expert
    #    worker's slots are all in one rotation, as in production
    base = {}

    def baselines():
        run.run(False, dict(ids = a_ids, max_new = args.bench_warmup), None)
        base["decode"] = run.run(False, dict(ids = a_ids, max_new = args.bench_tokens), None)
        base["prefill"] = run.run(False, None, short_b)
        base["mix"] = run.run(False, long_a, short_b, args.bench_lead)

    phase("as today (thread off)", baselines)

    # 3. References of the same-computation check, still with the thread off
    refs = {}

    def references():
        tok_a = base["decode"]["tokens"][DECODER]
        tok_b = base["mix"]["tokens"][READER]
        if not tok_a or not tok_b:
            raise RuntimeError("no tokens to force (a baseline run is missing)")
        refs["tok_a"], refs["tok_b"] = tok_a, tok_b
        refs["a1"] = run.run(False, dict(ids = a_ids, forced = tok_a, max_new = 0), None)
        refs["a2"] = run.run(False, dict(ids = a_ids, forced = tok_a, max_new = 0), None)
        forced_b = dict(ids = b_ids, forced = tok_b, max_new = 0)
        refs["b1"] = run.run(False, long_a, forced_b, args.bench_lead)
        refs["b2"] = run.run(False, long_a, forced_b, args.bench_lead)

    if "check" in args.bench_phases:
        phase("references of the check (thread off)", references)

    # 4. With the thread
    side = {}

    def threaded():
        run.run(True, None, dict(ids = b_ids, max_new = 8))         # the thread's own first pass
        side["prefill"] = run.run(True, None, short_b)
        side["mix"] = run.run(True, long_a, short_b, args.bench_lead)

    phase("with the prefill thread", threaded)

    checks = {}

    def check():
        checks["a"] = run.run(True, dict(ids = a_ids, forced = refs["tok_a"], max_new = 0), short_b, args.bench_lead)
        checks["b"] = run.run(True, long_a, dict(ids = b_ids, forced = refs["tok_b"], max_new = 0), args.bench_lead)

    if "check" in args.bench_phases and refs.get("b2") is not None:
        phase("same computation (thread on)", check)

    # ---------------------------------------------------------------------------------------
    #   Report
    # ---------------------------------------------------------------------------------------
    solo_rate = whole_rate(base["decode"], DECODER) if base.get("decode") else None
    w_alone = ingestion(base["prefill"]) if base.get("prefill") else None
    w_alone_t = ingestion(side["prefill"]) if side.get("prefill") else None
    w_today = ingestion(base["mix"]) if base.get("mix") else None
    w_side = ingestion(side["mix"]) if side.get("mix") else None

    def beside(rec, w):
        if rec is None or w is None:
            return None, 0
        return decode_rate(rec, DECODER, w["t0"], w["t1"])

    r_today, n_today = beside(base.get("mix"), w_today)
    r_side, n_side = beside(side.get("mix"), w_side)

    def pct(x, ref):
        return "-" if x is None or not ref else f"{100 * x / ref:.0f}%"

    summary = [
        ["`a` da solo", fmt(solo_rate), "100%", "-", "-", "-"],
        ["`b` da solo, come oggi", "-", "-", fmt(w_alone and w_alone["tok_s"], 0), fmt(w_alone and w_alone["seconds"]), "100%"],
        ["`b` da solo, sul thread", "-", "-", fmt(w_alone_t and w_alone_t["tok_s"], 0), fmt(w_alone_t and w_alone_t["seconds"]),
         pct(w_alone_t and w_alone_t["tok_s"], w_alone and w_alone["tok_s"])],
        ["insieme, come oggi", fmt(r_today), pct(r_today, solo_rate), fmt(w_today and w_today["tok_s"], 0),
         fmt(w_today and w_today["seconds"]), pct(w_today and w_today["tok_s"], w_alone and w_alone["tok_s"])],
        ["insieme, col thread", fmt(r_side), pct(r_side, solo_rate), fmt(w_side and w_side["tok_s"], 0),
         fmt(w_side and w_side["seconds"]), pct(w_side and w_side["tok_s"], w_alone and w_alone["tok_s"])],
    ]
    summary_header = [
        "", "`a`: tok/s (nei giri insieme: mentre `b` ingerisce)", "del suo da solo", "`b`: token di prompt al secondo",
        "secondi fino al primo token", "del suo da solo",
    ]

    detail = []
    for label, rec, w, n in (("come oggi", base.get("mix"), w_today, n_today), ("col thread", side.get("mix"), w_side, n_side)):
        if rec is None or w is None:
            continue
        g = gpu_mean(w)
        s = rec["side"]
        detail.append([
            label, w["chunks"], n, fmt(1000 * w["seconds"] / max(n, 1), 1) if n else "-",
            s["beside"], s["decode_steps"], fmt(s["t_chunks"]), fmt(rec["hit"], 3),
            " + ".join(f"{u:.0f}%" for u, _, _ in g) if g else "-", fmt(sum(w_ for _, w_, _ in g), 0) if g else "-",
        ])
    if detail:
        sections.append(
            "## La finestra di ingestione\n\nDall'accodamento di `b` al suo primo token, con `a` che genera. «Come oggi» è "
            "la libreria senza il thread: un passo di decode dopo ogni pezzo di prompt. «Col thread» i passaggi del pezzo "
            "girano sul secondo thread e il decode continua accanto.\n\n" + table(
                ["modo", "pezzi di prompt", "token di `a` nella finestra", "ms per token di `a`", "pezzi con decode accanto",
                 "passi di decode accanto ai pezzi", "secondi di passaggi sul thread", "selezioni servite dalla VRAM",
                 "uso delle schede", "W"], detail,
            )
        )

    if "check" in args.bench_phases:
        rows = []
        if refs.get("a2") is not None:
            rows.append(check_row("`a` da solo, ripetuto (rumore di fondo)", compare_forced(refs["a1"], refs["a2"], DECODER)))
        if checks.get("a") is not None:
            w = ingestion(checks["a"])
            # `b` joins the decode batch in the iteration that gives its first token
            inside = tokens_before(checks["a"], DECODER, w["t1"]) if w else 0
            started = tokens_at(checks["a"], DECODER, w["t0"]) if w else 0
            if inside > started:
                c = compare_forced(cut(refs["a1"], DECODER, inside), cut(checks["a"], DECODER, inside), DECODER)
                rows.append(check_row(
                    "`a` accanto all'ingestione di `b`, contro `a` da solo", c,
                    f" (posizioni {started}-{inside} generate durante l'ingestione)",
                ))
            else:
                notes.append("**Attenzione**: nel giro di controllo `a` non ha generato durante l'ingestione di `b`: "
                             "il confronto su `a` manca (alza `--bench-tokens` o `--bench-prompt-tokens`).")
        if refs.get("b2") is not None:
            rows.append(check_row("`b` ingerito come oggi, ripetuto (rumore di fondo)", compare_forced(refs["b1"], refs["b2"], READER)))
        if checks.get("b") is not None:
            rows.append(check_row("`b` ingerito sul thread accanto ad `a`, contro `b` ingerito come oggi",
                                  compare_forced(refs["b1"], checks["b"], READER)))
        sections.append(
            "## Stesso calcolo\n\n"
            "Token imposti, probabilità del token imposto posizione per posizione. Per `a` il riferimento è `a` da "
            "solo e si confrontano le posizioni generate prima che `b` entri nel batch (da lì il batch è di due job e "
            "i numeri cambiano comunque). Per `b` i token sono quelli che ha dato nel giro «insieme, come oggi», "
            "e in tutti i giri `b` genera accanto ad `a`. Un confronto vale solo se la riga di rumore di fondo "
            "corrispondente è identica.\n\n" + table(
                ["confronto", "posizioni", "prima posizione diversa", "scarto massimo di probabilità",
                 "posizioni col token più probabile cambiato", "esito"], rows,
            )
        )

    mem = []
    for label, rec in (("`b` da solo, come oggi", base.get("prefill")), ("insieme, come oggi", base.get("mix")),
                       ("`b` da solo, sul thread", side.get("prefill")), ("insieme, col thread", side.get("mix"))):
        if rec is not None:
            mem.append([label] + [gib(rec["peak"][d]) for d in env.device_ids] + [gib(rec["free"][d]) for d in env.device_ids])
    if mem:
        sections.append(
            "## Memoria delle schede\n\nPicco di memoria allocata da PyTorch durante il giro e memoria libera secondo il "
            "driver alla fine (il pool di PyTorch non restituisce: la libera scende al picco più alto visto fin lì). GiB.\n\n"
            + table(["giro"] + [f"picco cuda:{d}" for d in env.device_ids] + [f"libera cuda:{d}" for d in env.device_ids], mem)
        )

    head = [
        "# Prefill accanto al decode: misura",
        "",
        time.strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "## Sintesi",
        "",
        table(summary_header, summary),
        "",
    ]
    for note in notes:
        head += [note, ""]
    slots = env.slots() if env.slots is not None else None
    head += [
        "## Condizioni",
        "",
        f"- Banco di prova: `{os.path.basename(__file__)}` al commit {env.revision}",
        f"- ExLlamaV3 {env.version} da `{env.lib_path}`, torch {env.torch_version}",
        f"- Modello `{env.model_path}`",
        f"- Schede: {', '.join(env.devices) if env.devices else '-'}",
        f"- Bozza: {env.draft_desc}",
        f"- `a`: prompt di {int(a_ids.shape[-1])} token; `b`: documento più domanda, {int(b_ids.shape[-1])} token, "
        f"pezzi di {gen.max_chunk_size}; `b` viene accodato dopo {args.bench_lead} token di `a`",
        "- Prima di ogni giro la cache dei prompt viene azzerata: ogni giro ingerisce il documento da capo",
        "- Il banco non ha le finestre di decode del server (`--decode-slice-ms`): «come oggi» qui è la libreria da sola, "
        "che dà ad `a` un passo per pezzo; in produzione le finestre gliene danno di più, a spese di `b`",
        f"- Slot del worker degli esperti: {slots or '-'}"
        + (" (prima che il thread serva la prima volta sono tutti in una rotazione, come in produzione)" if slots else ""),
        f"- Un pezzo gira accanto al decode se ha almeno {gen.side_min_tokens} token e il batch di decode non supera "
        f"{gen.side_max_rows} righe per token; altrimenti il generatore lo aspetta",
        f"- Intervallo di cambio del GIL: {1000 * sys.getswitchinterval():.1f} ms",
        "- Campionamento greedy, nessuna condizione di stop; dopo l'adattamento iniziale nessuno sweep del piazzamento",
        "- I tok/s di `a` «mentre `b` ingerisce» contano i token arrivati tra l'accodamento di `b` e il suo primo token",
        "- Uso e potenza delle schede: media dei campioni di `nvidia-smi` nella finestra, saltando il primo mezzo secondo"
        + (f" (campionatore fermato da un errore: {gpu.error})" if gpu is not None and gpu.error else ""),
        "",
    ]
    return "\n".join(head) + "\n" + "\n\n".join(sections) + "\n"


def main():
    ap = argparse.ArgumentParser(add_help = False)
    ap.add_argument("--bench-prompt-tokens", type = int, default = 8192, help = "token del documento che `b` ingerisce (default 8192)")
    ap.add_argument("--bench-tokens", type = int, default = 600, help = "token di `a` nel giro da solo e nel controllo (default 600)")
    ap.add_argument("--bench-decoder-tokens", type = int, default = 6000,
                    help = "tetto ai token di `a` nei giri insieme: deve durare più dell'ingestione di `b` (default 6000)")
    ap.add_argument("--bench-reader-tokens", type = int, default = 64, help = "token generati da `b` dopo il prompt (default 64)")
    ap.add_argument("--bench-lead", type = int, default = 48, help = "token di `a` prima che `b` venga accodato (default 48)")
    ap.add_argument("--bench-warmup", type = int, default = 48, help = "token del giro di riscaldamento (default 48)")
    ap.add_argument("--bench-adapt", type = int, default = 3, help = "giri al massimo per adattare il piazzamento degli esperti (default 3, 0 = no)")
    ap.add_argument("--bench-adapt-tokens", type = int, default = 160, help = "token di `a` in un giro di adattamento (default 160)")
    ap.add_argument("--bench-phases", default = "speed,check", help = "speed (velocità), check (stesso calcolo); default entrambe")
    ap.add_argument("--bench-slots", type = int, default = 0,
                    help = "slot del worker degli esperti (EXL3_MOE_CPU_SLOTS, 3-8); default 0: quelli della libreria (4)")
    ap.add_argument("--bench-switch-ms", type = float, default = 0.0,
                    help = "intervallo di cambio del GIL in millisecondi (default 0: quello di Python, 5 ms)")
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
    args.bench_phases = [p.strip() for p in args.bench_phases.split(",") if p.strip()]
    unknown = [p for p in args.bench_phases if p not in ("speed", "check")]
    if unknown:
        raise SystemExit(f"--bench-phases: fasi sconosciute {unknown}; valgono speed, check")
    if args.bench_slots:
        # Read once, when the library is imported
        os.environ["EXL3_MOE_CPU_SLOTS"] = str(args.bench_slots)
    if args.bench_switch_ms > 0:
        sys.setswitchinterval(args.bench_switch_ms / 1000.0)

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
    from exllamav3.util.lane import cuda_devices
    from exllamav3.version import __version__ as exl3_version
    from openai.config import parse_args
    from openai.model_container import ModelContainer

    if not hasattr(Generator, "_prefill_round"):
        raise SystemExit(
            f"La libreria importata ({exllamav3.__file__}) non ha il prefill sul secondo thread: "
            "metti il checkout del fork davanti nel PYTHONPATH e riprova."
        )

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
    gen.stagger_decode = False

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

    document = long_prompt(lambda text: tokenizer.encode(text).shape[-1], PROMPTS[1], "reader", args.bench_prompt_tokens)
    prompts = {DECODER: encode(PROMPTS[0]), READER: encode(document)}
    room = getattr(cfg, "cache_size", None)
    need = int(prompts[READER].shape[-1]) + int(prompts[DECODER].shape[-1]) + args.bench_decoder_tokens + args.bench_reader_tokens + 1024
    if room is not None and need > room:
        raise SystemExit(f"--bench-prompt-tokens {args.bench_prompt_tokens}: servono {need} token di cache, --cache-size ne ha {room}.")

    def slots():
        hosts = list(getattr(model.config, "moe_cpu_hosts", {}).values())
        if not hosts:
            return None
        h = hosts[0]
        lanes = getattr(h, "_lanes", None)
        if lanes is None:
            return f"{h.num_slots}, non ancora divisi"
        return f"{h.num_slots}: {lanes[0][1]} per il decode e {lanes[1][1]} per il prefill"

    env = SimpleNamespace(
        torch = torch,
        gen = gen,
        model = model,
        Job = Job,
        sampler_cls = GreedySampler,
        prompts = prompts,
        sweep = sweep,
        slots = slots,
        device_ids = cuda_devices(model, draft_model),
        version = exl3_version,
        revision = revision,
        lib_path = os.path.dirname(exllamav3.__file__),
        torch_version = torch.__version__,
        model_path = cfg.model_path,
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        draft_desc = f"MTP, {gen.num_draft_tokens} token" if draft_model is not None else "nessuna",
    )

    gpu = None
    if not args.bench_no_gpu:
        gpu = GpuSampler()
        gpu.start()
    try:
        text = measure(env, args, gpu)
    finally:
        if gpu is not None:
            gpu.stop()

    out = args.bench_out
    if out is None:
        folder = "prof-runs" if os.path.isdir("prof-runs") else "."
        out = os.path.join(folder, time.strftime("prefill-bench-%Y%m%d-%H%M%S.md"))
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
