#!/usr/bin/env python3
"""
Probe: the chunks of one prompt in flight on the two cards at the same time.

A prompt is ingested a chunk at a time, and with the layers of the model split over two cards
each card works for half of a chunk's pass and waits for the other half. Nothing in the model
forbids starting the next chunk on the first card while the current one is on the second: every
layer keeps its own state (K/V pages, recurrent state) and needs only its own state after the
previous chunk and the output of the layer before it for this one. So:

    first card   A(0)  A(1)  A(2)  A(3) ...
    second card        B(0)  B(1)  B(2) ...

with A(k+1) issued after A(k) and B(k+1) after B(k). This script does exactly that with the
model's real passes (model/forward_cursor.py splits a pass where it changes card, as the
staggered decode does), outside the generator's bookkeeping: a job is started so that it owns
pages and a recurrent state, its prompt's chunks are run by hand in the order above from two
threads (even chunks on this thread, odd ones on the prefill thread's lane: util/lane.py), and
the job is cancelled. It measures the time against the same chunks one after the other, and
compares the logits of every chunk's last token between the two orders: a pipeline that fed a
layer the wrong state would change them.

It answers two questions before any of this goes into the generator: how much of the two halves
really overlaps (the CPU expert worker, the host and the DRAM are shared), and whether the order
is safe with the modules as they are.

Run it with the server STOPPED, from the server repository, with the server's own flags:

    cd ~/Desktop/exllamav3-openai-server
    python /path/to/exllamav3/experiments/pipeline_probe.py --probe-out prof-runs/pipeline.md <the server's flags>

Options of the probe start with --probe-; everything else goes to the server's argument parser.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stagger_bench import PROMPTS, bench_revision, fmt, long_prompt, table     # noqa: E402

MODES = ("in fila", "in fila sul thread di prefill", "in volo sulle due schede")


def plan_spans(total: int, chunk: int, page: int = 256) -> list:
    """Whole chunks of the first `total` tokens: [start, end) with every end on a page boundary."""
    chunk = chunk // page * page
    return [(a, a + chunk) for a in range(0, total - chunk + 1, chunk)]


def pipeline_order(n: int, stage_a, stage_b, submit):
    """
    Run stage_a(k) then stage_b(k, what stage_a returned) for k in range(n) from two threads,
    even k on the calling thread and odd k on the thread behind `submit` (which takes a function
    and returns something with .result()), keeping to the two rules of the pipeline: A(k+1)
    after A(k), B(k+1) after B(k). Returns per chunk (seconds in A, seconds in B, seconds waited
    between the two). The first exception stops both threads and is raised.
    """
    done_a = [threading.Event() for _ in range(n)]
    done_b = [threading.Event() for _ in range(n)]
    times = [None] * n
    errors = []

    def work(k: int):
        try:
            if k:
                done_a[k - 1].wait()
            if errors:
                return
            t0 = time.perf_counter()
            state = stage_a(k)
            t1 = time.perf_counter()
            done_a[k].set()
            if k:
                done_b[k - 1].wait()
            if errors:
                return
            t2 = time.perf_counter()
            stage_b(k, state)
            times[k] = (t1 - t0, time.perf_counter() - t2, t2 - t1)
        except BaseException as e:
            errors.append(e)
        finally:
            done_a[k].set()
            done_b[k].set()

    def lane(ks):
        for k in ks:
            work(k)

    other = submit(lambda: lane(range(1, n, 2)))
    lane(range(0, n, 2))
    other.result()
    if errors:
        raise errors[0]
    return times


def report(env, rows: list, checks: list, stage_rows: list, notes: list) -> str:
    head = [
        "# Chunk di prefill in volo sulle due schede: sonda",
        "",
        f"- sonda: `{os.path.basename(__file__)}` al commit {env['revision']}",
        f"- libreria: exllamav3 {env['version']} da `{env['lib_path']}`",
        f"- modello: `{env['model_path']}`",
        f"- schede: {', '.join(env['devices']) or 'nessuna'}; il passaggio cambia scheda al modulo {env['bound']} di {env['modules']}",
        f"- bozze: {env['draft']}; thread del worker degli esperti: {env['threads'] or 'sconosciuti'}; "
        f"slot del worker: {env['slots'] or 'non divisi'}",
        f"- prompt: {env['tokens']} token in {env['chunks']} pezzi da {env['chunk']}",
        "- memoria libera sulle schede a fine misura: " + ", ".join(f"{d} {m} MiB" for d, m in env["memory"]),
        "",
    ]
    if notes:
        head += notes + [""]
    body = [
        "## Tempi\n\n"
        "Gli stessi pezzi dello stesso prompt, a ogni giro da capo (job nuovo, niente in cache), eseguiti a mano "
        "fuori dal generatore: nessun checkpoint, nessuna registrazione delle pagine. «in fila» è il passaggio "
        "intero di un pezzo dopo l'altro (`Model.forward`, come oggi); «in fila sul thread di prefill» lo stesso "
        "dal secondo thread con i suoi slot del worker (come la prod con `--prefill-thread`); «in volo sulle due "
        "schede» fa partire il pezzo successivo sulla prima scheda mentre il precedente è sulla seconda.\n\n"
        + table(["giro", "modo", "secondi", "token/s", "contro «in fila»"], rows),
    ]
    if stage_rows:
        body.append(
            "## Dove passa il tempo di un pezzo in volo\n\n"
            "Mediana sui pezzi, in secondi di orologio del thread che lo esegue (attese del worker comprese): metà "
            "sulla prima scheda, attesa che il pezzo precedente lasci la seconda, metà sulla seconda (con la "
            "testa MTP).\n\n"
            + table(["giro", "prima scheda", "attesa", "seconda scheda"], stage_rows)
        )
    if checks:
        body.append(
            "## Il risultato è lo stesso?\n\n"
            "Logit dell'ultimo token di ogni pezzo, contro il primo giro «in fila».\n\n"
            + table(["giro", "modo", "pezzi identici", "scarto massimo dei logit", "stesso token più probabile"], checks)
        )
    return "\n".join(head) + "\n" + "\n\n".join(body) + "\n"


def main():
    ap = argparse.ArgumentParser(add_help = False)
    ap.add_argument("--probe-tokens", type = int, default = 32768, help = "token del prompt da ingerire (default 32768)")
    ap.add_argument("--probe-chunk", type = int, default = 0, help = "dimensione del pezzo (default: quella del server)")
    ap.add_argument("--probe-rounds", type = int, default = 2, help = "giri per modo (default 2)")
    ap.add_argument("--probe-out", default = None, help = "file .md del rapporto")
    ap.add_argument("--probe-server-repo", default = os.getcwd(), help = "repository del server (default: directory corrente)")
    ap.add_argument("--probe-help", action = "store_true")
    args, server_argv = ap.parse_known_args()
    if args.probe_help:
        ap.print_help()
        print(__doc__)
        return 0
    if args.probe_out is not None and not args.probe_out.endswith(".md"):
        raise SystemExit("--probe-out: il rapporto è un file .md")

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.abspath(args.probe_server_repo))
    import logging
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch
    import exllamav3
    from exllamav3 import Generator, Job
    from exllamav3.generator.sampler import GreedySampler
    from exllamav3.model.forward_cursor import ForwardCursor, cuda_stage_bounds
    from exllamav3.modules import block_sparse_mlp_cpu
    from exllamav3.version import __version__ as exl3_version
    from openai.config import parse_args
    from openai.model_container import ModelContainer

    cfg = parse_args(server_argv)
    container = ModelContainer(cfg)
    container.load_model_sync()
    model, tokenizer = container.model, container.tokenizer
    draft_model = container.draft_model
    gen_kwargs = {}
    if draft_model is not None:
        gen_kwargs.update(draft_model = draft_model, draft_cache = container.draft_cache, num_draft_tokens = cfg.draft_num_tokens)
    if getattr(cfg, "recurrent_checkpoint_interval", None) is not None:
        gen_kwargs["recurrent_checkpoint_interval"] = cfg.recurrent_checkpoint_interval
    chunk = args.probe_chunk or cfg.max_chunk_size
    gen = Generator(
        model = model, cache = container.cache, tokenizer = tokenizer, max_batch_size = container.max_batch_size,
        max_chunk_size = chunk, **gen_kwargs,
    )
    # nothing sweeps the expert placement during the probe, and the server's file is not written
    block_sparse_mlp_cpu.run_pending_swap_sweeps = lambda infer_params: None

    notes = []
    bounds = cuda_stage_bounds(model)
    if len(bounds) != 1:
        raise SystemExit(f"Il modello è su {len(bounds) + 1} scheda/e: la sonda ne vuole due.")
    bound = bounds[0]
    if not gen.mtp_draft:
        raise SystemExit("La sonda esegue il passaggio di prefill come con la testa MTP (--draft-mode mtp).")

    def count_tokens(text: str) -> int:
        return tokenizer.encode(text).shape[-1]

    text = long_prompt(count_tokens, PROMPTS[1], f"pipeline-{os.getpid()}", args.probe_tokens + chunk)
    chat = getattr(model, "default_chat_prompt", None)
    ids = tokenizer.encode(chat(text) if chat is not None else text, encode_special_tokens = True)
    spans = plan_spans(min(args.probe_tokens, int(ids.shape[-1]) - 1), chunk)
    if len(spans) < 3:
        raise SystemExit("Servono almeno tre pezzi: alza --probe-tokens o abbassa --probe-chunk.")
    total = spans[-1][1]

    def start_job(tag: str):
        job = Job(input_ids = ids, max_new_tokens = 8, sampler = GreedySampler(), identifier = tag)
        gen.enqueue(job)
        gen.iterate_start_jobs([])
        if job not in gen.active_jobs:
            raise RuntimeError("the job did not start (no room in the cache?)")
        if job.cached_pages:
            raise RuntimeError(f"{job.cached_pages} pages of the prompt came from the cache: the probe wants a cold prompt")
        return job

    def end_job(job):
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)
        gen.cancel(job)

    def chunk_inputs(job, k: int):
        """What Job.prefill hands the model for the chunk (its `run`), text only."""
        seq = job.sequences[0]
        a, b = spans[k]
        params = {
            "attn_mode": "flash_attn",
            "block_table": seq.block_index_tensor,
            "cache": gen.cache,
            "cache_seqlens": torch.tensor([a], dtype = torch.int32),
            "recurrent_states": [job.recurrent_state] if job.recurrent_state is not None else None,
            "indexed_embeddings": None,
            "inv_freq": None,
            "mm_span_prefix": 0,
        }
        params.update(gen.draft_model.draft_verifier_params)
        params["last_tokens_only"] = 1
        return seq.sequence_ids.torch_slice(a, b), params

    def after_pass(job, k: int, chunk_ids, params, out, carry: dict, outs: list):
        """The MTP head's own prefill of the chunk, as in Job.prefill, and the chunk's logits kept."""
        seq = job.sequences[0]
        target_hidden = params.get("export_states")[-1]
        prev = carry.get("h")
        if prev is None:
            prev = torch.zeros_like(target_hidden[:, :1, :])
        shifted = torch.cat((prev, target_hidden[:, :-1, :]), dim = 1)
        carry["h"] = target_hidden[:, -1:, :].clone()
        gen.draft_model.prefill(
            input_ids = chunk_ids,
            params = {
                "target_hidden": shifted,
                "attn_mode": "flash_attn",
                "block_table": seq.block_index_tensor,
                "cache": gen.draft_cache,
                "cache_seqlens": torch.tensor([spans[k][0]], dtype = torch.int32),
                "indexed_embeddings": None,
            },
        )
        outs[k] = out[..., -1, :].clone()

    def in_line(job) -> tuple:
        carry, outs = {}, [None] * len(spans)
        for k in range(len(spans)):
            chunk_ids, params = chunk_inputs(job, k)
            out = model.forward(input_ids = chunk_ids, params = params)
            after_pass(job, k, chunk_ids, params, out, carry, outs)
        return outs, None

    def on_side(fn):
        """Run fn on the prefill thread, as Generator._side_work runs a chunk."""
        lane = gen._side_lane

        def work():
            with torch.inference_mode(), lane.passing():
                try:
                    return fn()
                finally:
                    lane.synchronize()

        return gen._side_pool.submit(work)

    def in_line_on_side(job) -> tuple:
        return on_side(lambda: in_line(job)).result()

    def in_flight(job) -> tuple:
        carry, outs = {}, [None] * len(spans)

        def stage_a(k: int):
            chunk_ids, params = chunk_inputs(job, k)
            cur = ForwardCursor(model, chunk_ids, params)
            cur.run_to(bound)
            cur.hold()
            return cur, chunk_ids, params

        def stage_b(k: int, state):
            cur, chunk_ids, params = state
            cur.wake()
            out = cur.finish()
            after_pass(job, k, chunk_ids, params, out, carry, outs)

        times = pipeline_order(len(spans), stage_a, stage_b, on_side)
        return outs, times

    runners = dict(zip(MODES, (in_line, in_line_on_side, in_flight)))

    def timed(mode: str, tag: str) -> dict:
        job = start_job(tag)
        try:
            for d in range(torch.cuda.device_count()):
                torch.cuda.synchronize(d)
            t0 = time.perf_counter()
            with torch.inference_mode():
                outs, times = runners[mode](job)
            for d in range(torch.cuda.device_count()):
                torch.cuda.synchronize(d)
            dt = time.perf_counter() - t0
            return {"mode": mode, "seconds": dt, "outs": [o.float().cpu() for o in outs], "times": times}
        finally:
            end_job(job)

    rows, checks, stage_rows, runs = [], [], [], []
    slots = None
    try:
        with torch.inference_mode():
            print(" -- probe: warm-up", flush = True)
            timed(MODES[0], "warm")
            print(" -- probe: in line, before the worker's slots are split", flush = True)
            for i in range(args.probe_rounds):
                runs.append((i + 1, timed(MODES[0], f"line-{i}")))
            gen.prefill_thread = True
            if not gen._side_plan():
                raise RuntimeError(f"the prefill thread is not available: {gen.prefill_thread_reason}")
            hosts = list(getattr(model.config, "moe_cpu_hosts", {}).values())
            if hosts and getattr(hosts[0], "_lanes", None):
                slots = " + ".join(str(v[1]) for v in hosts[0]._lanes.values())
            print(" -- probe: in line on the prefill thread, and in flight", flush = True)
            for i in range(args.probe_rounds):
                order = MODES[1:] if i % 2 == 0 else MODES[:0:-1]
                for mode in order:
                    runs.append((i + 1, timed(mode, f"{'side' if mode == MODES[1] else 'flight'}-{i}")))
    except Exception as e:
        traceback.print_exc()
        notes.append(f"**Attenzione**: la sonda si è fermata con `{type(e).__name__}: {e}`; le righe che mancano non sono state misurate.")

    base = statistics.median([r["seconds"] for _, r in runs if r["mode"] == MODES[0]]) if runs else None
    ref = runs[0][1]["outs"] if runs else None
    for i, r in runs:
        rows.append([i, r["mode"], fmt(r["seconds"], 2), fmt(total / r["seconds"]),
                     f"{100 * (base / r['seconds'] - 1):+.1f}%" if r["mode"] != MODES[0] else "-"])
        if r["times"]:
            ts = [t for t in r["times"] if t]
            stage_rows.append([i] + [fmt(statistics.median([t[j] for t in ts]), 3) for j in (0, 2, 1)])
        if r is not runs[0][1]:
            same = sum(1 for a, b in zip(ref, r["outs"]) if torch.equal(a, b))
            worst = max(float((a - b).abs().max()) for a, b in zip(ref, r["outs"]))
            top = sum(1 for a, b in zip(ref, r["outs"]) if int(a.argmax()) == int(b.argmax()))
            bad = sum(1 for o in r["outs"] if not bool(torch.isfinite(o).all()))
            checks.append([i, r["mode"], f"{same}/{len(ref)}", f"{worst:.4f}", f"{top}/{len(ref)}"])
            if bad:
                notes.append(f"**Attenzione**: nel giro {i} «{r['mode']}» {bad} pezzi hanno logit non finiti.")

    hosts = list(getattr(model.config, "moe_cpu_hosts", {}).values())
    env = {
        "revision": bench_revision(), "version": exl3_version, "lib_path": os.path.dirname(exllamav3.__file__),
        "model_path": cfg.model_path, "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "bound": bound, "modules": len(model.fwd_modules), "draft": f"MTP, {gen.num_draft_tokens} token",
        "threads": getattr(hosts[0], "threads", None) if hosts else None, "slots": slots,
        "tokens": total, "chunks": len(spans), "chunk": spans[0][1] - spans[0][0],
        "memory": [(f"cuda:{i}", torch.cuda.mem_get_info(i)[0] // (1 << 20)) for i in range(torch.cuda.device_count())],
    }
    text_out = report(env, rows, checks, stage_rows, notes)
    out = args.probe_out or time.strftime("pipeline-probe-%Y%m%d-%H%M%S.md")
    with open(out, "w") as f:
        f.write(text_out)
    print(text_out)
    print(f" -- probe: report written to {out}", flush = True)
    return 0


if __name__ == "__main__":
    code = main()
    # No interpreter teardown: exit hooks of the server's modules (placement save among them) do
    # not belong in a probe, and the CPU expert worker ends with its parent (PDEATHSIG)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code or 0)
