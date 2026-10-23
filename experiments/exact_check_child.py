#!/usr/bin/env python3
"""
Child process of experiments/exact_check.py: one model load, the steps of one process of the
plan, one record per step appended (and synced) to records.jsonl in the process's directory.

The model is loaded the way the server loads it, through the server's own ModelContainer
(--server-path of the orchestrator, ahead on sys.path), so the expert split per card, the warm
start from the placement file and the library knobs the server sets before a load are
production's. A raw Generator is then driven from this process, as the fork's benches did.

Every library call goes through Watch.guard: a faulthandler timer that, past the step timeout,
writes the Python stacks of all threads to hang-stacks.txt and ends the process (the orchestrator
reports it). status.json says what the process is doing, for the orchestrator's own watchdog.

Not meant to be run by hand: the orchestrator writes the spec and the environment (the library
on PYTHONPATH or not, the exact mode, the worker's knobs).
"""

from __future__ import annotations

import os

# Before torch is imported, as the server does (golden rule 2 of its AGENTS.md)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")

import argparse
import faulthandler
import hashlib
import inspect
import json
import logging
import math
import random
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exact_check as EC      # noqa: E402  (standard library only)


class RunError(RuntimeError):
    """A run could not do what its spec says (our own logic, not the library's)."""


class LibraryError(RuntimeError):
    """A library call raised: the generator may be left inconsistent."""


# --------------------------------------------------------------------------------------------
#   Watchdog
# --------------------------------------------------------------------------------------------

class Watch:
    """
    faulthandler's C-level timer around each library call: it needs no GIL, so a call that
    never returns (a CUDA wait, a lock, a worker that died) still gets its stacks written, for
    every thread, and the process ended with status 1. SIGUSR1 writes the stacks too (the
    orchestrator sends it before it kills a process that went silent). status.json is rewritten
    at most once a second, and whenever the operation changes.
    """

    def __init__(self, proc_dir: str, step_timeout: float, clock = time.time):
        self.status_path = os.path.join(proc_dir, "status.json")
        self.hang = open(os.path.join(proc_dir, "hang-stacks.txt"), "a")
        self.step_timeout = float(step_timeout)
        self.clock = clock
        self.scenario = None
        self.run = None
        self.op = "start"
        self.timeout = self.step_timeout
        self.iterations = 0
        self.last_call_s = 0.0
        self._last = 0.0
        faulthandler.enable(file = self.hang, all_threads = True)
        try:
            faulthandler.register(signal.SIGUSR1, file = self.hang, all_threads = True, chain = False)
        except (AttributeError, ValueError, RuntimeError):
            pass

    def beat(self, op: str | None = None, timeout: float | None = None, force: bool = False):
        now = self.clock()
        if op is not None and op != self.op:
            self.op = op
            force = True
        if timeout is not None and timeout != self.timeout:
            self.timeout = timeout
            force = True
        if not force and now - self._last < 1.0:
            return
        self._last = now
        status = {"t": now, "pid": os.getpid(), "op": self.op, "run": self.run, "scenario": self.scenario,
                  "timeout": self.timeout, "iterations": self.iterations}
        tmp = self.status_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(status, f)
            os.replace(tmp, self.status_path)
        except OSError:
            pass

    @contextmanager
    def guard(self, op: str, timeout: float | None = None):
        t = float(timeout or self.step_timeout)
        self.beat(op, timeout = t)
        faulthandler.dump_traceback_later(t, repeat = False, file = self.hang, exit = True)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
            self.last_call_s = time.perf_counter() - t0
            self.beat()

    def dump(self, why: str):
        """The stacks now, from Python (a run that makes no progress, not a call that hangs)."""
        try:
            self.hang.write(f"\n--- {time.strftime('%H:%M:%S')} {why}\n")
            self.hang.flush()
            faulthandler.dump_traceback(file = self.hang, all_threads = True)
            self.hang.flush()
        except Exception:
            pass


# --------------------------------------------------------------------------------------------
#   What every sampled position leaves behind
# --------------------------------------------------------------------------------------------

class Digester:
    """
    Wraps a job's receive_logits (instance attribute: the generator calls it through the job)
    and records, for every sampled position, on the logits as they come out of the model:

      - an integer digest of their bits: sum over the row of (bits as int16/int32) * w, w a
        fixed odd 31-bit weight per vocabulary entry (the same in every process), in int64 with
        wrap-around. Integer addition is associative, so the digest does not depend on the order
        of the reduction: equal logits give equal digests, and different ones collide with
        probability about 2**-64;
      - the logit of the token the job emitted and the log-sum-exp over the real vocabulary
        (the tokenizer's actual_vocab_size: the sampler masks the padding in place, which a
        forced token skips, so the library's own token_probs would differ between a free and a
        forced run of the same computation);
      - the k most likely tokens with their log-probabilities;
      - the job's draft counters (accepted, rejected) as they stand when the position is
        sampled: host integers, so the MTP head's drafts can be compared position by position
        over a part of a run, not only as the job's totals.

    Everything is queued on the logits' stream as small kernels and read back after iterate()
    (resolve), where the generator has synchronized anyway: no host wait is added to the step.
    """

    def __init__(self, torch, vocab: int | None, k: int):
        self.torch = torch
        self.vocab = vocab
        self.k = max(1, int(k))
        self._weights = {}
        self.pending = {}
        self.errors = []

    def weights(self, n: int, device):
        key = (n, str(device))
        w = self._weights.get(key)
        if w is None:
            torch = self.torch
            g = torch.Generator().manual_seed(0x5EED1EC)
            w = torch.randint(1, 2 ** 31 - 1, (n,), generator = g, dtype = torch.int64) | 1
            w = w.to(device)
            self._weights[key] = w
        return w

    def wrap(self, job, name: str):
        orig = job.receive_logits
        dig = self

        def receive_logits(logits, *args, **kwargs):
            pre = None
            try:
                pre = dig._pre(logits)
            except Exception as e:
                dig._error(e)
            drafts = (int(getattr(job, "accepted_draft_tokens", 0) or 0), int(getattr(job, "rejected_draft_tokens", 0) or 0))
            out = orig(logits, *args, **kwargs)
            entry = None
            if pre is not None:
                try:
                    entry = dig._post(pre, out[0])
                except Exception as e:
                    dig._error(e)
            dig.pending.setdefault(name, []).append((entry, drafts))
            return out

        job.receive_logits = receive_logits

    def _error(self, e):
        if len(self.errors) < 10:
            self.errors.append(f"{type(e).__name__}: {e}")

    def _pre(self, logits):
        torch = self.torch
        row = logits.reshape(-1, logits.shape[-1])[0].contiguous()
        if row.dtype == torch.float32:
            bits = row.view(torch.int32)
        elif row.dtype in (torch.float16, torch.bfloat16):
            bits = row.view(torch.int16)
        else:
            bits = row.float().view(torch.int32)
        digest = (bits.to(torch.int64) * self.weights(row.shape[-1], row.device)).sum()
        lf = row.float()
        real = lf[: self.vocab] if self.vocab and self.vocab < lf.shape[-1] else lf
        lse = torch.logsumexp(real, 0)
        k = min(self.k, real.shape[-1])
        tv, ti = torch.topk(real, k)
        return lf, digest, lse, tv, ti

    def _post(self, pre, next_token):
        torch = self.torch
        lf, digest, lse, tv, ti = pre
        tok = next_token.reshape(-1)[:1].to(device = lf.device, dtype = torch.long)
        tl = lf.gather(0, tok)
        ints = torch.cat([digest.view(1), tok, ti.to(torch.int64)])
        floats = torch.cat([lse.view(1), tl.view(1), tv])
        return ints, floats

    def resolve(self, out: dict):
        """Read the queued values back into out[name] (lists per position)."""
        torch = self.torch
        pending, self.pending = self.pending, {}
        for name, pairs in pending.items():
            dst = out[name]
            entries = [e for e, _ in pairs]
            for _, (acc, rej) in pairs:
                dst.setdefault("acc", []).append(acc)
                dst.setdefault("rej", []).append(rej)
            good = [e for e in entries if e is not None]
            ints = floats = None
            if good:
                try:
                    ints = torch.stack([e[0] for e in good]).cpu().tolist()
                    floats = torch.stack([e[1] for e in good]).cpu().tolist()
                except Exception as e:
                    self._error(e)
                    ints = floats = None
            gi = 0
            for e in entries:
                if e is None or ints is None:
                    dst["digest"].append(None)
                    dst["tok"].append(None)
                    dst["logp"].append(None)
                    dst["topi"].append(None)
                    dst["topl"].append(None)
                    continue
                iv, fv = ints[gi], floats[gi]
                gi += 1
                lse = fv[0]
                dst["digest"].append(int(iv[0]))
                dst["tok"].append(int(iv[1]))
                dst["logp"].append(fv[1] - lse)
                dst["topi"].append([int(x) for x in iv[2:]])
                dst["topl"].append([v - lse for v in fv[2:]])


# --------------------------------------------------------------------------------------------
#   Prompts
# --------------------------------------------------------------------------------------------

class Prompts:
    """The token ids of a process's prompt keys, built once each, the same in every process
    (the same text through the same tokenizer). A document of an exact length is three pieces
    encoded apart: the chat template's head with an opening line, a pseudo-prose body cut to the
    token, and the question with the template's tail."""

    def __init__(self, torch, tokenizer, chat, specs: dict, answer_of):
        self.torch = torch
        self.tokenizer = tokenizer
        self.specs = specs
        self.answer_of = answer_of
        self.cache = {}
        marker = "⁣EXACT⁣"
        if chat is not None:
            text = chat(marker)
            self.head, _, self.tail = text.partition(marker)
        else:
            self.head, self.tail = "", ""

    def encode(self, text: str):
        ids = self.tokenizer.encode(text, encode_special_tokens = True)
        return ids.reshape(1, -1).to(self.torch.long)

    def ids(self, key: str):
        got = self.cache.get(key)
        if got is not None:
            return got
        spec = self.specs.get(key) or EC.prompt_spec_of(key)
        torch = self.torch
        if spec["kind"] == "question":
            ids = self.encode(self.head + EC.PROMPTS[spec["index"]] + self.tail)
        elif spec["kind"] == "doc":
            head = self.encode(self.head + "Read this document.\n\n")
            tail = self.encode("\n\nNow set the document aside. " + EC.PROMPTS[spec["question"]] + self.tail)
            need = spec["length"] - head.shape[-1] - tail.shape[-1]
            if need <= 0:
                raise RunError(f"prompt {key}: {spec['length']} tokens leave no room for the document")
            paragraphs = max(4, need // 100)
            while True:
                body = self.encode(EC.filler(spec["tag"], paragraphs))
                if body.shape[-1] >= need:
                    break
                paragraphs = int(paragraphs * 1.5) + 4
            ids = torch.cat([head, body[:, :need], tail], dim = -1)
            assert ids.shape[-1] == spec["length"], (key, ids.shape)
        elif spec["kind"] == "turn":
            base = self.ids(spec["base"])
            answer = self.answer_of(spec["answer"])
            ids = torch.cat([base, torch.tensor([answer], dtype = torch.long), self.encode(spec["suffix"])], dim = -1)
        else:
            raise RunError(f"unknown prompt kind {spec['kind']}")
        self.cache[key] = ids
        return ids


# --------------------------------------------------------------------------------------------
#   One run
# --------------------------------------------------------------------------------------------

def numeric_delta(a: dict | None, b: dict | None) -> dict | None:
    if a is None or b is None:
        return None
    out = {}
    for k, v in b.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(a.get(k), (int, float)):
            out[k] = v - a[k]
    return out


class Runner:
    """
    Executes a run spec over whatever the environment `x` provides (the real generator in a
    child, the toy model of the fork's tests in tests/test_exact_check.py):

      x.torch, x.gen, x.clock(), x.watch (guard, beat, dump), x.make_job(name, ids, max_new),
      x.prompt_ids(key), x.forced_tokens(ref), x.digester(), x.apply_switches(switches),
      x.forget(), x.stats(), x.state(), x.placement(), x.reset_memory(), x.sample_memory(),
      x.memory(), x.disk_metrics(), x.run_timeout
    """

    def __init__(self, x):
        self.x = x

    def run(self, spec: dict) -> dict:
        x = self.x
        rec = {"op": "run", "id": spec["id"], "scenario": spec.get("scenario"), "ok": False, "warnings": [],
               "warmup": bool(spec.get("warmup"))}
        t_start = x.clock()
        order = [j["name"] for j in spec["jobs"]]
        jr = {n: {"tokens": [], "digest": [], "tok": [], "logp": [], "topi": [], "topl": [], "acc": [], "rej": [],
                  "prefill": [], "enqueued": None, "first_token": None, "ended": None, "cancelled": False, "eos": False}
              for n in order}
        rec.update(order = order, jobs = jr, timeline = [], cancels = [])
        try:
            self._run(spec, rec, jr, order, t_start)
            rec["ok"] = True
        except LibraryError as e:
            rec["error"] = str(e)
            rec["fatal"] = True
            self._clear(rec)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc()[-2000:]
            self._clear(rec)
        rec["seconds"] = x.clock() - t_start
        return rec

    def _clear(self, rec):
        try:
            if self.x.gen.num_remaining_jobs():
                with self.x.watch.guard("clear_queue"):
                    self.x.gen.clear_queue()
        except Exception as e:
            rec["warnings"].append(f"clear_queue() dopo l'errore: {type(e).__name__}: {e}")
            rec["fatal"] = True

    def _call(self, op: str, fn, *args):
        """A library call under the watchdog; what it raises is the library's."""
        try:
            with self.x.watch.guard(op):
                return fn(*args)
        except Exception as e:
            raise LibraryError(f"{op}: {type(e).__name__}: {e}") from e

    def _run(self, spec, rec, jr, order, t_start):
        x = self.x
        torch, gen = x.torch, x.gen
        x.apply_switches(spec.get("switches") or {})
        if spec.get("forget", True):
            rec["forgot"] = x.forget()
        stats0 = x.stats()
        rec["placement_before"] = x.placement()
        x.reset_memory()
        specs = {j["name"]: j for j in spec["jobs"]}
        forced = {n: list(x.forced_tokens(js["forced"])) for n, js in specs.items() if "forced" in js}
        for n, toks in forced.items():
            if not toks:
                raise RunError(f"no tokens to force on {n}")
        dig = x.digester()
        jobs, by_id = {}, {}
        finished = set()

        def count(name):
            j = jobs.get(name)
            return 0 if j is None else max(0, int(getattr(j, "new_tokens", 0) or 0))

        def start(name):
            js = specs[name]
            ids = x.prompt_ids(js["prompt"])
            n_new = len(forced[name]) if name in forced else int(js["max_new"])
            job = x.make_job(name, ids, n_new)
            dig.wrap(job, name)
            self._call("enqueue", gen.enqueue, job)
            if name in forced:
                job.constrain_output_now(torch.tensor([forced[name]], dtype = torch.long))
            jobs[name] = job
            by_id[id(job)] = name
            jr[name].update(enqueued = x.clock(), prompt_tokens = int(ids.shape[-1]), forced = name in forced)

        waiting = [n for n in order if specs[n].get("start")]
        for n in order:
            if n not in waiting:
                start(n)
        until = spec.get("until") or {"all": True}
        cancels = list(spec.get("cancel") or [])
        after = spec.get("after_cancel")
        deadline = t_start + float(spec.get("timeout_s") or x.run_timeout)
        disk_first = bool(spec.get("disk_at_first_token"))
        iters = 0
        max_step = 0.0
        errors = []

        while True:
            for n in list(waiting):
                st = specs[n]["start"]
                other = st["after"]
                if count(other) >= st["tokens"]:
                    start(n)
                    waiting.remove(n)
                elif other in finished:
                    raise RunError(f"`{other}` ended after {count(other)} tokens, before `{n}` could start at {st['tokens']}")
            if not gen.num_remaining_jobs():
                if waiting:
                    raise RunError(f"nothing left to run while {waiting} wait to start")
                break
            t0 = time.perf_counter()
            results = self._call("iterate", gen.iterate)
            dt = time.perf_counter() - t0
            iters += 1
            x.watch.iterations += 1
            max_step = max(max_step, dt)
            now = x.clock()
            with x.watch.guard("resolve"):
                dig.resolve(jr)
            for r in results:
                name = by_id.get(id(r.get("job")))
                if name is None:
                    continue
                stage = r.get("stage")
                if stage == "error":
                    errors.append(f"`{name}`: {r.get('error')!r}")
                    finished.add(name)
                    continue
                if stage == "prefill":
                    jr[name]["prefill"].append([now, r.get("curr_progress"), r.get("max_progress")])
                    continue
                if stage != "streaming":
                    continue
                ids = r.get("token_ids")
                if ids is not None and ids.numel():
                    if jr[name]["first_token"] is None:
                        jr[name]["first_token"] = now
                        if disk_first:
                            jr[name]["disk_at_first_token"] = x.disk_metrics()
                    jr[name]["tokens"] += [int(t) for t in ids.reshape(-1).tolist()]
                if r.get("eos"):
                    finished.add(name)
                    jr[name].update(eos = True, eos_reason = r.get("eos_reason"), cached_tokens = r.get("cached_tokens"),
                                    ended = now)
            rec["timeline"].append([now, [count(n) for n in order]])
            x.sample_memory()
            if errors:
                raise RunError("job falliti: " + "; ".join(errors))
            for c in list(cancels):
                name = c["job"]
                j = jobs.get(name)
                if j is None or name in finished:
                    continue
                pf = jr[name]["prefill"]
                frac = (pf[-1][1] / pf[-1][2]) if pf and pf[-1][2] else 0.0
                if frac >= c["at_progress"]:
                    t0 = time.perf_counter()
                    self._call("cancel", gen.cancel, j)
                    rec["cancels"].append({"job": name, "progress": frac, "seconds": time.perf_counter() - t0,
                                           "tokens": {n: count(n) for n in order}})
                    finished.add(name)
                    jr[name]["cancelled"] = True
                    cancels.remove(c)
            end = bool(until.get("job")) and until["job"] in finished
            if after and rec["cancels"] and not cancels:
                base = rec["cancels"][-1]["tokens"].get(after["job"], 0)
                if count(after["job"]) >= base + after["tokens"] or after["job"] in finished:
                    end = True
            if end:
                for n, j in jobs.items():
                    if n not in finished:
                        self._call("cancel", gen.cancel, j)
                        jr[n]["cancelled"] = True
                        finished.add(n)
                if waiting:
                    rec["warnings"].append(f"job mai partiti: {waiting}")
                break
            if x.clock() > deadline:
                x.watch.dump(f"run {spec['id']} past its time")
                raise RunError(f"oltre il tempo del giro ({deadline - t_start:.0f} s), stack in hang-stacks.txt")
        if cancels:
            rec["warnings"].append("annullamento non avvenuto: " + ", ".join(c["job"] for c in cancels))
        with x.watch.guard("resolve"):
            dig.resolve(jr)
        if dig.errors:
            rec["warnings"].append("impronta dei logit non calcolata: " + "; ".join(dig.errors[:3]))
        for n, j in jobs.items():
            v = jr[n]
            v["accepted"] = int(getattr(j, "accepted_draft_tokens", 0) or 0)
            v["rejected"] = int(getattr(j, "rejected_draft_tokens", 0) or 0)
            v["resumed_from_tail"] = getattr(j, "resumed_from_tail", None)
            if v.get("cached_tokens") is None and hasattr(j, "cached_pages"):
                v["cached_tokens"] = int(getattr(j, "cached_pages", 0)) * 256 + int(getattr(j, "cached_tokens", 0) or 0)
            if v["first_token"] is not None and v["enqueued"] is not None:
                v["ttft"] = v["first_token"] - v["enqueued"]
            if len(v["digest"]) != len(v["tokens"]) and not v["cancelled"]:
                rec["warnings"].append(f"`{n}`: {len(v['digest'])} posizioni registrate per {len(v['tokens'])} token")
            if n in forced and not v["cancelled"] and v["tokens"] != forced[n][:len(v["tokens"])]:
                rec["warnings"].append(f"`{n}`: i token usciti non sono quelli imposti")
        rec["left_jobs"] = gen.num_remaining_jobs()
        if rec["left_jobs"]:
            rec["warnings"].append(f"{rec['left_jobs']} job rimasti nella coda alla fine del giro")
            self._call("clear_queue", gen.clear_queue)
        stats1 = x.stats()
        rec["stats"] = {k: numeric_delta(stats0.get(k), stats1.get(k)) for k in stats1}
        rec["state"] = x.state()
        rec["memory"] = x.memory()
        rec["placement_after"] = x.placement()
        rec["iterations"] = iters
        rec["max_step_s"] = max_step


# --------------------------------------------------------------------------------------------
#   Stress
# --------------------------------------------------------------------------------------------

STRESS_KINDS = ("short", "doc", "long", "turn", "cancel")
STRESS_WEIGHTS = (54, 18, 3, 15, 10)


def stress_schedule(seed: int, minutes: float, mean_gap: float = 5.0) -> list:
    """Arrivals over the stress's duration, from a seed: (second, kind, a uniform number, an
    integer for names and lengths). Kinds: a short question, a document of 8-33k tokens, a long
    prompt of 64-96k tokens (past the 32768-token checkpoints of a prompt, more than once), the
    next turn of a conversation that ended, a cancel of a job in any state. The same seed gives
    the same schedule."""
    rng = random.Random(seed)
    events, t = [], 0.0
    while True:
        t += rng.expovariate(1.0 / mean_gap)
        if t >= minutes * 60.0:
            break
        kind = rng.choices(STRESS_KINDS, weights = STRESS_WEIGHTS)[0]
        events.append((t, kind, rng.random(), rng.randrange(1 << 30)))
    return events


def run_stress(x, spec: dict) -> dict:
    """
    Concurrent traffic for a while, then the drain: every arrival from stress_schedule, at most
    eight jobs in the generator, idle work between jobs as the server's worker does. Records the
    longest calls, the exceptions, what the generator's counters moved.
    """
    gen, torch = x.gen, x.torch
    rec = {"op": "stress", "id": spec["id"], "scenario": spec.get("scenario"), "ok": False, "warnings": []}
    s = {"jobs": 0, "kinds": {}, "done": 0, "cancelled": 0, "job_errors": 0, "prompt_tokens": 0, "generated": 0,
         "iterations": 0, "exceptions": [], "step_limit": spec.get("step_limit"), "over_limit": 0, "drained": False}
    rec["stress"] = s
    events = stress_schedule(spec.get("seed", 0), float(spec["minutes"]), float(spec.get("mean_gap") or 5.0))
    short_lo, short_hi = spec.get("short_tokens") or (32, 512)
    doc_new = int(spec.get("doc_tokens") or 64)
    stats0 = x.stats()
    x.reset_memory()
    jobs = {}           # id -> (name, job, prompt ids)
    finished = []       # (prompt ids, output tokens) of jobs that ended by themselves
    outputs = {}
    steps = []
    slow = []
    t0 = x.clock()
    last_progress = t0
    step_timeout = x.watch.step_timeout
    limit = float(spec.get("step_limit") or 0)
    serial = 0
    doc_lengths = [EC.doc_length(0, n) for n in spec.get("doc_pages") or (32, 64, 130)]
    long_lengths = [EC.doc_length(0, n) for n in spec.get("long_pages") or (256, 384)]
    max_rq = getattr(x, "max_rq_tokens", None)

    def call(op, fn, *args):
        t = time.perf_counter()
        try:
            with x.watch.guard(op):
                return fn(*args)
        finally:
            dt = time.perf_counter() - t
            if op != "iterate":
                slow.append((dt, op))
            elif limit and dt > limit:
                s["over_limit"] += 1

    def enqueue(kind, u, k):
        nonlocal serial
        serial += 1
        name = f"{kind}{serial}"
        if kind == "short":
            ids = x.prompt_ids(EC.q(k % len(EC.PROMPTS)))
            max_new = short_lo + int(u * (short_hi - short_lo))
        elif kind in ("doc", "long"):
            lengths = doc_lengths if kind == "doc" else long_lengths
            length = lengths[k % len(lengths)] + (k >> 3) % 250
            ids = x.prompt_ids(EC.doc(f"s7d{serial}", length, k % len(EC.PROMPTS)))
            max_new = doc_new
        else:
            if not finished:
                return enqueue("short", u, k)
            pids, out = finished[k % len(finished)]
            suffix = x.prompt_ids_text(EC.TURN_SUFFIX)
            ids = torch.cat([pids, torch.tensor([out], dtype = torch.long), suffix], dim = -1)
            max_new = doc_new
        job = x.make_job(name, ids, max_new, max_rq_tokens = max_rq)
        call("enqueue", gen.enqueue, job)
        jobs[id(job)] = (name, job, ids)
        outputs[id(job)] = []
        s["jobs"] += 1
        s["kinds"][kind] = s["kinds"].get(kind, 0) + 1
        s["prompt_tokens"] += int(ids.shape[-1])

    def handle(results):
        nonlocal last_progress
        for r in results:
            key = id(r.get("job"))
            if key not in jobs:
                continue
            last_progress = x.clock()
            stage = r.get("stage")
            if stage == "error":
                s["job_errors"] += 1
                if len(s["exceptions"]) < 20:
                    s["exceptions"].append(f"job {jobs[key][0]}: {r.get('error')!r}")
                jobs.pop(key, None)
                continue
            if stage == "streaming":
                ids = r.get("token_ids")
                if ids is not None and ids.numel():
                    outputs[key] += ids.reshape(-1).tolist()
                    s["generated"] += int(ids.numel())
                if r.get("eos"):
                    name, job, pids = jobs.pop(key)
                    s["done"] += 1
                    if outputs[key]:
                        finished.append((pids, outputs[key]))
                        del finished[:-32]

    try:
        while True:
            now = x.clock() - t0
            draining = now >= spec["minutes"] * 60.0
            while events and events[0][0] <= now and not draining:
                _, kind, u, k = events.pop(0)
                if kind == "cancel":
                    live = [v for v in jobs.values()]
                    if live:
                        name, job, _ = live[k % len(live)]
                        call("cancel", gen.cancel, job)
                        jobs.pop(id(job), None)
                        s["cancelled"] += 1
                    continue
                if gen.num_remaining_jobs() >= 8:
                    continue
                enqueue(kind, u, k)
            if draining and not gen.num_remaining_jobs():
                s["drained"] = True
                break
            if gen.num_remaining_jobs():
                t = time.perf_counter()
                results = call("iterate", gen.iterate)
                steps.append(time.perf_counter() - t)
                s["iterations"] += 1
                x.watch.iterations += 1
                handle(results)
                x.sample_memory()
                if x.clock() - last_progress > step_timeout:
                    x.watch.dump("stress: no progress")
                    s["hang"] = f"nessun progresso per {step_timeout:.0f} s con {gen.num_remaining_jobs()} job"
                    break
                if draining and x.clock() - t0 - spec["minutes"] * 60.0 > float(spec.get("drain_timeout") or 600):
                    s["hang"] = "la coda non si svuota"
                    x.watch.dump("stress: queue does not drain")
                    break
            else:
                last_progress = x.clock()
                x.watch.beat()
                if getattr(gen, "has_idle_work", None) is not None and gen.has_idle_work():
                    call("idle_work", gen.idle_work)
                else:
                    time.sleep(0.05)
        s["drain_s"] = x.clock() - t0 - spec["minutes"] * 60.0
        # Idle work until there is none, as the server's worker does while idle
        t_idle = x.clock()
        while getattr(gen, "has_idle_work", None) is not None and gen.has_idle_work():
            call("idle_work", gen.idle_work)
            if x.clock() - t_idle > 300:
                rec["warnings"].append("il lavoro di inattività non finisce in 300 s")
                break
            x.watch.beat()
            time.sleep(0.25)
        rec["ok"] = not s.get("hang")
    except Exception as e:
        s["exceptions"].append(f"{type(e).__name__}: {e}")
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-3000:]
        rec["fatal"] = True
    s["seconds"] = x.clock() - t0
    steps.sort()
    if steps:
        s["max_step_s"] = steps[-1]
        s["p99_step_s"] = steps[min(len(steps) - 1, int(0.99 * len(steps)))]
    slow.sort(reverse = True)
    s["slowest"] = [(t, op) for t, op in slow[:10]] + [(t, "iterate") for t in steps[-5:][::-1]]
    s["slowest"].sort(reverse = True)
    stats1 = x.stats()
    for k in ("side", "stagger", "tail", "disk", "uncache", "recurrent"):
        s[k] = numeric_delta(stats0.get(k), stats1.get(k))
    s["free_min"] = (x.memory() or {}).get("free_min")
    s["state"] = x.state()
    return rec


# --------------------------------------------------------------------------------------------
#   The process
# --------------------------------------------------------------------------------------------

class Child:

    def __init__(self, spec: dict):
        self.spec = spec
        self.proc_dir = spec["proc_dir"]
        self.records_path = os.path.join(self.proc_dir, "records.jsonl")
        self.watch = Watch(self.proc_dir, spec["step_timeout"])
        self.records = {}
        self.gen = None
        self.gen_base = {}
        self.broken = None
        self.spent = {}
        self.free_min = {}

    # ---------------------------------------------------------------- records
    def write(self, rec: dict):
        try:
            EC.append_jsonl(self.records_path, rec)
        except ValueError:
            # (NaN and inf are written as JSON's extensions; anything else is a bug of ours)
            EC.append_jsonl(self.records_path, json.loads(json.dumps(rec, default = str)))
        if rec.get("id"):
            self.records[rec["id"]] = rec

    def forced_tokens(self, r: dict) -> list:
        if r is None:
            raise RunError("no forced tokens")
        if "tokens" in r:
            return r["tokens"]
        if "missing" in r:
            raise RunError(f"mancano i token imposti di {r['missing']} (quel processo non li ha prodotti)")
        rec = self.records.get(r["run"])
        if rec is None or not rec.get("ok"):
            raise RunError(f"mancano i token imposti: il giro {r['run']} non c'è o è fallito")
        toks = (rec.get("jobs") or {}).get(r["job"], {}).get("tokens") or []
        if r.get("limit"):
            toks = toks[:r["limit"]]
        return toks

    # ---------------------------------------------------------------- library facts
    def start_record(self):
        import torch
        import exllamav3
        from exllamav3 import Generator
        try:
            from exllamav3.version import __version__ as version
        except Exception:
            version = "?"
        try:
            import exllamav3_ext
            ext = getattr(exllamav3_ext, "__file__", "?")
        except Exception:
            ext = "?"
        lib = os.path.dirname(os.path.abspath(exllamav3.__file__))
        fork_path = os.path.abspath(self.spec["fork_path"])
        fork_features = hasattr(Generator, "prefill_pipeline_state")
        problem = None
        under_fork = lib.startswith(fork_path + os.sep)
        if self.spec["library"] == "fork" and not (fork_features and under_fork):
            problem = f"doveva importare il fork da {fork_path}, ha importato {lib}"
        if self.spec["library"] == "stock" and (fork_features or under_fork):
            problem = f"doveva importare la wheel installata, ha importato {lib} (con le funzioni del fork)"
        rec = {
            "op": "start", "id": "start", "ok": problem is None, "library": self.spec["library"], "library_path": lib,
            "version": version, "ext": ext, "fork_features": fork_features, "library_problem": problem,
            "torch": torch.__version__, "python": sys.version.split()[0], "pid": os.getpid(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")},
        }
        self.write(rec)
        return problem is None

    # ---------------------------------------------------------------- loading
    def load(self):
        import torch
        sys.path.insert(0, self.spec["server_path"])
        from openai.config import parse_args
        from openai.model_container import ModelContainer
        from exllamav3.modules import block_sparse_mlp_cpu as bsm
        self.torch = torch
        self.bsm = bsm
        cfg = parse_args(self.spec["server_argv"])
        container = ModelContainer(cfg)
        t0 = time.time()
        self.watch.scenario = None
        with self.watch.guard("load", self.spec["load_timeout"]):
            container.load_model_sync()
        self.cfg, self.container = cfg, container
        self.model, self.tokenizer = container.model, container.tokenizer
        # The library's own sweep, past the wrapper of the server's expert tier (which saves the
        # placement file after every sweep it sees); during the compared runs nothing sweeps
        loaded = bsm.run_pending_swap_sweeps
        self.sweep_loaded = loaded
        self.sweep_lib = getattr(loaded, "_exl3srv_orig", loaded)
        if self.spec.get("sweeps", "frozen") == "frozen":
            bsm.run_pending_swap_sweeps = lambda infer_params: None
        self.devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self.vocab = getattr(self.tokenizer, "actual_vocab_size", None)
        chat = getattr(self.model, "default_chat_prompt", None)
        self.prompts = Prompts(torch, self.tokenizer, chat, self.spec["prompts"], self.forced_tokens)
        rec = {"op": "load", "id": "load", "ok": True, "seconds": time.time() - t0, "placement": self.placement(),
               "split_modules": len(self._split_modules()), "free": self._free(), "vocab": self.vocab,
               "max_batch_size": container.max_batch_size, "draft": container.draft_model is not None,
               "sweeps": self.spec.get("sweeps", "frozen"), "qsa_raw_tail": self.qsa_raw_tail()}
        self.write(rec)

    def qsa_raw_tail(self) -> dict:
        """The QSA raw-key ring of the caches as loaded (Cache.qsa_raw_tail, cache/qsa.py; None on a
        library without it) and the VRAM it saves per card, bytes"""
        c = self.container
        out = {"main": getattr(c.cache, "qsa_raw_tail", None), "draft": getattr(c.draft_cache, "qsa_raw_tail", None),
               "layers": 0, "saved": {}}
        for cache in (c.cache, c.draft_cache):
            for layer in (getattr(cache, "layers", None) or {}).values():
                saved = getattr(layer, "raw_tail_saved_bytes", None)
                raw = getattr(layer, "raw_k", None)
                if saved is None or raw is None or not saved():
                    continue
                d = raw.device.index
                out["saved"][d] = out["saved"].get(d, 0) + int(saved())
                out["layers"] += 1
        return out

    def _split_modules(self):
        ip = getattr(getattr(self.model, "config", None), "infer_params", None)
        return list(getattr(ip, "moe_cpu_swap_modules", None) or [])

    def placement(self):
        mods = self._split_modules()
        if not mods:
            return None
        h = hashlib.sha1()
        with self.torch.inference_mode():
            for m in mods:
                mp = getattr(m, "_split_map", None)
                if mp is not None:
                    h.update(self._tensor_bytes(mp))
        return h.hexdigest()[:16]

    def _free(self):
        out = {}
        for d in self.devices:
            try:
                out[d] = int(self.torch.cuda.mem_get_info(d)[0])
            except Exception:
                pass
        return out

    # ---------------------------------------------------------------- generators
    def make_generator(self, step: dict):
        from exllamav3 import Generator
        torch = self.torch
        cfg, c = self.cfg, self.container
        if self.gen is not None:
            self.close_generator("replaced")
        for k, v in (step.get("env") or {}).items():
            if v == "":
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        kw = dict(model = self.model, cache = c.cache, tokenizer = self.tokenizer, max_batch_size = c.max_batch_size,
                  max_chunk_size = cfg.max_chunk_size)
        if cfg.cpu_cache_size and cfg.cpu_cache_size > 0:
            kw["cpu_cache_size"] = int(cfg.cpu_cache_size * 1024 ** 3)
        if c.draft_model is not None:
            kw.update(draft_model = c.draft_model, draft_cache = c.draft_cache, num_draft_tokens = cfg.draft_num_tokens)
        if cfg.recurrent_cache_size is not None:
            kw["recurrent_cache_size"] = int(cfg.recurrent_cache_size * 1024 ** 3)
        if cfg.recurrent_checkpoint_interval is not None:
            kw["recurrent_checkpoint_interval"] = cfg.recurrent_checkpoint_interval
        if cfg.recurrent_checkpoint_interval_pp is not None:
            kw["recurrent_checkpoint_interval_pp"] = cfg.recurrent_checkpoint_interval_pp
        disk = step.get("disk")
        if disk:
            root = os.path.join(self.spec["disk_root"], disk)
            os.makedirs(root, exist_ok = True)
            kw["disk_cache_dir"] = root
            kw["disk_cache_size"] = int(float(self.spec.get("disk_budget_gb", 16)) * 1024 ** 3)
        sw = step.get("switches") or {}
        params = inspect.signature(Generator.__init__).parameters
        for key, arg in (("stagger", "stagger_decode"), ("thread", "prefill_thread"), ("pipeline", "prefill_pipeline"),
                         ("tail", "recurrent_tail_checkpoint")):
            if sw.get(key):
                if arg not in params:
                    raise RunError(f"questa libreria non ha Generator({arg})")
                kw[arg] = sw[key]
        t0 = time.time()
        with self.watch.guard("generator"):
            gen = Generator(**kw)
        self.gen = gen
        self.broken = None
        warnings = []
        attached = False
        if self.spec.get("sweeps", "frozen") == "server":
            # As the server does when it makes its generator: the expert tier wraps the queue's
            # drain, and with --moe-swap-inline false a sweep runs only there (without this the
            # tier would defer every sweep for good)
            tier = getattr(c, "moe_tier", None)
            if tier is not None:
                try:
                    tier.attach_generator(gen, None)
                    attached = True
                except Exception as e:
                    warnings.append(f"il tier degli esperti del server non si aggancia al generatore: {type(e).__name__}: {e}")
        self.gen_base = {"stagger": getattr(gen, "stagger_decode", False), "thread": getattr(gen, "prefill_thread", False),
                         "pipeline": getattr(gen, "prefill_pipeline", False),
                         "tail": getattr(gen, "tail_checkpoint_interval", 0), "chunk": gen.max_chunk_size,
                         "stagger_min_jobs": getattr(gen, "stagger_min_jobs", None)}
        state = {}
        for name in ("stagger_decode", "prefill_thread", "prefill_pipeline", "tail_checkpoint_interval"):
            if hasattr(gen, name):
                state[name] = getattr(gen, name)
        if hasattr(self.bsm, "hist_prefill_weight"):
            state["hist_prefill_weight"] = self.bsm.hist_prefill_weight()
        if hasattr(c.cache, "qsa_raw_tail"):
            state["qsa_raw_tail"] = c.cache.qsa_raw_tail
        if gen.disk_page_cache is not None:
            state["disk_uncache"] = getattr(gen.disk_page_cache, "uncache", None)
        return {"op": "generator", "id": step["id"], "scenario": step.get("scenario"), "ok": True,
                "seconds": time.time() - t0, "switches_state": state, "disk": disk, "kwargs": sorted(kw),
                "tier_attached": attached, "warnings": warnings}

    def close_generator(self, why: str):
        gen, self.gen = self.gen, None
        if gen is None:
            return 0.0
        t0 = time.time()
        if self.broken and gen.disk_page_cache is not None:
            # A failed generator writes nothing more (upstream's own reason for abort())
            try:
                gen.disk_page_cache.abort()
            except Exception:
                pass
        with self.watch.guard("close", max(self.spec["step_timeout"], 900.0)):
            gen.close()
        return time.time() - t0

    def apply_switches(self, sw: dict):
        gen, base = self.gen, self.gen_base
        pipeline = sw.get("pipeline", base["pipeline"])
        thread = sw.get("thread", base["thread"]) or pipeline
        for key, attr, val in (("stagger", "stagger_decode", sw.get("stagger", base["stagger"])),
                               ("thread", "prefill_thread", thread), ("pipeline", "prefill_pipeline", pipeline)):
            if not hasattr(gen, attr):
                if val:
                    raise RunError(f"questa libreria non ha {attr}")
                continue
            setattr(gen, attr, bool(val))
        tail = int(sw.get("tail", base["tail"]) or 0)
        if hasattr(gen, "tail_checkpoint_interval"):
            gen.tail_checkpoint_interval = tail if gen.recurrent_cache is not None else 0
        elif tail:
            raise RunError("questa libreria non ha il checkpoint di coda")
        gen.max_chunk_size = int(sw.get("chunk") or base["chunk"])
        if base.get("stagger_min_jobs") is not None:
            gen.stagger_min_jobs = int(sw.get("stagger_min_jobs") or base["stagger_min_jobs"])

    def forget(self):
        """Make the next run ingest its prompts from their first token: a recurrent model
        resumes a cached prompt only from a page with a stashed checkpoint, so every checkpoint
        in RAM goes, through the library's own prune with a page table that calls them all
        stranded (as the fork's benches did; the page table itself is left alone)."""
        gen = self.gen
        rc = getattr(gen, "recurrent_cache", None)
        if rc is None:
            return None
        real = rc.pagetable
        rc.pagetable = SimpleNamespace(is_resumable = lambda key: False)
        try:
            with self.watch.guard("forget"):
                n = rc.prune_stranded()
        finally:
            rc.pagetable = real
        return n

    def stats(self) -> dict:
        gen = self.gen
        out = {}
        for key, attr in (("side", "side_stats"), ("stagger", "stagger_stats"), ("tail", "tail_stats")):
            v = getattr(gen, attr, None)
            if isinstance(v, dict):
                out[key] = dict(v)
        disk = getattr(gen, "disk_page_cache", None)
        if disk is not None:
            out["disk"] = dict(disk.metrics)
            if isinstance(getattr(disk, "uncache_stats", None), dict):
                out["uncache"] = dict(disk.uncache_stats)
        rc = getattr(gen, "recurrent_cache", None)
        if rc is not None and isinstance(getattr(rc, "metrics", None), dict):
            out["recurrent"] = dict(rc.metrics)
        pt = getattr(gen, "pagetable", None)
        if pt is not None and isinstance(getattr(pt, "metrics", None), dict):
            out["pagetable"] = dict(pt.metrics)
        return out

    def disk_metrics(self):
        disk = getattr(self.gen, "disk_page_cache", None)
        return dict(disk.metrics) if disk is not None else None

    def state(self) -> dict:
        gen = self.gen
        out = {}
        for key, attr in (("thread", "prefill_thread_state"), ("thread_reason", "prefill_thread_reason"),
                          ("pipeline", "prefill_pipeline_state"), ("pipeline_reason", "prefill_pipeline_reason")):
            try:
                out[key] = getattr(gen, attr)
            except Exception:
                out[key] = None
        out["stagger"] = bool(getattr(gen, "stagger_decode", False))
        out["tail"] = getattr(gen, "tail_checkpoint_interval", None)
        out["head_first"] = getattr(gen, "_pipe_head_first", None)
        return out

    def reset_memory(self):
        self.free_min = {}
        for d in self.devices:
            try:
                self.torch.cuda.reset_peak_memory_stats(d)
            except Exception:
                pass

    def sample_memory(self):
        for d in self.devices:
            try:
                free = int(self.torch.cuda.mem_get_info(d)[0])
            except Exception:
                continue
            if d not in self.free_min or free < self.free_min[d]:
                self.free_min[d] = free

    def memory(self) -> dict:
        out = {"peak": {}, "free_end": self._free(), "free_min": dict(self.free_min)}
        for d in self.devices:
            try:
                out["peak"][d] = int(self.torch.cuda.max_memory_allocated(d))
            except Exception:
                pass
        return out

    def env(self):
        """What the Runner needs, over this process."""
        from exllamav3 import Job
        from exllamav3.generator.sampler import GreedySampler

        def make_job(name, ids, max_new, max_rq_tokens = None):
            kw = {"max_rq_tokens": int(max_rq_tokens)} if max_rq_tokens else {}
            return Job(input_ids = ids, max_new_tokens = int(max_new), sampler = GreedySampler(), identifier = name, **kw)

        x = SimpleNamespace(
            torch = self.torch, clock = time.time, watch = self.watch, make_job = make_job,
            prompt_ids = self.prompts.ids, prompt_ids_text = self.prompts.encode,
            forced_tokens = self.forced_tokens, digester = lambda: Digester(self.torch, self.vocab, self.spec.get("top_k", 4)),
            apply_switches = self.apply_switches, forget = self.forget, stats = self.stats, state = self.state,
            placement = self.placement if self.spec.get("sweeps", "frozen") == "frozen" else (lambda: None),
            reset_memory = self.reset_memory, sample_memory = self.sample_memory, memory = self.memory,
            disk_metrics = self.disk_metrics, run_timeout = 3 * 3600.0,
            max_rq_tokens = getattr(self.container, "max_rq_tokens", None),
        )
        return x

    # ---------------------------------------------------------------- the other operations
    def disk_scan(self, step: dict) -> dict:
        gen = self.gen
        rec = {"op": "disk_scan", "id": step["id"], "scenario": step.get("scenario"), "ok": False}
        t0 = time.time()
        if gen is not None and gen.disk_page_cache is not None:
            with self.watch.guard("flush", max(self.spec["step_timeout"], 900.0)):
                rec["flushed"] = gen.flush_cache_to_disk(wait = True)
            rec["disk"] = dict(gen.disk_page_cache.metrics)
        with self.watch.guard("scan", max(self.spec["step_timeout"], 900.0)):
            rec["scan"] = EC.scan_disk(os.path.join(self.spec["disk_root"], step["disk"]))
        rec["seconds"] = time.time() - t0
        rec["ok"] = True
        return rec

    def reset_hist(self, step: dict) -> dict:
        mods = self._split_modules()
        with self.torch.inference_mode():
            for m in mods:
                if getattr(m, "_split_hist", None) is not None:
                    m._split_hist.zero_()
                pf = getattr(m, "_split_hist_pf", None)
                if pf is not None:
                    pf.zero_()
            if mods and hasattr(mods[0], "_split_sweep_layer_reset"):
                mods[0]._split_sweep_layer_reset()
        return {"op": "reset_hist", "id": step["id"], "scenario": step.get("scenario"), "ok": True,
                "modules": len(mods), "placement": self.placement()}

    def _tensor_bytes(self, t) -> bytes:
        """The raw bytes of a tensor whatever its dtype (numpy has no bfloat16)."""
        t = t.detach().to("cpu").contiguous()
        return t.reshape(-1).view(self.torch.uint8).numpy().tobytes() if t.numel() else b""

    def _layer_hashes(self, attr: str) -> list:
        out = []
        with self.torch.inference_mode():
            for m in self._split_modules():
                t = getattr(m, attr, None)
                if t is None:
                    out.append(None)
                    continue
                out.append(hashlib.sha1(self._tensor_bytes(t)).hexdigest()[:16])
        return out

    def sweep(self, step: dict) -> dict:
        mods = self._split_modules()
        rec = {"op": "sweep", "id": step["id"], "scenario": step.get("scenario"), "ok": False}
        if not mods:
            rec["error"] = "il modello non ha strati con piazzamento dinamico"
            return rec
        rec["hist_pre"] = self._layer_hashes("_split_hist")
        with self.torch.inference_mode():
            before = [m._split_map.detach().to("cpu").clone() for m in mods]
        ip = self.model.config.infer_params
        ip.moe_cpu_swap_pending = True
        t0 = time.time()
        with self.watch.guard("sweep", max(self.spec["step_timeout"], 900.0)):
            self.sweep_lib(ip)
        rec["seconds"] = time.time() - t0
        with self.torch.inference_mode():
            after = [m._split_map.detach().to("cpu").clone() for m in mods]
        rec["moved"] = [int((a != b).sum()) // 2 for a, b in zip(before, after)]
        rec["total_moved"] = sum(rec["moved"])
        rec["maps"] = self._layer_hashes("_split_map")
        rec["hist_post"] = self._layer_hashes("_split_hist")
        try:
            with open(os.path.join(self.proc_dir, f"{step['id']}.maps.json"), "w") as f:
                json.dump([a.tolist() for a in after], f)
        except OSError:
            pass
        rec["ok"] = True
        return rec

    # ---------------------------------------------------------------- the steps
    def step(self, st: dict, x):
        op = st["op"]
        if op == "generator":
            return self.make_generator(st)
        if op == "close":
            t = self.close_generator("asked")
            return {"op": "close", "id": st["id"], "scenario": st.get("scenario"), "ok": True, "seconds": t}
        if self.gen is None and op in ("run", "stress", "reset_hist", "sweep"):
            raise RunError("nessun generatore")
        if op == "run":
            if self.broken:
                return {"op": "run", "id": st["id"], "scenario": st.get("scenario"), "ok": False,
                        "error": f"saltato: il generatore è fallito prima ({self.broken})"}
            rec = Runner(x).run(st)
            if rec.get("fatal"):
                self.broken = rec.get("error")
            return rec
        if op == "stress":
            rec = run_stress(x, st)
            if rec.get("fatal"):
                self.broken = rec.get("error")
            return rec
        if op == "disk_scan":
            return self.disk_scan(st)
        if op == "reset_hist":
            return self.reset_hist(st)
        if op == "sweep":
            return self.sweep(st)
        raise RunError(f"operazione sconosciuta {op}")

    def main(self) -> int:
        if not self.start_record():
            return 3
        try:
            self.load()
        except Exception as e:
            self.write({"op": "load", "id": "load", "ok": False, "error": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()[-3000:]})
            return 4
        x = Gen(self)
        budgets = {int(k): float(v) for k, v in (self.spec.get("budgets") or {}).items()}
        for st in self.spec["steps"]:
            s = st.get("scenario")
            self.watch.scenario = s
            self.watch.run = st.get("id")
            self.watch.beat(st["op"], force = True)
            if s in budgets and self.spent.get(s, 0.0) > budgets[s]:
                self.write({"op": st["op"], "id": st.get("id"), "scenario": s, "ok": False,
                            "error": "saltato: tempo dello scenario esaurito"})
                continue
            t0 = time.time()
            try:
                rec = self.step(st, x)
            except Exception as e:
                rec = {"op": st["op"], "id": st.get("id"), "scenario": s, "ok": False,
                       "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
                if st["op"] in ("generator", "close"):
                    self.broken = rec["error"]
            self.spent[s] = self.spent.get(s, 0.0) + (time.time() - t0)
            self.write(rec)
            print(f" -- exact check child: {st.get('id')} {'ok' if rec.get('ok') else 'FAILED: ' + str(rec.get('error'))}"
                  f" ({time.time() - t0:.1f} s)", flush = True)
        self.watch.scenario = None
        self.watch.run = None
        try:
            if self.gen is not None:
                self.close_generator("end")
        except Exception as e:
            self.write({"op": "close", "id": "final.close", "ok": False, "error": f"{type(e).__name__}: {e}"})
        self.write({"op": "end", "id": "end", "ok": True})
        # Orderly release of the cards and of the CPU expert worker. The expert tier is not
        # closed: its close() saves the placement (to this process's copy; nothing is gained)
        try:
            with self.watch.guard("unload", 600):
                if self.container.draft_model is not None:
                    self.container.draft_model.unload()
                self.container.model.unload()
        except Exception as e:
            print(f" -- exact check child: unload failed ({e!r}); exiting anyway", flush = True)
        return 0


class Gen(SimpleNamespace):
    """The Runner's environment over a Child: the generator is read when asked, since a step can
    replace it."""

    def __init__(self, child: Child):
        base = child.env()
        super().__init__(**vars(base))
        self._child = child

    @property
    def gen(self):
        return self._child.gen


def main(argv = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required = True)
    args = ap.parse_args(argv)
    with open(args.spec) as f:
        spec = json.load(f)
    # The server's and the library's log lines into child.log, as the server's own log has them
    # (the expert tier's load summary, and the split it suggests when a layer does not fit)
    logging.basicConfig(level = logging.INFO, stream = sys.stdout,
                        format = "%(asctime)s %(levelname)s %(name)s: %(message)s")
    child = Child(spec)
    try:
        code = child.main()
    except Exception as e:
        traceback.print_exc()
        try:
            child.write({"op": "crash", "id": "crash", "ok": False, "error": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
        code = 5
    sys.stdout.flush()
    sys.stderr.flush()
    # No interpreter teardown: the server's exit hooks (a placement save among them) do not
    # belong here, and the CPU expert worker ends with its parent (PDEATHSIG)
    os._exit(code)


if __name__ == "__main__":
    main()
