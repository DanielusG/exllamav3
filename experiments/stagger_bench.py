#!/usr/bin/env python
"""
Bench of the staggered decode (Generator.stagger_decode) on the production model.

The model is loaded exactly as the OpenAI server loads it (same flags, through the server's own
ModelContainer), then a raw Generator is driven from this process. For 2, 3 and 4 jobs
generating at the same time it gives three things:

  - speed: the regular decode (one batch) against the staggered one, greedy sampling;
  - same computation: every job is generated again with its own tokens forced on it, in every
    mode, and the probabilities are compared position by position against the job's group
    alone. Two identical runs of this model do not give the same greedy tokens (the first
    near-tie flips and the texts part ways), so comparing tokens proves nothing; forced tokens
    keep every run on the same sequence and the group alone, run twice, gives the noise floor;
  - where the time goes: a short run with CUDA events around each card's part of the pass and
    around the handoffs to the CPU expert worker (stream time, no host synchronization).

Before measuring, the expert placement is adapted to the bench's own prompts with the
library's sweeps (the placement loaded from the server's file follows production traffic). The
server's placement file is never written.

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
import traceback
from types import SimpleNamespace

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

NAMES = "abcd"
_MISSING = object()


def _own_attr(obj, name: str):
    """The attribute as set on the instance itself (_MISSING when it comes from the class)."""
    return getattr(obj, "__dict__", {}).get(name, _MISSING)


def _restore_attr(obj, name: str, prev):
    if prev is _MISSING:
        try:
            delattr(obj, name)
        except AttributeError:
            pass
    else:
        setattr(obj, name, prev)


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
        self._prev = _own_attr(obj, name)
        orig = getattr(obj, name)
        timer = self

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                timer.total += time.perf_counter() - t0
                timer.calls += 1

        setattr(obj, name, timed)

    def take(self):
        total, calls = self.total, self.calls
        self.total, self.calls = 0.0, 0
        return total, calls

    def remove(self):
        _restore_attr(self._obj, self._name, self._prev)


class StreamProf:
    """
    Stream-time brackets (CUDA events, no host synchronization) for a profiled run: around each
    card's part of a decode pass and, inside it, around the two handoffs of a MoE layer to the
    CPU expert worker. The collect bracket (wait for the worker's flag, then the readback) is
    the stall a card could not hide behind its own expert work; the issue bracket is the wait
    for the slot's previous job plus the staging kernel.

    A pass bracket starts when the host starts issuing and ends when the card is done with what
    was issued: in a regular step the second card's bracket therefore includes the first half
    of the pass, which runs on the other card.

    Installed for the profiled runs only (begin/end): the wrappers cost a few events per layer.
    """

    def __init__(self, torch, model, gen, stage_devs):
        self.torch = torch
        self.model = model
        self.gen = gen
        self.stage_devs = stage_devs        # (first card, second card) of a pass; None when unknown
        self.step = 0                       # index of the iterate() call in progress
        self.fallbacks = 0                  # handoffs that left the fused path (not bracketed)
        self.pending = []                   # (kind, device index, step, start event, end event)
        self.resolved = []                  # (kind, device index, step, ms)
        self._undo = []

    def _swap(self, obj, name: str, make):
        prev = _own_attr(obj, name)
        setattr(obj, name, make(getattr(obj, name)))
        self._undo.append((obj, name, prev))

    def _mark(self, dev):
        with self.torch.cuda.device(dev):
            ev = self.torch.cuda.Event(enable_timing = True)
            ev.record()
        return ev

    def _bracket(self, kind: str, devs: list, fn):
        starts = [(d, self._mark(d)) for d in devs]
        r = fn()
        for d, ev0 in starts:
            self.pending.append((kind, d.index, self.step, ev0, self._mark(d)))
        return r

    def begin(self):
        prof = self
        self.pending, self.resolved, self.fallbacks, self.step = [], [], 0, 0
        d0, d1 = self.stage_devs
        both = [d for d in (d0, d1) if d is not None]

        def issue(orig):
            def wrapped(layer_idx, y, *args, **kwargs):
                r = prof._bracket("issue", [y.device], lambda: orig(layer_idx, y, *args, **kwargs))
                if r is None:
                    prof.fallbacks += 1
                return r
            return wrapped

        def collect(orig):
            def wrapped(handle, *args, **kwargs):
                # handle: (seq, slot, rows, h, ho, device, counts)
                return prof._bracket("collect", [handle[5]], lambda: orig(handle, *args, **kwargs))
            return wrapped

        def forward(orig):
            def wrapped(*args, **kwargs):
                return prof._bracket("pass", both, lambda: orig(*args, **kwargs))
            return wrapped

        def stagger_issue(orig):
            def wrapped(late, early, split):
                devs = []
                if late is not None and d1 is not None:
                    devs.append(d1)
                if early is not None and d0 is not None:
                    devs.append(d0)
                return prof._bracket("pass", devs, lambda: orig(late, early, split))
            return wrapped

        for host in getattr(self.model.config, "moe_cpu_hosts", {}).values():
            if hasattr(host, "submit_issue_fused"):
                self._swap(host, "submit_issue_fused", issue)
            if hasattr(host, "submit_collect_fused"):
                self._swap(host, "submit_collect_fused", collect)
        self._swap(self.model, "forward", forward)
        if hasattr(self.gen, "_stagger_issue"):
            self._swap(self.gen, "_stagger_issue", stagger_issue)

    def end(self):
        for obj, name, prev in reversed(self._undo):
            _restore_attr(obj, name, prev)
        self._undo = []
        for d in range(self.torch.cuda.device_count()):
            self.torch.cuda.synchronize(d)
        for kind, dev, step, ev0, ev1 in self.pending:
            self.resolved.append((kind, dev, step, ev0.elapsed_time(ev1)))
        self.pending = []

    @staticmethod
    def digest(resolved: list, first: int, last: int) -> dict:
        """Per (kind, card): ms and brackets per iterate() call, median and p90 of one bracket,
        over the calls first+1 .. last."""
        calls = max(1, last - first)
        by = {}
        for kind, dev, step, ms in resolved:
            if first < step <= last:
                by.setdefault((kind, dev), []).append(ms)
        out = {}
        for key, vals in by.items():
            vals.sort()
            out[key] = {
                "ms_call": sum(vals) / calls,
                "n_call": len(vals) / calls,
                "med": vals[len(vals) // 2],
                "p90": vals[min(len(vals) - 1, int(len(vals) * 0.9))],
            }
        return out


def bench_revision() -> str:
    """The commit of the checkout this file is in, for the report: a report must say which
    bench wrote it (a checkout that was not updated runs the old one without a word)."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(
            ["git", "-C", here, "log", "-1", "--format=%h %cs"],
            capture_output = True, text = True, timeout = 5,
        )
        if out.returncode == 0 and out.stdout.strip():
            dirty = subprocess.run(
                ["git", "-C", here, "status", "--porcelain", "--", os.path.basename(__file__)],
                capture_output = True, text = True, timeout = 5,
            ).stdout.strip()
            return out.stdout.strip() + (" (file modificato rispetto al commit)" if dirty else "")
    except Exception:
        pass
    return "sconosciuto"


def split_modules(model) -> list:
    """The MoE layers with a dynamic expert placement (empty without one)."""
    ip = getattr(getattr(model, "config", None), "infer_params", None)
    return list(getattr(ip, "moe_cpu_swap_modules", None) or [])


def placement_counts(torch, model):
    """(expert selections served from VRAM, all selections) counted by the split layers since
    the last sweep; None for a model without a dynamic placement. A sweep decays the counts, so
    two readings are comparable only with no sweep in between."""
    mods = split_modules(model)
    if not mods:
        return None
    gpu = total = 0.0
    with torch.inference_mode():
        for m in mods:
            hist = m._split_hist.detach().to("cpu", torch.float64)
            smap = m._split_map.detach().to("cpu")
            total += float(hist.sum())
            gpu += float(hist[smap < m.cpu_split_first].sum())
    return gpu, total


def placement_maps(torch, model) -> list:
    with torch.inference_mode():
        return [m._split_map.detach().to("cpu").clone() for m in split_modules(model)]


def hit_rate(c0, c1):
    if c0 is None or c1 is None:
        return None
    total = c1[1] - c0[1]
    return (c1[0] - c0[0]) / total if total > 0 else None


def stage_devices(torch, model, split):
    """The CUDA device of the modules before and after the stage boundary (None when unknown)."""
    def first(mods):
        for module, _, _ in mods:
            dev = getattr(module, "device", None)
            if dev is not None and torch.device(dev).type == "cuda":
                return torch.device(dev)
        return None
    if not split:
        return first(model.fwd_modules), None
    return first(model.fwd_modules[:split]), first(model.fwd_modules[split:])


class Bench:
    """A loaded model, its generator and the instruments; run() is one generation."""

    def __init__(self, env, gpu):
        self.env = env
        self.gpu = gpu
        self.timers = {}
        for key, obj, name in (("draft", env.gen, "iterate_draftmodel_mtp_gen"), ("forward", env.model, "forward")):
            if hasattr(obj, name):
                self.timers[key] = Timer(obj, name)

    def close(self):
        for t in self.timers.values():
            t.remove()

    def run(self, job_names, stagger: bool, tokens: int, forced: dict | None = None, profile: bool = False):
        """
        Generate for the named prompts at the same time: `tokens` tokens each with greedy
        sampling, or, with `forced` (name -> token ids), exactly those tokens, with the
        probability of each and the two most likely tokens of its position recorded. Returns
        the tokens of each job and the figures of the stretch in which ALL the jobs were
        generating (from the iteration after the last job's first token to the one before the
        first job's last).
        """
        env = self.env
        torch, gen = env.torch, env.gen
        prof = env.prof if profile else None
        gen.stagger_decode = stagger
        stats0 = dict(gen.stagger_stats)
        for t in self.timers.values():
            t.take()
        counts0 = placement_counts(torch, env.model)

        jobs = {}
        for name in job_names:
            kwargs = {}
            n_new = tokens
            if forced is not None:
                n_new = len(forced[name])
                kwargs = {"return_probs": True, "return_top_tokens": 2}
            jobs[name] = env.Job(
                input_ids = env.prompts[name],
                max_new_tokens = n_new,
                sampler = env.sampler_cls(),
                identifier = name,
                **kwargs,
            )
        by_job = {id(job): name for name, job in jobs.items()}
        gen.enqueue(list(jobs.values()))
        if forced is not None:
            for name, job in jobs.items():
                job.constrain_output_now(torch.tensor([forced[name]], dtype = torch.long))

        out = {name: [] for name in jobs}
        probs = {name: [] for name in jobs}
        top = {name: [] for name in jobs}       # per position: ([token 1, token 2], [p1, p2])
        done = set()
        timeline = []       # (t after the iteration, tokens so far per job, jobs done)
        marks = []          # timer and stagger totals after each iteration
        if prof is not None:
            prof.begin()
        try:
            while gen.num_remaining_jobs():
                if prof is not None:
                    prof.step = len(timeline)
                results = gen.iterate()
                now = time.perf_counter()
                for r in results:
                    name = by_job.get(id(r.get("job")))
                    if r.get("stage") == "error":
                        raise RuntimeError(f"job {name} failed: {r.get('error')!r}")
                    if r.get("stage") != "streaming" or name is None:
                        continue
                    ids = r.get("token_ids")
                    if ids is not None:
                        out[name] += ids.reshape(-1).tolist()
                    p = r.get("token_probs")
                    if p is not None:
                        probs[name] += p.reshape(-1).tolist()
                    kt, kp = r.get("top_k_tokens"), r.get("top_k_probs")
                    if kt is not None and kp is not None:
                        top[name] += list(zip(
                            kt.reshape(-1, kt.shape[-1]).tolist(),
                            kp.reshape(-1, kp.shape[-1]).tolist(),
                        ))
                    if r.get("eos"):
                        done.add(name)
                timeline.append((now, {n: j.new_tokens + getattr(j, "rq_new_tokens", 0) for n, j in jobs.items()}, len(done)))
                marks.append((
                    {k: (t.total, t.calls) for k, t in self.timers.items()},
                    dict(gen.stagger_stats),
                ))
        finally:
            if prof is not None:
                prof.end()

        # The stretch with every job generating
        first = next((i for i, (_, toks, _) in enumerate(timeline) if all(v >= 1 for v in toks.values())), None)
        last = next((i for i, (_, _, n_done) in enumerate(timeline) if n_done > 0), len(timeline)) - 1
        res = {
            "names": list(job_names),
            "stagger": stagger,
            "tokens": out,
            "probs": probs,
            "top": top,
            "accepted": sum(j.accepted_draft_tokens for j in jobs.values()),
            "rejected": sum(j.rejected_draft_tokens for j in jobs.values()),
            "hit": hit_rate(counts0, placement_counts(torch, env.model)),
            "window": None,
            "prof": None,
            "prof_fallbacks": prof.fallbacks if prof is not None else 0,
        }
        if first is not None and last > first + 4:
            t_a, toks_a, _ = timeline[first]
            t_b, toks_b, _ = timeline[last]
            n_tok = sum(toks_b.values()) - sum(toks_a.values())
            steps = last - first
            dur = max(t_b - t_a, 1e-9)
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
                "gpu": self.gpu.mean(t_a, t_b) if self.gpu is not None else None,
            }
            if prof is not None:
                res["prof"] = StreamProf.digest(prof.resolved, first, last)
        res["stagger_total"] = {k: gen.stagger_stats[k] - stats0[k] for k in stats0}
        return res


def compare_forced(ref: dict, run: dict, name: str):
    """
    One job of a forced run against the same job of a reference run (same forced tokens):
    from which position and how far the probability of the forced token moved, in how many
    positions the most likely token changed and how wide the reference's lead was where it did,
    and in how many positions of the run the forced token is the most likely one. None when
    either run recorded nothing.
    """
    p_r, p_x = ref["probs"].get(name, []), run["probs"].get(name, [])
    t_r, t_x = ref["top"].get(name, []), run["top"].get(name, [])
    forced = run["tokens"].get(name, [])
    n = min(len(p_r), len(p_x), len(t_r), len(t_x), len(forced))
    if n == 0:
        return None
    delta = [abs(a - b) for a, b in zip(p_r[:n], p_x[:n])]
    flips = [i for i in range(n) if t_r[i][0][0] != t_x[i][0][0]]
    return {
        "n": n,
        "first_delta": next((i for i, d in enumerate(delta) if d != 0.0), None),
        "max": max(delta),
        "mean": statistics.fmean(delta),
        "flips": len(flips),
        "first_flip": flips[0] if flips else None,
        "margin": max((t_r[i][1][0] - t_r[i][1][1] for i in flips), default = None),
        "on_top": sum(1 for i in range(n) if t_x[i][0][0] == forced[i]),
    }


def groups_of(n: int) -> list:
    """The two groups a staggered run of the first n jobs settles into (Generator._stagger_pick)."""
    first = n - n // 2
    return [NAMES[:first], NAMES[first:n]]


def fmt(x, nd = 1):
    return "-" if x is None else f"{x:.{nd}f}"


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def timing_row(label: str, r: dict) -> list:
    w = r["window"]
    if w is None:
        return [label, "finestra troppo corta"] + ["-"] * 10
    acc = r["accepted"] + r["rejected"]
    g = w["gpu"]
    st = w["stagger"]
    steps = max(1, w["steps"])
    if r["stagger"]:
        host = (f"bozza+ingressi {1000 * st['t_launch'] / steps:.2f}, emissione {1000 * st['t_issue'] / steps:.2f}, "
                f"campionamento {1000 * st['t_sample'] / steps:.2f}")
    else:
        tm = w["timers_ms_step"]
        draft, forward = tm.get("draft", 0.0), tm.get("forward", 0.0)
        host = f"bozza {draft:.2f}, emissione {forward:.2f}, resto {w['ms_step'] - draft - forward:.2f}"
    return [
        label,
        fmt(w["tok_s"]),
        " / ".join(fmt(v) for v in w["per_job"].values()),
        fmt(w["ms_step"], 2),
        fmt(w["tok_step"], 2),
        fmt(r["accepted"] / acc if acc else None, 3),
        fmt(r["hit"], 3),
        " + ".join(f"{u:.0f}%" for u, _, _ in g) if g else "-",
        " + ".join(f"{p:.0f} W" for _, p, _ in g) if g else "-",
        g[0][2] if g else 0,
        fmt(w["seconds"], 1),
        host,
    ]


TIMING_HEADER = [
    "modo", "tok/s totali", "tok/s per job", "ms per chiamata", "token per chiamata", "accettazione bozza",
    "selezioni servite dalla VRAM", "uso schede", "potenza", "campioni GPU", "finestra s",
    "tempo host per chiamata (ms)",
]


def check_row(name: str, label: str, c) -> list:
    if c is None:
        return [name, label, "nessun dato", "-", "-", "-", "-", "-", "-"]
    return [
        name,
        label,
        c["n"],
        c["on_top"],
        c["flips"] if not c["flips"] else f"{c['flips']} (primo all'indice {c['first_flip']})",
        fmt(c["margin"], 4),
        "-" if c["first_delta"] is None else c["first_delta"],
        "0 (identici)" if c["max"] == 0 else f"{c['max']:.4f}",
        f"{c['mean']:.5f}",
    ]


CHECK_HEADER = [
    "job", "confronto", "posizioni", "token imposto più probabile", "top-1 cambiati",
    "margine massimo dove cambia", "prima posizione con p diversa", "scarto massimo di p", "scarto medio di p",
]


def prof_row(label: str, r: dict) -> list:
    w, p = r["window"], r["prof"]
    if w is None or p is None:
        return [label, "finestra troppo corta"] + ["-"] * 7
    cards = sorted({dev for _, dev in p})

    def cell(kind: str, field: str, nd: int) -> str:
        return " / ".join(fmt(p.get((kind, d), {}).get(field), nd) for d in cards) if cards else "-"

    return [
        label,
        fmt(w["ms_step"], 2),
        cell("pass", "ms_call", 2),
        cell("collect", "ms_call", 2),
        cell("issue", "ms_call", 2),
        cell("collect", "n_call", 1),
        cell("collect", "med", 3),
        cell("collect", "p90", 3),
        r["prof_fallbacks"],
    ]


PROF_HEADER = [
    "giro", "ms per chiamata", "passo (scheda 0 / 1)", "attesa del worker (scheda 0 / 1)",
    "consegna al worker (scheda 0 / 1)", "raccolte per chiamata (scheda 0 / 1)",
    "una raccolta, mediana ms (scheda 0 / 1)", "una raccolta, p90 ms (scheda 0 / 1)", "consegne fuori dal percorso fuso",
]


def measure(env, args, gpu = None) -> str:
    """Everything between a loaded model and the text of the report."""
    torch, gen, model = env.torch, env.gen, env.model
    names = NAMES[:env.max_jobs]
    n_jobs_list = [n for n in env.n_jobs_list if 2 <= n <= env.max_jobs]
    bench = Bench(env, gpu)
    notes = []          # problems of the measure itself, shown right under the summary

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

    sections = []

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
        "Quota delle selezioni di esperti che cadono su esperti residenti in VRAM (il resto lo calcola il "
        "worker su CPU). Il piazzamento caricato dal file del server segue il traffico di produzione; qui "
        f"viene adattato ai prompt della misura: ogni giro genera {args.bench_adapt_tokens} token per job "
        "con tutti i job insieme, poi uno sweep della libreria scambia gli esperti. Dopo l'ultimo giro il "
        "piazzamento resta fermo per tutta la misura. Il file di piazzamento del server non viene scritto "
        "(gli sweep non passano dal tier del server, che è l'unico a salvarlo).",
    ]
    if adapt_rows:
        text.append(table(["giro", "selezioni servite dalla VRAM nel giro", "esperti scambiati dallo sweep che segue"], adapt_rows))
    if adapt_note:
        text.append(adapt_note)
    sections.append("## Piazzamento degli esperti\n\n" + "\n\n".join(text))

    # 2. Warm-up of every shape the measure uses
    def warmup():
        bench.run(names[:1], False, args.bench_warmup)
        for n in n_jobs_list:
            for g in groups_of(n):
                bench.run(g, False, args.bench_warmup)
            bench.run(names[:n], False, args.bench_warmup)
            bench.run(names[:n], True, args.bench_warmup)

    phase("warm-up", warmup)

    # 3. Speed
    solo = {}

    def one_at_a_time():
        for name in names:
            solo[name] = bench.run(name, False, args.bench_tokens)

    phase("one job at a time", one_at_a_time)
    rows = []
    for name, r in solo.items():
        w = r["window"]
        rows.append([
            name, len(r["tokens"][name]), fmt(w and w["tok_s"]), fmt(w and w["ms_step"], 2),
            fmt(w and w["tok_step"], 2), fmt(r["hit"], 3),
        ])
    sections.append("## Un job alla volta\n\n" + table(
        ["job", "token", "tok/s", "ms per passo", "token per passo", "selezioni servite dalla VRAM"], rows,
    ))

    summary = []
    for n in n_jobs_list:
        def timing(n = n):
            return bench.run(names[:n], False, args.bench_tokens), bench.run(names[:n], True, args.bench_tokens)

        pair = phase(f"{n} jobs", timing)
        if pair is None:
            continue
        batched, staggered = pair
        section = [f"## {n} job insieme\n", table(TIMING_HEADER, [timing_row("un batch", batched), timing_row("sfalsato", staggered)])]
        wb, ws = batched["window"], staggered["window"]
        if wb and ws:
            gain = ws["tok_s"] / wb["tok_s"] - 1.0
            summary.append([n, fmt(wb["tok_s"]), fmt(ws["tok_s"]), f"{100 * gain:+.1f}%"])
            section.append(f"\nSfalsato rispetto a un batch: **{100 * gain:+.1f}%** di tok/s totali.")
        st = staggered["stagger_total"]
        section.append(
            f"\nContatori del decode sfalsato sull'intero giro: {st['steps']} chiamate, {st['launched']} gruppi emessi, "
            f"{st['sampled']} campionati, {st['solo']} completati da soli, {st['drains']} svuotamenti. "
            f"Gruppi attesi: {' e '.join('`' + g + '`' for g in groups_of(n))}."
        )
        sections.append("\n".join(section))

    # 4. Same computation: forced tokens, probabilities against the group alone
    check_parts = []

    def check():
        ref_tokens = {name: solo[name]["tokens"][name] for name in solo}
        missing = [name for name in names if not ref_tokens.get(name)]
        if missing:
            raise RuntimeError(f"no reference tokens for {missing}")
        alone = {}      # group -> its two forced runs
        for n in n_jobs_list:
            for g in groups_of(n):
                if g not in alone:
                    alone[g] = (
                        bench.run(g, False, 0, forced = ref_tokens),
                        bench.run(g, False, 0, forced = ref_tokens),
                    )
            batched = bench.run(names[:n], False, 0, forced = ref_tokens)
            staggered = bench.run(names[:n], True, 0, forced = ref_tokens)
            rows = []
            unforced = []
            for g in groups_of(n):
                ref, again = alone[g]
                for name in g:
                    rows.append(check_row(name, f"`{g}` da solo, rifatto", compare_forced(ref, again, name)))
                    rows.append(check_row(name, f"sfalsato contro `{g}` da solo", compare_forced(ref, staggered, name)))
                    rows.append(check_row(name, f"un batch contro `{g}` da solo", compare_forced(ref, batched, name)))
                    for label, r in (("da solo", ref), ("sfalsato", staggered), ("un batch", batched)):
                        if r["tokens"][name] != ref_tokens[name]:
                            unforced.append(f"`{name}` ({label})")
            st = staggered["stagger_total"]
            part = [
                f"### {n} job\n",
                table(CHECK_HEADER, rows),
                f"\nGiro sfalsato a token imposti: {st['steps']} chiamate, {st['launched']} gruppi emessi, "
                f"{st['sampled']} campionati, {st['solo']} completati da soli, {st['drains']} svuotamenti.",
            ]
            if unforced:
                part.append(
                    "\n**Attenzione**: i token usciti non sono quelli imposti per " + ", ".join(unforced)
                    + "; le righe di quei job non confrontano la stessa sequenza."
                )
            check_parts.append("\n".join(part))

    phase("same computation (forced tokens)", check)
    sections.append(
        "## Stesso calcolo?\n\n"
        "Due giri identici di questo modello non danno gli stessi token greedy: al primo quasi-pareggio la scelta "
        "cambia e da lì i testi divergono, quindi confrontare i token non dice nulla. Qui ogni job viene generato "
        "di nuovo **imponendogli** i token che ha prodotto da solo, in tutti i modi: ogni giro elabora la stessa "
        "sequenza e a ogni posizione si registrano la probabilità del token imposto e i due token più probabili. "
        "Il riferimento è il gruppo del job fatto girare da solo; la riga «rifatto» confronta due giri identici "
        "del gruppo da solo ed è la misura di quanto il modello cambia da un giro all'altro senza che cambi "
        "nulla.\n\n"
        "- **token imposto più probabile**: in quante posizioni del giro in esame il token imposto è anche il più probabile\n"
        "- **top-1 cambiati**: posizioni in cui il token più probabile non è lo stesso nei due giri\n"
        "- **margine massimo dove cambia**: tra quelle posizioni, il distacco più grande tra primo e secondo token nel riferimento\n"
        "- **prima posizione con p diversa**: la prima posizione in cui la probabilità del token imposto non è "
        "identica, bit per bit, nei due giri (`-` quando lo è ovunque)\n"
        "- **scarto di p**: differenza, in valore assoluto, tra i due giri della probabilità del token imposto\n\n"
        + ("\n\n".join(check_parts) if check_parts else "Nessun confronto completato.")
    )

    # 5. Where the time goes: stream-time brackets
    prof_rows = []

    def profile():
        if env.prof is None:
            raise RuntimeError("no CUDA device: stream-time brackets not available")
        r = bench.run(names[:1], False, args.bench_prof_tokens, profile = True)
        prof_rows.append(prof_row(f"`{names[0]}` da solo", r))
        for n in n_jobs_list:
            r = bench.run(names[:n], False, args.bench_prof_tokens, profile = True)
            prof_rows.append(prof_row(f"{n} job, un batch", r))
            r = bench.run(names[:n], True, args.bench_prof_tokens, profile = True)
            prof_rows.append(prof_row(f"{n} job, sfalsato", r))

    phase("where the time goes", profile)
    sections.append(
        "## Dove va il tempo\n\n"
        f"Giri brevi a parte ({args.bench_prof_tokens} token per job) con eventi CUDA attorno alle parti del passo: "
        "tempi di flusso delle schede, senza sincronizzare l'host. Cifre per chiamata di `iterate()`, sul tratto in "
        "cui tutti i job generano.\n\n"
        "- **passo**: dall'inizio dell'emissione alla fine del lavoro emesso su quella scheda. Nel modo normale quello "
        "della scheda 1 comprende la prima metà del passo, che gira sulla scheda 0\n"
        "- **attesa del worker**: somma delle parentesi di raccolta, cioè il tempo in cui la scheda è ferma ad aspettare "
        "gli esperti calcolati su CPU (più la rilettura del risultato) dopo aver finito i propri\n"
        "- **consegna al worker**: somma delle parentesi di emissione di un job (attesa che lo slot sia libero, più il "
        "kernel che lo riempie)\n"
        "- **raccolte per chiamata**: quanti layer MoE la scheda ha chiuso in una chiamata\n\n"
        + (table(PROF_HEADER, prof_rows) if prof_rows else "Nessun giro di profilo completato.")
    )

    bench.close()

    split = env.split
    copies = env.copy_stats() if env.copy_stats is not None else None
    head = [
        "# Decode sfalsato: misura",
        "",
        time.strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "## Sintesi",
        "",
        table(["job insieme", "un batch tok/s", "sfalsato tok/s", "differenza"], summary) if summary else "Nessuna misura completa.",
        "",
    ]
    for note in notes:
        head += [note, ""]
    head += [
        "## Condizioni",
        "",
        f"- Banco di prova: `{os.path.basename(__file__)}` al commit {env.revision}",
        f"- ExLlamaV3 {env.version} da `{env.lib_path}`, torch {env.torch_version}",
        f"- Modello `{env.model_path}`",
        f"- Schede: {', '.join(env.devices) if env.devices else '-'}",
        f"- Confine tra le schede: modulo {split} di {env.n_modules}"
        + ("" if split else " (**decode sfalsato non disponibile**)"),
        f"- Bozza: {env.draft_desc}",
        f"- Token per job {args.bench_tokens}, moduli per turno {args.bench_interleave}, batch massimo {gen.max_batch_size}",
    ]
    if copies is not None:
        head.append(
            f"- Copie tra schede: {copies[0]} dirette, {copies[1]} passate dalla RAM"
            + (" (**una copia passata dalla RAM blocca l'host a metà passo**)" if copies[1] else "")
        )
    head += [
        "- Campionamento greedy, nessuna condizione di stop; dopo l'adattamento iniziale nessuno sweep del piazzamento",
        "- Le cifre di ogni riga valgono per il tratto in cui tutti i job generano; «chiamata» è una `iterate()`: "
        "nel modo normale un passo per tutti i job, nel modo sfalsato un passo per un gruppo",
        "- Uso e potenza delle schede: media dei campioni di `nvidia-smi` nel tratto, saltando il primo mezzo secondo"
        + (f" (campionatore fermato da un errore: {gpu.error})" if gpu is not None and gpu.error else ""),
        "",
    ]
    return "\n".join(head) + "\n" + "\n\n".join(sections) + "\n"


def main():
    ap = argparse.ArgumentParser(add_help = False)
    ap.add_argument("--bench-jobs", default = "2,3,4", help = "numeri di job da provare (default 2,3,4)")
    ap.add_argument("--bench-tokens", type = int, default = 400, help = "token generati da ogni job (default 400)")
    ap.add_argument("--bench-warmup", type = int, default = 48, help = "token dei giri di riscaldamento (default 48)")
    ap.add_argument("--bench-adapt", type = int, default = 8,
                    help = "giri al massimo per adattare il piazzamento degli esperti ai prompt della misura (default 8, 0 = no)")
    ap.add_argument("--bench-adapt-tokens", type = int, default = 160, help = "token per job in un giro di adattamento (default 160)")
    ap.add_argument("--bench-prof-tokens", type = int, default = 160, help = "token per job nei giri di profilo (default 160)")
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
    from exllamav3.version import __version__ as exl3_version
    from openai.config import parse_args
    from openai.model_container import ModelContainer

    if not hasattr(Generator, "iterate_staggered"):
        raise SystemExit(
            f"La libreria importata ({exllamav3.__file__}) non ha il decode sfalsato: "
            "installa il fork nell'ambiente (pip install -e) e riprova."
        )

    revision = bench_revision()
    print(f" -- bench: {os.path.basename(__file__)} at commit {revision}; phases: expert placement, warm-up, "
          "speed, same computation (forced tokens), where the time goes", flush = True)
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
    n_jobs_list = sorted({int(x) for x in args.bench_jobs.split(",") if x.strip()})
    max_jobs = min(max(n_jobs_list), gen.max_batch_size, len(PROMPTS))

    # The library's own sweep, past the wrapper the server's expert tier puts around it: the tier
    # saves the placement file after every sweep it sees, and the production file must not learn
    # from a bench. From here on nothing sweeps unless the bench asks: a sweep between two runs
    # moves experts between VRAM and the CPU worker, which changes the numerics of the layers it
    # touches and the load on the worker
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

    split = gen._stagger_plan()
    env = SimpleNamespace(
        torch = torch,
        gen = gen,
        model = model,
        Job = Job,
        sampler_cls = GreedySampler,
        prompts = {NAMES[i]: encode(PROMPTS[i]) for i in range(max_jobs)},
        max_jobs = max_jobs,
        n_jobs_list = n_jobs_list,
        sweep = sweep,
        prof = StreamProf(torch, model, gen, stage_devices(torch, model, split)) if torch.cuda.is_available() else None,
        split = split,
        n_modules = len(model.fwd_modules),
        version = exl3_version,
        revision = revision,
        lib_path = os.path.dirname(exllamav3.__file__),
        torch_version = torch.__version__,
        model_path = cfg.model_path,
        devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        draft_desc = f"MTP, {gen.num_draft_tokens} token" if draft_model is not None else "nessuna",
        copy_stats = lambda: (device_copy.stats["direct"], device_copy.stats["bounced"]),
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
