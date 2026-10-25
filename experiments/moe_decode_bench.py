#!/usr/bin/env python3
"""
Decode-size routed-expert MoE kernels on one card, measured apart from the model (a GPU window,
production stopped): what one MoE layer of a decode step costs for 1-8 rows, how that cost
depends on the number of distinct experts the rows pick, and what the extension's other paths
and its tuning knobs would cost at the same shapes. Nothing in the library is changed: the
script calls the published extension's kernels directly.

Why: the night profiling of 2026-10-24 (server repo, docs/REPORT-profiling-20261024.md, 2.3 and
2.6 b) read the ~2.5 ms of MoE kernel per extra row as rows re-reading experts that other rows of
the step also picked. The decode path does not do that. BlockSparseMLP.forward sends every batch
of 1..MAX_BSZN (8) rows to BC_BlockSparseMLP.run_bszN, i.e. to the fused "coop" kernels
(exllamav3_ext/quant/exl3_moe_coop_kernel.cuh): a rotation pre-kernel builds a run table that
sorts the (row, pick) slots by expert and cuts them into runs of at most ROWS = 8 slots of one
expert (build_runs), and kernels A (gate/up) and B (down) run one block per (run, column group):
the run's slots are the rows of one MMA tile, so an expert's weights are read once per call
whichever rows picked it (an expert can be picked at most once per row, so with at most 8 rows
it is always a single run). The per-row cost is the experts the extra row adds that no other row
picked, at what the kernels make of each.

What the script measures, per row count R = 1..8 and selection pattern:
  - "distinte": the rows pick disjoint experts (the most distinct experts R rows can pick);
  - "uguali": every row picks the same experts (one row's worth of distinct experts);
  - "sovrapposte F": each row keeps a fraction F of the previous row's GPU experts;
  with a share of the picks (1 - --hit) outside the GPU table, as the CPU-split experts are
  (masked in the kernel, never computed here). The paths:
  - coop: the routed launch as run_bszN issues it (rotation, A, B; the shared expert's output
    merged through its sigmoid gate in B's epilogue), and the shared expert's own one-expert
    launch before it;
  - fused: the prefill's grouped fused tier (ext.exl3_moe + ext.exl3_moe_gather) at the same
    rows, with the decode-size slot layout BlockSparseMLP.forward builds for it when every
    assignment fits the fused tier (no host readback of the counts), mirrored here;
  in one process per setting of the coop launcher's knobs (EXL3_MOE_COOP_WIDE, read once per
    process by the extension; EXL3_MOE_COOP_KSPLIT); the fused tier only in the default one.
  Times: the GPU time per call from a CUDA graph of the calls (no host in it) when capture
  works, else from back-to-back calls; and the host's issue time per call. The weights are
  random EXL3 trellises (every word decodes to some codebook value, and the kernels do the same
  work whatever the values) at the model's shapes and bit widths, read from the checkpoint's
  safetensors headers with --model (headers only, nothing loaded), in --layers copies of
  --experts experts so that consecutive calls read weights the L2 cache does not hold.
  Accuracy: for a few cases every path's output against a torch reference of the same expert
  MLPs (the dense reconstruct of each expert, as the upstream kernel tests do), per row, and
  every knob's coop output against the default's.

Run it with the production server STOPPED (it needs a card; about a GiB of VRAM), with the Python
of the environment that has the extension (FORK.md, "A routed expert is read once per decode
step", has the command lines for the workstation):

    <env python> experiments/moe_decode_bench.py --dry-run --model <model dir> --out <dir>
    <env python> experiments/moe_decode_bench.py --model <model dir> --out <dir>

The report (REPORT.md, Italian like the fork's other benches; code and comments in English) and
the raw records (records.jsonl, probes-*.pt) are in --out; nothing is written anywhere else.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

MAX_BSZN = 8                    # BlockSparseMLP.MAX_BSZN: rows of the decode-size path
COOP_ROWS = 8                   # exl3_moe_coop_kernel.cuh, ROWS: slots of one expert per run
MOE_ACT_SILU = 0                # exl3_moe_common.cuh / exl3_moe_coop.cuh (both 0 for SiLU)
CODEBOOKS = ("3inst", "mcg", "mul1")
TOL_ROW = 3e-3                  # per-row bound against the reference (upstream's coop test bound)

# The coop launcher's knobs, one process each (the extension reads them once per process)
SETTINGS = {
    "default": {},
    "wide0": {"EXL3_MOE_COOP_WIDE": "0"},
    "wide1": {"EXL3_MOE_COOP_WIDE": "1"},
    "ksplit2": {"EXL3_MOE_COOP_KSPLIT": "2"},
}
KNOB_VARS = ("EXL3_MOE_COOP_WIDE", "EXL3_MOE_COOP_KSPLIT")

# Production's shapes (Qwen3.8-Flash-Next 2.05 bpw), used without --model
DEFAULT_SHAPES = {
    "H": 2560, "Hi": 2560, "I": 640, "Ho": 2560, "K_gu": 2.0, "K_d": 2.0, "cb": 2,
    "sh_I": 640, "sh_K_gu": 4.0, "sh_K_d": 4.0, "sh_cb": 2, "topk": 10, "num_experts": 512,
    "source": "default",
}


# --------------------------------------------------------------------------------------------
#   Shapes from the checkpoint (safetensors headers, standard library only)
# --------------------------------------------------------------------------------------------

def read_safetensors_header(path: str) -> dict:
    """The JSON header of a safetensors file: {tensor name: {dtype, shape, data_offsets}}"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > (1 << 30):
            raise ValueError(f"{path}: header di {n} byte, non è un file safetensors")
        h = json.loads(f.read(n).decode("utf-8"))
    h.pop("__metadata__", None)
    return h


def checkpoint_headers(model_dir: str) -> dict:
    """Every tensor's header entry over the checkpoint's files: {name: entry}"""
    files = sorted(fn for fn in os.listdir(model_dir) if fn.endswith(".safetensors"))
    if not files:
        raise ValueError(f"{model_dir}: nessun file .safetensors")
    out = {}
    for fn in files:
        out.update(read_safetensors_header(os.path.join(model_dir, fn)))
    return out


def _cfg_value(cfg: dict, key: str, default = None):
    for c in (cfg, cfg.get("text_config") or {}):
        if key in c:
            return c[key]
    return default


_EXPERT_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.trellis$")
_SHARED_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.shared_experts?\.(gate|up|down)_proj\.trellis$")


def _k_of(shape: list) -> float:
    """Bits per weight of a trellis (k / 16, n / 16, 16 * K)"""
    return shape[2] / 16.0


def _cb_of(headers: dict, trellis_key: str) -> int:
    base = trellis_key[: -len(".trellis")]
    if f"{base}.mul1" in headers:
        return 2
    if f"{base}.mcg" in headers:
        return 1
    return 0


def model_shapes(model_dir: str, layer: int | None = None) -> dict:
    """
    The routed and shared experts' shapes and bit widths of one MoE layer (the first one with
    experts when `layer` is None), from the trellis shapes in the headers, the top-k from
    config.json; also the (K_gu, K_d) combinations found over every layer, which the bench does
    not need but the report shows.
    """
    headers = checkpoint_headers(model_dir)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    experts = {}            # layer -> {proj: shape} of expert 0
    shared = {}             # layer -> {proj: (shape, key)}
    per_layer_k = {}        # layer -> {(K_gu, K_d)}
    for name, e in headers.items():
        m = _EXPERT_RE.search(name)
        if m:
            li, ei, proj = int(m.group(1)), int(m.group(2)), m.group(3)
            if ei == 0:
                experts.setdefault(li, {})[proj] = (e["shape"], name)
            continue
        m = _SHARED_RE.search(name)
        if m:
            shared.setdefault(int(m.group(1)), {})[m.group(2)] = (e["shape"], name)
    if not experts:
        raise ValueError(f"{model_dir}: nessun tensore 'layers.N.mlp.experts.E.<gate|up|down>_proj.trellis' negli header")
    for li, ps in experts.items():
        if all(p in ps for p in ("gate", "up", "down")):
            per_layer_k[li] = (_k_of(ps["up"][0]), _k_of(ps["down"][0]))
    li = min(per_layer_k) if layer is None else layer
    if li not in per_layer_k:
        raise ValueError(f"layer {li}: esperti incompleti o assenti negli header")
    ps = experts[li]
    up_shape, up_key = ps["up"]
    down_shape = ps["down"][0]
    out = {
        "H": int(_cfg_value(cfg, "hidden_size", up_shape[0] * 16)),
        "Hi": up_shape[0] * 16, "I": up_shape[1] * 16, "Ho": down_shape[1] * 16,
        "K_gu": _k_of(up_shape), "K_d": _k_of(down_shape), "cb": _cb_of(headers, up_key),
        "topk": int(_cfg_value(cfg, "num_experts_per_tok", DEFAULT_SHAPES["topk"])),
        "num_experts": int(_cfg_value(cfg, "num_experts", DEFAULT_SHAPES["num_experts"])),
        "layer": li, "source": model_dir,
        "k_combos": sorted({f"{a:g}/{b:g}" for a, b in per_layer_k.values()}),
        "moe_layers": len(per_layer_k),
    }
    if _k_of(ps["gate"][0]) != out["K_gu"]:
        raise ValueError(f"layer {li}: gate e up a bit diversi ({_k_of(ps['gate'][0])} / {out['K_gu']}), il kernel coop non lo prevede")
    sh = shared.get(li, {})
    if all(p in sh for p in ("gate", "up", "down")):
        out.update(sh_I = sh["up"][0][1] * 16, sh_K_gu = _k_of(sh["up"][0]), sh_K_d = _k_of(sh["down"][0]),
                   sh_cb = _cb_of(headers, sh["up"][1]))
    else:
        out.update(sh_I = 0, sh_K_gu = 0.0, sh_K_d = 0.0, sh_cb = out["cb"])
    return out


# --------------------------------------------------------------------------------------------
#   Selections (pure Python): which experts each row picks
# --------------------------------------------------------------------------------------------

def gpu_counts(rows: int, topk: int, hit: float) -> list:
    """GPU-resident picks per row: hit * topk on average over the rows, an integer per row"""
    out = []
    for r in range(rows):
        a = math.floor(hit * topk * (r + 1) + 1e-9) - math.floor(hit * topk * r + 1e-9)
        out.append(max(0, min(topk, a)))
    return out


def parse_pattern(p: str) -> tuple:
    """("distinct" | "same" | "overlap", fraction)"""
    if p in ("distinct", "same"):
        return p, 0.0
    m = re.fullmatch(r"overlap:([0-9.]+)", p)
    if m and 0.0 <= float(m.group(1)) <= 1.0:
        return "overlap", float(m.group(1))
    raise ValueError(f"schema di selezione sconosciuto: {p!r} (distinct, same, overlap:F)")


def pattern_label(p: str) -> str:
    kind, f = parse_pattern(p)
    return {"distinct": "distinte", "same": "uguali"}.get(kind) or f"sovrapposte {f:g}"


def plan_rows(rows: int, topk: int, e_gpu: int, e_total: int, hit: float, pattern: str,
              rng: random.Random) -> list:
    """
    One selection: `rows` lists of `topk` distinct expert ids. GPU-resident ids are 0..e_gpu-1
    (the module's local table), the others e_gpu..e_total-1 (CPU-split experts: masked in the
    coop kernel, the sentinel bin in the fused tier). The pattern decides how the rows' GPU picks
    overlap; the CPU picks are fresh for each row.
    """
    kind, frac = parse_pattern(pattern)
    counts = gpu_counts(rows, topk, hit)
    if kind == "same":
        counts = [counts[0]] * rows
    keeps = [0] + [min(round(frac * min(counts[r], counts[r - 1])), counts[r], counts[r - 1]) if kind == "overlap" else 0
                   for r in range(1, rows)]
    need = counts[0] if kind == "same" else sum(n - k for n, k in zip(counts, keeps))    # fresh experts from the pool
    if need > e_gpu:
        raise ValueError(f"{rows} righe '{pattern}' vogliono fino a {need} esperti GPU distinti, la tabella ne ha {e_gpu}")
    if topk - min(counts) > e_total - e_gpu:
        raise ValueError(f"servono {topk - min(counts)} esperti CPU per riga, ce ne sono {e_total - e_gpu}")
    pool = list(range(e_gpu))
    rng.shuffle(pool)
    out = []
    used = 0
    prev = []
    for r in range(rows):
        n = counts[r]
        if kind == "distinct" or r == 0:
            gpu = pool[used: used + n]
            used += n
        elif kind == "same":
            gpu = list(prev)
        else:
            keep = keeps[r]
            kept = rng.sample(prev, keep)
            fresh = pool[used: used + n - keep]          # never in kept: those came from pool[:used]
            used += n - keep
            gpu = kept + fresh
        cpu = rng.sample(range(e_gpu, e_total), topk - len(gpu))
        row = gpu + cpu
        rng.shuffle(row)
        out.append(row)
        prev = gpu
    return out


def distinct_gpu(sel: list, e_gpu: int) -> int:
    """Distinct GPU-resident experts a selection reads"""
    return len({e for row in sel for e in row if e < e_gpu})


def gpu_picks(sel: list, e_gpu: int) -> int:
    """(row, pick) slots on GPU-resident experts: what a kernel that read an expert once per pick would read"""
    return sum(1 for row in sel for e in row if e < e_gpu)


def row_weights(rows: int, topk: int, rng: random.Random) -> list:
    """Routing weights per row, positive, summing to 1"""
    out = []
    for _ in range(rows):
        w = [rng.uniform(0.1, 1.0) for _ in range(topk)]
        s = sum(w)
        out.append([v / s for v in w])
    return out


def case_rng(seed: int, pattern: str, rows: int, s: int) -> random.Random:
    return random.Random(f"{seed}/{pattern}/{rows}/{s}")


# --------------------------------------------------------------------------------------------
#   The fused tier's decode-size layout (torch): BlockSparseMLP.forward, all-fused fast path
# --------------------------------------------------------------------------------------------

def fused_layout(torch, sel, rw, E: int, first: int = 0, flat_token = None, count_on_device: bool = False) -> dict:
    """
    What BlockSparseMLP.forward builds for ext.exl3_moe when every assignment fits the fused
    tier (num_tokens * top_k <= fused_rows: expert_count_list is None, FUSED_DET on): local
    expert ids with the sentinel bin E for picks outside [first, first + E), the assignments
    sorted by expert, the per-expert counts, the inverse permutation and the slot tables
    (base = start, kind = count > 0) of the one exl3_moe_gather. forward() counts with
    torch.bincount, which on CUDA reads the input's min and max back to the host (two syncs);
    count_on_device does the same count with a scatter_add, so that a CUDA graph can hold it.
    """
    T, k = sel.shape
    flat_g = sel.reshape(-1)
    flat_w = rw.reshape(-1)
    if flat_token is None:
        flat_token = torch.arange(T, device = sel.device).repeat_interleave(k)
    local = flat_g - first
    valid = (local >= 0) & (local < E)
    local = torch.where(valid, local, torch.full_like(local, E))
    order = local.argsort()
    token_sorted = flat_token[order]
    weight_sorted = flat_w[order]
    if count_on_device:
        expert_count = torch.zeros(E + 1, dtype = torch.long, device = sel.device)
        expert_count.scatter_add_(0, local, torch.ones_like(local))
    else:
        expert_count = torch.bincount(local, minlength = E + 1)
    A = local.shape[0]
    inv_order = torch.empty_like(order).scatter_(0, order, torch.arange(A, device = order.device))
    expert_start = torch.cumsum(expert_count, 0) - expert_count
    tables = torch.stack([expert_start, expert_start, (expert_count > 0).long()])
    return {"local": local, "order": order, "token_sorted": token_sorted, "weight_sorted": weight_sorted,
            "expert_count": expert_count, "inv_order": inv_order, "tables": tables, "n_slots": A}


def launcher_knobs(env) -> dict:
    """The coop launcher's knobs as the extension reads them (atoi of the variable; unset = none)"""
    def atoi(v):
        m = re.match(r"\s*([+-]?\d+)", v or "")
        return int(m.group(1)) if m else 0
    w, k = env.get("EXL3_MOE_COOP_WIDE"), env.get("EXL3_MOE_COOP_KSPLIT")
    return {"wide_mode": None if w is None else atoi(w), "ksplit_forced": None if k is None else atoi(k)}


def coop_geometry(Hi: int, I: int, Ho: int, slots: int, slots_max: int, gated: bool, sms: int, blackwell: bool,
                  wide_mode: int | None = None, ksplit_forced: int | None = None) -> dict:
    """
    The tile geometry exl3_moe_coop_launch picks for a launch (exl3_moe_coop.cu: moe_coop_pick_wide,
    moe_coop_pick_ksplit, the grids): per stage the wide (128-column, 4 x 4 warps) or narrow (32-column,
    16-way k split) tile, the split-k factor and the grid. Grids are sized for the no-duplicates case (slots x
    column groups x projections x split): blocks past the run count exit at once. The rotation pre-kernel
    runs for bsz > 1 only (with one row kernel A rotates its input in every block).
    """
    def pick_wide(kslices):
        if wide_mode is not None and wide_mode >= 0:
            return wide_mode != 0
        if not blackwell:
            return True
        return kslices >= 256 or (kslices >= 128 and slots >= 32)

    def pick_ksplit(base, kslices, max_split):
        if ksplit_forced is not None and ksplit_forced > 0:
            return min(ksplit_forced, max_split)
        want = -(-sms // max(base, 1))
        kcap = max(kslices // 64, 1)
        return max(1, min(min(want, kcap), max_split))

    nproj = 2 if gated else 1
    wa, wb = pick_wide(Hi // 16), pick_wide(I // 16)
    base_a = slots * nproj * (I // (128 if wa else 32))
    base_b = slots * (Ho // (128 if wb else 32))
    max_split = max(1, slots_max // slots)
    ka, kb = pick_ksplit(base_a, Hi // 16, max_split), pick_ksplit(base_b, I // 16, max_split)
    return {"a_wide": wa, "b_wide": wb, "a_ksplit": ka, "b_ksplit": kb, "a_grid": base_a * ka, "b_grid": base_b * kb}


def coop_ctr_len(slots_max: int, rows_max: int, I: int, Ho: int) -> int:
    """exl3_moe_coop.cuh, exl3_moe_coop_ctr_len"""
    return slots_max * (I // 128) + rows_max * (Ho // 128) + 2 + (slots_max + 1) + slots_max


# --------------------------------------------------------------------------------------------
#   The child: weights, buffers and the timed paths on one card
# --------------------------------------------------------------------------------------------

class Bench:
    """
    One card's worth of the bench. `ext` is the extension module (a fake one in the CPU tests);
    `on_tensor`, when set, is called with every weight tensor made (the fake resolves the
    pointer tables through it).
    """

    def __init__(self, torch, ext, cfg: dict, device, on_tensor = None):
        self.torch = torch
        self.ext = ext
        self.cfg = cfg
        self.dev = torch.device(device)
        self.on_tensor = on_tensor
        s = cfg["shapes"]
        self.H, self.Hi, self.I, self.Ho = s["H"], s["Hi"], s["I"], s["Ho"]
        self.topk = s["topk"]
        self.E = cfg["experts"]
        self.e_total = max(s["num_experts"], self.E + self.topk)
        self.mcg, self.mul1 = s["cb"] == 1, s["cb"] == 2
        self.gen = torch.Generator(device = self.dev).manual_seed(cfg["seed"])
        self.layers = [self._experts(self.E, self.Hi, self.I, self.Ho, s["K_gu"], s["K_d"]) for _ in range(cfg["layers"])]
        # One shared expert per layer copy, as the model has: a single one would stay in the L2
        self.shared = None
        if s.get("sh_I"):
            self.shared = [self._experts(1, self.Hi, s["sh_I"], self.Ho, s["sh_K_gu"], s["sh_K_d"])
                           for _ in range(cfg["layers"])]
        self.sh_gate_w = (self._randn((self.H,)) * 0.02).half()
        self._buffers()

    # ---- weights

    def _made(self, t):
        if self.on_tensor is not None:
            self.on_tensor(t)
        return t

    def _randn(self, shape):
        return self.torch.randn(shape, generator = self.gen, device = self.dev)

    def _trellis(self, k: int, n: int, K: float):
        words = int(round(16 * K))
        if words != 16 * K:
            raise ValueError(f"K = {K}: non è un multiplo di 1/16")
        t = self.torch.randint(-32768, 32768, (k // 16, n // 16, words), generator = self.gen, device = self.dev,
                               dtype = self.torch.int16)
        return self._made(t)

    def _scale(self, n: int):
        mag = self.torch.rand((n,), generator = self.gen, device = self.dev) * 0.2 + 0.9
        sign = self.torch.where(self.torch.rand((n,), generator = self.gen, device = self.dev) < 0.5, -1.0, 1.0)
        return self._made((mag * sign).half())

    def _experts(self, E: int, Hi: int, I: int, Ho: int, K_gu: float, K_d: float) -> dict:
        """E experts: {"g" | "u" | "d": [(trellis, suh, svh)] * E, "tabs": {p: (ptrs x 3)}, K_gu, K_d, I}"""
        ex = {"K_gu": K_gu, "K_d": K_d, "I": I}
        for p, (k, n, K) in (("g", (Hi, I, K_gu)), ("u", (Hi, I, K_gu)), ("d", (I, Ho, K_d))):
            ex[p] = [(self._trellis(k, n, K), self._scale(k), self._scale(n)) for _ in range(E)]
        ex["tabs"] = {p: tuple(self.torch.tensor([e[i].data_ptr() for e in ex[p]], dtype = self.torch.long, device = self.dev)
                               for i in range(3)) for p in ("g", "u", "d")}
        return ex

    def weights_digest(self) -> int:
        """A checksum of the first layer's trellises: the processes must have built the same weights"""
        t = self.torch
        acc = 0
        for p in ("g", "u", "d"):
            for tr, _, _ in self.layers[0][p][:4]:
                acc = (acc * 1000003 + int(tr.to(t.int64).sum().item())) & ((1 << 61) - 1)
        return acc

    # ---- buffers (BlockSparseMLP.load_local, BC_BlockSparseMLP's shared-expert scratch)

    def _buffers(self):
        t, d = self.torch, self.dev
        S = MAX_BSZN * self.topk
        self.slots_max = S
        self.c_had_g = t.empty((S, self.Hi), dtype = t.half, device = d)
        self.c_had_u = t.empty((S, self.Hi), dtype = t.half, device = d)
        self.c_gu_g = t.empty((S, self.I), dtype = t.half, device = d)
        self.c_gu_u = t.empty((S, self.I), dtype = t.half, device = d)
        self.c_act = t.empty((S, self.I), dtype = t.half, device = d)
        self.c_dout = t.empty((S, self.Ho), dtype = t.float, device = d)
        self.c_ctr = t.zeros((coop_ctr_len(S, MAX_BSZN, self.I, self.Ho),), dtype = t.int, device = d)
        self.c_out = t.empty((MAX_BSZN, self.H), dtype = t.float, device = d)
        if self.shared is not None:
            Is = self.shared[0]["I"]
            self.s_had_g = t.empty((MAX_BSZN, self.Hi), dtype = t.half, device = d)
            self.s_had_u = t.empty((MAX_BSZN, self.Hi), dtype = t.half, device = d)
            self.s_gu_g = t.empty((MAX_BSZN, Is), dtype = t.half, device = d)
            self.s_gu_u = t.empty((MAX_BSZN, Is), dtype = t.half, device = d)
            self.s_act = t.empty((MAX_BSZN, Is), dtype = t.half, device = d)
            self.s_dout = t.empty((MAX_BSZN, self.Ho), dtype = t.float, device = d)
            self.s_ctr = t.zeros((coop_ctr_len(MAX_BSZN, MAX_BSZN, Is, self.Ho),), dtype = t.int, device = d)
            self.s_out = t.zeros((MAX_BSZN, self.H), dtype = t.float, device = d)
            self.s_sel = t.zeros((MAX_BSZN, 1), dtype = t.long, device = d)
            self.s_rw = t.ones((MAX_BSZN, 1), dtype = t.half, device = d)
        # The fused tier's per-group staging (BlockSparseMLP.fused_mode_buffers)
        self.f_rows = 256 if self.mul1 else 128
        # The fused kernel takes one hidden width for the gate/up input and the down output, and the
        # mcg or mul1 codebook only (exl3_moe.cu)
        self.fused_ok = self.H == self.Hi == self.Ho and (self.mcg or self.mul1)
        C = self.ext.exl3_moe_max_concurrency(self.dev.index if self.dev.index is not None else 0)
        self.f_tg = t.empty((C, self.f_rows, self.H), dtype = t.half, device = d)
        self.f_tu = t.empty((C, self.f_rows, self.H), dtype = t.half, device = d)
        self.f_tig = t.empty((C, self.f_rows, self.I), dtype = t.half, device = d)
        self.f_tiu = t.empty((C, self.f_rows, self.I), dtype = t.half, device = d)

    # ---- inputs

    def inputs(self, pattern: str, rows: int, s: int, hit: float) -> dict:
        """Selection set s of a case on the device: x, sel, rw (and the plan, for the counts)"""
        t = self.torch
        rng = case_rng(self.cfg["seed"], pattern, rows, s)
        sel = plan_rows(rows, self.topk, self.E, self.e_total, hit, pattern, rng)
        rw = row_weights(rows, self.topk, rng)
        g = t.Generator(device = "cpu").manual_seed(rng.randrange(1 << 30))
        x = (t.randn((rows, self.H), generator = g) * 0.5).half().to(self.dev)
        return {
            "plan": sel,
            "x": x,
            "sel": t.tensor(sel, dtype = t.long, device = self.dev),
            "rw": t.tensor(rw, dtype = t.float).half().to(self.dev),
            "flat_token": t.arange(rows, device = self.dev).repeat_interleave(self.topk),
            "distinct": distinct_gpu(sel, self.E),
            "picks": gpu_picks(sel, self.E),
        }

    # ---- the paths

    def coop_shared(self, x, layer: int = 0):
        """The shared expert's one-expert launch (run_bszN, sh_coop): its output in s_out[:R]"""
        sh = self.shared[layer]
        R = x.shape[0]
        self.ext.exl3_moe_coop(
            x, self.s_sel[:R], self.s_rw[:R], -1, -1, self.Hi,
            *sh["tabs"]["g"], *sh["tabs"]["u"], *sh["tabs"]["d"],
            None, None, None,
            float(sh["K_gu"]), float(sh["K_gu"]), float(sh["K_d"]), self._sh_mcg(), self._sh_mul1(),
            MOE_ACT_SILU, 0.0, True,
            self.s_had_g, self.s_had_u, self.s_gu_g, self.s_gu_u, self.s_act, self.s_dout, self.s_ctr, self.s_out,
            None, None,
        )
        return self.s_out[:R]

    def _sh_mcg(self):
        return self.cfg["shapes"].get("sh_cb", self.cfg["shapes"]["cb"]) == 1

    def _sh_mul1(self):
        return self.cfg["shapes"].get("sh_cb", self.cfg["shapes"]["cb"]) == 2

    def coop_routed(self, layer: int, inp: dict, with_shared: bool):
        """The routed launch (run_bszN): rotation, A, B into c_out[:R], the shared output merged
        through the sigmoid gate when with_shared (as BlockSparseMLP's bc_sh_exp layers do)"""
        ly = self.layers[layer]
        s = self.cfg["shapes"]
        x = inp["x"]
        R = x.shape[0]
        sh_out = self.s_out[:R] if (with_shared and self.shared is not None) else None
        sh_w = self.sh_gate_w if sh_out is not None else None
        self.ext.exl3_moe_coop(
            x, inp["sel"], inp["rw"], 0, self.E, self.Hi,
            *ly["tabs"]["g"], *ly["tabs"]["u"], *ly["tabs"]["d"],
            None, None, None,
            float(s["K_gu"]), float(s["K_gu"]), float(s["K_d"]), self.mcg, self.mul1,
            MOE_ACT_SILU, 0.0, True,
            self.c_had_g, self.c_had_u, self.c_gu_g, self.c_gu_u, self.c_act, self.c_dout, self.c_ctr, self.c_out,
            sh_out, sh_w,
        )
        return self.c_out[:R]

    def coop_layer(self, layer: int, inp: dict):
        """Shared launch, then the routed one merging it: one MoE layer of a decode step"""
        if self.shared is not None:
            self.coop_shared(inp["x"], layer)
        return self.coop_routed(layer, inp, True)

    def fused_routed(self, layer: int, inp: dict, count_on_device: bool = False):
        """The prefill's fused tier at decode size, as forward() would run it for these rows"""
        t = self.torch
        ly = self.layers[layer]
        s = self.cfg["shapes"]
        x = inp["x"]
        R = x.shape[0]
        fhs_ext = t.zeros((R + 1, self.H), dtype = t.float, device = self.dev)
        out = fhs_ext[:R]
        lay = fused_layout(t, inp["sel"], inp["rw"], self.E, 0, inp["flat_token"], count_on_device)
        tables = lay["tables"]
        scratch = t.empty((max(lay["n_slots"], 1), self.H), dtype = t.float, device = self.dev)
        self.ext.exl3_moe(
            x, out, lay["expert_count"], lay["token_sorted"], lay["weight_sorted"],
            self.f_tg, self.f_tu, self.f_tig, self.f_tiu,
            MOE_ACT_SILU, float(s["K_gu"]), float(s["K_gu"]), float(s["K_d"]),
            *ly["tabs"]["g"], *ly["tabs"]["u"], *ly["tabs"]["d"],
            self.mcg, self.mul1, self.mcg, self.mul1, self.mcg, self.mul1,
            0.0, -1, scratch, tables[0], 1, self.f_rows, 16,
        )
        E = self.E
        self.ext.exl3_moe_gather(out, scratch, lay["local"], lay["inv_order"], tables[1, :E], tables[0, :E],
                                 tables[2, :E], lay["weight_sorted"])
        return out

    # ---- the reference: dense reconstruct of each expert (testlib.moe.exl3_linear_ref's steps)

    def _linear_ref(self, x, trellis, suh, svh, K: float, cb: int):
        t, ext = self.torch, self.ext
        xh = t.empty_like(x)
        ext.had_r_128(x, xh, suh, None, 1.0)
        W = t.empty((trellis.shape[0] * 16, trellis.shape[1] * 16), dtype = t.half, device = self.dev)
        ext.reconstruct(W, trellis, float(K), cb == 1, cb == 2)
        y = t.empty((x.shape[0], W.shape[1]), dtype = t.half, device = self.dev)
        ext.hgemm(xh, W, y)
        ext.had_r_128(y, y, None, svh, 1.0)
        return y

    def _mlp_ref(self, ex: dict, e: int, x, cb: int):
        g = self._linear_ref(x, *ex["g"][e], ex["K_gu"], cb)
        u = self._linear_ref(x, *ex["u"][e], ex["K_gu"], cb)
        self.ext.silu_mul(g, u, u, 0.0)
        return self._linear_ref(u, *ex["d"][e], ex["K_d"], cb).float()

    def reference(self, layer: int, inp: dict) -> dict:
        """{"routed": (R, H), "shared": (R, H) or None, "layer": routed + gate * shared} in fp32"""
        t = self.torch
        ly = self.layers[layer]
        x = inp["x"]
        R = x.shape[0]
        xp = x
        if self.Hi != self.H:
            xp = t.zeros((R, self.Hi), dtype = t.half, device = self.dev)
            xp[:, :self.H] = x
        cb = self.cfg["shapes"]["cb"]
        routed = t.zeros((R, self.H), dtype = t.float, device = self.dev)
        for b in range(R):
            for k in range(self.topk):
                e = int(inp["plan"][b][k])
                if e >= self.E:
                    continue
                w = float(inp["rw"][b, k])
                routed[b] += w * self._mlp_ref(ly, e, xp[b: b + 1], cb)[0, : self.H]
        out = {"routed": routed, "shared": None, "layer": routed.clone()}
        if self.shared is not None:
            shared = self._mlp_ref(self.shared[layer], 0, xp, self.cfg["shapes"].get("sh_cb", cb))[:, : self.H]
            gate = t.sigmoid((x.float() * self.sh_gate_w.float()).sum(1, keepdim = True))
            out["shared"] = shared
            out["layer"] = routed + gate * shared
        return out


# --------------------------------------------------------------------------------------------
#   Timing
# --------------------------------------------------------------------------------------------

def time_calls(torch, device, calls: list, iters: int, repeats: int, graph: bool) -> dict:
    """
    Time per call of `calls` cycled `iters` times: the GPU time from a CUDA graph of the calls
    (graph mode, no host in it) or from back-to-back calls (plain), and the host's issue time per
    call (plain, before any synchronization). On a CPU device the wall time of the calls.
    """
    dev = torch.device(device)
    cuda = dev.type == "cuda"
    out = {"iters": iters, "repeats": repeats}

    def issue():
        for i in range(iters):
            calls[i % len(calls)]()

    for f in calls:                        # warm-up: kernel attributes, allocator, graphs of the library
        f()
    if cuda:
        torch.cuda.synchronize(dev)

    # Plain: back-to-back calls
    gpu, host = [], []
    for _ in range(repeats):
        if cuda:
            a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            a.record()
        t0 = time.perf_counter()
        issue()
        t1 = time.perf_counter()
        if cuda:
            b.record()
            b.synchronize()
            gpu.append(a.elapsed_time(b) * 1000.0 / iters)
        else:
            gpu.append((t1 - t0) * 1e6 / iters)
        host.append((t1 - t0) * 1e6 / iters)
    out["plain_us"] = statistics.median(gpu)
    out["host_us"] = statistics.median(host)
    out["gpu_us"] = out["plain_us"]
    out["mode"] = "plain"

    if graph and cuda:
        try:
            s = torch.cuda.Stream(device = dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                for f in calls:
                    f()
            torch.cuda.current_stream(dev).wait_stream(s)
            torch.cuda.synchronize(dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                issue()
            g.replay()
            torch.cuda.synchronize(dev)
            ts = []
            for _ in range(repeats):
                a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
                a.record()
                g.replay()
                b.record()
                b.synchronize()
                ts.append(a.elapsed_time(b) * 1000.0 / iters)
            out["graph_us"] = statistics.median(ts)
            out["gpu_us"] = out["graph_us"]
            out["mode"] = "graph"
            del g
        except Exception as e:             # capture refused: the plain time stands
            out["graph_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            try:
                torch.cuda.synchronize(dev)
            except Exception:
                pass
    return out


# --------------------------------------------------------------------------------------------
#   The child process
# --------------------------------------------------------------------------------------------

PROBE_ROWS = (1, 2, 4, 8)


def append_jsonl(path: str, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def run_child(cfg: dict, setting: str, out_dir: str, torch = None, ext = None, on_tensor = None) -> int:
    """
    Everything one setting's process measures, its records appended to records.jsonl and its probe
    outputs saved to probes-<setting>.pt. torch / ext / on_tensor are injected by the CPU tests.
    """
    if torch is None:
        import torch as _torch
        torch = _torch
    if ext is None:
        sys.path.insert(0, REPO)
        from exllamav3.ext import exllamav3_ext as _ext
        ext = _ext
    rec_path = os.path.join(out_dir, "records.jsonl")
    device = cfg["device"]
    dev = torch.device(device)
    env = {k: os.environ.get(k) for k in KNOB_VARS}
    head = {"kind": "proc", "setting": setting, "env": env, "pid": os.getpid(), "torch": torch.__version__,
            "device": device, "t": time.time()}
    if dev.type == "cuda":
        head["gpu"] = torch.cuda.get_device_name(dev)
        head["sms"] = torch.cuda.get_device_properties(dev).multi_processor_count
    try:
        bench = Bench(torch, ext, cfg, device, on_tensor)
    except Exception as e:
        head.update(error = f"{type(e).__name__}: {e}")
        append_jsonl(rec_path, head)
        return 1
    head["weights_digest"] = bench.weights_digest()
    if dev.type == "cuda":
        head["weights_bytes"] = int(sum(tr.numel() * 2 + a.numel() * 2 + b.numel() * 2
                                        for ly in bench.layers for p in ("g", "u", "d") for tr, a, b in ly[p]))
    append_jsonl(rec_path, head)

    hit = cfg["hit"]
    L = cfg["layers"]
    n_sets = cfg["sets"]
    fused = setting == "default" and cfg.get("fused", True) and bench.fused_ok
    paths = ["coop_routed", "coop_layer"] + (["coop_shared"] if bench.shared is not None else []) + \
            (["fused_routed"] if fused else [])
    sms, blackwell = 36, True
    if dev.type == "cuda":
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        blackwell = torch.cuda.get_device_capability(dev)[0] >= 10
    knobs = launcher_knobs(os.environ)
    status = 0
    for pattern in cfg["patterns"]:
        for R in cfg["rows"]:
            try:
                sets = [bench.inputs(pattern, R, s, hit) for s in range(n_sets)]
            except ValueError as e:
                append_jsonl(rec_path, {"kind": "skip", "setting": setting, "pattern": pattern, "rows": R, "why": str(e)})
                continue
            for path in paths:
                if path == "coop_shared" and pattern != cfg["patterns"][0]:
                    continue                    # the shared launch does not depend on the selection
                calls = []
                for i in range(n_sets * L if n_sets * L <= 64 else 64):
                    inp, layer = sets[i % n_sets], i % L
                    if path == "coop_routed":
                        calls.append(lambda inp = inp, layer = layer: bench.coop_routed(layer, inp, True))
                    elif path == "coop_layer":
                        calls.append(lambda inp = inp, layer = layer: bench.coop_layer(layer, inp))
                    elif path == "coop_shared":
                        calls.append(lambda inp = inp, layer = layer: bench.coop_shared(inp["x"], layer))
                    else:
                        calls.append(lambda inp = inp, layer = layer: bench.fused_routed(layer, inp, cfg.get("graph", True)))
                rec = {"kind": "time", "setting": setting, "path": path, "pattern": pattern, "rows": R,
                       "slots": R * bench.topk, "distinct": statistics.mean(st["distinct"] for st in sets),
                       "picks": statistics.mean(st["picks"] for st in sets), "calls": len(calls)}
                if path in ("coop_routed", "coop_layer"):
                    rec["geometry"] = coop_geometry(bench.Hi, bench.I, bench.Ho, R * bench.topk, bench.slots_max,
                                                    True, sms, blackwell, **knobs)
                if path in ("coop_shared", "coop_layer") and bench.shared is not None:
                    rec["geometry_shared"] = coop_geometry(bench.Hi, bench.shared[0]["I"], bench.Ho, R, MAX_BSZN,
                                                           True, sms, blackwell, **knobs)
                try:
                    rec.update(time_calls(torch, device, calls, cfg["iters"], cfg["repeats"], cfg.get("graph", True)))
                    if path == "fused_routed" and cfg.get("graph", True):
                        # forward()'s own layout counts with torch.bincount (host syncs): its host time
                        plain = [lambda inp = sets[i % n_sets], layer = i % L: bench.fused_routed(layer, inp, False)
                                 for i in range(len(calls))]
                        t2 = time_calls(torch, device, plain, cfg["iters"], cfg["repeats"], False)
                        rec["bincount_plain_us"] = t2["plain_us"]
                        rec["bincount_host_us"] = t2["host_us"]
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                    status = 1
                append_jsonl(rec_path, rec)

    # Probes: the outputs of a few cases (layer 0, set 0), against the reference in the default process
    probes = {"setting": setting, "weights_digest": bench.weights_digest(), "cases": {}}
    for pattern in cfg["probe_patterns"]:
        for R in cfg["probe_rows"]:
            try:
                inp = bench.inputs(pattern, R, 0, hit)
            except ValueError:
                continue
            key = f"{pattern}/{R}"
            case = {"plan": inp["plan"]}
            try:
                case["coop_routed"] = bench.coop_routed(0, inp, False).clone().float().cpu()
                case["coop_layer"] = bench.coop_layer(0, inp).clone().float().cpu()
                if fused:
                    case["fused_routed"] = bench.fused_routed(0, inp, False).clone().float().cpu()
                if setting == "default":
                    ref = bench.reference(0, inp)
                    case["ref_routed"] = ref["routed"].cpu()
                    case["ref_layer"] = ref["layer"].cpu()
            except Exception as e:
                case["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                status = 1
            probes["cases"][key] = case
    torch.save(probes, os.path.join(out_dir, f"probes-{setting}.pt"))
    append_jsonl(rec_path, {"kind": "done", "setting": setting, "status": status, "t": time.time()})
    return status


# --------------------------------------------------------------------------------------------
#   Accuracy over the probes (torch on the CPU, in the orchestrator)
# --------------------------------------------------------------------------------------------

def row_errors(test, ref) -> list:
    """Per row: max |test - ref| over the row, divided by max |ref| over the row"""
    out = []
    for b in range(ref.shape[0]):
        scale = float(ref[b].abs().max())
        err = float((test[b] - ref[b]).abs().max())
        out.append(err / scale if scale > 0 else (0.0 if err == 0 else float("inf")))
    return out


def compare_probes(probe_sets: dict) -> list:
    """
    probe_sets: {setting: probes dict}. Rows of the accuracy table: per case and comparison, the
    worst row's relative error and whether every row is within TOL_ROW.
    """
    out = []
    base = probe_sets.get("default")
    if base is None:
        return out
    for key, bc in base["cases"].items():
        comps = []
        if "ref_routed" in bc:
            for path in ("coop_routed", "fused_routed"):
                if path in bc:
                    comps.append((f"{path} / riferimento", "default", path, bc["ref_routed"], bc[path]))
            if "coop_layer" in bc and "ref_layer" in bc:
                comps.append(("coop_layer / riferimento", "default", "coop_layer", bc["ref_layer"], bc["coop_layer"]))
        if "fused_routed" in bc and "coop_routed" in bc:
            comps.append(("fused_routed / coop_routed", "default", "fused_routed", bc["coop_routed"], bc["fused_routed"]))
        for setting, ps in probe_sets.items():
            if setting == "default":
                continue
            c = ps["cases"].get(key) or {}
            if c.get("plan") != bc.get("plan"):
                out.append({"case": key, "what": f"{setting}: selezione diversa dal default", "setting": setting,
                            "worst": None, "ok": False})
                continue
            for path in ("coop_routed", "coop_layer"):
                if path in c and path in bc:
                    comps.append((f"{path} {setting} / default", setting, path, bc[path], c[path]))
                    rkey = "ref_layer" if path == "coop_layer" else "ref_routed"
                    if rkey in bc:
                        comps.append((f"{path} {setting} / riferimento", setting, path, bc[rkey], c[path]))
        for what, setting, path, ref, test in comps:
            errs = row_errors(test, ref)
            out.append({"case": key, "what": what, "setting": setting, "path": path, "rows": len(errs),
                        "worst": max(errs) if errs else 0.0, "errs": errs,
                        "ok": all(e <= TOL_ROW for e in errs)})
        if "error" in bc:
            out.append({"case": key, "what": f"default: {bc['error']}", "setting": "default", "worst": None, "ok": False})
    return out


# --------------------------------------------------------------------------------------------
#   Report (Italian)
# --------------------------------------------------------------------------------------------

def fmt(x, nd = 1):
    if x is None:
        return "–"
    if isinstance(x, float):
        if math.isinf(x):
            return "∞"
        return f"{x:.{nd}f}"
    return str(x)


def fmt_e(x):
    return "–" if x is None else ("∞" if math.isinf(x) else f"{x:.1e}")


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def linear_fit(points: list) -> tuple | None:
    """Least squares y = a + b x over (x, y) points; None with fewer than two distinct x"""
    if len({x for x, _ in points}) < 2:
        return None
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    sxy = sum((x - mx) * (y - my) for x, y in points)
    b = sxy / sxx
    return my - b * mx, b


def geometry_text(g: dict | None) -> str:
    if not g:
        return "–"
    def one(w, grid, ks):
        return f"{'larga' if w else 'stretta'} {grid}" + (f" (split {ks})" if ks > 1 else "")
    return f"{one(g['a_wide'], g['a_grid'], g['a_ksplit'])} / {one(g['b_wide'], g['b_grid'], g['b_ksplit'])}"


def times_by(records: list) -> dict:
    """{(setting, path, pattern, rows): record} of the timing records"""
    return {(r["setting"], r["path"], r["pattern"], r["rows"]): r for r in records if r.get("kind") == "time"}


def render_report(cfg: dict, records: list, accuracy: list, finished: bool) -> str:
    s = cfg["shapes"]
    procs = [r for r in records if r.get("kind") == "proc"]
    done = {r["setting"]: r for r in records if r.get("kind") == "done"}
    tb = times_by(records)
    settings = [st for st in cfg["settings"] if any(p["setting"] == st for p in procs)]
    rows_list = cfg["rows"]
    patterns = cfg["patterns"]
    expert_mb = (2 * s["Hi"] * s["I"] * s["K_gu"] + s["I"] * s["Ho"] * s["K_d"]) / 8 / 1e6
    L = []
    L.append("# MoE del decode: un esperto per passo, misurato fuori dal modello")
    L.append("")
    L.append("Generato da `experiments/moe_decode_bench.py`" + ("" if finished else " (**non finito**: un processo manca o è in corso)") + ".")
    L.append("")
    L.append("## Condizioni")
    L.append("")
    head = procs[0] if procs else {}
    L.append(f"- scheda: {head.get('gpu', '?')} ({head.get('sms', '?')} SM), `{cfg['device']}`, torch {head.get('torch', '?')}")
    src = s.get("source", "default")
    L.append(f"- forme: H {s['H']} (Hi {s['Hi']}, Ho {s['Ho']}), I {s['I']}, top-k {s['topk']}, "
             f"esperti instradati a {s['K_gu']:g}/{s['K_d']:g} bit (gate-up/down), codebook {CODEBOOKS[s['cb']]}, "
             f"esperto condiviso " + (f"I {s['sh_I']} a {s['sh_K_gu']:g}/{s['sh_K_d']:g} bit" if s.get("sh_I") else "assente")
             + f"; da {'`' + src + '`' if src != 'default' else 'i valori di produzione scritti nello script'}"
             + (f" (layer {s['layer']}; combinazioni di bit sui {s.get('moe_layers')} layer MoE: {', '.join(s.get('k_combos', []))})" if "layer" in s else ""))
    L.append(f"- un esperto pesa {expert_mb:.2f} MB; {cfg['layers']} copie di {cfg['experts']} esperti "
             f"({cfg['layers'] * cfg['experts'] * expert_mb / 1e3:.2f} GB) perché chiamate consecutive leggano pesi fuori dalla L2; "
             f"{cfg['sets']} selezioni per caso; quota delle scelte su esperti in VRAM {cfg['hit']:g} (il resto è fuori tabella, come gli esperti sul worker CPU)")
    L.append(f"- tempi: mediana di {cfg['repeats']} ripetizioni di {cfg['iters']} chiamate; «GPU» dal grafo CUDA delle chiamate "
             "(senza host) quando la cattura riesce, altrimenti chiamate di fila; «host» il tempo di emissione per chiamata")
    for p in procs:
        env = ", ".join(f"{k}={v}" for k, v in (p.get("env") or {}).items() if v is not None) or "nessuna variabile"
        st = done.get(p["setting"])
        state = "finito" if st and st.get("status") == 0 else ("finito con errori" if st else "**non finito**")
        L.append(f"- processo `{p['setting']}`: {env}; {state}" + (f"; errore: {p['error']}" if p.get("error") else ""))
    digests = {p.get("weights_digest") for p in procs if p.get("weights_digest") is not None}
    if len(digests) > 1:
        L.append("- **Attenzione**: i processi non hanno costruito gli stessi pesi (checksum diversi): i confronti fra impostazioni non valgono")
    graph_err = [r for r in records if r.get("kind") == "time" and r.get("graph_error")]
    if graph_err:
        L.append(f"- **Attenzione**: cattura del grafo rifiutata in {len(graph_err)} casi (tempi di chiamate di fila), "
                 f"per esempio `{graph_err[0]['path']}`: {graph_err[0]['graph_error']}")
    errs = [r for r in records if r.get("kind") == "time" and r.get("error")]
    if errs:
        L.append(f"- **Attenzione**: {len(errs)} misure fallite, per esempio `{errs[0]['path']}` {errs[0]['pattern']} "
                 f"{errs[0]['rows']} righe: {errs[0]['error']}")
    L.append("")

    # 1. Read once
    L.append("## 1. Un esperto scelto da più righe si legge una volta")
    L.append("")
    L.append("Lancio instradato del decode (`coop_routed`, impostazione di default), µs GPU per layer. "
             "«uguali» è R righe che scelgono gli stessi esperti, «distinte» R righe con esperti tutti diversi: stesse righe, "
             "stessi slot, stessa griglia, cambiano solo gli esperti distinti da leggere. Se il kernel leggesse un esperto per "
             "riga che lo sceglie, «uguali» costerebbe come «distinte»; se lo legge una volta per chiamata, «uguali» resta "
             "vicino al suo costo a 2 righe (cresce solo il lavoro per riga: rotazione dell'ingresso, epiloghi). A una riga "
             "il kernel A ruota l'ingresso dentro ogni blocco, senza il pre-kernel: un percorso a parte.")
    L.append("")
    rows = []
    u2 = tb.get(("default", "coop_routed", "same", 2))
    for R in rows_list:
        d = tb.get(("default", "coop_routed", "distinct", R))
        u = tb.get(("default", "coop_routed", "same", R))
        ok = lambda a, b: a is not None and b is not None and a.get("gpu_us") and b.get("gpu_us")
        rows.append([R, fmt(d and d["distinct"]), fmt(d and d.get("gpu_us")), fmt(u and u["distinct"]), fmt(u and u.get("gpu_us")),
                     fmt(u["gpu_us"] / d["gpu_us"] if ok(u, d) else None, 2),
                     fmt(u["gpu_us"] / u2["gpu_us"] if ok(u, u2) else None, 2),
                     geometry_text((d or u or {}).get("geometry"))])
    L.append(table(["righe", "esperti GPU distinti (distinte)", "µs (distinte)", "esperti GPU distinti (uguali)", "µs (uguali)",
                    "uguali / distinte", "uguali / uguali a 2 righe", "tile A / B (blocchi)"], rows))
    L.append("")

    # 2. Cost model
    L.append("## 2. Quanto costa un esperto in più, e quanto il resto")
    L.append("")
    L.append("Retta µs = a + b × esperti distinti sui casi a 2-8 righe (a una riga il kernel ruota l'ingresso dentro ogni blocco, "
             "un percorso a parte), per impostazione e lancio; «GB/s marginali» sono i byte di un esperto in b µs.")
    L.append("")
    rows = []
    for st in settings:
        for path in ("coop_routed", "fused_routed"):
            pts = [(r["distinct"], r["gpu_us"]) for (s_, p_, pat, R), r in tb.items()
                   if s_ == st and p_ == path and R >= 2 and r.get("gpu_us") is not None]
            fit = linear_fit(pts)
            if fit is None:
                continue
            a, b = fit
            rows.append([f"`{st}`", f"`{path}`", len(pts), fmt(a), fmt(b, 2), fmt(expert_mb * 1e6 / (b * 1e3) if b > 0 else None, 0)])
    if rows:
        L.append(table(["impostazione", "lancio", "punti", "a (µs)", "b (µs per esperto)", "GB/s marginali"], rows))
    else:
        L.append("(nessun dato)")
    L.append("")

    # 3. Settings
    L.append("## 3. Le manopole del lanciatore coop")
    L.append("")
    L.append("`coop_layer` (condiviso + instradato, un layer MoE del passo), µs GPU per layer, per schema e righe; "
             "tra parentesi il rapporto con il default.")
    L.append("")
    rows = []
    for pattern in patterns:
        for R in rows_list:
            base = tb.get(("default", "coop_layer", pattern, R))
            if base is None:
                continue
            row = [pattern_label(pattern), R, fmt(base["distinct"])]
            for st in settings:
                r = tb.get((st, "coop_layer", pattern, R))
                if r is None or r.get("gpu_us") is None:
                    row.append("–")
                elif st == "default":
                    row.append(fmt(r["gpu_us"]))
                else:
                    row.append(f"{fmt(r['gpu_us'])} ({fmt(r['gpu_us'] / base['gpu_us'], 2) if base.get('gpu_us') else '–'})")
            rows.append(row)
    L.append(table(["schema", "righe", "esperti distinti"] + [f"`{st}`" for st in settings], rows) if rows else "(nessun dato)")
    L.append("")
    geo = []
    for R in rows_list:
        row = [R]
        for st in settings:
            r = tb.get((st, "coop_layer", patterns[0], R))
            row.append(geometry_text(r and r.get("geometry")) + " · condiviso " + geometry_text(r and r.get("geometry_shared"))
                       if r and r.get("geometry_shared") else geometry_text(r and r.get("geometry")))
        geo.append(row)
    L.append("La geometria che il lanciatore sceglie (lancio instradato, poi quello del condiviso), dal suo stesso calcolo "
             "riprodotto in Python:")
    L.append("")
    L.append(table(["righe"] + [f"`{st}`" for st in settings], geo))
    L.append("")
    sh_rows = []
    for R in rows_list:
        row = [R]
        for st in settings:
            r = tb.get((st, "coop_shared", patterns[0], R))
            row.append(fmt(r and r.get("gpu_us")))
        sh_rows.append(row)
    if any(c != "–" for r in sh_rows for c in r[1:]):
        L.append("Il lancio dell'esperto condiviso da solo (`coop_shared`), µs GPU:")
        L.append("")
        L.append(table(["righe"] + [f"`{st}`" for st in settings], sh_rows))
        L.append("")

    # 4. Fused tier
    L.append("## 4. Il livello fuso del prefill alle dimensioni del decode")
    L.append("")
    L.append("`fused_routed` (ext.exl3_moe + ext.exl3_moe_gather, la disposizione che forward() costruisce senza leggere i "
             "conteggi) contro `coop_routed`, impostazione di default. «host» è l'emissione di una chiamata; «bincount» "
             "il tempo di chiamate di fila col `torch.bincount` di forward(), che su CUDA legge min e max sull'host.")
    L.append("")
    rows = []
    for pattern in patterns:
        for R in rows_list:
            c = tb.get(("default", "coop_routed", pattern, R))
            f = tb.get(("default", "fused_routed", pattern, R))
            if f is None:
                continue
            rows.append([pattern_label(pattern), R, fmt(f["distinct"]), fmt(c and c.get("gpu_us")), fmt(f.get("gpu_us")),
                         fmt(c and c.get("host_us")), fmt(f.get("host_us")), fmt(f.get("bincount_plain_us"))])
    L.append(table(["schema", "righe", "esperti distinti", "coop µs GPU", "fuso µs GPU", "coop µs host", "fuso µs host",
                    "fuso con bincount, µs di fila"], rows) if rows else "(non misurato)")
    L.append("")

    # 5. Accuracy
    L.append("## 5. Stessa computazione")
    L.append("")
    L.append(f"Per riga: max |differenza| / max |riferimento| della riga; ogni riga deve stare entro {TOL_ROW:g} (il limite "
             "dei test upstream del kernel coop contro il riferimento denso). Il riferimento ricostruisce ogni esperto denso "
             "(reconstruct + GEMM fp16), come i test del kernel.")
    L.append("")
    if accuracy:
        rows = [[a["case"], a["what"], a.get("rows", "–"), fmt_e(a.get("worst")), "sì" if a["ok"] else "**no**"] for a in accuracy]
        L.append(table(["caso", "confronto", "righe", "riga peggiore", "entro"], rows))
        bad = [a for a in accuracy if not a["ok"]]
        L.append("")
        L.append(f"{len(accuracy) - len(bad)} confronti su {len(accuracy)} entro il limite." if accuracy else "")
    else:
        L.append("(nessun confronto: i file probes-*.pt mancano o sono illeggibili)")
    L.append("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------
#   Orchestrator
# --------------------------------------------------------------------------------------------

def child_env(base: dict, setting: str) -> dict:
    """The child's environment: the knobs of every setting removed, this setting's set"""
    env = {k: v for k, v in base.items() if k not in KNOB_VARS}
    env.update(SETTINGS[setting])
    env["PYTHONPATH"] = REPO + (os.pathsep + base["PYTHONPATH"] if base.get("PYTHONPATH") else "")
    return env


def vram_estimate(cfg: dict) -> int:
    s = cfg["shapes"]
    per_expert = (2 * s["Hi"] * s["I"] * s["K_gu"] + s["I"] * s["Ho"] * s["K_d"]) / 8
    shared = 0
    if s.get("sh_I"):
        shared = (2 * s["Hi"] * s["sh_I"] * s["sh_K_gu"] + s["sh_I"] * s["Ho"] * s["sh_K_d"]) / 8
    return int(cfg["layers"] * (cfg["experts"] * per_expert + shared) + 64 * 2 ** 20)


def time_estimate(cfg: dict) -> float:
    """Seconds, roughly: per setting the cases times paths times (warm-up, plain, graph), plus the reference"""
    cases = len(cfg["patterns"]) * len(cfg["rows"])
    per_case = 3 * cfg["repeats"] * cfg["iters"] * 150e-6 + 0.3
    total = 0.0
    for st in cfg["settings"]:
        paths = 3 + (1 if st == "default" and cfg.get("fused", True) else 0)
        total += 10 + cases * paths * per_case
    total += len(cfg["probe_patterns"]) * len(cfg["probe_rows"]) * 3 + 20
    return total


def load_probes(out_dir: str, settings: list) -> dict:
    out = {}
    try:
        import torch
    except Exception:
        return out
    for st in settings:
        p = os.path.join(out_dir, f"probes-{st}.pt")
        if os.path.exists(p):
            try:
                out[st] = torch.load(p, map_location = "cpu", weights_only = False)
            except Exception:
                pass
    return out


def write_report(cfg: dict, out_dir: str, finished: bool) -> str:
    records = read_jsonl(os.path.join(out_dir, "records.jsonl"))
    accuracy = compare_probes(load_probes(out_dir, cfg["settings"]))
    text = render_report(cfg, records, accuracy, finished)
    path = os.path.join(out_dir, "REPORT.md")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def run_all(cfg: dict, out_dir: str, python: str, child_cmd = None) -> int:
    """One child per setting, in order; the report after each. child_cmd (tests) replaces the real command"""
    status = 0
    cfg_path = os.path.join(out_dir, "config.json")
    for st in cfg["settings"]:
        cmd = (child_cmd or [python, os.path.abspath(__file__)]) + ["--child", cfg_path, "--setting", st]
        log = os.path.join(out_dir, f"child-{st}.log")
        print(f" -- {st}: {' '.join(f'{k}={v}' for k, v in SETTINGS[st].items()) or 'nessuna variabile'}", flush = True)
        t0 = time.time()
        with open(log, "w") as lf:
            rc = subprocess.call(cmd, env = child_env(dict(os.environ), st), stdout = lf, stderr = subprocess.STDOUT)
        print(f"    uscito con {rc} in {time.time() - t0:.0f} s (log: {log})", flush = True)
        if rc != 0:
            status = 1
        write_report(cfg, out_dir, False)
    path = write_report(cfg, out_dir, True)
    print(f" -- report: {path}", flush = True)
    return status


def int_list(text: str) -> list:
    return [int(x) for x in str(text).split(",") if x.strip()]


def parse_cli(argv: list | None = None):
    ap = argparse.ArgumentParser(description = "Banco dei kernel MoE del decode (1-8 righe) su una scheda, fuori dal modello")
    ap.add_argument("--out", help = "cartella del run (nuova o vuota)")
    ap.add_argument("--model", help = "cartella del modello EXL3: forme e bit dagli header safetensors (niente pesi caricati)")
    ap.add_argument("--model-layer", type = int, default = None, help = "layer MoE da cui prendere le forme (default il primo)")
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--settings", default = ",".join(SETTINGS), help = f"impostazioni, fra {', '.join(SETTINGS)}")
    ap.add_argument("--rows", default = "1,2,3,4,5,6,7,8")
    ap.add_argument("--patterns", default = "distinct,same,overlap:0.3", help = "distinct, same, overlap:F")
    ap.add_argument("--probe-rows", default = ",".join(str(r) for r in PROBE_ROWS))
    ap.add_argument("--probe-patterns", default = "distinct,same")
    ap.add_argument("--hit", type = float, default = 0.87, help = "quota delle scelte su esperti in VRAM")
    ap.add_argument("--experts", type = int, default = 96, help = "esperti per copia di layer (>= righe x top-k)")
    ap.add_argument("--layers", type = int, default = 6, help = "copie di layer, perché la L2 non tenga i pesi")
    ap.add_argument("--sets", type = int, default = 8, help = "selezioni diverse per caso")
    ap.add_argument("--iters", type = int, default = 64)
    ap.add_argument("--repeats", type = int, default = 7)
    ap.add_argument("--seed", type = int, default = 1)
    ap.add_argument("--no-graph", action = "store_true", help = "niente grafi CUDA: solo chiamate di fila")
    ap.add_argument("--no-fused", action = "store_true", help = "non misurare il livello fuso del prefill")
    ap.add_argument("--dry-run", action = "store_true", help = "forme, piano, VRAM e durata stimate; niente GPU")
    ap.add_argument("--child", help = argparse.SUPPRESS)
    ap.add_argument("--setting", help = argparse.SUPPRESS)
    ap.add_argument("--report", help = "rifà REPORT.md dai record di una cartella")
    return ap.parse_args(argv)


def build_config(args) -> dict:
    shapes = dict(DEFAULT_SHAPES)
    if args.model:
        shapes = model_shapes(args.model, args.model_layer)
    settings = [s for s in args.settings.split(",") if s]
    for s in settings:
        if s not in SETTINGS:
            raise SystemExit(f"impostazione sconosciuta: {s} (fra {', '.join(SETTINGS)})")
    if "default" not in settings:
        settings = ["default"] + settings
    patterns = [p for p in args.patterns.split(",") if p]
    for p in patterns + [p for p in args.probe_patterns.split(",") if p]:
        parse_pattern(p)
    rows = int_list(args.rows)
    if not rows or min(rows) < 1 or max(rows) > MAX_BSZN:
        raise SystemExit(f"righe fra 1 e {MAX_BSZN}")
    need = max(rows) * shapes["topk"]
    if args.experts < need:
        raise SystemExit(f"--experts {args.experts}: le selezioni «distinte» a {max(rows)} righe ne vogliono {need}")
    return {
        "shapes": shapes, "device": args.device, "settings": settings, "rows": rows, "patterns": patterns,
        "probe_rows": [r for r in int_list(args.probe_rows) if 1 <= r <= MAX_BSZN],
        "probe_patterns": [p for p in args.probe_patterns.split(",") if p],
        "hit": args.hit, "experts": args.experts, "layers": args.layers, "sets": args.sets,
        "iters": args.iters, "repeats": args.repeats, "seed": args.seed, "graph": not args.no_graph,
        "fused": not args.no_fused,
    }


def dry_run_text(cfg: dict) -> str:
    s = cfg["shapes"]
    lines = [
        f"forme ({s.get('source')}): H {s['H']} Hi {s['Hi']} I {s['I']} Ho {s['Ho']}, top-k {s['topk']}, "
        f"instradati {s['K_gu']:g}/{s['K_d']:g} bit, codebook {CODEBOOKS[s['cb']]}, condiviso "
        + (f"I {s['sh_I']} a {s['sh_K_gu']:g}/{s['sh_K_d']:g} bit" if s.get("sh_I") else "assente"),
    ]
    if s.get("k_combos"):
        lines.append(f"combinazioni di bit degli esperti sui {s['moe_layers']} layer MoE: {', '.join(s['k_combos'])} (forme dal layer {s['layer']})")
    lines += [
        f"impostazioni (un processo ciascuna): {', '.join(cfg['settings'])}",
        f"righe {cfg['rows']}, schemi {cfg['patterns']}, quota in VRAM {cfg['hit']:g}",
        f"VRAM: circa {vram_estimate(cfg) / 2 ** 30:.2f} GiB su {cfg['device']}",
        f"durata: circa {time_estimate(cfg) / 60:.1f} minuti",
    ]
    return "\n".join(lines)


def main(argv: list | None = None) -> int:
    args = parse_cli(argv)
    if args.child:
        with open(args.child) as f:
            cfg = json.load(f)
        return run_child(cfg, args.setting, os.path.dirname(os.path.abspath(args.child)))
    if args.report:
        with open(os.path.join(args.report, "config.json")) as f:
            cfg = json.load(f)
        print(write_report(cfg, args.report, True))
        return 0
    cfg = build_config(args)
    if args.dry_run:
        print(dry_run_text(cfg))
        return 0
    if not args.out:
        raise SystemExit("--out obbligatorio")
    out = os.path.abspath(args.out)
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} esiste e non è vuota")
    os.makedirs(out, exist_ok = True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent = 1)
    print(dry_run_text(cfg), flush = True)
    return run_all(cfg, out, sys.executable)


if __name__ == "__main__":
    sys.exit(main())
