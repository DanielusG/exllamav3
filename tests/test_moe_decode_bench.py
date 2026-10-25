"""
The decode MoE bench (experiments/moe_decode_bench.py) on this machine: no GPU, no extension.

- the selections (which experts each row picks, how the rows overlap, the GPU/CPU split of the
  picks), the fused tier's decode-size layout (every routed pick of every row exactly once, the
  results back in row order), the coop launcher's geometry against the grids of the night
  profiling's nsys capture, the shapes read from a checkpoint's safetensors headers;
- every path of the bench end to end on the CPU over a fake extension whose functions take
  exactly the arguments of the C++ declarations in exllamav3_ext (parsed from the headers, so a
  wrong argument count or type fails here and not in the GPU window) and compute the same expert
  MLPs as the reference: coop routed and shared, the fused tier, the reference, the accuracy
  comparison catching a wrong row;
- the orchestrator with a fake child: one process per setting, the knob variables of the parent
  never leaking into a child, the report.

What only the GPU window can tell: the times, and that the real kernels agree with the reference.
"""

import json
import math
import os
import random
import re
import struct
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(REPO, "experiments")
EXT_SRC = os.path.join(REPO, "exllamav3", "exllamav3_ext")
sys.path.insert(0, EXP)
import moe_decode_bench as MB       # noqa: E402


# --------------------------------------------------------------------------------------------
#   A fake extension, its signatures taken from the C++ declarations
# --------------------------------------------------------------------------------------------

DECLS = {
    "exl3_moe_coop": "quant/exl3_moe_coop.cuh",
    "exl3_moe": "quant/exl3_moe.cuh",
    "exl3_moe_gather": "quant/exl3_moe.cuh",
    "exl3_moe_max_concurrency": "quant/exl3_moe.cuh",
    "had_r_128": "quant/hadamard.cuh",
    "reconstruct": "quant/reconstruct.cuh",
    "hgemm": "hgemm.cuh",
    "silu_mul": "activation.cuh",
}


def _split_top(text):
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [p.strip() for p in out if p.strip()]


def cpp_params(func):
    """[(kind, name)] of a function's declaration: kind in tensor, optional, int, float, bool"""
    text = open(os.path.join(EXT_SRC, DECLS[func])).read()
    text = re.sub(r"//[^\n]*", "", text)
    m = re.search(r"\b(?:void|int)\s+" + func + r"\s*\(([^;{]*?)\)\s*;", text, re.S)
    assert m, f"declaration of {func} not found"
    out = []
    for p in _split_top(m.group(1)):
        name = re.findall(r"[A-Za-z_]\w*", p)[-1]
        typ = p[: p.rfind(name)]
        if "optional" in typ:
            kind = "optional"
        elif "Tensor" in typ:
            kind = "tensor"
        elif re.search(r"\bbool\b", typ):
            kind = "bool"
        elif re.search(r"\b(int|int64_t)\b", typ):
            kind = "int"
        elif re.search(r"\b(float|double)\b", typ):
            kind = "float"
        else:
            raise AssertionError(f"{func}: unknown parameter type {typ!r}")
        out.append((kind, name))
    return out


def check_args(func, args):
    params = cpp_params(func)
    assert len(args) == len(params), f"{func}: {len(args)} arguments, the declaration has {len(params)}"
    for a, (kind, name) in zip(args, params):
        ok = {
            "tensor": isinstance(a, torch.Tensor),
            "optional": a is None or isinstance(a, torch.Tensor),
            "int": isinstance(a, int) and not isinstance(a, bool),
            "float": isinstance(a, (int, float)) and not isinstance(a, bool),
            "bool": isinstance(a, bool),
        }[kind]
        assert ok, f"{func}: argument {name} ({kind}) got {type(a).__name__}"


def _check(cond, msg):
    if not cond:
        raise RuntimeError(msg)


class FakeExt:
    """
    The extension functions the bench calls, with the C++ signatures and the checks of the real
    ones that the bench's buffers must pass, computing the routed expert MLPs in torch. The weights
    are resolved from the pointer tables through the tensors the bench reports (track). A trellis
    decodes to a fixed dense matrix; the Hadamard rotations are identities, consistently in every
    path, so the coop, fused and reference outputs agree to fp16 rounding.
    """

    def __init__(self):
        self.reg = {}
        self.calls = {}

    def track(self, t):
        self.reg[t.data_ptr()] = t

    def _count(self, name, args):
        check_args(name, args)
        self.calls[name] = self.calls.get(name, 0) + 1

    @staticmethod
    def dense(trellis):
        kt, nt, words = trellis.shape
        idx = (torch.arange(16)[:, None] * 16 + torch.arange(16)[None, :]) % words
        t = trellis.to(torch.float32) / 32768.0
        tiles = t[:, :, idx]                                     # (kt, nt, 16, 16)
        return tiles.permute(0, 2, 1, 3).reshape(kt * 16, nt * 16) / math.sqrt(kt * 16)

    def _t(self, table, i):
        return self.reg[int(table[i])]

    def _lin(self, x, tabs, l):
        tr, suh, svh = (self._t(tabs[0], l), self._t(tabs[1], l), self._t(tabs[2], l))
        return ((x.float() * suh.float()) @ self.dense(tr)) * svh.float()

    def _mlp(self, x, g, u, d, l, H_out):
        a = (F.silu(self._lin(x, g, l)) * self._lin(x, u, l)).half()
        return self._lin(a, d, l)[..., :H_out]

    # ---- the functions

    def exl3_moe_max_concurrency(self, *args):
        self._count("exl3_moe_max_concurrency", args)
        return 4

    def exl3_moe_coop(self, *args):
        self._count("exl3_moe_coop", args)
        (x, sel, rw, min_e, max_e, Hi, gt, gs, gv, ut, us, uv, dt, ds, dv, gb, ub, db, Kg, Ku, Kd, mcg, mul1,
         act, limit, gated, had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out, sh_out, sh_w) = args
        # exl3_moe_coop (front end), exl3_moe_coop_prepare, exl3_moe_coop_run
        _check(x.dtype == torch.half and sel.dtype == torch.long and rw.dtype == torch.half, "dtypes")
        _check(sel.is_contiguous() and rw.is_contiguous() and sel.shape == rw.shape and sel.dim() == 2
               and sel.shape[0] == x.shape[0], "sel/rw")
        _check(ctr.dtype == torch.int and ctr.is_contiguous(), "ctr")
        for tb in (gt, gs, gv, ut, us, uv, dt, ds, dv):
            _check(tb.dtype == torch.long and tb.is_contiguous(), "pointer tables")
        _check(not (mcg and mul1) and 1 <= Kg <= 8 and 1 <= Ku <= 8 and 1 <= Kd <= 8 and (not gated or Kg == Ku), "K / codebook")
        I, Ho, H_out = gu_u.shape[-1], d_out.shape[-1], out.shape[-1]
        _check(Hi % 128 == 0 and I % 128 == 0 and Ho % 128 == 0 and H_out <= Ho, "Hi/I/Ho")
        for t_, dt_ in ((had_g, torch.half), (had_u, torch.half), (act_out, torch.half), (d_out, torch.float), (out, torch.float)):
            _check(t_.dtype == dt_, "scratch dtype")
        _check(gu_g.dtype == gu_u.dtype and gu_u.dtype in (torch.half, torch.float), "gu dtype")
        slots_max = min(had_g.numel() // Hi, had_u.numel() // Hi, gu_g.numel() // I, gu_u.numel() // I,
                        act_out.numel() // I, d_out.numel() // Ho)
        for t_, w in ((had_g, Hi), (had_u, Hi), (gu_g, I), (gu_u, I), (act_out, I), (d_out, Ho)):
            _check(t_.is_contiguous() and t_.shape[-1] == w, "scratch shape")
        _check(out.dim() >= 2 and out.stride(-1) == 1 and 1 <= slots_max <= 256, "out / slots")
        rows_max = out.shape[-2]
        _check(ctr.numel() >= MB.coop_ctr_len(slots_max, rows_max, I, Ho), "counter scratch too small")
        if sh_w is not None:
            _check(sh_w.dtype == torch.half and sh_w.is_contiguous(), "shared gate")
        bsz, H = x.shape
        topk = sel.shape[1]
        _check(x.stride(1) == 1 and H % 4 == 0 and H <= Hi, "x")
        _check(bsz * topk <= slots_max and bsz <= rows_max, "batch exceeds the scratch")
        _check(min_e < 0 or max_e - min_e <= ut.numel(), "expert range")
        if sh_out is not None:
            _check(sh_out.dtype == torch.float and sh_out.is_contiguous() and sh_out.shape[-1] == H
                   and sh_out.numel() >= bsz * H and H_out == H, "shared output")
            if sh_w is not None:
                _check(sh_w.numel() == H, "shared gate width")
        xp = torch.zeros((bsz, Hi), dtype = torch.half)
        xp[:, :H] = x
        for b in range(bsz):
            acc = torch.zeros(H_out)
            for k in range(topk):
                e, w = int(sel[b, k]), float(rw[b, k])
                ok = w != 0.0 and (min_e < 0 or min_e <= e < max_e)
                if not ok:
                    continue
                l = e - min_e if min_e >= 0 else e
                acc += w * self._mlp(xp[b: b + 1], (gt, gs, gv), (ut, us, uv), (dt, ds, dv), l, H_out)[0]
            if sh_out is not None:
                gate = torch.sigmoid((x[b].float() * sh_w.float()).sum()) if sh_w is not None else 1.0
                acc += gate * sh_out[b, :H_out]
            out[b] = acc

    def exl3_moe(self, *args):
        self._count("exl3_moe", args)
        (hidden, out, expert_count, token_sorted, weight_sorted, tg, tu, tig, tiu, act, Kg, Ku, Kd,
         gt, gs, gv, ut, us, uv, dt, ds, dv, gmcg, gmul1, umcg, umul1, dmcg, dmul1, limit, num_active,
         scratch, fused_base, lo, hi, m_tile) = args
        _check(hidden.dtype == torch.half and hidden.dim() == 2 and out.dtype == torch.float and out.shape == hidden.shape, "hidden/out")
        _check(expert_count.dtype == torch.long and expert_count.dim() == 1, "expert_count")
        _check(token_sorted.dtype == torch.long and token_sorted.shape == weight_sorted.shape, "sorted")
        _check(tg.dtype == torch.half and tg.dim() == 3 and tg.shape[2] == hidden.shape[1] and tg.shape == tu.shape, "temp state")
        _check(tig.dim() == 3 and tig.shape == tiu.shape and tig.shape[1] == tg.shape[1], "temp intermediate")
        _check(gmcg == umcg == dmcg and gmul1 == umul1 == dmul1 and gmcg != gmul1, "codebook")
        E = expert_count.numel() - 1
        for tb in (gt, gs, gv, ut, us, uv, dt, ds, dv):
            _check(tb.numel() == E, "pointer table size")
        if scratch is not None:
            _check(fused_base is not None and scratch.dtype == torch.float and fused_base.dtype == torch.long
                   and scratch.is_contiguous() and scratch.dim() == 2 and scratch.shape[1] == hidden.shape[1], "scratch")
        H = hidden.shape[1]
        cap = tg.shape[1]
        start = 0
        for e in range(E):
            c = int(expert_count[e])
            end = start + c
            if 0 < c <= cap and lo <= c <= hi:
                for r in range(c):
                    tok = int(token_sorted[start + r])
                    y = float(weight_sorted[start + r]) * self._mlp(hidden[tok: tok + 1], (gt, gs, gv), (ut, us, uv),
                                                                    (dt, ds, dv), e, H)[0]
                    if scratch is not None:
                        scratch[int(fused_base[e]) + r] = y
                    else:
                        out[tok] += y
            start = end

    def exl3_moe_gather(self, *args):
        self._count("exl3_moe_gather", args)
        out, scratch, flat_expert, inv_order, expert_start, slot_base, slot_kind, weight_sorted = args
        for t_ in (flat_expert, inv_order, expert_start, slot_base, slot_kind, weight_sorted):
            _check(t_.is_contiguous(), "gather: index tensors must be contiguous")
        _check(out.is_contiguous() and out.dim() == 2 and scratch.shape[1] == out.shape[1], "gather: shapes")
        T = out.shape[0]
        topk = flat_expert.numel() // T
        E = slot_kind.numel()
        for t in range(T):
            acc = torch.zeros(out.shape[1])
            for k in range(topk):
                a = t * topk + k
                e = int(flat_expert[a])
                if 0 <= e < E and int(slot_kind[e]):
                    pos = int(inv_order[a])
                    slot = int(slot_base[e]) + (pos - int(expert_start[e]))
                    w = float(weight_sorted[pos]) if int(slot_kind[e]) == 2 else 1.0
                    acc += scratch[slot] * w
            out[t] += acc

    def had_r_128(self, *args):
        self._count("had_r_128", args)
        inp, outp, pre, post, scale = args
        v = inp.float()
        if pre is not None:
            v = v * pre.float()
        if post is not None:
            v = v * post.float()
        outp.copy_((v * scale).to(outp.dtype))

    def reconstruct(self, *args):
        self._count("reconstruct", args)
        W, trellis, K, mcg, mul1 = args
        _check(W.shape == (trellis.shape[0] * 16, trellis.shape[1] * 16) and trellis.shape[2] == 16 * K, "reconstruct shape")
        W.copy_(self.dense(trellis).half())

    def hgemm(self, *args):
        self._count("hgemm", args)
        a, b, c = args
        c.copy_((a.float() @ b.float()).to(c.dtype))

    def silu_mul(self, *args):
        self._count("silu_mul", args)
        x, y, z, limit = args
        z.copy_((F.silu(x.float()) * y.float()).to(z.dtype))


def test_the_fake_takes_the_declared_signatures():
    # The parser reads the real declarations, and the checker refuses what pybind would refuse
    assert len(cpp_params("exl3_moe_coop")) == 36
    assert len(cpp_params("exl3_moe")) == 35
    assert len(cpp_params("exl3_moe_gather")) == 8
    assert cpp_params("exl3_moe_max_concurrency") == [("int", "device")]
    assert [k for k, _ in cpp_params("had_r_128")] == ["tensor", "tensor", "optional", "optional", "float"]
    with pytest.raises(AssertionError, match = "arguments"):
        check_args("hgemm", (torch.zeros(1), torch.zeros(1)))
    with pytest.raises(AssertionError, match = "int"):
        check_args("exl3_moe_max_concurrency", (0.0,))


# --------------------------------------------------------------------------------------------
#   Selections
# --------------------------------------------------------------------------------------------

def test_gpu_counts_average_the_hit_rate():
    for rows in range(1, 9):
        c = MB.gpu_counts(rows, 10, 0.87)
        assert len(c) == rows and all(0 <= n <= 10 for n in c)
        assert sum(c) == math.floor(8.7 * rows + 1e-9)
    assert MB.gpu_counts(3, 10, 1.0) == [10, 10, 10] and MB.gpu_counts(2, 10, 0.0) == [0, 0]


@pytest.mark.parametrize("pattern", ["distinct", "same", "overlap:0.3", "overlap:1"])
@pytest.mark.parametrize("rows", [1, 2, 5, 8])
def test_selection_rows(pattern, rows):
    topk, e_gpu, e_total = 10, 96, 512
    sel = MB.plan_rows(rows, topk, e_gpu, e_total, 0.87, pattern, random.Random(3))
    assert len(sel) == rows
    for row in sel:
        assert len(row) == topk and len(set(row)) == topk and all(0 <= e < e_total for e in row)
    counts = [sum(1 for e in row if e < e_gpu) for row in sel]
    kind, frac = MB.parse_pattern(pattern)
    if kind == "same":
        assert all(row == sel[0] or sorted(r for r in row if r < e_gpu) == sorted(r for r in sel[0] if r < e_gpu) for row in sel)
        assert MB.distinct_gpu(sel, e_gpu) == counts[0]
    else:
        assert counts == MB.gpu_counts(rows, topk, 0.87)
    if kind == "distinct":
        assert MB.distinct_gpu(sel, e_gpu) == sum(counts) == MB.gpu_picks(sel, e_gpu)
    if kind == "overlap":
        for r in range(1, rows):
            a = {e for e in sel[r - 1] if e < e_gpu}
            b = {e for e in sel[r] if e < e_gpu}
            assert len(a & b) == min(round(frac * min(len(a), len(b))), len(b))
    # Deterministic per seed
    assert sel == MB.plan_rows(rows, topk, e_gpu, e_total, 0.87, pattern, random.Random(3))


def test_selection_refuses_a_table_too_small():
    with pytest.raises(ValueError, match = "tabella"):
        MB.plan_rows(8, 10, 40, 512, 1.0, "distinct", random.Random(0))
    with pytest.raises(ValueError, match = "sconosciuto"):
        MB.parse_pattern("overlap:2")


# --------------------------------------------------------------------------------------------
#   The fused tier's layout
# --------------------------------------------------------------------------------------------

def test_fused_layout_takes_every_pick_once_and_maps_back():
    E, topk = 12, 4
    rng = random.Random(5)
    sel = torch.tensor(MB.plan_rows(5, topk, E, 40, 0.75, "overlap:0.5", rng))
    rw = torch.rand(sel.shape).half()
    for on_device in (False, True):
        lay = MB.fused_layout(torch, sel, rw, E, 0, None, on_device)
        A = sel.numel()
        local = lay["local"]
        assert torch.equal(local, torch.where(sel.reshape(-1) < E, sel.reshape(-1), torch.full_like(local, E)))
        # sorted by expert, each assignment once, the inverse permutation exact
        assert torch.equal(lay["order"].sort().values, torch.arange(A))
        assert torch.equal(local[lay["order"]], local[lay["order"]].sort().values)
        assert torch.equal(lay["order"][lay["inv_order"]], torch.arange(A))
        assert torch.equal(lay["expert_count"], torch.bincount(local, minlength = E + 1))
        # each (row, pick) slot of a GPU expert: token and weight where the gather looks for them
        st, base, kind = lay["tables"]
        assert torch.equal(st, base) and torch.equal(kind, (lay["expert_count"] > 0).long())
        for a in range(A):
            row, k = divmod(a, topk)
            pos = int(lay["inv_order"][a])
            assert int(lay["token_sorted"][pos]) == row and lay["weight_sorted"][pos] == rw[row, k]
            e = int(local[a])
            assert int(st[e]) <= pos < int(st[e]) + int(lay["expert_count"][e])
        assert lay["n_slots"] == A


# --------------------------------------------------------------------------------------------
#   The launcher's geometry against the night profiling's nsys capture
# --------------------------------------------------------------------------------------------

def test_coop_geometry_gives_the_grids_of_the_production_capture():
    """
    The decode capture of 2026-10-24 (T2b/decode_ctx60-0.sqlite, both cards, 36 SMs, Blackwell):
    routed exl3_moe_coop_a_kernel grids 400 / 800 / 1200 (narrow, 1-3 rows) and 400 (wide, 4 rows: the
    other register count), exl3_moe_coop_b_kernel 800 / 1600 / 2400 / 3200; the shared expert's launch
    40 / 80 / 120 / 160 and 80 / 160 / 240 / 320. Production's shapes: Hi = Ho = 2560, I = 640, top-k 10
    """
    want_a = {1: (False, 400), 2: (False, 800), 3: (False, 1200), 4: (True, 400)}
    for R in (1, 2, 3, 4):
        g = MB.coop_geometry(2560, 640, 2560, R * 10, 80, True, 36, True)
        assert (g["a_wide"], g["a_grid"]) == want_a[R] and (g["b_wide"], g["b_grid"]) == (False, 800 * R)
        assert g["a_ksplit"] == g["b_ksplit"] == 1
        s = MB.coop_geometry(2560, 640, 2560, R, 8, True, 36, True)
        assert (s["a_grid"], s["b_grid"], s["a_wide"], s["b_wide"]) == (40 * R, 80 * R, False, False)
    # The knobs as the extension reads them
    w1 = MB.coop_geometry(2560, 640, 2560, 10, 80, True, 36, True, **MB.launcher_knobs({"EXL3_MOE_COOP_WIDE": "1"}))
    assert w1["a_wide"] and w1["b_wide"] and w1["a_grid"] == 100 and w1["b_grid"] == 200
    k2 = MB.coop_geometry(2560, 640, 2560, 10, 80, True, 36, True, **MB.launcher_knobs({"EXL3_MOE_COOP_KSPLIT": "2"}))
    assert k2["a_ksplit"] == k2["b_ksplit"] == 2 and k2["a_grid"] == 800
    assert MB.launcher_knobs({}) == {"wide_mode": None, "ksplit_forced": None}
    assert MB.launcher_knobs({"EXL3_MOE_COOP_WIDE": "x"})["wide_mode"] == 0      # atoi("x") = 0: narrow


# --------------------------------------------------------------------------------------------
#   Shapes from the headers
# --------------------------------------------------------------------------------------------

def _write_safetensors(path, tensors):
    header = {}
    off = 0
    for name, (dtype, shape) in tensors.items():
        n = 2 * math.prod(shape)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"\0" * off)


def test_model_shapes_from_the_headers(tmp_path):
    pre = "model.language_model.layers"
    t = {}
    for li, (kgu, kd) in ((0, (2, 3)), (1, (2, 2))):
        for e in range(2):
            t[f"{pre}.{li}.mlp.experts.{e}.gate_proj.trellis"] = ("I16", [160, 40, 16 * kgu])
            t[f"{pre}.{li}.mlp.experts.{e}.up_proj.trellis"] = ("I16", [160, 40, 16 * kgu])
            t[f"{pre}.{li}.mlp.experts.{e}.down_proj.trellis"] = ("I16", [40, 160, 16 * kd])
            t[f"{pre}.{li}.mlp.experts.{e}.up_proj.mul1"] = ("I32", [1])
        t[f"{pre}.{li}.mlp.shared_expert.gate_proj.trellis"] = ("I16", [160, 40, 64])
        t[f"{pre}.{li}.mlp.shared_expert.up_proj.trellis"] = ("I16", [160, 40, 64])
        t[f"{pre}.{li}.mlp.shared_expert.down_proj.trellis"] = ("I16", [40, 160, 80])
        t[f"{pre}.{li}.mlp.shared_expert.up_proj.mul1"] = ("I32", [1])
    _write_safetensors(tmp_path / "model-00001.safetensors", dict(list(t.items())[: len(t) // 2]))
    _write_safetensors(tmp_path / "model-00002.safetensors", dict(list(t.items())[len(t) // 2:]))
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 2560, "num_experts_per_tok": 10,
                                                                     "num_experts": 512}}))
    s = MB.model_shapes(str(tmp_path))
    assert (s["H"], s["Hi"], s["I"], s["Ho"], s["K_gu"], s["K_d"], s["cb"]) == (2560, 2560, 640, 2560, 2.0, 3.0, 2)
    assert (s["sh_I"], s["sh_K_gu"], s["sh_K_d"], s["sh_cb"], s["topk"], s["num_experts"]) == (640, 4.0, 5.0, 2, 10, 512)
    assert s["layer"] == 0 and s["k_combos"] == ["2/2", "2/3"] and s["moe_layers"] == 2
    assert MB.model_shapes(str(tmp_path), 1)["K_d"] == 2.0
    with pytest.raises(ValueError):
        MB.model_shapes(str(tmp_path), 7)


# --------------------------------------------------------------------------------------------
#   The child on the CPU over the fake extension
# --------------------------------------------------------------------------------------------

def small_cfg(**over):
    shapes = {"H": 128, "Hi": 128, "I": 128, "Ho": 128, "K_gu": 2.0, "K_d": 3.0, "cb": 2, "sh_I": 128, "sh_K_gu": 4.0,
              "sh_K_d": 4.0, "sh_cb": 2, "topk": 4, "num_experts": 64, "source": "test"}
    cfg = {"shapes": shapes, "device": "cpu", "settings": ["default", "wide1"], "rows": [1, 2, 8],
           "patterns": ["distinct", "same"], "probe_rows": [1, 2, 8], "probe_patterns": ["distinct", "same"],
           "hit": 0.75, "experts": 32, "layers": 2, "sets": 2, "iters": 3, "repeats": 2, "seed": 7, "graph": True,
           "fused": True}
    cfg.update(over)
    return cfg


def test_every_path_computes_the_reference():
    ext = FakeExt()
    cfg = small_cfg()
    b = MB.Bench(torch, ext, cfg, "cpu", ext.track)
    assert b.fused_ok
    for pattern in ("distinct", "same", "overlap:0.5"):
        for R in (1, 3, 8):
            inp = b.inputs(pattern, R, 0, cfg["hit"])
            ref = b.reference(0, inp)
            routed = b.coop_routed(0, inp, False).clone()
            layer = b.coop_layer(0, inp).clone()
            fused = b.fused_routed(0, inp, False).clone()
            fused_dev = b.fused_routed(0, inp, True).clone()
            for test, want in ((routed, ref["routed"]), (fused, ref["routed"]), (fused_dev, ref["routed"]), (layer, ref["layer"])):
                errs = MB.row_errors(test, want)
                assert max(errs) <= 1e-2, (pattern, R, errs)
            # The shared expert goes into the layer only: the two differ by its gated output
            assert not torch.allclose(layer, routed)
            # CPU picks never computed: zeroing their weights changes nothing
            rw = inp["rw"].clone()
            rw[inp["sel"] >= b.E] = 0
            routed2 = b.coop_routed(0, dict(inp, rw = rw), False)
            assert torch.allclose(routed2, routed)
    assert ext.calls["exl3_moe_coop"] and ext.calls["exl3_moe"] and ext.calls["exl3_moe_gather"]


def test_the_comparison_catches_a_wrong_row():
    ext = FakeExt()
    b = MB.Bench(torch, ext, small_cfg(), "cpu", ext.track)
    inp = b.inputs("distinct", 4, 0, 0.75)
    ref = b.reference(0, inp)["routed"]
    out = b.coop_routed(0, inp, False).clone()
    assert max(MB.row_errors(out, ref)) <= 1e-2
    swapped = out[[1, 0, 2, 3]]
    errs = MB.row_errors(swapped, ref)
    assert errs[0] > MB.TOL_ROW and errs[1] > MB.TOL_ROW and errs[2] <= 1e-2
    probes = {"default": {"cases": {"distinct/4": {"plan": inp["plan"], "coop_routed": swapped, "ref_routed": ref}}}}
    rows = MB.compare_probes(probes)
    assert rows and not rows[0]["ok"]


def test_time_calls_on_the_cpu():
    n = {"c": 0}

    def f():
        n["c"] += 1
    r = MB.time_calls(torch, "cpu", [f, f], 5, 3, True)
    assert r["mode"] == "plain" and r["gpu_us"] >= 0 and r["host_us"] >= 0 and "graph_us" not in r
    assert n["c"] == 2 + 5 * 3


def test_run_child_records_and_probes(tmp_path):
    ext = FakeExt()
    cfg = small_cfg()
    assert MB.run_child(cfg, "default", str(tmp_path), torch, ext, ext.track) == 0
    recs = MB.read_jsonl(str(tmp_path / "records.jsonl"))
    kinds = [r["kind"] for r in recs]
    assert kinds[0] == "proc" and kinds[-1] == "done" and recs[-1]["status"] == 0
    times = [r for r in recs if r["kind"] == "time"]
    paths = {r["path"] for r in times}
    assert paths == {"coop_routed", "coop_layer", "coop_shared", "fused_routed"}
    for r in times:
        assert "error" not in r and r["gpu_us"] >= 0
        if r["path"] == "coop_routed":
            assert r["geometry"]["a_grid"] > 0
    same8 = next(r for r in times if r["path"] == "coop_routed" and r["pattern"] == "same" and r["rows"] == 8)
    dist8 = next(r for r in times if r["path"] == "coop_routed" and r["pattern"] == "distinct" and r["rows"] == 8)
    assert same8["distinct"] == 3 and dist8["distinct"] == 24          # hit 0.75 of top-4
    # The shared launch once per row count, whatever the selection
    assert sorted(r["rows"] for r in times if r["path"] == "coop_shared") == [1, 2, 8]
    probes = torch.load(tmp_path / "probes-default.pt", weights_only = False)
    case = probes["cases"]["distinct/8"]
    assert {"coop_routed", "coop_layer", "fused_routed", "ref_routed", "ref_layer"} <= set(case)
    acc = MB.compare_probes({"default": probes})
    assert acc and all(a["ok"] or a["worst"] <= 1e-2 for a in acc)


def test_a_wrong_argument_fails_on_the_cpu(monkeypatch):
    """The point of the fake: a call the C++ signature would refuse fails here"""
    ext = FakeExt()
    b = MB.Bench(torch, ext, small_cfg(), "cpu", ext.track)
    inp = b.inputs("distinct", 2, 0, 0.75)
    real = ext.exl3_moe_coop
    monkeypatch.setattr(ext, "exl3_moe_coop", lambda *a: real(*a[:-1]))
    with pytest.raises(AssertionError, match = "arguments"):
        b.coop_routed(0, inp, True)


# --------------------------------------------------------------------------------------------
#   Orchestrator, report
# --------------------------------------------------------------------------------------------

FAKE_CHILD = textwrap.dedent("""
    import json, os, sys
    sys.path.insert(0, {exp!r}); sys.path.insert(0, {tests!r})
    import torch
    import moe_decode_bench as MB
    from test_moe_decode_bench import FakeExt
    cfg_path = sys.argv[sys.argv.index("--child") + 1]
    setting = sys.argv[sys.argv.index("--setting") + 1]
    cfg = json.load(open(cfg_path))
    ext = FakeExt()
    sys.exit(MB.run_child(cfg, setting, os.path.dirname(cfg_path), torch, ext, ext.track))
""")


def test_orchestrator_one_process_per_setting(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    cfg = small_cfg(settings = ["default", "wide1", "ksplit2"], rows = [1, 2], patterns = ["distinct", "same"],
                    probe_rows = [1, 2])
    (out / "config.json").write_text(json.dumps(cfg))
    script = tmp_path / "child.py"
    script.write_text(FAKE_CHILD.format(exp = EXP, tests = os.path.dirname(os.path.abspath(__file__))))
    monkeypatch.setenv("EXL3_MOE_COOP_WIDE", "0")           # the parent's value must not reach a child
    monkeypatch.setenv("EXL3_MOE_COOP_KSPLIT", "4")
    rc = MB.run_all(cfg, str(out), sys.executable, child_cmd = [sys.executable, str(script)])
    assert rc == 0
    recs = MB.read_jsonl(str(out / "records.jsonl"))
    envs = {r["setting"]: r["env"] for r in recs if r["kind"] == "proc"}
    assert envs == {"default": {"EXL3_MOE_COOP_WIDE": None, "EXL3_MOE_COOP_KSPLIT": None},
                    "wide1": {"EXL3_MOE_COOP_WIDE": "1", "EXL3_MOE_COOP_KSPLIT": None},
                    "ksplit2": {"EXL3_MOE_COOP_WIDE": None, "EXL3_MOE_COOP_KSPLIT": "2"}}
    # The geometry each process recorded follows its own knob
    g = {r["setting"]: r["geometry"] for r in recs if r["kind"] == "time" and r["path"] == "coop_routed"
         and r["rows"] == 1 and r["pattern"] == "distinct"}
    assert g["wide1"]["a_wide"] and not g["default"]["a_wide"] and g["ksplit2"]["a_ksplit"] == 2
    # The fused tier in the default process only
    assert {r["setting"] for r in recs if r["kind"] == "time" and r["path"] == "fused_routed"} == {"default"}
    report = (out / "REPORT.md").read_text()
    for head in ("## 1. Un esperto scelto da più righe si legge una volta", "## 2.", "## 3.", "## 4.", "## 5. Stessa computazione"):
        assert head in report
    assert "**no**" not in report and "non finito" not in report
    assert "`wide1`" in report and "coop_routed wide1 / default" in report
    # --report rebuilds it from the records
    os.remove(out / "REPORT.md")
    assert MB.main(["--report", str(out)]) == 0 and (out / "REPORT.md").exists()


def test_build_config_and_dry_run():
    args = MB.parse_cli(["--out", "/nonexistent", "--settings", "wide1", "--rows", "1,8"])
    cfg = MB.build_config(args)
    assert cfg["settings"] == ["default", "wide1"] and cfg["rows"] == [1, 8]
    text = MB.dry_run_text(cfg)
    assert "VRAM" in text and "durata" in text
    assert 0.5 < MB.vram_estimate(cfg) / 2 ** 30 < 1.0               # 6 x 96 production experts, ~0.73 GB
    with pytest.raises(SystemExit):
        MB.build_config(MB.parse_cli(["--experts", "40"]))
    with pytest.raises(SystemExit):
        MB.build_config(MB.parse_cli(["--settings", "nope"]))
    with pytest.raises(SystemExit):
        MB.build_config(MB.parse_cli(["--rows", "9"]))
