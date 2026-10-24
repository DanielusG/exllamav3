"""
The QSA indexer's raw keys kept only for the blocks being built (Cache(qsa_raw_tail), EXL3_QSA_RAW_TAIL):
the raw-key ring of cache/qsa.py and its kernels in modules/attention_fn/qsa_triton.py.

No GPU is required. The kernels run in Triton's interpreter (TRITON_INTERPRET, numpy on the CPU; on a machine
with CUDA they run on the card instead): the full-plane kernels the ring replaces (_mla_plane_update_kernel,
_qsa_pool_update_kernel) and the ring's own, over the same random sequences of appends, and the pooled planes
must come out equal bit for bit after every append: prompt chunks of any length that start and end inside
blocks and pages, decode steps of up to 16 rows (the graph slots' longest) with their drafts rejected or
accepted, the MTP head's cache pattern (drafts one at a time, then the accepted ones written again), replays
from a page boundary, pages taken over by another sequence and complete pages shared by prefix, several
sequences per call. The graph slot's form is driven the way the slot launches it (the same arguments: raw
append, then pool update); the eager form through QSAIndexer.update_planes itself. The same comparison on the
library's torch references (update_planes_ref) runs without Triton. Two checks show that the comparison is not
blind: the eager kernels in the other order, and a speculative pass longer than the ring serves, both give
other pooled keys.

Then, in a process of their own (the interpreter is chosen when triton is imported), the kernels are compiled
for the cards' architecture (sm_120, with the ptxas of the triton wheel) from BCAttn._configure_qsa itself,
with the slot's own signatures, and the floating-point instructions of the ring's pool updates compared with
those of the full plane's. And the cache layer (shapes, page images, copy_page, tp_export), the Cache switch and
its refusals, the generator's check of the draft length, and the disk tier's fingerprint.

The source files under test are loaded on their own, with the rest of the package stubbed. Without triton
installed the kernel tests skip; so do the interpreted ones in a session that imported triton before this
file without the interpreter, on a machine without CUDA.
"""

import ast
import contextlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import types
from collections import Counter

import pytest
import torch

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "exllamav3")

# The process the compile checks run in (see _compile_report): triton without the interpreter
COMPILE_ENV = "EXL3_TEST_QSA_COMPILE"
COMPILING = os.environ.get(COMPILE_ENV) == "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Triton picks the interpreter for its own library functions when it is imported (and checks the switch
# again when it imports more of itself later), so the switch has to be on before the first import, and
# stays on for the session
INTERPRET = DEVICE.type == "cpu" and not COMPILING and "triton" not in sys.modules
if INTERPRET:
    os.environ["TRITON_INTERPRET"] = "1"
try:
    import triton
    HAVE_TRITON = True
except ImportError:
    triton = None
    HAVE_TRITON = False

needs_kernels = pytest.mark.skipif(
    not HAVE_TRITON or not (INTERPRET or DEVICE.type == "cuda"),
    reason = "triton not installed, or imported without the interpreter on a machine without CUDA",
)
needs_compiler = pytest.mark.skipif(
    not HAVE_TRITON or importlib.util.find_spec("triton.backends.nvidia") is None,
    reason = "triton or its NVIDIA backend not installed",
)


# --------------------------------------------------------------------------------------------
#   Loading
# --------------------------------------------------------------------------------------------

class _Any:
    def __init__(self, *a, **k):
        pass


class _TensorCache:
    """util.tensor.g_tensor_cache: fresh buffers every time"""

    def get_scratch(self, device, numel, dtype, tag, rows, static_rows):
        return torch.empty((numel,), dtype = dtype, device = DEVICE)

    def get(self, device, shape, dtype, tag):
        return torch.empty(shape, dtype = dtype, device = DEVICE)

    def get_bucketed(self, device, numel, dtype, tag):
        return torch.empty((numel,), dtype = dtype, device = DEVICE)


def _get_for_device(d, key, device, default = None):
    return d.get(key, default)


def _load(pkg: str):
    """The cache layers, the QSA indexer, its kernels and the decode graph's configuration, as package
    `pkg`"""

    def stub(name, **attrs):
        full = f"{pkg}.{name}" if name else pkg
        m = types.ModuleType(full)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[full] = m
        return m

    def real(name, fix = None):
        full = f"{pkg}.{name}"
        path = os.path.join(SRC, *name.split(".")) + ".py"
        text = open(path).read()
        if fix:
            text = fix(text)
        m = types.ModuleType(full)
        m.__file__ = path
        m.__package__ = full.rpartition(".")[0]
        sys.modules[full] = m
        exec(compile(text, path, "exec"), m.__dict__)
        return m

    # fp16.py and quant.py import the extension by its absolute name
    rel_ext = lambda text: text.replace("from exllamav3.ext import exllamav3_ext as ext", "from ..ext import exllamav3_ext as ext")

    stub("")
    for p in ("cache", "modules", "modules.attention_fn", "util", "generator"):
        stub(p)
    stub("ext", exllamav3_ext = types.SimpleNamespace())
    stub("model", Config = _Any)
    stub("model.config", Config = _Any)
    stub("util.tensor", g_tensor_cache = _TensorCache(), get_for_device = _get_for_device)
    stub("modules.module", Module = type("Module", (), {"__init__": lambda self, *a, **k: None}))
    stub("modules.linear", Linear = _Any)
    stub("modules.rmsnorm", RMSNorm = _Any)
    stub("modules.attention_fn.triton_paged", combine_subtiles = None, _qc_load_kt = None, _qc_load_v = None,
         _rot_h32 = None, _get_h32 = None, _paged_attn_decode_split_kernel = None,
         _paged_attn_decode_combine_kernel = None, _paged_kv_update_kernel = None,
         decode_row_layout = None, _normalize_window = None)
    stub("modules.attention_fn.dsa_triton", _dsa_indexer_fewq_kernel = None, _dsa_pool_expand_kernel = None)
    real("constants")
    real("version")
    real("util.backend")
    real("modules.attention_fn.smem")
    real("cache.cache")
    real("cache.fp16", rel_ext)
    real("cache.quant", rel_ext)
    real("cache.qsa")
    mods = types.SimpleNamespace(
        cache = sys.modules[f"{pkg}.cache.cache"],
        qsa = sys.modules[f"{pkg}.cache.qsa"],
        indexer = real("modules.qsa_indexer"),
    )
    if HAVE_TRITON:
        mods.kern = real("modules.attention_fn.qsa_triton")
        mods.mla = real("modules.attention_fn.mla_triton")
        mods.bc = real("modules.attention_fn.bc_attn")
    return mods


M = _load("exl3_rawtail_stub")
QSA = M.qsa
PAGE = sys.modules["exl3_rawtail_stub.constants"].PAGE_SIZE
P = 4           # compress ratio (Qwen3.8-Flash-Next)
D = 128         # indexer head dim (Qwen3.8-Flash-Next)
ROPE_R = 64     # the main attention's rotary width: the pooled keys' rope covers half of D
RING = QSA.RAW_TAIL_DEFAULT


# --------------------------------------------------------------------------------------------
#   Layers
# --------------------------------------------------------------------------------------------

def attention(head_dim = D, cr = P):
    """What a QSA cache layer reads of its attention module"""
    return types.SimpleNamespace(
        qsa_indexer = types.SimpleNamespace(head_dim = head_dim, compress_ratio = cr),
        num_kv_heads = 1, head_dim = 8,
    )


def make_layer(ring: int, num_pages: int, quant = False):
    if quant:
        layer = QSA.CacheLayer_qsa_quant(None, attention(), 1, num_pages * PAGE, 8, 8, raw_tail = ring)
    else:
        layer = QSA.CacheLayer_qsa(None, attention(), 1, num_pages * PAGE, raw_tail = ring)
    layer.alloc(DEVICE)
    return layer


# --------------------------------------------------------------------------------------------
#   The cache layer
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("quant", [False, True])
def test_the_ring_replaces_the_raw_plane_and_leaves_the_page_images(quant):
    full = make_layer(0, 6, quant)
    ring = make_layer(RING, 6, quant)
    assert full.raw_k.shape == (6, PAGE, D) and ring.raw_k.shape == (6, RING, D)
    assert ring.pooled.shape == full.pooled.shape == (6, PAGE // P, D)
    # page images (CPU and disk tiers), defrag rotation: K/V and the pooled keys, not the ring
    ft, rt = full.get_tensors(), ring.get_tensors()
    assert any(t is full.raw_k for t in ft)
    assert not any(t is ring.raw_k for t in rt)
    assert len(rt) == len(ft) - 1 and rt[-1] is ring.pooled
    saved = 6 * (PAGE - RING) * D * 2
    assert full.storage_size() - ring.storage_size() == saved
    assert ring.raw_tail_saved_bytes() == saved and full.raw_tail_saved_bytes() == 0
    # 56 KiB per page and layer with the default ring
    assert ring.raw_tail_saved_bytes() // 6 == 56 * 1024


@pytest.mark.parametrize("quant", [False, True])
def test_tp_export_carries_the_ring(quant):
    layer = make_layer(RING, 2, quant)
    exported = layer.tp_export(None)
    assert exported["args"]["raw_tail"] == RING
    rebuilt = exported["cls"](None, attention(), **exported["args"])
    assert rebuilt.raw_tail == RING and rebuilt.raw_k_shape == layer.raw_k_shape
    assert make_layer(0, 2, quant).tp_export(None)["args"]["raw_tail"] == 0


def test_copy_page_copies_the_ring_row():
    a, b = make_layer(RING, 4), make_layer(RING, 4)
    a.raw_k.copy_(torch.randn(a.raw_k.shape).half())
    a.pooled.copy_(torch.randn(a.pooled.shape).half())
    b.copy_page(a, 1, 3, 77)
    assert torch.equal(b.raw_k[3], a.raw_k[1])
    assert torch.equal(b.pooled[3, :20], a.pooled[1, :20]) and not b.pooled[3, 20:].any()
    # without the ring, as upstream: the first num_tokens raw keys
    a, b = make_layer(0, 4), make_layer(0, 4)
    a.raw_k.copy_(torch.randn(a.raw_k.shape).half())
    b.copy_page(a, 1, 3, 77)
    assert torch.equal(b.raw_k[3, :77], a.raw_k[1, :77]) and not b.raw_k[3, 77:].any()


def test_a_ring_too_short_for_the_graph_appends_is_refused():
    # 16 rows of a graph slot plus the 3 tokens of the block they start in
    with pytest.raises(ValueError, match = "decode graph"):
        make_layer(18, 2)
    assert make_layer(19, 2).raw_tail == 19
    assert QSA.raw_tail_max_rows(RING, P) == 29


def test_the_switch_values(monkeypatch):
    monkeypatch.delenv(QSA.RAW_TAIL_ENV, raising = False)
    assert QSA.resolve_raw_tail(None) == 0
    assert QSA.resolve_raw_tail(False) == 0 and QSA.resolve_raw_tail(0) == 0
    assert QSA.resolve_raw_tail(True) == RING and QSA.resolve_raw_tail(1) == RING
    assert QSA.resolve_raw_tail(64) == 64
    for env, ring in (("1", RING), ("0", 0), ("", 0), ("48", 48), ("yes", RING)):
        monkeypatch.setenv(QSA.RAW_TAIL_ENV, env)
        assert QSA.resolve_raw_tail(None) == ring, env
    # an argument wins over the environment
    monkeypatch.setenv(QSA.RAW_TAIL_ENV, "1")
    assert QSA.resolve_raw_tail(False) == 0
    with pytest.raises(ValueError):
        QSA.resolve_raw_tail(PAGE + 1)


# --------------------------------------------------------------------------------------------
#   The Cache switch
# --------------------------------------------------------------------------------------------

class FakeAttn:
    def __init__(self, layer_idx, qsa = True):
        self.layer_idx = layer_idx
        self.qsa_indexer = types.SimpleNamespace(head_dim = D, compress_ratio = P) if qsa else None
        self.num_kv_heads = 1
        self.head_dim = 8
        self.cache_layers = []

    def cache_layer_type(self, default, kwargs):
        # Attention.cache_layer_type
        if self.qsa_indexer is None:
            return default, kwargs
        if issubclass(default, M.qsa.CacheLayer_quant):
            return QSA.CacheLayer_qsa_quant, kwargs
        return QSA.CacheLayer_qsa, kwargs


class FakeModel:
    def __init__(self, recurrent = True, atomic_mm = False, qsa = True, mtp_head = False):
        self.config = None
        self.caps = {"recurrent_states": recurrent}
        if atomic_mm:
            self.caps["atomic_mm_prefill"] = True
        if mtp_head:
            # Qwen4ExpMTPModel: no recurrent layers of its own, it drafts for a target that has them
            self.caps["mtp_draft"] = True
        self.attn = [FakeAttn(3, qsa), FakeAttn(7, qsa), FakeAttn(9, qsa = False)]
        self.cache_weakrefs = {}
        self.recurrent_state_cls = None

    def get_cache_layers(self):
        return self.attn

    def get_layer_instances(self, idx):
        return [(idx, 0)]

    def get_recurrent_layers(self):
        return []


def make_cache(model, **kwargs):
    return M.cache.Cache(model, 4 * PAGE, layer_type = M.qsa.CacheLayer_quant, k_bits = 8, v_bits = 8, **kwargs)


def test_the_cache_hands_the_ring_to_its_qsa_layers_only(monkeypatch, capsys):
    monkeypatch.delenv(QSA.RAW_TAIL_ENV, raising = False)
    cache = make_cache(FakeModel(), qsa_raw_tail = True)
    assert cache.qsa_raw_tail == RING
    rings = {k: getattr(l, "raw_tail", None) for k, l in cache.layers.items()}
    assert rings == {(3, 0): RING, (7, 0): RING, (9, 0): None}
    assert type(cache.layers[3, 0]) is QSA.CacheLayer_qsa_quant
    assert type(cache.layers[9, 0]) is M.qsa.CacheLayer_quant
    out = capsys.readouterr().out
    assert "QSA raw keys: a ring of 32 tokens per page in 2 layers" in out
    # the environment, and off by default
    monkeypatch.setenv(QSA.RAW_TAIL_ENV, "1")
    assert make_cache(FakeModel()).qsa_raw_tail == RING
    monkeypatch.delenv(QSA.RAW_TAIL_ENV)
    capsys.readouterr()
    cache = make_cache(FakeModel())
    assert cache.qsa_raw_tail == 0
    assert all(getattr(l, "raw_tail", 0) == 0 for l in cache.layers.values())
    assert "QSA raw keys" not in capsys.readouterr().out


@pytest.mark.parametrize("model,why", [
    (FakeModel(recurrent = False), "no recurrent states"),
    (FakeModel(atomic_mm = True), "past a chunk's end"),
])
def test_the_cache_refuses_the_ring_for_models_it_cannot_serve(model, why, capsys):
    cache = make_cache(model, qsa_raw_tail = True)
    assert cache.qsa_raw_tail == 0
    assert all(getattr(l, "raw_tail", 0) == 0 for l in cache.layers.values())
    out = capsys.readouterr().out
    assert "refused" in out and why in out


def test_a_model_without_qsa_layers_reports_no_ring(capsys):
    assert make_cache(FakeModel(qsa = False), qsa_raw_tail = True).qsa_raw_tail == 0
    # an external draft model without QSA layers (or recurrent states) takes the argument quietly
    assert make_cache(FakeModel(qsa = False, recurrent = False), qsa_raw_tail = True).qsa_raw_tail == 0
    assert capsys.readouterr().out == ""


def test_the_mtp_heads_cache_takes_the_ring_and_its_generator_decides():
    """The MTP head has no recurrent states of its own (Qwen4ExpMTPModel's caps): its cache follows the
    target's generator, which refuses the ring for a target without them"""
    head = make_cache(FakeModel(recurrent = False, mtp_head = True), qsa_raw_tail = True)
    assert head.qsa_raw_tail == RING
    main = make_cache(FakeModel(), qsa_raw_tail = True)
    QSA.check_raw_tail([main, head], 2, FakeModel())
    with pytest.raises(ValueError, match = "no recurrent states"):
        QSA.check_raw_tail([make_cache(FakeModel(recurrent = False)), head], 2, FakeModel(recurrent = False))
    with pytest.raises(ValueError, match = "past a chunk's end"):
        QSA.check_raw_tail([main, head], 2, FakeModel(atomic_mm = True))
    # without a ring anywhere, any model
    QSA.check_raw_tail([make_cache(FakeModel(recurrent = False))], 2, FakeModel(recurrent = False))


def test_the_generator_refuses_drafts_longer_than_the_ring_serves():
    cache = make_cache(FakeModel(), qsa_raw_tail = True)
    QSA.check_raw_tail([cache, None], 29, FakeModel())      # 28 draft tokens plus the one they follow
    with pytest.raises(ValueError, match = "at most 28 draft tokens"):
        QSA.check_raw_tail([cache, None], 30, FakeModel())
    QSA.check_raw_tail([make_cache(FakeModel()), None], 1000, FakeModel())
    QSA.check_raw_tail([make_cache(FakeModel(), qsa_raw_tail = 64)], 61, FakeModel())


def test_the_generator_checks_its_longest_pass():
    """Generator.__init__ hands check_raw_tail its caches and its widest verification pass, the
    draft window plus one, once max_draft_tokens is known"""
    tree = ast.parse(open(os.path.join(SRC, "generator", "generator.py")).read())
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                and any(a.arg == "draft_cache" for a in n.args.args))
    calls = [n for n in ast.walk(init) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "check_raw_tail"]
    assert len(calls) == 1
    caches, rows, model = calls[0].args
    assert [e.id for e in caches.elts] == ["cache", "draft_cache"]
    assert ast.unparse(rows) == "self.max_draft_tokens + 1" and model.id == "model"
    assigned = [n.lineno for n in ast.walk(init) if isinstance(n, ast.Assign)
                and any(ast.unparse(t) == "self.max_draft_tokens" for t in n.targets)]
    assert assigned and assigned[0] < calls[0].lineno


# --------------------------------------------------------------------------------------------
#   The disk tier's fingerprint
# --------------------------------------------------------------------------------------------

def test_the_disk_tier_opens_another_directory_with_the_ring(tmp_path):
    """The page image changes (no raw keys in it), and so do the segments the fingerprint covers"""
    import test_tail_checkpoint as TC

    def tier_for(layers):
        # CPUPageCache's segment table: one entry per paged tensor of every layer
        segs, offset = [], 0
        for layer in layers:
            for t in layer.get_tensors():
                nbytes = t[0].numel() * t.element_size()
                segs.append((t, offset, tuple(t.shape[1:]), t.dtype))
                offset = (offset + nbytes + 255) // 256 * 256
        tier = types.SimpleNamespace(segments = segs, slab_size = offset, tp_groups = [], max_slots = 8,
                                     entries = {})
        tier.__contains__ = lambda h: False
        return tier

    class Tier(types.SimpleNamespace):
        def __contains__(self, h):
            return False

        def __len__(self):
            return 0

    dirs = {}
    for ring in (0, RING):
        layers = [make_layer(ring, 2, quant = True), make_layer(ring, 2, quant = True)]
        t = tier_for(layers)
        tier = Tier(**vars(t))
        cache = types.SimpleNamespace(layers = {(3, 0): layers[0], (7, 0): layers[1]}, model = None)
        disk = TC.DISK.DiskPageCache(str(tmp_path), [cache], tier, 1 << 30, min_state_distance = 0, min_free = 0)
        dirs[ring] = (disk.dir, disk.image_size)
        disk.close()
    assert dirs[0][0] != dirs[RING][0]
    assert dirs[0][1] - dirs[RING][1] == 2 * PAGE * D * 2


# --------------------------------------------------------------------------------------------
#   Appends: the full plane and the ring side by side
# --------------------------------------------------------------------------------------------

class Rope:
    """What the plane kernels read of the rope: inv_freq for ROPE_R // 2 frequencies, attn_factor"""

    def __init__(self, rope_r = ROPE_R):
        self.inv_freq = (1.0 / (10000.0 ** (torch.arange(0, rope_r, 2, dtype = torch.float) / rope_r))).to(DEVICE)
        self.attn_factor = 1.0

    def apply(self, *a, **k):
        pass


class Planes:
    """A full-plane layer and a ring layer fed the same appends. graph(): the slot's launches over
    both (the slot launches the ring's kernels with the full-plane kernels' arguments); eager(): the
    full plane's update_planes kernels against the ring's (pool update from the rows, then the
    append). Every call compares the pooled planes."""

    def __init__(self, num_pages, ring = RING, rope_r = ROPE_R, eager_order = "pool_first"):
        self.full = make_layer(0, num_pages)
        self.ring = make_layer(ring, num_pages)
        self.rope = Rope(rope_r)
        self.norm_w = (torch.randn(D) * 0.1).half().to(DEVICE)
        self.eps = 1e-6
        self.eager_order = eager_order
        self.calls = 0

    def _pool(self, kernel, layer, bt, seqlens, L, *extra, **cx):
        kernel[(bt.shape[0], L // P + 1)](
            layer.raw_k.view(-1, D), layer.pooled.view(-1, D), self.norm_w, self.rope.inv_freq,
            bt, seqlens, bt.shape[1], L, *extra,
            page_size = PAGE, P = P, D = D, ROPE_R = 2 * self.rope.inv_freq.numel(),
            attn_factor = float(self.rope.attn_factor), eps = self.eps, MAXPOOLS = 1, **cx,
        )

    def _full(self, rows, bt, seqlens, L):
        M.mla._mla_plane_update_kernel[(rows.shape[0] * L,)](
            rows, self.full.raw_k, bt, seqlens, bt.shape[1], L,
            page_size = PAGE, D = D, DST_D = 0, DST_OFF = 0,
        )
        self._pool(M.kern._qsa_pool_update_kernel, self.full, bt, seqlens, L)

    def _ring_append(self, rows, bt, seqlens, L):
        M.kern._qsa_raw_tail_append_kernel[(rows.shape[0] * L,)](
            rows, self.ring.raw_k, bt, seqlens, bt.shape[1], L,
            page_size = PAGE, D = D, RING = self.ring.raw_tail,
        )

    def graph(self, rows, bt, seqlens):
        L = rows.shape[1]
        assert L <= 16
        self._full(rows, bt, seqlens, L)
        self._ring_append(rows, bt, seqlens, L)
        self._pool(M.kern._qsa_pool_update_ring_kernel, self.ring, bt, seqlens, L, RING = self.ring.raw_tail)
        return self.check()

    def eager(self, rows, bt, seqlens):
        L = rows.shape[1]
        self._full(rows, bt, seqlens, L)
        if self.eager_order == "pool_first":
            self._pool(M.kern._qsa_pool_update_rows_kernel, self.ring, bt, seqlens, L, rows, RING = self.ring.raw_tail)
            self._ring_append(rows, bt, seqlens, L)
        else:
            self._ring_append(rows, bt, seqlens, L)
            self._pool(M.kern._qsa_pool_update_rows_kernel, self.ring, bt, seqlens, L, rows, RING = self.ring.raw_tail)
        return self.check()

    def check(self):
        self.calls += 1
        return torch.equal(self.full.pooled, self.ring.pooled)


class Seq:
    """A sequence's positions and pages, and the raw key of every position of its current text:
    replayed positions get the keys they had, positions past the accepted ones new keys"""

    def __init__(self, pages, gen, prefix_keys = None):
        self.pages = list(pages)
        self.pos = 0
        self.keys = torch.empty((len(self.pages) * PAGE, D), dtype = torch.half)
        self.valid = 0
        self.gen = gen
        if prefix_keys is not None:
            n = prefix_keys.shape[0]
            self.keys[:n] = prefix_keys
            self.pos = self.valid = n

    def rows(self, L, replay = False):
        s = self.pos
        if not replay:
            self.keys[s : s + L] = (torch.randn((L, D), generator = self.gen) * 2).half()
            self.valid = s + L
        return self.keys[s : s + L]


def batch(planes, seqs, L, form, max_pages, replay = False):
    rows = torch.stack([s.rows(L, replay) for s in seqs]).to(DEVICE).contiguous()
    bt = torch.zeros((len(seqs), max_pages), dtype = torch.int32)
    for i, s in enumerate(seqs):
        bt[i, :len(s.pages)] = torch.tensor(s.pages, dtype = torch.int32)
    bt = bt.to(DEVICE)
    seqlens = torch.tensor([s.pos for s in seqs], dtype = torch.int32, device = DEVICE)
    ok = (planes.graph if form == "graph" else planes.eager)(rows, bt, seqlens)
    return ok


def run_conversation(planes, seq, rnd, max_pages, prompt, steps, max_rows = 16, mtp_head = False):
    """A prompt in eager chunks, then decode steps through the graph slot (verification passes of up to
    max_rows rows, some of their drafts rejected) or, for the MTP head's cache, drafts written one at a
    time and the accepted ones written again. Returns False at the first difference."""
    while seq.pos < prompt:
        L = min(rnd.choice([1, 3, 7, 33, 100, 255, 256, 257, 600]), prompt - seq.pos)
        if not batch(planes, [seq], L, "eager", max_pages):
            return False
        seq.pos += L
    for _ in range(steps):
        if seq.pos + max_rows + 1 > len(seq.pages) * PAGE:
            break
        if mtp_head:
            # drafts at K..K+W-1, one row each, then K+1..K+A-1 again from the target's states
            W = rnd.randint(1, max_rows - 1)
            k0 = seq.pos
            for i in range(W):
                if not batch(planes, [seq], 1, "graph", max_pages):
                    return False
                seq.pos += 1
            A = rnd.randint(1, W + 1)
            seq.pos = k0 + 1
            if A > 1:
                if not batch(planes, [seq], A - 1, "graph", max_pages):
                    return False
            seq.pos = k0 + A
        else:
            L = rnd.randint(1, max_rows)
            if not batch(planes, [seq], L, "graph", max_pages):
                return False
            seq.pos += 1 + rnd.randint(0, L - 1)      # the accepted drafts
    return True


@needs_kernels
@pytest.mark.parametrize("seed", range(3))
def test_pooled_keys_are_the_full_planes_through_a_conversation(seed):
    rnd = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    pages = list(range(8))
    rnd.shuffle(pages)
    planes = Planes(8)
    seq = Seq(pages, gen)
    assert run_conversation(planes, seq, rnd, 8, prompt = rnd.randint(200, 700), steps = 40)
    assert planes.calls > 20


@needs_kernels
def test_pooled_keys_are_the_full_planes_in_the_mtp_heads_cache():
    rnd = random.Random(100)
    gen = torch.Generator().manual_seed(100)
    planes = Planes(6)
    seq = Seq(rnd.sample(range(6), 6), gen)
    assert run_conversation(planes, seq, rnd, 6, prompt = rnd.randint(100, 400), steps = 25, mtp_head = True)


@needs_kernels
def test_pooled_keys_are_the_full_planes_without_a_pass_through_part():
    """ROPE_R == D: the pooled keys are roped whole (the kernels' PASS == 0 branch)"""
    rnd = random.Random(7)
    planes = Planes(4, rope_r = D)
    seq = Seq(rnd.sample(range(4), 4), torch.Generator().manual_seed(7))
    assert run_conversation(planes, seq, rnd, 4, prompt = 133, steps = 20)


@needs_kernels
def test_pooled_keys_are_the_full_planes_for_several_sequences_per_call():
    """Batches as the generator makes them: sequences at their own positions, each in its own pages, one
    append length for the call (a graph slot of bsz x q_len, an eager chunk of several prompts)"""
    rnd = random.Random(11)
    gen = torch.Generator().manual_seed(11)
    pages = list(range(24))
    rnd.shuffle(pages)
    planes = Planes(24)
    seqs = [Seq(pages[i * 6 : (i + 1) * 6], gen) for i in range(4)]
    # prompts of different lengths: one eager call per sequence
    for s in seqs:
        L = rnd.randint(50, 400)
        assert batch(planes, [s], L, "eager", 6)
        s.pos += L
    # a batched eager chunk (same length for all) and batched decode steps
    assert batch(planes, seqs, 37, "eager", 6)
    for s in seqs:
        s.pos += 37
    for _ in range(15):
        L = rnd.randint(1, 4)
        assert batch(planes, seqs, L, "graph", 6)
        for s in seqs:
            s.pos += 1 + rnd.randint(0, L - 1)


@needs_kernels
def test_pooled_keys_are_the_full_planes_after_a_replay_and_on_reused_pages():
    """A banned-string rewind on a recurrent model replays from a page-aligned checkpoint, with the same
    raw keys; a finished sequence's pages go to another one, which starts from a page boundary (its prompt
    cache: complete pages shared by prefix, with no raw keys in the ring for them)"""
    rnd = random.Random(23)
    gen = torch.Generator().manual_seed(23)
    planes = Planes(16)
    a = Seq(list(range(8)), gen)
    assert run_conversation(planes, a, rnd, 8, prompt = 500, steps = 25)
    # replay from the last page boundary at or before the position, with the same keys
    end = a.pos
    a.pos = end // PAGE * PAGE
    while a.pos < end:
        L = min(rnd.randint(1, 200), end - a.pos)
        assert batch(planes, [a], L, "eager", 8, replay = True)
        a.pos += L
    assert run_conversation(planes, a, rnd, 8, prompt = 0, steps = 10)
    # b shares a's first two (complete) pages and takes over a's later pages, written in another order
    b = Seq([a.pages[0], a.pages[1]] + [7, 6, 5, 4, 3, 2], gen, prefix_keys = a.keys[: 2 * PAGE].clone())
    assert run_conversation(planes, b, rnd, 8, prompt = b.pos + 200, steps = 15)
    # and pages that were never written, the rest of the pool
    c = Seq(list(range(8, 16)), gen)
    assert run_conversation(planes, c, rnd, 8, prompt = 263, steps = 15)


@needs_kernels
def test_long_chunks_that_start_inside_a_block():
    """Prompt chunks of thousands of tokens (production: 4096) whose start is inside a block: the block's
    raw keys before the chunk come from the ring, the chunk's own from its rows"""
    gen = torch.Generator().manual_seed(5)
    planes = Planes(18)
    seq = Seq(list(range(17, -1, -1)), gen)
    for L in (6, 4096, 3):
        assert batch(planes, [seq], L, "eager", 18)
        seq.pos += L
    for L in (2, 16, 2):
        assert batch(planes, [seq], L, "graph", 18)
        seq.pos += 1


@needs_kernels
def test_the_comparison_sees_the_eager_kernels_in_the_other_order():
    """Not blind: with the ring's append before the pool update, a long chunk that starts inside a block
    takes the ring slots of that block's earlier raw keys, and the block's pooled key comes out wrong"""
    gen = torch.Generator().manual_seed(9)
    planes = Planes(4, eager_order = "append_first")
    seq = Seq([0, 1, 2, 3], gen)
    assert batch(planes, [seq], 34, "eager", 4)      # 34: position 32, 33 in ring slots 0, 1
    seq.pos += 34
    # the next chunk starts in block 8 (positions 32..35), reads 32 and 33 from the ring; written before
    # the pool update, its own positions 64 and 65 have taken slots 0 and 1
    assert not batch(planes, [seq], 40, "eager", 4)


@needs_kernels
def test_the_comparison_sees_a_pass_longer_than_the_ring_serves():
    """Not blind either: why the generator refuses drafts past raw_tail_max_rows. A 16-row ring takes
    verification passes of 13 rows; one of 15 rows, its drafts rejected, leaves the ring without the block
    the next pass starts in"""
    gen = torch.Generator().manual_seed(13)
    planes = Planes(4, ring = 0)
    planes.ring = make_layer(32, 4)
    planes.ring.raw_tail = 16      # past the constructor's check, to see what the check prevents
    planes.ring.raw_k = planes.ring.raw_k[:, :16].contiguous()
    seq = Seq([0, 1, 2, 3], gen)
    assert batch(planes, [seq], 2, "eager", 4)
    seq.pos += 2
    # a pass from position 2 (block 0 holds 0, 1 from before) over 2..16; drafts rejected: back to 3
    assert batch(planes, [seq], 15, "eager", 4)
    seq.pos = 3
    # the pass at 3 reads positions 0..2 from the ring: 16 has taken slot 0
    assert not batch(planes, [seq], 2, "graph", 4)


# --------------------------------------------------------------------------------------------
#   The eager path: QSAIndexer.update_planes itself
# --------------------------------------------------------------------------------------------

class Proj:
    """index_qk_proj: the rows given are the projection's output already ([q heads || raw key])"""

    def forward(self, x, params):
        return x


def make_indexer(rope, n_heads = 2):
    idx = M.indexer.QSAIndexer.__new__(M.indexer.QSAIndexer)
    idx.compress_ratio = P
    idx.head_dim = D
    idx.n_heads = n_heads
    idx.index_qk_proj = Proj()
    norm = lambda: types.SimpleNamespace(weight = types.SimpleNamespace(data = (torch.randn(D) * 0.1).half().to(DEVICE)),
                                         rms_norm_eps = 1e-6)
    idx.q_layernorm = norm()
    idx.k_layernorm = norm()
    return idx


@needs_kernels
def test_update_planes_with_the_ring_gives_the_full_planes_pooled_keys(monkeypatch):
    """The library's eager path, both layers through the same indexer: chunks of every kind, then decode
    steps the size of a fallback (bsz > 8 or a long verification pass), several sequences per call"""
    if DEVICE.type == "cpu":
        monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    rnd = random.Random(3)
    gen = torch.Generator().manual_seed(3)
    rope = Rope()
    idx = make_indexer(rope, n_heads = 1)
    full, ring = make_layer(0, 16), make_layer(RING, 16)
    pos = [0, 0]
    pages = [list(range(0, 8)), list(range(8, 16))]
    rnd.shuffle(pages[0])
    bt = torch.tensor(pages, dtype = torch.int32, device = DEVICE)
    hist = torch.empty((2, 8 * PAGE, (idx.n_heads + 1) * D), dtype = torch.half, device = DEVICE)
    for step in range(20):
        L = rnd.choice([1, 2, 5, 29, 64, 130]) if step < 8 else rnd.randint(1, 29)
        if max(pos) + L > 8 * PAGE:
            break
        for b in range(2):
            hist[b, pos[b] : pos[b] + L] = (torch.randn((L, hist.shape[2]), generator = gen) * 2).half()
        x = torch.stack([hist[b, pos[b] : pos[b] + L] for b in range(2)]).contiguous()
        seqlens = torch.tensor(pos, dtype = torch.int32)
        params = {"cache_seqlens": seqlens.to(DEVICE)}
        for layer in (full, ring):
            idx.update_planes(layer, x, rope, bt, seqlens, params)
        assert torch.equal(full.pooled, ring.pooled), step
        if step >= 8:
            # speculative: some of the rows rejected
            pos = [p + 1 + rnd.randint(0, L - 1) for p in pos]
        else:
            pos = [p + L for p in pos]
    # the ring holds what the ring kernels wrote: the last positions of each page, nothing for complete pages
    assert full.raw_k.abs().sum() > 0 and ring.raw_k.shape == (16, RING, D)


# --------------------------------------------------------------------------------------------
#   The library's torch references (no Triton)
# --------------------------------------------------------------------------------------------

class RefRope:
    def __init__(self, n = 16 * PAGE):
        inv = 1.0 / (10000.0 ** (torch.arange(0, ROPE_R, 2, dtype = torch.float) / ROPE_R))
        f = torch.outer(torch.arange(n, dtype = torch.float), inv)
        emb = torch.cat((f, f), dim = -1)
        self.cached_cos, self.cached_sin = emb.cos().to(DEVICE), emb.sin().to(DEVICE)

    def expand_cache(self, n):
        assert n <= self.cached_cos.shape[0]


class RefNorm:
    """RMSNorm with the +1 bias in fp32, out fp16 (what k_layernorm.forward computes; any fixed
    function would do, both layers go through the same one)"""

    def __init__(self):
        self.w = (torch.randn(D) * 0.1).to(DEVICE)

    def forward(self, x, params, out_dtype = None):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim = True) + 1e-6) * (self.w + 1.0)
        return y.half()


class RefProj:
    def forward(self, x, params):
        return x


def test_update_planes_ref_with_the_ring_gives_the_full_planes_pooled_keys():
    """update_planes_ref and its ring form, the torch statements of the same addressing: the chunk's own
    raw keys from the projection, the block it starts in from the ring, then the chunk's last raw_tail raw
    keys into the ring. Prompt chunks, speculative steps with rejected drafts, a replay"""
    rnd = random.Random(31)
    gen = torch.Generator().manual_seed(31)
    idx = M.indexer.QSAIndexer.__new__(M.indexer.QSAIndexer)
    idx.compress_ratio, idx.head_dim, idx.n_heads = P, D, 1
    idx.index_qk_proj = RefProj()
    idx.q_layernorm = types.SimpleNamespace(forward = lambda x, params: x)
    idx.k_layernorm = RefNorm()
    rope = RefRope()
    full, ring = make_layer(0, 8), make_layer(RING, 8)
    bt = torch.tensor([rnd.sample(range(8), 8)], dtype = torch.int32, device = DEVICE)
    keys = torch.empty((8 * PAGE, 2 * D), dtype = torch.half, device = DEVICE)
    pos, n_checked = 0, 0

    def append(L, replay = False):
        nonlocal n_checked
        if not replay:
            keys[pos : pos + L] = (torch.randn((L, 2 * D), generator = gen) * 2).half().to(DEVICE)
        x = keys[pos : pos + L].unsqueeze(0).contiguous()
        seqlens = torch.tensor([pos], dtype = torch.int32)
        for layer in (full, ring):
            idx.update_planes_ref(layer, x, rope, bt, seqlens, {})
        n_checked += 1
        return torch.equal(full.pooled, ring.pooled)

    for L in (5, 300, 1, 255, 600):
        assert append(L)
        pos += L
    for _ in range(80):
        L = rnd.randint(1, 16)
        assert append(L)
        pos += 1 + rnd.randint(0, L - 1)
    end, pos = pos, pos // PAGE * PAGE
    while pos < end:
        L = min(rnd.randint(1, 100), end - pos)
        assert append(L, replay = True)
        pos += L
    for _ in range(20):
        L = rnd.randint(1, 16)
        assert append(L)
        pos += 1 + rnd.randint(0, L - 1)
    # the full plane's pooled keys are not trivially equal: complete blocks were written
    assert n_checked > 100 and full.pooled.abs().sum() > 0


# --------------------------------------------------------------------------------------------
#   The graph slot: compiled for the cards, from BCAttn._configure_qsa
# --------------------------------------------------------------------------------------------

def _runtime_args(fn_name: str, module_path: str):
    tree = ast.parse(open(module_path).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    return [a.arg for a in fn.args.args if not (a.annotation is not None and "constexpr" in ast.unparse(a.annotation))]


def test_the_slot_kernels_take_the_full_plane_kernels_arguments():
    """The slot launches its raw append and pool update with a fixed argument list (attention.cpp, from
    set_qsa's raw plane): the ring's kernels must take exactly the same runtime arguments"""
    kern = os.path.join(SRC, "modules", "attention_fn", "qsa_triton.py")
    mla = os.path.join(SRC, "modules", "attention_fn", "mla_triton.py")
    assert _runtime_args("_qsa_raw_tail_append_kernel", kern) == _runtime_args("_mla_plane_update_kernel", mla)
    assert _runtime_args("_qsa_pool_update_ring_kernel", kern) == _runtime_args("_qsa_pool_update_kernel", kern)
    assert _runtime_args("_qsa_pool_update_rows_kernel", kern) == _runtime_args("_qsa_pool_update_kernel", kern) + ["rows_new"]


FP_OP = re.compile(r"^\s*(?:@!?%p\d+\s+)?((?:add|sub|mul|fma|div|rcp|sqrt|rsqrt|sin|cos|ex2|lg2|neg|abs|min|max|cvt)"
                   r"(?:\.[a-z0-9]+)*\.(?:f32|f16|f16x2|bf16|f64)(?:\.f32|\.f16)?)\b")


def _fp_ops(ptx: str) -> list:
    """The floating-point instructions of a PTX listing in program order, each as its opcode with every
    modifier (rounding, flush to zero, approximation, types)"""
    return [m.group(1) for m in map(FP_OP.match, ptx.splitlines()) if m]


def _compile(fn, signature, constexprs, num_warps, num_stages, attrs = None):
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    src = ASTSource(fn = fn, signature = signature, constexprs = constexprs, attrs = attrs or {})
    return triton.compile(src, target = GPUTarget("cuda", 120, 32),
                          options = {"num_warps": num_warps, "num_stages": num_stages})


def _slot_kernels(ring: int, q_len: int):
    """BCAttn._configure_qsa (regime 0) on a stand-in BCAttn, with _compile_kernel compiling for sm_120
    (the RTX 5060 Ti) instead of the current device: {kernel name: (compiled, signature, constexprs)},
    and the kernels configure_slot_qsa was given"""
    compiled = {}

    def compile_kernel(device, fn, signature, constexprs, num_warps, num_stages, pointer_range_32 = False):
        sig, attrs = M.bc.signature_to_ast(fn, signature, pointer_range_32)
        compiled[fn.__name__] = (_compile(fn, sig, constexprs, num_warps, num_stages, attrs), signature, constexprs)
        return fn.__name__

    slot = {}
    norm = types.SimpleNamespace(rms_norm_eps = 1e-6)
    bca = types.SimpleNamespace(
        device = torch.device("cuda", 0),
        qsa_idx = types.SimpleNamespace(n_heads = 16, head_dim = D, compress_ratio = P, q_layernorm = norm,
                                        k_layernorm = norm, scale = D ** -0.5, block_topk = 512),
        qsa_layer = types.SimpleNamespace(raw_tail = ring),
        module = types.SimpleNamespace(rope = Rope()),
        bc = types.SimpleNamespace(configure_slot_qsa = lambda *a: slot.setdefault("args", a)),
    )
    old = M.bc._compile_kernel
    M.bc._compile_kernel = compile_kernel
    try:
        M.bc.BCAttn._configure_qsa(bca, 1, q_len, 0)
    finally:
        M.bc._compile_kernel = old
    return compiled, slot["args"]


def _compile_report() -> dict:
    """Run in a process of its own (EXL3_TEST_QSA_COMPILE=1, triton without the interpreter): the slot's
    plane kernels as _configure_qsa compiles them with and without the ring, and the eager path's pool
    updates, reduced to what the tests compare"""
    report = {"slot": {}}
    for q_len in (1, 2, 16):
        entry = {}
        for name, ring in (("full", 0), ("ring", RING)):
            kernels, args = _slot_kernels(ring, q_len)
            entry[name] = {
                k: {"sig": sig, "cx": cx, "fp": _fp_ops(ck.asm["ptx"]), "cubin": len(ck.asm.get("cubin") or b"")}
                for k, (ck, sig, cx) in kernels.items()
            }
            entry[name + "_slot"] = list(args[7:9])
        report["slot"][str(q_len)] = entry
    base = {"raw_plane": "*fp16", "pool_plane": "*fp16", "k_norm_w": "*fp16", "inv_freq": "*fp32",
            "block_table": "*i32", "cache_seqlens": "*i32", "num_pages_per_row": "i32", "append_len": "i32"}
    cx = dict(page_size = PAGE, P = P, D = D, ROPE_R = ROPE_R, attn_factor = 1.0, eps = 1e-6, MAXPOOLS = 1)
    constexpr = lambda d: {k: "constexpr" for k in d}
    report["eager"] = {
        "full": _fp_ops(_compile(M.kern._qsa_pool_update_kernel, base | constexpr(cx), cx, 4, 2).asm["ptx"]),
        "rows": _fp_ops(_compile(M.kern._qsa_pool_update_rows_kernel, base | {"rows_new": "*fp16"}
                                 | constexpr(cx) | {"RING": "constexpr"}, cx | {"RING": RING}, 4, 2).asm["ptx"]),
    }
    return report


@pytest.fixture(scope = "module")
def compiled():
    tests = os.path.dirname(os.path.abspath(__file__))
    code = (f"import sys, json; sys.path.insert(0, {tests!r}); import test_qsa_raw_tail as T; "
            f"print('REPORT' + json.dumps(T._compile_report()))")
    env = dict(os.environ, **{COMPILE_ENV: "1"})
    env.pop("TRITON_INTERPRET", None)
    r = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True, timeout = 1200)
    assert r.returncode == 0, r.stderr[-6000:]
    return json.loads(next(l for l in r.stdout.splitlines() if l.startswith("REPORT"))[6:])


@needs_compiler
@pytest.mark.parametrize("q_len", ["1", "2", "16"])
def test_the_slot_compiles_the_rings_kernels_with_the_same_arithmetic(compiled, q_len):
    """_configure_qsa with and without the ring, compiled for sm_120: the ring's kernels in place of the
    full plane's, with the same runtime signature, and the pool update's floating-point instructions the
    same, in the same order, opcode for opcode (contractions, rounding modes and approximations included)"""
    entry = compiled["slot"][q_len]
    full, ring = entry["full"], entry["ring"]
    assert set(full) == {"_qsa_stage_kernel", "_mla_plane_update_kernel", "_qsa_pool_update_kernel"}
    assert set(ring) == {"_qsa_stage_kernel", "_qsa_raw_tail_append_kernel", "_qsa_pool_update_ring_kernel"}
    # what configure_slot_qsa is given as k_raw_append and k_pool_update
    assert entry["full_slot"] == ["_mla_plane_update_kernel", "_qsa_pool_update_kernel"]
    assert entry["ring_slot"] == ["_qsa_raw_tail_append_kernel", "_qsa_pool_update_ring_kernel"]
    rt = lambda sig: [(k, v) for k, v in sig.items() if v != "constexpr"]
    assert rt(ring["_qsa_raw_tail_append_kernel"]["sig"]) == rt(full["_mla_plane_update_kernel"]["sig"])
    assert rt(ring["_qsa_pool_update_ring_kernel"]["sig"]) == rt(full["_qsa_pool_update_kernel"]["sig"])
    assert ring["_qsa_pool_update_ring_kernel"]["cx"]["RING"] == RING
    assert ring["_qsa_pool_update_ring_kernel"]["cx"]["MAXPOOLS"] == int(q_len) // P + 1
    a, b = full["_qsa_pool_update_kernel"]["fp"], ring["_qsa_pool_update_ring_kernel"]["fp"]
    assert len(a) > 50 and Counter(a)["cvt.rn.f16.f32"] >= 4
    assert a == b
    # the stage kernel is the same one either way, and every kernel has its cubin
    assert ring["_qsa_stage_kernel"] == full["_qsa_stage_kernel"]
    assert all(k["cubin"] > 0 for k in ring.values())


@needs_compiler
def test_the_eager_rows_kernel_has_the_same_arithmetic(compiled):
    """The eager path's pool update (raw keys from the rows and the ring) against the full plane's, both
    compiled for sm_120 with the same constants"""
    a, b = compiled["eager"]["full"], compiled["eager"]["rows"]
    assert len(a) > 50 and a == b
