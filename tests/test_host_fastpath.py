"""
The decode step's host fast path (EXL3_HOST_FASTPATH=1, off by default; FORK.md, "The decode step's
host work, kept between calls").

The switch is read once, when a module is imported, so the sources under test are loaded twice, into
two packages: one with the switch off (today's code) and one with it on. The same calls are driven
through both, and what reaches the extension is compared call by call: the function, and for every
argument its value or, for a tensor, the buffer it lives in, its offset, shape, strides and dtype.
A buffer is named by what it is (the workspace cache's key of the entry it belongs to, the module
tensor, the test's input) or, for one made during the run, by the order it was first seen in on its
thread, so two runs compare even when their memory is laid out differently. Every torch operation
that is not a view (an allocation, a copy, a cast) is recorded in the same sequence, at the moment it
runs, with the same naming. With the switch on, the views a site used to make again at every call are
kept: those are the only operations the two sequences may differ in, and they are left out.

What the scenarios cover: gated residual sites in the order a pass meets them (attention and MLP
site of each block, the final mixer), batch shapes changing from pass to pass, rows past the fused
decode pair, stream stacks the plain path converts first, the weighted-copy handoff (to the next
site, and broken by a successor on another device), sublayer outputs of every dtype the plain path
casts and non-contiguous ones it copies, the conversion and capture passes, entries added to the
workspace cache between a mix and its apply, the cache emptied between passes, passes from threads
on lanes 1 and 2 (their own workspaces), two threads issuing at the same time, and the MoE routing
buffers at every row count up to past the kept ones.

No GPU or extension is required: the extension is a recorder.
"""

import importlib.util
import os
import random
import sys
import threading
import types

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------------------------
#   Loading
# --------------------------------------------------------------------------------------------

class RecordingExt(types.ModuleType):
    """The compiled extension as the modules under test see it: every call is handed to the
    world's recorder; the few functions whose result the Python side reads return one."""

    HAS_GR_MIX_TILED = False
    HAS_DET_GEMM = True

    def __init__(self):
        super().__init__("exllamav3_ext")
        self.recorder = None

    def hc_mix_num_chunks(self, R, n):
        return 4

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def call(*args, **kwargs):
            rec = self.recorder
            if rec is not None:
                rec.ext(name, args, kwargs)
            return None

        call.__name__ = name
        return call


def load_world(tag: str, fastpath: bool):
    """Load util/lane, util/tensor, util/device_copy, util/backend, modules/hyperconnections and
    modules/block_sparse_mlp_routing into a package of their own, with EXL3_HOST_FASTPATH set as
    asked while they import."""
    pkg = f"exl3_host_fastpath_{tag}"
    src = os.path.join(REPO, "exllamav3")
    ext = RecordingExt()

    def stub(name, **attrs):
        full = f"{pkg}.{name}" if name else pkg
        m = types.ModuleType(full)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[full] = m
        return m

    def real(name):
        full = f"{pkg}.{name}"
        spec = importlib.util.spec_from_file_location(full, os.path.join(src, *name.split(".")) + ".py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        return m

    class Module:
        def __init__(self, config = None, key = None, qmap = None):
            self.config, self.key, self.qmap, self.device = config, key, qmap, None

        def register_submodule(self, m):
            pass

        def load(self, device, **kwargs):
            self.device = device

        def unload(self):
            pass

    class RMSNorm(Module):
        def __init__(self, config, key, rms_norm_eps, **kwargs):
            super().__init__(config, key, None)

    old = os.environ.get("EXL3_HOST_FASTPATH")
    os.environ["EXL3_HOST_FASTPATH"] = "1" if fastpath else "0"
    try:
        stub("")
        for p in ("util", "modules", "model", "tokenizer"):
            stub(p)
        stub("ext", exllamav3_ext = ext)
        stub("model.config", Config = object)
        stub("modules.module", Module = Module)
        stub("modules.rmsnorm", RMSNorm = RMSNorm)
        stub("tokenizer.mm_embedding", FIRST_MM_EMBEDDING_INDEX = 1000000000)
        w = types.SimpleNamespace(
            tag = tag,
            fastpath = fastpath,
            ext = ext,
            lane = real("util.lane"),
            device_copy = real("util.device_copy"),
            tensor = real("util.tensor"),
            backend = real("util.backend"),
            hc = real("modules.hyperconnections"),
            routing = real("modules.block_sparse_mlp_routing"),
        )
    finally:
        if old is None:
            os.environ.pop("EXL3_HOST_FASTPATH", None)
        else:
            os.environ["EXL3_HOST_FASTPATH"] = old
    return w


OFF = load_world("off", False)
ON = load_world("on", True)


def test_the_switch_is_read_at_import():
    for w, on in ((OFF, False), (ON, True)):
        assert w.lane.HOST_FASTPATH is on and w.tensor.HOST_FASTPATH is on
        assert w.hc.HOST_FASTPATH is on and w.routing.HOST_FASTPATH is on
    # Off: the plain functions are the only ones; on: the plain ones are kept beside the fast ones
    GR0, GR1 = OFF.hc.GatedResidual, ON.hc.GatedResidual
    assert not hasattr(GR0, "_mix_fast") and not hasattr(GR0, "_apply_plain")
    assert not hasattr(OFF.routing, "_routing_buffers_plain")
    assert not hasattr(OFF.tensor.g_tensor_cache, "fast_views")
    assert GR1._mix_plain.__code__ is not GR1._mix.__code__ and GR1._apply_plain.__code__ is not GR1.apply_.__code__
    # The plain functions of the fast world are today's, line for line
    assert GR1._mix_plain.__code__.co_code == GR0._mix.__code__.co_code
    assert GR1._apply_plain.__code__.co_code == GR0.apply_.__code__.co_code
    assert ON.routing._routing_buffers_plain.__code__.co_code == OFF.routing._routing_buffers.__code__.co_code


# --------------------------------------------------------------------------------------------
#   Recording
# --------------------------------------------------------------------------------------------

class Recorder:
    """One world's record: extension calls and the torch operations that are not views, per
    thread, in order. Tensors are described as (buffer, byte offset, shape, strides, dtype).
    Dispatch modes are per thread: a thread the scenario starts enters its own (mode())."""

    def __init__(self, world):
        self.world = world
        self.named = {}              # storage data_ptr -> name, for the test's own tensors
        self.keep = []               # ... which are kept alive: their addresses stay theirs
        self.lock = threading.Lock()
        self.events = {}             # thread name -> [event]
        self.anon = {}               # thread name -> {storage data_ptr: ordinal}
        self._anon_count = {}

    def mode(self):
        return _RecordingMode(self)

    def name(self, tensors: dict):
        for n, t in tensors.items():
            self.keep.append(t)
            self.named[t.untyped_storage().data_ptr()] = n

    def _buffer(self, t: torch.Tensor):
        base = t.untyped_storage().data_ptr()
        for key, (_, v) in list(self.world.tensor.g_tensor_cache.cache.items()):
            if v.untyped_storage().data_ptr() == base:
                return "ws:" + key
        name = self.named.get(base)
        if name is not None:
            return name
        th = threading.current_thread().name
        an = self.anon.setdefault(th, {})
        if base not in an:
            an[base] = self._next_anon(th)
        return f"anon{an[base]}"

    def describe(self, a):
        if isinstance(a, torch.Tensor):
            if a.device.type == "meta":
                return ("meta", tuple(a.shape), str(a.dtype))
            base = a.untyped_storage().data_ptr()
            return (self._buffer(a), a.data_ptr() - base, tuple(a.shape), tuple(a.stride()), str(a.dtype))
        if isinstance(a, (list, tuple)):
            return type(a).__name__, tuple(self.describe(x) for x in a)
        if isinstance(a, (int, float, bool, str, type(None), torch.dtype, torch.device)):
            return a
        return type(a).__name__

    def _add(self, ev):
        th = threading.current_thread().name
        with self.lock:
            self.events.setdefault(th, []).append(ev)

    def ext(self, name, args, kwargs):
        self._add(("ext", name, tuple(self.describe(a) for a in args),
                   tuple(sorted((k, self.describe(v)) for k, v in kwargs.items()))))

    def op(self, func, args, kwargs, out):
        # An output in a storage none of the inputs is in is a new one: a new name, even where the
        # allocator handed out the address of one that has gone (the two worlds reuse memory in
        # their own order). In place and out= results keep their input's name.
        def tensors(x):
            if isinstance(x, torch.Tensor):
                yield x
            elif isinstance(x, (list, tuple)):
                for y in x:
                    yield from tensors(y)
        ins = {t.untyped_storage().data_ptr() for t in tensors((args, list(kwargs.values())))
               if t.device.type != "meta"}
        desc_in = tuple(self.describe(a) for a in args)
        desc_kw = tuple(sorted((k, self.describe(v)) for k, v in kwargs.items()))
        th = threading.current_thread().name
        an = self.anon.setdefault(th, {})
        for t in tensors(out):
            if t.device.type == "meta":
                continue
            base = t.untyped_storage().data_ptr()
            if base not in ins and base not in self.named and base:
                an[base] = self._next_anon(th)
        self._add(("op", str(func), desc_in, desc_kw, self.describe(out)))

    def _next_anon(self, th):
        n = self._anon_count.get(th, 0)
        self._anon_count[th] = n + 1
        return n


class _RecordingMode(TorchDispatchMode):

    def __init__(self, rec):
        super().__init__()
        self.rec = rec

    def __torch_dispatch__(self, func, types, args = (), kwargs = None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if not func.is_view:
            self.rec.op(func, args, kwargs, out)
        return out


def record(world, scenario):
    """Run scenario(world, rec) under the world's recorder; the events per thread."""
    rec = Recorder(world)
    world.ext.recorder = rec
    world.rec = rec
    try:
        with rec.mode():
            scenario(world, rec)
    finally:
        world.ext.recorder = None
        world.rec = None
    return rec.events


def assert_same(off_events, on_events):
    assert set(off_events) == set(on_events)
    for th in off_events:
        a, b = off_events[th], on_events[th]
        n = min(len(a), len(b))
        for i in range(n):
            assert a[i] == b[i], f"thread {th}, event {i}:\n off {a[i]}\n on  {b[i]}"
        assert len(a) == len(b), f"thread {th}: {len(a)} events off, {len(b)} on"
        assert any(e[0] == "ext" for e in a)


# --------------------------------------------------------------------------------------------
#   A toy pass through gated residual sites
# --------------------------------------------------------------------------------------------

H, D = 4, 256


def make_site(world, key, rank = 64, use_combine = True, seed = 0):
    g = torch.Generator().manual_seed(seed)
    gr = world.hc.GatedResidual(None, key, H, D, 1e-6, use_combine = use_combine,
                                out_dtype = None if use_combine else torch.half)
    gr.norm_w_raw = torch.randn(H * D, generator = g).half()
    down = torch.randn(rank, H * D, generator = g).half()
    up = torch.randn(H * D, rank, generator = g).half()
    inject = torch.randn(H, H * D, generator = g).half() if use_combine else None
    gr._prepare(down, up, inject, keep_source_weights = True)
    gr.device = torch.device("cpu")
    return gr


class ToyModel:
    """Blocks of two sites (attention, MLP) and a final mixer, wired as link_sites wires them.
    One block's sites have another rank (another table height, so another dots workspace)."""

    def __init__(self, world, blocks = 3):
        self.world = world
        self.blocks = []
        for i in range(blocks):
            rank = 128 if i == 1 else 64
            self.blocks.append((make_site(world, f"l{i}.attn_hc", rank, seed = 2 * i),
                                make_site(world, f"l{i}.mlp_hc", rank, seed = 2 * i + 1)))
        self.mixer = make_site(world, "mixer", use_combine = False, seed = 99)
        modules = [types.SimpleNamespace(attn_hc = a, mlp_hc = m) for a, m in self.blocks] + [self.mixer]
        world.hc.GatedResidual.link_sites(modules)

    def named(self):
        """name -> tensor, for the recorder"""
        out = {}
        sites = [(f"l{i}.attn", a) for i, (a, _) in enumerate(self.blocks)] + \
                [(f"l{i}.mlp", m) for i, (_, m) in enumerate(self.blocks)] + [("mixer", self.mixer)]
        for n, site in sites:
            for attr in ("proj_h", "upx_h", "w_h", "up_h", "norm_w", "norm_w_raw"):
                t = getattr(site, attr)
                if t is not None:
                    out[f"{n}.{attr}"] = t
        return out


def sublayer(y, kind):
    """A sublayer's output from its input, of the dtype and layout `kind` asks for."""
    b, s, d = y.shape
    if kind == "float":
        return y.float() * 0.5
    if kind == "half":
        return (y * 0.5).half()
    if kind == "bf16":
        return y.to(torch.bfloat16)
    if kind == "strided":
        # A non-contiguous output (apply_ copies it)
        t = torch.empty((b, s, 2 * d), dtype = torch.float)
        t[..., ::2] = y.float()
        return t[..., ::2]
    raise ValueError(kind)


def run_pass(model, b, s, kinds = ("float", "half"), params = None, between = None, streams = None):
    """One forward pass over the toy's sites as TransformerBlock drives them."""
    params = {} if params is None else params
    if streams is None:
        streams = torch.full((b, s, H, D), 0.25, dtype = torch.float)
    x = streams
    for i, (attn_hc, mlp_hc) in enumerate(model.blocks):
        for j, site in enumerate((attn_hc, mlp_hc)):
            post, comb, y, normed = site.mix_norm(x, params, None)
            y = y.half()
            out = sublayer(y, kinds[(i + j) % len(kinds)])
            if between is not None:
                between(i, j)
            x = site.apply_(x, out, post, comb, params)
    return model.mixer.forward(x, params)


def model_scenario(passes, kinds = ("float", "half"), mutate = None):
    def scenario(world, rec):
        model = ToyModel(world)
        rec.name(model.named())
        for n, (b, s) in enumerate(passes):
            if mutate is not None:
                mutate(world, model, n)
            run_pass(model, b, s, kinds)
    return scenario


SHAPES = [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (1, 1), (3, 2), (4, 2), (1, 8), (1, 9), (2, 5), (1, 1),
          (2, 2), (8, 1), (1, 3), (16, 1), (1, 2)]


def compare(scenario):
    a = record(OFF, scenario)
    b = record(ON, scenario)
    assert_same(a, b)
    return a, b


def reset(world):
    world.tensor.g_tensor_cache.drop_all()


@pytest.fixture(autouse = True)
def _fresh_caches():
    reset(OFF)
    reset(ON)
    yield
    reset(OFF)
    reset(ON)


def test_a_pass_issues_the_same_calls_through_every_batch_shape():
    off, on = compare(model_scenario(SHAPES))
    calls = [e for e in on["MainThread"] if e[0] == "ext"]
    # Rows past the fused pair (9, 10, 16) take the cuBLAS path (an rms_norm launch and torch's
    # matmuls, whose ops are in the same sequence too): the fast path left them alone
    assert {e[1] for e in calls} == {"gr_mix", "hc_apply", "rms_norm"}
    assert any(e[0] == "op" and e[1].startswith("aten.mm") for e in on["MainThread"])


def test_every_sublayer_dtype_and_layout():
    compare(model_scenario(SHAPES[:8], kinds = ("bf16", "strided", "half", "float")))


def test_stream_stacks_the_plain_path_converts_first():
    def scenario(world, rec):
        model = ToyModel(world)
        rec.name(model.named())
        site = model.blocks[0][0]
        for b, s in ((1, 1), (2, 2), (1, 3)):
            # half (MTP's draft stack), non-contiguous fp32, and contiguous fp32 again
            site.mix(torch.ones((b, s, H, D), dtype = torch.half), {})
            t = torch.ones((b, s, H, 2 * D))[..., ::2]
            site.mix(t, {})
            site.mix(torch.ones((b, s, H, D)), {})
            model.mixer.forward(torch.ones((b, s, H, D), dtype = torch.half), {})
    compare(scenario)


def test_the_final_mixer_and_cached_false():
    def scenario(world, rec):
        model = ToyModel(world)
        rec.name(model.named())
        for b, s in ((1, 1), (1, 4), (2, 1)):
            x = torch.ones((b, s, H, D))
            model.mixer.forward(x, {})
            model.mixer.forward(x, {"capture": True})            # conversion: fresh outputs
            model.mixer._mix(x, cached = False, params = {})
            model.mixer._mix(x, cached = True, params = {})
            model.blocks[0][1]._mix(x, cached = False, params = {})
    compare(scenario)


def test_conversion_passes_take_the_allocating_apply():
    def scenario(world, rec):
        model = ToyModel(world)
        rec.name(model.named())
        for p in ({"capture": True}, {"quant_preserve": True}, {}):
            run_pass(model, 1, 2, params = dict(p))
    compare(scenario)


def test_a_successor_on_another_device_breaks_the_handoff():
    def mutate(world, model, n):
        # The second block's MLP site's successor (the third block's attention site) sits on
        # another device from pass 2 on: no weighted copy for it
        if n == 2:
            nxt = model.blocks[1][1].next_site
            nxt.w_h = torch.empty(nxt.w_h.shape, dtype = nxt.w_h.dtype, device = "meta")
    compare(model_scenario(SHAPES[:6], mutate = mutate))


def test_entries_added_between_a_mix_and_its_apply():
    def scenario(world, rec):
        model = ToyModel(world)
        rec.name(model.named())
        gtc = world.tensor.g_tensor_cache

        def between(i, j):
            # a sublayer that takes a workspace of its own for the first time (a new entry)
            gtc.get_bucketed("cpu", 7 + 3 * i + j, torch.float, f"sublayer_{i}_{j}")

        for b, s in SHAPES[:6]:
            run_pass(model, b, s, between = between)
    compare(scenario)


def test_the_cache_emptied_between_passes():
    def mutate(world, model, n):
        if n in (2, 3, 7):
            world.tensor.g_tensor_cache.drop_all()
    compare(model_scenario(SHAPES[:10], mutate = mutate))


def test_a_rebuilt_table_is_read_again():
    def mutate(world, model, n):
        if n == 3:
            # tp_import and a reload build proj_h anew (_prepare)
            site = model.blocks[0][0]
            g = torch.Generator().manual_seed(7)
            site._prepare(torch.randn(64, H * D, generator = g).half(), torch.randn(H * D, 64, generator = g).half(),
                          torch.randn(H, H * D, generator = g).half(), keep_source_weights = True)
    a = record(OFF, model_scenario(SHAPES[:6], mutate = mutate))
    b = record(ON, model_scenario(SHAPES[:6], mutate = mutate))
    # (the new tables are not in the test's names: compare only that both worlds read the same
    # anonymous buffer at the same calls)
    assert_same(a, b)


def _on_lane(world, index, fn, name = None):
    """fn() on a thread that entered lane `index` (and the world's recorder, when one runs)."""
    out = {}

    def body():
        world.lane.Lane(index, [0, 1]).enter()
        rec = getattr(world, "rec", None)
        try:
            if rec is None:
                out["v"] = fn()
            else:
                with rec.mode():
                    out["v"] = fn()
        except Exception as e:
            out["e"] = e

    t = threading.Thread(target = body, name = name or f"lane{index}")
    t.start()
    t.join(30)
    assert not t.is_alive()
    if "e" in out:
        raise out["e"]
    return out["v"]


def test_passes_on_other_lanes_take_their_own_workspaces():
    def scenario(world, rec):
        model = ToyModel(world)
        rec.name(model.named())
        for n, (b, s) in enumerate(SHAPES[:9]):
            run_pass(model, b, s)
            _on_lane(world, 1 + n % 2, lambda: run_pass(model, b, s))
    off, on = compare(scenario)
    keys = {e[2][6][0] for e in on["lane1"] if e[0] == "ext" and e[1] == "gr_mix"}
    assert keys and all(k.endswith("/lane1") for k in keys)


def test_two_threads_issuing_at_once():
    """A thread on lane 1 and the main thread on lane 0 issue passes at the same time, many
    times: each thread's calls must be those it makes alone, whatever the interleaving."""

    def scenario(world, rec):
        model = ToyModel(world)
        rec.name(model.named())
        go = threading.Barrier(2)

        def worker():
            world.lane.Lane(1, [0]).enter()
            with rec.mode():
                go.wait()
                for b, s in SHAPES[:8] * 4:
                    run_pass(model, b, s)

        t = threading.Thread(target = worker, name = "lane1")
        t.start()
        go.wait()
        for b, s in SHAPES[3:11] * 4:
            run_pass(model, b, s)
            if random.random() < 0.2:
                world.tensor.g_tensor_cache.get_bucketed("cpu", random.randint(1, 1 << 12), torch.float, "noise")
        t.join(60)
        assert not t.is_alive()

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        random.seed(1)
        a = record(OFF, scenario)
        random.seed(1)
        b = record(ON, scenario)
    finally:
        sys.setswitchinterval(old)
    # Workspaces made by the other thread first get its ordinal in neither: compare names that
    # do not depend on who made what first (the cache's keys, the inputs and module tensors)
    assert_same(a, b)


# --------------------------------------------------------------------------------------------
#   The routing buffers
# --------------------------------------------------------------------------------------------

E, K, HID = 64, 4, 128


def make_cfg(world, seed = 0):
    g = torch.Generator().manual_seed(seed)
    return world.routing.RoutingCFG(
        gate_tensor = torch.randn(HID, E, generator = g).half(), gate_tensor_t = None,
        num_experts = E, num_experts_per_tok = K,
        router_logits_bsz1 = torch.empty(1, E).half(), routing_weights_bsz1 = torch.empty(1, K).half(),
        selected_experts_bsz1 = torch.empty(1, K, dtype = torch.long),
        e_score_correction_bias = None, e_score_bias_h = None, routed_scaling_factor = None,
        n_group = None, topk_group = None, per_expert_scale = None,
    )


def test_routing_buffers_at_every_row_count():
    rows = [1, 2, 3, 4, 2, 8, 6, 1, 64, 127, 128, 129, 200, 2, 4, 33, 2]

    def scenario(world, rec):
        cfgs = [make_cfg(world, 0), make_cfg(world, 1)]
        for i, bsz in enumerate(rows):
            for cfg in cfgs:
                y = torch.ones((bsz, HID), dtype = torch.half)
                sel, w = world.routing.routing_std(bsz, cfg, y, {})
                # what the MoE layer does with them before the next layer routes
                sel.view(-1)
                w.is_contiguous()
            if i == 9:
                world.tensor.g_tensor_cache.drop_all()
            if i % 4 == 3:
                _on_lane(world, 1, lambda: world.routing.routing_std(bsz, cfgs[0], torch.ones((bsz, HID)).half(), {}))
    compare(scenario)


def test_routing_buffers_are_the_plain_ones():
    gtc = ON.tensor.g_tensor_cache
    cfg = make_cfg(ON)
    for bsz in (2, 5, 2, 130):
        a = ON.routing._routing_buffers(cfg, bsz, torch.device("cpu"))
        b = ON.routing._routing_buffers_plain(cfg, bsz, torch.device("cpu"))
        for x, y in zip(a, b):
            assert (x.data_ptr(), x.shape, x.stride(), x.dtype) == (y.data_ptr(), y.shape, y.stride(), y.dtype) \
                or bsz > ON.routing.ROUTING_CACHE_ROWS
    # Kept, and taken again once the cache moved. The call that makes the entries keeps its views
    # under the count it read before making them, so the next call takes them once more
    ON.routing._routing_buffers(cfg, 3, torch.device("cpu"))
    a = ON.routing._routing_buffers(cfg, 3, torch.device("cpu"))
    assert ON.routing._routing_buffers(cfg, 3, torch.device("cpu"))[0] is a[0]
    gtc.drop_all()
    c = ON.routing._routing_buffers(cfg, 3, torch.device("cpu"))
    assert c[0] is not a[0]
    assert gtc.cache["cpu/(256,)/torch.float16/moe_route_logits"][1].data_ptr() == c[0].data_ptr()
    c = ON.routing._routing_buffers(cfg, 3, torch.device("cpu"))
    assert ON.routing._routing_buffers(cfg, 3, torch.device("cpu"))[0] is c[0]
    # the end state of two threads making the same entry at once (the cache holds the second)
    key = "cpu/(256,)/torch.float16/moe_route_logits"
    del gtc.cache[key]
    gtc.get_bucketed("cpu", 192, torch.half, "moe_route_logits")
    d = ON.routing._routing_buffers(cfg, 3, torch.device("cpu"))
    assert d[0].data_ptr() == gtc.cache[key][1].data_ptr() != c[0].data_ptr()


# --------------------------------------------------------------------------------------------
#   The workspace cache and the lane
# --------------------------------------------------------------------------------------------

def test_lanes_read_the_same_with_the_switch():
    for w in (OFF, ON):
        assert w.lane.current_lane() == 0 and w.lane.host_waits() is False
        assert _on_lane(w, 1, lambda: (w.lane.current_lane(), w.lane.host_waits())) == (1, w.lane._HOST_WAITS)
        assert _on_lane(w, 2, w.lane.current_lane) == 2
        # a thread that never entered one
        out = {}
        t = threading.Thread(target = lambda: out.setdefault("v", (w.lane.current_lane(), w.lane.host_waits())))
        t.start()
        t.join()
        assert out["v"] == (0, False)
        assert w.lane.current_lane() == 0


def _bucketed_calls(world, calls):
    """(key, refcount, backing data_ptr, slice geometry) of each call, and the cache's keys."""
    gtc = world.tensor.GTensorCache()
    out = []
    backings = {}
    for device, numel, dtype, tag, lane in calls:
        fn = lambda: gtc.get_bucketed(device, numel, dtype, tag)
        try:
            t = fn() if lane == 0 else _on_lane(world, lane, fn)
        except Exception as e:      # the plain path's own errors must be the fast path's too
            out.append(("raise", type(e).__name__))
            continue
        base = t.untyped_storage().data_ptr()
        name = backings.setdefault(base, len(backings))
        # (an empty slice has no address of its own)
        out.append((name, t.data_ptr() - base if t.numel() else None, tuple(t.shape), tuple(t.stride()), t.dtype))
    entries = sorted((k, refc, tuple(v.shape), v.dtype) for k, (refc, v) in gtc.cache.items())
    return out, entries


def test_get_bucketed_is_the_plain_lookup():
    rnd = random.Random(3)
    # an int device is cuda:N (raises here, the same way on both paths)
    devices = ["cpu", torch.device("cpu"), torch.device("cpu"), 0]
    tags = ["a", "b", "gr_mix_dots", 1, True, ("t",)]
    calls = []
    for _ in range(400):
        calls.append((rnd.choice(devices), rnd.randint(0, 5000), rnd.choice([torch.float, torch.half, torch.long]),
                      rnd.choice(tags), rnd.choice([0, 0, 0, 1, 2])))
    calls += [("cpu", 3, torch.float, ["unhashable"], 0), ("cpu", 3, torch.float, "a", 0), ("cpu", -1, torch.float, "a", 0),
              ("cpu", 1, float, "a", 0), ("cpu", 1, torch.float64, "a", 0)]
    assert _bucketed_calls(OFF, calls) == _bucketed_calls(ON, calls)


def test_get_bucketed_after_drop_all_makes_the_entry_again():
    gtc = ON.tensor.GTensorCache()
    a = gtc.get_bucketed("cpu", 100, torch.float, "x")
    g = gtc.fast_gen
    assert gtc.get_bucketed("cpu", 100, torch.float, "x").data_ptr() == a.data_ptr() and gtc.fast_gen == g
    gtc.drop_all()
    assert gtc.fast_gen == g + 1 and not gtc.cache and not gtc.fast_views
    b = gtc.get_bucketed("cpu", 100, torch.float, "x")
    assert gtc.fast_gen == g + 2 and list(gtc.cache) == ["cpu/(128,)/torch.float32/x"]
    assert gtc.cache["cpu/(128,)/torch.float32/x"][1].data_ptr() == b.data_ptr()


def test_the_kept_views_follow_the_cache():
    gtc = ON.tensor.g_tensor_cache
    model = ToyModel(ON)
    site = model.blocks[0][0]
    x = torch.ones((1, 2, H, D))
    site.mix(x, {})                                                 # makes the entries
    p1, _, y1 = site.mix(x, {})
    p2, _, y2 = site.mix(x, {})
    assert p1 is p2 and y1 is y2                                    # kept
    gtc.get_bucketed("cpu", 3, torch.float, "unrelated_new_entry")  # a new entry anywhere
    p3, _, y3 = site.mix(x, {})
    assert p3 is not p1 and p3.data_ptr() == p1.data_ptr()          # taken again, same backing
    assert site.mix(x, {})[0] is p3
    gtc.drop_all()
    p4, _, y4 = site.mix(x, {})
    key = "cpu/(512,)/torch.float16/gr_mix_mixed"
    assert gtc.cache[key][1].data_ptr() == y4.data_ptr()            # the new backing
    # unload forgets the kept table view
    site._proj_rows()
    assert "_fast_proj" in site.__dict__
    site.unload()
    assert "_fast_proj" not in site.__dict__


def test_an_apply_with_a_post_of_its_own():
    """apply_ hands the kept (R, H) view to the kernel only for the post its own site's mix
    returned; any other post is reshaped as the plain path does."""

    def scenario(world, rec):
        model = ToyModel(world)
        rec.name(model.named())
        site = model.blocks[0][0]
        for b, s in ((1, 1), (2, 2), (1, 3), (2, 2)):
            x = torch.ones((b, s, H, D))
            site.mix(x, {})                                  # a kept entry for this shape
            post, mixed = site._mix(x, cached = False, params = {})
            y = mixed.view(b, s, D).float()
            site.apply_(x, y, post.view(b, s, H), None, {})  # a fresh post
            p2 = torch.ones((b, s, 2 * H))[..., ::2]         # and a non-contiguous one
            site.apply_(x, y, p2, None, {})
            p, _, yy = site.mix(x, {})
            site.apply_(x, yy.float(), p, None, {})          # the kept one
    compare(scenario)


def test_an_entry_made_again_by_another_thread():
    """The end state of two threads making the same entry at once: the cache holds the second
    one. Kept views of the first are taken again (fast_gen moved), as the plain lookup would."""
    gtc = ON.tensor.g_tensor_cache
    model = ToyModel(ON)
    site = model.blocks[0][0]
    x = torch.ones((1, 2, H, D))
    for _ in range(2):
        p, _, y = site.mix(x, {})
    xw_key = ("gr_apply_xw", x.device, 2, H, D, 0)
    site.apply_(x, y.float(), p, None, {})
    site.apply_(x, y.float(), p, None, {})
    xw_old = gtc.fast_views[xw_key][1]
    for tag, n in (("gr_mix_mixed", 2 * D), ("gr_mix_post", 2 * H), ("gr_apply_xw", 2 * H * D)):
        key = [k for k in gtc.cache if k.endswith("/" + tag)][0]
        old = gtc.cache[key][1]
        del gtc.cache[key]
        gtc.get_bucketed("cpu", n, torch.float if tag != "gr_mix_mixed" else torch.half, tag)
        assert gtc.cache[key][1] is not old
    p2, _, y2 = site.mix(x, {})
    assert y2.data_ptr() == gtc.cache[[k for k in gtc.cache if k.endswith("/gr_mix_mixed")][0]][1].data_ptr()
    assert p2.data_ptr() == gtc.cache[[k for k in gtc.cache if k.endswith("/gr_mix_post")][0]][1].data_ptr()
    params = {}
    site.apply_(x, y2.float(), p2, None, params)
    xw_new = params["gr_weighted"][2]
    assert xw_new is not xw_old
    assert xw_new.data_ptr() == gtc.cache[[k for k in gtc.cache if k.endswith("/gr_apply_xw")][0]][1].data_ptr()
