"""
The prompt's last-page recurrent checkpoint taken inside the chunk that goes past it
(Generator.recurrent_checkpoint_in_pass, cache/recurrent_capture.py).

No GPU or model is required. Two parts.

The GDN scan's stop (_chunked_scan with split_at): over upstream's reference recurrence in torch,
the state handed over at the split is the final state a pass that ends there returns, its
sub-ranges are that pass's sub-ranges, and the scan goes on to the same output and final state.
The fla kernels and the module's conv window are checked on the cards (tests/modules/recurrent/).

The job and the generator: the real Job, PageTable, RecurrentCache, Generator prompt rounds
(inline, on the prefill thread, in the pipeline) over test_page_revert's stand-in model, two of
whose modules keep a recurrent state per slot (a scan over the token ids and a window of the
last ids, stashed as GDNLayerState stashes) and copy it at a capture's boundary the way the GDN
and PLE layers do. With the switch off the chunk ends at the last page boundary and a pass of
its own reads the tokens after it; with it on the chunk runs whole, and the checkpoint the
record takes is the one the shorter pass leaves: same key, same position, same values, the
format of a regular stash. The conversation's next turn resumes from it.
"""

import importlib.util
import os
import sys
import types

import pytest
import torch

import test_page_revert as PR
from test_page_revert import _fake_lane   # noqa: F401 (autouse fixture: lanes without CUDA)

pytestmark = pytest.mark.nogpu

GEN, JOB, PT = PR.GEN, PR.JOB, PR.PT
PAGE = PR.PAGE
PKG = JOB.__name__.rsplit(".generator", 1)[0]
REPO = PR.REPO


def _module(name: str):
    """A module of the package under test: imported, or (stubbed package) loaded from its source"""
    full = f"{PKG}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    if PKG == "exllamav3":
        return importlib.import_module(full)
    for i in range(1, len(name.split("."))):
        parent = f"{PKG}." + ".".join(name.split(".")[:i])
        if parent not in sys.modules:
            m = types.ModuleType(parent)
            m.__path__ = []
            sys.modules[parent] = m
    path = os.path.join(REPO, "exllamav3", *name.split(".")) + ".py"
    spec = importlib.util.spec_from_file_location(full, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[full] = m
    spec.loader.exec_module(m)
    return m


REC = _module("cache.recurrent")
CAP = _module("cache.recurrent_capture")
RUTIL = _module("cache.recurrent_util")
GDR = _module("modules.gated_delta_net_fn.gated_delta_rule")


# --------------------------------------------------------------------------------------------
#   The GDN scan's stop
# --------------------------------------------------------------------------------------------

def reference_fn(q, k, v, g, beta, initial_state = None, output_final_state = False, use_qk_l2norm_in_kernel = True):
    """chunk_gated_delta_rule's signature over upstream's reference recurrence (fp32 inputs, so the
    state stays fp32 between sub-ranges, as the chunk kernels keep it)"""
    o, s = GDR.torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state, True, use_qk_l2norm_in_kernel)
    return o, (s if output_final_state else None)


def scan_inputs(T, heads = 2, dk = 8, dv = 8, seed = 0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn((1, T, heads, dk), generator = gen)
    k = torch.randn((1, T, heads, dk), generator = gen)
    v = torch.randn((1, T, heads, dv), generator = gen)
    g = -torch.rand((1, T, heads), generator = gen)
    beta = torch.rand((1, T, heads), generator = gen)
    state = torch.randn((1, heads, dk, dv), generator = gen)
    return q, k, v, g, beta, state


def sliced(inputs, t0, t1):
    q, k, v, g, beta, _ = inputs
    return q[:, t0:t1], k[:, t0:t1], v[:, t0:t1], g[:, t0:t1], beta[:, t0:t1]


@pytest.mark.parametrize("T, sub, split", [(256, 64, 128), (300, 64, 256), (300, 100, 64), (300, 0, 192), (520, 128, 512)])
def test_the_scan_hands_over_the_state_a_pass_ending_there_returns(monkeypatch, T, sub, split):
    monkeypatch.setattr(GDR, "_scan_sub_range", sub)
    inputs = scan_inputs(T)
    state = inputs[-1]

    calls = []
    def fn(q, *a, **k):
        calls.append(q.shape[1])
        return reference_fn(q, *a, **k)

    # A pass of `split` tokens, and one over the rest from its state: today's two passes
    a_out, a_state = GDR._chunked_scan(fn, *sliced(inputs, 0, split), state.clone(), True)
    a_calls = list(calls)
    b_out, b_state = GDR._chunked_scan(fn, *sliced(inputs, split, T), a_state.clone(), True)
    calls.clear()

    handed = []
    out, final = GDR._chunked_scan(fn, *sliced(inputs, 0, T), state.clone(), True,
                                   split_at = split, on_split = lambda s: handed.append(s.clone()))
    assert len(handed) == 1
    assert torch.equal(handed[0], a_state), "the state at the split is not the final state of a pass ending there"
    assert calls[:len(a_calls)] == a_calls, "the sub-ranges before the split are not those of a pass ending there"
    assert torch.equal(out, torch.cat((a_out, b_out), dim = 1))
    assert torch.equal(final, b_state)


@pytest.mark.parametrize("split", [None, 0, 300, 400])
def test_a_split_outside_the_pass_changes_nothing(monkeypatch, split):
    monkeypatch.setattr(GDR, "_scan_sub_range", 128)
    inputs = scan_inputs(300)
    out0, s0 = GDR._chunked_scan(reference_fn, *sliced(inputs, 0, 300), inputs[-1].clone(), True)
    handed = []
    out1, s1 = GDR._chunked_scan(reference_fn, *sliced(inputs, 0, 300), inputs[-1].clone(), True,
                                 split_at = split, on_split = handed.append)
    assert not handed
    assert torch.equal(out0, out1) and torch.equal(s0, s1)


def test_the_scan_ranges_up_to_the_split_are_a_shorter_pass_s():
    assert GDR._scan_ranges(5000, 2048, None) == [(0, 2048), (2048, 4096), (4096, 5000)]
    assert GDR._scan_ranges(5000, 2048, 3840) == [(0, 2048), (2048, 3840), (3840, 5000)]
    assert GDR._scan_ranges(5000, 2048, 4096) == [(0, 2048), (2048, 4096), (4096, 5000)]
    assert GDR._scan_ranges(5000, 2048, 256) == [(0, 256), (256, 2304), (2304, 4352), (4352, 5000)]


# --------------------------------------------------------------------------------------------
#   Stand-in recurrent layers
# --------------------------------------------------------------------------------------------

WIN = 4


def step(scan: torch.Tensor, win: torch.Tensor, t: int):
    """One token through the stand-in layer: a decaying scan over the ids and a window of the last ids"""
    return scan * 0.75 + t, torch.cat((win[1:], torch.tensor([float(t)], dtype = win.dtype)))


def reference_state(ids: list) -> tuple:
    scan, win = torch.zeros(1, dtype = torch.float64), torch.zeros(WIN, dtype = torch.float64)
    for t in ids:
        scan, win = step(scan, win, t)
    return scan, win


class RecLayer:
    """A layer's state per slot, stashed as GDNLayerState stashes (a tuple of host copies)"""

    def __init__(self, slots):
        self.scan = torch.zeros((slots, 1), dtype = torch.float64)
        self.win = torch.zeros((slots, WIN), dtype = torch.float64)

    def clear(self, slot):
        self.scan[slot].zero_()
        self.win[slot].zero_()

    def get_checkpoint_size(self):
        return 8 * (1 + WIN)

    def stash(self, slot, position = 0, parity = 0):
        return (REC.host_copy(self.scan[slot]), REC.host_copy(self.win[slot]))

    def unstash(self, slot, stashed, position = 0, parity = 0):
        self.scan[slot].copy_(stashed[0])
        self.win[slot].copy_(stashed[1])


class RecState:
    """One sequence's state across the cache's layers (GDNState's interface, what the generator uses)"""

    pass_reads_position = False

    def __init__(self, cache, slot, position, stashed = None):
        self.cache = cache
        self.slot = slot
        self.position = position
        self.last_history = 0
        if stashed is None:
            for l in cache.recurrent_layers.values():
                l.clear(slot)
        else:
            assert stashed["position"] == position
            for k, l in cache.recurrent_layers.items():
                l.unstash(slot, stashed[k])
        self.checkpoint_size = sum(l.get_checkpoint_size() for l in cache.recurrent_layers.values())

    def stash(self):
        stashed = {"position": self.position, "checkpoint_size": self.checkpoint_size}
        for k, l in self.cache.recurrent_layers.items():
            stashed[k] = l.stash(self.slot, self.position)
        return stashed

    def free(self):
        self.cache.release_state(self)

    def rewind(self, num_tokens):
        self.position -= num_tokens

    def rollback_capacity(self):
        return 0

    def reset(self):
        self.position = 0

    def post_advance(self):
        pass


REC_MODULES = (1, 3)   # one on each pretend card


class RecKV(PR.KV):
    """test_page_revert's K/V stand-in with the recurrent layers of the stand-in modules"""

    def __init__(self, n_pages, slots = 4):
        super().__init__(n_pages)
        self.recurrent_layers = {(i, 0): RecLayer(slots) for i in REC_MODULES}
        self.free_slots = list(range(slots))

    def get_all_recurrent_layers(self):
        return self.recurrent_layers

    def get_new_state(self):
        return RecState(self, self.free_slots.pop(0), 0)

    def new_from_stashed(self, stashed, position):
        return RecState(self, self.free_slots.pop(0), position, stashed)

    def release_state(self, state):
        self.free_slots.append(state.slot)


class RecModule(PR.Module):
    """Writes its K/V like the other stand-in modules and advances its recurrent state token by token,
    copying it at a capture's boundary the way GatedDeltaNet and PLELayer do"""

    captures_recurrent_state = True

    def __init__(self, model, idx, device, captures = True):
        super().__init__(model, idx, device)
        self.captures = captures

    def forward(self, x, params):
        x = super().forward(x, params)
        rs = params.get("recurrent_states")
        if rs:
            st = rs[0]
            layer = params["cache"].recurrent_layers[(self.idx, 0)]
            ids = params["input_ids"][0].tolist()
            cap = params.get("recurrent_capture") if self.captures else None
            if cap is not None and not cap.wants(1, len(ids)):
                cap = None
            scan, win = layer.scan[st.slot].clone(), layer.win[st.slot].clone()
            for i, t in enumerate(ids):
                if cap is not None and i == cap.offset:
                    cap.take(layer, (scan.clone(), win.clone()))
                scan, win = step(scan, win, t)
            layer.scan[st.slot] = scan
            layer.win[st.slot] = win
        return x


class RecModel(PR.Model):

    def __init__(self, captures = True):
        super().__init__()
        self.caps = {"recurrent_states": True}
        for i in REC_MODULES:
            self.modules[i] = RecModule(self, i, f"cuda:{i // 2}", captures = captures or i == REC_MODULES[0])
        self.fwd_modules = [(m, 0, i) for i, m in enumerate(self.modules)]
        self.passes = []

    def get_recurrent_layers(self):
        return [self.modules[i] for i in REC_MODULES]

    def prepare_inputs(self, input_ids, params):
        self.passes.append(input_ids.shape[-1])
        return super().prepare_inputs(input_ids, params)

    def forward(self, input_ids, params):
        x = super().forward(input_ids, params)
        RUTIL.advance_recurrent_states(input_ids, params, self)
        return x

    prefill = forward


def make_generator(capture: bool, n_pages = 12, max_chunk_size = 2048, all_layers_capture = True):
    g = PR.make_generator(n_pages, max_chunk_size)
    g.model = RecModel(captures = all_layers_capture)
    g.cache = RecKV(n_pages)
    g.draft_cache = PR.KV(n_pages)
    g.pagetable = PT.PageTable(g, g.cache)
    g.max_total_tokens = g.cache.max_num_tokens
    g.recurrent_cache = REC.RecurrentCache(g.model, 1 << 30)
    g.recurrent_cache.pagetable = g.pagetable
    g.target_recurrent = True
    g.draft_recurrent = False
    g.recurrent_checkpoint_interval = 4096
    g.recurrent_checkpoint_interval_pp = 32768
    g.tail_checkpoint_interval = 0
    g.tail_stats = {"stashes": 0, "replaced": 0, "t_stash": 0.0,
                    "disk_withdrawn": 0, "disk_dropped": 0, "disk_left": 0}
    g.recurrent_checkpoint_in_pass = capture
    g.recurrent_capture_stats = {"captured": 0, "incomplete": 0}
    g._capture_warned = False
    return g


@pytest.fixture
def generators():
    made = []

    def make(*a, **k):
        g = make_generator(*a, **k)
        made.append(g)
        return g

    yield make
    for g in made:
        g.close()


# --------------------------------------------------------------------------------------------
#   Conversations
# --------------------------------------------------------------------------------------------

PROMPT = [100 + (i * 7) % 900 for i in range(3 * PAGE + 100)]   # three complete pages and a tail of 100
WHERE = ["inline", "thread", "pipeline"]


def ingest(g, ids) -> object:
    job = PR.make_job(g, ids)
    PR.ingest(g, job)
    return job


def checkpoint(g, job, pos):
    """The checkpoint stashed for the page that ends at pos, and its key"""
    seq = job.sequences[0]
    key = seq.allocated_pages[pos // PAGE - 1].phash
    return key, g.recurrent_cache.get(key)


def assert_holds(stashed, ids):
    """A stash (PairedState format) holds, for every stand-in layer, the state after these ids"""
    scan, win = reference_state(ids)
    target = stashed["target"]
    for k in ((i, 0) for i in REC_MODULES):
        s, w = target[k]
        assert torch.equal(s, scan) and torch.equal(w, win), f"layer {k}: not the state after {len(ids)} tokens"


def structure(obj):
    if isinstance(obj, torch.Tensor):
        return ("tensor", tuple(obj.shape), obj.dtype)
    if isinstance(obj, tuple):
        return ("tuple", tuple(structure(o) for o in obj))
    if isinstance(obj, dict):
        return ("dict", tuple((k, structure(v)) for k, v in obj.items()))
    return obj


def test_switched_off_the_chunk_ends_at_the_last_page_boundary(generators):
    g = generators(capture = False)
    job = ingest(g, PROMPT)
    lpb = (len(PROMPT) - 1) // PAGE * PAGE
    assert g.model.passes == [lpb, len(PROMPT) - 1 - lpb]
    key, stashed = checkpoint(g, job, lpb)
    assert stashed["position"] == lpb
    assert_holds(stashed, PROMPT[:lpb])
    assert g.recurrent_capture_stats == {"captured": 0, "incomplete": 0}


@pytest.mark.parametrize("where", WHERE)
def test_switched_on_one_pass_and_the_checkpoint_of_the_shorter_pass(generators, where):
    ref = generators(capture = False)
    PR.switch(ref, where)
    ref_job = ingest(ref, PROMPT)
    lpb = (len(PROMPT) - 1) // PAGE * PAGE
    ref_key, ref_stash = checkpoint(ref, ref_job, lpb)

    g = generators(capture = True)
    PR.switch(g, where)
    job = ingest(g, PROMPT)
    assert g.model.passes == [len(PROMPT) - 1], "the chunk was cut at the last page boundary"
    key, stashed = checkpoint(g, job, lpb)
    assert key == ref_key and stashed is not None
    assert stashed["position"] == lpb and stashed["checkpoint_size"] == ref_stash["checkpoint_size"]
    assert structure(stashed) == structure(ref_stash), "not the format of a regular stash"
    assert_holds(stashed, PROMPT[:lpb])
    for k in ref_stash["target"]:
        if k in ("position", "checkpoint_size"):
            assert stashed["target"][k] == ref_stash["target"][k]
            continue
        assert all(torch.equal(a, b) for a, b in zip(stashed["target"][k], ref_stash["target"][k]))
    assert g.recurrent_capture_stats == {"captured": 1, "incomplete": 0}
    assert job.last_recurrent_checkpoint_pos == lpb
    # The live state goes on past the boundary: after the whole prompt but its last token
    layer = g.cache.recurrent_layers[(REC_MODULES[0], 0)]
    scan, win = reference_state(PROMPT[:-1])
    assert torch.equal(layer.scan[job.recurrent_state.target.slot], scan)
    assert job.recurrent_state.position == len(PROMPT) - 1


@pytest.mark.parametrize("capture", [False, True])
def test_the_next_turn_resumes_from_the_last_page_checkpoint(generators, capture):
    g = generators(capture = capture)
    job = ingest(g, PROMPT)
    g.cancel(job)
    lpb = (len(PROMPT) - 1) // PAGE * PAGE
    g.model.passes.clear()
    nxt = PROMPT + [7] * 300
    job2 = ingest(g, nxt)
    assert job2.cached_pages == lpb // PAGE
    # The ingestion starts at the checkpoint, from its state
    assert sum(g.model.passes) == len(nxt) - 1 - lpb
    layer = g.cache.recurrent_layers[(REC_MODULES[1], 0)]
    scan, win = reference_state(nxt[:-1])
    assert torch.equal(layer.scan[job2.recurrent_state.target.slot], scan)
    assert torch.equal(layer.win[job2.recurrent_state.target.slot], win)


def test_a_chunk_that_ends_at_the_boundary_needs_no_capture(generators):
    g = generators(capture = True)
    ids = PROMPT[:3 * PAGE + 1]      # the last page boundary is where the chunk ends
    job = ingest(g, ids)
    assert g.model.passes == [3 * PAGE]
    _, stashed = checkpoint(g, job, 3 * PAGE)
    assert_holds(stashed, ids[:3 * PAGE])
    assert g.recurrent_capture_stats == {"captured": 0, "incomplete": 0}


def test_the_boundary_in_a_later_chunk(generators):
    g = generators(capture = True, max_chunk_size = 2 * PAGE)
    job = ingest(g, PROMPT)
    lpb = (len(PROMPT) - 1) // PAGE * PAGE
    assert g.model.passes == [2 * PAGE, len(PROMPT) - 1 - 2 * PAGE]
    _, stashed = checkpoint(g, job, lpb)
    assert_holds(stashed, PROMPT[:lpb])
    assert g.recurrent_capture_stats["captured"] == 1


def test_a_capture_a_layer_is_missing_from_takes_no_checkpoint(generators, capsys):
    g = generators(capture = True, all_layers_capture = False)
    job = ingest(g, PROMPT)
    lpb = (len(PROMPT) - 1) // PAGE * PAGE
    assert g.model.passes == [len(PROMPT) - 1]
    _, stashed = checkpoint(g, job, lpb)
    assert stashed is None
    assert g.recurrent_capture_stats == {"captured": 0, "incomplete": 1}
    assert "lacks a recurrent layer" in capsys.readouterr().out
    # The next turn finds no checkpoint there and ingests from the start
    g.cancel(job)
    job2 = ingest(g, PROMPT + [7] * 10)
    assert job2.cached_pages == 0


def test_only_a_plain_job_takes_its_checkpoint_in_the_pass():
    gen = types.SimpleNamespace(recurrent_checkpoint_in_pass = True)
    side = object()
    pair = types.SimpleNamespace(target = side, draft = None)

    def job(**k):
        j = types.SimpleNamespace(generator = gen, sequences = [object()], embeddings = [], recurrent_state = pair)
        j.__dict__.update(k)
        return j

    assert JOB.Job.can_capture_recurrent(job())
    assert not JOB.Job.can_capture_recurrent(job(embeddings = [object()]))
    assert not JOB.Job.can_capture_recurrent(job(sequences = [object(), object()]))
    assert not JOB.Job.can_capture_recurrent(job(recurrent_state = types.SimpleNamespace(target = side, draft = side)))
    assert not JOB.Job.can_capture_recurrent(job(recurrent_state = None))
    assert not JOB.Job.can_capture_recurrent(job(generator = types.SimpleNamespace()))


def test_the_generator_says_why_it_is_off():
    g = GEN.Generator.__new__(GEN.Generator)
    g.model = RecModel()
    g.recurrent_cache = None
    g.draft_recurrent = False
    assert "no recurrent state" in g._capture_unavailable()
    g.recurrent_cache = object()
    g.model.loaded_tp = True
    assert "tensor parallelism" in g._capture_unavailable()
    g.model.loaded_tp = False
    g.draft_recurrent = True
    assert "draft model" in g._capture_unavailable()
    g.draft_recurrent = False
    assert g._capture_unavailable() is None

    class Mamba(PR.Module):
        pass
    g.model.modules[3] = Mamba(g.model, 3, "cuda:1")
    assert "Mamba" in g._capture_unavailable()


def test_every_recurrent_module_of_the_production_models_copies_its_state():
    """The modules Qwen3.8-Flash-Next keeps recurrent state in (GDN layers, PLE layers) say they copy
    it; Mamba2 does not, and the switch stays off for a model with it"""
    src = {}
    for name in ("gated_delta_net", "ple", "mamba2"):
        with open(os.path.join(REPO, "exllamav3", "modules", f"{name}.py")) as f:
            src[name] = f.read()
    assert "class GatedDeltaNet(Module):\n\n    # Fork: copies its state" in src["gated_delta_net"]
    assert "class PLELayer(Module):\n\n    # Fork: copies its state" in src["ple"]
    assert "captures_recurrent_state" not in src["mamba2"]
