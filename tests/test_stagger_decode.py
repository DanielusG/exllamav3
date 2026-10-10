"""
Staggered decode (Generator.stagger_decode, Generator.iterate_staggered) against a toy model.

No GPU or model is required. The generator's real draft step, decode step and scheduler run over
a toy recurrent model whose modules sit on two pretend cards, with stub jobs that keep pages and
states the way Job does. The toy model makes every token depend on everything a pass could get
wrong: each layer folds the token, its position, the job's first page and its own carried state
into the hidden value, and the layers on the second card read their inputs from the generator's
pinned staging buffers only when they run (as an asynchronous upload would). A job therefore
yields the same tokens in a staggered run as in a regular one only if its pass saw its own
inputs, in order, exactly once.

When the compiled extension is not installed the two source files under test are loaded on
their own, with the rest of the package stubbed.
"""

import importlib.util
import os
import sys
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = 64


# --------------------------------------------------------------------------------------------
#   Loading
# --------------------------------------------------------------------------------------------

def _load_real():
    sys.path.insert(0, REPO)
    try:
        from exllamav3.generator import generator as gen_mod
        from exllamav3.model import forward_cursor as fc_mod
        from exllamav3.cache import recurrent_util as ru_mod
        return gen_mod, fc_mod, ru_mod
    finally:
        sys.path.remove(REPO)


def _load_stubbed():
    pkg = "exl3_stagger_stub"
    src = os.path.join(REPO, "exllamav3")

    def stub(name, **attrs):
        full = f"{pkg}.{name}" if name else pkg
        m = types.ModuleType(full)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[full] = m
        return m

    def real(name):
        full = f"{pkg}.{name}"
        path = os.path.join(src, *name.split(".")) + ".py"
        spec = importlib.util.spec_from_file_location(full, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        return m

    class _Any:
        def __init__(self, *a, **k):
            pass

    stub("")
    for p in ("model", "cache", "tokenizer", "util", "generator", "modules"):
        stub(p)
    sys.modules[f"{pkg}.tokenizer"].MMEmbedding = _Any
    sys.modules[f"{pkg}.util"].cuda_sync_active = lambda: None
    sys.modules[f"{pkg}.util"].profile_opt = lambda *a, **k: None
    stub("constants", PAGE_SIZE = 256)
    stub("model.model", Model = _Any)
    stub("cache.cache", Cache = _Any)
    stub("cache.recurrent", RecurrentCache = _Any, host_pool = _Any(), mp_host_pool_release = None)
    stub("tokenizer.tokenizer", Tokenizer = _Any)
    stub("util.memory", malloc_trim = lambda: None)
    stub("generator.pagetable", PageTable = _Any, is_content_hash = lambda h: False)
    stub("generator.cpu_cache", CPUPageCache = _Any)
    stub("generator.draft_confidence", DraftConfidenceCalibrator = _Any)
    stub("generator.job", Job = _Any)
    stub("generator.filter", Filter = _Any)
    stub("generator.sampler", Sampler = _Any)
    stub("generator.visualizer", CacheVisualizer = _Any)
    ru_mod = real("cache.recurrent_util")
    fc_mod = real("model.forward_cursor")
    gen_mod = real("generator.generator")
    return gen_mod, fc_mod, ru_mod


try:
    GEN, FC, RU = _load_real()
except Exception:
    GEN, FC, RU = _load_stubbed()

PAGE = GEN.PAGE_SIZE


class _Cuda:
    def __init__(self, real_cuda):
        self._real = real_cuda

    def synchronize(self, *a, **k):
        pass

    def __getattr__(self, name):
        return getattr(self._real, name)


class _Torch:
    """torch as the generator module sees it in these tests: host tensors only, so pinned
    allocations are plain ones and there is no stream to synchronize."""

    def __init__(self):
        self.cuda = _Cuda(torch.cuda)

    def zeros(self, *a, pin_memory = False, **k):
        return torch.zeros(*a, **k)

    def empty(self, *a, pin_memory = False, **k):
        return torch.zeros(*a, **k)

    def __getattr__(self, name):
        return getattr(torch, name)


@pytest.fixture(autouse = True)
def _host_torch(monkeypatch):
    monkeypatch.setattr(GEN, "torch", _Torch())
    monkeypatch.setattr(GEN, "cuda_sync_active", lambda: None)


# --------------------------------------------------------------------------------------------
#   Toy model
# --------------------------------------------------------------------------------------------

def mix(*vals):
    acc = 1469598103
    for v in vals:
        acc = (acc * 1103515245 + int(v) * 2654435761 + 12345) % 2147483647
    return acc


class ToyState:
    """Recurrent state of one job: a value per layer, plus the values after each token of the
    last pass (what a rewind of rejected draft positions goes back to)."""

    def __init__(self, slot, layers):
        self.slot = slot
        self.position = 0
        self.last_history = 0
        self.acc = {l: 0 for l in layers}
        self.hist = {l: [0] for l in layers}
        self.freed = False

    def post_advance(self):
        pass

    def rewind(self, n):
        if n:
            for l in self.acc:
                self.acc[l] = self.hist[l][-(n + 1)]
                self.hist[l] = self.hist[l][:-n]
            self.position -= n

    def free(self):
        self.freed = True


def layer_step(state, layer, tokens, pos0, page0):
    """Fold a row of tokens into one layer's carried value; returns the value after each."""
    acc = state.acc[layer]
    hist = [acc]
    for i, t in enumerate(tokens):
        acc = mix(acc, t, layer, pos0 + i, page0)
        hist.append(acc)
    state.acc[layer] = acc
    state.hist[layer] = hist
    return hist[1:]


class ToyModule:
    def __init__(self, model, idx, device):
        self.model = model
        self.idx = idx
        self.device = device
        self.caps = {}

    def prepare_for_device(self, x, params):
        return x

    def rows(self, params):
        return [id(s) for s in params["recurrent_states"]]


class ToyEmbed(ToyModule):
    def forward(self, x, params):
        if "input_ids" not in params:
            params["input_ids"] = x
        self.model.log.append((self.idx, self.rows(params)))
        return x.clone()


class ToyLayer(ToyModule):
    def forward(self, x, params):
        self.model.log.append((self.idx, self.rows(params)))
        # Read at the time the layer runs, like a device copy made when the pass gets here
        ids = params["input_ids"]
        seqlens = params["cache_seqlens"]
        table = params["block_table"]
        out = x.clone()
        for r, state in enumerate(params["recurrent_states"]):
            assert not state.freed, "a layer ran on a released state"
            vals = layer_step(state, self.idx, ids[r].tolist(), int(seqlens[r]), int(table[r, 0]))
            for i, v in enumerate(vals):
                out[r, i] = mix(out[r, i], v) % 1000003
        return out


class ToyHead(ToyModule):
    def __init__(self, model, idx, device):
        super().__init__(model, idx, device)
        self.caps = {"logits_output": True}

    def forward(self, x, params):
        self.model.log.append((self.idx, self.rows(params)))
        ids = params["input_ids"]
        rows, width = x.shape
        logits = torch.zeros((rows, width, VOCAB))
        for r in range(rows):
            for i in range(width):
                h = int(x[r, i])
                # One time in three the next token is the one the toy drafter guesses
                tok = (int(ids[r, i]) + 1) % VOCAB if h % 3 == 0 else h % VOCAB
                logits[r, i, tok] = 1.0
        params["export_states"] = [x.float().unsqueeze(-1)]
        return logits


class ToyHost:
    def __init__(self):
        self.passes = 0

    def begin_pass(self):
        self.passes += 1


class ToyModel:
    """An embedding and three layers on the first card, three layers and the head on the second."""

    def __init__(self):
        self.log = []
        self.loaded_tp = False
        self.caps = {}
        self.host = ToyHost()
        self.config = types.SimpleNamespace(moe_cpu_hosts = {"cuda:0": self.host}, infer_params = None)
        self.modules = [ToyEmbed(self, 0, "cuda:0")]
        self.modules += [ToyLayer(self, i, "cuda:0") for i in (1, 2, 3)]
        self.modules += [ToyLayer(self, i, "cuda:1") for i in (4, 5, 6)]
        self.modules += [ToyHead(self, 7, "cuda:1")]
        self.fwd_modules = [(m, 0, i) for i, m in enumerate(self.modules)]
        self.logit_layer_idx = len(self.modules) - 1
        self.layers = [m.idx for m in self.modules if isinstance(m, ToyLayer)]
        self._get_prefetch_layers = []

    def prepare_inputs(self, input_ids, params):
        return input_ids

    def prefetch_tokens(self, ids):
        pass

    def forward(self, input_ids, params):
        """The whole pass at once, written out here rather than through the cursor."""
        x = self.prepare_inputs(input_ids, params)
        self.host.begin_pass()
        for module, instance, _ in self.fwd_modules:
            params["layer_instance"] = instance
            x = module.forward(module.prepare_for_device(x, params), params)
        RU.advance_recurrent_states(input_ids, params, self)
        return x


class ToyDraft:
    """MTP-like drafter: always guesses the token after the one it is given."""

    def __init__(self):
        self.caps = {"mtp_draft": True}
        self.draft_verifier_params = {"export_state_norm_keys": ("mixer",)}
        self.prefills = 0

    def forward(self, batch_ids, params):
        assert params["target_hidden"].shape[0] == batch_ids.shape[0]
        return batch_ids.clone().float().view(-1, 1, 1)

    def sample_from_state(self, state, params):
        return (state.long().view(-1, 1) + 1) % VOCAB

    def prefill(self, ids, params):
        self.prefills += 1


# --------------------------------------------------------------------------------------------
#   Stub jobs
# --------------------------------------------------------------------------------------------

class ToyPage:
    def __init__(self):
        self.kv_position = 0
        self.sequence = torch.zeros((1, PAGE), dtype = torch.long)
        self.ref_count = 1
        self.can_revert = True
        self.phash = None


class ToyIds:
    def __init__(self, ids):
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def torch_slice(self, a, b):
        return torch.tensor([self.ids[a:b]], dtype = torch.long)

    def torch(self):
        return torch.tensor([self.ids], dtype = torch.long)

    def append(self, t):
        self.ids.append(int(t.item()))


class ToySeq:
    def __init__(self, ids, first_page, n_pages):
        self.sequence_ids = ToyIds(ids)
        self.input_ids = ToyIds(ids)
        self.kv_position = 0
        self.allocated_pages = [ToyPage() for _ in range(n_pages)]
        self.block_index_tensor = torch.arange(first_page, first_page + n_pages, dtype = torch.int32).view(1, -1)
        self.prefill_complete = False


class ToyJob:
    """What Generator needs from a Job, with the invariants a staggered step must not break
    turned into assertions: nothing but its own pass touches a job between the staging of its
    inputs and its sampling."""

    def __init__(self, model, name, prompt, max_new, prefill_iters = 1, rq_at = 0):
        self.model = model
        self.name = name
        self.prompt = list(prompt)
        self.max_new = max_new
        self.prefill_iters = prefill_iters
        self.rq_at = rq_at
        # A job's own pages, the same in every run
        self.first_page = 1000 * ord(name)
        self.out = []
        self.identifier = None
        self.skips = 0
        self.max_skips = 4
        self.embeddings = []
        self.filters = []
        self.filter_futures = []
        self.logit_masks = []
        self.alt_rope_offset = 0
        self.time_first_token = None
        self.checkpoint_rewound = False
        self.accepted_draft_tokens = 0
        self.rejected_draft_tokens = 0
        self.draft_stats = []
        self.recurrent_state = None
        self.mid_pass = False
        self.cancelled = False
        self.droppable = False      # clear_queue() may release a job in the middle of its pass
        self.requeues = 0
        self._reset(self.prompt)

    def _reset(self, ids):
        self.sequences = [ToySeq(ids, self.first_page, (len(ids) + self.max_new) // PAGE + 2)]
        self.new_tokens = 0
        self.mtp_last_hidden = None
        self.prefill_left = self.prefill_iters

    # Queueing
    def prepare_for_queue(self, generator, serial_number, rq = False):
        self.generator = generator
        self.serial_number = serial_number

    def current_new_pages_required(self):
        return 1

    def activate(self):
        pass

    def allocate_pages(self):
        self.recurrent_state = ToyState(self.serial_number, self.model.layers)

    def deallocate_pages(self):
        assert not self.mid_pass or self.droppable, f"{self.name}: pages released during its pass"
        if self.recurrent_state is not None:
            self.recurrent_state.free()
            self.recurrent_state = None

    # State
    def is_prefill_done(self):
        return all(seq.kv_position == len(seq.sequence_ids) - 1 for seq in self.sequences)

    def get_max_seq_len(self):
        if not self.is_prefill_done():
            return 0
        return max(len(seq.sequence_ids) for seq in self.sequences)

    def is_checkpoint_boundary(self, override_interval = None):
        return False

    def maybe_stash_recurrent(self, cache, interval = None):
        assert not self.mid_pass, f"{self.name}: checkpoint offered during its pass"

    def prefill(self, results):
        assert not self.mid_pass, f"{self.name}: prefill called during its pass"
        if self.is_prefill_done() and self.prefill_left == 0:
            return
        self.prefill_left -= 1
        if self.prefill_left > 0:
            return
        seq = self.sequences[0]
        n = len(seq.sequence_ids) - 1
        state = self.recurrent_state
        for layer in self.model.layers:
            layer_step(state, layer, seq.sequence_ids.ids[:n], 0, self.first_page)
        state.position = n
        left = n
        for page in seq.allocated_pages:
            page.kv_position = min(left, PAGE)
            left -= page.kv_position
        seq.kv_position = n
        seq.prefill_complete = True
        results.append({"job": self, "stage": "prefill", "eos": False, "serial": self.serial_number})

    def get_input_ids_list(self, draft_tokens = None, idx = 0, add_to_cache = False):
        input_ids_list = []
        for seq in self.sequences:
            ids = seq.sequence_ids.torch_slice(seq.kv_position, None)
            if draft_tokens is not None:
                ids = torch.cat((ids, draft_tokens[idx:idx + 1, :]), dim = -1)
            input_ids_list.append(ids)
            if add_to_cache:
                assert not self.mid_pass, f"{self.name}: issued twice"
                assert self.recurrent_state.position == seq.kv_position, f"{self.name}: state out of step"
                assert sum(p.kv_position for p in seq.allocated_pages) == seq.kv_position, \
                    f"{self.name}: pages out of step"
                self.mid_pass = True
                tokens_to_add = ids.shape[-1]
                skvp = seq.kv_position
                while tokens_to_add:
                    page = seq.allocated_pages[skvp // PAGE]
                    tokens_page = min(tokens_to_add, PAGE - page.kv_position)
                    page.sequence[:, page.kv_position:page.kv_position + tokens_page] = ids[:, :tokens_page]
                    page.kv_position += tokens_page
                    skvp += tokens_page
                    ids = ids[:, tokens_page:]
                    tokens_to_add -= tokens_page
        return input_ids_list

    # Sampling
    def prepare_logit_mask(self):
        pass

    def prepare_sampling_past_ids(self):
        pass

    def receive_logits(self, logits):
        self.mid_pass = False
        return torch.argmax(logits, dim = -1).view(1, 1), None, None, None

    def receive_sample(self, logits, next_token, next_k_tokens, next_k_probs, next_prob, results):
        tok = int(next_token.item())
        seq = self.sequences[0]
        seq.sequence_ids.append(next_token)
        seq.kv_position += 1
        self.new_tokens += 1
        self.out.append(tok)
        eos = len(self.out) >= self.max_new
        rq = not eos and self.rq_at and self.new_tokens >= self.rq_at
        results.append({"job": self, "stage": "streaming", "eos": eos, "serial": self.serial_number, "token": tok})
        return eos, next_token, rq

    def prepare_for_requeue(self):
        self.requeues += 1
        self._reset(self.sequences[0].sequence_ids.ids)
        return self


class ToyPagetable:
    def __init__(self, gen):
        self.gen = gen
        self.defrags = 0

    def num_unreferenced_pages(self):
        return 1 << 30

    def defrag(self):
        assert self.gen._flight is None, "defrag with a pass in flight"
        self.defrags += 1


def make_generator(draft = True, stagger = False, num_draft_tokens = 1, max_batch_size = 4, min_jobs = 2):
    model = ToyModel()
    g = GEN.Generator.__new__(GEN.Generator)
    g.model = model
    g.cache = types.SimpleNamespace(initialized = True)
    g.pagetable = ToyPagetable(g)
    g.pending_jobs = []
    g.active_jobs = []
    g.job_serial = 0
    g.max_batch_size = max_batch_size
    g.recurrent_cache = types.SimpleNamespace()
    g.visualizer = None
    g.ngram_match_min = 0
    g.dflash_draft = False
    g.draft_calibrator = None
    g._draft_conf_round = None
    g.record_draft_stats = True
    g.filter_pool = None
    g.sample_pinned = None
    g.staging_buffers = {}
    if draft:
        g.draft_model = ToyDraft()
        g.draft_cache = types.SimpleNamespace(initialized = True)
        g.mtp_draft = True
        g.num_draft_tokens = num_draft_tokens
        g.draft_input_ids_pinned = torch.zeros((max_batch_size, 1), dtype = torch.long)
        g.draft_ids_pinned = torch.zeros((max_batch_size, num_draft_tokens), dtype = torch.long)
    else:
        g.draft_model = None
        g.draft_cache = None
        g.mtp_draft = False
        g.num_draft_tokens = 0
    g.stagger_decode = stagger
    g.stagger_interleave = 1
    g.stagger_min_jobs = min_jobs
    g.stagger_stats = {
        "steps": 0, "launched": 0, "sampled": 0, "solo": 0, "drains": 0,
        "t_launch": 0.0, "t_issue": 0.0, "t_sample": 0.0,
    }
    g._step_group = None
    g._flight = None
    g._stagger_run = None
    g._lanes = {}
    g._stagger_backlog = []
    g._stagger_split = None
    g.on_queue_drained = g.pagetable.defrag
    return g


PROMPTS = {
    "a": [3, 14, 15, 9, 2],
    "b": [6, 5, 35, 8],
    "c": [9, 7, 9, 3, 23, 8, 4],
    "d": [62, 6, 43],
}


def run(names, stagger, draft = True, num_draft_tokens = 1, lengths = None, script = None, min_jobs = 2, **job_kw):
    """Drive iterate() until every job is done. script(generator, jobs, n) runs before the n-th
    iteration (to add, cancel or switch). Returns (tokens per job, generator, jobs)."""
    g = make_generator(draft = draft, stagger = stagger, num_draft_tokens = num_draft_tokens, min_jobs = min_jobs)
    jobs = {}
    lengths = lengths or {}

    def add(name):
        job = ToyJob(g.model, name, PROMPTS[name], lengths.get(name, 60), **job_kw)
        jobs[name] = job
        g.enqueue(job)
        return job

    for name in names:
        add(name)
    seen = {}
    n = 0
    while g.num_remaining_jobs():
        if script is not None:
            script(g, jobs, n, add)
        for r in g.iterate():
            if r["stage"] == "streaming":
                seen.setdefault(r["job"].name, []).append(r["token"])
        n += 1
        assert n < 5000, "the generator does not finish"
    assert g._flight is None
    for name, job in jobs.items():
        if not job.cancelled:
            assert seen.get(name, []) == job.out, f"{name}: results differ from what the job sampled"
    return {name: job.out for name, job in jobs.items()}, g, jobs


# --------------------------------------------------------------------------------------------
#   Tests
# --------------------------------------------------------------------------------------------

def test_stage_bounds_of_the_toy_model():
    assert FC.cuda_stage_bounds(ToyModel()) == [4]


def test_a_cursor_issued_in_pieces_is_the_whole_pass():
    outs = []
    for pieces in (None, (4,), (1, 2, 3, 4, 5, 6, 7)):
        model = ToyModel()
        state = ToyState(0, model.layers)
        ids = torch.tensor([[5, 6, 7]], dtype = torch.long)
        params = {
            "recurrent_states": [state],
            "recurrent_history": True,
            "cache_seqlens": torch.tensor([11], dtype = torch.int32),
            "block_table": torch.tensor([[77]], dtype = torch.int32),
        }
        if pieces is None:
            out = model.forward(ids, params)
        else:
            cur = FC.ForwardCursor(model, ids, params)
            for stop in pieces:
                cur.run_to(stop)
                cur.hold()
            out = cur.finish()
            assert cur.pos == cur.end
            with pytest.raises(AssertionError):
                cur.finish()
        assert state.position == 3 and state.last_history == 2
        outs.append((out.argmax(-1).tolist(), dict(state.acc)))
    assert outs[0] == outs[1] == outs[2]


@pytest.mark.parametrize("draft", [True, False])
@pytest.mark.parametrize("names", ["ab", "abc", "abcd"])
def test_staggered_tokens_are_the_regular_ones(names, draft):
    lengths = {"a": 70, "b": 55, "c": 90, "d": 40}
    ref, g0, _ = run(names, stagger = False, draft = draft, lengths = lengths)
    out, g, jobs = run(names, stagger = True, draft = draft, lengths = lengths)
    assert out == ref
    assert all(len(out[n]) == lengths[n] for n in names)
    assert g0.stagger_stats["steps"] == 0
    assert g.stagger_stats["steps"] > 20
    assert g.stagger_stats["drains"] == 0
    if draft:
        # Both outcomes of a draft are exercised, in both runs alike
        acc = sum(j.accepted_draft_tokens for j in jobs.values())
        rej = sum(j.rejected_draft_tokens for j in jobs.values())
        assert acc > 5 and rej > 5


def test_two_draft_tokens():
    ref, _, _ = run("abcd", stagger = False, num_draft_tokens = 2)
    out, g, _ = run("abcd", stagger = True, num_draft_tokens = 2)
    assert out == ref and g.stagger_stats["steps"] > 20


def test_the_two_halves_are_issued_in_turn_second_card_first():
    _, g, jobs = run("ab", stagger = True, lengths = {"a": 40, "b": 40})
    log = g.model.log
    # Hand-overs from a second-card module of one pass to a first-card module of the other:
    # four per step when the halves are issued a module at a time, one when they are issued
    # one after the other
    turns = 0
    for (m0, r0), (m1, r1) in zip(log, log[1:]):
        if m0 >= 4 and m1 < 4 and r0 != r1:
            turns += 1
    assert turns > 100
    stats = g.stagger_stats
    assert stats["launched"] > 20 and stats["sampled"] == stats["launched"]
    # The worker is woken for every pass, and again when its second half is issued
    assert g.model.host.passes >= 2 * stats["launched"]


def test_groups_stay_even():
    seen = []
    orig = GEN.Generator._stagger_pick

    def pick(ready, n_flying):
        launch = orig(ready, n_flying)
        seen.append((len(launch), n_flying))
        return launch

    GEN.Generator._stagger_pick = staticmethod(pick)
    try:
        run("abcd", stagger = True, lengths = {n: 50 for n in "abcd"})
    finally:
        GEN.Generator._stagger_pick = staticmethod(orig)
    assert seen[0] == (2, 0)
    assert seen[1:10] == [(2, 2)] * 9
    assert GEN.Generator._stagger_pick([1, 2, 3], 1) == [1, 2]
    assert GEN.Generator._stagger_pick([1], 2) == [1]
    assert GEN.Generator._stagger_pick([], 2) == []
    assert GEN.Generator._stagger_pick([1, 2, 3], 0) == [1, 2]


def test_shared_staging_buffers_would_corrupt_a_pass():
    """The check on the test itself: with one lane for both groups the pass waiting at the
    second card reads the other group's inputs, and the toy model shows it."""
    ref, _, _ = run("ab", stagger = False)
    orig = GEN.Generator._stagger_scope

    def one_lane(self, lane, jobs):
        return orig(self, 0, jobs)

    GEN.Generator._stagger_scope = one_lane
    try:
        try:
            out, _, _ = run("ab", stagger = True)
        except AssertionError:
            out = None
    finally:
        GEN.Generator._stagger_scope = orig
    assert out != ref


def test_cancel_of_a_job_in_flight_finishes_its_group_first():
    lengths = {"a": 60, "b": 60, "c": 60, "d": 60}
    ref, _, _ = run("abcd", stagger = False, lengths = lengths)

    def script(g, jobs, n, add):
        if n == 12:
            flight = g._flight
            assert flight is not None and len(flight.jobs) == 2
            victim = flight.jobs[0]
            victim.cancelled = True
            g.cancel(victim)
            assert g._flight is None
            assert victim not in g.active_jobs and victim not in g.pending_jobs
            # The other job of the group was sampled by the drain; its tokens wait for iterate()
            assert any(r["job"] is flight.jobs[1] for r in g._stagger_backlog)
            assert all(r["job"] is not victim for r in g._stagger_backlog)

    out, g, jobs = run("abcd", stagger = True, lengths = lengths, script = script)
    assert g.stagger_stats["drains"] == 1
    for name, job in jobs.items():
        if job.cancelled:
            assert 0 < len(out[name]) < lengths[name]
            assert out[name] == ref[name][:len(out[name])]
        else:
            assert out[name] == ref[name]


def test_cancel_of_a_job_at_rest_leaves_the_flight_alone():
    ref, _, _ = run("abcd", stagger = False)

    def script(g, jobs, n, add):
        if n == 12:
            flight = g._flight
            victim = next(j for j in g.active_jobs if all(j is not f for f in flight.jobs))
            victim.cancelled = True
            g.cancel(victim)
            assert g._flight is flight

    out, g, jobs = run("abcd", stagger = True, script = script)
    assert g.stagger_stats["drains"] == 0
    for name, job in jobs.items():
        if not job.cancelled:
            assert out[name] == ref[name]


def test_a_job_arriving_sends_the_decode_back_to_one_batch_while_it_prefills():
    lengths = {"a": 80, "b": 80, "c": 50}

    def script(g, jobs, n, add):
        if n == 15:
            add("c")

    ref, _, _ = run("ab", stagger = False, lengths = lengths, script = script, prefill_iters = 4)
    out, g, jobs = run("ab", stagger = True, lengths = lengths, script = script, prefill_iters = 4)
    assert out == ref
    assert len(out["c"]) == 50
    # The group in flight when the job arrived was completed on its own
    assert g.stagger_stats["solo"] >= 1


def test_switching_off_completes_the_flight_and_goes_on_regularly():
    ref, _, _ = run("abc", stagger = False)

    def script(g, jobs, n, add):
        if n == 10:
            g.stagger_decode = False
        if n == 30:
            assert g._flight is None
            g.stagger_decode = True

    out, g, _ = run("abc", stagger = True, script = script)
    assert out == ref


def test_requeue_inside_a_staggered_run():
    ref, _, _ = run("abc", stagger = False, rq_at = 17)
    out, g, jobs = run("abc", stagger = True, rq_at = 17)
    assert out == ref
    assert g.stagger_stats["steps"] > 20
    assert all(job.requeues == 3 for job in jobs.values())


def test_clear_queue_drops_a_flight():
    g = make_generator(stagger = True)
    jobs = [ToyJob(g.model, n, PROMPTS[n], 50) for n in "ab"]
    for j in jobs:
        g.enqueue(j)
    for _ in range(8):
        g.iterate()
    assert g._flight is not None
    for j in jobs:
        j.droppable = True
    g.clear_queue()
    assert g._flight is None and not g.num_remaining_jobs()
    assert g.iterate() == []


def test_not_available_falls_back_to_regular_steps():
    ref, _, _ = run("ab", stagger = False)
    g = make_generator(stagger = True)
    for m in g.model.modules:
        m.device = "cuda:0"
    assert g._stagger_plan() is None
    jobs = [ToyJob(g.model, n, PROMPTS[n], 60) for n in "ab"]
    for j in jobs:
        g.enqueue(j)
    while g.num_remaining_jobs():
        g.iterate()
    assert {j.name: j.out for j in jobs} == ref
    assert g.stagger_stats["steps"] == 0


def test_one_job_is_never_staggered():
    ref, g0, _ = run("a", stagger = False)
    out, g, _ = run("a", stagger = True)
    assert out == ref and g.stagger_stats["steps"] == 0


def test_below_the_threshold_the_decode_stays_one_batch():
    ref, _, _ = run("ab", stagger = False)
    out, g, _ = run("ab", stagger = True, min_jobs = 3)
    assert out == ref and g.stagger_stats["steps"] == 0


def test_staggering_ends_when_the_jobs_fall_below_the_threshold():
    # `a` leaves first, so the two jobs left are one in flight and one at rest: the case in
    # which the scheduler itself must refuse to issue (with `c` leaving first the two left are
    # in flight together and the decode falls back on its own)
    lengths = {"a": 20, "b": 70, "c": 55}
    ref, _, _ = run("abc", stagger = False, lengths = lengths)
    seen = {}

    def script(g, jobs, n, add):
        # The first iteration that starts with two jobs left: at most a pass is still in flight
        if "steps" not in seen and g.num_remaining_jobs() <= 2:
            seen["steps"] = g.stagger_stats["steps"]
            seen["flying"] = g._flight is not None

    out, g, _ = run("abc", stagger = True, lengths = lengths, min_jobs = 3, script = script)
    assert out == ref
    assert seen["steps"] > 5 and seen["flying"]         # staggered with three
    # One more call completes the pass in flight alone, then regular steps for the two left
    assert g.stagger_stats["steps"] == seen["steps"] + 1
    assert g.stagger_stats["solo"] == 1


def test_staggering_starts_again_when_a_job_joins():
    lengths = {"a": 90, "b": 90, "c": 40}
    seen = {}

    def script(g, jobs, n, add):
        if n == 10:
            seen["before"] = g.stagger_stats["steps"]
            add("c")

    out, g, _ = run("ab", stagger = True, lengths = lengths, min_jobs = 3, script = script)
    assert seen["before"] == 0 and g.stagger_stats["steps"] > 5
    assert out["a"] == run("a", stagger = False, lengths = lengths)[0]["a"]


def stretch_lines(capsys) -> list:
    return [l for l in capsys.readouterr().out.splitlines() if "calls in" in l]


def calls_of(line: str) -> int:
    return int(line.split(":")[1].split("calls")[0])


def test_each_stretch_of_staggered_calls_is_reported_in_one_line(capsys):
    """A server holds the generator behind its own worker: the lines on stdout are the only
    trace that the decode was staggered (and they are flushed: stdout of a service is not a
    terminal)."""
    _, g, _ = run("abc", True, min_jobs = 3)
    lines = [l for l in capsys.readouterr().out.splitlines() if "Staggered decode" in l]
    assert lines[0] == " -- Staggered decode: modules 0-3 on the first card, 4-7 on the second"
    assert len(lines) == 2 and g._stagger_run is None
    assert calls_of(lines[1]) == g.stagger_stats["steps"] and lines[1].endswith(" s, up to 3 jobs")

    # Nothing is reported for a decode that never staggers
    run("ab", True, min_jobs = 3)
    assert stretch_lines(capsys) == []

    # A job that takes several iterations to ingest its prompt ends the stretch; the next one
    # starts when it generates, with one job more
    def script(g, jobs, n, add):
        if n == 15:
            add("d")

    _, g, _ = run("abc", True, min_jobs = 3, lengths = {"a": 80, "b": 80, "c": 80, "d": 40}, script = script, prefill_iters = 4)
    lines = stretch_lines(capsys)
    assert len(lines) == 2, lines
    assert lines[0].endswith("up to 3 jobs") and lines[1].endswith("up to 4 jobs")
    assert sum(calls_of(l) for l in lines) == g.stagger_stats["steps"]


def test_a_stretch_ended_by_a_cancel_between_two_calls_is_reported_at_the_next_one(capsys):
    def script(g, jobs, n, add):
        if n == 12:
            # The pass in flight is completed by the cancel itself, outside iterate(); with two
            # jobs left the next call is a regular step and closes the stretch
            victim = g._flight.jobs[0]
            victim.cancelled = True
            g.cancel(victim)
            assert g._flight is None and g._stagger_run is not None
            capsys.readouterr()
        if n == 13:
            assert g._stagger_run is None

    _, g, _ = run("abc", True, min_jobs = 3, script = script)
    lines = stretch_lines(capsys)
    assert len(lines) == 1 and calls_of(lines[0]) == g.stagger_stats["steps"] == 12
