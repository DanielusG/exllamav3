"""
A page's earlier identity is not given back once a pass that writes into the page has been issued
(Job.forget_revert).

No GPU or model is required. The real page table (PageTable, CachePage), the real Job and the
generator's real prompt rounds (inline, on the prefill thread, in the pipeline) run over a stand-in
model whose modules write every token of a pass into a stand-in K/V cache, at its page and slot, and
can be made to fail or to wait part-way through the pass. Nobody here decodes: "a job generating" is
a job past its prompt, and the draft and decode steps are left out of iterate().

The page table takes a prompt's tail page and the pages for generation as unique pages, the least
valuable unreferenced pages, often complete pages of another conversation. It keeps the identity such
a page had (CachePage.backup) and gives it back when the job lets go of the page with can_revert still
set (sub_ref, revert). A prompt chunk cleared can_revert only at its record, after its passes, so a
chunk released without its record (a pass that fails, a cancel while the chunk is under way on the
prefill thread or in the pipeline) handed the other conversation its complete page back over this
job's K/V, and that conversation's next turn found the page by its hash and reused it.

When the compiled extension is not installed the source files under test are loaded on their own,
with the rest of the package stubbed.
"""

import importlib.util
import os
import sys
import threading
import time
import types
from contextlib import contextmanager

import pytest
import torch

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------------------------
#   Loading
# --------------------------------------------------------------------------------------------

def _load_real():
    sys.path.insert(0, REPO)
    try:
        from exllamav3.generator import generator as gen_mod
        from exllamav3.generator import job as job_mod
        from exllamav3.generator import pagetable as pt_mod
        return gen_mod, job_mod, pt_mod
    finally:
        sys.path.remove(REPO)


def _count_match(a, b, n):
    """exllamav3_ext.count_match_tensor: how many leading tokens of b match the first n of a"""
    m = min(n, b.shape[-1])
    eq = (a[0, :m] == b[0, :m]).tolist()
    return eq.index(False) if False in eq else m


def _load_stubbed():
    pkg = "exl3_revert_stub"
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
    ext = types.SimpleNamespace(count_match_tensor = _count_match)
    stub("ext", exllamav3_ext = ext)
    sys.modules[f"{pkg}.util"].cuda_sync_active = lambda: None
    sys.modules[f"{pkg}.util"].profile_opt = lambda *a, **k: None
    sys.modules[f"{pkg}.tokenizer"].MMEmbedding = _Any
    real("constants")
    stub("util.memory", malloc_trim = lambda: None)
    real("util.lane")
    real("util.device_copy")
    real("util.tensor")
    real("tokenizer.mm_embedding")
    stub("tokenizer.tokenizer", Tokenizer = _Any)
    real("cache.recurrent_util")
    sys.modules[f"{pkg}.cache"].RecurrentCache = real("cache.recurrent").RecurrentCache
    stub("cache.cache", Cache = _Any)
    stub("model.model", Model = _Any)
    real("model.forward_cursor")
    # (the page table imports the extension by the package's own name; only for the import)
    added = []
    for name, attrs in (("exllamav3", {}), ("exllamav3.ext", {"exllamav3_ext": ext})):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__dict__.update(attrs)
            sys.modules[name] = m
            added.append(name)
    try:
        pt_mod = real("generator.pagetable")
    finally:
        for name in added:
            sys.modules.pop(name, None)
    stub("generator.filter", Filter = _Any)
    stub("generator.loop_detect", LoopDetector = _Any)
    stub("generator.sampler", Sampler = _Any, DefaultSampler = _Any)
    job_mod = real("generator.job")
    stub("generator.cpu_cache", CPUPageCache = _Any)
    stub("generator.disk_cache", DiskPageCache = _Any)
    real("generator.draft_confidence")
    real("generator.draft_window")
    stub("generator.visualizer", CacheVisualizer = _Any)
    gen_mod = real("generator.generator")
    return gen_mod, job_mod, pt_mod


try:
    GEN, JOB, PT = _load_real()
except Exception:
    GEN, JOB, PT = _load_stubbed()

PAGE = PT.PAGE_SIZE


class FakeLane:
    """What the generator needs from a Lane, with no CUDA device behind it."""

    def __init__(self, index, devices):
        self.index = index
        self.devices = list(devices)

    def enter(self):
        pass

    @contextmanager
    def passing(self):
        yield

    def synchronize(self):
        pass


@pytest.fixture(autouse = True)
def _fake_lane(monkeypatch):
    monkeypatch.setattr(GEN, "Lane", FakeLane)
    monkeypatch.delenv("EXL3_PREFILL_PIPELINE_HEAD", raising = False)


# --------------------------------------------------------------------------------------------
#   Stand-in model and caches
# --------------------------------------------------------------------------------------------

class KV:
    """A cache whose K/V is the token each pass wrote at (page index, slot)."""

    def __init__(self, n_pages):
        self.initialized = True
        self.max_num_tokens = n_pages * PAGE
        self.kv = {}

    def write(self, ids, params):
        start = int(params["cache_seqlens"][0])
        table = params["block_table"]
        for i, t in enumerate(ids[0].tolist()):
            pos = start + i
            self.kv[(int(table[0, pos // PAGE]), pos % PAGE)] = t

    def page(self, page) -> list:
        return [self.kv.get((page.page_index, s)) for s in range(PAGE)]

    def copy_page(self, target, from_page, to_page, num_tokens):
        for s in range(num_tokens):
            target.kv[(to_page, s)] = self.kv.get((from_page, s))


class Module:
    """Writes the pass's tokens into the cache, then calls the model's hook (fail, wait)."""

    def __init__(self, model, idx, device):
        self.model = model
        self.idx = idx
        self.device = device
        self.caps = {}

    def prepare_for_device(self, x, params):
        return x

    def forward(self, x, params):
        ids = params["input_ids"]
        params["cache"].write(ids, params)
        if self.idx == len(self.model.modules) - 1:
            # (for the MTP head's update: the target's last hidden states)
            params["export_states"] = [torch.zeros((1, ids.shape[-1], 1))]
        if self.model.hook is not None:
            self.model.hook(self.idx, ids)
        return x


class Model:
    """Two modules on each of two pretend cards."""

    def __init__(self):
        self.loaded_tp = False
        self.caps = {}
        self.output_device = "cpu"
        self.config = types.SimpleNamespace(moe_cpu_hosts = {}, infer_params = None)
        self.modules = [Module(self, i, f"cuda:{i // 2}") for i in range(4)]
        self.fwd_modules = [(m, 0, i) for i, m in enumerate(self.modules)]
        self._get_prefetch_layers = []
        self.hook = None

    def prepare_inputs(self, input_ids, params):
        params["input_ids"] = input_ids
        return input_ids

    def forward(self, input_ids, params):
        x = self.prepare_inputs(input_ids, params)
        for module, _, _ in self.fwd_modules:
            x = module.forward(module.prepare_for_device(x, params), params)
        return x

    prefill = forward


class Draft:
    """The MTP head: its update writes the chunk into the draft cache."""

    def __init__(self):
        self.caps = {"mtp_draft": True}
        self.draft_verifier_params = {}
        self.modules = [types.SimpleNamespace(device = "cuda:0")]

    def prefill(self, input_ids, params):
        params["cache"].write(input_ids, params)


class Generating:
    """A job past its prompt: the prompt rounds then have somebody generating beside them."""

    def __init__(self):
        self.sequences = [None]
        self.serial_number = -1
        self.identifier = None

    def is_prefill_done(self):
        return True

    def prefill(self, results):
        pass

    def deallocate_pages(self):
        pass


def make_generator(n_pages = 3, max_chunk_size = 2048):
    g = GEN.Generator.__new__(GEN.Generator)
    g.model = Model()
    g.cache = KV(n_pages)
    g.draft_cache = KV(n_pages)
    g.draft_model = Draft()
    g.pagetable = PT.PageTable(g, g.cache)
    g.pending_jobs = []
    g.active_jobs = []
    g.job_serial = 0
    g.max_batch_size = 4
    g.max_chunk_size = max_chunk_size
    g.max_total_tokens = g.cache.max_num_tokens
    g.padded_vocab_size = 64
    g.recurrent_cache = None
    g.target_recurrent = False
    g.draft_recurrent = False
    g.mtp_draft = True
    g.dflash_draft = False
    g.num_draft_tokens = 1
    g.max_draft_tokens = 1
    g.ngram_match_min = 0
    g.ngram_corpus = None
    g.hybrid_draft_tokens = 0
    g.enable_defrag = False
    g.visualizer = None
    g.cpu_page_cache = None
    g.disk_page_cache = None
    g.disk_cache_write_interval = 0.0
    g.sample_pinned = None
    g.staging_buffers = {}
    g.draft_input_ids_pinned = None
    g.draft_ids_pinned = None
    g.filter_pool = types.SimpleNamespace(shutdown = lambda wait = True: None)
    g._stagger_init(False)
    g.stagger_decode = False
    g.prefill_thread = False
    g.side_min_tokens = 16
    g.side_max_rows = 8
    g.side_stats = {
        "chunks": 0, "beside": 0, "decode_steps": 0, "errors": 0,
        "t_chunks": 0.0, "t_waited": 0.0, "piped": 0, "ahead": 0,
    }
    g._side = None
    g._side_ok = None
    g._side_why = None
    g._side_lane = None
    g._side_pool = None
    g._side_turn = 0
    g._side_stretch = None
    g.prefill_pipeline = False
    g.pipe_depth = 2
    g._pipe = []
    g._pipe_ok = None
    g._pipe_why = None
    g._pipe_lanes = None
    g._pipe_pool = None
    g._pipe_stop = None
    g._pipe_head_first = False
    # Nobody decodes here: the draft and decode steps are left out, and so is the housekeeping of
    # a drained queue (defragmentation, the expert sweep)
    g.iterate_draftmodel_mtp_gen = lambda results: None
    g.iterate_gen = lambda results, draft_tokens = None: None
    g.on_queue_drained = lambda: None
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

A = [1000 + i for i in range(2 * PAGE + 1)]     # two complete pages and a token
B = [5000 + i for i in range(PAGE + 101)]       # a complete page and a tail of 100 tokens
C = [9000 + i for i in range(PAGE + 51)]


def make_job(g, ids, max_new_tokens = 1):
    job = JOB.Job(input_ids = torch.tensor([ids], dtype = torch.long), max_new_tokens = max_new_tokens)
    g.enqueue(job)
    return job


def ingest(g, job):
    """Iterate until the job is done with its prompt (a chunk beside a job generating is recorded
    by the first round that finds it done)."""
    deadline = time.time() + 10
    while not (job in g.active_jobs and job.is_prefill_done()):
        assert time.time() < deadline, "the prompt is not ingested"
        g.iterate()
        time.sleep(0.0005)


def conversation_a(g) -> list:
    """Conversation A ingested and let go: its two complete pages stay in the cache, unreferenced,
    under their content hashes. Returns their hashes."""
    a = make_job(g, A)
    ingest(g, a)
    g.cancel(a)
    pt = g.pagetable
    hashes = a.sequences[0].page_hashes[:2]
    for i, h in enumerate(hashes):
        page = pt.unreferenced_pages[h]
        assert page.kv_position == PAGE and g.cache.page(page) == A[i * PAGE:(i + 1) * PAGE]
    return hashes


def switch(g, where):
    g.prefill_thread = where != "inline"
    g.prefill_pipeline = where.startswith("pipeline")


def assert_a_s_next_turn_reads_a_s_own_pages(g, hashes):
    """The page that held A's second page (taken by another job as a unique page and written by
    it) is no longer under A's hash, and A's next turn reuses only pages that hold A's K/V."""
    pt = g.pagetable
    assert hashes[1] not in pt.unreferenced_pages and hashes[1] not in pt.referenced_pages, \
        "a page written by another job went back under A's hash"
    a2 = make_job(g, A + [7] * 100)
    g.iterate_start_jobs([])
    assert a2 in g.active_jobs
    seq = a2.sequences[0]
    for i, page in enumerate(seq.allocated_pages[:a2.cached_pages]):
        assert g.cache.page(page) == A[i * PAGE:(i + 1) * PAGE], \
            f"A's next turn reused page {i} with other K/V in it"
    assert a2.cached_pages == 1


def b_takes_a_s_second_page(g, hashes, b):
    """B's tail page is the page that held A's second page, kept for A until B writes into it."""
    tail = b.sequences[0].allocated_pages[1]
    assert tail.can_revert and tail.phash_revert == hashes[1] and tail.kv_position_revert == PAGE
    return tail


# --------------------------------------------------------------------------------------------
#   The hazard: a chunk released without its record
# --------------------------------------------------------------------------------------------

class FailedPass(RuntimeError):
    """What a failed pass raises (on the machine: torch.OutOfMemoryError, a kernel error)"""


@pytest.mark.parametrize("where, module", [
    ("inline", 1),
    ("thread", 1),
    ("pipeline", 1),        # in the first half, on the first card
    ("pipeline", 3),        # in the second half, the first card's modules have written
])
def test_a_chunk_whose_pass_fails_gives_no_page_back(generators, where, module):
    # B's only chunk (a complete page and the tail) fails part-way through its pass, after the
    # modules before the failing one have written into both of B's pages: the job is released
    # (Generator.reap_failed_job) without the record of the chunk
    g = generators()
    hashes = conversation_a(g)
    switch(g, where)
    b = make_job(g, B)
    g.iterate_start_jobs([])
    tail = b_takes_a_s_second_page(g, hashes, b)

    def fail(idx, ids):
        if idx == module and int(ids[0, 0]) == B[0]:
            raise FailedPass("out of memory")
    g.model.hook = fail
    results = g.iterate()
    g.model.hook = None

    assert [r["job"] for r in results if r["stage"] == "error"] == [b]
    assert isinstance(next(r["error"] for r in results if r["stage"] == "error"), FailedPass)
    assert b not in g.active_jobs and tail.ref_count == 0
    # The pass wrote B's tail into the page before it failed
    assert g.cache.page(tail)[:100] == B[PAGE:PAGE + 100]
    assert_a_s_next_turn_reads_a_s_own_pages(g, hashes)


@pytest.mark.parametrize("where", ["thread", "pipeline"])
def test_a_job_cancelled_with_its_chunk_under_way_gives_no_page_back(generators, where):
    # B's chunk is under way beside a job generating when B is cancelled: the generator lets the
    # chunk run to its end (Generator._side_drop) or to the end of the half it is in (_pipe_drop),
    # records nothing, and releases B's pages
    g = generators()
    hashes = conversation_a(g)
    switch(g, where)
    g.active_jobs.append(Generating())
    b = make_job(g, B)
    g.iterate_start_jobs([])
    tail = b_takes_a_s_second_page(g, hashes, b)
    go = threading.Event()
    reached = threading.Event()
    waited = []

    def wait(idx, ids):
        if idx == 0 and int(ids[0, 0]) == B[0]:
            waited.append(threading.current_thread().name)
            reached.set()
            assert go.wait(10), "the chunk was never let go"
    g.model.hook = wait

    g.iterate()
    under_way = g._side if where == "thread" else (g._pipe[0] if g._pipe else None)
    assert under_way is not None and under_way.job is b
    # The chunk's pass has written its first module's K/V and waits there, on its own thread
    assert reached.wait(10) and not under_way.future.done()
    assert len(waited) == 1 and waited[0] != threading.current_thread().name
    timer = threading.Timer(0.05, go.set)
    timer.start()
    g.cancel(b)
    timer.join()
    g.model.hook = None

    assert b not in g.active_jobs and tail.ref_count == 0
    assert g._side is None and g._pipe == []
    assert g.cache.page(tail)[:100] == B[PAGE:PAGE + 100]
    assert_a_s_next_turn_reads_a_s_own_pages(g, hashes)


# --------------------------------------------------------------------------------------------
#   What stays: a page no pass has written goes back, a run that records every chunk is as before
# --------------------------------------------------------------------------------------------

def test_a_job_cancelled_before_its_first_chunk_gives_its_page_back(generators):
    # C waits for its turn at the prefill thread (B's chunk is under way) with its pages allocated
    # and nothing planned: cancelled, its tail page goes back to A, whose K/V is still in it
    g = generators(n_pages = 5)
    hashes = conversation_a(g)
    switch(g, "thread")
    g.active_jobs.append(Generating())
    b = make_job(g, B)
    c = make_job(g, C)
    go = threading.Event()

    def wait(idx, ids):
        if idx == 0 and int(ids[0, 0]) == B[0]:
            assert go.wait(10), "the chunk was never let go"
    g.model.hook = wait

    g.iterate()
    assert g._side is not None and g._side.job is b and not getattr(c, "_side_turn", 0)
    tail = c.sequences[0].allocated_pages[1]
    assert tail.can_revert and tail.phash_revert == hashes[1]
    g.cancel(c)
    pt = g.pagetable
    assert pt.unreferenced_pages.get(hashes[1]) is tail
    assert tail.kv_position == PAGE and pt.get_live_page(hashes[1]) is tail
    assert g.cache.page(tail) == A[PAGE:2 * PAGE]

    go.set()
    ingest(g, b)
    g.model.hook = None


def test_a_page_past_what_the_job_wrote_still_goes_back(generators):
    # B ingests its prompt and is let go before it generates: the page it holds for generation was
    # never written, and goes back to A (whose first page it was)
    g = generators()
    hashes = conversation_a(g)
    b = make_job(g, B, max_new_tokens = 300)
    g.iterate_start_jobs([])
    pages = b.sequences[0].allocated_pages
    assert len(pages) == 3 and pages[2].phash_revert == hashes[0]
    ingest(g, b)
    assert not pages[1].can_revert and pages[2].can_revert
    g.cancel(b)
    pt = g.pagetable
    assert hashes[1] not in pt.unreferenced_pages
    assert pt.get_live_page(hashes[0]) is pages[2]
    assert g.cache.page(pages[2]) == A[:PAGE]


def test_the_decode_step_gives_the_identity_up_as_it_stages_its_pass(generators):
    # Upstream's decode step, which the change leaves alone: the tokens of the pass (the job's last
    # token and the drafts) go into the pages, and the pages give up their earlier identity, before
    # the forward pass is issued (get_input_ids_list, add_to_cache). The window here crosses into
    # the page held for generation
    g = generators(n_pages = 5)
    conversation_a(g)
    d = make_job(g, [3000 + i for i in range(2 * PAGE)], max_new_tokens = 10)
    ingest(g, d)
    seq = d.sequences[0]
    assert seq.kv_position == 2 * PAGE - 1
    ahead = seq.allocated_pages[2]
    assert ahead.can_revert
    d.get_input_ids_list(torch.tensor([[5]], dtype = torch.long), 0, add_to_cache = True)
    assert not ahead.can_revert and ahead.kv_position == 1


def snapshot(g) -> list:
    """The page table, random hashes left out (a process-wide counter), and the K/V."""
    def h(x):
        return x if x is not None and PT.is_content_hash(x) else None
    pages = [
        (p.page_index, h(p.phash), h(p.prev_hash), p.kv_position, p.ref_count, p.can_revert, p.access_serial,
         p.sequence[0, :p.kv_position].tolist())
        for p in g.pagetable.all_pages
    ]
    return [pages, dict(g.cache.kv), dict(g.draft_cache.kv)]


def converse(g, where) -> list:
    """Turns of three conversations whose chunks are all recorded, each let go when its prompt is
    in, each next one reusing what the ones before left and taking their pages; beside a job
    generating, in chunks of two pages. A snapshot after each step."""
    shots = []
    conversation_a(g)
    shots.append(snapshot(g))
    switch(g, where)
    g.active_jobs.append(Generating())
    g.max_chunk_size = 2 * PAGE
    turns = [
        (B, 300), (C, 300), (A + [7] * 100, 300), (B + [8] * 200, 300), (C + [9] * 300, 1),
        (A + [7] * 100 + [6] * 50, 300),
    ]
    for ids, max_new in turns:
        job = make_job(g, ids, max_new_tokens = max_new)
        ingest(g, job)
        shots.append(snapshot(g))
        g.cancel(job)
        shots.append(snapshot(g))
    return shots


@pytest.mark.parametrize("where", ["inline", "thread", "pipeline"])
def test_a_run_that_records_every_chunk_leaves_the_page_table_as_before(generators, monkeypatch, where):
    # Every page a chunk's plan gives up is given up by the chunk's record too, before anything can
    # release the page; without a failure or a cancel in between, the page table, and with it what
    # is reused and evicted afterwards, is the same as with the identity given up at the record only
    seen = {"early": 0, "reverted": 0}
    forget = getattr(JOB.Job, "forget_revert", None)
    revert = PT.CachePage.revert

    def counting_forget(self, seq, start, end):
        seen["early"] += sum(p.can_revert for p in seq.allocated_pages[start // PAGE:(end + PAGE - 1) // PAGE])
        forget(self, seq, start, end)

    def counting_revert(self):
        seen["reverted"] += self.kv_position_revert == PAGE
        revert(self)

    monkeypatch.setattr(PT.CachePage, "revert", counting_revert)
    if forget is not None:
        monkeypatch.setattr(JOB.Job, "forget_revert", counting_forget)
    now = converse(generators(n_pages = 6), where)
    # (pages given up at the plan that the record gives up later, and pages that went back whole
    # to the conversation they were taken from: the run is not one where nothing of it happens)
    assert seen["early"] >= 5 or forget is None
    assert seen["reverted"] >= 2

    monkeypatch.setattr(JOB.Job, "forget_revert", lambda self, seq, start, end: None, raising = False)
    before = converse(generators(n_pages = 6), where)
    assert now == before
