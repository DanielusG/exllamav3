"""
Content-addressed multimodal pages (Generator.mm_content_hash, EXL3_MM_CONTENT_HASH; FORK.md).

No GPU, model or compiled extension is required. The embedding container and its digest registry
(tokenizer/mm_embedding.py), the page table (PageTable, Sequence), the job's hash of a page completed during
generation (Job.hash_completed_page), the CPU page tier, upstream's DiskPageCache (writer thread, reader pool,
index) and the recurrent checkpoint cache are the real ones. The "VRAM" the CPU tier copies from and to is a host
tensor, and the tier is set up around it without its __init__ (which asserts CUDA cache tensors). A prompt is
"prefilled" by writing each page's tokens and a K/V pattern made from the page's hash, as the record of a chunk
leaves them. A second process is stood for by a fresh allocator and digest registry (an image then gets other
ids than it had), a fresh page table, CPU tier and checkpoint cache, and a new DiskPageCache on the same
directory.

What is checked: with the switch off, hashes, the persistable flag, the tier's page record and the restore are
upstream's, against an independent copy of upstream's hash; with it on, text pages hash as upstream's; the same
image in two processes gives the same page hashes, different images on the same ids or one image at other rows
or positions give other ones; the pages and the checkpoint of an image conversation reach the disk and a second
process resumes from them with its own ids in the pages; defrag and the page table's own validation after that;
an embedding without a digest falls back to upstream's rule for its chain; the digest covers what a forward
reads; every hashing site goes through the one helper.

When the compiled extension is not installed the source files under test are loaded on their own, with the rest
of the package stubbed.
"""

import ast
import hashlib
import importlib.util
import os
import sys
import types
from collections import deque

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "exllamav3")


def _load_real():
    sys.path.insert(0, REPO)
    try:
        from exllamav3.tokenizer import mm_embedding
        from exllamav3.generator import pagetable, job, cpu_cache, disk_cache
        from exllamav3.cache import recurrent
        return mm_embedding, pagetable, job, cpu_cache, disk_cache, recurrent
    finally:
        sys.path.remove(REPO)


def _load_stubbed():
    pkg = "exl3_mmhash_stub"

    def stub(name, **attrs):
        full = f"{pkg}.{name}" if name else pkg
        m = types.ModuleType(full)
        m.__path__ = []
        m.__dict__.update(attrs)
        sys.modules[full] = m
        return m

    def real(name):
        full = f"{pkg}.{name}"
        path = os.path.join(SRC, *name.split(".")) + ".py"
        spec = importlib.util.spec_from_file_location(full, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        return m

    def functions(name, names, **attrs):
        """A stub of module `name` with the real source of the functions `names` (the disk tier's checkpoint
        (de)serialization lives in model_tp_fn, which imports the extension)"""
        path = os.path.join(SRC, *name.split(".")) + ".py"
        tree = ast.parse(open(path).read())
        body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        m = stub(name, torch = torch, **attrs)
        exec(compile(ast.Module(body = body, type_ignores = []), path, "exec"), m.__dict__)
        return m

    class _Any:
        def __init__(self, *a, **k):
            pass

    stub("")
    for p in ("cache", "generator", "model", "tokenizer", "util"):
        stub(p)
    ext = types.SimpleNamespace()
    stub("ext", exllamav3_ext = ext)
    stub("util.memory", malloc_trim = lambda: None, check_host_memory = lambda *a, **k: None)
    stub("util.pinned_arena", PinnedArena = _Any)
    sys.modules[f"{pkg}.util"].profile_opt = lambda *a, **k: None
    stub("model.forward_cursor", ForwardCursor = _Any)
    stub("cache.cache", Cache = _Any)
    stub("generator.filter", Filter = _Any)
    stub("generator.loop_detect", LoopDetector = _Any)
    stub("generator.sampler", Sampler = _Any, DefaultSampler = _Any)
    unused = lambda *a, **k: None
    functions("model.model_tp_fn", ("pack_stash", "unpack_stash"),
              mp_disk_layout = unused, mp_disk_set_layout = unused, mp_disk_write = unused, mp_disk_read = unused,
              mp_disk_state_describe = unused, mp_disk_state_write = unused, mp_disk_state_read = unused)
    real("constants")
    real("version")
    mm = real("tokenizer.mm_embedding")
    sys.modules[f"{pkg}.tokenizer"].mm_embedding = mm
    sys.modules[f"{pkg}.tokenizer"].MMEmbedding = mm.MMEmbedding
    real("util.device_copy")
    real("util.lane")
    real("util.tensor")
    rec = real("cache.recurrent")
    sys.modules[f"{pkg}.cache"].RecurrentCache = rec.RecurrentCache
    cpu = real("generator.cpu_cache")
    disk = real("generator.disk_cache")
    # The page table imports the extension by its absolute name
    saved = {k: sys.modules.get(k) for k in ("exllamav3", "exllamav3.ext")}
    sys.modules["exllamav3"] = types.ModuleType("exllamav3")
    sys.modules["exllamav3"].__path__ = []
    sys.modules["exllamav3.ext"] = types.ModuleType("exllamav3.ext")
    sys.modules["exllamav3.ext"].exllamav3_ext = ext
    try:
        pt = real("generator.pagetable")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    job = real("generator.job")
    return mm, pt, job, cpu, disk, rec


try:
    MM, PT, JOB, CPU, DISK, REC = _load_real()
except Exception:
    MM, PT, JOB, CPU, DISK, REC = _load_stubbed()

PAGE = PT.PAGE_SIZE
FIRST = MM.FIRST_MM_EMBEDDING_INDEX
START, END = 151652, 151653     # vision start / end tokens (text ids)
HIDDEN = 16                     # width of a pretend embedding row
KV_WIDTH = 8                    # values per token of the pretend K/V cache
CKPT = 4 * 1024                 # bytes of a pretend checkpoint
ENV = "EXL3_MM_CONTENT_HASH"
# The switch in the tests of the new behaviour: each of them fails with it set to False
ON = True


def upstream_hash(ids: torch.Tensor, prev: bytes | None) -> bytes:
    """Upstream's page hash, written out again: blake2b-128 over the previous hash and the page's int64 ids"""
    h = hashlib.blake2b(digest_size = 16)
    if prev is not None:
        h.update(prev)
    h.update(np.ascontiguousarray(ids.numpy()).astype("<i8").tobytes())
    return h.digest()


# --------------------------------------------------------------------------------------------
#   Processes, images, prompts
# --------------------------------------------------------------------------------------------

@pytest.fixture(autouse = True)
def process(monkeypatch):
    """A fresh process for every test: allocator, digest registry, digests off unless a test turns them on"""
    monkeypatch.delenv(ENV, raising = False)
    new_process(monkeypatch)
    yield
    MM.set_content_digests(None)


def new_process(monkeypatch, skip: int = 0):
    """Stand for a new process: a fresh allocator (moved on by `skip` ids, so the same image gets other ids than
    in the last one) and a fresh digest registry"""
    alloc = MM.MMTokenAllocator()
    alloc.next_token_index += skip
    monkeypatch.setattr(MM, "global_allocator", alloc)
    monkeypatch.setattr(MM, "global_registry", MM.MMDigestRegistry())


def image(seed: int, rows: int = 600, digests: bool | None = True, **kw):
    """An embedding of `rows` rows whose values follow from seed, made with digests on, off, or as the
    environment says (None)"""
    MM.set_content_digests(digests)
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn((rows, HIDDEN), generator = g).half()
    ts = torch.tensor([[START] + [-1] * rows + [END]], dtype = torch.long)
    kw.setdefault("grid_thw", (1, 2, rows * 2))
    kw.setdefault("mrope_merge_size", 2)
    return MM.MMEmbedding(embeddings = emb, token_string = ts, **kw)


def text(seed: int, n: int) -> list[int]:
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randint(0, 150000, (n,), generator = g).tolist()


def prompt(*parts) -> torch.Tensor:
    ids = []
    for p in parts:
        ids += p.token_list if isinstance(p, MM.MMEmbedding) else list(p)
    return torch.tensor([ids], dtype = torch.long)


def hashes(ids: torch.Tensor, content: bool) -> list[bytes]:
    seq = PT.Sequence(ids, ids)
    if content:
        seq.prepare(False, PAGE, content_hash = True)
    else:
        seq.prepare(False, PAGE)
    return seq.page_hashes


def mm_pages(ids: torch.Tensor) -> list[int]:
    """The full pages holding a placeholder id"""
    n = (ids.shape[-1] - 1) // PAGE
    return [i for i in range(n) if bool((ids[0, i * PAGE:(i + 1) * PAGE] >= FIRST).any())]


# --------------------------------------------------------------------------------------------
#   The world: page table, CPU tier over host "VRAM", disk tier, checkpoint cache
# --------------------------------------------------------------------------------------------

class Gen:
    """What the page table, the checkpoint cache and Job.hash_completed_page read of a generator"""

    def __init__(self, mm_content_hash: bool):
        self.mm_content_hash = mm_content_hash
        self.enable_defrag = True
        self.draft_cache = None
        self.recurrent_cache = None


class Cache:
    """A cache as the page table and defrag see it: a size, no tensors to rotate (the K/V tensor is the
    CPU tier's segment)"""

    def __init__(self, pages: int):
        self.max_num_tokens = pages * PAGE
        self.model = types.SimpleNamespace(loaded_tp = False)

    def get_all_tensors(self):
        return []


def make_tier(kv: torch.Tensor, max_slots: int = 64):
    """The real CPUPageCache over one host segment, set up as __init__ does for a local (non-TP) cache: the
    copies are synchronous, so there are no stream events"""
    tier = CPU.CPUPageCache.__new__(CPU.CPUPageCache)
    tier.tp_groups = []
    tier.tp = False
    tier.segments = [(kv, 0, tuple(kv.shape[1:]), kv.dtype)]
    nbytes = kv[0].numel() * kv.element_size()
    tier.slab_size = CPU._align(nbytes, 4096)
    tier.slot_size = tier.slab_size
    tier.max_slots = max_slots
    tier.pagetable = None
    tier.disk_tier = None
    tier._devices = []
    tier.entries = {}
    tier.slot_views = []
    tier.free_slots = deque()
    tier.slot_events = {}
    tier.num_slots = 0
    tier._order = deque()
    tier._order_pops = 0
    tier._order_rebuild = max(64, max_slots // 8)
    tier.metrics = {"pushes": 0, "dedup_hits": 0, "restores": 0, "evictions": 0, "promotions": 0,
                    "promotion_failures": 0}
    tier.arena = types.SimpleNamespace(tensor = torch.zeros(max_slots * tier.slab_size, dtype = torch.uint8),
                                       close = lambda: None)
    return tier


class World:
    """One process's page table, tiers and checkpoint cache, the disk tier on `root` (None: no disk tier)"""

    def __init__(self, root, content: bool, pages: int = 32):
        self.gen = Gen(content)
        self.pt = PT.PageTable(self.gen, Cache(pages))
        self.kv = torch.zeros((pages, PAGE, KV_WIDTH), dtype = torch.float16)
        self.tier = make_tier(self.kv)
        self.tier.attach(self.pt)
        self.pt.cpu_tier = self.tier
        self.disk = None
        if root is not None:
            self.disk = DISK.DiskPageCache(str(root), [], self.tier, 1 << 30, fingerprint = "mm-content-hash-test",
                                           min_state_distance = 0, min_free = 0)
            self.tier.attach_disk(self.disk)
        self.rc = REC.RecurrentCache(types.SimpleNamespace(loaded_tp = False), max_size = 64 * CKPT)
        self.rc.pagetable = self.pt
        self.rc.disk_tier = self.disk
        self.gen.recurrent_cache = self.rc

    def close(self):
        if self.disk is not None:
            self.disk.close()


def kv_pattern(h: bytes) -> torch.Tensor:
    """The K/V a page's content leaves in the cache, a function of its hash"""
    g = torch.Generator().manual_seed(int.from_bytes(h[:6], "little"))
    return torch.randn((PAGE, KV_WIDTH), generator = g).half()


class State:
    """A recurrent state as the checkpoint cache sees it: a size and a stash in 1.6.2's paired shape"""
    checkpoint_size = CKPT

    def __init__(self, position: int):
        self.position = position

    def stash(self):
        t = REC.host_pool.take((CKPT // 4,), torch.float32)
        t.fill_(self.position)
        return {"position": self.position, "checkpoint_size": CKPT, "target": {"layer.0": t}}


def make_job(world, seq):
    """A job as Job.hash_completed_page sees one"""
    job = JOB.Job.__new__(JOB.Job)
    job.generator = world.gen
    job.pagetable = world.pt
    job.sequences = [seq]
    job.tail_checkpoint_key = None
    job.prompt_tail_key = None
    return job


def start(world, ids: torch.Tensor, content: bool, max_new: int = PAGE):
    """Queue and allocate a sequence the way Job.prepare_for_queue and Job.allocate_pages do"""
    seq = PT.Sequence(ids, ids)
    if content:
        seq.prepare(False, max_new, content_hash = True)
    else:
        seq.prepare(False, max_new)
    n, cached, _, stashed = seq.allocate_pages(world.pt, world.rc)
    return seq, cached, stashed


def prefill(world, seq):
    """What the record of the prompt's chunks leaves: each complete page's tokens, K/V and link; the last
    complete page's checkpoint"""
    seq_ids = seq.sequence_ids.torch()
    n = (len(seq.sequence_ids) - 1) // PAGE
    for i in range(seq.kv_position // PAGE, n):
        page = seq.allocated_pages[i]
        assert page.phash == seq.page_hashes[i]
        page.sequence.copy_(seq_ids[:, i * PAGE:(i + 1) * PAGE])
        page.kv_position = PAGE
        page.prev_hash = seq.allocated_pages[i - 1].phash if i else None
        world.kv[page.page_index].copy_(kv_pattern(page.phash))
    seq.kv_position = n * PAGE
    world.rc.put(seq.page_hashes[n - 1], State(seq.kv_position))
    return n


def generate(world, seq, tokens: list[int]):
    """Generated tokens one at a time, as Job.receive_sample takes them: a page that completes gets its hash
    from Job.hash_completed_page, its K/V, and a checkpoint at its end (the regular checkpoint of an answer)"""
    job = make_job(world, seq)
    page_idx = seq.kv_position // PAGE
    page = seq.allocated_pages[page_idx]
    page.prev_hash = seq.allocated_pages[page_idx - 1].phash if page_idx else None
    pos = seq.kv_position
    page.sequence[0, :len(seq.sequence_ids) - pos].copy_(seq.sequence_ids.torch()[0, pos:])
    page.kv_position = len(seq.sequence_ids) - pos
    seq.kv_position = len(seq.sequence_ids)
    completed = []
    for t in tokens:
        seq.sequence_ids.append(torch.tensor([[t]], dtype = torch.long))
        page = seq.allocated_pages[seq.kv_position // PAGE]
        page.sequence[0, page.kv_position] = t
        page.kv_position += 1
        seq.kv_position += 1
        if seq.kv_position % PAGE == 0:
            idx = seq.kv_position // PAGE - 1
            job.hash_completed_page(seq, idx)
            p = seq.allocated_pages[idx]
            world.kv[p.page_index].copy_(kv_pattern(p.phash))
            world.rc.put(p.phash, State(seq.kv_position))
            completed.append(p.phash)
    return completed


def persist(world, seq):
    """Queue the job's complete pages for the disk the way the generator's write-behind and idle flush do
    (Generator._queue_pages_to_disk: complete, content-hashed, persistable, through the CPU tier's store),
    then let the writer finish and close the tier as close() does"""
    for page in seq.allocated_pages:
        if page.kv_position != PAGE or not PT.is_content_hash(page.phash) or not page.persistable:
            continue
        if page.phash in world.tier or world.disk.has_page(page.phash) or page.phash in world.disk.pending_pages:
            continue
        world.tier.store(page, world.pt.access_serial)
    world.disk.flush()
    # A checkpoint whose page was still on its way stayed a candidate; the next write (or close) writes it
    world.disk.drain_ready_states()
    world.disk.flush()


def conversation(root, monkeypatch, content: bool):
    """Process A: a prompt with an image whose last rows are on the prompt's last, partial page, ingested, an
    answer generated past that page, its pages persisted. Process B: the same image embedded again (other ids),
    the next turn's prompt (A's prompt and answer) queued and allocated. Returns what both saw"""
    a = types.SimpleNamespace()
    image(0, rows = 4, digests = content)              # something embedded before: the image's ids move on
    x = image(7, digests = content)
    a.ids = prompt(text(1, 100), x, text(2, 10))       # 712 tokens: pages 0-1 complete, the image ends on page 2
    a.world = World(root, content)
    seq, a.cached, _ = start(a.world, a.ids, content)
    prefill(a.world, seq)
    a.answer = text(3, 60)
    a.completed = generate(a.world, seq, a.answer)
    a.hashes = [p.phash for p in seq.allocated_pages[:3]]
    a.persistable = [p.persistable for p in seq.allocated_pages[:3]]
    a.kv = {h: world_kv(a.world, h) for h in a.hashes}
    persist(a.world, seq)
    # Later the pages leave VRAM: PageTable.evict pushes them through the CPU tier's store, write-through
    for page in seq.allocated_pages[:3]:
        a.world.tier.store(page, a.world.pt.access_serial)
    a.world.disk.flush()
    a.on_disk = [a.world.disk.has_page(h) for h in a.hashes]
    a.ckpt_on_disk = a.world.disk.has_state(a.hashes[2])
    a.world.pt.deallocate_pages(seq.allocated_pages)
    a.world.close()
    a.metrics = dict(a.world.disk.metrics)

    b = types.SimpleNamespace()
    new_process(monkeypatch, skip = 1000)
    x2 = image(7, digests = content)
    b.x_ids = (x.first_index, x2.first_index)
    b.ids = torch.cat((prompt(text(1, 100), x2, text(2, 10)), torch.tensor([a.answer], dtype = torch.long)), dim = -1)
    b.world = World(root, content)
    b.has = [b.world.rc.has(h) for h in a.hashes]
    seq, b.cached, b.stashed = start(b.world, b.ids, content)
    b.seq = seq
    b.hashes = list(seq.page_hashes[:3])
    b.persistable = [p.persistable for p in seq.allocated_pages[:3]]
    return a, b


def world_kv(world, h: bytes) -> torch.Tensor:
    page = world.pt.get_live_page(h)
    return world.kv[page.page_index].clone()


# --------------------------------------------------------------------------------------------
#   With the switch off: upstream
# --------------------------------------------------------------------------------------------

def test_off_by_default_and_upstream_hashes(monkeypatch):
    """No digest is made and no id range registered; Sequence.prepare and Job.hash_completed_page hash every
    page, text and image ones, as upstream's function does; the content path is never taken; the pages from the
    first placeholder's on are not persistable"""
    assert not MM.content_digests()
    x = image(7, digests = None)
    assert x.content_digest is None and len(MM.global_registry) == 0
    monkeypatch.setattr(PT, "content_hash_checksum", None)
    monkeypatch.setattr(JOB, "content_hash_checksum", None)

    ids = prompt(text(1, 300), x, text(2, 300), x, text(3, 40))
    world = World(None, False)
    seq, cached, _ = start(world, ids, False)
    prev, ref = None, []
    for i in range((ids.shape[-1] - 1) // PAGE):
        prev = upstream_hash(ids[:, i * PAGE:(i + 1) * PAGE], prev)
        ref.append(prev)
    assert seq.page_hashes == ref and len(mm_pages(ids)) >= 4
    assert seq.page_hashes == [PT.tensor_hash_checksum(ids[:, i * PAGE:(i + 1) * PAGE], seq.page_hashes[i - 1] if i else None)
                               for i in range(len(ref))]
    first = int((ids[0] >= FIRST).nonzero()[0]) // PAGE
    assert seq.mm_first_page == first and not seq.content_hash
    assert [p.persistable for p in seq.allocated_pages] == [i < first for i in range(len(seq.allocated_pages))]

    prefill(world, seq)
    completed = generate(world, seq, text(4, PAGE))
    n = len(ref)
    assert completed[0] == upstream_hash(seq.sequence_ids.torch()[:, n * PAGE:(n + 1) * PAGE], ref[-1])
    world.pt.validate_pagetable([types.SimpleNamespace(sequences = [seq])])


def test_off_disk_record_and_restore_are_upstream(tmp_path, monkeypatch):
    """With the switch off an image conversation keeps upstream's rule: its pages from the image on are not
    written (skipped_mm), its checkpoint stays a candidate and is dropped, and the next process finds nothing
    of it past the text before the image; a restored text page takes the ids stored with it"""
    a, b = conversation(tmp_path, monkeypatch, content = False)
    assert a.persistable == [False, False, False] and a.on_disk == [False, False, False]
    assert a.metrics["skipped_mm"] == 3 and not a.ckpt_on_disk
    assert a.metrics["state_candidates_dropped"] >= 1
    assert b.hashes != a.hashes and b.cached == 0 and b.has == [False, False, False]

    # A text-only conversation: written, read back by the next process, the stored ids restored
    world = World(tmp_path, False)
    ids = torch.tensor([text(5, 3 * PAGE + 10)], dtype = torch.long)
    seq, _, _ = start(world, ids, False)
    prefill(world, seq)
    persist(world, seq)
    world.close()
    world = World(tmp_path, False)
    calls = []
    real_fetch = world.tier.fetch

    def fetch(h, page_index, serial):
        e = real_fetch(h, page_index, serial)
        calls.append(e["tokens"].clone())
        return e
    monkeypatch.setattr(world.tier, "fetch", fetch)
    seq, cached, stashed = start(world, ids, False)
    assert cached == 3 and stashed is not None and len(calls) == 3
    for i in range(3):
        assert torch.equal(seq.allocated_pages[i].sequence.view(-1), calls[i].view(-1))
        assert torch.equal(world.kv[seq.allocated_pages[i].page_index], kv_pattern(seq.page_hashes[i]))
    world.close()


# --------------------------------------------------------------------------------------------
#   With the switch on
# --------------------------------------------------------------------------------------------

def test_text_pages_hash_as_upstream():
    """Content hashing changes nothing for a page without a placeholder: the same function, the same bytes,
    the whole chain of a text prompt identical; text pages before an image identical too"""
    MM.set_content_digests(True)
    ids = torch.tensor([text(1, 5 * PAGE + 3)], dtype = torch.long)
    assert hashes(ids, True) == hashes(ids, False)
    x = image(7)
    ids = prompt(text(1, 2 * PAGE + 30), x, text(2, 40))
    on, off = hashes(ids, True), hashes(ids, False)
    assert on[:2] == off[:2] and on[2] != off[2]
    page = ids[:, :PAGE]
    assert PT.content_hash_checksum(page, None) == upstream_hash(page, None)
    assert PT.content_hash_checksum(page, b"p" * 16) == upstream_hash(page, b"p" * 16)


def test_same_image_in_two_processes_same_hashes(monkeypatch):
    """The same image embedded in two processes gets other ids and the same page hashes, for every page of
    the chain from the image on. With the switch off the hashes follow the ids and differ (fails)"""
    image(0, rows = 3, digests = ON)
    x = image(7, digests = ON)
    ids_a = prompt(text(1, 300), x, text(2, 700))
    h_a = hashes(ids_a, ON)
    new_process(monkeypatch, skip = 12345)
    x2 = image(7, digests = ON)
    ids_b = prompt(text(1, 300), x2, text(2, 700))
    assert x2.first_index != x.first_index and not torch.equal(ids_a, ids_b)
    assert x2.content_digest == x.content_digest
    h_b = hashes(ids_b, ON)
    assert mm_pages(ids_a) and h_a == h_b


def test_different_images_on_the_same_ids_differ(monkeypatch):
    """Two processes give two different images the same ids: upstream's hashes are the same (why upstream keeps
    such pages off the disk), content hashes differ from the image's first page on. With the switch off the
    hashes are equal (fails)"""
    MM.set_content_digests(True)
    x = image(7)
    ids_a = prompt(text(1, 300), x, text(2, 400))
    new_process(monkeypatch)
    MM.set_content_digests(True)
    y = image(8)
    ids_b = prompt(text(1, 300), y, text(2, 400))
    assert torch.equal(ids_a, ids_b) and x.content_digest != y.content_digest
    assert hashes(ids_a, False) == hashes(ids_b, False)
    new_process(monkeypatch)
    image(7, digests = ON)
    h_a = hashes(ids_a, ON)
    new_process(monkeypatch)
    image(8, digests = ON)
    h_b = hashes(ids_b, ON)
    first = mm_pages(ids_a)[0]
    assert h_a[:first] == h_b[:first]
    assert all(p != q for p, q in zip(h_a[first:], h_b[first:]))


def test_same_image_at_other_rows_or_positions_differs(monkeypatch):
    """One image, in two processes: moved by one token in the prompt, every page from the image on hashes
    differently, in the other process too; the same layout hashes the same in both (fails with the switch
    off). Within a page: other rows of the image, or the same rows one place later, give other hashes; the same
    rows from a second embedding of the same content (a re-embedded image) give the same one"""
    x = image(7, digests = ON)
    at_a = hashes(prompt(text(1, 300), x, text(2, 600)), ON)
    moved_a = hashes(prompt(text(1, 301), x, text(2, 599)), ON)
    new_process(monkeypatch, skip = 777)
    x2 = image(7, digests = ON)
    at_b = hashes(prompt(text(1, 300), x2, text(2, 600)), ON)
    first = mm_pages(prompt(text(1, 300), x2, text(2, 600)))[0]
    assert at_a == at_b and moved_a[:first] == at_a[:first]
    assert all(p != q for p, q in zip(moved_a[first:], at_a[first:]))

    rows = torch.tensor(x2.token_list[1:-1], dtype = torch.long)
    page = lambda r: r[:PAGE].view(1, -1)
    assert PT.content_hash_checksum(page(rows), None) != PT.content_hash_checksum(page(rows[PAGE:]), None)
    shifted = torch.cat((torch.tensor([5]), rows[:PAGE - 1]))
    assert PT.content_hash_checksum(page(rows), None) != PT.content_hash_checksum(page(shifted), None)
    x3 = image(7, digests = True)
    rows3 = torch.tensor(x3.token_list[1:-1], dtype = torch.long)
    assert not torch.equal(rows3, rows)
    assert PT.content_hash_checksum(page(rows3), b"q" * 16) == PT.content_hash_checksum(page(rows), b"q" * 16)
    # Two images of one content side by side in one page: one run, whichever embeddings the ids came from
    mixed = torch.cat((rows[-100:], rows3[:156])).view(1, -1)
    same = torch.cat((rows3[-100:], rows[:156])).view(1, -1)
    assert PT.content_hash_checksum(mixed, None) == PT.content_hash_checksum(same, None)


def test_image_conversation_persists_and_resumes_in_another_process(tmp_path, monkeypatch):
    """Process A ingests a prompt whose image ends on its last partial page, generates past that page
    (Job.hash_completed_page) and persists: every page is written, the checkpoint at the end of the image's
    last page too. Process B embeds the image again (other ids): the next turn's prompt hashes to A's pages,
    the checkpoint is found (RecurrentCache.has, from disk), allocate_pages resumes past the image with the
    pages restored from the disk tier and the checkpoint read back; the restored pages carry B's own ids, not
    A's, and the page table's validation holds. With the switch off: nothing past the text is written, and B
    resumes nowhere (fails)"""
    a, b = conversation(tmp_path, monkeypatch, content = ON)
    assert a.persistable == [True, True, True] and a.on_disk == [True, True, True] and a.ckpt_on_disk
    assert a.metrics["skipped_mm"] == 0 and a.metrics["state_writes"] >= 1
    assert b.x_ids[0] != b.x_ids[1]
    assert b.hashes == a.hashes and b.has[2] and b.persistable == [True, True, True]
    assert b.cached == 3 and b.stashed is not None and b.stashed["position"] == 3 * PAGE
    assert torch.equal(b.stashed["target"]["layer.0"], torch.full((CKPT // 4,), 3 * PAGE, dtype = torch.float32))
    world, seq = b.world, b.seq
    ids_b = seq.sequence_ids.torch()
    for i in range(3):
        page = seq.allocated_pages[i]
        assert page.kv_position == PAGE and page.phash == a.hashes[i]
        assert torch.equal(world.kv[page.page_index], a.kv[a.hashes[i]])
        assert torch.equal(page.sequence, ids_b[:, i * PAGE:(i + 1) * PAGE])
    assert not torch.equal(seq.allocated_pages[1].sequence, a.ids[:, PAGE:2 * PAGE])
    assert world.pt.metrics["alloc_disk_pages"] == 3
    world.pt.validate_pagetable([types.SimpleNamespace(sequences = [seq])])
    world.close()


def test_defrag_of_restored_image_pages_keeps_hashes(tmp_path, monkeypatch):
    """Pages restored from another process's disk records, moved by defrag: defrag moves pages by their links
    and never hashes; afterwards every complete page still hashes (from its ids, which are this process's) to
    the hash it is indexed under, and the next turn finds the whole chain in VRAM"""
    a, b = conversation(tmp_path, monkeypatch, content = ON)
    world, seq = b.world, b.seq
    assert b.cached == 3
    world.pt.deallocate_pages(seq.allocated_pages)
    # Scatter the chain over the cache so that defrag has something to do
    order = list(range(world.pt.max_pages))
    np.random.default_rng(3).shuffle(order)
    by_hash = {p.phash: p for p in world.pt.all_pages}
    pages = [by_hash[h] for h in a.hashes]
    others = [p for p in world.pt.all_pages if p not in pages]
    for p, idx in zip(pages + others, order):
        p.page_index = idx
    world.pt.all_pages.sort(key = lambda p: p.page_index)
    world.pt.last_defrag_serial = -10 ** 9
    before = {h: by_hash[h].page_index for h in a.hashes}
    world.pt.defrag()
    assert any(by_hash[h].page_index != before[h] for h in a.hashes)
    world.pt.validate_pagetable([])
    seq2, cached, _ = start(world, b.ids, ON)
    assert cached == 3 and [p.phash for p in seq2.allocated_pages[:3]] == a.hashes
    assert world.pt.metrics["alloc_disk_pages"] == 3
    world.pt.validate_pagetable([types.SimpleNamespace(sequences = [seq2])])
    world.close()


def test_same_content_other_ids_in_one_process(monkeypatch):
    """In one process an image embedded again (the server's embedding cache let it go) has other ids and the
    same digest: the second job finds the first one's pages in VRAM, and the page table's validation (which
    compares a job's ids with its pages') holds. With the switch off the second job finds nothing past the
    text (fails)"""
    x = image(7, digests = ON)
    world = World(None, ON)
    ids = prompt(text(1, 100), x, text(2, 500))
    seq, _, _ = start(world, ids, ON)
    n = prefill(world, seq)
    world.pt.deallocate_pages(seq.allocated_pages)
    x2 = image(7, digests = ON)
    ids2 = prompt(text(1, 100), x2, text(2, 500))
    seq2, cached, stashed = start(world, ids2, ON)
    assert cached == n and stashed is not None
    world.pt.validate_pagetable([types.SimpleNamespace(sequences = [seq2])])


def test_embedding_without_digest_falls_back(tmp_path, monkeypatch):
    """With the switch on, an embedding made with digests off has no digest: its pages hash as upstream's and
    the chain from its first placeholder on is not persistable, while the pages before it (an image with a
    digest included) are persisted. With the switch off nothing past the text before the first image is
    persistable (fails)"""
    x = image(7, digests = ON)
    y = image(8, digests = False)
    assert y.content_digest is None
    ids = prompt(text(1, 100), x, text(2, 400), y, text(3, 300))
    world = World(tmp_path, ON)
    seq, _, _ = start(world, ids, ON)
    first_y = int((ids[0] >= y.first_index).nonzero()[0]) // PAGE
    assert seq.mm_first_page == first_y
    flags = [p.persistable for p in seq.allocated_pages]
    assert flags == [i < first_y for i in range(len(flags))]
    prefill(world, seq)
    off = hashes(ids, False)
    assert seq.page_hashes[first_y] != off[first_y]
    # The page with y's first rows hashes as upstream's over the chain before it
    assert seq.page_hashes[first_y] == upstream_hash(ids[:, first_y * PAGE:(first_y + 1) * PAGE], seq.page_hashes[first_y - 1])
    persist(world, seq)
    assert all(world.disk.has_page(h) for h in seq.page_hashes[:first_y])
    assert not any(world.disk.has_page(h) for h in seq.page_hashes[first_y:])
    # The eviction path (PageTable.evict through the CPU tier's store) keeps it off the disk too
    world.tier.store(seq.allocated_pages[first_y], world.pt.access_serial)
    world.disk.flush()
    assert world.disk.metrics["skipped_mm"] == 1 and not world.disk.has_page(seq.page_hashes[first_y])
    world.close()


def test_job_queues_with_the_generator_switch(monkeypatch):
    """The real Job: prepare_for_queue fixes the generator's switch on each sequence (Sequence.prepare is called
    as upstream calls it when the switch is off), and a job queued again with its answer (a requeue, or the
    next turn) hashes its prompt pages as the first one hashed them, the page its answer completed included.
    With the switch off the job hashes as upstream; the cross-process half fails"""
    def queue(world, ids, embeddings, content):
        gen = types.SimpleNamespace(
            pagetable = world.pt, max_total_tokens = world.pt.max_pages * PAGE, max_draft_tokens = 0,
            recurrent_cache = world.rc, recurrent_checkpoint_interval = PAGE, mtp_draft = False, max_batch_size = 4,
            ngram_match_min = 0, hybrid_draft_tokens = 0, ngram_corpus = None, padded_vocab_size = 32,
            model = types.SimpleNamespace(caps = {}), mm_content_hash = content,
        )
        job = JOB.Job(input_ids = ids, max_new_tokens = 2 * PAGE, embeddings = embeddings)
        job.prepare_for_queue(gen, 1)
        return job

    calls = []
    real_prepare = PT.Sequence.prepare
    monkeypatch.setattr(PT.Sequence, "prepare", lambda self, *a, **k: calls.append(k) or real_prepare(self, *a, **k))
    x = image(7, digests = None)
    ids = prompt(text(1, 300), x, text(2, 300))
    job = queue(World(None, False), ids, [x], False)
    assert calls == [{}] and not job.sequences[0].content_hash and job.sequences[0].page_hashes == hashes(ids, False)

    calls.clear()
    x = image(7, digests = ON)
    ids = prompt(text(1, 300), x, text(2, 40))
    world = World(None, ON)
    job = queue(world, ids, [x], ON)
    seq = job.sequences[0]
    assert calls == ([{"content_hash": True}] if ON else [{}]) and seq.content_hash == ON
    seq.allocate_pages(world.pt, world.rc)
    prefill(world, seq)
    answer = text(3, 2 * PAGE - 50)
    completed = generate(world, seq, answer)
    chain = seq.page_hashes + completed

    new_process(monkeypatch, skip = 99)
    x2 = image(7, digests = ON)
    ids2 = torch.cat((prompt(text(1, 300), x2, text(2, 40)), torch.tensor([answer], dtype = torch.long)), dim = -1)
    job2 = queue(World(None, ON), ids2, [x2], ON)
    # Pages 0-2 from the prompt, 3 (the image's last rows and the answer's first tokens) and 4 from the answer
    assert len(chain) == 5 and bool((ids[0, 3 * PAGE:] >= FIRST).any())
    assert job2.sequences[0].page_hashes[:5] == chain


# --------------------------------------------------------------------------------------------
#   The digest, the registry, the switch
# --------------------------------------------------------------------------------------------

def test_digest_covers_what_a_forward_reads():
    """Each of: one value of the embedding, its dtype, its shape, a deepstack embedding, the MRoPE grid, the
    merge size, the alignment fields changes the digest; the same tensors again, on another object, do not"""
    g = torch.Generator().manual_seed(1)
    e = torch.randn((8, HIDDEN), generator = g).half()
    ds = [torch.randn((8, HIDDEN), generator = g).half()]
    base = dict(deepstack_embeddings = ds, grid_thw = (1, 4, 8), mrope_merge_size = 2, align = 0, align_phase = 0,
                align_lead = 0)
    d = MM.embedding_digest(e, **base)
    assert len(d) == MM.DIGEST_SIZE and d == MM.embedding_digest(e.clone(), **dict(base, deepstack_embeddings = [ds[0].clone()]))
    e2 = e.clone()
    e2[3, 5] += 1
    variants = [
        MM.embedding_digest(e2, **base),
        MM.embedding_digest(e.float(), **base),
        MM.embedding_digest(e.view(4, 2 * HIDDEN), **base),
        MM.embedding_digest(e, **dict(base, deepstack_embeddings = None)),
        MM.embedding_digest(e, **dict(base, deepstack_embeddings = [ds[0] * 2])),
        MM.embedding_digest(e, **dict(base, grid_thw = (1, 8, 4))),
        MM.embedding_digest(e, **dict(base, mrope_merge_size = 1)),
        MM.embedding_digest(e, **dict(base, align = 4, align_phase = 1, align_lead = 2)),
        MM.embedding_digest(e, **dict(base, grid_thw = torch.tensor([1, 8, 4]))),
    ]
    assert d not in variants and len(set(variants)) == len(variants) - 1     # the grid as a tensor or a tuple
    assert variants[-1] == variants[5]


def test_registry_lookup_and_canonical_form(monkeypatch):
    """Ranges are found by bisection, ends exclusive; an id in no range has no digest; the canonical form is
    -1 - row for a placeholder and runs of equal digests; the allocator going back (a new process in a test)
    clears what was registered"""
    reg = MM.MMDigestRegistry()
    alloc = MM.MMTokenAllocator()
    d1, d2 = b"1" * 16, b"2" * 16
    a = reg.allocate(alloc, 10, d1)
    alloc.allocate(5)                  # an embedding made without a digest
    b = reg.allocate(alloc, 20, d2)
    assert (a, b) == (FIRST, FIRST + 15) and len(reg) == 2
    assert reg.lookup(FIRST) == (d1, 0) and reg.lookup(FIRST + 9) == (d1, 9)
    assert reg.lookup(FIRST + 10) is None and reg.lookup(FIRST + 14) is None and reg.lookup(FIRST - 1) is None
    assert reg.lookup(FIRST + 15) == (d2, 0) and reg.lookup(FIRST + 35) is None
    ids = np.array([7, FIRST + 8, FIRST + 9, FIRST + 15, 3, FIRST + 16], dtype = np.int64)
    canon, runs = reg.canonical(ids)
    assert canon.tolist() == [7, -9, -10, -1, 3, -2] and runs == [(2, d1), (2, d2)]
    assert reg.canonical(np.array([1, FIRST + 12], dtype = np.int64)) is None
    assert reg.first_undigested(np.array([1, FIRST + 3, FIRST + 12, FIRST + 40], dtype = np.int64)) == 2
    assert reg.first_undigested(np.array([1, FIRST + 3], dtype = np.int64)) is None
    with pytest.raises(AssertionError):
        reg.allocate(alloc, 1, b"short")
    reg.allocate(MM.MMTokenAllocator(), 4, d1)
    assert len(reg) == 1 and reg.lookup(FIRST + 15) is None


def test_switch(monkeypatch):
    """Off by default; EXL3_MM_CONTENT_HASH=1 turns the digests on, set_content_digests overrides either way;
    an embedding keeps what it was made with; one made through the tensor-parallel import has none"""
    assert not MM.content_digests()
    assert image(1, digests = None).content_digest is None and len(MM.global_registry) == 0
    monkeypatch.setenv(ENV, "1")
    assert MM.content_digests()
    x = image(1, digests = None)
    assert x.content_digest is not None and MM.global_registry.lookup(x.first_index) == (x.content_digest, 0)
    MM.set_content_digests(False)
    assert not MM.content_digests()
    monkeypatch.setenv(ENV, "0")
    MM.set_content_digests(None)
    assert not MM.content_digests() and x.content_digest is not None
    imp = {"metadata": {}, "full_length": 3, "mm_length": 1, "first_index": FIRST, "last_index": FIRST + 1,
           "text_alias": "a", "grid_thw": None, "mrope_merge_size": None, "embeddings": None,
           "deepstack_embeddings": None}
    assert MM.MMEmbedding(imp = imp).content_digest is None


def test_generator_switch_reads_the_environment(monkeypatch):
    """Generator(mm_content_hash) and EXL3_MM_CONTENT_HASH, read off the source: the argument or the variable
    turns the generator's switch on, and with it the digests for the process"""
    path = os.path.join(SRC, "generator", "generator.py")
    tree = ast.parse(open(path).read())
    init = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                and any(a.arg == "mm_content_hash" for a in n.args.args))
    arg = next(a for a in init.args.args if a.arg == "mm_content_hash")
    default = init.args.defaults[len(init.args.defaults) - len(init.args.args) + init.args.args.index(arg)]
    assert isinstance(default, ast.Constant) and default.value is False
    src = ast.get_source_segment(open(path).read(), init)
    assert 'os.environ.get("EXL3_MM_CONTENT_HASH"' in src and "set_content_digests(True)" in src


def test_every_page_hash_goes_through_the_helper():
    """Read off the sources: upstream's page hash is reached only by the content helper (for its fallback)
    and by the sites that choose between the two on the sequence's or the generator's switch. A new site that
    hashed page ids on its own would hash image pages one way and the rest of the library another; take it
    through content_hash_checksum (run this after merging a newer upstream)"""
    names = {"tensor_hash_checksum", "_tensor_blake2b_checksum"}
    found = set()
    for dirpath, _, files in os.walk(SRC):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            tree = ast.parse(open(path).read())
            rel = os.path.relpath(path, SRC)

            def visit(node, scope):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        visit(child, scope + [child.name])
                        continue
                    if isinstance(child, ast.Name) and child.id in names and isinstance(child.ctx, ast.Load):
                        found.add((rel, ".".join(scope)))
                    visit(child, scope)
            visit(tree, [])
    assert found == {
        ("generator/pagetable.py", ""),                                     # tensor_hash_checksum = ...
        ("generator/pagetable.py", "content_hash_checksum"),
        ("generator/pagetable.py", "Sequence.prepare"),
        ("generator/pagetable.py", "PageTable.checksum"),
        ("generator/job.py", "Job.hash_completed_page"),
    }, found
