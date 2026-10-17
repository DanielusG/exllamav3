"""
The checkpoint that follows a generating job (Generator.tail_checkpoint_interval,
Job.is_tail_boundary, Job.maybe_stash_recurrent, RecurrentCache.discard) on 1.6.2.

No GPU or model is required. The job's real checkpoint rules and the real checkpoint cache run
over a stub job that keeps its position and its pages the way Job does: the prompt is ingested
in chunks, then tokens are generated one at a time, and after every move the generator's offer
(maybe_stash_recurrent) is made, as Generator.recurrent_checkpoint does. The disk tier is
upstream's DiskPageCache, writer thread and index included, over a stand-in for the CPU page
tier (pages are written straight from a host buffer: only their presence on disk matters to a
checkpoint).

1.6.2 points: a discarded stash goes back through RecurrentCache._release (host pool, the
tensor-parallel handles of both sides of a pair, a release deferred while the disk tier still
writes the stash); what the disk tier holds of a discarded checkpoint (a candidate, a queued
write, a written copy, a write under way); the keys through upstream's rekey; the "tail" flag
through the disk round trip; the generator's checkpoint offer, which passes over a job in
flight (staggered decode) or with a prompt chunk under way (prefill thread, pipeline).

When the compiled extension is not installed the source files under test are loaded on their
own, with the rest of the package stubbed.
"""

import ast
import importlib.util
import inspect
import os
import sys
import threading
import types
from unittest.mock import Mock

import pytest
import torch

import test_stagger_decode as TOY

pytestmark = pytest.mark.nogpu

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = TOY.GEN


def _load_real():
    sys.path.insert(0, REPO)
    try:
        from exllamav3.generator import job as job_mod
        from exllamav3.cache import recurrent as rec_mod
        from exllamav3.generator import disk_cache as disk_mod
        return job_mod, rec_mod, disk_mod
    finally:
        sys.path.remove(REPO)


def _load_stubbed():
    pkg = "exl3_tailckpt_stub"
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

    def functions(name, names, **attrs):
        """A stub of module `name` with the real source of the functions `names` (the disk tier's
        checkpoint (de)serialization lives in model_tp_fn, which imports the extension)"""
        path = os.path.join(src, *name.split(".")) + ".py"
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
    stub("ext", exllamav3_ext = types.SimpleNamespace())
    stub("util.memory", malloc_trim = lambda: None, check_host_memory = lambda *a, **k: None)
    stub("util.tensor", SeqTensor = _Any)
    stub("util.pinned_arena", PinnedArena = _Any)
    sys.modules[f"{pkg}.util"].profile_opt = lambda *a, **k: None
    sys.modules[f"{pkg}.tokenizer"].MMEmbedding = _Any
    stub("model.forward_cursor", ForwardCursor = _Any)
    stub("generator.pagetable", Sequence = _Any, tensor_hash_checksum = lambda ids, prev: b"", random_hash = lambda: b"")
    stub("generator.filter", Filter = _Any)
    stub("generator.loop_detect", LoopDetector = _Any)
    stub("generator.sampler", Sampler = _Any, DefaultSampler = _Any)
    unused = lambda *a, **k: None
    functions("model.model_tp_fn", ("pack_stash", "unpack_stash"),
              mp_disk_layout = unused, mp_disk_set_layout = unused, mp_disk_write = unused, mp_disk_read = unused,
              mp_disk_state_describe = unused, mp_disk_state_write = unused, mp_disk_state_read = unused)
    real("constants")
    real("version")
    rec = real("cache.recurrent")
    real("generator.cpu_cache")
    return real("generator.job"), rec, real("generator.disk_cache")


try:
    JOB, REC, DISK = _load_real()
except Exception:
    JOB, REC, DISK = _load_stubbed()

Job = JOB.Job
RecurrentCache = REC.RecurrentCache
PAGE = JOB.PAGE_SIZE
CKPT = 100      # bytes of a pretend checkpoint
TAIL_STATS = {"stashes": 0, "replaced": 0, "t_stash": 0.0, "disk_withdrawn": 0, "disk_dropped": 0, "disk_left": 0}


# --------------------------------------------------------------------------------------------
#   Stub job
# --------------------------------------------------------------------------------------------

class State:
    """What a job's recurrent state is to the cache: a size and a stash in 1.6.2's paired shape
    (PairedState.stash: the target side; the MTP head has no recurrent state). The tensor comes
    from the stash pool, as a real stash's do, so a release can be seen."""
    checkpoint_size = CKPT

    def __init__(self, job):
        self.job = job

    @property
    def position(self):
        return self.job.sequences[0].kv_position

    def stash(self):
        t = REC.host_pool.take((4,), torch.float32)
        t.fill_(self.position)
        return {"position": self.position, "checkpoint_size": CKPT, "target": {"layer.0": t}}

    def free(self):
        pass


class Ids(list):
    def torch_slice(self, a, b):
        return torch.tensor(self[a:b], dtype = torch.long).view(1, -1)


class Page:
    def __init__(self, phash):
        self.phash = phash
        self.prev_hash = None
        self.kv_position = 0
        self.can_revert = True
        self.access_serial = 0

    def update_hash(self, h):
        self.phash = h


def key(chain: str, page: int) -> bytes:
    """A page's content hash (RecurrentCache and the disk tier take a hash whose first 8 bytes are not all
    zero for one)"""
    return f"{chain}:{page}".encode()


def random_key(n: int) -> bytes:
    """A page still under its random hash (a page that completed during a banned-string hold)"""
    return bytes(8) + f"r{n}".encode()


def make_gen(tail = PAGE, interval = 4096, chunk = 2048, room = 100, disk = None):
    cache = RecurrentCache(types.SimpleNamespace(loaded_tp = False), max_size = room * CKPT)
    cache.disk_tier = disk
    return types.SimpleNamespace(
        recurrent_cache = cache,
        recurrent_checkpoint_interval = interval,
        recurrent_checkpoint_interval_pp = 32768,
        max_chunk_size = chunk,
        tail_checkpoint_interval = tail,
        tail_stats = dict(TAIL_STATS),
    )


def make_job(gen, chain: str, prompt: int, cached_pages: int = 0, resumed = None):
    """A job whose prompt is `prompt` tokens long (plus the token generation starts from), with
    the first cached_pages pages found in the cache and the state resumed from `resumed` (the
    stashed dict allocate_pages got, through the real Job.allocate_pages), None for a new state."""
    job = Job.__new__(Job)
    job.generator = gen
    job.sequences = [types.SimpleNamespace(
        kv_position = cached_pages * PAGE,
        sequence_ids = Ids([0] * (prompt + 1)),
        allocated_pages = [Page(key(chain, i)) for i in range(4096)],
    )]
    for p in job.sequences[0].allocated_pages[:cached_pages]:
        p.kv_position = PAGE
    job.cached_pages = cached_pages
    job.new_tokens = 0
    job.recurrent_state = State(job)
    job.last_recurrent_checkpoint_pos = None
    job.tail_checkpoint_key = None
    job.prompt_tail_key = None
    job.resumed_from_tail = False
    job.pagetable = types.SimpleNamespace(referenced_pages = {}, unreferenced_pages = {})
    if resumed is not None:
        resume(job, resumed, cached_pages)
    return job


def resume(job, stashed, cached_pages):
    """Job.allocate_pages, the real one, over a sequence whose allocation found `cached_pages` pages and
    the checkpoint `stashed` at the last of them"""
    seq, gen = job.sequences[0], job.generator
    seq.allocate_pages = lambda pt, rc, protected: (len(seq.allocated_pages), cached_pages, 0, stashed)
    gen.new_recurrent_state = lambda: State(job)
    gen.recurrent_state_from_stashed = lambda s, position: State(job)
    job.all_unique_hashes = []
    job.cached_pages = 0
    job.total_pages = 0
    job.non_sequential_pages = 0
    Job.allocate_pages(job)
    assert job.cached_pages == cached_pages and job.last_recurrent_checkpoint_pos == cached_pages * PAGE


def advance(job, to: int):
    seq = job.sequences[0]
    seq.kv_position = to
    for i in range(to // PAGE):
        seq.allocated_pages[i].kv_position = PAGE


def ingest(job):
    """The prompt in chunks, as Job.prefill: a pass of its own up to the last page boundary
    (offered with the page size), then the rest; the generator's offer after every chunk."""
    gen, seq = job.generator, job.sequences[0]
    cache = gen.recurrent_cache
    end = len(seq.sequence_ids) - 1
    last_page = end // PAGE * PAGE
    while seq.kv_position < end:
        start = seq.kv_position
        stop = min((start + gen.max_chunk_size) // PAGE * PAGE, end)
        if start < last_page <= stop:
            advance(job, last_page)
            job.maybe_stash_recurrent(cache, PAGE)
        else:
            advance(job, stop if stop > start else end)
        job.maybe_stash_recurrent(cache)


def step(job):
    """One token generated, no offer"""
    seq = job.sequences[0]
    seq.sequence_ids.append(0)
    job.new_tokens += 1
    advance(job, seq.kv_position + 1)


def generate(job, tokens: int, after = None):
    """One token at a time: the sequence grows, the position follows, the offer is made (then
    `after(job)`, the disk tier's write-behind in some tests)."""
    cache = job.generator.recurrent_cache
    held = []
    for _ in range(tokens):
        step(job)
        job.maybe_stash_recurrent(cache)
        held.append(len(cache))
        if after is not None:
            after(job)
    return held


def positions(cache) -> list:
    return sorted(v["position"] for v in cache.values())


# --------------------------------------------------------------------------------------------
#   The disk tier
# --------------------------------------------------------------------------------------------

class Tier:
    """The CPU page tier as DiskPageCache sees it when only checkpoints and their anchor pages matter: one
    segment, no tensor-parallel group, no entries in system RAM; a page is written from a host buffer of
    its own (DiskPageCache.write_page takes the slot, the writer thread reads it)."""

    def __init__(self):
        self.slab_size = 512
        self.segments = [(torch.zeros((1, 512), dtype = torch.uint8), 0, (512,), torch.uint8)]
        self.tp_groups = []
        self.max_slots = 64
        self.entries = {}
        self.slab = torch.zeros(512, dtype = torch.uint8)

    def __contains__(self, h):
        return False

    def __len__(self):
        return 0

    def slab_view(self, slot):
        return self.slab


def open_disk(root, min_state_distance = 0):
    return DISK.DiskPageCache(str(root), [], Tier(), 1 << 30, fingerprint = "tail-checkpoint-test",
                              min_state_distance = min_state_distance, min_free = 0)


def land(disk, chain: str, pages):
    """Write pages of a chain to disk and wait for them, then write the checkpoints whose page has landed
    (the main thread does that at its next call into the tier: drain_ready_states)"""
    for i in pages:
        tokens = torch.full((PAGE,), i, dtype = torch.long)
        disk.write_page(key(chain, i), key(chain, i - 1) if i else None, tokens, 0, [])
    disk.flush()
    disk.drain_ready_states()
    disk.flush()


def write_behind(disk, chain: str):
    """Generator.write_behind's rule for one job: every complete page but the last two before the
    position (they are queued on the way, not each token: land() waits for them)"""
    done = set()

    def after(job):
        upto = max(0, job.sequences[0].kv_position // PAGE - 2)
        new = [i for i in range(upto) if i not in done]
        if new:
            land(disk, chain, new)
            done.update(new)
    return after


def pooled(t) -> bool:
    return any(x is t for lst in REC.host_pool.free.values() for x in lst)


# --------------------------------------------------------------------------------------------
#   The rules (ported from the v1.5.4 fork)
# --------------------------------------------------------------------------------------------

def test_off_nothing_is_left_behind_an_answer():
    gen = make_gen(tail = 0)
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    at_prompt = positions(gen.recurrent_cache)
    assert at_prompt[-1] == 4864                    # the prompt's last page boundary
    generate(job, 1000)
    assert positions(gen.recurrent_cache) == at_prompt
    assert gen.tail_stats == TAIL_STATS
    assert job.tail_checkpoint_key is None and job.prompt_tail_key is None
    assert not any(v.get("tail") for v in gen.recurrent_cache.values())


def test_one_checkpoint_follows_the_answer_page_by_page():
    gen = make_gen()
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    at_prompt = positions(cache)
    held = generate(job, 1000)                      # positions 5001 .. 6000: pages end at 5120 .. 5888
    assert positions(cache) == at_prompt + [5888]
    assert max(held) == len(at_prompt) + 1          # never more than one beyond the prompt's
    assert cache[key("a", 22)]["tail"] is True and cache[key("a", 22)]["position"] == 5888
    assert not any(v.get("tail") for k, v in cache.items() if k != key("a", 22))
    assert gen.tail_stats["stashes"] == 4 and gen.tail_stats["replaced"] == 3
    assert cache.metrics["stash_replaced"] == 3 and cache.metrics["stash_evictions"] == 0
    assert job.tail_checkpoint_key == key("a", 22)


def test_an_answer_shorter_than_a_page_leaves_nothing():
    gen = make_gen()
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    at_prompt = positions(gen.recurrent_cache)
    generate(job, 100)
    assert positions(gen.recurrent_cache) == at_prompt and gen.tail_stats["stashes"] == 0


def test_the_position_generation_starts_from_is_not_a_tail_boundary():
    # a prompt that ends on a page boundary: its last page has its checkpoint already, and the
    # job has generated nothing yet when the generator makes its offer there
    gen = make_gen()
    job = make_job(gen, "a", prompt = 5120)
    ingest(job)
    assert positions(gen.recurrent_cache)[-1] == 5120 and gen.tail_stats["stashes"] == 0
    assert not job.is_tail_boundary()


def test_a_regular_checkpoint_takes_the_tail_one_s_place_and_stays():
    gen = make_gen(interval = 1024)
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 2100)         # nothing cached: the regular grid is 1024, 2048, 3072 ...
    ingest(job)                                     # one chunk up to the last page boundary, 2048
    assert positions(cache) == [2048]
    generate(job, 900)                              # up to 3000: tail at 2304, 2560, 2816
    assert positions(cache) == [2048, 2816] and cache[key("a", 10)]["tail"] is True
    generate(job, 200)                              # 3200: the regular one at 3072 takes its place
    assert positions(cache) == [2048, 3072]
    assert not cache[key("a", 11)].get("tail")
    assert job.tail_checkpoint_key is None
    generate(job, 200)                              # 3400: the tail one starts again after the regular one
    assert positions(cache) == [2048, 3072, 3328]
    assert cache[key("a", 12)]["tail"] is True


def test_a_job_that_resumed_from_a_tail_checkpoint_gives_up_its_prompt_s_last_page_one():
    gen = make_gen()
    cache = gen.recurrent_cache
    first = make_job(gen, "a", prompt = 5000)
    ingest(first)
    generate(first, 1000)
    left = cache[key("a", 22)]                      # at 5888, where the next turn resumes
    before = len(cache)

    turn = make_job(gen, "a", prompt = 7000, cached_pages = 23, resumed = left)
    assert turn.resumed_from_tail
    ingest(turn)
    assert turn.prompt_tail_key == key("a", 26)     # 6912, the new prompt's last page boundary
    assert len(cache) == before + 1
    generate(turn, 600)                             # 7001 .. 7600: pages end at 7168, 7424
    # one checkpoint more than before the turn, as without the feature: the answer's instead of the prompt's
    assert len(cache) == before + 1
    assert key("a", 26) not in cache and cache[key("a", 28)]["position"] == 7424
    assert key("a", 22) in cache                    # what the turn resumed from is still the fallback
    assert turn.prompt_tail_key is None


def test_a_job_that_did_not_keeps_its_prompt_s_last_page_one():
    gen = make_gen()
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    generate(job, 600)
    assert key("a", 18) in cache and job.prompt_tail_key == key("a", 18)       # 4864 stays
    assert cache[key("a", 20)]["tail"] is True                               # 5376 beside it
    # resumed from a checkpoint without the flag (the prompt's last page, a regular one): the same
    turn = make_job(gen, "a", prompt = 7000, cached_pages = 19, resumed = cache[key("a", 18)])
    assert not turn.resumed_from_tail


def test_no_room_is_ever_made_for_a_tail_checkpoint_that_moves():
    # a cache that holds exactly what is there after the first tail checkpoint: every later one
    # must find its room in the one it replaces, not in an eviction (which a disk tier would write)
    gen = make_gen(room = 3)
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    assert len(cache) == 2
    generate(job, 300)
    assert len(cache) == 3 and cache.metrics["stash_evictions"] == 0
    generate(job, 2000)
    assert len(cache) == 3 and cache.metrics["stash_evictions"] == 0
    assert positions(cache)[-1] == 7168


def test_discard_is_not_an_eviction_and_tolerates_a_missing_key():
    gen = make_gen()
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 600)
    ingest(job)
    k = key("a", 1)
    assert k in cache and cache.current_size == CKPT
    assert cache.discard(k) is True
    assert k not in cache and cache.current_size == 0
    assert cache.metrics["stash_replaced"] == 1 and cache.metrics["stash_evictions"] == 0
    assert cache.discard(k) is False and cache.metrics["stash_replaced"] == 1


def test_every_other_page_with_a_longer_interval():
    gen = make_gen(tail = 512)
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    generate(job, 1000)
    assert positions(gen.recurrent_cache)[-1] == 5632 and gen.tail_stats["stashes"] == 2


def test_a_generator_without_the_switch_is_upstream_s():
    # objects that predate the feature (a toy generator in another test) have neither attribute
    gen = make_gen()
    del gen.tail_checkpoint_interval, gen.tail_stats
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    at_prompt = positions(gen.recurrent_cache)
    generate(job, 1000)
    assert positions(gen.recurrent_cache) == at_prompt


def test_switched_off_while_a_job_runs_the_next_regular_checkpoint_replaces_the_tail_one():
    gen = make_gen(interval = 1024)
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 2100)
    ingest(job)
    generate(job, 300)                              # 2400: tail at 2304
    assert cache[key("a", 8)]["tail"] is True
    gen.tail_checkpoint_interval = 0
    generate(job, 500)                              # 2900: no tail checkpoint moves any more
    assert positions(cache) == [2048, 2304]
    generate(job, 200)                              # 3100: the regular one at 3072, and the tail one goes
    assert positions(cache) == [2048, 3072] and job.tail_checkpoint_key is None


# --------------------------------------------------------------------------------------------
#   Whose checkpoint it is: never one the cache already had
# --------------------------------------------------------------------------------------------

def test_a_checkpoint_already_in_ram_under_the_key_is_not_taken_over():
    gen = make_gen()
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    other = make_job(gen, "a", prompt = 5000)       # the same text generated again, by another job
    advance(other, 5376)
    cache.put(key("a", 20), State(other))
    theirs = cache[key("a", 20)]
    generate(job, 500)                              # 5500: crosses 5120 (tail), 5376 (theirs)
    assert key("a", 19) not in cache                # its own tail at 5120 went when it moved on
    assert job.tail_checkpoint_key is None          # 5376 is not its own
    generate(job, 300)                              # 5800: tail at 5632
    assert cache.get(key("a", 20)) is theirs and not theirs.get("tail")
    assert cache[key("a", 21)]["tail"] is True and gen.tail_stats["replaced"] == 1


def test_a_checkpoint_on_disk_under_the_key_is_neither_stashed_again_nor_removed(tmp_path):
    disk = open_disk(tmp_path)
    try:
        gen = make_gen(disk = disk)
        cache = gen.recurrent_cache
        land(disk, "a", list(range(0, 19)) + [20, 21, 22, 23])     # (page 19 not: its tail is a candidate)
        st = {"position": 5376, "checkpoint_size": CKPT, "target": {"layer.0": torch.ones(4)}}
        assert disk.write_state(key("a", 20), st)
        disk.flush()
        job = make_job(gen, "a", prompt = 5000)
        ingest(job)
        generate(job, 500)                          # 5500: tail at 5120, then 5376 is on disk already
        assert key("a", 20) not in cache and job.tail_checkpoint_key is None
        generate(job, 300)                          # 5800: tail at 5632; nothing of 5376 is touched
        disk.flush()
        assert disk.has_state(key("a", 20)) and os.path.exists(disk._state_path(key("a", 20)))
        assert gen.tail_stats["disk_withdrawn"] == 1 and gen.tail_stats["disk_dropped"] == 0
    finally:
        disk.close()


def test_a_replay_that_comes_back_to_its_own_tail_checkpoint_keeps_it():
    # (a banned-string rewind restores an older stash and replays: the offer can come again at the page of
    # the job's own tail checkpoint, which is then in the cache already, and still the job's)
    gen = make_gen()
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    generate(job, 130)                              # tail at 5120
    advance(job, 5120)
    job.last_recurrent_checkpoint_pos = 4864        # restored from the prompt's last page, replayed to 5120
    job.maybe_stash_recurrent(cache)
    assert job.tail_checkpoint_key == key("a", 19) and gen.tail_stats["replaced"] == 0
    generate(job, 256)
    assert key("a", 19) not in cache and gen.tail_stats["replaced"] == 1


def test_a_prompt_s_last_page_checkpoint_it_did_not_stash_is_not_given_up():
    gen = make_gen()
    cache = gen.recurrent_cache
    first = make_job(gen, "a", prompt = 5000)
    ingest(first)
    generate(first, 1000)
    left = cache[key("a", 22)]
    # a checkpoint at the new prompt's last page boundary is there already (another request with
    # the same prompt)
    other = make_job(gen, "a", prompt = 7000)
    advance(other, 6912)
    cache.put(key("a", 26), State(other))
    turn = make_job(gen, "a", prompt = 7000, cached_pages = 23, resumed = left)
    ingest(turn)
    assert turn.prompt_tail_key is None
    generate(turn, 600)
    assert key("a", 26) in cache


# --------------------------------------------------------------------------------------------
#   1.6.2: the release path, paired stashes, tensor parallelism
# --------------------------------------------------------------------------------------------

def test_a_discarded_stash_goes_back_to_the_pool():
    # through RecurrentCache._release: the buffers return to the stash pool before the next stash is
    # taken, which reuses them (the pool's point: no allocation churn for a checkpoint that moves)
    gen = make_gen()
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    generate(job, 130)                              # tail at 5120
    t = cache[key("a", 19)]["target"]["layer.0"]
    assert not pooled(t)
    generate(job, 256)                              # moved on to 5376
    assert key("a", 19) not in cache
    assert cache[key("a", 20)]["target"]["layer.0"] is t and torch.equal(t, torch.full((4,), 5376.0))


def test_a_paired_stash_frees_both_sides_in_tensor_parallel_mode():
    calls = []

    def model(side):
        return types.SimpleNamespace(loaded_tp = True,
                                     tp_dispatch_all = lambda fn, args: calls.append((side, fn.__name__, args)))

    cache = RecurrentCache(model("target"), 10 * CKPT, draft_model = model("draft"))
    k1, k2 = key("p", 1), key("p", 2)
    cache[k1] = {"position": 256, "checkpoint_size": 3 * CKPT, "target": {"tp_handle": 7}, "draft": {"tp_handle": 8}}
    cache[k2] = {"position": 512, "checkpoint_size": 2 * CKPT, "target": {"tp_handle": 9}}
    assert cache.update_total_size() == 5 * CKPT
    assert cache.discard(k1)
    assert calls == [("target", "mp_cache_recurrent_del", (id(cache), 7)),
                     ("draft", "mp_cache_recurrent_del", (id(cache), 8))]
    assert cache.current_size == 2 * CKPT and list(cache) == [k2]


# --------------------------------------------------------------------------------------------
#   1.6.2: the disk tier
# --------------------------------------------------------------------------------------------

def test_a_replaced_candidate_is_withdrawn_and_never_written(tmp_path):
    disk = open_disk(tmp_path)
    try:
        gen = make_gen(disk = disk)
        job = make_job(gen, "a", prompt = 5000)
        ingest(job)
        generate(job, 1000, after = write_behind(disk, "a"))
        # the tail moved on from each page before write-behind took that page: every one of them was a
        # candidate, withdrawn, and the release did not count it as dropped
        assert gen.tail_stats["replaced"] == 3 and gen.tail_stats["disk_withdrawn"] == 3
        assert disk.metrics["state_candidates_dropped"] == 0
        land(disk, "a", range(0, 23))               # every page, the tail's included, on disk now
        assert sorted(disk.states) == sorted([key("a", 15), key("a", 18), key("a", 22)])
        assert not any(key("a", p) in disk.state_candidates for p in (19, 20, 21))
    finally:
        disk.close()


def test_a_queued_write_is_taken_back(tmp_path):
    disk = open_disk(tmp_path)
    try:
        gen = make_gen(disk = disk)
        cache = gen.recurrent_cache
        land(disk, "a", range(0, 24))               # the answer's pages are on disk already
        job = make_job(gen, "a", prompt = 5000)
        ingest(job)
        disk.flush()
        size_before = disk.total_size
        generate(job, 119)                          # 5119
        # The writer takes ops under this lock: the tail's write at 5120 is still queued when it moves on
        with disk._cv:
            generate(job, 1)                        # 5120: the tail's write is queued (its page is on disk)
            assert key("a", 19) in disk.pending_states
            stash = cache[key("a", 19)]
            generate(job, 256)                      # 5376: moved on
            assert key("a", 19) not in disk.pending_states and key("a", 20) in disk.pending_states
            assert id(stash) not in disk._deferred_release
            # released at once, and the next stash took its buffers
            assert cache[key("a", 20)]["target"]["layer.0"] is stash["target"]["layer.0"]
        disk.flush()
        assert not disk.has_state(key("a", 19)) and disk.has_state(key("a", 20))
        assert not os.path.exists(disk._state_path(key("a", 19)))
        assert gen.tail_stats["disk_withdrawn"] == 1 and gen.tail_stats["disk_left"] == 0
        # the withdrawn write's reservation went with it
        assert disk.total_size == size_before + disk.states[key("a", 20)]["size"]
    finally:
        disk.close()


def test_a_written_copy_is_removed(tmp_path):
    disk = open_disk(tmp_path)
    try:
        gen = make_gen(disk = disk)
        land(disk, "a", range(0, 24))
        job = make_job(gen, "a", prompt = 5000)
        ingest(job)
        generate(job, 130)                          # tail at 5120, written at once (its page is on disk)
        disk.flush()
        assert disk.has_state(key("a", 19)) and os.path.exists(disk._state_path(key("a", 19)))
        generate(job, 256)                          # 5386: moved on to 5376
        disk.flush()
        assert not disk.has_state(key("a", 19)) and not os.path.exists(disk._state_path(key("a", 19)))
        assert disk.has_state(key("a", 20))
        assert gen.tail_stats["disk_dropped"] == 1 and gen.tail_stats["replaced"] == 1
    finally:
        disk.close()


def test_a_write_under_way_is_left_and_the_release_waits_for_it(tmp_path):
    disk = open_disk(tmp_path)
    try:
        gen = make_gen(disk = disk)
        cache = gen.recurrent_cache
        land(disk, "a", range(0, 24))
        job = make_job(gen, "a", prompt = 5000)
        ingest(job)
        disk.flush()
        started, go = threading.Event(), threading.Event()
        write = disk._do_write_state

        def slow(conn, phash, *a):
            if phash == key("a", 19):
                started.set()
                assert go.wait(10)
            return write(conn, phash, *a)

        disk._do_write_state = slow
        generate(job, 130)                          # tail at 5120: its write starts and stops half way
        assert started.wait(10)
        stash = cache[key("a", 19)]
        generate(job, 256)                          # moved on: the write cannot be taken back
        assert gen.tail_stats["disk_left"] == 1 and id(stash) in disk._deferred_release
        assert not pooled(stash["target"]["layer.0"]), "buffers back in the pool while the writer reads them"
        go.set()
        disk.flush()
        assert pooled(stash["target"]["layer.0"]) and not disk._deferred_release
        # the file is what the stash held (nothing reused its buffers while it was written)
        back = disk.read_state(key("a", 19))
        assert back["position"] == 5120 and torch.equal(back["target"]["layer.0"], torch.full((4,), 5120.0))
    finally:
        go.set()
        disk.close()


def test_an_evicted_tail_copy_on_disk_goes_with_the_next_move(tmp_path):
    disk = open_disk(tmp_path)
    try:
        gen = make_gen(disk = disk, room = 2)      # the prompt's two checkpoints fill it
        cache = gen.recurrent_cache
        land(disk, "a", range(0, 24))
        job = make_job(gen, "a", prompt = 5000)
        ingest(job)
        generate(job, 130)                          # tail at 5120: the oldest goes to make room
        disk.flush()
        # LRU pressure takes the tail checkpoint out of RAM (a third checkpoint of another job)
        other = make_job(gen, "b", prompt = 300)
        advance(other, 256)
        land(disk, "b", range(0, 1))
        cache.put(key("b", 0), State(other))
        cache.put(key("b", 0) + b"x", State(other))
        assert key("a", 19) not in cache and disk.has_state(key("a", 19))
        replaced = cache.metrics["stash_replaced"]
        generate(job, 256)                          # moved on: its copy on disk is this job's, it goes
        disk.flush()
        assert not disk.has_state(key("a", 19)) and gen.tail_stats["disk_dropped"] == 1
        # not counted as a replacement: nothing left RAM for it (the tracker of the server pairs the two)
        assert cache.metrics["stash_replaced"] == replaced and gen.tail_stats["replaced"] == 0
    finally:
        disk.close()


def test_the_idle_flush_writes_the_tail_a_job_left_and_none_it_replaced(tmp_path):
    disk = open_disk(tmp_path, min_state_distance = 4096)
    try:
        gen = make_gen(disk = disk)
        cache = gen.recurrent_cache
        job = make_job(gen, "a", prompt = 5000)
        ingest(job)
        generate(job, 1000, after = write_behind(disk, "a"))
        # The idle flush (Generator.flush_cache_to_disk, the real one): every complete page goes to disk
        # first, then the checkpoints anchored there, the last of each chain whatever its spacing
        land(disk, "a", range(0, 23))
        fake = types.SimpleNamespace(
            disk_page_cache = disk, cpu_page_cache = disk.cpu_tier, recurrent_cache = cache,
            pagetable = types.SimpleNamespace(unreferenced_pages = {}, referenced_pages = {}, max_pages = 64),
        )
        fake._queue_pages_to_disk = lambda pages, budget: (0, False)
        queued = GEN.Generator.flush_cache_to_disk(fake, wait = True)
        assert disk.has_state(key("a", 22)), queued        # the tail, 1024 tokens past the one before it
        assert not any(disk.has_state(key("a", p)) for p in (19, 20, 21))
        # min_state_distance 4096: of the prompt's, 4096 is kept and 4864 pruned (768 past it)
        assert disk.has_state(key("a", 15)) and not disk.has_state(key("a", 18))
    finally:
        disk.close()


@pytest.mark.parametrize("paired", [True, False])
def test_the_tail_flag_survives_the_disk_round_trip(tmp_path, paired):
    disk = open_disk(tmp_path)
    land(disk, "a", range(0, 23))
    gen = make_gen(disk = disk)
    job = make_job(gen, "a", prompt = 5000)
    if not paired:
        # a bare state's stash (one unnamed side for the disk tier)
        job.recurrent_state.stash = lambda: {"position": job.sequences[0].kv_position, "checkpoint_size": CKPT,
                                             "layer.0": torch.ones(4)}
    ingest(job)
    generate(job, 1000)                             # the tail at 5888 is written at once: its page is on disk
    disk.flush()
    assert disk.has_state(key("a", 22))
    disk.close()

    # a new process: the checkpoint comes from disk only
    disk2 = open_disk(tmp_path)
    try:
        gen2 = make_gen(disk = disk2)
        stashed = gen2.recurrent_cache.get_stashed(key("a", 22))
        assert stashed is not None and stashed.get("tail") is True and stashed["position"] == 5888
        # a regular checkpoint of the same chain has no flag after the round trip either
        assert not gen2.recurrent_cache.get_stashed(key("a", 18)).get("tail")
        turn = make_job(gen2, "a", prompt = 7000, cached_pages = 23, resumed = stashed)
        assert turn.resumed_from_tail
    finally:
        disk2.close()


def test_a_flag_set_after_the_put_would_not_reach_the_file(tmp_path):
    # Why put() takes the flag: a checkpoint whose page is on disk is laid out for writing inside put()
    disk = open_disk(tmp_path)
    try:
        land(disk, "a", range(0, 2))
        cache = RecurrentCache(types.SimpleNamespace(loaded_tp = False), 10 * CKPT)
        cache.disk_tier = disk
        job = make_job(make_gen(), "a", prompt = 600)
        advance(job, 256)
        cache.put(key("a", 0), State(job))
        cache[key("a", 0)]["tail"] = True
        disk.flush()
        assert not disk.read_state(key("a", 0)).get("tail")
    finally:
        disk.close()


# --------------------------------------------------------------------------------------------
#   1.6.2: the keys through upstream's rekey
# --------------------------------------------------------------------------------------------

def hold_and_hash(job, chain, page, monkeypatch):
    """Page `page` completes under a random hash (a banned-string hold), the tail checkpoint is stashed
    there, then the hold resolves: Job.hash_completed_page, the real one, gives the page its content hash"""
    seq = job.sequences[0]
    seq.allocated_pages[page].phash = random_key(page)
    monkeypatch.setattr(JOB, "tensor_hash_checksum", lambda ids, prev: key(chain, page))
    return lambda: Job.hash_completed_page(job, seq, page)


def test_a_tail_checkpoint_under_a_deferred_page_follows_its_rekey(monkeypatch):
    gen = make_gen()
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    resolve = hold_and_hash(job, "a", 19, monkeypatch)
    generate(job, 130)                              # tail at 5120 under the random hash
    assert job.tail_checkpoint_key == random_key(19) and random_key(19) in cache
    resolve()
    assert job.tail_checkpoint_key == key("a", 19) and key("a", 19) in cache and random_key(19) not in cache
    generate(job, 256)
    assert key("a", 19) not in cache and gen.tail_stats["replaced"] == 1


def test_a_rekey_that_does_not_move_leaves_the_key_where_the_checkpoint_is(monkeypatch):
    # someone's checkpoint in RAM under the final hash: upstream's rekey leaves both where they are
    gen = make_gen()
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    resolve = hold_and_hash(job, "a", 19, monkeypatch)
    generate(job, 130)
    other = make_job(gen, "a", prompt = 5000)
    advance(other, 5120)
    cache.put(key("a", 19), State(other))
    theirs = cache[key("a", 19)]
    resolve()
    assert job.tail_checkpoint_key == random_key(19)
    generate(job, 256)
    assert cache.get(key("a", 19)) is theirs and random_key(19) not in cache


def test_a_rekey_onto_a_key_the_disk_has_leaves_that_copy_alone(tmp_path, monkeypatch):
    disk = open_disk(tmp_path)
    try:
        gen = make_gen(disk = disk)
        cache = gen.recurrent_cache
        land(disk, "a", range(0, 24))
        assert disk.write_state(key("a", 19), {"position": 5120, "checkpoint_size": CKPT,
                                               "target": {"layer.0": torch.zeros(4)}})
        disk.flush()
        job = make_job(gen, "a", prompt = 5000)
        ingest(job)
        resolve = hold_and_hash(job, "a", 19, monkeypatch)
        generate(job, 130)
        resolve()                                   # moved in RAM; on disk the copy that was there stays
        assert key("a", 19) in cache and job.tail_checkpoint_key is None
        generate(job, 256)
        disk.flush()
        assert disk.has_state(key("a", 19)) and gen.tail_stats["disk_dropped"] == 0
    finally:
        disk.close()


def test_a_rekeyed_tail_checkpoint_written_by_the_rekey_goes_with_the_next_move(tmp_path, monkeypatch):
    # upstream's rekey_state writes the checkpoint under its final hash at once
    disk = open_disk(tmp_path)
    try:
        gen = make_gen(disk = disk)
        job = make_job(gen, "a", prompt = 5000)
        ingest(job)
        resolve = hold_and_hash(job, "a", 19, monkeypatch)
        generate(job, 130)
        resolve()
        disk.flush()
        assert disk.has_state(key("a", 19))
        generate(job, 256)
        disk.flush()
        assert not disk.has_state(key("a", 19)) and gen.tail_stats["disk_dropped"] == 1
    finally:
        disk.close()


# --------------------------------------------------------------------------------------------
#   1.6.2: beside the other fork features, and the generator's own parts
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("busy", ["flight", "side", "pipe"])
def test_no_tail_checkpoint_for_a_job_in_flight_or_with_a_chunk_under_way(busy):
    # Generator.recurrent_checkpoint, the real one: a job of the staggered group in flight, or whose
    # prompt chunk is under way on the prefill thread or in the pipeline, is passed over (its state is
    # ahead of its position); it is offered the checkpoint once at rest
    gen = make_gen()
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    for _ in range(120):
        step(job)                                   # 5120, no offer yet
    fake = types.SimpleNamespace(active_jobs = [job], recurrent_cache = cache, _flight = None, _side = None, _pipe = [])
    if busy == "flight":
        fake._flight = types.SimpleNamespace(jobs = [job])
    elif busy == "side":
        fake._side = types.SimpleNamespace(job = job)
    else:
        fake._pipe = [types.SimpleNamespace(job = job)]
    before = dict(cache)
    GEN.Generator.recurrent_checkpoint(fake)
    assert dict(cache) == before and gen.tail_stats["stashes"] == 0
    fake._flight, fake._side, fake._pipe = None, None, []
    GEN.Generator.recurrent_checkpoint(fake)
    assert cache[key("a", 19)]["tail"] is True


def test_a_draft_is_cut_at_a_tail_boundary():
    # the verification loop stops accepting drafts where is_checkpoint_boundary() says so, and the
    # tail checkpoint's positions are among them only while the switch is on
    src = inspect.getsource(GEN.Generator._iterate_gen_co)
    assert "cp_boundary = batch_states is not None and job.is_checkpoint_boundary()" in src
    gen = make_gen()
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    for _ in range(120):
        step(job)
    assert job.is_checkpoint_boundary()
    gen.tail_checkpoint_interval = 0
    assert not job.is_checkpoint_boundary()


def test_the_pipeline_s_question_about_a_later_position_gets_that_position_s_answer():
    # Generator._pipe_ahead asks is_checkpoint_boundary(at = end) of the position a chunk would end at
    gen = make_gen()
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    step(job)                                       # 5001: generating
    assert job.is_checkpoint_boundary(at = 5120) and job.is_tail_boundary(at = 5120)
    assert not job.is_checkpoint_boundary(at = 5121) and not job.is_checkpoint_boundary()


def test_the_requeue_checkpoint_takes_the_tail_one_s_place():
    # the generator stashes at the position a job requeues at (maybe_stash_recurrent with the page
    # size) after its last token: the tail checkpoint before it goes, as for any regular one
    gen = make_gen()
    cache = gen.recurrent_cache
    job = make_job(gen, "a", prompt = 5000, cached_pages = 3, resumed = {"position": 768, "checkpoint_size": CKPT})
    ingest(job)
    generate(job, 3191)                             # 8191: tail at 7936
    assert cache[key("a", 30)]["tail"] is True
    step(job)                                       # 8192, a multiple of the checkpoint interval
    job.maybe_stash_recurrent(cache, PAGE)
    assert key("a", 30) not in cache and not cache[key("a", 31)].get("tail")
    assert job.tail_checkpoint_key is None


def _generator(monkeypatch, recurrent = True, **kwargs):
    model = types.SimpleNamespace(config = types.SimpleNamespace(vocab_size = 256),
                                  caps = {"recurrent_states": recurrent}, loaded_tp = False)
    cache = types.SimpleNamespace(num_slots = 4, max_history = 0, reset_states = Mock())
    monkeypatch.setattr(GEN, "PageTable", lambda *a, **k: types.SimpleNamespace(max_pages = 16))
    return GEN.Generator(model, cache, tokenizer = None, **kwargs)


def test_the_generator_s_switch(monkeypatch):
    monkeypatch.delenv("EXL3_TAIL_CHECKPOINT", raising = False)
    assert "recurrent_tail_checkpoint" in inspect.signature(GEN.Generator.__init__).parameters
    g = _generator(monkeypatch)
    assert g.tail_checkpoint_interval == 0 and g.tail_stats == TAIL_STATS
    assert _generator(monkeypatch, recurrent_tail_checkpoint = True).tail_checkpoint_interval == PAGE
    assert _generator(monkeypatch, recurrent_tail_checkpoint = 1024).tail_checkpoint_interval == 1024
    with pytest.raises(AssertionError, match = "multiple of the page size"):
        _generator(monkeypatch, recurrent_tail_checkpoint = 300)
    # nothing to checkpoint without recurrent states
    assert _generator(monkeypatch, recurrent = False, recurrent_tail_checkpoint = True).tail_checkpoint_interval == 0
    monkeypatch.setenv("EXL3_TAIL_CHECKPOINT", "1")
    assert _generator(monkeypatch).tail_checkpoint_interval == PAGE
    monkeypatch.setenv("EXL3_TAIL_CHECKPOINT", "512")
    assert _generator(monkeypatch).tail_checkpoint_interval == 512
    monkeypatch.setenv("EXL3_TAIL_CHECKPOINT", "0")
    assert _generator(monkeypatch).tail_checkpoint_interval == 0


def test_the_names_the_server_reads():
    # exllamav3-openai-server: model_container.generator_takes("recurrent_tail_checkpoint") and its
    # tracker's RecurrentCache.metrics["stash_replaced"] (rule 25 of its AGENTS.md)
    assert "recurrent_tail_checkpoint" in inspect.signature(GEN.Generator.__init__).parameters
    assert RecurrentCache(types.SimpleNamespace(loaded_tp = False)).metrics["stash_replaced"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
