"""
The checkpoint that follows a generating job (Generator.tail_checkpoint_interval,
Job.is_tail_boundary, Job.maybe_stash_recurrent, RecurrentCache.discard).

No GPU or model is required. The job's real checkpoint rules and the real checkpoint cache run
over a stub job that keeps its position and its pages the way Job does: the prompt is ingested
in chunks, then tokens are generated one at a time, and after every move the generator's offer
(maybe_stash_recurrent) is made, as Generator.recurrent_checkpoint does.

When the compiled extension is not installed the two source files under test are loaded on
their own, with the rest of the package stubbed.
"""

import importlib.util
import os
import sys
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = 256


def _load_real():
    sys.path.insert(0, REPO)
    try:
        from exllamav3.generator import job as job_mod
        from exllamav3.cache import recurrent as rec_mod
        return job_mod, rec_mod
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

    class _Any:
        def __init__(self, *a, **k):
            pass

    stub("")
    for p in ("cache", "generator", "tokenizer", "util"):
        stub(p)
    stub("ext", exllamav3_ext = types.SimpleNamespace())
    stub("util.memory", malloc_trim = lambda: None)
    stub("util.tensor", SeqTensor = _Any)
    sys.modules[f"{pkg}.util"].profile_opt = lambda *a, **k: None
    sys.modules[f"{pkg}.tokenizer"].MMEmbedding = _Any
    stub("generator.pagetable", Sequence = _Any, tensor_hash_checksum = lambda ids, prev: b"", random_hash = lambda: b"")
    stub("generator.filter", Filter = _Any)
    stub("generator.loop_detect", LoopDetector = _Any)
    stub("generator.sampler", Sampler = _Any, DefaultSampler = _Any)
    real("constants")
    return real("generator.job"), real("cache.recurrent")


try:
    job_mod, rec_mod = _load_real()
except Exception:
    job_mod, rec_mod = _load_stubbed()

Job = job_mod.Job
RecurrentCache = rec_mod.RecurrentCache
CKPT = 100      # bytes of a pretend checkpoint


class State:
    """What a job's recurrent state is to the cache: a size and a stash."""
    checkpoint_size = CKPT

    def __init__(self, job):
        self.job = job

    def stash(self):
        return {"position": self.job.sequences[0].kv_position, "checkpoint_size": CKPT}


class Page:
    def __init__(self, phash):
        self.phash = phash
        self.kv_position = 0


def key(chain: str, page: int) -> bytes:
    return f"{chain}:{page}".encode()


def make_gen(tail = PAGE, interval = 4096, chunk = 2048, room = 100):
    cache = RecurrentCache(types.SimpleNamespace(loaded_tp = False), max_size = room * CKPT)
    return types.SimpleNamespace(
        recurrent_cache = cache,
        recurrent_checkpoint_interval = interval,
        recurrent_checkpoint_interval_pp = 32768,
        max_chunk_size = chunk,
        tail_checkpoint_interval = tail,
        tail_stats = {"stashes": 0, "replaced": 0, "t_stash": 0.0},
    )


def make_job(gen, chain: str, prompt: int, cached_pages: int = 0, resumed = None):
    """A job whose prompt is `prompt` tokens long (plus the token generation starts from), with
    the first cached_pages pages found in the cache and the state resumed from `resumed` (the
    stashed dict allocate_pages got, None for a new state)."""
    job = Job.__new__(Job)
    job.generator = gen
    job.sequences = [types.SimpleNamespace(
        kv_position = cached_pages * PAGE,
        sequence_ids = [0] * (prompt + 1),
        allocated_pages = [Page(key(chain, i)) for i in range(4096)],
    )]
    for p in job.sequences[0].allocated_pages[:cached_pages]:
        p.kv_position = PAGE
    job.cached_pages = cached_pages
    job.new_tokens = 0
    job.recurrent_state = State(job)
    job.last_recurrent_checkpoint_pos = cached_pages * PAGE if resumed is not None else None
    job.tail_checkpoint_key = None
    job.prompt_tail_key = None
    job.resumed_from_tail = isinstance(resumed, dict) and bool(resumed.get("tail"))
    return job


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


def generate(job, tokens: int):
    """One token at a time: the sequence grows, the position follows, the offer is made."""
    seq = job.sequences[0]
    cache = job.generator.recurrent_cache
    held = []
    for _ in range(tokens):
        seq.sequence_ids.append(0)
        job.new_tokens += 1
        advance(job, seq.kv_position + 1)
        job.maybe_stash_recurrent(cache)
        held.append(len(cache))
    return held


def positions(cache) -> list:
    return sorted(v["position"] for v in cache.values())


def test_off_nothing_is_left_behind_an_answer():
    gen = make_gen(tail = 0)
    job = make_job(gen, "a", prompt = 5000)
    ingest(job)
    at_prompt = positions(gen.recurrent_cache)
    assert at_prompt[-1] == 4864                    # the prompt's last page boundary
    generate(job, 1000)
    assert positions(gen.recurrent_cache) == at_prompt
    assert gen.tail_stats == {"stashes": 0, "replaced": 0, "t_stash": 0.0}
    assert job.tail_checkpoint_key is None


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
