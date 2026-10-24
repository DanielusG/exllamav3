# Fork notes

Personal fork of ExLlamaV3 for one machine: Qwen3.8-Flash-Next (MoE, CPU expert split) on two RTX
5060 Ti in layer split, served by `exllamav3-openai-server`. Branch `flash-next-1.6.3`, based on
upstream **v1.6.3**: the commits of `flash-next-1.6.2` (based on v1.6.2) picked in order onto
v1.6.3, then what 1.6.3 asked of them ("The move to 1.6.3", below). It has run in production under
`exllamav3-openai-server` since 2026-10-18 13:16, and on 2026-10-23 it was merged into this fork's
`master`, which until then mirrored upstream and is now the fork's official line: upstream is
followed through its release tags. On 2026-10-24 the fix of "A page a prompt chunk has written is
not handed back" and "Content-addressed multimodal pages" (branches `fix-page-revert` and
`mm-content-hash`, run in production together as `prova-oom-mm` from 10:51 that day) were merged
into it. The tuning of the earliest fork branches (`flash-next` and
`flash-next-placement`, on v1.5.4) was ported onto `flash-next-1.6.2` one feature at a time. Every
feature stays behind a switch of its own, off by default: with every switch off this branch
behaves as upstream v1.6.3, but for the fixes of upstream's own code, which have no switch and
change nothing in a run that does not reach what they fix (one so far: "A page a prompt chunk has
written is not handed back", below).

Everything on this branch is Python. The checkout is used without building anything: put it
ahead of the installed package on `PYTHONPATH` (the server's wrapper does
`export PYTHONPATH=<this checkout>`). The compiled extension is a top-level module
(`exllamav3_ext`) and keeps coming from turboderp's published **1.6.3** wheel,
`exllamav3-1.6.3+cu132.torch2.11.0-cp312-cp312-linux_x86_64.whl` (CUDA 13.2, torch 2.11,
Python 3.12; sha256 `624e7a09643ce7f86ef1d2e0994bd0bbb1ac5297a45800a075bf2a447314b96a`),
installed in the environment: on the workstation in the standard environment `tabbyapi`, in
place of its 1.5.4 wheel, with `pip install --no-deps` so torch is left as it is. It has to be that
wheel: the 1.6.2 sources call extension functions the 1.5.4 one does not have
(`exl3_moe_split_issue_fused`, `batched_scan_replay` among them), the 1.6.3 ones two that the
1.6.2 one does not have (`routing_ds3_group`, `routing_gemv_f32`, for Ling 3.x's grouped router),
and 1.6.3's top-P convention is in the fused sampler's kernel (`sampling_fused.cu`) as well as
in the Python samplers. Without the `PYTHONPATH` entry the same environment runs stock upstream
1.6.3.

The expert frequency profile: upstream's own fixes are on this base. 1.6.3 has the fix of #515
(345eca1c, the profile's persistence: "What changed with 1.6.3" under the placement weight) and
that of #523 ("expert placement loaded from a persisted profile causes degenerate generation"),
f35b51cc: `_split_saved_lists_ref` goes through `_split_perm`, so the sweep's checkpoint reads,
the demotion's install keys and the transfer verify follow the load-time permutation of a seeded
layer. PR #524, which encoded the permutation in the split map instead, was closed unmerged on
2026-10-17. Nothing of the profile is carried here; whether the server turns upstream's profile
on again is the server's call (rule 20 of its `AGENTS.md`).

## The move to 1.6.3 (2026-10-18)

The 19 commits of `flash-next-1.6.2` picked in order onto v1.6.3 (`git cherry-pick -x`,
authorship kept). One conflict: `.github/workflows/build.yml`, changed upstream and deleted here,
stays deleted (below). Three files merged by themselves, each read after the merge:

- `modules/block_sparse_mlp_cpu.py`: 1.6.3 counts the first registered split module's steps
  twice, `_swap_tick_count` (since the last sweep, which zeroes it: the sweep's trigger, as
  before) and `_split_step_count` (since load: the profile save's gate and period). The fork's
  `save_expert_stats` (the counts kept apart added at the prefill weight) sits under upstream's
  new gate; `_split_swap_tick` steps both counters and keeps the fork's guard on the inline sweep
  (not from a lane, not while a lane has a pass under way).
- `generator/disk_cache.py`: 1.6.3 reads the free space with `shutil.disk_usage` instead of
  `os.statvfs` (for Windows), the same figures on Linux; the page-cache hygiene is elsewhere in
  the file.
- `doc/env_vars.md`: upstream's top-P paragraph and `EXL3_TP_REDUCE_MAX_ISA`, beside the fork's
  `EXL3_DISK_CACHE_UNCACHE`.

Upstream changed none of the other files the fork changes (`generator.py`, `job.py`,
`cache/recurrent.py`, `model/moe_cpu_host.py`, `modules/row_table.py`, `util/tensor.py`); in
`gated_delta_net.py` the fork's class attribute of `GDNState` is beside 1.6.3's direct KDA gates
for Ling. The sections below say "What changed with 1.6.2" and "this base" for the port from
1.5.4: 1.6.3 changed none of those paths, so they hold as written; where 1.6.3 changed something,
a "What changed with 1.6.3" paragraph says so. What 1.6.3 asked of the fork, a commit each:

- the fakes of the split tests set `_split_step_count` too (24 tests failed on it);
- `Generator` carries on the class, with every switch off, the fork's state that `iterate()`
  reads (`stagger_decode`, `prefill_thread`, `_stagger_backlog`, `_stagger_run`, `_flight`,
  `_side`, `_pipe`; the empty queues as tuples): upstream's
  `tests/generator/test_failure_containment.py` makes a Generator with `__new__` and drives
  `iterate()` with upstream's attributes only. Stock 1.6.3 passes it, the fork failed on
  `_stagger_backlog`. `__init__` still sets every one of them on the instance;
- the exact check names its release in one constant (`BASE_RELEASE`, under the exact check);
- a test of the profile save that now fires beside a chunk under way (under the placement weight).

What else of 1.6.3 reaches this machine: the top-P convention (the token whose probability
crosses `top_p` is now kept, and in the fused sampler its ties at the cutoff too, down-weighted
together; with a top-P stack, which the server's Qwen3.8 sampling sets have, the same seed
samples other tokens than on 1.6.2), upstream's fixes of the expert profile (above), and the disk
tier's fingerprint (under the page-cache hygiene). The rest is Ling 3.x (architecture, grouped
router, direct KDA gates, MTP head), fp8 source weights with partial blocks, the native
tensor-parallel backend's AVX tier, and Windows.

Tests, on the development machine without a GPU (2026-10-18): the fork's own
(`python -m pytest --noconftest tests/test_*.py`, `EXL3_TEST_DISK_DIR` on btrfs) 306 passed on
`flash-next-1.6.2`, 309 here (three tests added); upstream's tests marked `nogpu`, with a
stand-in for the compiled extension, come out the same test by test here as on stock v1.6.3
(430 tests: 230 pass, 43 skip, 157 fail on both, those that need the extension's kernels).
This branch has run in production since 2026-10-18 13:16 (the server's switch to the 1.6.3 wheel,
with production's flags, which turn every switch of this branch on), without the exact check on
1.6.3 before it. Every measure below was made on 1.5.4 or 1.6.2, and where a section says a
feature has not yet run on this base, that is about its measures and checks.

## What differs from upstream

### No CI workflows

`.github/workflows/` is removed on this branch: `build.yml` (upstream's wheel build) and
`cache_keepalive.yml` (the keepalive of that build's compiler cache, new in 1.6.x). Nothing here
is released as a wheel, and a push that carries a workflow file needs a token with the
`workflow` scope.

### A page a prompt chunk has written is not handed back (a fix: no switch)

A fix of upstream's prompt ingestion (`Job._prefill_co`), found on 2026-10-23 by reading and
reproduced on the CPU with the real page table; the page table it guards (`generator/pagetable.py`)
is upstream's, unchanged. In force with every switch off.

**What upstream does.** A job's pages for generation and its prompt's tail (the last page, partial;
with the MTP head also the last full page when the prompt's length less one is a multiple of a
page: the page the head leaves out of the cached prefix) are unique pages.
`PageTable.allocate_pages` takes them from the unreferenced pages in eviction order
(`CachePage.add_ref_unique`: empty pages, then cached sequences from their tail), often complete
pages of other conversations. It keeps what such a page was (`CachePage.backup`: its hash, the hash
before it, its position, its access serial; `can_revert` set) and gives that back when the page's
last reference goes while `can_revert` is still set (`sub_ref`, `revert`): a page the job never
wrote stays the other conversation's, which can reuse it by its hash. The decode step gives the
backup up when it stages its tokens, before its pass (`Job.get_input_ids_list(add_to_cache =
True)`). A prompt chunk gave it up in its record, after its passes, with the pages' positions and
tokens; the record of a chunk that ends on a page boundary also gives up the page after it, which
it links to the chain.

**The hazard.** A chunk released after its passes were issued and without its record: the release
gives the other conversation's identity back to a page whose K/V (the target's, and the draft
cache's, which has the same page indices: the MTP head's update writes the chunk there too) is now
this job's, wholly or in part, at a full page's position. Nothing fails. The other conversation's
next turn finds the page by its hash in VRAM, ahead of the good copy the CPU tier took when the
page was evicted, and reuses it; once neither tier holds the good copy, the page's next eviction
pushes the bad one to the CPU tier, and the disk tier can write it and read it back after a
restart. It takes two things:

- a release between a chunk's issue and its record. Upstream: a pass that fails (CUDA out of
  memory, the host-memory guard, any kernel error), contained by `Generator.reap_failed_job`. This
  fork adds the cancel of a job whose chunk is under way on the prefill thread or in the pipeline
  (`Generator.cancel`: `_side_drop` and `_pipe_drop` let the chunk run to the end of what it is in,
  record nothing, and the pages go; `clear_queue` the same) and the failures there
  (`_side_collect`, `_pipe_fail`);
- a chunk that writes a page whose backup is still set: the chunk that writes the prompt's tail
  page (or the full page the MTP head leaves out), unless it starts on that page and is not the
  job's first, when the record of the chunk before it has given the backup up. On production's
  setup (recurrent states, the MTP head, chunks of 4096 and the prompt's last page as a pass of its
  own): the last-page pass when it is the job's first chunk (a prompt that ends less than a page
  past the checkpoint it resumes from: the same prompt again, or a turn that adds little past the
  previous answer's tail checkpoint, as a short tool result does), a prompt shorter than a page,
  and the last chunk of a prompt whose length less one is a multiple of a page. Without recurrent
  states, the last chunk of the prompt, unless it starts on the tail page.

**The fix.** `Job.forget_revert(seq, start, end)` gives up the backup of the pages at positions
start to end. `Job._prefill_co` calls it for the positions the chunk's passes write (after a
partial-page copy, with an atomic multimodal prefill's extension) right before it yields the
passes: on the generator's thread, before any of them can be issued, whoever issues them
(`Job.prefill` inline, the prefill thread, the pipeline's halves, any other driver of the
coroutine). The record gives up what it gave up before. What still goes back: the pages of a job
released before it planned a chunk (allocated, waiting for its turn), and the pages past what its
chunks and decode steps reached (the pages for generation of a job let go early).

**Successful runs are the same, bit for bit.** `can_revert` is read in one place, when a page's
last reference goes (`sub_ref`). Every page a chunk's plan gives up is given up by that chunk's
record too (the pages of an atomic multimodal extension past the chunk's end by the next chunk's
record, which feeds them again), and between the two the page is held by this job alone: a unique
page has a random hash no other job can claim, and a page shared by hash has no backup to give up.
Its last reference goes only with this job's release, at the job's end, at a requeue, or by a
cancel or a failure, and the first two come after every chunk is recorded. So where no failure or
cancel falls between a chunk's plan and its record, every page goes back, or does not, exactly as
before: the same page table, the same pages reused, evicted and pushed to the tiers, by the same
jobs, and the same passes. In one case pages that no pass wrote lose their way back: a chunk
planned in the pipeline but not yet handed to its threads (one too short to run beside the decode
waits for the chunks before it) when its job is cancelled. Its pages go as empty pages; the old
conversation gets them back from the CPU tier, which took them when they were evicted, or prefills
them again.

**Left as upstream** (read, not changed):

- the draft step writes the draft cache at the positions it drafts before the decode step stages
  them, in the same `iterate()` call; only an exception can come between the two, and it fails the
  iteration (upstream's `AsyncGenerator` keeps the error for every job after it, the server's
  worker reports itself poisoned and aborts the disk tier);
- a DFlash drafter writes its whole block into the draft cache, past the window the verification
  stages when the confidence cut shortens it, so at the job's end the page past its last position
  can go back with those slots of the draft cache written. The draft cache only (the tokens are the
  target's) and DFlash only; not used here.

Tests: `tests/test_page_revert.py` (no GPU, no extension): the real page table, the real `Job` and
the generator's real rounds (inline, on the prefill thread, in the pipeline) over a stand-in model
that writes every token into a stand-in K/V cache, at its page and slot, and can fail or wait
part-way through a pass. A conversation leaves two complete pages; another job takes the second as
its tail page and its only chunk writes into it; the chunk fails (inline, on the thread, in the
pipeline's first or second half) or the job is cancelled with the chunk under way (on the thread,
in the pipeline): the page is no longer under the first conversation's hash, and that
conversation's next turn reuses only its own pages. Those six fail without the fix: the page goes
back under the hash, and the next turn reuses both pages, the second with 100 slots of the other
job's K/V. What stays, with and without the fix: a job cancelled while it waits for its first chunk
gives its page back whole; a page for generation that no pass wrote goes back at the job's release;
the decode step gives the backup up as it stages its pass; and turns of three conversations with
every chunk recorded leave the same page table (hashes, positions, references, access serials,
`can_revert`) and the same K/V after every step as with the backup given up at the record only,
inline, on the thread and in the pipeline (the run has pages given up at the plan and pages that
went back whole). Marked `nogpu`; run with `--noconftest` without the compiled extension. The
fork's own tests: 321 passed (309 before); upstream's `tests/generator`, with a stand-in for the
extension, the same test by test as before (34 pass; the rest need the extension's kernels or a
GPU).

In production since 2026-10-24 10:51 (branch `prova-oom-mm`, merged into `master` that day), with
the server's retry of a request after a VRAM OOM: the server's test fault (`EXL3SRV_INJECT_OOM=1`,
the first chunk of a request fails where it is planned, before any pass) failed a request, which
was made again and completed. A real chunk failing or cancelled after its plan has not been seen
with the fix yet.

### Staggered decode (`Generator(stagger_decode = True)`, `EXL3_STAGGER_DECODE=1`)

With the layers split over two cards a regular decode step keeps one card idle at a time. With
`stagger_min_jobs` or more jobs generating, the jobs are decoded as two groups half a pass apart:
one group's second-card half and the other group's first-card half are issued together, a module
of each in turn. Off by default; with it off the generator behaves as upstream.

- `exllamav3/model/forward_cursor.py`: `ForwardCursor`, the layer-split forward pass issued in
  pieces (same modules, order and params as `Model.forward`), and the rules for using it.
- `exllamav3/generator/generator.py`:
  - `iterate_gen` is a wrapper around `_iterate_gen_co`, the same step as a coroutine that stops
    at its forward pass. The draft and decode steps iterate `self._step_jobs()` instead of
    `self.active_jobs` (the same list unless a staggered step narrows it to a group).
  - `iterate_staggered` and its helpers: the scheduler. Its docstring states what keeps a
    waiting pass safe; keep it true when touching the step functions.
  - `cancel` completes the group in flight before releasing one of its jobs; `clear_queue` and
    `close` drop it; `iterate` hands out results a cancel produced (`_stagger_backlog`), and does
    not offer prefill or a checkpoint to a job in flight.
- Not staggered: tensor-parallel loads, drafters other than MTP, the hybrid draft, models not on
  exactly two cards in a row, multi-sequence jobs, any iteration in which a job is ingesting its
  prompt, and fewer than `stagger_min_jobs` jobs generating (3; `EXL3_STAGGER_MIN_JOBS`).
- Each stretch of staggered calls is reported in one line when the decode goes back to regular
  steps (calls, seconds, most jobs at once); the line that says whether the decode can be
  staggered is printed once, at the first try. Both are flushed (stdout of a service is
  block-buffered). `stagger_stats` counts the rest.

**What changed with 1.6.2.** The fork's step functions were written against 1.5.4; on this base:

- *The recurrent layers' rewind reads what the last pass staged.* 1.6.2 keeps a GDN (and KDA,
  Mamba2) state in two rows per slot and settles a speculative pass by replaying the accepted
  prefix over the scan inputs that pass staged. Those inputs sit in buffers each layer's state
  (`GDNLayerState.spec_conv`, `spec_beta`, `spec_g`) holds once for every pass, addressed by
  batch row, where 1.5.4 kept per-slot history. A group in flight is settled when it is
  sampled, after the next group's first half has been through the same first-card layers and
  staged its own inputs there: without a remedy its rewind replays the other group's tokens, a
  wrong state with no error (the check on the toy model shows it, and so does the shape assert
  when the two groups differ in size). `_stagger_hold_spec` copies the in-flight pass's rows of
  the first card's layers out before the next group is issued (one multi-tensor copy per dtype on
  the first card's stream), and `_stagger_sample` hands those copies to the layer states for the
  time of the group's sampling. The second card's layers need nothing: a group is sampled
  before the next one reaches them. Cost: the copies on every staggered call with MTP (per
  first-card recurrent layer, the group's jobs x (1 + draft tokens) rows of its scan inputs),
  and VRAM on the first card for buffers the size of those layers' staging buffers.
- *The CPU embedding stages its output through one pinned buffer per shape* (`Embedding`,
  `pinned_staging`, relying on a host sync between two passes). Two groups' passes have none on
  the first card unless the MTP head's draft step happens to be there; the cursor copies a
  pinned host output into pinned memory of its own (torch's caching host allocator keeps a block
  until the uploads read from it have run).
- *The hybrid draft* (`hybrid_draft_tokens`) decides one long verification pass for the whole
  batch, and the staggered step does not draft it: refused, like the n-gram drafter.
- *The persistent disk tier.* Its write-behind at the end of `iterate()` takes a job in flight
  like any other: it queues no page within two of the sequence position, and the pass under way
  writes past that position (test: every page it queues is complete and never written again).
  The idle flush (`on_queue_drained`) never runs with a group in flight, whose jobs are active;
  the prefetch at `enqueue` touches no page in use. `close()`, which flushes, drops the group in
  flight instead of completing it: a retired generator issues nothing more, and may have failed.
- *A cache taken over by a newer generator* (1.6.2's ownership check): `iterate()` drops the group
  in flight before refusing to run, and `cancel` drops it instead of completing it.
- *NonFiniteLogits* on a job's first sampled token can come while a staggered call samples a
  group: as fatal as in a regular step; the generator keeps its own buffers and the new group in
  flight, for a `cancel` to complete or `clear_queue` to drop.
- *The CPU expert worker* still serves its job ring in publication order: the fused decode issue
  (`exl3_moe_split_issue_fused`) writes the job's descriptor into the ring from the host, in call
  order, and the worker waits for each job's data flag in turn. So the halves are still issued a
  module at a time; from one host thread nothing else needs a lock (two threads are the prefill
  thread's business).
- *The forward path* (`Model.forward`, `forward_ls`) is the same in the two releases; what moved
  is inside it and the cursor calls it as `forward_ls` does: `prepare_for_recurrence` (in
  `prepare_inputs`) reads the states' parity when the pass is staged, and
  `advance_recurrent_states` (at `finish()`) records `spec_row` / `spec_shape`. The cursor differs
  from `forward_ls` only for a pass left waiting: the workers are woken again when it resumes,
  the stream stack is copied when it waits, and a pinned host output is copied (above).
- Not checked beyond reading: promote-on-miss (`EXL3_MOE_CPU_PROMOTE`, off in production) runs its
  step at the first split module of every pass, which with two groups is in the middle of the
  other group's second half, and waits there on the host for the previous pass's snapshot
  events; its map flips and slot copies stay stream-ordered. The inline expert sweep that
  upstream falls back to when a sweep is badly overdue changes the placement between two layers
  of the group in flight (numerics only; the server defers sweeps to the queue's drain). Group
  sizes are batch sizes of their own for the graph-captured decode paths (more graph slots).

Measured on the machine **on the v1.5.4 base** (2026-10-10, bench at 91621dd, placement adapted
to the bench's prompts, 91% of the expert selections served from VRAM, MTP with 1 draft token);
not yet run on this base:

| jobs | one batch | staggered | | cards, one batch | cards, staggered |
|---|---|---|---|---|---|
| 2 | 94.9 tok/s | 93.2 tok/s | -1.7% | 38% + 40%, 144 W | 58% + 62%, 180 W |
| 3 | 117.7 tok/s | 128.6 tok/s | +9.3% | 44% + 42%, 151 W | 62% + 65%, 194 W |
| 4 | 128.6 tok/s | 154.4 tok/s | +20.1% | 40% + 40%, 149 W | 64% + 65%, 200 W |

- Same computation: with forced tokens the probabilities of a staggered run were bit-identical
  to those of each group alone, in every position of every job (9 job comparisons, 400
  positions each). The check is not blind: the same figures tell a job in a batch from the
  job alone (regular decode: the most likely token changes in 2-7 positions out of 400). On
  1.6.2 this is the check to repeat first: the rewind above is new.
- A staggered call is draft step (2 ms), issue (about 15 ms of host time for a pass, whatever
  the size of its group) and sampling (2-2.7 ms per job); the cards work during the issue and
  idle in the other two. With groups of one job the host is the limit of the issue too: the
  cards finish each half as the host finishes issuing it (a half pass of one job is about 12
  ms of stream time). Hence the threshold: two groups of one job cost two calls for what one
  batch of two gives in one step of the same length. With groups of two the cards' half pass
  (about 17 ms) and the host's issue are level, and what the cards still idle, 6-7.5 ms of a
  23 ms call with 4 jobs, is the draft and sampling phases.
- The CPU expert worker is out of the way only with a placement that fits the traffic: the
  cards waited for it 2-3 ms per call each at 91% of selections served from VRAM. On the
  placement tuned for production traffic the bench's prompts got 74%, and every multi-job
  figure, staggered or not, stopped at about 100 rows per second (4 jobs in one batch: 86
  tok/s against 129).

Why the two halves are issued a module at a time: the CPU expert worker serves its job ring in
publication order (`exllamav3_ext/cpu/moe_handoff.cu`), so two halves issued back to back would
stall the first card at its first MoE layer until the second card is done.

Tests: `tests/test_stagger_decode.py` (no GPU, no extension: toy model on two pretend cards, stub
jobs; tokens of a staggered run must equal those of a regular one). The toy's recurrent layers
stage a speculative pass's inputs once for every pass and replay them in the rewind, as 1.6.2's
do, and `test_the_library_still_stages_what_the_hold_copies` reads off the library's sources the
points the hold depends on (run it before taking a newer upstream: a library that staged
elsewhere would leave the hold copying nothing). Marked `nogpu`; upstream's `tests/conftest.py`
imports the package, so without the compiled extension run it with `--noconftest` (the file then
loads the sources it tests on their own).

The bench of the v1.5.4 branches (`experiments/stagger_bench.py`, `tests/test_stagger_bench.py`)
is not on this branch.

### Prompt ingestion beside the decode (`Generator(prefill_thread = True)`, `EXL3_PREFILL_THREAD=1`)

Upstream's `iterate()` runs a prefill chunk for every job still ingesting its prompt and then one
decode step: a job that is generating gets one token per chunk while another job's prompt goes in
(2.9 s per 2048-token chunk on this machine). With the switch on, the forward passes of a chunk
run on a second thread and the generator's thread goes on decoding. Off by default; with it off
the generator behaves as upstream. Run on the machine **on the v1.5.4 base** (the runs are below:
the exact form of the same-computation check, one to five jobs generating beside an ingestion,
and a test instance of the server); not yet run on this base. In the server: `--prefill-thread`
(rule 24 of its `AGENTS.md`).

The two threads issue on the same CUDA streams, so kernels never overlap on a card. That is not
a choice: the compiled EXL3 kernels (GEMM, MGEMM, the fused MoE kernel) coordinate their thread
blocks through one lock buffer per device, indexed from zero by every launch
(`quant/exl3_devctx.cu: get_locks`, `quant/exl3_gemm_inner.cuh: barrier_acquire(&locks[...])`),
and several decode paths go through buffers baked into a module at load time. What the second
thread buys is therefore host time and CPU expert worker time: decode steps fit in the time a
chunk spends waiting for the worker and for its own host, which is most of a chunk here (the
cards are at about 30% each during prefill). It is a finer sharing of the cards than the
server's former `--decode-slice-ms`, not two passes at once.

- `exllamav3/util/lane.py`: a lane is the number a thread that issues passes beside another one
  carries. Everything the library shares between passes on the strength of "one pass at a time"
  is kept once per lane: the workspaces of `g_tensor_cache` (lane in the key), the CPU expert
  worker's slots, a row table's pinned staging sets.
- `exllamav3/generator/job.py`: `Job.prefill` is a wrapper around `_prefill_co`, the same
  function as a coroutine that yields each chunk's forward passes as a function. The record of
  the chunk (positions, page hashes, checkpoint, progress event) stays in the coroutine.
- `exllamav3/generator/generator.py`: `_prefill_round` and the `_side_*` helpers. Its docstring
  states the rules; the ones that matter when touching it:
  - only forward passes run on the second thread. The page table, the recurrent cache and the
    results are touched by the generator's thread alone (the server's tracker relies on it);
  - one chunk under way at a time; a job whose chunk is under way gets no checkpoint and no
    decode step, and `cancel` / `clear_queue` / `close` let the chunk finish first;
  - a chunk and a decode step overlap only if they cannot meet inside a module: chunk of at
    least `side_min_tokens` tokens (33: one past the CPU expert worker's decode-size batches;
    a page until 2026-10-11, see the third session below), decode batch of at most 8 rows x
    tokens (`side_max_rows`, the verification pass counted at its widest). Past those
    thresholds no path is common to the two (decode graphs, fused decode kernels and
    single-row buffers up to 8 rows; every prefill tier above the worker's 32). A shorter
    chunk is waited for like an inline one;
  - with nobody generating the round waits for its chunks, which still run on the second
    thread: the workspaces of a prefill pass exist once.
- `exllamav3/model/moe_cpu_host.py`: `split_lanes` (a range of slots per lane: on one rotation
  two threads make a stream wait for itself, `tests/test_moe_host_lanes.py` shows it), the
  descriptor and the write of its data flag issued under one lock (the worker waits for the
  job at the head of its ring), and for a thread that shares the pass's stream the copies and
  flag waits of the chunked path on a stream of their own, with the host waiting for the last
  readback: a flag wait on the shared stream would stop the other thread's kernels for as long
  as the worker takes. The sources of those copies are held until that readback
  (`_handoff_held`: the first run on the machine computed NaN without it, see below).
- `exllamav3/modules/block_sparse_mlp_cpu.py`: no inline expert sweep while a second thread
  issues a pass; the copy path's selection translate of the second thread into a buffer of its
  own.
- `exllamav3/modules/row_table.py`: the pinned staging sets and the queued prefetches per lane.
- `Generator.prefill_thread_state` (`off`, `pending`, `on`, `unavailable`) and
  `prefill_thread_reason` say where prompt ingestion runs; `side_stats` counts the chunks, those
  with decode steps beside them, the steps, the errors, the seconds of the chunks and of the
  waits. One line on stdout per stretch of chunks ingested beside a decode.
- Not available (state `unavailable`, ingestion inline as upstream): tensor-parallel loads, a
  model not on CUDA devices, a draft model other than an MTP head, promote-on-miss (below), a
  CPU expert worker with a single slot.

**What changed with 1.6.2.** The fork's thread was written against 1.5.4; on this base:

- *The fused decode issue.* 1.6.2 issues a decode-size worker job in one extension call
  (`exl3_moe_split_issue_fused`): the descriptor into the shared ring and the ring's tail one
  higher, then the slot's consumed wait, the issue kernel and the data flag, with no lock of its
  own (it keeps the interpreter for the whole call). The chunked path publishes from Python, the
  tail read and the tail written bytecodes apart: a thread switch in between and both jobs get
  the same ring entry, one of them lost and a stream waiting for it for good. Both now go under
  `_issue_lock`, the sequence number included, the fused call and its throttle-and-retry inside;
  the fused issue takes its slot from its lane's range like the chunked path; descriptors go
  through upstream's `_set_job_seq` (the 32-bit flags with the high half beside them).
- *The streamed prefill and the pinned arena.* A chunk's hot experts stream to the card by DMA
  straight from the pinned arena (1.6.2), or through the stage ring for chunks past the pinned
  ones. While a chunk and a decode step overlap only the chunk streams (the decode is at most 8
  rows, below `stream_min_rows`), so the weight-staging slots, the stage ring and the
  per-device stream state have one user at a time and need no lock (two prompt lanes at once
  are the pipeline's business: `_stage_lock`, next section). On the prefill lane the worker's flags are waited for on the
  copy stream and the landed batch by the host: the pass's stream, which the decode shares,
  never waits for a DMA (`tests/test_moe_host_lanes.py` reads it off the sources). The arena
  blocks the DMA reads are rewritten only by the placement sweep (at the queue's drain, never
  inline while a lane is at work) and by promote-on-miss (not offered with the thread).
- *Promote-on-miss* (`EXL3_MOE_CPU_PROMOTE`, spare GPU slots with the pinned arena; production
  runs it off) runs a step at the first split layer of every pass, prompt chunks included, over
  one state for every split layer: map flips, host waits on snapshots, promotions into GPU slots
  a chunk under way may still read, demotions into worker slots a chunk's job may use. Nothing
  of it is per lane: with spares on any split layer the thread is `unavailable`
  (`prefill_thread_reason` starts with `promote-on-miss`).
- *The recurrent states.* A GDN, KDA or Mamba2 layer keeps two rows per slot and stages a
  speculative pass's scan inputs, by batch row, in buffers shared by every pass, which the
  decode's rewind replays after its sampling. A chunk runs through the same layers between a
  decode pass and its rewind, but a prompt chunk's pass has no recurrent history: it reads and
  writes the base row of its own slot and stages nothing (`tests/test_prefill_thread.py` reads
  that off `job.py`, `gated_delta_net.py` and `mamba2.py`, and runs the toy with a chunk layer
  between every pass and its rewind; with layers that would stage a chunk, the replay goes
  wrong). Checkpoints, `rekey` and the paired target / draft stashes stay on the generator's
  thread, after the chunk's record.
- *The hybrid draft* decides a long verification pass after the prefill round: the decode batch
  is counted at `1 + max_draft_tokens` per row, so with it a chunk is waited for.
- *The persistent disk tier.* Write-behind at the end of `iterate()` takes the job whose chunk is
  under way like any other: its sequence position is the chunk's start until the chunk is
  recorded, on this thread, and write-behind keeps two pages behind it (test: no page of the
  chunk is queued, and no page queued is written again). The prefetch at `enqueue` and the CPU
  tier's queued writes touch host slots and complete pages, never a chunk's. `close()`, which
  flushes, waits for the chunk under way and stops the thread first.
- *A cache taken over by a newer generator*: `iterate()` and `cancel` wait for the chunk under way
  (nothing more of it is issued) before refusing to run or dropping the job.
- *NonFiniteLogits* on a decode step leaves the chunk under way the generator's, for
  `clear_queue()` or `close()` to see through.
- *CUDA graphs.* The decode's graphed sites capture on the generator's thread, on a stream made
  for the capture (non-blocking) and in thread-local mode, so what the prefill thread launches
  on the pass's stream neither enters the graph nor breaks the capture; the capture's device
  synchronize waits for the chunk's work, which never waits for this thread. Prompt chunks take
  no graphed site (above 32 rows and 16 tokens). cuBLAS handles are per thread, their workspace
  one per device: safe on one stream.
- *The staggered decode* is never on while a job is ingesting; a group in flight when a prompt
  arrives is completed beside the chunk, as a regular decode step.
- Not checked beyond reading: DFlash and other draft models (refused), the n-gram drafter (allowed:
  no model of its own), images in a prompt.

What it does not do, and what would:

- A decode job still waits in the worker's ring behind the prefill chunks published before
  it, and the worker idles on a decode job's data flag while prefill chunks sit behind it. A
  ring and a thread of its own for the decode in `cpu/moe_handoff.cu` removes both; it needs a
  build of the extension.
- Kernels of the two threads on one card at the same time need the lock buffer, the GEMV and
  routing workspaces and the baked decode buffers per stream: also a build.
- Any host wait for a stream is a wait for both threads' kernels on that card.
- `EXL3_MOE_CPU_SLOTS` (4) is split 2 for the decode and the rest for the prefill, whose window
  of outstanding jobs was 4: set 6 to keep it.
- In the server (`--prefill-thread`, rule 24 of its `AGENTS.md`): the worker asks for the
  thread only of a library that has `Generator.prefill_thread_state`, reads that state at
  every iteration and its counters (`side_stats`). Its tests read these points off the fork's
  sources (`tests/moe_harness/build.sh <this checkout> flash-next-1.6.3 <dir>` in the server's
  repo).

Tests (no GPU, no extension): `tests/test_prefill_thread.py` (the real round over the toy model
of the staggered tests, chunks that last a given number of decode steps: same tokens as inline,
and what a chunk on another thread must not meet; the 1.6.2 points above),
`tests/test_moe_host_lanes.py` (the host's issue and collect code, the fused issue included,
against a model of streams, flags and the worker's ring), `tests/test_split_lanes.py` (the split
mixin's sweep guard and translate), `tests/test_row_table_lanes.py`, `tests/test_lane.py`.
Marked `nogpu`; run with `--noconftest` without the compiled extension.

The bench of the v1.5.4 branches (`experiments/prefill_bench.py`, `tests/test_prefill_bench.py`)
is not on this branch. On the v1.5.4 base it ran one job generating while another ingests a
document, as upstream and with the thread, against each alone; forced-token probabilities of
both jobs against the upstream path; the longest time the generating job went without a token
during the ingestion; memory of the cards. On this base the check to repeat first is its exact
form (third session below).

Measured on the machine **on the v1.5.4 base**; first run (2026-10-10, bench at ad6dfa4,
8289-token document, chunks of 2048):

- The scheduling does what it is for: 142 decode steps were taken beside 4 chunks, and the job
  that was generating got 259 tokens during the 8.1 s of the other job's ingestion (31.8 tok/s;
  60.4 alone, 1.2 as upstream). The ingestion took 7.3 s as upstream.
- **The computation was wrong.** After a prompt ingested on the second thread the probabilities
  of that job were NaN in all 64 positions, and the share of expert selections served from VRAM
  in that run was 0.675 against 0.952 as upstream: about what routing on noise gives. So the
  figures above are those of a pass computing garbage, not a measure of the feature.
- Cause found by reading, not confirmed on the machine: the copies of the chunked path's inputs
  run on the handoff stream when the worker gets to their job, after `_issue_compute` has
  returned; its temporaries were freed by then, and the allocator hands a freed block out again
  in the order of the stream it was made on, so the pass's next kernels wrote into memory those
  copies had yet to read. The sources are now held until the host has seen the last readback
  (`_handoff_held`). On one stream the same lifetime is safe, which is why upstream has it.
  (On 1.6.2 the same NaN would now stop the job at its first token: `NonFiniteLogits`.)
- Two identical runs as upstream do not give identical probabilities when each ingests its
  prompt again (largest shift 0.03-0.05, the most likely token changed in 1-4 positions of
  64-600), so "identical to upstream" cannot be the test. The bench repeats every compared run
  twice as upstream and judges the run with the thread against that floor; it counts values
  that are not finite, compares a prompt ingested on the thread with nobody decoding as well
  (which tells the thread from the overlap), and drops the state checkpoints between runs
  rather than resetting the page table (the reset was not behind the floor, see below).
- Memory: about 70 MiB less free on each card with the thread (0.39 and 0.37 GiB left).
- `EXL3_LANE_HOST_WAITS=0` keeps the second thread's worker waits on the pass's stream: nothing
  is gained, but the handoff stream is out of the picture, for telling whose fault a wrong
  number is.

Second run (2026-10-10, bench at af7c041, same case, with the fix):

- Same computation, as far as this bench can tell. No value that is not finite; the three
  comparisons with the thread are within the floor of the same run done twice as upstream
  (largest shift 0.036-0.078 against 0.047-0.057, mean shift 0.0040-0.0056 against
  0.0041-0.0050, most likely token changed in 1-3 positions against 1-4); expert selections
  served from VRAM 0.945 with the thread against 0.949 as upstream, 0.951 against 0.951 with
  nobody decoding. What it cannot tell: an error smaller than the floor, and a race that did
  not happen in this one run.
- Speed. The generating job got 354 tokens during the 11.3 s of the other job's ingestion
  (31.2 tok/s; 61.0 alone, 1.1 as upstream), 193 decode steps beside 4 chunks. The ingestion
  took 11.3 s against 7.9 s beside the same job as upstream and 7.3 s alone (731 prompt tok/s
  against 1139). With nobody decoding the thread costs nothing (1144 against 1139). The same
  work done one after the other takes 7.3 + 354 / 61.0 = 13.1 s, so the overlap saves 14%.
- The cards are not what limits: 48% + 42% during the window (42% + 31% as upstream), 151 W
  against 133 W.
- Memory: peak 30 MiB higher on each card; 0.39 and 0.37 GiB free at the end against 0.40 and
  0.44.
- The floor is still there without the page-table reset, and upstream's sources name a cause
  (read on 1.5.4). On a split model a prefill chunk of 32 rows or more sends the hot
  CPU-resident experts through `MoeCpuHost.submit_prefill`'s streamed path. Its fused tier
  calls `exl3_moe` without the slot scratch, which the kernel describes as "arrival order,
  which is not bit-reproducible run to run" (`quant/exl3_moe_kernel.cuh`), and its batched
  tier adds with a single atomic `index_add_` unless `EXL3_MOE_RECON_DET=1`
  (`modules/moe_batch_recon.py`). Decode does not take that path. The third session confirmed
  it.

Third session (2026-10-11, night, server stopped; production's flags with 8 worker threads,
MTP 1):

- **Same computation, exactly.** With `EXL3_MOE_STREAM_FUSED_T=0 EXL3_MOE_RECON_DET=1` two
  identical runs as upstream give identical probabilities in every position (so those two paths
  were the whole of the run-to-run noise), and the three comparisons with the thread come out
  identical too: the generating job beside an ingestion against itself alone, the prompt
  ingested on the thread alone, and beside a decode. That is the check to repeat after any
  change to this code, and first of all on this base.
- **The GIL is not what limits the overlap.** A switch interval of 5, 1, 0.2 and 0.05 ms gives
  the generating job the same 26-27 tok/s; py-spy with `--gil` has the generator's thread hold
  it 30% of the run and the prefill thread 7%, and nobody the rest of the time.
- **Neither are the cards**: 37% + 36% in the window.
- **The CPU expert worker is.** A decode step is a pipeline through it, layer by layer: with
  0.84 of the selections served from VRAM, 41 layers of 48 pick at least one CPU-resident
  expert among their selections, and each of those costs the worker 0.46 ms (GEMVs over the
  experts' weights: 0.42 of it, compute-bound on AVX2 at about 2 GB/s per thread) after 0.27 ms
  for the card to stage the inputs. That is the whole 30 ms step; the host, which issues a step
  in 18 ms, is ahead of it. A prompt alone keeps the worker busy about half the time (tail jobs
  of 64 rows, 3-9 ms each, 55 a second). Together they want more of it than there is: the step
  goes to 54 ms and the ingestion from 7.6 to 11.7 s. Fewer layers of a decode step in the
  worker is what changes that (the placement weighted on the decode: *Expert placement that
  follows the decode*, below).
- What was tried against that, none of it worth more than 10% and all of it the same 10%
  (30.6 tok/s and 11.7 s with nothing): a ring and a thread of their own for the decode's jobs
  in the worker (branch `flash-next-native`: 33.5 and 10.7 with 8 threads each), 12 worker
  threads in one ring (32.9 and 10.8), 16 (31.2 and 11.1), the pinned arena, a streaming
  threshold of 2, 4 and 16 instead of 8. Own CUDA streams for the lane were not tried: the
  cards are not the limit, and a block allocated on one stream and last used on another
  (`seq.mtp_carry_hidden` is one) is handed out again in the order of the first.
- **The last chunk of a prompt no longer stops the decode.** It is shorter than a page (a
  recurrent model ingests the last partial page as a pass of its own), and so is a whole short
  prompt; with the floor at a page it was waited for: 0.5-0.7 s without a token for the
  generating job, once per prompt ingested beside it. With the exact check at hand the floor
  went down to 33 tokens (`SIDE_MIN_TOKENS`; `_side_plan` raises it to one past the worker
  host's `stream_min_rows`; `EXL3_SIDE_MIN_TOKENS=256` is the old behaviour). Checked exactly
  with last chunks of 33, 48, 64, 65, 96, 100, 144, 145, 199 and 249 tokens running beside a
  decode: identical probabilities in the three comparisons every time, and the same accepted
  and rejected drafts job by job (the MTP head's cache does not show in the probabilities).
  The generating job's longest wait in the window: 78-120 ms. Below 33 tokens a chunk takes
  the worker's decode-size path and is still waited for: not checked, and short.
- **Several jobs generating.** With 2, 3 and 4 jobs generating (4, 6 and 8 decode rows with
  MTP 1) while a fifth ingests 8289 tokens on the thread, in the exact form: each generating
  job's probabilities identical to the same jobs with no prompt beside them (about 400 positions
  each, all generated during the ingestion), the reader's identical to an ingestion as upstream
  beside the same jobs, and its accepted and rejected drafts the same. With 5 jobs (10 rows:
  past `side_max_rows`, the chunks are waited for) the same.
- **In the server** (a test instance on another port, the 1.5.4 wheel, the v1.5.4 branch): a
  job generating kept 73% of its events per second beside a 15k-token prompt (longest wait
  172 ms), 89% beside six prompts of 88-428 tokens (95 ms), 81% beside two long prompts one
  after the other; a request abandoned in the middle of its ingestion left no error and the
  server went on answering. No error of the thread or of the engine in 36 chunks.

### Two prompt chunks in flight on the two cards (`Generator(prefill_pipeline = True)`, `EXL3_PREFILL_PIPELINE=1`)

Off by default. For a model whose layers are split over two cards, with the MTP head. Measured
on the machine **on the v1.5.4 base** on 2026-10-11 (below): same computation exactly, a long
prompt in a third less time alone, and beside jobs that are generating a quarter faster at their
expense. Not yet run on this base. In the server: `--prefill-pipeline` (rule 26 of its
`AGENTS.md`), which also asks for six worker slots (`ask_worker_slots` puts 6 in
`EXL3_MOE_CPU_SLOTS` and on `moe_cpu_host.TUNING.num_slots` before the first load).

**The idea.** A prompt is ingested a chunk at a time, and a chunk's pass works on one card at a
time: each card is idle for half of it. Every layer keeps its own state (K/V pages, recurrent
state) and needs that state after the previous chunk and the output of the layer before it for
this one, so nothing stops the next chunk from starting on the first card while this one is on
the second:

    first card    A(0)  A(1)  A(2) ...
    second card         B(0)  B(1) ...

with A(k+1) after A(k) and B(k+1) after B(k).

**What the switch does.** It turns the prefill thread on and adds a second one. A chunk's
passes are issued in pieces (`Job._prefill_co`: the `halves` of the yielded function, a
`ForwardCursor` stopped where the pass changes card, held, and finished by the other thread;
then the MTP head's update): the first half by the prefill thread, the second by the new one.
Each thread takes the chunks in the order they are handed over, which is the two rules above.
Up to two chunks are under way (`pipe_depth`, `EXL3_PREFILL_PIPELINE_DEPTH`), of two jobs or of
one. The pipeline is available only where the prefill thread is (so not with promote-on-miss,
a tensor-parallel load or a draft model other than MTP), and needs the MTP head (a prompt chunk
is then one forward pass a cursor can split), exactly two cards and three worker slots;
`prefill_pipeline_state` (`off`, `pending`, `on`, `unavailable`) and `prefill_pipeline_reason`
say which, `side_stats` counts the chunks `piped` (started with another under way) and
`ahead` (of those, planned before the previous chunk of the same job was recorded).

**The MTP head's update takes its turn on the head's card.** The head loads first and is on
the first card. Its update of a chunk needs the end of that chunk's pass, so it follows the
second half; run by the second half's thread it was on the first card while the next chunk's
first half was there, and although the kernels of the two never overlap (one stream), their
memory did: on the v1.5.4 base, 0.4 GiB more at the peak on the first card, and a 179k-token
prompt that no longer fitted at production's split. With the head on the first card the update
is therefore handed to the prefill thread when the second half is done, and runs between two
first halves (`_pipe_head_first`, `_pipe_after`): every prompt pass on the first card, the
head's included, is issued by that one thread. `EXL3_PREFILL_PIPELINE_HEAD=second` is the
earlier behaviour, for telling the two apart.

- `exllamav3/generator/generator.py`: `_pipe_round` and the `_pipe_*` helpers; its docstring
  has the rules. Everything the round of one chunk at a time says still holds, card by card:
  only forward passes leave the generator's thread, a chunk and a decode step overlap only if
  they cannot meet inside a module, a job with a chunk under way gets no checkpoint and no
  decode step, `cancel` lets its chunks finish the half they are in (and tells them to go no
  further). Chunks are recorded in the order they started, each once all its pieces (the
  head's update included) are done.
- **A chunk of the same job is planned ahead of the record of the one before it**
  (`_pipe_ahead`, `Job._prefill_co(results, at)`), and that is refused where it could go wrong:
  several sequences, embeddings, a state whose position a module reads during a pass (the
  state class says: `GDNState.pass_reads_position = False`; on 1.6.2 every side of the
  `PairedState` is asked, and a class that does not say counts as one that reads it), a chunk
  whose record may swap a page (below), and **any position where a recurrent checkpoint is
  due**. A checkpoint stashes the whole state at one position, and the first half of the next
  chunk would have moved the first card's layers past it; so the chunk that ends at a
  checkpoint is recorded with nothing else of its job under way, and the next one starts in the
  round after the checkpoint. That is one chunk without overlap per checkpoint: every
  `recurrent_checkpoint_interval_pp` tokens (32768), every `recurrent_checkpoint_interval` in
  the last two chunks of the prompt (with 4096 for both that interval and the chunk: the last
  two chunks are in line), and the prompt's last page. A prompt of one chunk gains nothing.
- **Two lanes** (`util/lane.py`). The first half issues on lane 1, the prefill thread's own,
  the second on lane 2. Each lane has its passes on one card (the head's update is on the first
  card's lane when the head is there, on the second's otherwise), so its workspaces are that
  card's: the two lanes together hold what the prefill thread's one lane held, and a card still
  runs one prompt chunk at a time beside the decode.
- `exllamav3/model/moe_cpu_host.py`: `split_lanes(lanes = 3)` gives the decode its two slots
  and the two halves an even share of the rest (with the default four: one each; the pipeline
  says so at start-up, and `EXL3_MOE_CPU_SLOTS=6` gives each two, which is what the probe and
  the server run with). Two lanes of prompt chunks at once is new to the host: the rotation of
  the weight-staging slots and the stager's ring of the streamed path were one lane's, and are
  now taken under `_stage_lock`, down to the enqueue of the copy the stager waits for.

**What changed with 1.6.2.** The fork's pipeline was written against 1.5.4; on this base:

- *Two lanes on the streamed prefill path.* With one lane of prompt chunks the streamed path had
  one user at a time (previous section); with two, each card's lane streams its layers' hot
  experts while the other does. They share the rotation of the weight-staging slots (`ws`), the
  sequence number (`wseq`), the stage ring and each slot's pinned side and stage flags; what is
  per device (the VRAM slots, their `wready_ev` / `wconsumed_ev`, `wslot_used`, the copy
  stream) has one lane on it, because each lane's layers are on one card and an MTP head with
  split experts has a host of its own. Under `_stage_lock`: the slot and its sequence number,
  the stage job, the record of the slot's tenant (`wslot_prev_seq`) and, for the staged path,
  the copy stream's wait for the stager, the copy and the `pinned_free` flag, so that every
  stream's waits on the stage flags come in the order of their sequence numbers. Without it two
  stage jobs can land in one ring entry and a lane waits for good
  (`tests/test_moe_host_lanes.py` runs both paths from two lanes against a model of the streams,
  the re-recorded events and the stager). The pinned-arena DMA takes its slot under the same
  lock; what the slot indexes there is per card. The rotation being shared, a lane can get the
  slot it used last, and its next DMA then waits for its last batch's compute instead of
  overlapping it: as on the v1.5.4 branch, measured there with this rotation. A host made
  without `__init__` (upstream's tests of the streamed path) takes a class-level lock. The
  sources a host-waiting lane holds for its copies (`_handoff_held`) are kept by both lanes now;
  the trim of a leaked entry tolerates the other lane's pop.
- *The recurrent states.* Paired target / draft states: the draft side is absent with an MTP
  head (1.6.2 refuses MTP heads with recurrent layers). A prompt chunk's pass reads and writes
  the base row of its own slot (`2 * slot + parity`; the parity moves only in a speculative
  rewind) and stages nothing in `spec_conv` / `spec_beta` / `spec_g`, so two chunks of one job
  in flight touch only that row, and consecutive first halves run on one thread and one
  stream: a chunk's first half reads what the previous chunk's first half wrote, in stream
  order (the toy asserts that a layer takes the chunks in order; chunk halves from two threads
  between a speculative pass and its rewind leave the decode's tokens as they are). A pass
  reads the state's slot and parity (`prepare_for_recurrence`), never its position, which the
  second half advances at its end. 1.6.2's record takes a state found past the chunk's end
  back to it (`rewind_to` on the pair, for the extension an atomic multimodal prefill feeds
  twice); with the next chunk planned ahead and through both cards by then (the generator's
  thread late to the record), that would take back its work. The generator marks the chunk
  under way when it plans the next one (`run.followed`) and the record leaves such a state
  alone; upstream's rewind is untouched with the switch off (the v1.5.4 branch keyed the
  rewind on `atomic_mm_prefill` instead). Checkpoints, `rekey` and the paired stashes stay on
  the generator's thread after the record, with nothing of the job under way; so do the
  persistent disk tier's checkpoint candidates (`note_state`, `state_candidates`).
- *The block table.* The record gives the pages a chunk completes their content hash
  (`Job.hash_completed_page`), and a page still under a random hash may be swapped there for
  another job's referenced page of the same hash: the table rebuilt, the page's reference
  dropped, while a chunk planned ahead still reads the table it was planned with. Such a chunk
  is not followed by one planned ahead. In practice the one page concerned is the last prompt
  page the MTP head leaves out of the cached prefix, which is in the prompt's last chunk anyway.
- *The persistent disk tier.* Write-behind reads the sequence position, which is the start of
  the job's oldest chunk under way until that chunk is recorded, all its pieces done (the head's
  update writes the draft cache's pages of the chunk); it keeps two pages behind it, so no page
  of a chunk in flight is queued (test: with two chunks of one job under way, one planned ahead,
  and the head on either card). The prefetch at `enqueue` touches host slots only. `close()`,
  `clear_queue()`, `cancel()` and a cache taken over stop both chunks at the half they are in
  and wait for it; `close()` then lets both threads go, before it flushes.
- *The MTP head on 1.6.2.* Flash-Next's head is full-attention blocks (QSA) and a MoE; no
  recurrent state of its own. Its input layer calls the target's embedding, which stages
  through a shared pinned buffer only for the decode (`pinned_staging`), not for a prompt
  chunk. With the head on the first card its update also issues a few kernels on the second
  card (the shifted hidden state, the carry) from the first card's thread: tensor ops, stream
  ordered behind the second half that made their inputs.
- *The hybrid draft*: the decode is counted at `1 + max_draft_tokens` rows, as for the thread.
- *Promote-on-miss*: the thread is not offered, nor the pipeline (`unavailable`, with the
  thread's reason).
- *The staggered decode* is never on while a job is ingesting; *graph capture* stays on the
  generator's thread, thread-local, on a stream of its own (a third thread changes nothing);
  *NonFiniteLogits* on a decode step leaves the chunks in flight to `clear_queue()` or `close()`;
  the inline expert *sweep* is held back on any lane but 0 and while either half is under way
  (`current_lane()`, `lanes_busy()`).
- *Memory: the quantized cache's prefill staging.* With a quantized cache, 1.6.2's two-pass
  prefill (`EXL3_QC_STAGING=1`, the default; chunks from 256 tokens on, `EXL3_QC_PF_TWO_PASS_MIN_Q`)
  stages the window an attention layer reads as fp16 K and V in a transient of one fixed size
  per call, freed when the layer returns: for QSA layers in their dense regime (Flash-Next's
  full attention and its head) `sparse_threshold + chunk` tokens' worth of pages, for other
  attention layers the whole pool. So a card holds one such transient per prompt pass under
  way on it. With the head's turn every prompt pass on the first card is issued by one thread,
  and the first card holds at most one of them for prompt chunks, as with one chunk at a time
  (the decode beside them takes the decode kernels, which stage nothing); the second card has
  its own for the second half. With `EXL3_PREFILL_PIPELINE_HEAD=second` the head's update and
  the next chunk's first half can each hold one on the first card at the same time. A chunk
  waiting between the two cards holds its activations as well (about 130 MiB on Flash-Next at
  chunk 4096, measured on the v1.5.4 base).
- `prefill_pipeline_state` is `off` once the prefill thread is switched off at run time.

What it costs when it runs: a second host thread (idle between chunks), one lane more in the
worker's slots, nothing written anywhere, about 130 MiB less free on the first card after a
long prompt (on the v1.5.4 base), and the jobs that are generating: both halves want the CPU
expert worker, which is what limits the overlap of one chunk with the decode already, so a
prompt that goes in faster beside them takes that time from them.

**Measured on the v1.5.4 base** (2026-10-11, server stopped; reports in
`~/exl3-pipeline-20261011/reports` on the machine):

- `experiments/prefill_bench.py` of the v1.5.4 branch, phases `speed,check,pipe`, exact form,
  six worker slots, a 33334-token document in chunks of 4096, split `23:262,25:285`. **Same
  computation**: the nine comparisons identical, the three with two chunks in flight among them
  (the prompt alone, the prompt beside a job generating, that job beside the prompt). The
  prompt alone: 1170 tokens a second in line, 1104 on the thread, **1540 in flight** (21.6 s
  against 30.2). Beside a job generating at 49.3 tok/s alone: on the thread the prompt at 614
  and the job at 29.1; in flight the prompt at 777 and the job at 24.5. The same work one after
  the other takes within 1% of either: in flight moves time from the decode to the prompt. Six
  chunks of ten were planned ahead.
- A test instance of the server, production's flags and split (`23:226,25:253`), the switch
  from the environment, six slots, against the same requests without it: prompts of 5.5k
  tokens the same (770 against 765 tokens a second); 29.5k alone 25.6 s against 33.8; 32k
  beside one job generating 865 tokens a second against 706, the job at 16.0 tok/s against 20.0
  (its longest wait 0.6 s against 0.2 in that run, 0.2 in the run before the head's turn: not
  explained); **179k alone 143 s against 207**. Free memory after all of it: 254 and 166 MiB
  against 384 and 146.
- Before the head's turn the same instance ran out of memory on the first card at the 179k
  prompt (24 MiB free before it). With the update left to the second thread and eight split
  units given back on the first card (`23:234`) the 179k prompt goes in, with 22 MiB to spare:
  the turn is the way out, not the split.
- The probe (`experiments/pipeline_probe.py` of the v1.5.4 branch: the chunks of one job run by
  hand from two threads; 32768 tokens in 8 chunks of 4096, exact form): in line 27.2-27.5 s, on
  the prefill thread 30.9 s, in flight 19.8 s (+38% against in line); the logits of every
  chunk's last token identical bit for bit in the three orders.

Not seen, on either base: images in a prompt (they are ingested one chunk at a time, in the same
pieces), several prompts at once on real traffic, a decode batch above eight rows, the checkpoint
disk tier filling while chunks are in flight. On this base, nothing yet: the exact form of the
bench's check with two chunks in flight is the one to repeat first.

Tests (no GPU, no extension): `tests/test_prefill_pipeline.py` (the real round over the toy
model, whose job turns the rules into assertions: one chunk per card at a time, the head's
update included, a layer takes the chunks in order, records in order with every piece done, a
checkpoint on a whole state and never past one, no page released under a half; the 1.6.2 points
above; and Job's own coroutine planning ahead the chunks it plans in turn, at production's
sizes), `tests/test_moe_host_lanes.py` (the split into three lanes, the streamed path from two
lanes on two cards, the lock's coverage), `tests/test_split_lanes.py` (lane 2 and the sweep,
the halves' translate buffers), `tests/test_lane.py`. Marked `nogpu`; run with `--noconftest`
without the compiled extension.

The bench of the v1.5.4 branches (`experiments/prefill_bench.py`, its phase `pipe`) is not on
this branch.

### A checkpoint that follows the answer (`Generator(recurrent_tail_checkpoint = True)`, `EXL3_TAIL_CHECKPOINT=1`)

Off by default; with it off the generator behaves as upstream. For models with recurrent states
(Qwen3.8-Flash-Next: 36 of 48 layers). In the server: `--recurrent-tail-checkpoint` (rule 25 of
its `AGENTS.md`), asked for only of a library whose `Generator` takes the argument; its tracker
reads `RecurrentCache.metrics["stash_replaced"]` to tell a replacement from a loss.

**What upstream does.** A cached prompt is resumable only from a page that has a stashed
recurrent state. Prompt ingestion stashes one at the prompt's last page boundary (a pass of
its own for it); generation stashes one every `recurrent_checkpoint_interval` tokens counted from
the end of the cached prefix, and one when a job is requeued. **Nothing is stashed when a job
ends** (1.6.2's disk tier writes the last checkpoint of each chain at idle, but it makes none).
The next turn of the conversation, whose prompt is the old prompt plus the answer plus the new
message, therefore resumes at the old prompt's last page and reads the whole answer again,
although its K/V pages are in the cache (three days of production on the v1.5.4 base: one
prefill token in nine, 2460 turns of 636 tokens on average).

**What the switch does.** While a job generates, one more checkpoint is kept, at the last page
boundary it has crossed (`Job.is_tail_boundary`, read by `is_checkpoint_boundary`, so a draft is
cut there as at any other checkpoint). When it moves on a page, the checkpoint of the page
before is taken out first (`RecurrentCache.discard`, counted in `stash_replaced`): the cache
holds one of them per job and never has to make room for it, which matters to a cache with a
disk tier below, where an eviction is a file written. A regular checkpoint due where the tail
one would move takes its place and stays, as upstream's. The job that ends leaves
its tail checkpoint behind, flagged (`"tail": True` in the stashed dict, given by
`RecurrentCache.put(key, state, tail = True)`).

A job whose state was resumed from a flagged checkpoint (`Job.resumed_from_tail`) has a client
that sends answers back as they were, so it also gives up the checkpoint of its own prompt's
last page when its first tail checkpoint comes: a turn then leaves one checkpoint, the answer's
instead of the prompt's, and the count per turn is upstream's. A job that did not resume from
one keeps both (the first turn of a conversation, or a client that rewrites the answer, e.g. by
dropping the reasoning: for that client the tail checkpoint is never used, and costs its
stashes and one more checkpoint per turn for nothing).

A job discards only checkpoints it stashed itself under a key the cache did not have, in RAM or
on disk (`Job._stash_with_tail`). A key that is there already is someone else's checkpoint of
the same text (generated again, or by a twin job): the tail checkpoint is not stashed over it
(the one in RAM is touched, as a regular put would), and a prompt's last-page checkpoint that
was there is not given up. So whatever the disk tier holds under a key a job tracks came from
that job's stash, and it can go with it.

`gen.tail_checkpoint_interval` is the distance in tokens (a page; a multiple of it to stash
less often, 0 = off) and is read at every step; a tail checkpoint still outstanding when it is
switched off goes at the job's next regular checkpoint. `gen.tail_stats` counts the stashes, the
checkpoints taken out of RAM for them, the seconds spent stashing, and what the disk tier held
of the ones taken out (`disk_withdrawn`, `disk_dropped`, `disk_left`, below).

**What changed with 1.6.2.** The fork's checkpoint was written against 1.5.4; on this base:

- *The release path.* 1.6.2 hands a stash back through `RecurrentCache._release`: the host pool
  gets its buffers, the tensor-parallel ranks drop their copies (one handle per side of a
  `PairedState` stash, target and draft), and the release waits while the disk tier still
  writes the stash (`release_after_write`). `discard` goes through it like an eviction (the
  v1.5.4 branch gave the buffers to the pool at once and read a top-level `tp_handle`, which a
  paired stash does not have) and recounts the size as `prune_stranded` does.
- *The persistent disk tier* (`generator/disk_cache.py`). A checkpoint there is a write candidate
  until its anchor page is on disk (`state_candidates`), a write queued or under way
  (`pending_states`), or a file (`states`). A replaced tail checkpoint is one nobody asked to
  keep, so before the release, under the tier's lock (`RecurrentCache._discard_on_disk`): a
  candidate is withdrawn (else `release_after_write` would write it if its page had landed,
  or count it in `state_candidates_dropped`, which the server's tracker reads), a write still in
  the queue is taken back as `DiskPageCache.rekey_state` takes one back (`disk_withdrawn` for
  both), a file is removed (`disk_dropped`; also when the RAM copy had been evicted before, its
  file written then: `discard` returns False and `stash_replaced` does not move, nothing left
  RAM for it). A write the writer thread has started, or that the ranks of a tensor-parallel
  side are doing, cannot be taken back (`disk_left`): the release waits for it and the file
  stays, for `prune_states` (which keeps it if it is `min_state_distance` past the checkpoint
  kept below it). In the steady state none of that is written: write-behind queues no page
  within two of the position, so a tail checkpoint has moved on before its page reaches the disk
  and its candidate is withdrawn. A file is written at once when the page is on disk already (the
  same text again) or when upstream's `rekey` moves the checkpoint (below). The idle flush
  (`flush_cache_to_disk`) writes the checkpoint a job left, as the last of its chain, whatever its
  spacing: what the next turn resumes from, also after a restart.
- *The flag through the disk tier.* A checkpoint whose page is on disk is laid out for writing
  inside `put`, so the flag is set there, before anything reads the stash (the v1.5.4 branch set
  it after `put` returned: such a file would have lost it). The tier keeps a stash's top-level
  ints in the file's header (`top` for a paired stash, the side's `ints` for a bare one) and
  `read_state` puts them back, so a job resuming from a copy read back from disk knows it was a
  tail checkpoint. A file written by a process without the switch has no flag: such a checkpoint
  counts as a regular one.
- *Keys.* 1.6.2 has `RecurrentCache.has(key)` (RAM or disk) and `rekey(old, new)`, which `Job`
  calls when a page that completed during a banned-string hold gets its content hash: it moves
  the RAM entry only when there is none under the new hash, and writes the stash under the new
  hash on disk (`rekey_state`). The job's tracked keys follow only a move that happened; onto a
  key the disk had already, the moved checkpoint is left to the cache like any other (the copy
  on disk is not this job's). A stash under a random hash is never written by the tier.
- *Paired states and fp16.* The stash is `{position, checkpoint_size, target, draft}` (the draft
  side is absent with an MTP head, which has no recurrent layers); the size accounting is the
  cache's own (`update_total_size`, by stash, whatever the sides' dtype).
- *With the other fork features.* A tail checkpoint is offered where every checkpoint is
  (`Generator.recurrent_checkpoint`, on the generator's thread, as `discard` and its disk side
  are): a job of the staggered group in flight, or with a prompt chunk under way on the prefill
  thread or in the pipeline, is passed over and offered it at rest, at the same position. The
  pipeline asks `is_checkpoint_boundary(at = end)` of a position a chunk would end at, and the
  tail boundary answers for that position (`_pipe_ahead` does not plan ahead for a job that has
  generated tokens anyway; a banned-string replay is one). `NonFiniteLogits` comes on a job's
  first token, before any tail checkpoint. A hybrid draft's long verification pass is cut at
  every tail boundary too (production runs MTP alone).

Differences from the v1.5.4 branch besides the above: a key already in the cache is not taken
over (the v1.5.4 branch flagged it and could later discard another job's checkpoint); a replay
that comes back to the job's own tail checkpoint keeps it as the job's.

**Measured on the v1.5.4 base** (2026-10-11, on the production model in the exact form of the
check, a prompt of 8.1k tokens and a forced answer of 1100, two rounds each way): generation
56.0 -> 55.7 tok/s (-0.7%), four stashes of 25 ms, the probabilities of the forced tokens
identical; the next turn resumes 768 tokens later (264 to read instead of 1032) and has its
first token in 0.88 s instead of 1.62. Not yet run on this base, where a stash is a different
size (fp16 states by default) and the disk tier is upstream's: the checks to make first are the
next turn's resume point with the disk tier on, its writes per turn (`tail_stats`, the tier's
`state_writes`), and the same forced-token check. Not seen on either base: real traffic, several
jobs at once, the staggered decode together with it.

Tests: `tests/test_tail_checkpoint.py` (no GPU, no extension: the job's real rules and the real
checkpoint cache over a stub job; upstream's `DiskPageCache`, writer thread and index, over a
stand-in for the CPU page tier, for every state a discarded checkpoint can be in on disk, the
idle flush and the flag through a new process; upstream's `rekey`; the generator's checkpoint
offer beside a group in flight and a chunk under way; the switch). Marked `nogpu`; run with
`--noconftest` without the compiled extension.

### Expert placement that follows the decode (`EXL3_MOE_HIST_PREFILL_WEIGHT`)

Off by default: at the default weight, 1, nothing of it runs and the placement is upstream's,
histogram for histogram. For a model with CPU split experts placed by the dynamic sweep. In the
server: `--moe-swap-prefill-weight` (rule 20 of its `AGENTS.md`), which sets the variable in
`apply_env` and shows `hist_prefill_weight()` as `/status.moe.prefill_weight`.

The library's dynamic placement ranks the experts of a split layer by how often they were
selected, every selection counting one. Prompt tokens outnumber generated ones by orders of
magnitude, so prompts decide the placement alone. But a selection that misses VRAM does not cost
the same in the two: a decode step waits for the worker in every layer that has one (third
session of the prefill thread, above), while a prompt chunk sends its hot CPU-resident experts to
the card and leaves the worker a tail.

Measured **on the v1.5.4 base** (2026-10-11, night, server stopped; Flash-Next 2.05 bpw,
production's flags with 8 worker threads, MTP 1; placement adapted to the bench's own traffic in
each case):

| placement learnt from | share served from VRAM, generating job | its speed alone | share, prompt | its ingestion alone |
|---|---|---|---|---|
| job + prompt, equal weights (upstream) | 0.843 | 60.6 tok/s | 0.953 | 7.64 s |
| the generating job only | 0.945 | 69.7 tok/s | 0.621 | 7.87 s |
| job + prompt, prompt weight 0 | 0.929 | 66.7-68.3 tok/s | 0.627 | 7.77 s |
| job + prompt, prompt weight 0.004 | 0.928 | 67.6-67.8 tok/s | 0.836 | 7.60 s |

With 12 worker threads and weight 0: 67.8-69.9 tok/s and 7.50 s. In the overlap the two weights
give 32-33 tok/s and 11.4-12.0 s (30.6 and 11.7 with equal weights).

- `exllamav3/modules/block_sparse_mlp_cpu.py`: while the weight is not 1, the selections of a
  prefill-size batch (at least the CPU expert host's `stream_min_rows` rows,
  `EXL3_MOE_STREAM_MIN_ROWS`, 32) are counted in `_split_hist_pf` instead of `_split_hist`
  (`_split_hist_apart`, which makes it on first use), and `run_pending_swap_sweeps` folds them
  into `_split_hist` with the weight, for every layer, before it sweeps any
  (`_split_hist_fold`): a reader wrapped around `_split_sweep_layer`, like the server's expert
  tier, sees the histogram the sweep uses, and so does one that reads the layers the swap budget
  does not reach after the sweep. `split_hits()` gives both histograms.
- `EXL3_MOE_HIST_PREFILL_WEIGHT` is read at every sweep, for the fold, and at every prefill-size
  batch of a split layer, for whether to count it apart: the weight in force decides, whenever
  it was set. 1 (the default, and what a value that is not a number reads as): nothing is
  counted apart, no second histogram is made, a sweep neither reads the variable nor folds. 0:
  the placement follows generated tokens only. A small weight (0.004 is one prompt selection in
  250, about the ratio of the two costs on that bench) lets prompts fill the slots the decode
  leaves undecided.
- What the server shows as hit rate is computed from `_split_hist` around a sweep: with a weight
  below 1 it is the weighted share, close to the generating jobs' own.

What it costs when it runs: nothing written anywhere; one more histogram per split layer (E
floats on its card); two small kernels per layer at each sweep; one read of the environment per
split layer and prefill-size batch.

**What changed with 1.6.2.** The fork's weight was written against 1.5.4; on this base:

- *Where the selections are counted.* A decode-size batch's in the fused issue
  (`exl3_moe_split_issue_fused`, which doubles as `moe_split_map` with the histogram in place),
  or in the copy path's map kernel when the fused issue declines a shape or is switched off
  (`EXL3_MOE_SPLIT_FUSED=0`); a prefill-size batch's in the copy path's map kernel, before
  `submit_prefill` (streamed or not). Which is which is the module's own fork between the two
  paths, `bsz < cpu_host.stream_min_rows`, the threshold promote-on-miss uses for a decode-size
  step and `_side_plan` keeps the prefill thread's floor above; it is the v1.5.4 branch's rule
  too. The weight is about what a miss costs, and that is the path's. The decode-size paths are
  untouched: they count into `_split_hist` in their kernels, and the choice is made on the
  prefill path only, in Python.
- *Lanes.* The rows decide, not the thread. A chunk beside the decode, on the prefill thread or
  on either lane of the pipeline, has more than `stream_min_rows` rows (`side_min_tokens` is one
  past it) and is counted apart, translated into the lane's own buffer as before. A chunk below
  `stream_min_rows` (a short prompt, or a prompt's last pages as a pass of their own), which the
  generator waits for anyway, takes the fused decode issue and costs the worker what a decode
  step costs: it counts one per selection, on any thread, as on the v1.5.4 branch. A
  prefill-size batch from the generator's own thread (a chunk run inline, a verification pass
  that wide) is counted apart. A layer's prefill-size batches come from one thread at a time
  (the generator's, the prefill thread's, or the lane of the layer's card), so the second
  histogram, made on first use, needs no lock; a decode and a chunk beside it count into two
  tensors, in the order of the stream they share.
- *The sweep.* `run_pending_swap_sweeps` is reached from the queue's drain (no chunk under way:
  `_side_drop` and `_pipe_drop` come first) and from `_split_swap_tick`'s inline fallback for an
  overdue sweep, which the prefill thread's guard refuses on a lane and while a lane has a pass
  under way (and the server defers to the drain). The fold is inside it, after the all-device
  synchronize: a sweep held back folds nothing, and the counts wait for the sweep that runs.
- *Promote-on-miss* (`EXL3_MOE_CPU_PROMOTE`, spare GPU slots with the pinned arena; off in
  production) replaces the sweep: at every step it decays and snapshots `_split_hist` and reads
  the step's routes from the difference. The weight is the sweep's only: when the first
  registered split module has spare slots (the mode `_split_swap_tick` follows) nothing is
  counted apart and every selection counts one, as upstream; `hist_prefill_weight()` still
  returns what the variable says.
- *The saved expert profile* (`save_expert_stats`, `_expert_stats_load`; the server keeps it off
  and points `EXPERT_STATS_DIR` elsewhere) saves what the sweep ranks by: `_split_hist` with the
  counts kept apart since the last sweep added at the current weight, without folding them in.
  The profile seeds the next load's placement; seeded from the raw counts it would start from
  the prompts' placement and the first sweeps would swap it back, a checkpoint read per swap. At
  weight 1, and under promote-on-miss, it is upstream's.
- *Load and unload.* `cpu_post_load` starts a layer with nothing counted apart; `cpu_unload`
  drops it with the rest of the placement state.

**What changed with 1.6.3.** Upstream's fix of #515 (345eca1c) changes when the saved profile is
written, and f35b51cc (#523) what a seeded layer's sweep reads; the weight is untouched:

- *The save's gate.* `save_expert_stats` now asks for `_split_step_count`, the first registered
  split module's steps since load, to be at least `_STATS_MIN_STEPS` (64), and `_split_swap_tick`
  saves every `_STATS_SAVE_INTERVAL` (4096) of them. On 1.6.2 both read `_swap_tick_count`, which
  every sweep zeroes: with the sweep in charge (one due every 128 steps by default) the periodic
  save fired only after 4096 steps without a sweep, and the save at unload only when the last
  sweep was 64 steps back or more (promote-on-miss never sweeps, so there both fired).
- *Where it now fires from.* `_split_swap_tick` is called by `cpu_split_submit`, on whichever
  thread issues the pass: the 4096th step can be a prompt chunk's on the prefill thread's lane,
  or the decode's while a chunk is under way. That is safe for the weight: the save adds the
  counts kept apart without folding them, so a save in the middle of a window changes nothing
  the next sweep reads (test). Its cost is the profile's own, not measured: a `tolist()` of every
  split layer's histograms (a device sync each) and a JSON file every 4096 steps; from a lane the
  syncs wait for both threads' kernels on the card, as any host wait there does.
- *A save at exit.* At the first split registration (`cpu_post_load` making a new
  `moe_cpu_swap_modules`) an `atexit` handler is registered with a weak reference to the
  `infer_params`, so a process that exits without unloading saves what it learned. It calls
  `save_expert_stats` by global name; after a regular unload the registry is empty and it saves
  nothing.
- *In the server* the profile stays off (rule 20 of its `AGENTS.md`): it replaces
  `save_expert_stats` and `_expert_stats_load` by name, so none of the three saves writes
  anything there.
- *A seeded layer's sweep.* `_split_saved_lists_ref` goes through `_split_perm`, so the sweep
  re-reads the expert it means into a GPU slot. Without a saved profile `_split_perm` is None and
  the function is 1.6.2's.

Differences from the v1.5.4 branch: that branch counted prompt chunks apart at any weight and
folded them in at weight 1, so the sweep ranked the same but `_split_hist` between two sweeps
held the decode's selections only (on this base the saved profile reads it there); here nothing
is apart at 1. Whether to count apart is decided at every batch, not once. Promote-on-miss and
the saved profile are 1.6.x's.

Not seen on real traffic, on either base: the bench has one generating job on one prompt, and
how far a placement learnt from many conversations' output gets is for production's telemetry to
say. Not yet run on this base: the bench's shares and speeds at weights 1, 0 and 0.004 are the
check to repeat.

Tests (no GPU, no extension): `tests/test_placement_weight.py` (the mixin on its own, with a host
that records what it is handed and counts in its fused issue as the kernel does: where the
selections are counted on each path and thread, at the host's threshold; what the sweep ranks by
at weights 1, 0 and in between; every layer folded once and before any is swept, and nothing
folded by a sweep held back; at weight 1 the same histograms, maps and swaps, bit for bit, as
with every selection counted as a decode-size one; promote-on-miss; the saved profile, also
when 1.6.3's periodic save comes from a lane or beside a chunk under way; load and unload).
Marked `nogpu`; run with `--noconftest` without the compiled extension.

### Page-cache hygiene in the disk tier (`EXL3_DISK_CACHE_UNCACHE=1`)

Off by default; with it off the tier makes upstream's system calls and not one more. For 1.6.x's
persistent disk tier (`generator/disk_cache.py`, `Generator(disk_cache_dir = ...)`). The
variable goes in the environment of the process, before the generator makes the tier. The server
(`openai/disk_tier.py`, `ask_page_cache_hygiene`) puts `EXL3_DISK_CACHE_UNCACHE=1` there itself
whenever `--disk-cache-dir` is given and the variable is not in the environment already (an
operator's value, `0` or empty included, is kept), and shows the tier's `uncache` and
`uncache_stats` on `/status.disk_cache`. Also in `doc/env_vars.md`.

**Why.** The tier writes one file per K/V page (about 4.45 MB on Flash-Next, `pages/<hh>/<phash>.kv`)
and one per recurrent checkpoint (about 57 MiB with fp16 states, `states/<phash>.rs`), about
180 GiB a day in production by estimate: buffered writes from the CPU tier's slabs (pinned) and
from stash buffers on the writer thread, reads by four reader threads with `readinto` into a CPU
tier slot. Every byte written or read is in a RAM slot already, so the kernel's page-cache copy
of the file is a second one. On production's 32 GB host, already short of memory, the page cache
also holds the hot rows of the 24.4 GB n-gram table, read per token with `pread`
(`modules/row_table.py`): the tier's copies push those rows out (to be read from the drive again
at the next token that needs them) and, behind them, anonymous memory to swap.

**What the switch does.** `DiskPageCache._uncache_file`, at four points, each only for what this
process writes or reads on its own:

- `_do_write_page`, non-TP page (`markers is None`): after the image is written, the file being
  complete (magic, header and tokens by `_create_page_file`, the image here), and before
  `_publish_page` renames it into place: `fdatasync`, then `posix_fadvise(fd, 0, 0, DONTNEED)`.
  `fdatasync` works on the inode, so it also covers the header and tokens written through the
  other descriptor; the advice covers the whole file, header page included.
- `_do_write_state`, a checkpoint with no tensor-parallel side (`layout["markers"] is None`):
  the same, after the last tensor and before `_publish_state`'s rename.
- `read_page_into`, no tensor-parallel group: the advice after `readinto` and its length check.
  The checksum that follows is computed over the slot, not the file, so it is unaffected and
  still drops a damaged page (test). The header reads of `_read_tokens` are of the same file and
  go with it.
- `read_state`, no tensor-parallel side: the advice after the last tensor is in its pooled buffer.

Why the sync: the advice does not drop dirty pages, and right after a buffered write they all
are (the earlier study's probe: 244 MB of 285 left; here, without the sync, every page of every
file stays: the residency test fails that way). Not used: `POSIX_FADV_RANDOM` / `NOREUSE` (no
effect for this pattern), `O_DIRECT` (header offsets unaligned, the checkpoint layout, and btrfs
falls back to buffered I/O for checksummed files).

**The power-loss window.** Upstream's comment in `_do_write_page`: the file gets its full size
before the image is written, so a power loss between the rename and the data's writeback can
leave a published page file of the right size with zeros in it, which the checksum finds on
read. With the sync before the rename the data is on the drive before the name; that closes the
window for pages, and for checkpoints, which have no checksum. The rename itself is not synced
(no `fsync` of the directory): after a power loss a file can be missing, which the reconcile of
the index at open already handles.

**Errors and platforms.** An `OSError` of the sync or of the advice is counted
(`uncache_stats["errors"]`), logged once, and the file is published as without the switch;
upstream does not sync, so it would not have seen the error either. Making a failed sync a write
failure (temporary file removed, page not published, `_write_failed`'s pause or disable) would be
the stricter choice; not taken, to keep the switch to page-cache hygiene. A write error of the
flush of Python's buffer before the sync is raised, as the close would raise it without the
switch. Without `os.posix_fadvise` (Windows, macOS) nothing is done (a warning when the tier is
made); without `os.fdatasync`, `os.fsync` takes its place.

- The switch is read when the tier is made; `DiskPageCache.uncache` can be changed at run time.
  `DiskPageCache.uncache_stats`: files synced and dropped after a write (`writes`), dropped after
  a read (`reads`), seconds in the sync on the writer thread (`sync_s`), `errors`.
- Not covered: tensor-parallel transfers. The ranks write and read their chunks of the file
  through `np.memmap` (`model_tp_fn.DiskRankWorker`); a write is already `msync`ed there
  (`mm.flush()`), so the advice after the mapping is gone is all the rank side would need, and
  the main process's header page of such a file with it. Left as it is: production runs a layer
  split. Not covered either: the header reads at open of files the index lacks (one page or a
  readahead window each, once).
- Where the generator waits for the writer, it waits longer: `CPUPageCache._evict_one` calls
  `flush()` when every entry of the CPU tier is busy (a write queued for each), and
  `flush_cache_to_disk(wait = True)` and `close()` wait for the queue. The writer goes through a
  page file in 4.7 ms instead of 2.5 (below), so about 950 MB/s instead of 1.8 GB/s into the page
  cache; prompt ingestion produces pages at 3-6 a second, the idle flush in bursts within its
  budget.
- With the tail checkpoint: a checkpoint write under way lasts about 13 ms more, the window in
  which a discard finds it under way (`disk_left`). In the steady state a tail checkpoint is
  withdrawn as a candidate before its page lands, and nothing of it is written.

**Cost, recurring.** At the estimated 180 GiB a day: 40-100 s a day of writer-thread time
(43k page files at 2.2 ms if it were all pages, 3.2k checkpoints at 13 ms if it were all
checkpoints), none on the generator's thread unless it waits for the writer as above; the
advice after a read about 0.4 ms per page read. Nothing more is written: the bytes reach the
drive anyway, sooner. What it frees at any moment depends on the kernel's reclaim and was not
measured on production's traffic.

**Measured locally** (2026-10-17, on the development machine, not the production workstation:
btrfs zstd:1 on NVMe, kernel 7.1.13; `DiskPageCache` and `CPUPageCache` of this branch on CPU, the K/V pool a host
tensor, as in the tail checkpoint's run; 64 page files of 4452352 random bytes and two
checkpoints of 57 MiB per variant, 405 MB; written through `CPUPageCache.store`, read back
through `CPUPageCache.promote` and `read_state` in a fresh tier on the same directory, after
dropping the files' pages, i.e. from the drive in both cases):

| | switch off | switch on |
|---|---|---|
| page file write, mean / p50 / max | 2.48 / 1.93 / 9.5 ms | 4.67 / 4.36 / 27.6 ms |
| checkpoint write | 16.5 ms | 29.5 ms |
| left in the page cache after the writes | 285 MB pages + 120 MB checkpoints | 0 |
| the same 35 s later (past the dirty expiry) | 285 + 120 MB | 0 |
| page read, per file | 3.36 ms (1201 MB/s) | 3.75 ms (1148 MB/s) |
| left in the page cache after the reads | 285 + 120 MB | 0 |

The pages and checkpoints came back bit-exact in both. `sync_s` 0.144 s for the 66 files. The
earlier read-only study measured the same write pattern outside the tier (no checksum) at 1.45
and 3.69 ms per page file.

**What changed with 1.6.3.** In the tier itself only the reading of the free space
(`shutil.disk_usage`, the same `f_bavail * f_frsize` on Linux); the switch's four points are
unchanged. But the fingerprint includes the library's version (`cache_fingerprint`:
`"exllamav3": __version__`), so every change of version starts with an empty disk cache: 1.6.3
opens a fingerprint directory of its own beside the one 1.6.2 wrote (where 1.6.2 ran with the
tier) and never reads the old one. The old one still counts against the root's budget (the new
directory's share is the budget minus its size) until an open of the tier finds the two together
over the budget: `_reclaim_root`, which runs only when the tier opens, then removes the least
recently used sibling whole, if its index has not been touched for 10 minutes. On the development
machine (2026-10-18, upstream's tier over the tests' stand-in, a 50 MB budget, an old directory
of 40 MB): the first open gave the new directory 10 MB; a burst of writes took it to 18.5 MB (a
write reserves its room at once, and only files already written can be evicted for it); the next
open removed the old directory and gave the new one the whole budget. A new directory that never
goes past its share leaves the old one where it is: removing `<root>/<old fingerprint>` by hand
at the switch, with the server stopped, gives the budget back at once.

Tests (no GPU, no extension): `tests/test_disk_uncache.py`. Upstream's `DiskPageCache`, writer
thread, reader pool and index, over a stand-in for the CPU tier with random slots: off by default
and then no call of the four (only the two renames); with the switch on, for a page and for a
checkpoint, a sync of the complete file through the descriptor that wrote it, then the advice on
the whole file, then the rename, in that order and nothing else; a read dropped without a sync,
bit-exact, the checksum still in force; errors of either call swallowed and logged once;
without `posix_fadvise` nothing, without `fdatasync` `fsync`; a tensor-parallel group's page and
a checkpoint's tensor-parallel side left alone. The residency test (`mincore` on the files) runs
where the advice drops pages: on tmpfs it skips, so set `EXL3_TEST_DISK_DIR` to an existing
directory on a real disk (a subdirectory is made there and removed). Upstream's own
`tests/cache/test_disk_cache.py` needs CUDA (`CPUPageCache` asserts CUDA cache tensors).
Marked `nogpu`; run with `--noconftest` without the compiled extension.

### Content-addressed multimodal pages (`Generator(mm_content_hash = True)`, `EXL3_MM_CONTENT_HASH=1`)

Off by default; with it off every page hash, persistable flag, disk record and call is upstream's
(test, and a comparison with master's own files below). For the persistent disk tier and models
with image input. In the server it would be one more constructor argument, asked for as
`--recurrent-tail-checkpoint` is (`generator_takes("mm_content_hash")`): the generator is made at
startup, before any image is embedded, and turning the switch on turns on the embeddings' digests
for the process.

**The problem.** Placeholder ids of image embeddings come from a process-wide counter
(`tokenizer/mm_embedding.py`, `MMTokenAllocator`, from 1000000000 up), and a page's hash is
blake2b over its token ids chained with the previous page's hash. A later process can give the same
ids to another image, so upstream keeps every page from the first placeholder on out of the disk
tier (`Sequence.mm_first_page`, `CachePage.persistable`, `DiskPageCache.write_page` counts them in
`skipped_mm`), and a recurrent checkpoint is written only once its anchor page is on disk (a
candidate until then, dropped when it leaves RAM). An image conversation then lives in VRAM, the
RAM tier and the RAM checkpoint cache only (production: 1 GiB, 17 checkpoints, LRU), and once its
checkpoints are gone the K/V pages left in VRAM are of no use to a recurrent model, which resumes
only at a checkpoint: the whole prompt again. In production, 18-23 Oct, after 30-70 minutes of other
traffic: 7 of 13 resumes of image sessions idle for 10 minutes or more re-read at least 16k tokens
from scratch (99-148k tokens, 132-233 s each), against 0 of 116 text sessions.

**What the switch does.**

- `tokenizer/mm_embedding.py`: an `MMEmbedding` made with digests on gets `content_digest`,
  computed once in `__init__` (`embedding_digest`: blake2b-128, personalization `exl3-mm-emb`,
  over the embedding and deepstack tensors' dtype, shape and bytes, the MRoPE grid and merge size
  and the three alignment fields, each length-prefixed), and its id range is handed out and
  registered with the digest in one step under a lock (`MMDigestRegistry.allocate`, the
  process-wide `global_registry`). The registry is append-only (an id keeps its digest for the life
  of the process: pages holding it outlive the embedding), its ranges increasing and disjoint, so a
  lookup is a bisection and takes no lock (a range is appended before its start). `set_content_digests`
  (the generator calls it) or `EXL3_MM_CONTENT_HASH` decide; an embedding keeps what it was made
  with; one made by the tensor-parallel import has none.
- `generator/pagetable.py`: `content_hash_checksum`, the one helper every hashing site uses with
  the switch on (`Sequence.prepare`, `Job.hash_completed_page`, `PageTable.checksum` for
  `CachePage.update_hash()` and `validate_pagetable`; `tests/test_mm_content_hash.py` reads off
  the sources that nothing else reaches upstream's hash). A page without placeholders hashes as
  `tensor_hash_checksum`, the same function on the same bytes. A page whose placeholders all have a
  digest hashes, under a personalization of its own (`exl3-mm-page`), over a self-delimiting
  input: (previous hash present, id count, run count), the previous hash, the ids with each
  placeholder as -1 minus its row in its embedding (a text id is >= 0), then one (count, 16-byte
  digest) pair per run of placeholders with one digest. The input reads back into the text ids,
  every placeholder's position, row and digest, and the previous hash, so two different images, or
  one image at other rows or other positions, never give the same input; two embeddings of one
  content (an image embedded again in the same process, or in another) give the same one.
- Persistability: a placeholder without a digest falls back to upstream for its page (upstream's
  hash) and for the chain after it (`Sequence.prepare` moves `mm_first_page` to the first
  undigested placeholder's page; before it, every page is persistable). A sequence fixes the way
  it hashes in `prepare` (`Sequence.content_hash`), from the generator's switch when the job is
  queued or requeued; a prompt without placeholders takes upstream's function directly.
- The stored ids. A page file and a RAM-tier entry hold the ids of the process that stored the
  page, which with the switch on can stand for another image in this one. A page restored from a
  tier (`PageTable.allocate_pages`, `own_ids`) therefore takes the claiming sequence's own ids,
  which hash to the page's hash here, instead of the stored ones; with the switch off the stored
  ones, as upstream. `validate_pagetable` compares a job's ids with its pages' by content
  (`same_content`): a page claimed by hash can hold another embedding's ids of the same content.

What else reads ids, checked: the forward passes read the pass's own ids against the job's own
embeddings (`modules/embedding.py`: row = id - `first_index`, the row the canonical form names;
`modules/ple.py` maps placeholders to `mm_token_id` before they enter its history, which is in the
recurrent state, so a checkpoint holds no placeholder id; `block_sparse_mlp_routing.py`,
`architecture/gemma4.py`, `deepseek_v4.py`, `util/rope.py` and `Job.mm_exact_spans` the same), and
restored K/V is never recomputed. Defrag moves pages by their links and never hashes or reads
`page.sequence` (the rehash at `PageTable` "Check individual hashes" is `validate_pagetable`'s,
a debugging aid the library does not call). The partial-page match of models without recurrent
state (`Job._prefill_co`, `count_match_tensor`) compares raw ids, now always this process's: an
embedding of the same content with other ids gives a shorter match, never a wrong one. Requeue
rebuilds the sequences and hashes them again in the generator's mode. `CPUPageCache.store` and
the disk tier's page record take `page.sequence` as they are: unchanged.

**The disk format and the fingerprint are unchanged**, so the first boot with the switch opens
the same fingerprint directory and keeps what is there. The stored tokens are only ever copied
back into a page, which the restore now does with the claimer's ids, and their checksum is over
the stored bytes. A process with the switch off (stock 1.6.3 included, which shares the
fingerprint) never computes a content hash: it never finds a content-hashed page, nor a text page
chained after one (the chain differs), and its own image pages are kept off the disk as before.
What the fingerprint covers (`cache_fingerprint`): the library's `__version__` (the fork says
1.6.3 too), `DISK_CACHE_FORMAT_VERSION`, the page size, the page image layout, and per cache the
model's `config.json` dict and every `*.safetensors` file's name, size, header and seven 64 KiB
samples, the cache layers and their bit widths, the recurrent layers and their state dtype. Not
torch, the compiled extension, the GPU, the image preprocessing (`preprocessor_config.json`,
Pillow) or numerics switches. A vision tower's weights are in the model's safetensors (the server's
BF16 tower swap changes the fingerprint); the embedding digest does not rely on that.

**Why a digest of the embedding, not of the image.** It is what a forward reads, whatever the
architecture: equal digests feed the model equal values, and a change anywhere upstream of the
embedding (tower weights, preprocessing, a library or Pillow version, kernels) changes the digest.
The other way errs on the safe side: a vision tower that does not compute the same bytes twice for
one image gives two digests, a cache miss as today, never a page reused for other values. A digest
of the pixels and of the preprocessing outcome is cheaper but would have to be proven to
determine the embedding (tower weights, numerics, every preprocessing input), per architecture,
and a non-deterministic tower would then reuse K/V computed from other embedding values. Within
one process the server keeps an image's embedding in an LRU keyed by the image bytes, so the same
object (and digest) comes back; recomputed after the LRU let it go, or after a restart, the digest
matches only if the tower is deterministic: so it was for one small image on the cards (below).

**Cost, measured on the development machine** (Ryzen AI 9 365 in powersave, Python 3.14, CPU torch;
not the workstation, whose Python 3.12 has another blake2b implementation; 2026-10-23):

- per image, once, on the thread that makes the embedding (the server's vision thread; hashlib
  releases the interpreter for the whole pass): `MMEmbedding` of 9000 rows of 2048 fp16 values
  (36.9 MB) 128.9 ms against 0.3 ms without, 4096 values (73.7 MB) 257.8 ms, i.e. blake2b at 0.29
  GB/s (OpenSSL's blake2b is 0.67 GB/s and sha256 0.97 GB/s on the same core: a faster hash is the
  lever if this matters). A deepstack tensor is hashed too, after a device-to-host copy;
- per page with placeholders 40 us against 10 us; `Sequence.prepare` of a 150k-token prompt with a
  9k-token image (36 such pages) 11.0 ms against 7.6-7.9 ms; a prompt without placeholders the same
  as upstream; a page completed during generation 13 us against 10 us;
- memory: about 100 bytes per embedding ever made, for the life of the process;
- disk: an image conversation's pages and checkpoints are now written like a text one's, about
  4.45 MB per page and 57 MiB per checkpoint kept (every 4096 tokens, `disk_cache_checkpoint_interval`);
  that week had 13 resumes of image sessions after 10 minutes idle or more, next to 116 of text
  ones. What it buys there: a resume of 99-148k tokens from disk (390-580 page files, about 1.7-2.6
  GB, plus a checkpoint) instead of 132-233 s of prefill, also after a restart; not measured.

**Not carried over: the server's tracker.** `exllamav3-openai-server`'s cache tracker extends a
finished job's chain with `tensor_hash_checksum` from the library (`openai/cache_tracker.py`,
`chain_hashes`). With the switch on that differs for a page completed during generation that holds
placeholders (the prompt's last, partial page when an image ends there, the usual layout for an
image followed by a short question): the session's chain then differs from the page table's from
that page on, with wrong labels on `/status` and in the cache log, and a branch the pruner cannot
find (left to the LRU, as without the pruner). Taking
`content_hash_checksum` when the generator's `mm_content_hash` is on (it equals upstream's hash
for every text page) is the server's change.

Verified (no GPU, no extension): `tests/test_mm_content_hash.py`, 16 tests, over the real page
table, `Job`'s hashing and `prepare_for_queue`, the CPU tier set up over host memory, upstream's
`DiskPageCache` (writer thread, readers, index) and the checkpoint cache, with a second process
stood for by a fresh allocator and registry: the same image in two processes hashes the same,
different images on the same ids and one image at other rows or positions differently; an image
whose last rows are on the prompt's partial page, an answer generated past it (the completed
page's hash), all of it persisted, then found by the second process with other ids (`has`,
`allocate_pages` resuming past the image, pages restored with the second process's ids, the
checkpoint read back), defrag after that and `validate_pagetable`; one process re-embedding the
image; an embedding without a digest; the digest's coverage; the registry; the switch; with it off,
upstream's hashes against an independent copy, `skipped_mm`, the dropped candidate and the stored
ids restored. The eight tests of the new behaviour fail with the switch off. Also, once, outside
the suite: 200 random prompts of text and images, with the switch off, give the same page hashes,
`mm_first_page`, persistable flags and completed-page hash on this branch as with master's own
`pagetable.py` and `job.py`. Fork suite 309 -> 325 passed; upstream's `tests/generator` with a
stand-in for the extension the same test by test (34 passed, 20 failed, 187 skipped); the server's
suite on this branch (harness from `mm-content-hash`) 818 passed / 4 skipped with the stubbed
extension and 820 / 2 with the real worker sources, as on `flash-next-1.6.3`.

On the cards (2026-10-24, production with the server's `--mm-content-hash`, branch `prova-oom-mm`):
a conversation with one 64-token image (7265 tokens, the image in its last pages) had all its 29
complete pages on the disk tier and `skipped_mm` stayed at 0; after a restart the same image got
the same `content_digest` in the new process (`cb5914e3df840c3e`, the server's IMG line), and the
conversation's next turn read its 29 pages and their checkpoint from disk (7424 tokens cached, 309
ingested). Not seen yet: the digest of a large photo (thousands of image tokens) in two processes,
an image session resumed from disk after its RAM checkpoints went in the same process, the
digest's own time on the workstation's Python (the server's IMG line times decode, forward and
digest together: 520-569 ms for the 64-token image).

### An MTP window verified with one readback (`Generator(mtp_batched_verify = True)`, `EXL3_MTP_BATCHED_VERIFY=1`)

Off by default; with it off the generator behaves as upstream. For the MTP head (whatever window
the step verifies while the head drafts, a hybrid round's included). In the server:
`--mtp-batched-verify` (rule 31 of its `AGENTS.md`). Written on this base (2026-10-23); not run on
the cards.

**What upstream does.** With drafts the decode step samples a job's window a position at a time:
a sampler call, then `Job.receive_sample`'s `.cpu()`, a readback that waits for the whole pass and
the sampling behind it, and the next position only once the one before matched its draft; job
after job. The path with one readback for the batch (`sample_pinned`, one synchronize) is only for
steps without drafts. With one draft token, n jobs and the head's acceptance a: n (1 + a) readbacks
a step for the sampling, plus the draft step's two blocking copies. Measured on the v1.5.4 base
(staggered decode, above): a call's sampling is 2-2.7 ms per job, and the cards idle in the draft and
sampling phases, 6-7.5 ms of a 23 ms call with 4 jobs.

**What the switch does.** `Generator._verify_windows`, before the serial loop: every job that can
(`Job.window_sampling_why` returns None) has its window sampled in one call (`Job.sample_window` →
`CustomSampler.forward_window`), position r as `receive_logits` would sample it once `drafts[:r]`
were accepted; the samples are compared with the drafts on the device and the drafts matched from
the first are counted there (`cumprod`); samples and counts of every such job come back through one
pinned buffer, in one copy and one wait (`_host_wait`: an event on the current stream of the logits'
card, where the sampling and the copy are, not a synchronize of the whole card). The serial loop then
takes those jobs' tokens from there and accepts by the count instead of comparing on the host; what
it checks on the host still ends a window where it did: a recurrent checkpoint boundary (the tail
checkpoint's included), a stop token, string or limit, a filter's end, a requeue, the loop detector.
Positions sampled past where a window ends are never used.

- *Masks.* Position 0's is the one `prepare_logit_mask` builds as before (the host side is now kept,
  `Job.host_logit_mask`). For position r every filter is fed `drafts[:r]` (`Filter.feed`, journaled)
  and asked for its mask, then all are rolled back (`Filter.rewind`) whatever happens in between
  (`Job._window_masks`): only filters with a native rollback (`rollback_tokens` overridden:
  `LLGuidanceFilter`) and no trigger token take this path, the others would replay their whole
  journal. A draft the previous position's mask refuses cannot be accepted, nor can anything after
  it; a draft that ends a filter with `eos_after_completed` ends the job if accepted: the rows after
  either repeat the last one, never read. A filter that completes leaves the rows after it
  unconstrained by it, as in the serial path. A filter that raises is put back and the job takes the
  serial path that step (`filter_error`), with nothing drawn. The masks of positions that end up
  rejected are computed for nothing: with one draft token, one mask more per step for a job with a
  filter (the serial path computes them only for accepted positions).
- *Past IDs.* The penalty kernels (`apply_rep_pens`, `apply_pres_freq_pens`) take one row, so a
  stack whose steps read past IDs (`window_past_ids`) must have them in the head of a fused stack
  (`window_head`): the head runs per row on the sequence followed by the drafts before it (one
  upload: the drafts go after the sequence in the job's pinned buffer, past its watermark), the fused
  tail once over all rows. The server's non-thinking set (presence 1.5) is such a stack; with
  presence 0 the step is a no-op and no past IDs are read (although `reqs_past_ids` is set).
- *The random stream.* One draw from the job's generator per window, where the serial path takes
  one per position. The fused sampler keys its Philox noise on (draw, flat index), so row 0 gets the
  noise the serial path would give that position with that draw, and every row's noise is
  independent of the others': each position is sampled from exactly its distribution, and a seed
  gives the same tokens run after run (for the same drafts, as the serial path), but other tokens
  than the serial path with the same seed. Greedy sampling gives the serial path's tokens: each row
  of the fused kernel is computed alone, its vocabulary blocks counted from the vocabulary size, not
  the batch.
- *Serial path*, per job and step (`mtp_verify_stats["fallbacks"]`, by reason): `first_token` (token
  healing, and the first token, whose NaN check is a readback anyway), `forced_tokens`
  (`constrain_output_now`), `probs` (`return_probs`, `return_top_tokens`), `banned_strings` (a
  partial match holds tokens back and blocks what it explored at the next position), `min_new_tokens`
  (stop tokens blocked, a limit the window could cross), `sampler` (a step that keeps state from one
  call to the next, adaptive-P; one that draws from torch's generator, `SS_Sample_mn`; penalties not
  in a fused head, e.g. with `EXL3_FUSED_SAMPLER=0`; a sampler that is not a `CustomSampler`),
  `filter`, `filter_error`, `job` (a job without `window_sampling_why`).
- *With the other features.* The staggered decode samples a group the same way, through its lane's
  staging buffers. A prompt chunk on the prefill thread or in the pipeline shares the pass's stream,
  so the wait waits for what the chunk issued before it, as any readback does (the event waits for
  nothing on the other streams of the card, a synchronize would). The tail checkpoint cuts the
  window on the host as before.
- `readback_stats` counts, whatever the switches, the decode steps that sampled and the readbacks of
  drafted and sampled tokens they and the MTP draft steps made (not a job's first token: its NaN
  check and its timing are a wait each). `mtp_verify_stats`: windows sampled at once, windows left to
  the serial path while the switch is on, the readbacks, the fallbacks.

What it costs when it runs: one draw per window instead of per position (the stream above), the rows
sampled past the first mismatch (a few kernels; with one draft token at most one row per job), the
masks above, one upload of the drafts and one of the past IDs per step. Nothing written anywhere.

Counted on the toy model (one in three drafts accepted, readbacks per step once every job has its
first token): with one draft token 3.3, 4.7, 6.0, 7.3 readbacks a step for 1-4 jobs serially, 3 with
the switch (the draft step's two blocking copies and one for the sampling); with two draft tokens
5.5-9.8 serially, 5 with the switch. Production's acceptance is higher (about 0.79 with one draft
token), so its serial count is higher. Not measured on the cards: what a readback costs there, and
whether the cards' idle time in the sampling phase goes.

Not checked beyond reading: a hybrid round's window, a TP model. An `LLGuidanceFilter` is checked on
CPU only, with a byte-level tokenizer of the test's own (`tests/test_window_sampling.py`, below; the
production tokenizer and grammar not). The compiled `fused_sampler` over a window, with and without
per-row masks (its per-row form is upstream's, used for one row and for batches without drafts), has
a GPU test in the same file, not yet run on the cards.

**The window's distributions** (`tests/test_window_sampling.py`). Speculative sampling is exact only
if every row of a window is sampled from exactly the distribution `receive_logits` would sample that
position from once the drafts before it are accepted, and independently of the other rows. For the
server's two stacks (thinking: temperature 1.0, top-P 0.95, top-K 20; non-thinking: temperature 0.7,
top-P 0.8, top-K 20, presence 1.5), over 4-row windows of designed logits (the token crossing P
carries 3.5% of the mass or more and the cumulative mass is at least 0.01 from P on both sides of it,
in fp32 and fp16, so rounding cannot move the kept set; the non-thinking rows told apart only by the
drafts in their past IDs), with and without a mask per row (rows 1-3 each banning a token of their
kept set, a different rank in each, row 0 none):
- row 0 against a one-row `forward()` with the same seed: the same token, bit for bit (512 seeds per
  stack, dtype and mask; on the card with the compiled extension, on CPU over a float64 stand-in);
- every row's frequencies against the exact distribution computed in float64 (temperature first,
  top-K, then top-P keeping the token that crosses it, 1.6.3's convention): no token outside the
  kept set, every token within 5 standard deviations of its probability, and a chi-square over the
  kept set (bins under 5 expected merged) with a p-value above 1e-6; 16384 windows per case on the
  card, 4096 over the stand-in;
- rows 0 and 1 (the same logits before the penalty and the mask) independent: a chi-square over
  the 4 x 4 table of their top-3 tokens and the rest, p above 1e-6;
- greedy windows give each row's argmax.
The same checks fail, over the stand-in, for noise shared across rows, for 1.6.2's top-P (the
crossing token dropped), for noise offset by a row, for a mask row given to the wrong position and
for drafts missing from a row's past IDs. With `llguidance` installed, a real `LLGuidanceFilter`
(a JSON grammar over the test's tokenizer): it takes the window path, and `Job._window_masks` gives,
position by position, the masks the serial path builds after accepting the drafts before it; a
refused draft's rows repeat row 0, a draft that completes the grammar leaves the rows after it
unconstrained (`eos_after_completed` false) or repeats its row (true). The GPU tests skip without
CUDA and the compiled extension; on the card the file makes about 74000 sampler calls, about a
minute and a quarter at 1 ms a call (assumed, not measured), plus the CPU tests' minute and a quarter:

    PYTHONPATH=/home/daniele/Desktop/exllamav3 /home/daniele/miniconda3/envs/tabbyapi/bin/python -m pytest --noconftest /home/daniele/Desktop/exllamav3/tests/test_window_sampling.py -q

Tests (no GPU, no extension): `tests/test_mtp_decode.py`: the generator over the toy model of the
staggered tests (the same tokens and the same drafts accepted as the serial path, with 1-3 draft
tokens, staggered, with checkpoint cuts and requeues, jobs declining or failing beside the others);
readbacks counted on tensors that stand for device memory (every host read of one, every explicit
wait, every synchronize), against `readback_stats`; the real `Job`'s masks against the serial
path's, position by position, packed and dense, a refused draft, a filter that completes, ends the
job or raises, the past IDs, the reasons; the real sampler's `forward_window` over a stand-in for the
extension against its own `forward()` one row at a time (greedy, masked, with presence), the first
row's draw, the stacks that qualify. Marked `nogpu`; run with `--noconftest` without the compiled
extension.

### The MTP head's update and next draft in one pass (`Generator(mtp_batched_update = True)`, `EXL3_MTP_BATCHED_UPDATE=1`)

Off by default; with it off the generator behaves as upstream. For the MTP head. In the server:
`--mtp-batched-update` (rule 31 of its `AGENTS.md`). Written on this base (2026-10-23); not run on
the cards.

**What upstream does.** The head drafts position K+1 from the token at K and the target's hidden
state at K-1, writing its own cache at K (`iterate_draftmodel_mtp_gen`). After the verification,
every job that accepted A-1 > 0 drafts gets a head pass of its own over K+1..K+A-1, the accepted
drafts paired with the target's hidden states at K..K+A-2 (`draft_model.prefill`, one job after
another); the job keeps the target's state at K+A-1 (`mtp_last_hidden`), and the next iteration
starts with a draft pass over every job at K+A. Each draft token is read back twice, into the drafts
and into the inputs of the next draft token, both blocking copies. With one draft token and n jobs:
up to n + 1 head passes and two readbacks a step.

**What the switch does.** After the verification (`Generator._mtp_update`):

- A job that goes on generating is folded (`_mtp_fold`): one pass of the head over every such job,
  positions K+1..K+W+1 (W the window), the token at K+i paired with the target's state at K+i-1: the
  accepted drafts for i < A, the token sampled after them for i = A, the rejected drafts and a filler
  past it. The head's output at K+A is what the next draft pass would compute: the first draft is
  taken from there (`sample_from_state` on those rows only) and kept on the job with the position it
  was made at (`Job.mtp_fold`), with the head's state there for the draft tokens after the first.
  Nothing is read back. The next draft step (`_mtp_take_folds`), when every job of the step has a
  fold made at the position it stands at, reads them back in one copy, into the drafts and from
  there into the next draft token's inputs, and runs the head from the second draft token on (with
  one draft token, not at all). Otherwise the head drafts for every job, as without the switch, and
  every fold is dropped: a fold is for one draft step. The head's cache then holds at K..K+A what the
  head pass per job and the next draft pass would have written (the draft pass at K+A writes it again
  with the same inputs).
- Jobs that accepted drafts and are not folded get one batched pass over K+1..K+S, S the most drafts
  a row accepted (`_mtp_update_pass`): a job that ended or was requeued (the pages of a job that ends
  may stop at K+W), and every job in a step whose window is not the head's own (a hybrid round, the
  confidence-calibrated draft, which drafts token by token and exports confidences). A job a
  banned-string rewind abandoned gets neither, as without the switch (its carry is gone; drafting
  pauses until the next target pass). `mtp_last_hidden` is set as before for every job, so the
  regular draft pass can always run.
- With the switch the regular draft loop also takes the next draft token's inputs from the copy it
  just read back (host to host), not from a second readback.

**Why writing past a job's accepted tokens is harmless.** The head's cache has the target's page
layout (one page table, `draft_cache.max_num_tokens == cache.max_num_tokens`), and attention reads
stop at each row's position (`cache_seqlens` plus the pass's own tokens). A job's position only moves
forward from K+A, and every position it moves through is written again before a pass reads it: the
next pass that reaches a position writes it first, as the verification pass writes over the target
cache's rejected drafts. The pages are the job's: up to K+W the verification pass wrote them; K+W+1
only for a job that goes on, whose next verification writes up to K+A+W, and `_fold_fits` checks the
page table and the block table all the same (a position on a page outside them would be written
through a zeroed block-table entry, page 0). Rewinds: a banned-string rewind is never folded; a
requeue is never folded and comes back as a new prompt whose ingestion updates the head over it. With
the staggered decode a group is folded when it is sampled, through its lane's staging buffers, its
jobs' pages its own; the pass is issued where the per-job passes were, between two halves of the
other group, which has its own copy of the embedding's pinned output (the cursor's); the fold is read
back at the group's next draft step, as the draft pass was. A fold of r jobs is r x (1 + W) rows, the
verification pass's count, which the prefill thread already counts against `side_max_rows`
(`_decode_rows`), so a chunk's head update and a fold meet inside the head's modules no more than a
draft pass and a chunk did.

**Exactness.** Speculative decoding keeps the target's tokens whatever the drafts. The head's cache
gets the same inputs as without the switch, in passes of other shapes (r x (1 + W) rows instead of
1 x (A - 1) and r x 1): on the cards its values can differ in the last bits, so its drafts, the
acceptance, and with them which target pass computes a position can differ. With greedy sampling
the tokens are then the same only as far as the target's logits at a position do not depend on the
pass that computes them (the exact check says; the serial path of upstream has the same exposure to
a draft that changes).

What it costs when it runs: the fold's rows past each job's accepted tokens (with one draft token,
one or two positions per job, in a pass that replaces two), the head's attention and MoE at a new
shape r x (1 + W) (the graph-captured decode paths capture a graph per shape: more graph slots on the
head's card, the first card, where VRAM is shortest), the rows of the batched update past a row's
own accepted drafts. Nothing written anywhere.

Counted on the toy model (one in three drafts accepted): one head pass a step instead of 1.3-3.3 for
1-4 jobs with one draft token (2 instead of 2.3-3.3 with two); readbacks a step 2.3-6.3 with this
switch alone and 2 with both switches (3 with two draft tokens), for 1-4 jobs, regular or staggered.
Not measured on the cards.

Not checked beyond reading: a tensor-parallel target (the fold calls the head and its
`sample_from_state` as the draft step does), MiMo's depth-specialized heads (the fold is a depth-0
step, `draft_step = 0`, as the update and the first draft step are), the graph captures above.

Tests (no GPU, no extension): `tests/test_mtp_decode.py`. A toy MTP head with a paged cache of its
own, in the target's page layout: a pass writes each position's token and hidden state, then
computes each output from the 16 slots before it; a slot must hold the job's token at that position
and the hidden state the target exported at the one before (or be a later draft step's own write,
read within its round), else the head drafts a token no target samples. With and without the switch
(and with the window verified at once): the same tokens and the same drafts accepted as without it,
no such draft, with 1-3 draft tokens, staggered, across requeues and checkpoint cuts, jobs joining
(the head drafts for all), folds refused by the page check; the check catches a missing update and a
draft taken at the wrong position; readbacks and head passes counted. Marked `nogpu`; run with
`--noconftest` without the compiled extension.

### The MTP draft window chosen from a measured cost curve (`Generator(draft_window = "cost")`, `EXL3_DRAFT_WINDOW=cost`)

Off by default (`"fixed"`); with it off the generator behaves as upstream. For the MTP head. In the
server: `--draft-window cost` (rule 32 of its `AGENTS.md`). Written on this base
(2026-10-23), on top of the two switches above; not run on the cards.

**What upstream does.** Every decode step verifies `num_draft_tokens` drafts per job. The
confidence-target draft (`dynamic_draft_tokens`) stops the head's loop once every row's running
product of estimated acceptance falls below a fixed target (0.4), whatever a position costs. What
it costs on this machine (production on 1.6.3, one request, 15-23k context): W=1 30.1 ms a step for
1.79 tokens, W=3 43.1 ms (37-41 once warm) for 2.52, about 5 ms per drafted position; predictable
stretches reach 70-84 tok/s at W=3 against about 59 at W=1, the rest lose. The decode paths' fast
kernels serve up to 8 rows (jobs x (1 + W): MoE fused decode, hyperconnections), so the curve has a
cliff: W=2 with three jobs, W=1 with five. One window serves a whole batch (or a staggered group).

**What the switch does** (DeepSeek's DSpark policy, with the head's own confidence in place of a
trained head; `generator/draft_window.py`). Each step takes the window W that maximizes

    (n + sum over the n jobs of a_j1 + a_j1 a_j2 + ... + a_j1 ... a_jW) / T(n, W)

- *Estimates.* a_jk is job j's k-th draft's acceptance given the ones before, from its winning
  logit through a `DraftConfidenceCalibrator` of the policy's own (the dynamic draft's bins and
  decay; the target it is made with is not used). The head exports the logit with each position
  (`export_draft_conf`), and it comes back with the drafts: a non-blocking copy into pinned memory
  just before the drafts' blocking readback on the same stream (`_dw_stage_conf`). The fold of
  `mtp_batched_update` exports it too, kept on the job with the draft, and `_mtp_take_folds` reads it
  back in the drafts' copy. No readback of its own: `readback_stats` counts the same per drafted
  position as before.
- *Labels.* Every verified position: with the window sampled at once (`mtp_batched_verify`) the
  target's sample at position r was made as if `drafts[:r]` were accepted, so every position of the
  window is a label, past the first mismatch too (with a filter, up to it: past a refused draft the
  rows repeat the last mask); serially, the accepted ones and the first that was not. A step that
  verifies nothing labels the first draft with the token the target sampled, so labels never stop
  (a calibrator that decays without labels goes back to its burn-in, whose estimate is 1.0: tried,
  it oscillated). Soft labels (the target's probability of each draft, DSpark's calibration) are
  not used: the fused sampler does not expose the processed distribution, a raw softmax would be
  wrong for greedy requests and for top-P/top-K stacks, and the hard labels of whole windows give
  n x W labels a step (the calibrator's burn-in, 64 labels, is a few steps).
- *The choice, as the head drafts.* The head drafts a position at a time, each with a readback
  already. After each position the windows drafted so far are scored with their estimates, the
  longer ones with a prediction (each job's next positions as likely as its last one), and drafting
  stops once no longer window is predicted to beat the best one drafted (`DraftWindowRound`). A
  position drafted and then not verified (its estimate turned out low) is a head pass for nothing,
  and such a step is not timed.
- *W = 0.* Allowed: past the 8-row cliff (five or six jobs, production runs `max_batch_size` 6)
  only W = 0 stays under it, and a stretch the head gets wrong pays nothing for its rows. The head's
  first draft is still made (its pass writes the head's cache at the job's position, and its logit
  decides between 0 and more; with `mtp_batched_update` it comes from the last fold at no cost); the
  step then takes upstream's path without drafts (one token per job, one synchronize).
- *T(n, W).* The median of the last 32 wall times of steps with n jobs and window W, regular and
  staggered steps apart. A step is timed from the end of the last decode step's sampling to the end
  of its own (`time.perf_counter()` where the host has just waited for the pass: after
  `_verify_windows`' wait with the switch, after the serial path's last readback without), so it
  holds everything a step costs, the last step's head update and what the caller does between two
  `iterate()` calls included. It is kept only when nothing else ran in it (`_dw_end`): the same jobs
  (none started, ended, requeued or cancelled), no prompt ingested in the iteration, no tail
  checkpoint taken, no cap on the window, no position drafted past the window, and in a staggered
  decode a call that also issued the next group (a staggered call's time goes to the group it
  sampled, whose window was chosen a call earlier). The first time of each (n, W) is dropped (the
  first step at a shape captures its graphs); a change of either MTP decode switch forgets the curve.
  The cliff is read from the times, not written anywhere.
- *Exploration.* A window with fewer than 4 times for these jobs is taken to time it (the one with
  the fewest first, only in a step that can be timed); a window the policy does not take is timed
  again after 512 steps, then after twice as many each time it still is not taken, up to 16 x 512,
  one such refresh at most every 32 steps: a window past the cliff is taken about six times in 20000
  steps once known (`tests/test_draft_window.py`). Until a window is known for these jobs the
  policy takes the longest, as without it; until the calibrator has its labels the estimates are 1.0.
- *A prompt chunk beside the decode* (`prefill_thread`, `prefill_pipeline`). A chunk may overlap a
  decode step of at most `side_max_rows` rows (8). `_decode_rows` counts the step at the narrowest
  window the policy may take, and while a chunk is under way the draft step caps the window so that
  n x (1 + W) stays within it (`_dw_begin`): with three jobs and W up to 3, chunks now run beside a
  decode of W <= 1, where at a fixed W=3 (12 rows) every chunk was waited for. With the hybrid draft
  the count stays at the widest (its window is decided after the prefill round).
- *With the head's update in one pass.* `_mtp_fold` folded only a window of `num_draft_tokens`
  without the calibrator. Its content does not depend on who drafted (position K+i pairs the token
  there with the target's state at K+i-1) and its width is the step's, so a window the policy chose
  is folded whatever its width; a window of 0 over K+1 (the head's cache at K was written this step,
  by the draft pass or by the fold whose draft the step took). Each window is then a shape of the
  fold as well as of the verification pass.
- *Other features.* The staggered decode chooses a window per group, at its draft step. A hybrid
  round that replaces the window is not the policy's: not timed, not labelled, not folded. Checkpoint
  boundaries, the tail checkpoint, stop conditions and requeues cut a window on the host as before;
  a banned-string rewind gives no labels. The confidence-target draft is not used while the policy
  is on. The switch can be flipped between steps (attribute of the same name).
- *Exactness.* Speculative decoding keeps the target's tokens whatever the window. A position is
  then computed by verification passes of other widths from step to step; on the cards their logits
  can differ in the last bits (the exposure two fixed windows already have against each other), so
  greedy text is the same only as far as no near-tie falls there. Scenario 9 of the exact check
  (below) exercises the policy, under the test hook of the next point.
- *A test hook, not for production: a window schedule* (`EXL3_DRAFT_WINDOW_SCHEDULE`, e.g. `3` or
  `0,1,2,3`; `Generator.draft_window_schedule`, a list or such a string, None for none). The windows
  the policy takes depend on the step times it measures and on what it explores, so two processes,
  or two runs in one, take different windows, and a run is not repeatable. With a schedule every
  decode step takes the next window of it, cycled per step (a staggered decode's group counts as a
  step), whatever the estimates, the times and the exploration; the window is clipped to what the
  step allows (`draft_window_set`, `num_draft_tokens`, and the cap beside a prompt chunk: the largest
  allowed window not above the scheduled one, else the smallest allowed), and the head drafts exactly
  that window (a window of 0 still drafts its first position, as above). Everything else is the cost
  path's: the confidences exported, the labels, the step timed when clean, the fold, the readbacks;
  `draft_window_stats["scheduled"]` counts such steps (not `explored`). The schedule starts again
  from its first window when the generator is given a job while it has none, and when it changes.
  Only with `draft_window = "cost"` (a fixed window takes no window from anybody; the generator says
  so at construction). Unset (the default, and production) the policy never enters the hook's code
  (`tests/test_draft_window.py::test_without_a_schedule_the_hook_is_never_entered`: the same rounds
  and the same stats with the hook replaced by one that raises).
- `draft_window_set` (`EXL3_DRAFT_WINDOW_SET`, e.g. `0,1,3`) limits the windows; default 0 to
  `num_draft_tokens`. Above it nothing is taken (buffers and the cache's `max_history` are sized by it).
- `draft_window_stats`: steps, windows taken (`chosen`), explored, scheduled, capped, overridden, timed, positions
  drafted, verified and accepted, expected against realised tokens of the steps it chose freely,
  labels, `curve` (`"r:n:W"` -> samples and median ms) and the calibrator's state.

What it costs when it runs: the head's winning logit (`max` in place of `argmax` over its logits),
one non-blocking copy of n floats per drafted position, Python per step (the policy's part, 22 us a
step with three jobs and 20 bins on the development machine), a head pass now and then for a
position not verified. On the cards each window
is a shape of its own: with windows 0-3 the verification pass takes 1-4 tokens per job instead of
the one width of a fixed window, and the fold as many (graph slots of the attention and statics per
shape and card, the head's on the first card, where VRAM is shortest: not measured; a window of 0
is a 1-token verification shape production never ran with MTP). The main cache needs
`max_history = num_draft_tokens` as for a fixed window of that size: on this base the recurrent
state is two rows per slot whatever `max_history` (1.5.4 kept a whole state per history step: the
111 MiB a token and slot of the server's rule 5); what grows is each GDN layer's conv ring and
staged scan inputs, 4 x fdim_qkv + 6 x num_v_heads bytes per layer, slot and history token
(`GDNLayerState`), on Flash-Next (36 GDN layers, fdim_qkv about 10k, 48 value heads: from its
57 MiB checkpoint) about 1.4-2 MiB per slot and draft token, 17-24 MiB with six slots from 1 to 3
draft tokens. Nothing written anywhere.

On the toy (`tests/test_draft_window.py`, acceptance one in three at every position, a clock the
target's pass moves by a synthetic cost of its shape): one job on a flat curve takes W=3, two jobs
on a steep one W=0, three jobs on a curve with a cliff past 8 rows W=1 and the windows past the
cliff only to time them; expected tokens within 10% of realised. The toy's acceptance is not
production's (about 0.79 at the first position, about 0.58 after it from the 1.6.3 run above), so
it says nothing of the tok/s on the cards.

Not checked beyond reading: a TP head's confidence (`tp_dispatch_lm_head_argmax(return_max = True)`);
a head that exports no confidence (the Qwen 3.5/3.8 heads do, through the target's head) gets the
calibrator's overall rate, which without labels stays 1.0, so the policy then weighs every draft as
certain; the graph shapes' VRAM, whether the step time's median is stable enough on the cards with the
prefill thread and the expert sweeps around it.

Tests (no GPU, no extension): `tests/test_draft_window.py`: the policy on its own (the curve's
medians, clean steps only, the first dropped; flat curve, steep curve with low acceptance, a cliff,
the head stopping early, jobs counted together, no measure, restricted windows, a cap, the switches;
20000 simulated steps: every window timed, the cliff rarely after); the generator over the toy with
a clock its passes move: the same tokens with the policy as without, with 1 and 3 draft tokens, with
and without either MTP decode switch, staggered, switched on and off while jobs run; the window
following flat, steep and cliffed curves; the head's confidence deciding step by step; every
readback counted, one per drafted position and one for the sampling with both switches; the window
capped beside a prompt chunk on the prefill thread, the chunks running beside where a fixed W=3
waits for them; the test hook: its parsing, the schedule taken in order whatever the estimates,
clipped to the windows allowed and to the cap, started again with each stretch of work and when it
changes, never entered without a schedule, and over the toy generator with schedules `3`,
`0,1,2,3` and `3,0,3,1`, with and without either MTP decode switch, one to four jobs, staggered: the
same tokens as a fixed window, the windows taken those of the schedule, the positions drafted
`max(1, W)`. Marked `nogpu`; run with `--noconftest` without the compiled extension.

### Raw keys of the block being built only (`Cache(qsa_raw_tail = True)`, `EXL3_QSA_RAW_TAIL=1`)

Off by default; with it off the cache layers, the kernels and the decode graph are upstream's. For
the QSA attention layers of Qwen3.8-Flash-Next (12 of its 48 layers, and the MTP head's). The switch
is the cache's, read when the `Cache` is made, so before the model loads: `Cache(qsa_raw_tail = True)`
(or a ring size in tokens), else `EXL3_QSA_RAW_TAIL=1` in the environment. In the server:
`--qsa-compact-raw-keys` (rule 33 of its `AGENTS.md`), passed to both caches only by a library whose
`Cache` takes the argument. Also in `doc/env_vars.md`.

**What upstream keeps.** `cache/qsa.py`, `QSAPlanes`: per page and QSA layer a raw indexer key for
every token (`raw_k`, 256 x 128 fp16, 64 KiB) beside the pooled keys (one per 4-token block,
16 KiB). The raw plane has one reader, the pooled-key rebuild of the blocks an append touches
(`_qsa_pool_update_kernel`: in the decode graph's slots, `attention.cpp` with the plane `set_qsa`
was given, and in `QSAIndexer.update_planes`), and of those blocks only the one the append starts
in has raw keys from before the append: at most compress_ratio - 1 = 3. The other uses of `raw_k`
in the library: `bc_attn.py` hands the plane to the slot, `attn.py` zeroes page 0 for the
autosplit's measuring pass, `build_mask` and the `*_ref` functions are the parity tests' torch
references (the non-cached path projects its raw keys and pools them without a plane). At
production's 370k-token cache that is 1445 pages x 64 KiB = 90 MiB per layer, 1.06 GiB for the 12
layers (5 on the first card, 7 on the second with production's device split) and 90 MiB more for
the MTP head (first card), plus 64 KiB per QSA layer in every page image the CPU and disk tiers
store.

**What the switch does.** The raw plane becomes a ring of 32 tokens per page (`RAW_TAIL_DEFAULT`;
8 KiB per page and layer): token t of a page at slot t % 256 % 32, i.e. the last 32 positions
written to the page. With 32 it saves 56 KiB per page and layer: at 370k 79 MiB per layer, about
474 MiB on the first card (5 layers and the MTP head) and 553 MiB on the second (computed from the
shapes, `QSAPlanes.raw_tail_saved_bytes`; `Cache` prints the total). Three kernels in
`modules/attention_fn/qsa_triton.py`, sharing the full plane kernel's arithmetic statement for
statement (`_qsa_pool_tail_block`: fp32 sums in token order, mean -> fp16, norm -> fp16, rope in
fp16), so the pooled keys, the block selection and the attention output are bit for bit the full
plane's:

- `_qsa_raw_tail_append_kernel`: the append, the last 32 rows of each sequence written.
- `_qsa_pool_update_ring_kernel`: the decode graph's pool update, every raw key read from the ring,
  after the append. It takes the full plane kernels' runtime arguments, in the same order, so the
  slot launches it unchanged; the slots' appends (up to 16 rows, `MAX_QLEN`) and the 3 tokens
  before them fit 32 distinct slots (`BCAttn._configure_qsa` asserts it per slot).
- `_qsa_pool_update_rows_kernel`: `update_planes`' pool update, for appends of any length (prompt
  chunks, the eager fallbacks): the append's own raw keys from the staged rows, the 3 before it from
  the ring. It runs before the append, whose writes may take those slots (a chunk's last 32
  positions can share slots with the block it started in).

**Where the ring holds what is read.** When an append starts at p, the ring must hold the raw keys
of [p - p % 4, p) as their last writes left them. A page is written in order from its start, so
that holds unless something wrote 32 positions further on in the same page since. The cases:

- *Speculative passes.* A pass of L rows at p is taken back to p' > p (the token it follows is
  kept), after positions up to p + L - 1 were written: the next pass reads [p' - 3, p'), so
  L + 3 <= 32, the same bound as a graph slot's. The generator checks its widest pass
  (`max_draft_tokens + 1`: the MTP, n-gram or hybrid draft window) against the ring of both caches
  when it is made (`check_raw_tail`, a ValueError): with 32, at most 28 draft tokens (production:
  1). The MTP head's cache writes its drafts one row at a time at K..K+W-1 and then the accepted
  K+1..K+A-1 again from the target's states: the same bound.
- *Banned-string rewinds.* GDN states have no in-place rollback (`rollback_capacity` 0), so on
  Flash-Next a rewind restores a page-aligned checkpoint and replays, rewriting every position from
  there. A model without recurrent states rewinds by truncation, to any depth; and its prompt
  cache copies a partial page (`Job._prefill_co`'s partial-page reuse, `copy_page` with a token
  count off a block boundary) and goes on in the middle of a block. `Cache` refuses the ring for
  such a model with a warning and keeps the full plane; also for a model whose prompt passes can
  run past a chunk's end and be taken back (`atomic_mm_prefill`). No QSA model is either today.
  The MTP head has no recurrent states of its own (`Qwen4ExpMTPModel`'s caps) and its cache
  follows the target's: an MTP head's cache takes the ring, and the generator refuses to start
  (`check_raw_tail`, `raw_tail_refusal` of its own model) if its model is one of the two.
- *Prompt cache, page sharing, tiers.* Only complete pages are reused (by hash in VRAM, from the
  CPU tier, from the disk tier), shared between jobs or stored, and a job starts at a page boundary;
  so do a requeued job, a resume from a tail checkpoint and a replay. A complete page's raw keys
  are never read again. `copy_page` copies the whole ring row (its one partial-page caller is the
  refused case above).
- *Defrag* rotates `get_tensors()`, which leaves the ring out: it runs with no page referenced, and
  a partial page's ring is read only by the sequence writing it.
- *Tensor parallelism.* `tp_export` carries the ring to the rank that holds a QSA layer whole (the
  ring is the same per-layer storage there). Not run.

**Page images and the disk tier.** The ring is not in `get_tensors()`, so the CPU tier's slots and
the disk tier's page files lose the raw keys: 64 KiB per QSA layer (832 KiB with the MTP head's
layer) of every page image. The disk tier's fingerprint covers the image layout
(`cache_fingerprint`'s `segments` and `image_size`), so the switch on and off are two fingerprint
directories: a page of one layout is never read under the other; going back opens the old one, and
the one not in use keeps its share of the root's budget until removed (rule 27 of the server).

- `Cache.qsa_raw_tail` is the ring in effect (0: off, refused, or no QSA layer); a layer's
  `raw_tail` the same per layer. `util/memory.py`'s report counts the ring as cache.
- `update_planes_ref` (the torch reference of the parity tests) has the ring's form too.
- The ring's size can be chosen (`qsa_raw_tail = 64`, `EXL3_QSA_RAW_TAIL=64`) for longer drafts;
  below 19 tokens (16 + 3) it is refused.

**What has been checked, and where** (2026-10-23, the development machine, no GPU): the kernels in
Triton's interpreter against the full plane's, the same random appends into both, the pooled planes
compared bit for bit after every append (prompt chunks of 1 to 4096 tokens starting inside blocks
and pages, decode passes of up to 16 rows with drafts rejected and accepted, the MTP head's
pattern, a replay from a page boundary, pages taken over by another sequence and shared by prefix,
several sequences per call, with and without a pass-through part in the pooled keys' rope), the
graph slot's launches with the slot's own arguments, the eager path through `update_planes`
itself; two checks that the comparison sees a difference (the eager kernels in the other order; a
pass longer than the ring serves). Then `BCAttn._configure_qsa` compiling for sm_120 with the
triton wheel's ptxas: the ring's kernels in place of the full plane's with the same runtime
signature, and the floating-point instructions of the ring's pool updates (graph and eager) the
same as the full plane's, in the same order, modifiers included. Nothing has run on the cards: the
compiled kernels' results there, the C++ slot launching them, the memory freed per card, and a
real model's outputs with the switch on and off are the GPU check below.

**The GPU check** (on the workstation, production stopped, this branch checked out in
`~/Desktop/exllamav3`; fish, one line each). First the kernels on the card against the full
plane's, no model (with CUDA the test file runs them there; it needs `pytest` in the environment):

    /home/daniele/miniconda3/envs/tabbyapi/bin/python -m pytest --noconftest /home/daniele/Desktop/exllamav3/tests/test_qsa_raw_tail.py -q

Then the model, scenario 8 of the exact check: the plan and the checks, then the run (about 13
minutes by the estimate: two loads). `REPORT.md` in the run directory says, per compared run,
whether the logits are identical bit for bit with the ring and without, and per card the memory
each load left free, the difference and what the library says the ring saves:

    /home/daniele/miniconda3/envs/tabbyapi/bin/python /home/daniele/Desktop/exllamav3/experiments/exact_check.py --dry-run --scenarios 8 --out /home/daniele/exl3-exact-check-qsa
    /home/daniele/miniconda3/envs/tabbyapi/bin/python /home/daniele/Desktop/exllamav3/experiments/exact_check.py --scenarios 8 --out /home/daniele/exl3-exact-check-qsa

Tests: `tests/test_qsa_raw_tail.py` (no GPU, no extension; the kernel tests need triton, and run on
the card where CUDA is available: on the workstation they are the first GPU check). Marked
`nogpu`; run with `--noconftest`. On a machine with no CUDA the file switches Triton's interpreter
on for the session when it is the first to import triton (the switch is read at the import); the
compile checks run in a process of their own.

### The decode step's host work, kept between calls (`EXL3_HOST_FASTPATH=1`)

Off by default. Read once, when each module that has a fast path is imported (`util/lane.py`,
`util/tensor.py`, `modules/hyperconnections.py`, `modules/block_sparse_mlp_routing.py`,
`modules/block_sparse_mlp.py`, `modules/block_sparse_mlp_cpu.py`, `modules/gated_delta_net.py`), so it
is set in the environment before the library is imported and holds for the whole process: no
`Generator` argument, no server flag (in the server's wrapper an `export`). Branch `host-fastpath`,
on `master` at fcd111d5. Also in `doc/env_vars.md`.

**Why.** The night profiling of 2026-10-24 (the server's `docs/REPORT-profiling-20261024.md`, 2.2, 2.4
and 2.6 a): the thread that issues a decode step (`exl3-generator-worker`) spends 25-30 ms of each
step preparing and launching about a thousand kernels (measured with the split profiler on, whose own
share is 3-4 ms of it: below), and with three requests staggered it waits for the cards 5-7% of the
time: the host is the step. Inclusive shares of that thread (py-spy): the
MoE module 27%, the gated residual sites 24% (96 a pass, ~90 µs each), the workspace cache's lookups
11%, GDN 8%. A good part of it is the same bookkeeping, call after call, for the same shapes: format
a workspace key string, look it up, slice the backing, view it to the shape.

**What the switch changes**, site by site. The same kernels, with the same arguments (for a tensor:
the same storage, offset, shape, strides and dtype), in the same order; none of the code below calls
a stream, an event or a synchronize. Only the Python around the calls changes.

- *Workspace cache* (`util/tensor.py`, `GTensorCache.get_bucketed`): the key string `make_key`
  formats for a (device, backing size, dtype, tag, lane) is kept (`fast_keys`); a later call looks
  the entry up by it, counts the reference and slices the backing as the plain lookup does. Kept
  only for a `torch.device` or `str` device, a `torch.dtype` and a `str` tag, where two equal tuples
  format the same string; anything else, and an entry not yet made, takes the plain path. `get` (the
  same code) and `drop_all` count every entry made and every emptying in `fast_gen`, under a lock.
- *Lanes* (`util/lane.py`): `current_lane()` and `host_waits()` read the thread-local's own dict
  (`_tls.__dict__.get`) instead of `getattr(_tls, name, default)`, which on a thread that never
  entered a lane raises and catches an AttributeError inside, at every workspace lookup and CPU host
  slot. The same values (`Lane.enter` is the only writer).
- *Gated residual sites* (`modules/hyperconnections.py`, `GatedResidual._mix`, `mix`, `apply_`:
  Qwen4Exp's 96 sites a pass, the final mixer, the MTP head's): for a decode-size call (rows up to
  `FUSED_MAX_R`, an fp32 contiguous stream stack: the fused pair's domain) the three workspace views
  `ext.gr_mix` writes (dots, post, mixed), their (b, s, ...) views and the weighted-copy workspace
  `apply_` hands to the next site are taken once per (device, b, s, table rows, streams, hidden,
  combine form, lane) and kept in `g_tensor_cache.fast_views` with the `fast_gen` read before they
  were taken; when the count has moved they are taken again. `proj_h[: proj_m]` is kept per site
  while `proj_h` is the same tensor and `proj_m` the same count (`_prepare` and `tp_import` make a
  new `proj_h`, `unload` drops the kept view). In `apply_`, a `post` that is the (b, s, H) view this
  site's mix returned is passed as the (R, H) view kept beside it, which is what
  `post.reshape(R, H).contiguous()` returns for it; any other `post` takes that expression. The
  conversion and capture passes (`quant_preserve`, `capture`), larger row counts, stacks the plain
  path converts first (fp16 MTP stacks, non-contiguous ones) and the mHC `HyperConnection` are left
  as they are.
- *MoE routing* (`modules/block_sparse_mlp_routing.py`, `_routing_buffers`): the three views of a
  multi-row call up to `ROUTING_CACHE_ROWS` (router logits, selected experts, weights) kept per
  (device, rows, experts, top-k, lane), as above. One-row calls have their own buffers and never
  went through it.
- *MoE fused decode tier* (`BlockSparseMLP.forward`, `BlockSparseMLP_CPU.cpu_split_combine`):
  `experts_cfg.out_bszn[:bsz].view(eshape)` kept per (bsz, eshape) while `out_bszn` is the same
  tensor (the output the fused decode kernels write, made at load and baked into `self.bc`), and the
  (rows, H) view of it that the CPU split's collect is handed (`submit_collect_fused`) kept beside
  it. What goes to the CPU expert worker, and in what order, is unchanged.
- *GDN* (`GatedDeltaNet.forward`'s fused decode path): it called `GDNLayerState.staged_views(bsz,
  seqlen)` only for the record of a speculative pass's shape (for the rewind) and discarded the three
  views; with the switch it calls `staged_shape(bsz, seqlen)`, the same check and the same record
  without the views (only for a pass not fed a token at a time, `spec_step is None`, and a state
  that is exactly a `GDNLayerState`).

**What it relies on.**
- A view is kept only of a tensor something else already keeps alive (a workspace cache entry, a
  module's buffer). Keeping one of a tensor that would otherwise be freed would change what the
  caching allocator hands out later, so later addresses. Workspace entries are only ever added,
  never replaced, so a view taken while the count is still the current one is a view of the backing
  `get_bucketed` would return now; the count is read before the views are taken, so an entry made
  meanwhile, on this thread or another, only makes the next call take them again.
- The lane is in every key, as in the workspace cache's own: the prefill thread's and the pipeline's
  passes keep their own views of their own workspaces. Dict reads and writes are atomic under the GIL;
  the count moves under a lock.
- The same view objects are now returned call after call where the plain path made new ones: safe
  as long as nothing changes a view's metadata in place. Nothing in the library does outside the
  loader (`resize_`, `as_strided_`, `set_`, `t_`, `squeeze_`, `unsqueeze_`, `transpose_`: one `set_`,
  in `loader/safetensors.py`, on a tensor of its own).
- A call served by kept views counts no reference on the workspace entries (the plain path counted
  one per lookup). Nothing reads the counts: `util/memory.py` lists the entries and drops the count,
  `model_ls.py` reads the keys.

**With the switch off.** The fast functions are defined under `if HOST_FASTPATH:` at module or class
level, and the plain ones are the code that was there. What the switch-off path runs that it did not
before: a module-constant test in `BlockSparseMLP.forward`'s fused tier, in `cpu_split_combine` and in
the GDN fused path's record. `GDNLayerState.staged_shape` exists and nothing calls it.

**The split profiler is not free.** Upstream's `EXL3_SPLIT_PROF=1` (fd11c82e, in 1.6.3; the night's
counters boot C set it with `EXL3_MOE_CPU_PROF=1`) brackets every CPU-split submit and collect with
two `torch.cuda.Event`s recorded on the current stream. Its own work (creating, recording and reading
the events) is 9.7-11.7% of the generator thread's samples in the night's three counters captures (one
request at 60k and at 125k, three at 60k), 10.5-15.2% of its issue time, and all of the 2.7% the
report counts in `torch.cuda.is_available()` (`_get_available_device_type`, from `current_stream()`
and events without a device) is inside it. The decode path makes no such call with the profiler off:
there is nothing of that kind to take out of production's step (the report's first item). The events
are recorded on the current device's stream, cuda:0, for the layers on cuda:1 too, so the profiler's
brackets of the second card's layers do not time that card's work. Not changed here (upstream's code).
Issue times measured with it on include it.

**Measured on the CPU** (`experiments/host_fastpath_bench.py`: the sources loaded off and on as the
tests load them, the extension a function that does nothing, Qwen3.8-Flash-Next's sizes: 4 streams of
2560, gated residual rank 320, 512 experts, top 10, 48 layers of which 36 GDN; the development machine,
Python 3.14, torch 2.14 CPU, one thread; µs a call, the best of 5):

| site | rows | off | on |
|---|---|---|---|
| a gated residual site (mix + apply) | 1-8 | 13.3-13.7 | 4.3-4.4 |
| `routing_std` | 1 | 0.78 | 0.80 |
| `routing_std` | 2-8 | 6.4-6.5 | 1.06-1.10 |
| MoE tail (routed sum + collect) | 1-8 | 2.45-2.50 | 1.12-1.17 |
| GDN record | 1-8 | 2.33-2.37 | 0.08 |
| `current_lane()` | | 0.22 | 0.08 |
| `get_bucketed` | | 1.45-1.49 | 0.79-0.83 |
| a pass: 96 sites, 48 MoE layers, 36 GDN | 1 | 1653 | 556 |
| a pass | 2, 4, 8 | 1877-1898 | 575-579 |

**On the cards, an estimate [S].** In the night's captures (counters boot, py-spy over the generator
thread) the frames the fast path no longer enters take 2.7-3.5 ms a step: `get_bucketed` under the
gated residual sites 2.1-2.4 ms, `_routing_buffers` 0.12-0.43, `staged_views` 0.23-0.40, and part of
`make_key` and `current_lane` elsewhere. On top of that, the views and slices the sites no longer
redo, which py-spy attributes to the sites' own lines: on the CPU they are a third of what a site
saves, and at the ratio `get_bucketed` shows between the two machines (about 4: 2.4 ms a step there
for 4 lookups x ~100 sites, 1.47 µs a lookup here) about 1.3 ms more. So 3-5 ms a step. With three
requests staggered, where the host is the step (about 27 ms of issue a step once the profiler's 3 ms
are taken out), that is a step of about 22-24 ms, +12-20% while the cards are not the limit; with one
request only the ~5 ms of gaps in which the cards wait for the host can come back, +10% at most. Not
measured.

**What has been checked, and where** (the development machine, no GPU, no extension):
`tests/test_host_fastpath.py`. The sources are loaded twice, into one package with the switch off and
one with it on; the same calls go through both and what reaches the extension is compared call by
call (the function and, per argument, its value or, for a tensor, the buffer it is in, named by what
it is, its offset, shape, strides and dtype), together with every torch operation that is not a view,
in order. Scenarios: the sites in the order a pass meets them in a three-block toy with the final
mixer, batch shapes changing from pass to pass, rows past the fused pair, stacks the plain path
converts, the weighted-copy handoff and a successor on another device, sublayer outputs of every dtype
and layout, the conversion and capture passes, entries made between a mix and its apply, the cache
emptied between passes, a table rebuilt, passes on lanes 1 and 2, two threads issuing at once, an
entry made again by another thread, the routing buffers at every row count up to past the kept ones,
the routed sum and its collect, the GDN record (against `staged_views`, and the call site's branch).
Thirty mutations of the fast path (a count check dropped, the lane, the batch shape or the dtype left
out of a key, an identity check dropped, a conversion let through, another shape recorded, ...) each
make a test fail, but one that cannot: the identity check in `_fast_rows_view` is redundant while the
kept view is held in the same dict. The whole suite passes with the switch off and with it on.
Not checked: anything on the cards (no CUDA tensor has gone through the fast path; equality on the
cards is the GPU check below), a real model's full forward (the toy has three blocks), mHC models
(their `HyperConnection` is untouched, the workspace cache's change applies to them), tensor
parallelism (the ranks import the library in their own processes and read the switch from the
environment they inherit), ROCm, a free-threaded Python.

**The GPU check** (on the workstation, production stopped, this branch checked out in
`~/Desktop/exllamav3`; fish, one line each). Scenario 11 of the exact check (about 10 minutes by the
estimate: two loads): the same runs in a process with `EXL3_HOST_FASTPATH=1` and one without, both
with the raw-key ring, with scenario 9's production generator (`s3`) and its fixed-window one
(`off`), expected identical everywhere against the plain process repeated:

    /home/daniele/miniconda3/envs/tabbyapi/bin/python /home/daniele/Desktop/exllamav3/experiments/exact_check.py --dry-run --scenarios 11 --device-split 23:233,25:253 --out /home/daniele/exl3-exact-check-s11
    /home/daniele/miniconda3/envs/tabbyapi/bin/python /home/daniele/Desktop/exllamav3/experiments/exact_check.py --scenarios 11 --device-split 23:233,25:253 --out /home/daniele/exl3-exact-check-s11

Then the issue time, with the night's method (`digest_issue.py`: py-spy at 200 Hz over the generator
thread, issue ms a step = (1 - share in the host wait) x window / steps), in two boots that differ in
`EXL3_HOST_FASTPATH` only, with the same frozen placement, a fixed window of 1 (as the night's P11c and
P12d) for one request at 60k and the three-request staggered case, and **without** `EXL3_SPLIT_PROF`
and `EXL3_MOE_CPU_PROF` (the counters mode sets them, and the first is 10-15% of the issue time).

Tests: `tests/test_host_fastpath.py` (no GPU, no extension; marked `nogpu`, run with
`--noconftest`), the scenario's plan and report in `tests/test_exact_check.py`.

## The overnight exact check (`experiments/exact_check.py`)

What it is: the same-computation check of the v1.5.4 benches (`stagger_bench.py`, `prefill_bench.py`,
`config_bench.py`, `pipeline_probe.py` of `flash-next-placement`), rebuilt for this branch and run
in one night before production switches to it. An orchestrator (standard library only) starts one
process per model load (`experiments/exact_check_child.py`: the model loaded by the server's own
`ModelContainer` from `--server-path`, so the expert split per card is the server's device plan,
`--moe-device-split`, with its warm start from a copy of the placement file), forces on every
compared job the tokens a reference run produced, and records at every position an integer digest
of the raw logits, the forced token's log-probability, the most likely tokens and the MTP head's
draft counters. A test run is **identical** when every compared position is; otherwise it is judged
against the same reference run repeated (the noise floor) and, where there is one, a control run.

The release it checks against is one constant, `BASE_RELEASE` (1.6.3): the stock processes must
find that wheel installed (its version, the local label aside), the server's checkout must pin
`exllamav3==BASE_RELEASE` in its `pyproject.toml` (by default `~/Desktop/exllamav3-openai-server`,
the standard checkout), and the report and the checks name it.

The seven scenarios: (1) every switch off against stock upstream (the installed wheel alone,
in two processes for a floor across processes) and the page files upstream's disk tier writes;
(2) the staggered decode with 2-5 jobs against each group alone; (3) the prefill thread beside 1-5
jobs, last chunks of 33-249 tokens; (4) the pipeline: a 33334-token prompt alone and beside a job,
`EXL3_PREFILL_PIPELINE_HEAD=second`, a cancel mid-ingestion, write-behind and the files read back
by a new generator, a 179k-token prompt with the memory of the cards; (5) the tail checkpoint on and
off, the next turn resumed from it in RAM and after a restart from the disk tier (a second process),
labelled identical or within the noise; (6) the placement weight unset against stock, the same
sweeps after the same traffic; (7) ten minutes of concurrent traffic with every production switch
on, a watchdog on every library call (faulthandler: stacks of all threads in the process's
`hang-stacks.txt`). On request (`--scenarios 8`, not in the default list): (8) the QSA
raw-key ring, the same forced runs in a process with the full raw plane and one with
`EXL3_QSA_RAW_TAIL=1` (a document of 12386 tokens past the sparse threshold with a forced answer of
512, the same beside three questions, the next turn from the pages in VRAM and, with a new generator,
from the disk tier), expected identical against the full plane's process repeated, and the memory
each load left free per card beside what the library says the ring saves (about 13 minutes by the
estimate, two loads). On request (`--scenarios 9`): (9) the production combination, below. On
request (`--scenarios 11`; 10 is the number of the branch `stream-beside-decode`'s): (11) the decode
step's host fast path, two processes that differ in `EXL3_HOST_FASTPATH` only, with scenario 9's
production generator and fixed-window one and its workload, expected identical everywhere (about 10
minutes, two loads; "The decode step's host work, kept between calls", above).

**Scenario 9, the production combination** (`--scenarios 9`, about 17 minutes by the estimate: two
loads, 78 runs, 87 comparisons). The MTP head drafting `--s9-window` tokens (3; the processes start
with the server's `--draft-num-tokens` at that value) with the batched verify, the batched head
update, the cost window and the raw-key ring, on top of what production already runs (staggered
decode, prefill thread, pipeline, tail checkpoint, the MTP head on the first card, the layer split of
`--device-split`). The MTP switches are `Generator` arguments, so one process with the full raw plane
holds a generator per set (`off`: none of the four; `v`: verify; `u`: update; `vu`: both, fixed W;
`s3`: both, cost window under the schedule `W`; `mix1`, `mix2`: both, cost window under the
schedules of `--s9-mixed`, `0,1,2,3` and `3,0,3,1`; `w1`: none, one draft token); the ring is a cache
layout made at the load, so a second process with `EXL3_QSA_RAW_TAIL=1` holds `s3` alone. The windows
come from the test hook of the draft window section: the policy left to itself chooses from step
times, and no two runs would compare. The workload of every set: a job alone (256 tokens), a batch of
three (staggered), a document of 12386 tokens ingested beside two jobs generating (prefill thread and
pipeline: two jobs of 1 + 3 rows stay within the 8 rows a chunk may overlap, so the chunks run beside
and the cost window's cap, 8 // 2 - 1 = 3, clips nothing), then the next turn resumed from the pages
in VRAM. Free runs are greedy and compared as far as their tokens go together (the position where
they part is compared for its logits only; a difference of the sampled token with identical logits
up to there is the sampling's, reported as such); forced runs take the tokens of `off`'s first free
run. A forced job takes the serial path of the verify (`forced_tokens`), so only free runs exercise
it, and the child digests a window sampled at once row by row, each position as the serial path
would have recorded it (`sample_window`, then `receive_sample` per position). The comparisons, each
group with its own table in the report and its expected verdict:
- (a) verify on against off, greedy free runs, update off (`v` against `off`) and on (`vu` against
  `u`): only how the sampled tokens come back to the host changes, and each row of the fused sampler
  is computed alone, so **identical** (logits, free tokens, drafts); an error if the run with verify
  sampled no window at once.
- (b) the cost window under the schedule `W` against the fixed W (`s3` against `vu`), free and
  forced: the same computation through the other path (the head drafts a position at a time with its
  confidence, the round stops at exactly W, the policy's times and labels go on deciding nothing):
  **identical**.
- (c) mixed schedules against the fixed W, forced: a position is computed by passes of other widths
  and rewound by other amounts, so **within the noise** against the shape meter (the same library,
  the same forced tokens, W=1 fixed against W=3 fixed: the rounding upstream itself shows when the
  verified rows change); a wrong rewind of the recurrent state or of the head's cache with windows of
  varying length would be far beyond it. An error if the schedule did not decide the steps or one
  window only was taken. The first mixed schedule repeated in the same process: **identical** (its
  floor; the document's job left out, since it joins the batch at a step that depends on time).
- (d) the batched head update on against off, fixed W, verify off (`u` against `off`, forced): the
  head's cache gets the same inputs in passes of other shapes, so its drafts and with them which pass
  verifies a position can change: **within the noise** against the shape meter, drafts reported,
  not judged; an error if no update was folded.
- (e) the ring against the full plane with production's MTP switches (`s3` in both processes, free
  and forced): **identical**, against `s3` repeated in the full plane's process.
- (f) everything on (the ring's process, `s3`) against everything off (`off`, fixed W), forced: a
  control row, **within the noise** against the shape meter (which also holds the noise between two
  processes that scenario 1 measures).
The decoders beside the document are compared only up to where the document joined the batch in
both runs (when a chunk is recorded depends on time); the turn requires a resumed prompt, the batch a
staggered call, the document's rows a chunk beside the decode with the thread and the pipeline on.
The report's table of runs gives, per run, the windows sampled at once and left to the serial path,
the folds, the scheduled steps, the windows taken, the drafts accepted and rejected, the chunks
beside and in flight. Then the memory each load left free per card (the ring's), the placement and
the libraries, as in scenario 8. The disk tier is not in this scenario (scenario 8 has the ring's
turn from the disk).

**The switches.** A switch of this branch left off in a process with every switch off must read as
off: the bookkeeping (`SWITCHES_OFF`, the child's `switches_state`) holds the MTP decode switches
(`mtp_batched_verify`, `mtp_batched_update` off), the draft window (`"fixed"`, upstream's) and the
schedule (none), next to the earlier ones; `EXL3_MTP_BATCHED_VERIFY`, `EXL3_MTP_BATCHED_UPDATE`,
`EXL3_DRAFT_WINDOW`, `EXL3_DRAFT_WINDOW_SET` and `EXL3_DRAFT_WINDOW_SCHEDULE` are removed from every
process's environment with the other switches. Since the merge of `master` of 2026-10-24 the same holds
for content-addressed multimodal pages: `mm_content_hash` (the generator's switch) and
`mm_content_digests` (the embeddings' digests, which the switch turns on for the whole process,
`tokenizer.mm_embedding.content_digests`) must read off, and `EXL3_MM_CONTENT_HASH` is removed with the
others. No scenario turns it on: the check's prompts are text, whose pages hash the same either way.
The host fast path too: `host_fastpath` (what the workspace cache's module read at import) must read
off, and `EXL3_HOST_FASTPATH` is removed with the others; scenario 11 sets it in one of its two
processes and warns when either did not read what it was given. Scenario 1 with every switch off
reports nothing about switches (`tests/test_exact_check.py`).

What is pinned: `EXL3_MOE_STREAM_FUSED_T=0 EXL3_MOE_RECON_DET=1 EXL3_MOE_FUSED_DET=1` (the MoE paths
whose sums depend on the arrival order of atomics; read on 1.6.2's sources, the same in 1.6.3) in
every process but the stress, no sweep of the expert placement during the compared runs (the library's
`run_pending_swap_sweeps` replaced by a no-op after the load: both of its callers read the module's
global), promote-on-miss off, six CPU worker slots everywhere (`EXL3_MOE_HOST_CORES=0` with the 12
worker threads, the guide's "12 as today"), the fork's switches from the environment removed. Graphs stay as production runs them (an uncaptured call issues the same
kernels, `exllamav3_ext/graph.cu`). Nothing is written outside the run directory; its disk cache is
removed at the end unless `--keep`.

How to run it on the workstation (fish; production stopped first, its graceful stop and then
`systemctl --user stop exllama-server.service`), the plan and the checks first, then the night.
What it expects there: the environment `tabbyapi` with the 1.6.3 wheel installed in place of the
1.5.4 one, this branch checked out in `~/Desktop/exllamav3` and the server's checkout for 1.6.3
(its pin, `exllamav3==1.6.3`) in `~/Desktop/exllamav3-openai-server`: the standard folders, no
worktrees; the server's is the default of `--server-path`, the fork's the one the script is in.
The 1.5.4 setup that used the same folders and environment (branch `flash-next-placement` over
the 1.5.4 wheel of `tabbyapi`) is gone since the move to 1.6.3: the 1.6.3 wheel took its place.
The dry run writes nothing and touches no card: with production still up it lists production's
process on the cards as a problem and the rest as it will be.

    /home/daniele/miniconda3/envs/tabbyapi/bin/python /home/daniele/Desktop/exllamav3/experiments/exact_check.py --dry-run --out /home/daniele/exl3-exact-check
    /home/daniele/miniconda3/envs/tabbyapi/bin/python /home/daniele/Desktop/exllamav3/experiments/exact_check.py --out /home/daniele/exl3-exact-check

How long: the dry run's estimate is about 1.9 hours for the default scenarios 1-7 (prefill 700
tokens/s, decode 60 tokens/s for one job and 60 x sqrt(n) for n, 1 minute per model load, 40-65 s
measured on 2026-10-24; ten loads), 13 minutes for scenario 8, 17 for scenario 9, 10 for scenario
11; a scenario
gets at most three times its estimate, and at least 30 minutes. The report, `REPORT.md` in the run
directory, is rewritten after every process and every ten minutes; a closed terminal does not stop
the night (`orchestrator.log` has what it printed), and `--rebuild <run directory>` makes the
report again from the records. A process whose
model does not load with the split given is run again with the split the server's error suggests
(at most `--split-retries` times), which the later processes keep; the report says so.

What the check cannot tell: a difference the digest misses (two rows of logits with the same
weighted sum of their bits: never for one changed value, about 2**-64 otherwise), a race that does
not fire in one run, whether the exact mode makes 1.6.3 repeatable (the floors of the first report
say it), and anything off the paths the scenarios take (images, a decode batch above
eight rows beside a chunk, tensor parallelism, the cost window choosing by itself, random sampling:
with a seed the window sampled at once draws other tokens than the serial path by design, so random
runs cannot compare bit for bit, and their distributions are `tests/test_window_sampling.py`'s).

The GPU window before the production combination (fish, one line each; production stopped as above):
scenario 1 with every switch off (the fork against stock upstream, which says the branch is upstream
when nothing is on), scenario 9, and the window sampling test on the card:

    /home/daniele/miniconda3/envs/tabbyapi/bin/python /home/daniele/Desktop/exllamav3/experiments/exact_check.py --dry-run --scenarios 1,9 --device-split 23:233,25:253 --out /home/daniele/exl3-exact-check-s1-s9
    /home/daniele/miniconda3/envs/tabbyapi/bin/python /home/daniele/Desktop/exllamav3/experiments/exact_check.py --scenarios 1 --device-split 23:233,25:253 --out /home/daniele/exl3-exact-check-s1
    /home/daniele/miniconda3/envs/tabbyapi/bin/python /home/daniele/Desktop/exllamav3/experiments/exact_check.py --scenarios 9 --device-split 23:233,25:253 --out /home/daniele/exl3-exact-check-s9
    PYTHONPATH=/home/daniele/Desktop/exllamav3 /home/daniele/miniconda3/envs/tabbyapi/bin/python -m pytest --noconftest /home/daniele/Desktop/exllamav3/tests/test_window_sampling.py -q

Tests (no GPU, no extension):
`tests/test_exact_check.py`, with `--noconftest` (comparisons, floors and controls, plan, estimate,
report, records, watchdog, orchestration and split retry with a fake child, the child over a fake
library and server, the Runner and the stress over the toy model of the staggered tests).
