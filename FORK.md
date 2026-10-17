# Fork notes

Personal fork of ExLlamaV3 for one machine: Qwen3.8-Flash-Next (MoE, CPU expert split) on two
RTX 5060 Ti in layer split, served by `exllamav3-openai-server`. Branch `flash-next-1.6.2`, based
on upstream **v1.6.2**. The tuning of the earlier fork branches (`flash-next` and
`flash-next-placement`, on v1.5.4) is ported onto it one feature at a time. Every feature stays
behind a switch of its own, off by default: with every switch off this branch behaves as
upstream v1.6.2.

Everything on this branch is Python. The checkout is used without building anything: put it
ahead of the installed package on `PYTHONPATH` (the server's wrapper does
`export PYTHONPATH=<this checkout>`). The compiled extension is a top-level module
(`exllamav3_ext`) and keeps coming from turboderp's published **1.6.2** wheel (CUDA 13.2,
torch 2.11, Python 3.12) installed in the environment. It has to be that wheel: the 1.6.2
sources call extension functions the 1.5.4 one does not have (`exl3_moe_split_issue_fused`,
`batched_scan_replay` among them). Without the `PYTHONPATH` entry the same environment runs
stock upstream 1.6.2.

Upstream fixes not on this branch: the fix of the expert frequency profile (upstream PR #524)
is still pending upstream. It is not carried here; the server keeps upstream's profile
disabled until a release has it.

## What differs from upstream

### No CI workflows

`.github/workflows/` is removed on this branch: `build.yml` (upstream's wheel build) and
`cache_keepalive.yml` (the keepalive of that build's compiler cache, new in 1.6.x). Nothing here
is released as a wheel, and a push that carries a workflow file needs a token with the
`workflow` scope.

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
  are the pipeline's business). On the prefill lane the worker's flags are waited for on the
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
  sources (`tests/moe_harness/build.sh <this checkout> flash-next-1.6.2 <dir>` in the server's
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
  worker is what changes that (the placement weighted on the decode, a later step of this
  branch).
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

### Still to come

Ported from the v1.5.4 branches in the steps after this one, each into a section of its own:

- two prompt chunks in flight on the two cards (`prefill_pipeline`);
- a recurrent checkpoint that follows the answer (`recurrent_tail_checkpoint`);
- expert placement weighted on the decode (`EXL3_MOE_HIST_PREFILL_WEIGHT`);
- page-cache hygiene in the disk tier (`generator/disk_cache.py`).
