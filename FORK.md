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
with every selection counted as a decode-size one; promote-on-miss; the saved profile; load and
unload). Marked `nogpu`; run with `--noconftest` without the compiled extension.

### Still to come

Ported from the v1.5.4 branches in the step after this one, into a section of its own:

- page-cache hygiene in the disk tier (`generator/disk_cache.py`).
