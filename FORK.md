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

### Still to come

Ported from the v1.5.4 branches in the steps after this one, each into a section of its own:

- prompt ingestion on a second thread, beside the decode (`prefill_thread`), with the short
  prompt chunks run beside the decode too, and issue lanes in the CPU expert host that two
  threads can share (1.6.2's fused issue of a decode job needs a lock around it);
- two prompt chunks in flight on the two cards (`prefill_pipeline`);
- a recurrent checkpoint that follows the answer (`recurrent_tail_checkpoint`);
- expert placement weighted on the decode (`EXL3_MOE_HIST_PREFILL_WEIGHT`);
- page-cache hygiene in the disk tier (`generator/disk_cache.py`).
