# Fork notes

Personal fork of ExLlamaV3 for one machine: Qwen3.8-Flash-Next (MoE, CPU expert split) on two
RTX 5060 Ti in layer split, served by `exllamav3-openai-server`. Branch `flash-next`, based on
the tag the server runs in production (**v1.5.4**). Upstream is not tracked beyond that tag
until the server's hooks are ported (see the server's AGENTS.md).

As long as a change touches Python only, the checkout can be used without building anything:
put it ahead of the installed package on `PYTHONPATH`. The compiled extension is a top-level
module (`exllamav3_ext`) and keeps coming from the installed 1.5.4 wheel.

## What differs from upstream

### No CI workflows

`.github/workflows/build.yml` (upstream's wheel build) is removed on this branch: nothing here
is released as a wheel, and a push that carries a workflow file needs a token with the
`workflow` scope.

### Staggered decode (`Generator(stagger_decode = True)`, `EXL3_STAGGER_DECODE=1`)

With the layers split over two cards a regular decode step keeps one card idle at a time. With
two or more jobs generating, the jobs are decoded as two groups half a pass apart: one group's
second-card half and the other group's first-card half are issued together, a module of each in
turn. Off by default; with it off the generator behaves as upstream.

- `exllamav3/model/forward_cursor.py`: `ForwardCursor`, the layer-split forward pass issued in
  pieces (same modules, order and params as `Model.forward`), and the rules for using it.
- `exllamav3/generator/generator.py`:
  - `iterate_gen` is now a wrapper around `_iterate_gen_co`, the same step as a coroutine
    that stops at its forward pass. The draft and decode steps iterate `self._step_jobs()`
    instead of `self.active_jobs` (identical unless a staggered step narrows it to a group).
  - `iterate_staggered` and its helpers: the scheduler. Its docstring states what keeps a
    waiting pass safe; keep it true when touching the step functions.
  - `cancel` completes the group in flight before releasing one of its jobs; `clear_queue`
    drops it; `iterate` hands out results a cancel produced (`_stagger_backlog`), and does not
    offer prefill or a checkpoint to a job in flight.
- Not staggered: tensor-parallel loads, drafters other than MTP, models not on exactly two
  cards in a row, multi-sequence jobs, any iteration in which a job is ingesting its prompt,
  and fewer than `stagger_min_jobs` jobs generating (3; `EXL3_STAGGER_MIN_JOBS`).

Measured on the machine (2026-10-10, bench at 91621dd, placement adapted to the bench's prompts,
91% of the expert selections served from VRAM, MTP with 1 draft token):

| jobs | one batch | staggered | | cards, one batch | cards, staggered |
|---|---|---|---|---|---|
| 2 | 94.9 tok/s | 93.2 tok/s | -1.7% | 38% + 40%, 144 W | 58% + 62%, 180 W |
| 3 | 117.7 tok/s | 128.6 tok/s | +9.3% | 44% + 42%, 151 W | 62% + 65%, 194 W |
| 4 | 128.6 tok/s | 154.4 tok/s | +20.1% | 40% + 40%, 149 W | 64% + 65%, 200 W |

- Same computation: with forced tokens the probabilities of a staggered run are bit-identical
  to those of each group alone, in every position of every job (9 job comparisons, 400
  positions each). The check is not blind: the same figures tell a job in a batch from the
  job alone (regular decode: the most likely token changes in 2-7 positions out of 400).
- A staggered call is draft step (2 ms), issue (about 15 ms of host time for a pass, whatever
  the size of its group) and sampling (2-2.7 ms per job); the cards work during the issue and
  idle in the other two. With groups of one job the host is the limit of the issue too: the
  cards finish each half as the host finishes issuing it (a half pass of one job is about 12
  ms of stream time). Hence the threshold: two groups of one job cost two calls for what one
  batch of two gives in one step of the same length. With groups of two the cards' half pass
  (about 17 ms) and the host's issue are level, and what the cards still idle, 6-7.5 ms of a
  23 ms call with 4 jobs, is the draft and sampling phases.
- The CPU expert worker is out of the way only with a placement that fits the traffic: the
  cards wait for it 2-3 ms per call each at 91% of selections served from VRAM. On the
  placement tuned for production traffic the bench's prompts got 74%, and every multi-job
  figure, staggered or not, stopped at about 100 rows per second (4 jobs in one batch: 86
  tok/s against 129).

Tests: `tests/test_stagger_decode.py` (no GPU, no extension: toy model on two pretend cards,
stub jobs; tokens of a staggered run must equal those of a regular one).

Bench on the real model: `experiments/stagger_bench.py` (server stopped; loads the model the
way the server does; report in Markdown). Three parts: speed (one batch against staggered),
same computation, where the time goes. What the first run on the machine taught:

- Two identical greedy runs of this model do not give the same tokens, so token equality
  proves nothing. The check forces on every job the tokens it produced alone and compares the
  probabilities position by position against the job's group alone; the group alone run twice
  is the noise floor.
- The placement loaded from the server's file follows production traffic, not the bench's
  prompts, and a poor placement loads the CPU expert worker. The bench first adapts it with
  the library's own sweeps, called past the wrapper of the server's expert tier (the tier
  saves the placement file after every sweep it sees), and reports the share of selections
  served from VRAM for every run.
- Host timers and `nvidia-smi` cannot tell a card that waits for the worker from an idle one:
  the profiled runs bracket each card's part of the pass and the worker handoffs with CUDA
  events (stream time, no host synchronization).

`tests/test_stagger_bench.py` runs the report side of the bench on fakes.

Why the two halves are issued a module at a time: the CPU expert worker serves its job ring in
publication order (`exllamav3_ext/cpu/moe_handoff.cu`), so two halves issued back to back would
stall the first card at its first MoE layer until the second card is done.
