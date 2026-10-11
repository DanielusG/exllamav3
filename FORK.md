# Fork notes

Personal fork of ExLlamaV3 for one machine: Qwen3.8-Flash-Next (MoE, CPU expert split) on two
RTX 5060 Ti in layer split, served by `exllamav3-openai-server`. Branch `flash-next`, based on
the tag the server runs in production (**v1.5.4**). Upstream is not tracked beyond that tag
until the server's hooks are ported (see the server's AGENTS.md).

As long as a change touches Python only, the checkout can be used without building anything:
put it ahead of the installed package on `PYTHONPATH`. The compiled extension is a top-level
module (`exllamav3_ext`) and keeps coming from the installed 1.5.4 wheel. Everything on this
branch is of that kind. How to build the extension on the machine, for a change that is not,
is at the end of this file.

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
way the server does; report in Markdown). Four parts, `--bench-phases` picks among them: speed
(one batch against staggered), same computation, where the time goes on the cards, where the
host's time goes. What the first runs on the machine taught:

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
- The host thread paces a staggered call, and timers around three functions do not say what
  its time is made of. With py-spy installed the bench has it sample this process's Python
  stacks during one more set of runs (one capture; a frame named after the run on the stack
  tells the runs apart) and the report splits a call into phases (draft, inputs, issue, wait
  for the cards, sampling) and the issue into the library's components, with the most sampled
  source lines. The stacks go to a second Markdown file next to the report.

`tests/test_stagger_bench.py` runs the report side of the bench on fakes.

Why the two halves are issued a module at a time: the CPU expert worker serves its job ring in
publication order (`exllamav3_ext/cpu/moe_handoff.cu`), so two halves issued back to back would
stall the first card at its first MoE layer until the second card is done.

### Prompt ingestion beside the decode (`Generator(prefill_thread = True)`, `EXL3_PREFILL_THREAD=1`)

Upstream's `iterate()` runs a prefill chunk for every job still ingesting its prompt and then one
decode step: a job that is generating gets one token per chunk while another job's prompt goes in
(2.9 s per 2048-token chunk on this machine). With the switch on, the forward passes of a chunk
run on a second thread and the generator's thread goes on decoding. Off by default; with it off
the generator behaves as upstream. Run on the machine on one case (one job generating, one
ingesting an 8289-token document; see the two runs below): same numbers as upstream within
upstream's own run-to-run noise, the generating job at half its solo speed during the ingestion,
the ingestion half as long again. Not in use in the server.

The two threads issue on the same CUDA streams, so kernels never overlap on a card. That is not
a choice: the compiled EXL3 kernels (GEMM, MGEMM, the fused MoE kernel) coordinate their thread
blocks through one lock buffer per device, indexed from zero by every launch
(`quant/exl3_devctx.cu: get_locks`, `quant/exl3_gemm_inner.cuh: barrier_acquire(&locks[...])`),
and several decode paths go through buffers baked into a module at load time. What the second
thread buys is therefore host time and CPU expert worker time: decode steps fit in the time a
chunk spends waiting for the worker and for its own host, which is most of a chunk here (the
cards are at about 30% each during prefill). It is a finer sharing of the cards than the
server's `--decode-slice-ms`, not two passes at once.

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
    decode step, and `cancel` / `clear_queue` let the chunk finish first;
  - a chunk and a decode step overlap only if they cannot meet inside a module: chunk of at
    least a page (`side_min_tokens`), decode batch of at most 8 rows x tokens
    (`side_max_rows`). Past those thresholds no path is common to the two (decode graphs and
    fused decode kernels up to 8, direct GEMM up to 144 rows). A shorter chunk, the tail of
    every prompt, is waited for like an inline one;
  - with nobody generating the round waits for its chunks, which still run on the second
    thread: the workspaces of a prefill pass exist once.
- `exllamav3/model/moe_cpu_host.py`: `split_lanes` (a range of slots per lane: on one rotation
  two threads make a stream wait for itself, `tests/test_moe_host_lanes.py` shows it), the
  descriptor and the write of its data flag issued under one lock (the worker waits for the
  job at the head of its ring), and for a thread that shares the pass's stream the copies and
  flag waits of the chunked path on a stream of their own, with the host waiting for the last
  readback: a flag wait on the shared stream would stop the other thread's kernels for as long
  as the worker takes.
- `exllamav3/modules/block_sparse_mlp_cpu.py`: no inline expert sweep while a second thread
  issues a pass.

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
  every iteration (off, pending, on, unavailable; `prefill_thread_reason` says why not) and,
  while the thread is at work, opens no decode window and keeps the long chunk: the thread's
  chunks are planned through `job._prefill_co`, so the gate it puts on `job.prefill` would
  hold nothing. Its tests read these points off the fork's sources
  (`tests/moe_harness/build.sh <this checkout> flash-next <dir>` in the server's repo).

Tests (no GPU, no extension): `tests/test_prefill_thread.py` (the real round over the toy model
of the staggered tests, chunks that last a given number of decode steps: same tokens as inline,
and what a chunk on another thread must not meet), `tests/test_moe_host_lanes.py` (the host's
issue and collect code against a model of streams, flags and the worker's ring),
`tests/test_row_table_lanes.py`, `tests/test_prefill_bench.py`.

Bench on the real model: `experiments/prefill_bench.py` (server stopped, same arguments as the
server, report in Markdown): one job generating while another ingests a document, as upstream
and with the thread, against each alone; forced-token probabilities of both jobs against the
upstream path; the longest time the generating job went without a token during the ingestion;
memory of the cards.

First run on the machine (2026-10-10, bench at ad6dfa4, 8289-token document, chunks of 2048):

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
- Two identical runs as upstream do not give identical probabilities when each ingests its
  prompt again (largest shift 0.03-0.05, the most likely token changed in 1-4 positions of
  64-600), so "identical to upstream" cannot be the test. The bench now repeats every compared
  run twice as upstream and judges the run with the thread against that floor; it counts values
  that are not finite, compares a prompt ingested on the thread with nobody decoding as well
  (which tells the thread from the overlap), and no longer resets the page table between runs
  (it drops the state checkpoints instead; the reset was not behind the floor, see below).
- Memory: about 70 MiB less free on each card with the thread (0.39 and 0.37 GiB left).
- `EXL3_LANE_HOST_WAITS=0` (`--bench-no-handoff`) keeps the second thread's worker waits on the
  pass's stream: nothing is gained, but the handoff stream is out of the picture, for telling
  whose fault a wrong number is.

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
  against 133 W. Which of the GIL and the worker's ring does has not been measured
  (`--bench-switch-ms` moves the first).
- Memory: peak 30 MiB higher on each card; 0.39 and 0.37 GiB free at the end against 0.40 and
  0.44.
- The floor is still there without the page-table reset, and upstream's sources name a cause
  (read, not confirmed on the machine). On a split model a prefill chunk of 32 rows or more
  sends the hot CPU-resident experts through `MoeCpuHost.submit_prefill`'s streamed path. Its
  fused tier calls `exl3_moe` without the slot scratch, which the kernel describes as "arrival
  order, which is not bit-reproducible run to run" (`quant/exl3_moe_kernel.cuh`), and its
  batched tier adds with a single atomic `index_add_` unless `EXL3_MOE_RECON_DET=1`
  (`modules/moe_batch_recon.py`: the default there is off, although the comment says it follows
  `EXL3_MOE_FUSED_DET`, which is on). Both as in v1.5.4. Decode does not take that path, and
  the staggered bench, whose repeated runs were identical with the same 55-token prompt, never
  dropped its prompts between runs. A run with
  `EXL3_MOE_STREAM_FUSED_T=0 EXL3_MOE_RECON_DET=1` would tell: if the repeated runs then read
  identical, the comparison with the thread becomes exact (its speed figures would not be
  production's: the fused tier is off).

Third session (2026-10-11, night, server stopped; bench and a probe built on it, production's
flags with 8 worker threads, MTP 1):

- **Same computation, exactly.** With `EXL3_MOE_STREAM_FUSED_T=0 EXL3_MOE_RECON_DET=1` two
  identical runs as upstream give identical probabilities in every position (so those two paths
  were the whole of the run-to-run noise, as read above), and the three comparisons with the
  thread come out identical too: the generating job beside an ingestion against itself alone,
  the prompt ingested on the thread alone, and beside a decode. That is the check to repeat
  after any change to this code:
  `EXL3_MOE_STREAM_FUSED_T=0 EXL3_MOE_RECON_DET=1 python experiments/prefill_bench.py --bench-phases check ...`
  (it costs no speed on this machine: 7.5 s for the prompt, 60 tok/s alone, as without).
- **The GIL is not what limits the overlap.** A switch interval of 5, 1, 0.2 and 0.05 ms gives
  the generating job the same 26-27 tok/s; py-spy with `--gil` has the generator's thread hold
  it 30% of the run and the prefill thread 7%, and nobody the rest of the time.
- **Neither are the cards**: 37% + 36% in the window.
- **The CPU expert worker is.** A decode step is a pipeline through it, layer by layer: with
  0.84 of the selections served from VRAM, 41 layers of 48 pick at least one CPU-resident
  expert among their 16 selections (2 rows x 8), and each of those costs the worker 0.46 ms
  (GEMVs over the experts' weights: 0.42 of it, compute-bound on AVX2 at about 2 GB/s per
  thread) after 0.27 ms for the card to stage the inputs. That is the whole 30 ms step; the
  host, which issues a step in 18 ms, is ahead of it. A prompt alone keeps the worker busy
  about half the time (tail jobs of 64 rows, 3-9 ms each, 55 a second). Together they want
  more of it than there is: the step goes to 54 ms and the ingestion from 7.6 to 11.7 s.
- What was tried against that, none of it worth more than 10% and all of it the same 10%
  (30.6 tok/s and 11.7 s with nothing): a ring and a thread of their own for the decode's jobs
  in the worker (branch `flash-next-native`: 33.5 and 10.7 with 8 threads each; with 2 the
  decode is slower than in one ring), 12 worker threads in one ring (32.9 and 10.8), 16 (31.2
  and 11.1), the pinned arena, a streaming threshold of 2, 4 and 16 instead of 8. Own CUDA
  streams for the lane were not tried: the cards are not the limit, and a block allocated on
  one stream and last used on another (`seq.mtp_carry_hidden` is one) is handed out again in
  the order of the first.
- What does change the picture is on the other side of the same finding: fewer layers of a
  decode step in the worker. See the next section.
- The generating job's longest wait in the window is 0.5-0.6 s: the last chunk of the prompt,
  96 tokens here, is shorter than a page and is waited for.

### Expert placement that follows the decode (`EXL3_MOE_HIST_PREFILL_WEIGHT`)

The library's dynamic placement ranks the experts of a split layer by how often they were
selected, every selection counting one. Prompt tokens outnumber generated ones by orders of
magnitude, so prompts decide the placement alone. But a selection that misses VRAM does not cost
the same in the two: a decode step waits for the worker in every layer that has one (above),
while a prompt chunk sends its hot CPU-resident experts to the card and leaves the worker a
tail.

Measured (same night, same flags; placement adapted to the bench's own traffic in each case):

| placement learnt from | share served from VRAM, generating job | its speed alone | share, prompt | its ingestion alone |
|---|---|---|---|---|
| job + prompt, equal weights (upstream) | 0.843 | 60.6 tok/s | 0.953 | 7.64 s |
| the generating job only | 0.945 | 69.7 tok/s | 0.621 | 7.87 s |
| job + prompt, prompt weight 0 | 0.929 | 66.7-68.3 tok/s | 0.627 | 7.77 s |
| job + prompt, prompt weight 0.004 | 0.928 | 67.6-67.8 tok/s | 0.836 | 7.60 s |

With 12 worker threads and weight 0: 67.8-69.9 tok/s and 7.50 s. In the overlap the two weights
give 32-33 tok/s and 11.4-12.0 s (30.6 and 11.7 with equal weights).

- `modules/block_sparse_mlp_cpu.py`: the selections of prefill-size batches (at least
  `EXL3_MOE_STREAM_MIN_ROWS` rows, 32) are counted in `_split_hist_pf` instead of `_split_hist`,
  and `run_pending_swap_sweeps` folds them into `_split_hist` with the weight, for every layer,
  before it sweeps any (`_split_hist_fold`): a reader wrapped around `_split_sweep_layer`, like
  the server's expert tier, sees the histogram the sweep uses. `split_hits()` gives both.
- `EXL3_MOE_HIST_PREFILL_WEIGHT` is read at every sweep. Default 1: upstream's ranking. 0: the
  placement follows generated tokens only. A small weight (0.004 is one prompt selection in
  250, about the ratio of the two costs on this bench) lets prompts fill the slots the decode
  leaves undecided.
- What the server shows as hit rate is computed from `_split_hist` around a sweep: with a
  weight below 1 it is the weighted share, close to the generating jobs' own.
- Not seen on real traffic: the bench has one generating job on one prompt. How far a
  placement learnt from many conversations' output gets is for production's telemetry to say.

Tests: `tests/test_placement_weight.py` (the mixin on its own: where the selections are
counted, what the sweep ranks by at weights 1, 0 and in between, the fold before any layer).

### Building the extension on the machine

Needed only for a change to `exllamav3_ext` (none on this branch). Found the hard way on
2026-10-11:

- The system's CUDA 13.4 (`/opt/cuda`) does not compile torch 2.11's headers, with any gcc:
  its front end drops a `typename` the host compiler then asks for (`ATen/core/List_inl.h`).
  The 1.5.4 wheel was built with CUDA 13.2: a conda prefix of that, used only as compiler
  (no root, 2.2 GB), does it:
  `conda create -p <prefix> --override-channels -c nvidia/label/cuda-13.2.0 -c conda-forge cuda-nvcc cuda-cudart-dev cuda-cccl cuda-driver-dev libcublas-dev libcurand-dev libcusparse-dev libcusolver-dev`
- Host compiler gcc 13 (`CC`, `CXX`, `CUDAHOSTCXX`): the environment's `libstdc++` is gcc 15's.
- `CUDA_HOME=<prefix> CPATH=<prefix>/targets/x86_64-linux/include LIBRARY_PATH=<prefix>/lib`
  `TORCH_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES= MAX_JOBS=14 python setup.py build_ext --inplace`
  in a checkout of its own. Without `TORCH_CUDA_ARCH_LIST` torch asks the cards for their
  architecture, which opens a CUDA context in the build process. 190 translation units,
  3 min 45 s; a change to a header rebuilds what includes it (100 s for the worker's three).
- The module lands in the checkout's root and wins over the wheel's when that root is ahead on
  `PYTHONPATH`. Built from this branch's sources it has the wheel's 183 symbols and gives the
  bench the same verdicts, hit rates and speeds.
- Not with the server running: one C++ unit takes 2.5 GB, and the machine has no memory to
  spare beside the model (the first attempt pushed 10 GB of the server into swap).
