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

### Still to come

Ported from the v1.5.4 branches in the steps after this one, each into a section of its own:

- prompt ingestion on a second thread, beside the decode (`prefill_thread`), with the short
  prompt chunks run beside the decode too, and issue lanes in the CPU expert host that two
  threads can share (1.6.2's fused issue of a decode job needs a lock around it);
- two prompt chunks in flight on the two cards (`prefill_pipeline`);
- a recurrent checkpoint that follows the answer (`recurrent_tail_checkpoint`);
- expert placement weighted on the decode (`EXL3_MOE_HIST_PREFILL_WEIGHT`);
- page-cache hygiene in the disk tier (`generator/disk_cache.py`).
