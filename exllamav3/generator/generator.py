from __future__ import annotations
import logging
import os
import torch
from ..model.model import Model
from ..model.forward_cursor import ForwardCursor, DirectForward, cuda_stage_bounds
from ..util.lane import Lane, cuda_devices, module_devices
from ..cache.cache import Cache
from ..cache.recurrent import RecurrentCache, PairedState, host_pool, mp_host_pool_release
from ..tokenizer.tokenizer import Tokenizer
from ..constants import PAGE_SIZE
from ..util import cuda_sync_active

logger = logging.getLogger(__name__)
from ..util.memory import malloc_trim
from .pagetable import PageTable, is_content_hash
from .cpu_cache import CPUPageCache
from .disk_cache import DiskPageCache
from .draft_confidence import DraftConfidenceCalibrator
from .draft_window import DraftWindowPolicy, parse_windows, parse_schedule
from .job import Job
from .filter import Filter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from .sampler import Sampler
from .visualizer import CacheVisualizer
import time
import threading
from ..tokenizer import MMEmbedding
from ..util import profile_opt

class _Flight:
    """A group of jobs whose decode step is under way (Generator.iterate_staggered): the step
    coroutine stopped at its forward pass, the pass itself, the lane of pinned staging buffers
    the step staged its inputs through, and the jobs."""

    __slots__ = ("step", "fwd", "lane", "jobs")

    def __init__(self, step, fwd, lane: int, jobs: list):
        self.step = step
        self.fwd = fwd
        self.lane = lane
        self.jobs = jobs


# Shortest prompt chunk that may run on the prefill thread beside a decode step: one token past
# the CPU expert worker's decode-size batches (Generator.__init__ says what stands behind it)
SIDE_MIN_TOKENS = 33


class _SidePass:
    """A chunk of a job's prompt being ingested on the prefill thread (Generator._prefill_round):
    the job, its prefill coroutine stopped at the chunk's forward passes, the future of those
    passes, whether decode steps may be taken beside them and how many have been so far.
    In flight on two cards (Generator._pipe_round) also the passes themselves (run; the future
    is None until they are handed to the threads), whether the chunk has been told to go no
    further (dropped), whether it started with another chunk under way (piped) and before the
    chunk of its own job that comes before it was recorded (ahead)."""

    __slots__ = ("job", "step", "future", "beside", "steps", "run", "dropped", "piped", "ahead")

    def __init__(self, job, step, future, beside, run = None, ahead = False):
        self.job = job
        self.step = step
        self.future = future
        self.beside = beside
        self.steps = 0
        self.run = run
        self.dropped = False
        self.piped = False
        self.ahead = ahead


class Generator:

    # The fork's state that iterate() reads, as __init__ leaves it with every switch off, for a
    # Generator made without __init__: upstream's tests/generator/test_failure_containment.py
    # makes one with __new__ and gives it only upstream's attributes to drive iterate() over stub
    # jobs. With these it takes upstream's paths (no group in flight or results kept back, prompt
    # ingestion inline, no chunk under way). __init__ sets every one of them on the instance, so
    # a generator made the regular way never reads these. The empty queues are tuples, so nothing
    # can be added to one shared by every such generator
    stagger_decode = False
    prefill_thread = False
    mtp_batched_verify = False
    mtp_batched_update = False
    draft_window = "fixed"
    draft_window_set = None
    draft_window_schedule = None
    draft_window_policy = None
    draft_window_stats = None
    _dw_round = None
    _dw_dirty = False
    _stagger_backlog = ()
    _stagger_run = None
    _flight = None
    _side = None
    _pipe = ()

    def __init__(
        self,
        model: Model,
        cache: Cache,
        tokenizer: Tokenizer,
        max_batch_size: int = 256,
        max_chunk_size: int = 2048,
        max_q_size: int = 8,
        draft_model: Model | None = None,
        draft_cache: Cache | None = None,
        num_draft_tokens: int | None = None,
        show_visualizer: bool = False,
        enable_defrag: bool = True,
        cpu_cache_size: int = 0,
        disk_cache_dir: str | None = None,
        disk_cache_size: int = 50 * 1024**3,
        disk_cache_flush_on_idle: bool = True,
        disk_cache_checkpoint_interval: int = 4096,
        disk_cache_write_interval: float = 2.0,
        recurrent_cache_size: int = 4 * 1024**3,
        recurrent_checkpoint_interval: int = None,
        recurrent_checkpoint_interval_pp: int = 32768,
        recurrent_tail_checkpoint: int | bool = False,
        ngram_match_min: int = 0,
        dynamic_draft_tokens: bool = False,
        draft_confidence: float = 0.4,
        record_draft_stats: bool = False,
        ngram_corpus: str | None = None,
        hybrid_draft_tokens: int = 0,
        stagger_decode: bool = False,
        prefill_thread: bool = False,
        prefill_pipeline: bool = False,
        mtp_batched_verify: bool = False,
        mtp_batched_update: bool = False,
        draft_window: str = "fixed",
        draft_window_set: list[int] | str | None = None,
        **kwargs
    ):
        """
        Initialize generator

        :param model:
            The model (loaded)

        :param cache:
            Paged cache

        :param tokenizer:
            Tokenizer

        :param max_batch_size:
            The maximum number of sequences to process in parallel. The generator will also limit this
            dynamically considering the available cache space.

        :param max_chunk_size:
            Maximum number of tokens to process in parallel during prefill (prompt ingestion). Should not
            exceed the model's max_input_len but can be lowered to trade off prompt speed for a shorter
            interruption to ongoing jobs when a new job is started.

        :param max_q_size:
            Maximum number of tokens to evaluate per sequence during generation. Leave this at the default
            (8) unless there's a good reason to increase it.

        :param draft_model:
            Draft model. Enables speculative decoding with draft, and must be specified along with
            draft_cache. Note that speculative decoding with many parallel jobs is likely not advantageous.

        :param draft_cache:
            Cache allocated for draft model. Must be same size as main cache.

        :param num_draft_tokens:
            Number of future tokens to draft. Default is 4 unless the draft model has a preference (e.g.
            from DFlash block size)

        :param ngram_match_min:
            Minimum number of tokens to match for n-gram draft (0 = disabled).

        :param ngram_corpus:
            Optional frozen SAM file shared by n-gram jobs; requires ngram_match_min > 0.

        :param hybrid_draft_tokens:
            Long suffix-match drafting on top of whichever drafting mode is active (or on its own). Every
            round, each job's suffix automaton is consulted; when the longest match of the current suffix
            against the job's earlier context is at least this many tokens, the draft becomes the same number
            of tokens that followed that earlier occurrence, unless they disagree with the regular drafter's
            own guess. The intent is to catch verbatim repetition of context (reciting a drafted reply, editing
            source) and gamble a long verification pass on advancing many tokens at prefill speed. A recurrent
            model's cache needs max_history of at least this many tokens.

        :param dynamic_draft_tokens:
            Adapt the per-round draft length to the workload. The draft is cut using a drafter-provided
            confidence score, calibrated online against observed acceptance rates (see draft_confidence). A
            DFlash drafter still runs at its fixed diffusion block size and only the verified window shrinks;
            AR and MTP drafters stop the drafting loop itself early, saving one drafter forward per pruned
            position. The acceptance behavior of the surviving positions is unchanged. Has no effect on n-gram
            drafting, whose draft length already adapts through the match length.

        :param draft_confidence:
            Used with dynamic_draft_tokens and a draft model: target acceptance probability for drafted
            positions, evaluated against an online mapping from drafter confidence scores to observed
            acceptance rates. Scale-free and portable across model pairs. For DFlash (block produced at fixed
            cost) the block is cut at the first position whose estimated conditional acceptance drops below
            the target. For AR and MTP drafters (one forward per position, and a position only pays off if
            everything before it is accepted too) drafting stops when the running product of conditional
            estimates - the probability that the next position actually contributes a token - falls below the
            target. Lower values keep longer drafts; higher values truncate more aggressively. Ignored for
            n-gram drafting.

        :param record_draft_stats:
            Append (position, window, accepted) per verification round to job.draft_stats, for analysis.

        :param stagger_decode:
            For a model whose layers are split over two GPUs: with stagger_min_jobs or more jobs generating
            (3; EXL3_STAGGER_MIN_JOBS), decode them as two groups half a pass apart, so one group runs on the
            second card while the other runs on the first, instead of one batch that keeps one card idle at a
            time. Each iterate() then returns the tokens of one group. Can be switched at any time (attribute
            of the same name); needs the MTP drafter or no drafter, and no hybrid draft. EXL3_STAGGER_DECODE=1
            in the environment turns it on for callers that do not pass the argument. See iterate_staggered().

        :param prefill_thread:
            Run the forward passes of prompt ingestion on a second thread, so that the jobs that are generating
            keep getting decode steps while another job's prompt is ingested, instead of one step per prefill
            chunk. The two threads issue on the same CUDA streams: the decode steps fit in the time the prefill
            spends waiting for the CPU expert worker and for its own host. Can be switched at any time (attribute
            of the same name); not available with a tensor-parallel load, a draft model other than MTP or
            promote-on-miss (prefill_thread_reason says why). EXL3_PREFILL_THREAD=1 in the environment turns it on
            for callers that do not pass the argument. See _prefill_round().

        :param prefill_pipeline:
            With the layers of the model on two cards, keep two prompt chunks under way: one on the second card
            while the next is on the first, each half issued by a thread of its own (it turns prefill_thread on).
            A prompt of several chunks then goes in faster; a prompt of one chunk gains nothing. A chunk is never
            started past a position where a recurrent checkpoint is due before the chunk that ends there is
            recorded, so every checkpoint interval costs the overlap of one chunk. Needs the MTP head and one
            slot more per lane in the CPU expert worker (EXL3_MOE_CPU_SLOTS=6 leaves each lane two); not
            available where the prefill thread is not. Can be switched at any time (attribute of the same name).
            EXL3_PREFILL_PIPELINE=1 in the environment turns it on for callers that do not pass the argument. See
            _pipe_round().

        :param mtp_batched_verify:
            With the MTP head drafting: sample every position of each job's speculative window in one
            sampler call, compare the samples with the drafts on the device, and read the tokens and the
            accepted lengths of every such job back at once, instead of a sampler call and a readback per
            position and job. Greedy sampling gives the same tokens; with random sampling a job draws one
            random number per window instead of one per position, so the same seed samples other tokens
            (as reproducibly). A job whose window cannot be sampled ahead (Job.window_sampling_why: token
            healing or its first token, forced tokens, probabilities asked for, banned strings, stop tokens
            still blocked by min_new_tokens, a stateful sampler step, a filter without native rollback or
            with a trigger token) takes the serial path that step. Can be switched at any time (attribute
            of the same name). EXL3_MTP_BATCHED_VERIFY=1 in the environment turns it on for callers that do
            not pass the argument. See _verify_windows().

        :param mtp_batched_update:
            With the MTP head drafting: after the verification, bring the head's cache up to the accepted
            tokens and draft the next token in one batched pass of the head over every job that goes on
            generating (_mtp_fold), instead of one pass of the head per job that accepted a draft and a
            draft pass of the head at the start of the next iteration. Positions past a job's accepted
            tokens are written too, with nothing that will be read: every one of them is written again
            before any pass reads it. The next draft step then reads the drafts back without running the
            head for the first draft token; it runs the head as before when a job of the step has none
            (a job just past its prompt, a job that ended or was requeued, a banned-string rewind, the
            confidence-calibrated draft). Speculative decoding stays exact; the head's cache gets the same
            inputs in passes of other shapes, so drafts, and with them which positions each target pass
            computes, can differ. Can be switched at any time (attribute of the same name).
            EXL3_MTP_BATCHED_UPDATE=1 in the environment turns it on for callers that do not pass the
            argument.

        :param draft_window:
            With the MTP head drafting: "fixed" (default) verifies num_draft_tokens drafts every step;
            "cost" chooses each step's window W (one for the batch, or for each group of a staggered
            decode) from 0 to num_draft_tokens to maximize the expected tokens per second,
            (n + sum over the n jobs of the running products of the drafts' estimated acceptance up
            to W) / T(n, W): the estimates from the head's winning logit, calibrated online against
            the verified drafts (DraftConfidenceCalibrator), T the measured wall time of such steps
            (median of the last ones, for every window timed at first and from time to time after).
            W = 0 verifies no draft that step (the head's first draft is still made). Speculative
            decoding keeps the target's tokens; the window decides what each pass verifies (on the
            cards passes of other widths can move a position's logits in the last bits, as between
            two fixed windows). Takes the place of dynamic_draft_tokens for the MTP head. Can be switched at any time (attribute of the same
            name). EXL3_DRAFT_WINDOW=cost in the environment turns it on for callers that do not pass
            the argument. See _dw_begin() and generator/draft_window.py.

        :param draft_window_set:
            The windows draft_window = "cost" may choose from, a list or "0,1,3" (EXL3_DRAFT_WINDOW_SET);
            None: every window from 0 to num_draft_tokens. Each window is a shape of the verification
            pass and of the head's update, with graphs of its own on the cards.

            A test hook, read from the environment only: EXL3_DRAFT_WINDOW_SCHEDULE="3" or "0,1,2,3"
            makes draft_window = "cost" take the next window of that schedule at every decode step
            (cycled, clipped to what the step allows) instead of the policy's choice, so that runs of
            the cost path can be compared bit for bit (the exact check). Attribute
            draft_window_schedule (a list or such a string, None for none); see generator/
            draft_window.py.

        :param show_visualizer:
            Open window to render visualization of cache (for debug/demonstration purposes)

        :param enable_defrag:
            Defragment cache periodically

        :param cpu_cache_size:
            Size in bytes of a second-tier page cache in pinned system memory, 0 (default) to disable. Complete
            K/V pages evicted from the GPU cache are stored there and restored on prompt-cache hits instead of
            being recomputed by prefill. In tensor-parallel mode each rank pins its own shard of the tier and
            the budget counts whole pages across all of them

        :param disk_cache_dir:
            Directory for a third, persistent tier behind the CPU page cache (requires cpu_cache_size > 0),
            None (default) to disable. Pages entering the CPU tier and recurrent checkpoints are written
            through to a per-model subdirectory (keyed by a fingerprint of the model, cache configuration and
            kernel sources) and read back, in this or a later process, when a prompt's pages are in no other
            tier. Page images and checkpoints are stored in a layout independent of the tensor-parallel
            split, so a directory is shared between TP loads with different splits and non-TP loads of the
            same model and cache configuration

        :param disk_cache_size:
            Budget in bytes for the whole disk_cache_dir, shared by every model and cache configuration
            stored under it (the least recently used of the others are removed whole when a load finds the
            root over budget). Default is 50 GB

        :param disk_cache_flush_on_idle:
            Also write every complete page still in VRAM (and every checkpoint) to the disk tier whenever the
            generator goes idle, so a context never has to be evicted first to be persisted. Default True;
            flush_cache_to_disk() does the same on demand

        :param disk_cache_checkpoint_interval:
            Minimum spacing, in tokens, between the recurrent checkpoints the disk tier keeps along any chain
            of cached pages, counted from the start of the chain; closer ones (and checkpoints whose page is
            gone) are pruned when the generator goes idle, so a resumed prompt replays at most this many
            tokens. Checkpoints take far more space than pages on hybrid models. Default 4096; 0 keeps all

        :param disk_cache_write_interval:
            While jobs run, each this many seconds, queue the complete pages of the active sequences for the disk
            tier (through the CPU tier). Without this, a page goes to disk only when it leaves VRAM or when the
            generator becomes idle, so a process that stops during a long prefill loses all of it. With this, it
            loses only the pages after the last checkpoint on disk. The last two pages before the position of
            each sequence are not queued: speculative decoding can still rewind them, and the MTP draft cache
            writes the last slot of a page one step later. 0 disables. Default 2.0

        :param recurrent_cache_size:
            Size of recurrent cache, in bytes. Recurrent cache resides in system RAM. Default is 4 GB.
            Ignored if model doesn't use recurrent states

        :param recurrent_checkpoint_interval:
            Minimum number of tokens between recurrent checkpoints in model output and the tail end of
            the prompt. Must be a multiple of the page size. Model architecture determines default

        :param recurrent_checkpoint_interval_pp]:
            Minimum number of tokens between recurrent checkpoints during prompt ingestion. Must be a
            multiple of the page size. Default is 32768 tokens

        :param recurrent_tail_checkpoint:
            Keep one more checkpoint behind a job that is generating, moved on every this many tokens
            (True: every page; a multiple of the page size). A job that ends then leaves a checkpoint
            at the last page boundary of its answer, and the next turn of the conversation resumes
            there instead of ingesting the answer again from the prompt's last page (no checkpoint
            is taken at the end of a generation otherwise: the next one falls a whole
            recurrent_checkpoint_interval after the prompt). The checkpoint replaces the one it
            left a page before, so a job holds one more at most and the cache never makes room for
            it; a job that itself resumed from such a checkpoint gives up its prompt's last-page one
            for it, since the client has shown it sends answers back as they were. Each move is a
            stash of the state (a copy to system RAM), and a draft is cut at the boundary:
            tail_stats has the count and the time. With the disk tier, a replaced checkpoint's write
            is withdrawn and a copy already written removed (a write under way is left to the tier's
            pruning); the one a job leaves is written at idle, as the last of its chain. Can be
            switched at any time (tail_checkpoint_interval, 0 = off). EXL3_TAIL_CHECKPOINT=1 (or a
            number of tokens) in the environment turns it on for callers that do not pass the
            argument. Default False

        :param kwargs:
        """

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        cfg = self.model.config
        self.padded_vocab_size = ((cfg.vocab_size + 31) // 32) * 32

        # Paging
        self.pagetable = PageTable(self, cache)
        # A cache has exactly one owning generator. Record the takeover and refuse to iterate a superseded generator
        cache.owner_serial = getattr(cache, "owner_serial", 0) + 1
        self.cache_owner_serial = cache.owner_serial
        self.max_total_tokens = PAGE_SIZE * self.pagetable.max_pages
        self._cache_stats_memo = (None, None)

        # Draft model
        self.draft_model = draft_model
        self.draft_cache = draft_cache
        if draft_model:
            assert not ngram_match_min, \
                "Cannot use both draft model and n-gram draft."
            assert draft_cache is not None, \
                "Must supply cache for draft model"
            assert draft_cache.max_num_tokens == cache.max_num_tokens, \
                "Cache and draft cache must be same size"
            if draft_model.caps.get("recurrent_states"):
                # An autoregressive drafter with recurrent layers drafts through a token-by-token
                # speculative pass on its own state pool (iterate_draftmodel_gen); a DFlash-family
                # drafter's sliding-window rings take the projected rows directly. MTP heads with
                # recurrent layers have neither path yet
                assert not draft_model.caps.get("mtp_draft"), "Recurrent MTP draft models are not supported."
            if num_draft_tokens:
                self.num_draft_tokens = num_draft_tokens
            else:
                self.num_draft_tokens = draft_model.caps.get("default_draft_size", 4)
            depths = draft_model.caps.get("mtp_depths")
            if depths is not None and self.num_draft_tokens > depths:
                print(f" !! Warning: the MTP head has {depths} depth-specialized layers; draft positions past "
                      f"{depths} reuse the last one, with decreasing acceptance (num_draft_tokens = "
                      f"{self.num_draft_tokens})")
        elif ngram_match_min:
            self.num_draft_tokens = num_draft_tokens if num_draft_tokens is not None else 4
        else:
            self.num_draft_tokens = 0

        self.ngram_corpus = None
        if ngram_corpus is not None:
            if ngram_match_min <= 0:
                raise ValueError("ngram_corpus requires ngram_match_min > 0")
            from .ngram import NgramCorpus
            self.ngram_corpus = NgramCorpus(ngram_corpus, tokenizer)
        self.ngram_match_min = ngram_match_min
        # The widest verification pass any round can take: the regular drafter's window or a hybrid draft
        self.hybrid_draft_tokens = max(0, hybrid_draft_tokens or 0)
        self.max_draft_tokens = max(self.num_draft_tokens, self.hybrid_draft_tokens)
        # A QSA raw-key ring (Cache(qsa_raw_tail), cache/qsa.py) keeps the block the next pass starts
        # in only for verification passes it was sized for
        if getattr(cache, "qsa_raw_tail", 0) or getattr(draft_cache, "qsa_raw_tail", 0):
            from ..cache.qsa import check_raw_tail
            check_raw_tail([cache, draft_cache], self.max_draft_tokens + 1, model)
        self.dynamic_draft = dynamic_draft_tokens and self.num_draft_tokens > 0
        self.record_draft_stats = record_draft_stats
        max_q_size = max(self.max_draft_tokens + 1, max_q_size)

        # Chunking/partitioning
        self.max_batch_size = max_batch_size
        self.max_chunk_size = max_chunk_size

        # Job queues
        self.job_serial = 0
        self.pending_jobs = []
        self.active_jobs = []

        # Filter threads
        self.filter_pool = ThreadPoolExecutor(max_workers = 16)
        self.filter_queue = []

        # Rounds that verified a hybrid (long suffix-match) draft
        self.hybrid_rounds = 0

        # Pinned staging buffer for batched token readback in iterate_gen
        self.sample_pinned = None
        self.staging_buffers = {}

        # Readbacks of drafted and sampled tokens by the decode steps (each a host wait for the GPU),
        # counted whatever the switches: decode steps that sampled, readbacks they and their draft
        # steps made. Not counted: a job's first token (its NaN check and timing are a wait each)
        self.readback_stats = {"steps": 0, "readbacks": 0}

        # Speculative windows sampled in one call each and read back together (_verify_windows):
        # windows sampled so, windows left to the serial path while the switch is on (and why)
        self.mtp_batched_verify = mtp_batched_verify or \
            os.environ.get("EXL3_MTP_BATCHED_VERIFY", "0") not in ("", "0")
        self.mtp_verify_stats = {"windows": 0, "serial": 0, "readbacks": 0, "fallbacks": {}}

        # The head's cache brought up to the accepted tokens, and the next draft made, in one pass
        # (_mtp_fold): head passes it issued, rows it brought up to date in them (each job's), rows
        # that also drafted their next token, draft steps that took those drafts without running the
        # head, draft steps that ran it with some job's draft left unused
        self.mtp_batched_update = mtp_batched_update or \
            os.environ.get("EXL3_MTP_BATCHED_UPDATE", "0") not in ("", "0")
        self.mtp_update_stats = {"passes": 0, "rows": 0, "folds": 0, "fold_drafts": 0, "fold_misses": 0}

        # Staggered decode (iterate_staggered)
        self._stagger_init(stagger_decode)

        # Prompt ingestion on a second thread (_prefill_round). _side is the chunk under way;
        # the thread and its lane (util/lane.py) are made at the first use. What may overlap:
        # a chunk of at least side_min_tokens tokens and a decode step of at most side_max_rows
        # rows x tokens. Those two never take the same path through a module, so no buffer
        # baked into a module at load time is used by both: the decode graphs, the fused decode
        # kernels and the single-row buffers serve up to 8 rows x tokens, and a chunk above the
        # CPU expert worker's decode-size batches (EXL3_MOE_STREAM_MIN_ROWS, 32: _side_plan
        # raises the floor to that) goes through the prefill tiers of every module, whether
        # the direct GEMM (up to 144 rows) or the reconstruct path; the workspaces of
        # g_tensor_cache are per lane besides. The floor was a page until the exact form of the
        # bench's check existed (FORK.md, measured on the v1.5.4 base): chunks of 33, 48, 64, 65,
        # 96, 100, 144, 145, 199 and 249 tokens beside a decode came out identical to upstream,
        # and the last chunk of a prompt, or a short prompt, no longer stopped the jobs that
        # were generating (their longest wait in the bench: 0.6 s before, 0.09 s after).
        # EXL3_SIDE_MIN_TOKENS overrides (256 is the old behaviour)
        self.prefill_thread = prefill_thread or os.environ.get("EXL3_PREFILL_THREAD", "0") not in ("", "0")
        self.side_min_tokens = max(SIDE_MIN_TOKENS, int(os.environ.get("EXL3_SIDE_MIN_TOKENS", "0") or 0))
        self.side_max_rows = 8
        self.side_stats = {
            "chunks": 0, "beside": 0, "decode_steps": 0, "errors": 0,
            "t_chunks": 0.0, "t_waited": 0.0, "piped": 0, "ahead": 0,
        }
        self._side = None
        self._side_ok = None
        self._side_why = None
        self._side_lane = None
        self._side_pool = None
        self._side_turn = 0
        # Chunks ingested beside a decode since the last report: [chunks, decode steps beside
        # them, seconds of chunks]; reported in one line once no job is ingesting any more
        self._side_stretch = None

        # Prompt chunks in flight on two cards (_pipe_round): each chunk issued in two halves,
        # the first by the prefill thread and the second by a thread of its own, so that a card
        # has one chunk at a time and each of the two lanes its kernels on one card. _pipe is
        # the chunks under way, oldest first, pipe_depth how many there may be (two: one per
        # card; the first half is the longer one on this machine, so a third would only wait).
        # The second thread, its lane and the module at which a chunk changes card are made
        # at the first use
        self.prefill_pipeline = prefill_pipeline or os.environ.get("EXL3_PREFILL_PIPELINE", "0") not in ("", "0")
        if self.prefill_pipeline:
            self.prefill_thread = True
        self.pipe_depth = max(1, int(os.environ.get("EXL3_PREFILL_PIPELINE_DEPTH", "2") or 2))
        self._pipe = []
        self._pipe_ok = None
        self._pipe_why = None
        self._pipe_lanes = None
        self._pipe_pool = None
        self._pipe_stop = None
        self._pipe_head_first = False

        # Buffers. Pinned: the draft input ids upload non-blocking from here every round (and
        # the DFlash2 selector reads its anchor from the same view), the drafted ids come back
        # into draft_ids_pinned
        if draft_model or ngram_match_min or self.hybrid_draft_tokens:
            self.draft_input_ids_pinned = torch.empty(
                (max_batch_size, 1),
                dtype = torch.long,
                pin_memory = True
            )
            self.draft_ids_pinned = torch.empty(
                (max_batch_size, self.max_draft_tokens),
                dtype = torch.long,
                pin_memory = True
            )

        # CPU page cache tier, and the disk tier behind it
        self.cpu_page_cache = None
        self.disk_page_cache = None
        self.disk_cache_flush_on_idle = disk_cache_flush_on_idle
        self.disk_cache_write_interval = disk_cache_write_interval
        self._write_behind_time = 0.0
        if cpu_cache_size:
            tier_caches = [cache] + ([draft_cache] if draft_cache is not None else [])
            self.cpu_page_cache = CPUPageCache(tier_caches, cpu_cache_size)
            self.cpu_page_cache.attach(self.pagetable)
            self.pagetable.cpu_tier = self.cpu_page_cache
            if disk_cache_dir:
                self.disk_page_cache = DiskPageCache(disk_cache_dir, tier_caches, self.cpu_page_cache, disk_cache_size,
                                                     min_state_distance = disk_cache_checkpoint_interval)
                self.cpu_page_cache.attach_disk(self.disk_page_cache)
        elif disk_cache_dir:
            raise ValueError("disk_cache_dir needs a CPU page cache tier to write through from (cpu_cache_size > 0)")

        # Visualizer
        if show_visualizer:
            self.visualizer = CacheVisualizer(self.pagetable.max_pages)
        else:
            self.visualizer = None

        # Defrag
        self.enable_defrag = enable_defrag

        # Recurrent cache. A job's recurrent state is a PairedState over the target and draft sides;
        # both sides checkpoint together into this one pool (see PairedState)
        self.recurrent_cache_size = recurrent_cache_size
        self.target_recurrent = bool(self.model.caps.get("recurrent_states"))
        self.draft_recurrent = bool(draft_model is not None and draft_model.caps.get("recurrent_states"))
        if self.target_recurrent:
            if self.max_draft_tokens > cache.max_history:
                raise ValueError(
                    f"Speculative passes of up to {self.max_draft_tokens} draft tokens need a cache with "
                    f"max_history >= {self.max_draft_tokens} on a recurrent model (this cache has "
                    f"max_history = {cache.max_history})"
                )
        if self.draft_recurrent and not draft_model.caps.get("dflash_draft") and self.num_draft_tokens > draft_cache.max_history:
            raise ValueError(
                f"A recurrent draft model drafting {self.num_draft_tokens} tokens needs a draft cache with "
                f"max_history >= {self.num_draft_tokens} (this cache has max_history = {draft_cache.max_history})"
            )
        if self.target_recurrent or self.draft_recurrent:
            self.recurrent_cache = RecurrentCache(self.model, recurrent_cache_size, draft_model)
            self.recurrent_cache.pagetable = self.pagetable
            self.recurrent_cache.disk_tier = self.disk_page_cache
            if self.disk_page_cache is not None:
                # Which model and cache stash under each side of a PairedState: a tensor-parallel side's
                # checkpoints are written and read by that model's ranks
                self.disk_page_cache.recurrent_sides = {
                    side: (m, c) for side, m, c, on in (
                        ("target", self.model, cache, self.target_recurrent),
                        ("draft", draft_model, draft_cache, self.draft_recurrent),
                    ) if on
                }
            # The new page table owns every page, so every state slot is ours too
            for c, recurrent in ((cache, self.target_recurrent), (draft_cache, self.draft_recurrent)):
                if recurrent:
                    c.reset_states()
                    # Limit batch size if cache has recurrent states
                    self.max_batch_size = min(self.max_batch_size, c.num_slots)
        else:
            self.recurrent_cache = None
        if recurrent_checkpoint_interval is None:
            # The target sets the checkpoint schedule; a recurrent draft under a non-recurrent target
            # lends it its own
            schedule_model = model if self.target_recurrent or not self.draft_recurrent else draft_model
            recurrent_checkpoint_interval = schedule_model.caps.get("default_recurrent_checkpoint_interval", 2048)

        assert recurrent_checkpoint_interval % PAGE_SIZE == 0 and recurrent_checkpoint_interval_pp % PAGE_SIZE == 0, \
            "checkpoint interval must be a multiple of the page size (256)"
        def ceil_span(a, b):
            return (a + b - 1) // b * b
        recurrent_checkpoint_interval = recurrent_checkpoint_interval
        self.recurrent_checkpoint_interval = recurrent_checkpoint_interval
        self.recurrent_checkpoint_interval_pp = ceil_span(recurrent_checkpoint_interval_pp, self.max_chunk_size)

        # The checkpoint that follows a generating job (Job.is_tail_boundary, Job.maybe_stash_recurrent):
        # tokens between two of its positions, 0 = none. Read at every step, so it can be switched while
        # jobs run
        if not recurrent_tail_checkpoint:
            env = os.environ.get("EXL3_TAIL_CHECKPOINT", "0").strip() or "0"
            recurrent_tail_checkpoint = True if env == "1" else int(env)
        if recurrent_tail_checkpoint is True:
            recurrent_tail_checkpoint = PAGE_SIZE
        assert recurrent_tail_checkpoint % PAGE_SIZE == 0, \
            "tail checkpoint interval must be a multiple of the page size (256)"
        self.tail_checkpoint_interval = int(recurrent_tail_checkpoint) if self.recurrent_cache is not None else 0
        # stashes: tail checkpoints taken; replaced: checkpoints they (or a regular checkpoint after
        # them) took out of RAM; t_stash: seconds spent taking them (the copy of the state, and the
        # wait for the step before it); disk_*: what the disk tier held of the ones taken out, a
        # write candidate or a queued write withdrawn, a written copy removed, a write under way left
        self.tail_stats = {"stashes": 0, "replaced": 0, "t_stash": 0.0,
                           "disk_withdrawn": 0, "disk_dropped": 0, "disk_left": 0}

        # Drafting mode
        if draft_model is not None and draft_model.caps.get("attach_target"):
            draft_model.attach_to(model)
        self.dflash_draft = self.draft_model is not None and self.draft_model.caps.get("dflash_draft", False)
        self.mtp_draft = self.draft_model is not None and self.draft_model.caps.get("mtp_draft", False)

        # Confidence-calibrated draft truncation (draft model + dynamic draft, any mode). For
        # DFlash the fixed-size drafted block is truncated before verification; for AR draft
        # models and MTP the drafting loop itself stops early, saving drafter forwards. Modes
        # whose TP lm_head path returns only the argmax simply never export confidence values,
        # leaving the calibrator inert
        self.draft_calibrator = None
        self._draft_conf_round = None
        if self.dynamic_draft and self.draft_model is not None:
            self.draft_calibrator = DraftConfidenceCalibrator(draft_confidence)

        # The MTP draft window chosen per step from a measured cost curve (_dw_begin). The policy
        # (generator/draft_window.py) is made at the first step that uses it; draft_window_stats
        # are its counters, the curve among them
        if draft_window == "fixed":
            draft_window = os.environ.get("EXL3_DRAFT_WINDOW", "").strip().lower() or "fixed"
        if draft_window not in ("fixed", "cost"):
            raise ValueError(f'draft_window must be "fixed" or "cost", not {draft_window!r}')
        self.draft_window = draft_window
        if draft_window_set is None:
            draft_window_set = os.environ.get("EXL3_DRAFT_WINDOW_SET")
        self.draft_window_set = parse_windows(draft_window_set)
        # The test hook's window schedule (draft_window.py): None unless the environment asks
        self.draft_window_schedule = parse_schedule(os.environ.get("EXL3_DRAFT_WINDOW_SCHEDULE"))
        if self.draft_window_schedule is not None:
            print(f" !! EXL3_DRAFT_WINDOW_SCHEDULE={os.environ.get('EXL3_DRAFT_WINDOW_SCHEDULE')}: a test hook, the "
                  "MTP draft window follows that schedule" + (" (draft_window = \"cost\" only: no effect with "
                  "the window fixed)" if self.draft_window != "cost" else ""), flush = True)
        self.draft_window_policy = None
        self.draft_window_stats = None
        self._dw_round = None
        self._dw_dirty = False
        if self.draft_window == "cost":
            if not self.mtp_draft:
                print(' !! draft_window = "cost" chooses the MTP head\'s window: no MTP head, the window stays fixed',
                      flush = True)
            else:
                if self.draft_window_set and max(self.draft_window_set) > self.num_draft_tokens:
                    print(f" !! Draft windows above num_draft_tokens ({self.num_draft_tokens}) are never taken: "
                          f"{self.draft_window_set}", flush = True)
                self._dw_policy()


    def num_remaining_jobs(self):
        return len(self.pending_jobs) + len(self.active_jobs)

    def num_active_jobs(self):
        return len(self.active_jobs)

    def num_pending_jobs(self):
        return len(self.pending_jobs)

    def get_cache_stats(self) -> dict:
        """
        Snapshot of cache occupancy and prompt-cache efficiency, intended for status displays.

        Returns a dict with:
            page_size: tokens per page
            max_tokens: total cache capacity, in tokens
            used_tokens: tokens in pages referenced by active jobs
            cached_tokens: tokens in complete pages that a later or concurrent job could actually reuse: every
                page on an unbroken chain down to a root, whether or not an active job currently references it.
                Complete pages whose ancestors were evicted (orphans) don't count, since a prefix match walks
                the chain from the root
            free_tokens: tokens in unreferenced pages holding no reusable content
            tier_cached_tokens: reusable tokens held only in the system-memory tier (sysmem K/V cache), by the
                same chain rule; 0 without a tier
            tier_max_tokens: capacity of the system-memory tier, 0 without one
            disk_cached_tokens: reusable tokens held only in the disk tier, by the same chain rule (a chain
                may run through all three tiers; the disk tier is persistent, so these survive a restart);
                0 without a disk tier
            disk_pages, disk_states: pages and recurrent checkpoints the disk tier holds for this model
            disk_used_bytes, disk_max_bytes: the disk tier's occupancy and budget (the budget is for the whole
                directory, every model and cache configuration under it); 0 without a disk tier
            alloc_pages: cumulative pages claimed by started jobs, the pages reserved for generation included
            alloc_prompt_pages: of those, the complete prompt pages, i.e. the ones a prompt-cache hit can serve
            alloc_cached_pages: of those, pages that were reused from the prompt cache
            alloc_tier_pages: of those, pages restored from the system-memory tier (including pages that were
                promoted there from disk first)
            alloc_disk_pages: of those, pages that came from the disk tier, whether the enqueue-time prefetch
                or the allocation itself promoted them; a page counts here the first time it is restored after
                its promotion and as an ordinary tier page from then on
            hit_rate: alloc_cached_pages / alloc_prompt_pages, or None before any job has started. The
                denominator is the prompt pages only, so this matches the per-request cached fraction rather
                than being diluted by the pages each job reserves for its output
            tier_hit_rate: alloc_tier_pages / alloc_prompt_pages, the part of hit_rate served from the
                system-memory tier (disk promotions included), or None before any job has started
            disk_hit_rate: alloc_disk_pages / alloc_prompt_pages, the part of tier_hit_rate that was read from
                disk, or None before any job has started
            active_jobs, pending_jobs: current job counts

        The chain walk is memoized on a signature of the page tables, so repeated calls between changes are
        cheap.
        """
        pt = self.pagetable
        tier = pt.cpu_tier
        disk = self.disk_page_cache

        # The chain walk is linear in the number of pages, which can be large with a big system-memory
        # tier, so it only reruns when something that can change the outcome has moved
        signature = (
            len(pt.referenced_pages),
            len(pt.unreferenced_pages),
            sum(1 for h in pt.referenced_pages if is_content_hash(h)),
            pt.metrics["alloc_pages"],
            pt.metrics["evictions"],
            pt.last_defrag_serial,
            len(tier) if tier is not None else 0,
            (tier.metrics["pushes"], tier.metrics["evictions"]) if tier is not None else None,
            (len(disk), len(disk.states), disk.metrics["page_writes"], disk.metrics["evictions"]) if disk is not None else None,
        )
        memo_signature, memo = self._cache_stats_memo
        if signature == memo_signature:
            return {**memo, "active_jobs": len(self.active_jobs), "pending_jobs": len(self.pending_jobs)}

        # prev-hash links of every complete page, per tier
        gpu_links = {}
        for pages in (pt.referenced_pages, pt.unreferenced_pages):
            for h, page in pages.items():
                if is_content_hash(h):
                    gpu_links[h] = page.prev_hash
        tier_links = {h: e["prev_hash"] for h, e in tier.entries.items()} if tier is not None else {}
        if disk is not None:
            with disk._lock:
                disk_links = {h: m["prev_hash"] for h, m in disk.pages.items()}
        else:
            disk_links = {}

        # A page is reusable when its whole chain is present in either tier
        resumable = {}
        def is_resumable(h):
            chain = []
            while h is not None and h not in resumable:
                chain.append(h)
                if h in gpu_links:
                    h = gpu_links[h]
                elif h in tier_links:
                    h = tier_links[h]
                elif h in disk_links:
                    h = disk_links[h]
                else:
                    resumable[h] = False
                    break
            result = True if h is None else resumable[h]
            for c in chain:
                resumable[c] = result
            return result

        cached = sum(1 for h in gpu_links if is_resumable(h))
        tier_cached = sum(1 for h in tier_links if h not in gpu_links and is_resumable(h))
        disk_cached = sum(1 for h in disk_links if h not in gpu_links and h not in tier_links and is_resumable(h))
        used = len(pt.referenced_pages)
        free = sum(1 for h in pt.unreferenced_pages if not is_content_hash(h))
        alloc = pt.metrics["alloc_pages"]
        alloc_prompt = pt.metrics["alloc_prompt_pages"]
        alloc_cached = pt.metrics["alloc_cached_pages"]
        alloc_tier = pt.metrics["alloc_tier_pages"]
        alloc_disk = pt.metrics["alloc_disk_pages"]
        stats = {
            "page_size": PAGE_SIZE,
            "max_tokens": pt.max_pages * PAGE_SIZE,
            "used_tokens": used * PAGE_SIZE,
            "cached_tokens": cached * PAGE_SIZE,
            "free_tokens": free * PAGE_SIZE,
            "tier_cached_tokens": tier_cached * PAGE_SIZE,
            "tier_max_tokens": tier.max_slots * PAGE_SIZE if tier is not None else 0,
            "disk_cached_tokens": disk_cached * PAGE_SIZE,
            "disk_pages": len(disk) if disk is not None else 0,
            "disk_states": len(disk.states) if disk is not None else 0,
            "disk_used_bytes": disk.total_size if disk is not None else 0,
            "disk_max_bytes": disk.root_size if disk is not None else 0,
            "alloc_pages": alloc,
            "alloc_prompt_pages": alloc_prompt,
            "alloc_cached_pages": alloc_cached,
            "alloc_tier_pages": alloc_tier,
            "alloc_disk_pages": alloc_disk,
            "hit_rate": alloc_cached / alloc_prompt if alloc_prompt else None,
            "tier_hit_rate": alloc_tier / alloc_prompt if alloc_prompt else None,
            "disk_hit_rate": alloc_disk / alloc_prompt if alloc_prompt else None,
            "active_jobs": len(self.active_jobs),
            "pending_jobs": len(self.pending_jobs),
        }
        self._cache_stats_memo = (signature, stats)
        return stats


    def clear_queue(self):
        """
        Abort all active and pending jobs
        """

        num_jobs = self.num_remaining_jobs()
        # A pass issued in part is abandoned with its jobs (their pages and states go below)
        self._stagger_drop()
        # A chunk under way on the prefill thread is left to finish first, and so are the
        # chunks in flight on two cards (each stops at the half it is in)
        self._side_drop()
        self._pipe_drop()
        for job in self.active_jobs + self.pending_jobs:
            job.deallocate_pages()
        self.active_jobs.clear()
        self.pending_jobs.clear()
        if num_jobs and not self.num_remaining_jobs():
            self.on_queue_drained()


    def enqueue(
        self,
        job: Job | list[Job]
    ) -> int | list[int]:
        """
        Adds a job or list of jobs to the queue.

        Each job is prepared against this generator before it is appended to the pending queue. Preparation assigns
        the current job_serial, hashes prompt pages for later cache lookup, and validates that the request can fit
        the configured cache and batch limits. The serial is then incremented and returned to the caller; it is also
        included in started, prefill and streaming result dictionaries so clients can correlate events with the
        logical request. Requeued jobs keep their original serial number internally so a long generation still
        appears as one stream.

        returns:
            int: (List of) unique serial number(s) for job(s)
        """

        if isinstance(job, list):
            serials = []
            for j in job:
                serials.append(self.enqueue(j))
            return serials

        job.prepare_for_queue(self, self.job_serial)
        self.job_serial += 1
        # The test hook's draft window schedule starts again with each stretch of work (a run of
        # the exact check takes it from its first entry, whatever ran before)
        if self.draft_window_policy is not None and not self.pending_jobs and not self.active_jobs:
            self.draft_window_policy.restart_schedule()
        self.pending_jobs.append(job)
        if self.disk_page_cache is not None:
            # Start reading the prompt's pages that only the disk tier holds, so the allocation finds them in
            # the CPU tier (or in flight) rather than waiting on the disk itself
            for seq in job.sequences:
                hashes = seq.page_hashes if seq.max_cached_pages is None else seq.page_hashes[:seq.max_cached_pages]
                self.disk_page_cache.prefetch(hashes, self.pagetable.access_serial)
        return job.serial_number


    def cancel(
        self,
        job: Job
    ):
        """
        Cancel a single pending or active job.

        Pending jobs are removed before they ever allocate cache pages. Active jobs first release their page
        references and recurrent state, then leave active_jobs so future iterations stop sampling them. If the
        cancellation drains the generator completely, the page table is defragmented immediately because no active
        block tables still depend on the current physical page ordering.
        """

        # The job's pass may be issued in part (staggered decode): its group is finished and
        # sampled first, so no kernel is left to write into pages and states about to be
        # released. The other jobs' tokens are handed out by the next iterate(). A generator
        # whose cache a newer one has taken over issues nothing more on it
        flight = self._flight
        if flight is not None and any(j is job for j in flight.jobs):
            if self._cache_superseded():
                self._stagger_drop()
            else:
                backlog = []
                self._stagger_drain(backlog)
                self._stagger_backlog += [r for r in backlog if r.get("job") is not job]

        # Likewise for a chunk of its prompt under way on the prefill thread: it is left to
        # finish, and not recorded (also when a newer generator has taken the cache over: the
        # thread then issues nothing more of it)
        if self._side is not None and self._side.job is job:
            self._side_drop()
        # In flight on two cards: its chunks stop at the half they are in, which is waited for
        self._pipe_drop(job)

        num_jobs = self.num_remaining_jobs()

        if job in self.pending_jobs:
            self.pending_jobs.remove(job)
        elif job in self.active_jobs:
            job.deallocate_pages()
            self.active_jobs.remove(job)
        if num_jobs and not self.num_remaining_jobs():
            self.on_queue_drained()


    @torch.inference_mode
    def iterate(self) -> list[dict]:
        """
        Performs inference on available jobs.

        :return:
            List of dicts:

            # Job has started
            {
                "job": Job  - reference to job
                "stage": "started"
                "identifier":  - optional identifier
                "serial": int  - job serial number
                "eos": bool  - always False at this stage
            }

            # Prefill is underway
            {
                "job": Job  - reference to job
                "stage": "prefill"
                "curr_progress": int  - prompt tokens ingested so far
                "max_progress": int  - total prompt tokens to ingest
                "identifier":  - optional identifier
                "serial": int   - job serial number
                "eos": bool  - always False at this stage
            }

            # Generation is underway
            {
                "job": Job  - reference to job
                "stage": "streaming"
                "identifier":  - optional identifier
                "serial": int   - job serial number
                "eos": bool  - True if stop condition has been met

                optional, if eos:
                    "eos_reason":  - one of:
                        "stop_token"  (stop token was reached)
                        "stop_string"  (stop string was completed)
                        "max_new_tokens"  (max_new_tokens reached)
                        "end_filter"  (filter reached end state with eos_after_completed = True)
                        "loop_detected"  (loop detection triggered)
                    optional, if "eos_reason" == "stop_token":
                        "eos_triggering_token_id": int
                        "eos_triggering_token_str": str
                    optional, if "eos_reason" == "stop_string":
                        "eos_triggering_string": str
                    "full_completion": str  - full text completion
                    "new_tokens": int  - number of tokens generated
                    "time_enqueued": float  - time from job was enqueued until it started, in seconds
                    "time_prefill": float  - time to first token, in seconds
                    "time_generate": float  - time to last token, in seconds
                    optional, if SD enabled:
                        "accepted_draft_tokens": int
                        "rejected_draft_tokens": int

                "text": str  - streamed text output. Does not include prefix from healed token, or stop string
                "token_ids": torch.Tensor  - output tokens, shape (1, n)
                "token_probs": torch.Tensor  - last sampling probability of output tokens, shape (1, n)
                "top_k_tokens": torch.Tensor  - shape (1, n, k)
                "top_k_probs": torch.Tensor  - shape (1, n, k)
                "logits": torch.Tensor  - shape (1, n, vocab_size)
            }
        """

        assert self.cache.initialized, \
            "Cache tensors were never allocated. Construct the Cache BEFORE calling model.load()"
        owner_serial = getattr(self.cache, "owner_serial", None)
        if owner_serial is not None and owner_serial != getattr(self, "cache_owner_serial", owner_serial):
            # Superseded: drop every job without touching the cache (the new owner may already have
            # reused these pages and state slots) and refuse to run. A chunk under way on the
            # prefill thread is waited for: nothing more of it may be issued (in flight on two
            # cards, the half each chunk is in)
            self._stagger_drop()
            self._side_drop()
            self._pipe_drop()
            self.pending_jobs.clear()
            self.active_jobs.clear()
            raise RuntimeError(
                "This Generator no longer owns its Cache: a newer Generator was created over the same "
                "Cache. Close or discard the old Generator before creating a new one."
            )
        assert self.draft_cache is None or self.draft_cache.initialized, \
            "Draft cache tensors were never allocated. Construct the draft Cache BEFORE calling draft_model.load()"

        results = []
        if self._stagger_backlog:
            results, self._stagger_backlog = self._stagger_backlog, []
        self.iterate_start_jobs(results)

        # With the cost-chosen draft window: a decode step with a prompt being ingested in the same
        # iteration is not timed (_dw_end)
        if self.draft_window == "cost":
            self._dw_dirty = any(not job.is_prefill_done() for job in self.active_jobs)

        # Perform one round of prefill
        self._prefill_round(results)

        # Recurrent checkpoints
        if self.recurrent_cache is not None:
            self.recurrent_checkpoint()

        # Two groups of jobs half a pass apart (layers split over two GPUs); the staggered step
        # drafts for its groups itself
        staggered = self._stagger_wanted()
        if staggered:
            self.iterate_staggered(results)

        # Generation with draft model
        elif self.draft_model:
            if self.dflash_draft:
                draft_tokens = self.iterate_draftmodel_dflash_gen(results)
            elif self.mtp_draft:
                draft_tokens = self.iterate_draftmodel_mtp_gen(results)
            else:
                draft_tokens = self.iterate_draftmodel_gen(results)

        # Generation with n-gram draft
        elif self.ngram_match_min:
            draft_tokens = self.iterate_ngram_gen(results)

        # Regular generation
        else:
            draft_tokens = None

        # Long suffix-match draft over the regular one, when a job is repeating its context
        if self.hybrid_draft_tokens and not staggered:
            draft_tokens = self.iterate_hybrid_gen(draft_tokens)

        if not staggered:
            self.iterate_gen(results, draft_tokens)

        # Back to regular steps (also after a cancel completed the pass in flight between calls)
        if self._stagger_run is not None and self._flight is None:
            self._stagger_run_end()

        # Visualization
        if self.visualizer:
            self.update_visualizer()

        # Disk tier: queue the complete pages of the active sequences (see disk_cache_write_interval)
        if self.disk_page_cache is not None and self.active_jobs and self.disk_cache_write_interval > 0:
            self.write_behind()

        # Finished iteration
        return results


    def close(self):
        """
        Release the host memory a retired generator would otherwise hold until it is garbage collected,
        which can be long after it stops being used: a generator sits in a reference cycle with its page
        table, and the jobs that failed on it keep it reachable through their exception tracebacks. The
        model and cache stay loaded for a replacement generator to use; this generator is not reusable
        afterwards. Safe to call more than once.

        Releases, in order of size: the recurrent checkpoint cache (up to recurrent_cache_size of system
        RAM, in the tensor-parallel ranks too), the CPU page cache tier (cpu_cache_size of pinned memory),
        the pinned staging and draft buffers, and the filter thread pool.
        """
        # A pass issued in part (staggered decode) is abandoned, not completed: nothing more is issued
        # by a retired generator, which may also have failed. What its first half writes is past every
        # complete page the flush below persists. Its second set of pinned buffers goes too
        self._stagger_drop()
        self._lanes = {}
        self._stagger_spec = None
        # A chunk under way on the prefill thread is waited for (it writes pages and states the
        # flush below and a replacement generator would read), and the thread goes. Chunks in
        # flight on two cards stop at the half they are in; both threads go
        self._side_drop()
        self._pipe_drop()
        for pool in (self._side_pool, self._pipe_pool):
            if pool is not None:
                pool.shutdown(wait = True)
        if self.disk_page_cache is not None:
            # Persist what is still only in VRAM or RAM, then let the writer drain before its sources go away.
            # After abort() (generator error), nothing is written
            if self.disk_cache_flush_on_idle and not self.disk_page_cache.write_disabled:
                self.flush_cache_to_disk()
            self.disk_page_cache.close()
            self.disk_page_cache = None
            if self.recurrent_cache is not None:
                self.recurrent_cache.disk_tier = None
        if self.recurrent_cache is not None:
            self.recurrent_cache.close()
        if self.cpu_page_cache is not None:
            self.cpu_page_cache.close()
            self.cpu_page_cache = None
            self.pagetable.cpu_tier = None
        self.sample_pinned = None
        self.staging_buffers = {}
        self.draft_input_ids_pinned = None
        self.draft_ids_pinned = None
        self.filter_pool.shutdown(wait = False)


    @torch.inference_mode()
    def flush_cache_to_disk(self, wait: bool = False, max_pages: int | None = None) -> dict:
        """
        Queue every complete, content-hashed page that is in VRAM but in neither the CPU nor the disk tier for
        persistence (through the CPU tier, whose store is the write-through point), and every recurrent
        checkpoint not yet on disk. Returns counts of what was queued; with wait = True returns once the
        writer has finished.

        max_pages limits the number of pages queued in one call. Each page takes a CPU-tier slot until its
        write is complete. If the pages are more than the free slots, the tier must wait for the writer, and
        this call stops the generator for that time. With a limit, the call stops early and sets "more" in the
        result. The idle loop then calls again (see idle_work).
        """
        disk = self.disk_page_cache
        tier = self.cpu_page_cache
        queued = {"pages": 0, "states": 0, "more": False}
        if disk is None:
            return queued
        pt = self.pagetable
        pages = list(pt.unreferenced_pages.values()) + list(pt.referenced_pages.values())
        queued["pages"], queued["more"] = self._queue_pages_to_disk(pages, max_pages)
        rc = self.recurrent_cache
        if rc is not None and rc.disk_tier is not None:
            # Checkpoints follow their pages: every complete page is queued above, so an anchored checkpoint
            # is written now, the rest stay candidates. The last checkpoint of each chain (no other checkpoint
            # in RAM is after it on its chain) is written also when it is closer than min_state_distance to
            # the one before: a conversation continues from the end of its chain
            keys = [h for h in rc.keys() if is_content_hash(h)]
            keyset = set(keys)
            covered = set()
            limit = pt.max_pages + len(tier) + len(disk)
            for k in keys:
                h = disk.chain_prev(k)
                steps = 0
                while h is not None and steps <= limit and h not in covered:
                    if h in keyset:
                        covered.add(h)
                    h = disk.chain_prev(h)
                    steps += 1
            for h in keys:
                stashed = rc.get(h)
                if stashed is None or not (disk.has_page(h) or h in disk.pending_pages):
                    continue
                if disk.write_state(h, stashed, force = h not in covered):
                    queued["states"] += 1
        queued["states_pruned"] = disk.prune_states()
        self._disk_backlog = queued["more"]
        if wait:
            disk.flush()
        return queued


    def _queue_pages_to_disk(self, pages, budget: int | None) -> tuple[int, bool]:
        """
        Queue the complete, content-hashed, persistable pages among `pages` that are in VRAM but in neither the
        CPU nor the disk tier, through the CPU tier (its store is the write-through point; the copy to host
        memory goes on the current stream, after the kernels that wrote the page and before any that could
        write it again). Stops after `budget` pages. Returns (pages queued, stopped early).
        """
        disk, tier, pt = self.disk_page_cache, self.cpu_page_cache, self.pagetable
        queued = 0
        for page in pages:
            h = page.phash
            if page.kv_position != PAGE_SIZE or not is_content_hash(h) or not page.persistable:
                continue
            if h in tier or disk.has_page(h) or h in disk.pending_pages:
                continue
            if budget is not None and queued >= budget:
                return queued, True
            tier.store(page, pt.access_serial)
            queued += 1
        return queued, False


    @torch.inference_mode()
    def write_behind(self):
        """
        Queue the complete pages of the active sequences for the disk tier, at most once per
        disk_cache_write_interval seconds, so a process that dies during a long prefill loses only what came
        after the last page (and checkpoint) written. The generator does not wait for the copies or the disk.

        The last two pages before a sequence's position are not queued: speculative decoding can still rewind
        them, and the MTP draft cache writes the last slot of a page one step later. One call queues at most
        a quarter of the CPU tier, less the writes still queued, so the tier never has to wait for the writer
        to free a slot.
        """
        now = time.monotonic()
        if now - self._write_behind_time < self.disk_cache_write_interval:
            return
        self._write_behind_time = now
        disk = self.disk_page_cache
        if disk.write_disabled:
            return
        budget = self.cpu_page_cache.max_slots // 4 - len(disk.pending_pages)
        if budget <= 0:
            return
        pages = []
        for job in self.active_jobs:
            for seq in job.sequences:
                if seq.allocated_pages:
                    pages.extend(seq.allocated_pages[:max(0, seq.kv_position // PAGE_SIZE - 2)])
        self._queue_pages_to_disk(pages, budget)


    def _idle_flush_budget(self) -> int:
        """Pages for one idle flush step: half the CPU tier, less the writes still queued"""
        return max(1, self.cpu_page_cache.max_slots // 2 - len(self.disk_page_cache.pending_pages))


    def has_idle_work(self) -> bool:
        """True if the idle flush stopped early and has more pages to queue"""
        return self.disk_page_cache is not None and getattr(self, "_disk_backlog", False)


    @torch.inference_mode()
    def idle_work(self):
        """Do one more idle flush step, if no job is queued. The async generator calls this while idle"""
        if self.has_idle_work() and not self.num_remaining_jobs():
            self.flush_cache_to_disk(max_pages = self._idle_flush_budget())


    @torch.inference_mode()
    def on_queue_drained(self):
        """
        Idle-transition housekeeping: drop recurrent checkpoints stranded by KV eviction, then defragment the
        page table (both need/prefer a moment with no active block tables), and return whatever host memory
        glibc is retaining from stash/offload churn (issue #277) — sub-threshold residue would otherwise sit
        in the arenas for the whole idle period.
        """
        if self.disk_page_cache is not None and self.disk_cache_flush_on_idle and \
                not self.disk_page_cache.write_disabled:
            # Before pruning: a checkpoint stranded in VRAM/RAM terms may be anchored by pages on disk. One
            # step only; the async generator does the rest while idle (idle_work)
            self.flush_cache_to_disk(max_pages = self._idle_flush_budget())
        if self.recurrent_cache is not None:
            self.recurrent_cache.prune_stranded()
            # The pruned checkpoints' buffers went back to the stash pool; drop them so the RAM
            # is actually returned (the next stash reallocates once)
            host_pool.release()
            if self.model.loaded_tp:
                self.model.tp_dispatch_all(mp_host_pool_release, ())
        self.pagetable.defrag()
        # Dynamic expert placement: apply any pending swap sweep now, between generations —
        # a placement change perturbs the logits slightly (same expert, different device
        # numerics) and must never land mid-stream
        from ..modules.block_sparse_mlp_cpu import run_pending_swap_sweeps
        run_pending_swap_sweeps(self.model.config.infer_params)
        malloc_trim()


    def recurrent_checkpoint(self):
        flying = self._flight.jobs if self._flight is not None else ()
        side_job = self._side.job if self._side is not None else None
        for job in self.active_jobs:
            # The states of a job in flight are half a pass ahead on the first card only; it was
            # offered its checkpoint at this very position before it was issued
            if flying and any(j is job for j in flying):
                continue
            # The state of a job whose prompt chunk is under way is being advanced by the prefill
            # thread; it was offered its checkpoint after the previous chunk. In flight on two
            # cards the same for any chunk of the job under way
            if job is side_job or any(p.job is job for p in self._pipe):
                continue
            job.maybe_stash_recurrent(self.recurrent_cache)


    def update_visualizer(self):
        chains = []
        for job in self.active_jobs:
            for seq in job.sequences:
                idx = job.serial_number
                chain = [page.page_index for page in seq.allocated_pages]
                chains.append((idx, chain))
        usage = [0] * self.pagetable.max_pages
        for page in self.pagetable.all_pages:
            usage[page.page_index] = page.kv_position / PAGE_SIZE
        self.visualizer.update(chains, usage)


    def iterate_draftmodel_gen(self, results: list):

        self._draft_conf_round = None

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens + 1)
            batch_size += 1
        if batch_size == 0:
            return None

        # Create block index table for batch
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        # Pinned staging (as iterate_gen), so the uploads are stream-ordered instead of blocking
        # pageable copies; the draft's own buffers, since the target's are still in flight
        max_pages_batch = (max_pages_batch + 15) // 16 * 16
        block_index = self._staging("draft_block_index", batch_size, max_pages_batch)
        block_index.zero_()
        cache_seqlens = self._staging("draft_cache_seqlens", batch_size)
        batch = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = seq.block_index_tensor[:, :max_pages_batch]
                block_index[batch:batch+1, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1

        # Indexed embeddings not supported when drafting
        # TODO: Allow multimodal draft model, perhaps with dummy embeddings?
        for job in self.active_jobs:
            assert not job.embeddings, \
                "MM embeddings not supported while using draft model."

        # Recurrent drafter: its states must stand exactly at the K/V position before the round (the
        # draft consumes every token it drafts, so this holds unless a hybrid round accepted past its
        # window); a lagging state first consumes the gap in a plain pass
        draft_states = None
        if self.draft_recurrent:
            draft_states = []
            for job in self.active_jobs:
                if not job.is_prefill_done(): continue
                seq = job.sequences[0]
                ds = job.recurrent_state.draft
                if ds.position < seq.kv_position:
                    gap = seq.sequence_ids.torch_slice(ds.position, seq.kv_position)
                    self.draft_model.prefill(gap, {
                        "attn_mode": "flash_attn",
                        "block_table": seq.block_index_tensor,
                        "cache": self.draft_cache,
                        "cache_seqlens": torch.tensor([ds.position], dtype = torch.int32),
                        "recurrent_states": [ds],
                    })
                assert ds.position == seq.kv_position, "recurrent draft state out of step with the K/V position"
                draft_states.append(ds)

        # Collect input IDs
        input_ids_list = []
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list()
            input_ids_list += job_ids
        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))

        def draft_params(step):
            p = {
                "attn_mode": "flash_attn",
                "block_table": block_index,
                "cache": self.draft_cache,
                "cache_seqlens": cache_seqlens,
            }
            if draft_states is not None:
                # One speculative pass over the drafted tokens, fed one step at a time; the round's
                # verification settles it (rewind to the accepted prefix) in iterate_gen
                p.update({"recurrent_states": draft_states, "recurrent_history": True, "recurrent_spec_step": step})
            return p

        # Greedy sample batched draft tokens. With a confidence calibrator, drafting stops early
        window = self.num_draft_tokens
        cal = self.draft_calibrator
        conf_cols = []
        reach = None
        for idx in range(window):
            batch_logits = self.draft_model.forward(input_ids = batch_ids, params = draft_params(idx))
            if cal is not None:
                conf, new_ids = torch.max(batch_logits, dim = -1)
            else:
                new_ids = torch.argmax(batch_logits, dim = -1)
            self.draft_ids_pinned[:batch_size, idx:idx+1].copy_(new_ids)
            batch_ids.copy_(new_ids)
            cache_seqlens += 1
            if cal is not None:
                c = conf.float().cpu()
                conf_cols.append(c)
                est = [cal.estimate(v) for v in c.view(-1).tolist()]
                reach = est if reach is None else [r * e for r, e in zip(reach, est)]
                if idx + 1 < window and max(reach) < cal.confidence:
                    window = idx + 1
                    break

        # The last drafted token goes into the draft cache too, so a fully accepted round leaves the
        # drafter at the next unprocessed token
        self.draft_model.prefill(input_ids = batch_ids, params = draft_params(window))

        if conf_cols:
            self._draft_conf_round = {
                "ids": self.draft_ids_pinned[:batch_size, :window],
                "conf": torch.cat(conf_cols, dim = 1),
                "window": window,
            }

        return self.draft_ids_pinned[:, :window]


    def iterate_draftmodel_mtp_gen(self, results: list):

        self._draft_conf_round = None
        self._dw_round = None
        jobs = self._step_jobs()

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens + 1)
            batch_size += 1
        if batch_size == 0:
            return None

        # Create block index table for batch
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        # Pinned staging (as iterate_gen), so the uploads are stream-ordered instead of blocking
        # pageable copies; the draft's own buffers, since the target's are still in flight
        max_pages_batch = (max_pages_batch + 15) // 16 * 16
        block_index = self._staging("draft_block_index", batch_size, max_pages_batch)
        block_index.zero_()
        cache_seqlens = self._staging("draft_cache_seqlens", batch_size)
        batch = 0
        for job in jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = seq.block_index_tensor[:, :max_pages_batch]
                block_index[batch:batch+1, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1

        # Collect input IDs
        input_ids_list = []
        mtp_hidden_list = []
        for job in jobs:
            if not job.is_prefill_done(): continue
            assert len(job.sequences) == 1, "Qwen3.5 MTP drafting does not currently support CFG/multi-sequence jobs"
            if job.mtp_last_hidden is None:
                # A one-token prompt has no token to prefill before the generation input.
                # Run one normal target step first; iterate_gen() will initialize MTP state.
                return None
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list()
            input_ids_list += job_ids
            mtp_hidden_list.append(job.mtp_last_hidden)
        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))
        temp_hidden = torch.cat(mtp_hidden_list, dim = 0)

        # Greedy sample batched draft tokens. As in iterate_draftmodel_gen, drafting stops once
        # every row's running product of estimated conditional acceptance probabilities falls
        # below the confidence target, keeping the first low-confidence token as the label probe
        window = self.num_draft_tokens
        cal = self.draft_calibrator
        conf_cols = []
        reach = None

        # With draft_window = "cost" the window is chosen as the head drafts (_dw_begin): the head's
        # confidences come back with each position's drafts, the policy says when to stop and which
        # window to verify. It takes the place of the confidence-target draft
        dw = self._dw_begin(batch_size)
        if dw is not None:
            cal = None

        # With mtp_batched_update the head's update of the last step made each job's first draft
        # already (_mtp_fold), at the position it stands at: read back, the head runs from the second
        # draft token on. One job without one and the head drafts for all, as always
        start = 0
        first_conf = None
        if self.mtp_batched_update:
            start, temp_hidden, first_conf = self._mtp_take_folds(
                jobs, batch_size, batch_ids, cache_seqlens, temp_hidden, dw is not None,
            )
        stop = False
        if dw is not None:
            dw.start = start
            window = dw.draft_limit()
            if start:
                stop = dw.add(first_conf)

        for idx in range(start, start if stop else window):
            params = {
                "target_hidden": temp_hidden,
                "attn_mode": "flash_attn",
                "block_table": block_index,
                "cache": self.draft_cache,
                "cache_seqlens": cache_seqlens,
                "draft_step": idx,   # heads specialized per depth pick their head from this
            }
            if cal is not None or dw is not None:
                params["export_draft_conf"] = True
            batch_state = self.draft_model.forward(batch_ids, params)
            lm_head = self.model.modules[self.model.logit_layer_idx]
            batch_state = lm_head.prepare_for_device(batch_state, params)
            new_ids = self.draft_model.sample_from_state(batch_state, params)
            conf_buf = self._dw_stage_conf(params.get("draft_conf"), batch_size) if dw is not None else None
            self.draft_ids_pinned[:batch_size, idx:idx+1].copy_(new_ids)
            if self.mtp_batched_update:
                # The next draft step's inputs from the copy just read back, not a second readback
                batch_ids.copy_(self.draft_ids_pinned[:batch_size, idx:idx+1])
                self._note_readbacks(1)
            else:
                batch_ids.copy_(new_ids)
                self._note_readbacks(2)
            cache_seqlens += 1
            temp_hidden = batch_state
            if dw is not None:
                # The confidences came back with the drafts (the blocking copy above waited for them)
                dw.passes += 1
                if dw.add(None if conf_buf is None else conf_buf.view(-1).tolist()):
                    break
                continue
            draft_conf = params.get("draft_conf")
            if cal is not None and draft_conf is not None:
                c = draft_conf.float().cpu()
                conf_cols.append(c)
                est = [cal.estimate(v) for v in c.view(-1).tolist()]
                reach = est if reach is None else [r * e for r, e in zip(reach, est)]
                if idx + 1 < window and max(reach) < cal.confidence:
                    window = idx + 1
                    break

        if dw is not None:
            # The window the policy chose; 0 verifies no draft (the step samples one token per job)
            window = dw.close()
            self._dw_round = dw
            if window == 0:
                # Not verified, but the token the step samples labels the first draft (_dw_end)
                dw.probe = self.draft_ids_pinned[:batch_size, 0].tolist()
                return None
            dw.drafts = self.draft_ids_pinned[:, :window]
            return dw.drafts

        if conf_cols and len(conf_cols) == window:
            self._draft_conf_round = {
                "ids": self.draft_ids_pinned[:batch_size, :window],
                "conf": torch.cat(conf_cols, dim = 1),
                "window": window,
            }

        return self.draft_ids_pinned[:, :window]


    # ---------------------------------------------------------------------------------------
    #   The MTP draft window chosen from a measured cost curve (draft_window = "cost")
    # ---------------------------------------------------------------------------------------

    def _dw_policy(self) -> DraftWindowPolicy | None:
        """The cost policy of the draft window when it is on (draft_window = "cost", the MTP head
        drafting), else None. Made at the first use."""
        if self.draft_window != "cost" or not self.mtp_draft:
            return None
        policy = self.draft_window_policy
        if policy is None:
            policy = self.draft_window_policy = DraftWindowPolicy(self.draft_window_set)
            self.draft_window_stats = policy.stats
        return policy


    def _dw_begin(self, n: int):
        """
        The draft step of n jobs starts: the round of the cost policy that chooses its window
        (DraftWindowRound), None when the policy is off.

        A prompt chunk under way on the prefill thread (or two in the pipeline) runs beside this
        step's decode only because the decode is at most side_max_rows rows x tokens, which
        _decode_rows counted at the narrowest window the policy may take: the window is capped so
        the verification pass stays within that (n x (1 + W) rows; the head's fold has as many).
        Such a step is not timed, and neither is one in an iteration that ingests a prompt.
        """
        policy = self._dw_policy()
        if policy is None:
            return None
        cap = None
        if self._side is not None or any(p.future is not None for p in self._pipe):
            cap = self.side_max_rows // max(1, n) - 1
        return policy.begin(
            "s" if self._step_group is not None else "r", n, self.num_draft_tokens, cap,
            timed = not self._dw_dirty and cap is None,
            switches = (bool(self.mtp_batched_verify), bool(self.mtp_batched_update)),
            schedule = self.draft_window_schedule,
        )


    def _dw_stage_conf(self, conf, n: int):
        """The head's confidences of a drafted position (draft_conf: the winning logit per row),
        copied into pinned memory without a wait of their own: the blocking readback of the drafts
        that follows on the same stream completes it. None when the head exported none (or not
        one per row)."""
        if conf is None or conf.numel() != n:
            return None
        buf = self._staging("dw_conf", n, 1, torch.float)
        buf.copy_(conf.reshape(n, 1).float(), non_blocking = True)
        return buf


    def _dw_end(self, dw, jobs, logit_mapping, accepted_lengths, rewound_jobs, windows, draft_tokens):
        """
        A decode step's sampling is done (the host has just waited for its pass): the step's time,
        the calibrator's labels and the counters of the cost policy. Returns the round when the step
        verified the policy's own window (not a hybrid round's), else None.

        The step's time is from the end of the last decode step's sampling to now, so it includes
        everything a step costs (the last step's head update, the draft passes, the pass, the
        sampling, the caller between two iterate() calls), and goes to the curve only if nothing
        else ran in it: the same jobs (no job started, ended or was requeued since), no prompt
        ingested in the iteration (_dw_dirty), no tail checkpoint taken, no cap on the window, the
        head drafting no more positions than the window verifies, and in a staggered decode a call
        that also issued the next group. A staggered call's time goes to the group it sampled.

        Labels: with the window sampled at once (mtp_batched_verify) the target's sample at every
        position of the window was made as if the drafts before it were accepted, so every
        position is a label (with a filter, up to the first draft not matched); serially, the
        accepted positions and the first one that was not. A window of 0 labels the first draft
        (made, not verified) with the token the step sampled, so the labels never stop while the
        policy verifies nothing.
        """
        policy = self.draft_window_policy
        if policy is None:
            return None
        t = time.perf_counter()
        staggered = self._step_group is not None
        own = dw is not None and (draft_tokens is dw.drafts if dw.window else draft_tokens is None)
        tail = getattr(self, "tail_stats", None)
        signature = (
            staggered, bool(self.mtp_batched_verify), bool(self.mtp_batched_update),
            tail.get("stashes") if tail else None, tuple(id(j) for j in self.active_jobs),
        )
        clean = own and not self._dw_dirty and not dw.capped and dw.passes == dw.expected_passes() \
            and (not staggered or self._flight is not None)
        policy.point(t, signature, dw.key() if own else None, clean, own and dw.target is not None)
        if dw is None:
            return None
        if not own:
            policy.overridden(dw)
            return None

        w = dw.window
        rows = [job for job, a, b in zip(jobs, logit_mapping[:-1], logit_mapping[1:]) if a != b]
        accepted = 0
        if len(rows) == dw.n == len(accepted_lengths):
            for row, job in enumerate(rows):
                n_acc = accepted_lengths[row] - 1
                accepted += n_acc
                if id(job) in rewound_jobs or not dw.conf:
                    continue
                window = windows.get(id(job)) if windows else None
                if w == 0:
                    # The first draft against the token the target sampled there
                    seq = job.sequences[0]
                    m = len(seq.sequence_ids)
                    labels = [(0, dw.probe[row] == seq.sequence_ids.torch_slice(m - 1, m).item())]
                elif window is not None:
                    # (with a filter, the rows past a draft its mask refused repeat the last mask:
                    # not labels)
                    drafts = draft_tokens[row, :w].tolist()
                    upto = min(w, window[1] + 1) if job.filters else w
                    labels = [(r, window[0][r] == drafts[r]) for r in range(upto)]
                else:
                    drafts = draft_tokens[row, :w].tolist()
                    labels = [(r, True) for r in range(n_acc)]
                    if n_acc < w:
                        seq = job.sequences[0]
                        m = len(seq.sequence_ids)
                        labels.append((n_acc, drafts[n_acc] == seq.sequence_ids.torch_slice(m - 1, m).item()))
                for r, ok in labels:
                    conf = dw.conf[r] if r < len(dw.conf) else None
                    if conf is not None:
                        policy.label(conf[row], ok)
        policy.finish(dw, sum(accepted_lengths), accepted)
        return dw


    # TODO: Refactor, share code with other draft fns
    def iterate_draftmodel_dflash_gen(self, results: list):

        self._draft_conf_round = None

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(
                max_seq_len,
                job.get_max_seq_len() + self.num_draft_tokens + 1,
                job.get_max_seq_len() + self.draft_model.config.block_size + 1
            )
            batch_size += 1
        if batch_size == 0:
            return None

        # The diffusion drafter always runs at its fixed block size, dynamic window truncates the drafted block
        window = self.num_draft_tokens

        # Create block index table for batch
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        # Pinned staging (as iterate_gen), so the uploads are stream-ordered instead of blocking
        # pageable copies; the draft's own buffers, since the target's are still in flight
        max_pages_batch = (max_pages_batch + 15) // 16 * 16
        block_index = self._staging("draft_block_index", batch_size, max_pages_batch)
        block_index.zero_()
        cache_seqlens = self._staging("draft_cache_seqlens", batch_size)
        batch = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = seq.block_index_tensor[:, :max_pages_batch]
                block_index[batch:batch+1, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1

        # Collect input IDs
        input_ids_list = []
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list()
            input_ids_list += job_ids
        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))

        # Run draft model. A drafter with sliding-window states appends its block to each row's
        # window ring; the rows the verification accepts are written over it afterwards
        # (update_kv_from_target below), which also rolls the state back to the K/V position
        params = {
            "attn_mode": "flash_attn",
            "block_table": block_index,
            "cache": self.draft_cache,
            "cache_seqlens": cache_seqlens,
        }
        if self.draft_recurrent:
            params["recurrent_states"] = [job.recurrent_state.draft for job in self.active_jobs if job.is_prefill_done()]
        if self.draft_calibrator is not None:
            params["export_draft_conf"] = True
        out_state = self.draft_model.forward(
            input_ids = batch_ids,
            params = params,
        )
        new_ids = self.draft_model.sample_from_state(out_state, params)

        # Draft models with a confidence head cap the usable draft length per round;
        # 0 means no draft position cleared the threshold, so skip drafting entirely
        conf_len = params.get("draft_confidence_len")
        if conf_len is not None:
            if conf_len == 0:
                return None
            window = min(window, conf_len)

        # Crop out the first token after sampling to keep batch contiguous for lm_head
        new_ids = new_ids[:, 1:]

        # Confidence-calibrated truncation: cut the batch window at the first position whose
        # drafter confidence falls below the calibrated threshold, taking the longest cut across
        # rows (jobs wanting less just reject earlier). The full block and
        # its scores are kept so iterate_gen can label the verified positions afterwards
        draft_conf = params.get("draft_conf")
        if self.draft_calibrator is not None and draft_conf is not None:
            conf = draft_conf[:batch_size, 1:].float().cpu()
            ids_full = new_ids[:batch_size, :].cpu()
            thr = self.draft_calibrator.threshold()
            below = conf[:, :window] < thr
            any_below = below.any(dim = 1)
            first_below = below.int().argmax(dim = 1)
            cuts = torch.where(any_below, first_below, torch.full_like(first_below, window))
            w_used = int(cuts.max().item())
            self._draft_conf_round = {"ids": ids_full, "conf": conf, "window": w_used}
            if w_used == 0:
                return None
            window = w_used

        self.draft_ids_pinned[:batch_size, :window].copy_(new_ids[:batch_size, :window])
        return self.draft_ids_pinned[:, :window]


    def iterate_ngram_gen(self, results: list):

        # Get shape of active batch
        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens + 1)
            batch_size += 1
        if batch_size == 0:
            return None

        # Generate draft
        window = self.num_draft_tokens
        draft_ids = []
        min_len = window
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            d = job.get_ngram_draft(window)
            min_len = min(min_len, d.shape[-1])
            draft_ids.append(d)

        if min_len == 0:
            return None

        # Trim to minimum length in batch
        draft_ids = torch.cat([d[:, :min_len] for d in draft_ids], dim = 0)
        return draft_ids


    def new_recurrent_state(self) -> PairedState:
        """A fresh (cleared) pair of recurrent states for a new sequence, one side per recurrent model"""
        target = draft = None
        try:
            target = self.cache.get_new_state() if self.target_recurrent else None
            draft = self.draft_cache.get_new_state() if self.draft_recurrent else None
            return PairedState(target, draft)
        except BaseException:
            if draft is not None:
                draft.free()
            if target is not None:
                target.free()
            raise


    def recurrent_state_from_stashed(self, stashed: dict, position: int) -> PairedState:
        """The pair restored from a checkpoint stash (PairedState.stash) at `position`"""
        target = draft = None
        try:
            target = self.cache.new_from_stashed(stashed["target"], position) if self.target_recurrent else None
            draft = self.draft_cache.new_from_stashed(stashed["draft"], position) if self.draft_recurrent else None
            return PairedState(target, draft)
        except BaseException:
            if draft is not None:
                draft.free()
            if target is not None:
                target.free()
            raise


    def iterate_hybrid_gen(self, draft_tokens: torch.Tensor | None):
        """
        Replace the round's draft with hybrid_draft_tokens tokens of suffix-match continuation when some job's
        current suffix matches at least that many tokens of its earlier context and the continuation agrees
        with the regular draft (the model's own short-range guess). The verification pass is rectangular, so
        when any row takes the long draft every row does: rows without a qualifying match keep their regular
        draft and are padded with whatever continuation their longest match offers, then with repeats of
        their last token, which verification rejects at the first mismatch. Returns the (rows, width) draft
        for iterate_gen, or the regular draft unchanged.
        """
        H = self.hybrid_draft_tokens
        jobs = [job for job in self.active_jobs if job.is_prefill_done()]
        if not jobs:
            return draft_tokens
        rows = []
        fire = False
        for i, job in enumerate(jobs):
            short = draft_tokens[i : i + 1] if draft_tokens is not None else None
            long, matched = job.get_hybrid_draft(H)
            # A qualifying row: a match of at least H tokens, H tokens following it, and no disagreement with
            # the regular drafter over the positions it covers
            ok = matched >= H and long.shape[-1] == H
            if ok and short is not None and short.shape[-1] and not torch.equal(long[:, :short.shape[-1]], short):
                ok = False
            fire = fire or ok
            rows.append((ok, short, long))
        if not fire:
            return draft_tokens

        out = torch.empty((len(jobs), H), dtype = torch.long)
        for i, (ok, short, long) in enumerate(rows):
            if ok:
                out[i] = long[0]
                continue
            parts = []
            if short is not None and short.shape[-1]:
                parts.append(short[0])
            if long.shape[-1]:
                parts.append(long[0])
            fill = torch.cat(parts) if parts else jobs[i].sequences[0].sequence_ids.torch()[0, -1:]
            fill = fill[:H]
            out[i, : fill.shape[-1]] = fill
            out[i, fill.shape[-1] :] = fill[-1]
        # The regular drafter's confidence record no longer describes the verified window
        self._draft_conf_round = None
        self.hybrid_rounds += 1
        return out


    def _staging(self, name, rows: int, width: int | None = None, dtype = torch.int32):
        """
        Reusable pinned staging buffer, keyed by name and row width, grown by rows on demand.
        """
        key = (name, width)
        buf = self.staging_buffers.get(key)
        if buf is None or buf.shape[0] < rows:
            alloc_rows = max(rows, 32)
            shape = (alloc_rows,) if width is None else (alloc_rows, width)
            buf = torch.zeros(shape, dtype = dtype, pin_memory = True)
            self.staging_buffers[key] = buf
        return buf[:rows]


    # ---------------------------------------------------------------------------------------
    #   Prompt ingestion, inline or on a second thread
    # ---------------------------------------------------------------------------------------

    def _prefill_round(self, results: list):
        """
        One round of prompt ingestion.

        As upstream, every job still ingesting its prompt gets one chunk and iterate() goes on
        to the decode step when they are done: a job that is generating gets one token per
        round of chunks. With prefill_thread the forward passes of a chunk run on a second
        thread instead (the yielded function of Job._prefill_co), with its own workspaces and
        CPU expert worker slots (util/lane.py), and this thread does not wait for them while a
        job is generating: it goes on to the decode step, and a later round records the chunk
        once it is done. So:

          - one chunk is under way at a time (self._side); with several jobs ingesting, they
            take turns at it;
          - a chunk and a decode step overlap only when they cannot meet inside a module
            (side_min_tokens, side_max_rows: see __init__). A shorter chunk (a prompt of a
            few tokens), or any chunk while the decode batch is larger than that, is waited
            for like an inline one;
          - everything that is not a forward pass stays on this thread: the plan of the chunk,
            its record (sequence and page positions, page hashes), the checkpoint, the results.
            The page table and the recurrent cache are never touched from the other thread;
          - a job whose chunk is under way is left alone until the chunk is recorded: no
            checkpoint (recurrent_checkpoint), no decode step (it is not done with its prompt
            until the record says so), and cancel() lets the chunk finish before it releases
            the job's pages and state;
          - the two threads issue on the same streams (util/lane.py says why), so kernels
            never overlap on a card and a job changes hands in stream order;
          - with no job generating there is nothing to do meanwhile and the round waits for
            its chunks, one per job as upstream. They still run on the prefill thread, so the
            workspaces of a prefill pass exist once.

        A failed chunk fails its job only, as an inline one (reap_failed_job).

        With prefill_pipeline and the layers on two cards, two chunks are under way instead of
        one: _pipe_round.
        """
        flying = self._flight.jobs if self._flight is not None else ()
        side = self.prefill_thread and self._side_plan()

        if self._pipe or (side and self.prefill_pipeline and self._side is None and self._pipe_plan()):
            self._pipe_round(results, flying, bool(side and self.prefill_pipeline))
            return

        if not side and self._side is None:
            for job in list(self.active_jobs):
                # A job whose pass is issued in part has pages ahead of its sequence position
                if flying and any(j is job for j in flying):
                    continue
                try:
                    job.prefill(results)
                except Exception as e:
                    self.reap_failed_job(job, e, results)
            return

        # Whether a chunk may be left to run beside this round's decode step: somebody is
        # generating, and the step is decode-sized for every module (side_max_rows). The width
        # of a verification pass is decided after this, by the draft step: counted at its
        # widest, the hybrid draft's long window included (max_draft_tokens)
        rows = self._decode_rows()
        beside = side and 0 < rows <= self.side_max_rows

        # The chunk under way: left to run beside this round's decode step, or recorded. A job
        # whose chunk is recorded here starts its next one in the next round, after the
        # checkpoint this round offers it
        served = None
        cur = self._side
        if cur is not None:
            if beside and not cur.future.done():
                cur.steps += 1
            else:
                served = cur.job
                self._side_collect(results, wait_all = not beside)

        waiting = []
        for job in list(self.active_jobs):
            if flying and any(j is job for j in flying):
                continue
            if job is served or (self._side is not None and job is self._side.job):
                continue
            if side and not job.is_prefill_done():
                waiting.append(job)
                continue
            # A job past its prompt (the call stamps its first prefill time), or any job when
            # the switch has just been turned off
            try:
                job.prefill(results)
            except Exception as e:
                self.reap_failed_job(job, e, results)

        # Whoever had a chunk longest ago goes first
        waiting.sort(key = lambda j: getattr(j, "_side_turn", 0))
        for job in waiting:
            if self._side is not None:
                break
            # (the pipeline has just been switched on: it takes over once no chunk is under way)
            if self.prefill_pipeline and self._pipe_ok is not False:
                break
            if not self._side_start(job, results):
                continue
            if beside and self._side.beside:
                self._side.steps += 1       # this round's decode step
            else:
                self._side_collect(results, wait_all = True)

        if self._side_stretch is not None and self._side is None \
                and all(job.is_prefill_done() for job in self.active_jobs):
            self._side_stretch_end()


    def _decode_rows(self) -> int:
        """Rows x tokens of the decode step the jobs past their prompt make together, the
        verification pass counted at its widest (max_draft_tokens: the hybrid draft's window is
        decided after the prefill round). With the cost-chosen draft window (and no hybrid draft)
        it is counted at the narrowest window the policy may take: when a chunk then runs beside
        the step, the draft step keeps the window within side_max_rows (_dw_begin)."""
        width = 1 + self.max_draft_tokens
        policy = self._dw_policy()
        if policy is not None and not self.hybrid_draft_tokens:
            width = 1 + policy.windows_for(self.num_draft_tokens)[0]
        rows = 0
        for job in self.active_jobs:
            if job.is_prefill_done():
                rows += len(job.sequences) * width
        return rows


    def _moe_hosts(self) -> list:
        """The CPU expert workers' hosts of the model and of the draft model (one per component)."""
        hosts = {}
        for m in (self.model, self.draft_model):
            for h in getattr(getattr(m, "config", None), "moe_cpu_hosts", {}).values():
                hosts[id(h)] = h
        return list(hosts.values())


    def _side_plan(self) -> bool:
        """Whether prompt chunks can go to a second thread; sets it up at the first call."""
        ok = self._side_ok
        if ok is None:
            why = None
            slots = None
            devices = []
            if self.model.loaded_tp:
                why = "tensor-parallel model"
            elif self.draft_model is not None and not self.mtp_draft:
                why = "draft model other than MTP"
            else:
                devices = cuda_devices(self.model, self.draft_model)
                if not devices:
                    why = "model not on a CUDA device"
            if why is None:
                why = self._side_promote_why()
            if why is None:
                for h in self._moe_hosts():
                    slots = h.split_lanes()
                    if slots is None:
                        why = "the CPU expert worker has a single slot (EXL3_MOE_CPU_SLOTS)"
                        break
                    # A chunk the worker's host would treat as a decode-size batch is not one
                    # that may run beside a decode step
                    self.side_min_tokens = max(
                        self.side_min_tokens, int(getattr(h, "stream_min_rows", 0) or 0) + 1)
            if why is None:
                self._side_lane = Lane(1, devices)
                self._side_pool = ThreadPoolExecutor(
                    max_workers = 1,
                    thread_name_prefix = "exl3-prefill",
                    initializer = self._side_lane.enter,
                )
                extra = "" if slots is None else \
                    f", CPU expert worker slots {slots[0]} for the decode and {slots[1]} for the prefill"
                print(f" -- Prefill thread: on{extra}", flush = True)
            else:
                print(f" !! Prefill thread not available: {why}", flush = True)
            self._side_why = why
            self._side_ok = ok = why is None
        return ok


    def _side_promote_why(self) -> str | None:
        """
        Why promote-on-miss rules the prefill thread out, None when no split layer has it.

        With spare GPU slots (EXL3_MOE_CPU_PROMOTE, on 1.6.x with the pinned arena) every pass
        runs a step of it at its first split layer (_split_promote_step), whatever its size: map
        flips of landed transfers, host waits on the previous step's snapshots, promotions into
        GPU slots and demotions into worker slots, on a side stream of their own. Its state is
        one object for every split layer and assumes one pass at a time: from two threads a
        promotion can take a GPU slot a chunk under way still reads, a demotion rewrite an arena
        block a chunk's worker job reads, and two steps interleave over the same state. Nothing
        of it is per lane, so the thread is not offered while any split layer has spares.
        """
        for m in (self.model, self.draft_model):
            ip = getattr(getattr(m, "config", None), "infer_params", None)
            for mod in getattr(ip, "moe_cpu_swap_modules", None) or ():
                if getattr(mod, "_split_spares", 0):
                    return "promote-on-miss (spare GPU slots on the split layers, EXL3_MOE_CPU_PROMOTE)"
        return None


    @property
    def prefill_thread_state(self) -> str:
        """
        Where prompt ingestion runs, for whoever holds the generator (a server that has its
        own answer to a prefill stalling a decode needs to know which of the two is at work):

          - "off": on the generator's thread, as upstream;
          - "pending": asked for, and the first round has yet to set the thread up;
          - "on": on the prefill thread (_prefill_round);
          - "unavailable": asked for, but not with this model (prefill_thread_reason says
            why); the generator behaves as upstream.
        """
        if not self.prefill_thread:
            return "off"
        if self._side_ok is None:
            return "pending"
        return "on" if self._side_ok else "unavailable"


    @property
    def prefill_thread_reason(self) -> str | None:
        """Why the prefill thread is not available, None when it is or is not known yet."""
        return self._side_why


    def _side_start(self, job, results: list) -> bool:
        """Plan the job's next chunk and hand its forward passes to the prefill thread. False
        when the round had nothing to compute for the job (pages found in the cache) or the
        plan failed (the job is released, as for an inline chunk)."""
        step = job._prefill_co(results)
        try:
            run = next(step)
        except StopIteration:
            return False
        except Exception as e:
            self.reap_failed_job(job, e, results)
            return False
        self._side_turn += 1
        job._side_turn = self._side_turn
        self._side = self._side_submit(job, step, run)
        return True


    def _side_submit(self, job, step, run) -> _SidePass:
        beside = getattr(run, "tokens", 0) >= self.side_min_tokens
        return _SidePass(job, step, self._side_pool.submit(self._side_work, run), beside)


    def _side_work(self, run) -> float:
        """On the prefill thread: a chunk's forward passes, waited for (a chunk that is done
        has run: its job may be released). Returns the seconds it took."""
        t0 = time.perf_counter()
        lane = self._side_lane
        with torch.inference_mode(), lane.passing():
            try:
                run()
            except BaseException:
                # What the failed pass had issued still runs: the job is released only after it
                try:
                    lane.synchronize()
                except Exception:
                    pass
                raise
            lane.synchronize()
        return time.perf_counter() - t0


    def _side_collect(self, results: list, wait_all: bool = False):
        """
        Wait for the chunk under way and record it. A job of several sequences has a chunk per
        sequence in one round: the next one is handed over at once and stays under way, unless
        wait_all (nothing is decoding meanwhile, or the thread is being left), which sees the
        job's round through.
        """
        while self._side is not None:
            cur, self._side = self._side, None
            stats = self.side_stats
            t0 = time.perf_counter()
            try:
                dt = cur.future.result()
                stats["t_waited"] += time.perf_counter() - t0
                stats["chunks"] += 1
                stats["t_chunks"] += dt
                if cur.steps:
                    stats["beside"] += 1
                    stats["decode_steps"] += cur.steps
                    stretch = self._side_stretch
                    if stretch is None:
                        stretch = self._side_stretch = [0, 0, 0.0]
                    stretch[0] += 1
                    stretch[1] += cur.steps
                    stretch[2] += dt
                run = cur.step.send(results)
            except StopIteration:
                return
            except Exception as e:
                stats["errors"] += 1
                cur.step.close()
                self.reap_failed_job(cur.job, e, results)
                return
            self._side = self._side_submit(cur.job, cur.step, run)
            if self._side.beside and not wait_all:
                self._side.steps += 1       # this round's decode step
                return


    def _side_drop(self):
        """Let the chunk under way finish and forget it (its job is being released: no kernel
        may be left writing into pages and states about to be handed to someone else). No-op
        with no chunk under way."""
        cur, self._side = self._side, None
        if cur is None:
            return
        try:
            cur.future.result()
        except Exception:
            pass
        cur.step.close()


    def _side_stretch_end(self):
        """Report the chunks ingested beside a decode since the last report (a server has no
        other trace of it; side_stats is for whoever holds the generator)."""
        stretch, self._side_stretch = self._side_stretch, None
        if stretch is not None:
            # (in flight on two cards the seconds are the stretch's, not the sum of the chunks')
            what = "Prefill thread" if len(stretch) == 3 else "Prefill pipeline"
            print(f" -- {what}: {stretch[0]} chunks in {stretch[2]:.1f} s, "
                  f"{stretch[1]} decode steps beside them", flush = True)


    # ---------------------------------------------------------------------------------------
    #   Prompt chunks in flight on two cards
    # ---------------------------------------------------------------------------------------

    def _pipe_round(self, results: list, flying, more: bool):
        """
        One round of prompt ingestion with two chunks under way (prefill_pipeline).

        The layers of the model are on two cards, and a chunk's pass works on one card at a time:
        while it is on the second, the first is free for the next chunk. Every layer keeps its
        own state (K/V pages, recurrent state) and needs that state after the previous chunk and
        the output of the layer before it for this one, so the next chunk may start on the first
        card as soon as this one has left it:

            first card    A(0)  A(1)  A(2) ...
            second card         B(0)  B(1) ...

        A chunk is issued in two halves (Job._prefill_co: halves), the first by the prefill
        thread and the second by a thread of its own, which also runs the MTP head's update
        unless the head is on the first card (_pipe_plan). Each thread takes the chunks in the
        order they were handed over, so A(k+1) follows A(k) and B(k+1) follows B(k) whatever the
        jobs; the second half waits for its first. What the round of one chunk at a time says
        (_prefill_round) holds here too, card by card: only forward passes leave this thread, a
        chunk and a decode step overlap only if they cannot meet inside a module, a job with a
        chunk under way gets no checkpoint and no decode step, cancel() lets its chunks finish
        the half they are in. What is new:

          - the chunks under way may be of two jobs, or two chunks of one job. The second is
            then planned ahead of the first's record (_pipe_ahead), and that is refused where
            a recurrent checkpoint is due at the position between the two: the state has to be
            whole there, and the first half of the next chunk would have moved the first
            card's layers past it. The chunk before a checkpoint is therefore recorded with
            nothing else of its job under way, and the next one starts in the round after the
            checkpoint. Chunks are recorded in the order they were started, so a job's
            sequence position (what the disk tier's write-behind reads) is the start of its
            oldest chunk under way until that chunk is recorded;
          - a chunk too short to run beside a decode step (side_min_tokens) waits, while
            somebody is generating, until the chunks before it are recorded, and is then
            waited for alone: the jobs that are generating stop for its passes only;
          - a decode batch too large for the decode paths (side_max_rows) gets one chunk per
            round, waited for, as without the pipeline;
          - with nobody generating the round waits for the oldest chunk and records it, having
            started the next one first: a round is still worth a chunk, and the cards are
            never left without one.

        more: whether new chunks are started (False when the switch has been turned off: the
        ones under way are seen through and the next round is a regular one).
        """
        pipe = self._pipe
        served = self._pipe_collect(results)

        waiting = []
        for job in list(self.active_jobs):
            if flying and any(j is job for j in flying):
                continue
            if not job.is_prefill_done():
                waiting.append(job)
                continue
            # A job past its prompt (the call stamps its first prefill time)
            try:
                job.prefill(results)
            except Exception as e:
                self.reap_failed_job(job, e, results)

        if more:
            self._pipe_fill(waiting, served, results)
            # Nobody is generating: the round is worth a chunk
            if not served and pipe and not self._decode_rows():
                served += self._pipe_collect(results, wait = True)
                self._pipe_fill([j for j in waiting if j in self.active_jobs], served, results)
        else:
            for p in pipe:
                if p.future is None:
                    self._pipe_submit(p)
            while pipe:
                self._pipe_collect(results, wait = True)

        # What the decode step that follows must not meet is waited for
        rows = self._decode_rows()
        under_way = [p for p in pipe if p.future is not None]
        if rows and under_way:
            if rows > self.side_max_rows or not all(p.beside for p in under_way):
                while pipe and pipe[0].future is not None:
                    self._pipe_collect(results, wait = True)
            else:
                self.side_stats["decode_steps"] += 1
                stretch = self._side_stretch
                if stretch is None:
                    stretch = self._side_stretch = [0, 0, 0.0, time.perf_counter()]
                stretch[1] += 1
                for p in under_way:
                    p.steps += 1

        if self._side_stretch is not None and not pipe \
                and all(job.is_prefill_done() for job in self.active_jobs):
            self._side_stretch_end()


    def _pipe_plan(self) -> bool:
        """Whether prompt chunks can be in flight on two cards; sets the second thread up at the
        first call, which is made with no chunk under way (the CPU expert worker's slots change
        hands). Only where the prefill thread is available (the caller has asked _side_plan)."""
        ok = self._pipe_ok
        if ok is None:
            why = None
            slots = None
            bounds = cuda_stage_bounds(self.model)
            if not self.mtp_draft:
                why = "only with the MTP head (a prompt chunk is then one forward pass, which can be issued in two halves)"
            elif len(bounds) != 1:
                why = f"the layers of the model are on {len(bounds) + 1} card(s), not on two"
            else:
                for h in self._moe_hosts():
                    slots = h.split_lanes(lanes = 3)
                    if slots is None or len(slots) < 3:
                        why = "the CPU expert worker has too few slots for a lane more (EXL3_MOE_CPU_SLOTS)"
                        break
            if why is None:
                stop = bounds[0]
                modules = [m for m, _, _ in self.model.fwd_modules]
                first = module_devices(modules[:stop])
                second = set(module_devices(modules[stop:]))
                # The MTP head's update of a chunk follows the chunk's second half. With the
                # head on the first card it takes its turn there, between two first halves, on
                # that card's thread: left to the second half's thread it ran while the next
                # chunk was on the first card, and the memory of the two passes added up there
                # (0.4 GiB on the machine, measured on the v1.5.4 base on 2026-10-11: a
                # 179k-token prompt no longer fitted). EXL3_PREFILL_PIPELINE_HEAD=second is
                # that behaviour, for telling the two apart
                head = set(cuda_devices(self.draft_model))
                self._pipe_head_first = bool(head) and head <= set(first) \
                    and os.environ.get("EXL3_PREFILL_PIPELINE_HEAD", "") != "second"
                if not self._pipe_head_first:
                    second |= head
                self._pipe_stop = stop
                # (the first half's lane is the prefill thread's own, entered by that thread;
                # what differs is the card each half waits for when it is done)
                self._pipe_lanes = (Lane(1, first), Lane(2, sorted(second)))
                self._pipe_pool = ThreadPoolExecutor(
                    max_workers = 1,
                    thread_name_prefix = "exl3-prefill2",
                    initializer = self._pipe_lanes[1].enter,
                )
                extra = ""
                if slots is not None:
                    extra = f", CPU expert worker slots {slots[0]} for the decode and {slots[1]} + {slots[2]} for the two halves"
                    if min(slots[1:]) < 2:
                        extra += " (EXL3_MOE_CPU_SLOTS=6 gives each half two)"
                if self._pipe_head_first:
                    extra += "; the MTP head's update takes its turn on the first card"
                print(f" -- Prefill pipeline: on, a chunk changes card at module {stop} of {len(modules)}{extra}", flush = True)
            else:
                print(f" !! Prefill pipeline not available: {why}", flush = True)
            self._pipe_why = why
            self._pipe_ok = ok = why is None
        return ok


    @property
    def prefill_pipeline_state(self) -> str:
        """
        Whether prompt chunks are in flight on two cards, for whoever holds the generator:

          - "off": not asked for (or the prefill thread has been switched off since);
          - "pending": asked for, and the first round has yet to set it up;
          - "on": two chunks may be under way (_pipe_round);
          - "unavailable": asked for, but not with this model (prefill_pipeline_reason says
            why); the prefill thread then ingests one chunk at a time, if it is available
            itself (the pipeline is not available where the thread is not).
        """
        if not self.prefill_pipeline or not self.prefill_thread:
            return "off"
        if self._side_ok is False:
            return "unavailable"
        if self._pipe_ok is None:
            return "pending"
        return "on" if self._pipe_ok else "unavailable"


    @property
    def prefill_pipeline_reason(self) -> str | None:
        """Why the pipeline is not available, None when it is or is not known yet."""
        if self.prefill_pipeline and self._side_ok is False:
            return self._side_why
        return self._pipe_why


    @staticmethod
    def _state_reads_position(state) -> bool:
        """Whether a module reads the position of this recurrent state while a pass runs (of
        either side of a PairedState); True for a state class that does not say
        (pass_reads_position), and for a job without a recurrent state."""
        if state is None:
            return True
        sides = getattr(state, "components", None)
        if sides is None:
            sides = [state]
        return not sides or any(getattr(s, "pass_reads_position", True) for s in sides)


    def _pipe_ahead(self, job, last) -> int | None:
        """
        The position from which the job's next chunk may be planned while `last`, its latest
        chunk, is under way; None when it has to wait for that chunk's record. Planning ahead
        is for the plain case, where nothing but the previous chunk stands between the job and
        the next one:

          - one sequence, no embeddings (their spans move the chunks' ends, and a chunk may
            read tokens again), not a job taken up again after tokens of its own;
          - a recurrent state whose position no module reads during a pass (the state class
            says so: pass_reads_position). A model without recurrent states may find pages of
            the cache at the head of a chunk, which the plan would have to skip;
          - no checkpoint at the position between the two chunks, whether the record of the
            first takes it (the prompt's last page) or the generator would after it
            (Job.is_checkpoint_boundary): see _pipe_round;
          - a record of `last` that leaves the block table alone: it gives the pages the chunk
            completes their content hashes, and a page that does not have it yet (allocated
            under a random one) may be swapped for another job's page of the same hash, the
            table rebuilt and the page's reference dropped (Job.hash_completed_page), while
            the next chunk's passes still read the table they were planned with.
        """
        run = last.run
        end = getattr(run, "end", None)
        if end is None or last.future is None or getattr(run, "checkpoint", True):
            return None
        if len(job.sequences) != 1 or job.embeddings or getattr(job, "new_tokens", 0) > 0:
            return None
        if self._state_reads_position(job.recurrent_state):
            return None
        seq = job.sequences[0]
        if end >= len(seq.sequence_ids) - 1:
            return None
        if job.is_checkpoint_boundary(at = end):
            return None
        hashes = getattr(seq, "page_hashes", None) or ()
        start = max(0, end - getattr(run, "tokens", 0)) // PAGE_SIZE
        for i in range(start, min(end // PAGE_SIZE, len(hashes))):
            if seq.allocated_pages[i].phash != hashes[i]:
                return None
        return end


    def _pipe_fill(self, waiting: list, served: list, results: list):
        """Start chunks while there is room for one: first the ones that were waiting for the
        chunks before them, then new ones, whoever had a chunk longest ago first."""
        pipe = self._pipe
        rows = self._decode_rows()
        for i, p in enumerate(pipe):
            if p.future is not None:
                continue
            alone = rows and not p.beside
            if alone and i:
                return
            self._pipe_submit(p)
            if alone:
                return

        depth = self.pipe_depth if rows <= self.side_max_rows else 1
        waiting.sort(key = lambda j: getattr(j, "_side_turn", 0))
        while len(pipe) < depth:
            entry = None
            for job in list(waiting):
                mine = [p for p in pipe if p.job is job]
                at = None
                if mine:
                    at = self._pipe_ahead(job, mine[-1])
                    if at is None:
                        continue
                elif any(j is job for j in served):
                    # Its next chunk starts in the next round, after the checkpoint this round
                    # offers it
                    continue
                step = job._prefill_co(results) if at is None else job._prefill_co(results, at)
                try:
                    run = next(step)
                except StopIteration:
                    # Nothing to compute (pages found in the cache)
                    waiting.remove(job)
                    continue
                except Exception as e:
                    waiting.remove(job)
                    self._pipe_fail(job, e, results)
                    continue
                if at is not None:
                    # The record of the chunk under way then leaves the state's position alone:
                    # this one's passes may have advanced it by then (Job._prefill_co)
                    mine[-1].run.followed = True
                self._side_turn += 1
                job._side_turn = self._side_turn
                beside = getattr(run, "tokens", 0) >= self.side_min_tokens
                entry = _SidePass(job, step, None, beside, run, at is not None)
                break
            if entry is None:
                return
            waiting.sort(key = lambda j: getattr(j, "_side_turn", 0))
            # Too short to run beside a decode step: once the others are recorded, and alone
            if pipe and rows and not entry.beside:
                pipe.append(entry)
                return
            pipe.append(entry)
            self._pipe_submit(entry)
            if rows and not entry.beside:
                return


    def _pipe_submit(self, entry: _SidePass):
        """
        Hand a chunk's pieces to their threads: the first half to the prefill thread, the rest
        of the pass to the second, and the MTP head's update after it, either on the second
        thread at once or, with the head on the first card (_pipe_plan), on the prefill thread
        when the half under way there is done. The chunk's future is that of the last piece.
        """
        first, rest, after = entry.run.halves
        entry.piped = any(p.future is not None for p in self._pipe if p is not entry)
        head_first = self._pipe_head_first
        done = entry.future = Future()
        fa = self._side_pool.submit(self._pipe_first, entry, first)
        fb = self._pipe_pool.submit(self._pipe_rest, entry, rest, fa, None if head_first else after)

        def chain(f):
            # (on the second thread, when its piece is done)
            try:
                state, dt = f.result()
                if state is None or not head_first:
                    done.set_result(dt)
                else:
                    self._side_pool.submit(self._pipe_after, entry, after, state, dt, done)
            except BaseException as e:
                done.set_exception(e)

        fb.add_done_callback(chain)


    def _pipe_first(self, entry: _SidePass, first):
        """On the prefill thread: the half of a chunk's pass on the first card, waited for.
        Returns what the second half needs and the seconds it took."""
        if entry.dropped:
            return None, 0.0
        t0 = time.perf_counter()
        lane = self._pipe_lanes[0]
        with torch.inference_mode(), lane.passing():
            try:
                state = first(self._pipe_stop)
            except BaseException:
                # What the failed half had issued still runs: the job is released only after it
                try:
                    lane.synchronize()
                except Exception:
                    pass
                raise
            lane.synchronize()
        return state, time.perf_counter() - t0


    def _pipe_rest(self, entry: _SidePass, rest, first, after):
        """On the second thread: the rest of a chunk's pass once its first half is done (a
        failed first half fails the chunk here), then the MTP head's update unless that is
        left to the first card's thread (after is None); waited for. Returns what the update
        needs, None for a chunk that was told to stop, and the seconds so far."""
        state, dt = first.result()
        if state is None or entry.dropped:
            return None, dt
        t0 = time.perf_counter()
        lane = self._pipe_lanes[1]
        with torch.inference_mode(), lane.passing():
            try:
                state = rest(state)
                if after is not None:
                    after(state)
            except BaseException:
                try:
                    lane.synchronize()
                except Exception:
                    pass
                raise
            lane.synchronize()
        return state, dt + time.perf_counter() - t0


    def _pipe_after(self, entry: _SidePass, after, state, dt: float, done: Future):
        """On the prefill thread, between two first halves: the MTP head's update of a chunk
        whose pass is done, waited for. Settles the chunk's future with the seconds of all
        its pieces."""
        try:
            if not entry.dropped:
                t0 = time.perf_counter()
                lane = self._pipe_lanes[0]
                with torch.inference_mode(), lane.passing():
                    try:
                        after(state)
                    except BaseException:
                        try:
                            lane.synchronize()
                        except Exception:
                            pass
                        raise
                    lane.synchronize()
                dt += time.perf_counter() - t0
            done.set_result(dt)
        except BaseException as e:
            done.set_exception(e)


    def _pipe_collect(self, results: list, wait: bool = False) -> list:
        """
        Record the chunks that are done, oldest first, and stop at the first one that is not
        (a chunk is recorded after the ones started before it); wait: for the oldest one first.
        Returns the jobs a chunk was recorded for. The next chunk of a job of several sequences
        takes its place at the end.
        """
        pipe = self._pipe
        stats = self.side_stats
        served = []
        while pipe and pipe[0].future is not None and (wait or pipe[0].future.done()):
            wait = False
            cur = pipe.pop(0)
            t0 = time.perf_counter()
            try:
                dt = cur.future.result()
                stats["t_waited"] += time.perf_counter() - t0
                stats["chunks"] += 1
                stats["t_chunks"] += dt
                stats["piped"] += cur.piped
                stats["ahead"] += cur.ahead
                if cur.steps:
                    stats["beside"] += 1
                    stretch = self._side_stretch
                    if stretch is not None and len(stretch) > 3:
                        stretch[0] += 1
                        stretch[2] = time.perf_counter() - stretch[3]
                served.append(cur.job)
                run = cur.step.send(results)
            except StopIteration:
                continue
            except Exception as e:
                stats["errors"] += 1
                cur.step.close()
                self._pipe_fail(cur.job, e, results)
                continue
            beside = getattr(run, "tokens", 0) >= self.side_min_tokens
            entry = _SidePass(cur.job, cur.step, None, beside, run)
            pipe.append(entry)
            if len(pipe) == 1 or (pipe[-2].future is not None and (beside or not self._decode_rows())):
                self._pipe_submit(entry)
        return served


    def _pipe_drop(self, job = None):
        """
        Stop the chunks under way of a job, or of every job, and forget them (the job is being
        released: no kernel may be left writing into pages and states about to be handed to
        someone else). Each is told to go no further than the half it is in, and that half is
        waited for. Nothing is recorded. No-op with none under way.
        """
        mine = [p for p in self._pipe if job is None or p.job is job]
        if not mine:
            return
        for p in mine:
            p.dropped = True
        for p in mine:
            if p.future is not None:
                try:
                    p.future.result()
                except Exception:
                    pass
            p.step.close()
        self._pipe[:] = [p for p in self._pipe if not any(p is q for q in mine)]


    def _pipe_fail(self, job, error, results: list):
        """A chunk of the job failed, at its plan or in its passes: the job is released as for
        an inline chunk (reap_failed_job), once its other chunks under way have stopped."""
        self._pipe_drop(job)
        self.reap_failed_job(job, error, results)


    # ---------------------------------------------------------------------------------------
    #   Staggered decode
    # ---------------------------------------------------------------------------------------

    def _stagger_init(self, stagger_decode: bool):
        """
        State of the staggered decode (iterate_staggered). _step_group narrows the draft and decode
        steps to one group of jobs; _flight is the group whose pass is issued up to the second card;
        _lanes holds the second set of pinned staging buffers (the first set is the generator's own);
        _stagger_backlog keeps results produced outside iterate(); _stagger_spec lists the first
        card's layer states that stage a speculative pass's scan inputs (_stagger_hold_spec).
        """
        self.stagger_decode = stagger_decode or os.environ.get("EXL3_STAGGER_DECODE", "0") not in ("", "0")
        self.stagger_interleave = 1
        # Jobs generating at the same time from which the decode is staggered. Every call issues
        # a whole pass, whatever the size of its group, and with the cards no longer the limit a
        # call costs what the host needs to issue it: two groups of one job take two calls for
        # the tokens one batch of two gives in a step of about the same length (measured on
        # Flash-Next, 2x RTX 5060 Ti, on the v1.5.4 base: -2% with 2 jobs, +9% with 3, +20% with 4)
        self.stagger_min_jobs = max(2, int(os.environ.get("EXL3_STAGGER_MIN_JOBS", "3")))
        self.stagger_stats = {
            "steps": 0, "launched": 0, "sampled": 0, "solo": 0, "drains": 0, "spec_holds": 0,
            "t_launch": 0.0, "t_issue": 0.0, "t_sample": 0.0,
        }
        self._step_group = None
        self._flight = None
        # Stretch of staggered calls in progress: [start, end of the last call, calls, most jobs
        # at once]; reported in one line when the decode goes back to regular steps
        self._stagger_run = None
        self._lanes = {}
        self._stagger_backlog = []
        self._stagger_split = None
        self._stagger_spec = None


    def _step_jobs(self) -> list:
        """Jobs the draft and decode steps work on: every active job, or one group of them
        while the decode is staggered."""
        group = self._step_group
        return self.active_jobs if group is None else group


    def _begin_forward(self, input_ids: torch.Tensor, params: dict):
        """The decode step's forward pass, not issued yet: whole (Model.forward) for a regular
        step, in pieces for a group of a staggered one."""
        if self._step_group is None:
            return DirectForward(self.model, input_ids, params)
        return ForwardCursor(self.model, input_ids, params)


    def _cache_superseded(self) -> bool:
        """Whether a newer generator has taken the cache over (iterate() then refuses to run)"""
        owner_serial = getattr(self.cache, "owner_serial", None)
        return owner_serial is not None and owner_serial != getattr(self, "cache_owner_serial", owner_serial)


    def _stagger_plan(self) -> int | None:
        """Index in model.fwd_modules of the first module on the second card, or None when the
        decode cannot be staggered: tensor-parallel load, not exactly two cards in a row, a
        drafter whose step is not group-aware (anything but MTP), or the hybrid draft (its long
        verification pass is decided for the whole batch at once, and the staggered step does not
        draft it)."""
        split = self._stagger_split
        if split is None:
            split = 0
            why = None
            if self.model.loaded_tp:
                why = "tensor-parallel model"
            elif self.ngram_match_min or (self.draft_model is not None and not self.mtp_draft):
                why = "drafter other than MTP"
            elif self.hybrid_draft_tokens:
                why = "hybrid draft"
            else:
                bounds = cuda_stage_bounds(self.model)
                if len(bounds) == 1:
                    split = bounds[0]
                else:
                    why = f"layers on {len(bounds) + 1} card(s), two needed"
            self._stagger_split = split
            if self.stagger_decode:
                # flush: under a service stdout is block-buffered, and these would surface hours
                # later or never
                if split:
                    print(f" -- Staggered decode: modules 0-{split - 1} on the first card, "
                          f"{split}-{len(self.model.fwd_modules) - 1} on the second", flush = True)
                else:
                    print(f" !! Staggered decode not available: {why}", flush = True)
        return split or None


    def _stagger_wanted(self) -> bool:
        """Whether this iteration's decode goes through iterate_staggered(): always while a pass
        is issued in part; otherwise with the switch on, stagger_min_jobs or more jobs generating
        and no job ingesting its prompt (an iteration that may run a prefill chunk stays one
        batch)."""
        if self._flight is not None:
            return True
        if not self.stagger_decode or self._stagger_plan() is None:
            return False
        ready = 0
        for job in self.active_jobs:
            if not job.is_prefill_done() or len(job.sequences) != 1:
                return False
            ready += 1
        return ready >= max(2, self.stagger_min_jobs)


    @staticmethod
    def _stagger_pick(ready: list, n_flying: int) -> list:
        """Jobs to issue now out of those at rest, given how many are in flight: all of them,
        minus what evens out the two groups (a job held back joins the other group at the next
        iteration). With nothing in flight this splits the jobs in two."""
        hold = max(0, (len(ready) - n_flying) // 2)
        return ready[:len(ready) - hold]


    def _new_lane(self) -> dict:
        lane = {"staging_buffers": {}, "sample_pinned": None, "_draft_conf_round": None}
        for name in ("draft_input_ids_pinned", "draft_ids_pinned"):
            buf = getattr(self, name, None)
            if buf is not None:
                lane[name] = torch.empty(buf.shape, dtype = buf.dtype, pin_memory = True)
        return lane


    @contextmanager
    def _stagger_scope(self, lane: int, jobs: list):
        """
        Run a piece of the draft or decode step for one group: the step functions see only its
        jobs, and stage through the pinned buffers of its lane. A group keeps its lane from the
        moment its inputs are staged until it is sampled, and the other group in the meantime
        uses the other lane: uploads from pinned memory are asynchronous, so a buffer rewritten
        for one group while the other group's pass has yet to reach the second card would hand
        that pass the wrong block table, cache lengths and token ids. Lane 0 is the generator's
        own set of buffers.
        """
        assert self._step_group is None, "nested staggered scope"
        own = None
        if lane:
            own = self._lanes.get(lane)
            if own is None:
                own = self._lanes[lane] = self._new_lane()
            saved = {k: getattr(self, k) for k in own}
            for k, v in own.items():
                setattr(self, k, v)
        self._step_group = jobs
        try:
            yield
        finally:
            self._step_group = None
            if own is not None:
                for k in own:
                    own[k] = getattr(self, k)
                for k, v in saved.items():
                    setattr(self, k, v)


    def _stagger_issue(self, late, early, split: int):
        """
        Issue the second-card half of one pass and the first-card half of another, a module of
        each in turn. The kernels land on the two cards' own streams, so the cards work at the
        same time. Taking turns is what keeps the CPU expert worker out of the way: it serves
        its ring strictly in the order the jobs were published, so two halves issued one after
        the other would make the first card wait at its first MoE layer until the second card
        is done with all of its own. The second-card module goes first in each turn.
        """
        n = max(1, int(self.stagger_interleave))
        if late is not None:
            late.wake()
        while True:
            more = False
            if late is not None and late.pos < late.end:
                late.run_to(late.pos + n)
                more = True
            if early is not None and early.pos < split:
                early.run_to(min(early.pos + n, split))
                more = True
            if not more:
                break


    def _stagger_spec_layers(self) -> list:
        """[layer state, buffers] for each layer state of the target cache on the first card that
        stages a speculative pass's scan inputs for the rewind after verification (GDNLayerState:
        GDN, KDA and Mamba2 layers). Found once; the buffers are made at the first hold."""
        layers = self._stagger_spec
        if layers is None:
            layers = []
            first = None
            for module, _, _ in self.model.fwd_modules[:self._stagger_split or 0]:
                dev = getattr(module, "device", None)
                if dev is not None and torch.device(dev).type == "cuda":
                    first = torch.device(dev).index
                    break
            if self.target_recurrent and first is not None:
                for l in self.cache.get_all_recurrent_layers().values():
                    if not all(hasattr(l, k) for k in ("spec_conv", "spec_beta", "spec_g", "spec_shape")):
                        continue
                    dev = getattr(l, "device", None)
                    if dev is not None and torch.device(dev).type == "cuda" and torch.device(dev).index == first:
                        layers.append([l, None])
            self._stagger_spec = layers
        return layers


    def _stagger_hold_spec(self, flight) -> list | None:
        """
        Copy out the scan inputs the speculative pass in flight staged on the first card, before
        the next group's first half writes its own over them.

        A recurrent layer's state (GDNLayerState) keeps the scan inputs of the last speculative
        pass through it, in buffers shared by every pass and addressed by batch row, and the
        rewind after verification replays them to settle a partial accept. A group in flight is
        settled when it is sampled, which is after the next group's first half has been through
        the same first-card layers (the second card's are settled before the next group gets
        there). The copies go on the first card's stream after the pass that wrote the rows and
        before the next group's; for the time of the group's sampling the layer states hand the
        copies to the rewind (_stagger_sample). One set of buffers serves every call: the rewind
        reading them is issued on the same stream before the next hold writes them again.

        Returns what to hand to _stagger_sample, None when the pass in flight is not speculative
        or the first card has no such layer.
        """
        if not flight.fwd.params.get("recurrent_history"):
            return None
        layers = self._stagger_spec_layers()
        if not layers:
            return None
        bsz, seqlen = flight.fwd.input_ids.shape
        shape = (bsz, seqlen, 1)
        n = bsz * seqlen
        held = []
        # One multi-tensor copy per dtype (the fused path takes a single dtype)
        copies = {}
        for entry in layers:
            l, bufs = entry
            assert l.spec_shape == shape, \
                f"staggered decode: a first-card layer staged a pass of shape {l.spec_shape}, the pass in flight is {shape}"
            if bufs is None:
                bufs = entry[1] = tuple(torch.empty_like(t) for t in (l.spec_conv, l.spec_beta, l.spec_g))
            for t, b in zip((l.spec_conv, l.spec_beta, l.spec_g), bufs):
                dsts, srcs = copies.setdefault(t.dtype, ([], []))
                dsts.append(b[:n])
                srcs.append(t[:n])
            held.append((l, bufs, shape))
        for dsts, srcs in copies.values():
            torch._foreach_copy_(dsts, srcs)
        self.stagger_stats["spec_holds"] += 1
        return held


    def _stagger_sample(self, flight, results: list, held: list | None = None):
        """Advance the states of a fully issued pass and resume its decode step to sample. With
        scan inputs held for it (_stagger_hold_spec), the first card's layer states present those
        to the rewind for the time of the sampling."""
        logits = flight.fwd.finish()
        swapped = []
        try:
            for l, bufs, shape in held or ():
                swapped.append((l, l.spec_conv, l.spec_beta, l.spec_g, l.spec_shape))
                l.spec_conv, l.spec_beta, l.spec_g = bufs
                l.spec_shape = shape
            with self._stagger_scope(flight.lane, flight.jobs):
                try:
                    flight.step.send((logits, results))
                except StopIteration:
                    pass
        finally:
            for l, conv, beta, g, shape in swapped:
                l.spec_conv, l.spec_beta, l.spec_g, l.spec_shape = conv, beta, g, shape
        self.stagger_stats["sampled"] += 1


    def _stagger_drain(self, results: list):
        """Finish the pass in flight on its own and sample it (a job of its group is being
        cancelled). No-op with nothing in flight. The pass is the last that went through the
        first card's layers, so its scan inputs are where its rewind looks for them."""
        flight, self._flight = self._flight, None
        if flight is None:
            return
        with torch.inference_mode():
            flight.fwd.wake()
            self._stagger_sample(flight, results)
        self.stagger_stats["drains"] += 1


    def _stagger_drop(self):
        """Abandon the pass in flight, if any (clear_queue(), close(), a cache taken over): nothing
        more of it is issued and its tokens are never sampled. The kernels of its first half run
        to their end, on the first card's stream ahead of anything issued after them."""
        flight, self._flight = self._flight, None
        if flight is not None:
            flight.step.close()
        self._stagger_backlog = []


    def iterate_staggered(self, results: list):
        """
        Decode step for a model whose layers are split over two GPUs, with the jobs in two
        groups half a pass apart.

        A regular step issues one pass for all the jobs: the first card runs its layers while
        the second one waits, then the other way round, and each card is idle for more than
        half of the step. Here one group is always "in flight": its pass has been issued up to
        the last module on the first card and it waits there. Each call

          1. drafts for the jobs at rest and stages their inputs (the new group);
          2. issues the second-card half of the group in flight together with the first-card
             half of the new group (_stagger_issue);
          3. samples the group that was in flight, which is now complete; its tokens are this
             call's results. The new group stays in flight for the next call.

        So every call returns the tokens of one group and the groups take turns. The groups
        are not fixed: whoever is at rest and ready is issued, minus what keeps the two groups
        even. With fewer than stagger_min_jobs jobs left, or when a job starts ingesting a prompt,
        nothing new is issued, the group in flight is completed alone and iterate() goes back to
        regular steps.

        What makes it safe to keep a pass waiting: its jobs' pages and recurrent states are
        referenced (nothing else can take them), its pinned staging buffers belong to its lane
        (_stagger_scope), its stream stack is owned (ForwardCursor.hold), and while two passes
        are issued in turn they are on different cards (the modules' workspaces are per card).
        The states' position advances only when the pass is complete, so a job in flight looks
        to the rest of iterate() as it did when it was issued; the state tensors of the
        first-card layers are already a pass ahead, which is why such a job is passed over at
        checkpoint time and its group is completed before any of its jobs is cancelled. What a
        speculative pass stages in the first card's recurrent layer states for its rewind is
        copied out before the next group writes there (_stagger_hold_spec). The disk tier's
        write-behind at the end of iterate() takes a job in flight like any other: it queues no
        page within two of the sequence position, and the pass under way writes past it.

        An exception in here is as fatal for the generator as one in a regular decode step
        (NonFiniteLogits on a job's first token, say): the group that was in flight is lost and
        the new one stays in flight, for a cancel() to complete or clear_queue() to drop.
        """
        stats = self.stagger_stats
        t0 = time.perf_counter()
        flight, self._flight = self._flight, None
        split = self._stagger_plan()

        # The new group: jobs at rest, unless the decode is going back to regular steps
        launch = []
        if self.stagger_decode and split is not None:
            ready = []
            for job in self.active_jobs:
                if not job.is_prefill_done() or len(job.sequences) != 1:
                    ready = []
                    break
                if flight is None or not any(j is job for j in flight.jobs):
                    ready.append(job)
            n_flying = len(flight.jobs) if flight is not None else 0
            if len(ready) + n_flying >= max(2, self.stagger_min_jobs):
                launch = self._stagger_pick(ready, n_flying)

        new = None
        if launch:
            lane = 1 - flight.lane if flight is not None else 0
            with self._stagger_scope(lane, launch):
                draft_tokens = self.iterate_draftmodel_mtp_gen(results) if self.mtp_draft else None
                step = self._iterate_gen_co(draft_tokens)
                try:
                    fwd = next(step)
                except StopIteration:
                    step = None
            if step is not None:
                new = _Flight(step, fwd, lane, launch)
                stats["launched"] += 1
        t1 = time.perf_counter()

        held = self._stagger_hold_spec(flight) if flight is not None and new is not None else None
        self._stagger_issue(
            flight.fwd if flight is not None else None,
            new.fwd if new is not None else None,
            split,
        )
        if new is not None:
            new.fwd.hold()
            self._flight = new
        t2 = time.perf_counter()

        if flight is not None:
            self._stagger_sample(flight, results, held)
            if new is None:
                stats["solo"] += 1
        t3 = time.perf_counter()

        stats["steps"] += 1
        stats["t_launch"] += t1 - t0
        stats["t_issue"] += t2 - t1
        stats["t_sample"] += t3 - t2

        run = self._stagger_run
        if run is None:
            run = self._stagger_run = [t0, t3, 0, 0]
        run[1] = t3
        run[2] += 1
        run[3] = max(run[3], (len(flight.jobs) if flight is not None else 0) + (len(new.jobs) if new is not None else 0))


    def _stagger_run_end(self):
        """Report the stretch of staggered calls that just ended. A server has no other trace
        of the decode having been staggered (stagger_stats is for whoever holds the generator)."""
        run, self._stagger_run = self._stagger_run, None
        if run is not None:
            print(f" -- Staggered decode: {run[2]} calls in {run[1] - run[0]:.1f} s, up to {run[3]} jobs", flush = True)


    def iterate_gen(self, results: list, draft_tokens: torch.Tensor | None = None):
        """
        One decode step for the jobs of this step: build the batch, run the forward pass, sample.
        """
        step = self._iterate_gen_co(draft_tokens)
        try:
            fwd = next(step)
        except StopIteration:
            return
        batch_logits = fwd.finish()
        # The pass holds the params, and with them the recurrent states: not kept during sampling
        fwd = None
        try:
            step.send((batch_logits, results))
        except StopIteration:
            pass


    def _iterate_gen_co(self, draft_tokens: torch.Tensor | None = None):
        """
        The decode step as a coroutine that stops once, at the forward pass: it yields the pass
        (something with finish(), not issued yet or issued in part) and is resumed with
        (logits, results) to sample from them. Returns without yielding when no job of the step is
        generating. Before the yield the jobs' inputs are staged (their page positions move, their
        sampling filters start); the results list is the one of the iterate() call that resumes
        the step, which need not be the call that started it (iterate_staggered).
        """
        # The cost policy's round of this step's window (iterate_draftmodel_mtp_gen made it, for
        # this step only)
        dw, self._dw_round = self._dw_round, None
        jobs = self._step_jobs()

        # Get shape of active batch
        # Only jobs that have finished prefill can participate in token generation. The maximum sequence length
        # determines how many cache pages the temporary block table needs for this iteration.
        batch_size = 0
        max_seq_len = 0
        for job in jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens)
            batch_size += len(job.sequences)
        if batch_size == 0:
            return
        if draft_tokens is not None:
            max_seq_len += draft_tokens.shape[-1]

        # Create block index table for batch
        # The model sees a compact batch, so build per-row mappings from logical page positions to physical cache
        # page indices, along with current cache lengths and optional MRoPE position offsets.
        # Block-table width is padded to a multiple of 16 pages so the pinned staging buffers
        # cover a few distinct widths only; the extra (zeroed) columns are never dereferenced
        # since the kernels bound their reads by the cache lengths
        max_pages_batch = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        max_pages_batch = (max_pages_batch + 15) // 16 * 16
        block_index = self._staging("block_index", batch_size, max_pages_batch)
        block_index.zero_()
        cache_seqlens = self._staging("cache_seqlens", batch_size)
        batch = 0
        use_offsets = "mrope" in self.model.caps
        positions = self._staging("positions", batch_size) if use_offsets else None
        for job in jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = seq.block_index_tensor[:, :max_pages_batch]
                block_index[batch:batch+1, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                if use_offsets:
                    positions[batch] = seq.kv_position + job.alt_rope_offset
                batch += 1

        # Collect input IDs and indexed embeddings
        # Jobs may contribute multiple sequence rows. logit_mapping records the slice of the model output that
        # belongs to each active job, while batch_jobs keeps the corresponding Job objects in compact-batch order.
        input_ids_list = []
        active_embeddings = []
        logit_mapping = []
        batch_jobs = []
        for job in jobs:
            logit_mapping.append(len(input_ids_list))
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list(draft_tokens, len(input_ids_list), add_to_cache = True)
            input_ids_list += job_ids
            batch_jobs.append(job)
            active_embeddings += job.embeddings
        logit_mapping.append(len(input_ids_list))
        ids_width = input_ids_list[0].shape[-1]
        if all(ids.shape[-1] == ids_width for ids in input_ids_list):
            ids_staging = self._staging("batch_ids", len(input_ids_list), ids_width, torch.long)
            batch_ids = torch.cat(input_ids_list, dim = 0, out = ids_staging)
        else:
            batch_ids = torch.cat(input_ids_list, dim = 0)

        # Collect recurrent states for batch
        # Recurrent models carry mutable state beside the K/V cache; pass one state object per compact batch job so
        # the model can advance or rewind it consistently with accepted draft tokens.
        batch_states = None
        if self.target_recurrent:
            batch_states = [job.recurrent_state.target for job in batch_jobs]

        # GPU workload is scheduled here, so launch any sampling filters that can run in the background
        for job in batch_jobs:
            if job.new_tokens < 0: continue
            assert len(job.filter_futures) == 0
            for f in job.filters:
                if not f.is_active: continue
                if f.use_background_worker():
                    job.filter_futures.append(self.filter_pool.submit(f.get_next_logit_mask))
                else:
                    job.filter_futures.append(None)

        # Get logit batch from model. Forward writes new K/V entries into the cache for the supplied positions and
        # returns logits for either one target token or a target-plus-draft verification window.
        params = {
            "attn_mode": "flash_attn",
            "block_table": block_index,
            "cache": self.cache,
            "cache_seqlens": cache_seqlens,
            "recurrent_states": batch_states,
            "indexed_embeddings": active_embeddings,
            "positions": positions,
            "recurrent_history": draft_tokens is not None,
            "pinned_staging": True,
        }
        if self.draft_model:
            params.update(self.draft_model.draft_verifier_params)
        batch_logits, results = yield self._begin_forward(batch_ids, params)

        # Keep only the fields needed below for draft-cache updates and drop the params dict so it cannot extend
        # references to recurrent state objects past this iteration.
        p_export_states = params.get("export_states")
        p_cache_seqlens = params.get("cache_seqlens")
        params = None

        # Background filters were launched before forward(); synchronous filters run now to overlap CPU work with
        # queued GPU execution as much as possible.
        for job in batch_jobs:
            if job.new_tokens < 0: continue
            assert len(job.logit_masks) == 0
            for f in job.filters:
                if not f.is_active: continue
                if f.use_background_worker():
                    job.logit_masks.append(None)
                else:
                    job.logit_masks.append(f.get_next_logit_mask())

        # Prepare past IDs (for sequences that need them for repetition penalty etc.) and logit masks where needed
        for job in batch_jobs:
            job.prepare_logit_mask()
            job.prepare_sampling_past_ids()
        self._note_readbacks(0, step = True)

        # TODO: Batch sampling
        # Pass to jobs to sample
        # Sampling is still job-by-job. Each accepted token updates job state, may emit a stream result, may finish
        # the job, or may request requeueing once the configured per-job token budget is reached.
        completed_jobs = []
        requeuing_jobs = []
        accepted_lengths = []
        rewound_jobs = set()
        j = 0

        # Reject the trailing draft positions after the last accepted token at index i: count them, roll back the
        # job's recurrent state and return cache pages to the accepted position
        def reject_remainder(job_, j_, i_, batch_states_):
            num_rejected = batch_logits.shape[1] - 1 - i_
            if num_rejected == 0:
                return 0
            job_.rejected_draft_tokens += num_rejected

            # Rewind recurrent states
            if batch_states_ is not None:
                batch_states_[j_].rewind(num_rejected)

            # Rewind cache position (draft model cache layout is always the same as target)
            for seq_ in job_.sequences:
                r = num_rejected
                while r:
                    pos = seq_.kv_position + r
                    page = seq_.allocated_pages[(pos - 1) // PAGE_SIZE]
                    rp = min(page.kv_position, r)
                    page.kv_position -= rp
                    r -= rp
            return num_rejected

        # Without a draft, each job samples exactly one independent token, so all sampler chains
        # are launched first and the results collected in a second pass: the first collect
        # absorbs the whole GPU tail and the remaining reads are cheap, instead of one full
        # launch-to-readback round trip per job. Sampler launch order (and with it each job's
        # RNG draw) matches the serial loop exactly. Per-job constraints (filters, penalties)
        # are unaffected because no job's token depends on another job's result.
        windows = None
        if draft_tokens is None and batch_logits.shape[1] == 1:
            # Single-token results stage through one pinned buffer and one synchronize, so the
            # batch pays one launch-to-readback round trip instead of one per job
            pinned = self.sample_pinned
            if pinned is None or pinned.shape[0] < batch_size:
                self.sample_pinned = pinned = torch.empty(
                    (max(batch_size, 32), 1), dtype = torch.long, pin_memory = True
                )
            launched = []
            for job, a, b in zip(jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a == b: continue
                token_logits = batch_logits[a:b, :, :]
                sampled = job.receive_logits(token_logits)
                next_token = sampled[0]
                if next_token.is_cuda and next_token.numel() == 1:
                    k = len(launched)
                    pinned[k:k + 1].copy_(next_token.view(1, 1), non_blocking = True)
                    sampled = (pinned[k:k + 1],) + sampled[1:]
                launched.append((job, token_logits, sampled))
            if launched:
                torch.cuda.synchronize(batch_logits.device)
                self._note_readbacks(1)
                # The sampled tokens are the next forward's inputs
                self.model.prefetch_tokens([t for _, _, s in launched for t in s[0].view(-1).tolist()])

            for job, token_logits, (next_token, next_k_tokens, next_k_probs, next_prob) in launched:
                eos, sampled_token, rq = job.receive_sample(
                    token_logits,
                    next_token,
                    next_k_tokens,
                    next_k_probs,
                    next_prob,
                    results,
                )
                # Single-token rewinds need no batch cleanup, but the post-batch draft carry update must still be
                # suppressed so it doesn't resurrect a stale MTP state
                if job.checkpoint_rewound:
                    job.checkpoint_rewound = False
                    rewound_jobs.add(id(job))

                # Requeue. Requeueing is only supported for single-sequence jobs because the replacement job
                # uses the full sequence generated so far as its next prompt.
                if len(job.sequences) == 1 and rq:
                    requeuing_jobs.append(job)
                elif eos:
                    completed_jobs.append(job)
                accepted_lengths.append(1)

        # With a draft, positions within a job are consumed strictly serially: acceptance of
        # position n gates position n+1, and constrained-decoding masks and sampling past IDs
        # must advance between positions.
        else:
            # With mtp_batched_verify the windows that can be sampled at once are, and read back
            # together: id(job) -> (sampled tokens, drafts matched from the first). The serial loop
            # below then takes their tokens from there, the rest as before
            windows = None
            if draft_tokens is not None and self.mtp_batched_verify and self.mtp_draft:
                windows = self._verify_windows(jobs, logit_mapping, batch_logits, draft_tokens)

            for idx, (job, a, b) in enumerate(zip(jobs, logit_mapping[:-1], logit_mapping[1:])):
                if a == b: continue
                job_logits = batch_logits[a:b, :, :]
                accepted_length = 1
                rejected = 0
                window = windows.get(id(job)) if windows else None

                for i in range(batch_logits.shape[1]):
                    token_logits = job_logits[:, i:i + 1, :]
                    if window is None:
                        next_token, next_k_tokens, next_k_probs, next_prob = job.receive_logits(
                            token_logits,
                        )
                        self._note_readbacks(1)
                    else:
                        next_token = torch.tensor([[window[0][i]]], dtype = torch.long)
                        next_k_tokens, next_k_probs, next_prob = None, None, None
                    eos, sampled_token, rq = job.receive_sample(
                        token_logits,
                        next_token,
                        next_k_tokens,
                        next_k_probs,
                        next_prob,
                        results,
                    )

                    # Requeue. Requeueing is only supported for single-sequence jobs because the replacement job
                    # uses the full sequence generated so far as its next prompt. Unconsumed draft positions must
                    # be rejected here so the recurrent state and page positions match the accepted sequence
                    # before the requeue stash.
                    if len(job.sequences) == 1 and rq:
                        if draft_tokens is not None:
                            rejected = reject_remainder(job, j, i, batch_states)
                        requeuing_jobs.append(job)
                        break

                    # EOS. Stop sampling this job immediately once a stop condition, filter condition or max
                    # token limit produces an EOS event. Draft positions from the EOS position onward were fed
                    # to verification but never resolved; count them as rejected so accepted + rejected equals
                    # the number of drafted positions (no cache/state rollback needed, the pages are released).
                    # The EOS stream event was already built inside receive_sample, so patch its counter too
                    if eos:
                        num_unresolved = batch_logits.shape[1] - 1 - i if draft_tokens is not None else 0
                        if num_unresolved:
                            job.rejected_draft_tokens += num_unresolved
                            for r_ in reversed(results):
                                if r_.get("eos") and r_["job"] is job:
                                    if "rejected_draft_tokens" in r_:
                                        r_["rejected_draft_tokens"] = job.rejected_draft_tokens
                                    break
                        completed_jobs.append(job)
                        break

                    # A banned-string match inside receive_sample just rewound the job, resetting cache pages and
                    # rolling back or replacing any recurrent state. The remaining logit positions extend a
                    # sequence that no longer exists, so abandon them; everything is already consistent, and the
                    # usual rejection rollback and state normalization must not run on top of the rewind.
                    if job.checkpoint_rewound:
                        job.checkpoint_rewound = False
                        rewound_jobs.add(id(job))
                        if draft_tokens is not None:
                            job.rejected_draft_tokens += batch_logits.shape[1] - 1 - i
                            rejected = -1
                        break

                    # Continue sampling from logit batch as long as result matches draft, unless hitting
                    # checkpoint mark. For speculative decoding, consume additional logits only while the
                    # sampled target token matches the draft token. A recurrent checkpoint boundary also stops
                    # draft acceptance so state can be stashed at an exact page boundary.
                    if draft_tokens is not None and i < batch_logits.shape[1] - 1:
                        cp_boundary = batch_states is not None and job.is_checkpoint_boundary()
                        if window is not None:
                            mismatch = i >= window[1]
                        else:
                            mismatch = draft_tokens[j, i].item() != sampled_token.item()
                        if mismatch or cp_boundary:
                            rejected = reject_remainder(job, j, i, batch_states)
                            break

                        # Accept draft token
                        else:
                            job.accepted_draft_tokens += 1
                            accepted_length += 1

                            # A window sampled at once had every position's mask and past IDs made
                            # ahead; receive_sample has fed the filters the token
                            if window is not None:
                                continue

                            # Advance filters
                            for f in job.filters:
                                if not f.is_active: continue
                                # Both lists take one entry per active filter (prepare_logit_mask indexes them
                                # in step)
                                if f.use_background_worker():
                                    job.filter_futures.append(self.filter_pool.submit(f.get_next_logit_mask))
                                    job.logit_masks.append(None)
                                else:
                                    job.logit_masks.append(f.get_next_logit_mask())
                                    job.filter_futures.append(None)

                            # Update masks and past IDs
                            job.prepare_logit_mask()
                            job.prepare_sampling_past_ids()

                # Make sure outgoing state is valid if entire draft was accepted
                if batch_states and draft_tokens is not None and rejected == 0:
                    batch_states[j].rewind(0)

                # Record per-round draft stats. Skip abandoned windows (banned-string rewind);
                # checkpoint-boundary truncations are rare enough to count as ordinary rejections.
                if draft_tokens is not None and rejected != -1:
                    if self.record_draft_stats:
                        job.draft_stats.append((
                            job.new_tokens,
                            draft_tokens.shape[-1],
                            accepted_length - 1,
                        ))

                # The last sampled token leads the next forward
                self.model.prefetch_tokens([sampled_token.item()])

                accepted_lengths.append(accepted_length)
                j += 1

        # The cost-chosen draft window: this step's time, the calibrator's labels, the counters
        if dw is not None or self.draft_window == "cost":
            dw = self._dw_end(dw, jobs, logit_mapping, accepted_lengths, rewound_jobs, windows, draft_tokens)

        # Update the draft-confidence calibration with this round's verification outcomes (any
        # draft-model mode)
        if self.draft_calibrator is not None and self._draft_conf_round is not None:
            st = self._draft_conf_round
            self._draft_conf_round = None
            cal = self.draft_calibrator
            row = 0
            for job, a_idx, b_idx in zip(jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a_idx == b_idx: continue
                accepted_length = accepted_lengths[row]
                conf = st["conf"][row]
                ids_full = st["ids"][row]
                row += 1
                if id(job) in rewound_jobs:
                    continue
                a = accepted_length - 1
                for i in range(a):
                    cal.add_label(conf[i].item(), True)
                if a < conf.shape[-1]:
                    seq = job.sequences[0]
                    n = len(seq.sequence_ids)
                    tail = seq.sequence_ids.torch_slice(n - 1, n).item()
                    cal.add_label(conf[a].item(), ids_full[a].item() == tail)
            cal.decay_step()

        # Settle a recurrent drafter's speculative pass: it consumed the round's input token and every
        # drafted token (last_history of them); the rejected drafts are undone, leaving its state at
        # the accepted prefix, i.e. at the new K/V position (or behind it after a long hybrid round)
        if self.draft_recurrent and draft_tokens is not None:
            accepted_idx = 0
            for job, a_idx, b_idx in zip(jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a_idx == b_idx:
                    continue
                accepted_drafts = accepted_lengths[accepted_idx] - 1
                accepted_idx += 1
                if id(job) in rewound_jobs or job.recurrent_state is None:
                    continue   # a checkpoint rewind replaced the pair
                ds = job.recurrent_state.draft
                ds.rewind(max(0, ds.last_history - accepted_drafts))

        # Accept new target_hidden if DFlash. DFlash draft models can update their cache from target-model hidden
        # states for the tokens accepted above, keeping draft and target cache layouts aligned.
        if self.dflash_draft:
            update_params = {
                "block_table": block_index,
                "cache_seqlens": p_cache_seqlens,
            }
            lengths = accepted_lengths
            if self.draft_recurrent:
                # A row whose pair a checkpoint rewind replaced gets its window from the replay
                # prefill instead; its length 0 skips the write (and the rollback)
                drafting = [job for job, a, b in zip(jobs, logit_mapping[:-1], logit_mapping[1:]) if a != b]
                update_params["recurrent_states"] = [job.recurrent_state.draft for job in drafting]
                lengths = [0 if id(job) in rewound_jobs or job.recurrent_state is None else n
                           for job, n in zip(drafting, accepted_lengths)]
            self.draft_model.update_kv_from_target(
                target_hidden = p_export_states,
                cache = self.draft_cache,
                lengths = lengths,
                params = update_params,
            )

        # Accept new target_hidden if MTP. MTP draft models can update their cache from target-model hidden
        # states for the tokens accepted above, keeping draft and target cache layouts aligned.
        if self.mtp_draft and self.mtp_batched_update:
            self._mtp_update(
                jobs, logit_mapping, batch_ids, block_index, p_cache_seqlens, p_export_states[-1],
                accepted_lengths, rewound_jobs, completed_jobs + requeuing_jobs, draft_tokens, dw,
            )
        elif self.mtp_draft:
            target_hidden = p_export_states[-1]
            accepted_idx = 0
            for job, a_idx, b_idx in zip(jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a_idx == b_idx:
                    continue
                accepted_length = accepted_lengths[accepted_idx]
                accepted_idx += 1

                # A banned-string rewind invalidated this job's carry; leave it unset so drafting pauses until the
                # next target forward provides a fresh one, and don't propagate hidden states from the abandoned
                # window into the draft cache
                if id(job) in rewound_jobs:
                    continue

                # Position K was drafted from the last target state already. Replace accepted
                # speculative positions K+1..K+A-1 with the corresponding target-state inputs.
                if accepted_length > 1:
                    self.draft_model.prefill(
                        batch_ids[a_idx:b_idx, 1:accepted_length],
                        {
                            "attn_mode": "flash_attn",
                            "block_table": block_index[a_idx:b_idx],
                            "cache": self.draft_cache,
                            "cache_seqlens": p_cache_seqlens[a_idx:b_idx] + 1,
                            "target_hidden": target_hidden[a_idx:b_idx, :accepted_length - 1, :],
                        },
                    )

                # The next unprocessed token is paired with the preceding target hidden state.
                job.mtp_last_hidden = target_hidden[
                    a_idx:b_idx, accepted_length - 1:accepted_length, :
                ].clone()

        # Release pages for completed jobs. Finished and requeued jobs no longer need their active page references.
        # Requeued recurrent jobs may stash the last checkpoint first so the next queued job can resume from cached
        # recurrent state.
        num_jobs = self.num_remaining_jobs()
        for job in completed_jobs + requeuing_jobs:
            if job in requeuing_jobs and self.recurrent_cache is not None:
                job.maybe_stash_recurrent(self.recurrent_cache, PAGE_SIZE)
            job.deallocate_pages()
            self.active_jobs.remove(job)

        # Requeue jobs. Puts replacement jobs at the front so long generations continue promptly after they yield
        # their cache pages.
        for job in requeuing_jobs:
            rq_job = job.prepare_for_requeue()
            self.pending_jobs.insert(0, rq_job)

        # Defrag. Physical page indices can only be compacted when no active block tables are using them.
        if num_jobs and not self.num_remaining_jobs():
            self.on_queue_drained()


    def _mtp_update(self, jobs, logit_mapping, batch_ids, block_index, cache_seqlens, target_hidden,
                    accepted_lengths, rewound_jobs, done_jobs, draft_tokens, dw = None):
        """
        The MTP head's update after a verification (mtp_batched_update), in place of a head pass per
        job that accepted a draft (positions K+1..K+A-1, the accepted drafts with the target's hidden
        states) and the next iteration's draft pass (position K+A, the token sampled after them, with
        the target's state at K+A-1).

        A job that goes on generating is folded (_mtp_fold): one pass of the head over all of them
        writes positions K+1..K+W+1 (W the window) and drafts from position K+A. The rest of the
        jobs that accepted a draft (a job that ended or was requeued, whose pages may end at K+W, or
        a window that is not the head's own: a hybrid round, the calibrated draft, which drafts step
        by step) get one batched pass over K+1..K+A-1 (_mtp_update_pass). A job a banned-string
        rewind abandoned gets neither (its carry is gone; drafting pauses until the next target pass,
        as without the switch).

        Why writing past a job's accepted tokens is harmless: the head's cache has the target's page
        layout, attention reads stop at each row's position, and a job's position only moves forward
        from K+A through positions written again before they are read (the next pass that reaches a
        position writes it first, as the target's verification pass does over rejected drafts). The
        positions are on pages the job owns: up to K+W the verification pass wrote them; past it a
        folded job still generates, so its next verification will write up to K+A+W (_fold_fits
        checks the page table all the same).

        With the window chosen by the cost policy (dw: the step verified the window it chose, of any
        width up to num_draft_tokens, 0 included) the jobs are folded too: the fold's content does
        not depend on the window (position K+i pairs the token there with the target's state at
        K+i-1, whoever drafted it), only its width does. A window of 0 is folded over K+1: the
        head's cache at K was written this step, by the draft pass or by the last step's fold
        whose first draft the step took.
        """
        width = batch_ids.shape[-1]
        if dw is not None:
            fold = True
        else:
            fold = draft_tokens is not None and draft_tokens.shape[-1] == self.num_draft_tokens and \
                self.draft_calibrator is None
        done = {id(job) for job in done_jobs}
        folds, updates = [], []
        row = 0
        for job, a, b in zip(jobs, logit_mapping[:-1], logit_mapping[1:]):
            if a == b:
                continue
            n = accepted_lengths[row]
            row += 1
            if id(job) in rewound_jobs:
                continue
            # As without the switch: what the regular draft pass starts from, should this job need one
            job.mtp_last_hidden = target_hidden[a:b, n - 1:n, :].clone()
            kv = int(cache_seqlens[a])
            if fold and id(job) not in done and job.sequences[0].kv_position == kv + n and \
                    self._fold_fits(job, kv + width):
                folds.append((job, a, n))
            elif n > 1:
                updates.append((job, a, n))
        if updates:
            self._mtp_update_pass(updates, batch_ids, block_index, cache_seqlens, target_hidden)
        if folds:
            self._mtp_fold(folds, batch_ids, cache_seqlens, target_hidden)


    @staticmethod
    def _fold_fits(job, last: int) -> bool:
        """Whether position `last` of a job is on a page it owns, in its block table"""
        seq = job.sequences[0]
        page = last // PAGE_SIZE
        return page < len(seq.allocated_pages) and page < seq.block_index_tensor.shape[-1]


    def _mtp_update_pass(self, rows, batch_ids, block_index, cache_seqlens, target_hidden):
        """One pass of the head over positions K+1..K+S of each row (S the most drafts a row
        accepted), the drafts with the target's hidden states: what a head pass per job writes, and
        past a row's own accepted drafts positions the verification pass wrote too (see _mtp_update)."""
        span = max(n for _, _, n in rows) - 1
        k = len(rows)
        ids = self._staging("mtp_update_ids", k, span, torch.long)
        seqlens = self._staging("mtp_update_seqlens", k)
        table = self._staging("mtp_update_table", k, block_index.shape[-1])
        hidden = []
        for r, (job, a, n) in enumerate(rows):
            ids[r].copy_(batch_ids[a, 1:1 + span])
            seqlens[r] = cache_seqlens[a] + 1
            table[r].copy_(block_index[a])
            hidden.append(target_hidden[a:a + 1, :span, :])
        self.draft_model.prefill(ids, {
            "attn_mode": "flash_attn",
            "block_table": table,
            "cache": self.draft_cache,
            "cache_seqlens": seqlens,
            "target_hidden": torch.cat(hidden, dim = 0),
        })
        self.mtp_update_stats["passes"] += 1
        self.mtp_update_stats["rows"] += k


    def _mtp_fold(self, rows, batch_ids, cache_seqlens, target_hidden):
        """
        One pass of the head over positions K+1..K+W+1 of each job that goes on generating, and its
        next draft (mtp_batched_update, see _mtp_update). Position K+i pairs the token at K+i with
        the target's hidden state at K+i-1: the accepted drafts for i < A, the token sampled after
        them for i = A, the rejected drafts (and a filler) past it. The head's output at K+A is
        what the next draft pass would compute, so the job's first draft is taken from there and
        kept on the job (mtp_fold) for the next draft step, with the head's state at that position
        for the draft tokens after it. Nothing is read back here: the next draft step does it.
        """
        width = batch_ids.shape[-1]
        k = len(rows)
        pages = max((int(cache_seqlens[a]) + width) // PAGE_SIZE + 1 for _, a, _ in rows)
        pages = (pages + 15) // 16 * 16
        ids = self._staging("mtp_fold_ids", k, width, torch.long)
        seqlens = self._staging("mtp_fold_seqlens", k)
        table = self._staging("mtp_fold_table", k, pages)
        table.zero_()
        hidden = []
        for r, (job, a, n) in enumerate(rows):
            seq = job.sequences[0]
            ids[r, :width - 1].copy_(batch_ids[a, 1:])
            ids[r, width - 1] = batch_ids[a, width - 1]
            last = len(seq.sequence_ids)
            ids[r, n - 1] = int(seq.sequence_ids.torch_slice(last - 1, last).item())
            seqlens[r] = cache_seqlens[a] + 1
            src = seq.block_index_tensor[:, :pages]
            table[r, :src.shape[-1]].copy_(src[0])
            hidden.append(target_hidden[a:a + 1, :width, :])
        params = {
            "target_hidden": torch.cat(hidden, dim = 0),
            "attn_mode": "flash_attn",
            "block_table": table,
            "cache": self.draft_cache,
            "cache_seqlens": seqlens,
            "draft_step": 0,
        }
        # The cost policy reads the first draft's confidence with the draft (_mtp_take_folds)
        with_conf = self._dw_policy() is not None
        if with_conf:
            params["export_draft_conf"] = True
        state = self.draft_model.forward(ids, params)
        picked = torch.cat([state[r:r + 1, n - 1:n, :] for r, (_, _, n) in enumerate(rows)], dim = 0)
        lm_head = self.model.modules[self.model.logit_layer_idx]
        picked = lm_head.prepare_for_device(picked, params)
        drafted = self.draft_model.sample_from_state(picked, params)
        conf = params.get("draft_conf") if with_conf else None
        deeper = self.num_draft_tokens > 1
        for r, (job, a, n) in enumerate(rows):
            job.mtp_fold = (
                int(cache_seqlens[a]) + n, drafted[r:r + 1], picked[r:r + 1] if deeper else None,
                conf[r:r + 1] if conf is not None else None,
            )
        stats = self.mtp_update_stats
        stats["passes"] += 1
        stats["rows"] += k
        stats["folds"] += k


    def _mtp_take_folds(self, jobs, batch_size, batch_ids, cache_seqlens, temp_hidden, with_conf = False):
        """
        The draft step's first draft token from the last step's fold (mtp_batched_update): when
        every job of the step has one made at the position it stands at, the drafts are read back
        (one copy) into the draft and into the inputs of the head's next draft token, the positions
        move on one, and the head's states at those positions are what it drafts on from. Returns
        (the first draft token still to make, the hidden states to make it from, the drafts'
        confidences as host values when with_conf and the folds made them, else None). Every job's
        fold is used or dropped here: a fold is for one draft step. The confidence-target draft
        needs the first draft's confidence and takes no fold; the cost policy (with_conf) gets it
        with the drafts, in the same readback.
        """
        ready = [job for job in jobs if job.is_prefill_done()]
        folds = [getattr(job, "mtp_fold", None) for job in ready]
        for job in ready:
            job.mtp_fold = None
        if (self.draft_calibrator is not None and not with_conf) or not folds or not all(
            f is not None and f[0] == job.sequences[0].kv_position for f, job in zip(folds, ready)
        ):
            if any(f is not None for f in folds):
                self.mtp_update_stats["fold_misses"] += 1
            return 0, temp_hidden, None
        conf = None
        if with_conf and all(len(f) > 3 and f[3] is not None for f in folds):
            conf = self._dw_stage_conf(torch.cat([f[3] for f in folds], dim = 0), batch_size)
        self.draft_ids_pinned[:batch_size, 0:1].copy_(torch.cat([f[1] for f in folds], dim = 0))
        self._note_readbacks(1)
        batch_ids.copy_(self.draft_ids_pinned[:batch_size, 0:1])
        cache_seqlens += 1
        if self.num_draft_tokens > 1:
            temp_hidden = torch.cat([f[2] for f in folds], dim = 0)
        self.mtp_update_stats["fold_drafts"] += 1
        return 1, temp_hidden, (conf.view(-1).tolist() if conf is not None else None)


    def _note_readbacks(self, n: int, step: bool = False):
        """Count readbacks of drafted or sampled tokens (readback_stats), and a decode step that
        samples. A Generator made without __init__ has no counters"""
        stats = getattr(self, "readback_stats", None)
        if stats is not None:
            stats["readbacks"] += n
            if step:
                stats["steps"] += 1


    @staticmethod
    def _host_wait(device):
        """
        Wait on the host for what has been issued so far on the current stream of `device`, the
        stream the decode step's sampling and its copies go to: not for the whole device, so a
        copy stream of the CPU expert worker's host or anything else on another stream of the card
        is not waited for (a chunk on the prefill thread shares the stream, and is waited for up to
        here, as by any readback). Nothing to wait for off CUDA.
        """
        device = torch.device(device)
        if device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            event.synchronize()


    def _verify_windows(self, jobs, logit_mapping, batch_logits, draft_tokens) -> dict | None:
        """
        Speculative windows sampled at once (mtp_batched_verify). The serial path samples a job's
        window a position at a time, each a sampler call and a readback, the next position only
        once the one before matched its draft. Here every job that can (Job.window_sampling_why)
        has its window sampled in one call (Job.sample_window: position r as if drafts[:r] were
        accepted), the samples are compared with the drafts on the device and the number of drafts
        matched from the first is counted there, and the tokens and counts of every such job come
        back in one copy and one wait. The serial loop then accepts from them exactly as from its
        own samples: a position whose draft did not match ends the window, and so do what it checks
        on the host, a recurrent checkpoint boundary (the tail checkpoint's too), a stop condition, a
        requeue (positions sampled past where the window ends are never used). Jobs that cannot take
        the serial path in the same step, after this wait.

        Returns {id(job): (tokens of the window, drafts matched)}, or None when no job's window was
        sampled so.
        """
        stats = self.mtp_verify_stats
        width = draft_tokens.shape[-1]
        rows = width + 1
        if width < 1:
            return None

        taken = []
        samples = []
        j = 0
        for job, a, b in zip(jobs, logit_mapping[:-1], logit_mapping[1:]):
            if a == b:
                continue
            row = j
            j += 1
            why_fn = getattr(job, "window_sampling_why", None)
            why = "job" if why_fn is None else why_fn()
            if why is None:
                sampled = job.sample_window(batch_logits[a:b, :, :], draft_tokens[row, :width].tolist())
                if sampled is None:
                    why = "filter_error"
            if why is not None:
                stats["serial"] += 1
                stats["fallbacks"][why] = stats["fallbacks"].get(why, 0) + 1
                continue
            taken.append((job, row))
            samples.append(sampled.view(1, rows))
        if not taken:
            return None

        # The drafts of the jobs taken, uploaded from pinned staging; samples and matched counts
        # come back through one pinned buffer, in one copy
        n = len(taken)
        device = batch_logits.device
        drafts = self._staging("verify_drafts", n, width, torch.long)
        for k, (job, row) in enumerate(taken):
            drafts[k].copy_(draft_tokens[row, :width])
        sampled = torch.cat(samples, dim = 0)
        drafted = drafts.to(device, non_blocking = True)
        matched = (sampled[:, :width] == drafted).long().cumprod(dim = 1).sum(dim = 1, keepdim = True)
        out = self._staging("verify_out", n, rows + 1, torch.long)
        out.copy_(torch.cat((sampled, matched), dim = 1), non_blocking = True)
        self._host_wait(device)
        self._note_readbacks(1)
        stats["windows"] += n
        stats["readbacks"] += 1

        return {id(job): (r[:rows], r[rows]) for (job, _), r in zip(taken, out.tolist())}


    def reap_failed_job(self, job, error, results: list):
        """
        Contain a per-job failure so the generator stays usable for other jobs: release the
        failed job's pages and recurrent state, drop it from the active set, and append an
        error result carrying the standard serial/stage/eos fields. The async wrapper
        delivers the raw exception to that job's consumer only (AsyncJob.__aiter__ re-raises
        queued exceptions) and the sync generate() API raises it, instead of the failure
        escaping to _run_iteration, which latches AsyncGenerator.error permanently and kills
        the iteration task for every job.
        """
        try:
            job.deallocate_pages()
        except Exception:
            logger.error(
                "reap_failed_job: deallocate_pages() raised while cleaning up failed "
                "job %s; pages or recurrent state may be stranded",
                job,
                exc_info = True,
            )
        if job in self.active_jobs:
            self.active_jobs.remove(job)
        results.append({
            "job": job,
            "serial": job.serial_number,
            "stage": "error",
            "eos": True,
            "error": error,
        })


    def iterate_start_jobs(self, results: list):
        """
        Move pending jobs into the active set when batch and cache capacity allow.

        Jobs are considered in queue order, but a job can be skipped temporarily if it would exceed max_batch_size
        or needs more fresh pages than are currently unreferenced. Later jobs may start if they fit, which improves
        utilization, but each skipped job accumulates a skip count; once any skipped job reaches max_skips, startup
        stops for this iteration to preserve approximate queue fairness. Started jobs allocate their cache pages
        immediately and emit a "started" event.
        """

        # Get current max batch
        current_max_batch = 0
        for job in self.active_jobs:
            current_max_batch += len(job.sequences)

        # Start new jobs if possible
        if (self.pagetable.num_unreferenced_pages() and
            len(self.pending_jobs) and
            current_max_batch < self.max_batch_size):

            skipped_jobs = []
            for job in self.pending_jobs.copy():

                if (len(job.sequences) + current_max_batch > self.max_batch_size or
                        job.current_new_pages_required() > self.pagetable.num_unreferenced_pages()):
                    skipped_jobs.append(job)
                    continue

                # Make sure the job we're about to add doesn't skip a job that's been skipped too many times
                for j in skipped_jobs:
                    if j.skips >= j.max_skips:
                        return
                for j in skipped_jobs:
                    j.skips += 1

                # Add job to active list
                self.pending_jobs.remove(job)
                self.active_jobs.append(job)
                job.activate()

                # Allocate pages for job
                try:
                    job.allocate_pages()
                except Exception as e:
                    self.reap_failed_job(job, e, results)
                    continue
                current_max_batch += len(job.sequences)

                r = {
                    "job": job,
                    "stage": "started",
                    "eos": False,
                    "serial": job.serial_number,
                }
                if job.identifier is not None:
                    r.update({ "identifier": job.identifier })
                results.append(r)


    def generate(
        self,
        prompt: list[tuple] | list[str] | tuple | str,
        max_new_tokens: int | None = None,
        min_new_tokens: int = 0,
        seed: int | None = None,
        sampler: Sampler | list[Sampler] | None = None,
        token_healing: bool = False,
        encode_special_tokens: bool = False,
        decode_special_tokens: bool = False,
        stop_conditions: list[int | str] | None = None,
        add_bos: bool = False,
        abort_event: threading.Event | None = None,
        completion_only: bool = False,
        filters: list[list[Filter]] | list[Filter] | None = None,
        return_last_results: bool = False,
        embeddings: list[MMEmbedding] | list[list[MMEmbedding]] | None = None,
        max_rq_tokens: int | None = None,
        stop_on_loop: tuple[int, int] = None,
        **kwargs
    ):
        """
        This is a utility function for easily generating one or more completions from one or more prompt strings. For
        more versatility and streaming functionality, use the async wrapper or create Job objects directly, enqueue()
        them and call iterate() to receive one token at a time

        :param prompt:
            If this argument is a list, its length determines the batch size, and the output will be a list of strings
            as well. Each prompt is either a string or a pair of prompts for CFG sampling. If CFG is used, sampler
            settings must contain cfg_scale.

        :param sampler:
            Sampler stack settings for all prompts in batch or list of samplers for each prompt.

        :param max_new_tokens:
            Max number of tokens to generate.

        :param min_new_tokens:
            Minimum number of tokens to generate before stop tokens become active. Until this number have been
            sampled, stop tokens are suppressed but stop strings will still end response.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param stop_conditions:
            List of strings and/or token IDs that will end generation. The stop condition is not included
            in the output.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :param filters:
            (List of) list of Filters to apply during generation. Each prompt in a batch needs
            its own filter list, or a value of None to disable filters for individual prompts.

        :param return_last_results:
            If True, returns the last results dict for each job

        :param embeddings:
            Optional list of MMEmbeddings to use for, or list of lists for batched generation

        :param max_rq_tokens:
            Maximum number of tokens before job is requeued. Rounded to nearest page boundary. This limits how
            many new pages are allocated in the cache for the job in any one round and allows a single job to use
            the full cache size without limiting concurrency for other jobs.

        :param stop_on_loop:
            Tuple of (window_size: int, min_reps: int), or None. If enabled, generation will end if the last
            window_size tokens sampled make up >= min_reps consecutive instances of a looping string.

        :return:
            Completion(s): (str or list[str] depending on the type of the input prompt argument)
            Optionally, last results: (dict or list[dict] depending on the type of the input prompt argument)
        """

        order = {}
        if isinstance(prompt, list):
            prompts = prompt
        else:
            prompts = [prompt]
            filters = [filters]
            embeddings = [embeddings]

        if not filters:
            filters = [None] * len(prompts)
        else:
            assert len(filters) == len(prompts) and \
                all((f is None or isinstance(f, list)) for f in filters), \
                "If using filters, must provide one filter list (or None-value) per prompt."

        if not embeddings:
            embeddings = [None] * len(prompts)
        else:
            assert len(embeddings) == len(prompts) and all((isinstance(f, list) or not f) for f in embeddings), \
                "Must provide one list of embeddings per prompt."

        prompts = prompt if isinstance(prompt, list) else [prompt]
        batch_size = len(prompts)
        for idx, p in enumerate(prompts):
            if isinstance(p, str):
                input_ids = self.tokenizer.encode(
                    p,
                    encode_special_tokens = encode_special_tokens,
                    add_bos = add_bos,
                    embeddings = embeddings[idx]
                )
            elif isinstance(p, tuple):
                input_ids = [self.tokenizer.encode(
                    p_,
                    encode_special_tokens = encode_special_tokens,
                    add_bos = add_bos,
                    embeddings = embeddings[idx]
                ) for p_ in p]
            else:
                assert False, "Unexpected type in prompt"

            if sampler is None or isinstance(sampler, Sampler):
                p_sampler = sampler
            elif isinstance(sampler, list):
                assert len(sampler) == len(prompts)
                p_sampler = sampler[idx]
            else:
                assert False, "Unexpected sampler type"

            job = Job(
                input_ids = input_ids,
                max_new_tokens = max_new_tokens,
                min_new_tokens = min_new_tokens,
                seed = seed,
                stop_conditions = stop_conditions,
                sampler = p_sampler,
                filters = filters[idx] or [],
                token_healing = token_healing,
                decode_special_tokens = decode_special_tokens,
                embeddings = embeddings[idx] or [],
                max_rq_tokens = max_rq_tokens,
                stop_on_loop = stop_on_loop,
            )

            if seed is not None: seed += 1

            serial = self.enqueue(job)
            order[serial] = idx

        # Collect outputs until all jobs finish
        completions = [""] * batch_size
        last_results = [None] * batch_size

        while self.num_remaining_jobs():
            results = self.iterate()

            for r in results:
                idx = order[r["serial"]]
                if r["stage"] == "error":
                    # A per-job failure was contained by reap_failed_job; surface the
                    # original exception to this caller rather than returning a silently
                    # truncated completion. Jobs from the same batch that are still in
                    # flight are left to the generator's normal queue handling.
                    raise r["error"]
                if r["stage"] == "streaming":
                    text = r.get("text", "")
                    completions[idx] += text
                if r["eos"]:
                    last_results[idx] = r
            if abort_event is not None and abort_event.is_set():
                self.clear_queue()
                return None

        # Return results
        if not completion_only:
            completions = [(p if isinstance(p, str) else p[0]) + c for p, c in zip(prompts, completions)]

        if not isinstance(prompt, list):
            completions = completions[0]
            last_results = last_results[0]

        if return_last_results:
            return completions, last_results
        else:
            return completions
