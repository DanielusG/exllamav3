#!/usr/bin/env python3
"""
Overnight bit-identity check of the flash-next fork against the upstream release it is based on
(BASE_RELEASE below), on the production model, with the server stopped.

This file is the orchestrator and needs nothing but the standard library: it plans the runs,
starts one child process per model load (experiments/exact_check_child.py, which imports torch,
ExLlamaV3 and the server's model loading), watches them, compares what they saved and writes the
report. A child runs either the fork (its checkout ahead of the installed package on PYTHONPATH)
or stock upstream (the installed wheel of that release alone), so the two can be compared run
for run.

Every compared run forces on each job the tokens a reference run produced, and records for every
position the bits of the logits (an integer digest over the raw values), the log-probability of
the forced token and the most likely tokens. "Identical" means identical logits in every compared
position; anything else is judged against the same run repeated (the noise floor).

Scenarios (--scenarios):
  1  the fork with every switch off against stock upstream (three processes: stock, fork,
     stock again for a floor between processes)
  2  staggered decode with 2-5 jobs generating, against each group of jobs alone
  3  prompt ingestion on the prefill thread beside 1-5 jobs generating (last chunks of 33-249
     tokens), against the same jobs with no prompt beside them and the prompt ingested as upstream
  4  two prompt chunks in flight (prefill pipeline): a prompt alone and beside a job, past 32768
     tokens, the MTP head's update on the first card against EXL3_PREFILL_PIPELINE_HEAD=second,
     a cancel in the middle of an ingestion, write-behind of the persistent disk tier (page CRCs),
     and a ~179k-token prompt at production's split in a process of its own (memory per card)
  5  the recurrent checkpoint that follows the answer: forced answer with it on and off, the next
     turn resumed from it, also after a restart with the disk tier (a second process)
  6  expert placement with the prefill weight unset against stock: same traffic, same sweeps,
     same placement decisions (in the processes of scenario 1)
  7  stress for hangs with every production switch on, watchdog on every library call

Run it with the production server STOPPED (it needs the cards), with the Python of the
environment that has the wheel of BASE_RELEASE, from any directory (FORK.md, "The overnight exact
check", has the command lines for the workstation):

    <env python> experiments/exact_check.py --dry-run --out <dir>   # plan, checks, time estimate
    <env python> experiments/exact_check.py --out <dir>             # the night
    <env python> experiments/exact_check.py --rebuild <dir>         # REPORT.md again from the records

The report (REPORT.md, Italian like the fork's other benches; code and comments in English) is
rewritten after every process, so a crash leaves the finished scenarios in it; every child
appends its records (synced) as it goes, and --rebuild makes the report from them. The night
survives a closed terminal (SIGHUP is ignored; orchestrator.log has what it printed); SIGTERM or
Ctrl-C stops it in order. Nothing is written outside the run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
FORK_DEFAULT = os.path.dirname(HERE)
CHILD_SCRIPT = os.path.join(HERE, "exact_check_child.py")

# The upstream release the fork's branch is on: the wheel the environment must have (stock runs it
# alone, the fork over it), the pin the server's checkout must carry, and the labels of the report
BASE_RELEASE = "1.6.3"
FORK_BRANCH = f"flash-next-{BASE_RELEASE}"

# --------------------------------------------------------------------------------------------
#   Defaults: production's configuration (the server's wrapper, AGENTS.md section 8)
# --------------------------------------------------------------------------------------------

MODEL_DEFAULT = "/home/daniele/models/Qwen3.8-Flash-Next-exl3-2.05bpw_h4_ng4/"
PLACEMENT_DEFAULT = "/home/daniele/models/qwen-flash-arena-cache/"
# The server's standard checkout, the one production runs. Its model loading has to be the one for
# this library: preflight() refuses a checkout whose pyproject does not pin exllamav3==BASE_RELEASE
SERVER_DEFAULT = "~/Desktop/exllamav3-openai-server"

# The paths of the MoE layers that are not bit-reproducible run to run, read on 1.6.2 (the three
# files are the same in 1.6.3, but for a comment in modules/block_sparse_mlp.py):
#   - the streamed prefill's fused tier calls exl3_moe without the slot scratch (arrival order of
#     the atomics; model/moe_cpu_host.py, _submit_prefill_streamed): EXL3_MOE_STREAM_FUSED_T=0
#     sends those experts to the tiers below;
#   - the batched reconstruct tier adds with one atomic index_add_ over the slab unless
#     EXL3_MOE_RECON_DET (modules/moe_batch_recon.py: its default follows EXL3_MOE_FUSED_DET with
#     a fallback of "0", i.e. atomic when neither is set);
#   - the GPU fused kernel's accumulation is deterministic by default on 1.6.x (FUSED_DET, the
#     default "1" in modules/block_sparse_mlp.py); pinned here so that nothing can turn it off.
# Graphs are left as production runs them: an uncaptured call issues the same kernels in the
# same order as the replay (exllamav3_ext/graph.cu); --env EXL3_GRAPHS=0 takes them out.
EXACT_ENV = {"EXL3_MOE_STREAM_FUSED_T": "0", "EXL3_MOE_RECON_DET": "1", "EXL3_MOE_FUSED_DET": "1"}

# MTP with one draft token, as production: a generating job is 1 + 1 decode rows, and the prefill
# thread runs a chunk beside the decode only up to Generator.side_max_rows rows
DRAFT_TOKENS = 1
SIDE_MAX_ROWS = 8

# Fork switches that the environment can turn on: scrubbed from every child, set only where a
# scenario asks for them (every EXL3_ variable of the caller is dropped anyway, see child_env)
FORK_ENV_SWITCHES = (
    "EXL3_STAGGER_DECODE", "EXL3_PREFILL_THREAD", "EXL3_PREFILL_PIPELINE", "EXL3_TAIL_CHECKPOINT",
    "EXL3_MOE_HIST_PREFILL_WEIGHT", "EXL3_DISK_CACHE_UNCACHE", "EXL3_PREFILL_PIPELINE_HEAD",
    "EXL3_MTP_BATCHED_VERIFY", "EXL3_MTP_BATCHED_UPDATE",
)

PROMPTS = [
    "Write a detailed technical essay, at least 1500 words long, on how a B-tree index works "
    "in a relational database: node layout, search, insertion with splits, deletion with "
    "merges, and why it suits disks.",
    "Write a long short story, at least 1500 words, about a lighthouse keeper who finds a "
    "message in a bottle every morning for a month. Use rich descriptions and dialogue.",
    "Write a complete Python module, with docstrings and comments, that implements an LRU "
    "cache, a trie with prefix search, and a priority queue based on a binary heap. Then "
    "explain each design choice at length.",
    "Explain at length, in at least 1500 words, the history of the steam engine from Newcomen "
    "to the locomotive, with dates, people, and the engineering problems each step solved.",
    "Describe in detail, in at least 1500 words, how a modern CPU executes instructions out of "
    "order: the reorder buffer, register renaming, reservation stations and branch prediction.",
    "Write a long travel diary, at least 1500 words, of a week spent walking across the Alps "
    "from village to village, with the people met, the food and the weather of every day.",
]
NAMES = "abcdef"
TURN_SUFFIX = "\n\nContinue from there, in two sentences."

_WORDS = (
    "the of and to in is that it was for on are as with his they at be this from have or one had by "
    "word but not what all were we when your can said there use an each which she do how their if "
    "will up other about out many then them these so some her would make like him into time has look "
    "two more write go see number no way could people my than first water been call who oil its now "
    "find long down day did get come made may part"
).split()


def filler(tag: str, paragraphs: int) -> str:
    """Deterministic pseudo-prose, different for every tag from its first words on (the page
    table shares identical pages between jobs; conversations in production share none). The
    same text in every process: the RNG is seeded from the tag alone."""
    rng = random.Random(tag)
    out = []
    for i in range(paragraphs):
        words = [rng.choice(_WORDS) for _ in range(90)]
        sentences = [" ".join(words[j:j + 15]).capitalize() + "." for j in range(0, 90, 15)]
        out.append(f"[{tag}-{i + 1}] " + " ".join(sentences))
    return "\n\n".join(out)


def groups_of(n: int) -> list:
    """The two groups a staggered run of the first n jobs settles into (Generator._stagger_pick:
    with nothing in flight the jobs are split in two, the larger half first)."""
    first = n - n // 2
    return [NAMES[:first], NAMES[first:n]]


def doc_length(last_chunk: int, pages: int = 32) -> int:
    """Prompt length whose last prefill pass is last_chunk tokens long. Job._prefill_co ingests
    tokens [0, len - 1) in chunks ending on page boundaries, and a recurrent model ingests the
    last partial page as a pass of its own: (len - 1) mod 256 tokens."""
    return pages * 256 + last_chunk + 1


# --------------------------------------------------------------------------------------------
#   Small helpers
# --------------------------------------------------------------------------------------------

def fmt(x, nd = 1):
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def fmt_g(x):
    """A log-probability difference: 0 as such, otherwise three significant digits."""
    if x is None:
        return "-"
    if x == 0:
        return "0"
    if math.isinf(x):
        return "inf"
    return f"{x:.3g}"


def table(header: list, rows: list) -> str:
    def cell(c):
        return str(c).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(cell(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)


def atomic_write(path: str, text: str):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding = "utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: str, obj: dict):
    """One record per line, synced: a process killed in the night keeps what it wrote."""
    line = json.dumps(obj, separators = (",", ":"), allow_nan = True)
    with open(path, "a", encoding = "utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str) -> list:
    """The records of a file, a truncated last line (a process killed while writing) left out."""
    out = []
    try:
        with open(path, encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def file_digest(path: str | None) -> str | None:
    """sha1 of a file's bytes, None when there is no file to read."""
    if not path:
        return None
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def git_revision(path: str) -> str:
    try:
        out = subprocess.run(["git", "-C", path, "log", "-1", "--format=%h %cs"],
                             capture_output = True, text = True, timeout = 10)
        if out.returncode != 0 or not out.stdout.strip():
            return "sconosciuto"
        branch = subprocess.run(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output = True, text = True, timeout = 10).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                               capture_output = True, text = True, timeout = 10).stdout.strip()
        return out.stdout.strip() + (f" ({branch})" if branch else "") + (" con modifiche non registrate" if dirty else "")
    except Exception:
        return "sconosciuto"


def placement_name(model_path: str) -> str:
    """The name the server gives the placement file of a model in a directory (openai/
    moe_tier.py, resolve_placement_path, with the model directory as the server resolves it:
    pathlib's resolve(), so symbolic links followed)."""
    name = os.path.basename(os.path.realpath(os.path.expanduser(model_path)))
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "model").strip("._") or "model"
    return f"moe-placement.{name}.json"


def resolve_placement_file(path: str, model_path: str) -> str:
    """The file the server reads for --moe-placement-file: a directory holds
    moe-placement.<model directory name>.json, anything else is the file itself."""
    path = os.path.expanduser(path)
    if path.endswith(os.sep) or os.path.isdir(path):
        return os.path.join(path, placement_name(model_path))
    return path


# --------------------------------------------------------------------------------------------
#   Disk tier files (upstream 1.6.2's DiskPageCache layout, unchanged in 1.6.3), shared with the child
# --------------------------------------------------------------------------------------------

PAGE_MAGIC = b"EXL3KVP1"
STATE_MAGIC = b"EXL3RCS1"


def _read_header(f, magic: bytes):
    if f.read(len(magic)) != magic:
        return None
    raw = f.read(8)
    if len(raw) != 8:
        return None
    hlen = int.from_bytes(raw[:4], "little")
    tlen = int.from_bytes(raw[4:], "little")
    meta = json.loads(f.read(hlen))
    return meta, tlen, len(magic) + 8 + hlen


def scan_disk(root: str) -> dict:
    """
    Every page file and checkpoint file under a disk tier's root (<root>/<fingerprint>/pages/
    <hh>/<phash>.kv, states/<phash>.rs), read back from the drive: for a page, the CRC32 its
    header carries (over the tokens and the image, written from the slab with the image) and
    whether the bytes on disk still give it; for a checkpoint, a CRC32 of its data (the image
    after the header, which carries ints and the tail flag). Temporary files are left out.
    """
    pages, states, bad = {}, {}, []
    fingerprints = []
    if not os.path.isdir(root):
        return {"pages": pages, "states": states, "bad": ["no such directory"], "fingerprints": fingerprints}
    for fp in sorted(os.listdir(root)):
        base = os.path.join(root, fp)
        if not os.path.isdir(base):
            continue
        fingerprints.append(fp)
        pdir = os.path.join(base, "pages")
        for sub in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
            for name in sorted(os.listdir(os.path.join(pdir, sub))):
                if not name.endswith(".kv"):
                    continue
                path = os.path.join(pdir, sub, name)
                try:
                    with open(path, "rb") as f:
                        head = _read_header(f, PAGE_MAGIC)
                        if head is None:
                            bad.append(name)
                            continue
                        meta, tlen, _ = head
                        crc = zlib.crc32(f.read(tlen))
                        size = 0
                        while True:
                            block = f.read(1 << 22)
                            if not block:
                                break
                            size += len(block)
                            crc = zlib.crc32(block, crc)
                    want = meta.get("crc32")
                    pages[name[:-3]] = {"crc": want, "crc_disk": crc, "ok": want is None or want == crc,
                                        "bytes": size, "prev": meta.get("prev_hash")}
                except (OSError, ValueError):
                    bad.append(name)
        sdir = os.path.join(base, "states")
        for name in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
            if not name.endswith(".rs"):
                continue
            path = os.path.join(sdir, name)
            try:
                with open(path, "rb") as f:
                    head = _read_header(f, STATE_MAGIC)
                    if head is None:
                        bad.append(name)
                        continue
                    meta, _, _ = head
                    crc, size = 0, 0
                    while True:
                        block = f.read(1 << 22)
                        if not block:
                            break
                        size += len(block)
                        crc = zlib.crc32(block, crc)
                states[name[:-3]] = {"crc": crc, "bytes": size, "position": meta.get("position"),
                                     "tail": bool((meta.get("top") or {}).get("tail"))}
            except (OSError, ValueError):
                bad.append(name)
    return {"pages": pages, "states": states, "bad": bad, "fingerprints": fingerprints}


# --------------------------------------------------------------------------------------------
#   The plan
# --------------------------------------------------------------------------------------------

def q(i: int) -> str:
    return f"q{i}"


def doc(tag: str, length: int, question: int = 1) -> str:
    return f"doc/{tag}/{length}/{question}"


def ref(proc: str, run: str, job: str, limit: int | None = None) -> dict:
    r = {"proc": proc, "run": run, "job": job}
    if limit:
        r["limit"] = limit
    return r


def jobspec(name: str, prompt: str, max_new: int | None = None, forced: dict | None = None,
            start: dict | None = None) -> dict:
    j = {"name": name, "prompt": prompt}
    if forced is not None:
        j["forced"] = forced
    else:
        j["max_new"] = int(max_new)
    if start is not None:
        j["start"] = start
    return j


def run_step(rid: str, scenario: int, jobs: list, *, forget: bool = True, switches: dict | None = None,
             until: dict | None = None, cancel: list | None = None, after_cancel: dict | None = None,
             warmup: bool = False, disk_at_first_token: bool = False) -> dict:
    return {
        "op": "run", "id": rid, "scenario": scenario, "jobs": jobs, "forget": forget,
        "switches": switches or {}, "until": until, "cancel": cancel or [], "after_cancel": after_cancel,
        "warmup": warmup, "disk_at_first_token": disk_at_first_token,
    }


def gen_step(gid: str, scenario: int, switches: dict | None = None, disk: str | None = None,
             env: dict | None = None) -> dict:
    """A new generator over the loaded model and cache (the previous one is closed first)."""
    return {"op": "generator", "id": gid, "scenario": scenario, "switches": switches or {}, "disk": disk, "env": env or {}}


def cmp(scenario: int, cid: str, label: str, test: dict, refr: dict, *, expect: str = "identical",
        floor=None, floor2=None, yards=None, upto=None, drafts: bool = True, requires: dict | None = None,
        soft: dict | None = None) -> dict:
    """
    One probability comparison: `test` against `refr` (same forced tokens), judged against the
    same reference run repeated (`floor`, else `floor2`). `yards` are controls: for a comparison
    expected within the noise they are the yardstick; for one expected identical, a control that
    differs from the reference says the test cannot be identical for a reason of upstream's own
    (the row is then within that control's noise, said so). `upto`: an int, {"window": job} (the
    positions generated during that job's ingestion) or {"before_first_end": True}. `requires`:
    what the test run must have exercised, else the row is an error; `soft`: the same, as a note.
    """
    return {
        "kind": "probs", "scenario": scenario, "id": cid, "label": label, "test": test, "ref": refr,
        "expect": expect, "floor": floor, "floor2": floor2, "yards": yards or [], "upto": upto,
        "drafts": drafts, "requires": requires or {}, "soft": soft or {},
    }


class Proc:
    """One child process: one model load, its steps in order."""

    def __init__(self, name: str, library: str, scenarios: list, *, env: dict | None = None,
                 server_args: list | None = None, sweeps: str = "frozen"):
        self.name = name
        self.library = library          # "fork" | "stock"
        self.scenarios = list(scenarios)
        self.env = dict(env or {})
        self.server_args = list(server_args or [])
        self.sweeps = sweeps            # "frozen" | "server" (as loaded) for the whole process
        self.steps = []
        self.prompts = {}

    def add(self, *steps):
        self.steps.extend(steps)

    def as_dict(self) -> dict:
        return {"name": self.name, "library": self.library, "scenarios": self.scenarios, "env": self.env,
                "server_args": self.server_args, "sweeps": self.sweeps, "steps": self.steps, "prompts": self.prompts}


SCENARIO_TITLES = {
    1: f"Fork con ogni interruttore spento contro upstream {BASE_RELEASE}",
    2: "Decode sfalsato",
    3: "Ingestione sul thread di prefill accanto al decode",
    4: "Due pezzi di prompt in volo sulle due schede",
    5: "Checkpoint che segue la risposta",
    6: "Piazzamento degli esperti col peso del prefill non impostato",
    7: "Stress: blocchi",
}


def prompt_specs(steps: list, extra: dict) -> dict:
    """The prompt table of a process: every key its runs use, turns from `extra`."""
    out = {}
    for st in steps:
        for j in st.get("jobs", []) if st.get("op") == "run" else []:
            key = j["prompt"]
            if key in extra:
                out[key] = extra[key]
                base = extra[key].get("base")
                if base:
                    out[base] = prompt_spec_of(base)
            else:
                out[key] = prompt_spec_of(key)
    return out


def prompt_spec_of(key: str) -> dict:
    if key.startswith("q"):
        return {"kind": "question", "index": int(key[1:])}
    if key.startswith("doc/"):
        _, tag, length, question = key.split("/")
        return {"kind": "doc", "tag": tag, "length": int(length), "question": int(question)}
    raise ValueError(f"unknown prompt key {key}")


def build_plan(args) -> dict:
    """Processes, comparisons and checks for the scenarios asked for."""
    sc = set(args.scenarios)
    procs = []
    comps = []
    T = args.tokens
    D = args.decoder_tokens
    R = args.reader_tokens
    L = args.lead
    W = args.warmup_tokens

    # ---------------------------------------------------------------- scenarios 1 and 6
    if sc & {1, 6}:
        tag = "s1" if 1 in sc else "s6"
        names = [(f"{tag}-stock-a", "stock"), (f"{tag}-fork", "fork"), (f"{tag}-stock-b", "stock")]
        P0, P1, P2 = (n for n, _ in names)
        r8 = doc("s1r", doc_length(96), 1)
        turn_key = "turn/s1"
        turn_spec = {"kind": "turn", "base": r8, "answer": ref(P0, "s1.mix.free", "r"), "suffix": TURN_SUFFIX}
        for name, lib in names:
            p = Proc(name, lib, sorted(sc & {1, 6}), server_args = [])
            p.add(gen_step("g0", min(sc & {1, 6})))
            p.add(run_step(f"{tag}.warm.1", min(sc & {1, 6}), [jobspec("a", q(0), W)], warmup = True))
            p.add(run_step(f"{tag}.warm.3", min(sc & {1, 6}), [jobspec(x, q(i), W) for i, x in enumerate("abc")], warmup = True))
            p.add(run_step(f"{tag}.warm.r", min(sc & {1, 6}), [jobspec("r", doc("s1w", doc_length(64, 8), 2), 8)], warmup = True))
            if 1 in sc:
                for i, x in enumerate("abc"):
                    p.add(run_step(f"s1.free.{x}", 1, [jobspec(x, q(i), T)]))
                for k in (1, 2):
                    p.add(run_step(f"s1.a.ref{k}", 1, [jobspec("a", q(0), forced = ref(P0, "s1.free.a", "a"))]))
                for k in (1, 2):
                    p.add(run_step(f"s1.abc.ref{k}", 1, [jobspec(x, q(i), forced = ref(P0, f"s1.free.{x}", x))
                                                          for i, x in enumerate("abc")]))
                p.add(run_step("s1.mix.free", 1, [jobspec("a", q(0), 6000), jobspec("r", r8, R, start = {"after": "a", "tokens": L})],
                               until = {"job": "r"}))
                for k in (1, 2):
                    p.add(run_step(f"s1.mix.ref{k}", 1, [jobspec("a", q(0), 6000),
                                                          jobspec("r", r8, forced = ref(P0, "s1.mix.free", "r"), start = {"after": "a", "tokens": L})],
                                   until = {"job": "r"}))
                # The next turn: resumed from the prompt's last-page checkpoint of the run before
                p.add(run_step("s1.turn.free", 1, [jobspec("t", turn_key, 32)], forget = False))
                for k in (1, 2):
                    p.add(run_step(f"s1.turn.prep{k}", 1, [jobspec("r", r8, forced = ref(P0, "s1.mix.free", "r"))]))
                    p.add(run_step(f"s1.turn.ref{k}", 1, [jobspec("t", turn_key, forced = ref(P0, "s1.turn.free", "t"))], forget = False))
                # Upstream's disk tier with every switch off: the page images it writes
                p.add(gen_step("g1", 1, disk = f"s1-{name}"))
                p.add(run_step("s1.disk.run", 1, [jobspec("r", r8, forced = ref(P0, "s1.mix.free", "r"))]))
                p.add({"op": "disk_scan", "id": "s1.disk.scan", "scenario": 1, "disk": f"s1-{name}"})
                p.add(gen_step("g2", 1))
            if 6 in sc:
                r6 = doc("s6r", doc_length(96), 3)
                if name == P0:
                    for i, x in enumerate("abc"):
                        p.add(run_step(f"s6.free.{x}", 6, [jobspec(x, q(i), args.s6_tokens)]))
                    p.add(run_step("s6.free.r", 6, [jobspec("r", r6, 16)]))
                p.add({"op": "reset_hist", "id": "s6.reset", "scenario": 6})
                for rnd in range(1, args.s6_rounds + 1):
                    p.add(run_step(f"s6.r{rnd}.traffic", 6,
                                   [jobspec(x, q(i), forced = ref(P0, f"s6.free.{x}", x)) for i, x in enumerate("abc")]
                                   + [jobspec("r", r6, forced = ref(P0, "s6.free.r", "r"), start = {"after": "a", "tokens": L})],
                                   until = {"all": True}))
                    p.add({"op": "sweep", "id": f"s6.r{rnd}.sweep", "scenario": 6})
            p.prompts = prompt_specs(p.steps, {turn_key: turn_spec})
            procs.append(p)

        if 1 in sc:
            def three(run, job, label, drafts = True):
                comps.append(cmp(1, f"s1.{run}.{job}", label, ref(P1, f"s1.{run}.ref1", job), ref(P0, f"s1.{run}.ref1", job),
                                 floor = (ref(P0, f"s1.{run}.ref1", job), ref(P2, f"s1.{run}.ref1", job)),
                                 floor2 = (ref(P0, f"s1.{run}.ref1", job), ref(P0, f"s1.{run}.ref2", job)), drafts = drafts))
            three("a", "a", "`a` da solo: fork contro stock")
            for x in "abc":
                three("abc", x, f"`{x}` in un batch di tre: fork contro stock")
            three("mix", "r", "`r` (documento di 8289 token) accodato accanto ad `a` che genera, come upstream: fork contro stock")
            three("turn", "t", "turno successivo ripreso dal checkpoint dell'ultima pagina del prompt: fork contro stock")
            comps.append(cmp(1, "s1.disk.r", "`r` con il tier su disco di upstream acceso: fork contro stock",
                             ref(P1, "s1.disk.run", "r"), ref(P0, "s1.disk.run", "r"),
                             floor = (ref(P0, "s1.disk.run", "r"), ref(P2, "s1.disk.run", "r"))))
            comps.append({"kind": "crc", "scenario": 1, "id": "s1.disk.crc",
                          "label": "file delle pagine scritti dal tier su disco (CRC): fork contro stock",
                          "test": ref(P1, "s1.disk.scan", ""), "ref": ref(P0, "s1.disk.scan", ""),
                          "floor": (ref(P0, "s1.disk.scan", ""), ref(P2, "s1.disk.scan", ""))})
            for x in "abc":
                comps.append({"kind": "tokens", "scenario": 1, "id": f"s1.free.{x}",
                              "label": f"token greedy liberi di `{x}`: fork contro stock",
                              "test": ref(P1, f"s1.free.{x}", x), "ref": ref(P0, f"s1.free.{x}", x)})
            comps.append({"kind": "same_placement", "scenario": 1, "id": "s1.placement", "procs": [P0, P1, P2]})
            comps.append({"kind": "switches_off", "scenario": 1, "id": "s1.switches", "proc": P1})
            comps.append({"kind": "library", "scenario": 1, "id": "s1.library", "procs": {P0: "stock", P1: "fork", P2: "stock"}})
        if 6 in sc:
            # The placement every process starts the traffic from (after scenario 1, if it ran)
            comps.append({"kind": "same_placement", "scenario": 6, "id": "s6.start", "procs": [P0, P1, P2], "record": "s6.reset"})
            if 1 not in sc:
                comps.append({"kind": "switches_off", "scenario": 6, "id": "s6.switches", "proc": P1})
                comps.append({"kind": "library", "scenario": 6, "id": "s6.library", "procs": {P0: "stock", P1: "fork", P2: "stock"}})
            comps.append({"kind": "placement", "scenario": 6, "id": "s6.placement",
                          "label": "decisioni del piazzamento dopo ogni sweep: fork (peso non impostato) contro stock",
                          "test": P1, "ref": P0, "floor": (P0, P2), "rounds": args.s6_rounds})
            for k in range(1, args.s6_rounds + 1):
                for x in "abc":
                    comps.append(cmp(6, f"s6.r{k}.{x}", f"giro {k}, `{x}`: fork contro stock (stesso traffico prima dello sweep)",
                                     ref(P1, f"s6.r{k}.traffic", x), ref(P0, f"s6.r{k}.traffic", x),
                                     floor = (ref(P0, f"s6.r{k}.traffic", x), ref(P2, f"s6.r{k}.traffic", x)), drafts = False))

    # ---------------------------------------------------------------- scenario 2
    if 2 in sc:
        P = "s2-fork"
        p = Proc(P, "fork", [2])
        ns = [n for n in args.s2_jobs if 2 <= n <= 5]
        p.add(gen_step("g0", 2))
        for n in ns:
            names = NAMES[:n]
            p.add(run_step(f"s2.warm.{n}", 2, [jobspec(x, q(i), W) for i, x in enumerate(names)], warmup = True))
            p.add(run_step(f"s2.warm.{n}.st", 2, [jobspec(x, q(i), W) for i, x in enumerate(names)], warmup = True,
                           switches = {"stagger": True, "stagger_min_jobs": 2}))
        for i, x in enumerate(NAMES[:max(ns)]):
            p.add(run_step(f"s2.free.{x}", 2, [jobspec(x, q(i), T)]))
        groups = []
        for n in ns:
            for g in groups_of(n):
                if g not in groups:
                    groups.append(g)
        for g in groups:
            for k in (1, 2):
                p.add(run_step(f"s2.g.{g}.ref{k}", 2, [jobspec(x, q(NAMES.index(x)), forced = ref(P, f"s2.free.{x}", x)) for x in g]))
        for n in ns:
            jobs = [jobspec(x, q(i), forced = ref(P, f"s2.free.{x}", x)) for i, x in enumerate(NAMES[:n])]
            p.add(run_step(f"s2.n{n}.batch", 2, jobs))
            p.add(run_step(f"s2.n{n}.stagger", 2, jobs, switches = {"stagger": True, "stagger_min_jobs": 2}))
        p.prompts = prompt_specs(p.steps, {})
        procs.append(p)
        # Positions up to the iteration in which the first job of the run ended: the groups are
        # the reference's until then. Past it the staggered decode re-forms them
        # (Generator._stagger_pick splits the remaining group in two once the other is empty, and
        # below stagger_min_jobs it goes back to one batch), which the group alone does not
        first_end = {"before_first_end": True}
        for n in ns:
            for g in groups_of(n):
                for x in g:
                    comps.append(cmp(2, f"s2.n{n}.{x}", f"{n} job, `{x}` sfalsato contro il gruppo `{g}` da solo",
                                     ref(P, f"s2.n{n}.stagger", x), ref(P, f"s2.g.{g}.ref1", x),
                                     floor = (ref(P, f"s2.g.{g}.ref1", x), ref(P, f"s2.g.{g}.ref2", x)),
                                     upto = first_end, requires = {"stagger_calls": True}))
                    comps.append(cmp(2, f"s2.n{n}.{x}.batch", f"{n} job, `{x}` in un batch normale contro il gruppo `{g}` da solo",
                                     ref(P, f"s2.n{n}.batch", x), ref(P, f"s2.g.{g}.ref1", x), expect = "info",
                                     upto = first_end))

    # ---------------------------------------------------------------- scenario 3
    if 3 in sc:
        P = "s3-fork"
        p = Proc(P, "fork", [3])
        p.add(gen_step("g0", 3))
        decs = [n for n in args.s3_decoders if 2 <= n <= 5]
        nmax = max(decs + [1])
        rb = doc("s3rb", doc_length(96), 1)
        p.add(run_step("s3.warm.1", 3, [jobspec("a", q(0), W)], warmup = True))
        p.add(run_step("s3.warm.n", 3, [jobspec(x, q(i), W) for i, x in enumerate(NAMES[:nmax])], warmup = True))
        p.add(run_step("s3.warm.r", 3, [jobspec("r", doc("s3w", doc_length(64, 8), 2), 8)], warmup = True))
        for i, x in enumerate(NAMES[:nmax]):
            p.add(run_step(f"s3.free.{x}", 3, [jobspec(x, q(i), D)]))
        sets = [1] + decs
        for n in sets:
            for k in (1, 2):
                p.add(run_step(f"s3.dec{n}.ref{k}", 3, [jobspec(x, q(i), forced = ref(P, f"s3.free.{x}", x))
                                                         for i, x in enumerate(NAMES[:n])]))

        def reader_runs(case: str, n: int, prompt: str, thread: bool):
            decoders = NAMES[:n]
            sw = {"thread": True} if thread else {}
            free_dec = [jobspec(x, q(i), 6000) for i, x in enumerate(decoders)]
            forced_dec = [jobspec(x, q(i), forced = ref(P, f"s3.free.{x}", x)) for i, x in enumerate(decoders)]
            after = {"after": "a", "tokens": L}
            if not thread:
                p.add(run_step(f"s3.{case}.free", 3, free_dec + [jobspec("r", prompt, R, start = after)], until = {"job": "r"}))
                for k in (1, 2):
                    p.add(run_step(f"s3.{case}.ref{k}", 3, free_dec + [jobspec("r", prompt, forced = ref(P, f"s3.{case}.free", "r"), start = after)],
                                   until = {"job": "r"}))
                # Control, as upstream: `r` queued later, so that it joins the decode batch with
                # the jobs further on in their text, as it does after an ingestion on the thread
                # (the jobs generate meanwhile). Tells a difference of the thread from one that
                # comes from what the other rows of the batch hold
                p.add(run_step(f"s3.{case}.shift", 3, free_dec + [jobspec("r", prompt, forced = ref(P, f"s3.{case}.free", "r"),
                                                                          start = {"after": "a", "tokens": L + args.shift})],
                               until = {"job": "r"}))
            else:
                p.add(run_step(f"s3.{case}.thr.reader", 3, free_dec + [jobspec("r", prompt, forced = ref(P, f"s3.{case}.free", "r"), start = after)],
                               until = {"job": "r"}, switches = sw))
                p.add(run_step(f"s3.{case}.thr.dec", 3, forced_dec + [jobspec("r", prompt, R, start = after)],
                               until = {"job": "r"}, switches = sw))

        cases = [(f"L{lc}", 1, doc(f"s3r{lc}", doc_length(lc), 1)) for lc in args.s3_last_chunks]
        cases += [(f"n{n}", n, rb) for n in decs]
        for case, n, prompt in cases:
            reader_runs(case, n, prompt, False)
        p.add(run_step("s3.alone.free", 3, [jobspec("r", rb, R)]))
        for k in (1, 2):
            p.add(run_step(f"s3.alone.ref{k}", 3, [jobspec("r", rb, forced = ref(P, "s3.alone.free", "r"))]))
        # From here on the prefill thread (its first round splits the CPU expert worker's slots)
        p.add(run_step("s3.warm.thr", 3, [jobspec("a", q(0), 6000), jobspec("r", doc("s3w", doc_length(64, 8), 2), 8, start = {"after": "a", "tokens": 16})],
                       until = {"job": "r"}, switches = {"thread": True}, warmup = True))
        for case, n, prompt in cases:
            reader_runs(case, n, prompt, True)
        p.add(run_step("s3.alone.thr", 3, [jobspec("r", rb, forced = ref(P, "s3.alone.free", "r"))], switches = {"thread": True}))
        p.prompts = prompt_specs(p.steps, {})
        procs.append(p)
        for case, n, prompt in cases:
            what = f"ultimo pezzo di {case[1:]} token" if case.startswith("L") else f"{n} job che generano"
            # Past side_max_rows decode rows the chunks are waited for, as an inline one: the
            # comparison still holds, with no chunk beside the decode to ask for
            beside = n * (1 + DRAFT_TOKENS) <= SIDE_MAX_ROWS
            if not beside:
                what += f" ({n * (1 + DRAFT_TOKENS)} righe di decode, oltre {SIDE_MAX_ROWS}: pezzi attesi)"
            comps.append(cmp(3, f"s3.{case}.reader", f"{what}: `r` ingerito sul thread contro come upstream, accanto agli stessi job",
                             ref(P, f"s3.{case}.thr.reader", "r"), ref(P, f"s3.{case}.ref1", "r"),
                             floor = (ref(P, f"s3.{case}.ref1", "r"), ref(P, f"s3.{case}.ref2", "r")),
                             yards = [(ref(P, f"s3.{case}.ref1", "r"), ref(P, f"s3.{case}.shift", "r"))],
                             requires = {"thread": "on"}, soft = {"all_beside": case.startswith("L")}))
            for x in NAMES[:n]:
                comps.append(cmp(3, f"s3.{case}.{x}", f"{what}: `{x}` accanto all'ingestione sul thread contro gli stessi job senza prompt accanto",
                                 ref(P, f"s3.{case}.thr.dec", x), ref(P, f"s3.dec{n}.ref1", x),
                                 floor = (ref(P, f"s3.dec{n}.ref1", x), ref(P, f"s3.dec{n}.ref2", x)),
                                 upto = {"window": "r"}, drafts = False,
                                 requires = {"thread": "on", "beside": True} if beside else {"thread": "on"}))
        comps.append(cmp(3, "s3.alone", "`r` da solo, ingerito sul thread contro come upstream",
                         ref(P, "s3.alone.thr", "r"), ref(P, "s3.alone.ref1", "r"),
                         floor = (ref(P, "s3.alone.ref1", "r"), ref(P, "s3.alone.ref2", "r")), requires = {"thread": "on"}))

    # ---------------------------------------------------------------- scenario 4
    if 4 in sc:
        P = "s4-fork"
        p = Proc(P, "fork", [4], env = {"EXL3_DISK_CACHE_UNCACHE": "1"})
        r32 = doc("s4r", args.s4_prompt, 1)
        pipe = {"pipeline": True, "thread": True}
        after = {"after": "a", "tokens": L}
        p.add(gen_step("g0", 4))
        p.add(run_step("s4.warm.1", 4, [jobspec("a", q(0), W)], warmup = True))
        p.add(run_step("s4.warm.r", 4, [jobspec("r", doc("s4w", doc_length(64, 8), 2), 8)], warmup = True))
        p.add(run_step("s4.free.a", 4, [jobspec("a", q(0), D)]))
        for k in (1, 2):
            p.add(run_step(f"s4.a.ref{k}", 4, [jobspec("a", q(0), forced = ref(P, "s4.free.a", "a"))]))
        p.add(run_step("s4.alone.free", 4, [jobspec("r", r32, R)]))
        for k in (1, 2):
            p.add(run_step(f"s4.alone.ref{k}", 4, [jobspec("r", r32, forced = ref(P, "s4.alone.free", "r"))]))
        p.add(run_step("s4.mix.free", 4, [jobspec("a", q(0), 6000), jobspec("r", r32, R, start = after)], until = {"job": "r"}))
        for k in (1, 2):
            p.add(run_step(f"s4.mix.ref{k}", 4, [jobspec("a", q(0), 6000), jobspec("r", r32, forced = ref(P, "s4.mix.free", "r"), start = after)],
                           until = {"job": "r"}))
        # Control, as upstream: `r` queued later (see scenario 3)
        p.add(run_step("s4.mix.shift", 4, [jobspec("a", q(0), 6000), jobspec("r", r32, forced = ref(P, "s4.mix.free", "r"),
                                                                             start = {"after": "a", "tokens": L + args.shift})],
                       until = {"job": "r"}))
        # From here on two chunks in flight (the first round splits the worker's slots in three)
        p.add(run_step("s4.warm.pipe", 4, [jobspec("r", doc("s4w", doc_length(64, 8), 2), 8)], switches = pipe, warmup = True))
        p.add(run_step("s4.alone.pipe", 4, [jobspec("r", r32, forced = ref(P, "s4.alone.free", "r"))], switches = pipe))
        p.add(run_step("s4.mix.pipe.reader", 4, [jobspec("a", q(0), 6000), jobspec("r", r32, forced = ref(P, "s4.mix.free", "r"), start = after)],
                       until = {"job": "r"}, switches = pipe))
        p.add(run_step("s4.mix.pipe.dec", 4, [jobspec("a", q(0), forced = ref(P, "s4.free.a", "a")), jobspec("r", r32, R, start = after)],
                       until = {"job": "r"}, switches = pipe))
        p.add(run_step("s4.cancel", 4, [jobspec("a", q(0), 6000), jobspec("r", r32, R, start = after)], switches = pipe,
                       cancel = [{"job": "r", "at_progress": 0.5}], after_cancel = {"job": "a", "tokens": 32}))
        p.add(run_step("s4.after.a", 4, [jobspec("a", q(0), forced = ref(P, "s4.free.a", "a", limit = 64))], switches = pipe))
        p.add(run_step("s4.after.alone", 4, [jobspec("r", r32, forced = ref(P, "s4.alone.free", "r"))], switches = pipe))
        # The MTP head's update left to the second half's thread (read when the pipeline is planned)
        p.add(gen_step("g1", 4, switches = pipe, env = {"EXL3_PREFILL_PIPELINE_HEAD": "second"}))
        p.add(run_step("s4.head2.alone", 4, [jobspec("r", r32, forced = ref(P, "s4.alone.free", "r"))], switches = pipe))
        # Upstream's disk tier: the same ingestion in line and in flight, then read back
        p.add(gen_step("g2", 4, disk = "s4-inline", env = {"EXL3_PREFILL_PIPELINE_HEAD": ""}))
        p.add(run_step("s4.disk.inline", 4, [jobspec("r", r32, forced = ref(P, "s4.alone.free", "r"))], disk_at_first_token = True))
        p.add({"op": "disk_scan", "id": "s4.disk.inline.scan", "scenario": 4, "disk": "s4-inline"})
        p.add(gen_step("g3", 4, switches = pipe, disk = "s4-pipe"))
        p.add(run_step("s4.disk.pipe", 4, [jobspec("r", r32, forced = ref(P, "s4.alone.free", "r"))], switches = pipe, disk_at_first_token = True))
        p.add({"op": "disk_scan", "id": "s4.disk.pipe.scan", "scenario": 4, "disk": "s4-pipe"})
        p.add(gen_step("g4", 4, switches = pipe, disk = "s4-pipe"))
        p.add(run_step("s4.disk.reread", 4, [jobspec("r", r32, forced = ref(P, "s4.alone.free", "r"))], switches = pipe, forget = False))
        p.prompts = prompt_specs(p.steps, {})
        procs.append(p)
        # The MTP head's update on the first card (the fork's default) unless asked otherwise
        need_pipe = {"pipeline": "on", "piped": True}
        comps += [
            cmp(4, "s4.alone", f"`r` ({args.s4_prompt} token) da solo, in volo contro in linea",
                ref(P, "s4.alone.pipe", "r"), ref(P, "s4.alone.ref1", "r"),
                floor = (ref(P, "s4.alone.ref1", "r"), ref(P, "s4.alone.ref2", "r")), requires = dict(need_pipe, ahead = True),
                soft = {"head_first": True}),
            cmp(4, "s4.mix.r", "`r` accanto ad `a` che genera, in volo contro in linea",
                ref(P, "s4.mix.pipe.reader", "r"), ref(P, "s4.mix.ref1", "r"),
                floor = (ref(P, "s4.mix.ref1", "r"), ref(P, "s4.mix.ref2", "r")),
                yards = [(ref(P, "s4.mix.ref1", "r"), ref(P, "s4.mix.shift", "r"))], requires = need_pipe),
            cmp(4, "s4.mix.a", "`a` accanto all'ingestione in volo contro `a` da solo",
                ref(P, "s4.mix.pipe.dec", "a"), ref(P, "s4.a.ref1", "a"),
                floor = (ref(P, "s4.a.ref1", "a"), ref(P, "s4.a.ref2", "a")), upto = {"window": "r"}, drafts = False,
                requires = {"pipeline": "on", "beside": True}),
            cmp(4, "s4.after.a", "dopo l'annullamento: `a` da solo (64 posizioni) contro `a` da solo prima",
                ref(P, "s4.after.a", "a"), ref(P, "s4.a.ref1", "a"),
                floor = (ref(P, "s4.a.ref1", "a"), ref(P, "s4.a.ref2", "a")), upto = 64, drafts = False),
            cmp(4, "s4.after.r", "dopo l'annullamento: `r` da solo in volo contro in linea",
                ref(P, "s4.after.alone", "r"), ref(P, "s4.alone.ref1", "r"),
                floor = (ref(P, "s4.alone.ref1", "r"), ref(P, "s4.alone.ref2", "r")), requires = need_pipe),
            cmp(4, "s4.head2", "`r` da solo in volo con EXL3_PREFILL_PIPELINE_HEAD=second contro in linea",
                ref(P, "s4.head2.alone", "r"), ref(P, "s4.alone.ref1", "r"),
                floor = (ref(P, "s4.alone.ref1", "r"), ref(P, "s4.alone.ref2", "r")), requires = dict(need_pipe, head_first = False)),
            cmp(4, "s4.disk.inline", "`r` in linea col tier su disco contro senza",
                ref(P, "s4.disk.inline", "r"), ref(P, "s4.alone.ref1", "r"),
                floor = (ref(P, "s4.alone.ref1", "r"), ref(P, "s4.alone.ref2", "r"))),
            cmp(4, "s4.disk.pipe", "`r` in volo col tier su disco contro in linea senza",
                ref(P, "s4.disk.pipe", "r"), ref(P, "s4.alone.ref1", "r"),
                floor = (ref(P, "s4.alone.ref1", "r"), ref(P, "s4.alone.ref2", "r")), requires = need_pipe),
            # Resumed against ingested: the MTP head's drafts may legitimately differ (its state at
            # the resume point is rebuilt), so only the target's logits are judged
            cmp(4, "s4.disk.reread", "`r` ripreso dal disco (pagine e checkpoint riletti) contro la sua ingestione in volo",
                ref(P, "s4.disk.reread", "r"), ref(P, "s4.disk.pipe", "r"),
                floor = (ref(P, "s4.alone.ref1", "r"), ref(P, "s4.alone.ref2", "r")), drafts = False,
                requires = {"resumed": True}),
            {"kind": "crc", "scenario": 4, "id": "s4.disk.crc",
             "label": "file delle pagine scritti durante l'ingestione in volo contro in linea (CRC)",
             "test": ref(P, "s4.disk.pipe.scan", ""), "ref": ref(P, "s4.disk.inline.scan", ""), "floor": None,
             "write_behind": ref(P, "s4.disk.pipe", "r")},
            {"kind": "cancel", "scenario": 4, "id": "s4.cancel", "run": ref(P, "s4.cancel", "r")},
        ]
        if args.s4_long:
            PL = "s4-long"
            pl = Proc(PL, "fork", [4])
            r179 = doc("s4l", args.s4_long, 1)
            pl.add(gen_step("g0", 4))
            pl.add(run_step("s4l.warm.r", 4, [jobspec("r", doc("s4w", doc_length(64, 8), 2), 8)], warmup = True))
            pl.add(run_step("s4l.free", 4, [jobspec("r", r179, 32)]))
            if args.s4_long_floor:
                pl.add(run_step("s4l.ref2", 4, [jobspec("r", r179, forced = ref(PL, "s4l.free", "r"))]))
            pl.add(run_step("s4l.pipe", 4, [jobspec("r", r179, forced = ref(PL, "s4l.free", "r"))], switches = pipe))
            pl.prompts = prompt_specs(pl.steps, {})
            procs.append(pl)
            comps.append(cmp(4, "s4l.pipe", f"`r` di {args.s4_long} token, in volo contro in linea (riferimento: il giro libero)",
                             ref(PL, "s4l.pipe", "r"), ref(PL, "s4l.free", "r"),
                             floor = (ref(PL, "s4l.free", "r"), ref(PL, "s4l.ref2", "r")) if args.s4_long_floor else None,
                             requires = need_pipe))

    # ---------------------------------------------------------------- scenario 5
    if 5 in sc:
        PA, PB = "s5-fork-a", "s5-fork-b"
        p5 = doc("s5", doc_length(1, 32), 2)
        tail = {"tail": 256}
        turn_key = "turn/s5"
        turn_spec = {"kind": "turn", "base": p5, "answer": ref(PA, "s5.t1.free", "x"), "suffix": TURN_SUFFIX}
        A = ref(PA, "s5.t1.free", "x")
        B = ref(PA, "s5.t2.full.free", "y")
        p = Proc(PA, "fork", [5], env = {"EXL3_DISK_CACHE_UNCACHE": "1"})
        p.add(gen_step("g0", 5))
        p.add(run_step("s5.warm", 5, [jobspec("x", doc("s5w", doc_length(64, 8), 2), 8)], warmup = True))
        p.add(run_step("s5.t1.free", 5, [jobspec("x", p5, args.s5_answer)]))
        for k in (1, 2):
            p.add(run_step(f"s5.t1.off{k}", 5, [jobspec("x", p5, forced = A)]))
        p.add(run_step("s5.t1.on", 5, [jobspec("x", p5, forced = A)], switches = tail))
        p.add(run_step("s5.t2.full.free", 5, [jobspec("y", turn_key, args.s5_next)]))
        p.add(run_step("s5.t2.full.c2048", 5, [jobspec("y", turn_key, forced = B)], switches = {"chunk": 2048}))
        p.add(run_step("s5.t2.full.ref2", 5, [jobspec("y", turn_key, forced = B)]))
        p.add(run_step("s5.t2.tail.prep", 5, [jobspec("x", p5, forced = A)], switches = tail))
        p.add(run_step("s5.t2.tail", 5, [jobspec("y", turn_key, forced = B)], switches = tail, forget = False))
        p.add(run_step("s5.t2.up.prep", 5, [jobspec("x", p5, forced = A)]))
        p.add(run_step("s5.t2.up", 5, [jobspec("y", turn_key, forced = B)], forget = False))
        # The disk tier shared with the next process: the first turn with the checkpoint, then
        # the close, which persists what is only in VRAM or RAM (the tail checkpoint with it)
        p.add(gen_step("g1", 5, switches = {"tail": True}, disk = "s5"))
        p.add(run_step("s5.t1.disk", 5, [jobspec("x", p5, forced = A)], switches = tail))
        p.add({"op": "close", "id": "s5.close", "scenario": 5})
        p.add({"op": "disk_scan", "id": "s5.disk.scan", "scenario": 5, "disk": "s5"})
        p.prompts = prompt_specs(p.steps, {turn_key: turn_spec})
        procs.append(p)
        pb = Proc(PB, "fork", [5], env = {"EXL3_DISK_CACHE_UNCACHE": "1"})
        pb.add(gen_step("g0", 5, switches = {"tail": True}, disk = "s5"))
        pb.add(run_step("s5.t2.restart", 5, [jobspec("y", turn_key, forced = B)], switches = tail, forget = False))
        pb.prompts = prompt_specs(pb.steps, {turn_key: turn_spec})
        procs.append(pb)
        comps += [
            # The tail boundary cuts a draft as any checkpoint does (Job.is_checkpoint_boundary):
            # with the switch on the drafted positions are others, so only the logits are judged
            cmp(5, "s5.t1.on", f"risposta imposta ({args.s5_answer} token): checkpoint di coda acceso contro spento",
                ref(PA, "s5.t1.on", "x"), ref(PA, "s5.t1.off1", "x"),
                floor = (ref(PA, "s5.t1.off1", "x"), ref(PA, "s5.t1.off2", "x")), drafts = False,
                requires = {"tail_stashes": True}),
            cmp(5, "s5.t1.disk", "la stessa col tier su disco acceso, contro senza",
                ref(PA, "s5.t1.disk", "x"), ref(PA, "s5.t1.on", "x"),
                floor = (ref(PA, "s5.t1.off1", "x"), ref(PA, "s5.t1.off2", "x"))),
            cmp(5, "s5.t2.full", "turno dopo da capo, ripetuto (pezzi di 4096)",
                ref(PA, "s5.t2.full.ref2", "y"), ref(PA, "s5.t2.full.free", "y"), floor = None),
            cmp(5, "s5.t2.tail", "turno dopo ripreso dal checkpoint di coda (in RAM) contro da capo",
                ref(PA, "s5.t2.tail", "y"), ref(PA, "s5.t2.full.free", "y"), expect = "noise",
                yards = [(ref(PA, "s5.t2.full.free", "y"), ref(PA, "s5.t2.up", "y")),
                         (ref(PA, "s5.t2.full.free", "y"), ref(PA, "s5.t2.full.c2048", "y"))],
                drafts = False, requires = {"resumed_from_tail": True}),
            cmp(5, "s5.t2.up", "metro: turno dopo ripreso come upstream (ultima pagina del prompt) contro da capo",
                ref(PA, "s5.t2.up", "y"), ref(PA, "s5.t2.full.free", "y"), expect = "info"),
            cmp(5, "s5.t2.c2048", "metro: turno dopo da capo con pezzi di 2048 contro 4096",
                ref(PA, "s5.t2.full.c2048", "y"), ref(PA, "s5.t2.full.free", "y"), expect = "info"),
            cmp(5, "s5.t2.restart", "turno dopo ripreso dal checkpoint di coda letto dal disco in un processo nuovo, contro lo stesso ripreso dalla RAM",
                ref(PB, "s5.t2.restart", "y"), ref(PA, "s5.t2.tail", "y"), floor = None,
                requires = {"resumed_from_tail": True, "same_resume": ref(PA, "s5.t2.tail", "y")}),
            {"kind": "resume", "scenario": 5, "id": "s5.resume",
             "runs": [("ripreso dal checkpoint di coda (RAM)", ref(PA, "s5.t2.tail", "y")),
                      ("ripreso dal checkpoint di coda dopo il riavvio (disco)", ref(PB, "s5.t2.restart", "y")),
                      ("ripreso come upstream", ref(PA, "s5.t2.up", "y")),
                      ("da capo", ref(PA, "s5.t2.full.free", "y"))]},
        ]

    # ---------------------------------------------------------------- scenario 7
    if 7 in sc:
        P = "s7-fork"
        sargs = ["--moe-swap-prefill-weight", "0.004"]
        if args.pinned_arena:
            sargs += ["--moe-pinned-arena", "true"]
        p = Proc(P, "fork", [7], env = {"EXL3_DISK_CACHE_UNCACHE": "1"}, server_args = sargs, sweeps = "server")
        p.add(gen_step("g0", 7, switches = {"stagger": True, "thread": True, "pipeline": True, "tail": True}, disk = "s7"))
        p.add({"op": "stress", "id": "s7.stress", "scenario": 7, "minutes": args.s7_minutes, "seed": args.seed,
               "step_limit": args.s7_step_limit, "drain_timeout": args.s7_drain_timeout})
        p.add({"op": "close", "id": "s7.close", "scenario": 7})
        p.prompts = {}
        procs.append(p)
        comps.append({"kind": "stress", "scenario": 7, "id": "s7.stress", "run": ref(P, "s7.stress", ""), "close": ref(P, "s7.close", "")})

    return {"procs": [pr.as_dict() for pr in procs], "comparisons": comps, "scenarios": sorted(sc)}


# --------------------------------------------------------------------------------------------
#   Time estimate
# --------------------------------------------------------------------------------------------

QUESTION_TOKENS = 64        # a question in the chat template, for the estimate only


def prompt_len(key: str, prompts: dict, lengths: dict) -> int:
    spec = prompts.get(key) or prompt_spec_of(key)
    if spec["kind"] == "question":
        return QUESTION_TOKENS
    if spec["kind"] == "doc":
        return spec["length"]
    if spec["kind"] == "turn":
        a = spec.get("answer") or {}
        answer = lengths.get((a.get("proc"), a.get("run"), a.get("job")), 1100)
        return prompt_len(spec["base"], prompts, lengths) + answer + 12
    return 0


def estimate_plan(plan: dict, args) -> dict:
    """
    Seconds per scenario and process, from a few rates (args.assume_*): prompt ingestion at one
    rate whatever runs beside it, n jobs generating together at decode * sqrt(n) tokens per
    second in total, every process a model load, a few seconds per run for the rest. A model,
    not a measure: the report says how long things really took.
    """
    pre = args.assume_prefill
    dec = args.assume_decode
    out = {"procs": {}, "scenarios": {}, "total": 0.0, "runs": {}, "ingest": {}, "generate": {}}
    lengths = {}            # (proc, run, job) -> tokens it generates, for forced references
    for proc in plan["procs"]:
        name = proc["name"]
        secs = {s: 0.0 for s in proc["scenarios"]}
        load = args.assume_load_min * 60.0
        for st in proc["steps"]:
            s = st.get("scenario")
            op = st["op"]
            t = 0.0
            if op == "run":
                jobs = st["jobs"]
                each = dec / math.sqrt(max(len(jobs), 1))       # tokens per second of one job
                own, ingest = {}, {}
                for j in jobs:
                    if "forced" in j:
                        f = j["forced"]
                        n = lengths.get((f["proc"], f["run"], f["job"]), args.tokens)
                        if f.get("limit"):
                            n = min(n, f["limit"])
                    else:
                        n = j["max_new"]
                    own[j["name"]] = n
                    plen = prompt_len(j["prompt"], proc["prompts"], lengths)
                    if not st.get("forget", True) and (proc["prompts"].get(j["prompt"]) or {}).get("kind") == "turn":
                        plen = min(plen, 1400)      # resumed from a checkpoint: the rest only
                    ingest[j["name"]] = plen
                until = st.get("until") or {}
                if until.get("job"):
                    # The jobs beside it go on until it is done: its prompt and tokens decide
                    who = until["job"]
                    lead = args.lead / dec if any(j.get("start") for j in jobs if j["name"] == who) else 0.0
                    t = lead + sum(ingest.values()) / pre + own[who] / each
                    for j in jobs:
                        if "forced" not in j and j["name"] != who:
                            own[j["name"]] = min(j["max_new"], int(t * each))
                elif st.get("after_cancel"):
                    t = args.lead / dec + 0.5 * sum(ingest.values()) / pre + st["after_cancel"]["tokens"] / each
                    for j in jobs:
                        own[j["name"]] = min(own[j["name"]], int(t * each))
                else:
                    t = sum(ingest.values()) / pre + max(own.values(), default = 0) / each
                for j in jobs:
                    lengths[(name, st["id"], j["name"])] = own[j["name"]]
                t += 3.0
                out["ingest"][s] = out["ingest"].get(s, 0) + sum(ingest.values())
                out["generate"][s] = out["generate"].get(s, 0) + sum(own.values())
                out["runs"][s] = out["runs"].get(s, 0) + 1
            elif op == "generator":
                t = 5.0
            elif op == "disk_scan":
                t = 20.0
            elif op == "sweep":
                t = 10.0
            elif op == "close":
                t = 20.0
            elif op == "stress":
                t = st["minutes"] * 60.0 + 180.0
            elif op == "reset_hist":
                t = 1.0
            secs[s] = secs.get(s, 0.0) + t
        # The load is charged to the process's first scenario
        first = proc["scenarios"][0]
        secs[first] = secs.get(first, 0.0) + load
        out["procs"][name] = {"load": load, "scenarios": secs, "total": sum(secs.values())}
        for s, v in secs.items():
            out["scenarios"][s] = out["scenarios"].get(s, 0.0) + v
        out["total"] += sum(secs.values())
    return out


def scenario_budgets(estimate: dict, args) -> dict:
    """Wall-clock budget per scenario, seconds: --scenario-timeout (minutes) when given, else
    three times the estimate and at least 30 minutes."""
    out = {}
    for s, v in estimate["scenarios"].items():
        if args.scenario_timeout:
            out[s] = args.scenario_timeout * 60.0
        else:
            out[s] = max(1800.0, 3.0 * v)
    return out


# --------------------------------------------------------------------------------------------
#   Comparisons
# --------------------------------------------------------------------------------------------

IDENT = "identico"
NOISE = "entro il rumore"
DIFF = "**DIVERSO**"
ERR = "errore"
INFO = "informativo"
PASS = "superato"
HANG = "**BLOCCO**"
FAIL = "**FALLITO**"
NOTRUN = "non eseguito"

SEVERITY = [DIFF, HANG, FAIL, ERR, NOISE, IDENT, PASS]


def worst(verdicts: list) -> str:
    vs = [v for v in verdicts if v not in (INFO, None)]
    if not vs:
        return NOTRUN
    for v in SEVERITY:
        if v in vs:
            return v
    return vs[0]


def job_view(rec: dict | None, job: str) -> dict | None:
    if rec is None:
        return None
    return (rec.get("jobs") or {}).get(job)


def tokens_at(rec: dict, name: str, t: float) -> int:
    """Tokens `name` had generated by time t (after the last iteration that ended by then)."""
    order = rec.get("order") or []
    if name not in order:
        return 0
    k = order.index(name)
    n = 0
    for when, counts in rec.get("timeline") or []:
        if when > t:
            break
        n = counts[k]
    return n


def tokens_before(rec: dict, name: str, t: float) -> int:
    """Tokens `name` had generated before the iteration that ended at time t."""
    order = rec.get("order") or []
    if name not in order:
        return 0
    k = order.index(name)
    n = 0
    for when, counts in rec.get("timeline") or []:
        if when >= t:
            break
        n = counts[k]
    return n


def first_sample_time(rec: dict, name: str):
    """End time of the first iteration in which `name` had sampled a token (the generator's own
    count, Job.new_tokens, which the timeline holds); None if it never did. An emitted token can
    come later than its sample (text held for an incomplete character), so the timeline, not the
    first emitted token, says when a job joined the decode batch."""
    order = rec.get("order") or []
    if name not in order:
        return None
    k = order.index(name)
    for when, counts in rec.get("timeline") or []:
        if counts[k] >= 1:
            return when
    return None


def ingestion_window(rec: dict, reader: str):
    """(enqueue time, end of the iteration of its first sampled token) of `reader` in a run,
    None when it got no token."""
    j = job_view(rec, reader)
    if not j or j.get("enqueued") is None:
        return None
    end = first_sample_time(rec, reader)
    if end is None:
        end = j.get("first_token")
    if end is None:
        return None
    return j["enqueued"], end


def first_end_time(rec: dict):
    """End time of the iteration in which the first job of a run ended (eos), None if none did."""
    ends = [j.get("ended") for j in (rec.get("jobs") or {}).values() if j.get("ended") is not None]
    return min(ends) if ends else None


def _drafts_upto(v: dict, n: int):
    """The job's draft counters (accepted, rejected) as they were when position n was sampled,
    from the per-position record (Digester); None when the record does not have them."""
    acc, rej = v.get("acc"), v.get("rej")
    if not isinstance(acc, list) or not isinstance(rej, list) or len(acc) < n or len(rej) < n:
        return None
    return list(zip(acc[:n], rej[:n]))


def compare_jobs(refv: dict, testv: dict, upto: int | None = None) -> dict | None:
    """
    One job of a test run against the same job of a reference run (the same forced tokens),
    position by position over the first `upto` positions: whether the logits are the same bits
    (digest), the log-probability of the forced token, the most likely tokens, and the MTP
    head's accepted and rejected drafts up to each position.
    """
    if refv is None or testv is None:
        return None
    keys = ("tokens", "logp", "digest", "topi", "topl")
    n = min(len(refv.get(k) or []) for k in keys)
    n = min(n, min(len(testv.get(k) or []) for k in keys))
    if upto is not None:
        n = min(n, upto)
    if n <= 0:
        return None
    rt, tt = refv["tokens"][:n], testv["tokens"][:n]
    tokens_ok = rt == tt
    nonfinite = 0
    deltas = []
    same = []
    flips = 0
    digests = True
    for i in range(n):
        a, b = refv["logp"][i], testv["logp"][i]
        finite = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
        if not finite:
            nonfinite += 1
        else:
            deltas.append(abs(a - b))
        da, db = refv["digest"][i], testv["digest"][i]
        if da is None or db is None:
            digests = False
            dig_same = True
        else:
            dig_same = da == db
        ok = finite and dig_same and a == b and refv["topi"][i] == testv["topi"][i] and refv["topl"][i] == testv["topl"][i]
        same.append(ok)
        if (refv["topi"][i] or [None])[0] != (testv["topi"][i] or [None])[0]:
            flips += 1
    first = next((i for i, s in enumerate(same) if not s), None)
    # Drafts: position by position where both records have the counters; otherwise the job's
    # totals, which say something only when every position of both jobs is compared
    dt, dr = _drafts_upto(testv, n), _drafts_upto(refv, n)
    if dt is not None and dr is not None:
        drafts_same = dt == dr
        drafts, drafts_ref = (dt[-1] if dt else (0, 0)), (dr[-1] if dr else (0, 0))
    else:
        drafts, drafts_ref = (testv.get("accepted"), testv.get("rejected")), (refv.get("accepted"), refv.get("rejected"))
        whole = n == len(testv.get("tokens") or []) == len(refv.get("tokens") or [])
        drafts_same = (drafts == drafts_ref) if whole else None
    return {
        "n": n,
        "same": sum(same),
        "first_diff": first,
        "max": max(deltas) if deltas else (0.0 if not nonfinite else None),
        "mean": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "flips": flips,
        "nonfinite": nonfinite,
        "tokens_ok": tokens_ok,
        "digests": digests,
        "identical": first is None and nonfinite == 0,
        "drafts": drafts,
        "drafts_ref": drafts_ref,
        "drafts_same": drafts_same,
    }


def within(test: dict, floor: dict) -> bool:
    """A comparison is within a floor that moves when its largest shift is at most twice the
    floor's and its changed top tokens at most twice the floor's plus two (the old benches' rule,
    on log-probabilities here)."""
    if floor is None or floor.get("max") is None or test.get("max") is None:
        return False
    return test["max"] <= 2.0 * floor["max"] and test["flips"] <= 2 * floor["flips"] + 2


class Results:
    """What the processes saved: proc -> run id -> record, and the processes' own outcome."""

    def __init__(self):
        self.records = {}       # proc -> {run id: record}
        self.procs = {}         # proc -> outcome dict (exit, reason, load_s, wall_s, ...)
        self.meta = {}          # proc -> the child's "start" record (library, versions, devices)

    def rec(self, r: dict | None) -> dict | None:
        if r is None:
            return None
        return self.records.get(r["proc"], {}).get(r["run"])

    def load_proc(self, name: str, path: str):
        recs = {}
        for obj in read_jsonl(path):
            if obj.get("op") == "start":
                self.meta[name] = obj
            if obj.get("id"):
                recs[obj["id"]] = obj
        self.records[name] = recs


def requirement_problems(req: dict, rec: dict, job: str, res: Results) -> list:
    """What the test run had to exercise and did not (each a reason to call the row an error,
    or, for the soft requirements, a note on it)."""
    out = []
    state = rec.get("state") or {}
    stats = rec.get("stats") or {}
    side = stats.get("side") or {}
    if req.get("thread") and state.get("thread") != req["thread"]:
        out.append(f"thread di prefill «{state.get('thread')}» ({state.get('thread_reason') or '-'})")
    if req.get("pipeline") and state.get("pipeline") != req["pipeline"]:
        out.append(f"pezzi in volo «{state.get('pipeline')}» ({state.get('pipeline_reason') or '-'})")
    if req.get("piped") and not side.get("piped"):
        out.append("nessun pezzo partito con un altro in volo")
    if req.get("ahead") and not side.get("ahead"):
        out.append("nessun pezzo pianificato in anticipo")
    if req.get("beside") and not side.get("beside"):
        out.append("nessun pezzo è girato accanto al decode")
    if req.get("all_beside") and side.get("chunks") and (side.get("beside") or 0) < side["chunks"]:
        out.append(f"{side['chunks'] - (side.get('beside') or 0)} dei {side['chunks']} pezzi sul thread sono stati attesi "
                   "senza decode accanto (l'ultimo pezzo corto non è stato messo alla prova accanto al decode)")
    if "head_first" in req and state.get("head_first") is not None and bool(state.get("head_first")) != bool(req["head_first"]):
        if req["head_first"]:
            out.append("l'aggiornamento della testa MTP non ha fatto il suo turno sulla prima scheda (testa non sulla "
                       "prima scheda?): il confronto con EXL3_PREFILL_PIPELINE_HEAD=second confronta la stessa cosa")
        else:
            out.append("l'aggiornamento della testa MTP ha fatto il suo turno sulla prima scheda nonostante "
                       "EXL3_PREFILL_PIPELINE_HEAD=second")
    if req.get("stagger_calls") and not (stats.get("stagger") or {}).get("steps"):
        out.append("nessuna chiamata sfalsata")
    if req.get("tail_stashes") and not (stats.get("tail") or {}).get("stashes"):
        out.append("nessun checkpoint di coda preso")
    j = job_view(rec, job) or {}
    if req.get("resumed") and not j.get("cached_tokens"):
        out.append("nessun token ripreso dalla cache")
    if req.get("resumed_from_tail") and not j.get("resumed_from_tail"):
        out.append(f"non ripreso da un checkpoint di coda (ripresi {j.get('cached_tokens')} token)")
    same = req.get("same_resume")
    if same:
        other = job_view(res.rec(same), same["job"]) or {}
        if other.get("cached_tokens") != j.get("cached_tokens"):
            out.append(f"ripreso a {j.get('cached_tokens')} token, il riferimento a {other.get('cached_tokens')}")
    return out


def evaluate_probs(c: dict, res: Results) -> dict:
    row = {"id": c["id"], "scenario": c["scenario"], "label": c["label"], "expect": c["expect"], "notes": []}
    t_rec, r_rec = res.rec(c["test"]), res.rec(c["ref"])
    missing = [what for what, rr, rf in (("prova", t_rec, c["test"]), ("riferimento", r_rec, c["ref"])) if rr is None]
    if missing:
        row.update(verdict = ERR, why = "manca il giro di " + " e di ".join(missing))
        return row
    for what, rr in (("prova", t_rec), ("riferimento", r_rec)):
        if not rr.get("ok", False):
            row.update(verdict = ERR, why = f"il giro di {what} è fallito: {rr.get('error') or '?'}")
            return row
    upto = c.get("upto")
    window = None
    if isinstance(upto, dict) and upto.get("window"):
        w = ingestion_window(t_rec, upto["window"])
        if w is None:
            row.update(verdict = ERR, why = f"`{upto['window']}` non ha dato token nel giro di prova")
            return row
        inside = tokens_before(t_rec, c["test"]["job"], w[1])
        started = tokens_at(t_rec, c["test"]["job"], w[0])
        window = (started, inside)
        if inside <= started:
            row.update(verdict = ERR, why = "nessuna posizione generata durante l'ingestione", window = window)
            return row
        upto = inside
    elif isinstance(upto, dict) and upto.get("before_first_end"):
        t_end = first_end_time(t_rec)
        upto = None if t_end is None else tokens_before(t_rec, c["test"]["job"], t_end)
        if upto is not None and upto <= 0:
            row.update(verdict = ERR, why = "un job è finito prima che si potesse confrontare una posizione")
            return row
    test = compare_jobs(job_view(r_rec, c["ref"]["job"]), job_view(t_rec, c["test"]["job"]), upto)
    if test is None:
        row.update(verdict = ERR, why = "nessuna posizione registrata")
        return row
    row.update(test = test, window = window)

    def pair(p):
        if p is None:
            return None
        a, b = res.rec(p[0]), res.rec(p[1])
        if a is None or b is None or not a.get("ok") or not b.get("ok"):
            return None
        return compare_jobs(job_view(a, p[0]["job"]), job_view(b, p[1]["job"]), upto)

    floor = pair(c.get("floor"))
    floor2 = pair(c.get("floor2"))
    yards = [y for y in (pair(p) for p in c.get("yards") or []) if y is not None]
    row.update(floor = floor, floor2 = floor2, yards = yards)
    probs = requirement_problems(c.get("requires") or {}, t_rec, c["test"]["job"], res)
    for line in requirement_problems(c.get("soft") or {}, t_rec, c["test"]["job"], res):
        row["notes"].append(f"**Attenzione**: {line}")
    if not test["tokens_ok"]:
        row.update(verdict = ERR, why = "i token usciti non sono quelli imposti: le due sequenze non sono le stesse")
        return row
    if c["expect"] == "info":
        row["verdict"] = INFO
        return row
    if probs:
        row.update(verdict = ERR, why = "la prova non ha esercitato la funzione: " + "; ".join(probs))
        return row
    judge_drafts = bool(c.get("drafts")) and test["drafts_same"] is not None
    if test["nonfinite"]:
        row.update(verdict = DIFF, why = f"valori non finiti in {test['nonfinite']} posizioni")
        return row
    if test["identical"] and (not judge_drafts or test["drafts_same"]):
        row["verdict"] = IDENT
        return row
    if test["identical"]:
        # Same logits everywhere, but the MTP head proposed differently: its cache differs
        fl = floor or floor2
        if fl is not None and fl.get("drafts_same") is False:
            row.update(verdict = NOISE, why = "bozze diverse, come nel giro ripetuto")
        else:
            row.update(verdict = DIFF, why = "stessi logit, bozze accettate/rifiutate diverse")
        return row
    biggest = None
    for y in yards:
        if biggest is None or (y.get("max") or 0) > (biggest.get("max") or 0):
            biggest = y
    if c["expect"] == "noise":
        if biggest is None:
            row.update(verdict = DIFF, why = "manca il metro")
        elif within(test, biggest):
            row["verdict"] = NOISE
        else:
            row.update(verdict = DIFF, why = "oltre il metro")
        return row
    fl = floor if floor is not None else floor2
    # A control that is not identical itself leaves room too, for a reason of upstream's own
    moved = [y for y in yards if not y["identical"]]
    if fl is not None and not fl["identical"] and within(test, fl):
        row["verdict"] = NOISE
    elif moved and any(within(test, y) for y in moved):
        row.update(verdict = NOISE, why = "come il giro di controllo, che differisce anche lui dal riferimento come upstream")
    elif fl is None:
        row.update(verdict = DIFF, why = "manca il giro ripetuto: nessun metro")
    elif fl["identical"]:
        row.update(verdict = DIFF, why = "il giro ripetuto è identico"
                   + (" e la prova è oltre il giro di controllo" if moved else ""))
    else:
        row.update(verdict = DIFF, why = "oltre il rumore del giro ripetuto"
                   + (" e del giro di controllo" if moved else ""))
    return row


def crc_compare(a: dict, b: dict) -> dict:
    """Two disk scans: the same page files (by content hash), the same CRCs, every file intact."""
    pa, pb = a.get("pages") or {}, b.get("pages") or {}
    common = sorted(set(pa) & set(pb))
    differ = [h for h in common if pa[h]["crc_disk"] != pb[h]["crc_disk"]]
    broken = [h for h, v in list(pa.items()) + list(pb.items()) if not v.get("ok")]
    only_a, only_b = sorted(set(pa) - set(pb)), sorted(set(pb) - set(pa))
    # Checkpoints are reported, not judged: which ones reach the disk depends on when the idle
    # flush and the write candidates met their pages, not only on the computation
    sa, sb = a.get("states") or {}, b.get("states") or {}
    s_common = sorted(set(sa) & set(sb))
    s_differ = [h for h in s_common if sa[h].get("crc") != sb[h].get("crc")]
    return {"common": len(common), "differ": len(differ), "broken": len(broken), "only_ref": len(only_a),
            "only_test": len(only_b), "pages_ref": len(pa), "pages_test": len(pb),
            "states_ref": len(sa), "states_test": len(sb), "states_common": len(s_common),
            "states_differ": len(s_differ),
            "bad": len(a.get("bad") or []) + len(b.get("bad") or []),
            "identical": bool(common) and not differ and not broken and not only_a and not only_b}


def evaluate_crc(c: dict, res: Results) -> dict:
    row = {"id": c["id"], "scenario": c["scenario"], "label": c["label"], "expect": "identical", "notes": []}
    ta, tb = res.rec(c["ref"]), res.rec(c["test"])
    if ta is None or tb is None or not ta.get("ok") or not tb.get("ok"):
        row.update(verdict = ERR, why = "manca una delle due letture del disco")
        return row
    m = crc_compare(ta["scan"], tb["scan"])
    row["crc"] = m
    floor = None
    if c.get("floor"):
        fa, fb = res.rec(c["floor"][0]), res.rec(c["floor"][1])
        if fa is not None and fb is not None and fa.get("ok") and fb.get("ok"):
            floor = crc_compare(fa["scan"], fb["scan"])
    row["crc_floor"] = floor
    wb = c.get("write_behind")
    if wb is not None:
        j = job_view(res.rec(wb), wb["job"]) or {}
        at = (j.get("disk_at_first_token") or {}).get("page_writes")
        row["write_behind"] = at
        if not at:
            row["notes"].append("**Attenzione**: nessuna pagina scritta prima del primo token di `r`: la scrittura "
                                "durante l'ingestione non è stata esercitata")
    if m["pages_ref"] == 0:
        row.update(verdict = ERR, why = "nessuna pagina sul disco")
    elif m["identical"]:
        row["verdict"] = IDENT
    elif floor is not None and not floor["identical"]:
        row.update(verdict = NOISE, why = "anche il confronto ripetuto differisce")
    else:
        row.update(verdict = DIFF, why = f"{m['differ']} pagine con CRC diverso, {m['broken']} file rovinati, "
                                         f"{m['only_ref']} + {m['only_test']} pagine solo da una parte")
    return row


def evaluate_placement(c: dict, res: Results) -> dict:
    row = {"id": c["id"], "scenario": c["scenario"], "label": c["label"], "expect": "identical", "notes": []}

    def sweeps(proc):
        recs = res.records.get(proc, {})
        return [recs.get(f"s6.r{k}.sweep") for k in range(1, c["rounds"] + 1)]

    def diff(pa, pb):
        a, b = sweeps(pa), sweeps(pb)
        out = []
        for k, (x, y) in enumerate(zip(a, b), 1):
            if x is None or y is None or not x.get("ok") or not y.get("ok"):
                out.append((k, None))
                continue
            same = x["maps"] == y["maps"] and x["hist_pre"] == y["hist_pre"] and x["moved"] == y["moved"]
            out.append((k, {"same": same, "maps_same": x["maps"] == y["maps"], "hist_same": x["hist_pre"] == y["hist_pre"],
                            "moved": (x.get("total_moved"), y.get("total_moved"))}))
        return out

    test = diff(c["ref"], c["test"])
    row["rounds"] = test
    if not test or any(v is None for _, v in test):
        row.update(verdict = ERR, why = "manca uno sweep")
        return row
    floor = diff(*c["floor"]) if c.get("floor") else None
    row["floor_rounds"] = floor
    if all(v["same"] for _, v in test):
        row["verdict"] = IDENT
    elif floor and all(v is not None for _, v in floor) and not all(v["same"] for _, v in floor):
        row.update(verdict = NOISE, why = "anche stock contro stock differisce")
    else:
        row.update(verdict = DIFF, why = "decisioni diverse")
    return row


def evaluate_tokens(c: dict, res: Results) -> dict:
    row = {"id": c["id"], "scenario": c["scenario"], "label": c["label"], "expect": "info", "verdict": INFO, "notes": []}
    a = job_view(res.rec(c["ref"]), c["ref"]["job"])
    b = job_view(res.rec(c["test"]), c["test"]["job"])
    if a is None or b is None:
        row["why"] = "manca un giro"
        return row
    ta, tb = a.get("tokens") or [], b.get("tokens") or []
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if first is None and len(ta) != len(tb):
        first = min(len(ta), len(tb))
    row["why"] = "uguali" if first is None else f"si separano alla posizione {first} di {min(len(ta), len(tb))}"
    return row


def evaluate_cancel(c: dict, res: Results) -> dict:
    row = {"id": c["id"], "scenario": c["scenario"], "label": "annullamento a metà di un'ingestione in volo", "expect": "identical", "notes": []}
    rec = res.rec(c["run"])
    if rec is None:
        row.update(verdict = ERR, why = "manca il giro")
        return row
    cancels = rec.get("cancels") or []
    row["cancels"] = cancels
    if not rec.get("ok"):
        row.update(verdict = FAIL, why = f"il giro è fallito: {rec.get('error')}")
    elif not cancels:
        row.update(verdict = ERR, why = "l'annullamento non è avvenuto (l'ingestione è finita prima?)")
    elif rec.get("left_jobs"):
        row.update(verdict = FAIL, why = f"{rec['left_jobs']} job rimasti nella coda")
    else:
        row["verdict"] = PASS
        row["why"] = ", ".join(f"`{x['job']}` annullato al {100 * x['progress']:.0f}% in {x['seconds']:.2f} s" for x in cancels)
    return row


def evaluate_stress(c: dict, res: Results) -> dict:
    row = {"id": c["id"], "scenario": c["scenario"], "label": "traffico concorrente con ogni interruttore di produzione acceso",
           "expect": "identical", "notes": []}
    rec = res.rec(c["run"])
    outcome = res.procs.get(c["run"]["proc"]) or {}
    if rec is None:
        if outcome.get("hang"):
            row.update(verdict = HANG, why = outcome.get("reason") or "nessun ritorno da una chiamata")
        else:
            row.update(verdict = ERR, why = outcome.get("reason") or "manca il giro")
        return row
    s = rec.get("stress") or {}
    row["stress"] = s
    close = res.rec(c.get("close"))
    problems = []
    if s.get("exceptions"):
        problems.append(f"{len(s['exceptions'])} eccezioni: " + "; ".join(s["exceptions"][:3]))
    if s.get("job_errors"):
        problems.append(f"{s['job_errors']} job falliti")
    if not s.get("drained"):
        problems.append("la coda non si è svuotata")
    if s.get("max_step_s") is not None and s.get("step_limit") and s["max_step_s"] > s["step_limit"]:
        problems.append(f"un passo di {s['max_step_s']:.1f} s, oltre {s['step_limit']} s")
    if close is not None and not close.get("ok"):
        problems.append(f"close() fallita: {close.get('error')}")
    if s.get("hang") or outcome.get("hang"):
        row.update(verdict = HANG, why = s.get("hang") or outcome.get("reason"))
    elif problems:
        row.update(verdict = FAIL, why = "; ".join(problems))
    else:
        row["verdict"] = PASS
    return row


def evaluate_resume(c: dict, res: Results) -> dict:
    row = {"id": c["id"], "scenario": c["scenario"], "label": "da dove riparte il turno dopo", "expect": "info", "verdict": INFO, "notes": []}
    rows = []
    for label, r in c["runs"]:
        j = job_view(res.rec(r), r["job"]) or {}
        prompt, cached = j.get("prompt_tokens"), j.get("cached_tokens")
        rows.append([label, prompt if prompt is not None else "-", cached if cached is not None else "-",
                     (prompt - cached) if prompt is not None and cached is not None else "-",
                     {True: "sì", False: "no"}.get(j.get("resumed_from_tail"), "-"),
                     fmt(j.get("ttft"), 2)])
    row["table"] = rows
    return row


EVALUATORS = {
    "probs": evaluate_probs, "crc": evaluate_crc, "placement": evaluate_placement, "tokens": evaluate_tokens,
    "cancel": evaluate_cancel, "stress": evaluate_stress, "resume": evaluate_resume,
}


def evaluate_all(plan: dict, res: Results) -> list:
    """One row per comparison of the plan (the capture checks are capture_checks' business)."""
    rows = []
    for c in plan["comparisons"]:
        kind = c.get("kind", "probs")
        fn = EVALUATORS.get(kind)
        if fn is None:
            continue
        try:
            row = fn(c, res)
        except Exception as e:      # a broken record must not take the report down
            row = {"id": c.get("id"), "scenario": c.get("scenario"), "label": c.get("label", c.get("id")),
                   "expect": c.get("expect", "identical"), "verdict": ERR, "why": f"confronto non calcolabile: {e!r}", "notes": []}
        row["kind"] = kind
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------
#   Capture checks (the "Attenzione" lines)
# --------------------------------------------------------------------------------------------

SWITCHES_OFF = {
    "stagger_decode": False, "prefill_thread": False, "prefill_pipeline": False,
    "tail_checkpoint_interval": 0, "hist_prefill_weight": 1.0, "disk_uncache": False,
    "mtp_batched_verify": False, "mtp_batched_update": False,
}


def switch_is_off(name: str, value) -> bool:
    """Whether a generator's switch, as the child read it, is at upstream's behaviour. By type:
    in Python True == 1 == 1.0, and a switch left on must not pass for a weight of one."""
    if value is None:
        return True
    want = SWITCHES_OFF.get(name)
    if isinstance(want, bool) or isinstance(value, bool):
        return value is False
    try:
        return float(value) == float(want)
    except (TypeError, ValueError):
        return False


def capture_checks(plan: dict, res: Results) -> dict:
    """scenario -> list of Attenzione lines about the capture itself, never about the results."""
    out = {}

    def note(s, text):
        out.setdefault(s, []).append(f"**Attenzione**: {text}")

    for proc in plan["procs"]:
        name = proc["name"]
        o = res.procs.get(name)
        s0 = proc["scenarios"][0]
        if o is None:
            continue
        if o.get("skipped"):
            note(s0, f"il processo `{name}` non è partito: {o['skipped']}")
            continue
        if o.get("exit") not in (0, None) or o.get("killed"):
            tail = o.get("log_tail") or ""
            note(s0, f"il processo `{name}` si è fermato ({o.get('reason') or 'uscita ' + str(o.get('exit'))})"
                 + (f"; ultime righe del suo log: `{tail}`" if tail else ""))
        meta = res.meta.get(name) or {}
        if meta.get("library_problem"):
            note(s0, f"`{name}`: {meta['library_problem']}")
        recs = res.records.get(name, {})
        for rid, rec in recs.items():
            sc_of = rec.get("scenario") or s0
            if rec.get("op") in ("load", "run", "disk_scan", "sweep", "close", "generator", "reset_hist", "stress") \
                    and rec.get("ok") is False:
                note(sc_of, f"`{name}` / `{rid}` è fallito: {rec.get('error')}")
            for msg in rec.get("warnings") or []:
                note(sc_of, f"`{name}` / `{rid}`: {msg}")
            if rec.get("op") == "run" and proc.get("sweeps") == "frozen" and rec.get("placement_before") \
                    and rec.get("placement_after") and rec["placement_before"] != rec["placement_after"]:
                note(sc_of, f"`{name}` / `{rid}`: il piazzamento degli esperti è cambiato durante il giro")
            # A run that dropped the checkpoints first must ingest every prompt from its first
            # token: a job that resumed means the reference and the test did not compute the same
            if rec.get("op") == "run" and rec.get("ok") and "forgot" in rec:
                for jn, jv in (rec.get("jobs") or {}).items():
                    if jv.get("cached_tokens"):
                        note(sc_of, f"`{name}` / `{rid}`: `{jn}` ha ripreso {jv['cached_tokens']} token dalla cache in un "
                                    "giro che doveva ingerire i prompt da capo")
    for c in plan["comparisons"]:
        kind = c.get("kind")
        if kind == "same_placement":
            hashes = {}
            where = c.get("record", "load")
            for pname in c["procs"]:
                loaded = (res.records.get(pname, {}).get(where) or {}).get("placement")
                if loaded:
                    hashes[pname] = loaded
            if len(set(hashes.values())) > 1:
                note(c["scenario"], f"il piazzamento ({'al caricamento' if where == 'load' else 'all inizio del traffico'}) "
                     "non è lo stesso nei processi ("
                     + ", ".join(f"`{k}` {v}" for k, v in hashes.items()) + "): i confronti tra processi non valgono")
        elif kind == "switches_off":
            for rid, rec in sorted(res.records.get(c["proc"], {}).items()):
                if rec.get("op") != "generator":
                    continue
                on = [k for k, v in (rec.get("switches_state") or {}).items() if not switch_is_off(k, v)]
                if on:
                    note(c["scenario"], f"nel fork del confronto con stock non tutti gli interruttori sono spenti "
                                        f"(`{rid}`): {', '.join(on)}")
        elif kind == "library":
            for pname, want in c["procs"].items():
                meta = res.meta.get(pname)
                if meta is None:
                    continue
                got = "fork" if meta.get("fork_features") else "stock"
                if got != want:
                    note(c["scenario"], f"`{pname}` doveva girare su {want} e ha importato {got} da `{meta.get('library_path')}`")
    return out


# --------------------------------------------------------------------------------------------
#   Report
# --------------------------------------------------------------------------------------------

EXPECT_TEXT = {"identical": "identico", "noise": "entro il rumore", "info": "-"}


def _cmp_text(v: dict | None) -> str:
    if v is None:
        return "-"
    return "identico" if v["identical"] else f"max {fmt_g(v['max'])}, top-1 {v['flips']}"


def floor_text(row: dict) -> str:
    """The yardstick of a row: the run repeated, or for a row expected within the noise the
    widest of its yardsticks; a control run, where there is one, after it."""
    fl = row.get("floor") or row.get("floor2")
    ys = row.get("yards") or []
    widest = max(ys, key = lambda v: v.get("max") or 0) if ys else None
    if row.get("expect") == "noise":
        return _cmp_text(widest)
    text = _cmp_text(fl)
    if widest is not None:
        text += f"; controllo: {_cmp_text(widest)}"
    return text


def probs_table(rows: list) -> str:
    out = []
    for r in rows:
        t = r.get("test")
        win = r.get("window")
        pos = "-" if t is None else (f"{t['n']}" + (f" ({win[1] - win[0]} durante l'ingestione)" if win else ""))
        drafts = "-"
        if t is not None and t["drafts"][0] is not None:
            drafts = f"{t['drafts'][0]}/{t['drafts'][1]} contro {t['drafts_ref'][0]}/{t['drafts_ref'][1]}"
        verdict = r["verdict"] + (f" ({r['why']})" if r.get("why") and r["verdict"] != IDENT else "")
        out.append([
            r["label"], EXPECT_TEXT.get(r.get("expect"), "-"), pos,
            "-" if t is None else f"{t['same']}" + ("" if t["digests"] else " (senza impronta)"),
            "-" if t is None or t["first_diff"] is None else t["first_diff"],
            "-" if t is None else fmt_g(t["max"]),
            "-" if t is None else fmt_g(t["mean"]),
            "-" if t is None else t["flips"],
            drafts, floor_text(r), verdict,
        ])
    return table(["confronto", "atteso", "posizioni", "identiche", "prima diversa", "max |Δ log p|", "media |Δ log p|",
                  "top-1 cambiati", "bozze acc./rif. (prova contro riferimento)", "metro", "esito"], out)


def summary_row(s: int, rows: list, notes: list) -> list:
    rel = [r for r in rows if r.get("scenario") == s and r.get("verdict") is not None]
    verdict = worst([r["verdict"] for r in rel])
    if verdict == NOTRUN and any(n for n in notes):
        verdict = ERR
    tested = [r for r in rel if r.get("verdict") not in (INFO,) and r.get("test")]
    mx = None
    pos = 0
    for r in tested:
        t = r["test"]
        pos += t["n"]
        if t.get("max") is not None:
            mx = t["max"] if mx is None else max(mx, t["max"])
    floors = [r.get("floor") or r.get("floor2") for r in tested if (r.get("floor") or r.get("floor2"))]
    if not floors:
        ftxt = "-"
    elif all(f["identical"] for f in floors):
        ftxt = "identico"
    else:
        ftxt = f"max {fmt_g(max((f.get('max') or 0) for f in floors))}"
    return [f"{s}. {SCENARIO_TITLES[s]}", verdict, fmt_g(mx) if tested else "-", pos if tested else "-", ftxt]


SCENARIO_TEXT = {
    1: f"Tre processi con lo stesso carico di lavoro, uno per caricamento del modello: upstream {BASE_RELEASE} dalla wheel installata "
       "(`stock-a`), il fork con ogni interruttore spento (`fork`) e di nuovo upstream (`stock-b`). I token imposti sono quelli "
       "che `stock-a` ha prodotto da solo. Il metro di ogni confronto è `stock-a` contro `stock-b`: due processi della stessa "
       "libreria. Per il tier su disco si confrontano anche i file delle pagine scritti, per CRC.",
    2: "Ogni job viene rigenerato coi token che ha dato da solo, nel decode sfalsato e in un batch normale, e confrontato col "
       "suo gruppo (i job con cui il decode sfalsato lo mette in un passaggio) fatto girare da solo. Si confrontano le "
       "posizioni campionate prima dell'iterazione in cui finisce il primo job del giro: da lì il decode sfalsato rifà i "
       "gruppi (o torna a un batch solo) e il gruppo da solo no. Le righe «in un batch normale» dicono se il confronto vede "
       "la differenza di composizione del batch: sono informative.",
    3: "Per ogni caso un documento `r` viene accodato mentre i job generano: ingerito come upstream (un passo di decode per "
       "pezzo) e sul thread di prefill. Si confrontano `r` sul thread contro `r` come upstream accanto agli stessi job, e i "
       "job che generano contro gli stessi job senza nessun prompt accanto, sulle posizioni generate prima che `r` entri "
       "nel batch. Nei casi «ultimo pezzo di N token» il documento è lungo 32 pagine più N + 1 token: il suo ultimo "
       "passaggio di prefill è di N token. Il controllo di ogni caso è `r` accodato più tardi, come upstream: entra nel "
       "batch con i job più avanti nel loro testo, come dopo un'ingestione sul thread; se differisce dal riferimento, la "
       "riga di `r` sul thread si giudica col suo rumore (la differenza viene da cosa tengono le altre righe del batch, "
       "non dal thread). Con 5 job che generano (10 righe di decode con MTP 1, oltre le 8 di `side_max_rows`) i pezzi "
       "vengono attesi come in linea: il confronto vale lo stesso.",
    4: "Un documento di più pezzi ingerito con due pezzi in volo sulle due schede, contro lo stesso ingerito in linea; "
       "accanto a un job che genera; dopo un annullamento a metà ingestione; con la testa MTP aggiornata dal thread della "
       "seconda metà (`EXL3_PREFILL_PIPELINE_HEAD=second`); col tier su disco di upstream acceso (i file delle pagine "
       "scritti durante l'ingestione contro quelli dell'ingestione in linea, poi riletti da un generatore nuovo); e un "
       "prompt di circa 179k token in un processo a sé, con la memoria delle schede. `r` accanto ad `a` ha il controllo "
       "dello scenario 3 (`r` accodato più tardi, in linea).",
    5: "Una risposta lunga imposta con e senza il checkpoint che segue la risposta (atteso identico: lo stash è una copia); "
       "il turno dopo ripreso da quel checkpoint contro lo stesso turno ingerito da capo (atteso solo entro il rumore: chi "
       "riprende ha calcolato la risposta un token alla volta e legge il resto in un passaggio solo, chi parte da capo legge "
       "tutto a pezzi; il metro è quanto differisce da capo il turno ripreso come fa upstream, e il turno da capo con pezzi di "
       "2048); lo stesso turno ripreso dopo un riavvio, col checkpoint letto dal disco da un secondo processo (atteso identico "
       "al ripreso dalla RAM).",
    6: "Negli stessi tre processi dello scenario 1, dopo di esso: i contatori delle selezioni degli esperti azzerati, poi "
       "giri di traffico imposto (tre job e un documento) ciascuno seguito da uno sweep del piazzamento della libreria. Si "
       "confrontano, giro per giro, le mappe di piazzamento dopo lo sweep, gli istogrammi prima dello sweep e gli scambi.",
    7: "Traffico concorrente generato da un seme: domande brevi, documenti di 8-33k token, qualche prompt di 64-96k, turni "
       "successivi e annullamenti (di job in coda, in ingestione o che generano), con "
       "decode sfalsato, thread di prefill, pezzi in volo, checkpoint di coda, tier su disco con `EXL3_DISK_CACHE_UNCACHE=1`, "
       "MTP 1, peso del prefill 0.004 e sweep del piazzamento come in produzione. Superato: nessuna chiamata della libreria "
       "oltre il cane da guardia, nessuna eccezione, la coda si svuota e `close()` finisce.",
}


def render_report(plan: dict, res: Results, meta: dict, finished: bool) -> str:
    rows = evaluate_all(plan, res)
    notes = capture_checks(plan, res)
    sc = plan["scenarios"]
    out = [
        f"# Controllo di identità bit a bit: fork {BASE_RELEASE} contro upstream",
        "",
        f"{meta.get('started', '')} - " + (meta.get("ended", "") if finished else "in corso (rapporto parziale)"),
        "",
        "## Sintesi",
        "",
        table(["scenario", "esito", "max |Δ log p|", "posizioni confrontate", "rumore di fondo"],
              [summary_row(s, rows, notes.get(s, [])) for s in sc]),
        "",
        "- **identico**: logit identici bit per bit in ogni posizione confrontata (impronta intera sui valori grezzi), "
        "stessi token più probabili e, dove si confrontano, stesse bozze accettate e rifiutate",
        "- **entro il rumore**: non identico, ma lo scarto massimo è al più il doppio di quello del metro e i token più "
        "probabili cambiati al più il doppio più due",
        "- **DIVERSO**: oltre il metro, o non identico con un metro identico",
        "- **errore**: il confronto non si è potuto fare o la prova non ha esercitato la funzione (il motivo è nella riga)",
        "",
    ]
    for line in meta.get("notes") or []:
        out += [f"Nota dei controlli preliminari: {line}", ""]
    for line in meta.get("end_notes") or []:
        out += [line, ""]
    for ch in meta.get("split_changes") or []:
        out += [f"**Attenzione**: con `--moe-device-split {ch['old']}` il processo `{ch['proc']}` non ha caricato il "
                f"modello; dal suo secondo tentativo in poi lo split è `{ch['new']}` (la stima che il server ha dato "
                "nel suo log). I processi prima di questo hanno lo split di prima: i confronti tra processi con split "
                "diversi non valgono (vedi le righe sul piazzamento).", ""]
    if meta.get("interrupted"):
        out += ["**Attenzione**: il giro è stato interrotto (segnale o Ctrl-C): gli scenari dopo l'interruzione mancano.", ""]
    for s in sc:
        for line in notes.get(s, []):
            out += [line, ""]
    out += conditions_section(plan, res, meta)
    for s in sc:
        out += scenario_section(s, plan, res, rows)
    out += [
        "## Come leggere",
        "",
        "Ogni job confrontato genera gli stessi token in tutti i giri: gli si impongono i token che un giro di riferimento ha "
        "prodotto da solo (greedy). A ogni posizione si registrano un'impronta dei logit (somma pesata intera dei bit dei "
        "valori, quindi uguale solo per logit uguali), la log-probabilità del token imposto e i token più probabili. Due giri "
        "dello stesso calcolo danno impronte uguali; il metro di un confronto è lo stesso giro di riferimento ripetuto "
        "(nello stesso processo o, per lo scenario 1, in un altro processo della stessa libreria).",
        "",
        "Le bozze della testa MTP (accettate e rifiutate) si confrontano posizione per posizione, coi contatori del job "
        "letti a ogni campionamento; dove il confronto non le giudica (checkpoint di coda acceso, che taglia le bozze; "
        "turni ripresi contro turni da capo) restano nella tabella.",
        "",
        "Il rapporto riporta misure; le righe che iniziano con «Attenzione» riguardano solo problemi della misura stessa.",
        "",
    ]
    return "\n".join(out) + "\n"


def conditions_section(plan: dict, res: Results, meta: dict) -> list:
    out = ["## Condizioni", ""]
    out.append(f"- Banco: `experiments/exact_check.py` al commit {meta.get('fork_revision', '-')}, fork in `{meta.get('fork_path', '-')}`")
    out.append(f"- Server (caricamento del modello come in produzione): `{meta.get('server_path', '-')}` al commit {meta.get('server_revision', '-')}")
    out.append(f"- Python `{meta.get('python', '-')}`; exllamav3 installato {meta.get('installed_version', '-')}")
    out.append(f"- Modello `{meta.get('model', '-')}`; argomenti del server comuni: `{' '.join(meta.get('server_argv_shown', []))}`")
    out.append(f"- Modo esatto: {meta.get('exact_text', '-')}")
    out.append("- Piazzamento degli esperti: ogni processo parte da una copia sua del file di piazzamento (l'originale non viene "
               "scritto; avvio a caldo e verifica del server) e nessuno sweep gira durante i giri confrontati "
               "(`run_pending_swap_sweeps` della libreria, chiamata sia allo svuotamento della coda sia dal ripiego in linea "
               "di uno sweep in ritardo, sostituita da una funzione vuota dopo il caricamento; lo scenario 6 chiama quella "
               "della libreria quando vuole); promote-on-miss spento (`EXL3_MOE_CPU_PROMOTE=0`); profilo degli esperti della "
               "libreria spento dal server. Lo stress (7) tiene gli sweep come in produzione (allo svuotamento della coda)")
    out.append(f"- Slot del worker degli esperti: `EXL3_MOE_CPU_SLOTS={meta.get('slots', '-')}` in ogni processo; "
               f"`EXL3_MOE_HOST_CORES={meta.get('host_cores') if meta.get('host_cores') not in (None, '') else 'della libreria'}`")
    out.append("- Campionamento greedy, nessuna condizione di stop, nessun requeue (nessun `max_rq_tokens` nei giri confrontati; "
               "lo stress usa quello del server); i checkpoint ricorrenti vengono tolti prima di ogni giro (ogni giro ingerisce "
               "i suoi prompt dal primo token) tranne dove il giro riprende di proposito; torre visiva non caricata")
    out.append("- Scritture: tutto sotto la cartella del giro (copie del piazzamento, cache su disco, `XDG_CACHE_HOME`, "
               "`TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `PYTHONPYCACHEPREFIX`); la cache su disco viene tolta alla fine "
               "salvo `--keep`")
    devs = None
    for m in res.meta.values():
        if m.get("devices"):
            devs = m["devices"]
            break
    out.append(f"- Schede: {', '.join(devs) if devs else '-'}")
    out.append("")
    prow = []
    for proc in plan["procs"]:
        name = proc["name"]
        o = res.procs.get(name) or {}
        m = res.meta.get(name) or {}
        state = "da fare"
        if o.get("skipped"):
            state = "saltato"
        elif o.get("killed"):
            state = f"fermato: {o.get('reason')}"
        elif o.get("exit") is not None:
            state = "finito" if o["exit"] == 0 else f"uscito con {o['exit']}"
        elif o.get("running"):
            state = "in corso"
        prow.append([f"`{name}`", proc["library"], ", ".join(str(s) for s in proc["scenarios"]),
                     f"{m.get('version', '-')} da `{m.get('library_path', '-')}`" if m else "-",
                     fmt(o.get("load_s"), 0), fmt(o.get("wall_s"), 0), len(res.records.get(name, {})), state])
    out.append(table(["processo", "libreria", "scenari", "exllamav3 importato", "caricamento s", "durata s", "record", "stato"], prow))
    out.append("")
    return out


def headline(r: dict) -> str:
    return f"**{r['label']}**: {r['verdict']}" + (f" ({r['why']})" if r.get("why") else "")


def scenario_section(s: int, plan: dict, res: Results, rows: list) -> list:
    out = [f"## {s}. {SCENARIO_TITLES[s]}", "", SCENARIO_TEXT[s], ""]
    rel = [r for r in rows if r.get("scenario") == s]
    probs = [r for r in rel if r.get("kind") == "probs"]
    if probs:
        out += [probs_table(probs), ""]
        for r in probs:
            for n in r.get("notes") or []:
                out += [f"{n} (riga «{r['label']}»)", ""]
    toks = [r for r in rel if r.get("kind") == "tokens"]
    if toks:
        out += [table(["confronto", "esito"], [[r["label"], r.get("why", "-")] for r in toks]), ""]
    for r in rel:
        kind = r.get("kind")
        if kind == "crc":
            out += [headline(r), ""]
            m = r.get("crc")
            if m:
                out += [table(["", "pagine", "in comune", "CRC diversi", "file rovinati", "checkpoint"],
                              [["riferimento", m["pages_ref"], m["common"], m["differ"], m["broken"], m["states_ref"]],
                               ["prova", m["pages_test"], m["common"], m["differ"], "", m["states_test"]]]), ""]
                if m.get("states_common") is not None:
                    out += [f"Checkpoint sul disco in comune: {m['states_common']}, con CRC dei dati diverso: "
                            f"{m.get('states_differ', 0)} (riportato, non giudicato).", ""]
            if r.get("write_behind") is not None:
                out += [f"Pagine già scritte sul disco al primo token di `r`: {r['write_behind']}.", ""]
            out += [n for n in r.get("notes") or []] + ([""] if r.get("notes") else [])
        elif kind == "placement":
            out += [headline(r), ""]
            trs = []
            floor = dict(r.get("floor_rounds") or [])
            for k, v in r.get("rounds") or []:
                f = floor.get(k)
                trs.append([k, "-" if v is None else ("sì" if v["maps_same"] else "no"),
                            "-" if v is None else ("sì" if v["hist_same"] else "no"),
                            "-" if v is None else f"{v['moved'][0]} / {v['moved'][1]}",
                            "-" if f is None else ("sì" if f["same"] else "no")])
            if trs:
                out += [table(["giro", "mappe uguali", "istogrammi uguali", "esperti scambiati (stock / fork)",
                               "stock-a contro stock-b uguali"], trs), ""]
        elif kind == "cancel":
            out += [headline(r), ""]
        elif kind == "resume":
            out += [f"**{r['label']}**", "", table(["turno", "token del prompt", "ripresi", "da rileggere", "da un checkpoint di coda",
                                                   "secondi al primo token"], r.get("table") or []), ""]
        elif kind == "stress":
            st = r.get("stress") or {}
            out += [headline(r), ""]
            if st:
                out += [table(["misura", "valore"], [
                    ["durata del traffico s", fmt(st.get("seconds"), 0)],
                    ["job accodati (brevi / documenti 8-33k / prompt lunghi 64-96k / turni dopo)",
                     f"{st.get('jobs', 0)} ({st.get('kinds', {}).get('short', 0)} / {st.get('kinds', {}).get('doc', 0)} / "
                     f"{st.get('kinds', {}).get('long', 0)} / {st.get('kinds', {}).get('turn', 0)})"],
                    ["finiti / annullati / falliti", f"{st.get('done', 0)} / {st.get('cancelled', 0)} / {st.get('job_errors', 0)}"],
                    ["token di prompt / generati", f"{st.get('prompt_tokens', 0)} / {st.get('generated', 0)}"],
                    ["chiamate a iterate() / la più lunga s / p99 s", f"{st.get('iterations', 0)} / {fmt(st.get('max_step_s'), 2)} / {fmt(st.get('p99_step_s'), 2)}"],
                    ["chiamate oltre il limite di un passo", st.get("over_limit", 0)],
                    ["svuotamento della coda s", fmt(st.get("drain_s"), 1)],
                    ["pezzi sul thread / accanto al decode / in volo con un altro", f"{st.get('side', {}).get('chunks', '-')} / {st.get('side', {}).get('beside', '-')} / {st.get('side', {}).get('piped', '-')}"],
                    ["chiamate sfalsate", (st.get("stagger") or {}).get("steps", "-")],
                    ["checkpoint di coda presi / sostituiti", f"{(st.get('tail') or {}).get('stashes', '-')} / {(st.get('tail') or {}).get('replaced', '-')}"],
                    ["disco: pagine scritte / lette / checkpoint scritti / errori di scrittura",
                     f"{(st.get('disk') or {}).get('page_writes', '-')} / {(st.get('disk') or {}).get('page_reads', '-')} / "
                     f"{(st.get('disk') or {}).get('state_writes', '-')} / {(st.get('disk') or {}).get('write_errors', '-')}"],
                    ["memoria libera minima per scheda MiB", ", ".join(f"cuda:{d} {v // (1 << 20)}" for d, v in sorted((st.get("free_min") or {}).items()))],
                ]), ""]
                if st.get("exceptions"):
                    out += ["Eccezioni: " + "; ".join(f"`{e}`" for e in st["exceptions"][:10]), ""]
                if st.get("slowest"):
                    out += ["Chiamate più lunghe: " + ", ".join(f"{fmt(t, 2)} s ({op})" for t, op in st["slowest"][:8]), ""]
    # Memory and timing per run, for the scenarios where they are asked for
    if s in (3, 4, 5):
        out += runs_table(s, plan, res)
    hangs = []
    for proc in plan["procs"]:
        if s in proc["scenarios"]:
            o = res.procs.get(proc["name"]) or {}
            if o.get("hang_file"):
                hangs.append(f"- `{proc['name']}`: stack di tutti i thread in `{o['hang_file']}`")
    if hangs:
        out += ["Stack dei thread scritti da faulthandler (cane da guardia, segnale dell'orchestratore o errore fatale):", ""] \
            + hangs + [""]
    return out


def runs_table(s: int, plan: dict, res: Results) -> list:
    rows = []
    for proc in plan["procs"]:
        if s not in proc["scenarios"]:
            continue
        recs = res.records.get(proc["name"], {})
        for st in proc["steps"]:
            if st.get("op") != "run" or st.get("scenario") != s or st.get("warmup"):
                continue
            rec = recs.get(st["id"])
            if rec is None:
                continue
            mem = rec.get("memory") or {}
            peak = mem.get("peak") or {}
            free = mem.get("free_min") or {}
            reader = None
            for name in rec.get("order") or []:
                w = ingestion_window(rec, name)
                if w is not None and (job_view(rec, name) or {}).get("prompt_tokens", 0) > 1000:
                    reader = (name, w)
            ing = "-"
            if reader:
                name, (t0, t1) = reader
                ptok = (job_view(rec, name) or {}).get("prompt_tokens") or 0
                ing = f"{t1 - t0:.1f} s ({ptok / max(t1 - t0, 1e-9):.0f} tok/s)"
            side = (rec.get("stats") or {}).get("side") or {}
            rows.append([f"`{proc['name']}` / `{st['id']}`", fmt(rec.get("seconds"), 1), ing,
                         f"{side.get('chunks', 0)} / {side.get('beside', 0)} / {side.get('piped', 0)} / {side.get('ahead', 0)}",
                         ", ".join(f"{int(v) / (1 << 30):.2f}" for _, v in sorted(peak.items())) or "-",
                         ", ".join(f"{int(v) // (1 << 20)}" for _, v in sorted(free.items())) or "-",
                         fmt(rec.get("max_step_s"), 2)])
    if not rows:
        return []
    return ["### I giri", "", "Tempi, pezzi sul thread (pezzi / accanto al decode / con un altro in volo / pianificati in anticipo), "
            "picco di memoria allocata da PyTorch per scheda (GiB) e memoria libera minima vista dal driver durante il giro (MiB).", "",
            table(["giro", "secondi", "ingestione del documento", "pezzi sul thread", "picco GiB per scheda",
                   "libera minima MiB per scheda", "chiamata più lunga s"], rows), ""]


# --------------------------------------------------------------------------------------------
#   Orchestration
# --------------------------------------------------------------------------------------------

def child_env(base_env: dict, proc: dict, args, run_dir: str) -> dict:
    """The environment of a child: the caller's minus every EXL3_ variable and PYTHONPATH, then
    the allocator, the library path, the exact mode, the worker's knobs and what the scenario
    asks for."""
    env = {k: v for k, v in base_env.items() if not k.startswith("EXL3_") and k not in (
        "PYTHONPATH", "PYTORCH_CUDA_ALLOC_CONF", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES")}
    env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    env["PYTHONUNBUFFERED"] = "1"
    # Nothing written outside the run directory: the library's caches, Triton's kernels, the
    # driver's JIT cache and the bytecode of the server's and the fork's sources (shared by the
    # processes of the night, so each is made once)
    caches = os.path.join(run_dir, "caches")
    env["XDG_CACHE_HOME"] = os.path.join(run_dir, "xdg-cache")
    env["TRITON_CACHE_DIR"] = os.path.join(caches, "triton")
    env["CUDA_CACHE_PATH"] = os.path.join(caches, "cuda")
    env["PYTHONPYCACHEPREFIX"] = os.path.join(caches, "pycache")
    if proc["library"] == "fork":
        env["PYTHONPATH"] = os.path.abspath(os.path.expanduser(args.fork_path))
    if args.exact and 7 not in proc["scenarios"]:
        env.update(EXACT_ENV)
    env["EXL3_MOE_CPU_SLOTS"] = str(args.worker_slots)
    env["EXL3_MOE_CPU_PROMOTE"] = "0"
    if str(args.host_cores).strip():
        env["EXL3_MOE_HOST_CORES"] = str(args.host_cores).strip()
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    env.update(proc.get("env") or {})
    return env


def base_server_argv(args, placement_dir: str) -> list:
    argv = [
        "--model-path", os.path.expanduser(args.model),
        "--max-seq-len", str(args.max_seq_len), "--cache-size", str(args.cache_size),
        "--cache-mode", args.cache_mode, "--max-batch-size", str(args.max_batch),
        "--max-chunk-size", str(args.chunk), "--tensor-parallel", "false",
        "--autosplit-reserve", args.autosplit_reserve,
        "--moe-device-split", args.device_split, "--moe-cpu-threads", str(args.worker_threads),
        "--moe-layer-budget", args.layer_budget,
        "--moe-swap-floor", "2", "--moe-swap-hyst", "1.5", "--moe-swap-max", "256", "--moe-swap-inline", "false",
        "--draft-mode", "mtp", "--draft-num-tokens", str(DRAFT_TOKENS), "--draft-cache-mode", args.draft_cache_mode,
        "--sysmem-kv-cache", str(args.sysmem_kv_cache), "--recurrent-checkpoint-interval", str(args.ckpt_interval),
        "--recurrent-cache-size", str(args.recurrent_cache_gb),
        "--attn-per-sequence", "true" if args.attn_per_sequence else "false",
    ]
    if placement_dir:
        argv += ["--moe-placement-file", placement_dir + os.sep]
    argv += list(args.server_arg or [])
    return argv


SPLIT_HINT = re.compile(r"try --moe-device-split (\S+) \(estimate\)")


def split_suggestion(log_text: str) -> str | None:
    """The split the server's expert tier suggests when a layer did not fit on the card the
    plan gives it (openai/moe_tier.py, _explain_overflow: "try --moe-device-split X
    (estimate)"); the last one in the text, None without one."""
    found = SPLIT_HINT.findall(log_text or "")
    return found[-1] if found else None


class Orchestrator:

    def __init__(self, plan: dict, args, run_dir: str, child_cmd: list | None = None, clock = time.time,
                 sleep = time.sleep, out = None):
        self.plan = plan
        self.args = args
        self.run_dir = run_dir
        self.child_cmd = child_cmd or [args.python, CHILD_SCRIPT]
        self.res = Results()
        self.meta = {}
        self.clock = clock
        self.sleep = sleep
        self.out = out if out is not None else sys.stdout
        self.estimate = estimate_plan(plan, args)
        self.budgets = scenario_budgets(self.estimate, args)
        self.spent = {s: 0.0 for s in plan["scenarios"]}
        self.report_path = os.path.join(run_dir, "REPORT.md")
        self.log_path = os.path.join(run_dir, "orchestrator.log")
        self.split_retries = 0
        self._current = None
        self._current_name = None

    # ---------------------------------------------------------------- output
    def say(self, text: str):
        """A line on the terminal and in orchestrator.log. A terminal that went away (a closed
        window, a dropped ssh session) must not end the night: writes to it are allowed to fail."""
        line = f"{time.strftime('%H:%M:%S')} {text}"
        try:
            with open(self.log_path, "a", encoding = "utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        try:
            print(line, file = self.out, flush = True)
        except (OSError, ValueError):
            pass

    # ---------------------------------------------------------------- refs
    def resolve(self, proc: dict) -> tuple:
        """The process's steps with every reference to another process's tokens replaced by the
        tokens themselves (those processes have finished). Returns (steps, prompts, problems)."""
        problems = []

        def tokens_of(r):
            rec = self.res.rec(r)
            if rec is None or not rec.get("ok"):
                return None
            j = job_view(rec, r["job"])
            if not j or not j.get("tokens"):
                return None
            toks = list(j["tokens"])
            if r.get("limit"):
                toks = toks[:r["limit"]]
            return toks

        def fix(r):
            if r is None or r["proc"] == proc["name"]:
                return r
            toks = tokens_of(r)
            if toks is None:
                problems.append(f"{r['proc']}/{r['run']}/{r['job']}")
                return {"missing": f"{r['proc']}/{r['run']}/{r['job']}"}
            return {"tokens": toks}

        steps = json.loads(json.dumps(proc["steps"]))
        for st in steps:
            for j in st.get("jobs", []) if st.get("op") == "run" else []:
                if "forced" in j:
                    j["forced"] = fix(j["forced"])
        prompts = json.loads(json.dumps(proc["prompts"]))
        for key, spec in prompts.items():
            if spec.get("kind") == "turn" and isinstance(spec.get("answer"), dict) and "proc" in spec["answer"]:
                spec["answer"] = fix(spec["answer"])
        return steps, prompts, problems

    # ---------------------------------------------------------------- report
    def write_report(self, finished: bool = False):
        try:
            text = render_report(self.plan, self.res, self.meta, finished)
        except Exception as e:
            text = f"# Controllo di identità bit a bit\n\nRapporto non generato: `{e!r}`\n"
        try:
            atomic_write(self.report_path, text)
            summary = {"procs": self.res.procs, "spent": self.spent, "budgets": self.budgets, "meta": self.meta,
                       "finished": finished}
            atomic_write(os.path.join(self.run_dir, "summary.json"), json.dumps(summary, indent = 1, default = str))
        except OSError as e:
            self.say(f"exact check: report not written ({e})")

    # ---------------------------------------------------------------- one process
    def run_proc(self, proc: dict):
        """One process of the plan; once more with the split the server suggests when the model
        did not load with the one given (--split-retries), that split then kept for the rest."""
        while True:
            self._run_proc_once(proc)
            o = self.res.procs.get(proc["name"]) or {}
            log_text = o.pop("log_text", "")
            if not o.get("load_failed") or o.get("killed") or self.split_retries >= self.args.split_retries:
                return
            hint = split_suggestion(log_text)
            if not hint or hint == self.args.device_split:
                return
            self.split_retries += 1
            self.meta.setdefault("split_changes", []).append({"proc": proc["name"], "old": self.args.device_split, "new": hint})
            self.say(f" -- exact check: {proc['name']} did not load with --moe-device-split {self.args.device_split}; "
                     f"again with {hint} (the server's estimate), kept from here on")
            self.args.device_split = hint
            self.meta["server_argv_shown"] = [a for a in base_server_argv(self.args, "<copia del piazzamento>") if a]
            self.write_report()

    def _run_proc_once(self, proc: dict):
        name = proc["name"]
        pdir = os.path.join(self.run_dir, "procs", name)
        os.makedirs(pdir, exist_ok = True)
        attempts = (self.res.procs.get(name) or {}).get("attempts", 0) + 1
        outcome = {"running": True, "attempts": attempts}
        self.res.procs[name] = outcome
        steps, prompts, problems = self.resolve(proc)
        remaining = {s: max(0.0, self.budgets.get(s, 3600.0) - self.spent.get(s, 0.0)) for s in proc["scenarios"]}
        if all(v <= 0 for v in remaining.values()):
            outcome.update(running = False, skipped = "tempo degli scenari esaurito")
            return
        # A private copy of the placement file for every process: the server's tier saves it
        # after the sweeps it sees, and nothing may reach the original
        placement_dir = ""
        if self.args.placement_copy:
            placement_dir = os.path.join(pdir, "placement")
            os.makedirs(placement_dir, exist_ok = True)
            # Under the name the server looks for in a directory, whatever the original is called;
            # a retry starts again from the original, not from what the failed load left
            shutil.copyfile(self.args.placement_copy, os.path.join(placement_dir, placement_name(self.args.model)))
        spec = {
            "proc": name, "library": proc["library"], "run_dir": self.run_dir, "proc_dir": pdir,
            "server_path": os.path.abspath(os.path.expanduser(self.args.server_path)),
            "fork_path": os.path.abspath(os.path.expanduser(self.args.fork_path)),
            "server_argv": base_server_argv(self.args, placement_dir) + list(proc.get("server_args") or []),
            "steps": steps, "prompts": prompts, "sweeps": proc.get("sweeps", "frozen"),
            "step_timeout": self.args.step_timeout, "load_timeout": self.args.load_timeout,
            "budgets": remaining, "top_k": self.args.top_k, "disk_budget_gb": self.args.disk_budget_gb,
            "disk_root": os.path.join(self.run_dir, "disk-cache"), "missing_refs": problems,
        }
        spec_path = os.path.join(pdir, "spec.json")
        atomic_write(spec_path, json.dumps(spec, indent = 1))
        env = child_env(os.environ, proc, self.args, self.run_dir)
        atomic_write(os.path.join(pdir, "env.json"), json.dumps({k: v for k, v in env.items()
                                                                 if k.startswith(("EXL3_", "PYTHON", "PYTORCH", "XDG", "CUDA", "TRITON"))}, indent = 1))
        log_path = os.path.join(pdir, "child.log")
        records_path = os.path.join(pdir, "records.jsonl")
        status_path = os.path.join(pdir, "status.json")
        hang_path = os.path.join(pdir, "hang-stacks.txt")
        log_start = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        # A status file of an earlier attempt must not pass for a sign of life of this one
        try:
            os.remove(status_path)
        except OSError:
            pass
        cmd = list(self.child_cmd) + ["--spec", spec_path]
        t0 = self.clock()
        log = open(log_path, "ab")
        try:
            child = subprocess.Popen(cmd, cwd = pdir, env = env, stdout = log, stderr = subprocess.STDOUT,
                                     stdin = subprocess.DEVNULL, start_new_session = True)
        except OSError as e:
            log.close()
            outcome.update(running = False, skipped = f"avvio fallito: {e}")
            return
        self._current, self._current_name = child, name
        hard = self.args.load_timeout + sum(remaining.values()) + 600.0
        killed = None
        last_scn = None
        last_t = t0
        next_report = t0 + self.args.report_every
        while True:
            code = child.poll()
            now = self.clock()
            st = None
            try:
                with open(status_path) as f:
                    st = json.load(f)
                st_mtime = os.path.getmtime(status_path)
            except (OSError, ValueError):
                st_mtime = t0
            # Time spent per scenario, from what the child says it is working on
            scn = (st or {}).get("scenario")
            if last_scn is not None:
                self.spent[last_scn] = self.spent.get(last_scn, 0.0) + (now - last_t)
            last_scn, last_t = scn, now
            if code is not None:
                break
            op_timeout = float((st or {}).get("timeout") or self.args.step_timeout)
            if now - st_mtime > op_timeout + self.args.grace:
                killed = f"nessun segno di vita per {now - st_mtime:.0f} s durante `{(st or {}).get('op', '?')}`"
            elif now - t0 > hard:
                killed = f"oltre il tempo del processo ({hard / 60:.0f} min)"
            if killed:
                self.say(f" -- exact check: {name}: {killed}; stacks, then the process group goes")
                self.kill(child, hang = True)
                break
            if self.args.report_every > 0 and now >= next_report:
                # What the process has recorded so far, in the report (a long process otherwise
                # shows in it only once it is over)
                next_report = now + self.args.report_every
                self.res.load_proc(name, records_path)
                self.write_report()
            self.sleep(self.args.poll)
        code = child.wait()
        log.close()
        self.reap_group(child.pid)
        self._current, self._current_name = None, None
        self.res.load_proc(name, records_path)
        recs = self.res.records.get(name, {})
        load_rec = recs.get("load") or {}
        outcome.update(running = False, exit = code, wall_s = self.clock() - t0, load_s = load_rec.get("seconds"))
        hang_text = ""
        try:
            with open(hang_path) as f:
                hang_text = f.read()
        except OSError:
            pass
        if killed:
            outcome.update(killed = True, reason = killed, hang = True)
        elif code != 0:
            st = None
            try:
                with open(status_path) as f:
                    st = json.load(f)
            except (OSError, ValueError):
                pass
            if hang_text.strip() and "Timeout" in hang_text:
                outcome.update(hang = True, reason = f"il cane da guardia è scattato durante `{(st or {}).get('op', '?')}` "
                                                     f"(`{(st or {}).get('run', '-')}`, oltre {float((st or {}).get('timeout') or self.args.step_timeout):.0f} s)")
            elif code is not None and code < 0:
                outcome["reason"] = f"terminato dal segnale {-code}"
            else:
                outcome["reason"] = f"uscito con {code}"
        if hang_text.strip():
            outcome["hang_file"] = os.path.relpath(hang_path, self.run_dir)
        # The model never loaded: what the child said about it, for a retry with another split
        if not load_rec.get("ok") and not any(r.get("op") == "run" for r in recs.values()):
            outcome["load_failed"] = True
            outcome["log_text"] = self.log_since(log_path, log_start)
        if code != 0 or killed:
            outcome["log_tail"] = self.log_tail(log_path)

    def log_since(self, path: str, start: int, limit: int = 1 << 20) -> str:
        """What a child wrote to its log in this attempt (the last `limit` bytes of it)."""
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(start, size - limit))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def log_tail(self, path: str, lines: int = 3) -> str:
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8192))
                text = f.read().decode("utf-8", "replace")
            return " / ".join(l.strip() for l in text.strip().splitlines()[-lines:])[:600]
        except OSError:
            return ""

    def kill(self, child, hang: bool):
        """Stacks first (the child registered SIGUSR1 with faulthandler; the signal goes to it
        alone, the CPU expert worker would just die of it), then the whole group."""
        if hang:
            try:
                os.kill(child.pid, signal.SIGUSR1)
            except OSError:
                pass
            t_end = self.clock() + 20
            while self.clock() < t_end and child.poll() is None:
                self.sleep(0.5)
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except OSError:
            pass

    def reap_group(self, pgid: int, timeout: float = 60.0):
        """Wait until nothing of the child's process group is left (the CPU expert worker is a
        grandchild): the next process needs the cards and the RAM."""
        t_end = self.clock() + timeout
        while self.clock() < t_end:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return True
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
            self.sleep(0.5)
        return False

    # ---------------------------------------------------------------- the night
    def run(self) -> int:
        os.makedirs(os.path.join(self.run_dir, "procs"), exist_ok = True)
        atomic_write(os.path.join(self.run_dir, "plan.json"), json.dumps(self.plan, indent = 1))
        self.meta["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.write_report()

        def on_signal(signum, frame):
            raise KeyboardInterrupt

        # SIGTERM and Ctrl-C end the night in order (the process under way is stopped, the report
        # written); a hangup (the terminal closed) does not end it
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            for proc in self.plan["procs"]:
                self.say(f" -- exact check: process {proc['name']} (scenari {proc['scenarios']})")
                self.run_proc(proc)
                o = self.res.procs.get(proc["name"]) or {}
                self.say(f" -- exact check: process {proc['name']}: "
                         + (o.get("skipped") or o.get("reason") or f"finished in {o.get('wall_s', 0) / 60:.0f} min"))
                self.write_report()
        except KeyboardInterrupt:
            if self._current is not None:
                name = self._current_name
                self.kill(self._current, hang = False)
                self.reap_group(self._current.pid)
                if name is not None:
                    self.res.load_proc(name, os.path.join(self.run_dir, "procs", name, "records.jsonl"))
                    self.res.procs.setdefault(name, {}).update(running = False, killed = True, reason = "interrotto")
            self.meta["interrupted"] = True
        finally:
            for sig, h in old.items():
                try:
                    signal.signal(sig, h)
                except (TypeError, ValueError):
                    pass
            self.meta["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
            # The original placement file is only read: copied for every process
            src = getattr(self.args, "placement_copy", None)
            if src and self.meta.get("placement_sha") and file_digest(src) != self.meta["placement_sha"]:
                self.meta.setdefault("end_notes", []).append(
                    f"**Attenzione**: il file di piazzamento originale {src} è cambiato durante la notte (qualcun "
                    "altro lo ha scritto: i processi partivano ognuno da una copia presa in quel momento)")
            if not self.args.keep:
                shutil.rmtree(os.path.join(self.run_dir, "disk-cache"), ignore_errors = True)
            self.write_report(finished = True)
            self.say(f" -- exact check: report in {self.report_path}")
        return 0


def rebuild_report(run_dir: str) -> str:
    """REPORT.md again from what a run directory holds (plan.json, every process's records,
    summary.json for the processes' outcomes): the night's figures without running anything,
    e.g. after the orchestrator itself was killed. Returns the report's path."""
    with open(os.path.join(run_dir, "plan.json"), encoding = "utf-8") as f:
        plan = json.load(f)
    summary = {}
    try:
        with open(os.path.join(run_dir, "summary.json"), encoding = "utf-8") as f:
            summary = json.load(f)
    except (OSError, ValueError):
        pass
    res = Results()
    for proc in plan["procs"]:
        path = os.path.join(run_dir, "procs", proc["name"], "records.jsonl")
        if os.path.exists(path):
            res.load_proc(proc["name"], path)
    res.procs = dict(summary.get("procs") or {})
    for name, o in res.procs.items():
        if isinstance(o, dict) and o.get("running"):
            o.update(running = False, killed = True, reason = o.get("reason") or "interrotto (orchestratore fermato)")
    meta = dict(summary.get("meta") or {})
    finished = bool(summary.get("finished"))
    if not finished:
        meta["interrupted"] = True
    text = render_report(plan, res, meta, True)
    path = os.path.join(run_dir, "REPORT.md")
    atomic_write(path, text)
    return path


# --------------------------------------------------------------------------------------------
#   Preflight and dry run
# --------------------------------------------------------------------------------------------

def gpu_processes() -> list | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"], capture_output = True, text = True, timeout = 20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def existing_dir(path: str) -> str:
    """The directory itself if it exists, else its nearest existing ancestor."""
    path = os.path.abspath(path)
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


LIBRARY_PROBE = (
    "import importlib.metadata as m, importlib.util as u, json\n"
    "out = {}\n"
    "try:\n"
    "    out['version'] = m.version('exllamav3')\n"
    "except Exception as e:\n"
    "    out['version'] = None\n"
    "for name in ('exllamav3', 'exllamav3_ext'):\n"
    "    try:\n"
    "        spec = u.find_spec(name)\n"
    "        out[name] = spec.origin if spec is not None else None\n"
    "    except Exception as e:\n"
    "        out[name] = None\n"
    "print(json.dumps(out))\n"
)


def probe_library(python: str, pythonpath: str | None) -> dict:
    """Where `python` finds exllamav3 and its compiled extension, with the fork ahead on the path
    or not; nothing is imported (find_spec) and the cards are hidden."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    out = subprocess.run([python, "-c", LIBRARY_PROBE], capture_output = True, text = True, timeout = 120,
                         env = env, cwd = "/")
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if out.returncode != 0 or not lines:
        raise OSError(out.stderr.strip()[-300:] or f"exit {out.returncode}")
    return json.loads(lines[-1])


def preflight(args, plan: dict) -> tuple:
    """(problems that stop the night, notes) about the machine and the paths. Writes nothing."""
    problems, notes = [], []
    model = os.path.expanduser(args.model)
    if not os.path.isdir(model):
        problems.append(f"modello non trovato: {model}")
    if args.placement:
        pf = resolve_placement_file(args.placement, args.model)
        if not os.path.isfile(pf):
            problems.append(f"file di piazzamento non trovato: {pf} (--placement, o --placement '' per partire senza)")
        args.placement_copy = pf if os.path.isfile(pf) else None
    else:
        args.placement_copy = None
        if args.layer_budget == "profile":
            problems.append("--layer-budget profile vuole il file di piazzamento (--placement)")
    server = os.path.abspath(os.path.expanduser(args.server_path))
    if not os.path.isfile(os.path.join(server, "openai", "config.py")):
        problems.append(f"server non trovato in {server} (--server-path)")
    else:
        try:
            with open(os.path.join(server, "pyproject.toml")) as f:
                py = f.read()
            if f"exllamav3=={BASE_RELEASE}" not in py.replace(" ", ""):
                problems.append(f"il server in {server} non fissa exllamav3=={BASE_RELEASE} nel pyproject: non è "
                                f"il server per {BASE_RELEASE} (--server-path, default {SERVER_DEFAULT})")
        except OSError:
            notes.append(f"{server}/pyproject.toml non leggibile")
    fork = os.path.abspath(os.path.expanduser(args.fork_path))
    gsrc = os.path.join(fork, "exllamav3", "generator", "generator.py")
    try:
        with open(gsrc) as f:
            g = f.read()
        for needle in ("prefill_pipeline_state", "def iterate_staggered", "tail_checkpoint_interval"):
            if needle not in g:
                problems.append(f"il fork in {fork} non ha `{needle}` (--fork-path)")
    except OSError:
        problems.append(f"fork non trovato: {gsrc}")
    args.installed_version = "?"
    try:
        stock = probe_library(args.python, None)
        forked = probe_library(args.python, fork)
        args.installed_version = stock.get("version") or "?"
        if str(stock.get("version") or "").split("+")[0] != BASE_RELEASE:
            problems.append(f"in {args.python} exllamav3 è {stock.get('version') or 'assente'}, serve la wheel {BASE_RELEASE}")
        if (stock.get("exllamav3") or "").startswith(fork + os.sep):
            problems.append(f"senza PYTHONPATH {args.python} trova exllamav3 nel fork ({stock.get('exllamav3')}): "
                            "il confronto con upstream non avrebbe upstream")
        if not (forked.get("exllamav3") or "").startswith(fork + os.sep):
            problems.append(f"con PYTHONPATH={fork} exllamav3 viene da {forked.get('exllamav3')}, non dal fork")
        ext = forked.get("exllamav3_ext")
        if not ext:
            problems.append(f"exllamav3_ext (l'estensione compilata della wheel) non si trova con {args.python}")
        elif ext.startswith(fork + os.sep):
            problems.append(f"exllamav3_ext viene dal fork ({ext}), non dalla wheel {BASE_RELEASE} dell'ambiente")
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        problems.append(f"{args.python} non risponde: {e}")
    apps = gpu_processes()
    if apps is None:
        notes.append("nvidia-smi non risponde: non so se le schede sono libere")
    elif apps:
        problems.append("processi sulle schede (il server di produzione va fermato): " + "; ".join(apps))
    where = existing_dir(args.out)
    if os.path.exists(os.path.join(args.out, "plan.json")):
        problems.append(f"{args.out} contiene già un giro (plan.json): i record si mescolerebbero, scegli un'altra --out "
                        "(--rebuild per rifarne il rapporto)")
    try:
        if os.path.exists(args.out) and not os.path.isdir(args.out):
            raise OSError(f"{args.out} esiste e non è una cartella")
        if not os.access(where, os.W_OK):
            raise OSError(f"{where} non è scrivibile")
        st = os.statvfs(where)
        free_gb = st.f_bavail * st.f_frsize / 1024 ** 3
        if free_gb < 40:
            notes.append(f"solo {free_gb:.0f} GB liberi in {where}: il tier su disco smette di scrivere sotto "
                         "min(16 GiB, 2% del filesystem) liberi")
        with open("/proc/mounts") as f:
            mounts = [l.split() for l in f if len(l.split()) >= 3]
        best = max((m for m in mounts if where == m[1] or where.startswith(m[1].rstrip("/") + "/")),
                   key = lambda m: len(m[1]), default = None)
        if best is not None and best[2] in ("tmpfs", "ramfs"):
            problems.append(f"la cartella del giro è su {best[2]} ({best[1]}): il tier su disco e la sua igiene della "
                            "page cache vanno misurati su un disco vero (--out)")
    except OSError as e:
        problems.append(f"cartella del giro non utilizzabile: {e}")
    caller = sorted(k for k in os.environ if k.startswith("EXL3_"))
    if caller:
        notes.append("variabili EXL3_ dell'ambiente tolte ai processi (passale con --env se servono): " + ", ".join(caller))
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        notes.append(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r} tolto ai processi, che vedono tutte le "
                     "schede (--env CUDA_VISIBLE_DEVICES=... per altro)")
    if server and os.path.isfile(os.path.join(server, "openai", "config.py")) and not problems:
        bad = validate_server_args(args, plan, server)
        problems += bad
    return problems, notes


def validate_server_args(args, plan: dict, server: str) -> list:
    """Every process's server arguments through the server's own parser (in the child's Python,
    which has its dependencies), before anything loads."""
    code = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from openai.config import parse_args\n"
        "bad = []\n"
        "for name, argv in json.loads(sys.argv[2]).items():\n"
        "    try:\n"
        "        parse_args(argv)\n"
        "    except (SystemExit, Exception) as e:\n"
        "        bad.append(f'{name}: {e!r}')\n"
        "print(json.dumps(bad))\n"
    )
    argvs = {p["name"]: base_server_argv(args, os.path.dirname(args.placement_copy or "") if args.placement_copy else "")
             + list(p.get("server_args") or []) for p in plan["procs"]}
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(CUDA_VISIBLE_DEVICES = "", PYTHONDONTWRITEBYTECODE = "1")
    try:
        out = subprocess.run([args.python, "-c", code, server, json.dumps(argvs)], capture_output = True, text = True,
                             timeout = 120, cwd = existing_dir(args.out), env = env)
        if out.returncode != 0:
            return [f"il parser del server non si importa: {out.stderr.strip()[-300:]}"]
        lines = [l for l in out.stdout.strip().splitlines() if l.startswith("[")]
        return [f"argomenti del server rifiutati: {b}" for b in json.loads(lines[-1])] if lines else []
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        return [f"controllo degli argomenti del server non riuscito: {e}"]


def dry_run_text(plan: dict, args, est: dict, budgets: dict, problems: list, notes: list) -> str:
    lines = ["Controllo di identità bit a bit: piano", ""]
    lines.append(f"Cartella del giro: {args.out}")
    lines.append(f"Ipotesi della stima: prefill {args.assume_prefill:.0f} token/s (qualunque cosa giri accanto; 700-720 nel "
                 f"traffico vero del 9-11 ottobre), decode {args.assume_decode:.0f} token/s per un job e "
                 f"{args.assume_decode:.0f} x radice(n) in tutto per n job insieme (63/86/103/117 token/s per 1-4 job, "
                 f"misurati il 9 ottobre con --attn-per-sequence), {args.assume_load_min:.0f} min per caricamento del "
                 "modello (mai misurato: il rapporto dice quello vero), pochi secondi per giro di contorno. Con lo stesso "
                 "rapporto si può rifare il conto: --assume-prefill, --assume-decode, --assume-load-min.")
    lines.append("")
    for s in plan["scenarios"]:
        procs = [p["name"] for p in plan["procs"] if s in p["scenarios"]]
        lines.append(f"Scenario {s} - {SCENARIO_TITLES[s]}")
        lines.append(f"  processi: {', '.join(procs)}")
        lines.append(f"  giri: {est['runs'].get(s, 0)}, token di prompt ingeriti: {est['ingest'].get(s, 0)}, "
                     f"token generati: {est['generate'].get(s, 0)}")
        n_cmp = len([c for c in plan["comparisons"] if c["scenario"] == s and c.get("kind", "probs") in ("probs", "crc", "placement", "cancel", "stress")])
        lines.append(f"  confronti: {n_cmp}")
        lines.append(f"  stima: {est['scenarios'].get(s, 0) / 60:.0f} min (tempo massimo concesso: {budgets.get(s, 0) / 60:.0f} min)")
        lines.append("")
    lines.append("Processi, in ordine:")
    for p in plan["procs"]:
        e = est["procs"][p["name"]]
        n_runs = len([st for st in p["steps"] if st["op"] == "run"])
        lines.append(f"  {p['name']:<12} {p['library']:<5} scenari {','.join(str(s) for s in p['scenarios']):<5} "
                     f"{n_runs:>3} giri  {e['total'] / 60:6.0f} min")
    lines.append("")
    lines.append(f"Totale stimato: {est['total'] / 3600:.1f} ore ({est['total'] / 60:.0f} min)")
    lines.append("")
    for n in notes:
        lines.append(f"Nota: {n}")
    for pr in problems:
        lines.append(f"PROBLEMA: {pr}")
    if not problems:
        lines.append("Controlli preliminari: nessun problema.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
#   CLI
# --------------------------------------------------------------------------------------------

def int_list(text: str) -> list:
    return [int(x) for x in str(text).split(",") if x.strip()]


def parse_cli(argv: list | None = None):
    ap = argparse.ArgumentParser(
        description = f"Controllo notturno di identità bit a bit del fork {FORK_BRANCH} contro upstream {BASE_RELEASE}, "
                      "sul modello di produzione, a server fermo. Rapporto in REPORT.md nella cartella del giro.")
    ap.add_argument("--scenarios", default = "1,2,3,4,5,6,7", help = "scenari da eseguire, es. 1,2 (default tutti)")
    ap.add_argument("--dry-run", action = "store_true", help = "stampa il piano, i controlli e la stima, non carica niente")
    ap.add_argument("--out", default = None, help = "cartella del giro (default ./exact-check-AAAAMMGG-HHMMSS)")
    ap.add_argument("--keep", action = "store_true", help = "non cancellare la cache su disco del giro alla fine")
    ap.add_argument("--force", action = "store_true", help = "parte anche con problemi nei controlli preliminari")
    ap.add_argument("--python", default = sys.executable, help = f"Python dell'ambiente con la wheel {BASE_RELEASE} (default questo)")
    ap.add_argument("--fork-path", default = FORK_DEFAULT, help = "checkout del fork (default quello di questo file)")
    ap.add_argument("--server-path", default = SERVER_DEFAULT, help = f"checkout del server per {BASE_RELEASE} (default {SERVER_DEFAULT})")
    ap.add_argument("--model", default = MODEL_DEFAULT, help = "cartella del modello")
    ap.add_argument("--placement", default = PLACEMENT_DEFAULT,
                    help = "file (o cartella) del piazzamento degli esperti del server: viene copiato, mai scritto; '' per nessuno")
    ap.add_argument("--device-split", default = "23:226,25:253", help = "split degli esperti per scheda (--moe-device-split)")
    ap.add_argument("--layer-budget", default = "profile", help = "--moe-layer-budget del server (default profile)")
    ap.add_argument("--cache-mode", default = "8,8")
    ap.add_argument("--draft-cache-mode", default = "4,4")
    ap.add_argument("--cache-size", type = int, default = 370000)
    ap.add_argument("--max-seq-len", type = int, default = 250000)
    ap.add_argument("--max-batch", type = int, default = 6)
    ap.add_argument("--chunk", type = int, default = 4096)
    ap.add_argument("--ckpt-interval", type = int, default = 4096, help = "recurrent_checkpoint_interval")
    ap.add_argument("--recurrent-cache-gb", type = float, default = 1.0)
    ap.add_argument("--sysmem-kv-cache", type = int, default = 1000, help = "tier K/V in RAM, MB")
    ap.add_argument("--autosplit-reserve", default = "96,96")
    ap.add_argument("--worker-threads", type = int, default = 12)
    ap.add_argument("--worker-slots", type = int, default = 6, help = "EXL3_MOE_CPU_SLOTS in ogni processo (default 6)")
    ap.add_argument("--host-cores", default = "0",
                    help = "EXL3_MOE_HOST_CORES in ogni processo: 0 con 12 thread (la scelta «12 come oggi» della guida "
                           "di migrazione, 3.4), '' per lasciare quello della libreria (1)")
    ap.add_argument("--attn-per-sequence", type = lambda v: v.lower() in ("1", "true", "yes", "on"), default = True)
    ap.add_argument("--server-arg", action = "append", help = "argomento in più per il server, ripetibile")
    ap.add_argument("--env", action = "append", help = "VAR=VALORE nell'ambiente di ogni processo, ripetibile")
    ap.add_argument("--no-exact", dest = "exact", action = "store_false",
                    help = "senza EXL3_MOE_STREAM_FUSED_T=0 EXL3_MOE_RECON_DET=1 (il rumore di fondo torna)")
    ap.add_argument("--tokens", type = int, default = 320, help = "token imposti per job negli scenari 1 e 2")
    ap.add_argument("--decoder-tokens", type = int, default = 640, help = "token imposti ai job che generano negli scenari 3 e 4")
    ap.add_argument("--reader-tokens", type = int, default = 64, help = "token del documento dopo il prompt")
    ap.add_argument("--lead", type = int, default = 48, help = "token di `a` prima che il documento venga accodato")
    ap.add_argument("--shift", type = int, default = 37,
                    help = "token in più di `a` prima che il documento venga accodato nei giri di controllo (scenari 3 e 4)")
    ap.add_argument("--warmup-tokens", type = int, default = 32)
    ap.add_argument("--top-k", type = int, default = 4, help = "token più probabili registrati per posizione")
    ap.add_argument("--s2-jobs", type = int_list, default = [2, 3, 4, 5])
    ap.add_argument("--s3-decoders", type = int_list, default = [2, 3, 4, 5])
    ap.add_argument("--s3-last-chunks", type = int_list, default = [33, 48, 64, 65, 96, 100, 144, 145, 199, 249])
    ap.add_argument("--s4-prompt", type = int, default = 33334, help = "token del documento dello scenario 4 (oltre 32768)")
    ap.add_argument("--s4-long", type = int, default = 179301, help = "token del prompt lungo (0 = niente processo a sé)")
    ap.add_argument("--s4-long-floor", type = lambda v: v.lower() in ("1", "true", "yes", "on"), default = True,
                    help = "ripete il prompt lungo in linea per il metro (default sì)")
    ap.add_argument("--s5-answer", type = int, default = 1100)
    ap.add_argument("--s5-next", type = int, default = 32)
    ap.add_argument("--s6-rounds", type = int, default = 4)
    ap.add_argument("--s6-tokens", type = int, default = 160)
    ap.add_argument("--s7-minutes", type = float, default = 10.0)
    ap.add_argument("--s7-step-limit", type = float, default = 60.0, help = "secondi oltre i quali un passo dello stress è un fallimento")
    ap.add_argument("--s7-drain-timeout", type = float, default = 600.0)
    ap.add_argument("--pinned-arena", action = "store_true", help = "scenario 7 con l'arena degli esperti bloccata in RAM")
    ap.add_argument("--seed", type = int, default = 20261018)
    ap.add_argument("--disk-budget-gb", type = float, default = 16.0, help = "budget di ogni cartella del tier su disco")
    ap.add_argument("--step-timeout", type = float, default = 300.0, help = "cane da guardia di ogni chiamata della libreria, s")
    ap.add_argument("--load-timeout", type = float, default = 2400.0, help = "tempo massimo di un caricamento del modello, s")
    ap.add_argument("--split-retries", type = int, default = 2,
                    help = "quante volte rifare un processo che non carica il modello con lo split suggerito dal server "
                           "(che resta poi per i processi dopo); 0 per mai")
    ap.add_argument("--rebuild", default = None, metavar = "CARTELLA",
                    help = "rifà REPORT.md da una cartella di un giro (plan.json e i record), senza far girare niente")
    ap.add_argument("--scenario-timeout", type = float, default = 0.0, help = "minuti per scenario (default 3x la stima, almeno 30)")
    ap.add_argument("--grace", type = float, default = 180.0, help = argparse.SUPPRESS)
    ap.add_argument("--poll", type = float, default = 2.0, help = argparse.SUPPRESS)
    ap.add_argument("--report-every", type = float, default = 600.0, help = argparse.SUPPRESS)
    ap.add_argument("--child-cmd", default = None, help = argparse.SUPPRESS)
    ap.add_argument("--assume-prefill", type = float, default = 700.0)
    ap.add_argument("--assume-decode", type = float, default = 60.0)
    ap.add_argument("--assume-load-min", type = float, default = 5.0)
    args = ap.parse_args(argv)
    args.scenarios = sorted({int(x) for x in str(args.scenarios).split(",") if x.strip()})
    unknown = [s for s in args.scenarios if s not in SCENARIO_TITLES]
    if unknown:
        ap.error(f"scenari sconosciuti: {unknown}")
    if args.out is None:
        args.out = os.path.abspath(time.strftime("exact-check-%Y%m%d-%H%M%S"))
    args.out = os.path.abspath(os.path.expanduser(args.out))
    args.fork_path = os.path.abspath(os.path.expanduser(args.fork_path))
    if args.chunk % 256:
        ap.error("--chunk: multiplo di 256")
    return args


def main(argv: list | None = None) -> int:
    args = parse_cli(argv)
    if args.rebuild:
        print(rebuild_report(os.path.abspath(os.path.expanduser(args.rebuild))))
        return 0
    plan = build_plan(args)
    est = estimate_plan(plan, args)
    budgets = scenario_budgets(est, args)
    problems, notes = preflight(args, plan)
    if args.dry_run:
        print(dry_run_text(plan, args, est, budgets, problems, notes))
        return 1 if problems else 0
    if problems and not args.force:
        print(dry_run_text(plan, args, est, budgets, problems, notes))
        print("Fermo: correggi i problemi sopra (o --force).")
        return 2
    os.makedirs(args.out, exist_ok = True)
    child_cmd = json.loads(args.child_cmd) if args.child_cmd else None
    orch = Orchestrator(plan, args, args.out, child_cmd = child_cmd)
    orch.say(dry_run_text(plan, args, est, budgets, problems, notes))
    orch.meta.update(
        fork_path = args.fork_path, fork_revision = git_revision(args.fork_path),
        server_path = os.path.abspath(os.path.expanduser(args.server_path)),
        server_revision = git_revision(os.path.abspath(os.path.expanduser(args.server_path))),
        python = args.python, installed_version = getattr(args, "installed_version", "?"), model = args.model,
        server_argv_shown = [a for a in base_server_argv(args, "<copia del piazzamento>") if a],
        exact_text = (" ".join(f"{k}={v}" for k, v in EXACT_ENV.items()) + " in ogni processo tranne lo stress (7)") if args.exact
        else "spento (--no-exact)",
        slots = args.worker_slots, host_cores = args.host_cores, notes = notes,
        placement_sha = file_digest(getattr(args, "placement_copy", None)),
    )
    return orch.run()


if __name__ == "__main__":
    sys.exit(main())
