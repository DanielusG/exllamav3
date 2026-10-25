"""
The overnight exact check (experiments/exact_check.py, experiments/exact_check_child.py) on this
machine: no GPU, no model, no extension.

- the orchestrator's own logic: comparisons and verdicts, the noise floor, the disk scan, the
  plan (every forced token and every comparison names a run that exists, in an order the
  library allows), the time estimate and the dry run, the report, incremental records;
- the watchdog, in a process of its own: a call that does not return leaves the stacks of every
  thread and ends the process;
- the orchestration, with a fake child process in place of the real one: specs, references
  across processes, a child that hangs or crashes, the report rewritten after every process;
- the child's Runner over the real generator and the toy model of test_stagger_decode (with the
  prefill thread of test_prefill_thread): forced tokens, logit digests, the ingestion window, a
  cancel, the staggered decode, and a wrong number on the prefill thread caught.

What only the first real run can tell: whether the library still has the shapes the child reads
(Job's results, the generator's counters, the server's container), and the numbers.
"""

import json
import math
import os
import subprocess
import sys
import textwrap
import threading
import time
import types
import zlib

import pytest
import torch

import test_stagger_decode as toy
from test_stagger_decode import ToyJob, _host_torch, make_generator      # noqa: F401
from test_prefill_thread import Slots, _fake_lane, lasting, long_prompt    # noqa: F401

pytestmark = pytest.mark.nogpu

# (the toy's autouse fixtures: host tensors only, a lane with no device behind it)
_FIXTURES = (_host_torch, _fake_lane)

EXP = os.path.join(toy.REPO, "experiments")
sys.path.insert(0, EXP)
import exact_check as EC            # noqa: E402
import exact_check_child as CH      # noqa: E402

CHUNK = 16


# --------------------------------------------------------------------------------------------
#   Comparisons
# --------------------------------------------------------------------------------------------

def view(tokens, digest = None, logp = None, top = None, accepted = 3, rejected = 1):
    n = len(tokens)
    return {
        "tokens": list(tokens),
        "digest": list(digest if digest is not None else range(100, 100 + n)),
        "logp": list(logp if logp is not None else [-0.5 - 0.01 * i for i in range(n)]),
        "topi": [list(t) for t in (top if top is not None else [[tok, 0] for tok in tokens])],
        "topl": [[-0.5, -2.0] for _ in range(n)],
        "accepted": accepted, "rejected": rejected,
    }


def test_identical_runs_compare_identical():
    a = view([5, 6, 7, 8])
    c = EC.compare_jobs(a, json.loads(json.dumps(a)))
    assert c["identical"] and c["same"] == 4 and c["first_diff"] is None and c["max"] == 0.0
    assert c["tokens_ok"] and c["digests"] and c["flips"] == 0 and c["nonfinite"] == 0


def test_a_different_digest_is_a_difference_even_with_equal_probabilities():
    a = view([5, 6, 7, 8])
    b = view([5, 6, 7, 8], digest = [100, 101, 999, 103])
    c = EC.compare_jobs(a, b)
    assert not c["identical"] and c["first_diff"] == 2 and c["max"] == 0.0 and c["same"] == 3


def test_log_probability_shifts_flips_and_the_window():
    a = view([5, 6, 7, 8])
    b = view([5, 6, 7, 8], digest = [100, 101, 7, 8], logp = [-0.5, -0.51, -0.9, -0.2], top = [[5, 0], [6, 0], [9, 7], [8, 0]])
    c = EC.compare_jobs(a, b)
    assert c["first_diff"] == 2 and c["flips"] == 1
    assert c["max"] == pytest.approx(0.38) and c["mean"] == pytest.approx((0.38 + 0.33) / 4)
    # Only the positions before the window's end
    assert EC.compare_jobs(a, b, upto = 2)["identical"]


def test_values_that_are_not_finite_are_counted_not_hidden():
    a = view([5, 6, 7])
    b = view([5, 6, 7], logp = [-0.5, float("nan"), float("-inf")])
    c = EC.compare_jobs(a, b)
    assert c["nonfinite"] == 2 and not c["identical"] and c["max"] == 0.0


def test_forced_tokens_not_honoured_are_seen():
    c = EC.compare_jobs(view([5, 6, 7]), view([5, 9, 7]))
    assert not c["tokens_ok"]


def test_the_rule_against_a_floor_that_moves():
    floor = {"max": 0.04, "flips": 2}
    assert EC.within({"max": 0.08, "flips": 6}, floor)
    assert not EC.within({"max": 0.081, "flips": 0}, floor)
    assert not EC.within({"max": 0.01, "flips": 7}, floor)
    assert not EC.within({"max": 0.01, "flips": 0}, None)


class Res(EC.Results):
    """Results from records written here."""

    def __init__(self, recs):
        super().__init__()
        for (proc, rid), rec in recs.items():
            rec.setdefault("ok", True)
            rec.setdefault("id", rid)
            self.records.setdefault(proc, {})[rid] = rec


def runrec(**jobs):
    return {"ok": True, "jobs": jobs, "order": list(jobs), "timeline": [], "stats": {}, "state": {}}


def probs_cmp(**kw):
    base = dict(test = EC.ref("p", "t", "a"), refr = EC.ref("p", "r1", "a"), floor = (EC.ref("p", "r1", "a"), EC.ref("p", "r2", "a")))
    base.update(kw)
    return EC.cmp(1, "x", "label", base.pop("test"), base.pop("refr"), **base)


def test_verdicts_against_the_floor():
    same = view([1, 2, 3, 4])
    moved = view([1, 2, 3, 4], digest = [0, 0, 0, 0], logp = [-0.5, -0.52, -0.52, -0.53])
    far = view([1, 2, 3, 4], digest = [0, 0, 0, 0], logp = [-0.5, -0.9, -0.52, -0.53])

    def verdict(test, r2, **kw):
        res = Res({("p", "t"): runrec(a = test), ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = r2)})
        return EC.evaluate_probs(probs_cmp(**kw), res)

    assert verdict(same, same)["verdict"] == EC.IDENT
    r = verdict(moved, same)
    assert r["verdict"] == EC.DIFF and "identico" in r["why"]
    assert verdict(moved, moved)["verdict"] == EC.NOISE                 # within twice the floor
    assert verdict(far, moved)["verdict"] == EC.DIFF                    # 0.4 against a floor of 0.03
    # Same logits but other drafts: the MTP head computed something else
    r = verdict(dict(same, accepted = 2, rejected = 2), same)
    assert r["verdict"] == EC.DIFF and "bozze" in r["why"]
    assert verdict(dict(same, accepted = 2, rejected = 2), same, drafts = False)["verdict"] == EC.IDENT
    # Information only
    assert verdict(far, same, expect = "info")["verdict"] == EC.INFO
    # No floor at all
    res = Res({("p", "t"): runrec(a = moved), ("p", "r1"): runrec(a = same)})
    assert EC.evaluate_probs(probs_cmp(floor = None), res)["verdict"] == EC.DIFF


def test_capture_problems_make_errors_not_differences():
    same = view([1, 2, 3])
    res = Res({("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same)})
    assert EC.evaluate_probs(probs_cmp(), res)["verdict"] == EC.ERR                         # no test run
    res = Res({("p", "t"): dict(runrec(a = same), ok = False, error = "boom"), ("p", "r1"): runrec(a = same)})
    r = EC.evaluate_probs(probs_cmp(), res)
    assert r["verdict"] == EC.ERR and "boom" in r["why"]
    res = Res({("p", "t"): runrec(a = view([1, 9, 3])), ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same)})
    assert EC.evaluate_probs(probs_cmp(), res)["verdict"] == EC.ERR                         # not the forced tokens
    # The test run did not exercise what it is about
    t = dict(runrec(a = same), state = {"thread": "unavailable", "thread_reason": "tensor-parallel model"})
    res = Res({("p", "t"): t, ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same)})
    r = EC.evaluate_probs(probs_cmp(requires = {"thread": "on"}), res)
    assert r["verdict"] == EC.ERR and "tensor-parallel" in r["why"]
    t = dict(runrec(a = same), stats = {"stagger": {"steps": 0}})
    res = Res({("p", "t"): t, ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same)})
    assert EC.evaluate_probs(probs_cmp(requires = {"stagger_calls": True}), res)["verdict"] == EC.ERR


def test_the_positions_generated_during_an_ingestion():
    # a generated 2 tokens by t=1, the reader was enqueued at 1.5, got its first token at 4.0
    t = runrec(a = view(list(range(10))), r = dict(view([1]), enqueued = 1.5, first_token = 4.0))
    t["order"] = ["a", "r"]
    t["timeline"] = [[1.0, [2, 0]], [2.0, [4, 0]], [3.0, [6, 0]], [4.0, [8, 1]], [5.0, [10, 2]]]
    ref = runrec(a = view(list(range(10))))
    res = Res({("p", "t"): t, ("p", "r1"): ref, ("p", "r2"): ref})
    r = EC.evaluate_probs(probs_cmp(upto = {"window": "r"}, drafts = False), res)
    assert r["verdict"] == EC.IDENT and r["window"] == (2, 6) and r["test"]["n"] == 6
    # A decoder that generated nothing in the window tests nothing
    t["timeline"] = [[1.0, [2, 0]], [4.0, [2, 1]]]
    assert EC.evaluate_probs(probs_cmp(upto = {"window": "r"}, drafts = False), res)["verdict"] == EC.ERR


def test_two_runs_are_compared_only_while_their_decode_batches_are_the_same():
    # Scenario 10 on the cards, 2026-10-25: the second job joined the first's decode at a's 22nd
    # token in one run and at its 24th in the other (wall time, before `r` was queued), and `r`
    # joined at a's 640th / b's 627th in one and a's 397th / b's 365th in the other (exact mode
    # ingests it faster with the switch off). Past those points the two runs part by design
    def run(b_at, r_at_a, r_at_b, n = 700):
        line, a_n, b_n, t = [], 0, 0, 0.0
        b_t = r_t = None
        while a_n < n:
            t += 1.0
            a_n += 1
            if a_n > b_at:
                b_n += 1
                if b_t is None:
                    b_t = t
            if r_t is None and (a_n > r_at_a or (b_n > r_at_b and b_t is not None)):
                r_t = t
            line.append([t, [a_n, b_n, 1 if r_t is not None else 0]])
        rec = runrec(a = view(list(range(n))), b = view(list(range(n))), r = dict(view([1]), enqueued = 48.5))
        rec["order"], rec["timeline"] = ["a", "b", "r"], line
        return rec
    on, off = run(22, 640, 627), run(24, 397, 375)
    assert EC.same_batch_upto(on, off, "a", "a") == 22
    assert EC.same_batch_upto(on, off, "b", "b") == min(EC.tokens_before(on, "b", EC.first_sample_time(off, "r")),
                                                        EC.tokens_before(off, "b", EC.first_sample_time(off, "r")))
    # The same history: as far as the records go
    assert EC.same_batch_upto(on, on, "a", "a") == 700
    # In a comparison, with the reader's window in both runs; the floor of the jobs alone
    # (no reader in it) is taken over the comparison's own positions
    alone = runrec(a = view(list(range(700))), b = view(list(range(700))))
    res = Res({("p", "t"): on, ("p", "r1"): off, ("p", "a1"): alone, ("p", "a2"): alone})
    c = EC.cmp(10, "x", "label", EC.ref("p", "t", "b"), EC.ref("p", "r1", "b"),
               floor = (EC.ref("p", "a1", "b"), EC.ref("p", "a2", "b")),
               upto = {"window": "r", "both": True, "batch": True}, drafts = False)
    row = EC.evaluate_probs(c, res)
    assert row["verdict"] == EC.IDENT and row["test"]["n"] == EC.same_batch_upto(on, off, "b", "b")
    assert row["floor"] is not None and row["floor"]["n"] == row["test"]["n"]
    c["test"], c["ref"] = EC.ref("p", "t", "a"), EC.ref("p", "r1", "a")
    c["floor"] = (EC.ref("p", "a1", "a"), EC.ref("p", "a2", "a"))
    row = EC.evaluate_probs(c, res)
    assert row["verdict"] == EC.ERR and "durante l'ingestione" in row["why"]     # 22 < the 48 before `r`


def test_a_comparison_expected_within_the_noise_takes_its_yardsticks():
    same = view([1, 2, 3, 4])
    resumed = view([1, 2, 3, 4], digest = [0] * 4, logp = [-0.5, -0.52, -0.52, -0.53])
    upstream = view([1, 2, 3, 4], digest = [1] * 4, logp = [-0.5, -0.54, -0.5, -0.5])
    res = Res({("p", "t"): runrec(a = resumed), ("p", "r1"): runrec(a = same), ("p", "up"): runrec(a = upstream)})
    c = EC.cmp(5, "y", "label", EC.ref("p", "t", "a"), EC.ref("p", "r1", "a"), expect = "noise",
               yards = [(EC.ref("p", "r1", "a"), EC.ref("p", "up", "a"))])
    assert EC.evaluate_probs(c, res)["verdict"] == EC.NOISE
    c["yards"] = []
    assert EC.evaluate_probs(c, res)["verdict"] == EC.DIFF


def free_cmp(**kw):
    return probs_cmp(free = True, **kw)


def test_free_runs_compare_up_to_where_their_tokens_part():
    same = view([1, 2, 3, 4])
    res = Res({("p", "t"): runrec(a = same), ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same)})
    assert EC.evaluate_probs(free_cmp(), res)["verdict"] == EC.IDENT
    # The same logits up to and at the position, another token there: the sampling, not the computation
    sampled = view([1, 2, 9, 4], top = [[1, 0], [2, 0], [3, 0], [4, 0]], logp = [-0.5, -0.51, -3.0, -0.53])
    res = Res({("p", "t"): runrec(a = sampled), ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same)})
    r = EC.evaluate_probs(free_cmp(), res)
    assert r["verdict"] == EC.DIFF and "campionamento" in r["why"] and r["test"]["n"] == 3 and r["test"]["parted"] == 2
    assert r["test"]["max"] == pytest.approx(0.0) and any("si separano alla posizione 2" in n for n in r["notes"])
    # Other logits at the position where they part: a difference of the computation, judged on the floor
    moved = view([1, 2, 9, 4], digest = [100, 101, 7, 103], top = [[1, 0], [2, 0], [9, 3], [4, 0]])
    res = Res({("p", "t"): runrec(a = moved), ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same)})
    r = EC.evaluate_probs(free_cmp(), res)
    assert r["verdict"] == EC.DIFF and r["test"]["first_diff"] == 2 and "campionamento" not in r["why"]
    # ... and within a floor whose own free runs part the same way
    res = Res({("p", "t"): runrec(a = moved), ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = moved)})
    r = EC.evaluate_probs(free_cmp(), res)
    assert r["verdict"] == EC.NOISE, r
    # Forced runs whose tokens differ are still an error
    res = Res({("p", "t"): runrec(a = sampled), ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same)})
    assert EC.evaluate_probs(probs_cmp(), res)["verdict"] == EC.ERR
    # Parting past the positions compared is no parting
    res = Res({("p", "t"): runrec(a = sampled), ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same)})
    assert EC.evaluate_probs(free_cmp(upto = 2), res)["verdict"] == EC.IDENT


def test_a_window_bound_in_both_runs():
    # The reader joins the batch after 6 positions of `a` in the test run and after 4 in the reference
    def run(first):
        t = runrec(a = view(list(range(10))), r = dict(view([1]), enqueued = 1.5, first_token = first))
        t["order"] = ["a", "r"]
        t["timeline"] = [[1.0, [2, 0]], [2.0, [4, 0]], [3.0, [6, 0]], [4.0, [8, 1]], [5.0, [10, 2]]]
        for when, counts in t["timeline"]:
            if when >= first:
                counts[1] = max(1, counts[1])
        return t
    res = Res({("p", "t"): run(4.0), ("p", "r1"): run(3.0), ("p", "r2"): run(4.0)})
    r = EC.evaluate_probs(probs_cmp(upto = {"window": "r"}, drafts = False), res)
    assert r["test"]["n"] == 6
    r = EC.evaluate_probs(probs_cmp(upto = {"window": "r", "both": True}, drafts = False), res)
    assert r["verdict"] == EC.IDENT and r["test"]["n"] == 4
    # The floor's two runs bound their own comparison
    assert r["floor"]["n"] == 4


def test_what_the_mtp_decode_rows_ask_of_the_test_run():
    same = view([1, 2, 3])

    def verdict(stats, **req):
        t = dict(runrec(a = same), stats = stats)
        res = Res({("p", "t"): t, ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same)})
        return EC.evaluate_probs(probs_cmp(requires = req), res)

    good = {"verify": {"windows": 3}, "update": {"fold_drafts": 2}, "window": {"scheduled": 4, "w0": 1, "w3": 3}}
    assert verdict(good, verify_windows = True, folds = True, scheduled = True, windows_varied = True)["verdict"] == EC.IDENT
    r = verdict(dict(good, verify = {"windows": 0, "serial": 7}), verify_windows = True)
    assert r["verdict"] == EC.ERR and "7 finestre lasciate al percorso seriale" in r["why"]
    assert "mtp_batched_update" in verdict(dict(good, update = {"folds": 0}), folds = True)["why"]
    assert "programma" in verdict(dict(good, window = {"scheduled": 0}), scheduled = True)["why"]
    r = verdict(dict(good, window = {"scheduled": 4, "w0": 0, "w3": 4}), windows_varied = True)
    assert r["verdict"] == EC.ERR and "una sola finestra presa ([3])" in r["why"]


def test_the_worst_verdict_decides_a_scenario():
    assert EC.worst([EC.IDENT, EC.NOISE, EC.INFO]) == EC.NOISE
    assert EC.worst([EC.IDENT, EC.ERR, EC.DIFF]) == EC.DIFF
    assert EC.worst([EC.INFO]) == EC.NOTRUN
    assert EC.worst([EC.PASS, EC.HANG]) == EC.HANG


# --------------------------------------------------------------------------------------------
#   The disk tier's files
# --------------------------------------------------------------------------------------------

def write_page(root, phash: bytes, tokens: bytes, image: bytes, crc = None):
    d = os.path.join(root, "fp0123456789abcd", "pages", phash[:1].hex())
    os.makedirs(d, exist_ok = True)
    meta = {"phash": phash.hex(), "prev_hash": None, "fingerprint": "x", "image_size": len(image), "page_size": 256}
    meta["crc32"] = zlib.crc32(image, zlib.crc32(tokens)) if crc is None else crc
    header = json.dumps(meta).encode()
    with open(os.path.join(d, phash.hex() + ".kv"), "wb") as f:
        f.write(EC.PAGE_MAGIC + len(header).to_bytes(4, "little") + len(tokens).to_bytes(4, "little"))
        f.write(header + tokens + image)


def write_state(root, phash: bytes, data: bytes, tail = False):
    d = os.path.join(root, "fp0123456789abcd", "states")
    os.makedirs(d, exist_ok = True)
    header = json.dumps({"phash": phash.hex(), "position": 512, "top": {"tail": tail} if tail else {}}).encode()
    with open(os.path.join(d, phash.hex() + ".rs"), "wb") as f:
        f.write(EC.STATE_MAGIC + len(header).to_bytes(4, "little") + bytes(4) + header + data)


def test_the_disk_scan_reads_back_every_page(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    for root in (a, b):
        write_page(root, b"\x01" * 16, b"t" * 512, os.urandom(0) + b"img" * 1000)
        write_page(root, b"\x02" * 16, b"u" * 512, b"other" * 999)
        write_state(root, b"\x01" * 16, b"state" * 100, tail = True)
    sa, sb = EC.scan_disk(a), EC.scan_disk(b)
    assert len(sa["pages"]) == 2 and all(p["ok"] for p in sa["pages"].values())
    assert sa["states"][(b"\x01" * 16).hex()]["tail"] and sa["states"][(b"\x01" * 16).hex()]["position"] == 512
    assert EC.crc_compare(sa, sb)["identical"]
    # A page whose bytes changed on the drive no longer gives the CRC its header carries
    write_page(b, b"\x02" * 16, b"u" * 512, b"othe!" * 999, crc = sa["pages"][(b"\x02" * 16).hex()]["crc"])
    m = EC.crc_compare(sa, EC.scan_disk(b))
    assert not m["identical"] and m["broken"] == 1 and m["differ"] == 1
    # A page on one side only
    write_page(a, b"\x03" * 16, b"v" * 512, b"x" * 10)
    m = EC.crc_compare(EC.scan_disk(a), EC.scan_disk(b))
    assert m["only_ref"] == 1 and not m["identical"]
    assert EC.scan_disk(str(tmp_path / "none"))["bad"] == ["no such directory"]


# --------------------------------------------------------------------------------------------
#   The plan
# --------------------------------------------------------------------------------------------

def args_for(*extra):
    return EC.parse_cli(["--out", "/nonexistent-exact-check"] + list(extra))


def all_refs(plan):
    """Every (proc, run) a forced token, a turn's answer or a comparison names."""
    out = []
    for p in plan["procs"]:
        for st in p["steps"]:
            for j in st.get("jobs", []) if st["op"] == "run" else []:
                if "forced" in j:
                    out.append((p["name"], st["id"], j["forced"]))
        for spec in p["prompts"].values():
            if spec.get("kind") == "turn":
                out.append((p["name"], None, spec["answer"]))
    return out


@pytest.mark.parametrize("scenarios", [None, "8", "9", "10", "11", "12"])
def test_the_plan_refers_only_to_runs_that_come_before(scenarios):
    plan = EC.build_plan(args_for(*(["--scenarios", scenarios] if scenarios else [])))
    seen = set()
    order = {}
    for p in plan["procs"]:
        ids = [st["id"] for st in p["steps"]]
        assert len(ids) == len(set(ids)), f"{p['name']}: run ids repeat"
        for k, st in enumerate(p["steps"]):
            order[(p["name"], st["id"])] = (plan["procs"].index(p), k)
            seen.add((p["name"], st["id"]))
    for proc, rid, r in all_refs(plan):
        assert (r["proc"], r["run"]) in seen, (proc, rid, r)
        if rid is not None:
            # a forced reference is to a run that is over by then
            assert order[(r["proc"], r["run"])] < order[(proc, rid)], (proc, rid, r)
    for c in plan["comparisons"]:
        for key in ("test", "ref"):
            r = c.get(key)
            if isinstance(r, dict) and r.get("run"):
                assert (r["proc"], r["run"]) in seen, (c["id"], r)
        for pair in [c.get("floor"), c.get("floor2")] + list(c.get("yards") or []):
            if pair and isinstance(pair[0], dict):
                for r in pair:
                    assert (r["proc"], r["run"]) in seen, (c["id"], r)


def test_inline_references_come_before_the_thread_and_the_pipeline():
    """The prefill thread splits the CPU expert worker's slots at its first round, and the
    pipeline once more: every run without them comes first, in a process, as in the benches."""
    plan = EC.build_plan(args_for("--scenarios", "3,4"))
    for p in plan["procs"]:
        level = 0
        for st in p["steps"]:
            if st["op"] == "generator":
                level = 0
            if st["op"] != "run":
                continue
            sw = st.get("switches") or {}
            now = 2 if sw.get("pipeline") else 1 if sw.get("thread") else 0
            assert now >= level or st["id"].startswith("s4.disk"), (p["name"], st["id"])
            level = max(level, now)


def test_the_documents_end_in_the_last_chunks_asked_for():
    for lc in (33, 48, 64, 65, 96, 100, 144, 145, 199, 249):
        n = EC.doc_length(lc)
        assert (n - 1) % 256 == lc and n > 8192
    plan = EC.build_plan(args_for("--scenarios", "3"))
    lengths = {spec["length"] for p in plan["procs"] for spec in p["prompts"].values() if spec["kind"] == "doc"}
    assert {EC.doc_length(lc) for lc in (33, 249)} <= lengths


def test_the_groups_of_a_staggered_run():
    assert EC.groups_of(2) == ["a", "b"] and EC.groups_of(3) == ["ab", "c"]
    assert EC.groups_of(4) == ["ab", "cd"] and EC.groups_of(5) == ["abc", "de"]


def test_scenario_1_compares_the_fork_with_two_stock_processes():
    plan = EC.build_plan(args_for("--scenarios", "1"))
    libs = [(p["name"], p["library"]) for p in plan["procs"]]
    assert libs == [("s1-stock-a", "stock"), ("s1-fork", "fork"), ("s1-stock-b", "stock")]
    # The same runs in the three, with the stock process's tokens forced in all of them
    runs = [[st["id"] for st in p["steps"] if st["op"] == "run"] for p in plan["procs"]]
    assert runs[0] == runs[1] == runs[2]
    for p in plan["procs"]:
        for st in p["steps"]:
            for j in st.get("jobs", []) if st["op"] == "run" else []:
                if "forced" in j:
                    assert j["forced"]["proc"] == "s1-stock-a"
    c = next(c for c in plan["comparisons"] if c["id"] == "s1.a.a")
    assert c["test"]["proc"] == "s1-fork" and c["floor"][1]["proc"] == "s1-stock-b"


def test_scenario_6_shares_the_processes_of_scenario_1_or_has_its_own():
    both = EC.build_plan(args_for("--scenarios", "1,6"))
    assert [p["name"] for p in both["procs"]] == ["s1-stock-a", "s1-fork", "s1-stock-b"]
    steps = [st["id"] for st in both["procs"][1]["steps"]]
    assert steps.index("s6.reset") > steps.index("s1.disk.scan")
    alone = EC.build_plan(args_for("--scenarios", "6"))
    assert [p["name"] for p in alone["procs"]] == ["s6-stock-a", "s6-fork", "s6-stock-b"]
    assert all(any(st["op"] == "sweep" for st in p["steps"]) for p in alone["procs"])


def test_scenario_8_runs_the_same_forced_runs_with_and_without_the_ring():
    assert 8 not in args_for().scenarios
    plan = EC.build_plan(args_for("--scenarios", "8"))
    procs = [(p["name"], p["library"], p["env"]) for p in plan["procs"]]
    assert procs == [("s8-fork-full", "fork", {}), ("s8-fork-ring", "fork", {"EXL3_QSA_RAW_TAIL": "1"})]
    full, ring = plan["procs"]
    forced = lambda p: [st["id"] for st in p["steps"] if st["op"] == "run" and not st.get("warmup")
                        and all("forced" in j for j in st["jobs"])]
    # (the full plane's process ingests the document once more before the free turn it forces on the others)
    assert [x for x in forced(full) if x != "s8.turn.prep0"] == forced(ring) and "s8.disk.turn" in forced(ring)
    # the references are the full plane's
    for st in ring["steps"]:
        for j in st.get("jobs", []) if st["op"] == "run" else []:
            if "forced" in j:
                assert j["forced"]["proc"] == "s8-fork-full"
    # the document is past the indexer's sparse threshold (Qwen3.8-Flash-Next: about 2k tokens)
    lengths = [spec["length"] for spec in ring["prompts"].values() if spec["kind"] == "doc"]
    assert max(lengths) > 8192
    probs = [c for c in plan["comparisons"] if c["kind"] == "probs"]
    assert len(probs) == 8
    for c in probs:
        assert c["test"]["proc"] == "s8-fork-ring" and c["ref"]["proc"] == "s8-fork-full" and c["expect"] == "identical"
        assert {r["proc"] for r in c["floor"]} == {"s8-fork-full"} and c["test"]["run"] == c["ref"]["run"]
    kinds = {c["kind"] for c in plan["comparisons"]}
    assert {"raw_tail", "same_placement", "library"} <= kinds
    # the fork's switch is scrubbed from every child unless a process asks for it
    assert "EXL3_QSA_RAW_TAIL" in EC.FORK_ENV_SWITCHES
    assert not EC.switch_is_off("qsa_raw_tail", 32) and EC.switch_is_off("qsa_raw_tail", 0)


def test_scenario_8_reports_the_ring_and_the_memory_per_card():
    plan = EC.build_plan(args_for("--scenarios", "8"))
    res = fake_results(plan)
    mib = 1 << 20
    res.records["s8-fork-full"]["load"].update(
        free = {"0": 1000 * mib, "1": 800 * mib}, draft = True,
        qsa_raw_tail = {"main": 0, "draft": 0, "layers": 0, "saved": {}})
    res.records["s8-fork-ring"]["load"].update(
        free = {"0": 1474 * mib, "1": 1353 * mib}, draft = True,
        qsa_raw_tail = {"main": 32, "draft": 32, "layers": 13, "saved": {"0": 474 * mib, "1": 553 * mib}})
    report = EC.render_report(plan, res, {"started": "s", "ended": "e"}, True)
    assert summary_cells(report)[0][1] == EC.IDENT
    assert "anello di 32 token in 13 strati" in report
    assert "| cuda:0 | 1000 | 1474 | 474 | 474 |" in report and "| cuda:1 | 800 | 1353 | 553 | 553 |" in report
    # the ring missing where it was asked for (a library without it, or refused), or there where it was not
    res.records["s8-fork-ring"]["load"]["qsa_raw_tail"] = {"main": None, "draft": None, "layers": 0, "saved": {}}
    report = EC.render_report(plan, res, {"started": "s"}, True)
    assert summary_cells(report)[0][1] == EC.FAIL and "non aveva l'anello" in report
    res.records["s8-fork-ring"]["load"]["qsa_raw_tail"] = {"main": 32, "draft": 0, "layers": 12, "saved": {}}
    assert "testa MTP non aveva l'anello" in EC.render_report(plan, res, {"started": "s"}, True)


def test_scenario_9_holds_every_generator_in_one_process_and_the_ring_in_another():
    assert 9 not in args_for().scenarios
    args = args_for("--scenarios", "9")
    plan = EC.build_plan(args)
    procs = [(p["name"], p["library"], p["env"], p["server_args"]) for p in plan["procs"]]
    assert procs == [("s9-fork-full", "fork", {}, ["--draft-num-tokens", "3"]),
                     ("s9-fork-ring", "fork", {"EXL3_QSA_RAW_TAIL": "1"}, ["--draft-num-tokens", "3"])]
    full, ring = plan["procs"]
    gens = {st["id"]: st["switches"] for st in full["steps"] if st["op"] == "generator"}
    assert list(gens) == ["g.off", "g.v", "g.u", "g.vu", "g.s3", "g.mix1", "g.mix2", "g.w1"]
    prod = {"stagger": True, "thread": True, "pipeline": True, "tail": True}
    for g, sw in gens.items():
        assert {k: sw[k] for k in prod} == prod, g
    assert gens["g.off"] == prod and gens["g.v"] == dict(prod, verify = True) and gens["g.u"] == dict(prod, update = True)
    assert gens["g.s3"] == dict(prod, verify = True, update = True, window = "cost", schedule = [3])
    assert gens["g.mix1"]["schedule"] == [0, 1, 2, 3] and gens["g.mix2"]["schedule"] == [3, 0, 3, 1]
    assert gens["g.w1"] == dict(prod, draft_tokens = 1)
    # The ring's process: the production generator only, with the same runs as the full plane's
    assert [st["id"] for st in ring["steps"] if st["op"] == "generator"] == ["g.s3"]
    runs = lambda p, pre: [st["id"] for st in p["steps"] if st["op"] == "run" and st["id"].startswith(pre)]
    assert runs(ring, "s9.s3.free1") == runs(full, "s9.s3.free1") and runs(ring, "s9.s3.forced1") == runs(full, "s9.s3.forced1")
    # Every forced token is a free run of `off` in the full plane's process, the warm-ups come first
    for p in plan["procs"]:
        ids = [st["id"] for st in p["steps"] if st["op"] == "run"]
        assert ids[:3] == ["s9.warm.a", "s9.warm.abc", "s9.warm.doc"]
        for st in p["steps"]:
            for j in st.get("jobs", []) if st["op"] == "run" else []:
                if "forced" in j:
                    assert j["forced"]["proc"] == "s9-fork-full" and j["forced"]["run"].startswith("s9.off.free1.")
    # The document beside two jobs generating: past the sparse threshold, several chunks, and two
    # jobs of 1 + W rows within side_max_rows (the chunks run beside the decode, the cost window's
    # cap clips no window up to W)
    doc_run = next(st for st in full["steps"] if st["id"] == "s9.off.free1.doc")
    assert [j["name"] for j in doc_run["jobs"]] == ["a", "b", "r"] and doc_run["until"] == {"job": "r"}
    assert 2 * (1 + args.s9_window) <= EC.SIDE_MAX_ROWS and EC.SIDE_MAX_ROWS // 2 - 1 >= args.s9_window
    lengths = [spec["length"] for spec in full["prompts"].values() if spec["kind"] == "doc"]
    assert max(lengths) > 3 * args.chunk
    turn = next(st for st in full["steps"] if st["id"] == "s9.off.free1.turn")
    assert not turn["forget"] and full["prompts"]["turn/s9"]["answer"] == EC.ref("s9-fork-full", "s9.off.free1.doc", "r")


def test_scenario_9_s_comparisons_and_what_they_expect():
    plan = EC.build_plan(args_for("--scenarios", "9"))
    probs = [c for c in plan["comparisons"] if c["kind"] == "probs"]
    by = {}
    for c in probs:
        by.setdefault(c["group"], []).append(c)
    assert {g: len(v) for g, v in by.items()} == {"a1": 8, "a2": 8, "b": 16, "c": 16, "c0": 7, "d": 8, "e": 16, "f": 8}
    for g in ("a1", "a2", "b", "c0", "e"):
        assert all(c["expect"] == "identical" for c in by[g]), g
    for g in ("c", "d", "f"):
        for c in by[g]:
            # Against the shape meter: the same forced tokens, fixed W=1 against fixed W=3
            assert c["expect"] == "noise" and not c["drafts"]
            assert [(y[0]["run"].rsplit(".", 1)[0], y[1]["run"].rsplit(".", 1)[0]) for y in c["yards"]] == \
                [("s9.off.forced1", "s9.w1.forced1")]
    # (a) compares free runs, which exercise the batched verify; the forced (b) and (e) rows do not ask for it
    for c in by["a1"] + by["a2"]:
        assert c["free"] and c["requires"].get("verify_windows")
    assert all(c["free"] == ("free" in c["test"]["run"]) for c in by["b"] + by["e"])
    assert all(("verify_windows" in c["requires"]) == c["free"] for c in by["b"] + by["e"])
    assert all(c["requires"].get("scheduled") for c in by["b"] + by["c"] + by["c0"] + by["e"] + by["f"])
    assert all(c["requires"].get("windows_varied") for c in by["c"])
    assert all(c["requires"].get("folds") for c in by["d"] + by["f"])
    # (e) and (f) take the test from the ring's process
    assert {c["test"]["proc"] for c in by["e"] + by["f"]} == {"s9-fork-ring"}
    assert {c["ref"]["proc"] for c in probs} == {"s9-fork-full"}
    # The jobs beside the document: up to where it joins the batch in either run, chunks beside asked for
    for c in probs:
        if c["test"]["run"].endswith(".doc") and c["test"]["job"] in "ab":
            assert c["upto"] == {"window": "r", "both": True}
            assert c["requires"]["beside"] and c["requires"]["pipeline"] == "on" and c["soft"] == {"piped": True}
    # The mixed schedule repeated: no document, which joins at a step that depends on wall time
    assert not any(c["test"]["job"] == "r" for c in by["c0"])
    kinds = {c["kind"] for c in plan["comparisons"]}
    assert {"raw_tail", "same_placement", "library"} <= kinds


def test_scenario_9_s_estimate_fits_the_gpu_window_and_its_options():
    args = args_for("--scenarios", "9")
    plan = EC.build_plan(args)
    est = EC.estimate_plan(plan, args)
    # Two loads of a minute, about 20 minutes in all
    assert 12 * 60 <= est["total"] <= 22 * 60, est["total"] / 60
    assert len(plan["procs"]) == 2
    other = args_for("--scenarios", "9", "--s9-mixed", "1,3/0,3", "--s9-window", "3")
    assert other.s9_mixed == [[1, 3], [0, 3]]
    gens = EC.s9_generators(other)
    assert gens["mix1"]["schedule"] == [1, 3] and gens["mix2"]["schedule"] == [0, 3]
    with pytest.raises(SystemExit):
        EC.parse_cli(["--scenarios", "9", "--s9-window", "2", "--s9-mixed", "0,3"])


def test_scenario_9_reports_each_group_with_what_it_expects():
    plan = EC.build_plan(args_for("--scenarios", "9"))
    res = fake_results(plan)
    mib = 1 << 20
    res.records["s9-fork-full"]["load"].update(draft = True, free = {"0": 1000 * mib},
                                               qsa_raw_tail = {"main": 0, "draft": 0, "layers": 0, "saved": {}})
    res.records["s9-fork-ring"]["load"].update(draft = True, free = {"0": 1474 * mib},
                                               qsa_raw_tail = {"main": 32, "draft": 32, "layers": 13, "saved": {"0": 474 * mib}})
    report = EC.render_report(plan, res, {"started": "s", "ended": "e"}, True)
    assert summary_cells(report)[0][1] == EC.IDENT, summary_cells(report)
    for g, title, text in EC.S9_GROUPS:
        assert f"### {title}: {EC.IDENT}" in report and text in report
    assert "**identico**" in report and "**entro il rumore**" in report
    # The counters of every run: windows sampled at once, folds, the schedule's steps, the windows taken
    assert "| `s9-fork-full` / `s9.v.free1.a` | 10.0 | 6 / 0 | 5 / 4 | 7 / 7 | 0:2 3:5 |" in report
    # A row of (a) whose run sampled no window at once is an error, not a pass
    res.records["s9-fork-full"]["s9.v.free1.a"]["stats"]["verify"] = {"windows": 0, "serial": 9}
    rows = {r["id"]: r for r in EC.evaluate_all(plan, res)}
    r = rows["s9.a1.v.free1.a.a"]
    assert r["verdict"] == EC.ERR and "mtp_batched_verify non esercitato" in r["why"]


def test_scenario_10_judges_the_prompt_against_its_yardsticks_and_the_jobs_beside_as_identical():
    plan = EC.build_plan(args_for("--scenarios", "10"))
    (p,) = plan["procs"]
    assert p["env"] == {"EXL3_MOE_STREAM_T_BESIDE": "1"}
    sw = {st["id"]: st.get("switches") or {} for st in p["steps"] if st["op"] == "run"}
    on = {k for k, v in sw.items() if v.get("beside_t") == 1}
    off = {k for k, v in sw.items() if v.get("beside_t") == 0}
    assert on and off and all(sw[k].get("pipeline") for k in on | off)
    # The first run in flight sets the prefill thread up with the switch as the process made it:
    # Generator._side_plan lists the hosts that have a value then, once per generator
    first = next(st for st in p["steps"] if (st.get("switches") or {}).get("pipeline"))
    assert "beside_t" not in first["switches"] and first.get("warmup")
    assert sw["s10.n1.alone.all"] == {"stream_t": 1}
    # The prompt's logits move with the switch (within the bigger yardstick); everything else is the same
    differ = {f"s10.n{n}.{k}" for n in (1, 2) for k in ("on", "shift", "alone.all")} | {"s10.long.on"}
    res = fake_results(plan, differ = differ)
    recs = res.records["s10-fork"]
    for rid, rec in recs.items():
        if rid in on:
            beside = 0 if rid == "s10.alone.on" else 40
            rec["stats"]["stream"] = {"normal": 12, "beside": beside}
        elif rid in off:
            rec["stats"]["stream"] = {"normal": 0, "beside": 0}
    rows = {r["id"]: r for r in EC.evaluate_all(plan, res)}
    for n in (1, 2):
        assert rows[f"s10.n{n}.reader"]["verdict"] == EC.NOISE, rows[f"s10.n{n}.reader"]
        assert rows[f"s10.n{n}.reader.off"]["verdict"] == EC.IDENT
        for i, x in enumerate(EC.NAMES[:n]):
            # Only the job that joins last is judged (the others' batch changes by wall time)
            assert rows[f"s10.n{n}.{x}"]["verdict"] == (EC.IDENT if i == n - 1 else EC.INFO), rows[f"s10.n{n}.{x}"]
            assert rows[f"s10.n{n}.{x}.alone"]["verdict"] == EC.INFO
    assert rows["s10.alone"]["verdict"] == EC.IDENT
    assert rows["s10.long"]["verdict"] == EC.INFO
    report = EC.render_report(plan, res, {"started": "s", "ended": "e"}, True)
    assert summary_cells(report)[0][1] == EC.NOISE, summary_cells(report)
    # The memory per run is in the table of the runs
    assert "`s10-fork` / `s10.long.on`" in report
    # A run that did not plan a layer at the lower threshold did not test the switch: an error
    recs["s10.n2.on"]["stats"]["stream"] = {"normal": 52, "beside": 0}
    recs["s10.alone.on"]["stats"]["stream"] = {"normal": 12, "beside": 3}
    rows = {r["id"]: r for r in EC.evaluate_all(plan, res)}
    assert rows["s10.n2.reader"]["verdict"] == EC.ERR and "soglia bassa" in rows["s10.n2.reader"]["why"]
    assert rows["s10.alone"]["verdict"] == EC.ERR and "doveva essere spento" in rows["s10.alone"]["why"]
    # Over the yardsticks is a difference
    recs["s10.n2.on"]["stats"]["stream"] = {"normal": 12, "beside": 40}
    recs["s10.n2.on"]["jobs"]["r"]["logp"] = [-3.0] * 8
    rows = {r["id"]: r for r in EC.evaluate_all(plan, res)}
    assert rows["s10.n2.reader"]["verdict"] == EC.DIFF and "metro" in rows["s10.n2.reader"]["why"]


def test_the_child_turns_the_lower_threshold_and_the_cards_threshold_per_run():
    """The child's two knobs of scenario 10 on hosts shaped as the fork's MoeCpuHost: the lower
    threshold beside a decode only on hosts made with the variable, given back when a run leaves
    it out; every card's own threshold overridden and given back; the counters summed."""
    made = types.SimpleNamespace(stream_t_beside = 1, decode_beside = False,
                                 beside_stats = {"normal": [0, 3, 2], "beside": [0, 5, 4], "small": [0, 1, 0]},
                                 sstate = {0: {"stream_t": 23}, 1: {"stream_t": 11}})
    plain = types.SimpleNamespace(stream_t_beside = 0, beside_stats = None, sstate = {0: {"stream_t": 23}})
    ch = object.__new__(CH.Child)
    ch.model = types.SimpleNamespace(config = types.SimpleNamespace(moe_cpu_hosts = {0: made, 1: made}))
    ch.container = types.SimpleNamespace(draft_model = types.SimpleNamespace(config = types.SimpleNamespace(moe_cpu_hosts = {0: plain})))
    ch.gen = types.SimpleNamespace(_beside_hosts = (made,))
    ch.stream_t_base = {}
    assert ch.moe_hosts() == [made, plain]
    ch.apply_stream_switches({"beside_t": 0, "stream_t": 1})
    assert made.stream_t_beside == 0 and plain.stream_t_beside == 0
    assert [made.sstate[0]["stream_t"], made.sstate[1]["stream_t"], plain.sstate[0]["stream_t"]] == [1, 1, 1]
    st = ch.state()
    assert st["beside_t"] == [0, 0] and st["beside_hosts"] == 1
    assert st["stream_t"] == {"0:cuda:0": 1, "0:cuda:1": 1, "1:cuda:0": 1}
    ch.apply_stream_switches({})
    assert made.stream_t_beside == 1
    assert [made.sstate[0]["stream_t"], made.sstate[1]["stream_t"], plain.sstate[0]["stream_t"]] == [23, 11, 23]
    assert ch.stats()["stream"] == {"normal": 5, "beside": 9, "small": 1}
    # Asked of a process none of whose hosts the variable made: an error, not a silent no-op
    ch.model = types.SimpleNamespace(config = types.SimpleNamespace(moe_cpu_hosts = {0: plain}))
    ch.container = types.SimpleNamespace(draft_model = None)
    with pytest.raises(CH.RunError):
        ch.apply_stream_switches({"beside_t": 1})
    assert "stream" not in ch.stats()
    # The cards' threshold before any streamed layer made their state
    plain.sstate = {}
    with pytest.raises(CH.RunError):
        ch.apply_stream_switches({"stream_t": 1})


def test_scenario_10_needs_decoders_whose_rows_a_chunk_may_overlap():
    with pytest.raises(SystemExit):
        EC.build_plan(args_for("--scenarios", "10", "--s10-decoders", "5,6"))
    plan = EC.build_plan(args_for("--scenarios", "10", "--s10-decoders", "1", "--s10-long", "0"))
    ids = [st["id"] for st in plan["procs"][0]["steps"]]
    assert not any(i.startswith("s10.long") for i in ids) and not any(i.startswith("s10.n2") for i in ids)


def test_scenario_11_runs_the_same_generators_in_two_processes_that_differ_in_the_host_fast_path_only():
    assert 11 not in args_for().scenarios
    args = args_for("--scenarios", "11")
    plan = EC.build_plan(args)
    procs = [(p["name"], p["library"], p["env"], p["server_args"]) for p in plan["procs"]]
    assert procs == [("s11-fork-plain", "fork", {"EXL3_QSA_RAW_TAIL": "1"}, ["--draft-num-tokens", "3"]),
                     ("s11-fork-hostfast", "fork", {"EXL3_QSA_RAW_TAIL": "1", "EXL3_HOST_FASTPATH": "1"},
                      ["--draft-num-tokens", "3"])]
    assert "EXL3_HOST_FASTPATH" in EC.FORK_ENV_SWITCHES
    plain, fast = plan["procs"]
    gens9 = EC.s9_generators(args)
    for p in plan["procs"]:
        gens = {st["id"]: st["switches"] for st in p["steps"] if st["op"] == "generator"}
        assert gens == {"g.s3": gens9["s3"], "g.off": gens9["off"]}
        ids = [st["id"] for st in p["steps"] if st["op"] == "run"]
        assert ids[:3] == ["s11.warm.a", "s11.warm.abc", "s11.warm.doc"]
        # Every forced token is a free run of `s3` in the process without the fast path
        for st in p["steps"]:
            for j in st.get("jobs", []) if st["op"] == "run" else []:
                if "forced" in j:
                    assert j["forced"]["proc"] == "s11-fork-plain" and j["forced"]["run"].startswith("s11.s3.free1.")
    runs = lambda p, pre: [st["id"] for st in p["steps"] if st["op"] == "run" and st["id"].startswith(pre)]
    for pre in ("s11.s3.free1", "s11.s3.forced1", "s11.off.forced1"):
        assert runs(fast, pre) == runs(plain, pre) and len(runs(plain, pre)) == 4
    # The plain process repeats each set it is compared on, for the floor
    assert runs(plain, "s11.s3.free2") and runs(plain, "s11.s3.forced2") and runs(plain, "s11.off.forced2")
    assert not runs(fast, "s11.s3.free2") and not runs(fast, "s11.off.free")
    assert plain["prompts"]["turn/s11"]["answer"] == EC.ref("s11-fork-plain", "s11.s3.free1.doc", "r")


def test_scenario_11_expects_identical_everywhere_against_the_plain_process_repeated():
    plan = EC.build_plan(args_for("--scenarios", "11"))
    probs = [c for c in plan["comparisons"] if c["kind"] == "probs"]
    by = {}
    for c in probs:
        by.setdefault(c["group"], []).append(c)
    assert {g: len(v) for g, v in by.items()} == {"a": 8, "b": 8, "c": 8}
    for c in probs:
        assert c["expect"] == "identical"
        assert c["test"]["proc"] == "s11-fork-hostfast" and c["ref"]["proc"] == "s11-fork-plain"
        assert c["test"]["run"] == c["ref"]["run"] and c["test"]["job"] == c["ref"]["job"]
        assert {r["proc"] for r in c["floor"]} == {"s11-fork-plain"}
        assert [r["run"].rsplit(".", 1)[0] for r in c["floor"]] == [c["ref"]["run"].rsplit(".", 1)[0],
                                                                    c["ref"]["run"].rsplit(".", 1)[0][:-1] + "2"]
    assert all(c["free"] and c["requires"].get("verify_windows") and c["requires"].get("scheduled") for c in by["a"])
    assert all(not c["free"] and c["requires"].get("scheduled") for c in by["b"])
    assert all(not c["free"] and "scheduled" not in c["requires"] for c in by["c"])
    for c in probs:
        if c["test"]["run"].endswith(".doc") and c["test"]["job"] in "ab":
            assert c["upto"] == {"window": "r", "both": True}
            assert c["requires"]["beside"] and c["requires"]["pipeline"] == "on" and c["soft"] == {"piped": True}
    state = next(c for c in plan["comparisons"] if c["kind"] == "switch_state")
    assert state["name"] == "host_fastpath" and state["expect"] == {"s11-fork-plain": False, "s11-fork-hostfast": True}
    assert {"same_placement", "library"} <= {c["kind"] for c in plan["comparisons"]}
    est = EC.estimate_plan(plan, args_for("--scenarios", "11"))
    assert 5 * 60 <= est["total"] <= 15 * 60, est["total"] / 60


def test_scenario_11_says_when_a_process_did_not_run_the_fast_path_it_was_meant_to():
    plan = EC.build_plan(args_for("--scenarios", "11"))
    res = fake_results(plan)
    for p in plan["procs"]:
        for rec in res.records[p["name"]].values():
            if rec["op"] == "generator":
                rec["switches_state"]["host_fastpath"] = p["name"] == "s11-fork-hostfast"
    report = EC.render_report(plan, res, {"started": "s", "ended": "e"}, True)
    assert summary_cells(report)[0][1] == EC.IDENT, summary_cells(report)
    assert f"## 11. {EC.SCENARIO_TITLES[11]}" in report and "| `s11-fork-hostfast` / `s11.s3.free1.a` |" in report
    assert "host_fastpath" not in report
    # The variable not taken (a library without the switch, or one that read it otherwise): said
    # under the scenario, as every problem of the capture itself
    res.records["s11-fork-hostfast"]["g.s3"]["switches_state"]["host_fastpath"] = False
    report = EC.render_report(plan, res, {"started": "s", "ended": "e"}, True)
    assert "**Attenzione**: `s11-fork-hostfast` (`g.s3`) ha `host_fastpath` = False, doveva essere True" in report
    del res.records["s11-fork-hostfast"]["g.s3"]["switches_state"]["host_fastpath"]
    assert "non dice `host_fastpath`" in EC.render_report(plan, res, {"started": "s", "ended": "e"}, True)
    # A run of the fast path that differs from the plain one is a difference, not noise
    res = fake_results(plan)
    other = fake_results(plan, differ = ("s11.s3.forced1.abc",))
    res.records["s11-fork-hostfast"]["s11.s3.forced1.abc"] = other.records["s11-fork-hostfast"]["s11.s3.forced1.abc"]
    rows = {r["id"]: r for r in EC.evaluate_all(plan, res)}
    assert rows["s11.b.s3.forced1.abc.a"]["verdict"] == EC.DIFF
    assert rows["s11.b.s3.forced1.a.a"]["verdict"] == EC.IDENT


def test_scenario_12_runs_the_same_decode_in_a_process_without_and_one_with_the_variables():
    assert 12 not in args_for().scenarios
    args = args_for("--scenarios", "12")
    plan = EC.build_plan(args)
    procs = [(p["name"], p["library"], p["env"], p["server_args"]) for p in plan["procs"]]
    assert procs == [("s12-fork-base", "fork", {}, ["--draft-num-tokens", "3"]),
                     ("s12-fork-test", "fork", {"EXL3_MOE_COOP_WIDE": "1"}, ["--draft-num-tokens", "3"])]
    base, test = plan["procs"]
    # No switch of the fork: W fixed, then W = 1 over the same caches
    for p in plan["procs"]:
        gens = {st["id"]: st["switches"] for st in p["steps"] if st["op"] == "generator"}
        assert gens == {"g.w3": {}, "g.w1": {"draft_tokens": 1}}
    runs = lambda p: [st["id"] for st in p["steps"] if st["op"] == "run" and not st.get("warmup")]
    assert runs(base)[:2] == ["s12.free.a", "s12.free.b"]
    assert runs(base)[2:] == runs(test)
    assert runs(test) == [f"s12.{g}.{wl}.ref{k}" for g in ("w3", "w1") for wl in ("a", "ab") for k in (1, 2)]
    # Every forced job takes the tokens of its own free run alone, in the base process
    for p in plan["procs"]:
        for st in p["steps"]:
            for j in st.get("jobs", []) if st["op"] == "run" else []:
                if "forced" in j:
                    assert j["forced"] == EC.ref("s12-fork-base", f"s12.free.{j['name']}", j["name"])
    # The rows the variables act on: 1 + W per job, two jobs within the decode-size path's 8
    assert 2 * (1 + args.s12_window) <= EC.S12_MAX_ROWS
    probs = [c for c in plan["comparisons"] if c["kind"] == "probs"]
    assert len(probs) == 6
    for c in probs:
        assert c["expect"] == "noise" and not c["drafts"]
        assert c["test"]["proc"] == "s12-fork-test" and c["ref"]["proc"] == "s12-fork-base"
        assert c["test"]["run"] == c["ref"]["run"] and c["test"]["job"] == c["ref"]["job"]
        assert c["requires"] == {"env": {"proc": "s12-fork-test", "vars": {"EXL3_MOE_COOP_WIDE": "1"}},
                                 "env_absent": {"proc": "s12-fork-base", "vars": ["EXL3_MOE_COOP_WIDE"]}}
        assert {r["proc"] for pair in c["yards"] + [c["floor"]] for r in pair} == {"s12-fork-base"}
        wl, job = c["test"]["run"].split(".")[2], c["test"]["job"]
        # the shape meter for every job, the batch meter for `a`
        assert (EC.ref("s12-fork-base", f"s12.w3.{wl}.ref1", job), EC.ref("s12-fork-base", f"s12.w1.{wl}.ref1", job)) in c["yards"]
        assert len(c["yards"]) == (2 if job == "a" else 1)
    labels = sorted(c["label"] for c in probs)
    assert any("W=3 (8 righe)" in l for l in labels) and any("W=1 (2 righe)" in l for l in labels)
    kinds = {c["kind"] for c in plan["comparisons"]}
    assert {"same_placement", "library"} <= kinds
    # Another set of variables, and what the CLI refuses
    other = EC.build_plan(args_for("--scenarios", "12", "--s12-env", "EXL3_MOE_COOP_KSPLIT=2, EXL3_MOE_SHARED_COOP=0",
                                   "--s12-window", "1"))
    assert other["procs"][1]["env"] == {"EXL3_MOE_COOP_KSPLIT": "2", "EXL3_MOE_SHARED_COOP": "0"}
    assert [st["id"] for st in other["procs"][0]["steps"] if st["op"] == "generator"] == ["g.w1"]
    assert len([c for c in other["comparisons"] if c["kind"] == "probs"]) == 3
    for bad in (["--s12-env", "EXL3_MOE_FUSED_DET=0"], ["--s12-env", "WIDE=1"], ["--s12-env", ""],
                ["--s12-window", "4"], ["--s12-window", "0"]):
        with pytest.raises(SystemExit):
            EC.parse_cli(["--scenarios", "12"] + bad)
    est = EC.estimate_plan(plan, args)
    assert 3 * 60 <= est["total"] <= 20 * 60, est["total"] / 60       # two loads, about 5 minutes


def test_scenario_12_s_rows_need_the_variables_where_they_were_asked_for():
    plan = EC.build_plan(args_for("--scenarios", "12"))
    res = fake_results(plan)
    res.meta["s12-fork-test"]["env"] = {"EXL3_MOE_COOP_WIDE": "1", "EXL3_MOE_FUSED_DET": "1"}
    res.meta["s12-fork-base"]["env"] = {"EXL3_MOE_FUSED_DET": "1"}
    rows = [r for r in EC.evaluate_all(plan, res) if r["kind"] == "probs"]
    assert rows and all(r["verdict"] == EC.IDENT for r in rows), [(r["id"], r.get("why")) for r in rows]
    report = EC.render_report(plan, res, {"started": "s", "ended": "e"}, True)
    assert "## 12. " + EC.SCENARIO_TITLES[12] in report and EC.SCENARIO_TEXT[12] in report
    # The test process without the variable compares the same thing: an error, not a pass
    del res.meta["s12-fork-test"]["env"]["EXL3_MOE_COOP_WIDE"]
    rows = [r for r in EC.evaluate_all(plan, res) if r["kind"] == "probs"]
    assert all(r["verdict"] == EC.ERR and "non aveva EXL3_MOE_COOP_WIDE=1" in r["why"] for r in rows)
    # ... and so does a base process that has it
    res.meta["s12-fork-test"]["env"]["EXL3_MOE_COOP_WIDE"] = "1"
    res.meta["s12-fork-base"]["env"]["EXL3_MOE_COOP_WIDE"] = "1"
    rows = [r for r in EC.evaluate_all(plan, res) if r["kind"] == "probs"]
    assert all(r["verdict"] == EC.ERR and "il riferimento non è senza" in r["why"] for r in rows)


def test_the_exact_environment_of_the_children():
    args = args_for("--env", "EXL3_HOST_MEM_RESERVE_MB=1024")
    base = {"PATH": "/bin", "EXL3_STAGGER_DECODE": "1", "EXL3_MOE_CPU_SPLIT_STATS": "/x", "PYTHONPATH": "/somewhere"}
    stock = EC.child_env(base, {"library": "stock", "scenarios": [1], "env": {}}, args, "/run")
    fork = EC.child_env(base, {"library": "fork", "scenarios": [4], "env": {"EXL3_DISK_CACHE_UNCACHE": "1"}}, args, "/run")
    stress = EC.child_env(base, {"library": "fork", "scenarios": [7], "env": {}}, args, "/run")
    assert "PYTHONPATH" not in stock and fork["PYTHONPATH"] == args.fork_path
    for env in (stock, fork, stress):
        assert "EXL3_STAGGER_DECODE" not in env and "EXL3_MOE_CPU_SPLIT_STATS" not in env
        assert env["EXL3_MOE_CPU_PROMOTE"] == "0" and env["EXL3_MOE_CPU_SLOTS"] == "6"
        assert env["PYTORCH_CUDA_ALLOC_CONF"] == "backend:cudaMallocAsync" and env["XDG_CACHE_HOME"] == "/run/xdg-cache"
        assert env["EXL3_HOST_MEM_RESERVE_MB"] == "1024"
    assert stock["EXL3_MOE_STREAM_FUSED_T"] == "0" and stock["EXL3_MOE_RECON_DET"] == "1"
    assert fork["EXL3_DISK_CACHE_UNCACHE"] == "1" and "EXL3_DISK_CACHE_UNCACHE" not in stock
    # The stress runs production's numerics
    assert "EXL3_MOE_STREAM_FUSED_T" not in stress and "EXL3_MOE_RECON_DET" not in stress


def test_production_flags_for_the_server():
    argv = EC.base_server_argv(args_for(), "/run/procs/x/placement")
    joined = " ".join(argv)
    for part in ("--moe-device-split 23:226,25:253", "--cache-mode 8,8", "--draft-cache-mode 4,4", "--draft-num-tokens 1",
                 "--max-chunk-size 4096", "--recurrent-checkpoint-interval 4096", "--max-batch-size 6",
                 "--moe-cpu-threads 12", "--moe-placement-file /run/procs/x/placement/", "--tensor-parallel false"):
        assert part in joined, part
    assert "--prefill-thread" not in joined and "--moe-swap-prefill-weight" not in joined


def test_the_placement_file_keeps_the_name_the_server_looks_for(tmp_path):
    model = tmp_path / "Qwen3.8 Flash"
    model.mkdir()
    assert EC.placement_name(str(model) + "/") == "moe-placement.Qwen3.8_Flash.json"
    d = tmp_path / "cache"
    d.mkdir()
    assert EC.resolve_placement_file(str(d), str(model)) == str(d / "moe-placement.Qwen3.8_Flash.json")
    assert EC.resolve_placement_file(str(d / "mine.json"), str(model)) == str(d / "mine.json")


def test_the_estimate_and_the_dry_run():
    args = args_for()
    plan = EC.build_plan(args)
    est = EC.estimate_plan(plan, args)
    assert set(est["scenarios"]) == {1, 2, 3, 4, 5, 6, 7}
    assert est["total"] == pytest.approx(sum(est["scenarios"].values()))
    # Every process is a load
    assert est["total"] >= len(plan["procs"]) * args.assume_load_min * 60
    # The stress lasts what it is told
    assert est["scenarios"][7] >= args.s7_minutes * 60
    # Slower rates, longer night
    slow = EC.estimate_plan(plan, args_for("--assume-prefill", "350", "--assume-decode", "30"))
    assert slow["total"] > est["total"] * 1.3
    budgets = EC.scenario_budgets(est, args)
    assert all(budgets[s] >= max(1800.0, 2.9 * est["scenarios"][s]) for s in budgets)
    assert EC.scenario_budgets(est, args_for("--scenario-timeout", "10"))[2] == 600.0
    text = EC.dry_run_text(plan, args, est, budgets, ["modello non trovato: /x"], ["una nota"])
    for s in plan["scenarios"]:
        assert f"Scenario {s} - {EC.SCENARIO_TITLES[s]}" in text
    assert "Totale stimato" in text and "PROBLEMA: modello non trovato" in text and "Nota: una nota" in text
    assert "prefill 700 token/s" in text and "1 min per caricamento del modello (40-65 s misurati" in text


def test_the_cli_refuses_what_it_cannot_do():
    with pytest.raises(SystemExit):
        EC.parse_cli(["--scenarios", "1,99"])
    with pytest.raises(SystemExit):
        EC.parse_cli(["--chunk", "1000"])
    a = EC.parse_cli(["--scenarios", "3,1,3", "--out", "/tmp/x"])
    assert a.scenarios == [1, 3]


# --------------------------------------------------------------------------------------------
#   Records and report
# --------------------------------------------------------------------------------------------

def test_records_survive_a_process_killed_while_writing(tmp_path):
    path = str(tmp_path / "records.jsonl")
    EC.append_jsonl(path, {"id": "a", "x": 1})
    EC.append_jsonl(path, {"id": "b", "x": float("nan")})
    with open(path, "a") as f:
        f.write('{"id": "c", "x": [1, 2')         # cut short
    recs = EC.read_jsonl(path)
    assert [r["id"] for r in recs] == ["a", "b"] and math.isnan(recs[1]["x"])
    assert EC.read_jsonl(str(tmp_path / "none.jsonl")) == []


def fake_results(plan, *, differ = (), missing = ()):
    """Records for every run of a plan, as identical as a deterministic computation gives them;
    `differ` run ids get other logits, `missing` processes wrote nothing."""
    res = EC.Results()
    for p in plan["procs"]:
        if p["name"] in missing:
            continue
        recs = {}
        res.meta[p["name"]] = {"op": "start", "fork_features": p["library"] == "fork", "library_path": "/lib", "version": EC.BASE_RELEASE}
        recs["load"] = {"op": "load", "id": "load", "ok": True, "seconds": 200.0, "placement": "abc"}
        for st in p["steps"]:
            if st["op"] == "generator":
                recs[st["id"]] = {"op": "generator", "id": st["id"], "ok": True, "switches_state": {"stagger_decode": False}}
            elif st["op"] == "run":
                jobs = {}
                for j in st["jobs"]:
                    n = 8
                    v = view([7] * n)
                    if st["id"] in differ:
                        v["digest"] = [1] * n
                        v["logp"] = [-0.4] * n
                    v.update(enqueued = 1.0, first_token = 3.0, prompt_tokens = 9000, cached_tokens = 512, resumed_from_tail = True)
                    if st.get("disk_at_first_token"):
                        v["disk_at_first_token"] = {"page_writes": 40}
                    jobs[j["name"]] = v
                order = list(jobs)
                # The reader `r` samples its first token at 3.0, the others generate from the start
                line = [[0.5, [0] * len(order)], [2.0, [0 if n == "r" else 4 for n in order]],
                        [3.0, [1 if n == "r" else 6 for n in order]]]
                recs[st["id"]] = {"op": "run", "id": st["id"], "scenario": st["scenario"], "ok": True, "jobs": jobs,
                                  "order": order, "timeline": line,
                                  "stats": {"side": {"chunks": 3, "beside": 3, "piped": 2, "ahead": 1}, "stagger": {"steps": 5},
                                            "tail": {"stashes": 4}, "verify": {"windows": 6, "serial": 0},
                                            "update": {"folds": 5, "fold_drafts": 4},
                                            "window": {"scheduled": 7, "steps": 7, "w0": 2, "w1": 0, "w2": 0, "w3": 5}},
                                  "state": {"thread": "on", "pipeline": "on"}, "seconds": 10.0, "left_jobs": 0,
                                  "cancels": [{"job": "r", "progress": 0.51, "seconds": 0.3}] if st.get("cancel") else []}
            elif st["op"] == "disk_scan":
                recs[st["id"]] = {"op": "disk_scan", "id": st["id"], "ok": True,
                                  "scan": {"pages": {"aa": {"crc": 1, "crc_disk": 1, "ok": True}}, "states": {}, "bad": []}}
            elif st["op"] == "sweep":
                recs[st["id"]] = {"op": "sweep", "id": st["id"], "ok": True, "maps": ["m"], "hist_pre": ["h"], "moved": [3], "total_moved": 3}
            elif st["op"] == "stress":
                recs[st["id"]] = {"op": "stress", "id": st["id"], "ok": True,
                                  "stress": {"drained": True, "exceptions": [], "job_errors": 0, "max_step_s": 2.0, "step_limit": 60,
                                             "jobs": 40, "kinds": {"short": 30, "doc": 6, "turn": 4}}}
            elif st["op"] in ("close", "reset_hist"):
                recs[st["id"]] = {"op": st["op"], "id": st["id"], "ok": True}
        res.records[p["name"]] = recs
        res.procs[p["name"]] = {"exit": 0, "wall_s": 100.0, "load_s": 200.0}
    return res


def summary_cells(report):
    body = report.split("## Sintesi", 1)[1].split("\n\n", 2)[1]
    return [[c.strip() for c in l.strip("|").split("|")] for l in body.splitlines()[2:]]


def test_the_report_of_a_night_where_everything_is_the_same():
    args = args_for()
    plan = EC.build_plan(args)
    report = EC.render_report(plan, fake_results(plan), {"started": "s", "ended": "e"}, True)
    rows = summary_cells(report)
    assert [r[0].split(".")[0] for r in rows] == ["1", "2", "3", "4", "5", "6", "7"]
    assert [r[1] for r in rows] == [EC.IDENT] * 6 + [EC.PASS], rows
    assert rows[0][3] != "-" and rows[0][4] == "identico"
    assert "**Attenzione**" not in report
    for s in plan["scenarios"]:
        assert f"## {s}. {EC.SCENARIO_TITLES[s]}" in report


def test_the_report_says_where_it_differs():
    args = args_for("--scenarios", "2,3")
    plan = EC.build_plan(args)
    report = EC.render_report(plan, fake_results(plan, differ = {"s2.n3.stagger"}), {"started": "s"}, False)
    rows = summary_cells(report)
    assert rows[0][1] == EC.DIFF and rows[1][1] == EC.IDENT
    assert "rapporto parziale" in report


def test_a_process_that_never_ran_leaves_errors_and_a_line_about_it():
    args = args_for("--scenarios", "5")
    plan = EC.build_plan(args)
    res = fake_results(plan, missing = {"s5-fork-b"})
    res.procs["s5-fork-b"] = {"exit": 4, "reason": "uscito con 4", "log_tail": "CUDA out of memory"}
    report = EC.render_report(plan, res, {}, True)
    assert summary_cells(report)[0][1] == EC.ERR
    assert "**Attenzione**: il processo `s5-fork-b` si è fermato (uscito con 4)" in report and "CUDA out of memory" in report


def test_capture_problems_are_attention_lines():
    args = args_for("--scenarios", "1")
    plan = EC.build_plan(args)
    res = fake_results(plan)
    res.records["s1-stock-b"]["load"]["placement"] = "other"
    res.meta["s1-stock-a"]["fork_features"] = True
    res.records["s1-fork"]["g0"]["switches_state"] = {"stagger_decode": True}
    res.records["s1-fork"]["s1.a.ref1"]["placement_before"] = "p1"
    res.records["s1-fork"]["s1.a.ref1"]["placement_after"] = "p2"
    notes = EC.capture_checks(plan, res)[1]
    text = "\n".join(notes)
    assert "il piazzamento (al caricamento) non è lo stesso" in text
    assert "`s1-stock-a` doveva girare su stock" in text
    assert "non tutti gli interruttori sono spenti (`g0`): stagger_decode" in text
    assert "il piazzamento degli esperti è cambiato durante il giro" in text


# --------------------------------------------------------------------------------------------
#   The watchdog
# --------------------------------------------------------------------------------------------

def test_a_call_that_does_not_return_leaves_the_stacks_and_ends_the_process(tmp_path):
    script = tmp_path / "hang.py"
    script.write_text(textwrap.dedent(f"""
        import sys, threading, time
        sys.path.insert(0, {EXP!r})
        import exact_check_child as CH
        w = CH.Watch({str(tmp_path)!r}, 1.0)
        w.run = "s9.run"
        def beside():
            while True:
                time.sleep(0.05)
        threading.Thread(target = beside, name = "the-other-thread", daemon = True).start()
        def sleepy_library_call():
            time.sleep(30)
        with w.guard("iterate"):
            sleepy_library_call()
        print("not reached")
    """))
    t0 = time.time()
    out = subprocess.run([sys.executable, str(script)], capture_output = True, text = True, timeout = 60)
    assert out.returncode == 1 and "not reached" not in out.stdout
    assert time.time() - t0 < 20
    stacks = (tmp_path / "hang-stacks.txt").read_text()
    assert "Timeout" in stacks and "sleepy_library_call" in stacks and "beside" in stacks
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["op"] == "iterate" and status["run"] == "s9.run" and status["timeout"] == 1.0


def test_a_call_that_returns_in_time_disarms_the_watchdog(tmp_path):
    w = CH.Watch(str(tmp_path), 0.5)
    try:
        with w.guard("iterate"):
            time.sleep(0.05)
        time.sleep(0.8)         # past the timeout: nothing fires once the call is over
        w.dump("asked")
        assert "asked" in (tmp_path / "hang-stacks.txt").read_text()
    finally:
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
        faulthandler.enable()


# --------------------------------------------------------------------------------------------
#   Orchestration, with a fake child
# --------------------------------------------------------------------------------------------

FAKE_CHILD = textwrap.dedent("""
    import json, os, sys, time, zlib
    sys.path.insert(0, @EXP@)
    import exact_check as EC
    spec = json.load(open(sys.argv[sys.argv.index("--spec") + 1]))
    pdir = spec["proc_dir"]
    path = os.path.join(pdir, "records.jsonl")
    mode = os.environ.get("FAKE_MODE", "")
    def status(op, run = None, scenario = None, timeout = 5):
        json.dump({"t": time.time(), "op": op, "run": run, "scenario": scenario, "timeout": timeout},
                  open(os.path.join(pdir, "status.json"), "w"))
    status("start")
    EC.append_jsonl(path, {"op": "start", "id": "start", "ok": True, "fork_features": spec["library"] == "fork",
                           "library_path": "/fake", "version": EC.BASE_RELEASE, "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")}})
    argv = spec["server_argv"]
    split = argv[argv.index("--moe-device-split") + 1]
    if mode.startswith("splitfail:") and split in mode[len("splitfail:"):].split("|"):
        # As the server's expert tier says it when a layer does not fit where the plan puts it
        print("ERROR openai.moe_tier: MoE device split: a block of that kind takes about 400 MiB there, plus the "
              "largest transient (300 MiB measured) and the library's 256 MiB margin. One unit of split on cuda:0 is "
              "56 MiB: try --moe-device-split 23:240,25:253 (estimate), or give that card one layer less", flush = True)
        EC.append_jsonl(path, {"op": "load", "id": "load", "ok": False, "error": "MoeDeviceSplitError: does not fit"})
        sys.exit(4)
    EC.append_jsonl(path, {"op": "load", "id": "load", "ok": True, "seconds": 1.0, "placement": "p0", "split": split})
    if mode == "crash:" + spec["proc"]:
        sys.exit(5)
    local = {}
    for st in spec["steps"]:
        status(st["op"], st.get("id"), st.get("scenario"))
        if mode == "hang:" + spec["proc"] + ":" + st.get("id", ""):
            time.sleep(600)
        rec = {"op": st["op"], "id": st.get("id"), "scenario": st.get("scenario"), "ok": True}
        if st["op"] == "run":
            jobs = {}
            for j in st["jobs"]:
                f = j.get("forced")
                if f is None:
                    toks = [(len(j["prompt"]) * 7 + i * 3) % 50 for i in range(min(j["max_new"], 12))]
                elif "tokens" in f:
                    toks = f["tokens"]
                elif "missing" in f:
                    rec.update(ok = False, error = "missing " + f["missing"]); break
                else:
                    toks = local[f["run"]][f["job"]][:f.get("limit") or None]
                jobs[j["name"]] = {"tokens": toks, "digest": [zlib.crc32(repr((j["prompt"], toks[:i + 1])).encode()) for i in range(len(toks))],
                                   "logp": [-0.1] * len(toks), "topi": [[t, 0] for t in toks], "topl": [[-0.1, -3.0]] * len(toks),
                                   "accepted": 1, "rejected": 1, "enqueued": 1.0, "first_token": 2.0}
            local[st["id"]] = {k: v["tokens"] for k, v in jobs.items()}
            line = [[0.5, [0] * len(jobs)], [1.5, [0 if n == "r" else 5 for n in jobs]], [2.5, [1 if n == "r" else 9 for n in jobs]]]
            rec.update(jobs = jobs, order = list(jobs), timeline = line,
                       stats = {"stagger": {"steps": 4}, "side": {"chunks": 2, "beside": 2}}, state = {"thread": "on"})
        elif st["op"] == "generator":
            rec["switches_state"] = {"stagger_decode": False, "hist_prefill_weight": 1.0} if spec["library"] == "fork" else {}
        elif st["op"] == "disk_scan":
            rec["scan"] = {"pages": {"aa": {"crc": 7, "crc_disk": 7, "ok": True}}, "states": {}, "bad": []}
        EC.append_jsonl(path, rec)
    EC.append_jsonl(path, {"op": "end", "id": "end", "ok": True})
""")


def orchestrate(tmp_path, monkeypatch, mode = "", scenarios = "2", extra = ()):
    script = tmp_path / "fake_child.py"
    script.write_text(FAKE_CHILD.replace("@EXP@", repr(EXP)))
    out = tmp_path / "run"
    args = EC.parse_cli(["--scenarios", scenarios, "--out", str(out), "--s2-jobs", "2", "--step-timeout", "1",
                         "--load-timeout", "5", "--placement", "", "--layer-budget", "uniform"] + list(extra))
    args.placement_copy = None
    args.grace = 1.0
    args.poll = 0.1
    monkeypatch.setenv("FAKE_MODE", mode)
    monkeypatch.setenv("EXL3_STAGGER_DECODE", "1")
    plan = EC.build_plan(args)
    orch = EC.Orchestrator(plan, args, str(out), child_cmd = [sys.executable, str(script)])
    orch.run()
    return orch, out


def test_a_night_with_a_fake_child(tmp_path, monkeypatch):
    orch, out = orchestrate(tmp_path, monkeypatch)
    report = (out / "REPORT.md").read_text()
    assert summary_cells(report)[0][1] == EC.IDENT, report
    spec = json.loads((out / "procs" / "s2-fork" / "spec.json").read_text())
    assert spec["library"] == "fork" and spec["steps"][0]["op"] == "generator"
    env = json.loads((out / "procs" / "s2-fork" / "env.json").read_text())
    assert "EXL3_STAGGER_DECODE" not in env and env["EXL3_MOE_RECON_DET"] == "1"
    assert orch.res.procs["s2-fork"]["exit"] == 0 and "spent" in json.loads((out / "summary.json").read_text())


def test_tokens_of_another_process_travel_in_the_spec(tmp_path, monkeypatch):
    orch, out = orchestrate(tmp_path, monkeypatch, scenarios = "1")
    p0 = orch.res.records["s1-stock-a"]
    spec = json.loads((out / "procs" / "s1-fork" / "spec.json").read_text())
    forced = next(j["forced"] for st in spec["steps"] if st["id"] == "s1.a.ref1" for j in st["jobs"])
    assert forced == {"tokens": p0["s1.free.a"]["jobs"]["a"]["tokens"]}
    turn = spec["prompts"]["turn/s1"]["answer"]
    assert turn == {"tokens": p0["s1.mix.free"]["jobs"]["r"]["tokens"]}
    # In the process that made them they stay references
    spec0 = json.loads((out / "procs" / "s1-stock-a" / "spec.json").read_text())
    assert next(j["forced"] for st in spec0["steps"] if st["id"] == "s1.a.ref1" for j in st["jobs"])["proc"] == "s1-stock-a"
    report = (out / "REPORT.md").read_text()
    assert summary_cells(report)[0][1] == EC.IDENT, report


def test_a_child_that_goes_silent_is_stopped_and_the_night_goes_on(tmp_path, monkeypatch):
    orch, out = orchestrate(tmp_path, monkeypatch, mode = "hang:s1-stock-a:s1.a.ref2", scenarios = "1")
    o = orch.res.procs["s1-stock-a"]
    assert o["killed"] and o["hang"] and "nessun segno di vita" in o["reason"]
    # The next processes ran, and found no tokens to force
    assert orch.res.procs["s1-fork"]["exit"] == 0
    report = (out / "REPORT.md").read_text()
    assert "**Attenzione**: il processo `s1-stock-a` si è fermato" in report
    assert summary_cells(report)[0][1] in (EC.ERR, EC.DIFF)


def test_a_child_that_crashes_leaves_the_report_of_the_others(tmp_path, monkeypatch):
    orch, out = orchestrate(tmp_path, monkeypatch, mode = "crash:s3-fork", scenarios = "2,3")
    report = (out / "REPORT.md").read_text()
    rows = summary_cells(report)
    assert rows[0][1] == EC.IDENT and rows[1][1] == EC.ERR
    assert "il processo `s3-fork` si è fermato (uscito con 5)" in report


# --------------------------------------------------------------------------------------------
#   The child's Runner over the real generator and the toy model
# --------------------------------------------------------------------------------------------

class FakeWatch:
    step_timeout = 30.0
    iterations = 0

    def guard(self, op, timeout = None):
        from contextlib import nullcontext
        return nullcontext()

    def beat(self, *a, **k):
        pass

    def dump(self, why):
        pass


def toy_env(steps = 5, slots = 4, num_draft_tokens = 1, window_jobs = False):
    g = make_generator(draft = True, min_jobs = 2, num_draft_tokens = num_draft_tokens)
    g.mtp_verify_stats = {"windows": 0, "serial": 0, "readbacks": 0, "fallbacks": {}}
    g.readback_stats = {"steps": 0, "readbacks": 0}
    g.model.host = Slots(slots)
    g.model.config.moe_cpu_hosts = {"cuda:0": g.model.host}
    g.side_min_tokens = CHUNK
    g.max_chunk_size = CHUNK
    records = {}

    class BenchJob(ToyJob):
        """A Job as the child makes it, over the toy job: forced tokens, results with the keys of
        the real ones."""

        def __init__(self, name, ids, max_new):
            super().__init__(g.model, name, ids.reshape(-1).tolist(), max_new, chunk = CHUNK)
            self.identifier = name
            self.forced = None
            self.hurry = threading.Event()
            if steps:
                self.on_chunk = lasting(steps)

        def constrain_output_now(self, tokens):
            self.forced = tokens.reshape(-1).tolist()

        def receive_logits(self, logits):
            self.mid_pass = False
            if self.forced is not None:
                return torch.tensor([[self.forced[len(self.out)]]]), None, None, None
            return torch.argmax(logits, dim = -1).view(1, 1), None, None, None

        def receive_sample(self, logits, next_token, next_k_tokens, next_k_probs, next_prob, results):
            eos, tok, rq = super().receive_sample(logits, next_token, next_k_tokens, next_k_probs, next_prob, results)
            results[-1]["token_ids"] = next_token.view(1, 1)
            return eos, tok, rq

    class WindowJob(BenchJob):
        """With Job's window interface (Generator.mtp_batched_verify): greedy rows, a forced job
        declines as the real one does"""

        def window_sampling_why(self):
            return "forced_tokens" if self.forced is not None else None

        def sample_window(self, logits, drafts):
            self.mid_pass = False
            return torch.argmax(logits, dim = -1).view(1, -1)

    if window_jobs:
        BenchJob = WindowJob

    prompts = {"q0": toy.PROMPTS["a"], "q1": toy.PROMPTS["b"], "q2": toy.PROMPTS["c"], "q3": toy.PROMPTS["d"]}

    def prompt_ids(key):
        if key in prompts:
            return torch.tensor([prompts[key]], dtype = torch.long)
        spec = EC.prompt_spec_of(key)
        if spec["kind"] == "question":
            return torch.tensor([long_prompt(spec["index"] + 1, 4 + spec["index"])], dtype = torch.long)
        return torch.tensor([long_prompt(len(spec["tag"]), spec["length"])], dtype = torch.long)

    def forced_tokens(r):
        rec = records[r["run"]]
        toks = rec["jobs"][r["job"]]["tokens"]
        return toks[:r["limit"]] if r.get("limit") else toks

    def apply_switches(sw):
        pipeline = bool(sw.get("pipeline"))
        g.prefill_thread = bool(sw.get("thread")) or pipeline
        g.prefill_pipeline = pipeline
        g.stagger_decode = bool(sw.get("stagger"))
        g.stagger_min_jobs = int(sw.get("stagger_min_jobs") or 2)

    def make_job(name, ids, max_new, **kw):
        # (max_rq_tokens: the toy job never requeues)
        return BenchJob(name, ids, max_new)

    x = types.SimpleNamespace(
        torch = torch, gen = g, clock = time.time, watch = FakeWatch(), make_job = make_job, prompt_ids = prompt_ids,
        forced_tokens = forced_tokens, digester = lambda: CH.Digester(torch, None, 2), apply_switches = apply_switches,
        forget = lambda: 0, stats = lambda: {"side": dict(g.side_stats), "stagger": dict(g.stagger_stats),
                                             "verify": {k: v for k, v in g.mtp_verify_stats.items() if k != "fallbacks"}},
        state = lambda: {"thread": g.prefill_thread_state, "pipeline": g.prefill_pipeline_state},
        placement = lambda: None, reset_memory = lambda: None, sample_memory = lambda: None, memory = lambda: {},
        disk_metrics = lambda: None, run_timeout = 120.0,
    )
    x.records = records
    return x


@pytest.fixture
def env():
    x = toy_env()
    yield x
    for pool in (x.gen._side_pool, x.gen._pipe_pool):
        if pool is not None:
            pool.shutdown(wait = True)


def do(x, spec):
    rec = CH.Runner(x).run(spec)
    assert rec["ok"], rec.get("trace") or rec.get("error")
    x.records[spec["id"]] = rec
    return rec


def res_of(x):
    return Res({("p", rid): rec for rid, rec in x.records.items()})


DOC = EC.doc("r", 8 * CHUNK + 1, 1)


def test_forced_runs_of_the_same_computation_compare_identical(env):
    free = do(env, EC.run_step("free", 2, [EC.jobspec("a", "q0", 120)]))
    assert len(free["jobs"]["a"]["tokens"]) == 120
    v = free["jobs"]["a"]
    # A digest, the emitted token and the most likely ones at every position
    assert len(v["digest"]) == len(v["tokens"]) == len(v["logp"]) == len(v["topi"]) == 120
    assert v["tok"] == v["tokens"] and all(d is not None for d in v["digest"])
    for k in (1, 2):
        do(env, EC.run_step(f"ref{k}", 2, [EC.jobspec("a", "q0", forced = EC.ref("p", "free", "a"))]))
    c = EC.compare_jobs(env.records["free"]["jobs"]["a"], env.records["ref1"]["jobs"]["a"])
    assert c["identical"] and c["n"] == 120 and c["tokens_ok"]
    r = EC.evaluate_probs(EC.cmp(2, "x", "l", EC.ref("p", "ref1", "a"), EC.ref("p", "free", "a"),
                                 floor = (EC.ref("p", "ref1", "a"), EC.ref("p", "ref2", "a"))), res_of(env))
    assert r["verdict"] == EC.IDENT


def test_the_digest_tells_a_batch_from_a_job_alone_in_the_toy(env):
    """The toy's job alone and in a batch compute the same tokens (it is row-independent), so
    the comparison sees the same digests; the check is not blind either, see the next test."""
    do(env, EC.run_step("fa", 2, [EC.jobspec("a", "q0", 80)]))
    do(env, EC.run_step("fb", 2, [EC.jobspec("b", "q1", 80)]))
    jobs = [EC.jobspec("a", "q0", forced = EC.ref("p", "fa", "a")), EC.jobspec("b", "q1", forced = EC.ref("p", "fb", "b"))]
    do(env, EC.run_step("batch", 2, jobs))
    st = do(env, EC.run_step("stagger", 2, jobs, switches = {"stagger": True, "stagger_min_jobs": 2}))
    assert st["stats"]["stagger"]["steps"] > 0
    for name in "ab":
        assert EC.compare_jobs(env.records["batch"]["jobs"][name], st["jobs"][name])["identical"]


def test_a_reader_beside_a_decoder_on_the_prefill_thread(env):
    after = {"after": "a", "tokens": 6}
    do(env, EC.run_step("free.a", 3, [EC.jobspec("a", "q0", 400)]))
    for k in (1, 2):
        do(env, EC.run_step(f"dec.ref{k}", 3, [EC.jobspec("a", "q0", forced = EC.ref("p", "free.a", "a"))]))
    do(env, EC.run_step("mix.free", 3, [EC.jobspec("a", "q0", 2000), EC.jobspec("r", DOC, 12, start = after)], until = {"job": "r"}))
    for k in (1, 2):
        do(env, EC.run_step(f"mix.ref{k}", 3, [EC.jobspec("a", "q0", 2000),
                                               EC.jobspec("r", DOC, forced = EC.ref("p", "mix.free", "r"), start = after)],
                            until = {"job": "r"}))
    thr = {"thread": True}
    t_reader = do(env, EC.run_step("thr.reader", 3, [EC.jobspec("a", "q0", 2000),
                                                     EC.jobspec("r", DOC, forced = EC.ref("p", "mix.free", "r"), start = after)],
                                   until = {"job": "r"}, switches = thr))
    t_dec = do(env, EC.run_step("thr.dec", 3, [EC.jobspec("a", "q0", forced = EC.ref("p", "free.a", "a")),
                                               EC.jobspec("r", DOC, 12, start = after)], until = {"job": "r"}, switches = thr))
    # The reader went in eight chunks beside the decode; the decoder was cancelled at its end
    assert t_reader["state"]["thread"] == "on" and t_reader["stats"]["side"]["beside"] == 8
    assert len(t_reader["jobs"]["r"]["prefill"]) == 8 and t_reader["jobs"]["a"]["cancelled"]
    res = res_of(env)
    reader = EC.evaluate_probs(EC.cmp(3, "r", "l", EC.ref("p", "thr.reader", "r"), EC.ref("p", "mix.ref1", "r"),
                                      floor = (EC.ref("p", "mix.ref1", "r"), EC.ref("p", "mix.ref2", "r")),
                                      requires = {"thread": "on"}), res)
    assert reader["verdict"] == EC.IDENT and reader["test"]["n"] == 12
    dec = EC.evaluate_probs(EC.cmp(3, "a", "l", EC.ref("p", "thr.dec", "a"), EC.ref("p", "dec.ref1", "a"),
                                   floor = (EC.ref("p", "dec.ref1", "a"), EC.ref("p", "dec.ref2", "a")),
                                   upto = {"window": "r"}, drafts = False, requires = {"thread": "on", "beside": True}), res)
    assert dec["verdict"] == EC.IDENT, dec
    started, inside = dec["window"]
    assert started >= 6 and inside - started >= 8 * 5
    assert t_dec["jobs"]["r"]["first_token"] > t_dec["jobs"]["r"]["enqueued"]


def test_a_wrong_number_on_the_prefill_thread_is_caught(env, monkeypatch):
    real = toy.layer_step

    def layer_step(state, layer, tokens, pos0, page0):
        out = real(state, layer, tokens, pos0, page0)
        if threading.current_thread().name.startswith("exl3-prefill") and layer == 3 and pos0 == 2 * CHUNK:
            state.acc[layer] += 1
        return out

    monkeypatch.setattr(toy, "layer_step", layer_step)
    do(env, EC.run_step("free", 3, [EC.jobspec("r", DOC, 16)]))
    for k in (1, 2):
        do(env, EC.run_step(f"ref{k}", 3, [EC.jobspec("r", DOC, forced = EC.ref("p", "free", "r"))]))
    do(env, EC.run_step("thr", 3, [EC.jobspec("r", DOC, forced = EC.ref("p", "free", "r"))], switches = {"thread": True}))
    r = EC.evaluate_probs(EC.cmp(3, "r", "l", EC.ref("p", "thr", "r"), EC.ref("p", "ref1", "r"),
                                 floor = (EC.ref("p", "ref1", "r"), EC.ref("p", "ref2", "r"))), res_of(env))
    assert r["verdict"] == EC.DIFF and "identico" in r["why"]
    assert r["test"]["first_diff"] == 0 or r["test"]["first_diff"] is not None


def test_a_cancel_in_the_middle_of_an_ingestion(env):
    rec = do(env, EC.run_step("cancel", 4, [EC.jobspec("a", "q0", 3000), EC.jobspec("r", DOC, 12, start = {"after": "a", "tokens": 6})],
                              switches = {"thread": True}, cancel = [{"job": "r", "at_progress": 0.5}],
                              after_cancel = {"job": "a", "tokens": 10}))
    assert len(rec["cancels"]) == 1 and rec["cancels"][0]["progress"] >= 0.5
    assert rec["jobs"]["r"]["cancelled"] and rec["jobs"]["r"]["first_token"] is None
    assert rec["left_jobs"] == 0 and not env.gen.num_remaining_jobs()
    row = EC.evaluate_cancel({"scenario": 4, "id": "c", "run": EC.ref("p", "cancel", "r")}, res_of(env))
    assert row["verdict"] == EC.PASS
    # The generator goes on: the same decoder again gives the same tokens as before
    do(env, EC.run_step("after", 4, [EC.jobspec("a", "q0", 50)]))
    do(env, EC.run_step("again", 4, [EC.jobspec("a", "q0", 50)]))
    assert EC.compare_jobs(env.records["after"]["jobs"]["a"], env.records["again"]["jobs"]["a"])["identical"]


def test_a_start_that_can_never_come_is_an_error_not_a_hang(env):
    rec = CH.Runner(env).run(EC.run_step("never", 3, [EC.jobspec("a", "q0", 4),
                                                       EC.jobspec("r", DOC, 4, start = {"after": "a", "tokens": 50})]))
    assert not rec["ok"] and "before `r` could start" in rec["error"]
    assert not env.gen.num_remaining_jobs()


def test_the_stress_schedule_is_the_seed_s():
    a, b = CH.stress_schedule(7, 2.0), CH.stress_schedule(7, 2.0)
    assert a == b and a and all(0 < t < 120 for t, *_ in a)
    assert CH.stress_schedule(8, 2.0) != a
    kinds = {k for _, k, _, _ in CH.stress_schedule(1, 60.0)}
    assert kinds == {"short", "doc", "long", "turn", "cancel"}


@pytest.mark.parametrize("names", ["a", "abc"])
def test_windows_sampled_at_once_leave_the_records_of_the_serial_path(monkeypatch, names):
    """With mtp_batched_verify a job's positions skip receive_logits (Job.sample_window); the
    Digester records them from the window's rows, so a free run with the verify on has the same
    records, position for position, as one with it off: digests, emitted tokens, the most likely
    ones and the draft counters"""
    monkeypatch.setattr(toy.ToyLayerState, "ROWS", 4 * 4)
    recs = {}
    for verify in (False, True):
        x = toy_env(num_draft_tokens = 3, window_jobs = True)
        x.gen.mtp_batched_verify = verify
        jobs = [EC.jobspec(n, f"q{i}", 90) for i, n in enumerate(names)]
        recs[verify] = do(x, EC.run_step("free", 9, jobs, switches = {"stagger": len(names) > 2}))
        assert not recs[verify]["warnings"], recs[verify]["warnings"]
        if verify:
            assert recs[verify]["stats"]["verify"]["windows"] > 20 and recs[verify]["stats"]["verify"]["serial"] == 0
    for n in names:
        off, on = recs[False]["jobs"][n], recs[True]["jobs"][n]
        assert len(on["digest"]) == len(on["tokens"]) == 90 and all(d is not None for d in on["digest"])
        for key in ("tokens", "digest", "tok", "logp", "topi", "topl", "acc", "rej"):
            assert on[key] == off[key], (n, key)
        assert on["accepted"] == off["accepted"] > 5
    # The same comparison the report makes, on free runs
    res = Res({("p", "on"): recs[True], ("p", "off"): recs[False]})
    c = EC.cmp(9, "x", "l", EC.ref("p", "on", names[0]), EC.ref("p", "off", names[0]), free = True,
               floor = (EC.ref("p", "off", names[0]), EC.ref("p", "on", names[0])))
    assert EC.evaluate_probs(c, res)["verdict"] == EC.IDENT


def test_the_digester_takes_each_position_of_a_window_from_its_row():
    """A window of four rows of which the generator takes two (the second draft rejected): two
    entries, each its own row's, then a serial position clears the window"""
    dig = CH.Digester(torch, None, 2)
    taken = []

    class J:
        accepted_draft_tokens = 0
        rejected_draft_tokens = 0

        def receive_logits(self, logits):
            return torch.argmax(logits, dim = -1).view(1, 1), None, None, None

        def sample_window(self, logits, drafts):
            return torch.argmax(logits, dim = -1).view(1, -1)

        def receive_sample(self, logits, next_token, *rest):
            taken.append(int(next_token.view(-1)[0]))
            return False, next_token, False

    j = J()
    dig.wrap(j, "a")
    logits = torch.tensor([[[0.0, 5.0, 1.0], [3.0, 0.0, 1.0], [0.0, 1.0, 7.0], [9.0, 0.0, 0.0]]])
    rows = j.sample_window(logits, [1, 0, 2])
    j.receive_sample(logits[:, 0:1], torch.tensor([[int(rows[0, 0])]]))
    j.accepted_draft_tokens += 1
    j.receive_sample(logits[:, 1:2], torch.tensor([[int(rows[0, 1])]]))
    j.rejected_draft_tokens += 2
    # The next step takes the serial path: its position is receive_logits', not the window's third row
    tok = j.receive_logits(torch.tensor([[[1.0, 0.0, 4.0]]]))[0]
    j.receive_sample(None, tok)
    out = {"a": {"digest": [], "tok": [], "logp": [], "topi": [], "topl": []}}
    dig.resolve(out)
    v = out["a"]
    assert taken == [1, 0, 2] and v["tok"] == [1, 0, 2] and v["acc"] == [0, 1, 1] and v["rej"] == [0, 0, 2]
    # Each entry is its own row's: the digest of a row alone, as the serial path would record it
    alone = CH.Digester(torch, None, 2)
    for r, t in ((0, 1), (1, 0)):
        o = {"x": {"digest": [], "tok": [], "logp": [], "topi": [], "topl": []}}
        alone.pending = {"x": [(alone._post(alone._pre(logits[:, r:r + 1]), torch.tensor([[t]])), (0, 0))]}
        alone.resolve(o)
        assert o["x"]["digest"][0] == v["digest"][r] and o["x"]["logp"][0] == pytest.approx(v["logp"][r])
    assert not dig.errors


def test_the_digester_is_the_same_in_every_process():
    d1, d2 = CH.Digester(torch, 6, 3), CH.Digester(torch, 6, 3)
    logits = torch.tensor([[[0.5, -1.0, 2.0, 0.25, 3.0, -0.5, 9.0, 9.0]]])
    out = {"x": {"digest": [], "tok": [], "logp": [], "topi": [], "topl": []}}
    out2 = json.loads(json.dumps(out))
    for d, o in ((d1, out), (d2, out2)):
        pre = d._pre(logits)
        d.pending = {"x": [(d._post(pre, torch.tensor([[2]])), (3, 1))]}
        d.resolve(o)
    assert out == out2
    v = out["x"]
    # Over the real vocabulary only (the last two entries are padding)
    assert v["topi"][0] == [4, 2, 0] and v["tok"] == [2]
    lse = math.log(sum(math.exp(z) for z in [0.5, -1.0, 2.0, 0.25, 3.0, -0.5]))
    assert v["logp"][0] == pytest.approx(2.0 - lse, rel = 1e-6)
    # One bit flipped in the padding changes the digest, nothing else
    other = logits.clone()
    other.view(-1).view(torch.int32)[7] ^= 1
    o3 = {"x": {"digest": [], "tok": [], "logp": [], "topi": [], "topl": []}}
    d1.pending = {"x": [(d1._post(d1._pre(other), torch.tensor([[2]])), (3, 1))]}
    d1.resolve(o3)
    assert o3["x"]["digest"] != v["digest"] and o3["x"]["logp"] == v["logp"]
    # The job's draft counters as they stood at the position
    assert v["acc"] == [3] and v["rej"] == [1]


# --------------------------------------------------------------------------------------------
#   The library points the child reads (checked on the sources of this branch)
# --------------------------------------------------------------------------------------------

def _src(*parts):
    with open(os.path.join(toy.REPO, "exllamav3", *parts)) as f:
        return f.read()


def test_the_library_still_has_what_the_child_reads():
    job = _src("generator", "job.py")
    gen = _src("generator", "generator.py")
    for name in ("return_probs", "identifier", "max_new_tokens", "def constrain_output_now", "self.accepted_draft_tokens",
                 "self.rejected_draft_tokens", "self.resumed_from_tail", '"cached_tokens"', '"curr_progress"',
                 '"max_progress"', "class NonFiniteLogits"):
        assert name in job, name
    # The generator samples through the job object, one position per call
    assert "job.receive_logits(token_logits)" in gen and "job.receive_logits(\n" in gen
    for name in ("def flush_cache_to_disk", "def has_idle_work", "def idle_work", "def close", "self.side_stats",
                 "self.stagger_stats", "self.tail_stats", "def prefill_thread_state", "def prefill_pipeline_state",
                 "EXL3_PREFILL_PIPELINE_HEAD", "self.stagger_min_jobs", "self.tail_checkpoint_interval"):
        assert name in gen, name
    # The exact mode's two switches, and the static weights of the streamed prefill
    host = _src("model", "moe_cpu_host.py")
    assert 'os.environ.get("EXL3_MOE_STREAM_FUSED_T", 256)' in host
    assert 'EXL3_MOE_RECON_DET' in _src("modules", "moe_batch_recon.py")
    # The digest reads the token the job returns first
    assert "return next_token, next_k_tokens, next_k_probs, next_prob" in job
    # A window sampled at once goes through the job's sample_window (its logits before the sampler
    # sees them), then receive_sample once per position the generator takes from it, in order: the
    # Digester takes each such position's row from there
    assert "sampled = job.sample_window(batch_logits[a:b, :, :], draft_tokens[row, :width].tolist())" in gen
    assert "next_token = torch.tensor([[window[0][i]]], dtype = torch.long)" in gen
    assert gen.count("eos, sampled_token, rq = job.receive_sample(") == 2
    # The MTP decode's switches and counters, and the test hook's schedule (the cost window)
    for name in ("mtp_batched_verify: bool = False", "mtp_batched_update: bool = False", 'draft_window: str = "fixed"',
                 "self.mtp_verify_stats = {", "self.mtp_update_stats = {", "self.readback_stats = {",
                 "self.draft_window_stats = policy.stats", "schedule = self.draft_window_schedule,"):
        assert name in gen, name
    assert '"scheduled": 0' in _src("generator", "draft_window.py")
    # The exact mode's other pins: the GPU fused kernel's deterministic accumulation (default on),
    # the batched reconstruct's fallback default (atomic unless set), the streamed fused tier
    # without the slot scratch (why EXL3_MOE_STREAM_FUSED_T=0), graphs off only on request
    assert 'FUSED_DET = os.environ.get("EXL3_MOE_FUSED_DET", "1") != "0"' in _src("modules", "block_sparse_mlp.py")
    assert 'os.environ.get("EXL3_MOE_RECON_DET", os.environ.get("EXL3_MOE_FUSED_DET", "0"))' in _src("modules", "moe_batch_recon.py")
    assert "float(spec[\"act_limit\"] or 0.0), n_act, None, None, lo, hi, mt" in host
    assert 'getenv("EXL3_GRAPHS")' in _src("exllamav3_ext", "graph.cu")
    # The sweep is reached through the module's global, from the drain and the overdue fallback:
    # a function-local import in the generator (read at every call), no import of it elsewhere
    cpu = _src("modules", "block_sparse_mlp_cpu.py")
    assert "\n        from ..modules.block_sparse_mlp_cpu import run_pending_swap_sweeps\n" in gen
    assert "        run_pending_swap_sweeps(ip)" in cpu
    for root, _, files in os.walk(os.path.join(toy.REPO, "exllamav3")):
        for fn in files:
            if fn.endswith(".py") and "conversion" not in root:
                with open(os.path.join(root, fn)) as f:
                    for line in f:
                        if "import" in line and "run_pending_swap_sweeps" in line:
                            assert line.startswith("        "), (fn, line)
    # Promote-on-miss is read at load, with the value the orchestrator sets
    assert 'int(os.environ.get("EXL3_MOE_CPU_PROMOTE", 4))' in cpu
    assert "self.actual_vocab_size" in _src("tokenizer", "tokenizer.py")
    for name in ("_split_hist", "_split_map", "def _split_sweep_layer_reset", "moe_cpu_swap_pending", "moe_cpu_swap_modules"):
        assert name in cpu, name
    # The forget trick: prune_stranded asks the page table attached to the cache
    rc = _src("cache", "recurrent.py")
    assert "stranded = [k for k in self if not self.pagetable.is_resumable(k)]" in rc
    # The disk files' layout the scan reads
    disk = _src("generator", "disk_cache.py")
    assert 'PAGE_MAGIC = b"EXL3KVP1"' in disk and 'STATE_MAGIC = b"EXL3RCS1"' in disk
    assert 'meta["crc32"] = crc' in disk and "zlib.crc32(memoryview(slab.numpy()), zlib.crc32(tokens_bytes))" in disk
    assert 'phash[:1].hex(), phash.hex() + ".kv"' in disk


def test_a_switch_left_on_does_not_pass_for_off():
    assert EC.switch_is_off("stagger_decode", False) and not EC.switch_is_off("stagger_decode", True)
    assert EC.switch_is_off("hist_prefill_weight", 1.0) and not EC.switch_is_off("hist_prefill_weight", 0.004)
    assert EC.switch_is_off("tail_checkpoint_interval", 0) and not EC.switch_is_off("tail_checkpoint_interval", 256)
    assert not EC.switch_is_off("tail_checkpoint_interval", True) and EC.switch_is_off("disk_uncache", None)
    for name in ("mtp_batched_verify", "mtp_batched_update"):
        assert EC.switch_is_off(name, False) and not EC.switch_is_off(name, True)
    # The draft window: off is upstream's fixed window, and no test schedule
    assert EC.switch_is_off("draft_window", "fixed") and not EC.switch_is_off("draft_window", "cost")
    assert EC.switch_is_off("draft_window_schedule", None) and not EC.switch_is_off("draft_window_schedule", [3])
    assert not EC.switch_is_off("draft_window_schedule", "3") and not EC.switch_is_off("an_unknown_switch", 0)
    for name in ("EXL3_DRAFT_WINDOW", "EXL3_DRAFT_WINDOW_SET", "EXL3_DRAFT_WINDOW_SCHEDULE"):
        assert name in EC.FORK_ENV_SWITCHES
    # Content-addressed multimodal pages (master's, 2026-10-24): the generator's switch and the digests it
    # turns on for the process
    assert "EXL3_MM_CONTENT_HASH" in EC.FORK_ENV_SWITCHES
    for name in ("mm_content_hash", "mm_content_digests"):
        assert EC.switch_is_off(name, False) and not EC.switch_is_off(name, True)
    # The decode step's host fast path (read at import)
    assert "EXL3_HOST_FASTPATH" in EC.FORK_ENV_SWITCHES
    assert EC.switch_is_off("host_fastpath", False) and not EC.switch_is_off("host_fastpath", True)


def test_scenario_1_with_every_switch_of_this_branch_off_says_nothing_about_switches():
    plan = EC.build_plan(args_for("--scenarios", "1"))
    res = fake_results(plan)
    off = {"stagger_decode": False, "prefill_thread": False, "prefill_pipeline": False, "tail_checkpoint_interval": 0,
           "mtp_batched_verify": False, "mtp_batched_update": False, "draft_window": "fixed",
           "draft_window_schedule": None, "hist_prefill_weight": 1.0, "qsa_raw_tail": 0, "disk_uncache": False,
           "mm_content_hash": False, "mm_content_digests": False}
    for rid, rec in res.records["s1-fork"].items():
        if rec.get("op") == "generator":
            rec["switches_state"] = dict(off)
    assert not EC.capture_checks(plan, res).get(1)
    res.records["s1-fork"]["g0"]["switches_state"]["draft_window"] = "cost"
    text = "\n".join(EC.capture_checks(plan, res)[1])
    assert "non tutti gli interruttori sono spenti (`g0`): draft_window" in text
    res.records["s1-fork"]["g0"]["switches_state"]["draft_window"] = "fixed"
    res.records["s1-fork"]["g0"]["switches_state"]["mm_content_digests"] = True
    text = "\n".join(EC.capture_checks(plan, res)[1])
    assert "non tutti gli interruttori sono spenti (`g0`): mm_content_digests" in text


# --------------------------------------------------------------------------------------------
#   The child process itself, over a fake library and a fake server
# --------------------------------------------------------------------------------------------

FAKE_LIB = {
    "exllamav3/__init__.py": "from .generator import Generator, Job\n",
    "exllamav3/version.py": f"__version__ = '{EC.BASE_RELEASE}'\n",
    "exllamav3/generator/sampler.py": "class GreedySampler:\n    pass\n",
    "exllamav3/modules/__init__.py": "",
    "exllamav3/modules/block_sparse_mlp_cpu.py": textwrap.dedent('''
        import torch
        SWEEPS = []
        def run_pending_swap_sweeps(ip):
            if not getattr(ip, "moe_cpu_swap_pending", False):
                return
            ip.moe_cpu_swap_pending = False
            SWEEPS.append(1)
            for m in ip.moe_cpu_swap_modules:
                hot = int(torch.argmax(m._split_hist))
                cold = 0
                mp = m._split_map.clone()
                mp[hot], mp[cold] = m._split_map[cold], m._split_map[hot]
                m._split_map.copy_(mp)
                m._split_hist.mul_(0.5)
        @@FORK_WEIGHT@@
    '''),
    "exllamav3/generator/__init__.py": textwrap.dedent('''
        import os, json, zlib, torch
        from types import SimpleNamespace
        FORK = @@FORK@@

        class Job:
            def __init__(self, input_ids, max_new_tokens = None, sampler = None, identifier = None, **kw):
                self.ids = input_ids.reshape(-1).tolist()
                self.max_new = max_new_tokens
                self.identifier = identifier
                self.new_tokens = 0
                self.forced = None
                self.accepted_draft_tokens = 0
                self.rejected_draft_tokens = 0
                self.resumed_from_tail = False
                self.cached_pages = 0
                self.cached_tokens = 0
                self.done_prefill = 0
                self.generator = None
            def constrain_output_now(self, t):
                assert self.generator is not None
                self.forced = t.reshape(-1).tolist()
            def receive_logits(self, logits):
                if self.forced is not None:
                    return torch.tensor([[self.forced[self.new_tokens]]]), None, None, None
                return torch.argmax(logits, dim = -1).view(1, 1), None, None, None

        class Generator:
            def __init__(self, model, cache, tokenizer, max_batch_size = 4, max_chunk_size = 2048, cpu_cache_size = 0,
                         draft_model = None, draft_cache = None, num_draft_tokens = None, recurrent_cache_size = None,
                         recurrent_checkpoint_interval = None, recurrent_checkpoint_interval_pp = 32768,
                         disk_cache_dir = None, disk_cache_size = None, @@FORK_ARGS@@ **kwargs):
                self.model, self.max_chunk_size = model, max_chunk_size
                self.active, self.pending = [], []
                self.pagetable = SimpleNamespace(metrics = {"alloc": 0}, is_resumable = lambda k: True)
                self.recurrent_cache = Rc(self.pagetable)
                self.disk_page_cache = Disk(disk_cache_dir) if disk_cache_dir else None
                self.kwargs = dict(kwargs)
                if FORK:
                    self.stagger_decode, self.prefill_thread, self.prefill_pipeline = stagger_decode, prefill_thread, prefill_pipeline
                    self.tail_checkpoint_interval = 256 if recurrent_tail_checkpoint else 0
                    self.stagger_min_jobs = 3
                    self.side_stats = {"chunks": 0, "beside": 0, "piped": 0}
                    self.stagger_stats = {"steps": 0}
                    self.tail_stats = {"stashes": 0}
                    self.mtp_batched_verify, self.mtp_batched_update = mtp_batched_verify, mtp_batched_update
                    self.draft_window = draft_window
                    self.num_draft_tokens = num_draft_tokens or 0
                    self.mtp_verify_stats = {"windows": 0, "serial": 0, "fallbacks": {}}
                    self.mtp_update_stats = {"folds": 0, "fold_drafts": 0}
                    self.draft_window_stats = {"steps": 0, "scheduled": 0, "chosen": {}} if draft_window == "cost" else None
            if FORK:
                draft_window_schedule = None
                prefill_thread_state = property(lambda self: "on" if self.prefill_thread else "off")
                prefill_pipeline_state = property(lambda self: "on" if self.prefill_pipeline else "off")
            def enqueue(self, job):
                job.generator = self
                self.pending.append(job)
            def num_remaining_jobs(self):
                return len(self.active) + len(self.pending)
            def iterate(self):
                out = []
                while self.pending:
                    j = self.pending.pop(0)
                    self.active.append(j)
                    out.append({"job": j, "stage": "started"})
                for j in list(self.active):
                    if j.done_prefill < len(j.ids) - 1:
                        j.done_prefill = min(len(j.ids) - 1, j.done_prefill + self.max_chunk_size)
                        if FORK and self.prefill_thread:
                            self.side_stats["chunks"] += 1
                        out.append({"job": j, "stage": "prefill", "curr_progress": j.done_prefill, "max_progress": len(j.ids) - 1})
                        continue
                    seed = zlib.crc32(repr(j.ids).encode())
                    logits = torch.tensor([[[float((seed * 31 + t * 7 + len(j.ids)) % 97) for t in range(64)]]])
                    tok, *_ = j.receive_logits(logits)
                    j.ids.append(int(tok))
                    j.new_tokens += 1
                    eos = j.new_tokens >= j.max_new
                    r = {"job": j, "stage": "streaming", "token_ids": tok.view(1, 1), "eos": eos}
                    if eos:
                        r.update(eos_reason = "max_new_tokens", cached_tokens = 0)
                        self.active.remove(j)
                    out.append(r)
                return out
            def cancel(self, job):
                for q in (self.active, self.pending):
                    if job in q:
                        q.remove(job)
            def clear_queue(self):
                self.active.clear()
                self.pending.clear()
            def flush_cache_to_disk(self, wait = False):
                d = os.path.join(self.disk_page_cache.root, "fp", "pages", "ab")
                os.makedirs(d, exist_ok = True)
                tokens, image = b"tt", b"image" * 10
                meta = json.dumps({"crc32": zlib.crc32(image, zlib.crc32(tokens))}).encode()
                with open(os.path.join(d, "ab01.kv"), "wb") as f:
                    f.write(b"EXL3KVP1" + len(meta).to_bytes(4, "little") + len(tokens).to_bytes(4, "little") + meta + tokens + image)
                return {"pages": 1}
            def close(self):
                self.closed = True

        class Rc(dict):
            def __init__(self, pt):
                super().__init__()
                self.pagetable = pt
                self.metrics = {"stash_pruned": 0}
                self["k"] = 1
            def prune_stranded(self):
                gone = [k for k in self if not self.pagetable.is_resumable(k)]
                for k in gone:
                    del self[k]
                return len(gone)

        class Disk:
            def __init__(self, root):
                self.root = root
                self.metrics = {"page_writes": 0}
                self.uncache = os.environ.get("EXL3_DISK_CACHE_UNCACHE", "0") not in ("", "0")
    '''),
}

FAKE_SERVER = {
    "openai/__init__.py": "",
    "openai/config.py": textwrap.dedent('''
        from types import SimpleNamespace
        def parse_args(argv):
            assert "--model-path" in argv
            return SimpleNamespace(max_chunk_size = 16, cpu_cache_size = 0.5, draft_num_tokens = 1, recurrent_cache_size = 1.0,
                                   recurrent_checkpoint_interval = 256, recurrent_checkpoint_interval_pp = None, argv = argv)
    '''),
    "openai/model_container.py": textwrap.dedent('''
        import torch
        from types import SimpleNamespace
        from exllamav3.modules import block_sparse_mlp_cpu as bsm
        class Tok:
            actual_vocab_size = 60
            def encode(self, text, encode_special_tokens = False):
                return torch.tensor([[ord(c) % 60 for c in text]])
        class Mod:
            def __init__(self):
                self._split_map = torch.arange(8)
                self._split_hist = torch.zeros(8)
                self.resets = 0
            def _split_sweep_layer_reset(self):
                self.resets += 1
        class Tier:
            attached = []
            def attach_generator(self, gen, cache_log = None):
                # As the server's: the drain wrapped, so that a deferred sweep runs there
                Tier.attached.append(gen)
                gen._exl3srv_moe_drain = True
        class ModelContainer:
            def __init__(self, cfg):
                self.cfg = cfg
            def load_model_sync(self):
                ip = SimpleNamespace(moe_cpu_swap_modules = [Mod(), Mod()])
                ip.moe_cpu_swap_modules[0]._split_hist[5] = 3.0
                self.model = SimpleNamespace(config = SimpleNamespace(infer_params = ip), unload = lambda: None,
                                             default_chat_prompt = lambda p: "<u>" + p + "</u>")
                self.tokenizer = Tok()
                self.cache = object()
                self.draft_model = None
                self.draft_cache = None
                self.max_batch_size = 4
                self.max_rq_tokens = 4096
                self.moe_tier = Tier()
                # The server's tier wraps the library's sweep
                orig = bsm.run_pending_swap_sweeps
                def wrapped(ip):
                    return orig(ip)
                wrapped._exl3srv_orig = orig
                bsm.run_pending_swap_sweeps = wrapped
    '''),
}


def write_tree(root, files, fork):
    for rel, text in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok = True)
        text = text.replace("@@FORK@@", "True" if fork else "False")
        text = text.replace("@@FORK_ARGS@@", "stagger_decode = False, prefill_thread = False, prefill_pipeline = False, "
                                             "recurrent_tail_checkpoint = False, mtp_batched_verify = False, "
                                             "mtp_batched_update = False, draft_window = 'fixed'," if fork else "")
        text = text.replace("@@FORK_WEIGHT@@", "def hist_prefill_weight():\n    return 1.0" if fork else "")
        with open(path, "w") as f:
            f.write(text)


def run_child(tmp_path, library, steps, prompts = None, lib_on_path = None, sweeps = "frozen"):
    fork_dir = tmp_path / "fork"
    stock_dir = tmp_path / "site"
    write_tree(str(fork_dir), FAKE_LIB, True)
    write_tree(str(stock_dir), FAKE_LIB, False)
    write_tree(str(tmp_path / "server"), FAKE_SERVER, False)
    pdir = tmp_path / f"proc-{library}"
    pdir.mkdir()
    spec = {"proc": library, "library": library, "run_dir": str(tmp_path), "proc_dir": str(pdir),
            "server_path": str(tmp_path / "server"), "fork_path": str(fork_dir), "server_argv": ["--model-path", "/m"],
            "steps": steps, "prompts": prompts or {}, "sweeps": sweeps, "step_timeout": 30, "load_timeout": 60,
            "budgets": {}, "top_k": 3, "disk_budget_gb": 1, "disk_root": str(tmp_path / "disk-cache")}
    (pdir / "spec.json").write_text(json.dumps(spec))
    env = dict(os.environ)
    lib = lib_on_path or (fork_dir if library == "fork" else stock_dir)
    env["PYTHONPATH"] = str(lib)
    out = subprocess.run([sys.executable, os.path.join(EXP, "exact_check_child.py"), "--spec", str(pdir / "spec.json")],
                         cwd = str(pdir), env = env, capture_output = True, text = True, timeout = 120)
    recs = {r["id"]: r for r in EC.read_jsonl(str(pdir / "records.jsonl"))}
    return out, recs


def test_the_child_over_a_fake_library(tmp_path):
    steps = [
        EC.gen_step("g0", 1),
        EC.run_step("free", 1, [EC.jobspec("a", "q0", 12)]),
        EC.run_step("forced", 1, [EC.jobspec("a", "q0", forced = EC.ref("fork", "free", "a"))], switches = {"thread": True}),
        EC.run_step("doc", 1, [EC.jobspec("r", EC.doc("t", 600, 1), 4)]),
        EC.gen_step("g1", 1, disk = "d1", switches = {"stagger": True}),
        {"op": "disk_scan", "id": "scan", "scenario": 1, "disk": "d1"},
        {"op": "reset_hist", "id": "reset", "scenario": 6},
        {"op": "sweep", "id": "sweep", "scenario": 6},
        {"op": "close", "id": "close", "scenario": 1},
    ]
    out, recs = run_child(tmp_path, "fork", steps)
    assert out.returncode == 0, out.stdout + out.stderr
    assert recs["start"]["ok"] and recs["start"]["fork_features"] and recs["load"]["ok"]
    assert recs["load"]["vocab"] == 60 and recs["load"]["split_modules"] == 2
    assert all(recs[k]["ok"] for k in ("g0", "free", "forced", "doc", "g1", "scan", "reset", "sweep", "close", "end")), recs
    # Forced on the free run's tokens, the same logits
    assert recs["forced"]["jobs"]["a"]["tokens"] == recs["free"]["jobs"]["a"]["tokens"]
    assert EC.compare_jobs(recs["free"]["jobs"]["a"], recs["forced"]["jobs"]["a"])["identical"]
    assert recs["forced"]["state"]["thread"] == "on" and recs["forced"]["stats"]["side"]["chunks"] >= 1
    assert recs["free"]["forgot"] == 1 and recs["forced"]["forgot"] == 0
    assert recs["forced"]["placement_before"] == recs["forced"]["placement_after"] is not None
    # A document of the exact length asked for
    assert recs["doc"]["jobs"]["r"]["prompt_tokens"] == 600
    # The fork's switches as the generator reads them, the disk tier written and read back
    assert recs["g0"]["switches_state"] == {"stagger_decode": False, "prefill_thread": False, "prefill_pipeline": False,
                                            "tail_checkpoint_interval": 0, "hist_prefill_weight": 1.0,
                                            "mtp_batched_verify": False, "mtp_batched_update": False, "draft_window": "fixed",
                                            "draft_window_schedule": None}
    assert recs["g1"]["switches_state"]["stagger_decode"] is True and "disk_cache_dir" in recs["g1"]["kwargs"]
    assert recs["scan"]["flushed"] == {"pages": 1} and recs["scan"]["scan"]["pages"]["ab01"]["ok"]
    # The sweep: the library's own (past the server's wrapper), never during the runs
    assert recs["sweep"]["total_moved"] == 0 or recs["sweep"]["total_moved"] >= 1
    assert recs["reset"]["modules"] == 2
    status = json.loads((tmp_path / "proc-fork" / "status.json").read_text())
    assert status["op"] in ("unload", "close", "end", "start") or status["run"] is None


@pytest.mark.parametrize("value, want", [(None, False), ("0", False), ("1", True)])
def test_the_child_says_whether_the_library_runs_the_host_fast_path(tmp_path, monkeypatch, value, want):
    # The switch is read at import by the library's modules: the child reports what the workspace
    # cache's module read, not the variable
    monkeypatch.setitem(FAKE_LIB, "exllamav3/util/__init__.py", "")
    monkeypatch.setitem(FAKE_LIB, "exllamav3/util/tensor.py",
                        "import os\nHOST_FASTPATH = os.environ.get('EXL3_HOST_FASTPATH', '0') not in ('', '0')\n")
    if value is None:
        monkeypatch.delenv("EXL3_HOST_FASTPATH", raising = False)
    else:
        monkeypatch.setenv("EXL3_HOST_FASTPATH", value)
    out, recs = run_child(tmp_path, "fork", [EC.gen_step("g0", 11)])
    assert out.returncode == 0, out.stdout + out.stderr
    assert recs["g0"]["switches_state"]["host_fastpath"] is want
    assert EC.switch_is_off("host_fastpath", recs["g0"]["switches_state"]["host_fastpath"]) is not want


def test_the_child_hands_the_mtp_decode_switches_to_the_generator(tmp_path):
    steps = [
        EC.gen_step("g0", 9, switches = {"verify": True, "update": True, "window": "cost", "schedule": [0, 3]}),
        EC.run_step("free", 9, [EC.jobspec("a", "q0", 6)]),
        EC.gen_step("g1", 9, switches = {"draft_tokens": 1}),
    ]
    out, recs = run_child(tmp_path, "fork", steps)
    assert out.returncode == 0, out.stdout + out.stderr
    st = recs["g0"]["switches_state"]
    assert st["mtp_batched_verify"] is True and st["mtp_batched_update"] is True
    assert st["draft_window"] == "cost" and st["draft_window_schedule"] == [0, 3]
    assert not any(EC.switch_is_off(k, st[k]) for k in ("mtp_batched_verify", "draft_window", "draft_window_schedule"))
    # The counters of the MTP decode, every window from 0 counted
    stats = recs["free"]["stats"]
    assert stats["verify"] == {"windows": 0, "serial": 0} and stats["update"] == {"folds": 0, "fold_drafts": 0}
    assert stats["window"] == {"steps": 0, "scheduled": 0, "w0": 0} and recs["g0"]["num_draft_tokens"] == 0
    # Another window than the process's, without an MTP head: refused
    assert not recs["g1"]["ok"] and "draft_tokens" in recs["g1"]["error"]


def test_the_child_refuses_a_library_that_is_not_the_one_asked_for(tmp_path):
    out, recs = run_child(tmp_path, "stock", [EC.gen_step("g0", 1)], lib_on_path = tmp_path / "fork")
    assert out.returncode == 3
    assert not recs["start"]["ok"] and "wheel installata" in recs["start"]["library_problem"]
    assert "load" not in recs


def test_a_stock_child_never_asks_for_a_fork_switch(tmp_path):
    steps = [EC.gen_step("g0", 1), EC.run_step("free", 1, [EC.jobspec("a", "q0", 6)]),
             EC.run_step("thread", 1, [EC.jobspec("a", "q0", 6)], switches = {"thread": True})]
    out, recs = run_child(tmp_path, "stock", steps)
    assert out.returncode == 0, out.stdout + out.stderr
    assert recs["g0"]["switches_state"] == {} and recs["free"]["ok"]
    assert not recs["thread"]["ok"] and "prefill_thread" in recs["thread"]["error"]


# --------------------------------------------------------------------------------------------
#   Where a comparison stops, controls, soft requirements
# --------------------------------------------------------------------------------------------

def test_a_staggered_row_stops_where_the_first_job_ended():
    """Past the first job's end the staggered decode re-forms its groups, the group alone does
    not: positions sampled from that iteration on are left out, so is a draft count there."""
    ref = view(list(range(10)), accepted = 4, rejected = 2)
    ref.update(acc = [0, 1, 1, 2, 2, 3, 3, 4, 4, 4], rej = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    t = runrec(a = dict(ref), b = dict(view(list(range(6))), ended = 3.0))
    t["jobs"]["a"] = dict(ref, digest = ref["digest"][:8] + [0, 0], acc = ref["acc"][:8] + [9, 9], ended = 5.0)
    t["order"] = ["a", "b"]
    t["timeline"] = [[1.0, [2, 2]], [2.0, [5, 4]], [3.0, [8, 6]], [4.0, [10, 6]]]
    res = Res({("p", "t"): t, ("p", "r1"): runrec(a = ref), ("p", "r2"): runrec(a = ref)})
    r = EC.evaluate_probs(probs_cmp(upto = {"before_first_end": True}), res)
    # b ended in the iteration that ended at 3.0: a's positions before it are 5
    assert r["verdict"] == EC.IDENT and r["test"]["n"] == 5 and r["test"]["drafts_same"]
    # Without the limit the re-formed groups' positions count
    assert EC.evaluate_probs(probs_cmp(), res)["verdict"] == EC.DIFF


def test_drafts_are_compared_position_by_position_or_not_at_all():
    a = dict(view([1, 2, 3, 4]), acc = [0, 1, 1, 2], rej = [0, 0, 1, 1])
    b = dict(a, acc = [0, 1, 2, 2])
    c = EC.compare_jobs(a, b)
    assert c["identical"] and c["drafts_same"] is False and c["drafts"] == (2, 1)
    assert EC.compare_jobs(a, b, upto = 2)["drafts_same"] is True
    # Without the per-position counters only the totals of whole jobs say something
    x, y = view([1, 2, 3, 4], accepted = 3), view([1, 2, 3, 4], accepted = 2)
    assert EC.compare_jobs(x, y)["drafts_same"] is False and EC.compare_jobs(x, y, upto = 3)["drafts_same"] is None
    res = Res({("p", "t"): runrec(a = y), ("p", "r1"): runrec(a = x), ("p", "r2"): runrec(a = x)})
    assert EC.evaluate_probs(probs_cmp(upto = 3), res)["verdict"] == EC.IDENT


def test_a_control_that_moves_explains_a_difference_and_one_that_does_not_cannot():
    same = view([1, 2, 3, 4])
    moved = view([1, 2, 3, 4], digest = [0] * 4, logp = [-0.5, -0.52, -0.52, -0.53])
    far = view([1, 2, 3, 4], digest = [0] * 4, logp = [-0.5, -0.9, -0.52, -0.53])

    def verdict(test, control):
        res = Res({("p", "t"): runrec(a = test), ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same),
                   ("p", "ctl"): runrec(a = control)})
        return EC.evaluate_probs(probs_cmp(yards = [(EC.ref("p", "r1", "a"), EC.ref("p", "ctl", "a"))]), res)

    r = verdict(moved, moved)
    assert r["verdict"] == EC.NOISE and "controllo" in r["why"]
    r = verdict(far, moved)
    assert r["verdict"] == EC.DIFF and "controllo" in r["why"]
    r = verdict(moved, same)
    assert r["verdict"] == EC.DIFF and "identico" in r["why"]
    assert verdict(same, moved)["verdict"] == EC.IDENT
    # The control shows in the yardstick column
    assert "controllo" in EC.floor_text(verdict(moved, moved))


def test_soft_requirements_leave_a_note_and_hard_ones_an_error():
    same = view([1, 2, 3])
    t = dict(runrec(a = same), stats = {"side": {"chunks": 3, "beside": 2}}, state = {"thread": "on", "head_first": False})
    res = Res({("p", "t"): t, ("p", "r1"): runrec(a = same), ("p", "r2"): runrec(a = same)})
    r = EC.evaluate_probs(probs_cmp(requires = {"thread": "on"}, soft = {"all_beside": True, "head_first": True}), res)
    assert r["verdict"] == EC.IDENT and len(r["notes"]) == 2
    assert all(n.startswith("**Attenzione**") for n in r["notes"])
    assert "1 dei 3 pezzi" in r["notes"][0] and "prima scheda" in r["notes"][1]
    # EXL3_PREFILL_PIPELINE_HEAD=second not honoured is an error
    t["state"]["head_first"] = True
    r = EC.evaluate_probs(probs_cmp(requires = {"head_first": False}), res)
    assert r["verdict"] == EC.ERR and "nonostante" in r["why"]
    # The notes reach the report
    rows = [dict(r, kind = "probs", scenario = 4)]
    plan = {"procs": [], "comparisons": [], "scenarios": [4]}
    text = "\n".join(EC.scenario_section(4, plan, EC.Results(), rows))
    assert "nonostante" in text


def test_the_plan_asks_for_chunks_beside_the_decode_only_up_to_eight_rows():
    plan = EC.build_plan(args_for("--scenarios", "3"))
    comps = {c["id"]: c for c in plan["comparisons"]}
    for n in (2, 3, 4):
        assert comps[f"s3.n{n}.a"]["requires"] == {"thread": "on", "beside": True}
    assert comps["s3.n5.a"]["requires"] == {"thread": "on"} and "pezzi attesi" in comps["s3.n5.a"]["label"]
    # Every reader row has its control, the last-chunk cases a soft check of the last chunk
    assert all(comps[f"s3.{case}.reader"]["yards"] for case in ("L33", "L249", "n2", "n5"))
    assert comps["s3.L33.reader"]["soft"] == {"all_beside": True} and comps["s3.n2.reader"]["soft"] == {"all_beside": False}
    runs = {st["id"]: st for p in plan["procs"] for st in p["steps"]}
    assert runs["s3.L33.shift"]["jobs"][-1]["start"] == {"after": "a", "tokens": 48 + 37}
    # Staggered rows stop at the first job's end; the tail checkpoint's rows do not judge drafts
    plan = EC.build_plan(args_for("--scenarios", "2,4,5"))
    comps = {c["id"]: c for c in plan["comparisons"]}
    assert comps["s2.n3.c"]["upto"] == {"before_first_end": True}
    assert not comps["s5.t1.on"]["drafts"] and not comps["s5.t2.tail"]["drafts"] and comps["s5.t2.restart"]["drafts"]
    assert comps["s4.head2"]["requires"]["head_first"] is False and comps["s4.alone"]["soft"] == {"head_first": True}
    assert comps["s4.mix.r"]["yards"][0][1]["run"] == "s4.mix.shift"


def test_a_run_that_was_to_ingest_from_scratch_and_resumed_is_an_attention_line():
    args = args_for("--scenarios", "2")
    plan = EC.build_plan(args)
    res = fake_results(plan)
    res.records["s2-fork"]["s2.n2.batch"]["forgot"] = 3
    res.records["s2-fork"]["s2.n2.batch"]["jobs"]["b"]["cached_tokens"] = 0
    res.records["s2-fork"]["s2.n2.batch"]["jobs"]["a"]["cached_tokens"] = 512
    text = "\n".join(EC.capture_checks(plan, res)[2])
    assert "`s2-fork` / `s2.n2.batch`: `a` ha ripreso 512 token" in text
    for v in res.records["s2-fork"]["s2.n2.batch"]["jobs"].values():
        v["cached_tokens"] = 0
    assert not EC.capture_checks(plan, res).get(2)


# --------------------------------------------------------------------------------------------
#   The night's robustness: split retry, rebuild, dry run, terminal
# --------------------------------------------------------------------------------------------

def test_the_split_the_server_suggests_is_read_from_its_log():
    log = ("x\nERROR openai.moe_tier: MoE device split: ... try --moe-device-split 23:240,25:253 (estimate), or give\n"
           "later: try --moe-device-split 23:250,25:253 (estimate)\n")
    assert EC.split_suggestion(log) == "23:250,25:253"
    assert EC.split_suggestion("no hint here") is None


def test_a_model_that_does_not_load_is_retried_with_the_suggested_split(tmp_path, monkeypatch):
    orch, out = orchestrate(tmp_path, monkeypatch, mode = "splitfail:23:226,25:253", scenarios = "2,5")
    o = orch.res.procs["s2-fork"]
    assert o["exit"] == 0 and o["attempts"] == 2 and "log_text" not in o
    assert orch.args.device_split == "23:240,25:253"
    # Every later process starts with it
    spec = json.loads((out / "procs" / "s5-fork-a" / "spec.json").read_text())
    argv = spec["server_argv"]
    assert argv[argv.index("--moe-device-split") + 1] == "23:240,25:253"
    assert orch.res.procs["s5-fork-a"]["attempts"] == 1
    report = (out / "REPORT.md").read_text()
    assert "**Attenzione**: con `--moe-device-split 23:226,25:253` il processo `s2-fork` non ha caricato" in report
    assert summary_cells(report)[0][1] == EC.IDENT, report


def test_a_split_that_keeps_failing_is_tried_a_bounded_number_of_times(tmp_path, monkeypatch):
    orch, out = orchestrate(tmp_path, monkeypatch, mode = "splitfail:23:226,25:253|23:240,25:253", scenarios = "2",
                            extra = ("--split-retries", "1"))
    o = orch.res.procs["s2-fork"]
    assert o["exit"] == 4 and o["attempts"] == 2 and o["load_failed"]
    report = (out / "REPORT.md").read_text()
    assert "`s2-fork` / `load` è fallito" in report and summary_cells(report)[0][1] == EC.ERR


def test_the_report_is_rebuilt_from_the_run_directory(tmp_path, monkeypatch):
    orch, out = orchestrate(tmp_path, monkeypatch, scenarios = "1")
    before = (out / "REPORT.md").read_text()
    (out / "REPORT.md").unlink()
    # As after an orchestrator killed in the middle: the last process still "running"
    summary = json.loads((out / "summary.json").read_text())
    summary["procs"]["s1-stock-b"]["running"] = True
    summary["finished"] = False
    (out / "summary.json").write_text(json.dumps(summary))
    path = EC.rebuild_report(str(out))
    after = open(path).read()
    assert summary_cells(after) == summary_cells(before)
    assert "il giro è stato interrotto" in after and "`s1-stock-b` si è fermato (interrotto" in after
    assert EC.main(["--rebuild", str(out)]) == 0


def test_the_dry_run_writes_nothing_and_a_used_directory_is_refused(tmp_path, capsys):
    out = tmp_path / "night"
    args = EC.parse_cli(["--out", str(out), "--python", sys.executable, "--placement", "", "--layer-budget", "uniform",
                         "--server-path", str(tmp_path / "no-server")])
    plan = EC.build_plan(args)
    problems, notes = EC.preflight(args, plan)
    assert not out.exists()
    assert any("server non trovato" in p for p in problems)
    out.mkdir()
    (out / "plan.json").write_text("{}")
    problems, _ = EC.preflight(args, plan)
    assert any("contiene già un giro" in p for p in problems)
    assert EC.main(["--dry-run", "--out", str(tmp_path / "other"), "--placement", "", "--layer-budget", "uniform"]) in (0, 1)
    assert not (tmp_path / "other").exists() and "Totale stimato" in capsys.readouterr().out


def test_the_wheel_and_the_server_s_pin_are_the_release_the_branch_is_on(tmp_path, monkeypatch):
    # One constant names upstream's release: the installed wheel, the pin of the server's checkout
    # (by default the standard one) and the report's labels follow it
    assert (EC.BASE_RELEASE, EC.FORK_BRANCH) == ("1.6.3", "flash-next-1.6.3")
    assert EC.SERVER_DEFAULT == "~/Desktop/exllamav3-openai-server"
    server = tmp_path / "server"
    (server / "openai").mkdir(parents = True)
    (server / "openai" / "config.py").write_text("")
    fork = EC.FORK_DEFAULT
    args = EC.parse_cli(["--out", str(tmp_path / "night"), "--placement", "", "--layer-budget", "uniform",
                         "--server-path", str(server)])
    plan = EC.build_plan(args)
    monkeypatch.setattr(EC, "gpu_processes", lambda: [])
    monkeypatch.setattr(EC, "validate_server_args", lambda *a: [])

    def problems(pin, version):
        (server / "pyproject.toml").write_text(f'dependencies = [\n    "exllamav3=={pin}",\n]\n')
        monkeypatch.setattr(EC, "probe_library", lambda python, path: {
            "version": version, "exllamav3_ext": "/env/lib/exllamav3_ext.so",
            "exllamav3": os.path.join(fork if path else "/env/lib", "exllamav3", "__init__.py")})
        return " | ".join(EC.preflight(args, plan)[0])

    ok = problems("1.6.3", "1.6.3+cu132.torch2.11.0")
    assert "fissa" not in ok and "wheel" not in ok
    old = problems("1.6.2", "1.6.2+cu132.torch2.11.0")
    assert "non fissa exllamav3==1.6.3" in old and "exllamav3 è 1.6.2+cu132.torch2.11.0, serve la wheel 1.6.3" in old
    assert "serve la wheel 1.6.3" in problems("1.6.3", "1.6.30")
    assert EC.SCENARIO_TITLES[1].endswith("upstream 1.6.3")


def test_a_terminal_that_went_away_does_not_end_the_night(tmp_path):
    class Gone:
        def write(self, *_):
            raise OSError(5, "Input/output error")

        def flush(self):
            raise OSError(5, "Input/output error")

    args = args_for("--scenarios", "2")
    orch = EC.Orchestrator(EC.build_plan(args), args, str(tmp_path), out = Gone())
    orch.say("still here")
    assert "still here" in (tmp_path / "orchestrator.log").read_text()


def test_the_children_write_nothing_outside_the_run_directory():
    args = args_for("--host-cores", "0")
    env = EC.child_env({"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                       {"library": "fork", "scenarios": [2], "env": {}}, args, "/run")
    for key in ("XDG_CACHE_HOME", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "PYTHONPYCACHEPREFIX"):
        assert env[key].startswith("/run/"), key
    assert "CUDA_VISIBLE_DEVICES" not in env and "PYTHONDONTWRITEBYTECODE" not in env
    assert env["EXL3_MOE_HOST_CORES"] == "0" and env["EXL3_MOE_FUSED_DET"] == "1"
    env = EC.child_env({}, {"library": "stock", "scenarios": [1], "env": {}}, args_for("--host-cores", ""), "/run")
    assert "EXL3_MOE_HOST_CORES" not in env


# --------------------------------------------------------------------------------------------
#   The stress over the toy model
# --------------------------------------------------------------------------------------------

@pytest.fixture
def quick_env():
    """The toy with chunks that take no time of their own (the stress cancels jobs at any point,
    which the timed chunks of the other tests do not expect)."""
    x = toy_env(steps = 0)
    yield x
    for pool in (x.gen._side_pool, x.gen._pipe_pool):
        if pool is not None:
            pool.shutdown(wait = True)


def test_the_stress_over_the_toy_drains_and_counts(quick_env):
    env = quick_env
    # (the toy job keys its pages on a one-character name)
    make, names = env.make_job, {}
    env.make_job = lambda name, ids, max_new, **kw: make(names.setdefault(name, chr(0x4E00 + len(names))), ids, max_new, **kw)
    env.prompt_ids_text = lambda text: torch.tensor([[ord(c) % toy.VOCAB for c in text[:6]]], dtype = torch.long)
    env.apply_switches({"thread": True, "stagger": True, "stagger_min_jobs": 2})
    spec = {"op": "stress", "id": "s7.stress", "scenario": 7, "minutes": 0.04, "seed": 3, "step_limit": 30,
            "drain_timeout": 60, "mean_gap": 0.03, "short_tokens": (4, 12), "doc_tokens": 6,
            "doc_pages": (1,), "long_pages": (2,)}
    rec = CH.run_stress(env, spec)
    s = rec["stress"]
    assert rec["ok"], rec
    assert s["drained"] and not s["exceptions"] and s["job_errors"] == 0 and not env.gen.num_remaining_jobs()
    assert s["jobs"] > 5 and s["done"] + s["cancelled"] == s["jobs"]
    assert s["iterations"] > 0 and s["max_step_s"] < 30 and s["side"]["chunks"] >= 1


def test_the_child_records_ends_and_drafts_and_runs_the_stress_with_the_servers_sweeps(tmp_path):
    steps = [
        EC.gen_step("g0", 7, disk = "s7"),
        EC.run_step("free", 7, [EC.jobspec("a", "q0", 5), EC.jobspec("b", "q1", 9)]),
        {"op": "stress", "id": "s7.stress", "scenario": 7, "minutes": 0.03, "seed": 5, "step_limit": 30,
         "drain_timeout": 30, "mean_gap": 0.05, "short_tokens": (2, 6), "doc_tokens": 3, "doc_pages": (1,),
         "long_pages": (2,)},
        {"op": "close", "id": "s7.close", "scenario": 7},
    ]
    out, recs = run_child(tmp_path, "fork", steps, sweeps = "server")
    assert out.returncode == 0, out.stdout + out.stderr
    assert recs["g0"]["ok"] and not recs["g0"]["warnings"]
    # The server's tier hooked on the generator, as the server does when it makes one
    assert recs["load"]["sweeps"] == "server" and recs["g0"]["tier_attached"]
    jobs = recs["free"]["jobs"]
    assert jobs["a"]["ended"] is not None and jobs["b"]["ended"] >= jobs["a"]["ended"]
    assert len(jobs["a"]["acc"]) == len(jobs["a"]["digest"]) == 5 and len(jobs["b"]["rej"]) == 9
    st = recs["s7.stress"]
    assert st["ok"], st
    s = st["stress"]
    assert s["drained"] and not s["exceptions"] and s["jobs"] >= 1 and s["done"] + s["cancelled"] == s["jobs"]
    assert recs["s7.close"]["ok"]
