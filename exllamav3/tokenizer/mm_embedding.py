from __future__ import annotations
import bisect
import hashlib
import json
import os
import struct
import threading
import numpy as np
import torch

FIRST_MM_EMBEDDING_INDEX = 1000000000

# Assume no model will have more than one billion regular text tokens, and assign dynamic token IDs starting from
# that index.

class MMTokenAllocator:

    next_token_index: int

    def __init__(self):
        self.next_token_index = FIRST_MM_EMBEDDING_INDEX

    def allocate(self, num_tokens):
        idx = self.next_token_index
        self.next_token_index += num_tokens
        return idx

global_allocator = MMTokenAllocator()


# ---------------------------------------------------------------------------------------------------------------
# Fork: content-addressed multimodal pages (Generator(mm_content_hash = True) or EXL3_MM_CONTENT_HASH=1; off by
# default, FORK.md). Placeholder ids are handed out per process, so a page that holds them is hashed over ids
# that can stand for another image in another process, and upstream keeps it, and the rest of its chain, off
# the disk tier. With the switch on, an embedding gets a digest of its content when it is made, and the range of
# ids it is handed is registered with that digest (global_registry): the page table then hashes a placeholder as
# (digest, row of the embedding) instead of as its id (generator.pagetable.content_hash_checksum), the same in
# every process. An embedding made with the switch off has no digest, and a page that holds its ids is hashed
# and kept as upstream does.
# ---------------------------------------------------------------------------------------------------------------

DIGEST_SIZE = 16
_content_digests = None


def set_content_digests(on: bool | None):
    """
    Give every embedding made from now on a digest of its content (True), none (False), or follow
    EXL3_MM_CONTENT_HASH (None, the default). Generator(mm_content_hash = True) sets True. An embedding keeps
    what it was made with.
    """
    global _content_digests
    _content_digests = on


def content_digests() -> bool:
    """Whether an embedding made now gets a content digest"""
    if _content_digests is not None:
        return _content_digests
    return os.environ.get("EXL3_MM_CONTENT_HASH", "0") not in ("", "0")


def _update(hasher, b: bytes | memoryview):
    """Length-prefixed, so that no two sequences of fields give the same input"""
    hasher.update(struct.pack("<Q", len(b)))
    hasher.update(b)


def _plain(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, (tuple, list)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def _hash_tensor(hasher, t: torch.Tensor):
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    t = t.contiguous()
    _update(hasher, json.dumps([str(t.dtype), list(t.shape)]).encode())
    _update(hasher, memoryview(t.reshape(-1).view(torch.uint8).numpy()))


def embedding_digest(
    embeddings: torch.Tensor,
    deepstack_embeddings: list[torch.Tensor] | None = None,
    grid_thw: tuple | None = None,
    mrope_merge_size: int | None = None,
    align: int = 0,
    align_phase: int = 0,
    align_lead: int = 0,
) -> bytes:
    """
    Digest of everything a forward pass reads of an embedding: the rows of the embedding and of its deepstack
    embeddings (dtype, shape and bytes: the row count is the placeholder count), the MRoPE grid and merge size,
    and the alignment fields. What the vision tower was given (pixels, preprocessing) is not hashed, its output
    is: two embeddings with the same digest feed the model the same values, whatever made them, and a tower
    that does not compute the same bytes twice gives two digests (a cache miss, never a page reused for other
    values). The metadata dict and the text alias are not read by a forward pass.
    """
    hasher = hashlib.blake2b(digest_size = DIGEST_SIZE, person = b"exl3-mm-emb")
    meta = {
        "grid_thw": _plain(grid_thw),
        "mrope_merge_size": _plain(mrope_merge_size),
        "align": [align, align_phase, align_lead],
        "deepstack": None if deepstack_embeddings is None else len(deepstack_embeddings),
    }
    _update(hasher, json.dumps(meta, sort_keys = True).encode())
    _hash_tensor(hasher, embeddings)
    for de in deepstack_embeddings or ():
        _hash_tensor(hasher, de)
    return hasher.digest()


class MMDigestRegistry:
    """
    The ranges of placeholder ids handed out together with a content digest, in the order they were handed out,
    so increasing and disjoint (the allocator only moves on). Append-only: an id keeps its digest for the life of
    the process, since pages that hold it can outlive the embedding (in VRAM, in the CPU tier, in a requeued
    job). An entry is ~100 bytes. Lookups take no lock: a range is appended before its start is, and a reader
    never sees a start without its range.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._table = ([], [])      # (first ids, [(first, end, digest)]), same order
        self._vectors = None        # (count, first ids, ends, ranges) as arrays, rebuilt as the table grows

    def allocate(self, allocator: MMTokenAllocator, num_tokens: int, digest: bytes) -> int:
        """Hand out num_tokens ids from the allocator and register them with digest, as one step"""
        assert isinstance(digest, bytes) and len(digest) == DIGEST_SIZE
        with self._lock:
            first = allocator.allocate(num_tokens)
            starts, ranges = self._table
            if ranges and first < ranges[-1][1]:
                # The allocator went back (a test standing for a new process does that): the ids handed out
                # before may stand for other content from now on
                starts, ranges = [], []
                self._table = (starts, ranges)
            ranges.append((first, first + num_tokens, digest))
            starts.append(first)
        return first

    def __len__(self):
        return len(self._table[0])

    def lookup(self, token_id: int) -> tuple[bytes, int] | None:
        """(digest, row of the embedding) of a placeholder id, None when it has no registered digest"""
        starts, ranges = self._table
        i = bisect.bisect_right(starts, token_id) - 1
        if i < 0:
            return None
        first, end, digest = ranges[i]
        return (digest, token_id - first) if token_id < end else None

    def _arrays(self):
        starts, ranges = self._table
        n = len(starts)
        v = self._vectors
        if v is None or v[0] != n or v[3] is not ranges:
            v = (
                n,
                np.array([r[0] for r in ranges[:n]], dtype = np.int64),
                np.array([r[1] for r in ranges[:n]], dtype = np.int64),
                ranges,
            )
            self._vectors = v
        return v

    def _resolve(self, vals: np.ndarray):
        """Range index of each placeholder id in vals, and a mask of the ids that have one"""
        _, st, en, ranges = self._arrays()
        if not st.size:
            return None, np.zeros(vals.shape, dtype = bool), ranges, st
        idx = np.searchsorted(st, vals, side = "right") - 1
        ok = (idx >= 0) & (vals < en[np.maximum(idx, 0)])
        return idx, ok, ranges, st

    def first_undigested(self, ids: np.ndarray) -> int | None:
        """Position of the first placeholder id in the 1-D int64 ids that has no registered digest, or None"""
        pos = np.flatnonzero(ids >= FIRST_MM_EMBEDDING_INDEX)
        if not pos.size:
            return None
        _, ok, _, _ = self._resolve(ids[pos])
        bad = np.flatnonzero(~ok)
        return int(pos[bad[0]]) if bad.size else None

    def canonical(self, ids: np.ndarray) -> tuple[np.ndarray, list[tuple[int, bytes]]] | None:
        """
        The 1-D int64 ids as content: (ids with each placeholder id replaced by -1 minus its row in its
        embedding, the digests of the placeholders in order as runs (count, digest), consecutive placeholders
        with the same digest in one run). Two embeddings with the same digest give the same result. None when
        a placeholder id has no registered digest.
        """
        pos = np.flatnonzero(ids >= FIRST_MM_EMBEDDING_INDEX)
        if not pos.size:
            return ids, []
        vals = ids[pos]
        idx, ok, ranges, st = self._resolve(vals)
        if not ok.all():
            return None
        canon = ids.copy()
        canon[pos] = -1 - (vals - st[idx])
        cut = (np.flatnonzero(idx[1:] != idx[:-1]) + 1).tolist()
        runs = []
        for a, b in zip([0] + cut, cut + [idx.size]):
            digest = ranges[int(idx[a])][2]
            if runs and runs[-1][1] == digest:
                runs[-1] = (runs[-1][0] + b - a, digest)
            else:
                runs.append((b - a, digest))
        return canon, runs


global_registry = MMDigestRegistry()


class MMEmbedding:
    """
    Container for one embedding (image etc.) and associated metadata
    """

    def __init__(
        self,
        embeddings: torch.Tensor | None = None,
        token_string: torch.Tensor | None = None,
        text_alias: str | None = None,
        deepstack_embeddings: list[torch.Tensor] | None = None,
        grid_thw: tuple | None = None,
        mrope_merge_size: int | None = None,
        align: int = 0,
        align_phase: int = 0,
        align_lead: int = 0,
        imp: dict | None = None
    ):
        """
        :param embeddings:
            Embeddings, shape (num_tokens, input_dim)

        :param token_string:
            Tokenized representation, with -1 as a placeholder for the MM embeddings

        :param text_alias:
            Text string to represent this embedding for tokenizing

        :param align:
            Position alignment for embeddings whose block must start on a fixed phase of a
            model-side grouping (DeepSeek-V4: the CSA compressor's 4-token groups). The
            token_string then begins with align_lead spare rows (pads) and the tokenizer emits
            only the suffix that places row align_lead at a prompt position p with
            p % align == align_phase. 0 = no alignment.
        """

        if imp:
            self.metadata = imp["metadata"]
            self.full_length = imp["full_length"]
            self.mm_length = imp["mm_length"]
            self.first_index = imp["first_index"]
            self.last_index = imp["last_index"]
            self.text_alias = imp["text_alias"]
            self.grid_thw = imp["grid_thw"]
            self.mrope_merge_size = imp["mrope_merge_size"]
            self.embeddings = imp["embeddings"]
            self.deepstack_embeddings = imp["deepstack_embeddings"]
            self.align = imp.get("align", 0)
            self.align_phase = imp.get("align_phase", 0)
            self.align_lead = imp.get("align_lead", 0)
            self.content_digest = imp.get("content_digest")
            self.token_string = None
            self.token_list = None
            return

        global global_allocator

        if deepstack_embeddings is not None:
            assert all(de.shape == embeddings.shape for de in deepstack_embeddings), \
                "Deepstack embeddings shape mismatch"

        self.metadata = {}
        self.full_length = token_string.shape[-1]
        self.mm_length = embeddings.shape[-2]
        # Fork: with content digests on (content_digests), the digest is computed here, once, and the ids are
        # handed out and registered with it in one step (MMDigestRegistry.allocate)
        if content_digests():
            self.content_digest = embedding_digest(
                embeddings, deepstack_embeddings, grid_thw, mrope_merge_size, align, align_phase, align_lead
            )
            self.first_index = global_registry.allocate(global_allocator, self.mm_length, self.content_digest)
        else:
            self.content_digest = None
            self.first_index = global_allocator.allocate(self.mm_length)
        self.last_index = self.first_index + self.mm_length
        self.embeddings = embeddings
        self.deepstack_embeddings = deepstack_embeddings
        self.text_alias = text_alias or f"<$EMB_{self.first_index}$>"

        # MRoPE
        self.grid_thw = grid_thw
        self.mrope_merge_size = mrope_merge_size

        # Position alignment (see align)
        assert align == 0 or (0 <= align_phase < align and 0 <= align_lead < align and align_lead < self.full_length), \
            "MMEmbedding: align_lead / align_phase must be smaller than align"
        self.align = align
        self.align_phase = align_phase
        self.align_lead = align_lead

        # not exported for TP
        r = torch.arange(self.first_index, self.first_index + self.mm_length, dtype = torch.long)
        m = (token_string == -1)
        token_string.masked_scatter_(m, r)
        self.token_string = token_string
        self.token_list = token_string[0].tolist()

    def token_list_at(self, position: int) -> list[int]:
        """Token ids to splice in when the embedding's first emitted row lands at prompt
        position `position`: the whole token_list, or, for aligned embeddings, the suffix that
        skips the spare leading rows not needed to put row align_lead on the required phase."""
        if not self.align:
            return self.token_list
        n_pad = (self.align_phase - position) % self.align
        assert n_pad <= self.align_lead
        return self.token_list[self.align_lead - n_pad:]


def send_embeddings(producer, ies: list[MMEmbedding]):
    return {
        "method": "list",
        "data": [
            {
                "metadata": ie.metadata,
                "full_length": ie.full_length,
                "mm_length": ie.mm_length,
                "first_index": ie.first_index,
                "last_index": ie.last_index,
                "text_alias": ie.text_alias,
                "grid_thw": ie.grid_thw,
                "mrope_merge_size": ie.mrope_merge_size,
                "align": ie.align,
                "align_phase": ie.align_phase,
                "align_lead": ie.align_lead,
                "embeddings": producer.send(ie.embeddings, cache_id = id(ie.embeddings)),
                "deepstack_embeddings": [
                    producer.send(dse, cache_id = id(dse))
                    for dse in ie.deepstack_embeddings
                ] if ie.deepstack_embeddings is not None else None
            }
            for ie in ies
        ]
    }


def recv_embeddings(consumer, recv) -> list[MMEmbedding]:
    result = []
    assert recv.get("method") == "list", "Consumer expected list"
    for imp in recv["data"]:
        imp["embeddings"] = consumer.recv(imp["embeddings"])
        imp["deepstack_embeddings"] = [
            consumer.recv(dse) for dse in imp["deepstack_embeddings"]
        ] if imp.get("deepstack_embeddings") else None
        result.append(MMEmbedding(imp = imp))
    return result