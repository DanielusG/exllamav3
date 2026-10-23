from __future__ import annotations
from typing_extensions import override
import os
import numpy as np
import torch
from .cache import CacheLayer
from .fp16 import CacheLayer_fp16
from .quant import CacheLayer_quant
from ..constants import PAGE_SIZE


# Raw keys of the block being built only (EXL3_QSA_RAW_TAIL=1, Cache(qsa_raw_tail = True)): the
# per-page raw plane shrinks to a ring of RAW_TAIL_DEFAULT tokens. RAW_TAIL_GRAPH_ROWS is the longest
# append a decode graph slot makes (BCAttn's MAX_QLEN): the slot's pool update reads that append's
# raw keys back from the ring, so the ring holds an append plus the compress_ratio - 1 tokens of the
# block it starts in
RAW_TAIL_ENV = "EXL3_QSA_RAW_TAIL"
RAW_TAIL_DEFAULT = 32
RAW_TAIL_GRAPH_ROWS = 16


def resolve_raw_tail(value: int | bool | None) -> int:
    """
    Ring size in tokens (0 = the full raw plane) for Cache(qsa_raw_tail = value): None reads
    EXL3_QSA_RAW_TAIL, True (or "1") is RAW_TAIL_DEFAULT, an int above 1 is the ring itself
    """
    if value is None:
        env = os.environ.get(RAW_TAIL_ENV, "0").strip()
        try:
            value = int(env or "0")
        except ValueError:
            value = RAW_TAIL_DEFAULT
    if value is True:
        value = RAW_TAIL_DEFAULT
    value = int(value)
    if value == 1:
        value = RAW_TAIL_DEFAULT
    if value < 0 or value > PAGE_SIZE:
        raise ValueError(f"qsa_raw_tail: ring of {value} tokens, must be between 2 and {PAGE_SIZE}")
    return value


def raw_tail_max_rows(raw_tail: int, compress_ratio: int) -> int:
    """
    The longest append a ring of raw_tail tokens serves, for both of its uses: a decode graph slot
    reads its whole append back from the ring, and an append whose tail can be taken back
    (speculative rows) leaves in the ring the block the next append starts in. Either way an
    append and the compress_ratio - 1 tokens before it must fall in distinct ring slots
    """
    return raw_tail - (compress_ratio - 1)


def raw_tail_refusal(model) -> str | None:
    """
    Why a model's generator cannot use the ring, None if it can. A model without recurrent states
    reuses a partial page from the prompt cache (Job's partial-page reuse, copy_page with a token
    count off a block boundary) and rewinds banned strings by truncation, to any depth; a model
    whose prompt passes can run past a chunk's end takes them back (atomic_mm_prefill). Either
    would read raw keys the ring no longer holds
    """
    caps = getattr(model, "caps", None) or {}
    if not caps.get("recurrent_states"):
        return "the model has no recurrent states"
    if caps.get("atomic_mm_prefill"):
        return "the model's prompt passes can run past a chunk's end"
    return None


def check_raw_tail(caches: list, max_rows: int, model = None):
    """
    Refuse a generator whose longest speculative pass (max_rows rows per sequence: the draft
    window plus the token it follows) does not fit the QSA raw-key ring of one of its caches, or
    whose model cannot use the ring at all (raw_tail_refusal; the MTP head's cache follows its
    target, the generator's model)
    """
    for cache in caches:
        if cache is None:
            continue
        for layer in cache.layers.values():
            ring = getattr(layer, "raw_tail", 0)
            if not ring:
                continue
            why = raw_tail_refusal(model) if model is not None else None
            if why:
                raise ValueError(f"QSA raw-key ring (qsa_raw_tail, {RAW_TAIL_ENV}) in a cache of this generator: {why}")
            if max_rows > raw_tail_max_rows(ring, layer.compress_ratio):
                raise ValueError(
                    f"QSA raw-key ring of {ring} tokens (qsa_raw_tail, {RAW_TAIL_ENV}) is too short for "
                    f"verification passes of {max_rows} rows: at most "
                    f"{raw_tail_max_rows(ring, layer.compress_ratio) - 1} draft tokens, or a larger ring"
                )


class QSAPlanes:
    """
    Mixin adding the QSA indexer's side planes to a K/V cache layer: per-token RAW indexer keys
    (unnormed, unroped) and per-4-token-block POOLED keys (fp32 mean -> norm -> rope at block
    start, written once a block completes). PAGE_SIZE is a multiple of the compress ratio, so
    blocks never straddle pages and page sharing / copy-on-write / defrag carry the planes along
    with the KV they describe. The planes stay fp16 whatever the K/V storage is: they are the
    input to every block score, and 128 + 32 values per token is small next to the K/V.

    raw_tail > 0 (the fork's switch, see resolve_raw_tail): the only reader of the raw plane is the
    pooled-key rebuild of the blocks an append touches, and of those only the block the append
    starts in has raw keys from before the append (at most compress_ratio - 1 of them). So the raw
    plane keeps a ring of raw_tail tokens per page instead, token t of a page at slot
    t % page_size % raw_tail: the last raw_tail positions written to a page. A complete page needs
    no raw keys, and the ring stays out of get_tensors (page images in the CPU and disk tiers,
    defrag rotation): only complete pages are stored, restored or kept by hash, and a page is
    written again only from its start, or within the speculative rows the generator checks
    (check_raw_tail). The planes' kernels address the ring through the same block table
    (qsa_triton._qsa_raw_tail_append_kernel, _qsa_pool_update_ring_kernel and
    _qsa_pool_update_rows_kernel); the pooled keys come out bit for bit as from the full plane.
    """

    def _init_planes(self, attention, max_num_tokens: int, raw_tail: int = 0):
        idx = attention.qsa_indexer
        assert idx is not None
        self.index_head_dim = idx.head_dim
        self.compress_ratio = idx.compress_ratio
        assert PAGE_SIZE % self.compress_ratio == 0
        num_pages = max_num_tokens // PAGE_SIZE
        self.raw_tail = raw_tail
        if raw_tail and raw_tail_max_rows(raw_tail, self.compress_ratio) < RAW_TAIL_GRAPH_ROWS:
            raise ValueError(
                f"QSA raw-key ring of {raw_tail} tokens cannot serve the decode graph's appends of "
                f"{RAW_TAIL_GRAPH_ROWS} rows (at least {RAW_TAIL_GRAPH_ROWS + self.compress_ratio - 1})"
            )
        self.raw_k_shape = (num_pages, raw_tail or PAGE_SIZE, self.index_head_dim)
        self.pooled_shape = (num_pages, PAGE_SIZE // self.compress_ratio, self.index_head_dim)
        self.raw_k = None
        self.pooled = None

    @override
    def alloc(self, device: torch.device):
        super().alloc(device)
        self.raw_k = torch.zeros(self.raw_k_shape, dtype = torch.half, device = device)
        self.pooled = torch.zeros(self.pooled_shape, dtype = torch.half, device = device)

    @override
    def free(self):
        super().free()
        self.raw_k = None
        self.pooled = None

    @override
    def copy_page(self, source, from_page: int, to_page: int, num_tokens: int):
        super().copy_page(source, from_page, to_page, num_tokens)
        if self.raw_tail:
            # The whole ring: it holds the destination's partial block only if the source wrote
            # it among its last raw_tail positions. The one caller that copies a partial block,
            # the prompt cache's partial-page reuse, runs for models without recurrent states,
            # for which Cache refuses the ring
            self.raw_k[to_page].copy_(source.raw_k[from_page], non_blocking = True)
        else:
            self.raw_k[to_page, :num_tokens].copy_(source.raw_k[from_page, :num_tokens], non_blocking = True)
        nb = (num_tokens + self.compress_ratio - 1) // self.compress_ratio
        self.pooled[to_page, :nb].copy_(source.pooled[from_page, :nb], non_blocking = True)

    @override
    def get_tensors(self):
        if self.raw_tail:
            return super().get_tensors() + [self.pooled]
        return super().get_tensors() + [self.raw_k, self.pooled]

    @override
    def storage_size(self):
        return super().storage_size() + \
            (np.prod(self.raw_k_shape) + np.prod(self.pooled_shape)) * torch.half.itemsize

    def raw_tail_saved_bytes(self) -> int:
        """VRAM the ring saves against the full raw plane (0 without it)"""
        if not self.raw_tail:
            return 0
        return self.raw_k_shape[0] * (PAGE_SIZE - self.raw_tail) * self.index_head_dim * torch.half.itemsize


class CacheLayer_qsa(QSAPlanes, CacheLayer_fp16):
    """fp16 KV cache layer with the QSA indexer planes."""

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
        raw_tail: int = 0,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)
        self._init_planes(attention, max_num_tokens, raw_tail)

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_qsa,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "raw_tail": self.raw_tail,
            }
        }


class CacheLayer_qsa_quant(QSAPlanes, CacheLayer_quant):
    """Quantized KV cache layer (CacheLayer_quant packing, read online by the dense and the
    gathered sparse attention kernels) with the fp16 QSA indexer planes."""

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int,
        v_bits: int,
        compand_a: float = 0.0,
        raw_tail: int = 0,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens, k_bits, v_bits, compand_a)
        self._init_planes(attention, max_num_tokens, raw_tail)

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_qsa_quant,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.k_bits,
                "v_bits": self.v_bits,
                "raw_tail": self.raw_tail,
            }
        }
