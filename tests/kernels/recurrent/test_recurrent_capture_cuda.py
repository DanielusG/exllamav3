"""
The recurrent checkpoint in pass (fork: Generator.recurrent_checkpoint_in_pass, cache/recurrent_capture.py) on
the kernels it relies on. A layer copies its state at a page boundary inside a pass; the checkpoint is right
when that copy is what a pass ending at the boundary leaves in the layer's slot, from the same inputs, bit for
bit. Checked here on the device, where the scan, the conv state write and the PLE stream run as kernels (the
CPU suite's tests/test_recurrent_capture.py checks the split on a torch reference and the generator's side):

- the GDN scan: gated_delta_rule_fn with split_at hands on_split the fp32 state after split_at tokens. Up to
  split_at it runs the vendored fla chunk kernel over the sub-ranges a pass of split_at tokens runs, so the
  state, stored as the pool stores it, is the pool row such a pass writes, and the output rows before
  split_at are that pass's;
- the GDN conv window: conv_window over a pass's inputs is what causal_conv1d_update's state write leaves in
  the slot after a pass of `end` tokens, on each path the prefill takes: the Triton kernels over a token-major
  fp16 or fp32 projection (the default, and EXL3_GDN_PROJ_FP32) and over channel-major bf16, and the CUDA
  kernel (channel-major bf16, up to 32 tokens: never a capture, whose boundary is a page in, but the same
  window);
- the PLE layer: forward_streams with split_at hands on_split the conv state a pass of split_at positions
  leaves, on its fast path (ext.ple_forward_streams over the same row slabs), with and without a carried
  state, the slabs up to the boundary shorter or longer than EXL3_PLE_SUB_CHUNK.
"""
from types import SimpleNamespace

import pytest
import torch

from exllamav3.modules.gated_delta_net_fn import causal_conv1d_update, conv_window, gated_delta_rule_fn
from exllamav3.modules import ple as ple_mod
from exllamav3.modules.ple import PLELayer


def _gdn_inputs(device, seqlen, nk, nv, dk, dv, seed):
    g_ = torch.Generator(device = "cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator = g_)
    qkv_dim = 2 * nk * dk + nv * dv
    mixed_qkv = (r(1, seqlen, qkv_dim) * 0.25).bfloat16().to(device)
    g = (r(1, seqlen, nv) * 0.5 - 1.0).float().to(device)
    beta = torch.sigmoid(r(1, seqlen, nv)).bfloat16().to(device)
    # One pool row (GDNLayerState: (2 * max_batch_size, 1, HV, K, V))
    state0 = (r(1, nv, dk, dv) * 0.05).float().to(device)
    return mixed_qkv, g, beta, state0


def _scan(mixed_qkv, g, beta, pool, row, nk, nv, dk, dv, split_at = None, on_split = None):
    # The chunked path reads its pool rows from params (recurrent_slots_scan: GDNState's 2 * slot + parity)
    rows = torch.tensor([row], dtype = torch.int)
    return gated_delta_rule_fn(
        mixed_qkv = mixed_qkv, beta = beta, g = g, recurrent_state = pool, recurrent_slots = rows.to(mixed_qkv.device),
        history = False, save_state = True, num_k_heads = nk, num_v_heads = nv, k_dim = nk * dk, v_dim = nv * dv,
        k_head_dim = dk, v_head_dim = dv, params = {"recurrent_slots_scan": rows}, split_at = split_at,
        on_split = on_split,
    )


@pytest.mark.parametrize("state_dtype", [torch.half, torch.float], ids = ["fp16_pool", "fp32_pool"])
@pytest.mark.parametrize(
    "seqlen,split_at,nk,nv,dk,dv",
    [
        (611, 512, 2, 4, 128, 128),          # a prompt's last chunk: the boundary, then the last partial page
        (4040, 3840, 2, 4, 128, 128),        # the boundary deep in a chunk past the 2048 sub-range
        (2304, 2048, 2, 4, 128, 128),        # the boundary on a sub-range's end
        (1279, 256, 16, 32, 128, 128),       # Qwen3.8-Flash-Next's heads, the boundary a page in
    ],
)
@torch.inference_mode()
def test_the_scan_s_state_at_the_split_is_the_state_a_pass_ending_there_stores(device, state_dtype, seqlen, split_at,
                                                                                 nk, nv, dk, dv):
    mixed_qkv, g, beta, state0 = _gdn_inputs(device, seqlen, nk, nv, dk, dv, seed = seqlen)
    # A pool of three rows; the pass works in row 1
    pool = torch.zeros((3, 1, nv, dk, dv), dtype = state_dtype, device = device)
    pool[1].copy_(state0)
    start = pool[1].clone()

    # The pass that ends at the boundary
    short = pool.clone()
    out_short = _scan(mixed_qkv[:, :split_at], g[:, :split_at], beta[:, :split_at], short, 1, nk, nv, dk, dv)

    # The whole pass with the copy at the boundary, as GatedDeltaNet.forward takes it
    got = []
    out_full = _scan(mixed_qkv, g, beta, pool, 1, nk, nv, dk, dv, split_at = split_at,
                     on_split = lambda st: got.append(st.to(pool.dtype, copy = True).unsqueeze(1)))
    torch.cuda.synchronize()
    assert len(got) == 1
    captured = got[0]
    assert captured.shape == short[1:2].shape and captured.dtype == short.dtype
    assert torch.equal(captured, short[1:2]), \
        f"captured state differs from the stored one by {(captured.float() - short[1:2].float()).abs().max().item():.3e}"
    assert torch.equal(out_full[:, :split_at], out_short), "output rows before the boundary differ"
    # (the copy is the layer's own: the pass went on past the boundary and wrote its final state into the
    # pool's row, which the comparison above did not see)
    assert not torch.equal(pool[1], start) and not torch.equal(pool[1:2], captured)


@torch.inference_mode()
def test_a_scan_without_a_split_or_one_outside_the_pass_calls_nothing(device):
    nk, nv, dk, dv = 2, 4, 128, 128
    mixed_qkv, g, beta, state0 = _gdn_inputs(device, 700, nk, nv, dk, dv, seed = 1)
    for split_at in (None, 0, 700, 900):
        pool = torch.zeros((2, 1, nv, dk, dv), dtype = torch.half, device = device)
        pool[0].copy_(state0)
        calls = []
        _scan(mixed_qkv, g, beta, pool, 0, nk, nv, dk, dv, split_at = split_at, on_split = calls.append)
        assert calls == [], split_at


def _conv_case(device, token_major, in_dtype, dim, kc, n, seed):
    g_ = torch.Generator(device = "cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator = g_)
    x = (r(1, n, dim) * 2.0).to(in_dtype).to(device)
    if not token_major:
        x = x.transpose(1, 2).to(torch.bfloat16).contiguous()
    weight = (r(dim, kc) * 0.5).bfloat16().to(device)
    bias = (r(dim) * 0.1).bfloat16().to(device)
    # The slot pool as GDNLayerState makes it: conv_kernel_size + max_history columns, bf16
    pool = (r(3, dim, kc + 3) * 0.3).bfloat16().to(device)
    return x, weight, bias, pool


@pytest.mark.parametrize(
    "token_major,in_dtype,end,n",
    [
        (True, torch.half, 256, 355),        # the default prefill: fp16 projection read in place
        (True, torch.half, 512, 611),
        (True, torch.float, 3840, 4040),     # EXL3_GDN_PROJ_FP32
        (False, torch.bfloat16, 512, 611),   # EXL3_GDN_CONV_TOKEN_MAJOR=0: channel-major bf16, Triton
        (False, torch.bfloat16, 16, 30),     # channel-major bf16, the CUDA kernel (32 tokens or fewer)
    ],
    ids = ["tm_fp16_256", "tm_fp16_512", "tm_fp32_3840", "cm_bf16_512", "cm_bf16_cuda"],
)
@torch.inference_mode()
def test_the_conv_window_is_what_the_conv_s_state_write_leaves(device, token_major, in_dtype, end, n):
    dim, kc = 512, 4
    x, weight, bias, pool = _conv_case(device, token_major, in_dtype, dim, kc, n, seed = end)
    slots = torch.tensor([1], dtype = torch.int, device = device)
    head = x[:, :end] if token_major else x[:, :, :end].contiguous()
    short = pool.clone()
    causal_conv1d_update(head.contiguous(), short, slots, weight, bias, token_major = token_major)
    window = conv_window(x, end, kc, token_major, torch.empty((dim, kc), dtype = pool.dtype, device = device))
    torch.cuda.synchronize()
    assert torch.equal(window, short[1, :, :kc]), \
        f"window differs by {(window.float() - short[1, :, :kc].float()).abs().max().item():.3e}"
    # (the rest of the slot and the other slots are the conv's business, untouched by the copy)
    assert torch.equal(pool[0], short[0]) and torch.equal(pool[2], short[2])


def _ple(device, H, D, ple_dim, ksize, dilation, emb, seed):
    """A PLELayer with what forward_streams reads, without a config or an n-gram table: the embedding rows
    are given (the n-gram gather is per row; its rows do not depend on the pass)"""
    g_ = torch.Generator(device = "cpu").manual_seed(seed)
    r = lambda *s, scale = 1.0: (torch.randn(*s, generator = g_) * scale)

    def linear(n_in, n_out):
        inner = SimpleNamespace(weight = r(n_in, n_out, scale = n_in ** -0.5).half().to(device), bias = None)
        return SimpleNamespace(quant_type = "fp16", inner = inner)

    def norm():
        return SimpleNamespace(weight = SimpleNamespace(data = r(H * D, scale = 0.3).half().to(device)), rms_norm_eps = 1e-6)

    m = PLELayer.__new__(PLELayer)
    m.hc_mult, m.hidden_size = H, D
    m.conv_dilation = dilation
    m.conv_state_len = (ksize - 1) * dilation
    m.gate_scale = D ** -0.5
    m.key_proj = linear(ple_dim, H * D)
    m.value_proj = linear(ple_dim, D)
    m.norm_key, m.norm_query, m.norm_conv = norm(), norm(), norm()
    m.conv_w = r(H * D, 1, ksize, scale = 0.5).half().to(device).contiguous()
    ctx = 2
    m.ple_embedding = SimpleNamespace(forward = lambda hist, params: emb[:, : hist.shape[1] - ctx])
    return m, ctx


@pytest.mark.parametrize("with_state", [False, True], ids = ["fresh", "carried"])
@pytest.mark.parametrize("seq,split_at,sub", [(611, 512, 1024), (4040, 3840, 1024), (1300, 1024, 1024), (700, 256, 0)])
@torch.inference_mode()
def test_the_ple_conv_state_at_the_split_is_what_a_pass_ending_there_leaves(device, monkeypatch, with_state, seq,
                                                                              split_at, sub):
    monkeypatch.setattr(ple_mod, "_ple_sub_range", sub)
    H, D, ple_dim, ksize, dilation = 4, 256, 64, 4, 3
    g_ = torch.Generator(device = "cpu").manual_seed(seq + split_at)
    emb = torch.randn(1, seq, ple_dim, generator = g_).half().to(device)
    m, ctx = _ple(device, H, D, ple_dim, ksize, dilation, emb, seed = 7)
    streams = (torch.randn(1, seq, H, D, generator = g_) * 2.0).float().to(device)
    hist = torch.zeros((1, ctx + seq), dtype = torch.long)
    cs = (torch.randn(1, H * D, m.conv_state_len, generator = g_) * 0.3).half().to(device) if with_state else None

    short_streams = streams[:, :split_at].clone()
    _, short_cs = m.forward_streams(short_streams, hist[:, : ctx + split_at], {}, conv_state = cs)
    got = []
    full_streams = streams.clone()
    _, _ = m.forward_streams(full_streams, hist, {}, conv_state = cs, split_at = split_at, on_split = got.append)
    torch.cuda.synchronize()
    assert len(got) == 1
    want = short_cs[:, :, -m.conv_state_len:]
    assert got[0].shape == want.shape and got[0].dtype == want.dtype
    assert torch.equal(got[0], want), f"conv state differs by {(got[0].float() - want.float()).abs().max().item():.3e}"
    # The rows before the boundary went through the same slabs
    assert torch.equal(full_streams[:, :split_at], short_streams)
