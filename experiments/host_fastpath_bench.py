"""
Host cost of the decode step's bookkeeping with EXL3_HOST_FASTPATH off and on (FORK.md, "The decode
step's host work, kept between calls"), on the CPU, without a GPU or the extension.

The sources are loaded twice as tests/test_host_fastpath.py loads them (the switch is read at
import), with the extension replaced by a function that does nothing, and the sites the switch
touches are driven with Qwen3.8-Flash-Next's sizes (4 streams of 2560, gated residual rank 320,
512 experts, top 10, 48 layers of which 36 GDN, MoE split on every layer). What is timed is the
Python and torch bookkeeping around the kernels only: a kernel launch here costs one empty call.

    python experiments/host_fastpath_bench.py [--rows 1,2,4,8] [--repeat 7]

Prints, per site and per pass of the model's sites at each row count, the time per call off and on.
"""

import argparse
import importlib.util
import os
import sys
import timeit
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

H, D, RANK, E, K, LAYERS, GDN_LAYERS = 4, 2560, 320, 512, 10, 48, 36


def load_test_module():
    spec = importlib.util.spec_from_file_location("host_fastpath_tests", os.path.join(REPO, "tests", "test_host_fastpath.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def site(world, key, use_combine = True):
    gr = world.hc.GatedResidual(None, key, H, D, 1e-6, use_combine = use_combine,
                                out_dtype = None if use_combine else torch.half)
    gr.norm_w_raw = torch.zeros(H * D).half()
    inject = torch.zeros(H, H * D).half() if use_combine else None
    gr._prepare(torch.zeros(RANK, H * D).half(), torch.zeros(H * D, RANK).half(), inject, keep_source_weights = True)
    return gr


def build(world, T):
    sites = []
    for i in range(LAYERS):
        sites.append((site(world, f"l{i}.attn"), site(world, f"l{i}.mlp")))
    mixer = site(world, "mixer", use_combine = False)
    world.hc.GatedResidual.link_sites([types.SimpleNamespace(attn_hc = a, mlp_hc = m) for a, m in sites] + [mixer])
    cfg = world.routing.RoutingCFG(
        gate_tensor = torch.zeros(D, E).half(), gate_tensor_t = None, num_experts = E, num_experts_per_tok = K,
        router_logits_bsz1 = torch.empty(1, E).half(), routing_weights_bsz1 = torch.empty(1, K).half(),
        selected_experts_bsz1 = torch.empty(1, K, dtype = torch.long), e_score_correction_bias = None,
        e_score_bias_h = None, routed_scaling_factor = None, n_group = None, topk_group = None, per_expert_scale = None)
    moes = [T.make_moe(world, D) for _ in range(LAYERS)]
    return sites, mixer, cfg, moes


def gdn_methods():
    T = sys.modules["host_fastpath_tests"]
    src = T._gdn_source()
    return T._method(src, "GDNLayerState", "staged_views"), T._method(src, "GDNLayerState", "staged_shape")


class GDNState:
    def __init__(self):
        self.max_history, self.max_batch_size = 3, 6
        rows = 6 * 4
        self.spec_conv = torch.zeros((rows, 8192), dtype = torch.bfloat16)
        self.spec_beta = torch.zeros((rows, 32), dtype = torch.bfloat16)
        self.spec_g = torch.zeros((rows, 32))
        self.spec_shape = None


def time_call(fn, repeat, number):
    return min(timeit.repeat(fn, number = number, repeat = repeat)) / number * 1e6   # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default = "1,2,4,8")
    ap.add_argument("--repeat", type = int, default = 7)
    args = ap.parse_args()
    T = load_test_module()
    sys.modules["host_fastpath_tests"] = T
    views, shape_only = gdn_methods()
    rows = [int(r) for r in args.rows.split(",")]
    torch.set_num_threads(1)
    results = {}
    with torch.inference_mode():
        for world in (T.OFF, T.ON):
            sites, mixer, cfg, moes = build(world, T)
            gtc = world.tensor.g_tensor_cache
            for R in rows:
                b, s = (1, R) if R <= 4 else (R // 2, 2)
                x = torch.zeros((b, s, H, D))
                yf = torch.zeros((b, s, D))
                z = torch.zeros((b * s, D)).half()
                st = GDNState()
                a0, m0 = sites[0]
                eshape = torch.Size((b, s, D))

                def one_site():
                    params = {}
                    post, _, y, _ = a0.mix_norm(x, params, None)
                    a0.apply_(x, yf, post, None, params)

                def routing():
                    world.routing.routing_std(b * s, cfg, z, {})

                def moe_tail():
                    T.moe_tail(world, moes[0], b * s, eshape, ("fused", None))

                def gdn():
                    (shape_only if world.fastpath else views)(st, b, s)

                def lane():
                    world.lane.current_lane()

                def bucketed():
                    gtc.get_bucketed(x.device, 1000, torch.float, "bench")

                def full_pass():
                    params = {}
                    for i, (a, m) in enumerate(sites):
                        post, _, y, _ = a.mix_norm(x, params, None)
                        if i < GDN_LAYERS:
                            (shape_only if world.fastpath else views)(st, b, s)
                        a.apply_(x, yf, post, None, params)
                        post, _, y, _ = m.mix_norm(x, params, None)
                        world.routing.routing_std(b * s, cfg, z, {})
                        T.moe_tail(world, moes[i], b * s, eshape, ("fused", None))
                        world.lane.current_lane()          # the CPU host's slot of the layer
                        m.apply_(x, yf, post, None, params)
                    mixer.forward(x, params)

                for name, fn, n in (("site (mix + apply)", one_site, 2000), ("routing_std", routing, 4000),
                                    ("MoE tail (routed sum + collect)", moe_tail, 4000), ("GDN record", gdn, 10000),
                                    ("current_lane", lane, 50000), ("get_bucketed", bucketed, 20000),
                                    ("pass: 96 sites, 48 MoE, 36 GDN", full_pass, 20)):
                    fn()
                    fn()
                    results[(name, R, world.fastpath)] = time_call(fn, args.repeat, n)
    names = []
    for (name, _, _) in results:
        if name not in names:
            names.append(name)
    print(f"Python {sys.version.split()[0]}, torch {torch.__version__}, one thread; us per call (min of {args.repeat})\n")
    print("| site | rows | off | on | saved |\n|---|---|---|---|---|")
    for name in names:
        for R in rows:
            a, b_ = results[(name, R, False)], results[(name, R, True)]
            print(f"| {name} | {R} | {a:.2f} | {b_:.2f} | {a - b_:.2f} ({100 * (a - b_) / a:.0f}%) |")


if __name__ == "__main__":
    main()
