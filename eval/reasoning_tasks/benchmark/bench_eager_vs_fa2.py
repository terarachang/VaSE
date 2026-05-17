"""
Per-layer attention benchmark at decode-step shapes: eager vs flash_attention_2.

Toggles between MHA (default, H_kv=None) and GQA (H_kv specified, eager pays
repeat_kv to expand K/V from H_kv -> H_q heads). FA2 handles GQA natively in
the kernel so it never pays repeat_kv.

Default MHA finding: eager beats FA2 at decode (FA2's tile kernel is degenerate
for q_len=1). GQA flips the verdict because eager's repeat_kv materialization
dominates.
"""
import argparse
from statistics import mean
from typing import Callable

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func


device = 'cuda'
dtype = torch.bfloat16


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def eager_decode(q, k, v, group=1):
    k = repeat_kv(k, group)
    v = repeat_kv(v, group)
    D = q.size(-1)
    aw = torch.matmul(q, k.transpose(2, 3)) * (D ** -0.5)
    aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(aw, v)


def fa2_from_hf_layout(q_bhsd, k_bhsd, v_bhsd):
    # Realistic HF flow: cache stored as (B, H, S, D); FA2 needs (B, S, H, D).
    # maybe_contiguous inside the kernel skips the copy because the last dim is
    # still stride-1 after transpose(1, 2).
    q = q_bhsd.transpose(1, 2)
    k = k_bhsd.transpose(1, 2)
    v = v_bhsd.transpose(1, 2)
    return flash_attn_func(q, k, v, causal=True)


def fa2_native(q_bshd, k_bshd, v_bshd):
    return flash_attn_func(q_bshd, k_bshd, v_bshd, causal=True)


def benchmark(run: Callable, num_warmups: int = 3, num_trials: int = 20) -> float:
    """Benchmark `func` by running it `num_trials`.  Return the average time."""
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()
    times: list[float] = []
    for trial in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        run()
        end_event.record()
        torch.cuda.synchronize()
        times.append((start_event.elapsed_time(end_event)))
    return mean(times)


def make_qkv(args, kv_len):
    q = torch.randn(args.B, args.H_q, 1, args.D, dtype=dtype, device=device)
    h_kv = args.H_kv if args.gqa else args.H_q
    k = torch.randn(args.B, h_kv, kv_len, args.D, dtype=dtype, device=device)
    v = torch.randn(args.B, h_kv, kv_len, args.D, dtype=dtype, device=device)
    return q, k, v


def sweep(args):
    mode = f"GQA (H_q={args.H_q}, H_kv={args.H_kv})" if args.gqa else f"MHA (H={args.H_q})"
    group = args.H_q // args.H_kv if args.gqa else 1
    print(f"Decode-step attention: B={args.B}, {mode}, D={args.D}")
    print(f"{'kv_len':>7} | {'eager (ms)':>11} | {'fa2 (ms)':>9} | {'speedup':>7}")
    print("-" * 50)
    for kv_len in args.kv_lens:
        q, k, v = make_qkv(args, kv_len)
        eager = benchmark(lambda: eager_decode(q, k, v, group))
        fa2 = benchmark(lambda: fa2_from_hf_layout(q, k, v))
        print(f"{kv_len:>7} | {eager:>11.3f} | {fa2:>9.3f} | {eager/fa2:>6.2f}x")


def breakdown(args):
    kv_len = args.breakdown_kv
    group = args.H_q // args.H_kv if args.gqa else 1
    print(f"\nBreakdown at kv={kv_len}:")
    q_bhsd, k_bhsd, v_bhsd = make_qkv(args, kv_len)
    q_bshd = q_bhsd.transpose(1, 2).contiguous()
    k_bshd = k_bhsd.transpose(1, 2).contiguous()
    v_bshd = v_bhsd.transpose(1, 2).contiguous()

    print(f"  FA2 from HF layout (BHSD input):  "
          f"{benchmark(lambda: fa2_from_hf_layout(q_bhsd, k_bhsd, v_bhsd)):.3f} ms")
    print(f"  FA2 from native BSHD layout:      "
          f"{benchmark(lambda: fa2_native(q_bshd, k_bshd, v_bshd)):.3f} ms")

    # Eager broken into its kernel launches (including repeat_kv if GQA).
    print(f"\nEager kernel-by-kernel:")
    ms_rk_k = ms_rk_v = 0.0
    if args.gqa:
        ms_rk_k = benchmark(lambda: repeat_kv(k_bhsd, group))
        ms_rk_v = benchmark(lambda: repeat_kv(v_bhsd, group))
    k_full = repeat_kv(k_bhsd, group)
    v_full = repeat_kv(v_bhsd, group)
    scale = args.D ** -0.5
    ms_qk = benchmark(lambda: torch.matmul(q_bhsd, k_full.transpose(2, 3)) * scale)
    aw = torch.matmul(q_bhsd, k_full.transpose(2, 3)) * scale
    ms_sm = benchmark(lambda: F.softmax(aw, dim=-1, dtype=torch.float32).to(q_bhsd.dtype))
    aw_sm = F.softmax(aw, dim=-1, dtype=torch.float32).to(q_bhsd.dtype)
    ms_av = benchmark(lambda: torch.matmul(aw_sm, v_full))
    if args.gqa:
        print(f"  repeat_kv(K):            {ms_rk_k:.3f} ms")
        print(f"  repeat_kv(V):            {ms_rk_v:.3f} ms")
    print(f"  Q @ K^T (+scale):        {ms_qk:.3f} ms")
    print(f"  softmax (fp32 cast):     {ms_sm:.3f} ms")
    print(f"  attn @ V:                {ms_av:.3f} ms")
    print(f"  sum: {ms_rk_k + ms_rk_v + ms_qk + ms_sm + ms_av:.3f} ms")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=8, help="batch size")
    p.add_argument("--H_q", type=int, default=32, help="number of query heads")
    p.add_argument("--H_kv", type=int, default=None,
                   help="number of KV heads. If set, enables GQA mode (eager calls repeat_kv).")
    p.add_argument("--D", type=int, default=128, help="head dim")
    p.add_argument("--kv_lens", type=int, nargs='+',
                   default=[1024, 2048, 4096, 8192, 16384])
    p.add_argument("--breakdown_kv", type=int, default=16384)
    args = p.parse_args()
    args.gqa = args.H_kv is not None
    if args.gqa:
        assert args.H_q % args.H_kv == 0, "H_q must be divisible by H_kv for GQA"
    return args


if __name__ == "__main__":
    args = parse_args()
    sweep(args)
    breakdown(args)
