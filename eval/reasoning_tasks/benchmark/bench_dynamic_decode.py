"""
Dynamic-decoding benchmark: end-to-end decode latency from prompt_len to
max_seqlen with q_len=1 per step.

- eager and flash_attn_func use HF's DynamicCache to grow the KV cache via
  torch.cat each step (the realistic HF inference path).
- flash_attn_with_kvcache uses a preallocated [B, max_seqlen, H_kv, D] buffer
  and writes new K/V in-kernel.

Qwen3 GQA shape (H_q=40, H_kv=8, D=128) by default. Speed only — no numerical
comparison.
"""
import argparse
from statistics import mean
from typing import Callable

import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import repeat_kv


device = 'cuda'
dtype = torch.bfloat16


def benchmark(run: Callable, num_warmups: int = 1, num_trials: int = 3) -> float:
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(num_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return mean(times)


def eager_attention(q_h, k_h, v_h, n_rep):
    # All inputs [B, H, S, D]. q_len=1 with causal attends to all of k.
    D = q_h.shape[-1]
    k_h = repeat_kv(k_h, n_rep)
    v_h = repeat_kv(v_h, n_rep)
    scale = 1.0 / (D ** 0.5)
    attn = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q_h.dtype)
    return torch.matmul(attn, v_h)


def run_eager(args, num_steps, n_rep):
    cache = DynamicCache()
    k_prompt = torch.randn(args.B, args.H_kv, args.prompt_len, args.D, dtype=dtype, device=device)
    v_prompt = torch.randn(args.B, args.H_kv, args.prompt_len, args.D, dtype=dtype, device=device)
    cache.update(k_prompt, v_prompt, layer_idx=0)

    q_new = torch.randn(args.B, args.H_q, 1, args.D, dtype=dtype, device=device)
    k_new = torch.randn(args.B, args.H_kv, 1, args.D, dtype=dtype, device=device)
    v_new = torch.randn(args.B, args.H_kv, 1, args.D, dtype=dtype, device=device)

    for _ in range(num_steps):
        k_full, v_full = cache.update(k_new, v_new, layer_idx=0)
        eager_attention(q_new, k_full, v_full, n_rep)


def run_func(args, num_steps):
    cache = DynamicCache()
    k_prompt = torch.randn(args.B, args.H_kv, args.prompt_len, args.D, dtype=dtype, device=device)
    v_prompt = torch.randn(args.B, args.H_kv, args.prompt_len, args.D, dtype=dtype, device=device)
    cache.update(k_prompt, v_prompt, layer_idx=0)

    # flash_attn_func expects [B, S, H, D]; DynamicCache stores [B, H, S, D],
    # so we transpose after .update each step (this matches what HF does in
    # _flash_attention_forward).
    q_new = torch.randn(args.B, 1, args.H_q, args.D, dtype=dtype, device=device)
    k_new = torch.randn(args.B, args.H_kv, 1, args.D, dtype=dtype, device=device)
    v_new = torch.randn(args.B, args.H_kv, 1, args.D, dtype=dtype, device=device)

    for _ in range(num_steps):
        k_full, v_full = cache.update(k_new, v_new, layer_idx=0)  # [B, H_kv, S, D]
        flash_attn_func(q_new, k_full.transpose(1, 2), v_full.transpose(1, 2), causal=True)


def run_kvcache(args, num_steps):
    max_len = args.prompt_len + num_steps
    k_cache = torch.zeros(args.B, max_len, args.H_kv, args.D, dtype=dtype, device=device)
    v_cache = torch.zeros(args.B, max_len, args.H_kv, args.D, dtype=dtype, device=device)
    k_cache[:, :args.prompt_len] = torch.randn(
        args.B, args.prompt_len, args.H_kv, args.D, dtype=dtype, device=device)
    v_cache[:, :args.prompt_len] = torch.randn(
        args.B, args.prompt_len, args.H_kv, args.D, dtype=dtype, device=device)

    cache_seqlens = torch.full((args.B,), args.prompt_len, dtype=torch.int32, device=device)
    q_new = torch.randn(args.B, 1, args.H_q, args.D, dtype=dtype, device=device)
    k_new = torch.randn(args.B, 1, args.H_kv, args.D, dtype=dtype, device=device)
    v_new = torch.randn(args.B, 1, args.H_kv, args.D, dtype=dtype, device=device)

    for _ in range(num_steps):
        flash_attn_with_kvcache(
            q_new, k_cache, v_cache,
            k=k_new, v=v_new,
            cache_seqlens=cache_seqlens,
            causal=True,
        )
        cache_seqlens.add_(1)


def main():
    args = parse_args()
    num_steps = args.max_seqlen - args.prompt_len
    n_rep = args.H_q // args.H_kv
    print(f"Dynamic decode (q_len=1, causal=True): B={args.B}, "
          f"H_q={args.H_q}, H_kv={args.H_kv}, D={args.D}, dtype={dtype}")
    print(f"prompt_len={args.prompt_len}, max_seqlen={args.max_seqlen}, "
          f"decode_steps={num_steps}")
    print("-" * 80)

    t_eager = benchmark(lambda: run_eager(args, num_steps, n_rep))
    t_func = benchmark(lambda: run_func(args, num_steps))
    t_kvc = benchmark(lambda: run_kvcache(args, num_steps))

    print(f"{'kernel':<40} {'total (ms)':>12} {'per step (us)':>15} {'vs kvc':>10}")
    print("-" * 80)
    for name, t in [
        ("eager + DynamicCache", t_eager),
        ("flash_attn_func + DynamicCache", t_func),
        ("flash_attn_with_kvcache (preallocated)", t_kvc),
    ]:
        print(f"{name:<40} {t:>12.2f} {t * 1000 / num_steps:>15.2f} {t / t_kvc:>9.2f}x")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=8, help="batch size")
    p.add_argument("--H_q", type=int, default=40, help="query heads (Qwen3-14B: 40)")
    p.add_argument("--H_kv", type=int, default=8, help="kv heads (Qwen3-14B: 8)")
    p.add_argument("--D", type=int, default=128, help="head dim")
    p.add_argument("--prompt_len", type=int, default=256)
    p.add_argument("--max_seqlen", type=int, default=4096)
    args = p.parse_args()
    assert args.H_q % args.H_kv == 0
    return args


if __name__ == "__main__":
    main()
