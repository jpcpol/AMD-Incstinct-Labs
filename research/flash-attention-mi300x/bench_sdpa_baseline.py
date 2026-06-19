#!/usr/bin/env python3
"""
bench_sdpa_baseline.py — Phase-1: fresh F.sdpa latency for gap comparison.

Measures torch.nn.functional.scaled_dot_product_attention (fused ROCm backend)
on the same shapes as bench_multiwave_vs_sdpa.hip so the gap can be computed
offline: gap = multiwave_us / sdpa_us.

This is the "production side" of H-MW3. Run on the VM after running the HIP
benchmark, then paste both tables into the results file.

Run:
    python bench_sdpa_baseline.py
    python bench_sdpa_baseline.py --seqs 512,1024,2048,4096 --D 128 --nH 32 --nKV 8
"""
from __future__ import annotations

import argparse
import time

import torch


def bench_sdpa(q, k, v, warmup: int = 20, iters: int = 100) -> float:
    """Return median latency in µs."""
    fn = torch.nn.functional.scaled_dot_product_attention
    for _ in range(warmup):
        fn(q, k, v, is_causal=True)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    return times[len(times) // 2]


def tflops(seq: int, nH: int, D: int, us: float) -> float:
    # causal: 0.5× work, 2 matmuls, 2 flops/MAC
    work = 0.5 * 2.0 * 2.0 * nH * seq * seq * D
    return work / (us * 1e-6) / 1e12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs",  default="512,1024,2048,4096")
    ap.add_argument("--D",     type=int, default=128)
    ap.add_argument("--nH",    type=int, default=32)
    ap.add_argument("--nKV",   type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    seqs  = [int(s) for s in args.seqs.split(",")]
    D, nH, nKV, B = args.D, args.nH, args.nKV, args.batch
    device = "cuda"

    props = torch.cuda.get_device_properties(0)
    print(f"Device : {props.name}")
    print(f"Config : GQA {nH}q/{nKV}kv, D={D}, causal=True, batch={B}")
    print(f"Backend: F.sdpa (fused ROCm)\n")

    # correctness smoke: compare SDPA vs fp32 reference on small shape
    s0 = 256
    q0 = torch.randn(B, nH,  s0, D, device=device, dtype=torch.float16)
    k0 = torch.randn(B, nKV, s0, D, device=device, dtype=torch.float16)
    v0 = torch.randn(B, nKV, s0, D, device=device, dtype=torch.float16)
    # expand KV for GQA
    k0e = k0.repeat_interleave(nH // nKV, dim=1)
    v0e = v0.repeat_interleave(nH // nKV, dim=1)
    out_f16 = torch.nn.functional.scaled_dot_product_attention(
        q0, k0e, v0e, is_causal=True)
    out_f32 = torch.nn.functional.scaled_dot_product_attention(
        q0.float(), k0e.float(), v0e.float(), is_causal=True)
    rel = (out_f16.float() - out_f32).abs().max().item() / (out_f32.abs().mean().item() + 1e-6)
    print(f"Correctness smoke (seq={s0}, fp16 vs fp32): rel={rel:.4f} "
          f"[{'OK' if rel < 0.05 else 'CHECK'}]\n")

    print(f"{'seq':>6} {'sdpa µs':>10} {'sdpa TFLOPS':>12}")
    print(f"{'---':>6} {'-------':>10} {'-----------':>12}")

    results = []
    for seq in seqs:
        q = torch.randn(B, nH,  seq, D, device=device, dtype=torch.float16)
        k = torch.randn(B, nKV, seq, D, device=device, dtype=torch.float16)
        v = torch.randn(B, nKV, seq, D, device=device, dtype=torch.float16)
        ke = k.repeat_interleave(nH // nKV, dim=1)
        ve = v.repeat_interleave(nH // nKV, dim=1)
        us = bench_sdpa(q, ke, ve)
        tf = tflops(seq, nH, D, us)
        print(f"{seq:>6} {us:>10.1f} {tf:>12.2f}")
        results.append((seq, us, tf))

    # Print in copy-paste format for results file
    print("\n--- paste into results file ---")
    print(f"# F.sdpa baseline — D={D}, GQA {nH}q/{nKV}kv, causal, batch={B}")
    for seq, us, tf in results:
        print(f"  sdpa seq={seq}: {us:.1f} µs  {tf:.2f} TFLOPS")

    # Gap table using 2-C multiwave numbers as placeholder
    # (replace with actual bench_multiwave_vs_sdpa.hip output)
    print("\n--- gap table template ---")
    print("Copy multiwave W=1 D=128 numbers from bench_multiwave_vs_sdpa output,")
    print("then compute gap = multiwave_us / sdpa_us:\n")
    print(f"{'seq':>6} {'multiwave µs':>14} {'sdpa µs':>10} {'gap (×)':>10}")
    print(f"{'---':>6} {'------------':>14} {'-------':>10} {'-------':>10}")
    for seq, sdpa_us, _ in results:
        print(f"{seq:>6} {'<from HIP>':>14} {sdpa_us:>10.1f} {'<mw/sdpa>':>10}")


if __name__ == "__main__":
    main()
