#!/usr/bin/env python3
"""
compare_vs_production_fa.py — Step 2-C: position our kernel vs production Flash Attention.

Runs the SAME shapes (D=128, GQA, causal, seqLen sweep) through:
  1. our kernel        (torch.ops.amdinstinct.flash_attn, via the 2-A op)
  2. a production FA    (in priority order, whichever is importable):
        a. torch.nn.functional.scaled_dot_product_attention (PyTorch's fused FA backend
           on ROCm — the realistic "what vLLM-class stacks use" baseline, zero install)
        b. flash_attn (the ROCm flash-attention package, if installed)
        c. Composable Kernel FA (if a CK python binding is present)

and reports latency / achieved TFLOPS side by side.

FRAMING (per Paso2_comparacion_vias.md, via C): the value is NOT winning — production
FA has years of tuning, fp8 pipelines, and full MFMA scheduling. The value is measuring
the GAP and stating what of our DPP+DME finding is transferable to a production kernel.
We expect to be slower in absolute terms; the honest deliverable is "research kernel vs
production kernel, here is the gap and here is what's portable."

Run (on the VM, after building the 2-A op):
    python compare_vs_production_fa.py --seqs 512,1024,2048,4096

Scope: prefill, fp16, causal, GQA. Correctness is checked against SDPA on a small shape
before timing (same numbers within fp16 tol), so the comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


def load_our_op():
    try:
        import amdinstinct_fa  # noqa: F401
    except Exception:
        import glob
        sos = glob.glob("amdinstinct_fa*.so") + glob.glob("build/**/amdinstinct_fa*.so",
                                                           recursive=True)
        if not sos:
            print("ERROR: build the op first (python setup.py build_ext --inplace).",
                  file=sys.stderr)
            sys.exit(1)
        torch.ops.load_library(sos[0])


def pick_production_backend():
    """Return (name, fn(q,k,v)->o) for the best available production FA.

    q/k/v here are torch's standard [B, heads, seq, D] (we adapt our head-major op
    separately). SDPA is always present on torch-rocm and is the realistic baseline.
    """
    # b: explicit flash_attn package
    try:
        from flash_attn import flash_attn_func  # type: ignore

        def fa_pkg(q, k, v):
            # flash_attn expects [B, seq, heads, D]
            qa, ka, va = (t.transpose(1, 2).contiguous() for t in (q, k, v))
            o = flash_attn_func(qa, ka, va, causal=True)
            return o.transpose(1, 2)
        return ("flash_attn(pkg)", fa_pkg)
    except Exception:
        pass

    # a: SDPA fused backend (always available on torch-rocm)
    def sdpa(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return ("F.sdpa(fused)", sdpa)


def our_kernel(q_hm, k_hm, v_hm):
    """Our op on head-major [heads, seq, D] tensors."""
    return torch.ops.amdinstinct.flash_attn(q_hm, k_hm, v_hm)


def bench(fn, *tensors, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*tensors)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*tensors)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # µs/iter


def tflops(seq, nH, D, lat_us, causal=True):
    # 2 matmuls (QKᵀ and PV), each seq*seq*D MACs per head, 2 flops/MAC.
    work = 2.0 * 2.0 * nH * seq * seq * D
    if causal:
        work *= 0.5
    return work / (lat_us * 1e-6) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="512,1024,2048,4096")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--nH", type=int, default=32)
    ap.add_argument("--nKV", type=int, default=8)   # Llama-3 style GQA
    args = ap.parse_args()

    load_our_op()
    dev = "cuda"
    seqs = [int(s) for s in args.seqs.split(",")]
    D, nH, nKV = args.D, args.nH, args.nKV
    name, prod = pick_production_backend()

    # correctness check vs SDPA (full MHA, small seq) before timing
    s0 = 256
    q = torch.randn(nH, s0, D, device=dev, dtype=torch.float16)
    k = torch.randn(nH, s0, D, device=dev, dtype=torch.float16)  # MHA for the check
    v = torch.randn(nH, s0, D, device=dev, dtype=torch.float16)
    ours = our_kernel(q, k, v).float()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0).float(), k.unsqueeze(0).float(), v.unsqueeze(0).float(),
        is_causal=True).squeeze(0)
    rel = (ours - ref).abs().max().item() / (ref.abs().mean().item() + 1e-6)
    print(f"correctness vs SDPA (seq{s0}, MHA): rel={rel:.4f} "
          f"[{'OK' if rel < 0.05 else 'CHECK'}]\n")

    print(f"Comparison: amdinstinct.flash_attn vs {name}  (D={D}, GQA {nH}q/{nKV}kv, causal)")
    print(f"{'seq':>6} {'ours µs':>10} {'ours TFLOPS':>12} "
          f"{'prod µs':>10} {'prod TFLOPS':>12} {'ours/prod':>10}")
    for seq in seqs:
        qh = torch.randn(nH,  seq, D, device=dev, dtype=torch.float16)
        kh = torch.randn(nKV, seq, D, device=dev, dtype=torch.float16)
        vh = torch.randn(nKV, seq, D, device=dev, dtype=torch.float16)
        # production backend wants [B, heads, seq, D] with kv expanded for GQA
        qb = qh.unsqueeze(0)
        kb = kh.repeat_interleave(nH // nKV, dim=0).unsqueeze(0)
        vb = vh.repeat_interleave(nH // nKV, dim=0).unsqueeze(0)

        lat_ours = bench(our_kernel, qh, kh, vh)
        lat_prod = bench(prod, qb, kb, vb)
        tf_ours = tflops(seq, nH, D, lat_ours)
        tf_prod = tflops(seq, nH, D, lat_prod)
        print(f"{seq:>6} {lat_ours:>10.1f} {tf_ours:>12.2f} "
              f"{lat_prod:>10.1f} {tf_prod:>12.2f} {lat_ours/lat_prod:>9.2f}×")

    print("\nFRAMING: this is a research kernel vs a production kernel. A gap is expected")
    print("and is the point — it measures where DPP-softmax + DME-prefetch land relative")
    print("to a tuned MFMA pipeline, and what of that finding is portable upstream.")


if __name__ == "__main__":
    main()
