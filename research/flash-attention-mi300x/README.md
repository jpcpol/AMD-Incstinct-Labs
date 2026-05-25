# flash-attention-mi300x

Flash Attention for AMD MI300X using DME async prefetch and FA3-style compute/memory pipelining.

## Problem

MI300X has 5.3 TB/s HBM3 bandwidth — 58% more than H100's 3.35 TB/s. Yet no Flash Attention implementation exploits this:

- The Data Movement Engine (DME) can async-load HBM→LDS without occupying CUs, but all current backends ignore it
- FA3-style pipelining (overlap compute with memory) is not implemented for AMD
- The best current option (FA2 Triton FP8) crashes the compiler when autotune is also enabled

## Approach

### Stage 1: Fix the autotune + FP8 compiler bug

The bug: enabling `TRITON_FA_AUTOTUNE=1` with FP8 dtype causes `hipcc` internal error. Fix upstream in triton-lang ROCm backend.

### Stage 2: DME-pipelined attention

Use `amdgpu.global_load_async_to_lds` (MLIR) / `__builtin_amdgcn_global_load_lds` to issue async HBM→LDS loads for Q, K, V tiles while MFMA units process the previous tile.

```
Cycle N:    MFMA(Q[i] × K[i])     +    DME_load(Q[i+1], K[i+1])
Cycle N+1:  MFMA(Q[i+1] × K[i+1]) +   DME_load(V[i+1])
Cycle N+2:  MFMA(scores × V[i])   +    DME_load(next block)
```

### Stage 3: FA3-style warp specialization

Split warps into producer (load) and consumer (MFMA) groups, using LDS as a ping-pong buffer. The DME handles HBM→LDS, consumer warps never stall on memory.

## Benchmarks

Target: exceed FA2 Triton on MI300X by exploiting the HBM bandwidth advantage.

| Backend | BW utilization | TFLOPS | VRAM savings |
| --- | --- | --- | --- |
| PyTorch SDPA (baseline) | ~30% HBM | — | 0% |
| FA2 Triton FP8 (current best) | ~55% HBM | — | ~24% |
| DME-pipelined (this work) | target: >80% HBM | TBD | TBD |

## Files

```
include/
    flash_attention_dme.hpp    # C++ interface
src/
    fa_dme_kernel.hip          # HIP kernel with DME intrinsics
    fa_triton_fix.py           # Triton autotune+FP8 patch
benchmarks/
    bench_fa.py                # Comparison vs FA2 Triton / PyTorch SDPA
```

## Status

- [ ] Reproduce autotune+FP8 crash, isolate minimal repro
- [ ] DME async load prototype for single attention block
- [ ] Full pipelined attention kernel
- [ ] Benchmarks vs FA2 Triton on NanoGPT workload
- [ ] Upstream: fix to triton-lang ROCm backend

## Gap Reference

[Gap 2 — Flash Attention](../../docs/gap-analysis.md#2-flash-attention--dme-and-fa3-style-pipelining-missing)
