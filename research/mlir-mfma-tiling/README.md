# mlir-mfma-tiling

MLIR pass for automatic MFMA tile-size selection on AMD Instinct MI300X.

## Problem

The MLIR AMDGPU dialect supports `linalg.matmul → amdgpu.mfma` lowering, but tile size selection is entirely manual. There is no equivalent to:

- cuBLAS internal auto-tuning
- CUTLASS v3 automatic kernel selection
- Triton's auto-tuning (which requires explicit config grids)

For each GEMM shape (M, N, K), a developer must manually choose:
- MFMA variant (e.g., `mfma.f32.32x32x8.bf16` vs `mfma.f32.16x16x32.fp8.fp8`)
- Thread tile sizes (how many MFMA outputs per thread)
- Block tile sizes (how much of M/N/K per CTA)
- LDS buffer layout to avoid bank conflicts
- Number of pipeline stages

## Proposed MLIR Pass

A new pass `--amdgpu-mfma-tile-select` that operates on `linalg.matmul` (and `linalg.batch_matmul`) ops:

```
Input:  linalg.matmul ins(%A: memref<MxKxbf16>, %B: memref<KxNxbf16>) outs(%C: memref<MxNxf32>)
Output: Tiled linalg.matmul with:
          - Optimal MFMA variant selected
          - Block tiles matched to L2 (32 MB on MI300X)
          - Thread tiles at max register utilization without spilling
          - LDS layout with zero bank conflicts
```

### Selection Logic

```
Given (M, N, K, dtype):
1. Select MFMA variant:
   - FP8 input → mfma.f32.32x32x16.fp8 (highest throughput)
   - BF16 input → mfma.f32.32x32x8.bf16
   - FP16 input → mfma.f32.32x32x8.f16
   - FP64 input → mfma.f64.16x16x4.f64

2. Compute max tile for L2:
   tile_size = sqrt(L2_size / dtype_size / 2)  // 2 for A+B

3. Estimate register usage; reduce tile if occupancy < 2 waves/CU

4. Select LDS swizzle pattern to avoid 32-bank conflicts
```

## Files

```
lib/
    AMDGPUMFMATileSelect.cpp   # Main MLIR pass implementation
    AMDGPUMFMATileSelect.h
    CMakeLists.txt
tests/
    tile_select_bf16.mlir      # Test: BF16 GEMM tiling
    tile_select_fp8.mlir       # Test: FP8 GEMM tiling
    tile_select_large.mlir     # Test: Large M/N/K (LLM shapes)
benchmarks/
    bench_gemm.py              # vs rocBLAS on MI300X GEMM shapes
```

## Upstream Target

This pass is a candidate for direct contribution to LLVM/MLIR upstream in the AMDGPU dialect. AMD's MLIR infrastructure team maintains the `amdgpu` dialect — the relevant module is `mlir/lib/Dialect/AMDGPU/`.

## Status

- [ ] Research: survey existing tile selection heuristics in Triton and IREE
- [ ] Prototype: rule-based selection for BF16 GEMM
- [ ] Extend to FP8 and INT8 variants
- [ ] Autotuning fallback for edge-case shapes
- [ ] Benchmark suite: standard LLM GEMM shapes (batch=1..8, M=1..8192, N/K=4096..16384)
- [ ] Upstream proposal to MLIR

## Gap Reference

[Gap 6 — MLIR MFMA Auto-Tiling](../../docs/gap-analysis.md#6-mlir-mfma-auto-tiling--open)
