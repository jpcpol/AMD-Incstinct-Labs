# flash-attention-mi300x

Flash Attention for AMD MI300X — incremental build exploiting DPP zero-LDS reductions
and DME async prefetch for FA3-style compute/memory pipelining on CDNA3.

## Problem

MI300X has 5.3 TB/s HBM3 bandwidth — 58% more than H100's 3.35 TB/s. The gap in
Flash Attention is not just bandwidth: the key insight from this project is that
**DPP reductions use zero LDS**, which relaxes the LDS pressure that limits FA tile
size. All current backends use ds_bpermute (LDS roundtrip) for row-softmax, which
steals LDS budget from K/V tiles.

## Build order (four independent steps)

Each step is separately benchmarked and validated before proceeding.

| Step | File | Status | What it adds |
| ---- | ---- | ------ | ------------ |
| 1 | `kernels/fa_naive.hip` | written | FA2 baseline: portable reductions, sync loads |
| 2 | `kernels/fa_dpp.hip` | written | DPP rowmax/rowsum + two-phase reduction for Bc=64/128/256 |
| 3 | `kernels/fa_dme.hip` | planned | Double-buffer K/V via DME async — requires dme-abstraction validation first |
| 4 | `kernels/fa_mfma.hip` | planned | MFMA intrinsics + tile-size tuning (ties into mlir-mfma-tiling) |

## Step 1: fa_naive (baseline)

**File:** `kernels/fa_naive.hip`  
**Run:** `bash scripts/run_fa_naive.sh` (from `amd-instinct-labs/` root)

Layout:
- One wave (64 threads) per query row
- `kBc=64`: K/V tile fits in one wave64 reduction exactly
- `kBr=1`: one query row per wave (simplest correct layout)
- Portable wave reductions (`__shfl_down` → ds_bpermute on CDNA3)
- Synchronous K/V loads from HBM (no DME)

Output:

- Correctness vs CPU reference (expected: max_rel_err < 1e-3)
- Median/P99 latency, TFLOPS achieved, HBM bandwidth
- rocprofv3 `SQ_INSTS_LDS` baseline (step 2 will reduce this to ~0 for reductions)

## The key argument (why DPP matters for FA, not just microbenchmarks)

```
LDS budget per CU = 64 KB
Tile storage: K_j (Bc×d) + V_j (Bc×d) + Q (Br×d)

With ds_bpermute reductions:
  LDS = tiles + reduction scratch → tiles must shrink
  → smaller Bc → more iterations → worse MFMA utilization

With DPP reductions (zero LDS):
  LDS = tiles only → maximum tile size
  → larger Bc → fewer iterations → better MFMA utilization
```

This is the argument that connects the Run 2 wave-primitives result to FA.
Step 2 measures this directly: fix the kernel, vary Bc, compare DPP vs portable.

## Dependencies

- `../../wave-primitives/include/wave_primitives/wave_reduce_dpp.hpp` — validated Run 2
- `../../dme-abstraction/include/dme/dme_copy.hpp` — step 3 only, needs Run 4 validation
- MFMA via `__builtin_amdgcn_mfma_*` — step 4 only

## Gap Reference

[Gap 2 — Flash Attention](../../docs/gap-analysis.md#2-flash-attention--dme-and-fa3-style-pipelining-missing)
