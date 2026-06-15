# hipCUB upstream issue — DPP gap on CDNA3

Target: https://github.com/ROCm/hipCUB/issues/new

## Title

[CDNA3/gfx942] WarpReduce::Sum uses ds_bpermute (LDS) instead of DPP — 1.79× slower than native DPP path

## Body

### Summary

On AMD Instinct MI300X (CDNA3, gfx942, ROCm 7.2), `hipCUB::WarpReduce<float>::Sum()`
generates **6× `ds_bpermute`** instructions — an LDS round-trip per reduction step —
instead of using the DPP (Data-Parallel Primitives) cross-lane datapath available on CDNA3.

A full-DPP implementation using `__builtin_amdgcn_update_dpp` with `row_shr` + `row_bcast`
control codes eliminates all LDS traffic and is **1.79× faster than hipCUB** on the same hardware.

### Measured Results

Hardware: MI300X VF (gfx942), ROCm 7.2, hipCUB 4.2.0, kReps=4096

| Implementation | Median (µs) | vs hipCUB | SQ_INSTS_LDS |
|---|---|---|---|
| `wave::dpp::reduce_sum` (full DPP, lane-63 result) | **210** | **1.79×** | **0** |
| `wave::dpp::reduce_sum` (full DPP, broadcast) | **279** | **1.35×** | **0** |
| `hipCUB::WarpReduce<float>::Sum()` | 377 | 1.00× | 4.2M |
| portable `__shfl_down` reduce | 922 | 0.41× | 25.2M |

`SQ_INSTS_LDS` measured with rocprofv3 (`SQ_INSTS_LDS` hardware counter, kReps=4096 iterations).

### Root Cause

On CDNA3, `__shfl_down_sync` lowers to `ds_bpermute_b32` (requires LDS), not to a DPP
instruction. Confirmed by inspecting generated ISA with `--save-temps`.

The DPP alternative uses only `v_add_f32_dpp` VALU instructions:

```
row_shr:1  (0x111) — combines lane (i-1) within each 16-lane row
row_shr:2  (0x112)
row_shr:4  (0x114)
row_shr:8  (0x118) — per-row result lands on lanes 15/31/47/63
row_bcast15 (0x142) — propagate across 16-lane rows
row_bcast31 (0x143) — propagate across 32-lane bank boundary
```

Result lands on lane 63. Broadcast to all lanes via `v_readlane_b32` (optional).

### Reference Implementation

Header-only, correctness PASS 12/12 on MI300X (f32/f64/i32/i64):

https://github.com/jpcpol/AMD-Incstinct-Labs/blob/main/research/wave-primitives/include/wave_primitives/wave_reduce_dpp.hpp

Benchmark: `research/wave-primitives/benchmarks/bench_reduce_dpp.hip`
rocprofv3 data: `research/wave-primitives/results/2026-06-03_rocprofv3_lds_mi300x.txt`

### Suggestion

`hipCUB::WarpReduce` and `hipCUB::WarpScan` could use `__builtin_amdgcn_update_dpp`
on CDNA3 targets (`__gfx9__` or explicit `gfx942`) to eliminate the LDS round-trip.

The scan DPP implementation achieves `SQ_INSTS_LDS = 0` and matches hipCUB WarpScan
within measurement noise (1.028×); the reduce path outperforms it by 1.35–1.79×.

Note: the DPP advantage is **wave-scoped only**. Block-level reductions (CUB BlockReduce
equivalent) still require one LDS step for inter-wave communication — both hipCUB and a
hand-tuned two-phase DPP approach measure identically at that scope (1.000×).
