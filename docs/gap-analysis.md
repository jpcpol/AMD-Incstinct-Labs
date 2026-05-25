# Gap Analysis: ROCm vs CUDA

> **Status per gap**: verified against AMD primary sources  
> **Last updated**: 2026-05-25  
> **Hardware target**: AMD Instinct MI300X (gfx942), ROCm 6.2+

---

## Summary

| Gap | Status | Our Work |
| --- | --- | --- |
| Dynamic Parallelism | ❌ Closed "not planned" — no roadmap | Planned: host-side dispatch framework |
| Flash Attention DME | 🟡 FA2 works, DME unused, FA3-style absent | [flash-attention-mi300x](../research/flash-attention-mi300x/) |
| FP8 Quantization | ✅ Resolved in ROCm 6.2 | No action needed |
| Warp-agnostic primitives | 🟡 hipCUB exists but is thin wrapper | [wave-primitives](../research/wave-primitives/) |
| RCCL / Infinity Fabric | 🟡 xGMI optimized, SDMA bottleneck open | [infinity-fabric-allreduce](../research/infinity-fabric-allreduce/) |
| MLIR MFMA auto-tiling | ❌ Manual tile selection, no auto-tuning | [mlir-mfma-tiling](../research/mlir-mfma-tiling/) |
| DME abstraction | ❌ MLIR low-level only, no high-level API | [dme-abstraction](../research/dme-abstraction/) |
| MI300A unified memory | ❌ No frameworks exploit shared CPU+GPU HBM | Planned |
| Profiling tooling | 🟡 rocprofv3 exists, no integrated visualizer | Low priority |

---

## 1. Dynamic Parallelism — No Roadmap

**Primary source**: [GitHub Issue #3511](https://github.com/ROCm/HIP/issues/3511) — closed "not planned", June 2024.

HIP does not support device-side kernel launches. The feature request was silently discarded with no milestone or assignee.

**Affected workloads**: GNNs with variable depth, Adaptive Mesh Refinement, BVH traversal, adaptive sort algorithms.

**Proposed approach**: Host-side dispatch framework using HIP streams and dependency graphs that provides a device-transparent API:

```cpp
hip::dynamic_dispatch([&](auto& ctx) {
    if (ctx.needs_subdivision()) {
        ctx.launch_child(refine_kernel, ...);
    }
});
```

---

## 2. Flash Attention — DME and FA3-Style Pipelining Missing

**Sources**: ZD Tech Substack — Flash Attention on ROCm; AMD ROCm Blog — Flash Attention MI300X.

Five implementations available on MI300X. Best: FA2 Triton FP8 — but compiler crashes when autotune and FP8 are both enabled simultaneously.

**Open gaps**:

1. **autotune + FP8 crash** — must choose one or the other
2. **DME not used** — `amdgpu.global_load_async_to_lds` not leveraged in any implementation
3. **FA3-style compute/memory pipelining** not implemented for AMD
4. **ALiBi** not supported in all backends

**Opportunity**: MI300X has 5.3 TB/s HBM3 vs 3.35 TB/s on H100. Overlapping Q/K/V HBM loads with MFMA compute via DME should yield significant gains not achievable on NVIDIA hardware.

See [`research/flash-attention-mi300x/`](../research/flash-attention-mi300x/) for implementation work.

---

## 3. FP8 Quantization — RESOLVED

**Source**: AMD ROCm 6.2 release (2024), bitsandbytes ROCm blog.

ROCm 6.2 shipped FP8 GEMM in PyTorch and JAX via hipBLASLt, official bitsandbytes support, and standardized FP8 headers. No work needed here.

---

## 4. Warp-Agnostic Primitives — Open

**Source**: hipCUB documentation; gap confirmed by absence of wave-aware library.

hipCUB wraps rocPRIM but does not resolve the fundamental problem: code that assumes `warpSize=32` produces incorrect results on wave64 without manual changes.

```cpp
// CUDA — correct
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);

// HIP wave64 — wrong: loop starts at 16 but wavefront has 64 lanes
// Correct starting offset is 32
```

**Proposed**: `wave-primitives` — zero-overhead header-only library that detects wave size at compile time and provides correct `warp_reduce`, `warp_scan`, `warp_ballot`, `warp_shuffle` for both wave32 and wave64, with an API matching CUB/CUDA for easy migration.

See [`research/wave-primitives/`](../research/wave-primitives/).

---

## 5. RCCL / Infinity Fabric — SDMA Bottleneck

**Sources**: AMD RCCL xGMI blog; arXiv 2410.00801 — Infinity Fabric data movement analysis.

RCCL achieves 310–330 GB/s aggregate on 8× MI300X (theoretical 336 GB/s). Identified bottlenecks:

1. **SDMA engines** tuned for PCIe-4.0, not Infinity Fabric GPU-GPU bandwidth. Workaround: `HSA_ENABLE_PEER_SDMA=0`
2. **Slowest link bottleneck**: in full-mesh topology, the slowest pair (45.21 GB/s) caps the AllReduce
3. **Ring/Tree algorithms** inherited from NCCL — not designed for 7-link full-mesh topology

**Proposed**: Topology-aware AllReduce that reduces on fast local links first, then broadcasts across all 7 simultaneous links.

See [`research/infinity-fabric-allreduce/`](../research/infinity-fabric-allreduce/).

---

## 6. MLIR MFMA Auto-Tiling — Open

**Evidence**: MLIR AMDGPU dialect — `linalg.matmul → amdgpu.mfma` lowering is possible but tile size selection is manual. No cuBLAS-equivalent auto-tuning exists.

**Proposed**: MLIR pass that analyzes GEMM shapes (M, N, K), selects the optimal MFMA variant (32×32×8 BF16, 16×16×32 FP8, etc.), and determines tile sizes for L1/L2 cache on MI300X. Direct upstream contribution to MLIR.

See [`research/mlir-mfma-tiling/`](../research/mlir-mfma-tiling/).

---

## 7. DME Abstraction — Open, Exclusive to CDNA3

**Evidence**: `amdgpu.tensor_load_to_lds` / `amdgpu.tensor_store_from_lds` exist in MLIR but with no high-level API. The DME accelerates multi-dimensional tensor copies HBM→LDS with arbitrary layouts. No NVIDIA equivalent.

**Proposed**: C++ library exposing DME as a usable API:

```cpp
dme::tensor_prefetch(src_hbm, dst_lds, {batch, heads, seq_len, head_dim});
```

See [`research/dme-abstraction/`](../research/dme-abstraction/).

---

## 8. MI300A Unified Memory — Open, Hardware Unique

128 GB HBM3 shared between Zen 4 CPU and GPU with no copy. No framework targets this programming model.

**Proposed**: First library defining zero-copy CPU↔GPU programming patterns — adaptive algorithms that select CPU or GPU execution at runtime based on cache state.

---

## AMD Contribution Path

For research-grade contributions, AMD's process is:

1. Publish preprint on arXiv with benchmarks
2. Implement in a public repo (this repo)
3. Open GitHub Discussion in the relevant AMD repo (MIOpen, composable_kernel, RCCL, triton-lang)
4. Submit PR to `develop` branch with design document + before/after metrics

For Composable Kernel (high-performance kernels): PRs < 1,000 lines, design doc required for major changes, `clang-format` enforced.

- GitHub Discussions: [ROCm/ROCm Discussions](https://github.com/ROCm/ROCm/discussions)
- ROCm Discord: [discord.gg/rocm](https://discord.gg/rocm)
