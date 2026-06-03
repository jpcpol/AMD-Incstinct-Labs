# AMD Instinct Labs

Independent research targeting confirmed gaps between the ROCm software stack and the hardware capabilities of AMD Instinct MI300X (CDNA3, gfx942).

The MI300X has meaningful hardware advantages over competing accelerators that remain unexploited at the software level. This project identifies each gap against AMD primary sources, implements a reference solution, benchmarks it against the current ROCm baseline, and contributes upstream.

---

## The Problem

AMD's MI300X is the highest-memory-bandwidth AI accelerator commercially available, yet the software stack consistently leaves performance on the table:

| Hardware advantage | Current software reality |
| --- | --- |
| 5.3 TB/s HBM3 (1.58× H100) | Flash Attention ignores the Data Movement Engine — Q/K/V loads block compute |
| 7-link full-mesh Infinity Fabric | RCCL uses Ring/Tree AllReduce designed for linear PCIe topologies |
| Wave64 SIMT (64-lane wavefront) | Every CUDA warp-level primitive assumes 32 lanes — broken on CDNA without manual porting |
| DME: async HBM→LDS tensor copies | No C++ API exists — only MLIR low-level intrinsics |
| MI300A: 128 GB unified CPU+GPU HBM | No framework defines programming patterns for zero-copy CPU↔GPU algorithms |
| MFMA matrix units (FP8/BF16/FP32) | Tile size selection for GEMM lowering is manual — no cuBLAS-equivalent auto-tuning |

These are not speculative gaps. Each one is verified against AMD's own documentation, GitHub issues, and arXiv research. The full analysis with primary source citations is in [`docs/gap-analysis.md`](docs/gap-analysis.md).

---

## Documentation

| Document | Description |
| --- | --- |
| [`docs/gap-analysis.md`](docs/gap-analysis.md) | Full gap analysis with per-gap status verified against AMD primary sources |
| [`docs/research-outline.md`](docs/research-outline.md) | Research outline: hypotheses, methodology, metrics, and result placeholders — pre-registered before benchmark execution |

---

## Research Areas

| Area | Status | What it solves |
| --- | --- | --- |
| [wave-primitives](research/wave-primitives/) | **First result on MI300X** | Header-only library: correct `reduce`, `scan`, `ballot`, `shuffle` for both wave32 (RDNA/NVIDIA) and wave64 (CDNA). DPP-based float32 reduction measured **1.35–1.79× faster than hipCUB** in one configuration — bounded result, see Key Result below. |
| [flash-attention-mi300x](research/flash-attention-mi300x/) | In progress | Flash Attention using DME async prefetch to overlap HBM loads with MFMA compute; FA3-style pipelining for AMD |
| [infinity-fabric-allreduce](research/infinity-fabric-allreduce/) | Planned | Topology-aware AllReduce that uses all 7 Infinity Fabric links simultaneously instead of serializing through a ring |
| [dme-abstraction](research/dme-abstraction/) | Planned | C++ API over the CDNA3 Data Movement Engine — the only CDNA3 feature with no NVIDIA equivalent and no usable interface |
| [mlir-mfma-tiling](research/mlir-mfma-tiling/) | Planned | MLIR pass for automatic MFMA tile-size selection: analyze GEMM shape → select variant → emit optimal tiled code |

---

## Key Result (MI300X VF, ROCm 7.2, 2026-06-03 — bounded experimental claim)

For **wave64 reduction and scan** under a specific configuration, header-only
primitives on the CDNA3 DPP datapath measured faster than hipCUB — with **zero
LDS traffic confirmed by rocprofv3**:

| Primitive (`<float>`, wave64) | DPP median | vs hipCUB | LDS (measured) |
| --- | --- | --- | --- |
| `wave::dpp::reduce_sum` (full DPP) | 210–279 µs | **1.35–1.79×** | **0** |
| `wave::dpp::scan_inclusive_sum` (full DPP) | 343 µs | **1.028×** | **0** |
| (portable `__shfl` baselines) | 922 / 966 µs | 0.36–0.41× | 25.2M |
| (hipCUB WarpReduce / WarpScan) | 377 / 352 µs | 1.00× | 4.2M / 0 |

Generalized across **float/double/int32/int64 × sum/max/min** (12/12 correct).

**Why (the part that generalizes):** the generic HIP `__shfl_*` routes every
cross-lane step through Local Data Share (`ds_bpermute`). The DPP path keeps the
whole wave64 operation on the cross-lane VALU datapath — in-row via `row_shr`,
cross-row/bank via `row_bcast` cascades — reaching **zero `ds_bpermute`**.
rocprofv3 measures `SQ_INSTS_LDS = 0` for both DPP primitives vs 25.2M for the
portable path, with near-identical VALU — isolating LDS elimination as the cause.
The unifying rule: *whoever removes all LDS-mediated cross-lane traffic wins on
CDNA3 wave64* — true for reduce (we beat hipCUB) and scan (we edge it).

**What this is and isn't.** Still a *bounded* result: a **virtualized** MI300X,
clocks unpinned, and CDNA3-vs-gfx942 generality untested (no MI250/RDNA3 access).
It is **not** a blanket "we beat hipCUB" claim. Open items before strengthening it
— other types, rocprofv3 LDS-traffic counters, and CDNA3-vs-gfx942 hardware
generality — are tracked in [`docs/research-outline.md` §10.10](docs/research-outline.md).
That said, any ROCm code reducing via `__shfl_down` (much of the PyTorch/JAX/vLLM
HIP surface) routes cross-lane traffic through LDS, so the direction of the
finding has broad reach.

Full data, methodology, and the (refuted) pre-registered hypothesis that led here:
[`docs/research-outline.md` §10](docs/research-outline.md).

---

## Methodology

Each research area follows a five-step process: confirm the gap against primary sources → survey existing implementations → implement targeting gfx942 → benchmark against the ROCm baseline (p50/p99 latency, rocprofv3 hardware counters) → contribute upstream via arXiv preprint + AMD GitHub Discussion + PR.

Hypotheses are pre-registered before benchmark execution to prevent result-driven framing. For the full methodology, benchmark protocol, and per-hypothesis success criteria see [`docs/research-outline.md`](docs/research-outline.md).

---

## Hardware

**Primary target**: AMD Instinct MI300X

| Specification | Value |
| --- | --- |
| Architecture | CDNA3 (gfx942) |
| Compute units | 304 (8 XCDs × 38 CUs) |
| HBM3 capacity | 192 GB |
| HBM3 bandwidth | 5.3 TB/s |
| MFMA peak (FP8) | 2,614 TFLOPS |
| MFMA peak (BF16) | 1,307 TFLOPS |
| Wavefront size | 64 lanes (wave64) |
| L2 cache | 32 MB aggregate |
| Infinity Fabric links | 7 bidirectional per GPU |

**Secondary target**: AMD Instinct MI300A (same gfx942, 128 GB HBM3 shared with Zen 4 CPU)

---

## Build

**Requirements**: ROCm 7.0+, CMake 3.21+

> ROCm 7.0+ is required for `warpSize` early-fold — the compiler mechanism that enables loop unrolling without template dispatch. ROCm 7.2 is the tested version.

```bash
git clone https://github.com/jpcpol/AMD-Incstinct-Labs.git
cd AMD-Incstinct-Labs

cmake -B build -DGPU_TARGETS=gfx942 -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

**Full test + benchmark suite** (recommended on AMD Developer Cloud):

```bash
cd research/wave-primitives
chmod +x scripts/run_all.sh
./scripts/run_all.sh        # compiles, runs all tests and benchmarks, saves to results/
```

**Quick single-file correctness test** (no CMake):

```bash
hipcc -O2 --offload-arch=gfx942 \
      -I research/wave-primitives/include \
      research/wave-primitives/tests/test_correctness.hip \
      -o test_correctness
./test_correctness
```

Expected output on MI300X:

```
[ENV]  warpSize = 64 (wave64 detected)
[PASS] reduce_sum  float
[PASS] reduce_sum  double
[PASS] reduce_max  float
[PASS] reduce_min  float
[PASS] scan_inclusive_sum
[PASS] scan_exclusive_sum
[PASS] ballot  (mask = 0x5555555555555555)
[PASS] popcount  (cnt = 32)
All tests passed.
```

---

## Testing on AMD Developer Cloud

All benchmarks in this project are designed to run on [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) — AMD's own MI300X cloud for developers ($1.99/hour per GPU). New accounts receive $100 in credits via the [AMD AI Developer Program](https://www.amd.com/en/developer/ai-dev-program.html).

Recommended setup: **Vanilla ROCm** image (Ubuntu 24.04, ROCm 7.2), accessed via SSH. The full benchmark suite runs in under 10 minutes.

---

## Prior Work and Relationship to Existing Libraries

This project does not replace hipCUB, Composable Kernel, or MIOpen. It targets the gaps those libraries leave:

- **hipCUB**: excellent block- and warp-level primitives, but does not provide a standalone *header-only* wave-level API correct for both wave32 and wave64. Its `WarpReduce` uses the DPP datapath yet still emits a residual `ds_bpermute` for the 32-lane bank cross; our full-DPP reduction avoids it (`row_bcast31`) and is 1.35–1.79× faster on MI300X (see Key Result)
- **Composable Kernel**: high-performance GEMM and attention kernels, but no public DME abstraction layer
- **RCCL**: functional AllReduce, but Ring/Tree topology assumptions leave bandwidth on the table for Infinity Fabric full-mesh clusters
- **HipKittens** (HazyResearch, 2025): tile-level framework for GEMM/attention, does not expose wave-level primitives

Key finding from HipKittens incorporated here: wave specialization (dedicated producer/consumer waves) achieves only ~80% peak on CDNA3 due to AMD's static register allocation. All primitives in this library use uniform wave patterns.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: open an issue with the gap you want to target, include the primary source confirming it's open, and propose a measurable outcome. PRs without benchmarks are not accepted.

## License

MIT — see [LICENSE](LICENSE).
