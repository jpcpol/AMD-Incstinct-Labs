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
| [wave-primitives](research/wave-primitives/) | In progress | Header-only library: correct `reduce`, `scan`, `ballot`, `shuffle` for both wave32 (RDNA/NVIDIA) and wave64 (CDNA) |
| [flash-attention-mi300x](research/flash-attention-mi300x/) | In progress | Flash Attention using DME async prefetch to overlap HBM loads with MFMA compute; FA3-style pipelining for AMD |
| [infinity-fabric-allreduce](research/infinity-fabric-allreduce/) | Planned | Topology-aware AllReduce that uses all 7 Infinity Fabric links simultaneously instead of serializing through a ring |
| [dme-abstraction](research/dme-abstraction/) | Planned | C++ API over the CDNA3 Data Movement Engine — the only CDNA3 feature with no NVIDIA equivalent and no usable interface |
| [mlir-mfma-tiling](research/mlir-mfma-tiling/) | Planned | MLIR pass for automatic MFMA tile-size selection: analyze GEMM shape → select variant → emit optimal tiled code |

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

- **hipCUB**: excellent block-level primitives, but does not provide a standalone wave-level API correct for both wave32 and wave64
- **Composable Kernel**: high-performance GEMM and attention kernels, but no public DME abstraction layer
- **RCCL**: functional AllReduce, but Ring/Tree topology assumptions leave bandwidth on the table for Infinity Fabric full-mesh clusters
- **HipKittens** (HazyResearch, 2025): tile-level framework for GEMM/attention, does not expose wave-level primitives

Key finding from HipKittens incorporated here: wave specialization (dedicated producer/consumer waves) achieves only ~80% peak on CDNA3 due to AMD's static register allocation. All primitives in this library use uniform wave patterns.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: open an issue with the gap you want to target, include the primary source confirming it's open, and propose a measurable outcome. PRs without benchmarks are not accepted.

## License

MIT — see [LICENSE](LICENSE).
