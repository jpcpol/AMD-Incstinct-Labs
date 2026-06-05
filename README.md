# AMD Instinct Labs

Independent research on the AMD Instinct MI300X (CDNA3, gfx942) — targeting
confirmed gaps between the ROCm software stack and the hardware's actual capabilities.

---

## Navigation

| Where to go | What you'll find |
| --- | --- |
| [Key Result](#key-result) | The main finding in one table |
| [Research Areas](#research-areas) | Five areas, their status, and links |
| [Repository Layout](#repository-layout) | Every directory and file explained |
| [How to Run](#how-to-run) | Build, test, benchmark — one command |
| [Methodology](#methodology) | How hypotheses, benchmarks, and claims work |
| [Prior Work](#prior-work-and-relationship-to-existing-libraries) | What this adds vs hipCUB / CK / RCCL |
| [Hardware](#hardware) | MI300X specifications |

---

## Key Result

Measured on MI300X VF (gfx942, ROCm 7.2, hipCUB 4.2.0, 2026-06-03).
Full methodology and caveats: [`docs/research-outline.md §10`](docs/research-outline.md).

| Primitive (`float`, wave64) | Median (µs) | vs hipCUB | SQ_INSTS_LDS (rocprofv3) |
| --- | --- | --- | --- |
| `wave::dpp::reduce_sum` (full DPP, broadcast) | 279 | **1.35×** | **0** |
| `wave::dpp::reduce_sum` (full DPP, lane-63) | 210 | **1.79×** | **0** |
| `wave::dpp::scan_inclusive_sum` (full DPP) | 343 | **1.028×** | **0** |
| hipCUB WarpReduce::Sum | 377 | 1.00× | 4.2M |
| hipCUB WarpScan::InclusiveSum | 352 | 1.00× | 0 |
| portable `__shfl_down` reduce | 922 | 0.41× | 25.2M |
| portable `__shfl_up` scan | 966 | 0.36× | 25.2M |

**The finding:** `__shfl_*` lowers to `ds_bpermute` on CDNA3 — every cross-lane
step is an LDS round-trip. A full-DPP implementation stays on the cross-lane VALU
datapath (`row_shr` in-row + `row_bcast` cascade cross-row), reaching
**zero LDS traffic** by hardware counter. The same pattern holds for both
reduction (1.79× vs hipCUB) and scan (~parity, 1.028×), making this look
architectural rather than algorithm-specific.

Generalized across **float/double/int32/int64 × sum/max/min** — 12/12 correctness PASS.

**Caveats:** virtualized MI300X, clocks unpinned, single hardware point (no MI250/RDNA3),
microbenchmark only (no real-kernel demonstration yet). Stated as a bounded claim, not
a general "we beat hipCUB" assertion. Open items tracked in
[`docs/research-outline.md §10.10`](docs/research-outline.md).

---

## Research Areas

| Area | Status | Key finding / goal | Directory |
| --- | --- | --- | --- |
| **wave-primitives** | **Results on MI300X** | Full-DPP geometry eliminates LDS cross-lane traffic; 1.35–1.79× over hipCUB reduce, 1.028× scan | [`research/wave-primitives/`](research/wave-primitives/) |
| flash-attention-mi300x | In progress | Flash Attention using DME async prefetch to overlap HBM loads with MFMA compute | [`research/flash-attention-mi300x/`](research/flash-attention-mi300x/) |
| infinity-fabric-allreduce | Planned | Topology-aware AllReduce using all 7 Infinity Fabric links (vs Ring/Tree) | [`research/infinity-fabric-allreduce/`](research/infinity-fabric-allreduce/) |
| dme-abstraction | Planned | C++ API over the CDNA3 Data Movement Engine — no equivalent in NVIDIA toolchain | [`research/dme-abstraction/`](research/dme-abstraction/) |
| mlir-mfma-tiling | Planned | MLIR pass for automatic MFMA tile-size selection from GEMM shape | [`research/mlir-mfma-tiling/`](research/mlir-mfma-tiling/) |

---

## Repository Layout

```text
amd-instinct-labs/
│
├── README.md                          ← this file (index)
├── CONTRIBUTING.md                    ← contribution rules (benchmarks required)
├── CMakeLists.txt                     ← root build (HIP/CMake 3.21+)
│
├── docs/
│   ├── gap-analysis.md                ← 8 gaps verified against AMD primary sources
│   └── research-outline.md            ← pre-registered hypotheses + Run 1/2 results (§10)
│
├── benchmarks/
│   └── README.md                      ← benchmark protocol (warmup, kReps, statistics)
│
└── research/
    │
    ├── wave-primitives/               ← ACTIVE — first results available
    │   ├── README.md                  ← area overview, result table, geometry explanation
    │   ├── include/wave_primitives/
    │   │   ├── wave_reduce_dpp.hpp    ← full-DPP reduce (sum/max/min × 4 types)
    │   │   ├── wave_scan_dpp.hpp      ← full-DPP scan (inclusive/exclusive)
    │   │   └── detail/config.hpp      ← WP_AMD / WP_NVIDIA compile-time detection
    │   ├── tests/
    │   │   ├── probe_scan_dpp.hip     ← lane-by-lane scan correctness (4 types × 4 ops)
    │   │   ├── probe_reduce_dpp.hip   ← lane-by-lane reduce correctness
    │   │   └── probe_dpp_bcast_map.hip← empirical DPP control-code → lane routing
    │   ├── benchmarks/
    │   │   ├── bench_reduce_dpp.hip   ← reduce timing vs hipCUB WarpReduce
    │   │   ├── bench_reduce_dpp_types.hip ← f32/f64/i32/i64 × sum/max/min
    │   │   ├── bench_scan_dpp.hip     ← scan timing vs hipCUB WarpScan
    │   │   └── bench_block_reduce.hip ← two-phase block reduce vs hipCUB BlockReduce
    │   └── scripts/
    │       └── run_all_validation.sh  ← compile + run all probes + benchmarks in one pass
    │
    ├── flash-attention-mi300x/        ← IN PROGRESS
    │   ├── README.md
    │   └── docs/design.md             ← FA3-style DME pipeline design
    │
    ├── dme-abstraction/               ← PLANNED
    │   └── README.md
    │
    ├── infinity-fabric-allreduce/     ← PLANNED
    │   ├── README.md
    │   └── docs/design.md
    │
    └── mlir-mfma-tiling/              ← PLANNED
        └── README.md
```

---

## How to Run

**Requirements:** ROCm 7.0+, CMake 3.21+

> ROCm 7.0+ is required: `warpSize` is early-folded by the compiler, enabling
> full loop unrolling without template dispatch. ROCm 7.2 is the tested version.

### Full validation suite (recommended — runs on AMD Developer Cloud)

```bash
git clone https://github.com/jpcpol/AMD-Incstinct-Labs.git
cd AMD-Incstinct-Labs/research/wave-primitives
bash scripts/run_all_validation.sh
```

Compiles all probes and benchmarks with `hipcc`, runs correctness first, then
benchmarks. Results saved to `results/`. Typical runtime < 10 minutes on MI300X.

### CMake build

```bash
cmake -B build -DGPU_TARGETS=gfx942 -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Quick single-file correctness check (no CMake)

```bash
hipcc -O3 --offload-arch=gfx942 \
      -I research/wave-primitives/include \
      research/wave-primitives/tests/probe_scan_dpp.hip \
      -o probe_scan_dpp && ./probe_scan_dpp
```

Expected output on MI300X:

```text
  f32 sum/max/min/excl ALL PASS
  i32 sum/max/min/excl ALL PASS
  f64 sum/max/min/excl ALL PASS
  i64 sum/max/min/excl ALL PASS

ALL TYPES/OPS PASS
```

---

## Testing on AMD Developer Cloud

All benchmarks run on [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
— AMD's own MI300X cloud ($1.99/hr). New accounts receive $100 via the
[AMD AI Developer Program](https://www.amd.com/en/developer/ai-dev-program.html).

Recommended image: **Vanilla ROCm** (Ubuntu 24.04, ROCm 7.2). Full suite runs in
under 10 minutes.

---

## Methodology

Each research area follows five steps:

1. Confirm the gap against AMD primary sources (ISA reference, ROCm docs, GitHub issues)
2. Survey existing implementations (hipCUB, Composable Kernel, RCCL, arXiv)
3. Implement targeting gfx942 — header-only where possible
4. Benchmark against the ROCm baseline: p50/p99 latency + rocprofv3 hardware counters
5. Contribute upstream: arXiv preprint + AMD GitHub Discussion + PR

Hypotheses are **pre-registered before benchmark execution** to prevent
result-driven framing. The full pre-registration, outcomes (including refutations),
and open questions are in [`docs/research-outline.md`](docs/research-outline.md).

---

## The Gaps

The MI300X has hardware advantages that the current software stack does not exploit:

| Hardware advantage | Current software reality |
| --- | --- |
| 5.3 TB/s HBM3 (1.58× H100) | Flash Attention ignores the Data Movement Engine — Q/K/V loads block compute |
| 7-link full-mesh Infinity Fabric | RCCL uses Ring/Tree AllReduce designed for linear PCIe topologies |
| Wave64 SIMT (64-lane wavefront) | `__shfl_*` lowers to `ds_bpermute` (LDS) — **being addressed here** |
| DME: async HBM→LDS tensor copies | No C++ API exists — only MLIR low-level intrinsics |
| MI300A: 128 GB unified CPU+GPU HBM | No framework defines zero-copy CPU↔GPU programming patterns |
| MFMA matrix units (FP8/BF16/FP32) | Tile size for GEMM lowering is manual — no cuBLAS-equivalent auto-tuning |

Full analysis with primary source citations: [`docs/gap-analysis.md`](docs/gap-analysis.md).

---

## Prior Work and Relationship to Existing Libraries

This project does not replace hipCUB, Composable Kernel, or RCCL. It targets
the specific gaps those libraries leave open:

- **hipCUB WarpReduce/WarpScan**: correct and fast, but uses DPP + 1 residual
  `ds_bpermute` for the 32-lane bank cross. Our `row_bcast31` step avoids it —
  1.35–1.79× faster for reduce on MI300X (§10.3). No standalone header-only API.
- **Composable Kernel**: high-performance GEMM and attention, but no public
  DME abstraction layer.
- **RCCL**: functional AllReduce, but Ring/Tree topology assumptions leave
  bandwidth on the table for Infinity Fabric full-mesh clusters.
- **HipKittens** (HazyResearch, arXiv 2511.08083, 2025): tile-level framework
  for GEMM/attention on CDNA3, no wave-level primitives. Key finding incorporated
  here: wave specialization achieves only ~80% peak FLOPS on CDNA3 due to static
  register allocation — all primitives here use uniform wave patterns.

---

## Hardware

**Primary target:** AMD Instinct MI300X

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

**Secondary target:** AMD Instinct MI300A (same gfx942, 128 GB HBM3 shared with Zen 4 CPU)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: open an issue with the
gap you want to target, include the primary source confirming it is open, and
propose a measurable outcome. PRs without benchmarks are not accepted.

## License

MIT — see [LICENSE](LICENSE).
