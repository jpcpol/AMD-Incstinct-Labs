# Research Outline: Wave-Size-Agnostic Primitives for AMD CDNA3

> **Status**: Pre-registration — written before benchmark results  
> **Target**: arXiv cs.DC + AMD ROCm upstream contribution  
> **Hardware**: AMD Instinct MI300X (gfx942, ROCm 7.2)  
> **Repository**: github.com/jpcpol/AMD-Incstinct-Labs

---

## 1. Motivation and Problem Statement

Every CUDA kernel that uses warp-level primitives — shuffle reductions, ballot masks, scan operations — carries an implicit assumption: the warp contains exactly 32 threads. This assumption is correct on all NVIDIA hardware and on AMD RDNA consumer GPUs, but it is **wrong** on AMD CDNA datacenter GPUs, where the wavefront is 64 lanes.

The consequence is concrete: a standard warp reduction loop

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
```

produces incorrect results on CDNA without modification. The loop processes only half the wavefront (lanes 0–31), silently discarding the contributions of lanes 32–63. The ballot mask `0xffffffff` covers 32 bits, leaving 32 lanes unrepresented.

This is the primary portability barrier between CUDA and HIP for any kernel that uses intra-warp communication. The AMD documentation acknowledges it. The `hipCUB` library wraps around it with block-level primitives that require shared memory. No standalone, header-only library provides correct wave-level primitives for both wave32 and wave64.

This gap is not theoretical. Every major ML framework porting effort to ROCm — PyTorch, JAX, vLLM — contains code that either works around this issue manually, uses hipCUB at the block level (incurring unnecessary shared memory overhead), or silently produces wrong results on CDNA architectures.

---

## 2. Background

### 2.1 AMD CDNA3 Architecture (MI300X)

The AMD Instinct MI300X is based on the CDNA3 architecture (gfx942). Relevant hardware properties for this research:

- **Wavefront size**: 64 lanes (wave64) — double the NVIDIA warp size
- **SIMD units**: 4 per Compute Unit, each executing one wavefront
- **LDS (Local Data Share)**: 64 KB per CU — shared memory in HIP terminology
- **Register file**: 512 × 32-bit VGPRs per SIMD unit
- **L2 cache**: 32 MB aggregate across all XCDs
- **HBM3 bandwidth**: 5.3 TB/s

The double wavefront size means that wave-level operations cover twice as many elements as on NVIDIA hardware. This is an advantage for throughput but requires explicit handling in software.

### 2.2 Existing Solutions

**hipCUB WarpReduce / WarpScan**: Correct implementations that wrap AMD's native shuffle instructions. Require `TempStorage` — a shared memory allocation per warp per operation. This makes them unsuitable for use inside register-intensive kernels where shared memory budget is tight (e.g., GEMM kernels, attention kernels with large tile sizes).

**rocm-examples**: AMD's official example repository contains `warp_size_reduction` and `template_warp_size_reduction` demonstrating correct reduce patterns using `warpSize`. These are pedagogical, not library-quality: they use private template helpers (`static_for`, `divide<2>`) from an unreleased internal library and cover only sum reduction.

**HipKittens** (HazyResearch, arXiv 2511.08083, 2025): Tile-level primitives for GEMM and attention on CDNA3. Does not expose wave-level scan, ballot, or shuffle abstractions. Key finding from this work incorporated here: *wave specialization (dedicated producer/consumer waves) achieves only ~80% peak FLOPS on CDNA3* due to AMD's static register allocation. Our library uses uniform wave patterns.

**Gap**: No published, standalone, header-only library provides the full set of wave-level primitives (reduce, scan, ballot, shuffle) correct for both wave32 and wave64 with zero shared memory for the register-only operations.

### 2.3 ROCm 7 warpSize Early-Fold

Starting with ROCm 7.0, `warpSize` is *early-folded* by the compiler: it is treated as a compile-time constant in loop bounds, enabling full loop unrolling without requiring `template<int WarpSize>` dispatch. This is a significant change that simplifies portable wave-agnostic code.

```cpp
// ROCm 7+: compiler unrolls this to 6 iterations on wave64, 5 on wave32
for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);
```

Our library relies on this behavior. Pre-ROCm 7 compatibility is out of scope.

---

## 3. Research Questions

**RQ1** — Correctness:  
Can a header-only C++ library provide wave-level reduce, scan, ballot, and shuffle primitives that produce correct results on both wave32 (RDNA, NVIDIA) and wave64 (CDNA) hardware using a single code path, without preprocessor conditionals on warp size?

**RQ2** — Performance of register-only scan:  
Does eliminating shared memory from the scan primitive (`wave::scan_inclusive_sum`) result in measurably lower latency compared to `hipCUB::WarpScan` on MI300X, given that hipCUB requires a `TempStorage` allocation?

**RQ3** — warpSize early-fold effectiveness:  
Does ROCm 7's `warpSize` early-fold produce the same assembly as an explicit `template<int WarpSize=64>` instantiation, confirming that no runtime overhead is introduced?

**RQ4** — Half-precision accumulation accuracy:  
What is the quantitative accuracy difference between accumulating `__half` values natively (N rounding steps in fp16) versus promoting to float32 during accumulation? Does float32 accumulation introduce measurable latency overhead?

**RQ5** — Block-level composition:  
Does a two-phase block reduction built on `wave::reduce_sum` (register reduce + minimal LDS for inter-wave communication) match the latency of `hipCUB::BlockReduce` across standard block sizes (128, 256, 512, 1024)?

---

## 4. Hypotheses

Each hypothesis is stated before results are collected (pre-registration). Each is falsifiable and references a measurable quantity.

---

**H1** — Register-only scan is faster than hipCUB WarpScan on MI300X

*Statement*: `wave::scan_inclusive_sum` will exhibit lower median kernel launch latency than `hipcub::WarpScan::InclusiveSum` when measured on MI300X (gfx942) with 1024 blocks × 64 threads.

*Rationale*: hipCUB WarpScan allocates `TempStorage` in shared memory even for the shuffle-based path. Our implementation uses only registers (`shfl_up` + conditional add). LDS allocation, even when not actively bank-conflicting, introduces extra resource reservation that limits occupancy.

*Expected outcome*: wave::scan is 10–30% faster. If hipCUB is faster, the compiler's shared memory optimizations exceed our expectations.

*Falsification condition*: median latency of `wave::scan_inclusive_sum` ≥ median latency of `hipCUB::WarpScan::InclusiveSum`.

---

**H2** — wave::reduce_sum matches hipCUB WarpReduce for float

*Statement*: `wave::reduce_sum<float>` and `hipcub::WarpReduce<float>::Sum` will exhibit statistically equivalent median latency (within 5%) on MI300X.

*Rationale*: Both implementations ultimately use `__shfl_down` for the same number of iterations (6 for wave64). hipCUB's TempStorage for WarpReduce is minimal for the Sum operation. We expect convergence in the generated ISA.

*Expected outcome*: within ±5%. If our implementation is faster, it suggests less register pressure from the zero-allocation approach.

*Falsification condition*: median latency difference > 10% in either direction.

---

**H3** — Float32 accumulation for __half reduce costs < 10% latency vs native fp16

*Statement*: `wave::reduce_sum(__half)` (which promotes to float32 internally) will be no more than 10% slower than a naive fp16 reduction that keeps accumulation in fp16 throughout.

*Rationale*: The promotion (`__half2float`) and demotion (`__float2half`) are single-instruction operations on CDNA3. The MFMA and VALU units operate on 32-bit registers regardless. The cost difference is two type conversions per reduction step (6 steps for wave64 = 12 extra instructions).

*Expected outcome*: < 5% overhead. The accuracy benefit is unambiguous: naive fp16 accumulation of 64 values of 0.1f produces ~10× higher absolute error than float32 accumulation.

*Falsification condition*: latency overhead > 15%.

---

**H4** — Two-phase wave block reduction matches hipCUB BlockReduce latency within 15%

*Statement*: `kernel_wave_block_reduce` (phase 1: wave reduce per wavefront; phase 2: first wave reduces partial sums from LDS) will match `hipcub::BlockReduce` median latency within ±15% for block sizes 256, 512, and 1024 on MI300X.

*Rationale*: hipCUB BlockReduce uses an optimized two-phase pattern internally. Our implementation uses the same algorithmic structure but with a smaller LDS footprint (`blockDim.x / warpSize` floats vs hipCUB's larger TempStorage). At block size 1024 on wave64, our LDS usage is 16 floats = 64 bytes; hipCUB may use considerably more.

*Expected outcome*: within ±10% for block sizes ≥ 256. Block size 128 may favor hipCUB due to fewer waves (2 waves on wave64).

*Falsification condition*: latency difference > 25% for any tested block size.

---

**H5** — warpSize early-fold generates equivalent ISA to explicit template dispatch

*Statement*: The assembly generated for `wave::reduce_sum<float>` (using `warpSize`-based loop) will contain exactly 6 `v_add_f32` + `ds_swizzle_b32` / `v_readlane_b32` instructions in sequence, with no branch instructions for the loop control — identical to what a `template<int WarpSize=64>` version would generate.

*Rationale*: ROCm 7 documentation explicitly states `warpSize` is early-folded. The ISA should reflect a fully unrolled loop.

*Verification method*: `hipcc --save-temps` + `llvm-objdump -d` on the `.co` file, manual ISA inspection.

*Falsification condition*: presence of any branch instruction within the reduction sequence, or loop count ≠ 6.

---

## 5. Methodology

### 5.1 Hardware and Software Configuration

| Parameter | Value |
| --- | --- |
| GPU | AMD Instinct MI300X (gfx942) |
| Memory | 192 GB HBM3 |
| ROCm version | 7.2 |
| Compiler | hipcc (amdclang++ 18) |
| Compile flags | `-O3 --offload-arch=gfx942 -std=c++17` |
| OS | Ubuntu 24.04 LTS |
| Platform | AMD Developer Cloud |

### 5.2 Benchmark Protocol

All performance measurements follow this protocol, implemented in `benchmarks/bench_common.hpp`:

1. **Correctness first**: verify results against CPU reference before any timing run. A benchmark that produces wrong answers is invalid regardless of latency.
2. **Warmup**: 100 kernel launches (not timed) to ensure JIT compilation and cache warming are complete.
3. **Measurement**: 1000 timed iterations using `hipEventRecord` / `hipEventElapsedTime`.
4. **Statistics**: report **median** (p50) and **P99** latency. Median is robust to outliers from OS scheduling jitter. P99 captures worst-case behavior relevant to production pipelines.
5. **Throughput**: derived from median latency and total elements processed (Gelems/s).
6. **Clock pinning**: `rocm-smi --setperflevel high` before benchmark runs to disable frequency scaling.

### 5.3 ISA Analysis (H5)

For assembly verification:

```bash
hipcc -O3 --offload-arch=gfx942 --save-temps \
      -I include research/wave-primitives/tests/test_correctness.hip
llvm-objdump -d --mcpu=gfx942 *.co | grep -A 50 "reduce_sum"
```

### 5.4 Accuracy Measurement (H3)

Input: 64 lanes × 0.1f (cannot be represented exactly in fp16 or bfloat16).
Expected sum: 6.4f.
Measure: `|result - 6.4|` for each strategy.

---

## 6. Metrics and Success Criteria

| Hypothesis | Primary metric | Success threshold | Failure threshold |
| --- | --- | --- | --- |
| H1 (scan vs hipCUB) | Median latency ratio wave/cub | < 1.0 (wave faster) | > 1.0 (wave slower) |
| H2 (reduce vs hipCUB) | Median latency ratio | 0.95 – 1.05 | < 0.85 or > 1.15 |
| H3 (fp16 acc overhead) | Latency overhead % | < 10% | > 15% |
| H4 (block reduce) | Latency ratio vs hipCUB | 0.85 – 1.15 | < 0.70 or > 1.30 |
| H5 (ISA unroll) | Branch count in reduction | 0 branches | ≥ 1 branch |

Secondary metrics collected for all benchmarks: P99 latency, VALU utilization (via rocprofv3), L2 cache hit rate, shared memory allocation per kernel (from rocprofv3 kernel stats).

---

## 7. Scope and Limitations

### In scope

- Wave-level reduce (sum, max, min, and/or or), scan (inclusive/exclusive), ballot, shuffle primitives
- Types: float, double, int32, int64, \_\_half, hip\_bfloat16
- Single GPU (MI300X, gfx942)
- ROCm 7.2 compiler behavior

### Out of scope

- Multi-GPU or multi-node scenarios
- Custom user-defined types or non-commutative operations
- RDNA GPU validation (wave32 path is implemented and tested for correctness, but no RDNA hardware benchmarks)
- NVIDIA hardware (portability layer is implemented; performance parity with CUDA is not evaluated)
- Dynamic parallelism or cooperative groups patterns
- Async / persistent kernel patterns

### Known limitations

1. **Single hardware point**: results are for MI300X specifically. MI250X (CDNA2, gfx90a) may differ due to different LDS latency characteristics.
2. **Micro-benchmark gap**: wave-level primitives are rarely the bottleneck in isolation. The relevant metric in production is the improvement when embedded in a real kernel (Flash Attention, GEMM). That measurement is deferred to future work.
3. **hipCUB version dependency**: hipCUB implementation details may change across ROCm versions. Results are pinned to ROCm 7.2.

---

## 8. Expected Contributions

**C1 — Open-source library**  
`wave-primitives`: the first header-only C++ library providing correct wave-level reduce, scan, ballot, and shuffle for both wave32 and wave64, with zero shared memory for register-only operations. MIT license. Candidate for upstream contribution to `ROCm/hipCUB`.

**C2 — Empirical characterization**  
First published performance comparison of register-only vs shared-memory scan on AMD CDNA3 wave64 hardware. Quantitative data on the warpSize early-fold mechanism's effectiveness in ROCm 7.

**C3 — Portability pattern documentation**  
Formal documentation of the wave32/wave64 portability problem with reproducible examples, replacing the scattered tribal knowledge currently distributed across GitHub issues and informal discussions.

**C4 — Benchmark suite**  
Reproducible benchmark suite targeting MI300X, runnable on AMD Developer Cloud, covering all supported types and block sizes.

---

## 9. Target Venues

| Venue | Type | Why |
| --- | --- | --- |
| arXiv cs.DC | Preprint | Primary dissemination — enables AMD maintainer discovery |
| AMD ROCm Blog | Technical article | Direct path to AMD developer audience |
| SC26 Workshop (GPU Computing) | Workshop paper | Community visibility, peer review |
| IEEE TPDS | Journal (future) | If extended with Flash Attention DME results |

Submission sequence: arXiv first (no embargo), then workshop.

---

## 10. Results

> *This section is intentionally blank. It will be completed after benchmark execution on AMD Instinct MI300X hardware.*

### 10.1 Environment

| Field | Value |
| --- | --- |
| GPU | — |
| ROCm | — |
| Date | — |
| Clock state | — |

### 10.2 H1 — Scan vs hipCUB WarpScan

| Kernel | Median (µs) | P99 (µs) | Gelems/s |
| --- | --- | --- | --- |
| wave::scan_inclusive_sum | — | — | — |
| hipCUB WarpScan::InclusiveSum | — | — | — |

Outcome: [ CONFIRMED / REFUTED / INCONCLUSIVE ]

### 10.3 H2 — Reduce vs hipCUB WarpReduce

| Kernel | Median (µs) | P99 (µs) | Gelems/s |
| --- | --- | --- | --- |
| wave::reduce_sum float | — | — | — |
| hipCUB WarpReduce::Sum | — | — | — |

Outcome: [ CONFIRMED / REFUTED / INCONCLUSIVE ]

### 10.4 H3 — Float32 accumulation overhead

| Strategy | Median (µs) | Error vs expected |
| --- | --- | --- |
| wave::reduce_sum __half (fp32 acc) | — | — |
| Naive fp16 reduce | — | — |

Outcome: [ CONFIRMED / REFUTED / INCONCLUSIVE ]

### 10.5 H4 — Block reduce vs hipCUB BlockReduce

| Block size | wave (µs) | hipCUB (µs) | Ratio |
| --- | --- | --- | --- |
| 128 | — | — | — |
| 256 | — | — | — |
| 512 | — | — | — |
| 1024 | — | — | — |

Outcome: [ CONFIRMED / REFUTED / INCONCLUSIVE ]

### 10.6 H5 — ISA unroll verification

Reduction instruction count: —  
Branch instructions in reduction sequence: —  
Outcome: [ CONFIRMED / REFUTED ]

### 10.7 Discussion

*To be written after results.*

### 10.8 Conclusions

*To be written after results.*
