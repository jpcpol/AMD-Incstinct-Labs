# wave-primitives

Header-only C++ library for wave64-optimized warp-level operations on AMD Instinct (CDNA3).

## Key Finding

The central result of this library is **not** wave-size portability — it is that the CDNA3
DPP (Data-Parallel Primitives) cross-lane datapath can replace LDS-mediated cross-lane
communication entirely for wave64 reduction and scan.

The generic HIP `__shfl_down` / `__shfl_up` lower to `ds_bpermute` — an LDS round-trip per
step. A full-DPP implementation issues **zero LDS instructions** (rocprofv3-measured,
`SQ_INSTS_LDS = 0`) and outperforms hipCUB on both algorithms:

| Primitive (`float`, wave64) | Median (µs) | vs hipCUB | SQ_INSTS_LDS |
| --- | --- | --- | --- |
| `wave::dpp::reduce_sum` (full DPP, broadcast) | 279 | **1.35×** | **0** |
| `wave::dpp::reduce_sum` (full DPP, lane-63) | 210 | **1.79×** | **0** |
| `wave::dpp::scan_inclusive_sum` (full DPP) | 343 | **1.028×** | **0** |
| hipCUB WarpReduce::Sum | 377 | 1.00× | 4.2M |
| hipCUB WarpScan::InclusiveSum | 352 | 1.00× | 0 |
| portable `__shfl_down` reduce | 922 | 0.41× | 25.2M |
| portable `__shfl_up` scan | 966 | 0.36× | 25.2M |

Measured on MI300X VF (gfx942, ROCm 7.2, hipCUB 4.2.0, kReps=4096). See
[`docs/research-outline.md §10`](../../docs/research-outline.md) for full methodology
and caveats.

## Why it works

```text
__shfl_*  →  ds_bpermute  →  LDS round-trip per step  →  25.2M LDS insts
   vs
DPP       →  cross-lane VALU datapath  →  zero LDS  →  lower latency
```

**Reduction geometry (wave64):**

- In-row prefix (16 lanes): `row_shr 1,2,4,8` — row totals accumulate on lanes 15/31/47/63
- Cross-row carry: cascaded `row_bcast15` (0x142) + `row_bcast31` (0x143) — no LDS, no shfl
- Broadcast to all lanes: `v_readlane_b32` (optional; omit for lane-63 result)

**Scan geometry (wave64):**

- In-row prefix: same `row_shr` cascade
- Exclusive shift: `row_shr1` (0x111) + `row_bcast15` carry-patch at lanes 16/32/48

Both geometries verified empirically lane-by-lane; DPP control codes confirmed against
AMD ISA reference.

## API

```cpp
#include "wave_primitives/wave_reduce_dpp.hpp"
#include "wave_primitives/wave_scan_dpp.hpp"

// Reduction — result on all lanes (broadcast)
float  r = wave::dpp::reduce_sum(my_float);
double d = wave::dpp::reduce_sum(my_double);
int    i = wave::dpp::reduce_sum(my_int32);

// Reduction — result on lane 63 only (fastest path, no readlane)
float r63 = wave::dpp::reduce_sum_lane63(my_float);  // see header

// Max/min
float mx = wave::dpp::reduce_max(my_float);
float mn = wave::dpp::reduce_min(my_float);

// Inclusive/exclusive prefix scan
float inc = wave::dpp::scan_inclusive_sum(my_float);
float exc = wave::dpp::scan_exclusive_sum(my_float);
```

All primitives: `float`, `double`, `int32_t`, `int64_t` — **12/12 correctness PASS**
(MI300X, ROCm 7.2). 64-bit DPP compiles directly on amdclang 22 (no hi/lo split).

Wave32 / NVIDIA fallback: portable `__shfl_*` path — same API, no DPP.

## Build (header-only)

```cmake
target_include_directories(your_target PRIVATE path/to/wave-primitives/include)
```

No CMake required for single-file use:

```bash
hipcc -O3 --offload-arch=gfx942 \
      -I research/wave-primitives/include \
      research/wave-primitives/tests/probe_scan_dpp.hip \
      -o probe_scan_dpp && ./probe_scan_dpp
```

Expected output:

```text
  f32 sum/max/min/excl ALL PASS
  i32 sum/max/min/excl ALL PASS
  f64 sum/max/min/excl ALL PASS
  i64 sum/max/min/excl ALL PASS

ALL TYPES/OPS PASS
```

## Status

| Primitive | DPP path | Types | Correctness | Benchmark |
| --- | --- | --- | --- | --- |
| `reduce_sum/max/min` | Full DPP (zero LDS) | f32/f64/i32/i64 | PASS 12/12 | Run 2 ✓ |
| `scan_inclusive_sum` | Full DPP (zero LDS) | f32/f64/i32/i64 | PASS 12/12 | Run 2 ✓ |
| `scan_exclusive_sum` | Full DPP (zero LDS) | f32 | PASS | Run 2 ✓ |
| `ballot` / `popcount` | Portable (correct) | — | PASS | — |
| `__half` / `bf16` | Not yet | — | — | — |
| Block reduce | Two-phase (LDS inter-wave) | f32 | — | Deferred (kReps) |

## Probes and tests

| File | Purpose |
| --- | --- |
| `tests/probe_scan_dpp.hip` | Lane-by-lane scan correctness (all 4 types × 4 ops) |
| `tests/probe_reduce_dpp.hip` | Lane-by-lane reduce correctness |
| `tests/probe_dpp_bcast_map.hip` | Verify DPP control code → lane routing empirically |
| `benchmarks/bench_reduce_dpp.hip` | Timing vs hipCUB WarpReduce |
| `benchmarks/bench_reduce_dpp_types.hip` | Timing across f32/f64/i32/i64 × sum/max/min |
| `benchmarks/bench_scan_dpp.hip` | Timing vs hipCUB WarpScan |

## Gap Reference

[Gap 4 — Warp-Agnostic Primitives](../../docs/gap-analysis.md#4-warp-agnostic-primitives--open)
