# wave-primitives

Header-only C++ library for wave32/wave64-agnostic warp-level operations on AMD GPUs.

## Problem

Every CUDA kernel using warp-level primitives assumes `warpSize=32`. On AMD CDNA (MI300X, MI250X), the wavefront is 64 lanes. This breaks:

- Shuffle reductions: loop starting at `offset=16` only covers half the wavefront
- Ballot masks: `0xffffffff` is 32 bits, `0xffffffffffffffff` needed for wave64
- Any fixed-width mask in `__shfl_*_sync`, `__ballot_sync`, `__activemask`

hipCUB wraps rocPRIM but does not fix this at the primitive level.

## Solution

Compile-time detection of wave size via `warpSize`. All primitives resolve to the correct implementation with zero runtime overhead.

```cpp
#include "wave_primitives.hpp"

// Correct on both wave32 (RDNA/CUDA) and wave64 (CDNA)
float result = wave::reduce_sum(my_val);
uint64_t mask = wave::ballot(predicate);
float shuffled = wave::shfl_down(val, offset);
```

## API

| Function | Description |
| --- | --- |
| `wave::reduce_sum(val)` | Warp-level sum reduction |
| `wave::reduce_max(val)` | Warp-level max reduction |
| `wave::scan_inclusive(val)` | Inclusive prefix scan |
| `wave::ballot(pred)` | Ballot — returns `uint64_t` (covers wave64) |
| `wave::shfl_down(val, offset)` | Shuffle down |
| `wave::shfl_xor(val, mask)` | Shuffle XOR |
| `wave::lane_id()` | Current lane index |
| `wave::active_mask()` | Active lane mask |

## Build

Header-only — just include:

```cmake
target_include_directories(your_target PRIVATE path/to/wave-primitives/include)
```

## Status

- [ ] Core reduce/scan primitives
- [ ] Ballot and mask operations
- [ ] Shuffle family
- [ ] Unit tests (gfx942 + gfx1100 for cross-architecture validation)
- [ ] Benchmarks vs hipCUB

## Gap Reference

[Gap 4 — Warp-Agnostic Primitives](../../docs/gap-analysis.md#4-warp-agnostic-primitives--open)
