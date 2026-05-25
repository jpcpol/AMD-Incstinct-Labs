# dme-abstraction

High-level C++ API for the CDNA3 Data Movement Engine — a feature exclusive to MI300X/MI300A with no NVIDIA equivalent.

## What is the DME

The Data Movement Engine on CDNA3 handles multi-dimensional tensor copies from HBM to LDS (Local Data Share) without occupying compute units. It supports arbitrary tensor layouts and strides — not just linear copies.

At the MLIR level, it's exposed as:

```mlir
amdgpu.tensor_load_to_lds %src_hbm, %dst_lds, {layout attributes}
amdgpu.tensor_store_from_lds %src_lds, %dst_hbm, {layout attributes}
```

No C++ API exists. Users must either use MLIR intrinsics directly or use raw `__builtin_amdgcn_global_load_lds` with manual layout computation.

## Proposed API

```cpp
#include "dme/tensor_prefetch.hpp"

// Prefetch a 4D tensor tile from HBM into LDS
dme::tensor_prefetch(
    src_hbm,                              // HBM pointer
    dst_lds,                              // LDS pointer
    {batch, heads, seq_len, head_dim},    // tensor shape
    {stride_b, stride_h, stride_s, 1},   // HBM strides
    dme::layout::row_major               // layout hint
);

// Async version: returns token, overlaps with compute
auto token = dme::async_prefetch(...);
// ... MFMA compute ...
dme::wait(token);  // ensure load complete before reading LDS
```

## Implementation Path

1. Thin wrapper around `amdgpu.tensor_load_to_lds` via HIP inline assembly / LLVM intrinsics
2. Layout computation helper: converts multi-dimensional indexing to byte offsets
3. Async token interface using `rocdl.asyncmark` / `rocdl.wait.asyncmark`

## Files

```
include/
    dme/tensor_prefetch.hpp    # Main API
    dme/layout.hpp             # Layout descriptors and stride helpers
    dme/async_token.hpp        # Async completion tokens
src/
    dme_intrinsics.hip         # Thin wrappers over LLVM intrinsics
tests/
    test_basic_load.hip        # Correctness: DME load matches memcpy
    test_async_overlap.hip     # Verify compute+load overlap via rocprof
benchmarks/
    bench_bandwidth.hip        # DME bandwidth vs direct buffer load
```

## Status

- [ ] Research: verify exact intrinsic names in ROCm 6.2 (`__builtin_amdgcn_global_load_lds` availability)
- [ ] Basic sync API prototype
- [ ] Async token interface
- [ ] Bandwidth benchmark: DME vs raw buffer load vs `hipMemcpyAsync`
- [ ] Integration with `flash-attention-mi300x`

## Gap Reference

[Gap 7 — DME Abstraction](../../docs/gap-analysis.md#7-dme-abstraction--open-exclusive-to-cdna3)
