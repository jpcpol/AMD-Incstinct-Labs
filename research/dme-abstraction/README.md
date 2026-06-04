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

## Implementation Path — incremental (de-risked)

We build bottom-up: get the *simplest* DME copy working and verified on real
hardware first, then layer the tensor API on top. This avoids designing the 4D
tensor descriptor before we even know the 1D builtin's exact semantics.

1. **1D element / tile copy** (`dme_copy.hpp`) — direct wrapper over
   `__builtin_amdgcn_global_load_lds`. ← *current step*
2. Empirical probe of the intrinsic's offset + wait semantics (`probe_dme.hip`).
3. Bandwidth benchmark: DME copy vs raw buffer load vs `hipMemcpyAsync`.
4. Async token interface (overlap DME load with MFMA compute).
5. Multi-dimensional tensor API (the `tensor_prefetch` vision above).
6. Integration with `flash-attention-mi300x`.

## The underlying intrinsic (ROCm 7.x, gfx942)

```cpp
void __builtin_amdgcn_global_load_lds(
    void* global_ptr,  // per-lane HBM source  (addrspace 1 or 7)
    void* lds_ptr,     // uniform LDS dest      (addrspace 3)
    int   size,        // bytes: 1/2/4/8 (immediate)
    int   offset,      // immediate byte offset — applied to BOTH ptrs (to verify)
    int   aux);        // cache policy (sc0 = streaming, recommended on gfx942)
```

## Files (current)

| File | Purpose |
| --- | --- |
| `include/dme/dme_copy.hpp` | C++ API: `copy_element`, `copy_element_stream`, `copy_tile_1d`, `wait` |
| `tests/probe_dme.hip` | Empirical probe of the builtin's offset + wait semantics |

## Open questions for Run 4 (probe answers these)

1. **Offset semantics** — does `offset` apply to global ptr, LDS ptr, or both?
2. **Wait encoding** — is `__builtin_amdgcn_s_waitcnt(0)` correct, or is a
   specific `lgkmcnt`/`vmcnt` encoding needed? (Wrong → corrupted data in probe.)
3. **rocprofv3** — does the copy route through the DME (off the VMEM path)?

## Status

- [x] Research: confirm `__builtin_amdgcn_global_load_lds` signature (ROCm 7.x)
- [x] 1D sync API prototype (`dme_copy.hpp`) — written, **unvalidated**
- [x] Hardware probe written (`probe_dme.hip`)
- [ ] Hardware validation (Run 4 — blocked on GPU availability)
- [ ] Bandwidth benchmark: DME vs raw buffer load vs `hipMemcpyAsync`
- [ ] Async token interface
- [ ] Multi-dimensional tensor API
- [ ] Integration with `flash-attention-mi300x`

## Gap Reference

[Gap 7 — DME Abstraction](../../docs/gap-analysis.md#7-dme-abstraction--open-exclusive-to-cdna3)
