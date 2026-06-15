# dme-abstraction

C++ API for the CDNA3 Data Movement Engine — a hardware feature exclusive to MI300X/MI300A with no NVIDIA equivalent.

## What is the DME

The Data Movement Engine on CDNA3 issues async HBM→LDS copies without occupying the VMEM pipeline. This lets compute units (MFMA, VALU) run concurrently with the load, hiding HBM latency.

```text
Without DME:  load → stall → compute   (serial)
With DME:     async prefetch ─────────┐
              compute (MFMA) ─────────┤ overlap
              barrier + use ──────────┘
```

The hardware signature of successful overlap: `SQ_INSTS_VMEM_RD` drops by 44.7% in Flash Attention Step 3 vs naive (measured with rocprofv3 on MI300X, T-025).

**No public C++ API exists for the DME.** The only access is via the low-level intrinsic `__builtin_amdgcn_global_load_lds`. This library wraps it.

## Files

| File | Purpose | Status |
| ---- | ------- | ------ |
| `include/dme/dme_copy.hpp` | 1D element + tile async copy API | **Validated** (MI300X, Run 4) |
| `include/dme/tensor_prefetch.hpp` | 2D strided tile prefetch + AsyncTile2D manager | Written (pending hardware run) |
| `tests/probe_dme.hip` | Empirical probe: offset, wait, collision semantics | **Validated** (Run 4) |
| `tests/bench_dme.hip` | Throughput: DME vs raw VMEM load | **Validated** (Run 4) |
| `tests/bench_dme_async.hip` | DME double-buffer overlap vs sync | **Validated** (Run 4, 1.02-1.04×) |

## Validated semantics (probe_dme.hip, gfx942, ROCm 7.2)

1. **Routing**: per-lane pointer advance is the correct model. `offset` must be 0.
2. **Offset**: must be a compile-time constant. Do NOT use for runtime routing.
3. **Size**: valid values are 1, 2, 4 bytes. size=8 is NOT supported on gfx942.
4. **Cache**: `aux=1` (sc0, streaming) bypasses L1/L2 — recommended for K/V tiles.
5. **Collision**: aliased LDS targets produce 0 (not undefined, not last-writer).
6. **Wait**: `__builtin_amdgcn_s_waitcnt(0)` is sufficient.

## API overview

### dme_copy.hpp — 1D API (validated)

```cpp
// Single async element (4-byte, per-lane)
dme::copy_element<float>(src_hbm + lane, &tile_lds[lane]);
dme::copy_element_stream<float>(...);   // sc0 streaming variant

// Cooperative tile (kElems fp32, multiple of 64)
dme::copy_tile_1d<float, kElems>(src_hbm, tile_lds);
dme::copy_tile_1d_stream<float, kElems>(...);

// Barriers
dme::wait();       // lgkmcnt=0 + vmcnt=0
dme::wait_lds();   // lgkmcnt=0 only
```

### tensor_prefetch.hpp — 2D API (written, pending validation)

```cpp
// Packed tile [kRows, kCols] fp16 — contiguous in HBM
dme::prefetch_tile2d_fp16<Bc, D>(base_hbm, dst_lds, tid, nthreads);

// Strided tile — rows separated by stride_row in HBM (KV-cache layout)
dme::prefetch_tile2d_strided_fp16<Bc, D>(base_hbm, dst_lds, stride_row, tid, nthreads);

// Double-buffer manager (RAII ping/pong)
dme::AsyncTile2D<Bc, D> tile;
tile.prologue(base_hbm, buf, tid, nthreads);
for (int j = 0; j < n_tiles; ++j) {
    tile.prefetch_next(base_hbm, buf, j+1, n_tiles, tid, nthreads);
    // ... compute on tile.cur() ...
    tile.wait_next(j+1, n_tiles);
    tile.advance();
}
```

## Integration

**Flash Attention (validated):** `fa_dme.hip` uses the 1D API directly. It achieves 1.18× vs naive attention (82.4 µs vs 97.2 µs, CDNA3 MI300X) with 44.7% fewer VMEM reads (rocprofv3 counters).

**cdna3 runtime (planned):** `tensor_prefetch.hpp` targets the KV-cache layout `[n_heads, max_seq, D]` used in `cdna3::runtime::Session`. The `prefetch_tile2d_strided_fp16` variant handles non-contiguous cache rows (stride_row = max_seq).

## Gap Reference

[Gap 7 — DME Abstraction](../../docs/gap-analysis.md)

This gap is partially closed: the 1D API is validated and integrated into Flash Attention. The 2D tensor API awaits hardware validation.
