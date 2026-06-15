# cdna3-prims — Composition Guide & Evidence

How the three validated primitives compose into real CDNA3 kernels, and the
hardware-counter evidence behind each. Every number here is measured on MI300X VF
(gfx942, ROCm 7.2) and traces back to the project's papers and results files — this
facade adds packaging, not new claims.

---

## The three primitives

| Primitive | Header | What it gives you | Evidence |
|-----------|--------|-------------------|----------|
| **DPP wave reduction** | `cdna3/wave.hpp` | zero-LDS cross-lane reduce/scan | `SQ_INSTS_LDS = 0` (rocprofv3) |
| **DME async copy** | `cdna3/dme.hpp` | HBM→LDS movement off the CUs | −44.7% `SQ_INSTS_VMEM_RD` in FA |
| **MFMA tile** | `cdna3/mfma.hpp` | verified 16×16×16 lane mapping | only-matching layout @ err 0.001 |

---

## 1. DPP wave reduction — the zero-LDS evidence

Generic `__shfl_*` lowers to `ds_bpermute` on CDNA3 — an LDS round-trip per step. The
DPP path lowers to the cross-lane VALU datapath, zero LDS:

| Kernel | SQ_INSTS_LDS | vs hipCUB |
|--------|--------------|-----------|
| `wave::dpp::reduce_sum` | **0** | 1.35–1.79× faster |
| `wave::reduce_sum` (`__shfl`) | 25.2M | 0.41× |
| `hipCUB WarpReduce::Sum` | 4.2M | 1.00× |
| `wave::dpp::scan_inclusive_sum` | **0** | 1.028× |

**When it pays:** reduction-dominated kernels — short rows, multiple reductions per
row, register-bound kernels with a tight LDS budget. **When it doesn't:** memory-bound
kernels (wide-row LayerNorm/Softmax) where HBM traffic dominates — the margin collapses
to 1–3% (still fastest, still zero-LDS, but not the headline). Honest boundary, stated
so you choose the primitive deliberately.

The zero-LDS property is **wave-scoped**: block-level reductions that need an inter-wave
LDS gather match hipCUB exactly (1.000×). cdna3-prims gives you the wave-level zero-LDS
path; above one wave you pay the same LDS round-trip everyone does.

## 2. DME async copy — the overlap evidence

DME moves data HBM→LDS without occupying the CUs. The win is **overlap**: prefetch the
next tile while computing the current one. In Flash Attention:

| Kernel | SQ_INSTS_VMEM_RD |
|--------|------------------|
| `fa_naive` | 311,296 |
| `fa_dme` (DME prefetch) | **172,032 (−44.7%)** |

The −44.7% is the hardware signature of the load/compute overlap. The pattern is in
`examples/async_tiled_gemm.hpp`. The two rules that preserve the overlap: (1) issue the
prefetch for tile t+1 *before* computing tile t; (2) `wait_lds()` *after* the compute,
just before consuming — earlier serializes and erases the gain.

## 3. MFMA tile — the verified mapping

The register-to-matrix lane mapping of `v_mfma_f32_16x16x16f16` is not obvious from the
ISA docs. It was determined empirically (`probe_mfma_mapping.hip`: a full 16×16×16
GEMM-vs-reference enumerating all layout combinations; only one matched at err 0.001):

```
A  in : lane L, frag f → A[row = L%16      ][k   = (L/16)*4 + f]
B  in : lane L, frag f → B[k   = (L/16)*4+f][col = L%16        ]
C/out : lane L, frag f → C[row = (L/16)*4+f][col = L%16        ]
```

`cdna3::mfma::load_A / load_B / store_C` encapsulate this index math so you never write
`(L/16)*4+f` again. `cdna3::mfma::mma(a,b,c)` is the fused matrix-core call.

**MFMA boundary (honest):** at small head dim (D=64) the 16×16×16 tiles are too small to
saturate the cores — the dot-product path can win (paper §5.8). At D=128 the matrix
cores reach 10.45 TFLOPS vs 6.19 at D=64 (+69%). Use MFMA when the tiles are full.

---

## 4. How they compose — the canonical CDNA3 kernel

The three primitives compose into the pattern that defines a well-formed CDNA3 kernel:

```
  ┌─ dme::copy_tile_1d_stream  (prefetch next tile, async, off the CUs)
  │     ║  overlap window
  ├─ cdna3::mfma::mma          (matrix-core accumulate on current tile)
  │     ║
  ├─ cdna3::ex::online_softmax_step  (zero-LDS DPP reduction for the rescale)
  └─ dme::wait_lds             (consume the prefetched tile)
```

This is exactly the Flash Attention inner loop (paper §5.8): DME prefetch + MFMA +
DPP softmax. Each primitive removes a different bottleneck — DPP removes LDS contention
on the cross-lane reduction, DME removes the VMEM stall on the tile load, MFMA does the
matmul on the matrix cores. The validated FA kernel combined them for an 18% end-to-end
speedup (97.2 → 82.4 µs) with each contribution isolated by counters.

`examples/fused_softmax.hpp` and `examples/async_tiled_gemm.hpp` package the softmax and
the prefetched-matmul halves of this loop as reusable building blocks.

---

## 5. Scope & status

- **Stage 1 (this):** facade over the three validated primitives + composition examples.
  Rests only on hardware-validated code.
- **Stage 3 (later):** an `attention` module, after the multi-wave and decode kernels
  are validated on a VM session (see `docs/framework-vision.md`).
- Header-only, gfx942-first, ROCm 6.2+. Non-CDNA3: wave primitives fall back to portable
  `__shfl`; dme/mfma are CDNA3-specific (guard with `__gfx9__`).

Source evidence: `docs/paper-draft.md` §5.2/§5.8, `research/*/results/`.
