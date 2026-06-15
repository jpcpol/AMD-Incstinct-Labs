# cdna3-prims

Composable, header-only CDNA3 (AMD Instinct MI300X / gfx942) primitives that expose the
hardware the current ROCm software ecosystem leaves on the table: **zero-LDS DPP
cross-lane reductions**, **DME async data movement**, and the **verified MFMA 16×16×16
lane mapping** — behind one clean include.

```cpp
#include <cdna3/cdna3.hpp>

wave::dpp::reduce_sum(v);                  // zero ds_bpermute, 1.35–1.79× vs hipCUB
dme::copy_tile_1d_stream<__half, N>(...);  // async HBM→LDS, off the CUs
cdna3::mfma::mma(a, b, c);                 // verified 16×16×16 matrix-core tile
```

This is the **framework consolidation** of the AMD-Instinct Labs research: findings that
were separate artifacts (a wave-primitives library, a DME wrapper, an FA kernel) become
composable infrastructure a third-party ROCm kernel can include. It is a **facade above**
the independent research areas, not a fork of them — there is one source of truth per
primitive.

## What's validated (Stage 1 rests only on this)

| Primitive | Evidence (MI300X, gfx942, ROCm 7.2) |
|-----------|--------------------------------------|
| DPP wave reduce/scan | `SQ_INSTS_LDS = 0` (rocprofv3); 1.35–1.79× vs hipCUB |
| DME async copy | −44.7% `SQ_INSTS_VMEM_RD` in Flash Attention |
| MFMA lane mapping | only-matching layout @ err 0.001 (full GEMM probe) |

See [`docs/composition.md`](docs/composition.md) for the full evidence and the canonical
compose-into-a-kernel pattern.

## Layout

```
cdna3-prims/
  include/cdna3/
    wave.hpp     — re-exports wave::dpp::* (validated)
    dme.hpp      — re-exports dme::*        (validated)
    mfma.hpp     — typed 16×16×16 tile accessors (the verified mapping; new)
    cdna3.hpp    — umbrella include
  examples/
    fused_softmax.hpp     — DPP row-softmax building block
    async_tiled_gemm.hpp  — DME-prefetch + MFMA tile-loop building block
  docs/composition.md     — evidence + composition guide
  tests/compile_smoke.hip — header include + API smoke (compile-only)
```

Header-only · gfx942-first · ROCm 6.2+ (7+ recommended). Non-CDNA3 targets fall back to
portable `__shfl` for the wave primitives; dme/mfma are CDNA3-specific.

---

## The framework roadmap — closing the A → B → C cycle

The end goal is a complete framework with an LLM runtime ready to use. Three stages,
each gated on the previous being **validated on hardware** — never built on unproven
ground (the discipline the whole project has used).

### Stage A — Composable primitives library  ·  **IN PROGRESS (cold)**

The facade + examples + docs in this directory. Rests only on the three validated
primitives. Fully buildable and useful with no further GPU time.

- [x] `cdna3/` facade headers (wave, dme, mfma, umbrella)
- [x] composition examples (fused_softmax, async_tiled_gemm)
- [x] `docs/composition.md` with the rocprofv3 evidence
- [x] compile-smoke test
- [ ] **(VM, small)** compile-validate the smoke test + examples on 1×MI300X
- [ ] **(cold)** CI-style promotion of the existing probes into a correctness suite
- [ ] **(cold)** upstream/community write-up of the primitives facade

**Exit criterion for A:** the facade compiles and the examples run on MI300X; the
primitives' evidence is reproducible from the test suite.

### Stage B — Modular attention module  ·  **PLANNED (gated on A + kernel validation)**

Add an `attention` module to the library: a configurable FA for CDNA3 (prefill/decode,
GQA, causal) built FROM the Stage-A primitives, with DPP/DME/MFMA as the swappable
strategies. This is where the written-but-unvalidated kernels earn their place.

Roadmap (each step gated):
1. **(VM)** Validate `fa_multiwave.hip` (occupancy fix) on 1×MI300X — H-MW1/2/3.
2. **(VM)** Validate `fa_decode.hip` + the decode torch op — H-DEC1/2/3.
3. **(cold)** Refactor the two validated kernels to consume `cdna3::*` primitives
   (so the attention module IS a composition of Stage A, not a parallel codebase).
4. **(cold)** `cdna3/attention.hpp`: a unified entry point — `cdna3::attn::prefill(...)`
   and `cdna3::attn::decode(...)` — selecting kernel + strategy from config.
5. **(VM)** End-to-end re-validation: the attention module reproduces 2-A (full Qwen2.5,
   top-1 preserved) through the new unified API.
6. **(cold)** Benchmark module vs production SDPA (extend `compare_vs_production_fa.py`);
   report the gap honestly, track multi-wave's narrowing of it.

**Exit criterion for B:** `cdna3::attn::{prefill,decode}` runs a real LLM end-to-end via
the unified API, validated on hardware, with the occupancy gap measured.

### Stage C — Minimal LLM inference runtime  ·  **TENTATIVE (gated on B validated)**

Wrap the validated attention module into a small, honest research inference runtime for
MI300X — the closing of the cycle: a usable framework, not a kernel collection.

Tentative roadmap (revisit after B; scope deliberately modest vs vLLM):
1. **(cold)** `runtime/` design note: scope (greedy/batch-1 first), KV-cache manager
   (contiguous fp16; paged is future), the layer loop (attention via Stage B + MLP).
2. **(cold)** Model loader: reuse the 2-A HF-weight export path; support the small
   open-weight GQA models already validated (Qwen2.5, Llama-3.2-1B class).
3. **(VM)** Prefill+decode generation loop on the validated attention module; validate
   the generated sequence vs HF greedy (the 2-A/decode criteria, now at runtime level).
4. **(cold)** A minimal serving API (single-stream generate) + a benchmark harness:
   tokens/sec, time-to-first-token, vs a production baseline — framed research-vs-prod
   (the 2-C discipline), not a vLLM-beater claim.
5. **(cold/VM)** Optional: batched decode, paged KV-cache — explicitly future work,
   only if the single-stream runtime proves the core out.

**Exit criterion for C (the cycle closed):** a documented, header-+-runtime framework
where `cdna3`-powered attention runs a real small LLM end-to-end (prefill→decode→tokens)
through a clean API, with honest benchmarks vs production, on validated hardware.

### Why this ordering (the discipline)

Each stage rests on the previous being **validated**, never assumed. A is buildable now
from proven code. B absorbs the pending kernel validation as library modules (turning
"validate two loose kernels" into "validate the attention module"). C wraps a validated B
into a runtime. The cycle closes with a usable framework — and at no point is a layer
built on unproven ground, the failure mode the project has consistently avoided.

CAL-L4 note: this framework is an AMD-side artifact. CAL-L4 needs the *data* (delivered)
and, far out, an RCC proxy; the framework is not on the CAL critical path. See
`docs/framework-vision.md` §6.

---

## Out of scope — G3 Infinity Fabric AllReduce

The original research plan included a G3 area (topology-aware AllReduce over the
MI300X Infinity Fabric full-mesh, 8×GPU). **This is permanently out of scope:**
8×MI300X instances are not available on the AMD Developer Cloud and are not
expected to become available. The single-GPU framework (A→B→C) is the complete
research artifact. Multi-GPU AllReduce remains an open gap in the ROCm ecosystem
and is documented as such in `docs/gap-analysis.md`.
