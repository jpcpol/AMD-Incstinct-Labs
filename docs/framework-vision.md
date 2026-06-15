# Framework Vision — from research findings to composable CDNA3 infrastructure

**Status:** design / vision note (cold, 2026-06). No code committed under this yet.
**Purpose:** decide whether — and in what form — the validated findings of this project
should consolidate into a reusable framework, before writing any of it.

---

## 0. The question this note answers

The project has produced findings that today live as **separate artifacts**: a
wave-primitives library, a Flash Attention kernel, a DME wrapper, a verified MFMA lane
mapping. Each proves a finding. None of them is, on its own, something an external ROCm
developer can *compose* into their own kernel without copy-pasting.

A framework would turn findings into **infrastructure**. The question is not "should we
build more" — it is "is there a backbone that unifies what we have, and what is the
minimal, honest form of it given what is actually validated."

This note evaluates three candidate forms against the validated base and recommends one
with a staged roadmap. It deliberately does NOT commit code — that follows, gated on the
decision here and on validation of the pieces each form depends on.

## 1. Inventory: what is validated vs. what is written

The single most important discipline for a framework is to build only on validated
ground. Mixing in unvalidated kernels repeats the error the CAL-L3 work punished
(building the operator before understanding the object).

| Artifact | Status | Framework-ready? |
|----------|--------|------------------|
| **Wave-primitives DPP** (zero-LDS reduce/scan, 1.35–1.79× vs hipCUB) | ✅ Validated on MI300X, rocprofv3-verified, header-only | **Yes** — the strongest piece |
| **DME async abstraction** (`dme_copy.hpp`, −44.7% VMEM reads) | ✅ Validated semantics on MI300X | **Yes** |
| **Verified MFMA lane mapping** (`v_mfma_f32_16x16x16f16` wave64) | ✅ Empirically determined | **Yes** — a reusable fact |
| **Flash Attention prefill** (fa_robust + 2-A/2-B/2-C) | ✅ Validated, runs a real LLM | **Yes**, as a reference app |
| Multi-wave FA (`fa_multiwave.hip`) | ⚠️ Written, NOT validated | No — gated on VM |
| Decode/KV-cache (`fa_decode.hip`) | ⚠️ Written, NOT validated | No — gated on VM |
| Infinity Fabric AllReduce (G3) | ⬜ Not started | No — needs 8×GPU |

**The validated base is the three low-level primitives (DPP, DME, MFMA-mapping) plus a
reference FA app.** Any framework must rest on these, not on the unvalidated kernels.

## 2. The three candidate forms

### Option A — Composable CDNA3 primitives library (lowest level)

Package the validated primitives as a header-only, dependency-free library that any HIP
kernel includes: `wave::dpp::*` (already exists), `dme::copy_*` (exists), and the MFMA
lane-mapping helpers as typed tile accessors. The unifying idea: **expose exactly the
CDNA3 hardware the current software ecosystem leaves on the table** — DPP cross-lane
without LDS, DME async movement, correct MFMA tiling — behind one clean, documented API.

- **Rests on:** only validated pieces. Lowest risk.
- **Audience:** ROCm kernel authors (the people in the AMD community thread).
- **Adoption story:** "drop-in primitives; your kernel keeps zero-LDS cross-lane and
  async DME without reverse-engineering the ISA." Upstreamable.
- **Effort:** moderate — mostly consolidation + docs + tests of existing code.
- **Risk:** low. The pieces are proven; the work is packaging and API design.

### Option B — Modular Flash Attention framework (mid level)

A configurable FA for CDNA3: prefill/decode, GQA, causal, with DPP/DME/MFMA as
swappable strategies — a research-grade mini-FlashAttention.

- **Rests on:** multi-wave + decode (BOTH unvalidated). Higher risk.
- **Audience:** narrower — people who want an FA they can configure on CDNA3.
- **Effort:** high — requires the unvalidated kernels to pass first.
- **Risk:** medium-high. Gated on VM validation that hasn't happened.

### Option C — Minimal LLM inference runtime (highest level)

A small inference runtime (torch op + KV-cache + decode) as a research alternative to
vLLM on MI300X.

- **Rests on:** decode + cache management + scheduling, mostly unvalidated, and competes
  with a heavily-tuned production system (2-C already showed an 8.5–15× gap).
- **Audience:** broad but already served by vLLM/SGLang.
- **Effort:** very high. Risk: high — maximum surface, least validated, hardest to
  differentiate honestly.

## 3. Recommendation: Option A, staged toward B

**Build Option A now.** It is the only form that rests entirely on validated ground, it
matches the project's existing invariants (header-only, gfx942-first, benchmark-backed),
and it is the honest packaging of what the research actually proved. It is also the form
with the clearest upstream/adoption path — the AMD community contacts are exactly its
audience.

**Treat B as the second stage, gated on validating multi-wave + decode.** Once those
kernels pass on a 1×MI300X session, the FA pieces become library-quality and Option A
naturally grows an `attention` module — that is B, earned rather than assumed. C stays
out of scope: it competes with production systems where 2-C already measured the gap
honestly, and the differentiation would be thin.

This mirrors the discipline the whole project has used: ship the validated core, declare
the rest as staged/gated, never over-claim.

## 4. Proposed architecture (Option A)

A single header-only library — call it **`cdna3-prims`** (name TBD) — layered as a new
facade ABOVE the existing research areas, NOT as a cross-dependency between them (which
would violate the project invariant that research areas stay independent). It re-exports
the validated primitives under one coherent namespace and documents how they compose.

```
cdna3-prims/                       (new: a facade + docs + composition examples)
  include/cdna3/
    wave.hpp        → re-exports wave::dpp::* (from research/wave-primitives)
    dme.hpp         → re-exports dme::copy_*  (from research/flash-attention's dme_copy)
    mfma.hpp        → typed 16×16×16 tile accessors built on the verified lane mapping
    cdna3.hpp       → umbrella include
  examples/
    fused_softmax.hpp     → DPP softmax, the §5.6 pattern, as a reusable building block
    async_tiled_gemm.hpp  → DME prefetch + MFMA, the FA inner-loop pattern, generalized
  docs/
    composition.md  → how the primitives compose; the rocprofv3 evidence per primitive
  tests/            → the existing probes, promoted to a CI-style correctness suite
```

Key property: the library **does not move or fork** the research-area code. It is a thin,
documented facade with composition examples. The research areas remain independent
(invariant preserved); the framework is the integration layer the user includes.

## 5. What this is NOT (scope discipline)

- ❌ NOT a vLLM competitor (Option C) — out of scope, gap measured honestly in 2-C.
- ❌ NOT built on unvalidated kernels — multi-wave/decode join only after they pass.
- ❌ NOT a cross-dependency between research areas — it is a facade above them.
- ❌ NOT a claim of production readiness — it is research-grade composable primitives,
  with the same honest-boundaries framing as the papers.

## 6. Relationship to the papers and CAL

- **AMD paper (cs.DC):** the framework is the *artifact* the paper describes — "here is
  the library that packages these primitives." It strengthens the paper's reach claim
  (§8: "the fix is local and composable") by making "composable" literally true.
- **CAL-L4:** the framework has **no direct role** in the CAL-L4 roadmap. CAL needs the
  *data* (the O(n²) baseline, done) and eventually the RCC proxy (far future). The
  framework is an AMD-side deliverable. The one thin thread: a clean DME/MFMA facade
  would be the substrate IF the RCC's V-extraction-in-prefill is ever built — but that is
  speculative and not a reason to build the framework now.

## 7. Cold roadmap (no GPU needed for stage 1)

| Stage | Work | Needs GPU? |
|-------|------|------------|
| **1 — Facade + docs** | `cdna3/` headers re-exporting validated primitives; composition.md with the existing rocprofv3 evidence; promote probes to a test suite | No (consolidation of validated code) |
| **2 — Examples** | `fused_softmax.hpp`, `async_tiled_gemm.hpp` as reusable building blocks extracted from the validated FA | No (refactor of validated code) |
| **3 — Validate-then-absorb** | After a VM session validates multi-wave + decode, add an `attention` module (→ Option B) | Yes (the pending VM validation) |
| **4 — Upstream** | Propose the primitives facade to the ROCm community / as a paper artifact | No |

Stages 1–2 and 4 are **fully cold** — real work with real return, no GPU. Stage 3 is the
natural home for the pending VM validation, turning it from "validate two loose kernels"
into "validate two kernels that then become a library module" — more motivating and more
useful.

## 8. Decision requested

Adopt Option A (staged toward B), build Stage 1 cold now? Or revisit the form.
