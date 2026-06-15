// ---------------------------------------------------------------------------
// cdna3/cdna3.hpp — umbrella header for the cdna3-prims facade.
//
// One include gives a ROCm kernel author the full set of validated CDNA3 primitives
// that the current software ecosystem leaves on the table:
//
//   #include <cdna3/cdna3.hpp>
//
//   wave::dpp::reduce_sum(v)            // zero-LDS cross-lane reduction (gfx9)
//   dme::copy_tile_1d_stream<T,N>(...)  // async HBM→LDS movement off the CUs
//   cdna3::mfma::mma(a, b, c)           // verified 16×16×16 matrix-core tile
//   cdna3::mfma::load_A / load_B / store_C   // the verified lane mapping, typed
//
// Design: this is a FACADE above the independent research areas, not a cross-
// dependency between them. It re-exports validated libraries (wave-primitives,
// dme-abstraction) and adds typed MFMA accessors (the one new piece). Each research
// area stays independently buildable (project invariant); cdna3-prims is the
// integration layer a third party includes.
//
// Header-only · gfx942-first · ROCm 6.2+ (7+ recommended). On non-CDNA3 targets the
// wave primitives fall back to portable __shfl paths; the dme and mfma pieces are
// CDNA3-specific (guard with __gfx9__ at the call site if portability is needed).
//
// Evidence and composition patterns: docs/composition.md.
// Status: Stage B (attention module) — adds cdna3::attn::{prefill,decode} above
// the validated primitives. Include <cdna3/attention.hpp> explicitly to get the
// attention module (it includes the kernel .hip files; not included here to avoid
// pulling main() into every consumer). See README.md for the A→B→C roadmap.
// ---------------------------------------------------------------------------
#pragma once

#include "wave.hpp"   // wave::dpp::* (validated, zero-LDS) + portable fallbacks
#include "dme.hpp"    // dme::* async copy (validated, −44.7% VMEM reads in FA)
#include "mfma.hpp"   // cdna3::mfma::* verified 16×16×16 lane mapping (new, typed)
// attention.hpp — cdna3::attn::{prefill,decode}  → include separately (pulls kernel .hip)

namespace cdna3 {

// Library version. Bump on API changes.
static constexpr int VERSION_MAJOR = 0;
static constexpr int VERSION_MINOR = 2;   // Stage B: attention module added

} // namespace cdna3
