// ---------------------------------------------------------------------------
// cdna3/dme.hpp — facade re-export of the validated DME async-copy abstraction.
//
// Re-exports dme::* (the Data Movement Engine C++ wrapper) under the cdna3 facade.
// Includes the existing validated header (research/dme-abstraction) — one source of
// truth, no fork.
//
// What this gives you (validated on MI300X — DME async double-buffer in Flash
// Attention delivered −44.7% SQ_INSTS_VMEM_RD, the hardware signature of overlap):
//   dme::copy_element<T>(dst, src)             — single async element copy
//   dme::copy_element_stream<T>(dst, src)      — streaming-cache variant (sc0=1)
//   dme::copy_tile_1d<T,kElems>(dst, src)      — async tile copy
//   dme::copy_tile_1d_stream<T,kElems>(...)    — streaming tile copy
//   dme::wait()  /  dme::wait_lds()            — completion barriers
//
// The validated usage pattern: prefetch K_{j+1}/V_{j+1} via copy_*_stream while the
// current tile's MFMA accumulation runs, then wait_lds() before consuming. See
// docs/composition.md (async_tiled_gemm example) for the FA inner-loop pattern.
// ---------------------------------------------------------------------------
#pragma once

// Facade path relative to cdna3-prims/include/cdna3/ → ../../../research/dme-abstraction/include.
#include "../../../research/dme-abstraction/include/dme/dme_copy.hpp"

// Names stay in the dme:: namespace — the facade re-exposes, it does not rename.
