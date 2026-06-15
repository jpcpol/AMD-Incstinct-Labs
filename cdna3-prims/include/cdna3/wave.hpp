// ---------------------------------------------------------------------------
// cdna3/wave.hpp — facade re-export of the validated wave-primitives library.
//
// Re-exports wave::dpp::* (zero-LDS DPP reductions/scans) and the portable wave::*
// fallbacks under the cdna3 facade. This does NOT fork the library — it includes the
// existing validated headers (research/wave-primitives), so there is one source of
// truth. The facade exists so a kernel author writes a single `#include <cdna3/cdna3.hpp>`
// and gets the whole composable primitive set.
//
// What this gives you (validated on MI300X, rocprofv3-verified):
//   wave::dpp::reduce_sum/max/min(v)        — zero ds_bpermute, 1.35–1.79× vs hipCUB
//   wave::dpp::reduce_*_bcast(v)            — result broadcast to all lanes
//   wave::dpp::scan_inclusive_sum(v)        — zero-LDS inclusive scan, 1.028× vs hipCUB
//   wave::dpp::scan_exclusive_sum(v)
//   wave::reduce_*/scan_* (portable)        — wave32/wave64-agnostic __shfl fallback
//
// See docs/composition.md for the rocprofv3 evidence (SQ_INSTS_LDS = 0) and when the
// DPP advantage applies (reduction-dominated kernels; collapses to 1–3% memory-bound).
// ---------------------------------------------------------------------------
#pragma once

// The validated wave-primitives library lives in its own research area. The facade
// path is relative to cdna3-prims/include/cdna3/ → ../../../research/wave-primitives/include.
#include "../../../research/wave-primitives/include/wave_primitives/wave_primitives.hpp"
#include "../../../research/wave-primitives/include/wave_primitives/wave_reduce_dpp.hpp"
#include "../../../research/wave-primitives/include/wave_primitives/wave_scan_dpp.hpp"

// Note: the umbrella above pulls reduce/scan/shuffle/ballot; the two _dpp headers add
// the CDNA3-optimized zero-LDS paths explicitly. All names stay in their original
// namespaces (wave::, wave::dpp::) — the facade re-exposes, it does not rename.
