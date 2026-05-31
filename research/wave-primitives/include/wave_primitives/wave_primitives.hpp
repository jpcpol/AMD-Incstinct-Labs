#pragma once

// ---------------------------------------------------------------------------
// wave_primitives — umbrella header
//
// Header-only library for wave32/wave64-agnostic wave-level operations.
// Targets AMD Instinct CDNA (wave64) and RDNA / NVIDIA (wave32).
//
// Requires: ROCm 6.2+ (ROCm 7+ recommended for warpSize early-fold)
//           hipcc / amdclang++ --offload-arch=gfx942 (MI300X)
//
// Usage:
//   #include "wave_primitives/wave_primitives.hpp"
//
//   __global__ void my_kernel(float* out, const float* in, int n) {
//       float val = (blockIdx.x * blockDim.x + threadIdx.x < n)
//                   ? in[blockIdx.x * blockDim.x + threadIdx.x] : 0.0f;
//       float sum = wave::reduce_sum(val);     // correct on wave32 and wave64
//       if (wave::lane_id() == 0) out[blockIdx.x] = sum;
//   }
// ---------------------------------------------------------------------------

#include "wave_shuffle.hpp"
#include "wave_ballot.hpp"
#include "wave_reduce.hpp"
#include "wave_scan.hpp"
