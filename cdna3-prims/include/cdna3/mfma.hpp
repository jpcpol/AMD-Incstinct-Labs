// ---------------------------------------------------------------------------
// cdna3/mfma.hpp — typed accessors for the verified v_mfma_f32_16x16x16f16
// wave64 register-to-matrix lane mapping on gfx942 (MI300X / CDNA3).
//
// This is the one piece of the cdna3-prims facade that is NEW code (the wave and
// dme headers re-export existing validated libraries). It packages the empirically
// verified MFMA lane mapping (probe_mfma_mapping.hip — only one layout combination
// matched a full 16×16×16 GEMM-vs-reference at err 0.001) as named, typed helpers,
// so kernel authors don't reverse-engineer the ISA layout again.
//
// VERIFIED MAPPING (probe_mfma_mapping.hip, gfx942):
//   A  input : lane L, frag f → A[row = L%16      ][k   = (L/16)*4 + f]
//   B  input : lane L, frag f → B[k   = (L/16)*4+f][col = L%16        ]
//   C/D out  : lane L, frag f → C[row = (L/16)*4+f][col = L%16        ]
//
// A is read row-major over (row,k); B and the accumulator are read transposed over
// (k/row, col). This mapping is reusable for ANY fp16 16×16×16 MFMA kernel on gfx942.
//
// Header-only, gfx942-first. Falls back to a compile error on non-CDNA3 (the MFMA
// builtin is CDNA-specific); callers that need portability should guard with
// __gfx9__ at the call site.
// ---------------------------------------------------------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace cdna3 {
namespace mfma {

// Tile geometry of v_mfma_f32_16x16x16f16: 16×16 output, K=16, 4 fragments/lane.
static constexpr int M = 16;
static constexpr int N = 16;
static constexpr int K = 16;
static constexpr int FRAGS = 4;   // elements each lane holds of A, B, and C

using fp16x4 = __attribute__((ext_vector_type(4))) __fp16;
using f32x4  = __attribute__((ext_vector_type(4))) float;

__device__ __forceinline__ __fp16 to_fp16(__half h){ return *reinterpret_cast<__fp16*>(&h); }

// The fused MFMA: C[16×16] += A[16×16] · B[16×16], fp16 inputs, fp32 accumulate.
// a/b are this lane's 4 fragments in the verified input layout; c is the running
// accumulator in the verified output layout. Returns the updated accumulator.
__device__ __forceinline__ f32x4 mma(fp16x4 a, fp16x4 b, f32x4 c){
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}

// --- typed loaders: pack a lane's 4 fragments from a tile in LDS/registers ---------
// These encapsulate the verified index math so callers never write (L/16)*4+f again.

// A operand: lane L wants A[row=L%16][k=(L/16)*4+f] for f in [0,4).
// `A_tile` is row-major [M][K] (ldA = K by default).
__device__ __forceinline__ fp16x4 load_A(const __half* A_tile, int lane, int ldA = K){
    fp16x4 a;
    int row = lane % 16;
    int k0  = (lane / 16) * 4;
    #pragma unroll
    for (int f = 0; f < 4; ++f) a[f] = to_fp16(A_tile[row * ldA + (k0 + f)]);
    return a;
}

// B operand: lane L wants B[k=(L/16)*4+f][col=L%16] for f in [0,4).
// `B_tile` is row-major [K][N] (ldB = N by default).
__device__ __forceinline__ fp16x4 load_B(const __half* B_tile, int lane, int ldB = N){
    fp16x4 b;
    int col = lane % 16;
    int k0  = (lane / 16) * 4;
    #pragma unroll
    for (int f = 0; f < 4; ++f) b[f] = to_fp16(B_tile[(k0 + f) * ldB + col]);
    return b;
}

// C/D output: lane L holds C[row=(L/16)*4+f][col=L%16] for f in [0,4).
// Store this lane's 4 accumulator fragments back to a row-major [M][N] tile.
__device__ __forceinline__ void store_C(float* C_tile, f32x4 c, int lane, int ldC = N){
    int col = lane % 16;
    int r0  = (lane / 16) * 4;
    #pragma unroll
    for (int f = 0; f < 4; ++f) C_tile[(r0 + f) * ldC + col] = c[f];
}

// Same store but emitting fp16 (common for chained matmuls, e.g. P·V after Q·Kᵀ).
__device__ __forceinline__ void store_C_fp16(__half* C_tile, f32x4 c, int lane, int ldC = N){
    int col = lane % 16;
    int r0  = (lane / 16) * 4;
    #pragma unroll
    for (int f = 0; f < 4; ++f) C_tile[(r0 + f) * ldC + col] = __float2half(c[f]);
}

} // namespace mfma
} // namespace cdna3
