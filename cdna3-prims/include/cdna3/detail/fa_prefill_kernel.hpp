// ---------------------------------------------------------------------------
// cdna3/detail/fa_prefill_kernel.hpp — multi-wave prefill kernel (header-only).
//
// Contains ONLY __global__ template fa_multiwave_kernel<D,W> and the minimal
// supporting types. No main(), no launcher, no benchmark harness.
// Single source of truth; standalone driver (fa_multiwave.hip) includes this.
//
// Consumed by cdna3/attention.hpp (Stage-B unified API).
//
// MFMA access: fragments a/b are assembled directly from Q_lds/K_lds using the
// verified lane mapping (no extra sub-tile copies). cdna3::mfma::mma() provides
// the matrix-core intrinsic; the load index math follows the verified mapping:
//   A: q_row = L%16, k_base = (L/16)*4 + f  → Q_lds[wave][q_row*D + k_base]
//   B: col   = jc*MT + (L%MT), k_base = kd*MT + (L/MT)*4 + f → K_lds[col*D + k_base]
// ---------------------------------------------------------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../cdna3.hpp"   // wave::dpp::*, cdna3::mfma::*

using wave::dpp::reduce_max_bcast;
using wave::dpp::reduce_sum_bcast;
using _fp16x4 = cdna3::mfma::fp16x4;
using _f32x4  = cdna3::mfma::f32x4;

__device__ __forceinline__ __fp16 _h2f16(__half h){ return *reinterpret_cast<__fp16*>(&h); }

static constexpr int _Br = 16;
static constexpr int _Bc = 64;
static constexpr int _MT = 16;   // MFMA tile edge = cdna3::mfma::M = 16

struct PrefillKCfg { int D, seqQ, seqK, nQHeads, nKVHeads, batch, causal; };

// ---------------------------------------------------------------------------
// fa_multiwave_kernel<D, W>
//   W waves per block; each wave owns Br=16 query rows.
//   K/V tiles (Bc×D) loaded cooperatively by ALL W*64 lanes, reused by all waves.
//   Softmax: full 64-lane DPP reduce per row (zero LDS, via wave::dpp::*).
//   Q·Kᵀ / P·V: MFMA via cdna3::mfma::mma; fragments built from LDS directly
//   using the verified lane-index math (no extra sub-tile staging).
// ---------------------------------------------------------------------------
template<int D, int W>
__global__ void fa_multiwave_kernel(const __half* __restrict__ Q, const __half* __restrict__ K,
                                    const __half* __restrict__ V, __half* __restrict__ O,
                                    PrefillKCfg c)
{
    constexpr int nD  = D / _MT;    // number of 16-wide k-slices to cover D
    constexpr int nBc = _Bc / _MT;  // number of 16-wide key-column groups to cover Bc
    const int wave = threadIdx.x / 64;
    const int L    = threadIdx.x % 64;
    const int qblocks = c.seqQ / (W * _Br);

    const int sblk  = blockIdx.x % qblocks;
    const int qhead = (blockIdx.x / qblocks) % c.nQHeads;
    const int batch = blockIdx.x / qblocks / c.nQHeads;
    const int group  = c.nQHeads / c.nKVHeads;
    const int kvhead = qhead / group;
    const int qrow0  = sblk * (W * _Br) + wave * _Br;

    const __half* Qblk  = Q + (((size_t)batch*c.nQHeads + qhead)*c.seqQ + qrow0)*D;
    const __half* Kbase = K + ((size_t)batch*c.nKVHeads + kvhead)*c.seqK*D;
    const __half* Vbase = V + ((size_t)batch*c.nKVHeads + kvhead)*c.seqK*D;
    __half*       Oblk  = O + (((size_t)batch*c.nQHeads + qhead)*c.seqQ + qrow0)*D;

    __shared__ __half K_lds[_Bc*D];
    __shared__ __half V_lds[_Bc*D];
    __shared__ __half Q_lds[W][_Br*D];
    __shared__ __half S_lds[W][_Br*_Bc];  // half saves 8192B at W=4,D=64 (66KB→58KB)
    __shared__ __half P_lds[W][_Br*_Bc];
    __shared__ float  O_lds[W][_Br*D];
    __shared__ float  m_lds[W][_Br];
    __shared__ float  l_lds[W][_Br];

    for (int i=L; i<_Br*D; i+=64) Q_lds[wave][i] = Qblk[i];
    for (int i=L; i<_Br*D; i+=64) O_lds[wave][i] = 0.f;
    if (L < _Br){ m_lds[wave][L] = -1e38f; l_lds[wave][L] = 0.f; }
    const float scale = 1.f / sqrtf((float)D);
    const int last_q = qrow0 + _Br - 1;
    int kv_tiles = c.seqK / _Bc;
    if (c.causal){ int mt = (last_q / _Bc) + 1; if (mt < kv_tiles) kv_tiles = mt; }
    __syncthreads();

    for (int kv = 0; kv < kv_tiles; ++kv){
        // All W*64 lanes cooperatively load K and V for this tile.
        const __half* Kt = Kbase + (size_t)kv*_Bc*D;
        const __half* Vt = Vbase + (size_t)kv*_Bc*D;
        for (int i = threadIdx.x; i < _Bc*D; i += W*64){ K_lds[i] = Kt[i]; V_lds[i] = Vt[i]; }
        __syncthreads();

        // S[wave][row][col] = (Q[row] · K[col]) * scale  — per-wave MFMA.
        // Verified lane mapping (no extra staging):
        //   A frag: Q_lds[wave][q_row * D + kd*_MT + (L/_MT)*4 + f]   q_row = L % _MT
        //   B frag: K_lds[key   * D + kd*_MT + (L/_MT)*4 + f]         key = jc*_MT + (L%_MT)
        //   C out : S_lds[wave][(L/_MT)*4+f * _Bc + jc*_MT + (L%_MT)]
        #pragma unroll
        for (int jc = 0; jc < nBc; ++jc){
            _f32x4 acc = {0,0,0,0};
            #pragma unroll
            for (int kd = 0; kd < nD; ++kd){
                _fp16x4 a, b;
                int q_row = L % _MT;
                int k_base = kd * _MT + (L / _MT) * 4;
                #pragma unroll
                for (int f = 0; f < 4; ++f) a[f] = _h2f16(Q_lds[wave][q_row*D + k_base + f]);
                int key = jc * _MT + (L % _MT);
                #pragma unroll
                for (int f = 0; f < 4; ++f) b[f] = _h2f16(K_lds[key*D + k_base + f]);
                acc = cdna3::mfma::mma(a, b, acc);
            }
            #pragma unroll
            for (int f = 0; f < 4; ++f){
                int sr = (L / _MT)*4 + f;
                int sc = jc * _MT + (L % _MT);
                float s = acc[f] * scale;
                if (c.causal){ int gq = qrow0+sr, gk = kv*_Bc+sc; if (gk > gq) s = -1e30f; }
                S_lds[wave][sr*_Bc + sc] = __float2half(s);
            }
        }
        __syncthreads();

        // Softmax via cdna3-prims: zero-LDS DPP broadcast reduce (all 64 lanes per row).
        // Running (m,l) lives in LDS; lane 0 commits the scalar update and O rescale.
        #pragma unroll
        for (int row = 0; row < _Br; ++row){
            float s    = __half2float(S_lds[wave][row*_Bc + L]);
            float rmax = reduce_max_bcast(s);
            float mold = m_lds[wave][row];
            float mnew = fmaxf(mold, rmax);
            float p    = __expf(s - mnew);
            float rsum = reduce_sum_bcast(p);
            P_lds[wave][row*_Bc + L] = __float2half(p);
            if (L == 0){
                float resc = __expf(mold - mnew);
                for (int d = 0; d < D; ++d) O_lds[wave][row*D + d] *= resc;
                m_lds[wave][row] = mnew;
                l_lds[wave][row] = resc * l_lds[wave][row] + rsum;
            }
        }
        __syncthreads();

        // O += P·V  — per-wave MFMA, same verified mapping.
        //   A frag (P): P_lds[wave][(L%_MT)*_Bc + kb*_MT + (L/_MT)*4 + f]
        //   B frag (V): V_lds[(kb*_MT + (L/_MT)*4 + f)*D + jd*_MT + (L%_MT)]
        #pragma unroll
        for (int jd = 0; jd < nD; ++jd){
            _f32x4 acc = {0,0,0,0};
            #pragma unroll
            for (int kb = 0; kb < nBc; ++kb){
                _fp16x4 a, b;
                int p_row   = L % _MT;
                int bc_base = kb * _MT + (L / _MT) * 4;
                #pragma unroll
                for (int f = 0; f < 4; ++f) a[f] = _h2f16(P_lds[wave][p_row*_Bc + bc_base + f]);
                int d_col = jd * _MT + (L % _MT);
                #pragma unroll
                for (int f = 0; f < 4; ++f) b[f] = _h2f16(V_lds[(bc_base+f)*D + d_col]);
                acc = cdna3::mfma::mma(a, b, acc);
            }
            #pragma unroll
            for (int f = 0; f < 4; ++f){
                int orr = (L / _MT)*4 + f;
                int oc  = jd * _MT + (L % _MT);
                O_lds[wave][orr*D + oc] += acc[f];
            }
        }
        __syncthreads();
    }

    // Final normalize: O /= l (guard fully-masked rows).
    if (L < _Br) l_lds[wave][L] = (l_lds[wave][L] > 0.f) ? l_lds[wave][L] : 1.f;
    __syncthreads();
    for (int i = L; i < _Br*D; i += 64){
        int row = i / D;
        Oblk[i] = __float2half(O_lds[wave][i] / l_lds[wave][row]);
    }
}
