// ---------------------------------------------------------------------------
// fa_mfma2.hip — Flash Attention Step 4 (v2): MFMA, 1-wave-per-block design
//
// Fixes the fa_mfma.hip structural bug: that kernel used Br=16 INDEPENDENT
// warps, but v_mfma_f32_16x16x16f16 needs ONE wave's 64 lanes to hold all 16
// rows of the tile. This version uses exactly ONE wave (64 lanes) per query
// block, so the MFMA semantics line up.
//
// Architecture:
//   block = (64) — one wave. Each block processes Br=16 query rows over the
//   full K/V sequence. grid = batch*heads*(seqQ/Br).
//
//   Per KV tile (Bc=64 keys):
//     1. S = Q·Kᵀ   via MFMA  (16×64 = 4 col-tiles, each summing 4 D-tiles)
//     2. S written to LDS (16×Bc fp32) using verified MFMA output mapping
//     3. online softmax over S rows (lanes 0..15 each own one query row)
//     4. P (= exp(S-m)) written to LDS as fp16
//     5. O += P·V   via MFMA  (16×64 = 4 D-tiles, each summing 4 Bc-tiles)
//
// MFMA lane mapping (verified, probe_mfma_mapping.hip, gfx942):
//   A in : lane L,f → A[row=L%16     ][k  =(L/16)*4+f]
//   B in : lane L,f → B[k  =(L/16)*4+f][col=L%16      ]
//   C/out: lane L,f → C[row=(L/16)*4+f][col=L%16      ]
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//         -I../wave-primitives/include/wave_primitives \
//         -I../dme-abstraction/include  fa_mfma2.hip -o fa_mfma2
// ---------------------------------------------------------------------------
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

static constexpr int Br  = 16;
static constexpr int Bc  = 64;
static constexpr int D   = 64;
static constexpr int kN  = 512;

static constexpr int kBatch  = 1;
static constexpr int kHeads  = 8;
static constexpr int kSeqQ   = 512;
static constexpr int kBlocks = kBatch * kHeads * kSeqQ / Br;
static constexpr int kWarmup = 20;
static constexpr int kIter   = 200;

static constexpr int MT  = 16;        // MFMA tile
static constexpr int nD  = D  / MT;   // 4
static constexpr int nBc = Bc / MT;   // 4

using fp16x4 = __attribute__((ext_vector_type(4))) __fp16;
using f32x4  = __attribute__((ext_vector_type(4))) float;

__device__ __forceinline__ __fp16 h2f16(__half h){ return *reinterpret_cast<__fp16*>(&h); }
__device__ __forceinline__ __fp16 f2f16(float f){ __half h=__float2half(f); return *reinterpret_cast<__fp16*>(&h); }
__device__ __forceinline__
f32x4 mfma(fp16x4 a, fp16x4 b, f32x4 c){ return __builtin_amdgcn_mfma_f32_16x16x16f16(a,b,c,0,0,0); }

// ---------------------------------------------------------------------------
__global__ void fa_mfma2_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ O,
    int N)
{
    const int qblk  = blockIdx.x % (kSeqQ / Br);
    const int head  = (blockIdx.x / (kSeqQ / Br)) % kHeads;
    const int batch = blockIdx.x / (kSeqQ / Br) / kHeads;
    const int L = threadIdx.x;   // 0..63, single wave

    const __half* Qblk  = Q + ((size_t)batch*kHeads+head)*kSeqQ*D + qblk*Br*D;
    const __half* Kbase = K + ((size_t)batch*kHeads+head)*N*D;
    const __half* Vbase = V + ((size_t)batch*kHeads+head)*N*D;
    __half*       Oblk  = O + ((size_t)batch*kHeads+head)*kSeqQ*D + qblk*Br*D;

    __shared__ __half Q_lds[Br * D];     // 16×64 fp16  = 2 KB
    __shared__ __half K_lds[Bc * D];     // 64×64 fp16  = 8 KB
    __shared__ __half V_lds[Bc * D];     // 64×64 fp16  = 8 KB
    __shared__ float  S_lds[Br * Bc];    // 16×64 fp32  = 4 KB
    __shared__ __half P_lds[Br * Bc];    // 16×64 fp16  = 2 KB
    __shared__ float  O_lds[Br * D];     // 16×64 fp32  = 4 KB

    // Load Q block into LDS (Br*D=1024 elems, 64 lanes → 16 each)
    #pragma unroll
    for (int i = L; i < Br*D; i += 64) Q_lds[i] = Qblk[i];

    // Per-row softmax running state — lanes 0..15 own query rows 0..15
    float m_run = -1e38f, l_run = 0.f;
    // Each lane also keeps its row's O accumulator in LDS (O_lds), init 0
    #pragma unroll
    for (int i = L; i < Br*D; i += 64) O_lds[i] = 0.f;

    const float scale = 1.f / sqrtf((float)D);
    const int kv_tiles = N / Bc;
    __syncthreads();

    for (int kv = 0; kv < kv_tiles; ++kv) {
        const __half* Kt = Kbase + (size_t)kv*Bc*D;
        const __half* Vt = Vbase + (size_t)kv*Bc*D;

        // Load K,V tiles into LDS (Bc*D=4096 each, 64 lanes → 64 each)
        #pragma unroll
        for (int i = L; i < Bc*D; i += 64) { K_lds[i] = Kt[i]; V_lds[i] = Vt[i]; }
        __syncthreads();

        // -------- S = Q · Kᵀ  via MFMA --------
        // S[16×Bc]. For each Bc col-tile jc (4 tiles of 16 keys):
        //   S_frag = sum over D-tiles kd of  Q_tile[kd] · (K_tile[jc,kd])ᵀ
        // A = Q: A[row=query, k=D]; lane L,f → Q_lds[(L%16)*D + kd*MT + (L/16)*4+f]
        // B = Kᵀ: we need Kᵀ[k=D][col=key]. K_lds is [key][D] row-major, so
        //   Kᵀ[d][key] = K_lds[key*D + d]. B[k=d][col=key]: lane L,f →
        //   k=(L/16)*4+f (the D index), col=L%16 (the key index within tile jc)
        //   → K_lds[(jc*MT + L%16)*D + kd*MT + (L/16)*4+f]
        #pragma unroll
        for (int jc = 0; jc < nBc; ++jc) {
            f32x4 acc = {0,0,0,0};
            #pragma unroll
            for (int kd = 0; kd < nD; ++kd) {
                fp16x4 a, b;
                int q_row = L % MT;
                int d_idx = kd*MT + (L/MT)*4;
                #pragma unroll
                for (int f=0; f<4; ++f) a[f] = h2f16(Q_lds[q_row*D + d_idx + f]);
                int key = jc*MT + (L % MT);
                #pragma unroll
                for (int f=0; f<4; ++f) {
                    int dd = kd*MT + (L/MT)*4 + f;
                    b[f] = h2f16(K_lds[key*D + dd]);
                }
                acc = mfma(a, b, acc);
            }
            // Write S_frag to S_lds. Output mapping: lane L,f → [row=(L/16)*4+f][col=L%16]
            // col within tile jc → global key = jc*MT + L%16
            #pragma unroll
            for (int f=0; f<4; ++f) {
                int s_row = (L/MT)*4 + f;
                int s_col = jc*MT + (L % MT);
                S_lds[s_row*Bc + s_col] = acc[f] * scale;
            }
        }
        __syncthreads();

        // -------- Online softmax over S rows --------
        // Lanes 0..15 each own one query row; lanes 16..63 idle here.
        if (L < Br) {
            int row = L;
            float row_max = -1e38f;
            #pragma unroll
            for (int c = 0; c < Bc; ++c) row_max = fmaxf(row_max, S_lds[row*Bc + c]);
            float m_new = fmaxf(m_run, row_max);
            float row_sum = 0.f;
            #pragma unroll
            for (int c = 0; c < Bc; ++c) {
                float p = __expf(S_lds[row*Bc + c] - m_new);
                P_lds[row*Bc + c] = __float2half(p);
                row_sum += p;
            }
            float rescale = __expf(m_run - m_new);
            float l_new = rescale * l_run + row_sum;
            // rescale this row's O accumulator
            #pragma unroll
            for (int d = 0; d < D; ++d) O_lds[row*D + d] *= rescale;
            m_run = m_new;
            l_run = l_new;
        }
        __syncthreads();

        // -------- O += P · V  via MFMA --------
        // O[16×D]. For each D col-tile jd (4 tiles of 16):
        //   O_frag = sum over Bc-tiles kb of  P_tile[kb] · V_tile[kb,jd]
        // A = P: A[row=query][k=Bc]; lane L,f → P_lds[(L%16)*Bc + kb*MT + (L/16)*4+f]
        // B = V: B[k=Bc][col=D]; lane L,f → V_lds[(kb*MT + L/16*4+f)*D + jd*MT + L%16]
        #pragma unroll
        for (int jd = 0; jd < nD; ++jd) {
            f32x4 acc = {0,0,0,0};
            #pragma unroll
            for (int kb = 0; kb < nBc; ++kb) {
                fp16x4 a, b;
                int p_row = L % MT;
                int bc_idx = kb*MT + (L/MT)*4;
                #pragma unroll
                for (int f=0; f<4; ++f) a[f] = h2f16(P_lds[p_row*Bc + bc_idx + f]);
                int d_col = jd*MT + (L % MT);
                #pragma unroll
                for (int f=0; f<4; ++f) {
                    int kk = kb*MT + (L/MT)*4 + f;
                    b[f] = h2f16(V_lds[kk*D + d_col]);
                }
                acc = mfma(a, b, acc);
            }
            // Accumulate into O_lds. Output: lane L,f → [row=(L/16)*4+f][col=L%16]
            #pragma unroll
            for (int f=0; f<4; ++f) {
                int o_row = (L/MT)*4 + f;
                int o_col = jd*MT + (L % MT);
                O_lds[o_row*D + o_col] += acc[f];
            }
        }
        __syncthreads();
    }

    // Normalize and write. Lanes 0..15 own rows; need l_run per row.
    // l_run lives in lanes 0..15 only — broadcast via LDS.
    __shared__ float l_share[Br];
    if (L < Br) l_share[L] = l_run;
    __syncthreads();
    // All 64 lanes write O block: Br*D=1024 elems
    #pragma unroll
    for (int i = L; i < Br*D; i += 64) {
        int row = i / D;
        Oblk[i] = __float2half(O_lds[i] / l_share[row]);
    }
}

// ---------------------------------------------------------------------------
void attn_cpu_ref(const std::vector<float>& Q, const std::vector<float>& K,
                  const std::vector<float>& V, std::vector<float>& O,
                  int seq_q, int seq_k, int d)
{
    const float scale = 1.f / sqrtf((float)d);
    for (int q = 0; q < seq_q; ++q) {
        std::vector<float> S(seq_k);
        float mx=-1e38f;
        for (int k=0;k<seq_k;++k){ float dot=0; for(int dd=0;dd<d;++dd) dot+=Q[q*d+dd]*K[k*d+dd]; S[k]=dot*scale; mx=std::max(mx,S[k]); }
        float se=0; for(int k=0;k<seq_k;++k){ S[k]=expf(S[k]-mx); se+=S[k]; }
        for(int k=0;k<seq_k;++k) S[k]/=se;
        for(int dd=0;dd<d;++dd){ float o=0; for(int k=0;k<seq_k;++k) o+=S[k]*V[k*d+dd]; O[q*d+dd]=o; }
    }
}

int main() {
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
    printf("Device : %s  (gfx%d)\n", prop.name, prop.major*100+prop.minor);
    printf("Config : Br=%d Bc=%d D=%d seqQ=%d seqK=%d heads=%d\n", Br,Bc,D,kSeqQ,kN,kHeads);
    printf("Design : 1 wave/block, MFMA Q·Kᵀ and P·V (16×16×16 fp16)\n\n");

    const size_t QKV=(size_t)kBatch*kHeads*kSeqQ*D, KV=(size_t)kBatch*kHeads*kN*D;
    std::vector<float> hQf(QKV),hKf(KV),hVf(KV);
    std::mt19937 rng(42); std::normal_distribution<float> dist(0.f,0.5f);
    for(auto&v:hQf)v=dist(rng); for(auto&v:hKf)v=dist(rng); for(auto&v:hVf)v=dist(rng);
    std::vector<__half> hQ(QKV),hK(KV),hV(KV);
    for(size_t i=0;i<QKV;++i)hQ[i]=__float2half(hQf[i]);
    for(size_t i=0;i<KV;++i){hK[i]=__float2half(hKf[i]);hV[i]=__float2half(hVf[i]);}

    __half *dQ,*dK,*dV,*dO;
    hipMalloc(&dQ,QKV*2);hipMalloc(&dK,KV*2);hipMalloc(&dV,KV*2);hipMalloc(&dO,QKV*2);
    hipMemcpy(dQ,hQ.data(),QKV*2,hipMemcpyHostToDevice);
    hipMemcpy(dK,hK.data(),KV*2,hipMemcpyHostToDevice);
    hipMemcpy(dV,hV.data(),KV*2,hipMemcpyHostToDevice);

    dim3 grid(kBlocks), block(64);
    hipLaunchKernelGGL(fa_mfma2_kernel,grid,block,0,0,dQ,dK,dV,dO,kN);
    hipError_t e=hipDeviceSynchronize();
    if(e!=hipSuccess){printf("KERNEL ERROR: %s\n",hipGetErrorString(e));return 1;}

    std::vector<__half> hOg(QKV); hipMemcpy(hOg.data(),dO,QKV*2,hipMemcpyDeviceToHost);
    std::vector<float> hOr(kSeqQ*D);
    attn_cpu_ref(hQf,hKf,hVf,hOr,kSeqQ,kN,D);
    float err=0; for(int i=0;i<kSeqQ*D;++i) err=std::max(err,fabsf(__half2float(hOg[i])-hOr[i]));
    printf("--- Correctness (first head) ---\n  max |gpu-ref| = %.5f  [%s]\n\n",
           err, err<0.1f?"PASS":"FAIL");
    if(err>=0.1f){printf("Abort benchmark.\n");return 1;}

    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    for(int i=0;i<kWarmup;++i) hipLaunchKernelGGL(fa_mfma2_kernel,grid,block,0,0,dQ,dK,dV,dO,kN);
    hipDeviceSynchronize();
    std::vector<float> ts(kIter);
    for(int i=0;i<kIter;++i){ hipEventRecord(t0); hipLaunchKernelGGL(fa_mfma2_kernel,grid,block,0,0,dQ,dK,dV,dO,kN); hipEventRecord(t1); hipEventSynchronize(t1); hipEventElapsedTime(&ts[i],t0,t1); ts[i]*=1000.f; }
    std::sort(ts.begin(),ts.end());
    float med=ts[kIter/2], p99=ts[(int)(kIter*0.99f)];
    double flops=2.0*2.0*Br*Bc*D*(kN/Bc)*(double)kBlocks;
    printf("=== FA MFMA v2 — benchmark ===\n");
    printf("  Median: %.1f µs | P99: %.1f µs | %.2f TFLOPS\n", med,p99,flops/med/1e6);
    printf("\n  fa_dme  (Step 3): 82.4 µs\n  fa_mfma2(Step 4): %.1f µs\n  Speedup: %.3fx\n",
           med, 82.4f/med);
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
    return 0;
}
