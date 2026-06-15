// ---------------------------------------------------------------------------
// validate_decode_dme.hip — correctness test for fa_decode_dme_kernel.
//
// Compares cdna3::attn::decode_dme (reads strided KV-cache with DME) against
// the already-validated cdna3::attn::decode (reads contiguous packed K/V).
// Both must agree to max_err < 0.01 fp16 (matching Stage-B tolerance).
//
// Test matrix:
//   D = 64, 128
//   N = 64, 128, 256, 512  (must be multiple of kBc=64)
//   GQA ratio 1:1 (MHA) and 7:1
//
// Compile (from cdna3-prims/tests/):
//   /opt/rocm/bin/hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//     -I../include validate_decode_dme.hip -o validate_decode_dme
// ---------------------------------------------------------------------------
#include <cstdio>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../include/cdna3/attention.hpp"
#include "../include/cdna3/runtime/forward.hpp"   // pack_kv_kernel (for reference path)

// ---------------------------------------------------------------------------
// Build a packed contiguous K/V [nKVH, N, D] from a strided cache
// [nKVH, max_seq, D] on host (for the reference decode path).
// ---------------------------------------------------------------------------
static void pack_host(const std::vector<__half>& cache, int nKVH, int N, int D,
                      int max_seq, std::vector<__half>& packed)
{
    packed.resize((size_t)nKVH * N * D);
    for (int h = 0; h < nKVH; ++h)
        for (int n = 0; n < N; ++n)
            for (int d = 0; d < D; ++d)
                packed[(size_t)h*N*D + n*D + d] =
                    cache[(size_t)h*max_seq*D + n*D + d];
}

struct Config { int D, N, nQH, nKVH; const char* tag; };

int main(){
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
    printf("Device : %s\n\n", prop.name);
    printf("validate_decode_dme — fa_decode_dme_kernel vs fa_decode_kernel\n");
    printf("================================================================\n\n");

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.f, 0.5f);

    Config configs[] = {
        { 64,  64,  8, 8, "D=64  N=64  MHA"},
        { 64, 128,  8, 8, "D=64  N=128 MHA"},
        { 64, 256, 14, 2, "D=64  N=256 GQA7:1"},
        { 64, 512, 14, 2, "D=64  N=512 GQA7:1"},
        {128,  64,  8, 8, "D=128 N=64  MHA"},
        {128, 256,  8, 8, "D=128 N=256 MHA"},
        {128, 512, 14, 2, "D=128 N=512 GQA7:1"},
    };
    int n_configs = (int)(sizeof(configs)/sizeof(configs[0]));

    int max_seq = 1024;   // stride for the cache
    int all_pass = 1;

    for (int ci = 0; ci < n_configs; ++ci) {
        auto& cfg = configs[ci];
        int D = cfg.D, N = cfg.N, nQH = cfg.nQH, nKVH = cfg.nKVH;

        // Generate random fp16 Q, and a strided KV-cache [nKVH, max_seq, D]
        size_t q_elems    = (size_t)nQH  * D;
        size_t cache_elems= (size_t)nKVH * max_seq * D;
        size_t packed_elems = (size_t)nKVH * N * D;

        std::vector<__half> hQ(q_elems), hCache_K(cache_elems, __float2half(0.f)),
                                          hCache_V(cache_elems, __float2half(0.f));
        for (auto& v : hQ)         v = __float2half(dist(rng));
        // Fill only the first N positions of the cache
        for (int h = 0; h < nKVH; ++h)
            for (int n = 0; n < N; ++n)
                for (int d = 0; d < D; ++d){
                    hCache_K[(size_t)h*max_seq*D + n*D + d] = __float2half(dist(rng));
                    hCache_V[(size_t)h*max_seq*D + n*D + d] = __float2half(dist(rng));
                }

        // Pack for reference decode path
        std::vector<__half> hPacked_K, hPacked_V;
        pack_host(hCache_K, nKVH, N, D, max_seq, hPacked_K);
        pack_host(hCache_V, nKVH, N, D, max_seq, hPacked_V);

        // Device alloc
        __half *d_Q, *d_CacheK, *d_CacheV, *d_PackK, *d_PackV, *d_O_ref, *d_O_dme;
        hipMalloc(&d_Q,      q_elems*2);
        hipMalloc(&d_CacheK, cache_elems*2);
        hipMalloc(&d_CacheV, cache_elems*2);
        hipMalloc(&d_PackK,  packed_elems*2);
        hipMalloc(&d_PackV,  packed_elems*2);
        hipMalloc(&d_O_ref,  q_elems*2);
        hipMalloc(&d_O_dme,  q_elems*2);

        hipMemcpy(d_Q,      hQ.data(),       q_elems*2,      hipMemcpyHostToDevice);
        hipMemcpy(d_CacheK, hCache_K.data(), cache_elems*2,  hipMemcpyHostToDevice);
        hipMemcpy(d_CacheV, hCache_V.data(), cache_elems*2,  hipMemcpyHostToDevice);
        hipMemcpy(d_PackK,  hPacked_K.data(),packed_elems*2, hipMemcpyHostToDevice);
        hipMemcpy(d_PackV,  hPacked_V.data(),packed_elems*2, hipMemcpyHostToDevice);

        // Reference: decode() on contiguous packed K/V
        cdna3::attn::DecodeCfg ref_cfg{D, N, nQH, nKVH};
        cdna3::attn::decode(d_Q, d_PackK, d_PackV, d_O_ref, ref_cfg);
        hipDeviceSynchronize();

        // DME: decode_dme() on strided cache
        cdna3::attn::DecodeDmeCfg dme_cfg{D, N, nQH, nKVH, max_seq};
        cdna3::attn::decode_dme(d_Q, d_CacheK, d_CacheV, d_O_dme, dme_cfg);
        hipDeviceSynchronize();

        // Compare
        std::vector<__half> hO_ref(q_elems), hO_dme(q_elems);
        hipMemcpy(hO_ref.data(), d_O_ref, q_elems*2, hipMemcpyDeviceToHost);
        hipMemcpy(hO_dme.data(), d_O_dme, q_elems*2, hipMemcpyDeviceToHost);

        float max_err = 0.f, rms = 0.f;
        for (size_t i = 0; i < q_elems; ++i){
            float diff = fabsf(__half2float(hO_ref[i]) - __half2float(hO_dme[i]));
            max_err = std::max(max_err, diff);
            rms += diff * diff;
        }
        rms = sqrtf(rms / (float)q_elems);
        bool pass = (max_err < 0.01f);
        if (!pass) all_pass = 0;
        printf("  %-22s  max_err=%.5f  rms=%.5f  [%s]\n",
               cfg.tag, max_err, rms, pass ? "PASS" : "FAIL");

        hipFree(d_Q); hipFree(d_CacheK); hipFree(d_CacheV);
        hipFree(d_PackK); hipFree(d_PackV); hipFree(d_O_ref); hipFree(d_O_dme);
    }

    printf("\n%s\n", all_pass ? "ALL PASS — fa_decode_dme_kernel validated."
                              : "FAILURES DETECTED — investigate kernel.");
    return all_pass ? 0 : 1;
}
