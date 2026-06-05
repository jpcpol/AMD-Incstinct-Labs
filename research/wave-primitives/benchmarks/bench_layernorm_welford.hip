#include "wave_primitives/wave_reduce_dpp.hpp"
#include "bench_common.hpp"
#include <hipcub/hipcub.hpp>
#include <cmath>

// ---------------------------------------------------------------------------
// bench_layernorm_welford.hip — One-pass Welford LayerNorm
//
// vs. bench_layernorm.hip (two-pass):
//   two-pass: read row → reduce_sum (mean) → re-compute deviations → reduce_sum (var)
//   one-pass: read row once, accumulate Welford (count,mean,M2) per lane,
//             then parallel-merge across lanes with a single DPP reduction over
//             a (mean, M2) pair — one HBM pass, one wave reduction.
//
// Parallel Welford merge (Chan et al. 1979):
//   Given groups A=(nA, mA, MA) and B=(nB, mB, MB):
//     n = nA + nB
//     delta = mB - mA
//     mean  = mA + delta * nB / n
//     M2    = MA + MB + delta^2 * nA*nB / n
//
// Implementation: reduce left-to-right across the 64 lanes using DPP
// row_shr shifts and row_bcast carries, each step merging two Welford groups.
// The result on lane 63 is the wave-wide (mean, M2) — no LDS needed.
//
// Layout: one wave (64 lanes) per row, each lane processes kElemsPerLane cols.
// Backends compared: Welford-DPP, two-pass-DPP, two-pass-portable.
// ---------------------------------------------------------------------------

#ifndef ELEMS_PER_LANE
#define ELEMS_PER_LANE 8
#endif

static constexpr int kRows         = 65536;
static constexpr int kElemsPerLane = ELEMS_PER_LANE;
static constexpr int kCols         = 64 * kElemsPerLane;
static constexpr float kEps        = 1e-5f;
static constexpr int kWarmup       = 50;
static constexpr int kIter         = 500;

// ---------------------------------------------------------------------------
// Welford parallel merge helpers (device)
// ---------------------------------------------------------------------------

struct WelfordState {
    float mean;
    float M2;
    int   n;
};

__device__ __forceinline__
WelfordState welford_merge(WelfordState a, WelfordState b) {
    if (b.n == 0) return a;
    if (a.n == 0) return b;
    int   n     = a.n + b.n;
    float delta = b.mean - a.mean;
    float mean  = a.mean + delta * (float)b.n / (float)n;
    float M2    = a.M2 + b.M2 + delta * delta * (float)a.n * (float)b.n / (float)n;
    return {mean, M2, n};
}

// ---------------------------------------------------------------------------
// DPP Welford wave-reduce: merge 64 WelfordStates using DPP shifts.
//
// Strategy: reduce the wave in 6 steps (log2(64)) using __builtin_amdgcn_update_dpp
// to shift the neighbor's (mean, M2, n) into the current lane and merge.
// Each step halves the number of "active" groups. The final result lands on lane 63.
// We then broadcast mean and M2 from lane 63 to all lanes using v_readlane_b32.
//
// All float moves use DPP (row_shr), n is an integer — also moved via DPP
// (dpp works on 32-bit words; we cast int to float for the move, cast back).
// ---------------------------------------------------------------------------

__device__ __forceinline__
WelfordState dpp_welford_reduce(WelfordState s) {
    // Steps within each 16-lane row: row_shr 1, 2, 4, 8
    // Steps across rows: row_bcast15, row_bcast31
    // After each step, merge if the neighbor is a valid predecessor.

    // We use a simpler sequential-merge DPP scan approach:
    // lane L merges with lane L-stride if L >= stride.
    // This is equivalent to a binary tree reduction in wave order.

    // Use DPP to implement reduce via successive halving:
    // stride 1: lanes 1,3,5,...63 receive from lanes 0,2,4,...62 → merge
    // stride 2: lanes 2,6,...62 receive accumulated from stride-1 → merge
    // stride 4,8 within row, then bcast15 and bcast31 across rows.

    // --- row_shr1 (DPP ctrl 0x111): lane L ← lane L-1 within 16-lane row ---
    // bound_ctrl=true fills lane 0 of each row with identity (n=0 → skip merge)
    float nb_mean, nb_M2;
    int   nb_n;

    #define DPP_SHIFT(ctrl, bcast_from_lane)                                        \
        nb_mean = __builtin_amdgcn_update_dpp(0.0f, s.mean, ctrl, 0xf, 0xf, true); \
        nb_M2   = __builtin_amdgcn_update_dpp(0.0f, s.M2,   ctrl, 0xf, 0xf, true); \
        nb_n    = __builtin_amdgcn_update_dpp(0,    s.n,    ctrl, 0xf, 0xf, true); \
        s = welford_merge(s, {nb_mean, nb_M2, nb_n})

    DPP_SHIFT(0x111, 0);   // row_shr1  — stride 1
    DPP_SHIFT(0x112, 0);   // row_shr2  — stride 2
    DPP_SHIFT(0x114, 0);   // row_shr4  — stride 4
    DPP_SHIFT(0x118, 0);   // row_shr8  — stride 8
    // Now lanes 15/31/47/63 each hold the sum of their 16-lane row.
    // row_bcast15 (0x142): lane 15 → lanes 16-31, lane 31 → 32-47, lane 47 → 48-63
    DPP_SHIFT(0x142, 15);  // row_bcast15 — cross-row carry
    DPP_SHIFT(0x143, 31);  // row_bcast31 — bank carry

    #undef DPP_SHIFT

    // Broadcast final mean and M2 from lane 63 to all lanes.
    float final_mean = __shfl(s.mean, 63, 64);
    float final_M2   = __shfl(s.M2,   63, 64);
    return {final_mean, final_M2, kCols};
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// One-pass Welford with DPP reduce
__global__ void layernorm_welford_dpp(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ out)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const float* xrow   = x   + (size_t)row * kCols;
    float*       outrow = out + (size_t)row * kCols;

    float vals[kElemsPerLane];

    // Accumulate per-lane Welford state (sequential over kElemsPerLane elements)
    WelfordState s = {0.0f, 0.0f, 0};
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        float v = xrow[lane * kElemsPerLane + e];
        vals[e] = v;
        ++s.n;
        float delta = v - s.mean;
        s.mean += delta / (float)s.n;
        float delta2 = v - s.mean;
        s.M2 += delta * delta2;
    }

    // One DPP wave-reduce to merge all 64 per-lane Welford states
    WelfordState w = dpp_welford_reduce(s);
    float inv_std = rsqrtf(w.M2 / (float)kCols + kEps);

    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        int col = lane * kElemsPerLane + e;
        outrow[col] = (vals[e] - w.mean) * inv_std * gamma[col] + beta[col];
    }
}

// Two-pass DPP (baseline from bench_layernorm.hip)
__global__ void layernorm_twopass_dpp(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ out)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const float* xrow   = x   + (size_t)row * kCols;
    float*       outrow = out + (size_t)row * kCols;

    float vals[kElemsPerLane];
    float lsum = 0.0f;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        vals[e] = xrow[lane * kElemsPerLane + e];
        lsum += vals[e];
    }
    float mean = wave::dpp::reduce_sum_bcast(lsum) / kCols;

    float lvar = 0.0f;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        float d = vals[e] - mean;
        lvar += d * d;
    }
    float inv_std = rsqrtf(wave::dpp::reduce_sum_bcast(lvar) / kCols + kEps);

    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        int col = lane * kElemsPerLane + e;
        outrow[col] = (vals[e] - mean) * inv_std * gamma[col] + beta[col];
    }
}

// Two-pass portable (shfl_down, worst case)
__global__ void layernorm_twopass_portable(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ out)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const float* xrow   = x   + (size_t)row * kCols;
    float*       outrow = out + (size_t)row * kCols;

    float vals[kElemsPerLane];
    float lsum = 0.0f;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        vals[e] = xrow[lane * kElemsPerLane + e];
        lsum += vals[e];
    }
    for (int o = 32; o > 0; o >>= 1) lsum += __shfl_down(lsum, o, 64);
    float mean = __shfl(lsum, 0, 64) / kCols;

    float lvar = 0.0f;
    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        float d = vals[e] - mean;
        lvar += d * d;
    }
    for (int o = 32; o > 0; o >>= 1) lvar += __shfl_down(lvar, o, 64);
    float inv_std = rsqrtf(__shfl(lvar, 0, 64) / kCols + kEps);

    #pragma unroll
    for (int e = 0; e < kElemsPerLane; ++e) {
        int col = lane * kElemsPerLane + e;
        outrow[col] = (vals[e] - mean) * inv_std * gamma[col] + beta[col];
    }
}

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);
    if (dev.wave_size != 64) { printf("Requires wave64.\n"); return 1; }

    const size_t n = (size_t)kRows * kCols;
    float *d_x, *d_gamma, *d_beta, *d_out;
    HIP_CHECK(hipMalloc(&d_x,     n     * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gamma, kCols * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_beta,  kCols * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out,   n     * sizeof(float)));

    {
        std::vector<float> hx(n), hg(kCols, 1.0f), hb(kCols, 0.0f);
        for (size_t i = 0; i < n; ++i)
            hx[i] = (float)((i * 1103515245u + 12345u) & 0xff) / 32.0f - 4.0f;
        HIP_CHECK(hipMemcpy(d_x,     hx.data(), n     * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_gamma, hg.data(), kCols * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_beta,  hb.data(), kCols * sizeof(float), hipMemcpyHostToDevice));
    }

    auto check = [&](const char* name, auto kern) {
        hipLaunchKernelGGL(kern, dim3(kRows), dim3(64), 0, 0, d_x, d_gamma, d_beta, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> h(kCols);
        HIP_CHECK(hipMemcpy(h.data(), d_out, kCols * sizeof(float), hipMemcpyDeviceToHost));
        float m = 0.f, v = 0.f;
        for (int i = 0; i < kCols; ++i) m += h[i];
        m /= kCols;
        for (int i = 0; i < kCols; ++i) v += (h[i]-m)*(h[i]-m);
        v /= kCols;
        bool ok = fabsf(m) < 1e-3f && fabsf(v - 1.0f) < 1e-2f;
        printf("  %-25s row0: mean=%8.5f var=%8.5f [%s]\n", name, m, v, ok?"PASS":"FAIL");
    };

    printf("\n--- Correctness (gamma=1, beta=0: mean≈0, var≈1) ---\n");
    check("Welford DPP (one-pass)",  layernorm_welford_dpp);
    check("two-pass DPP",            layernorm_twopass_dpp);
    check("two-pass portable",       layernorm_twopass_portable);

    printf("\nConfig: %d rows x %d cols (ELEMS_PER_LANE=%d)\n\n", kRows, kCols, kElemsPerLane);
    print_bench_header("LayerNorm — Welford one-pass DPP vs two-pass DPP vs two-pass portable");

    const double elems = (double)n;
    auto rw = time_kernel([&](){ hipLaunchKernelGGL(layernorm_welford_dpp,     dim3(kRows),dim3(64),0,0,d_x,d_gamma,d_beta,d_out);}, elems, kWarmup, kIter);
    auto rd = time_kernel([&](){ hipLaunchKernelGGL(layernorm_twopass_dpp,     dim3(kRows),dim3(64),0,0,d_x,d_gamma,d_beta,d_out);}, elems, kWarmup, kIter);
    auto rp = time_kernel([&](){ hipLaunchKernelGGL(layernorm_twopass_portable,dim3(kRows),dim3(64),0,0,d_x,d_gamma,d_beta,d_out);}, elems, kWarmup, kIter);

    print_bench_result("Welford DPP (one-pass)",  rw);
    print_bench_result("two-pass DPP",            rd);
    print_bench_result("two-pass portable",       rp);
    print_speedup("Welford DPP", "two-pass DPP",      rw, rd);
    print_speedup("Welford DPP", "two-pass portable", rw, rp);

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_gamma));
    HIP_CHECK(hipFree(d_beta));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
