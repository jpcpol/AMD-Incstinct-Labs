#include "wave_primitives/wave_primitives.hpp"
#include "bench_common.hpp"

// ---------------------------------------------------------------------------
// bench_h3_fp16acc.hip — H3: float32-accumulation overhead for __half reduce
//
// H3 (pre-registered): "Float32 accumulation for __half reduce costs < 10%
//   latency vs native fp16." Falsification threshold: > 15%.
//
// The original bench_all_types measured this at the launch-bound floor (~5.7 µs),
// where every kernel is indistinguishable. This version amortizes the launch
// overhead with an internal kReps loop (the same technique used for the DPP
// reduce/scan microbenchmarks), so the per-reduction latency difference becomes
// measurable.
//
// Two strategies for reducing a wave of __half values:
//   A) fp32 acc  — promote each __half to float, reduce in float32 (our library)
//   B) naive fp16 — reduce directly in fp16 via __hadd (no promotion)
//
// To prevent the compiler from hoisting the reduction out of the kReps loop,
// the reduced value is fed back into the lane's input each iteration (data
// dependence chain), and the final value is written to global memory.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//         -I../include benchmarks/bench_h3_fp16acc.hip -o bench_h3_fp16acc
// ---------------------------------------------------------------------------

static constexpr int kBlocks = 1024;
static constexpr int kWarmup = 100;
static constexpr int kIter   = 1000;
static constexpr int kReps   = 4096;   // internal amortization loop

// --- A: fp32 accumulation (promote each fp16 to float, reduce in float) -----
// Exactly ONE wave reduction per rep — matches kernel B's operation count.
// The only difference vs B is the accumulation precision (float32 vs fp16).
// Each rep round-trips val through fp16 storage so both kernels carry the same
// per-lane fp16 input distribution into the reduction.
__global__ void kernel_half_fp32acc(const __half* in, float* out) {
    __half val = in[blockIdx.x * blockDim.x + threadIdx.x];
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float v = __half2float(val);                 // promote
        float s = wave::reduce_sum(v);               // reduce in float32 (ONE reduce)
        // perturb + store back to fp16 to keep the dependence chain
        val = __float2half(s * 0.5f + 1e-3f * (float)threadIdx.x);
    }
    if (wave::lane_id() == 0) out[blockIdx.x] = __half2float(val);
}

// --- B: naive fp16 reduction (reduce directly in __half via __hadd) ---------
__global__ void kernel_half_naive(const __half* in, float* out) {
    __half val = in[blockIdx.x * blockDim.x + threadIdx.x];
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        __half v = val;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1)  // ONE reduce
            v = __hadd(v, wave::shfl_down(v, offset));
        val = __hadd(__hmul(v, __float2half(0.5f)),
                     __float2half(1e-3f * (float)threadIdx.x));
    }
    if (wave::lane_id() == 0) out[blockIdx.x] = __half2float(val);
}

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);

    const int total = kBlocks * dev.wave_size;

    __half* d_in;
    float*  d_out_a;
    float*  d_out_b;
    HIP_CHECK(hipMalloc(&d_in,    total   * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_out_a, kBlocks * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_b, kBlocks * sizeof(float)));

    std::vector<__half> h_in(total, __float2half(0.1f));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), total * sizeof(__half), hipMemcpyHostToDevice));

    printf("\n--- H3: fp32-acc vs naive-fp16 __half reduce (kReps=%d amortized) ---\n", kReps);
    print_bench_header("reduce_sum __half  (1024 blocks x wave_size)");

    auto ra = time_kernel([&]() {
        hipLaunchKernelGGL(kernel_half_fp32acc, dim3(kBlocks), dim3(dev.wave_size), 0, 0, d_in, d_out_a);
    }, total, kWarmup, kIter);

    auto rb = time_kernel([&]() {
        hipLaunchKernelGGL(kernel_half_naive,   dim3(kBlocks), dim3(dev.wave_size), 0, 0, d_in, d_out_b);
    }, total, kWarmup, kIter);

    print_bench_result("__half  (fp32 acc)",   ra);
    print_bench_result("__half  (naive fp16)", rb);

    float overhead = (ra.median_us - rb.median_us) / rb.median_us * 100.0f;
    printf("\nfp32-acc overhead vs naive fp16: %+.1f%%\n", overhead);
    printf("H3 threshold: < 10%% (success), > 15%% (falsified)\n");
    printf("H3 outcome: %s\n",
           overhead < 10.0f ? "CONFIRMED (< 10%)" :
           overhead < 15.0f ? "INCONCLUSIVE (10-15%)" : "FALSIFIED (> 15%)");

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out_a));
    HIP_CHECK(hipFree(d_out_b));
    return 0;
}
