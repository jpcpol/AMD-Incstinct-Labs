#include "wave_primitives/wave_primitives.hpp"
#include "bench_common.hpp"

// ---------------------------------------------------------------------------
// bench_bf16acc.hip — G2: float32-accumulation overhead for bf16 reduce
//
// Companion to bench_h3_fp16acc.hip (which answered H3 for __half: fp32-acc
// is 10% FASTER than native fp16 on CDNA3). H3's pre-registration left bf16
// "assumed but not separately amortized." This closes that gap.
//
// Two strategies for reducing a wave of hip_bfloat16 values:
//   A) fp32 acc  — promote each bf16 to float, reduce in float32 (our library)
//   B) naive bf16 — reduce in bf16 precision (convert back to bf16 each step)
//
// kReps internal loop amortizes launch overhead (the ~5.7 µs floor that made
// the original bench_all_types measurement indistinguishable). Dependence chain
// kept live by feeding the reduced value back each iteration.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 -I../include \
//         benchmarks/bench_bf16acc.hip -o bench_bf16acc
// ---------------------------------------------------------------------------

static constexpr int kBlocks = 1024;
static constexpr int kWarmup = 100;
static constexpr int kIter   = 1000;
static constexpr int kReps   = 4096;

// --- A: fp32 accumulation (promote bf16 -> float, reduce in float) ----------
__global__ void kernel_bf16_fp32acc(const hip_bfloat16* in, float* out) {
    hip_bfloat16 val = in[blockIdx.x * blockDim.x + threadIdx.x];
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float v = float(val);                        // promote
        float s = wave::reduce_sum(v);               // reduce in float32 (ONE reduce)
        val = hip_bfloat16(s * 0.5f + 1e-3f * (float)threadIdx.x);  // perturb + store bf16
    }
    if (wave::lane_id() == 0) out[blockIdx.x] = float(val);
}

// --- B: naive bf16 reduction (reduce in bf16 precision) ---------------------
__global__ void kernel_bf16_naive(const hip_bfloat16* in, float* out) {
    hip_bfloat16 val = in[blockIdx.x * blockDim.x + threadIdx.x];
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        float v = float(val);
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            hip_bfloat16 a = hip_bfloat16(v);
            hip_bfloat16 b = wave::shfl_down(a, offset);
            v = float(hip_bfloat16(float(a) + float(b)));  // round back to bf16 each step
        }
        val = hip_bfloat16(v * 0.5f + 1e-3f * (float)threadIdx.x);
    }
    if (wave::lane_id() == 0) out[blockIdx.x] = float(val);
}

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);
    const int total = kBlocks * dev.wave_size;

    hip_bfloat16* d_in; float *d_a, *d_b;
    HIP_CHECK(hipMalloc(&d_in, total*sizeof(hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_a, kBlocks*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, kBlocks*sizeof(float)));
    std::vector<hip_bfloat16> h_in(total, hip_bfloat16(0.1f));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), total*sizeof(hip_bfloat16), hipMemcpyHostToDevice));

    printf("\n--- G2: fp32-acc vs naive-bf16 reduce (kReps=%d amortized) ---\n", kReps);
    print_bench_header("reduce_sum bf16  (1024 blocks x wave_size)");

    auto ra = time_kernel([&](){ hipLaunchKernelGGL(kernel_bf16_fp32acc, dim3(kBlocks), dim3(dev.wave_size), 0, 0, d_in, d_a); }, total, kWarmup, kIter);
    auto rb = time_kernel([&](){ hipLaunchKernelGGL(kernel_bf16_naive,   dim3(kBlocks), dim3(dev.wave_size), 0, 0, d_in, d_b); }, total, kWarmup, kIter);

    print_bench_result("bf16 (fp32 acc)",  ra);
    print_bench_result("bf16 (naive bf16)", rb);

    float overhead = (ra.median_us - rb.median_us) / rb.median_us * 100.0f;
    printf("\nfp32-acc overhead vs naive bf16: %+.1f%%\n", overhead);
    printf("(Compare to H3/__half result: fp32-acc was -10.1%%, i.e. faster.)\n");

    HIP_CHECK(hipFree(d_in)); HIP_CHECK(hipFree(d_a)); HIP_CHECK(hipFree(d_b));
    return 0;
}
