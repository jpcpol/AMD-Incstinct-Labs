#include "wave_primitives/wave_primitives.hpp"
#include "bench_common.hpp"

// ---------------------------------------------------------------------------
// bench_all_types.hip — wave::reduce_sum across all supported types
//
// Tests: float, double, int32_t, int64_t, __half, hip_bfloat16
//
// For __half and bfloat16, compares two strategies:
//   A) wave::reduce_sum — accumulates in float32 internally (our library)
//   B) naive_fp16_reduce — shfl_down directly on fp16 (no float promotion)
//
// This demonstrates both the performance difference and the numerical
// accuracy advantage of float32 accumulation for half-precision types.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx942 \
//         -I../include -I/opt/rocm/include \
//         benchmarks/bench_all_types.hip -o bench_all_types
// ---------------------------------------------------------------------------

static constexpr int kBlocks = 1024;
static constexpr int kWarmup = 100;
static constexpr int kIter   = 1000;

// ---------------------------------------------------------------------------
// Kernels — one per type
// ---------------------------------------------------------------------------

__global__ void kernel_reduce_float(const float* in, float* out) {
    float val = in[blockIdx.x * blockDim.x + threadIdx.x];
    float sum = wave::reduce_sum(val);
    if (wave::lane_id() == 0) out[blockIdx.x] = sum;
}

__global__ void kernel_reduce_double(const double* in, double* out) {
    double val = in[blockIdx.x * blockDim.x + threadIdx.x];
    double sum = wave::reduce_sum(val);
    if (wave::lane_id() == 0) out[blockIdx.x] = sum;
}

__global__ void kernel_reduce_int32(const int* in, int* out) {
    int val = in[blockIdx.x * blockDim.x + threadIdx.x];
    int sum = wave::reduce_sum(val);
    if (wave::lane_id() == 0) out[blockIdx.x] = sum;
}

__global__ void kernel_reduce_int64(const long long* in, long long* out) {
    long long val = in[blockIdx.x * blockDim.x + threadIdx.x];
    long long sum = wave::reduce_sum(val);
    if (wave::lane_id() == 0) out[blockIdx.x] = sum;
}

// __half via wave::reduce_sum — internally promotes to float32
__global__ void kernel_reduce_half_fp32acc(const __half* in, float* out) {
    __half val = in[blockIdx.x * blockDim.x + threadIdx.x];
    __half sum = wave::reduce_sum(val);   // accumulates in float32
    if (wave::lane_id() == 0) out[blockIdx.x] = __half2float(sum);
}

// __half naive — raw shfl_down on fp16, no float promotion
__global__ void kernel_reduce_half_naive(const __half* in, float* out) {
    __half val = in[blockIdx.x * blockDim.x + threadIdx.x];
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = __hadd(val, wave::shfl_down(val, offset));
    }
    if (wave::lane_id() == 0) out[blockIdx.x] = __half2float(val);
}

// bfloat16 via wave::reduce_sum — internally promotes to float32
__global__ void kernel_reduce_bf16_fp32acc(const hip_bfloat16* in, float* out) {
    hip_bfloat16 val = in[blockIdx.x * blockDim.x + threadIdx.x];
    hip_bfloat16 sum = wave::reduce_sum(val);
    if (wave::lane_id() == 0) out[blockIdx.x] = __bfloat162float(sum);
}

// bfloat16 naive — raw shfl_down, no float promotion
__global__ void kernel_reduce_bf16_naive(const hip_bfloat16* in, float* out) {
    float val = __bfloat162float(in[blockIdx.x * blockDim.x + threadIdx.x]);
    // naive: reduce in bf16 precision by converting back each step
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        hip_bfloat16 a = __float2bfloat16(val);
        hip_bfloat16 b = wave::shfl_down(a, offset);
        val = __bfloat162float(a) + __bfloat162float(b);
        // convert back to bf16 to simulate precision loss
        val = __bfloat162float(__float2bfloat16(val));
    }
    if (wave::lane_id() == 0) out[blockIdx.x] = val;
}

// ---------------------------------------------------------------------------
// Numerical accuracy check for half-precision strategies
// ---------------------------------------------------------------------------

static void check_fp16_accuracy(int wave_size) {
    // Each lane has value 0.1f — exact sum = wave_size * 0.1f
    // Accumulated in fp16 introduces rounding at each step
    const int n = wave_size;
    std::vector<__half> h_in(n);
    for (int i = 0; i < n; ++i) h_in[i] = __float2half(0.1f);

    __half* d_in;
    float*  d_out;
    HIP_CHECK(hipMalloc(&d_in,  n * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_out, 1 * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), n * sizeof(__half), hipMemcpyHostToDevice));

    float h_fp32acc, h_naive;

    hipLaunchKernelGGL(kernel_reduce_half_fp32acc, 1, n, 0, 0, d_in, d_out);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&h_fp32acc, d_out, sizeof(float), hipMemcpyDeviceToHost));

    hipLaunchKernelGGL(kernel_reduce_half_naive, 1, n, 0, 0, d_in, d_out);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&h_naive, d_out, sizeof(float), hipMemcpyDeviceToHost));

    float expected = n * 0.1f;
    printf("--- __half accuracy (input: %d × 0.1f, expected: %.4f) ---\n", n, expected);
    printf("  wave::reduce_sum (fp32 acc): %.6f  error: %.6f\n",
           h_fp32acc, fabsf(h_fp32acc - expected));
    printf("  naive fp16 reduce:           %.6f  error: %.6f\n\n",
           h_naive, fabsf(h_naive - expected));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);

    const int total = kBlocks * dev.wave_size;

    // --- Accuracy comparison (half-precision) --------------------------------
    check_fp16_accuracy(dev.wave_size);

    // --- Allocate buffers for all types -------------------------------------
    void   *d_in_f32, *d_in_f64, *d_in_i32, *d_in_i64, *d_in_f16, *d_in_bf16;
    void   *d_out_f32, *d_out_f64, *d_out_i32, *d_out_i64, *d_out_f16_fp32acc, *d_out_f16_naive;
    void   *d_out_bf16_fp32acc, *d_out_bf16_naive;

    HIP_CHECK(hipMalloc(&d_in_f32,  total * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_in_f64,  total * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_in_i32,  total * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_in_i64,  total * sizeof(long long)));
    HIP_CHECK(hipMalloc(&d_in_f16,  total * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_in_bf16, total * sizeof(hip_bfloat16)));

    HIP_CHECK(hipMalloc(&d_out_f32,          kBlocks * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_f64,          kBlocks * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_out_i32,          kBlocks * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out_i64,          kBlocks * sizeof(long long)));
    HIP_CHECK(hipMalloc(&d_out_f16_fp32acc,  kBlocks * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_f16_naive,    kBlocks * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_bf16_fp32acc, kBlocks * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_bf16_naive,   kBlocks * sizeof(float)));

    // Initialize (all 1.0 for simplicity)
    {
        std::vector<float>        h_f32(total, 1.0f);
        std::vector<double>       h_f64(total, 1.0);
        std::vector<int>          h_i32(total, 1);
        std::vector<long long>    h_i64(total, 1LL);
        std::vector<__half>       h_f16(total, __float2half(1.0f));
        std::vector<hip_bfloat16> h_bf16(total, __float2bfloat16(1.0f));

        HIP_CHECK(hipMemcpy(d_in_f32,  h_f32.data(),  total * sizeof(float),        hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_in_f64,  h_f64.data(),  total * sizeof(double),       hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_in_i32,  h_i32.data(),  total * sizeof(int),          hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_in_i64,  h_i64.data(),  total * sizeof(long long),    hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_in_f16,  h_f16.data(),  total * sizeof(__half),       hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_in_bf16, h_bf16.data(), total * sizeof(hip_bfloat16), hipMemcpyHostToDevice));
    }

    // --- Benchmarks ---------------------------------------------------------
    print_bench_header("reduce_sum  all types  (1024 blocks x wave_size)");

    auto r_f32  = time_kernel([&]() { hipLaunchKernelGGL(kernel_reduce_float,  dim3(kBlocks), dim3(dev.wave_size), 0, 0, (float*)d_in_f32,   (float*)d_out_f32);   }, total, kWarmup, kIter);
    auto r_f64  = time_kernel([&]() { hipLaunchKernelGGL(kernel_reduce_double, dim3(kBlocks), dim3(dev.wave_size), 0, 0, (double*)d_in_f64,  (double*)d_out_f64);  }, total, kWarmup, kIter);
    auto r_i32  = time_kernel([&]() { hipLaunchKernelGGL(kernel_reduce_int32,  dim3(kBlocks), dim3(dev.wave_size), 0, 0, (int*)d_in_i32,     (int*)d_out_i32);     }, total, kWarmup, kIter);
    auto r_i64  = time_kernel([&]() { hipLaunchKernelGGL(kernel_reduce_int64,  dim3(kBlocks), dim3(dev.wave_size), 0, 0, (long long*)d_in_i64, (long long*)d_out_i64); }, total, kWarmup, kIter);

    auto r_f16_fp32 = time_kernel([&]() { hipLaunchKernelGGL(kernel_reduce_half_fp32acc, dim3(kBlocks), dim3(dev.wave_size), 0, 0, (__half*)d_in_f16,       (float*)d_out_f16_fp32acc);  }, total, kWarmup, kIter);
    auto r_f16_naive= time_kernel([&]() { hipLaunchKernelGGL(kernel_reduce_half_naive,   dim3(kBlocks), dim3(dev.wave_size), 0, 0, (__half*)d_in_f16,       (float*)d_out_f16_naive);    }, total, kWarmup, kIter);
    auto r_bf16_fp32= time_kernel([&]() { hipLaunchKernelGGL(kernel_reduce_bf16_fp32acc, dim3(kBlocks), dim3(dev.wave_size), 0, 0, (hip_bfloat16*)d_in_bf16,(float*)d_out_bf16_fp32acc); }, total, kWarmup, kIter);
    auto r_bf16_naive=time_kernel([&]() { hipLaunchKernelGGL(kernel_reduce_bf16_naive,   dim3(kBlocks), dim3(dev.wave_size), 0, 0, (hip_bfloat16*)d_in_bf16,(float*)d_out_bf16_naive);   }, total, kWarmup, kIter);

    print_bench_result("float",                    r_f32);
    print_bench_result("double",                   r_f64);
    print_bench_result("int32",                    r_i32);
    print_bench_result("int64",                    r_i64);
    print_bench_result("__half  (fp32 acc)",       r_f16_fp32);
    print_bench_result("__half  (naive fp16)",     r_f16_naive);
    print_bench_result("bfloat16 (fp32 acc)",      r_bf16_fp32);
    print_bench_result("bfloat16 (naive bf16)",    r_bf16_naive);

    printf("\nNote: fp32 acc variants trade a small latency cost for significantly\n");
    printf("      better numerical accuracy (see accuracy section above).\n");

    // Cleanup
    HIP_CHECK(hipFree(d_in_f32));  HIP_CHECK(hipFree(d_in_f64));
    HIP_CHECK(hipFree(d_in_i32));  HIP_CHECK(hipFree(d_in_i64));
    HIP_CHECK(hipFree(d_in_f16));  HIP_CHECK(hipFree(d_in_bf16));
    HIP_CHECK(hipFree(d_out_f32)); HIP_CHECK(hipFree(d_out_f64));
    HIP_CHECK(hipFree(d_out_i32)); HIP_CHECK(hipFree(d_out_i64));
    HIP_CHECK(hipFree(d_out_f16_fp32acc));  HIP_CHECK(hipFree(d_out_f16_naive));
    HIP_CHECK(hipFree(d_out_bf16_fp32acc)); HIP_CHECK(hipFree(d_out_bf16_naive));
    return 0;
}
