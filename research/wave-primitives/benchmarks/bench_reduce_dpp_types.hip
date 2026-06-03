#include "wave_primitives/wave_primitives.hpp"
#include "wave_primitives/wave_reduce_dpp.hpp"
#include "bench_common.hpp"
#include <hipcub/hipcub.hpp>
#include <cstdint>

// Validate + benchmark the generalized DPP reductions across types and ops.
//   sum/max/min  ×  float, double, int32, int64
// Correctness is checked against a CPU reference before any timing.
// Result is on lane 63 (no broadcast) for the bench kernels; we read lane 63.

static constexpr int kBlocks = 1024;
static constexpr int kWarmup = 100;
static constexpr int kIter   = 1000;
static constexpr int kReps   = 4096;

// --- correctness kernels: one reduction over one wave, result -> out[0] -----
template <typename T, int OP>  // OP: 0=sum 1=max 2=min
__global__ void verify_kernel(const T* in, T* out) {
    T v = in[threadIdx.x];
    T s;
    if      (OP == 0) s = wave::dpp::reduce_sum_bcast(v);
    else if (OP == 1) s = wave::dpp::reduce_max_bcast(v);
    else              s = wave::dpp::reduce_min_bcast(v);
    if (wave::lane_id() == 0) out[0] = s;  // bcast → valid on all lanes
}

// --- amortized bench kernels (result on lane 63) ----------------------------
template <typename T, int OP>
__global__ void bench_kernel(const T* __restrict__ in, T* __restrict__ out) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    T val = in[gid];
    T acc = T(0);
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        T s;
        if      (OP == 0) s = wave::dpp::reduce_sum(val);
        else if (OP == 1) s = wave::dpp::reduce_max(val);
        else              s = wave::dpp::reduce_min(val);
        acc = acc + s;
        // per-lane perturbation to defeat DCE / constant folding
        val = s + static_cast<T>(threadIdx.x) + T(1);
    }
    if (wave::lane_id() == 63) out[blockIdx.x] = acc;
}

template <typename T>
void run_type(const char* type_name, const DeviceInfo& dev) {
    const int W = dev.wave_size;
    const int total = kBlocks * W;
    const double work = static_cast<double>(total) * kReps;

    std::vector<T> h(total);
    for (int i = 0; i < total; ++i) h[i] = static_cast<T>(i % W);
    T *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in,  total   * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_out, kBlocks * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_in, h.data(), total * sizeof(T), hipMemcpyHostToDevice));

    // CPU references over one wave of values 0..W-1
    T exp_sum = T(0), exp_max = static_cast<T>(W - 1), exp_min = T(0);
    for (int i = 0; i < W; ++i) exp_sum = exp_sum + static_cast<T>(i);

    auto verify = [&](int op, T expected, const char* opn, void(*k)(const T*, T*)) {
        hipLaunchKernelGGL(k, dim3(1), dim3(W), 0, 0, d_in, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        T r{};
        HIP_CHECK(hipMemcpy(&r, d_out, sizeof(T), hipMemcpyDeviceToHost));
        double diff = static_cast<double>(r) - static_cast<double>(expected);
        bool ok = (diff < 0 ? -diff : diff) < 0.5;
        printf("    %-4s %-6s got %12.1f  expected %12.1f  [%s]\n",
               type_name, opn, static_cast<double>(r), static_cast<double>(expected),
               ok ? "PASS" : "FAIL");
        return ok;
    };

    printf("  --- %s correctness ---\n", type_name);
    bool ok = true;
    ok &= verify(0, exp_sum, "sum", verify_kernel<T,0>);
    ok &= verify(1, exp_max, "max", verify_kernel<T,1>);
    ok &= verify(2, exp_min, "min", verify_kernel<T,2>);
    if (!ok) { printf("  >>> %s FAILED — skipping bench\n", type_name); HIP_CHECK(hipFree(d_in)); HIP_CHECK(hipFree(d_out)); return; }

    auto bs = time_kernel([&](){ hipLaunchKernelGGL((bench_kernel<T,0>), dim3(kBlocks), dim3(W),0,0,d_in,d_out);}, work,kWarmup,kIter);
    auto bm = time_kernel([&](){ hipLaunchKernelGGL((bench_kernel<T,1>), dim3(kBlocks), dim3(W),0,0,d_in,d_out);}, work,kWarmup,kIter);
    auto bn = time_kernel([&](){ hipLaunchKernelGGL((bench_kernel<T,2>), dim3(kBlocks), dim3(W),0,0,d_in,d_out);}, work,kWarmup,kIter);
    char lbl[64];
    snprintf(lbl,sizeof lbl,"%s dpp::reduce_sum", type_name); print_bench_result(lbl, bs);
    snprintf(lbl,sizeof lbl,"%s dpp::reduce_max", type_name); print_bench_result(lbl, bm);
    snprintf(lbl,sizeof lbl,"%s dpp::reduce_min", type_name); print_bench_result(lbl, bn);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
}

int main() {
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);
    print_bench_header("Generalized DPP reductions — sum/max/min × float/double/int32/int64");
    run_type<float>  ("f32", dev);
    run_type<double> ("f64", dev);
    run_type<int32_t>("i32", dev);
    run_type<int64_t>("i64", dev);
    return 0;
}
