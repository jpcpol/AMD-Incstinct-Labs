#include "wave_primitives/wave_primitives.hpp"
#include "wave_primitives/wave_scan_dpp.hpp"
#include "bench_common.hpp"
#include <cstdint>

// DPP scan_inclusive_sum latency across types (float/double/int32/int64).
// Correctness for all types/ops is covered by tests/probe_scan_dpp.hip.

static constexpr int kBlocks = 1024;
static constexpr int kWarmup = 100;
static constexpr int kIter   = 1000;
static constexpr int kReps   = 4096;

template <typename T> __global__ void bench_scan(const T* __restrict__ in, T* __restrict__ out) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    T val = in[gid]; T acc = T(0);
    #pragma unroll 1
    for (int r = 0; r < kReps; ++r) {
        T s = wave::dpp::scan_inclusive_sum(val);
        acc = acc + s;
        val = s + (T)threadIdx.x + (T)1;
    }
    if (threadIdx.x == 63) out[blockIdx.x] = acc;
}

template <typename T>
void run(const char* tn, const DeviceInfo& dev) {
    const int total = kBlocks * dev.wave_size;
    const double work = (double)total * kReps;
    T *d_in,*d_out;
    HIP_CHECK(hipMalloc(&d_in,total*sizeof(T))); HIP_CHECK(hipMalloc(&d_out,kBlocks*sizeof(T)));
    { std::vector<T> h(total); for(int i=0;i<total;i++) h[i]=(T)1; HIP_CHECK(hipMemcpy(d_in,h.data(),total*sizeof(T),hipMemcpyHostToDevice)); }
    auto r = time_kernel([&](){ hipLaunchKernelGGL((bench_scan<T>),dim3(kBlocks),dim3(dev.wave_size),0,0,d_in,d_out);},work,kWarmup,kIter);
    char lbl[64]; snprintf(lbl,sizeof lbl,"%s dpp::scan_inclusive_sum",tn); print_bench_result(lbl,r);
    HIP_CHECK(hipFree(d_in)); HIP_CHECK(hipFree(d_out));
}

int main(){
    const DeviceInfo dev = get_device_info(); print_device_info(dev);
    print_bench_header("DPP scan_inclusive_sum across types (kReps amortized)");
    run<float>("f32",dev); run<double>("f64",dev); run<int32_t>("i32",dev); run<int64_t>("i64",dev);
    return 0;
}
