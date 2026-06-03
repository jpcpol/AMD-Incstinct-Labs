#include "wave_primitives/wave_primitives.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Correctness tests for wave_primitives
//
// Run on AMD Developer Cloud (MI300X, gfx942, wave64):
//   hipcc -O2 --offload-arch=gfx942 \
//         -I../include tests/test_correctness.hip -o test_correctness
//   ./test_correctness
//
// Expected output on wave64 (MI300X):
//   [ENV]  warpSize = 64 (wave64 detected)
//   [PASS] reduce_sum  float
//   [PASS] reduce_sum  double
//   [PASS] reduce_max  float
//   [PASS] reduce_min  float
//   [PASS] scan_inclusive_sum
//   [PASS] scan_exclusive_sum
//   [PASS] ballot
//   [PASS] popcount
//   All tests passed.
// ---------------------------------------------------------------------------

#define HIP_CHECK(cmd) do {                                       \
    hipError_t e = (cmd);                                         \
    if (e != hipSuccess) {                                        \
        fprintf(stderr, "HIP error %s at %s:%d\n",               \
                hipGetErrorString(e), __FILE__, __LINE__);        \
        exit(1);                                                  \
    }                                                             \
} while(0)

// ---------------------------------------------------------------------------
// Test kernels
// ---------------------------------------------------------------------------

// reduce_sum: each lane contributes its lane_id. Expected sum = 0+1+...+(warpSize-1)
__global__ void kernel_reduce_sum_float(float* out) {
    const float val = static_cast<float>(wave::lane_id());
    const float sum = wave::reduce_sum(val);
    if (wave::lane_id() == 0) out[blockIdx.x] = sum;
}

__global__ void kernel_reduce_sum_double(double* out) {
    const double val = static_cast<double>(wave::lane_id());
    const double sum = wave::reduce_sum(val);
    if (wave::lane_id() == 0) out[blockIdx.x] = sum;
}

__global__ void kernel_reduce_max_float(float* out) {
    // Each lane contributes sin(lane_id) — max should be sin(warpSize-1) or near 1.0
    // Simpler: each lane contributes lane_id, max should be warpSize-1.
    const float val = static_cast<float>(wave::lane_id());
    const float mx  = wave::reduce_max(val);
    if (wave::lane_id() == 0) out[blockIdx.x] = mx;
}

__global__ void kernel_reduce_min_float(float* out) {
    const float val = static_cast<float>(wave::lane_id()) + 1.0f; // 1..warpSize
    const float mn  = wave::reduce_min(val);
    if (wave::lane_id() == 0) out[blockIdx.x] = mn;
}

__global__ void kernel_scan_inclusive(float* out) {
    const float val    = 1.0f; // all ones → inclusive scan = 1, 2, 3, ..., warpSize
    const float prefix = wave::scan_inclusive_sum(val);
    out[blockIdx.x * warpSize + wave::lane_id()] = prefix;
}

__global__ void kernel_scan_exclusive(float* out) {
    const float val    = 1.0f; // all ones → exclusive scan = 0, 1, 2, ..., warpSize-1
    const float prefix = wave::scan_exclusive_sum(val);
    out[blockIdx.x * warpSize + wave::lane_id()] = prefix;
}

__global__ void kernel_ballot(unsigned long long* out_mask, int* out_popcount) {
    // Even lanes vote true
    const int predicate = (wave::lane_id() % 2 == 0) ? 1 : 0;
    const unsigned long long mask = wave::ballot(predicate);
    const int cnt = wave::popcount(predicate);
    if (wave::lane_id() == 0) {
        out_mask[blockIdx.x]     = mask;
        out_popcount[blockIdx.x] = cnt;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

static bool check_float(const char* name, float got, float expected, float tol = 1e-3f) {
    if (fabsf(got - expected) <= tol) {
        printf("[PASS] %s  (got %.1f, expected %.1f)\n", name, got, expected);
        return true;
    }
    printf("[FAIL] %s  got %.6f, expected %.6f\n", name, got, expected);
    return false;
}

static bool check_double(const char* name, double got, double expected, double tol = 1e-6) {
    if (fabs(got - expected) <= tol) {
        printf("[PASS] %s  (got %.1f, expected %.1f)\n", name, got, expected);
        return true;
    }
    printf("[FAIL] %s  got %.10f, expected %.10f\n", name, got, expected);
    return false;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    // Query actual warpSize for this device
    int device = 0;
    int wave_size = 0;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipDeviceGetAttribute(&wave_size, hipDeviceAttributeWarpSize, device));
    printf("[ENV]  warpSize = %d (%s detected)\n",
           wave_size, wave_size == 64 ? "wave64" : "wave32");

    const int num_blocks = 1;
    bool all_pass = true;

    // ------- reduce_sum float -----------------------------------------------
    {
        float* d_out;
        HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
        hipLaunchKernelGGL(kernel_reduce_sum_float, dim3(num_blocks), dim3(wave_size), 0, 0, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        float h_out;
        HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(float), hipMemcpyDeviceToHost));
        // Expected: 0 + 1 + ... + (warpSize-1) = warpSize*(warpSize-1)/2
        float expected = static_cast<float>(wave_size * (wave_size - 1) / 2);
        all_pass &= check_float("reduce_sum  float", h_out, expected);
        HIP_CHECK(hipFree(d_out));
    }

    // ------- reduce_sum double ----------------------------------------------
    {
        double* d_out;
        HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(double)));
        hipLaunchKernelGGL(kernel_reduce_sum_double, dim3(num_blocks), dim3(wave_size), 0, 0, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        double h_out;
        HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(double), hipMemcpyDeviceToHost));
        double expected = static_cast<double>(wave_size * (wave_size - 1) / 2);
        all_pass &= check_double("reduce_sum  double", h_out, expected);
        HIP_CHECK(hipFree(d_out));
    }

    // ------- reduce_max float -----------------------------------------------
    {
        float* d_out;
        HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
        hipLaunchKernelGGL(kernel_reduce_max_float, dim3(num_blocks), dim3(wave_size), 0, 0, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        float h_out;
        HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(float), hipMemcpyDeviceToHost));
        float expected = static_cast<float>(wave_size - 1);
        all_pass &= check_float("reduce_max  float", h_out, expected);
        HIP_CHECK(hipFree(d_out));
    }

    // ------- reduce_min float -----------------------------------------------
    {
        float* d_out;
        HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
        hipLaunchKernelGGL(kernel_reduce_min_float, dim3(num_blocks), dim3(wave_size), 0, 0, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        float h_out;
        HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(float), hipMemcpyDeviceToHost));
        all_pass &= check_float("reduce_min  float", h_out, 1.0f);
        HIP_CHECK(hipFree(d_out));
    }

    // ------- scan_inclusive_sum ---------------------------------------------
    {
        float* d_out;
        HIP_CHECK(hipMalloc(&d_out, num_blocks * wave_size * sizeof(float)));
        hipLaunchKernelGGL(kernel_scan_inclusive, dim3(num_blocks), dim3(wave_size), 0, 0, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> h_out(wave_size, 0.0f);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, wave_size * sizeof(float), hipMemcpyDeviceToHost));
        bool pass = true;
        for (int i = 0; i < wave_size; ++i) {
            if (fabsf(h_out[i] - static_cast<float>(i + 1)) > 0.5f) {
                printf("[FAIL] scan_inclusive_sum lane %d: got %.1f expected %.1f\n",
                       i, h_out[i], static_cast<float>(i + 1));
                pass = false;
                break;
            }
        }
        if (pass) printf("[PASS] scan_inclusive_sum\n");
        all_pass &= pass;
        HIP_CHECK(hipFree(d_out));
    }

    // ------- scan_exclusive_sum ---------------------------------------------
    {
        float* d_out;
        HIP_CHECK(hipMalloc(&d_out, num_blocks * wave_size * sizeof(float)));
        hipLaunchKernelGGL(kernel_scan_exclusive, dim3(num_blocks), dim3(wave_size), 0, 0, d_out);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> h_out(wave_size, 0.0f);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, wave_size * sizeof(float), hipMemcpyDeviceToHost));
        bool pass = true;
        for (int i = 0; i < wave_size; ++i) {
            if (fabsf(h_out[i] - static_cast<float>(i)) > 0.5f) {
                printf("[FAIL] scan_exclusive_sum lane %d: got %.1f expected %.1f\n",
                       i, h_out[i], static_cast<float>(i));
                pass = false;
                break;
            }
        }
        if (pass) printf("[PASS] scan_exclusive_sum\n");
        all_pass &= pass;
        HIP_CHECK(hipFree(d_out));
    }

    // ------- ballot + popcount ----------------------------------------------
    {
        unsigned long long* d_mask;
        int* d_cnt;
        HIP_CHECK(hipMalloc(&d_mask, num_blocks * sizeof(unsigned long long)));
        HIP_CHECK(hipMalloc(&d_cnt,  num_blocks * sizeof(int)));
        hipLaunchKernelGGL(kernel_ballot, dim3(num_blocks), dim3(wave_size), 0, 0, d_mask, d_cnt);
        HIP_CHECK(hipDeviceSynchronize());
        unsigned long long h_mask;
        int h_cnt;
        HIP_CHECK(hipMemcpy(&h_mask, d_mask, sizeof(unsigned long long), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(&h_cnt,  d_cnt,  sizeof(int), hipMemcpyDeviceToHost));

        // Even lanes vote: bits 0,2,4,...,(warpSize-2) set → mask = 0x5555...
        unsigned long long expected_mask = 0;
        for (int i = 0; i < wave_size; i += 2) expected_mask |= (1ULL << i);
        int expected_cnt = wave_size / 2;

        bool pass_mask = (h_mask == expected_mask);
        bool pass_cnt  = (h_cnt == expected_cnt);

        if (pass_mask) printf("[PASS] ballot  (mask = 0x%llx)\n", h_mask);
        else           printf("[FAIL] ballot  got 0x%llx expected 0x%llx\n", h_mask, expected_mask);

        if (pass_cnt)  printf("[PASS] popcount  (cnt = %d)\n", h_cnt);
        else           printf("[FAIL] popcount  got %d expected %d\n", h_cnt, expected_cnt);

        all_pass &= pass_mask && pass_cnt;
        HIP_CHECK(hipFree(d_mask));
        HIP_CHECK(hipFree(d_cnt));
    }

    printf("\n%s\n", all_pass ? "All tests passed." : "SOME TESTS FAILED.");
    return all_pass ? 0 : 1;
}
