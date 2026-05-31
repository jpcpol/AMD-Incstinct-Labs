#pragma once

// Architecture detection
#if defined(__HIP_PLATFORM_AMD__) || defined(__AMDGCN__)
    #define WP_AMD 1
    #define WP_NVIDIA 0
#elif defined(__HIP_PLATFORM_NVIDIA__) || defined(__CUDA_ARCH__)
    #define WP_AMD 0
    #define WP_NVIDIA 1
#else
    #error "wave_primitives: unsupported platform"
#endif

// On AMD, __ballot returns uint64_t covering the full wavefront.
// On NVIDIA, __ballot_sync returns uint32_t (32 lanes).
// We normalize to uint64_t everywhere.
#if WP_NVIDIA
    // Provide a uniform 64-bit mask type — upper 32 bits are always 0 on NVIDIA.
    using wp_mask_t = unsigned long long;
    #define WP_FULL_MASK 0xffffffffULL
#else
    using wp_mask_t = unsigned long long;
    #define WP_FULL_MASK 0xffffffffffffffffULL
#endif

// warpSize is early-folded by the compiler in ROCm 7+ and CUDA 12+,
// meaning it behaves like a compile-time constant in loops and templates.
// We rely on this behavior — no explicit template<int WarpSize> dispatch needed.
