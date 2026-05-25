cmake_minimum_required(VERSION 3.21)
project(amd-instinct-labs LANGUAGES CXX)

# HIP language support
enable_language(HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

# Default GPU target for MI300X / MI300A (gfx942)
if(NOT DEFINED GPU_TARGETS)
    set(GPU_TARGETS "gfx942" CACHE STRING "AMD GPU targets")
endif()

set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})

message(STATUS "Building for GPU targets: ${GPU_TARGETS}")

# ROCm path detection
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

# Find ROCm packages
find_package(hip REQUIRED)
find_package(rocblas QUIET)
find_package(miopen QUIET)
find_package(rccl QUIET)

# Compiler flags
set(HIP_COMPILE_FLAGS
    -O3
    -Wall
    -Wextra
    --offload-arch=${GPU_TARGETS}
)

# Subdirectories
add_subdirectory(research/wave-primitives)
add_subdirectory(research/flash-attention-mi300x)
add_subdirectory(research/infinity-fabric-allreduce)
add_subdirectory(research/dme-abstraction)
add_subdirectory(research/mlir-mfma-tiling)
