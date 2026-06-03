cmake_minimum_required(VERSION 3.21)

add_library(wave-primitives INTERFACE)
target_include_directories(wave-primitives INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# --- Correctness test -------------------------------------------------------
add_executable(wave_test_correctness tests/test_correctness.hip)
target_link_libraries(wave_test_correctness PRIVATE wave-primitives hip::host)
set_target_properties(wave_test_correctness PROPERTIES HIP_ARCHITECTURES "${GPU_TARGETS}")

# --- Benchmarks (require hipCUB) --------------------------------------------
find_path(HIPCUB_INCLUDE_DIR hipcub/hipcub.hpp
    HINTS "${ROCM_PATH}/include" "/opt/rocm/include"
)

if(HIPCUB_INCLUDE_DIR)
    message(STATUS "wave-primitives: hipCUB found at ${HIPCUB_INCLUDE_DIR} — benchmarks enabled")

    set(BENCH_COMPILE_FLAGS -O3)
    set(BENCH_INCLUDES wave-primitives hip::host)

    foreach(bench reduce scan all_types block_reduce reduce_dpp reduce_dpp_types)
        add_executable(wave_bench_${bench} benchmarks/bench_${bench}.hip)
        target_link_libraries(wave_bench_${bench} PRIVATE ${BENCH_INCLUDES})
        target_include_directories(wave_bench_${bench} PRIVATE ${HIPCUB_INCLUDE_DIR})
        target_compile_options(wave_bench_${bench} PRIVATE ${BENCH_COMPILE_FLAGS})
        set_target_properties(wave_bench_${bench} PROPERTIES HIP_ARCHITECTURES "${GPU_TARGETS}")
    endforeach()
else()
    message(STATUS "wave-primitives: hipCUB not found — benchmarks skipped (install ROCm or set ROCM_PATH)")
endif()
