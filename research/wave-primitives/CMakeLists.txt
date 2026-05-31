cmake_minimum_required(VERSION 3.21)

# wave-primitives — header-only library + tests + benchmarks

add_library(wave-primitives INTERFACE)
target_include_directories(wave-primitives INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# --- Correctness test -------------------------------------------------------
add_executable(wave_test_correctness tests/test_correctness.hip)
target_link_libraries(wave_test_correctness PRIVATE wave-primitives hip::host)
set_target_properties(wave_test_correctness PROPERTIES
    HIP_ARCHITECTURES "${GPU_TARGETS}"
)

# --- Benchmark vs hipCUB ----------------------------------------------------
find_package(hipcub QUIET HINTS "${ROCM_PATH}/include/hipcub")

if(hipcub_FOUND OR EXISTS "${ROCM_PATH}/include/hipcub/hipcub.hpp")
    add_executable(wave_bench_reduce benchmarks/bench_reduce.hip)
    target_link_libraries(wave_bench_reduce PRIVATE wave-primitives hip::host)
    target_include_directories(wave_bench_reduce PRIVATE "${ROCM_PATH}/include")
    set_target_properties(wave_bench_reduce PROPERTIES
        HIP_ARCHITECTURES "${GPU_TARGETS}"
    )
    message(STATUS "wave-primitives: hipCUB found — bench_reduce enabled")
else()
    message(STATUS "wave-primitives: hipCUB not found — bench_reduce skipped")
endif()
