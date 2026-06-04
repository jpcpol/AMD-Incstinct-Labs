cmake_minimum_required(VERSION 3.21)

set(FA_INC
    ${CMAKE_SOURCE_DIR}/research/wave-primitives/include
    ${CMAKE_SOURCE_DIR}/research/wave-primitives/benchmarks
)

# Step 1: naive FA (portable reductions, sync loads) — the baseline
add_executable(fa_naive kernels/fa_naive.hip)
target_include_directories(fa_naive PRIVATE ${FA_INC})
target_compile_definitions(fa_naive PRIVATE FA_HEAD_DIM=64 FA_BC=64)
set_target_properties(fa_naive PROPERTIES
    CXX_STANDARD 17
    COMPILE_FLAGS "--offload-arch=${GPU_TARGETS}"
)

# Step 2: DPP reductions (three Bc variants to test LDS-relaxation hypothesis)
foreach(BC 64 128 256)
    add_executable(fa_dpp_bc${BC} kernels/fa_dpp.hip)
    target_include_directories(fa_dpp_bc${BC} PRIVATE ${FA_INC})
    target_compile_definitions(fa_dpp_bc${BC} PRIVATE FA_HEAD_DIM=64 FA_BC=${BC})
    set_target_properties(fa_dpp_bc${BC} PROPERTIES
        CXX_STANDARD 17
        COMPILE_FLAGS "--offload-arch=${GPU_TARGETS}"
    )
endforeach()

# Placeholder targets for steps 3-4 (not implemented yet)
# add_executable(fa_dme     kernels/fa_dme.hip)       # step 3: DME prefetch
# add_executable(fa_mfma    kernels/fa_mfma.hip)      # step 4: MFMA tuning
