cmake_minimum_required(VERSION 3.21)

# wave-primitives is header-only — nothing to compile yet
# Future: add test executables here
add_library(wave-primitives INTERFACE)
target_include_directories(wave-primitives INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
