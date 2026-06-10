#!/usr/bin/env python3
"""
setup_decode.py — build the amdinstinct.flash_attn_decode op (decode step).

Build (on the VM, ROCm container with torch-rocm matching hipcc):
    python setup_decode.py build_ext --inplace

Same ROCm ABI notes as setup.py (the prefill op). The decode kernel is header-only
(fa_decode_kernel.hpp); only fa_decode_op.hip compiles. The 2-A session showed the
clean path is torch 2.10.0+rocm7.0 against the system hipcc 7.2 — use the same.
"""
import os
from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"torch cpp_extension unavailable ({exc}). Run on the VM with torch-rocm.")

GPU_ARCH = os.environ.get("GPU_ARCH", "gfx942")

setup(
    name="amdinstinct_fa_decode",
    ext_modules=[
        CUDAExtension(
            name="amdinstinct_fa_decode",
            sources=["fa_decode_op.hip"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", f"--offload-arch={GPU_ARCH}"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
