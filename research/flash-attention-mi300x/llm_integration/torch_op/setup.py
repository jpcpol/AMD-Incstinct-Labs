#!/usr/bin/env python3
"""
setup.py — build the amdinstinct.flash_attn PyTorch extension (Step 2-A).

Build (on the VM, inside the ROCm container with torch-rocm installed):
    python setup.py build_ext --inplace
then in Python:
    import torch
    torch.ops.load_library("./amdinstinct_fa.*.so")   # or just `import` after install
    o = torch.ops.amdinstinct.flash_attn(q, k, v)

WHY this is the fragile step (per Paso2_comparacion_vias.md, via A): the torch-rocm
ABI must match the hipcc used here. Common failure modes and fixes, documented so the
VM session debugs fast instead of burning hours:

  - "RuntimeError: ... undefined symbol": torch ABI mismatch. Ensure the torch in the
    container is the ROCm build (`torch.version.hip` is not None) and rebuild clean
    (`rm -rf build *.so`).
  - hipcc not finding torch headers: BuildExtension injects them; if it doesn't, the
    container's torch is likely CPU-only. Check `python -c "import torch;print(torch.version.hip)"`.
  - GPU arch: we force gfx942. If the VM is a different CDNA, change GPU_ARCH below.
  - The kernel is header-only (fa_attn_kernel.hpp); only fa_torch_op.hip compiles.

This file is staged COLD. It is not run locally (no GPU). The first real build is on
the VM — but everything that can be decided in advance (sources, flags, arch, ABI
notes) is fixed here so the VM session is build-and-run, not design.
"""
import os
from setuptools import setup

# torch's HIP extension helpers. Import guarded so the file is at least importable
# for inspection on a CPU box (it won't build there, but won't crash on import-check).
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    _HAVE_TORCH = True
except Exception as exc:  # noqa: BLE001
    _HAVE_TORCH = False
    _IMPORT_ERR = exc

GPU_ARCH = os.environ.get("GPU_ARCH", "gfx942")  # MI300X by default; override via env

if not _HAVE_TORCH:
    raise SystemExit(
        f"torch.utils.cpp_extension unavailable ({_IMPORT_ERR}).\n"
        "Run this on the VM inside the ROCm container with torch-rocm installed."
    )

setup(
    name="amdinstinct_fa",
    ext_modules=[
        CUDAExtension(                       # CUDAExtension == HIP extension under ROCm
            name="amdinstinct_fa",
            sources=["fa_torch_op.hip"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                # nvcc key is reused by torch's ROCm path for hipcc flags
                "nvcc": ["-O3", "-std=c++17", f"--offload-arch={GPU_ARCH}"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
