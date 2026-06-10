// ---------------------------------------------------------------------------
// fa_torch_op.hip — Step 2-A: wrap the verified FA kernel as a PyTorch custom op.
//
// Registers torch.ops.amdinstinct.flash_attn(q, k, v) -> o, where q/k/v/o are
// fp16 CUDA(=HIP) tensors in HEAD-MAJOR layout [heads, seq, D] (contiguous).
// Causal masking + GQA are intrinsic to the kernel (decoder LLM prefill).
//
// This op does ONLY attention — projections + RoPE stay in PyTorch (the model does
// them). That isolates the kernel under test and avoids re-implementing proj/RoPE in
// HIP (which B already validated separately). The op is what gets monkey-patched into
// a HF model's attention in patch_hf_attention.py.
//
// Build: see setup.py (python setup.py build_ext --inplace) or CMake. Requires a
// torch-rocm install whose ABI matches hipcc. The kernel itself is header-only
// (fa_attn_kernel.hpp) and was validated in Step 2-B.
//
// Contract (asserted at runtime):
//   q : [nQH, seq, D] fp16 contiguous on GPU
//   k : [nKVH, seq, D] fp16
//   v : [nKVH, seq, D] fp16
//   nQH % nKVH == 0 (GQA), D in {64,128}, seq % 16 == 0, kv-seq % 64 == 0
//   returns o : [nQH, seq, D] fp16
// ---------------------------------------------------------------------------
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include "../fa_attn_kernel.hpp"

static void check(const torch::Tensor& t, const char* name){
    TORCH_CHECK(t.is_cuda(),       name, " must be a GPU (HIP) tensor");
    TORCH_CHECK(t.scalar_type()==torch::kHalf, name, " must be fp16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim()==3,        name, " must be [heads, seq, D]");
}

torch::Tensor flash_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v){
    check(q,"q"); check(k,"k"); check(v,"v");
    const int nQH  = q.size(0);
    const int seq  = q.size(1);
    const int D    = q.size(2);
    const int nKVH = k.size(0);
    TORCH_CHECK(k.size(1)==seq && v.size(1)==seq, "k/v seq must match q");
    TORCH_CHECK(k.size(2)==D && v.size(2)==D,     "k/v head_dim must match q");
    TORCH_CHECK(nQH % nKVH == 0, "nQH must be a multiple of nKVH (GQA)");
    TORCH_CHECK(D==64 || D==128,  "D must be 64 or 128 (instantiated templates)");
    TORCH_CHECK(seq % fa_attn::Br == 0, "seq must be a multiple of 16 (Br)");
    TORCH_CHECK(seq % fa_attn::Bc == 0, "kv seq must be a multiple of 64 (Bc)");

    auto o = torch::empty_like(q);   // [nQH, seq, D] fp16
    auto stream = at::hip::getCurrentHIPStream().stream();

    hipError_t e = fa_attn::launch(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()),
        seq, nQH, nKVH, D, stream);
    TORCH_CHECK(e==hipSuccess, "fa_attn launch failed: ", hipGetErrorString(e));
    return o;
}

TORCH_LIBRARY(amdinstinct, m){
    m.def("flash_attn(Tensor q, Tensor k, Tensor v) -> Tensor");
}
TORCH_LIBRARY_IMPL(amdinstinct, CUDA, m){   // ROCm registers under the CUDA key
    m.impl("flash_attn", flash_attn);
}
