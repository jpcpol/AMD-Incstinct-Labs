// ---------------------------------------------------------------------------
// fa_decode_op.hip — Step decode: wrap the split-KV decode kernel as a torch op.
//
// Registers torch.ops.amdinstinct.flash_attn_decode(q, k_cache, v_cache) -> o.
//   q       : [nQH, D]      fp16, the new token's query (per head)
//   k_cache : [nKVH, N, D]  fp16, the KV-cache (N = current length)
//   v_cache : [nKVH, N, D]  fp16
//   returns : [nQH, D]      fp16
// GQA + (trivial) causal are intrinsic. W (split-KV waves) picked by D here (4).
//
// Does ONLY decode attention — projections/RoPE/cache-append stay in PyTorch. Same
// philosophy as the prefill op (2-A). Build: torch_op/setup_decode.py.
// ---------------------------------------------------------------------------
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include "../fa_decode_kernel.hpp"

static void chk(const torch::Tensor& t, const char* n, int dims){
    TORCH_CHECK(t.is_cuda(), n, " must be a GPU tensor");
    TORCH_CHECK(t.scalar_type()==torch::kHalf, n, " must be fp16");
    TORCH_CHECK(t.is_contiguous(), n, " must be contiguous");
    TORCH_CHECK(t.dim()==dims, n, " wrong rank");
}

torch::Tensor flash_attn_decode(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache){
    chk(q,"q",2); chk(k_cache,"k_cache",3); chk(v_cache,"v_cache",3);
    const int nQH  = q.size(0);
    const int D    = q.size(1);
    const int nKVH = k_cache.size(0);
    const int N    = k_cache.size(1);
    TORCH_CHECK(k_cache.size(2)==D && v_cache.size(2)==D, "cache head_dim mismatch");
    TORCH_CHECK(v_cache.size(0)==nKVH && v_cache.size(1)==N, "k/v cache shape mismatch");
    TORCH_CHECK(nQH % nKVH == 0, "nQH must be a multiple of nKVH (GQA)");
    TORCH_CHECK(D==64 || D==128, "D must be 64 or 128");

    auto o = torch::empty({nQH, D}, q.options());
    auto stream = at::hip::getCurrentHIPStream().stream();
    fa_decode::DCfg c{D, N, nQH, nKVH};
    const int W = 4;   // split-KV waves; 4 is the default (see DESIGN_decode_kvcache.md)

    hipError_t e = fa_decode::launch(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()),
        c, W, stream);
    TORCH_CHECK(e==hipSuccess, "fa_decode launch failed: ", hipGetErrorString(e));
    return o;
}

TORCH_LIBRARY_FRAGMENT(amdinstinct, m){
    m.def("flash_attn_decode(Tensor q, Tensor k_cache, Tensor v_cache) -> Tensor");
}
TORCH_LIBRARY_IMPL(amdinstinct, CUDA, m){
    m.impl("flash_attn_decode", flash_attn_decode);
}
