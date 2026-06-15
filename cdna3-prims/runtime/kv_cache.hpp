// ---------------------------------------------------------------------------
// cdna3/runtime/kv_cache.hpp — contiguous fp16 KV-cache for batch-1 decode.
//
// Pre-allocated at session init. Each decode step appends K/V for one token
// and calls cdna3::attn::decode with N = cache.len.
//
// Layout: [n_layers × n_kv_heads × max_seq_len × head_dim] (contiguous fp16).
// No paging — greedy batch-1 only. Paged KV-cache is future work.
// ---------------------------------------------------------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdlib>
#include <cassert>

namespace cdna3 {
namespace runtime {

struct KVCache {
    __half* K = nullptr;   // device: [n_layers × n_kv_heads × max_seq_len × head_dim]
    __half* V = nullptr;
    int n_layers;
    int n_kv_heads;
    int max_seq_len;
    int head_dim;
    int len = 0;           // tokens currently stored

    void allocate(int nl, int nkvh, int max_seq, int hdim) {
        n_layers    = nl;
        n_kv_heads  = nkvh;
        max_seq_len = max_seq;
        head_dim    = hdim;
        size_t bytes = (size_t)nl * nkvh * max_seq * hdim * sizeof(__half);
        hipMalloc(&K, bytes);
        hipMalloc(&V, bytes);
        hipMemset(K, 0, bytes);
        hipMemset(V, 0, bytes);
    }

    void free_buffers() {
        if (K) { hipFree(K); K = nullptr; }
        if (V) { hipFree(V); V = nullptr; }
    }

    // Pointer to K slice for a given layer, starting at token offset `pos`.
    __half* K_layer(int layer, int pos = 0) const {
        return K + ((size_t)layer * n_kv_heads * max_seq_len + pos) * head_dim;
    }
    __half* V_layer(int layer, int pos = 0) const {
        return V + ((size_t)layer * n_kv_heads * max_seq_len + pos) * head_dim;
    }
};

} // namespace runtime
} // namespace cdna3
