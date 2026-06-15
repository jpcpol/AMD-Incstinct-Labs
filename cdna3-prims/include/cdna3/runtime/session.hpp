// ---------------------------------------------------------------------------
// cdna3/runtime/session.hpp — Stage-C minimal LLM inference session.
//
// Wraps the validated cdna3::attn attention module into a greedy, batch-1
// generation loop: prefill the prompt, then decode token-by-token using a
// contiguous fp16 KV-cache. The attention path is cdna3::attn::{prefill,decode}
// (Stage-B validated); the surrounding ops are the reference kernels in
// forward.hpp.
//
// fp32-acc hardening (C4): d_hidden, d_norm, d_gate, d_up, d_mlp are float to
// eliminate 24-layer residual drift. Weights remain fp16. Attention I/O stays
// fp16. H3/T-030 confirmed fp32-acc is faster AND more precise on CDNA3.
//
// ## Usage
//   cdna3::runtime::Session sess("/tmp/qwen25_1b", /*max_seq*/4096);
//   auto out = sess.generate(input_ids, {/*max_new_tokens*/256});
// ---------------------------------------------------------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <string>
#include <cmath>

#include "../attention.hpp"   // cdna3::attn::{prefill,decode}
#include "model_loader.hpp"
#include "kv_cache.hpp"
#include "forward.hpp"

namespace cdna3 {
namespace runtime {

struct GenerateCfg {
    int max_new_tokens = 64;
    int eos_token_id   = -1;   // stop early if produced; -1 = run to max
};

class Session {
    ModelLoader loader_;
    const ModelWeights* w_ = nullptr;
    KVCache kv_;
    int max_seq_;

    // scratch device buffers (allocated for the widest stage: prefill of prompt)
    // fp32-acc: hidden state and MLP activations in float; attn I/O stays __half.
    float  *d_hidden=nullptr, *d_norm=nullptr;           // [T, H] float
    __half *d_q=nullptr, *d_k=nullptr, *d_v=nullptr;    // [T, qd/kvd] fp16
    __half *d_attn_tok=nullptr;                          // [T, qd] fp16 (attn output)
    float  *d_oproj=nullptr;                             // [T, H] float (O-proj result)
    float  *d_gate=nullptr, *d_up=nullptr, *d_mlp=nullptr; // [T, F/H] float
    __half *d_logits=nullptr;                            // [1, V] fp16
    __half *d_kpack=nullptr, *d_vpack=nullptr;           // [n_kv_heads, Nkeys, D] fp16
    float  *d_cos=nullptr, *d_sin=nullptr;
    int    *d_ids=nullptr, *d_argmax=nullptr;
    int    cap_tokens_=0;

    void alloc_scratch(int T){
        if (T <= cap_tokens_) return;
        free_scratch();
        const auto& c = w_->cfg;
        int H=c.hidden, F=c.ffn_dim, V=c.vocab_size;
        int qd=c.n_heads*c.head_dim, kvd=c.n_kv_heads*c.head_dim;
        // float buffers (4 bytes each)
        hipMalloc(&d_hidden,(size_t)T*H*4);
        hipMalloc(&d_norm,  (size_t)T*H*4);
        hipMalloc(&d_oproj, (size_t)T*H*4);
        hipMalloc(&d_gate,  (size_t)T*F*4);
        hipMalloc(&d_up,    (size_t)T*F*4);
        hipMalloc(&d_mlp,   (size_t)T*H*4);
        // fp16 buffers (2 bytes each)
        hipMalloc(&d_q,     (size_t)T*qd*2);
        hipMalloc(&d_k,     (size_t)T*kvd*2);
        hipMalloc(&d_v,     (size_t)T*kvd*2);
        hipMalloc(&d_attn_tok,(size_t)T*qd*2);
        hipMalloc(&d_logits,(size_t)V*2);                   // always 1 token
        hipMalloc(&d_kpack, (size_t)c.n_kv_heads*max_seq_*c.head_dim*2);
        hipMalloc(&d_vpack, (size_t)c.n_kv_heads*max_seq_*c.head_dim*2);
        hipMalloc(&d_cos,   (size_t)max_seq_*c.head_dim*4);
        hipMalloc(&d_sin,   (size_t)max_seq_*c.head_dim*4);
        hipMalloc(&d_ids,   (size_t)T*sizeof(int));
        hipMalloc(&d_argmax,sizeof(int));
        cap_tokens_ = T;
    }
    void free_scratch(){
        for (void* p : {(void*)d_hidden,(void*)d_norm,(void*)d_oproj,
                        (void*)d_gate,(void*)d_up,(void*)d_mlp,
                        (void*)d_q,(void*)d_k,(void*)d_v,(void*)d_attn_tok,
                        (void*)d_logits,(void*)d_kpack,(void*)d_vpack,
                        (void*)d_cos,(void*)d_sin,(void*)d_ids,(void*)d_argmax})
            if (p) hipFree(p);
        d_hidden=d_norm=d_oproj=nullptr;
        d_gate=d_up=d_mlp=nullptr;
        d_q=d_k=d_v=d_attn_tok=d_logits=nullptr;
        d_kpack=d_vpack=nullptr; d_cos=d_sin=nullptr;
        d_ids=d_argmax=nullptr; cap_tokens_=0;
    }

    // Precompute RoPE cos/sin for absolute positions [pos0, pos0+T) into d_cos/d_sin.
    void fill_rope(int pos0, int T){
        const auto& c = w_->cfg;
        int D = c.head_dim;
        std::vector<float> cosv((size_t)T*D), sinv((size_t)T*D);
        for (int s=0;s<T;++s){
            int pos = pos0 + s;
            for (int d=0; d<D/2; ++d){
                float freq = 1.f / powf(c.rope_theta, (2.f*d)/D);
                float ang = pos * freq;
                cosv[s*D + d] = cosf(ang);  cosv[s*D + d + D/2] = cosf(ang);
                sinv[s*D + d] = sinf(ang);  sinv[s*D + d + D/2] = sinf(ang);
            }
        }
        hipMemcpy(d_cos, cosv.data(), (size_t)T*D*4, hipMemcpyHostToDevice);
        hipMemcpy(d_sin, sinv.data(), (size_t)T*D*4, hipMemcpyHostToDevice);
    }

    static dim3 grid1d(size_t n, int tpb=256){ return dim3((unsigned)((n+tpb-1)/tpb)); }

    // Run all transformer layers for T tokens starting at cache position pos0.
    // d_hidden holds [T, H] float input; on return holds the final hidden (float).
    void run_layers(int T, int pos0, bool is_prefill){
        const auto& c = w_->cfg;
        int H=c.hidden, F=c.ffn_dim, D=c.head_dim;
        int qd=c.n_heads*D, kvd=c.n_kv_heads*D;
        dim3 tb(16,16);

        // float hidden → fp16 QKV (±bias for Qwen2)
        auto gemm_qkv=[&](const __half* W, const __half* B, __half* Y, int N, int K){
            dim3 gr((N+15)/16,(T+15)/16);
            hipLaunchKernelGGL(gemm_f32in_f16out_bias_kernel,gr,tb,0,0,d_norm,W,B,Y,T,N,K);
        };
        // float hidden → float MLP (gate/up/down)
        auto gemm_mlp=[&](const __half* W, const float* X, float* Y, int M, int N, int K){
            dim3 gr((N+15)/16,(M+15)/16);
            hipLaunchKernelGGL(gemm_f32in_f32out_kernel,gr,tb,0,0,X,W,Y,M,N,K);
        };
        // fp16 attn_out → float O-proj
        auto gemm_oproj=[&](const __half* W, int N, int K){
            dim3 gr((N+15)/16,(T+15)/16);
            hipLaunchKernelGGL(gemm_f16in_f32out_kernel,gr,tb,0,0,d_attn_tok,W,d_oproj,T,N,K);
        };

        fill_rope(pos0, T);

        for (int li=0; li<c.n_layers; ++li){
            const auto& L = w_->layers[li];

            // 1. attn RMSNorm: float hidden → float norm
            hipLaunchKernelGGL(rmsnorm_f32_kernel,dim3(T),dim3(256),0,0,d_hidden,L.norm_attn,d_norm,T,H,1e-6f);

            // 2. QKV projections: float norm → fp16 Q/K/V (±Qwen2 bias)
            gemm_qkv(L.Wq,L.bq,d_q, qd, H);
            gemm_qkv(L.Wk,L.bk,d_k, kvd,H);
            gemm_qkv(L.Wv,L.bv,d_v, kvd,H);

            // 3. RoPE in-place on fp16 Q and K
            hipLaunchKernelGGL(rope_kernel,grid1d((size_t)T*c.n_heads*D),dim3(256),0,0,d_q,d_cos,d_sin,T,c.n_heads,D);
            hipLaunchKernelGGL(rope_kernel,grid1d((size_t)T*c.n_kv_heads*D),dim3(256),0,0,d_k,d_cos,d_sin,T,c.n_kv_heads,D);

            // 4. append K/V into head-major cache at [pos0, pos0+T)
            append_kv(li, T, pos0);

            // 5. attention: one query at a time via cdna3::attn::decode (fp16 I/O)
            for (int q = 0; q < T; ++q){
                int Nkeys = pos0 + q + 1;
                hipLaunchKernelGGL(pack_kv_kernel,grid1d((size_t)c.n_kv_heads*Nkeys*D),dim3(256),0,0,
                                   kv_.K_layer(li,0),d_kpack,c.n_kv_heads,Nkeys,D,max_seq_);
                hipLaunchKernelGGL(pack_kv_kernel,grid1d((size_t)c.n_kv_heads*Nkeys*D),dim3(256),0,0,
                                   kv_.V_layer(li,0),d_vpack,c.n_kv_heads,Nkeys,D,max_seq_);
                const __half* q_in  = d_q       + (size_t)q*qd;
                __half*       q_out = d_attn_tok + (size_t)q*qd;
                attn::DecodeCfg dc{D,Nkeys,c.n_heads,c.n_kv_heads};
                attn::decode(q_in, d_kpack, d_vpack, q_out, dc);
            }

            // 6. O projection: fp16 attn_out → float d_oproj; residual add into float hidden
            gemm_oproj(L.Wo, H, qd);
            hipLaunchKernelGGL(add_f32_kernel,grid1d((size_t)T*H),dim3(256),0,0,d_hidden,d_oproj,(size_t)T*H);

            // 7. MLP RMSNorm: float hidden → float norm
            hipLaunchKernelGGL(rmsnorm_f32_kernel,dim3(T),dim3(256),0,0,d_hidden,L.norm_mlp,d_norm,T,H,1e-6f);

            // 8. gate/up: float norm → float; silu-mul in float; down: float → float; residual
            gemm_mlp(L.Wgate,d_norm,d_gate,T,F,H);
            gemm_mlp(L.Wup,  d_norm,d_up,  T,F,H);
            hipLaunchKernelGGL(silu_mul_f32_kernel,grid1d((size_t)T*F),dim3(256),0,0,d_gate,d_up,(size_t)T*F);
            gemm_mlp(L.Wdown,d_gate,d_mlp,T,H,F);
            hipLaunchKernelGGL(add_f32_kernel,grid1d((size_t)T*H),dim3(256),0,0,d_hidden,d_mlp,(size_t)T*H);
        }

        // final norm: float hidden → float d_norm; copy back to d_hidden
        hipLaunchKernelGGL(rmsnorm_f32_kernel,dim3(T),dim3(256),0,0,d_hidden,w_->final_norm,d_norm,T,H,1e-6f);
        hipMemcpy(d_hidden,d_norm,(size_t)T*H*4,hipMemcpyDeviceToDevice);
    }

    // Scatter token-major K/V [T, n_kv_heads, D] into cache head-major slot at pos0.
    void append_kv(int layer, int T, int pos0){
        const auto& c = w_->cfg;
        int D=c.head_dim, KVH=c.n_kv_heads;
        // Use repack kernel variant: write each (s,h,d) of d_k to cache[h, pos0+s, d].
        // We can reuse repack_kernel with the cache pointer offset baked into stride —
        // but cache stride is max_seq, not T, so do an explicit scatter.
        hipLaunchKernelGGL(scatter_kv_kernel,grid1d((size_t)T*KVH*D),dim3(256),0,0,
                           d_k,kv_.K_layer(layer,0),T,KVH,D,pos0,max_seq_);
        hipLaunchKernelGGL(scatter_kv_kernel,grid1d((size_t)T*KVH*D),dim3(256),0,0,
                           d_v,kv_.V_layer(layer,0),T,KVH,D,pos0,max_seq_);
    }

public:
    Session(const std::string& model_dir, int max_seq=4096)
        : loader_(model_dir), max_seq_(max_seq){
        loader_.load();
        w_ = &loader_.weights();
        const auto& c = w_->cfg;
        kv_.allocate(c.n_layers, c.n_kv_heads, max_seq, c.head_dim);
    }
    ~Session(){ free_scratch(); kv_.free_buffers(); }

    // Greedy generate. Returns full token sequence (input + generated).
    std::vector<int> generate(const std::vector<int>& input_ids, const GenerateCfg& cfg){
        const auto& c = w_->cfg;
        int H=c.hidden;
        std::vector<int> seq = input_ids;
        int Tin = (int)input_ids.size();
        alloc_scratch(std::max(Tin, 1));

        // --- Prefill ---
        hipMemcpy(d_ids, input_ids.data(), Tin*sizeof(int), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(embed_gather_f32_kernel,dim3(Tin),dim3(256),0,0,
                           w_->embed_tokens,d_ids,d_hidden,Tin,H);
        kv_.len = 0;
        run_layers(Tin, /*pos0*/0, /*is_prefill*/true);
        kv_.len = Tin;

        int next = greedy_from_last(Tin);
        seq.push_back(next);

        // --- Decode loop ---
        for (int step=1; step<cfg.max_new_tokens; ++step){
            if (cfg.eos_token_id>=0 && next==cfg.eos_token_id) break;
            int pos = kv_.len;
            if (pos >= max_seq_) break;
            int one = next;
            hipMemcpy(d_ids,&one,sizeof(int),hipMemcpyHostToDevice);
            hipLaunchKernelGGL(embed_gather_f32_kernel,dim3(1),dim3(256),0,0,
                               w_->embed_tokens,d_ids,d_hidden,1,H);
            run_layers(1, pos, /*is_prefill*/false);
            kv_.len += 1;
            next = greedy_from_last(1);
            seq.push_back(next);
        }
        return seq;
    }

    // Diagnostic: run prefill only, return fp32 logits of the last prompt token.
    std::vector<float> prefill_logits(const std::vector<int>& input_ids){
        const auto& c = w_->cfg;
        int H=c.hidden, V=c.vocab_size;
        int Tin=(int)input_ids.size();
        alloc_scratch(std::max(Tin,1));
        hipMemcpy(d_ids, input_ids.data(), Tin*sizeof(int), hipMemcpyHostToDevice);
        hipLaunchKernelGGL(embed_gather_f32_kernel,dim3(Tin),dim3(256),0,0,
                           w_->embed_tokens,d_ids,d_hidden,Tin,H);
        kv_.len=0;
        run_layers(Tin,0,true);
        kv_.len=Tin;
        // lm_head: float hidden[last] → fp16 logits
        const float* last = d_hidden + (size_t)(Tin-1)*H;
        dim3 tb(16,16), gr((V+15)/16,1);
        hipLaunchKernelGGL(gemm_f32in_f16out_kernel,gr,tb,0,0,last,w_->lm_head,d_logits,1,V,H);
        std::vector<__half> h(V); hipMemcpy(h.data(),d_logits,(size_t)V*2,hipMemcpyDeviceToHost);
        std::vector<float> out(V);
        for (int i=0;i<V;++i) out[i]=__half2float(h[i]);
        return out;
    }

private:
    // Compute logits for the LAST of T float hidden rows, return argmax token.
    int greedy_from_last(int T){
        const auto& c = w_->cfg;
        int H=c.hidden, V=c.vocab_size;
        const float* last = d_hidden + (size_t)(T-1)*H;
        dim3 tb(16,16), gr((V+15)/16,1);
        hipLaunchKernelGGL(gemm_f32in_f16out_kernel,gr,tb,0,0,last,w_->lm_head,d_logits,1,V,H);
        hipLaunchKernelGGL(argmax_kernel,dim3(1),dim3(256),0,0,d_logits,V,d_argmax);
        int tok; hipMemcpy(&tok,d_argmax,sizeof(int),hipMemcpyDeviceToHost);
        return tok;
    }
};

} // namespace runtime
} // namespace cdna3
