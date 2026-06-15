// ---------------------------------------------------------------------------
// cdna3/runtime/model_loader.hpp — weight loader for the Stage-C LLM runtime.
//
// Loads a model exported from HuggingFace (fp16 binary dumps + meta.json)
// into device memory. Supports Llama-3/Qwen2 GQA architectures.
//
// ## Export flow (Python side, on VM):
//
//   python runtime/export_model.py \
//       --model Qwen/Qwen2.5-1.5B \
//       --out /tmp/qwen25_1b
//
//   Produces: meta.json  +  layer_N_{q,k,v,o,gate,up,down}_proj.bin
//             embed_tokens.bin  lm_head.bin  norm.bin
//             layer_N_{input,post}_norm.bin
//
// ## C++ side:
//
//   ModelLoader loader("/tmp/qwen25_1b");
//   loader.load();  // mallocs + H2D copies
//   // Access: loader.layers[i].Wq  (device ptr, fp16)
//
// ## Tensor layout (all fp16, device):
//
//   Wq/Wk/Wv/Wo : [hidden × hidden] or [hidden × (n_kv_heads × head_dim)]
//   Wgate/Wup   : [ffn_dim × hidden]
//   Wdown        : [hidden × ffn_dim]
//   embed_tokens : [vocab_size × hidden]
//   lm_head      : [vocab_size × hidden]
//   norm.weight  : [hidden]
// ---------------------------------------------------------------------------
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

// Minimal JSON parser for meta.json (no external dependency).
// Only handles flat key:"value" and key:number pairs.
#include <sstream>
#include <map>

namespace cdna3 {
namespace runtime {

struct ModelConfig {
    int n_layers;
    int hidden;       // hidden_size
    int n_heads;      // num_attention_heads
    int n_kv_heads;   // num_key_value_heads
    int head_dim;     // hidden / n_heads
    int ffn_dim;      // intermediate_size
    int vocab_size;
    float rope_theta;
};

struct LayerWeights {
    __half* Wq   = nullptr;   // [hidden × (n_heads × head_dim)]
    __half* Wk   = nullptr;   // [hidden × (n_kv_heads × head_dim)]
    __half* Wv   = nullptr;
    __half* Wo   = nullptr;   // [hidden × hidden]
    __half* Wgate= nullptr;   // [ffn_dim × hidden]
    __half* Wup  = nullptr;
    __half* Wdown= nullptr;   // [hidden × ffn_dim]
    __half* norm_attn = nullptr;   // [hidden]
    __half* norm_mlp  = nullptr;   // [hidden]
};

struct ModelWeights {
    ModelConfig cfg;
    std::vector<LayerWeights> layers;
    __half* embed_tokens = nullptr;   // [vocab_size × hidden]
    __half* lm_head      = nullptr;
    __half* final_norm   = nullptr;   // [hidden]
};

// ---------------------------------------------------------------------------
// Load a flat binary tensor file → device memory.
// Returns device pointer (caller owns).
// ---------------------------------------------------------------------------
static __half* load_tensor(const std::string& path, size_t n_elems) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    std::vector<uint16_t> buf(n_elems);
    f.read(reinterpret_cast<char*>(buf.data()), n_elems * 2);
    if (!f) throw std::runtime_error("short read: " + path);
    __half* dev;
    hipMalloc(&dev, n_elems * sizeof(__half));
    hipMemcpy(dev, buf.data(), n_elems * sizeof(__half), hipMemcpyHostToDevice);
    return dev;
}

// ---------------------------------------------------------------------------
// Parse meta.json (minimal — only int/float fields needed).
// ---------------------------------------------------------------------------
static std::map<std::string,std::string> parse_meta(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open meta.json: " + path);
    std::string src((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    std::map<std::string,std::string> kv;
    size_t pos = 0;
    while (pos < src.size()) {
        size_t qs = src.find('"', pos); if (qs == std::string::npos) break;
        size_t qe = src.find('"', qs+1); if (qe == std::string::npos) break;
        std::string key = src.substr(qs+1, qe-qs-1);
        size_t colon = src.find(':', qe); if (colon == std::string::npos) break;
        size_t vs = colon+1;
        while (vs<src.size() && (src[vs]==' '||src[vs]=='\t'||src[vs]=='\n')) ++vs;
        std::string val;
        if (vs<src.size() && src[vs]=='"'){
            size_t ve=src.find('"',vs+1); val=src.substr(vs+1,ve-vs-1); pos=ve+1;
        } else {
            size_t ve=vs;
            while(ve<src.size()&&src[ve]!=','&&src[ve]!='}'&&src[ve]!='\n') ++ve;
            val=src.substr(vs,ve-vs); pos=ve;
        }
        kv[key]=val;
    }
    return kv;
}

// ---------------------------------------------------------------------------
// ModelLoader
// ---------------------------------------------------------------------------
class ModelLoader {
    std::string dir_;
    ModelWeights w_;

public:
    explicit ModelLoader(const std::string& model_dir) : dir_(model_dir) {}

    const ModelWeights& weights() const { return w_; }

    void load() {
        // Parse config
        auto meta = parse_meta(dir_ + "/meta.json");
        w_.cfg.n_layers   = std::stoi(meta["n_layers"]);
        w_.cfg.hidden     = std::stoi(meta["hidden"]);
        w_.cfg.n_heads    = std::stoi(meta["n_heads"]);
        w_.cfg.n_kv_heads = std::stoi(meta["n_kv_heads"]);
        w_.cfg.head_dim   = std::stoi(meta["head_dim"]);
        w_.cfg.ffn_dim    = std::stoi(meta["ffn_dim"]);
        w_.cfg.vocab_size = std::stoi(meta["vocab_size"]);
        w_.cfg.rope_theta = std::stof(meta.count("rope_theta") ? meta["rope_theta"] : "10000.0");

        const auto& c = w_.cfg;
        int H = c.hidden, F = c.ffn_dim, V = c.vocab_size;
        int qd = c.n_heads   * c.head_dim;
        int kvd= c.n_kv_heads* c.head_dim;

        auto p = [&](const std::string& name) { return dir_ + "/" + name + ".bin"; };

        // Embedding and final norm
        w_.embed_tokens = load_tensor(p("embed_tokens"), (size_t)V*H);
        w_.lm_head      = load_tensor(p("lm_head"),      (size_t)V*H);
        w_.final_norm   = load_tensor(p("final_norm"),   H);

        // Per-layer weights
        w_.layers.resize(c.n_layers);
        for (int i=0; i<c.n_layers; ++i) {
            auto lp = [&](const char* t) {
                return dir_ + "/layer_" + std::to_string(i) + "_" + t + ".bin";
            };
            auto& lw = w_.layers[i];
            lw.Wq       = load_tensor(lp("q_proj"),    (size_t)qd *H);
            lw.Wk       = load_tensor(lp("k_proj"),    (size_t)kvd*H);
            lw.Wv       = load_tensor(lp("v_proj"),    (size_t)kvd*H);
            lw.Wo       = load_tensor(lp("o_proj"),    (size_t)H  *H);
            lw.Wgate    = load_tensor(lp("gate_proj"), (size_t)F  *H);
            lw.Wup      = load_tensor(lp("up_proj"),   (size_t)F  *H);
            lw.Wdown    = load_tensor(lp("down_proj"), (size_t)H  *F);
            lw.norm_attn= load_tensor(lp("norm_attn"), H);
            lw.norm_mlp = load_tensor(lp("norm_mlp"),  H);
        }

        printf("ModelLoader: loaded %d layers, hidden=%d, heads=%d/%d (GQA), ffn=%d\n",
               c.n_layers, H, c.n_heads, c.n_kv_heads, F);
    }

    ~ModelLoader() {
        for (auto& lw : w_.layers) {
            hipFree(lw.Wq); hipFree(lw.Wk); hipFree(lw.Wv); hipFree(lw.Wo);
            hipFree(lw.Wgate); hipFree(lw.Wup); hipFree(lw.Wdown);
            hipFree(lw.norm_attn); hipFree(lw.norm_mlp);
        }
        hipFree(w_.embed_tokens); hipFree(w_.lm_head); hipFree(w_.final_norm);
    }
};

} // namespace runtime
} // namespace cdna3
