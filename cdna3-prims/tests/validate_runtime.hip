// ---------------------------------------------------------------------------
// validate_runtime.hip — Stage-C C3 end-to-end runtime validation.
//
// Loads a real GQA model (exported via runtime/export_model.py), runs greedy
// generation through cdna3::runtime::Session, and compares the generated token
// sequence against a reference produced by HuggingFace greedy decoding
// (exported to ref_tokens.txt).
//
// Compile (from cdna3-prims/tests/):
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//     -I../include validate_runtime.hip -o validate_runtime
//
// Run:
//   ./validate_runtime <model_dir> <prompt_ids_file> <ref_tokens_file> <max_new>
//
//   model_dir       : directory from export_model.py (meta.json + *.bin)
//   prompt_ids_file : text file, one int per line (input token ids)
//   ref_tokens_file : text file, one int per line (HF greedy output, full seq)
//   max_new         : number of tokens to generate
//
// Exit criterion (Stage C): generated == ref (top-1 match) → cycle closed.
// ---------------------------------------------------------------------------
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>

#include "../include/cdna3/runtime/session.hpp"

static std::vector<int> load_ids(const std::string& path){
    std::vector<int> v; std::ifstream f(path);
    int x; while (f >> x) v.push_back(x);
    return v;
}

int main(int argc, char** argv){
    if (argc < 5){
        printf("usage: %s <model_dir> <prompt_ids> <ref_tokens> <max_new>\n", argv[0]);
        return 1;
    }
    std::string model_dir = argv[1];
    std::vector<int> prompt = load_ids(argv[2]);
    std::vector<int> ref    = load_ids(argv[3]);
    int max_new = atoi(argv[4]);

    hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
    printf("Device: %s (gfx%d)\n", prop.name, prop.major*100+prop.minor);
    printf("Stage-C runtime validation — cdna3::runtime::Session\n");
    printf("====================================================\n");
    printf("Model: %s   prompt_len=%zu   max_new=%d\n\n",
           model_dir.c_str(), prompt.size(), max_new);

    cdna3::runtime::Session sess(model_dir, /*max_seq*/2048);
    cdna3::runtime::GenerateCfg cfg;
    cfg.max_new_tokens = max_new;

    auto out = sess.generate(prompt, cfg);

    // Compare generated tail vs reference.
    size_t gen_start = prompt.size();
    int matched = 0, compared = 0;
    bool ok = true;
    printf("pos  generated  reference  match\n");
    for (size_t i = gen_start; i < out.size() && i < ref.size(); ++i){
        bool m = (out[i] == ref[i]);
        if (m) ++matched; ++compared;
        if (!m) ok = false;
        if (i - gen_start < 20)   // print first 20 generated
            printf("%3zu  %9d  %9d  %s\n", i-gen_start, out[i], ref[i], m?"ok":"MISMATCH");
    }
    printf("\n----------------------------------------------------\n");
    printf("Matched %d/%d generated tokens vs HF greedy\n", matched, compared);
    if (ok && compared > 0)
        printf("Stage-C exit criterion MET — runtime reproduces HF greedy top-1.\n");
    else
        printf("MISMATCH — runtime diverges from HF greedy. Investigate layer numerics.\n");
    return (ok && compared > 0) ? 0 : 1;
}
