// ---------------------------------------------------------------------------
// diag_logits.hip — Stage-C C3 numerical diagnostic.
//
// Compares the runtime's first-step logits (after prefill) against the HF
// reference logits (ref_logits0.bin). Tells us whether a greedy mismatch is
// small numerical drift (top-1 preserved, close values) or a structural bug.
//
//   ./diag_logits <model_dir> <prompt_ids> <ref_logits0.bin>
// ---------------------------------------------------------------------------
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>

#include "../include/cdna3/runtime/session.hpp"

static std::vector<int> load_ids(const std::string& p){
    std::vector<int> v; std::ifstream f(p); int x; while (f>>x) v.push_back(x); return v;
}

int main(int argc, char** argv){
    if (argc < 4){ printf("usage: %s <model_dir> <prompt_ids> <ref_logits0.bin>\n",argv[0]); return 1; }
    std::string dir=argv[1];
    auto prompt=load_ids(argv[2]);

    cdna3::runtime::Session sess(dir, 2048);
    auto mine = sess.prefill_logits(prompt);
    int V = (int)mine.size();

    // load reference fp32 logits
    std::vector<float> ref(V);
    { std::ifstream f(argv[3], std::ios::binary);
      f.read(reinterpret_cast<char*>(ref.data()), (size_t)V*4); }

    // top-10 of each
    auto top10=[&](const std::vector<float>& L, const char* tag){
        std::vector<int> idx(L.size()); for (size_t i=0;i<L.size();++i) idx[i]=(int)i;
        std::partial_sort(idx.begin(), idx.begin()+10, idx.end(),
                          [&](int a,int b){ return L[a]>L[b]; });
        printf("%s top-10:\n", tag);
        for (int i=0;i<10;++i) printf("   id=%6d  logit=%8.3f\n", idx[i], L[idx[i]]);
        return idx[0];
    };
    int my_top = top10(mine, "RUNTIME");
    int rf_top = top10(ref,  "HF REF ");

    // correlation / max abs diff over vocab
    double sd=0, sm=0, sr=0, mx=0;
    for (int i=0;i<V;++i){ double d=mine[i]-ref[i]; sd+=d*d; sm+=mine[i]; sr+=ref[i];
                           mx=std::max(mx,(double)std::fabs(d)); }
    printf("\nargmax: runtime=%d  ref=%d  %s\n", my_top, rf_top,
           my_top==rf_top?"MATCH":"DIFFER");
    printf("rms diff over vocab = %.4f   max abs diff = %.4f\n", std::sqrt(sd/V), mx);
    printf("(small rms + top-1 match => numerical drift, expected in fp16 greedy)\n");
    return 0;
}
