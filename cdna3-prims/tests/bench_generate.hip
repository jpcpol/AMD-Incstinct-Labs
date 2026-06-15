// ---------------------------------------------------------------------------
// bench_generate.hip — Stage-C C5 throughput benchmark.
//
// Measures two key LLM serving metrics for cdna3::runtime::Session:
//
//   TTFT  (time-to-first-token): prefill latency for several prompt lengths.
//         This exercises the layer loop × T tokens: T QKV GEMMs, T×T attention
//         (1-query-at-a-time decode path), T MLP GEMMs.
//
//   TPS   (tokens/second): decode throughput — one new token per step,
//         kv-cache grows from prompt_len to prompt_len+N_gen.
//
// Both are wall-clock (hipDeviceSynchronize fences). kReps warm-up runs
// precede the timed runs; median is reported.
//
// Compile (from cdna3-prims/tests/):
//   /opt/rocm/bin/hipcc -O3 --offload-arch=gfx942 -std=c++17 \
//     -I../include bench_generate.hip -o bench_generate
//
// Run:
//   ./bench_generate <model_dir>
// ---------------------------------------------------------------------------
#include <cstdio>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>

#include <hip/hip_runtime.h>
#include "../include/cdna3/runtime/session.hpp"

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static double now_ms(){
    hipDeviceSynchronize();
    return std::chrono::duration_cast<Ms>(Clock::now().time_since_epoch()).count();
}

static double median(std::vector<double>& v){
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2-1]+v[n/2])/2.0 : v[n/2];
}

// Build a synthetic prompt of length T: repeating token id 1 (safe for any vocab).
static std::vector<int> make_prompt(int T){
    return std::vector<int>(T, 1);
}

int main(int argc, char** argv){
    if (argc < 2){
        printf("usage: %s <model_dir>\n", argv[0]);
        return 1;
    }
    std::string model_dir = argv[1];

    hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
    printf("Device  : %s\n", prop.name);
    printf("Model   : %s\n\n", model_dir.c_str());

    // -------------------------------------------------------------------------
    // TTFT benchmark: prefill latency for several prompt lengths.
    // Session is re-created per prompt length so KV-cache is fresh.
    // -------------------------------------------------------------------------
    const int TTFT_REPS  = 7;   // runs per config (median of 7)
    const int TTFT_WARM  = 2;   // warm-up runs before timing
    const int N_GEN_ZERO = 1;   // generate exactly 1 token (= TTFT)

    const int prompt_lens[] = {5, 16, 32, 64};
    const int n_prompt_lens = (int)(sizeof(prompt_lens)/sizeof(prompt_lens[0]));

    printf("## TTFT (time-to-first-token)\n\n");
    printf("| prompt_len | TTFT median (ms) | TTFT P95 (ms) |\n");
    printf("|------------|-----------------|---------------|\n");

    for (int pi = 0; pi < n_prompt_lens; ++pi){
        int plen = prompt_lens[pi];
        auto prompt = make_prompt(plen);

        cdna3::runtime::Session sess(model_dir, /*max_seq*/512);
        cdna3::runtime::GenerateCfg cfg;
        cfg.max_new_tokens = N_GEN_ZERO;
        cfg.eos_token_id   = -1;

        // warm-up
        for (int w = 0; w < TTFT_WARM; ++w){
            sess.generate(prompt, cfg);
        }

        std::vector<double> times;
        times.reserve(TTFT_REPS);
        for (int r = 0; r < TTFT_REPS; ++r){
            double t0 = now_ms();
            sess.generate(prompt, cfg);
            double t1 = now_ms();
            times.push_back(t1 - t0);
        }

        double med = median(times);
        std::sort(times.begin(), times.end());
        double p95 = times[(size_t)(0.95 * times.size())];

        printf("| %10d | %15.2f | %13.2f |\n", plen, med, p95);
    }

    // -------------------------------------------------------------------------
    // TPS benchmark: decode throughput — N_GEN tokens after a short prompt.
    // Session is created once; the generate() call produces N_GEN tokens.
    // We measure wall-clock / N_GEN = ms-per-token, report tokens/sec.
    // -------------------------------------------------------------------------
    const int TPS_REPS  = 5;
    const int TPS_WARM  = 1;
    const int PROMPT_LEN_TPS = 5;
    const int N_GEN_TPS      = 32;   // 32 decode steps

    printf("\n## TPS (decode throughput, prompt_len=%d, gen=%d tokens)\n\n",
           PROMPT_LEN_TPS, N_GEN_TPS);
    printf("| run | total_ms | ms/token | tokens/sec |\n");
    printf("|-----|----------|----------|------------|\n");

    auto prompt_tps = make_prompt(PROMPT_LEN_TPS);
    cdna3::runtime::Session sess_tps(model_dir, /*max_seq*/512);
    cdna3::runtime::GenerateCfg cfg_tps;
    cfg_tps.max_new_tokens = N_GEN_TPS;
    cfg_tps.eos_token_id   = -1;

    // warm-up
    for (int w = 0; w < TPS_WARM; ++w){
        sess_tps.generate(prompt_tps, cfg_tps);
    }

    std::vector<double> tps_times;
    tps_times.reserve(TPS_REPS);
    for (int r = 0; r < TPS_REPS; ++r){
        double t0 = now_ms();
        sess_tps.generate(prompt_tps, cfg_tps);
        double t1 = now_ms();
        double total = t1 - t0;
        double ms_tok = total / N_GEN_TPS;
        printf("| %3d | %8.1f | %8.3f | %10.1f |\n",
               r+1, total, ms_tok, 1000.0/ms_tok);
        tps_times.push_back(total);
    }

    double med_total = median(tps_times);
    double med_ms_tok = med_total / N_GEN_TPS;
    printf("\nMedian TPS: %.1f tokens/sec  (%.3f ms/token, total %.1f ms for %d tokens)\n\n",
           1000.0/med_ms_tok, med_ms_tok, med_total, N_GEN_TPS);

    // -------------------------------------------------------------------------
    // TTFT breakdown: prefill vs first decode step (5-token prompt)
    // -------------------------------------------------------------------------
    printf("## TTFT breakdown (prompt_len=5)\n\n");

    const int plen5 = 5;
    auto prompt5 = make_prompt(plen5);
    cdna3::runtime::Session sess5(model_dir, 512);

    // time prefill+decode(1) = full TTFT
    cdna3::runtime::GenerateCfg cfg1; cfg1.max_new_tokens=1; cfg1.eos_token_id=-1;
    for (int w=0;w<2;++w) sess5.generate(prompt5,cfg1);
    std::vector<double> ttft5;
    for (int r=0;r<TTFT_REPS;++r){
        double t0=now_ms(); sess5.generate(prompt5,cfg1); double t1=now_ms();
        ttft5.push_back(t1-t0);
    }

    // time decode(1) only by calling with single-token prompt to a warm cache
    // (proxy: compare TTFT(5) vs TPS-single-step)
    // single decode step = TPS time per token (from tps_times[0] / N_GEN_TPS)
    double single_decode_ms = tps_times[0] / N_GEN_TPS;
    double prefill_only_ms  = median(ttft5) - single_decode_ms;

    printf("TTFT (5-tok prompt)  : %.2f ms  (median)\n", median(ttft5));
    printf("  prefill estimate   : %.2f ms  (TTFT - 1 decode step)\n", prefill_only_ms);
    printf("  decode step        : %.3f ms\n\n", single_decode_ms);

    printf("Notes:\n");
    printf("  - Attention path: cdna3::attn::decode, 1-query-at-a-time (correctness-first)\n");
    printf("  - GEMM path: naive fp16→float accumulation (research reference, not rocBLAS)\n");
    printf("  - Bottleneck: O(T^2) pack_kv_kernel + cdna3::attn::decode per prefill step\n");
    printf("  - Decode is O(N_gen × n_layers × D^2) — linear in generated length\n");

    return 0;
}
