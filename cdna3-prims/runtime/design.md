# cdna3-prims Runtime — Stage-C Design Note

**Status:** Stage C, step C1 (cold design). Gated on Stage B validated (DONE 2026-06-15).  
**Scope:** A minimal, honest research LLM inference runtime for 1×MI300X, built on the
validated `cdna3::attn` attention module. Not a vLLM replacement — a research vehicle.

---

## Goals and non-goals

**Goals:**
- Greedy decode, batch-size 1, fp16 weights
- Prefill → decode generation loop validated vs HuggingFace greedy output
- Supports small open-weight GQA models (Qwen2.5-1.5B class, Llama-3.2-1B class)
- Honest benchmarks: tokens/sec, time-to-first-token (TTFT), vs a production baseline
- Clean API: `runtime::Session` that wraps attention + MLP + layer loop

**Non-goals (explicitly future work):**
- Batched decode (paged KV-cache, continuous batching)
- Quantization (int8/fp8 weights)
- Multi-GPU (gated on hardware availability, not planned)
- Flash attention for MLP (MLP is memory-bound, not the bottleneck here)

---

## Architecture

```
cdna3-prims/runtime/
  design.md               ← this file (C1)
  model_loader.hpp        ← weight loading from HF safetensors export (C2)
  kv_cache.hpp            ← contiguous fp16 KV-cache manager (C2)
  layer.hpp               ← single transformer layer: attn + MLP (C3)
  session.hpp             ← Session: load model, generate() loop (C3)
  benchmarks/
    bench_generate.cpp    ← tokens/sec, TTFT, vs production baseline (C4)
```

---

## Layer loop (the spine)

Each transformer layer executes:

```
1. LayerNorm(x)                      → x_norm        [validated: DPP two-pass, T-018/T-019]
2. cdna3::attn::prefill(x_norm, ...)  → attn_out     [validated: Stage B, 2026-06-15]
   OR cdna3::attn::decode(x_norm, ...)
3. x = x + attn_out  (residual)
4. LayerNorm(x)                       → x_norm2
5. MLP: gate_proj * silu(up_proj) → down_proj         [naive fp16 GEMM, not the bottleneck]
6. x = x + mlp_out  (residual)
```

MLP uses rocBLAS for GEMM — not a research target; the attention path is.

---

## KV-cache manager

Contiguous fp16, pre-allocated at session start:

```cpp
struct KVCache {
    __half* K;  // [n_layers × n_kv_heads × max_seq_len × head_dim]
    __half* V;
    int     len;   // current filled length
};
```

No paging. `max_seq_len` set at session init (default: 4096). Each decode step appends
one token's K/V and calls `cdna3::attn::decode` with N = cache.len.

Rationale: paged KV-cache is the right production choice for batched/long contexts, but
for greedy batch-1 the contiguous layout is simpler and lets the research question
(does our attention kernel work end-to-end?) be answered without a separate allocator.

---

## Model loader (C2)

Reuses the HF export path from the 2-A integration session:
- HuggingFace weights exported to safetensors (fp16, no quantization)
- Loader maps tensor names to layer structs
- Supported architectures: LlamaForCausalLM, Qwen2ForCausalLM (same attention pattern)

The model loader is a thin wrapper — it does not implement training or fine-tuning.

---

## Session API (C3)

```cpp
namespace cdna3::runtime {

struct GenerateCfg {
    int max_new_tokens;
    int temperature_numerator;  // 0 = greedy (temp = 0)
};

class Session {
public:
    Session(const std::string& model_dir, int max_seq_len = 4096);
    std::vector<int> generate(const std::vector<int>& input_ids,
                              const GenerateCfg& cfg = {});
};

} // namespace cdna3::runtime
```

`generate()` runs:
1. Prefill all input tokens → logits for last token → argmax (greedy)
2. Decode loop: feed predicted token → update KV-cache → logits → argmax
3. Stop on EOS or max_new_tokens

---

## Benchmark harness (C4)

`bench_generate.cpp` measures:
- **TTFT** (time-to-first-token): prefill latency for prompt lengths 128/512/1024
- **TPS** (tokens/second): decode throughput for 256 generated tokens
- **Baseline**: same prompt through `transformers` pipeline on same hardware (CPU or vLLM)

Output format: markdown table, framed as research-vs-prod comparison.

---

## Exit criterion for Stage C

A documented session that:
1. Loads a real GQA model (Qwen2.5-1.5B or Llama-3.2-1B)
2. Generates a 256-token sequence from a fixed prompt
3. Top-1 token sequence matches HuggingFace greedy output
4. Benchmark output with TTFT and TPS

This closes the cycle: **cdna3-prims** delivers a usable, validated, research-grade
inference runtime built entirely on the three hardware primitives discovered in this project.

---

## Implementation order (C2 → C3 → C4)

| Step | Mode | Deliverable | Gate |
|------|------|-------------|------|
| C2   | cold | `model_loader.hpp`, `kv_cache.hpp` | design review |
| C3   | VM   | `layer.hpp`, `session.hpp`, correctness vs HF | Stage B ✓ |
| C4   | cold | `bench_generate.cpp`, benchmark harness | C3 VM pass |
