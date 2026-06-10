# Design Note — Decode / KV-cache Flash Attention

**Cold design.** 2-A validated the kernel in **prefill** (seqQ = seqK = N, the full
N×N attention). Decode is the other half of LLM inference and the one production serves
most: generate one token at a time, attending the new query against a growing KV-cache.
This note designs the decode kernel + its torch-op path before writing HIP, so the VM
session is implement-and-validate. Single-GPU — validates on 1×MI300X.

---

## 1. Why decode is NOT a small variant of prefill

| | Prefill (validated, 2-A) | Decode (this note) |
|---|---|---|
| Query rows | N (the whole prompt) | **1** (the new token) |
| Keys/values | N | **N (cache), grows by 1 each step** |
| Attention shape | N×N | **1×N** |
| Compute regime | compute-bound (MFMA-friendly) | **memory-bound** (stream the KV-cache once) |
| Reuse of Q | Br=16 rows reuse each K tile | a single query vector reused over all N keys |
| Right parallelization | wide query block (Axis-A) | **split-KV across waves (Axis-B)** |
| MFMA | 16×16×16 tiles fit naturally | **a 1×N·D matvec — MFMA tiles are mostly empty** |

The consequence that drives the design: with **one** query row, the MFMA 16×16×16 path
(built for 16 query rows) wastes 15/16 of every tile. Decode is a **matrix-vector**
problem (q·Kᵀ is 1×N, P·V is 1×N · N×D), memory-bound on streaming the KV-cache. The
1-wave/block prefill kernel would idle almost everything. The right structure is the
mirror image of the multi-wave prefill design: **split the N keys across waves**, each
wave reduces its key-range partial (m, l, O_partial), then combine across waves.

## 2. Design: split-KV decode kernel

```
One (batch, qhead) → one block of W waves.  grid = batch · nQHeads.
The single query vector q[D] is loaded once (broadcast to all waves).
The N cached keys are split into W contiguous ranges; wave w handles range w.

Per wave w:
  for each key k in its range:
     s_k = (q · K[k]) * scale          # dot product, D-long, lane-parallel over D
     online-softmax update of (m_w, l_w, O_w[D]) with s_k and V[k]
  → wave w holds a partial (m_w, l_w, O_w) over its key range

Across waves (one LDS combine, the FlashAttention split-K merge):
  m = max_w m_w
  l = Σ_w exp(m_w − m) · l_w
  O = Σ_w exp(m_w − m) · O_w           # rescale each wave's partial to the global max
  O /= l                                # final normalize
  write O[D]
```

This is exactly Axis-B from `DESIGN_multiwave_fa.md` (deferred there as "decode"). The
split-K merge is the standard FlashAttention combine; the only subtlety is doing it
correctly across waves via LDS (each wave publishes its (m_w, l_w, O_w[D]); one wave
reduces).

**Why split-KV and not 1-wave-streams-all:** with N in the thousands and one query, a
single wave streaming all N keys is latency-bound on the sequential key loop. Splitting
across W waves cuts that loop W× and the combine is cheap (W partials). This is what
makes decode fast; it is memory-bound, so the win is keeping all waves streaming the
cache in parallel.

## 3. The dot-product, not MFMA

For decode, q·K[k] is a single D-length dot product per key. Map D across the lanes
(D=128 → 2 elements/lane over 64 lanes; D=64 → 1/lane) and use the **verified DPP
wave-reduce** to sum the partial products — the same primitive the project is built on,
now in its natural home (a true reduction, not a matmul). No MFMA: the 16×16×16 tiles
are empty with one query row, so the dot-product path is both simpler and faster here.
This also means decode is a clean second showcase of the DPP reduction in a real kernel.

## 4. KV-cache layout + the torch-op path

Decode needs the cache to persist across steps. Two pieces:

- **Kernel signature:** `decode_attn(q[1,nQH,D], K_cache[nKVH,N,D], V_cache[nKVH,N,D]) → o[1,nQH,D]`,
  with N = current cache length (runtime). The cache is owned by the caller (PyTorch),
  appended each step.
- **Torch op:** `torch.ops.amdinstinct.flash_attn_decode(q, k_cache, v_cache)`. In the
  validation harness we run real Qwen2.5 decode: prefill with the existing path (or HF),
  then generate T tokens, at each step appending the new K/V to the cache and calling our
  decode op for the attention, comparing the generated token sequence against HF's
  `model.generate` greedy output.

**PASS criterion (mirrors 2-A):** the generated token *sequence* matches HF greedy for T
tokens (the decode analog of "same model"). Per-step logit cosine reported as diagnostic.

## 5. What carries over (de-risks the rewrite)

- The DPP wave-reduce (now for the q·K dot product and the softmax) — verified.
- The online-softmax math + the split-K combine (combine is new but standard).
- GQA kvhead mapping, causal (trivial in decode: the new token attends ALL cached keys).
- The torch-op build + monkey-patch machinery from 2-A (the op signature changes; the
  scaffolding — setup.py, registration, patch harness — is reused).

## 6. Pre-registered hypotheses

- **H-DEC1 (correctness):** real Qwen2.5 greedy decode for T=32 tokens on our decode op
  produces the SAME token sequence as HF `model.generate`. Accept if sequences identical
  (allowing the known fp16 last-token cosine decay not to flip argmax).
- **H-DEC2 (split-KV scaling):** decode latency drops with W (more waves streaming the
  cache) up to the memory-bandwidth limit. Accept if W=4 beats W=1 at N≥1024.
- **H-DEC3 (regime):** decode is memory-bound — latency scales ~linearly with N (the
  cache length), not N² (it's 1×N, not N×N). Accept if the N-sweep exponent ≈ 1.0.

## 7. Scope

- ✅ Single-token greedy decode with a growing KV-cache, split-KV across waves, DPP
  dot-product + softmax, GQA + (trivial) causal, validated vs HF greedy on Qwen2.5.
- ❌ NOT batched decode (multiple sequences) — batch=1 first.
- ❌ NOT paged/quantized KV-cache (vLLM-style) — contiguous fp16 cache first.
- ❌ NOT speculative decode / beam search — greedy only.
- ❌ NOT expected to beat production decode kernels — same honest framing as 2-C.

## 8. Cold deliverables

1. This design note (pre-registration timestamp).
2. `fa_decode.hip` — the split-KV decode kernel (standalone, with CPU ref + N-sweep,
   mirroring fa_robust's harness so correctness + scaling are checked the same way).
3. `llm_integration/torch_op/fa_decode_op.hip` — the decode torch op
   (`flash_attn_decode`), reusing the 2-A build/registration pattern.
4. `llm_integration/torch_op/validate_decode.py` — real Qwen2.5 greedy-decode harness
   vs HF `model.generate` (H-DEC1).
5. VM guion (local, `Proyecto/Decode_guion_vm.md`).
