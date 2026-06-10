#!/usr/bin/env python3
"""
validate_full_model.py — Step 2-A: end-to-end logit validation.

Monkey-patches a small HF decoder model's attention to call our HIP kernel
(torch.ops.amdinstinct.flash_attn) during prefill, runs a prompt, and compares the
full-model output logits against the unpatched fp16 HF forward on the SAME input.

This is the literal answer to the original critique ("does the microbenchmark serve a
real model?"): if patched logits ≈ reference logits, our kernel produces *the same
model*. Step 2-B already proved one layer is numerically correct; 2-A proves the whole
stack runs on our kernel and the argmax/top-k predictions are unchanged.

Run (on the VM, after building the op):
    python validate_full_model.py --model Qwen/Qwen2.5-0.5B --seq 256 \
        --prompt "The capital of France is"

Scope (honest): prefill only (no KV-cache decode path); fp16 kernel vs fp16 HF ref.
Metric is logit agreement on the LAST position (next-token distribution) + top-1/top-5
match, which is what actually matters for "same model". Full-sequence max logit error
is also reported.
"""
from __future__ import annotations

import argparse
import sys

import torch


def load_op():
    """Load the compiled extension; it self-registers torch.ops.amdinstinct.flash_attn."""
    try:
        import amdinstinct_fa  # noqa: F401  (import triggers TORCH_LIBRARY registration)
    except Exception:
        # fall back to explicit .so load if not installed on the path
        import glob
        sos = glob.glob("amdinstinct_fa*.so") + glob.glob("build/**/amdinstinct_fa*.so",
                                                           recursive=True)
        if not sos:
            print("ERROR: build the op first (python setup.py build_ext --inplace).",
                  file=sys.stderr)
            sys.exit(1)
        torch.ops.load_library(sos[0])
    assert hasattr(torch.ops.amdinstinct, "flash_attn"), "op not registered"


def make_patched_attention(orig_module, cfg):
    """Return a forward() that does Q/K/V proj + RoPE in torch, attention in our kernel.

    Mirrors the layout contract of fa_attn_kernel.hpp: head-major [heads, seq, D],
    fp16, causal+GQA intrinsic. RoPE is applied in torch exactly as Step 2-B's export
    did (rotate-half), so the numerics match the validated path.
    """
    H   = cfg.hidden_size
    nH  = cfg.num_attention_heads
    nKV = getattr(cfg, "num_key_value_heads", nH)
    D   = H // nH
    theta = getattr(cfg, "rope_theta", 10000.0)

    def rope(t, cos, sin):              # t: [heads, seq, D]
        d = t.shape[-1]
        t1, t2 = t[..., : d // 2], t[..., d // 2 :]
        rot = torch.cat((-t2, t1), dim=-1)
        return t * cos + rot * sin

    def forward(hidden_states, *args, **kwargs):
        # hidden_states: [1, seq, H]  (batch 1, prefill)
        x = hidden_states
        bsz, seq, _ = x.shape
        assert bsz == 1, "2-A validation is batch-1 prefill"
        q = orig_module.q_proj(x).view(seq, nH,  D).transpose(0, 1).contiguous()   # [nH, seq, D]
        k = orig_module.k_proj(x).view(seq, nKV, D).transpose(0, 1).contiguous()
        v = orig_module.v_proj(x).view(seq, nKV, D).transpose(0, 1).contiguous()

        # RoPE (rotate-half), built like export_attn_layer.py.
        pos = torch.arange(seq, device=x.device)
        inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=x.device).float() / D))
        ang = pos.float().unsqueeze(-1) * inv.unsqueeze(0)        # [seq, D/2]
        cos = torch.cos(ang).repeat(1, 2).to(x.dtype)            # [seq, D]
        sin = torch.sin(ang).repeat(1, 2).to(x.dtype)
        q = rope(q, cos, sin)
        k = rope(k, cos, sin)

        o = torch.ops.amdinstinct.flash_attn(q.half(), k.half(), v.half())  # [nH, seq, D]
        o = o.transpose(0, 1).reshape(1, seq, nH * D)            # [1, seq, H]
        out = orig_module.o_proj(o.to(x.dtype))
        # RETURN SHAPE = the #1 fragility of 2-A. HF attention modules' return arity
        # changed across versions: old = (attn_output, attn_weights, past_kv); current
        # (transformers >=4.4x) = (attn_output, attn_weights). The decoder layer unpacks
        # exactly as many as its version expects. We return 2-tuple (current default).
        # If the VM run raises "too many/few values to unpack", flip RETURN_ARITY below
        # and re-run with --layers 1 first to confirm before patching all layers.
        if RETURN_ARITY == 2:
            return (out, None)
        if RETURN_ARITY == 3:
            return (out, None, None)
        return out  # arity 1: some custom modules return the tensor directly

    return forward


# Set per the transformers version on the VM. Default 2 matches transformers >=4.4x.
# Bisect with --layers 1 if unpacking fails; this is the documented fragile point.
RETURN_ARITY = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--layers", type=int, default=-1,
                    help="patch only the first N layers (-1 = all); useful to bisect bugs")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_op()
    dev = "cuda"  # HIP exposes the CUDA device API
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(dev)
    model.eval()
    cfg = model.config

    ids = tok(args.prompt, return_tensors="pt").input_ids[:, : args.seq].to(dev)

    # Reference forward (unpatched).
    with torch.no_grad():
        ref_logits = model(ids).logits.float()   # [1, seq, vocab]

    # Patch attention forwards.
    import types
    layers = model.model.layers
    n_patch = len(layers) if args.layers < 0 else min(args.layers, len(layers))
    for i in range(n_patch):
        attn = layers[i].self_attn
        attn.forward = types.MethodType(
            lambda self, hs, *a, **k: make_patched_attention(self, cfg)(hs, *a, **k), attn)

    with torch.no_grad():
        patched_logits = model(ids).logits.float()

    # Compare. The meaningful metric is the last-position next-token distribution.
    last_ref = ref_logits[0, -1]
    last_pat = patched_logits[0, -1]
    max_abs = (patched_logits - ref_logits).abs().max().item()
    scale = ref_logits.abs().mean().item()
    top1_ref = last_ref.argmax().item()
    top1_pat = last_pat.argmax().item()
    top5_ref = set(last_ref.topk(5).indices.tolist())
    top5_pat = set(last_pat.topk(5).indices.tolist())
    cos = torch.nn.functional.cosine_similarity(last_ref, last_pat, dim=0).item()

    print(f"=== Step 2-A: full-model logit validation ({args.model}) ===")
    print(f"  patched {n_patch}/{len(layers)} attention layers with amdinstinct.flash_attn")
    print(f"  seq={ids.shape[1]} D={cfg.hidden_size // cfg.num_attention_heads} "
          f"nH={cfg.num_attention_heads} nKV={getattr(cfg,'num_key_value_heads',cfg.num_attention_heads)}")
    print(f"  full-seq max|logit_patched - logit_ref| = {max_abs:.4f}  "
          f"(ref logit scale mean|·|={scale:.3f}, rel={max_abs/max(scale,1e-6):.3f})")
    print(f"  last-token cosine(ref, patched) = {cos:.5f}")
    print(f"  top-1 next token: ref={top1_ref} patched={top1_pat}  "
          f"{'MATCH' if top1_ref==top1_pat else 'MISMATCH'}")
    print(f"  top-5 overlap: {len(top5_ref & top5_pat)}/5")
    ok = (top1_ref == top1_pat) and (cos > 0.99)
    print(f"  VERDICT: {'PASS — same model under our kernel' if ok else 'CHECK — divergence, bisect with --layers'}")
    print("  (prefill only, fp16 kernel vs fp16 HF ref; KV-cache decode is future work)")


if __name__ == "__main__":
    main()
