#!/usr/bin/env python3
"""
export_attn_layer.py — Step 2-B: export one real attention layer from a HF model
for numerical validation of the HIP Flash Attention kernel.

This is the "numerical bridge" between the synthetic-shapes kernel (fa_robust) and
a real LLM: it takes ONE attention layer of a small open-weight model (Llama/Mistral
family, matching the TCO PP-bound workload), runs a forward pass with a known input,
and dumps to binary:
  - the layer input hidden states           (seq x hidden, fp16)
  - the Q/K/V/O projection weights          (fp16)
  - RoPE config (theta, dims) and GQA config (n_heads, n_kv_heads, head_dim)
  - the reference attention output           (seq x hidden, fp32) -- ground truth

The HIP side (validate_attn_layer.hip) reloads these, reconstructs the layer
(proj -> RoPE -> fa_robust kernel -> O proj), and checks max|gpu - ref|.

Why one layer (not the whole model): isolates the attention kernel's numerical
correctness with REAL weights + RoPE + GQA, without the fragility of a full
custom-op build. De-risks Step 2-A (full PyTorch integration).

Usage (on the VM, has torch+transformers):
  python export_attn_layer.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --layer 0 --seq 256 --out ./attn_layer_dump

Outputs a directory of .bin files + a meta.json describing shapes/dtypes.

NOTE: run this on the GPU VM (or any box with torch+transformers). It does NOT
need the GPU for the export itself (CPU forward of one layer is fine), but the
model download needs network. Pre-stage the model if the VM has no HF access.
"""
import argparse, json, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="HF model id (Llama/Mistral family, small)")
    ap.add_argument("--layer", type=int, default=0, help="which decoder layer to export")
    ap.add_argument("--seq", type=int, default=256, help="sequence length of the probe input")
    ap.add_argument("--out", default="./attn_layer_dump", help="output directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        print("ERROR: needs torch + transformers. pip install torch transformers", file=sys.stderr)
        sys.exit(1)

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.model} (fp16, CPU) ...")
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.eval()

    # Locate the decoder layer's self-attention. Llama/Mistral share this structure.
    layer = model.model.layers[args.layer].self_attn
    H   = cfg.hidden_size
    nH  = cfg.num_attention_heads
    nKV = getattr(cfg, "num_key_value_heads", nH)   # GQA: may be < nH
    D   = H // nH
    theta = getattr(cfg, "rope_theta", 10000.0)
    print(f"  hidden={H} n_heads={nH} n_kv_heads={nKV} head_dim={D} rope_theta={theta}")
    print(f"  GQA group = {nH // nKV} (q heads per kv head)")

    # Probe input: random hidden states (deterministic). Shape [1, seq, H].
    x = torch.randn(1, args.seq, H, dtype=torch.float16)

    # Reference output: run the real attention layer forward (causal).
    # We build position ids and a causal mask the way the model expects.
    with torch.no_grad():
        pos = torch.arange(args.seq).unsqueeze(0)
        # transformers attention modules vary in signature across versions; we call
        # the projections + SDPA manually to stay version-robust and to mirror what
        # the HIP side reconstructs.
        q = layer.q_proj(x)                       # [1, seq, nH*D]
        k = layer.k_proj(x)                       # [1, seq, nKV*D]
        v = layer.v_proj(x)                       # [1, seq, nKV*D]
        q = q.view(1, args.seq, nH,  D).transpose(1, 2)   # [1, nH, seq, D]
        k = k.view(1, args.seq, nKV, D).transpose(1, 2)
        v = v.view(1, args.seq, nKV, D).transpose(1, 2)

        # RoPE (Llama style). Build cos/sin and apply.
        inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        ang = pos.float().unsqueeze(-1) * inv_freq.unsqueeze(0)   # [1, seq, D/2]
        cos = torch.cos(ang).repeat(1, 1, 2)   # [1, seq, D]
        sin = torch.sin(ang).repeat(1, 1, 2)
        def rope(t):  # t: [1, heads, seq, D]
            d = t.shape[-1]
            t1, t2 = t[..., :d//2], t[..., d//2:]
            rot = torch.cat((-t2, t1), dim=-1)
            c = cos.unsqueeze(1).to(t.dtype); s = sin.unsqueeze(1).to(t.dtype)
            return t * c + rot * s
        q = rope(q); k = rope(k)

        # GQA: expand kv heads to match q heads for the reference SDPA.
        rep = nH // nKV
        k_e = k.repeat_interleave(rep, dim=1)
        v_e = v.repeat_interleave(rep, dim=1)

        out = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k_e.float(), v_e.float(), is_causal=True)   # [1, nH, seq, D]
        out = out.transpose(1, 2).reshape(1, args.seq, nH*D)       # [1, seq, H]
        o_proj = layer.o_proj(out.to(torch.float16)).float()       # [1, seq, H] fp32 ref

    # Dump tensors. fp16 weights/inputs (what the kernel uses), fp32 reference.
    def save(name, t, dtype):
        a = t.detach().cpu().numpy().astype(dtype)
        a.tofile(os.path.join(args.out, name + ".bin"))
        return list(a.shape), str(a.dtype)

    meta = {"model": args.model, "layer": args.layer, "seq": args.seq,
            "hidden": H, "n_heads": nH, "n_kv_heads": nKV, "head_dim": D,
            "rope_theta": theta, "causal": True, "tensors": {}}

    # input + projection weights (Linear weight is [out, in]); store as-is, HIP applies.
    meta["tensors"]["x"]      = save("x",      x.view(args.seq, H),                 np.float16)
    meta["tensors"]["q_w"]    = save("q_w",    layer.q_proj.weight,                 np.float16)
    meta["tensors"]["k_w"]    = save("k_w",    layer.k_proj.weight,                 np.float16)
    meta["tensors"]["v_w"]    = save("v_w",    layer.v_proj.weight,                 np.float16)
    meta["tensors"]["o_w"]    = save("o_w",    layer.o_proj.weight,                 np.float16)
    meta["tensors"]["ref"]    = save("ref",    o_proj.view(args.seq, H),            np.float32)
    # precomputed cos/sin so the HIP side doesn't reimplement RoPE math (just reads)
    meta["tensors"]["cos"]    = save("cos",    cos.view(args.seq, D),               np.float32)
    meta["tensors"]["sin"]    = save("sin",    sin.view(args.seq, D),               np.float32)

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nExported to {args.out}/")
    print(json.dumps(meta, indent=2))
    print("\nNext: validate_attn_layer.hip reads this and checks max|gpu - ref|.")


if __name__ == "__main__":
    main()
