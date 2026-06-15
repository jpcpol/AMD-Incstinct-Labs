#!/usr/bin/env python3
"""
export_model.py — Stage-C model export for cdna3-prims runtime.

Loads a HuggingFace LLaMA/Qwen2 model in fp16, exports all weights as flat
fp16 binary files + meta.json. The C++ ModelLoader reads this directory.

Usage (on GPU VM, needs torch + transformers):
  python export_model.py \\
      --model Qwen/Qwen2.5-1.5B \\
      --out /tmp/qwen25_1b

  python export_model.py \\
      --model meta-llama/Llama-3.2-1B \\
      --out /tmp/llama32_1b

Output layout:
  meta.json
  embed_tokens.bin, lm_head.bin, final_norm.bin
  layer_N_q_proj.bin, layer_N_k_proj.bin, ...  (one file per tensor per layer)
"""
import argparse, json, os, sys
import numpy as np


def save_fp16(arr: "np.ndarray", path: str):
    arr.astype(np.float16).tofile(path)
    print(f"  {os.path.basename(path):50s}  {arr.shape}  {arr.nbytes//1024:>6} KB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model ID")
    ap.add_argument("--out",   required=True, help="output directory")
    ap.add_argument("--cpu",   action="store_true", default=True,
                    help="load on CPU (default; avoids GPU VRAM for export)")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        print("ERROR: pip install torch transformers", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {args.model} ...")
    config = AutoConfig.from_pretrained(args.model)
    model  = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True)
    model.eval()

    # Detect architecture: Llama vs Qwen2 differ only in naming
    sd = model.state_dict()

    # Determine head_dim
    n_heads    = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    hidden     = config.hidden_size
    head_dim   = hidden // n_heads
    ffn_dim    = config.intermediate_size
    n_layers   = config.num_hidden_layers
    vocab_size = config.vocab_size
    rope_theta = getattr(config, "rope_theta", 10000.0)

    meta = dict(
        n_layers=n_layers, hidden=hidden, n_heads=n_heads,
        n_kv_heads=n_kv_heads, head_dim=head_dim,
        ffn_dim=ffn_dim, vocab_size=vocab_size,
        rope_theta=rope_theta, model_id=args.model,
    )
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta.json: {meta}")
    print("Exporting tensors ...")

    def t(key):
        return sd[key].float().numpy()

    # Detect key prefix (model.embed_tokens vs embed_tokens)
    prefix = "model." if "model.embed_tokens.weight" in sd else ""

    save_fp16(t(f"{prefix}embed_tokens.weight"), os.path.join(args.out, "embed_tokens.bin"))
    save_fp16(t(f"{prefix}norm.weight"),          os.path.join(args.out, "final_norm.bin"))

    # lm_head may be tied to embed_tokens
    lm_key = "lm_head.weight"
    lm_arr = t(lm_key) if lm_key in sd else t(f"{prefix}embed_tokens.weight")
    save_fp16(lm_arr, os.path.join(args.out, "lm_head.bin"))

    for i in range(n_layers):
        lp = f"{prefix}layers.{i}"

        def lk(suffix): return f"{lp}.{suffix}.weight"

        save_fp16(t(lk("self_attn.q_proj")),    os.path.join(args.out, f"layer_{i}_q_proj.bin"))
        save_fp16(t(lk("self_attn.k_proj")),    os.path.join(args.out, f"layer_{i}_k_proj.bin"))
        save_fp16(t(lk("self_attn.v_proj")),    os.path.join(args.out, f"layer_{i}_v_proj.bin"))
        save_fp16(t(lk("self_attn.o_proj")),    os.path.join(args.out, f"layer_{i}_o_proj.bin"))
        save_fp16(t(lk("mlp.gate_proj")),        os.path.join(args.out, f"layer_{i}_gate_proj.bin"))
        save_fp16(t(lk("mlp.up_proj")),          os.path.join(args.out, f"layer_{i}_up_proj.bin"))
        save_fp16(t(lk("mlp.down_proj")),        os.path.join(args.out, f"layer_{i}_down_proj.bin"))

        # Layer norm weights (RMSNorm)
        save_fp16(t(f"{lp}.input_layernorm.weight"),      os.path.join(args.out, f"layer_{i}_norm_attn.bin"))
        save_fp16(t(f"{lp}.post_attention_layernorm.weight"), os.path.join(args.out, f"layer_{i}_norm_mlp.bin"))

    total_mb = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)) // (1024*1024)
    print(f"\nDone. {args.out}  total {total_mb} MB")


if __name__ == "__main__":
    main()
