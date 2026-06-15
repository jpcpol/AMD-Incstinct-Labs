#!/usr/bin/env python3
"""
make_reference.py — produce the HF greedy reference for Stage-C C3 validation.

Given a model and a prompt string, runs HuggingFace greedy decoding on CPU and
writes:
  prompt_ids.txt  — input token ids (one per line)
  ref_tokens.txt  — full token sequence (prompt + greedy generated), one per line

The C++ runtime (validate_runtime) reads these and must reproduce ref_tokens.

Usage (on VM venv):
  python make_reference.py --model Qwen/Qwen2.5-0.5B \\
      --prompt "The capital of France is" --max_new 32 --out /tmp/ref
"""
import argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max_new", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids
    print(f"prompt: {args.prompt!r}")
    print(f"prompt_ids: {ids[0].tolist()}")

    # Diagnostic: logits of the last prompt token (first decode step).
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()   # [vocab]
    topv, topi = torch.topk(logits, 10)
    print("HF first-step top-10 logits:")
    for v, idx in zip(topv.tolist(), topi.tolist()):
        print(f"   id={idx:6d}  logit={v:8.3f}  {tok.decode([idx])!r}")
    # Dump full first-step logits (fp32) for numerical comparison.
    logits.numpy().tofile(os.path.join(args.out, "ref_logits0.bin"))

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                             num_beams=1, temperature=None, top_p=None, top_k=None)

    full = out[0].tolist()
    gen  = full[ids.shape[1]:]
    print(f"generated_ids: {gen}")
    print(f"generated text: {tok.decode(gen)!r}")

    with open(os.path.join(args.out, "prompt_ids.txt"), "w") as f:
        for t in ids[0].tolist(): f.write(f"{t}\n")
    with open(os.path.join(args.out, "ref_tokens.txt"), "w") as f:
        for t in full: f.write(f"{t}\n")
    print(f"\nWrote prompt_ids.txt ({ids.shape[1]} ids) and "
          f"ref_tokens.txt ({len(full)} ids) to {args.out}")


if __name__ == "__main__":
    main()
