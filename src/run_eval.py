"""P0 eval: run a chat model on the task file, score, aggregate.

Greedy decoding, batched. Answer is parsed as the first integer in the model's
output and compared to the gold integer. Emits per-example results (jsonl) and a
summary (json) grouped by (condition, n_lines, interference_load).
"""

import argparse
import collections
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_INT_RE = re.compile(r"-?\d+")


def parse_int(text):
    m = _INT_RE.search(text)
    return int(m.group()) if m else None


def load_rows(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/p0.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="results/overwrite_task/p0_qwen7b.jsonl")
    ap.add_argument("--summary", default="results/overwrite_task/p0_qwen7b.summary.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument(
        "--attn-impl",
        default=None,
        help="attn_implementation override (e.g. 'eager'); default = library default (sdpa)",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = load_rows(args.data, args.limit)
    print(f"loaded {len(rows)} examples from {args.data}")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto"}
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    def build_input(prompt):
        msgs = [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    results = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = [build_input(r["prompt"]) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        for r, text in zip(batch, decoded):
            pred = parse_int(text)
            results.append(
                {
                    "id": r["id"],
                    "condition": r["condition"],
                    "n_lines": r["n_lines"],
                    "interference_load": r["interference_load"],
                    "gold": r["gold"],
                    "pred": pred,
                    "raw": text.strip(),
                    "correct": int(pred == r["gold"]),
                }
            )
        print(f"  {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # aggregate
    agg = collections.defaultdict(lambda: [0, 0])
    for r in results:
        key = (r["condition"], r["n_lines"], r["interference_load"])
        agg[key][0] += r["correct"]
        agg[key][1] += 1
    summary = {
        "model": args.model,
        "data": args.data,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl or "default",
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "n": len(results),
        "cells": [],
    }
    for (cond, n_lines, I), (c, t) in sorted(agg.items()):
        summary["cells"].append(
            {
                "condition": cond,
                "n_lines": n_lines,
                "interference_load": I,
                "accuracy": round(c / t, 4),
                "n": t,
            }
        )
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===")
    for cell in summary["cells"]:
        print(
            f"{cell['condition']:12s} n_lines={cell['n_lines']:3d} "
            f"I={cell['interference_load']:2d}  acc={cell['accuracy']:.3f} (n={cell['n']})"
        )


if __name__ == "__main__":
    main()
