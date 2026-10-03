"""Eval for the parallel-maintenance task: multi-stream scoring.

Parses `A=.., B=.., ...` from the model output and scores each stream against
gold. Reports, per (condition, K, U): mean per-stream accuracy and the
all-streams-correct rate (the real capacity measure). Greedy, batched.
"""

import argparse
import collections
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_PAIR_RE = re.compile(r"([A-Z])\s*=\s*(-?\d+)")


def parse_pairs(text):
    return {m.group(1): int(m.group(2)) for m in _PAIR_RE.finditer(text)}


def load_rows(path, limit=None):
    rows = [json.loads(l) for l in open(path)]
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/parallel.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="results/parallel_qwen7b.jsonl")
    ap.add_argument("--summary", default="results/parallel_qwen7b.summary.json")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = load_rows(args.data, args.limit)
    print(f"loaded {len(rows)} examples from {args.data}")

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
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
            pred = parse_pairs(text)
            gold = r["gold"]
            per_stream = {s: int(pred.get(s) == v) for s, v in gold.items()}
            n_correct = sum(per_stream.values())
            results.append(
                {
                    "id": r["id"],
                    "condition": r["condition"],
                    "K": r["K"],
                    "U": r["U"],
                    "n_streams": len(gold),
                    "n_correct": n_correct,
                    "all_correct": int(n_correct == len(gold)),
                    "raw": text.strip(),
                }
            )
        print(f"  {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # aggregate: per-stream accuracy = correct streams / total streams; all-correct rate
    agg = collections.defaultdict(lambda: [0, 0, 0])  # [correct_streams, total_streams, all_correct]
    for r in results:
        key = (r["condition"], r["K"], r["U"])
        agg[key][0] += r["n_correct"]
        agg[key][1] += r["n_streams"]
        agg[key][2] += r["all_correct"]
    n_per = collections.Counter((r["condition"], r["K"], r["U"]) for r in results)

    summary = {"model": args.model, "n": len(results), "cells": []}
    for key in sorted(agg):
        cond, K, U = key
        cs, ts, ac = agg[key]
        summary["cells"].append(
            {
                "condition": cond,
                "K": K,
                "U": U,
                "per_stream_acc": round(cs / ts, 4),
                "all_correct_rate": round(ac / n_per[key], 4),
                "n": n_per[key],
            }
        )
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary (per-stream acc | all-correct rate) ===")
    for c in summary["cells"]:
        print(
            f"{c['condition']:11s} K={c['K']:2d} U={c['U']:2d}  "
            f"stream_acc={c['per_stream_acc']:.3f}  all_correct={c['all_correct_rate']:.3f} (n={c['n']})"
        )


if __name__ == "__main__":
    main()
