"""Hard-tier frontier break-point probe (Stage F, task 3).

Question: models that saturate the standard battery (Qwen-72B, GPT-4o; plus
Llama-70B which already degrades) — do they break at extreme interference, and
when they do, is the error signature still within-stale?

Generates a hard grid with gen_tasks.make_example (I in {32, 64}, long
contexts, plus a simple control at max length for parse/contamination sanity),
then reuses the OpenRouter machinery from openrouter_sweep.
"""

import argparse
import json
import random
from pathlib import Path

from gen_tasks import make_example
from openrouter_sweep import run_requests, summarize

CELLS = [
    # (condition, n_lines, I)
    ("simple", 320, 0),
    ("interference", 160, 32),
    ("interference", 320, 32),
    ("interference", 320, 64),
]


def build_hard_rows(n_per_cell, seed):
    rng = random.Random(seed)
    rows = []
    uid = 0
    for cond, n_lines, I in CELLS:
        for _ in range(n_per_cell):
            ex = make_example(cond, n_lines, I, rng)
            ex["id"] = uid
            ex["openrouter_source"] = "hard_tier"
            uid += 1
            rows.append(ex)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=(
        "qwen/qwen-2.5-72b-instruct,"
        "meta-llama/llama-3.1-70b-instruct,"
        "openai/gpt-4o"
    ))
    ap.add_argument("--n-per-cell", type=int, default=50)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--out", default="results/hard_tier_and_transport/hard_tier/sweep.jsonl")
    ap.add_argument("--summary", default="results/hard_tier_and_transport/hard_tier/sweep.summary.json")
    ap.add_argument("--data-out", default="data/hard_tier.jsonl")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    rows = build_hard_rows(args.n_per_cell, args.seed)
    Path(args.data_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.data_out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"hard-tier data: {len(rows)} rows -> {args.data_out}", flush=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results = run_requests(
        rows, models, args.max_tokens, args.concurrency, args.retries,
        args.timeout, out_path=args.out,
    )
    summary = summarize(results, {
        "task": "hard_tier_break_point",
        "cells": CELLS,
        "n_per_cell": args.n_per_cell,
        "seed": args.seed,
        "models": models,
    })
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}, {args.summary}", flush=True)


if __name__ == "__main__":
    main()
