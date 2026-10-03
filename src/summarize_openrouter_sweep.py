"""Summarize an OpenRouter sweep JSONL produced outside the sandbox."""

import argparse
import collections
import json
from pathlib import Path


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows, budget):
    groups = collections.defaultdict(lambda: collections.Counter())
    for row in rows:
        key = (
            row["model"],
            row["openrouter_source"],
            row["condition"],
            row["n_lines"],
            row["interference_load"],
        )
        groups[key]["n"] += 1
        groups[key]["correct"] += int(row["correct"])
        if not row["correct"]:
            groups[key]["wrong"] += 1
            groups[key][row["answer_type"]] += 1

    cells = []
    for (model, source, condition, n_lines, load), counter in sorted(groups.items(), key=lambda item: str(item[0])):
        wrong = counter["wrong"]
        cells.append(
            {
                "model": model,
                "openrouter_source": source,
                "condition": condition,
                "n_lines": n_lines,
                "interference_load": load,
                "n": counter["n"],
                "accuracy": counter["correct"] / counter["n"],
                "within_stale_rate_among_errors": counter["within_stale"] / wrong if wrong else None,
                "other_rate_among_errors": counter["other"] / wrong if wrong else None,
            }
        )

    by_model = collections.Counter(row["model"] for row in rows)
    expected = int(budget.get("n_calls", 0) or 0)
    return {
        "models": budget.get("models", sorted(by_model)),
        "budget": budget,
        "n": len(rows),
        "expected_n": expected,
        "complete": bool(expected and len(rows) == expected),
        "counts_by_model": dict(sorted(by_model.items())),
        "cells": cells,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="results/llama_and_scale/openrouter/sweep_v2.jsonl")
    ap.add_argument("--budget", default="results/llama_and_scale/openrouter/sweep_v2_budget.json")
    ap.add_argument("--summary", default="results/llama_and_scale/openrouter/sweep_v2.summary.json")
    args = ap.parse_args()

    rows = load_jsonl(args.jsonl)
    with open(args.budget) as f:
        budget = json.load(f)
    summary = summarize(rows, budget)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.summary} complete={summary['complete']} n={summary['n']}/{summary['expected_n']}")


if __name__ == "__main__":
    main()
