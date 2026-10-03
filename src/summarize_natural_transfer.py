"""Summarize naturalistic Stage E transfer arms."""

import argparse
import json
from pathlib import Path


def read_json(path):
    with open(path) as f:
        return json.load(f)


def summarize_arm(summary, high_loads):
    by_family = {}
    simple = []
    for cell in summary.get("cells", []):
        family = cell.get("stage_d_cell")
        if cell.get("condition") == "simple":
            simple.append(cell)
        if cell.get("condition") != "interference":
            continue
        if cell.get("interference_load") not in high_loads:
            continue
        slot = by_family.setdefault(
            family,
            {"n": 0, "correct": 0.0, "stale": 0.0, "error": 0.0},
        )
        n = cell["n"]
        slot["n"] += n
        slot["correct"] += cell["accuracy"] * n
        slot["stale"] += cell.get("stale_rate", 0.0) * n
        slot["error"] += (1.0 - cell["accuracy"]) * n
    out = {}
    for family, vals in by_family.items():
        n = vals["n"]
        errors = vals["error"]
        out[family] = {
            "accuracy": vals["correct"] / n if n else None,
            "n": n,
            "stale_rate": vals["stale"] / n if n else None,
            "within_stale_rate_among_errors": vals["stale"] / errors if errors else 0.0,
        }
    return out, simple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/llama_and_scale/natural/qwen_baseline_ai.summary.json")
    ap.add_argument("--arm-l", required=True)
    ap.add_argument("--random-heads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-label", default="seed")
    ap.add_argument("--high-loads", default="4,8")
    args = ap.parse_args()

    high_loads = {int(x) for x in args.high_loads.split(",") if x}
    summaries = {
        "baseline": read_json(args.baseline),
        "arm_l": read_json(args.arm_l),
        "random_heads": read_json(args.random_heads),
    }
    high_i = {}
    for arm, summary in summaries.items():
        arm_summary, _simple = summarize_arm(summary, high_loads)
        for family, vals in arm_summary.items():
            high_i.setdefault(family, {})[arm] = vals

    for family, arms in high_i.items():
        base = arms.get("baseline", {}).get("accuracy")
        for arm, vals in arms.items():
            vals["gain_vs_baseline"] = None if base is None or vals["accuracy"] is None else vals["accuracy"] - base
        if "arm_l" in arms and "random_heads" in arms:
            arms["arm_l_vs_random_gap"] = arms["arm_l"]["accuracy"] - arms["random_heads"]["accuracy"]

    out = {
        "stage": "E",
        "track": "naturalistic_transfer",
        "seed_label": args.seed_label,
        "high_loads": sorted(high_loads),
        "high_i_summary": high_i,
        "source_summaries": {
            "baseline": args.baseline,
            "arm_l": args.arm_l,
            "random_heads": args.random_heads,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
