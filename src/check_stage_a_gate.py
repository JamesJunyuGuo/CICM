"""Task 2c acceptance gate for Stage A fp32 eager extraction."""

import argparse
import collections
import json
import sys


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _cell_key(row):
    return (row["condition"], int(row["n_lines"]), int(row["interference_load"]))


def compute_cells(rows):
    counts = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        key = _cell_key(row)
        counts[key][0] += int(row["correct"])
        counts[key][1] += 1
    return {
        key: {"accuracy": correct / total, "n": total}
        for key, (correct, total) in counts.items()
    }


def load_anchor_cells(path):
    with open(path) as f:
        summary = json.load(f)
    cells = {}
    for cell in summary["cells"]:
        key = (
            cell["condition"],
            int(cell["n_lines"]),
            int(cell["interference_load"]),
        )
        cells[key] = {"accuracy": float(cell["accuracy"]), "n": int(cell["n"])}
    return cells


def compare_to_anchor(observed, anchor, tolerance=0.04, simple_floor=0.95):
    rows = []
    failures = []
    for key in sorted(set(observed).intersection(anchor)):
        obs = observed[key]
        ref = anchor[key]
        delta = obs["accuracy"] - ref["accuracy"]
        abs_delta = abs(delta)
        passed = abs_delta <= tolerance
        label = f"{key[0]}@{key[1]} I={key[2]}"
        if not passed:
            failures.append(
                f"{label} delta {delta:+.4f} exceeds tolerance +/-{tolerance:.4f}"
            )
        if key == ("simple", 40, 0) and obs["accuracy"] < simple_floor:
            passed = False
            failures.append(
                f"simple@40 accuracy {obs['accuracy']:.4f} below floor {simple_floor:.4f}"
            )
        rows.append(
            {
                "condition": key[0],
                "n_lines": key[1],
                "interference_load": key[2],
                "observed_accuracy": obs["accuracy"],
                "anchor_accuracy": ref["accuracy"],
                "delta": delta,
                "abs_delta": abs_delta,
                "observed_n": obs["n"],
                "anchor_n": ref["n"],
                "passed": passed,
            }
        )

    simple_key = ("simple", 40, 0)
    if simple_key not in observed:
        failures.append("simple@40 missing from extraction index")
    if not rows:
        failures.append("no overlapping cells between extraction index and anchor")

    return {"passed": not failures, "cells": rows, "failures": failures}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="results/stage_a/extract_index.jsonl")
    ap.add_argument(
        "--anchor", default="results/stage_a_check_sdpa.summary.json"
    )
    ap.add_argument("--out", default="results/stage_a/task2c_gate.json")
    ap.add_argument("--tolerance", type=float, default=0.04)
    ap.add_argument("--simple-floor", type=float, default=0.95)
    args = ap.parse_args()

    observed = compute_cells(load_jsonl(args.index))
    anchor = load_anchor_cells(args.anchor)
    result = compare_to_anchor(
        observed, anchor, tolerance=args.tolerance, simple_floor=args.simple_floor
    )
    result["index"] = args.index
    result["anchor"] = args.anchor
    result["tolerance"] = args.tolerance
    result["simple_floor"] = args.simple_floor
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
