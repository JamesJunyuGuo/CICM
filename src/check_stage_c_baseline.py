"""C0 gate: validate Stage C baseline battery before any training."""

import argparse
import json
from pathlib import Path


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def cells_for(summary, **filters):
    out = []
    for cell in summary["cells"]:
        ok = True
        for key, value in filters.items():
            if cell.get(key) != value:
                ok = False
                break
        if ok:
            out.append(cell)
    return out


def find_cell(summary, **filters):
    matches = cells_for(summary, **filters)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"multiple cells for {filters}: {matches}")
    return matches[0]


def check_gate(summary, simple_threshold=0.98):
    simple = cells_for(
        summary,
        task_family="overwrite",
        condition="simple",
        interference_load=0,
    )
    min_simple = min((cell["accuracy"] for cell in simple), default=0.0)
    id40 = [
        find_cell(
            summary,
            task_family="overwrite",
            condition="interference",
            n_lines=40,
            interference_load=load,
        )
        for load in [2, 4, 8]
    ]
    missing = [load for load, cell in zip([2, 4, 8], id40) if cell is None]
    id40_acc = [None if cell is None else cell["accuracy"] for cell in id40]
    degradation = (
        not missing
        and id40_acc[0] >= id40_acc[1]
        and id40_acc[1] >= id40_acc[2]
        and id40_acc[0] > id40_acc[2]
    )
    passed = min_simple >= simple_threshold and degradation
    return {
        "pass": bool(passed),
        "simple_threshold": simple_threshold,
        "min_simple_accuracy": min_simple,
        "id_n40_accuracy_by_I": {"2": id40_acc[0], "4": id40_acc[1], "8": id40_acc[2]},
        "id_n40_degradation": bool(degradation),
        "missing_cells": missing,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/stage_c/baseline.summary.json")
    ap.add_argument("--out", default="results/stage_c/c0_baseline_gate.json")
    ap.add_argument("--simple-threshold", type=float, default=0.98)
    args = ap.parse_args()
    result = check_gate(load_summary(args.summary), args.simple_threshold)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit("C0 baseline gate failed")


if __name__ == "__main__":
    main()
