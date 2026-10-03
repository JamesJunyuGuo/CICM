"""Error signature analysis for single-variable P0 stale-binding evals."""

import argparse
import collections
import json
from pathlib import Path


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _init_counts():
    return {"n": 0, "n_wrong": 0, "within_stale": 0, "cross_value": 0, "other": 0}


def _finish_counts(counts):
    wrong = counts["n_wrong"]
    out = dict(counts)
    out["within_stale_rate_among_wrong"] = counts["within_stale"] / wrong if wrong else 0.0
    out["cross_value_rate_among_wrong"] = counts["cross_value"] / wrong if wrong else 0.0
    out["other_rate_among_wrong"] = counts["other"] / wrong if wrong else 0.0
    out["error_rate"] = wrong / counts["n"] if counts["n"] else 0.0
    return out


def summarize_p0_errors(data_rows, result_rows, model=None):
    data_by_id = {row["id"]: row for row in data_rows}
    all_values_by_id = {
        row["id"]: set(row.get("stale_values", [])) | {row.get("gold")}
        for row in data_rows
    }
    global_values = set().union(*all_values_by_id.values()) if all_values_by_id else set()
    overall = _init_counts()
    by_cell = collections.defaultdict(_init_counts)

    for result in result_rows:
        row = data_by_id[result["id"]]
        key = (row["condition"], row["n_lines"], row["interference_load"])
        pred = result.get("pred")
        correct = int(result.get("correct", pred == row["gold"]))
        for bucket in [overall, by_cell[key]]:
            bucket["n"] += 1
        if correct:
            continue

        stale = set(row.get("stale_values", []))
        if pred in stale:
            kind = "within_stale"
        elif pred in global_values - all_values_by_id[row["id"]]:
            kind = "cross_value"
        else:
            kind = "other"
        for bucket in [overall, by_cell[key]]:
            bucket["n_wrong"] += 1
            bucket[kind] += 1

    cells = []
    for (condition, n_lines, load), counts in sorted(by_cell.items()):
        cells.append(
            {
                "condition": condition,
                "n_lines": n_lines,
                "interference_load": load,
                **_finish_counts(counts),
            }
        )
    return {
        "model": model,
        "n": len(result_rows),
        "overall": _finish_counts(overall),
        "cells": cells,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/p0.jsonl")
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    summary = summarize_p0_errors(load_jsonl(args.data), load_jsonl(args.results), args.model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
