"""Build stable-label sidecar and optional stable-primary artifacts."""

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_stable_rows(extract_rows, sdpa_by_id):
    rows = []
    for row in extract_rows:
        rid = row["id"]
        if rid not in sdpa_by_id:
            raise KeyError(f"id={rid} missing from sdpa rows")
        sdpa = sdpa_by_id[rid]
        fp32_correct = int(row["correct"])
        sdpa_correct = int(sdpa["correct"])
        if fp32_correct and sdpa_correct:
            stable_label = "stable_correct"
        elif not fp32_correct and not sdpa_correct:
            stable_label = "stable_wrong"
        else:
            stable_label = "flipped"
        rows.append(
            {
                "row_index": row["row_index"],
                "id": rid,
                "condition": row["condition"],
                "n_lines": row["n_lines"],
                "interference_load": row["interference_load"],
                "fp32_correct": fp32_correct,
                "sdpa_correct": sdpa_correct,
                "fp32_pred": row.get("pred"),
                "sdpa_pred": sdpa.get("pred"),
                "fp32_raw": row.get("raw"),
                "sdpa_raw": sdpa.get("raw"),
                "stable_label": stable_label,
                "stable_primary": int(stable_label != "flipped"),
            }
        )
    return rows


def summarize_stability(rows):
    counts = collections.Counter(row["stable_label"] for row in rows)
    by_cell_counts = collections.defaultdict(collections.Counter)
    for row in rows:
        key = (row["condition"], row["n_lines"], row["interference_load"])
        by_cell_counts[key][row["stable_label"]] += 1

    by_cell = []
    for (condition, n_lines, load), counter in sorted(by_cell_counts.items()):
        n = sum(counter.values())
        by_cell.append(
            {
                "condition": condition,
                "n_lines": n_lines,
                "interference_load": load,
                "n": n,
                "stable_correct": counter["stable_correct"],
                "stable_wrong": counter["stable_wrong"],
                "flipped": counter["flipped"],
                "flip_rate": counter["flipped"] / n if n else 0.0,
                "agreement_rate": (counter["stable_correct"] + counter["stable_wrong"]) / n
                if n
                else 0.0,
            }
        )

    return {
        "n": len(rows),
        "counts": {
            "stable_correct": counts["stable_correct"],
            "stable_wrong": counts["stable_wrong"],
            "flipped": counts["flipped"],
        },
        "by_cell": by_cell,
    }


def make_stable_primary_index(extract_rows, stable_rows):
    by_row_index = {row["row_index"]: row for row in stable_rows}
    out = []
    for row in extract_rows:
        stable = by_row_index[row["row_index"]]
        if not stable["stable_primary"]:
            continue
        new_row = dict(row)
        new_row.update(stable)
        new_row["correct"] = int(stable["stable_label"] == "stable_correct")
        out.append(new_row)
    return out


def write_subset_npz(npz_in, npz_out, row_indices, total_rows):
    data = np.load(npz_in)
    arrays = {}
    row_indices = np.asarray(row_indices, dtype=np.int64)
    for key in data.files:
        value = data[key]
        if value.shape and value.shape[0] == total_rows:
            arrays[key] = value[row_indices]
        else:
            arrays[key] = value
    arrays["stable_subset"] = np.array("stable_primary")
    np.savez_compressed(npz_out, **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-index", default="results/stage_a/extract_index.jsonl")
    ap.add_argument("--sdpa", default="results/stage_a_check_sdpa.jsonl")
    ap.add_argument("--out", default="results/stage_a/stable_labels.jsonl")
    ap.add_argument("--summary", default="results/stage_a/stable_labels.summary.json")
    ap.add_argument("--npz-in", default="results/stage_a/extract.npz")
    ap.add_argument("--stable-index-out", default="results/stage_a/extract_stable_index.jsonl")
    ap.add_argument("--stable-npz-out", default="results/stage_a/extract_stable.npz")
    args = ap.parse_args()

    extract_rows = load_jsonl(args.extract_index)
    sdpa_rows = load_jsonl(args.sdpa)
    sdpa_by_id = {row["id"]: row for row in sdpa_rows}

    stable_rows = build_stable_rows(extract_rows, sdpa_by_id)
    summary = summarize_stability(stable_rows)
    stable_index = make_stable_primary_index(extract_rows, stable_rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, stable_rows)
    write_jsonl(args.stable_index_out, stable_index)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    write_subset_npz(
        args.npz_in,
        args.stable_npz_out,
        [row["row_index"] for row in stable_index],
        len(extract_rows),
    )

    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}, {args.stable_index_out}, {args.stable_npz_out}")


if __name__ == "__main__":
    main()
