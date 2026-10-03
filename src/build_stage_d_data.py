"""Build Stage D broader contextual-management eval data."""

import argparse
import collections
import json
from pathlib import Path

from gen_stage_d_context import make_all_stage_d_examples


FAMILY_FILES = {
    "ruler_style": "eval_ruler_style.jsonl",
    "babilong_style": "eval_babilong_style.jsonl",
    "michelangelo_style": "eval_michelangelo_style.jsonl",
    "state_suite": "eval_state_suite.jsonl",
}


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_stage_d_rows(n_per_cell=100, seed=20):
    return make_all_stage_d_examples(n_per_cell=n_per_cell, seed=seed)


def summarize(rows):
    by_family = collections.Counter(row["task_family"] for row in rows)
    by_cell = collections.Counter(row["stage_d_cell"] for row in rows)
    by_answer_type = collections.Counter(row["answer_type"] for row in rows)
    return {
        "n_total": len(rows),
        "by_family": dict(sorted(by_family.items())),
        "by_cell": dict(sorted(by_cell.items())),
        "by_answer_type": dict(sorted(by_answer_type.items())),
    }


def write_stage_d_outputs(out_dir, n_per_cell=100, seed=20):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = make_stage_d_rows(n_per_cell=n_per_cell, seed=seed)
    write_jsonl(out / "eval_all.jsonl", rows)

    family_paths = {}
    for family, filename in FAMILY_FILES.items():
        family_rows = [row for row in rows if row["task_family"] == family]
        path = out / filename
        write_jsonl(path, family_rows)
        family_paths[family] = str(path)

    manifest = {
        "stage": "D",
        "description": "Broader contextual-management eval data: RULER-style, BABILong-style, Michelangelo-style, and controlled state-suite probes.",
        "seed": seed,
        "n_per_cell": n_per_cell,
        "eval_all_path": str(out / "eval_all.jsonl"),
        "family_paths": family_paths,
        **summarize(rows),
        "notes": [
            "These are benchmark-inspired controlled generators, not official external benchmark splits.",
            "Rows preserve current/stale span metadata for future Stage A/B-style analysis.",
        ],
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/stage_d")
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()

    manifest = write_stage_d_outputs(args.out_dir, args.n_per_cell, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
