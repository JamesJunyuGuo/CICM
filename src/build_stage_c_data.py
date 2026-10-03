"""Build Stage C train/eval datasets and manifest."""

import argparse
import json
import os
import random
from pathlib import Path

from gen_parallel import make_example as make_parallel_example
from gen_paraphrase import make_paraphrase_example
from gen_tasks import make_example as make_overwrite_example


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_top_heads(path, top_k=8):
    with open(path) as f:
        summary = json.load(f)
    return summary["top_heads_by_correct_p_last"][:top_k]


def make_train_rows(n_examples, seed):
    rng = random.Random(seed)
    rows = []
    for idx in range(n_examples):
        I = rng.randint(1, 6)
        min_lines = I + 1 + 3
        n_lines = rng.randint(max(12, min_lines), 48)
        row = make_overwrite_example("interference", n_lines, I, rng)
        row["id"] = idx
        row["task_family"] = "overwrite"
        row["split"] = "train"
        rows.append(row)
    return rows


def make_eval_rows(n_per_cell, seed):
    rng = random.Random(seed)
    rows = []
    uid = 0

    def add(row, family, split, cell):
        nonlocal uid
        row["id"] = uid
        uid += 1
        row["task_family"] = family
        row["split"] = split
        row["stage_c_cell"] = cell
        rows.append(row)

    for n_lines in [20, 40, 80, 160]:
        for _ in range(n_per_cell):
            add(
                make_overwrite_example("simple", n_lines, 0, rng),
                "overwrite",
                "eval",
                "simple_control",
            )

    for n_lines in [20, 40, 80]:
        for I in [2, 4, 8]:
            for _ in range(n_per_cell):
                add(
                    make_overwrite_example("interference", n_lines, I, rng),
                    "overwrite",
                    "eval",
                    "id_heldout",
                )

    for n_lines in [80, 160]:
        for I in [12, 16]:
            for _ in range(n_per_cell):
                add(
                    make_overwrite_example("interference", n_lines, I, rng),
                    "overwrite",
                    "eval",
                    "ood_load",
                )

    for n_lines in [20, 40, 80]:
        for I in [2, 4, 8]:
            for _ in range(n_per_cell):
                add(
                    make_paraphrase_example("interference", n_lines, I, rng),
                    "paraphrase",
                    "eval",
                    "ood_surface",
                )

    for condition in ["serial", "interleaved"]:
        for K in [4, 8]:
            for _ in range(n_per_cell):
                add(
                    make_parallel_example(condition, K, 4, rng),
                    "parallel",
                    "eval",
                    "parallel_transfer",
                )
    return rows


def make_smoke_rows(eval_rows, n_per_group=2):
    buckets = {}
    for row in eval_rows:
        key = (
            row["stage_c_cell"],
            row["task_family"],
            row.get("condition"),
            row.get("n_lines"),
            row.get("interference_load"),
            row.get("K"),
            row.get("U"),
        )
        buckets.setdefault(key, []).append(row)
    smoke = []
    for key in sorted(buckets, key=str):
        smoke.extend(buckets[key][:n_per_group])
    return smoke


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/stage_c")
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--eval-n-per-cell", type=int, default=100)
    ap.add_argument("--train-seed", type=int, default=10)
    ap.add_argument("--eval-seed", type=int, default=2)
    ap.add_argument("--a1-summary", default="results/stage_a/a1_stable_summary.json")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = make_train_rows(args.train_n, args.train_seed)
    eval_rows = make_eval_rows(args.eval_n_per_cell, args.eval_seed)
    train_path = out / "train_overwrite.jsonl"
    eval_path = out / "eval_battery.jsonl"
    smoke_path = out / "eval_smoke.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(eval_path, eval_rows)
    write_jsonl(smoke_path, make_smoke_rows(eval_rows))

    manifest = {
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "eval_smoke_path": str(smoke_path),
        "train_seed": args.train_seed,
        "eval_seed": args.eval_seed,
        "train_n": len(train),
        "eval_n": len(eval_rows),
        "eval_n_per_cell": args.eval_n_per_cell,
        "target_heads": load_top_heads(args.a1_summary, 8),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {train_path} ({len(train)})")
    print(f"wrote {eval_path} ({len(eval_rows)})")
    print(f"wrote {smoke_path}")
    print(f"wrote {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
