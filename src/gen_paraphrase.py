"""Natural-language overwrite task generator for Stage C OOD-surface eval."""

import argparse
import json
import os
import random

from gen_tasks import _rand_names, _rand_values


def make_paraphrase_example(condition, n_lines, I, rng=None, rng_seed=None, tail_min=3):
    if rng is None:
        rng = random.Random(rng_seed)
    if condition == "simple":
        names = _rand_names(n_lines, rng)
        values = _rand_values(n_lines, rng)
        lines = [(names[i], values[i]) for i in range(n_lines)]
        rng.shuffle(lines)
        q_idx = rng.randrange(0, n_lines - tail_min)
        target, gold = lines[q_idx]
        target_writes = [q_idx]
        stale_values = []
    elif condition == "interference":
        assert n_lines >= I + 1 + tail_min
        n_distractor = n_lines - (I + 1)
        names = _rand_names(1 + n_distractor, rng)
        target = names[0]
        distractors = names[1:]
        target_values = _rand_values(I + 1, rng)
        stale_values = target_values[:-1]
        gold = target_values[-1]
        distractor_values = _rand_values(n_distractor, rng)
        last_allowed = n_lines - 1 - tail_min
        target_writes = sorted(rng.sample(range(0, last_allowed + 1), I + 1))
        target_pos = set(target_writes)
        distractor_positions = [i for i in range(n_lines) if i not in target_pos]
        lines = [None] * n_lines
        for idx, pos in enumerate(target_writes):
            lines[pos] = (target, target_values[idx])
        for name, value, pos in zip(distractors, distractor_values, distractor_positions):
            lines[pos] = (name, value)
    else:
        raise ValueError(condition)

    statements = [
        f"The budget for {name} is now {value}." for name, value in lines
    ]
    prompt = (
        "Read the following chronological budget updates. If a code is updated "
        "more than once, the latest update is the current one.\n\n"
        + "\n".join(statements)
        + f"\n\nQuestion: What is the current budget for {target}?\n"
        "Answer with only the integer value."
    )
    return {
        "task_family": "paraphrase",
        "condition": condition,
        "n_lines": n_lines,
        "interference_load": 0 if condition == "simple" else I,
        "target_var": target,
        "gold": gold,
        "prompt": prompt,
        "target_write_positions": target_writes,
        "stale_values": stale_values,
    }


def build_dataset(out_path, n_per_cell=100, seed=2):
    rng = random.Random(seed)
    rows = []
    uid = 0
    for n_lines in [20, 40, 80]:
        for _ in range(n_per_cell):
            row = make_paraphrase_example("simple", n_lines, 0, rng)
            row["id"] = uid
            uid += 1
            rows.append(row)
        for I in [2, 4, 8]:
            for _ in range(n_per_cell):
                row = make_paraphrase_example("interference", n_lines, I, rng)
                row["id"] = uid
                uid += 1
                rows.append(row)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/head_adapters/paraphrase_eval.jsonl")
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    n = build_dataset(args.out, args.n_per_cell, args.seed)
    print(f"wrote {n} examples to {args.out}")
