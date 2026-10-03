"""P0 task generator: "current value under stale interference".

We probe one context-management primitive: reading the *current* value of a
variable when earlier (stale) values of the SAME variable are present as
distractors. Two conditions share the exact same surface format and the same
number of assignment lines; they differ ONLY in whether the queried variable is
overwritten:

  - simple        : every variable is assigned exactly once (query = plain lookup).
  - interference  : the queried variable is assigned (I+1) times; the answer is the
                    LAST assignment. The I earlier values are stale distractors.

Length is matched across conditions at a fixed `n_lines`, so any accuracy gap is
attributable to the overwrite/interference structure, not to context length.

Recency control: at least `tail_min` distractor lines are placed AFTER the
target's final write, so "return the current value" cannot be solved by simply
copying the last line.

Each example also carries metadata (all write positions of the target, its stale
values, current value, query line index) so later mechanistic passes (probing /
attention analysis) can reuse the exact same data without regeneration.
"""

import argparse
import json
import random
import string
import re


def _rand_names(n, rng):
    """n distinct short variable names."""
    names = set()
    while len(names) < n:
        names.add("".join(rng.choice(string.ascii_lowercase) for _ in range(2)))
    return list(names)


def _rand_values(n, rng, lo=10, hi=9999):
    """n distinct integer values (distinct so a wrong answer is unambiguous)."""
    vals = set()
    while len(vals) < n:
        vals.add(rng.randint(lo, hi))
    return list(vals)


def make_example(condition, n_lines, I, rng, tail_min=3):
    """Build one example.

    condition: "simple" or "interference".
    n_lines:   total number of `var = value` lines (matched across conditions).
    I:         interference load = number of stale writes of the target
               (only used for "interference"; must satisfy n_lines >= I + 1 + tail_min).
    """
    if condition == "simple":
        n_vars = n_lines
        names = _rand_names(n_vars, rng)
        values = _rand_values(n_vars, rng)
        lines = [(names[i], values[i]) for i in range(n_vars)]
        rng.shuffle(lines)
        # query a variable that is NOT in the last tail_min lines (recency control)
        q_idx = rng.randrange(0, n_lines - tail_min)
        target, gold = lines[q_idx]
        target_writes = [q_idx]
        stale_values = []
    elif condition == "interference":
        assert n_lines >= I + 1 + tail_min, (
            f"n_lines={n_lines} too small for I={I} with tail_min={tail_min}"
        )
        n_distractor = n_lines - (I + 1)
        names = _rand_names(1 + n_distractor, rng)
        target = names[0]
        distractor_names = names[1:]
        # I+1 distinct values for the target: last one is current (gold)
        tvals = _rand_values(I + 1, rng)
        current = tvals[-1]
        stale_values = tvals[:-1]
        dvals = _rand_values(n_distractor, rng)

        # choose chronological positions for the target's writes; final write must
        # leave >= tail_min lines after it.
        last_allowed = n_lines - 1 - tail_min
        write_positions = sorted(rng.sample(range(0, last_allowed + 1), I + 1))
        pos_set = set(write_positions)
        distractor_positions = [p for p in range(n_lines) if p not in pos_set]

        lines = [None] * n_lines
        for k, p in enumerate(write_positions):
            lines[p] = (target, tvals[k])
        for name, val, p in zip(distractor_names, dvals, distractor_positions):
            lines[p] = (name, val)

        gold = current
        target_writes = write_positions
    else:
        raise ValueError(condition)

    body = "\n".join(f"{name} = {val}" for name, val in lines)
    prompt = (
        "Here is a list of variable assignments, in order. A variable may be "
        "assigned more than once; only its most recent assignment counts.\n\n"
        f"{body}\n\n"
        f"Question: What is the current value of {target}?\n"
        "Answer with only the integer value."
    )
    return {
        "condition": condition,
        "n_lines": n_lines,
        "interference_load": (0 if condition == "simple" else I),
        "target_var": target,
        "gold": gold,
        "prompt": prompt,
        # metadata for later mechanistic reuse:
        "target_write_positions": target_writes,
        "stale_values": stale_values,
    }


_ASSIGNMENT_RE = re.compile(r"^([A-Za-z]+) = (-?\d+)$")


def _assignment_lines_from_prompt(prompt):
    assignments = []
    for line in prompt.splitlines():
        match = _ASSIGNMENT_RE.match(line.strip())
        if match:
            assignments.append((match.group(1), int(match.group(2))))
    return assignments


def _fresh_same_length_name(existing, rng, length):
    alphabet = string.ascii_lowercase
    while True:
        name = "".join(rng.choice(alphabet) for _ in range(length))
        if name not in existing:
            existing.add(name)
            return name


def _prompt_from_assignments(assignments, target):
    body = "\n".join(f"{name} = {val}" for name, val in assignments)
    return (
        "Here is a list of variable assignments, in order. A variable may be "
        "assigned more than once; only its most recent assignment counts.\n\n"
        f"{body}\n\n"
        f"Question: What is the current value of {target}?\n"
        "Answer with only the integer value."
    )


def make_counterfactual_pair(corrupted, rng=None):
    """Create the Stage G clean/corrupted pair without changing generator semantics.

    The clean member keeps the same line geometry and values, but stale writes
    of the queried variable are renamed to fresh same-length distractor names.
    The target variable therefore has only the final write in the clean prompt.
    """
    if corrupted.get("condition") != "interference":
        raise ValueError("counterfactual pairs require an interference example")
    if rng is None:
        rng = random.Random(0)

    assignments = _assignment_lines_from_prompt(corrupted["prompt"])
    if len(assignments) != corrupted["n_lines"]:
        raise ValueError(
            f"parsed {len(assignments)} assignments, expected {corrupted['n_lines']}"
        )

    target = corrupted["target_var"]
    write_positions = list(corrupted["target_write_positions"])
    if not write_positions:
        raise ValueError("missing target_write_positions")
    stale_positions = write_positions[:-1]
    current_position = write_positions[-1]

    existing = {name for name, _ in assignments}
    clean_assignments = list(assignments)
    replacements = {}
    for pos in stale_positions:
        old_name, value = assignments[pos]
        if old_name != target:
            raise ValueError(f"expected target var at stale position {pos}, found {old_name}")
        new_name = _fresh_same_length_name(existing, rng, len(target))
        clean_assignments[pos] = (new_name, value)
        replacements[pos] = new_name

    final_name, final_value = assignments[current_position]
    if final_name != target or int(final_value) != int(corrupted["gold"]):
        raise ValueError("final target write does not match gold")

    span_equal = all(
        len(assignments[pos][0]) == len(clean_assignments[pos][0])
        and assignments[pos][1] == clean_assignments[pos][1]
        for pos in write_positions
    )
    if not span_equal:
        raise ValueError("counterfactual pair failed same-length/value geometry check")

    clean = dict(corrupted)
    clean["condition"] = "counterfactual_clean"
    clean["prompt"] = _prompt_from_assignments(clean_assignments, target)
    clean["target_write_positions"] = [current_position]
    clean["stale_values"] = []
    clean["counterfactual_from_id"] = corrupted.get("id")
    clean["counterfactual_stale_write_positions"] = stale_positions
    clean["counterfactual_all_write_positions"] = write_positions
    clean["counterfactual_replacements"] = replacements

    return {
        "corrupted": dict(corrupted),
        "clean": clean,
        "corrupted_assignments": assignments,
        "clean_assignments": clean_assignments,
        "span_equal": span_equal,
        "replacements": replacements,
    }


def build_dataset(out_path, n_per_cell=100, seed=0):
    rng = random.Random(seed)
    n_lines_grid = [20, 40, 80]
    I_grid = [2, 4, 8]
    rows = []
    uid = 0
    for n_lines in n_lines_grid:
        # simple baseline at each length
        for _ in range(n_per_cell):
            ex = make_example("simple", n_lines, 0, rng)
            ex["id"] = uid
            uid += 1
            rows.append(ex)
        # interference at each load (skip loads that don't fit this length)
        for I in I_grid:
            if n_lines < I + 1 + 3:
                continue
            for _ in range(n_per_cell):
                ex = make_example("interference", n_lines, I, rng)
                ex["id"] = uid
                uid += 1
                rows.append(ex)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/p0.jsonl")
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = build_dataset(args.out, args.n_per_cell, args.seed)
    print(f"wrote {n} examples to {args.out}")
