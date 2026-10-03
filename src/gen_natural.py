"""Naturalistic stale-binding generator for Stage E.

The rows keep the synthetic stale-binding contract while changing the surface
format to dialogue, agent-log, and document scenarios. Values are distinct
multi-digit integers so stale-error attribution remains unambiguous.
"""

import argparse
import collections
import json
import random
from pathlib import Path


FAMILIES = ("dialogue", "agent_log", "document")
LOADS = (1, 2, 4, 8)
LENGTH_VARIANTS = ("short", "long")


def _rand_values(rng, n, lo=1000, hi=9999):
    values = set()
    while len(values) < n:
        values.add(rng.randint(lo, hi))
    return list(values)


def _target_for_family(family, rng):
    if family == "dialogue":
        return rng.choice(["city code", "travel date code", "ticket quantity"])
    if family == "agent_log":
        return rng.choice(["db.timeout", "cache.limit", "retry.window"])
    if family == "document":
        return rng.choice(["quarterly estimate", "shipment count", "risk score"])
    raise ValueError(family)


def _distractor_names(family):
    if family == "dialogue":
        return ["meal code", "seat row", "hotel rating", "loyalty number", "bag count"]
    if family == "agent_log":
        return ["ui.theme", "worker.batch", "api.retries", "log.level", "queue.depth"]
    if family == "document":
        return ["regional total", "backup forecast", "audit sample", "legacy figure", "appendix value"]
    raise ValueError(family)


def _target_line(family, target, value, write_idx, is_current):
    if family == "dialogue":
        if write_idx == 0:
            return f"User: Please set the {target} to {value} for this booking."
        return f"User: Correction, revise the {target} to {value}; use this newer value."
    if family == "agent_log":
        action = "set" if write_idx == 0 else "update"
        return f"tool.log: {action} config {target} to {value}"
    if family == "document":
        if write_idx == 0:
            return f"The report initially listed the {target} as {value}."
        return f"Erratum {write_idx}: the {target} was later revised to {value}."
    raise ValueError(family)


def _distractor_line(family, name, value, idx):
    if family == "dialogue":
        speaker = "Assistant" if idx % 2 else "User"
        return f"{speaker}: Separately, record the {name} as {value}."
    if family == "agent_log":
        return f"tool.log: set config {name} to {value}"
    if family == "document":
        return f"Background note {idx}: the {name} remained {value}."
    raise ValueError(family)


def _tail_line(family, idx):
    if family == "dialogue":
        return [
            "Assistant: I will keep the latest revision as authoritative.",
            "User: Ignore any older notes if they conflict.",
            "Assistant: The booking record now has several unrelated details.",
            "User: Please answer from the final state only.",
            "Assistant: Understood; earlier revisions are stale.",
        ][idx % 5]
    if family == "agent_log":
        return [
            "tool.log: health check completed without changing the target key",
            "tool.log: read-only audit snapshot created",
            "tool.log: unrelated cleanup task finished",
            "tool.log: final query will inspect current config state",
            "tool.log: no further writes to the target key occurred",
        ][idx % 5]
    if family == "document":
        return [
            "A later paragraph discussed methodology without changing the figure.",
            "The appendix added context but no new value for the target figure.",
            "Reviewers were told to use the latest erratum when conflicts appear.",
            "No subsequent sentence revised the target figure again.",
            "The closing note summarized unrelated assumptions.",
        ][idx % 5]
    raise ValueError(family)


def _question(family, target):
    if family == "dialogue":
        return f"Question: What is the final integer value for the {target}?"
    if family == "agent_log":
        return f"Question: What is the current integer value of config key {target}?"
    if family == "document":
        return f"Question: What is the latest integer value of the {target}?"
    raise ValueError(family)


def _char_spans(prompt, values):
    spans = []
    cursor = 0
    for value in values:
        text = str(value)
        start = prompt.find(text, cursor)
        if start < 0:
            raise ValueError(f"value {value} not found after offset {cursor}")
        end = start + len(text)
        spans.append({"char_start": start, "char_end": end, "value": value})
        cursor = end
    return spans


def make_natural_example(family, condition, load, length_variant, rng, tail_min=3):
    if condition not in {"simple", "interference"}:
        raise ValueError(condition)
    if condition == "simple" and load != 0:
        raise ValueError("simple rows must use load=0")

    target = _target_for_family(family, rng)
    n_stale = 0 if condition == "simple" else load
    n_target_values = n_stale + 1
    n_distractors = 5 if length_variant == "short" else 13
    tail_units = tail_min if length_variant == "short" else tail_min + 3
    values = _rand_values(rng, n_target_values + n_distractors)
    target_values = values[:n_target_values]
    distractor_values = values[n_target_values:]
    stale_values = target_values[:-1]
    gold = target_values[-1]

    distractor_names = _distractor_names(family)
    distractors = [
        _distractor_line(family, distractor_names[i % len(distractor_names)], value, i)
        for i, value in enumerate(distractor_values)
    ]
    target_lines = [
        _target_line(family, target, value, idx, idx == len(target_values) - 1)
        for idx, value in enumerate(target_values)
    ]

    prefix_count = 1 if length_variant == "short" else 4
    middle_distractors = distractors[: max(0, n_distractors - tail_units - prefix_count)]
    prefix_distractors = distractors[max(0, n_distractors - tail_units - prefix_count) : max(0, n_distractors - tail_units)]
    tail_distractors = distractors[max(0, n_distractors - tail_units) :]

    context_lines = []
    context_lines.extend(prefix_distractors[:prefix_count])
    target_update_positions = []
    for idx, target_line in enumerate(target_lines):
        context_lines.append(target_line)
        target_update_positions.append(len(context_lines) - 1)
        if idx < len(target_lines) - 1 and middle_distractors:
            context_lines.append(middle_distractors.pop(0))
    context_lines.extend(middle_distractors)
    context_lines.extend(tail_distractors)
    while len(context_lines) - 1 - target_update_positions[-1] < tail_min:
        context_lines.append(_tail_line(family, len(context_lines)))

    intro = {
        "dialogue": "Read the transcript. Later user revisions override earlier ones.",
        "agent_log": "Read the tool log in order. Later writes override earlier writes.",
        "document": "Read the report. Later errata override earlier stated figures.",
    }[family]
    question = _question(family, target)
    prompt = (
        f"{intro}\n\n"
        + "\n".join(context_lines)
        + "\n\n"
        + question
        + "\nAnswer with only the integer value."
    )

    spans = _char_spans(prompt, target_values)
    all_values = list(target_values) + list(distractor_values)
    return {
        "id": None,
        "task_family": "naturalistic",
        "stage_d_cell": family,
        "natural_family": family,
        "condition": condition,
        "length_variant": length_variant,
        "n_lines": len(context_lines),
        "interference_load": load,
        "target": target,
        "target_var": target,
        "gold": gold,
        "prompt": prompt,
        "question": question,
        "answer_type": "int",
        "stale_values": stale_values,
        "all_values": all_values,
        "context_lines": context_lines,
        "target_update_positions": target_update_positions,
        "target_write_positions": target_update_positions,
        "target_update_char_spans": spans,
        "current_spans": [spans[-1]],
        "stale_spans": spans[:-1],
        "tail_units_after_final_update": len(context_lines) - 1 - target_update_positions[-1],
        "metadata": {
            "generator": "gen_natural.py",
            "family": family,
            "load": load,
            "length_variant": length_variant,
        },
    }


def make_natural_rows(n_per_cell=100, seed=20):
    rng = random.Random(seed)
    rows = []
    uid = 0
    for family in FAMILIES:
        for length in LENGTH_VARIANTS:
            for _ in range(n_per_cell):
                row = make_natural_example(family, "simple", 0, length, rng)
                row["id"] = f"natural-{uid:05d}"
                uid += 1
                rows.append(row)
            for load in LOADS:
                for _ in range(n_per_cell):
                    row = make_natural_example(family, "interference", load, length, rng)
                    row["id"] = f"natural-{uid:05d}"
                    uid += 1
                    rows.append(row)
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def summarize(rows):
    by_family = collections.Counter(row["natural_family"] for row in rows)
    by_cell = collections.Counter(
        (row["natural_family"], row["condition"], row["interference_load"], row["length_variant"])
        for row in rows
    )
    return {
        "n_total": len(rows),
        "by_family": dict(sorted(by_family.items())),
        "by_cell": {
            "|".join(map(str, key)): value for key, value in sorted(by_cell.items())
        },
    }


def write_natural_outputs(out_dir, n_per_cell=100, seed=20):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = make_natural_rows(n_per_cell=n_per_cell, seed=seed)
    write_jsonl(out / "eval_all.jsonl", rows)
    family_paths = {}
    for family in FAMILIES:
        path = out / f"eval_{family}.jsonl"
        write_jsonl(path, [row for row in rows if row["natural_family"] == family])
        family_paths[family] = str(path)
    manifest = {
        "stage": "E",
        "track": "naturalistic",
        "description": "Natural-language stale-binding suite with dialogue, agent-log, and document families.",
        "seed": seed,
        "n_per_cell": n_per_cell,
        "eval_all_path": str(out / "eval_all.jsonl"),
        "family_paths": family_paths,
        **summarize(rows),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/natural_transfer/natural")
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--print-examples", type=int, default=0)
    args = ap.parse_args()

    manifest = write_natural_outputs(args.out_dir, args.n_per_cell, args.seed)
    print(json.dumps(manifest, indent=2))
    if args.print_examples:
        rows = make_natural_rows(n_per_cell=1, seed=args.seed)
        for family in FAMILIES:
            print(f"\n=== {family} examples ===")
            for row in [r for r in rows if r["natural_family"] == family][: args.print_examples]:
                print(row["prompt"])
                print(f"gold={row['gold']} stale={row['stale_values']}")


if __name__ == "__main__":
    main()
