"""Stage N few-shot latest-binding task generation for base Pythia models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


MODEL_DEFAULT = "EleutherAI/pythia-160m"
TEMPLATES = ("arrow", "current", "latest")
VAR_CANDIDATES = (
    "alpha", "beta", "gamma", "delta", "theta", "sigma", "omega", "vector",
    "color", "music", "route", "style", "topic", "mode", "level", "shape",
    "fruit", "drink", "sport", "animal", "city", "movie", "book", "game",
)


def dump_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def one_token_strings(tokenizer, candidates: Iterable[str]) -> list[str]:
    out = []
    for value in candidates:
        ids = tokenizer.encode(" " + value, add_special_tokens=False)
        if len(ids) == 1 and tokenizer.decode(ids).strip() == value:
            out.append(value)
    return out


def value_token_id(tokenizer, value: str) -> int:
    ids = tokenizer.encode(" " + str(value), add_special_tokens=False)
    if len(ids) != 1 or tokenizer.decode(ids).strip() != str(value):
        raise ValueError(f"value is not one canonical leading-space token: {value!r} -> {ids}")
    return int(ids[0])


def _fixed_examples(template: str) -> list[tuple[list[tuple[str, str]], str, str]]:
    examples = [
        ([('color', '6'), ('color', '3')], 'color', '3'),
        ([('music', '8'), ('route', '4'), ('music', '1')], 'music', '1'),
        ([('shape', '7'), ('topic', '2'), ('mode', '5')], 'topic', '2'),
        ([('drink', '9'), ('drink', '0'), ('sport', '4'), ('drink', '5')], 'drink', '5'),
    ]
    if template not in TEMPLATES:
        raise ValueError(template)
    return examples


def _render_block(events: list[dict] | list[tuple[str, str]], target: str, answer: str | None, template: str) -> str:
    assignments = [(e["var"], e["value"]) if isinstance(e, dict) else e for e in events]
    body = "\n".join(f"{var} = {value}" for var, value in assignments)
    if template == "arrow":
        suffix = f"=> {target} =" + (f" {answer}" if answer is not None else "")
        return f"Assignments:\n{body}\n{suffix}"
    if template == "current":
        suffix = f"Current {target} =" + (f" {answer}" if answer is not None else "")
        return f"Record:\n{body}\n{suffix}"
    if template == "latest":
        suffix = f"Latest value of {target} =" + (f" {answer}" if answer is not None else "")
        return f"Updates in time order:\n{body}\n{suffix}"
    raise ValueError(template)


def render_prompt(events: list[dict], target: str, template: str) -> tuple[str, int]:
    exemplars = [
        _render_block(example_events, example_target, answer, template)
        for example_events, example_target, answer in _fixed_examples(template)
    ]
    prefix = "\n\n".join(exemplars) + "\n\n"
    task_start = len(prefix)
    return prefix + _render_block(events, target, None, template), task_start


def _char_to_token(offsets, start: int, end: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end]


def annotate_spans(row: dict, tokenizer) -> dict:
    prompt = row["prompt"]
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    cursor = int(row["task_char_start"])
    writes = []
    for event in row["events"]:
        needle = f"{event['var']} = {event['value']}"
        line_start = prompt.find(needle, cursor)
        if line_start < 0:
            raise ValueError(f"assignment missing from prompt: {needle}")
        var_start = line_start
        var_end = var_start + len(event["var"])
        value_start = line_start + len(event["var"]) + 3
        value_end = value_start + len(event["value"])
        var_tokens = _char_to_token(offsets, var_start, var_end)
        value_tokens = _char_to_token(offsets, value_start, value_end)
        if len(var_tokens) != 1 or len(value_tokens) != 1:
            raise ValueError(
                f"non-single-token write {needle}: var={var_tokens}, value={value_tokens}"
            )
        writes.append(
            {
                **event,
                "var_token": int(var_tokens[0]),
                "value_token": int(value_tokens[0]),
                "var_token_id": int(enc["input_ids"][var_tokens[0]]),
                "value_token_id": int(enc["input_ids"][value_tokens[0]]),
            }
        )
        cursor = value_end
    row = {**row, "writes": writes, "prompt_tokens": len(enc["input_ids"])}
    row["current_value_span"] = [
        w["value_token"] for w in writes if w["var"] == row["target_var"] and w["is_current"]
    ]
    row["stale_value_spans"] = [
        w["value_token"] for w in writes if w["var"] == row["target_var"] and not w["is_current"]
    ]
    row["cross_value_spans"] = [w["value_token"] for w in writes if w["var"] != row["target_var"]]
    row["target_identity_spans"] = [w["var_token"] for w in writes if w["var"] == row["target_var"]]
    row["all_identity_spans"] = [w["var_token"] for w in writes]
    if len(row["current_value_span"]) != 1:
        raise ValueError(f"expected one current span: {row['id']}")
    return row


def _semantic_events(variant: str, k: int, rng: random.Random, variables: list[str], values: list[str], n_lines: int) -> tuple[list[dict], str, str]:
    if n_lines < k + 4:
        raise ValueError("n_lines must leave room for post-current distractors")
    chosen_vars = rng.sample(variables, n_lines + 1)
    target = chosen_vars[0]
    chosen_values = rng.sample(values, n_lines + 6)
    target_values = chosen_values[: k + 1]
    remaining_values = iter(chosen_values[k + 1 :])

    if variant == "single":
        events = [
            {
                "position": position,
                "var": target,
                "value": value,
                "is_target": True,
                "is_current": position == k,
            }
            for position, value in enumerate(target_values)
        ]
        return events, target, target_values[-1]

    last_allowed = n_lines - 3
    target_positions = sorted(rng.sample(range(last_allowed + 1), k + 1))
    events: list[dict | None] = [None] * n_lines
    for write_idx, (position, value) in enumerate(zip(target_positions, target_values)):
        events[position] = {
            "position": position,
            "var": target,
            "value": value,
            "is_target": True,
            "is_current": write_idx == k,
        }

    open_positions = [i for i, event in enumerate(events) if event is None]
    if variant == "multi":
        distractor_vars = chosen_vars[1:4]
        per_var_seen: dict[str, int] = {var: 0 for var in distractor_vars}
        for index, position in enumerate(open_positions):
            var = distractor_vars[index % len(distractor_vars)]
            per_var_seen[var] += 1
            events[position] = {
                "position": position,
                "var": var,
                "value": next(remaining_values),
                "is_target": False,
                "is_current": False,
            }
        for var in distractor_vars:
            var_rows = [event for event in events if event is not None and event["var"] == var]
            if var_rows:
                var_rows[-1]["is_current"] = True
    else:
        raise ValueError(variant)

    concrete = [event for event in events if event is not None]
    return concrete, target, target_values[-1]


def make_row(
    *,
    variant: str,
    template: str,
    k: int,
    seed: int,
    index: int,
    tokenizer,
    n_lines: int = 12,
    variables: list[str] | None = None,
    values: list[str] | None = None,
) -> dict:
    # Template changes only the surface form; semantic_id therefore denotes the
    # same assignment sequence across all three formats.
    rng = random.Random(seed * 1_000_003 + k * 10_007 + index * 101 + (0 if variant == "single" else 1))
    variables = list(variables) if variables is not None else one_token_strings(tokenizer, VAR_CANDIDATES)
    values = list(values) if values is not None else one_token_strings(tokenizer, (str(i) for i in range(10, 40)))
    reserved_vars = {v for events, _, _ in _fixed_examples(template) for v, _ in events}
    reserved_values = {x for events, _, _ in _fixed_examples(template) for _, x in events}
    variables = [v for v in variables if v not in reserved_vars]
    values = [v for v in values if v not in reserved_values]
    if len(variables) < n_lines + 1 or len(values) < n_lines + 6:
        raise ValueError("insufficient one-token vocabulary")
    events, target, gold = _semantic_events(variant, k, rng, variables, values, n_lines)
    prompt, task_char_start = render_prompt(events, target, template)
    target_writes = [event for event in events if event["var"] == target]
    stale = [event["value"] for event in target_writes if not event["is_current"]]
    cross = [event["value"] for event in events if event["var"] != target]
    row = {
        "id": f"n_{variant}_{template}_k{k}_s{seed}_{index:04d}",
        "semantic_id": f"n_{variant}_k{k}_s{seed}_{index:04d}",
        "variant": variant,
        "template": template,
        "k": k,
        "seed": seed,
        "index": index,
        "n_lines": n_lines,
        "target_var": target,
        "gold": gold,
        "gold_token_id": value_token_id(tokenizer, gold),
        "stale_values": stale,
        "stale_token_ids": [value_token_id(tokenizer, value) for value in stale],
        "cross_values": cross,
        "cross_token_ids": [value_token_id(tokenizer, value) for value in cross],
        "events": events,
        "prompt": prompt,
        "task_char_start": task_char_start,
    }
    return annotate_spans(row, tokenizer)


def make_clean_counterfactual(row: dict, tokenizer) -> dict:
    used_vars = {event["var"] for event in row["events"]}
    candidates = [v for v in one_token_strings(tokenizer, VAR_CANDIDATES) if v not in used_vars]
    stale_events = [e for e in row["events"] if e["var"] == row["target_var"] and not e["is_current"]]
    if len(candidates) < len(stale_events):
        raise ValueError("not enough foil variables")
    foils = iter(candidates)
    clean_events = []
    for event in row["events"]:
        copied = dict(event)
        if copied["var"] == row["target_var"] and not copied["is_current"]:
            copied["var"] = next(foils)
            copied["is_target"] = False
            copied["is_current"] = True
        clean_events.append(copied)
    prompt, task_char_start = render_prompt(clean_events, row["target_var"], row["template"])
    clean = {
        **{k: v for k, v in row.items() if k not in {"writes", "prompt", "events"}},
        "id": row["id"] + "__clean",
        "pair_id": row["id"],
        "counterfactual_clean": True,
        "events": clean_events,
        "prompt": prompt,
        "task_char_start": task_char_start,
        "stale_values": [],
        "stale_token_ids": [],
        "cross_values": [e["value"] for e in clean_events if e["var"] != row["target_var"]],
        "cross_token_ids": [value_token_id(tokenizer, e["value"]) for e in clean_events if e["var"] != row["target_var"]],
    }
    clean = annotate_spans(clean, tokenizer)
    corrupt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    clean_ids = tokenizer(clean["prompt"], add_special_tokens=False)["input_ids"]
    if len(corrupt_ids) != len(clean_ids):
        raise ValueError(f"counterfactual token length mismatch: {len(corrupt_ids)} vs {len(clean_ids)}")
    return clean


def validate_row(row: dict, tokenizer) -> None:
    assert row["prompt"].endswith("=")
    expected_stale = 0 if row.get("counterfactual_clean") else row["k"]
    assert len(row["stale_values"]) == expected_stale
    assert row["gold"] not in row["stale_values"]
    assert not (set(row["stale_values"]) & set(row["cross_values"]))
    assert row["gold"] not in row["cross_values"]
    assert len(row["current_value_span"]) == 1
    for write in row["writes"]:
        assert write["value_token_id"] == value_token_id(tokenizer, write["value"])


def generate(args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    variables = one_token_strings(tokenizer, VAR_CANDIDATES)
    values = one_token_strings(tokenizer, (str(i) for i in range(10, 40)))
    rows = []
    for seed in args.seeds:
        for k in args.k_values:
            for variant in args.variants:
                for template in args.templates:
                    for index in range(args.n_per_seed_cell):
                        row = make_row(
                            variant=variant,
                            template=template,
                            k=k,
                            seed=seed,
                            index=index,
                            tokenizer=tokenizer,
                            n_lines=args.n_lines,
                            variables=variables,
                            values=values,
                        )
                        validate_row(row, tokenizer)
                        rows.append(row)
    dump_jsonl(args.out, rows)
    summary = {
        "model": args.model,
        "n": len(rows),
        "seeds": args.seeds,
        "k_values": args.k_values,
        "variants": args.variants,
        "templates": args.templates,
        "n_per_seed_cell": args.n_per_seed_cell,
        "n_per_aggregated_cell": args.n_per_seed_cell * len(args.seeds),
        "single_token_values": sorted({row["gold"] for row in rows}),
        "prompt_token_range": [min(r["prompt_tokens"] for r in rows), max(r["prompt_tokens"] for r in rows)],
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def self_test(args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    for template in TEMPLATES:
        for variant in ("single", "multi"):
            for k in (0, 2, 5):
                row = make_row(variant=variant, template=template, k=k, seed=0, index=0, tokenizer=tokenizer)
                validate_row(row, tokenizer)
                if k:
                    clean = make_clean_counterfactual(row, tokenizer)
                    validate_row(clean, tokenizer)
    print("pythia_gen self-test: PASS")


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_csv(value: str) -> list[str]:
    return [part for part in value.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    test = sub.add_parser("self-test")
    test.add_argument("--model", default=MODEL_DEFAULT)
    test.set_defaults(func=self_test)
    gen = sub.add_parser("generate")
    gen.add_argument("--model", default=MODEL_DEFAULT)
    gen.add_argument("--out", default="results/stage_n/behavior/tasks.jsonl")
    gen.add_argument("--summary", default="results/stage_n/behavior/tasks.summary.json")
    gen.add_argument("--seeds", type=parse_csv_ints, default=[11, 29, 47])
    gen.add_argument("--k-values", type=parse_csv_ints, default=[0, 1, 2, 3, 4, 5])
    gen.add_argument("--variants", type=parse_csv, default=["single", "multi"])
    gen.add_argument("--templates", type=parse_csv, default=list(TEMPLATES))
    gen.add_argument("--n-per-seed-cell", type=int, default=72)
    gen.add_argument("--n-lines", type=int, default=12)
    gen.set_defaults(func=generate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
