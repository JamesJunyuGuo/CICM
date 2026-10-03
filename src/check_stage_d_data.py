"""Sanity checks for Stage D broader contextual-management data."""

import argparse
import collections
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "id",
    "task_family",
    "stage_d_cell",
    "condition",
    "prompt",
    "context_lines",
    "question",
    "gold",
    "answer_type",
    "current_spans",
    "stale_spans",
    "stale_values",
    "metadata",
}

EXPECTED_FAMILIES = {
    "ruler_style",
    "babilong_style",
    "michelangelo_style",
    "state_suite",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _span_errors(row, span_key):
    errors = []
    lines = row.get("context_lines")
    if not isinstance(lines, list) or not lines:
        return [f"id={row.get('id')} has invalid context_lines"]
    for idx, span in enumerate(row.get(span_key, [])):
        if not isinstance(span, dict):
            errors.append(f"id={row.get('id')} {span_key}[{idx}] is not an object")
            continue
        line_no = span.get("line")
        if not isinstance(line_no, int) or not 0 <= line_no < len(lines):
            errors.append(f"id={row.get('id')} {span_key}[{idx}] has invalid line={line_no}")
        if "value" not in span:
            errors.append(f"id={row.get('id')} {span_key}[{idx}] missing value")
    return errors


def check_rows(rows):
    errors = []
    seen_ids = set()
    for offset, row in enumerate(rows):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            errors.append(f"offset={offset} missing fields: {missing}")
            continue
        row_id = row["id"]
        if row_id in seen_ids:
            errors.append(f"duplicate id={row_id}")
        seen_ids.add(row_id)
        if row["task_family"] not in EXPECTED_FAMILIES:
            errors.append(f"id={row_id} unexpected task_family={row['task_family']}")
        if not row["current_spans"]:
            errors.append(f"id={row_id} has no current_spans")
        if row["answer_type"] not in {"int", "string", "list", "dict"}:
            errors.append(f"id={row_id} unexpected answer_type={row['answer_type']}")
        if "Question:" not in row["prompt"]:
            errors.append(f"id={row_id} prompt missing question")
        if not all(line in row["prompt"] for line in row["context_lines"]):
            errors.append(f"id={row_id} prompt does not contain all context lines")
        errors.extend(_span_errors(row, "current_spans"))
        errors.extend(_span_errors(row, "stale_spans"))
    families = {row.get("task_family") for row in rows}
    missing_families = sorted(EXPECTED_FAMILIES - families)
    if missing_families:
        errors.append(f"missing families: {missing_families}")
    return errors


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stage_d/eval_all.jsonl")
    ap.add_argument("--out", default="data/stage_d/check_summary.json")
    args = ap.parse_args()

    rows = load_jsonl(args.data)
    errors = check_rows(rows)
    summary = {
        "data": args.data,
        "pass": not errors,
        "errors": errors[:50],
        "n_errors": len(errors),
        **summarize(rows),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
