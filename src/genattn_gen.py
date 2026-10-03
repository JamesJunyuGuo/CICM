"""Generate Stage P event-aligned diagnostic-query branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


CHECKPOINTS = ("tau1", "tau2", "tau3", "tau4", "tauQ")
TEMPLATES = ("direct", "profile", "timeline")
SLOTS = ("color", "music", "route", "style", "topic", "level", "shape", "fruit")
VALUE_REGIMES = {
    "pythia_numeric": tuple(str(value) for value in range(20, 28)),
    "crossmodel_words": ("red", "blue", "green", "black", "white", "silver", "gold", "pink"),
}
QUERY = "What is the latest value of the {slot} preference?\nLatest value of {slot} ="


def dump_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PromptBuilder:
    def __init__(self, prefix: str = "") -> None:
        self.parts = [prefix]
        self.length = len(prefix)
        self.markers: list[dict] = []

    def add(self, text: str, marks: list[tuple[str, int, int, dict]] = ()) -> None:
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        for kind, left, right, metadata in marks:
            self.markers.append({
                "kind": kind,
                "char_start": start + left,
                "char_end": start + right,
                **metadata,
            })

    def build(self) -> tuple[str, list[dict]]:
        return "".join(self.parts), list(self.markers)


def _write_line(template: str, slot: str, value: str, update: bool) -> tuple[str, list[tuple[str, int, int, dict]]]:
    if template == "direct":
        prefix, middle, suffix = "", " = ", "\n"
    elif template == "profile":
        prefix, middle, suffix = "Preference ", " = ", "\n"
    elif template == "timeline":
        prefix, middle, suffix = "Update: ", " = ", "\n"
    else:
        raise ValueError(template)
    slot_start = len(prefix)
    value_start = slot_start + len(slot) + len(middle)
    text = prefix + slot + middle + value + suffix
    return text, [
        ("identity", slot_start, slot_start + len(slot), {"slot": slot}),
        ("value", value_start, value_start + len(value), {"slot": slot, "value": value}),
    ]


def _reminder_line(template: str, slot: str, value: str, index: int) -> tuple[str, list[tuple[str, int, int, dict]]]:
    if template == "direct":
        prefix, middle, suffix = f"Historical mention {index}: ", " was ", "\n"
    elif template == "profile":
        prefix, middle, suffix = f"Earlier profile note {index}: ", " was ", "\n"
    elif template == "timeline":
        prefix, middle, suffix = f"Past timeline note {index}: ", " was ", "\n"
    else:
        raise ValueError(template)
    slot_start = len(prefix)
    value_start = slot_start + len(slot) + len(middle)
    text = prefix + slot + middle + value + suffix
    return text, [
        ("identity", slot_start, slot_start + len(slot), {"slot": slot}),
        ("value", value_start, value_start + len(value), {"slot": slot, "value": value}),
    ]


def _filler_line(template: str, slot: str, index: int) -> tuple[str, list[tuple[str, int, int, dict]]]:
    if template == "direct":
        prefix, suffix = f"No assignment in note {index} for ", "\n"
    elif template == "profile":
        prefix, suffix = f"No profile change in note {index} for ", "\n"
    elif template == "timeline":
        prefix, suffix = f"No timeline update in note {index} for ", "\n"
    else:
        raise ValueError(template)
    slot_start = len(prefix)
    text = prefix + slot + suffix
    return text, [("identity", slot_start, slot_start + len(slot), {"slot": slot})]


def _query_line(slot: str) -> tuple[str, list[tuple[str, int, int, dict]]]:
    first_prefix = "What is the latest value of the "
    middle = " preference?\nLatest value of "
    suffix = " ="
    text = first_prefix + slot + middle + slot + suffix
    first_start = len(first_prefix)
    second_start = first_start + len(slot) + len(middle)
    return text, [
        ("query_anchor", first_start, first_start + len(slot), {"slot": slot}),
        ("query_anchor", second_start, second_start + len(slot), {"slot": slot}),
    ]


def _few_shot_prefix(template: str, value_regime: str) -> str:
    examples = (
        [
            ("color", "10", "13", "music", "16"),
            ("music", "11", "14", "route", "17"),
            ("route", "12", "15", "style", "18"),
        ]
        if value_regime == "pythia_numeric"
        else [
            ("color", "north", "south", "music", "warm"),
            ("music", "east", "west", "route", "cool"),
            ("route", "high", "low", "style", "quiet"),
        ]
    )
    blocks = []
    for slot, old, current, other_slot, _other_value in examples:
        first, _ = _write_line(template, slot, old, False)
        update, _ = _write_line(template, slot, current, True)
        distractor, _ = _filler_line(template, other_slot, 1)
        reminder, _ = _reminder_line(template, slot, old, 1)
        blocks.append(first + update + distractor + reminder + QUERY.format(slot=slot) + f" {current}\n")
    instruction = (
        "Read assignments in time order. Historical mentions and no-assignment notes are not updates. "
        "Return the latest assigned value for the requested name.\n"
    )
    return instruction + "Examples:\n" + "\n".join(blocks) + "\nNow answer the next record.\n"


def _branch(
    *, template: str, target: str, foil: str, old: str, current: str,
    distractors: list[tuple[str, str]], checkpoint: str, control: bool,
    reminder_count: int, value_regime: str,
) -> dict:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(checkpoint)
    builder = PromptBuilder(_few_shot_prefix(template, value_regime))
    checkpoint_index = CHECKPOINTS.index(checkpoint)
    first_slot = foil if control and checkpoint_index >= 1 else target
    text, marks = _write_line(template, first_slot, old, False)
    builder.add(text, marks)
    if checkpoint_index >= 1:
        text, marks = _write_line(template, target, current, True)
        builder.add(text, marks)
    if checkpoint_index >= 2:
        for filler_index, (slot, _value) in enumerate(distractors[:4], 1):
            text, marks = _filler_line(template, slot, filler_index)
            builder.add(text, marks)
    if checkpoint_index >= 3:
        reminder_slot = foil if control else target
        visible_reminders = 1 if checkpoint == "tau4" else reminder_count
        for reminder_index in range(1, visible_reminders + 1):
            text, marks = _reminder_line(template, reminder_slot, old, reminder_index)
            builder.add(text, marks)
    if checkpoint_index >= 4:
        for filler_index, (slot, _value) in enumerate(distractors[4:8], 5):
            text, marks = _filler_line(template, slot, filler_index)
            builder.add(text, marks)
    query_text, query_marks = _query_line(target)
    builder.add(query_text, query_marks)
    prompt, markers = builder.build()
    active_current = old if checkpoint == "tau1" else current
    for marker in markers:
        if marker["kind"] == "value":
            marker["role"] = (
                "current" if marker["slot"] == target and marker["value"] == active_current
                else "stale" if marker["slot"] == target and marker["value"] == old and checkpoint != "tau1"
                else "matched_comparator" if control and marker["slot"] == foil and marker["value"] == old and checkpoint != "tau1"
                else "other"
            )
        elif marker["kind"] == "identity":
            marker["role"] = "target_identity" if marker["slot"] == target else "other_identity"
    stale_count = sum(marker.get("role") == "stale" for marker in markers)
    return {
        "checkpoint": checkpoint,
        "control": control,
        "prompt": prompt,
        "markers": markers,
        "gold": active_current,
        "stale_values": [] if checkpoint == "tau1" or control else [old],
        "comparison_values": [] if checkpoint == "tau1" else [old],
        "stale_trace_count": stale_count,
        "query_text": query_text,
    }


def make_item(seed: int, index: int, template: str, reminder_count: int, value_regime: str = "crossmodel_words") -> dict:
    rng = random.Random(seed * 1_000_003 + index * 101)
    target, foil, *remaining_slots = rng.sample(list(SLOTS), len(SLOTS))
    other_slots = (remaining_slots * 2)[:8]
    values = VALUE_REGIMES[value_regime]
    current = values[(seed + index) % len(values)]
    old = rng.choice([value for value in values if value != current])
    other_values = [values[(seed + index + offset + 1) % len(values)] for offset in range(8)]
    distractors = list(zip(other_slots[:8], other_values[:8]))
    branches = []
    for checkpoint in CHECKPOINTS:
        branches.append(_branch(
            template=template, target=target, foil=foil, old=old, current=current,
            distractors=distractors, checkpoint=checkpoint, control=False,
            reminder_count=reminder_count, value_regime=value_regime,
        ))
        branches.append(_branch(
            template=template, target=target, foil=foil, old=old, current=current,
            distractors=distractors, checkpoint=checkpoint, control=True,
            reminder_count=reminder_count, value_regime=value_regime,
        ))
    semantic_id = f"p_s{seed}_{index:04d}"
    return {
        "id": f"{semantic_id}_{template}",
        "semantic_id": semantic_id,
        "seed": seed,
        "index": index,
        "template": template,
        "target_slot": target,
        "foil_slot": foil,
        "old_value": old,
        "current_value": current,
        "distractors": [{"slot": slot, "value": value} for slot, value in distractors],
        "reminder_count": reminder_count,
        "value_regime": value_regime,
        "branches": branches,
    }


def validate_item(item: dict) -> None:
    branches = {(row["checkpoint"], row["control"]): row for row in item["branches"]}
    if set(branches) != {(tau, control) for tau in CHECKPOINTS for control in (False, True)}:
        raise ValueError("missing checkpoint/control branch")
    for checkpoint in CHECKPOINTS:
        corrupt = branches[(checkpoint, False)]
        control = branches[(checkpoint, True)]
        if len(corrupt["prompt"]) != len(control["prompt"]):
            raise ValueError(f"character length mismatch at {checkpoint}")
        if corrupt["query_text"] != control["query_text"]:
            raise ValueError(f"query mismatch at {checkpoint}")
        for branch in (corrupt, control):
            for marker in branch["markers"]:
                observed = branch["prompt"][marker["char_start"]:marker["char_end"]]
                expected = marker.get("value", marker.get("slot"))
                if observed != expected:
                    raise ValueError(f"marker mismatch: {observed!r} != {expected!r}")
        if checkpoint != "tau1" and control["stale_trace_count"] != 0:
            raise ValueError("no-overwrite control contains a target stale trace")
    queries = {branches[(tau, False)]["query_text"] for tau in CHECKPOINTS}
    if len(queries) != 1:
        raise ValueError("diagnostic query changed across checkpoints")


def generate(args) -> None:
    rows = []
    for seed in args.seeds:
        for index in range(args.n_per_seed):
            for template in args.templates:
                row = make_item(seed, index, template, args.reminder_count, args.value_regime)
                validate_item(row)
                rows.append(row)
    dump_jsonl(args.out, rows)
    digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    lengths = [len(branch["prompt"]) for row in rows for branch in row["branches"]]
    summary = {
        "stage": "P-generation",
        "n_items": len(rows),
        "n_branches": len(rows) * len(CHECKPOINTS) * 2,
        "seeds": args.seeds,
        "templates": args.templates,
        "n_per_seed": args.n_per_seed,
        "checkpoints": list(CHECKPOINTS),
        "reminder_count": args.reminder_count,
        "value_regime": args.value_regime,
        "stale_trace_counts_by_checkpoint": {
            checkpoint: next(
                branch["stale_trace_count"]
                for branch in rows[0]["branches"]
                if branch["checkpoint"] == checkpoint and not branch["control"]
            )
            for checkpoint in CHECKPOINTS
        },
        "value_counts": dict(Counter(row["current_value"] for row in rows)),
        "prompt_character_range": [min(lengths), max(lengths)],
        "fixed_query_protocol": "same formatted diagnostic query on every independent branch",
        "control_protocol": "same-character-length target-to-foil replacement for superseded traces",
        "data_sha256": digest,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def self_test(_args) -> None:
    for value_regime in VALUE_REGIMES:
        for template in TEMPLATES:
            row = make_item(11, 0, template, 3, value_regime)
            validate_item(row)
    print("genattn_gen self-test: PASS")


def parse_csv(value: str) -> list[str]:
    return [part for part in value.split(",") if part]


def parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(func=self_test)
    gen = sub.add_parser("generate")
    gen.add_argument("--out", default="results/stage_p/data/event_items.jsonl")
    gen.add_argument("--summary", default="results/stage_p/data/event_items.summary.json")
    gen.add_argument("--seeds", type=parse_int_csv, default=[11, 29, 47])
    gen.add_argument("--templates", type=parse_csv, default=list(TEMPLATES))
    gen.add_argument("--n-per-seed", type=int, default=24)
    gen.add_argument("--reminder-count", type=int, default=3)
    gen.add_argument("--value-regime", choices=tuple(VALUE_REGIMES), default="crossmodel_words")
    gen.set_defaults(func=generate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
