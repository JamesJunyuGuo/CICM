"""Stage R9-C: fixed stale-key routing transfer to natural CICM dialogue."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pythia_eval import dump_json
from pythia_gen import dump_jsonl, load_jsonl
from cicm_eval import (
    classify_stage_l_response,
    contains_value,
    load_model_and_tokenizer,
    render_prompt,
)
from stale_binding_attenuation import _random_summary, clustered_paired_bootstrap
from stale_binding_routing import (
    RoutingBiasContext,
    _char_span_to_tokens,
    _heads_by_layer,
    install_qwen2_routing_patch,
    routing_bias,
)


SLOT_SCHEMA = {
    "music_genre": {
        "label": "music genre",
        "values": ("jazz", "rock", "classical", "electronic", "folk", "hiphop", "blues"),
    },
    "diet": {
        "label": "meal style",
        "values": ("vegan", "vegetarian", "keto", "mediterranean", "paleo", "pescatarian", "gluten free"),
    },
    "learning_style": {
        "label": "learning style",
        "values": ("visual", "hands on", "lecture", "reading", "discussion", "self paced", "tutoring"),
    },
}

DECISION_CODES = ("amber", "birch", "coral", "denim", "ember", "frost", "grove")
DEFAULT_BETAS = (0.5, 1.0, 2.0, 4.0, 8.0)


def _value_matches(text: str, value: str):
    return re.finditer(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", text, re.IGNORECASE)


def _message_offsets(prompt: str, messages: list[dict]) -> list[int]:
    starts = []
    cursor = 0
    for index, message in enumerate(messages):
        content = message["content"]
        if index == 0 and message.get("role") == "system":
            # Some chat templates relocate the system text (Mistral places it
            # beside the final user turn). Record it without moving the cursor
            # used to recover the chronological conversation messages.
            starts.append(prompt.find(content))
            continue
        start = prompt.find(content, cursor)
        if start < 0:
            raise ValueError(f"message content is absent from rendered prompt: {content[:60]!r}")
        starts.append(start)
        cursor = start + len(content)
    return starts


def _merge_messages_for_restricted_chat_template(
    messages: list[dict], *, relocate_system: bool
) -> list[dict]:
    """Normalize only the role constraint rejected by a chat template."""
    output = copy.deepcopy(messages)
    system_content = None
    if relocate_system and output and output[0].get("role") == "system":
        system_content = output.pop(0)["content"]
    if system_content is not None:
        if not output or output[0].get("role") != "user":
            raise ValueError("cannot relocate system content without a leading user turn")
        output[0]["content"] = f"{system_content}\n\n{output[0]['content']}"

    merged = []
    for message in output:
        if merged and merged[-1].get("role") == message.get("role"):
            merged[-1]["content"] = (
                f"{merged[-1]['content']}\n\n{message['content']}"
            )
        else:
            merged.append(message)
    return merged


def render_natural_prompt(tokenizer, messages: list[dict]) -> str:
    """Render unchanged unless the tokenizer rejects system/adjacent roles."""
    try:
        return render_prompt(tokenizer, messages)
    except Exception as exc:
        error = str(exc).lower()
        supported_repair = (
            "system role not supported" in error
            or "conversation roles must alternate" in error
        )
        if not supported_repair:
            raise
        normalized = _merge_messages_for_restricted_chat_template(
            messages,
            relocate_system="system role not supported" in error,
        )
        return render_prompt(tokenizer, normalized)


def _target_slot(query: str) -> str:
    query_lower = query.lower()
    matches = [slot for slot, item in SLOT_SCHEMA.items() if item["label"] in query_lower]
    if len(matches) != 1:
        raise ValueError(f"expected one target slot in final query, found {matches}")
    return matches[0]


def _is_stale_reminder(text: str) -> bool:
    lowered = text.lower()
    return "still thinking about" in lowered and "compare options" in lowered


def infer_natural_route(messages: list[dict], tokenizer) -> dict:
    """Infer target state and stale mention keys from dialogue text only."""
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("the final dialogue message must be the user query")
    target_slot = _target_slot(messages[-1]["content"])
    values = SLOT_SCHEMA[target_slot]["values"]

    mentions = []
    for message_index, message in enumerate(messages[:-1]):
        if message.get("role") != "user":
            continue
        content = message["content"]
        for value in values:
            for match in _value_matches(content, value):
                mentions.append(
                    {
                        "message_index": message_index,
                        "char_start": match.start(),
                        "char_end": match.end(),
                        "value": value,
                        "kind": "stale_reminder" if _is_stale_reminder(content) else "binding_update",
                    }
        )
    mentions.sort(key=lambda item: (item["message_index"], item["char_start"]))
    updates = [mention for mention in mentions if mention["kind"] == "binding_update"]
    if not updates:
        raise ValueError(f"no update-intent mention found for {target_slot}")
    current = updates[-1]
    stale_mentions = [mention for mention in mentions if mention["value"] != current["value"]]
    if not stale_mentions:
        raise ValueError(f"no superseded mention found for {target_slot}")

    prompt = render_natural_prompt(tokenizer, messages)
    starts = _message_offsets(prompt, messages)
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]

    def attach_positions(mention):
        start = starts[mention["message_index"]] + mention["char_start"]
        end = starts[mention["message_index"]] + mention["char_end"]
        return {
            **mention,
            "prompt_char_start": start,
            "prompt_char_end": end,
            "token_positions": _char_span_to_tokens(offsets, start, end),
        }

    mentions = [attach_positions(mention) for mention in mentions]
    stale_mentions = [mention for mention in mentions if mention["value"] != current["value"]]
    current_mentions = [
        mention
        for mention in mentions
        if mention["kind"] == "binding_update" and mention["value"] == current["value"]
    ]
    current_mention = current_mentions[-1]

    all_value_positions = []
    for message_index, message in enumerate(messages[:-1]):
        content = message["content"]
        for item in SLOT_SCHEMA.values():
            for value in item["values"]:
                for match in _value_matches(content, value):
                    start = starts[message_index] + match.start()
                    end = starts[message_index] + match.end()
                    all_value_positions.extend(_char_span_to_tokens(offsets, start, end))

    return {
        "target_slot": target_slot,
        "target_slot_label": SLOT_SCHEMA[target_slot]["label"],
        "current_value": current["value"],
        "current_positions": list(current_mention["token_positions"]),
        "stale_mentions": stale_mentions,
        "stale_positions": [
            position for mention in stale_mentions for position in mention["token_positions"]
        ],
        "all_value_positions": sorted(set(all_value_positions)),
        "query_char_start": starts[-1],
        "prompt": prompt,
        "_message_starts": starts,
        "_offset_mapping": offsets,
    }


def _metadata_positions(row: dict, tokenizer, route: dict | None = None) -> dict:
    route = route or infer_natural_route(row["messages"], tokenizer)
    starts = route["_message_starts"]
    offsets = route["_offset_mapping"]
    stale_positions = []
    current_positions = []
    for mention in row["value_mentions"]:
        start = starts[mention["message_index"]] + mention["char_start"]
        end = starts[mention["message_index"]] + mention["char_end"]
        positions = _char_span_to_tokens(offsets, start, end)
        if mention["is_current"]:
            current_positions.extend(positions)
        else:
            stale_positions.extend(positions)
    return {
        "stale_positions": stale_positions,
        "current_positions": current_positions,
    }


def audit_natural_routes(rows: list[dict], tokenizer) -> tuple[dict, list[dict]]:
    mismatches = []
    routes = []
    for row in rows:
        route = infer_natural_route(row["messages"], tokenizer)
        expected = _metadata_positions(row, tokenizer, route)
        if (
            route["target_slot"] != row["slot"]
            or route["current_value"] != row["current_value"]
            or route["stale_positions"] != expected["stale_positions"]
            or route["current_positions"] != expected["current_positions"]
        ):
            mismatches.append(
                {
                    "id": row["id"],
                    "inferred_slot": route["target_slot"],
                    "expected_slot": row["slot"],
                    "inferred_current": route["current_value"],
                    "expected_current": row["current_value"],
                    "inferred_stale_positions": route["stale_positions"],
                    "expected_stale_positions": expected["stale_positions"],
                    "inferred_current_positions": route["current_positions"],
                    "expected_current_positions": expected["current_positions"],
                }
            )
        route.pop("_message_starts", None)
        route.pop("_offset_mapping", None)
        routes.append(route)
    summary = {
        "n": len(rows),
        "matches": len(rows) - len(mismatches),
        "accuracy": (len(rows) - len(mismatches)) / len(rows) if rows else 0.0,
        "mismatches": mismatches,
    }
    return summary, routes


def _decision_mapping(row: dict) -> dict[str, str]:
    digest = hashlib.sha256(f"R9C|{row['id']}".encode()).digest()
    shift = int.from_bytes(digest[:2], "big") % len(DECISION_CODES)
    codes = DECISION_CODES[shift:] + DECISION_CODES[:shift]
    return dict(zip(row["slot_values"], codes))


def build_derived_row(row: dict) -> dict:
    output = copy.deepcopy(row)
    mapping = _decision_mapping(row)
    table = "\n".join(f"- {value} -> {mapping[value]}" for value in row["slot_values"])
    output["messages"][-1]["content"] = (
        f"Decision table for my {row['slot_label']}:\n{table}\n"
        f"Using the latest selected {row['slot_label']} from this conversation, "
        "which action code applies? Reply with only the code."
    )
    output["query"] = output["messages"][-1]["content"]
    output["task_type"] = "derived_decision"
    output["decision_mapping"] = mapping
    output["current_code"] = mapping[row["current_value"]]
    output["stale_codes"] = sorted({mapping[value] for value in row["stale_values"]})
    return output


def add_current_state_recap(messages: list[dict], route: dict) -> list[dict]:
    output = copy.deepcopy(messages)
    slot_label = route.get("target_slot_label", SLOT_SCHEMA[route["target_slot"]]["label"])
    output.insert(
        len(output) - 1,
        {
            "role": "assistant",
            "content": (
                f"Current state recap: the latest selected {slot_label} "
                f"is {route['current_value']}."
            ),
        },
    )
    return output


def classify_natural_task_response(row: dict, response: str) -> dict:
    if row["task_type"] == "retrieval":
        return classify_stage_l_response(response, row)
    current_hit = contains_value(response.lower(), row["current_code"])
    stale_hits = [code for code in row["stale_codes"] if contains_value(response.lower(), code)]
    if current_hit:
        label = "correct_current"
    elif stale_hits:
        label = "within_stale"
    else:
        label = "other"
    return {
        "label": label,
        "current_hit": current_hit,
        "stale_hits": stale_hits,
        "same_slot_other_hits": [],
        "cross_slot_hits": [],
    }


def _semantic_id(row: dict) -> str:
    cell = row["factorial_cell"]
    return "|".join(
        (
            row["slot"],
            row["current_value"],
            str(row["k_overwrites"]),
            cell["same_slot_stale_distance_bin"],
            cell["recent_other_slot_distance_bin"],
        )
    )


def prepare_task_rows(base_rows: list[dict]) -> dict[str, list[dict]]:
    retrieval = []
    derived = []
    for source in base_rows:
        row = copy.deepcopy(source)
        row["task_type"] = "retrieval"
        row["semantic_id"] = _semantic_id(row)
        retrieval.append(row)
        decision = build_derived_row(source)
        decision["semantic_id"] = _semantic_id(decision)
        derived.append(decision)
    return {"retrieval": retrieval, "derived_decision": derived}


def _limit_cells(rows: list[dict], limit_per_cell: int | None) -> list[dict]:
    if not limit_per_cell:
        return rows
    grouped = defaultdict(list)
    for row in rows:
        cell = row["factorial_cell"]
        key = (cell["same_slot_stale_distance_bin"], cell["recent_other_slot_distance_bin"])
        if len(grouped[key]) < limit_per_cell:
            grouped[key].append(row)
    return [row for key in sorted(grouped) for row in grouped[key]]


def split_factorial_rows(rows: list[dict], calibration_fraction: float = 0.2):
    grouped = defaultdict(list)
    for row in rows:
        cell = row["factorial_cell"]
        key = (cell["same_slot_stale_distance_bin"], cell["recent_other_slot_distance_bin"])
        grouped[key].append(row)
    calibration = []
    confirmation = []
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key],
            key=lambda row: hashlib.sha256(f"R9C-SPLIT|{row['id']}".encode()).digest(),
        )
        n_calibration = max(1, int(round(len(ordered) * calibration_fraction)))
        if n_calibration >= len(ordered):
            n_calibration = len(ordered) - 1
        if n_calibration < 1:
            raise ValueError(f"factorial cell {key} needs at least two rows")
        calibration.extend(ordered[:n_calibration])
        confirmation.extend(ordered[n_calibration:])
    return calibration, confirmation


def load_routing_heads(path: str | Path) -> list[tuple[int, int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    source = payload.get("topk_heads") or payload.get("top_heads") or payload.get("heads")
    if not source:
        raise ValueError(f"no head list found in {path}")
    heads = [(int(row["layer"]), int(row["head"])) for row in source]
    if len(heads) != 8 or len(set(heads)) != 8:
        raise ValueError(f"expected the frozen eight-head Qwen7B set, found {len(heads)}")
    return heads


def choose_beta(curve: list[dict], minimum_preservation: float = 0.95) -> dict:
    eligible = [
        row for row in curve if float(row["correct_preservation"]) >= minimum_preservation
    ]
    if not eligible:
        raise ValueError("no beta satisfies calibration preservation")
    return max(
        eligible,
        key=lambda row: (float(row["net_accuracy_gain"]), -float(row["beta"])),
    )


def _generate_arm(
    model,
    tokenizer,
    device,
    rows,
    routes,
    heads,
    beta,
    batch_size,
    max_new_tokens,
    *,
    positions=None,
    boost_positions=None,
    boost_beta=0.0,
    recap=False,
):
    import torch

    records = []
    first_scores = []
    heads_by_layer = _heads_by_layer(heads)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_routes = routes[start : start + batch_size]
            batch_positions = positions[start : start + batch_size] if positions is not None else [route["stale_positions"] for route in batch_routes]
            batch_boost_positions = (
                boost_positions[start : start + batch_size]
                if boost_positions is not None
                else [tuple() for _ in batch_routes]
            )
            messages = [
                add_current_state_recap(row["messages"], route) if recap else row["messages"]
                for row, route in zip(batch_rows, batch_routes)
            ]
            prompts = [render_natural_prompt(tokenizer, item) for item in messages]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            width = int(enc["input_ids"].shape[1])
            lengths = enc["attention_mask"].sum(dim=1).detach().cpu().tolist()
            adjusted_positions = []
            adjusted_boost_positions = []
            for length, values, positive_values in zip(
                lengths, batch_positions, batch_boost_positions
            ):
                pad = width - int(length)
                adjusted_positions.append(tuple(pad + int(value) for value in values))
                adjusted_boost_positions.append(
                    tuple(pad + int(value) for value in positive_values)
                )
            context = RoutingBiasContext(
                beta=float(beta),
                key_positions=tuple(adjusted_positions),
                query_positions=tuple(width - 1 for _ in batch_rows),
                heads_by_layer=heads_by_layer,
                boost_beta=float(boost_beta),
                boost_positions=tuple(adjusted_boost_positions),
            )
            with routing_bias(context), torch.inference_mode():
                generated = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            first_scores.append(generated.scores[0].detach().float().cpu().numpy())
            generated_ids = generated.sequences[:, width:]
            for row, token_ids in zip(batch_rows, generated_ids):
                response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                records.append(
                    {
                        "id": row["id"],
                        "semantic_id": row["semantic_id"],
                        "task_type": row["task_type"],
                        "factorial_cell": row["factorial_cell"],
                        "response": response,
                        **classify_natural_task_response(row, response),
                    }
                )
    finally:
        tokenizer.padding_side = old_padding_side
    return records, np.concatenate(first_scores, axis=0)


def make_random_position_sets(rows, routes, tokenizer, n_random: int, seed: int):
    output = []
    seen = set()
    attempt = 0
    valid_by_row = []
    for row, route in zip(rows, routes):
        offsets = tokenizer(
            route["prompt"], add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        excluded = set(route["all_value_positions"])
        valid = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start
            and token_end <= route["query_char_start"]
            and index not in excluded
        ]
        if len(valid) < len(route["stale_positions"]):
            raise ValueError(f"not enough random positions for {row['id']}")
        valid_by_row.append(valid)
    while len(output) < n_random:
        attempt += 1
        if attempt > 10_000:
            raise RuntimeError("could not construct unique natural random-position sets")
        positions_by_row = []
        for row, route, valid in zip(rows, routes, valid_by_row):
            count = len(route["stale_positions"])
            row_seed = int.from_bytes(
                hashlib.sha256(f"{seed}|{attempt}|{row['id']}|{row['task_type']}".encode()).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(row_seed)
            positions_by_row.append(tuple(sorted(int(value) for value in rng.choice(valid, count, replace=False))))
        key = tuple(positions_by_row)
        if key in seen:
            continue
        seen.add(key)
        output.append(positions_by_row)
    return output


def make_random_balanced_position_sets(rows, routes, tokenizer, n_random: int, seed: int):
    output = []
    seen = set()
    valid_by_row = []
    for row, route in zip(rows, routes):
        offsets = tokenizer(
            route["prompt"], add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        excluded = set(route["all_value_positions"])
        valid = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start
            and token_end <= route["query_char_start"]
            and index not in excluded
        ]
        required = len(route["stale_positions"]) + len(route["current_positions"])
        if len(valid) < required:
            raise ValueError(f"not enough balanced random positions for {row['id']}")
        valid_by_row.append(valid)
    attempt = 0
    while len(output) < n_random:
        attempt += 1
        if attempt > 10_000:
            raise RuntimeError("could not construct unique balanced random-position sets")
        negative_by_row = []
        positive_by_row = []
        for row, route, valid in zip(rows, routes, valid_by_row):
            negative_count = len(route["stale_positions"])
            positive_count = len(route["current_positions"])
            row_seed = int.from_bytes(
                hashlib.sha256(f"R9C-BAL|{seed}|{attempt}|{row['id']}|{row['task_type']}".encode()).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(row_seed)
            selected = [
                int(value)
                for value in rng.choice(
                    valid, negative_count + positive_count, replace=False
                )
            ]
            negative_by_row.append(tuple(sorted(selected[:negative_count])))
            positive_by_row.append(tuple(sorted(selected[negative_count:])))
        key = (tuple(negative_by_row), tuple(positive_by_row))
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {"negative": negative_by_row, "positive": positive_by_row}
        )
    return output


def _metrics(rows, baseline, arm, n_boot: int, seed: int) -> dict:
    if [row["id"] for row in baseline] != [row["id"] for row in arm]:
        raise ValueError("baseline and arm rows are not aligned")
    base_correct = np.asarray([row["label"] == "correct_current" for row in baseline])
    arm_correct = np.asarray([row["label"] == "correct_current" for row in arm])
    gain = arm_correct.astype(float) - base_correct.astype(float)
    groups = [row["semantic_id"] for row in rows]
    bootstrap = clustered_paired_bootstrap(gain, groups, n_boot=n_boot, seed=seed)
    transitions = Counter(
        f"{before['label']}->{after['label']}" for before, after in zip(baseline, arm)
    )
    by_mode = {}
    for mode in sorted({row["label"] for row in baseline}):
        mask = np.asarray([row["label"] == mode for row in baseline])
        by_mode[mode] = {
            "n": int(mask.sum()),
            "to_correct": float(arm_correct[mask].mean()) if mask.any() else math.nan,
        }
    return {
        "n": len(rows),
        "baseline_counts": dict(Counter(row["label"] for row in baseline)),
        "arm_counts": dict(Counter(row["label"] for row in arm)),
        "baseline_accuracy": float(base_correct.mean()),
        "arm_accuracy": float(arm_correct.mean()),
        "net_accuracy_gain": float(gain.mean()),
        "paired_net_gain": bootstrap,
        "correct_preservation": float(arm_correct[base_correct].mean()) if base_correct.any() else math.nan,
        "within_stale_correction": float(arm_correct[np.asarray([row["label"] == "within_stale" for row in baseline])].mean()) if any(row["label"] == "within_stale" for row in baseline) else math.nan,
        "by_baseline_mode": by_mode,
        "transitions": dict(sorted(transitions.items())),
    }


def _per_cell(rows, baseline, arm):
    output = {}
    for same in ("near", "far"):
        for other in ("far", "mid", "near2"):
            indices = [
                index
                for index, row in enumerate(rows)
                if row["factorial_cell"]["same_slot_stale_distance_bin"] == same
                and row["factorial_cell"]["recent_other_slot_distance_bin"] == other
            ]
            base = np.asarray([baseline[index]["label"] == "correct_current" for index in indices], dtype=float)
            target = np.asarray([arm[index]["label"] == "correct_current" for index in indices], dtype=float)
            output[f"same_{same}__other_{other}"] = {
                "n": len(indices),
                "baseline_accuracy": float(base.mean()) if len(base) else math.nan,
                "arm_accuracy": float(target.mean()) if len(target) else math.nan,
                "net_accuracy_gain": float((target - base).mean()) if len(base) else math.nan,
            }
    return output


def _control_summary(records):
    return {
        "n": len(records),
        "net_accuracy_gain": _random_summary([row["net_accuracy_gain"] for row in records]),
        "correct_preservation": _random_summary([row["correct_preservation"] for row in records]),
    }


def _task_gate(metric, per_cell, random_summary, audit_exact, identity_exact, minimum_pool):
    counts = metric["baseline_counts"]
    gate = {
        "route_audit_exact": audit_exact,
        "identity_exact": identity_exact,
        "adequate_correct_pool": counts.get("correct_current", 0) >= minimum_pool,
        "adequate_within_stale_pool": counts.get("within_stale", 0) >= minimum_pool,
        "net_gain_ci_above_zero": metric["paired_net_gain"]["ci"][0] > 0.0,
        "correct_preservation": metric["correct_preservation"] >= 0.95,
        "exceeds_random_position95": metric["net_accuracy_gain"] > random_summary["net_accuracy_gain"]["interval95"][1],
        "positive_in_four_cells": sum(item["net_accuracy_gain"] > 0.0 for item in per_cell.values()) >= 4,
    }
    gate["all_pass"] = all(gate.values())
    return gate


def run(args) -> None:
    install_qwen2_routing_patch()
    base_rows = _limit_cells(load_jsonl(args.data), args.limit_per_cell)
    if not base_rows:
        raise ValueError("natural CICM input is empty")
    calibration_base, confirmation_base = split_factorial_rows(
        base_rows, args.calibration_fraction
    )
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    device = next(model.parameters()).device
    audit, _ = audit_natural_routes(base_rows, tokenizer)
    if audit["mismatches"]:
        raise RuntimeError("natural route audit failed")
    heads = load_routing_heads(args.head_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration_rows = prepare_task_rows(calibration_base)["retrieval"]
    calibration_routes = [
        infer_natural_route(row["messages"], tokenizer) for row in calibration_rows
    ]
    calibration_baseline, calibration_scores = _generate_arm(
        model,
        tokenizer,
        device,
        calibration_rows,
        calibration_routes,
        [],
        0.0,
        args.batch_size,
        args.max_new_tokens,
    )
    curve = []
    calibration_boost_positions = (
        [route["current_positions"] for route in calibration_routes]
        if args.routing_mode == "balanced"
        else None
    )
    for beta_index, beta in enumerate(args.betas):
        arm, arm_scores = _generate_arm(
            model,
            tokenizer,
            device,
            calibration_rows,
            calibration_routes,
            heads,
            beta,
            args.batch_size,
            args.max_new_tokens,
            boost_positions=calibration_boost_positions,
            boost_beta=beta if args.routing_mode == "balanced" else 0.0,
        )
        metric = _metrics(
            calibration_rows,
            calibration_baseline,
            arm,
            min(500, args.n_boot),
            args.seed + beta_index,
        )
        curve.append(
            {
                "beta": beta,
                "baseline_accuracy": metric["baseline_accuracy"],
                "arm_accuracy": metric["arm_accuracy"],
                "net_accuracy_gain": metric["net_accuracy_gain"],
                "correct_preservation": metric["correct_preservation"],
                "within_stale_correction": metric["within_stale_correction"],
                "max_abs_first_token_score_change": float(
                    np.max(np.abs(arm_scores - calibration_scores))
                ),
                "mean_abs_first_token_score_change": float(
                    np.mean(np.abs(arm_scores - calibration_scores))
                ),
            }
        )
    try:
        operating_point = choose_beta(curve, args.minimum_preservation)
        operating_status = "eligible"
    except ValueError:
        operating_point = {
            "beta": 0.0,
            "baseline_accuracy": curve[0]["baseline_accuracy"],
            "arm_accuracy": curve[0]["baseline_accuracy"],
            "net_accuracy_gain": 0.0,
            "correct_preservation": 1.0,
            "within_stale_correction": 0.0,
        }
        operating_status = "no_eligible_beta_identity_fallback"
    beta = float(operating_point["beta"])

    task_rows = prepare_task_rows(confirmation_base)

    task_summaries = {}
    for task_index, (task_type, rows) in enumerate(task_rows.items()):
        routes = [infer_natural_route(row["messages"], tokenizer) for row in rows]
        routed_boost_positions = (
            [route["current_positions"] for route in routes]
            if args.routing_mode == "balanced"
            else None
        )
        baseline, baseline_scores = _generate_arm(
            model, tokenizer, device, rows, routes, [], 0.0, args.batch_size, args.max_new_tokens
        )
        identity, identity_scores = _generate_arm(
            model, tokenizer, device, rows, routes, heads, 0.0, args.batch_size, args.max_new_tokens
        )
        identity_max = float(np.max(np.abs(identity_scores - baseline_scores)))
        identity_response_changes = sum(
            before["response"] != after["response"] for before, after in zip(baseline, identity)
        )
        if identity_max != 0.0 or identity_response_changes:
            raise RuntimeError(f"identity gate failed for {task_type}")

        routed, routed_scores = _generate_arm(
            model, tokenizer, device, rows, routes, heads, beta, args.batch_size, args.max_new_tokens,
            boost_positions=routed_boost_positions,
            boost_beta=beta if args.routing_mode == "balanced" else 0.0,
        )
        opposite, opposite_scores = _generate_arm(
            model, tokenizer, device, rows, routes, heads, -beta, args.batch_size, args.max_new_tokens,
            boost_positions=routed_boost_positions,
            boost_beta=-beta if args.routing_mode == "balanced" else 0.0,
        )
        recap, _ = _generate_arm(
            model, tokenizer, device, rows, routes, [], 0.0, args.batch_size, args.max_new_tokens, recap=True
        )

        if args.routing_mode == "balanced":
            random_sets = make_random_balanced_position_sets(
                rows,
                routes,
                tokenizer,
                args.n_random_positions,
                args.seed + 1000 * (task_index + 1),
            )
        else:
            random_sets = [
                {"negative": values, "positive": None}
                for values in make_random_position_sets(
                    rows,
                    routes,
                    tokenizer,
                    args.n_random_positions,
                    args.seed + 1000 * (task_index + 1),
                )
            ]
        random_records = []
        for index, control in enumerate(random_sets):
            random_arm, _ = _generate_arm(
                model,
                tokenizer,
                device,
                rows,
                routes,
                heads,
                beta,
                args.batch_size,
                args.max_new_tokens,
                positions=control["negative"],
                boost_positions=control["positive"],
                boost_beta=beta if args.routing_mode == "balanced" else 0.0,
            )
            metric = _metrics(rows, baseline, random_arm, 200, args.seed + 3000 + index)
            random_records.append(
                {
                    "index": index,
                    "net_accuracy_gain": metric["net_accuracy_gain"],
                    "correct_preservation": metric["correct_preservation"],
                }
            )
            print(f"{task_type} random-position {index + 1}/{len(random_sets)}", flush=True)

        routed_metric = _metrics(rows, baseline, routed, args.n_boot, args.seed + 10 * task_index)
        opposite_metric = _metrics(rows, baseline, opposite, args.n_boot, args.seed + 100 + 10 * task_index)
        recap_metric = _metrics(rows, baseline, recap, args.n_boot, args.seed + 200 + 10 * task_index)
        per_cell = _per_cell(rows, baseline, routed)
        random_summary = _control_summary(random_records)
        gate = _task_gate(
            routed_metric,
            per_cell,
            random_summary,
            audit_exact=not audit["mismatches"],
            identity_exact=identity_max == 0.0 and identity_response_changes == 0,
            minimum_pool=args.minimum_pool,
        )
        task_dir = out_dir / task_type
        task_dir.mkdir(parents=True, exist_ok=True)
        dump_jsonl(task_dir / "baseline_rows.jsonl", baseline)
        dump_jsonl(task_dir / "routed_rows.jsonl", routed)
        dump_jsonl(task_dir / "opposite_rows.jsonl", opposite)
        dump_jsonl(task_dir / "recap_rows.jsonl", recap)
        dump_jsonl(task_dir / "random_position_controls.jsonl", random_records)
        task_summaries[task_type] = {
            "routed": routed_metric,
            "opposite": opposite_metric,
            "recap": recap_metric,
            "per_factorial_cell": per_cell,
            "random_positions": random_summary,
            "identity_max_abs_first_token_score_change": identity_max,
            "identity_response_changes": identity_response_changes,
            "routed_max_abs_first_token_score_change": float(
                np.max(np.abs(routed_scores - baseline_scores))
            ),
            "routed_mean_abs_first_token_score_change": float(
                np.mean(np.abs(routed_scores - baseline_scores))
            ),
            "opposite_max_abs_first_token_score_change": float(
                np.max(np.abs(opposite_scores - baseline_scores))
            ),
            "gate": gate,
        }

    retrieval_pass = task_summaries["retrieval"]["gate"]["all_pass"]
    derived_pass = task_summaries["derived_decision"]["gate"]["all_pass"]
    if retrieval_pass and derived_pass:
        verdict = "controlled_natural_dialogue_and_derived_correction_supported"
    elif retrieval_pass:
        verdict = "controlled_natural_dialogue_retrieval_correction_only"
    else:
        verdict = "natural_dialogue_transfer_gate_failed"
    summary = {
        "stage": (
            "R10-balanced-natural-dialogue-routing"
            if args.routing_mode == "balanced"
            else "R9-C-natural-dialogue-routing"
        ),
        "verdict": verdict,
        "model": args.model,
        "dtype": args.dtype,
        "data": args.data,
        "n_base_rows": len(base_rows),
        "n_calibration_rows": len(calibration_base),
        "n_confirmation_rows": len(confirmation_base),
        "calibration": {
            "task": "retrieval",
            "fraction_within_cell": args.calibration_fraction,
            "curve": curve,
            "operating_point": operating_point,
            "operating_status": operating_status,
        },
        "beta": beta,
        "routing_mode": args.routing_mode,
        "frozen_heads": [{"layer": layer, "head": head} for layer, head in heads],
        "runtime_inputs": {
            "automatic_route": "raw messages, declared slot schema, and tokenizer offsets only",
            "forbidden_route_metadata": ["current_value", "stale_values", "value_mentions", "competition", "saved token spans"],
            "answer_value_injected_by_routing": False,
            "answer_value_injected_by_recap_baseline": True,
        },
        "route_audit": audit,
        "tasks": task_summaries,
        "scope": (
            "Qwen2.5-7B-Instruct routing transfer on controlled natural CICM grammar; "
            "not unique-head evidence, unrestricted dialogue generalization, cross-model transfer, or closed-API deployment"
        ),
    }
    dump_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"verdict": verdict, "gates": {key: value["gate"] for key, value in task_summaries.items()}}, indent=2), flush=True)


def write_report(path: str | Path, summary: dict) -> None:
    lines = [
        f"# Stage {summary['stage']}: Natural-dialogue routing",
        "",
        f"**Verdict:** `{summary['verdict']}`.",
        "",
        "The R9-B correction-efficacy authorization does not reverse its failed head-specificity gate.",
        "",
        "| Task | baseline acc. | routed acc. | net gain | 95% CI | preservation | stale correction | recap acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task_type, result in summary["tasks"].items():
        routed = result["routed"]
        recap = result["recap"]
        lines.append(
            f"| {task_type} | {routed['baseline_accuracy']:.3f} | {routed['arm_accuracy']:.3f} | "
            f"{routed['net_accuracy_gain']:.3f} | {routed['paired_net_gain']['ci']} | "
            f"{routed['correct_preservation']:.3f} | {routed['within_stale_correction']:.3f} | "
            f"{recap['arm_accuracy']:.3f} |"
        )
    lines.extend(["", "## Gates", ""])
    for task_type, result in summary["tasks"].items():
        lines.append(f"### {task_type}")
        lines.append("")
        lines.extend(
            f"- {key}: **{'PASS' if value else 'FAIL'}**"
            for key, value in result["gate"].items()
        )
        lines.append("")
    lines.extend(["## Scope", "", summary["scope"]])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/cicm/cicm_natural_factorial_otherdist_l0.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--head-summary", default="results/overwrite_attention_bias/summary.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--betas", type=lambda value: [float(item) for item in value.split(",")], default=list(DEFAULT_BETAS))
    parser.add_argument("--minimum-preservation", type=float, default=0.95)
    parser.add_argument("--routing-mode", choices=("stale_only", "balanced"), default="stale_only")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-pool", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--n-random-positions", type=int, default=16)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--limit-per-cell", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
