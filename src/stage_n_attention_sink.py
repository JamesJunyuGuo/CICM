"""Inference-only attention-sink diagnostics for Stage N-S."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pythia_gen import TEMPLATES, VAR_CANDIDATES, dump_jsonl, load_jsonl, make_row, one_token_strings


MODEL_SMOKE = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_HEADLINE = "Qwen/Qwen2.5-3B-Instruct"
SEEDS = (11, 29, 47)
QWEN_VALUE_CANDIDATES = (
    "red", "blue", "green", "gold", "black", "white", "pink", "teal", "brown", "gray",
    "silver", "orange", "purple", "yellow", "violet", "indigo", "cyan", "coral", "navy", "beige",
    "apple", "grape", "lemon", "mango", "peach", "berry", "melon", "cherry", "plum", "pear",
    "jazz", "rock", "folk", "metal", "blues", "dance", "piano", "drums", "guitar", "violin",
    "north", "south", "east", "west", "left", "right", "upper", "lower", "front", "back",
    "circle", "square", "triangle", "oval", "star", "heart", "large", "small", "fast", "slow",
    "spring", "summer", "autumn", "winter", "morning", "evening", "monday", "friday", "early", "late",
)
QWEN_FIXED_EXAMPLES = (
    ((("color", 0), ("color", 1)), "color", 1),
    ((("music", 2), ("route", 3), ("music", 4)), "music", 4),
    ((("shape", 5), ("topic", 6), ("mode", 7)), "topic", 6),
    ((("drink", 8), ("drink", 9), ("sport", 3), ("drink", 7)), "drink", 7),
)


def dump_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_split(identifier: str) -> str:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return "discovery" if int.from_bytes(digest[:8], "big") % 2 == 0 else "heldout"


def classify_token(row: dict, token_id: int) -> str:
    if int(token_id) == int(row["gold_token_id"]):
        return "correct_current"
    if int(token_id) in {int(value) for value in row.get("stale_token_ids", [])}:
        return "within_stale"
    if int(token_id) in {int(value) for value in row.get("cross_token_ids", [])}:
        return "cross_variable"
    return "other"


def answer_token_id(tokenizer, prompt: str, value: str) -> int:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    combined_ids = tokenizer.encode(prompt + " " + value, add_special_tokens=False)
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"answer retokenizes the generation boundary: {value!r}")
    answer_ids = combined_ids[len(prompt_ids) :]
    if len(answer_ids) != 1:
        raise ValueError(f"answer is not one token at generation boundary: {value!r} -> {answer_ids}")
    return int(answer_ids[0])


def qwen_single_token_values(tokenizer, candidates) -> list[str]:
    values = []
    for value in candidates:
        bare_ids = tokenizer.encode(value, add_special_tokens=False)
        context_ids = tokenizer.encode(" " + value, add_special_tokens=False)
        if len(bare_ids) != 1 or len(context_ids) != 1:
            continue
        if tokenizer.decode(bare_ids).strip() != value:
            continue
        if tokenizer.decode(context_ids).strip() != value:
            continue
        values.append(value)
    return values


def _render_qwen_block(events, target: str, answer: str | None, template: str) -> str:
    assignments = [
        (event["var"], event["value"]) if isinstance(event, dict) else event
        for event in events
    ]
    body = "\n".join(f"{variable} = {value}" for variable, value in assignments)
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


def render_qwen_prompt(
    events: list[dict],
    target: str,
    template: str,
    fixed_values: tuple[str, ...],
) -> tuple[str, int]:
    if len(fixed_values) != 10:
        raise ValueError("Qwen fixed demonstrations require exactly ten labels")
    exemplars = []
    for indexed_events, example_target, answer_index in QWEN_FIXED_EXAMPLES:
        assignments = [(variable, fixed_values[index]) for variable, index in indexed_events]
        exemplars.append(
            _render_qwen_block(
                assignments,
                example_target,
                fixed_values[answer_index],
                template,
            )
        )
    prefix = "\n\n".join(exemplars) + "\n\n"
    return prefix + _render_qwen_block(events, target, None, template), len(prefix)


def make_qwen_clean_counterfactual(
    row: dict,
    tokenizer,
    fixed_values: tuple[str, ...],
) -> dict:
    used_variables = {event["var"] for event in row["events"]}
    foil_candidates = [
        value
        for value in one_token_strings(tokenizer, VAR_CANDIDATES)
        if value not in used_variables
    ]
    stale_events = [
        event
        for event in row["events"]
        if event["var"] == row["target_var"] and not event["is_current"]
    ]
    if len(foil_candidates) < len(stale_events):
        raise ValueError("not enough one-token foil variables")
    foils = iter(foil_candidates)
    clean_events = []
    for event in row["events"]:
        copied = dict(event)
        if copied["var"] == row["target_var"] and not copied["is_current"]:
            copied["var"] = next(foils)
            copied["is_target"] = False
            copied["is_current"] = True
        clean_events.append(copied)
    prompt, task_char_start = render_qwen_prompt(
        clean_events,
        row["target_var"],
        row["template"],
        fixed_values,
    )
    return {
        **{key: value for key, value in row.items() if key not in {"writes", "prompt", "events"}},
        "id": row["id"] + "__clean",
        "pair_id": row["id"],
        "counterfactual_clean": True,
        "events": clean_events,
        "prompt": prompt,
        "task_char_start": task_char_start,
        "stale_values": [],
        "stale_token_ids": [],
        "cross_values": [
            event["value"]
            for event in clean_events
            if event["var"] != row["target_var"]
        ],
    }


def structural_sink_positions(input_ids: list[int], special_ids: set[int]) -> list[int]:
    return sorted({0, *(index for index, token_id in enumerate(input_ids) if token_id in special_ids)})


def _char_to_tokens(offsets, start: int, end: int) -> list[int]:
    return [index for index, (left, right) in enumerate(offsets) if right > start and left < end]


def annotate_chat_row(raw_row: dict, tokenizer) -> dict:
    raw_prompt = str(raw_row["prompt"])
    if "\n" not in raw_prompt:
        raise ValueError("task prompt must end with a separate query line")
    user_content, assistant_prefill = raw_prompt.rsplit("\n", 1)
    chat_prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_prefill},
        ],
        tokenize=False,
        continue_final_message=True,
    )
    if not chat_prompt.endswith(assistant_prefill):
        raise ValueError("chat template did not preserve the assistant answer prefill")
    encoding = tokenizer(
        chat_prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = [int(token_id) for token_id in encoding["input_ids"]]
    offsets = encoding["offset_mapping"]
    content_start = chat_prompt.find(user_content)
    if content_start < 0:
        raise ValueError("task context is absent from the rendered chat template")
    query_start = chat_prompt.rfind(assistant_prefill)

    cursor = content_start + int(raw_row["task_char_start"])
    writes = []
    for event in raw_row["events"]:
        needle = f"{event['var']} = {event['value']}"
        line_start = chat_prompt.find(needle, cursor)
        if line_start < 0:
            raise ValueError(f"assignment missing from chat prompt: {needle}")
        var_start = line_start
        var_end = var_start + len(event["var"])
        value_start = line_start + len(event["var"]) + 3
        value_end = value_start + len(event["value"])
        var_tokens = _char_to_tokens(offsets, var_start, var_end)
        value_tokens = _char_to_tokens(offsets, value_start, value_end)
        if len(var_tokens) != 1 or len(value_tokens) != 1:
            raise ValueError(
                f"non-single-token chat write {needle}: var={var_tokens}, value={value_tokens}"
            )
        writes.append(
            {
                **event,
                "var_token": int(var_tokens[0]),
                "value_token": int(value_tokens[0]),
                "var_token_id": input_ids[var_tokens[0]],
                "value_token_id": input_ids[value_tokens[0]],
            }
        )
        cursor = value_end

    special_ids = {int(token_id) for token_id in tokenizer.all_special_ids}
    row = {
        **raw_row,
        "raw_prompt": raw_prompt,
        "raw_task_char_start": int(raw_row["task_char_start"]),
        "prompt": chat_prompt,
        "task_char_start": content_start + int(raw_row["task_char_start"]),
        "prompt_input_ids": input_ids,
        "prompt_tokens": len(input_ids),
        "chat_content_char_start": content_start,
        "chat_query_char_start": query_start,
        "answer_boundary_mode": "assistant_prefill_continuation",
        "answer_continuation_prefix": " ",
        "sink_positions": structural_sink_positions(input_ids, special_ids),
        "writes": writes,
        "gold_token_id": answer_token_id(tokenizer, chat_prompt, str(raw_row["gold"])),
        "stale_token_ids": [
            answer_token_id(tokenizer, chat_prompt, str(value))
            for value in raw_row.get("stale_values", [])
        ],
        "cross_token_ids": [
            answer_token_id(tokenizer, chat_prompt, str(value))
            for value in raw_row.get("cross_values", [])
        ],
    }
    row["current_value_span"] = [
        write["value_token"]
        for write in writes
        if write["var"] == row["target_var"] and write.get("is_current", False)
    ]
    row["stale_value_spans"] = [
        write["value_token"]
        for write in writes
        if write["var"] == row["target_var"] and not write.get("is_current", False)
    ]
    row["cross_value_spans"] = [
        write["value_token"] for write in writes if write["var"] != row["target_var"]
    ]
    row["target_identity_spans"] = [
        write["var_token"] for write in writes if write["var"] == row["target_var"]
    ]
    if len(row["current_value_span"]) != 1:
        raise ValueError(f"expected exactly one current value span: {row['id']}")
    return row


def validate_pair_alignment(clean: dict, corrupt: dict) -> dict:
    if len(clean["prompt_input_ids"]) != len(corrupt["prompt_input_ids"]):
        return {"valid": False, "reason": "prompt token length mismatch"}
    clean_spans = [int(write["value_token"]) for write in clean["writes"]]
    corrupt_spans = [int(write["value_token"]) for write in corrupt["writes"]]
    if clean_spans != corrupt_spans:
        return {"valid": False, "reason": "write span mismatch"}
    if list(clean["current_value_span"]) != list(corrupt["current_value_span"]):
        return {"valid": False, "reason": "current span mismatch"}
    return {"valid": True, "reason": "aligned"}


def expand_gqa_values(values, num_query_heads: int):
    num_kv_heads = int(values.shape[-2])
    if num_query_heads % num_kv_heads:
        raise ValueError(
            f"query heads ({num_query_heads}) must be divisible by KV heads ({num_kv_heads})"
        )
    return values.repeat_interleave(num_query_heads // num_kv_heads, dim=-2)


def sink_components(attention, values, sink_mask, eps: float = 1e-8) -> dict:
    import torch

    if attention.ndim != 2 or values.ndim != 3:
        raise ValueError("expected attention [heads, seq] and values [heads, seq, dim]")
    if attention.shape[:2] != values.shape[:2]:
        raise ValueError("attention/value head and sequence axes must match")
    sink_mask = sink_mask.to(device=attention.device, dtype=torch.bool)
    if sink_mask.ndim != 1 or sink_mask.shape[0] != attention.shape[1]:
        raise ValueError("sink_mask must have shape [seq]")

    sink_attention = attention[:, sink_mask]
    content_attention = attention[:, ~sink_mask]
    sink_values = values[:, sink_mask, :]
    content_values = values[:, ~sink_mask, :]
    sink_mass = sink_attention.sum(dim=-1)
    content_gate = content_attention.sum(dim=-1)
    sink_residual = torch.einsum("hs,hsd->hd", sink_attention, sink_values)
    content_weighted = torch.einsum("hs,hsd->hd", content_attention, content_values)
    conditional_sink = sink_residual / sink_mass.clamp_min(eps).unsqueeze(-1)
    conditional_content = content_weighted / content_gate.clamp_min(eps).unsqueeze(-1)
    exact_output = sink_residual + content_weighted
    reconstructed = (
        sink_mass.unsqueeze(-1) * conditional_sink
        + content_gate.unsqueeze(-1) * conditional_content
    )
    return {
        "sink_mass": sink_mass,
        "content_gate": content_gate,
        "conditional_sink": conditional_sink,
        "conditional_content": conditional_content,
        "sink_residual": sink_residual,
        "simplified_gated_output": content_gate.unsqueeze(-1) * conditional_content,
        "exact_output": exact_output,
        "reconstructed_output": reconstructed,
        "degenerate_sink": sink_mass <= eps,
        "degenerate_content": content_gate <= eps,
    }


def intervention_outputs(
    *, clean_attention, corrupt_attention, clean_values, corrupt_values, sink_mask, eps: float = 1e-8
) -> dict:
    corrupt = sink_components(corrupt_attention, corrupt_values, sink_mask, eps)
    clean = sink_components(clean_attention, clean_values, sink_mask, eps)
    sink_mask = sink_mask.to(device=corrupt_attention.device, dtype=bool)

    clean_content_attention = clean_attention[:, ~sink_mask]
    clean_content_distribution = clean_content_attention / clean["content_gate"].clamp_min(eps).unsqueeze(-1)
    routed_content = np_or_torch_einsum(
        "hs,hsd->hd", clean_content_distribution, corrupt_values[:, ~sink_mask, :]
    )
    gate_output = (
        clean["sink_mass"].unsqueeze(-1) * corrupt["conditional_sink"]
        + clean["content_gate"].unsqueeze(-1) * corrupt["conditional_content"]
    )
    routing_output = (
        corrupt["sink_mass"].unsqueeze(-1) * corrupt["conditional_sink"]
        + corrupt["content_gate"].unsqueeze(-1) * routed_content
    )
    value_output = np_or_torch_einsum("hs,hsd->hd", corrupt_attention, clean_values)
    return {
        "identity": corrupt["exact_output"],
        "gate": gate_output,
        "routing": routing_output,
        "value": value_output,
        "full": clean["exact_output"],
    }


def np_or_torch_einsum(equation: str, left, right):
    import torch

    return torch.einsum(equation, left, right)


def choose_mechanism_k(rows: list[dict]) -> dict:
    candidates = []
    by_k = defaultdict(list)
    for row in rows:
        by_k[int(row["k"])].append(row)
    for k, subset in sorted(by_k.items()):
        valid = [row for row in subset if row.get("pair_valid", True)]
        per_seed = Counter(int(row["seed"]) for row in valid)
        per_template = Counter(str(row["template"]) for row in valid)
        n_current = max((int(row.get("n_current", 0)) for row in subset), default=0)
        n_stale = max((int(row.get("n_within_stale", 0)) for row in subset), default=0)
        gate = (
            len(valid) >= 90
            and len(per_seed) == 3
            and min(per_seed.values()) >= 20
            and len(per_template) == 3
            and min(per_template.values()) >= 20
        )
        candidates.append(
            {
                "k": k,
                "n_pairs": len(valid),
                "per_seed": dict(per_seed),
                "per_template": dict(per_template),
                "n_current": n_current,
                "n_within_stale": n_stale,
                "balance_objective": min(n_current, n_stale),
                "gate_pass": gate,
            }
        )
    passing = [candidate for candidate in candidates if candidate["gate_pass"]]
    chosen = max(passing, key=lambda row: (row["balance_objective"], -row["k"])) if passing else None
    return {
        "gate_pass": chosen is not None,
        "chosen_k": None if chosen is None else chosen["k"],
        "chosen": chosen,
        "candidates": candidates,
    }


def paired_bootstrap(values, n_boot: int = 2000, seed: int = 0) -> dict:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("paired_bootstrap requires a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_boot, len(array)))
    boot = array[indices].mean(axis=1)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def paired_length_controlled_delta(
    rows: list[dict],
    *,
    metric: str,
    n_boot: int = 2000,
    n_shuffle: int = 2000,
    seed: int = 0,
) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["id"]), str(row["side"]))].append(row)
    collapsed = []
    for (identifier, side), subset in sorted(grouped.items()):
        collapsed.append(
            {
                "id": identifier,
                "side": side,
                "prompt_tokens": float(np.mean([row["prompt_tokens"] for row in subset])),
                "value": float(np.mean([row[metric] for row in subset])),
            }
        )
    values = np.asarray([row["value"] for row in collapsed], dtype=float)
    log_length = np.log(np.asarray([row["prompt_tokens"] for row in collapsed], dtype=float))
    design = np.column_stack([np.ones(len(collapsed)), log_length])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - design @ coefficients
    for row, value in zip(collapsed, residual):
        row["residual"] = float(value)
    by_id = defaultdict(dict)
    for row in collapsed:
        by_id[row["id"]][row["side"]] = row
    paired = [value for value in by_id.values() if {"clean", "corrupt"} <= set(value)]
    if not paired:
        raise ValueError("no complete clean/corrupt pairs for length control")
    raw_delta = np.asarray(
        [pair["corrupt"]["value"] - pair["clean"]["value"] for pair in paired],
        dtype=float,
    )
    controlled_delta = np.asarray(
        [pair["corrupt"]["residual"] - pair["clean"]["residual"] for pair in paired],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_shuffle, len(controlled_delta)))
    shuffled = (signs * controlled_delta[None, :]).mean(axis=1)
    return {
        "n_pairs": len(paired),
        "raw_delta": float(raw_delta.mean()),
        "raw": paired_bootstrap(raw_delta, n_boot=n_boot, seed=seed + 1),
        "length_coefficients": coefficients.tolist(),
        "length_controlled_delta": float(controlled_delta.mean()),
        "length_controlled": paired_bootstrap(
            controlled_delta, n_boot=n_boot, seed=seed + 2
        ),
        "shuffle95": np.quantile(shuffled, [0.025, 0.975]).tolist(),
        "n_shuffle": int(n_shuffle),
    }


def ratio_of_means_bootstrap(
    numerator,
    denominator,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if numerator.shape != denominator.shape or numerator.ndim != 1 or len(numerator) == 0:
        raise ValueError("ratio bootstrap requires equal non-empty one-dimensional arrays")
    mean_denominator = float(denominator.mean())
    ratio = float(numerator.mean() / mean_denominator) if mean_denominator != 0 else math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(numerator), size=(n_boot, len(numerator)))
    numerator_boot = numerator[indices].mean(axis=1)
    denominator_boot = denominator[indices].mean(axis=1)
    valid = np.abs(denominator_boot) > 1e-12
    ratios = numerator_boot[valid] / denominator_boot[valid]
    return {
        "n": int(len(numerator)),
        "ratio": ratio,
        "ci95": np.quantile(ratios, [0.025, 0.975]).tolist() if len(ratios) else [math.nan, math.nan],
        "valid_bootstrap": int(len(ratios)),
        "n_boot": int(n_boot),
    }


def validate_sink_heads(rows: list[dict]) -> dict[str, dict]:
    by_head = defaultdict(list)
    for row in rows:
        by_head[(int(row["layer"]), int(row["head"]))].append(row)
    summaries = {}
    for (layer, head), subset in sorted(by_head.items()):
        concentration = float(np.mean([row["sink_mass"] for row in subset]))
        candidate_share = float(np.mean([row["candidate_share"] for row in subset]))
        value_ratio = float(np.mean([row["sink_to_non_sink_value_norm"] for row in subset]))
        residual_ratio = float(np.mean([row["sink_residual_ratio"] for row in subset]))
        template_pass = {}
        for template in sorted({str(row["template"]) for row in subset}):
            group = [row for row in subset if str(row["template"]) == template]
            group_sink = float(np.mean([row["sink_mass"] for row in group]))
            group_share = float(np.mean([row["candidate_share"] for row in group]))
            group_value = float(
                np.mean([row["sink_to_non_sink_value_norm"] for row in group])
            )
            template_pass[template] = bool(
                group_sink >= 4.0 * group_share and group_value <= 0.5
            )
        valid = bool(
            concentration >= 4.0 * candidate_share
            and value_ratio <= 0.5
            and residual_ratio <= 0.25
            and sum(template_pass.values()) >= 2
        )
        summaries[f"{layer}.{head}"] = {
            "layer": layer,
            "head": head,
            "n": len(subset),
            "mean_sink_mass": concentration,
            "mean_candidate_share": candidate_share,
            "mean_sink_to_non_sink_value_norm": value_ratio,
            "mean_sink_residual_ratio": residual_ratio,
            "template_pass": template_pass,
            "valid_sink": valid,
        }
    return summaries


def layer_matched_random_sets(
    selected: list[tuple[int, int]],
    n_layers: int,
    n_heads: int,
    n_sets: int,
    seed: int,
) -> list[list[tuple[int, int]]]:
    del n_layers
    rng = random.Random(seed)
    selected_set = set(selected)
    layer_counts = Counter(layer for layer, _head in selected)
    out = []
    for _ in range(n_sets):
        candidate = []
        for layer, count in sorted(layer_counts.items()):
            available = [
                (layer, head)
                for head in range(n_heads)
                if (layer, head) not in selected_set
            ]
            if len(available) < count:
                raise ValueError(f"not enough random heads in layer {layer}")
            candidate.extend(rng.sample(available, count))
        out.append(sorted(candidate))
    return out


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def prepare_command(args) -> None:
    tokenizer = _load_tokenizer(args.model)
    valid_values = qwen_single_token_values(tokenizer, QWEN_VALUE_CANDIDATES)
    if len(valid_values) < 28:
        raise ValueError(
            f"Qwen adapter found {len(valid_values)} usable labels; at least 28 are required"
        )
    fixed_values = tuple(valid_values[:10])
    task_values = valid_values[10:]
    rows = []
    for seed in args.seeds:
        for k in args.k_values:
            for template in args.templates:
                for index in range(args.per_cell):
                    raw = make_row(
                        variant="single",
                        template=template,
                        k=k,
                        seed=seed,
                        index=index,
                        tokenizer=tokenizer,
                        values=task_values,
                    )
                    raw["prompt"], raw["task_char_start"] = render_qwen_prompt(
                        raw["events"],
                        raw["target_var"],
                        raw["template"],
                        fixed_values,
                    )
                    rows.append(annotate_chat_row(raw, tokenizer))
    dump_jsonl(args.out, rows)
    summary = {
        "stage": "N-S-prepare",
        "model": args.model,
        "n": len(rows),
        "seeds": list(args.seeds),
        "k_values": list(args.k_values),
        "templates": list(args.templates),
        "per_seed_template_k": int(args.per_cell),
        "fixed_demonstration_values": list(fixed_values),
        "task_value_vocabulary": task_values,
        "prompt_token_range": [
            min(row["prompt_tokens"] for row in rows),
            max(row["prompt_tokens"] for row in rows),
        ],
        "single_token_values": sorted(
            {value for row in rows for value in [row["gold"], *row["stale_values"]]}
        ),
        "structural_sink_position_count_range": [
            min(len(row["sink_positions"]) for row in rows),
            max(len(row["sink_positions"]) for row in rows),
        ],
        "scoring": "one-token greedy, program-verifiable candidates at assistant boundary",
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)


def load_qwen_model(model_name: str, dtype: str, require_cuda: bool = True):
    import torch
    from transformers import AutoModelForCausalLM

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Qwen model forward passes require a Slurm GPU allocation")
    tokenizer = _load_tokenizer(model_name)
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def _padded_batch(rows: list[dict], pad_token_id: int, device):
    import torch

    lengths = torch.tensor([len(row["prompt_input_ids"]) for row in rows], device=device)
    width = int(lengths.max().item())
    input_ids = torch.full(
        (len(rows), width), int(pad_token_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        ids = torch.tensor(row["prompt_input_ids"], dtype=torch.long, device=device)
        input_ids[index, : len(ids)] = ids
        attention_mask[index, : len(ids)] = 1
    return input_ids, attention_mask, lengths - 1


def predict_rows(model, tokenizer, device, rows: list[dict], batch_size: int) -> list[dict]:
    import torch

    evaluated = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        input_ids, attention_mask, query = _padded_batch(batch, tokenizer.pad_token_id, device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = outputs.logits[torch.arange(len(batch), device=device), query]
        predictions = logits.argmax(dim=-1).detach().cpu().tolist()
        for local_index, (row, prediction) in enumerate(zip(batch, predictions)):
            stale_ids = [int(value) for value in row.get("stale_token_ids", [])]
            stale_logits = [float(logits[local_index, token_id].item()) for token_id in stale_ids]
            if stale_logits:
                strongest_index = int(np.argmax(stale_logits))
                strongest_stale_id = stale_ids[strongest_index]
                strongest_stale_logit = stale_logits[strongest_index]
                strongest_stale_value = row["stale_values"][strongest_index]
            else:
                strongest_stale_id = int(row["gold_token_id"])
                strongest_stale_logit = float(logits[local_index, strongest_stale_id].item())
                strongest_stale_value = row["gold"]
            gold_logit = float(logits[local_index, int(row["gold_token_id"])].item())
            evaluated.append(
                {
                    **row,
                    "pred_token_id": int(prediction),
                    "prediction": tokenizer.decode([int(prediction)]),
                    "label": classify_token(row, int(prediction)),
                    "gold_logit": gold_logit,
                    "strongest_stale_token_id": int(strongest_stale_id),
                    "strongest_stale_value": strongest_stale_value,
                    "strongest_stale_logit": strongest_stale_logit,
                    "current_minus_stale_logit": gold_logit - strongest_stale_logit,
                }
            )
        print(f"evaluated {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return evaluated


def _group_summary(rows: list[dict]) -> dict:
    counts = Counter(row["label"] for row in rows)
    n = len(rows)
    errors = n - counts["correct_current"]
    return {
        "n": n,
        "counts": dict(counts),
        "accuracy": counts["correct_current"] / n if n else math.nan,
        "within_stale_rate": counts["within_stale"] / n if n else math.nan,
        "within_stale_share_errors": counts["within_stale"] / errors if errors else math.nan,
    }


def behavior_summary(rows: list[dict], model_name: str, dtype: str) -> dict:
    by_k = defaultdict(list)
    by_cell = defaultdict(list)
    for row in rows:
        by_k[int(row["k"])].append(row)
        by_cell[(int(row["k"]), row["template"], int(row["seed"]))].append(row)
    return {
        "stage": "N-S-behavior",
        "model": model_name,
        "dtype": dtype,
        "decoding": "one-token greedy over full vocabulary",
        "n": len(rows),
        "by_k": {str(key): _group_summary(value) for key, value in sorted(by_k.items())},
        "by_cell": {
            f"k{k}__{template}__seed{seed}": _group_summary(value)
            for (k, template, seed), value in sorted(by_cell.items())
        },
    }


def boundary_control_gate(
    summary: dict,
    min_k0_accuracy: float = 0.80,
    max_other_rate: float = 0.20,
) -> dict:
    k0 = summary.get("by_k", {}).get("0", {})
    total_n = sum(int(group.get("n", 0)) for group in summary.get("by_k", {}).values())
    other_n = sum(
        int(group.get("counts", {}).get("other", 0))
        for group in summary.get("by_k", {}).values()
    )
    k0_accuracy = float(k0.get("accuracy", math.nan))
    other_rate = other_n / total_n if total_n else math.nan
    checks = {
        "k0_accuracy": bool(math.isfinite(k0_accuracy) and k0_accuracy >= min_k0_accuracy),
        "other_rate": bool(math.isfinite(other_rate) and other_rate <= max_other_rate),
    }
    return {
        "stage": "N-S-boundary-gate",
        "model": summary.get("model"),
        "n": total_n,
        "k0_accuracy": k0_accuracy,
        "overall_other_rate": other_rate,
        "thresholds": {
            "min_k0_accuracy": min_k0_accuracy,
            "max_other_rate": max_other_rate,
        },
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def boundary_gate_command(args) -> None:
    result = boundary_control_gate(
        load_json(args.behavior_summary),
        min_k0_accuracy=args.min_k0_accuracy,
        max_other_rate=args.max_other_rate,
    )
    dump_json(args.out, result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["gate_pass"]:
        raise SystemExit(2)


def behavior_command(args) -> None:
    rows = load_jsonl(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    model, tokenizer, device = load_qwen_model(args.model, args.dtype, require_cuda=not args.allow_cpu)
    evaluated = predict_rows(model, tokenizer, device, rows, args.batch_size)
    dump_jsonl(args.out, evaluated)
    summary = behavior_summary(evaluated, args.model, args.dtype)
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)


def _raw_task_projection(row: dict) -> dict:
    keys = (
        "id",
        "semantic_id",
        "variant",
        "template",
        "k",
        "seed",
        "index",
        "n_lines",
        "target_var",
        "gold",
        "stale_values",
        "cross_values",
        "events",
    )
    projected = {key: row[key] for key in keys if key in row}
    projected["prompt"] = row["raw_prompt"]
    projected["task_char_start"] = int(row["raw_task_char_start"])
    return projected


def pair_command(args) -> None:
    behavior_rows = load_jsonl(args.behavior)
    stale_rows = [row for row in behavior_rows if row["label"] == "within_stale"]
    source_rows = list(stale_rows)
    if args.smoke_any_pairs:
        seen = {row["id"] for row in source_rows}
        source_rows.extend(
            row
            for row in behavior_rows
            if int(row["k"]) > 0 and row["id"] not in seen
        )
        source_rows = source_rows[: args.smoke_any_pairs]
    model, tokenizer, device = load_qwen_model(args.model, args.dtype, require_cuda=not args.allow_cpu)
    valid_values = qwen_single_token_values(tokenizer, QWEN_VALUE_CANDIDATES)
    if len(valid_values) < 10:
        raise ValueError("cannot reconstruct the fixed Qwen demonstration vocabulary")
    fixed_values = tuple(valid_values[:10])
    clean_rows = []
    corrupt_by_pair = {}
    for corrupt in source_rows:
        raw_clean = make_qwen_clean_counterfactual(
            _raw_task_projection(corrupt),
            tokenizer,
            fixed_values,
        )
        clean = annotate_chat_row(raw_clean, tokenizer)
        clean["pair_id"] = corrupt["id"]
        clean_rows.append(clean)
        corrupt_by_pair[corrupt["id"]] = corrupt
    clean_evaluated = predict_rows(model, tokenizer, device, clean_rows, args.batch_size)

    counts_by_k = defaultdict(Counter)
    for row in behavior_rows:
        counts_by_k[int(row["k"])][row["label"]] += 1
    candidates = []
    rejected = Counter()
    for clean in clean_evaluated:
        corrupt = corrupt_by_pair[clean["pair_id"]]
        if clean["label"] != "correct_current" and not args.smoke_any_pairs:
            rejected["clean_not_current"] += 1
            continue
        alignment = validate_pair_alignment(clean, corrupt)
        if not alignment["valid"]:
            rejected[alignment["reason"]] += 1
            continue
        k = int(corrupt["k"])
        clean["comparison_stale_token_id"] = int(corrupt["strongest_stale_token_id"])
        clean["comparison_stale_value"] = corrupt["strongest_stale_value"]
        corrupt["comparison_stale_token_id"] = int(corrupt["strongest_stale_token_id"])
        corrupt["comparison_stale_value"] = corrupt["strongest_stale_value"]
        candidates.append(
            {
                "id": corrupt["id"],
                "semantic_id": corrupt["semantic_id"],
                "k": k,
                "seed": int(corrupt["seed"]),
                "template": corrupt["template"],
                "split": stable_split(corrupt["semantic_id"]),
                "pair_valid": True,
                "smoke_nonheadline": bool(
                    args.smoke_any_pairs
                    and (corrupt["label"] != "within_stale" or clean["label"] != "correct_current")
                ),
                "n_current": counts_by_k[k]["correct_current"],
                "n_within_stale": counts_by_k[k]["within_stale"],
                "clean": clean,
                "corrupted": corrupt,
            }
        )
    if args.smoke_any_pairs:
        chosen = candidates[: args.smoke_any_pairs]
        selection = {
            "gate_pass": bool(chosen),
            "chosen_k": None,
            "smoke_only": True,
            "rule": "aligned pairs only; labels are not a result",
        }
    else:
        selection = choose_mechanism_k(candidates)
        chosen = (
            [row for row in candidates if int(row["k"]) == int(selection["chosen_k"])]
            if selection["gate_pass"]
            else []
        )
    dump_jsonl(args.all_out, candidates)
    dump_jsonl(args.out, chosen)
    summary = {
        "stage": "N-S-pairs",
        "model": args.model,
        "candidate_stale_failures": len(stale_rows),
        "clean_evaluated": len(clean_evaluated),
        "valid_pairs_all_k": len(candidates),
        "chosen_pair_count": len(chosen),
        "rejected": dict(rejected),
        "selection": selection,
        "split_counts": dict(Counter(row["split"] for row in chosen)),
        "requirements": (
            "corrupt=within_stale; clean=current; equal chat-token length; aligned write/current spans"
        ),
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if args.require_gate and not selection["gate_pass"]:
        raise SystemExit(2)


class QwenCapture:
    def __init__(self, model):
        self.model = model
        self.values = {}
        self.head_outputs = {}
        self.handles = []

    def __enter__(self):
        for layer_index, layer in enumerate(self.model.model.layers):
            def value_hook(_module, _inputs, output, index=layer_index):
                self.values[index] = output.detach()

            def output_pre_hook(_module, inputs, index=layer_index):
                self.head_outputs[index] = inputs[0].detach()

            self.handles.append(layer.self_attn.v_proj.register_forward_hook(value_hook))
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(output_pre_hook))
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def capture_batch(model, tokenizer, device, rows: list[dict]) -> dict:
    import torch

    input_ids, attention_mask, query = _padded_batch(rows, tokenizer.pad_token_id, device)
    with QwenCapture(model) as capture, torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    answer_logits = outputs.logits[torch.arange(len(rows), device=device), query]
    num_query_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.config.hidden_size // num_query_heads)
    values = {}
    head_outputs = {}
    attention = {}
    for layer_index in range(int(model.config.num_hidden_layers)):
        value = capture.values[layer_index].view(
            len(rows), input_ids.shape[1], num_kv_heads, head_dim
        )
        values[layer_index] = expand_gqa_values(value, num_query_heads).permute(0, 2, 1, 3)
        heads = capture.head_outputs[layer_index].view(
            len(rows), input_ids.shape[1], num_query_heads, head_dim
        )
        head_outputs[layer_index] = torch.stack(
            [heads[index, int(query[index].item())] for index in range(len(rows))]
        )
        layer_attention = outputs.attentions[layer_index]
        attention[layer_index] = torch.stack(
            [layer_attention[index, :, int(query[index].item()), :] for index in range(len(rows))]
        )
    return {
        "answer_logits": answer_logits.detach(),
        "attention": attention,
        "values": values,
        "head_outputs": head_outputs,
        "query": query,
        "width": int(input_ids.shape[1]),
    }


def _sink_mask(row: dict, width: int, device):
    import torch

    mask = torch.zeros(width, dtype=torch.bool, device=device)
    mask[[int(position) for position in row["sink_positions"]]] = True
    return mask


def layer_interventions(clean_capture: dict, corrupt_capture: dict, clean_rows: list[dict], corrupt_rows: list[dict], layer: int) -> dict:
    arms = {name: [] for name in ("identity", "gate", "routing", "value", "full")}
    for index, (clean_row, corrupt_row) in enumerate(zip(clean_rows, corrupt_rows)):
        if clean_row["sink_positions"] != corrupt_row["sink_positions"]:
            raise ValueError(f"sink positions are not aligned: {corrupt_row['id']}")
        mask = _sink_mask(
            corrupt_row,
            corrupt_capture["width"],
            corrupt_capture["attention"][layer].device,
        )
        outputs = intervention_outputs(
            clean_attention=clean_capture["attention"][layer][index],
            corrupt_attention=corrupt_capture["attention"][layer][index],
            clean_values=clean_capture["values"][layer][index],
            corrupt_values=corrupt_capture["values"][layer][index],
            sink_mask=mask,
        )
        # Causal anchors must use the tensors that actually entered o_proj.
        # Reconstructing A @ V in BF16 is close but not an exact no-op.
        outputs["identity"] = corrupt_capture["head_outputs"][layer][index]
        outputs["full"] = clean_capture["head_outputs"][layer][index]
        for arm, value in outputs.items():
            arms[arm].append(value)
    return {arm: np_or_torch_stack(values) for arm, values in arms.items()}


def np_or_torch_stack(values):
    import torch

    return torch.stack(values, dim=0)


def patched_answer_logits(
    model,
    tokenizer,
    device,
    rows: list[dict],
    replacements: dict[int, object],
    selected_heads: dict[int, list[int]],
):
    import torch

    input_ids, attention_mask, query = _padded_batch(rows, tokenizer.pad_token_id, device)
    num_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // num_heads)
    handles = []
    for layer_index, heads in selected_heads.items():
        replacement = replacements[layer_index]

        def pre_hook(_module, inputs, index=layer_index, head_ids=tuple(heads), source=replacement):
            hidden = inputs[0].clone()
            shaped = hidden.view(len(rows), hidden.shape[1], num_heads, head_dim)
            for batch_index in range(len(rows)):
                shaped[batch_index, int(query[batch_index].item()), list(head_ids), :] = source[
                    batch_index, list(head_ids), :
                ].to(shaped.dtype)
            return (shaped.reshape_as(hidden),)

        handles.append(model.model.layers[layer_index].self_attn.o_proj.register_forward_pre_hook(pre_hook))
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        return outputs.logits[torch.arange(len(rows), device=device), query].detach()
    finally:
        for handle in handles:
            handle.remove()


def _current_stale_gap(logits, rows: list[dict]):
    import torch

    gaps = []
    for index, row in enumerate(rows):
        gold = int(row["gold_token_id"])
        stale = int(row.get("comparison_stale_token_id", row["strongest_stale_token_id"]))
        gaps.append(logits[index, gold] - logits[index, stale])
    return torch.stack(gaps)


def smoke_mechanism_command(args) -> None:
    import torch

    pairs = load_jsonl(args.pairs)[: args.limit]
    if not pairs:
        raise RuntimeError("smoke requires at least one aligned pair")
    clean_rows = [pair["clean"] for pair in pairs]
    corrupt_rows = [pair["corrupted"] for pair in pairs]
    model, tokenizer, device = load_qwen_model(args.model, args.dtype, require_cuda=True)

    input_ids, attention_mask, query = _padded_batch(corrupt_rows, tokenizer.pad_token_id, device)
    with torch.no_grad():
        baseline_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    baseline_logits = baseline_output.logits[torch.arange(len(corrupt_rows), device=device), query]
    corrupt_capture = capture_batch(model, tokenizer, device, corrupt_rows)
    clean_capture = capture_batch(model, tokenizer, device, clean_rows)
    noop_max = float((corrupt_capture["answer_logits"] - baseline_logits).abs().max().item())

    reconstruction_max = 0.0
    for layer in range(int(model.config.num_hidden_layers)):
        reconstructed = torch.einsum(
            "bhs,bhsd->bhd",
            corrupt_capture["attention"][layer],
            corrupt_capture["values"][layer],
        )
        reconstruction_max = max(
            reconstruction_max,
            float((reconstructed - corrupt_capture["head_outputs"][layer]).abs().max().item()),
        )

    layer = int(args.layer)
    head = int(args.head)
    interventions = layer_interventions(
        clean_capture, corrupt_capture, clean_rows, corrupt_rows, layer
    )
    baseline_gap = _current_stale_gap(baseline_logits, corrupt_rows)
    arm_rows = []
    arm_logits = {}
    for arm in ("identity", "gate", "routing", "value", "full"):
        logits = patched_answer_logits(
            model,
            tokenizer,
            device,
            corrupt_rows,
            {layer: interventions[arm]},
            {layer: [head]},
        )
        arm_logits[arm] = logits
        gaps = _current_stale_gap(logits, corrupt_rows)
        for index, pair in enumerate(pairs):
            arm_rows.append(
                {
                    "id": pair["id"],
                    "arm": arm,
                    "layer": layer,
                    "head": head,
                    "baseline_gap": float(baseline_gap[index].item()),
                    "patched_gap": float(gaps[index].item()),
                    "gap_change": float((gaps[index] - baseline_gap[index]).item()),
                    "smoke_nonheadline": True,
                }
            )
    identity_max = float((arm_logits["identity"] - baseline_logits).abs().max().item())
    finite = all(math.isfinite(row["gap_change"]) for row in arm_rows)
    summary = {
        "stage": "N-S-qwen15-smoke",
        "model": args.model,
        "dtype": args.dtype,
        "n_pairs": len(pairs),
        "layer": layer,
        "head": head,
        "hooked_noop_max_abs_answer_logit_diff": noop_max,
        "identity_max_abs_answer_logit_diff": identity_max,
        "attention_value_reconstruction_max_abs": reconstruction_max,
        "all_arm_gap_changes_finite": finite,
        "hard_gates": {
            "hooked_noop": noop_max <= 1e-6,
            "identity": identity_max <= 1e-6,
            "aligned_pairs": all(
                validate_pair_alignment(pair["clean"], pair["corrupted"])["valid"]
                for pair in pairs
            ),
            "finite_interventions": finite,
        },
        "note": "Pipeline-integrity smoke only; labels and effects are not findings.",
    }
    summary["smoke_pass"] = all(summary["hard_gates"].values())
    dump_jsonl(args.out, arm_rows)
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["smoke_pass"]:
        raise SystemExit(2)


def _baseline_answer_logits(model, tokenizer, device, rows: list[dict]):
    import torch

    input_ids, attention_mask, query = _padded_batch(rows, tokenizer.pad_token_id, device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    return outputs.logits[torch.arange(len(rows), device=device), query].detach()


def observational_rows(model, capture: dict, rows: list[dict], side: str) -> list[dict]:
    import torch

    out = []
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // num_heads)
    unembed = model.lm_head.weight.detach()
    for layer in range(num_layers):
        attention = capture["attention"][layer]
        values = capture["values"][layer]
        heads = capture["head_outputs"][layer]
        o_weight = model.model.layers[layer].self_attn.o_proj.weight.detach()
        for index, row in enumerate(rows):
            width = capture["width"]
            sink_mask = _sink_mask(row, width, attention.device)
            components = sink_components(attention[index], values[index], sink_mask)
            value_norm = values[index].float().norm(dim=-1)
            sink_value_norm = value_norm[:, sink_mask].mean(dim=-1)
            valid_positions = torch.arange(width, device=attention.device) < int(
                row["prompt_tokens"]
            )
            content_positions = valid_positions & ~sink_mask
            content_value_norm = value_norm[:, content_positions].median(dim=-1).values
            value_ratio = sink_value_norm / content_value_norm.clamp_min(1e-8)
            residual_ratio = components["sink_residual"].float().norm(dim=-1) / components[
                "exact_output"
            ].float().norm(dim=-1).clamp_min(1e-8)
            current_positions = [int(position) for position in row["current_value_span"]]
            stale_value = row.get("comparison_stale_value", row.get("strongest_stale_value"))
            stale_positions = [
                int(write["value_token"])
                for write in row["writes"]
                if write["var"] == row["target_var"]
                and not write.get("is_current", False)
                and write["value"] == stale_value
            ]
            current_mass = attention[index, :, current_positions].sum(dim=-1)
            stale_mass = (
                attention[index, :, stale_positions].sum(dim=-1)
                if stale_positions
                else torch.zeros(num_heads, device=attention.device, dtype=attention.dtype)
            )
            conditional_current = current_mass / components["content_gate"].clamp_min(1e-8)
            conditional_stale = stale_mass / components["content_gate"].clamp_min(1e-8)
            gold = int(row["gold_token_id"])
            stale = int(row.get("comparison_stale_token_id", row["strongest_stale_token_id"]))
            direction = (unembed[gold] - unembed[stale]).float()
            projected = torch.einsum(
                "mhd,hd->hm",
                o_weight.float().view(o_weight.shape[0], num_heads, head_dim),
                heads[index].float(),
            )
            dla = torch.mv(projected, direction)
            metrics = {
                "sink_mass": components["sink_mass"].float().cpu().numpy(),
                "content_gate": components["content_gate"].float().cpu().numpy(),
                "sink_value_norm": sink_value_norm.float().cpu().numpy(),
                "content_value_norm": content_value_norm.float().cpu().numpy(),
                "value_ratio": value_ratio.float().cpu().numpy(),
                "residual_ratio": residual_ratio.float().cpu().numpy(),
                "conditional_current": conditional_current.float().cpu().numpy(),
                "conditional_stale": conditional_stale.float().cpu().numpy(),
                "head_output_norm": heads[index].float().norm(dim=-1).cpu().numpy(),
                "dla": dla.float().cpu().numpy(),
            }
            for head in range(num_heads):
                out.append(
                    {
                        "id": row.get("pair_id", row["id"]),
                        "semantic_id": row["semantic_id"],
                        "split": row.get("split", stable_split(row["semantic_id"])),
                        "side": side,
                        "label": "correct_current" if side == "clean" else "within_stale",
                        "template": row["template"],
                        "seed": int(row["seed"]),
                        "k": int(row["k"]),
                        "prompt_tokens": int(row["prompt_tokens"]),
                        "layer": layer,
                        "head": head,
                        "sink_mass": float(metrics["sink_mass"][head]),
                        "content_gate": float(metrics["content_gate"][head]),
                        "candidate_share": len(row["sink_positions"]) / int(row["prompt_tokens"]),
                        "sink_value_norm": float(metrics["sink_value_norm"][head]),
                        "non_sink_value_norm_median": float(metrics["content_value_norm"][head]),
                        "sink_to_non_sink_value_norm": float(metrics["value_ratio"][head]),
                        "sink_residual_ratio": float(metrics["residual_ratio"][head]),
                        "conditional_current_attention": float(metrics["conditional_current"][head]),
                        "conditional_stale_attention": float(metrics["conditional_stale"][head]),
                        "current_minus_stale_conditional_attention": float(
                            metrics["conditional_current"][head]
                            - metrics["conditional_stale"][head]
                        ),
                        "head_output_norm": float(metrics["head_output_norm"][head]),
                        "direct_logit_attribution": float(metrics["dla"][head]),
                    }
                )
    return out


def _head_map(heads: list[tuple[int, int]]) -> dict[int, list[int]]:
    mapped = defaultdict(list)
    for layer, head in heads:
        mapped[int(layer)].append(int(head))
    return {layer: sorted(values) for layer, values in mapped.items()}


def _patch_rows(
    *, arm: str, logits, baseline_logits, rows: list[dict], control_index: int | None = None
) -> list[dict]:
    import torch

    baseline_gap = _current_stale_gap(baseline_logits, rows)
    patched_gap = _current_stale_gap(logits, rows)
    predictions = logits.argmax(dim=-1)
    out = []
    for index, row in enumerate(rows):
        predicted = int(predictions[index].item())
        record = {
            "id": row["id"],
            "semantic_id": row["semantic_id"],
            "split": "heldout",
            "arm": arm,
            "baseline_gap": float(baseline_gap[index].item()),
            "patched_gap": float(patched_gap[index].item()),
            "gap_change": float((patched_gap[index] - baseline_gap[index]).item()),
            "pred_token_id": predicted,
            "predicted_label": classify_token(row, predicted),
            "stale_to_current_flip": bool(predicted == int(row["gold_token_id"])),
        }
        if control_index is not None:
            record["control_index"] = int(control_index)
        out.append(record)
    return out


def _rank_heads(
    model,
    tokenizer,
    device,
    clean_capture: dict,
    corrupt_capture: dict,
    clean_rows: list[dict],
    corrupt_rows: list[dict],
    max_layers: int | None,
    max_heads: int | None,
) -> tuple[list[dict], dict]:
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    layers = range(num_layers if max_layers is None else min(num_layers, max_layers))
    heads = range(num_heads if max_heads is None else min(num_heads, max_heads))
    baseline = corrupt_capture["answer_logits"]
    baseline_gap = _current_stale_gap(baseline, corrupt_rows)
    all_selected = [(layer, head) for layer in layers for head in heads]
    all_logits = patched_answer_logits(
        model,
        tokenizer,
        device,
        corrupt_rows,
        {layer: clean_capture["head_outputs"][layer] for layer in layers},
        _head_map(all_selected),
    )
    all_effect = _current_stale_gap(all_logits, corrupt_rows) - baseline_gap
    ranking = []
    total = len(all_selected)
    for position, (layer, head) in enumerate(all_selected, start=1):
        logits = patched_answer_logits(
            model,
            tokenizer,
            device,
            corrupt_rows,
            {layer: clean_capture["head_outputs"][layer]},
            {layer: [head]},
        )
        effect = _current_stale_gap(logits, corrupt_rows) - baseline_gap
        ranking.append(
            {
                "layer": layer,
                "head": head,
                "n_discovery": len(corrupt_rows),
                "mean_gap_change": float(effect.mean().item()),
                "median_gap_change": float(effect.median().item()),
                "all_head_mean_gap_change": float(all_effect.mean().item()),
            }
        )
        if position % 16 == 0 or position == total:
            print(f"ranked heads {position}/{total}", flush=True)
    ranking.sort(key=lambda row: row["mean_gap_change"], reverse=True)
    positive = [row for row in ranking if row["mean_gap_change"] > 0]
    top = positive[:12] if positive else ranking[:12]
    cumulative = []
    selected = []
    crossing = None
    for prefix, row in enumerate(top, start=1):
        selected.append((int(row["layer"]), int(row["head"])))
        logits = patched_answer_logits(
            model,
            tokenizer,
            device,
            corrupt_rows,
            {layer: clean_capture["head_outputs"][layer] for layer, _head in selected},
            _head_map(selected),
        )
        effect = _current_stale_gap(logits, corrupt_rows) - baseline_gap
        mean_effect = float(effect.mean().item())
        fraction = (
            mean_effect / float(all_effect.mean().item())
            if float(all_effect.mean().item()) > 0
            else math.nan
        )
        cumulative.append(
            {"prefix": prefix, "heads": [f"{l}.{h}" for l, h in selected], "mean_gap_change": mean_effect, "fraction_of_all": fraction}
        )
        if crossing is None and math.isfinite(fraction) and fraction >= 0.8:
            crossing = prefix
    selected_count = crossing if crossing is not None else min(12, len(top))
    selected_heads = [
        (int(row["layer"]), int(row["head"])) for row in top[:selected_count]
    ]
    summary = {
        "all_head_mean_gap_change": float(all_effect.mean().item()),
        "all_head_anchor_positive": bool(float(all_effect.mean().item()) > 0),
        "first_80pct_prefix": crossing,
        "compactness_gate": crossing is not None,
        "selected_heads": [f"{layer}.{head}" for layer, head in selected_heads],
        "cumulative": cumulative,
        "n_ranked": len(ranking),
    }
    return ranking, summary


def _norm_matched_donor_replacements(
    targeted: dict[int, object],
    corrupt_capture: dict,
    selected_heads: list[tuple[int, int]],
    permutation,
) -> dict[int, object]:
    replacements = {}
    selected_map = _head_map(selected_heads)
    for layer, heads in selected_map.items():
        base = corrupt_capture["head_outputs"][layer]
        source = targeted[layer][permutation]
        replacement = base.clone()
        for head in heads:
            target_delta = targeted[layer][:, head] - base[:, head]
            donor_delta = source[:, head] - base[:, head]
            scale = target_delta.float().norm(dim=-1) / donor_delta.float().norm(dim=-1).clamp_min(1e-8)
            replacement[:, head] = base[:, head] + donor_delta * scale.to(donor_delta.dtype).unsqueeze(-1)
        replacements[layer] = replacement
    return replacements


def full_mechanism_command(args) -> None:
    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_jsonl(args.pairs)
    discovery_pairs = [pair for pair in pairs if pair["split"] == "discovery"]
    heldout_pairs = [pair for pair in pairs if pair["split"] == "heldout"]
    if not discovery_pairs or not heldout_pairs:
        raise RuntimeError("full mechanism requires non-empty discovery and held-out pair pools")
    model, tokenizer, device = load_qwen_model(args.model, args.dtype, require_cuda=True)

    discovery_clean = [pair["clean"] for pair in discovery_pairs]
    discovery_corrupt = [pair["corrupted"] for pair in discovery_pairs]
    heldout_clean = [pair["clean"] for pair in heldout_pairs]
    heldout_corrupt = [pair["corrupted"] for pair in heldout_pairs]
    for pair, clean, corrupt in zip(discovery_pairs + heldout_pairs, discovery_clean + heldout_clean, discovery_corrupt + heldout_corrupt):
        clean["split"] = pair["split"]
        corrupt["split"] = pair["split"]

    discovery_corrupt_capture = capture_batch(model, tokenizer, device, discovery_corrupt)
    discovery_clean_capture = capture_batch(model, tokenizer, device, discovery_clean)
    heldout_corrupt_capture = capture_batch(model, tokenizer, device, heldout_corrupt)
    heldout_clean_capture = capture_batch(model, tokenizer, device, heldout_clean)
    heldout_baseline = _baseline_answer_logits(model, tokenizer, device, heldout_corrupt)
    noop_max = float(
        (heldout_baseline - heldout_corrupt_capture["answer_logits"]).abs().max().item()
    )

    observations = []
    observations.extend(observational_rows(model, discovery_clean_capture, discovery_clean, "clean"))
    observations.extend(observational_rows(model, discovery_corrupt_capture, discovery_corrupt, "corrupt"))
    observations.extend(observational_rows(model, heldout_clean_capture, heldout_clean, "clean"))
    observations.extend(observational_rows(model, heldout_corrupt_capture, heldout_corrupt, "corrupt"))
    dump_jsonl(out_dir / "observational_rows.jsonl", observations)
    sink_validation = validate_sink_heads(
        [row for row in observations if row["split"] == "discovery"]
    )
    valid_sink_heads = {
        tuple(int(part) for part in key.split("."))
        for key, value in sink_validation.items()
        if value["valid_sink"]
    }

    ranking, ranking_summary = _rank_heads(
        model,
        tokenizer,
        device,
        discovery_clean_capture,
        discovery_corrupt_capture,
        discovery_clean,
        discovery_corrupt,
        args.max_layers,
        args.max_heads,
    )
    dump_jsonl(out_dir / "head_ranking.jsonl", ranking)
    selected = [
        tuple(int(part) for part in key.split("."))
        for key in ranking_summary["selected_heads"]
    ]
    headline_heads = sorted(set(selected) & valid_sink_heads)

    patch_rows = []
    identity_max = math.nan
    full_anchor = math.nan
    if headline_heads:
        interventions = {
            layer: layer_interventions(
                heldout_clean_capture,
                heldout_corrupt_capture,
                heldout_clean,
                heldout_corrupt,
                layer,
            )
            for layer in sorted({layer for layer, _head in headline_heads})
        }
        selected_map = _head_map(headline_heads)
        targeted_logits = {}
        for arm in ("identity", "gate", "routing", "value", "full"):
            replacements = {layer: interventions[layer][arm] for layer in selected_map}
            logits = patched_answer_logits(
                model, tokenizer, device, heldout_corrupt, replacements, selected_map
            )
            targeted_logits[arm] = logits
            patch_rows.extend(
                _patch_rows(
                    arm=arm,
                    logits=logits,
                    baseline_logits=heldout_baseline,
                    rows=heldout_corrupt,
                )
            )
        identity_max = float((targeted_logits["identity"] - heldout_baseline).abs().max().item())
        full_anchor = float(
            (
                _current_stale_gap(targeted_logits["full"], heldout_corrupt)
                - _current_stale_gap(heldout_baseline, heldout_corrupt)
            ).mean().item()
        )

        random_sets = layer_matched_random_sets(
            headline_heads,
            int(model.config.num_hidden_layers),
            int(model.config.num_attention_heads),
            args.n_random,
            args.seed,
        )
        all_layers = sorted({layer for random_set in random_sets for layer, _head in random_set})
        all_interventions = {
            layer: (
                interventions[layer]
                if layer in interventions
                else layer_interventions(
                    heldout_clean_capture,
                    heldout_corrupt_capture,
                    heldout_clean,
                    heldout_corrupt,
                    layer,
                )
            )
            for layer in all_layers
        }
        for control_index, random_set in enumerate(random_sets):
            random_map = _head_map(random_set)
            for random_arm in ("gate", "routing", "value"):
                logits = patched_answer_logits(
                    model,
                    tokenizer,
                    device,
                    heldout_corrupt,
                    {
                        layer: all_interventions[layer][random_arm]
                        for layer in random_map
                    },
                    random_map,
                )
                patch_rows.extend(
                    _patch_rows(
                        arm=f"random_{random_arm}",
                        logits=logits,
                        baseline_logits=heldout_baseline,
                        rows=heldout_corrupt,
                        control_index=control_index,
                    )
                )
            if (control_index + 1) % 8 == 0:
                print(f"random controls {control_index + 1}/{len(random_sets)}", flush=True)

        targeted_gate = {layer: interventions[layer]["gate"] for layer in selected_map}
        rng = np.random.default_rng(args.seed + 1)
        for control_index in range(args.n_donor):
            if len(heldout_corrupt) < 2:
                raise RuntimeError("mismatched-donor control requires at least two held-out pairs")
            permutation = rng.permutation(len(heldout_corrupt))
            while np.any(permutation == np.arange(len(permutation))):
                permutation = rng.permutation(len(heldout_corrupt))
            replacements = _norm_matched_donor_replacements(
                targeted_gate,
                heldout_corrupt_capture,
                headline_heads,
                torch.as_tensor(permutation, device=device),
            )
            logits = patched_answer_logits(
                model, tokenizer, device, heldout_corrupt, replacements, selected_map
            )
            patch_rows.extend(
                _patch_rows(
                    arm="mismatched_donor",
                    logits=logits,
                    baseline_logits=heldout_baseline,
                    rows=heldout_corrupt,
                    control_index=control_index,
                )
            )
            if (control_index + 1) % 8 == 0:
                print(f"donor controls {control_index + 1}/{args.n_donor}", flush=True)
    dump_jsonl(out_dir / "patch_rows.jsonl", patch_rows)

    summary = {
        "stage": "N-S-full-mechanism",
        "model": args.model,
        "dtype": args.dtype,
        "n_pairs": len(pairs),
        "n_discovery": len(discovery_pairs),
        "n_heldout": len(heldout_pairs),
        "hooked_noop_max_abs_answer_logit_diff": noop_max,
        "sink_validation": sink_validation,
        "n_valid_sink_heads": len(valid_sink_heads),
        "valid_sink_heads": [f"{layer}.{head}" for layer, head in sorted(valid_sink_heads)],
        "ranking": ranking_summary,
        "headline_heads": [f"{layer}.{head}" for layer, head in headline_heads],
        "identity_max_abs_answer_logit_diff": identity_max,
        "full_output_anchor_mean_gap_change": full_anchor,
        "controls": {"n_random": args.n_random, "n_mismatched_donor": args.n_donor},
        "hard_gates": {
            "hooked_noop": noop_max <= 1e-6,
            "valid_sink_in_causal_set": bool(headline_heads),
            "identity": bool(math.isfinite(identity_max) and identity_max <= 1e-6),
            "full_output_anchor": bool(math.isfinite(full_anchor) and full_anchor > 0),
        },
        "artifacts": {
            "observational_rows": str(out_dir / "observational_rows.jsonl"),
            "head_ranking": str(out_dir / "head_ranking.jsonl"),
            "patch_rows": str(out_dir / "patch_rows.jsonl"),
        },
    }
    summary["integrity_pass"] = all(summary["hard_gates"].values())
    dump_json(out_dir / "mechanism.summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def _arm_effect_summary(rows: list[dict], arm: str, n_boot: int, seed: int) -> dict:
    subset = [row for row in rows if row["arm"] == arm]
    if not subset:
        return {"arm": arm, "n": 0, "mean": math.nan, "ci95": [math.nan, math.nan]}
    effects = np.asarray([row["gap_change"] for row in subset], dtype=float)
    result = paired_bootstrap(effects, n_boot=n_boot, seed=seed)
    result["arm"] = arm
    result["correct_flip_rate"] = float(
        np.mean([bool(row["stale_to_current_flip"]) for row in subset])
    )
    return result


def _control_effect_summary(rows: list[dict], arm: str) -> dict:
    subset = [row for row in rows if row["arm"] == arm]
    grouped = defaultdict(list)
    for row in subset:
        grouped[int(row["control_index"])].append(float(row["gap_change"]))
    means = np.asarray([np.mean(values) for _key, values in sorted(grouped.items())], dtype=float)
    return {
        "arm": arm,
        "n_sets": int(len(means)),
        "mean": float(means.mean()) if len(means) else math.nan,
        "quantile95": np.quantile(means, [0.025, 0.975]).tolist() if len(means) else [math.nan, math.nan],
        "set_means": means.tolist(),
    }


def model_report_label(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def analyze_command(args) -> None:
    out_dir = Path(args.out_dir)
    mechanism = load_json(out_dir / "mechanism.summary.json")
    observations = load_jsonl(out_dir / "observational_rows.jsonl")
    patch_rows = load_jsonl(out_dir / "patch_rows.jsonl")
    headline_heads = {
        tuple(int(part) for part in key.split(".")) for key in mechanism["headline_heads"]
    }
    heldout_observations = [
        row
        for row in observations
        if row["split"] == "heldout" and (int(row["layer"]), int(row["head"])) in headline_heads
    ]
    if heldout_observations:
        gate_delta = paired_length_controlled_delta(
            heldout_observations,
            metric="content_gate",
            n_boot=args.n_boot,
            n_shuffle=args.n_shuffle,
            seed=args.seed,
        )
    else:
        gate_delta = {
            "n_pairs": 0,
            "raw_delta": math.nan,
            "length_controlled_delta": math.nan,
            "shuffle95": [math.nan, math.nan],
        }

    targeted = {
        arm: _arm_effect_summary(patch_rows, arm, args.n_boot, args.seed + index)
        for index, arm in enumerate(("identity", "gate", "routing", "value", "full"))
    }
    controls = {
        "gate_random": _control_effect_summary(patch_rows, "random_gate"),
        "routing_random": _control_effect_summary(patch_rows, "random_routing"),
        "value_random": _control_effect_summary(patch_rows, "random_value"),
        "gate_donor": _control_effect_summary(patch_rows, "mismatched_donor"),
    }
    by_arm_id = defaultdict(dict)
    for row in patch_rows:
        if row["arm"] in {"gate", "full"}:
            by_arm_id[row["id"]][row["arm"]] = float(row["gap_change"])
    paired_ratio = [value for value in by_arm_id.values() if {"gate", "full"} <= set(value)]
    gate_fraction = (
        ratio_of_means_bootstrap(
            np.asarray([row["gate"] for row in paired_ratio]),
            np.asarray([row["full"] for row in paired_ratio]),
            n_boot=args.n_boot,
            seed=args.seed + 20,
        )
        if paired_ratio
        else {"n": 0, "ratio": math.nan, "ci95": [math.nan, math.nan]}
    )

    def exceeds_control(arm: str, control_key: str) -> bool:
        upper = controls[control_key]["quantile95"][1]
        return bool(math.isfinite(upper) and targeted[arm]["mean"] > upper)

    gate_specific = exceeds_control("gate", "gate_random") and exceeds_control(
        "gate", "gate_donor"
    )
    routing_specific = exceeds_control("routing", "routing_random")
    value_specific = exceeds_control("value", "value_random")
    delta = gate_delta["length_controlled_delta"]
    shuffle_low, shuffle_high = gate_delta["shuffle95"]
    observed_outside_null = bool(
        math.isfinite(delta) and (delta < shuffle_low or delta > shuffle_high)
    )
    causal_direction_matches = bool(
        math.isfinite(delta)
        and math.isfinite(targeted["gate"]["mean"])
        and (-delta) * targeted["gate"]["mean"] > 0
    )
    integrity_pass = bool(mechanism["integrity_pass"])
    model_label = model_report_label(mechanism["model"])
    full_support = bool(
        integrity_pass
        and observed_outside_null
        and causal_direction_matches
        and gate_specific
        and gate_fraction["ratio"] >= 0.5
        and gate_fraction["ci95"][0] > 0
    )
    if not integrity_pass:
        verdict = "uninformative_stop"
    elif full_support:
        verdict = "full_attention_sink_explanation"
    elif gate_specific and routing_specific:
        verdict = "joint_gate_and_routing"
    elif routing_specific and not gate_specific:
        verdict = "routing_not_sink"
    elif value_specific and targeted["value"]["mean"] >= max(
        targeted["gate"]["mean"], targeted["routing"]["mean"]
    ):
        verdict = "value_side_effect"
    elif gate_specific:
        verdict = "partial_gate_effect"
    else:
        verdict = "clean_sink_null"

    summary = {
        "stage": "N-S-analysis",
        "model": mechanism["model"],
        "n_pairs": mechanism["n_pairs"],
        "n_discovery": mechanism["n_discovery"],
        "n_heldout": mechanism["n_heldout"],
        "headline_heads": mechanism["headline_heads"],
        "gate_observation_corrupt_minus_clean": gate_delta,
        "targeted_arms": targeted,
        "controls": controls,
        "gate_fraction_of_full_ratio_of_means": gate_fraction,
        "reading_gates": {
            "mechanism_integrity": integrity_pass,
            "observed_gate_shift_outside_shuffle": observed_outside_null,
            "causal_direction_matches_observation": causal_direction_matches,
            "gate_exceeds_random_and_donor": gate_specific,
            "routing_exceeds_random": routing_specific,
            "value_exceeds_random": value_specific,
            "gate_at_least_half_full": bool(
                math.isfinite(gate_fraction["ratio"]) and gate_fraction["ratio"] >= 0.5
            ),
        },
        "verdict": verdict,
        "full_support": full_support,
        "scope": f"Static inference-only diagnosis on {model_label}; no training and no developmental claim.",
    }
    dump_json(out_dir / "sink.summary.json", summary)
    from stage_n_attention_sink_figs import make_figure

    make_figure(summary, out_dir / "figures")
    report = [
        f"# Stage N-S Report: Attention-Sink Diagnosis on {model_label}",
        "",
        f"**Pre-registered verdict:** `{verdict}`.",
        "",
        "This is an inference-only static mechanism test. No weights were trained or updated.",
        "",
        "## Substrate and integrity",
        "",
        f"- Exact causal pairs: {mechanism['n_pairs']} "
        f"({mechanism['n_discovery']} discovery / {mechanism['n_heldout']} held-out).",
        f"- Valid sink heads: {mechanism['n_valid_sink_heads']}; causally selected sink heads: "
        f"{', '.join(mechanism['headline_heads']) or 'none'}.",
        f"- Hooked no-op maximum answer-logit difference: "
        f"{mechanism['hooked_noop_max_abs_answer_logit_diff']:.3g}.",
        f"- Identity maximum answer-logit difference: "
        f"{mechanism['identity_max_abs_answer_logit_diff']:.3g}.",
        f"- Full-output anchor mean gap change: "
        f"{mechanism['full_output_anchor_mean_gap_change']:.4f}.",
        "",
        "## Gate observation",
        "",
        f"Corrupt-minus-clean content-gate delta: raw {gate_delta['raw_delta']:.4f}; "
        f"length-controlled {gate_delta['length_controlled_delta']:.4f}; "
        f"shuffle95 [{gate_delta['shuffle95'][0]:.4f}, {gate_delta['shuffle95'][1]:.4f}].",
        "",
        "## Held-out causal decomposition",
        "",
        "| Arm | Mean gap change | 95% CI | Correct flip rate |",
        "|---|---:|---:|---:|",
    ]
    for arm in ("identity", "gate", "routing", "value", "full"):
        value = targeted[arm]
        report.append(
            f"| {arm} | {value['mean']:.4f} | [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}] | "
            f"{value.get('correct_flip_rate', math.nan):.3f} |"
        )
    report.extend(
        [
            "",
            f"Gate/full ratio of means: {gate_fraction['ratio']:.3f} "
            f"[{gate_fraction['ci95'][0]:.3f}, {gate_fraction['ci95'][1]:.3f}].",
            "",
            "## Pre-registered adjudication",
            "",
        ]
    )
    for key, value in summary["reading_gates"].items():
        report.append(f"- {key}: **{'PASS' if value else 'FAIL'}**")
    report.extend(
        [
            "",
            f"The claim is bounded to a static {model_label} diagnosis on the frozen controlled "
            "single-variable substrate. It does not replace the Pythia training-dynamics test "
            "and does not establish a cross-model developmental law.",
            "",
        ]
    )
    (out_dir.parent / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_csv(value: str) -> list[str]:
    return [part for part in value.split(",") if part]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--model", default=MODEL_HEADLINE)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--summary", required=True)
    prepare.add_argument("--seeds", type=parse_csv_ints, default=list(SEEDS))
    prepare.add_argument("--k-values", type=parse_csv_ints, default=[0, 1, 2, 3, 4, 5])
    prepare.add_argument("--templates", type=parse_csv, default=list(TEMPLATES))
    prepare.add_argument("--per-cell", type=int, default=72)
    prepare.set_defaults(func=prepare_command)

    behavior = subparsers.add_parser("behavior")
    behavior.add_argument("--model", default=MODEL_HEADLINE)
    behavior.add_argument("--data", required=True)
    behavior.add_argument("--out", required=True)
    behavior.add_argument("--summary", required=True)
    behavior.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    behavior.add_argument("--batch-size", type=int, default=64)
    behavior.add_argument("--limit", type=int)
    behavior.add_argument("--allow-cpu", action="store_true")
    behavior.set_defaults(func=behavior_command)

    boundary_gate = subparsers.add_parser("boundary-gate")
    boundary_gate.add_argument("--behavior-summary", required=True)
    boundary_gate.add_argument("--out", required=True)
    boundary_gate.add_argument("--min-k0-accuracy", type=float, default=0.80)
    boundary_gate.add_argument("--max-other-rate", type=float, default=0.20)
    boundary_gate.set_defaults(func=boundary_gate_command)

    pair = subparsers.add_parser("pair")
    pair.add_argument("--model", default=MODEL_HEADLINE)
    pair.add_argument("--behavior", required=True)
    pair.add_argument("--all-out", required=True)
    pair.add_argument("--out", required=True)
    pair.add_argument("--summary", required=True)
    pair.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    pair.add_argument("--batch-size", type=int, default=64)
    pair.add_argument("--allow-cpu", action="store_true")
    pair.add_argument("--smoke-any-pairs", type=int, default=0)
    pair.add_argument("--require-gate", action="store_true")
    pair.set_defaults(func=pair_command)

    smoke = subparsers.add_parser("smoke-mechanism")
    smoke.add_argument("--model", default=MODEL_SMOKE)
    smoke.add_argument("--pairs", required=True)
    smoke.add_argument("--out", required=True)
    smoke.add_argument("--summary", required=True)
    smoke.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    smoke.add_argument("--limit", type=int, default=4)
    smoke.add_argument("--layer", type=int, default=0)
    smoke.add_argument("--head", type=int, default=0)
    smoke.set_defaults(func=smoke_mechanism_command)

    full = subparsers.add_parser("full-mechanism")
    full.add_argument("--model", default=MODEL_HEADLINE)
    full.add_argument("--pairs", required=True)
    full.add_argument("--out-dir", required=True)
    full.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    full.add_argument("--n-random", type=int, default=64)
    full.add_argument("--n-donor", type=int, default=64)
    full.add_argument("--seed", type=int, default=20260810)
    full.add_argument("--max-layers", type=int)
    full.add_argument("--max-heads", type=int)
    full.set_defaults(func=full_mechanism_command)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--out-dir", required=True)
    analyze.add_argument("--n-boot", type=int, default=2000)
    analyze.add_argument("--n-shuffle", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=20260810)
    analyze.set_defaults(func=analyze_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
