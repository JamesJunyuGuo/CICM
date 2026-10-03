"""Model-specific causal-head discovery and adaptive attention routing."""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import gc
import hashlib
import json
import math
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import stale_binding_natural_routing as natural
from pythia_eval import dump_json
from pythia_gen import dump_jsonl, load_jsonl
from cicm_eval import load_model_and_tokenizer
from crossmodel_routing import (
    _record_attention,
    _repeat_kv,
    capture_head_attention,
)


@dataclass(frozen=True)
class AdaptiveRoutingContext:
    margin: float
    stale_positions: tuple[tuple[int, ...], ...]
    current_positions: tuple[tuple[int, ...], ...]
    query_positions: tuple[int, ...]
    heads_by_layer: dict[int, tuple[int, ...]]
    direction: int = 1
    max_beta: float = 8.0
    gate_scale: float = 1.0
    gates_by_layer: dict[int, object] | None = None


_ADAPTIVE_CONTEXT = contextvars.ContextVar("stage_r12_adaptive_context", default=None)


@contextlib.contextmanager
def adaptive_routing(context: AdaptiveRoutingContext | None):
    token = _ADAPTIVE_CONTEXT.set(context)
    try:
        yield
    finally:
        _ADAPTIVE_CONTEXT.reset(token)


def _cell_key(row: dict) -> tuple[str, str]:
    cell = row["factorial_cell"]
    return (
        cell["same_slot_stale_distance_bin"],
        cell["recent_other_slot_distance_bin"],
    )


def split_calibration_discovery_validation(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[_cell_key(row)].append(row)
    discovery = []
    validation = []
    for key in sorted(grouped):
        ordered = sorted(
            grouped[key],
            key=lambda row: hashlib.sha256(
                f"R12-DV|{row['id']}".encode()
            ).digest(),
        )
        split = len(ordered) // 2
        if split < 1 or split == len(ordered):
            raise ValueError(f"R12 cell {key} needs at least two calibration rows")
        discovery.extend(ordered[:split])
        validation.extend(ordered[split:])
    return discovery, validation


def split_r12_rows(
    rows: list[dict],
    calibration_fraction: float,
    calibration_limit_per_cell: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Construct the canonical split before any calibration-only smoke limit."""
    calibration, confirmation = natural.split_factorial_rows(
        rows, calibration_fraction
    )
    if calibration_limit_per_cell:
        calibration = natural._limit_cells(
            calibration, calibration_limit_per_cell
        )
    return calibration, confirmation


def apply_adaptive_margin_bias(attn_scores, layer_idx: int, context: AdaptiveRoutingContext | None):
    if context is None or float(context.gate_scale) == 0.0:
        return attn_scores
    heads = context.heads_by_layer.get(int(layer_idx), ())
    layer_gates = None
    if context.gates_by_layer is not None:
        layer_gates = context.gates_by_layer.get(int(layer_idx))
        if layer_gates is not None and not heads:
            heads = tuple(range(attn_scores.shape[1]))
    if not heads:
        return attn_scores
    if len(context.stale_positions) != attn_scores.shape[0]:
        raise ValueError("adaptive stale positions do not match attention batch")
    if len(context.current_positions) != attn_scores.shape[0]:
        raise ValueError("adaptive current positions do not match attention batch")
    if len(context.query_positions) != attn_scores.shape[0]:
        raise ValueError("adaptive query positions do not match attention batch")
    if context.direction not in (-1, 1):
        raise ValueError("adaptive direction must be -1 or 1")

    import torch

    out = attn_scores.clone()
    for batch_index, (query, stale, current) in enumerate(
        zip(
            context.query_positions,
            context.stale_positions,
            context.current_positions,
        )
    ):
        query = int(query)
        if query >= out.shape[-2]:
            continue
        if not stale or not current:
            raise ValueError("adaptive routing requires stale and current positions")
        selected = tuple(stale) + tuple(current)
        if min(selected) < 0 or max(selected) >= out.shape[-1]:
            raise ValueError("adaptive key position outside attention tensor")
        for head in heads:
            head = int(head)
            stale_scores = out[batch_index, head, query, tuple(stale)]
            current_scores = out[batch_index, head, query, tuple(current)]
            if context.direction == 1:
                delta = torch.logsumexp(current_scores, 0) - torch.logsumexp(stale_scores, 0)
            else:
                delta = torch.logsumexp(stale_scores, 0) - torch.logsumexp(current_scores, 0)
            beta = (0.5 * torch.clamp(float(context.margin) - delta, min=0.0)).clamp(
                max=float(context.max_beta)
            )
            gate = float(context.gate_scale)
            if layer_gates is not None:
                gate = gate * layer_gates[head]
            shift = beta.detach() * gate
            if context.direction == 1:
                out[batch_index, head, query, tuple(stale)] -= shift
                out[batch_index, head, query, tuple(current)] += shift
            else:
                out[batch_index, head, query, tuple(stale)] += shift
                out[batch_index, head, query, tuple(current)] -= shift
    return out


def _llama_adaptive_eager_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    del kwargs
    import torch
    import torch.nn as nn

    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[
            :, :, :, : key_states.shape[-2]
        ]
    attn_weights = apply_adaptive_margin_bias(
        attn_weights, module.layer_idx, _ADAPTIVE_CONTEXT.get()
    )
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    _record_attention(attn_weights, module.layer_idx)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def _gemma2_adaptive_eager_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    softcap=None,
    **kwargs,
):
    """Gemma-2 eager attention with routing on its final pre-softmax logits."""
    del kwargs
    import torch
    import torch.nn as nn

    if scaling is None:
        scaling = module.head_dim**-0.5
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = torch.tanh(attn_weights / softcap) * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[
            :, :, :, : key_states.shape[-2]
        ]
    attn_weights = apply_adaptive_margin_bias(
        attn_weights, module.layer_idx, _ADAPTIVE_CONTEXT.get()
    )
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    _record_attention(attn_weights, module.layer_idx)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def validate_model_family(actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"loaded model_type={actual}, expected {expected}")


def install_adaptive_patch(model_family: str) -> None:
    if model_family == "llama":
        import transformers.models.llama.modeling_llama as modeling
    elif model_family == "mistral":
        import transformers.models.mistral.modeling_mistral as modeling
    elif model_family == "qwen2":
        import transformers.models.qwen2.modeling_qwen2 as modeling
    elif model_family == "gemma2":
        import transformers.models.gemma2.modeling_gemma2 as modeling
    else:
        raise ValueError(f"unsupported adaptive-routing family: {model_family}")
    if model_family == "gemma2":
        modeling.eager_attention_forward = _gemma2_adaptive_eager_attention
    else:
        modeling.eager_attention_forward = _llama_adaptive_eager_attention


def rank_causal_heads(
    failure_gradients: np.ndarray,
    correct_gradients: np.ndarray,
    target_attention_mass: np.ndarray,
) -> list[dict]:
    if failure_gradients.shape != correct_gradients.shape:
        raise ValueError("failure and correct gradients must align")
    if failure_gradients.shape != target_attention_mass.shape or failure_gradients.ndim != 2:
        raise ValueError("head arrays must align as [layer, head]")
    rows = []
    for layer in range(failure_gradients.shape[0]):
        for head in range(failure_gradients.shape[1]):
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "failure_gradient": float(failure_gradients[layer, head]),
                    "correct_gradient": float(correct_gradients[layer, head]),
                    "target_attention_mass": float(target_attention_mass[layer, head]),
                }
            )
    return sorted(rows, key=lambda row: (-row["failure_gradient"], row["layer"], row["head"]))


def choose_validation_configuration(
    rows: list[dict], minimum_preservation: float = 0.95
) -> dict:
    eligible = [
        row
        for row in rows
        if float(row["correct_preservation"]) >= minimum_preservation
        and float(row["net_accuracy_gain"]) > 0.0
    ]
    if not eligible:
        raise ValueError("no positive validation configuration preserves correct answers")
    return max(
        eligible,
        key=lambda row: (
            float(row["net_accuracy_gain"]),
            -int(row["size"]),
            -float(row["margin"]),
        ),
    )


def discovery_pool_gate(
    n_correct: int,
    n_within_stale: int,
    minimum_correct: int,
    minimum_within_stale: int,
) -> dict:
    gate = {
        "n_correct": int(n_correct),
        "n_within_stale": int(n_within_stale),
        "minimum_correct": int(minimum_correct),
        "minimum_within_stale": int(minimum_within_stale),
        "correct_pass": int(n_correct) >= int(minimum_correct),
        "within_stale_pass": int(n_within_stale) >= int(minimum_within_stale),
    }
    gate["all_pass"] = gate["correct_pass"] and gate["within_stale_pass"]
    return gate


def _heads_by_layer(heads: list[tuple[int, int]]) -> dict[int, tuple[int, ...]]:
    grouped = defaultdict(list)
    for layer, head in heads:
        grouped[int(layer)].append(int(head))
    return {layer: tuple(sorted(values)) for layer, values in grouped.items()}


def make_layer_matched_random_head_sets(
    selected_heads: list[tuple[int, int]],
    n_heads: int,
    n_random: int,
    seed: int,
) -> list[list[tuple[int, int]]]:
    """Sample unique controls with the selected set's per-layer head counts."""
    selected_by_layer = _heads_by_layer(selected_heads)
    selected = set(selected_heads)
    for layer, heads in selected_by_layer.items():
        if len(heads) > n_heads - len(heads):
            raise ValueError(f"layer {layer} lacks enough non-selected control heads")
    rng = np.random.default_rng(seed)
    output = []
    seen = set()
    attempts = 0
    while len(output) < n_random:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("could not construct unique layer-matched head sets")
        candidate = []
        for layer, layer_heads in sorted(selected_by_layer.items()):
            choices = [
                head for head in range(n_heads) if (int(layer), head) not in selected
            ]
            sampled = rng.choice(choices, size=len(layer_heads), replace=False)
            candidate.extend((int(layer), int(head)) for head in sorted(sampled.tolist()))
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def confirmation_gate(
    metric: dict,
    per_cell: dict,
    random_position_summary: dict,
    random_head_summary: dict,
    *,
    route_audit_exact: bool,
    identity_exact: bool,
    identity_required: bool = True,
    minimum_pool: int,
    minimum_preservation: float,
) -> dict:
    counts = metric["baseline_counts"]
    gain = float(metric["net_accuracy_gain"])
    gate = {
        "route_audit_exact": bool(route_audit_exact),
        "identity_exact": bool(identity_exact),
        "identity_required": bool(identity_required),
        "adequate_correct_pool": counts.get("correct_current", 0) >= minimum_pool,
        "adequate_within_stale_pool": counts.get("within_stale", 0) >= minimum_pool,
        "net_gain_ci_above_zero": float(metric["paired_net_gain"]["ci"][0]) > 0.0,
        "correct_preservation": float(metric["correct_preservation"])
        >= minimum_preservation,
        "exceeds_random_position95": gain
        > float(random_position_summary["net_accuracy_gain"]["interval95"][1]),
        "exceeds_random_head95": gain
        > float(random_head_summary["net_accuracy_gain"]["interval95"][1]),
        "positive_in_four_cells": sum(
            float(item["net_accuracy_gain"]) > 0.0 for item in per_cell.values()
        )
        >= 4,
    }
    required_checks = [
        value
        for name, value in gate.items()
        if name not in {"identity_exact", "identity_required"}
    ]
    if identity_required:
        required_checks.append(bool(identity_exact))
    gate["all_pass"] = all(required_checks)
    return gate


def _adjust_route_positions(enc, routes: list[dict]):
    width = int(enc["input_ids"].shape[1])
    lengths = enc["attention_mask"].sum(dim=1).detach().cpu().tolist()
    stale = []
    current = []
    for length, route in zip(lengths, routes):
        pad = width - int(length)
        stale.append(tuple(pad + int(value) for value in route["stale_positions"]))
        current.append(tuple(pad + int(value) for value in route["current_positions"]))
    return width, tuple(stale), tuple(current)


def _value_first_token_ids(tokenizer, value: str) -> tuple[int, ...]:
    variants = (value, value.capitalize(), " " + value, " " + value.capitalize())
    output = []
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids and int(ids[0]) not in output:
            output.append(int(ids[0]))
    if not output:
        raise ValueError(f"value has no tokenization: {value!r}")
    return tuple(output)


def _candidate_sets(tokenizer, row: dict) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    current = _value_first_token_ids(tokenizer, row["current_value"])
    stale = tuple(_value_first_token_ids(tokenizer, value) for value in row["stale_values"])
    current_set = set(current)
    if any(current_set.intersection(values) for values in stale):
        raise ValueError(f"current/stale first-token collision for {row['id']}")
    return current, stale


def partition_candidate_discovery(tokenizer, rows, routes, baseline):
    """Remove only rows undefined under the first-token discovery objective."""
    if not (len(rows) == len(routes) == len(baseline)):
        raise ValueError("candidate discovery inputs must have equal lengths")
    kept_rows = []
    kept_routes = []
    kept_baseline = []
    exclusions = []
    for row, route, record in zip(rows, routes, baseline):
        try:
            _candidate_sets(tokenizer, row)
        except ValueError as error:
            if "current/stale first-token collision" not in str(error):
                raise
            exclusions.append(
                {
                    "id": row["id"],
                    "reason": "current_stale_first_token_collision",
                }
            )
            continue
        kept_rows.append(row)
        kept_routes.append(route)
        kept_baseline.append(record)
    return kept_rows, kept_routes, kept_baseline, exclusions


def _candidate_gap(logits, tokenizer, row: dict):
    import torch

    current, stale_groups = _candidate_sets(tokenizer, row)
    current_ids = torch.as_tensor(current, dtype=torch.long, device=logits.device)
    current_score = torch.logsumexp(logits.index_select(0, current_ids), 0)
    stale_scores = []
    for group in stale_groups:
        ids = torch.as_tensor(group, dtype=torch.long, device=logits.device)
        stale_scores.append(torch.logsumexp(logits.index_select(0, ids), 0))
    return current_score - torch.stack(stale_scores).max()


def _candidate_prediction_label(pred_id: int, tokenizer, row: dict) -> str:
    current, stale_groups = _candidate_sets(tokenizer, row)
    if int(pred_id) in current:
        return "correct_current"
    if any(int(pred_id) in values for values in stale_groups):
        return "within_stale"
    return "other"


def _generate_adaptive_arm(
    model,
    tokenizer,
    rows: list[dict],
    routes: list[dict],
    heads: list[tuple[int, int]],
    margin: float,
    batch_size: int,
    max_new_tokens: int,
    *,
    direction: int = 1,
    gate_scale: float = 1.0,
    stale_positions: list[tuple[int, ...]] | None = None,
    current_positions: list[tuple[int, ...]] | None = None,
    recap: bool = False,
):
    import torch

    records = []
    first_scores = []
    device = next(model.parameters()).device
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_routes = routes[start : start + batch_size]
            batch_stale = (
                stale_positions[start : start + batch_size]
                if stale_positions is not None
                else [route["stale_positions"] for route in batch_routes]
            )
            batch_current = (
                current_positions[start : start + batch_size]
                if current_positions is not None
                else [route["current_positions"] for route in batch_routes]
            )
            messages = [
                natural.add_current_state_recap(row["messages"], route)
                if recap
                else row["messages"]
                for row, route in zip(batch_rows, batch_routes)
            ]
            prompts = [natural.render_natural_prompt(tokenizer, item) for item in messages]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            route_overrides = [
                {
                    **route,
                    "stale_positions": tuple(stale_values),
                    "current_positions": tuple(current_values),
                }
                for route, stale_values, current_values in zip(
                    batch_routes, batch_stale, batch_current
                )
            ]
            width, stale, current = _adjust_route_positions(enc, route_overrides)
            context = AdaptiveRoutingContext(
                margin=float(margin),
                stale_positions=stale,
                current_positions=current,
                query_positions=tuple(width - 1 for _ in batch_rows),
                heads_by_layer=_heads_by_layer(heads),
                direction=int(direction),
                gate_scale=float(gate_scale),
            )
            with adaptive_routing(context), torch.inference_mode():
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
                        **natural.classify_natural_task_response(row, response),
                    }
                )
    finally:
        tokenizer.padding_side = old_padding_side
    return records, np.concatenate(first_scores, axis=0)


def causal_gradient_discovery(
    model,
    tokenizer,
    rows: list[dict],
    routes: list[dict],
    baseline: list[dict],
    *,
    margin: float,
    max_beta: float,
    batch_size: int,
) -> dict:
    import torch

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = next(model.parameters()).device
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    failure_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    correct_sum = np.zeros_like(failure_sum)
    counts = Counter()
    audit_matches = 0
    audit_total = 0
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_routes = routes[start : start + batch_size]
            batch_baseline = baseline[start : start + batch_size]
            prompts = [
                natural.render_natural_prompt(tokenizer, row["messages"])
                for row in batch_rows
            ]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            width, stale, current = _adjust_route_positions(enc, batch_routes)
            gates = {
                layer: torch.zeros(n_heads, dtype=torch.float32, device=device, requires_grad=True)
                for layer in range(n_layers)
            }
            context = AdaptiveRoutingContext(
                margin=float(margin),
                stale_positions=stale,
                current_positions=current,
                query_positions=tuple(width - 1 for _ in batch_rows),
                heads_by_layer={},
                max_beta=float(max_beta),
                gates_by_layer=gates,
            )
            with adaptive_routing(context):
                output = model(**enc, use_cache=False, return_dict=True)
            logits = output.logits[:, width - 1, :]
            gaps = torch.stack(
                [_candidate_gap(logits[index], tokenizer, row) for index, row in enumerate(batch_rows)]
            )
            pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
            for pred_id, row, record in zip(pred_ids, batch_rows, batch_baseline):
                if record["label"] in ("correct_current", "within_stale"):
                    audit_total += 1
                    audit_matches += int(
                        _candidate_prediction_label(pred_id, tokenizer, row) == record["label"]
                    )

            gate_list = [gates[layer] for layer in range(n_layers)]
            for label, target in (("within_stale", failure_sum), ("correct_current", correct_sum)):
                indices = [index for index, record in enumerate(batch_baseline) if record["label"] == label]
                if not indices:
                    continue
                selected = gaps[torch.as_tensor(indices, dtype=torch.long, device=device)].sum()
                gradients = torch.autograd.grad(
                    selected,
                    gate_list,
                    retain_graph=True,
                    allow_unused=True,
                )
                for layer, gradient in enumerate(gradients):
                    if gradient is not None:
                        target[layer] += gradient.detach().float().cpu().numpy()
                counts[label] += len(indices)
            del output, logits, gaps, gates
            print(f"causal gradients {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    finally:
        tokenizer.padding_side = old_padding_side
    if not counts["within_stale"] or not counts["correct_current"]:
        raise RuntimeError(f"causal discovery pools are inadequate: {dict(counts)}")
    return {
        "failure_gradients": failure_sum / counts["within_stale"],
        "correct_gradients": correct_sum / counts["correct_current"],
        "n_within_stale": counts["within_stale"],
        "n_correct": counts["correct_current"],
        "candidate_audit": {
            "matches": audit_matches,
            "n": audit_total,
            "accuracy": audit_matches / audit_total if audit_total else 0.0,
        },
    }


def _write_report(path: Path, summary: dict) -> None:
    selected = summary.get("validation", {}).get("selected")
    lines = [
        f"# {summary['stage']} causal-head discovery: {summary['model']}",
        "",
        f"**Verdict:** `{summary['verdict']}`.",
        "",
        f"Rows: discovery={summary['n_discovery']}, validation={summary['n_validation']}, "
        f"reserved confirmation={summary['n_confirmation']}.",
        "",
        f"Candidate-token audit: {summary['head_discovery']['candidate_audit']['accuracy']:.3f} "
        f"({summary['head_discovery']['candidate_audit']['matches']}/"
        f"{summary['head_discovery']['candidate_audit']['n']}).",
        f"First-token collision exclusions (discovery only): "
        f"{len(summary['head_discovery'].get('candidate_exclusions', []))}.",
        "",
        "## Causally ranked heads",
        "",
    ]
    for row in summary["head_discovery"]["top_heads"]:
        lines.append(
            f"- L{row['layer']}H{row['head']}: failure gradient="
            f"{row['failure_gradient']:.5f}, correct gradient="
            f"{row['correct_gradient']:.5f}, target mass="
            f"{row['target_attention_mass']:.5f}"
        )
    lines.extend(["", "## Validation", ""])
    if selected is None:
        lines.append("No eligible configuration was selected.")
    else:
        lines.extend(
            [
                f"Selected top-{selected['size']} at margin={selected['margin']}.",
                "",
                f"Validation accuracy: {selected['baseline_accuracy']:.3f} -> "
                f"{selected['arm_accuracy']:.3f}; gain={selected['net_accuracy_gain']:.3f}; "
                f"preservation={selected['correct_preservation']:.3f}; "
                f"within-stale correction={selected['within_stale_correction']:.3f}.",
                "",
                f"Opposite-direction gain: {summary['validation']['opposite']['net_accuracy_gain']:.3f}.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_confirmation_report(path: Path, summary: dict) -> None:
    retrieval = summary["tasks"]["retrieval"]
    lines = [
        f"# {summary['stage']} held-out confirmation: {summary['model']}",
        "",
        f"**Verdict:** `{summary['verdict']}`.",
        "",
        f"Frozen before confirmation: top-{summary['selected']['size']} causal heads, "
        f"adaptive margin={summary['selected']['margin']}.",
        "",
        (
            f"All {summary['n_confirmation']} rows are untouched held-out confirmation rows."
            if not summary["excluded_from_primary_confirmation"]
            else f"All {summary['n_confirmation']} rows were generated; the primary analysis "
            f"uses {summary['n_confirmation_primary']} rows after excluding "
            f"{len(summary['excluded_from_primary_confirmation'])} calibration-overlap row(s)."
        ),
        "",
        "| Task | baseline | routed | gain | 95% CI | preservation | stale correction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task_name, task in summary["tasks"].items():
        metric = task["routed"]
        lines.append(
            f"| {task_name} | {metric['baseline_accuracy']:.3f} | "
            f"{metric['arm_accuracy']:.3f} | {metric['net_accuracy_gain']:.3f} | "
            f"{metric['paired_net_gain']['ci']} | {metric['correct_preservation']:.3f} | "
            f"{metric['within_stale_correction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Retrieval controls",
            "",
            f"- Opposite-direction gain: {retrieval['opposite']['net_accuracy_gain']:.3f}.",
            f"- Explicit-recap gain: {retrieval['recap']['net_accuracy_gain']:.3f}.",
            "- Random-position gain interval: "
            f"{retrieval['random_positions']['net_accuracy_gain']['interval95']}.",
            "- Layer-matched random-head gain interval: "
            f"{retrieval['random_heads']['net_accuracy_gain']['interval95']}.",
            "",
            "## Preregistered retrieval gate",
            "",
        ]
    )
    for name, passed in retrieval["gate"].items():
        lines.append(f"- {name}: {passed}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_CONFIRMATION_TASK_FILES = {
    "baseline": "baseline_rows.jsonl",
    "routed": "routed_rows.jsonl",
    "opposite": "opposite_rows.jsonl",
    "recap": "recap_rows.jsonl",
    "random_positions": "random_position_controls.jsonl",
    "random_heads": "random_head_controls.jsonl",
}


def _load_completed_confirmation_task(
    task_dir: Path,
    rows: list[dict],
    *,
    n_random_positions: int,
    expected_random_head_sets: list[list[tuple[int, int]]],
) -> dict | None:
    """Load only a task checkpoint that reached all hard gates and controls."""
    paths = {name: task_dir / filename for name, filename in _CONFIRMATION_TASK_FILES.items()}
    present = {name for name, path in paths.items() if path.is_file()}
    if not present:
        return None
    if present != set(paths):
        missing = sorted(set(paths) - present)
        raise RuntimeError(
            f"partial confirmation checkpoint in {task_dir}; missing {missing}"
        )

    artifacts = {name: load_jsonl(path) for name, path in paths.items()}
    expected_ids = [row["id"] for row in rows]
    expected_task_types = [row["task_type"] for row in rows]
    for name in ("baseline", "routed", "opposite", "recap"):
        records = artifacts[name]
        if [record.get("id") for record in records] != expected_ids:
            raise RuntimeError(f"{task_dir}/{paths[name].name} has wrong held-out row order")
        if [record.get("task_type") for record in records] != expected_task_types:
            raise RuntimeError(f"{task_dir}/{paths[name].name} has wrong task type")

    position_records = artifacts["random_positions"]
    if [record.get("index") for record in position_records] != list(
        range(n_random_positions)
    ):
        raise RuntimeError(f"{task_dir} has incomplete random-position controls")

    head_records = artifacts["random_heads"]
    if [record.get("index") for record in head_records] != list(
        range(len(expected_random_head_sets))
    ):
        raise RuntimeError(f"{task_dir} has incomplete random-head controls")
    recorded_head_sets = [
        [(int(head["layer"]), int(head["head"])) for head in record.get("heads", [])]
        for record in head_records
    ]
    normalized_expected = [
        [(int(layer), int(head)) for layer, head in heads]
        for heads in expected_random_head_sets
    ]
    if recorded_head_sets != normalized_expected:
        raise RuntimeError(f"{task_dir} random-head controls do not match this run")
    return artifacts


def _identity_diagnostics(
    baseline: list[dict],
    identity: list[dict],
    baseline_scores: np.ndarray,
    identity_scores: np.ndarray,
) -> dict:
    if baseline_scores.shape != identity_scores.shape:
        raise ValueError("identity score arrays must have the same shape")
    finite = np.isfinite(baseline_scores) & np.isfinite(identity_scores)
    differences = np.abs(identity_scores - baseline_scores)
    finite_differences = differences[finite]
    changed_ids = [
        before["id"]
        for before, after in zip(baseline, identity)
        if before["response"] != after["response"]
    ]
    return {
        "max_abs_first_token_score_change": (
            float(finite_differences.max()) if finite_differences.size else None
        ),
        "different_finite_score_entries": int(
            np.count_nonzero(finite_differences != 0.0)
        ),
        "baseline_nonfinite_score_entries": int(
            np.count_nonzero(~np.isfinite(baseline_scores))
        ),
        "identity_nonfinite_score_entries": int(
            np.count_nonzero(~np.isfinite(identity_scores))
        ),
        "score_arrays_exact": bool(
            np.array_equal(baseline_scores, identity_scores, equal_nan=True)
            and np.isfinite(baseline_scores).all()
            and np.isfinite(identity_scores).all()
        ),
        "identity_response_changes": len(changed_ids),
        "changed_response_ids": changed_ids[:20],
    }


def _identity_gate_passes(diagnostic: dict) -> bool:
    return bool(
        diagnostic["score_arrays_exact"]
        and int(diagnostic["identity_response_changes"]) == 0
    )


def _summarize_confirmation_task(
    *,
    rows: list[dict],
    evaluation_indices: list[int],
    artifacts: dict,
    task_index: int,
    args,
    route_audit_exact: bool,
    identity_exact: bool,
    identity_score_change: float,
    identity_response_changes: int,
    routed_score_change: float | None,
    resumed_from: Path | None,
) -> dict:
    def held_out(records):
        return [records[index] for index in evaluation_indices]

    evaluation_rows = [rows[index] for index in evaluation_indices]
    baseline = artifacts["baseline"]
    routed = artifacts["routed"]
    opposite = artifacts["opposite"]
    recap = artifacts["recap"]
    routed_metric = natural._metrics(
        evaluation_rows,
        held_out(baseline),
        held_out(routed),
        args.n_boot,
        args.seed + 10 * task_index,
    )
    opposite_metric = natural._metrics(
        evaluation_rows,
        held_out(baseline),
        held_out(opposite),
        args.n_boot,
        args.seed + 100 + 10 * task_index,
    )
    recap_metric = natural._metrics(
        evaluation_rows,
        held_out(baseline),
        held_out(recap),
        args.n_boot,
        args.seed + 200 + 10 * task_index,
    )
    per_cell = natural._per_cell(
        evaluation_rows, held_out(baseline), held_out(routed)
    )
    all_rows_sensitivity = natural._metrics(
        rows,
        baseline,
        routed,
        args.n_boot,
        args.seed + 900 + 10 * task_index,
    )
    random_position_summary = natural._control_summary(artifacts["random_positions"])
    random_head_summary = natural._control_summary(artifacts["random_heads"])
    gate = confirmation_gate(
        routed_metric,
        per_cell,
        random_position_summary,
        random_head_summary,
        route_audit_exact=route_audit_exact,
        identity_exact=identity_exact,
        identity_required=not args.exclude_identity_from_gate,
        minimum_pool=args.minimum_pool,
        minimum_preservation=args.minimum_preservation,
    )
    return {
        "routed": routed_metric,
        "all_960_rows_sensitivity": all_rows_sensitivity,
        "opposite": opposite_metric,
        "recap": recap_metric,
        "per_factorial_cell": per_cell,
        "random_positions": random_position_summary,
        "random_heads": random_head_summary,
        "identity_max_abs_first_token_score_change": identity_score_change,
        "identity_response_changes": identity_response_changes,
        "routed_max_abs_first_token_score_change": routed_score_change,
        "gate": gate,
        "execution": {
            "resumed_from_complete_task_artifacts": (
                str(resumed_from) if resumed_from is not None else None
            ),
            "resume_validation": (
                "All arm row IDs/task types and all preregistered control indices/head sets matched. "
                "Identity exactness follows from the original writer ordering: task artifacts are "
                "written only after the identity hard gate passes."
                if resumed_from is not None
                else None
            ),
            "identity_gate_override_used": bool(
                not identity_exact
                and (
                    args.continue_after_identity_failure
                    or args.exclude_identity_from_gate
                )
            ),
            "identity_excluded_from_gate": bool(args.exclude_identity_from_gate),
        },
    }


def _run_confirmation(args) -> None:
    import torch

    if not args.frozen_summary:
        raise ValueError("--confirmation-only requires --frozen-summary")
    frozen = json.loads(Path(args.frozen_summary).read_text(encoding="utf-8"))
    if frozen.get("verdict") != "validation_gate_passed_confirmation_not_run":
        raise RuntimeError("frozen R12 validation did not pass")
    selected = frozen["validation"]["selected"]
    if selected is None:
        raise RuntimeError("frozen R12 summary has no selected configuration")
    selected_heads = [
        (int(row["layer"]), int(row["head"]))
        for row in frozen["head_discovery"]["top_heads"][: int(selected["size"])]
    ]
    margin = float(selected["margin"])

    install_adaptive_patch(args.model_family)
    if args.limit_per_cell is not None:
        raise ValueError("confirmation may not use --limit-per-cell")
    base_rows = load_jsonl(args.data)
    calibration_base, confirmation_base = split_r12_rows(
        base_rows, args.calibration_fraction
    )
    frozen_dir = Path(args.frozen_summary).parent
    frozen_rows = load_jsonl(frozen_dir / "discovery/baseline_rows.jsonl") + load_jsonl(
        frozen_dir / "validation/baseline_rows.jsonl"
    )
    confirmation_ids = {row["id"] for row in confirmation_base}
    overlap_ids = sorted(
        {row["id"] for row in frozen_rows}.intersection(confirmation_ids)
    )
    if len(calibration_base) != int(frozen["n_calibration"]):
        if not args.allow_one_row_confirmation_overlap or len(overlap_ids) != 1:
            raise RuntimeError(
                "frozen calibration differs from the canonical split; expected the "
                "explicit one-row-overlap override"
            )
    elif overlap_ids:
        raise RuntimeError("canonical frozen calibration unexpectedly overlaps confirmation")
    if args.limit_per_cell is None and len(confirmation_base) != 960:
        raise RuntimeError(f"expected 960 untouched confirmation rows, found {len(confirmation_base)}")

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    validate_model_family(model.config.model_type, args.model_family)
    audit, _ = natural.audit_natural_routes(base_rows, tokenizer)
    if audit["mismatches"]:
        raise RuntimeError("R12 confirmation route audit failed")
    random_head_sets = make_layer_matched_random_head_sets(
        selected_heads,
        int(model.config.num_attention_heads),
        args.n_random_heads,
        args.seed + 7000,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_from_dir and not args.resume_completed_tasks:
        raise ValueError("--resume-from-dir requires --resume-completed-tasks")
    resume_root = Path(args.resume_from_dir) if args.resume_from_dir else out_dir
    task_summaries = {}
    for task_index, (task_name, rows) in enumerate(
        natural.prepare_task_rows(confirmation_base).items()
    ):
        task_dir = out_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        routes = [natural.infer_natural_route(row["messages"], tokenizer) for row in rows]
        evaluation_indices = [
            index for index, row in enumerate(rows) if row["id"] not in overlap_ids
        ]
        evaluation_rows = [rows[index] for index in evaluation_indices]

        def held_out(records):
            return [records[index] for index in evaluation_indices]

        resume_task_dir = resume_root / task_name
        completed = None
        if args.resume_completed_tasks:
            completed = _load_completed_confirmation_task(
                resume_task_dir,
                rows,
                n_random_positions=args.n_random_positions,
                expected_random_head_sets=random_head_sets,
            )
        if completed is not None:
            if resume_task_dir.resolve() != task_dir.resolve():
                for name, filename in _CONFIRMATION_TASK_FILES.items():
                    dump_jsonl(task_dir / filename, completed[name])
            task_summary = _summarize_confirmation_task(
                rows=rows,
                evaluation_indices=evaluation_indices,
                artifacts=completed,
                task_index=task_index,
                args=args,
                route_audit_exact=not audit["mismatches"],
                identity_exact=True,
                identity_score_change=0.0,
                identity_response_changes=0,
                routed_score_change=None,
                resumed_from=resume_task_dir,
            )
            dump_json(task_dir / "task_summary.json", task_summary)
            task_summaries[task_name] = task_summary
            print(f"{task_name} resumed from {resume_task_dir}", flush=True)
            continue

        baseline, baseline_scores = _generate_adaptive_arm(
            model, tokenizer, rows, routes, [], 0.0, args.batch_size,
            args.max_new_tokens, gate_scale=0.0,
        )
        identity, identity_scores = _generate_adaptive_arm(
            model, tokenizer, rows, routes, selected_heads, margin, args.batch_size,
            args.max_new_tokens, gate_scale=0.0,
        )
        identity_diagnostic = _identity_diagnostics(
            baseline, identity, baseline_scores, identity_scores
        )
        dump_json(task_dir / "identity_diagnostic.json", identity_diagnostic)
        identity_score_change = identity_diagnostic[
            "max_abs_first_token_score_change"
        ]
        if identity_score_change is None:
            identity_score_change = float("inf")
        identity_response_changes = identity_diagnostic["identity_response_changes"]
        identity_exact = _identity_gate_passes(identity_diagnostic)
        if args.identity_diagnostic_only:
            repeated_baseline, repeated_baseline_scores = _generate_adaptive_arm(
                model, tokenizer, rows, routes, [], 0.0, args.batch_size,
                args.max_new_tokens, gate_scale=0.0,
            )
            repeated_baseline_diagnostic = _identity_diagnostics(
                baseline,
                repeated_baseline,
                baseline_scores,
                repeated_baseline_scores,
            )
            identity_vs_repeat_diagnostic = _identity_diagnostics(
                repeated_baseline,
                identity,
                repeated_baseline_scores,
                identity_scores,
            )
            diagnostic_payload = {
                "task": task_name,
                "baseline_vs_identity": identity_diagnostic,
                "baseline_vs_repeat": repeated_baseline_diagnostic,
                "repeat_vs_identity": identity_vs_repeat_diagnostic,
            }
            dump_json(task_dir / "identity_repeat_diagnostic.json", diagnostic_payload)
            print(
                json.dumps(diagnostic_payload, indent=2, sort_keys=True),
                flush=True,
            )
            return
        allow_identity_failure = bool(
            args.continue_after_identity_failure or args.exclude_identity_from_gate
        )
        if not identity_exact and not allow_identity_failure:
            raise RuntimeError(
                f"R12 confirmation identity gate failed for {task_name}: "
                f"{json.dumps(identity_diagnostic, sort_keys=True)}"
            )
        if not identity_exact:
            print(
                "WARNING: continuing after identity-gate failure for descriptive "
                f"targeted-vs-random analysis: {json.dumps(identity_diagnostic, sort_keys=True)}",
                flush=True,
            )

        routed, routed_scores = _generate_adaptive_arm(
            model, tokenizer, rows, routes, selected_heads, margin, args.batch_size,
            args.max_new_tokens,
        )
        opposite, _ = _generate_adaptive_arm(
            model, tokenizer, rows, routes, selected_heads, margin, args.batch_size,
            args.max_new_tokens, direction=-1,
        )
        recap, _ = _generate_adaptive_arm(
            model, tokenizer, rows, routes, [], 0.0, args.batch_size,
            args.max_new_tokens, gate_scale=0.0, recap=True,
        )

        position_controls = natural.make_random_balanced_position_sets(
            rows, routes, tokenizer, args.n_random_positions,
            args.seed + 1000 * (task_index + 1),
        )
        random_position_records = []
        for index, control in enumerate(position_controls):
            arm, _ = _generate_adaptive_arm(
                model, tokenizer, rows, routes, selected_heads, margin,
                args.batch_size, args.max_new_tokens,
                stale_positions=control["negative"],
                current_positions=control["positive"],
            )
            metric = natural._metrics(
                evaluation_rows,
                held_out(baseline),
                held_out(arm),
                min(args.n_boot, 200),
                args.seed + 3000 + index,
            )
            random_position_records.append(
                {
                    "index": index,
                    "net_accuracy_gain": metric["net_accuracy_gain"],
                    "correct_preservation": metric["correct_preservation"],
                }
            )
            print(
                f"{task_name} random-position {index + 1}/{len(position_controls)}",
                flush=True,
            )

        random_head_records = []
        for index, heads in enumerate(random_head_sets):
            arm, _ = _generate_adaptive_arm(
                model, tokenizer, rows, routes, heads, margin, args.batch_size,
                args.max_new_tokens,
            )
            metric = natural._metrics(
                evaluation_rows,
                held_out(baseline),
                held_out(arm),
                min(args.n_boot, 200),
                args.seed + 5000 + index,
            )
            random_head_records.append(
                {
                    "index": index,
                    "heads": [
                        {"layer": int(layer), "head": int(head)}
                        for layer, head in heads
                    ],
                    "net_accuracy_gain": metric["net_accuracy_gain"],
                    "correct_preservation": metric["correct_preservation"],
                }
            )
            print(
                f"{task_name} random-head {index + 1}/{len(random_head_sets)}",
                flush=True,
            )

        dump_jsonl(task_dir / "baseline_rows.jsonl", baseline)
        dump_jsonl(task_dir / "routed_rows.jsonl", routed)
        dump_jsonl(task_dir / "opposite_rows.jsonl", opposite)
        dump_jsonl(task_dir / "recap_rows.jsonl", recap)
        dump_jsonl(task_dir / "random_position_controls.jsonl", random_position_records)
        dump_jsonl(task_dir / "random_head_controls.jsonl", random_head_records)
        artifacts = {
            "baseline": baseline,
            "routed": routed,
            "opposite": opposite,
            "recap": recap,
            "random_positions": random_position_records,
            "random_heads": random_head_records,
        }
        task_summary = _summarize_confirmation_task(
            rows=rows,
            evaluation_indices=evaluation_indices,
            artifacts=artifacts,
            task_index=task_index,
            args=args,
            route_audit_exact=not audit["mismatches"],
            identity_exact=identity_exact,
            identity_score_change=identity_score_change,
            identity_response_changes=identity_response_changes,
            routed_score_change=float(np.max(np.abs(routed_scores - baseline_scores))),
            resumed_from=None,
        )
        dump_json(task_dir / "task_summary.json", task_summary)
        task_summaries[task_name] = task_summary

    retrieval_pass = task_summaries["retrieval"]["gate"]["all_pass"]
    derived_pass = task_summaries["derived_decision"]["gate"]["all_pass"]
    if retrieval_pass and derived_pass:
        verdict = "retrieval_and_derived_correction_supported"
    elif retrieval_pass:
        verdict = "retrieval_correction_only"
    else:
        verdict = "confirmation_gate_failed"
    summary = {
        "stage": f"{args.stage_label}-causal-head-routing-confirmation",
        "verdict": verdict,
        "model": args.model_label,
        "data": args.data,
        "frozen_summary": args.frozen_summary,
        "selected": {
            "size": int(selected["size"]),
            "margin": margin,
            "heads": [
                {"layer": int(layer), "head": int(head)}
                for layer, head in selected_heads
            ],
        },
        "n_base_rows": len(base_rows),
        "n_calibration": len(calibration_base),
        "n_confirmation": len(confirmation_base),
        "n_confirmation_primary": len(confirmation_base) - len(overlap_ids),
        "excluded_from_primary_confirmation": overlap_ids,
        "route_audit": audit,
        "n_random_positions": args.n_random_positions,
        "n_random_heads": args.n_random_heads,
        "tasks": task_summaries,
        "confirmation_opened": True,
        "scope": (
            "Model-specific causal-head routing evaluated on the canonical held-out "
            "confirmation split; derived decision is transfer-only"
        ),
    }
    dump_json(out_dir / "summary.json", summary)
    _write_confirmation_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"verdict": verdict, "retrieval_gate": task_summaries["retrieval"]["gate"]}, indent=2), flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args) -> None:
    import torch

    install_adaptive_patch(args.model_family)
    base_rows = load_jsonl(args.data)
    calibration_base, confirmation_base = split_r12_rows(
        base_rows, args.calibration_fraction, args.limit_per_cell
    )
    discovery_base, validation_base = split_calibration_discovery_validation(
        calibration_base
    )
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    validate_model_family(model.config.model_type, args.model_family)
    audit, _ = natural.audit_natural_routes(calibration_base, tokenizer)
    if audit["mismatches"]:
        raise RuntimeError("R12 route audit failed")

    discovery_rows = natural.prepare_task_rows(discovery_base)["retrieval"]
    discovery_routes = [
        natural.infer_natural_route(row["messages"], tokenizer)
        for row in discovery_rows
    ]
    baseline, _ = _generate_adaptive_arm(
        model,
        tokenizer,
        discovery_rows,
        discovery_routes,
        [],
        args.discovery_margin,
        args.batch_size,
        args.max_new_tokens,
        gate_scale=0.0,
    )
    stale_attention, current_attention = capture_head_attention(
        model,
        tokenizer,
        discovery_rows,
        discovery_routes,
        args.capture_batch_size,
        prompt_renderer=natural.render_natural_prompt,
    )
    gradient_rows, gradient_routes, gradient_baseline, candidate_exclusions = (
        partition_candidate_discovery(
            tokenizer, discovery_rows, discovery_routes, baseline
        )
    )
    gradients = causal_gradient_discovery(
        model,
        tokenizer,
        gradient_rows,
        gradient_routes,
        gradient_baseline,
        margin=args.discovery_margin,
        max_beta=args.max_beta,
        batch_size=args.gradient_batch_size,
    )
    if gradients["candidate_audit"]["accuracy"] < args.minimum_candidate_audit:
        raise RuntimeError(
            "candidate-token audit below threshold: "
            f"{gradients['candidate_audit']['accuracy']:.3f}"
        )
    minimum_correct = (
        args.minimum_correct_discovery_pool
        if args.minimum_correct_discovery_pool is not None
        else args.minimum_discovery_pool
    )
    minimum_within_stale = (
        args.minimum_stale_discovery_pool
        if args.minimum_stale_discovery_pool is not None
        else args.minimum_discovery_pool
    )
    pool_gate = discovery_pool_gate(
        gradients["n_correct"],
        gradients["n_within_stale"],
        minimum_correct,
        minimum_within_stale,
    )
    failure_mask = np.asarray([row["label"] == "within_stale" for row in baseline])
    if not pool_gate["all_pass"]:
        raise RuntimeError(
            "discovery pool gate failed after first-token collision exclusions: "
            f"correct={gradients['n_correct']}/{minimum_correct}, "
            f"within_stale={gradients['n_within_stale']}/{minimum_within_stale}"
        )
    target_mass = (stale_attention + current_attention)[failure_mask].mean(axis=0)
    ranked = rank_causal_heads(
        gradients["failure_gradients"],
        gradients["correct_gradients"],
        target_mass,
    )
    positive = [row for row in ranked if row["failure_gradient"] > 0.0]
    if len(positive) < 8:
        raise RuntimeError(f"R12 found only {len(positive)} positive causal heads")
    candidates = positive[: args.candidate_count]

    out_dir = Path(args.out_dir)
    discovery_dir = out_dir / "discovery"
    validation_dir = out_dir / "validation"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(discovery_dir / "baseline_rows.jsonl", baseline)
    np.savez_compressed(
        discovery_dir / "head_arrays.npz",
        failure_gradients=gradients["failure_gradients"].astype(np.float32),
        correct_gradients=gradients["correct_gradients"].astype(np.float32),
        target_attention_mass=target_mass.astype(np.float32),
    )
    dump_json(
        discovery_dir / "head_summary.json",
        {
            "rule": "identity-point gradient of current-minus-strongest-stale first-token logit gap",
            "discovery_margin": args.discovery_margin,
            "max_beta": args.max_beta,
            "n_correct": gradients["n_correct"],
            "n_within_stale": gradients["n_within_stale"],
            "candidate_audit": gradients["candidate_audit"],
            "candidate_exclusions": candidate_exclusions,
            "pool_gate": pool_gate,
            "top_heads": candidates,
            "all_heads": ranked,
        },
    )

    validation_rows = natural.prepare_task_rows(validation_base)["retrieval"]
    validation_routes = [
        natural.infer_natural_route(row["messages"], tokenizer)
        for row in validation_rows
    ]
    validation_baseline, validation_scores = _generate_adaptive_arm(
        model,
        tokenizer,
        validation_rows,
        validation_routes,
        [],
        0.0,
        args.batch_size,
        args.max_new_tokens,
        gate_scale=0.0,
    )
    dump_jsonl(validation_dir / "baseline_rows.jsonl", validation_baseline)
    curve = []
    arm_records = []
    for size in args.head_sizes:
        if size > len(candidates):
            continue
        heads = [(row["layer"], row["head"]) for row in candidates[:size]]
        for margin in args.margins:
            arm, scores = _generate_adaptive_arm(
                model,
                tokenizer,
                validation_rows,
                validation_routes,
                heads,
                margin,
                args.batch_size,
                args.max_new_tokens,
            )
            metric = natural._metrics(
                validation_rows,
                validation_baseline,
                arm,
                min(args.n_boot, 500),
                args.seed + 100 * size + int(10 * margin),
            )
            curve.append(
                {
                    "size": size,
                    "margin": margin,
                    "baseline_accuracy": metric["baseline_accuracy"],
                    "arm_accuracy": metric["arm_accuracy"],
                    "net_accuracy_gain": metric["net_accuracy_gain"],
                    "correct_preservation": metric["correct_preservation"],
                    "within_stale_correction": metric["within_stale_correction"],
                    "max_abs_first_token_score_change": float(
                        np.max(np.abs(scores - validation_scores))
                    ),
                }
            )
            for record in arm:
                arm_records.append({"size": size, "margin": margin, **record})
            print(f"validation size={size} margin={margin}", flush=True)
    dump_jsonl(validation_dir / "all_arm_rows.jsonl", arm_records)

    selected = None
    opposite_metric = None
    verdict = "validation_gate_failed"
    try:
        selected = choose_validation_configuration(
            curve, minimum_preservation=args.minimum_preservation
        )
        selected_heads = [
            (row["layer"], row["head"])
            for row in candidates[: int(selected["size"])]
        ]
        selected_arm, selected_scores = _generate_adaptive_arm(
            model,
            tokenizer,
            validation_rows,
            validation_routes,
            selected_heads,
            float(selected["margin"]),
            args.batch_size,
            args.max_new_tokens,
        )
        identity, identity_scores = _generate_adaptive_arm(
            model,
            tokenizer,
            validation_rows,
            validation_routes,
            selected_heads,
            float(selected["margin"]),
            args.batch_size,
            args.max_new_tokens,
            gate_scale=0.0,
        )
        if any(a["response"] != b["response"] for a, b in zip(validation_baseline, identity)):
            raise RuntimeError("R12 identity response gate failed")
        if float(np.max(np.abs(identity_scores - validation_scores))) != 0.0:
            raise RuntimeError("R12 identity score gate failed")
        opposite, _ = _generate_adaptive_arm(
            model,
            tokenizer,
            validation_rows,
            validation_routes,
            selected_heads,
            float(selected["margin"]),
            args.batch_size,
            args.max_new_tokens,
            direction=-1,
        )
        opposite_metric = natural._metrics(
            validation_rows,
            validation_baseline,
            opposite,
            min(args.n_boot, 500),
            args.seed + 999,
        )
        dump_jsonl(validation_dir / "selected_rows.jsonl", selected_arm)
        dump_jsonl(validation_dir / "opposite_rows.jsonl", opposite)
        if (
            selected["within_stale_correction"] > 0.0
            and selected["net_accuracy_gain"] > opposite_metric["net_accuracy_gain"]
        ):
            verdict = "validation_gate_passed_confirmation_not_run"
    except ValueError:
        pass

    summary = {
        "stage": f"{args.stage_label}-causal-head-routing",
        "verdict": verdict,
        "model": args.model_label,
        "data": args.data,
        "n_base_rows": len(base_rows),
        "n_calibration": len(calibration_base),
        "n_discovery": len(discovery_rows),
        "n_validation": len(validation_rows),
        "n_confirmation": len(confirmation_base),
        "route_audit": audit,
        "split_order": "canonical 240/960 split before optional calibration-only limiting",
        "head_discovery": {
            "candidate_audit": gradients["candidate_audit"],
            "n_correct": gradients["n_correct"],
            "n_within_stale": gradients["n_within_stale"],
            "candidate_exclusions": candidate_exclusions,
            "pool_gate": pool_gate,
            "top_heads": candidates,
        },
        "validation": {
            "curve": curve,
            "selected": selected,
            "opposite": opposite_metric,
        },
        "confirmation_opened": False,
        "scope": "calibration-only discovery/validation; no held-out correction claim",
    }
    dump_json(out_dir / "summary.json", summary)
    _write_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"verdict": verdict, "selected": selected}, indent=2), flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--model-family",
        choices=("llama", "mistral", "qwen2", "gemma2"),
        default="llama",
    )
    parser.add_argument("--stage-label", default="R12-llama")
    parser.add_argument(
        "--data",
        default="data/cicm/cicm_natural_factorial_otherdist_l0.jsonl",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--discovery-margin", type=float, default=1.0)
    parser.add_argument("--max-beta", type=float, default=8.0)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--head-sizes", type=lambda value: [int(x) for x in value.split(",")], default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--margins", type=lambda value: [float(x) for x in value.split(",")], default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--minimum-preservation", type=float, default=0.95)
    parser.add_argument("--minimum-candidate-audit", type=float, default=0.95)
    parser.add_argument("--minimum-discovery-pool", type=int, default=5)
    parser.add_argument("--minimum-correct-discovery-pool", type=int)
    parser.add_argument("--minimum-stale-discovery-pool", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--capture-batch-size", type=int, default=1)
    parser.add_argument("--gradient-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-random-positions", type=int, default=16)
    parser.add_argument("--n-random-heads", type=int, default=64)
    parser.add_argument("--minimum-pool", type=int, default=100)
    parser.add_argument("--limit-per-cell", type=int)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--confirmation-only", action="store_true")
    parser.add_argument("--frozen-summary")
    parser.add_argument("--resume-completed-tasks", action="store_true")
    parser.add_argument("--resume-from-dir")
    parser.add_argument("--identity-diagnostic-only", action="store_true")
    parser.add_argument("--continue-after-identity-failure", action="store_true")
    parser.add_argument("--exclude-identity-from-gate", action="store_true")
    parser.add_argument("--allow-one-row-confirmation-overlap", action="store_true")
    args = parser.parse_args()
    if args.confirmation_only:
        _run_confirmation(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
