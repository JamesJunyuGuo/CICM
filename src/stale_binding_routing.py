"""Stage R9: update-aware stale-key suppression at test time."""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pythia_circuit import stable_split
from pythia_eval import dump_json, load_model
from pythia_gen import dump_jsonl, load_jsonl
from stale_binding_attenuation import (
    _limited_calibration,
    _limited_confirmation,
    _metrics_with_ci,
    _per_template,
    _random_summary,
    _stable_calibration,
    make_layer_matched_random_sets,
)
from stale_binding_correction import _record_predictions, load_heads, protocol_split


DEFAULT_BETAS = (0.5, 1.0, 2.0, 4.0, 8.0)
_ROUTING_CONTEXT = contextvars.ContextVar("stage_r_routing_context", default=None)
_QWEN2_PATCHED = False


@dataclass(frozen=True)
class RoutingBiasContext:
    beta: float
    key_positions: tuple[tuple[int, ...], ...]
    query_positions: tuple[int, ...]
    heads_by_layer: dict[int, tuple[int, ...]]
    boost_beta: float = 0.0
    boost_positions: tuple[tuple[int, ...], ...] = ()


def apply_stale_key_bias(attn_scores, layer_idx: int, ctx: RoutingBiasContext | None):
    """Subtract beta only from selected answer-query/head/key score cells."""
    if ctx is None or (float(ctx.beta) == 0.0 and float(ctx.boost_beta) == 0.0):
        return attn_scores
    heads = ctx.heads_by_layer.get(int(layer_idx), ())
    if not heads or not (any(ctx.key_positions) or any(ctx.boost_positions)):
        return attn_scores
    if len(ctx.key_positions) != attn_scores.shape[0]:
        raise ValueError("routing key positions do not match attention batch")
    if len(ctx.query_positions) != attn_scores.shape[0]:
        raise ValueError("routing query positions do not match attention batch")
    if ctx.boost_positions and len(ctx.boost_positions) != attn_scores.shape[0]:
        raise ValueError("routing boost positions do not match attention batch")
    if all(int(position) >= attn_scores.shape[-2] for position in ctx.query_positions):
        return attn_scores

    out = attn_scores.clone()
    boost_positions = ctx.boost_positions or tuple(() for _ in ctx.key_positions)
    for batch_index, (query_position, key_positions, positive_positions) in enumerate(
        zip(ctx.query_positions, ctx.key_positions, boost_positions)
    ):
        if not key_positions and not positive_positions:
            continue
        query_position = int(query_position)
        # Generation reuses this context for cached one-token decode calls. The
        # intervention is defined only on the full prompt prefill query.
        if query_position >= out.shape[-2]:
            continue
        if not 0 <= query_position < out.shape[-2]:
            raise ValueError(f"query position {query_position} outside attention tensor")
        selected_positions = tuple(key_positions) + tuple(positive_positions)
        key_index = np.asarray(selected_positions, dtype=np.int64)
        if key_index.min() < 0 or key_index.max() >= out.shape[-1]:
            raise ValueError("routing key position outside attention tensor")
        for head in heads:
            if key_positions:
                out[batch_index, int(head), query_position, key_positions] -= float(ctx.beta)
            if positive_positions:
                out[batch_index, int(head), query_position, positive_positions] += float(ctx.boost_beta)
    return out


def install_qwen2_routing_patch() -> None:
    global _QWEN2_PATCHED
    if _QWEN2_PATCHED:
        return

    import torch
    import torch.nn as nn
    import transformers.models.qwen2.modeling_qwen2 as qwen2

    def routing_eager_attention_forward(
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
        key_states = qwen2.repeat_kv(key, module.num_key_value_groups)
        value_states = qwen2.repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = apply_stale_key_bias(
            attn_weights, module.layer_idx, _ROUTING_CONTEXT.get()
        )
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output.transpose(1, 2).contiguous(), attn_weights

    qwen2.eager_attention_forward = routing_eager_attention_forward
    _QWEN2_PATCHED = True


@contextlib.contextmanager
def routing_bias(ctx: RoutingBiasContext):
    token = _ROUTING_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _ROUTING_CONTEXT.reset(token)


_QUERY_PATTERNS = (
    (re.compile(r"(?m)^=>\s*([A-Za-z][\w-]*)\s*=\s*$"), "Assignments:"),
    (re.compile(r"(?m)^Current\s+([A-Za-z][\w-]*)\s*=\s*$"), "Record:"),
    (
        re.compile(r"(?m)^Latest value of\s+([A-Za-z][\w-]*)\s*=\s*$"),
        "Updates in time order:",
    ),
)
_ASSIGNMENT = re.compile(
    r"(?m)^([A-Za-z][\w-]*)\s*=\s*([A-Za-z][\w-]*)\s*$"
)


def _char_span_to_tokens(offsets, start: int, end: int) -> list[int]:
    positions = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]
    if not positions:
        raise ValueError(f"character span {start}:{end} maps to no tokens")
    return positions


def infer_structured_route(prompt: str, tokenizer) -> dict:
    """Infer superseded value-token positions from raw structured prompt only."""
    matches = []
    for pattern, header in _QUERY_PATTERNS:
        found = list(pattern.finditer(prompt))
        if found:
            matches.append((found[-1], header))
    if len(matches) != 1:
        raise ValueError(f"expected one current-state query, found {len(matches)}")
    query_match, header = matches[0]
    target_var = query_match.group(1)
    block_start = prompt.rfind(header, 0, query_match.start())
    if block_start < 0:
        raise ValueError(f"missing final block header {header!r}")

    writes = []
    for match in _ASSIGNMENT.finditer(prompt, block_start, query_match.start()):
        if match.group(1) == target_var:
            writes.append(
                {
                    "value": match.group(2),
                    "char_start": match.start(2),
                    "char_end": match.end(2),
                }
            )
    if not writes:
        raise ValueError(f"no writes found for queried variable {target_var!r}")

    encoded = tokenizer(
        prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = encoded["offset_mapping"]
    for write in writes:
        write["token_positions"] = _char_span_to_tokens(
            offsets, write["char_start"], write["char_end"]
        )
    return {
        "target_var": target_var,
        "values": [write["value"] for write in writes],
        "stale_positions": [
            position for write in writes[:-1] for position in write["token_positions"]
        ],
        "current_positions": list(writes[-1]["token_positions"]),
    }


def audit_routes(rows: list[dict], tokenizer) -> dict:
    mismatches = []
    routes = []
    for row in rows:
        route = infer_structured_route(row["prompt"], tokenizer)
        routes.append(route)
        expected_stale = [int(value) for value in row.get("stale_value_spans", [])]
        expected_current = [int(value) for value in row["current_value_span"]]
        if (
            route["target_var"] != row["target_var"]
            or route["stale_positions"] != expected_stale
            or route["current_positions"] != expected_current
        ):
            mismatches.append(
                {
                    "id": row["id"],
                    "inferred_target": route["target_var"],
                    "expected_target": row["target_var"],
                    "inferred_stale": route["stale_positions"],
                    "expected_stale": expected_stale,
                    "inferred_current": route["current_positions"],
                    "expected_current": expected_current,
                }
            )
    return {
        "n": len(rows),
        "matches": len(rows) - len(mismatches),
        "accuracy": (len(rows) - len(mismatches)) / len(rows) if rows else 0.0,
        "mismatches": mismatches,
        "routes": routes,
    }


def choose_beta(rows: list[dict], minimum_preservation: float = 0.95) -> dict:
    eligible = [
        dict(row)
        for row in rows
        if float(row["correct_preservation"]) >= minimum_preservation
    ]
    if not eligible:
        raise ValueError("no beta satisfies the calibration preservation constraint")
    return max(
        eligible,
        key=lambda row: (float(row["net_accuracy_gain"]), -float(row["beta"])),
    )


def _heads_by_layer(heads: list[tuple[int, int]]) -> dict[int, tuple[int, ...]]:
    grouped = {}
    for layer, head in heads:
        grouped.setdefault(int(layer), []).append(int(head))
    return {layer: tuple(values) for layer, values in grouped.items()}


def make_random_position_sets(
    rows: list[dict], tokenizer, *, n_random: int, seed: int
) -> list[list[tuple[int, ...]]]:
    output = []
    seen = set()
    attempt = 0
    while len(output) < n_random:
        attempt += 1
        if attempt > 10_000:
            raise RuntimeError("could not construct unique random-position controls")
        positions_by_row = []
        for row in rows:
            token_count = len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
            excluded = set(int(value) for value in row.get("stale_value_spans", []))
            excluded.update(int(value) for value in row.get("current_value_span", []))
            excluded.update(int(value) for value in row.get("target_identity_spans", []))
            for write in row.get("writes", []):
                excluded.add(int(write["value_token"]))
                excluded.add(int(write["var_token"]))
            valid = [index for index in range(token_count - 1) if index not in excluded]
            count = len(row.get("stale_value_spans", []))
            if len(valid) < count:
                raise ValueError(f"not enough control positions for id={row['id']}")
            row_seed = int.from_bytes(
                hashlib.sha256(f"{seed}|{attempt}|{row['id']}".encode()).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(row_seed)
            selected = tuple(sorted(int(value) for value in rng.choice(valid, count, replace=False)))
            positions_by_row.append(selected)
        key = tuple(positions_by_row)
        if key in seen:
            continue
        seen.add(key)
        output.append(positions_by_row)
    return output


def evaluate_routing(
    model,
    tokenizer,
    device,
    rows,
    heads,
    beta,
    key_positions,
    batch_size,
    *,
    return_logits=False,
):
    import torch

    records = []
    query_logits = []
    heads_by_layer = _heads_by_layer(heads)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_positions = key_positions[start : start + batch_size]
        enc = tokenizer(
            [row["prompt"] for row in batch_rows],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        query = enc["attention_mask"].sum(dim=1) - 1
        ctx = RoutingBiasContext(
            beta=float(beta),
            key_positions=tuple(tuple(int(value) for value in values) for values in batch_positions),
            query_positions=tuple(int(value) for value in query.detach().cpu().tolist()),
            heads_by_layer=heads_by_layer,
        )
        with routing_bias(ctx), torch.no_grad():
            output = model(**enc, use_cache=False, return_dict=True)
        batch = torch.arange(len(batch_rows), device=device)
        logits = output.logits[batch, query]
        records.extend(_record_predictions(batch_rows, logits, tokenizer))
        if return_logits:
            query_logits.append(logits.detach().float().cpu().numpy())
    logits = np.concatenate(query_logits) if return_logits else None
    return records, logits


def _curve(model, tokenizer, device, rows, baseline, heads, positions, betas, batch_size):
    curve = []
    for beta in betas:
        arm, _ = evaluate_routing(
            model, tokenizer, device, rows, heads, beta, positions, batch_size
        )
        metric = _metrics_with_ci(rows, baseline, arm, n_boot=200, seed=7300 + int(beta * 10))
        curve.append(
            {
                "beta": float(beta),
                **{
                    key: metric[key]
                    for key in (
                        "correction",
                        "correct_preservation",
                        "baseline_accuracy",
                        "arm_accuracy",
                        "net_accuracy_gain",
                    )
                },
            }
        )
    return curve


def _control_metrics(
    model,
    tokenizer,
    device,
    rows,
    baseline,
    controls,
    beta,
    batch_size,
    *,
    heads=None,
    positions=None,
):
    records = []
    for index, control in enumerate(controls):
        control_heads = control if heads is None else heads
        control_positions = control if positions is None else positions
        arm, _ = evaluate_routing(
            model,
            tokenizer,
            device,
            rows,
            control_heads,
            beta,
            control_positions,
            batch_size,
        )
        metric = _metrics_with_ci(rows, baseline, arm, n_boot=200, seed=8100 + index)
        records.append(
            {
                "index": index,
                "net_accuracy_gain": metric["net_accuracy_gain"],
                "correction": metric["correction"],
                "correct_preservation": metric["correct_preservation"],
            }
        )
        print(f"control {index + 1}/{len(controls)}", flush=True)
    return records


def _summarize_controls(records: list[dict]) -> dict:
    return {
        "n": len(records),
        "net_accuracy_gain": _random_summary(
            [row["net_accuracy_gain"] for row in records]
        ),
        "correction": _random_summary([row["correction"] for row in records]),
        "correct_preservation": _random_summary(
            [row["correct_preservation"] for row in records]
        ),
    }


def parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def run(args) -> None:
    install_qwen2_routing_patch()
    calibration = [
        row
        for row in load_jsonl(args.calibration_pool)
        if protocol_split(row["semantic_id"], stable_split(row["semantic_id"]))
        == "calibration"
    ]
    calibration = _limited_calibration(calibration, args.smoke_per_group)
    confirmation = _limited_confirmation(
        load_jsonl(args.confirmation_rows), args.smoke_per_group
    )
    k6_rows = [row for row in confirmation if int(row["k"]) == 6]
    k0_rows = [row for row in confirmation if int(row["k"]) == 0]
    if not calibration or not k6_rows or not k0_rows:
        raise ValueError("calibration and both confirmation cells must be nonempty")

    heads = load_heads(args.head_summary)
    model, tokenizer, device = load_model(args.model, args.dtype)
    n_heads = int(model.config.num_attention_heads)

    calibration_audit = audit_routes(calibration, tokenizer)
    confirmation_audit = audit_routes(confirmation, tokenizer)
    if calibration_audit["mismatches"] or confirmation_audit["mismatches"]:
        raise RuntimeError("automatic structured-route audit failed")

    calibration_positions = [
        tuple(int(value) for value in row.get("stale_value_spans", []))
        for row in calibration
    ]
    calibration_base, _ = evaluate_routing(
        model,
        tokenizer,
        device,
        calibration,
        [],
        0.0,
        calibration_positions,
        args.batch_size,
    )
    calibration, calibration_base, calibration_mismatches = _stable_calibration(
        calibration, calibration_base
    )
    calibration_positions = [
        tuple(int(value) for value in row.get("stale_value_spans", []))
        for row in calibration
    ]
    counts = Counter(row["label"] for row in calibration)
    if min(counts["correct_current"], counts["within_stale"]) < 2:
        raise RuntimeError(f"calibration lost a class: {counts}")

    curve = _curve(
        model,
        tokenizer,
        device,
        calibration,
        calibration_base,
        heads,
        calibration_positions,
        args.betas,
        args.batch_size,
    )
    try:
        operating = choose_beta(curve, args.minimum_preservation)
        operating_status = "eligible"
    except ValueError:
        operating = {
            "beta": 0.0,
            "correction": 0.0,
            "correct_preservation": 1.0,
            "baseline_accuracy": sum(
                row["label"] == "correct_current" for row in calibration_base
            )
            / len(calibration_base),
            "arm_accuracy": sum(
                row["label"] == "correct_current" for row in calibration_base
            )
            / len(calibration_base),
            "net_accuracy_gain": 0.0,
        }
        operating_status = "no_eligible_beta_identity_fallback"
    beta = float(operating["beta"])

    oracle_positions = [
        tuple(int(value) for value in row.get("stale_value_spans", [])) for row in k6_rows
    ]
    automatic_positions = [
        tuple(route["stale_positions"])
        for route in confirmation_audit["routes"]
        if route["stale_positions"]
    ]
    if len(automatic_positions) != len(k6_rows):
        raise RuntimeError("automatic confirmation routes are not aligned with k=6 rows")

    baseline, baseline_logits = evaluate_routing(
        model,
        tokenizer,
        device,
        k6_rows,
        [],
        0.0,
        oracle_positions,
        args.batch_size,
        return_logits=True,
    )
    identity, identity_logits = evaluate_routing(
        model,
        tokenizer,
        device,
        k6_rows,
        heads,
        0.0,
        oracle_positions,
        args.batch_size,
        return_logits=True,
    )
    identity_max = float(np.max(np.abs(identity_logits - baseline_logits)))
    identity_changes = sum(
        before["pred_token_id"] != after["pred_token_id"]
        for before, after in zip(baseline, identity)
    )
    if identity_max != 0.0 or identity_changes:
        raise RuntimeError("beta=0 identity gate failed")

    oracle, _ = evaluate_routing(
        model,
        tokenizer,
        device,
        k6_rows,
        heads,
        beta,
        oracle_positions,
        args.batch_size,
    )
    automatic, _ = evaluate_routing(
        model,
        tokenizer,
        device,
        k6_rows,
        heads,
        beta,
        automatic_positions,
        args.batch_size,
    )
    opposite, _ = evaluate_routing(
        model,
        tokenizer,
        device,
        k6_rows,
        heads,
        -beta,
        oracle_positions,
        args.batch_size,
    )
    oracle_metric = _metrics_with_ci(
        k6_rows, baseline, oracle, args.n_boot, args.seed + 1000
    )
    automatic_metric = _metrics_with_ci(
        k6_rows, baseline, automatic, args.n_boot, args.seed + 2000
    )
    opposite_metric = _metrics_with_ci(
        k6_rows, baseline, opposite, args.n_boot, args.seed + 3000
    )
    oracle_per_template = _per_template(k6_rows, baseline, oracle)
    automatic_per_template = _per_template(k6_rows, baseline, automatic)

    random_heads = make_layer_matched_random_sets(
        heads,
        n_heads=n_heads,
        n_random=args.n_random_heads,
        seed=args.seed + 4000,
    )
    random_head_records = _control_metrics(
        model,
        tokenizer,
        device,
        k6_rows,
        baseline,
        random_heads,
        beta,
        args.batch_size,
        positions=oracle_positions,
    )
    random_head_summary = _summarize_controls(random_head_records)

    random_positions = make_random_position_sets(
        k6_rows,
        tokenizer,
        n_random=args.n_random_positions,
        seed=args.seed + 5000,
    )
    random_position_records = _control_metrics(
        model,
        tokenizer,
        device,
        k6_rows,
        baseline,
        random_positions,
        beta,
        args.batch_size,
        heads=heads,
    )
    random_position_summary = _summarize_controls(random_position_records)

    k0_positions = [tuple() for _ in k0_rows]
    k0_base, k0_base_logits = evaluate_routing(
        model,
        tokenizer,
        device,
        k0_rows,
        [],
        0.0,
        k0_positions,
        args.batch_size,
        return_logits=True,
    )
    k0_target, k0_target_logits = evaluate_routing(
        model,
        tokenizer,
        device,
        k0_rows,
        heads,
        beta,
        k0_positions,
        args.batch_size,
        return_logits=True,
    )
    k0_metric = _metrics_with_ci(
        k0_rows, k0_base, k0_target, args.n_boot, args.seed + 6000
    )
    k0_max = float(np.max(np.abs(k0_target_logits - k0_base_logits)))
    k0_changes = sum(
        before["pred_token_id"] != after["pred_token_id"]
        for before, after in zip(k0_base, k0_target)
    )

    oracle_gate = {
        "eligible_calibration_beta": operating_status == "eligible",
        "identity_exact_zero": identity_max == 0.0 and identity_changes == 0,
        "net_gain_ci_above_zero": oracle_metric["paired_net_gain"]["ci"][0] > 0.0,
        "net_gain_exceeds_random_head95": (
            oracle_metric["net_accuracy_gain"]
            > random_head_summary["net_accuracy_gain"]["interval95"][1]
        ),
        "net_gain_exceeds_random_position95": (
            oracle_metric["net_accuracy_gain"]
            > random_position_summary["net_accuracy_gain"]["interval95"][1]
        ),
        "correct_preservation": oracle_metric["correct_preservation"] >= 0.95,
        "multi_template_support": sum(
            metric["net_accuracy_gain"] > 0.0
            for metric in oracle_per_template.values()
        )
        >= 2,
        "k0_exact_noop": k0_max == 0.0 and k0_changes == 0,
        "beats_opposite": (
            oracle_metric["net_accuracy_gain"] > opposite_metric["net_accuracy_gain"]
        ),
    }
    oracle_gate["all_pass"] = all(oracle_gate.values())
    retained_fraction = (
        automatic_metric["net_accuracy_gain"] / oracle_metric["net_accuracy_gain"]
        if oracle_metric["net_accuracy_gain"] > 0.0
        else 0.0
    )
    automatic_gate = {
        "route_audit_exact": not calibration_audit["mismatches"]
        and not confirmation_audit["mismatches"],
        "net_gain_ci_above_zero": (
            automatic_metric["paired_net_gain"]["ci"][0] > 0.0
        ),
        "net_gain_exceeds_random_head95": (
            automatic_metric["net_accuracy_gain"]
            > random_head_summary["net_accuracy_gain"]["interval95"][1]
        ),
        "net_gain_exceeds_random_position95": (
            automatic_metric["net_accuracy_gain"]
            > random_position_summary["net_accuracy_gain"]["interval95"][1]
        ),
        "correct_preservation": automatic_metric["correct_preservation"] >= 0.95,
        "multi_template_support": sum(
            metric["net_accuracy_gain"] > 0.0
            for metric in automatic_per_template.values()
        )
        >= 2,
        "retains_half_oracle_gain": retained_fraction >= 0.5,
        "k0_exact_noop": k0_max == 0.0 and k0_changes == 0,
    }
    automatic_gate["all_pass"] = all(automatic_gate.values())
    if automatic_gate["all_pass"] and oracle_gate["all_pass"]:
        verdict = "limited_deployable_structured_log_correction_supported"
    elif oracle_gate["all_pass"]:
        verdict = "oracle_routing_operator_only"
    else:
        verdict = "stale_key_suppression_gate_failed"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "baseline_k6_rows.jsonl", baseline)
    dump_jsonl(out_dir / "oracle_k6_rows.jsonl", oracle)
    dump_jsonl(out_dir / "automatic_k6_rows.jsonl", automatic)
    dump_jsonl(out_dir / "opposite_k6_rows.jsonl", opposite)
    dump_jsonl(out_dir / "baseline_k0_rows.jsonl", k0_base)
    dump_jsonl(out_dir / "automatic_k0_rows.jsonl", k0_target)
    dump_jsonl(out_dir / "random_head_controls.jsonl", random_head_records)
    dump_jsonl(out_dir / "random_position_controls.jsonl", random_position_records)

    def audit_summary(audit):
        return {key: value for key, value in audit.items() if key != "routes"}

    summary = {
        "stage": "R9-update-aware-stale-key-suppression",
        "model": args.model,
        "dtype": args.dtype,
        "frozen_heads": [{"layer": layer, "head": head} for layer, head in heads],
        "runtime_inputs": {
            "oracle": "stored stale spans; mechanism upper bound only",
            "automatic": "raw prompt plus tokenizer offsets only",
            "answer_value_or_token_injected": False,
        },
        "calibration": {
            "requested": len(calibration) + len(calibration_mismatches),
            "retained": len(calibration),
            "mismatches": calibration_mismatches,
            "counts": dict(counts),
            "curve": curve,
            "operating_point": operating,
            "operating_status": operating_status,
        },
        "route_audit": {
            "calibration": audit_summary(calibration_audit),
            "confirmation": audit_summary(confirmation_audit),
        },
        "confirmation": {
            "k6_n": len(k6_rows),
            "k0_n": len(k0_rows),
            "baseline_k6_counts": dict(Counter(row["label"] for row in baseline)),
            "oracle": oracle_metric,
            "automatic": automatic_metric,
            "automatic_retained_oracle_gain": retained_fraction,
            "opposite": opposite_metric,
            "oracle_per_template": oracle_per_template,
            "automatic_per_template": automatic_per_template,
            "random_heads": random_head_summary,
            "random_positions": random_position_summary,
            "k0": k0_metric,
            "identity_max_abs_logit_change": identity_max,
            "identity_prediction_changes": identity_changes,
            "k0_max_abs_logit_change": k0_max,
            "k0_prediction_changes": k0_changes,
        },
        "oracle_gate": oracle_gate,
        "automatic_gate": automatic_gate,
        "verdict": verdict,
        "scope": (
            "open-weight Qwen2.5-1.5B with explicitly structured update logs; "
            "not a natural-dialogue, cross-model, or closed-API repair"
        ),
    }
    dump_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"verdict": verdict, "beta": beta, "oracle_gate": oracle_gate, "automatic_gate": automatic_gate}, indent=2), flush=True)


def write_report(path: str | Path, summary: dict) -> None:
    confirmation = summary["confirmation"]
    lines = [
        "# Stage R9: Update-aware stale-key suppression",
        "",
        f"**Verdict:** `{summary['verdict']}`.",
        "",
        "| Route | n | beta | correction | preservation | net gain | paired 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("oracle", "automatic"):
        metric = confirmation[label]
        lines.append(
            f"| {label} | {confirmation['k6_n']} | "
            f"{summary['calibration']['operating_point']['beta']:.2f} | "
            f"{metric['correction']:.3f} | {metric['correct_preservation']:.3f} | "
            f"{metric['net_accuracy_gain']:.3f} | {metric['paired_net_gain']['ci']} |"
        )
    lines.extend(
        [
            "",
            f"Random-head net-gain 95%: {confirmation['random_heads']['net_accuracy_gain']['interval95']}.",
            f"Random-position net-gain 95%: {confirmation['random_positions']['net_accuracy_gain']['interval95']}.",
            f"Opposite-sign net gain: {confirmation['opposite']['net_accuracy_gain']:.3f}.",
            f"k=0 maximum logit change: {confirmation['k0_max_abs_logit_change']:.1f}.",
            "",
            "## Oracle gates",
            "",
        ]
    )
    lines.extend(
        f"- {key}: **{'PASS' if value else 'FAIL'}**"
        for key, value in summary["oracle_gate"].items()
    )
    lines.extend(["", "## Automatic structured-log gates", ""])
    lines.extend(
        f"- {key}: **{'PASS' if value else 'FAIL'}**"
        for key, value in summary["automatic_gate"].items()
    )
    lines.extend(["", "## Scope", "", summary["scope"]])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--calibration-pool", default="results/stage_o/partC_qwen15b/matched_pool.jsonl")
    parser.add_argument("--head-summary", default="results/stage_o/partC_qwen15b/ablation/ablation_heldout.summary.json")
    parser.add_argument("--confirmation-rows", default="results/stage_r/attenuation/confirmation_tasks.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--betas", type=parse_floats, default=list(DEFAULT_BETAS))
    parser.add_argument("--minimum-preservation", type=float, default=0.95)
    parser.add_argument("--n-random-heads", type=int, default=64)
    parser.add_argument("--n-random-positions", type=int, default=16)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--smoke-per-group", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
