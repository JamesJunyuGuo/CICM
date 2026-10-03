"""Stage R11: model-specific balanced attention routing on Llama and Qwen.

The script discovers eight query heads on the frozen calibration split, then
delegates the unchanged R10 intervention and confirmation protocol to the
natural-dialogue runner. Frozen R10 files are read-only inputs, never outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import gc
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import stale_binding_natural_routing as natural
from pythia_eval import dump_json
from pythia_gen import dump_jsonl, load_jsonl
from cicm_eval import load_model_and_tokenizer, render_prompt
from stale_binding_routing import (
    _ROUTING_CONTEXT,
    apply_stale_key_bias,
    install_qwen2_routing_patch,
)


_CAPTURE_CONTEXT = contextvars.ContextVar("stage_r11_capture_context", default=None)


@dataclass
class AttentionCaptureContext:
    stale_positions: tuple[tuple[int, ...], ...]
    current_positions: tuple[tuple[int, ...], ...]
    query_positions: tuple[int, ...]
    by_layer: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@contextlib.contextmanager
def attention_capture(ctx: AttentionCaptureContext):
    token = _CAPTURE_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _CAPTURE_CONTEXT.reset(token)


def _repeat_kv(hidden_states, repetitions: int):
    if repetitions == 1:
        return hidden_states
    batch, key_heads, length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, key_heads, repetitions, length, head_dim
    )
    return expanded.reshape(batch, key_heads * repetitions, length, head_dim)


def _record_attention(attn_weights, layer_idx: int) -> None:
    import torch

    ctx = _CAPTURE_CONTEXT.get()
    if ctx is None:
        return
    if len(ctx.query_positions) != attn_weights.shape[0]:
        raise ValueError("capture query positions do not match attention batch")
    stale_rows = []
    current_rows = []
    for batch_index, (query, stale, current) in enumerate(
        zip(ctx.query_positions, ctx.stale_positions, ctx.current_positions)
    ):
        query = int(query)
        if not 0 <= query < attn_weights.shape[-2]:
            raise ValueError("capture query position outside attention tensor")
        if not stale or not current:
            raise ValueError("capture requires stale and current key positions")
        if min(stale + current) < 0 or max(stale + current) >= attn_weights.shape[-1]:
            raise ValueError("capture key position outside attention tensor")
        query_weights = attn_weights[batch_index, :, query, :]
        stale_index = torch.as_tensor(stale, dtype=torch.long, device=query_weights.device)
        current_index = torch.as_tensor(current, dtype=torch.long, device=query_weights.device)
        stale_rows.append(query_weights.index_select(-1, stale_index).sum(dim=-1))
        current_rows.append(query_weights.index_select(-1, current_index).sum(dim=-1))
    ctx.by_layer[int(layer_idx)] = (
        np.stack([row.detach().float().cpu().numpy() for row in stale_rows]),
        np.stack([row.detach().float().cpu().numpy() for row in current_rows]),
    )


def _generic_routing_eager_attention(
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
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = apply_stale_key_bias(
        attn_weights, module.layer_idx, _ROUTING_CONTEXT.get()
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


def install_model_routing_patches() -> None:
    """Install one tested eager-attention implementation for Qwen2 and Llama."""
    install_qwen2_routing_patch()
    import transformers.models.llama.modeling_llama as llama
    import transformers.models.qwen2.modeling_qwen2 as qwen2

    llama.eager_attention_forward = _generic_routing_eager_attention
    qwen2.eager_attention_forward = _generic_routing_eager_attention


def capture_head_attention(
    model,
    tokenizer,
    rows,
    routes,
    batch_size: int,
    prompt_renderer=render_prompt,
):
    import torch

    stale_batches = []
    current_batches = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    expected_layers = int(model.config.num_hidden_layers)
    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_routes = routes[start : start + batch_size]
            prompts = [
                prompt_renderer(tokenizer, row["messages"])
                for row in batch_rows
            ]
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            width = int(enc["input_ids"].shape[1])
            lengths = enc["attention_mask"].sum(dim=1).detach().cpu().tolist()
            stale_positions = []
            current_positions = []
            for length, route in zip(lengths, batch_routes):
                pad = width - int(length)
                stale_positions.append(
                    tuple(pad + int(value) for value in route["stale_positions"])
                )
                current_positions.append(
                    tuple(pad + int(value) for value in route["current_positions"])
                )
            ctx = AttentionCaptureContext(
                stale_positions=tuple(stale_positions),
                current_positions=tuple(current_positions),
                query_positions=tuple(width - 1 for _ in batch_rows),
            )
            with attention_capture(ctx), torch.inference_mode():
                model.model(**enc, use_cache=False)
            if sorted(ctx.by_layer) != list(range(expected_layers)):
                raise RuntimeError(
                    f"captured layers {sorted(ctx.by_layer)}; expected 0..{expected_layers - 1}"
                )
            stale_batches.append(
                np.stack([ctx.by_layer[layer][0] for layer in range(expected_layers)], axis=1)
            )
            current_batches.append(
                np.stack([ctx.by_layer[layer][1] for layer in range(expected_layers)], axis=1)
            )
            print(
                f"captured attention {min(start + batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )
    finally:
        tokenizer.padding_side = old_padding_side
    return np.concatenate(stale_batches, axis=0), np.concatenate(current_batches, axis=0)


def select_heads(
    stale_attention: np.ndarray,
    current_attention: np.ndarray,
    labels: list[str],
    *,
    count: int = 8,
    minimum_pool: int = 5,
) -> dict:
    if stale_attention.shape != current_attention.shape or stale_attention.ndim != 3:
        raise ValueError("attention arrays must have matching [row, layer, head] shapes")
    if stale_attention.shape[0] != len(labels):
        raise ValueError("labels do not align with attention rows")
    correct = np.asarray([label == "correct_current" for label in labels])
    failures = np.asarray([label == "within_stale" for label in labels])
    if int(correct.sum()) < minimum_pool or int(failures.sum()) < minimum_pool:
        raise RuntimeError(
            f"head discovery needs {minimum_pool} correct and stale rows; "
            f"found {int(correct.sum())} and {int(failures.sum())}"
        )
    eps = 1e-8
    log_ratio = np.log((stale_attention + eps) / (current_attention + eps))
    correct_mean = log_ratio[correct].mean(axis=0)
    failure_mean = log_ratio[failures].mean(axis=0)
    score = failure_mean - correct_mean
    records = []
    for layer in range(score.shape[0]):
        for head in range(score.shape[1]):
            records.append(
                {
                    "layer": layer,
                    "head": head,
                    "score": float(score[layer, head]),
                    "failure_mean_log_stale_current": float(failure_mean[layer, head]),
                    "correct_mean_log_stale_current": float(correct_mean[layer, head]),
                }
            )
    ranked = sorted(records, key=lambda row: (-row["score"], row["layer"], row["head"]))
    positive = [row for row in ranked if row["score"] > 0.0]
    if len(positive) < count:
        raise RuntimeError(f"only {len(positive)} heads have a positive discovery score")
    return {
        "rule": "failure-minus-correct mean log stale/current attention ratio",
        "n_correct": int(correct.sum()),
        "n_within_stale": int(failures.sum()),
        "top_heads": positive[:count],
        "all_heads": ranked,
    }


def write_crossmodel_report(path: Path, summary: dict) -> None:
    lines = [
        f"# Stage R11: {summary['model_label']} balanced attention routing",
        "",
        f"**Verdict:** `{summary['verdict']}`.",
        "",
        f"Calibration selected beta={summary['beta']} and eight model-specific heads. ",
        "Head discovery and beta selection used calibration only.",
        "",
        "| Task | baseline | routed | gain | 95% CI | preservation | stale correction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, task in summary["tasks"].items():
        metric = task["routed"]
        lines.append(
            f"| {name} | {metric['baseline_accuracy']:.3f} | {metric['arm_accuracy']:.3f} | "
            f"{metric['net_accuracy_gain']:.3f} | {metric['paired_net_gain']['ci']} | "
            f"{metric['correct_preservation']:.3f} | {metric['within_stale_correction']:.3f} |"
        )
    lines.extend(["", "## Selected heads", ""])
    lines.extend(
        f"- L{row['layer']}H{row['head']}: score={row['score']:.4f}"
        for row in summary["head_discovery"]["top_heads"]
    )
    lines.extend(["", "## Gates", ""])
    for name, task in summary["tasks"].items():
        lines.append(f"### {name}")
        lines.extend(
            f"- {key}: **{'PASS' if value else 'FAIL'}**"
            for key, value in task["gate"].items()
        )
        lines.append("")
    lines.extend(["## Scope", "", summary["scope"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    install_model_routing_patches()
    base_rows = natural._limit_cells(load_jsonl(args.data), args.limit_per_cell)
    calibration_base, _ = natural.split_factorial_rows(
        base_rows, args.calibration_fraction
    )
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    if model.config.model_type != args.model_family:
        raise ValueError(
            f"loaded model_type={model.config.model_type}, expected {args.model_family}"
        )
    audit, _ = natural.audit_natural_routes(base_rows, tokenizer)
    if audit["mismatches"]:
        raise RuntimeError("natural route audit failed before head discovery")
    calibration_rows = natural.prepare_task_rows(calibration_base)["retrieval"]
    calibration_routes = [
        natural.infer_natural_route(row["messages"], tokenizer)
        for row in calibration_rows
    ]
    device = next(model.parameters()).device
    baseline, _ = natural._generate_arm(
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
    stale_attention, current_attention = capture_head_attention(
        model,
        tokenizer,
        calibration_rows,
        calibration_routes,
        args.capture_batch_size,
    )
    discovery = select_heads(
        stale_attention,
        current_attention,
        [row["label"] for row in baseline],
        count=8,
        minimum_pool=args.minimum_discovery_pool,
    )
    out_dir = Path(args.out_dir)
    discovery_dir = out_dir / "head_discovery"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(discovery_dir / "baseline_rows.jsonl", baseline)
    dump_json(
        discovery_dir / "head_summary.json",
        {
            "model": args.model_label,
            "calibration_n": len(calibration_rows),
            **discovery,
        },
    )
    np.savez_compressed(
        discovery_dir / "attention_mass.npz",
        stale=stale_attention.astype(np.float16),
        current=current_attention.astype(np.float16),
    )

    del model
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args.head_summary = str(discovery_dir / "head_summary.json")
    args.routing_mode = "balanced"
    natural.run(args)

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    retrieval_gate = summary["tasks"]["retrieval"]["gate"]
    if not retrieval_gate["adequate_correct_pool"] or not retrieval_gate["adequate_within_stale_pool"]:
        verdict = "substrate_infeasible"
    elif retrieval_gate["all_pass"]:
        verdict = "model_retrieval_correction_supported"
    else:
        verdict = "model_correction_gate_failed"
    summary.update(
        {
            "stage": "R11-crossmodel-balanced-routing",
            "verdict": verdict,
            "model": args.model_label,
            "model_label": args.model_label,
            "model_path": args.model,
            "model_family": args.model_family,
            "head_discovery": discovery,
            "scope": (
                f"Model-specific balanced attention routing on {args.model_label} over the "
                "controlled natural CICM grammar; not unrestricted-dialogue, closed-API, "
                "or unique-circuit evidence."
            ),
        }
    )
    dump_json(summary_path, summary)
    write_crossmodel_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"verdict": verdict, "model": args.model_label}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/cicm/cicm_natural_factorial_otherdist_l0.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-family", choices=("llama", "qwen2"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--betas", type=lambda value: [float(item) for item in value.split(",")], default=list(natural.DEFAULT_BETAS))
    parser.add_argument("--minimum-preservation", type=float, default=0.95)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-pool", type=int, default=100)
    parser.add_argument("--minimum-discovery-pool", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--capture-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--n-random-positions", type=int, default=16)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--limit-per-cell", type=int)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
