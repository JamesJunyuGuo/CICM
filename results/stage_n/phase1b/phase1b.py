"""Stage N Phase 1b: held-out ablation and upstream QK attribution."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import pythia_circuit as core  # noqa: E402
from pythia_eval import classify_token, dump_json, load_model  # noqa: E402
from pythia_gen import dump_jsonl, load_jsonl  # noqa: E402


TARGET_HEADS = [(8, 10), (8, 2)]
UPSTREAM_LAYERS = tuple(range(8))
MAX_COMPACT_COMPONENTS = 12


def component_name(component: tuple[str, int, int | None]) -> str:
    kind, layer, index = component
    return f"L{layer}H{index}" if kind == "head" else f"L{layer}MLP"


def component_dict(component: tuple[str, int, int | None]) -> dict:
    kind, layer, index = component
    return {"kind": kind, "layer": layer, "head": index, "name": component_name(component)}


def upstream_components() -> list[tuple[str, int, int | None]]:
    return [
        *(('head', layer, head) for layer in UPSTREAM_LAYERS for head in range(12)),
        *(('mlp', layer, None) for layer in UPSTREAM_LAYERS),
    ]


def quantile_summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "quantile95": np.quantile(values, [0.025, 0.975]).tolist(),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def slice_rows(rows, offset: int = 0, limit: int | None = None):
    stop = None if limit is None else offset + limit
    return rows[offset:stop]


def merge_identity_gates(gates: list[dict]) -> dict:
    keys = gates[0].keys()
    return {
        key: (
            sum(int(gate[key]) for gate in gates)
            if key == "prediction_changes"
            else max(float(gate[key]) for gate in gates)
        )
        for key in keys
    }


def split_name(row: dict) -> str:
    return core.stable_split(row["semantic_id"])


def discovery_stale_ranking(harvest_dir: Path) -> tuple[list[tuple[int, int]], list[dict]]:
    rows = load_jsonl(harvest_dir / "harvest_index.jsonl")
    data = np.load(harvest_dir / "harvest.npz")
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    discovery = np.asarray([split_name(row) == "discovery" for row in rows])
    correct = discovery & (labels == "correct_current")
    failure = discovery & (labels == "within_stale")
    denominator = data["attn_stale"] + data["attn_current"]
    # Match the frozen Phase-1 ratio definition exactly for near-zero-attention heads.
    stale_ratio = data["attn_stale"] / np.maximum(denominator, 1e-12)
    table = []
    for layer in range(stale_ratio.shape[1]):
        for head in range(stale_ratio.shape[2]):
            attention_shift = float(
                np.nanmean(stale_ratio[failure, layer, head])
                - np.nanmean(stale_ratio[correct, layer, head])
            )
            failure_dla = float(np.mean(data["head_dla"][failure, layer, head]))
            score = attention_shift * max(0.0, -failure_dla)
            table.append(
                {
                    "layer": layer,
                    "head": head,
                    "discovery_attention_shift": attention_shift,
                    "discovery_failure_dla": failure_dla,
                    "discovery_score": score,
                }
            )
    table.sort(key=lambda row: row["discovery_score"], reverse=True)
    for rank, row in enumerate(table, 1):
        row["rank"] = rank
    return [(row["layer"], row["head"]) for row in table], table


def evaluate_random_ablation(
    model,
    tokenizer,
    device,
    rows: list[dict],
    baseline_labels: np.ndarray,
    random_sets: list[list[tuple[int, int]]],
    target_label: str,
    batch_size: int,
) -> tuple[list[float], list[dict]]:
    rates = []
    records = []
    baseline = (baseline_labels == target_label).astype(float)
    for index, heads in enumerate(random_sets):
        result = core.evaluate_ablation_arm(model, tokenizer, device, rows, heads, batch_size)
        labels = np.asarray(result["labels"])
        effect = (labels == target_label).astype(float) - baseline
        rates.append(float(np.mean(effect)))
        records.append(
            {
                "random_index": index,
                "heads": [{"layer": layer, "head": head} for layer, head in heads],
                "paired_effect": float(np.mean(effect)),
            }
        )
        if (index + 1) % 8 == 0 or index + 1 == len(random_sets):
            print(f"random ablation {target_label} {index + 1}/{len(random_sets)}", flush=True)
    return rates, records


def ablate_heldout(args) -> None:
    rows = load_jsonl(args.pool)
    split_counts = Counter((row["label"], split_name(row)) for row in rows)
    correct_rows = [
        row for row in rows if row["label"] == "correct_current" and split_name(row) == "evaluation"
    ]
    failure_rows = [
        row for row in rows if row["label"] == "within_stale" and split_name(row) == "evaluation"
    ]
    cumulative = json.loads(Path(args.cumulative_summary).read_text(encoding="utf-8"))
    current_heads = [
        (int(row["layer"]), int(row["head"]))
        for row in cumulative["selected_heads"][: args.max_heads]
    ]
    stale_ranking, stale_table = discovery_stale_ranking(Path(args.harvest_dir))
    stale_heads = stale_ranking[: len(current_heads)]

    model, tokenizer, device = load_model(args.model, args.dtype)
    baseline_correct = core.evaluate_ablation_arm(
        model, tokenizer, device, correct_rows, [], args.batch_size
    )
    targeted_current = core.evaluate_ablation_arm(
        model, tokenizer, device, correct_rows, current_heads, args.batch_size
    )
    baseline_failure = core.evaluate_ablation_arm(
        model, tokenizer, device, failure_rows, [], args.batch_size
    )
    targeted_stale = core.evaluate_ablation_arm(
        model, tokenizer, device, failure_rows, stale_heads, args.batch_size
    )

    baseline_correct_labels = np.asarray(baseline_correct["labels"])
    targeted_current_labels = np.asarray(targeted_current["labels"])
    baseline_failure_labels = np.asarray(baseline_failure["labels"])
    targeted_stale_labels = np.asarray(targeted_stale["labels"])
    current_effect = (targeted_current_labels == "within_stale").astype(float) - (
        baseline_correct_labels == "within_stale"
    ).astype(float)
    stale_effect = (targeted_stale_labels == "correct_current").astype(float) - (
        baseline_failure_labels == "correct_current"
    ).astype(float)

    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    current_random_sets = core.layer_matched_random_sets(
        current_heads, n_layers, n_heads, args.n_random, args.seed
    )
    stale_random_sets = core.layer_matched_random_sets(
        stale_heads, n_layers, n_heads, args.n_random, args.seed + 1
    )
    current_random_rates, current_random_records = evaluate_random_ablation(
        model,
        tokenizer,
        device,
        correct_rows,
        baseline_correct_labels,
        current_random_sets,
        "within_stale",
        args.batch_size,
    )
    stale_random_rates, stale_random_records = evaluate_random_ablation(
        model,
        tokenizer,
        device,
        failure_rows,
        baseline_failure_labels,
        stale_random_sets,
        "correct_current",
        args.batch_size,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "ablation_random_sets.jsonl", [
        *({"arm": "current", **row} for row in current_random_records),
        *({"arm": "stale", **row} for row in stale_random_records),
    ])
    phase1 = json.loads(Path(args.phase1_ablation).read_text(encoding="utf-8"))
    stale_random_q = np.quantile(stale_random_rates, [0.025, 0.975])
    current_random_q = np.quantile(current_random_rates, [0.025, 0.975])
    summary = {
        "model": args.model,
        "protocol": {
            "split": "stable_split(semantic_id), rank on discovery and evaluate on evaluation only",
            "split_counts": {
                label: {
                    split: split_counts[(label, split)]
                    for split in ("discovery", "evaluation")
                }
                for label in ("correct_current", "within_stale")
            },
            "n_random": args.n_random,
            "random_control": "same evaluation rows; layer-matched same-size head sets",
        },
        "current_binding": {
            "selection": "Phase-1 path-patch prefix ranked on discovery only",
            "heads": [{"layer": layer, "head": head} for layer, head in current_heads],
            "evaluation_n": len(correct_rows),
            "baseline_stale_rate": float(np.mean(baseline_correct_labels == "within_stale")),
            "targeted_stale_rate": float(np.mean(targeted_current_labels == "within_stale")),
            "paired_effect": core.paired_bootstrap(current_effect, args.n_boot, args.seed + 10),
            "random_effect": quantile_summary(current_random_rates),
            "heldout_exceeds_random95": bool(float(np.mean(current_effect)) > current_random_q[1]),
            "phase1_same_pool_effect": phase1["current_binding_ablation"][
                "paired_stale_rate_increase"
            ]["mean"],
        },
        "stale_promoting": {
            "selection": "discovery stale-ratio shift times max(0, negative discovery failure DLA)",
            "heads": [{"layer": layer, "head": head} for layer, head in stale_heads],
            "ranking": stale_table,
            "evaluation_n": len(failure_rows),
            "baseline_correct_rate": float(np.mean(baseline_failure_labels == "correct_current")),
            "targeted_correct_rate": float(np.mean(targeted_stale_labels == "correct_current")),
            "paired_effect": core.paired_bootstrap(stale_effect, args.n_boot, args.seed + 20),
            "random_effect": quantile_summary(stale_random_rates),
            "heldout_exceeds_random95": bool(float(np.mean(stale_effect)) > stale_random_q[1]),
            "phase1_same_pool_effect": phase1["stale_component_ablation"][
                "paired_correct_rate_increase"
            ]["mean"],
            "same_pool_minus_heldout": float(
                phase1["stale_component_ablation"]["paired_correct_rate_increase"]["mean"]
                - np.mean(stale_effect)
            ),
        },
        "identity_reproduction": {
            "correct_baseline_label_mismatches": int(
                np.sum(baseline_correct_labels != "correct_current")
            ),
            "failure_baseline_label_mismatches": int(
                np.sum(baseline_failure_labels != "within_stale")
            ),
        },
    }
    dump_json(out_dir / "ablation_heldout.summary.json", summary)
    print(json.dumps({
        "current_heldout": summary["current_binding"]["paired_effect"],
        "stale_heldout": summary["stale_promoting"]["paired_effect"],
        "stale_exceeds_random": summary["stale_promoting"]["heldout_exceeds_random95"],
    }, indent=2))


class ForwardCache:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.head_inputs: dict[int, object] = {}
        self.mlp_outputs: dict[int, object] = {}
        self.qkv_outputs: dict[int, object] = {}

    def __enter__(self):
        for layer in UPSTREAM_LAYERS:
            block = self.model.gpt_neox.layers[layer]
            self.handles.append(
                block.attention.dense.register_forward_pre_hook(
                    lambda _module, inputs, layer=layer: self.head_inputs.__setitem__(
                        layer, inputs[0].detach().clone()
                    )
                )
            )
            self.handles.append(
                block.mlp.register_forward_hook(
                    lambda _module, _inputs, output, layer=layer: self.mlp_outputs.__setitem__(
                        layer, output.detach().clone()
                    )
                )
            )
        for layer in sorted({layer for layer, _head in TARGET_HEADS}):
            module = self.model.gpt_neox.layers[layer].attention.query_key_value
            self.handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, layer=layer: self.qkv_outputs.__setitem__(
                        layer, output.detach().clone()
                    )
                )
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()


def run_capture(model, enc) -> tuple[object, ForwardCache]:
    import torch

    cache = ForwardCache(model)
    with cache, torch.no_grad():
        output = model(
            **enc,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    return output, cache


def row_positions(rows: list[dict], side: str, query) -> list[list[int]]:
    if side == "query":
        return [[int(position)] for position in query.detach().cpu().tolist()]
    if side == "key":
        return [
            [int(write["value_token"]) for write in row["writes"] if write["var"] == row["target_var"]]
            for row in rows
        ]
    raise ValueError(side)


def flatten_patch_positions(positions: list[list[int]]) -> tuple[list[int], list[int]]:
    batch_indices = []
    token_indices = []
    for batch_index, row_positions_ in enumerate(positions):
        batch_indices.extend([batch_index] * len(row_positions_))
        token_indices.extend(int(position) for position in row_positions_)
    return batch_indices, token_indices


def register_component_patch(
    model,
    source: ForwardCache,
    components: list[tuple[str, int, int | None]],
    positions: list[list[int]],
):
    import torch

    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // n_heads)
    by_layer_heads: dict[int, list[int]] = defaultdict(list)
    mlp_layers = set()
    for kind, layer, index in components:
        if kind == "head":
            by_layer_heads[layer].append(int(index))
        else:
            mlp_layers.add(layer)
    flat_batch, flat_tokens = flatten_patch_positions(positions)
    handles = []
    for layer, heads in by_layer_heads.items():
        def head_hook(_module, inputs, layer=layer, heads=tuple(heads)):
            value = inputs[0].clone()
            value_heads = value.view(value.shape[0], value.shape[1], n_heads, head_dim)
            source_heads = source.head_inputs[layer].view(
                value.shape[0], value.shape[1], n_heads, head_dim
            )
            batch_index = torch.as_tensor(flat_batch, device=value.device)[:, None]
            token_index = torch.as_tensor(flat_tokens, device=value.device)[:, None]
            head_index = torch.as_tensor(heads, device=value.device)[None, :]
            value_heads[batch_index, token_index, head_index, :] = source_heads[
                batch_index, token_index, head_index, :
            ]
            return (value_heads.reshape_as(value),)
        handles.append(
            model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(head_hook)
        )
    for layer in mlp_layers:
        def mlp_hook(_module, _inputs, output, layer=layer):
            value = output.clone()
            batch_index = torch.as_tensor(flat_batch, device=value.device)
            token_index = torch.as_tensor(flat_tokens, device=value.device)
            value[batch_index, token_index] = source.mlp_outputs[layer][batch_index, token_index]
            return value
        handles.append(model.gpt_neox.layers[layer].mlp.register_forward_hook(mlp_hook))
    return handles


def register_qkv_patch(
    model,
    source_qkv: dict[int, object],
    target: tuple[int, int],
    side: str,
    query,
    write_positions: list[list[int]],
):
    layer, head = target
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // n_heads)

    def hook(_module, _inputs, output):
        value = output.clone().view(output.shape[0], output.shape[1], n_heads, 3 * head_dim)
        source = source_qkv[layer].view(
            output.shape[0], output.shape[1], n_heads, 3 * head_dim
        )
        if side in {"query", "both"}:
            for batch_index, position in enumerate(query.detach().cpu().tolist()):
                value[batch_index, int(position), head, :head_dim] = source[
                    batch_index, int(position), head, :head_dim
                ]
        if side in {"key", "both"}:
            for batch_index, positions in enumerate(write_positions):
                for position in positions:
                    value[batch_index, position, head, head_dim : 2 * head_dim] = source[
                        batch_index, position, head, head_dim : 2 * head_dim
                    ]
        return value.reshape_as(output)

    return model.gpt_neox.layers[layer].attention.query_key_value.register_forward_hook(hook)


def run_patched(
    model,
    enc,
    *,
    component_source: ForwardCache | None = None,
    components: list[tuple[str, int, int | None]] | None = None,
    component_positions: list[list[int]] | None = None,
    qkv_source: dict[int, object] | None = None,
    qkv_target: tuple[int, int] | None = None,
    qkv_side: str | None = None,
    query=None,
    write_positions: list[list[int]] | None = None,
):
    import torch

    handles = []
    capture = {}
    if components:
        handles.extend(
            register_component_patch(model, component_source, components, component_positions)
        )
    if qkv_target is not None:
        handles.append(
            register_qkv_patch(
                model, qkv_source, qkv_target, qkv_side, query, write_positions
            )
        )
    for layer in sorted({layer for layer, _head in TARGET_HEADS}):
        handles.append(
            model.gpt_neox.layers[layer].attention.query_key_value.register_forward_hook(
                lambda _module, _inputs, output, layer=layer: capture.__setitem__(
                    layer, output.detach().clone()
                )
            )
        )
    try:
        with torch.no_grad():
            output = model(
                **enc,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        return output, capture
    finally:
        for handle in handles:
            handle.remove()


def target_metrics(model, enc, rows: list[dict], output, qkv_outputs) -> dict:
    import torch
    from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

    query = core.query_positions(enc)
    final_gap, predictions = core.gaps_from_logits(output.logits, rows, query)
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // n_heads)
    position_ids = torch.arange(enc["input_ids"].shape[1], device=enc["input_ids"].device)
    position_ids = position_ids.unsqueeze(0).expand(enc["input_ids"].shape[0], -1)
    metrics = {}
    for layer, head in TARGET_HEADS:
        qkv = qkv_outputs[layer].view(
            enc["input_ids"].shape[0], enc["input_ids"].shape[1], n_heads, 3 * head_dim
        ).transpose(1, 2)
        query_states, key_states, _value_states = qkv.chunk(3, dim=-1)
        cos, sin = model.gpt_neox.rotary_emb(output.hidden_states[layer], position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        qk_margin = []
        current_attention = []
        stale_attention = []
        for batch_index, row in enumerate(rows):
            q_position = int(query[batch_index].item())
            q = query_states[batch_index, head, q_position]
            scores = torch.matmul(key_states[batch_index, head], q) / math.sqrt(head_dim)
            current = int(row["current_value_span"][0])
            stale = [int(position) for position in row["stale_value_spans"]]
            qk_margin.append(float(scores[current].item() - scores[stale].max().item()))
            attention = output.attentions[layer][batch_index, head, q_position]
            current_attention.append(float(attention[current].item()))
            stale_attention.append(float(attention[stale].sum().item()))
        metrics[f"L{layer}H{head}"] = {
            "qk_margin": np.asarray(qk_margin, dtype=np.float64),
            "current_attention": np.asarray(current_attention, dtype=np.float64),
            "stale_attention": np.asarray(stale_attention, dtype=np.float64),
            "attention_preference": np.asarray(current_attention, dtype=np.float64)
            - np.asarray(stale_attention, dtype=np.float64),
            "final_gap": np.asarray(final_gap, dtype=np.float64),
            "predictions": predictions,
        }
    return metrics


def baseline_attention_features(output, enc, rows: list[dict]) -> dict:
    import torch

    query = core.query_positions(enc)
    n_layers = len(UPSTREAM_LAYERS)
    n_heads = int(output.attentions[0].shape[1])
    previous = np.zeros((len(rows), n_layers, n_heads), dtype=np.float32)
    latest = np.zeros_like(previous)
    for layer in UPSTREAM_LAYERS:
        attention = output.attentions[layer]
        for batch_index, row in enumerate(rows):
            length = int(query[batch_index].item()) + 1
            destinations = torch.arange(1, length, device=attention.device)
            previous[batch_index, layer] = (
                attention[batch_index, :, destinations, destinations - 1]
                .mean(dim=-1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            q_position = int(query[batch_index].item())
            current = int(row["current_value_span"][0])
            stale = [int(position) for position in row["stale_value_spans"]]
            latest[batch_index, layer] = (
                attention[batch_index, :, q_position, current]
                - attention[batch_index, :, q_position, stale].mean(dim=-1)
            ).detach().float().cpu().numpy()
    return {"previous_token": previous, "latest_write": latest}


def metric_record(metrics: dict, index: int) -> dict:
    return {
        head: {
            key: (
                int(values[index])
                if key == "predictions"
                else float(values[index])
            )
            for key, values in head_metrics.items()
        }
        for head, head_metrics in metrics.items()
    }


def upstream_scan(args) -> None:
    pairs = load_jsonl(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    components = upstream_components()
    model, tokenizer, device = load_model(args.model, args.dtype)
    records = []
    identity = {
        "b1_max_abs_final_gap": 0.0,
        "b1_max_abs_qk_margin": 0.0,
        "b1_max_abs_attention_preference": 0.0,
        "b1_prediction_changes": 0,
        "b2_max_abs_final_gap": 0.0,
        "b2_max_abs_qk_margin": 0.0,
        "b2_max_abs_attention_preference": 0.0,
        "b2_prediction_changes": 0,
        "b2_paths_checked": len(components) * 2,
    }
    for start in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[start : start + args.batch_size]
        clean_rows = [pair["clean"] for pair in batch_pairs]
        corrupt_rows = [pair["corrupted"] for pair in batch_pairs]
        clean_enc = core.batch_encode(tokenizer, device, clean_rows)
        corrupt_enc = core.batch_encode(tokenizer, device, corrupt_rows)
        query = core.query_positions(corrupt_enc)
        if clean_enc["input_ids"].shape != corrupt_enc["input_ids"].shape:
            raise ValueError("clean/corrupt batch shape mismatch")
        write_positions = row_positions(corrupt_rows, "key", query)
        clean_output, clean_cache = run_capture(model, clean_enc)
        corrupt_output, corrupt_cache = run_capture(model, corrupt_enc)
        clean_metrics = target_metrics(
            model, clean_enc, corrupt_rows, clean_output, clean_cache.qkv_outputs
        )
        corrupt_metrics = target_metrics(
            model, corrupt_enc, corrupt_rows, corrupt_output, corrupt_cache.qkv_outputs
        )
        attention_features = baseline_attention_features(corrupt_output, corrupt_enc, corrupt_rows)

        b1 = defaultdict(dict)
        for target in TARGET_HEADS:
            target_name = f"L{target[0]}H{target[1]}"
            for side in ("query", "key", "both"):
                patched_output, patched_qkv = run_patched(
                    model,
                    corrupt_enc,
                    qkv_source=clean_cache.qkv_outputs,
                    qkv_target=target,
                    qkv_side=side,
                    query=query,
                    write_positions=write_positions,
                )
                b1[target_name][side] = target_metrics(
                    model, corrupt_enc, corrupt_rows, patched_output, patched_qkv
                )[target_name]
                identity_output, identity_qkv = run_patched(
                    model,
                    corrupt_enc,
                    qkv_source=corrupt_cache.qkv_outputs,
                    qkv_target=target,
                    qkv_side=side,
                    query=query,
                    write_positions=write_positions,
                )
                identity_metrics = target_metrics(
                    model, corrupt_enc, corrupt_rows, identity_output, identity_qkv
                )[target_name]
                identity["b1_max_abs_final_gap"] = max(
                    identity["b1_max_abs_final_gap"],
                    float(np.max(np.abs(identity_metrics["final_gap"] - corrupt_metrics[target_name]["final_gap"]))),
                )
                identity["b1_max_abs_qk_margin"] = max(
                    identity["b1_max_abs_qk_margin"],
                    float(np.max(np.abs(identity_metrics["qk_margin"] - corrupt_metrics[target_name]["qk_margin"]))),
                )
                identity["b1_max_abs_attention_preference"] = max(
                    identity["b1_max_abs_attention_preference"],
                    float(np.max(np.abs(identity_metrics["attention_preference"] - corrupt_metrics[target_name]["attention_preference"]))),
                )
                identity["b1_prediction_changes"] += int(
                    np.sum(np.asarray(identity_metrics["predictions"]) != np.asarray(corrupt_metrics[target_name]["predictions"]))
                )

        b2 = defaultdict(dict)
        for side in ("query", "key"):
            positions = row_positions(corrupt_rows, side, query)
            for component in components:
                name = component_name(component)
                patched_output, patched_qkv = run_patched(
                    model,
                    corrupt_enc,
                    component_source=clean_cache,
                    components=[component],
                    component_positions=positions,
                )
                b2[side][name] = target_metrics(
                    model, corrupt_enc, corrupt_rows, patched_output, patched_qkv
                )
                identity_output, identity_qkv = run_patched(
                    model,
                    corrupt_enc,
                    component_source=corrupt_cache,
                    components=[component],
                    component_positions=positions,
                )
                identity_metrics = target_metrics(
                    model, corrupt_enc, corrupt_rows, identity_output, identity_qkv
                )
                for target_name in identity_metrics:
                    identity["b2_max_abs_final_gap"] = max(
                        identity["b2_max_abs_final_gap"],
                        float(np.max(np.abs(identity_metrics[target_name]["final_gap"] - corrupt_metrics[target_name]["final_gap"]))),
                    )
                    identity["b2_max_abs_qk_margin"] = max(
                        identity["b2_max_abs_qk_margin"],
                        float(np.max(np.abs(identity_metrics[target_name]["qk_margin"] - corrupt_metrics[target_name]["qk_margin"]))),
                    )
                    identity["b2_max_abs_attention_preference"] = max(
                        identity["b2_max_abs_attention_preference"],
                        float(np.max(np.abs(identity_metrics[target_name]["attention_preference"] - corrupt_metrics[target_name]["attention_preference"]))),
                    )
                    identity["b2_prediction_changes"] += int(
                        np.sum(np.asarray(identity_metrics[target_name]["predictions"]) != np.asarray(corrupt_metrics[target_name]["predictions"]))
                    )
            all_output, all_qkv = run_patched(
                model,
                corrupt_enc,
                component_source=clean_cache,
                components=components,
                component_positions=positions,
            )
            b2[side]["all_upstream"] = target_metrics(
                model, corrupt_enc, corrupt_rows, all_output, all_qkv
            )
            all_identity_output, all_identity_qkv = run_patched(
                model,
                corrupt_enc,
                component_source=corrupt_cache,
                components=components,
                component_positions=positions,
            )
            all_identity_metrics = target_metrics(
                model, corrupt_enc, corrupt_rows, all_identity_output, all_identity_qkv
            )
            for target_name in all_identity_metrics:
                identity["b2_max_abs_final_gap"] = max(
                    identity["b2_max_abs_final_gap"],
                    float(np.max(np.abs(all_identity_metrics[target_name]["final_gap"] - corrupt_metrics[target_name]["final_gap"]))),
                )
                identity["b2_max_abs_qk_margin"] = max(
                    identity["b2_max_abs_qk_margin"],
                    float(np.max(np.abs(all_identity_metrics[target_name]["qk_margin"] - corrupt_metrics[target_name]["qk_margin"]))),
                )
                identity["b2_max_abs_attention_preference"] = max(
                    identity["b2_max_abs_attention_preference"],
                    float(np.max(np.abs(all_identity_metrics[target_name]["attention_preference"] - corrupt_metrics[target_name]["attention_preference"]))),
                )
                identity["b2_prediction_changes"] += int(
                    np.sum(np.asarray(all_identity_metrics[target_name]["predictions"]) != np.asarray(corrupt_metrics[target_name]["predictions"]))
                )

        for local, pair in enumerate(batch_pairs):
            corrupt = pair["corrupted"]
            competitor = int(corrupt["strongest_stale_token_id"])
            competitor_positions = [
                int(write["value_token"])
                for write in corrupt["writes"]
                if int(write["value_token_id"]) == competitor
            ]
            current_position = int(corrupt["current_value_span"][0])
            competitor_position = max(competitor_positions)
            record = {
                "id": pair["id"],
                "semantic_id": corrupt["semantic_id"],
                "split": core.stable_split(corrupt["semantic_id"]),
                "template": corrupt["template"],
                "seed": int(corrupt["seed"]),
                "query_position": int(query[local].item()),
                "current_position": current_position,
                "competitor_position": competitor_position,
                "recency_gap": current_position - competitor_position,
                "clean": metric_record(clean_metrics, local),
                "corrupt": metric_record(corrupt_metrics, local),
                "b1": {
                    target_name: {
                        side: {
                            key: (int(values[local]) if key == "predictions" else float(values[local]))
                            for key, values in metrics.items()
                        }
                        for side, metrics in sides.items()
                    }
                    for target_name, sides in b1.items()
                },
                "b2": {
                    side: {
                        name: metric_record(metrics, local)
                        for name, metrics in component_metrics.items()
                    }
                    for side, component_metrics in b2.items()
                },
                "attention_features": {
                    "previous_token": attention_features["previous_token"][local].tolist(),
                    "latest_write": attention_features["latest_write"][local].tolist(),
                },
            }
            records.append(record)
        print(f"upstream scan {min(start + args.batch_size, len(pairs))}/{len(pairs)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "upstream_scan_rows.jsonl", records)
    dump_json(
        out_dir / "upstream_scan.summary.json",
        {
            "model": args.model,
            "n_pairs": len(records),
            "split_counts": dict(Counter(row["split"] for row in records)),
            "targets": [f"L{layer}H{head}" for layer, head in TARGET_HEADS],
            "components_scanned": len(components),
            "component_paths": len(components) * 2,
            "identity_gate": identity,
        },
    )
    print(json.dumps(identity, indent=2))


def recovery_array(rows, path: list[str], target: str, metric: str) -> np.ndarray:
    def get(row, keys):
        value = row
        for key in keys:
            value = value[key]
        return float(value)

    clean = np.asarray([get(row, ["clean", target, metric]) for row in rows])
    corrupt = np.asarray([get(row, ["corrupt", target, metric]) for row in rows])
    patched = np.asarray([get(row, path + [metric]) for row in rows])
    return core.recovery_fraction(patched, corrupt, clean)


def rank_upstream(args) -> None:
    rows = load_jsonl(args.scan_rows)
    components = upstream_components()
    discovery = np.asarray([row["split"] == "discovery" for row in rows])
    rankings = {}
    for target in ("L8H10", "L8H2"):
        rankings[target] = {}
        for side in ("query", "key"):
            table = []
            for component in components:
                name = component_name(component)
                qk = recovery_array(rows, ["b2", side, name, target], target, "qk_margin")
                final = recovery_array(rows, ["b2", side, name, target], target, "final_gap")
                qk_mean = float(np.nanmean(qk[discovery]))
                final_mean = float(np.nanmean(final[discovery]))
                table.append(
                    {
                        **component_dict(component),
                        "discovery_qk_recovery": qk_mean,
                        "discovery_final_gap_recovery": final_mean,
                        "discovery_joint_score": min(qk_mean, final_mean),
                    }
                )
            table.sort(key=lambda row: row["discovery_joint_score"], reverse=True)
            for rank, row in enumerate(table, 1):
                row["rank"] = rank
            rankings[target][side] = table
    dump_json(
        args.out,
        {
            "selection_rule": "rank on discovery by min(mean QK recovery, mean final-gap recovery)",
            "compact_rule": "smallest top-12 prefix reaching 80% of positive all-upstream discovery restoration for both QK and final gap",
            "rankings": rankings,
        },
    )


def matched_random_component_sets(
    selected: list[tuple[str, int, int | None]], n_sets: int, seed: int
) -> list[list[tuple[str, int, int | None]]]:
    rng = random.Random(seed)
    universe = upstream_components()
    selected_set = set(selected)
    out = []
    for _ in range(n_sets):
        used = set()
        choice = []
        for kind, layer, index in selected:
            if kind == "head":
                candidates = [
                    component
                    for component in universe
                    if component[0] == "head"
                    and component[1] == layer
                    and component not in selected_set
                    and component not in used
                ]
            else:
                candidates = [
                    component
                    for component in universe
                    if component[0] == "mlp"
                    and component not in selected_set
                    and component not in used
                ]
                candidates.sort(key=lambda component: (abs(component[1] - layer), component[1]))
                nearest = min(3, len(candidates))
                candidates = candidates[:nearest]
            if not candidates:
                candidates = [
                    component
                    for component in universe
                    if component[0] == kind
                    and component not in selected_set
                    and component not in used
                ]
            if not candidates:
                candidates = [
                    component
                    for component in universe
                    if component not in selected_set and component not in used
                ]
                candidates.sort(key=lambda component: (abs(component[1] - layer), component[0]))
                candidates = candidates[: max(1, min(12, len(candidates)))]
            picked = rng.choice(candidates)
            choice.append(picked)
            used.add(picked)
        out.append(choice)
    return out


def upstream_validate(args) -> None:
    pairs = load_jsonl(args.pairs)
    pairs = slice_rows(pairs, args.offset, args.limit)
    ranking_payload = json.loads(Path(args.ranking).read_text(encoding="utf-8"))
    model, tokenizer, device = load_model(args.model, args.dtype)
    records = []
    random_definitions = {}
    identity = {
        "max_abs_final_gap": 0.0,
        "max_abs_qk_margin": 0.0,
        "max_abs_attention_preference": 0.0,
        "prediction_changes": 0,
    }
    for start in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[start : start + args.batch_size]
        clean_rows = [pair["clean"] for pair in batch_pairs]
        corrupt_rows = [pair["corrupted"] for pair in batch_pairs]
        clean_enc = core.batch_encode(tokenizer, device, clean_rows)
        corrupt_enc = core.batch_encode(tokenizer, device, corrupt_rows)
        query = core.query_positions(corrupt_enc)
        clean_output, clean_cache = run_capture(model, clean_enc)
        corrupt_output, corrupt_cache = run_capture(model, corrupt_enc)
        clean_metrics = target_metrics(model, clean_enc, corrupt_rows, clean_output, clean_cache.qkv_outputs)
        corrupt_metrics = target_metrics(model, corrupt_enc, corrupt_rows, corrupt_output, corrupt_cache.qkv_outputs)
        validation = defaultdict(lambda: defaultdict(dict))
        for target in ("L8H10", "L8H2"):
            for side_index, side in enumerate(("query", "key")):
                ranked_rows = ranking_payload["rankings"][target][side]
                ranked = [
                    (row["kind"], int(row["layer"]), None if row["head"] is None else int(row["head"]))
                    for row in ranked_rows
                ]
                positions = row_positions(corrupt_rows, side, query)
                for count in range(1, MAX_COMPACT_COMPONENTS + 1):
                    components = ranked[:count]
                    output, qkv = run_patched(
                        model,
                        corrupt_enc,
                        component_source=clean_cache,
                        components=components,
                        component_positions=positions,
                    )
                    validation[target][side][f"prefix_{count}"] = target_metrics(
                        model, corrupt_enc, corrupt_rows, output, qkv
                    )[target]

                discovery_indices = [
                    index
                    for index, pair in enumerate(batch_pairs)
                    if core.stable_split(pair["corrupted"]["semantic_id"]) == "discovery"
                ]
                # The selected size is finalized after all batches from discovery metrics.
                random_sets = matched_random_component_sets(
                    ranked[:MAX_COMPACT_COMPONENTS], args.n_random, args.seed + 100 * side_index + (0 if target == "L8H10" else 1000)
                )
                random_definitions[f"{target}_{side}"] = {
                    "selected_top12": [component_dict(component) for component in ranked[:MAX_COMPACT_COMPONENTS]],
                    "sets": [
                        [component_dict(component) for component in component_set]
                        for component_set in random_sets
                    ],
                }
                for random_index, components in enumerate(random_sets):
                    output, qkv = run_patched(
                        model,
                        corrupt_enc,
                        component_source=clean_cache,
                        components=components,
                        component_positions=positions,
                    )
                    validation[target][side][f"random12_{random_index}"] = target_metrics(
                        model, corrupt_enc, corrupt_rows, output, qkv
                    )[target]
                identity_output, identity_qkv = run_patched(
                    model,
                    corrupt_enc,
                    component_source=corrupt_cache,
                    components=ranked[:MAX_COMPACT_COMPONENTS],
                    component_positions=positions,
                )
                identity_metrics = target_metrics(
                    model, corrupt_enc, corrupt_rows, identity_output, identity_qkv
                )[target]
                identity["max_abs_final_gap"] = max(
                    identity["max_abs_final_gap"],
                    float(np.max(np.abs(identity_metrics["final_gap"] - corrupt_metrics[target]["final_gap"]))),
                )
                identity["max_abs_qk_margin"] = max(
                    identity["max_abs_qk_margin"],
                    float(np.max(np.abs(identity_metrics["qk_margin"] - corrupt_metrics[target]["qk_margin"]))),
                )
                identity["max_abs_attention_preference"] = max(
                    identity["max_abs_attention_preference"],
                    float(np.max(np.abs(identity_metrics["attention_preference"] - corrupt_metrics[target]["attention_preference"]))),
                )
                identity["prediction_changes"] += int(
                    np.sum(np.asarray(identity_metrics["predictions"]) != np.asarray(corrupt_metrics[target]["predictions"]))
                )
        for local, pair in enumerate(batch_pairs):
            records.append(
                {
                    "id": pair["id"],
                    "semantic_id": pair["corrupted"]["semantic_id"],
                    "split": core.stable_split(pair["corrupted"]["semantic_id"]),
                    "clean": metric_record(clean_metrics, local),
                    "corrupt": metric_record(corrupt_metrics, local),
                    "validation": {
                        target: {
                            side: {
                                name: {
                                    key: (int(values[local]) if key == "predictions" else float(values[local]))
                                    for key, values in metrics.items()
                                }
                                for name, metrics in paths.items()
                            }
                            for side, paths in sides.items()
                        }
                        for target, sides in validation.items()
                    },
                }
            )
        print(f"upstream validate {min(start + args.batch_size, len(pairs))}/{len(pairs)}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "upstream_validate_rows.jsonl", records)
    dump_json(
        out_dir / "upstream_validate.summary.json",
        {
            "model": args.model,
            "n_pairs": len(records),
            "prefixes": MAX_COMPACT_COMPONENTS,
            "random_sets_per_target_side": args.n_random,
            "random_matching": "same type and layer where possible; nearest-layer same type for MLPs; nearest-layer fallback only if the selected set exhausts a type",
            "random_definitions": random_definitions,
            "identity_gate": identity,
            "note": "random12 controls are generated from the discovery top-12 composition; final selected-prefix controls are derived conservatively in CPU analysis",
        },
    )


def merge_upstream_validation(args) -> None:
    shard_root = Path(args.shard_root)
    summaries = []
    records = []
    for shard_index in range(args.n_shards):
        shard_dir = shard_root / f"shard_{shard_index}"
        shard_rows = load_jsonl(shard_dir / "upstream_validate_rows.jsonl")
        shard_summary = json.loads(
            (shard_dir / "upstream_validate.summary.json").read_text(encoding="utf-8")
        )
        if len(shard_rows) != shard_summary["n_pairs"]:
            raise ValueError(f"shard {shard_index} row-count mismatch")
        records.extend(shard_rows)
        summaries.append(shard_summary)
    identifiers = [row["id"] for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate validation pair across shards")
    reference_random = summaries[0]["random_definitions"]
    if any(summary["random_definitions"] != reference_random for summary in summaries[1:]):
        raise ValueError("random component definitions differ across shards")
    if len(records) != args.expected_pairs:
        raise ValueError(f"expected {args.expected_pairs} validation pairs, got {len(records)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "upstream_validate_rows.jsonl", records)
    dump_json(
        out_dir / "upstream_validate.summary.json",
        {
            "model": summaries[0]["model"],
            "n_pairs": len(records),
            "prefixes": summaries[0]["prefixes"],
            "random_sets_per_target_side": summaries[0]["random_sets_per_target_side"],
            "random_matching": summaries[0]["random_matching"],
            "random_definitions": reference_random,
            "identity_gate": merge_identity_gates(
                [summary["identity_gate"] for summary in summaries]
            ),
            "shards": args.n_shards,
            "note": "merged disjoint stable-order pair shards; random definitions verified identical",
        },
    )


def bootstrap_spearman(x, y, n_boot: int, seed: int) -> dict:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    observed = float(spearmanr(x, y).statistic) if len(x) > 2 else math.nan
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        indices = rng.integers(0, len(x), len(x))
        value = spearmanr(x[indices], y[indices]).statistic
        if np.isfinite(value):
            boot.append(float(value))
    return {
        "n": int(len(x)),
        "rho": observed,
        "bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist() if boot else [math.nan, math.nan],
    }


def summarize_restoration(values, mask, n_boot, seed) -> dict:
    return core.paired_bootstrap(np.asarray(values)[mask], n_boot, seed)


def analyze_upstream(args) -> None:
    scan = load_jsonl(args.scan_rows)
    validation = load_jsonl(args.validate_rows)
    ranking_payload = json.loads(Path(args.ranking).read_text(encoding="utf-8"))
    evaluation = np.asarray([row["split"] == "evaluation" for row in scan])
    discovery = ~evaluation
    validation_discovery = np.asarray([row["split"] == "discovery" for row in validation])
    validation_evaluation = ~validation_discovery
    summary = {
        "protocol": {
            "split_counts": {"discovery": int(discovery.sum()), "evaluation": int(evaluation.sum())},
            "b1": "report evaluation only; side dominance requires bootstrap CI of query-minus-key QK recovery to exclude zero",
            "b2_ranking": ranking_payload["selection_rule"],
            "b2_compact": ranking_payload["compact_rule"],
            "small_set_max": MAX_COMPACT_COMPONENTS,
            "b3_gate": "aggregate held-out effect-recency Spearman CI excludes zero and at least half of selected head feeders are top-decile previous-token or latest-write heads",
        },
        "b1": {},
        "b2": {},
        "b3": {},
        "b4": {"run": False, "reason": "existing Phase-1 harvest is k=4 only; a new k=1..5 activation sweep is not cheap reuse"},
    }
    for target_index, target in enumerate(("L8H10", "L8H2")):
        clean = {metric: np.asarray([row["clean"][target][metric] for row in scan]) for metric in ("qk_margin", "attention_preference", "final_gap")}
        corrupt = {metric: np.asarray([row["corrupt"][target][metric] for row in scan]) for metric in clean}
        b1_target = {"sides": {}}
        side_recovery = {}
        for side_index, side in enumerate(("query", "key", "both")):
            b1_target["sides"][side] = {}
            side_recovery[side] = {}
            for metric_index, metric in enumerate(clean):
                patched = np.asarray([row["b1"][target][side][metric] for row in scan])
                recovery = core.recovery_fraction(patched, corrupt[metric], clean[metric])
                side_recovery[side][metric] = recovery
                b1_target["sides"][side][metric] = summarize_restoration(
                    recovery, evaluation, args.n_boot, args.seed + 1000 * target_index + 100 * side_index + metric_index
                )
                b1_target["sides"][side][f"raw_change_{metric}"] = core.paired_bootstrap(
                    (patched - corrupt[metric])[evaluation], args.n_boot, args.seed + 4000 + metric_index
                )
        difference = side_recovery["query"]["qk_margin"] - side_recovery["key"]["qk_margin"]
        difference_summary = summarize_restoration(
            difference, evaluation, args.n_boot, args.seed + 5000 + target_index
        )
        if difference_summary["ci"][0] > 0:
            dominant = "query"
        elif difference_summary["ci"][1] < 0:
            dominant = "key"
        else:
            dominant = "mixed_or_unresolved"
        b1_target["query_minus_key_qk_recovery"] = difference_summary
        b1_target["dominant_side"] = dominant
        summary["b1"][target] = b1_target

        summary["b2"][target] = {}
        summary["b3"][target] = {}
        for side_index, side in enumerate(("query", "key")):
            ranked_rows = ranking_payload["rankings"][target][side]
            ranked_names = [row["name"] for row in ranked_rows]
            all_qk = recovery_array(
                scan, ["b2", side, "all_upstream", target], target, "qk_margin"
            )
            all_final = recovery_array(
                scan, ["b2", side, "all_upstream", target], target, "final_gap"
            )
            all_discovery_qk = float(np.nanmean(all_qk[discovery]))
            all_discovery_final = float(np.nanmean(all_final[discovery]))
            prefix_table = []
            for count in range(1, MAX_COMPACT_COMPONENTS + 1):
                name = f"prefix_{count}"
                qk = recovery_array(
                    validation, ["validation", target, side, name], target, "qk_margin"
                )
                final = recovery_array(
                    validation, ["validation", target, side, name], target, "final_gap"
                )
                prefix_table.append(
                    {
                        "count": count,
                        "discovery_qk_recovery": float(np.nanmean(qk[validation_discovery])),
                        "discovery_final_gap_recovery": float(np.nanmean(final[validation_discovery])),
                        "evaluation_qk": summarize_restoration(qk, validation_evaluation, args.n_boot, args.seed + 6000 + count),
                        "evaluation_final_gap": summarize_restoration(final, validation_evaluation, args.n_boot, args.seed + 7000 + count),
                    }
                )
            crossing = None
            if all_discovery_qk > 0 and all_discovery_final > 0:
                for row in prefix_table:
                    if (
                        row["discovery_qk_recovery"] >= 0.8 * all_discovery_qk
                        and row["discovery_final_gap_recovery"] >= 0.8 * all_discovery_final
                    ):
                        crossing = row["count"]
                        break
            selected_count = MAX_COMPACT_COMPONENTS
            selected = ranked_rows[:selected_count]
            selected_row = prefix_table[selected_count - 1]

            random_qk = []
            random_final = []
            for random_index in range(args.n_random):
                name = f"random12_{random_index}"
                qk = recovery_array(
                    validation, ["validation", target, side, name], target, "qk_margin"
                )
                final = recovery_array(
                    validation, ["validation", target, side, name], target, "final_gap"
                )
                random_qk.append(float(np.nanmean(qk[validation_evaluation])))
                random_final.append(float(np.nanmean(final[validation_evaluation])))
            qk_random_summary = quantile_summary(random_qk)
            final_random_summary = quantile_summary(random_final)
            compact = crossing is not None
            exceeds_random = bool(
                selected_row["evaluation_qk"]["mean"] > qk_random_summary["quantile95"][1]
                and selected_row["evaluation_final_gap"]["mean"] > final_random_summary["quantile95"][1]
            )
            summary["b2"][target][side] = {
                "all_upstream_discovery": {"qk": all_discovery_qk, "final_gap": all_discovery_final},
                "all_upstream_evaluation": {
                    "qk": summarize_restoration(all_qk, evaluation, args.n_boot, args.seed + 8000),
                    "final_gap": summarize_restoration(all_final, evaluation, args.n_boot, args.seed + 8001),
                },
                "prefixes": prefix_table,
                "selected_count": selected_count,
                "selected_components": selected,
                "first_80pct_crossing": crossing,
                "crossed_80pct_by_12": compact,
                "selected_evaluation": {
                    "qk": selected_row["evaluation_qk"],
                    "final_gap": selected_row["evaluation_final_gap"],
                },
                "random12_evaluation": {"qk": qk_random_summary, "final_gap": final_random_summary},
                "matched_control_exceeded": exceeds_random,
                "root_cause_gate_pass": bool(compact and exceeds_random),
            }

            previous = np.asarray([row["attention_features"]["previous_token"] for row in scan])
            latest = np.asarray([row["attention_features"]["latest_write"] for row in scan])
            head_previous_means = np.nanmean(previous[evaluation], axis=0)
            head_latest_means = np.nanmean(latest[evaluation], axis=0)
            previous_threshold = float(np.quantile(head_previous_means, 0.9))
            latest_threshold = float(np.quantile(head_latest_means, 0.9))
            feeder_rows = []
            structured_count = 0
            head_count = 0
            recency_gap = np.asarray([row["recency_gap"] for row in scan], dtype=np.float64)
            for feeder_index, feeder in enumerate(selected):
                name = feeder["name"]
                qk = recovery_array(
                    scan, ["b2", side, name, target], target, "qk_margin"
                )
                correlation = bootstrap_spearman(
                    recency_gap[evaluation], qk[evaluation], args.n_boot, args.seed + 9000 + feeder_index
                )
                row = {**feeder, "effect_recency_correlation": correlation}
                if feeder["kind"] == "head":
                    head_count += 1
                    layer = int(feeder["layer"])
                    head = int(feeder["head"])
                    previous_value = float(head_previous_means[layer, head])
                    latest_value = float(head_latest_means[layer, head])
                    structured = bool(
                        previous_value >= previous_threshold or latest_value >= latest_threshold
                    )
                    structured_count += int(structured)
                    row.update(
                        {
                            "previous_token_attention": previous_value,
                            "latest_write_attention_advantage": latest_value,
                            "top_decile_structure": structured,
                        }
                    )
                feeder_rows.append(row)
            selected_qk = recovery_array(
                validation,
                ["validation", target, side, f"prefix_{selected_count}"],
                target,
                "qk_margin",
            )
            aggregate_correlation = bootstrap_spearman(
                recency_gap[evaluation],
                selected_qk[validation_evaluation],
                args.n_boot,
                args.seed + 10000 + target_index * 10 + side_index,
            )
            correlation_nonzero = bool(
                aggregate_correlation["bootstrap95"][0] > 0
                or aggregate_correlation["bootstrap95"][1] < 0
            )
            structure_pass = bool(head_count > 0 and structured_count / head_count >= 0.5)
            summary["b3"][target][side] = {
                "selected_feeders": feeder_rows,
                "previous_token_top_decile_threshold": previous_threshold,
                "latest_write_top_decile_threshold": latest_threshold,
                "structured_head_fraction": structured_count / head_count if head_count else math.nan,
                "aggregate_effect_recency_correlation": aggregate_correlation,
                "positionality_gate_pass": bool(correlation_nonzero and structure_pass),
            }

    scan_summary = json.loads(Path(args.scan_summary).read_text(encoding="utf-8"))
    validate_summary = json.loads(Path(args.validate_summary).read_text(encoding="utf-8"))
    summary["identity_gate"] = {
        "scan": scan_summary["identity_gate"],
        "validation": validate_summary["identity_gate"],
        "pass": bool(
            max(
                scan_summary["identity_gate"]["b1_max_abs_final_gap"],
                scan_summary["identity_gate"]["b1_max_abs_qk_margin"],
                scan_summary["identity_gate"]["b1_max_abs_attention_preference"],
                scan_summary["identity_gate"]["b2_max_abs_final_gap"],
                scan_summary["identity_gate"]["b2_max_abs_qk_margin"],
                scan_summary["identity_gate"]["b2_max_abs_attention_preference"],
                validate_summary["identity_gate"]["max_abs_final_gap"],
                validate_summary["identity_gate"]["max_abs_qk_margin"],
                validate_summary["identity_gate"]["max_abs_attention_preference"],
            )
            == 0.0
            and scan_summary["identity_gate"]["b1_prediction_changes"] == 0
            and scan_summary["identity_gate"]["b2_prediction_changes"] == 0
            and validate_summary["identity_gate"]["prediction_changes"] == 0
        ),
    }
    dump_json(args.out, summary)


def self_test(_args) -> None:
    assert slice_rows(list(range(6)), offset=2, limit=3) == [2, 3, 4]
    assert merge_identity_gates(
        [
            {"max_abs_final_gap": 0.0, "prediction_changes": 1},
            {"max_abs_final_gap": 0.25, "prediction_changes": 2},
        ]
    ) == {"max_abs_final_gap": 0.25, "prediction_changes": 3}
    batch_indices, token_indices = flatten_patch_positions([[3], [5, 7], [11]])
    assert batch_indices == [0, 1, 1, 2]
    assert token_indices == [3, 5, 7, 11]
    assert len(upstream_components()) == 104
    assert component_name(("head", 3, 4)) == "L3H4"
    assert component_name(("mlp", 2, None)) == "L2MLP"
    random_sets = matched_random_component_sets(
        [("head", 2, 1), ("mlp", 4, None)], 8, 0
    )
    assert len(random_sets) == 8
    assert all(len(row) == 2 and len(set(row)) == 2 for row in random_sets)
    values = core.recovery_fraction(np.asarray([0.0]), np.asarray([-1.0]), np.asarray([1.0]))
    assert np.allclose(values, 0.5)
    toy = [{
        "clean": {"L8H10": {"qk_margin": 1.0}},
        "corrupt": {"L8H10": {"qk_margin": -1.0}},
        "validation": {"L8H10": {"query": {"prefix_1": {"qk_margin": 0.0}}}},
    }]
    restored = recovery_array(
        toy, ["validation", "L8H10", "query", "prefix_1"], "L8H10", "qk_margin"
    )
    assert np.allclose(restored, 0.5)
    print("stage_n_phase1b self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(func=self_test)

    ablate = sub.add_parser("ablate-heldout")
    ablate.add_argument("--pool", required=True)
    ablate.add_argument("--harvest-dir", required=True)
    ablate.add_argument("--cumulative-summary", required=True)
    ablate.add_argument("--phase1-ablation", required=True)
    ablate.add_argument("--out-dir", required=True)
    ablate.add_argument("--model", default="EleutherAI/pythia-160m")
    ablate.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ablate.add_argument("--batch-size", type=int, default=32)
    ablate.add_argument("--max-heads", type=int, default=12)
    ablate.add_argument("--n-random", type=int, default=64)
    ablate.add_argument("--n-boot", type=int, default=2000)
    ablate.add_argument("--seed", type=int, default=20260809)
    ablate.set_defaults(func=ablate_heldout)

    scan = sub.add_parser("upstream-scan")
    scan.add_argument("--pairs", required=True)
    scan.add_argument("--out-dir", required=True)
    scan.add_argument("--model", default="EleutherAI/pythia-160m")
    scan.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    scan.add_argument("--batch-size", type=int, default=16)
    scan.add_argument("--limit", type=int)
    scan.set_defaults(func=upstream_scan)

    rank = sub.add_parser("rank-upstream")
    rank.add_argument("--scan-rows", required=True)
    rank.add_argument("--out", required=True)
    rank.set_defaults(func=rank_upstream)

    validate = sub.add_parser("upstream-validate")
    validate.add_argument("--pairs", required=True)
    validate.add_argument("--ranking", required=True)
    validate.add_argument("--out-dir", required=True)
    validate.add_argument("--model", default="EleutherAI/pythia-160m")
    validate.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    validate.add_argument("--batch-size", type=int, default=16)
    validate.add_argument("--offset", type=int, default=0)
    validate.add_argument("--limit", type=int)
    validate.add_argument("--n-random", type=int, default=64)
    validate.add_argument("--seed", type=int, default=20260809)
    validate.set_defaults(func=upstream_validate)

    merge = sub.add_parser("merge-upstream-validation")
    merge.add_argument("--shard-root", required=True)
    merge.add_argument("--n-shards", type=int, required=True)
    merge.add_argument("--expected-pairs", type=int, default=96)
    merge.add_argument("--out-dir", required=True)
    merge.set_defaults(func=merge_upstream_validation)

    analyze = sub.add_parser("analyze-upstream")
    analyze.add_argument("--scan-rows", required=True)
    analyze.add_argument("--scan-summary", required=True)
    analyze.add_argument("--validate-rows", required=True)
    analyze.add_argument("--validate-summary", required=True)
    analyze.add_argument("--ranking", required=True)
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--n-random", type=int, default=64)
    analyze.add_argument("--n-boot", type=int, default=2000)
    analyze.add_argument("--seed", type=int, default=20260809)
    analyze.set_defaults(func=analyze_upstream)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
