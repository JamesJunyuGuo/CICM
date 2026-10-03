"""Stage N exhaustive circuit analysis for GPT-NeoX/Pythia-160m."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pythia_eval import classify_token, dump_json, load_model, wilson
from pythia_gen import dump_jsonl, load_jsonl


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def paired_bootstrap(values, n_boot: int = 2000, seed: int = 0) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "n": int(len(values)),
        "mean": float(values.mean()) if len(values) else math.nan,
        "ci": bootstrap_ci(values, n_boot=n_boot, seed=seed),
    }


def stable_split(identifier: str) -> str:
    """Deterministic discovery/evaluation split for circuit selection."""
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return "discovery" if digest[0] % 2 == 0 else "evaluation"


def summarize_binary(values, n_boot: int = 2000, seed: int = 0) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "rate": float(values.mean()) if len(values) else math.nan,
        "wilson95": wilson(int(values.sum()), int(len(values))),
        "bootstrap95": bootstrap_ci(values, n_boot=n_boot, seed=seed),
    }


def pooled_log_length_residuals(values, lengths) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(lengths) & (lengths > 0)
    residuals = np.full(len(values), np.nan, dtype=np.float64)
    if finite.sum() < 2:
        return residuals
    design = np.column_stack([np.ones(finite.sum()), np.log(lengths[finite])])
    coef, *_ = np.linalg.lstsq(design, values[finite], rcond=None)
    residuals[finite] = values[finite] - design @ coef
    return residuals


def shuffle_delta_null(values, labels, a: str, b: str, n_shuffle: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    finite = np.isfinite(values) & np.isin(labels, [a, b])
    values = values[finite]
    labels = labels[finite].copy()
    rng = np.random.default_rng(seed)
    out = np.empty(n_shuffle, dtype=np.float64)
    for i in range(n_shuffle):
        rng.shuffle(labels)
        out[i] = values[labels == a].mean() - values[labels == b].mean()
    return out


def delta_summary(values, rows: list[dict], a: str, b: str, n_boot: int, n_shuffle: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    lengths = np.asarray([row["prompt_tokens"] for row in rows], dtype=np.float64)
    residuals = pooled_log_length_residuals(values, lengths)
    a_values = values[labels == a]
    b_values = values[labels == b]
    a_resid = residuals[labels == a]
    b_resid = residuals[labels == b]
    raw = float(np.nanmean(a_values) - np.nanmean(b_values))
    controlled = float(np.nanmean(a_resid) - np.nanmean(b_resid))
    raw_boot = []
    controlled_boot = []
    rng = np.random.default_rng(seed)
    a_values = a_values[np.isfinite(a_values)]
    b_values = b_values[np.isfinite(b_values)]
    a_resid = a_resid[np.isfinite(a_resid)]
    b_resid = b_resid[np.isfinite(b_resid)]
    for _ in range(n_boot):
        raw_boot.append(
            rng.choice(a_values, len(a_values), replace=True).mean()
            - rng.choice(b_values, len(b_values), replace=True).mean()
        )
        controlled_boot.append(
            rng.choice(a_resid, len(a_resid), replace=True).mean()
            - rng.choice(b_resid, len(b_resid), replace=True).mean()
        )
    raw_null = shuffle_delta_null(values, labels, a, b, n_shuffle, seed + 1)
    controlled_null = shuffle_delta_null(residuals, labels, a, b, n_shuffle, seed + 2)
    return {
        "group_a": a,
        "group_b": b,
        "n_a": int(np.sum(labels == a)),
        "n_b": int(np.sum(labels == b)),
        "raw_delta": raw,
        "raw_ci": [float(np.quantile(raw_boot, 0.025)), float(np.quantile(raw_boot, 0.975))],
        "raw_shuffle95": [float(np.quantile(raw_null, 0.025)), float(np.quantile(raw_null, 0.975))],
        "length_controlled_delta": controlled,
        "length_controlled_ci": [
            float(np.quantile(controlled_boot, 0.025)),
            float(np.quantile(controlled_boot, 0.975)),
        ],
        "length_controlled_shuffle95": [
            float(np.quantile(controlled_null, 0.025)),
            float(np.quantile(controlled_null, 0.975)),
        ],
    }


class ComponentCapture:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.head_inputs: dict[int, object] = {}
        self.mlp_outputs: dict[int, object] = {}
        self.residual_outputs: dict[int, object] = {}
        self.final_residual = None

    def clear(self) -> None:
        self.head_inputs.clear()
        self.mlp_outputs.clear()
        self.residual_outputs.clear()
        self.final_residual = None

    def __enter__(self):
        def dense_pre(layer):
            def hook(_module, inputs):
                self.head_inputs[layer] = inputs[0].detach()
            return hook

        def mlp_hook(layer):
            def hook(_module, _inputs, output):
                self.mlp_outputs[layer] = output.detach()
            return hook

        def layer_hook(layer):
            def hook(_module, _inputs, output):
                self.residual_outputs[layer] = output[0].detach()
            return hook

        def final_pre(_module, inputs):
            self.final_residual = inputs[0].detach()

        for layer, block in enumerate(self.model.gpt_neox.layers):
            self.handles.append(block.attention.dense.register_forward_pre_hook(dense_pre(layer)))
            self.handles.append(block.mlp.register_forward_hook(mlp_hook(layer)))
            self.handles.append(block.register_forward_hook(layer_hook(layer)))
        self.handles.append(self.model.gpt_neox.final_layer_norm.register_forward_pre_hook(final_pre))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()


def encode_row(tokenizer, device, row: dict):
    return tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(device)


def competitor_token_id(row: dict) -> int:
    if row.get("label") == "within_stale":
        return int(row["pred_token_id"])
    candidate = row.get("strongest_stale_token_id")
    if candidate is not None:
        return int(candidate)
    stale = row.get("stale_token_ids", [])
    return int(stale[0]) if stale else int(row["gold_token_id"])


def layernorm_direct(component, final_residual, layernorm, logit_direction) -> float:
    import torch

    component = component.float()
    residual = final_residual.float()
    centered = component - component.mean(dim=-1, keepdim=True)
    scale = torch.sqrt(residual.var(dim=-1, unbiased=False, keepdim=True) + layernorm.eps)
    normalized_component = centered / scale * layernorm.weight.float()
    return float(torch.dot(normalized_component, logit_direction.float()).item())


def one_row_harvest(model, tokenizer, device, row: dict, capture: ComponentCapture) -> dict:
    import torch

    capture.clear()
    enc = encode_row(tokenizer, device, row)
    with torch.no_grad():
        output = model(
            **enc,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    query = enc["input_ids"].shape[1] - 1
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // n_heads)
    gold_id = int(row["gold_token_id"])
    stale_id = competitor_token_id(row)
    logits = output.logits[0, query]
    logit_direction = model.embed_out.weight[gold_id] - model.embed_out.weight[stale_id]
    final_residual = capture.final_residual[0, query]

    residuals = [output.hidden_states[0][0, query].detach().float().cpu().numpy()]
    for layer in range(n_layers):
        residuals.append(capture.residual_outputs[layer][0, query].detach().float().cpu().numpy())
    residuals.append(output.hidden_states[-1][0, query].detach().float().cpu().numpy())

    logit_lens = []
    for residual in residuals[:-1]:
        normalized = model.gpt_neox.final_layer_norm(torch.as_tensor(residual, device=device, dtype=final_residual.dtype))
        lens_logits = model.embed_out(normalized)
        logit_lens.append(float((lens_logits[gold_id] - lens_logits[stale_id]).item()))
    logit_lens.append(float((logits[gold_id] - logits[stale_id]).item()))

    head_dla = np.zeros((n_layers, n_heads), dtype=np.float32)
    mlp_dla = np.zeros(n_layers, dtype=np.float32)
    for layer in range(n_layers):
        dense = model.gpt_neox.layers[layer].attention.dense
        head_input = capture.head_inputs[layer][0, query]
        for head in range(n_heads):
            sl = slice(head * head_dim, (head + 1) * head_dim)
            component = torch.matmul(head_input[sl], dense.weight[:, sl].T)
            head_dla[layer, head] = layernorm_direct(
                component, final_residual, model.gpt_neox.final_layer_norm, logit_direction
            )
        mlp_dla[layer] = layernorm_direct(
            capture.mlp_outputs[layer][0, query],
            final_residual,
            model.gpt_neox.final_layer_norm,
            logit_direction,
        )

    current_mass = np.zeros((n_layers, n_heads), dtype=np.float32)
    stale_mass = np.zeros((n_layers, n_heads), dtype=np.float32)
    identity_mass = np.zeros((n_layers, n_heads), dtype=np.float32)
    current_positions = row["current_value_span"]
    stale_positions = row["stale_value_spans"]
    identity_positions = row["target_identity_spans"]
    for layer, attention in enumerate(output.attentions):
        final_attention = attention[0, :, query, :].detach().float().cpu().numpy()
        current_mass[layer] = final_attention[:, current_positions].sum(axis=1)
        if stale_positions:
            stale_mass[layer] = final_attention[:, stale_positions].sum(axis=1)
        identity_mass[layer] = final_attention[:, identity_positions].sum(axis=1)
    return {
        "hidden": np.stack(residuals, axis=0),
        "logit_lens": np.asarray(logit_lens, dtype=np.float32),
        "head_dla": head_dla,
        "mlp_dla": mlp_dla,
        "attn_current": current_mass,
        "attn_stale": stale_mass,
        "attn_identity": identity_mass,
        "final_gap": float((logits[gold_id] - logits[stale_id]).item()),
    }


def harvest(args) -> None:
    rows = load_jsonl(args.pool)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer, device = load_model(args.model, args.dtype)
    arrays = defaultdict(list)
    index_rows = []
    with ComponentCapture(model) as capture:
        for index, row in enumerate(rows):
            result = one_row_harvest(model, tokenizer, device, row, capture)
            for key, value in result.items():
                arrays[key].append(value)
            index_rows.append(
                {
                    "id": row["id"],
                    "semantic_id": row["semantic_id"],
                    "label": row["label"],
                    "variant": row["variant"],
                    "template": row["template"],
                    "seed": row["seed"],
                    "k": row["k"],
                    "prompt_tokens": row["prompt_tokens"],
                    "current_value": row["gold"],
                    "gold_token_id": row["gold_token_id"],
                    "competitor_token_id": competitor_token_id(row),
                }
            )
            if (index + 1) % 20 == 0 or index + 1 == len(rows):
                print(f"harvested {index + 1}/{len(rows)}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "harvest.npz", **{key: np.asarray(value) for key, value in arrays.items()})
    dump_jsonl(out_dir / "harvest_index.jsonl", index_rows)
    dump_json(
        out_dir / "harvest.summary.json",
        {
            "model": args.model,
            "dtype": args.dtype,
            "n": len(rows),
            "label_counts": dict(Counter(row["label"] for row in rows)),
            "hidden_shape": list(np.asarray(arrays["hidden"]).shape),
            "attention_shape": list(np.asarray(arrays["attn_current"]).shape),
        },
    )


def fit_probe_layer(hidden: np.ndarray, rows: list[dict], layer: int, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    x = hidden[:, layer, :].astype(np.float32)
    value_labels = np.asarray([row["current_value"] for row in rows], dtype=object)
    groups = np.asarray([row["semantic_id"] for row in rows], dtype=object)
    encoder = LabelEncoder()
    y = encoder.fit_transform(value_labels)
    splitter = GroupKFold(n_splits=5)
    probabilities = np.full((len(rows), len(encoder.classes_)), np.nan, dtype=np.float64)
    predictions = np.full(len(rows), -1, dtype=np.int64)
    for train, test in splitter.split(x, y, groups):
        scaler = StandardScaler()
        train_x = scaler.fit_transform(x[train])
        test_x = scaler.transform(x[test])
        classifier = LogisticRegression(
            C=1.0,
            penalty="l2",
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        classifier.fit(train_x, y[train])
        fold_probability = classifier.predict_proba(test_x)
        for local, class_id in enumerate(classifier.classes_):
            probabilities[test, class_id] = fold_probability[:, local]
        predictions[test] = classifier.predict(test_x)
    class_to_index = {str(value): i for i, value in enumerate(encoder.classes_)}
    true_scores = np.asarray(
        [probabilities[i, class_to_index[str(value)]] for i, value in enumerate(value_labels)],
        dtype=np.float64,
    )
    valid = np.isfinite(true_scores)
    return {
        "layer": layer,
        "n": int(valid.sum()),
        "n_classes": len(encoder.classes_),
        "cv_accuracy": float(accuracy_score(y[valid], predictions[valid])),
        "classes": encoder.classes_.tolist(),
        "probabilities": probabilities,
        "true_scores": true_scores,
        "value_labels": value_labels,
    }


def value_shuffle_null(probe: dict, rows: list[dict], target_label: str, n_shuffle: int, seed: int) -> np.ndarray:
    labels = probe["value_labels"].copy()
    probabilities = probe["probabilities"]
    class_to_index = {str(value): i for i, value in enumerate(probe["classes"])}
    group = np.asarray([row["label"] == target_label for row in rows], dtype=bool)
    semantic_groups = np.asarray([row["semantic_id"] for row in rows], dtype=object)
    unique_groups = sorted(set(semantic_groups.tolist()))
    group_labels = np.asarray([
        labels[np.flatnonzero(semantic_groups == semantic_id)[0]] for semantic_id in unique_groups
    ], dtype=object)
    rng = np.random.default_rng(seed)
    out = np.empty(n_shuffle, dtype=np.float64)
    for iteration in range(n_shuffle):
        shuffled_group_labels = group_labels.copy()
        rng.shuffle(shuffled_group_labels)
        shuffled_map = dict(zip(unique_groups, shuffled_group_labels))
        shuffled_labels = np.asarray([shuffled_map[semantic_id] for semantic_id in semantic_groups])
        scores = np.asarray(
            [probabilities[i, class_to_index[str(value)]] for i, value in enumerate(shuffled_labels)],
            dtype=np.float64,
        )
        out[iteration] = np.nanmean(scores[group])
    return out


def attention_head_table(data, rows: list[dict], n_boot: int, n_shuffle: int, seed: int) -> list[dict]:
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    current = data["attn_current"]
    stale = data["attn_stale"]
    identity = data["attn_identity"]
    ratio = stale / np.maximum(current + stale, 1e-12)
    table = []
    for layer in range(current.shape[1]):
        for head in range(current.shape[2]):
            correct = labels == "correct_current"
            failure = labels == "within_stale"
            delta = float(
                ratio[failure, layer, head].mean() - ratio[correct, layer, head].mean()
            )
            controlled = delta_summary(
                ratio[:, layer, head],
                rows,
                "within_stale",
                "correct_current",
                n_boot,
                n_shuffle,
                seed + layer * 100 + head,
            )
            table.append(
                {
                    "layer": layer,
                    "head": head,
                    "correct_current_attention": float(current[correct, layer, head].mean()),
                    "failure_current_attention": float(current[failure, layer, head].mean()),
                    "correct_stale_attention": float(stale[correct, layer, head].mean()),
                    "failure_stale_attention": float(stale[failure, layer, head].mean()),
                    "correct_identity_attention": float(identity[correct, layer, head].mean()),
                    "failure_identity_attention": float(identity[failure, layer, head].mean()),
                    "failure_minus_correct_stale_ratio": delta,
                    "within_vs_correct": controlled,
                }
            )
    return table


def analyze(args) -> None:
    out_dir = Path(args.harvest_dir)
    rows = load_jsonl(out_dir / "harvest_index.jsonl")
    data = np.load(out_dir / "harvest.npz")
    hidden = data["hidden"]
    probe_layers = []
    probes = []
    selected_layers = (
        list(range(hidden.shape[1]))
        if args.probe_layers is None
        else [layer if layer >= 0 else hidden.shape[1] + layer for layer in args.probe_layers]
    )
    if hidden.shape[1] - 1 not in selected_layers:
        raise ValueError("probe layer selection must include the final checkpoint")
    for layer in selected_layers:
        probe = fit_probe_layer(hidden, rows, layer, args.seed + layer)
        probes.append(probe)
        probe_layers.append(
            {
                "layer": layer,
                "cv_accuracy": probe["cv_accuracy"],
                "valid_oos_score_n": int(np.isfinite(probe["true_scores"]).sum()),
                "valid_oos_score_fraction": float(np.isfinite(probe["true_scores"]).mean()),
                "within_stale_true_current_score": float(
                    np.nanmean(probe["true_scores"][[r["label"] == "within_stale" for r in rows]])
                ),
                "correct_true_current_score": float(
                    np.nanmean(probe["true_scores"][[r["label"] == "correct_current" for r in rows]])
                ),
            }
        )
        print(f"fit grouped probe checkpoint {layer} ({len(probes)}/{len(selected_layers)})", flush=True)
    final_probe = probes[selected_layers.index(hidden.shape[1] - 1)]
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    within_mask = labels == "within_stale"
    observed = float(np.nanmean(final_probe["true_scores"][within_mask]))
    null = value_shuffle_null(final_probe, rows, "within_stale", args.n_shuffle, args.seed + 100)
    probe_delta = delta_summary(
        final_probe["true_scores"],
        rows,
        "within_stale",
        "correct_current",
        args.n_boot,
        args.n_shuffle,
        args.seed + 200,
    )
    head_table = attention_head_table(data, rows, args.n_boot, args.n_shuffle, args.seed + 400)
    ratio = data["attn_stale"] / np.maximum(data["attn_stale"] + data["attn_current"], 1e-12)
    final_ratio = ratio[:, -1, :].mean(axis=1)
    attention_delta = delta_summary(
        final_ratio,
        rows,
        "within_stale",
        "correct_current",
        args.n_boot,
        args.n_shuffle,
        args.seed + 300,
    )
    dla_heads = []
    for layer in range(data["head_dla"].shape[1]):
        for head in range(data["head_dla"].shape[2]):
            values = data["head_dla"][:, layer, head]
            dla_heads.append(
                {
                    "layer": layer,
                    "head": head,
                    "correct_mean": float(values[labels == "correct_current"].mean()),
                    "failure_mean": float(values[labels == "within_stale"].mean()),
                    "correct_minus_failure": float(
                        values[labels == "correct_current"].mean()
                        - values[labels == "within_stale"].mean()
                    ),
                }
            )
    summary = {
        "stage": "N-3a-3c",
        "model": args.model,
        "n": len(rows),
        "label_counts": dict(Counter(labels.tolist())),
        "probe": {
            "protocol": "standardized multinomial logistic regression, 5-fold GroupKFold by semantic_id",
            "layers": probe_layers,
            "final_layer": probe_layers[-1],
            "within_stale_true_current_score": observed,
            "value_label_shuffle95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
            "above_shuffle95": bool(observed > np.quantile(null, 0.975)),
            "within_vs_correct": probe_delta,
        },
        "attention": {
            "definition": "stale/(stale+current), averaged over all heads at the final layer",
            "within_vs_correct": attention_delta,
            "all_heads": head_table,
        },
        "direct_logit_attribution": {
            "head_table": dla_heads,
            "mlp_correct_mean": data["mlp_dla"][labels == "correct_current"].mean(axis=0).tolist(),
            "mlp_failure_mean": data["mlp_dla"][labels == "within_stale"].mean(axis=0).tolist(),
            "logit_lens_correct_mean": data["logit_lens"][labels == "correct_current"].mean(axis=0).tolist(),
            "logit_lens_failure_mean": data["logit_lens"][labels == "within_stale"].mean(axis=0).tolist(),
        },
    }
    dump_json(args.summary, summary)
    dump_jsonl(args.probe_rows, [
        {**row, "final_true_current_score": float(score)}
        for row, score in zip(rows, final_probe["true_scores"])
    ])
    print(json.dumps({"summary": args.summary, "probe_gate": summary["probe"]["above_shuffle95"]}, indent=2))


def fit_probe_chunk(args) -> None:
    out_dir = Path(args.harvest_dir)
    rows = load_jsonl(out_dir / "harvest_index.jsonl")
    hidden = np.load(out_dir / "harvest.npz")["hidden"]
    layer_rows = []
    for layer in args.layers:
        probe = fit_probe_layer(hidden, rows, layer, args.seed + layer)
        labels = np.asarray([row["label"] for row in rows], dtype=object)
        layer_rows.append(
            {
                "layer": layer,
                "cv_accuracy": probe["cv_accuracy"],
                "valid_oos_score_n": int(np.isfinite(probe["true_scores"]).sum()),
                "valid_oos_score_fraction": float(np.isfinite(probe["true_scores"]).mean()),
                "within_stale_true_current_score": float(
                    np.nanmean(probe["true_scores"][labels == "within_stale"])
                ),
                "correct_true_current_score": float(
                    np.nanmean(probe["true_scores"][labels == "correct_current"])
                ),
            }
        )
        print(f"fit grouped probe checkpoint {layer}", flush=True)
    dump_json(args.out, {"layers": layer_rows, "n": len(rows)})


def merge_probe_chunks(args) -> None:
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    layers = {int(row["layer"]): row for row in summary["probe"]["layers"]}
    for chunk in args.chunks:
        payload = json.loads(Path(chunk).read_text(encoding="utf-8"))
        for row in payload["layers"]:
            layers[int(row["layer"])] = row
    summary["probe"]["layers"] = [layers[layer] for layer in sorted(layers)]
    summary["probe"]["layer_curve_source"] = "exact same grouped probe fit in layer-parallel CPU jobs"
    dump_json(args.summary, summary)
    print(json.dumps({"summary": args.summary, "layers": sorted(layers)}, indent=2))


class PatchCache:
    """Capture clean component states needed for final-position patching."""

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.head_inputs = {}
        self.mlp_outputs = {}
        self.residual_outputs = {}

    def __enter__(self):
        for layer, block in enumerate(self.model.gpt_neox.layers):
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
            self.handles.append(
                block.register_forward_hook(
                    lambda _module, _inputs, output, layer=layer: self.residual_outputs.__setitem__(
                        layer, output[0].detach().clone()
                    )
                )
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()


def batch_encode(tokenizer, device, rows: list[dict]):
    return tokenizer(
        [row["prompt"] for row in rows],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)


def query_positions(enc):
    return enc["attention_mask"].sum(dim=1) - 1


def gaps_from_logits(logits, rows: list[dict], query):
    import torch

    batch = torch.arange(len(rows), device=logits.device)
    gold = torch.as_tensor([int(row["gold_token_id"]) for row in rows], device=logits.device)
    stale = torch.as_tensor([competitor_token_id(row) for row in rows], device=logits.device)
    answer_logits = logits[batch, query]
    gaps = answer_logits[batch, gold] - answer_logits[batch, stale]
    predictions = answer_logits.argmax(dim=-1)
    return gaps.detach().float().cpu().numpy(), predictions.detach().cpu().tolist()


def run_logits(model, enc):
    import torch

    with torch.no_grad():
        return model(**enc, use_cache=False, return_dict=True).logits


def patch_head_inputs(model, enc, query, source: dict, heads: list[tuple[int, int]]):
    import torch

    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // n_heads)
    by_layer = defaultdict(list)
    for layer, head in heads:
        by_layer[int(layer)].append(int(head))
    handles = []
    batch = torch.arange(query.shape[0], device=query.device)
    for layer, selected in by_layer.items():
        def hook(_module, inputs, layer=layer, selected=tuple(selected)):
            value = inputs[0].clone()
            for head in selected:
                sl = slice(head * head_dim, (head + 1) * head_dim)
                value[batch, query, sl] = source[layer][batch, query, sl]
            return (value,)
        handles.append(model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(hook))
    try:
        return run_logits(model, enc)
    finally:
        for handle in handles:
            handle.remove()


def zero_head_inputs(model, enc, query, heads: list[tuple[int, int]]):
    import torch

    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // n_heads)
    by_layer = defaultdict(list)
    for layer, head in heads:
        by_layer[int(layer)].append(int(head))
    handles = []
    batch = torch.arange(query.shape[0], device=query.device)
    for layer, selected in by_layer.items():
        def hook(_module, inputs, selected=tuple(selected)):
            value = inputs[0].clone()
            for head in selected:
                sl = slice(head * head_dim, (head + 1) * head_dim)
                value[batch, query, sl] = 0
            return (value,)
        handles.append(model.gpt_neox.layers[layer].attention.dense.register_forward_pre_hook(hook))
    try:
        return run_logits(model, enc)
    finally:
        for handle in handles:
            handle.remove()


def patch_component(model, enc, query, source, layer: int, component: str):
    import torch

    batch = torch.arange(query.shape[0], device=query.device)
    if component == "mlp":
        def hook(_module, _inputs, output):
            value = output.clone()
            value[batch, query] = source[layer][batch, query]
            return value
        handle = model.gpt_neox.layers[layer].mlp.register_forward_hook(hook)
    elif component == "residual":
        def hook(_module, _inputs, output):
            hidden = output[0].clone()
            hidden[batch, query] = source[layer][batch, query]
            return (hidden,) + tuple(output[1:])
        handle = model.gpt_neox.layers[layer].register_forward_hook(hook)
    else:
        raise ValueError(component)
    try:
        return run_logits(model, enc)
    finally:
        handle.remove()


def recovery_fraction(patched, corrupt, clean) -> np.ndarray:
    denominator = np.asarray(clean) - np.asarray(corrupt)
    return (np.asarray(patched) - np.asarray(corrupt)) / np.where(
        np.abs(denominator) >= 1e-4, denominator, np.nan
    )


def path_patch(args) -> None:
    pairs = load_jsonl(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("no clean/corrupted pairs")
    model, tokenizer, device = load_model(args.model, args.dtype)
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    all_heads = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    records = []
    for start in range(0, len(pairs), args.batch_size):
        batch_pairs = pairs[start : start + args.batch_size]
        clean_rows = [pair["clean"] for pair in batch_pairs]
        corrupt_rows = [pair["corrupted"] for pair in batch_pairs]
        clean_enc = batch_encode(tokenizer, device, clean_rows)
        corrupt_enc = batch_encode(tokenizer, device, corrupt_rows)
        clean_query = query_positions(clean_enc)
        corrupt_query = query_positions(corrupt_enc)
        if clean_enc["input_ids"].shape != corrupt_enc["input_ids"].shape or not np.array_equal(
            clean_query.cpu().numpy(), corrupt_query.cpu().numpy()
        ):
            raise ValueError("paired prompts must have matching padded shapes and query positions")
        with PatchCache(model) as clean_cache:
            clean_logits = run_logits(model, clean_enc)
        with PatchCache(model) as corrupt_cache:
            corrupt_logits = run_logits(model, corrupt_enc)
        clean_gap, _ = gaps_from_logits(clean_logits, corrupt_rows, corrupt_query)
        corrupt_gap, corrupt_pred = gaps_from_logits(corrupt_logits, corrupt_rows, corrupt_query)
        head_gaps = np.zeros((len(batch_pairs), n_layers, n_heads), dtype=np.float32)
        for layer, head in all_heads:
            logits = patch_head_inputs(
                model, corrupt_enc, corrupt_query, clean_cache.head_inputs, [(layer, head)]
            )
            head_gaps[:, layer, head], _ = gaps_from_logits(logits, corrupt_rows, corrupt_query)
        mlp_gaps = np.zeros((len(batch_pairs), n_layers), dtype=np.float32)
        residual_gaps = np.zeros((len(batch_pairs), n_layers), dtype=np.float32)
        for layer in range(n_layers):
            logits = patch_component(
                model, corrupt_enc, corrupt_query, clean_cache.mlp_outputs, layer, "mlp"
            )
            mlp_gaps[:, layer], _ = gaps_from_logits(logits, corrupt_rows, corrupt_query)
            logits = patch_component(
                model, corrupt_enc, corrupt_query, clean_cache.residual_outputs, layer, "residual"
            )
            residual_gaps[:, layer], _ = gaps_from_logits(logits, corrupt_rows, corrupt_query)
        all_logits = patch_head_inputs(
            model, corrupt_enc, corrupt_query, clean_cache.head_inputs, all_heads
        )
        all_gap, _ = gaps_from_logits(all_logits, corrupt_rows, corrupt_query)
        identity_logits = patch_head_inputs(
            model, corrupt_enc, corrupt_query, corrupt_cache.head_inputs, all_heads
        )
        identity_gap, identity_pred = gaps_from_logits(identity_logits, corrupt_rows, corrupt_query)
        for local, pair in enumerate(batch_pairs):
            records.append(
                {
                    "id": pair["id"],
                    "semantic_id": pair["corrupted"]["semantic_id"],
                    "split": stable_split(pair["corrupted"]["semantic_id"]),
                    "clean_gap": float(clean_gap[local]),
                    "corrupt_gap": float(corrupt_gap[local]),
                    "corrupt_pred_token_id": int(corrupt_pred[local]),
                    "head_patched_gaps": head_gaps[local].tolist(),
                    "mlp_patched_gaps": mlp_gaps[local].tolist(),
                    "residual_patched_gaps": residual_gaps[local].tolist(),
                    "all_heads_patched_gap": float(all_gap[local]),
                    "identity_patched_gap": float(identity_gap[local]),
                    "identity_pred_token_id": int(identity_pred[local]),
                }
            )
        print(f"path patched {min(start + args.batch_size, len(pairs))}/{len(pairs)}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "path_patch_rows.jsonl"
    dump_jsonl(raw_path, records)
    clean = np.asarray([row["clean_gap"] for row in records])
    corrupt = np.asarray([row["corrupt_gap"] for row in records])
    head_gaps = np.asarray([row["head_patched_gaps"] for row in records])
    mlp_gaps = np.asarray([row["mlp_patched_gaps"] for row in records])
    residual_gaps = np.asarray([row["residual_patched_gaps"] for row in records])
    discovery = np.asarray([row["split"] == "discovery" for row in records])
    evaluation = ~discovery
    if discovery.sum() < args.min_split or evaluation.sum() < args.min_split:
        raise ValueError(f"underpowered split: discovery={discovery.sum()} evaluation={evaluation.sum()}")
    individual_recovery = recovery_fraction(head_gaps, corrupt[:, None, None], clean[:, None, None])
    ranking = sorted(
        all_heads,
        key=lambda lh: float(np.nanmean(individual_recovery[discovery, lh[0], lh[1]])),
        reverse=True,
    )
    dump_json(
        out_dir / "path_patch_first_pass.summary.json",
        {
            "model": args.model,
            "n_pairs": len(records),
            "split_counts": {"discovery": int(discovery.sum()), "evaluation": int(evaluation.sum())},
            "selection_rule": "rank individual clean-to-corrupt head patches on discovery split only",
            "ranking": [
                {
                    "rank": rank + 1,
                    "layer": layer,
                    "head": head,
                    "discovery_recovery": float(np.nanmean(individual_recovery[discovery, layer, head])),
                    "evaluation_recovery": float(np.nanmean(individual_recovery[evaluation, layer, head])),
                    "evaluation_ci": bootstrap_ci(
                        individual_recovery[evaluation, layer, head], args.n_boot, args.seed + rank
                    ),
                }
                for rank, (layer, head) in enumerate(ranking)
            ],
            "all_head_joint_evaluation": paired_bootstrap(
                recovery_fraction(
                    np.asarray([row["all_heads_patched_gap"] for row in records])[evaluation],
                    corrupt[evaluation],
                    clean[evaluation],
                ),
                args.n_boot,
                args.seed + 1000,
            ),
            "identity_gate": {
                "max_abs_gap_change": float(
                    np.max(np.abs(np.asarray([r["identity_patched_gap"] for r in records]) - corrupt))
                ),
                "prediction_changes": int(
                    sum(r["identity_pred_token_id"] != r["corrupt_pred_token_id"] for r in records)
                ),
            },
            "mlp_recovery_mean": np.nanmean(
                recovery_fraction(mlp_gaps, corrupt[:, None], clean[:, None]), axis=0
            ).tolist(),
            "residual_recovery_mean": np.nanmean(
                recovery_fraction(residual_gaps, corrupt[:, None], clean[:, None]), axis=0
            ).tolist(),
        },
    )
    cumulative_patch(
        model=model,
        tokenizer=tokenizer,
        device=device,
        pairs=[pair for pair, keep in zip(pairs, evaluation) if keep],
        ranking=ranking,
        out_dir=out_dir,
        n_boot=args.n_boot,
        seed=args.seed + 2000,
        batch_size=args.batch_size,
    )


def cumulative_patch(*, model, tokenizer, device, pairs, ranking, out_dir, n_boot, seed, batch_size) -> None:
    records = []
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        clean_rows = [pair["clean"] for pair in batch_pairs]
        corrupt_rows = [pair["corrupted"] for pair in batch_pairs]
        clean_enc = batch_encode(tokenizer, device, clean_rows)
        corrupt_enc = batch_encode(tokenizer, device, corrupt_rows)
        query = query_positions(corrupt_enc)
        with PatchCache(model) as clean_cache:
            clean_logits = run_logits(model, clean_enc)
        clean_gap, _ = gaps_from_logits(clean_logits, corrupt_rows, query)
        corrupt_gap, _ = gaps_from_logits(run_logits(model, corrupt_enc), corrupt_rows, query)
        cumulative = np.zeros((len(batch_pairs), len(ranking)), dtype=np.float32)
        for count in range(1, len(ranking) + 1):
            logits = patch_head_inputs(
                model, corrupt_enc, query, clean_cache.head_inputs, ranking[:count]
            )
            cumulative[:, count - 1], _ = gaps_from_logits(logits, corrupt_rows, query)
        all_heads = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
        all_logits = patch_head_inputs(model, corrupt_enc, query, clean_cache.head_inputs, all_heads)
        all_gap, _ = gaps_from_logits(all_logits, corrupt_rows, query)
        for local, pair in enumerate(batch_pairs):
            records.append(
                {
                    "id": pair["id"],
                    "clean_gap": float(clean_gap[local]),
                    "corrupt_gap": float(corrupt_gap[local]),
                    "cumulative_gaps": cumulative[local].tolist(),
                    "all_heads_gap": float(all_gap[local]),
                }
            )
        print(f"cumulative patched {min(start + batch_size, len(pairs))}/{len(pairs)}", flush=True)
    dump_jsonl(out_dir / "cumulative_patch_rows.jsonl", records)
    clean = np.asarray([row["clean_gap"] for row in records])
    corrupt = np.asarray([row["corrupt_gap"] for row in records])
    cumulative = np.asarray([row["cumulative_gaps"] for row in records])
    all_recovery = recovery_fraction(
        np.asarray([row["all_heads_gap"] for row in records]), corrupt, clean
    )
    target = 0.8 * float(np.nanmean(all_recovery))
    cumulative_recovery = recovery_fraction(cumulative, corrupt[:, None], clean[:, None])
    means = np.nanmean(cumulative_recovery, axis=0)
    crossing = next((index + 1 for index, value in enumerate(means) if value >= target), None)
    selected_count = crossing if crossing is not None else len(ranking)
    clean_candidate = bool(
        crossing is not None
        and selected_count <= 12
        and means[selected_count - 1] >= 0.5
    )
    dump_json(
        out_dir / "cumulative_patch.summary.json",
        {
            "evaluation_n": len(records),
            "criterion": "smallest discovery-ranked prefix reaching 80% of held-out all-head joint recovery",
            "all_head_joint": paired_bootstrap(all_recovery, n_boot, seed),
            "target_recovery": target,
            "selected_count": selected_count,
            "selected_heads": [
                {"layer": layer, "head": head} for layer, head in ranking[:selected_count]
            ],
            "selected_recovery": paired_bootstrap(
                cumulative_recovery[:, selected_count - 1], n_boot, seed + 1
            ),
            "clean_candidate_before_ablation": clean_candidate,
            "cumulative": [
                {
                    "count": index + 1,
                    "mean_recovery": float(means[index]),
                    "ci": bootstrap_ci(cumulative_recovery[:, index], n_boot, seed + index + 2),
                }
                for index in range(len(ranking))
            ],
        },
    )


def selected_heads_from_summary(path: str | Path, max_heads: int | None = None) -> list[tuple[int, int]]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    heads = [(int(row["layer"]), int(row["head"])) for row in summary["selected_heads"]]
    return heads if max_heads is None else heads[:max_heads]


def qkov(args) -> None:
    import torch
    from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

    rows = load_jsonl(args.pool)
    if args.limit:
        rows = rows[: args.limit]
    selected = selected_heads_from_summary(args.cumulative_summary, args.max_key_heads)
    if not selected:
        raise ValueError("no selected heads for QK/OV analysis")
    model, tokenizer, device = load_model(args.model, args.dtype)
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    raw = []
    for index, row in enumerate(rows):
        enc = encode_row(tokenizer, device, row)
        position_ids = torch.arange(enc["input_ids"].shape[1], device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(
                **enc,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
        query_position = enc["input_ids"].shape[1] - 1
        writes = {int(write["value_token"]): write for write in row["writes"]}
        target_positions = [
            int(write["value_token"]) for write in row["writes"] if write["var"] == row["target_var"]
        ]
        candidate_ids = [int(row["gold_token_id"])] + [int(x) for x in row["stale_token_ids"]]
        for layer, head in selected:
            block = model.gpt_neox.layers[layer]
            hidden = output.hidden_states[layer]
            normalized = block.input_layernorm(hidden)
            qkv = block.attention.query_key_value(normalized).view(
                1, normalized.shape[1], model.config.num_attention_heads, 3 * head_dim
            ).transpose(1, 2)
            query_states, key_states, value_states = qkv.chunk(3, dim=-1)
            cos, sin = model.gpt_neox.rotary_emb(hidden, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            q = query_states[0, head, query_position]
            scores = torch.matmul(key_states[0, head], q) / math.sqrt(head_dim)
            current_position = int(row["current_value_span"][0])
            stale_positions = [int(x) for x in row["stale_value_spans"]]
            identity_positions = [int(x) for x in row["target_identity_spans"]]
            current_score = float(scores[current_position].item())
            stale_scores = [float(scores[position].item()) for position in stale_positions]
            identity_scores = [float(scores[position].item()) for position in identity_positions]
            target_scores = np.asarray([float(scores[position].item()) for position in target_positions])
            target_relative_positions = np.asarray(target_positions, dtype=np.float64)
            if len(target_positions) > 1:
                target_relative_positions = (
                    target_relative_positions - target_relative_positions.mean()
                ) / max(target_relative_positions.std(), 1e-8)
                positional_slope = float(np.polyfit(target_relative_positions, target_scores, 1)[0])
            else:
                positional_slope = math.nan
            dense = block.attention.dense
            sl = slice(head * head_dim, (head + 1) * head_dim)
            ov_rows = []
            for position in target_positions:
                value = value_states[0, head, position]
                component = torch.matmul(value, dense.weight[:, sl].T)
                source_id = int(writes[position]["value_token_id"])
                source_logit = torch.dot(component.float(), model.embed_out.weight[source_id].float())
                alternatives = [token_id for token_id in candidate_ids if token_id != source_id]
                alternative_logits = [
                    torch.dot(component.float(), model.embed_out.weight[token_id].float())
                    for token_id in alternatives
                ]
                ov_rows.append(
                    {
                        "position": position,
                        "is_current": position == current_position,
                        "source_token_id": source_id,
                        "copy_margin": float(
                            source_logit.item()
                            - (torch.stack(alternative_logits).max().item() if alternative_logits else 0.0)
                        ),
                    }
                )
            attention = output.attentions[layer][0, head, query_position]
            raw.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "layer": layer,
                    "head": head,
                    "qk_current": current_score,
                    "qk_stale_max": max(stale_scores) if stale_scores else math.nan,
                    "qk_stale_mean": float(np.mean(stale_scores)) if stale_scores else math.nan,
                    "qk_latest_margin": current_score - max(stale_scores) if stale_scores else math.nan,
                    "qk_identity_mean": float(np.mean(identity_scores)),
                    "qk_position_slope": positional_slope,
                    "attention_current": float(attention[current_position].item()),
                    "attention_stale_sum": float(attention[stale_positions].sum().item()),
                    "ov_writes": ov_rows,
                }
            )
        if (index + 1) % 20 == 0 or index + 1 == len(rows):
            print(f"QK/OV {index + 1}/{len(rows)}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "qkov_rows.jsonl", raw)
    summary_heads = []
    for layer, head in selected:
        head_rows = [row for row in raw if row["layer"] == layer and row["head"] == head]
        labels = sorted({row["label"] for row in head_rows})
        by_label = {}
        for label in labels:
            subset = [row for row in head_rows if row["label"] == label]
            by_label[label] = {
                "n": len(subset),
                "qk_latest_margin_mean": float(np.nanmean([row["qk_latest_margin"] for row in subset])),
                "qk_position_slope_mean": float(np.nanmean([row["qk_position_slope"] for row in subset])),
                "current_attention_mean": float(np.mean([row["attention_current"] for row in subset])),
                "stale_attention_sum_mean": float(np.mean([row["attention_stale_sum"] for row in subset])),
                "current_ov_copy_margin_mean": float(
                    np.mean([
                        write["copy_margin"]
                        for row in subset for write in row["ov_writes"] if write["is_current"]
                    ])
                ),
                "stale_ov_copy_margin_mean": float(
                    np.mean([
                        write["copy_margin"]
                        for row in subset for write in row["ov_writes"] if not write["is_current"]
                    ])
                ),
            }
        summary_heads.append({"layer": layer, "head": head, "by_label": by_label})
    dump_json(
        out_dir / "qkov.summary.json",
        {
            "model": args.model,
            "n_rows": len(rows),
            "head_selection": f"first {args.max_key_heads} heads from held-out cumulative-patch set",
            "heads": summary_heads,
            "qk_definition": "answer-position rotated query dot write-position rotated key / sqrt(head_dim)",
            "ov_definition": "source-token direct-unembedding margin of the unweighted head value through W_O",
        },
    )


def induction(args) -> None:
    import torch

    model, tokenizer, device = load_model(args.model, args.dtype)
    token_ids = []
    for value in range(10, 100):
        encoded = tokenizer.encode(" " + str(value), add_special_tokens=False)
        if len(encoded) == 1:
            token_ids.append(int(encoded[0]))
    rng = random.Random(args.seed)
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    induction_scores = np.zeros((args.n_sequences, n_layers, n_heads), dtype=np.float32)
    prefix_scores = np.zeros_like(induction_scores)
    baselines = np.zeros_like(induction_scores)
    for sample in range(args.n_sequences):
        first = rng.sample(token_ids, args.sequence_length)
        ids = torch.as_tensor([first + first], device=device)
        with torch.no_grad():
            output = model(
                input_ids=ids,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        for layer, attention in enumerate(output.attentions):
            weights = attention[0].detach().float().cpu().numpy()
            per_head_induction = []
            per_head_prefix = []
            per_head_baseline = []
            for offset in range(args.sequence_length - 1):
                query = args.sequence_length + offset
                target = offset + 1
                prefix = offset
                per_head_induction.append(weights[:, query, target])
                per_head_prefix.append(weights[:, query, prefix])
                non_targets = [position for position in range(args.sequence_length) if position != target]
                per_head_baseline.append(weights[:, query, non_targets].mean(axis=1))
            induction_scores[sample, layer] = np.mean(per_head_induction, axis=0)
            prefix_scores[sample, layer] = np.mean(per_head_prefix, axis=0)
            baselines[sample, layer] = np.mean(per_head_baseline, axis=0)
    means = induction_scores.mean(axis=0)
    table = []
    flattened = means.reshape(-1)
    for layer in range(n_layers):
        for head in range(n_heads):
            value = means[layer, head]
            table.append(
                {
                    "layer": layer,
                    "head": head,
                    "induction_score": float(value),
                    "induction_minus_baseline": float(
                        (induction_scores[:, layer, head] - baselines[:, layer, head]).mean()
                    ),
                    "prefix_match_score": float(prefix_scores[:, layer, head].mean()),
                    "percentile_among_heads": float((flattened <= value).mean()),
                    "ci": bootstrap_ci(
                        induction_scores[:, layer, head], args.n_boot, args.seed + layer * 100 + head
                    ),
                }
            )
    selected = selected_heads_from_summary(args.cumulative_summary, args.max_key_heads)
    selected_rows = [row for row in table if (row["layer"], row["head"]) in selected]
    dump_json(
        args.summary,
        {
            "model": args.model,
            "n_sequences": args.n_sequences,
            "sequence_length_per_copy": args.sequence_length,
            "definition": "attention from token i in the second copy to token i+1 in the first copy",
            "all_heads": table,
            "circuit_heads": selected_rows,
            "anchor_gate": {
                "criterion": "at least one inspected circuit head is in the top induction-score decile",
                "pass": any(row["percentile_among_heads"] >= 0.9 for row in selected_rows),
            },
        },
    )


def classify_predictions(rows: list[dict], predictions: list[int]) -> list[str]:
    return [classify_token(row, int(prediction)) for row, prediction in zip(rows, predictions)]


def layer_matched_random_sets(
    selected: list[tuple[int, int]], n_layers: int, n_heads: int, n_sets: int, seed: int
) -> list[list[tuple[int, int]]]:
    rng = random.Random(seed)
    counts = Counter(layer for layer, _head in selected)
    selected_set = set(selected)
    out = []
    universe = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    for _ in range(n_sets):
        choice = []
        for layer, count in sorted(counts.items()):
            candidates = [(layer, head) for head in range(n_heads) if (layer, head) not in selected_set]
            if len(candidates) >= count:
                choice.extend(rng.sample(candidates, count))
            else:
                choice.extend(candidates)
        remaining = len(selected) - len(choice)
        if remaining:
            candidates = [head for head in universe if head not in selected_set and head not in choice]
            choice.extend(rng.sample(candidates, remaining))
        out.append(choice)
    return out


def evaluate_ablation_arm(model, tokenizer, device, rows, heads, batch_size):
    predictions = []
    gaps = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        enc = batch_encode(tokenizer, device, batch_rows)
        query = query_positions(enc)
        logits = zero_head_inputs(model, enc, query, heads) if heads else run_logits(model, enc)
        batch_gaps, batch_predictions = gaps_from_logits(logits, batch_rows, query)
        gaps.extend(batch_gaps.tolist())
        predictions.extend(batch_predictions)
    return {
        "gaps": np.asarray(gaps, dtype=np.float64),
        "labels": classify_predictions(rows, predictions),
        "predictions": predictions,
    }


def ablate(args) -> None:
    rows = load_jsonl(args.pool)
    correct_rows = [row for row in rows if row["label"] == "correct_current"][: args.max_per_class]
    failure_rows = [row for row in rows if row["label"] == "within_stale"][: args.max_per_class]
    if not correct_rows or not failure_rows:
        raise ValueError("ablation requires both correct and within-stale rows")
    cumulative = json.loads(Path(args.cumulative_summary).read_text(encoding="utf-8"))
    selected_all = [(int(row["layer"]), int(row["head"])) for row in cumulative["selected_heads"]]
    current_heads = selected_all[: args.max_ablation_heads]
    mechanism = json.loads(Path(args.mechanism_summary).read_text(encoding="utf-8"))
    attention = {
        (int(row["layer"]), int(row["head"])): row
        for row in mechanism["attention"]["all_heads"]
    }
    dla = {
        (int(row["layer"]), int(row["head"])): row
        for row in mechanism["direct_logit_attribution"]["head_table"]
    }
    all_heads = sorted(attention)
    stale_ranking = sorted(
        all_heads,
        key=lambda key: (
            attention[key]["failure_minus_correct_stale_ratio"]
            * max(0.0, -dla[key]["failure_mean"])
        ),
        reverse=True,
    )
    stale_heads = stale_ranking[: len(current_heads)]
    model, tokenizer, device = load_model(args.model, args.dtype)
    baseline_correct = evaluate_ablation_arm(
        model, tokenizer, device, correct_rows, [], args.batch_size
    )
    current_ablation = evaluate_ablation_arm(
        model, tokenizer, device, correct_rows, current_heads, args.batch_size
    )
    baseline_failure = evaluate_ablation_arm(
        model, tokenizer, device, failure_rows, [], args.batch_size
    )
    stale_ablation = evaluate_ablation_arm(
        model, tokenizer, device, failure_rows, stale_heads, args.batch_size
    )
    random_current = layer_matched_random_sets(
        current_heads,
        int(model.config.num_hidden_layers),
        int(model.config.num_attention_heads),
        args.n_random,
        args.seed,
    )
    random_stale = layer_matched_random_sets(
        stale_heads,
        int(model.config.num_hidden_layers),
        int(model.config.num_attention_heads),
        args.n_random,
        args.seed + 1,
    )
    random_current_rates = []
    random_stale_rates = []
    for index, heads in enumerate(random_current):
        result = evaluate_ablation_arm(model, tokenizer, device, correct_rows, heads, args.batch_size)
        random_current_rates.append(np.mean(np.asarray(result["labels"]) == "within_stale"))
        if (index + 1) % 8 == 0:
            print(f"current random ablations {index + 1}/{len(random_current)}", flush=True)
    for index, heads in enumerate(random_stale):
        result = evaluate_ablation_arm(model, tokenizer, device, failure_rows, heads, args.batch_size)
        random_stale_rates.append(np.mean(np.asarray(result["labels"]) == "correct_current"))
        if (index + 1) % 8 == 0:
            print(f"stale random ablations {index + 1}/{len(random_stale)}", flush=True)
    baseline_correct_labels = np.asarray(baseline_correct["labels"])
    current_labels = np.asarray(current_ablation["labels"])
    baseline_failure_labels = np.asarray(baseline_failure["labels"])
    stale_labels = np.asarray(stale_ablation["labels"])
    current_effect = (current_labels == "within_stale").astype(float) - (
        baseline_correct_labels == "within_stale"
    ).astype(float)
    stale_effect = (stale_labels == "correct_current").astype(float) - (
        baseline_failure_labels == "correct_current"
    ).astype(float)
    current_summary = paired_bootstrap(current_effect, args.n_boot, args.seed + 10)
    stale_summary = paired_bootstrap(stale_effect, args.n_boot, args.seed + 20)
    summary = {
        "model": args.model,
        "current_binding_ablation": {
            "heads": [{"layer": layer, "head": head} for layer, head in current_heads],
            "n": len(correct_rows),
            "baseline_stale_rate": float(np.mean(baseline_correct_labels == "within_stale")),
            "ablated_stale_rate": float(np.mean(current_labels == "within_stale")),
            "paired_stale_rate_increase": current_summary,
            "random_same_size_stale_rate": {
                "n_sets": len(random_current_rates),
                "mean": float(np.mean(random_current_rates)),
                "quantile95": np.quantile(random_current_rates, [0.025, 0.975]).tolist(),
            },
        },
        "stale_component_ablation": {
            "selection_rule": "rank failure stale-attention shift times negative failure DLA; fixed before ablation",
            "heads": [{"layer": layer, "head": head} for layer, head in stale_heads],
            "n": len(failure_rows),
            "baseline_correct_rate": float(np.mean(baseline_failure_labels == "correct_current")),
            "ablated_correct_rate": float(np.mean(stale_labels == "correct_current")),
            "paired_correct_rate_increase": stale_summary,
            "random_same_size_correct_rate": {
                "n_sets": len(random_stale_rates),
                "mean": float(np.mean(random_stale_rates)),
                "quantile95": np.quantile(random_stale_rates, [0.025, 0.975]).tolist(),
            },
        },
        "causal_concentration_gate": {
            "patch_small_set": bool(cumulative["clean_candidate_before_ablation"]),
            "current_ablation_directional": bool(current_summary["mean"] > 0),
            "current_ablation_ci_excludes_zero": bool(current_summary["ci"][0] > 0),
            "patch_plus_ablation_pass": bool(
                cumulative["clean_candidate_before_ablation"] and current_summary["ci"][0] > 0
            ),
            "note": "Final clean-circuit support additionally requires interpretable QK/OV; adjudicate in REPORT.md.",
        },
    }
    dump_json(args.summary, summary)


def self_test(_args) -> None:
    corrupt = np.asarray([-2.0, -1.0])
    clean = np.asarray([2.0, 3.0])
    patched = np.asarray([0.0, 1.0])
    assert np.allclose(recovery_fraction(patched, corrupt, clean), 0.5)
    rows = [
        {"label": "within_stale", "prompt_tokens": 100},
        {"label": "within_stale", "prompt_tokens": 110},
        {"label": "correct_current", "prompt_tokens": 100},
        {"label": "correct_current", "prompt_tokens": 110},
    ]
    summary = delta_summary(np.asarray([2.0, 2.0, 1.0, 1.0]), rows, "within_stale", "correct_current", 20, 20, 0)
    assert abs(summary["raw_delta"] - 1.0) < 1e-8
    random_sets = layer_matched_random_sets([(1, 1), (2, 2)], 4, 4, 10, 0)
    assert all(len(heads) == 2 for heads in random_sets)
    assert stable_split("example") in {"discovery", "evaluation"}
    print("pythia_circuit self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(func=self_test)

    harvest_parser = sub.add_parser("harvest")
    harvest_parser.add_argument("--pool", required=True)
    harvest_parser.add_argument("--out-dir", required=True)
    harvest_parser.add_argument("--model", default="EleutherAI/pythia-160m")
    harvest_parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    harvest_parser.add_argument("--limit", type=int)
    harvest_parser.set_defaults(func=harvest)

    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--harvest-dir", required=True)
    analyze_parser.add_argument("--summary", required=True)
    analyze_parser.add_argument("--probe-rows", required=True)
    analyze_parser.add_argument("--model", default="EleutherAI/pythia-160m")
    analyze_parser.add_argument("--seed", type=int, default=20260807)
    analyze_parser.add_argument("--n-boot", type=int, default=2000)
    analyze_parser.add_argument("--n-shuffle", type=int, default=2000)
    analyze_parser.add_argument(
        "--probe-layers",
        type=lambda value: [int(part) for part in value.split(",") if part],
        help="comma-separated residual checkpoints; must include final checkpoint, or omit for all",
    )
    analyze_parser.set_defaults(func=analyze)

    chunk_parser = sub.add_parser("fit-probe-chunk")
    chunk_parser.add_argument("--harvest-dir", required=True)
    chunk_parser.add_argument("--layers", type=lambda value: [int(x) for x in value.split(",")], required=True)
    chunk_parser.add_argument("--out", required=True)
    chunk_parser.add_argument("--seed", type=int, default=20260807)
    chunk_parser.set_defaults(func=fit_probe_chunk)

    merge_parser = sub.add_parser("merge-probe-chunks")
    merge_parser.add_argument("--summary", required=True)
    merge_parser.add_argument("--chunks", nargs="+", required=True)
    merge_parser.set_defaults(func=merge_probe_chunks)

    patch_parser = sub.add_parser("path-patch")
    patch_parser.add_argument("--pairs", required=True)
    patch_parser.add_argument("--out-dir", required=True)
    patch_parser.add_argument("--model", default="EleutherAI/pythia-160m")
    patch_parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    patch_parser.add_argument("--batch-size", type=int, default=16)
    patch_parser.add_argument("--limit", type=int)
    patch_parser.add_argument("--min-split", type=int, default=16)
    patch_parser.add_argument("--n-boot", type=int, default=2000)
    patch_parser.add_argument("--seed", type=int, default=20260807)
    patch_parser.set_defaults(func=path_patch)

    qkov_parser = sub.add_parser("qkov")
    qkov_parser.add_argument("--pool", required=True)
    qkov_parser.add_argument("--cumulative-summary", required=True)
    qkov_parser.add_argument("--out-dir", required=True)
    qkov_parser.add_argument("--model", default="EleutherAI/pythia-160m")
    qkov_parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    qkov_parser.add_argument("--limit", type=int, default=96)
    qkov_parser.add_argument("--max-key-heads", type=int, default=12)
    qkov_parser.set_defaults(func=qkov)

    induction_parser = sub.add_parser("induction")
    induction_parser.add_argument("--cumulative-summary", required=True)
    induction_parser.add_argument("--summary", required=True)
    induction_parser.add_argument("--model", default="EleutherAI/pythia-160m")
    induction_parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    induction_parser.add_argument("--n-sequences", type=int, default=64)
    induction_parser.add_argument("--sequence-length", type=int, default=24)
    induction_parser.add_argument("--max-key-heads", type=int, default=12)
    induction_parser.add_argument("--n-boot", type=int, default=2000)
    induction_parser.add_argument("--seed", type=int, default=20260807)
    induction_parser.set_defaults(func=induction)

    ablate_parser = sub.add_parser("ablate")
    ablate_parser.add_argument("--pool", required=True)
    ablate_parser.add_argument("--cumulative-summary", required=True)
    ablate_parser.add_argument("--mechanism-summary", required=True)
    ablate_parser.add_argument("--summary", required=True)
    ablate_parser.add_argument("--model", default="EleutherAI/pythia-160m")
    ablate_parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ablate_parser.add_argument("--batch-size", type=int, default=32)
    ablate_parser.add_argument("--max-per-class", type=int, default=96)
    ablate_parser.add_argument("--max-ablation-heads", type=int, default=12)
    ablate_parser.add_argument("--n-random", type=int, default=64)
    ablate_parser.add_argument("--n-boot", type=int, default=2000)
    ablate_parser.add_argument("--seed", type=int, default=20260807)
    ablate_parser.set_defaults(func=ablate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
