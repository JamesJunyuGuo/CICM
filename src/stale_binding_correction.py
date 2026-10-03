"""Stage R: ITI-style, mechanism-guided stale-binding correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np

from pythia_circuit import paired_bootstrap, stable_split
from pythia_eval import classify_token, dump_json, load_model
from pythia_gen import dump_jsonl, load_jsonl


CORRECT = "correct_current"
STALE = "within_stale"
DEFAULT_ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)


def protocol_split(semantic_id: str, frozen_split: str) -> str:
    """Keep evaluation frozen; deterministically split old discovery rows."""
    if frozen_split == "evaluation":
        return "evaluation"
    if frozen_split != "discovery":
        raise ValueError(f"unknown frozen split: {frozen_split}")
    digest = hashlib.sha256(f"stage-r-calibration|{semantic_id}".encode()).digest()
    return "calibration" if int.from_bytes(digest[:4], "big") % 4 == 0 else "fit"


def apply_head_shifts(hidden, query, directions, scales, alpha: float, n_heads: int):
    """Add per-head directions at each batch row's answer-query position."""
    import torch

    if float(alpha) == 0.0 or not directions:
        return hidden
    if hidden.shape[-1] % n_heads:
        raise ValueError("o_proj input does not split evenly over query heads")
    value = hidden.clone()
    head_dim = hidden.shape[-1] // n_heads
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    for head, direction in directions.items():
        head = int(head)
        vector = torch.as_tensor(direction, dtype=hidden.dtype, device=hidden.device)
        if vector.numel() != head_dim:
            raise ValueError(f"head {head} direction has {vector.numel()} dims, expected {head_dim}")
        sl = slice(head * head_dim, (head + 1) * head_dim)
        value[batch, query, sl] += float(alpha) * float(scales[head]) * vector
    return value


def make_matched_random_directions(
    directions: dict[tuple[int, int], np.ndarray],
    scales: dict[tuple[int, int], float],
    *,
    n_random: int,
    seed: int,
) -> list[dict[tuple[int, int], np.ndarray]]:
    """Sample unit directions; scales remain the target per-head sigma values."""
    del scales  # The caller applies the unchanged target scales.
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_random):
        one = {}
        for key, target in directions.items():
            vector = rng.normal(size=np.asarray(target).shape).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-12:
                raise RuntimeError("sampled a zero random direction")
            one[key] = vector / norm
        out.append(one)
    return out


def choose_operating_alpha(rows: list[dict], minimum_preservation: float = 0.95) -> dict:
    eligible = []
    for row in rows:
        if float(row["correct_preservation"]) < minimum_preservation:
            continue
        item = dict(row)
        item["net_gain"] = float(row["correction"]) - (
            1.0 - float(row["correct_preservation"])
        )
        eligible.append(item)
    if not eligible:
        raise ValueError("no alpha satisfies the calibration preservation constraint")
    return max(eligible, key=lambda row: (row["net_gain"], -float(row["alpha"])))


def parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def load_heads(path: str | Path) -> list[tuple[int, int]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    heads = [(int(row["layer"]), int(row["head"])) for row in payload["heads"]]
    if len(heads) != 12 or len(set(heads)) != 12:
        raise ValueError(f"expected the frozen 12-head set, found {len(heads)}")
    return heads


class HeadStateCapture(AbstractContextManager):
    def __init__(self, model, query, heads):
        self.model = model
        self.query = query
        self.heads = list(heads)
        self.handles = []
        self.values = {}

    def __enter__(self):
        import torch

        by_layer = defaultdict(list)
        for layer, head in self.heads:
            by_layer[layer].append(head)
        n_heads = int(self.model.config.num_attention_heads)
        head_dim = int(self.model.config.hidden_size // n_heads)
        for layer, layer_heads in by_layer.items():
            def hook(_module, inputs, layer=layer, layer_heads=tuple(layer_heads)):
                hidden = inputs[0]
                batch = torch.arange(hidden.shape[0], device=hidden.device)
                for head in layer_heads:
                    sl = slice(head * head_dim, (head + 1) * head_dim)
                    self.values[(layer, head)] = hidden[batch, self.query, sl].detach().float().cpu()

            self.handles.append(
                self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook)
            )
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


class HeadShiftHook(AbstractContextManager):
    def __init__(self, model, query, directions, scales, alpha):
        self.model = model
        self.query = query
        self.directions = directions
        self.scales = scales
        self.alpha = float(alpha)
        self.handles = []

    def __enter__(self):
        by_layer = defaultdict(dict)
        by_layer_scale = defaultdict(dict)
        for (layer, head), direction in self.directions.items():
            by_layer[layer][head] = direction
            by_layer_scale[layer][head] = self.scales[(layer, head)]
        n_heads = int(self.model.config.num_attention_heads)
        for layer, layer_directions in by_layer.items():
            layer_scales = by_layer_scale[layer]

            def hook(
                _module,
                inputs,
                layer_directions=layer_directions,
                layer_scales=layer_scales,
            ):
                shifted = apply_head_shifts(
                    inputs[0],
                    self.query,
                    layer_directions,
                    layer_scales,
                    self.alpha,
                    n_heads,
                )
                if shifted is inputs[0]:
                    return None
                return (shifted,) + tuple(inputs[1:])

            self.handles.append(
                self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook)
            )
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


def _query_batch(model, tokenizer, device, rows):
    enc = tokenizer(
        [row["prompt"] for row in rows],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)
    query = enc["attention_mask"].sum(dim=1) - 1
    return enc, query


def _record_predictions(rows, logits, tokenizer):
    records = []
    pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    for local, (row, pred_id) in enumerate(zip(rows, pred_ids)):
        gold = int(row["gold_token_id"])
        stale_ids = [int(value) for value in row.get("stale_token_ids", [])]
        stale_logits = [float(logits[local, value].item()) for value in stale_ids]
        strongest = max(stale_logits) if stale_logits else math.nan
        records.append(
            {
                "id": row["id"],
                "semantic_id": row["semantic_id"],
                "stored_label": row.get("label"),
                "pred_token_id": int(pred_id),
                "prediction": tokenizer.decode([int(pred_id)]),
                "label": classify_token(row, int(pred_id)),
                "gold_logit": float(logits[local, gold].item()),
                "strongest_stale_logit": strongest,
                "current_minus_stale_logit": (
                    float(logits[local, gold].item()) - strongest if stale_ids else math.nan
                ),
            }
        )
    return records


def capture_baseline(model, tokenizer, device, rows, heads, batch_size):
    import torch

    activations = {head: [] for head in heads}
    records = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        enc, query = _query_batch(model, tokenizer, device, batch_rows)
        with HeadStateCapture(model, query, heads) as capture, torch.no_grad():
            output = model(**enc, use_cache=False, return_dict=True)
        batch = torch.arange(len(batch_rows), device=device)
        logits = output.logits[batch, query]
        records.extend(_record_predictions(batch_rows, logits, tokenizer))
        for head in heads:
            activations[head].append(capture.values[head].numpy())
        print(f"baseline capture {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return records, {head: np.concatenate(values) for head, values in activations.items()}


def _fit_one_direction(x: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    ).fit(scaler.transform(x), y)
    raw = model.coef_[0].astype(np.float32) / np.maximum(
        scaler.scale_.astype(np.float32), 1e-6
    )
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise RuntimeError("linear probe produced an invalid direction")
    direction = raw / norm
    sigma = float(np.std(x @ direction, ddof=0))
    if not np.isfinite(sigma) or sigma <= 1e-9:
        raise RuntimeError("probe direction has zero projection variance")
    return direction, sigma


def fit_directions(activations, rows, fit_indices, seed, *, shuffled=False):
    y = np.asarray([rows[index]["label"] == CORRECT for index in fit_indices], dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("direction fit needs both correct and stale rows")
    if shuffled:
        y = np.random.default_rng(seed).permutation(y)
    directions = {}
    scales = {}
    for offset, (head, values) in enumerate(activations.items()):
        direction, sigma = _fit_one_direction(values[fit_indices], y, seed + offset)
        directions[head] = direction
        scales[head] = sigma
    return directions, scales


def evaluate_arm(
    model,
    tokenizer,
    device,
    rows,
    directions,
    scales,
    alpha,
    batch_size,
    *,
    mediation_heads=None,
    return_logits=False,
):
    import torch

    records = []
    query_logits = []
    eps = 1e-8
    mediation_heads = list(mediation_heads or [])
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        enc, query = _query_batch(model, tokenizer, device, batch_rows)
        with HeadShiftHook(model, query, directions, scales, alpha), torch.no_grad():
            output = model(
                **enc,
                use_cache=False,
                output_attentions=bool(mediation_heads),
                return_dict=True,
            )
        batch = torch.arange(len(batch_rows), device=device)
        logits = output.logits[batch, query]
        if return_logits:
            query_logits.append(logits.detach().float().cpu().numpy())
        batch_records = _record_predictions(batch_rows, logits, tokenizer)
        if mediation_heads:
            for local, (row, record) in enumerate(zip(batch_rows, batch_records)):
                margins = []
                for layer, head in mediation_heads:
                    attention = output.attentions[layer][local, head, query[local]]
                    current = attention[row["current_value_span"]].sum()
                    stale = attention[row["stale_value_spans"]].sum()
                    margins.append(float(torch.log((current + eps) / (stale + eps)).item()))
                record["downstream_attention_margin"] = float(np.mean(margins))
        records.extend(batch_records)
    return records, np.concatenate(query_logits) if return_logits else None


def summarize_against_baseline(baseline, arm):
    if [row["id"] for row in baseline] != [row["id"] for row in arm]:
        raise ValueError("baseline and arm rows are not aligned")
    stale = np.asarray([row["label"] == STALE for row in baseline])
    correct = np.asarray([row["label"] == CORRECT for row in baseline])
    arm_correct = np.asarray([row["label"] == CORRECT for row in arm])
    base_correct = np.asarray([row["label"] == CORRECT for row in baseline])
    return {
        "n": len(baseline),
        "baseline_counts": dict(Counter(row["label"] for row in baseline)),
        "arm_counts": dict(Counter(row["label"] for row in arm)),
        "correction": float(arm_correct[stale].mean()) if stale.any() else math.nan,
        "correct_preservation": float(arm_correct[correct].mean()) if correct.any() else math.nan,
        "baseline_accuracy": float(base_correct.mean()),
        "arm_accuracy": float(arm_correct.mean()),
        "net_accuracy_gain": float((arm_correct.astype(float) - base_correct.astype(float)).mean()),
        "paired_gain": (arm_correct.astype(float) - base_correct.astype(float)).tolist(),
    }


def _shuffle_null(values, labels, n_shuffle, seed):
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    return np.asarray([roc_auc_score(rng.permutation(labels), values) for _ in range(n_shuffle)])


def probe_sanity(directions, scales, activations, rows, fit_indices, eval_indices, n_shuffle, seed):
    from sklearn.metrics import roc_auc_score

    del scales
    eval_labels = np.asarray([rows[index]["label"] == CORRECT for index in eval_indices], dtype=int)
    per_head = []
    aggregate = np.zeros(len(eval_indices), dtype=np.float64)
    for head, direction in directions.items():
        fit_projection = activations[head][fit_indices] @ direction
        mean = float(fit_projection.mean())
        std = max(float(fit_projection.std()), 1e-8)
        score = (activations[head][eval_indices] @ direction - mean) / std
        auc = float(roc_auc_score(eval_labels, score))
        aggregate += score
        per_head.append({"layer": head[0], "head": head[1], "heldout_auc": auc})
    aggregate /= len(directions)
    auc = float(roc_auc_score(eval_labels, aggregate))
    null = _shuffle_null(aggregate, eval_labels, n_shuffle, seed)
    return {
        "heldout_n": len(eval_indices),
        "aggregate_auc": auc,
        "shuffle_n": int(n_shuffle),
        "shuffle_interval95": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "above_shuffle95": bool(auc > np.quantile(null, 0.975)),
        "per_head": per_head,
    }


def grouped_probe_cv(activations, rows, discovery_indices, n_shuffle, seed):
    """Out-of-fold probe check with current-value identity held out by group."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    discovery_indices = np.asarray(discovery_indices, dtype=int)
    y = np.asarray(
        [rows[index]["label"] == CORRECT for index in discovery_indices], dtype=int
    )
    groups = np.asarray(
        [int(rows[index]["gold_token_id"]) for index in discovery_indices], dtype=int
    )
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("grouped probe CV needs at least two current-value groups")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    score = np.full(len(discovery_indices), np.nan, dtype=np.float64)
    for fold, (train_local, test_local) in enumerate(
        splitter.split(np.zeros(len(y)), y, groups)
    ):
        train = discovery_indices[train_local]
        test = discovery_indices[test_local]
        fold_score = np.zeros(len(test), dtype=np.float64)
        for offset, (head, values) in enumerate(activations.items()):
            direction, _ = _fit_one_direction(values[train], y[train_local], seed + 100 * fold + offset)
            train_projection = values[train] @ direction
            mean = float(train_projection.mean())
            std = max(float(train_projection.std()), 1e-8)
            fold_score += (values[test] @ direction - mean) / std
        score[test_local] = fold_score / len(activations)
    if not np.isfinite(score).all():
        raise RuntimeError("grouped probe CV left non-finite out-of-fold scores")
    auc = float(roc_auc_score(y, score))
    null = _shuffle_null(score, y, n_shuffle, seed + 5000)
    return {
        "n": int(len(y)),
        "n_splits": int(n_splits),
        "group": "current value token id",
        "aggregate_auc": auc,
        "shuffle_n": int(n_shuffle),
        "shuffle_interval95": [
            float(np.quantile(null, 0.025)),
            float(np.quantile(null, 0.975)),
        ],
        "above_shuffle95": bool(auc > np.quantile(null, 0.975)),
    }


def _stable_rows(rows, records):
    kept_rows = []
    kept_records = []
    mismatches = []
    for row, record in zip(rows, records):
        if record["label"] == row.get("label"):
            kept_rows.append(row)
            kept_records.append(record)
        else:
            mismatches.append({"id": row["id"], "stored": row.get("label"), "rerun": record["label"]})
    return kept_rows, kept_records, mismatches


def _subset(values, indices):
    return [values[index] for index in indices]


def _random_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "interval95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "min": float(values.min()),
        "max": float(values.max()),
    }


def adjudicate(output_gate, *, attention_mediation, upstream_confirmation):
    """Separate behavioral correction from confirmation of its mechanism."""
    if not all(output_gate.values()):
        return "correction_gate_failed"
    if attention_mediation and upstream_confirmation:
        return "limited_mechanism_guided_correction"
    return "head_state_output_correction_mediation_unverified"


def _mediation_summary(baseline, targeted, random_runs, n_boot, seed):
    base = np.asarray([row["downstream_attention_margin"] for row in baseline])
    target = np.asarray([row["downstream_attention_margin"] for row in targeted])
    if not random_runs:
        raise ValueError("mediation needs at least one random direction")
    random_value = np.asarray(
        [
            [row["downstream_attention_margin"] for row in random_rows]
            for random_rows in random_runs
        ]
    )
    if random_value.shape[1] != len(base):
        raise ValueError("mediation random runs are not aligned with baseline")
    target_delta = target - base
    random_delta = random_value - base
    random_mean_per_example = random_delta.mean(axis=0)
    target_vs_random = target_delta - random_mean_per_example
    random_direction_means = random_delta.mean(axis=1)
    random_interval = [
        float(np.quantile(random_direction_means, 0.025)),
        float(np.quantile(random_direction_means, 0.975)),
    ]
    return {
        "n_random_directions": int(len(random_runs)),
        "targeted_minus_baseline": paired_bootstrap(target_delta, n_boot, seed),
        "random_minus_baseline": paired_bootstrap(
            random_mean_per_example, n_boot, seed + 1
        ),
        "targeted_minus_random": paired_bootstrap(target_vs_random, n_boot, seed + 2),
        "random_direction_mean_interval95": random_interval,
        "targeted_mean_exceeds_random95": bool(
            target_delta.mean() > random_interval[1]
        ),
    }


def _limit_per_split_label(rows, per_class):
    if not per_class:
        return rows
    counts = Counter()
    selected = []
    for row in rows:
        frozen = stable_split(row["semantic_id"])
        split = protocol_split(row["semantic_id"], frozen)
        key = (split, row["label"])
        if counts[key] >= per_class:
            continue
        counts[key] += 1
        selected.append(row)
    return selected


def precheck(args):
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = load_jsonl(args.pool)
    index = load_jsonl(Path(args.harvest_dir) / "harvest_index.jsonl")
    if [row["id"] for row in rows] != [row["id"] for row in index]:
        raise ValueError("harvest and pool order differ")
    data = np.load(Path(args.harvest_dir) / "harvest.npz")
    heads = load_heads(args.head_summary)
    eps = 1e-8
    score = np.stack(
        [
            np.log(
                (data["attn_stale"][:, layer, head] + eps)
                / (data["attn_current"][:, layer, head] + eps)
            )
            for layer, head in heads
        ],
        axis=1,
    ).mean(axis=1)
    labels = np.asarray([row["label"] == STALE for row in rows], dtype=int)
    frozen = np.asarray([stable_split(row["semantic_id"]) for row in rows])
    discovery = frozen == "discovery"
    evaluation = frozen == "evaluation"
    threshold = float(np.quantile(score[discovery & (labels == 0)], 0.9))
    triggered = score >= threshold
    summary = {
        "stage": "R0-attention-risk-precheck",
        "score": "mean frozen-head log(stale_attention/current_attention)",
        "discovery": {
            "n": int(discovery.sum()),
            "auc": float(roc_auc_score(labels[discovery], score[discovery])),
            "average_precision": float(average_precision_score(labels[discovery], score[discovery])),
        },
        "evaluation": {
            "n": int(evaluation.sum()),
            "auc": float(roc_auc_score(labels[evaluation], score[evaluation])),
            "average_precision": float(average_precision_score(labels[evaluation], score[evaluation])),
            "stale_trigger_rate": float(triggered[evaluation & (labels == 1)].mean()),
            "correct_trigger_rate": float(triggered[evaluation & (labels == 0)].mean()),
        },
        "threshold_rule": "discovery correct-score 90th percentile",
        "threshold": threshold,
        "reading": "diagnostic precheck only; it does not establish correction",
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)


def run(args):
    rows = _limit_per_split_label(load_jsonl(args.pool), args.smoke_per_class)
    heads = load_heads(args.head_summary)
    model, tokenizer, device = load_model(args.model, args.dtype)
    baseline_records, activations = capture_baseline(
        model, tokenizer, device, rows, heads, args.batch_size
    )
    stable_rows, stable_baseline, mismatches = _stable_rows(rows, baseline_records)
    stability = len(stable_rows) / max(len(rows), 1)
    if stability < 0.9:
        raise RuntimeError(f"baseline stability below 0.9: {stability:.3f}")
    if mismatches:
        kept_ids = {row["id"] for row in stable_rows}
        keep = np.asarray([row["id"] in kept_ids for row in rows])
        activations = {head: values[keep] for head, values in activations.items()}
    rows = stable_rows
    baseline_records = stable_baseline

    splits = [protocol_split(row["semantic_id"], stable_split(row["semantic_id"])) for row in rows]
    split_counts = Counter((split, row["label"]) for split, row in zip(splits, rows))
    for split in ("fit", "calibration", "evaluation"):
        for label in (CORRECT, STALE):
            if split_counts[(split, label)] < (2 if args.smoke_per_class else 20):
                raise RuntimeError(f"underpowered split {split}/{label}: {split_counts[(split, label)]}")
    indices = {split: [i for i, value in enumerate(splits) if value == split] for split in set(splits)}
    directions, scales = fit_directions(activations, rows, indices["fit"], args.seed)
    shuffled_directions, _ = fit_directions(
        activations, rows, indices["fit"], args.seed + 10000, shuffled=True
    )
    probe = probe_sanity(
        directions,
        scales,
        activations,
        rows,
        indices["fit"],
        indices["evaluation"],
        args.n_shuffle,
        args.seed,
    )
    probe["grouped_discovery_cv"] = grouped_probe_cv(
        activations,
        rows,
        indices["fit"] + indices["calibration"],
        args.n_shuffle,
        args.seed,
    )

    calibration_rows = _subset(rows, indices["calibration"])
    calibration_base = _subset(baseline_records, indices["calibration"])
    calibration_curve = []
    for alpha in args.alphas:
        arm, _ = evaluate_arm(
            model, tokenizer, device, calibration_rows, directions, scales, alpha, args.batch_size
        )
        metric = summarize_against_baseline(calibration_base, arm)
        calibration_curve.append({"alpha": alpha, **{k: metric[k] for k in (
            "correction", "correct_preservation", "net_accuracy_gain"
        )}})
    operating = choose_operating_alpha(
        calibration_curve, minimum_preservation=args.minimum_preservation
    )
    alpha = float(operating["alpha"])

    evaluation_rows = _subset(rows, indices["evaluation"])
    evaluation_base = _subset(baseline_records, indices["evaluation"])
    identity_rows, identity_logits = evaluate_arm(
        model,
        tokenizer,
        device,
        evaluation_rows,
        directions,
        scales,
        0.0,
        args.batch_size,
        return_logits=True,
    )
    _, baseline_logits = evaluate_arm(
        model,
        tokenizer,
        device,
        evaluation_rows,
        {},
        {},
        0.0,
        args.batch_size,
        return_logits=True,
    )
    identity_max = float(np.max(np.abs(identity_logits - baseline_logits)))
    if identity_max != 0.0 or [row["label"] for row in identity_rows] != [row["label"] for row in evaluation_base]:
        raise RuntimeError(f"alpha=0 identity gate failed: max logit delta={identity_max}")

    targeted_rows, _ = evaluate_arm(
        model, tokenizer, device, evaluation_rows, directions, scales, alpha, args.batch_size
    )
    opposite_rows, _ = evaluate_arm(
        model, tokenizer, device, evaluation_rows, directions, scales, -alpha, args.batch_size
    )
    shuffled_rows, _ = evaluate_arm(
        model, tokenizer, device, evaluation_rows, shuffled_directions, scales, alpha, args.batch_size
    )
    heldout = {
        "targeted": summarize_against_baseline(evaluation_base, targeted_rows),
        "opposite": summarize_against_baseline(evaluation_base, opposite_rows),
        "label_shuffled": summarize_against_baseline(evaluation_base, shuffled_rows),
    }
    random_sets = make_matched_random_directions(
        directions, scales, n_random=args.n_random, seed=args.seed + 20000
    )
    random_records = []
    random_metrics = []
    for random_index, random_directions in enumerate(random_sets):
        arm, _ = evaluate_arm(
            model, tokenizer, device, evaluation_rows, random_directions, scales, alpha, args.batch_size
        )
        metric = summarize_against_baseline(evaluation_base, arm)
        random_metrics.append(metric)
        random_records.append({"random_index": random_index, **metric})
        print(f"random direction {random_index + 1}/{len(random_sets)}", flush=True)

    no_overwrite = [
        row for row in load_jsonl(args.no_overwrite_rows)
        if int(row.get("k", -1)) == 0 and row.get("variant") == "single"
    ]
    if args.smoke_per_class:
        no_overwrite = no_overwrite[: 2 * args.smoke_per_class]
    no_base, _ = evaluate_arm(model, tokenizer, device, no_overwrite, {}, {}, 0.0, args.batch_size)
    no_target, _ = evaluate_arm(
        model, tokenizer, device, no_overwrite, directions, scales, alpha, args.batch_size
    )
    no_control = summarize_against_baseline(no_base, no_target)

    earliest = min(layer for layer, _ in heads)
    mediation_heads = [(layer, head) for layer, head in heads if layer > earliest]
    stale_eval = [
        row for row, record in zip(evaluation_rows, evaluation_base) if record["label"] == STALE
    ]
    med_base, _ = evaluate_arm(
        model, tokenizer, device, stale_eval, {}, {}, 0.0, args.mediation_batch_size,
        mediation_heads=mediation_heads,
    )
    med_target, _ = evaluate_arm(
        model, tokenizer, device, stale_eval, directions, scales, alpha, args.mediation_batch_size,
        mediation_heads=mediation_heads,
    )
    n_mediation_random = min(args.n_mediation_random, len(random_sets))
    med_random_runs = []
    for random_directions in random_sets[:n_mediation_random]:
        med_random, _ = evaluate_arm(
            model,
            tokenizer,
            device,
            stale_eval,
            random_directions,
            scales,
            alpha,
            args.mediation_batch_size,
            mediation_heads=mediation_heads,
        )
        med_random_runs.append(med_random)
    mediation = _mediation_summary(
        med_base, med_target, med_random_runs, args.n_boot, args.seed + 30000
    )

    upstream_directions = {
        head: direction
        for head, direction in directions.items()
        if head[0] <= args.upstream_max_layer
    }
    upstream_scales = {head: scales[head] for head in upstream_directions}
    upstream_shuffled = {
        head: shuffled_directions[head] for head in upstream_directions
    }
    if len(upstream_directions) < 2:
        raise RuntimeError(
            f"upstream validity arm has only {len(upstream_directions)} heads"
        )
    upstream_curve = []
    for upstream_alpha in args.alphas:
        arm, _ = evaluate_arm(
            model,
            tokenizer,
            device,
            calibration_rows,
            upstream_directions,
            upstream_scales,
            upstream_alpha,
            args.batch_size,
        )
        metric = summarize_against_baseline(calibration_base, arm)
        upstream_curve.append(
            {
                "alpha": upstream_alpha,
                **{
                    key: metric[key]
                    for key in (
                        "correction",
                        "correct_preservation",
                        "net_accuracy_gain",
                    )
                },
            }
        )
    upstream_operating = choose_operating_alpha(
        upstream_curve, minimum_preservation=args.minimum_preservation
    )
    upstream_alpha = float(upstream_operating["alpha"])
    upstream_target_rows, _ = evaluate_arm(
        model,
        tokenizer,
        device,
        evaluation_rows,
        upstream_directions,
        upstream_scales,
        upstream_alpha,
        args.batch_size,
    )
    upstream_opposite_rows, _ = evaluate_arm(
        model,
        tokenizer,
        device,
        evaluation_rows,
        upstream_directions,
        upstream_scales,
        -upstream_alpha,
        args.batch_size,
    )
    upstream_shuffled_rows, _ = evaluate_arm(
        model,
        tokenizer,
        device,
        evaluation_rows,
        upstream_shuffled,
        upstream_scales,
        upstream_alpha,
        args.batch_size,
    )
    upstream_no_target, _ = evaluate_arm(
        model,
        tokenizer,
        device,
        no_overwrite,
        upstream_directions,
        upstream_scales,
        upstream_alpha,
        args.batch_size,
    )
    upstream_metrics = {
        "targeted": summarize_against_baseline(evaluation_base, upstream_target_rows),
        "opposite": summarize_against_baseline(evaluation_base, upstream_opposite_rows),
        "label_shuffled": summarize_against_baseline(
            evaluation_base, upstream_shuffled_rows
        ),
        "no_overwrite": summarize_against_baseline(no_base, upstream_no_target),
    }
    upstream_gain_values = np.asarray(upstream_metrics["targeted"].pop("paired_gain"))
    upstream_gain = paired_bootstrap(
        upstream_gain_values, args.n_boot, args.seed + 50000
    )
    for arm in ("opposite", "label_shuffled", "no_overwrite"):
        upstream_metrics[arm].pop("paired_gain")
    upstream_med_target, _ = evaluate_arm(
        model,
        tokenizer,
        device,
        stale_eval,
        upstream_directions,
        upstream_scales,
        upstream_alpha,
        args.mediation_batch_size,
        mediation_heads=mediation_heads,
    )
    upstream_med_random_runs = []
    for random_directions in random_sets[:n_mediation_random]:
        upstream_random = {
            head: random_directions[head] for head in upstream_directions
        }
        upstream_med_random, _ = evaluate_arm(
            model,
            tokenizer,
            device,
            stale_eval,
            upstream_random,
            upstream_scales,
            upstream_alpha,
            args.mediation_batch_size,
            mediation_heads=mediation_heads,
        )
        upstream_med_random_runs.append(upstream_med_random)
    upstream_mediation = _mediation_summary(
        med_base,
        upstream_med_target,
        upstream_med_random_runs,
        args.n_boot,
        args.seed + 60000,
    )
    upstream_gate = {
        "gain_ci_above_zero": upstream_gain["ci"][0] > 0.0,
        "correct_preservation": (
            upstream_metrics["targeted"]["correct_preservation"] >= 0.95
        ),
        "no_overwrite_drop_le_002": (
            upstream_metrics["no_overwrite"]["net_accuracy_gain"] >= -0.02
        ),
        "direction_specificity": (
            upstream_metrics["targeted"]["net_accuracy_gain"]
            > upstream_metrics["opposite"]["net_accuracy_gain"]
            and upstream_metrics["targeted"]["net_accuracy_gain"]
            > upstream_metrics["label_shuffled"]["net_accuracy_gain"]
        ),
        "attention_mediation_vs_random": (
            upstream_mediation["targeted_minus_random"]["ci"][0] > 0.0
            and upstream_mediation["targeted_mean_exceeds_random95"]
        ),
    }
    upstream_gate["all_pass"] = all(upstream_gate.values())
    upstream_confirmation = {
        "max_layer": args.upstream_max_layer,
        "heads": [
            {"layer": layer, "head": head} for layer, head in upstream_directions
        ],
        "calibration_curve": upstream_curve,
        "operating_point": upstream_operating,
        "heldout": upstream_metrics,
        "paired_gain": upstream_gain,
        "mediation": upstream_mediation,
        "gate": upstream_gate,
    }

    target_gain = np.asarray(heldout["targeted"].pop("paired_gain"))
    target_ci = paired_bootstrap(target_gain, args.n_boot, args.seed + 40000)
    for arm in ("opposite", "label_shuffled"):
        heldout[arm].pop("paired_gain")
    random_correction = _random_summary([row["correction"] for row in random_metrics])
    random_gain = _random_summary([row["net_accuracy_gain"] for row in random_metrics])
    random_upper = random_correction["interval95"][1]
    random_gain_upper = random_gain["interval95"][1]
    med_ci = mediation["targeted_minus_random"]["ci"]
    output_gate = {
        "identity_exact_zero": identity_max == 0.0,
        "targeted_gain_ci_above_zero": target_ci["ci"][0] > 0.0,
        "correction_exceeds_random95": heldout["targeted"]["correction"] > random_upper,
        "net_gain_exceeds_random95": (
            heldout["targeted"]["net_accuracy_gain"] > random_gain_upper
        ),
        "correct_preservation": heldout["targeted"]["correct_preservation"] >= 0.95,
        "no_overwrite_drop_le_002": no_control["net_accuracy_gain"] >= -0.02,
        "direction_specificity": (
            heldout["targeted"]["net_accuracy_gain"] > heldout["opposite"]["net_accuracy_gain"]
            and heldout["targeted"]["net_accuracy_gain"] > heldout["label_shuffled"]["net_accuracy_gain"]
        ),
    }
    gate = {
        **output_gate,
        "attention_mediation_vs_random": (
            med_ci[0] > 0.0 and mediation["targeted_mean_exceeds_random95"]
        ),
        "upstream_nonfinal_effect": upstream_gate["all_pass"],
    }
    gate["all_pass"] = all(gate.values())
    verdict = adjudicate(
        output_gate,
        attention_mediation=gate["attention_mediation_vs_random"],
        upstream_confirmation=gate["upstream_nonfinal_effect"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "baseline_rows.jsonl", baseline_records)
    dump_jsonl(out_dir / "targeted_rows.jsonl", targeted_rows)
    dump_jsonl(out_dir / "random_controls.jsonl", random_records)
    summary = {
        "stage": "R1-ITI-style-test-time-correction",
        "model": args.model,
        "dtype": args.dtype,
        "decoding": "one-token greedy over full vocabulary",
        "runtime_inputs": "global learned head directions only; no current/stale value, token id, or span",
        "frozen_heads": [{"layer": layer, "head": head} for layer, head in heads],
        "baseline_stability": {
            "requested": len(rows) + len(mismatches),
            "retained": len(rows),
            "rate": stability,
            "mismatches": mismatches,
        },
        "split_counts": {f"{key[0]}__{key[1]}": value for key, value in split_counts.items()},
        "probe_sanity": probe,
        "calibration_curve": calibration_curve,
        "operating_point": operating,
        "heldout": heldout,
        "targeted_paired_gain": target_ci,
        "random_correction": random_correction,
        "random_net_gain": random_gain,
        "no_overwrite": no_control,
        "identity_max_abs_logit_change": identity_max,
        "mediation_heads": [{"layer": layer, "head": head} for layer, head in mediation_heads],
        "mediation": mediation,
        "upstream_confirmation": upstream_confirmation,
        "gate": gate,
        "verdict": verdict,
        "scope": "controlled Qwen2.5-1.5B existence test; not a general or deployable repair",
    }
    dump_json(out_dir / "summary.json", summary)
    write_report(out_dir / "REPORT.md", summary)
    print(json.dumps({"summary": str(out_dir / 'summary.json'), "verdict": verdict, "gate": gate}, indent=2))


def write_report(path, summary):
    target = summary["heldout"]["targeted"]
    random = summary["random_correction"]
    random_gain = summary["random_net_gain"]
    mediation = summary["mediation"]
    med = summary["mediation"]["targeted_minus_random"]
    upstream = summary["upstream_confirmation"]
    lines = [
        "# Stage R: Mechanism-guided inference-time correction",
        "",
        f"**Verdict:** `{summary['verdict']}`.",
        "",
        "The intervention uses one global correct-vs-stale direction per frozen causal head. "
        "It receives no answer value, token id, or write span at inference time.",
        "",
        "## Held-out result",
        "",
        "| alpha | correction | preserve correct | net accuracy gain | random correction 95% | random net-gain 95% |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {summary['operating_point']['alpha']:.3g} | {target['correction']:.3f} | "
        f"{target['correct_preservation']:.3f} | {target['net_accuracy_gain']:.3f} | "
        f"[{random['interval95'][0]:.3f}, {random['interval95'][1]:.3f}] | "
        f"[{random_gain['interval95'][0]:.3f}, {random_gain['interval95'][1]:.3f}] |",
        "",
        f"Paired net-gain 95% CI: {summary['targeted_paired_gain']['ci']}. "
        f"No-overwrite net change: {summary['no_overwrite']['net_accuracy_gain']:.3f}.",
        "",
        "## Mechanism mediation",
        "",
        f"Against {mediation['n_random_directions']} matched-norm random directions, "
        f"targeted-minus-random downstream attention-margin change was {med['mean']:.3f} "
        f"(paired 95% CI {med['ci']}); targeted exceeded the empirical random "
        f"95th percentile: {mediation['targeted_mean_exceeds_random95']}.",
        "",
        "## Pre-final validity arm",
        "",
        f"Using {len(upstream['heads'])} frozen heads at layers <="
        f"{upstream['max_layer']}, held-out net gain was "
        f"{upstream['heldout']['targeted']['net_accuracy_gain']:.3f} "
        f"(95% CI {upstream['paired_gain']['ci']}); mediation-versus-random CI was "
        f"{upstream['mediation']['targeted_minus_random']['ci']}.",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {key}: **{'PASS' if value else 'FAIL'}**" for key, value in summary["gate"].items())
    lines.extend([
        "",
        "## Scope",
        "",
        summary["scope"],
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("precheck")
    check.add_argument("--pool", default="results/pythia_crossscale/partC_qwen15b/matched_pool.jsonl")
    check.add_argument("--harvest-dir", default="results/pythia_crossscale/partC_qwen15b/mechanism")
    check.add_argument("--head-summary", default="results/pythia_crossscale/partC_qwen15b/ablation/ablation_heldout.summary.json")
    check.add_argument("--summary", default="results/attention_rerouting/precheck.summary.json")
    check.set_defaults(func=precheck)

    experiment = sub.add_parser("run")
    experiment.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    experiment.add_argument("--pool", default="results/pythia_crossscale/partC_qwen15b/matched_pool.jsonl")
    experiment.add_argument("--head-summary", default="results/pythia_crossscale/partC_qwen15b/ablation/ablation_heldout.summary.json")
    experiment.add_argument("--no-overwrite-rows", default="results/pythia_crossscale/partD_qwen15b/rows.jsonl")
    experiment.add_argument("--out-dir", required=True)
    experiment.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    experiment.add_argument("--batch-size", type=int, default=64)
    experiment.add_argument("--mediation-batch-size", type=int, default=8)
    experiment.add_argument("--alphas", type=parse_floats, default=list(DEFAULT_ALPHAS))
    experiment.add_argument("--minimum-preservation", type=float, default=0.95)
    experiment.add_argument("--n-random", type=int, default=64)
    experiment.add_argument("--n-mediation-random", type=int, default=16)
    experiment.add_argument("--n-boot", type=int, default=2000)
    experiment.add_argument("--n-shuffle", type=int, default=2000)
    experiment.add_argument("--seed", type=int, default=20260910)
    experiment.add_argument("--smoke-per-class", type=int)
    experiment.add_argument("--upstream-max-layer", type=int, default=25)
    experiment.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
