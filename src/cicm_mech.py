import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cicm_eval import classify_stage_l_response, dump_json, dump_jsonl, load_jsonl, render_prompt
from cicm_stats import group_delta_with_pooled_length_control, null_summary


def true_label_scores(proba: np.ndarray, classes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    class_to_idx = {str(label): i for i, label in enumerate(classes)}
    out = np.full(len(labels), np.nan, dtype=np.float64)
    for i, label in enumerate(labels):
        idx = class_to_idx.get(str(label))
        if idx is not None:
            out[i] = float(proba[i, idx])
    return out


def value_label_shuffle_null(
    proba: np.ndarray,
    classes: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    group: str,
    n_shuffle: int = 2000,
    seed: int = 2001,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=object)
    groups = np.asarray(groups, dtype=object)
    out = np.empty(n_shuffle, dtype=np.float64)
    for i in range(n_shuffle):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        scores = true_label_scores(proba, classes, shuffled)
        selected = scores[groups == group]
        selected = selected[np.isfinite(selected)]
        out[i] = selected.mean() if len(selected) else math.nan
    return out


def empirical_null_summary(null_values, observed: float) -> dict:
    arr = np.asarray(null_values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "ci": [math.nan, math.nan],
            "p_two_sided_empirical": math.nan,
            "above_null_95": False,
            "below_null_95": False,
            "exceeds_null_95": False,
        }
    lo, hi = np.quantile(arr, [0.025, 0.975])
    lower_tail = (np.sum(arr <= observed) + 1) / (arr.size + 1)
    upper_tail = (np.sum(arr >= observed) + 1) / (arr.size + 1)
    return {
        "n": int(arr.size),
        "ci": [float(lo), float(hi)],
        "p_two_sided_empirical": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        "above_null_95": bool(observed > hi),
        "below_null_95": bool(observed < lo),
        "exceeds_null_95": bool(observed < lo or observed > hi),
    }


def cell_key(row: dict) -> str:
    cell = row.get("factorial_cell") or row.get("competition", {}).get("factorial_cell") or {}
    same = cell.get("same_slot_stale_distance_bin", "unknown")
    other = cell.get("recent_other_slot_distance_bin", "unknown")
    return f"same_{same}__other_{other}"


def bootstrap_delta_values(a, b, *, n_boot: int, seed: int) -> list[float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = a[rng.integers(0, len(a), size=len(a))].mean() - b[
            rng.integers(0, len(b), size=len(b))
        ].mean()
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def shuffle_group_delta_null(a, b, *, n_shuffle: int, seed: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.full(n_shuffle, math.nan, dtype=np.float64)
    values = np.concatenate([a, b])
    labels = np.array([1] * len(a) + [0] * len(b), dtype=np.int8)
    rng = np.random.default_rng(seed)
    out = np.empty(n_shuffle, dtype=np.float64)
    for i in range(n_shuffle):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        out[i] = values[shuffled == 1].mean() - values[shuffled == 0].mean()
    return out


def delta_from_global_residuals(
    rows: list[dict],
    values,
    residuals,
    *,
    group_a_label: str,
    group_b_label: str = "correct_current",
    cell: str | None = None,
    n_boot: int = 2000,
    n_shuffle: int = 2000,
    seed: int = 1009,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    selected = np.ones(len(rows), dtype=bool)
    if cell is not None:
        selected = np.asarray([cell_key(row) == cell for row in rows], dtype=bool)
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    finite = selected & np.isfinite(values) & np.isfinite(residuals)
    a_mask = finite & (labels == group_a_label)
    b_mask = finite & (labels == group_b_label)
    raw_a, raw_b = values[a_mask], values[b_mask]
    resid_a, resid_b = residuals[a_mask], residuals[b_mask]
    raw_delta = float(raw_a.mean() - raw_b.mean()) if len(raw_a) and len(raw_b) else math.nan
    length_delta = float(resid_a.mean() - resid_b.mean()) if len(resid_a) and len(resid_b) else math.nan
    return {
        "group_a": group_a_label,
        "group_b": group_b_label,
        "cell": cell or "overall",
        "n_pool": int(finite.sum()),
        "n_a": int(a_mask.sum()),
        "n_b": int(b_mask.sum()),
        "raw_delta": raw_delta,
        "raw_delta_ci": bootstrap_delta_values(raw_a, raw_b, n_boot=n_boot, seed=seed + 1),
        "raw_shuffle_null": null_summary(
            shuffle_group_delta_null(raw_a, raw_b, n_shuffle=n_shuffle, seed=seed + 2),
            raw_delta,
        ),
        "length_controlled_delta": length_delta,
        "length_controlled_delta_ci": bootstrap_delta_values(
            resid_a, resid_b, n_boot=n_boot, seed=seed + 3
        ),
        "length_controlled_shuffle_null": null_summary(
            shuffle_group_delta_null(resid_a, resid_b, n_shuffle=n_shuffle, seed=seed + 4),
            length_delta,
        ),
    }


def decodability_summary_for_mask(
    rows: list[dict],
    proba: np.ndarray,
    classes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    mask,
    *,
    n_shuffle: int,
    seed: int,
) -> dict:
    mask = np.asarray(mask, dtype=bool)
    selected = scores[mask]
    selected = selected[np.isfinite(selected)]
    observed = float(selected.mean()) if len(selected) else math.nan
    group_labels = np.where(mask, "target", "background")
    null = value_label_shuffle_null(
        proba,
        classes,
        labels,
        group_labels,
        group="target",
        n_shuffle=n_shuffle,
        seed=seed,
    )
    out = empirical_null_summary(null, observed)
    out.update(
        {
            "n": int(len(selected)),
            "mean_true_current_score": observed,
            "above_value_label_shuffle_95": bool(out["above_null_95"]),
        }
    )
    return out


def build_l1_failure_pool_summary(
    rows: list[dict],
    probe: dict,
    *,
    n_boot: int,
    n_shuffle: int,
    seed: int,
) -> dict:
    from cicm_stats import pooled_length_residuals

    scores = np.asarray(probe["true_current_score"], dtype=np.float64)
    lengths = np.asarray([float(row.get("prompt_tokens", 0) or 0) for row in rows], dtype=np.float64)
    residuals = pooled_length_residuals(scores, lengths)
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    cells = sorted({cell_key(row) for row in rows})
    classes = np.asarray(probe["classes"], dtype=object)
    value_labels = np.asarray(probe["labels"], dtype=object)
    out = {
        "length_control": {
            "scope": "all_l1_rows",
            "n_pool": int(np.isfinite(scores).sum()),
            "formula": "probe_true_current_score ~ log(prompt_tokens)",
        },
        "overall_label_counts": dict(Counter(labels.tolist())),
        "failure_modes": {},
        "per_cell_note": (
            "Per-cell rows are the inferential unit. Overall rows are included only as an "
            "equal-cell-size diagnostic and must not be used to claim deployment prevalence."
        ),
    }
    for mode in ["within_stale", "cross_slot"]:
        mode_mask = labels == mode
        mode_summary = {
            "decodability": decodability_summary_for_mask(
                rows,
                probe["proba"],
                classes,
                value_labels,
                scores,
                mode_mask,
                n_shuffle=n_shuffle,
                seed=seed + (11 if mode == "within_stale" else 31),
            ),
            "vs_correct": delta_from_global_residuals(
                rows,
                scores,
                residuals,
                group_a_label=mode,
                n_boot=n_boot,
                n_shuffle=n_shuffle,
                seed=seed + (101 if mode == "within_stale" else 301),
            ),
            "by_cell": {},
        }
        for idx, cell in enumerate(cells):
            cell_mask = np.asarray([cell_key(row) == cell for row in rows], dtype=bool)
            mode_summary["by_cell"][cell] = {
                "label_counts": dict(Counter(labels[cell_mask].tolist())),
                "decodability": decodability_summary_for_mask(
                    rows,
                    probe["proba"],
                    classes,
                    value_labels,
                    scores,
                    mode_mask & cell_mask,
                    n_shuffle=n_shuffle,
                    seed=seed + 1000 + idx + (0 if mode == "within_stale" else 100),
                ),
                "vs_correct": delta_from_global_residuals(
                    rows,
                    scores,
                    residuals,
                    group_a_label=mode,
                    cell=cell,
                    n_boot=n_boot,
                    n_shuffle=n_shuffle,
                    seed=seed + 2000 + idx + (0 if mode == "within_stale" else 100),
                ),
            }
        out["failure_modes"][mode] = mode_summary
    return out


def l1_adjudication_status(summary: dict) -> dict:
    counts = summary.get("label_counts", {})
    n_within = int(counts.get("within_stale", 0) or 0)
    if n_within == 0:
        return {
            "status": "not_adjudicable_no_within_stale",
            "l1_gate_pass": False,
            "reading": (
                "L-0 produced no within-stale failures, so L-1 retention-vs-selection is not adjudicable. "
                "This is a behavioral-gate miss, not evidence for retention failure."
            ),
        }
    above_null = bool(
        summary.get("probe", {})
        .get("within_stale_true_current_score_shuffle_null", {})
        .get("above_null_95", False)
    )
    if above_null:
        return {
            "status": "selection",
            "l1_gate_pass": True,
            "reading": (
                "Current value decodable on within-stale trials (present but not selected) → **SELECTION "
                "failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not "
                "identify (headline mechanism result). Nuance: true-current probe score is lower on "
                "within-stale than correct trials after length control, and stale-token attention is "
                "strongly elevated, so Stage L shows selection failure with a small retention/retrieval "
                "degradation rather than a pure no-loss story."
            ),
        }
    return {
        "status": "retention",
        "l1_gate_pass": False,
        "reading": "Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.",
    }


def resolve_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def load_model_and_tokenizer(model_name: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype),
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def find_value_spans(tokenizer, prompt: str, values: list[str]) -> list[tuple[int, int]]:
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    spans = []
    for value in values:
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", re.IGNORECASE)
        for match in pattern.finditer(prompt):
            token_indices = [
                idx
                for idx, (start, end) in enumerate(offsets)
                if start != end and not (end <= match.start() or start >= match.end())
            ]
            if token_indices:
                spans.append((min(token_indices), max(token_indices) + 1))
    return spans


def reduce_final_attention(attentions, current_spans, stale_spans) -> dict:
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    out = {
        "attn_current": np.zeros((n_layers, n_heads), dtype=np.float32),
        "attn_stale": np.zeros((n_layers, n_heads), dtype=np.float32),
        "attn_total": np.zeros((n_layers, n_heads), dtype=np.float32),
    }
    for layer_idx, layer_attn in enumerate(attentions):
        final_attn = layer_attn[0, :, -1, :].detach().float().cpu()
        out["attn_total"][layer_idx] = final_attn.sum(dim=-1).numpy()
        for name, spans in [("attn_current", current_spans), ("attn_stale", stale_spans)]:
            for start, end in spans:
                out[name][layer_idx] += final_attn[:, start:end].sum(dim=-1).numpy()
    return out


def final_hidden(hidden_states) -> np.ndarray:
    return np.stack(
        [state[0, -1, :].detach().float().cpu().numpy().astype(np.float16) for state in hidden_states],
        axis=0,
    )


def harvest(args) -> None:
    import torch

    rows = load_jsonl(args.rows)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    device = next(model.parameters()).device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden, attn_current, attn_stale, attn_total, index_rows = [], [], [], [], []
    for i, row in enumerate(rows):
        prompt = render_prompt(tokenizer, row["messages"])
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        current_spans = find_value_spans(tokenizer, prompt, [row["current_value"]])
        stale_spans = find_value_spans(tokenizer, prompt, row.get("stale_values", []))
        with torch.no_grad():
            outputs = model(**enc, output_attentions=True, output_hidden_states=True, use_cache=False)
        reduced = reduce_final_attention(outputs.attentions, current_spans, stale_spans)
        hidden.append(final_hidden(outputs.hidden_states))
        attn_current.append(reduced["attn_current"])
        attn_stale.append(reduced["attn_stale"])
        attn_total.append(reduced["attn_total"])
        index_rows.append(
            {
                **row,
                "row_index": len(index_rows),
                "prompt_tokens": int(enc["input_ids"].shape[1]),
                "current_token_spans": current_spans,
                "stale_token_spans": stale_spans,
                "current_span_count": len(current_spans),
                "stale_span_count": len(stale_spans),
            }
        )
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"harvested {i + 1}/{len(rows)}", flush=True)
    np.savez_compressed(
        out_dir / "l1_harvest.npz",
        hidden=np.stack(hidden, axis=0),
        attn_current=np.stack(attn_current, axis=0),
        attn_stale=np.stack(attn_stale, axis=0),
        attn_total=np.stack(attn_total, axis=0),
    )
    dump_jsonl(out_dir / "l1_index.jsonl", index_rows)
    dump_json(
        out_dir / "l1_harvest.summary.json",
        {
            "stage": "l1_harvest",
            "model": args.model,
            "dtype": args.dtype,
            "n": len(index_rows),
            "out_dir": str(out_dir),
            "span_counts": {
                "current_missing": int(sum(1 for row in index_rows if not row["current_token_spans"])),
                "stale_missing": int(sum(1 for row in index_rows if not row["stale_token_spans"])),
            },
        },
    )
    print(f"wrote {out_dir / 'l1_harvest.npz'} and {out_dir / 'l1_index.jsonl'}")


def fit_current_value_probe(hidden: np.ndarray, rows: list[dict], *, layer: int, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    x = hidden[:, layer, :].astype(np.float32)
    labels = np.asarray([row["current_value"] for row in rows], dtype=object)
    groups = np.asarray([row["id"] for row in rows], dtype=object)
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    if len(enc.classes_) < 2:
        raise ValueError("need at least two current-value classes")
    if len(rows) >= 10:
        splits = list(GroupKFold(n_splits=min(5, len(rows))).split(x, y, groups))
        splitter = f"GroupKFold(n_splits={min(5, len(rows))})"
    else:
        splits = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=seed).split(x, y))
        splitter = "StratifiedKFold(n_splits=2)"
    proba = np.full((len(rows), len(enc.classes_)), np.nan, dtype=np.float64)
    pred = np.full(len(rows), -1, dtype=np.int64)
    for train_idx, test_idx in splits:
        train_classes = np.unique(y[train_idx])
        if len(train_classes) < 2:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
            random_state=seed,
        )
        clf.fit(x_train, y[train_idx])
        fold_proba = clf.predict_proba(x_test)
        for local_col, class_id in enumerate(clf.classes_):
            proba[test_idx, class_id] = fold_proba[:, local_col]
        pred[test_idx] = clf.predict(x_test)
    valid = np.isfinite(proba).any(axis=1)
    score = true_label_scores(proba, enc.classes_, labels)
    return {
        "layer": int(layer if layer >= 0 else hidden.shape[1] + layer),
        "splitter": splitter,
        "classes": enc.classes_.tolist(),
        "n": int(valid.sum()),
        "cv_accuracy": float(accuracy_score(y[valid], pred[valid])) if valid.any() else math.nan,
        "true_current_score": score,
        "proba": proba,
        "labels": labels,
    }


def analyze(args) -> None:
    mech_dir = Path(args.mech_dir)
    rows = load_jsonl(mech_dir / "l1_index.jsonl")
    data = np.load(mech_dir / "l1_harvest.npz")
    hidden = data["hidden"]
    layer = args.layer if args.layer is not None else hidden.shape[1] - 1
    probe = fit_current_value_probe(hidden, rows, layer=layer, seed=args.seed)
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    scores = probe["true_current_score"]
    failure_pool_summary = build_l1_failure_pool_summary(
        rows,
        probe,
        n_boot=args.n_boot,
        n_shuffle=args.n_shuffle,
        seed=args.seed + 500,
    )
    within_scores = scores[labels == "within_stale"]
    within_scores = within_scores[np.isfinite(within_scores)]
    observed_within = float(within_scores.mean()) if len(within_scores) else math.nan
    value_null = value_label_shuffle_null(
        probe["proba"],
        np.asarray(probe["classes"], dtype=object),
        probe["labels"],
        labels,
        group="within_stale",
        n_shuffle=args.n_shuffle,
        seed=args.seed + 10,
    )
    score_rows = [{**row, "probe_true_current_score": float(score) if np.isfinite(score) else math.nan} for row, score in zip(rows, scores)]
    probe_delta = group_delta_with_pooled_length_control(
        score_rows,
        scores,
        group_a="within_stale",
        group_b="correct_current",
        label_key="label",
        n_boot=args.n_boot,
        n_shuffle=args.n_shuffle,
        seed=args.seed + 100,
    )
    attn_ratio = np.divide(
        data["attn_stale"],
        data["attn_stale"] + data["attn_current"],
        out=np.full_like(data["attn_stale"], np.nan, dtype=np.float32),
        where=(data["attn_stale"] + data["attn_current"]) > 0,
    )
    attn_value = np.nanmean(attn_ratio, axis=2)[:, -1]
    attn_delta = group_delta_with_pooled_length_control(
        rows,
        attn_value,
        group_a="within_stale",
        group_b="correct_current",
        label_key="label",
        n_boot=args.n_boot,
        n_shuffle=args.n_shuffle,
        seed=args.seed + 200,
    )
    value_null_summary = empirical_null_summary(value_null, observed_within)
    summary = {
        "stage": "l1_analyze",
        "mech_dir": str(mech_dir),
        "n_boot": args.n_boot,
        "n_shuffle": args.n_shuffle,
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "factorial_cell_counts": dict(Counter(cell_key(row) for row in rows)),
        "stage_a_probe_reuse": {
            "status": "protocol_reused_weights_not_loadable",
            "note": (
                "No serialized Stage-A linear probe weights exist in results/overwrite_attention_extraction, and the "
                "Stage-A synthetic label space is incompatible with Stage-L natural values. "
                "L-1 therefore freezes the Stage-A probe protocol: decision-position residual "
                "stream, linear multinomial probe, standardization, GroupKFold, and shuffle null."
            ),
        },
        "probe": {
            "layer": probe["layer"],
            "splitter": probe["splitter"],
            "n": probe["n"],
            "n_classes": len(probe["classes"]),
            "cv_accuracy": probe["cv_accuracy"],
            "within_stale_true_current_score_mean": observed_within,
            "within_stale_true_current_score_shuffle_null": value_null_summary,
            "within_vs_correct_true_current_score": probe_delta,
            "failure_pool_summary": failure_pool_summary,
        },
        "attention_stale_ratio_within_vs_correct": attn_delta,
    }
    status = l1_adjudication_status(summary)
    summary["l1_status"] = status["status"]
    summary["l1_gate_pass"] = status["l1_gate_pass"]
    summary["l1_reading"] = status["reading"]
    dump_json(args.summary_out, summary)
    dump_jsonl(args.probe_rows_out, score_rows)
    append_l1_report(args.report_out, summary)
    print(
        json.dumps(
            {"summary": args.summary_out, "probe_rows": args.probe_rows_out, "gate": summary["l1_gate_pass"]},
            indent=2,
        )
    )


def _fmt_delta(row: dict) -> str:
    return (
        f"raw {row['raw_delta']:.4f} CI [{row['raw_delta_ci'][0]:.4f}, {row['raw_delta_ci'][1]:.4f}], "
        f"length-controlled {row['length_controlled_delta']:.4f} CI "
        f"[{row['length_controlled_delta_ci'][0]:.4f}, {row['length_controlled_delta_ci'][1]:.4f}], "
        f"shuffle95 {row['length_controlled_shuffle_null']['ci']}"
    )


def _fmt_float(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def _fmt_bool(value) -> str:
    return "yes" if bool(value) else "no"


def _cell_parts(cell: str) -> tuple[str, str]:
    match = re.match(r"same_(.+)__other_(.+)", cell)
    if not match:
        return cell, "unknown"
    return match.group(1), match.group(2)


def _failure_mode_table(mode_summary: dict) -> list[str]:
    lines = [
        "| same-slot stale | other-slot distance | labels in cell | fail n | true-current score | above shuffle95 | raw delta vs correct | length-controlled delta vs correct | length-null95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, row in sorted(mode_summary["by_cell"].items(), key=lambda kv: _cell_parts(kv[0])):
        same, other = _cell_parts(cell)
        dec = row["decodability"]
        delta = row["vs_correct"]
        length_null = delta["length_controlled_shuffle_null"]["ci"]
        lines.append(
            "| "
            f"{same} | {other} | `{row['label_counts']}` | {dec['n']} | "
            f"{_fmt_float(dec['mean_true_current_score'])} | {_fmt_bool(dec['above_value_label_shuffle_95'])} | "
            f"{_fmt_float(delta['raw_delta'])} | {_fmt_float(delta['length_controlled_delta'])} | "
            f"[{_fmt_float(length_null[0])}, {_fmt_float(length_null[1])}] |"
        )
    return lines


def append_l1_report(path: str | Path, summary: dict) -> None:
    p = Path(path)
    existing = p.read_text(encoding="utf-8") if p.exists() else "# Stage L Report\n"
    marker = "## L-1 Mechanism"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    probe = summary["probe"]
    null = probe["within_stale_true_current_score_shuffle_null"]
    verdict = summary.get("l1_reading") or l1_adjudication_status(summary)["reading"]
    pool = probe.get("failure_pool_summary", {})
    within = pool.get("failure_modes", {}).get("within_stale", {})
    cross = pool.get("failure_modes", {}).get("cross_slot", {})
    stage_a_reuse = summary.get("stage_a_probe_reuse", {})
    lines = [
        marker,
        "",
        "- Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result).",
        "- Current value decodable on cross-slot trials → **recency-over-identity selection failure**: the target-slot value is represented, but a recent other-slot value is selected.",
        "- Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.",
        "",
        "Analysis discipline applied here: per-cell rows are the result; aggregate rows are included only as equal-cell-size diagnostics and are not prevalence claims. L-0 crossover is treated as a conditioned tie/recency-gradient, not a clean global reversal.",
        "",
        f"- Stage-A probe reuse: `{stage_a_reuse.get('status', 'unknown')}`. {stage_a_reuse.get('note', '')}",
        f"- Label counts: `{summary['label_counts']}`",
        f"- Factorial cell counts: `{summary.get('factorial_cell_counts', {})}`",
        f"- Probe layer: {probe['layer']}; splitter: `{probe['splitter']}`; n_classes={probe['n_classes']}; CV accuracy={probe['cv_accuracy']:.4f}.",
        f"- Length control: `{pool.get('length_control', {}).get('formula', 'n/a')}` fit once on `{pool.get('length_control', {}).get('scope', 'n/a')}` with n={pool.get('length_control', {}).get('n_pool', 'n/a')}.",
        f"- Within-stale true-current score: {probe['within_stale_true_current_score_mean']:.4f}; value-label shuffle95 {null['ci']}; above null: `{null['above_null_95']}`.",
        f"- Probe true-current score within_stale - correct_current: {_fmt_delta(probe['within_vs_correct_true_current_score'])}.",
        f"- Attention stale/(stale+current) within_stale - correct_current: {_fmt_delta(summary['attention_stale_ratio_within_vs_correct'])}.",
        f"- L-1 status: `{summary.get('l1_status')}`.",
        f"- L-1 gate pass: `{summary['l1_gate_pass']}`.",
        f"- Pre-registered reading selected: {verdict}",
        "",
        "### Within-Stale Failure Pool",
        "",
        f"- Overall diagnostic: n={within.get('decodability', {}).get('n', 0)}, true-current score={_fmt_float(within.get('decodability', {}).get('mean_true_current_score'))}, above value-label shuffle95={_fmt_bool(within.get('decodability', {}).get('above_value_label_shuffle_95'))}.",
        "",
        *(_failure_mode_table(within) if within else []),
        "",
        "### Cross-Slot Failure Pool",
        "",
        f"- Overall diagnostic: n={cross.get('decodability', {}).get('n', 0)}, true-current score={_fmt_float(cross.get('decodability', {}).get('mean_true_current_score'))}, above value-label shuffle95={_fmt_bool(cross.get('decodability', {}).get('above_value_label_shuffle_95'))}.",
        "",
        *(_failure_mode_table(cross) if cross else []),
        "",
    ]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(existing + "\n".join(lines), encoding="utf-8")


def class_means(hidden: np.ndarray, rows: list[dict], *, layer: int) -> dict[str, np.ndarray]:
    grouped = defaultdict(list)
    for h, row in zip(hidden[:, layer, :].astype(np.float32), rows):
        grouped[row["current_value"]].append(h)
    return {label: np.stack(items, axis=0).mean(axis=0) for label, items in grouped.items()}


def choose_l2_competitor_value(row: dict) -> tuple[str | None, str]:
    failure_mode = row.get("label", "other")
    if failure_mode == "within_stale":
        values = row.get("stale_hits") or row.get("stale_values") or []
        return (values[0] if values else None), "within_stale"
    if failure_mode == "cross_slot":
        values = row.get("cross_slot_hits") or []
        if not values:
            recent = row.get("competition", {}).get("recent_other_slot_nearest") or {}
            value = recent.get("value")
            if value:
                values = [value]
        return (values[0] if values else None), "cross_slot"
    return None, str(failure_mode)


def fit_probe_direction_model(hidden: np.ndarray, rows: list[dict], *, layer: int, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    x = hidden[:, layer, :].astype(np.float32)
    labels = np.asarray([row["current_value"] for row in rows], dtype=object)
    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    clf = LogisticRegression(
        C=1.0,
        penalty="l2",
        max_iter=400,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
        random_state=seed,
    )
    clf.fit(x_scaled, y)
    directions = clf.coef_.astype(np.float32) / np.maximum(scaler.scale_.astype(np.float32), 1e-6)
    return {
        "classes": enc.classes_.tolist(),
        "class_to_index": {str(label): int(i) for i, label in enumerate(enc.classes_)},
        "directions": directions.astype(np.float32),
        "layer": int(layer if layer >= 0 else hidden.shape[1] + layer),
        "source": "full_data_linear_probe_coef_raw_space",
    }


def l2_probe_direction(direction_model: dict, current_value: str, competitor_value: str | None) -> np.ndarray | None:
    if competitor_value is None:
        return None
    class_to_index = direction_model["class_to_index"]
    current_idx = class_to_index.get(str(current_value))
    competitor_idx = class_to_index.get(str(competitor_value))
    if current_idx is None or competitor_idx is None:
        return None
    directions = direction_model["directions"]
    return (directions[current_idx] - directions[competitor_idx]).astype(np.float32)


def paired_bootstrap_delta(a, b, n_boot=2000, seed=42) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    diff = a - b
    if len(diff) == 0:
        return {"n": 0, "mean": math.nan, "ci": [math.nan, math.nan]}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = diff[rng.integers(0, len(diff), size=len(diff))].mean()
    return {"n": int(len(diff)), "mean": float(diff.mean()), "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}


def generate_without_hook(model, tokenizer, row: dict, max_new_tokens: int) -> str:
    import torch

    prompt = render_prompt(tokenizer, row["messages"])
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = generated[:, enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def intervention_generate(model, tokenizer, row: dict, direction: np.ndarray, alpha: float, layer: int, max_new_tokens: int) -> str:
    import torch

    prompt = render_prompt(tokenizer, row["messages"])
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    direction_t = torch.tensor(direction, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    prompt_len = int(enc["input_ids"].shape[1])

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] == prompt_len and alpha != 0.0:
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + alpha * direction_t
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    gen_ids = generated[:, enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def effective_steer_direction(
    direction: np.ndarray, *, alpha: float, residual_norm: float, alpha_mode: str
) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float32)
    if alpha_mode == "absolute":
        return (float(alpha) * direction).astype(np.float32)
    if alpha_mode == "residual_ratio":
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9 or not np.isfinite(norm):
            return np.zeros_like(direction, dtype=np.float32)
        return (direction / norm * (float(alpha) * float(residual_norm))).astype(np.float32)
    raise ValueError(f"unsupported alpha_mode: {alpha_mode}")


def build_l2_record(row: dict, *, arm: str, response: str, failure_mode: str, competitor_value: str | None, **extra) -> dict:
    label = classify_stage_l_response(response, row)
    return {
        **row,
        "baseline_label": row.get("label"),
        "failure_mode": failure_mode,
        "competitor_value": competitor_value,
        "arm": arm,
        "intervention_response": response,
        **label,
        **extra,
    }


def parse_alpha_list(alpha: float, alpha_list: str | None) -> list[float]:
    if not alpha_list:
        return [float(alpha)]
    values = []
    for part in alpha_list.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("--alpha-list did not contain any numeric alpha values")
    return values


def alpha_key(value: float) -> str:
    return f"{float(value):g}"


def intervene(args) -> None:
    summary = json.loads(Path(args.l1_summary).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not summary.get("l1_gate_pass") and not args.force:
        skipped = {
            "stage": "l2_intervention",
            "status": "skipped_l1_gate_failed",
            "l1_summary": args.l1_summary,
            "pre_registered_reading": summary.get(
                "l1_reading",
                "Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.",
            ),
        }
        dump_json(out_dir / "l2_summary.json", skipped)
        append_l2_report(args.report_out, skipped)
        print(json.dumps(skipped, indent=2))
        return

    import torch

    mech_dir = Path(args.mech_dir)
    rows = load_jsonl(mech_dir / "l1_index.jsonl")
    data = np.load(mech_dir / "l1_harvest.npz")
    hidden = data["hidden"]
    hidden_layer = hidden.shape[1] - 1
    hook_layer = args.layer if args.layer is not None else hidden.shape[1] - 2
    hook_residual_index = min(int(hook_layer) + 1, hidden.shape[1] - 1)
    direction_model = fit_probe_direction_model(hidden, rows, layer=hidden_layer, seed=args.seed)
    target_rows = [row for row in rows if row["label"] in {"within_stale", "cross_slot"}]
    if args.limit:
        target_rows = target_rows[: args.limit]
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    rng = np.random.default_rng(args.seed)
    arm_rows = []
    alpha_values = parse_alpha_list(args.alpha, args.alpha_list)
    for i, row in enumerate(target_rows):
        competitor_value, failure_mode = choose_l2_competitor_value(row)
        target_dir = l2_probe_direction(direction_model, row["current_value"], competitor_value)
        if target_dir is None:
            continue
        norm = float(np.linalg.norm(target_dir))
        row_index = int(row.get("row_index", i))
        residual_norm = float(np.linalg.norm(hidden[row_index, hook_residual_index].astype(np.float32)))
        default_response = generate_without_hook(model, tokenizer, row, args.max_new_tokens)
        identity_response = intervention_generate(
            model, tokenizer, row, np.zeros_like(target_dir), 0.0, hook_layer, args.max_new_tokens
        )
        for alpha_value in alpha_values:
            random_dir = rng.normal(size=target_dir.shape).astype(np.float32)
            random_dir = random_dir / max(float(np.linalg.norm(random_dir)), 1e-9) * norm
            arm_rows.append(
                build_l2_record(
                    row,
                    arm="default_nohook",
                    response=default_response,
                    failure_mode=failure_mode,
                    competitor_value=competitor_value,
                    direction_norm=0.0,
                    random_direction_norm=0.0,
                    alpha=float(alpha_value),
                    alpha_mode=args.alpha_mode,
                    residual_norm=residual_norm,
                    effective_steer_norm=0.0,
                    hook_layer=int(hook_layer),
                    hook_residual_index=int(hook_residual_index),
                    direction_source=direction_model["source"],
                )
            )
            arm_rows.append(
                build_l2_record(
                    row,
                    arm="identity",
                    response=identity_response,
                    failure_mode=failure_mode,
                    competitor_value=competitor_value,
                    direction_norm=0.0,
                    random_direction_norm=0.0,
                    alpha=float(alpha_value),
                    alpha_mode=args.alpha_mode,
                    residual_norm=residual_norm,
                    effective_steer_norm=0.0,
                    hook_layer=int(hook_layer),
                    hook_residual_index=int(hook_residual_index),
                    direction_source=direction_model["source"],
                )
            )
            for arm, direction in [
                ("random_matched_norm", random_dir),
                ("targeted", target_dir),
            ]:
                effective_direction = effective_steer_direction(
                    direction,
                    alpha=float(alpha_value),
                    residual_norm=residual_norm,
                    alpha_mode=args.alpha_mode,
                )
                response = intervention_generate(
                    model, tokenizer, row, effective_direction, 1.0, hook_layer, args.max_new_tokens
                )
                arm_rows.append(
                    build_l2_record(
                        row,
                        arm=arm,
                        response=response,
                        failure_mode=failure_mode,
                        competitor_value=competitor_value,
                        direction_norm=norm,
                        random_direction_norm=float(np.linalg.norm(random_dir))
                        if arm == "random_matched_norm"
                        else None,
                        alpha_mode=args.alpha_mode,
                        residual_norm=residual_norm,
                        effective_steer_norm=float(np.linalg.norm(effective_direction)),
                        alpha=float(alpha_value),
                        hook_layer=int(hook_layer),
                        hook_residual_index=int(hook_residual_index),
                        direction_source=direction_model["source"],
                    )
                )
        if (i + 1) % 10 == 0 or i + 1 == len(target_rows):
            print(f"intervened {i + 1}/{len(target_rows)}", flush=True)
    dump_jsonl(out_dir / "l2_rows.jsonl", arm_rows)
    l2_summary = summarize_l2(arm_rows, n_boot=args.n_boot, seed=args.seed)
    l2_summary.update(
        {
            "stage": "l2_intervention",
            "status": "complete",
            "alpha": args.alpha,
            "alpha_list": alpha_values,
            "alpha_mode": args.alpha_mode,
            "hidden_layer": int(hidden_layer),
            "hook_layer": int(hook_layer),
            "hook_residual_index": int(hook_residual_index),
            "direction_source": direction_model["source"],
            "target_failure_modes": ["within_stale", "cross_slot"],
        }
    )
    dump_json(out_dir / "l2_summary.json", l2_summary)
    append_l2_report(args.report_out, l2_summary)
    print(json.dumps({"summary": str(out_dir / "l2_summary.json"), "rows": str(out_dir / "l2_rows.jsonl")}, indent=2))


def _summarize_l2_pairs(paired: list[dict], *, n_boot: int, seed: int, target_error_label: str | None = None) -> dict:
    out = {"n_paired": len(paired), "arm_counts": {}}
    arms_to_report = ["identity", "random_matched_norm", "targeted"]
    for arm in arms_to_report:
        labels = [arms[arm]["label"] for arms in paired]
        counts = Counter(labels)
        target_hits = [
            arms[arm]["label"] == (target_error_label or arms[arm].get("failure_mode"))
            for arms in paired
        ]
        out["arm_counts"][arm] = {
            "counts": dict(counts),
            "target_error_rate": float(np.mean(target_hits)) if target_hits else math.nan,
            "within_stale_rate": counts.get("within_stale", 0) / len(labels) if labels else math.nan,
            "cross_slot_rate": counts.get("cross_slot", 0) / len(labels) if labels else math.nan,
            "correct_rate": counts.get("correct_current", 0) / len(labels) if labels else math.nan,
            "any_error_rate": 1.0 - counts.get("correct_current", 0) / len(labels) if labels else math.nan,
        }
    target_reduction = [
        (arms["identity"]["label"] == (target_error_label or arms["identity"].get("failure_mode")))
        - (arms["targeted"]["label"] == (target_error_label or arms["targeted"].get("failure_mode")))
        for arms in paired
    ]
    random_reduction = [
        (arms["identity"]["label"] == (target_error_label or arms["identity"].get("failure_mode")))
        - (arms["random_matched_norm"]["label"] == (target_error_label or arms["random_matched_norm"].get("failure_mode")))
        for arms in paired
    ]
    correct_gain_targeted = [
        (arms["targeted"]["label"] == "correct_current") - (arms["identity"]["label"] == "correct_current")
        for arms in paired
    ]
    correct_gain_random = [
        (arms["random_matched_norm"]["label"] == "correct_current")
        - (arms["identity"]["label"] == "correct_current")
        for arms in paired
    ]
    out["targeted_minus_random_target_error_reduction"] = paired_bootstrap_delta(
        target_reduction, random_reduction, n_boot=n_boot, seed=seed + 30
    )
    out["targeted_minus_random_correct_gain"] = paired_bootstrap_delta(
        correct_gain_targeted, correct_gain_random, n_boot=n_boot, seed=seed + 40
    )
    return out


def summarize_l2(rows: list[dict], *, n_boot: int, seed: int) -> dict:
    by_id = defaultdict(dict)
    for row in rows:
        by_id[(row["id"], alpha_key(row.get("alpha", 1.0)))][row["arm"]] = row
    required = {"default_nohook", "identity", "random_matched_norm", "targeted"}
    paired = [arms for arms in by_id.values() if required <= set(arms)]
    out = _summarize_l2_pairs(paired, n_boot=n_boot, seed=seed)
    identity_mismatch = [
        arms["default_nohook"].get("intervention_response", "")
        != arms["identity"].get("intervention_response", "")
        for arms in paired
    ]
    out["identity_gate_mismatches"] = int(sum(identity_mismatch))
    out["by_failure_mode"] = {}
    for failure_mode in ["within_stale", "cross_slot"]:
        mode_pairs = [arms for arms in paired if arms["identity"].get("failure_mode") == failure_mode]
        out["by_failure_mode"][failure_mode] = _summarize_l2_pairs(
            mode_pairs,
            n_boot=n_boot,
            seed=seed + (100 if failure_mode == "within_stale" else 200),
            target_error_label=failure_mode,
        )
    out["by_alpha"] = {}
    for key in sorted({alpha_key(arms["identity"].get("alpha", 1.0)) for arms in paired}, key=lambda x: float(x)):
        alpha_pairs = [arms for arms in paired if alpha_key(arms["identity"].get("alpha", 1.0)) == key]
        alpha_summary = _summarize_l2_pairs(alpha_pairs, n_boot=n_boot, seed=seed + int(float(key) * 1000))
        alpha_summary["by_failure_mode"] = {}
        for failure_mode in ["within_stale", "cross_slot"]:
            mode_pairs = [arms for arms in alpha_pairs if arms["identity"].get("failure_mode") == failure_mode]
            alpha_summary["by_failure_mode"][failure_mode] = _summarize_l2_pairs(
                mode_pairs,
                n_boot=n_boot,
                seed=seed + int(float(key) * 1000) + (100 if failure_mode == "within_stale" else 200),
                target_error_label=failure_mode,
            )
        out["by_alpha"][key] = alpha_summary
    if out["by_alpha"]:
        out["best_alpha_by_targeted_correct_gain"] = max(
            (float(key) for key in out["by_alpha"]),
            key=lambda a: out["by_alpha"][alpha_key(a)]["arm_counts"]["targeted"]["correct_rate"],
        )
        out["best_alpha_by_targeted_target_error_reduction"] = min(
            (float(key) for key in out["by_alpha"]),
            key=lambda a: out["by_alpha"][alpha_key(a)]["arm_counts"]["targeted"]["target_error_rate"],
        )
    ci = out["targeted_minus_random_target_error_reduction"]["ci"]
    out["l2_gate_pass"] = bool(ci[0] > 0)
    return out


def append_l2_report(path: str | Path, summary: dict) -> None:
    p = Path(path)
    existing = p.read_text(encoding="utf-8") if p.exists() else "# Stage L Report\n"
    marker = "## L-2 Causality"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    if summary.get("status") == "skipped_l1_gate_failed":
        body = [
            marker,
            "",
            "- Status: skipped because L-1 gate failed.",
            f"- L-1 reading: {summary.get('pre_registered_reading')}",
            "",
        ]
    else:
        headline = summary.get("targeted_minus_random_target_error_reduction", {})
        reading = (
            "Targeted reduces within-stale errors, random control does not (CI of the difference excludes 0) → **a causal lever for stale-binding on realistic-controlled data** — the result Stage G's external transfer could not obtain (the impact result)."
            if summary.get("l2_gate_pass")
            else "Targeted ≈ random, or no reduction → **clean negative**: the mechanism is readable and the representation is movable, but stale-binding is not a clean causal lever even here (consistent with Stage G external + the project's causal history). Report exactly so and STOP — do not tune the intervention post hoc to force an effect."
        )
        body = [
            marker,
            "",
            "- Targeted reduces within-stale errors, random control does not (CI of the difference excludes 0) → **a causal lever for stale-binding on realistic-controlled data** — the result Stage G's external transfer could not obtain (the impact result).",
            "- Targeted ≈ random, or no reduction → **clean negative**: the mechanism is readable and the representation is movable, but stale-binding is not a clean causal lever even here (consistent with Stage G external + the project's causal history). Report exactly so and STOP — do not tune the intervention post hoc to force an effect.",
            "",
            f"- n paired: {summary['n_paired']}",
            f"- direction source: `{summary.get('direction_source', 'unknown')}`; hidden_layer={summary.get('hidden_layer')}; hook_layer={summary.get('hook_layer')}; hook_residual_index={summary.get('hook_residual_index')}; alpha_mode=`{summary.get('alpha_mode', 'absolute')}`; alpha_list=`{summary.get('alpha_list', [summary.get('alpha')])}`.",
            f"- arm counts: `{summary['arm_counts']}`",
            f"- identity gate mismatches: {summary['identity_gate_mismatches']}",
            f"- targeted-minus-random target-error reduction: `{headline}`",
            f"- targeted-minus-random correct gain: `{summary.get('targeted_minus_random_correct_gain')}`",
            f"- by alpha: `{summary.get('by_alpha')}`",
            f"- best alpha by targeted correct gain: `{summary.get('best_alpha_by_targeted_correct_gain')}`.",
            f"- best alpha by targeted target-error reduction: `{summary.get('best_alpha_by_targeted_target_error_reduction')}`.",
            f"- by failure mode: `{summary.get('by_failure_mode')}`",
            f"- L-2 gate pass: `{summary['l2_gate_pass']}`.",
            f"- Pre-registered reading selected: {reading}",
            "",
        ]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(existing + "\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--rows", default="results/cicm/l0_rows.jsonl")
    h.add_argument("--out-dir", default="results/cicm/l1")
    h.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    h.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    h.add_argument("--limit", type=int)

    a = sub.add_parser("analyze")
    a.add_argument("--mech-dir", default="results/cicm/l1")
    a.add_argument("--summary-out", default="results/cicm/l1_summary.json")
    a.add_argument("--probe-rows-out", default="results/cicm/l1_probe_rows.jsonl")
    a.add_argument("--report-out", default="results/cicm/REPORT.md")
    a.add_argument("--layer", type=int)
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--n-shuffle", type=int, default=2000)
    a.add_argument("--seed", type=int, default=20260721)

    it = sub.add_parser("intervene")
    it.add_argument("--mech-dir", default="results/cicm/l1")
    it.add_argument("--l1-summary", default="results/cicm/l1_summary.json")
    it.add_argument("--out-dir", default="results/cicm/l2")
    it.add_argument("--report-out", default="results/cicm/REPORT.md")
    it.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    it.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    it.add_argument("--layer", type=int)
    it.add_argument("--alpha", type=float, default=1.0)
    it.add_argument("--alpha-list")
    it.add_argument("--alpha-mode", choices=["absolute", "residual_ratio"], default="absolute")
    it.add_argument("--max-new-tokens", type=int, default=16)
    it.add_argument("--limit", type=int)
    it.add_argument("--n-boot", type=int, default=2000)
    it.add_argument("--seed", type=int, default=20260721)
    it.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.cmd == "harvest":
        harvest(args)
    elif args.cmd == "analyze":
        analyze(args)
    else:
        intervene(args)


if __name__ == "__main__":
    main()
