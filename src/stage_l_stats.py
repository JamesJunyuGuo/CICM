import math
from collections import Counter

import numpy as np


def labels_for(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([row.get(key, "other") for row in rows], dtype=object)


def lengths_for(rows: list[dict]) -> np.ndarray:
    return np.asarray([float(row.get("prompt_tokens", row.get("prompt_len", 0)) or 0) for row in rows], dtype=np.float64)


def pooled_length_residuals(values, lengths) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(lengths)
    residuals = np.full(values.shape, np.nan, dtype=np.float64)
    if mask.sum() < 3 or np.allclose(lengths[mask], lengths[mask][0]):
        residuals[mask] = values[mask] - values[mask].mean()
        return residuals
    x = np.column_stack([np.ones(mask.sum()), np.log(np.maximum(lengths[mask], 1.0))])
    coef, *_ = np.linalg.lstsq(x, values[mask], rcond=None)
    residuals[mask] = values[mask] - x @ coef
    return residuals


def _delta(labels: np.ndarray, values: np.ndarray, group_a: str, group_b: str) -> float:
    a = values[labels == group_a]
    b = values[labels == group_b]
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return math.nan
    return float(a.mean() - b.mean())


def bootstrap_delta(labels, values, group_a, group_b, n_boot=2000, seed=1009) -> list[float]:
    labels = np.asarray(labels, dtype=object)
    values = np.asarray(values, dtype=np.float64)
    a = values[labels == group_a]
    b = values[labels == group_b]
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        deltas[i] = a[rng.integers(0, len(a), size=len(a))].mean() - b[
            rng.integers(0, len(b), size=len(b))
        ].mean()
    return [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]


def shuffle_label_null(labels, values, group_a, group_b, n_shuffle=2000, seed=2001) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n_shuffle, dtype=np.float64)
    for i in range(n_shuffle):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        out[i] = _delta(shuffled, values, group_a, group_b)
    return out


def null_summary(null_values, observed: float) -> dict:
    arr = np.asarray(null_values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "ci": [math.nan, math.nan], "p_two_sided": math.nan, "exceeds_null_95": False}
    lo, hi = np.quantile(arr, [0.025, 0.975])
    p = (np.sum(np.abs(arr) >= abs(observed)) + 1) / (arr.size + 1)
    return {
        "n": int(arr.size),
        "ci": [float(lo), float(hi)],
        "p_two_sided": float(p),
        "exceeds_null_95": bool(observed < lo or observed > hi),
    }


def group_delta_with_pooled_length_control(
    rows: list[dict],
    values,
    *,
    group_a: str,
    group_b: str,
    label_key: str = "label",
    n_boot: int = 2000,
    n_shuffle: int = 2000,
    seed: int = 1009,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    labels = labels_for(rows, label_key)
    lengths = lengths_for(rows)
    finite = np.isfinite(values)
    labels = labels[finite]
    lengths = lengths[finite]
    values = values[finite]
    residuals = pooled_length_residuals(values, lengths)
    raw_delta = _delta(labels, values, group_a, group_b)
    length_delta = _delta(labels, residuals, group_a, group_b)
    counts = Counter(labels.tolist())
    raw_null = shuffle_label_null(labels, values, group_a, group_b, n_shuffle=n_shuffle, seed=seed + 1)
    length_null = shuffle_label_null(labels, residuals, group_a, group_b, n_shuffle=n_shuffle, seed=seed + 2)
    return {
        "group_a": group_a,
        "group_b": group_b,
        "label_key": label_key,
        "n_pool": int(len(values)),
        "n_a": int(counts.get(group_a, 0)),
        "n_b": int(counts.get(group_b, 0)),
        "raw_delta": raw_delta,
        "raw_delta_ci": bootstrap_delta(labels, values, group_a, group_b, n_boot=n_boot, seed=seed + 3),
        "raw_shuffle_null": null_summary(raw_null, raw_delta),
        "length_controlled_delta": length_delta,
        "length_controlled_delta_ci": bootstrap_delta(
            labels, residuals, group_a, group_b, n_boot=n_boot, seed=seed + 4
        ),
        "length_controlled_shuffle_null": null_summary(length_null, length_delta),
    }
