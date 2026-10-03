import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=True)


def dump_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def char_span_to_token_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    token_indices = []
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if end <= char_start or start >= char_end:
            continue
        token_indices.append(idx)
    if not token_indices:
        raise ValueError(f"no token offsets overlap char span [{char_start}, {char_end})")
    return min(token_indices), max(token_indices) + 1


def find_text_char_span(prompt: str, needle: str, *, start: int = 0) -> tuple[int, int]:
    if not needle:
        raise ValueError("empty needle")
    pos = prompt.find(needle, start)
    if pos >= 0:
        return pos, pos + len(needle)
    match = re.search(re.escape(needle), prompt, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    compact_prompt = re.sub(r"\s+", " ", prompt)
    compact_needle = re.sub(r"\s+", " ", needle).strip()
    pos = compact_prompt.lower().find(compact_needle.lower())
    if pos < 0:
        raise ValueError(f"text span not found: {needle[:120]!r}")
    raise ValueError(f"text span only matched after whitespace normalization: {needle[:120]!r}")


def token_span_for_text(tokenizer, prompt: str, text: str, *, start: int = 0) -> dict:
    char_start, char_end = find_text_char_span(prompt, text, start=start)
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    token_start, token_end = char_span_to_token_span(enc["offset_mapping"], char_start, char_end)
    decoded = tokenizer.decode(enc["input_ids"][token_start:token_end])
    return {
        "char_start": char_start,
        "char_end": char_end,
        "token_start": token_start,
        "token_end": token_end,
        "decoded": decoded,
    }


def length_residualize(values, lengths):
    values = np.asarray(values, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    if values.size == 0:
        return values
    if values.size < 3 or np.allclose(lengths, lengths[0]):
        return values - values.mean()
    x = np.column_stack([np.ones_like(lengths), np.log(np.maximum(lengths, 1.0))])
    coef, *_ = np.linalg.lstsq(x, values, rcond=None)
    return values - x @ coef


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 1009) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [math.nan, math.nan]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = arr[rng.integers(0, arr.size, size=arr.size)].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_values(values, lengths=None) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": math.nan, "ci": [math.nan, math.nan], "length_residualized_mean": math.nan}
    out = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "ci": bootstrap_ci(arr),
    }
    if lengths is not None:
        residuals = length_residualize(arr, lengths)
        out["length_residualized_mean"] = float(residuals.mean())
        out["length_residualized_ci"] = bootstrap_ci(residuals)
    return out


def summarize_dp_logit_gaps(rows: list[dict], gaps: np.ndarray) -> dict:
    if gaps.shape[0] != len(rows):
        raise ValueError(f"rows/gaps mismatch: {len(rows)} vs {gaps.shape[0]}")
    groups = {}
    for label in sorted({row.get("final_error_type", "other") for row in rows}):
        idx = [i for i, row in enumerate(rows) if row.get("final_error_type", "other") == label]
        layer_gaps = gaps[idx]
        lengths = [rows[i].get("prompt_tokens", 0) for i in idx]
        groups[label] = {
            "n": len(idx),
            "final_layer_mean_gap": float(layer_gaps[:, -1].mean()) if idx else math.nan,
            "max_mean_gap": float(layer_gaps.mean(axis=0).max()) if idx else math.nan,
            "first_positive_mean_layer": int(np.argmax(layer_gaps.mean(axis=0) > 0))
            if idx and np.any(layer_gaps.mean(axis=0) > 0)
            else None,
            "final_layer_gap_ci": bootstrap_ci(layer_gaps[:, -1]) if idx else [math.nan, math.nan],
            "final_layer_length_residualized": summarize_values(layer_gaps[:, -1], lengths),
        }
    return {"groups": groups}


def summarize_label_counts(rows: list[dict], key: str = "final_error_type") -> dict:
    return dict(Counter(row.get(key, "other") for row in rows))


def mean_attention_ratio(attn_a: np.ndarray, attn_b: np.ndarray) -> np.ndarray:
    denom = attn_a + attn_b
    return np.divide(attn_a, denom, out=np.full_like(attn_a, np.nan, dtype=np.float32), where=denom > 0)


def summarize_attention_groups(rows: list[dict], values: np.ndarray, label_key: str = "final_error_type") -> dict:
    out = {}
    for label in sorted({row.get(label_key, "other") for row in rows}):
        idx = [i for i, row in enumerate(rows) if row.get(label_key, "other") == label]
        if not idx:
            continue
        group_values = values[idx]
        lengths = np.array([rows[i].get("prompt_tokens", 0) for i in idx], dtype=np.float64)
        final_layer = group_values[:, -1]
        out[label] = {
            "n": len(idx),
            "mean_by_layer": np.nanmean(group_values, axis=0).tolist(),
            "final_layer": summarize_values(final_layer, lengths),
        }
    return out


def compare_groups(rows: list[dict], values: np.ndarray, a: str, b: str, label_key: str = "final_error_type") -> dict:
    idx_a = [i for i, row in enumerate(rows) if row.get(label_key, "other") == a]
    idx_b = [i for i, row in enumerate(rows) if row.get(label_key, "other") == b]
    if not idx_a or not idx_b:
        return {"a": a, "b": b, "n_a": len(idx_a), "n_b": len(idx_b), "delta_final_layer": math.nan}
    va = values[idx_a, -1]
    vb = values[idx_b, -1]
    return {
        "a": a,
        "b": b,
        "n_a": len(idx_a),
        "n_b": len(idx_b),
        "delta_final_layer": float(va.mean() - vb.mean()),
        "delta_final_layer_ci": bootstrap_ci(
            np.array([va[np.random.randint(0, len(va))] - vb[np.random.randint(0, len(vb))] for _ in range(2000)])
        ),
    }
