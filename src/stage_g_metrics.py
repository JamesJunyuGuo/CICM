"""Small shared metrics for Stage G circuit and benchmark runs."""

import collections
import math


def _first_token_id(tokenizer, text):
    encoded = tokenizer(str(text), add_special_tokens=False)
    ids = encoded["input_ids"]
    if not ids:
        return None
    return int(ids[0])


def first_token_logit_diff(logits_by_token_id, gold, stale, tokenizer):
    """Return logit(gold first token) - logit(stale first token), or None on collision."""
    gold_id = _first_token_id(tokenizer, gold)
    stale_id = _first_token_id(tokenizer, stale)
    if gold_id is None or stale_id is None or gold_id == stale_id:
        return None
    return float(logits_by_token_id[gold_id]) - float(logits_by_token_id[stale_id])


def assert_self_patch_identity(component_deltas, tolerance=1e-6):
    """Hard gate for self-patching: every measured drift must be numerically zero."""
    failures = {
        str(name): float(delta)
        for name, delta in component_deltas.items()
        if abs(float(delta)) > tolerance
    }
    if failures:
        raise AssertionError(
            f"self-patching identity gate failed at tolerance={tolerance}: {failures}"
        )
    return True


def _finite_mean(vals):
    finite = [float(v) for v in vals if math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else math.nan


def _finite_median(vals):
    finite = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not finite:
        return math.nan
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return (finite[mid - 1] + finite[mid]) / 2


def summarize_patch_effects(rows):
    by_component = collections.defaultdict(list)
    for row in rows:
        component = row["component"]
        layer = row.get("layer")
        if component.startswith("residual_l"):
            component = "residual_stream_late" if layer is not None and layer >= 20 else "residual_stream_early_mid"
        elif component.startswith("late_mlp_output_l"):
            component = "late_mlp_output"
        by_component[component].append(row["delta_recovery"])
    effects = []
    for component, vals in by_component.items():
        finite_n = sum(1 for v in vals if math.isfinite(float(v)))
        effects.append(
            {
                "component": component,
                "n": finite_n,
                "delta_recovery_mean": _finite_mean(vals),
                "delta_recovery_median": _finite_median(vals),
            }
        )
    return effects
