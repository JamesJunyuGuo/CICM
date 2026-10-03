"""Stage B causal intervention: bias attention toward write spans."""

import argparse
import contextlib
import contextvars
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_stage_a import classify_prediction, resolve_torch_dtype
from run_eval import parse_int
from span_utils import build_chat_prompt, map_target_value_spans


_CURRENT_ATTENTION_BIAS = contextvars.ContextVar("stage_b_attention_bias", default=None)
_QWEN2_PATCHED = False
_LLAMA_PATCHED = False


@dataclass(frozen=True)
class AttentionBiasContext:
    alpha: float
    span_token_start: int
    span_token_end: int
    heads_by_layer: dict


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def parse_alphas(text):
    return [float(x) for x in text.split(",") if x.strip()]


def resolve_sanity_baseline_mode(mode, model_name):
    if mode != "auto":
        return mode
    if model_name == "Qwen/Qwen2.5-7B-Instruct":
        return "stage_a"
    return "same_model"


def alpha_tag(alpha):
    if float(alpha).is_integer():
        return str(int(alpha))
    return str(alpha).replace(".", "p")


def apply_attention_bias(attn_scores, layer_idx, ctx):
    """Return attention scores with ln(alpha) added to selected head/span cells.

    The input shape is [batch, heads, query_len, key_len]. Bias applies from all
    query positions to the configured key columns. For alpha=1 this returns the
    original tensor object, preserving the exact identity path.
    """
    if ctx is None or ctx.alpha == 1.0:
        return attn_scores
    heads = ctx.heads_by_layer.get(layer_idx)
    if not heads:
        return attn_scores
    key_len = attn_scores.shape[-1]
    start = int(ctx.span_token_start)
    end = int(ctx.span_token_end)
    if start < 0 or end <= start:
        raise ValueError(f"invalid span {start}:{end}")
    if end > key_len:
        raise ValueError(f"span {start}:{end} exceeds key length {key_len}")

    biased = attn_scores.clone()
    head_idx = torch.tensor(tuple(heads), dtype=torch.long, device=attn_scores.device)
    biased[:, head_idx, :, start:end] += math.log(float(ctx.alpha))
    return biased


def install_qwen2_attention_patch():
    """Patch installed Transformers Qwen2 eager attention with Stage-B bias hook."""
    global _QWEN2_PATCHED
    if _QWEN2_PATCHED:
        return

    import transformers.models.qwen2.modeling_qwen2 as qwen2

    def stage_b_eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = qwen2.repeat_kv(key, module.num_key_value_groups)
        value_states = qwen2.repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        ctx = _CURRENT_ATTENTION_BIAS.get()
        attn_weights = apply_attention_bias(
            attn_weights, layer_idx=module.layer_idx, ctx=ctx
        )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    qwen2.eager_attention_forward = stage_b_eager_attention_forward
    _QWEN2_PATCHED = True


def install_llama_attention_patch():
    """Patch installed Transformers Llama eager attention with Stage-B bias hook."""
    global _LLAMA_PATCHED
    if _LLAMA_PATCHED:
        return

    import transformers.models.llama.modeling_llama as llama

    def stage_b_eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = llama.repeat_kv(key, module.num_key_value_groups)
        value_states = llama.repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        ctx = _CURRENT_ATTENTION_BIAS.get()
        attn_weights = apply_attention_bias(
            attn_weights, layer_idx=module.layer_idx, ctx=ctx
        )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    llama.eager_attention_forward = stage_b_eager_attention_forward
    _LLAMA_PATCHED = True


def install_attention_patch(model_name):
    lower = model_name.lower()
    if "qwen" in lower:
        install_qwen2_attention_patch()
    elif "llama" in lower:
        install_llama_attention_patch()
    else:
        raise ValueError(f"no Stage-B attention patch registered for model={model_name}")


@contextlib.contextmanager
def attention_bias(ctx):
    token = _CURRENT_ATTENTION_BIAS.set(ctx)
    try:
        yield
    finally:
        _CURRENT_ATTENTION_BIAS.reset(token)


def build_scope_heads(scope, num_layers, num_heads, top_summary=None, top_k=8):
    if scope == "all":
        return {layer: tuple(range(num_heads)) for layer in range(num_layers)}
    if scope == "late":
        return {
            layer: tuple(range(num_heads))
            for layer in range(18, min(27, num_layers - 1) + 1)
        }
    if scope == "topk":
        if not top_summary:
            raise ValueError("topk scope requires Stage-A A1 summary")
        heads_by_layer = {}
        n_valid = 0
        for item in top_summary["top_heads_by_correct_p_last"]:
            layer = int(item["layer"])
            head = int(item["head"])
            if layer >= num_layers or head >= num_heads:
                continue
            heads_by_layer.setdefault(layer, []).append(head)
            n_valid += 1
            if n_valid >= top_k:
                break
        if n_valid == 0:
            raise ValueError(
                f"no topk heads fit model shape {num_layers}x{num_heads}"
            )
        return {layer: tuple(heads) for layer, heads in heads_by_layer.items()}
    raise ValueError(f"unknown scope: {scope}")


def _is_within_stale_wrong(row):
    return (
        row["condition"] == "interference"
        and row["stable_label"] == "stable_wrong"
        and row.get("answer_type") == "stale"
    )


def build_example_sets(
    extract_rows,
    stable_rows,
    correct_subsample_n=200,
    seed=1,
    wrong_filter="any",
):
    stable_by_id = {row["id"]: row for row in stable_rows}
    merged = []
    for row in extract_rows:
        stable = stable_by_id.get(row["id"])
        if stable is None:
            continue
        item = dict(row)
        item.update(
            {
                "stable_label": stable["stable_label"],
                "fp32_pred": stable.get("fp32_pred"),
                "sdpa_pred": stable.get("sdpa_pred"),
                "stable_primary": stable.get("stable_primary"),
            }
        )
        merged.append(item)

    w_rows_all = [
        row
        for row in merged
        if row["condition"] == "interference"
        and row["stable_label"] == "stable_wrong"
    ]
    if wrong_filter == "any":
        w_rows = w_rows_all
    elif wrong_filter == "within_stale":
        w_rows = [row for row in w_rows_all if _is_within_stale_wrong(row)]
    else:
        raise ValueError(f"unknown wrong_filter={wrong_filter}")
    c_candidates = [
        row
        for row in merged
        if row["condition"] == "interference"
        and row["stable_label"] == "stable_correct"
    ]
    s_rows = [
        row
        for row in merged
        if row["condition"] == "simple" and row["stable_label"] == "stable_correct"
    ]

    if len(c_candidates) > correct_subsample_n:
        rng = random.Random(seed)
        chosen = {row["id"] for row in rng.sample(c_candidates, correct_subsample_n)}
        c_rows = [row for row in c_candidates if row["id"] in chosen]
    else:
        c_rows = c_candidates
    return {"W": w_rows, "C": c_rows, "S": s_rows}


def cap_rows(rows, limit):
    if limit is None:
        return rows
    return rows[:limit]


def _decode_span(tokenizer, input_ids, write):
    span_ids = input_ids[write["token_start"] : write["token_end"]]
    return tokenizer.decode(span_ids)


def row_interference_load(row):
    load = row.get("interference_load")
    return 0 if load is None else int(load)


def select_span(row, span_info, target_kind):
    if target_kind == "current":
        return span_info["current"], str(row["gold"])
    if target_kind == "answered_stale":
        write_idx = row.get("answered_write_idx")
        if write_idx is None:
            return None, None
        write_idx = int(write_idx)
        stale_writes = span_info["stale"]
        if write_idx < 0 or write_idx >= len(stale_writes):
            return None, None
        write = stale_writes[write_idx]
        return write, str(write["value"])
    raise ValueError(f"unknown target_kind: {target_kind}")


def assert_stage_a_span_alignment(row, span_info):
    starts = [int(w["token_start"]) for w in span_info["writes"]]
    ends = [int(w["token_end"]) for w in span_info["writes"]]
    if starts != row["span_token_starts"] or ends != row["span_token_ends"]:
        raise ValueError(
            f"Stage-A span mismatch for id={row['id']}: "
            f"{starts}/{ends} != {row['span_token_starts']}/{row['span_token_ends']}"
        )


def prepare_example(tokenizer, data_by_id, index_row, device):
    data_row = data_by_id[index_row["id"]]
    templated = build_chat_prompt(data_row["prompt"], tokenizer)
    span_info = map_target_value_spans(data_row, tokenizer)
    assert_stage_a_span_alignment(index_row, span_info)

    enc = tokenizer(templated, return_tensors="pt").to(device)
    input_ids = enc["input_ids"][0].detach().cpu().tolist()
    decoded_spans = [
        _decode_span(tokenizer, input_ids, write) for write in span_info["writes"]
    ]
    if decoded_spans != index_row["write_values"]:
        raise ValueError(
            f"span decode mismatch for id={index_row['id']}: "
            f"{decoded_spans} != {index_row['write_values']}"
        )
    return data_row, enc, span_info


def decode_generation(model, tokenizer, enc, max_new_tokens, ctx=None):
    with torch.no_grad():
        if ctx is None:
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            with attention_bias(ctx):
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
    gen = out[:, enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def generate_without_intervention(
    model,
    tokenizer,
    data_by_id,
    index_row,
    max_new_tokens,
    device,
):
    data_row, enc, _span_info = prepare_example(tokenizer, data_by_id, index_row, device)
    raw = decode_generation(model, tokenizer, enc, max_new_tokens, ctx=None)
    pred = parse_int(raw)
    label = classify_prediction(data_row, pred)
    return {
        "raw": raw,
        "pred": pred,
        "correct": int(label["correct"]),
        "answer_type": label["answer_type"],
        "answered_write_idx": label["answered_write_idx"],
    }


def generate_with_intervention(
    model,
    tokenizer,
    data_by_id,
    index_row,
    heads_by_layer,
    alpha,
    target_kind,
    max_new_tokens,
    device,
):
    data_row, enc, span_info = prepare_example(tokenizer, data_by_id, index_row, device)

    write, boosted_value = select_span(index_row, span_info, target_kind)
    if write is None:
        return {
            "skipped": True,
            "skip_reason": f"no span for target_kind={target_kind}",
            "boosted_value": boosted_value,
        }

    ctx = AttentionBiasContext(
        alpha=float(alpha),
        span_token_start=int(write["token_start"]),
        span_token_end=int(write["token_end"]),
        heads_by_layer=heads_by_layer,
    )
    raw = decode_generation(model, tokenizer, enc, max_new_tokens, ctx=ctx)
    pred = parse_int(raw)
    label = classify_prediction(data_row, pred)
    return {
        "skipped": False,
        "raw": raw,
        "pred": pred,
        "correct": int(label["correct"]),
        "answer_type": label["answer_type"],
        "answered_write_idx": label["answered_write_idx"],
        "target_span": {
            "write_idx": int(write["write_idx"]),
            "token_start": int(write["token_start"]),
            "token_end": int(write["token_end"]),
            "value": str(write["value"]),
            "kind": write["kind"],
        },
        "boosted_value": boosted_value,
    }


def make_record(base_row, run_spec, result):
    baseline_pred = base_row["pred"]
    pred = result.get("pred")
    gold = base_row["gold"]
    boosted_value = result.get("boosted_value")
    boosted_pred = None if pred is None else str(pred)
    return {
        "id": base_row["id"],
        "condition": base_row["condition"],
        "n_lines": base_row["n_lines"],
        "interference_load": row_interference_load(base_row),
        "set": run_spec["set"],
        "run": run_spec["run"],
        "scope": run_spec["scope"],
        "alpha": run_spec["alpha"],
        "target_kind": run_spec["target_kind"],
        "stable_label": base_row["stable_label"],
        "gold": gold,
        "baseline_pred": baseline_pred,
        "baseline_raw": base_row.get("raw"),
        "baseline_source": base_row.get("baseline_source", "stage_a_extract"),
        "baseline_correct": int(base_row["correct"]),
        "baseline_answer_type": base_row.get("answer_type"),
        "baseline_answered_write_idx": base_row.get("answered_write_idx"),
        "pred": pred,
        "raw": result.get("raw"),
        "correct": result.get("correct"),
        "answer_type": result.get("answer_type"),
        "answered_write_idx": result.get("answered_write_idx"),
        "skipped": bool(result.get("skipped", False)),
        "skip_reason": result.get("skip_reason"),
        "target_span": result.get("target_span"),
        "boosted_value": boosted_value,
        "flip_to_correct": int(
            run_spec["set"] == "W" and baseline_pred != gold and pred == gold
        ),
        "harm": int(run_spec["set"] == "C" and baseline_pred == gold and pred != gold),
        "matches_baseline_pred": int(pred == baseline_pred),
        "matches_boosted_value": int(
            boosted_value is not None and boosted_pred == str(boosted_value)
        ),
    }


def make_run_specs(example_sets, alphas, limit, seed, sanity_n=50):
    specs = []
    rng = random.Random(seed)
    wc = example_sets["W"] + example_sets["C"]
    if len(wc) > sanity_n:
        chosen = {row["id"] for row in rng.sample(wc, sanity_n)}
        sanity_rows = [row for row in wc if row["id"] in chosen]
    else:
        sanity_rows = wc
    sanity_rows = cap_rows(sanity_rows, limit)
    specs.append(
        {
            "run": "sanity_alpha1_identity",
            "set": "sanity",
            "scope": "all",
            "alpha": 1.0,
            "target_kind": "current",
            "rows": sanity_rows,
        }
    )

    for set_name in ("W", "C"):
        rows = cap_rows(example_sets[set_name], limit)
        for scope in ("late", "topk"):
            for alpha in alphas:
                specs.append(
                    {
                        "run": "dose_response",
                        "set": set_name,
                        "scope": scope,
                        "alpha": float(alpha),
                        "target_kind": "current",
                        "rows": rows,
                    }
                )
        specs.append(
            {
                "run": "scope_comparison",
                "set": set_name,
                "scope": "all",
                "alpha": 4.0,
                "target_kind": "current",
                "rows": rows,
            }
        )

    specs.append(
        {
            "run": "positive_control_answered_stale",
            "set": "W",
            "scope": "late",
            "alpha": 4.0,
            "target_kind": "answered_stale",
            "rows": cap_rows(example_sets["W"], limit),
        }
    )
    specs.append(
        {
            "run": "simple_noop",
            "set": "S",
            "scope": "late",
            "alpha": 4.0,
            "target_kind": "current",
            "rows": cap_rows(example_sets["S"], limit),
        }
    )
    return specs


def _mean(records, key):
    vals = [row[key] for row in records if row.get(key) is not None and not row["skipped"]]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def summarize(records, metadata):
    groups = {}
    for row in records:
        key = (
            row["run"],
            row["set"],
            row["scope"],
            alpha_tag(row["alpha"]),
            row["target_kind"],
        )
        groups.setdefault(key, []).append(row)

    group_summaries = []
    for key, rows in sorted(groups.items()):
        run, set_name, scope, alpha, target_kind = key
        valid = [row for row in rows if not row["skipped"]]
        group_summaries.append(
            {
                "run": run,
                "set": set_name,
                "scope": scope,
                "alpha": float(alpha.replace("p", ".")),
                "target_kind": target_kind,
                "n": len(rows),
                "n_valid": len(valid),
                "n_skipped": len(rows) - len(valid),
                "accuracy": _mean(rows, "correct"),
                "flip_to_correct_rate": _mean(rows, "flip_to_correct"),
                "harm_rate": _mean(rows, "harm"),
                "matches_baseline_pred_rate": _mean(rows, "matches_baseline_pred"),
                "matches_boosted_value_rate": _mean(rows, "matches_boosted_value"),
            }
        )

    per_i = []
    for summary_row in group_summaries:
        matching = [
            row
            for row in records
            if row["run"] == summary_row["run"]
            and row["set"] == summary_row["set"]
            and row["scope"] == summary_row["scope"]
            and float(row["alpha"]) == float(summary_row["alpha"])
            and row["target_kind"] == summary_row["target_kind"]
        ]
        for i_value in sorted({row["interference_load"] for row in matching}):
            i_rows = [row for row in matching if row["interference_load"] == i_value]
            if not i_rows:
                continue
            per_i.append(
                {
                    "run": summary_row["run"],
                    "set": summary_row["set"],
                    "scope": summary_row["scope"],
                    "alpha": summary_row["alpha"],
                    "target_kind": summary_row["target_kind"],
                    "interference_load": i_value,
                    "n": len(i_rows),
                    "n_valid": len([row for row in i_rows if not row["skipped"]]),
                    "accuracy": _mean(i_rows, "correct"),
                    "flip_to_correct_rate": _mean(i_rows, "flip_to_correct"),
                    "harm_rate": _mean(i_rows, "harm"),
                    "matches_boosted_value_rate": _mean(
                        i_rows, "matches_boosted_value"
                    ),
                }
            )

    sanity = [
        row
        for row in records
        if row["run"] == "sanity_alpha1_identity" and not row["skipped"]
    ]
    sanity_pass = bool(sanity) and all(row["pred"] == row["baseline_pred"] for row in sanity)
    return {
        **metadata,
        "n_records": len(records),
        "sanity_gate": {
            "n": len(sanity),
            "pass": sanity_pass,
            "n_mismatch": sum(row["pred"] != row["baseline_pred"] for row in sanity),
        },
        "groups": group_summaries,
        "per_interference_load": per_i,
    }


def find_group(summary, run, set_name, scope, alpha, target_kind="current"):
    for row in summary["groups"]:
        if (
            row["run"] == run
            and row["set"] == set_name
            and row["scope"] == scope
            and float(row["alpha"]) == float(alpha)
            and row["target_kind"] == target_kind
        ):
            return row
    return None


def make_figures(summary, fig_dir):
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"error": f"matplotlib unavailable: {exc}"}

    alphas = sorted(
        {
            float(row["alpha"])
            for row in summary["groups"]
            if row["run"] == "dose_response" and row["set"] == "W"
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for scope in ("late", "topk"):
        flip = []
        harm = []
        xs = []
        for alpha in alphas:
            w = find_group(summary, "dose_response", "W", scope, alpha)
            c = find_group(summary, "dose_response", "C", scope, alpha)
            if w is not None:
                xs.append(alpha)
                flip.append(w["flip_to_correct_rate"])
                harm.append(None if c is None else c["harm_rate"])
        axes[0].plot(xs, flip, marker="o", label=scope)
        axes[1].plot(xs, harm, marker="o", label=scope)
    axes[0].set_title("W flip to correct")
    axes[1].set_title("C harm")
    for ax in axes:
        ax.set_xlabel("alpha")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("rate")
    out = Path(fig_dir) / "dose_response.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return {"dose_response": str(out)}


def verdict_from_summary(summary):
    late_rates = []
    topk_rates = []
    monotone = False
    best_late = 0.0
    best_topk = 0.0
    best_alpha = None
    for alpha in (2.0, 4.0, 8.0):
        late = find_group(summary, "dose_response", "W", "late", alpha)
        topk = find_group(summary, "dose_response", "W", "topk", alpha)
        if late and late["flip_to_correct_rate"] is not None:
            late_rates.append(late["flip_to_correct_rate"])
            if late["flip_to_correct_rate"] > best_late:
                best_late = late["flip_to_correct_rate"]
                best_alpha = alpha
        if topk and topk["flip_to_correct_rate"] is not None:
            topk_rates.append(topk["flip_to_correct_rate"])
            best_topk = max(best_topk, topk["flip_to_correct_rate"])
    if len(late_rates) >= 2:
        monotone = all(b >= a for a, b in zip(late_rates, late_rates[1:]))

    working_alpha = best_alpha if best_alpha is not None else 4.0
    harm = find_group(summary, "dose_response", "C", "late", working_alpha)
    pos = find_group(
        summary,
        "positive_control_answered_stale",
        "W",
        "late",
        4.0,
        target_kind="answered_stale",
    )
    simple = find_group(summary, "simple_noop", "S", "late", 4.0)
    harm_rate = None if harm is None else harm["harm_rate"]
    pos_rate = None if pos is None else pos["matches_boosted_value_rate"]
    simple_acc = None if simple is None else simple["accuracy"]

    confirmed = (
        best_late >= 0.3
        and monotone
        and pos_rate is not None
        and pos_rate >= 0.5
        and simple_acc is not None
        and simple_acc >= 0.95
        and harm_rate is not None
        and harm_rate <= 0.1
    )
    refuted = best_late < 0.05 and pos_rate is not None and pos_rate >= 0.5
    localization = "undetermined"
    if best_late > 0:
        localization = "head-localized" if best_topk >= 0.7 * best_late else "layer-diffuse"

    if confirmed:
        status = f"transport-selectivity confirmed; {localization}"
    elif refuted:
        status = "transport-selectivity refuted; likely downstream of attention transport"
    else:
        status = f"mixed or inconclusive; {localization}"
    return {
        "status": status,
        "best_late_flip": best_late,
        "best_topk_flip": best_topk,
        "best_alpha": best_alpha,
        "late_monotone": monotone,
        "harm_at_working_alpha": harm_rate,
        "positive_control_match": pos_rate,
        "simple_accuracy": simple_acc,
    }


def write_report(summary, report_path):
    verdict = verdict_from_summary(summary)
    lines = [
        "# Stage B Report",
        "",
        "## Verdict",
        "",
        f"{verdict['status']}. Best late-scope W flip-to-correct rate was "
        f"{verdict['best_late_flip']:.3f} at alpha={verdict['best_alpha']}; "
        f"best top-k flip rate was {verdict['best_topk_flip']:.3f}. "
        f"Late dose-response monotone: {verdict['late_monotone']}.",
        "",
        "## Gates and Controls",
        "",
        f"- Alpha=1 identity gate: {summary['sanity_gate']['pass']} "
        f"(n={summary['sanity_gate']['n']}, "
        f"mismatches={summary['sanity_gate']['n_mismatch']}).",
        f"- Positive-control boosted-value match: "
        f"{verdict['positive_control_match']}.",
        f"- Simple no-op accuracy: {verdict['simple_accuracy']}.",
        f"- Harm on C at working alpha: {verdict['harm_at_working_alpha']}.",
        "",
        "## Primary Metrics",
        "",
        "| run | set | scope | alpha | target | n_valid | acc | flip | harm | boosted_match |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["groups"]:
        lines.append(
            "| {run} | {set} | {scope} | {alpha:g} | {target_kind} | "
            "{n_valid} | {accuracy} | {flip_to_correct_rate} | {harm_rate} | "
            "{matches_boosted_value_rate} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Per-I Breakdown",
            "",
            "| run | set | scope | alpha | I | n_valid | acc | flip | harm | boosted_match |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["per_interference_load"]:
        lines.append(
            "| {run} | {set} | {scope} | {alpha:g} | {interference_load} | "
            "{n_valid} | {accuracy} | {flip_to_correct_rate} | {harm_rate} | "
            "{matches_boosted_value_rate} |".format(**row)
        )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stage_a.jsonl")
    ap.add_argument("--extract-index", default="results/stage_a/extract_index.jsonl")
    ap.add_argument("--stable-labels", default="results/stage_a/stable_labels.jsonl")
    ap.add_argument("--a1-summary", default="results/stage_a/a1_stable_summary.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="results/stage_b/interventions.jsonl")
    ap.add_argument("--summary", default="results/stage_b/summary.json")
    ap.add_argument("--report", default="results/stage_b/REPORT.md")
    ap.add_argument("--fig-dir", default="results/stage_b/figs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alphas", default="2,4,8")
    ap.add_argument("--correct-subsample-n", type=int, default=200)
    ap.add_argument("--sanity-n", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--wrong-filter",
        choices=["any", "within_stale"],
        default="any",
        help="Filter stable-wrong W examples; Stage E Llama uses within_stale.",
    )
    ap.add_argument(
        "--sanity-baseline",
        choices=["auto", "stage_a", "same_model"],
        default="auto",
        help=(
            "Baseline for alpha=1 sanity gate. auto uses Stage-A predictions for "
            "Qwen2.5-7B and a no-bias same-model baseline for smoke models."
        ),
    )
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    data_rows = load_jsonl(args.data)
    data_by_id = {row["id"]: row for row in data_rows}
    extract_rows = load_jsonl(args.extract_index)
    stable_rows = load_jsonl(args.stable_labels)
    with open(args.a1_summary) as f:
        top_summary = json.load(f)

    example_sets = build_example_sets(
        extract_rows,
        stable_rows,
        correct_subsample_n=args.correct_subsample_n,
        seed=args.seed,
        wrong_filter=args.wrong_filter,
    )
    alphas = parse_alphas(args.alphas)
    run_specs = make_run_specs(
        example_sets, alphas=alphas, limit=args.limit, seed=args.seed, sanity_n=args.sanity_n
    )
    sanity_baseline_mode = resolve_sanity_baseline_mode(
        args.sanity_baseline, args.model
    )

    install_attention_patch(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_torch_dtype(args.dtype),
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    scopes = {
        scope: build_scope_heads(scope, num_layers, num_heads, top_summary)
        for scope in ("all", "late", "topk")
    }

    metadata = {
        "model": args.model,
        "dtype": args.dtype,
        "data": args.data,
        "extract_index": args.extract_index,
        "stable_labels": args.stable_labels,
        "a1_summary": args.a1_summary,
        "limit": args.limit,
        "alphas": alphas,
        "seed": args.seed,
        "wrong_filter": args.wrong_filter,
        "sanity_baseline": sanity_baseline_mode,
        "max_new_tokens": args.max_new_tokens,
        "example_set_counts": {k: len(v) for k, v in example_sets.items()},
        "num_layers": num_layers,
        "num_heads": num_heads,
        "topk_heads": [
            {"layer": layer, "head": head}
            for layer, heads in scopes["topk"].items()
            for head in heads
        ],
    }

    records = []
    with open(args.out, "w") as out_f:
        for spec in run_specs:
            rows = spec["rows"]
            print(
                f"run={spec['run']} set={spec['set']} scope={spec['scope']} "
                f"alpha={spec['alpha']} target={spec['target_kind']} n={len(rows)}",
                flush=True,
            )
            heads_by_layer = scopes[spec["scope"]]
            for i, row in enumerate(rows):
                record_base = row
                if (
                    spec["run"] == "sanity_alpha1_identity"
                    and sanity_baseline_mode == "same_model"
                ):
                    baseline = generate_without_intervention(
                        model=model,
                        tokenizer=tokenizer,
                        data_by_id=data_by_id,
                        index_row=row,
                        max_new_tokens=args.max_new_tokens,
                        device=device,
                    )
                    record_base = dict(row)
                    record_base.update(
                        {
                            "pred": baseline["pred"],
                            "raw": baseline["raw"],
                            "correct": baseline["correct"],
                            "answer_type": baseline["answer_type"],
                            "answered_write_idx": baseline["answered_write_idx"],
                            "baseline_source": "same_model_no_bias",
                        }
                    )
                result = generate_with_intervention(
                    model=model,
                    tokenizer=tokenizer,
                    data_by_id=data_by_id,
                    index_row=row,
                    heads_by_layer=heads_by_layer,
                    alpha=spec["alpha"],
                    target_kind=spec["target_kind"],
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                rec = make_record(record_base, spec, result)
                records.append(rec)
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                if i == 0 or (i + 1) % 25 == 0 or (i + 1) == len(rows):
                    print(
                        f"  {i + 1}/{len(rows)} id={row['id']} pred={rec['pred']} "
                        f"baseline={rec['baseline_pred']}",
                        flush=True,
                    )

            if spec["run"] == "sanity_alpha1_identity":
                sanity_summary = summarize(records, metadata)["sanity_gate"]
                if not sanity_summary["pass"]:
                    summary = summarize(records, metadata)
                    summary["figures"] = {}
                    summary["verdict"] = verdict_from_summary(summary)
                    write_json(args.summary, summary)
                    write_report(summary, args.report)
                    raise SystemExit(
                        "alpha=1 identity gate failed: "
                        f"{sanity_summary['n_mismatch']} mismatches"
                    )

    summary = summarize(records, metadata)
    summary["figures"] = make_figures(summary, args.fig_dir)
    summary["verdict"] = verdict_from_summary(summary)
    write_json(args.summary, summary)
    write_report(summary, args.report)
    print(f"wrote {args.out}, {args.summary}, {args.report}", flush=True)


if __name__ == "__main__":
    main()
