"""Stage G-1 circuit-level stale-binding analysis."""

import argparse
import collections
import json
import math
import os
import random
import string
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/x-jguo7-codextmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from extract_stage_a import classify_prediction, resolve_torch_dtype
from gen_tasks import (
    _assignment_lines_from_prompt,
    _prompt_from_assignments,
    make_counterfactual_pair,
    make_example,
)
from run_eval import parse_int
from span_utils import build_chat_prompt, map_target_value_spans
from stage_c_adapters import load_head_sliced_adapter
from stage_g_metrics import assert_self_patch_identity, summarize_patch_effects


DEFAULT_TOP_HEADS = "results/stage_a/a1_stable_summary.json"
ARM_L_ADAPTER = "results/stage_c/adapters_full_ai/arm_l"


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_top_heads(path, top_k=8, num_layers=None, num_heads=None):
    summary = read_json(path)
    heads = []
    for item in summary["top_heads_by_correct_p_last"]:
        layer = int(item["layer"])
        head = int(item["head"])
        if num_layers is not None and (layer >= num_layers or head >= num_heads):
            continue
        heads.append((layer, head))
        if len(heads) >= top_k:
            break
    if not heads:
        raise ValueError("no valid top heads for this model")
    return heads


def load_model(model_name, dtype, adapter_dir=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_torch_dtype(dtype),
        device_map="auto",
        attn_implementation="eager",
    )
    if adapter_dir:
        load_head_sliced_adapter(model, adapter_dir)
    model.eval()
    return model


def load_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def first_token_id(tokenizer, value):
    ids = tokenizer(str(value), add_special_tokens=False)["input_ids"]
    return int(ids[0]) if ids else None


def forward_logits(model, tokenizer, row, output_attentions=False, output_hidden_states=False):
    text = build_chat_prompt(row["prompt"], tokenizer)
    enc = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(
            **enc,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )
    return out, enc


def generate_pred(model, tokenizer, row, max_new_tokens=16):
    text = build_chat_prompt(row["prompt"], tokenizer)
    enc = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    raw = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return raw, parse_int(raw)


def logit_diff_from_output(out, gold_id, stale_id):
    logits = out.logits[0, -1]
    return float((logits[gold_id] - logits[stale_id]).detach().float().cpu())


def assert_pair_span_equality(pair, tokenizer):
    corrupted_all = pair["corrupted"]
    clean_all = pair["clean"]
    # Raw same-length variable replacement should preserve token offsets for values.
    corr_map = map_target_value_spans(corrupted_all, tokenizer)
    clean_map = map_target_value_spans(clean_all, tokenizer)
    corr_current = corr_map["current"]
    clean_current = clean_map["current"]
    if (corr_current["token_start"], corr_current["token_end"]) != (
        clean_current["token_start"],
        clean_current["token_end"],
    ):
        raise ValueError(
            f"current span mismatch for id={corrupted_all.get('id')}: "
            f"{corr_current} vs {clean_current}"
        )
    return True


def make_token_span_equal_pair(corrupted, rng, tokenizer, max_tries=200):
    try:
        return make_tokenizer_aware_pair(corrupted, tokenizer)
    except ValueError:
        pass
    last_error = None
    for _ in range(max_tries):
        pair = make_counterfactual_pair(corrupted, rng)
        try:
            assert_pair_span_equality(pair, tokenizer)
            return pair
        except ValueError as exc:
            last_error = exc
    raise ValueError(f"could not sample token-span-equal counterfactual: {last_error}")


def _line_token_count(tokenizer, name, value):
    return len(tokenizer(f"{name} = {value}\n", add_special_tokens=False)["input_ids"])


def make_tokenizer_aware_pair(corrupted, tokenizer):
    assignments = _assignment_lines_from_prompt(corrupted["prompt"])
    target = corrupted["target_var"]
    write_positions = list(corrupted["target_write_positions"])
    stale_positions = write_positions[:-1]
    existing = {name for name, _ in assignments}
    clean_assignments = list(assignments)
    replacements = {}
    candidates = [
        a + b
        for a in string.ascii_lowercase
        for b in string.ascii_lowercase
        if a + b not in existing
    ]
    for pos in stale_positions:
        old_name, value = assignments[pos]
        if old_name != target:
            raise ValueError(f"expected target at position {pos}")
        old_count = _line_token_count(tokenizer, old_name, value)
        chosen = None
        for name in candidates:
            if name in replacements.values():
                continue
            if _line_token_count(tokenizer, name, value) == old_count:
                chosen = name
                break
        if chosen is None:
            raise ValueError(f"no tokenizer-compatible replacement for {target} at {pos}")
        clean_assignments[pos] = (chosen, value)
        replacements[pos] = chosen
    clean = dict(corrupted)
    clean["condition"] = "counterfactual_clean"
    clean["prompt"] = _prompt_from_assignments(clean_assignments, target)
    clean["target_write_positions"] = [write_positions[-1]]
    clean["stale_values"] = []
    clean["counterfactual_from_id"] = corrupted.get("id")
    clean["counterfactual_stale_write_positions"] = stale_positions
    clean["counterfactual_all_write_positions"] = write_positions
    clean["counterfactual_replacements"] = replacements
    pair = {
        "corrupted": dict(corrupted),
        "clean": clean,
        "corrupted_assignments": assignments,
        "clean_assignments": clean_assignments,
        "span_equal": True,
        "replacements": replacements,
    }
    assert_pair_span_equality(pair, tokenizer)
    return pair


def build_pairs(args):
    rng = random.Random(args.seed)
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, args.dtype)
    selected = []
    candidates = 0
    collisions = 0
    per_load = collections.Counter()
    target_per_load = args.n_pairs // len(args.loads)
    while min([per_load[I] for I in args.loads] or [0]) < target_per_load:
        if candidates >= args.max_candidates:
            break
        for load in args.loads:
            if candidates >= args.max_candidates:
                break
            if per_load[load] >= target_per_load:
                continue
            corrupted = make_example("interference", args.n_lines, load, rng)
            corrupted["id"] = f"stage-g-{load}-{candidates}"
            candidates += 1
            try:
                pair = make_token_span_equal_pair(corrupted, rng, tokenizer)
            except ValueError:
                continue
            raw_c, pred_c = generate_pred(model, tokenizer, corrupted)
            class_c = classify_prediction(corrupted, pred_c)
            raw_q, pred_q = generate_pred(model, tokenizer, pair["clean"])
            clean_ok = pred_q == pair["clean"]["gold"]
            if not (class_c["answer_type"] == "stale" and clean_ok):
                continue
            stale_value = corrupted["stale_values"][class_c["answered_write_idx"]]
            gold_id = first_token_id(tokenizer, corrupted["gold"])
            stale_id = first_token_id(tokenizer, stale_value)
            if gold_id == stale_id:
                collisions += 1
                continue
            selected.append(
                {
                    "id": corrupted["id"],
                    "interference_load": load,
                    "corrupted": corrupted,
                    "clean": pair["clean"],
                    "answered_stale": stale_value,
                    "answered_write_idx": class_c["answered_write_idx"],
                    "corrupted_raw": raw_c,
                    "clean_raw": raw_q,
                    "gold_first_token_id": gold_id,
                    "stale_first_token_id": stale_id,
                }
            )
            per_load[load] += 1
            print(f"selected {len(selected)}/{args.n_pairs} yield={len(selected)}/{candidates}", flush=True)
            if len(selected) >= args.n_pairs:
                break
    write_jsonl(args.out_pairs, selected)
    write_json(
        args.out_summary,
        {
            "model": args.model,
            "dtype": args.dtype,
            "n_pairs": len(selected),
            "candidates": candidates,
            "yield": len(selected) / candidates if candidates else 0.0,
            "max_candidates": args.max_candidates,
            "target_met": len(selected) >= args.n_pairs,
            "first_token_collision_drops": collisions,
            "per_load": dict(per_load),
            "span_equality_asserted": True,
        },
    )
    print(f"wrote {args.out_pairs}")


def layer_output_as_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def patch_layer_output(model, layer_idx, source_final_hidden):
    layer = model.model.layers[layer_idx]

    def hook(_module, _inputs, output):
        hidden = layer_output_as_tensor(output).clone()
        hidden[:, -1, :] = source_final_hidden.to(hidden.device, hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return layer.register_forward_hook(hook)


def collect_layer_final_outputs(model, tokenizer, row):
    saved = {}
    handles = []
    for idx, layer in enumerate(model.model.layers):
        def make_hook(layer_idx):
            def hook(_module, _inputs, output):
                saved[layer_idx] = layer_output_as_tensor(output)[:, -1, :].detach().clone()
            return hook
        handles.append(layer.register_forward_hook(make_hook(idx)))
    try:
        out, _ = forward_logits(model, tokenizer, row)
    finally:
        for h in handles:
            h.remove()
    return out, saved


def run_with_residual_patch(model, tokenizer, row, layer_idx, source_final_hidden):
    handle = patch_layer_output(model, layer_idx, source_final_hidden)
    try:
        out, _ = forward_logits(model, tokenizer, row)
    finally:
        handle.remove()
    return out


def attention_write_mass(attentions, span_info, heads):
    masses = []
    for layer, head in heads:
        final = attentions[layer][0, head, -1, :].detach().float().cpu()
        write_vals = []
        for write in span_info["writes"]:
            write_vals.append(float(final[write["token_start"] : write["token_end"]].sum()))
        masses.append(write_vals)
    return np.asarray(masses, dtype=np.float32)


class ComponentCapture:
    def __init__(self, model, heads, mlp_layers=None):
        self.model = model
        self.heads = heads
        self.mlp_layers = set(mlp_layers or [])
        self.records = {}
        self.handles = []
        self.n_heads = model.config.num_attention_heads
        self.n_kv = getattr(model.config, "num_key_value_heads", self.n_heads)
        self.groups = self.n_heads // self.n_kv
        self.head_dim = model.config.hidden_size // self.n_heads

    def __enter__(self):
        for layer_idx in sorted({layer for layer, _ in self.heads}):
            attn = self.model.model.layers[layer_idx].self_attn
            self.handles.append(attn.q_proj.register_forward_hook(self._hook(layer_idx, "q")))
            self.handles.append(attn.k_proj.register_forward_hook(self._hook(layer_idx, "k")))
            self.handles.append(attn.o_proj.register_forward_pre_hook(self._pre_hook(layer_idx, "head_output")))
        for layer_idx in sorted(self.mlp_layers):
            self.handles.append(self.model.model.layers[layer_idx].mlp.register_forward_hook(self._hook(layer_idx, "mlp")))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()

    def _hook(self, layer_idx, name):
        def hook(_module, _inputs, output):
            self.records[(layer_idx, name)] = output.detach().clone()
        return hook

    def _pre_hook(self, layer_idx, name):
        def hook(_module, inputs):
            self.records[(layer_idx, name)] = inputs[0].detach().clone()
        return hook


def collect_component_outputs(model, tokenizer, row, heads, mlp_layers):
    with ComponentCapture(model, heads, mlp_layers=mlp_layers) as cap:
        out, _ = forward_logits(model, tokenizer, row)
    return out, cap.records, cap


def _head_slice(cap, head, kind):
    if kind == "k":
        head = head // cap.groups
    return slice(head * cap.head_dim, (head + 1) * cap.head_dim)


def _write_positions(span_info, variant):
    if variant == "all":
        writes = span_info["writes"]
    elif variant == "current":
        writes = [span_info["current"]]
    elif variant == "stale":
        writes = span_info["stale"]
    else:
        raise ValueError(f"unknown write-position variant: {variant}")
    return sorted({pos for write in writes for pos in range(write["token_start"], write["token_end"])})


def patch_projection(model, heads, cap, source_records, projection, positions=None):
    handles = []
    layers = sorted({layer for layer, _ in heads})
    for layer_idx in layers:
        attn = model.model.layers[layer_idx].self_attn

        def make_q_hook(layer):
            def hook(_module, _inputs, output):
                patched = output.clone()
                src = source_records[(layer, "q")].to(patched.device, patched.dtype)
                for _layer, head in heads:
                    if _layer != layer:
                        continue
                    sl = _head_slice(cap, head, "q")
                    patched[:, -1, sl] = src[:, -1, sl]
                return patched
            return hook

        def make_k_hook(layer):
            def hook(_module, _inputs, output):
                patched = output.clone()
                src = source_records[(layer, "k")].to(patched.device, patched.dtype)
                for _layer, head in heads:
                    if _layer != layer:
                        continue
                    sl = _head_slice(cap, head, "k")
                    for pos in positions or []:
                        patched[:, pos, sl] = src[:, pos, sl]
                return patched
            return hook

        def make_o_pre_hook(layer):
            def hook(_module, inputs):
                patched = inputs[0].clone()
                src = source_records[(layer, "head_output")].to(patched.device, patched.dtype)
                for _layer, head in heads:
                    if _layer != layer:
                        continue
                    sl = _head_slice(cap, head, "q")
                    patched[:, -1, sl] = src[:, -1, sl]
                return (patched,) + tuple(inputs[1:])
            return hook

        if projection == "q":
            handles.append(attn.q_proj.register_forward_hook(make_q_hook(layer_idx)))
        elif projection == "k":
            handles.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))
        elif projection == "head_output":
            handles.append(attn.o_proj.register_forward_pre_hook(make_o_pre_hook(layer_idx)))
        else:
            raise ValueError(f"unknown projection patch: {projection}")
    return handles


def run_with_projection_patch(model, tokenizer, row, heads, cap, source_records, projection, positions=None):
    handles = patch_projection(model, heads, cap, source_records, projection, positions=positions)
    try:
        out, _ = forward_logits(model, tokenizer, row)
    finally:
        for h in handles:
            h.remove()
    return out


def collect_mlp_outputs(model, tokenizer, row, mlp_layers):
    saved = {}
    handles = []
    for layer_idx in mlp_layers:
        def make_hook(layer):
            def hook(_module, _inputs, output):
                saved[layer] = output[:, -1, :].detach().clone()
            return hook
        handles.append(model.model.layers[layer_idx].mlp.register_forward_hook(make_hook(layer_idx)))
    try:
        out, _ = forward_logits(model, tokenizer, row)
    finally:
        for h in handles:
            h.remove()
    return out, saved


def run_with_mlp_patch(model, tokenizer, row, layer_idx, source_final_output):
    mlp = model.model.layers[layer_idx].mlp

    def hook(_module, _inputs, output):
        patched = output.clone()
        patched[:, -1, :] = source_final_output.to(patched.device, patched.dtype)
        return patched

    handle = mlp.register_forward_hook(hook)
    try:
        out, _ = forward_logits(model, tokenizer, row)
    finally:
        handle.remove()
    return out


def append_patch_row(rows, pair, component, clean_delta, corrupt_delta, patched_delta, layer=None):
    gap = clean_delta - corrupt_delta
    row = {
        "id": pair["id"],
        "component": component,
        "clean_delta": clean_delta,
        "corrupt_delta": corrupt_delta,
        "patched_delta": patched_delta,
        "delta_recovery": (patched_delta - corrupt_delta) / gap if abs(gap) > 1e-9 else math.nan,
    }
    if layer is not None:
        row["layer"] = layer
    rows.append(row)


class QKCapture:
    def __init__(self, model, heads):
        self.model = model
        self.heads = heads
        self.records = {}
        self.handles = []
        self.n_heads = model.config.num_attention_heads
        self.n_kv = getattr(model.config, "num_key_value_heads", self.n_heads)
        self.groups = self.n_heads // self.n_kv
        self.head_dim = model.config.hidden_size // self.n_heads

    def __enter__(self):
        layers = sorted({layer for layer, _ in self.heads})
        for layer_idx in layers:
            attn = self.model.model.layers[layer_idx].self_attn
            self.handles.append(attn.q_proj.register_forward_hook(self._hook(layer_idx, "q")))
            self.handles.append(attn.k_proj.register_forward_hook(self._hook(layer_idx, "k")))
            self.handles.append(attn.v_proj.register_forward_hook(self._hook(layer_idx, "v")))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()

    def _hook(self, layer_idx, name):
        def hook(_module, _inputs, output):
            self.records[(layer_idx, name)] = output.detach().float().cpu()
        return hook

    def q_head(self, layer, head, pos):
        arr = self.records[(layer, "q")][0]
        return arr[pos, head * self.head_dim : (head + 1) * self.head_dim].numpy()

    def k_head(self, layer, head, pos):
        kv_head = head // self.groups
        arr = self.records[(layer, "k")][0]
        return arr[pos, kv_head * self.head_dim : (kv_head + 1) * self.head_dim].numpy()

    def v_head(self, layer, head, pos):
        kv_head = head // self.groups
        arr = self.records[(layer, "v")][0]
        return arr[pos, kv_head * self.head_dim : (kv_head + 1) * self.head_dim].numpy()


def safe_cos(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else math.nan


def geometry_for_rows(model, tokenizer, pairs, heads, limit=None):
    rows = pairs[:limit] if limit else pairs
    margins = []
    identities = []
    recencies = []
    alignments = []
    ov_rows = []
    for pair in rows:
        row = pair["corrupted"]
        span = map_target_value_spans(row, tokenizer)
        with QKCapture(model, heads) as cap:
            out, _ = forward_logits(model, tokenizer, row, output_attentions=True)
        for layer, head in heads:
            q = cap.q_head(layer, head, -1)
            current_pos = span["current"]["token_start"]
            current_k = cap.k_head(layer, head, current_pos)
            stale_ks = [cap.k_head(layer, head, w["token_start"]) for w in span["stale"]]
            cur_score = float(np.dot(q, current_k))
            stale_scores = [float(np.dot(q, k)) for k in stale_ks]
            margins.append(
                {
                    "id": pair["id"],
                    "layer": layer,
                    "head": head,
                    "I": pair["interference_load"],
                    "current_minus_stale_mean": cur_score - float(np.mean(stale_scores)),
                }
            )
            rec_axis = current_k - np.mean(stale_ks, axis=0)
            id_axis = np.mean([current_k] + stale_ks, axis=0)
            recencies.append(safe_cos(current_k, np.mean(stale_ks, axis=0)))
            identities.append(float(np.mean([safe_cos(current_k, k) for k in stale_ks])))
            alignments.append(
                {
                    "id": pair["id"],
                    "layer": layer,
                    "head": head,
                    "I": pair["interference_load"],
                    "query_identity_alignment": safe_cos(q, id_axis),
                    "query_recency_alignment": safe_cos(q, rec_axis),
                }
            )
            # OV sanity: contribution from the value vector at the most attended write.
            attn = out.attentions[layer][0, head, -1, :].detach().float().cpu()
            write_weights = [
                float(attn[w["token_start"] : w["token_end"]].sum()) for w in span["writes"]
            ]
            write = span["writes"][int(np.argmax(write_weights))]
            v = cap.v_head(layer, head, write["token_start"])
            o_proj = model.model.layers[layer].self_attn.o_proj.weight.detach().float().cpu().numpy()
            sl = slice(head * cap.head_dim, (head + 1) * cap.head_dim)
            resid = v @ o_proj[:, sl].T
            lm = model.lm_head.weight.detach().float().cpu().numpy()
            value_id = first_token_id(tokenizer, write["value"])
            gold_id = pair["gold_first_token_id"]
            stale_id = pair["stale_first_token_id"]
            logits = lm @ resid
            ov_rows.append(
                {
                    "id": pair["id"],
                    "layer": layer,
                    "head": head,
                    "write_kind": write["kind"],
                    "value_token_logit": float(logits[value_id]),
                    "gold_token_logit": float(logits[gold_id]),
                    "answered_stale_token_logit": float(logits[stale_id]),
                }
            )
    return {
        "margins": margins,
        "mean_current_stale_key_cosine": float(np.nanmean(recencies)) if recencies else math.nan,
        "mean_identity_key_cosine": float(np.nanmean(identities)) if identities else math.nan,
        "alignments": alignments,
        "ov_rows": ov_rows,
    }


def summarize_geometry(geom):
    margins = np.array([r["current_minus_stale_mean"] for r in geom["margins"]], dtype=np.float64)
    identity_align = np.array([r["query_identity_alignment"] for r in geom["alignments"]], dtype=np.float64)
    recency_align = np.array([r["query_recency_alignment"] for r in geom["alignments"]], dtype=np.float64)
    return {
        "mean_current_minus_stale_qk": float(np.nanmean(margins)) if margins.size else math.nan,
        "median_current_minus_stale_qk": float(np.nanmedian(margins)) if margins.size else math.nan,
        "mean_query_identity_alignment": float(np.nanmean(identity_align)) if identity_align.size else math.nan,
        "mean_query_recency_alignment": float(np.nanmean(recency_align)) if recency_align.size else math.nan,
        "mean_identity_key_cosine": geom["mean_identity_key_cosine"],
        "mean_current_stale_key_cosine": geom["mean_current_stale_key_cosine"],
        "n_head_examples": int(margins.size),
    }


def make_figures(summary, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    effects = summary["patching"]["component_effects"]
    names = [e["component"] for e in effects]
    vals = [e["delta_recovery_mean"] for e in effects]
    plt.figure(figsize=(8, 4))
    plt.bar(names, vals, color="#476A6F")
    plt.ylabel("Mean Δ recovery")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "F11_path_patching_effects.png", dpi=200)
    plt.savefig(out_dir / "F11_path_patching_effects.pdf")
    plt.close()

    base = summary["geometry"]["baseline"]
    arm = summary["geometry"].get("arm_l", {})
    labels = ["QK margin", "query identity", "query recency"]
    base_vals = [
        base.get("mean_current_minus_stale_qk", math.nan),
        base.get("mean_query_identity_alignment", math.nan),
        base.get("mean_query_recency_alignment", math.nan),
    ]
    arm_vals = [
        arm.get("mean_current_minus_stale_qk", math.nan),
        arm.get("mean_query_identity_alignment", math.nan),
        arm.get("mean_query_recency_alignment", math.nan),
    ]
    x = np.arange(len(labels))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.18, base_vals, width=0.36, label="baseline", color="#7A8B99")
    plt.bar(x + 0.18, arm_vals, width=0.36, label="Arm L", color="#C97B5A")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "F12_qk_geometry.png", dpi=200)
    plt.savefig(out_dir / "F12_qk_geometry.pdf")
    plt.close()
    write_json(out_dir / "F11_F12.data.json", summary)


def run_circuit(args):
    pairs = read_jsonl(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, args.dtype)
    heads = load_top_heads(
        args.top_heads,
        top_k=8,
        num_layers=len(model.model.layers),
        num_heads=int(model.config.num_attention_heads),
    )
    n_layers = len(model.model.layers)
    residual_layers = list(range(n_layers)) if args.residual_stride <= 1 else list(range(0, n_layers, args.residual_stride))
    if residual_layers[-1] != n_layers - 1:
        residual_layers.append(n_layers - 1)
    late_mlp_layers = [layer for layer in range(20, min(28, n_layers))]

    patch_rows = []
    identity_deltas = {}
    attention_rows = []
    for i, pair in enumerate(pairs):
        clean_out, clean_layer_outputs = collect_layer_final_outputs(model, tokenizer, pair["clean"])
        corrupt_out, corrupt_layer_outputs = collect_layer_final_outputs(model, tokenizer, pair["corrupted"])
        gold_id = pair["gold_first_token_id"]
        stale_id = pair["stale_first_token_id"]
        clean_delta = logit_diff_from_output(clean_out, gold_id, stale_id)
        corrupt_delta = logit_diff_from_output(corrupt_out, gold_id, stale_id)
        gap = clean_delta - corrupt_delta
        for layer in residual_layers:
            self_out = run_with_residual_patch(
                model, tokenizer, pair["corrupted"], layer, corrupt_layer_outputs[layer]
            )
            self_delta = logit_diff_from_output(self_out, gold_id, stale_id)
            identity_deltas[f"{pair['id']}:residual_l{layer}"] = self_delta - corrupt_delta
            patched = run_with_residual_patch(
                model, tokenizer, pair["corrupted"], layer, clean_layer_outputs[layer]
            )
            patched_delta = logit_diff_from_output(patched, gold_id, stale_id)
            append_patch_row(
                patch_rows,
                pair,
                f"residual_l{layer}",
                clean_delta,
                corrupt_delta,
                patched_delta,
                layer=layer,
            )
        out_c, _ = forward_logits(model, tokenizer, pair["corrupted"], output_attentions=True)
        out_q, _ = forward_logits(model, tokenizer, pair["clean"], output_attentions=True)
        span_c = map_target_value_spans(pair["corrupted"], tokenizer)
        span_q = map_target_value_spans(pair["clean"], tokenizer)
        mass_c = attention_write_mass(out_c.attentions, span_c, heads)
        mass_q = attention_write_mass(out_q.attentions, span_q, heads)
        c_last = mass_c[:, -1]
        c_stale = mass_c[:, :-1].mean(axis=1)
        q_current = mass_q[:, -1]
        attn_effect = float(np.mean(q_current - c_last))
        attention_rows.append(
            {
                "id": pair["id"],
                "component": "selection_head_attention_pattern_proxy",
                "delta_recovery": attn_effect,
                "corrupted_current_mass": float(np.mean(c_last)),
                "corrupted_stale_mass": float(np.mean(c_stale)),
                "clean_current_mass": float(np.mean(q_current)),
            }
        )
        _clean_component_out, clean_component_records, cap = collect_component_outputs(
            model, tokenizer, pair["clean"], heads, late_mlp_layers
        )
        _corrupt_component_out, corrupt_component_records, _ = collect_component_outputs(
            model, tokenizer, pair["corrupted"], heads, late_mlp_layers
        )
        for component, projection, positions in [
            ("selection_head_output_proxy", "head_output", None),
            ("qk_query_final", "q", None),
        ]:
            self_out = run_with_projection_patch(
                model, tokenizer, pair["corrupted"], heads, cap, corrupt_component_records, projection, positions
            )
            self_delta = logit_diff_from_output(self_out, gold_id, stale_id)
            identity_deltas[f"{pair['id']}:{component}"] = self_delta - corrupt_delta
            patched = run_with_projection_patch(
                model, tokenizer, pair["corrupted"], heads, cap, clean_component_records, projection, positions
            )
            patched_delta = logit_diff_from_output(patched, gold_id, stale_id)
            append_patch_row(patch_rows, pair, component, clean_delta, corrupt_delta, patched_delta)
        for variant, component in [
            ("all", "qk_key_all_writes"),
            ("current", "qk_key_current_write"),
            ("stale", "qk_key_stale_writes"),
        ]:
            positions = _write_positions(span_c, variant)
            self_out = run_with_projection_patch(
                model, tokenizer, pair["corrupted"], heads, cap, corrupt_component_records, "k", positions
            )
            self_delta = logit_diff_from_output(self_out, gold_id, stale_id)
            identity_deltas[f"{pair['id']}:{component}"] = self_delta - corrupt_delta
            patched = run_with_projection_patch(
                model, tokenizer, pair["corrupted"], heads, cap, clean_component_records, "k", positions
            )
            patched_delta = logit_diff_from_output(patched, gold_id, stale_id)
            append_patch_row(patch_rows, pair, component, clean_delta, corrupt_delta, patched_delta)
        clean_mlp_out, clean_mlp_outputs = collect_mlp_outputs(model, tokenizer, pair["clean"], late_mlp_layers)
        corrupt_mlp_out, corrupt_mlp_outputs = collect_mlp_outputs(model, tokenizer, pair["corrupted"], late_mlp_layers)
        del clean_mlp_out, corrupt_mlp_out
        for layer in late_mlp_layers:
            self_out = run_with_mlp_patch(
                model, tokenizer, pair["corrupted"], layer, corrupt_mlp_outputs[layer]
            )
            self_delta = logit_diff_from_output(self_out, gold_id, stale_id)
            identity_deltas[f"{pair['id']}:late_mlp_output_l{layer}"] = self_delta - corrupt_delta
            patched = run_with_mlp_patch(
                model, tokenizer, pair["corrupted"], layer, clean_mlp_outputs[layer]
            )
            patched_delta = logit_diff_from_output(patched, gold_id, stale_id)
            append_patch_row(
                patch_rows,
                pair,
                f"late_mlp_output_l{layer}",
                clean_delta,
                corrupt_delta,
                patched_delta,
                layer=layer,
            )
        print(f"patched {i + 1}/{len(pairs)}", flush=True)

    assert_self_patch_identity(identity_deltas, tolerance=args.identity_tolerance)
    write_jsonl(args.patch_rows, patch_rows + attention_rows)

    effects = summarize_patch_effects(patch_rows + attention_rows)

    geom_base = geometry_for_rows(model, tokenizer, pairs, heads, limit=args.geometry_limit)
    geom_base_summary = summarize_geometry(geom_base)
    geom_arm_summary = {}
    if args.arm_l_adapter:
        del model
        torch.cuda.empty_cache()
        arm_model = load_model(args.model, args.dtype, adapter_dir=args.arm_l_adapter)
        geom_arm = geometry_for_rows(arm_model, tokenizer, pairs, heads, limit=args.geometry_limit)
        geom_arm_summary = summarize_geometry(geom_arm)
        write_jsonl(args.geometry_rows.replace(".jsonl", ".arm_l.jsonl"), geom_arm["margins"] + geom_arm["alignments"])
    write_jsonl(args.geometry_rows, geom_base["margins"] + geom_base["alignments"])
    write_jsonl(args.ov_rows, geom_base["ov_rows"])

    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "pairs": args.pairs,
        "n_pairs": len(pairs),
        "top_heads": [{"layer": l, "head": h} for l, h in heads],
        "hard_gates": {
            "self_patching_identity": True,
            "self_patching_identity_tolerance": args.identity_tolerance,
        },
        "patching": {
            "residual_layers": residual_layers,
            "late_mlp_layers": late_mlp_layers,
            "component_effects": effects,
        },
        "geometry": {
            "baseline": geom_base_summary,
            "arm_l": geom_arm_summary,
        },
        "ov_sanity": {
            "n": len(geom_base["ov_rows"]),
            "mean_value_token_logit": float(np.mean([r["value_token_logit"] for r in geom_base["ov_rows"]])) if geom_base["ov_rows"] else math.nan,
            "mean_gold_token_logit": float(np.mean([r["gold_token_logit"] for r in geom_base["ov_rows"]])) if geom_base["ov_rows"] else math.nan,
            "mean_answered_stale_token_logit": float(np.mean([r["answered_stale_token_logit"] for r in geom_base["ov_rows"]])) if geom_base["ov_rows"] else math.nan,
        },
        "limitations": [
            "Attention-pattern patching is reported as an allocation proxy; residual stream patching uses actual activation patch hooks.",
            "Q/K geometry is measured on the first token of each value span.",
        ],
    }
    write_json(args.summary, summary)
    make_figures(summary, args.figure_dir)
    print(f"wrote {args.summary}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-pairs")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--n-lines", type=int, default=40)
    p.add_argument("--loads", type=lambda s: [int(x) for x in s.split(",")], default=[4, 8])
    p.add_argument("--n-pairs", type=int, default=120)
    p.add_argument("--max-candidates", type=int, default=5000)
    p.add_argument("--out-pairs", default="results/stage_g/circuit/pairs.jsonl")
    p.add_argument("--out-summary", default="results/stage_g/circuit/pairs.summary.json")
    p.set_defaults(func=build_pairs)

    r = sub.add_parser("run")
    r.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    r.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    r.add_argument("--pairs", default="results/stage_g/circuit/pairs.jsonl")
    r.add_argument("--top-heads", default=DEFAULT_TOP_HEADS)
    r.add_argument("--arm-l-adapter", default=ARM_L_ADAPTER)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--geometry-limit", type=int, default=None)
    r.add_argument("--residual-stride", type=int, default=4)
    r.add_argument("--identity-tolerance", type=float, default=2e-5)
    r.add_argument("--patch-rows", default="results/stage_g/circuit/patch_rows.jsonl")
    r.add_argument("--geometry-rows", default="results/stage_g/circuit/qk_geometry_rows.jsonl")
    r.add_argument("--ov-rows", default="results/stage_g/circuit/ov_sanity_rows.jsonl")
    r.add_argument("--summary", default="results/stage_g/circuit/summary.json")
    r.add_argument("--figure-dir", default="results/stage_g/figures")
    r.set_defaults(func=run_circuit)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
