"""Capture Stage P event-aligned attention, QK margins, logits, and hidden states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from genattn_gen import CHECKPOINTS, dump_jsonl, load_jsonl


def dump_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_model(model_name: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def model_layers(model):
    if hasattr(model, "gpt_neox"):
        return list(model.gpt_neox.layers), "gpt_neox"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers), str(model.config.model_type)
    raise TypeError(f"unsupported model architecture: {type(model).__name__}")


def attention_module(layer, architecture: str):
    return layer.attention if architecture == "gpt_neox" else layer.self_attn


def marker_token_positions(prompt: str, markers: list[dict], tokenizer) -> tuple[dict, list[int]]:
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    by_role: dict[str, list[list[int]]] = defaultdict(list)
    query_positions: list[int] = []
    for marker in markers:
        positions = [
            index for index, (left, right) in enumerate(offsets)
            if right > int(marker["char_start"]) and left < int(marker["char_end"])
        ]
        if not positions:
            raise ValueError(f"marker maps to no tokens: {marker}")
        observed = prompt[int(marker["char_start"]):int(marker["char_end"])]
        expected = marker.get("value", marker.get("slot"))
        if observed != expected:
            raise ValueError(f"marker text mismatch: {observed!r} != {expected!r}")
        if marker["kind"] == "query_anchor":
            query_positions.extend(positions)
        elif marker["kind"] == "value":
            by_role[str(marker["role"])].append(positions)
        elif marker["kind"] == "identity":
            by_role[str(marker["role"])].append(positions)
    return {
        "input_ids": list(encoded["input_ids"]),
        "offsets": offsets,
        "current_spans": by_role.get("current", []),
        "stale_spans": by_role.get("stale", []),
        "matched_comparator_spans": by_role.get("matched_comparator", []),
        "other_spans": by_role.get("other", []),
        "target_identity_spans": by_role.get("target_identity", []),
        "other_identity_spans": by_role.get("other_identity", []),
        "query_anchor_positions": sorted(set(query_positions)),
    }, list(encoded["input_ids"])


def _flatten(spans: list[list[int]]) -> list[int]:
    return sorted({position for span in spans for position in span})


def one_token_id(tokenizer, value: str) -> int:
    token_ids = tokenizer.encode(" " + str(value), add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Stage P value is not one leading-space token: {value!r} -> {token_ids}")
    return int(token_ids[0])


class QKRecorder:
    """Read-only pre-hooks that reproduce the model's post-RoPE Q/K tensors."""

    def __init__(self, model) -> None:
        self.model = model
        self.layers, self.architecture = model_layers(model)
        self.handles = []
        self.query_index = -1
        self.marker_positions: list[int] = []
        self.q: dict[int, object] = {}
        self.k: dict[int, object] = {}

    def set_positions(self, query_index: int, marker_positions: list[int]) -> None:
        self.query_index = int(query_index)
        self.marker_positions = list(marker_positions)
        self.q.clear()
        self.k.clear()

    @staticmethod
    def _apply_rotary(module, q, k, position_embeddings):
        model_type = str(module.config.model_type)
        cos, sin = position_embeddings
        if model_type == "gpt_neox":
            from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
        elif model_type == "qwen2":
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        elif model_type == "llama":
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        elif model_type == "gemma2":
            from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb
        else:
            raise TypeError(f"unsupported rotary architecture: {model_type}")
        return apply_rotary_pos_emb(q, k, cos, sin)

    def _hook(self, layer_index: int):
        import torch

        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            if hidden is None or position_embeddings is None:
                raise RuntimeError("attention pre-hook did not receive hidden_states and position_embeddings")
            if self.architecture == "gpt_neox":
                batch, sequence, _ = hidden.shape
                head_dim = int(module.head_size)
                qkv = module.query_key_value(hidden).view(batch, sequence, -1, 3 * head_dim).transpose(1, 2)
                q, k, _ = qkv.chunk(3, dim=-1)
            else:
                batch, sequence, _ = hidden.shape
                head_dim = int(module.head_dim)
                q = module.q_proj(hidden).view(batch, sequence, -1, head_dim).transpose(1, 2)
                k = module.k_proj(hidden).view(batch, sequence, -1, head_dim).transpose(1, 2)
            q, k = self._apply_rotary(module, q, k, position_embeddings)
            self.q[layer_index] = q[0, :, self.query_index].detach().float().cpu()
            self.k[layer_index] = k[0, :, self.marker_positions].detach().float().cpu()
            return None

        return hook

    def __enter__(self):
        for index, layer in enumerate(self.layers):
            module = attention_module(layer, self.architecture)
            self.handles.append(module.register_forward_pre_hook(self._hook(index), with_kwargs=True))
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def qk_and_attention_arrays(model, output, recorder: QKRecorder, token_info: dict):
    import torch

    current_positions = _flatten(token_info["current_spans"])
    comparison_spans = token_info["stale_spans"] or token_info["matched_comparator_spans"]
    stale_positions = _flatten(comparison_spans)
    other_positions = _flatten(token_info["other_spans"])
    identity_positions = _flatten(token_info["target_identity_spans"])
    marker_positions = current_positions + [p for p in stale_positions + other_positions + identity_positions if p not in current_positions]
    local = {position: index for index, position in enumerate(marker_positions)}
    n_layers = len(output.attentions)
    n_heads = int(model.config.num_attention_heads)
    ratio = np.full((n_layers, n_heads), np.nan, dtype=np.float32)
    qk_margin = np.full_like(ratio, np.nan)
    attn_current = np.zeros_like(ratio)
    attn_stale = np.zeros_like(ratio)
    attn_other = np.zeros_like(ratio)
    attn_identity = np.zeros_like(ratio)
    epsilon = 1e-9
    for layer_index, attention in enumerate(output.attentions):
        weights = attention[0, :, -1].detach().float().cpu()
        attn_current[layer_index] = weights[:, current_positions].sum(dim=1).numpy()
        if stale_positions:
            attn_stale[layer_index] = weights[:, stale_positions].sum(dim=1).numpy()
            ratio[layer_index] = np.log((attn_current[layer_index] + epsilon) / (attn_stale[layer_index] + epsilon))
        if other_positions:
            attn_other[layer_index] = weights[:, other_positions].sum(dim=1).numpy()
        if identity_positions:
            attn_identity[layer_index] = weights[:, identity_positions].sum(dim=1).numpy()
        if stale_positions:
            q = recorder.q[layer_index]
            k = recorder.k[layer_index]
            n_kv = k.shape[0]
            groups = n_heads // n_kv
            k_for_query = k.repeat_interleave(groups, dim=0)
            scaling = float(attention_module(recorder.layers[layer_index], recorder.architecture).scaling)
            scores = torch.einsum("hd,hpd->hp", q, k_for_query) * scaling
            current_score = scores[:, [local[p] for p in current_positions]].mean(dim=1)
            stale_trace_scores = []
            for span in comparison_spans:
                stale_trace_scores.append(scores[:, [local[p] for p in span]].mean(dim=1))
            strongest_stale = torch.stack(stale_trace_scores, dim=1).max(dim=1).values
            qk_margin[layer_index] = (current_score - strongest_stale).numpy()
    return {
        "read_ratio": ratio,
        "qk_margin": qk_margin,
        "attn_current": attn_current,
        "attn_stale": attn_stale,
        "attn_other": attn_other,
        "attn_identity": attn_identity,
    }


def classify_prediction(prediction: int, gold: int, stale_ids: list[int]) -> str:
    if prediction == gold:
        return "correct_current"
    if prediction in set(stale_ids):
        return "within_stale"
    return "other"


def capture_one(model, tokenizer, device, item: dict, branch: dict, recorder: QKRecorder):
    import torch

    token_info, input_ids_list = marker_token_positions(branch["prompt"], branch["markers"], tokenizer)
    current_positions = _flatten(token_info["current_spans"])
    comparison_spans = token_info["stale_spans"] or token_info["matched_comparator_spans"]
    stale_positions = _flatten(comparison_spans)
    other_positions = _flatten(token_info["other_spans"])
    identity_positions = _flatten(token_info["target_identity_spans"])
    if not current_positions:
        raise ValueError("branch has no current-value marker")
    marker_positions = current_positions + [p for p in stale_positions + other_positions + identity_positions if p not in current_positions]
    encoded = tokenizer(branch["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
    query_index = int(encoded["input_ids"].shape[1] - 1)
    recorder.set_positions(query_index, marker_positions)
    with torch.no_grad():
        output = model(
            **encoded,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    gold_id = one_token_id(tokenizer, branch["gold"])
    stale_ids = [one_token_id(tokenizer, value) for value in branch["stale_values"]]
    comparison_ids = [one_token_id(tokenizer, value) for value in branch.get("comparison_values", branch["stale_values"])]
    logits = output.logits[0, query_index]
    prediction = int(logits.argmax().item())
    strongest_stale_logit = max((float(logits[token_id].item()) for token_id in comparison_ids), default=math.nan)
    arrays = qk_and_attention_arrays(model, output, recorder, token_info)
    hidden = torch.stack([state[0, query_index].detach().float().cpu() for state in output.hidden_states]).numpy()
    metadata = {
        "id": item["id"],
        "semantic_id": item["semantic_id"],
        "seed": item["seed"],
        "template": item["template"],
        "checkpoint": branch["checkpoint"],
        "checkpoint_index": CHECKPOINTS.index(branch["checkpoint"]),
        "control": bool(branch["control"]),
        "target_slot": item["target_slot"],
        "gold": branch["gold"],
        "gold_token_id": gold_id,
        "stale_values": branch["stale_values"],
        "stale_token_ids": stale_ids,
        "comparison_token_ids": comparison_ids,
        "comparison_kind": "semantic_stale" if stale_ids else "matched_foil" if comparison_ids else "none",
        "stale_trace_count": branch["stale_trace_count"],
        "prediction_token_id": prediction,
        "prediction": tokenizer.decode([prediction]),
        "label": classify_prediction(prediction, gold_id, stale_ids),
        "behavioral_margin": float(logits[gold_id].item() - strongest_stale_logit) if comparison_ids else math.nan,
        "prefix_tokens": len(input_ids_list),
        # The nearest comparator is the strongest position-only competitor; keep
        # the farthest distance as an audit field rather than mixing the two.
        "competitor_distance": query_index - max(stale_positions) if stale_positions else 0,
        "competitor_distance_nearest": query_index - max(stale_positions) if stale_positions else 0,
        "competitor_distance_farthest": query_index - min(stale_positions) if stale_positions else 0,
        "current_distance": query_index - max(current_positions),
        "current_span_lengths": [len(span) for span in token_info["current_spans"]],
        "stale_span_lengths": [len(span) for span in token_info["stale_spans"]],
        "query_anchor_tokens": token_info["query_anchor_positions"],
        "current_positions": current_positions,
        "stale_positions": _flatten(token_info["stale_spans"]),
        "comparison_positions": stale_positions,
        "prompt": branch["prompt"],
    }
    return metadata, {**arrays, "hidden": hidden}


def noop_gate(model, tokenizer, device, item: dict) -> dict:
    import torch

    branch = next(row for row in item["branches"] if row["checkpoint"] == "tauQ" and not row["control"])
    encoded = tokenizer(branch["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        baseline = model(**encoded, use_cache=False, output_attentions=True, return_dict=True).logits
    recorder = QKRecorder(model)
    token_info, _ = marker_token_positions(branch["prompt"], branch["markers"], tokenizer)
    positions = _flatten(token_info["current_spans"] + token_info["stale_spans"] + token_info["matched_comparator_spans"] + token_info["other_spans"] + token_info["target_identity_spans"])
    recorder.set_positions(encoded["input_ids"].shape[1] - 1, positions)
    with recorder, torch.no_grad():
        captured = model(**encoded, use_cache=False, output_attentions=True, return_dict=True).logits
    change = float((captured - baseline).abs().max().item())
    return {"max_abs_logit_change": change, "exact_zero": bool(change == 0.0), "n": 1}


def tokenization_gate(items: list[dict], tokenizer) -> dict:
    failures = Counter()
    max_value_span = 0
    for item in items:
        branches = {(row["checkpoint"], row["control"]): row for row in item["branches"]}
        for checkpoint in CHECKPOINTS:
            corrupt = branches[(checkpoint, False)]
            control = branches[(checkpoint, True)]
            corrupt_info, corrupt_ids = marker_token_positions(corrupt["prompt"], corrupt["markers"], tokenizer)
            control_info, control_ids = marker_token_positions(control["prompt"], control["markers"], tokenizer)
            if len(corrupt_ids) != len(control_ids):
                failures["control_length"] += 1
            if corrupt["query_text"] != control["query_text"]:
                failures["fixed_query"] += 1
            for value in {corrupt["gold"], *corrupt.get("comparison_values", corrupt["stale_values"])}:
                try:
                    one_token_id(tokenizer, value)
                except ValueError:
                    failures["candidate_not_single_token"] += 1
            spans = corrupt_info["current_spans"] + corrupt_info["stale_spans"] + control_info["matched_comparator_spans"]
            if spans:
                max_value_span = max(max_value_span, max(map(len, spans)))
    return {
        "n_items": len(items),
        "n_checkpoint_pairs": len(items) * len(CHECKPOINTS),
        "failures": dict(failures),
        "max_value_span_tokens": max_value_span,
        "pass": not failures,
    }


def capture(args) -> None:
    items = load_jsonl(args.data)
    if args.limit_items:
        items = items[:args.limit_items]
    model, tokenizer, device = load_model(args.model, args.dtype)
    token_gate = tokenization_gate(items, tokenizer)
    if not token_gate["pass"]:
        dump_json(Path(args.out_dir) / "tokenization_gate.summary.json", token_gate)
        raise SystemExit("tokenization gate failed")
    noop = noop_gate(model, tokenizer, device, items[0])
    dump_json(Path(args.out_dir) / "hooked_noop.summary.json", noop)
    if not noop["exact_zero"]:
        raise SystemExit("hooked-noop exact-zero gate failed")
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    metadata = []
    with QKRecorder(model) as recorder:
        for item_index, item in enumerate(items):
            for branch in item["branches"]:
                row, values = capture_one(model, tokenizer, device, item, branch, recorder)
                metadata.append(row)
                for key, value in values.items():
                    arrays[key].append(value)
            if (item_index + 1) % 4 == 0 or item_index + 1 == len(items):
                print(f"captured {item_index + 1}/{len(items)} items", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "capture.npz", **{
        key: np.asarray(value, dtype=np.float16 if key == "hidden" else np.float32)
        for key, value in arrays.items()
    })
    dump_jsonl(out_dir / "capture_index.jsonl", metadata)
    layers, architecture = model_layers(model)
    summary = {
        "stage": "P-capture",
        "model": args.model_label or args.model,
        "model_path": args.model,
        "architecture": architecture,
        "dtype": args.dtype,
        "device": str(device),
        "n_items": len(items),
        "data_path": args.data,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "n_branches": len(metadata),
        "n_layers": len(layers),
        "n_query_heads": int(model.config.num_attention_heads),
        "n_key_value_heads": int(getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)),
        "gqa_scope": "head axes are query heads; each query head uses its shared KV head" if getattr(model.config, "num_key_value_heads", model.config.num_attention_heads) != model.config.num_attention_heads else "standard multi-head attention",
        "tokenization_gate": token_gate,
        "hooked_noop_gate": noop,
        "array_shapes": {key: list(np.asarray(value).shape) for key, value in arrays.items()},
        "capture_protocol": "eager attention, output_attentions, no KV cache, greedy next-token logits",
        "comparison_protocol": "overwrite arms compare current to semantic stale traces; no-overwrite arms compare current to exact-position foil-old traces without labeling the foil as stale",
    }
    dump_json(out_dir / "capture.summary.json", summary)
    dump_json(out_dir / "tokenization_gate.summary.json", token_gate)
    print(json.dumps(summary, indent=2), flush=True)


def self_test(_args) -> None:
    assert classify_prediction(1, 1, [2]) == "correct_current"
    assert classify_prediction(2, 1, [2]) == "within_stale"
    assert classify_prediction(3, 1, [2]) == "other"
    print("genattn_capture self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(func=self_test)
    run = sub.add_parser("capture")
    run.add_argument("--data", default="results/event_trace/data/event_items.jsonl")
    run.add_argument("--model", required=True)
    run.add_argument("--model-label")
    run.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    run.add_argument("--limit-items", type=int, default=0)
    run.add_argument("--out-dir", required=True)
    run.set_defaults(func=capture)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
