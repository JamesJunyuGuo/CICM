"""Stage O Part C/D adapters for base-model tokenizer gates and Qwen2 hooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from contextlib import contextmanager
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pythia_eval import dump_json, wilson
from pythia_gen import (
    TEMPLATES,
    VAR_CANDIDATES,
    _fixed_examples,
    annotate_spans,
    dump_jsonl,
    load_jsonl,
    make_row,
    one_token_strings,
    value_token_id,
    validate_row,
)
from stage_n_attention_sink import (
    QWEN_VALUE_CANDIDATES,
    make_qwen_clean_counterfactual,
    qwen_single_token_values,
    render_qwen_prompt,
)


SEEDS = (11, 29, 47)
PART_D_SPECS = (
    ("qwen05b", "Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
    ("qwen15b", "Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
    ("qwen3b", "Qwen2.5-3B", "Qwen/Qwen2.5-3B"),
    ("mistral7b", "Mistral-7B-v0.3", "mistralai/Mistral-7B-v0.3"),
    ("llama32_1b", "Llama-3.2-1B", "meta-llama/Llama-3.2-1B"),
    ("gemma2_2b", "Gemma-2-2B", "google/gemma-2-2b"),
    ("gemma2_9b", "Gemma-2-9B", "google/gemma-2-9b"),
)


def query_to_kv_head(query_head: int, n_query_heads: int, n_kv_heads: int) -> int:
    if n_query_heads % n_kv_heads:
        raise ValueError("query-head count must be divisible by KV-head count")
    if not 0 <= query_head < n_query_heads:
        raise ValueError("query head is out of range")
    return query_head // (n_query_heads // n_kv_heads)


def select_exact_pairs(rows: list[dict], per_template_split: int = 16) -> list[dict]:
    strata = defaultdict(list)
    for row in rows:
        strata[(row["template"], row["split"])].append(row)
    expected = {(template, split) for template in TEMPLATES for split in ("discovery", "evaluation")}
    missing = [key for key in sorted(expected) if len(strata[key]) < per_template_split]
    if missing:
        raise ValueError(f"underpowered exact-pair strata: {missing}")
    selected = []
    for key in sorted(expected):
        selected.extend(sorted(strata[key], key=lambda row: row["id"])[:per_template_split])
    return sorted(selected, key=lambda row: row["id"])


def compute_caps(n_variables: int, n_task_values: int, target_n_lines: int = 15) -> dict:
    n_lines = min(target_n_lines, n_variables - 1, n_task_values - 6)
    if n_lines < 4:
        raise ValueError("tokenizer leaves no valid few-shot substrate")
    # The frozen generator requires three post-current slots so that the same
    # grid remains legal for the one multi-variable taxonomy arm.
    structural_cap = n_lines - 4
    k_cap = structural_cap - structural_cap % 2
    return {
        "target_n_lines": target_n_lines,
        "n_lines": n_lines,
        "k_cap": k_cap,
        "k_grid": list(range(0, k_cap + 1, 2)),
        "constraints": {
            "variables": n_variables - 1,
            "task_values": n_task_values - 6,
            "post_current": structural_cap,
        },
    }


def _one_token_at_span(tokenizer, text: str, start: int, end: int) -> bool:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions = [
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if right > start and left < end
    ]
    return len(positions) == 1


def _assignment_variables(tokenizer, candidates):
    values = []
    for value in candidates:
        text = f"Record:\n{value} = red\n"
        start = text.index(value)
        if _one_token_at_span(tokenizer, text, start, start + len(value)):
            values.append(value)
    return values


def _assignment_values(tokenizer, candidates):
    values = []
    for value in qwen_single_token_values(tokenizer, candidates):
        text = f"Record:\nalpha = {value}\n"
        start = text.index(value)
        if _one_token_at_span(tokenizer, text, start, start + len(value)):
            values.append(value)
    return values


def tokenizer_audit(args) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    all_variables = _assignment_variables(tokenizer, VAR_CANDIDATES)
    reserved_variables = {
        variable
        for template in TEMPLATES
        for events, _target, _answer in _fixed_examples(template)
        for variable, _value in events
    }
    usable_variables = [
        value for value in all_variables if value not in reserved_variables
    ]
    all_values = _assignment_values(tokenizer, QWEN_VALUE_CANDIDATES)
    if len(all_values) < 16:
        raise ValueError("deterministic value candidate list leaves fewer than 16 labels")
    fixed_values = all_values[:10]
    task_values = all_values[10:]
    fixed_value_gate = len(fixed_values) == 10
    caps = compute_caps(len(usable_variables), len(task_values), args.target_n_lines)
    payload = {
        "stage": args.stage,
        "model": args.model,
        "tokenizer_class": tokenizer.__class__.__name__,
        "single_token_variables_before_reserved_filter": all_variables,
        "reserved_demo_variables": sorted(reserved_variables),
        "usable_variables": usable_variables,
        "task_values": task_values,
        "all_single_token_value_candidates": all_values,
        "fixed_demo_values": fixed_values,
        "fixed_demo_single_token_gate": fixed_value_gate,
        "caps": caps,
        "deterministic_rule": (
            "Filter the fixed candidate lists in listed order by canonical bare, "
            "leading-space, and exact assignment-span single-token encoding; do not "
            "add post-hoc labels."
        ),
    }
    dump_json(args.summary, payload)
    print(json.dumps(payload, indent=2), flush=True)


def generate_data(args) -> None:
    from transformers import AutoTokenizer

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    n_lines = int(audit["caps"]["n_lines"])
    permitted = set(int(value) for value in audit["caps"]["k_grid"])
    k_values = args.k_values if args.k_values is not None else sorted(permitted)
    if not set(k_values) <= permitted:
        raise ValueError(f"requested k outside tokenizer-audited grid: {k_values}")
    rows = []
    for seed in args.seeds:
        for k in k_values:
            for variant in args.variants:
                for template in args.templates:
                    for index in range(args.n_per_seed_cell):
                        row = make_row(
                            variant=variant,
                            template=template,
                            k=k,
                            seed=seed,
                            index=index,
                            tokenizer=tokenizer,
                            n_lines=n_lines,
                            variables=audit["usable_variables"],
                            values=audit["task_values"],
                        )
                        row["prompt"], row["task_char_start"] = render_qwen_prompt(
                            row["events"],
                            row["target_var"],
                            row["template"],
                            tuple(audit["fixed_demo_values"]),
                        )
                        row = annotate_spans(row, tokenizer)
                        validate_row(row, tokenizer)
                        rows.append(row)
    dump_jsonl(args.out, rows)
    summary = {
        "stage": args.stage,
        "model_tokenizer": args.model,
        "audit": args.audit,
        "n": len(rows),
        "seeds": args.seeds,
        "templates": args.templates,
        "variants": args.variants,
        "k_values": k_values,
        "n_lines": n_lines,
        "n_per_seed_cell": args.n_per_seed_cell,
        "prompt_token_range": [
            min(row["prompt_tokens"] for row in rows),
            max(row["prompt_tokens"] for row in rows),
        ],
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_part_c(args) -> None:
    from pythia_eval import matched_pool

    source_rows = load_jsonl(args.behavior_rows)
    source_summary = json.loads(Path(args.behavior_summary).read_text(encoding="utf-8"))
    if source_summary.get("model") != args.model:
        raise ValueError("Part D behavior source uses the wrong model")
    k_rows = [
        row for row in source_rows
        if int(row["k"]) == args.k and row["variant"] == "single"
    ]
    counts = Counter(row["label"] for row in k_rows)
    if counts["correct_current"] < 150 or counts["within_stale"] < 120:
        raise ValueError(f"fixed-k behavior pool is underpowered: {dict(counts)}")
    pool = matched_pool(source_rows, args.k, args.seed, args.per_stratum_cap)
    pool_counts = Counter(row["label"] for row in pool)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_out = out_dir / "behavior_rows.jsonl"
    summary_out = out_dir / "behavior.summary.json"
    pool_out = out_dir / "matched_pool.jsonl"
    dump_jsonl(behavior_out, source_rows)
    shutil.copy2(args.behavior_summary, summary_out)
    dump_jsonl(pool_out, pool)
    payload = {
        "stage": "O-Part-C-full-behavior",
        "model": args.model,
        "fixed_k": args.k,
        "full_protocol_n": len(source_rows),
        "fixed_k_n": len(k_rows),
        "fixed_k_counts": dict(counts),
        "matched_pool_n": len(pool),
        "matched_pool_counts": dict(pool_counts),
        "behavior_reuse": {
            "reason": "Part D used the identical 3-seed x 3-template x 72-per-cell protocol",
            "source_rows": args.behavior_rows,
            "source_rows_sha256": _sha256(Path(args.behavior_rows)),
            "source_summary": args.behavior_summary,
            "source_summary_sha256": _sha256(Path(args.behavior_summary)),
            "inference_rerun": False,
        },
    }
    dump_json(out_dir / "behavior_reuse.summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def _raw_qwen_clean(corrupt: dict, tokenizer, fixed_values: tuple[str, ...]) -> dict:
    raw = {
        key: corrupt[key]
        for key in (
            "id", "semantic_id", "variant", "template", "k", "seed", "index",
            "n_lines", "target_var", "gold", "gold_token_id", "stale_values",
            "stale_token_ids", "cross_values", "cross_token_ids", "events",
        )
    }
    raw["prompt"] = corrupt["prompt"]
    raw["task_char_start"] = int(corrupt["task_char_start"])
    clean = make_qwen_clean_counterfactual(raw, tokenizer, fixed_values)
    clean["cross_token_ids"] = [value_token_id(tokenizer, value) for value in clean["cross_values"]]
    clean = annotate_spans(clean, tokenizer)
    validate_row(clean, tokenizer)
    return clean


def _pair_alignment(clean: dict, corrupt: dict, tokenizer) -> tuple[bool, str]:
    clean_ids = tokenizer(clean["prompt"], add_special_tokens=False)["input_ids"]
    corrupt_ids = tokenizer(corrupt["prompt"], add_special_tokens=False)["input_ids"]
    if len(clean_ids) != len(corrupt_ids):
        return False, "prompt token length mismatch"
    clean_spans = [int(write["value_token"]) for write in clean["writes"]]
    corrupt_spans = [int(write["value_token"]) for write in corrupt["writes"]]
    if clean_spans != corrupt_spans:
        return False, "write span mismatch"
    if list(clean["current_value_span"]) != list(corrupt["current_value_span"]):
        return False, "current span mismatch"
    return True, "aligned"


def stale_competitor(row: dict) -> tuple[int, str]:
    """Return the stale token that the model actually selected."""
    pred_token_id = int(row["pred_token_id"])
    stale_ids = [int(token_id) for token_id in row["stale_token_ids"]]
    matches = [
        value
        for token_id, value in zip(stale_ids, row["stale_values"])
        if token_id == pred_token_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"predicted token id {pred_token_id} is not among stale token ids "
            f"exactly once: {stale_ids}"
        )
    return pred_token_id, matches[0]


def stable_stale_subset(rows: list[dict], baseline_labels: list[str]) -> tuple[list[dict], list[dict]]:
    if len(rows) != len(baseline_labels):
        raise ValueError("rows and baseline labels must have equal length")
    stable = []
    mismatches = []
    for row, label in zip(rows, baseline_labels):
        if label == "within_stale":
            stable.append(row)
        else:
            mismatches.append({"id": row["id"], "baseline_label": label})
    return stable, mismatches


def bounded_sequence_length(requested: int, unique_token_count: int) -> int:
    if unique_token_count < 2:
        raise ValueError("induction test requires at least two unique single-token values")
    return min(requested, unique_token_count)


def build_part_c_pairs(args) -> None:
    from pythia_circuit import stable_split
    from pythia_eval import load_model, predict_rows

    rows = [
        row for row in load_jsonl(args.behavior)
        if int(row["k"]) == args.k and row["variant"] == "single"
        and row["label"] == "within_stale"
    ]
    model, tokenizer, device = load_model(args.model, args.dtype)
    valid_values = qwen_single_token_values(tokenizer, QWEN_VALUE_CANDIDATES)
    fixed_values = tuple(valid_values[:10])
    clean_rows = [_raw_qwen_clean(row, tokenizer, fixed_values) for row in rows]
    evaluated = predict_rows(model, tokenizer, device, clean_rows, args.batch_size)
    corrupt = {row["id"]: row for row in rows}
    candidates = []
    rejected = Counter()
    for clean in evaluated:
        original = corrupt[clean["pair_id"]]
        if clean["label"] != "correct_current":
            rejected["clean_not_current"] += 1
            continue
        valid, reason = _pair_alignment(clean, original, tokenizer)
        if not valid:
            rejected[reason] += 1
            continue
        stale_id, stale_value = stale_competitor(original)
        for row in (clean, original):
            row["comparison_stale_token_id"] = stale_id
            row["comparison_stale_value"] = stale_value
        candidates.append(
            {
                "id": original["id"],
                "semantic_id": original["semantic_id"],
                "template": original["template"],
                "split": stable_split(original["semantic_id"]),
                "clean": clean,
                "corrupted": original,
            }
        )
    chosen = select_exact_pairs(candidates, args.per_template_split)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "exact_pairs_all.jsonl", candidates)
    dump_jsonl(out_dir / "exact_pairs.jsonl", chosen)
    smoke_pairs = []
    for split in ("discovery", "evaluation"):
        smoke_pairs.extend([row for row in chosen if row["split"] == split][:2])
    dump_jsonl(out_dir / "exact_pairs_smoke.jsonl", smoke_pairs)
    summary = {
        "stage": "O-Part-C-exact-pairs",
        "model": args.model,
        "fixed_k": args.k,
        "candidate_failures": len(rows),
        "clean_correct_aligned": len(candidates),
        "selected_exact_pairs": len(chosen),
        "selected_counts": {
            "template": dict(Counter(row["template"] for row in chosen)),
            "split": dict(Counter(row["split"] for row in chosen)),
            "template_x_split": {
                f"{template}__{split}": sum(
                    row["template"] == template and row["split"] == split for row in chosen
                )
                for template in TEMPLATES for split in ("discovery", "evaluation")
            },
        },
        "rejected": dict(rejected),
        "pair_gate": bool(len(chosen) >= 90),
        "requirements": "corrupt stale, clean current, equal token length, aligned write spans",
    }
    dump_json(out_dir / "exact_pairs.summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["pair_gate"]:
        raise SystemExit(2)


class _QwenBlockFacade:
    def __init__(self, block):
        self._block = block
        self.attention = SimpleNamespace(dense=block.self_attn.o_proj)
        self.mlp = block.mlp

    def register_forward_hook(self, *args, **kwargs):
        return self._block.register_forward_hook(*args, **kwargs)


class QwenCoreFacade:
    """Expose the small subset of GPT-NeoX names used by frozen Stage-N tools."""

    def __init__(self, model):
        self._model = model
        self.config = model.config
        self.gpt_neox = SimpleNamespace(
            layers=[_QwenBlockFacade(block) for block in model.model.layers],
            final_layer_norm=model.model.norm,
        )
        self.embed_out = model.lm_head

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)


def load_qwen_facade(model_name: str, dtype: str):
    from pythia_eval import load_model

    model, tokenizer, device = load_model(model_name, dtype)
    return QwenCoreFacade(model), tokenizer, device


@contextmanager
def patched_core_qwen_loader(*modules):
    originals = [module.load_model for module in modules]
    for module in modules:
        module.load_model = load_qwen_facade
    try:
        yield
    finally:
        for module, original in zip(modules, originals):
            module.load_model = original


class QwenMechanismCapture:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.head_inputs = {}
        self.mlp_outputs = {}
        self.residual_outputs = {}
        self.final_residual = None

    def clear(self):
        self.head_inputs.clear()
        self.mlp_outputs.clear()
        self.residual_outputs.clear()
        self.final_residual = None

    def __enter__(self):
        for layer_index, block in enumerate(self.model.model.layers):
            self.handles.append(
                block.self_attn.o_proj.register_forward_pre_hook(
                    lambda _module, inputs, layer=layer_index: self.head_inputs.__setitem__(
                        layer, inputs[0].detach()
                    )
                )
            )
            self.handles.append(
                block.mlp.register_forward_hook(
                    lambda _module, _inputs, output, layer=layer_index: self.mlp_outputs.__setitem__(
                        layer, output.detach()
                    )
                )
            )

            def layer_hook(_module, _inputs, output, layer=layer_index):
                hidden = output[0] if isinstance(output, tuple) else output
                self.residual_outputs[layer] = hidden.detach()

            self.handles.append(block.register_forward_hook(layer_hook))
        self.handles.append(
            self.model.model.norm.register_forward_pre_hook(
                lambda _module, inputs: setattr(self, "final_residual", inputs[0].detach())
            )
        )
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def rmsnorm_direct(component, final_residual, norm, logit_direction) -> float:
    import torch

    component = component.float()
    residual = final_residual.float()
    scale = torch.sqrt(residual.pow(2).mean(dim=-1, keepdim=True) + norm.variance_epsilon)
    normalized = component / scale * norm.weight.float()
    return float(torch.dot(normalized, logit_direction.float()).item())


def _competitor_token_id(row: dict) -> int:
    if row.get("label") == "within_stale":
        return int(row["pred_token_id"])
    if row.get("comparison_stale_token_id") is not None:
        return int(row["comparison_stale_token_id"])
    return int(row["strongest_stale_token_id"])


def qwen_one_row_harvest(model, tokenizer, device, row: dict, capture: QwenMechanismCapture):
    import torch

    capture.clear()
    encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        output = model(
            **encoded,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    query = encoded["input_ids"].shape[1] - 1
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size // n_heads)
    gold_id = int(row["gold_token_id"])
    stale_id = _competitor_token_id(row)
    logits = output.logits[0, query]
    direction = model.lm_head.weight[gold_id] - model.lm_head.weight[stale_id]
    final_residual = capture.final_residual[0, query]

    residuals = [output.hidden_states[0][0, query].detach().float().cpu().numpy()]
    residuals.extend(
        capture.residual_outputs[layer][0, query].detach().float().cpu().numpy()
        for layer in range(n_layers)
    )
    residuals.append(output.hidden_states[-1][0, query].detach().float().cpu().numpy())
    logit_lens = []
    for residual in residuals[:-1]:
        tensor = torch.as_tensor(residual, device=device, dtype=final_residual.dtype)
        lens = model.lm_head(model.model.norm(tensor))
        logit_lens.append(float((lens[gold_id] - lens[stale_id]).item()))
    logit_lens.append(float((logits[gold_id] - logits[stale_id]).item()))

    head_dla = np.zeros((n_layers, n_heads), dtype=np.float32)
    mlp_dla = np.zeros(n_layers, dtype=np.float32)
    for layer in range(n_layers):
        head_input = capture.head_inputs[layer][0, query]
        projection = model.model.layers[layer].self_attn.o_proj
        for head in range(n_heads):
            sl = slice(head * head_dim, (head + 1) * head_dim)
            component = torch.matmul(head_input[sl], projection.weight[:, sl].T)
            head_dla[layer, head] = rmsnorm_direct(
                component, final_residual, model.model.norm, direction
            )
        mlp_dla[layer] = rmsnorm_direct(
            capture.mlp_outputs[layer][0, query], final_residual, model.model.norm, direction
        )

    current = np.zeros((n_layers, n_heads), dtype=np.float32)
    stale = np.zeros_like(current)
    identity = np.zeros_like(current)
    for layer, attention in enumerate(output.attentions):
        weights = attention[0, :, query].detach().float().cpu().numpy()
        current[layer] = weights[:, row["current_value_span"]].sum(axis=1)
        stale[layer] = weights[:, row["stale_value_spans"]].sum(axis=1)
        identity[layer] = weights[:, row["target_identity_spans"]].sum(axis=1)
    return {
        "hidden": np.stack(residuals),
        "logit_lens": np.asarray(logit_lens, dtype=np.float32),
        "head_dla": head_dla,
        "mlp_dla": mlp_dla,
        "attn_current": current,
        "attn_stale": stale,
        "attn_identity": identity,
        "final_gap": float((logits[gold_id] - logits[stale_id]).item()),
    }


def harvest_part_c(args) -> None:
    from pythia_eval import load_model

    rows = load_jsonl(args.pool)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer, device = load_model(args.model, args.dtype)
    arrays = defaultdict(list)
    index_rows = []
    with QwenMechanismCapture(model) as capture:
        for index, row in enumerate(rows):
            result = qwen_one_row_harvest(model, tokenizer, device, row, capture)
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
                    "competitor_token_id": _competitor_token_id(row),
                }
            )
            if (index + 1) % 20 == 0 or index + 1 == len(rows):
                print(f"harvested {index + 1}/{len(rows)}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "harvest.npz", **{key: np.asarray(value) for key, value in arrays.items()})
    dump_jsonl(out_dir / "harvest_index.jsonl", index_rows)
    summary = {
        "stage": "O-Part-C-Qwen-harvest",
        "model": args.model,
        "dtype": args.dtype,
        "n": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "hidden_shape": list(np.asarray(arrays["hidden"]).shape),
        "attention_shape": list(np.asarray(arrays["attn_current"]).shape),
        "gqa_scope": "All head axes index query heads; keys/values are shared within KV groups.",
    }
    dump_json(out_dir / "harvest.summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def analyze_part_c(args) -> None:
    import pythia_circuit as core

    core.analyze(args)
    path = Path(args.summary)
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["stage"] = "O-Part-C-probe-attention-DLA"
    summary["gqa_scope"] = "Per-head quantities index query heads; Qwen2.5-1.5B has two shared KV heads."
    dump_json(path, summary)


def path_patch_part_c(args) -> None:
    import pythia_circuit as core

    with patched_core_qwen_loader(core):
        core.path_patch(args)
    first_pass = json.loads((Path(args.out_dir) / "path_patch_first_pass.summary.json").read_text())
    gate = first_pass["identity_gate"]
    gate["exact_zero"] = bool(gate["max_abs_gap_change"] == 0.0 and gate["prediction_changes"] == 0)
    first_pass["stage"] = "O-Part-C-Qwen-path-patch"
    first_pass["gqa_scope"] = "Exhaustive patches index query-head output slices before o_proj."
    dump_json(Path(args.out_dir) / "path_patch_first_pass.summary.json", first_pass)
    if not gate["exact_zero"]:
        raise SystemExit(2)


def ablate_part_c(args) -> None:
    import pythia_circuit as core
    import pythia_crossscale as crossscale

    rows = load_jsonl(args.pool)
    index_rows = load_jsonl(Path(args.harvest_dir) / "harvest_index.jsonl")
    if [row["id"] for row in rows] != [row["id"] for row in index_rows]:
        raise ValueError("harvest index does not match the matched pool")
    data = np.load(Path(args.harvest_dir) / "harvest.npz")
    labels = np.asarray([row["label"] for row in rows], dtype=object)
    splits = np.asarray([core.stable_split(row["semantic_id"]) for row in rows], dtype=object)
    stale_ratio = data["attn_stale"] / np.maximum(
        data["attn_stale"] + data["attn_current"], 1e-12
    )
    ranking = crossscale.rank_stale_promoting_heads(
        stale_ratio, data["head_dla"], labels, splits
    )
    cumulative = json.loads(Path(args.cumulative_summary).read_text(encoding="utf-8"))
    selected_n = min(int(cumulative["selected_count"]), args.max_heads)
    if selected_n < 1:
        raise ValueError("path-patch summary selected no heads")
    target_heads = [
        (int(row["layer"]), int(row["head"])) for row in ranking[:selected_n]
    ]
    requested_rows = [
        row
        for row, split in zip(rows, splits)
        if split == "evaluation" and row["label"] == "within_stale"
    ]
    if not requested_rows:
        raise ValueError("held-out split has no within-stale rows")

    with patched_core_qwen_loader(core, crossscale):
        model, tokenizer, device = crossscale.load_model(args.model, args.dtype)
        baseline = core.evaluate_ablation_arm(
            model, tokenizer, device, requested_rows, [], args.batch_size
        )
        evaluation_rows, mismatch_details = stable_stale_subset(
            requested_rows, list(baseline["labels"])
        )
        if len(evaluation_rows) < 0.9 * len(requested_rows):
            raise RuntimeError(
                "held-out baseline stability below 90%: "
                f"{len(evaluation_rows)}/{len(requested_rows)}"
            )
        targeted = core.evaluate_ablation_arm(
            model, tokenizer, device, evaluation_rows, target_heads, args.batch_size
        )
        target_effect = (
            np.asarray(targeted["labels"], dtype=object) == "correct_current"
        ).astype(float)
        n_layers = int(model.config.num_hidden_layers)
        n_heads = int(model.config.num_attention_heads)
        random_sets = core.layer_matched_random_sets(
            target_heads, n_layers, n_heads, args.n_random, args.seed
        )
        random_rows = []
        random_rates = []
        for index, heads in enumerate(random_sets):
            result = core.evaluate_ablation_arm(
                model, tokenizer, device, evaluation_rows, heads, args.batch_size
            )
            effect = np.asarray(result["labels"], dtype=object) == "correct_current"
            random_rates.append(float(np.mean(effect)))
            random_rows.append(
                {
                    "random_index": index,
                    "heads": [
                        {"layer": layer, "head": head} for layer, head in heads
                    ],
                    "paired_effect": float(np.mean(effect)),
                }
            )
            if (index + 1) % 8 == 0 or index + 1 == len(random_sets):
                print(
                    f"held-out random ablation {index + 1}/{len(random_sets)}",
                    flush=True,
                )

    random = crossscale._random_summary(random_rates)
    effect = core.paired_bootstrap(target_effect, args.n_boot, args.seed + 100)
    summary = {
        "stage": "O-Part-C-Qwen-heldout-ablation",
        "model": args.model,
        "dtype": args.dtype,
        "protocol": {
            "ranking": (
                "discovery stale-attention shift times max(0, negative discovery failure DLA)"
            ),
            "evaluation": "stable_split evaluation within-stale rows only",
            "baseline_stability": (
                "exclude only rows whose no-ablation rerun no longer returns within-stale; "
                "all targeted and random arms use the same retained rows"
            ),
            "minimum_stability_fraction": 0.9,
            "n_random": args.n_random,
            "random_control": "same held-out rows and layer-matched same-size head sets",
            "selected_n_rule": f"min(path-patch selected_count, {args.max_heads})",
        },
        "heads": [{"layer": layer, "head": head} for layer, head in target_heads],
        "ranking": ranking,
        "evaluation_n_requested": len(requested_rows),
        "evaluation_n": len(evaluation_rows),
        "baseline_label_mismatches": len(mismatch_details),
        "baseline_mismatch_details": mismatch_details,
        "targeted_correct_rate_increase": effect,
        "random_effect": random,
        "targeted_minus_random_mean": float(effect["mean"] - random["mean"]),
        "heldout_exceeds_random95": bool(effect["mean"] > random["quantile95"][1]),
        "gqa_scope": "Ablation zeros query-head output slices; shared KV tensors are unchanged.",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "ablation_random_sets.jsonl", random_rows)
    dump_json(out_dir / "ablation_heldout.summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def induction_part_c(args) -> None:
    import torch
    from pythia_circuit import bootstrap_ci, selected_heads_from_summary
    from pythia_eval import load_model

    model, tokenizer, device = load_model(args.model, args.dtype)
    valid_values = qwen_single_token_values(tokenizer, QWEN_VALUE_CANDIDATES)
    token_ids = sorted({value_token_id(tokenizer, value) for value in valid_values})
    requested_length = int(args.sequence_length)
    sequence_length = bounded_sequence_length(requested_length, len(token_ids))
    rng = random.Random(args.seed)
    n_layers = int(model.config.num_hidden_layers)
    n_heads = int(model.config.num_attention_heads)
    induction_scores = np.zeros(
        (args.n_sequences, n_layers, n_heads), dtype=np.float32
    )
    prefix_scores = np.zeros_like(induction_scores)
    baselines = np.zeros_like(induction_scores)
    for sample in range(args.n_sequences):
        first = rng.sample(token_ids, sequence_length)
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
            for offset in range(sequence_length - 1):
                query = sequence_length + offset
                target = offset + 1
                prefix = offset
                per_head_induction.append(weights[:, query, target])
                per_head_prefix.append(weights[:, query, prefix])
                non_targets = [
                    position for position in range(sequence_length) if position != target
                ]
                per_head_baseline.append(weights[:, query, non_targets].mean(axis=1))
            induction_scores[sample, layer] = np.mean(per_head_induction, axis=0)
            prefix_scores[sample, layer] = np.mean(per_head_prefix, axis=0)
            baselines[sample, layer] = np.mean(per_head_baseline, axis=0)
        if (sample + 1) % 8 == 0 or sample + 1 == args.n_sequences:
            print(f"Qwen induction {sample + 1}/{args.n_sequences}", flush=True)

    means = induction_scores.mean(axis=0)
    flattened = means.reshape(-1)
    table = []
    for layer in range(n_layers):
        for head in range(n_heads):
            value = means[layer, head]
            table.append(
                {
                    "layer": layer,
                    "head": head,
                    "induction_score": float(value),
                    "induction_minus_baseline": float(
                        (
                            induction_scores[:, layer, head]
                            - baselines[:, layer, head]
                        ).mean()
                    ),
                    "prefix_match_score": float(prefix_scores[:, layer, head].mean()),
                    "percentile_among_heads": float((flattened <= value).mean()),
                    "ci": bootstrap_ci(
                        induction_scores[:, layer, head],
                        args.n_boot,
                        args.seed + layer * 100 + head,
                    ),
                }
            )
    selected = selected_heads_from_summary(args.cumulative_summary, args.max_key_heads)
    selected_rows = [
        row for row in table if (row["layer"], row["head"]) in selected
    ]
    summary = {
        "stage": "O-Part-C-Qwen-induction",
        "model": args.model,
        "n_sequences": args.n_sequences,
        "sequence_length_per_copy": sequence_length,
        "requested_sequence_length_per_copy": requested_length,
        "available_unique_single_token_values": len(token_ids),
        "sequence_length_adjusted_to_tokenizer": bool(
            sequence_length != requested_length
        ),
        "value_vocabulary": (
            "QWEN_VALUE_CANDIDATES filtered to canonical leading-space single tokens"
        ),
        "definition": (
            "attention from token i in the second copy to token i+1 in the first copy"
        ),
        "all_heads": table,
        "circuit_heads": selected_rows,
        "anchor_gate": {
            "criterion": (
                "at least one inspected circuit head is in the top induction-score decile"
            ),
            "pass": any(
                row["percentile_among_heads"] >= 0.9 for row in selected_rows
            ),
        },
    }
    dump_json(args.summary, summary)
    print(json.dumps(summary, indent=2), flush=True)


def qkov_part_c(args) -> None:
    import torch
    from pythia_circuit import selected_heads_from_summary
    from pythia_eval import load_model
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    rows = load_jsonl(args.pool)
    if args.limit:
        rows = rows[: args.limit]
    selected = selected_heads_from_summary(args.cumulative_summary, args.max_key_heads)
    if not selected:
        raise ValueError("no selected heads for QK/OV analysis")
    model, tokenizer, device = load_model(args.model, args.dtype)
    n_query = int(model.config.num_attention_heads)
    n_kv = int(model.config.num_key_value_heads)
    head_dim = int(model.config.hidden_size // n_query)
    raw = []
    for index, row in enumerate(rows):
        encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
        width = encoded["input_ids"].shape[1]
        position_ids = torch.arange(width, device=device).unsqueeze(0)
        with torch.no_grad():
            output = model(
                **encoded, use_cache=False, output_hidden_states=True,
                output_attentions=True, return_dict=True,
            )
        query_position = width - 1
        writes = {int(write["value_token"]): write for write in row["writes"]}
        target_positions = [
            int(write["value_token"]) for write in row["writes"]
            if write["var"] == row["target_var"]
        ]
        candidate_ids = [int(row["gold_token_id"])] + [int(value) for value in row["stale_token_ids"]]
        for layer, head in selected:
            block = model.model.layers[layer]
            hidden = output.hidden_states[layer]
            normalized = block.input_layernorm(hidden)
            query_states = block.self_attn.q_proj(normalized).view(1, width, n_query, head_dim).transpose(1, 2)
            key_states = block.self_attn.k_proj(normalized).view(1, width, n_kv, head_dim).transpose(1, 2)
            value_states = block.self_attn.v_proj(normalized).view(1, width, n_kv, head_dim).transpose(1, 2)
            cos, sin = model.model.rotary_emb(normalized, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            kv_head = query_to_kv_head(head, n_query, n_kv)
            query = query_states[0, head, query_position]
            scores = torch.matmul(key_states[0, kv_head], query) / math.sqrt(head_dim)
            current_position = int(row["current_value_span"][0])
            stale_positions = [int(value) for value in row["stale_value_spans"]]
            identity_positions = [int(value) for value in row["target_identity_spans"]]
            current_score = float(scores[current_position].item())
            stale_scores = [float(scores[position].item()) for position in stale_positions]
            identity_scores = [float(scores[position].item()) for position in identity_positions]
            target_scores = np.asarray([float(scores[position].item()) for position in target_positions])
            relative = np.asarray(target_positions, dtype=np.float64)
            if len(relative) > 1:
                relative = (relative - relative.mean()) / max(relative.std(), 1e-8)
                positional_slope = float(np.polyfit(relative, target_scores, 1)[0])
            else:
                positional_slope = math.nan
            projection = block.self_attn.o_proj
            sl = slice(head * head_dim, (head + 1) * head_dim)
            ov_rows = []
            for position in target_positions:
                value = value_states[0, kv_head, position]
                component = torch.matmul(value, projection.weight[:, sl].T)
                source_id = int(writes[position]["value_token_id"])
                source_logit = torch.dot(component.float(), model.lm_head.weight[source_id].float())
                alternatives = [token_id for token_id in candidate_ids if token_id != source_id]
                alternative_logits = [
                    torch.dot(component.float(), model.lm_head.weight[token_id].float())
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
                    "id": row["id"], "label": row["label"], "layer": layer,
                    "head": head, "shared_kv_head": kv_head,
                    "qk_current": current_score,
                    "qk_stale_max": max(stale_scores),
                    "qk_stale_mean": float(np.mean(stale_scores)),
                    "qk_latest_margin": current_score - max(stale_scores),
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
    head_summaries = []
    for layer, head in selected:
        head_rows = [row for row in raw if row["layer"] == layer and row["head"] == head]
        by_label = {}
        for label in sorted({row["label"] for row in head_rows}):
            subset = [row for row in head_rows if row["label"] == label]
            by_label[label] = {
                "n": len(subset),
                "qk_latest_margin_mean": float(np.nanmean([row["qk_latest_margin"] for row in subset])),
                "qk_position_slope_mean": float(np.nanmean([row["qk_position_slope"] for row in subset])),
                "current_attention_mean": float(np.mean([row["attention_current"] for row in subset])),
                "stale_attention_sum_mean": float(np.mean([row["attention_stale_sum"] for row in subset])),
                "current_ov_copy_margin_mean": float(np.mean([
                    write["copy_margin"] for record in subset
                    for write in record["ov_writes"] if write["is_current"]
                ])),
                "stale_ov_copy_margin_mean": float(np.mean([
                    write["copy_margin"] for record in subset
                    for write in record["ov_writes"] if not write["is_current"]
                ])),
            }
        head_summaries.append(
            {
                "layer": layer,
                "head": head,
                "shared_kv_head": query_to_kv_head(head, n_query, n_kv),
                "by_label": by_label,
            }
        )
    summary = {
        "stage": "O-Part-C-Qwen-QK-OV",
        "model": args.model,
        "n_rows": len(rows),
        "head_selection": f"first {args.max_key_heads} heads from held-out cumulative-patch set",
        "gqa_scope": (
            f"Metrics index {n_query} query heads; each query head is paired with one of "
            f"{n_kv} shared key/value heads by contiguous GQA group."
        ),
        "qk_definition": "answer-position rotated query dot write-position shared rotated key / sqrt(head_dim)",
        "ov_definition": "source-token direct-unembedding margin of the shared value through the query-head o_proj slice",
        "heads": head_summaries,
    }
    dump_json(out_dir / "qkov.summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def report_part_c(args) -> None:
    root = Path(args.root)
    behavior = json.loads((root / "behavior.summary.json").read_text(encoding="utf-8"))
    reuse = json.loads((root / "behavior_reuse.summary.json").read_text(encoding="utf-8"))
    pairs = json.loads((root / "pairs/exact_pairs.summary.json").read_text(encoding="utf-8"))
    mechanism = json.loads((root / "mechanism/mechanism.summary.json").read_text(encoding="utf-8"))
    first = json.loads((root / "circuit/path_patch_first_pass.summary.json").read_text(encoding="utf-8"))
    cumulative = json.loads((root / "circuit/cumulative_patch.summary.json").read_text(encoding="utf-8"))
    ablation = json.loads((root / "ablation/ablation_heldout.summary.json").read_text(encoding="utf-8"))
    qkov = json.loads((root / "qkov/qkov.summary.json").read_text(encoding="utf-8"))
    induction = json.loads((root / "induction.summary.json").read_text(encoding="utf-8"))
    noop = json.loads((root / "hooked_noop_cpu.summary.json").read_text(encoding="utf-8"))

    top_rows = []
    for head in qkov["heads"][:3]:
        correct = head["by_label"].get("correct_current", {})
        stale = head["by_label"].get("within_stale", {})
        qk_shift = bool(stale["qk_latest_margin_mean"] < correct["qk_latest_margin_mean"])
        ov_positive = bool(
            min(correct["current_ov_copy_margin_mean"], stale["current_ov_copy_margin_mean"]) > 0
        )
        top_rows.append(
            {
                "layer": head["layer"], "head": head["head"],
                "shared_kv_head": head["shared_kv_head"],
                "qk_correct": correct["qk_latest_margin_mean"],
                "qk_stale": stale["qk_latest_margin_mean"],
                "ov_correct": correct["current_ov_copy_margin_mean"],
                "ov_stale": stale["current_ov_copy_margin_mean"],
                "qk_shift": qk_shift, "ov_positive": ov_positive,
            }
        )
    qkov_gate = sum(row["qk_shift"] and row["ov_positive"] for row in top_rows) >= 2
    gates = {
        "hooked_noop_exact_identity": bool(noop["pass"] and noop["identity_exact_zero"]),
        "exact_pair_power": bool(pairs["pair_gate"] and pairs["selected_exact_pairs"] >= 90),
        "selection_component": bool(mechanism["probe"]["above_shuffle95"]),
        "compact_recovery_set": bool(cumulative["clean_candidate_before_ablation"]),
        "heldout_stale_ablation_specific": bool(ablation["heldout_exceeds_random95"]),
        "qk_shift_with_ov_preserved": qkov_gate,
    }
    core_gates = (
        "selection_component", "compact_recovery_set",
        "heldout_stale_ablation_specific", "qk_shift_with_ov_preserved",
    )
    if all(gates[key] for key in core_gates):
        verdict = "mechanism_shape_replication_supported"
    elif gates["selection_component"] and any(gates[key] for key in core_gates[1:]):
        verdict = "mechanism_shape_replication_partial"
    else:
        verdict = "mechanism_shape_replication_not_supported"
    payload = {
        "stage": "O-Part-C-full",
        "model": "Qwen/Qwen2.5-1.5B",
        "fixed_k": reuse["fixed_k"],
        "verdict": verdict,
        "gates": gates,
        "probe": mechanism["probe"],
        "path_patch": {
            "identity_gate": first["identity_gate"],
            "selected_count": cumulative["selected_count"],
            "selected_heads": cumulative["selected_heads"],
            "heldout_recovery": cumulative["selected_recovery"],
            "all_head_joint": cumulative["all_head_joint"],
        },
        "ablation": ablation,
        "qkov_top3": top_rows,
        "induction": induction["anchor_gate"],
        "claim_boundary": (
            "Tests cross-family mechanism shape only: selection, compact causal recovery, "
            "specific stale-promoting ablation, and QK-shift/OV-preserved. It does not "
            "claim head-for-head correspondence with Pythia."
        ),
    }
    dump_json(root / "summary.json", payload)
    lines = [
        "# Stage O Part C: Qwen2.5-1.5B mechanism-shape replication",
        "",
        f"Pre-registered verdict: **{verdict}**.",
        "",
        payload["claim_boundary"],
        "",
        "## Behavior and integrity gates",
        "",
        f"The identical Part-D full behavior run was reused without new inference "
        f"(SHA-256 recorded in `behavior_reuse.summary.json`). At fixed k={reuse['fixed_k']}, "
        f"the pool contains {reuse['fixed_k_counts']['correct_current']} correct and "
        f"{reuse['fixed_k_counts']['within_stale']} within-stale rows; the matched mechanism "
        f"pool contains {reuse['matched_pool_counts']['correct_current']} per class.",
        "",
        f"The 1.5B hooked no-op changes logits by exactly "
        f"{noop['identity_max_abs_logit_change']:.1f} at gamma=1 and actively changes them "
        f"by {noop['active_gamma0_max_abs_logit_change']:.3f} at gamma=0. "
        f"Pair construction yielded {pairs['selected_exact_pairs']} exact clean/corrupt pairs, "
        "balanced across surface template and discovery/held-out split.",
        "",
        "## Selection probe",
        "",
        f"On within-stale failures, the grouped out-of-sample current-value score is "
        f"{mechanism['probe']['within_stale_true_current_score']:.3f}; its value-label shuffle "
        f"95% interval is [{mechanism['probe']['value_label_shuffle95'][0]:.3f}, "
        f"{mechanism['probe']['value_label_shuffle95'][1]:.3f}]. The length-controlled "
        f"failure-minus-correct delta is "
        f"{mechanism['probe']['within_vs_correct']['length_controlled_delta']:+.3f}.",
        "",
        "## Exhaustive query-head patching and held-out ablation",
        "",
        f"Discovery ranking selects {cumulative['selected_count']} of 336 query heads. "
        f"Their held-out recovery is {cumulative['selected_recovery']['mean']:.3f} "
        f"[{cumulative['selected_recovery']['ci'][0]:.3f}, "
        f"{cumulative['selected_recovery']['ci'][1]:.3f}]. The identity patch has maximum "
        f"absolute gap change {first['identity_gate']['max_abs_gap_change']:.1f} and "
        f"{first['identity_gate']['prediction_changes']} prediction changes.",
        "",
        f"The discovery-ranked stale-promoting set raises held-out correction by "
        f"{ablation['targeted_correct_rate_increase']['mean']:.3f}; the 64 layer-matched "
        f"random-set 95% range is [{ablation['random_effect']['quantile95'][0]:.3f}, "
        f"{ablation['random_effect']['quantile95'][1]:.3f}].",
        "",
        "## QK/OV and induction anchor",
        "",
        "Qwen uses grouped-query attention: each row below is a query head paired with its shared KV head.",
        "",
        "| Query head | Shared KV | QK margin correct | QK margin stale | Current OV correct | Current OV stale | Shape |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in top_rows:
        shape = "QK shift + OV preserved" if row["qk_shift"] and row["ov_positive"] else "partial"
        lines.append(
            f"| L{row['layer']}H{row['head']} | {row['shared_kv_head']} | "
            f"{row['qk_correct']:+.3f} | {row['qk_stale']:+.3f} | "
            f"{row['ov_correct']:+.3f} | {row['ov_stale']:+.3f} | {shape} |"
        )
    lines.extend(
        [
            "",
            f"The repeated-token induction anchor is "
            f"**{'PASS' if induction['anchor_gate']['pass'] else 'FAIL'}**.",
            "",
            "## Pre-registered readings",
            "",
        ]
    )
    for key, value in gates.items():
        lines.append(f"- {key}: **{'PASS' if value else 'FAIL'}**")
    lines.extend(["", "Partial outcomes are retained as partial; no held-out reranking was performed.", ""])
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "gates": gates}, indent=2), flush=True)


def projected_recon_gate(rows: list[dict], model: str, multiplier: int = 9) -> dict:
    candidates = []
    for k in sorted({int(row["k"]) for row in rows if int(row["k"]) > 0}):
        subset = [row for row in rows if int(row["k"]) == k]
        counts = Counter(row["label"] for row in subset)
        per_template = defaultdict(Counter)
        for row in subset:
            per_template[row["template"]][row["label"]] += 1
        projected_correct = counts["correct_current"] * multiplier
        projected_stale = counts["within_stale"] * multiplier
        gate = bool(
            projected_correct >= 150
            and projected_stale >= 120
            and len(per_template) == 3
            and all(
                values["correct_current"] * multiplier >= 25
                and values["within_stale"] * multiplier >= 25
                for values in per_template.values()
            )
        )
        candidates.append(
            {
                "k": k,
                "recon_n": len(subset),
                "recon_counts": dict(counts),
                "projected_full_counts": {
                    "correct_current": projected_correct,
                    "within_stale": projected_stale,
                },
                "projected_per_template": {
                    template: {
                        label: count * multiplier for label, count in values.items()
                    }
                    for template, values in per_template.items()
                },
                "projected_gate": gate,
                "balance_objective": min(projected_correct, projected_stale),
            }
        )
    passing = [row for row in candidates if row["projected_gate"]]
    best = max(passing, key=lambda row: (row["balance_objective"], -row["k"])) if passing else None
    k0 = [row for row in rows if int(row["k"]) == 0]
    return {
        "model": model,
        "k0_accuracy": (
            sum(row["label"] == "correct_current" for row in k0) / len(k0)
            if k0
            else None
        ),
        "projection_multiplier": multiplier,
        "candidates": candidates,
        "gate_pass": best is not None,
        "best": best,
    }


def adjudicate_recon(args) -> None:
    arms = [
        projected_recon_gate(load_jsonl(path), model, args.multiplier)
        for model, path in (
            (args.model_05, args.rows_05),
            (args.model_15, args.rows_15),
        )
    ]
    passing = [arm for arm in arms if arm["gate_pass"]]
    if passing:
        passing.sort(
            key=lambda arm: (
                -arm["best"]["balance_objective"],
                0 if arm["model"] == args.model_05 else 1,
            )
        )
        selected = passing[0]
        status = "substrate_selected"
    else:
        selected = None
        status = "substrate_infeasible"
    payload = {
        "stage": "O-Part-C-Step-0",
        "status": status,
        "arms": arms,
        "selected_model": None if selected is None else selected["model"],
        "selected_k": None if selected is None else selected["best"]["k"],
        "tie_break": "0.5B when projected balance objectives tie",
        "gqa_scope": (
            "Per-head quantities are QUERY-head quantities. Qwen2.5 has shared KV "
            "heads within GQA groups; output patching remains query-head specific."
        ),
    }
    noop = (
        json.loads(Path(args.noop_summary).read_text(encoding="utf-8"))
        if args.noop_summary
        else None
    )
    payload["hooked_noop"] = noop
    payload["hook_preparation_pass"] = bool(noop and noop.get("pass"))
    dump_json(args.summary, payload)
    lines = [
        "# Stage O Part C Step 0",
        "",
        f"Substrate verdict: **{status}**.",
        f"Hooked-noop gate: **{'PASS' if payload['hook_preparation_pass'] else 'FAIL'}**.",
        "",
        "| Model | k=0 recon accuracy | Best k | Projected correct | Projected stale | Gate |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for arm in arms:
        best = arm["best"] or {}
        counts = best.get("projected_full_counts", {})
        lines.append(
            f"| {arm['model']} | {arm['k0_accuracy']:.3f} | {best.get('k', '--')} | "
            f"{counts.get('correct_current', '--')} | {counts.get('within_stale', '--')} | "
            f"{'pass' if arm['gate_pass'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            payload["gqa_scope"],
            "",
            "No Part C full mechanism run was started.",
        ]
    )
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def summarize_part_d(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = []
    for key, label, model_name in PART_D_SPECS:
        directory = Path(args.root) / f"partD_{key}"
        behavior = json.loads(
            (directory / "behavior.summary.json").read_text(encoding="utf-8")
        )
        taxonomy = json.loads(
            (directory / "taxonomy_behavior.summary.json").read_text(encoding="utf-8")
        )
        audit = json.loads(
            (directory / "tokenizer_gate.summary.json").read_text(encoding="utf-8")
        )
        cells = []
        for k_text, cell in sorted(behavior["by_k"].items(), key=lambda item: int(item[0])):
            n = int(cell["n"])
            correct = int(cell["counts"].get("correct_current", 0))
            errors = n - correct
            stale = int(cell["counts"].get("within_stale", 0))
            stale_ci = wilson(stale, errors) if errors else (float("nan"), float("nan"))
            cells.append(
                {
                    "k": int(k_text),
                    "n": n,
                    "accuracy": float(cell["accuracy"]),
                    "accuracy_ci": cell["accuracy_wilson95"],
                    "counts": cell["counts"],
                    "stale_share_errors": stale / errors if errors else None,
                    "stale_share_ci": list(stale_ci),
                }
            )
        cap = max(row["k"] for row in cells)
        cap_templates = []
        for template in TEMPLATES:
            cell = behavior["by_cell"][f"k{cap}__single__{template}"]
            cap_templates.append(float(cell["accuracy"]))
        taxonomy_counts = Counter()
        for cell in taxonomy["by_cell"].values():
            taxonomy_counts.update(cell["counts"])
        models.append(
            {
                "key": key,
                "label": label,
                "model": model_name,
                "tokenizer_gate": audit,
                "cells": cells,
                "k0_gate_pass": bool(cells[0]["accuracy"] >= 0.95),
                "cap": cap,
                "template_accuracy_at_cap": dict(zip(TEMPLATES, cap_templates)),
                "template_range_at_cap": max(cap_templates) - min(cap_templates),
                "taxonomy": {
                    "n": int(sum(taxonomy_counts.values())),
                    "counts": dict(taxonomy_counts),
                },
            }
        )
    payload = {
        "stage": "O-Part-D",
        "models": models,
        "claim_boundary": (
            "Behavioral universality matrix only; no probe, circuit, or prevalence claim."
        ),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "summary.json", payload)

    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    colors = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00", "#7A5195")
    markers = ("o", "s", "^", "D", "v", "P", "X")
    for model, color, marker in zip(models, colors, markers):
        x = np.asarray([row["k"] for row in model["cells"]])
        y = np.asarray([row["accuracy"] for row in model["cells"]])
        ci = np.asarray([row["accuracy_ci"] for row in model["cells"]])
        axis.plot(
            x,
            y,
            color=color,
            marker=marker,
            markersize=4.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=1.6,
            label=model["label"],
        )
        axis.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.12, linewidth=0)
    axis.axhline(0.95, color="#666666", linewidth=1.0, linestyle="--", label="k=0 gate")
    axis.set_xlabel("Number of overwrites (k)")
    axis.set_ylabel("Current-value accuracy")
    axis.set_ylim(0, 1.03)
    axis.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "dose_curve.png", dpi=240)
    fig.savefig(out_dir / "dose_curve.pdf")
    plt.close(fig)

    lines = [
        "# Stage O Part D: cross-family behavioral sweep",
        "",
        payload["claim_boundary"],
        "",
        "| Family/size | k=0 acc | acc at cap | errors at cap | stale share (95% CI) | cross | other | template range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        first, last = model["cells"][0], model["cells"][-1]
        counts = last["counts"]
        errors = last["n"] - counts.get("correct_current", 0)
        share = last["stale_share_errors"]
        ci = last["stale_share_ci"]
        share_text = "--" if share is None else f"{share:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(
            f"| {model['label']} | {first['accuracy']:.3f} | {last['accuracy']:.3f} | "
            f"{errors} | {share_text} | {counts.get('cross_variable', 0)} | "
            f"{counts.get('other', 0)} | {model['template_range_at_cap']:.3f} |"
        )
    lines.extend(["", "![Cross-family dose curves](dose_curve.png)", ""])
    for model in models:
        lines.extend(
            [
                f"## {model['label']}",
                "",
                "| k | n | accuracy | correct | stale | cross | other |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cell in model["cells"]:
            counts = cell["counts"]
            lines.append(
                f"| {cell['k']} | {cell['n']} | {cell['accuracy']:.3f} | "
                f"{counts.get('correct_current', 0)} | {counts.get('within_stale', 0)} | "
                f"{counts.get('cross_variable', 0)} | {counts.get('other', 0)} |"
            )
        lines.extend(
            [
                "",
                f"Tokenizer cap: n_lines={model['tokenizer_gate']['caps']['n_lines']}, "
                f"k_cap={model['cap']}. Multi-variable taxonomy counts: "
                f"{model['taxonomy']['counts']}.",
                "",
            ]
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / 'summary.json'), "report": str(out_dir / 'REPORT.md')}, indent=2))


def apply_query_head_scale(hidden, query, heads, gamma: float, n_query_heads: int):
    import torch

    if hidden.shape[-1] % n_query_heads:
        raise ValueError("Qwen o_proj input does not split over query heads")
    if float(gamma) == 1.0:
        return hidden
    value = hidden.clone()
    head_dim = hidden.shape[-1] // n_query_heads
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    for head in heads:
        sl = slice(int(head) * head_dim, (int(head) + 1) * head_dim)
        value[batch, query, sl] *= gamma
    return value


class QwenHeadOutputHook:
    """Pre-o_proj query-head hook; KV sharing does not change this tensor layout."""

    def __init__(self, model, query, selected, gamma):
        self.model = model
        self.query = query
        self.selected = selected
        self.gamma = gamma
        self.handles = []

    def __enter__(self):
        by_layer = defaultdict(list)
        for layer, head in self.selected:
            by_layer[int(layer)].append(int(head))
        n_heads = int(self.model.config.num_attention_heads)
        for layer, heads in by_layer.items():
            def hook(_module, inputs, heads=tuple(heads)):
                return (
                    apply_query_head_scale(
                        inputs[0], self.query, heads, self.gamma, n_heads
                    ),
                )

            self.handles.append(
                self.model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook)
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def hooked_noop(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(args.data)[: args.limit]
    if not rows:
        raise ValueError("hooked-noop needs at least one task row")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    enc = tokenizer(
        [row["prompt"] for row in rows],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    query = enc["attention_mask"].sum(dim=1) - 1
    with torch.no_grad():
        baseline = model(**enc, use_cache=False, return_dict=True).logits
    selected = [(0, 0)]
    with QwenHeadOutputHook(model, query, selected, 1.0), torch.no_grad():
        identity = model(**enc, use_cache=False, return_dict=True).logits
    with QwenHeadOutputHook(model, query, selected, 0.0), torch.no_grad():
        active = model(**enc, use_cache=False, return_dict=True).logits
    identity_max = float((identity - baseline).abs().max().item())
    active_max = float((active - baseline).abs().max().item())
    payload = {
        "stage": "O-Part-C-Qwen-hook-smoke",
        "model": args.model,
        "device": "cpu",
        "n": len(rows),
        "num_query_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
        "selected_query_heads": [{"layer": 0, "head": 0}],
        "identity_max_abs_logit_change": identity_max,
        "identity_exact_zero": bool(identity_max == 0.0),
        "active_gamma0_max_abs_logit_change": active_max,
        "hook_active": bool(active_max > 0.0),
        "pass": bool(identity_max == 0.0 and active_max > 0.0),
        "gqa_scope": (
            "The hook indexes query-head output slices before o_proj; KV heads are "
            "shared and future QK margins must use the corresponding shared key."
        ),
    }
    dump_json(args.summary, payload)
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["pass"]:
        raise SystemExit(2)


def cache_model(args) -> None:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id=args.model, cache_dir=args.cache_dir)
    payload = {"model": args.model, "cache_dir": args.cache_dir, "snapshot": snapshot}
    dump_json(args.summary, payload)
    print(json.dumps(payload, indent=2), flush=True)


def self_test(_args) -> None:
    import torch

    hidden = torch.ones(1, 2, 8)
    result = apply_query_head_scale(hidden, torch.tensor([1]), [1], 0.0, 2)
    assert torch.equal(result[0, 1, :4], torch.ones(4))
    assert torch.equal(result[0, 1, 4:], torch.zeros(4))
    print("qwen_small_circuit self-test: PASS")


def parse_csv(value):
    return [part for part in value.split(",") if part]


def parse_int_csv(value):
    return [int(part) for part in value.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("tokenizer-audit")
    audit.add_argument("--model", required=True)
    audit.add_argument("--target-n-lines", type=int, default=15)
    audit.add_argument("--stage", default="O-tokenizer-gate")
    audit.add_argument("--summary", required=True)
    audit.set_defaults(func=tokenizer_audit)
    generate = sub.add_parser("generate-data")
    generate.add_argument("--model", required=True)
    generate.add_argument("--audit", required=True)
    generate.add_argument("--out", required=True)
    generate.add_argument("--summary", required=True)
    generate.add_argument("--stage", default="O-cross-family-data")
    generate.add_argument("--seeds", type=parse_int_csv, default=list(SEEDS))
    generate.add_argument("--templates", type=parse_csv, default=list(TEMPLATES))
    generate.add_argument("--variants", type=parse_csv, default=["single"])
    generate.add_argument("--k-values", type=parse_int_csv)
    generate.add_argument("--n-per-seed-cell", type=int, default=72)
    generate.set_defaults(func=generate_data)
    prepare_c = sub.add_parser("prepare-part-c")
    prepare_c.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    prepare_c.add_argument("--behavior-rows", required=True)
    prepare_c.add_argument("--behavior-summary", required=True)
    prepare_c.add_argument("--out-dir", required=True)
    prepare_c.add_argument("--k", type=int, default=6)
    prepare_c.add_argument("--seed", type=int, default=20260818)
    prepare_c.add_argument("--per-stratum-cap", type=int, default=28)
    prepare_c.set_defaults(func=prepare_part_c)
    pairs_c = sub.add_parser("build-part-c-pairs")
    pairs_c.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    pairs_c.add_argument("--behavior", required=True)
    pairs_c.add_argument("--out-dir", required=True)
    pairs_c.add_argument("--k", type=int, default=6)
    pairs_c.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    pairs_c.add_argument("--batch-size", type=int, default=64)
    pairs_c.add_argument("--per-template-split", type=int, default=16)
    pairs_c.set_defaults(func=build_part_c_pairs)
    harvest_c = sub.add_parser("harvest-part-c")
    harvest_c.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    harvest_c.add_argument("--pool", required=True)
    harvest_c.add_argument("--out-dir", required=True)
    harvest_c.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    harvest_c.add_argument("--limit", type=int)
    harvest_c.set_defaults(func=harvest_part_c)
    analyze_c = sub.add_parser("analyze-part-c")
    analyze_c.add_argument("--harvest-dir", required=True)
    analyze_c.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    analyze_c.add_argument("--summary", required=True)
    analyze_c.add_argument("--probe-rows", required=True)
    analyze_c.add_argument("--probe-layers", type=parse_int_csv)
    analyze_c.add_argument("--n-boot", type=int, default=2000)
    analyze_c.add_argument("--n-shuffle", type=int, default=2000)
    analyze_c.add_argument("--seed", type=int, default=20260818)
    analyze_c.set_defaults(func=analyze_part_c)
    patch_c = sub.add_parser("path-patch-part-c")
    patch_c.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    patch_c.add_argument("--pairs", required=True)
    patch_c.add_argument("--out-dir", required=True)
    patch_c.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    patch_c.add_argument("--batch-size", type=int, default=8)
    patch_c.add_argument("--limit", type=int)
    patch_c.add_argument("--min-split", type=int, default=16)
    patch_c.add_argument("--n-boot", type=int, default=2000)
    patch_c.add_argument("--seed", type=int, default=20260818)
    patch_c.set_defaults(func=path_patch_part_c)
    ablate_c = sub.add_parser("ablate-part-c")
    ablate_c.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ablate_c.add_argument("--pool", required=True)
    ablate_c.add_argument("--harvest-dir", required=True)
    ablate_c.add_argument("--cumulative-summary", required=True)
    ablate_c.add_argument("--out-dir", required=True)
    ablate_c.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ablate_c.add_argument("--batch-size", type=int, default=32)
    ablate_c.add_argument("--max-heads", type=int, default=12)
    ablate_c.add_argument("--n-random", type=int, default=64)
    ablate_c.add_argument("--n-boot", type=int, default=2000)
    ablate_c.add_argument("--seed", type=int, default=20260818)
    ablate_c.set_defaults(func=ablate_part_c)
    qkov_c = sub.add_parser("qkov-part-c")
    qkov_c.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    qkov_c.add_argument("--pool", required=True)
    qkov_c.add_argument("--cumulative-summary", required=True)
    qkov_c.add_argument("--out-dir", required=True)
    qkov_c.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    qkov_c.add_argument("--limit", type=int, default=96)
    qkov_c.add_argument("--max-key-heads", type=int, default=12)
    qkov_c.set_defaults(func=qkov_part_c)
    induction_c = sub.add_parser("induction-part-c")
    induction_c.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    induction_c.add_argument("--cumulative-summary", required=True)
    induction_c.add_argument("--summary", required=True)
    induction_c.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    induction_c.add_argument("--n-sequences", type=int, default=64)
    induction_c.add_argument("--sequence-length", type=int, default=24)
    induction_c.add_argument("--max-key-heads", type=int, default=12)
    induction_c.add_argument("--n-boot", type=int, default=2000)
    induction_c.add_argument("--seed", type=int, default=20260818)
    induction_c.set_defaults(func=induction_part_c)
    report_c = sub.add_parser("report-part-c")
    report_c.add_argument("--root", default="results/stage_o/partC_qwen15b")
    report_c.set_defaults(func=report_part_c)
    recon = sub.add_parser("adjudicate-recon")
    recon.add_argument("--model-05", default="Qwen/Qwen2.5-0.5B")
    recon.add_argument("--rows-05", required=True)
    recon.add_argument("--model-15", default="Qwen/Qwen2.5-1.5B")
    recon.add_argument("--rows-15", required=True)
    recon.add_argument("--multiplier", type=int, default=9)
    recon.add_argument("--summary", required=True)
    recon.add_argument("--noop-summary")
    recon.add_argument("--report", required=True)
    recon.set_defaults(func=adjudicate_recon)
    part_d = sub.add_parser("summarize-part-d")
    part_d.add_argument("--root", default="results/stage_o")
    part_d.add_argument("--out-dir", required=True)
    part_d.set_defaults(func=summarize_part_d)
    noop = sub.add_parser("hooked-noop")
    noop.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    noop.add_argument("--data", required=True)
    noop.add_argument("--limit", type=int, default=2)
    noop.add_argument("--summary", required=True)
    noop.set_defaults(func=hooked_noop)
    cache = sub.add_parser("cache-model")
    cache.add_argument("--model", required=True)
    cache.add_argument("--cache-dir", required=True)
    cache.add_argument("--summary", required=True)
    cache.set_defaults(func=cache_model)
    test = sub.add_parser("self-test")
    test.set_defaults(func=self_test)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
