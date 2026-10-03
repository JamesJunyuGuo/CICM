"""Causal key-side and query-side patching for Stage P matched branches."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from genattn_capture import attention_module, dump_json, load_model, marker_token_positions, model_layers, one_token_id
from genattn_gen import dump_jsonl, load_jsonl


def _current_positions(row: dict, tokenizer) -> list[int]:
    # Capture metadata already passed the tokenizer-offset gate; re-tokenize only
    # to verify the saved positions against this exact model invocation.
    return [int(position) for position in row["current_positions"]]


class ProjectionPatcher:
    """Capture or replace pre-RoPE Q/K projection slices without changing shapes."""

    def __init__(self, model, selected_heads: list[tuple[int, int]], side: str, positions: list[int], source=None):
        if side not in {"query", "key"}:
            raise ValueError(side)
        self.model = model
        self.selected_heads = selected_heads
        self.side = side
        self.positions = positions
        self.source = source
        self.captured = {}
        self.handles = []
        self.layers, self.architecture = model_layers(model)

    def _selected_for_layer(self, layer: int, module) -> list[int]:
        query_heads = [head for target_layer, head in self.selected_heads if target_layer == layer]
        if self.side == "query" or self.architecture == "gpt_neox":
            return sorted(set(query_heads))
        n_query = int(module.config.num_attention_heads)
        n_kv = int(module.config.num_key_value_heads)
        groups = n_query // n_kv
        return sorted({head // groups for head in query_heads})

    def _hook(self, layer_index: int, module):
        import torch

        heads = self._selected_for_layer(layer_index, module)

        def hook(_projection, _inputs, output):
            if not heads:
                return output
            value = output
            if self.architecture == "gpt_neox":
                batch, sequence, _ = output.shape
                n_heads = int(module.config.num_attention_heads)
                head_dim = int(module.head_size)
                view = output.view(batch, sequence, n_heads, 3 * head_dim)
                component = 0 if self.side == "query" else 1
                slices = [
                    view[:, self.positions, head, component * head_dim:(component + 1) * head_dim]
                    for head in heads
                ]
                if self.source is None:
                    self.captured[layer_index] = [part.detach().clone() for part in slices]
                    return output
                value = output.clone()
                patched = value.view(batch, sequence, n_heads, 3 * head_dim)
                for local, head in enumerate(heads):
                    patched[:, self.positions, head, component * head_dim:(component + 1) * head_dim] = self.source[layer_index][local].to(output.device, output.dtype)
                return value
            n_heads = int(module.config.num_attention_heads if self.side == "query" else module.config.num_key_value_heads)
            head_dim = int(module.head_dim)
            view = output.view(output.shape[0], output.shape[1], n_heads, head_dim)
            slices = [view[:, self.positions, head].detach().clone() for head in heads]
            if self.source is None:
                self.captured[layer_index] = slices
                return output
            value = output.clone()
            patched = value.view(output.shape[0], output.shape[1], n_heads, head_dim)
            for local, head in enumerate(heads):
                patched[:, self.positions, head] = self.source[layer_index][local].to(output.device, output.dtype)
            return value

        return hook

    def __enter__(self):
        for layer_index, layer in enumerate(self.layers):
            module = attention_module(layer, self.architecture)
            if not self._selected_for_layer(layer_index, module):
                continue
            projection = module.query_key_value if self.architecture == "gpt_neox" else (module.q_proj if self.side == "query" else module.k_proj)
            self.handles.append(projection.register_forward_hook(self._hook(layer_index, module)))
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def forward_logits(model, tokenizer, device, row: dict, patcher: ProjectionPatcher | None = None):
    import torch

    encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
    context = patcher if patcher is not None else _NullContext()
    with context, torch.no_grad():
        logits = model(**encoded, use_cache=False, return_dict=True).logits[0, -1]
    gold_id = int(row["gold_token_id"])
    stale_ids = [int(token_id) for token_id in row["stale_token_ids"]]
    stale_logits = [float(logits[token_id].item()) for token_id in stale_ids]
    strongest = max(stale_logits) if stale_logits else float("nan")
    prediction = int(logits.argmax().item())
    label = "correct_current" if prediction == gold_id else "within_stale" if prediction in set(stale_ids) else "other"
    return logits.detach().cpu(), {
        "prediction_token_id": prediction,
        "label": label,
        "margin": float(logits[gold_id].item() - strongest) if stale_logits else float("nan"),
    }


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return None


def capture_source(model, tokenizer, device, row: dict, heads: list[tuple[int, int]], side: str, positions: list[int]):
    patcher = ProjectionPatcher(model, heads, side, positions, source=None)
    forward_logits(model, tokenizer, device, row, patcher)
    return patcher.captured


def patch_arm(model, tokenizer, device, clean: dict, corrupt: dict, heads: list[tuple[int, int]], side: str):
    query_position = len(tokenizer(corrupt["prompt"], add_special_tokens=False)["input_ids"]) - 1
    positions = [query_position] if side == "query" else _current_positions(corrupt, tokenizer)
    clean_positions = [query_position] if side == "query" else _current_positions(clean, tokenizer)
    if positions != clean_positions:
        raise ValueError(f"clean/corrupt {side} positions differ")
    clean_source = capture_source(model, tokenizer, device, clean, heads, side, clean_positions)
    corrupt_logits, baseline = forward_logits(model, tokenizer, device, corrupt)
    identity_source = capture_source(model, tokenizer, device, corrupt, heads, side, positions)
    identity_logits, identity = forward_logits(
        model, tokenizer, device, corrupt,
        ProjectionPatcher(model, heads, side, positions, source=identity_source),
    )
    patched_logits, patched = forward_logits(
        model, tokenizer, device, corrupt,
        ProjectionPatcher(model, heads, side, positions, source=clean_source),
    )
    return {
        "baseline": baseline,
        "identity": identity,
        "patched": patched,
        "identity_max_abs_logit_change": float((identity_logits - corrupt_logits).abs().max().item()),
        "patched_max_abs_logit_change": float((patched_logits - corrupt_logits).abs().max().item()),
        "source": clean_source,
    }


def bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    for iteration in range(n_boot):
        indices = rng.choice(len(values), len(values), replace=True)
        samples[iteration] = values[indices].mean()
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def summarize_path(rows: list[dict], path: str, n_boot: int, seed: int) -> dict:
    identity_changes = np.asarray([row[path]["identity_max_abs_logit_change"] for row in rows])
    margin_changes = np.asarray([
        row[path]["patched"]["margin"] - row[path]["baseline"]["margin"] for row in rows
    ])
    corrections = np.asarray([row[path]["patched"]["label"] == "correct_current" for row in rows], dtype=np.float64)
    return {
        "n": len(rows),
        "identity_max_abs_logit_change": float(identity_changes.max(initial=0.0)),
        "identity_exact_zero": bool(np.all(identity_changes == 0.0)),
        "mean_margin_change": float(margin_changes.mean()),
        "margin_change_bootstrap95": bootstrap_mean(margin_changes, n_boot, seed),
        "correction_rate": float(corrections.mean()),
        "correction_rate_bootstrap95": bootstrap_mean(corrections, n_boot, seed + 1),
        "patched_label_counts": dict(Counter(row[path]["patched"]["label"] for row in rows)),
    }


def run(args) -> None:
    analysis = json.loads(Path(args.analysis_summary).read_text(encoding="utf-8"))
    checkpoint = analysis["earliest_stale_favoring_checkpoint"]
    if checkpoint is None:
        raise SystemExit("analysis found no stale-favoring checkpoint; causal patching is gated off")
    heads = [(int(row["layer"]), int(row["head"])) for row in analysis["selected_patch_heads"]]
    if not heads:
        raise SystemExit("analysis selected no discovery heads")
    pairs = [row for row in load_jsonl(args.pairs) if row["split"] == "evaluation"]
    if args.limit_pairs:
        pairs = pairs[:args.limit_pairs]
    if not pairs:
        raise SystemExit("no held-out exact patch pairs")
    model, tokenizer, device = load_model(args.model, args.dtype)
    output_rows = []
    key_invariance = []
    for index, pair in enumerate(pairs):
        row = {"pair_id": pair["pair_id"], "semantic_id": pair["semantic_id"], "checkpoint": checkpoint}
        for scope, clean_name, corrupt_name in (
            ("early", "early_clean", "early_corrupt"),
            ("final", "final_clean", "final_corrupt"),
        ):
            clean = pair[clean_name]
            corrupt = pair[corrupt_name]
            if clean["prefix_tokens"] != corrupt["prefix_tokens"]:
                raise ValueError("patch pair is not exact length")
            for side in ("query", "key"):
                result = patch_arm(model, tokenizer, device, clean, corrupt, heads, side)
                row[f"{scope}_{side}"] = {key: value for key, value in result.items() if key != "source"}
                if side == "key":
                    if scope == "early":
                        row["_early_key_source"] = result["source"]
                    else:
                        differences = []
                        for layer in result["source"]:
                            for early_tensor, final_tensor in zip(row["_early_key_source"][layer], result["source"][layer]):
                                differences.append(float((early_tensor - final_tensor).abs().max().item()))
                        key_invariance.append(max(differences, default=0.0))
        row.pop("_early_key_source", None)
        output_rows.append(row)
        if (index + 1) % 4 == 0 or index + 1 == len(pairs):
            print(f"patched {index + 1}/{len(pairs)} held-out pairs", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "patch_rows.jsonl", output_rows)
    paths = [f"{scope}_{side}" for scope in ("early", "final") for side in ("query", "key")]
    summaries = {
        path: summarize_path(output_rows, path, args.n_boot, args.seed + offset * 10)
        for offset, path in enumerate(paths)
    }
    identity_pass = all(summary["identity_exact_zero"] for summary in summaries.values())
    summary = {
        "stage": "P-causal-patching",
        "model": args.model_label,
        "checkpoint": checkpoint,
        "selection_scope": "heads ranked on discovery only; all effects below use stable_split evaluation only",
        "selected_heads": [{"layer": layer, "head": head} for layer, head in heads],
        "n_pairs": len(output_rows),
        "paths": summaries,
        "identity_patch_exact_zero_all_paths": identity_pass,
        "key_cache_invariance_max_abs": float(max(key_invariance, default=0.0)),
        "key_cache_invariance_exact": bool(all(value == 0.0 for value in key_invariance)),
        "gqa_scope": "key patches deduplicate query heads onto shared KV heads" if getattr(model.config, "num_key_value_heads", model.config.num_attention_heads) != model.config.num_attention_heads else "standard multi-head attention",
    }
    dump_json(out_dir / "patch.summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not identity_pass:
        raise SystemExit(2)


def self_test(_args) -> None:
    assert bootstrap_mean(np.asarray([0.0, 1.0]), 20, 1)[0] <= 0.5
    print("genattn_patch self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("self-test")
    test.set_defaults(func=self_test)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--model-label", required=True)
    run_parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    run_parser.add_argument("--analysis-summary", required=True)
    run_parser.add_argument("--pairs", required=True)
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--limit-pairs", type=int, default=0)
    run_parser.add_argument("--n-boot", type=int, default=2000)
    run_parser.add_argument("--seed", type=int, default=20260822)
    run_parser.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
