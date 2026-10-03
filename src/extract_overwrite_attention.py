"""Stage A extraction: behavior, write attention, and final-token hiddens."""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_eval import parse_int
from span_utils import build_chat_prompt, map_target_value_spans


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def is_stage_a_row(row):
    if row["condition"] == "simple":
        return row["n_lines"] == 40
    if row["condition"] != "interference":
        return False
    n_lines = row["n_lines"]
    load = row["interference_load"]
    return (n_lines == 40 and load in {2, 4, 8}) or (
        n_lines in {20, 80} and load == 4
    )


def select_stage_a_rows(rows, limit=None):
    selected = [row for row in rows if is_stage_a_row(row)]
    return selected[:limit] if limit else selected


def filter_rows_by_index_order(rows, index_rows):
    by_id = {row.get("id"): row for row in rows}
    filtered = []
    missing = []
    for index_row in index_rows:
        row_id = index_row.get("id")
        if row_id not in by_id:
            missing.append(row_id)
        else:
            filtered.append(by_id[row_id])
    if missing:
        preview = ", ".join(str(x) for x in missing[:5])
        raise ValueError(f"{len(missing)} ids from index were not found in data: {preview}")
    return filtered


def classify_prediction(row, pred):
    if pred == row["gold"]:
        return {"correct": True, "answer_type": "correct", "answered_write_idx": None}
    if pred in row["stale_values"]:
        return {
            "correct": False,
            "answer_type": "stale",
            "answered_write_idx": row["stale_values"].index(pred),
        }
    return {"correct": False, "answer_type": "other", "answered_write_idx": None}


def resolve_torch_dtype(dtype_name):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_name}")


def load_model_with_optional_adapter(model_name, dtype_name, adapter_dir, adapter_kind, adapter_mode):
    torch_dtype = resolve_torch_dtype(dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation="eager",
    )
    if adapter_dir:
        if adapter_kind == "generic":
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        elif adapter_kind in {"head_sliced", "random_head_sliced"}:
            if adapter_mode == "off":
                from head_adapters import wrap_head_sliced_adapters

                with open(Path(adapter_dir) / "adapter_config.json") as f:
                    config = json.load(f)
                wrap_head_sliced_adapters(
                    model,
                    config["heads"],
                    rank=int(config["rank"]),
                    alpha=float(config["alpha"]),
                )
            else:
                from head_adapters import load_head_sliced_adapter

                load_head_sliced_adapter(model, adapter_dir)
        else:
            raise ValueError(f"unknown adapter_kind={adapter_kind}")
    model.eval()
    return model


def adapter_disabled_context(model, adapter_kind, adapter_mode):
    if adapter_mode != "off" or adapter_kind != "generic":
        return contextlib.nullcontext()
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("generic adapter-off requested but model lacks disable_adapter()")
    return model.disable_adapter()


def _decode_span(tokenizer, input_ids, write):
    span_ids = input_ids[write["token_start"] : write["token_end"]]
    return tokenizer.decode(span_ids)


def _reduce_attentions(attentions, writes, max_writes):
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    attn_writes = np.zeros((n_layers, n_heads, max_writes), dtype=np.float32)
    attn_total = np.zeros((n_layers, n_heads), dtype=np.float32)
    attn_bos = np.zeros((n_layers, n_heads), dtype=np.float32)

    for layer_idx, layer_attn in enumerate(attentions):
        final_attn = layer_attn[0, :, -1, :].detach().float().cpu()
        attn_total[layer_idx] = final_attn.sum(dim=-1).numpy()
        attn_bos[layer_idx] = final_attn[:, 0].numpy()
        for write in writes:
            s = write["token_start"]
            e = write["token_end"]
            attn_writes[layer_idx, :, write["write_idx"]] = (
                final_attn[:, s:e].sum(dim=-1).numpy()
            )
    return attn_writes, attn_total, attn_bos


def _hidden_final_token(hidden_states):
    return np.stack(
        [h[0, -1, :].detach().to(torch.float16).cpu().numpy() for h in hidden_states],
        axis=0,
    )


def _generate_text(model, tokenizer, enc, max_new_tokens):
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[:, enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/extraction_task.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="results/overwrite_attention_extraction/extract.npz")
    ap.add_argument("--index-out", default="results/overwrite_attention_extraction/extract_index.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--ids-from-index", default=None)
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument(
        "--adapter-kind",
        choices=["generic", "head_sliced", "random_head_sliced"],
        default=None,
    )
    ap.add_argument("--adapter-mode", choices=["on", "off"], default="on")
    ap.add_argument("--no-hidden", action="store_true")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.index_out).parent.mkdir(parents=True, exist_ok=True)

    rows = select_stage_a_rows(load_jsonl(args.data))
    if args.ids_from_index:
        rows = filter_rows_by_index_order(rows, load_jsonl(args.ids_from_index))
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded {len(rows)} Stage-A examples from {args.data}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model_with_optional_adapter(
        args.model,
        args.dtype,
        args.adapter_dir,
        args.adapter_kind,
        args.adapter_mode,
    )
    device = next(model.parameters()).device

    max_writes = max(len(row["stale_values"]) + 1 for row in rows)
    attn_writes_all = []
    attn_total_all = []
    attn_bos_all = []
    hidden_all = []
    index_rows = []

    for i, row in enumerate(rows):
        templated = build_chat_prompt(row["prompt"], tokenizer)
        span_info = map_target_value_spans(row, tokenizer)
        enc = tokenizer(templated, return_tensors="pt").to(device)

        input_ids = enc["input_ids"][0].detach().cpu().tolist()
        decoded_spans = [
            _decode_span(tokenizer, input_ids, write) for write in span_info["writes"]
        ]
        expected_spans = [str(v) for v in row["stale_values"] + [row["gold"]]]
        spans_ok = decoded_spans == expected_spans
        if not spans_ok:
            raise ValueError(
                f"span decode mismatch for id={row.get('id')}: "
                f"{decoded_spans} != {expected_spans}"
            )

        with adapter_disabled_context(model, args.adapter_kind, args.adapter_mode):
            with torch.no_grad():
                outputs = model(
                    **enc,
                    output_attentions=True,
                    output_hidden_states=not args.no_hidden,
                    use_cache=False,
                )
            raw = _generate_text(model, tokenizer, enc, args.max_new_tokens).strip()
        pred = parse_int(raw)
        label = classify_prediction(row, pred)
        attn_writes, attn_total, attn_bos = _reduce_attentions(
            outputs.attentions, span_info["writes"], max_writes
        )

        attn_writes_all.append(attn_writes)
        attn_total_all.append(attn_total)
        attn_bos_all.append(attn_bos)
        if not args.no_hidden:
            hidden_all.append(_hidden_final_token(outputs.hidden_states))

        index_rows.append(
            {
                "row_index": i,
                "id": row.get("id"),
                "condition": row["condition"],
                "n_lines": row["n_lines"],
                "interference_load": row["interference_load"],
                "target_var": row["target_var"],
                "gold": row["gold"],
                "stale_values": row["stale_values"],
                "pred": pred,
                "raw": raw,
                "correct": int(label["correct"]),
                "answer_type": label["answer_type"],
                "answered_write_idx": label["answered_write_idx"],
                "write_count": len(span_info["writes"]),
                "write_values": expected_spans,
                "target_write_positions": row["target_write_positions"],
                "span_token_starts": [w["token_start"] for w in span_info["writes"]],
                "span_token_ends": [w["token_end"] for w in span_info["writes"]],
                "spans_decode_ok": spans_ok,
            }
        )

        if i == 0:
            print(
                "sanity first example: "
                f"decoded_spans={decoded_spans} "
                f"attn_writes_sum={float(attn_writes.sum()):.6f}",
                flush=True,
            )
        print(f"  {i + 1}/{len(rows)} id={row.get('id')} pred={pred}", flush=True)

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    arrays = {
        "attn_writes": np.stack(attn_writes_all, axis=0),
        "attn_total": np.stack(attn_total_all, axis=0),
        "attn_bos": np.stack(attn_bos_all, axis=0),
        "max_writes": np.array(max_writes, dtype=np.int64),
        "model": np.array(args.model),
        "dtype": np.array(args.dtype),
    }
    if not args.no_hidden:
        arrays["hidden"] = np.stack(hidden_all, axis=0)
    np.savez_compressed(
        args.out,
        **arrays,
    )
    with open(args.index_out, "w") as f:
        for row in index_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "model": args.model,
        "data": args.data,
        "n": len(index_rows),
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "ids_from_index": args.ids_from_index,
        "adapter_dir": args.adapter_dir,
        "adapter_kind": args.adapter_kind,
        "adapter_mode": args.adapter_mode,
        "hidden_saved": not args.no_hidden,
        "out": args.out,
        "index_out": args.index_out,
        "accuracy": float(np.mean([r["correct"] for r in index_rows])),
    }
    summary_path = os.path.splitext(args.out)[0] + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}, {args.index_out}, {summary_path}", flush=True)


if __name__ == "__main__":
    main()
