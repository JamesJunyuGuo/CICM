import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_logit_lens import first_value_token_id, should_drop_shared_first_token
from icf_generate import build_instructional_forgetting_jobs, messages_to_prompt
from icf_k1d import build_dp_messages, strip_dp_options
from icf_matchers import classify_instructional_forgetting_response, extract_forgotten_spans
from icf_mech import dump_json, dump_jsonl, load_json, token_span_for_text


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {dtype_name}")


def load_model(model_name: str, dtype_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_torch_dtype(dtype_name),
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    return model


def generate_text(model, tokenizer, enc, max_new_tokens: int) -> str:
    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = generated[:, enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()


def reduce_attention(attentions, spans: dict[str, tuple[int, int] | None]) -> dict[str, np.ndarray]:
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    out = {name: np.zeros((n_layers, n_heads), dtype=np.float32) for name in spans}
    out["total"] = np.zeros((n_layers, n_heads), dtype=np.float32)
    out["bos"] = np.zeros((n_layers, n_heads), dtype=np.float32)
    for layer_idx, layer_attn in enumerate(attentions):
        final_attn = layer_attn[0, :, -1, :].detach().float().cpu()
        out["total"][layer_idx] = final_attn.sum(dim=-1).numpy()
        out["bos"][layer_idx] = final_attn[:, 0].numpy()
        for name, span in spans.items():
            if span is None:
                continue
            start, end = span
            out[name][layer_idx] = final_attn[:, start:end].sum(dim=-1).numpy()
    return out


def final_hidden(hidden_states) -> np.ndarray:
    return np.stack(
        [state[0, -1, :].detach().to(torch.float16).cpu().numpy() for state in hidden_states],
        axis=0,
    )


def _get_norm_module(model):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise AttributeError("could not find final norm module")


def dp_new_old_logit_gap(model, tokenizer, hidden_states, row: dict) -> tuple[np.ndarray, bool]:
    new_id, _ = first_value_token_id(tokenizer, row["new_op"])
    old_id, _ = first_value_token_id(tokenizer, row["old_op"])
    if should_drop_shared_first_token(new_id, [old_id]):
        return np.full((len(hidden_states),), np.nan, dtype=np.float32), False
    norm = _get_norm_module(model)
    lm_head = model.lm_head
    gaps = []
    with torch.no_grad():
        for state in hidden_states:
            h = state[0, -1, :].unsqueeze(0)
            logits = lm_head(norm(h))
            gaps.append(float((logits[0, new_id] - logits[0, old_id]).detach().float().cpu()))
    return np.asarray(gaps, dtype=np.float32), True


def span_tuple(info: dict) -> tuple[int, int]:
    return int(info["token_start"]), int(info["token_end"])


def build_dp_prompt_and_spans(row: dict, tokenizer) -> tuple[str, dict, dict]:
    row = dict(row)
    row["question"] = strip_dp_options(row.get("question", ""))
    messages = build_dp_messages(row, freeform=True, include_old=True)
    prompt = messages_to_prompt(tokenizer, messages)
    old_info = token_span_for_text(tokenizer, prompt, row["old_preference"])
    new_info = token_span_for_text(tokenizer, prompt, row["new_preference"], start=old_info["char_end"])
    meta = {
        "old_preference_span": old_info,
        "new_preference_span": new_info,
        "span_decode_ok": bool(old_info["decoded"].strip()) and bool(new_info["decoded"].strip()),
    }
    return prompt, {"old_pref": span_tuple(old_info), "new_pref": span_tuple(new_info)}, meta


def _token_span_for_last_text(tokenizer, prompt: str, text: str) -> dict:
    pos = prompt.rfind(text)
    if pos < 0:
        return token_span_for_text(tokenizer, prompt, text)
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    from icf_mech import char_span_to_token_span

    token_start, token_end = char_span_to_token_span(enc["offset_mapping"], pos, pos + len(text))
    return {
        "char_start": pos,
        "char_end": pos + len(text),
        "token_start": token_start,
        "token_end": token_end,
        "decoded": tokenizer.decode(enc["input_ids"][token_start:token_end]),
    }


def build_if_prompt_and_spans(row: dict, tokenizer) -> tuple[str, dict, dict]:
    job = build_instructional_forgetting_jobs([row])[0]
    prompt = messages_to_prompt(tokenizer, job["messages"])
    forget_info = _token_span_for_last_text(tokenizer, prompt, row.get("forget_instruction", ""))
    value_infos = []
    value_span = None
    for span_text in extract_forgotten_spans(row):
        try:
            info = token_span_for_text(tokenizer, prompt[: forget_info["char_start"]], span_text)
        except Exception:
            continue
        value_infos.append({**info, "text": span_text})
    if value_infos:
        value_span = (
            min(info["token_start"] for info in value_infos),
            max(info["token_end"] for info in value_infos),
        )
    meta = {
        "forget_instruction_span": forget_info,
        "forget_value_spans": value_infos,
        "value_span_found": bool(value_infos),
    }
    return prompt, {"forget_instruction": span_tuple(forget_info), "forget_value": value_span}, meta


def build_rows(scenario: str, input_path: str, limit: int | None) -> list[dict]:
    rows = load_json(input_path)
    if scenario == "instructional_forgetting":
        kept = []
        for row in rows:
            spans = extract_forgotten_spans(row)
            if 1 <= len(spans) <= 20 and all(len(span) <= 80 for span in spans):
                kept.append(row)
        rows = kept
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["dynamic_preference", "instructional_forgetting"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        padding_side="left",
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model, args.dtype)
    device = next(model.parameters()).device
    rows = build_rows(args.scenario, args.input, args.limit)
    print(f"loaded {len(rows)} {args.scenario} rows", flush=True)

    hidden = []
    dp_logit_gaps = []
    dp_logit_gap_valid = []
    attn_arrays = defaultdict(list)  # type: ignore[name-defined]
    index_rows = []
    response_rows = []

    for idx, row in enumerate(rows):
        try:
            if args.scenario == "dynamic_preference":
                prompt, spans, meta = build_dp_prompt_and_spans(row, tokenizer)
            else:
                prompt, spans, meta = build_if_prompt_and_spans(row, tokenizer)
        except Exception as exc:
            print(f"span build failed id={row.get('id')}: {exc}", flush=True)
            continue

        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**enc, output_attentions=True, output_hidden_states=True, use_cache=False)
        raw = generate_text(model, tokenizer, enc, args.max_new_tokens)
        reduced = reduce_attention(outputs.attentions, spans)

        for key, arr in reduced.items():
            attn_arrays[key].append(arr)
        hidden.append(final_hidden(outputs.hidden_states))
        if args.scenario == "dynamic_preference":
            gap, valid = dp_new_old_logit_gap(model, tokenizer, outputs.hidden_states, row)
            dp_logit_gaps.append(gap)
            dp_logit_gap_valid.append(valid)
        prompt_tokens = int(enc["input_ids"].shape[1])

        index_row = {
            "row_index": len(index_rows),
            "id": str(row.get("id")),
            "scenario": args.scenario,
            "prompt_tokens": prompt_tokens,
            "raw": raw,
            "dp_logit_gap_valid": dp_logit_gap_valid[-1] if args.scenario == "dynamic_preference" else None,
            **meta,
        }
        if args.scenario == "dynamic_preference":
            response_row = {
                **row,
                "bare_question": strip_dp_options(row.get("question", "")),
                "llm_response_exist_old": raw,
                "model": args.model,
                "format": "freeform_local_mech",
            }
        else:
            error_type = classify_instructional_forgetting_response(raw, row)
            response_row = {
                **row,
                "instruction_forget_reply": raw,
                "forget_response": raw,
                "error_type": error_type,
                "final_error_type": error_type,
                "forget_correct": error_type == "correct_forget",
                "model": args.model,
            }
            index_row["final_error_type"] = error_type
        index_rows.append(index_row)
        response_rows.append(response_row)

        print(f"  {idx + 1}/{len(rows)} id={row.get('id')} tokens={prompt_tokens} label={response_row.get('final_error_type', 'pending')}", flush=True)
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    arrays = {
        "hidden": np.stack(hidden, axis=0),
        "model": np.array(args.model),
        "dtype": np.array(args.dtype),
    }
    for key, vals in attn_arrays.items():
        arrays[f"attn_{key}"] = np.stack(vals, axis=0)
    if args.scenario == "dynamic_preference":
        arrays["new_old_logit_gap"] = np.stack(dp_logit_gaps, axis=0)
        arrays["new_old_logit_gap_valid"] = np.asarray(dp_logit_gap_valid, dtype=np.bool_)
    npz_path = out_dir / f"{args.scenario}.npz"
    np.savez_compressed(npz_path, **arrays)
    dump_jsonl(out_dir / f"{args.scenario}_index.jsonl", index_rows)
    if args.scenario == "dynamic_preference":
        dump_json(out_dir / "dynamic_preference_local_responses.json", response_rows)
    else:
        dump_json(out_dir / "instructional_forgetting_local_responses.json", response_rows)
    summary = {
        "scenario": args.scenario,
        "model": args.model,
        "dtype": args.dtype,
        "n": len(index_rows),
        "max_new_tokens": args.max_new_tokens,
        "npz": str(npz_path),
        "index": str(out_dir / f"{args.scenario}_index.jsonl"),
        "responses": str(out_dir / f"{args.scenario}_local_responses.json"),
        "value_span_found": sum(1 for row in index_rows if row.get("value_span_found", True)),
    }
    dump_json(out_dir / f"{args.scenario}_extract_summary.json", summary)
    print(f"wrote {npz_path}", flush=True)


if __name__ == "__main__":
    main()
