import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=True)


def dump_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_value(response_norm: str, value: str) -> bool:
    value_norm = normalize_text(value)
    if not value_norm:
        return False
    return bool(re.search(rf"(^| )({re.escape(value_norm)})( |$)", response_norm))


def classify_stage_l_response(response: str, row: dict) -> dict:
    response_norm = normalize_text(response)
    current = row["current_value"]
    stale_values = list(row.get("stale_values", []))
    slot_values = list(row.get("slot_values", []))
    all_values = list(row.get("all_controlled_values", slot_values))

    current_hit = contains_value(response_norm, current)
    stale_hits = [value for value in stale_values if contains_value(response_norm, value)]
    same_slot_other_hits = [
        value
        for value in slot_values
        if value != current and value not in stale_values and contains_value(response_norm, value)
    ]
    cross_hits = [value for value in all_values if value not in slot_values and contains_value(response_norm, value)]

    if current_hit:
        label = "correct_current"
    elif stale_hits:
        label = "within_stale"
    elif same_slot_other_hits:
        label = "same_slot_other"
    elif cross_hits:
        label = "cross_slot"
    else:
        label = "other"
    return {
        "label": label,
        "current_hit": bool(current_hit),
        "stale_hits": stale_hits,
        "same_slot_other_hits": same_slot_other_hits,
        "cross_slot_hits": cross_hits,
    }


def resolve_dtype(name: str):
    import torch

    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model_and_tokenizer(model_name: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype),
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def generate_rows(args) -> list[dict]:
    import torch

    rows = load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    device = next(model.parameters()).device
    out_rows = []
    for i, row in enumerate(rows):
        prompt = render_prompt(tokenizer, row["messages"])
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = generated[:, enc["input_ids"].shape[1] :]
        response = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        label = classify_stage_l_response(response, row)
        out = {
            **row,
            "prompt_tokens": int(enc["input_ids"].shape[1]),
            "response": response,
            "model": args.model,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            **label,
        }
        out_rows.append(out)
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print(f"generated {i + 1}/{len(rows)}", flush=True)
    return out_rows


def score_existing(input_path: str) -> list[dict]:
    rows = load_jsonl(input_path)
    out = []
    for row in rows:
        response = row.get("response", "")
        out.append({**row, **classify_stage_l_response(response, row)})
    return out


def summarize_l0(rows: list[dict]) -> dict:
    by_dose = {}
    for dose in sorted({row["k_overwrites"] for row in rows}):
        group = [row for row in rows if row["k_overwrites"] == dose]
        counts = Counter(row["label"] for row in group)
        failures = len(group) - counts.get("correct_current", 0)
        by_dose[str(dose)] = {
            "n": len(group),
            "counts": dict(counts),
            "accuracy": counts.get("correct_current", 0) / len(group) if group else None,
            "within_stale_failure_share": counts.get("within_stale", 0) / failures if failures else None,
        }
    counts = Counter(row["label"] for row in rows)
    failures = len(rows) - counts.get("correct_current", 0)
    value_counts = defaultdict(Counter)
    for row in rows:
        value_counts[row["slot"]][row["current_value"]] += 1
    return {
        "stage": "l0",
        "n": len(rows),
        "overall_counts": dict(counts),
        "overall_accuracy": counts.get("correct_current", 0) / len(rows) if rows else None,
        "overall_within_stale_failure_share": counts.get("within_stale", 0) / failures if failures else None,
        "by_dose": by_dose,
        "slot_current_value_count_min": {
            slot: min(counter.values()) if counter else 0 for slot, counter in value_counts.items()
        },
    }


def write_l0_report(path: str | Path, summary: dict, *, data_path: str, rows_path: str) -> None:
    lines = [
        "# Stage L Report",
        "",
        "## Positioning and Expected Value",
        "",
        "Stage L fills the gap left by ICF-Bench and Stage G: realistic preference-dialogue surface, controlled repeated bindings, program-verifiable labels, identifiable mechanism probes, and in-distribution causal tests.",
        "",
        "Honest expected value: the mechanism step is likely to land (repeated bindings make the probe identifiable). The causal step is high-value but uncertain -- prior causal transfer results were weak, so both causal branches are pre-registered as publishable.",
        "",
        "## L-0 Behavior",
        "",
        f"- Data: `{data_path}`",
        f"- Rows: `{rows_path}`",
        f"- n: {summary['n']}",
        f"- Overall counts: `{summary['overall_counts']}`",
        f"- Overall accuracy: {summary['overall_accuracy']:.4f}" if summary["overall_accuracy"] is not None else "- Overall accuracy: n/a",
        f"- Overall within-stale share among failures: {summary['overall_within_stale_failure_share']:.4f}"
        if summary["overall_within_stale_failure_share"] is not None
        else "- Overall within-stale share among failures: n/a",
        "",
        "Per dose:",
        "",
    ]
    for dose, row in summary["by_dose"].items():
        acc = row["accuracy"]
        stale = row["within_stale_failure_share"]
        acc_s = f"{acc:.4f}" if acc is not None else "n/a"
        stale_s = f"{stale:.4f}" if stale is not None else "n/a"
        lines.append(
            f"- k={dose}: n={row['n']}, counts=`{row['counts']}`, "
            f"accuracy={acc_s}, within_stale_failure_share={stale_s}"
        )
    lines.extend(
        [
            "",
            "L-1 pre-registered readings:",
            "",
            "- Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result).",
            "- Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.",
            "",
            "L-2 pre-registered readings (kill-gate):",
            "",
            "- Targeted reduces within-stale errors, random control does not (CI of the difference excludes 0) → **a causal lever for stale-binding on realistic-controlled data** — the result Stage G's external transfer could not obtain (the impact result).",
            "- Targeted ≈ random, or no reduction → **clean negative**: the mechanism is readable and the representation is movable, but stale-binding is not a clean causal lever even here (consistent with Stage G external + the project's causal history). Report exactly so and STOP — do not tune the intervention post hoc to force an effect.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--data", default="data/cicm/cicm.jsonl")
    gen.add_argument("--out", default="results/cicm/l0_rows.jsonl")
    gen.add_argument("--summary-out", default="results/cicm/l0_summary.json")
    gen.add_argument("--report-out", default="results/cicm/REPORT.md")
    gen.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    gen.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    gen.add_argument("--max-new-tokens", type=int, default=16)
    gen.add_argument("--limit", type=int)

    score = sub.add_parser("score")
    score.add_argument("--input", required=True)
    score.add_argument("--out", default="results/cicm/l0_rows.rescored.jsonl")
    score.add_argument("--summary-out", default="results/cicm/l0_summary.json")
    score.add_argument("--report-out", default="results/cicm/REPORT.md")
    score.add_argument("--data", default="data/cicm/cicm.jsonl")

    args = parser.parse_args()
    if args.cmd == "generate":
        rows = generate_rows(args)
        rows_path = args.out
        data_path = args.data
    else:
        rows = score_existing(args.input)
        rows_path = args.out
        data_path = args.data
    summary = summarize_l0(rows)
    dump_jsonl(rows_path, rows)
    dump_json(args.summary_out, summary)
    write_l0_report(args.report_out, summary, data_path=data_path, rows_path=rows_path)
    print(json.dumps({"rows": rows_path, "summary": args.summary_out, "report": args.report_out}, indent=2))


if __name__ == "__main__":
    main()
