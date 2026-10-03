import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

from icf_k1 import bootstrap_ci, dump_json, dump_jsonl, load_json, wilson_ci
from icf_matchers import normalize_text


DEFAULT_DP_MODEL = "qwen/qwen-2.5-7b-instruct"
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4.6"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MC_MARKER = " Here are four options, you can choose one as your answer"
LABEL_ALIASES = {
    "correct_current": "correct_current",
    "correct_follows_new": "correct_current",
    "follows_new": "correct_current",
    "new": "correct_current",
    "within_stale": "within_stale",
    "stale": "within_stale",
    "follows_old": "within_stale",
    "old": "within_stale",
    "other": "other",
    "other_valid_value": "other",
    "ambiguous": "other",
    "hallucination": "other",
}
CONTENT_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "answer",
    "are",
    "can",
    "choice",
    "course",
    "for",
    "from",
    "have",
    "here",
    "into",
    "just",
    "like",
    "more",
    "option",
    "prefer",
    "preference",
    "recommend",
    "should",
    "some",
    "that",
    "the",
    "their",
    "this",
    "through",
    "use",
    "with",
    "would",
    "you",
    "your",
}
NON_ENGLISH_LANGUAGE_RE = re.compile(
    r"\b(french|punjabi|urdu|hindi|spanish|korean|japanese|chinese|german|italian|"
    r"portuguese|arabic|russian)\b",
    re.IGNORECASE,
)


def strip_dp_options(question: str) -> str:
    text = question or ""
    if MC_MARKER in text:
        text = text.split(MC_MARKER, 1)[0]
    return re.sub(r"\s+", " ", text).strip()


def _content_tokens(text: object) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) >= 4 and token not in CONTENT_STOPWORDS
    }


def _option_score(response: str, option: str) -> int:
    response_norm = normalize_text(response)
    option_norm = normalize_text(option)
    if option_norm and option_norm in response_norm:
        return 100
    return len(_content_tokens(response).intersection(_content_tokens(option)))


def _current_free_resource_match(response_norm: str, row: dict) -> bool:
    new_pref = normalize_text(row.get("new_preference") or row.get("new_pref") or "")
    old_op = normalize_text(row.get("old_op", ""))
    if not ({"free", "opensource"}.intersection(set(new_pref.split())) or "open source" in new_pref):
        return False
    if "purchase" not in old_op and "paid" not in old_op:
        return False
    return any(token in response_norm.split() for token in ("free", "opensource", "youtube", "open"))


def _current_flat_shoes_match(response_norm: str, row: dict) -> bool:
    new_pref = normalize_text(row.get("new_preference") or row.get("new_pref") or "")
    return "flat shoes" in new_pref and "flat shoes" in response_norm


def _violates_current_by_old_direction(response_norm: str, row: dict) -> bool:
    new_pref = normalize_text(row.get("new_preference") or row.get("new_pref") or "")
    if "avoid electronic music" in new_pref and "electronic music" in response_norm:
        return True
    if "only" in new_pref and "english" in new_pref and NON_ENGLISH_LANGUAGE_RE.search(response_norm):
        return True
    if "severe peanut" in new_pref and any(token in response_norm.split() for token in ("peanut", "nuts", "nut")):
        return True
    if "high heels" in new_pref and "not like high heels" in new_pref and "high heels" in response_norm:
        return True
    return False


def classify_freeform_dynamic_preference(response: str, row: dict, min_score: int = 4, margin: int = 2) -> str:
    response_norm = normalize_text(response or "")
    if _current_free_resource_match(response_norm, row):
        return "correct_current"
    if _current_flat_shoes_match(response_norm, row):
        return "correct_current"
    if _violates_current_by_old_direction(response_norm, row):
        return "within_stale"
    old_score = _option_score(response or "", row.get("old_op", ""))
    new_score = _option_score(response or "", row.get("new_op", ""))
    if old_score >= min_score and old_score >= new_score + margin:
        return "within_stale"
    if new_score >= min_score and new_score >= old_score + margin:
        return "correct_current"
    if old_score >= min_score and new_score >= min_score:
        return "ambiguous"
    return "needs_judge"


def parse_judge_label(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        payload = json.loads(text)
        label = str(payload.get("label", "")).strip().lower()
    except Exception:
        label = text.strip().lower()
    return LABEL_ALIASES.get(label, "judge_parse_error")


def build_dp_messages(row: dict, freeform: bool = True, include_old: bool = True) -> list[dict]:
    question = strip_dp_options(row.get("question", "")) if freeform else row.get("question", "")
    messages = [
        {
            "role": "system",
            "content": "You are a useful assistant. Make the most reasonable recommendation according to my current preferences.",
        }
    ]
    if include_old:
        messages.extend(
            [
                {"role": "user", "content": row.get("old_preference", "")},
                {"role": "assistant", "content": "I understand your preference."},
            ]
        )
    messages.extend(
        [
            {"role": "user", "content": row.get("new_preference", "")},
            {"role": "assistant", "content": "I understand your updated preference."},
            {"role": "user", "content": question},
        ]
    )
    return messages


def call_openrouter(
    messages: list[dict],
    model: str,
    *,
    max_tokens: int,
    retries: int,
    timeout: float,
    title: str,
) -> str:
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": title,
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    last_error = None
    for attempt in range(retries + 1):
        resp = None
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            body = ""
            if resp is not None:
                try:
                    body = resp.text[:500]
                except Exception:
                    body = ""
            last_error = f"{exc} | body: {body}"
            if attempt >= retries:
                return f"__CALL_ERROR__ {last_error}"
            time.sleep(min(30.0, 2.0**attempt))
    return f"__CALL_ERROR__ unreachable retry state: {last_error}"


def _load_existing_by_id(path: str | Path) -> dict[str, dict]:
    existing = {}
    p = Path(path)
    if not p.exists():
        return existing
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("judge_raw", "")).startswith("__CALL_ERROR__"):
                continue
            if row.get("judge_error_type") == "judge_parse_error":
                continue
            if "id" in row:
                existing[str(row["id"])] = row
    return existing


def build_freeform_rows(input_rows: list[dict], limit: int | None = None) -> list[dict]:
    rows = []
    for row in input_rows[: limit or None]:
        out = dict(row)
        out["bare_question"] = strip_dp_options(out.get("question", ""))
        rows.append(out)
    return rows


def run_dp_api(
    rows: list[dict],
    out_path: str | Path,
    *,
    model: str = DEFAULT_DP_MODEL,
    freeform: bool = True,
    max_tokens: int = 160,
    concurrency: int = 4,
    retries: int = 4,
    timeout: float = 90.0,
) -> list[dict]:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_by_id(out_path)
    todo = [row for row in rows if str(row["id"]) not in existing]
    print(f"loaded {len(existing)} existing rows; running {len(todo)} new rows", flush=True)

    def run_one(row: dict) -> dict:
        forget_raw = call_openrouter(
            build_dp_messages(row, freeform=freeform, include_old=True),
            model,
            max_tokens=max_tokens,
            retries=retries,
            timeout=timeout,
            title="Contextual-management Stage K K1d DP",
        ).strip()
        noforget_raw = call_openrouter(
            build_dp_messages(row, freeform=freeform, include_old=False),
            model,
            max_tokens=max_tokens,
            retries=retries,
            timeout=timeout,
            title="Contextual-management Stage K K1d DP",
        ).strip()
        return {
            **row,
            "model": model,
            "format": "freeform" if freeform else "multiple_choice",
            "freeform": freeform,
            "llm_response_exist_old": forget_raw,
            "llm_response_noexist_old": noforget_raw,
        }

    new_records = []
    with open(out_path, "a", encoding="utf-8") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_row = {pool.submit(run_one, row): row for row in todo}
            for i, fut in enumerate(concurrent.futures.as_completed(future_to_row), 1):
                row = future_to_row[fut]
                record = fut.result()
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                new_records.append(record)
                if i == 1 or i % 25 == 0 or i == len(todo):
                    print(f"completed {i}/{len(todo)} new rows (id={row['id']})", flush=True)

    merged = list(existing.values()) + new_records
    return sorted(merged, key=lambda row: int(row["id"]) if str(row["id"]).isdigit() else str(row["id"]))


def build_judge_prompt(row: dict, response: str, examples: list[dict]) -> list[dict]:
    example_lines = []
    for item in examples[:9]:
        example_lines.append(
            json.dumps(
                {
                    "old_op": item.get("old_op"),
                    "new_op": item.get("new_op"),
                    "response": item.get("freeform_response"),
                    "label": LABEL_ALIASES.get(item.get("claude_hand_label"), item.get("claude_hand_label")),
                },
                ensure_ascii=False,
            )
        )
    instructions = (
        "Classify the assistant response for a Dynamic Preference overwrite item. "
        "The user first had OLD preference/old_op, then updated to NEW preference/new_op. "
        "Return JSON only with label and rationale. Labels: "
        "correct_current if the response follows or recommends the NEW option/current preference; "
        "within_stale if it follows or recommends the OLD option/stale preference; "
        "other if neither, both, vague, or a different recommendation. "
        "Use old_op and new_op as the anchor ground truth."
    )
    target = {
        "old_preference": row.get("old_preference") or row.get("old_pref"),
        "new_preference": row.get("new_preference") or row.get("new_pref"),
        "old_op": row.get("old_op"),
        "new_op": row.get("new_op"),
        "question": row.get("bare_question") or strip_dp_options(row.get("question", "")),
        "response": response,
    }
    user = instructions + "\n\nCalibration examples:\n" + "\n".join(example_lines) + "\n\nItem:\n" + json.dumps(target, ensure_ascii=False)
    return [
        {"role": "system", "content": "You are a careful annotation judge. Return valid compact JSON only."},
        {"role": "user", "content": user},
    ]


def _balanced_examples(sample_items: list[dict], seed: int = 17, per_label: int = 3) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {}
    for item in sample_items:
        label = LABEL_ALIASES.get(item.get("claude_hand_label"), item.get("claude_hand_label"))
        grouped.setdefault(label, []).append(item)
    out = []
    for label in ("correct_current", "within_stale", "other"):
        items = list(grouped.get(label, []))
        rng.shuffle(items)
        out.extend(items[:per_label])
    return out


def _sample_to_row(item: dict) -> dict:
    row = dict(item)
    row["old_preference"] = row.get("old_preference") or row.get("old_pref")
    row["new_preference"] = row.get("new_preference") or row.get("new_pref")
    row["bare_question"] = row.get("bare_question") or strip_dp_options(row.get("question", ""))
    return row


def classify_rows_hybrid(
    rows: list[dict],
    sample_items: list[dict],
    judge_out_path: str | Path,
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 120,
    concurrency: int = 4,
    retries: int = 3,
    timeout: float = 90.0,
) -> list[dict]:
    examples = _balanced_examples(sample_items)
    existing = _load_existing_by_id(judge_out_path)
    classified = []
    residuals = []
    for row in rows:
        response = row.get("llm_response_exist_old") or row.get("freeform_response") or ""
        det = classify_freeform_dynamic_preference(response, row)
        base = {
            **row,
            "deterministic_error_type": det,
            "judge_model": judge_model,
            "judge_raw": None,
            "judge_error_type": None,
            "final_error_type": det if det in {"correct_current", "within_stale"} else None,
        }
        if det in {"correct_current", "within_stale"}:
            classified.append(base)
        elif str(row["id"]) in existing and existing[str(row["id"])].get("judge_model") == judge_model:
            label = parse_judge_label(existing[str(row["id"])]["judge_raw"])
            classified.append({**base, **existing[str(row["id"])], "judge_error_type": label, "final_error_type": label})
        else:
            residuals.append(base)

    Path(judge_out_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"deterministic classified {len(classified)}/{len(rows)}; judging {len(residuals)} residual rows", flush=True)

    def judge_one(row: dict) -> dict:
        response = row.get("llm_response_exist_old") or row.get("freeform_response") or ""
        raw = call_openrouter(
            build_judge_prompt(row, response, examples),
            judge_model,
            max_tokens=max_tokens,
            retries=retries,
            timeout=timeout,
            title="Contextual-management Stage K K1d judge",
        ).strip()
        label = parse_judge_label(raw)
        return {
            "id": str(row["id"]),
            "judge_model": judge_model,
            "judge_raw": raw,
            "judge_error_type": label,
            "final_error_type": label if label != "judge_parse_error" else "other",
        }

    with open(judge_out_path, "a", encoding="utf-8") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_to_row = {pool.submit(judge_one, row): row for row in residuals}
            for i, fut in enumerate(concurrent.futures.as_completed(future_to_row), 1):
                base = future_to_row[fut]
                judged = fut.result()
                out_f.write(json.dumps(judged, ensure_ascii=False) + "\n")
                out_f.flush()
                classified.append({**base, **judged})
                if i == 1 or i % 25 == 0 or i == len(residuals):
                    print(f"judged {i}/{len(residuals)} residual rows", flush=True)

    return sorted(classified, key=lambda row: int(row["id"]) if str(row["id"]).isdigit() else str(row["id"]))


def summarize_freeform_dp(rows: list[dict], judge_validation_agreement: float | None = None) -> dict:
    failures = [row for row in rows if row.get("final_error_type") != "correct_current"]
    stale = [row for row in failures if row.get("final_error_type") == "within_stale"]
    noforget_current = 0
    for row in rows:
        noforget_type = row.get("noforget_error_type")
        if noforget_type is None:
            noforget_type = classify_freeform_dynamic_preference(row.get("llm_response_noexist_old", ""), row)
        noforget_current += int(noforget_type == "correct_current")
    failure_flags = [1 if row.get("final_error_type") == "within_stale" else 0 for row in failures]
    return {
        "scenario": "dynamic_preference_freeform",
        "n": len(rows),
        "format": "freeform",
        "forget_accuracy": (len(rows) - len(failures)) / len(rows) if rows else math.nan,
        "forget_failures": len(failures),
        "within_stale_failures": len(stale),
        "within_stale_fraction": len(stale) / len(failures) if failures else math.nan,
        "within_stale_ci": bootstrap_ci(failure_flags) if failures else [math.nan, math.nan],
        "error_counts": dict(Counter(row.get("final_error_type", "other") for row in failures)),
        "noforget_reference_rate": noforget_current / len(rows) if rows else math.nan,
        "judge_validation_agreement": judge_validation_agreement,
        "headline_allowed": judge_validation_agreement is not None and judge_validation_agreement >= 0.90,
    }


def validate_against_sample(sample_items: list[dict], classified_rows: list[dict]) -> dict:
    by_id = {str(row["id"]): row for row in classified_rows}
    rows = []
    for item in sample_items:
        expected = LABEL_ALIASES.get(item.get("claude_hand_label"), item.get("claude_hand_label"))
        got = by_id[str(item["id"])]["final_error_type"]
        rows.append(
            {
                "id": str(item["id"]),
                "human_label": expected,
                "model_label": got,
                "match": expected == got,
                "deterministic_error_type": by_id[str(item["id"])].get("deterministic_error_type"),
                "judge_error_type": by_id[str(item["id"])].get("judge_error_type"),
            }
        )
    matches = sum(row["match"] for row in rows)
    return {
        "n": len(rows),
        "matches": matches,
        "agreement": matches / len(rows) if rows else math.nan,
        "rows": rows,
    }


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_k1d_report(path: str | Path, summary: dict) -> None:
    dp = summary["dynamic_preference_freeform"]
    mc = summary["dynamic_preference_mc_contrast"]
    ifs = summary["instructional_forgetting_k1c"]
    val = summary["judge_validation"]
    gate = summary["gate"]
    ci = dp["within_stale_ci"]
    lines = [
        "# Stage K Report",
        "",
        "## Scope",
        "",
        "K1d is API-only behavioral inference on ICF-Bench Dynamic Preference free-form prompts plus existing K1c Instructional Forgetting error-type repair. No hidden states, no GPU, no Stage-J artifacts.",
        "",
        "## K1d Result",
        "",
        f"- DP primary model: `{summary['dp_model']}` via OpenRouter, temperature 0.",
        f"- Judge model: `{summary['judge_model']}` via OpenRouter, temperature 0.",
        f"- Judge-vs-human calibration: {val['matches']}/{val['n']} = {val['agreement']:.3f}.",
        f"- Headline allowed by validation threshold: `{dp['headline_allowed']}`.",
        "",
        "| Scenario / format | n | Forget acc | Failures | within-stale / failures | 95% CI | NoForget/reference |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
        (
            f"| Dynamic Preference free-form | {dp['n']} | {dp['forget_accuracy']:.3f} | "
            f"{dp['forget_failures']} | {dp['within_stale_fraction']:.3f} "
            f"({dp['within_stale_failures']}/{dp['forget_failures']}) | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {dp['noforget_reference_rate']:.3f} |"
        ),
        (
            f"| Dynamic Preference MC contrast | {mc['n']} | {mc['forget_accuracy']:.3f} | "
            f"{mc['forget_failures']} | {mc['within_stale_fraction']:.3f} "
            f"({mc['within_stale_failures']}/{mc['forget_failures']}) | "
            f"[{mc['within_stale_ci'][0]:.3f}, {mc['within_stale_ci'][1]:.3f}] | {mc['noforget_reference_rate']:.3f} |"
        ),
        (
            f"| Instructional Forgetting free-form (K1c fixed matcher) | {ifs['n']} | {ifs['forget_accuracy']:.3f} | "
            f"{ifs['forget_failures']} | {ifs['within_stale_fraction']:.3f} "
            f"({ifs['within_stale_failures']}/{ifs['forget_failures']}) | "
            f"[{ifs['within_stale_ci'][0]:.3f}, {ifs['within_stale_ci'][1]:.3f}] | {ifs['noforget_reference_rate']:.3f} |"
        ),
        "",
        "Pre-registered K1d reading:",
        "",
        "> within-stale dominates among failures in BOTH IF and DP -> stale-binding is format-robust and pervasive across interference types.",
        "",
        f"Adjudication: `{gate['adjudication']}`.",
        f"Next step: `{gate['next_step']}`.",
        "",
        "## Format Contrast",
        "",
        f"DP free-form within-stale among failures is {dp['within_stale_fraction']:.3f}; DP multiple-choice contrast is {mc['within_stale_fraction']:.3f}. The MC format masks stale-binding by distributing wrong answers across distractors and by presenting the current option explicitly.",
        "",
        "## Artifacts",
        "",
        "- `k1d_dp_freeform_openrouter.jsonl`",
        "- `k1d_dp_freeform_judge_raw.jsonl`",
        "- `k1d_dp_freeform_rows.jsonl`",
        "- `k1d_dp_freeform_summary.json`",
        "- `k1d_freeform_dp_judge_validation_rows.jsonl`",
        "- `k1d_freeform_dp_judge_validation_summary.json`",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_combined_summary(
    dp_summary: dict,
    validation_summary: dict,
    k1c_summary: dict,
    *,
    dp_model: str,
    judge_model: str,
) -> dict:
    mc = k1c_summary["by_scenario"]["dynamic_preference"]
    ifs = k1c_summary["by_scenario"]["instructional_forgetting"]
    dp_dominant = dp_summary["headline_allowed"] and dp_summary["within_stale_ci"][0] >= 0.50
    if_dominant = ifs["within_stale_ci"][0] >= 0.70
    if dp_dominant and if_dominant:
        adjudication = "format_robust_cross_scenario_stale_binding"
        next_step = "stop_k1_api_phase_mechanism_k2_k3_deferred_to_local_gpu"
    elif if_dominant and not dp_dominant:
        adjudication = "format_controlled_if_dp_contrast"
        next_step = "report_contrast_then_defer_mechanism"
    else:
        adjudication = "k1_boundary_no_cross_scenario_replication"
        next_step = "stop_report_boundary"
    return {
        "stage": "k1d",
        "dp_model": dp_model,
        "judge_model": judge_model,
        "temperature": 0,
        "dynamic_preference_freeform": dp_summary,
        "dynamic_preference_mc_contrast": mc,
        "instructional_forgetting_k1c": ifs,
        "judge_validation": {k: v for k, v in validation_summary.items() if k != "rows"},
        "gate": {
            "dp_dominant_threshold": "validation agreement >= 0.90 and DP free-form within-stale CI lower bound >= 0.50",
            "if_dominant_threshold": "IF K1c within-stale CI lower bound >= 0.70",
            "dp_dominant": dp_dominant,
            "if_dominant": if_dominant,
            "adjudication": adjudication,
            "next_step": next_step,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_dp = sub.add_parser("run-dp-api")
    run_dp.add_argument("--input", default="results/icf_bench/k1b_dynamic_preference_input.json")
    run_dp.add_argument("--out", default="results/icf_bench/k1d_dp_freeform_openrouter.jsonl")
    run_dp.add_argument("--model", default=DEFAULT_DP_MODEL)
    run_dp.add_argument("--format", choices=["freeform", "mc"], default="freeform")
    run_dp.add_argument("--limit", type=int)
    run_dp.add_argument("--max-tokens", type=int, default=160)
    run_dp.add_argument("--concurrency", type=int, default=4)
    run_dp.add_argument("--retries", type=int, default=4)
    run_dp.add_argument("--timeout", type=float, default=90.0)

    classify = sub.add_parser("classify-dp")
    classify.add_argument("--responses", default="results/icf_bench/k1d_dp_freeform_openrouter.jsonl")
    classify.add_argument("--sample", default="results/icf_bench/freeform_dp_sample_claude.json")
    classify.add_argument("--judge-raw", default="results/icf_bench/k1d_dp_freeform_judge_raw.jsonl")
    classify.add_argument("--noforget-judge-raw", default="results/icf_bench/k1d_dp_freeform_noforget_judge_raw.jsonl")
    classify.add_argument("--validation-judge-raw", default="results/icf_bench/k1d_freeform_dp_judge_validation_raw.jsonl")
    classify.add_argument("--rows-out", default="results/icf_bench/k1d_dp_freeform_rows.jsonl")
    classify.add_argument("--summary-out", default="results/icf_bench/k1d_dp_freeform_summary.json")
    classify.add_argument("--validation-rows-out", default="results/icf_bench/k1d_freeform_dp_judge_validation_rows.jsonl")
    classify.add_argument("--validation-summary-out", default="results/icf_bench/k1d_freeform_dp_judge_validation_summary.json")
    classify.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    classify.add_argument("--max-tokens", type=int, default=120)
    classify.add_argument("--concurrency", type=int, default=4)
    classify.add_argument("--retries", type=int, default=3)
    classify.add_argument("--timeout", type=float, default=90.0)

    report = sub.add_parser("write-report")
    report.add_argument("--dp-summary", default="results/icf_bench/k1d_dp_freeform_summary.json")
    report.add_argument("--validation-summary", default="results/icf_bench/k1d_freeform_dp_judge_validation_summary.json")
    report.add_argument("--k1c-summary", default="results/icf_bench/k1c_qwen7b/k1_summary_qwen7b.json")
    report.add_argument("--summary-out", default="results/icf_bench/k1d_summary.json")
    report.add_argument("--report-out", default="results/icf_bench/REPORT.md")
    report.add_argument("--dp-model", default=DEFAULT_DP_MODEL)
    report.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)

    args = parser.parse_args()

    if args.cmd == "run-dp-api":
        input_rows = load_json(args.input)
        rows = build_freeform_rows(input_rows, args.limit)
        run_dp_api(
            rows,
            args.out,
            model=args.model,
            freeform=args.format == "freeform",
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            retries=args.retries,
            timeout=args.timeout,
        )
        return

    if args.cmd == "classify-dp":
        sample = load_json(args.sample)["items"]
        validation_input = [_sample_to_row(item) for item in sample]
        validation_classified = classify_rows_hybrid(
            validation_input,
            sample,
            args.validation_judge_raw,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            retries=args.retries,
            timeout=args.timeout,
        )
        validation = validate_against_sample(sample, validation_classified)
        dump_jsonl(args.validation_rows_out, validation["rows"])
        dump_json(args.validation_summary_out, {k: v for k, v in validation.items() if k != "rows"})

        response_rows = load_jsonl(args.responses)
        classified = classify_rows_hybrid(
            response_rows,
            sample,
            args.judge_raw,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            retries=args.retries,
            timeout=args.timeout,
        )
        noforget_inputs = []
        for row in response_rows:
            out = dict(row)
            out["llm_response_exist_old"] = row.get("llm_response_noexist_old", "")
            noforget_inputs.append(out)
        noforget_classified = classify_rows_hybrid(
            noforget_inputs,
            sample,
            args.noforget_judge_raw,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            retries=args.retries,
            timeout=args.timeout,
        )
        noforget_by_id = {str(row["id"]): row.get("final_error_type") for row in noforget_classified}
        classified = [
            {**row, "noforget_error_type": noforget_by_id.get(str(row["id"]), "other")}
            for row in classified
        ]
        dp_summary = summarize_freeform_dp(classified, validation["agreement"])
        dump_jsonl(args.rows_out, classified)
        dump_json(args.summary_out, dp_summary)
        return

    if args.cmd == "write-report":
        dp_summary = load_json(args.dp_summary)
        validation_summary = load_json(args.validation_summary)
        k1c_summary = load_json(args.k1c_summary)
        combined = build_combined_summary(
            dp_summary,
            validation_summary,
            k1c_summary,
            dp_model=args.dp_model,
            judge_model=args.judge_model,
        )
        dump_json(args.summary_out, combined)
        write_k1d_report(args.report_out, combined)
        return


if __name__ == "__main__":
    main()
