import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from icf_matchers import (
    classify_dynamic_preference_response,
    classify_instructional_forgetting_response,
    extract_forgotten_spans,
)


OPTION_RE = re.compile(r"(?ms)^\s*([A-D])\.\s*(.*?)(?=^\s*[A-D]\.\s*|\Z)")


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_options_from_question(question: str) -> list[str]:
    options = []
    for _letter, text in OPTION_RE.findall(question or ""):
        options.append(re.sub(r"\s+", " ", text).strip())
    return options


def wilson_ci(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def bootstrap_ci(values: list[int], n_boot: int = 2000, seed: int = 1009) -> list[float]:
    if not values:
        return [math.nan, math.nan]
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return [lo, hi]


def classify_dynamic_preference_file(path: str | Path, model: str) -> list[dict]:
    rows = []
    for item in load_json(path):
        row = dict(item)
        row["options"] = parse_options_from_question(row.get("question", ""))
        error_type = classify_dynamic_preference_response(row.get("llm_response_exist_old", ""), row)
        noforget_type = classify_dynamic_preference_response(row.get("llm_response_noexist_old", ""), row)
        rows.append(
            {
                "scenario": "dynamic_preference",
                "id": str(row.get("id")),
                "model": model,
                "forget_response": row.get("llm_response_exist_old", ""),
                "noforget_response": row.get("llm_response_noexist_old", ""),
                "forget_correct": error_type == "correct_current",
                "noforget_recovers_reference": noforget_type in {"correct_current", "within_stale"},
                "error_type": error_type,
                "noforget_type": noforget_type,
                "old_value": row.get("old_op", ""),
                "current_value": row.get("new_op", ""),
                "n_options": len(row["options"]),
                "kept": len(row["options"]) == 4 and bool(row.get("old_op")) and bool(row.get("new_op")),
                "drop_reason": None if len(row["options"]) == 4 else "options_not_parseable",
            }
        )
    return rows


def classify_instructional_forgetting_file(path: str | Path, model: str) -> list[dict]:
    rows = []
    for item in load_json(path):
        spans = extract_forgotten_spans(item)
        error_type = classify_instructional_forgetting_response(item.get("instruction_forget_reply", ""), item)
        noforget_type = classify_instructional_forgetting_response(item.get("instruction_noforget_reply", ""), item)
        kept = 1 <= len(spans) <= 20 and all(len(span) <= 80 for span in spans)
        rows.append(
            {
                "scenario": "instructional_forgetting",
                "id": str(item.get("id")),
                "model": model,
                "forget_response": item.get("instruction_forget_reply", ""),
                "noforget_response": item.get("instruction_noforget_reply", ""),
                "forget_correct": error_type == "correct_forget",
                "noforget_recovers_reference": noforget_type == "within_stale",
                "error_type": error_type,
                "noforget_type": noforget_type,
                "old_value": spans,
                "current_value": "FORGET/REFUSE",
                "kept": kept,
                "drop_reason": None if kept else "no_short_extractable_span",
            }
        )
    return rows


def summarize_classifications(rows: list[dict], ci_method: str = "bootstrap") -> dict:
    kept_rows = [row for row in rows if row.get("kept", True)]
    by_scenario = {}
    qualifying = []
    for scenario in sorted({row["scenario"] for row in kept_rows}):
        scenario_rows = [row for row in kept_rows if row["scenario"] == scenario]
        failures = [row for row in scenario_rows if not row.get("forget_correct", False)]
        stale = [row for row in failures if row.get("error_type") == "within_stale"]
        counts = Counter(row.get("error_type", "other") for row in failures)
        if ci_method == "wilson":
            ci = wilson_ci(len(stale), len(failures))
        else:
            ci = bootstrap_ci([1 if row.get("error_type") == "within_stale" else 0 for row in failures])
        fraction = len(stale) / len(failures) if failures else math.nan
        by_scenario[scenario] = {
            "n": len(scenario_rows),
            "forget_failures": len(failures),
            "within_stale_failures": len(stale),
            "within_stale_fraction": fraction,
            "within_stale_ci": ci,
            "error_counts": dict(counts),
            "forget_accuracy": sum(1 for row in scenario_rows if row.get("forget_correct", False))
            / len(scenario_rows)
            if scenario_rows
            else math.nan,
            "noforget_reference_rate": sum(1 for row in scenario_rows if row.get("noforget_recovers_reference"))
            / len(scenario_rows)
            if scenario_rows
            else math.nan,
        }
        if len(failures) > 0 and ci[0] >= 0.70:
            qualifying.append(scenario)

    drop_counts = defaultdict(Counter)
    for row in rows:
        if not row.get("kept", True):
            drop_counts[row["scenario"]][row.get("drop_reason") or "dropped"] += 1

    two_scenario_pass = len(qualifying) >= 2
    dissociation = None
    proceed = two_scenario_pass
    next_step = "stop_report_boundary"
    if two_scenario_pass:
        next_step = "proceed_to_k2_measure_mechanism"
    elif (
        "instructional_forgetting" in qualifying
        and "dynamic_preference" in by_scenario
        and "dynamic_preference" not in qualifying
    ):
        dissociation = "if_stale_dp_not_dominant"
        proceed = True
        next_step = "proceed_to_k2_k3_explain_if_dp_contrast"

    return {
        "stage": "k1",
        "by_scenario": by_scenario,
        "drop_counts": {scenario: dict(counts) for scenario, counts in drop_counts.items()},
        "gate": {
            "criterion": "within-stale dominates (pooled CI lower bound >= 0.70) in >=2 program-verifiable scenarios",
            "qualifying_scenarios": qualifying,
            "pass": two_scenario_pass,
            "dissociation": dissociation,
            "proceed": proceed,
            "next_step": next_step,
            "reading_if_pass": "within-stale dominates (pooled CI lower bound >= 0.70) in >=2 program-verifiable scenarios -> the stale-binding error signature replicates on independent real dialogue; proceed to K2.",
            "reading_if_dissociation": "IF stale dominates but clean DP does not -> real IF-vs-DP dissociation; proceed to K2/K3 to explain the contrast.",
            "reading_if_fail": "Otherwise -> report the boundary honestly (the synthetic signature is partly setup-specific / real-world failure is more heterogeneous).",
        },
    }


def write_report(path: str | Path, summary: dict, model: str, source: str) -> None:
    lines = [
        "# Stage K Report",
        "",
        "## Scope",
        "",
        "Stage K tests stale-binding as a real-deployment phenomenon on ICF-Bench. This report is isolated from Stage J and uses only `results/icf_bench/` artifacts.",
        "",
        "## K1 Behavioral Error Signature",
        "",
        f"- Model/source: `{model}` from `{source}`",
        "- Headline: within-stale fraction among Forget-form failures, not accuracy drop.",
        "- Program-verifiable scenarios: Dynamic Preference and Instructional Forgetting.",
        "- LLM-judge labels are not used for the headline.",
        "",
        "Pre-registered reading:",
        "",
        "> within-stale dominates (pooled CI lower bound >= 0.70) in >=2 program-verifiable scenarios -> the stale-binding error signature replicates on independent real dialogue; proceed to K2. Otherwise -> report the boundary honestly (the synthetic signature is partly setup-specific / real-world failure is more heterogeneous) and STOP.",
        "",
        "| Scenario | n kept | Forget acc | Forget failures | within-stale / failures | 95% CI | no-forget reference rate |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for scenario, vals in summary["by_scenario"].items():
        ci = vals["within_stale_ci"]
        lines.append(
            f"| {scenario} | {vals['n']} | {vals['forget_accuracy']:.3f} | {vals['forget_failures']} | "
            f"{vals['within_stale_fraction']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {vals['noforget_reference_rate']:.3f} |"
        )
    gate = summary["gate"]
    if gate.get("pass"):
        gate_line = "K1 two-scenario stale-dominance gate: `PASS`; proceed to K2."
    elif gate.get("dissociation") == "if_stale_dp_not_dominant":
        gate_line = (
            "K1 two-scenario stale-dominance gate: `MISS`; pre-registered IF-vs-DP "
            "dissociation branch: `PROCEED` to K2/K3 contrast explanation."
        )
    else:
        gate_line = "K1 two-scenario stale-dominance gate: `MISS`; report boundary."
    lines.extend(
        [
            "",
            gate_line,
            f"Qualifying scenarios: {', '.join(gate['qualifying_scenarios']) or 'none'}.",
            f"Next step: `{gate.get('next_step', 'unknown')}`.",
            "",
            "## Artifacts",
            "",
            "- `k1_rows.jsonl`",
            "- `k1_summary.json`",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic-preference", required=True)
    parser.add_argument("--instructional-forgetting", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-rows", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--source", default="existing_answer_files")
    args = parser.parse_args()

    rows = []
    rows.extend(classify_dynamic_preference_file(args.dynamic_preference, args.model))
    rows.extend(classify_instructional_forgetting_file(args.instructional_forgetting, args.model))
    summary = summarize_classifications(rows)
    summary["model"] = args.model
    summary["source"] = args.source
    dump_jsonl(args.out_rows, rows)
    dump_json(args.summary_out, summary)
    write_report(args.report_out, summary, args.model, args.source)


if __name__ == "__main__":
    main()
