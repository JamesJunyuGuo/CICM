import argparse
import json
from collections import defaultdict
from pathlib import Path

from icf_k1 import parse_options_from_question
from icf_matchers import (
    classify_dynamic_preference_response,
    classify_instructional_forgetting_response,
)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _bool_from_string(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def dynamic_preference_alignment_rows(path: str | Path, annotator: str) -> list[dict]:
    rows = []
    for item in load_json(path):
        row = dict(item)
        row["options"] = parse_options_from_question(row.get("question", ""))
        forget_type = classify_dynamic_preference_response(row.get("llm_response_Forget", ""), row)
        noforget_type = classify_dynamic_preference_response(row.get("llm_response_NoForget", ""), row)
        rows.append(
            {
                "scenario": "dynamic_preference",
                "annotator_file": annotator,
                "id": str(row.get("id")),
                "matcher_forget": forget_type == "correct_current",
                "human_forget": _bool_from_string(row.get("human_forget")),
                "matcher_noforget": noforget_type in {"correct_current", "within_stale"},
                "human_noforget": _bool_from_string(row.get("human_noforget")),
                "forget_type": forget_type,
                "noforget_type": noforget_type,
            }
        )
    return rows


def instructional_forgetting_alignment_rows(path: str | Path, annotator: str) -> list[dict]:
    rows = []
    for item in load_json(path):
        forget_type = classify_instructional_forgetting_response(item.get("instruction_forget_reply", ""), item)
        noforget_type = classify_instructional_forgetting_response(item.get("instruction_noforget_reply", ""), item)
        rows.append(
            {
                "scenario": "instructional_forgetting",
                "annotator_file": annotator,
                "id": str(item.get("id")),
                "matcher_forget": forget_type == "correct_forget",
                "human_forget": _bool_from_string(item.get("human_forget")),
                "matcher_noforget": noforget_type == "within_stale",
                "human_noforget": _bool_from_string(item.get("human_noforget")),
                "forget_type": forget_type,
                "noforget_type": noforget_type,
            }
        )
    return rows


def summarize_human_alignment(rows: list[dict]) -> dict:
    by_scenario = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)
    for scenario, scenario_rows in sorted(grouped.items()):
        n = len(scenario_rows)
        forget_matches = sum(row["matcher_forget"] == row["human_forget"] for row in scenario_rows)
        noforget_matches = sum(row["matcher_noforget"] == row["human_noforget"] for row in scenario_rows)
        by_scenario[scenario] = {
            "n": n,
            "forget_agreement": forget_matches / n if n else 0.0,
            "noforget_agreement": noforget_matches / n if n else 0.0,
            "overall_agreement": (forget_matches + noforget_matches) / (2 * n) if n else 0.0,
        }
    return {"stage": "k1b_matcher_validation", "by_scenario": by_scenario}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-dir", default="external_data/icf_bench/human_llm_align")
    parser.add_argument("--out-rows", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    human_dir = Path(args.human_dir)
    rows: list[dict] = []
    for path in sorted(human_dir.glob("dynamic_preference_human_llm_align_50_*.json")):
        rows.extend(dynamic_preference_alignment_rows(path, path.name))
    for path in sorted(human_dir.glob("instructional_forgetting_human_llm_align_50_*.json")):
        rows.extend(instructional_forgetting_alignment_rows(path, path.name))
    summary = summarize_human_alignment(rows)
    dump_jsonl(args.out_rows, rows)
    dump_json(args.summary_out, summary)


if __name__ == "__main__":
    main()
