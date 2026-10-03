import argparse
import json
from collections import Counter
from pathlib import Path


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _question_with_options(question: str, options: list[str]) -> str:
    option_lines = "\n".join(f"{letter}. {option}" for letter, option in zip("ABCD", options))
    return (
        f"{question} Here are four options, you can choose one as your answer, "
        "just return the content of option, no more additional descriptions or omissions.\n"
        f"{option_lines}"
    )


def build_dynamic_preference_rows(old_rows: list[dict], new_rows: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    drops: Counter[str] = Counter()

    for old_idx, old in enumerate(old_rows):
        new_idx = old.get("from_implicit_id")
        if not isinstance(new_idx, int) or new_idx < 0 or new_idx >= len(new_rows):
            drops["missing_new_pair"] += 1
            continue

        new = new_rows[new_idx]
        options = list(new.get("options") or [])
        old_op = old.get("source_option", "")
        new_op = new.get("aligned_op", "")
        if len(options) != 4:
            drops["non_four_options"] += 1
            continue
        if old_op not in options or new_op not in options:
            drops["paired_option_missing"] += 1
            continue

        rows.append(
            {
                "id": len(rows),
                "source_old_index": old_idx,
                "source_new_index": new_idx,
                "old_preference": old.get("preference", ""),
                "new_preference": new.get("preference", ""),
                "old_op": old_op,
                "new_op": new_op,
                "options": options,
                "question": _question_with_options(new.get("question", ""), options),
            }
        )

    summary = {
        "stage": "k1b_dynamic_preference_prepare",
        "old_rows": len(old_rows),
        "new_rows": len(new_rows),
        "paired_rows": len(rows),
        "drop_counts": dict(drops),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="external_data/icf_bench/dynamic_preference/dynamic_preference_old.json")
    parser.add_argument("--new", default="external_data/icf_bench/dynamic_preference/dynamic_preference_new.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    rows, summary = build_dynamic_preference_rows(load_json(args.old), load_json(args.new))
    dump_json(args.out, rows)
    dump_json(args.summary_out, summary)


if __name__ == "__main__":
    main()
