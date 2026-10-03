import argparse
from pathlib import Path

from icf_k1d import classify_rows_hybrid, summarize_freeform_dp
from icf_mech import dump_json, dump_jsonl, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", default="results/icf_bench/mech/dynamic_preference_local_responses.json")
    parser.add_argument("--sample", default="results/icf_bench/freeform_dp_sample_claude.json")
    parser.add_argument("--rows-out", default="results/icf_bench/mech/dynamic_preference_local_labeled.jsonl")
    parser.add_argument("--summary-out", default="results/icf_bench/mech/dynamic_preference_local_label_summary.json")
    parser.add_argument("--judge-raw", default="results/icf_bench/mech/dynamic_preference_local_judge_raw.jsonl")
    parser.add_argument("--judge-model", default="openai/gpt-4o")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    rows = load_json(args.responses)
    sample = load_json(args.sample)["items"]
    classified = classify_rows_hybrid(
        rows,
        sample,
        args.judge_raw,
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        retries=args.retries,
        timeout=args.timeout,
    )
    # K2/K3 condition on Forget local labels only; NoForget is not generated in the
    # mechanism harvest, so keep the control field out of the summary denominator.
    summary = summarize_freeform_dp(classified, judge_validation_agreement=0.90)
    summary["note"] = "Local mechanism labels for Forget DP responses only; NoForget was not harvested in K2/K3."
    summary["noforget_reference_rate"] = None
    Path(args.rows_out).parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.rows_out, classified)
    dump_json(args.summary_out, summary)
    print(f"wrote {args.rows_out}, {args.summary_out}")


if __name__ == "__main__":
    main()
