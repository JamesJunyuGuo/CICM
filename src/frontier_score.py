#!/usr/bin/env python3
"""Deterministic scoring + aggregation for Stage Q responses.

Extracts the integer answer, classifies against the program-known value sets,
and aggregates accuracy + error signature per (model, k, condition).
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict


def extract_int(text):
    if not text:
        return None
    nums = re.findall(r"\d{3,5}", text.replace(",", ""))
    if not nums:
        return None
    # "reply with only the number" -> usually one; else the last stated number
    return int(nums[-1])


def classify(rec):
    if rec.get("error") or rec.get("response") is None:
        return "none"
    g = extract_int(rec["response"])
    if g is None:
        return "none"
    if g == rec["current_value"]:
        return "correct"
    if g in set(rec["stale_values"]):
        return "stale"
    if g in set(rec["other_values"]):
        return "cross"
    return "invented"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = defaultdict(lambda: defaultdict(int))  # (model,k,ctrl) -> label -> n
    for path in args.responses:
        for line in open(path):
            rec = json.loads(line)
            key = (rec["model"], rec["k"], rec["control"])
            cells[key][classify(rec)] += 1

    rows = []
    print(f"{'model':<28} {'k':>4} {'cond':<5} {'n':>4} {'acc':>6} "
          f"{'stale':>6} {'cross':>6} {'inv':>6} {'none':>6} {'stale%err':>9}")
    for (model, k, ctrl) in sorted(cells, key=lambda x: (x[0], x[2], x[1])):
        c = cells[(model, k, ctrl)]
        n = sum(c.values())
        acc = c["correct"] / n if n else 0
        err = n - c["correct"]
        stale_share_err = c["stale"] / err if err else 0.0
        rows.append({"model": model, "k": k, "control": ctrl, "n": n,
                     "accuracy": round(acc, 4),
                     "stale": c["stale"], "cross": c["cross"],
                     "invented": c["invented"], "none": c["none"],
                     "stale_share_of_errors": round(stale_share_err, 4)})
        print(f"{model:<28} {k:>4} {'ctrl' if ctrl else 'ovw':<5} {n:>4} "
              f"{acc:>6.2f} {c['stale']:>6} {c['cross']:>6} {c['invented']:>6} "
              f"{c['none']:>6} {stale_share_err:>9.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
