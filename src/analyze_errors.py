"""Classify parallel-task errors (CPU-only, reuses saved outputs).

For each wrong stream, is the predicted value:
  - within_stale : a *stale* (earlier) value of the SAME stream  -> within-stream interference
  - cross_leak   : a value belonging to a DIFFERENT stream        -> cross-stream leakage
  - missing      : no value parsed for that stream
  - other        : a number not present in the context (hallucinated)

Joins results/parallel_qwen7b.jsonl (raw outputs) with data/parallel.jsonl
(prompts -> per-stream assignment history + gold). All values are distinct
within an example, so classification is unambiguous.
"""

import collections
import json
import re

DATA = "data/parallel.jsonl"
RESULTS = "results/parallel_qwen7b.jsonl"

_ASSIGN_RE = re.compile(r"\[([A-Z])\]\s*=\s*(-?\d+)")
_PAIR_RE = re.compile(r"([A-Z])\s*=\s*(-?\d+)")


def per_stream_history(prompt):
    """{stream: [values in assignment order]} parsed from the prompt body."""
    hist = collections.defaultdict(list)
    for s, v in _ASSIGN_RE.findall(prompt):
        hist[s].append(int(v))
    return hist


def parse_pred(raw):
    # only take the answer region: pairs like A=123 ; take last occurrence per stream
    d = {}
    for s, v in _PAIR_RE.findall(raw):
        d[s] = int(v)
    return d


def main():
    data = {json.loads(l)["id"]: json.loads(l) for l in open(DATA)}
    rows = [json.loads(l) for l in open(RESULTS)]

    # focus: realistic regime, the axis that actually bites
    cells = [("interleaved", 4), ("serial", 4), ("interleaved", 2)]
    for cond, U in cells:
        for K in [2, 3, 4]:
            counts = collections.Counter()
            wrong = 0
            total_streams = 0
            for r in rows:
                if r["condition"] != cond or r["K"] != K or r["U"] != U:
                    continue
                ex = data[r["id"]]
                hist = per_stream_history(ex["prompt"])
                gold = {k: int(v) for k, v in ex["gold"].items()}
                pred = parse_pred(r["raw"])
                all_vals = {v for vs in hist.values() for v in vs}
                for s, g in gold.items():
                    total_streams += 1
                    p = pred.get(s)
                    if p == g:
                        continue
                    wrong += 1
                    own = set(hist[s])
                    own_stale = own - {g}
                    others = all_vals - own
                    if p is None:
                        counts["missing"] += 1
                    elif p in own_stale:
                        counts["within_stale"] += 1
                    elif p in others:
                        counts["cross_leak"] += 1
                    else:
                        counts["other"] += 1
            if wrong == 0:
                print(f"{cond:11s} K={K} U={U}: no errors (acc=1.0)")
                continue
            frac = {k: round(counts[k] / wrong, 3) for k in ["within_stale", "cross_leak", "missing", "other"]}
            print(
                f"{cond:11s} K={K} U={U}: errors={wrong}/{total_streams} "
                f"({wrong/total_streams:.2f})  "
                f"within_stale={frac['within_stale']}  cross_leak={frac['cross_leak']}  "
                f"missing={frac['missing']}  other={frac['other']}"
            )


if __name__ == "__main__":
    main()
