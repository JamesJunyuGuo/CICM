"""Parallel-maintenance task generator.

Probes whether the model can hold K independent "workstreams" at once (the
LLM-level substrate of dispatching K parallel sub-agents). Each line overwrites
one stream's current value; the answer is every stream's most recent value.

Two conditions share K, updates-per-stream, total line count, and answers, and
differ ONLY in ordering:

  - interleaved : stream updates are shuffled together (must juggle K streams).
  - serial      : each stream's updates are contiguous (do one stream at a time).

interleaved-minus-serial accuracy = the pure cost of *parallel* maintenance /
cross-stream interference, isolated from arithmetic, length, and per-stream depth.

Recency cannot solve it: all K streams are queried at once, so the single
globally-last line answers at most one stream. Values are all numeric and
distinct within an example, so the only cue for segregation is the stream label.
"""

import argparse
import json
import random
import string

LABELS = list(string.ascii_uppercase)  # A, B, C, ... (K <= 26)


def make_example(condition, K, U, rng, val_lo=100, val_hi=99999):
    """condition: 'interleaved' | 'serial'; K streams; U updates per stream."""
    streams = LABELS[:K]
    n_lines = K * U
    # distinct values across the whole example (unambiguous scoring)
    values = set()
    while len(values) < n_lines:
        values.add(rng.randint(val_lo, val_hi))
    values = list(values)
    rng.shuffle(values)

    if condition == "interleaved":
        slot_streams = [s for s in streams for _ in range(U)]
        rng.shuffle(slot_streams)
    elif condition == "serial":
        block_order = streams[:]
        rng.shuffle(block_order)
        slot_streams = [s for s in block_order for _ in range(U)]
    else:
        raise ValueError(condition)

    lines = []
    per_stream_vals = {s: [] for s in streams}
    vi = iter(values)
    for s in slot_streams:
        v = next(vi)
        lines.append((s, v))
        per_stream_vals[s].append(v)
    gold = {s: per_stream_vals[s][-1] for s in streams}  # last write wins

    body = "\n".join(f"[{s}] = {v}" for s, v in lines)
    ask = ", ".join(f"{s}=<value>" for s in streams)
    prompt = (
        f"You are tracking {K} independent workstreams ({', '.join(streams)}). "
        "Each line sets one stream's current value; the streams are independent. "
        "A stream may be updated several times; only its most recent value counts.\n\n"
        f"{body}\n\n"
        "Question: What is the current value of every stream?\n"
        f"Answer on a single line exactly as: {ask}"
    )
    return {
        "condition": condition,
        "K": K,
        "U": U,
        "streams": streams,
        "gold": gold,  # {stream: value}
        "prompt": prompt,
    }


def build_dataset(out_path, n_per_cell=100, seed=0):
    rng = random.Random(seed)
    K_grid = [2, 3, 4, 6, 8]
    U_grid = [2, 4]
    rows = []
    uid = 0
    for cond in ["interleaved", "serial"]:
        for K in K_grid:
            for U in U_grid:
                for _ in range(n_per_cell):
                    ex = make_example(cond, K, U, rng)
                    ex["id"] = uid
                    uid += 1
                    rows.append(ex)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


if __name__ == "__main__":
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/parallel.jsonl")
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = build_dataset(args.out, args.n_per_cell, args.seed)
    print(f"wrote {n} examples to {args.out}")
