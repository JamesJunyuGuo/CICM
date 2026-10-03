#!/usr/bin/env python3
"""Multi-slot maintenance-under-interference generator (Stage Q, agent scale).

S slots, each updated U times (distinct integers, last = current), all writes
interleaved in one long conversation with filler; then ONE slot is queried.
Stresses holding many current bindings at once. Program-verifiable.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

SLOTS = [f"{a} {b}" for a in
         ["booking", "invoice", "gate", "locker", "badge", "table", "parcel",
          "desk", "room", "seat", "zone", "dock", "aisle", "bin", "rack",
          "slot", "lane", "berth", "kiosk", "vault", "shelf", "cart", "tray",
          "panel", "meter", "valve", "sensor", "relay", "module", "unit"]
         for b in ["code", "number", "id"]]
FILLER = [
    "The weather's been pleasant lately.", "I cleaned out my inbox this morning.",
    "Traffic was light on the way over.", "That documentary was pretty good.",
    "The cafe downtown has a quiet corner.", "We might repaint the hallway.",
]


def _distinct_ints(rng, n, lo=1000, hi=9999):
    s = set()
    while len(s) < n:
        s.add(rng.randint(lo, hi))
    return list(s)


def make_item(rng, n_slots, updates, control, filler=0, total_writes=None):
    """Load- and length-matched multi-slot item with a designated queried slot.
    Overwrite arm: queried slot gets `updates` writes. Control arm: queried slot
    gets 1 write. In BOTH, total writes = `total_writes` (the remainder spread
    over the other slots) so length and working-memory load match; the ONLY
    difference is whether the queried slot itself is overwritten."""
    slots = rng.sample(SLOTS, n_slots)
    qslot = slots[0]
    uq = updates  # queried slot mentioned `updates` times in BOTH arms (freq-matched)
    W = total_writes if total_writes else (n_slots * updates)
    counts = {s: 1 for s in slots}
    counts[qslot] = uq
    remaining = W - sum(counts.values())
    others = slots[1:]
    while remaining > 0 and others:
        counts[rng.choice(others)] += 1
        remaining -= 1
    vals = {s: _distinct_ints(rng, counts[s]) for s in slots}
    if control:  # freq-matched: queried slot restated with the SAME value (no overwrite)
        vals[qslot] = [vals[qslot][0]] * counts[qslot]
    # interleave preserving per-slot order
    order = []
    idx = {s: 0 for s in slots}
    pending = [s for s in slots if counts[s] > 0]
    while pending:
        s = rng.choice(pending)
        order.append((s, idx[s])); idx[s] += 1
        if idx[s] >= counts[s]:
            pending.remove(s)

    msgs = [{"role": "user", "content": "Track these values for me; I'll set and revise "
             "several. Answer the final question with the most recent value only."},
            {"role": "assistant", "content": "Understood."}]
    for (s, i) in order:
        v = vals[s][i]
        verb = "Set" if i == 0 else "Change"
        msgs.append({"role": "user", "content": f"{verb} the {s} to {v}."})
        msgs.append({"role": "assistant", "content": "Noted."})
        if filler:
            msgs.append({"role": "user", "content": " ".join(rng.choice(FILLER) for _ in range(filler))})
            msgs.append({"role": "assistant", "content": "Ok."})

    q = qslot  # always query the designated slot (overwritten in ovw arm, stable in ctrl)
    current = vals[q][-1]
    stale = [] if control else vals[q][:-1]  # control restates one value: no genuine stale
    other = [vv for s in slots if s != q for vv in vals[s]]
    msgs.append({"role": "user", "content": f"What is the current {q}? Reply with only the number."})
    return {"target": q, "n_slots": n_slots, "updates": uq, "control": control,
            "current_value": current, "stale_values": stale, "other_values": other,
            "k": updates, "total_writes": len(order), "answer": current, "messages": msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots", default="10,20")
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--filler", type=int, default=0)
    ap.add_argument("--total-writes", type=int, default=0, help="0=n_slots*updates")
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for ns in [int(x) for x in args.slots.split(",")]:
        for control in (False, True):
            for j in range(args.n):
                it = make_item(rng, ns, args.updates, control, args.filler,
                               args.total_writes or None)
                it["item_id"] = f"s{ns}_{'ctrl' if control else 'ovw'}_{j:03d}"
                rows.append(it)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} multi-slot items to {args.out}")


if __name__ == "__main__":
    main()
