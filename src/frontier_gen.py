#!/usr/bin/env python3
"""CICM-Hard item generator (Stage Q).

Program-verifiable update-tracking items with integer values (unambiguous
exact-match scoring). Each overwrite item has k stale writes of one target slot
plus a length/turn-matched no-overwrite control. Distractor slots supply the
cross-slot competitor pool. See docs/stage_q_frontier_replication_spec.md.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

TARGET_SLOTS = [
    "booking reference", "meeting room number", "delivery slot code",
    "project budget code", "seat assignment number", "invoice number",
]
DISTRACTOR_SLOTS = [
    "loyalty number", "gate number", "locker code", "badge number",
    "table number", "parcel id", "desk number", "ticket count",
]


def _distinct_ints(rng, n, lo=1000, hi=9999):
    s = set()
    while len(s) < n:
        s.add(rng.randint(lo, hi))
    return list(s)


# neutral filler sentences (NO 3-5 digit numbers, so scoring is never confused)
_FILLER_BANK = [
    "The team reviewed the onboarding notes and agreed the schedule looked reasonable.",
    "Someone mentioned the weather had been unusually mild for the season.",
    "We briefly discussed whether the kickoff should be a call or an email thread.",
    "The venue confirmed that parking would be available near the main entrance.",
    "A colleague shared a summary of last week's planning workshop.",
    "There was a short tangent about the best coffee near the office.",
    "The document formatting was tidied up and a few typos were corrected.",
    "Everyone agreed the agenda should stay focused and end on time.",
    "The travel desk noted that the usual hotel had good reviews this year.",
    "We recapped the goals and made sure nothing important was missing.",
    "The catering options were described as flexible and easy to adjust.",
    "A brief note reminded us to keep the summary concise for the newsletter.",
    "The support channel had been quiet, which everyone took as a good sign.",
    "It was suggested that the next sync could be shorter than usual.",
    "The slides were reorganized so the overview came before the details.",
]


def _filler_paragraph(rng, n_sentences):
    return " ".join(rng.choice(_FILLER_BANK) for _ in range(n_sentences))


def make_item(rng, k, control, n_distractor_slots=3, n_remention=0, filler=0):
    """One item. Overwrite: k+1 target writes (last = current). Control: 1 target
    write + k extra distractor writes so total turns match the overwrite arm."""
    slots = rng.sample(TARGET_SLOTS, 1) + rng.sample(DISTRACTOR_SLOTS, n_distractor_slots)
    target = slots[0]
    distractors = slots[1:]
    n_target_vals = 1 if control else (k + 1)
    vals = _distinct_ints(rng, n_target_vals + n_distractor_slots + (k if control else 0))
    ti = 0
    target_vals = vals[ti:ti + n_target_vals]; ti += n_target_vals
    dist_vals = vals[ti:ti + n_distractor_slots]; ti += n_distractor_slots
    extra_vals = vals[ti:ti + (k if control else 0)]

    # Target writes MUST stay in temporal order (current = the last target write).
    # Distractor writes are sprinkled at random positions without disturbing that
    # order, so the most recent value of the target slot is unambiguously current.
    target_events = [("target", i) for i in range(n_target_vals)]
    dist_events = [("dist", s, v) for s, v in zip(distractors, dist_vals)]
    if control:  # extra distractor writes to match the overwrite turn count
        pool = distractors or DISTRACTOR_SLOTS
        for j, v in enumerate(extra_vals):
            dist_events.append(("dist", pool[j % len(pool)] + f" #{j+2}", v))
    events = list(target_events)
    for de in dist_events:
        events.insert(rng.randint(0, len(events)), de)  # preserves target order

    # optional post-update re-mentions: reference a STALE value of the target in a
    # non-authoritative way AFTER the current update, so the last *mention* of the
    # slot is stale while the last *update* is current. Defeats a "return the last
    # number" heuristic; operationalizes identity-recency competition. Control has
    # no stale values, so it re-mentions an unrelated other-slot value instead.
    rementions = []
    if n_remention > 0:
        stale_pool = target_vals[:-1]  # empty in control
        pool = stale_pool if (not control and stale_pool) else dist_vals
        for j in range(n_remention):
            v = rng.choice(pool) if pool else _distinct_ints(rng, 1)[0]
            slot = target if not control else (distractors[0] if distractors else target)
            rementions.append(("remention", slot, v))

    msgs = [{"role": "user",
             "content": "You are helping me keep track of some booking details. "
                        "I will give and revise several values; answer my final "
                        "question using only the most recent value."},
            {"role": "assistant", "content": "Got it, go ahead."}]
    tw = 0
    for ev in events:
        if ev[0] == "target":
            i = ev[1]
            v = target_vals[i]
            if i == 0:
                msgs.append({"role": "user", "content": f"Set the {target} to {v}."})
            else:
                msgs.append({"role": "user", "content": f"Actually, change the {target} to {v}."})
            tw += 1
        else:
            _, s, v = ev
            msgs.append({"role": "user", "content": f"Separately, note the {s} as {v}."})
        # brief assistant acks keep it a real dialogue
        msgs.append({"role": "assistant", "content": "Noted."})
        # long neutral filler spreads updates across a long context
        if filler > 0:
            msgs.append({"role": "user",
                         "content": "Some background while we're at it. " + _filler_paragraph(rng, filler)})
            msgs.append({"role": "assistant", "content": "Thanks, noted."})
    # post-update re-mentions (non-authoritative references, not "change to")
    remention_phrasings = [
        "Just to confirm, the earlier {s} of {v} is no longer valid, right?",
        "For the record, we previously had the {s} as {v}.",
        "Earlier I mentioned {v} for the {s}; that was the old one.",
    ]
    for j, (_, s, v) in enumerate(rementions):
        msgs.append({"role": "user",
                     "content": remention_phrasings[j % len(remention_phrasings)].format(s=s, v=v)})
        msgs.append({"role": "assistant", "content": "Understood."})
    msgs.append({"role": "user",
                 "content": f"What value is the {target} set to now (the value it was most "
                            f"recently changed to)? Reply with only the number, no words."})

    current = target_vals[-1]
    stale = [] if control else target_vals[:-1]
    return {
        "target": target, "k": k, "control": control, "n_remention": n_remention,
        "current_value": current, "stale_values": stale,
        "other_values": list(dist_vals) + list(extra_vals),
        "answer": current, "messages": msgs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--loads", default="1,2,4,8,16")
    ap.add_argument("--n", type=int, default=20, help="items per (load x condition)")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--n-distractor-slots", type=int, default=3)
    ap.add_argument("--remention", type=int, default=0,
                    help="post-update stale re-mentions (identity-recency stress)")
    ap.add_argument("--filler", type=int, default=0,
                    help="neutral filler sentences per gap (long-context stress)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    loads = [int(x) for x in args.loads.split(",")]
    rows = []
    for k in loads:
        for control in (False, True):
            for j in range(args.n):
                it = make_item(rng, k, control, args.n_distractor_slots, args.remention, args.filler)
                it["item_id"] = f"k{k}_{'ctrl' if control else 'ovw'}_{j:03d}"
                rows.append(it)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} items to {args.out} "
          f"(loads={loads}, n={args.n}/cell, 2 conditions)")


if __name__ == "__main__":
    main()
