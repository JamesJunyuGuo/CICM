#!/usr/bin/env python3
"""Naturalistic semantic-competition generator (Stage Q, option B).

Preference/instruction updates in natural prose, with paraphrased/implicit
updates and semantically ADJACENT values (so selection cannot rely on gross
dissimilarity or a fixed "set X to Y" pattern). Still program-verifiable: the
current value is the last update and scoring is canonical-label match.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

# semantically adjacent value pools (within-slot competitors are all similar)
POOLS = {
    "meal plan": ["keto", "paleo", "low-carb", "gluten-free", "vegan",
                  "mediterranean", "pescatarian"],
    "workout routine": ["yoga", "pilates", "HIIT", "weightlifting", "running",
                         "swimming", "cycling"],
    "music preference": ["jazz", "blues", "classical", "lo-fi", "ambient",
                         "folk", "bossa nova"],
    "coffee order": ["espresso", "latte", "cappuccino", "americano",
                     "flat white", "macchiato", "cortado"],
    "commute mode": ["subway", "bus", "biking", "walking", "rideshare",
                     "scooter", "train"],
    "news source": ["Reuters", "AP", "BBC", "NPR", "Bloomberg", "Guardian",
                    "Axios"],
}
FIRST = [
    "For my {slot}, let's go with {v}.",
    "Set my {slot} to {v}.",
    "I'd like my {slot} to be {v} for now.",
]
UPDATE = [  # paraphrased / implicit (some don't repeat the slot name)
    "Actually, switch my {slot} to {v}.",
    "On second thought, make it {v}.",
    "Scrap that — {v} instead.",
    "I changed my mind, let's do {v}.",
    "Hmm, let's go with {v} instead.",
    "Update that to {v} please.",
    "Nah, {v} suits me better.",
]
DISTRACT = ["Also, for my {slot}, I prefer {v}.",
            "Separately, set my {slot} to {v}."]
REMENTION = [  # mention a stale value naturally, NOT as an update
    "I did enjoy {v} for my {slot} back in the day, but anyway.",
    "Remember when we tried {v}? Good times.",
    "My friend still swears by {v}, funny enough.",
]
FILLER = [
    "The weather's been pleasant lately, nothing dramatic.",
    "I finally cleaned out my inbox this morning.",
    "Traffic was light on the way over today.",
    "That documentary everyone mentioned was pretty good.",
    "I might repaint the hallway this weekend.",
    "The new cafe downtown has a nice quiet corner.",
]


def make_item(rng, k, control, n_distractor=3, n_remention=0, filler=0):
    slot = rng.choice(list(POOLS))
    pool = POOLS[slot][:]
    rng.shuffle(pool)
    n_vals = 1 if control else (k + 1)
    vals = pool[:n_vals]                      # distinct, semantically adjacent
    current = vals[-1]
    stale = [] if control else vals[:-1]

    other_slots = [s for s in POOLS if s != slot]
    rng.shuffle(other_slots)
    dslots = other_slots[:n_distractor]
    dvals = [rng.choice(POOLS[s]) for s in dslots]

    # ordered target updates (current = last) + interspersed distractors
    tevents = [("t", i) for i in range(n_vals)]
    devents = [("d", s, v) for s, v in zip(dslots, dvals)]
    if control:  # extra distractor updates to match turn count
        for j in range(k):
            s = other_slots[(n_distractor + j) % len(other_slots)]
            devents.append(("d", s, rng.choice(POOLS[s])))
    events = list(tevents)
    for de in devents:
        events.insert(rng.randint(0, len(events)), de)

    msgs = [{"role": "user", "content": "Hey, help me keep track of a few preferences "
             "as we chat; I'll update some of them."},
            {"role": "assistant", "content": "Sure, I'll keep track."}]
    for ev in events:
        if ev[0] == "t":
            i = ev[1]; v = vals[i]
            tmpl = rng.choice(FIRST) if i == 0 else rng.choice(UPDATE)
            msgs.append({"role": "user", "content": tmpl.format(slot=slot, v=v)})
        else:
            _, s, v = ev
            msgs.append({"role": "user", "content": rng.choice(DISTRACT).format(slot=s, v=v)})
        msgs.append({"role": "assistant", "content": "Got it."})
        if filler > 0:
            msgs.append({"role": "user", "content": " ".join(rng.choice(FILLER) for _ in range(filler))})
            msgs.append({"role": "assistant", "content": "Ha, noted."})
    # post-update stale re-mentions (natural, non-authoritative)
    rem_pool = stale if (not control and stale) else dvals
    for j in range(n_remention):
        v = rng.choice(rem_pool) if rem_pool else rng.choice(POOLS[slot])
        s = slot if (not control and stale) else dslots[0]
        msgs.append({"role": "user", "content": REMENTION[j % len(REMENTION)].format(slot=s, v=v)})
        msgs.append({"role": "assistant", "content": "Understood."})
        if filler > 0:  # spread the post-update tail so current is far, stale near
            msgs.append({"role": "user", "content": " ".join(rng.choice(FILLER) for _ in range(filler))})
            msgs.append({"role": "assistant", "content": "Sure."})
    msgs.append({"role": "user", "content": f"What is my {slot} set to right now? "
                 f"Reply with only the value, no other words."})

    return {"slot": slot, "k": k, "control": control, "n_remention": n_remention,
            "current_value": current, "stale_values": stale, "other_values": dvals,
            "answer": current, "messages": msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--loads", default="1,2,3,4,6")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--n-distractor", type=int, default=3)
    ap.add_argument("--remention", type=int, default=0)
    ap.add_argument("--filler", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    loads = [int(x) for x in args.loads.split(",")]
    rows = []
    for k in loads:
        for control in (False, True):
            for j in range(args.n):
                it = make_item(rng, k, control, args.n_distractor, args.remention, args.filler)
                it["item_id"] = f"k{k}_{'ctrl' if control else 'ovw'}_{j:03d}"
                rows.append(it)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} NL items to {args.out} (loads={loads}, n={args.n}/cell)")


if __name__ == "__main__":
    main()
