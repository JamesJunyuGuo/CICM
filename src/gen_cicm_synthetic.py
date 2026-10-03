import argparse
import json
import random
from collections import Counter
from pathlib import Path


DEFAULT_DOSES = [1, 2, 3, 4, 6]


def controlled_slots() -> dict[str, list[str]]:
    shared_values = ["classic", "modern", "minimal", "immersive", "social", "quiet", "budget"]
    return {
        "music_genre": list(shared_values),
        "diet": list(shared_values),
        "learning_style": list(shared_values),
        "transport": list(shared_values),
        "movie_style": list(shared_values),
        "exercise": list(shared_values),
        "workspace": list(shared_values),
        "cuisine": list(shared_values),
        "reading_format": list(shared_values),
        "meeting_time": list(shared_values),
        "vacation": list(shared_values),
        "pet_preference": list(shared_values),
        "drink": list(shared_values),
        "game_style": list(shared_values),
        "clothing": list(shared_values),
        "communication": list(shared_values),
    }


SLOT_LABELS = {
    "music_genre": "music genre",
    "diet": "meal style",
    "learning_style": "learning style",
    "transport": "transportation mode",
    "movie_style": "movie style",
    "exercise": "exercise routine",
    "workspace": "workspace setting",
    "cuisine": "cuisine",
    "reading_format": "reading format",
    "meeting_time": "meeting time",
    "vacation": "vacation setting",
    "pet_preference": "pet preference",
    "drink": "drink",
    "game_style": "game style",
    "clothing": "clothing style",
    "communication": "communication channel",
}


def load_prefeval_seed_rows(path: str | None) -> list[dict]:
    if not path:
        path = "external_data/icf_bench/dynamic_preference/dynamic_preference_new.json"
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def default_filler_pool(prefeval_rows: list[dict]) -> list[tuple[str, str]]:
    fillers = []
    for row in prefeval_rows:
        user = row.get("conversation_query") or row.get("implicit_query") or row.get("question")
        assistant = row.get("conversation_assistant_acknowledgment") or row.get("explanation")
        if user and assistant:
            fillers.append((str(user).strip(), str(assistant).strip()))
    if fillers:
        return fillers
    return [
        ("Could you keep that in mind while we plan?", "Yes, I will keep the latest preference in mind."),
        ("I may ask for a recommendation after a few details.", "Understood. I will use your latest update."),
        ("Let's continue with the planning context.", "Sure. I am tracking the relevant preference updates."),
    ]


def slot_label(slot: str) -> str:
    return SLOT_LABELS.get(slot, slot.replace("_", " "))


def set_turn(slot: str, value: str, variant: int) -> tuple[str, str]:
    label = slot_label(slot)
    user_templates = [
        "For my {label}, set my preference profile to {value}.",
        "Please remember that my current {label} profile value is {value}.",
        "Update my {label}: use the {value} preference profile now.",
        "Use {value} as my {label} profile going forward.",
    ]
    assistant_templates = [
        "Noted. Your {label} profile is {value}.",
        "Got it. I will treat {value} as your {label} profile.",
        "Understood. The current {label} profile value is {value}.",
        "I have updated {label} to {value}.",
    ]
    return (
        user_templates[variant % len(user_templates)].format(label=label, value=value),
        assistant_templates[variant % len(assistant_templates)].format(label=label, value=value),
    )


def query_turn(slot: str, style: str = "direct") -> str:
    label = slot_label(slot)
    if style == "natural":
        return (
            f"I need the value I should use now for my {label}. "
            f"Reply with just that one value."
        )
    if style == "referential":
        return (
            f"For the {label} we revised earlier, what value should I use now? "
            f"Return only the value."
        )
    return (
        f"What is my current {label} profile value? "
        f"Answer with only the current value, no explanation."
    )


def system_prompt(style: str = "explicit") -> str:
    if style == "plain":
        return "You are a helpful assistant in a continuing conversation. Answer the user's final question using the conversation."
    if style == "minimal":
        return "You are a helpful assistant."
    return (
        "Track the user's latest preference updates. "
        "When asked, answer with only the current value for the requested preference slot."
    )


def choose_stale_values(values: list[str], current: str, k: int, rng: random.Random) -> list[str]:
    pool = [value for value in values if value != current]
    if len(pool) >= k:
        return rng.sample(pool, k)
    stale = []
    while len(stale) < k:
        stale.extend(rng.sample(pool, len(pool)))
    return stale[:k]


def choose_distractor_update(
    slots: dict[str, list[str]],
    target_slot: str,
    row_index: int,
    update_index: int,
) -> tuple[str, str]:
    other_slots = sorted(slot for slot in slots if slot != target_slot)
    slot = other_slots[(row_index + update_index) % len(other_slots)]
    values = list(slots[slot])
    value = values[(row_index + 2 * update_index) % len(values)]
    return slot, value


def build_stage_l_rows(
    *,
    n_per_dose: int,
    doses: list[int] | None = None,
    seed: int = 0,
    slots: dict[str, list[str]] | None = None,
    prefeval_rows: list[dict] | None = None,
    filler_pool: list[tuple[str, str]] | None = None,
    post_final_filler_turns: int = 0,
    post_final_distractor_updates: int = 0,
    system_style: str = "explicit",
    query_style: str = "direct",
) -> list[dict]:
    rng = random.Random(seed)
    doses = doses or list(DEFAULT_DOSES)
    slots = slots or controlled_slots()
    prefeval_rows = prefeval_rows or []
    filler_pool = filler_pool or default_filler_pool(prefeval_rows)
    all_values = sorted({value for values in slots.values() for value in values})
    slot_names = sorted(slots)
    shared_vocab = all(set(values) == set(all_values) for values in slots.values())
    slot_value_pairs = [(slot, value) for slot, values in slots.items() for value in values]
    rng.shuffle(slot_value_pairs)
    rows = []
    pair_cursor = 0
    value_cursor = 0
    slot_cursor = 0
    filler_cursor = 0
    for dose in doses:
        for dose_index in range(n_per_dose):
            if shared_vocab:
                slot = slot_names[slot_cursor % len(slot_names)]
                current = all_values[value_cursor % len(all_values)]
                slot_cursor += 1
                value_cursor += 1
            else:
                slot, current = slot_value_pairs[pair_cursor % len(slot_value_pairs)]
                pair_cursor += 1
            values = list(slots[slot])
            stale_values = choose_stale_values(values, current, dose, rng)
            binding_values = stale_values + [current]
            messages = [
                {
                    "role": "system",
                    "content": system_prompt(system_style),
                }
            ]
            mentions = []
            distractor_updates = []
            for binding_index, value in enumerate(binding_values):
                user, assistant = set_turn(slot, value, dose_index + binding_index)
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": assistant})
                mentions.append(
                    {
                        "binding_index": binding_index,
                        "value": value,
                        "slot": slot,
                        "is_current": value == current and binding_index == len(binding_values) - 1,
                        "message_index": len(messages) - 2,
                    }
                )
                if binding_index < len(binding_values) - 1:
                    user_fill, assistant_fill = filler_pool[filler_cursor % len(filler_pool)]
                    filler_cursor += 1
                    messages.append({"role": "user", "content": user_fill})
                    messages.append({"role": "assistant", "content": assistant_fill})
            for _ in range(post_final_filler_turns):
                user_fill, assistant_fill = filler_pool[filler_cursor % len(filler_pool)]
                filler_cursor += 1
                messages.append({"role": "user", "content": user_fill})
                messages.append({"role": "assistant", "content": assistant_fill})
            for update_index in range(post_final_distractor_updates):
                distractor_slot, distractor_value = choose_distractor_update(
                    slots, slot, dose_index, update_index
                )
                user, assistant = set_turn(distractor_slot, distractor_value, dose_index + update_index)
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": assistant})
                distractor_updates.append(
                    {
                        "slot": distractor_slot,
                        "value": distractor_value,
                        "message_index": len(messages) - 2,
                    }
                )
            query = query_turn(slot, query_style)
            messages.append({"role": "user", "content": query})
            rows.append(
                {
                    "id": f"stage_l_{dose}_{dose_index:05d}",
                    "base_dataset": "PrefEval via ICF-Bench dynamic_preference",
                    "slot": slot,
                    "slot_label": slot_label(slot),
                    "slot_values": values,
                    "all_controlled_values": all_values,
                    "current_value": current,
                    "stale_values": stale_values,
                    "binding_values": binding_values,
                    "k_overwrites": int(dose),
                    "filler_len": int(2 * dose),
                    "messages": messages,
                    "query": query,
                    "value_mentions": mentions,
                    "distractor_updates": distractor_updates,
                    "hardening": {
                        "post_final_filler_turns": int(post_final_filler_turns),
                        "post_final_distractor_updates": int(post_final_distractor_updates),
                        "system_style": system_style,
                        "query_style": query_style,
                    },
                    "seed": seed,
                }
            )
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_datasheet(path: str | Path, rows: list[dict], slots: dict[str, list[str]], args) -> None:
    value_counts = Counter(row["current_value"] for row in rows)
    dose_counts = Counter(row["k_overwrites"] for row in rows)
    lines = [
        "# Stage L CICM Datasheet",
        "",
        "Base: PrefEval-style preference material from `external_data/icf_bench/dynamic_preference/`.",
        "Surface: natural multi-turn preference updates plus filler turns; labels are controlled vocabulary values.",
        "Scoring: program-verifiable normalized matching against controlled values; no LLM judge for headlines.",
        "",
        f"Seed: `{args.seed}`",
        f"Rows: `{len(rows)}`",
        f"Dose counts: `{dict(sorted(dose_counts.items()))}`",
        f"Slots: `{len(slots)}`",
        f"Slot-value assignments: `{sum(len(v) for v in slots.values())}`",
        f"Unique controlled values: `{len(set(value for values in slots.values() for value in values))}`",
        f"Current-value count min/max: `{min(value_counts.values())}` / `{max(value_counts.values())}`",
        f"Hardening: `{getattr(args, 'hardening_name', 'v1')}`",
        f"Post-final filler turns: `{getattr(args, 'post_final_filler_turns', 0)}`",
        f"Post-final distractor updates: `{getattr(args, 'post_final_distractor_updates', 0)}`",
        f"System style: `{getattr(args, 'system_style', 'explicit')}`",
        f"Query style: `{getattr(args, 'query_style', 'direct')}`",
        "",
        "The default profile uses repeated cross-slot preference values so the current-value probe is identifiable.",
        "With `--n-per-dose 210`, each unique controlled value appears as current exactly 150 times.",
        "",
        "## Slots",
        "",
    ]
    for slot, values in slots.items():
        lines.append(f"- `{slot}`: {', '.join(values)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cicm/cicm.jsonl")
    parser.add_argument("--datasheet", default="data/cicm/DATASHEET.md")
    parser.add_argument("--prefeval", default="external_data/icf_bench/dynamic_preference/dynamic_preference_new.json")
    parser.add_argument("--n-per-dose", type=int, default=210)
    parser.add_argument("--doses", default="1,2,3,4,6")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--target-current-per-value", type=int)
    parser.add_argument("--hardening-name", default="v1")
    parser.add_argument("--post-final-filler-turns", type=int, default=0)
    parser.add_argument("--post-final-distractor-updates", type=int, default=0)
    parser.add_argument("--system-style", choices=["explicit", "plain", "minimal"], default="explicit")
    parser.add_argument("--query-style", choices=["direct", "natural", "referential"], default="direct")
    args = parser.parse_args()

    slots = controlled_slots()
    doses = [int(x) for x in args.doses.split(",") if x.strip()]
    n_per_dose = args.n_per_dose
    if args.target_current_per_value:
        n_values = sum(len(values) for values in slots.values())
        n_per_dose = max(n_per_dose, (args.target_current_per_value * n_values + len(doses) - 1) // len(doses))
    rows = build_stage_l_rows(
        n_per_dose=n_per_dose,
        doses=doses,
        seed=args.seed,
        slots=slots,
        prefeval_rows=load_prefeval_seed_rows(args.prefeval),
        post_final_filler_turns=args.post_final_filler_turns,
        post_final_distractor_updates=args.post_final_distractor_updates,
        system_style=args.system_style,
        query_style=args.query_style,
    )
    write_jsonl(args.out, rows)
    write_datasheet(args.datasheet, rows, slots, args)
    summary = {
        "out": args.out,
        "datasheet": args.datasheet,
        "n": len(rows),
        "doses": doses,
        "n_per_dose": n_per_dose,
        "target_current_per_value": args.target_current_per_value,
        "hardening_name": args.hardening_name,
        "post_final_filler_turns": args.post_final_filler_turns,
        "post_final_distractor_updates": args.post_final_distractor_updates,
        "system_style": args.system_style,
        "query_style": args.query_style,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
