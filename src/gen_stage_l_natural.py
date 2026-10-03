import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_DOSES = [1, 2, 3, 4, 6]
FACTORIAL_CELLS = [
    ("near", "near"),
    ("near", "far"),
    ("far", "near"),
    ("far", "far"),
]
SAME_DISTANCE_BINS = ["near", "far"]


def natural_controlled_slots() -> dict[str, list[str]]:
    return {
        "music_genre": ["jazz", "rock", "classical", "electronic", "folk", "hiphop", "blues"],
        "diet": ["vegan", "vegetarian", "keto", "mediterranean", "paleo", "pescatarian", "gluten free"],
        "learning_style": ["visual", "hands on", "lecture", "reading", "discussion", "self paced", "tutoring"],
        "transport": ["train", "bus", "bicycle", "subway", "rideshare", "walking", "carpool"],
        "movie_style": ["comedy", "drama", "thriller", "documentary", "animation", "romance", "science fiction"],
        "exercise": ["yoga", "running", "swimming", "cycling", "pilates", "weightlifting", "hiking"],
        "workspace": [
            "quiet office",
            "coworking space",
            "home desk",
            "library table",
            "cafe corner",
            "meeting room",
            "standing desk",
        ],
        "cuisine": ["italian", "japanese", "mexican", "thai", "indian", "korean", "greek"],
        "reading_format": ["paperback", "ebook", "audiobook", "hardcover", "newsletter", "comic", "article"],
        "meeting_time": ["morning", "noon", "evening", "weekday", "weekend", "midday", "night"],
        "vacation": ["beach", "mountains", "city", "countryside", "cruise", "cabin", "resort"],
        "pet_preference": ["dog", "cat", "bird", "fish", "rabbit", "reptile", "none"],
        "drink": ["coffee", "tea", "smoothie", "sparkling water", "juice", "lemonade", "cocoa"],
        "game_style": ["puzzle", "strategy", "cooperative", "competitive", "roleplaying", "sandbox", "arcade"],
        "clothing": ["casual", "formal", "sporty", "vintage", "business", "streetwear", "minimalist"],
        "communication": [
            "email",
            "phone call",
            "text message",
            "video call",
            "voice memo",
            "in person",
            "instant message",
        ],
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


def slot_label(slot: str) -> str:
    return SLOT_LABELS.get(slot, slot.replace("_", " "))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_value(text: str, value: str) -> bool:
    text_norm = normalize_text(text)
    value_norm = normalize_text(value)
    return bool(value_norm and re.search(rf"(^| )({re.escape(value_norm)})( |$)", text_norm))


def fallback_template_pack(slot: str) -> dict:
    label = slot_label(slot)
    return {
        "slot": slot,
        "slot_label": label,
        "system": "You are a helpful assistant in an ongoing preference-planning conversation.",
        "opening": {
            "user": f"I am updating a few personal preferences for planning. Please keep track of my {label} as we go.",
            "assistant": "Understood. I will follow the conversation and use the latest update when you ask.",
        },
        "updates": [
            {
                "user": f"For my {label}, I want to use {{value}} from now on.",
                "assistant": "Got it. I will use the latest version of that preference.",
            },
            {
                "user": f"I changed my mind about my {label}; make it {{value}} now.",
                "assistant": "Understood. I will treat that as the current update.",
            },
            {
                "user": f"Please revise the {label} preference to {{value}}.",
                "assistant": "Noted. I will rely on the revised preference.",
            },
            {
                "user": f"Actually, for the {label}, the option I want is {{value}}.",
                "assistant": "Okay. I will keep that as the latest preference.",
            },
        ],
        "fillers": [
            {
                "user": "Before the final recommendation, can we keep the plan practical for a normal week?",
                "assistant": "Yes. I will keep the plan realistic and avoid adding unnecessary complications.",
            },
            {
                "user": "Also keep the overall tone low pressure; I may still revise details later.",
                "assistant": "That makes sense. I will leave room for later adjustments.",
            },
            {
                "user": "Please make sure the final answer is concise when I ask for a specific detail.",
                "assistant": "Sure. If you ask for one detail, I will answer with that detail only.",
            },
            {
                "user": "I am trying to keep all of these preferences organized across the conversation.",
                "assistant": "Understood. I will use the most recent relevant update.",
            },
        ],
        "queries": [
            f"For the {label} I revised earlier, what should I use now? Return only the value.",
            f"What is my current {label}? Reply with only the value.",
        ],
    }


def validate_template_pack(pack: dict, slot: str, values: list[str]) -> dict:
    required = ["system", "opening", "updates", "fillers", "queries"]
    for key in required:
        if key not in pack:
            raise ValueError(f"template pack for {slot} missing {key}")
    if len(pack["updates"]) < 2 or len(pack["fillers"]) < 2 or len(pack["queries"]) < 1:
        raise ValueError(f"template pack for {slot} is too small")
    meta_terms = ["controlled value", "controlled values", "placeholder", "{value}"]
    for group in ["system", "queries"]:
        texts = pack[group] if isinstance(pack[group], list) else [pack[group]]
        for text in texts:
            text_s = str(text)
            text_norm = text_s.lower()
            if any(term in text_norm for term in meta_terms):
                raise ValueError(f"{group} template for {slot} contains meta placeholder language")
    for update in pack["updates"]:
        user = str(update.get("user", ""))
        assistant = str(update.get("assistant", ""))
        if user.count("{value}") != 1:
            raise ValueError(f"update template for {slot} must contain exactly one {{value}}")
        if "{value}" in assistant:
            raise ValueError(f"assistant update template for {slot} must not repeat {{value}}")
    for group in ["system", "queries"]:
        texts = pack[group] if isinstance(pack[group], list) else [pack[group]]
        for text in texts:
            if any(contains_value(str(text), value) for value in values):
                raise ValueError(f"{group} template for {slot} contains a controlled value")
    for turn in [pack["opening"], *pack["fillers"]]:
        for role in ["user", "assistant"]:
            text = str(turn.get(role, ""))
            if "{value}" in text:
                raise ValueError(f"non-update template for {slot} contains {{value}}")
            if any(contains_value(text, value) for value in values):
                raise ValueError(f"non-update template for {slot} contains a controlled value")
    return pack


def generate_template_pack_openai(slot: str, values: list[str], model: str) -> dict:
    from openai import OpenAI

    label = slot_label(slot)
    client = OpenAI()
    prompt = {
        "slot": slot,
        "slot_label": label,
        "controlled_values": values,
        "instructions": [
            "Return JSON only.",
            "Write coherent, natural preference-dialogue templates for this slot.",
            "The user update templates must contain exactly one literal placeholder {value}.",
            "Assistant update templates must acknowledge naturally but must not repeat {value}.",
            "Opening, filler, and query templates must not contain any controlled value.",
            "Avoid words from the controlled values except the literal {value} placeholder.",
            "Queries must ask for the current value and request only the value in the final answer.",
        ],
        "schema": {
            "system": "string",
            "opening": {"user": "string", "assistant": "string"},
            "updates": [{"user": "string with {value}", "assistant": "string"}],
            "fillers": [{"user": "string", "assistant": "string"}],
            "queries": ["string"],
        },
    }
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You generate compact JSON template packs for controlled experiments."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    pack = json.loads(content)
    pack["slot"] = slot
    pack["slot_label"] = label
    pack["system"] = "You are a helpful assistant in an ongoing preference-planning conversation."
    return validate_template_pack(pack, slot, values)


def load_template_packs(
    *,
    slots: dict[str, list[str]],
    template_source: str,
    templates_in: str | None = None,
    templates_out: str | None = None,
    openai_model: str = "gpt-4.1",
) -> dict[str, dict]:
    packs: dict[str, dict] = {}
    if templates_in:
        with Path(templates_in).open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        packs.update(loaded)
    for slot, values in slots.items():
        if slot in packs:
            packs[slot] = validate_template_pack(packs[slot], slot, values)
            continue
        if template_source == "openai":
            try:
                packs[slot] = generate_template_pack_openai(slot, values, openai_model)
            except Exception as exc:
                print(f"template_openai_fallback slot={slot} error={exc}", flush=True)
                packs[slot] = fallback_template_pack(slot)
        else:
            packs[slot] = fallback_template_pack(slot)
        packs[slot] = validate_template_pack(packs[slot], slot, values)
    if templates_out:
        Path(templates_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(templates_out).open("w", encoding="utf-8") as f:
            json.dump(packs, f, ensure_ascii=False, indent=2)
    return packs


def choose_stale_values(values: list[str], current: str, k: int, rng: random.Random) -> list[str]:
    pool = [value for value in values if value != current]
    if len(pool) >= k:
        return rng.sample(pool, k)
    stale = []
    while len(stale) < k:
        stale.extend(rng.sample(pool, len(pool)))
    return stale[:k]


def choose_distractor_update(
    slots: dict[str, list[str]], target_slot: str, row_index: int, update_index: int
) -> tuple[str, str]:
    other_slots = sorted(slot for slot in slots if slot != target_slot)
    slot = other_slots[(row_index + update_index) % len(other_slots)]
    values = list(slots[slot])
    value = values[(row_index + 2 * update_index) % len(values)]
    return slot, value


def render_update(pack: dict, value: str, variant: int) -> tuple[str, str, int, int]:
    template = pack["updates"][variant % len(pack["updates"])]
    user = template["user"].format(value=value)
    start = user.index(value)
    end = start + len(value)
    return user, template["assistant"], start, end


def render_stale_reminder(slot: str, value: str) -> tuple[str, str, int, int]:
    label = slot_label(slot)
    user = f"I am still thinking about {value} when I compare options for my {label}."
    assistant = "Understood. I will keep that context in mind while following your updates."
    start = user.index(value)
    end = start + len(value)
    return user, assistant, start, end


def anchored_query(slot: str, variant: int = 0) -> str:
    label = slot_label(slot)
    templates = [
        (
            "For my {label} specifically, what is the latest value I chose for that "
            "target slot, not any other preference? Reply with only the value."
        ),
        (
            "I am asking only about my {label}. What value is current for that slot "
            "now, not any other preference? Reply with only the value."
        ),
        (
            "Look back at the updates for my {label}. What is the current value for "
            "that exact slot, not any other preference? Reply with only the value."
        ),
    ]
    return templates[variant % len(templates)].format(label=label)


def build_competition_metadata(row: dict) -> dict:
    query_index = len(row["messages"]) - 1

    def summarize_mention(mention: dict, competitor_type: str) -> dict:
        return {
            "type": competitor_type,
            "slot": mention["slot"],
            "value": mention["value"],
            "message_index": mention["message_index"],
            "distance_messages_to_query": query_index - mention["message_index"],
            "distance_bin": mention.get("distance_bin"),
            "mention_kind": mention.get("mention_kind", "binding_update"),
            "binding_index": mention.get("binding_index"),
        }

    current = next(m for m in row["value_mentions"] if m["is_current"])
    stale = [m for m in row["value_mentions"] if not m["is_current"]]
    recent_other = [
        {
            "type": "recent_other_slot",
            "slot": update["slot"],
            "value": update["value"],
            "message_index": update["message_index"],
            "distance_messages_to_query": query_index - update["message_index"],
            "distance_bin": update.get("distance_bin"),
            "recency_rank_from_query": rank,
        }
        for rank, update in enumerate(reversed(row.get("distractor_updates", [])), 1)
    ]
    same_slot = [summarize_mention(m, "same_slot_stale") for m in stale]
    same_slot_nearest = min(same_slot, key=lambda item: item["distance_messages_to_query"]) if same_slot else None
    recent_other_nearest = (
        min(recent_other, key=lambda item: item["distance_messages_to_query"]) if recent_other else None
    )
    return {
        "target_current": summarize_mention(current, "target_current"),
        "same_slot_stale": same_slot,
        "same_slot_stale_nearest": same_slot_nearest,
        "recent_other_slot": recent_other,
        "recent_other_slot_nearest": recent_other_nearest,
        "competitor_types": ["same_slot_stale", "recent_other_slot"],
        "factorial_cell": row.get("factorial_cell"),
    }


def build_natural_stage_l_rows(
    *,
    n_per_dose: int,
    doses: list[int] | None = None,
    seed: int = 0,
    slots: dict[str, list[str]] | None = None,
    template_packs: dict[str, dict] | None = None,
    post_final_filler_turns: int = 4,
    post_final_distractor_updates: int = 2,
    factorial_competition: bool = False,
    factorial_target_current_mode: str = "balanced",
    other_distance_bins: list[str] | None = None,
    extra_interference_turns: int = 0,
) -> list[dict]:
    if factorial_competition:
        return build_factorial_natural_stage_l_rows(
            n_per_dose=n_per_dose,
            doses=doses,
            seed=seed,
            slots=slots,
            template_packs=template_packs,
            post_final_filler_turns=post_final_filler_turns,
            post_final_distractor_updates=post_final_distractor_updates,
            factorial_target_current_mode=factorial_target_current_mode,
            other_distance_bins=other_distance_bins,
            extra_interference_turns=extra_interference_turns,
        )
    rng = random.Random(seed)
    doses = doses or list(DEFAULT_DOSES)
    slots = slots or natural_controlled_slots()
    template_packs = template_packs or {slot: fallback_template_pack(slot) for slot in slots}
    all_values = sorted({value for values in slots.values() for value in values})
    slot_value_pairs = [(slot, value) for slot, values in slots.items() for value in values]
    rng.shuffle(slot_value_pairs)
    rows = []
    pair_cursor = 0
    for dose in doses:
        for dose_index in range(n_per_dose):
            slot, current = slot_value_pairs[pair_cursor % len(slot_value_pairs)]
            pair_cursor += 1
            values = list(slots[slot])
            pack = template_packs[slot]
            stale_values = choose_stale_values(values, current, dose, rng)
            binding_values = stale_values + [current]
            messages = [{"role": "system", "content": pack["system"]}]
            messages.append({"role": "user", "content": pack["opening"]["user"]})
            messages.append({"role": "assistant", "content": pack["opening"]["assistant"]})
            mentions = []
            distractor_updates = []
            filler_cursor = dose_index
            for binding_index, value in enumerate(binding_values):
                user, assistant, char_start, char_end = render_update(pack, value, dose_index + binding_index)
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": assistant})
                mentions.append(
                    {
                        "binding_index": binding_index,
                        "value": value,
                        "slot": slot,
                        "is_current": binding_index == len(binding_values) - 1,
                        "message_index": len(messages) - 2,
                        "role": "user",
                        "char_start": char_start,
                        "char_end": char_end,
                        "mention_kind": "binding_update",
                    }
                )
                if binding_index < len(binding_values) - 1:
                    filler = pack["fillers"][filler_cursor % len(pack["fillers"])]
                    filler_cursor += 1
                    messages.append({"role": "user", "content": filler["user"]})
                    messages.append({"role": "assistant", "content": filler["assistant"]})
            for _ in range(post_final_filler_turns):
                filler = pack["fillers"][filler_cursor % len(pack["fillers"])]
                filler_cursor += 1
                messages.append({"role": "user", "content": filler["user"]})
                messages.append({"role": "assistant", "content": filler["assistant"]})
            for update_index in range(post_final_distractor_updates):
                distractor_slot, distractor_value = choose_distractor_update(
                    slots, slot, dose_index, update_index
                )
                distractor_pack = template_packs[distractor_slot]
                user, assistant, char_start, char_end = render_update(
                    distractor_pack, distractor_value, dose_index + update_index
                )
                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": assistant})
                distractor_updates.append(
                    {
                        "slot": distractor_slot,
                        "value": distractor_value,
                        "message_index": len(messages) - 2,
                        "role": "user",
                        "char_start": char_start,
                        "char_end": char_end,
                        "distance_bin": "near",
                    }
                )
            query = anchored_query(slot, dose_index)
            messages.append({"role": "user", "content": query})
            row = {
                "id": f"stage_l_nat_{dose}_{dose_index:05d}",
                "base_dataset": "Stage L natural CICM",
                "slot": slot,
                "slot_label": slot_label(slot),
                "slot_values": values,
                "all_controlled_values": all_values,
                "current_value": current,
                "stale_values": stale_values,
                "binding_values": binding_values,
                "k_overwrites": int(dose),
                "filler_len": int(dose + post_final_filler_turns),
                "messages": messages,
                "query": query,
                "query_anchor": {
                    "target_slot": slot,
                    "target_slot_label": slot_label(slot),
                    "explicit_identity_anchor": True,
                    "distractors_retained": True,
                },
                "value_mentions": mentions,
                "distractor_updates": distractor_updates,
                "hardening": {
                    "natural_values": True,
                    "post_final_filler_turns": int(post_final_filler_turns),
                    "post_final_distractor_updates": int(post_final_distractor_updates),
                    "template_source": pack.get("template_source", "unknown"),
                },
                "seed": seed,
            }
            row["competition"] = build_competition_metadata(row)
            rows.append(row)
    validate_natural_rows(rows)
    return rows


def add_filler_turns(messages: list[dict], pack: dict, filler_cursor: int, n_turns: int) -> int:
    for _ in range(n_turns):
        filler = pack["fillers"][filler_cursor % len(pack["fillers"])]
        filler_cursor += 1
        messages.append({"role": "user", "content": filler["user"]})
        messages.append({"role": "assistant", "content": filler["assistant"]})
    return filler_cursor


def add_target_update(
    *,
    messages: list[dict],
    mentions: list[dict],
    pack: dict,
    slot: str,
    value: str,
    binding_index: int,
    is_current: bool,
    variant: int,
    distance_bin: str | None = None,
) -> None:
    user, assistant, char_start, char_end = render_update(pack, value, variant)
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    mentions.append(
        {
            "binding_index": binding_index,
            "value": value,
            "slot": slot,
            "is_current": is_current,
            "message_index": len(messages) - 2,
            "role": "user",
            "char_start": char_start,
            "char_end": char_end,
            "mention_kind": "binding_update",
            "distance_bin": distance_bin,
        }
    )


def add_stale_reminder(
    *,
    messages: list[dict],
    mentions: list[dict],
    slot: str,
    value: str,
    binding_index: int,
    distance_bin: str,
) -> None:
    user, assistant, char_start, char_end = render_stale_reminder(slot, value)
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    mentions.append(
        {
            "binding_index": binding_index,
            "value": value,
            "slot": slot,
            "is_current": False,
            "message_index": len(messages) - 2,
            "role": "user",
            "char_start": char_start,
            "char_end": char_end,
            "mention_kind": "stale_reminder",
            "distance_bin": distance_bin,
        }
    )


def add_distractor_update(
    *,
    messages: list[dict],
    distractor_updates: list[dict],
    template_packs: dict[str, dict],
    distractor_slot: str,
    distractor_value: str,
    variant: int,
    distance_bin: str,
) -> None:
    distractor_pack = template_packs[distractor_slot]
    user, assistant, char_start, char_end = render_update(distractor_pack, distractor_value, variant)
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    distractor_updates.append(
        {
            "slot": distractor_slot,
            "value": distractor_value,
            "message_index": len(messages) - 2,
            "role": "user",
            "char_start": char_start,
            "char_end": char_end,
            "distance_bin": distance_bin,
        }
    )


def build_factorial_natural_stage_l_rows(
    *,
    n_per_dose: int,
    doses: list[int] | None = None,
    seed: int = 0,
    slots: dict[str, list[str]] | None = None,
    template_packs: dict[str, dict] | None = None,
    post_final_filler_turns: int = 4,
    post_final_distractor_updates: int = 1,
    factorial_target_current_mode: str = "balanced",
    other_distance_bins: list[str] | None = None,
    extra_interference_turns: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    doses = doses or list(DEFAULT_DOSES)
    slots = slots or natural_controlled_slots()
    template_packs = template_packs or {slot: fallback_template_pack(slot) for slot in slots}
    if other_distance_bins is None:
        cells = list(FACTORIAL_CELLS)
    else:
        valid_other_bins = {"far", "mid", "near", "near2"}
        if any(bin_name not in valid_other_bins for bin_name in other_distance_bins):
            raise ValueError(f"unsupported other_distance_bins: {other_distance_bins}")
        cells = [(same_bin, other_bin) for same_bin in SAME_DISTANCE_BINS for other_bin in other_distance_bins]
    all_values = sorted({value for values in slots.values() for value in values})
    slot_value_pairs = [(slot, value) for slot, values in slots.items() for value in values]
    rng.shuffle(slot_value_pairs)
    rows = []
    pair_cursor = 0
    for dose_rank, dose in enumerate(doses):
        for dose_index in range(n_per_dose):
            same_bin, other_bin = cells[dose_index % len(cells)]
            replicate = dose_index // len(cells)
            if factorial_target_current_mode == "balanced":
                target_current_bin = "near" if replicate % 2 == 0 else "far"
            elif factorial_target_current_mode == "far":
                target_current_bin = "far"
            elif factorial_target_current_mode == "far_with_near_controls":
                target_current_bin = "near" if replicate == 1 and dose_rank < 2 else "far"
            else:
                raise ValueError(f"unsupported factorial_target_current_mode: {factorial_target_current_mode}")
            slot, current = slot_value_pairs[pair_cursor % len(slot_value_pairs)]
            pair_cursor += 1
            values = list(slots[slot])
            pack = template_packs[slot]
            stale_values = choose_stale_values(values, current, dose, rng)
            binding_values = stale_values + [current]
            messages = [{"role": "system", "content": pack["system"]}]
            messages.append({"role": "user", "content": pack["opening"]["user"]})
            messages.append({"role": "assistant", "content": pack["opening"]["assistant"]})
            mentions = []
            distractor_updates = []
            filler_cursor = dose_index

            for binding_index, value in enumerate(stale_values):
                add_target_update(
                    messages=messages,
                    mentions=mentions,
                    pack=pack,
                    slot=slot,
                    value=value,
                    binding_index=binding_index,
                    is_current=False,
                    variant=dose_index + binding_index,
                    distance_bin="far",
                )
                filler_cursor = add_filler_turns(messages, pack, filler_cursor, 1)

            if target_current_bin == "far":
                add_target_update(
                    messages=messages,
                    mentions=mentions,
                    pack=pack,
                    slot=slot,
                    value=current,
                    binding_index=len(binding_values) - 1,
                    is_current=True,
                    variant=dose_index + len(binding_values),
                    distance_bin="far",
                )

            if other_bin == "far":
                for update_index in range(post_final_distractor_updates):
                    distractor_slot, distractor_value = choose_distractor_update(
                        slots, slot, dose_index, update_index
                    )
                    add_distractor_update(
                        messages=messages,
                        distractor_updates=distractor_updates,
                        template_packs=template_packs,
                        distractor_slot=distractor_slot,
                        distractor_value=distractor_value,
                        variant=dose_index + update_index,
                        distance_bin=other_bin,
                    )

            filler_cursor = add_filler_turns(
                messages,
                pack,
                filler_cursor,
                max(4, post_final_filler_turns) + max(0, extra_interference_turns),
            )

            if other_bin == "mid":
                for update_index in range(post_final_distractor_updates):
                    distractor_slot, distractor_value = choose_distractor_update(
                        slots, slot, dose_index, update_index
                    )
                    add_distractor_update(
                        messages=messages,
                        distractor_updates=distractor_updates,
                        template_packs=template_packs,
                        distractor_slot=distractor_slot,
                        distractor_value=distractor_value,
                        variant=dose_index + update_index,
                        distance_bin=other_bin,
                    )
                filler_cursor = add_filler_turns(messages, pack, filler_cursor, 3)

            if same_bin == "near":
                add_stale_reminder(
                    messages=messages,
                    mentions=mentions,
                    slot=slot,
                    value=stale_values[-1],
                    binding_index=len(stale_values) - 1,
                    distance_bin="near",
                )

            if other_bin in {"near", "near2"}:
                for update_index in range(post_final_distractor_updates):
                    distractor_slot, distractor_value = choose_distractor_update(
                        slots, slot, dose_index, update_index
                    )
                    add_distractor_update(
                        messages=messages,
                        distractor_updates=distractor_updates,
                        template_packs=template_packs,
                        distractor_slot=distractor_slot,
                        distractor_value=distractor_value,
                        variant=dose_index + update_index,
                        distance_bin=other_bin,
                    )

            if target_current_bin == "near":
                add_target_update(
                    messages=messages,
                    mentions=mentions,
                    pack=pack,
                    slot=slot,
                    value=current,
                    binding_index=len(binding_values) - 1,
                    is_current=True,
                    variant=dose_index + len(binding_values),
                    distance_bin="near",
                )

            query = anchored_query(slot, dose_index)
            messages.append({"role": "user", "content": query})
            row = {
                "id": f"stage_l_natfact_{dose}_{dose_index:05d}",
                "base_dataset": "Stage L natural CICM factorial competition",
                "slot": slot,
                "slot_label": slot_label(slot),
                "slot_values": values,
                "all_controlled_values": all_values,
                "current_value": current,
                "stale_values": stale_values,
                "binding_values": binding_values,
                "k_overwrites": int(dose),
                "filler_len": int(dose + max(4, post_final_filler_turns)),
                "messages": messages,
                "query": query,
                "query_anchor": {
                    "target_slot": slot,
                    "target_slot_label": slot_label(slot),
                    "explicit_identity_anchor": True,
                    "distractors_retained": True,
                },
                "value_mentions": mentions,
                "distractor_updates": distractor_updates,
                "factorial_cell": {
                    "same_slot_stale_distance_bin": same_bin,
                    "recent_other_slot_distance_bin": other_bin,
                    "target_current_distance_bin": target_current_bin,
                },
                "hardening": {
                    "natural_values": True,
                    "factorial_competition": True,
                    "factorial_target_current_mode": factorial_target_current_mode,
                    "post_final_filler_turns": int(post_final_filler_turns),
                    "post_final_distractor_updates": int(post_final_distractor_updates),
                    "extra_interference_turns": int(extra_interference_turns),
                    "template_source": pack.get("template_source", "unknown"),
                },
                "seed": seed,
            }
            row["competition"] = build_competition_metadata(row)
            row["competition"]["factorial_cell"] = row["factorial_cell"]
            rows.append(row)
    validate_natural_rows(rows)
    return rows


def validate_natural_rows(rows: list[dict]) -> None:
    for row in rows:
        if row["current_value"] in row["stale_values"]:
            raise ValueError(f"{row['id']} current value also appears as stale")
        if len(row["stale_values"]) != row["k_overwrites"]:
            raise ValueError(f"{row['id']} stale count does not match k")
        if row.get("query_anchor", {}).get("target_slot") != row["slot"]:
            raise ValueError(f"{row['id']} query is not anchored to target slot")
        if "not any other preference" not in row.get("query", "").lower():
            raise ValueError(f"{row['id']} query does not disambiguate other preferences")
        if row.get("competition", {}).get("target_current", {}).get("value") != row["current_value"]:
            raise ValueError(f"{row['id']} competition target_current mismatch")
        current_mentions = [m for m in row["value_mentions"] if m["is_current"]]
        if len(current_mentions) != 1:
            raise ValueError(f"{row['id']} must have exactly one current target mention")
        for mention in row["value_mentions"]:
            msg = row["messages"][mention["message_index"]]
            text = msg["content"]
            if msg["role"] != mention["role"]:
                raise ValueError(f"{row['id']} mention role mismatch")
            if text[mention["char_start"] : mention["char_end"]] != mention["value"]:
                raise ValueError(f"{row['id']} mention span mismatch")
            if mention["slot"] != row["slot"]:
                raise ValueError(f"{row['id']} target mention slot mismatch")
        for update in row.get("distractor_updates", []):
            msg = row["messages"][update["message_index"]]
            text = msg["content"]
            if text[update["char_start"] : update["char_end"]] != update["value"]:
                raise ValueError(f"{row['id']} distractor span mismatch")
            if update["slot"] == row["slot"]:
                raise ValueError(f"{row['id']} distractor updates target slot")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_datasheet(path: str | Path, rows: list[dict], slots: dict[str, list[str]], args) -> None:
    current_counts = Counter((row["slot"], row["current_value"]) for row in rows)
    dose_counts = Counter(row["k_overwrites"] for row in rows)
    factorial_counts = Counter(
        (
            row.get("factorial_cell", {}).get("same_slot_stale_distance_bin"),
            row.get("factorial_cell", {}).get("recent_other_slot_distance_bin"),
        )
        for row in rows
        if row.get("factorial_cell")
    )
    target_current_counts = Counter(
        row.get("factorial_cell", {}).get("target_current_distance_bin")
        for row in rows
        if row.get("factorial_cell")
    )
    lines = [
        "# Stage L Natural CICM Datasheet",
        "",
        "Purpose: upgrade Stage L data realism while preserving program-verifiable stale-binding labels.",
        "",
        "## Construction",
        "",
        "- Slot values are natural and slot-specific, not a shared generic value pool.",
        "- A strong model may generate reusable dialogue templates, but the program injects exact controlled values.",
        "- The program records target-slot `value_mentions` with role, message index, and character spans.",
        "- Final queries are programmatically anchored to the target slot to remove query ambiguity.",
        "- Distractor slot updates are retained after the final target binding as recent-other-slot competitors.",
        "- Each row records `competition` metadata for target current, same-slot stale, and recent other-slot values.",
        "- Scoring remains deterministic exact matching against the fixed controlled vocabulary.",
        "",
        f"Rows: `{len(rows)}`",
        f"Seed: `{args.seed}`",
        f"Dose counts: `{dict(sorted(dose_counts.items()))}`",
        f"Post-final filler turns: `{args.post_final_filler_turns}`",
        f"Post-final distractor updates: `{args.post_final_distractor_updates}`",
        f"Factorial competition: `{getattr(args, 'factorial_competition', False)}`",
        f"Factorial target-current mode: `{getattr(args, 'factorial_target_current_mode', 'balanced')}`",
        f"Other-slot distance bins: `{getattr(args, 'other_distance_bins', None)}`",
        f"Extra interference turns: `{getattr(args, 'extra_interference_turns', 0)}`",
        f"Competition cell counts: `{dict(sorted(factorial_counts.items())) if factorial_counts else {}}`",
        f"Target-current distance-bin counts: `{dict(sorted(target_current_counts.items())) if target_current_counts else {}}`",
        f"Template source: `{args.template_source}`",
        f"OpenAI model for template generation: `{args.openai_model}`",
        f"Current binding count min/max over slot-value pairs: `{min(current_counts.values())}` / `{max(current_counts.values())}`",
        "",
        "## Natural Value Sets",
        "",
    ]
    for slot, values in slots.items():
        lines.append(f"- `{slot}` ({slot_label(slot)}): {', '.join(values)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_doses(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/stage_l/cicm_natural.jsonl")
    parser.add_argument("--datasheet", default="data/stage_l/cicm_natural_DATASHEET.md")
    parser.add_argument("--templates-in")
    parser.add_argument("--templates-out")
    parser.add_argument("--template-source", choices=["fallback", "openai"], default="fallback")
    parser.add_argument("--openai-model", default="gpt-4.1")
    parser.add_argument("--n-per-dose", type=int, default=210)
    parser.add_argument("--target-current-per-value", type=int)
    parser.add_argument("--doses", default="1,2,3,4,6")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--slot-limit", type=int)
    parser.add_argument("--post-final-filler-turns", type=int, default=4)
    parser.add_argument("--post-final-distractor-updates", type=int, default=2)
    parser.add_argument("--factorial-competition", action="store_true")
    parser.add_argument(
        "--factorial-target-current-mode",
        choices=["balanced", "far", "far_with_near_controls"],
        default="balanced",
    )
    parser.add_argument("--other-distance-bins", default="")
    parser.add_argument("--extra-interference-turns", type=int, default=0)
    args = parser.parse_args()

    slots = natural_controlled_slots()
    if args.slot_limit:
        slots = dict(list(slots.items())[: args.slot_limit])
    doses = parse_doses(args.doses)
    n_per_dose = args.n_per_dose
    if args.target_current_per_value:
        n_pairs = sum(len(values) for values in slots.values())
        n_per_dose = max(n_per_dose, (args.target_current_per_value * n_pairs + len(doses) - 1) // len(doses))
    packs = load_template_packs(
        slots=slots,
        template_source=args.template_source,
        templates_in=args.templates_in,
        templates_out=args.templates_out,
        openai_model=args.openai_model,
    )
    for pack in packs.values():
        pack["template_source"] = args.template_source
    rows = build_natural_stage_l_rows(
        n_per_dose=n_per_dose,
        doses=doses,
        seed=args.seed,
        slots=slots,
        template_packs=packs,
        post_final_filler_turns=args.post_final_filler_turns,
        post_final_distractor_updates=args.post_final_distractor_updates,
        factorial_competition=args.factorial_competition,
        factorial_target_current_mode=args.factorial_target_current_mode,
        other_distance_bins=[x for x in args.other_distance_bins.split(",") if x.strip()] or None,
        extra_interference_turns=args.extra_interference_turns,
    )
    write_jsonl(args.out, rows)
    write_datasheet(args.datasheet, rows, slots, args)
    print(
        json.dumps(
            {
                "out": args.out,
                "datasheet": args.datasheet,
                "templates_out": args.templates_out,
                "n": len(rows),
                "doses": doses,
                "n_per_dose": n_per_dose,
                "slots": len(slots),
                "slot_value_pairs": sum(len(v) for v in slots.values()),
                "factorial_competition": args.factorial_competition,
                "factorial_target_current_mode": args.factorial_target_current_mode,
                "other_distance_bins": args.other_distance_bins,
                "extra_interference_turns": args.extra_interference_turns,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
