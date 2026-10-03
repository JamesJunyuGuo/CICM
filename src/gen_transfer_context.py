"""Stage D broader contextual-management task generators.

These generators are controlled, metadata-rich analogues of external long-context
benchmarks. They do not claim to reproduce official RULER/BABILong/Michelangelo
splits; they prepare compatible probes for state tracking, tracing, aggregation,
latent lists, and coreference while preserving current/stale span metadata for
later Stage A/B-style analysis.
"""

import random
import string


NAMES = [
    "Avery",
    "Blair",
    "Casey",
    "Devon",
    "Emery",
    "Finley",
    "Gray",
    "Harper",
    "Indigo",
    "Jordan",
    "Kai",
    "Logan",
]

OBJECTS = [
    "budget",
    "server",
    "ticket",
    "route",
    "sample",
    "dataset",
    "shipment",
    "policy",
]

PLACES = ["Oslo", "Lima", "Seoul", "Cairo", "Dublin", "Quito", "Riga", "Perth"]
COLORS = ["red", "blue", "green", "amber", "violet", "silver", "black", "white"]


def _value(rng, lo=100, hi=9999):
    return rng.randint(lo, hi)


def _distinct_values(rng, n, lo=100, hi=9999):
    vals = set()
    while len(vals) < n:
        vals.add(_value(rng, lo, hi))
    return list(vals)


def _span(line, value, role, entity=None, attribute=None):
    return {
        "line": int(line),
        "value": value,
        "role": role,
        "entity": entity,
        "attribute": attribute,
    }


def _row(
    *,
    task_family,
    stage_d_cell,
    context_lines,
    question,
    gold,
    answer_type,
    current_spans,
    stale_spans=None,
    stale_values=None,
    condition="interference",
    target_entity=None,
    target_attribute=None,
    metadata=None,
):
    prompt = "\n".join(context_lines)
    prompt = f"{prompt}\n\nQuestion: {question}\nAnswer with only the requested value."
    return {
        "task_family": task_family,
        "stage_d_cell": stage_d_cell,
        "condition": condition,
        "prompt": prompt,
        "context_lines": list(context_lines),
        "question": question,
        "gold": gold,
        "answer_type": answer_type,
        "current_spans": list(current_spans),
        "stale_spans": list(stale_spans or []),
        "stale_values": list(stale_values or []),
        "target_entity": target_entity,
        "target_attribute": target_attribute,
        "metadata": metadata or {},
    }


def make_ruler_retrieval(rng, n_distractors=24):
    key = f"needle_{rng.choice(string.ascii_lowercase)}{rng.randint(10, 99)}"
    gold = _value(rng)
    lines = []
    insert_at = rng.randrange(2, n_distractors - 2)
    for idx in range(n_distractors):
        if idx == insert_at:
            lines.append(f"Record {idx}: The value for {key} is {gold}.")
        else:
            lines.append(f"Record {idx}: The value for decoy_{idx} is {_value(rng)}.")
    return _row(
        task_family="ruler_style",
        stage_d_cell="retrieval",
        context_lines=lines,
        question=f"What is the value for {key}?",
        gold=gold,
        answer_type="int",
        current_spans=[_span(insert_at, gold, "current", key, "value")],
        condition="retrieval",
        target_entity=key,
        target_attribute="value",
    )


def make_ruler_multi_hop(rng, hops=4, n_distractors=20):
    nodes = [f"N{rng.randint(10, 99)}" for _ in range(hops + 1)]
    lines = [f"Trace rule: {nodes[i]} points to {nodes[i + 1]}." for i in range(hops)]
    for idx in range(n_distractors):
        lines.append(f"Distractor rule: D{idx} points to D{idx + 1}.")
    rng.shuffle(lines)
    current_line = next(i for i, line in enumerate(lines) if line.endswith(f"{nodes[-1]}."))
    return _row(
        task_family="ruler_style",
        stage_d_cell="multi_hop_tracing",
        context_lines=lines,
        question=f"Starting at {nodes[0]} and following all Trace rules, where do you end?",
        gold=nodes[-1],
        answer_type="string",
        current_spans=[_span(current_line, nodes[-1], "current", nodes[0], "endpoint")],
        condition="multi_hop",
        target_entity=nodes[0],
        target_attribute="endpoint",
        metadata={"path": nodes},
    )


def make_ruler_aggregation(rng, n_items=7, n_distractors=12):
    target = f"group_{rng.choice(string.ascii_lowercase)}"
    vals = _distinct_values(rng, n_items, 1, 20)
    lines = [f"Item {idx}: {target} contributes {val}." for idx, val in enumerate(vals)]
    for idx in range(n_distractors):
        lines.append(f"Item x{idx}: other_group contributes {rng.randint(1, 20)}.")
    rng.shuffle(lines)
    current_spans = [
        _span(i, int(line.rsplit(" ", 1)[-1].rstrip(".")), "current", target, "contribution")
        for i, line in enumerate(lines)
        if f"{target} contributes" in line
    ]
    return _row(
        task_family="ruler_style",
        stage_d_cell="aggregation",
        context_lines=lines,
        question=f"What is the sum of all contributions for {target}?",
        gold=sum(vals),
        answer_type="int",
        current_spans=current_spans,
        condition="aggregation",
        target_entity=target,
        target_attribute="sum",
        metadata={"values": vals},
    )


def make_babilong_fact_chain(rng, hops=3):
    people = rng.sample(NAMES, hops + 1)
    lines = [f"{people[i]} handed the token to {people[i + 1]}." for i in range(hops)]
    lines += [f"{name} checked a blue folder." for name in rng.sample(NAMES, 3)]
    rng.shuffle(lines)
    current_line = next(i for i, line in enumerate(lines) if people[-1] in line)
    return _row(
        task_family="babilong_style",
        stage_d_cell="fact_chaining",
        context_lines=lines,
        question=f"If the token starts with {people[0]}, who has it after all handoffs?",
        gold=people[-1],
        answer_type="string",
        current_spans=[_span(current_line, people[-1], "current", "token", "holder")],
        condition="chain",
        target_entity="token",
        target_attribute="holder",
        metadata={"chain": people},
    )


def make_babilong_counting(rng, n_events=12):
    target = rng.choice(OBJECTS)
    total = 0
    lines = []
    current_spans = []
    for idx in range(n_events):
        amount = rng.randint(1, 5)
        if rng.random() < 0.65:
            total += amount
            current_spans.append(_span(idx, amount, "current", target, "count_delta"))
            lines.append(f"Log {idx}: add {amount} {target} units.")
        else:
            lines.append(f"Log {idx}: add {amount} unrelated units.")
    return _row(
        task_family="babilong_style",
        stage_d_cell="counting",
        context_lines=lines,
        question=f"How many {target} units were added in total?",
        gold=total,
        answer_type="int",
        current_spans=current_spans,
        condition="counting",
        target_entity=target,
        target_attribute="count",
    )


def make_babilong_list_set(rng, n_ops=10):
    bag = rng.choice(OBJECTS)
    items = rng.sample(COLORS + PLACES, 6)
    present = []
    lines = []
    current_spans = []
    stale_spans = []
    stale_values = []
    for idx in range(n_ops):
        item = rng.choice(items)
        if item not in present or rng.random() < 0.6:
            if item not in present:
                present.append(item)
            current_spans.append(_span(idx, item, "current", bag, "member"))
            lines.append(f"Step {idx}: add {item} to the {bag} list.")
        else:
            present.remove(item)
            stale_spans.append(_span(idx, item, "stale", bag, "member"))
            stale_values.append(item)
            lines.append(f"Step {idx}: remove {item} from the {bag} list.")
    gold = sorted(present)
    return _row(
        task_family="babilong_style",
        stage_d_cell="list_set",
        context_lines=lines,
        question=f"Which items are currently in the {bag} list? Return a comma-separated list.",
        gold=gold,
        answer_type="list",
        current_spans=current_spans,
        stale_spans=stale_spans,
        stale_values=stale_values,
        condition="list_set",
        target_entity=bag,
        target_attribute="members",
    )


def make_michelangelo_coreference(rng, rounds=8):
    codename = f"agent-{rng.choice(string.ascii_uppercase)}"
    holders = rng.sample(NAMES, min(rounds, len(NAMES)))
    lines = []
    stale_spans = []
    stale_values = []
    for idx, holder in enumerate(holders):
        if idx == 0:
            lines.append(f"Round {idx}: {codename} refers to {holder}.")
        else:
            stale_spans.append(_span(idx - 1, holders[idx - 1], "stale", codename, "referent"))
            stale_values.append(holders[idx - 1])
            lines.append(f"Round {idx}: From now on, {codename} refers to {holder}.")
    return _row(
        task_family="michelangelo_style",
        stage_d_cell="multi_round_coreference",
        context_lines=lines,
        question=f"At the end, who does {codename} refer to?",
        gold=holders[-1],
        answer_type="string",
        current_spans=[_span(len(lines) - 1, holders[-1], "current", codename, "referent")],
        stale_spans=stale_spans,
        stale_values=stale_values,
        condition="coreference",
        target_entity=codename,
        target_attribute="referent",
    )


def make_michelangelo_latent_list(rng, n_ops=12):
    board = f"board-{rng.randint(10, 99)}"
    items = rng.sample(COLORS + PLACES + OBJECTS, 8)
    active = []
    lines = []
    stale_spans = []
    stale_values = []
    current_spans = []
    for idx in range(n_ops):
        item = rng.choice(items)
        op = rng.choice(["append", "replace", "delete"])
        if op == "append" or not active:
            if item not in active:
                active.append(item)
            current_spans.append(_span(idx, item, "current", board, "item"))
            lines.append(f"Operation {idx}: append {item} to {board}.")
        elif op == "delete":
            victim = rng.choice(active)
            active.remove(victim)
            stale_spans.append(_span(idx, victim, "stale", board, "item"))
            stale_values.append(victim)
            lines.append(f"Operation {idx}: delete {victim} from {board}.")
        else:
            victim = rng.choice(active)
            pos = active.index(victim)
            active[pos] = item
            stale_spans.append(_span(idx, victim, "stale", board, "item"))
            stale_values.append(victim)
            current_spans.append(_span(idx, item, "current", board, "item"))
            lines.append(f"Operation {idx}: replace {victim} with {item} on {board}.")
    return _row(
        task_family="michelangelo_style",
        stage_d_cell="latent_list",
        context_lines=lines,
        question=f"What is the final ordered list on {board}? Return comma-separated items.",
        gold=active,
        answer_type="list",
        current_spans=current_spans,
        stale_spans=stale_spans,
        stale_values=stale_values,
        condition="latent_list",
        target_entity=board,
        target_attribute="ordered_items",
    )


def make_state_entity(rng, n_updates=6):
    entity = rng.choice(NAMES)
    attribute = rng.choice(["location", "status", "owner"])
    values = rng.sample(PLACES + COLORS + OBJECTS, n_updates)
    lines = [f"Update {idx}: {entity}'s {attribute} is now {val}." for idx, val in enumerate(values)]
    return _row(
        task_family="state_suite",
        stage_d_cell="entity_state",
        context_lines=lines,
        question=f"What is {entity}'s current {attribute}?",
        gold=values[-1],
        answer_type="string",
        current_spans=[_span(len(lines) - 1, values[-1], "current", entity, attribute)],
        stale_spans=[_span(i, val, "stale", entity, attribute) for i, val in enumerate(values[:-1])],
        stale_values=values[:-1],
        target_entity=entity,
        target_attribute=attribute,
    )


def make_state_multi_attribute(rng, n_updates=10):
    entity = rng.choice(OBJECTS)
    attributes = ["owner", "location", "priority"]
    latest = {}
    current_line_by_attr = {}
    lines = []
    stale_spans = []
    stale_values = []
    for idx in range(n_updates):
        attr = rng.choice(attributes)
        val = rng.choice(NAMES if attr == "owner" else PLACES if attr == "location" else COLORS)
        if attr in latest:
            stale_spans.append(_span(current_line_by_attr[attr], latest[attr], "stale", entity, attr))
            stale_values.append(latest[attr])
        latest[attr] = val
        current_line_by_attr[attr] = idx
        lines.append(f"Update {idx}: {entity}.{attr} = {val}.")
    target_attr = rng.choice(sorted(latest))
    return _row(
        task_family="state_suite",
        stage_d_cell="multi_attribute",
        context_lines=lines,
        question=f"What is the current {target_attr} of {entity}?",
        gold=latest[target_attr],
        answer_type="string",
        current_spans=[
            _span(current_line_by_attr[target_attr], latest[target_attr], "current", entity, target_attr)
        ],
        stale_spans=[s for s in stale_spans if s["attribute"] == target_attr],
        stale_values=[s["value"] for s in stale_spans if s["attribute"] == target_attr],
        target_entity=entity,
        target_attribute=target_attr,
    )


def make_state_implicit_invalidation(rng):
    entity = rng.choice(OBJECTS)
    old_owner, new_owner = rng.sample(NAMES, 2)
    old_place, new_place = rng.sample(PLACES, 2)
    lines = [
        f"Morning note: {entity} was assigned to {old_owner} in {old_place}.",
        f"Later note: {new_owner} took responsibility for {entity} after the handoff.",
        f"Routing note: anything handled by {new_owner} is processed in {new_place}.",
    ]
    return _row(
        task_family="state_suite",
        stage_d_cell="implicit_invalidation",
        context_lines=lines,
        question=f"Who is currently responsible for {entity}?",
        gold=new_owner,
        answer_type="string",
        current_spans=[_span(1, new_owner, "current", entity, "owner")],
        stale_spans=[_span(0, old_owner, "stale", entity, "owner")],
        stale_values=[old_owner],
        condition="implicit_invalidation",
        target_entity=entity,
        target_attribute="owner",
        metadata={"old_place": old_place, "new_place": new_place},
    )


def make_state_parallel_entities(rng, n_entities=4, updates_per_entity=4):
    entities = rng.sample(OBJECTS + NAMES, n_entities)
    latest = {}
    lines = []
    stale_spans = []
    stale_values = []
    line_for = {}
    schedule = [entity for entity in entities for _ in range(updates_per_entity)]
    rng.shuffle(schedule)
    for idx, entity in enumerate(schedule):
        val = _value(rng)
        if entity in latest:
            stale_spans.append(_span(line_for[entity], latest[entity], "stale", entity, "value"))
            stale_values.append(latest[entity])
        latest[entity] = val
        line_for[entity] = idx
        lines.append(f"Stream update {idx}: {entity} has value {val}.")
    return _row(
        task_family="state_suite",
        stage_d_cell="parallel_entities",
        context_lines=lines,
        question="What is the current value of every tracked entity? Return entity=value pairs.",
        gold={entity: latest[entity] for entity in entities},
        answer_type="dict",
        current_spans=[
            _span(line_for[entity], latest[entity], "current", entity, "value")
            for entity in entities
        ],
        stale_spans=stale_spans,
        stale_values=stale_values,
        condition="parallel_entities",
        metadata={"entities": entities},
    )


GENERATORS = [
    make_ruler_retrieval,
    make_ruler_multi_hop,
    make_ruler_aggregation,
    make_babilong_fact_chain,
    make_babilong_counting,
    make_babilong_list_set,
    make_michelangelo_coreference,
    make_michelangelo_latent_list,
    make_state_entity,
    make_state_multi_attribute,
    make_state_implicit_invalidation,
    make_state_parallel_entities,
]


def make_all_stage_d_examples(n_per_cell, seed):
    rng = random.Random(seed)
    rows = []
    uid = 0
    for make in GENERATORS:
        for _ in range(n_per_cell):
            row = make(rng)
            row["id"] = uid
            row["split"] = "eval"
            uid += 1
            rows.append(row)
    return rows
