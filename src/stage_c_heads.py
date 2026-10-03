"""Head-list utilities for Stage C target and random controls."""

import json
import random

import numpy as np

from analyze_a1_attention import compute_p_last


def ranked_heads_from_stage_a(npz_path, index_path):
    rows = [json.loads(line) for line in open(index_path)]
    data = np.load(npz_path)
    attn_writes = data["attn_writes"]
    write_counts = np.array([row["write_count"] for row in rows], dtype=np.int64)
    p_last = compute_p_last(attn_writes, write_counts)
    correct = np.array([bool(row["correct"]) for row in rows])
    heatmap = p_last[correct].mean(axis=0)
    n_layers, n_heads = heatmap.shape
    flat = heatmap.reshape(-1)
    ranked = []
    for idx in np.argsort(flat)[::-1]:
        ranked.append(
            {
                "layer": int(idx // n_heads),
                "head": int(idx % n_heads),
                "mean_p_last_correct": float(flat[idx]),
            }
        )
    return ranked


def target_heads(a1_summary_path, top_k=8, num_layers=None, num_heads=None):
    with open(a1_summary_path) as f:
        summary = json.load(f)
    heads = []
    for item in summary["top_heads_by_correct_p_last"]:
        layer = int(item["layer"])
        head = int(item["head"])
        if num_layers is not None and (layer >= num_layers or head >= num_heads):
            continue
        heads.append(
            {
                "layer": layer,
                "head": head,
                "mean_p_last_correct": item.get("mean_p_last_correct"),
            }
        )
        if len(heads) >= top_k:
            break
    return heads


def random_heads_outside_top(
    ranked_heads,
    seed,
    top_exclude=32,
    k=8,
    num_layers=None,
    num_heads=None,
):
    excluded = {(h["layer"], h["head"]) for h in ranked_heads[:top_exclude]}
    candidates = []
    if num_layers is not None and num_heads is not None:
        all_pairs = [
            {"layer": layer, "head": head}
            for layer in range(num_layers)
            for head in range(num_heads)
        ]
    else:
        all_pairs = ranked_heads
    for item in all_pairs:
        pair = (int(item["layer"]), int(item["head"]))
        if pair not in excluded:
            candidates.append({"layer": pair[0], "head": pair[1]})
    rng = random.Random(seed)
    return rng.sample(candidates, k)
