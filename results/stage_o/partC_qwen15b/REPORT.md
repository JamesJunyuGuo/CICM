# Stage O Part C: Qwen2.5-1.5B mechanism-shape replication

Pre-registered verdict: **mechanism_shape_replication_partial**.

Tests cross-family mechanism shape only: selection, compact causal recovery, specific stale-promoting ablation, and QK-shift/OV-preserved. It does not claim head-for-head correspondence with Pythia.

## Behavior and integrity gates

The identical Part-D full behavior run was reused without new inference (SHA-256 recorded in `behavior_reuse.summary.json`). At fixed k=6, the pool contains 352 correct and 296 within-stale rows; the matched mechanism pool contains 233 per class.

The 1.5B hooked no-op changes logits by exactly 0.0 at gamma=1 and actively changes them by 0.233 at gamma=0. Pair construction yielded 96 exact clean/corrupt pairs, balanced across surface template and discovery/held-out split.

## Selection probe

On within-stale failures, the grouped out-of-sample current-value score is 0.236; its value-label shuffle 95% interval is [0.012, 0.059]. The length-controlled failure-minus-correct delta is -0.020.

## Exhaustive query-head patching and held-out ablation

Discovery ranking selects 23 of 336 query heads. Their held-out recovery is 0.825 [0.786, 0.861]. The identity patch has maximum absolute gap change 0.0 and 0 prediction changes.

The discovery-ranked stale-promoting set raises held-out correction by 0.286; the 64 layer-matched random-set 95% range is [0.085, 0.217].

## QK/OV and induction anchor

Qwen uses grouped-query attention: each row below is a query head paired with its shared KV head.

| Query head | Shared KV | QK margin correct | QK margin stale | Current OV correct | Current OV stale | Shape |
|---|---:|---:|---:|---:|---:|---|
| L19H3 | 0 | +0.381 | +0.061 | +0.480 | +0.414 | QK shift + OV preserved |
| L19H1 | 0 | -0.778 | -1.026 | +0.419 | +0.354 | QK shift + OV preserved |
| L22H7 | 1 | -0.109 | -0.748 | +2.081 | +1.783 | QK shift + OV preserved |

The repeated-token induction anchor is **PASS**.

## Pre-registered readings

- hooked_noop_exact_identity: **PASS**
- exact_pair_power: **PASS**
- selection_component: **PASS**
- compact_recovery_set: **FAIL**
- heldout_stale_ablation_specific: **PASS**
- qk_shift_with_ov_preserved: **PASS**

Partial outcomes are retained as partial; no held-out reranking was performed.
