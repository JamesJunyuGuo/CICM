# Stage L Report
## L-1 Mechanism

- Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result).
- Current value decodable on cross-slot trials → **recency-over-identity selection failure**: the target-slot value is represented, but a recent other-slot value is selected.
- Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.

Analysis discipline applied here: per-cell rows are the result; aggregate rows are included only as equal-cell-size diagnostics and are not prevalence claims. L-0 crossover is treated as a conditioned tie/recency-gradient, not a clean global reversal.

- Stage-A probe reuse: `protocol_reused_weights_not_loadable`. No serialized Stage-A linear probe weights exist in results/overwrite_attention_extraction, and the Stage-A synthetic label space is incompatible with Stage-L natural values. L-1 therefore freezes the Stage-A probe protocol: decision-position residual stream, linear multinomial probe, standardization, GroupKFold, and shuffle null.
- Label counts: `{'within_stale': 634, 'correct_current': 458, 'cross_slot': 88, 'other': 20}`
- Factorial cell counts: `{'same_near__other_far': 200, 'same_near__other_mid': 200, 'same_near__other_near2': 200, 'same_far__other_far': 200, 'same_far__other_mid': 200, 'same_far__other_near2': 200}`
- Probe layer: 32; splitter: `GroupKFold(n_splits=5)`; n_classes=21; CV accuracy=0.8967.
- Length control: `probe_true_current_score ~ log(prompt_tokens)` fit once on `all_l1_rows` with n=1200.
- Within-stale true-current score: 0.8220; value-label shuffle95 [0.03214537198819564, 0.06352049026177288]; above null: `True`.
- Probe true-current score within_stale - correct_current: raw -0.0904 CI [-0.1197, -0.0612], length-controlled -0.0732 CI [-0.1014, -0.0422], shuffle95 [-0.02921991409445593, 0.032633713904033745].
- Attention stale/(stale+current) within_stale - correct_current: raw 0.0626 CI [0.0428, 0.0830], length-controlled 0.0116 CI [-0.0047, 0.0281], shuffle95 [-0.01648651211006095, 0.016859855471849098].
- L-1 status: `selection`.
- L-1 gate pass: `True`.
- Pre-registered reading selected: Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result). Nuance: true-current probe score is lower on within-stale than correct trials after length control, and stale-token attention is strongly elevated, so Stage L shows selection failure with a small retention/retrieval degradation rather than a pure no-loss story.

### Within-Stale Failure Pool

- Overall diagnostic: n=634, true-current score=0.8220, above value-label shuffle95=yes.

| same-slot stale | other-slot distance | labels in cell | fail n | true-current score | above shuffle95 | raw delta vs correct | length-controlled delta vs correct | length-null95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| far | far | `{'correct_current': 131, 'other': 14, 'within_stale': 30, 'cross_slot': 25}` | 30 | 0.9100 | yes | -0.0104 | 0.0052 | [-0.0875, 0.0709] |
| far | mid | `{'within_stale': 69, 'correct_current': 102, 'cross_slot': 29}` | 69 | 0.8444 | yes | -0.0487 | -0.0361 | [-0.0756, 0.0697] |
| far | near2 | `{'correct_current': 116, 'within_stale': 74, 'cross_slot': 10}` | 74 | 0.6988 | yes | -0.2053 | -0.1921 | [-0.0818, 0.0774] |
| near | far | `{'within_stale': 171, 'correct_current': 27, 'other': 2}` | 171 | 0.8826 | yes | -0.0377 | -0.0011 | [-0.0939, 0.1142] |
| near | mid | `{'within_stale': 114, 'cross_slot': 23, 'correct_current': 60, 'other': 3}` | 114 | 0.7859 | yes | -0.1409 | -0.1222 | [-0.0875, 0.0896] |
| near | near2 | `{'correct_current': 22, 'within_stale': 176, 'other': 1, 'cross_slot': 1}` | 176 | 0.8144 | yes | -0.1339 | -0.1241 | [-0.1177, 0.1478] |

### Cross-Slot Failure Pool

- Overall diagnostic: n=88, true-current score=0.8628, above value-label shuffle95=yes.

| same-slot stale | other-slot distance | labels in cell | fail n | true-current score | above shuffle95 | raw delta vs correct | length-controlled delta vs correct | length-null95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| far | far | `{'correct_current': 131, 'other': 14, 'within_stale': 30, 'cross_slot': 25}` | 25 | 0.9578 | yes | 0.0374 | 0.0419 | [-0.0866, 0.0689] |
| far | mid | `{'within_stale': 69, 'correct_current': 102, 'cross_slot': 29}` | 29 | 0.8699 | yes | -0.0232 | -0.0169 | [-0.1076, 0.0935] |
| far | near2 | `{'correct_current': 116, 'within_stale': 74, 'cross_slot': 10}` | 10 | 0.8539 | yes | -0.0502 | -0.0284 | [-0.1491, 0.1013] |
| near | far | `{'within_stale': 171, 'correct_current': 27, 'other': 2}` | 0 | nan | no | nan | nan | [nan, nan] |
| near | mid | `{'within_stale': 114, 'cross_slot': 23, 'correct_current': 60, 'other': 3}` | 23 | 0.7484 | yes | -0.1784 | -0.1512 | [-0.1235, 0.1068] |
| near | near2 | `{'correct_current': 22, 'within_stale': 176, 'other': 1, 'cross_slot': 1}` | 1 | 0.9998 | no | 0.0515 | 0.0719 | [-0.8486, 0.0959] |
