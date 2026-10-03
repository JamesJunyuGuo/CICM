# Stage L Report
## L-1 Mechanism

- Current value decodable on within-stale trials (present but not selected) → **SELECTION failure replicates on realistic-controlled data** — the mechanism ICF-Bench could not identify (headline mechanism result).
- Current value decodable on cross-slot trials → **recency-over-identity selection failure**: the target-slot value is represented, but a recent other-slot value is selected.
- Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.

Analysis discipline applied here: per-cell rows are the result; aggregate rows are included only as equal-cell-size diagnostics and are not prevalence claims. L-0 crossover is treated as a conditioned tie/recency-gradient, not a clean global reversal.

- Stage-A probe reuse: `protocol_reused_weights_not_loadable`. No serialized Stage-A linear probe weights exist in results/overwrite_attention_extraction, and the Stage-A synthetic label space is incompatible with Stage-L natural values. L-1 therefore freezes the Stage-A probe protocol: decision-position residual stream, linear multinomial probe, standardization, GroupKFold, and shuffle null.
- Label counts: `{'within_stale': 8, 'correct_current': 9, 'other': 2, 'cross_slot': 1}`
- Factorial cell counts: `{'same_near__other_far': 4, 'same_near__other_mid': 4, 'same_near__other_near2': 3, 'same_far__other_far': 3, 'same_far__other_mid': 3, 'same_far__other_near2': 3}`
- Probe layer: 24; splitter: `GroupKFold(n_splits=5)`; n_classes=20; CV accuracy=0.0000.
- Length control: `probe_true_current_score ~ log(prompt_tokens)` fit once on `all_l1_rows` with n=0.
- Within-stale true-current score: nan; value-label shuffle95 [0.004462857386083236, 0.2246637229110117]; above null: `False`.
- Probe true-current score within_stale - correct_current: raw nan CI [nan, nan], length-controlled nan CI [nan, nan], shuffle95 [nan, nan].
- Attention stale/(stale+current) within_stale - correct_current: raw 0.1363 CI [0.0318, 0.2526], length-controlled 0.1351 CI [0.0413, 0.2704], shuffle95 [-0.11326184115043078, 0.1125923839655384].
- L-1 status: `retention`.
- L-1 gate pass: `False`.
- Pre-registered reading selected: Current value NOT decodable (lost) → RETENTION failure; differs from synthetic; report.

### Within-Stale Failure Pool

- Overall diagnostic: n=0, true-current score=nan, above value-label shuffle95=no.

| same-slot stale | other-slot distance | labels in cell | fail n | true-current score | above shuffle95 | raw delta vs correct | length-controlled delta vs correct | length-null95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| far | far | `{'correct_current': 2, 'cross_slot': 1}` | 0 | nan | no | nan | nan | [nan, nan] |
| far | mid | `{'correct_current': 3}` | 0 | nan | no | nan | nan | [nan, nan] |
| far | near2 | `{'correct_current': 2, 'other': 1}` | 0 | nan | no | nan | nan | [nan, nan] |
| near | far | `{'within_stale': 3, 'correct_current': 1}` | 0 | nan | no | nan | nan | [nan, nan] |
| near | mid | `{'correct_current': 1, 'within_stale': 3}` | 0 | nan | no | nan | nan | [nan, nan] |
| near | near2 | `{'within_stale': 2, 'other': 1}` | 0 | nan | no | nan | nan | [nan, nan] |

### Cross-Slot Failure Pool

- Overall diagnostic: n=0, true-current score=nan, above value-label shuffle95=no.

| same-slot stale | other-slot distance | labels in cell | fail n | true-current score | above shuffle95 | raw delta vs correct | length-controlled delta vs correct | length-null95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| far | far | `{'correct_current': 2, 'cross_slot': 1}` | 0 | nan | no | nan | nan | [nan, nan] |
| far | mid | `{'correct_current': 3}` | 0 | nan | no | nan | nan | [nan, nan] |
| far | near2 | `{'correct_current': 2, 'other': 1}` | 0 | nan | no | nan | nan | [nan, nan] |
| near | far | `{'within_stale': 3, 'correct_current': 1}` | 0 | nan | no | nan | nan | [nan, nan] |
| near | mid | `{'correct_current': 1, 'within_stale': 3}` | 0 | nan | no | nan | nan | [nan, nan] |
| near | near2 | `{'within_stale': 2, 'other': 1}` | 0 | nan | no | nan | nan | [nan, nan] |
