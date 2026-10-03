# Stage K Mechanism Report

## Scope

K2/K3 use local `Qwen/Qwen2.5-7B-Instruct` GPU harvests with fp32 eager attention. Outputs are isolated under `results/icf_bench/mech/`; Stage-J and K1d artifacts were not modified.

## K3 DP Retention vs Selection

Frozen Stage-A-style linear readout: final-token hidden states are passed through the model final norm and unembedding; the measured gap is NEW aligned choice first-token logit minus OLD aligned choice first-token logit.

- n kept after shared-first-token drop: 543 / 783
- correct_current: n=397, final NEW-OLD gap=0.353, CI=[-0.108, 0.821]
- within_stale: n=127, final NEW-OLD gap=1.216, CI=[0.378, 2.054]
- other: n=19, final NEW-OLD gap=-0.231, CI=[-2.109, 1.683]

Pre-registered reading: NEW preference still linearly decodable on within-stale trials -> SELECTION failure replicates; NEW not decodable -> RETENTION/retrievability failure.

## K2 DP Attention

- Labels: {'correct_current': 564, 'within_stale': 183, 'other': 36}
- OLD/(OLD+NEW) within_stale minus correct final-layer delta: 0.0411

## K2 IF Attention

- n total: 931; value span found: 930
- Labels all: {'deflect': 17, 'within_stale': 862, 'other': 38, 'correct_forget': 14}
- forbidden-value ratio within_stale minus correct_forget final-layer delta: 0.0580
- forget-instruction ratio within_stale minus correct_forget final-layer delta: -0.0580

## Artifacts

- `dynamic_preference.npz` / `dynamic_preference_index.jsonl`
- `instructional_forgetting.npz` / `instructional_forgetting_index.jsonl`
- `dynamic_preference_local_labeled.jsonl`
- `k2k3_summary.json`

## K2/K3b Reanalysis Fix

This section reuses the saved local Qwen2.5-7B fp32-eager harvest. No regeneration was run. Length control is pooled across the full valid pool before comparing failure modes.

Pre-registered adjudication:

> Length-controlled selection signal survives (probe decodes NEW on within-stale above the shuffle null, length-controlled) → **selection-not-retention replicates on real dialogue** (clean headline); the stale-attention deltas that survive length control + shuffle null support the signature.

> Signal vanishes after length control / falls into the shuffle null → the raw effect was length-driven / not robust; report honestly (a clean, informative negative — grounding-style length artifact, consistent with Stage J's lesson).

### K3 Trained Linear Probe

- Scope: saved-activation feasible behavior-selection proxy, not a strict option-id probe. Each DP old/new pair appears once, so the stricter candidate-id NEW-vs-OLD probe is not identifiable from the current saved decision-position activations without a forward-pass-only supplement.
- Candidate-index diagnostic: new_unique=783, old_unique=783, pair_unique=783, max_new_count=1, max_old_count=1.
- Final-layer GroupKFold behavior-selection probe AUC (NEW-selected vs OLD-selected behavior): 0.691.
- Mean p(NEW) correct_current: 0.761; within_stale: 0.504.
- Within-stale p(NEW): mean 0.504 CI [0.441, 0.564], shuffle95 [0.6500882230376547, 0.7463450543861716].
- Within-stale p(NEW) above shuffle-null 95%: `False`.
- Probe p(NEW) within_stale - correct_current: raw -0.2574 CI [-0.3236, -0.1896], length-controlled -0.2505 CI [-0.3190, -0.1789], shuffle95 [-0.06601553099681778, 0.06243080971163455].
- K2/K3b verdict: trained-probe selection headline does NOT pass; report this as a mixed/negative mechanism result.

### K3 Logit-Lens Auxiliary

- NEW-OLD first-token gap within_stale - correct_current: raw 0.8622 CI [-0.0932, 1.7958], length-controlled 0.8029 CI [-0.1829, 1.7723], shuffle95 [-0.9617868374806604, 0.9323651379312959].
- Shared-first-token dropped: 240.

### K2 DP Attention

- OLD/(OLD+NEW) attention ratio within_stale - correct_current: raw 0.0411 CI [0.0265, 0.0546], length-controlled 0.0408 CI [0.0266, 0.0555], shuffle95 [-0.013893718032949677, 0.01425885618139589].
- DP stale-attention support after length control + shuffle null: `True`.

### K2 IF Attention

- Value/(value+forget-instruction) ratio within_stale - correct_forget: raw 0.0580 CI [-0.0052, 0.1101], length-controlled 0.0673 CI [-0.0004, 0.1245], shuffle95 [-0.07054196559406083, 0.0687590288819112].
- Correct-forget baseline n=14 (small; Qwen almost never truly forgets).
- IF attention support after length control + shuffle null: `False`.

### Artifacts

- `k2k3b_summary.json`
- `k2k3b_probe_rows.jsonl`
