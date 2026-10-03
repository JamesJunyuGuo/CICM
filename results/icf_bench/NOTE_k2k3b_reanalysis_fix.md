# K2/K3b — mechanism RE-ANALYSIS fix (mostly CPU; harvest already done)

The K2/K3 directions are encouraging and Paper-1-consistent, but four holes block a clean
"selection failure on real dialogue" claim. Fix them by RE-ANALYZING the already-harvested
activations (the 1:13:29 GPU harvest is done — do NOT re-generate). Only if the needed
hidden states were not saved, do a FORWARD-PASS-ONLY re-harvest over the saved local
generations (no re-generation). Write new artifacts under `results/icf_bench/mech/` with a
`_k2k3b` suffix; do not overwrite the prior run.

## Fix 1 (biggest) — do LENGTH CONTROL correctly, for every headline delta
The current per-group `length_residualized_*` centers EACH group at ~0 (residualized
within-group), which destroys the between-group signal — so the reported deltas (+1.216,
+0.041, +0.058) are RAW, uncontrolled for length. ICF-Bench conflates interference with
length, so this must be fixed.
- Correct procedure: pool ALL trials; fit ONE OLS `y ~ log(generation_length)` (do NOT
  include a group term); take residuals `r_i = y_i - ŷ_i`; then
  `delta = mean(r | within_stale) − mean(r | correct)` with paired/unpaired bootstrap CI.
- Report BOTH the raw delta and the length-controlled delta side by side, for: K3 NEW−OLD
  logit gap, K2 DP old_pref_ratio, K2 IF value_vs_instruction_ratio.

## Fix 2 — K3: use a TRAINED linear probe (Stage-A method), not the logit lens
The logit-lens-over-option-first-tokens is a questionable proxy for free-form (correct-trial
gap is non-significant and SMALLER than within-stale — counterintuitive; 31% dropped on
shared first token). Do the Stage-A probe the spec called for:
- From the saved residual-stream hidden states at the decision position, train a linear probe
  to decode NEW vs OLD preference (target from `new_op`/`old_op`), on correct + held-out
  trials, with GroupKFold CV + standardization.
- Then measure decodability of NEW on WITHIN-STALE trials.
- **Pre-registered reading:** NEW decodable on within-stale above the shuffle null (present
  but not selected) → SELECTION failure replicates Paper 1 on real dialogue. Not decodable →
  RETENTION failure. Keep the logit-lens as a secondary check and note the ordering anomaly.

## Fix 3 — shuffle nulls on every headline
Permute the group labels (within-stale vs correct) ≥1000×; recompute each delta / probe
metric; report the null band. Headline only if the observed exceeds the null.

## Fix 4 — IF reporting hygiene
- Report value_vs_instruction as ONE number; drop the redundant complementary
  `instruction_ratio` delta (it is −1× the value_ratio by construction, not a second finding).
- Flag the `correct_forget` baseline is only n=14 explicitly; if a cleaner/larger correct
  baseline exists (e.g. the tiny correct set is unavoidable since Qwen almost never forgets),
  state the limitation. Apply Fix 1 (length control) + Fix 3 (shuffle null) here too.

## Pre-registered adjudication (quote verbatim)
- Length-controlled selection signal survives (probe decodes NEW on within-stale above the
  shuffle null, length-controlled) → **selection-not-retention replicates on real dialogue**
  (clean headline); the stale-attention deltas that survive length control + shuffle null
  support the signature.
- Signal vanishes after length control / falls into the shuffle null → the raw effect was
  length-driven / not robust; report honestly (a clean, informative negative — grounding-
  style length artifact, consistent with Stage J's lesson).
- Either outcome is final for this round; do not add analyses post hoc to rescue a sign.

## Guardrails
- Condition on failure MODE (within-stale vs correct); pool-then-residualize for length;
  GroupKFold CV + standardization for the probe; shuffle nulls + paired bootstrap on every
  headline; report raw AND length-controlled.
- Reuse saved activations (CPU); forward-pass-only re-harvest over saved generations if
  hidden states are missing (no re-generation, no new sampling).
- New `src/icf_*` files; write `results/icf_bench/mech/*_k2k3b*`; do not touch Stage-J or the
  frozen K1d/K2K3 prior artifacts. Update `results/icf_bench/mech/REPORT.md` with a K2K3b
  section quoting the readings.
