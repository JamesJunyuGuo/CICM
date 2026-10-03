# Stage N cross-scale descriptive mechanism transfer

Status: frozen before any 410M/1B/2.8B inference.

## Question

Does the descriptive mechanism found in Pythia-160M transfer within the Pythia
family when only parameter scale changes: current-value information remains
decodable on stale failures, while retrieval heads shift attention and QK score
from the current write toward stale writes?

This extension does not search for a minimal circuit, establish head identity
across model sizes, or claim that the 160M causal circuit itself is conserved.

## Frozen task and behavior gate

- Task bytes are frozen to
  `results/stage_n/behavior_full_cpu/tasks.jsonl`, SHA256
  `22e79200d953cd347c3c1ea3ae2f94d2634a45ef70e8e37549f6fa80cafc2fdf`.
- Models: final checkpoints of `EleutherAI/pythia-160m` (reference),
  `pythia-410m`, `pythia-1b`, and `pythia-2.8b`.
- Greedy one-token full-vocabulary decoding, float32 behavior inference, the
  same tokenizer-derived token IDs, templates, seeds, variants, and `k=0..5`.
- Mechanism substrate is fixed at `k=4`, selected by Phase 1 on 160M. It is not
  retuned per scale.
- A scale enters mechanism analysis only if its `k=4` single-variable rows have
  at least 150 correct and 120 within-stale responses, with at least 25 of each
  label in every one of the three surface templates. Its matched pool is
  balanced within template and seed with the existing cap of 48 per stratum.
- Failure of this gate means the fixed task does not identify a matched
  failure-conditioned mechanism substrate at that scale. The gate will not be
  weakened after seeing results.

## Frozen mechanism readings

1. **Selection reading.** Fit the same L2 multinomial logistic probe to the
   final answer-position residual using 5-fold GroupKFold by `semantic_id`.
   Report the out-of-sample probability assigned to the current value on
   within-stale failures, its 2,000-permutation value-label null, and the raw
   and pooled-log-length-controlled failure-minus-correct difference.
2. **Attention reading.** For every head, measure
   `stale_mass / (stale_mass + current_mass)` at the answer position.
3. **QK reading.** For every head, measure rotated
   `QK(current write) - max QK(stale writes)` at the answer position.
4. For attention and QK, regress each head's value on log prompt length once on
   the full matched pool, then compare within-stale and correct residuals.
   Report raw and controlled deltas side by side.
5. Use 2,000 within-(template, seed) label permutations. Head significance is
   family-wise controlled with the permutation maximum/minimum across every
   head in that model. Also report whether the same head passes the directional
   attention and QK thresholds; do not infer homologous head IDs across sizes.

## Decision rule

A scale supports the descriptive transfer only when all are true:

- the fixed `k=4` behavior gate passes;
- the failure-pool current-value probe score is above its shuffle-null upper
  bound;
- at least one head has a length-controlled negative QK-margin shift beyond
  the model-wide family-wise shuffle bound;
- at least one of those same heads has a length-controlled positive
  stale-attention-ratio shift beyond its model-wide family-wise shuffle bound.

Partial passes are reported component by component. A negative result is a
failure of descriptive transfer under this fixed task, not proof that no other
mechanism exists at that model scale.
