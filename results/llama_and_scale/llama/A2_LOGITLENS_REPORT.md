# Llama A2 Logit-Lens Report

## Verdict

Llama-3.1-8B reproduces the late-layer resolution divergence seen in the Qwen
mechanistic story. On stable within-stale examples, correct cases move toward
the current value in late layers, while wrong cases remain strongly stale-biased.
The divergence is clearest at I=2 and I=4, and still present at I=8 even though
both groups are harder and the correct group does not cross positive on average.

This is consistent with the Stage-B-lite causal result: Llama is not controlled
by a small top-8 head set alone. The current-value decision can be moved by
late/all-scope attention-logit boosts, but the selected top heads are too weak
as an isolated intervention. Mechanistically, the decision signal looks
late-layer and distributed rather than concentrated in a small head subset.

## Inputs

- Extraction: `results/llama_and_scale/llama/extract_stable_ai_local.npz`
- Stable index: `results/llama_and_scale/llama/extract_stable_index_ai_local.jsonl`
- Summary: `results/llama_and_scale/llama/a2_stable_median_ai_summary.json`
- Figure: `results/llama_and_scale/llama/figs/a2_stable_median_ai/a2_logitlens_gap_by_layer.png`
- Wrong filter: `within_stale`

## Acceptance Checks

- Stable examples analyzed: 900 total stable labels.
- Stable wrong examples retained for A2: 482 within-stale wrong examples.
- A2 wrong examples used after token-overlap filtering: 480.
- Excluded wrong examples: 2.
- No GPU rerun was needed for this report; this is analysis-only on existing
  Llama extraction outputs.

## Final-Layer Current-vs-Stale Gap

The A2 gap is defined as logit-lens support for the current value minus support
for the stale value. Positive means the layer favors the current value; negative
means it favors a stale value.

| interference I | correct n | correct mean final | correct median final | wrong n | wrong mean final | wrong median final | mean diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 72 | 2.464 | 2.000 | 75 | -3.280 | -2.625 | 5.744 |
| 4 | 41 | 0.820 | 0.500 | 104 | -4.982 | -4.625 | 5.802 |
| 8 | 55 | -1.949 | -2.250 | 89 | -5.256 | -4.875 | 3.307 |

## Late-Layer Shape

For I=2, correct cases become positive in late layers and peak at layer 28
with a gap of 3.072; the final-layer gap remains positive at 2.448. Wrong cases
never cross positive and become increasingly stale-biased in late layers,
ending at -3.275.

For I=4, correct cases only cross positive very late, around layer 28, and end
at 0.802. Wrong cases again never cross positive and end much lower at -4.981.

For I=8, the task is harder: correct cases remain negative on average and the
median is also negative (-2.250). This is not a mean-skew/outlier artifact.
Correct examples are still substantially less stale-biased than wrong examples
(-1.949 vs -5.256 by the mean final gap), so the divergence persists, but the
absolute sign is anomalous. The most conservative reading is that some Llama
correct answers at high interference are produced through a pathway this
first-token final-position logit lens does not see.

## I=8 Anomaly Adjudication

The I=8 correct-gap anomaly is real under this probe: both the mean and median
final-layer gaps are negative. We therefore should not claim that Llama's
correct high-interference answers always have gold-current dominance in the
final-position logit lens. The cross-model claim should be narrower and more
accurate: correct and wrong trajectories separate late, but Llama can sometimes
answer correctly even when this specific logit-lens readout remains
stale-favored.

## Connection To Stage-B-Lite

Stage-B-lite shows the same localization picture from a causal angle:

| intervention | alpha | W flip-to-correct | C harm |
| --- | ---: | ---: | ---: |
| late scope | 4 | 0.465 | 0.005 |
| top-8 heads | 4 | 0.035 | 0.035 |
| late scope | 8 | 0.608 | 0.025 |
| top-8 heads | 8 | 0.068 | 0.060 |

The logit-lens result says Llama's answer resolution separates in late layers.
The causal result says boosting only Llama's top-8 A1 heads barely repairs wrong
examples, while broader late-scope boosting repairs many more. Together, these
support a layer-diffuse selection bottleneck: the current value is not simply
recoverable by pushing a small set of write-tracking heads, but late computation
can still be shifted toward the current binding.
