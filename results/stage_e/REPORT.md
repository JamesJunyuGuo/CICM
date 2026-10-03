# Stage E Report

## Current Status

The OpenRouter behavioral sweep is complete: `results/stage_e/openrouter/sweep_final.jsonl`
contains 4000 rows across 5 models, summarized in
`results/stage_e/openrouter/sweep_final.summary.json`. F8 has been regenerated
from that final sweep.

The Qwen2.5-7B retrieval-head overlap check is also complete:
`results/stage_e/retrieval_heads/full2_ai/summary.json` covers 200 fresh simple
examples from seed 30.

## Retrieval-Head Overlap

The simplified Wu-style retrieval detector finds only partial overlap with the
Stage-A stale-binding heads. The top-16 retrieval heads overlap the Stage-A
top-8 `p_last` heads at two heads: `(22, 1)` and `(23, 11)`. Across all heads,
the Spearman correlation between retrieval score and Stage-A `p_last` score is
-0.021.

This is useful evidence for the paper framing: the stale-binding heads are not
just generic retrieval/copy heads. There is some shared machinery, especially in
late layer 22/23 heads, but the all-head ranking is essentially uncorrelated.

| model | examples | accuracy | copied answer tokens | overlap with top-8 p_last | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | 200 | 0.975 | 771 | 2 | -0.021 |

## Scale Sweep

F8 is `results/figures/F8_openrouter_scale_sweep.{png,pdf}` with sidecar
`results/figures/F8_openrouter_scale_sweep.data.json`. It has two panels:
accuracy vs interference load I for each model, and within-stale share of
errors per model.

The main scale result is that larger models shift the failure threshold upward
but do not remove the stale-binding signature. Qwen-7B degrades strongly by
I=8/12/16, while Qwen-72B and GPT-4o are near ceiling on this sweep. Llama-70B
is the most important persistence result: it is high at ordinary P0 loads but
still degrades at harder loads, with accuracy 0.97 -> 0.86 -> 0.77 at
I=8/12/16.

When Llama-70B fails, it fails in the same way: across all interference cells it
has 43 errors and all 43 are within-stale errors. This gives a clean
cross-scale observation: scale moves the onset of failure, but the errors that
remain are still stale-binding errors rather than random hallucinations.

| model | I=8 acc | I=12 acc | I=16 acc | wrong errors | within-stale share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen-7B | 0.567 | 0.400 | 0.360 | 240 | 0.996 |
| Llama-8B | 0.167 | 0.150 | 0.100 | 510 | 0.959 |
| Llama-70B | 0.967 | 0.860 | 0.770 | 43 | 1.000 |
| Qwen-72B | 1.000 | 0.990 | 0.990 | 3 | 0.667 |
| GPT-4o | 0.993 | 1.000 | 1.000 | 1 | 1.000 |

## Llama A2 Logit Lens

Llama-3.1-8B shows a late-layer resolution divergence on the stable within-stale
subset. Correct examples move toward the current value late in the network,
while wrong examples remain stale-biased.

| interference I | correct mean final | correct median final | wrong mean final | wrong median final |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2.464 | 2.000 | -3.280 | -2.625 |
| 4 | 0.820 | 0.500 | -4.982 | -4.625 |
| 8 | -1.949 | -2.250 | -5.256 | -4.875 |

The I=8 anomaly is real under this probe: the correct-group mean and median are
both negative. The late correct-vs-wrong separation remains, but some Llama
correct high-I answers appear to use a pathway not visible to this first-token
final-position logit lens.

The full A2 deliverable is
`results/stage_e/llama/A2_LOGITLENS_REPORT.md`; the plotted curve is
`results/stage_e/llama/figs/a2_stable_gpu_local/a2_logitlens_gap_by_layer.png`.

## Link To Stage-B-Lite

The A2 result is consistent with the causal Stage-B-lite picture. Llama's
late-scope boost repairs many wrong examples, but the top-8-head boost remains
weak:

| intervention | alpha | W flip-to-correct | C harm |
| --- | ---: | ---: | ---: |
| late scope | 4 | 0.465 | 0.005 |
| top-8 heads | 4 | 0.035 | 0.035 |
| late scope | 8 | 0.608 | 0.025 |
| top-8 heads | 8 | 0.068 | 0.060 |

This supports a layer-diffuse selection bottleneck for Llama: answer resolution
separates late, but it is not concentrated enough in the top-8 A1 heads for a
small-head intervention to be sufficient.
