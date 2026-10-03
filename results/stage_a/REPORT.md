# Stage A Report

## Setup

- Data: `data/stage_a.jsonl`, generated with `gen_tasks.py --n-per-cell 150 --seed 1`.
- Extraction: `Qwen/Qwen2.5-7B-Instruct`, eager attention, `--dtype float32`, Slurm job `18932864`.
- Kernel incident: the earlier bf16-eager run was invalidated; fp32-eager artifacts are used here.
- Gate v2: passed by adjudication in Addendum 2; the old +/-0.04 per-cell gate is superseded.
- Stable-label sidecar: `results/stage_a/stable_labels.jsonl`, joining fp32 extraction to `results/stage_a_check_sdpa.jsonl` by `id`.

Primary analysis uses stable labels only:

- stable-correct: 531
- stable-wrong: 194
- flipped/excluded from primary: 175

## Plots

Primary stable subset:

- `results/stage_a/figs/stable/a1_p_last_heatmap.png`
- `results/stage_a/figs/stable/a1_p_last_correct_vs_wrong.png`
- `results/stage_a/figs/stable/a1_content_vs_position.png`
- `results/stage_a/figs/stable/a2_logitlens_gap_by_layer.png`

Robustness appendix:

- `results/stage_a/figs/full/a1_p_last_heatmap.png`
- `results/stage_a/figs/full/a1_p_last_correct_vs_wrong.png`
- `results/stage_a/figs/full/a1_content_vs_position.png`
- `results/stage_a/figs/full/a2_logitlens_gap_by_layer.png`

## Stable Labels And Flip Rates

| condition | n_lines | I | stable-correct | stable-wrong | flipped | flip rate |
|---|---:|---:|---:|---:|---:|---:|
| interference | 20 | 4 | 83 | 41 | 26 | 0.173 |
| interference | 40 | 2 | 107 | 12 | 31 | 0.207 |
| interference | 40 | 4 | 72 | 39 | 39 | 0.260 |
| interference | 40 | 8 | 56 | 60 | 34 | 0.227 |
| interference | 80 | 4 | 68 | 42 | 40 | 0.267 |
| simple | 40 | 0 | 145 | 0 | 5 | 0.033 |

The 17-27% flip rate on interference cells is itself informative: many
current-vs-stale decisions sit on narrow numerical margins.

## A1 Attention Selection

Primary stable subset:

| condition | n_lines | I | n | accuracy | mean p_last |
|---|---:|---:|---:|---:|---:|
| simple | 40 | 0 | 145 | 1.000 | 1.000 |
| interference | 20 | 4 | 124 | 0.669 | 0.201 |
| interference | 40 | 2 | 119 | 0.899 | 0.381 |
| interference | 40 | 4 | 111 | 0.649 | 0.226 |
| interference | 40 | 8 | 116 | 0.483 | 0.129 |
| interference | 80 | 4 | 110 | 0.618 | 0.231 |

Core wrong-stale comparison:

- `attn(answered stale write) / attn(all writes)`: 0.185
- `attn(current/last write) / attn(all writes)`: 0.188
- stale-answer wrong examples: 192

This does not support a simple H2 story where attention selection locks onto the
answered stale write. The current write receives comparable or slightly higher
write-local attention even on stable wrong cases.

Candidate selection heads by correct-example `p_last`:

1. layer 1 head 5: 0.774
2. layer 23 head 11: 0.754
3. layer 22 head 2: 0.722
4. layer 0 head 1: 0.705
5. layer 20 head 23: 0.694

Several later heads lose `p_last` on stable wrong examples, especially layer 23
head 11 and layer 20 head 23, but the A1 evidence is weaker than the A2 evidence
because answered-stale attention is not elevated over current-write attention.

Content-vs-position:

- At fixed `n_lines=40`, error increases with I: regression slope `+0.0657`.
- At fixed `n_lines=40`, mean `p_last` decreases with I: slope `-0.0396`.
- At fixed `I=4`, length/position does not explain the behavioral effect as cleanly.

This supports a content-load effect beyond a pure position-only account.

## A2 Logit Lens

Primary stable subset after first-token collision filtering:

- kept examples: 368 / 725 stable-primary rows
- stale-example first-token drop rate: 0.366

Mean gold-minus-best-stale final-layer gap:

| I | stable correct final gap | stable wrong final gap | wrong max gap | interpretation |
|---:|---:|---:|---:|---|
| 2 | +8.69 | -4.33 | +0.53 | mixed; gold briefly leads early, then loses late |
| 4 | +5.52 | -4.41 | -0.14 | gold never leads on wrong examples |
| 8 | +5.33 | -8.27 | -0.50 | gold never leads on wrong examples |

For stable correct examples, the gold token becomes strongly favored late
(`first_positive_layer=23`). For stable wrong examples at I=4 and I=8, the gold
token never leads at any layer. I=2 wrong cases show a small early/mid positive
gap, but the final gap is negative.

Full-set robustness matches the same pattern:

- I=2 wrong final gap: -3.47, max gap +0.22
- I=4 wrong final gap: -3.92, max gap -0.20
- I=8 wrong final gap: -7.56, max gap -0.48

## Verdict

Stage A supports **H1-dominant retention/retrievability failure**, with a small
low-load mixed component.

The decisive evidence is A2 on the stable subset: for stable wrong examples at
I=4 and I=8, the gold/current value never beats the best stale value at any
layer under the logit lens. This is the H1 signature: the current value is not
recoverably represented at the query position in a way the model's own final
norm and unembedding can read out. The I=2 stable-wrong subset is mixed: gold
can briefly lead early, but it is overtaken late, so low-load failures may
include a late readout/selection component.

A1 does not rescue a selection-only H2 account. Attention to the answered stale
write is not higher than attention to the current write on stable wrong cases
(0.185 vs 0.188). The implicated heads are therefore best treated as candidate
current-write retention/readout-support heads, not proven stale-selection heads.

The effect is content-driven: increasing the number of stale competitors at
fixed length increases stable error rate and reduces `p_last`, while the length
control at fixed I is weaker. The flip-rate analysis shows the behavioral margin
is narrow, but the stable-label analysis preserves the same qualitative H1
pattern.
