# Stage N-S Report: Attention-Sink Diagnostic on Qwen2.5-7B

## Pre-registered verdict

**`no_mechanism_substrate`.**

The assistant-prefill repair restored a valid one-token behavioral measurement,
but no overwrite count passed the frozen exact-pair gate. The attention-sink
mechanism analysis therefore stopped before all-head harvesting, causal ranking,
or gate-versus-routing intervention.

## Boundary validation

The original chat rendering closed the completion task inside the user turn, so
Qwen began its answer with explanatory tokens such as `Based`, `Latest`, and
`Assign`. The repaired rendering places the final query in an assistant
prefill and scores the actual leading-space continuation token.

Both 216-row boundary smokes passed:

| Portal | Job | Elapsed | k=0 accuracy | Other rate |
|---|---:|---:|---:|---:|
| H100 | 19775124 | 1:44 | 1.000 | 0.000 |
| A100 | 19775128 | 3:01 | 1.000 | 0.000 |

The 3,888-row full runs also passed the boundary gate. H100 k=0 accuracy was
0.9892 with an overall other rate of 0.0036; A100 k=0 accuracy was 0.9892 with
an overall other rate of 0.0041.

## Behavior

Each overwrite count contains 648 examples. Rates below are percentages of all
examples in the corresponding k cell.

| k | H100 accuracy | H100 within-stale | A100 accuracy | A100 within-stale |
|---:|---:|---:|---:|---:|
| 0 | 98.92 | 0.00 | 98.92 | 0.00 |
| 1 | 99.54 | 0.31 | 99.54 | 0.31 |
| 2 | 98.77 | 0.93 | 98.77 | 0.77 |
| 3 | 98.15 | 1.39 | 98.15 | 1.23 |
| 4 | 95.06 | 4.78 | 95.37 | 4.48 |
| 5 | 94.29 | 5.71 | 94.44 | 5.56 |

The repaired task shows a bounded stale-binding dose response at larger
overwrite counts. The near-identical portal results differ by at most two
classified rows in a k cell and lead to the same gate decision.

## Exact causal-pair gate

A k value must provide at least 90 exact stale-corrupt/current-clean pairs, at
least 20 pairs from every seed, and at least 20 pairs from every template.

| k | H100 exact pairs | A100 exact pairs | Gate |
|---:|---:|---:|:---|
| 2 | 1 | 1 | fail |
| 3 | 3 | 4 | fail |
| 4 | 28 | 26 | fail |
| 5 | 37 | 36 | fail |

At k=5, the H100 pool had 14/12/11 pairs by seed and 21/14/2 by
arrow/current/latest template. The A100 pool had 16/9/11 by seed and 20/13/3 by
template. Both fail the total-count and stratification requirements. Across all
k values, 69 H100 and 67 A100 clean-responsive aligned pairs were found, but the
frozen rule forbids pooling k values after observing outcomes.

## Scope

This is a valid behavioral result and a valid negative substrate gate. It is not
evidence for or against an attention-sink explanation because the preregistered
mechanism analysis was not identifiable at the required power. No mechanism
full run was submitted, and no thresholds or task difficulty were retuned.
