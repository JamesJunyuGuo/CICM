# Frozen Stage R10 paper summary

**Status:** frozen on 2026-09-11
**Verdict:** `controlled_natural_dialogue_retrieval_correction_only`

The canonical H100 run used 240 calibration and
960 held-out confirmation examples from the 1,200-row
natural CICM factorial. Retrieval accuracy increased from
37.6% to
68.9%: a raw paired gain of
31.2 percentage points, with a
semantic-ID clustered 95% bootstrap interval of
[27.9,
35.8] points. The intervention
preserved 100.0% of initially correct
retrievals and corrected 42.7% of
within-slot stale errors. All six factorial cells improved. Sixteen matched
random-position controls averaged 0.15 points,
whereas reversing the route changed accuracy by
-16.0 points.

The independent A100 run reproduced the result:
36.9% to
68.9%
(32.0 points). The derived decision task
did not improve (H100 -0.42 points;
A100 -1.04 points).


## What beta means

Let \(s_{lh}(q,j)\) denote the pre-softmax query--key attention score from the
final prompt token \(q\) to context token \(j\), in layer \(l\) and head \(h\).
For the frozen set of eight routing heads, balanced routing applies

\[
s'_{lh}(q,j)=
\begin{cases}
s_{lh}(q,j)-\beta, & j\text{ is part of a superseded same-slot value},\\
s_{lh}(q,j)+\beta, & j\text{ is part of the latest same-slot value},\\
s_{lh}(q,j), & \text{otherwise}.
\end{cases}
\]

The attention weights are then computed normally as
\(A'_{lh}(q,\cdot)=\operatorname{softmax}(s'_{lh}(q,\cdot))\). Thus
\(\beta\) is a dimensionless additive bias in attention-logit units, not a
percentage change in attention, a residual-stream steering norm, or a learned
parameter. For one routed current key and one routed stale key, the operation
adds \(2\beta\) to their log-attention odds:

\[
\log\frac{A'_{lh}(q,j_{\mathrm{current}})}
{A'_{lh}(q,j_{\mathrm{stale}})}
=
\log\frac{A_{lh}(q,j_{\mathrm{current}})}
{A_{lh}(q,j_{\mathrm{stale}})}+2\beta.
\]

This pairwise identity describes the local attention operation; it does not
imply an \(e^{2\beta}\) change in the model's output probability, because
multiple keys, heads, layers, and the rest of the network still contribute.
We selected \(\beta\) only on the 240-example calibration split from the fixed
grid \(\{0.5,1,2,4,8\}\), maximizing retrieval gain subject to preserving at
least 95% of initially correct calibration answers. The selected value was
\(\beta=8\), which was then frozen before evaluating the 960 confirmation
examples. Its magnitude should not be compared across model architectures
without recalibration because their attention-score scales can differ.

This choice is calibrated but still heuristic. On calibration, retrieval gain
rose monotonically across the tested grid (3.3, 6.2, 13.8, 20.8, and 37.1
percentage points for \(\beta=0.5,1,2,4,8\), respectively), with 100% correct
preservation at every point. Because \(\beta=8\) is also the upper grid boundary,
the experiment establishes it only as the best eligible value tested; it does
not locate a saturation point, establish an optimum, or justify transferring
the same numerical value to another model.

## How the test-time correction is applied

1. From the raw dialogue, a deterministic detector uses the declared slot
   schema to identify the queried slot, its latest update, and superseded
   same-slot mentions. Tokenizer offsets map those mentions to key positions.
   The detector does not read the gold answer, error label, saved experimental
   metadata, or the model's baseline response.
2. The model weights remain frozen. During prompt prefill, only the final
   answer-query row in the eight preselected heads is changed using the formula
   above; every other query, key, and head is untouched. The \(\beta=0\) arm is
   therefore an exact identity operation.
3. Greedy decoding then proceeds normally with the model's cache. The hook is
   inactive on later one-token decoding steps. A deployed routed answer needs
   one model decode, with no training, gradient update, second judging model, or
   post-hoc check that the baseline answer was wrong.

This is test-time correction in the precise sense that it changes an internal
attention computation for one forward pass while leaving the parameters and
prompt text unchanged. It currently requires an open-weight model with
attention-hook access and a parseable update structure; it is not directly
available through a closed text-only API.

## Paper-facing claim

A training-free attention-routing intervention improves direct current-binding
retrieval in controlled natural dialogue on Qwen2.5-7B-Instruct while
preserving initially correct retrievals. It does not improve the derived
decision task.

## Scope boundary

Do not claim unrestricted-dialogue or closed-API deployment, arbitrary
downstream decision improvement, cross-model correction, or uniqueness of the
frozen head set. H100 is the canonical numerical run; A100 is a hardware
replication.
