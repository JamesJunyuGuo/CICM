# Cross-model test-time repair: complete paper-writing reference

Snapshot: 2026-09-18. This is an internal writing reference, not manuscript text.
All tables are extracted from saved canonical summaries and checked against raw
response arms. No model was rerun, no new configuration was selected, and no
new GPU job was submitted. Local paths are for author audit only; omit them
from the paper. Original numerical artifacts remain unchanged. A separate
Mistral record's fixed-margin wording was corrected to match the adaptive code.

Reading guide: Sections 1--2 give the claim and headline numbers; 3--4 explain
the method and evaluation; 5--6 give controls, transfer, and per-cell effects;
7 contains reproducible model configurations; 8--10 cover limitations and
paper wording; 11--13 preserve real cases, mirrors, and the audit trail.

## 1. Main result and how to present it

**A model-calibrated, training-free attention-routing intervention improves
direct current-binding retrieval across Qwen, Llama, Mistral, and Gemma on
the controlled natural-dialogue factorial.** It shifts attention away from
superseded values and toward the current write, without replacing the answer
or changing model weights. Parameters and head identities are model-specific.

The strongest controlled replication is the adaptive protocol on Qwen-3B,
Llama-3B/8B, and Mistral-7B. Gemma-9B adds a large positive replication with
an explicitly qualified identity diagnostic. Qwen-7B is the earlier fixed-bias
demonstration; Qwen-14B provides a small positive, not a large-scale success of
the later adaptive protocol. Do not hide the small 14B result.

Recommended story: diagnosis of incorrect selection -> automatic detection of
current/stale spans -> model-specific selection of influential attention heads
-> intervention during inference -> held-out corrections with high observed
correct-answer preservation -> limits of downstream transfer.

Do not claim that all contextual forgetting is solved, that all models share
the same heads, or that repair improves monotonically with parameter count.
These are instruction-tuned checkpoints and one shared controlled dataset,
not seven independent datasets or a production deployment study.

## 2. Primary retrieval results

Accuracy, preservation, and correction are percentages. Gain is percentage
points (pp). Raw gain weights rows equally; the separate clustered estimate
and its 95% CI weight semantic clusters equally. Never label the latter as a
row-weighted confidence interval. Llama-8B uses 959 primary rows, not 960.

| Model | Policy | N | Before % | After % | Raw gain pp | Cluster gain pp [95% CI] | Stale corrected | Correct preserved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B | adaptive | 960 | 16.46 | 95.21 | +78.75 | +78.37 [+74.51, +82.21] | 678/711 (95.36%) | 156/158 (98.73%) |
| Qwen2.5-7B | fixed | 960 | 37.60 | 68.85 | +31.25 | +31.80 [+27.88, +35.76] | 208/487 (42.71%) | 361/361 (100.00%) |
| Qwen2.5-14B | fixed | 960 | 40.52 | 41.67 | +1.15 | +1.17 [+0.14, +2.24] | 13/491 (2.65%) | 383/389 (98.46%) |
| Llama-3.2-3B | adaptive | 960 | 27.71 | 93.12 | +65.42 | +64.60 [+59.40, +69.63] | 514/541 (95.01%) | 266/266 (100.00%) |
| Llama-3.1-8B | adaptive | 959 | 41.71 | 88.63 | +46.92 | +46.90 [+41.99, +51.93] | 395/467 (84.58%) | 400/400 (100.00%) |
| Mistral-7B-v0.3 | adaptive | 960 | 5.42 | 81.35 | +75.94 | +75.01 [+70.33, +79.12] | 564/674 (83.68%) | 51/52 (98.08%) |
| Gemma-2-9B | adaptive | 960 | 47.08 | 91.88 | +44.79 | +45.06 [+39.63, +50.99] | 407/455 (89.45%) | 452/452 (100.00%) |

Gemma is identity-qualified; Llama-8B excludes one exposed item. All rates describe this factorial, not natural prevalence.


Numerical highlights for selection: the five adaptive configurations yield raw gains of +44.79 to +78.75 pp; stale-correction rates 83.68%--95.36%; and observed correct preservation 98.08%--100%. This range includes the qualified Gemma run and is not an average or a scaling law.


## 3. What the intervention actually does

### 3.1 Inputs and automatic routing

At inference, the router receives the raw dialogue, a declared slot/value
schema, tokenizer offsets, and the model's frozen calibrated configuration.
It identifies the target slot from the anchored final question, scans prior
user turns for known values of that slot, and treats the latest update-intent
mention as current. Superseded values define stale key spans. The controlled
reminder phrase containing both `still thinking about` and `compare options`
is recognized as a reminder, not as a new update. The current span is the last
binding-update mention of the current value; stale spans include mentions
whose value differs from it. Assistant text does not determine update intent.

This is a schema- and grammar-aware parser, **not an unrestricted-language
intent detector**. It knows value strings from the schema and derives the
current value from text. It does not consume the confirmation gold field,
saved gold spans, failure labels, or the baseline model response. Experimental
metadata is used separately for route auditing and scoring. Exact route-audit
agreement is an in-distribution parser check, not evidence of general parsing.

Important deployment distinction: because the parser identifies the current
value, it could itself answer this direct retrieval task or provide a recap.
Thus routing is an automatic model-internal correction, but these experiments
do not establish that it is practically preferable to external state tracking.
The recap comparison below is necessary, not optional.

### 3.2 Fixed-bias variant: Qwen-7B and Qwen-14B

For the final prompt query at one selected head, let s_j be its scaled,
masked pre-softmax score for key j. Let C be current-write value-token
positions and S be stale-value positions. Only that query row is changed:

\[
s'_j = s_j + \beta\,\mathbf{1}[j\in C]
                  - \beta\,\mathbf{1}[j\in S].
\]

Every other key score is unchanged. Softmax and the value/output computation
then run normally. Beta is in attention-logit units, not a percentage of
attention, a residual-vector norm, or a trained weight. For any current/stale
key pair, the local log-attention odds increase by 2 beta. This identity does
not imply the same change in the model's answer odds.

Qwen-7B uses eight heads frozen from Stage B. Qwen-14B discovers eight heads
on its own calibration rows by the failure-minus-correct contrast in
log(stale attention/current attention). Both choose beta on calibration from
{0.5, 1, 2, 4, 8}, maximizing gain with at least 95% correct preservation.
These earlier runs do not have a separate discovery/validation subdivision.

### 3.3 Adaptive variant: model-specific causal-gradient head selection

The later protocol calibrates a desired attention-mass log-odds margin m,
rather than reusing a raw beta across models. At each selected head and item:

\[
\delta = \log\sum_{j\in C}e^{s_j}-\log\sum_{j\in S}e^{s_j},\qquad
\beta = \min\left(8,\max\left(0,\frac{m-\delta}{2}\right)\right).
\]

Apply the same balanced score update above. For finite accessible scores,
delta is exactly log(A(C)/A(S)): the common softmax denominator cancels.
If delta already exceeds m, nothing is changed. If the cap is inactive and
delta < m, the new local log-odds equal m (up to numerical precision).
At the cap, delta rises by at most 16. This accounts for the existing local
attention competition; it is **not** residual-norm normalization, a guaranteed
output-logit margin, or a scale-invariant universal optimum. m and the head
set still need per-model calibration. The margin does not specify total
attention paid to C and S relative to all other context.

The adaptive identity arm disables the route with gate_scale=0. Setting m=0
alone is not generally an identity: a negative delta can still induce a shift.

No K/V vectors or unembedding logits are directly overwritten. GQA models are
intervened on per **query head**, after shared keys are expanded to query
heads. Gemma uses its native score scaling and softcap, then mask and routing,
then float32 softmax. The bias is applied after softcapping, not before it.
Only the final prompt token's row is routed during prefill. Subsequent cached
single-token decoding steps are not hooked at that query index. Indirect
downstream effects of the changed prompt state can persist.

### 3.4 How heads and the operating point are selected

On discovery rows, freeze model weights and introduce a differentiable gate
for each head's routing operation. At gate=0, compute the derivative of the
current-minus-strongest-stale first-token answer-logit gap. Candidate scores
aggregate tokenization variants with logsumexp. Average gradients over
baseline within-stale rows and rank heads by this value. Correct-row
gradients and absolute target attention are recorded, but **the actual ranking
does not subtract correct gradients or enforce an attention-mass threshold**.
This is a local intervention-sensitivity proxy, not exhaustive finite-effect
patching or proof that every selected head individually repairs answers.

Discovery uses m=1 and a beta cap of 8. Current/stale first-token collisions
are excluded only from the discovery objective; full-response evaluation and
confirmation retain these rows. On independent validation rows, test nested
positive-gradient head sets from {1,2,4,8,16,32} and m in {0.5,1,2,4}.
Choose positive retrieval gain with preservation >=95%; maximize gain, breaking
ties toward fewer heads and lower m. Candidate availability can truncate the
head-size grid; the complete evaluated curves are retained below. No model
weight is trained; calibration nevertheless uses labeled data and gradients.

Freeze the chosen set and m before confirmation. At application time there
is no gradient computation, judge, gold answer lookup, or baseline-error test:
the same rule is applied to every item, including originally correct items.
One routed decode suffices. "One decode" does not establish negligible
latency: this implementation uses eager attention and has no controlled
production latency/memory benchmark against optimized serving kernels.

Llama-8B is an exception to the full 120/120 calibration protocol: its frozen
configuration came from the 12-discovery/12-validation smoke. One smoke item
overlapped confirmation; the primary analysis excludes that item. Do not
describe this run as having a fresh 120/120 calibration or 960 unseen rows.

## 4. Dataset, scoring, statistics, and comparability

The same 1,200 controlled natural dialogues have six equal cells:
same-slot stale distance {near,far} x other-slot distance {far,mid,near2}.
The current target write stays in the far regime. Each cell has 200 rows;
hashing row IDs with `R9C-SPLIT` allocates 40 to calibration and 160 to
confirmation. The adaptive full calibration divides its 240 rows into
120 discovery and 120 validation by a second within-cell ID hash `R12-DV`.
This is an item split, **not** a held-out vocabulary, template, or slot split.
Semantic clusters can span calibration and confirmation.

There are three slots (diet, learning_style, music_genre), seven natural values
per slot, and 400 rows per slot. Overwrite counts are {1,2,3,4,6}, 240 rows each.
Those counts describe this dataset, not the earlier k=1..6 dose experiment.
All questions explicitly name the target slot. Old-value reminders and
other-slot competitors remain present. See the full per-cell results before
interpreting any aggregate: the equal-cell aggregate is not real-world error
prevalence or an average improvement under a natural user distribution.

Retrieval labels use the saved deterministic value-matching classifier, not
an LLM judge. Its priority is current hit > stale hit > unused same-slot value
> cross-slot value > other. No unused-same-slot category occurs in these saved
primary retrieval arms. **Any current-value hit wins even when another controlled
value is also present.** This means programmatic matching of generated text,
not whole-string equality or a guarantee of an unambiguous answer. Mixed-value
counts and a conservative sensitivity are reported below. The derived-decision
task replaces the final query
with a value-to-action-code table and asks for the code associated with the
current value. It uses retrieval-calibrated heads/settings without task-specific
retuning. It is a transfer test, not an agent workflow or broad reasoning test.

Define c_i and r_i as baseline and routed correctness indicators. Report:

\[
\Delta_{\rm row}=\frac{1}{N}\sum_i(r_i-c_i),\quad
\mathrm{preservation}=\frac{\sum_i c_i r_i}{\sum_i c_i},\quad
\mathrm{stale\ correction}=\frac{\sum_{i:\,\mathrm{baseline\ stale}}r_i}
{\#\{i:\mathrm{baseline\ stale}\}}.
\]

The net gain includes fixes of cross-slot/other errors and losses of previously
correct answers; it is not identical to the stale-correction rate. A semantic
cluster is (slot, current value, k, same-slot-distance bin, other-slot-distance
bin). The saved CI averages gains within each cluster, samples clusters with
replacement 2,000 times, and averages sampled cluster means. Thus it estimates
an **equal-cluster**, not a row-weighted, effect. The current files contain 210
confirmation clusters. No new bootstrap or p-value is introduced in this report.
The CIs do not include uncertainty from redoing calibration/head selection.

Control intervals are empirical 2.5th--97.5th percentiles across random sets,
not confidence intervals for targeted-minus-random paired differences. Layer-
matched random heads use the same number per selected layer, excluding selected
heads, with the same adaptive policy. Random positions match positive/negative
token counts, are drawn before the query, and exclude all controlled value
tokens. **These are not matched-residual-norm controls**; adaptive beta can
differ across heads/positions. Opposite routing recomputes the margin in the
reverse direction; it is not guaranteed to have the same realized beta.

Greedy decoding, bf16 model weights, maximum 8 new tokens; primary confirmations
use 16 random-position sets and adaptive runs additionally 64 random-head sets.
R10 Qwen-7B's default analysis seed is 20260910; R11 Qwen-14B and adaptive
runs use default 20260911
(runbooks do not override these). Bootstrap/control streams use task/arm offsets.
Hardware and inference batch sizes are listed with each run. No repeat-seed
calibration variance or latency improvement is established.


## 5. Control results and downstream transfer

### 5.1 Retrieval controls

All gains below are row-weighted pp. Random bands are empirical ranges across control sets, not paired-effect CIs.

| Model | Targeted gain | Random positions mean [range] | Random heads mean [range] | Opposite gain | Recap accuracy % | Targeted - recap pp |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B | +78.75 | +0.08 [-0.27, +0.38]; n=16 | -9.98 [-14.42, -1.28]; n=64 | -16.35 | 99.27 | -4.06 |
| Qwen2.5-7B | +31.25 | +0.15 [-0.42, +0.59]; n=16 | not run in this protocol | -16.04 | 88.02 | -19.17 |
| Qwen2.5-14B | +1.15 | +0.18 [+0.00, +0.38]; n=16 | not run in this protocol | -1.77 | 98.54 | -56.88 |
| Llama-3.2-3B | +65.42 | -0.17 [-0.48, +0.07]; n=16 | -5.05 [-12.19, +0.88]; n=64 | -27.08 | 68.85 | +24.27 |
| Llama-3.1-8B | +46.92 | -0.36 [-0.69, +0.00]; n=16 | -2.52 [-10.90, +3.05]; n=64 | -28.15 | 65.38 | +23.25 |
| Mistral-7B-v0.3 | +75.94 | +0.03 [-0.10, +0.21]; n=16 | -0.49 [-1.50, +0.45]; n=64 | -3.65 | 98.44 | -17.08 |
| Gemma-2-9B | +44.79 | +0.19 [-0.07, +0.38]; n=16 | -3.04 [-7.07, +1.32]; n=64 | -27.71 | 82.81 | +9.06 |


Recap comes from the same automatic parser, not an external judge. It explicitly inserts the inferred current value in an assistant message before the query. Targeted-minus-recap above is descriptive; no paired significance test against recap was added.


### 5.2 Derived decision (no retuning)

| Model | N | Before % | After % | Raw gain pp | Cluster gain pp [95% CI] | Correct preserved | Recap accuracy % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B | 960 | 26.56 | 28.54 | +1.98 | +2.09 [+1.03, +3.15] | 253/255 (99.22%) | 77.29 |
| Qwen2.5-7B | 960 | 31.04 | 30.63 | -0.42 | -0.71 [-1.79, +0.24] | 288/298 (96.64%) | 86.04 |
| Qwen2.5-14B | 960 | 28.02 | 27.92 | -0.10 | -0.08 [-0.24, +0.00] | 268/269 (99.63%) | 88.85 |
| Llama-3.2-3B | 960 | 23.12 | 23.44 | +0.31 | +0.17 [-1.25, +1.56] | 203/222 (91.44%) | 88.23 |
| Llama-3.1-8B | 959 | 18.87 | 18.98 | +0.10 | -0.04 [-2.31, +2.24] | 145/181 (80.11%) | 83.52 |
| Mistral-7B-v0.3 | 960 | 7.19 | 7.40 | +0.21 | +0.22 [-0.22, +0.71] | 67/69 (97.10%) | 89.90 |
| Gemma-2-9B | incomplete | not assessed | not assessed | not assessed | not assessed | not assessed | not assessed |


Qwen-3B provides a small positive transfer result; the other completed arms do not support a consistent downstream benefit. A CI including zero is insufficient evidence of benefit, not proof of exactly zero effect. Gemma is missing, not null.


### 5.3 Derived-decision controls and stale corrections

These use the same frozen retrieval configuration. Random intervals are empirical ranges in pp.

| Model | Stale-code -> correct | Random positions mean [range] | Random heads mean [range] | Opposite gain pp |
| --- | --- | --- | --- | --- |
| Qwen2.5-3B | 21/702 (2.99%) | +0.16 [-0.17, +0.48] | -0.32 [-1.19, +0.42] | +0.00 |
| Qwen2.5-7B | 6/656 (0.91%) | -0.25 [-0.59, +0.20] | not run | -0.83 |
| Qwen2.5-14B | 0/678 (0.00%) | +0.01 [-0.17, +0.23] | not run | +0.10 |
| Llama-3.2-3B | 16/598 (2.68%) | +0.38 [+0.01, +0.62] | +0.33 [-0.10, +0.77] | -0.42 |
| Llama-3.1-8B | 28/612 (4.58%) | +0.17 [-0.17, +0.38] | +0.18 [-0.42, +0.77] | -1.88 |
| Mistral-7B-v0.3 | 1/552 (0.18%) | +0.20 [+0.04, +0.42] | +0.20 [-0.10, +0.42] | +0.31 |


## 6. Per-cell retrieval effects and error modes

### 6.1 All six cells

Cells name nearest same-slot stale and other-slot distance. Entries are before -> after accuracy (raw gain pp). N=160 per cell except Llama-8B near/mid N=159.

| Cell | Qwen2.5-3B | Qwen2.5-7B | Qwen2.5-14B | Llama-3.2-3B | Llama-3.1-8B | Mistral-7B-v0.3 | Gemma-2-9B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| same_far__other_far | 33.12 -> 85.00 (+51.88) | 72.50 -> 95.00 (+22.50) | 67.50 -> 66.88 (-0.62) | 50.62 -> 84.38 (+33.75) | 75.00 -> 94.38 (+19.38) | 15.00 -> 71.25 (+56.25) | 73.12 -> 87.50 (+14.37) |
| same_far__other_mid | 31.87 -> 98.12 (+66.25) | 55.62 -> 93.12 (+37.50) | 61.25 -> 65.00 (+3.75) | 29.38 -> 96.88 (+67.50) | 50.62 -> 78.75 (+28.12) | 6.88 -> 80.00 (+73.12) | 73.12 -> 96.88 (+23.75) |
| same_far__other_near2 | 28.75 -> 98.75 (+70.00) | 61.88 -> 93.12 (+31.25) | 82.50 -> 83.12 (+0.62) | 41.88 -> 96.88 (+55.00) | 62.50 -> 91.88 (+29.38) | 6.88 -> 94.38 (+87.50) | 85.00 -> 96.25 (+11.25) |
| same_near__other_far | 0.00 -> 90.62 (+90.62) | 1.25 -> 22.50 (+21.25) | 5.00 -> 6.25 (+1.25) | 20.00 -> 85.62 (+65.62) | 13.75 -> 95.62 (+81.88) | 0.00 -> 81.88 (+81.88) | 7.50 -> 99.38 (+91.88) |
| same_near__other_mid | 0.00 -> 98.75 (+98.75) | 16.25 -> 55.00 (+38.75) | 18.75 -> 19.38 (+0.62) | 15.00 -> 96.25 (+81.25) | 37.74 -> 86.16 (+48.43) | 0.62 -> 81.25 (+80.62) | 38.12 -> 89.38 (+51.25) |
| same_near__other_near2 | 5.00 -> 100.00 (+95.00) | 18.12 -> 54.37 (+36.25) | 8.12 -> 9.38 (+1.25) | 9.38 -> 98.75 (+89.38) | 10.62 -> 85.00 (+74.38) | 3.12 -> 79.38 (+76.25) | 5.62 -> 81.88 (+76.25) |


### 6.2 Repairs separated by baseline failure mode

| Model | Stale -> correct | Cross-slot -> correct | Other -> correct | Initially correct lost | Positive cells |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B | 678/711 (95.36%) | 59/63 (93.65%) | 21/28 (75.00%) | 2 | 6/6 |
| Qwen2.5-7B | 208/487 (42.71%) | 61/72 (84.72%) | 31/40 (77.50%) | 0 | 6/6 |
| Qwen2.5-14B | 13/491 (2.65%) | 3/44 (6.82%) | 1/36 (2.78%) | 6 | 5/6 |
| Llama-3.2-3B | 514/541 (95.01%) | 79/92 (85.87%) | 35/61 (57.38%) | 0 | 6/6 |
| Llama-3.1-8B | 395/467 (84.58%) | 51/88 (57.95%) | 4/4 (100.00%) | 0 | 6/6 |
| Mistral-7B-v0.3 | 564/674 (83.68%) | 4/9 (44.44%) | 162/225 (72.00%) | 1 | 6/6 |
| Gemma-2-9B | 407/455 (89.45%) | 17/42 (40.48%) | 6/11 (54.55%) | 0 | 6/6 |


### 6.3 Full retrieval response composition

Counts are correct / within-stale / cross-slot / other, using the primary denominator.

| Model | Baseline | Targeted | Opposite | Recap |
| --- | --- | --- | --- | --- |
| Qwen2.5-3B | 158 / 711 / 63 / 28 | 914 / 26 / 2 / 18 | 1 / 932 / 6 / 21 | 953 / 6 / 0 / 1 |
| Qwen2.5-7B | 361 / 487 / 72 / 40 | 661 / 281 / 10 / 8 | 207 / 712 / 28 / 13 | 845 / 110 / 2 / 3 |
| Qwen2.5-14B | 389 / 491 / 44 / 36 | 400 / 476 / 43 / 41 | 372 / 506 / 43 / 39 | 946 / 14 / 0 / 0 |
| Llama-3.2-3B | 266 / 541 / 92 / 61 | 894 / 6 / 16 / 44 | 6 / 865 / 40 / 49 | 661 / 246 / 11 / 42 |
| Llama-3.1-8B | 400 / 467 / 88 / 4 | 850 / 67 / 38 / 4 | 130 / 762 / 65 / 2 | 627 / 303 / 29 / 0 |
| Mistral-7B-v0.3 | 52 / 674 / 9 / 225 | 781 / 73 / 10 / 96 | 17 / 737 / 6 / 200 | 945 / 4 / 0 / 11 |
| Gemma-2-9B | 452 / 455 / 42 / 11 | 882 / 48 / 25 / 5 | 186 / 738 / 30 / 6 | 795 / 164 / 1 / 0 |


### 6.4 Scoring sensitivity: mixed-value responses

Post-hoc CPU check, not a replacement for the frozen classifier. 'Unambiguous-current' here means a current hit with no other controlled-value hit; it is still not whole-string exact match. The sensitivity marks mixed-current responses incorrect in both arms. No new CI was computed.

| Model | Mixed-current baseline | Mixed-current routed | Unambiguous-current raw gain pp | Unambiguous correct preserved |
| --- | --- | --- | --- | --- |
| Qwen2.5-3B | 0 | 0 | +78.75 | 156/158 (98.73%) |
| Qwen2.5-7B | 1 | 1 | +31.25 | 360/360 (100.00%) |
| Qwen2.5-14B | 4 | 4 | +1.15 | 379/385 (98.44%) |
| Llama-3.2-3B | 0 | 0 | +65.42 | 266/266 (100.00%) |
| Llama-3.1-8B | 0 | 1 | +46.82 | 400/400 (100.00%) |
| Mistral-7B-v0.3 | 9 | 0 | +76.88 | 43/43 (100.00%) |
| Gemma-2-9B | 0 | 0 | +44.79 | 452/452 (100.00%) |


## 7. Frozen model configurations and calibration

All layer/head indices below are zero-based query-head indices. Lists are in frozen ranking order, not a cross-model alignment.


### Qwen2.5-3B

- Checkpoint: `Qwen/Qwen2.5-3B-Instruct`.
- Canonical source: `results/stage_r/crossmodel/qwen25_3b/r14_causal_full_h100/summary.json`.
- Hardware: H100; inference batch size 12.

- Frozen calibration: `results/stage_r/crossmodel/qwen25_3b/r14_causal_calibration_h100/summary.json`.
- Discovery/validation rows: 120/120; discovery correct/stale pools: 20/83.
- First-token discovery exclusions: 6; candidate audit: `{"accuracy": 0.9902912621359223, "matches": 102, "n": 103}`.
- Selected top-32, m=4.0; beta cap=8.


Validation grid: net retrieval gain pp (a trailing ! means preservation <95%). Rows are head counts; columns are target margins. These are selection data, not held-out estimates.


| Heads | m=0.5 | m=1 | m=2 | m=4 |
| --- | --- | --- | --- | --- |
| 1 | +5.00 | +8.33 | +15.83 | +27.50 |
| 2 | +15.00 | +18.33 | +30.83 | +45.83 |
| 4 | +30.83 | +40.00 | +53.33 | +61.67 |
| 8 | +45.83 | +55.00 | +61.67 | +70.83 |
| 16 | +53.33 | +60.83 | +67.50 | +76.67 |
| 32 | +62.50 | +65.00 | +69.17 | +79.17 |


Selected validation point: 17.50% -> 96.67%; preservation 100.00%; stale correction 97.59%.


Frozen heads: L26H5, L33H9, L32H7, L31H7, L27H1, L33H15, L29H2, L33H0, L31H5, L33H14, L26H10, L30H11, L29H1, L31H3, L29H3, L32H1, L31H0, L33H3, L26H12, L27H4, L31H12, L26H0, L32H3, L31H15, L24H5, L33H10, L32H0, L33H12, L32H6, L32H13, L27H5, L28H10.


Identity diagnostic: response changes=0; max first-token score difference=0.0.


### Qwen2.5-7B

- Checkpoint: `Qwen/Qwen2.5-7B-Instruct`.
- Canonical source: `results/stage_r/routing/balanced_full_h100/summary.json`.
- Hardware: H100; inference batch size 12.

- Selected fixed beta=8.0; eight heads.

| Calibration beta | Before % | After % | Gain pp | Preservation % | Stale corrected % |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 37.08 | 40.42 | +3.33 | 100.00 | 4.24 |
| 1.0 | 37.08 | 43.33 | +6.25 | 100.00 | 10.17 |
| 2.0 | 37.08 | 50.83 | +13.75 | 100.00 | 22.03 |
| 4.0 | 37.08 | 57.92 | +20.83 | 100.00 | 30.51 |
| 8.0 | 37.08 | 74.17 | +37.08 | 100.00 | 54.24 |


Frozen heads: L1H5, L23H11, L22H2, L22H1, L0H1, L20H23, L19H19, L18H8.


Identity diagnostic: response changes=0; max first-token score difference=0.0.


### Qwen2.5-14B

- Checkpoint: `Qwen/Qwen2.5-14B-Instruct`.
- Canonical source: `results/stage_r/crossmodel/qwen25_14b/full_h100/summary.json`.
- Hardware: H100; inference batch size 3.

- Selected fixed beta=2.0; eight heads.

| Calibration beta | Before % | After % | Gain pp | Preservation % | Stale corrected % |
| --- | --- | --- | --- | --- | --- |
| 0.5 | 40.00 | 40.42 | +0.42 | 100.00 | 0.78 |
| 1.0 | 40.00 | 41.67 | +1.67 | 100.00 | 3.12 |
| 2.0 | 40.00 | 42.50 | +2.50 | 100.00 | 3.91 |
| 4.0 | 40.00 | 42.50 | +2.50 | 100.00 | 3.91 |
| 8.0 | 40.00 | 41.25 | +1.25 | 100.00 | 2.34 |


Frozen heads: L38H35, L38H38, L38H36, L35H11, L36H23, L43H34, L37H16, L43H32.


Identity diagnostic: response changes=0; max first-token score difference=0.0.


### Llama-3.2-3B

- Checkpoint: `meta-llama/Llama-3.2-3B-Instruct`.
- Canonical source: `results/stage_r/crossmodel/llama32_3b/r13_causal_full_h100/summary.json`.
- Hardware: H100; inference batch size 12.

- Frozen calibration: `results/stage_r/crossmodel/llama32_3b/r13_causal_calibration_h100/summary.json`.
- Discovery/validation rows: 120/120; discovery correct/stale pools: 34/61.
- First-token discovery exclusions: 6; candidate audit: `{"accuracy": 0.9894736842105263, "matches": 94, "n": 95}`.
- Selected top-32, m=4.0; beta cap=8.


Validation grid: net retrieval gain pp (a trailing ! means preservation <95%). Rows are head counts; columns are target margins. These are selection data, not held-out estimates.


| Heads | m=0.5 | m=1 | m=2 | m=4 |
| --- | --- | --- | --- | --- |
| 1 | +2.50 | +4.17 | +7.50 | +11.67 |
| 2 | +4.17 | +5.83 | +13.33 | +17.50 |
| 4 | +8.33 | +11.67 | +18.33 | +30.83 |
| 8 | +9.17 | +15.00 | +21.67 | +40.00 |
| 16 | +18.33 | +23.33 | +35.83 | +52.50 |
| 32 | +26.67 | +32.50 | +46.67 | +60.00 |


Selected validation point: 28.33% -> 88.33%; preservation 100.00%; stale correction 89.39%.


Frozen heads: L21H20, L16H15, L13H5, L26H10, L15H18, L26H11, L18H9, L24H4, L18H10, L17H9, L27H16, L18H19, L21H18, L27H10, L26H21, L20H20, L14H2, L13H17, L24H3, L19H19, L19H11, L16H14, L19H0, L22H5, L18H18, L23H11, L21H19, L26H9, L22H10, L17H0, L16H17, L12H3.


Identity diagnostic: response changes=0; max first-token score difference=0.0.


### Llama-3.1-8B

- Checkpoint: `meta-llama/Llama-3.1-8B-Instruct`.
- Canonical source: `results/stage_r/crossmodel/llama31_8b/r12_causal_full_h100/summary.json`.
- Hardware: H100; inference batch size 12.

- Frozen calibration: `results/stage_r/crossmodel/llama31_8b/r12_causal_smoke_h100/summary.json`.
- Discovery/validation rows: 12/12; discovery correct/stale pools: 4/7.
- First-token discovery exclusions: 0; candidate audit: `{"accuracy": 1.0, "matches": 11, "n": 11}`.
- Selected top-16, m=4.0; beta cap=8.


Validation grid: net retrieval gain pp (a trailing ! means preservation <95%). Rows are head counts; columns are target margins. These are selection data, not held-out estimates.


| Heads | m=0.5 | m=1 | m=2 | m=4 |
| --- | --- | --- | --- | --- |
| 1 | +0.00 | +0.00 | +8.33 | +8.33 |
| 2 | +0.00 | +0.00 | +8.33 | +16.67 |
| 4 | +8.33 | +8.33 | +8.33 | +16.67 |
| 8 | +8.33 | +8.33 | +16.67 | +41.67 |
| 16 | +8.33 | +16.67 | +41.67 | +50.00 |
| 32 | +33.33 | +33.33 | +33.33 | +41.67 |


Selected validation point: 50.00% -> 100.00%; preservation 100.00%; stale correction 100.00%.


Frozen heads: L24H27, L30H15, L17H24, L16H1, L24H26, L27H4, L22H15, L31H21, L26H13, L18H20, L27H6, L20H12, L19H13, L18H16, L18H28, L14H7.


Identity diagnostic: response changes=0; max first-token score difference=0.0.


### Mistral-7B-v0.3

- Checkpoint: `mistralai/Mistral-7B-Instruct-v0.3`.
- Canonical source: `results/stage_r/crossmodel/mistral7b_instruct/r13_causal_full_a100/summary.json`.
- Hardware: A100; inference batch size 8.

- Frozen calibration: `results/stage_r/crossmodel/mistral7b_instruct/r13_causal_calibration_a100/summary.json`.
- Discovery/validation rows: 120/120; discovery correct/stale pools: 9/78.
- First-token discovery exclusions: 6; candidate audit: `{"accuracy": 0.9770114942528736, "matches": 85, "n": 87}`.
- Selected top-32, m=2.0; beta cap=8.


Validation grid: net retrieval gain pp (a trailing ! means preservation <95%). Rows are head counts; columns are target margins. These are selection data, not held-out estimates.


| Heads | m=0.5 | m=1 | m=2 | m=4 |
| --- | --- | --- | --- | --- |
| 1 | +4.17 | +4.17 | +5.83 | +11.67 |
| 2 | +5.00 | +5.83 | +10.83 | +25.00 |
| 4 | +5.00 | +7.50 | +17.50 | +33.33 ! |
| 8 | +12.50 | +22.50 ! | +39.17 ! | +49.17 ! |
| 16 | +33.33 | +45.00 | +62.50 | +74.17 ! |
| 32 | +45.83 | +56.67 | +68.33 | +75.00 ! |


Selected validation point: 10.00% -> 78.33%; preservation 100.00%; stale correction 80.95%.


Frozen heads: L19H9, L19H16, L16H29, L18H3, L31H19, L21H7, L19H8, L16H1, L31H17, L18H12, L25H29, L31H18, L31H4, L24H15, L18H30, L24H5, L30H1, L26H17, L21H11, L24H21, L31H6, L20H14, L28H25, L29H22, L27H29, L29H9, L30H2, L20H6, L17H0, L31H31, L22H1, L26H6.


Identity diagnostic: response changes=0; max first-token score difference=0.0.


### Gemma-2-9B

- Checkpoint: `google/gemma-2-9b-it`.
- Canonical source: `results/stage_r/crossmodel/gemma2_9b_it/r14_causal_full_h100/retrieval/task_summary.json`.
- Hardware: H100; inference batch size 6.

- Frozen calibration: `results/stage_r/crossmodel/gemma2_9b_it/r14_causal_calibration_h100/summary.json`.
- Discovery/validation rows: 120/120; discovery correct/stale pools: 53/56.
- First-token discovery exclusions: 0; candidate audit: `{"accuracy": 1.0, "matches": 109, "n": 109}`.
- Selected top-32, m=4.0; beta cap=8.


Validation grid: net retrieval gain pp (a trailing ! means preservation <95%). Rows are head counts; columns are target margins. These are selection data, not held-out estimates.


| Heads | m=0.5 | m=1 | m=2 | m=4 |
| --- | --- | --- | --- | --- |
| 1 | +0.00 | +0.00 | +0.83 | +0.83 |
| 2 | +0.00 | +0.00 | +0.83 | +3.33 |
| 4 | +0.83 | +1.67 | +4.17 | +5.83 |
| 8 | +2.50 | +4.17 | +6.67 | +18.33 |
| 16 | +4.17 | +5.83 | +12.50 | +35.00 |
| 32 | +5.00 | +9.17 | +21.67 | +40.83 |


Selected validation point: 50.00% -> 90.83%; preservation 100.00%; stale correction 90.20%.


Frozen heads: L28H8, L28H2, L28H12, L35H1, L40H14, L32H7, L39H8, L33H1, L35H0, L37H12, L31H15, L36H12, L30H1, L31H13, L32H1, L28H4, L33H7, L30H15, L35H14, L32H13, L34H14, L26H0, L37H1, L26H4, L29H1, L37H6, L28H9, L32H15, L34H15, L25H13, L25H7, L36H2.


Identity diagnostic: response changes=7; max first-token score difference=1.8125.


## 8. Qualifications that must survive manuscript compression

**Gemma numerical identity.** The completed retrieval task records seven changed
responses at the identity setting relative to the first baseline (maximum
first-token score difference 1.8125). A separate diagnostic reproduces exactly
those differences between the first and repeated baselines; repeated baseline
and identity are exactly equal. No non-finite scores were observed. This
suggests execution-order/state dependence, not a demonstrated scientific null,
but the underlying cause remains unresolved. The identity check was explicitly
excluded from acceptance before the final run. Do not call it an exact first-
baseline identity pass. See the saved-report sensitivity below.

**Gemma completion.** Retrieval and all its controls completed. Derived decision
timed out during random-head evaluation and did not persist its response arms
or summary. It is unassessed, not a negative result. No additional jobs are
requested by this writing closeout.

**Llama-8B split exception.** The smoke-frozen configuration saw 24 rows,
including `stage_l_natfact_1_00043` from the eventual confirmation partition.
The primary sample excludes this ID; 959 rows are eligible. The 960-row
sensitivity must not replace the primary result. Later model calibrations
split before limiting and do not have this overlap. The earlier validity note
recommended recalibration; PI instead authorized confirmation with exclusion.

**Mistral preservation precision.** Only 52 confirmation items were initially
correct, and 51 remain correct. This supports "little observed damage" rather
than zero harm or a precise universal preservation probability. Its original
internal all_pass field remains false because the correct pool was below 100;
that does not erase a large measured correction. The manuscript should report
the actual count, not hide it behind either a pass or fail label.

**Scope and baseline strength.** A schema-aware parser already finds the current
value. The explicit-recap baseline inserts it into the prompt and often beats
routing. Routing therefore demonstrates an internal intervention lever without
rewriting the prompt, not superiority to simple external memory management.
Open-weight attention-hook access is required; text-only API deployment is not
shown. Neither arbitrary conversation nor agent-level reliability was tested.

**No causal overreach.** Targeted-vs-random effects support the selected operation
influencing retrieval. They do not prove a unique minimal circuit, pure QK-only
root cause, identical heads across families, or absence of late-layer answer
injection effects. Gradients incorporate downstream computation, and several
selected sets include final layers. Earlier Stage-L final-layer steering and
Stage-R routing are different interventions and must not be merged.

**Scale.** Llama-3B and 8B both improve under the adaptive method, but they are
different model versions and have different calibration sizes. Qwen-3B/7B/14B
mix adaptive and fixed protocols, so a decreasing or increasing efficacy
scaling law cannot be inferred. The valid statement is demonstrated correction
at multiple tested sizes, with model-dependent magnitude and limitations.

## 9. Earlier repair attempts: retain the non-successes

These historical arms are not interchangeable with natural-dialogue adaptive
confirmation. They explain the method's development, not additional independent
replications. Keep their records even if the main text is concise.

- R initial global per-head direction steering, Qwen-1.5B: approximately +0.8 pp
  held-out gain with a CI crossing zero, and no superiority to matched random
  directions. Source: `results/stage_r/full_h100/summary.json` and REPORT.
- R2 fixed causal-head attenuation, Qwen-1.5B: no eligible nonidentity calibration
  gamma; the fresh-seed confirmation defaulted to gamma=1 (identity). Its zero
  gain is not evidence that every attenuation strength is ineffective.
  Source: `results/stage_r/attenuation/full_h100/summary.json` and REPORT.
- R9 structured-log stale-key suppression, Qwen-1.5B: approximately +30.1 pp,
  62.8% stale correction, 97.2% preservation, automatic and oracle routes agree;
  targeted gain did NOT exceed the random-head reference range. This supports
  routing efficacy, not specificity of that head set. Source:
  `results/stage_r/routing/full_h100/summary.json` and REPORT.
- R11 Llama-8B descriptive-head smoke did not repair retrieval. Replacing that
  pipeline with R12 changes both head selection and bias policy; the later
  success cannot isolate which change was individually necessary. Source:
  `results/stage_r/crossmodel/llama31_8b/smoke_h100/summary.json`.
- Historical R10 paper summary forbids a cross-model claim because it predates
  R11--R14; that restriction describes R10 alone, not the accumulated evidence.
  Do not retroactively replace old verdicts with the new synthesis.

## 10. Paper-ready wording and presentation choices

Suggested main claim:

> We translate the diagnosis into a training-free attention-routing intervention.
> Using model-specific calibration, it improves current-binding retrieval across
> four model families on controlled natural dialogues, while retaining a high
> fraction of initially correct answers. The intervention reads update structure
> from the dialogue and changes attention scores, rather than supplying the
> answer through the output logits. Its scope is schema-aware retrieval with
> access to model internals; downstream decision gains are not consistent.

Suggested methods paragraph:

> A deterministic parser identifies current and superseded value spans from the
> dialogue. We bias selected query heads toward current spans and away from stale
> spans at the final prompt position. In the adaptive variant, each bias is the
> capped amount needed to reach a calibrated current-versus-stale attention-mass
> margin. We rank heads by intervention gradients on discovery examples, select
> the head count and margin on validation examples, and freeze the configuration
> for confirmation. Model weights remain unchanged, and inference uses neither
> gold annotations nor baseline error labels. The earlier fixed-bias runs and
> the Llama-8B smoke-calibration exception are reported separately.

Recommended compact main display:

1. Before/after accuracy dots for all seven checkpoints; mark fixed vs adaptive
   protocols and a Gemma qualification symbol. Include Qwen-14B, not only large
   gains. A separate gain plot can use the equal-cluster means with their stored
   CIs; the summaries do not supply CIs for the individual before/after rates.
2. Stale correction and correct preservation with numerator/denominator labels.
   Avoid implying that 51/52 is as precise as 452/452. Do not invent CI bars for
   these rates from the net-gain CI; repeated semantic clusters matter.
3. Targeted gain against random-head/position ranges; include recap accuracy
   and opposite-direction results in a compact table. Do not silently omit
   a simple baseline because it beats routing.
4. A method inset: raw dialogue -> schema-aware span parser -> model-specific
   heads -> balanced adaptive score bias -> ordinary generation. Explain beta
   and margin with the two short equations above, not a large symbol inventory.

Existing Qwen-7B-only assets are `results/stage_r/figures/`:
`R10_correction_evidence`, `R10_inference_transitions`, and
`R10_attention_routing_method` (PDF/PNG; generator `src/stage_r_figures.py`).
The method panel is schematic, not a measured attention heatmap. These figures
predate adaptive cross-model results; neither their fixed-beta diagram nor
their old captions should be presented as a summary of all models. The new
tables and companion exact-data file provide the inputs for replacement plots.

Use the per-cell table to support robustness across the designed competitors.
Keep derived-decision transfer visible as a concise limitation or auxiliary
table. Full head lists, calibration curves, arm compositions, and provenance
can live in supplementary materials; local paths belong only in this record.

Avoid: "we solve forgetting", "universally deployable", "zero damage across all
models", "all seven use the same intervention configuration", "all prerequisites
passed", "Gemma downstream null", "routing beats recap on every model", or a
monotonic repair-vs-scale claim. Internal gate names need not appear in prose,
but their substantive limitations must not disappear.


### Additional Gemma identity sensitivity

Excluding the 7 diagnostic-discrepant IDs from the saved paired arms leaves N=953: baseline 451, routed 875, raw gain +44.49 pp; stale corrected 402/450 (89.33%); correct preserved 451/451. No new CI was computed. This does not prove numerical equivalence of execution paths or recover the missing repeated-baseline output rows.


With targeted responses fixed, changing at most seven baseline classifications bounds the raw gain to [+44.06, +45.52] pp. This conditional bound does not cover arbitrary unobserved state-dependent effects on the targeted arm. Diagnostic IDs: `stage_l_natfact_1_00099, stage_l_natfact_2_00016, stage_l_natfact_2_00054, stage_l_natfact_2_00149, stage_l_natfact_3_00099, stage_l_natfact_4_00142, stage_l_natfact_4_00180`.


## 11. Real saved response examples

Selection rule: first ID in lexicographic order with baseline within-stale -> routed correct for each model. These are illustrations selected by outcome, not an unbiased case sample. Text is copied from saved responses and source dialogue, not reconstructed.


### Qwen2.5-3B: `stage_l_natfact_1_00000`

Slot: diet; current value: `mediterranean`; stale values: `["gluten free"]`.


- Message 3, user: I want to switch to gluten free meals.


- Message 7, user: Can you set my meal style to mediterranean?


- Message 35, user: I am still thinking about gluten free when I compare options for my meal style.


Final query:

> For my meal style specifically, what is the latest value I chose for that target slot, not any other preference? Reply with only the value.

Baseline response:

```text
gluten free
```

Routed response:

```text
mediterranean
```


### Qwen2.5-7B: `stage_l_natfact_1_00008`

Slot: learning_style; current value: `discussion`; stale values: `["hands on"]`.


- Message 3, user: I'd say hands on is the method that helps me most.


- Message 7, user: I think discussion works best for me overall.


- Message 29, user: I am still thinking about hands on when I compare options for my learning style.


Final query:

> Look back at the updates for my learning style. What is the current value for that exact slot, not any other preference? Reply with only the value.

Baseline response:

```text
hands on
```

Routed response:

```text
discussion
```


### Qwen2.5-14B: `stage_l_natfact_1_00034`

Slot: learning_style; current value: `reading`; stale values: `["discussion"]`.


- Message 3, user: I think discussion works best for me overall.


- Message 7, user: I actually prefer reading when I'm trying to understand something new.


Final query:

> I am asking only about my learning style. What value is current for that slot now, not any other preference? Reply with only the value.

Baseline response:

```text
Discussion
```

Routed response:

```text
Reading
```


### Llama-3.2-3B: `stage_l_natfact_1_00001`

Slot: learning_style; current value: `visual`; stale values: `["reading"]`.


- Message 3, user: I think reading works best for me overall.


- Message 7, user: I actually prefer visual when I'm trying to understand something new.


- Message 41, user: I am still thinking about reading when I compare options for my learning style.


Final query:

> I am asking only about my learning style. What value is current for that slot now, not any other preference? Reply with only the value.

Baseline response:

```text
Reading
```

Routed response:

```text
Visual
```


### Llama-3.1-8B: `stage_l_natfact_1_00001`

Slot: learning_style; current value: `visual`; stale values: `["reading"]`.


- Message 3, user: I think reading works best for me overall.


- Message 7, user: I actually prefer visual when I'm trying to understand something new.


- Message 41, user: I am still thinking about reading when I compare options for my learning style.


Final query:

> I am asking only about my learning style. What value is current for that slot now, not any other preference? Reply with only the value.

Baseline response:

```text
Reading
```

Routed response:

```text
Visual
```


### Mistral-7B-v0.3: `stage_l_natfact_1_00000`

Slot: diet; current value: `mediterranean`; stale values: `["gluten free"]`.


- Message 3, user: I want to switch to gluten free meals.


- Message 7, user: Can you set my meal style to mediterranean?


- Message 35, user: I am still thinking about gluten free when I compare options for my meal style.


Final query:

> For my meal style specifically, what is the latest value I chose for that target slot, not any other preference? Reply with only the value.

Baseline response:

```text
Gluten free.
```

Routed response:

```text
Mediterranean
```


### Gemma-2-9B: `stage_l_natfact_1_00002`

Slot: music_genre; current value: `folk`; stale values: `["rock"]`.


- Message 3, user: Can you change my selection to rock?


- Message 7, user: Make folk my preferred style.


- Message 29, user: I am still thinking about rock when I compare options for my music genre.


Final query:

> Look back at the updates for my music genre. What is the current value for that exact slot, not any other preference? Reply with only the value.

Baseline response:

```text
rock
```

Routed response:

```text
folk
```


## 12. Hardware mirrors and historical records

Mirrors share data/protocol and are not extra model replications. Preserve both, but use the canonical run consistently.

| Mirror | Before % | After % | Gain pp | Correct preserved | Stale corrected | Source |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-7B A100 | 36.88 | 68.85 | +31.98 | 354/354 (100.00%) | 198/478 (41.42%) | `results/stage_r/routing/balanced_full_a100/summary.json` |
| Qwen2.5-14B A100 | 40.52 | 41.56 | +1.04 | 388/389 (99.74%) | 10/494 (2.02%) | `results/stage_r/crossmodel/qwen25_14b/full_a100/summary.json` |


### Earlier prototypes: exact summary values

These are separate substrates/protocols, not rows in the natural-dialogue comparison. Values here are read from historical summaries; their raw prototype arms were not re-audited by this builder.

| Prototype | N | Before % | After % | Raw gain pp | Saved gain 95% interval pp | Preservation % | Stale correction % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R global direction | 236 | 49.58 | 50.42 | +0.85 | [-0.85, +2.54] | 99.15 | 2.52 |
| R2 attenuation | 648 | 49.85 | 49.85 | +0.00 | [+0.00, +0.00] | 100.00 | 0.00 |
| R9 automatic suppression | 648 | 49.85 | 79.94 | +30.09 | [+25.15, +35.03] | 97.21 | 62.77 |
| R11 Llama descriptive-head smoke | 48 | 43.75 | 41.67 | -2.08 | [-8.33, +0.00] | 95.24 | 0.00 |


R initial uses a row-paired interval; later prototype summaries label their semantic-cluster intervals explicitly. Do not mix their uncertainty estimands with the seven-model canonical comparison. R2's operating point is identity; R9 fails head specificity.


## 13. Internal audit and artifact map

This section is for writers, not for verbatim inclusion in the manuscript.


| Model | Original retrieval gate fields |
| --- | --- |
| Qwen2.5-3B | `{"adequate_correct_pool": true, "adequate_within_stale_pool": true, "all_pass": true, "correct_preservation": true, "exceeds_random_head95": true, "exceeds_random_position95": true, "identity_exact": true, "net_gain_ci_above_zero": true, "positive_in_four_cells": true, "route_audit_exact": true}` |
| Qwen2.5-7B | `{"adequate_correct_pool": true, "adequate_within_stale_pool": true, "all_pass": true, "correct_preservation": true, "exceeds_random_position95": true, "identity_exact": true, "net_gain_ci_above_zero": true, "positive_in_four_cells": true, "route_audit_exact": true}` |
| Qwen2.5-14B | `{"adequate_correct_pool": true, "adequate_within_stale_pool": true, "all_pass": true, "correct_preservation": true, "exceeds_random_position95": true, "identity_exact": true, "net_gain_ci_above_zero": true, "positive_in_four_cells": true, "route_audit_exact": true}` |
| Llama-3.2-3B | `{"adequate_correct_pool": true, "adequate_within_stale_pool": true, "all_pass": true, "correct_preservation": true, "exceeds_random_head95": true, "exceeds_random_position95": true, "identity_exact": true, "net_gain_ci_above_zero": true, "positive_in_four_cells": true, "route_audit_exact": true}` |
| Llama-3.1-8B | `{"adequate_correct_pool": true, "adequate_within_stale_pool": true, "all_pass": true, "correct_preservation": true, "exceeds_random_head95": true, "exceeds_random_position95": true, "identity_exact": true, "net_gain_ci_above_zero": true, "positive_in_four_cells": true, "route_audit_exact": true}` |
| Mistral-7B-v0.3 | `{"adequate_correct_pool": false, "adequate_within_stale_pool": true, "all_pass": false, "correct_preservation": true, "exceeds_random_head95": true, "exceeds_random_position95": true, "identity_exact": true, "net_gain_ci_above_zero": true, "positive_in_four_cells": true, "route_audit_exact": true}` |
| Gemma-2-9B | `{"adequate_correct_pool": true, "adequate_within_stale_pool": true, "all_pass": true, "correct_preservation": true, "exceeds_random_head95": true, "exceeds_random_position95": true, "identity_exact": false, "identity_required": false, "net_gain_ci_above_zero": true, "positive_in_four_cells": true, "route_audit_exact": true}` |


For each canonical run, the task directory contains `baseline_rows.jsonl`, `routed_rows.jsonl`, `opposite_rows.jsonl`, `recap_rows.jsonl`, and random-control records. The summary contains all response transitions, per-cell results, and paired bootstrap estimates. Gemma retrieval's task summary is authoritative because no combined summary was emitted.


Method implementation: `src/stale_binding_natural_routing.py` (parser, fixed routing orchestration, splitting, scoring, recap, metrics); `src/stale_binding_routing.py` (fixed bias hook); `src/stage_r_crossmodel_routing.py` (descriptive head ranking); `src/stage_r_llama_causal_heads.py` (adaptive hook, gradient discovery, validation, controls); `src/stale_binding_attenuation.py` (cluster bootstrap). Runbooks: `results/stage_r/runbooks/`.


Audit performed by this builder: aligned unique IDs and counts for all saved baseline/targeted/opposite/recap arms; exact response-category transitions; raw gains and equal-cluster point estimates; per-cell counts/accuracy; presence and count of random-control records. Saved bootstrap endpoints are read, not recomputed. This is a saved-artifact audit, not a fresh scoring/parser/model validation.


Companion `repair_writeup_data.json` retains exact selected summaries, all calibration curves, and SHA-256 hashes of every read artifact. Regenerate with:

```bash
/anvil/projects/x-cis250190/software/envs/verl/bin/python src/stage_r_repair_writeup.py
```
