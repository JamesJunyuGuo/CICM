# Stage R10 figure captions

## R10_correction_evidence

**Balanced attention routing improves direct current-binding retrieval.**
**a,** Retrieval calibration selects beta=8 from a fixed dose grid; the dashed
line is the unmodified baseline. **b,** On 960 held-out confirmation examples,
balanced routing increases accuracy by 31.2 percentage points (semantic-ID
clustered 95% bootstrap interval), whereas 16 count-matched random-position
routes remain near zero and reversing the route is harmful. An explicit recap
is shown as a deployment baseline because it places the answer in the prompt.
**c,** The gain is positive in every identity-by-recency cell (n=160 per cell).

## R10_inference_transitions

**The intervention corrects retrieval errors but does not transfer to a derived
decision task.** Rows condition on the baseline response type and colors show
the response after routing. In direct retrieval, all initially correct answers
remain correct, 42.7% of within-slot stale responses are repaired, and 84.7% of
cross-slot responses move to the current value. The same route produces no net
benefit on the derived decision task.

## R10_attention_routing_method

**Training-free attention routing changes which contextual write is read.** A
real held-out example contains several superseded meal-style values followed
by the current value. An automatic dialogue-only detector identifies the
relevant writes; at the answer query, the intervention subtracts beta from
old-write attention logits and adds beta to the current-write logit in eight
frozen heads. Here beta is an additive bias to the pre-softmax query--key score,
selected on the calibration split; the hook acts during prompt prefill only,
after which greedy decoding proceeds normally with unchanged model weights.
The unmodified model returns a stale value, while the routed decode returns the
current value. Panel b depicts the intervention operator, not a measured
attention heatmap.
