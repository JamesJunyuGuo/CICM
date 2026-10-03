"""Audit saved Stage-R repair arms and assemble a paper-writing reference.

CPU-only standard-library code. No inference, fitting, or artifact mutation.
Run from the repository root; outputs are a new synthesis and audit JSON.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/stage_r"
DATA = ROOT / "data/stage_l/cicm_natural_factorial_otherdist_l0.jsonl"
RUNS = [
    ("Qwen2.5-3B", "crossmodel/qwen25_3b/r14_causal_full_h100", "adaptive", 12),
    ("Qwen2.5-7B", "routing/balanced_full_h100", "fixed", 12),
    ("Qwen2.5-14B", "crossmodel/qwen25_14b/full_h100", "fixed", 3),
    ("Llama-3.2-3B", "crossmodel/llama32_3b/r13_causal_full_h100", "adaptive", 12),
    ("Llama-3.1-8B", "crossmodel/llama31_8b/r12_causal_full_h100", "adaptive", 12),
    ("Mistral-7B-v0.3", "crossmodel/mistral7b_instruct/r13_causal_full_a100", "adaptive", 8),
    ("Gemma-2-9B", "crossmodel/gemma2_9b_it/r14_causal_full_h100", "adaptive", 6),
]
SOURCES = {}


def read_json(path):
    path = Path(path)
    payload = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def read_rows(path):
    path = Path(path)
    payload = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fingerprint(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def pct(value):
    return f"{100 * value:.2f}"


def gain(value):
    return f"{100 * value:+.2f}"


def interval(values):
    return f"[{gain(values[0])}, {gain(values[1])}]"


def fraction(metric, label):
    n = metric["baseline_counts"].get(label, 0)
    k = metric["transitions"].get(f"{label}->correct_current", 0)
    return f"{k}/{n} ({pct(k / n)}%)" if n else "not observed (n=0)"


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ]) + "\n"


def audit_arm(rows, baseline, metric):
    ids = [r["id"] for r in baseline]
    require(ids == [r["id"] for r in rows], "unaligned arms")
    require(len(set(ids)) == len(ids) == metric["n"], "duplicate IDs/count mismatch")
    require(dict(Counter(r["label"] for r in baseline)) == metric["baseline_counts"], "baseline counts")
    require(dict(Counter(r["label"] for r in rows)) == metric["arm_counts"], "arm counts")
    trans = dict(Counter(f"{a['label']}->{b['label']}" for a, b in zip(baseline, rows)))
    require(trans == metric["transitions"], "transition mismatch")
    groups = defaultdict(list)
    for a, b in zip(baseline, rows):
        groups[a["semantic_id"]].append(int(b["label"] == "correct_current") - int(a["label"] == "correct_current"))
    total = sum(sum(v) for v in groups.values()) / len(rows)
    cluster_mean = sum(sum(v) / len(v) for v in groups.values()) / len(groups)
    require(math.isclose(total, metric["net_accuracy_gain"], abs_tol=1e-12), "raw gain mismatch")
    require(math.isclose(cluster_mean, metric["paired_net_gain"]["mean"], abs_tol=1e-12), "cluster gain mismatch")
    require(len(groups) == metric["paired_net_gain"]["n_clusters"], "cluster count mismatch")


def mixed_current(row):
    return row["label"] == "correct_current" and any(
        row.get(key) for key in ("stale_hits", "same_slot_other_hits", "cross_slot_hits")
    )


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def load_run(name, rel, protocol, batch):
    folder = BASE / rel
    if name == "Gemma-2-9B":
        source = folder / "retrieval/task_summary.json"
        calpath = folder.parent / "r14_causal_calibration_h100/summary.json"
        calibration = read_json(calpath)
        s = calibration["validation"]["selected"]
        summary = {
            "model": calibration["model"], "frozen_summary": str(calpath.relative_to(ROOT)),
            "selected": {"size": s["size"], "margin": s["margin"], "heads": calibration["head_discovery"]["top_heads"][:s["size"]]},
            "tasks": {"retrieval": read_json(source)},
            "excluded_from_primary_confirmation": [],
        }
    else:
        source = folder / "summary.json"
        summary = read_json(source)
        calibration = read_json(ROOT / summary["frozen_summary"]) if "frozen_summary" in summary else summary
    excluded = set(summary.get("excluded_from_primary_confirmation", []))
    raw_tasks = {}
    for task, result in summary["tasks"].items():
        arms = {arm: read_rows(folder / task / f"{arm}_rows.jsonl") for arm in ("baseline", "routed", "opposite", "recap")}
        require(all(len(v) == 960 for v in arms.values()), f"{name}/{task}: generated count")
        arms = {k: [r for r in v if r["id"] not in excluded] for k, v in arms.items()}
        for arm in ("routed", "opposite", "recap"):
            audit_arm(arms[arm], arms["baseline"], result[arm])
        for control, filename in (("random_positions", "random_position_controls.jsonl"), ("random_heads", "random_head_controls.jsonl")):
            if control not in result:
                continue
            records = read_rows(folder / task / filename)
            require(len(records) == result[control]["n"], f"{name}/{task}: incomplete controls")
            require(sorted(r["index"] for r in records) == list(range(len(records))), "control indices")
            values = [r["net_accuracy_gain"] for r in records]
            stats = result[control]["net_accuracy_gain"]
            require(math.isclose(sum(values) / len(values), stats["mean"], abs_tol=1e-12), "control mean")
            require(all(math.isclose(quantile(values, q), v, abs_tol=1e-12) for q, v in zip((.025, .975), stats["interval95"])), "control range")
            if control == "random_heads":
                selected = summary["selected"]["heads"]
                chosen = {(h["layer"], h["head"]) for h in selected}
                layer_counts = Counter(layer for layer, _ in chosen)
                for record in records:
                    heads = {(h["layer"], h["head"]) for h in record["heads"]}
                    require(len(heads) == len(chosen) and not heads.intersection(chosen), "random head set")
                    require(Counter(layer for layer, _ in heads) == layer_counts, "random head layer matching")
        for cell, expected in result["per_factorial_cell"].items():
            indices = [i for i, row in enumerate(arms["baseline"]) if
                       f"same_{row['factorial_cell']['same_slot_stale_distance_bin']}__other_{row['factorial_cell']['recent_other_slot_distance_bin']}" == cell]
            require(len(indices) == expected["n"], "cell count mismatch")
            for arm, field in (("baseline", "baseline_accuracy"), ("routed", "arm_accuracy")):
                value = sum(arms[arm][i]["label"] == "correct_current" for i in indices) / len(indices)
                require(math.isclose(value, expected[field], abs_tol=1e-12), "cell accuracy mismatch")
        raw_tasks[task] = arms
    return {"name": name, "path": str(source.relative_to(ROOT)), "folder": rel,
            "protocol": protocol, "batch": batch, "summary": summary, "calibration": calibration,
            "raw": raw_tasks}


INTRO = r"""# Cross-model test-time repair: complete paper-writing reference

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
"""

METHODS = r"""
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
"""

LIMITS = r"""
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
"""


def build():
    data = read_rows(DATA)
    by_id = {r["id"]: r for r in data}
    runs = [load_run(*args) for args in RUNS]
    canonical_ids = {r["id"] for r in runs[0]["raw"]["retrieval"]["baseline"]}
    for run in runs:
        ids = {r["id"] for r in run["raw"]["retrieval"]["baseline"]}
        excluded = set(run["summary"].get("excluded_from_primary_confirmation", []))
        require(ids == canonical_ids - excluded, "cross-model split mismatch")
    out = [INTRO]
    rows = []
    for run in runs:
        m = run["summary"]["tasks"]["retrieval"]["routed"]
        b = m["paired_net_gain"]
        rows.append([run["name"], run["protocol"], m["n"], pct(m["baseline_accuracy"]), pct(m["arm_accuracy"]), gain(m["net_accuracy_gain"]), f"{gain(b['mean'])} {interval(b['ci'])}", fraction(m, "within_stale"), fraction(m, "correct_current")])
    out.append(table(["Model", "Policy", "N", "Before %", "After %", "Raw gain pp", "Cluster gain pp [95% CI]", "Stale corrected", "Correct preserved"], rows))
    out.append("Gemma is identity-qualified; Llama-8B excludes one exposed item. All rates describe this factorial, not natural prevalence.\n")
    adaptive = [r["summary"]["tasks"]["retrieval"]["routed"] for r in runs if r["protocol"] == "adaptive"]
    out.append(f"\nNumerical highlights for selection: the five adaptive configurations yield raw gains of {gain(min(m['net_accuracy_gain'] for m in adaptive))} to {gain(max(m['net_accuracy_gain'] for m in adaptive))} pp; stale-correction rates {pct(min(m['within_stale_correction'] for m in adaptive))}%--{pct(max(m['within_stale_correction'] for m in adaptive))}%; and observed correct preservation {pct(min(m['correct_preservation'] for m in adaptive))}%--100%. This range includes the qualified Gemma run and is not an average or a scaling law.\n")
    out.append(METHODS)
    out.append("\n## 5. Control results and downstream transfer\n\n### 5.1 Retrieval controls\n\nAll gains below are row-weighted pp. Random bands are empirical ranges across control sets, not paired-effect CIs.\n")
    rows = []
    for run in runs:
        t = run["summary"]["tasks"]["retrieval"]
        def control(key):
            if key not in t:
                return "not run in this protocol"
            r = t[key]["net_accuracy_gain"]
            return f"{gain(r['mean'])} {interval(r['interval95'])}; n={t[key]['n']}"
        rows.append([run["name"], gain(t["routed"]["net_accuracy_gain"]), control("random_positions"), control("random_heads"), gain(t["opposite"]["net_accuracy_gain"]), pct(t["recap"]["arm_accuracy"]), gain(t["routed"]["arm_accuracy"] - t["recap"]["arm_accuracy"])])
    out.append(table(["Model", "Targeted gain", "Random positions mean [range]", "Random heads mean [range]", "Opposite gain", "Recap accuracy %", "Targeted - recap pp"], rows))
    out.append("\nRecap comes from the same automatic parser, not an external judge. It explicitly inserts the inferred current value in an assistant message before the query. Targeted-minus-recap above is descriptive; no paired significance test against recap was added.\n")
    out.append("\n### 5.2 Derived decision (no retuning)\n")
    rows = []
    for run in runs:
        task = run["summary"]["tasks"].get("derived_decision")
        if task is None:
            rows.append([run["name"], "incomplete", "not assessed", "not assessed", "not assessed", "not assessed", "not assessed", "not assessed"])
            continue
        m = task["routed"]
        b = m["paired_net_gain"]
        rows.append([run["name"], m["n"], pct(m["baseline_accuracy"]), pct(m["arm_accuracy"]), gain(m["net_accuracy_gain"]), f"{gain(b['mean'])} {interval(b['ci'])}", fraction(m, "correct_current"), pct(task["recap"]["arm_accuracy"])])
    out.append(table(["Model", "N", "Before %", "After %", "Raw gain pp", "Cluster gain pp [95% CI]", "Correct preserved", "Recap accuracy %"], rows))
    out.append("\nQwen-3B provides a small positive transfer result; the other completed arms do not support a consistent downstream benefit. A CI including zero is insufficient evidence of benefit, not proof of exactly zero effect. Gemma is missing, not null.\n")
    out.append("\n### 5.3 Derived-decision controls and stale corrections\n\nThese use the same frozen retrieval configuration. Random intervals are empirical ranges in pp.\n")
    rows = []
    for run in runs:
        t = run["summary"]["tasks"].get("derived_decision")
        if t is None:
            continue
        def derived_control(key):
            if key not in t:
                return "not run"
            v = t[key]["net_accuracy_gain"]
            return f"{gain(v['mean'])} {interval(v['interval95'])}"
        rows.append([run["name"], fraction(t["routed"], "within_stale"), derived_control("random_positions"), derived_control("random_heads"), gain(t["opposite"]["net_accuracy_gain"])])
    out.append(table(["Model", "Stale-code -> correct", "Random positions mean [range]", "Random heads mean [range]", "Opposite gain pp"], rows))
    out.append("\n## 6. Per-cell retrieval effects and error modes\n\n### 6.1 All six cells\n\nCells name nearest same-slot stale and other-slot distance. Entries are before -> after accuracy (raw gain pp). N=160 per cell except Llama-8B near/mid N=159.\n")
    keys = sorted(runs[0]["summary"]["tasks"]["retrieval"]["per_factorial_cell"])
    rows = []
    for key in keys:
        cells = [key]
        for run in runs:
            v = run["summary"]["tasks"]["retrieval"]["per_factorial_cell"][key]
            cells.append(f"{pct(v['baseline_accuracy'])} -> {pct(v['arm_accuracy'])} ({gain(v['net_accuracy_gain'])})")
        rows.append(cells)
    out.append(table(["Cell", *(r["name"] for r in runs)], rows))
    out.append("\n### 6.2 Repairs separated by baseline failure mode\n")
    rows = []
    for run in runs:
        m = run["summary"]["tasks"]["retrieval"]["routed"]
        lost = m["baseline_counts"].get("correct_current", 0) - m["transitions"].get("correct_current->correct_current", 0)
        positives = sum(v["net_accuracy_gain"] > 0 for v in run["summary"]["tasks"]["retrieval"]["per_factorial_cell"].values())
        rows.append([run["name"], fraction(m, "within_stale"), fraction(m, "cross_slot"), fraction(m, "other"), lost, f"{positives}/6"])
    out.append(table(["Model", "Stale -> correct", "Cross-slot -> correct", "Other -> correct", "Initially correct lost", "Positive cells"], rows))
    out.append("\n### 6.3 Full retrieval response composition\n\nCounts are correct / within-stale / cross-slot / other, using the primary denominator.\n")
    rows = []
    for run in runs:
        t = run["summary"]["tasks"]["retrieval"]
        counts = [t["routed"]["baseline_counts"], *(t[a]["arm_counts"] for a in ("routed", "opposite", "recap"))]
        rows.append([run["name"], *(" / ".join(str(c.get(k, 0)) for k in ("correct_current", "within_stale", "cross_slot", "other")) for c in counts)])
    out.append(table(["Model", "Baseline", "Targeted", "Opposite", "Recap"], rows))
    out.append("\n### 6.4 Scoring sensitivity: mixed-value responses\n\nPost-hoc CPU check, not a replacement for the frozen classifier. 'Unambiguous-current' here means a current hit with no other controlled-value hit; it is still not whole-string exact match. The sensitivity marks mixed-current responses incorrect in both arms. No new CI was computed.\n")
    rows = []
    for run in runs:
        a = run["raw"]["retrieval"]["baseline"]
        b = run["raw"]["retrieval"]["routed"]
        bc = [r["label"] == "correct_current" and not mixed_current(r) for r in a]
        rc = [r["label"] == "correct_current" and not mixed_current(r) for r in b]
        retained = sum(x and y for x, y in zip(bc, rc))
        rows.append([run["name"], sum(mixed_current(r) for r in a), sum(mixed_current(r) for r in b), gain((sum(rc) - sum(bc)) / len(a)), f"{retained}/{sum(bc)} ({pct(retained / sum(bc))}%)"])
    out.append(table(["Model", "Mixed-current baseline", "Mixed-current routed", "Unambiguous-current raw gain pp", "Unambiguous correct preserved"], rows))
    out.append("\n## 7. Frozen model configurations and calibration\n\nAll layer/head indices below are zero-based query-head indices. Lists are in frozen ranking order, not a cross-model alignment.\n")
    for run in runs:
        s, c = run["summary"], run["calibration"]
        out.append(f"\n### {run['name']}\n\n- Checkpoint: `{s['model']}`.\n- Canonical source: `{run['path']}`.\n- Hardware: {'A100' if 'a100' in run['folder'] else 'H100'}; inference batch size {run['batch']}.\n")
        if run["protocol"] == "adaptive":
            selected = s["selected"]
            heads = selected["heads"]
            discovery = c["head_discovery"]
            out.append(f"- Frozen calibration: `{s['frozen_summary']}`.\n- Discovery/validation rows: {c['n_discovery']}/{c['n_validation']}; discovery correct/stale pools: {discovery['n_correct']}/{discovery['n_within_stale']}.\n- First-token discovery exclusions: {len(discovery.get('candidate_exclusions', []))}; candidate audit: `{json.dumps(discovery['candidate_audit'], sort_keys=True)}`.\n- Selected top-{selected['size']}, m={selected['margin']}; beta cap=8.\n")
            curve = c["validation"]["curve"]
            out.append("\nValidation grid: net retrieval gain pp (a trailing ! means preservation <95%). Rows are head counts; columns are target margins. These are selection data, not held-out estimates.\n\n")
            margins = sorted({row["margin"] for row in curve})
            cells = {(row["size"], row["margin"]): row for row in curve}
            rows = []
            for size in sorted({row["size"] for row in curve}):
                vals = []
                for margin in margins:
                    v = cells.get((size, margin))
                    vals.append("not tested" if v is None else gain(v["net_accuracy_gain"]) + (" !" if v["correct_preservation"] < .95 else ""))
                rows.append([size, *vals])
            out.append(table(["Heads", *(f"m={m:g}" for m in margins)], rows))
            v = c["validation"]["selected"]
            out.append(f"\nSelected validation point: {pct(v['baseline_accuracy'])}% -> {pct(v['arm_accuracy'])}%; preservation {pct(v['correct_preservation'])}%; stale correction {pct(v['within_stale_correction'])}%.\n")
        else:
            heads = s["frozen_heads"]
            out.append(f"- Selected fixed beta={s['beta']}; eight heads.\n")
            out.append(table(["Calibration beta", "Before %", "After %", "Gain pp", "Preservation %", "Stale corrected %"], [[v["beta"], pct(v["baseline_accuracy"]), pct(v["arm_accuracy"]), gain(v["net_accuracy_gain"]), pct(v["correct_preservation"]), pct(v["within_stale_correction"])] for v in s["calibration"]["curve"]]))
        out.append("\nFrozen heads: " + ", ".join(f"L{h['layer']}H{h['head']}" for h in heads) + ".\n")
        t = s["tasks"]["retrieval"]
        out.append(f"\nIdentity diagnostic: response changes={t['identity_response_changes']}; max first-token score difference={t['identity_max_abs_first_token_score_change']}.\n")
    out.append(LIMITS)
    gemma = next(r for r in runs if r["name"] == "Gemma-2-9B")
    diagpath = BASE / gemma["folder"] / "retrieval/identity_diagnostic.json"
    diag = read_json(diagpath)
    repeatpath = BASE / "crossmodel/gemma2_9b_it/r14_identity_diag_h100/retrieval/identity_repeat_diagnostic.json"
    repeat = read_json(repeatpath)
    require(repeat["baseline_vs_identity"] == diag == repeat["baseline_vs_repeat"], "Gemma diagnostic discrepancy")
    require(repeat["repeat_vs_identity"]["score_arrays_exact"], "Gemma repeat identity differs")
    changed = set(diag["changed_response_ids"])
    pairs = [(a, b) for a, b in zip(gemma["raw"]["retrieval"]["baseline"], gemma["raw"]["retrieval"]["routed"]) if a["id"] not in changed]
    n = len(pairs)
    base = sum(a["label"] == "correct_current" for a, b in pairs)
    target = sum(b["label"] == "correct_current" for a, b in pairs)
    stale = sum(a["label"] == "within_stale" for a, b in pairs)
    fixed = sum(a["label"] == "within_stale" and b["label"] == "correct_current" for a, b in pairs)
    kept = sum(a["label"] == b["label"] == "correct_current" for a, b in pairs)
    out.append(f"\n### Additional Gemma identity sensitivity\n\nExcluding the {len(changed)} diagnostic-discrepant IDs from the saved paired arms leaves N={n}: baseline {base}, routed {target}, raw gain {gain((target-base)/n)} pp; stale corrected {fixed}/{stale} ({pct(fixed/stale)}%); correct preserved {kept}/{base}. No new CI was computed. This does not prove numerical equivalence of execution paths or recover the missing repeated-baseline output rows.\n\n")
    full = gemma["summary"]["tasks"]["retrieval"]["routed"]
    raw_gain = full["net_accuracy_gain"]
    out.append(f"With targeted responses fixed, changing at most seven baseline classifications bounds the raw gain to {interval([raw_gain - len(changed)/960, raw_gain + len(changed)/960])} pp. This conditional bound does not cover arbitrary unobserved state-dependent effects on the targeted arm. Diagnostic IDs: `{', '.join(sorted(changed))}`.\n")
    out.append("\n## 11. Real saved response examples\n\nSelection rule: first ID in lexicographic order with baseline within-stale -> routed correct for each model. These are illustrations selected by outcome, not an unbiased case sample. Text is copied from saved responses and source dialogue, not reconstructed.\n")
    for run in runs:
        arms = run["raw"]["retrieval"]
        candidates = [(a, b) for a, b in zip(arms["baseline"], arms["routed"]) if a["label"] == "within_stale" and b["label"] == "correct_current"]
        a, b = min(candidates, key=lambda pair: pair[0]["id"])
        d = by_id[a["id"]]
        out.append(f"\n### {run['name']}: `{a['id']}`\n\nSlot: {d['slot']}; current value: `{d['current_value']}`; stale values: `{json.dumps(d['stale_values'])}`.\n")
        indices = sorted({m["message_index"] for m in d["value_mentions"] if m["slot"] == d["slot"]})
        for i in indices:
            msg = d["messages"][i]
            out.append(f"\n- Message {i}, {msg['role']}: {msg['content']}\n")
        out.append(f"\nFinal query:\n\n> {d['messages'][-1]['content']}\n\nBaseline response:\n\n```text\n{a['response']}\n```\n\nRouted response:\n\n```text\n{b['response']}\n```\n")
    out.append("\n## 12. Hardware mirrors and historical records\n\nMirrors share data/protocol and are not extra model replications. Preserve both, but use the canonical run consistently.\n")
    mirrors = []
    for name, rel in [("Qwen2.5-7B A100", "routing/balanced_full_a100/summary.json"), ("Qwen2.5-14B A100", "crossmodel/qwen25_14b/full_a100/summary.json")]:
        s = read_json(BASE / rel)
        m = s["tasks"]["retrieval"]["routed"]
        mirrors.append([name, pct(m["baseline_accuracy"]), pct(m["arm_accuracy"]), gain(m["net_accuracy_gain"]), fraction(m, "correct_current"), fraction(m, "within_stale"), f"`results/stage_r/{rel}`"])
    out.append(table(["Mirror", "Before %", "After %", "Gain pp", "Correct preserved", "Stale corrected", "Source"], mirrors))
    out.append("\n### Earlier prototypes: exact summary values\n\nThese are separate substrates/protocols, not rows in the natural-dialogue comparison. Values here are read from historical summaries; their raw prototype arms were not re-audited by this builder.\n")
    historical = {}
    rows = []
    for label, rel in [("R global direction", "full_h100/summary.json"), ("R2 attenuation", "attenuation/full_h100/summary.json"), ("R9 automatic suppression", "routing/full_h100/summary.json"), ("R11 Llama descriptive-head smoke", "crossmodel/llama31_8b/smoke_h100/summary.json")]:
        d = read_json(BASE / rel)
        historical[label] = d
        if label == "R global direction":
            m, paired = d["heldout"]["targeted"], d["targeted_paired_gain"]
        elif label == "R2 attenuation":
            m = d["confirmation"]["targeted"]
            paired = m["paired_net_gain"]
        elif label == "R9 automatic suppression":
            m = d["confirmation"]["automatic"]
            paired = m["paired_net_gain"]
        else:
            m = d["tasks"]["retrieval"]["routed"]
            paired = m["paired_net_gain"]
        rows.append([label, m["n"], pct(m["baseline_accuracy"]), pct(m["arm_accuracy"]), gain(m["net_accuracy_gain"]), interval(paired["ci"]), pct(m["correct_preservation"]), pct(m.get("correction", m.get("within_stale_correction")))])
    out.append(table(["Prototype", "N", "Before %", "After %", "Raw gain pp", "Saved gain 95% interval pp", "Preservation %", "Stale correction %"], rows))
    out.append("\nR initial uses a row-paired interval; later prototype summaries label their semantic-cluster intervals explicitly. Do not mix their uncertainty estimands with the seven-model canonical comparison. R2's operating point is identity; R9 fails head specificity.\n")
    out.append("\n## 13. Internal audit and artifact map\n\nThis section is for writers, not for verbatim inclusion in the manuscript.\n\n")
    out.append(table(["Model", "Original retrieval gate fields"], [[r["name"], "`" + json.dumps(r["summary"]["tasks"]["retrieval"]["gate"], sort_keys=True) + "`"] for r in runs]))
    out.append("\nFor each canonical run, the task directory contains `baseline_rows.jsonl`, `routed_rows.jsonl`, `opposite_rows.jsonl`, `recap_rows.jsonl`, and random-control records. The summary contains all response transitions, per-cell results, and paired bootstrap estimates. Gemma retrieval's task summary is authoritative because no combined summary was emitted.\n\n")
    out.append("Method implementation: `src/stale_binding_natural_routing.py` (parser, fixed routing orchestration, splitting, scoring, recap, metrics); `src/stale_binding_routing.py` (fixed bias hook); `src/stage_r_crossmodel_routing.py` (descriptive head ranking); `src/stage_r_llama_causal_heads.py` (adaptive hook, gradient discovery, validation, controls); `src/stale_binding_attenuation.py` (cluster bootstrap). Runbooks: `results/stage_r/runbooks/`.\n\n")
    out.append("Audit performed by this builder: aligned unique IDs and counts for all saved baseline/targeted/opposite/recap arms; exact response-category transitions; raw gains and equal-cluster point estimates; per-cell counts/accuracy; presence and count of random-control records. Saved bootstrap endpoints are read, not recomputed. This is a saved-artifact audit, not a fresh scoring/parser/model validation.\n\n")
    out.append("Companion `repair_writeup_data.json` retains exact selected summaries, all calibration curves, and SHA-256 hashes of every read artifact. Regenerate with:\n\n```bash\nPYTHONPATH=src python src/stage_r_repair_writeup.py\n```\n")
    output = BASE / "PAPER_REPAIR_SYNTHESIS.md"
    output.write_text("\n".join(out), encoding="utf-8")
    for filename in ("stage_r_repair_writeup.py", "stale_binding_natural_routing.py", "stale_binding_routing.py", "stage_r_crossmodel_routing.py", "stage_r_llama_causal_heads.py", "stale_binding_attenuation.py", "stage_l_eval.py"):
        fingerprint(ROOT / "src" / filename)
    audit = {"snapshot": "2026-09-18", "inference_rerun": False,
             "runs": [{k: v for k, v in r.items() if k != "raw"} for r in runs],
             "historical_summary_only": historical,
             "sha256": SOURCES}
    (BASE / "repair_writeup_data.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"Verified {len(runs)} canonical models; wrote {output.relative_to(ROOT)}")
    print(f"Audited {sum(len(r['raw']) for r in runs)} task results; source hashes: {len(SOURCES)}")


if __name__ == "__main__":
    build()
