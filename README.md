<h1 align="center">CICM: Controlled In-Context Memory</h1>

<p align="center">
  <b>When Context Changes: Understanding Update Failures in LLMs</b><br>
  Code, data, and results for studying <i>stale binding</i>: why language models answer from a value the context has already replaced.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.38866"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.38866-b31b1b.svg"></a>
  <a href="LICENSE"><img alt="Code: MIT" src="https://img.shields.io/badge/code-MIT-2E7D5B.svg"></a>
  <a href="data/LICENSE.md"><img alt="Data: CC BY-NC 4.0" src="https://img.shields.io/badge/data-CC%20BY--NC%204.0-7B4FA0.svg"></a>
  <img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-3776AB.svg">
</p>

<p align="center">
  <a href="#the-failure-in-one-real-dialogue">Example</a> ·
  <a href="#a-guided-tour-of-the-results">Results tour</a> ·
  <a href="#the-cicm-dataset">Dataset</a> ·
  <a href="#evaluate-your-model-on-cicm">Evaluate your model</a> ·
  <a href="REPRODUCING.md">Reproduce the paper</a> ·
  <a href="#citation">Cite</a>
</p>

<p align="center">
  <img src="figures/overview.png" width="940" alt="Overview: an old value returns; the update is retained but not selected; attention controls selection"><br>
  <sub>Schematic overview. Every number in the sections below is measured.</sub>
</p>

A user says *keto*, later changes it to *gluten-free*, and then asks what their meal plan is.
Models answer **keto**. The error is not caused by context length (equally long contexts
without revisions are solved) and not by forgetting (the current value is still linearly
readable from the hidden state). What fails is **selection**: attention at the answer position
lands on the old assignment. Re-routing that attention at inference time, with no retraining,
restores the current value.

**Why it matters.** Assistants and agents increasingly run over long contexts in which
preferences, instructions, and variable values get revised. A model that retains everything
but answers from the superseded version is not controllable, and aggregate accuracy or
multiple-choice evaluation will not show the problem. CICM makes it measurable, and the
mechanism results say where to intervene.

| At a glance | |
|---|---|
| **Dataset** | CICM: 1,200 natural preference dialogues, 3 variables × 7 values, overwrite dose 1–6, 2 × 3 identity-by-recency factorial, program-known labels |
| **Models** | Qwen2.5 3B/7B/14B/72B, Llama-3.1 8B/70B, Llama-3.2 3B, Mistral-7B, Gemma-2 2B/9B, Pythia 160M–2.8B; GPT-4o, GPT-4.1, Claude Sonnet 4.5, DeepSeek-V3 and others via API |
| **Evidence** | behavior, linear probes, attention measurements, residual steering, component patching, exhaustive head analysis, training dynamics, a test-time repair |
| **Hardware** | one H100 for everything local; API for behavior-only sweeps |

---

## The failure in one real dialogue

The hero figure is a schematic. Here is one of the 1,200 CICM dialogues as Qwen2.5-7B saw it
(38 messages; only the turns that carry controlled values are shown):

| Turn | Speaker | Message | Role in the dialogue |
|:---:|:---|:---|:---|
| 3 | user | I want to switch to **gluten free** meals. | old value · 34 turns before the question |
| 7 | user | Can you set my meal style to **mediterranean**? | current value · 30 before |
| 13 | user | I'd say **discussion** is the method that helps me most. | another variable · 24 before |
| 35 | user | I am still thinking about **gluten free** when I compare options for my meal style. | old value mentioned again · 2 before |
| 37 | user | For my meal style specifically, what is the latest value I chose? Reply with only the value. | question |
| | **model** | **gluten free** ✗ | expected: **mediterranean** |

The assistant acknowledged the update at turn 8, then answered with the value it replaced.
Each dialogue records the position of every value mention, so answers are scored by exact
matching against a fixed vocabulary, never by an LLM judge.

---

## A guided tour of the results

### 1. Old values, not length, cause the failure

<p align="center"><img src="figures/phenomenon.png" width="940" alt="Phenomenon: answer composition by number of old values; scaling; multiple choice masks the error"></p>

With no rewrites, accuracy is 100% at every context length. As old values of the queried
variable accumulate, accuracy falls from 91% to 38%, and **331 of 333 errors return an earlier
value of the same variable** (two copy another variable, none are hallucinated). Larger
models move the break point rather than remove it: Qwen2.5-72B and GPT-4o hold through 16 old
values and fall to 78% and 86% at 64. Multiple-choice evaluation hides the signature: on
ICF-Bench, 82% of free-form failures return the old preference, only 29% do when the options
are listed, at the same overall accuracy.
<sub>Sources: `results/overwrite_task/`, `results/hard_tier_and_transport/`, `results/icf_bench/`.</sub>

### 2. Two pulls on the answer: variable identity vs. recency

<p align="center"><img src="figures/identity_vs_recency.png" width="940" alt="Identity-by-recency factorial on CICM"></p>

CICM moves two competitors independently: the queried variable's old value (near or far from
the question) and another variable's recent value (far, middle, or two turns before). With the
old value nearby, **94.5% of answers are that old value**, even when another variable was
updated two turns ago. With both competitors far, accuracy is 74.5% and the two error types
tie. Identity dominates; recency only catches up when identity's pull is weak.
<sub>Source: `results/cicm/natural_factorial_otherdist_l0/`.</sub>

### 3. The update is retained; attention selects the old one

<p align="center"><img src="figures/mechanism_attention.png" width="940" alt="Probe decodability, final-layer attention share, per-layer drift, cross-model shift"></p>

On failures, a linear probe still reads the current value from the final hidden state with
probability **0.85** (Qwen2.5-7B) and **0.82** (Llama-3.1-8B), against 0.03–0.06 with shuffled
labels. The information is there. What differs is where the answer position looks: on Qwen
failures it puts **80% of its attention on old assignments** (58% when it answers correctly),
and the extra pull appears in the last third of the network, where the answer is decided.
Introducing a conflicting update shifts attention toward the old value in all six open models
tested, including the ones that still answer correctly.
<sub>Sources: `results/cicm/natural_factorial_otherdist_l1/`, `results/cicm/natural_factorial_otherdist_llama/l1/`, `results/event_trace/`.</sub>

### 4. Selection is causal, and it happens late

<p align="center"><img src="figures/causal_steering.png" width="940" alt="Steering the final hidden state toward the current value"></p>

Adding the probe direction at the answer position removes most errors as the dose grows
(Qwen 97% → 54% still wrong, Llama 86% → 6%), while an equally large random direction and the
same vector at a middle layer do nothing. Component patching points the same way: replacing
late hidden states restores 78% of the answer-score gap, old-assignment keys alone restore 49%,
early and middle states restore none. In Pythia-160M, where every head can be examined,
disabling the 12 heads that most favor old values corrects **38% of unseen failures** against
9% for random heads from the same layers.
<sub>Sources: `results/cicm/validity_l2/`, `results/patching_and_external_benchmarks/circuit/`, `results/pythia_circuit/`.</sub>

### 5. Re-routing attention repairs the answer, with no retraining

<p align="center"><img src="figures/attention_rerouting_seven_models.png" width="940" alt="Before and after attention re-routing on seven models"></p>

If the failure is selection, moving attention from the old assignment to the current one
should fix it. It does. Lowering the query's attention logits to old-assignment tokens and
raising them for the current assignment, in a small set of heads chosen on calibration
dialogues, raises held-out accuracy by **+32 to +78 points on six of seven models** (Qwen2.5-14B
is the exception at +1), with 98–100% of already-correct answers preserved. Reversing the
route hurts; random positions do nothing. Weights are never changed.
<sub>Source: `results/attention_rerouting/CROSS_MODEL_SYNTHESIS.md` and the per-model records in `results/attention_rerouting/crossmodel/`.</sub>

### 6. When the wrong answer forms (Pythia-160M)

<p align="center"><img src="figures/timing_repair.png" width="940" alt="Timing: repair at the update vs at the end of the conversation"></p>

Repairing the current value's attention key fixes 93% of failures if the question follows the
update immediately, but only 9% once the rest of the conversation precedes it. Later context
rebuilds the competition, so selection has to be maintained, not fixed once.
<sub>Source: `results/event_trace/`.</sub>

---

## The CICM dataset

`data/cicm/cicm_natural_factorial_otherdist_l0.jsonl` holds the 1,200 primary dialogues
(seed 20260723; datasheet alongside). Three variables with seven natural values each (music
genre, meal style, learning style); each variable/current-value pair occurs 57–58 times so a
classifier can be trained on recurring values and tested on new dialogues. GPT-4.1 wrote
reusable templates for openings, acknowledgments, and filler turns; the generator program
inserted every controlled value, position, and question, so no API call is needed to use the
data.

<details>
<summary><b>Row format</b> (click to expand)</summary>

Each line is one dialogue with program-known structure:

```jsonc
{
  "id": "stage_l_natfact_1_00000",
  "slot": "diet",  "slot_label": "meal style",
  "current_value": "mediterranean",          // the answer
  "stale_values": ["gluten free"],           // old values of the queried variable
  "k_overwrites": 1,                         // overwrite dose (1, 2, 3, 4, or 6)
  "messages": [ {"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ... ],
  "query": "For my meal style specifically, what is the latest value I chose ...",
  "value_mentions": [                        // every controlled value, with its position
    {"value": "gluten free", "slot": "diet", "is_current": false,
     "message_index": 3, "role": "user", "char_start": 20, "char_end": 31,
     "mention_kind": "binding_update", "distance_bin": "far"},
    ...
  ],
  "factorial_cell": {"same_slot_stale_distance_bin": "near",      // old value near / far
                     "recent_other_slot_distance_bin": "far",     // other variable far / mid / near2
                     "target_current_distance_bin": "far"},
  "competition": { "target_current": {...}, "same_slot_stale": [...], "recent_other_slot": [...] }
}
```

Answers are normalized and matched against the fixed vocabulary, giving one of four outcomes:
current value, old value of the same variable, value of another variable, other.
</details>

## Evaluate your model on CICM

```bash
pip install -r requirements.txt && export PYTHONPATH=src

# 1. score a chat model (four-way answer classification, per dose and per factorial cell)
python src/cicm_eval.py generate --data data/cicm/cicm_natural_factorial_otherdist_l0.jsonl \
    --out results/my_run/rows.jsonl --summary-out results/my_run/summary.json \
    --report-out results/my_run/REPORT.md --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-new-tokens 16

# 2. is the current value still decodable, and where does attention go on failures?
python src/cicm_mech.py harvest --rows results/my_run/rows.jsonl --out-dir results/my_run/mech \
    --model Qwen/Qwen2.5-7B-Instruct --dtype float32
python src/cicm_mech.py analyze --mech-dir results/my_run/mech --summary-out results/my_run/mech/summary.json \
    --probe-rows-out results/my_run/mech/probe_rows.jsonl --report-out results/my_run/mech/REPORT.md
```

Full map of scripts, result directories, and commands for every part of the paper:
**[REPRODUCING.md](REPRODUCING.md)**.

## What is in the repository

```
data/cicm/          CICM dialogues, datasheets, pilot variants
data/               synthetic overwrite tasks; BABILong / Entity Tracking / RULER-VT subsets
src/                every script, flat: generators, evaluation, probes, attention, steering,
                    patching, Pythia circuit, frontier panel, test-time re-routing, figures
results/<topic>/    per-topic REPORT.md, *.summary.json, figures, per-example records
figures/            the paper's figures as PDF and PNG
tests/              CPU-only tests
```

## Citation

```bibtex
@article{guo2026context,
  title   = {When Context Changes: Understanding Update Failures in LLMs},
  author  = {Guo, Junyu and Fang, Yuchen and Gu, Shangding and Spanos, Costas and Demmel, James and Lavaei, Javad},
  journal = {arXiv preprint arXiv:2609.38866},
  year    = {2026}
}
```

## License

Code: MIT ([`LICENSE`](LICENSE)). Datasets and per-example model outputs: CC BY-NC 4.0
([`data/LICENSE.md`](data/LICENSE.md)), because CICM's preference vocabulary derives from
PrefEval, which carries that license. Questions and issues: please open a GitHub issue.
