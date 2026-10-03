# CICM: Controlled In-Context Memory

Code, data, and saved results for the paper **"When Do LLMs Use Updated Information?"**
(under double-blind review).

When a value is revised inside a language model's context ("meal style = gluten free … meal
style = mediterranean … *what is my meal style?*"), models often answer with the **old value
of the same variable**. We call this failure **stale binding**. This repository contains:

- **CICM**, a probe set of 1,200 natural preference dialogues in which every update, every
  competing value, and the correct answer are known to the program, so answers are scored by
  exact matching rather than by an LLM judge;
- the controlled overwrite tasks, the external-benchmark evaluations (BABILong, Entity
  Tracking, RULER-VT, ICF-Bench), and a frontier-model API panel;
- the mechanism pipeline: linear probes, attention measurements, residual steering, component
  patching, an exhaustive head analysis in Pythia-160M, and a training-dynamics trace;
- a training-free, test-time **attention-routing correction** with model-specific head
  discovery, evaluated on held-out dialogues with random-position, random-head, and
  opposite-direction controls;
- every `REPORT.md`, aggregate summary, and figure behind the paper, plus the exact job
  scripts that produced them.

## Key findings

| | Finding | Where |
|---|---|---|
| **Phenomenon** | Same-length contexts *without* revisions are solved (100%); accuracy falls from 91% to 38% as old values accumulate. Of 333 errors, 331 return an earlier value of the queried variable, 2 copy another variable, 0 are hallucinated. | `results/p0_qwen7b.summary.json`, `results/stage_e/qwen_p0_errors.summary.json` |
| **Scale** | Qwen2.5-72B and GPT-4o stay near ceiling through 16 old values but fall to 78% and 86% with 64; every error is an old value. Eight non-reasoning frontier chat models all score lower under overwrites than under a frequency/length-matched restatement control (gaps 8 to 50 points). | `results/stage_f/hard_tier/`, `results/stage_e/openrouter/`, `results/stage_q/REPORT.md` |
| **Identity vs. recency (CICM)** | With the old value near the question, 94.5% of answers are that old value even when another variable was updated two turns ago. With both competitors far, accuracy is 74.5% and the two error types tie (16.0% vs 16.5%). | `results/stage_l/natural_factorial_otherdist_l0/` |
| **Format masking** | On ICF-Bench, 82.1% of free-form failures return the old preference; in multiple-choice format only 29.4% do, at the same overall accuracy. | `results/stage_k/` |
| **Retained, not selected** | A linear probe reads the current value from the final hidden state on failures with mean probability 0.85 (Qwen2.5-7B) and 0.82 (Llama-3.1-8B), vs 0.03–0.06 with shuffled labels. On Qwen failures the answer position puts 80% of its attention on old assignments (58% when correct); Llama shows no such attention signature. | `results/stage_l/natural_factorial_otherdist_l1/`, `.../otherdist_llama/l1/` |
| **Causality** | Adding the probe direction at the final layer removes 39.5 (Qwen) and 69.5 (Llama) points more error than a matched random direction; the same vector mid-network does nothing. Patching late hidden states restores 78% of the answer-score gap; old-assignment keys alone restore 49%. | `results/stage_l/validity_l2/`, `results/stage_g/circuit/` |
| **Exhaustive small model** | In Pythia-160M, 12 of 144 heads recover 81% of the all-head patching effect; disabling the 12 heads that most favor old values corrects 38.4% of unseen failures vs 8.7% for layer-matched random heads. Copying, binding, and old-value errors emerge at the same training step (~1k). | `results/stage_n/`, `results/stage_o/` |
| **Timing** | Repairing the current value's attention key fixes 93% of failures if the question follows the update immediately, but only 9% once the rest of the conversation precedes it. | `results/stage_p/` |
| **Test-time correction** | Re-routing attention from old to current assignments in a calibrated head set raises held-out CICM retrieval on Qwen2.5-7B from 37.6% to 68.9% (+31.2, 95% CI [27.9, 35.8]) with 100% preservation of correct answers; reversing the route costs 16 points, random positions change nothing. With model-specific causal head discovery, held-out gains are +45 to +78 points on Qwen2.5-3B, Llama-3.2-3B, Llama-3.1-8B, Mistral-7B, and Gemma-2-9B (preservation 98–100%), but only +1.2 on Qwen2.5-14B. Mistral's run fails one preregistered gate (too few baseline-correct dialogues to bound preservation) and Gemma's identity check was excluded from its gate; the per-model records state this. | `results/stage_r/PAPER_REPAIR_SYNTHESIS.md`, `results/stage_r/crossmodel/` |
| **Dose law** | A one-layer softmax account predicts current-vs-old log-odds falling as −log k. Fitting only the intercept at low k predicts held-out counts in five controlled settings; it fails for Llama-3.1-70B and for CICM dialogues, in opposite directions. | `results/theory_free_energy/` |

Every number in the paper traces to a `*.summary.json` or `REPORT.md` under `results/`;
the figure scripts read those files and never recompute.

## Repository layout

```
src/            all scripts, flat (import each other by module name; run with PYTHONPATH=src)
tests/          pytest suite (tests/conftest.py puts src/ on sys.path)
data/           generated datasets, fixed seeds (CICM in data/stage_l/)
results/        per-stage REPORT.md, aggregate *.summary.json, figures, small per-row records
slurm/          the job scripts that produced every result (Purdue Anvil; edit account/paths)
paper_assets/   HTML sources and vector PDFs of the paper figures, LaTeX result tables
tools/          html2pdf.py: renders an HTML figure to a vector PDF with headless Firefox
```

Scripts are named by experiment stage. The stages, in the order the paper uses them:

| Stage | Question | Main scripts | Results |
|---|---|---|---|
| P0 / A / B | Controlled overwrite task; attention and logit-lens extraction; biasing attention toward writes | `gen_tasks.py`, `run_eval.py`, `analyze_p0_errors.py`, `extract_stage_a.py`, `analyze_a1_attention.py`, `analyze_a2_logitlens.py`, `intervene_stage_b.py` | `results/p0_qwen7b.*`, `results/stage_a/`, `results/stage_b/` |
| C / D | Head-sliced low-rank adapters (0.008% of parameters) and their transfer | `build_stage_c_data.py`, `train_stage_c.py`, `stage_c_eval.py`, `eval_stage_d.py` | `results/stage_c/`, `results/stage_d/` |
| E / F | Llama replication, retrieval-head control, cross-scale API sweep, hard-tier break points | `openrouter_sweep.py`, `hard_tier_sweep.py`, `retrieval_heads.py`, `gen_natural.py`, `stage_f_*.py` | `results/stage_e/`, `results/stage_f/` |
| G | Component patching; BABILong / Entity Tracking / RULER-VT | `stage_g_circuit.py`, `stage_g_external.py` | `results/stage_g/`, `data/stage_g/` |
| K | ICF-Bench: error signatures, format masking, mechanism probes | `icf_*.py` | `results/stage_k/` |
| **L** | **CICM**: generation, behavior, probes, attention, steering, geometry | `gen_stage_l_natural.py`, `stage_l_eval.py`, `stage_l_api.py`, `stage_l_mech.py`, `stage_l_geometry.py`, `stage_l_stats.py` | `data/stage_l/`, `results/stage_l/` |
| N / O | Pythia-160M head-level circuit, training dynamics, cross-scale and cross-family transfer | `pythia_gen.py`, `pythia_eval.py`, `pythia_circuit.py`, `pythia_dynamics.py`, `pythia_crossscale.py`, `pythia_correction.py`, `qwen_small_circuit.py` | `results/stage_n/`, `results/stage_o/` |
| P | Event-aligned trace: when the wrong answer forms | `genattn_gen.py`, `genattn_capture.py`, `genattn_patch.py`, `genattn_analyze.py` | `results/stage_p/` |
| Q | Frontier API panel under matched interference | `frontier_gen*.py`, `frontier_run.py`, `frontier_score*.py` | `results/stage_q/` |
| R | Test-time attention routing, causal head discovery, cross-model confirmation | `stale_binding_routing.py`, `stale_binding_natural_routing.py`, `stage_r_llama_causal_heads.py`, `stage_r_crossmodel_routing.py` | `results/stage_r/` |
| Theory | Held-out test of the −log k dose law | `theory_free_energy.py` | `results/theory_free_energy/` |
| Figures | Paper figures and tables from saved summaries | `paper_*.py`, `stage_l_paper_full_assets.py`, `figs.py`, `tools/html2pdf.py` | `paper_assets/`, `results/figures/` |

## The CICM dataset

Primary file: `data/stage_l/cicm_natural_factorial_otherdist_l0.jsonl` (1,200 dialogues,
seed 20260723; datasheet alongside it). Each dialogue is a multi-turn preference conversation
with:

- **3 variables × 7 values**: music genre {jazz, rock, classical, electronic, folk, hiphop,
  blues}; meal style {vegan, vegetarian, keto, mediterranean, paleo, pescatarian, gluten free};
  learning style {visual, hands on, lecture, reading, discussion, self paced, tutoring}. Every
  variable/current-value pair occurs 57–58 times, so a classifier can be trained on recurring
  values and tested on new dialogues.
- **Overwrite dose** k ∈ {1, 2, 3, 4, 6} old values of the queried variable (240 rows each).
- **A 2 × 3 competition factorial** (200 rows per cell): the nearest old value of the queried
  variable is *near* or *far* from the question, crossed with the nearest value of *another*
  variable being *far*, *middle*, or *two turns* before the question. The current value is
  always far.
- **Program-known labels**: every value mention is recorded with role, message index, and
  character span; the final question names the queried variable explicitly. Answers are
  normalized and matched against the fixed vocabulary, giving one of four outcomes: current
  value, old value of the same variable, value of another variable, other.

GPT-4.1 wrote reusable templates for openings, acknowledgments, and filler turns; the generator
program inserted every controlled value, position, and question. Template packs and all rows
are included, so no API call is needed to use the data. Earlier CICM variants used for
pilots and hardness calibration (`cicm.jsonl`, `l0b_*.jsonl`) are kept with their datasheets.

## Quickstart

```bash
pip install -r requirements.txt          # Python 3.10; torch 2.6, transformers 4.51
export PYTHONPATH=src
python -m pytest tests -q                # CPU-only unit tests

# 1. Controlled overwrite task: generate, evaluate, classify errors
python src/gen_tasks.py --out data/p0.jsonl --n-per-cell 100 --seed 0
python src/run_eval.py --data data/p0.jsonl --model Qwen/Qwen2.5-7B-Instruct \
    --out results/p0_qwen7b.jsonl --summary results/p0_qwen7b.summary.json \
    --batch-size 32 --max-new-tokens 16

# 2. CICM behavior (local model; the paper's primary run used the same prompts via OpenRouter,
#    see src/stage_l_api.py)
python src/stage_l_eval.py generate \
    --data data/stage_l/cicm_natural_factorial_otherdist_l0.jsonl \
    --out  results/stage_l/my_run/rows.jsonl \
    --summary-out results/stage_l/my_run/summary.json --report-out results/stage_l/my_run/REPORT.md \
    --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-new-tokens 16

# 3. Probe + attention harvest on the scored rows (fp32 eager attention), then analysis
python src/stage_l_mech.py harvest --rows results/stage_l/my_run/rows.jsonl \
    --out-dir results/stage_l/my_run/mech --model Qwen/Qwen2.5-7B-Instruct --dtype float32
python src/stage_l_mech.py analyze --mech-dir results/stage_l/my_run/mech \
    --summary-out results/stage_l/my_run/mech/l1_summary.json \
    --probe-rows-out results/stage_l/my_run/mech/l1_probe_rows.jsonl \
    --report-out results/stage_l/my_run/mech/REPORT.md --n-boot 2000 --n-shuffle 2000

# 4. Residual steering with matched random control and alpha = 0 identity gate
python src/stage_l_mech.py intervene --mech-dir results/stage_l/my_run/mech \
    --l1-summary results/stage_l/my_run/mech/l1_summary.json \
    --out-dir results/stage_l/my_run/steer --report-out results/stage_l/my_run/steer/REPORT.md \
    --model Qwen/Qwen2.5-7B-Instruct --dtype float32 --alpha-list 1,2,4,8,16 \
    --max-new-tokens 16 --n-boot 2000 --seed 20260722

# 5. Test-time attention routing (calibration/confirmation split is built inside)
python src/stale_binding_natural_routing.py --out-dir results/stage_r/routing/my_run \
    --routing-mode balanced --minimum-pool 100 --n-random-positions 16 --n-boot 2000 \
    --batch-size 12 --max-new-tokens 8

# 6. Held-out test of the dose law (CPU, reads saved summaries)
python src/theory_free_energy.py --out-dir results/theory_free_energy \
    --bootstrap-reps 5000 --seed 20260810
```

GPU steps were run on one H100 (80 GB); ≤14B models fit without sharding. Mechanistic runs
use `--dtype float32` with eager attention because bf16 kernel differences flip near-tied
answers (`results/stage_a/REPORT.md`). API sweeps read `OPENROUTER_API_KEY` or
`OPENAI_API_KEY` from the environment.

The `slurm/` scripts are the exact submissions behind each result directory. They hardcode a
cluster account, partition, project path, and Hugging Face cache; edit the header variables
before reuse. Scripts named `*_smoke*` are the small pre-flight runs that gate each full run.

## Conventions worth knowing

- **Controls ship with every claim**: same-length no-overwrite controls, shuffled-label probe
  baselines, matched-norm random steering directions, layer-matched random head sets, random
  routing positions, opposite-direction routing, and exact identity gates (alpha = 0 or a no-op
  hook must reproduce the baseline bit for bit).
- **Calibration/confirmation discipline** (Stage R): head sets and margins are frozen on 240
  calibration dialogues before the 960 confirmation dialogues are opened; confirmation is
  reported with semantic-ID-clustered bootstrap intervals and per-cell results.
- **Error signatures, not accuracy**, are the primary readout: classifying *which* context value
  a model returns is what separates stale binding from length effects, forgetting, and
  cross-variable confusion.
- Each experiment writes raw per-example rows, an aggregate `*.summary.json`, and a
  `REPORT.md`; decoding is greedy with fixed seeds, and the model name and parameters are
  recorded in every summary. Larger raw artifacts (`*.npz` harvests, steering rows) are not
  committed; `results/RAW_ARCHIVES.md` lists what is and is not here.

## Scope

The experiments characterize a controlled failure and its mechanism; they do not estimate how
often it occurs in deployed conversations. CICM is a controlled probe set, not a realism
benchmark. The attention signature is Qwen-specific, the head-level causal analysis is
established in Pythia-160M, and the test-time correction is evaluated on controlled, parseable
dialogue with a declared variable schema. See the paper's limitations section.

## License

Code is released under the MIT License (`LICENSE`). Datasets and per-example model outputs are
released under CC BY-NC 4.0 (`data/LICENSE.md`) because CICM's preference vocabulary derives
from PrefEval, which carries that license. The Source Sans 3 fonts in `paper_assets/fonts/`
are under the SIL Open Font License.

## Citation

The paper is under double-blind review; a BibTeX entry will be added here once it is public.
Until then, please cite this repository.
