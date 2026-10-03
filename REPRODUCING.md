# Reproducing the results

Everything in the paper traces to a `*.summary.json` or `REPORT.md` under `results/`. This
page maps each stage of the study to the scripts that ran it, the result directory it wrote,
and the exact commands. The `slurm/` directory holds the job scripts behind every result
directory; they hardcode a cluster account, partition, project path, and Hugging Face cache,
so edit the header variables before reuse. Scripts named `*_smoke*` are the small pre-flight
runs that gate each full run.

## Environment

```bash
pip install -r requirements.txt     # Python 3.10; torch 2.6, transformers 4.51
export PYTHONPATH=src               # src/ is flat; scripts import each other by module name
python -m pytest tests -q           # 208 CPU-only tests
```

GPU steps were run on one H100 (80 GB); models up to 14B fit without sharding. Mechanistic
runs use `--dtype float32` with eager attention because bf16 kernel differences flip near-tied
answers (`results/stage_a/REPORT.md`). API sweeps read `OPENROUTER_API_KEY` or
`OPENAI_API_KEY` from the environment.

## Stages

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

## Commands

```bash
# 1. Controlled overwrite task: generate, evaluate, classify errors
python src/gen_tasks.py --out data/p0.jsonl --n-per-cell 100 --seed 0
python src/run_eval.py --data data/p0.jsonl --model Qwen/Qwen2.5-7B-Instruct \
    --out results/p0_qwen7b.jsonl --summary results/p0_qwen7b.summary.json \
    --batch-size 32 --max-new-tokens 16

# 2. CICM behavior with a local model (the paper's primary run sent the same prompts
#    through OpenRouter; see src/stage_l_api.py)
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

# 5. Test-time attention routing (the calibration/confirmation split is built inside)
python src/stale_binding_natural_routing.py --out-dir results/stage_r/routing/my_run \
    --routing-mode balanced --minimum-pool 100 --n-random-positions 16 --n-boot 2000 \
    --batch-size 12 --max-new-tokens 8
#    model-specific causal head discovery + adaptive routing (Llama, Mistral, Gemma, Qwen-3B):
#    see slurm/stage_r/run_r1{2,3,4}_*_causal_{calibration,full}_h100.slurm

# 6. Held-out test of the dose law (CPU, reads saved summaries)
python src/theory_free_energy.py --out-dir results/theory_free_energy \
    --bootstrap-reps 5000 --seed 20260810
```

## Conventions

- **Controls ship with every claim**: same-length no-overwrite controls, shuffled-label probe
  baselines, matched-norm random steering directions, layer-matched random head sets, random
  routing positions, opposite-direction routing, and exact identity gates (alpha = 0 or a
  no-op hook must reproduce the baseline bit for bit).
- **Calibration/confirmation discipline** (Stage R): head sets and margins are frozen on 240
  calibration dialogues before the 960 confirmation dialogues are opened; confirmation is
  reported with semantic-ID-clustered bootstrap intervals and per-cell results.
- **Error signatures, not accuracy**, are the primary readout: classifying *which* context
  value a model returns is what separates stale binding from length effects, forgetting, and
  cross-variable confusion.
- Each experiment writes raw per-example rows, an aggregate `*.summary.json`, and a
  `REPORT.md`; decoding is greedy with fixed seeds, and the model name and parameters are
  recorded in every summary. Large raw artifacts (`*.npz` harvests, steering rows) are not
  committed; `results/RAW_ARCHIVES.md` lists what is and is not here.

## Scope

The experiments characterize a controlled failure and its mechanism; they do not estimate how
often it occurs in deployed conversations. CICM is a controlled probe set, not a realism
benchmark. The attention signature is Qwen-specific, the head-level causal analysis is
established in Pythia-160M, and the test-time correction is evaluated on controlled, parseable
dialogue with a declared variable schema. See the paper's limitations section.
