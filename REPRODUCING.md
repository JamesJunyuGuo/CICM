# Reproducing the results

Every number in the paper traces to a `*.summary.json` or `REPORT.md` under `results/`. This
page maps each part of the study to the scripts that ran it, the result directory it wrote, and
the exact commands. The job scripts used on the cluster are not included; the commands below
are the ones they ran.

## Environment

```bash
pip install -r requirements.txt     # Python 3.10; torch 2.6, transformers 4.51
export PYTHONPATH=src               # src/ is flat; scripts import each other by module name
python -m pytest tests -q           # CPU-only tests
```

GPU steps were run on one H100 (80 GB); models up to 14B fit without sharding. Mechanistic
runs use `--dtype float32` with eager attention because bf16 kernel differences flip near-tied
answers (`results/overwrite_attention_extraction/REPORT.md`). API sweeps read
`OPENROUTER_API_KEY` or `OPENAI_API_KEY` from the environment.

## Map of the study

| Part of the paper | Scripts (`src/`) | Results |
|---|---|---|
| Controlled overwrite task: behavior and error signature | `gen_tasks.py`, `run_eval.py`, `analyze_error_signature.py`, `gen_parallel.py`, `run_eval_parallel.py` | `results/overwrite_task/` |
| Attention and logit-lens extraction on that task | `extract_overwrite_attention.py`, `analyze_write_attention.py`, `analyze_logit_lens.py`, `build_stable_labels.py`, `check_extraction_gate.py` | `results/overwrite_attention_extraction/` |
| Biasing attention toward writes (first causal test) | `intervene_attention_bias.py` | `results/overwrite_attention_bias/` |
| Head-sliced low-rank adapters and their transfer | `build_adapter_data.py`, `train_adapters.py`, `adapter_eval.py`, `head_adapters.py`, `head_temperature_arm.py`, `build_transfer_data.py`, `eval_transfer.py`, `gen_natural.py` | `results/head_adapters/`, `results/adapter_transfer/` |
| Llama replication, retrieval-head control, cross-scale API sweep | `openrouter_sweep.py`, `summarize_openrouter_sweep.py`, `retrieval_heads.py`, `summarize_natural_transfer.py` | `results/llama_and_scale/` |
| Hard-tier break points; transport-share law | `hard_tier_sweep.py`, `adapter_p_last_gain.py`, `transport_share_law.py` | `results/hard_tier_and_transport/` |
| Component patching; BABILong / Entity Tracking / RULER-VT | `component_patching.py`, `external_benchmarks.py`, `patching_metrics.py`, `patching_report.py` | `results/patching_and_external_benchmarks/` |
| ICF-Bench: error signatures, format masking, mechanism probes | `icf_generate.py`, `icf_prepare.py`, `icf_k1.py` (scoring), `icf_k1d.py` (free-form preference task), `icf_matchers.py`, `icf_validate.py`, `icf_mech*.py`, `icf_k2k3b.py` (probe scores) | `results/icf_bench/` |
| **CICM**: generation, behavior, probes, attention, steering, geometry | `gen_cicm.py`, `cicm_eval.py`, `cicm_api_eval.py`, `cicm_mech.py`, `cicm_geometry.py`, `cicm_stats.py`, `cicm_figures.py` | `data/cicm/`, `results/cicm/` |
| Pythia-160M head-level circuit and training dynamics | `pythia_gen.py`, `pythia_eval.py`, `pythia_circuit.py`, `pythia_phase1b.py`, `pythia_dynamics.py`, `pythia_figs.py`, `attention_sink.py` | `results/pythia_circuit/` |
| Cross-scale and cross-family transfer of the small-model mechanism | `pythia_crossscale.py`, `pythia_correction.py`, `pythia_scale_transfer.py`, `qwen_small_circuit.py` | `results/pythia_crossscale/` |
| Event-aligned trace: when the wrong answer forms | `genattn_gen.py`, `genattn_capture.py`, `genattn_patch.py`, `genattn_analyze.py`, `genattn_figs.py` | `results/event_trace/` |
| Frontier API panel under matched interference | `frontier_gen*.py`, `frontier_run.py`, `frontier_score*.py` | `results/frontier_panel/` |
| Test-time attention re-routing, causal head discovery, cross-model confirmation | `stale_binding_routing.py`, `stale_binding_natural_routing.py`, `causal_head_routing.py`, `crossmodel_routing.py`, `stale_binding_correction.py`, `stale_binding_attenuation.py`, `rerouting_synthesis.py`, `rerouting_figures.py` | `results/attention_rerouting/` |
| Held-out test of the −log k dose law | `theory_free_energy.py` | `results/dose_law/` |
| Figures from saved summaries | `figs.py`, `figure_style.py`, `cicm_figures.py`, `pythia_figs.py`, `genattn_figs.py`, `rerouting_figures.py` | `figures/`, `results/figures/` |

## Commands

```bash
# 1. Controlled overwrite task: generate, evaluate, classify errors
python src/gen_tasks.py --out data/p0.jsonl --n-per-cell 100 --seed 0
python src/run_eval.py --data data/p0.jsonl --model Qwen/Qwen2.5-7B-Instruct \
    --out results/overwrite_task/p0_qwen7b.jsonl \
    --summary results/overwrite_task/p0_qwen7b.summary.json --batch-size 32 --max-new-tokens 16

# 2. CICM behavior with a local model (the paper's primary run sent the same prompts
#    through OpenRouter; see src/cicm_api_eval.py)
python src/cicm_eval.py generate \
    --data data/cicm/cicm_natural_factorial_otherdist_l0.jsonl \
    --out  results/cicm/my_run/rows.jsonl \
    --summary-out results/cicm/my_run/summary.json --report-out results/cicm/my_run/REPORT.md \
    --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --max-new-tokens 16

# 3. Probe + attention harvest on the scored rows (fp32 eager attention), then analysis
python src/cicm_mech.py harvest --rows results/cicm/my_run/rows.jsonl \
    --out-dir results/cicm/my_run/mech --model Qwen/Qwen2.5-7B-Instruct --dtype float32
python src/cicm_mech.py analyze --mech-dir results/cicm/my_run/mech \
    --summary-out results/cicm/my_run/mech/l1_summary.json \
    --probe-rows-out results/cicm/my_run/mech/l1_probe_rows.jsonl \
    --report-out results/cicm/my_run/mech/REPORT.md --n-boot 2000 --n-shuffle 2000

# 4. Residual steering with matched random control and alpha = 0 identity gate
python src/cicm_mech.py intervene --mech-dir results/cicm/my_run/mech \
    --l1-summary results/cicm/my_run/mech/l1_summary.json \
    --out-dir results/cicm/my_run/steer --report-out results/cicm/my_run/steer/REPORT.md \
    --model Qwen/Qwen2.5-7B-Instruct --dtype float32 --alpha-list 1,2,4,8,16 \
    --max-new-tokens 16 --n-boot 2000 --seed 20260722

# 5. Test-time attention re-routing on Qwen2.5-7B (calibration/confirmation split built inside)
python src/stale_binding_natural_routing.py --out-dir results/attention_rerouting/routing/my_run \
    --routing-mode balanced --minimum-pool 100 --n-random-positions 16 --n-boot 2000 \
    --batch-size 12 --max-new-tokens 8
#    model-specific causal head discovery + adaptive routing (Llama, Mistral, Gemma, Qwen-3B):
#    src/causal_head_routing.py, calibration first, then --confirmation-only with the frozen summary

# 6. Held-out test of the dose law (CPU, reads saved summaries)
python src/theory_free_energy.py --out-dir results/dose_law --bootstrap-reps 5000 --seed 20260810
```

## Conventions

- **Controls ship with every claim**: same-length no-overwrite controls, shuffled-label probe
  baselines, matched-norm random steering directions, layer-matched random head sets, random
  routing positions, opposite-direction routing, and exact identity gates (alpha = 0 or a
  no-op hook must reproduce the baseline bit for bit).
- **Calibration/confirmation discipline** (re-routing): head sets and margins are frozen on
  240 calibration dialogues before the 960 confirmation dialogues are opened; confirmation is
  reported with semantic-ID-clustered bootstrap intervals and per-cell results.
- **Error signatures, not accuracy**, are the primary readout: classifying *which* context
  value a model returns is what separates stale binding from length effects, forgetting, and
  cross-variable confusion.
- Each experiment writes raw per-example rows, an aggregate `*.summary.json`, and a
  `REPORT.md`; decoding is greedy with fixed seeds, and the model name and parameters are
  recorded in every summary. Large raw artifacts (`*.npz` attention harvests, steering rows)
  are not included; every script that produces them is.
- Result directories keep the sub-run names of the original runs (`full_h100` / `full_a100`
  are independent replications on two GPU types; `smoke_*` and `calibration_*` are the
  pre-flight and calibration runs that each held-out run depends on).

## Scope

The experiments characterize a controlled failure and its mechanism; they do not estimate how
often it occurs in deployed conversations. CICM is a controlled probe set, not a realism
benchmark. The attention signature is Qwen-specific, the head-level causal analysis is
established in Pythia-160M, and the test-time correction is evaluated on controlled, parseable
dialogue with a declared variable schema. See the paper's limitations section.
