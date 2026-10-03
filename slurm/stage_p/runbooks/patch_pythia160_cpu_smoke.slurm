#!/bin/bash
#SBATCH --account=cis250190
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:08:00
#SBATCH --job-name=p_patch_cpu_sm
#SBATCH --output=results/stage_p/logs/p_patch_cpu_sm_%j.out
#SBATCH --error=results/stage_p/logs/p_patch_cpu_sm_%j.err

set -euo pipefail
: "${LIMIT_PAIRS:=4}"
: "${OUT_DIR:=results/stage_p/models/pythia160/patch_cpu_smoke}"
module purge
module load conda
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
MODEL=/anvil/scratch/x-jguo7/hf_cache/hub/models--EleutherAI--pythia-160m/snapshots/b56d9bee36300031aeea723b73c4d62ac7fa71a2
export PYTHONPATH="$PROJ/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd "$PROJ"
"$PY" -u src/genattn_patch.py run \
  --model "$MODEL" --model-label pythia-160m --dtype float32 \
  --analysis-summary results/stage_p/models/pythia160/analysis/analysis.summary.json \
  --pairs results/stage_p/models/pythia160/analysis/patch_pairs.jsonl \
  --out-dir "$OUT_DIR" \
  --limit-pairs "$LIMIT_PAIRS" --n-boot 2000
