#!/usr/bin/env bash
# Download the one missing frozen Pythia checkpoint before offline inference.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH -J n_scale_dep_1b
#SBATCH -o results/stage_n/scale_transfer/logs/prepare_1b_%j.out
#SBATCH -e results/stage_n/scale_transfer/logs/prepare_1b_%j.err

set -euo pipefail
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
module purge
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0

SECONDS=0
echo "host=$(hostname) start=$(date --iso-8601=seconds)"
"$PY" -c 'from huggingface_hub import snapshot_download; print(snapshot_download("EleutherAI/pythia-1b", allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]))'
echo "elapsed_seconds=$SECONDS end=$(date --iso-8601=seconds)"
echo DONE

