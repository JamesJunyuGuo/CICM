#!/bin/bash
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=o_b_full
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partB_full_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/partB_full_%j.err

# Work estimate: 7 gammas x 65 arms over about 802 task prompts plus
# 200x512 LM tokens per arm = roughly 8.3e7 token-forwards. A 65-70 minute
# 160M/A100 estimate plus about 25% buffer gives a 90-minute request.
set -euo pipefail
module purge
module load modtree/gpu
module load conda
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export PYTHONPATH="$ROOT/src"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT"
SECONDS=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" src/pythia_correction.py run \
  --lm-batch-size 32 \
  --lm-segments results/stage_o/partB_correction/dependencies/wikitext_segments.npy \
  --out-dir results/stage_o/partB_correction/full_a100
echo "elapsed_seconds=$SECONDS"
