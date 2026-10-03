#!/usr/bin/env bash
# Stage L L-0b local hardness sweep duplicate on GPU/A100 portal.
#SBATCH -A cis250190-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:45:00
#SBATCH -J l_l0b_sweep_a100
#SBATCH -o logs/l_l0b_sweep_a100_%j.out
#SBATCH -e logs/l_l0b_sweep_a100_%j.err

set -euo pipefail

export OUT_SUFFIX=a100
exec bash /anvil/scratch/x-jguo7/agent_eval/Contextual_management/slurm/stage_l_l0b_sweep.slurm
