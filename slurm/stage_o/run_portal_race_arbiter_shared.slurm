#!/bin/bash
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:35:00
#SBATCH --requeue
#SBATCH --job-name=o_race
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/race_%x_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_o/logs/race_%x_%j.err

set -euo pipefail
ROOT=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
exec bash "$ROOT/results/stage_o/portal_race_arbiter.sh"
