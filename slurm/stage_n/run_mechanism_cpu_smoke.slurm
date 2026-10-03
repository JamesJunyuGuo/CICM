#!/usr/bin/env bash
# CPU-only hook validation for the Pythia-160m mechanism pipeline.
# The powered path-patching result remains a GPU run.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:18:00
#SBATCH -J stage_n_mech_cpu
#SBATCH -o results/stage_n/logs/mechanism_cpu_%j.out
#SBATCH -e results/stage_n/logs/mechanism_cpu_%j.err

set -euo pipefail
PROJ=/anvil/scratch/x-jguo7/agent_eval/Contextual_management
cd "$PROJ"
module purge
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJ/src${PYTHONPATH:+:$PYTHONPATH}"
BEHAVIOR_DIR=results/stage_n/behavior_full_cpu
OUT=results/stage_n/mechanism_smoke_cpu
mkdir -p "$OUT"
SECONDS=0

"$PY" src/pythia_circuit.py harvest --pool "$BEHAVIOR_DIR/matched_pool.jsonl" --out-dir "$OUT" --limit 90
"$PY" src/pythia_circuit.py analyze --harvest-dir "$OUT" --summary "$OUT/mechanism.summary.json" --probe-rows "$OUT/probe_rows.jsonl" --n-boot 20 --n-shuffle 20 --probe-layers=-1
"$PY" src/pythia_circuit.py path-patch --pairs "$BEHAVIOR_DIR/pairs.jsonl" --out-dir "$OUT" --batch-size 8 --limit 8 --min-split 1 --n-boot 20
"$PY" src/pythia_circuit.py qkov --pool "$BEHAVIOR_DIR/matched_pool.jsonl" --cumulative-summary "$OUT/cumulative_patch.summary.json" --out-dir "$OUT" --limit 8
"$PY" src/pythia_circuit.py induction --cumulative-summary "$OUT/cumulative_patch.summary.json" --summary "$OUT/induction.summary.json" --n-sequences 4 --n-boot 20
"$PY" src/pythia_circuit.py ablate --pool "$BEHAVIOR_DIR/matched_pool.jsonl" --cumulative-summary "$OUT/cumulative_patch.summary.json" --mechanism-summary "$OUT/mechanism.summary.json" --summary "$OUT/ablation.summary.json" --max-per-class 8 --n-random 2 --n-boot 20

echo "elapsed_seconds=$SECONDS"
echo DONE
