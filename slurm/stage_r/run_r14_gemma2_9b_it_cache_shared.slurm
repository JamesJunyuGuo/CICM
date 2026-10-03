#!/bin/bash
# Gemma-2-9B-IT cache hydration is CPU/network work. A prior 7B snapshot took
# about 80 seconds; 15 minutes allows for the larger gated snapshot and checks.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --job-name=r14_g9_cache
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r14_gemma2_9b_it_cache_shared_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r14_gemma2_9b_it_cache_shared_%j.err

set -euo pipefail
module purge
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
: "${HF_TOKEN:?HF_TOKEN must be exported when the cache job is submitted}"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

"$PY" -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="google/gemma-2-9b-it", allow_patterns=["config.json", "generation_config.json", "model-*.safetensors", "model.safetensors.index.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json"])'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" -c 'import json; from pathlib import Path; from huggingface_hub import snapshot_download; from transformers import AutoConfig, AutoTokenizer; name="google/gemma-2-9b-it"; path=Path(snapshot_download(repo_id=name, local_files_only=True)); index=json.loads((path/"model.safetensors.index.json").read_text()); missing=sorted({f for f in index["weight_map"].values() if not (path/f).is_file()}); assert not missing, f"missing model shards: {missing}"; config=AutoConfig.from_pretrained(path, local_files_only=True); tokenizer=AutoTokenizer.from_pretrained(path, local_files_only=True); assert config.model_type == "gemma2"; assert tokenizer.chat_template; print(f"offline_cache_check_ok path={path} shards={len(set(index['"'"'weight_map'"'"'].values()))} model_type={config.model_type}")'
