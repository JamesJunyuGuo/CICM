#!/bin/bash
# Restore the deleted Llama-3.2-3B-Instruct snapshot on the shared portal.
# The first Xet-backed attempt made no payload progress in 10 minutes. This
# isolated retry disables Xet and requests 30 minutes for the HTTP fallback.
#SBATCH -A cis250190
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=r13_l3_cache
#SBATCH -o /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_llama3b_cache_shared_%j.out
#SBATCH -e /anvil/scratch/x-jguo7/agent_eval/Contextual_management/results/stage_r/logs/r13_llama3b_cache_shared_%j.err

set -euo pipefail
module purge
module load conda
PY=/anvil/projects/x-cis250190/software/envs/verl/bin/python
: "${HF_TOKEN:?HF_TOKEN must be exported when the cache job is submitted}"
export HF_HOME=/anvil/scratch/x-jguo7/cache/huggingface_llama3b_r13
unset TRANSFORMERS_CACHE || true
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" HF_HUB_CACHE="$HF_HOME/hub"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

"$PY" -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="meta-llama/Llama-3.2-3B-Instruct", allow_patterns=["config.json", "generation_config.json", "model-*.safetensors", "model.safetensors.index.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"])'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" -c 'import json; from pathlib import Path; from huggingface_hub import snapshot_download; from transformers import AutoConfig, AutoTokenizer; name="meta-llama/Llama-3.2-3B-Instruct"; path=Path(snapshot_download(repo_id=name, local_files_only=True)); index=json.loads((path/"model.safetensors.index.json").read_text()); shards=sorted(set(index["weight_map"].values())); missing=sorted(f for f in shards if not (path/f).is_file() or (path/f).stat().st_size == 0); assert not missing, f"missing or empty model shards: {missing}"; config=AutoConfig.from_pretrained(path, local_files_only=True); tokenizer=AutoTokenizer.from_pretrained(path, local_files_only=True); assert config.model_type == "llama"; assert tokenizer.chat_template; print(f"offline_cache_check_ok path={path} shards={len(shards)}")'
