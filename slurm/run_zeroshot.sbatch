#!/bin/bash
#SBATCH -J cip_zeroshot           # Job name
#SBATCH -o slurm-zeroshot-%j.out
#SBATCH -e slurm-zeroshot-%j.err
#SBATCH -p gh                     # Grace-Hopper GPU partition
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -t 04:00:00               # 4 hours (server startup + 16k API calls)
#SBATCH --mail-type=end,fail
#
# B6: Zero-shot LLM classification (accuracy ceiling)
# Launches a vLLM server on the same node, then classifies against it.
#
# Usage:
#   sbatch slurm/run_zeroshot.sbatch
#   sbatch -A <allocation> slurm/run_zeroshot.sbatch
#   sbatch --export=ALL,MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B slurm/run_zeroshot.sbatch

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

module load gcc cuda python3
source "${SCRATCH}/envs/cip_classifier/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export HF_HOME="${SCRATCH}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_CACHE_DIR="${SCRATCH}/.cache/flashinfer"
export TRITON_CACHE_DIR="${SCRATCH}/.cache/triton"
export VLLM_USE_FLASHINFER_SAMPLER=0

# Fix transformers version for vLLM (setfit may have downgraded it)
pip install -q "transformers>=4.56" 2>/dev/null || true
pip install -q aiohttp 2>/dev/null || true

# Configurable model
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
SERVER_PORT=30000
SERVER_URL="http://localhost:${SERVER_PORT}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

echo "============================================="
echo "B6: Zero-Shot LLM Classifier"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model:  $MODEL"
echo "Max samples: $MAX_SAMPLES"
echo "Date:   $(date)"
echo "============================================="

# ---------------------------------------------------------------------------
# Launch vLLM server
# ---------------------------------------------------------------------------

echo ""
echo ">>> Launching vLLM server..."

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" &
SERVER_PID=$!

# ---------------------------------------------------------------------------
# Wait for server readiness
# ---------------------------------------------------------------------------

MAX_WAIT=1800
ELAPSED=0
INTERVAL=15

echo "Waiting for server at $SERVER_URL ..."

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -sf "${SERVER_URL}/v1/models" > /dev/null 2>&1; then
        echo "Server is ready! (took ${ELAPSED}s)"
        break
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Server not ready after ${MAX_WAIT}s. Exiting."
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# ---------------------------------------------------------------------------
# Run zero-shot classification
# ---------------------------------------------------------------------------

echo ""
echo ">>> Running zero-shot classification..."

cip-classifier zeroshot \
    -c configs/default.yaml -c configs/vista.yaml \
    --server-url "$SERVER_URL" \
    --model-name "$MODEL" \
    --max-samples "$MAX_SAMPLES" \
    --concurrency 16

echo ""
echo ">>> B6 Zero-shot completed: $(date)"

# Cleanup
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Server stopped."
