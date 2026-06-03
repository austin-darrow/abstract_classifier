#!/bin/bash
#SBATCH -J abstract_gen_1node       # Job name
#SBATCH -o slurm-generate-%j.out    # Output log
#SBATCH -e slurm-generate-%j.err    # Error log
#SBATCH -p gh                       # Grace-Hopper partition
#SBATCH -N 1                        # Single node (for distilled models)
#SBATCH -n 1                        # 1 task
#SBATCH -t 08:00:00                 # 8 hours
#SBATCH --mail-type=end,fail
#
# Single-node generation job for SMALLER models that fit on 1 GH200 (120GB):
#   - DeepSeek-R1-Distill-Qwen-32B  (fits easily)
#   - DeepSeek-R1-Distill-Llama-70B (fits at BF16, ~140GB with KV cache)
#   - DeepSeek-R1-Distill-Qwen-14B  (fits easily)
#
# For the full DeepSeek-R1 (671B, 8 nodes), use:
#   sbatch slurm/generate_multinode.sbatch
#
# This script launches a vLLM server on the single allocated node,
# waits for it to be ready, then runs generation — all within one job.
#
# Usage:
#   sbatch slurm/generate_singlenode.sbatch
#   sbatch -A <allocation> slurm/generate_singlenode.sbatch
#
# Override model or samples:
#   sbatch --export=ALL,MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,SAMPLES_PER_FIELD=50 slurm/generate_singlenode.sbatch

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

# Prevent any JIT caches from writing to $HOME (quota limited)
export FLASHINFER_CACHE_DIR="${SCRATCH}/.cache/flashinfer"
export TRITON_CACHE_DIR="${SCRATCH}/.cache/triton"

# Disable flashinfer sampler — use PyTorch-native sampling instead
# (avoids JIT compilation issues with Vista's module environment)
export VLLM_USE_FLASHINFER_SAMPLER=0

# Configurable model (default: 32B distill for single-node)
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
SERVER_PORT=30000
SERVER_URL="http://localhost:${SERVER_PORT}"

echo "============================================="
echo "Abstract Generation (Single Node)"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model:  $MODEL"
echo "Date:   $(date)"
echo "============================================="

# ---------------------------------------------------------------------------
# Launch vLLM server locally (single GPU)
# ---------------------------------------------------------------------------

echo ""
echo ">>> Launching vLLM server (single node)..."

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
# Run generation
# ---------------------------------------------------------------------------

echo ""
echo ">>> Starting abstract generation..."

EXTRA_ARGS=""
if [[ -n "${SAMPLES_PER_FIELD:-}" ]]; then
    EXTRA_ARGS="--samples $SAMPLES_PER_FIELD"
fi

python -m cip_classifier generate \
    --config config/default.yaml \
    --config config/generate.yaml \
    --server-url "$SERVER_URL" \
    $EXTRA_ARGS

echo ""
echo ">>> Running train/test split..."

python -m cip_classifier split \
    --config config/default.yaml \
    --config config/generate.yaml

echo ""
echo ">>> All done: $(date)"
ls -la data/generated/

# Cleanup
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "Server stopped."
