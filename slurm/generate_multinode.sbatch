#!/bin/bash
#SBATCH -J deepseek_generate        # Job name
#SBATCH -o slurm-generate-%j.out    # Output log
#SBATCH -e slurm-generate-%j.err    # Error log
#SBATCH -p gh                       # Grace-Hopper GPU partition
#SBATCH -N 9                        # 9 nodes (1 GH200 each = 9 GPUs total)
#SBATCH -n 9                        # 9 tasks (1 per node)
#SBATCH -t 12:00:00                 # 12 hours
#SBATCH --mail-type=begin,end,fail
#
# Single-job pipeline: launches DeepSeek-R1 (671B) inference server across
# 9 GH200 nodes using MPI (per Vista multi-node pattern), waits for readiness,
# then runs the abstract generation client on the head node.
#
# The model is loaded in FP8 precision (~75GB per GPU across 9 GPUs).
#
# Usage:
#   sbatch slurm/launch_server.sbatch
#   sbatch -A <allocation> slurm/launch_server.sbatch
#
# Override samples per field:
#   sbatch --export=ALL,SAMPLES_PER_FIELD=50 slurm/launch_server.sbatch

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

module load gcc cuda python3

source "${SCRATCH}/envs/cip_classifier/bin/activate"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Cache large models on $SCRATCH to avoid re-downloading
export HF_HOME="${SCRATCH}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# Prevent any JIT caches from writing to $HOME (quota limited)
export FLASHINFER_CACHE_DIR="${SCRATCH}/.cache/flashinfer"
export TRITON_CACHE_DIR="${SCRATCH}/.cache/triton"

# Disable flashinfer sampler — use PyTorch-native sampling instead
export VLLM_USE_FLASHINFER_SAMPLER=0

# Avoid CUDA memory fragmentation — allows allocator to use non-contiguous segments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable DeepGEMM JIT kernels — NVCC JIT compilation fails on Vista GH200
# nodes (SM90). E8M0 is a Blackwell/SM100 feature that should not be enabled.
# These must be exported so Ray worker actors inherit them.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
export VLLM_DEEP_GEMM_WARMUP=skip

# Point JIT caches to $SCRATCH in case partial compilation is attempted
export DG_CACHE_DIR="${SCRATCH}/.cache/deep_gemm"
export CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc 2>/dev/null)) 2>/dev/null)}"

# NCCL settings for Vista InfiniBand
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# Single GPU per node
export CUDA_VISIBLE_DEVICES=0

# ---------------------------------------------------------------------------
# Determine node roles
# ---------------------------------------------------------------------------

MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
MASTER_PORT=5000
SERVER_PORT=30000
NUM_NODES=$SLURM_NNODES

echo "============================================="
echo "DeepSeek-R1 Abstract Generation Pipeline"
echo "============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Nodes:       $NUM_NODES"
echo "Master:      $MASTER_NODE"
echo "Dist port:   $MASTER_PORT"
echo "API port:    $SERVER_PORT"
echo "Model:       deepseek-ai/DeepSeek-R1"
echo "Precision:   FP8"
echo "Date:        $(date)"
echo "============================================="

# ---------------------------------------------------------------------------
# Launch vLLM with Ray for multi-node distributed inference.
# Ray head starts on rank 0, workers join from other nodes via MPI.
# ---------------------------------------------------------------------------

# Ensure Ray is available
if ! command -v ray &>/dev/null; then
    echo "ERROR: 'ray' not found. Install with: pip install 'ray[default]>=2.30'"
    exit 1
fi

# ---------------------------------------------------------------------------
# Pre-download model weights on head node to avoid Lustre flock race conditions.
# All workers share the same $SCRATCH cache, so a single download suffices.
# ---------------------------------------------------------------------------

MODEL="deepseek-ai/DeepSeek-R1"

echo ">>> Pre-downloading model weights (single node)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}', allow_patterns=['*.safetensors', '*.json', '*.py', 'tokenizer*'])
print('Download complete.')
"
echo ">>> Model cached at ${HF_HOME}/hub"

# Also clean any stale .incomplete / .lock files from prior failed runs
find "${HF_HOME}/hub" -name "*.incomplete" -delete 2>/dev/null || true
find "${HF_HOME}/hub" -name "*.lock" -delete 2>/dev/null || true

LAUNCH_SCRIPT="${SCRATCH}/.vllm_worker_${SLURM_JOB_ID}.sh"
cat > "$LAUNCH_SCRIPT" << 'WORKER_EOF'
#!/bin/bash
HOST=$1
NODES=$2
SERVER_PORT=$3
RAY_PORT=6379

# Prevent workers from trying to download (already cached on shared $SCRATCH)
export HF_HUB_OFFLINE=1

# Disable DeepGEMM inside Ray workers — must be set before vLLM imports
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
export VLLM_DEEP_GEMM_WARMUP=skip
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOCAL_RANK=${OMPI_COMM_WORLD_RANK}

echo "[Rank $LOCAL_RANK] $(hostname) starting..."

if [ "$LOCAL_RANK" -eq 0 ]; then
    # Head node: start Ray head (num-gpus=1 so vLLM can schedule a worker here)
    ray start --head --port=$RAY_PORT --num-gpus=1 --block &
    RAY_HEAD_PID=$!

    # Wait until all worker nodes have joined the Ray cluster
    EXPECTED_GPUS=$NODES
    MAX_RAY_WAIT=120
    RAY_ELAPSED=0
    echo "[Rank 0] Waiting for $EXPECTED_GPUS GPUs in Ray cluster..."
    while [ $RAY_ELAPSED -lt $MAX_RAY_WAIT ]; do
        GPU_COUNT=$(python3 -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo "0")
        if [ "$GPU_COUNT" -ge "$EXPECTED_GPUS" ]; then
            echo "[Rank 0] Ray cluster has $GPU_COUNT GPUs (expected $EXPECTED_GPUS). Proceeding."
            break
        fi
        echo "[Rank 0] Ray cluster has $GPU_COUNT/$EXPECTED_GPUS GPUs. Waiting..."
        sleep 5
        RAY_ELAPSED=$((RAY_ELAPSED + 5))
    done

    if [ "$GPU_COUNT" -lt "$EXPECTED_GPUS" ]; then
        echo "[Rank 0] WARNING: Only $GPU_COUNT/$EXPECTED_GPUS GPUs after ${MAX_RAY_WAIT}s. Launching anyway."
    fi

    echo "[Rank 0] Launching vLLM server (pipeline-parallel=$NODES)..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model deepseek-ai/DeepSeek-R1 \
        --pipeline-parallel-size "$NODES" \
        --tensor-parallel-size 1 \
        --distributed-executor-backend ray \
        --trust-remote-code \
        --gpu-memory-utilization 0.90 \
        --max-model-len 8192 \
        --max-num-batched-tokens 1024 \
        --quantization fp8 \
        --enforce-eager \
        --host 0.0.0.0 \
        --port "$SERVER_PORT"
else
    # Worker nodes: connect to Ray head
    echo "[Rank $LOCAL_RANK] Joining Ray cluster at ${HOST}:${RAY_PORT}..."
    ray start --address="${HOST}:${RAY_PORT}" --num-gpus=1 --block
fi
WORKER_EOF
chmod +x "$LAUNCH_SCRIPT"

echo ""
echo ">>> Launching vLLM server across $NUM_NODES nodes via MPI..."
echo ""

mpirun -np "$NUM_NODES" --map-by ppr:1:node "$LAUNCH_SCRIPT" "$MASTER_NODE" "$NUM_NODES" "$SERVER_PORT" &
SERVER_PID=$!

# ---------------------------------------------------------------------------
# Wait for server to become healthy
# ---------------------------------------------------------------------------

SERVER_URL="http://${MASTER_NODE}:${SERVER_PORT}"
MAX_WAIT=1800  # 15 minutes for model loading
ELAPSED=0
INTERVAL=15

echo "Waiting for server at $SERVER_URL ..."

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -sf "${SERVER_URL}/v1/models" > /dev/null 2>&1; then
        echo "Server is ready! (took ${ELAPSED}s)"
        break
    fi
    echo "  Waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: Server not ready after ${MAX_WAIT}s. Killing workers and exiting."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -f "$LAUNCH_SCRIPT"
    exit 1
fi

# ---------------------------------------------------------------------------
# Run abstract generation client
# ---------------------------------------------------------------------------

echo ""
echo ">>> Starting abstract generation..."
echo ""

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
echo ">>> Generation complete. Running train/test split..."

python -m cip_classifier split \
    --config config/default.yaml \
    --config config/generate.yaml

echo ""
echo ">>> All done: $(date)"
echo "Output:"
ls -la data/generated/

# ---------------------------------------------------------------------------
# Cleanup: kill the server processes
# ---------------------------------------------------------------------------

echo ""
echo ">>> Shutting down inference server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
ray stop 2>/dev/null || true
rm -f "$LAUNCH_SCRIPT"
echo "Done."
