#!/bin/bash
#SBATCH -J cip_gen_targeted       # Job name
#SBATCH -o slurm-gen-targeted-%j.out
#SBATCH -e slurm-gen-targeted-%j.err
#SBATCH -p gh                     # Grace-Hopper GPU partition
#SBATCH -N 9                      # 9 nodes (DeepSeek-R1 671B needs multi-node)
#SBATCH -n 9                      # 1 task per node
#SBATCH -t 18:00:00               # 18 hours
#SBATCH --mail-type=end,fail
#
# D2.1: Generate additional synthetic abstracts for low-performing fields.
# Launches DeepSeek-R1 (671B, FP8) across 9 GH200 nodes, waits for readiness,
# then generates targeted abstracts for bottom ~15 fields.
#
# Output: data/generated/targeted/abstracts_raw.jsonl
# After completion, run: python scripts/generate_targeted.py --merge
#
# Usage:
#   sbatch -A <alloc> slurm/run_generate_targeted.sbatch
#   sbatch --export=ALL,SAMPLES_PER_CIP=50 -A <alloc> slurm/run_generate_targeted.sbatch

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
export FLASHINFER_CACHE_DIR="${SCRATCH}/.cache/flashinfer"
export TRITON_CACHE_DIR="${SCRATCH}/.cache/triton"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
export VLLM_DEEP_GEMM_WARMUP=skip
export DG_CACHE_DIR="${SCRATCH}/.cache/deep_gemm"
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export CUDA_VISIBLE_DEVICES=0

# ---------------------------------------------------------------------------
# Node roles
# ---------------------------------------------------------------------------

MASTER_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
MASTER_PORT=5000
SERVER_PORT=30000
NUM_NODES=$SLURM_NNODES
TARGET_PER_FIELD="${TARGET_PER_FIELD:-300}"

echo "============================================="
echo "D2.1: Targeted Abstract Generation"
echo "============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Nodes:       $NUM_NODES"
echo "Master:      $MASTER_NODE"
echo "API port:    $SERVER_PORT"
echo "Target/field: $TARGET_PER_FIELD"
echo "Model:       deepseek-ai/DeepSeek-R1"
echo "Precision:   FP8"
echo "Date:        $(date)"
echo "============================================="

# ---------------------------------------------------------------------------
# Launch vLLM with Ray (multi-node distributed inference)
# ---------------------------------------------------------------------------

LAUNCH_SCRIPT="${SCRATCH}/.vllm_worker_${SLURM_JOB_ID}.sh"
cat > "$LAUNCH_SCRIPT" << 'WORKER_EOF'
#!/bin/bash
HOST=$1
NODES=$2
SERVER_PORT=$3
RAY_PORT=6379

export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
export VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
export VLLM_DEEP_GEMM_WARMUP=skip
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOCAL_RANK=${OMPI_COMM_WORLD_RANK}
echo "[Rank $LOCAL_RANK] $(hostname) starting..."

if [ "$LOCAL_RANK" -eq 0 ]; then
    ray start --head --port=$RAY_PORT --num-gpus=1 --block &
    RAY_HEAD_PID=$!

    EXPECTED_GPUS=$NODES
    MAX_RAY_WAIT=120
    RAY_ELAPSED=0
    echo "[Rank 0] Waiting for $EXPECTED_GPUS GPUs in Ray cluster..."
    while [ $RAY_ELAPSED -lt $MAX_RAY_WAIT ]; do
        GPU_COUNT=$(python3 -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo "0")
        if [ "$GPU_COUNT" -ge "$EXPECTED_GPUS" ]; then
            echo "[Rank 0] Ray cluster has $GPU_COUNT GPUs. Proceeding."
            break
        fi
        sleep 5
        RAY_ELAPSED=$((RAY_ELAPSED + 5))
    done

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
    echo "[Rank $LOCAL_RANK] Joining Ray cluster at ${HOST}:${RAY_PORT}..."
    ray start --address="${HOST}:${RAY_PORT}" --num-gpus=1 --block
fi
WORKER_EOF
chmod +x "$LAUNCH_SCRIPT"

echo ""
echo ">>> Launching vLLM server across $NUM_NODES nodes..."
echo ""

mpirun -np "$NUM_NODES" --map-by ppr:1:node "$LAUNCH_SCRIPT" "$MASTER_NODE" "$NUM_NODES" "$SERVER_PORT" &
SERVER_PID=$!

# ---------------------------------------------------------------------------
# Wait for server readiness
# ---------------------------------------------------------------------------

SERVER_URL="http://${MASTER_NODE}:${SERVER_PORT}"
MAX_WAIT=1800
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
    echo "ERROR: Server not ready after ${MAX_WAIT}s. Exiting."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    rm -f "$LAUNCH_SCRIPT"
    exit 1
fi

# ---------------------------------------------------------------------------
# Run targeted generation
# ---------------------------------------------------------------------------

echo ""
echo ">>> Starting targeted abstract generation (target ${TARGET_PER_FIELD}/field)..."
echo ""

python scripts/generate_targeted.py \
    --target-per-field "$TARGET_PER_FIELD" \
    --server-url "$SERVER_URL" \
    --config configs/default.yaml configs/generate.yaml

echo ""
echo ">>> Generation complete. Merging with existing training data..."

python scripts/generate_targeted.py --merge

echo ""
echo ">>> All done: $(date)"
echo "Output:"
ls -la data/generated/targeted/
echo ""
ls -la data/generated/train_d2.jsonl

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

echo ""
echo ">>> Shutting down inference server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
ray stop 2>/dev/null || true
rm -f "$LAUNCH_SCRIPT"
echo "Done."
