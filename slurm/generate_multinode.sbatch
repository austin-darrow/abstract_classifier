#!/bin/bash
#SBATCH -J deepseek_generate        # Job name
#SBATCH -o slurm-generate-%j.out    # Output log
#SBATCH -e slurm-generate-%j.err    # Error log
#SBATCH -p gh                       # Grace-Hopper GPU partition
#SBATCH -N 8                        # 8 nodes (1 H100 each = 8 GPUs total)
#SBATCH -n 8                        # 8 tasks (1 per node)
#SBATCH -t 12:00:00                 # 12 hours
#SBATCH --mail-type=begin,end,fail
#
# Single-job pipeline: launches DeepSeek-R1 (671B) inference server across
# 8 GH200 nodes using MPI (per Vista multi-node pattern), waits for readiness,
# then runs the abstract generation client on the head node.
#
# The model is loaded in FP8 precision (~84GB per GPU across 8 GPUs).
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
# Create the per-node launch script (Vista MPI pattern)
# ---------------------------------------------------------------------------

LAUNCH_SCRIPT=$(mktemp /tmp/sglang_worker_XXXX.sh)
chmod +x "$LAUNCH_SCRIPT"

cat > "$LAUNCH_SCRIPT" << 'WORKER_EOF'
#!/bin/bash
HOST=$1
NODES=$2
SERVER_PORT=$3

# MPI sets OMPI_COMM_WORLD_RANK for each process
LOCAL_RANK=${OMPI_COMM_WORLD_RANK}

echo "[Rank $LOCAL_RANK] $(hostname) starting SGLang worker..."

ARGS=(
    --model-path deepseek-ai/DeepSeek-R1
    --tp "$NODES"
    --dist-init-addr "${HOST}:5000"
    --nnodes "$NODES"
    --node-rank "$LOCAL_RANK"
    --trust-remote-code
    --mem-fraction-static 0.88
    --max-total-tokens 32768
    --quantization fp8
)

# Only rank 0 (master) exposes the HTTP server
if [ "$LOCAL_RANK" -eq 0 ]; then
    ARGS+=(--host 0.0.0.0 --port "$SERVER_PORT")
fi

python3 -m sglang.launch_server "${ARGS[@]}"
WORKER_EOF

# ---------------------------------------------------------------------------
# Launch SGLang server across all nodes via MPI
# ---------------------------------------------------------------------------

echo ""
echo ">>> Launching SGLang server across $NUM_NODES nodes via MPI..."
echo ""

# Launch in background so we can run the client on this node after server is ready
mpirun -np "$NUM_NODES" --map-by ppr:1:node "$LAUNCH_SCRIPT" "$MASTER_NODE" "$NUM_NODES" "$SERVER_PORT" &
SERVER_PID=$!

# ---------------------------------------------------------------------------
# Wait for server to become healthy
# ---------------------------------------------------------------------------

SERVER_URL="http://${MASTER_NODE}:${SERVER_PORT}"
MAX_WAIT=900  # 15 minutes for model loading
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
rm -f "$LAUNCH_SCRIPT"
echo "Done."
