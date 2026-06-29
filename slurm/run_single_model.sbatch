#!/bin/bash
#SBATCH -J cip_single_model       # Job name
#SBATCH -o slurm-single-model-%j.out
#SBATCH -p gh                     # Grace-Hopper GPU partition
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -t 04:00:00               # 4 hours
#SBATCH --mail-type=end,fail

# D2.3: Train single unified 315-class model on enriched data + evaluate.
# Depends on: D2.1 (targeted data merged) and D2.2 (detailed silver labels)
# Usage: sbatch -A <alloc> slurm/run_single_model.sbatch

set -euo pipefail

module load gcc cuda python3
source "${SCRATCH}/envs/cip_classifier/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export TOKENIZERS_PARALLELISM=false

echo "=== D2.3: Single Unified Model (315 classes) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

# Check that enriched training data exists
TRAIN_DATA="data/generated/train_d2.jsonl"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: $TRAIN_DATA not found."
    echo "Run D2.1 (generate_targeted.py --merge) first."
    exit 1
fi

echo "Training data: $TRAIN_DATA"
wc -l "$TRAIN_DATA"
echo ""

# Train + evaluate + compare with two-model approach
python scripts/run_single_model.py \
    --train-data "$TRAIN_DATA" \
    --epochs 8 \
    --lr 3e-5 \
    --batch-size 16 \
    --freeze-layers 8 \
    --compare

echo ""
echo "Single model training + evaluation completed: $(date)"
