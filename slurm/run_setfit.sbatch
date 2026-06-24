#!/bin/bash
#SBATCH -J cip_setfit             # Job name
#SBATCH -o slurm-setfit-%j.out
#SBATCH -p gh                     # Grace-Hopper GPU partition
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -t 12:00:00               # 4 hours (SetFit contrastive is slow with 74 classes)
#SBATCH --mail-type=end,fail
#
# B4: SetFit contrastive fine-tuning classifier
#
# Usage:
#   sbatch slurm/run_setfit.sbatch
#   sbatch -A <allocation> slurm/run_setfit.sbatch

set -euo pipefail

module load gcc cuda python3
source "${SCRATCH}/envs/cip_classifier/bin/activate"
pip install -q "setfit>=1.1" "transformers>=4.40,<4.50" 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "=== B4: SetFit Classifier ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cip-classifier setfit \
    -c configs/default.yaml -c configs/vista.yaml -c configs/train.yaml \
    --train-data data/generated/train_with_silver.jsonl \
    --batch-size 32 --bf16 --num-iterations 20

echo ""
echo "B4 SetFit completed: $(date)"
