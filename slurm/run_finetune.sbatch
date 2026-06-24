#!/bin/bash
#SBATCH -J cip_finetune           # Job name
#SBATCH -o slurm-finetune-%j.out
#SBATCH -p gh                     # Grace-Hopper GPU partition
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -t 04:00:00               # 4 hours (3 epochs on 16k abstracts)
#SBATCH --mail-type=end,fail
#
# B5: Full encoder fine-tune (SciBERT) with classification head
#
# Usage:
#   sbatch slurm/run_finetune.sbatch
#   sbatch -A <allocation> slurm/run_finetune.sbatch

set -euo pipefail

module load gcc cuda python3
source "${SCRATCH}/envs/cip_classifier/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "=== B5: Full Fine-Tune (SciBERT) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

cip-classifier finetune \
    -c configs/default.yaml -c configs/vista.yaml -c configs/train.yaml \
    --train-data data/generated/train_with_silver.jsonl \
    --epochs 5

echo ""
echo "B5 Fine-tune completed: $(date)"
