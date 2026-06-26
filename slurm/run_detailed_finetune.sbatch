#!/bin/bash
#SBATCH -J cip_detailed           # Job name
#SBATCH -o slurm-detailed-%j.out
#SBATCH -p gh                     # Grace-Hopper GPU partition
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -t 04:00:00               # 4 hours (8 epochs on 16k abstracts, ~418 classes)
#SBATCH --mail-type=end,fail
#
# Train SciBERT on detailed CIP fields (~315-418 classes)
# Same best hyperparams as C4 major-field winner.
#
# Usage:
#   sbatch slurm/run_detailed_finetune.sbatch
#   sbatch -A <allocation> slurm/run_detailed_finetune.sbatch

set -euo pipefail

module load gcc cuda python3
source "${SCRATCH}/envs/cip_classifier/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "=== Detailed Field Fine-Tune (SciBERT, ~418 classes) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

python scripts/run_detailed_finetune.py \
    --train-data data/generated/train.jsonl \
    --test-data data/generated/test.jsonl \
    --epochs 8 \
    --lr 3e-5 \
    --batch-size 16 \
    --seed 42

echo ""
echo "Detailed field fine-tune completed: $(date)"
