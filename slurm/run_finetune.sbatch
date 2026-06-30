#!/bin/bash
#SBATCH -J cip_finetune           # Job name
#SBATCH -o slurm-finetune-%j.out
#SBATCH -e slurm-finetune-%j.err
#SBATCH -p gh                     # Grace-Hopper GPU partition
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -t 04:00:00               # 4 hours
#SBATCH --mail-type=end,fail
#
# D2: Retrain two-model SciBERT with D2 data (targeted + silver labels)
# Best hyperparams from Phase C: freeze_layers=8, lr=3e-5, epochs=8
#
# Usage:
#   sbatch slurm/run_finetune.sbatch
#   sbatch -A <allocation> slurm/run_finetune.sbatch

set -euo pipefail

module load gcc cuda python3
source "${SCRATCH}/envs/cip_classifier/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "=== D2: Two-Model Fine-Tune (SciBERT + D2 data) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

TRAIN_DATA="${TRAIN_DATA:-data/generated/train_d2.jsonl}"
echo "Training data: ${TRAIN_DATA}"
wc -l "${TRAIN_DATA}"
echo ""

cip-classifier finetune \
    -c configs/default.yaml -c configs/vista.yaml -c configs/train.yaml \
    --train-data "${TRAIN_DATA}" \
    --epochs 8 \
    --lr 3e-5 \
    --freeze-layers 8

echo ""
echo ">>> Generating comparison charts..."
cip-classifier compare \
    -c configs/default.yaml -c configs/vista.yaml

echo ""
echo "D2 Fine-tune completed: $(date)"
