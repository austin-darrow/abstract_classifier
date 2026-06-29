#!/bin/bash
#SBATCH -J cip_hierarchical       # Job name
#SBATCH -o slurm-hierarchical-%j.out
#SBATCH -p gh                     # Grace-Hopper GPU partition
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -t 00:30:00               # 30 min (inference only, no training)
#SBATCH --mail-type=end,fail
#
# Hierarchical inference: combine major-field + detailed-field models
# via constrained decoding. Compares 3 strategies vs flat baseline.
#
# Requires both models to already be trained:
#   - Major: output/sweep/models/scibert_scivocab_uncased_lr3e-05_ep8_bs16_ls0.0_wd0.01_linear/
#   - Detailed: output/models/detailed_finetune/
#
# Usage:
#   sbatch slurm/run_hierarchical.sbatch

set -euo pipefail

module load gcc cuda python3
source "${SCRATCH}/envs/cip_classifier/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export TOKENIZERS_PARALLELISM=false

echo "=== Hierarchical Inference (Constrained Decoding) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

python scripts/run_hierarchical_inference.py \
    --top-k-major 5 \
    --major-alpha 2.0 \
    --batch-size 64

echo ""
echo "Hierarchical inference completed: $(date)"
