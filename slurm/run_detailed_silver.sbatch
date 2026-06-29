#!/bin/bash
#SBATCH -J cip_detailed_silver    # Job name
#SBATCH -o slurm-detailed-silver-%j.out
#SBATCH -p gh                     # Grace-Hopper GPU partition
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -t 01:00:00               # 1 hour (inference only)
#SBATCH --mail-type=end,fail

# D2.2: Generate detailed-level silver labels from hierarchical model on real TACC.
# Usage: sbatch -A <alloc> slurm/run_detailed_silver.sbatch

set -euo pipefail

module load gcc cuda python3
source "${SCRATCH}/envs/cip_classifier/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export TOKENIZERS_PARALLELISM=false

MAJOR_MODEL="output/sweep/models/scibert_scivocab_uncased_lr3e-05_ep8_bs16_ls0.0_wd0.01_linear_freeze8"

echo "=== D2.2: Generate Detailed Silver Labels ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

python scripts/generate_detailed_silver.py \
    --major-model "$MAJOR_MODEL" \
    --detailed-model output/models/detailed_finetune \
    --threshold 0.80 \
    --batch-size 64

echo ""
echo "Detailed silver label generation completed: $(date)"
