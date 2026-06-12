#!/bin/bash
#SBATCH -J cip_classify          # Job name
#SBATCH -o slurm-classify-%j.out
#SBATCH -p gh                    # Grace-Hopper GPU partition
#SBATCH -N 1                     # 1 node
#SBATCH -n 1                     # 1 task
#SBATCH -t 04:00:00              # 4 hours max (large dataset)
#SBATCH --mail-type=end,fail
#
# Classify abstracts against the CIP FAISS index.
# Requires: build_index step completed first (index files exist).
#
# Usage:
#   sbatch slurm/classify.sbatch
#   sbatch -A <allocation> slurm/classify.sbatch

set -euo pipefail

# Load modules
module load gcc cuda python3

# Activate environment
source "${SCRATCH}/envs/cip_classifier/bin/activate"

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "=== Classify Abstracts ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

python scripts/run_baseline.py \
    --config configs/default.yaml configs/vista.yaml \
    --step classify

echo ""
echo "=== Evaluate Results ==="
python scripts/evaluate.py \
    --config configs/default.yaml configs/vista.yaml

echo ""
echo "Completed: $(date)"
