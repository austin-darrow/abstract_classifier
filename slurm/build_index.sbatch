#!/bin/bash
#SBATCH -J cip_build_index       # Job name
#SBATCH -o slurm-build-index-%j.out
#SBATCH -p gh                    # Grace-Hopper GPU partition
#SBATCH -N 1                     # 1 node
#SBATCH -n 1                     # 1 task
#SBATCH -t 02:00:00              # 2 hours max
#SBATCH --mail-type=end,fail
#
# Build FAISS index from CIP taxonomy on a single GPU node.
#
# Usage:
#   sbatch slurm/build_index.sbatch
#   sbatch -A <allocation> slurm/build_index.sbatch

set -euo pipefail

# Load modules
module load gcc cuda python3

# Activate environment
source "${SCRATCH}/envs/cip_classifier/bin/activate"

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "=== Build Index ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

python -m cip_classifier build-index \
    --config config/default.yaml \
    --config config/vista.yaml

echo ""
echo "Completed: $(date)"
