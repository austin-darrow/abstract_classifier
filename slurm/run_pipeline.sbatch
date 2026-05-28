#!/bin/bash
#SBATCH -J cip_pipeline          # Job name
#SBATCH -o slurm-pipeline-%j.out
#SBATCH -p gh                    # Grace-Hopper GPU partition
#SBATCH -N 1                     # 1 node
#SBATCH -n 1                     # 1 task
#SBATCH -t 06:00:00              # 6 hours max (full pipeline)
#SBATCH --mail-type=end,fail
#
# Run the full CIP classifier pipeline end-to-end on a single GPU node.
#
# Usage:
#   sbatch slurm/run_pipeline.sbatch
#   sbatch -A <allocation> slurm/run_pipeline.sbatch

set -euo pipefail

# Load modules
module load gcc cuda python3

# Activate environment
source "${SCRATCH}/envs/cip_classifier/bin/activate"

# Change to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "=== Full CIP Classifier Pipeline ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

python -m cip_classifier run-all \
    --config config/default.yaml \
    --config config/vista.yaml

echo ""
echo "Pipeline completed: $(date)"
