#!/bin/bash
#SBATCH -J cip_zeroshot           # Job name
#SBATCH -o slurm-zeroshot-%j.out
#SBATCH -p gh                     # Grace-Hopper GPU partition
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -t 02:00:00               # 2 hours (API calls, mostly network-bound)
#SBATCH --mail-type=end,fail
#
# B6: Zero-shot LLM classification (accuracy ceiling)
# Requires a running inference server (vLLM).
#
# Usage:
#   sbatch slurm/run_zeroshot.sbatch
#   sbatch -A <allocation> slurm/run_zeroshot.sbatch
#
# NOTE: You may need to start a vLLM server on another node first,
#       or override --server-url to point to an existing endpoint.
#       Use --max-samples to cap cost (e.g. 500 for a quick ceiling estimate).

set -euo pipefail

module load gcc cuda python3
source "${SCRATCH}/envs/cip_classifier/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "=== B6: Zero-Shot LLM Classifier ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Run on a subsample first for speed/cost control
# Remove --max-samples for full evaluation
cip-classifier zeroshot -c configs/default.yaml -c configs/vista.yaml \
    --max-samples 1000 \
    --concurrency 16

echo ""
echo "B6 Zero-shot completed: $(date)"
