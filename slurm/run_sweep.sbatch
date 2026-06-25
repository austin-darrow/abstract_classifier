#!/bin/bash
#SBATCH -J cip_sweep              # Job name
#SBATCH -o slurm-sweep-%j.out
#SBATCH -p gh                     # Grace-Hopper GPU partition
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -t 24:00:00               # 24 hours (runs multiple configs sequentially)
#SBATCH --mail-type=end,fail
#
# C4: Hyperparameter sweep for SciBERT fine-tuning
#
# Usage:
#   # Full sweep (will take multiple submissions):
#   sbatch slurm/run_sweep.sbatch
#
#   # Run a specific tier:
#   sbatch slurm/run_sweep.sbatch --tier 1
#
#   # Resume (skip completed):
#   sbatch slurm/run_sweep.sbatch --resume
#
#   # Single config (for debugging):
#   sbatch slurm/run_sweep.sbatch --config-idx 0

set -euo pipefail

module load gcc cuda python3
source "${SCRATCH}/envs/cip_classifier/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "=== C4: Hyperparameter Sweep ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo ""

# Pre-download models to avoid mid-sweep download failures
export HF_HOME="${SCRATCH}/.cache/huggingface"
echo "Pre-caching models (HF_HOME=$HF_HOME)..."
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
models = [
    'allenai/scibert_scivocab_uncased',
    'microsoft/deberta-v3-base',
    'microsoft/deberta-v3-large',
    'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
]
for m in models:
    print(f'  Caching {m}...')
    AutoTokenizer.from_pretrained(m)
    AutoModelForSequenceClassification.from_pretrained(m, num_labels=74)
print('All models cached.')
" 2>&1 | tail -20
echo ""

# Pass through any extra args from sbatch command line
# e.g., sbatch slurm/run_sweep.sbatch --tier 1
python scripts/sweep_finetune.py \
    --train-data data/generated/train_with_silver.jsonl \
    --resume \
    "$@"

echo ""
echo "=== Generating analysis ==="
python scripts/analyze_sweep.py

echo ""
echo "C4 Sweep completed: $(date)"
