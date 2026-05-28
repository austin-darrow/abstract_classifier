#!/bin/bash
# One-time environment setup for Vista (TACC)
# Run this from a compute node (idev -p gh-dev -N 1 -n 1 -t 00:30:00)
#
# Usage:
#   idev -p gh-dev -N 1 -n 1 -t 00:30:00
#   bash slurm/setup_env.sh

set -euo pipefail

echo "=== CIP Classifier: Vista Environment Setup ==="

# Load required modules
module load gcc cuda python3
echo "Modules loaded: gcc, cuda, python3"

# Create virtual environment on $SCRATCH (faster I/O, no quota issues)
VENV_DIR="${SCRATCH}/envs/cip_classifier"

if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, delete it first: rm -rf $VENV_DIR"
else
    echo "Creating virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# Activate and install
source "$VENV_DIR/bin/activate"
echo "Python: $(which python3)"
echo "Pip: $(which pip3)"

# Install PyTorch with CUDA support
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Install the project in editable mode
cd "$(dirname "$0")/.."
pip3 install -e .

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run pipeline:  python -m cip_classifier --help"
