#!/bin/bash
# One-time environment setup for Vista (TACC)
# Run this from a compute node (idev -p gh-dev -N 1 -n 1 -t 01:00:00)
#
# Usage:
#   idev -p gh-dev -N 1 -n 1 -t 01:00:00
#   bash slurm/setup_env.sh

set -euo pipefail

echo "=== CIP Classifier: Vista Environment Setup ==="

# ---------------------------------------------------------------------------
# Prevent pip from writing to $HOME (quota is tiny on Vista)
# ---------------------------------------------------------------------------
export PIP_CACHE_DIR="${SCRATCH}/.cache/pip"
export PIP_NO_WARN_SCRIPT_LOCATION=1
export PYTHONUSERBASE="${SCRATCH}/.local"
mkdir -p "$PIP_CACHE_DIR"

# Load required modules
module load gcc cuda python3
echo "Modules loaded: gcc, cuda, python3"
echo "Python: $(python3 --version) at $(which python3)"

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
echo "Venv Python: $(which python3)"
echo "Venv Pip:    $(which pip3)"

# Upgrade pip
pip3 install --upgrade pip

# Install PyTorch with CUDA support
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# Install the project in editable mode
cd "$(dirname "$0")/.."
pip3 install -e .

# Install generation dependencies (SGLang for inference server, httpx for client)
pip3 install "sglang[all]" httpx

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run pipeline:  python -m cip_classifier --help"
