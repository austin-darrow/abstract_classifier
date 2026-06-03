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

# Install SGLang first (it pulls compatible torch + CUDA versions automatically)
pip3 install "sglang[all]"

# Reinstall torchvision to match whatever torch version sglang installed
pip3 install --force-reinstall torchvision --no-deps
pip3 install torchvision

# Install the project in editable mode (httpx and other deps)
cd "$(dirname "$0")/.."
pip3 install -e .

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run pipeline:  python -m cip_classifier --help"
