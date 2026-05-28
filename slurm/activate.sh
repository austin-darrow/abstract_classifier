#!/bin/bash
# Source this file to activate the CIP Classifier environment on Vista.
# Usage: source slurm/activate.sh

module load gcc cuda python3
source "${SCRATCH}/envs/cip_classifier/bin/activate"
echo "CIP Classifier environment activated. Run: python -m cip_classifier --help"
