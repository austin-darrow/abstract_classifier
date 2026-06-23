#!/bin/bash
# Cancel a running generation job (server + client are in the same job).
#
# Usage:
#   bash slurm/stop_server.sh          # cancel most recent generation job
#   bash slurm/stop_server.sh <jobid>  # cancel specific job

set -euo pipefail

if [[ -n "${1:-}" ]]; then
    JOB_ID="$1"
else
    # Find the most recent deepseek_generate or abstract_gen job
    JOB_ID=$(squeue -u "$USER" -n deepseek_generate,abstract_gen_1node -h -o "%i" | head -n1)
fi

if [[ -z "$JOB_ID" ]]; then
    echo "No running generation job found."
    exit 0
fi

echo "Cancelling job: $JOB_ID"
scancel "$JOB_ID"
echo "Done."
