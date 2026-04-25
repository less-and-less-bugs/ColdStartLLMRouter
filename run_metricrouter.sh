#!/bin/bash
# Run multistage AgenticRouter; PYTHONPATH must be the project root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"

if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate concept 2>/dev/null || true
fi

python "$SCRIPT_DIR/src/components/agenticrouter_multistage.py" "$@"
