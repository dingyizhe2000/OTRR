#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/../results" && "$(basename "$SCRIPT_DIR")" == "code" ]]; then
  PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
else
  PROJECT_ROOT="$SCRIPT_DIR"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/real_data/nhanes_p43_final/supplement/semisynthetic_invariance/baselines}"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_nhanes_semisynthetic_invariance.py" \
  --output "$OUTPUT_ROOT" \
  --repetitions 10 \
  --workers 4 \
  "$@"
