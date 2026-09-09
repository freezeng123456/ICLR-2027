#!/bin/bash
set -euo pipefail
CODE_DIR="$1"
RUN_DIR="$2"
CELL_ID="${SLURM_ARRAY_TASK_ID:?}"
PYTHON=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$CODE_DIR"
PRIOR=gp
if [ "$CELL_ID" -ge 6 ]; then PRIOR=jump; fi
N_CONTEXT=8
if [ "$((CELL_ID % 6))" -ge 3 ]; then N_CONTEXT=24; fi
SEED="$((20260910 + CELL_ID % 3))"
CELL_DIR="$RUN_DIR/${PRIOR}_n${N_CONTEXT}_s${SEED}"
trap 'touch "$RUN_DIR/failed_${CELL_ID}"' ERR
"$PYTHON" -u pilot_context.py --prior "$PRIOR" --n-context "$N_CONTEXT" --seed "$SEED" --device cuda --output "$CELL_DIR"
