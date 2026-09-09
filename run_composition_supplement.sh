#!/bin/bash
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
COMPOSITION_CODE=$1
COMPOSITION_ROOT=$2
COMPOSITION_COMMIT=$3
COMPOSITION_MODE=${4:-matrix}
COMPOSITION_PYTHON=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
mkdir -p "$COMPOSITION_ROOT"
cd "$COMPOSITION_CODE"
if [ "$COMPOSITION_MODE" = smoke ]; then
  "$COMPOSITION_PYTHON" run_composition_supplement.py --root "$COMPOSITION_ROOT" --commit "$COMPOSITION_COMMIT" --smoke
else
  "$COMPOSITION_PYTHON" run_composition_supplement.py --root "$COMPOSITION_ROOT" --commit "$COMPOSITION_COMMIT" --task "$SLURM_ARRAY_TASK_ID" --tasks 12
fi
