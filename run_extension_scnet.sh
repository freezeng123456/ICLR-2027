#!/bin/bash
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
EXTENSION_CODE=$1
EXTENSION_ROOT=$2
EXTENSION_COMMIT=$3
EXTENSION_MODE=$4
EXTENSION_TRAINING_ROOT=${5:-unused}
EXTENSION_PYTHON=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
cd "$EXTENSION_CODE"
if [ "$EXTENSION_MODE" = train ]; then
  "$EXTENSION_PYTHON" learned_sbi.py --root "$EXTENSION_ROOT" --seed "$SLURM_ARRAY_TASK_ID"
elif [ "$EXTENSION_MODE" = smoke ]; then
  "$EXTENSION_PYTHON" run_composition_extension.py --root "$EXTENSION_ROOT" --commit "$EXTENSION_COMMIT" --smoke
else
  "$EXTENSION_PYTHON" run_composition_extension.py --root "$EXTENSION_ROOT" --commit "$EXTENSION_COMMIT" --kind "$EXTENSION_MODE" --training-root "$EXTENSION_TRAINING_ROOT" --task "$SLURM_ARRAY_TASK_ID" --tasks 40
fi
