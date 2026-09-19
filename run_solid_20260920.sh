set -euo pipefail
CODE=$1
ROOT=$2
TRAINING=$3
COMMIT=$4
PYTHON=$5
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TMPDIR=/data/workspace/iclr-solid-20260920/tmp
cd "$CODE"
"$PYTHON" run_solid_20260920.py --root "$ROOT" --training-root "$TRAINING" --commit "$COMMIT" --mode prepare
trap 'result=$?; printf "%s\n" "$result" > "$ROOT/launcher.exit"' EXIT
"$PYTHON" -m pip freeze > "$ROOT/pip-freeze.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv > "$ROOT/gpu.csv"
"$PYTHON" -m pytest tests/test_solid_20260920.py -q > "$ROOT/tests.log"
"$PYTHON" run_solid_20260920.py --root "$ROOT" --training-root "$TRAINING" --commit "$COMMIT" --mode smoke
"$PYTHON" run_solid_20260920.py --root "$ROOT" --training-root "$TRAINING" --commit "$COMMIT" --mode diagnostics
"$PYTHON" run_solid_20260920.py --root "$ROOT" --training-root "$TRAINING" --commit "$COMMIT" --mode run --hours 6
