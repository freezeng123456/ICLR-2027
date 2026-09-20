set -euo pipefail
code="$1"
root="$2"
commit="$3"
python="$4"
training="$5"
cd "$code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$root/tmp"
export MPLCONFIGDIR="$root/tmp/mpl"
export XDG_CACHE_HOME="$root/tmp/cache"
export ICLR_TRAINING_ROOT="$training"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
trap 'status=$?; printf "%s\n" "$status" > "$root/exit-pipeline.txt"' EXIT
ICLR_TEST_DEVICE=cuda "$python" -m pytest -q tests/test_learned_factor_bridge_20260921.py tests/test_learned_factor_reference_20260921.py tests/test_learned_factor_study_20260921.py > "$root/tests-cuda.log" 2>&1
"$python" run_learned_factor_study_20260921.py --root "$root/run" --training-root "$training" --source-commit "$commit" --phase development --device cuda > "$root/development.log" 2>&1
if "$python" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["expand_confirmation"])' "$root/run/selection.json"; then
    "$python" run_learned_factor_study_20260921.py --root "$root/run" --training-root "$training" --source-commit "$commit" --phase confirmation --device cuda > "$root/confirmation.log" 2>&1
fi
