set -euo pipefail
task_root=$1
task_python=$2
task_commit=$3
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR="$task_root/tmp"
export CUDA_VISIBLE_DEVICES=0
export ICLR_TEST_DEVICE=cuda
cd "$task_root/anchored-confirmation-code"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_root/exit-anchored-confirmation.txt"' EXIT
test "$(cat "$task_root/exit-explore.txt")" = 0
test "$(cat "$task_root/exit-factorization.txt")" = 0
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
task_remaining=$("$task_python" -c 'import json,sys; print(4-sum(json.load(open(p))["seconds"] for p in sys.argv[1:])/3600)' "$task_root/run-explore/state.json" "$task_root/run-factorization/state.json")
"$task_python" -m pytest tests/test_anchored_cuda_20260921.py -q > "$task_root/tests-anchored-confirmation.log" 2>&1
"$task_python" -u run_anchored_confirmation_20260921.py --root "$task_root/run-anchored-confirmation" --training-root "$task_root/training" --commit "$task_commit" --hours-bound "$task_remaining"
