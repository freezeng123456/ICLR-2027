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
cd "$task_root/diagnostic-code-v2"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_root/exit-factorization.txt"' EXIT
test "$(cat "$task_root/exit-explore.txt")" = 0
test -f "$task_root/run-explore/confirmation/done"
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
task_remaining=$("$task_python" -c 'import json,sys; print(4-json.load(open(sys.argv[1]))["seconds"]/3600)' "$task_root/run-explore/state.json")
"$task_python" -m pytest tests/test_factorized_baseline_20260921.py -q > "$task_root/tests-factorization.log" 2>&1
"$task_python" -u run_factorization_diagnostic_20260921.py --root "$task_root/run-factorization" --training-root "$task_root/training" --commit "$task_commit" --hours "$task_remaining"
