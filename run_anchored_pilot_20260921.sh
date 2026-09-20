set -euo pipefail
task_root=$1
task_python=$2
task_commit=$3
task_training=$4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR="$task_root/tmp"
export CUDA_VISIBLE_DEVICES=
unset ICLR_TEST_DEVICE
mkdir -p "$TMPDIR"
cd "$task_root/anchored-pilot-code"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_root/exit-anchored-pilot.txt"' EXIT
"$task_python" -m pytest tests/test_anchored_tail_20260921.py -q > "$task_root/tests-anchored-pilot.log" 2>&1
nice -n 10 "$task_python" -u run_anchored_pilot_20260921.py --root "$task_root/run-anchored-pilot" --training-root "$task_training" --commit "$task_commit"
