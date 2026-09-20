set -euo pipefail
task_root=$1
task_python=$2
task_commit=$3
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$task_root/tmp"
export CUDA_VISIBLE_DEVICES=0
cd "$task_root/matched-code"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_root/exit-matched.txt"' EXIT
"$task_python" -u run_matched_anchor_20260921.py --base "$task_root/run-anchor" --root "$task_root/run-matched" --commit "$task_commit"
