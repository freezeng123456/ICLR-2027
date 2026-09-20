set -euo pipefail
task_root=$1
task_python=$2
task_commit=$3
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$task_root/tmp"
export CUDA_VISIBLE_DEVICES=0
export ICLR_TEST_DEVICE=cuda
cd "$task_root/anchor-code"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_root/exit-anchor.txt"' EXIT
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
"$task_python" -m pytest tests/test_optimized_anchored_tail_20260921.py tests/test_anchored_cuda_20260921.py tests/test_factorized_baseline_20260921.py tests/test_paper_anchor_20260921.py -q > "$task_root/tests-anchor.log" 2>&1
"$task_python" -u run_paper_anchor_20260921.py --root "$task_root/run-anchor" --training-root /data/workspace/iclr-dual-20260921/training --commit "$task_commit" --hours 2
