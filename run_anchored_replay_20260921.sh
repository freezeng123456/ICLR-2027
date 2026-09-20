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
cd "$task_root/anchored-replay-code"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_root/exit-anchored-replay.txt"' EXIT
test "$(cat "$task_root/exit-explore.txt")" = 0
test "$(cat "$task_root/exit-factorization.txt")" = 0
test "$(cat "$task_root/exit-anchored-confirmation.txt")" = 0
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
task_remaining=$("$task_python" -c 'import os,sys,time; print(4-(time.time()-os.stat(sys.argv[1]).st_mtime)/3600)' "$task_root/hardware-explore.csv")
"$task_python" -m pytest tests/test_optimized_anchored_tail_20260921.py tests/test_anchored_cuda_20260921.py -q > "$task_root/tests-anchored-replay.log" 2>&1
"$task_python" -u run_anchored_replay_20260921.py --root "$task_root/run-anchored-replay" --original-root "$task_root/run-anchored-confirmation" --training-root "$task_root/training" --commit "$task_commit" --hours "$task_remaining"
