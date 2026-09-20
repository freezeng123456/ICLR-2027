set -euo pipefail

task_root=$1
task_python=$2
task_commit=$3
task_line=$4
task_training=$5
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR="$task_root/tmp"
export CUDA_VISIBLE_DEVICES=0
export ICLR_TEST_DEVICE=cuda
mkdir -p "$TMPDIR"
cd "$task_root/code"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_root/exit-$task_line.txt"' EXIT
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$task_root/hardware-$task_line.csv"
"$task_python" -m pytest tests/test_certified_composition_torch.py tests/test_random_kernel_moments_20260921.py tests/test_dual_20260921.py -q > "$task_root/tests-$task_line.log" 2>&1
"$task_python" -u run_dual_20260921.py --root "$task_root/run-$task_line" --training-root "$task_training" --line "$task_line" --commit "$task_commit" --hours 4 --phase all
