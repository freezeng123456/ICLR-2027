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
cd "$task_root/twist-code"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_root/exit-twist.txt"' EXIT
test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
"$task_python" -m pytest tests/test_gaussian_twisted_path_20260921.py tests/test_twisted_sensor_sampler_20260921.py -q > "$task_root/tests-twist.log" 2>&1
"$task_python" -u run_twist_pilot_20260921.py --root "$task_root/run-twist" --source-commit "$task_commit" --device cuda
