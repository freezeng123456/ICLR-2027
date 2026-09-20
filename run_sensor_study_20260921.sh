set -euo pipefail
task_root=$1
task_python=$2
task_commit=$3
task_mode=$4
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR="$task_root/tmp"
cd "$task_root/sensor-code"
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_root/exit-sensor-$task_mode.txt"' EXIT
if [ "$task_mode" = run ]; then
  test "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0
else
  export CUDA_VISIBLE_DEVICES=""
fi
"$task_python" -m pytest tests/test_nonseparable_sensor_model_20260921.py tests/test_nonseparable_sensor_sampler_20260921.py tests/test_sensor_study_20260921.py tests/test_matrix_tail_certificate_20260921.py -q > "$task_root/tests-sensor-$task_mode.log" 2>&1
"$task_python" -u run_sensor_study_20260921.py --root "$task_root/run-sensor" --mode "$task_mode" --commit "$task_commit" --hours 2
