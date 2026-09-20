set -euo pipefail
task_root=$1
task_python=$2
task_commit=$3
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_root/exit-sensor-pipeline.txt"' EXIT
nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used --format=csv > "$task_root/hardware-sensor.csv"
bash "$task_root/sensor-code/run_sensor_study_20260921.sh" "$task_root" "$task_python" "$task_commit" prepare
bash "$task_root/sensor-code/run_sensor_study_20260921.sh" "$task_root" "$task_python" "$task_commit" run
