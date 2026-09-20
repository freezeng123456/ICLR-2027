import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    report = dict(recorded_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        record_stage=args.stage, source_commit=args.source_commit, python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, cuda=torch.version.cuda,
        available_cpu_count=os.cpu_count(), environment={name: os.environ.get(name) for name in
        ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES"]})
    report["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
                                             "--format=csv,noheader"], text=True).strip()
    report["gpu_processes_at_recording"] = subprocess.check_output(["nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).splitlines()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
