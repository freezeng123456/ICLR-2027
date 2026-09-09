import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from exp_conditioning import train as train_gp
from exp_jump import train as train_jump


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description="使用独立目录保存 PFN 训练配置、检查点与状态")
    parser.add_argument("--prior", choices=["gp", "jump"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--width", type=int, choices=[64, 128, 256], default=128)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 20 or args.batch_size < 1 or args.threads < 1:
        parser.error("steps 至少 20，batch-size 和 threads 必须为正数")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前解释器无法访问 CUDA")
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    config = {**vars(args), "output": str(output), "commit": commit, "dirty": dirty,
              "python": platform.python_version(), "torch": torch.__version__,
              "numpy": np.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.machine(),
              "hostname": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "dtype": "float32", "deterministic_algorithms": True,
              "source_sha256": {name: sha256(repo / name) for name in
                                ["train_repro.py", "exp_conditioning.py", "exp_jump.py",
                                 "identifiability.py", "prior_jump.py"]}}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "status").write_text("RUNNING\n", encoding="utf-8")
    started = time.perf_counter()
    checkpoint = output / "model.pt"
    train = train_gp if args.prior == "gp" else train_jump
    model = train(args.steps, bs=args.batch_size, ckpt=checkpoint, d_model=args.width,
                  seed=args.seed, device=args.device)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("训练结果包含非有限参数")
    summary = {"steps_completed": args.steps, "elapsed_seconds": time.perf_counter() - started,
               "checkpoint_sha256": sha256(checkpoint),
               "parameters": sum(p.numel() for p in model.parameters()), "finite_parameters": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "status").write_text("COMPLETE\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
