import argparse
import csv
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys


def command(args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--stage", choices=["smoke", "matrix"], required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(exist_ok=True, parents=True)
    code = Path(__file__).resolve().parent
    if args.stage == "smoke":
        node_text = command(["scontrol", "show", "nodes", "-o"])
        (root / "nodes.txt").write_text(node_text)
        free = 0
        for line in node_text.splitlines():
            fields = dict(part.split("=", 1) for part in shlex.split(line) if "=" in part)
            if "xhhgnormal" not in fields.get("Partitions", "").split(","):
                continue
            if any(state in fields.get("State", "") for state in ["DOWN", "DRAIN", "FAIL", "UNKNOWN"]):
                continue
            configured = dict(part.split("=", 1) for part in fields.get("CfgTRES", "").split(",") if "=" in part)
            allocated = dict(part.split("=", 1) for part in fields.get("AllocTRES", "").split(",") if "=" in part)
            free += int(configured.get("gres/gpu", 0)) - int(allocated.get("gres/gpu", 0))
        association = command(["sacctmgr", "-nP", "show", "assoc", "where", "account=acu722p2q8", "format=Account,User,Partition,GrpTRES,GrpSubmitJobs,QOS"])
        (root / "association.txt").write_text(association)
        parent = [row for row in csv.reader(io.StringIO(association), delimiter="|") if len(row) > 4 and row[1] == "" and row[2] == ""]
        assert len(parent) == 1
        tres = dict(part.split("=", 1) for part in parent[0][3].split(","))
        gpu_cap = int(tres["gres/gpu"])
        assert int(parent[0][4]) >= 86
        qos = command(["sacctmgr", "-nP", "show", "qos", "user_zenghang", "format=Name,MaxTRESPU,GrpTRES,MaxSubmitPU"])
        (root / "qos.txt").write_text(qos)
        concurrency = min(40, free, gpu_cap)
        assert concurrency > 0
        provenance = {"commit": args.commit, "code": str(code), "root": str(root), "python": sys.executable, "free_partition_gpus": free, "account_gpu_cap": gpu_cap, "concurrency": concurrency, "expected_oracle_cells": 1080, "expected_learned_cells": 1000, "expected_training_runs": 5, "authorization": "user unrestricted compute 2026-09-09 23:51 CST"}
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2))
        subprocess.run([sys.executable, "run_composition_extension.py", "--root", str(root / "checks"), "--commit", args.commit, "--check", "--device", "cpu"], check=True)
        job = command(["sbatch", "--parsable", "--time=00:10:00", "--output=" + str(root / "smoke_%j.log"), str(code / "run_extension_scnet.sh"), str(code), str(root / "smoke"), args.commit, "smoke"])
        (root / "smoke_job.txt").write_text(job + "\n")
        print(json.dumps(dict(provenance, smoke_job=job)))
    else:
        assert (root / "smoke" / "task_000" / "done").is_file()
        assert not (root / "jobs.json").exists()
        provenance = json.loads((root / "provenance.json").read_text())
        concurrency = provenance["concurrency"]
        for kind in ["oracle", "learned"]:
            subprocess.run([sys.executable, "run_composition_extension.py", "--root", str(root / kind), "--commit", args.commit, "--kind", kind, "--manifest-only"], check=True)
        training = command(["sbatch", "--parsable", f"--array=0-4%{min(5, concurrency)}", "--output=" + str(root / "train_%A_%a.log"), str(code / "run_extension_scnet.sh"), str(code), str(root / "training"), args.commit, "train"])
        oracle = command(["sbatch", "--parsable", f"--array=0-39%{concurrency}", "--output=" + str(root / "oracle_%A_%a.log"), str(code / "run_extension_scnet.sh"), str(code), str(root / "oracle"), args.commit, "oracle"])
        learned = command(["sbatch", "--parsable", f"--array=0-39%{concurrency}", f"--dependency=afterok:{training}", "--output=" + str(root / "learned_%A_%a.log"), str(code / "run_extension_scnet.sh"), str(code), str(root / "learned"), args.commit, "learned", str(root / "training")])
        jobs = {"training": training, "oracle": oracle, "learned": learned}
        (root / "jobs.json").write_text(json.dumps(jobs, indent=2))
        print(json.dumps(jobs))


if __name__ == "__main__":
    main()
