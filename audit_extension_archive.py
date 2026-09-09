import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def audit(root, output):
    manifest = json.loads((root / "recovery_manifest.json").read_text())
    expected_paths = set()
    for index, item in enumerate(manifest["files"]):
        path = root / item["path"]
        assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], path
        expected_paths.add(item["path"])
        if (index + 1) % 2000 == 0:
            print(json.dumps({"verified_files": index + 1}), flush=True)
    actual_paths = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name != "recovery_manifest.json"}
    assert actual_paths == expected_paths
    rows = []
    for kind, count in [("training", 5), ("oracle", 40), ("learned", 40)]:
        with (root / f"{kind}_slurm.tsv").open() as stream:
            records = list(csv.DictReader(stream, delimiter="|"))
        assert len(records) == count
        for row in records:
            assert row["State"] == "COMPLETED" and row["ExitCode"] == "0:0"
            allocation = dict(item.split("=", 1) for item in row["AllocTRES"].split(",") if "=" in item)
            assert allocation["gres/gpu"] == "1" and allocation["cpu"] == "4"
        rows.extend(records)
    for kind in ["oracle", "learned"]:
        source = json.loads((root / kind / "manifest.json").read_text())
        assert source["commit"] == manifest["source_commit"]
        for name, checksum in source["source_hashes"].items():
            content = subprocess.check_output(["git", "show", f"{source['commit']}:{name}"])
            assert hashlib.sha256(content).hexdigest() == checksum, name
        for task in (root / kind).glob("task_*"):
            runtime = json.loads((task / "runtime.json").read_text())
            assert runtime["source_hashes"] == source["source_hashes"]
    with (root / "failed_attempt_slurm.tsv").open() as stream:
        failed_attempt = list(csv.DictReader(stream, delimiter="|"))
    failed_hours = 0
    expanded_states = Counter()
    for row in failed_attempt:
        allocation = dict(item.split("=", 1) for item in row["AllocTRES"].split(",") if "=" in item)
        failed_hours += int(allocation.get("gres/gpu", 0)) * int(row["ElapsedRaw"]) / 3600
        interval = re.search(r"_\[(\d+)-(\d+)(?:%\d+)?\]$", row["JobID"])
        expanded_states[row["State"]] += int(interval[2]) - int(interval[1]) + 1 if interval else 1
    report = {"status": "passed", "verified_files": len(expected_paths), "verified_scheduler_tasks": len(rows), "allocated_gpu_hours": sum(int(row["ElapsedRaw"]) for row in rows) / 3600, "source_commit": manifest["source_commit"],
              "earlier_storage_failure": {"scheduler_records": len(failed_attempt), "logical_array_tasks": sum(expanded_states.values()), "task_states": dict(expanded_states), "allocated_gpu_hours": failed_hours, "logs": "failed_attempt/", "scope": "Operational attempt retained separately; cancelled array ranges are expanded for task counts; none of its partial cells enter the formal scientific results"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "archive_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
