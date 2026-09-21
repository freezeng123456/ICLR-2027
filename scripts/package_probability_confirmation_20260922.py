import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


SOURCE_COMMIT = "9b03772540d0e9f5710de1f26c3f96a7b8408215"
METHODS = {"growth", "uniform", "variance", "paired", "betting"}
PROBLEMS = {
    f"rare_p{p:g}_b{b:g}": {
        "name": f"rare_p{p:g}_b{b:g}",
        "positive_beta": [8.0, 2.0],
        "negative_beta": [p, 1 - p],
        "positive_scale": 1.0,
        "negative_scale": b / p,
    }
    for p in [0.002, 0.02]
    for b in [0.2, 0.4, 0.8, 1.2]
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_resources(resource_text):
    resources = list(csv.DictReader(io.StringIO(resource_text), delimiter="|"))
    if len(resources) != 3 or {row["JobID"] for row in resources} != {"24232628", "24232628.batch", "24232628.extern"}:
        raise ValueError("Slurm accounting must include the job and both expected steps")
    primary = next(row for row in resources if row["JobID"] == "24232628")
    if primary["State"] != "COMPLETED" or primary["ExitCode"] != "0:0" or primary["AllocCPUS"] != "4":
        raise ValueError("Slurm does not confirm a successful four-CPU job")
    if primary["Timelimit"] != "01:30:00" or primary["ReqMem"] != "8G":
        raise ValueError("Slurm resource request differs from the authorized budget")
    if any(row["State"] != "COMPLETED" or row["ExitCode"] != "0:0" for row in resources):
        raise ValueError("A Slurm job step did not exit successfully")
    return primary


def validate_directory(root, smoke=False):
    provenance = json.loads((root / "provenance.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    summaries = json.loads((root / "summaries.json").read_text())
    expected = 5 if smoke else 20000
    problems = {"rare_p0.002_b0.2"} if smoke else set(PROBLEMS)
    seeds = range(8000, 8001) if smoke else range(9000, 9500)
    budget = 100 if smoke else 4096
    arguments = provenance["arguments"]
    if provenance["source_commit"] != SOURCE_COMMIT or provenance["slurm_job_id"] != "24232628":
        raise ValueError("Unexpected source revision or Slurm job")
    if provenance["expected_cells"] != expected or arguments["budget"] != budget:
        raise ValueError("Unexpected matrix or query budget")
    if arguments["workers"] != (1 if smoke else 4) or arguments["repetitions"] != len(seeds) or arguments["seed_start"] != seeds.start:
        raise ValueError("Worker or seed specification differs")
    if provenance["confseq_commit"] != "5ffe733ca2447a2e28c2c91f3b00086173f2ab2c":
        raise ValueError("Unexpected official implementation")
    if not (root / "done").is_file() or completion["completed_cells"] != expected or completion["expected_cells"] != expected:
        raise ValueError("Missing complete matrix marker")
    key = lambda item: (item["problem"], item["method"], item["seed"])
    expected_keys = {(problem, method, seed) for problem in problems for method in METHODS for seed in seeds}
    if len(summaries) != expected or {key(item) for item in summaries} != expected_keys:
        raise ValueError("Consolidated matrix differs from protocol")
    indexed = {key(item): item for item in summaries}
    directories = sorted(root.glob("*/*/seed_*"))
    if len(directories) != expected:
        raise ValueError("Actual cell directory count differs")
    for directory in directories:
        if {path.name for path in directory.iterdir()} != {"config.json", "summary.json", "trace.npz", "done"}:
            raise ValueError("Incomplete or unexpected cell files")
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        if summary != indexed[key(summary)]:
            raise ValueError("Raw cell summary differs from consolidation")
        if config["problem"] != PROBLEMS[summary["problem"]]:
            raise ValueError("Problem parameters differ from frozen protocol")
        if config != {"problem": PROBLEMS[summary["problem"]], "method": summary["method"], "seed": summary["seed"], "budget": budget, "check_every_queries": 16, "pilot_queries": 4, "alpha": 0.05}:
            raise ValueError("Cell configuration differs from frozen protocol")
    return {"cells": expected, "source_commit": provenance["source_commit"], "queries": sum(item["calls"] for item in summaries), "runtime_seconds": completion["runtime_seconds"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    root, archive = args.root.resolve(), args.archive.resolve()
    if root != Path("/work/home/zenghang/probability_confirmation_20260922"):
        raise ValueError("Only the canonical authorized SCNet run may be packaged")
    if archive.exists() or (root / "recovery_evidence").exists():
        raise FileExistsError("Recovery artifacts already exist; inspect them before retrying")
    if (root / "job.status").read_text().strip() != "COMPLETE_PENDING_AUDIT":
        raise ValueError("The experiment has not completed")
    if (root / "audit.status").read_text().strip() != "COMPLETE":
        raise ValueError("The independent audit has not completed")
    for name, expected in [("confirmation", 20000), ("smoke", 5)]:
        analysis = json.loads((root / name / "analysis.json").read_text())
        if not analysis["audit_passed"] or analysis["audited_cells"] != expected:
            raise ValueError("The independent audit does not cover the complete directory")
    resource_text = subprocess.check_output([
        "sacct", "-j", "24232628", "-P", "-o",
        "JobID,State,ExitCode,Start,End,Elapsed,TimeLimit,TotalCPU,AllocCPUS,ReqMem,MaxRSS,ReqTRES,AllocTRES,NodeList",
    ], text=True)
    validate_resources(resource_text)
    checks = {name: validate_directory(root / name, smoke=name == "smoke") for name in ["confirmation", "smoke"]}
    source_hash = digest(root / "source.tar.gz")
    for name in checks:
        provenance = json.loads((root / name / "provenance.json").read_text())
        if provenance["source_archive_sha256"] != source_hash:
            raise ValueError("Source archive does not match recorded provenance")
    evidence = root / "recovery_evidence"
    evidence.mkdir()
    (evidence / "slurm_accounting.psv").write_text(resource_text)
    (evidence / "structural_validation.json").write_text(json.dumps({
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "source_archive_sha256": source_hash,
        "scope": "complete files, exact configurations and successful job exit; independent trace audit recorded in confirmation/analysis.json, smoke/analysis.json and audit.log",
    }, indent=2) + "\n")
    names = [path.name for path in sorted(root.iterdir()) if path.name != "code"]
    files = []
    for name in names:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(path)
        files.extend(sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path])
    if any(path.is_symlink() for path in files):
        raise ValueError("Symlinks are not accepted in research result archives")
    manifest = {str(path.relative_to(root)): {"sha256": digest(path), "bytes": path.stat().st_size} for path in sorted(files)}
    manifest_path = evidence / "files.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    files.append(manifest_path)
    with tarfile.open(archive, "x:gz", compresslevel=1) as bundle:
        for path in sorted(files):
            bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    receipt = {"archive": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size, "files": len(files), "cells": 20005}
    archive.with_suffix(archive.suffix + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
