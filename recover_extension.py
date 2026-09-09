import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def recover(root, output):
    jobs = json.loads((root / "jobs.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    records = []
    for kind, expected in [("oracle", 1080), ("learned", 1000)]:
        tasks = sorted((root / kind).glob("task_*"))
        assert len(tasks) == 40 and all((task / "done").is_file() for task in tasks)
        cells = sorted((root / kind).glob("task_*/cell_*"))
        assert len(cells) == expected
        assert all((cell / "extension_done").is_file() for cell in cells)
        manifest = json.loads((root / kind / "manifest.json").read_text())
        assert manifest["commit"] == provenance["commit"] and len(manifest["cells"]) == expected
    for seed in range(5):
        directory = root / "training" / f"training_{seed}"
        assert (directory / "done").is_file()
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["status"] == "completed" and summary["updates"] == 40000
    for kind, job in jobs.items():
        table = subprocess.check_output(["sacct", "-j", job, "-X", "-P", "--format=JobID,State,ExitCode,ElapsedRaw,ReqTRES,AllocTRES,NodeList"], text=True)
        (root / f"{kind}_slurm.tsv").write_text(table)
        rows = list(csv.DictReader(io.StringIO(table), delimiter="|"))
        assert len(rows) == (5 if kind == "training" else 40), (kind, len(rows))
        assert all(row["State"] == "COMPLETED" and row["ExitCode"] == "0:0" for row in rows)
        for row in rows:
            allocated = dict(item.split("=", 1) for item in row["AllocTRES"].split(",") if "=" in item)
            assert allocated["gres/gpu"] == "1" and allocated["cpu"] == "4"
        records.extend(rows)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "recovery_manifest.json":
            assert not path.is_symlink()
    files = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(root.rglob("*")) if path.is_file() and path.name != "recovery_manifest.json"]
    recovery = {"source_commit": provenance["commit"], "files": files, "verified_oracle_cells": 1080, "verified_learned_cells": 1000,
                "verified_training_runs": 5, "verified_scheduler_tasks": len(records), "allocated_gpu_hours": sum(int(row["ElapsedRaw"]) for row in records) / 3600,
                "status": "recovered and scheduler verified; independent scientific audit pending"}
    (root / "recovery_manifest.json").write_text(json.dumps(recovery, indent=2))
    assert not output.exists()
    canonical = {}
    checksums = {item["path"]: (item["sha256"], item["bytes"]) for item in files}
    with tarfile.open(output, "w:gz", compresslevel=1) as archive:
        archive.add(root, arcname=root.name, recursive=False)
        for path in sorted(root.rglob("*")):
            relative = str(path.relative_to(root))
            name = str(Path(root.name) / relative)
            if path.is_dir():
                archive.add(path, arcname=name, recursive=False)
                continue
            info = archive.gettarinfo(str(path), arcname=name)
            key = checksums.get(relative)
            if key is not None and key in canonical:
                info.type = tarfile.LNKTYPE
                info.linkname = canonical[key]
                info.size = 0
                archive.addfile(info)
            else:
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
                if key is not None:
                    canonical[key] = name
    checksum = digest(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{checksum}  {output.name}\n")
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size, "sha256": checksum, "files": len(files), "gpu_hours": recovery["allocated_gpu_hours"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.root, args.output)
