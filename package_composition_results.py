import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tarfile


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def package(root, archive):
    root = Path(root)
    archive = Path(archive)
    if archive.exists():
        raise FileExistsError(archive)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = manifest["cells"]
    cells = sorted(root.glob("task_*/cell_*"))
    ids = []
    summaries = []
    for cell in cells:
        config = json.loads((cell / "config.json").read_text())
        index = config["cell_id"]
        ids.append(index)
        assert config["commit"] == manifest["commit"]
        for key, value in expected[index].items():
            assert config[key] == value, (index, key)
        for filename in ("config.json", "metrics.csv", "run.log", "summary.json", "done"):
            assert (cell / filename).is_file(), (index, filename)
        if config.get("kind") == "one_step":
            assert (cell / "diagnostics.npz").is_file()
        else:
            assert (cell / "samples.npz").is_file()
            assert (cell / "reference.npz").is_file()
        summary = json.loads((cell / "summary.json").read_text())
        assert summary["status"] == "completed"
        for key, value in summary.items():
            if isinstance(value, float):
                assert math.isfinite(value), (index, key, value)
        flat = {key: value for key, value in summary.items() if key != "runtime"}
        flat["relative_path"] = str(cell.relative_to(root))
        summaries.append(flat)
    assert sorted(ids) == list(range(len(expected))), (len(ids), len(expected))
    task_dirs = sorted(root.glob("task_*"))
    assert len(task_dirs) == manifest["tasks"]
    for task in task_dirs:
        assert (task / "done").is_file()
        summary = json.loads((task / "summary.json").read_text())
        assert summary["completed_cells"] == summary["expected_cells"]
    fields = sorted(set().union(*(row.keys() for row in summaries)))
    with (root / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(summaries, key=lambda row: row["cell_id"]))
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "recovery_manifest.json":
            assert not path.is_symlink()
            files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)})
    report = {"status": "completed", "expected_cells": len(expected), "verified_cells": len(cells), "verified_tasks": len(task_dirs), "source_commit": manifest["commit"], "files": files, "archive_format": "gzip tar; identical regular files are represented by internal hard links", "scope": "File/configuration/completion/hash validation; scientific metrics require separate recomputation."}
    report_path = root / "recovery_manifest.json"
    report_path.write_text(json.dumps(report, indent=2))
    files.append({"path": report_path.name, "bytes": report_path.stat().st_size, "sha256": digest(report_path)})
    seen = {}
    unique_bytes = 0
    with tarfile.open(archive, "w:gz", compresslevel=6, format=tarfile.PAX_FORMAT) as stream:
        for record in files:
            path = root / record["path"]
            arcname = f"{root.name}/{record['path']}"
            info = stream.gettarinfo(str(path), arcname=arcname)
            identity = (record["bytes"], record["sha256"])
            if record["bytes"] > 16384 and identity in seen:
                info.type = tarfile.LNKTYPE
                info.linkname = seen[identity]
                info.size = 0
                stream.addfile(info)
            else:
                with path.open("rb") as source:
                    stream.addfile(info, source)
                seen[identity] = arcname
                unique_bytes += record["bytes"]
    archive_hash = digest(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{archive_hash}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "sha256": archive_hash, "archive_bytes": archive.stat().st_size, "logical_bytes": sum(row["bytes"] for row in files), "unique_payload_bytes": unique_bytes, "cells": len(cells), "tasks": len(task_dirs)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    package(args.root, args.archive)
