import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--raw-sha256", required=True)
    parser.add_argument("--audit-archive", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert sha256(args.raw_archive) == args.raw_sha256
    assert sha256(args.audit_archive) == args.audit_sha256
    manifest = json.loads((args.root / "manifest.json").read_text())
    state = json.loads((args.root / "state.json").read_text())
    audit = json.loads((args.audit / "audit.json").read_text())
    assert state["status"] == "completed" and state["completed"] == state["expected"] == 630
    assert (args.root / "launcher.exit").read_text().strip() == "0"
    assert audit["status"] == "passed" and audit["audited_cells"] == 630
    assert audit["sensitivity_configurations_checked"] == 66
    assert audit["input_checks"]["checkpoints_reloaded"] == 5
    assert audit["input_checks"]["data_seeds_reconstructed"] == 20
    with (args.audit / "cells.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert sorted(int(row["cell_id"]) for row in rows) == list(range(630))
    count = 0
    for expected in manifest["cells"]:
        cell = args.root / "cells" / f"cell_{expected['cell_id']:04d}"
        assert (cell / "solid_done").read_text().strip() == "completed"
        config = json.loads((cell / "config.json").read_text())
        assert all(config[key] == value for key, value in expected.items())
        receipt = json.loads((cell / "receipt.json").read_text())
        assert receipt["status"] == "completed"
        for name, digest in receipt["files"].items():
            assert sha256(cell / name) == digest, (cell, name)
            count += 1
    artifacts = ["audit.json", "input_checks.json", "reference_checks.json", "statistics.json", "cells.csv"]
    artifacts += [f"{stem}.{extension}" for stem in ["heldout", "boundary", "tail_sensitivity"] for extension in ["pdf", "png"]]
    report = dict(status="passed", verified_at=datetime.now(timezone.utc).isoformat(),
                  sampling_commit=manifest["commit"], cells=630, sensitivity_configurations=66,
                  checked_cell_file_hashes=count, formal_run_seconds=state["seconds"],
                  raw_archive_sha256=args.raw_sha256, audit_archive_sha256=args.audit_sha256,
                  artifact_sha256={name: sha256(args.audit / name) for name in artifacts},
                  raw_root=str(args.root.resolve()), audit_root=str(args.audit.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
