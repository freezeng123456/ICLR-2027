import argparse
import hashlib
import json
from pathlib import Path
import subprocess


RUNS = {
    "decision_evidence_development_20260922": 6500,
    "decision_evidence_smoke_20260922": 6,
    "rare_counterevidence_development_20260922": 4000,
    "official_betting_control_20260922": 800,
    "official_empbern_control_20260922": 800,
    "matched_confirmation_local_smoke_20260922": 5,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-tracked", action="store_true")
    args = parser.parse_args()
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0"))
    receipts = []
    for name, expected in RUNS.items():
        root = Path("results") / name
        manifest = json.loads((root / "SHA256.json").read_text())
        files = {str(path.relative_to(root)): path for path in root.rglob("*") if path.is_file() and path.name != "SHA256.json"}
        if manifest.keys() != files.keys():
            raise ValueError(f"Manifest coverage differs: {name}")
        if len(list(root.glob("*/*/seed_*/done"))) != expected or not (root / "done").is_file():
            raise ValueError(f"Incomplete cell records: {name}")
        for relative, path in files.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[relative]:
                raise ValueError(f"Digest mismatch: {path}")
        all_files = list(files.values()) + [root / "SHA256.json"]
        if args.require_tracked and any(str(path) not in tracked for path in all_files):
            raise ValueError(f"Untracked experiment artifacts: {name}")
        receipts.append({"run": name, "cells": expected, "files": len(all_files), "bytes": sum(path.stat().st_size for path in all_files), "all_hashes_verified": True, "all_files_in_git_index": bool(args.require_tracked)})
    root = Path("results/probability_pivot_development_review_20260922")
    report = {"status": "passed", "covered_cells_including_smoke": sum(RUNS.values()), "experiment_files": sum(row["files"] for row in receipts), "runs": receipts}
    (root / "archive_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.iterdir()) if path.is_file() and path.name != "SHA256.json"}
    (root / "SHA256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
