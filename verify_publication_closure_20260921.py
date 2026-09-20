import base64
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "results/continued_iteration_20260921"
RAW = ROOT.parent / "publish-all-20260921"


def git(directory, *args):
    return subprocess.check_output(["git", *args], cwd=directory, text=True).strip()


def download_check(directory, revision, path, expected):
    oid = git(directory, "rev-parse", revision + ":" + path)
    blob = json.loads(subprocess.check_output(["gh", "api", "repos/freezeng123456/ICLR-2027/git/blobs/" + oid], timeout=60))
    if blob.get("encoding") != "base64" or blob.get("sha") != oid:
        raise ValueError("unexpected GitHub object")
    content = base64.b64decode(blob["content"])
    digest = hashlib.sha256(content).hexdigest()
    if len(content) != blob["size"] or digest != expected:
        raise ValueError(f"GitHub content mismatch: {path}")
    return dict(revision=revision, path=path, github_blob=oid, bytes=len(content), sha256=digest)


def main():
    historical = json.loads((REPORT / "historical-publication-receipt.json").read_text())
    continued = json.loads((REPORT / "continued-publication-receipt.json").read_text())
    inventories = [json.loads((ROOT / "work" / name / "inventory.json").read_text())
                   for name in ("publication-20260921", "publication-continued-20260921")]
    receipts = [historical, continued]
    if any(value["status"] != "published_and_verified" for value in receipts):
        raise ValueError("unfinished publication")
    research_revision = git(ROOT, "rev-parse", "HEAD")
    raw_revision = continued["commit"]
    for branch, expected in (("research/iclr-review-dual-20260921", research_revision),
                             ("results/complete-research-20260921", raw_revision)):
        if git(ROOT, "ls-remote", "origin", "refs/heads/" + branch).split()[0] != expected:
            raise ValueError("remote revision mismatch")
    paths = set(git(RAW, "ls-tree", "-r", "--name-only", raw_revision, "results/raw_research_20260921", "results/raw_continued_iteration_20260921").splitlines())
    entries = [entry for inventory in inventories for entry in inventory["entries"]]
    if len({entry["path"] for entry in entries}) != 64904 or any(entry["path"] not in paths for entry in entries):
        raise ValueError("incomplete combined raw tree")
    downloads = []
    for index in range(5):
        path = f"results/raw_research_20260921/extension/run/training/training_{index}/final.pt"
        expected = next(entry["sha256"] for entry in entries if entry["path"] == path)
        downloads.append(download_check(RAW, raw_revision, path, expected))
    for path in ("output/pdf/ICLR_2027_research_draft_continued_20260921.pdf",
                 "results/continued_iteration_20260921/delivery-verification.json",
                 "results/continued_iteration_20260921/learned-independent-audit.json",
                 "results/continued_iteration_20260921/sensor-independent-audit.json"):
        expected = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        downloads.append(download_check(ROOT, research_revision, path, expected))
    unique = {entry["sha256"]: entry["bytes"] for entry in entries if entry["kind"] == "file"}
    result = dict(status="published_and_verified", research_content_snapshot=research_revision,
                  raw_branch_revision=raw_revision, raw_manifest_paths=len(entries),
                  raw_tree_entries_including_indexes=len(paths), regular_files=sum(e["kind"] == "file" for e in entries),
                  symlinks=sum(e["kind"] == "symlink" for e in entries),
                  logical_bytes=sum(e.get("bytes", 0) for e in entries),
                  unique_sha256_files=len(unique), unique_file_bytes=sum(unique.values()),
                  complete_nonempty_roots_sampled=sum(len(r["fresh_github_downloads"]) for r in receipts),
                  additional_fresh_downloads=downloads,
                  verification_scope="All manifest paths checked against Git tree anchored by matching GitHub commit; one fresh object per nonempty canonical root, five checkpoints and four research artifacts downloaded and SHA-256 checked. Not a full redownload of all raw bytes.",
                  exclusions="Credentials, dependencies, caches and duplicate transport containers; unique retained scientific contents are indexed.")
    output = REPORT / "publication-summary.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "additional_fresh_downloads"}, indent=2))


if __name__ == "__main__":
    main()
