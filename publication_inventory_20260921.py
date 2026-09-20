import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


SECRET = re.compile(rb"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[A-Z0-9]{16})")
TEXT_SUFFIXES = {".py", ".sh", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".md", ".txt", ".tex", ".bib", ".log", ".csv", ".tsv"}


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_file(item):
    source, destination = item
    if source.is_symlink():
        if not source.exists():
            raise FileNotFoundError(f"broken source link: {source}")
        return dict(path=destination, kind="symlink", target=os.readlink(source))
    size = source.stat().st_size
    if source.suffix in TEXT_SUFFIXES and SECRET.search(source.read_bytes()):
        raise RuntimeError(f"credential-like content requires review: {source}")
    return dict(path=destination, kind="file", bytes=size, sha256=digest_file(source))


def outgoing_scan(repository, revision):
    objects = subprocess.check_output(["git", "rev-list", "--objects", revision, "--not", "--remotes=origin"], cwd=repository, text=True).splitlines()
    checked, findings, total = 0, [], 0
    for row in objects:
        oid, _, name = row.partition(" ")
        if subprocess.check_output(["git", "cat-file", "-t", oid], cwd=repository, text=True).strip() != "blob":
            continue
        size = int(subprocess.check_output(["git", "cat-file", "-s", oid], cwd=repository, text=True))
        total += size
        if size > 100 * 1024 ** 2:
            findings.append(dict(path=name, reason="GitHub single-file size limit"))
        if Path(name).suffix in TEXT_SUFFIXES:
            data = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=repository)
            if SECRET.search(data):
                findings.append(dict(path=name, reason="credential-like content"))
            checked += 1
    return dict(checked_text_blobs=checked, new_blob_bytes=total, findings=findings)


def inventory(plan):
    items, roots = [], []
    for entry in plan["roots"]:
        source, prefix = Path(entry["source"]), entry["destination"]
        if not source.is_dir():
            raise NotADirectoryError(source)
        count = 0
        for directory, dirs, files in os.walk(source, followlinks=False):
            for name in list(dirs):
                path = Path(directory) / name
                if path.is_symlink():
                    items.append((path, (Path(prefix) / path.relative_to(source)).as_posix()))
                    dirs.remove(name)
                    count += 1
            for name in files:
                path = Path(directory) / name
                if name.endswith(".pyc") or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                    raise ValueError(f"canonical root contains a runtime cache: {path}")
                items.append((path, (Path(prefix) / path.relative_to(source)).as_posix()))
                count += 1
        roots.append(dict(destination=prefix, source=str(source), files=count))
    for entry in plan.get("files", []):
        items.append((Path(entry["source"]), entry["destination"]))
    if len({item[1] for item in items}) != len(items):
        raise ValueError("duplicate destination path")
    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(inspect_file, items))
    unique = {row["sha256"]: row["bytes"] for row in files if row["kind"] == "file"}
    oversized = [row for row in files if row.get("bytes", 0) > 100 * 1024 ** 2]
    return dict(status="inventoried", roots=roots, entries=sorted(files, key=lambda row: row["path"]),
                paths=len(files), total_file_bytes=sum(row.get("bytes", 0) for row in files),
                unique_sha256_files=len(unique), unique_file_bytes=sum(unique.values()), oversized=oversized,
                secret_scan="credential signature scan; text only; no claim of exhaustive secret detection")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.plan) == bool(args.repository):
        raise ValueError("provide exactly one of --plan or --repository")
    report = inventory(json.loads(args.plan.read_text())) if args.plan else outgoing_scan(args.repository, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"entries", "roots"}}, indent=2), flush=True)
    if report.get("findings") or report.get("oversized"):
        raise RuntimeError("publication inventory requires intervention")


if __name__ == "__main__":
    main()
