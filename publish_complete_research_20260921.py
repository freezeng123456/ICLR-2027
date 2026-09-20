import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from publication_inventory_20260921 import digest_file


def git(directory, *args, capture=False):
    command = ["git", "-c", "pack.window=0", "-c", "pack.threads=2", *args]
    if capture:
        return subprocess.check_output(command, cwd=directory, text=True).strip()
    subprocess.run(command, cwd=directory, check=True)


def materialize(plan, report, target):
    mappings = [(row["destination"], Path(row["source"])) for row in plan["roots"]]
    standalone = {row["destination"]: Path(row["source"]) for row in plan.get("files", [])}
    for row in report["entries"]:
        destination = target / row["path"]
        if row["path"] in standalone:
            source = standalone[row["path"]]
        else:
            matches = [(prefix, path) for prefix, path in mappings if row["path"].startswith(prefix + "/")]
            if len(matches) != 1:
                raise ValueError(f"ambiguous source mapping: {row['path']}")
            prefix, base = matches[0]
            source = base / row["path"][len(prefix) + 1:]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if row["kind"] == "symlink":
            if not destination.is_symlink():
                if destination.exists():
                    raise FileExistsError(destination)
                os.symlink(row["target"], destination)
            if os.readlink(destination) != row["target"]:
                raise RuntimeError("publication symlink differs")
        else:
            if destination.is_symlink():
                raise RuntimeError("unexpected destination symlink")
            if destination.exists():
                if digest_file(destination) != row["sha256"]:
                    raise RuntimeError(f"existing publication content differs: {destination}")
            else:
                # 完成结果保持只读使用；硬链接避免重复占用本地存储。
                os.link(source, destination)
    for row in report["entries"]:
        if row["kind"] == "symlink" and not (target / row["path"]).exists():
            raise RuntimeError(f"publication link target missing: {row['path']}")


def batches(entries, limit):
    known, current, size = set(), [], 0
    for row in entries:
        addition = row.get("bytes", 0) if row.get("sha256") not in known else 0
        if current and size + addition > limit:
            yield current
            current, size = [], 0
        current.append(row["path"])
        size += addition
        if "sha256" in row:
            known.add(row["sha256"])
    if current:
        yield current


def remote_head(target, branch):
    rows = git(target, "ls-remote", "origin", "refs/heads/" + branch, capture=True).splitlines()
    if len(rows) != 1:
        raise RuntimeError("remote branch was not received")
    return rows[0].split()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--branch", default="results/complete-research-20260921")
    parser.add_argument("--manifest-prefix", default="results/raw_research_20260921")
    parser.add_argument("--batch-mib", type=int, default=400)
    args = parser.parse_args()
    plan, report = json.loads(args.plan.read_text()), json.loads(args.inventory.read_text())
    target = args.target.resolve()
    prefix = Path(args.manifest_prefix)
    if prefix.is_absolute() or ".." in prefix.parts or prefix.parts[0] != "results":
        raise ValueError("publication prefix must stay beneath results")
    if any(not row["path"].startswith(prefix.as_posix() + "/") for row in report["entries"]):
        raise ValueError("inventory entries must stay beneath their own manifest prefix")
    if git(target, "branch", "--show-current", capture=True) != args.branch:
        raise ValueError("publisher is on the wrong branch")
    if report["oversized"]:
        raise ValueError("oversized files require a separate storage decision")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(args.state.read_text()) if args.state.exists() else dict(batches=[], status="materializing")
    def save_state():
        args.state.write_text(json.dumps(state, indent=2) + "\n")
    save_state()
    materialize(plan, report, target)
    manifest_path = target / prefix / "FILE_MANIFEST.json"
    published = {key: value for key, value in report.items() if key != "roots"}
    published["roots"] = [{key: value for key, value in row.items() if key != "source"} for row in report["roots"]]
    manifest_path.write_text(json.dumps(published, indent=2) + "\n")
    chunks = list(batches(report["entries"], args.batch_mib * 1024 ** 2))
    if state["batches"] and (state.get("expected_paths") != report["paths"]
                             or state.get("manifest_sha256") != digest_file(manifest_path)):
        raise RuntimeError("resume requires the same frozen publication inventory")
    state.update(status="publishing", total_batches=len(chunks), expected_paths=report["paths"], manifest_sha256=digest_file(manifest_path))
    save_state()
    for index, paths in enumerate(chunks):
        if index < len(state["batches"]):
            continue
        if index == 0:
            paths += [(prefix / name).as_posix() for name in ("FILE_MANIFEST.json", "README.md")]
        pathspec = args.state.parent / "batch-paths.nul"
        pathspec.write_bytes(b"\0".join(path.encode() for path in paths) + b"\0")
        started = time.monotonic()
        git(target, "add", "-f", "--pathspec-from-file=" + str(pathspec.resolve()), "--pathspec-file-nul")
        if git(target, "diff", "--cached", "--name-only", capture=True):
            git(target, "commit", "--quiet", "-m", f"Publish complete research raw records {index + 1}/{len(chunks)}")
        commit = git(target, "rev-parse", "HEAD", capture=True)
        git(target, "push", "origin", "HEAD:refs/heads/" + args.branch)
        received = remote_head(target, args.branch)
        if received != commit:
            raise RuntimeError("published revision does not match local commit")
        state["batches"].append(dict(index=index, files=len(paths), commit=commit, remote_commit=received, seconds=time.monotonic() - started))
        save_state()
        print(json.dumps(dict(batch=index + 1, batches=len(chunks), files=len(paths), commit=commit)), flush=True)
    commit = git(target, "rev-parse", "HEAD", capture=True)
    tree_paths = git(target, "ls-tree", "-r", "--name-only", "HEAD", prefix.as_posix(), capture=True).splitlines()
    expected = {row["path"] for row in report["entries"]}
    if not expected.issubset(set(tree_paths)):
        raise RuntimeError("published tree omits manifest paths")
    # 使用 GitHub 内容 API；保持系统证书校验，并逐目录保存下载核验进度。
    checked = state.get("fresh_github_downloads", []) if state.get("download_revision") == commit else []
    checked_paths = {row["path"] for row in checked}
    state.update(status="verifying_github_downloads", download_revision=commit, fresh_github_downloads=checked)
    save_state()
    for root in report["roots"]:
        choices = [row for row in report["entries"] if row["path"].startswith(root["destination"] + "/") and row["kind"] == "file"]
        if not choices:
            continue
        samples = [row for row in choices if row["path"].endswith("samples.npz")]
        row = (samples or choices)[0]
        if row["path"] in checked_paths:
            continue
        oid = git(target, "rev-parse", commit + ":" + row["path"], capture=True)
        endpoint = "repos/freezeng123456/ICLR-2027/git/blobs/" + oid
        blob = json.loads(subprocess.check_output(["gh", "api", endpoint], timeout=60))
        if blob.get("encoding") != "base64" or blob.get("sha") != oid:
            raise RuntimeError("unexpected GitHub blob encoding or object identifier")
        content = base64.b64decode(blob["content"])
        if len(content) != row["bytes"] or blob["size"] != row["bytes"]:
            raise RuntimeError("downloaded GitHub blob size differs")
        actual = hashlib.sha256(content).hexdigest()
        if actual != row["sha256"]:
            raise RuntimeError(f"GitHub download differs: {row['path']}")
        checked.append(dict(path=row["path"], sha256=actual, github_blob=oid))
        save_state()
        print(json.dumps(dict(download_verified=row["path"], checked_roots=len(checked))), flush=True)
    state.update(status="published_and_verified", commit=commit, remote_commit=remote_head(target, args.branch),
                 published_manifest_paths=len(expected), tree_entries=len(tree_paths), fresh_github_downloads=checked)
    save_state()
    print(json.dumps({key: state[key] for key in ["status", "commit", "published_manifest_paths", "tree_entries"]}), flush=True)


if __name__ == "__main__":
    main()
