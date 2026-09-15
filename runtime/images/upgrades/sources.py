"""Resolve immutable Git inputs and apply bounded, path-admitted patches."""

from __future__ import annotations

import io
from pathlib import Path
import shutil
import tarfile
import re

from .contracts import allowed, encoded, oid, relative, require, sha
from .io import checked, command


def git(root, *args, seconds=120, limit=1024 * 1024):
    return checked(
        [
            "git",
            "-c",
            "core.hooksPath=" + str(Path(root) / ".no-hooks"),
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.longpaths=true",
            "-c",
            "protocol.ext.allow=never",
            "-C",
            str(root),
            *args,
        ],
        seconds=seconds,
        limit=limit,
    )


def discover(source, cache, budgets):
    mirror = Path(cache) / source["id"]
    if not mirror.exists():
        mirror.mkdir(parents=True)
        git(mirror, "init", "--bare")
    repository = source["repository"]
    ref = source["ref"]
    if oid(ref):
        target = ref
    else:
        rows = (
            git(
                mirror, "ls-remote", repository, ref, seconds=budgets["command_seconds"]
            )
            .decode()
            .splitlines()
        )
        require(len(rows) == 1, "Tracking ref did not resolve to one commit")
        target = rows[0].split()[0]
    require(oid(target), "Git discovery did not return a full commit")
    for revision in dict.fromkeys((source["baseline"], target)):
        present = command(
            ["git", "-C", str(mirror), "cat-file", "-e", revision + "^{commit}"],
            seconds=10,
        )
        if present["returncode"]:
            git(
                mirror,
                "fetch",
                "--no-tags",
                "--depth=1",
                repository,
                revision,
                seconds=budgets["command_seconds"],
                limit=budgets["output_bytes"],
            )
        require(
            git(mirror, "rev-parse", revision + "^{commit}").decode().strip()
            == revision,
            "Fetched commit identity differs",
        )
    changed = (
        git(mirror, "diff", "--name-only", source["baseline"], target, "--")
        .decode()
        .splitlines()
    )
    return {
        "id": source["id"],
        "repository": repository,
        "baseline": source["baseline"],
        "target": target,
        "changed_paths": changed,
        "native_changed": [p for p in changed if allowed(p, source["native_paths"])],
        "mirror": str(mirror),
    }


def materialize(record, revision, target, limit, excluded=()):
    target = Path(target)
    require(not target.exists(), "Snapshot destination already exists")
    modes = git(record["mirror"], "ls-tree", "-r", revision).decode().splitlines()
    require(
        not any(line.startswith("160000 ") for line in modes),
        "Submodule source requires an explicitly reviewed materialization recipe",
    )
    data = git(record["mirror"], "archive", "--format=tar", revision, limit=limit)
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = [
            item
            for item in archive.getmembers()
            if not allowed(item.name.rstrip("/"), excluded)
        ]
        require(
            sum(item.size for item in members) <= limit, "Source size exceeds budget"
        )
        for item in members:
            relative(item.name.rstrip("/"))
            require(
                ".git" not in Path(item.name).parts and (item.isdir() or item.isfile()),
                "Source archive contains a link or special file: "
                + item.name
                + "; use an explicitly reviewed source recipe",
            )
        archive.extractall(target, members=members, filter="data")
    git(target, "init")
    git(target, "config", "core.longpaths", "true")
    git(target, "add", "--all")
    return target


def inventory(root):
    result = {}
    root = Path(root)
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root)
        if ".git" in name.parts:
            continue
        require(not path.is_symlink(), "Candidate source contains a symlink")
        if path.is_file():
            result[name.as_posix()] = sha(
                encoded(
                    {
                        "bytes": sha(path.read_bytes()),
                        "executable": bool(path.stat().st_mode & 0o111),
                    }
                )
            )
    return result


def tree_digest(root):
    return sha(encoded(inventory(root)))


def copy_snapshot(source, target):
    require(not Path(target).exists(), "Candidate snapshot already exists")
    shutil.copytree(source, target)
    return Path(target)


def apply_patch(root, patch, editable=None, protected=()):
    # Approved carried patches have an immutable policy hash. Only agent edits
    # need a full before/after path audit; accepted trees are hashed separately.
    before = inventory(root) if editable is not None else None
    result = command(
        [
            "git",
            "-c",
            "core.hooksPath=" + str(Path(root) / ".no-hooks"),
            "-C",
            str(root),
            "apply",
            "--check",
            "--whitespace=nowarn",
            "-",
        ],
        input_bytes=patch,
        seconds=30,
    )
    if result["returncode"] or result["uncertain"]:
        return False, result["stderr"].decode(errors="replace")[:4000]
    checked(
        ["git", "-C", str(root), "apply", "--whitespace=nowarn", "-"],
        input_bytes=patch,
        seconds=30,
    )
    if editable is None:
        return True, []
    after = inventory(root)
    changed = sorted(
        p for p in set(before) | set(after) if before.get(p) != after.get(p)
    )
    require(
        changed and all(allowed(p, editable) for p in changed),
        "Agent patch modified an unapproved path or made no change",
    )
    require(
        not any(allowed(p, protected) for p in changed),
        "Agent patch modified a native or protected path",
    )
    return True, changed


def apply_fragments(root, patch):
    """Preserve clean file-level hunks and report rejected fragments for repair."""
    parts = re.split(rb"(?=^diff --git )", patch, flags=re.MULTILINE)
    parts = [part for part in parts if part.strip()]
    failed = []
    for part in parts:
        success, detail = apply_patch(root, part)
        if not success:
            failed.append({"patch": part.decode(), "error": detail})
    return not failed, failed


def complete_patch(root, upstream_tree):
    git(root, "add", "--all")
    return git(
        root,
        "diff",
        "--cached",
        "--binary",
        "--full-index",
        upstream_tree,
        "--",
        limit=16 * 1024 * 1024,
    )
