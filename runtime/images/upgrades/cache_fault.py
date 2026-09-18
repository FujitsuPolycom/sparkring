"""Inventory and recoverably corrupt only run-owned synthetic SparkCache chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

MARKER = ".sparkring-test-cache-owner"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def owned(root, owner):
    supplied = Path(root).absolute()
    if (
        any(path.is_symlink() for path in (supplied, *supplied.parents))
        or supplied != supplied.resolve()
    ):
        raise ValueError("Test cache cannot traverse symlinks")
    if not (supplied / MARKER).is_file() or (supplied / MARKER).read_text() != owner:
        raise ValueError("Cache directory is not owned by this qualification run")
    return supplied


def inventory(root, owner, persistent):
    root = owned(root, owner)
    selected = (root / persistent).resolve()
    if not selected.is_relative_to(root) or selected == root:
        raise ValueError("Persistent test cache must be contained under its owned root")
    rows = []
    for path in sorted(selected.rglob("*.spcc")):
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(selected)
        ):
            raise ValueError("Cache chunk is not a contained regular file")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            }
        )
    return rows


def corrupt(root, owner, persistent, *, workers_stopped=False):
    if not workers_stopped:
        raise ValueError("Stop both owned serving workers before fault injection")
    root = owned(root, owner)
    rows = inventory(root, owner, persistent)
    if not rows:
        raise ValueError("No published synthetic chunks exist to corrupt")
    backup = root / "fault-backup"
    backup.mkdir(exist_ok=False)
    records = []
    for index, row in enumerate(rows):
        path = root / row["path"]
        if row["bytes"] < 64:
            raise ValueError(
                "Published chunk is too short for bounded payload corruption"
            )
        destination = backup / (str(index) + ".spcc")
        shutil.copyfile(path, destination)
        if sha(destination) != row["sha256"]:
            raise ValueError("Fault backup does not reproduce the original chunk")
        offset = row["bytes"] // 2
        record = {
            **row,
            "backup": destination.relative_to(root).as_posix(),
            "offset": offset,
        }
        records.append(record)
        # Journal the recoverable original before changing the test chunk.
        (backup / "journal.json").write_text(
            json.dumps(
                {
                    "schema": "sparkring-cache-fault/v1",
                    "owner": owner,
                    "chunks": records,
                },
                indent=2,
            )
        )
        with path.open("r+b") as stream:
            stream.seek(offset)
            value = stream.read(1)
            stream.seek(offset)
            stream.write(bytes([value[0] ^ 1]))
        record["corrupt_sha256"] = sha(path)
    (backup / "journal.json").write_text(
        json.dumps(
            {"schema": "sparkring-cache-fault/v1", "owner": owner, "chunks": records},
            indent=2,
        )
    )
    return records


def repair(root, owner, *, workers_stopped=False):
    if not workers_stopped:
        raise ValueError(
            "Stop both owned serving workers before restoring fault backups"
        )
    root = owned(root, owner)
    journal = json.loads((root / "fault-backup/journal.json").read_text())
    if (
        journal.get("owner") != owner
        or journal.get("schema") != "sparkring-cache-fault/v1"
    ):
        raise ValueError("Fault journal belongs to another run")
    result = []
    for row in journal["chunks"]:
        target = (root / row["path"]).resolve()
        backup = (root / row["backup"]).resolve()
        if (
            not target.is_relative_to(root)
            or not backup.is_relative_to(root / "fault-backup")
            or sha(backup) != row["sha256"]
        ):
            raise ValueError("Fault recovery path or backup differs")
        current = sha(target) if target.exists() else None
        corrupted = row.get("corrupt_sha256")
        if corrupted is None:
            original = bytearray(backup.read_bytes())
            offset = row["offset"]
            if type(offset) is not int or not 0 <= offset < len(original):
                raise ValueError("Fault journal offset is invalid")
            original[offset] ^= 1
            corrupted = hashlib.sha256(original).hexdigest()
        if current not in (None, row["sha256"], corrupted):
            result.append({"path": row["path"], "result": "replacement preserved"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(backup, target)
        result.append(
            {"path": row["path"], "result": "original restored", "sha256": sha(target)}
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "corrupt", "repair"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--persistent", default="persistent")
    parser.add_argument("--workers-stopped", action="store_true")
    args = parser.parse_args()
    if args.action == "inventory":
        result = inventory(args.root, args.owner, args.persistent)
    elif args.action == "corrupt":
        result = corrupt(
            args.root, args.owner, args.persistent, workers_stopped=args.workers_stopped
        )
    else:
        result = repair(args.root, args.owner, workers_stopped=args.workers_stopped)
    print(json.dumps(result), flush=True)
