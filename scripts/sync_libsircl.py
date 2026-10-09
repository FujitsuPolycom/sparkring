"""Vendor a libsircl snapshot into spark_transport/libsircl, or check the vendored copy against its manifest.

A libsircl snapshot is a directory that libsircl's own workspace writes:
``tree/`` with the library's files, ``FILES.sha256`` with one
``<sha256>  ./<path>`` line per file of ``tree/``, and ``MANIFEST``, whose
``Tree digest`` line states the SHA-256 of ``FILES.sha256``. That digest
identifies the snapshot; its first eight hexadecimal digits name it.

``sync SNAPSHOT --tree-digest DIGEST`` requires the SHA-256 of the snapshot's
``FILES.sha256`` to equal ``DIGEST`` and the digest ``MANIFEST`` states,
every file of ``tree/`` to match its line and every line to name a file. It
then replaces the contents of the target directory with the snapshot's files
except those ``EXCLUDED`` names, and writes two files of its own:

- ``SNAPSHOT.sha256``: the snapshot's ``FILES.sha256``, byte for byte, so the
  tree digest and every file's digest stay checkable;
- ``SNAPSHOT.json`` (``sparkring-libsircl-snapshot/v2``): the tree digest,
  the library version (the tree's ``VERSION``), the number of vendored files,
  the exclusion rules, every excluded path and every rewritten file with its
  snapshot and vendored SHA-256.

A few of the library's documents name directories of the workspace that
wrote the snapshot. ``REWRITES`` replaces those passages in the vendored
copy. Before it changes the target, the sync refuses a snapshot whose
vendored text still matches ``WORKSPACE_SHAPES`` (a local Windows user path
or a reference to that workspace's own directories), and a target that is
neither absent, empty nor a vendored copy (a directory holding a
``SNAPSHOT.json`` of a ``sparkring-libsircl-snapshot`` schema).

``check`` requires every vendored file to match ``SNAPSHOT.sha256`` (a
rewritten file: the vendored SHA-256 its record states, for the snapshot
SHA-256 that ``SNAPSHOT.sha256`` lists), every listed file that is not
excluded to be present, no other file to be there, apart from the outputs of
libsircl's own build and Python caches (``IGNORED``), and no vendored text to
match ``WORKSPACE_SHAPES``. Neither action changes a file that the snapshot
does not name.

    python3 scripts/sync_libsircl.py sync SNAPSHOT_DIRECTORY --tree-digest TREE_DIGEST
    python3 scripts/sync_libsircl.py check
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "spark_transport" / "libsircl"
SCHEMA = "sparkring-libsircl-snapshot/v2"
SCHEMAS = ("sparkring-libsircl-snapshot/v1", SCHEMA)
MANIFEST = "SNAPSHOT.sha256"
RECORD = "SNAPSHOT.json"
OWN = (MANIFEST, RECORD)
# Snapshot paths that stay with the snapshot: (rule, reason). A rule ending in "/" matches a top-level
# directory; any other rule matches a path component.
EXCLUDED = (
    ("__pycache__", "compiled Python cache, not source"),
    ("verification/", "run evidence that STATUS.md summarizes; its logs name the build workstation's local paths "
                      "and the host names of the Sparks it ran on"),
    ("requests/", "change requests for SIRCL's own package; their landing scripts and notes name directories of "
                  "the workspace that wrote the snapshot"),
)
# Passages of vendored documents that name directories of the workspace that wrote the snapshot:
# (path, pattern, replacement); every match is replaced. When a snapshot rewords a passage so that its
# pattern stops matching, WORKSPACE_SHAPES refuses the snapshot until the rule follows.
REWRITES = (
    ("README.md", rb"is the clean-room implementation tree copied to\n  `\.\./sircl-current`;",
     b"is a copy of SIRCL's clean-room implementation tree;\n "),
    ("README.md", rb"`requests/` holds\n  changes the library needs inside SIRCL's package, prepared for its lead\.",
     b"The library's change\n  requests to SIRCL's package stay with the libsircl snapshot and are not vendored."),
    ("STATUS.md", rb"is the lead workspace's `ring8/cleanroom/impl` tree, copied byte for byte to\n"
                  rb"`\.\./sircl-current` \(",
     b"is SIRCL's clean-room implementation tree, copied byte for byte\n("),
    ("STATUS.md", rb" \(`requests/README\.md`\)", b" (kept with the libsircl snapshot, not vendored)"),
    ("STATUS.md", rb"naming the lead implementation tree's `spark_transport/sircl`",
     b"naming the SIRCL reference tree's `spark_transport/sircl`"),
    ("STATUS.md", rb"The lead implementation tree holds the same\ntext with CRLF endings\. ",
     b"This repository's SIRCL package (`spark_transport/sircl`) holds the same\nbytes. "),
    ("RUNBOOK.md", rb"`REF` is the SIRCL reference tree\n\(`\.\./sircl-current/spark_transport/sircl`\); `LOCK` is "
                   rb"`ring8/cleanroom/impl/\.build/gpu-lock\.sh` of the\nlead workspace\. ",
     b"`REF` is the SIRCL reference tree's\n`spark_transport/sircl` directory; `LOCK` is the script of that GPU lock.\n"),
    ("SOURCE_SNAPSHOT.json", rb"The copy at \.\./sircl-current is byte-identical", b"The copy is byte-identical"),
    ("SOURCE_SNAPSHOT.json", rb'"lead": "[^"]*"',
     b'"reference": "SIRCL\'s clean-room implementation tree (its local path is not vendored)"'),
    ("SOURCE_SNAPSHOT.json", rb'"selected": "lead"', b'"selected": "reference"'),
    # The reference tree's run results (spark_transport/sircl/sircl-ring-results), which the SIRCL tree this
    # repository vendors does not hold, are left out of the vendored file list.
    ("SOURCE_SNAPSHOT.json", rb'excluding build outputs \(\.build, \.pytest_tmp\) and caches\."',
     b'excluding build outputs (.build, .pytest_tmp) and caches. This copy of the record leaves the reference '
     b'tree\'s run results out of its file list."'),
    ("SOURCE_SNAPSHOT.json", rb'\n  "spark_transport/sircl/sircl-ring-results/[^"\n]*": \{\n   "sha256": "[0-9a-f]{64}",'
                             rb'\n   "bytes": [0-9]+\n  \},', b""),
)
# Text that no vendored file may hold: a local Windows user path (a drive or WSL mount followed by Users,
# or AppData), references to the snapshot workspace's own directories and paths of the SIRCL reference
# tree's run results.
WORKSPACE_SHAPES = re.compile(
    rb"[A-Za-z]:(?:\\|/)+Users(?:\\|/)|/mnt/[a-z]/Users/|AppData(?:\\|/)|sircl-current|cleanroom/impl|sircl-ring-results/"
    rb"|\blead (?:workspace|scratchpad|implementation tree)|\bits lead\b", re.I)
# Outputs of libsircl's own build (make's default BUILD directory) and Python caches, which check tolerates.
IGNORED = ("build", "__pycache__")
_LINE = re.compile(r"([0-9a-f]{64}) [ *]\./(.+)")
_TREE_DIGEST = re.compile(r"Tree digest[^:\n]*:\s*([0-9a-f]{64})")
_DIGEST = re.compile(r"[0-9a-f]{64}")


class SnapshotError(ValueError):
    """A snapshot or the vendored copy does not match its manifest."""


def require(condition, message):
    if not condition:
        raise SnapshotError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def parse_manifest(data):
    """``{path: sha256}`` of a ``FILES.sha256`` text; refuses malformed, duplicate or escaping paths."""
    entries = {}
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        found = _LINE.fullmatch(line)
        require(found is not None, f"manifest line {number} is not '<sha256>  ./<path>'")
        digest, path = found.groups()
        relative = PurePosixPath(path)
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in path
                and str(relative) == path, f"manifest line {number} names an unsafe path: {path}")
        require(path not in entries, f"manifest lists {path} twice")
        require(path not in OWN, f"the snapshot holds {path}, a name the vendored copy reserves")
        entries[path] = digest
    require(entries, "the manifest lists no file")
    return entries


def excluded(path):
    """The reason ``path`` stays with the snapshot, or None when it is vendored."""
    parts = PurePosixPath(path).parts
    for rule, reason in EXCLUDED:
        if rule.endswith("/") and parts[0] == rule[:-1]:
            return reason
        if not rule.endswith("/") and rule in parts:
            return reason
    return None


def read_snapshot(directory, tree_digest):
    """``(manifest bytes, {path: sha256}, version)`` of a snapshot directory after checking it whole."""
    directory = Path(directory)
    require(_DIGEST.fullmatch(tree_digest or ""), "--tree-digest is the snapshot's 64-digit SHA-256")
    data = (directory / "FILES.sha256").read_bytes()
    require(sha256(data) == tree_digest, f"FILES.sha256 has SHA-256 {sha256(data)}, not the tree digest "
                                         f"{tree_digest}")
    stated = _TREE_DIGEST.search((directory / "MANIFEST").read_text(encoding="utf-8"))
    require(stated is not None and stated.group(1) == tree_digest,
            "the snapshot's MANIFEST does not state this tree digest")
    entries = parse_manifest(data)
    tree = directory / "tree"
    present = {path.relative_to(tree).as_posix() for path in tree.rglob("*") if path.is_file() or path.is_symlink()}
    require(not (present - set(entries)), "tree/ holds files the manifest does not list: "
            + ", ".join(sorted(present - set(entries))[:5]))
    require(not (set(entries) - present), "the manifest lists files tree/ lacks: "
            + ", ".join(sorted(set(entries) - present)[:5]))
    for path, digest in sorted(entries.items()):
        source = tree / path
        require(not source.is_symlink(), f"tree/{path} is a symbolic link")
        require(sha256(source.read_bytes()) == digest, f"tree/{path} differs from its manifest line")
    version = (tree / "VERSION").read_text(encoding="utf-8").strip() if "VERSION" in entries else None
    require(version and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version), "the tree's VERSION is not x.y.z")
    return data, entries, version


def rewrite(path, data):
    """``data`` of the snapshot file ``path`` with the ``REWRITES`` passages of that path replaced."""
    for rule_path, pattern, replacement in REWRITES:
        if rule_path != path:
            continue
        data = re.sub(pattern, lambda _: replacement, data)
    return data


def workspace_references(files):
    """The paths of the text files in ``{path: bytes}`` that match ``WORKSPACE_SHAPES``."""
    return sorted(path for path, data in files.items() if b"\0" not in data and WORKSPACE_SHAPES.search(data))


def record(tree_digest, version, entries, rewritten):
    vendored = [path for path in entries if excluded(path) is None]
    return {"schema": SCHEMA, "snapshot": tree_digest[:8], "tree_digest": tree_digest, "version": version,
            "manifest": MANIFEST, "files": len(vendored),
            "excluded": {"rules": [{"match": rule, "reason": reason} for rule, reason in EXCLUDED],
                         "paths": sorted(path for path in entries if excluded(path) is not None)},
            "rewritten": {path: dict(rewritten[path]) for path in sorted(rewritten)}}


def encoded(value):
    return (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()


def replaceable(target):
    """Refuses a ``target`` that is neither absent, empty nor a vendored copy, before anything is removed."""
    if not target.exists():
        return
    require(target.is_dir() and not target.is_symlink(), f"the target {target} is not a directory")
    if not any(target.iterdir()):
        return
    try:
        schema = json.loads((target / RECORD).read_text(encoding="utf-8")).get("schema")
    except (OSError, ValueError, AttributeError):
        schema = None
    require(schema in SCHEMAS, f"the target {target} is neither empty nor a vendored libsircl copy (it holds no "
                               f"{RECORD} of a libsircl snapshot schema); the sync replaces only such a directory")


def sync(snapshot, tree_digest, target=TARGET):
    """Replace ``target``'s contents with the snapshot's vendored files and the two manifest files."""
    data, entries, version = read_snapshot(snapshot, tree_digest)
    target = Path(target)
    tree = Path(snapshot) / "tree"
    files = {path: rewrite(path, (tree / path).read_bytes()) for path in sorted(entries) if excluded(path) is None}
    found = workspace_references(files)
    require(not found, "the snapshot's vendored text names directories of its workspace after the rewrites: "
            + ", ".join(found[:5]) + "; extend REWRITES or EXCLUDED")
    replaceable(target)
    if target.exists():
        for child in target.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    target.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    (target / MANIFEST).write_bytes(data)
    rewritten = {path: {"snapshot_sha256": entries[path], "sha256": sha256(content)}
                 for path, content in files.items() if sha256(content) != entries[path]}
    value = record(tree_digest, version, entries, rewritten)
    (target / RECORD).write_bytes(encoded(value))
    return check(target)


def check(target=TARGET):
    """A summary of the vendored copy after checking it against ``SNAPSHOT.sha256`` and ``SNAPSHOT.json``."""
    target = Path(target)
    data = (target / MANIFEST).read_bytes()
    entries = parse_manifest(data)
    value = json.loads((target / RECORD).read_text(encoding="utf-8"))
    tree_digest = sha256(data)
    version = (target / "VERSION").read_text(encoding="utf-8").strip() if (target / "VERSION").is_file() else None
    rewritten = value.get("rewritten") if isinstance(value, dict) else None
    require(isinstance(rewritten, dict) and all(
        path in entries and excluded(path) is None and isinstance(digests, dict)
        and set(digests) == {"snapshot_sha256", "sha256"} and digests["snapshot_sha256"] == entries[path]
        and _DIGEST.fullmatch(str(digests["sha256"])) and digests["sha256"] != entries[path]
        for path, digests in rewritten.items()),
        f"{RECORD} does not describe {MANIFEST} (tree digest {tree_digest[:12]}); run the sync again")
    require(value == record(tree_digest, version, entries, rewritten),
            f"{RECORD} does not describe {MANIFEST} (tree digest {tree_digest[:12]}); run the sync again")
    present = set()
    for path in target.rglob("*"):
        relative = path.relative_to(target)
        if relative.parts[0] in IGNORED or "__pycache__" in relative.parts or not (path.is_file() or path.is_symlink()):
            continue
        present.add(relative.as_posix())
    present -= set(OWN)
    wanted = {path for path in entries if excluded(path) is None}
    require(not (present - wanted), "the vendored copy holds files its snapshot does not vendor: "
            + ", ".join(sorted(present - wanted)[:5]))
    require(not (wanted - present), "the vendored copy lacks files of its snapshot: "
            + ", ".join(sorted(wanted - present)[:5]))
    expected = {path: rewritten[path]["sha256"] if path in rewritten else entries[path] for path in wanted}
    changed = sorted(path for path in wanted if (target / path).is_symlink()
                     or sha256((target / path).read_bytes()) != expected[path])
    require(not changed, "vendored files differ from the snapshot (update them only with the sync): "
            + ", ".join(changed[:5]))
    found = workspace_references({path: (target / path).read_bytes() for path in wanted})
    require(not found, "vendored files name directories of the snapshot's workspace: " + ", ".join(found[:5]))
    return {"snapshot": value["snapshot"], "tree_digest": tree_digest, "version": version, "files": len(wanted),
            "excluded": len(value["excluded"]["paths"]), "rewritten": len(rewritten)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    actions = parser.add_subparsers(dest="action", required=True)
    synced = actions.add_parser("sync", help="replace the vendored copy with a checked snapshot")
    synced.add_argument("snapshot", type=Path, help="the snapshot directory: tree/, FILES.sha256 and MANIFEST")
    synced.add_argument("--tree-digest", required=True, help="the SHA-256 of the snapshot's FILES.sha256")
    checked = actions.add_parser("check", help="check the vendored copy against its manifest")
    for action in (synced, checked):
        action.add_argument("--target", type=Path, default=TARGET, help="the vendored directory")
    args = parser.parse_args(argv)
    try:
        result = (sync(args.snapshot, args.tree_digest, args.target) if args.action == "sync"
                  else check(args.target))
    except (SnapshotError, OSError, ValueError, KeyError) as error:
        print(f"sync_libsircl: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
