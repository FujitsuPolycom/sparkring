"""Build-time only: verify preimages and payloads, apply files, report postimage hashes."""
from __future__ import annotations
import ast
import hashlib
import json
import os
import py_compile
import re
from pathlib import Path
import stat

def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_files(records, *, root=Path("/")):
    for absolute, expected in records.items():
        path = root / absolute.lstrip("/")
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise RuntimeError(f"Symlink target refused: {absolute}")
        if expected is None:
            if path.exists():
                raise RuntimeError(f"Expected absent file: {absolute}")
        elif not path.is_file() or sha(path.read_bytes()) != expected:
            raise RuntimeError(f"File identity mismatch: {absolute}")


def verify_symbols(data, symbols):
    tree = ast.parse(data)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    top = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for symbol in symbols:
        if "." not in symbol:
            if symbol not in top:
                raise RuntimeError(f"Required top-level symbol missing: {symbol}")
            continue
        cls, member = symbol.split(".", 1)
        members = set()
        for node in classes.get(cls, ast.ClassDef(name="", bases=[], keywords=[], body=[], decorator_list=[])).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                members.add(node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                members.add(node.target.id)
            elif isinstance(node, ast.Assign):
                members.update(t.id for t in node.targets if isinstance(t, ast.Name))
        if member not in members:
            raise RuntimeError(f"Required class member missing: {symbol}")


def apply_manifest(expected_manifest_sha256):
    context = Path(__file__).parent
    encoded = (context / "patch-manifest.json").read_bytes()
    if sha(encoded) != expected_manifest_sha256:
        raise RuntimeError("Frozen build manifest hash mismatch")
    manifest = json.loads(encoded)
    verify_files(manifest["preimages"])
    payload = {}
    for target, record in manifest["replacements"].items():
        source = context / record["payload"]
        data = source.read_bytes()
        if sha(data) != record["sha256"]:
            raise RuntimeError(f"Frozen payload mismatch: {target}")
        if target.endswith(".py"):
            compile(data, target, "exec")
            verify_symbols(data, record.get("required_symbols", []))
        payload[target] = data
    # Validate every input and replacement before modifying the image.
    compiled_caches = set()
    for target, data in payload.items():
        path = Path(target)
        previous = path.stat() if path.exists() else None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.chmod(path, stat.S_IMODE(previous.st_mode) if previous else 0o644)
        if previous:
            os.chown(path, previous.st_uid, previous.st_gid)
        if path.suffix == ".py":
            cache = path.parent / "__pycache__"
            if cache.is_dir():
                for pyc in cache.glob(path.stem + ".*.pyc"):
                    if pyc.is_file() and not pyc.is_symlink():
                        match = re.search(r"\.opt-(\d+)\.pyc$", pyc.name)
                        py_compile.compile(str(path), cfile=str(pyc), doraise=True,
                                           optimize=int(match.group(1)) if match else 0,
                                           invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
                        compiled_caches.add(pyc)
    verify_files({name: row["sha256"] for name, row in manifest["replacements"].items()})
    verify_files({name: expected for name, expected in manifest["preimages"].items()
                  if name not in manifest["replacements"]})
    postimages = {name: row["sha256"] for name, row in manifest["replacements"].items()}
    # Make the filesystem diff independent of host build time. This touches
    # only timestamps, including directories affected by file/bytecode updates.
    touched = {Path(name) for name in payload} | compiled_caches
    parents = set()
    for path in touched:
        os.utime(path, ns=(0, 0))
        parents.update(path.parents)
        cache = path.parent / "__pycache__"
        if cache.is_dir():
            parents.add(cache)
    for path in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        os.utime(path, ns=(0, 0))
    print(json.dumps({"patch_manifest_sha256": sha(encoded), "files_replaced": len(payload),
                      "postimages": postimages, "attestation_regeneration_required": True}))
    return postimages


def apply(site_packages: Path) -> dict[str, str]:
    """Apply the pinned image layout and return replacement file SHA-256 digests."""
    if site_packages != Path("/usr/local/lib/python3.12/dist-packages"):
        raise ValueError("Checkpoint payload requires /usr/local/lib/python3.12/dist-packages")
    return apply_manifest("0970d29ec33e9f8525a2cc55989ab0deb937bff88b5f16d4d5280035359e4c55")



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate or install the explicit recurrent checkpoint payload.")
    parser.add_argument("command", choices=("verify-context", "verify-preimages", "apply"))
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    context = Path(__file__).resolve().parent
    encoded = (context / "patch-manifest.json").read_bytes()
    if sha(encoded) != args.manifest_sha256:
        raise RuntimeError("Manifest identity mismatch")
    manifest = json.loads(encoded)
    for target, record in manifest["replacements"].items():
        source = (context / record["payload"]).resolve()
        if not source.is_relative_to(context) or sha(source.read_bytes()) != record["sha256"]:
            raise RuntimeError(f"Payload identity mismatch: {target}")
        if target.endswith(".py"):
            compile(source.read_bytes(), target, "exec")
            verify_symbols(source.read_bytes(), record.get("required_symbols", []))
    if args.command == "verify-preimages":
        verify_files(manifest["preimages"])
    if args.command == "apply":
        apply_manifest(args.manifest_sha256)
    else:
        print(json.dumps({"manifest_sha256": args.manifest_sha256,
                          "payloads_verified": len(manifest["replacements"])}))


if __name__ == "__main__":
    main()
