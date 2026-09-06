"""Install four exact continuation sources after checkpoint ownership verification."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
MANIFEST_SHA256 = "ba458f84d1c07f5e070620629afc1259ca206feffb23ea9bf4f00931f79fbe7a"


def package(context=HERE):
    raw = (context / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Continuation source manifest differs")
    manifest = json.loads(raw)
    compressed = (context / "source.tar.gz").read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest["source_archive_sha256"]:
        raise ValueError("Continuation source archive differs")
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(compressed), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.name not in manifest["files"] or member.name in sources:
                raise ValueError("Continuation archive member differs")
            content = archive.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != manifest["files"][member.name]["after_sha256"]:
                raise ValueError("Continuation source payload differs")
            compile(content, member.name, "exec")
            sources[member.name] = content
    if sources.keys() != manifest["files"].keys():
        raise ValueError("Continuation source inventory differs")
    return manifest, sources


def apply(site: Path, ownership: dict, context=HERE):
    manifest, sources = package(context)
    rows = {row["path"]: row for row in ownership["files"]}
    if len(rows) != len(ownership["files"]):
        raise ValueError("Duplicate ownership dependency")
    writes = []
    for name, source in sources.items():
        expected = manifest["files"][name]
        target = site / name
        if not target.resolve().is_relative_to(site.resolve()):
            raise ValueError("Continuation source target escapes installation")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected["before_sha256"]:
            raise ValueError(f"Continuation runtime preimage differs: {name}")
        if name not in rows or rows[name]["sha256"] != expected["before_sha256"]:
            raise ValueError(f"Continuation ownership preimage differs: {name}")
        writes.append((target, source))
    # Validate all four source and ownership preimages before the first write.
    for target, source in writes:
        target.write_bytes(source)
    for name, expected in manifest["files"].items():
        rows[name]["sha256"] = expected["after_sha256"]
    return {"manifest_sha256": MANIFEST_SHA256, "files": manifest["files"]}
