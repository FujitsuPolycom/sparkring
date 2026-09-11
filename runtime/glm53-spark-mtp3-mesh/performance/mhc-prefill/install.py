"""Install exact mHC sources after validating runtime and ownership preimages."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

HERE = Path(__file__).resolve().parent
MANIFEST_SHA256 = "823c7b2d239a564bbbf4e2923a00e6f0c65558fbd9d44b93315098e9aeec4044"


def package(context=HERE, *, preimages=False):
    raw = (context / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("mHC source manifest differs")
    manifest = json.loads(raw)
    archive_name = "preimages.tar.gz" if preimages else "source.tar.gz"
    archive_hash = "preimage_archive_sha256" if preimages else "source_archive_sha256"
    source_hash = "before_sha256" if preimages else "after_sha256"
    expected = {
        name: row[source_hash]
        for name, row in manifest["files"].items()
        if row[source_hash] is not None
    }
    compressed = (context / archive_name).read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest[archive_hash]:
        raise ValueError("mHC source archive differs")
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(compressed), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or member.name not in expected
                or member.name in sources
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
            ):
                raise ValueError("mHC archive member differs")
            content = archive.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != expected[member.name]:
                raise ValueError("mHC source payload differs")
            compile(content, member.name, "exec")
            sources[member.name] = content
    if sources.keys() != expected.keys():
        raise ValueError("mHC source inventory differs")
    return manifest, sources


def apply(site: Path, ownership: dict, context=HERE):
    manifest, sources = package(context)
    rows = {row["path"]: row for row in ownership["files"]}
    if len(rows) != len(ownership["files"]):
        raise ValueError("Duplicate ownership dependency")
    # GDN attention exports recurrent checkpoint state. Its installed source
    # must match the checkpoint ownership contract before mHC changes it.
    checkpoint = "vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py"
    if checkpoint not in rows:
        raise ValueError("Missing mHC checkpoint ownership dependency")
    writes = []
    for name, source in sources.items():
        expected = manifest["files"][name]
        target = site / name
        if not target.resolve().is_relative_to(site.resolve()):
            raise ValueError("mHC source target escapes installation")
        before = expected["before_sha256"]
        if before is None:
            if target.exists() or name in rows:
                raise ValueError(f"mHC new source already exists: {name}")
        elif (
            not target.is_file()
            or hashlib.sha256(target.read_bytes()).hexdigest() != before
        ):
            raise ValueError(f"mHC runtime preimage differs: {name}")
        if name in rows and rows[name]["sha256"] != before:
            raise ValueError(f"mHC ownership preimage differs: {name}")
        writes.append((target, source))
    # Every source and overlapping ownership preimage passes before any write.
    for target, source in writes:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    for name, expected in manifest["files"].items():
        if hashlib.sha256((site / name).read_bytes()).hexdigest() != expected["after_sha256"]:
            raise ValueError(f"mHC runtime postimage differs: {name}")
    for name, expected in manifest["files"].items():
        if name in rows:
            rows[name]["sha256"] = expected["after_sha256"]
    return {"manifest_sha256": MANIFEST_SHA256, "files": manifest["files"]}
