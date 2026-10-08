"""Give a kraken-line installer image the vLLM and B12X Python sources that read the GLM-5.3-Flash CSF checkpoint.

The parent, `dev-20261004-kraken-cuda1342-nccl2323-status034`, installs vLLM
`d51b4181` and B12X `a850d8eb`: SparkRing's merges of Local Inference Lab's
`integration/karmic-kraken-beta` branches of 2026-10-04. The CSF checkpoint
of GLM-5.3-Flash (`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`)
stores its routed experts' block scales compressed; the `nvfp4_csf`
quantization and loader that read them are in the later merges vLLM
`bc9ea774` and B12X `cc36aa6f`. Neither merge changes a compiled extension:
every file that differs from the parent's commits under `vllm/` and `b12x/` is
a Python source, and none is deleted or renamed.

This layer replaces 47 of those sources and adds two, all in the serving
interpreter's site-packages. The source manifest
(`runtime/images/compositions/kraken-csf-sources-20261007/sources.json`)
pins each file's SHA-256 in the parent (null for an addition) and in the
merge; the build refuses a parent file or a payload file with another digest,
so the result is the merges' trees over the parent's compiled extensions. The
six B12X sources that the prepared RoCEnante transport hashes at startup are
the same in both trees and are not written, so the prepared transport, its
manifest and its receipt fields are unchanged.

The files come from a directory that holds them at their site-packages paths
(`--sources`): the read-only CSF source overlay that `build_overlay.sh` wrote
from the same payload, or the payload archive `csf_payload.tar` unpacked;
`--payload` reads the archive itself. /opt/sparkring/receipts/derived-kraken-csf-sources.json
records every file with its inherited and resulting SHA-256.

`prepare`, `record` and `build` are those of runtime/images/derived_layer.py;
`prepare` takes the source directory or archive in addition.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.images import derived_layer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "runtime/images/compositions/kraken-csf-sources-20261007/sources.json"
SCHEMA = "sparkring-source-payload/v1"
PACKAGES = ("vllm", "b12x")
PROVENANCE = "/opt/sparkring/receipts/derived-kraken-csf-sources.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def manifest(path=MANIFEST):
    """The source manifest after checking that it pins only site-packages Python sources of vLLM and B12X."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    require(record.get("schema") == SCHEMA, f"{path} is not a {SCHEMA} manifest")
    require(record.get("site_packages") == derived_layer.SITE,
            "The manifest's site-packages directory differs from the derived layer's")
    require(set(record.get("sources", {})) == set(PACKAGES), "The manifest names the vLLM and B12X merge commits")
    rows = record.get("files")
    require(isinstance(rows, list) and rows, "The manifest lists no files")
    paths = [row.get("path") for row in rows]
    require(paths == sorted(set(paths)), "The manifest lists each file once, sorted")
    for row in rows:
        require(set(row) == {"path", "package", "inherited_sha256", "sha256", "bytes"},
                "Each file records path, package, inherited_sha256, sha256 and bytes")
        relative = PurePosixPath(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts and relative.parts[0] == row["package"]
                and row["package"] in PACKAGES and relative.suffix == ".py",
                "A source file is a Python file of the vllm or b12x package: " + row["path"])
        target = derived_layer.SITE + row["path"]
        require(target not in derived_layer.TRANSPORT_SOURCES,
                "The prepared transport verifies this B12X source; the layer must not write it: " + row["path"])
        require(row["inherited_sha256"] is None or _SHA256.fullmatch(row["inherited_sha256"] or ""),
                "inherited_sha256 is a SHA-256 or null: " + row["path"])
        require(isinstance(row["sha256"], str) and _SHA256.fullmatch(row["sha256"]) and row["sha256"]
                != row["inherited_sha256"], "sha256 is the merged file's SHA-256, differing from the parent's: "
                + row["path"])
        require(type(row["bytes"]) is int and row["bytes"] > 0, "bytes is the merged file's size: " + row["path"])
    for row in record.get("transport_sources", []):
        require(derived_layer.SITE + row["path"] in derived_layer.TRANSPORT_SOURCES,
                "transport_sources lists the B12X sources the prepared transport verifies: " + row["path"])
    return record


def pins(record):
    """``{image path: (inherited SHA-256, resulting SHA-256)}`` for ``derived_layer.Layer``."""
    return {derived_layer.SITE + row["path"]: (row["inherited_sha256"], row["sha256"]) for row in record["files"]}


# Readers of the merged files, keyed by path relative to site-packages.

def directory_reader(root):
    """Files under ``root`` at their site-packages paths, such as the CSF source overlay."""
    root = Path(root)

    def read(relative):
        path = root.joinpath(*PurePosixPath(relative).parts)
        require(path.is_file() and not path.is_symlink(), f"{relative} is not a regular file under {root}")
        return path.read_bytes()
    return read


def payload_reader(archive, record):
    """Files of the payload archive after checking its SHA-256 and that it holds exactly the manifest's files."""
    data = Path(archive).read_bytes()
    expected = record["payload"]["sha256"]
    require(hashlib.sha256(data).hexdigest() == expected, f"{archive} is not the payload {expected[:12]}")
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
        for member in bundle.getmembers():
            require(member.isfile(), "The payload holds a member that is not a regular file: " + member.name)
            files[member.name] = bundle.extractfile(member).read()
    require(sorted(files) == [row["path"] for row in record["files"]], "The payload's files differ from the manifest's")

    def read(relative):
        return files[relative]
    return read


def layer(read_source, record=None):
    """The ``derived_layer.Layer`` whose files ``read_source`` supplies, each pinned by the manifest."""
    record = manifest() if record is None else record
    commits = record["sources"]

    def replace(read, receipt):
        require(read_source is not None, "Preparing this layer needs --sources or --payload")
        written = {}
        for row in record["files"]:
            data = read_source(row["path"])
            require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
                    f"{row['path']} differs from the merge's file; the source directory is not the CSF payload")
            written[derived_layer.SITE + row["path"]] = data
        return written

    return derived_layer.Layer(
        name="kraken-csf-sources",
        purpose=("vLLM and B12X Python sources of SparkRing's merges vLLM " + commits["vllm"]["commit"][:12]
                 + " and B12X " + commits["b12x"]["commit"][:12] + " of Local Inference Lab's "
                 "integration/karmic-kraken-beta, which read the GLM-5.3-Flash CSF checkpoint (nvfp4_csf), over "
                 "the parent's compiled extensions; source manifest "
                 "runtime/images/compositions/kraken-csf-sources-20261007/sources.json"),
        replace=replace,
        provenance=PROVENANCE,
        pins=pins(record),
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] != ["prepare"]:
        # record and build read the prepared context's plan only.
        return derived_layer.main(layer(None), argv)
    parser = argparse.ArgumentParser(prog="derive_kraken_csf_sources.py prepare",
                                     description="Write the layer's build context; does not build.")
    parser.add_argument("--parent-lock", required=True, type=Path, help="installer lock of the parent image")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sources", type=Path,
                        help="directory holding the merged files at their site-packages paths, such as the CSF "
                             "source overlay")
    source.add_argument("--payload", type=Path, help="the payload archive csf_payload.tar")
    parser.add_argument("--base-receipt", type=Path, help="the parent's " + derived_layer.BASE_RECEIPT)
    parser.add_argument("--toolchain-receipt", type=Path, help="the parent's " + derived_layer.TOOLCHAIN_RECEIPT)
    parser.add_argument("--parent-root", type=Path,
                        help="exported parent root filesystem; default: read from the local parent image")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv[1:])
    try:
        require(bool(args.base_receipt) == bool(args.toolchain_receipt), "Supply both copied receipts or neither")
        record = manifest()
        lock = derived_layer.load_lock(args.parent_lock)
        require(lock["name"] == record["parent_release"] and lock["image_id"] == record["parent_image_id"],
                f"This layer derives from {record['parent_release']}, not {lock['name']}")
        read_source = (directory_reader(args.sources) if args.sources is not None
                       else payload_reader(args.payload, record))
        fallback = (derived_layer.root_reader(args.parent_root) if args.parent_root
                    else derived_layer.docker_reader(lock["image_id"]))
        read = (derived_layer.receipt_reader(args.base_receipt, args.toolchain_receipt, fallback)
                if args.base_receipt else fallback)
        result = derived_layer.prepare_layer(layer(read_source, record), lock, read, args.output)
    except (ValueError, OSError, KeyError, tarfile.TarError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
