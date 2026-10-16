"""Prepare the pinned SparkCache source layer using authenticated parent receipts.

No image or hardware access occurs here. The shared derived-layer builder
builds and admits the context separately, including the inherited native files.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.images import derived_layer as layer

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "runtime/images/compositions/installer-sparkcache-8abbe9da67b8/sources.json"
PACKAGE = layer.SITE + "sparkcache/"


def resolve_descriptor(record, base, repository):
    """Resolve inherited hashes only from a receipt authenticated by the parent lock.

    No inherited SparkCache commit is claimed: the parent lock pins the exact
    receipt, which pins its installed bytes. Unknown runtime modules require
    explicit review rather than silently retaining a mixed package.
    """
    layer.require(record.get("schema") == "sparkring-sparkcache-layer/v1", "Unsupported SparkCache sources")
    for target, sha in record["native_libraries"].items():
        layer.require(base["files"].get(target) == sha,
                      "Parent SparkCache native library differs; review a native rebuild: " + target)
    for target in base["files"]:
        relative = target.removeprefix(PACKAGE)
        if (target.startswith(PACKAGE) and target.endswith(".py")
                and not PurePosixPath(relative).name.startswith("test_")
                and "tests" not in PurePosixPath(relative).parts
                and "native" not in PurePosixPath(relative).parts):
            layer.require(target in record["files"], "Parent has an unreviewed SparkCache module: " + target)
    files = {}
    for target, row in record["files"].items():
        layer.require(target.startswith(PACKAGE), "Source layer may only replace SparkCache files")
        layer._python_file(target)
        source = repository / row["source"]
        layer.require(not source.is_symlink() and source.resolve().is_relative_to(repository),
                      "SparkCache source escapes the repository")
        layer.pinned_bytes(source.read_bytes(), row["sha256"])
        files[target] = {**row, "inherited_sha256": base["files"].get(target)}
    return {"schema": layer.SCHEMA, "id": record["id"], "purpose": record["purpose"],
            "parent_lock": record["parent_lock"], "provenance": record["provenance"], "files": files}


def prepare(base_receipt, toolchain_receipt, output, *, sources=SOURCES, repository=ROOT):
    repository = Path(repository).resolve()
    sources = Path(sources)
    record = json.loads(sources.read_text(encoding="utf-8"))
    parent = repository / record["parent_lock"]
    layer.require(parent.resolve().is_relative_to(repository), "Parent lock escapes the repository")
    layer.pinned_bytes(parent.read_bytes(), record["parent_lock_sha256"])
    lock = layer.load_lock(parent)
    read = layer.receipt_reader(base_receipt, toolchain_receipt)
    base_raw = read(layer.BASE_RECEIPT)
    layer.require(layer.digest(base_raw) == lock["parent_receipt_sha256"],
                  "External-base receipt differs from the pinned 2026.10.4 image")
    descriptor = resolve_descriptor(record, json.loads(base_raw), repository)
    # Install the source inventory so the resulting receipt binds commit, tree,
    # native-library expectations and all resulting source digests.
    commit = record["source"]["commit"]
    layer.require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
                  "SparkCache source commit must be an immutable Git revision")
    source_target = PACKAGE + f"runtime_patches/sparkring-source-{commit[:12]}.json"
    layer.require(source_target not in json.loads(base_raw)["files"], "Source inventory already exists")
    descriptor["files"][source_target] = {
        "source": sources.resolve().relative_to(repository).as_posix(),
        "sha256": layer.digest(sources.read_bytes()), "inherited_sha256": None}
    result = layer.prepare_layer(descriptor, lock, read, output, identity={"repository": repository},
                                 descriptor_sha256=layer.digest(layer.canonical_json(descriptor)))
    (Path(output) / "descriptor.json").write_bytes(layer.canonical_json(descriptor))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--toolchain-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.base_receipt, args.toolchain_receipt, args.output, sources=args.sources)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
