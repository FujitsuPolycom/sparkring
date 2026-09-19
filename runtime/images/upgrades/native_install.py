"""Install source-built wheels while preserving the foundation's other payload.

The descriptor and compiler receipt are trusted build inputs. Installation does
not infer serving qualification or rewrite feature/cache compatibility contracts.
"""

from __future__ import annotations

import argparse
import base64
import csv
import configparser
import copy
import hashlib
import importlib.metadata as metadata
import json
import io
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import re
import zipfile

SITE = Path("/opt/venv/lib/python3.12/site-packages")
ROOT = Path("/opt/sparkring")
RECEIPT = ROOT / "receipts/candidate-installed.json"
NATIVE = ROOT / "receipts/native-installed.json"
ENTRYPOINT = ROOT / "bin/native-image.py"
DEPENDENCY_PACKAGES = {
    "flashinfer-python": "flashinfer",
    "flashinfer-jit-cache": "flashinfer_jit_cache",
}
# These distributions share the cuda namespace. They must never be treated as
# owning the entire cuda directory, including cuda-pathfinder's sibling files.
CUDA_RUNTIME_VERSIONS = {
    "cuda-python": "13.3.1",
    "cuda-bindings": "13.3.1",
    "cuda-core": "1.0.1",
}
CUDA_PROTECTED_VERSIONS = {
    "cuda-pathfinder": "1.8.1",
    "nvidia-cutlass-dsl": "4.6.2",
    "torch": "2.13.0",
}
SGLANG_PREFIX = Path("/opt/sglang")
FLASHINFER_GLOBAL_BUILD_HELPERS = ("build_backend.py", "build_utils.py")
PREPARED_TRANSPORT_PROFILE = "tp2-rocenante-adaptive-prepared"
PARENT_RECEIPTS = {
    "native": "/opt/sparkring/receipts/native-installed.json",
    "candidate": "/opt/sparkring/receipts/candidate-installed.json",
}


def normalize_parent_receipt(raw, kind):
    """Expose verified native contract selections without rewriting the receipt."""
    require(kind in PARENT_RECEIPTS, "Unknown foundation receipt kind")
    parent = json.loads(raw)
    require(
        isinstance(parent, dict)
        and parent.get("schema") == f"sparkring-{kind}-installed/v1"
        and isinstance(parent.get("files"), dict) and parent["files"]
        and isinstance(parent.get("versions"), dict) and parent["versions"],
        "Invalid foundation installed receipt",
    )
    require(isinstance(parent.get("removed_files", []), list), "Invalid removed-file inventory")
    if kind == "native":
        require("removed_files" in parent, "Native foundation lacks removed-file inventory")
        active = parent.get("active_contracts")
        require(isinstance(active, list) and all(isinstance(name, str) for name in active),
                "Native foundation lacks declared active contracts")
        require(len(active) == len(set(active)), "Duplicate native active contract")
        contracts = {}
        for name in active:
            path = PurePosixPath(name)
            digest = parent["files"].get(name)
            require(
                path.parent == PurePosixPath("/opt/sparkring/contracts")
                and str(path) == name and "\\" not in name and ".." not in path.parts
                and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
                "Native active contract is not owned with a valid hash: " + name,
            )
            contracts[name] = {"sha256": digest}
        require("boundary_runtime" in parent, "Native foundation lacks boundary selection")
        parent["integration_contracts"] = contracts
    else:
        require(isinstance(parent.get("integration_contracts", {}), dict),
                "Invalid candidate contract selections")
    return parent


def read_parent_foundation(descriptor=None):
    """A present native receipt is authoritative, including when it is invalid."""
    kind, path = ("native", NATIVE) if os.path.lexists(NATIVE) else ("candidate", RECEIPT)
    require(path.is_file() and not path.is_symlink(), "Invalid foundation receipt path")
    raw = path.read_bytes()
    evidence = dict(kind=kind, path=str(path), sha256=hashlib.sha256(raw).hexdigest())
    if descriptor is not None:
        declared = descriptor.get("parent_receipt")
        require(
            (declared == evidence if declared is not None else kind == "candidate")
            and descriptor.get("parent_installed_sha256") == evidence["sha256"],
            "Foundation installed receipt selection differs",
        )
    parent = normalize_parent_receipt(raw, kind)
    verify_installed_state(parent)
    return parent, raw, evidence


def retain_parent_receipt(raw, evidence):
    """Keep the exact parent attestation when native-installed.json is replaced."""
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == evidence["sha256"], "Retained parent receipt bytes differ")
    archive = ROOT / "receipts" / ("foundation-" + digest + ".json")
    if os.path.lexists(archive):
        require(archive.is_file() and not archive.is_symlink() and sha(archive) == digest,
                "Retained parent receipt destination differs")
    else:
        with archive.open("xb") as stream:
            stream.write(raw)
    return {str(archive): digest}, {**evidence, "retained_path": str(archive)}


def feature_asset_scope(name):
    """Classify only explicitly owned image-extension paths."""
    path = PurePosixPath(name)
    require(
        path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and str(path) == name,
        "Invalid image-extension asset path",
    )
    owner = PurePosixPath("/opt/sparkring")
    if (
        any(
            path != root and path.is_relative_to(root)
            for root in (owner / "features", owner / "qwen4-prefill")
        )
        or name == "/opt/venv/lib/python3.12/site-packages/sparkring_features.pth"
    ):
        return "feature"
    bundle = owner / "transports" / PREPARED_TRANSPORT_PROFILE
    if (
        (path != bundle and path.is_relative_to(bundle))
        or path == owner / "transports/sparkring_transport_selector.py"
        or name == "/opt/venv/lib/python3.12/site-packages/sparkring_transport.pth"
    ):
        return "transport"
    if path == owner / "licenses/components.md" or (
        path.parent == owner / "releases/shared"
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.json", path.name)
    ):
        return "fresh-metadata"
    raise ValueError("Image-extension asset is outside its reviewed owner: " + name)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(files):
    for name, expected in files.items():
        path = Path(name)
        require(
            path.is_absolute()
            and ".." not in path.parts
            and path.is_file()
            and sha(path) == expected,
            "Installed payload differs: " + name,
        )


def distribution_files(name):
    result = {}
    distribution = metadata.distribution(name)
    for entry in distribution.files or []:
        path = Path(distribution.locate_file(entry)).resolve()
        require(
            path.is_relative_to(SITE.parent.parent.parent),
            "Distribution RECORD escapes the virtual environment",
        )
        if path.is_file() and path.suffix not in (".pyc", ".pyo"):
            result[str(path)] = sha(path)
    require(result, "Distribution has no installed file inventory: " + name)
    return result


def selected_distribution_files(name):
    try:
        return distribution_files(name)
    except metadata.PackageNotFoundError:
        if name not in CUDA_RUNTIME_VERSIONS:
            raise
        return {}


def validate_runtime_dependencies(dependencies):
    require(set(dependencies) <= set(DEPENDENCY_PACKAGES) | set(CUDA_RUNTIME_VERSIONS),
            "Unreviewed runtime dependency migration")
    selected = set(dependencies) & set(CUDA_RUNTIME_VERSIONS)
    if selected:
        require(selected == set(CUDA_RUNTIME_VERSIONS)
                and all(dependencies[name].get("version") == version
                        for name, version in CUDA_RUNTIME_VERSIONS.items()),
                "CUDA runtime migration requires the complete reviewed version tuple")


def cuda_protected_inputs(dependencies):
    if not set(dependencies) & set(CUDA_RUNTIME_VERSIONS):
        return {}, {}
    versions = {name: metadata.version(name) for name in CUDA_PROTECTED_VERSIONS}
    require(versions == CUDA_PROTECTED_VERSIONS, "CUDA migration protected versions differ")
    files = {}
    for name in ("cuda-pathfinder", "nvidia-cutlass-dsl"):
        files.update(distribution_files(name))
    # Inventory siblings even if the old foundation omitted their RECORDs.
    # Only selected exact RECORD/wheel paths may change during installation.
    namespace = SITE / "cuda"
    require(not namespace.is_symlink(), "CUDA namespace is a symlink")
    for path in namespace.rglob("*"):
        require(not path.is_symlink(), "CUDA namespace contains a symlink")
        if path.is_file() and path.suffix not in (".pyc", ".pyo"):
            files[str(path)] = sha(path)
    return versions, files


def cuda_wheel_paths(name, wheel, version):
    require(name in CUDA_RUNTIME_VERSIONS and version == CUDA_RUNTIME_VERSIONS[name],
            "Unreviewed CUDA runtime wheel version")
    paths, roots = wheel_install_paths({name: wheel})
    owner = name.replace("-", "_") + "-" + version + ".dist-info"
    require(roots == [SITE / owner], "CUDA wheel metadata namespace differs")
    scope = {"cuda-bindings": ("cuda", "bindings"), "cuda-core": ("cuda", "core")}.get(name)
    for path in paths:
        file = Path(path)
        require(file.is_relative_to(SITE), "CUDA wheel escapes its namespace")
        parts = file.relative_to(SITE).parts
        require(parts[0] == owner or (scope is not None and parts[:2] == scope and len(parts) > 2),
                "CUDA wheel modifies an unreviewed namespace: " + str(path))
    return paths


def cuda_record_paths(name, files):
    """An old RECORD cannot grant blanket ownership outside the same namespace."""
    scope = {"cuda-bindings": ("cuda", "bindings"), "cuda-core": ("cuda", "core")}.get(name)
    for path in files:
        file = Path(path)
        require(file.is_relative_to(SITE), "CUDA RECORD escapes its namespace")
        parts = file.relative_to(SITE).parts
        own_metadata = (parts[0].startswith(name.replace("-", "_") + "-")
                        and parts[0].endswith(".dist-info"))
        require(own_metadata or (scope is not None and parts[:2] == scope and len(parts) > 2),
                "CUDA RECORD modifies an unreviewed namespace: " + str(path))
    return set(files)


def package_owned(path, package_names, metadata_roots, exact_paths=()):
    path = Path(path)
    if str(path) in exact_paths:
        return True
    if any(path.is_relative_to(SITE / name) for name in package_names):
        return True
    if any(path.is_relative_to(root) for root in metadata_roots):
        return True
    # These exact console entrypoints belong to the selected distributions.
    # B12X's policy-based CLI is removed by the preparation-based source package;
    # its removal must be tracked as owned, without admitting other venv programs.
    scripts = {
        "vllm": {"vllm"},
        "flashinfer": {"flashinfer"},
        "b12x": {"b12x-generate-gpu-profile", "b12x-inspect-model-policy"},
    }
    return path in {
        Path("/opt/venv/bin") / entrypoint
        for name in package_names
        for entrypoint in scripts.get(name, ())
    }


def distribution_ownership(paths):
    """Read every installed distribution's RECORD claims for selected paths."""
    paths = set(map(str, paths))
    result = {path: [] for path in paths}
    for distribution in metadata.distributions():
        owner = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        for entry in distribution.files or []:
            if str(entry).endswith((".pyc", ".pyo")):
                continue
            path = str(Path(distribution.locate_file(entry)).resolve())
            if path not in paths:
                continue
            digest = entry.hash
            expected = None
            if digest is not None and digest.mode == "sha256":
                expected = base64.urlsafe_b64decode(
                    digest.value + "=" * (-len(digest.value) % 4)
                ).hex()
            result[path].append(
                {
                    "distribution": owner,
                    "version": distribution.version,
                    "record_hash_mode": digest.mode if digest else None,
                    "record_hash_value": digest.value if digest else None,
                    "record_sha256": expected,
                }
            )
    return result


def wheel_install_paths(wheels):
    """Resolve reviewed wheel payload and console-script destinations for audit."""
    result, metadata_roots = set(), []
    for wheel in wheels.values():
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            records = [name for name in names if name.endswith(".dist-info/METADATA")]
            require(len(records) == 1, "Wheel lacks one metadata owner")
            metadata_root = records[0].split("/")[0]
            metadata_roots.append(SITE / metadata_root)
            destinations = set()
            for name in names:
                if name.endswith("/"):
                    continue
                parts = PurePosixPath(name).parts
                require(
                    parts
                    and not PurePosixPath(name).is_absolute()
                    and ".." not in parts
                    and "\\" not in name,
                    "Wheel destination escapes site-packages",
                )
                if parts[0].endswith(".data"):
                    require(
                        len(parts) > 2 and parts[1] in ("purelib", "platlib"),
                        "Unreviewed wheel relocation",
                    )
                    parts = parts[2:]
                destination = str(SITE.joinpath(*parts))
                require(
                    destination not in destinations,
                    "Duplicate relocated wheel destination",
                )
                destinations.add(destination)
            entrypoints = metadata_root + "/entry_points.txt"
            if entrypoints in names:
                config = configparser.ConfigParser(interpolation=None)
                config.optionxform = str
                config.read_string(archive.read(entrypoints).decode())
                for name in (
                    config["console_scripts"] if "console_scripts" in config else ()
                ):
                    require(
                        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name),
                        "Invalid wheel console-script path",
                    )
                    destinations.add(str(Path("/opt/venv/bin") / name))
            result.update(destinations)
    return result, metadata_roots


def audit_selected_ownership(
    selected_files, package_names, metadata_roots, parent_files, planned_paths=(), exact_paths=()
):
    """Refuse unexplained RECORD extras/collisions before pip changes any files."""
    paths = {path for files in selected_files.values() for path in files}
    owners = distribution_ownership(paths | set(planned_paths))
    selected_names = set(selected_files)
    helpers = {str(SITE / name) for name in FLASHINFER_GLOBAL_BUILD_HELPERS}
    backups, extras, conflicts, unowned = {}, {}, {}, []
    for name in sorted(paths):
        foreign = [
            item for item in owners[name] if item["distribution"] not in selected_names
        ]
        extra = not package_owned(name, package_names, metadata_roots, exact_paths)
        if extra:
            extras[name] = owners[name]
        if foreign:
            conflicts[name] = owners[name]
        if name in helpers and name in selected_files.get("flashinfer-python", {}):
            path = Path(name)
            data = path.read_bytes()
            require(
                not path.is_symlink() and parent_files.get(name) == sha(path),
                "Build helper lacks verified inherited ownership: " + name,
            )
            for owner in owners[name]:
                expected = owner["record_sha256"]
                owner["record_matches_baseline_bytes"] = (
                    None if expected is None else expected == sha(path)
                )
            backups[name] = {
                "bytes": data,
                "mode": path.stat().st_mode & 0o777,
                "sha256": sha(path),
                "owners": owners[name],
            }
        elif extra or foreign:
            unowned.append(
                {"path": name, "unexplained_extra": extra, "owners": owners[name]}
            )
    for name in sorted(set(planned_paths)):
        unknown_existing = name in exact_paths and Path(name).exists() and name not in paths
        if (
            not package_owned(name, package_names, metadata_roots, exact_paths)
            or unknown_existing
            or any(item["distribution"] not in selected_names for item in owners[name])
        ):
            unowned.append(
                {
                    "path": name,
                    "candidate_wheel_destination": True,
                    "owners": owners[name],
                }
            )
    require(
        not unowned,
        "Selected distributions have unreviewed ownership conflicts: "
        + json.dumps(unowned),
    )
    return {
        "selected_record_files": {
            name: len(files) for name, files in selected_files.items()
        },
        "extras": extras,
        "collisions": conflicts,
        "preserved_helpers": {
            name: {key: value for key, value in item.items() if key != "bytes"}
            for name, item in backups.items()
        },
    }, backups


def isolate_flashinfer_build_helpers(wheel, directory, version):
    """Remove only redundant root build helpers from the reviewed runtime wheel."""
    require(version == "0.6.18.post1", "Unreviewed FlashInfer helper-isolation version")
    wheel = Path(wheel)
    directory.mkdir(exist_ok=True)
    destination = directory / wheel.name
    require(not destination.exists(), "Normalized installation wheel already exists")
    removed = {}
    with zipfile.ZipFile(wheel) as source:
        names = source.namelist()
        require(len(names) == len(set(names)), "Duplicate runtime wheel paths")
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        require(len(record_names) == 1, "Runtime wheel must have exactly one RECORD")
        for name in FLASHINFER_GLOBAL_BUILD_HELPERS:
            duplicate = "flashinfer/data/" + name
            require(
                name in names
                and duplicate in names
                and source.read(name) == source.read(duplicate),
                "FlashInfer build helper is not an identical packaged-data duplicate: "
                + name,
            )
            removed[name] = {
                "sha256": hashlib.sha256(source.read(name)).hexdigest(),
                "retained_copy": duplicate,
            }
        record_name = record_names[0]
        rows = list(csv.reader(io.StringIO(source.read(record_name).decode())))
        require(
            {name for name, *_ in rows if name in removed} == set(removed),
            "Build helpers are absent from the publisher RECORD",
        )
        record = io.StringIO(newline="")
        csv.writer(record, lineterminator="\n").writerows(
            row for row in rows if row[0] not in removed
        )
        with zipfile.ZipFile(destination, "w") as output:
            for item in source.infolist():
                if item.filename in removed:
                    continue
                if item.filename == record_name:
                    output.writestr(copy.copy(item), record.getvalue())
                    continue
                with (
                    source.open(item) as original,
                    output.open(copy.copy(item), "w") as target,
                ):
                    while block := original.read(1024 * 1024):
                        target.write(block)
        with zipfile.ZipFile(destination) as output:
            require(
                set(output.namelist()) == set(names) - set(removed),
                "Installation wheel membership differs",
            )
            for name in output.namelist():
                if name == record_name:
                    continue
                with source.open(name) as original, output.open(name) as installed:
                    require(
                        hashlib.file_digest(original, "sha256").digest()
                        == hashlib.file_digest(installed, "sha256").digest(),
                        "Helper isolation changed runtime/package payload: " + name,
                    )
    return destination, {
        "profile": "flashinfer-0.6.18.post1-build-helper-isolation",
        "publisher_wheel_sha256": sha(wheel),
        "installation_wheel_sha256": sha(destination),
        "removed_root_helpers": removed,
        "other_payload_hashes_unchanged": True,
    }


def restore_build_helpers(backups):
    for name in (str(SITE / item) for item in FLASHINFER_GLOBAL_BUILD_HELPERS):
        require(
            not Path(name).exists() and not Path(name).is_symlink(),
            "Filtered build helper unexpectedly exists after pip: " + name,
        )
    for name, item in backups.items():
        path = Path(name)
        with path.open("xb") as stream:
            stream.write(item["bytes"])
        path.chmod(item["mode"])
        require(
            sha(path) == item["sha256"],
            "Restored inherited build helper differs: " + name,
        )


def isolated_sglang_inventory():
    """Verify the separate runtime receipt and inventory its Python environment."""
    directory = ROOT / "sglang"
    if not directory.exists():
        return None
    receipt = directory / "installed.json"
    manifest = directory / "manifest.json"
    record = read(receipt)
    require(
        record.get("schema") == "sparkring-sglang-installed/v1"
        and record.get("files")
        and record["composition_sha256"] == sha(manifest),
        "Isolated SGLang receipt or manifest differs",
    )
    require(
        record["vllm_parent_receipt_sha256"] == sha(RECEIPT),
        "Isolated SGLang parent receipt differs",
    )
    files = dict(record["files"])
    files.update({str(receipt): sha(receipt), str(manifest): sha(manifest)})
    files[str(ROOT / "bin/sglang-python")] = sha(ROOT / "bin/sglang-python")
    prefix = Path(read(manifest)["sglang_base"]["python_prefix"])
    require(
        prefix == SGLANG_PREFIX and prefix.is_dir(),
        "Isolated SGLang Python prefix differs",
    )
    for path in prefix.rglob("*"):
        if path.is_file() and path.suffix not in (".pyc", ".pyo"):
            files[str(path)] = sha(path)
    verify_files(files)
    return {
        "receipt_sha256": sha(receipt),
        "composition_sha256": sha(manifest),
        "files": files,
        "serving_qualified": False,
    }


def verified_inherited_feature_update(parent):
    """Retain migration provenance only while its owned catalogs/assets match.

    This preserves the original record, including historical API preimages;
    it does not rebind those preimages or qualify them against replacement wheels.
    """
    update = parent.get("feature_update")
    if update is None:
        return None
    require(
        isinstance(update, dict) and update.get("serving_qualified") is False
        and isinstance(update.get("descriptor_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", update["descriptor_sha256"])
        and isinstance(update.get("assets"), dict) and update["assets"],
        "Invalid inherited feature-update record",
    )

    def owned(name, expected):
        require(isinstance(name, str) and isinstance(expected, str)
                and re.fullmatch(r"[0-9a-f]{64}", expected),
                "Invalid inherited feature evidence")
        path = Path(name)
        require(
            path.is_absolute() and ".." not in path.parts
            and not any(part.is_symlink() for part in (path, *path.parents))
            and parent["files"].get(name) == expected
            and path.is_file() and sha(path) == expected,
            "Inherited feature evidence is unowned or changed: " + name,
        )
        return path

    catalog = ROOT / "features/capabilities.json"
    require(update.get("catalog") == str(catalog), "Inherited feature catalog differs")
    owned(str(catalog), update.get("catalog_sha256"))
    archived = owned(update.get("parent_catalog"), update.get("parent_capabilities_sha256"))
    require(archived.parent == ROOT / "receipts"
            and re.fullmatch(r"features-parent-[0-9a-f]{16}\.json", archived.name),
            "Inherited parent feature catalog escapes its owner")
    require(update["assets"].get(str(catalog), {}).get("sha256") == update["catalog_sha256"],
            "Inherited feature assets omit the selected catalog")
    for name, asset in update["assets"].items():
        require(isinstance(asset, dict), "Invalid inherited feature asset")
        target = owned(name, asset.get("sha256"))
        # Map configurable test roots to the same production ownership rules.
        if target.is_relative_to(ROOT):
            logical = "/opt/sparkring/" + target.relative_to(ROOT).as_posix()
        elif target.is_relative_to(SITE):
            logical = "/opt/venv/lib/python3.12/site-packages/" + target.relative_to(SITE).as_posix()
        else:
            logical = name
        feature_asset_scope(logical)
    previous, current = read(archived), read(catalog)
    require(previous.get("schema") == "sparkring-image-capabilities/v1"
            and isinstance(previous.get("features"), dict),
            "Unsupported inherited parent feature catalog")
    verify_feature_dispositions(previous, current)
    return copy.deepcopy(update)


def install_feature_update(context, descriptor, parent=None):
    """Apply reviewed owned feature changes after preserving the parent catalog."""
    selected = descriptor.get("feature_update")
    if selected is None:
        return {}, verified_inherited_feature_update(parent) if parent is not None else None
    require(
        selected.get("file") == "feature-update.json",
        "Feature descriptor is not an owned context file",
    )
    manifest_path = context / selected["file"]
    require(
        sha(manifest_path) == selected["sha256"], "Feature-update descriptor differs"
    )
    manifest = read(manifest_path)
    require(
        manifest.get("schema") == "sparkring-native-feature-update/v1"
        and manifest.get("assets"),
        "Unknown or empty feature update",
    )
    catalog = ROOT / "features/capabilities.json"
    require(
        sha(catalog) == manifest.get("parent_capabilities_sha256"),
        "Parent feature catalog differs",
    )
    require(
        str(catalog) in manifest["assets"],
        "Feature update must declare its child catalog",
    )
    payloads = {}
    for name, asset in manifest["assets"].items():
        target = Path(name)
        scope = feature_asset_scope(name)
        require(
            not any(part.is_symlink() for part in (target, *target.parents)),
            "Feature target is outside its owner or symlinked: " + name,
        )
        parent_hash = asset.get("parent_sha256")
        require(
            scope != "fresh-metadata" or parent_hash is None,
            "Release and license metadata may only use a fresh destination",
        )
        require(
            (
                not target.exists()
                if parent_hash is None
                else target.is_file() and sha(target) == parent_hash
            ),
            "Feature target preimage differs: " + name,
        )
        source = context / "feature-assets" / name.lstrip("/")
        require(
            source.is_file() and sha(source) == asset["sha256"],
            "Feature asset differs: " + name,
        )
        data = source.read_bytes()
        if target.suffix == ".py":
            compile(data, name, "exec")
        if scope == "fresh-metadata" and target.suffix == ".json":
            require(
                isinstance(json.loads(data), dict),
                "Release metadata must be a JSON object",
            )
        payloads[target] = data
    child = json.loads(payloads[catalog])
    parent = read(catalog)
    verify_feature_dispositions(parent, child)
    transport_bundles = verify_transport_assets(payloads)
    retained = (
        ROOT
        / "receipts"
        / ("features-parent-" + descriptor["input_sha256"][:16] + ".json")
    )
    require(not retained.exists(), "Retained parent feature catalog already exists")
    retained.write_bytes(catalog.read_bytes())
    for target, data in payloads.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    files = {str(path): sha(path) for path in [retained, *payloads]}
    receipt = {
        "descriptor_sha256": selected["sha256"],
        "parent_catalog": str(retained),
        "parent_capabilities_sha256": manifest["parent_capabilities_sha256"],
        "catalog": str(catalog),
        "catalog_sha256": sha(catalog),
        "assets": manifest["assets"],
        "transport_bundles": transport_bundles,
        "serving_qualified": False,
    }
    return files, receipt


def verify_transport_assets(payloads):
    """Require the complete prepared transport and its installed API preimages."""
    transport = ROOT / "transports"
    if not any(
        path.is_relative_to(transport) or path == SITE / "sparkring_transport.pth"
        for path in payloads
    ):
        return {}
    bundle = transport / PREPARED_TRANSPORT_PROFILE
    manifest_path = bundle / "manifest.json"
    require(
        manifest_path in payloads,
        "Transport update omits the complete prepared bundle manifest",
    )
    manifest = json.loads(payloads[manifest_path])
    require(
        manifest.get("schema") == "sparkring-transport-bundle/v1"
        and manifest.get("name") == PREPARED_TRANSPORT_PROFILE
        and isinstance(manifest.get("files"), dict)
        and manifest["files"],
        "Prepared transport manifest identity or inventory differs",
    )
    actual = {
        path.relative_to(bundle).as_posix(): path
        for path in payloads
        if path != manifest_path and path.is_relative_to(bundle)
    }
    require(
        set(actual) == set(manifest["files"]),
        "Prepared transport payload is incomplete or contains unlisted files",
    )
    for relative, expected in manifest["files"].items():
        parts = PurePosixPath(relative)
        require(
            not parts.is_absolute()
            and ".." not in parts.parts
            and "\\" not in relative
            and ":" not in relative
            and re.fullmatch(r"[0-9a-f]{64}", expected)
            and hashlib.sha256(payloads[actual[relative]]).hexdigest() == expected,
            "Prepared transport payload differs: " + relative,
        )
    if bundle.exists():
        existing = {
            path.relative_to(bundle).as_posix()
            for path in bundle.rglob("*")
            if path.is_file()
            and path != manifest_path
            and "__pycache__" not in path.parts
        }
        require(
            existing <= set(actual),
            "Prepared transport directory contains unlisted inherited files",
        )
    preimages = manifest.get("image_source_preimages", {})
    required = {
        str(SITE / "b12x" / name)
        for name in (
            "preparation/__init__.py",
            "preparation/types.py",
            "preparation/tuning.py",
            "preparation/session.py",
            "_lib/compile_plan.py",
            "_lib/program_cache.py",
        )
    }
    require(
        isinstance(preimages, dict) and required <= set(preimages),
        "Prepared transport omits required B12X API preimages",
    )
    for name, expected in preimages.items():
        path = Path(name)
        require(
            path.is_absolute()
            and ".." not in path.parts
            and path.is_relative_to(SITE / "b12x")
            and not any(part.is_symlink() for part in (path, *path.parents))
            and path.is_file()
            and sha(path) == expected,
            "Installed prepared-transport API preimage differs: " + name,
        )
    return {
        PREPARED_TRANSPORT_PROFILE: {
            "manifest": str(manifest_path),
            "manifest_sha256": hashlib.sha256(payloads[manifest_path]).hexdigest(),
            "files": {
                str(actual[name]): expected
                for name, expected in manifest["files"].items()
            },
            "image_source_preimages": preimages,
            "serving_qualified": False,
        }
    }


def verify_feature_dispositions(parent, child):
    require(
        child.get("schema") == "sparkring-image-capabilities/v1"
        and child.get("features"),
        "Child feature catalog is unsupported or empty",
    )
    removed = set(parent["features"]) - set(child["features"])
    unsupported = child.get("unsupported_features", {})
    require(
        removed <= set(unsupported),
        "Removed features lack an explicit unsupported disposition",
    )
    for name in removed:
        require(
            unsupported[name].get("reason")
            and unsupported[name].get("replacement") in child["features"],
            "Removed feature lacks its reason or replacement: " + name,
        )


def selected_boundary_identity(parent):
    """Use an owned versioned cache identity when the foundation declares one."""
    native = parent.get("schema") == "sparkring-native-installed/v1"
    selected = (parent.get("boundary_runtime") if native else
                parent.get("cache_extension", {}).get("boundary_runtime"))
    if selected is None:
        if native:
            return None
        return ROOT / "contracts/boundary-runtime.json"
    require(isinstance(selected, dict) and isinstance(selected.get("path"), str)
            and isinstance(selected.get("sha256"), str), "Invalid boundary selection")
    path = Path(selected["path"])
    require(
        path.parent == ROOT / "contracts" and not path.is_symlink(),
        "Selected boundary identity escapes its owner",
    )
    require(
        parent["files"].get(str(path)) == selected["sha256"] == sha(path),
        "Selected boundary identity differs from the installed receipt",
    )
    return path


def install_source_binding(context, descriptor, compiler):
    selected = descriptor.get("source_binding")
    if selected is None:
        return {}
    for name in ("file", "proof_file"):
        require(
            Path(selected[name]).name == selected[name],
            "Binding artifact is not a context file",
        )
    contract_file, proof_file = (
        context / selected["file"],
        context / selected["proof_file"],
    )
    require(
        sha(contract_file) == selected["sha256"]
        and sha(proof_file) == selected["proof_sha256"],
        "Binding artifacts differ from installation inputs",
    )
    contract, proof = read(contract_file), read(proof_file)
    schema = proof.get("schema")
    if schema == "sparkring-binding-migration/v1":
        trees = proof.get("component_trees", {})
        require(
            trees == compiler["source_trees"]
            and proof.get("input_sha256") == descriptor["input_sha256"]
            and proof.get("contract_sha256")
            == hashlib.sha256(
                json.dumps(
                    contract, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest(),
            "Migration proof does not bind compiled components",
        )
        oracles = proof.get("oracles", [])
        require(
            oracles and {item.get("component") for item in oracles} == set(trees),
            "Migration proof omits a component oracle",
        )
        for item in oracles:
            receipt = item.get("receipt", {})
            require(
                receipt.get("schema") == "sparkring-upgrade-gate/v1"
                and receipt.get("input_sha256") == descriptor["input_sha256"]
                and receipt.get("subject_sha256") == trees[item["component"]]
                and receipt.get("variant") == "candidate"
                and receipt.get("outcome") == "passed"
                and receipt.get("skipped") == 0
                and type(receipt.get("assertions")) is int
                and receipt["assertions"] > 0,
                "Migration oracle does not describe compiled source",
            )
    else:
        require(
            schema == "sparkring-binding-equivalence/v1"
            and proof.get("oracle", {}).get("input_sha256")
            == descriptor["input_sha256"]
            and proof["oracle"].get("subject_sha256")
            == compiler["source_trees"]["vllm"]
            and proof["oracle"].get("variant") == "candidate"
            and proof["oracle"].get("outcome") == "passed"
            and proof["oracle"].get("skipped") == 0
            and type(proof["oracle"].get("assertions")) is int
            and proof["oracle"]["assertions"] > 0,
            "Binding proof does not describe the compiled source",
        )
    require(
        proof.get("candidate_tree_sha256") == compiler["source_trees"]["vllm"]
        and proof.get("serving_qualified") is False,
        "Binding proof does not describe the compiled source",
    )
    require(
        contract.get("schema") == "sparkring-vllm-kv-block-lease-contract/v1",
        "Unknown installed source-binding schema",
    )
    require(
        isinstance(contract.get("files"), list) and contract["files"],
        "Source binding has no verified files",
    )
    expected_name = (
        "vllm-connector-jobs-source-" + proof["candidate_tree_sha256"][:16] + ".json"
    )
    destination = ROOT / "contracts" / expected_name
    require(
        str(destination) == selected["destination"] and not destination.exists(),
        "Binding destination is not a fresh owned identity",
    )
    for row in contract["files"]:
        path = (SITE / row["path"]).resolve()
        require(
            path.is_relative_to(SITE.resolve()), "Bound source escapes site-packages"
        )
        require(
            sha(path) == row["sha256"],
            "Installed source differs from its rebound contract",
        )
    proof_destination = (
        ROOT
        / "receipts"
        / ("source-binding-" + descriptor["input_sha256"][:16] + ".json")
    )
    require(not proof_destination.exists(), "Binding proof destination already exists")
    destination.write_bytes(contract_file.read_bytes())
    proof_destination.write_bytes(proof_file.read_bytes())
    return {
        str(destination): sha(destination),
        str(proof_destination): sha(proof_destination),
    }


def install(context):
    context = Path(context)
    descriptor = read(context / "descriptor.json")
    compiler = read(context / "compiler-result.json")
    require(
        descriptor.get("schema") == "sparkring-native-install/v1",
        "Unknown native installation descriptor",
    )
    require(
        compiler.get("schema") == "sparkring-native-wheel-result/v1"
        and compiler["descriptor_sha256"] == descriptor["compiler_descriptor_sha256"]
        and compiler["source_trees"] == descriptor["source_trees"],
        "Compiler receipt does not bind accepted source",
    )
    parent, raw_parent, parent_receipt = read_parent_foundation(descriptor)
    isolated_sglang = isolated_sglang_inventory()
    boundary_path = selected_boundary_identity(parent)
    boundary = read(boundary_path) if boundary_path is not None and boundary_path.exists() else None
    if boundary is not None:
        require(
            boundary.get("schema") == "sparkcache-boundary-runtime/v1",
            "Unknown boundary runtime identity schema",
        )
        verify_files(
            {str(SITE / name): expected for name, expected in boundary["files"].items()}
        )
    require(
        metadata.version("torch") == compiler["torch_version"],
        "Native compiler and runtime Torch ABI differ",
    )
    protected_torch = {
        name: metadata.version(name) for name in ("torch", "torchvision", "torchaudio")
    }
    before = dict(parent["files"])
    dependencies = descriptor.get("runtime_dependencies", {})
    validate_runtime_dependencies(dependencies)
    protected_cuda_versions, cuda_namespace_before = cuda_protected_inputs(dependencies)
    before.update(cuda_namespace_before)
    packages = ["vllm", "b12x", *dependencies]
    package_names = [DEPENDENCY_PACKAGES.get(name, name) for name in packages if name not in CUDA_RUNTIME_VERSIONS]
    wheel_records = {**compiler["wheels"], **dependencies}
    selected = []
    selected_files, wheel_paths, normalizations = {}, {}, {}
    metadata_roots, exact_paths, cuda_destinations = [], set(), set()
    for name in packages:
        selected_files[name] = selected_distribution_files(name)
        before.update(selected_files[name])
        metadata_roots.extend(SITE.glob(name.replace("-", "_") + "-*.dist-info"))
        record = wheel_records[name]
        filename = record["file"]
        require(
            Path(filename).name == filename and filename.endswith(".whl"),
            "Wheel is not a contained context file",
        )
        path = context / "wheels" / filename
        require(sha(path) == record["sha256"], "Compiled wheel bytes differ")
        if name in CUDA_RUNTIME_VERSIONS:
            destinations = cuda_wheel_paths(name, path, record["version"])
            require(not cuda_destinations & destinations, "CUDA wheels collide in their shared namespace")
            cuda_destinations.update(destinations)
            exact_paths.update(cuda_record_paths(name, selected_files[name]))
            exact_paths.update(destinations)
        if name == "flashinfer-python":
            path, normalizations[name] = isolate_flashinfer_build_helpers(
                path, context / "installation-wheels", record["version"]
            )
        wheel_paths[name] = path
        selected.append(str(path))
    planned_paths, installed_metadata_roots = wheel_install_paths(wheel_paths)
    require(not set(parent.get("removed_files", [])) & set(planned_paths),
            "Native wheel reintroduces a foundation removed file")
    ownership_audit, preserved_helpers = audit_selected_ownership(
        selected_files,
        package_names,
        [*metadata_roots, *installed_metadata_roots],
        parent["files"],
        planned_paths,
        exact_paths,
    )
    prior_errors = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    ).stdout.splitlines()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            *selected,
        ],
        check=True,
    )
    if "flashinfer-python" in normalizations:
        restore_build_helpers(preserved_helpers)
    require(
        {name: metadata.version(name) for name in protected_torch} == protected_torch,
        "Wheel installation changed protected Torch versions",
    )
    require({name: metadata.version(name) for name in protected_cuda_versions} == protected_cuda_versions,
            "Wheel installation changed protected CUDA dependencies")
    after_errors = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    ).stdout.splitlines()
    require(
        set(after_errors) <= set(prior_errors),
        "Native wheel introduces unsatisfied dependencies: "
        + str(sorted(set(after_errors) - set(prior_errors))),
    )
    files, removed = {}, list(parent.get("removed_files", []))
    for path, expected in before.items():
        if package_owned(path, package_names, metadata_roots, exact_paths):
            if not Path(path).exists():
                removed.append(path)
        else:
            require(
                Path(path).is_file() and sha(path) == expected,
                "Unrelated foundation bytes changed: " + path,
            )
            files[path] = expected
    for name in packages:
        require(
            metadata.version(name) == wheel_records[name]["version"],
            "Installed distribution version differs",
        )
        files.update(distribution_files(name))
        if name in CUDA_RUNTIME_VERSIONS:
            continue
        # Include package additions that a retained foundation RECORD omitted.
        for path in (SITE / DEPENDENCY_PACKAGES.get(name, name)).rglob("*"):
            if path.is_file() and path.suffix not in (".pyc", ".pyo"):
                files[str(path)] = sha(path)
    files.update(install_source_binding(context, descriptor, compiler))
    feature_files, feature_update = install_feature_update(context, descriptor, parent)
    files.update(feature_files)
    if isolated_sglang is not None:
        verify_files(isolated_sglang["files"])
        files.update(isolated_sglang["files"])
    # Optional cache/feature additions remain unchanged and receive explicit hashes.
    for directory in (
        SITE / "sparkcache",
        ROOT / "contracts",
        ROOT / "features",
        ROOT / "transports",
    ):
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file() and path.suffix not in (".pyc", ".pyo"):
                    files[str(path)] = sha(path)
    for path in SITE.glob("*.pth"):
        files[str(path)] = sha(path)
    ENTRYPOINT.parent.mkdir(parents=True, exist_ok=True)
    ENTRYPOINT.write_bytes(Path(__file__).read_bytes())
    files[str(ENTRYPOINT)] = sha(ENTRYPOINT)
    retained_files, parent_receipt = retain_parent_receipt(raw_parent, parent_receipt)
    files.update(retained_files)
    boundary_record = None
    if boundary is not None:
        # A compiled runtime gets a distinct attestation. This is identity
        # regeneration, not evidence that its connector interfaces are compatible.
        bound_files = {
            str(Path(name).relative_to(SITE)): expected
            for name, expected in files.items()
            if Path(name).is_relative_to(SITE)
            and Path(name).relative_to(SITE).parts[0] in ("vllm", "b12x", "sparkcache")
        }
        boundary_native = ROOT / (
            "contracts/boundary-native-" + descriptor["input_sha256"][:16] + ".json"
        )
        require(
            not boundary_native.exists(),
            "Boundary attestation destination already exists",
        )
        boundary_native.write_text(
            json.dumps(
                {
                    "schema": boundary["schema"],
                    "files": bound_files,
                    "external": boundary["external"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        files[str(boundary_native)] = sha(boundary_native)
        boundary_record = {
            "path": str(boundary_native),
            "sha256": sha(boundary_native),
            "cache_namespace_suffix": descriptor["input_sha256"][:16],
            "serving_qualified": False,
        }
    receipt = {
        "schema": "sparkring-native-installed/v1",
        "input_sha256": descriptor["input_sha256"],
        "parent_image_id": descriptor["parent_image_id"],
        "parent_installed_sha256": descriptor["parent_installed_sha256"],
        "parent_receipt": parent_receipt,
        "compiler": compiler,
        "files": files,
        "removed_files": sorted(set(removed)),
        "versions": {
            **parent["versions"],
            **{name: metadata.version(name) for name in packages},
            **protected_cuda_versions,
        },
        "foundation_dependency_exceptions": prior_errors,
        "runtime_dependencies": dependencies,
        "protected_cuda_dependencies": protected_cuda_versions,
        "runtime_dependency_normalizations": normalizations,
        "distribution_ownership_audit": ownership_audit,
        "feature_update": feature_update,
        "isolated_sglang": (
            {key: value for key, value in isolated_sglang.items() if key != "files"}
            | {"files_verified": len(isolated_sglang["files"])}
        )
        if isolated_sglang
        else None,
        "active_contracts": sorted(
            descriptor.get("active_contracts", parent.get("integration_contracts", {}))
        ),
        "boundary_runtime": boundary_record,
        "serving_qualified": False,
    }
    NATIVE.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return verify()


def verify_installed_state(value):
    """Verify files, absent paths, versions and features before any replacement."""
    verify_files(value["files"])
    for path in value.get("removed_files", []):
        require(not Path(path).exists(), "Removed package file reappeared: " + path)
    for name, expected in value["versions"].items():
        require(
            metadata.version(name) == expected, "Installed version differs: " + name
        )
    features = {}
    capabilities = ROOT / "features/capabilities.json"
    if capabilities.exists():
        manifest = read(capabilities)
        require(
            manifest.get("schema") == "sparkring-image-capabilities/v1",
            "Unknown feature manifest schema",
        )
        for name, feature in manifest.get("features", {}).items():
            paths = {}
            for relative, expected in feature["files"].items():
                path = (ROOT / "features" / relative).resolve()
                require(
                    path.is_relative_to((ROOT / "features").resolve()),
                    "Feature inventory escapes its directory",
                )
                paths[str(path)] = expected
            require(paths, "Feature has no source inventory")
            verify_files(paths)
            features[name] = paths
    verified_inherited_feature_update(value)
    return features


def verify():
    value = read(NATIVE)
    require(
        value.get("schema") == "sparkring-native-installed/v1" and value.get("files"),
        "Missing native installed inventory",
    )
    features = verify_installed_state(value)
    return {
        "schema": "sparkring-native-verification/v1",
        "receipt_sha256": sha(NATIVE),
        "source_trees": value["compiler"]["source_trees"],
        "files_verified": len(value["files"]),
        "input_sha256": value["input_sha256"],
        "serving_qualified": False,
        "features": sorted(features),
        "isolated_sglang": value.get("isolated_sglang"),
    }


def main():
    if sys.argv[1:2] == ["install"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("action")
        parser.add_argument("--context", type=Path, required=True)
        print(json.dumps(install(parser.parse_args().context)))
        return
    if sys.argv[1:2] == ["verify"]:
        print(json.dumps(verify()))
        return
    require(sys.argv[1:2] == ["serve"], "Expected install, verify or serve")
    verify()
    argv = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", *sys.argv[2:]]
    os.execve(argv[0], argv, os.environ)


if __name__ == "__main__":
    main()
