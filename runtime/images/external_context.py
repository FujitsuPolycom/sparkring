"""Prepare a pinned SparkRing source composition over an external ARM64 image.

All inputs are local artifacts named by a manifest. Preparation does not pull an
image, build native code, run Docker or change a deployment. The base image keeps
ownership of vLLM/B12X native and generated files absent from the source archives.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import pprint
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


SITE = "/usr/local/lib/python3.12/dist-packages/"
TRANSPORT = "tp2-rocenante-adaptive-prepared"
FEATURE_ROOT = "/opt/sparkring/features/"
PREFILL_FILES = (
    "package_prefill.py",
    "qwen4_prefill_bootstrap.py",
    "qwen4_hc_fusion.py",
    "qwen4_mtp_gemm.py",
    "qwen4_fused_gate_kernel.py",
)
ASSET_ROOTS = (
    "sparkcache", "sparkcache-overrides", "sparkcache-native", "licenses",
    "transports", "features", "nccl-lib",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def relative_path(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and str(path) == name and path.parts
            and not path.is_absolute() and ".." not in path.parts and "\\" not in name,
            "Invalid artifact path: " + str(name))
    return path


def pinned_artifact(root, record):
    path = root / record["path"]
    require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
            "Invalid artifact digest: " + str(path))
    require(path.is_file() and file_sha(path) == record["sha256"],
            "Artifact digest differs: " + str(path))
    return path


def source_entries(archive, component):
    result = {}
    with tarfile.open(archive) as source:
        for member in source:
            path = PurePosixPath(member.name)
            if not path.parts or path.parts[0] != component:
                continue
            relative_path(member.name)
            require(member.isfile() or member.isdir(),
                    "Source archive contains a non-regular entry: " + member.name)
            if not member.isfile():
                continue
            require(member.name not in result, "Duplicate source entry: " + member.name)
            require(not (".so" in path.name or path.suffix in {".dll", ".pyd", ".a", ".o"}),
                    "Source archive contains a framework binary: " + member.name)
            require(path.suffix != ".pyc", "Source archive contains bytecode: " + member.name)
            result[member.name] = (
                source.extractfile(member).read(), 0o755 if member.mode & 0o111 else 0o644,
            )
    require(result, "Source archive lacks package: " + component)
    return result


def assignment(raw, name):
    text = raw.decode()
    nodes = [node for node in ast.parse(text).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == name
                     for target in node.targets)]
    require(len(nodes) == 1, "Source-binding declaration changed: " + name)
    return text, nodes[0]


def assign(raw, name, value):
    text, node = assignment(raw, name)
    lines = text.splitlines(keepends=True)
    return ("".join(lines[:node.lineno - 1]) + name + " = "
            + pprint.pformat(value, sort_dicts=True) + "\n"
            + "".join(lines[node.end_lineno:])).encode()


def verify_assets(root, export_path, receipt_path):
    """Bind exported assets to their parent receipt or explicit native digest."""
    exported = json.loads(export_path.read_bytes())
    parent = json.loads(receipt_path.read_bytes())
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", exported["parent_image"]) is not None,
            "Asset parent must be an immutable image ID")
    require(file_sha(receipt_path) == exported["parent_receipt_sha256"],
            "Exported parent receipt differs")
    actual = {}
    for directory in ASSET_ROOTS:
        require((root / directory).is_dir(), "Missing asset directory: " + directory)
        for path in sorted((root / directory).rglob("*")):
            if path.is_symlink():
                require(directory == "nccl-lib" and path.resolve().parent == path.parent.resolve()
                        and path.resolve().is_file(), "Unsupported asset symlink: " + str(path))
            if path.is_file() and path.suffix != ".pyc":
                actual[path.relative_to(root).as_posix()] = file_sha(path)
    require(set(actual) == set(exported["files"]), "Exported asset inventory differs")
    for name, digest in actual.items():
        relative_path(name)
        row = exported["files"][name]
        require(digest == row["sha256"], "Exported asset digest differs: " + name)
        # Patched NCCL was exported separately from the native-installed receipt.
        # Its digest is pinned by the export manifest itself, never inferred from
        # the unmodified upstream NCCL distribution.
        if not name.startswith("nccl-lib/"):
            require(parent["files"].get(row["origin"]) == digest,
                    "Asset does not match parent receipt: " + name)
    nccl_files = [name for name in actual
                  if re.fullmatch(r"nccl-lib/libnccl\.so\.\d+\.\d+\.\d+", name)]
    require(len(nccl_files) == 1, "Exactly one versioned NCCL library is required")
    nccl = nccl_files[0]
    require(set(name for name in actual if name.startswith("nccl-lib/"))
            == {nccl, "nccl-lib/libnccl.so", "nccl-lib/libnccl.so.2"},
            "Unexpected NCCL export inventory")
    require(actual[nccl] == actual["nccl-lib/libnccl.so"]
            == actual["nccl-lib/libnccl.so.2"], "NCCL aliases do not resolve to its payload")
    return exported, nccl


def load_inputs(manifest_path):
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest.get("schema") == "sparkring-external-inputs/v1", "Unknown inputs schema")
    require(manifest.get("platform") == "linux/arm64", "Only Linux ARM64 is supported")
    base = manifest["base"]
    require(re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", base["reference"]) is not None,
            "Base image must be digest-pinned")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", base["config_id"]) is not None,
            "Base image config ID must be pinned")
    root = manifest_path.resolve().parent
    paths = {name: pinned_artifact(root, manifest[name])
             for name in ("installer", "base_inventory", "parent_cache_contract")}
    inventory = json.loads(paths["base_inventory"].read_bytes())
    require(inventory["architecture"] == "aarch64", "Base inventory is not ARM64")
    require(set(manifest["sources"]) == {"vllm", "b12x"}, "Both framework sources are required")
    sources = {}
    for name, row in manifest["sources"].items():
        for revision in ("commit", "upstream", "baseline_commit"):
            require(re.fullmatch(r"[0-9a-f]{40}", row[revision]) is not None,
                    "Source revision must be a full Git SHA: " + name + "/" + revision)
        require(inventory["packages"][name]["root"] == SITE + name,
                "Unsupported Python site for " + name)
        sources[name] = {
            key: source_entries(pinned_artifact(root, row[key]), name)
            for key in ("archive", "baseline_archive")
        }
    assets = manifest["assets"]
    asset_root = root / assets["root"]
    export_path = pinned_artifact(root, assets["export_manifest"])
    receipt_path = pinned_artifact(root, assets["parent_receipt"])
    exported, nccl = verify_assets(asset_root, export_path, receipt_path)
    controller = manifest["prefill_controller"]
    require(set(controller["files"]) == set(PREFILL_FILES), "Prefill controller inventory differs")
    controller_root = root / controller["root"]
    for name, digest in controller["files"].items():
        pinned_artifact(controller_root, {"path": name, "sha256": digest})
    return manifest, paths, inventory, sources, asset_root, exported, nccl, controller_root


class Composition:
    def __init__(self, context):
        self.context = context
        self.files = {}
        self.contents = {}
        self.links = {}

    def add(self, name, data, mode=0o644, before=None):
        relative = name.lstrip("/")
        relative_path(relative)
        target = self.context / "payload" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.files[name] = {"before": before, "after": sha(data), "payload": relative, "mode": mode}
        self.contents[name] = data

    def copy_tree(self, source, destination):
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix != ".pyc":
                self.add(destination + "/" + path.relative_to(source).as_posix(),
                         path.read_bytes(), 0o755 if ".so" in path.name else 0o644)

    def overlay_sources(self, inventory, sources):
        for component, archives in sources.items():
            existing = inventory["packages"][component]["files"]
            candidate = archives["archive"]
            for relative, (data, mode) in candidate.items():
                name = relative[len(component) + 1:]
                before = existing.get(name, {}).get("sha256")
                self.contents[SITE + relative] = data
                if before != sha(data):
                    self.add(SITE + relative, data, mode, before)
            for relative in sorted(archives["baseline_archive"].keys() - candidate.keys()):
                name = relative[len(component) + 1:]
                if name in existing:
                    self.files[SITE + relative] = {"before": existing[name]["sha256"], "after": None}
            # The baseline Git archive bounds deletions. Base-image vendor and
            # generated files outside it retain their original owner and bytes.

    def bind_transport(self):
        transport = "/opt/sparkring/transports/" + TRANSPORT + "/manifest.json"
        manifest = json.loads(self.contents[transport])
        manifest["image_source_preimages"] = {
            SITE + name.split("/site-packages/")[1]:
            sha(self.contents[SITE + name.split("/site-packages/")[1]])
            for name in manifest["image_source_preimages"]
        }
        manifest["qualification"] = {
            "cpu": "pending composed-source checks", "gpu_rdma": "pending", "serving": "pending",
        }
        self.add(transport, json_bytes(manifest))
        selector = "/opt/sparkring/transports/sparkring_transport_selector.py"
        self.add(selector, assign(self.contents[selector], "HOST_SOURCE_PREFIX", SITE + "b12x/"))
        self.add(SITE + "sparkring_transport.pth", (
            "/opt/sparkring/transports\n"
            "import sparkring_transport_selector; sparkring_transport_selector.install_from_environment()\n"
        ).encode())
        return sha(self.contents[transport])

    def bind_features(self, assets, controller, transport_digest):
        bindings = {SITE + relative: sha(self.contents[SITE + relative]) for relative in (
            "vllm/models/qwen4_exp/nvidia/hyperconnection.py", "b12x/sequence/mtp_feedback/_kernels.py",
        )}
        prefill = self.context / "prefill-bundle"
        with tempfile.TemporaryDirectory() as temporary:
            binding_file = Path(temporary) / "bindings.json"
            binding_file.write_bytes(json_bytes(bindings))
            subprocess.run([
                sys.executable, "-B", str(controller / "package_prefill.py"),
                "--destination", str(prefill), "--source-bindings", str(binding_file),
            ], check=True, capture_output=True, text=True)
        prefill_digest = file_sha(prefill / "manifest.json")
        for path in sorted(prefill.iterdir()):
            self.add(FEATURE_ROOT + "qwen4-prefill/" + path.name, path.read_bytes())
        self.copy_tree(assets / "features/qwen-collectives", FEATURE_ROOT + "qwen-collectives")
        collective = FEATURE_ROOT + "qwen-collectives/qwen38_collective_policy.py"
        _, node = assignment(self.contents[collective], "SOURCE_HASHES")
        bindings = {}
        for module, expected in ast.literal_eval(node.value).items():
            raw = self.contents[SITE + module.replace(".", "/") + ".py"]
            # Git archives may convert line endings. Rebinding is permitted only
            # after proving normalization recovers the parent's source identity.
            require(sha(raw.replace(b"\r\n", b"\n")) == expected,
                    "Collective adapter source identity changed: " + module)
            bindings[module] = sha(raw)
        self.add(collective, assign(self.contents[collective], "SOURCE_HASHES", bindings))
        self.add(FEATURE_ROOT + "sparkring_features.py",
                 (assets / "features/sparkring_features.py").read_bytes())
        capabilities = json.loads((assets / "features/capabilities.json").read_bytes())
        capabilities["features"]["qwen-collectives"]["files"] = {
            name: sha(self.contents[FEATURE_ROOT + name])
            for name in capabilities["features"]["qwen-collectives"]["files"]
        }
        feature = capabilities["features"]["qwen4-prefill"]
        feature["manifest_sha256"] = prefill_digest
        feature["files"] = {"qwen4-prefill/" + path.name: file_sha(path)
                            for path in prefill.iterdir()}
        feature["scope"] = "TP4 sharded HC prefill fusion; native KK decode path retained. GPU qualification pending."
        capabilities["transport_profiles"][TRANSPORT]["manifest_sha256"] = transport_digest
        capabilities["qualification"] = "Development composition. Source and GPU qualification are separate receipts."
        capabilities["inherited_capabilities"] = [
            "SparkCache Qwen hybrid integration assets; composition qualification pending",
            "Patched dual-domain NCCL", "Adaptive prepared RoCEnante source bundle",
        ]
        self.add(FEATURE_ROOT + "capabilities.json", json_bytes(capabilities))
        self.add(SITE + "sparkring_features.pth", (
            "/opt/sparkring/features\n"
            "import sparkring_features; sparkring_features.install_or_exit()\n"
        ).encode())


def prepare(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    require(not output.exists(), "Output already exists: " + str(output))
    manifest, paths, inventory, sources, assets, exported, nccl, controller = load_inputs(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".external-context-", dir=output.parent) as temporary:
        context = Path(temporary) / "context"
        context.mkdir()
        composition = Composition(context)
        composition.overlay_sources(inventory, sources)
        for directory, destination in (
            ("sparkcache", SITE + "sparkcache"),
            ("sparkcache-overrides", SITE + "sparkcache"),
            ("sparkcache-native", "/opt/sparkring/sparkcache"),
            ("licenses", "/opt/sparkring/licenses"),
            ("transports", "/opt/sparkring/transports"),
        ):
            composition.copy_tree(assets / directory, destination)
        nccl_name = PurePosixPath(nccl).name
        composition.add("/opt/local-inference/nccl/lib/" + nccl_name,
                        (assets / nccl).read_bytes(), 0o755)
        composition.links = {
            "/opt/local-inference/nccl/lib/libnccl.so.2": nccl_name,
            "/opt/local-inference/nccl/lib/libnccl.so": "libnccl.so.2",
        }
        transport_digest = composition.bind_transport()
        composition.bind_features(assets, controller, transport_digest)
        contract = json.loads(paths["parent_cache_contract"].read_bytes())
        for row in contract["files"]:
            require(SITE + row["path"] in composition.contents,
                    "Cache contract source is missing: " + row["path"])
            row["sha256"] = sha(composition.contents[SITE + row["path"]])
        contract["base_commit"] = manifest["sources"]["vllm"]["commit"]
        contract["qualification"] = "Development source binding. Protected CPU source and GPU restore qualification remain required."
        contract_raw = json_bytes(contract)
        contract_path = "/opt/sparkring/contracts/vllm-connector-jobs-eugr-" + sha(contract_raw)[:16] + ".json"
        composition.add(contract_path, contract_raw)
        source_pins = {
            name: {
                **{key: row[key] for key in ("commit", "upstream", "baseline_commit")},
                "archive": Path(row["archive"]["path"]).name,
                "archive_sha256": row["archive"]["sha256"],
                "baseline_archive": Path(row["baseline_archive"]["path"]).name,
                "baseline_archive_sha256": row["baseline_archive"]["sha256"],
            } for name, row in manifest["sources"].items()
        }
        descriptor = {
            "schema": "sparkring-external-composition/v1", "platform": "linux/arm64",
            "base": manifest["base"], "sources": source_pins,
            "base_inventory_sha256": manifest["base_inventory"]["sha256"],
            "installer_sha256": manifest["installer"]["sha256"],
            "provenance": {
                "preparer_sha256": file_sha(Path(__file__)),
                "input_manifest_sha256": file_sha(manifest_path),
                "parent_cache_contract_sha256": manifest["parent_cache_contract"]["sha256"],
                "prefill_controller_files": manifest["prefill_controller"]["files"],
                "asset_export_sha256": manifest["assets"]["export_manifest"]["sha256"],
                "asset_export": exported,
            },
            "files": composition.files, "symlinks": composition.links,
            "capabilities": {
                "features": ["qwen-collectives", "qwen4-prefill"],
                "sparkcache_contract": contract_path,
                "transport_profile": TRANSPORT, "transport_manifest_sha256": transport_digest,
                "hc_projection_tp": True, "hc_prefill_row_ownership": "off", "serving_qualified": False,
            },
        }
        (context / "composition.json").write_bytes(json_bytes(descriptor))
        shutil.copyfile(paths["base_inventory"], context / "base-inventory.json")
        shutil.copyfile(paths["installer"], context / "external_base.py")
        (context / "Dockerfile").write_bytes((
            "FROM " + descriptor["base"]["reference"] + "\n"
            "ENV PYTHONDONTWRITEBYTECODE=1\n"
            "COPY . /tmp/sparkring-context\n"
            "RUN python3 /tmp/sparkring-context/external_base.py install --context /tmp/sparkring-context\n"
            'ENTRYPOINT ["python3", "/opt/sparkring/bin/external-base.py"]\n'
            'CMD ["verify"]\n'
        ).encode())
        context.rename(output)
    return {
        "context": str(output), "source_changes_and_assets": len(composition.files),
        "composition_sha256": file_sha(output / "composition.json"),
        "capabilities": descriptor["capabilities"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output)))


if __name__ == "__main__":
    main()
