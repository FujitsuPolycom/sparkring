"""Build a native-wheel candidate from accepted source on a leased ARM64 host."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.images.upgrades.contracts import beneath, load_policy, read, require, sha  # noqa: E402
from runtime.images.upgrades.contract_rebind import rebind  # noqa: E402
from runtime.images.upgrades.contract_migration import migrate  # noqa: E402
from runtime.images.upgrades.io import checked, write_json  # noqa: E402
from runtime.images.upgrades.sources import tree_digest, native_digest  # noqa: E402
from runtime.images.upgrades.native_worker import wheel_record  # noqa: E402
from runtime.images.upgrades.native_install import (  # noqa: E402
    PARENT_RECEIPTS, feature_asset_scope, normalize_parent_receipt,
    CUDA_RUNTIME_VERSIONS, cuda_wheel_paths, validate_runtime_dependencies,
)


def read_parent_image(image_id):
    """Read the most-specific receipt, preserving its exact bytes and identity."""
    script = (
        "import os, pathlib, sys\n"
        f"paths = {PARENT_RECEIPTS!r}\n"
        "kind = 'native' if os.path.lexists(paths['native']) else 'candidate'\n"
        "path = pathlib.Path(paths[kind])\n"
        "assert path.is_file() and not path.is_symlink(), 'Invalid foundation receipt path'\n"
        "sys.stdout.buffer.write(kind.encode() + b'\\n' + path.read_bytes())\n"
    )
    result = docker(
        "run", "--rm", "--pull", "never", "--runtime", "runc", "--network", "none",
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", "512m", "--cpus", "1",
        "--env", "SPARKRING_FEATURES=", "--env", "SPARKRING_TRANSPORT_PROFILE=",
        "--env", "PYTHONPATH=", "--env", "CUDA_VISIBLE_DEVICES=",
        "--env", "NVIDIA_VISIBLE_DEVICES=void",
        "--entrypoint", "/opt/venv/bin/python", image_id, "-c", script,
        limit=64 * 1024**2,
    )
    kind, separator, raw = result.partition(b"\n")
    require(separator, "Foundation receipt selection is missing")
    kind = kind.decode("ascii")
    parent = normalize_parent_receipt(raw, kind)
    return parent, raw, dict(kind=kind, path=PARENT_RECEIPTS[kind], sha256=sha(raw))


def prepare_runtime_dependencies(policy, context):
    """Stage exact reviewed dependency wheels without resolving packages online."""
    result = {}
    selections = policy["foundation"].get("runtime_dependencies", [])
    for selected in selections:
        require(isinstance(selected, dict)
                and set(selected) == {"path", "sha256", "name", "version", "source_url"},
                "Runtime dependency fields differ")
    require(len({item["name"] for item in selections}) == len(selections), "Duplicate runtime dependency")
    validate_runtime_dependencies({item["name"]: item for item in selections})
    for selected in selections:
        path = beneath(policy["_root"], selected["path"])
        require(
            path.suffix == ".whl" and sha(path.read_bytes()) == selected["sha256"],
            "Runtime dependency wheel differs from policy",
        )
        record = wheel_record(path, selected["name"])
        require(
            record["version"] == selected["version"],
            "Runtime dependency version differs",
        )
        if selected["name"] in CUDA_RUNTIME_VERSIONS:
            cuda_wheel_paths(selected["name"], path, selected["version"])
        destination = context / "wheels" / path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, destination)
        result[selected["name"]] = {**record, "source_url": selected["source_url"]}
    return result


def prepare_feature_update(policy, context):
    selected = policy["foundation"].get("feature_update")
    if selected is None:
        return None
    require(
        set(selected) == {"manifest", "sha256"}, "Feature-update policy fields differ"
    )
    path = beneath(policy["_root"], selected["manifest"])
    require(
        sha(path.read_bytes()) == selected["sha256"], "Feature-update manifest differs"
    )
    manifest = read(path)
    require(
        manifest.get("schema") == "sparkring-native-feature-update/v1"
        and manifest.get("assets"),
        "Unknown or empty feature update",
    )
    for target, asset in manifest["assets"].items():
        scope = feature_asset_scope(target)
        require(scope != "fresh-metadata" or asset.get("parent_sha256") is None,
                "Release and license metadata must use fresh destinations")
        source = beneath(policy["_root"], asset["source"])
        require(
            sha(source.read_bytes()) == asset["sha256"],
            "Feature asset differs: " + target,
        )
        destination = context / "feature-assets" / target.lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(path, context / "feature-update.json")
    return {"file": "feature-update.json", "sha256": selected["sha256"]}


def docker(*args, seconds=120, limit=4 * 1024**2):
    return checked(
        ["docker", "--host", "unix:///var/run/docker.sock", *map(str, args)],
        seconds=seconds,
        limit=limit,
    )


def validate_recipe(recipe):
    require(
        isinstance(recipe, dict)
        and set(recipe) - {"wheel_metadata_profile"}
        == {
            "schema",
            "architecture",
            "jobs",
            "cpus",
            "memory_bytes",
            "build_seconds",
            "torch_version",
            "build_type",
            "network",
        },
        "Native recipe fields differ",
    )
    require(recipe["schema"] == "sparkring-native-recipe/v1", "Unknown native recipe")
    require(
        recipe.get("wheel_metadata_profile") in (None, "gb10-dsl462-quack064"),
        "Unknown wheel metadata profile",
    )
    require(
        recipe["build_type"] in ("Release", "RelWithDebInfo"),
        "Native build_type must explicitly select Release or RelWithDebInfo",
    )
    require(recipe["architecture"] in ("12.1", "12.1a"), "Select the GB10 architecture")
    require(
        type(recipe["jobs"]) is int and 1 <= recipe["jobs"] <= 20,
        "Native jobs must be between 1 and 20",
    )
    require(
        type(recipe["cpus"]) is int and recipe["jobs"] <= recipe["cpus"] <= 20,
        "Native CPU limit must cover compiler jobs",
    )
    require(
        type(recipe["memory_bytes"]) is int
        and 8 * 1024**3 <= recipe["memory_bytes"] <= 104 * 1024**3,
        "Native memory limit is outside the GB10 recipe",
    )
    require(
        type(recipe["build_seconds"]) is int and 60 <= recipe["build_seconds"] <= 43200,
        "Native compiler timeout is invalid",
    )
    require(
        recipe["network"] in ("none", "bridge"),
        "Native compilation supports isolated or explicit bridge networking",
    )
    require(
        isinstance(recipe["torch_version"], str) and recipe["torch_version"],
        "Explicit compiler Torch ABI required",
    )
    return recipe


def select_parent_binding(parent, destination, raw_contract, raw_extension=None):
    """Select an owned contract, or prove its source extension supersedes one base."""
    active = set(parent.get("integration_contracts", {}))
    contract_hash = sha(raw_contract)
    require(
        parent.get("files", {}).get(destination) == contract_hash,
        "Selected parent contract is not owned with matching bytes",
    )
    if destination in active:
        return active, None
    extension = parent.get("source_extension", {})
    descriptor_path = "/opt/sparkring/receipts/source-extension.json"
    require(
        raw_extension is not None
        and extension.get("provenance")
        and extension.get("descriptor_sha256")
        == sha(raw_extension)
        == parent["files"].get(descriptor_path),
        "Inactive contract lacks verified source-extension provenance",
    )
    descriptor = json.loads(raw_extension)
    require(
        descriptor.get("schema") == "sparkring-source-extension/v1"
        and descriptor.get("id") == extension.get("id")
        and descriptor.get("parent", {}).get("image_id")
        == extension.get("parent_image_id")
        and descriptor["parent"].get("receipt_sha256")
        == extension.get("parent_receipt_sha256")
        and descriptor.get("provenance") == extension["provenance"],
        "Source-extension descriptor identity differs",
    )
    require(
        descriptor.get("integration_contracts", {}).get(destination, {}).get("sha256")
        == contract_hash,
        "Source extension does not declare the selected contract",
    )
    contract = json.loads(raw_contract)
    require(
        contract.get("schema") == "sparkring-vllm-kv-block-lease-contract/v1"
        and isinstance(contract.get("files"), list)
        and contract["files"],
        "Source-extension contract has no source inventory",
    )
    rows = set()
    for item in contract["files"]:
        name = item.get("path", "")
        path = PurePosixPath(name)
        require(
            name
            and not path.is_absolute()
            and ".." not in path.parts
            and "\\" not in name
            and path.parts[0] in ("vllm", "b12x")
            and name not in rows
            and re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", ""))
            and parent["files"].get("/opt/venv/lib/python3.12/site-packages/" + name)
            == item["sha256"],
            "Source-extension contract does not match installed source: " + name,
        )
        rows.add(name)
    base_hash = contract.get("semantic_review", {}).get("base_contract_sha256")
    require(
        isinstance(base_hash, str) and re.fullmatch(r"[0-9a-f]{64}", base_hash),
        "Source-extension contract does not identify its reviewed base",
    )
    bases = [
        name
        for name in active
        if name.startswith("/opt/sparkring/contracts/vllm-connector-jobs-")
        and parent["integration_contracts"][name].get("sha256") == base_hash
    ]
    require(
        len(bases) == 1,
        "Source-extension base is absent or ambiguous among active contracts",
    )
    require(
        parent["files"].get(bases[0]) == base_hash,
        "Reviewed active base contract does not match its owned bytes",
    )
    active.remove(bases[0])
    active.add(destination)
    return active, {
        "selected_contract": destination,
        "selected_contract_sha256": contract_hash,
        "superseded_contract": bases[0],
        "superseded_contract_sha256": base_hash,
        "source_extension_id": extension["id"],
        "source_extension_descriptor_sha256": sha(raw_extension),
        "installed_source_files_verified": len(rows),
        "serving_qualified": False,
    }


def prepare_source_binding(policy, bundle, paths, parent, context):
    config = policy["foundation"].get("source_binding")
    if config is None:
        return None, None
    contract_path = beneath(policy["_root"], config["contract"])
    original_destination = "/opt/sparkring/contracts/" + contract_path.name
    raw_extension = None
    if original_destination not in parent.get("integration_contracts", {}):
        raw_extension = docker(
            "run",
            "--rm",
            "--pull",
            "never",
            "--runtime",
            "runc",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "64m",
            "--cpus",
            "1",
            "--entrypoint",
            "/bin/cat",
            policy["foundation"]["image_id"],
            "/opt/sparkring/receipts/source-extension.json",
            limit=16 * 1024**2,
        )
    active, selection = select_parent_binding(
        parent, original_destination, contract_path.read_bytes(), raw_extension
    )
    source = bundle["sources"]["vllm"]
    oracle = next(
        (
            item
            for item in source.get("oracles", [])
            if item["gate"] == config.get("oracle")
        ),
        None,
    )
    if "migration" in config:
        migration = config["migration"]
        manifest_file = beneath(policy["_root"], migration["path"])
        require(
            sha(manifest_file.read_bytes()) == migration["sha256"],
            "Migration policy changed",
        )
        contract, proof = migrate(
            read(contract_path),
            read(manifest_file),
            paths,
            bundle["sources"],
            input_sha256=bundle["input_sha256"],
        )
    else:
        require(oracle is not None, "Accepted vLLM source lacks its binding oracle")
        contract, proof = rebind(
            read(contract_path),
            config["reference_source"],
            paths["vllm"],
            source["target_commit"],
            oracle,
            input_sha256=bundle["input_sha256"],
        )
    destination = (
        "/opt/sparkring/contracts/vllm-connector-jobs-source-"
        + proof["candidate_tree_sha256"][:16]
        + ".json"
    )
    if selection is not None:
        proof["parent_binding_selection"] = selection
    write_json(context / "source-binding.json", contract)
    write_json(context / "source-binding-proof.json", proof)
    active.remove(original_destination)
    active.add(destination)
    return {
        "file": "source-binding.json",
        "sha256": sha((context / "source-binding.json").read_bytes()),
        "proof_file": "source-binding-proof.json",
        "proof_sha256": sha((context / "source-binding-proof.json").read_bytes()),
        "destination": destination,
    }, sorted(active)


def select_native_cache(policy, native_inputs, compiler_id, recipe):
    """Reuse exact native inputs or take an explicitly authorized full-build path."""
    cached = policy["foundation"].get("native_cache")
    if cached is None:
        return None, None, {"mode": "compile", "reason": "No native cache selected"}
    cache_path = Path(cached["manifest"])
    require(
        sha(cache_path.read_bytes()) == cached["sha256"],
        "Native-cache manifest differs",
    )
    record = read(cache_path)
    require(
        record.get("schema") == "sparkring-native-cache/v1",
        "Unsupported native cache manifest",
    )
    expected = {
        "native_inputs": native_inputs,
        "compiler_image_id": compiler_id,
        "architecture": recipe["architecture"],
        "torch_version": recipe["torch_version"],
        "build_type": recipe["build_type"],
    }
    require(
        set(expected) - {"build_type"} <= set(record),
        "Native-cache compatibility fields missing",
    )
    # Legacy manifests do not prove optimization mode and cannot be reused.
    changed = sorted(key for key, value in expected.items() if record.get(key) != value)
    decision = {"manifest_sha256": cached["sha256"], "changed_inputs": changed}
    if changed:
        require(
            cached.get("on_input_change", "refuse") == "rebuild"
            and policy.get("build", {}).get("supports_native_rebuild") is True,
            "Native build inputs or compiler image changed; rebuild is required",
        )
        return None, None, {**decision, "mode": "compile"}
    return record, cache_path.parent, {**decision, "mode": "reuse"}


def build(policy_path, bundle_path, output, result_path):
    policy, bundle = load_policy(policy_path), read(bundle_path)
    recipe = validate_recipe(policy.get("native"))
    require(bundle["policy_sha256"] == policy["_digest"], "Native build policy differs")
    require(
        set(bundle["sources"]) == {"vllm", "b12x"},
        "Native adapter requires vLLM/B12X sources",
    )
    require(policy.get("platform") == "linux/arm64", "Native recipe is ARM64 only")
    parent_id = policy["foundation"]["image_id"]
    compiler_id = policy["foundation"].get("compiler_image_id", parent_id)
    require(
        re.fullmatch(r"sha256:[0-9a-f]{64}", parent_id),
        "Immutable local foundation image ID required",
    )
    parent = json.loads(docker("image", "inspect", parent_id))[0]
    require(
        parent["Id"] == parent_id
        and parent["Os"] == "linux"
        and parent["Architecture"] == "arm64",
        "Foundation image/platform differs",
    )
    compiler_info = json.loads(docker("image", "inspect", compiler_id))[0]
    require(
        compiler_info["Id"] == compiler_id and compiler_info["Architecture"] == "arm64",
        "Compiler image/platform differs",
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    name = (
        "sr-upgrade-native-"
        + bundle["input_sha256"][:12]
        + "-"
        + sha(str(output).encode())[:8]
    )
    work = output / "compiler"
    work.mkdir()
    context = output / "image"
    context.mkdir()
    paths = {
        name: Path(item["candidate_path"]).resolve()
        for name, item in bundle["sources"].items()
    }
    require(
        len({p.parent for p in paths.values()}) == 1,
        "Accepted component sources must share a root",
    )
    source_root = next(iter(paths.values())).parent
    source_trees = {key: tree_digest(path) for key, path in paths.items()}
    require(
        source_trees
        == {
            key: item["candidate_tree_sha256"]
            for key, item in bundle["sources"].items()
        },
        "Accepted native source bytes differ",
    )
    source_descriptor = {
        "schema": "sparkring-native-wheel-build/v1",
        "input_sha256": bundle["input_sha256"],
        "sources": {key: {"tree_sha256": value} for key, value in source_trees.items()},
        "architecture": recipe["architecture"],
        "jobs": recipe["jobs"],
        "torch_version": recipe["torch_version"],
        "build_type": recipe["build_type"],
        "distribution_version": "0.26.1rc0+sparkring.native."
        + bundle["input_sha256"][:12],
    }
    if recipe.get("wheel_metadata_profile"):
        source_descriptor["wheel_metadata_profile"] = recipe["wheel_metadata_profile"]
    native_inputs = {
        source["id"]: native_digest(paths[source["id"]], source["native_paths"])
        for source in policy["sources"]
    }
    source_descriptor["native_inputs"] = native_inputs
    record, cache_directory, decision = select_native_cache(
        policy, native_inputs, compiler_id, recipe
    )
    source_descriptor["native_cache_decision"] = decision
    if record is not None:
        source_descriptor["native_cache"] = record
    descriptor_path = output / "compiler-descriptor.json"
    write_json(descriptor_path, source_descriptor)
    worker = Path(__file__).with_name("native_worker.py")
    argv = [
        "create",
        "--name",
        name,
        "--label",
        "sparkring.upgrade.input=" + bundle["input_sha256"],
        "--runtime",
        "runc",
        "--network",
        recipe["network"],
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        str(recipe["memory_bytes"]),
        "--memory-swap",
        str(recipe["memory_bytes"]),
        "--cpus",
        str(recipe["cpus"]),
        "--pids-limit",
        "4096",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "NVIDIA_VISIBLE_DEVICES=void",
        "--env",
        "CUDA_VISIBLE_DEVICES=",
        "--env",
        "SPARKRING_FEATURES=",
        "--env",
        "SPARKRING_TRANSPORT_PROFILE=",
        "--env",
        "PYTHONPATH=",
        "--mount",
        f"type=bind,src={source_root},dst=/source,readonly",
        "--mount",
        f"type=bind,src={work},dst=/work",
        "--mount",
        f"type=bind,src={descriptor_path},dst=/compiler-descriptor.json,readonly",
        "--mount",
        f"type=bind,src={worker},dst=/native-worker.py,readonly",
        "--entrypoint",
        "/opt/venv/bin/python",
        compiler_id,
        "/native-worker.py",
        "--descriptor",
        "/compiler-descriptor.json",
        "--sources",
        "/source",
        "--work",
        "/work",
    ]
    write_json(
        output / "owned-resource.json",
        {"container": name, "image": parent_id, "input_sha256": bundle["input_sha256"]},
    )
    if cache_directory:
        position = argv.index("--entrypoint")
        argv[position:position] = [
            "--mount",
            f"type=bind,src={cache_directory},dst=/native-cache,readonly",
        ]
    docker(*argv)
    print(
        json.dumps(
            {
                "compiler_container": name,
                "tail_command": "docker logs -f --tail 80 " + name,
            }
        ),
        flush=True,
    )
    try:
        logs = docker(
            "start",
            "--attach",
            name,
            seconds=recipe["build_seconds"],
            limit=policy["budgets"]["output_bytes"],
        )
        (output / "compiler.log").write_bytes(logs)
        state = json.loads(docker("inspect", name))[0]["State"]
        require(
            not state["Running"] and state["ExitCode"] == 0,
            "Native compiler did not exit successfully; retain its container/logs",
        )
        compiled = read(work / "result.json")
        require(
            compiled["descriptor_sha256"] == sha(descriptor_path.read_bytes())
            and compiled["source_trees"] == source_trees
            and compiled["build_type"] == recipe["build_type"],
            "Compiler receipt identifies other inputs",
        )
    except BaseException:
        # The caller marks failed actions uncertain and requires explicit cleanup.
        # Retain the named compiler container so its state and logs remain inspectable.
        raise
    parent_receipt, raw_parent, parent_selection = read_parent_image(parent_id)
    require(
        parent_receipt["versions"]["torch"] == compiled["torch_version"],
        "Compiled wheel and foundation Torch ABI differ",
    )
    wheels = context / "wheels"
    wheels.mkdir(exist_ok=True)
    for record in compiled["wheels"].values():
        path = work / "wheels" / record["file"]
        require(
            sha(path.read_bytes()) == record["sha256"], "Compiled wheel hash differs"
        )
        shutil.copyfile(path, wheels / path.name)
    install_descriptor = {
        "schema": "sparkring-native-install/v1",
        "input_sha256": bundle["input_sha256"],
        "parent_image_id": parent_id,
        "parent_installed_sha256": sha(raw_parent),
        "parent_receipt": parent_selection,
        "compiler_descriptor_sha256": compiled["descriptor_sha256"],
        "source_trees": source_trees,
        "runtime_dependencies": prepare_runtime_dependencies(policy, context),
        "feature_update": prepare_feature_update(policy, context),
    }
    binding, active = prepare_source_binding(
        policy, bundle, paths, parent_receipt, context
    )
    if binding is not None:
        install_descriptor["source_binding"] = binding
        install_descriptor["active_contracts"] = active
    write_json(context / "descriptor.json", install_descriptor)
    write_json(context / "compiler-result.json", compiled)
    shutil.copyfile(
        Path(__file__).with_name("native_install.py"), context / "native_install.py"
    )
    shutil.copyfile(
        Path(__file__).with_name("Dockerfile.native"), context / "Dockerfile"
    )
    tag = "local/sparkring-upgrade:native-" + bundle["input_sha256"][:16]
    parent_tag = "local/sparkring-upgrade-foundation:" + parent_id[7:23]
    docker("tag", parent_id, parent_tag)
    require(
        json.loads(docker("image", "inspect", parent_tag))[0]["Id"] == parent_id,
        "Foundation build alias differs",
    )
    docker(
        "build",
        "--network",
        "none",
        "--pull=false",
        "--build-arg",
        "PARENT=" + parent_tag,
        "--tag",
        tag,
        context,
        seconds=recipe["build_seconds"],
        limit=policy["budgets"]["output_bytes"],
    )
    info = json.loads(docker("image", "inspect", tag))[0]
    verification = json.loads(
        docker(
            "run",
            "--rm",
            "--runtime",
            "runc",
            "--network",
            "none",
            info["Id"],
            "verify",
            seconds=300,
        )
    )
    require(
        verification["source_trees"] == source_trees
        and verification["files_verified"] > 0
        and verification["serving_qualified"] is False,
        "Native installed verification differs",
    )
    require(
        set(policy.get("required_features", [])) <= set(verification["features"]),
        "Native image omits required feature files",
    )
    result = {
        "schema": "sparkring-upgrade-build/v1",
        "input_sha256": bundle["input_sha256"],
        "source_trees": source_trees,
        "image_id": info["Id"],
        "image_tag": tag,
        "platform": "linux/arm64",
        "installed_verified": True,
        "features": verification["features"],
        "native_rebuilt": compiled.get("native_rebuilt", True),
        "verification": verification,
        "scope": "Compiled runtime and preserved foundation inventory; profile/feature admission remains independently gated.",
    }
    write_json(result_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "bundle", "output", "result"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(build(args.policy, args.bundle, args.output, args.result)),
        flush=True,
    )
