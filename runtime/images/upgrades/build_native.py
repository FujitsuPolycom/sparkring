"""Build a native-wheel candidate from accepted source on a leased ARM64 host."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.images.upgrades.contracts import beneath, load_policy, read, require, sha  # noqa: E402
from runtime.images.upgrades.contract_rebind import rebind  # noqa: E402
from runtime.images.upgrades.io import checked, write_json  # noqa: E402
from runtime.images.upgrades.sources import tree_digest, native_digest  # noqa: E402


def docker(*args, seconds=120, limit=4 * 1024**2):
    return checked(
        ["docker", "--host", "unix:///var/run/docker.sock", *map(str, args)],
        seconds=seconds,
        limit=limit,
    )


def validate_recipe(recipe):
    require(
        isinstance(recipe, dict)
        and set(recipe)
        == {
            "schema",
            "architecture",
            "jobs",
            "cpus",
            "memory_bytes",
            "build_seconds",
            "torch_version",
            "network",
        },
        "Native recipe fields differ",
    )
    require(recipe["schema"] == "sparkring-native-recipe/v1", "Unknown native recipe")
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


def prepare_source_binding(policy, bundle, paths, parent, context):
    config = policy["foundation"].get("source_binding")
    if config is None:
        return None, None
    contract_path = beneath(policy["_root"], config["contract"])
    original_destination = "/opt/sparkring/contracts/" + contract_path.name
    active = set(parent.get("integration_contracts", {}))
    require(
        original_destination in active
        and parent["files"].get(original_destination)
        == sha(contract_path.read_bytes()),
        "Source binding does not name the active parent contract",
    )
    source = bundle["sources"]["vllm"]
    oracle = next(
        (
            item
            for item in source.get("oracles", [])
            if item["gate"] == config["oracle"]
        ),
        None,
    )
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
    }
    require(set(expected) <= set(record), "Native-cache compatibility fields missing")
    changed = sorted(key for key, value in expected.items() if record[key] != value)
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
        "distribution_version": "0.26.1rc0+sparkring.native."
        + bundle["input_sha256"][:12],
    }
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
            and compiled["source_trees"] == source_trees,
            "Compiler receipt identifies other inputs",
        )
    except BaseException:
        # The caller marks failed actions uncertain and requires explicit cleanup.
        # Retain the named compiler container so its state and logs remain inspectable.
        raise
    raw_parent = docker(
        "run",
        "--rm",
        "--runtime",
        "runc",
        "--network",
        "none",
        "--read-only",
        "--entrypoint",
        "/bin/cat",
        parent_id,
        "/opt/sparkring/receipts/candidate-installed.json",
        limit=64 * 1024**2,
    )
    parent_receipt = json.loads(raw_parent)
    require(
        parent_receipt["versions"]["torch"] == compiled["torch_version"],
        "Compiled wheel and foundation Torch ABI differ",
    )
    wheels = context / "wheels"
    wheels.mkdir()
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
        "compiler_descriptor_sha256": compiled["descriptor_sha256"],
        "source_trees": source_trees,
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
