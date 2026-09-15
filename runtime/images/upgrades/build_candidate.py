"""Reference source-overlay build adapter; native-input drift is rejected.

Use a separately reviewed native-build recipe when compiled inputs differ.
This adapter preserves inherited transport/cache/feature bytes, but does not
transfer their source-interface or serving qualification to the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.images.upgrades.contracts import load_policy, read, require  # noqa: E402
from runtime.images.upgrades.io import checked, write_json  # noqa: E402
from runtime.images.upgrades.sources import git, inventory, tree_digest  # noqa: E402


def digest(data):
    return hashlib.sha256(data).hexdigest()


def docker(*args, seconds=600, limit=1024 * 1024):
    return checked(
        ["docker", "--host", "unix:///var/run/docker.sock", *args],
        seconds=seconds,
        limit=limit,
    )


def build(policy_path, bundle_path, output, result_path):
    policy, bundle = load_policy(policy_path), read(bundle_path)
    require(
        bundle["policy_sha256"] == policy["_digest"], "Build policy identity differs"
    )
    require(
        set(bundle["sources"]) == {"vllm", "b12x"},
        "Reference overlay builder supports vLLM/B12X; register a reviewed adapter for another runtime family",
    )
    require(
        not any(s["native_changed"] for s in bundle["sources"].values()),
        "Native/build inputs changed; source-overlay reuse is refused",
    )
    foundation = policy["foundation"]
    parent = foundation["image"]
    require(
        "@sha256:" in parent and foundation["image_id"].startswith("sha256:"),
        "Immutable foundation reference and ID required",
    )
    info = json.loads(docker("image", "inspect", foundation["image_id"]))[0]
    require(
        info["Id"] == foundation["image_id"]
        and info["Os"] + "/" + info["Architecture"] == policy["platform"],
        "Foundation platform/identity differs",
    )
    require(
        parent in info.get("RepoDigests", []),
        "Foundation is not locally registered under the pinned digest; pull it explicitly",
    )
    parent_raw = docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--pull",
        "never",
        "--entrypoint",
        "/bin/cat",
        foundation["image_id"],
        "/opt/sparkring/receipts/candidate-installed.json",
        limit=64 * 1024 * 1024,
    )
    parent_receipt = json.loads(parent_raw)
    require(parent_receipt.get("files"), "Foundation inventory is empty")
    feature_raw = docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--pull",
        "never",
        "--entrypoint",
        "/bin/cat",
        foundation["image_id"],
        "/opt/sparkring/features/capabilities.json",
    )
    features = json.loads(feature_raw)
    require(
        set(policy.get("required_features", [])) <= set(features.get("features", {})),
        "Required feature is absent from the foundation capability manifest",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "parent-installed.json").write_bytes(parent_raw)
    descriptor = {
        "schema": "sparkring-candidate-image/v1",
        "composition_id": policy["name"] + "-" + bundle["input_sha256"][:12],
        "parent_image_id": foundation["image_id"],
        "distribution_version": parent_receipt["versions"]["vllm"].split("+")[0]
        + "+sparkring.upgrade."
        + bundle["input_sha256"][:12],
        "components": {},
        "parent_authored_files": {},
        "integration_contracts": {},
    }
    for name, record in bundle["sources"].items():
        source = Path(record["candidate_path"])
        require(
            tree_digest(source) == record["candidate_tree_sha256"],
            "Accepted source snapshot changed",
        )
        git(source, "add", "--all")
        tree = git(source, "write-tree").decode().strip()
        archive = output / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for relative in inventory(source):
                path = source / relative
                require(not path.is_symlink(), "Source archive cannot contain symlinks")
                stream.add(path, arcname=relative, recursive=False)
        descriptor["components"][name] = {
            "base_commit": record["target_commit"],
            "tree": tree,
            "archive": archive.name,
            "archive_sha256": digest(archive.read_bytes()),
            "patch_sha256": record["patch_sha256"],
        }
        prefix = "/opt/venv/lib/python3.12/site-packages/"
        descriptor["parent_authored_files"][name] = [
            p[len(prefix) :]
            for p in parent_receipt["files"]
            if p.startswith(prefix + name + "/")
            and not any(
                s in Path(p).name for s in (".so", ".dll", ".dylib", ".a", ".o")
            )
        ]
    write_json(output / "descriptor.json", descriptor)
    shutil.copyfile(
        ROOT / "runtime/images/candidate_image.py", output / "candidate_image.py"
    )
    shutil.copyfile(ROOT / "runtime/images/Dockerfile.candidate", output / "Dockerfile")
    tag = (
        "local/sparkring-upgrade:"
        + policy["name"]
        + "-"
        + Path(result_path).parents[1].name.lower()
    )
    docker(
        "build",
        "--network",
        "none",
        "--pull=false",
        "--build-arg",
        "PARENT=" + parent,
        "--platform",
        policy["platform"],
        "--tag",
        tag,
        str(output),
        seconds=policy["budgets"]["command_seconds"],
        limit=policy["budgets"]["output_bytes"],
    )
    image = json.loads(docker("image", "inspect", tag))[0]
    verification = json.loads(
        docker(
            "run", "--rm", "--network", "none", "--pull", "never", image["Id"], "verify"
        )
    )
    require(
        verification.get("files_verified", 0) > 0
        and verification.get("serving_qualified") is False,
        "Installed verification is incomplete",
    )
    result = {
        "schema": "sparkring-upgrade-build/v1",
        "input_sha256": bundle["input_sha256"],
        "image_id": image["Id"],
        "image_tag": tag,
        "platform": policy["platform"],
        "installed_verified": True,
        "features": sorted(features["features"]),
        "features_scope": "Inherited inventory preserved; activation and source contracts require independent gates.",
        "source_trees": {
            name: s["candidate_tree_sha256"] for name, s in bundle["sources"].items()
        },
        "verification": verification,
        "native_rebuilt": False,
    }
    write_json(result_path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("policy", "bundle", "output", "result"):
        parser.add_argument("--" + field, required=True, type=Path)
    args = parser.parse_args()
    build(args.policy, args.bundle, args.output, args.result)
