"""Create a pullable one-layer copy of an admitted native SparkRing image.

This local operator step never pushes an image. It exports a stopped container,
imports the filesystem as one layer, preserves the supported runtime
configuration, and verifies the installed receipt before and after conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


DOCKER = ("docker", "--host", "unix:///var/run/docker.sock")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
TARGET = re.compile(r"[a-z0-9][a-z0-9._/-]{0,191}:[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
CONTAINER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")
HEADROOM = 10 * 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def output(*args):
    return subprocess.check_output([*DOCKER, *args], text=True).strip()


def inspect(reference):
    value = json.loads(output("image", "inspect", reference))
    require(isinstance(value, list) and len(value) == 1, "Expected one image")
    return value[0]


def quoted(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def configuration_changes(config):
    """Encode every Docker-import configuration field supported by this tool."""
    for name in ("Healthcheck", "OnBuild", "Shell", "Volumes"):
        require(not config.get(name), "Flattening does not support Config." + name)
    changes = []
    for assignment in config.get("Env") or []:
        require("=" in assignment, "Invalid environment assignment")
        key, value = assignment.split("=", 1)
        changes.append("ENV " + key + "=" + quoted(value))
    for key, value in sorted((config.get("Labels") or {}).items()):
        changes.append("LABEL " + key + "=" + quoted(value))
    if config.get("Entrypoint") is not None:
        changes.append("ENTRYPOINT " + quoted(config["Entrypoint"]))
    if config.get("Cmd") is not None:
        changes.append("CMD " + quoted(config["Cmd"]))
    if config.get("WorkingDir"):
        changes.append("WORKDIR " + config["WorkingDir"])
    if config.get("User"):
        changes.append("USER " + config["User"])
    if config.get("StopSignal"):
        changes.append("STOPSIGNAL " + config["StopSignal"])
    for port in sorted(config.get("ExposedPorts") or {}):
        changes.append("EXPOSE " + port)
    return changes


def runtime_configuration(config):
    names = (
        "Env",
        "Labels",
        "Entrypoint",
        "Cmd",
        "WorkingDir",
        "User",
        "StopSignal",
        "ExposedPorts",
    )
    return {name: config.get(name) for name in names}


def installed_verification(image):
    raw = subprocess.check_output(
        [
            *DOCKER,
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "/opt/venv/bin/python",
            image,
            "/opt/sparkring/bin/native-image.py",
            "verify",
        ],
        text=True,
    )
    value = json.loads(raw.splitlines()[-1])
    require(
        value.get("schema") == "sparkring-native-verification/v1"
        and type(value.get("files_verified")) is int
        and value["files_verified"] > 0,
        "Installed runtime verification did not pass",
    )
    return value


def docker_root_free_bytes():
    root = json.loads(output("info", "--format", "{{json .DockerRootDir}}"))
    require(isinstance(root, str) and os.path.isabs(root), "Invalid Docker root")
    return shutil.disk_usage(root).free


def stream_flatten(container, target, changes):
    export = subprocess.Popen([*DOCKER, "export", container], stdout=subprocess.PIPE)
    try:
        command = [*DOCKER, "import", "--platform", "linux/arm64"]
        for change in changes:
            command.extend(("--change", change))
        command.extend(("-", target))
        require(export.stdout is not None, "Docker export did not expose stdout")
        subprocess.run(command, stdin=export.stdout, check=True)
        export.stdout.close()
        require(export.wait() == 0, "Docker export failed")
    finally:
        if export.poll() is None:
            export.terminate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--target-tag", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(args.execute, "Flattening requires explicit execution")
    require(IMAGE_ID.fullmatch(args.source_image), "Source must be an immutable image ID")
    require(TARGET.fullmatch(args.target_tag), "Target must be a versioned local tag")
    require(CONTAINER.fullmatch(args.container), "Invalid temporary container name")
    require(not args.output.exists(), "Output directory already exists")
    args.output.mkdir(parents=True)

    before = inspect(args.source_image)
    require(
        before.get("Id") == args.source_image
        and before.get("Os") == "linux"
        and before.get("Architecture") == "arm64",
        "Source must be the selected Linux ARM64 image",
    )
    require(
        not output("ps", "-aq", "--filter", "name=^" + args.container + "$"),
        "Temporary container already exists",
    )
    try:
        inspect(args.target_tag)
    except subprocess.CalledProcessError:
        pass
    else:
        raise ValueError("Target tag already exists")
    minimum = int(before["Size"]) * 2 + HEADROOM
    free = docker_root_free_bytes()
    require(free >= minimum, "Docker root lacks flattening headroom")
    changes = configuration_changes(before["Config"])
    source_verification = installed_verification(args.source_image)
    container = output("create", "--name", args.container, args.source_image)
    require(re.fullmatch(r"[0-9a-f]{64}", container), "Docker did not create one container")
    try:
        stream_flatten(container, args.target_tag, changes)
    finally:
        subprocess.run([*DOCKER, "rm", "-f", container], check=False)
    after = inspect(args.target_tag)
    require(
        len(after.get("RootFS", {}).get("Layers", [])) == 1,
        "Flattened image must contain one root filesystem layer",
    )
    require(
        after.get("Os") == "linux"
        and after.get("Architecture") == "arm64"
        and runtime_configuration(after["Config"])
        == runtime_configuration(before["Config"]),
        "Flattened runtime configuration differs",
    )
    target_verification = installed_verification(after["Id"])
    require(target_verification == source_verification, "Installed verification differs")
    proof = {
        "schema": "sparkring-flattened-image/v1",
        "source_image_id": args.source_image,
        "source_rootfs_layers": len(before["RootFS"]["Layers"]),
        "target_image_id": after["Id"],
        "target_tag": args.target_tag,
        "target_rootfs_layers": 1,
        "runtime_configuration_equal": True,
        "installed_verification": target_verification,
        "minimum_free_bytes": minimum,
        "observed_free_bytes": free,
        "external_publication": False,
    }
    data = json.dumps(proof, indent=2, sort_keys=True) + "\n"
    (args.output / "flatten.json").write_text(data)
    print(
        json.dumps(
            {
                "image_id": after["Id"],
                "proof": str(args.output / "flatten.json"),
                "proof_sha256": hashlib.sha256(data.encode()).hexdigest(),
                "external_publication": False,
            }
        )
    )


if __name__ == "__main__":
    main()
