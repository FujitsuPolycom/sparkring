"""Copy an exact local image through a fabric-tunneled loopback registry.

Run on rank 0. Registry traffic never uses an external publication destination.
The registry runs as the selected rank's SSH user and owns only a marked,
run-specific temporary directory. Model-serving containers are not modified.
"""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.images.upgrades.contracts import (  # noqa: E402
    Uncertain,
    load_policy,
    require,
)
from runtime.images.upgrades.execution import builder_lease_valid  # noqa: E402
from runtime.images.upgrades.hardware import Pair  # noqa: E402
from runtime.images.upgrades.io import write_json  # noqa: E402
from runtime.images.upgrades.maintenance_build import bound_document  # noqa: E402


def validate(config):
    require(
        config.get("schema") == "sparkring-image-transfer/v1"
        and set(config) - {"registry_rank"}
        == {
            "schema",
            "hosts",
            "hostnames",
            "gate_id",
            "fabric_peer",
            "host_key_alias",
            "registry_image_id",
            "temporary_parent",
            "port",
            "transfer_seconds",
        },
        "Unknown image-transfer configuration",
    )
    require(
        type(config.get("registry_rank", 1)) is int
        and config.get("registry_rank", 1) in (0, 1),
        "Registry storage must select rank 0 or rank 1",
    )
    require(
        len(config["hosts"]) == len(config["hostnames"]) == 2,
        "Image transfer requires two explicit ranks",
    )
    require(
        re.fullmatch(r"[a-z_][a-z0-9_-]*@[A-Za-z0-9.-]+", config["fabric_peer"])
        and re.fullmatch(r"[A-Za-z0-9.-]+", config["host_key_alias"]),
        "Invalid fabric SSH identity",
    )
    require(
        config["fabric_peer"].split("@")[0] == config["hosts"][1].split("@")[0],
        "Fabric and management SSH users differ",
    )
    require(
        re.fullmatch(r"sha256:[0-9a-f]{64}", config["registry_image_id"]),
        "An immutable, preinstalled registry image is required",
    )
    parent = PurePosixPath(config["temporary_parent"])
    require(
        parent.is_absolute()
        and ".." not in parent.parts
        and len(parent.parts) >= 4
        and str(parent) == config["temporary_parent"],
        "Transfer storage must be an explicit dedicated absolute directory",
    )
    require(
        not any(char in str(parent) for char in (",", "\n", "\r", "\x00")),
        "Transfer path cannot contain mount syntax or control characters",
    )
    require(
        type(config["port"]) is int
        and 1024 <= config["port"] <= 65535
        and type(config["transfer_seconds"]) is int
        and 30 <= config["transfer_seconds"] <= 3600,
        "Invalid bounded transfer port/deadline",
    )
    return config


def ssh_prefix(config):
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "HostKeyAlias=" + config["host_key_alias"],
    ]


def docker(*args):
    return ["docker", "--host", "unix:///var/run/docker.sock", *args]


def inspect(pair, rank, identifier):
    return json.loads(pair.call(rank, docker("inspect", identifier)))[0]


def image_info(pair, rank, image_id):
    value = json.loads(pair.call(rank, docker("image", "inspect", image_id)))[0]
    require(
        value["Id"] == image_id
        and value["Architecture"] == "arm64"
        and value["Os"] == "linux",
        "Transferred image identity or platform differs",
    )
    return value


def registry_command(config, *, name, path, user):
    require(re.fullmatch(r"[0-9]+:[0-9]+", user), "Invalid registry filesystem user")
    return docker(
        "create",
        "--name",
        name,
        "--label",
        "sparkring.upgrade.transfer=" + name,
        "--pull",
        "never",
        "--runtime",
        "runc",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        user,
        "--memory",
        "1g",
        "--cpus",
        "2",
        "--pids-limit",
        "128",
        "--tmpfs",
        "/tmp:rw,nosuid,size=67108864",
        "--publish",
        f"127.0.0.1:{config['port']}:5000",
        "--mount",
        f"type=bind,src={path},dst=/var/lib/registry",
        config["registry_image_id"],
    )


def temporary_directory(pair, config, name, *, remove=False):
    script = """
import json,os,pathlib,shutil,sys
parent=pathlib.Path(sys.argv[1]); name=sys.argv[2]; remove=sys.argv[3]=='remove'
assert parent.is_dir() and parent.resolve()==parent and not parent.is_symlink()
path=parent/name
assert path.parent==parent and path.resolve()==path and not path.is_symlink()
if remove:
    assert (path/'.owner').read_text()==name
    shutil.rmtree(path)
else:
    path.mkdir(mode=0o700)
    (path/'.owner').write_text(name)
print(json.dumps({'path':str(path),'user':str(os.getuid())+':'+str(os.getgid())}))
"""
    return json.loads(
        pair.call(
            config.get("registry_rank", 1),
            [
                "python3",
                "-c",
                script,
                config["temporary_parent"],
                name,
                "remove" if remove else "create",
            ],
        )
    )


def transfer(pair, config, image_id, run_id, output):
    validate(config)
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", image_id), "Exact image ID required")
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{1,35}", run_id), "Invalid transfer run ID")
    require(
        socket.gethostname() == config["hostnames"][0]
        and getpass.getuser() == config["hosts"][0].split("@")[0],
        "Image transfer must run as rank 0's leased SSH user",
    )
    registry_rank = config.get("registry_rank", 1)
    image_info(pair, 0, image_id)
    image_info(pair, registry_rank, config["registry_image_id"])
    # The approved management identity and fabric endpoint must reach the same host.
    fabric = pair.call(0, [*ssh_prefix(config), config["fabric_peer"], "hostname"])
    require(
        fabric.decode().strip() == config["hostnames"][1],
        "Fabric peer differs from rank 1",
    )
    name = "sr-upgrade-transfer-" + run_id
    tag = f"127.0.0.1:{config['port']}/sparkring/native:{image_id[7:]}"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "sparkring-image-transfer-result/v1",
        "run_id": run_id,
        "image_id": image_id,
        "rank1_verified": False,
        "registry": tag,
        "registry_rank": registry_rank,
        "external_publication": False,
        "failure": None,
        "cleanup_errors": [],
    }
    tunnel = None
    directory = container = None
    try:
        directory = temporary_directory(pair, config, name)
        write_json(output / "transfer.json", result, replace=True)
        pair.call(registry_rank, registry_command(config, name=name, **directory))
        container = inspect(pair, registry_rank, name)
        require(
            container["Image"] == config["registry_image_id"]
            and container["Config"]["Labels"].get("sparkring.upgrade.transfer") == name,
            "Created registry differs from transfer ownership",
        )
        result["registry_container_id"] = container["Id"]
        write_json(output / "transfer.json", result, replace=True)
        pair.call(registry_rank, docker("start", container["Id"]))
        port = config["port"]
        tunnel = subprocess.Popen(
            [
                *ssh_prefix(config),
                "-o",
                "ExitOnForwardFailure=yes",
                "-N",
                "-L" if registry_rank == 1 else "-R",
                f"127.0.0.1:{port}:127.0.0.1:{port}",
                config["fabric_peer"],
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        for _ in range(20):
            require(tunnel.poll() is None, "Fabric registry tunnel exited")
            if registry_rank == 0:
                # Probe through the reverse tunnel on the receiving host, not
                # directly against the sender's already-listening registry.
                probe = """import sys,urllib.request
try:
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
  'http://127.0.0.1:'+sys.argv[1]+'/v2/',timeout=2) as response:
  print(response.status)
except OSError:
 pass
"""
                ready = pair.call(1, ["python3", "-c", probe, str(port)], seconds=5)
                if ready.strip() == b"200":
                    break
                time.sleep(0.5)
                continue
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                    f"http://127.0.0.1:{port}/v2/", timeout=2
                ) as response:
                    require(response.status == 200, "Registry is not ready")
                break
            except OSError:
                time.sleep(0.5)
        else:
            require(False, "Registry startup deadline exceeded")
        pair.call(0, docker("tag", image_id, tag))
        print("Transferring image layers through the leased fabric tunnel.", flush=True)
        pair.call(0, docker("push", tag), seconds=config["transfer_seconds"])
        pair.call(1, docker("pull", tag), seconds=config["transfer_seconds"])
        received = json.loads(pair.call(1, docker("image", "inspect", tag)))[0]
        require(received["Id"] == image_id, "Receiving image hash differs")
        image_info(pair, 1, image_id)
        result["rank1_verified"] = True
    except BaseException as error:
        result["failure"] = str(error)
        raise
    finally:
        try:
            if tunnel is not None:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
                    tunnel.wait(timeout=10)
            if container is not None:
                actual = inspect(pair, registry_rank, container["Id"])
                require(
                    actual["Image"] == config["registry_image_id"]
                    and actual["Config"]["Labels"].get("sparkring.upgrade.transfer")
                    == name,
                    "Refusing cleanup of a registry with different ownership",
                )
                pair.call(
                    registry_rank, docker("stop", "--time", "30", container["Id"])
                )
                pair.call(registry_rank, docker("rm", container["Id"]))
            if directory is not None:
                # A failed create can leave a container even when inspect failed.
                ids = pair.call(
                    registry_rank,
                    docker(
                        "ps",
                        "--all",
                        "--quiet",
                        "--filter",
                        "name=^/" + name + "$",
                    ),
                )
                require(
                    not ids.strip(), "Registry container remains; retain its storage"
                )
                temporary_directory(pair, config, name, remove=True)
        except BaseException as error:
            result["cleanup_errors"].append(str(error))
        write_json(output / "transfer.json", result, replace=True)
        if result["cleanup_errors"]:
            raise Uncertain(
                "Image-transfer cleanup needs inspection: "
                + "; ".join(result["cleanup_errors"])
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "builder-lease", "hardware-lease", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("config", "approved-policy", "image-id", "run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(args.execute, "Image transfer requires explicit execution")
    policy = load_policy(args.policy)
    require(
        policy["_digest"] == args.approved_policy, "Approved transfer policy differs"
    )
    builder_lease_valid(args.builder_lease, policy)
    config = validate(bound_document(policy, args.config))
    pair = Pair(
        policy,
        args.hardware_lease,
        config["hosts"],
        run_id=args.run_id,
        gate_id=config["gate_id"],
        hostnames=config["hostnames"],
    )
    print(json.dumps(transfer(pair, config, args.image_id, args.run_id, args.output)))


if __name__ == "__main__":
    main()
