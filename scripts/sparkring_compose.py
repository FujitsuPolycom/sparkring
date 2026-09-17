"""Generate and coordinate per-host Compose deployments with shared plan receipts."""

from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import json
import ipaddress
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.common import compose, ports, profiles, qwen_flash_next  # noqa: E402
from scripts import deploy_engine  # noqa: E402

# Check repository inputs before importing host code. The controller provides
# the inventory; an installed host helper cannot attest to its own identity.
BOOTSTRAP = """import base64, hashlib, json, pathlib, sys, zlib
root = pathlib.Path(sys.argv[1]).resolve()
payload = json.loads(zlib.decompress(base64.b64decode(sys.argv[2], validate=True)))
for name, expected in payload['manifest']['inputs'].items():
    path = (root / name).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes().replace(b'\\r\\n', b'\\n')).hexdigest() != expected:
        raise SystemExit('Host repository differs: ' + name)
sys.path.insert(0, str(root))
from scripts.sparkring_compose import host_operation
host_operation(sys.argv[3], payload)
"""


def packed(value):
    return base64.b64encode(zlib.compress(compose.encoded(value).encode())).decode()


def unpacked(value):
    return json.loads(zlib.decompress(base64.b64decode(value, validate=True)))


def host_argv(manifest, files, rank, operation):
    payload = {"manifest": manifest, "files": files, "rank": rank["rank"]}
    # The mesh check reads root-owned marker executable and attachment records.
    prefix = ["sudo", "-n"] if "fabric" in rank and operation == "preflight" else []
    return prefix + [
        "python3",
        "-I",
        "-B",
        "-c",
        BOOTSTRAP,
        rank["repository"],
        packed(payload),
        operation,
    ]


def plan(manifest, files, operation):
    if operation not in ("check", "start", "stop"):
        raise ValueError("Unknown Compose lifecycle operation")
    ranks = manifest["site"]["ranks"]

    def phase(name, hosts, risk="read-only", verify=None):
        actions = []
        for rank in hosts:
            action = {
                "host": rank["host"],
                "argv": host_argv(manifest, files, rank, name),
                "risk": risk,
                "timeout": 1000 if name == "ready" else 300,
            }
            if verify:
                action["verify"] = {
                    "argv": host_argv(manifest, files, rank, verify),
                    "stdout": "ok",
                    "timeout": 300,
                }
            actions.append(action)
        return {"id": name, "actions": actions}

    if operation == "check":
        phases = [phase("preflight", ranks)]
    elif operation == "start":
        phases = [
            phase("preflight", ranks),
            phase("stage", ranks, "mutates-host", "staged"),
            phase("admit", ranks, "mutates-host", "admitted"),
            phase("create", ranks, "starts-model", "created"),
            phase("start-worker", ranks[1:], "starts-model", "running"),
            phase("start-api", ranks[:1], "starts-model", "running"),
            phase("ready", ranks[:1]),
            phase("running", ranks),
        ]
    else:
        phases = [phase("owned", ranks), phase("stop", ranks, "stops-model", "stopped")]
    return deploy_engine.seal_plan(
        {
            "schema": "sparkring-deploy-plan/v1",
            "deployment": manifest["id"],
            "operation": operation,
            "phases": phases,
        }
    )


def run(argv, **kwargs):
    if argv[0] == "docker" and argv[1:3] != ["--context", "default"]:
        argv = ["docker", "--context", "default", *argv[1:]]
    kwargs.setdefault("check", True)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("timeout", 240)
    return subprocess.run(argv, **kwargs)


def container(spec):
    # A daemon failure must not be confused with an absent container.
    result = run(
        ["docker", "container", "ls", "--all", "--no-trunc", "--format", "{{.Names}}"]
    )
    if spec.name not in result.stdout.splitlines():
        return None
    return json.loads(run(["docker", "container", "inspect", spec.name]).stdout)[0]


def check_project_containers(spec, *, owned_id=None):
    """Compose selects a project by labels even when a container was renamed."""
    compose.check_project_containers(spec.name, owned_id=owned_id, run=run)


def owned(spec, manifest):
    info = container(spec)
    labels = info.get("Config", {}).get("Labels", {}) if info else {}
    if (
        not info
        or labels.get(compose.LABEL) != manifest["id"]
        or labels.get("com.docker.compose.project") != spec.name
        or labels.get("com.docker.compose.service") != "model"
        or info.get("Image") != spec.image_id
    ):
        raise ValueError("Container is absent or does not belong to this deployment")
    config, host = info.get("Config", {}), info.get("HostConfig", {})
    environment = dict(
        value.split("=", 1) for value in config.get("Env", []) if "=" in value
    )
    mounts = sorted(
        (m.get("Source"), m.get("Destination"), m.get("RW"), m.get("Type"))
        for m in info.get("Mounts", [])
    )
    expected_mounts = sorted(
        (m.source, m.target, not m.read_only, "bind") for m in spec.mounts
    )
    devices = host.get("Devices") or []
    requests = host.get("DeviceRequests") or []
    health = ["CMD", *spec.health_command] if spec.health_command else ["NONE"]
    if (
        config.get("Entrypoint") != list(spec.entrypoint)
        or config.get("Cmd") != list(spec.command)
        or any(environment.get(key) != value for key, value in spec.environment.items())
        or mounts != expected_mounts
        or host.get("NetworkMode") != spec.network_mode
        or host.get("IpcMode") != spec.ipc_mode
        or host.get("Memory") != spec.memory
        or host.get("MemorySwap") != spec.memory_swap
        or host.get("RestartPolicy", {}).get("Name") != spec.restart_policy
        or host.get("Init") is not spec.init
        or host.get("Privileged") is not False
        or config.get("Healthcheck", {}).get("Test") != health
        or len(requests) != 1
        or requests[0].get("Driver") != "nvidia"
        or requests[0].get("Count") != spec.gpu_count
        or requests[0].get("Capabilities") != [["gpu"]]
        or sorted(
            (d.get("PathOnHost"), d.get("PathInContainer"), d.get("CgroupPermissions"))
            for d in devices
        )
        != sorted((d, d, "rwm") for d in spec.devices)
        or not any(
            limit.get("Name") == "memlock"
            and limit.get("Soft") == spec.memlock
            and limit.get("Hard") == spec.memlock
            for limit in (host.get("Ulimits") or [])
        )
    ):
        raise ValueError("Container settings differ from the deployment specification")
    return info


def stage_path(rank, manifest):
    base = Path(rank["deployment_root"])
    if not base.is_dir() or base.resolve() != base:
        raise ValueError(
            "Deployment root must already exist and have no symlink components"
        )
    target = base / manifest["id"]
    if target.resolve() != target:
        raise ValueError("Deployment directory cannot be a symlink")
    return target


def staged(target, expected):
    for name, text in expected.items():
        path = target / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != text.encode()
        ):
            raise ValueError("Staged deployment differs: " + name)


def require_idle_gpu():
    if run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).stdout.strip():
        raise ValueError("GPU already has a compute workload")


def preflight(rank, site, spec, image, profile, manifest):
    if sys.platform != "linux":
        raise ValueError("Compose serving requires Linux hosts")
    paths = [
        Path(rank[name]).resolve(strict=True)
        for name in ("model", "cache", "repository", "deployment_root")
    ]
    for i, path in enumerate(paths):
        if not path.is_dir():
            raise ValueError("Required host directory is absent")
        if any(
            path.is_relative_to(other) or other.is_relative_to(path)
            for other in paths[i + 1 :]
        ):
            raise ValueError("Host paths overlap after resolving symlinks")
    if not os.access(paths[1], os.W_OK):
        raise ValueError("Cache directory is not writable")
    qwen_flash_next.verify_model_paths(profile, paths[0], paths[1])
    if qwen_flash_next.node_count(profile) == 4:
        from runtime.common import qwen_mesh
        qwen_mesh.check(rank["fabric"], rank["rank"], rank["hcas"], rank["gid"], rank["host_ip"])
    if not Path("/dev/infiniband").is_dir():
        raise ValueError("RDMA device directory is absent")
    addresses = json.loads(
        run(["ip", "-j", "address", "show", "dev", rank["interface"]]).stdout
    )
    if rank["host_ip"] not in [
        entry.get("local") for item in addresses for entry in item.get("addr_info", [])
    ]:
        raise ValueError("Rank address is not assigned to its bootstrap interface")
    for hca in rank["hcas"]:
        port = Path("/sys/class/infiniband") / hca / "ports/1"
        if "ACTIVE" not in (port / "state").read_text():
            raise ValueError("RDMA port is not active: " + hca)
        gid = (port / "gids" / str(rank["gid"])).read_text().strip()
        if not gid or int(gid.replace(":", ""), 16) == 0:
            raise ValueError("Selected RDMA GID is empty: " + hca)
    info = json.loads(run(["docker", "image", "inspect", image]).stdout)[0]
    if (
        info.get("Id") != spec.image_id
        or info.get("Os") != "linux"
        or info.get("Architecture") != "arm64"
    ):
        raise ValueError(
            "Pull the registered Linux ARM64 image digest before deployment"
        )
    present = container(spec)
    if present is not None:
        owned(spec, manifest)
    check_project_containers(spec, owned_id=present["Id"] if present else None)
    gpus = run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]
    ).stdout.splitlines()
    if len(gpus) != 1:
        raise ValueError("This adapter requires one GPU per host")
    if not (present and present["State"].get("Running")):
        require_idle_gpu()
    if rank["rank"] == 0 and not (present and present["State"].get("Running")):
        for flag in ("--port", "--master-port"):
            port = int(spec.command[spec.command.index(flag) + 1])
            bind_address = (
                spec.command[spec.command.index("--host") + 1]
                if flag == "--port"
                else (
                    "::"
                    if ipaddress.ip_address(site["master"]).version == 6
                    else "0.0.0.0"
                )
            )
            ports.check_tcp_bind(bind_address, port)
    compose.check_equivalence(spec, image, compose.compose_text(spec, image), run=run)


def wait_ready(spec, manifest, *, seconds=900, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + seconds
    while clock() < deadline:
        info = owned(spec, manifest)
        state = info["State"]
        if not state.get("Running"):
            raise ValueError("API container exited before readiness")
        health = state.get("Health", {}).get("Status")
        if health == "healthy":
            return
        if health == "unhealthy":
            raise ValueError("API container health check failed")
        sleep(2)
    raise ValueError("API health deadline exceeded; inspect deployment logs")


def host_operation(operation, payload):
    manifest = payload["manifest"]
    expected, files = compose.build(manifest["profile"], manifest["site"], **compose.selection_options(manifest))
    if manifest != expected or files != payload["files"]:
        raise ValueError("Host/controller configuration differs")
    number = payload["rank"]
    rank = manifest["site"]["ranks"][number]
    specs, image = compose.specifications(manifest["profile"], manifest["site"], **compose.selection_options(manifest))
    spec = replace(
        specs[number],
        labels={compose.LABEL: manifest["id"], "io.sparkring.rank": str(number)},
    )
    metadata, _ = profiles.load(manifest["profile"])
    profile = qwen_flash_next.read(ROOT / metadata["configuration"]["path"])
    target = stage_path(rank, manifest)
    exports = {
        "compose.yaml": files[f"rank{number}/compose.yaml"],
        "deployment.json": compose.encoded(manifest),
        "container.json": files[f"rank{number}/container.json"],
    }
    command = compose.compose_command(spec.name, target / "compose.yaml")
    if operation == "preflight":
        preflight(rank, manifest["site"], spec, image, profile, manifest)
    elif operation == "admit":
        # Verification containers have no GPU, network or host mounts. This is
        # deliberately separate from the read-only host preflight.
        staged(target, exports)

        def image_run(argv, **kwargs):
            kwargs.setdefault("text", False)
            return run(argv, **kwargs)

        result = qwen_flash_next.verify_image(
            spec.image_id,
            cache_enabled=profile.get("image_extension") == "lil-r37-cache64",
            feature_enabled=profile.get("image_extension") == "lil-r37-shared",
            local_source_extension=manifest.get("local_source_extension"),
            run=image_run,
        )
        receipt = {
            "deployment": manifest["id"],
            "image_id": spec.image_id,
            "verification": result,
        }
        path = target / "admission.json"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(compose.encoded(receipt))
        path.chmod(0o600)
    elif operation == "admitted":
        receipt = qwen_flash_next.read(target / "admission.json")
        info = json.loads(run(["docker", "image", "inspect", image]).stdout)[0]
        if (
            receipt["deployment"] != manifest["id"]
            or receipt["image_id"] != spec.image_id
            or info["Id"] != spec.image_id
        ):
            raise ValueError("Image admission no longer matches the deployment")
    elif operation == "stage":
        target.mkdir(mode=0o700, exist_ok=False)
        for name, text in exports.items():
            path = target / name
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            path.chmod(0o600)
    elif operation == "staged":
        staged(target, exports)
    elif operation == "create":
        staged(target, exports)
        if container(spec) is not None:
            raise ValueError(
                "Container name already exists; refusing adoption or recreation"
            )
        check_project_containers(spec)
        compose.check_equivalence(spec, image, exports["compose.yaml"], run=run)
        run(command + ["create", "--no-build", "--no-recreate", "--pull", "never", "model"])
    elif operation == "created":
        if owned(spec, manifest)["State"]["Status"] not in ("created", "running"):
            raise ValueError("Created container has exited or changed state")
    elif operation in ("start-worker", "start-api"):
        staged(target, exports)
        if owned(spec, manifest)["State"]["Status"] != "created":
            raise ValueError(
                "Start requires the exact stopped container created by this plan"
            )
        # Creation and image admission can finish well before startup or resume.
        require_idle_gpu()
        run(command + ["start", "model"])
    elif operation == "running":
        if not owned(spec, manifest)["State"].get("Running"):
            raise ValueError("Rank container is not running")
    elif operation == "ready":
        wait_ready(spec, manifest)
    elif operation == "owned":
        owned(spec, manifest)
    elif operation == "stop":
        # Stop by the inspected immutable container ID; never remove containers,
        # volumes, cache files or networks as part of coordinated stop.
        info = owned(spec, manifest)
        run(["docker", "stop", "--time", "60", info["Id"]])
    elif operation == "stopped":
        if owned(spec, manifest)["State"].get("Running"):
            raise ValueError("Rank container is still running")
    else:
        raise ValueError("Unsupported host operation")
    print("ok")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring compose")
    sub = parser.add_subparsers(dest="operation", required=True)
    render = sub.add_parser(
        "render", help="generate a private deployment without contacting hosts"
    )
    render.add_argument("profile")
    render.add_argument("--site", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--local-image-id", help="pin a source-equivalent local Development rebuild; published releases reject overrides")
    render.add_argument("--local-source-extension", help="select a registered local source-extension test instead of the public image")
    render.add_argument("--local-kv-cache-gib", type=int, help="select the local TP4 40 GiB KV alternative; requires a source extension")
    render.add_argument("--local-master-port", type=int, help="isolated source-extension test bootstrap port")
    check = sub.add_parser(
        "check", help="check canonical inputs and resolved Compose equivalence"
    )
    check.add_argument("--deployment", type=Path, required=True)
    check.add_argument(
        "--hosts", action="store_true", help="also run read-only checks over SSH"
    )
    for operation in ("start", "stop"):
        command = sub.add_parser(
            operation, help="coordinate hosts using a reviewed, hashed plan"
        )
        command.add_argument("--deployment", type=Path, required=True)
        command.add_argument(
            "--approve", help="execute the plan identified by this exact SHA-256"
        )
        command.add_argument(
            "--resume",
            action="store_true",
            help="recheck a receipt and resume safe actions",
        )
    args = parser.parse_args(argv)
    try:
        if args.operation == "render":
            manifest = compose.render(
                args.profile, compose.read_site(args.site), args.output,
                local_image_id=args.local_image_id,
                local_source_extension=args.local_source_extension,
                local_kv_cache_gib=args.local_kv_cache_gib,
                local_master_port=args.local_master_port,
            )
            print(
                compose.encoded({"deployment": str(args.output), "id": manifest["id"]})
            )
            return 0
        manifest = compose.check(args.deployment)
        _, files = compose.load_deployment(args.deployment)
        if args.operation == "check" and not args.hosts:
            print(
                "Canonical inputs and resolved Compose settings agree. No serving qualification is implied."
            )
            return 0
        document = plan(manifest, files, args.operation)
        if args.operation != "check" and not args.approve:
            plan_path = args.deployment / (args.operation + "-plan.json")
            plan_path.write_text(
                compose.encoded(document), encoding="utf-8", newline="\n"
            )
            plan_path.chmod(0o600)
            print(
                compose.encoded(
                    {
                        "plan": str(plan_path),
                        "sha256": document["sha256"],
                        "profile": manifest["profile"],
                        "image": manifest["image"],
                        "site": manifest["site"],
                        "phases": [
                            {
                                "operation": phase["id"],
                                "hosts": [a["host"] for a in phase["actions"]],
                                "risk": phase["actions"][0]["risk"],
                            }
                            for phase in document["phases"]
                        ],
                    }
                )
            )
            print("Review this plan, then repeat with --approve " + document["sha256"])
            return 0
        if args.operation == "check":
            runner = deploy_engine.CommandRunner()
            for action in document["phases"][0]["actions"]:
                result = runner(action["host"], action["argv"], action["timeout"])
                if not deploy_engine.verified(result, {"stdout": "ok"}):
                    raise ValueError(
                        action["host"] + ": host preflight failed: " + result["stderr"]
                    )
            print(
                "Host prerequisites passed; image payload admission occurs during start."
            )
            return 0
        receipt = args.deployment / (args.operation + "-receipt.json")
        deploy_engine.execute_plan(
            document,
            receipt,
            args.approve,
            resume=args.resume,
            allow_model_actions=True,
        )
        print("Completed. Receipt: " + str(receipt))
        return 0
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print("Compose error: " + str(exc), file=sys.stderr)
        if hasattr(args, "deployment"):
            receipt = args.deployment / (args.operation + "-receipt.json")
            if receipt.is_file():
                print("Inspect the per-host failure in " + str(receipt), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
