"""Rank-local installer operations, invoked only by the approved controller plan."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from runtime.common import compose, glm_native_candidate, installer, native_candidate, profiles, qwen_flash_next, setup
from runtime.common.container_spec import expected_inspection
from scripts import deploy_engine

POSIX_STATS = os.name == "posix"


def run(argv, **kwargs):
    if argv[0] == "docker":
        argv = ["docker", "--context", "default", *argv[1:]]
    kwargs.setdefault("check", True)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("timeout", 7200)
    kwargs.setdefault("text", True)
    return subprocess.run(argv, **kwargs)


def plain(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Installer paths must be absolute without symlink components")
    return path


def image_info(lock):
    return json.loads(run(["docker", "image", "inspect", lock["selection"]["image_id"]]).stdout)[0]


def admit_image(lock):
    card = lock["selection"]
    def native_run(argv, **kwargs):
        # Native admission deliberately reads the installed receipt as bytes.
        kwargs.setdefault("text", False)
        return run(argv, **kwargs)
    if card["profile"].startswith("glm53-"):
        return glm_native_candidate.observe(card["image_id"], card["release"], run=native_run)
    return native_candidate.verify_image(card["image_id"], card["release"], run=native_run)


def model_files(path):
    path = plain(path)
    result = {}
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path)
        if relative.parts[0] in (".cache", ".git"):
            continue
        if item.is_symlink():
            raise ValueError("Use a complete local checkpoint directory, not external symlinks")
        if item.is_file():
            digest = hashlib.sha256()
            with item.open("rb") as stream:
                while chunk := stream.read(8 << 20):
                    digest.update(chunk)
            result[relative.as_posix()] = digest.hexdigest()
    if "config.json" not in result or "model.safetensors.index.json" not in result:
        raise ValueError("Checkpoint configuration/index are absent")
    index = profiles.read_json(path / "model.safetensors.index.json")
    shards = set(index["weight_map"].values())
    if not shards or not shards <= result.keys():
        raise ValueError("Checkpoint is missing indexed weight shards")
    return result


def model_file_stats(path):
    """Linux inode/change-time fingerprints for an already checksum-verified tree."""
    root = plain(path)
    result = {}
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if relative.parts[0] in (".cache", ".git"):
            continue
        if item.is_symlink():
            raise ValueError("Checkpoint verification cannot follow symlinks")
        if item.is_file():
            value = item.stat()
            result[relative.as_posix()] = [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
    return result


def verify_model(lock, row, receipt_path, *, receipt=None, measured=None):
    card = lock["selection"]
    receipt = profiles.read_json(receipt_path) if receipt is None else receipt
    if measured is None:
        before = model_file_stats(row["model"])
        if POSIX_STATS and receipt.get("file_stats") == before:
            measured = receipt["files"]
        else:
            measured = model_files(row["model"])
            if model_file_stats(row["model"]) != before:
                raise ValueError("Checkpoint changed during checksum verification")
    if (receipt["repository"] != card["model_repository"] or receipt["revision"] != card["model_revision"]
            or receipt["path"] != row["model"] or receipt["files"] != measured):
        raise ValueError("Checkpoint differs from its recorded revision/files")
    model = installer.checkpoint_contract(card)
    for filename, key in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
        if receipt["files"][filename] != model[key]:
            raise ValueError("Checkpoint metadata differs from the selected profile: " + filename)
    if card["profile"] in compose.SUPPORTED:
        sums = profiles.ROOT / "profiles/qwen38-flash-next-tp2/SHA256SUMS"
        for line in sums.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            if receipt["files"].get(name.lstrip("*")) != digest:
                raise ValueError("Checkpoint shard checksum differs: " + name)
    return receipt


def container(spec):
    names = run(["docker", "container", "ls", "--all", "--format", "{{.Names}}" ]).stdout.splitlines()
    return json.loads(run(["docker", "inspect", spec.name]).stdout)[0] if spec.name in names else None


def owned(spec, info, image):
    if info is None:
        raise ValueError("Deployment container is absent")
    expected = expected_inspection(spec, image, backend="compose")
    config, host = info["Config"], info["HostConfig"]
    host = dict(host)
    # Docker uses null for absent optional lists. NVIDIA-backed containers can
    # also receive label=disable on hosts where SELinux is not an active policy.
    for field in ("CapAdd", "SecurityOpt"):
        if host.get(field) is None and expected["host_config"][field] == []:
            host[field] = []
    if host.get("SecurityOpt") == ["label=disable"] and not expected["host_config"]["SecurityOpt"]:
        options = json.loads(run(["docker", "info", "--format", "{{json .SecurityOptions}}"] ).stdout)
        if isinstance(options, list) and not any("selinux" in str(value).lower() for value in options):
            host["SecurityOpt"] = []
    environment = dict(item.split("=", 1) for item in config.get("Env", []) if "=" in item)
    mounts = {item["Destination"]: {k: item[k] for k in ("Source", "Type", "RW")} for item in info["Mounts"]}
    if (info["Image"] != spec.image_id or config.get("Cmd") != expected["cmd"]
            or config.get("Entrypoint") != expected["entrypoint"] or environment != expected["env"]
            or mounts != expected["mounts"] or not deploy_engine.matches(config.get("Labels", {}), expected["labels"])
            or config.get("Healthcheck") != expected["healthcheck"]
            or config.get("WorkingDir", "") != expected["working_dir"] or config.get("User", "") != expected["user"]
            or not deploy_engine.matches(host, expected["host_config"])
            or len(host.get("Devices") or []) != len(spec.devices)
            or len(host.get("DeviceRequests") or []) != 1):
        raise ValueError("Existing container differs from this locked deployment; refusing adoption")
    compose.check_project_containers(spec.name, owned_id=info["Id"], run=run)
    return info


def require_idle():
    if run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).stdout.strip():
        raise ValueError("GPU has a workload; stop only the intended workload before retrying")


def http_json(port, path, body=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
        return json.loads(payload) if payload.strip() else {}


def perform(operation, lock, number):
    installer.validate(lock)
    row = lock["site"]["ranks"][number]
    workspace = plain(lock["site"]["workspace"])
    owner = profiles.read_json(workspace / ".installer-owner.json")
    if owner != {"deployment": lock["id"]}:
        raise ValueError("Workspace belongs to another deployment")
    state = plain(workspace / "installer")
    if operation in ("image", "model", "create"):
        state.mkdir(mode=0o700, exist_ok=True)
    card = lock["selection"]
    image_receipt = state / "image.json"
    model_receipt = state / "model.json"
    if operation.startswith("mesh-"):
        from runtime.host import native_mesh
        if operation == "mesh-prepare":
            return native_mesh.prepare_local(lock, number)
        if operation == "mesh-prepared":
            value = lock["site_input"]["native_mesh"]
            owner, _ = native_mesh.modules()
            owner.external_marker_attestation(binary=Path(value["site"]["marker_binary"]))
            return {"ok": True}
        if operation == "mesh-install-local":
            import sys
            return native_mesh.install_local(lock, number, json.load(sys.stdin))
        return native_mesh.operate_local(lock, number, operation)

    if operation == "image":
        run(["docker", "info"])
        ids = run(["docker", "image", "ls", "--quiet", "--no-trunc"]).stdout.splitlines()
        if card["image_id"] not in ids:
            docker_path = run(["docker", "info", "--format", "{{.DockerRootDir}}"]).stdout.strip()
            budget = setup.storage_plan(card, model_path=row["model"], cache_path=row["cache"],
                                        docker_path=docker_path, reuse_model=row["reuse_verified_model"])
            if not budget["passed"]:
                raise ValueError("Insufficient space for image/checkpoint/cache: " + json.dumps(budget["filesystems"]))
            run(["docker", "pull", "--platform", "linux/arm64", card["image_reference"]])
        receipt = admit_image(lock)
        deploy_engine.save_receipt(image_receipt, receipt)
        return {"ok": True}
    if operation == "image-check":
        if admit_image(lock) != profiles.read_json(image_receipt):
            raise ValueError("Image verification differs from the saved receipt")
        return {"ok": True}
    if operation in ("model", "model-check"):
        if operation == "model-check" or model_receipt.exists():
            verify_model(lock, row, model_receipt)
            return {"ok": True}
        model, cache = plain(row["model"]), plain(row["cache"])
        docker_path = run(["docker", "info", "--format", "{{.DockerRootDir}}"]).stdout.strip()
        report = setup.storage_plan(card, model_path=model, cache_path=cache, docker_path=docker_path,
                                     reuse_model=row["reuse_verified_model"], reuse_image=True)
        if not report["passed"]:
            raise ValueError("Insufficient destination storage: " + json.dumps(report["filesystems"]))
        if model.exists() and any(model.iterdir()) and not row["reuse_verified_model"]:
            raise ValueError("Nonempty model path has no installer receipt; explicitly declare an independently verified copy or choose a fresh path")
        model.mkdir(parents=True, exist_ok=True)
        if lock["backend"] != "glm-managed":
            cache.mkdir(parents=True, exist_ok=True)
        if not row["reuse_verified_model"]:
            code = "from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],local_dir='/model')"
            run(["docker", "run", "--rm", "--pull", "never", "--runtime", "runc", "--user", f"{os.getuid()}:{os.getgid()}",
                 "--env", "HF_HOME=/tmp/huggingface", "--mount", f"type=bind,src={model},dst=/model",
                 "--entrypoint", "/opt/venv/bin/python", card["image_id"], "-c", code,
                 card["model_repository"], card["model_revision"]])
        before = model_file_stats(model)
        hashes = model_files(model)
        if model_file_stats(model) != before:
            raise ValueError("Checkpoint changed while preparing its checksum receipt")
        receipt = {"repository": card["model_repository"], "revision": card["model_revision"],
                   "path": row["model"], "files": hashes, "file_stats": before,
                   "origin": "operator-declared-verified-copy" if row["reuse_verified_model"] else "pinned-hub-download"}
        verify_model(lock, row, model_receipt, receipt=receipt, measured=receipt["files"])
        deploy_engine.save_receipt(model_receipt, receipt)
        return {"ok": True}

    if lock["backend"] == "glm-managed":
        if operation == "receipt":
            return profiles.read_json(image_receipt)
        if operation == "smoke":
            connection = installer.connection(lock)
            name, port = connection["model"], connection["port"]
        else:
            raise ValueError("Managed GLM uses the existing host lifecycle coordinator")
    else:
        spec = installer.specifications(lock, only_rank=number)[0]
        image = image_info(lock)
        info = container(spec)
        if info:
            owned(spec, info, image)
        if operation == "status":
            return {"rank": number, "present": info is not None,
                    "running": bool(info and info["State"].get("Running")),
                    "health": info["State"].get("Health", {}).get("Status") if info else None}
        if operation == "owned":
            if info:
                owned(spec, info, image)
            return {"ok": True}
        if operation in ("stop", "stopped"):
            if operation == "stop" and info and info["State"].get("Running"):
                run(["docker", "stop", "--time", "60", info["Id"]])
            elif operation == "stopped" and info and info["State"].get("Running"):
                raise ValueError("Container is still running")
            return {"ok": True}
        if operation == "running":
            if not info or not info["State"].get("Running"):
                raise ValueError("Rank is not running")
            return {"ok": True}
        if operation == "created":
            owned(spec, info, image)
            return {"ok": True}
        if operation == "container-record":
            owned(spec, info, image)
            if info["State"].get("Running"):
                raise ValueError("Native installation requires stopped containers")
            return {"Id": info["Id"], "Image": info["Image"], "Name": info["Name"],
                    "State": {"Running": False}, "HostConfig": {"RestartPolicy": info["HostConfig"]["RestartPolicy"]},
                    "Config": {"Env": info["Config"].get("Env", [])}}
        if operation in ("preflight", "create", "start"):
            receipt = admit_image(lock)
            verify_model(lock, row, model_receipt)
            if card["profile"].startswith("glm53-"):
                actual = installer.specifications(lock, receipt=receipt, local=True, only_rank=number)[0]
                if actual != spec:
                    raise ValueError("Observed native-image settings differ from the planned Compose file")
            else:
                metadata, _ = profiles.load(card["profile"])
                profile = profiles.read_json(profiles.local_path(metadata["configuration"]["path"]))
                qwen_flash_next.verify_model_paths(profile, Path(row["model"]), Path(row["cache"]))
            if card["nodes"] == 4 and not (operation == "create" and "native_mesh" in lock["site_input"]):
                from runtime.common import qwen_mesh
                qwen_mesh.check(row["fabric"], number, row["hcas"], row["gid"], row["host_ip"])
            if not (info and info["State"].get("Running")):
                require_idle()
            compose.check_project_containers(spec.name, owned_id=info["Id"] if info else None, run=run)
            text = compose.compose_text(spec, card["image_reference"])
            compose.check_equivalence(spec, card["image_reference"], text, run=run)
            target = plain(Path(row["deployment_root"]) / lock["id"])
            if operation == "create":
                target.mkdir(parents=True, exist_ok=True)
                path = plain(target / "compose.yaml")
                if path.exists() and path.read_text() != text:
                    raise ValueError("Saved Compose file was edited; reinitialize explicitly")
                if not path.exists():
                    installer.write(path, text)
                if not info:
                    run(compose.compose_command(spec.name, path) + ["create", "--no-build", "--no-recreate", "--pull", "never", "model"])
            elif operation == "start":
                owned(spec, info, image)
                if not info["State"].get("Running"):
                    run(["docker", "start", info["Id"]])
            return {"ok": True}
        if operation == "ready":
            if number != 0:
                if not info or not info["State"].get("Running"):
                    raise ValueError("Worker is not running")
                return {"ok": True}
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                current = owned(spec, container(spec), image)
                if not current["State"].get("Running"):
                    raise ValueError("API rank exited during startup")
                health = current["State"].get("Health", {}).get("Status")
                if health == "healthy":
                    return {"ok": True}
                if health == "unhealthy":
                    raise ValueError("API health check failed; inspect this rank's logs")
                time.sleep(2)
            raise ValueError("Readiness exceeded 30 minutes; inspect logs before restarting")
        name = spec.command[spec.command.index("--served-model-name") + 1]
        port = int(spec.command[spec.command.index("--port") + 1])
    if operation == "smoke":
        listed = http_json(port, "/v1/models")
        if name not in [entry["id"] for entry in listed["data"]]:
            raise ValueError("API serves a different model")
        response = http_json(port, "/v1/chat/completions", {"model": name,
                             "messages": [{"role": "user", "content": "Reply only READY"}],
                             "max_tokens": 64, "temperature": 0,
                             "chat_template_kwargs": {"enable_thinking": False}})
        if not response.get("choices") or not response["choices"][0]["message"].get("content", "").strip():
            raise ValueError("Smoke request returned no answer")
        return {"ok": True, "model": name, "scope": "One short generation; not cache/performance qualification"}
    raise ValueError("Unknown rank operation: " + operation)
