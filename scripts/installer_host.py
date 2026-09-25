"""Rank-local installer operations, invoked only by the approved controller plan."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import urllib.request

from runtime.common import compose, glm_native_candidate, installer, native_candidate, profiles, qwen_flash_next, setup
from runtime.common.container_spec import expected_inspection
from scripts import deploy_engine

POSIX_STATS = os.name == "posix"


class CommandError(subprocess.CalledProcessError):
    """A failed command whose message carries the tail of its own error output."""

    def __str__(self):
        detail = self.stderr or self.output or ""
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        lines = [line.strip() for line in detail.strip().splitlines() if line.strip()][-3:]
        return super().__str__() + (": " + " | ".join(lines) if lines else "")


def run(argv, **kwargs):
    if argv[0] == "docker" and argv[1:3] != ["--context", "default"]:
        argv = ["docker", "--context", "default", *argv[1:]]
    kwargs.setdefault("check", True)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("timeout", 7200)
    kwargs.setdefault("text", True)
    try:
        return subprocess.run(argv, **kwargs)
    except subprocess.CalledProcessError as error:
        raise CommandError(error.returncode, error.cmd, error.output, error.stderr) from None


def plain(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Installer paths must be absolute without symlink components")
    return path


def image_info(lock):
    return json.loads(run(["docker", "image", "inspect", lock["selection"]["image_id"]]).stdout)[0]


ADMISSIONS = Path("/var/lib/sparkring/admissions")


def admit_image(lock):
    card = lock["selection"]
    if "image_runtime" in lock:
        from runtime.common import installer_image, loader_policy
        # Admission starts three isolated containers. Its result depends on the
        # content-addressed image, the lock, the profile and this host's kernel,
        # Docker and loader policy, so it is reused while all of them match.
        current = json.loads(run(["docker", "image", "inspect", card["image_id"]]).stdout)[0]["Id"]
        key = hashlib.sha256(json.dumps({
            "lock": lock["image_runtime"], "profile": card["profile"], "nodes": card["nodes"],
            "kernel": os.uname().release if hasattr(os, "uname") else "",
            "docker": run(["docker", "version", "--format", "{{.Server.Version}}"]).stdout.strip(),
            "policy": hashlib.sha256(loader_policy.PROFILE.read_bytes()).hexdigest(),
        }, sort_keys=True).encode()).hexdigest()
        record = ADMISSIONS / (key + ".json")
        if record.is_file() and not record.is_symlink():
            saved = profiles.read_json(record)
            if saved.get("image_id") == current:
                return saved["receipt"]
        receipt = installer_image.admit(lock["image_runtime"], run=run, profile=card["profile"], nodes=card["nodes"])
        loader_policy.check(card["image_id"], run=run)
        try:
            ADMISSIONS.mkdir(parents=True, exist_ok=True, mode=0o700)
            deploy_engine.save_receipt(record, {"image_id": current, "receipt": receipt})
        except OSError:
            pass
        return receipt
    def native_run(argv, **kwargs):
        # Native admission deliberately reads the installed receipt as bytes.
        kwargs.setdefault("text", False)
        return run(argv, **kwargs)
    if card["profile"].startswith("glm53-"):
        return glm_native_candidate.observe(card["image_id"], card["release"], run=native_run)
    return native_candidate.verify_image(card["image_id"], card["release"], run=native_run)


def _file_sha256(item):
    digest = hashlib.sha256()
    with item.open("rb") as stream:
        while chunk := stream.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(path):
    """SHA-256 of every checkpoint file.

    Files are hashed concurrently: hashlib releases the GIL for large updates,
    so threads spread the work across cores until storage bandwidth is the limit.
    """
    from concurrent.futures import ThreadPoolExecutor
    path = plain(path)
    files = []
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path)
        if relative.parts[0] in (".cache", ".git"):
            continue
        if item.is_symlink():
            raise ValueError("Use a complete local checkpoint directory, not external symlinks")
        if item.is_file():
            files.append((relative.as_posix(), item))
    workers = max(1, min(16, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = pool.map(_file_sha256, [item for _, item in files])
        result = {name: digest for (name, _), digest in zip(files, digests)}
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


def checksum_manifest(profile):
    """Per-file SHA-256 pins for a profile's checkpoint revision, if recorded."""
    own = profiles.ROOT / "profiles" / profile / "SHA256SUMS"
    if own.is_file():
        return own
    if profile in compose.EXAMPLES:
        return profiles.ROOT / "profiles/qwen38-flash-next-tp2/SHA256SUMS"
    return None


CHECKPOINTS = Path("/var/lib/sparkring/checkpoints")


def _checkpoint_record(model):
    return CHECKPOINTS / (hashlib.sha256(str(model).encode()).hexdigest() + ".json")


def remembered_checkpoint(card, model, stats):
    """File hashes from an earlier verification of this exact tree on this host.

    Reused only on Linux while every file's device, inode, size, modification
    and change time are unchanged; any difference forces full hashing.
    """
    record = _checkpoint_record(model)
    if not POSIX_STATS or not record.is_file():
        return None
    saved = profiles.read_json(record)
    if (saved.get("repository"), saved.get("revision"), saved.get("path")) != (card["model_repository"], card["model_revision"], str(model)):
        return None
    return saved["files"] if saved.get("file_stats") == stats else None


def remember_checkpoint(receipt):
    """Best-effort: the record only saves re-hashing, so failing to write it is harmless."""
    if not POSIX_STATS:
        return
    try:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True, mode=0o700)
        deploy_engine.save_receipt(_checkpoint_record(receipt["path"]), receipt)
    except OSError:
        pass


def pinned_differences(profile, files):
    """Names whose recorded pin differs from, or is absent in, a measured tree."""
    sums = checksum_manifest(profile)
    if sums is None:
        return []
    differences = []
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if files.get(name.lstrip("*")) != digest:
            differences.append(name.lstrip("*"))
    return differences


def hub_download(lock, model):
    """Fetch the pinned revision into ``model``; existing identical files are kept."""
    card = lock["selection"]
    code = "from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],local_dir='/model')"
    run(["docker", "run", "--rm", "--pull", "never", "--runtime", "runc", "--user", f"{os.getuid()}:{os.getgid()}",
         "--env", "HF_HOME=/tmp/huggingface", "--mount", f"type=bind,src={model},dst=/model",
         "--entrypoint", "python3" if "image_runtime" in lock else "/opt/venv/bin/python", card["image_id"], "-c", code,
         card["model_repository"], card["model_revision"]])


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
    sums = checksum_manifest(card["profile"])
    if sums is not None:
        for line in sums.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            if receipt["files"].get(name.lstrip("*")) != digest:
                raise ValueError("Checkpoint shard checksum differs: " + name)
    return receipt


def transfer_model(operation, lock, row, state):
    """Receive only a verified checkpoint manifest into an owned destination."""
    import shutil
    import sys
    from pathlib import PurePosixPath
    receipt_path = plain(state / "model.json")
    model = plain(row["model"])
    marker = plain(state / "model-transfer.json")
    if operation == "model-present":
        from runtime.host.assets import metadata_matches
        return {"present": (not marker.exists() or receipt_path.exists()) and metadata_matches(model, installer.checkpoint_contract(lock["selection"]))}
    if operation == "model-transfer-manifest":
        receipt = verify_model(lock, row, receipt_path)
        return {"repository": receipt["repository"], "revision": receipt["revision"], "files": receipt["files"],
                "sizes": {name: value[2] for name, value in model_file_stats(model).items()}}
    if operation == "model-reuse-receipt":
        previous = json.load(sys.stdin)
        source = plain(previous["workspace"])
        if profiles.read_json(source / ".installer-owner.json") != {"deployment": previous["deployment"]}:
            raise ValueError("Previous checkpoint receipt belongs to another deployment")
        saved = plain(source / "installer/model.json")
        if receipt_path.exists() or not saved.is_file():
            return {"reused": False}
        receipt = profiles.read_json(saved)
        if (receipt.get("repository"), receipt.get("revision"), receipt.get("path")) != (lock["selection"]["model_repository"], lock["selection"]["model_revision"], row["model"]):
            return {"reused": False}
        receipt = verify_model(lock, row, saved)
        state.mkdir(parents=True, exist_ok=True)
        deploy_engine.save_receipt(receipt_path, receipt)
        if not lock["backend"].startswith("glm-"):
            plain(row["cache"]).mkdir(parents=True, exist_ok=True)
        return {"reused": True}
    manifest = json.load(sys.stdin)
    card = lock["selection"]
    if (set(manifest) != {"repository", "revision", "files", "sizes"}
            or manifest["repository"] != card["model_repository"] or manifest["revision"] != card["model_revision"]
            or set(manifest["files"]) != set(manifest["sizes"])):
        raise ValueError("Checkpoint transfer identity differs")
    for name, digest in manifest["files"].items():
        if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
                or any(part in ("..", ".cache", ".git") for part in PurePosixPath(name).parts)
                or any(c in name for c in "\r\n\0\\") or str(PurePosixPath(name)) != name
                or not re.fullmatch(r"[0-9a-f]{64}", digest) or type(manifest["sizes"][name]) is not int
                or manifest["sizes"][name] < 0):
            raise ValueError("Unsafe checkpoint transfer manifest")
    expected = {"deployment": lock["id"], "path": str(model), "manifest": manifest}
    if marker.exists():
        if profiles.read_json(marker) != expected:
            raise ValueError("Another checkpoint transfer owns this destination")
    elif model.exists() and any(model.iterdir()):
        raise ValueError("Checkpoint destination is nonempty and has no owned transfer receipt")
    if operation == "model-transfer-prepare":
        policy = profiles.read_json(installer.ROOT / "profiles/storage-planning.json")
        ancestor = model
        while not ancestor.exists():
            ancestor = ancestor.parent
        remaining = sum(size for name, size in manifest["sizes"].items() if not (model / name).is_file())
        required = remaining + max(manifest["sizes"].values(), default=0) + policy["cache_and_jit_allowance_gib"] * 1024**3
        if shutil.disk_usage(ancestor).free < required:
            raise ValueError("Insufficient checkpoint transfer space; existing model remains running")
        model.mkdir(parents=True, exist_ok=True)
        state.mkdir(parents=True, exist_ok=True)
        deploy_engine.save_receipt(marker, expected)
        return {"ok": True}
    if operation != "model-transfer-complete" or not marker.exists():
        raise ValueError("Checkpoint transfer was not prepared")
    before = model_file_stats(model)
    hashes = model_files(model)
    if hashes != manifest["files"] or before != model_file_stats(model):
        raise ValueError("Transferred checkpoint differs from its verified source; repeat install to resume")
    receipt = {"repository": manifest["repository"], "revision": manifest["revision"], "path": str(model),
               "files": hashes, "file_stats": before, "origin": "verified-fabric-copy"}
    verify_model(lock, row, receipt_path, receipt=receipt, measured=hashes)
    deploy_engine.save_receipt(receipt_path, receipt)
    remember_checkpoint(receipt)
    if not lock["backend"].startswith("glm-"):
        plain(row["cache"]).mkdir(parents=True, exist_ok=True)
    return {"ok": True}


def container(spec):
    names = run(["docker", "container", "ls", "--all", "--format", "{{.Names}}" ]).stdout.splitlines()
    return json.loads(run(["docker", "inspect", spec.name]).stdout)[0] if spec.name in names else None


def owned(spec, info, image):
    if info is None:
        raise ValueError("Deployment container is absent")
    expected = expected_inspection(spec, image, backend="compose")
    if "io.sparkring.image-lock" in spec.labels:
        from runtime.common import loader_policy
        expected["host_config"]["SecurityOpt"] = [loader_policy.inspection_option() if option.startswith("seccomp=") else option
                                                  for option in spec.security_opt]
    config, host = info["Config"], info["HostConfig"]
    host = dict(host)
    # Docker uses null for absent optional lists. NVIDIA-backed containers can
    # also receive label=disable on hosts where SELinux is not an active policy.
    for field in ("CapAdd", "SecurityOpt"):
        if host.get(field) is None and expected["host_config"][field] == []:
            host[field] = []
    if "label=disable" in host.get("SecurityOpt", []) and "label=disable" not in expected["host_config"]["SecurityOpt"]:
        options = json.loads(run(["docker", "info", "--format", "{{json .SecurityOptions}}"] ).stdout)
        if isinstance(options, list) and not any("selinux" in str(value).lower() for value in options):
            host["SecurityOpt"] = [option for option in host["SecurityOpt"] if option != "label=disable"]
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


def smoke_request(card):
    """Chat-template settings for the smoke request, owned by the serving profile.

    Profiles whose template always reasons (GLM-5.3) request low effort instead
    of disabling thinking, which would merge reasoning into the answer text.
    """
    source = profiles.load(card["profile"])[0]["configuration"]
    if source["format"] == "serving-profile":
        config = profiles.read_json(profiles.local_path(source["path"]))
        if "smoke" in config:
            return config["smoke"]
    return {"chat_template_kwargs": {"enable_thinking": False}}


def http_json(port, path, body=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
        return json.loads(payload) if payload.strip() else {}


def runtime_binding(lock, row, info, *, root="/"):
    """Installer assertions for the exact stopped/inspected container, not attestation."""
    from runtime.host import node
    identity = node.observation_identity(root=root)
    node_id = identity["node_id"]
    if node_id is None or row.get("node_id", node_id) != node_id:
        raise ValueError("Runtime binding requires this host's expected persistent node identity")
    labels = info.get("Config", {}).get("Labels", {})
    if (not re.fullmatch(r"[0-9a-f]{64}", info.get("Id", "")) or info.get("Image") != lock["selection"]["image_id"]
            or labels.get(compose.LABEL) != lock["id"] or labels.get("io.sparkring.rank") != str(row["rank"])):
        raise ValueError("Runtime binding requires the exact owned container and image")
    return {"schema": "sparkring-runtime-binding/v1", "deployment_id": lock["id"], "node_id": node_id,
            "container_id": info["Id"], "image_id": info["Image"], "rank": row["rank"]}


def local_binding_path(lock, row, *, root="/"):
    """Reject remote/unknown mounts before inspecting the binding source file."""
    from runtime.common import installer_image
    name = installer_image.binding_path(lock, row)
    path = Path(root) / name.lstrip("/")
    candidates = []
    for line in (Path(root) / "proc/self/mountinfo").read_text().splitlines():
        before, separator, after = line.partition(" - ")
        if not separator or len(before.split()) < 5 or not after.split():
            continue
        mount = Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), before.split()[4]))
        if path.is_relative_to(mount):
            candidates.append((len(mount.parts), after.split()[0]))
    deepest = max((depth for depth, _ in candidates), default=-1)
    kinds = {kind for depth, kind in candidates if depth == deepest}
    if not kinds or not kinds <= {"ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "tmpfs", "ramfs"}:
        raise ValueError("Runtime binding must use a verified local filesystem, not a NAS or unknown mount")
    # Check parents first, so a symlink is rejected before following it to a
    # deeper component that could live on a remote filesystem.
    for item in reversed((path, *path.parents)):
        if item.is_symlink():
            raise ValueError("Runtime binding path contains a symlink")
    return path


def read_runtime_binding(path):
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16384:
            raise ValueError("Runtime binding must be a small regular local file")
        raw = stream.read(16385)
        if len(raw) > 16384:
            raise ValueError("Runtime binding grew beyond its size limit")
    return json.loads(raw)


def check_runtime_binding(lock, row, info, *, root="/", finalize=False):
    from runtime.host import node
    expected = runtime_binding(lock, row, info, root=root)
    path = local_binding_path(lock, row, root=root)
    actual = read_runtime_binding(path)
    if actual == expected:
        return expected
    if not finalize or info.get("State", {}).get("Running"):
        raise ValueError("Runtime binding differs; finalize it while the owned container is stopped")
    node.save(path.parent, path.name, expected, mode=0o644)
    return expected


def model_observation(lock, row, info, *, root="/", now=time.time):
    """Allowlisted identities from the inspected container and this host only."""
    from runtime.host import node
    identity = node.observation_identity(root=root)
    expected_node = row.get("node_id")
    state = info.get("State", {}) if info else {}
    return {"schema": "sparkring-model-observation/v1", "source": "installer-docker-inspect", "observed_at": now(),
            "deployment_id": lock["id"], "rank": row["rank"], **identity,
            "expected_node_id": expected_node,
            "node_identity_matches": identity["node_id"] == expected_node if expected_node and identity["node_id"] else None,
            "container_id": info["Id"] if info else None, "container_name": info.get("Name", "").lstrip("/") if info else None,
            "container_started_at": state.get("StartedAt"), "image_id": info["Image"] if info else None,
            "expected_image_id": lock["selection"]["image_id"],
            "present": info is not None, "running": bool(state.get("Running")),
            "health": state.get("Health", {}).get("Status")}


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
    if operation in ("model-present", "model-reuse-receipt") or operation.startswith("model-transfer-"):
        return transfer_model(operation, lock, row, state)
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
        # A config-ID-only docker save/load yields an untagged image which the
        # default image listing can omit. Admission addresses the exact ID.
        existing = run(["docker", "image", "inspect", card["image_id"]], check=False)
        if existing.returncode:
            if card["image_reference"] == card["image_id"]:
                raise ValueError("Pinned local image is absent; preload " + card["image_id"] + " on every rank before up")
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
        if not lock["backend"].startswith("glm-"):
            cache.mkdir(parents=True, exist_ok=True)
        if not row["reuse_verified_model"]:
            hub_download(lock, model)
        before = model_file_stats(model)
        hashes = remembered_checkpoint(card, model, before) or model_files(model)
        origin = "operator-declared-verified-copy" if row["reuse_verified_model"] else "pinned-hub-download"
        if row["reuse_verified_model"] and pinned_differences(card["profile"], hashes):
            # A reused copy that differs from the pinned per-file checksums is
            # synchronized in place: the hub client re-downloads only files whose
            # content differs from the pinned revision.
            hub_download(lock, model)
            before = model_file_stats(model)
            hashes = model_files(model)
            origin = "reused-copy-repaired-from-pinned-revision"
        if model_file_stats(model) != before:
            raise ValueError("Checkpoint changed while preparing its checksum receipt")
        receipt = {"repository": card["model_repository"], "revision": card["model_revision"],
                   "path": row["model"], "files": hashes, "file_stats": before, "origin": origin}
        verify_model(lock, row, model_receipt, receipt=receipt, measured=receipt["files"])
        deploy_engine.save_receipt(model_receipt, receipt)
        remember_checkpoint(receipt)
        return {"ok": True}

    if lock["backend"] == "glm-existing-mesh":
        if operation == "receipt":
            return profiles.read_json(image_receipt)
        from runtime.host import glm_existing_mesh
        return glm_existing_mesh.perform(operation, lock, number, state)
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
            return model_observation(lock, row, info)
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
            if "image_runtime" in lock:
                check_runtime_binding(lock, row, info)
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
            # Installer-owned containers name the admitted local image by its
            # configuration ID. A copy received over the fabric has no registry
            # digest, so a digest reference would not resolve with --pull never.
            local_image = card["image_id"] if "image_runtime" in lock else card["image_reference"]
            text = compose.compose_text(spec, local_image)
            compose.check_equivalence(spec, local_image, text, run=run)
            target = plain(Path(row["deployment_root"]) / lock["id"])
            if operation == "create":
                target.mkdir(parents=True, exist_ok=True)
                path = plain(target / "compose.yaml")
                if path.exists() and path.read_text() != text:
                    raise ValueError("Saved Compose file was edited; reinitialize explicitly")
                if not path.exists():
                    installer.write(path, text)
                if not info:
                    if "image_runtime" in lock:
                        binding = local_binding_path(lock, row)
                        if read_runtime_binding(binding) is None:
                            installer.write(binding, {})
                    run(compose.compose_command(spec.name, path) + ["create", "--no-build", "--no-recreate", "--pull", "never", "model"])
                if "image_runtime" in lock:
                    created = owned(spec, container(spec), image)
                    check_runtime_binding(lock, row, created, finalize=True)
            elif operation == "start":
                owned(spec, info, image)
                if "image_runtime" in lock:
                    check_runtime_binding(lock, row, info)
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
                             "max_tokens": 256, "temperature": 0, **smoke_request(card)})
        if not response.get("choices") or not (response["choices"][0]["message"].get("content") or "").strip():
            raise ValueError("Smoke request returned no answer")
        result = {"ok": True, "model": name, "scope": "One short generation; not cache/performance qualification"}
        if "image_runtime" in lock:
            # The shared image carries the runtime-status dashboard; its absence
            # means the status plugin or runtime binding did not load.
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/sparkring/status/view", timeout=60) as page:
                if page.status != 200 or page.headers.get_content_type() != "text/html":
                    raise ValueError("Runtime-status dashboard is unavailable")
            result["dashboard"] = "/v1/sparkring/status/view"
        return result
    raise ValueError("Unknown rank operation: " + operation)
