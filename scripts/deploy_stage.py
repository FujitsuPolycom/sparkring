"""Stage pinned runtime inputs for the managed mesh without starting a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path, PurePosixPath
import secrets
import shlex
import subprocess
import sys
import tarfile
import base64
import inspect
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROFILE = ROOT / "runtime/glm53-spark-mtp3-mesh"
LAUNCH_FILES = frozenset(
    {
        "site.json",
        "fabric.json",
        "launch-rank.sh",
        "fabric-plan.json",
        *(f"rank{rank}.env" for rank in range(4)),
    }
)


def sha(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def source_archive(root, output):
    """Package tracked source only; reject symlinks and omit private/generated files."""
    root = Path(root).resolve()
    output = Path(output)
    names = (
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    selected = []
    for name in filter(None, names):
        if name.split("/")[0] not in (
            "scripts",
            "runtime",
            "spark_transport",
            "third_party",
        ):
            continue
        p = root / name
        if any(part in (".private", "__pycache__") for part in p.parts) or p.suffix in (
            ".pyc",
            ".pyo",
        ):
            continue
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError("Source symlink is not supported: " + name)
        if p.is_file():
            selected.append(name)
    if "scripts/deploy_stage.py" not in selected:
        raise ValueError("Deployment source must be tracked before packaging")
    if output.exists():
        raise FileExistsError(output)
    manifest = {name: sha(root / name) for name in selected}
    with tarfile.open(output, "w:gz") as archive:
        for name in sorted(selected):
            archive.add(root / name, arcname=name, recursive=False)
    return {"sha256": sha(output), "files": manifest}


def extract_source(archive_path, destination, manifest):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = []
        for item in members:
            p = PurePosixPath(item.name)
            if (
                p.is_absolute()
                or ".." in p.parts
                or not item.isfile()
                or str(p) != item.name
            ):
                raise ValueError("Unsafe source archive entry")
            names.append(item.name)
        if len(set(names)) != len(names) or set(names) != set(manifest):
            raise ValueError("Source archive differs from manifest")
        destination.mkdir(parents=True)
        for item in members:
            target = destination / item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(item) as src, target.open("xb") as dst:
                while chunk := src.read(8 << 20):
                    dst.write(chunk)
            if sha(target) != manifest[item.name]:
                raise ValueError("Source file checksum mismatch: " + item.name)


def prepare_secrets(directory):
    """Persist one epoch and key; never print key material or replace existing secrets."""
    directory = Path(directory)
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError("Secret directory cannot contain symlinks")
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    key = directory / "health.key"
    epoch = directory / "epoch.txt"
    if key.is_symlink() or epoch.is_symlink():
        raise ValueError("Secret files cannot be symlinks")
    if key.exists() != epoch.exists():
        raise ValueError("Incomplete secret state; inspect it before recovery")
    if not key.exists():
        with key.open("xb") as f:
            f.write(secrets.token_bytes(32))
        key.chmod(0o600)
        with epoch.open("x") as f:
            f.write(secrets.token_hex(16) + "\n")
        epoch.chmod(0o600)
    if key.stat().st_size != 32:
        raise ValueError("Shared key must be 32 bytes")
    value = epoch.read_text().strip()
    if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid shared epoch")
    return value


class StageRunner:
    def remote(self, host, argv, *, input=None, timeout=600):
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                host,
                shlex.join(argv),
            ],
            input=input,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(
                f"{host}: {argv[0]} failed: "
                + result.stderr.decode(errors="replace")[-2000:]
            )
        return result.stdout.decode(errors="strict")

    def copy(self, source, destination):
        subprocess.run(
            ["scp", "-3", "-o", "BatchMode=yes", str(source), str(destination)],
            check=True,
            timeout=14400,
        )

    def copy_verified(self, source, host, destination, checksum):
        """Publish a transferred file only after its checksum matches; never overwrite."""
        parent = str(PurePosixPath(destination).parent)
        self.remote(host, ["mkdir", "-p", "--", parent])
        temp = self.remote(
            host, ["mktemp", "-p", parent, ".sparkring-copy-XXXXXXXX"]
        ).strip()
        if str(PurePosixPath(temp).parent) != parent or not PurePosixPath(
            temp
        ).name.startswith(".sparkring-copy-"):
            raise ValueError("Unexpected transfer staging path")
        try:
            self.copy(source, host + ":" + temp)
            if self.remote(host, ["sha256sum", "--", temp]).split()[0] != checksum:
                raise ValueError("Transferred file checksum mismatch")
            self.remote(host, ["ln", "--", temp, destination])
        finally:
            self.remote(host, ["rm", "--", temp])


def _validate_staging_tree(root):
    """Reject redirected or shared write targets before staging trusted artifacts."""
    import os
    from pathlib import Path
    import stat

    root = Path(root)
    for path in (root, *root.parents):
        if path.is_symlink():
            raise ValueError("Staging path contains a symlink: " + str(path))
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError("Staging workspace must be a directory")

    def unreadable(error):
        raise error

    for directory, names, files in os.walk(root, followlinks=False, onerror=unreadable):
        for name in names + files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("Staging path contains a symlink: " + str(path))
            if stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise ValueError(
                        "Staging file has external hard links: " + str(path)
                    )
            elif not stat.S_ISDIR(info.st_mode):
                raise ValueError("Staging path is not a regular file or directory")


def _workspace_local(workspace, owner, operation, token=None, *, base="/srv/sparkring"):
    """Check or lock one dedicated workspace; uncertain operations retain their lock."""
    import json
    import os
    from pathlib import Path
    import re

    root = Path(workspace)
    if root.parent != Path(base) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", root.name
    ):
        raise ValueError("Invalid dedicated workspace")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", owner):
        raise ValueError("Invalid deployment owner")
    _validate_staging_tree(root)
    uid = int(os.environ.get("SUDO_UID", os.getuid()))
    gid = int(os.environ.get("SUDO_GID", os.getgid()))
    marker = root / "deployment-owner.json"
    lock = root / ".stage-operation"
    if root.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != {"owner": owner}:
            raise ValueError("Workspace has no matching ownership record")
        if root.stat().st_uid not in (0, uid):
            raise ValueError("Workspace belongs to another login user")
    if operation == "check":
        if lock.exists():
            raise ValueError(
                "Remote staging is active or interrupted; inspect its lock"
            )
        return {"checked": True, "uid": uid, "gid": gid}
    if not token or not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError("Staging operation requires a valid token")
    if operation == "acquire":
        if not root.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            root.mkdir(mode=0o750)
            os.chown(root, uid, gid)
            with marker.open("x") as stream:
                json.dump({"owner": owner}, stream)
            os.chown(marker, uid, gid)
        lock.mkdir(mode=0o700)
        with (lock / "owner.json").open("x") as stream:
            json.dump({"owner": owner, "token": token}, stream)
        return {"locked": True}
    if not lock.is_dir() or json.loads((lock / "owner.json").read_text()) != {
        "owner": owner,
        "token": token,
    }:
        raise ValueError("Remote staging lock belongs to another operation")
    if operation == "verify":
        return {"checked": True}
    if operation == "release":
        (lock / "owner.json").unlink()
        lock.rmdir()
        return {"released": True}
    raise ValueError("Unknown workspace operation")


def workspace_operation(run, host, workspace, owner, operation, token=None):
    code = "\n\n".join(
        inspect.getsource(fn) for fn in (_validate_staging_tree, _workspace_local)
    )
    code += (
        "\nimport json\nprint(json.dumps(_workspace_local("
        + ",".join(repr(value) for value in (workspace, owner, operation, token))
        + ")))\n"
    )
    return json.loads(run.remote(host, ["sudo", "-n", "python3", "-c", code]))


class WorkspaceRunner:
    """Recheck owned workspace paths before each staging command or file transfer."""

    def __init__(self, run, hosts, workspace, owner, token):
        self.run, self.hosts = run, set(hosts)
        self.workspace, self.owner, self.token = workspace, owner, token

    def guard(self, host):
        if host not in self.hosts:
            raise ValueError("Host is outside this staging operation")
        workspace_operation(
            self.run, host, self.workspace, self.owner, "verify", self.token
        )

    def remote(self, host, argv, **kwargs):
        self.guard(host)
        return self.run.remote(host, argv, **kwargs)

    def copy(self, source, destination):
        for value in (source, destination):
            if ":" in str(value):
                host, path = str(value).split(":", 1)
                if (
                    not PurePosixPath(path).is_relative_to(self.workspace)
                    or ".." in PurePosixPath(path).parts
                ):
                    raise ValueError("Transfer path escapes the staging workspace")
                self.guard(host)
        return self.run.copy(source, destination)

    def copy_verified(self, source, host, destination, checksum):
        self.guard(host)
        if (
            not PurePosixPath(destination).is_relative_to(self.workspace)
            or ".." in PurePosixPath(destination).parts
        ):
            raise ValueError("Transfer path escapes the staging workspace")
        self.guard(str(source).split(":", 1)[0])
        return self.run.copy_verified(source, host, destination, checksum)


def put_secret(run, host, path, data):
    code = """import os,pathlib,sys
p=pathlib.Path(sys.argv[1]);data=sys.stdin.buffer.read()
if len(data)!=32:raise SystemExit('Invalid shared-key length')
if any(v.is_symlink() for v in (p,*p.parents)):raise SystemExit('Shared-key path cannot contain symlinks')
p.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
if p.exists():
 if p.is_symlink() or p.read_bytes()!=data:raise SystemExit('Shared-key conflict')
else:
 with p.open('xb') as f:f.write(data)
p.chmod(0o600)
os.chown(p,0,0);p.parent.chmod(0o700)
"""
    run.remote(host, ["sudo", "-n", "python3", "-c", code, path], input=data)


def stage(preparation, local_state, *, run=None, source_root=ROOT):
    """Prepare sources, image, model and canonical launch; no GPU/model is started."""
    state = Path(local_state)
    _validate_staging_tree(state)
    if any(p.is_symlink() for p in (state, *state.parents)):
        raise ValueError("Staging directory cannot contain symlinks")
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = state / "stage.lock"
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError(
            "Staging is active or interrupted; inspect stage.lock before retrying"
        ) from exc
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()) + "\n")
        return _stage(preparation, state, run=run, source_root=source_root)
    finally:
        lock.unlink()


def _stage(preparation, local_state, *, run=None, source_root=ROOT):
    if os.name != "posix":
        raise ValueError("Run staging from a Linux/WSL controller")
    run = run or StageRunner()
    from scripts.deploy_suite import (
        check_network,
        lifecycle_capabilities,
        require_verified_network,
    )

    require_verified_network(preparation)
    if preparation.get("lifecycle_capabilities", []) != lifecycle_capabilities(PROFILE):
        raise ValueError("Staging source lifecycle differs from reviewed preparation")

    def probe(host, argv, timeout):
        return {
            "returncode": 0,
            "stdout": run.remote(host, argv, timeout=timeout),
            "stderr": "",
        }

    # Saved evidence permits planning; a fresh probe catches drift before mutation.
    preparation = check_network(preparation, probe)
    spec = preparation["spec"]
    workspace = spec["workspace"]
    if (
        not workspace.startswith("/srv/sparkring/")
        or len(PurePosixPath(workspace).parts) != 4
        or ".." in PurePosixPath(workspace).parts
    ):
        raise ValueError("Invalid workspace")
    local_state = Path(local_state).resolve()
    local_state.mkdir(parents=True, exist_ok=True)
    local_state.chmod(0o700)
    binding = local_state / "deployment.json"
    if binding.exists():
        if json.loads(binding.read_text()) != spec:
            raise ValueError("Staging directory belongs to different deployment inputs")
    else:
        with binding.open("x") as stream:
            json.dump(spec, stream)
    public = json.loads((PROFILE / "public-image.json").read_text())
    pins = json.loads((PROFILE / "pins.json").read_text())
    hosts = spec["hosts"]
    if [h["rank"] for h in hosts] != list(range(4)) or len(
        {h["host"] for h in hosts}
    ) != 4:
        raise ValueError("Four distinct ordered hosts are required")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", h["host"]) for h in hosts):
        raise ValueError("Invalid SSH host")
    expected_model = workspace + "/models/" + pins["target"]["revision"]
    expected_paths = {
        "model_roots": [expected_model] * 4,
        "cache_roots": [workspace + "/cache"] * 4,
        "bundle_root": workspace + "/artifacts/mtp3-mesh-bundle",
        "marker_binary": workspace + "/artifacts/mlx5-rdma-tx-marker",
    }
    if any(spec["site"].get(k) != v for k, v in expected_paths.items()):
        raise ValueError("Runtime paths must match the dedicated workspace")
    # Every host must pass the read-only ownership check before the first host changes.
    identities = {}
    for host in hosts:
        identities[host["host"]] = workspace_operation(
            run, host["host"], workspace, spec["owner"], "check"
        )
    token = secrets.token_hex(16)
    operation = local_state / "remote-operation.json"
    operation.write_text(
        json.dumps({"token": token, "workspace": workspace, "hosts": identities})
    )
    operation.chmod(0o600)
    raw_runner = run
    for host in hosts:
        workspace_operation(
            raw_runner, host["host"], workspace, spec["owner"], "acquire", token
        )
    # A failure or timeout leaves remote locks intact: an SSH child may still be writing.
    run = WorkspaceRunner(raw_runner, identities, workspace, spec["owner"], token)
    seed = hosts[0]["host"]
    epoch = prepare_secrets(local_state / "private")
    preparation = {**preparation, "epoch": epoch}
    archive = local_state / "source.tar.gz"
    receipt = local_state / "source-manifest.json"
    if archive.exists():
        source = json.loads(receipt.read_text())
        if sha(archive) != source["sha256"]:
            raise ValueError("Source archive checksum mismatch")
        for name, digest in source["files"].items():
            if sha(Path(source_root) / name) != digest:
                raise ValueError("Source changed; use a separate staging directory")
    else:
        source = source_archive(source_root, archive)
        receipt.write_text(json.dumps(source, indent=2))
    controller_source = local_state / "source"
    if not controller_source.exists():
        extract_source(archive, controller_source, source["files"])
    _validate_staging_tree(controller_source)
    controller_files = {
        path.relative_to(controller_source).as_posix(): sha(path)
        for path in controller_source.rglob("*")
        if path.is_file()
    }
    if controller_files != source["files"]:
        raise ValueError("Controller source snapshot differs from the staged archive")
    source_remote = workspace + "/source"
    bootstrap = """import hashlib,json,pathlib,sys,tarfile
a=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);manifest=json.loads(pathlib.Path(sys.argv[3]).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
if sha(a)!=manifest['sha256']:raise SystemExit('Archive checksum mismatch')
if any(p.is_symlink() for p in (out,*out.parents)):raise SystemExit('Source cannot contain symlinks')
if not out.exists():
 with tarfile.open(a,'r:gz') as t:
  members=t.getmembers();names=[m.name for m in members]
  if len(names)!=len(set(names)) or set(names)!=set(manifest['files']):raise SystemExit('Archive list mismatch')
  if any(not m.isfile() or pathlib.PurePosixPath(m.name).is_absolute() or '..' in pathlib.PurePosixPath(m.name).parts for m in members):raise SystemExit('Unsafe archive')
  out.mkdir()
  for m in members:
   p=out/m.name;p.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(m) as src,p.open('xb') as dst:
    for b in iter(lambda:src.read(8388608),b''):dst.write(b)
for n,h in manifest['files'].items():
 if (out/n).is_symlink() or any(p.is_symlink() for p in (out/n).parents):raise SystemExit('Source contains a symlink')
 if sha(out/n)!=h:raise SystemExit('Source file checksum mismatch: '+n)
"""
    for host in hosts:
        target = host["host"]
        run.copy(archive, target + ":" + workspace + "/source.tar.gz")
        run.copy(receipt, target + ":" + workspace + "/source-manifest.json")
        run.remote(
            target,
            [
                "python3",
                "-c",
                bootstrap,
                workspace + "/source.tar.gz",
                source_remote,
                workspace + "/source-manifest.json",
            ],
        )
        put_secret(
            run,
            target,
            workspace + "/private/health.key",
            (local_state / "private/health.key").read_bytes(),
        )
        run.remote(
            target,
            [
                "mkdir",
                "-p",
                workspace + "/artifacts",
                workspace + "/models",
                workspace + "/cache",
                workspace + "/receipts",
            ],
        )
    run.remote(seed, ["docker", "pull", public["public_reference"]], timeout=14400)
    run.remote(
        seed,
        [
            "docker",
            "image",
            "save",
            "-o",
            workspace + "/image.tar",
            public["public_reference"],
        ],
        timeout=3600,
    )
    image_sha = run.remote(seed, ["sha256sum", workspace + "/image.tar"]).split()[0]
    for h in hosts[1:]:
        run.copy(
            seed + ":" + workspace + "/image.tar",
            h["host"] + ":" + workspace + "/image.tar",
        )
        if (
            run.remote(h["host"], ["sha256sum", workspace + "/image.tar"]).split()[0]
            != image_sha
        ):
            raise ValueError("Transferred image archive mismatch")
        run.remote(
            h["host"], ["docker", "load", "-i", workspace + "/image.tar"], timeout=3600
        )
    for h in hosts:
        if (
            run.remote(
                h["host"],
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    public["public_reference"],
                ],
            ).strip()
            != public["config_image_id"]
        ):
            raise ValueError("Image identity mismatch")
    model = spec["site"]["model_roots"][0]
    run.remote(seed, ["mkdir", "-p", model])
    download = (
        "from huggingface_hub import snapshot_download; snapshot_download(repo_id="
        + repr(pins["target"]["repository"])
        + ",revision="
        + repr(pins["target"]["revision"])
        + ",local_dir='/download')"
    )
    run.remote(
        seed,
        [
            "docker",
            "run",
            "--rm",
            "--user",
            str(identities[seed]["uid"]) + ":" + str(identities[seed]["gid"]),
            "-e",
            "HOME=/download",
            "-e",
            "HF_HOME=/download/.cache/huggingface",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=void",
            "-e",
            "HF_HUB_OFFLINE=0",
            "-e",
            "TRANSFORMERS_OFFLINE=0",
            "-v",
            model + ":/download",
            "--entrypoint",
            "python3",
            public["public_reference"],
            "-c",
            download,
        ],
        timeout=14400,
    )
    manifest_code = """import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);rows={}
if any(p.is_symlink() for p in (root,*root.parents)):raise SystemExit('Model path contains a symlink')
for p in root.rglob('*'):
 if p.is_symlink():raise SystemExit('Model directory contains a symlink')
 if p.is_file() and '.cache' not in p.relative_to(root).parts:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8388608),b''):h.update(b)
  rows[p.relative_to(root).as_posix()]=h.hexdigest()
print(json.dumps(rows))
"""
    model_files = json.loads(
        run.remote(seed, ["python3", "-c", manifest_code, model], timeout=7200)
    )
    if (
        model_files.get("config.json") != pins["target"]["config_sha256"]
        or model_files.get("model.safetensors.index.json")
        != pins["target"]["index_sha256"]
    ):
        raise ValueError("Target metadata mismatch")
    for h in hosts[1:]:
        run.remote(h["host"], ["mkdir", "-p", model])
        present = json.loads(
            run.remote(h["host"], ["python3", "-c", manifest_code, model], timeout=7200)
        )
        if any(model_files.get(n) != digest for n, digest in present.items()):
            raise ValueError(
                "Destination model contains different files; refusing overwrite"
            )
        for name, digest in model_files.items():
            if name not in present:
                run.copy_verified(
                    seed + ":" + model + "/" + name,
                    h["host"],
                    model + "/" + name,
                    digest,
                )
        other = json.loads(
            run.remote(h["host"], ["python3", "-c", manifest_code, model], timeout=7200)
        )
        if other != model_files:
            raise ValueError("Target weight-file verification failed")
    preparation["source"] = source
    preparation["model_files"] = model_files
    preparation["controller_launch"] = str(local_state / "launch")
    preparation["controller_source"] = str(controller_source)
    for name, value in [
        ("preparation.json", preparation),
        ("site.json", spec["site"]),
        ("fabric.json", spec["fabric"]),
    ]:
        file = local_state / name
        file.write_text(json.dumps(value, indent=2))
        for h in hosts:
            run.copy(file, h["host"] + ":" + workspace + "/" + name)
    from scripts.deploy_engine import plan_digest
    from scripts.deploy_trust import trusted_check, trusted_script

    canonical_files = None
    staging_digest = plan_digest(preparation)
    for h in hosts:
        result = json.loads(
            run.remote(
                h["host"],
                trusted_script(
                    workspace,
                    staging_digest,
                    script="scripts/deploy_stage.py",
                    args=["finish-host", "--workspace", workspace],
                    require_launch=False,
                ),
                timeout=180,
            )
        )
        expected_files = result["canonical_launch_files"]
        if set(expected_files) != LAUNCH_FILES or (
            canonical_files is not None and expected_files != canonical_files
        ):
            raise ValueError("Canonical launch differs across hosts")
        canonical_files = expected_files
    # Read back the rendered launch, so controller-side readiness and native tests
    # use the same hashed inputs as the four hosts.
    pack_launch = """import base64,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);names={'rank0.env','rank1.env','rank2.env','rank3.env','site.json','fabric.json','launch-rank.sh','fabric-plan.json'}
if {f.name for f in p.iterdir()}!=names:raise SystemExit('Unexpected launch file list')
print(json.dumps({n:base64.b64encode((p/n).read_bytes()).decode() for n in names}))
"""
    for h in hosts:
        launch_files = json.loads(
            run.remote(
                h["host"], ["python3", "-I", "-c", pack_launch, workspace + "/launch"]
            )
        )
        install_launch_copy(
            local_state / "launch", launch_files, expected=canonical_files
        )
    preparation["launch_files"] = canonical_files
    final_input = local_state / "preparation.json"
    final_input.write_text(json.dumps(preparation, indent=2))
    for h in hosts:
        run.copy(final_input, h["host"] + ":" + workspace + "/preparation.json")
    for h in hosts:
        run.remote(h["host"], trusted_check(workspace, plan_digest(preparation)))
    (local_state / "prepared.json").write_text(json.dumps(preparation, indent=2))
    for host in hosts:
        workspace_operation(
            raw_runner, host["host"], workspace, spec["owner"], "release", token
        )
    operation.unlink()
    return preparation


def install_launch_copy(output, files, *, expected=None):
    """Keep a verified controller copy without overwriting different launch inputs."""
    output = Path(output)
    _validate_staging_tree(output)
    decoded = {}
    for name, value in files.items():
        if PurePosixPath(name).name != name or name in ("", ".", ".."):
            raise ValueError("Invalid rendered launch filename")
        decoded[name] = base64.b64decode(value, validate=True)
    record = json.loads(decoded["fabric-plan.json"])
    if set(decoded) != LAUNCH_FILES or set(record["files"]) != LAUNCH_FILES - {
        "fabric-plan.json"
    }:
        raise ValueError("Rendered launch file list mismatch")
    for name, digest in record["files"].items():
        if hashlib.sha256(decoded[name]).hexdigest() != digest:
            raise ValueError("Rendered launch checksum mismatch")
    actual = {
        name: hashlib.sha256(value).hexdigest() for name, value in decoded.items()
    }
    if expected is not None and actual != expected:
        raise ValueError("Rendered launch differs from canonical source output")
    if output.exists() and {p.name for p in output.iterdir()} != LAUNCH_FILES:
        raise ValueError("Controller launch file list changed")
    output.mkdir(exist_ok=True, mode=0o700)
    for name, value in decoded.items():
        target = output / name
        if target.is_symlink() or (target.exists() and target.read_bytes() != value):
            raise ValueError("Controller launch inputs changed")
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(value)
            target.chmod(0o600)
    return actual


def finish_host(workspace):
    """Extract pinned host helpers and render the canonical stopped launch."""
    workspace = Path(workspace)
    # Root-only secrets and staging locks are checked by the privileged outer guard.
    _validate_staging_tree(workspace / "artifacts")
    _validate_staging_tree(workspace / "launch")
    preparation = json.loads((workspace / "preparation.json").read_text())
    spec = preparation["spec"]
    for name, expected in (
        ("site.json", spec["site"]),
        ("fabric.json", spec["fabric"]),
    ):
        path = workspace / name
        if path.is_symlink() or json.loads(path.read_text()) != expected:
            raise ValueError("Staging render inputs differ from approved preparation")
    public = json.loads((PROFILE / "public-image.json").read_text())
    pins = json.loads((PROFILE / "pins.json").read_text())
    artifact = Path(spec["site"]["bundle_root"])
    marker = Path(spec["site"]["marker_binary"])
    for path in (artifact, marker):
        if not path.is_relative_to(workspace) or ".." in path.parts:
            raise ValueError("Extracted artifact escapes the staging workspace")
    if not artifact.exists() and not marker.exists():
        identifier = subprocess.check_output(
            [
                "docker",
                "create",
                "--entrypoint",
                "/bin/true",
                public["public_reference"],
            ],
            text=True,
        ).strip()
        try:
            subprocess.run(
                ["docker", "cp", identifier + ":/opt/spark-sircl", str(artifact)],
                check=True,
            )
            subprocess.run(
                [
                    "docker",
                    "cp",
                    identifier + ":/opt/sparkring/bin/mlx5-rdma-tx-marker",
                    str(marker),
                ],
                check=True,
            )
        finally:
            subprocess.run(["docker", "rm", identifier], check=True)
    image_receipt = json.loads((PROFILE / "image-receipt.json").read_text())
    if (
        sha(artifact / "sparkring-overlay-manifest.json")
        != pins["canonical_bundle_manifest_sha256"]
        or sha(marker) != image_receipt["inside_image"]["marker_binary_sha256"]
    ):
        raise ValueError("Extracted artifact mismatch or incomplete extraction")
    sys.path.insert(0, str(PROFILE))
    module_spec = importlib.util.spec_from_file_location(
        "mesh_profile", PROFILE / "profile.py"
    )
    profile = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(profile)
    output = workspace / "launch"
    with tempfile.TemporaryDirectory(
        prefix=".canonical-render-", dir=workspace / "artifacts"
    ) as directory:
        canonical = Path(directory) / "launch"
        profile.render(
            workspace / "site.json", artifact, canonical, PROFILE / "image-receipt.json"
        )
        files = {
            p.name: base64.b64encode(p.read_bytes()).decode()
            for p in canonical.iterdir()
        }
        launch_files = install_launch_copy(output, files)
    verify_host(workspace, allow_unanchored=True, quiet=True)
    print(json.dumps({"canonical_launch_files": launch_files}))


def verify_host(
    workspace, preparation_sha256=None, *, allow_unanchored=False, quiet=False
):
    """Bind staged source and launch files to the controller's preparation document."""
    from scripts.deploy_engine import plan_digest

    workspace = Path(workspace)
    document = json.loads((workspace / "preparation.json").read_text())
    if preparation_sha256 and plan_digest(document) != preparation_sha256:
        raise ValueError("Host preparation differs from controller inputs")
    if document["spec"]["workspace"] != str(workspace):
        raise ValueError("Host workspace differs from preparation")
    for name, digest in document["source"]["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
            raise ValueError("Invalid source manifest path")
        p = workspace / "source" / name
        if any(v.is_symlink() for v in (p, *p.parents)) or sha(p) != digest:
            raise ValueError("Host source changed: " + name)
    output = workspace / "launch"
    expected = document.get("launch_files")
    if expected is None and not allow_unanchored:
        raise ValueError("Preparation does not bind rendered launch files")
    if {p.name for p in output.iterdir()} != LAUNCH_FILES:
        raise ValueError("Rendered launch file list changed")
    if expected is not None and (
        set(expected) != LAUNCH_FILES
        or any(
            (output / name).is_symlink() or sha(output / name) != digest
            for name, digest in expected.items()
        )
    ):
        raise ValueError("Rendered launch differs from approved preparation")
    record = json.loads((output / "fabric-plan.json").read_text())
    if set(record["files"]) != LAUNCH_FILES - {"fabric-plan.json"}:
        raise ValueError("Rendered launch manifest omits required files")
    for name, digest in record["files"].items():
        if (
            PurePosixPath(name).name != name
            or (output / name).is_symlink()
            or sha(output / name) != digest
        ):
            raise ValueError("Rendered launch changed")
    # Rendering normalizes the site, but the original reviewed input is retained.
    for name, expected in (
        ("site.json", document["spec"]["site"]),
        ("fabric.json", document["spec"]["fabric"]),
    ):
        if json.loads((workspace / name).read_text()) != expected:
            raise ValueError("Staged site inputs changed")
    if (
        record.get("site_sha256") != sha(workspace / "site.json")
        or record.get("topology_sha256") != sha(workspace / "fabric.json")
        or record.get("image_receipt_sha256")
        != sha(workspace / "source/runtime/glm53-spark-mtp3-mesh/image-receipt.json")
    ):
        raise ValueError("Rendered launch belongs to different deployment inputs")
    if (
        json.loads((output / "site.json").read_text())
        != dict(document["spec"]["site"], topology_file="fabric.json")
        or json.loads((output / "fabric.json").read_text())
        != document["spec"]["fabric"]
    ):
        raise ValueError("Rendered site differs from reviewed deployment inputs")
    if not quiet:
        print(json.dumps({"verified": True}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    stage_parser = sub.add_parser("stage")
    stage_parser.add_argument("--preparation", type=Path, required=True)
    stage_parser.add_argument("--state", type=Path, required=True)
    stage_parser.add_argument("--execute", action="store_true")
    finish = sub.add_parser("finish-host")
    finish.add_argument("--workspace", type=Path, required=True)
    verify = sub.add_parser("verify-host")
    verify.add_argument("--workspace", type=Path, required=True)
    verify.add_argument("--preparation-sha256", required=True)
    args = p.parse_args(argv)
    if args.command == "finish-host":
        finish_host(args.workspace)
    elif args.command == "verify-host":
        verify_host(args.workspace, args.preparation_sha256)
    elif not args.execute:
        print(
            "Plan: stage tracked source, one image/model download, verified copies, shared key, and stopped launch files. No model will start. Add --execute after network verification."
        )
    else:
        stage(json.loads(args.preparation.read_text()), args.state)
        print("Runtime files prepared; no model started.")


if __name__ == "__main__":
    main()
