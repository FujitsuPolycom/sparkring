"""Produce a local source-image receipt using CPU-only container verification."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from archive_utils import read_archive, sha
from receipt_contract import validate_receipt
from native_files import mode as native_mode

HERE = Path(__file__).resolve().parent
IMAGE_ROOT = "/opt/sparkcache-jj-runtime"
TOOLS = {"archive_utils.py", "install_sources.py", "verify_sources.py",
         "build_snapshot.py", "build_nccl.py", "receipt_contract.py", "nvcc_deterministic.py", "profile_assets.py", "native_files.py"}


def trusted_closure(context, lock_bytes):
    """Bind the prepared manifest and tool bytes to the local reviewed recipe."""
    receipt = json.loads((context / "context-receipt.json").read_bytes())
    context_bytes = (context / "image-context.tar").read_bytes()
    if sha(context_bytes) != receipt["context_sha256"]:
        raise ValueError("Prepared context archive differs from its receipt")
    files = read_archive(context_bytes)
    if set(files) != {"Dockerfile", "payload.tar"}:
        raise ValueError("Prepared context has an unexpected file set")
    payload = files["payload.tar"][0]
    if sha(payload) != receipt["payload_sha256"]:
        raise ValueError("Prepared payload differs from its receipt")
    entries = read_archive(payload)
    prefix = IMAGE_ROOT.lstrip("/") + "/"
    closure = {name: entries[prefix + name][0] for name in TOOLS | {"manifest.json", "source-lock.json"}}
    manifest_bytes = closure["manifest.json"]
    if (sha(manifest_bytes) != receipt["manifest_sha256"]
            or manifest_bytes != (context / "manifest.json").read_bytes()
            or closure["source-lock.json"] != lock_bytes):
        raise ValueError("Prepared manifest or source lock differs")
    manifest = json.loads(manifest_bytes)
    if manifest.get("source_lock_sha256") != sha(lock_bytes) or set(manifest["tool_hashes"]) != TOOLS:
        raise ValueError("Prepared verifier closure differs")
    for name in TOOLS:
        if (closure[name] != (HERE / name).read_bytes()
                or sha(closure[name]) != manifest["tool_hashes"][name]):
            raise ValueError(f"Prepared tool differs from trusted local recipe: {name}")
    return closure


def verify_embedded_closure(image, closure):
    """Copy files from a stopped container; no candidate program is executed."""
    identifier = subprocess.check_output([
        "docker", "create", "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--entrypoint", "/bin/true", image]).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", identifier):
        raise ValueError("Docker create did not return a container ID")
    try:
        for name, expected in sorted(closure.items()):
            archive = subprocess.check_output(["docker", "cp", f"{identifier}:{IMAGE_ROOT}/{name}", "-"])
            files = read_archive(archive)
            if set(files) != {name} or files[name][0] != expected:
                raise ValueError(f"Embedded verifier input differs from trusted context: {name}")
    finally:
        subprocess.run(["docker", "rm", identifier], check=True, capture_output=True)


def verify(image, profile, lock_path, output, context):
    if output.exists():
        raise ValueError("Receipt output must be absent")
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    closure = trusted_closure(context, lock_bytes)
    selected_native_mode = native_mode(json.loads(closure["manifest.json"]))
    if profile not in lock["profiles"]:
        raise ValueError("Profile is not declared by the source lock")
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]
    if inspected.get("Os") != "linux" or inspected.get("Architecture") != "arm64":
        raise ValueError("Source image requires Linux ARM64")
    parent = json.loads(subprocess.check_output(["docker", "image", "inspect", lock["parent"]["image_id"]]))[0]
    layers = parent["RootFS"]["Layers"]
    if inspected["RootFS"]["Layers"][:len(layers)] != layers:
        raise ValueError("Image does not retain its declared parent layers")
    verify_embedded_closure(inspected["Id"], closure)
    # Isolated Python ignores image PYTHONPATH and script-directory shadow modules.
    # Imports resolve to the read-only host-verified closure, never candidate tools.
    program = ("import sys;sys.path.insert(0,'/sparkring-verifier');"
               "from pathlib import Path;import json;from verify_sources import verify;"
               f"print(json.dumps(verify(root=Path('{IMAGE_ROOT}'))))")
    with tempfile.TemporaryDirectory(prefix="sparkring-verifier-") as directory:
        for name, data in closure.items():
            (Path(directory) / name).write_bytes(data)
        argv = ["docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--mount",
                f"type=bind,source={Path(directory).resolve()},target=/sparkring-verifier,readonly",
                "--entrypoint", "python3", inspected["Id"], "-I", "-S", "-B", "-c", program]
        inside = json.loads(subprocess.check_output(argv))
    digest = hashlib.sha256(lock_bytes).hexdigest()
    if inside.get("source_lock_sha256") != digest:
        raise ValueError("Built image uses another source lock")
    if inside.get("native_mode", "compile") != selected_native_mode:
        raise ValueError("Image native mode differs from trusted context")
    receipt = {"schema": "sparkring-source-image-receipt/v1", "status": "research-only",
               "image_id": inspected["Id"], "image_reference": inspected["Id"], "platform": "linux/arm64",
               "checks_passed": True, "profile": profile, "source_lock_sha256": digest,
               "native_mode": selected_native_mode,
               "inside_image": inside, "verification_command": argv,
               "limitation": "CPU file and dependency verification does not qualify rebuilt GPU or serving behavior."}
    validate_receipt(receipt, lock)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("glm53-tp4-lock.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True,
                        help="Trusted directory produced by prepare_image.py for this image")
    args = parser.parse_args()
    print(json.dumps(verify(args.image, args.profile, args.lock, args.output, args.context), indent=2))
