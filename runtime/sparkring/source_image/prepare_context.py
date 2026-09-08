"""Prepare an offline, source-locked ARM64 context; never invokes Docker."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tomllib

from archive_utils import STARTUP_PATHS, WARMUP_SOURCE, make_archive, read_archive, relative_path, sha, transform_warmup
from native_files import ARCHIVE as NATIVE_ARCHIVE, describe as describe_native_files

HERE = Path(__file__).resolve().parent
SITE = "/usr/local/lib/python3.12/dist-packages"
ROOT = "/opt/sparkcache-jj-runtime"
PUBLIC_BASE = "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:11a556a54041fd823d152a7f051ac4f7c617dc539030df26e93008392fee0746"
CRITICAL = {
    "/opt/spark-sircl/libspark_transport_capi.so": "056243fad27d224b82e437925ffa2aed42037e6bd29f239f56076a832f6ca5cb",
    "/opt/spark-sircl/sparkring-overlay-manifest.json": "c0fd5567442b08b908cc193f36d0864e262573c7e5d232509479a823cface742",
    "/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so": "2657cdd2e54a097c9544e4c79ae62c0646db6db123ff24e4f0c384238c3a1e8d",
}
NCCL = {"/opt/sparkring/nccl/libnccl.so.2.30.7":
        "5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3"}
GENERATED_PREFIXES = (
    "vllm/vllm_flash_attn/", "vllm/third_party/triton_kernels/",
    "vllm/third_party/flashmla/", "vllm/third_party/deep_gemm/",
    "vllm/third_party/fmha_sm100/", "vllm/third_party/tml_fa4/",
)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def startup_archive(spec, epoch):
    """Export only the explicitly supported startup component from a clean pin."""
    revision = spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Startup component requires a full commit ID")
    files = {}
    for source in STARTUP_PATHS:
        name = Path(source).name
        data = (Path(spec["checkout"]) / name).read_bytes()
        if sha(data) != spec["files"].get(name):
            raise ValueError("Startup input differs from composition lock")
        files[source] = (data, 0o755 if name == "serve_with_warmup.py" else 0o644)
    data = make_archive(files, epoch)
    record = {k: v for k, v in spec.items() if k not in ("checkout", "files")}
    record.update({"schema": "sparkring-startup-source-override/v1",
                   "archive_sha256": sha(data),
                   "files": {p: sha(v[0]) for p, v in files.items()},
                   "install_paths": STARTUP_PATHS})
    installed = {STARTUP_PATHS[p]: sha(v[0]) for p, v in files.items()}
    transform = spec.get("warmup_transform")
    if transform is not None:
        raw = files[WARMUP_SOURCE][0]
        changed = transform_warmup(raw, transform)
        installed[STARTUP_PATHS[WARMUP_SOURCE]] = sha(changed)
        record["transform"] = {
            "name": transform,
            "purpose": "Use the qualified GLM warmup thinking request without changing public serving reasoning behavior",
            "source_file": WARMUP_SOURCE, "raw_sha256": sha(raw),
            "installed_sha256": sha(changed),
        }
    record["installed_files"] = installed
    return data, record


def source_archive(name, spec, epoch):
    repo = Path(spec["checkout"]).resolve()
    revision = spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Sources require full commit IDs")
    base = spec["base_revision"]
    if git(repo, "rev-parse", "HEAD").decode().strip() != base:
        raise ValueError(f"{name}: source base differs from lock")
    tree = git(repo, "write-tree").decode().strip()
    if tree != spec["result_tree"]:
        raise ValueError(f"{name}: assembled Git tree differs from lock")
    if git(repo, "diff", "--name-only").strip() or git(repo, "ls-files", "--others", "--exclude-standard").strip():
        raise ValueError(f"{name}: source has unreviewed files")
    package = name
    selected = [package]
    if name == "vllm":
        for path in ("LICENSE", "NOTICE"):
            if (repo / path).is_file():
                selected.append(path)
    if name != "vllm":
        for path in ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in", "setup.cfg"):
            if (repo / path).is_file():
                selected.append(path)
        if (repo / "setup.py").exists():
            # This SparkCache hook only excludes test modules from its wheel;
            # it does not import runtime packages, compile code or access CUDA.
            reviewed = "d34d46facfccb00d901e3f129fd126f20dcaa0814ec6b20238589e4015b2f9c0"
            if name != "sparkcache" or sha(git(repo, "show", f"{tree}:setup.py")) != reviewed:
                raise ValueError("Executable setup.py requires a separate build review")
            selected.append("setup.py")
    raw = git(repo, "-c", "core.autocrlf=false", "-c", "core.eol=lf", "archive", "--format=tar", tree, *selected)
    files = read_archive(raw)
    if not any(p.startswith(package + "/") for p in files):
        raise ValueError(f"Missing package: {name}")
    project = None
    if name != "vllm":
        project = tomllib.loads(files["pyproject.toml"][0].decode())
        if project["build-system"]["build-backend"] != "setuptools.build_meta":
            raise ValueError("Only reviewed setuptools metadata builds are supported")
        if project["project"].get("dynamic") or not project["project"].get("version"):
            raise ValueError("Distribution metadata must have a static version")
    data = make_archive(files, epoch)
    public = {k: v for k, v in spec.items() if k != "checkout"}
    public.update({
        "package": package,
        "package_tree": git(repo, "rev-parse", f"{tree}:{package}").decode().strip(),
        "archive_sha256": sha(data),
        "files": {p: sha(value[0]) for p, value in files.items()},
    })
    if project:
        public["distribution_version"] = project["project"]["version"]
        public["entry_points"] = project["project"].get("entry-points", {})
        public["console_scripts"] = project["project"].get("scripts", {})
    return data, public


def retained_files(base_files):
    result = {}
    for absolute, digest in base_files.items():
        if not absolute.startswith(SITE + "/"):
            continue
        relative = absolute[len(SITE) + 1:]
        relative_path(relative)
        if (relative.startswith("vllm/") and "__pycache__" not in relative.split("/")
                and (relative.endswith(".so") or relative in ("vllm/_version.py", "vllm/vllm-rs")
                     or relative.startswith(GENERATED_PREFIXES))):
            result[relative] = digest
    if not result:
        raise ValueError("Base receipt does not enumerate retained vLLM dependencies")
    return result


def prepare(spec, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output must be a new directory; existing contexts are immutable")
    if spec["base_image"] != PUBLIC_BASE:
        raise ValueError("This recipe requires its declared digest-pinned ARM64 parent")
    if set(spec["sources"]) != {"vllm", "b12x", "sparkcache"}:
        raise ValueError("Exactly vllm, b12x and sparkcache sources are required")
    epoch = spec["source_date_epoch"]
    if type(epoch) is not int or epoch < 0:
        raise ValueError("source_date_epoch must be a nonnegative integer")
    base_bytes = Path(spec["base_receipt"]).read_bytes()
    if sha(base_bytes) != spec["base_receipt_sha256"]:
        raise ValueError("Base receipt hash mismatch")
    base_files = json.loads(base_bytes)["files"]
    if any(base_files.get(p) != digest for p, digest in CRITICAL.items()):
        raise ValueError("Parent lacks the declared transport, bundle manifest, or native cache placement artifacts")
    # Inherited binary and toolchain identities are checked separately from Python sources.
    runtime = json.loads(json.dumps(spec["runtime"]))
    if not runtime.get("expected_distributions") or not runtime.get("description"):
        raise ValueError("Explicit inherited runtime expectations are required")
    for path, digest in NCCL.items():
        if runtime.get("required_files", {}).get(path, digest) != digest:
            raise ValueError("Retained patched NCCL identity cannot be changed")
    runtime["required_files"] = {**runtime.get("required_files", {}), **NCCL}
    for absolute, digest in base_files.items():
        if not absolute.startswith("/") or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid base receipt file")
        relative_path(absolute[1:])
    manifest = {
        "schema": "sparkcache-jj-runtime-source/v1", "status": "research-only",
        "base_image": PUBLIC_BASE, "platform": "linux/arm64",
        "base_config_id": "sha256:6921a6c163ea40b603e19a0332330efe3dbccbf4dce9f6cbbf6b756c9231835a",
        "base_receipt_sha256": sha(base_bytes), "base_files": base_files,
        "runtime": runtime, "r27_binary_parity": False,
        "source_date_epoch": epoch, "site_packages": SITE, "sources": {},
        "retained_allowlist": retained_files(base_files), "critical": {**CRITICAL, **NCCL},
        "retained_vllm_native_source": "3633d61c3c7b04bb4d598cadbdc342f3be40482d",
        "warmup_argv": ["/opt/sparkring/bin/serve-with-warmup.py"],
        "startup_entrypoint": ["python3", "-S", "-B", ROOT + "/verify_sources.py", "--serve"],
    }
    payload = {}
    for name, source in sorted(spec["sources"].items()):
        data, record = source_archive(name, source, epoch)
        manifest["sources"][name] = record
        payload[name + ".tar"] = (data, 0o644)
    if spec.get("startup_override") is not None:
        data, record = startup_archive(spec["startup_override"], epoch)
        manifest["startup_override"] = record
        payload["startup.tar"] = (data, 0o644)
    source_paths = {p: h for source in manifest["sources"].values()
                    for p, h in source["files"].items()}
    # Maintained source takes precedence over generated Python support files;
    # native binaries must never be replaced by an unbuilt source overlay.
    for path, digest in list(manifest["retained_allowlist"].items()):
        if path in source_paths:
            if path.endswith(".so") and source_paths[path] != digest:
                raise ValueError(f"Source tries to replace retained binary: {path}")
            del manifest["retained_allowlist"][path]
    if spec.get("native_snapshot") is not None:
        from build_snapshot import RECIPE
        if spec["native_snapshot"] != RECIPE:
            raise ValueError("Unsupported native snapshot build recipe")
        manifest["native_snapshot"] = dict(RECIPE)
    lock_bytes = Path(spec["source_lock"]).read_bytes()
    lock = json.loads(lock_bytes)
    manifest["runtime_profiles"] = lock["profiles"]
    manifest["source_lock_sha256"] = sha(lock_bytes)
    payload["source-lock.json"] = (lock_bytes, 0o644)
    if spec.get("profile_assets") is not None:
        manifest["profile_assets"] = spec["profile_assets"]
        payload["profile-assets.tar"] = (Path(spec["profile_assets_archive"]).read_bytes(), 0o644)
    manifest["nccl_build"] = spec["nccl_build"]
    payload["nccl.tar"] = (Path(spec["nccl_archive"]).read_bytes(), 0o644)
    manifest["native_mode"] = "compile"
    if spec.get("native_files_archive") is not None:
        native_data = Path(spec["native_files_archive"]).read_bytes()
        manifest["native_files"] = describe_native_files(
            native_data, lock, manifest,
            {name: payload[name + ".tar"][0] for name in ("nccl", "sparkcache")})
        manifest["native_mode"] = "pinned"
        payload[NATIVE_ARCHIVE] = (native_data, 0o644)
    manifest["critical"][lock["runtime"]["nccl_path"]] = lock["runtime"]["nccl_sha256"]
    for name in ("snapshot", "placement"):
        manifest["critical"][lock["runtime"][name + "_path"]] = lock["runtime"][name + "_sha256"]
    for name in ("archive_utils.py", "install_sources.py", "verify_sources.py", "build_snapshot.py", "build_nccl.py", "receipt_contract.py", "nvcc_deterministic.py", "profile_assets.py", "native_files.py"):
        payload[name] = ((HERE / name).read_bytes(), 0o755 if name == "nvcc_deterministic.py" else 0o644)
    manifest["tool_hashes"] = {p: sha(v[0]) for p, v in payload.items() if p.endswith(".py")}
    payload["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n", 0o644)
    # ADD a canonical tar to avoid host-created COPY directory metadata.
    packed = make_archive({ROOT[1:] + "/" + p: value for p, value in payload.items()}, epoch)
    docker = [f"FROM {PUBLIC_BASE}", "USER root", "ADD payload.tar /",
              f"RUN python3 -S -B {ROOT}/install_sources.py stage && \\",
              *([f"    python3 -S -B {ROOT}/build_nccl.py && \\"] if manifest["native_mode"] == "compile" else []),
              f"    env -u PYTHONPATH UV_OFFLINE=1 UV_NO_CACHE=1 uv --no-cache pip install --system --no-deps --no-build-isolation --reinstall {ROOT}/b12x-source {ROOT}/sparkcache-source && \\",
              *([f"    python3 -S -B {ROOT}/build_snapshot.py && \\"] if manifest.get("native_snapshot") and manifest["native_mode"] == "compile" else []),
              f"    python3 -S -B {ROOT}/install_sources.py finalize && \\",
              f"    python3 -S -B {ROOT}/verify_sources.py",
              'LABEL org.sparkring.runtime.status="research-only"',
              'LABEL org.sparkring.runtime.r27-binary-parity="false"']
    for name, source in sorted(manifest["sources"].items()):
        docker.append(f'LABEL org.sparkring.source.{name}="{source["revision"]}"')
    if manifest.get("startup_override"):
        docker.append(f'LABEL org.sparkring.source.startup="{manifest["startup_override"]["revision"]}"')
    docker.append("ENTRYPOINT " + json.dumps(manifest["startup_entrypoint"]))
    files = {"Dockerfile": (("\n".join(docker) + "\n").encode(), 0o644), "payload.tar": (packed, 0o644)}
    context = make_archive(files, epoch)
    output.mkdir(parents=True)
    for name, (data, _) in files.items():
        (output / name).write_bytes(data)
    (output / "manifest.json").write_bytes(payload["manifest.json"][0])
    (output / "image-context.tar").write_bytes(context)
    receipt = {"schema": "sparkcache-jj-build-context/v1", "base_image": PUBLIC_BASE,
               "context_sha256": sha(context), "payload_sha256": sha(packed),
               "manifest_sha256": sha(payload["manifest.json"][0]),
               "source_revisions": {k: v["revision"] for k, v in manifest["sources"].items()},
               "built": False, "gpu_qualified": False}
    if manifest.get("startup_override"):
        receipt["startup_revision"] = manifest["startup_override"]["revision"]
    (output / "context-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(Path(args.spec).read_text()), args.output), indent=2))
