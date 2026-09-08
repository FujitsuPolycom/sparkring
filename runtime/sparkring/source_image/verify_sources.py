"""Verify all source/native bytes before entering the existing warmup gate."""
import importlib.machinery
import json
import os
from pathlib import Path
import sys

from archive_utils import STARTUP_PATHS, inventory, sha
from install_sources import ROOT, distribution_records, host_path, read_manifest, verify_map
from receipt_contract import file_map_hash
from profile_assets import destinations, verify_assets


def verify(root=ROOT, rootfs=Path("/")):
    manifest = read_manifest(root)
    state = json.loads((root / "installed-state.json").read_bytes())
    if state["source_manifest_sha256"] != sha((root / "manifest.json").read_bytes()):
        raise RuntimeError("Manifest changed after source installation")
    if state["retained_files"] != manifest["retained_allowlist"]:
        raise RuntimeError("Retained allowlist changed after installation")
    verify_map(rootfs, manifest["critical"])
    verify_map(rootfs, state["protected_files"])
    verify_map(rootfs, state["generated_install_files"])
    if manifest.get("native_snapshot") is not None:
        receipt_bytes = (root / "snapshot-build-receipt.json").read_bytes()
        if sha(receipt_bytes) != state.get("snapshot_build_receipt_sha256"):
            raise RuntimeError("Native snapshot build receipt changed")
        receipt = json.loads(receipt_bytes)
        verify_map(rootfs, receipt["files"])
    startup = manifest.get("startup_override")
    if startup is not None:
        if startup["install_paths"] != STARTUP_PATHS:
            raise RuntimeError("Unsupported startup override mapping")
        startup_expected = startup["installed_files"]
        if set(startup_expected) != set(STARTUP_PATHS.values()):
            raise RuntimeError("Startup override installed file set differs")
        if state.get("startup_files") != startup_expected:
            raise RuntimeError("Installed startup component differs from source manifest")
        verify_map(rootfs, startup_expected)
    elif state.get("startup_files"):
        raise RuntimeError("Undeclared startup override")
    site = host_path(rootfs, manifest["site_packages"])
    packages = {}
    for name, source in manifest["sources"].items():
        expected = {p[len(name) + 1:]: h for p, h in source["files"].items()
                    if p.startswith(name + "/")}
        expected.update({p[len(name) + 1:]: h for p, h in state["retained_files"].items()
                         if p.startswith(name + "/")})
        actual = inventory(site / name)
        # Python may emit bytecode after serving. No extra source/data/native file is accepted.
        actual = {p: h for p, h in actual.items()
                  if not ("__pycache__" in p.split("/") and p.endswith(".pyc"))}
        if actual != expected:
            bad = sorted(p for p in set(actual) | set(expected) if actual.get(p) != expected.get(p))
            raise RuntimeError(f"Installed {name} file set/content differs: {bad[:8]}")
        packages[name] = {"revision": source["revision"], "file_map_sha256": file_map_hash(expected), "files": len(expected)}
    if distribution_records(site) != state["installed_distributions"]:
        raise RuntimeError("Runtime distribution metadata differs")
    result = {"schema": "sparkcache-jj-source-witness/v1", "checks_passed": True,
            "source_revisions": {k: v["revision"] for k, v in manifest["sources"].items()},
            "source_manifest_sha256": state["source_manifest_sha256"],
            "inherited_runtime": {k: v["version"] for k, v in state["installed_distributions"].items()},
            "r27_binary_parity": False, "gpu_qualified": False}
    if startup is not None:
        result["startup_component"] = {"revision": startup["revision"], "files": startup_expected,
                                       "transform": startup.get("transform")}
    lock_bytes = (root / "source-lock.json").read_bytes()
    if sha(lock_bytes) != manifest["source_lock_sha256"]:
        raise RuntimeError("Source lock differs from prepared context")
    lock = json.loads(lock_bytes)
    if manifest.get("runtime_profiles") != lock["profiles"]:
        raise RuntimeError("Runtime profile declarations differ from source lock")
    if state.get("profile_assets", {}) != destinations(lock):
        raise RuntimeError("Installed profile asset inventory differs from source lock")
    result["transport_profiles"] = verify_assets(lock, rootfs)
    if result["transport_profiles"] != state.get("transport_profiles", {}):
        raise RuntimeError("Installed transport witness differs from its build record")
    identities = {
        "bundle_manifest_sha256": "/opt/spark-sircl/sparkring-overlay-manifest.json",
        "transport_sha256": "/opt/spark-sircl/libspark_transport_capi.so",
        "marker_source_sha256": "/opt/sparkring/src/mtp3-mesh/mlx5_rdma_tx_rewrite_probe.c",
        "marker_binary_sha256": "/opt/sparkring/bin/mlx5-rdma-tx-marker",
        "nccl_sha256": lock["runtime"]["nccl_path"],
        "snapshot_sha256": lock["runtime"]["snapshot_path"],
        "placement_sha256": lock["runtime"]["placement_path"],
    }
    for field, path in identities.items():
        digest = sha(host_path(rootfs, path).read_bytes())
        if digest != lock["runtime"][field]:
            raise RuntimeError(f"Runtime identity differs: {field}")
        result[field] = digest
    warmup = lock["runtime"]["readiness_warmup"]
    if sha(host_path(rootfs, "/opt/sparkring/bin/warmup_dflash.py").read_bytes()) != warmup["helper_sha256"]:
        raise RuntimeError("Readiness warmup helper differs from source lock")
    result["readiness_warmup"] = warmup
    native = {p: h for p, h in state["retained_files"].items() if p.endswith(".so")}
    if "torch" in sys.modules:
        raise RuntimeError("CPU-only verifier unexpectedly imported Torch")
    result.update(source_lock_sha256=sha(lock_bytes), packages=packages,
                  retained_vllm_native_sha256=file_map_hash(native),
                  cuda_initialized=False, model_loaded=False)
    for name, expected in lock["sources"].items():
        if (packages[name]["file_map_sha256"] != expected["installed_file_map_sha256"]
                or packages[name]["files"] != expected["installed_file_count"]):
            raise RuntimeError(f"Installed package inventory differs from source lock: {name}")
    if result["retained_vllm_native_sha256"] != lock["runtime"]["retained_vllm_native_sha256"]:
        raise RuntimeError("Retained vLLM native inventory differs from source lock")
    return result


def check_import_paths(manifest):
    # PathFinder discovers top-level packages without importing their code.
    # -S omits site-packages, so append the normal site location after PYTHONPATH.
    search = [p for p in sys.path if p] + [manifest["site_packages"]]
    for name in manifest["sources"]:
        spec = importlib.machinery.PathFinder.find_spec(name, search)
        expected = Path(manifest["site_packages"]) / name
        if spec is None or not spec.submodule_search_locations or any(
                Path(p).resolve() != expected.resolve() for p in spec.submodule_search_locations):
            raise RuntimeError(f"Runtime import path shadows verified package: {name}")


def serving_argv(manifest, profile, arguments, executable=sys.executable):
    """Select an attested profile entrypoint after common source verification."""
    if profile and profile not in manifest.get("runtime_profiles", {}):
        raise ValueError("Serving profile is absent from the verified image")
    if profile == "glm53-flash-nvfp4-tp2-mtp3":
        return [executable, "/opt/sparkring/transports/entrypoint.py", "serve", *arguments]
    return manifest["warmup_argv"] + arguments


if __name__ == "__main__":
    if not sys.flags.no_site:
        raise RuntimeError("Run source verification with python3 -S -B")
    print(json.dumps(verify()), flush=True)
    if "--serve" in sys.argv:
        manifest = read_manifest(ROOT)
        check_import_paths(manifest)
        argv = serving_argv(manifest, os.environ.get("SOURCE_IMAGE_PROFILE", ""),
                            sys.argv[sys.argv.index("--serve") + 1:])
        os.execv(argv[0], argv)
