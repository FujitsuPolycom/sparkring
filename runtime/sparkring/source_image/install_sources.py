"""Container-only source installation with no framework imports or CUDA use."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from archive_utils import STARTUP_PATHS, WARMUP_SOURCE, extract_checked, inventory, sha, under, transform_warmup
from profile_assets import install_assets, verify_assets

ROOT = Path(__file__).resolve().parent


def distribution_records(site):
    result = {}
    for dist in importlib.metadata.distributions(path=[str(site)]):
        name = dist.metadata.get("Name", "").lower().replace("_", "-")
        if name:
            result[name] = {"version": dist.version,
                            "entry_points": {f"{e.group}:{e.name}": e.value for e in dist.entry_points}}
    return result


def host_path(rootfs, absolute):
    if not absolute.startswith("/"):
        raise ValueError("Expected an absolute container path")
    return under(rootfs, absolute[1:])


def verify_map(rootfs, files):
    for absolute, digest in files.items():
        path = host_path(rootfs, absolute)
        if not path.is_file() or sha(path.read_bytes()) != digest:
            raise RuntimeError(f"File does not match manifest: {absolute}")


def read_manifest(root):
    manifest = json.loads((root / "manifest.json").read_bytes())
    if manifest["schema"] != "sparkcache-jj-runtime-source/v1":
        raise ValueError("Unsupported source manifest")
    for name, digest in manifest["tool_hashes"].items():
        if sha(under(root, name).read_bytes()) != digest:
            raise RuntimeError(f"Runtime verifier/tool changed: {name}")
    return manifest


def stage(root=ROOT, rootfs=Path("/")):
    manifest = read_manifest(root)
    if (root / "base-state.json").exists():
        raise RuntimeError("Installation already staged; do not reuse its state")
    verify_map(rootfs, manifest["base_files"])
    site = host_path(rootfs, manifest["site_packages"])
    distributions = distribution_records(site)
    for name, expected in manifest["runtime"]["expected_distributions"].items():
        if distributions.get(name, {}).get("version") != expected:
            raise RuntimeError(f"Inherited runtime version differs: {name}")
    protected = {p: h for p, h in manifest["base_files"].items()
                 if not any(p.startswith(str(manifest["site_packages"]) + "/" + n + "/")
                            for n in manifest["sources"])}
    protected.update(manifest["runtime"].get("required_files", {}))
    verify_map(rootfs, protected)
    # Digest-pinned base supplies wrappers; retain their full before/after bytes.
    warmup_dir = host_path(rootfs, "/opt/sparkring/bin")
    if not host_path(rootfs, manifest["warmup_argv"][0]).is_file():
        raise RuntimeError("Existing warmup entrypoint is missing")
    protected.update({"/opt/sparkring/bin/" + p: h for p, h in inventory(warmup_dir).items()})
    for name, source in manifest["sources"].items():
        data = (root / (name + ".tar")).read_bytes()
        if sha(data) != source["archive_sha256"]:
            raise RuntimeError(f"Source archive changed: {name}")
        extract_checked(data, root / (name + "-source"), source["files"])
    startup = manifest.get("startup_override")
    if startup is not None:
        if startup["install_paths"] != STARTUP_PATHS or set(startup["files"]) != set(STARTUP_PATHS):
            raise RuntimeError("Unsupported startup override file mapping")
        data = (root / "startup.tar").read_bytes()
        if sha(data) != startup["archive_sha256"]:
            raise RuntimeError("Startup archive changed")
        extract_checked(data, root / "startup-source", startup["files"])
    state = {"schema": "sparkcache-jj-base-state/v1", "distributions": distributions,
             "protected_files": protected, "retained_files": manifest["retained_allowlist"],
             "source_manifest_sha256": sha((root / "manifest.json").read_bytes())}
    (root / "base-state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return state


def replace_package(site, name, source_root, expected, retained):
    installed = under(site, name)
    # Validation precedes deletion and includes parent symlinks.
    old = inventory(installed)
    for relative, digest in expected.items():
        if relative.startswith(name + "/"):
            if sha(under(source_root, relative).read_bytes()) != digest:
                raise RuntimeError(f"Staged source changed: {relative}")
    for relative, digest in retained.items():
        if relative.startswith(name + "/"):
            path = under(site, relative)
            if not path.is_file() or sha(path.read_bytes()) != digest:
                raise RuntimeError(f"Retained dependency changed: {relative}")
    for relative in old:
        full = name + "/" + relative
        if full not in retained:
            under(installed, relative).unlink()
    for relative, digest in expected.items():
        if not relative.startswith(name + "/"):
            continue
        original = under(source_root, relative)
        if sha(original.read_bytes()) != digest:
            raise RuntimeError(f"Staged source changed: {relative}")
        target = under(site, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
        target.chmod(original.stat().st_mode & 0o777)


def finalize(root=ROOT, rootfs=Path("/")):
    manifest = read_manifest(root)
    state = json.loads((root / "base-state.json").read_bytes())
    if state["source_manifest_sha256"] != sha((root / "manifest.json").read_bytes()):
        raise RuntimeError("Manifest changed during installation")
    if state["retained_files"] != manifest["retained_allowlist"]:
        raise RuntimeError("Retained allowlist changed during installation")
    site = host_path(rootfs, manifest["site_packages"])
    verify_map(rootfs, state["protected_files"])
    for name, source in manifest["sources"].items():
        replace_package(site, name, root / (name + "-source"), source["files"], state["retained_files"])
    startup = manifest.get("startup_override")
    if startup is not None:
        if startup["install_paths"] != STARTUP_PATHS:
            raise RuntimeError("Unsupported startup override mapping")
        # All preimages were checked above; validate every replacement before any write.
        replacements = {}
        for source, destination in STARTUP_PATHS.items():
            data = under(root / "startup-source", source).read_bytes()
            if sha(data) != startup["files"][source]:
                raise RuntimeError(f"Startup source changed: {source}")
            transform = startup.get("transform")
            if source == WARMUP_SOURCE and transform is not None:
                if transform["source_file"] != source or sha(data) != transform["raw_sha256"]:
                    raise RuntimeError("Warmup transform preimage differs")
                data = transform_warmup(data, transform["name"])
                if sha(data) != transform["installed_sha256"]:
                    raise RuntimeError("Warmup transform output differs")
            if sha(data) != startup["installed_files"][destination]:
                raise RuntimeError(f"Startup installed bytes differ: {destination}")
            replacements[destination] = data
        state["startup_preimages"] = {p: state["protected_files"].get(p) for p in replacements}
        for destination, data in replacements.items():
            target = host_path(rootfs, destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755 if destination.endswith("/serve-with-warmup.py") else 0o644)
            state["protected_files"][destination] = sha(data)
        state["startup_files"] = {p: sha(data) for p, data in replacements.items()}
    distributions = distribution_records(site)
    for name, before in state["distributions"].items():
        if name not in ("b12x", "sparkcache") and distributions.get(name) != before:
            raise RuntimeError(f"Dependency metadata changed during install: {name}")
    if set(distributions) - set(state["distributions"]) - {"b12x", "sparkcache"}:
        raise RuntimeError("Unexpected dependency was installed")
    for name in ("b12x", "sparkcache"):
        source = manifest["sources"][name]
        observed = distributions.get(name, {})
        if observed.get("version") != source["distribution_version"]:
            raise RuntimeError(f"Installed distribution version differs: {name}")
        expected = {f"{group}:{entry}": value for group, entries in source["entry_points"].items()
                    for entry, value in entries.items()}
        expected.update({"console_scripts:" + entry: value for entry, value in source["console_scripts"].items()})
        if observed["entry_points"] != expected:
            raise RuntimeError(f"Installed entry points differ: {name}")
    state["installed_distributions"] = distributions
    if manifest.get("profile_assets") is not None:
        lock_bytes = (root / "source-lock.json").read_bytes()
        if sha(lock_bytes) != manifest["source_lock_sha256"]:
            raise ValueError("Profile source lock changed during installation")
        lock = json.loads(lock_bytes)
        state["profile_assets"] = install_assets((root / "profile-assets.tar").read_bytes(),
                                                 manifest["profile_assets"], lock, rootfs)
        state["transport_profiles"] = verify_assets(lock, rootfs)
        state["protected_files"].update(state["profile_assets"])
    if manifest.get("native_snapshot") is not None:
        from build_snapshot import LIBRARY, RECIPE
        receipt_bytes = (root / "snapshot-build-receipt.json").read_bytes()
        receipt = json.loads(receipt_bytes)
        if (receipt["source_manifest_sha256"] != state["source_manifest_sha256"]
                or receipt["sparkcache_revision"] != manifest["sources"]["sparkcache"]["revision"]
                or receipt["recipe"] != RECIPE or manifest["native_snapshot"] != RECIPE
                or set(receipt["files"]) != {LIBRARY}):
            raise RuntimeError("Native snapshot receipt does not match the source build")
        verify_map(rootfs, receipt["files"])
        state["snapshot_build_receipt_sha256"] = sha(receipt_bytes)
        state["protected_files"].update(receipt["files"])
    # Preserve all metadata and generated scripts after a reviewed offline install.
    generated = {}
    for name in ("b12x", "sparkcache"):
        for directory in site.glob(name + "-*.dist-info"):
            generated.update({manifest["site_packages"] + "/" + directory.name + "/" + p: h
                              for p, h in inventory(directory).items()})
        for script in manifest["sources"][name]["console_scripts"]:
            path = host_path(rootfs, "/usr/local/bin/" + script)
            if not path.is_file():
                raise RuntimeError(f"Missing generated console script: {script}")
            generated["/usr/local/bin/" + script] = sha(path.read_bytes())
    state["generated_install_files"] = generated
    (root / "installed-state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    # This normalizes touched source/install outputs, not unrelated runtime files.
    directories = [root, *[site / n for n in manifest["sources"]],
                   *[site / Path(p).relative_to(manifest["site_packages"]).parts[0]
                     for p in generated if p.startswith(manifest["site_packages"] + "/")]]
    paths = {p for directory in directories for p in [directory, *directory.rglob("*")]}
    paths.update(host_path(rootfs, p) for p in generated)
    paths.update(host_path(rootfs, p) for p in state.get("startup_files", {}))
    paths.update(host_path(rootfs, p) for p in state.get("profile_assets", {}))
    if startup is not None:
        paths.add(host_path(rootfs, "/opt/sparkring/bin"))
    paths.update((site, root.parent, host_path(rootfs, "/usr/local/bin")))
    for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
        if path.exists() and not path.is_symlink():
            os.utime(path, (manifest["source_date_epoch"],) * 2)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("stage", "finalize"))
    args = parser.parse_args()
    if not sys.flags.no_site:
        raise RuntimeError("Run this installer with python3 -S -B")
    result = stage() if args.phase == "stage" else finalize()
    print(json.dumps({"phase": args.phase, "checks_passed": True,
                      "source_manifest_sha256": result["source_manifest_sha256"]}))
