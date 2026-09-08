"""Package and verify profile-specific assets without importing GPU libraries."""
import json
from pathlib import Path

from archive_utils import make_archive, read_archive, sha, under
from receipt_contract import file_map_hash


SITE_HOOK = "/usr/local/lib/python3.12/dist-packages/sparkring_transport.pth"
TRANSPORT_ROOT = "/opt/sparkring/transports/"
PROFILE_ROOT = "/opt/sparkring/profiles/"


def destinations(lock):
    result = {}
    for record in lock.get("profile_assets", {}).values():
        path = record["destination"]
        if (path != SITE_HOOK and not path.startswith((TRANSPORT_ROOT, PROFILE_ROOT))):
            raise ValueError("Profile asset destination is outside its installation roots")
        under(Path("/"), path[1:])
        if path in result:
            raise ValueError("Two profile assets share an installation path")
        result[path] = record["sha256"]
    return result


def prepare_assets(repository, lock, epoch):
    expected = destinations(lock)
    files = {}
    for source, record in lock.get("profile_assets", {}).items():
        data = under(repository, source).read_bytes()
        if sha(data) != record["sha256"]:
            raise ValueError(f"Profile source asset differs from lock: {source}")
        files[record["destination"][1:]] = (data, 0o644)
    archive = make_archive(files, epoch)
    return archive, {"archive_sha256": sha(archive), "files": expected}


def install_assets(data, record, lock, rootfs):
    expected = destinations(lock)
    if record["files"] != expected or sha(data) != record["archive_sha256"]:
        raise ValueError("Profile asset archive differs from its locked inventory")
    files = read_archive(data, {path[1:]: digest for path, digest in expected.items()})
    paths = {path: under(rootfs, path) for path in files}
    # Check all preimages before writing any file. The locked parent has no
    # profile assets; matching files permit a repeated offline validation.
    for relative, target in paths.items():
        if target.exists() and (not target.is_file() or sha(target.read_bytes()) != expected["/" + relative]):
            raise ValueError("An installed file conflicts with a profile asset")
    for relative, (payload, mode) in files.items():
        paths[relative].parent.mkdir(parents=True, exist_ok=True)
        paths[relative].write_bytes(payload)
        paths[relative].chmod(mode)
    return expected


def verify_assets(lock, rootfs):
    for path, digest in destinations(lock).items():
        if sha(under(rootfs, path[1:]).read_bytes()) != digest:
            raise ValueError(f"Installed profile asset differs: {path}")
    witnesses = {}
    for profile in lock["profiles"].values():
        if profile.get("installed_profile_path") is not None:
            profile_path = under(rootfs, profile["installed_profile_path"][1:])
            if sha(profile_path.read_bytes()) != profile["profile_sha256"]:
                raise ValueError("Installed serving profile differs from source lock")
        name = profile.get("transport_profile")
        if name is None:
            continue
        bundle = under(rootfs, (TRANSPORT_ROOT + name)[1:])
        data = (bundle / "manifest.json").read_bytes()
        if sha(data) != profile["transport_manifest_sha256"]:
            raise ValueError("Installed transport manifest differs from serving profile")
        manifest = json.loads(data)
        expected = manifest["files"]
        if any(path.is_symlink() for path in bundle.rglob("*")):
            raise ValueError("Installed transport contains a symbolic link")
        actual = {path.relative_to(bundle).as_posix(): sha(path.read_bytes())
                  for path in bundle.rglob("*") if path.is_file()
                  and path != bundle / "manifest.json" and "__pycache__" not in path.parts}
        if actual != expected:
            raise ValueError("Installed transport inventory differs from its manifest")
        witnesses[name] = {"manifest_sha256": sha(data), "files_sha256": file_map_hash(expected),
                           "files": len(expected), "package": "b12x.comm.roce"}
    return witnesses
