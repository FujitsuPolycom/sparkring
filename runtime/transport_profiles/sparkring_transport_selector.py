"""Select a content-bound communication package before B12X imports it."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import site
import sys


ROOT = Path(__file__).resolve().parent
PROFILE = "tp2-rocenante-adaptive"
PROFILE_ENV = "SPARKRING_TRANSPORT_PROFILE"
DIGEST_ENV = "SPARKRING_TRANSPORT_MANIFEST_SHA256"
PACKAGE = "b12x.comm.roce"
HOOK_NAME = "sparkring_transport.pth"


def startup_hook(root: Path = ROOT) -> str:
    return (
        root.as_posix() + "\n"
        "import sparkring_transport_selector; "
        "sparkring_transport_selector.install_from_environment()\n"
    )


def verify_bundle(name: str, digest: str, root: Path = ROOT) -> Path:
    """Verify the manifest and the complete package before installing a finder."""
    if name != PROFILE:
        raise ValueError("Unknown SparkRing transport profile: " + name)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("A transport manifest SHA-256 is required")
    bundle = root / name
    manifest_path = bundle / "manifest.json"
    if bundle.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Transport bundle and manifest must not be symbolic links")
    data = manifest_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Transport manifest SHA-256 differs from the selected profile")
    manifest = json.loads(data)
    if manifest["schema"] != "sparkring-transport-bundle/v1" or manifest["name"] != name:
        raise ValueError("Transport manifest identity differs from the selected profile")
    expected = manifest["files"]
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path != manifest_path
        and "__pycache__" not in path.parts
    }
    if actual != set(expected):
        raise ValueError("Transport bundle file inventory differs from its manifest")
    for relative, expected_hash in expected.items():
        name_parts = PurePosixPath(relative)
        if (name_parts.is_absolute() or ".." in name_parts.parts
                or "\\" in relative or ":" in relative):
            raise ValueError("Invalid path in transport manifest: " + relative)
        path = bundle.joinpath(*name_parts.parts)
        ancestry = (path, *path.parents[:len(name_parts.parts)])
        if any(parent.is_symlink() for parent in ancestry):
            raise ValueError("Transport source must not be a symbolic link")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Transport source SHA-256 differs: " + relative)
    return bundle / "roce"


class SourceLoader(importlib.machinery.SourceFileLoader):
    """Compile the verified source instead of accepting an unrelated bytecode cache."""

    def get_code(self, fullname):
        return self.source_to_code(self.get_data(self.path), self.path)


class TransportFinder(importlib.abc.MetaPathFinder):
    """Override only b12x.comm.roce; model kernels keep their installed package."""

    def __init__(self, package: Path, digest: str):
        self.package = package
        self.digest = digest

    def find_spec(self, fullname, path=None, target=None):
        if fullname != PACKAGE and not fullname.startswith(PACKAGE + "."):
            return None
        relative = fullname[len(PACKAGE):].lstrip(".").split(".")
        target = self.package.joinpath(*relative) if relative != [""] else self.package
        package = target.is_dir()
        source = target / "__init__.py" if package else target.with_suffix(".py")
        if not source.is_file():
            raise ImportError("The selected transport has no module " + fullname)
        return importlib.util.spec_from_file_location(
            fullname, source, loader=SourceLoader(fullname, str(source)),
            submodule_search_locations=[str(target)] if package else None,
        )


def install_from_environment(root: Path = ROOT) -> None:
    """Abort Python startup on an invalid selection, including .pth execution.

    Python ignores Exception raised by a .pth hook. SystemExit deliberately
    escapes that handler so a worker cannot fall back to a different transport.
    """
    name = os.environ.get(PROFILE_ENV, "")
    if not name:
        return
    digest = os.environ.get(DIGEST_ENV, "")
    try:
        package = verify_bundle(name, digest, root)
        existing = [item for item in sys.meta_path if isinstance(item, TransportFinder)]
        if existing:
            if len(existing) != 1 or existing[0].package != package or existing[0].digest != digest:
                raise ValueError("A different SparkRing transport is already selected")
            return
        if any(key == PACKAGE or key.startswith(PACKAGE + ".") for key in sys.modules):
            raise ValueError("Select the transport before importing b12x.comm.roce")
        sys.meta_path.insert(0, TransportFinder(package, digest))
    except Exception as error:
        raise SystemExit("SparkRing transport activation failed: " + str(error)) from error


def require_active(*, require_startup_hook: bool = True) -> None:
    """Gate the serving entrypoint before vLLM or GPU initialization."""
    name = os.environ.get(PROFILE_ENV, "")
    if not name:
        raise RuntimeError("The serving profile must select a transport explicitly")
    digest = os.environ.get(DIGEST_ENV, "")
    package = verify_bundle(name, digest)
    if not sys.meta_path or not isinstance(sys.meta_path[0], TransportFinder):
        raise RuntimeError("The selected transport import hook is not active")
    finder = sys.meta_path[0]
    if finder.package != package or finder.digest != digest:
        raise RuntimeError("The active transport differs from the serving profile")
    if require_startup_hook:
        installed = [Path(path) / HOOK_NAME for path in site.getsitepackages()]
        if not any(path.is_file() and path.read_text() == startup_hook() for path in installed):
            raise RuntimeError("Install the transport .pth hook for spawned Python workers")


__all__ = ["install_from_environment", "require_active", "startup_hook", "verify_bundle"]
