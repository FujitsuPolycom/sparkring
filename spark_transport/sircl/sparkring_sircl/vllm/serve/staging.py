"""The package tree a serving container imports SIRCL and its vLLM plugin from.

A staged tree holds every source file of ``sparkring_sircl`` (Python, the
native C source and headers, JSON data), including the vLLM adapter, plus a
generated ``sparkring_sircl-<version>.dist-info`` directory beside the package
and SparkRing's RoCE GID resolver (``spark_roce_gid.py``, which
:mod:`sparkring_sircl.roce_gid` imports) at the top of the tree.
vLLM discovers plugins through ``importlib.metadata`` entry points, which scan
every ``sys.path`` directory for ``*.dist-info``; mounting the tree at a
``PYTHONPATH`` directory therefore makes the ``sircl`` platform and general
plugins visible without installing anything into the image.

The tree is content-addressed: its digest covers every file's relative path
and bytes, generated files included, and names its directory on the Sparks
(``<remote_dir>/serve/src/<digest>``), which is written once.

The native library is not part of the tree. The stage step builds it inside
the serving image into ``<remote_dir>/build-cache`` as
``roce_proxy-<16 hex digits of the C source's SHA-256>.so``
(:mod:`sparkring_sircl.build`), the file the ring harness builds from the same
source.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import tarfile
from collections.abc import Mapping
from pathlib import Path

from ... import __version__
from ... import build as native_build
from ... import roce_gid

PACKAGE = Path(__file__).resolve().parents[2]          # .../sparkring_sircl
PROJECT = PACKAGE.parent                               # .../spark_transport/sircl
SUFFIXES = (".py", ".c", ".h", ".json")
DISTRIBUTION = "sparkring_sircl"
# The entry points of pyproject.toml; checked against it by the CPU tests.
ENTRY_POINTS: Mapping[str, Mapping[str, str]] = {
    "vllm.general_plugins": {"sircl": "sparkring_sircl.vllm.plugin:register"},
    "vllm.platform_plugins": {"sircl": "sparkring_sircl.vllm.platform:activate"},
}


@dataclasses.dataclass(frozen=True)
class StagedTree:
    files: tuple[tuple[str, bytes], ...]     # (relative POSIX path, content), sorted

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        for name, content in self.files:
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(content).digest())
        return digest.hexdigest()[:16]

    @property
    def size(self) -> int:
        return sum(len(content) for _, content in self.files)

    def names(self) -> list[str]:
        return [name for name, _ in self.files]

    def tar(self) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, content in self.files:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode = 0o644
                info.mtime = 0
                archive.addfile(info, io.BytesIO(content))
        return buffer.getvalue()


def entry_points_text(entry_points: Mapping[str, Mapping[str, str]] = ENTRY_POINTS) -> str:
    sections = []
    for group in sorted(entry_points):
        lines = [f"[{group}]"] + [f"{name} = {target}" for name, target in sorted(entry_points[group].items())]
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def dist_info(version: str = __version__) -> dict[str, bytes]:
    folder = f"{DISTRIBUTION}-{version}.dist-info"
    metadata = (f"Metadata-Version: 2.1\nName: sparkring-sircl\nVersion: {version}\n"
                "Summary: SIRCL ring sessions and their vLLM adapter (staged by the serve launcher)\n")
    return {
        f"{folder}/METADATA": metadata.encode(),
        f"{folder}/entry_points.txt": entry_points_text().encode(),
        f"{folder}/top_level.txt": b"sparkring_sircl\n",
    }


def package_files(package: Path = PACKAGE) -> list[Path]:
    files = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package).parts
        if not path.is_file() or "__pycache__" in relative or path.suffix not in SUFFIXES:
            continue
        files.append(path)
    return files


def staged_tree(package: Path = PACKAGE) -> StagedTree:
    entries = {path.relative_to(package.parent).as_posix(): path.read_bytes()
               for path in package_files(package)}
    entries.update(dist_info())
    resolver = roce_gid.source_file()
    entries[resolver.name] = resolver.read_bytes()
    return StagedTree(tuple(sorted(entries.items())))


def library_name() -> str:
    """File name of the native library the stage step builds (same rule as the build cache)."""
    return native_build.library_path(directory=Path("/")).name
