"""Build of the point-to-point native library (``p2p/_p2p_proxy.c``).

The library is one plain C file compiled like SIRCL's collective library
(:mod:`sparkring_sircl.build`: the same compiler choice and flags, against
``ibverbs``, ``pthread`` and ``dl``) and cached beside it in the build-cache
directory (``SIRCL_BUILD_CACHE_DIR``, else ``<XDG cache home>/sircl/roce``) as
``p2p_proxy-<16 hex digits of the source's SHA-256>.so``. A cached object of
the same source hash is loaded without compiling; the object is written to a
temporary name and renamed into place, so concurrent builders never load a
partial file.

``python -m sparkring_sircl.p2p.build`` builds it into the cache (a
deployment runs it beside ``sircl-prepare`` so that setup never compiles);
``SIRCL_P2P_NATIVE_LIBRARY`` names a prebuilt library to load instead.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .. import build as core_build

SOURCE = Path(__file__).resolve().parent / "_p2p_proxy.c"
LIBRARY_STEM = "p2p_proxy"


def source_digest(source: Path = SOURCE) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()[:16]


def library_path(source: Path = SOURCE, directory: Path | None = None) -> Path:
    return (directory or core_build.cache_dir()) / f"{LIBRARY_STEM}-{source_digest(source)}.so"


def build(source: Path = SOURCE, directory: Path | None = None, *, force: bool = False) -> Path:
    """Return the cached library for ``source``, compiling it when absent (or when ``force``)."""
    target = library_path(source, directory)
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    command = core_build.compile_command(core_build.find_compiler(), source, temporary)
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise core_build.BuildError(
            "building SIRCL's point-to-point library failed:\n  " + shlex.join(command) + "\n"
            + (process.stderr or process.stdout).strip()
        )
    os.replace(temporary, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sparkring_sircl.p2p.build",
                                     description="Build SIRCL's point-to-point native library into the build cache.")
    parser.add_argument("--cache-dir", type=Path, help="build-cache directory (default: SIRCL_BUILD_CACHE_DIR or "
                        "<XDG cache>/sircl/roce)")
    parser.add_argument("--force", action="store_true", help="rebuild even when cached")
    parser.add_argument("--print-path", action="store_true", help="print the library path and exit")
    args = parser.parse_args(argv)
    if args.print_path:
        print(library_path(directory=args.cache_dir))
        return 0
    try:
        path = build(directory=args.cache_dir, force=args.force)
    except core_build.BuildError as error:
        print(error, file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
