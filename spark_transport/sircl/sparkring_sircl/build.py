"""Build of the native ring-session library (``oneshot/_roce_proxy.c``).

The progress thread and the verbs code are one plain C file without a CUDA
dependency. It is compiled with the host C compiler (``CC``, else ``gcc``,
``cc``, ``clang``) as ``-O2 -std=gnu11 -shared -fPIC`` against ``ibverbs``,
``pthread`` and ``dl``, and cached as ``roce_proxy-<16 hex digits of the
source's SHA-256>.so`` in the build-cache directory: ``SIRCL_BUILD_CACHE_DIR``,
else ``<XDG cache home>/sircl/roce``. The object is written to a temporary
name and renamed into place, so concurrent builders never load a partial
file; a cached object of the same source hash is loaded without compiling.

The native binding builds on first load when no cached object exists. A
deployment runs ``sircl-prepare`` (this module's :func:`main`) at image build
or host preparation instead, so that session construction never compiles.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "oneshot" / "_roce_proxy.c"
LIBRARY_STEM = "roce_proxy"
COMPILERS = ("gcc", "cc", "clang")


class BuildError(RuntimeError):
    """The native library could not be built."""


def source_digest(source: Path = SOURCE) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()[:16]


def cache_dir() -> Path:
    configured = os.environ.get("SIRCL_BUILD_CACHE_DIR")
    if configured:
        return Path(configured)
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(root) / "sircl" / "roce"


def library_path(source: Path = SOURCE, directory: Path | None = None) -> Path:
    return (directory or cache_dir()) / f"{LIBRARY_STEM}-{source_digest(source)}.so"


def find_compiler() -> list[str]:
    configured = os.environ.get("CC")
    candidates = ([configured] if configured else []) + list(COMPILERS)
    for candidate in candidates:
        argv = shlex.split(candidate)
        if argv and shutil.which(argv[0]):
            return argv
    raise BuildError(
        f"no C compiler found (tried {', '.join(candidates)}); building SIRCL's native library "
        "needs a C compiler and the libibverbs development headers"
    )


def compile_command(compiler: list[str], source: Path, output: Path) -> list[str]:
    return [*compiler, "-O2", "-std=gnu11", "-shared", "-fPIC", "-o", str(output), str(source),
            "-libverbs", "-lpthread", "-ldl"]


def build(source: Path = SOURCE, directory: Path | None = None, *, force: bool = False) -> Path:
    """Return the cached library for ``source``, compiling it when absent (or when ``force``)."""
    target = library_path(source, directory)
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    command = compile_command(find_compiler(), source, temporary)
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise BuildError(
            "building SIRCL's native library failed:\n  " + shlex.join(command) + "\n"
            + (process.stderr or process.stdout).strip()
        )
    os.replace(temporary, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sircl-prepare",
        description="Build SIRCL's native ring-session library into the build cache.",
    )
    parser.add_argument("--cache-dir", type=Path, help="build-cache directory (default: "
                        "SIRCL_BUILD_CACHE_DIR or <XDG cache>/sircl/roce)")
    parser.add_argument("--force", action="store_true", help="rebuild even when cached")
    parser.add_argument("--print-path", action="store_true", help="print the library path and exit")
    args = parser.parse_args(argv)
    if args.print_path:
        print(library_path(directory=args.cache_dir))
        return 0
    try:
        path = build(directory=args.cache_dir, force=args.force)
    except BuildError as error:
        print(error, file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
