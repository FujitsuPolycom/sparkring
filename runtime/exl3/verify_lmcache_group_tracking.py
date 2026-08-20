#!/usr/bin/env python3
"""Offline verification that an LMCache package tracks KV layout per layer group.

An LMCache package that assumes one packed KV format across all layers cannot
serve a checkpoint whose registration declares several kernel groups: a single
staging-buffer size cannot satisfy groups that differ in block size or element
size. Multiprocess per-group tracking was added upstream in LMCache pull
request 3171 (merge commit ``384d79df5c3a023ccfebedc2b69b094b0d7b7084``, merged
2026-05-14 into ``dev``), which introduced ``lmcache/v1/kv_layer_groups.py`` and
carried ``block_size`` into the kernel-group identity.

This checker reads an extracted or installed package directory and decides
whether that capability is present, whether the symbols this repository's
serving recipe binds to still exist, and whether two known defects are still
live. It parses source text and never imports the package, so it runs on a
CPU-only workstation with no CUDA toolchain and no LMCache dependencies
installed.

Safety class: OFFLINE. Reads only the named directory.

Usage::

    python runtime/exl3/verify_lmcache_group_tracking.py --package-dir PATH
    python runtime/exl3/verify_lmcache_group_tracking.py --package-dir PATH
        --expect-version 0.5.3+glm52dcp4.2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = "sparkring-lmcache-group-tracking/v1"

EXIT_OK = 0
EXIT_FAIL = 2
EXIT_CONFIG_ERROR = 3

# Module introduced by pull request 3171. Its presence is the decisive artifact
# of per-group tracking; packages predating that change have no counterpart.
GROUP_MODULE = "v1/kv_layer_groups.py"

# The kernel-group identity must separate layers whose block sizes differ.
# Without block_size in the identity, a compressed and an uncompressed group
# collapse into one entry and the transfer is sized from the wrong geometry.
IDENTITY_SYMBOL = "KernelGroupIdentity"
IDENTITY_FIELD = "block_size"

# Padded views taken from the engine's unified KV pool are described by an
# explicit physical stride rather than being assumed contiguous.
STRIDE_FIELD = "block_stride_elems"

# The REGISTER_KV_CACHE payload carries the engine block size so the server
# sizes buffers per group rather than from one global block size. A package
# without this field is wire-incompatible with one that has it.
PROTOCOL_FIELD = "vllm_block_size"

# The heartbeat guard is defective when it tests the container for existence
# rather than for contents: an empty dictionary is not None, so the branch is
# taken and the heartbeat thread never starts. The server then reaps a live
# engine while stores continue and lookups stop silently.
HEARTBEAT_DEFECT = re.compile(r"if\s+(?:self\.)?_?heartbeats\s+is\s+not\s+None\s*:")

# Symbol added by this repository's four-local-server topology patch. Its
# presence identifies a tree carrying that local change.
LOCAL_TOPOLOGY_SYMBOL = "local_server_url_for_worker"

# Rejection text emitted by packages that permit only one cache server for MLA
# plus decode context parallelism. A tree containing it cannot run the
# one-server-per-rank topology this repository deploys.
SINGLE_SERVER_REJECTION = "requires one LMCache server"

VERSION_PATTERN = re.compile(
    r'^__version__\s*(?::[^=]+)?=\s*[\'"]([^\'"]+)[\'"]', re.MULTILINE
)
VERSION_SOURCES = ("__init__.py", "_version.py", "version.py")


class ConfigError(Exception):
    """The named directory cannot be read as an LMCache package."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - surfaced as a config error
        raise ConfigError(f"cannot read {path}: {exc}") from exc


def python_sources(package: Path) -> list[Path]:
    """Return every Python source file in the package, in stable order."""
    return sorted(package.rglob("*.py"))


def declared_version(package: Path) -> str | None:
    """Return the package's declared ``__version__``, or None when absent.

    The value is parsed from source text rather than imported: importing
    LMCache pulls in CUDA extensions that are absent on a build workstation.
    """
    for name in VERSION_SOURCES:
        candidate = package / name
        if not candidate.is_file():
            continue
        match = VERSION_PATTERN.search(_read(candidate))
        if match:
            return match.group(1)
    return None


def find_symbol(sources: Iterable[Path], package: Path, symbol: str) -> list[str]:
    """Return package-relative paths of files mentioning ``symbol``."""
    hits = []
    for path in sources:
        if symbol in _read(path):
            hits.append(path.relative_to(package).as_posix())
    return hits


def find_pattern(
    sources: Iterable[Path], package: Path, pattern: re.Pattern[str]
) -> list[str]:
    """Return ``path:line`` locations at which ``pattern`` matches."""
    hits = []
    for path in sources:
        for number, line in enumerate(_read(path).splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(package).as_posix()}:{number}")
    return hits


def module_defines(package: Path, relative: str, symbol: str) -> bool:
    """Return whether ``relative`` defines ``symbol`` as a class or assignment."""
    path = package / relative
    if not path.is_file():
        return False
    definition = re.compile(
        rf"^\s*(?:class\s+{re.escape(symbol)}\b|{re.escape(symbol)}\s*=)",
        re.MULTILINE,
    )
    return bool(definition.search(_read(path)))


def _check(name: str, required: bool, passed: bool, detail: Any) -> dict[str, Any]:
    return {
        "name": name,
        "required": required,
        "passed": passed,
        "detail": detail,
    }


def verify(
    package: Path,
    expect_version: str | None = None,
    connector_module: str = "integration/vllm/lmcache_mp_connector.py",
    connector_class: str = "LMCacheMPConnector",
) -> dict[str, Any]:
    """Verify per-group tracking and binding stability for a package tree."""
    if not package.is_dir():
        raise ConfigError(f"not a directory: {package}")
    if not (package / "__init__.py").is_file():
        raise ConfigError(
            f"{package} has no __init__.py; expected the lmcache package "
            "directory itself, not its parent"
        )

    sources = python_sources(package)
    if not sources:
        raise ConfigError(f"{package} contains no Python sources")

    version = declared_version(package)
    group_module_present = (package / GROUP_MODULE).is_file()

    checks: list[dict[str, Any]] = []

    checks.append(
        _check(
            "group_module_present",
            True,
            group_module_present,
            {"path": GROUP_MODULE},
        )
    )
    checks.append(
        _check(
            "kernel_group_identity_defined",
            True,
            module_defines(package, GROUP_MODULE, IDENTITY_SYMBOL),
            {"module": GROUP_MODULE, "symbol": IDENTITY_SYMBOL},
        )
    )
    identity_carries_block_size = group_module_present and (
        IDENTITY_FIELD in _read(package / GROUP_MODULE)
    )
    checks.append(
        _check(
            "identity_separates_block_size",
            True,
            identity_carries_block_size,
            {"module": GROUP_MODULE, "field": IDENTITY_FIELD},
        )
    )

    stride_hits = find_symbol(sources, package, STRIDE_FIELD)
    checks.append(
        _check(
            "padded_stride_described",
            True,
            bool(stride_hits),
            {"field": STRIDE_FIELD, "files": stride_hits},
        )
    )

    protocol_hits = find_symbol(sources, package, PROTOCOL_FIELD)
    checks.append(
        _check(
            "registration_carries_engine_block_size",
            True,
            bool(protocol_hits),
            {"field": PROTOCOL_FIELD, "files": protocol_hits},
        )
    )

    checks.append(
        _check(
            "recipe_connector_symbol_present",
            True,
            module_defines(package, connector_module, connector_class),
            {"module": connector_module, "symbol": connector_class},
        )
    )

    heartbeat_hits = find_pattern(sources, package, HEARTBEAT_DEFECT)
    checks.append(
        _check(
            "heartbeat_guard_tests_contents",
            True,
            not heartbeat_hits,
            {"defective_locations": heartbeat_hits},
        )
    )

    rejection_hits = find_symbol(sources, package, SINGLE_SERVER_REJECTION)
    checks.append(
        _check(
            "multi_server_topology_permitted",
            True,
            not rejection_hits,
            {"rejection_text": SINGLE_SERVER_REJECTION, "files": rejection_hits},
        )
    )

    if expect_version is not None:
        checks.append(
            _check(
                "declared_version_matches",
                True,
                version == expect_version,
                {"expected": expect_version, "observed": version},
            )
        )

    # Reported but not required: upstream multi-server support may supersede the
    # local topology patch, so its absence is not by itself a failure.
    checks.append(
        _check(
            "local_topology_patch_applied",
            False,
            bool(find_symbol(sources, package, LOCAL_TOPOLOGY_SYMBOL)),
            {
                "symbol": LOCAL_TOPOLOGY_SYMBOL,
                "files": find_symbol(sources, package, LOCAL_TOPOLOGY_SYMBOL),
            },
        )
    )

    failed = [
        check["name"] for check in checks if check["required"] and not check["passed"]
    ]
    return {
        "schema": SCHEMA,
        "package_dir": package.as_posix(),
        "declared_version": version,
        "source_file_count": len(sources),
        "checks": checks,
        "failed_checks": failed,
        "verdict": "fail" if failed else "pass",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        required=True,
        help="path to the lmcache package directory (the one holding __init__.py)",
    )
    parser.add_argument(
        "--expect-version",
        default=None,
        help="require this exact declared __version__",
    )
    parser.add_argument(
        "--connector-module",
        default="integration/vllm/lmcache_mp_connector.py",
        help="package-relative module that must define the recipe's connector",
    )
    parser.add_argument(
        "--connector-class",
        default="LMCacheMPConnector",
        help="connector class name pinned by the serving recipe",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify(
            Path(args.package_dir),
            expect_version=args.expect_version,
            connector_module=args.connector_module,
            connector_class=args.connector_class,
        )
    except ConfigError as exc:
        print(f"lmcache-group-tracking: CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK if report["verdict"] == "pass" else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
