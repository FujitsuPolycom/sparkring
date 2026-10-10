"""Record or check glm_dcp_decode_comm's pins against vLLM, b12x and SIRCL trees.

``python tools/refresh_pins.py [--vllm DIR] [--b12x DIR] [--sircl DIR] [--sircl-only | --package-only] [--check]``

The pins live in ``glm_dcp_decode_comm/__init__.py``: ``FILE_CHECKS`` (one
SHA-256 per vLLM, b12x and SIRCL file the plugin relies on), the attention
module's ``ATTENTION_SHA256``, ``SIRCL_VERSION`` (the ``__version__`` of the
SIRCL tree) and ``PACKAGE_SHA256`` (the package's other modules). Digests read
CRLF line endings as LF (``glm_dcp_decode_comm.digest``).

Without ``--check`` the script rewrites those values for the given trees (each
package directory, for example ``.../site-packages/vllm``; a tree not given is
found on ``sys.path``) and prints every value that changed. ``--sircl-only``
rewrites only the ``sparkring_sircl`` pins and ``SIRCL_VERSION`` (and needs no
vLLM or b12x tree); ``--package-only`` rewrites only ``PACKAGE_SHA256``. With
``--check`` it changes nothing, prints
every pin that differs and exits 1 when any does. A re-pinned file is a new
build of the plugin's dependencies: its reason in ``FILE_CHECKS`` must still
hold, and the GPU checks and the audit-mode qualification run again.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT / "glm_dcp_decode_comm"
SOURCE = PACKAGE_DIR / "__init__.py"


def _plugin():
    sys.path.insert(0, str(PROJECT))
    import glm_dcp_decode_comm

    return glm_dcp_decode_comm


def _root(name: str, given: str | None) -> Path:
    if given:
        return Path(given).resolve()
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"{name} is not on the path; pass its directory")
    return Path(list(spec.submodule_search_locations)[0]).resolve()


def refresh(text: str, plugin, roots: dict[str, Path] | None) -> tuple[str, list[str]]:
    """``text`` with every pin recomputed (``roots`` None: the package's own modules only) and the changes."""
    changes: list[str] = []

    def sub(pattern: str, value: str, label: str) -> None:
        nonlocal text
        match = re.search(pattern, text)
        if match is None:
            raise SystemExit(f"{label}: not found in {SOURCE}")
        if match.group(2) != value:
            changes.append(f"{label}: {match.group(2)} -> {value}")
            text = text[:match.start(2)] + value + text[match.end(2):]

    if roots is not None:
        for check in plugin.FILE_CHECKS:
            if check.package not in roots:
                continue
            path = roots[check.package] / check.path
            value = plugin.digest(path) if path.exists() else "missing"
            sub(rf'(FileCheck\("{re.escape(check.package)}", "{re.escape(check.path)}",\s*")([0-9a-f]{{64}}|missing)',
                value, f"{check.package}/{check.path}")
        if "vllm" in roots:
            attention = roots["vllm"] / plugin.ATTENTION_PATH
            sub(r'(ATTENTION_SHA256 = ")([0-9a-f]{64}|missing)', plugin.digest(attention), plugin.ATTENTION_PATH)
        if plugin.SIRCL_PACKAGE in roots:
            version = plugin.sircl_version(roots[plugin.SIRCL_PACKAGE]) or "unknown"
            sub(r'(SIRCL_VERSION = ")([^"]*)', version, "SIRCL_VERSION")
    for name in plugin.PACKAGE_SHA256:
        sub(rf'("{re.escape(name)}": ")([0-9a-f]{{64}})', plugin.digest(PACKAGE_DIR / name), f"package {name}")
    return text, changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vllm")
    parser.add_argument("--b12x")
    parser.add_argument("--sircl")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--sircl-only", action="store_true")
    scope.add_argument("--package-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    plugin = _plugin()
    if args.package_only:
        roots = None
    elif args.sircl_only:
        roots = {plugin.SIRCL_PACKAGE: _root(plugin.SIRCL_PACKAGE, args.sircl)}
    else:
        roots = {"vllm": _root("vllm", args.vllm), "b12x": _root("b12x", args.b12x),
                 plugin.SIRCL_PACKAGE: _root(plugin.SIRCL_PACKAGE, args.sircl)}
    text = SOURCE.read_text(encoding="utf-8")
    updated, changes = refresh(text, plugin, roots)
    for line in changes:
        print(line)
    if args.check:
        print(f"{len(changes)} pin(s) differ" if changes else "every pin matches")
        return 1 if changes else 0
    if updated != text:
        SOURCE.write_text(updated, encoding="utf-8", newline="\n")
    print(f"{len(changes)} pin(s) rewritten in {SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
