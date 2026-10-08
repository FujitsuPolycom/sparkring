"""What a serving container will import, checked inside the serving image by the stage step.

``python3 -m sparkring_sircl.vllm.serve.probe [--build]`` runs in a
short-lived container of the serving image with the staged tree on
``PYTHONPATH`` (the stage step starts it). With ``--build`` it first builds
the native libraries into ``SIRCL_BUILD_CACHE_DIR``: the collective library
(:mod:`sparkring_sircl.build`) and the point-to-point library
(:mod:`sparkring_sircl.p2p.build`), which the serving ranks load from that
cache by its source hash.
It then prints one line, ``SIRCL-SERVE-PROBE <json>``, describing:

- ``package``: the file ``import sparkring_sircl`` resolves to under the
  serving process's module path. SparkRing's serving entrypoint prepends
  ``/opt/sparkring/addons/python`` and ``/opt/sparkring/toolchain/python``
  (when present) to ``PYTHONPATH`` before it starts vLLM
  (``runtime/images/external_base.py`` and ``toolchain_runtime.py``), so the
  probe resolves the package in that order without importing it;
- ``entry_points``: every ``vllm.general_plugins`` and
  ``vllm.platform_plugins`` entry point visible under that path, with its
  distribution. vLLM keeps one plugin per name, so a second ``sircl`` entry
  point would replace SIRCL's;
- ``modules``: the file ``import vllm`` and ``import b12x`` resolve to under
  the same path (a source overlay first on ``PYTHONPATH`` replaces the
  image's);
- ``vllm``: whether the files of that vLLM match a pinned build
  (:mod:`sparkring_sircl.vllm.pins`); the platform and communicator need only
  the interfaces they use, so a mismatch is reported, not refused;
- ``shims``: the pinned build each version-pinned shim
  (:mod:`sparkring_sircl.vllm.shims`) would install for, or None where that
  vLLM's files match none;
- ``shim_status``: every catalogued shim's status in that vLLM
  (:func:`sparkring_sircl.vllm.catalog.tree_status`: verified,
  applicable-unverified or absent), which the stage step prints and
  ``serve shims --probe`` renders;
- ``library``: the native library path and whether it exists;
- ``p2p_library``: the point-to-point library path and whether it exists.

The probe imports neither torch nor vLLM. :func:`evaluate` turns the record
into blockers and notes on the operator's machine: a launch with a source
overlay needs ``vllm`` and ``b12x`` from the overlay, and a launch that needs
a shim (prefill row ownership on a group without NCCL) needs its pinned
build.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.metadata
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PREFIX = "SIRCL-SERVE-PROBE "
SERVE_PATH_PREFIXES = ("/opt/sparkring/addons/python", "/opt/sparkring/toolchain/python")
GROUPS = ("vllm.general_plugins", "vllm.platform_plugins")
OURS = {
    "vllm.general_plugins": "sparkring_sircl.vllm.plugin:register",
    "vllm.platform_plugins": "sparkring_sircl.vllm.platform:activate",
}


def serve_path(path: Sequence[str] = tuple(sys.path), prefixes: Sequence[str] = SERVE_PATH_PREFIXES) -> list[str]:
    """``path`` with the serving entrypoint's prepended directories that exist."""
    result = [item for item in path if item not in prefixes]
    for prefix in reversed(prefixes):
        if os.path.isdir(prefix):
            result.insert(0, prefix)
    return result


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def entry_points(path: Sequence[str]) -> dict[str, list[list[str]]]:
    """Entry points of :data:`GROUPS` as ``importlib.metadata.entry_points`` sees them under ``path``."""
    found: dict[str, list[list[str]]] = {group: [] for group in GROUPS}
    seen: set[str] = set()
    for distribution in importlib.metadata.distributions(path=list(path)):
        name = distribution.metadata["Name"] or "?"
        if _normalized(name) in seen:      # the first distribution of a name wins, as in entry_points()
            continue
        seen.add(_normalized(name))
        for point in distribution.entry_points:
            if point.group in found:
                found[point.group].append([point.name, point.value, name])
    return found


MODULES = ("vllm", "b12x")


def module_origins(path: Sequence[str], names: Sequence[str] = MODULES) -> dict[str, str | None]:
    """The file each top-level package resolves to under ``path`` (its ``__init__.py``), without importing it."""
    origins: dict[str, str | None] = {}
    for name in names:
        spec = importlib.machinery.PathFinder.find_spec(name, list(path))
        locations = list(spec.submodule_search_locations or ()) if spec is not None else []
        origins[name] = (spec.origin if spec is not None and spec.origin else
                         os.path.join(locations[0], "__init__.py") if locations else None)
    return origins


def collect(*, build: bool = False) -> dict[str, Any]:
    path = serve_path()
    record: dict[str, Any] = {"python": sys.version.split()[0], "serve_path": path[:6]}
    spec = importlib.machinery.PathFinder.find_spec("sparkring_sircl", path)
    record["package"] = spec.origin if spec is not None else None
    record["entry_points"] = entry_points(path)
    record["modules"] = module_origins(path)
    try:
        from sparkring_sircl.vllm import catalog, pins, shims

        origin = record["modules"].get("vllm")
        root = Path(origin).parent if origin else None
        report = pins.check(root)
        record["vllm"] = {"root": str(report.root) if report.root else None, "matches": list(report.matches),
                          "describe": report.describe()}
        record["shims"] = {}
        for name, shim in shims.SHIMS.items():
            matched = pins.matching_build(root, shim.files) if root is not None else None
            record["shims"][name] = matched.name if matched is not None else None
        record["shim_status"] = catalog.tree_status(root) if root is not None else None
    except Exception as error:  # noqa: BLE001 - reported to the operator
        record["vllm"] = {"error": f"{type(error).__name__}: {error}"}
    try:
        from sparkring_sircl import build as native

        path_ = native.build() if build else native.library_path()
        record["library"] = {"path": str(path_), "exists": path_.exists()}
    except Exception as error:  # noqa: BLE001 - reported to the operator
        record["library"] = {"error": f"{type(error).__name__}: {error}"}
    try:
        from sparkring_sircl.p2p import build as p2p_native

        path_ = p2p_native.build() if build else p2p_native.library_path()
        record["p2p_library"] = {"path": str(path_), "exists": path_.exists()}
    except Exception as error:  # noqa: BLE001 - reported to the operator
        record["p2p_library"] = {"error": f"{type(error).__name__}: {error}"}
    return record


def evaluate(record: Mapping[str, Any], *, staged_root: str, library: str, overlay: str | None = None,
             required_shims: Sequence[str] = (), mhc_sizes: tuple[int, int] | None = None
             ) -> tuple[list[str], list[str]]:
    """(blockers, notes) for one Spark's probe record.

    ``overlay`` is the container path of a source overlay that ``vllm`` and
    ``b12x`` must resolve to; ``required_shims`` are the pinned shims the
    launch cannot serve without. ``mhc_sizes`` (tensor, decode-context
    parallelism) is given for a launch with mHC prefill row ownership on and
    decode-context parallelism above 1: the pinned build the vLLM's mHC files
    match must start mHC prefill row ownership at those sizes
    (``pins.mhc_admits``).
    """
    blockers, notes = [], []
    package = record.get("package") or ""
    if not package.startswith(staged_root.rstrip("/") + "/"):
        blockers.append(f"import sparkring_sircl resolves to {package or 'nothing'}, not the staged tree "
                        f"{staged_root}; another copy earlier on the serving path would shadow SIRCL")
    points = record.get("entry_points") or {}
    for group, value in OURS.items():
        named = [entry for entry in points.get(group, []) if entry[0] == "sircl"]
        if [entry[1] for entry in named] != [value]:
            blockers.append(f"{group} entry points named sircl are {named or 'missing'}; vLLM must find "
                            f"exactly {value}")
    modules = record.get("modules") or {}
    if overlay is not None:
        for name in MODULES:
            origin = modules.get(name) or ""
            if not origin.startswith(overlay.rstrip("/") + "/"):
                blockers.append(f"import {name} resolves to {origin or 'nothing'}, not the source overlay {overlay}")
    vllm = record.get("vllm") or {}
    if vllm.get("error") or not vllm.get("root"):
        blockers.append(f"the image's vLLM could not be located: {vllm.get('error', 'not installed')}")
    elif vllm.get("matches"):
        notes.append(f"vLLM at {vllm['root']} matches pinned build {', '.join(vllm['matches'])}")
    else:
        shims = record.get("shims") or {}
        matched = ", ".join(f"{name} ({build})" for name, build in sorted(shims.items()) if build) or "none"
        notes.append(f"vLLM at {vllm['root']} matches no pinned build; the platform plugin and communicator check "
                     f"their hooks by capability, and a pinned shim installs only where its own files match (shims "
                     f"that match: {matched}):\n" + str(vllm.get("describe", "")))
    shims = record.get("shims")
    for name in required_shims:
        if isinstance(shims, Mapping) and name in shims and shims[name] is None:
            blockers.append(f"this launch needs the {name} shim, and the files it wraps in the vLLM at "
                            f"{vllm.get('root')} match no pinned build: the ranks would refuse at setup"
                            + ("; --mhc-prefill-shard off serves without it" if name == "mhc_prefill_shard" else
                               "; --env VLLM_QWEN3_8_HC_PREFILL_MODE=off serves without it"
                               if name == "qwen_hc_prefill_shard" else ""))
    if mhc_sizes is not None and isinstance(shims, Mapping) and shims.get("mhc_prefill_shard"):
        from .. import pins

        build = shims["mhc_prefill_shard"]
        if tuple(mhc_sizes) not in pins.mhc_admits(build):
            tensor, dcp = mhc_sizes
            blockers.append(f"the vLLM at {vllm.get('root')} is pinned build {build}, whose GLM-5.3-Flash model "
                            "starts mHC prefill row ownership only at "
                            + ", ".join(f"TP{t}/DCP{d}" for t, d in pins.mhc_admits(build))
                            + f" and refuses TP{tensor} with DCP {dcp} at startup; --mhc-prefill-shard off "
                            "serves without it")
    native = record.get("library") or {}
    if native.get("error"):
        blockers.append(f"the native library could not be built: {native['error']}")
    elif not native.get("exists") or os.path.basename(str(native.get("path", ""))) != library:
        blockers.append(f"the native library is {native.get('path')} (exists: {native.get('exists')}); "
                        f"the plan expects {library}")
    p2p = record.get("p2p_library")
    if isinstance(p2p, Mapping):
        if p2p.get("error"):
            blockers.append(f"the point-to-point library could not be built: {p2p['error']}")
        elif not p2p.get("exists"):
            blockers.append(f"the point-to-point library {p2p.get('path')} does not exist; the serving ranks load "
                            "it from the build cache")
    return blockers, notes


def parse(text: str) -> dict[str, Any] | None:
    """The probe record in a container's output, or None."""
    for line in text.splitlines():
        if line.startswith(PREFIX):
            try:
                return json.loads(line[len(PREFIX):])
            except ValueError:
                return None
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sparkring_sircl.vllm.serve.probe")
    parser.add_argument("--build", action="store_true", help="build the native libraries first")
    args = parser.parse_args(argv)
    print(PREFIX + json.dumps(collect(build=args.build), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
