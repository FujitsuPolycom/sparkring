"""The vendor manifest must be a closed, dependency-free import set.

``scripts/sync_vendor.py`` declares which flat modules the wheel ships,
and the adapter imports its siblings by bare name. Two properties make
that manifest checkable instead of asserted: every source-tree module a
vendored module names is itself vendored, and every vendored module
imports with nothing but the standard library on the path.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import textwrap
import unittest

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR = PLUGIN_ROOT / "src" / "sparkring" / "_vendor"
REPO_ROOT = PLUGIN_ROOT.parent
SOURCE = REPO_ROOT / "spark_transport" / "integrations" / "vllm"
EXPERIMENTS = REPO_ROOT / "spark_transport" / "experiments"

sys.path.insert(0, str(PLUGIN_ROOT / "scripts"))
from sync_vendor import MODULES  # noqa: E402

# Repository packages a vendored module names on purpose without
# vendoring them. Each import is function-local and gated on its own
# environment variable, so the wheel is complete without the package and
# the adapter runs unchanged when it is absent.
ALLOWED_EXPERIMENT_PACKAGES = {
    # spark_graph_status_reporter reports the controller's installation
    # snapshot only under SPARK_ADAPTIVE_MTP_CONTROL=1.
    "adaptive_mtp_controller",
    # spark_q2r_probe_bridge reads route-capture provenance and live
    # capture control only inside its SPARK_Q2R_PROBE=1 install path.
    "moe_round_floor",
    # spark_q2r_probe_bridge installs phase timing from the same
    # SPARK_Q2R_PROBE=1 path.
    "q2r_phase_timing",
}

# The one seam that imports a name no static analysis can see:
# spark_tp4_query_row_provider resolves the module named in
# VLLM_SPARK_TP4_QUERY_ROW_PROVIDER at row-resolution time. No provider
# is vendored — the wheel is complete without one, and
# sparkring.preflight reports a configured provider's importability as
# its own check. Any other module importing a computed name would be an
# unaudited hole in the manifest.
DYNAMIC_IMPORT_MODULES = {"spark_tp4_query_row_provider"}

# Heavy dependencies the wheel must not need at import time. The probe
# blocks them outright so the result does not depend on what the host
# happens to have installed.
DEFERRED_DEPENDENCIES = ("torch", "vllm")

_IMPORT_PROBE = textwrap.dedent(
    """
    import importlib
    import importlib.abc
    import sys

    vendor, names, blocked = sys.argv[1], sys.argv[2], sys.argv[3]


    class _Absent(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in blocked.split(","):
                raise ModuleNotFoundError(
                    f"{fullname} is absent in this probe"
                )
            return None


    sys.meta_path.insert(0, _Absent())
    # Keep the interpreter's own standard-library entries and nothing
    # else: no repository source root, no site-packages. A bare-name
    # import must therefore resolve inside the vendored directory.
    roots = (sys.base_prefix, sys.prefix)
    stdlib = [
        entry for entry in sys.path
        if entry and entry.startswith(roots)
        and "site-packages" not in entry
    ]
    sys.path[:] = [vendor] + stdlib
    for name in blocked.split(","):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{name} must be absent in this probe")
    for name in names.split(","):
        importlib.import_module(name)
    print("VENDOR-IMPORT-OK")
    """
)


def _import_call_target(node: ast.Call) -> tuple[bool, str | None]:
    """Return (is an import call, constant module name or None)."""

    function = node.func
    if isinstance(function, ast.Name):
        called = function.id
    elif isinstance(function, ast.Attribute):
        called = function.attr
    else:
        return False, None
    if called not in ("import_module", "__import__"):
        return False, None
    first = node.args[0] if node.args else None
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return True, first.value
    return True, None


def _static_import_targets(tree: ast.Module) -> set[str]:
    """Every module name the source names as a constant, at any depth.

    ``ast.walk`` reaches function-local imports as well as module-level
    ones, which is what the closure claim needs: the adapter defers most
    of its sibling imports into the call that uses them.
    """

    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Relative imports cannot name a flat source-tree module.
            if node.level == 0 and node.module:
                targets.add(node.module)
        elif isinstance(node, ast.Call):
            is_import, name = _import_call_target(node)
            if is_import and name is not None:
                targets.add(name)
    return targets


def _parse(name: str) -> ast.Module:
    path = VENDOR / f"{name}.py"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@unittest.skipUnless(
    SOURCE.is_dir(), "source tree not present (wheel-installed run)"
)
class VendorClosureTest(unittest.TestCase):
    def test_manifest_covers_every_source_tree_import(self) -> None:
        for name in MODULES:
            for target in sorted(_static_import_targets(_parse(name))):
                root = target.split(".")[0]
                if (SOURCE / f"{root}.py").is_file():
                    with self.subTest(module=name, imports=target):
                        self.assertIn(
                            root,
                            MODULES,
                            f"{name}.py imports {target}, which lives "
                            "in spark_transport/integrations/vllm/ but "
                            "is not vendored; add it to "
                            "scripts/sync_vendor.py MODULES and re-run "
                            "the script",
                        )
                elif (EXPERIMENTS / root).is_dir():
                    with self.subTest(module=name, imports=target):
                        self.assertIn(
                            root,
                            ALLOWED_EXPERIMENT_PACKAGES,
                            f"{name}.py imports the experiment package "
                            f"{root}, which the wheel does not ship; "
                            "vendor it or record why it is an optional "
                            "seam in ALLOWED_EXPERIMENT_PACKAGES",
                        )

    def test_only_the_row_policy_seam_imports_a_computed_name(self) -> None:
        computed = set()
        for name in MODULES:
            for node in ast.walk(_parse(name)):
                if not isinstance(node, ast.Call):
                    continue
                is_import, constant = _import_call_target(node)
                if is_import and constant is None:
                    computed.add(name)
        self.assertEqual(
            computed,
            DYNAMIC_IMPORT_MODULES,
            "a vendored module imports a name computed at runtime that "
            "no static closure covers; document the seam in "
            "DYNAMIC_IMPORT_MODULES or replace it with a constant",
        )


class VendorImportIsolationTest(unittest.TestCase):
    def test_every_vendored_module_imports_without_heavy_deps(
        self,
    ) -> None:
        """Deferred-import discipline: the wheel installs and imports on
        a host with neither torch nor vLLM, so every heavy import must
        sit inside the function that uses it."""

        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("VLLM_SPARK", "SPARK_TP4", "PYTHON"))
        }
        # -I -S: no user site, no site-packages, no script directory, and
        # PYTHONPATH ignored, so the probe starts from the standard
        # library alone.
        result = subprocess.run(
            [
                sys.executable, "-I", "-S", "-c", _IMPORT_PROBE,
                str(VENDOR), ",".join(MODULES),
                ",".join(DEFERRED_DEPENDENCIES),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=environment,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"vendored import probe failed:\n{result.stderr[-4000:]}",
        )
        self.assertIn("VENDOR-IMPORT-OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
