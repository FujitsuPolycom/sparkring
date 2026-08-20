"""Packaged-artifact contract: build the wheel, install it, use it.

The rest of the suite imports from the source checkout, which cannot
detect broken package data, missing vendored modules, or wrong
entry-point metadata in the artifact a user would install. This module
builds the wheel once, installs it into an isolated target directory,
and asserts three contracts in subprocesses that see only that
directory: mode-unset ``register()`` imports neither vLLM nor any
adapter module, the installed ``sparkring-preflight`` console script
runs and prints the documented check records, and mode-set
``register()`` exits 78 instead of returning to vLLM when installation
cannot complete.

The build passes ``--no-build-isolation`` because it must stay OFFLINE:
PEP 517 isolation installs the ``setuptools>=68`` build requirement
from a package index, which fails on a network-isolated host. That
requirement must therefore be satisfied by the running interpreter, and
the class is skipped when it is not.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Mirrors the [build-system] requires pin in pyproject.toml. Without
# build isolation the backend is imported from this interpreter, so the
# pin has to hold here instead of in a fetched build environment.
_MINIMUM_SETUPTOOLS = 68

# Field names of sparkring.preflight.Check, which --json emits per check.
_CHECK_FIELDS = ("name", "passed", "detail", "severity")


def _setuptools_shortfall() -> str | None:
    """Describe why an isolation-free build cannot run here, or None."""

    try:
        version = importlib.metadata.version("setuptools")
    except importlib.metadata.PackageNotFoundError:
        return "setuptools is not installed"
    major, _, _ = version.partition(".")
    if not major.isdigit() or int(major) < _MINIMUM_SETUPTOOLS:
        return f"setuptools {version} is installed"
    return None


def _run_pip(*arguments: str, what: str) -> None:
    """Run one pip command, failing with the tail of its stderr."""

    completed = subprocess.run(
        [sys.executable, "-m", "pip", *arguments],
        capture_output=True, text=True, timeout=600,
    )
    if completed.returncode != 0:
        raise AssertionError(f"{what} failed:\n{completed.stderr[-2000:]}")


class WheelContractTest(unittest.TestCase):
    tmp: pathlib.Path
    target: pathlib.Path

    @classmethod
    def setUpClass(cls) -> None:
        shortfall = _setuptools_shortfall()
        if shortfall is not None:
            raise unittest.SkipTest(
                "an offline wheel build needs setuptools>="
                f"{_MINIMUM_SETUPTOOLS} importable in {sys.executable}; "
                f"{shortfall}"
            )
        # One build and one install serve every test in this class; the
        # build dominates the runtime of the module.
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.tmp = pathlib.Path(temporary.name)
        cls.target = cls.tmp / "site"
        wheel_dir = cls.tmp / "wheel"
        # --no-index keeps the build from reaching a package index at
        # all: pip fails locally instead of opening a connection.
        _run_pip(
            "wheel", "--no-deps", "--no-build-isolation", "--no-index",
            "--wheel-dir", str(wheel_dir), str(PLUGIN_ROOT),
            what="wheel build",
        )
        wheels = list(wheel_dir.glob("sparkring-*.whl"))
        if len(wheels) != 1:
            raise AssertionError(f"expected one wheel: {wheels}")
        _run_pip(
            "install", "--no-deps", "--no-index",
            "--target", str(cls.target), str(wheels[0]),
            what="wheel install",
        )

    def _probe_env(self) -> dict[str, str]:
        """Environment in which only the install target is importable.

        An inherited PYTHONPATH would let the source checkout satisfy
        imports the artifact is supposed to satisfy, and an inherited
        VLLM_SPARK*/SPARK_TP4* variable would change what the installed
        code does.
        """

        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("VLLM_SPARK", "SPARK_TP4", "PYTHON"))
        }
        environment["PYTHONPATH"] = str(self.target)
        return environment

    def _run_probe(self, source: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            capture_output=True, text=True, timeout=120,
            env=self._probe_env(), cwd=self.tmp,
        )

    def test_installed_register_without_mode_is_a_noop(self) -> None:
        probe = """
            import importlib.metadata
            import sys

            entry_points = importlib.metadata.entry_points(
                group="vllm.general_plugins"
            )
            matches = [e for e in entry_points if e.name == "sparkring"]
            assert len(matches) == 1, f"entry points: {entry_points}"
            register = matches[0].load()
            assert callable(register)

            for name in ("VLLM_SPARK_TP4_MODE",):
                import os
                os.environ.pop(name, None)
            register()

            forbidden = [
                name for name in sys.modules
                if name == "vllm"
                or name.startswith("vllm.")
                or name.startswith("spark_tp4")
                or name.startswith("spark_collective")
            ]
            assert not forbidden, f"mode-unset register imported: {forbidden}"
            print("WHEEL-CONTRACT-OK")
            """
        check = self._run_probe(probe)
        self.assertEqual(
            check.returncode, 0,
            f"wheel probe failed:\n{check.stderr[-2000:]}",
        )
        self.assertIn("WHEEL-CONTRACT-OK", check.stdout)

    def test_installed_preflight_script_reports_json_checks(self) -> None:
        """The console script the wheel generates runs and reports checks.

        ``pip install --target`` writes console scripts to ``bin``, named
        with the platform's launcher suffix.
        """

        scripts = sorted((self.target / "bin").glob("sparkring-preflight*"))
        self.assertEqual(len(scripts), 1, f"installed scripts: {scripts}")
        run = subprocess.run(
            [str(scripts[0]), "--json"],
            capture_output=True, text=True, timeout=120,
            env=self._probe_env(), cwd=self.tmp,
        )
        try:
            checks = json.loads(run.stdout)
        except json.JSONDecodeError as error:
            self.fail(
                f"--json output is not JSON ({error}); "
                f"exit {run.returncode}, stdout {run.stdout[-2000:]!r}, "
                f"stderr {run.stderr[-2000:]!r}"
            )
        self.assertIsInstance(checks, list)
        self.assertTrue(checks, "preflight reported no checks")
        for check in checks:
            self.assertEqual(set(check), set(_CHECK_FIELDS), f"shape: {check}")
            self.assertIsInstance(check["name"], str)
            self.assertIsInstance(check["passed"], bool)
            self.assertIsInstance(check["detail"], str)
            self.assertIn(check["severity"], ("required", "info"))

        by_name = {check["name"]: check for check in checks}
        self.assertIn("env-mode", by_name)
        # A mode variable inherited from the caller would land here, so
        # this also proves the probe environment is clean.
        self.assertIn("mode unset", by_name["env-mode"]["detail"])
        # main() returns 1 when a required check failed and 0 otherwise;
        # which checks fail depends on the host, the relation does not.
        required_failures = sorted(
            check["name"] for check in checks
            if check["severity"] == "required" and not check["passed"]
        )
        self.assertEqual(
            run.returncode, 1 if required_failures else 0,
            f"exit {run.returncode} with required failures "
            f"{required_failures}",
        )

    def test_installed_register_with_mode_exits_78(self) -> None:
        """A mode-set register() that cannot install must not return.

        The refusal is only reachable where installation genuinely
        fails. A host without a resolvable vLLM integration point
        satisfies that condition, and the probe reports the integration
        point before it enables the mode.
        """

        probe = """
            import importlib.metadata
            import os

            from sparkring._compat import compat_report

            if compat_report()["ok"]:
                print("VLLM-INTEGRATION-PRESENT")
                raise SystemExit(0)

            os.environ["VLLM_SPARK_TP4_MODE"] = "shadow"
            entry_points = importlib.metadata.entry_points(
                group="vllm.general_plugins"
            )
            matches = [e for e in entry_points if e.name == "sparkring"]
            assert len(matches) == 1, f"entry points: {entry_points}"
            matches[0].load()()
            print("REGISTER-RETURNED")
            """
        check = self._run_probe(probe)
        if "VLLM-INTEGRATION-PRESENT" in check.stdout:
            self.skipTest(
                "this host resolves the vLLM integration point, so "
                "mode-set register() is not forced to fail here"
            )
        self.assertEqual(
            check.returncode, 78,
            f"stdout {check.stdout[-2000:]!r}, "
            f"stderr {check.stderr[-2000:]!r}",
        )
        self.assertIn("FATAL", check.stderr)


if __name__ == "__main__":
    unittest.main()
