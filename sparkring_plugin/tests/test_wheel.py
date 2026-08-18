"""Packaged-artifact contract: build the wheel, install it, use it.

The rest of the suite imports from the source checkout, which cannot
detect broken package data, missing vendored modules, or wrong
entry-point metadata in the artifact a user would install. This test
builds the wheel from the checkout, installs it into an isolated
target directory, and — in a subprocess that sees only that
directory — discovers the ``vllm.general_plugins`` entry point and
verifies that mode-unset ``register()`` is a no-op that imports
neither vLLM nor any adapter module.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent


class WheelContractTest(unittest.TestCase):
    def test_wheel_installs_and_registers_as_a_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wheel_dir = pathlib.Path(tmp) / "wheel"
            target = pathlib.Path(tmp) / "site"
            build = subprocess.run(
                [
                    sys.executable, "-m", "pip", "wheel", "--no-deps",
                    "--wheel-dir", str(wheel_dir), str(PLUGIN_ROOT),
                ],
                capture_output=True, text=True, timeout=600,
            )
            self.assertEqual(
                build.returncode, 0, f"wheel build failed:\n{build.stderr[-2000:]}"
            )
            wheels = list(wheel_dir.glob("sparkring-*.whl"))
            self.assertEqual(len(wheels), 1, f"expected one wheel: {wheels}")
            install = subprocess.run(
                [
                    sys.executable, "-m", "pip", "install", "--no-deps",
                    "--target", str(target), str(wheels[0]),
                ],
                capture_output=True, text=True, timeout=600,
            )
            self.assertEqual(
                install.returncode, 0,
                f"wheel install failed:\n{install.stderr[-2000:]}",
            )

            probe = textwrap.dedent(
                """
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

                scripts = importlib.metadata.entry_points(
                    group="console_scripts"
                )
                assert any(e.name == "sparkring-preflight" for e in scripts)
                print("WHEEL-CONTRACT-OK")
                """
            )
            probe_env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("VLLM_SPARK", "SPARK_TP4", "PYTHON"))
            }
            probe_env["PYTHONPATH"] = str(target)
            check = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True, text=True, timeout=120,
                env=probe_env,
                cwd=tmp,
            )
            self.assertEqual(
                check.returncode, 0,
                f"wheel probe failed:\n{check.stderr[-2000:]}",
            )
            self.assertIn("WHEEL-CONTRACT-OK", check.stdout)


if __name__ == "__main__":
    unittest.main()
