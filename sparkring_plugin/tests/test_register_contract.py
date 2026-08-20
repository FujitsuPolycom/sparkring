"""Fail-closed exit status of the ``vllm.general_plugins`` entry point.

``register()`` ends the process with ``os._exit(78)`` when a
``VLLM_SPARK_*`` mode is set and installation fails, so the contract
cannot be observed in-process: ``os._exit`` would end the test runner
with it. Each case therefore runs a child interpreter that imports
``sparkring.plugin``, calls ``register()``, and reports its exit status.

Precondition for the refusal cases: vLLM must not be importable. They
pin the path where the compatibility gate finds no
``CudaCommunicator.all_reduce`` and refuses, which is what makes an
enabled mode fail on a host without the fork. With a real vLLM present
the gate can pass and installation proceeds instead, so those cases
skip.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = PLUGIN_ROOT / "src"

# The exit status register() uses for a fatal install failure with a
# mode enabled. A half-installed transport must never carry collectives,
# so the process ends before vLLM can serve traffic.
FATAL_EXIT = 78

_CHILD = textwrap.dedent(
    """
    from sparkring.plugin import register

    register()
    print("REGISTER-RETURNED")
    """
)


def _vllm_importable() -> bool:
    try:
        return importlib.util.find_spec("vllm") is not None
    except BaseException:
        # A vLLM that is present but not resolvable is also not the
        # absent-vLLM precondition the refusal cases need.
        return True


_absent_vllm_only = unittest.skipIf(
    _vllm_importable(),
    "vLLM is importable; the compatibility gate can pass instead of "
    "refusing",
)


def _register(**mode_env: str) -> subprocess.CompletedProcess:
    """Call register() in a child holding only the named mode variables.

    The child inherits neither the parent's ``VLLM_SPARK*`` /
    ``SPARK_TP4*`` settings nor its ``PYTHON*`` settings, and runs in an
    empty directory, so its only import root is the checkout's ``src``.
    """

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VLLM_SPARK", "SPARK_TP4", "PYTHON"))
    }
    environment["PYTHONPATH"] = str(SRC)
    environment.update(mode_env)
    with tempfile.TemporaryDirectory() as empty:
        return subprocess.run(
            [sys.executable, "-c", _CHILD],
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
            cwd=empty,
        )


class RegisterExitContractTest(unittest.TestCase):
    def _assert_fails_closed(
        self, result: subprocess.CompletedProcess
    ) -> None:
        self.assertEqual(
            result.returncode,
            FATAL_EXIT,
            f"expected exit {FATAL_EXIT}, got {result.returncode}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}",
        )
        self.assertNotIn("REGISTER-RETURNED", result.stdout)
        self.assertIn(
            "FATAL: SparkRing plugin installation failed", result.stderr
        )
        self.assertIn(
            "terminating before vLLM can serve traffic", result.stderr
        )
        self.assertIn(
            "SparkRing cannot locate the vLLM integration point",
            result.stderr,
        )
        self.assertIn("CudaCommunicator.all_reduce", result.stderr)

    def _assert_returns(
        self, result: subprocess.CompletedProcess
    ) -> None:
        self.assertEqual(
            result.returncode,
            0,
            f"expected exit 0, got {result.returncode}\n"
            f"stderr:\n{result.stderr[-2000:]}",
        )
        self.assertIn("REGISTER-RETURNED", result.stdout)

    @_absent_vllm_only
    def test_shadow_mode_exits_fatal(self) -> None:
        self._assert_fails_closed(_register(VLLM_SPARK_TP4_MODE="shadow"))

    @_absent_vllm_only
    def test_custom_mode_exits_fatal(self) -> None:
        self._assert_fails_closed(_register(VLLM_SPARK_TP4_MODE="custom"))

    @_absent_vllm_only
    def test_family_mode_alone_exits_fatal(self) -> None:
        """A family mode enables the plugin without the core mode, so it
        must reach the same refusal rather than install partially."""
        self._assert_fails_closed(
            _register(VLLM_SPARK_TP4_DCP_MODE="custom")
        )

    def test_disabled_core_mode_returns(self) -> None:
        self._assert_returns(_register(VLLM_SPARK_TP4_MODE="disabled"))

    def test_no_mode_variable_returns(self) -> None:
        self._assert_returns(_register())


if __name__ == "__main__":
    unittest.main()
