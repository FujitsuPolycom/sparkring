"""GPU-free behavioral tests for the plugin entry point."""

from __future__ import annotations

import os
import pathlib
import sys
import unittest
from unittest.mock import patch

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT / "src"))

from sparkring import plugin  # noqa: E402

_SPARK_ENV = [
    key
    for key in (
        "VLLM_SPARK_TP4_MODE",
        "VLLM_SPARK_TP4_ALLGATHER_MODE",
        "VLLM_SPARK_TP4_VOCAB_MODE",
        "VLLM_SPARK_TP4_DCP_MODE",
        "VLLM_SPARK_TP4_EAGER_WIDTHS",
    )
]


def _clean_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key not in _SPARK_ENV
    }


class PluginRegisterTest(unittest.TestCase):
    def test_register_is_a_noop_without_mode_env(self) -> None:
        with patch.dict(os.environ, _clean_env(), clear=True):
            self.assertIsNone(plugin.register())

    def test_enabled_reflects_core_and_family_modes(self) -> None:
        with patch.dict(os.environ, _clean_env(), clear=True):
            self.assertFalse(plugin._enabled())
            os.environ["VLLM_SPARK_TP4_MODE"] = "disabled"
            self.assertFalse(plugin._enabled())
            os.environ["VLLM_SPARK_TP4_MODE"] = "shadow"
            self.assertTrue(plugin._enabled())
            del os.environ["VLLM_SPARK_TP4_MODE"]
            os.environ["VLLM_SPARK_TP4_DCP_MODE"] = "custom"
            self.assertTrue(plugin._enabled())

    def test_register_impl_without_mode_imports_but_installs_nothing(
        self,
    ) -> None:
        """With every mode unset, _register_impl must complete without vLLM
        or torch present: backend.install() returns before touching vLLM."""
        with patch.dict(os.environ, _clean_env(), clear=True):
            plugin._register_impl()
        self.assertIn(plugin._VENDOR, sys.path)
        import spark_tp4_backend  # noqa: F401  (vendored, importable)

    def test_register_impl_with_mode_fails_without_vllm(self) -> None:
        """With shadow mode enabled on a host without vLLM, installation
        must raise (which register() converts to a fatal exit)."""
        environment = _clean_env()
        environment["VLLM_SPARK_TP4_MODE"] = "shadow"
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaises(BaseException):
                plugin._register_impl()

    def test_native_library_env_defaults_to_packaged_path(self) -> None:
        with patch.dict(os.environ, _clean_env(), clear=True):
            with patch.object(
                plugin.os.path, "exists", return_value=True
            ):
                try:
                    plugin._register_impl()
                except BaseException:
                    pass
                self.assertEqual(
                    os.environ.get("SPARK_TP4_LIBRARY"),
                    plugin._NATIVE_LIBRARY,
                )


if __name__ == "__main__":
    unittest.main()
