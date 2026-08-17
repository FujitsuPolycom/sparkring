"""GPU-free tests for vLLM feature detection."""

from __future__ import annotations

import pathlib
import sys
import types
import unittest
from unittest.mock import patch

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT / "src"))

from sparkring import _compat  # noqa: E402


def _fake_vllm(version: str, communicator: type | None) -> dict:
    """Build a sys.modules patch dict emulating a vLLM install."""

    vllm = types.ModuleType("vllm")
    vllm.__version__ = version
    distributed = types.ModuleType("vllm.distributed")
    device = types.ModuleType("vllm.distributed.device_communicators")
    cuda = types.ModuleType(
        "vllm.distributed.device_communicators.cuda_communicator"
    )
    if communicator is not None:
        cuda.CudaCommunicator = communicator
    return {
        "vllm": vllm,
        "vllm.distributed": distributed,
        "vllm.distributed.device_communicators": device,
        "vllm.distributed.device_communicators.cuda_communicator": cuda,
    }


class _GoodCommunicator:
    def all_reduce(self, input_):
        return input_


class _NoAllReduce:
    pass


class _WrongArity:
    def all_reduce(self, input_, extra):
        return input_


class CompatTest(unittest.TestCase):
    def test_resolves_good_communicator(self) -> None:
        with patch.dict(
            sys.modules, _fake_vllm("0.27.1", _GoodCommunicator)
        ):
            report = _compat.compat_report()
        self.assertTrue(report["ok"])
        self.assertEqual(report["vllm_version"], "0.27.1")
        self.assertIn("CudaCommunicator", report["communicator_path"])
        self.assertEqual(report["detail"], "resolved")

    def test_unknown_version_still_resolves_with_note(self) -> None:
        with patch.dict(
            sys.modules, _fake_vllm("0.99.0", _GoodCommunicator)
        ):
            report = _compat.compat_report()
        self.assertTrue(report["ok"])
        self.assertIn("known-good", report["detail"])

    def test_missing_all_reduce_fails(self) -> None:
        with patch.dict(
            sys.modules, _fake_vllm("0.27.1", _NoAllReduce)
        ):
            report = _compat.compat_report()
        self.assertFalse(report["ok"])
        self.assertIn("no all_reduce", report["detail"])

    def test_wrong_arity_fails(self) -> None:
        with patch.dict(
            sys.modules, _fake_vllm("0.27.1", _WrongArity)
        ):
            report = _compat.compat_report()
        self.assertFalse(report["ok"])
        self.assertIn("expected (self, input_)", report["detail"])

    def test_absent_vllm_reports_none_version(self) -> None:
        hidden = {
            name: None
            for name in list(sys.modules)
            if name == "vllm" or name.startswith("vllm.")
        }
        with patch.dict(sys.modules, hidden, clear=False):
            report = _compat.compat_report()
        self.assertFalse(report["ok"])
        self.assertIsNone(report["vllm_version"])

    def test_require_compatible_raises_with_version(self) -> None:
        with patch.dict(
            sys.modules, _fake_vllm("0.31.0", _NoAllReduce)
        ):
            with self.assertRaisesRegex(RuntimeError, "0.31.0"):
                _compat.require_compatible()

    def test_require_compatible_passes_on_good(self) -> None:
        with patch.dict(
            sys.modules, _fake_vllm("0.27.1", _GoodCommunicator)
        ):
            self.assertIsNone(_compat.require_compatible())


if __name__ == "__main__":
    unittest.main()
