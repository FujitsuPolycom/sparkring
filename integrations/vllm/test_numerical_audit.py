"""CPU-only tests for deterministic TP4 numerical-audit inputs."""

from __future__ import annotations

import unittest

import torch

from tp4_numerical_audit import ELEMENTS, make_rank_input


class NumericalAuditInputTest(unittest.TestCase):
    def test_inputs_are_deterministic_bfloat16_vectors(self) -> None:
        first = make_rank_input(7, 2)
        second = make_rank_input(7, 2)

        self.assertEqual(first.shape, (ELEMENTS,))
        self.assertEqual(first.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(torch.isfinite(first).all()))

    def test_sequence_or_rank_changes_the_input(self) -> None:
        baseline = make_rank_input(0, 0)

        self.assertFalse(torch.equal(baseline, make_rank_input(1, 0)))
        self.assertFalse(torch.equal(baseline, make_rank_input(0, 1)))

    def test_cancellation_case_has_a_finite_fp32_ground_truth(self) -> None:
        inputs = [make_rank_input(1, rank) for rank in range(4)]
        truth = torch.stack([tensor.float() for tensor in inputs]).sum(dim=0)

        self.assertTrue(bool(torch.isfinite(truth).all()))
        self.assertGreater(float(truth.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()


def test_invalid_iterations_fail_before_transport_setup(monkeypatch):
    import pytest
    import tp4_numerical_audit as audit
    def unexpected(*args, **kwargs):
        raise AssertionError("transport setup reached")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(audit.torch.cuda, "set_device", unexpected)
    monkeypatch.setattr(audit, "_NativeSession", unexpected)
    for count in ("0", "-1"):
        monkeypatch.setenv("ITERATIONS", count)
        with pytest.raises(ValueError, match="ITERATIONS must be positive"):
            audit.main()


def test_native_smoke_rejects_empty_run_before_native_construction(monkeypatch):
    import pytest
    import runpy
    import spark_tp4_backend
    from pathlib import Path
    def unexpected(*args, **kwargs):
        raise AssertionError("native setup reached")
    monkeypatch.setattr(spark_tp4_backend, "_NativeSession", unexpected)
    monkeypatch.setenv("RANK", "0")
    for count in ("0", "-1"):
        monkeypatch.setenv("ITERATIONS", count)
        with pytest.raises(ValueError, match="ITERATIONS must be positive"):
            runpy.run_path(str(Path(__file__).with_name("test_native_tp4.py")), run_name="__main__")


def test_probe_entrypoints_supply_native_payload_bytes(monkeypatch):
    import pytest
    import runpy
    from pathlib import Path
    import tp4_numerical_audit as audit
    import spark_tp4_backend as backend
    calls = []
    class SetupObserved(Exception):
        pass
    def session(rank, payload_bytes):
        calls.append((rank, payload_bytes))
        raise SetupObserved()
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("ITERATIONS", "1")
    monkeypatch.setattr(audit.torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(audit.dist, "init_process_group", lambda **_: None)
    monkeypatch.setattr(audit, "_NativeSession", session)
    monkeypatch.setattr(backend, "_NativeSession", session)
    with pytest.raises(SetupObserved):
        audit.main()
    with pytest.raises(SetupObserved):
        runpy.run_path(str(Path(__file__).with_name("test_native_tp4.py")), run_name="__main__")
    assert calls == [(0, 12288), (0, 12288)]
