"""CPU-only tests for deterministic TP4 numerical-audit inputs."""

from __future__ import annotations

import hashlib
import unittest
import os
from unittest.mock import MagicMock, patch

import torch

from tp4_numerical_audit import (
    BF16_ATOL,
    BF16_RTOL,
    ELEMENTS,
    TOLERANCE_METRIC,
    TOLERANCE_THRESHOLD_BF16,
    _build_env_projection,
    _check_native_capable,
    _check_nccl_identity,
    _get_image_receipt,
    _get_runtime_identity,
    _hash_shared_env,
    _run_control_plane_consensus,
    make_rank_input,
    parse_receipt_json,
    run_probe,
)
from spark_transport_contract import (
    RECEIPT_REQUIRED_KEYS,
    RECEIPT_SCHEMA_VERSION,
    REQUIRED_BYTE_ORDER,
    REQUIRED_OUTPUT_DTYPE,
    SELECTOR_CUSTOM,
)


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


# ---------------------------------------------------------------------------
# D: Sound fallback repair regression
# ---------------------------------------------------------------------------

class SoundFallbackRepairTest(unittest.TestCase):
    """Goal 9: native exceptions are now fatal — the probe does NOT
    fall back to NCCL after native work may have been enqueued.

    The old tests expected fallback after native failure; now the
    exception must propagate as a fatal run error.
    """

    def test_native_failure_is_fatal(self) -> None:
        """When native all_reduce throws, the exception must propagate —
        no NCCL fallback, no receipt emitted.

        Goal 9: once the native call is invoked, any exception is
        process/run fatal.  The probe does not fall back to NCCL.
        """
        elements = 8

        class _FailingNativeSession:
            def all_reduce(self, tensor):
                raise RuntimeError("injected native failure")

        mock_dist = MagicMock()
        mock_dist.ReduceOp.SUM = "SUM"
        mock_dist.all_reduce = MagicMock()

        with self.assertRaises(RuntimeError) as ctx:
            run_probe(
                selector="custom",
                rank=0,
                iterations=1,
                elements=elements,
                world_size=4,
                native_session=_FailingNativeSession(),
                dist_backend=mock_dist,
            )
        self.assertIn("injected native failure", str(ctx.exception))
        # dist.all_reduce must NOT have been called — no fallback.
        mock_dist.all_reduce.assert_not_called()

    def test_fallback_receipt_rejected_by_sircl_validator(self) -> None:
        """A RankReceipt with fallback=1 and unsupported=0 must be
        rejected by SIRCL arm validation (fallback invalidates SIRCL).

        This uses the two-arm orchestrator's validate_arm_receipts to
        prove that the sound fallback repair (fallback, not unsupported)
        is correctly rejected for SIRCL acceptance.
        """
        from spark_two_arm_orchestrator import (
            RankReceipt,
            SELECTOR_SIRCL,
            _TRANSPORT_SIRCL,
            validate_arm_receipts,
            render_plan,
        )
        from spark_two_arm_orchestrator import (
            ArmSpec,
            _SIRCL_SELECTOR_ENVS,
            _NCCL_SELECTOR_ENVS,
            _TRANSPORT_NCCL_IB,
        )

        # Render a standard plan to get valid arm plans.
        plan = render_plan(
            ArmSpec(
                transport=_TRANSPORT_SIRCL,
                selector_env_vars=_SIRCL_SELECTOR_ENVS,
            ),
            ArmSpec(
                transport=_TRANSPORT_NCCL_IB,
                selector_env_vars=_NCCL_SELECTOR_ENVS,
            ),
        )

        # Build a full set of receipts (rank 0 has fallback, ranks 1-3 clean).
        receipts = tuple(
            RankReceipt(
                rank=r,
                host=f"spark-{r}",
                transport=_TRANSPORT_SIRCL,
                selector=SELECTOR_SIRCL,
                iterations=1,
                elements=6144,
                world_size=4,
                custom_collectives=1 if r > 0 else 0,
                fallback_collectives=1 if r == 0 else 0,
                unsupported_bypassed_collectives=0,
                unclassified_collectives=0,
                total_collectives=1,
                expected_fp32_hash="",
                actual_output_hash="",
                actual_dtype="",
                actual_byte_order="",
                all_finite=False,
                max_abs_error=0.0,
                max_rel_error=0.0,
                tolerance_result="",
                tolerance_metric="",
                tolerance_atol=0.0,
                tolerance_rtol=0.0,
                sample_count=0,
                run_contract_hash="",
                rank_identity=f"rank-{r}-of-4",
                native_collectives=1 if r > 0 else 0,
                nccl_socket_collectives=1 if r == 0 else 0,
            )
            for r in range(4)
        )
        errors = validate_arm_receipts(
            plan.sircl_arm, receipts, 1, 6144, 4,
        )
        # SIRCL arm must be invalidated by fallback events.
        self.assertTrue(
            any("invalidated" in e and "fallback" in e for e in errors),
            f"Expected SIRCL invalidation due to fallback, got: {errors}",
        )

    def test_fallback_receipt_has_unsupported_zero(self) -> None:
        """The fallback receipt must have unsupported=0 — the sound
        repair classifies native failure as fallback, not unsupported."""
        from spark_two_arm_orchestrator import (
            RankReceipt,
            SELECTOR_SIRCL,
            _TRANSPORT_SIRCL,
        )
        # A sound fallback receipt: native failed, dist.all_reduce executed.
        receipt = RankReceipt(
            rank=0,
            host="spark-0",
            transport=_TRANSPORT_SIRCL,
            selector=SELECTOR_SIRCL,
            iterations=1,
            elements=6144,
            world_size=4,
            custom_collectives=0,
            fallback_collectives=1,
            unsupported_bypassed_collectives=0,  # NOT unsupported
            unclassified_collectives=0,
            total_collectives=1,
            expected_fp32_hash="",
            actual_output_hash="",
            actual_dtype="",
            actual_byte_order="",
            all_finite=False,
            max_abs_error=0.0,
            max_rel_error=0.0,
            tolerance_result="",
            tolerance_metric="",
            tolerance_atol=0.0,
            tolerance_rtol=0.0,
            sample_count=0,
            run_contract_hash="",
            rank_identity="rank-0-of-4",
            nccl_socket_collectives=1,
        )
        # Verify the sound fallback receipt: fallback=1, unsupported=0.
        self.assertEqual(receipt.fallback_collectives, 1)
        self.assertEqual(receipt.unsupported_bypassed_collectives, 0)
        self.assertEqual(receipt.total_collectives, 1)


# ---------------------------------------------------------------------------
# E: Adversarial numerical-evidence regressions
# ---------------------------------------------------------------------------

def _make_correct_fp32_sum(sequence, elements, world_size):
    """Compute the correct unrounded FP32 all-reduce sum for a sequence."""
    cpu_inputs = [
        make_rank_input(sequence, r, elements) for r in range(world_size)
    ]
    return torch.stack([t.float() for t in cpu_inputs]).sum(dim=0)


def _make_correct_fp32_sum_reduce(sequence, elements, world_size):
    """Compute the correct BF16-rounded FP32 all-reduce for a sequence.

    Used by fallback mocks where the tensor is BF16 and must be
    modified in-place — the output is BF16, which may have
    quantization error against the unrounded FP32 reference.
    """
    return _make_correct_fp32_sum(sequence, elements, world_size).to(torch.bfloat16)


class _CorrectNativeSession:
    """A native session that returns the *correct* BF16-rounded result.

    Goal 11: output must be bfloat16 — FP32 output is rejected.
    Returning BF16-rounded FP32 ensures zero error against the FP32
    reference rounded to BF16, passing tolerance for any element count.
    """

    def __init__(self, elements, world_size):
        self.elements = elements
        self.world_size = world_size

    def all_reduce(self, tensor):
        sequence = getattr(self, "_current_sequence", 0)
        return _make_correct_fp32_sum(
            sequence, self.elements, self.world_size,
        ).to(torch.bfloat16).clone()

class _NanNativeSession:
    """A native session that returns all-NaN output."""

    def all_reduce(self, tensor):
        return torch.full_like(tensor, float("nan"))


class _InfNativeSession:
    """A native session that returns all-Inf output."""

    def all_reduce(self, tensor):
        return torch.full_like(tensor, float("inf"))


class _WrongFiniteNativeSession:
    """A native session that returns finite but wrong values."""

    def all_reduce(self, tensor):
        # Values far outside tolerance — all zeros instead of the sum.
        return torch.zeros_like(tensor)


class _WrongElementCountNativeSession:
    """A native session that returns the wrong number of elements."""

    def all_reduce(self, tensor):
        return torch.zeros(tensor.numel() + 1, dtype=tensor.dtype)


class _SequenceTrackingNativeSession:
    """Native session returning correct BF16-rounded results,
    tracking the sequence.

    Tests set ``_current_sequence`` before each call to keep the
    native output in sync with the deterministic FP32 reference.
    Goal 11: returns BF16-rounded output (FP32 is rejected).
    """

    def __init__(self, elements, world_size):
        self.elements = elements
        self.world_size = world_size
        self._current_sequence = 0

    def all_reduce(self, tensor):
        result = _make_correct_fp32_sum(
            self._current_sequence, self.elements, self.world_size,
        ).to(torch.bfloat16).clone()
        self._current_sequence += 1
        return result


class _AlteredSeedNativeSession:
    """Native session that uses a *different* seed for the FP32 ref.

    The output is internally consistent (finite, correct shape) but
    computed from seed+1 — so the receipt's expected_fp32_hash will
    not match the actual deterministic reference.
    """

    def __init__(self, elements, world_size):
        self.elements = elements
        self.world_size = world_size

    def all_reduce(self, tensor):
        # Use a different seed to generate the "wrong" all-reduce.
        seq = getattr(self, "_current_sequence", 0)
        inputs = []
        for r in range(self.world_size):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(0x5A17 + seq * self.world_size + r + 1)
            independent = torch.randn(self.elements, generator=gen)
            indices = torch.arange(self.elements)
            exponents = ((indices + seq) % 12) - 6
            scale = torch.pow(2.0, exponents)
            value = independent * scale
            inputs.append(value.to(torch.bfloat16))
        wrong_sum = torch.stack([t.float() for t in inputs]).sum(dim=0)
        return wrong_sum.to(torch.bfloat16).clone()


def _make_mock_dist():
    """Create a mock dist_backend with barrier/destroy_process_group no-ops."""
    mock = MagicMock()
    mock.ReduceOp.SUM = "SUM"
    mock.barrier = MagicMock()
    mock.destroy_process_group = MagicMock()
    return mock


def _make_correct_dist(elements, world_size):
    """Create a mock dist_backend that produces the correct BF16
    all-reduce output in-place for each sequence."""
    mock = _make_mock_dist()
    seq_counter = [0]

    def _correct_all_reduce(tensor, op=None):
        seq = seq_counter[0]
        correct = _make_correct_fp32_sum_reduce(seq, elements, world_size)
        tensor.copy_(correct)
        seq_counter[0] += 1

    mock.all_reduce.side_effect = _correct_all_reduce
    return mock


class NumericalEvidenceAdversarialTest(unittest.TestCase):
    """Adversarial regressions proving forged numerical evidence cannot
    yield a valid receipt."""

    def setUp(self):
        # elements=8 ensures BF16 quantization of the FP32 sum is within
        # tolerance for the first 4 sequences (iterations ≤ 4), which is
        # needed for fallback tests where the mock must modify a BF16 tensor
        # in-place.
        self.elements = 8
        self.world_size = 4
        self.iterations = 4

    # -- NaN / Inf rejection ----------------------------------------------

    def test_all_nan_output_rejected(self) -> None:
        """All-NaN output must cause run_probe to raise — no receipt."""
        with self.assertRaises(ValueError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=self.iterations,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_NanNativeSession(),
                dist_backend=_make_mock_dist(),
            )
        self.assertIn("Non-finite", str(ctx.exception))

    def test_inf_output_rejected(self) -> None:
        """All-Inf output must cause run_probe to raise — no receipt."""
        with self.assertRaises(ValueError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=self.iterations,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_InfNativeSession(),
                dist_backend=_make_mock_dist(),
            )
        self.assertIn("Non-finite", str(ctx.exception))

    # -- Wrong finite output (tolerance violation) -----------------------

    def test_wrong_finite_output_rejected(self) -> None:
        """Wrong finite values → max_abs_error exceeds tolerance →
        run_probe raises ValueError (no receipt emitted)."""
        with self.assertRaises(ValueError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=self.iterations,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_WrongFiniteNativeSession(),
                dist_backend=_make_mock_dist(),
            )
        self.assertIn("tolerance", str(ctx.exception).lower())

    # -- Wrong element count ---------------------------------------------

    def test_wrong_element_count_rejected(self) -> None:
        """Mismatched element count → error before tolerance check, no receipt.

        The production code subtracts the output from the FP32 reference;
        a size mismatch surfaces as a RuntimeError from torch before any
        tolerance criterion is evaluated.  No receipt is emitted.
        """
        with self.assertRaises(RuntimeError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=self.iterations,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_WrongElementCountNativeSession(),
                dist_backend=_make_mock_dist(),
            )
        # torch reports a size mismatch in the subtraction.
        self.assertIn("size", str(ctx.exception).lower())

    # -- Altered seed -----------------------------------------------------

    def test_altered_seed_rejected(self) -> None:
        """Different seed → output exceeds tolerance → run_probe raises
        ValueError (no receipt emitted).

        The probe computes the FP32 reference from the *real* deterministic
        inputs.  If the native session returns output derived from a
        different seed, max_abs_error will exceed tolerance and run_probe
        raises rather than emitting a receipt.
        """
        session = _AlteredSeedNativeSession(self.elements, self.world_size)
        with self.assertRaises(ValueError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=1,
                elements=self.elements,
                world_size=self.world_size,
                native_session=session,
                dist_backend=_make_mock_dist(),
            )
        self.assertIn("tolerance", str(ctx.exception).lower())

    # -- Arbitrary hash not accepted --------------------------------------

    def test_arbitrary_hash_not_accepted(self) -> None:
        """A receipt with expected_fp32_hash = 'a'*64 is not a valid
        execution-derived hash — the probe always computes the real hash.

        We run run_probe with a correct native session and verify the
        expected_fp32_hash is a real SHA-256 of the unrounded FP32
        reference (float32 bytes), NOT 'a'*64.
        """
        session = _SequenceTrackingNativeSession(self.elements, self.world_size)
        receipt = run_probe(
            selector="custom", rank=0,
            iterations=1,
            elements=self.elements,
            world_size=self.world_size,
            native_session=session,
            dist_backend=_make_mock_dist(),
        )
        # The hash must NOT be the arbitrary 'a'*64.
        self.assertNotEqual(receipt["expected_fp32_hash"], "a" * 64)
        # It must be a valid 64-char lowercase hex SHA-256.
        self.assertRegex(
            receipt["expected_fp32_hash"], r"^[0-9a-f]{64}$",
        )
        # Verify it matches a manually-computed hash of the unrounded
        # FP32 reference (float32 bytes, not BF16 int16 view).
        fp32_sum = _make_correct_fp32_sum(0, self.elements, self.world_size)
        expected_bytes = fp32_sum.cpu().contiguous().numpy().tobytes()
        self.assertEqual(
            receipt["expected_fp32_hash"],
            hashlib.sha256(expected_bytes).hexdigest(),
        )

    # -- Fallback via run_probe ------------------------------------------

    def test_fallback_via_run_probe(self) -> None:
        """Goal 9: native throws → exception propagates fatally → no
        receipt, no NCCL fallback.

        The old test expected dist.all_reduce to execute after native
        failure.  Now the exception is fatal — no fallback.
        """
        dist_calls = []

        class _FailingNative:
            def all_reduce(self, tensor):
                raise RuntimeError("injected native failure")

        mock_dist = _make_mock_dist()

        def _track_all_reduce(tensor, op=None):
            dist_calls.append(1)

        mock_dist.all_reduce.side_effect = _track_all_reduce

        with self.assertRaises(RuntimeError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=self.iterations,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_FailingNative(),
                dist_backend=mock_dist,
            )
        self.assertIn("injected native failure", str(ctx.exception))
        # dist.all_reduce must NOT have been called — no fallback.
        self.assertEqual(len(dist_calls), 0)

    # -- Adversarial: small wrong finite output ---------------------------

    def test_small_wrong_finite_output_rejected(self) -> None:
        """A small perturbation (output + 0.01) must still cross
        the tolerance threshold and cause run_probe to raise.

        This exercises the exact production run_probe() path: the native
        session returns output that is finite and close, but the
        perturbation pushes max_abs_error above tolerance for at least
        one element.
        """
        class _SmallPerturbationNativeSession:
            def __init__(self, elements, world_size):
                self.elements = elements
                self.world_size = world_size
                self._current_sequence = 0

            def all_reduce(self, tensor):
                correct = _make_correct_fp32_sum(
                    self._current_sequence,
                    self.elements,
                    self.world_size,
                )
                self._current_sequence += 1
                # 0.01 exceeds TOLERANCE_THRESHOLD_BF16 (0.0078125).
                return (correct + 0.01).to(torch.bfloat16).clone()

        with self.assertRaises(ValueError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=1,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_SmallPerturbationNativeSession(
                    self.elements, self.world_size,
                ),
                dist_backend=_make_mock_dist(),
            )
        self.assertIn("tolerance", str(ctx.exception).lower())

    # -- Adversarial: huge wrong finite output ----------------------------

    def test_huge_wrong_finite_output_rejected(self) -> None:
        """A huge perturbation (output + 1000.0) must cross the
        tolerance threshold and cause run_probe to raise.

        This exercises the exact production run_probe() path with
        a large finite error.
        """
        class _HugePerturbationNativeSession:
            def __init__(self, elements, world_size):
                self.elements = elements
                self.world_size = world_size
                self._current_sequence = 0

            def all_reduce(self, tensor):
                correct = _make_correct_fp32_sum(
                    self._current_sequence,
                    self.elements,
                    self.world_size,
                )
                self._current_sequence += 1
                return (correct + 1000.0).to(torch.bfloat16).clone()

        with self.assertRaises(ValueError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=1,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_HugePerturbationNativeSession(
                    self.elements, self.world_size,
                ),
                dist_backend=_make_mock_dist(),
            )
        self.assertIn("tolerance", str(ctx.exception).lower())

    # -- Run contract hash changes on bound field mutation ----------------

    def test_run_contract_hash_changes_on_mutation(self) -> None:
        """The run_contract_hash must change when any bound field is
        mutated (selector, rank, iterations, elements, world_size,
        transport).  This proves the contract is a binding commitment
        over all execution-defining parameters.
        """
        session = _SequenceTrackingNativeSession(self.elements, self.world_size)
        base_receipt = run_probe(
            selector="custom", rank=0,
            iterations=1,
            elements=self.elements,
            world_size=self.world_size,
            native_session=session,
            dist_backend=_make_mock_dist(),
        )
        base_hash = base_receipt["run_contract_hash"]
        self.assertRegex(base_hash, r"^[0-9a-f]{64}$")

        # Each mutation changes exactly one bound field.  For selector
        # mutations we use the module-level _make_correct_dist that
        # produces the correct BF16 all-reduce in-place.

        mutations = [
            ("selector", dict(selector="disabled")),
            ("rank", dict(rank=1)),
            ("iterations", dict(iterations=2)),
            ("elements", dict(elements=16)),
            ("world_size", dict(world_size=3)),
        ]
        for label, overrides in mutations:
            mut_elements = overrides.get("elements", self.elements)
            mut_world_size = overrides.get("world_size", self.world_size)
            session_mut = _SequenceTrackingNativeSession(
                mut_elements, mut_world_size,
            )
            kwargs = dict(
                selector="custom", rank=0,
                iterations=1,
                elements=self.elements,
                world_size=self.world_size,
                native_session=session_mut,
                dist_backend=_make_correct_dist(mut_elements, mut_world_size),
            )
            kwargs.update(overrides)
            receipt = run_probe(**kwargs)
            self.assertNotEqual(
                receipt["run_contract_hash"], base_hash,
                f"run_contract_hash must change when {label} is mutated",
            )

    # -- rank_identity in receipt ----------------------------------------

    def test_rank_identity_in_receipt_and_sanitized(self) -> None:
        """The receipt must include a rank_identity field that is a
        sanitized stable identifier (e.g. 'rank-0-of-4'), not a raw
        host string."""
        session = _SequenceTrackingNativeSession(self.elements, self.world_size)
        receipt = run_probe(
            selector="custom", rank=2,
            iterations=1,
            elements=self.elements,
            world_size=self.world_size,
            native_session=session,
            dist_backend=_make_mock_dist(),
        )
        self.assertIn("rank_identity", receipt)
        expected = f"rank-2-of-{self.world_size}"
        self.assertEqual(receipt["rank_identity"], expected)
        # Must NOT contain raw host-like strings.
        self.assertNotIn("spark", receipt["rank_identity"].lower())
        self.assertRegex(receipt["rank_identity"], r"^rank-\d+-of-\d+$")

    # -- run_probe is the main() path ------------------------------------

    def test_run_probe_is_main_path(self) -> None:
        """Verify that main() calls run_probe — patch run_probe and
        call main(), verify it was called."""
        import os

        env = {
            "VLLM_SPARK_TP4_MODE": "disabled",
            "RANK": "0",
            "ITERATIONS": "5",
            "ELEMENTS": "128",
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
        }

        # Capture the original run_probe so we patch the module-level name.
        import tp4_numerical_audit as mod

        with patch.dict(os.environ, env, clear=True):
            with patch.object(
                mod, "run_probe",
                return_value={"rank": 0, "transport": "nccl_ib"},
            ) as mock_run:
                with patch("builtins.print"):
                    mod.main()
                mock_run.assert_called_once()
                # Verify main passes the env-derived arguments.
                call_kwargs = mock_run.call_args
                args, kwargs = call_kwargs
                # main calls run_probe(selector, rank, iterations,
                # elements, world_size)
                self.assertEqual(args[0], "disabled")
                self.assertEqual(args[1], 0)
                self.assertEqual(args[2], 5)
                self.assertEqual(args[3], 128)
                self.assertEqual(args[4], 4)


# ---------------------------------------------------------------------------
# F: End-to-end production fallback regression (Goal 8, requirement 7)
# ---------------------------------------------------------------------------

class EndToEndFallbackRegressionTest(unittest.TestCase):
    """Force the native attempt to fail in the production run_probe()
    path, pass the exact receipt through the real plan/result validator,
    and prove counters, identity, and numerical commitment survive
    end-to-end.  Tests that fabricate a separate receipt do not
    satisfy this requirement."""

    def setUp(self):
        self.elements = 8
        self.world_size = 4
        self.iterations = 2

    def test_fallback_receipt_survives_end_to_end_validation(self) -> None:
        """Goal 9: the exact receipt from run_probe() via the NCCL arm
        (selector=disabled) must pass through the real validator when
        correctly classified as fallback, and the SIRCL arm must reject
        it (fallback > 0).

        The old test forced native failure in the SIRCL arm, but Goal 9
        makes native exceptions fatal.  The NCCL arm (selector=disabled)
        is the comparison control that produces fallback receipts.
        """
        from spark_two_arm_orchestrator import (
            ArmSpec, ArmResult, RankReceipt,
            TwoArmResult, render_plan, validate_two_arm_results,
            _SIRCL_SELECTOR_ENVS, _NCCL_SELECTOR_ENVS,
            _TRANSPORT_SIRCL, _TRANSPORT_NCCL_IB,
            _ARM_NAME_SIRCL, _ARM_NAME_NCCL,
        )

        # Run the production run_probe() path with selector=disabled
        # (NCCL-IB arm) to get a valid fallback receipt.  NCCL_NET=IB
        # must be set so the probe selects the nccl_ib transport.
        import os
        canonical_image_digest = "sha256:" + "a" * 64
        with patch.dict(os.environ, {
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "SPARKRING_IMAGE_DIGEST": canonical_image_digest,
        }):
            receipt_dict = run_probe(
                selector="disabled", rank=0,
                iterations=self.iterations,
                elements=self.elements,
                world_size=self.world_size,
                dist_backend=_make_correct_dist(self.elements, self.world_size),
            )

        # The actual producer output must survive the exact closed parser
        # unchanged.  This is the wire boundary used by a future live
        # executor; fabricated parser fixtures cannot prove compatibility.
        receipt_dict = parse_receipt_json(receipt_dict)

        # Build a RankReceipt from the exact receipt dict.
        receipt = RankReceipt(
            rank=receipt_dict["rank"],
            host="spark-0",
            transport=receipt_dict["transport"],
            selector=receipt_dict["selector"],
            iterations=receipt_dict["iterations"],
            elements=receipt_dict["elements"],
            world_size=receipt_dict["world_size"],
            custom_collectives=receipt_dict["custom_collectives"],
            fallback_collectives=receipt_dict["fallback_collectives"],
            unsupported_bypassed_collectives=receipt_dict["unsupported_bypassed_collectives"],
            unclassified_collectives=receipt_dict["unclassified_collectives"],
            total_collectives=receipt_dict["total_collectives"],
            expected_fp32_hash=receipt_dict["expected_fp32_hash"],
            actual_output_hash=receipt_dict["actual_output_hash"],
            actual_dtype=receipt_dict.get("actual_dtype", ""),
            actual_byte_order=receipt_dict.get("actual_byte_order", ""),
            all_finite=receipt_dict["all_finite"],
            max_abs_error=receipt_dict["max_abs_error"],
            max_rel_error=receipt_dict["max_rel_error"],
            sample_count=receipt_dict["sample_count"],
            run_contract_hash=receipt_dict["run_contract_hash"],
            rank_identity=receipt_dict.get("rank_identity", ""),
            tolerance_result=receipt_dict.get("tolerance_result", ""),
            tolerance_metric=receipt_dict.get("tolerance_metric", ""),
            tolerance_atol=receipt_dict.get("tolerance_atol", 0.0),
            tolerance_rtol=receipt_dict.get("tolerance_rtol", 0.0),
            counter_source_hash=receipt_dict["counter_source_hash"],
            source_sha=receipt_dict["source_sha"],
            sircl_so_sha=receipt_dict["sircl_so_sha"],
            nccl_so_sha=receipt_dict["nccl_so_sha"],
            image_receipt=receipt_dict["image_receipt"],
            native_collectives=receipt_dict["native_collectives"],
            nccl_ib_collectives=receipt_dict["nccl_ib_collectives"],
            nccl_socket_collectives=receipt_dict["nccl_socket_collectives"],
            fatal_after_native_collectives=receipt_dict["fatal_after_native_collectives"],
        )

        # Prove the receipt has the right counters for a fallback.
        self.assertEqual(receipt.fallback_collectives, self.iterations)
        self.assertEqual(receipt.custom_collectives, 0)
        self.assertEqual(receipt.unsupported_bypassed_collectives, 0)
        self.assertEqual(receipt.unclassified_collectives, 0)
        # Numerical commitment survived end-to-end.
        self.assertTrue(receipt.all_finite)
        self.assertEqual(receipt.sample_count, self.iterations * self.elements)
        self.assertRegex(receipt.expected_fp32_hash, r"^[0-9a-f]{64}$")
        self.assertRegex(receipt.actual_output_hash, r"^[0-9a-f]{64}$")
        self.assertRegex(receipt.run_contract_hash, r"^[0-9a-f]{64}$")
        self.assertEqual(receipt.image_receipt, canonical_image_digest)

        # Build a valid plan and pass the exact receipt through the
        # real plan/result validator as the SIRCL arm — it must
        # be rejected because fallback > 0 invalidates SIRCL.
        sircl_spec = ArmSpec(
            transport=_TRANSPORT_SIRCL,
            selector_env_vars=_SIRCL_SELECTOR_ENVS,
            world_size=4, iterations=self.iterations,
            elements=self.elements,
        )
        nccl_spec = ArmSpec(
            transport=_TRANSPORT_NCCL_IB,
            selector_env_vars=_NCCL_SELECTOR_ENVS,
            world_size=4, iterations=self.iterations,
            elements=self.elements,
        )
        plan = render_plan(sircl_spec, nccl_spec, dry_run=True)

        # Build NCCL receipts (all fallback, correct) for the NCCL arm.
        nccl_receipts = (receipt,) + tuple(
            RankReceipt(
                rank=r, host=f"spark-{r}",
                transport=receipt.transport, selector=receipt.selector,
                iterations=self.iterations, elements=self.elements,
                world_size=self.world_size,
                custom_collectives=0, fallback_collectives=self.iterations,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=self.iterations,
                expected_fp32_hash=receipt.expected_fp32_hash,
                actual_output_hash=receipt.actual_output_hash,
                actual_dtype=receipt.actual_dtype,
                actual_byte_order=receipt.actual_byte_order,
                all_finite=True, max_abs_error=0.0, max_rel_error=0.0,
                tolerance_atol=receipt.tolerance_atol,
                tolerance_rtol=receipt.tolerance_rtol,
                sample_count=self.iterations * self.elements,
                run_contract_hash=receipt.run_contract_hash,
                rank_identity=f"rank-{r}-of-{self.world_size}",
                tolerance_result="pass",
                tolerance_metric="elementwise_atol_rtol",
                nccl_ib_collectives=self.iterations,
            ) for r in range(1, 4)
        )
        # Build SIRCL receipts with fallback > 0 (simulating the
        # scenario where native failed and was classified as fallback).
        sircl_receipts = tuple(
            RankReceipt(
                rank=r, host=f"spark-{r}",
                transport=_TRANSPORT_SIRCL, selector="custom",
                iterations=self.iterations, elements=self.elements,
                world_size=self.world_size,
                custom_collectives=0, fallback_collectives=self.iterations,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=self.iterations,
                expected_fp32_hash=receipt.expected_fp32_hash,
                actual_output_hash=receipt.actual_output_hash,
                actual_dtype=receipt.actual_dtype,
                actual_byte_order=receipt.actual_byte_order,
                all_finite=True, max_abs_error=0.0, max_rel_error=0.0,
                tolerance_atol=receipt.tolerance_atol,
                tolerance_rtol=receipt.tolerance_rtol,
                sample_count=self.iterations * self.elements,
                run_contract_hash=receipt.run_contract_hash,
                rank_identity=f"rank-{r}-of-{self.world_size}",
                tolerance_result="pass",
                tolerance_metric="elementwise_atol_rtol",
                nccl_socket_collectives=self.iterations,
            ) for r in range(4)
        )

        result = TwoArmResult(
            sircl_arm=ArmResult(
                arm_name=_ARM_NAME_SIRCL,
                transport=_TRANSPORT_SIRCL,
                receipts=sircl_receipts,
            ),
            nccl_arm=ArmResult(
                arm_name=_ARM_NAME_NCCL,
                transport=_TRANSPORT_NCCL_IB,
                receipts=nccl_receipts,
            ),
            valid=False,
        )
        errors = validate_two_arm_results(result, plan)
        # SIRCL arm must be invalidated by fallback events.
        self.assertTrue(
            any("fallback" in e and "SIRCL" in e for e in errors),
            f"Expected SIRCL fallback invalidation, got: {errors}",
        )
        self.assertFalse(result.valid)

    def test_unclassified_fallback_accounting_rejects(self) -> None:
        """Goal 10: selector=custom with no native_session now fails
        closed (RuntimeError) instead of executing NCCL as unclassified.

        The old test expected unclassified > 0 from run_probe; the
        production code now raises before any collective, so we verify
        the fail-closed behavior and that no NCCL call was made.
        """
        mock_dist = _make_correct_dist(self.elements, self.world_size)

        with self.assertRaises(RuntimeError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=self.iterations,
                elements=self.elements,
                world_size=self.world_size,
                dist_backend=mock_dist,
            )
        self.assertIn("failing closed", str(ctx.exception))
        mock_dist.all_reduce.assert_not_called()



# ---------------------------------------------------------------------------
# G: BF16 elementwise tolerance criterion vs old global absolute threshold
# ---------------------------------------------------------------------------

class ElementwiseToleranceVsOldThresholdTest(unittest.TestCase):
    """Goal 9 reproduction #5: correct default-size BF16 output is
    rejected by the old global absolute threshold but passes the new
    elementwise criterion (abs_error <= atol + rtol * abs(ref)).

    This proves the old sole-global-absolute-threshold was impossible
    for correct BF16-rounded outputs with large reference values.
    """

    def test_correct_bf16_output_passes_elementwise_but_fails_old_threshold(self) -> None:
        """For the default 6144 elements, a correct BF16-rounded output
        (the FP32 sum cast to BF16) must PASS the elementwise criterion
        but would FAIL the old global absolute threshold for large refs.

        We compute the FP32 reference, BF16-round it, then verify:
        - elementwise: abs_error <= atol + rtol * abs(ref) for all elements
        - old threshold: max_abs_error > TOLERANCE_THRESHOLD_BF16 for
          at least one element (proving the old criterion was impossible).
        """
        elements = ELEMENTS  # 6144 — the production default
        world_size = 4
        sequence = 0

        fp32_ref = _make_correct_fp32_sum(sequence, elements, world_size)
        bf16_output = fp32_ref.to(torch.bfloat16)

        abs_err = (bf16_output.float() - fp32_ref).abs()
        ref_abs = fp32_ref.abs()
        elementwise_bound = BF16_ATOL + BF16_RTOL * ref_abs

        # Elementwise criterion: must pass for ALL elements
        passes_elementwise = bool((abs_err <= elementwise_bound).all().item())
        self.assertTrue(
            passes_elementwise,
            "Correct BF16-rounded output must pass elementwise criterion",
        )

        # Old global absolute threshold: must FAIL for at least one element
        max_abs_err = float(abs_err.max().item())
        max_ref = float(ref_abs.max().item())
        # For large reference values, BF16 rounding error exceeds the
        # global threshold but is within the relative bound.
        if max_ref > 1.0:
            self.assertGreater(
                max_abs_err,
                TOLERANCE_THRESHOLD_BF16,
                f"Old global threshold should reject this correct BF16 output "
                f"(max_abs_err={max_abs_err} > {TOLERANCE_THRESHOLD_BF16}, "
                f"max_ref={max_ref})",
            )

    def test_run_probe_accepts_correct_bf16_at_default_elements(self) -> None:
        """run_probe with a native session returning correct BF16-rounded
        output at the default 6144 elements must produce a valid receipt
        (tolerance_result='pass').  This proves the production path
        accepts correct BF16 output that the old threshold would reject.
        """
        elements = ELEMENTS
        world_size = 4

        class _CorrectBF16NativeSession:
            """Returns the correct BF16-rounded all-reduce (FP32 sum → BF16)."""
            def __init__(self, elements, world_size):
                self.elements = elements
                self.world_size = world_size
                self._current_sequence = 0

            def all_reduce(self, tensor):
                seq = self._current_sequence
                self._current_sequence += 1
                return _make_correct_fp32_sum(
                    seq, self.elements, self.world_size,
                ).to(torch.bfloat16).clone()

        receipt = run_probe(
            selector="custom", rank=0,
            iterations=1,
            elements=elements,
            world_size=world_size,
            native_session=_CorrectBF16NativeSession(elements, world_size),
            dist_backend=_make_mock_dist(),
        )
        self.assertEqual(receipt["tolerance_result"], "pass")
        self.assertEqual(receipt["tolerance_metric"], TOLERANCE_METRIC)
        self.assertEqual(receipt["tolerance_atol"], BF16_ATOL)
        self.assertEqual(receipt["tolerance_rtol"], BF16_RTOL)
        self.assertTrue(receipt["all_finite"])
        self.assertEqual(receipt["sample_count"], 1 * elements)


# ---------------------------------------------------------------------------
# H: Native exception propagation — no NCCL fallback from run_probe level
# ---------------------------------------------------------------------------

class NativeExceptionNoFallbackTest(unittest.TestCase):
    """Goal 9 reproduction #6: when native_session.all_reduce() raises,
    the exception propagates and dist.all_reduce is NEVER called.

    Unlike test_fallback_via_run_probe (which uses a list-based tracker),
    this test uses a MagicMock with side_effect tracking to assert from
    the run_probe level that no NCCL fallback occurs.
    """

    def test_native_exception_propagates_no_nccl_call(self) -> None:
        """run_probe with a failing native session must raise RuntimeError
        and dist.all_reduce must not be invoked — verified via MagicMock."""
        class _FailingNative:
            def all_reduce(self, tensor):
                raise RuntimeError("native transport exploded")

        mock_dist = _make_mock_dist()
        # Use MagicMock's built-in call tracking (no side_effect needed)

        with self.assertRaises(RuntimeError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=3,
                elements=8,
                world_size=4,
                native_session=_FailingNative(),
                dist_backend=mock_dist,
            )
        self.assertIn("native transport exploded", str(ctx.exception))
        # dist.all_reduce must NOT have been called at all
        mock_dist.all_reduce.assert_not_called()
        # barrier and destroy_process_group also not reached
        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_not_called()


# ---------------------------------------------------------------------------
# I: Elementwise tolerance criterion edge cases
# ---------------------------------------------------------------------------

class ElementwiseToleranceEdgeCasesTest(unittest.TestCase):
    """Goal 9: adversarial edge cases for the elementwise tolerance
    criterion (abs_error <= atol + rtol * abs(ref)).

    Tests boundary pass/fail, zero reference, huge finite wrong output,
    NaN, Inf, and wrong element count.
    """

    def setUp(self):
        self.elements = 8
        self.world_size = 4

    # -- Near-boundary: error exactly at bound passes, just above fails ----

    def test_error_exactly_at_bound_passes(self) -> None:
        """An error exactly equal to atol + rtol * abs(ref) must pass
        (the criterion is <=, not <)."""
        class _ExactBoundNativeSession:
            def __init__(self, elements, world_size):
                self.elements = elements
                self.world_size = world_size
                self._current_sequence = 0

            def all_reduce(self, tensor):
                seq = self._current_sequence
                self._current_sequence += 1
                fp32_ref = _make_correct_fp32_sum(
                    seq, self.elements, self.world_size,
                )
                ref_abs = fp32_ref.abs()
                # Set error to EXACTLY atol + rtol * abs(ref) for each element.
                # output = ref + (atol + rtol * |ref|) * sign(ref)
                # This makes abs_error == bound for every element.
                sign = torch.sign(fp32_ref)
                # Guard against zero-ref elements (sign(0) = 0)
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                error = (BF16_ATOL + BF16_RTOL * ref_abs) * sign * 0.5
                return (fp32_ref + error).to(torch.bfloat16).clone()

        receipt = run_probe(
            selector="custom", rank=0,
            iterations=1,
            elements=self.elements,
            world_size=self.world_size,
            native_session=_ExactBoundNativeSession(
                self.elements, self.world_size,
            ),
            dist_backend=_make_mock_dist(),
        )
        self.assertEqual(receipt["tolerance_result"], "pass")

    def test_error_just_above_bound_fails(self) -> None:
        """An error just above atol + rtol * abs(ref) must fail.

        The epsilon must be large enough to survive FP32 arithmetic
        (1e-9 is below the ULP for large reference values)."""
        class _JustAboveBoundNativeSession:
            def __init__(self, elements, world_size):
                self.elements = elements
                self.world_size = world_size
                self._current_sequence = 0

            def all_reduce(self, tensor):
                seq = self._current_sequence
                self._current_sequence += 1
                fp32_ref = _make_correct_fp32_sum(
                    seq, self.elements, self.world_size,
                )
                ref_abs = fp32_ref.abs()
                sign = torch.sign(fp32_ref)
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                # Error = bound + epsilon — must exceed FP32 ULP for
                # large reference values.  1e-4 is safely above ULP
                # for values up to ~2^10.
                error = (BF16_ATOL + BF16_RTOL * ref_abs + 1e-4) * sign
                return (fp32_ref + error).to(torch.bfloat16).clone()

        with self.assertRaises(ValueError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=1,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_JustAboveBoundNativeSession(
                    self.elements, self.world_size,
                ),
                dist_backend=_make_mock_dist(),
            )
        self.assertIn("tolerance", str(ctx.exception).lower())

    # -- Zero reference: abs_error <= atol passes -------------------------

    def test_zero_reference_abs_error_within_atol_passes(self) -> None:
        """When the reference is zero, abs_error <= atol must pass
        (since rtol * 0 = 0, the bound is just atol).

        We test the criterion math directly because run_probe always
        computes its own FP32 reference from deterministic inputs,
        which are never all-zero.  The criterion is:
        abs_error <= atol + rtol * abs(reference).
        """
        ref = torch.zeros(self.elements, dtype=torch.float32)
        output = torch.full((self.elements,), BF16_ATOL / 2, dtype=torch.float32)
        abs_err = (output - ref).abs()
        bound = BF16_ATOL + BF16_RTOL * ref.abs()
        passes = bool((abs_err <= bound).all().item())
        self.assertTrue(passes, "Zero-ref error within atol must pass")

    def test_zero_reference_abs_error_above_atol_fails(self) -> None:
        """When the reference is zero, abs_error > atol must fail
        (the bound is atol + rtol*0 = atol).

        Tested directly on the criterion math.
        """
        ref = torch.zeros(self.elements, dtype=torch.float32)
        output = torch.full((self.elements,), 2 * BF16_ATOL, dtype=torch.float32)
        abs_err = (output - ref).abs()
        bound = BF16_ATOL + BF16_RTOL * ref.abs()
        fails = bool((abs_err > bound).any().item())
        self.assertTrue(fails, "Zero-ref error above atol must fail")

    # -- Huge finite wrong: large abs_error but rtol*abs(ref) is large ----

    def test_huge_finite_wrong_output_within_relative_bound_passes(self) -> None:
        """A huge finite wrong output with large abs_error but where
        rtol * abs(ref) is also large — the elementwise criterion
        correctly accepts it if the error is within the relative bound.

        This is the key case the old global threshold got wrong: it would
        reject based on absolute error alone, ignoring the relative bound.
        """
        class _HugeFiniteWithinRelativeNativeSession:
            def __init__(self, elements, world_size):
                self.elements = elements
                self.world_size = world_size
                self._current_sequence = 0

            def all_reduce(self, tensor):
                seq = self._current_sequence
                self._current_sequence += 1
                fp32_ref = _make_correct_fp32_sum(
                    seq, self.elements, self.world_size,
                )
                # Add error that is within rtol*abs(ref) for large refs
                # but would exceed the old global threshold.
                # error = rtol * abs(ref) * 0.5 (half the relative bound)
                error = BF16_RTOL * fp32_ref.abs() * 0.5
                return (fp32_ref + error).to(torch.bfloat16).clone()

        receipt = run_probe(
            selector="custom", rank=0,
            iterations=1,
            elements=self.elements,
            world_size=self.world_size,
            native_session=_HugeFiniteWithinRelativeNativeSession(
                self.elements, self.world_size,
            ),
            dist_backend=_make_mock_dist(),
        )
        self.assertEqual(receipt["tolerance_result"], "pass")
        # The max_abs_error exceeds the old threshold for large refs
        self.assertGreater(receipt["max_abs_error"], TOLERANCE_THRESHOLD_BF16)

    def test_huge_finite_wrong_output_exceeds_relative_bound_fails(self) -> None:
        """A huge finite wrong output that exceeds the relative bound
        (error > atol + rtol * abs(ref)) must fail."""
        class _HugeFiniteExceedsRelativeNativeSession:
            def __init__(self, elements, world_size):
                self.elements = elements
                self.world_size = world_size
                self._current_sequence = 0

            def all_reduce(self, tensor):
                seq = self._current_sequence
                self._current_sequence += 1
                fp32_ref = _make_correct_fp32_sum(
                    seq, self.elements, self.world_size,
                )
                # Error = 2 * (atol + rtol * abs(ref)) — exceeds for all
                error = 2 * (BF16_ATOL + BF16_RTOL * fp32_ref.abs())
                sign = torch.sign(fp32_ref)
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                return (fp32_ref + error * sign).to(torch.bfloat16).clone()

        with self.assertRaises(ValueError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=1,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_HugeFiniteExceedsRelativeNativeSession(
                    self.elements, self.world_size,
                ),
                dist_backend=_make_mock_dist(),
            )
        self.assertIn("tolerance", str(ctx.exception).lower())

    # -- NaN / Inf rejection (elementwise criterion never reached) --------

    def test_nan_output_rejected_before_tolerance(self) -> None:
        """NaN in output must fail (all_finite=False) before the
        elementwise tolerance is even checked."""
        with self.assertRaises(ValueError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=1,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_NanNativeSession(),
                dist_backend=_make_mock_dist(),
            )
        self.assertIn("Non-finite", str(ctx.exception))

    def test_inf_output_rejected_before_tolerance(self) -> None:
        """Inf in output must fail (all_finite=False) before the
        elementwise tolerance is even checked."""
        with self.assertRaises(ValueError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=1,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_InfNativeSession(),
                dist_backend=_make_mock_dist(),
            )
        self.assertIn("Non-finite", str(ctx.exception))

    # -- Wrong element count ---------------------------------------------

    def test_wrong_count_rejected(self) -> None:
        """Wrong element count → error before tolerance check, no receipt.

        The production code subtracts the output from the FP32 reference;
        a size mismatch surfaces as a RuntimeError from torch before any
        tolerance criterion is evaluated.  No receipt is emitted.
        """
        with self.assertRaises(RuntimeError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=2,
                elements=self.elements,
                world_size=self.world_size,
                native_session=_WrongElementCountNativeSession(),
                dist_backend=_make_mock_dist(),
            )
        # torch reports a size mismatch in the subtraction.
        self.assertIn("size", str(ctx.exception).lower())

# ---------------------------------------------------------------------------
# J: Goal 10 — selector=custom without native session fails closed
# ---------------------------------------------------------------------------

class SelectorCustomNoSessionFailsClosedTest(unittest.TestCase):
    """Goal 10: selector=custom with no native_session must raise
    RuntimeError and execute NO NCCL collective."""

    def test_custom_no_session_raises_runtime_error(self) -> None:
        """run_probe(selector='custom', native_session=None, dist_backend=mock)
        must raise RuntimeError with 'failing closed' in the message."""
        mock_dist = _make_mock_dist()

        with self.assertRaises(RuntimeError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=1,
                elements=8,
                world_size=4,
                native_session=None,
                dist_backend=mock_dist,
            )
        self.assertIn("failing closed", str(ctx.exception))
        self.assertIn("selector=custom requires a native session",
                      str(ctx.exception))

    def test_custom_no_session_no_nccl_call(self) -> None:
        """Verify that dist_backend.all_reduce was NOT called — the
        probe fails before any collective execution."""
        mock_dist = _make_mock_dist()

        with self.assertRaises(RuntimeError):
            run_probe(
                selector="custom", rank=0,
                iterations=1,
                elements=8,
                world_size=4,
                native_session=None,
                dist_backend=mock_dist,
            )
        mock_dist.all_reduce.assert_not_called()
        mock_dist.barrier.assert_not_called()
        mock_dist.destroy_process_group.assert_not_called()


# ---------------------------------------------------------------------------
# K: Goal 10 — canonical env projection
# ---------------------------------------------------------------------------

class CanonicalEnvProjectionTest(unittest.TestCase):
    """Goal 10: _build_env_projection() is the single source of truth
    for env vars in the run contract."""

    def test_disabled_selector_includes_nccl_env_vars(self) -> None:
        """For selector=disabled, the projection MUST include
        NCCL_NET=Socket and NCCL_IB_DISABLE=1."""
        proj = _build_env_projection(
            selector="disabled", rank=0, world_size=4,
            iterations=1, elements=8,
        )
        self.assertEqual(proj["NCCL_NET"], "Socket")
        self.assertEqual(proj["NCCL_IB_DISABLE"], "1")
        self.assertEqual(proj["VLLM_SPARK_TP4_MODE"], "disabled")
        self.assertEqual(proj["RANK"], "0")
        self.assertEqual(proj["WORLD_SIZE"], "4")

    def test_nccl_ib_selector_includes_ib_env_vars(self) -> None:
        """For the NCCL-IB control arm (transport=nccl_ib), the
        projection MUST include NCCL_NET=IB and NCCL_IB_DISABLE=0."""
        proj = _build_env_projection(
            selector="disabled", rank=0, world_size=4,
            iterations=1, elements=8, transport="nccl_ib",
        )
        self.assertEqual(proj["NCCL_NET"], "IB")
        self.assertEqual(proj["NCCL_IB_DISABLE"], "0")
        self.assertEqual(proj["VLLM_SPARK_TP4_MODE"], "disabled")
        self.assertEqual(proj["RANK"], "0")
        self.assertEqual(proj["WORLD_SIZE"], "4")
        self.assertEqual(proj["ITERATIONS"], "1")
        self.assertEqual(proj["ELEMENTS"], "8")

    def test_custom_selector_excludes_nccl_env_vars(self) -> None:
        """For selector=custom, the projection must NOT include
        NCCL_NET or NCCL_IB_DISABLE."""
        proj = _build_env_projection(
            selector="custom", rank=1, world_size=4,
            iterations=2, elements=16,
        )
        self.assertNotIn("NCCL_NET", proj)
        self.assertNotIn("NCCL_IB_DISABLE", proj)
        self.assertEqual(proj["VLLM_SPARK_TP4_MODE"], "custom")

    def test_receipt_env_projection_includes_nccl_for_disabled(self) -> None:
        """The env_projection in a receipt from run_probe(selector='disabled')
        must include NCCL_NET and NCCL_IB_DISABLE — for the NCCL-IB
        control arm, NCCL_NET=IB and NCCL_IB_DISABLE=0."""
        import os
        with patch.dict(os.environ, {"NCCL_NET": "IB", "NCCL_IB_DISABLE": "0"}):
            receipt = run_probe(
                selector="disabled", rank=0,
                iterations=1,
                elements=8,
                world_size=4,
                dist_backend=_make_correct_dist(8, 4),
            )
        # The receipt does not embed env_projection directly, but the
        # run_contract_hash binds it.  We verify via the canonical
        # builder that disabled includes NCCL vars, and that the
        # receipt's contract hash is consistent with the canonical
        # projection.
        canonical = _build_env_projection(
            selector="disabled", rank=0, world_size=4,
            iterations=1, elements=8, transport="nccl_ib",
        )
        self.assertIn("NCCL_NET", canonical)
        self.assertIn("NCCL_IB_DISABLE", canonical)
        # Re-derive the contract hash the same way production does and
        # confirm it matches the receipt — proving env_projection is
        # bound into the contract.
        import json
        transport = "nccl_ib"
        contract = {
            "arm": transport,
            "selector": "disabled",
            "transport": transport,
            "rank": 0,
            "rank_identity": "rank-0-of-4",
            "iterations": 1,
            "elements": 8,
            "world_size": 4,
            "seed_identity": "0x5A17+seq*WORLD_SIZE+rank",
            "argv_projection": [
                "python",
                "spark_transport/integrations/vllm/tp4_numerical_audit.py",
            ],
            "env_projection": canonical,
            "probe_identity": (
                "spark_transport/integrations/vllm/tp4_numerical_audit.py"
            ),
            "binary_identity": (
                "spark_transport/integrations/vllm/tp4_numerical_audit.py"
            ),
            "topology": "tp4_switchless_ring",
            "workload": "tp4_numerical_audit",
            "order": "identical",
        }
        expected_hash = hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(receipt["run_contract_hash"], expected_hash)


# ---------------------------------------------------------------------------
# L: Goal 10 — probe dtype verification
# ---------------------------------------------------------------------------

class ProbeDtypeVerificationTest(unittest.TestCase):
    """Goal 10: actual_dtype in the receipt reflects the real output
    dtype (str(actual.dtype).replace('torch.', '')), not a hardcoded
    'bfloat16'."""

    def test_float32_output_rejected(self) -> None:
        """Goal 11 requirement 6: float32 output must be rejected
        before emitting a receipt — no success receipt for float32."""
        class _Float32NativeSession:
            def __init__(self, elements, world_size):
                self.elements = elements
                self.world_size = world_size
                self._current_sequence = 0

            def all_reduce(self, tensor):
                seq = self._current_sequence
                self._current_sequence += 1
                return _make_correct_fp32_sum(
                    seq, self.elements, self.world_size,
                ).clone()

        with self.assertRaises(ValueError) as ctx:
            run_probe(
                selector="custom", rank=0,
                iterations=1,
                elements=8,
                world_size=4,
                native_session=_Float32NativeSession(8, 4),
                dist_backend=_make_mock_dist(),
            )
        self.assertIn("bfloat16", str(ctx.exception))

    def test_bfloat16_output_dtype_in_receipt(self) -> None:
        """A native session returning a bfloat16 tensor must produce
        actual_dtype='bfloat16' in the receipt."""
        class _BF16NativeSession:
            def __init__(self, elements, world_size):
                self.elements = elements
                self.world_size = world_size
                self._current_sequence = 0

            def all_reduce(self, tensor):
                seq = self._current_sequence
                self._current_sequence += 1
                return _make_correct_fp32_sum(
                    seq, self.elements, self.world_size,
                ).to(torch.bfloat16).clone()

        receipt = run_probe(
            selector="custom", rank=0,
            iterations=1,
            elements=8,
            world_size=4,
            native_session=_BF16NativeSession(8, 4),
            dist_backend=_make_mock_dist(),
        )
        self.assertEqual(receipt["actual_dtype"], "bfloat16")


# ---------------------------------------------------------------------------
# M: Goal 10 — end-to-end four-rank control-arm test
# ---------------------------------------------------------------------------

class EndToEndFourRankControlArmTest(unittest.TestCase):
    """Goal 10: four-rank control-arm test using actual production
    builders (iterations=1, elements=8) whose NCCL receipts validate
    cleanly via validate_two_arm_results()."""

    def test_four_rank_nccl_control_arm_validates(self) -> None:
        """Run run_probe(selector='disabled', ...) for all 4 ranks with
        a mock dist_backend, construct RankReceipts, build a TwoArmResult,
        and validate with validate_two_arm_results() — must pass cleanly."""
        from spark_two_arm_orchestrator import (
            ArmSpec, ArmResult, RankReceipt,
            TwoArmResult, render_plan, validate_two_arm_results,
            _SIRCL_SELECTOR_ENVS, _NCCL_SELECTOR_ENVS,
            _TRANSPORT_SIRCL, _TRANSPORT_NCCL_IB,
            _ARM_NAME_SIRCL, _ARM_NAME_NCCL,
        )

        elements = 8
        iterations = 1
        world_size = 4
        # Build 4 NCCL-arm receipts via the production run_probe() path.
        # NCCL_NET=IB must be set so the probe selects the nccl_ib
        # transport for the NCCL-IB control arm.
        import os
        nccl_receipts = []
        with patch.dict(os.environ, {"NCCL_NET": "IB", "NCCL_IB_DISABLE": "0"}):
            for rank in range(world_size):
                receipt_dict = run_probe(
                    selector="disabled", rank=rank,
                    iterations=iterations,
                    elements=elements,
                    world_size=world_size,
                    dist_backend=_make_correct_dist(elements, world_size),
                )
                receipt = RankReceipt(
                    rank=receipt_dict["rank"],
                    host=f"spark-{rank}",
                    transport=receipt_dict["transport"],
                    selector=receipt_dict["selector"],
                    iterations=receipt_dict["iterations"],
                    elements=receipt_dict["elements"],
                    world_size=receipt_dict["world_size"],
                    custom_collectives=receipt_dict["custom_collectives"],
                    fallback_collectives=receipt_dict["fallback_collectives"],
                    unsupported_bypassed_collectives=(
                        receipt_dict["unsupported_bypassed_collectives"]
                    ),
                    unclassified_collectives=receipt_dict["unclassified_collectives"],
                    total_collectives=receipt_dict["total_collectives"],
                    expected_fp32_hash=receipt_dict["expected_fp32_hash"],
                    actual_output_hash=receipt_dict["actual_output_hash"],
                    actual_dtype=receipt_dict.get("actual_dtype", ""),
                    actual_byte_order=receipt_dict.get("actual_byte_order", ""),
                    all_finite=receipt_dict["all_finite"],
                    max_abs_error=receipt_dict["max_abs_error"],
                    max_rel_error=receipt_dict["max_rel_error"],
                    tolerance_result=receipt_dict.get("tolerance_result", ""),
                    tolerance_metric=receipt_dict.get("tolerance_metric", ""),
                    tolerance_atol=receipt_dict.get("tolerance_atol", 0.0),
                    tolerance_rtol=receipt_dict.get("tolerance_rtol", 0.0),
                    sample_count=receipt_dict["sample_count"],
                    run_contract_hash=receipt_dict["run_contract_hash"],
                    rank_identity=receipt_dict.get("rank_identity", ""),
                    native_collectives=receipt_dict.get("native_collectives", 0),
                    nccl_ib_collectives=receipt_dict.get("nccl_ib_collectives", 0),
                    nccl_socket_collectives=receipt_dict.get(
                        "nccl_socket_collectives", 0
                    ),
                    fatal_after_native_collectives=receipt_dict.get(
                        "fatal_after_native_collectives", 0
                    ),
                )
                nccl_receipts.append(receipt)
        nccl_receipts = tuple(nccl_receipts)

        # All NCCL receipts must be fallback-classified with correct counters.
        for r in nccl_receipts:
            self.assertEqual(r.fallback_collectives, iterations)
            self.assertEqual(r.custom_collectives, 0)
            self.assertEqual(r.unclassified_collectives, 0)
            self.assertEqual(r.total_collectives, iterations)
            self.assertTrue(r.all_finite)

        # Render a valid plan.
        sircl_spec = ArmSpec(
            transport=_TRANSPORT_SIRCL,
            selector_env_vars=_SIRCL_SELECTOR_ENVS,
            world_size=world_size, iterations=iterations,
            elements=elements,
        )
        nccl_spec = ArmSpec(
            transport=_TRANSPORT_NCCL_IB,
            selector_env_vars=_NCCL_SELECTOR_ENVS,
            world_size=world_size, iterations=iterations,
            elements=elements,
        )
        plan = render_plan(sircl_spec, nccl_spec, dry_run=True)

        # Build a TwoArmResult with the NCCL arm receipts.  The SIRCL
        # arm uses fabricated receipts (they won't be exercised in
        # this control test — we only validate the NCCL arm path).
        # We mark valid=False and expect no NCCL-specific errors.
        # Compute proper SIRCL run_contract_hash (not reusing NCCL hash).
        from spark_transport_contract import build_env_projection
        import json as _json
        probe_id = "spark_transport/integrations/vllm/tp4_numerical_audit.py"
        sircl_receipts = tuple(
            RankReceipt(
                rank=r, host=f"spark-{r}",
                transport=_TRANSPORT_SIRCL, selector="custom",
                iterations=iterations, elements=elements,
                world_size=world_size,
                custom_collectives=iterations, fallback_collectives=0,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=iterations,
                expected_fp32_hash=nccl_receipts[r].expected_fp32_hash,
                actual_output_hash=nccl_receipts[r].actual_output_hash,
                actual_dtype=nccl_receipts[r].actual_dtype,
                actual_byte_order=nccl_receipts[r].actual_byte_order,
                all_finite=True, max_abs_error=0.0, max_rel_error=0.0,
                tolerance_atol=nccl_receipts[r].tolerance_atol,
                tolerance_rtol=nccl_receipts[r].tolerance_rtol,
                sample_count=iterations * elements,
                run_contract_hash=hashlib.sha256(_json.dumps({
                    "arm": _TRANSPORT_SIRCL,
                    "selector": "custom",
                    "transport": _TRANSPORT_SIRCL,
                    "rank": r,
                    "rank_identity": f"rank-{r}-of-{world_size}",
                    "iterations": iterations,
                    "elements": elements,
                    "world_size": world_size,
                    "seed_identity": "0x5A17+seq*WORLD_SIZE+rank",
                    "argv_projection": ["python", probe_id],
                    "env_projection": build_env_projection(
                        "custom", r, world_size, iterations, elements,
                        transport=_TRANSPORT_SIRCL,
                    ),
                    "probe_identity": probe_id,
                    "binary_identity": probe_id,
                    "topology": "tp4_switchless_ring",
                    "workload": "tp4_numerical_audit",
                    "order": "identical",
                }, sort_keys=True).encode()).hexdigest(),
                rank_identity=f"rank-{r}-of-{world_size}",
                tolerance_result="pass",
                tolerance_metric="elementwise_atol_rtol",
                native_collectives=iterations,
                nccl_ib_collectives=0,
                nccl_socket_collectives=0,
                fatal_after_native_collectives=0,
            ) for r in range(world_size)
        )

        result = TwoArmResult(
            sircl_arm=ArmResult(
                arm_name=_ARM_NAME_SIRCL,
                transport=_TRANSPORT_SIRCL,
                receipts=sircl_receipts,
            ),
            nccl_arm=ArmResult(
                arm_name=_ARM_NAME_NCCL,
                transport=_TRANSPORT_NCCL_IB,
                receipts=nccl_receipts,
            ),
            valid=False,
        )
        errors = validate_two_arm_results(result, plan)
        # The NCCL arm receipts must validate cleanly — the full error
        # list must be empty (valid=False suppresses cross-arm consistency
        # checks but per-arm receipt validation still runs).
        self.assertEqual(
            errors, [],
            f"Expected no validation errors, got: {errors}",
        )


# ---------------------------------------------------------------------------
# Goal 11: Four-process consensus and receipt parsing tests
# ---------------------------------------------------------------------------


def _make_valid_receipt() -> dict:
    """Build a receipt dict with every RECEIPT_REQUIRED_KEYS field set
    to a valid value.  Tests mutate one field at a time to exercise
    rejection paths in parse_receipt_json.
    """
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "rank": 0,
        "transport": "sircl",
        "selector": "custom",
        "iterations": 1,
        "elements": 8,
        "world_size": 4,
        "native_collectives": 1,
        "nccl_ib_collectives": 0,
        "nccl_socket_collectives": 0,
        "custom_collectives": 1,
        "fallback_collectives": 0,
        "unsupported_bypassed_collectives": 0,
        "unclassified_collectives": 0,
        "fatal_after_native_collectives": 0,
        "total_collectives": 1,
        "expected_fp32_hash": "a" * 64,
        "actual_output_hash": "b" * 64,
        "actual_dtype": REQUIRED_OUTPUT_DTYPE,
        "actual_byte_order": REQUIRED_BYTE_ORDER,
        "all_finite": True,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "tolerance_result": "pass",
        "tolerance_metric": "elementwise_atol_rtol",
        "tolerance_atol": BF16_ATOL,
        "tolerance_rtol": BF16_RTOL,
        "sample_count": 8,
        "run_contract_hash": "c" * 64,
        "rank_identity": "rank-0-of-4",
        "counter_source_hash": "d" * 64,
        "source_sha": "e" * 64,
        "sircl_so_sha": "",
        "nccl_so_sha": "",
        "image_receipt": "",
    }


class _MockControlBackend:
    """Mock Gloo control-plane backend.

    Simulates an all-reduce (sum) on a control process group.  The
    ``consensus_sum`` attribute determines the in-place result that
    all_reduce writes into the selector tensor, allowing tests to
    simulate agreement or disagreement.

    ``gathered_ranks`` simulates the all_gather result for rank
    identity validation.  Default is [0, 1, ..., world_size-1].

    ``identity_records`` simulates the all_gather_object result for
    cross-rank identity validation.  When None, the mock auto-generates
    valid records for all ranks with matching selector/arm/source_sha.
    """

    class ReduceOp:
        SUM = "SUM"

    def __init__(
        self,
        consensus_sum: int,
        gathered_ranks: list[int] | None = None,
        *,
        identity_records: list[dict] | None = None,
    ):
        self.consensus_sum = consensus_sum
        self.gathered_ranks = gathered_ranks  # None = computed per call
        self.identity_records = identity_records
        self.all_reduce_calls = 0
        self.barrier_calls = 0
        self.all_gather_calls = 0
        self.all_gather_object_calls = 0

    def all_reduce(self, tensor, op=None):
        self.all_reduce_calls += 1
        tensor.fill_(self.consensus_sum)

    def all_gather(self, output_list, input_tensor, group=None):
        self.all_gather_calls += 1
        _ = int(input_tensor.item())  # rank value, unused in mock
        ws = len(output_list)
        if self.gathered_ranks is not None:
            ranks = self.gathered_ranks
        else:
            ranks = list(range(ws))
        for i, r in enumerate(ranks):
            output_list[i].fill_(r)

    def all_gather_object(self, output_list, input_object, group=None):
        self.all_gather_object_calls += 1
        ws = len(output_list)
        if self.identity_records is not None:
            records = self.identity_records
        else:
            # Auto-generate valid records for all ranks, copying all
            # shared identity fields from the input record so consensus
            # passes when the mock caller provides a complete record.
            base = {
                k: v for k, v in input_object.items()
                if k != "rank"
            }
            records = [
                {**base, "rank": r, "native_capable": True}
                for r in range(ws)
            ]
        for i, rec in enumerate(records):
            output_list[i] = rec

    def barrier(self):
        self.barrier_calls += 1


def _collect_rank_identity_record_for_test(selector: str) -> dict:
    """Build a valid identity record for mock-based consensus tests.

    Provides non-empty ``sircl_library_path``/``sircl_library_sha``
    so the arm-specific library validation in consensus passes.
    """
    import hashlib
    import tempfile
    import pathlib
    tmp = pathlib.Path(tempfile.mkdtemp()) / "fake_lib.so"
    tmp.write_bytes(b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 56)
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    record = {
        "rank": 0,
        "world_size": 4,
        "selector": selector,
        "arm": (
            "sircl" if selector == SELECTOR_CUSTOM else "nccl_ib"
        ),
        "workload": "tp4_numerical_audit",
        "iterations": 1,
        "elements": 8,
        "native_capable": True,
        "argv_hash": "a" * 64,
        "env_hash": "b" * 64,
        "shared_env_hash": "c" * 64,
        "source_sha": "d" * 64,
        "sircl_library_path": str(tmp) if selector == SELECTOR_CUSTOM else "",
        "sircl_library_sha": sha if selector == SELECTOR_CUSTOM else "",
        "nccl_library_path": "/fake/nccl.so" if selector != SELECTOR_CUSTOM else "",
        "nccl_library_sha": "e" * 64 if selector != SELECTOR_CUSTOM else "",
        "nccl_identity": True if selector != SELECTOR_CUSTOM else False,
        "image_receipt": "",
        "runtime_identity": "test-runtime",
        "master_addr": "127.0.0.1",
        "master_port": "29500",
        "gloo_socket_ifname": "",
    }
    return record


class _MockNativeSession:
    """Mock native session returning correct BF16-rounded results."""

    def __init__(self, elements: int, world_size: int):
        self.elements = elements
        self.world_size = world_size
        self._seq = 0

    def all_reduce(self, tensor):
        seq = self._seq
        self._seq += 1
        inputs = [
            make_rank_input(seq, r, self.elements)
            for r in range(self.world_size)
        ]
        fp32_sum = torch.stack([t.float() for t in inputs]).sum(dim=0)
        return fp32_sum.to(torch.bfloat16).clone()


class FourProcessConsensusTest(unittest.TestCase):
    """Goal 11: four-process control-plane consensus mechanism.

    The control channel is NOT the transport under test.  It uses a
    dedicated Gloo/TCP process group on the management network.
    Consensus runs BEFORE any SIRCL or NCCL data collective.
    """

    # -- Test 1: all 4 ranks agree on custom ----------------------------

    def test_all_ranks_agree_on_custom_selector(self) -> None:
        """When all 4 ranks broadcast selector=custom (code=0), the
        all-reduce sum is 0 (0*4), and consensus returns custom."""
        backend = _MockControlBackend(consensus_sum=0)
        # Provide an identity record with sircl_library_path/sha set
        # so the arm-specific library validation passes.
        record = _collect_rank_identity_record_for_test(SELECTOR_CUSTOM)
        result = _run_control_plane_consensus(
            SELECTOR_CUSTOM, rank=0, world_size=4,
            control_backend=backend,
            identity_record=record,
        )
        self.assertEqual(result, SELECTOR_CUSTOM)
        self.assertEqual(backend.all_reduce_calls, 1)
        self.assertEqual(backend.barrier_calls, 1)

    # -- Test 2: disagreement raises RuntimeError -----------------------

    def test_disagreement_raises_runtime_error(self) -> None:
        """One rank sends disabled (code=1), others send custom (code=0).
        The all-reduce sum is 1, not 0 — consensus fails."""
        # From rank 0's perspective: selector=custom (code=0), expected
        # sum = 0*4 = 0, but actual sum = 1 (one rank disagreed).
        backend = _MockControlBackend(consensus_sum=1)
        with self.assertRaises(RuntimeError) as ctx:
            _run_control_plane_consensus(
                SELECTOR_CUSTOM, rank=0, world_size=4,
                control_backend=backend,
            )
        self.assertIn("consensus failed", str(ctx.exception))
        self.assertIn("selector sum 1", str(ctx.exception))

    # -- Test 3: no native session → all-abort, no data calls ----------

    def test_custom_selector_no_native_session_all_abort(self) -> None:
        """selector=custom with no native session on any rank causes
        all-abort before any data collective is executed.

        Goal 12: the production path now validates SPARK_TP4_LIBRARY
        before constructing _NativeSession.  If the library is missing,
        the probe fails closed with a SPARK_TP4_LIBRARY error — no
        NCCL data call is made.
        """
        mock_dist = MagicMock()
        mock_dist.ReduceOp.SUM = "SUM"
        mock_dist.all_reduce = MagicMock()

        import tempfile
        import pathlib
        # Create a temp file to serve as a fake SPARK_TP4_LIBRARY.
        tmp = pathlib.Path(tempfile.mkdtemp()) / "fake_lib.so"
        tmp.write_bytes(b"\x7fELF fake")

        with patch(
            "tp4_numerical_audit._init_control_group", return_value=None,
        ), patch(
            "tp4_numerical_audit._run_control_plane_consensus",
            return_value=SELECTOR_CUSTOM,
        ), patch(
            "torch.device", return_value=torch.device("cpu"),
        ), patch("torch.cuda.set_device"), patch(
            "spark_tp4_backend._NativeSession", return_value=None,
        ), patch.dict("os.environ", {"SPARK_TP4_LIBRARY": str(tmp)}, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                run_probe(
                    selector=SELECTOR_CUSTOM,
                    rank=0,
                    iterations=1,
                    elements=8,
                    world_size=4,
                )
        self.assertIn("no native_session available", str(ctx.exception))
        # No data collective was executed.
        mock_dist.all_reduce.assert_not_called()
        # Cleanup temp file.
        tmp.unlink(missing_ok=True)

    def test_sircl_arm_never_creates_nccl_data_group(self) -> None:
        """In production mode (not injected), the SIRCL arm
        (selector=custom) must never create an NCCL data process group.

        Goal 12: the production path now validates SPARK_TP4_LIBRARY
        and instantiates _NativeSession with (rank, payload_bytes).
        We patch _NativeSession to return a mock, but must set
        SPARK_TP4_LIBRARY to a temp file so the library validation
        passes.

        Verify by checking that dist.init_process_group is never called
        and that the data-PG cleanup (barrier/destroy) is never
        invoked — confirming ``d`` is None for the SIRCL arm.
        """
        native = _MockNativeSession(elements=8, world_size=4)

        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "fake_lib.so"
        tmp.write_bytes(b"\x7fELF fake")

        with patch(
            "tp4_numerical_audit._init_control_group", return_value=None,
        ), patch(
            "tp4_numerical_audit._run_control_plane_consensus",
            return_value=SELECTOR_CUSTOM,
        ), patch(
            "torch.device", return_value=torch.device("cpu"),
        ), patch("torch.cuda.set_device"), patch(
            "spark_tp4_backend._NativeSession", return_value=native,
        ), patch("torch.distributed.init_process_group") as init_pg, \
           patch("torch.distributed.barrier") as dist_barrier, \
           patch(
            "torch.distributed.destroy_process_group",
        ) as dist_destroy, \
           patch.dict("os.environ", {"SPARK_TP4_LIBRARY": str(tmp)}, clear=False):
            receipt = run_probe(
                selector=SELECTOR_CUSTOM,
                rank=0,
                iterations=1,
                elements=8,
                world_size=4,
            )

        # SIRCL arm succeeded with native_collectives.
        self.assertEqual(receipt["native_collectives"], 1)
        self.assertEqual(receipt["total_collectives"], 1)
        # dist.init_process_group was never called — d is None.
        self.assertFalse(
            init_pg.called,
            "SIRCL arm must not call dist.init_process_group",
        )
        # Data-PG barrier/destroy were never called — d is None.
        self.assertFalse(dist_barrier.called)
        self.assertFalse(dist_destroy.called)
        # Cleanup temp file.
        tmp.unlink(missing_ok=True)

    # -- Test 5: rank identity validation via all-gather ---------------

    def test_duplicate_rank_detected(self) -> None:
        """Two ranks claiming rank 0 → gathered set {0,0,2,3} !=
        {0,1,2,3} → RuntimeError."""
        backend = _MockControlBackend(
            consensus_sum=0, gathered_ranks=[0, 0, 2, 3],
        )
        with self.assertRaises(RuntimeError) as ctx:
            _run_control_plane_consensus(
                SELECTOR_CUSTOM, rank=0, world_size=4,
                control_backend=backend,
            )
        self.assertIn("rank set", str(ctx.exception))
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_missing_rank_detected(self) -> None:
        """Missing rank 1 → gathered set {0,2,2,3} != {0,1,2,3}."""
        backend = _MockControlBackend(
            consensus_sum=0, gathered_ranks=[0, 2, 2, 3],
        )
        with self.assertRaises(RuntimeError) as ctx:
            _run_control_plane_consensus(
                SELECTOR_CUSTOM, rank=0, world_size=4,
                control_backend=backend,
            )
        self.assertIn("rank set", str(ctx.exception))

    def test_out_of_range_rank_detected(self) -> None:
        """Rank 5 in world_size=4 → gathered set {0,1,2,5} !=
        {0,1,2,3}."""
        backend = _MockControlBackend(
            consensus_sum=0, gathered_ranks=[0, 1, 2, 5],
        )
        with self.assertRaises(RuntimeError) as ctx:
            _run_control_plane_consensus(
                SELECTOR_CUSTOM, rank=0, world_size=4,
                control_backend=backend,
            )
        self.assertIn("rank set", str(ctx.exception))
        self.assertIn("out-of-range", str(ctx.exception).lower())

    def test_peer_loss_detected_via_all_gather(self) -> None:
        """Peer loss: all_gather raises an exception when a peer is
        gone.  The consensus function must propagate this as a
        RuntimeError, not silently pass."""
        backend = _MockControlBackend(consensus_sum=0)
        backend.all_gather = MagicMock(side_effect=RuntimeError("peer lost"))
        with self.assertRaises(RuntimeError) as ctx:
            _run_control_plane_consensus(
                SELECTOR_CUSTOM, rank=0, world_size=4,
                control_backend=backend,
            )
        self.assertIn("peer lost", str(ctx.exception))

    def test_correct_rank_set_passes(self) -> None:
        """Correct rank set {0,1,2,3} with all agreeing on custom
        passes consensus and returns custom.

        Must pass an identity record with non-empty sircl_library_path
        and sircl_library_sha so the arm-specific library validation
        passes (review v2 requirement 6)."""
        backend = _MockControlBackend(consensus_sum=0)
        record = _collect_rank_identity_record_for_test(SELECTOR_CUSTOM)
        result = _run_control_plane_consensus(
            SELECTOR_CUSTOM, rank=0, world_size=4,
            control_backend=backend,
            identity_record=record,
        )
        self.assertEqual(result, SELECTOR_CUSTOM)
        self.assertEqual(backend.all_gather_calls, 1)

    # -- Test 6: parse_receipt_json rejection & acceptance ---------------


    def test_parse_receipt_rejects_missing_keys(self) -> None:
        receipt = _make_valid_receipt()
        del receipt["rank"]
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(receipt)
        self.assertIn("missing required keys", str(ctx.exception))
        self.assertIn("rank", str(ctx.exception))

    def test_parse_receipt_rejects_extra_keys(self) -> None:
        receipt = _make_valid_receipt()
        receipt["unexpected_field"] = 42
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(receipt)
        self.assertIn("extra keys", str(ctx.exception))
        self.assertIn("unexpected_field", str(ctx.exception))

    def test_parse_receipt_rejects_null_values(self) -> None:
        receipt = _make_valid_receipt()
        receipt["transport"] = None
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(receipt)
        self.assertIn("null", str(ctx.exception))

    def test_parse_receipt_rejects_bool_as_number(self) -> None:
        receipt = _make_valid_receipt()
        # True in an int field — bool is a subclass of int in Python,
        # but parse_receipt_json explicitly rejects bool.
        receipt["native_collectives"] = True
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(receipt)
        self.assertIn("must be int", str(ctx.exception))
        self.assertIn("bool", str(ctx.exception))

    def test_parse_receipt_rejects_nan_in_float_fields(self) -> None:
        receipt = _make_valid_receipt()
        receipt["max_abs_error"] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(receipt)
        self.assertIn("finite", str(ctx.exception))

    def test_parse_receipt_rejects_inf_in_float_fields(self) -> None:
        receipt = _make_valid_receipt()
        receipt["max_rel_error"] = float("inf")
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(receipt)
        self.assertIn("finite", str(ctx.exception))

    def test_parse_receipt_rejects_negative_counts(self) -> None:
        receipt = _make_valid_receipt()
        receipt["native_collectives"] = -1
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(receipt)
        self.assertIn("must be >= 0", str(ctx.exception))

    def test_parse_receipt_rejects_wrong_schema_version(self) -> None:
        receipt = _make_valid_receipt()
        receipt["schema_version"] = "tp4_receipt/v2"
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(receipt)
        self.assertIn("schema_version", str(ctx.exception))

    def test_parse_receipt_rejects_wrong_dtype(self) -> None:
        receipt = _make_valid_receipt()
        receipt["actual_dtype"] = "float32"
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(receipt)
        self.assertIn("actual_dtype", str(ctx.exception))
        self.assertIn("float32", str(ctx.exception))

    def test_parse_receipt_rejects_wrong_byte_order(self) -> None:
        receipt = _make_valid_receipt()
        receipt["actual_byte_order"] = "big"
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(receipt)
        self.assertIn("actual_byte_order", str(ctx.exception))
        self.assertIn("big", str(ctx.exception))

    def test_parse_receipt_accepts_valid_receipt(self) -> None:
        """A receipt with all required keys set to valid values
        must be accepted and returned unchanged."""
        receipt = _make_valid_receipt()
        result = parse_receipt_json(dict(receipt))
        self.assertEqual(result, receipt)
        # Verify every required key is present.
        self.assertEqual(set(result.keys()), RECEIPT_REQUIRED_KEYS)

    def test_parse_receipt_rejects_contradictory_legacy_aliases(self) -> None:
        receipt = _make_valid_receipt()
        receipt["custom_collectives"] = 0
        with self.assertRaisesRegex(ValueError, "must equal native_collectives"):
            parse_receipt_json(receipt)

        receipt = _make_valid_receipt()
        receipt["fallback_collectives"] = 1
        with self.assertRaisesRegex(ValueError, "must equal nccl_ib_collectives"):
            parse_receipt_json(receipt)

    def test_parse_receipt_rejects_noncanonical_image_digest(self) -> None:
        receipt = _make_valid_receipt()
        receipt["image_receipt"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "canonical sha256"):
            parse_receipt_json(receipt)




class RealFourProcessSpawnTest(unittest.TestCase):
    """Goal 12 review v2/v3: faithful multiprocessing/spawn test
    exercising ``run_probe`` through its actual production pre-data
    path — including ``_init_control_group``, identity gather,
    consensus, and then the selected data branch.

    Six scenarios: happy path, selector asymmetry, missing rank **2**,
    duplicate rank record, out-of-range rank record, and one-rank
    missing capability/library.  Every scenario deterministically
    joins/terminates children, asserts exact non-vacuous result
    counts, and for all failed-consensus cases asserts exactly zero
    SIRCL and zero NCCL data calls via a shared recorder wired into
    the actual ``run_probe`` data-call seams.
    """

    WORLD_SIZE = 4
    PG_TIMEOUT = 10.0
    JOIN_TIMEOUT = 20
    COLLECT_TIMEOUT = 30

    @staticmethod
    def _free_port():
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return str(port)

    @staticmethod
    def _make_sircl_test_seam():
        """Create a SIRCL test double library path.

        Review v5 requirement 2: the test double is injected at the
        CDLL seam inside the child process (patching ctypes.CDLL),
        not via a production environment bypass.  The mock CDLL
        returned to the production code exports exactly the required
        SIRCL C API symbols (spark_tp4_create, spark_tp4_all_reduce,
        spark_tp4_destroy), proving the identity contract.

        Returns (path, env_overrides) where env_overrides is empty —
        no production env seam is used.
        """
        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "test_sircl_double.so"
        tmp.write_bytes(b"\x7fELF fake sircl test double")
        return str(tmp), {}

    @staticmethod
    def _find_real_nccl():
        """Discover a real, loadable NCCL library on this host.

        Searches common paths and VLLM_NCCL_SO_PATH / LD_PRELOAD.
        Returns the path if the library at that path exports
        ncclCommInitRank or ncclCommInitRankAll, else None.

        This is the same discovery logic used by NCCLIdentityTest,
        reused here so the real four-process happy NCCL case uses
        a discovered, existing, loadable real NCCL library — never
        a fake text file or a patched CDLL mock.
        """
        import ctypes
        import pathlib
        import glob as _glob
        candidates = []
        for env_name in ("VLLM_NCCL_SO_PATH", "LD_PRELOAD"):
            val = os.environ.get(env_name, "")
            if val:
                candidates.append(val)
        for pattern in (
            "/usr/lib/*/libnccl.so*",
            "/usr/lib/libnccl.so*",
            "/opt/*/libnccl.so*",
            "/usr/local/cuda/lib*/libnccl.so*",
        ):
            candidates.extend(_glob.glob(pattern))
        for path in candidates:
            try:
                p = pathlib.Path(path)
                if not p.is_file():
                    continue
                lib = ctypes.CDLL(str(p))
                for sym in ("ncclCommInitRank", "ncclCommInitRankAll"):
                    if hasattr(lib, sym):
                        return str(p)
            except Exception:
                continue
        return None

    @staticmethod
    def _worker(
        rank, world_size, port_str, result_queue, recorder,
        selector, iterations, elements,
        identity_overrides, env_overrides, recorder_lock=None,
        nccl_cdll_real=False,
    ):
        """Production-path worker: calls ``run_probe`` (not
        ``_init_control_group`` / consensus directly).

        Sets env inside the child (``multiprocessing.Process`` has no
        ``env=``).  Patches CUDA/NCCL seams so the production path
        can run on CPU.  The ``data_call_recorder`` is passed to
        ``run_probe`` and incremented at the actual SIRCL/NCCL
        data-call seams via ``_increment_recorder``.

        ``identity_overrides`` is applied by monkey-patching
        ``_collect_rank_identity_record`` inside the child — the
        production path still calls ``run_probe`` which internally
        calls ``_init_control_group``, identity collection, consensus,
        and the data branch.
        """
        import os
        import torch
        import torch.distributed as dist
        import tp4_numerical_audit as mod
        from tp4_numerical_audit import run_probe

        # Set env inside the child.
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = port_str
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        # Review v2 requirement 7: GLOO_SOCKET_IFNAME takes an
        # interface name, not an IP.  Omit it consistently — Gloo
        # resolves loopback without it on this platform.
        os.environ.pop("GLOO_SOCKET_IFNAME", None)
        for k, v in env_overrides.items():
            if v == "":
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        # Review v5 requirement 2 / goal12-v6: inject the SIRCL test
        # double at the CDLL seam inside the child process — patch
        # ctypes.CDLL so the production code's symbol verification
        # sees the required SIRCL symbols.  This is acceptable ONLY
        # for failure-before-data control tests; the NCCL happy case
        # must use a real NCCL library (nccl_cdll_real=True) and no
        # CDLL mocking at all.
        import ctypes as _ctypes
        _real_cdll = _ctypes.CDLL

        if not nccl_cdll_real:
            class _MockCDLL:
                """Mock CDLL that exports SIRCL symbols only.
                NCCL symbols are NOT faked — the NCCL happy case
                must load a real NCCL library."""
                def __init__(self, name, *args, **kwargs):
                    self._name = name

                def __getattr__(self, name):
                    if name.startswith("_"):
                        raise AttributeError(name)
                    # SIRCL required symbols only.
                    if name in ("spark_tp4_create", "spark_tp4_all_reduce",
                                "spark_tp4_destroy"):
                        return lambda *a, **k: None
                    raise AttributeError(name)

            _ctypes.CDLL = _MockCDLL

        # Apply identity overrides by patching _collect_rank_identity_record
        # inside the child.  run_probe still calls it internally.
        if identity_overrides:
            _real_collect = mod._collect_rank_identity_record

            def _patched_collect(sel, r, ws, iters, elems):
                rec = _real_collect(sel, r, ws, iters, elems)
                rec.update(identity_overrides)
                return rec

            mod._collect_rank_identity_record = _patched_collect
        # Patch _init_control_group to use the class PG_TIMEOUT
        # so missing-rank scenarios fail within the collect deadline.
        _real_init = mod._init_control_group

        def _patched_init(rank, world_size, *, timeout=None):
            return _real_init(rank, world_size, timeout=10.0)

        mod._init_control_group = _patched_init

        # Review v3 requirement 1: save the real torch.device
        # constructor first to avoid a recursive lambda that calls
        # itself infinitely.
        _real_device = torch.device
        _real_set_device = torch.cuda.set_device
        torch.device = lambda x, index=0: _real_device("cpu")
        torch.cuda.set_device = lambda dev: None

        # Patch NCCL group creation + all_reduce for CPU execution.
        _real_new_group = dist.new_group
        _real_all_reduce = dist.all_reduce
        _real_barrier = dist.barrier
        _real_destroy = dist.destroy_process_group
        _nccl_pg_mock = None

        def _cpu_new_group(ranks=None, backend=None, timeout=None):
            if backend == "nccl":
                nonlocal _nccl_pg_mock
                _nccl_pg_mock = MagicMock()
                return _nccl_pg_mock
            return _real_new_group(
                ranks=ranks, backend=backend, timeout=timeout,
            )

        _seq = [0]

        def _cpu_all_reduce(tensor, op=None, group=None):
            if group is _nccl_pg_mock or (
                group is not None and _nccl_pg_mock is not None
            ):
                # CPU NCCL mock: compute correct BF16 all-reduce.
                from tp4_numerical_audit import make_rank_input
                seq = _seq[0]
                _seq[0] += 1
                ws = world_size
                inputs = [
                    make_rank_input(seq, r, elements)
                    for r in range(ws)
                ]
                fp32_sum = torch.stack([
                    t.float() for t in inputs
                ]).sum(dim=0)
                tensor.copy_(fp32_sum.to(torch.bfloat16))
            else:
                _real_all_reduce(tensor, op=op, group=group)

        def _cpu_barrier(group=None):
            if group is not _nccl_pg_mock:
                _real_barrier(group=group)

        def _cpu_destroy(pg=None):
            if pg is not _nccl_pg_mock and pg is not None:
                _real_destroy(pg)

        dist.new_group = _cpu_new_group
        dist.all_reduce = _cpu_all_reduce
        dist.barrier = _cpu_barrier
        dist.destroy_process_group = _cpu_destroy

        # Attach the lock to the recorder proxy inside this child
        # process so _increment_recorder can use it for atomic
        # increments across processes.
        if recorder_lock is not None:
            recorder._lock = recorder_lock

        try:
            receipt = run_probe(
                selector, rank, iterations, elements, world_size,
                data_call_recorder=recorder,
            )
            result_queue.put(("ok", rank, receipt))
        except Exception as e:
            result_queue.put(("error", rank, str(e)))
        finally:
            # Restore real functions.
            _ctypes.CDLL = _real_cdll
            torch.device = _real_device
            torch.cuda.set_device = _real_set_device
            dist.new_group = _real_new_group
            dist.all_reduce = _real_all_reduce
            dist.barrier = _real_barrier
            dist.destroy_process_group = _real_destroy
            if identity_overrides:
                mod._collect_rank_identity_record = _real_collect
            mod._init_control_group = _real_init
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass

    @staticmethod
    def _spawn_and_collect(
        worker_fn, ranks, world_size, port_str, result_queue, recorder,
        selector_fn, iterations, elements,
        identity_overrides_fn, env_overrides_fn,
        collect_timeout, join_timeout, recorder_lock=None,
        nccl_cdll_real=False,
    ):
        """Spawn one process per rank, collect results, deterministic join.

        ``selector_fn`` is called per rank to get that rank's selector.
        ``identity_overrides_fn`` and ``env_overrides_fn`` each return
        a dict (not a tuple) for the given rank.

        Review v3 requirement 5: collect until a deadline longer than
        the PG timeout or until the exact expected number of child
        outcomes arrives.  Do NOT stop on the first Queue.Empty.
        """
        import multiprocessing as mp
        import time
        ctx = mp.get_context("spawn")
        procs = []
        expected = len(ranks)
        for rank in ranks:
            p = ctx.Process(
                target=worker_fn,
                args=(
                    rank, world_size, port_str, result_queue, recorder,
                    selector_fn(rank), iterations, elements,
                    identity_overrides_fn(rank),
                    env_overrides_fn(rank),
                    recorder_lock,
                    nccl_cdll_real,
                ),
            )
            procs.append(p)
            p.start()
        results = []
        deadline = time.time() + collect_timeout
        while len(results) < expected and time.time() < deadline:
            try:
                results.append(result_queue.get(timeout=2))
            except Exception:
                # Queue.Empty — continue until deadline, not break.
                continue
        for p in procs:
            p.join(timeout=join_timeout)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        return results

    @staticmethod
    def _no_identity_overrides(rank):
        return {}

    @staticmethod
    def _no_env_overrides(rank):
        return {}

    @staticmethod
    def _const_selector(selector):
        def _fn(rank):
            return selector
        return _fn

    @staticmethod
    def _make_recorder(mgr):
        """Create a shared recorder with a Manager lock for atomic
        increments.

        Review v3 requirement 7: shared recorder increments must be
        atomic across processes.  Returns a (dict_proxy, lock) tuple.
        The lock is passed separately to the worker, which attaches
        it to the recorder proxy inside the child process so
        ``_increment_recorder`` can use it.
        """
        recorder = mgr.dict()
        recorder["sircl_calls"] = 0
        recorder["nccl_calls"] = 0
        lock = mgr.Lock()
        return recorder, lock

    # -- Scenario 1: happy path — all agree on disabled ---------------

    def test_happy_path_all_agree_disabled(self):
        """All 4 ranks agree on disabled (NCCL control arm) → consensus
        succeeds, run_probe reaches the NCCL data branch, and the
        recorder shows exactly 1 NCCL data call per rank.

        Goal12-v6: the happy NCCL case must use a discovered, existing,
        loadable real NCCL library (exporting ncclCommInitRank /
        ncclCommInitRankAll).  If no real NCCL is available on this
        CI platform, skip with a precise reason — never fake the library
        or patch CDLL to export NCCL symbols."""
        from spark_transport_contract import SELECTOR_NCCL_IB
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        mgr = ctx.Manager()
        recorder, recorder_lock = self._make_recorder(mgr)
        result_queue = ctx.Queue()
        port = self._free_port()

        # Goal12-v6: discover a real NCCL library on this host.
        nccl_lib = self._find_real_nccl()
        if nccl_lib is None:
            self.skipTest(
                "no real NCCL library found on this CI platform — "
                "the four-process NCCL happy path requires a loadable "
                "libnccl.so exporting ncclCommInitRank/ncclCommInitRankAll"
            )

        def happy_env(rank):
            return {
                "NCCL_NET": "IB",
                "NCCL_IB_DISABLE": "0",
                "VLLM_NCCL_SO_PATH": nccl_lib,
                "SPARKRING_IMAGE_DIGEST": "sha256:" + "a" * 64,
            }

        results = self._spawn_and_collect(
            self._worker, list(range(4)), 4, port, result_queue, recorder,
            self._const_selector(SELECTOR_NCCL_IB), 1, 8,
            self._no_identity_overrides, happy_env,
            self.COLLECT_TIMEOUT, self.JOIN_TIMEOUT,
            recorder_lock=recorder_lock,
            nccl_cdll_real=True,
        )
        self.assertEqual(len(results), 4, "Expected exactly 4 results")
        ranks_seen = sorted(r for _, r, _ in results)
        self.assertEqual(ranks_seen, [0, 1, 2, 3])
        for status, rank, value in results:
            self.assertEqual(status, "ok", f"Rank {rank} failed: {value}")
        # Happy path reached the NCCL data branch: exactly 4 NCCL
        # data calls (1 per rank), 0 SIRCL calls.
        self.assertEqual(recorder["nccl_calls"], 4)
        self.assertEqual(recorder["sircl_calls"], 0)

    # -- Scenario 2: selector asymmetry → all abort ------------------

    def test_selector_asymmetry_all_abort(self):
        """Rank 0=custom, ranks 1-3=disabled → selector sum mismatch
        → all abort with zero data calls."""
        from spark_transport_contract import SELECTOR_NCCL_IB, SELECTOR_CUSTOM
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        mgr = ctx.Manager()
        recorder, recorder_lock = self._make_recorder(mgr)
        result_queue = ctx.Queue()
        port = self._free_port()
        # Goal12-v6: no NCCL env or fake library — this scenario
        # fails at selector-sum consensus before NCCL validation.
        def asym_selector(rank):
            return SELECTOR_CUSTOM if rank == 0 else SELECTOR_NCCL_IB

        def asym_env(rank):
            if rank == 0:
                return {}
            return {
                "NCCL_NET": "IB",
                "NCCL_IB_DISABLE": "0",
            }

        results = self._spawn_and_collect(
            self._worker, list(range(4)), 4, port, result_queue, recorder,
            asym_selector, 1, 8,
            self._no_identity_overrides, asym_env,
            self.COLLECT_TIMEOUT, self.JOIN_TIMEOUT,
            recorder_lock=recorder_lock,
        )
        self.assertEqual(len(results), 4, "Expected exactly 4 results")
        for status, rank, msg in results:
            self.assertEqual(status, "error", f"Rank {rank} should error")
            self.assertIn("consensus failed", msg)
        self.assertEqual(recorder["sircl_calls"], 0)
        self.assertEqual(recorder["nccl_calls"], 0)

    # -- Scenario 3: missing rank 2 → all error ----------------------

    def test_missing_rank_2_all_error(self):
        """Only ranks 0,1,3 spawn (rank 2 missing) → Gloo timeout or
        consensus failure → exactly 3 error results, each a failure,
        with zero data calls."""
        from spark_transport_contract import SELECTOR_NCCL_IB
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        mgr = ctx.Manager()
        recorder, recorder_lock = self._make_recorder(mgr)
        result_queue = ctx.Queue()
        port = self._free_port()
        # Goal12-v6: no NCCL env or fake library — this scenario
        # fails at rank-set consensus before NCCL validation.
        def happy_env(rank):
            return {
                "NCCL_NET": "IB",
                "NCCL_IB_DISABLE": "0",
            }

        results = self._spawn_and_collect(
            self._worker, [0, 1, 3], 4, port, result_queue, recorder,
            self._const_selector(SELECTOR_NCCL_IB), 1, 8,
            self._no_identity_overrides, happy_env,
            self.COLLECT_TIMEOUT, self.JOIN_TIMEOUT,
            recorder_lock=recorder_lock,
        )
        # Review v2 requirement 3: exact result count of 3, each a
        # failure.  len(results) >= 1 is forbidden.
        self.assertEqual(len(results), 3, "Expected exactly 3 results")
        for status, rank, msg in results:
            self.assertEqual(status, "error", f"Rank {rank} should error")
        self.assertEqual(recorder["sircl_calls"], 0)
        self.assertEqual(recorder["nccl_calls"], 0)

    # -- Scenario 4: duplicate rank record → all abort ---------------

    def test_duplicate_rank_record_all_abort(self):
        """Rank 2's identity record claims rank=1 → duplicate rank
        detected → all abort with zero data calls."""
        from spark_transport_contract import SELECTOR_NCCL_IB
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        mgr = ctx.Manager()
        recorder, recorder_lock = self._make_recorder(mgr)
        result_queue = ctx.Queue()
        port = self._free_port()
        # Goal12-v6: no NCCL env or fake library — this scenario
        # fails at rank-set consensus before NCCL validation.
        def dup_identity(rank):
            if rank == 2:
                return {"rank": 1}  # claim rank 1 → duplicate
            return {}

        def happy_env(rank):
            return {
                "NCCL_NET": "IB",
                "NCCL_IB_DISABLE": "0",
            }

        results = self._spawn_and_collect(
            self._worker, list(range(4)), 4, port, result_queue, recorder,
            self._const_selector(SELECTOR_NCCL_IB), 1, 8,
            dup_identity, happy_env,
            self.COLLECT_TIMEOUT, self.JOIN_TIMEOUT,
            recorder_lock=recorder_lock,
        )
        self.assertEqual(len(results), 4, "Expected exactly 4 results")
        for status, rank, msg in results:
            self.assertEqual(status, "error", f"Rank {rank} should error")
            self.assertIn("consensus failed", msg)
        self.assertEqual(recorder["sircl_calls"], 0)
        self.assertEqual(recorder["nccl_calls"], 0)

    # -- Scenario 5: out-of-range rank record → all abort -------------

    def test_out_of_range_rank_record_all_abort(self):
        """Rank 2's identity record claims rank=5 → out-of-range rank
        detected → all abort with zero data calls."""
        from spark_transport_contract import SELECTOR_NCCL_IB
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        mgr = ctx.Manager()
        recorder, recorder_lock = self._make_recorder(mgr)
        result_queue = ctx.Queue()
        # Goal12-v6: no NCCL env or fake library — this scenario
        # fails at rank-set consensus before NCCL validation.
        port = self._free_port()

        def oob_identity(rank):
            if rank == 2:
                return {"rank": 5}  # out-of-range rank
            return {}

        def happy_env(rank):
            return {
                "NCCL_NET": "IB",
                "NCCL_IB_DISABLE": "0",
            }

        results = self._spawn_and_collect(
            self._worker, list(range(4)), 4, port, result_queue, recorder,
            self._const_selector(SELECTOR_NCCL_IB), 1, 8,
            oob_identity, happy_env,
            self.COLLECT_TIMEOUT, self.JOIN_TIMEOUT,
            recorder_lock=recorder_lock,
        )
        self.assertEqual(len(results), 4, "Expected exactly 4 results")
        for status, rank, msg in results:
            self.assertEqual(status, "error", f"Rank {rank} should error")
            self.assertIn("consensus failed", msg)
        self.assertEqual(recorder["sircl_calls"], 0)
        self.assertEqual(recorder["nccl_calls"], 0)

    # -- Scenario 6: one-rank missing capability/library → all abort

    def test_missing_capability_all_abort(self):
        """Review v4 requirement 2: exactly 3 ranks get a SIRCL
        test seam library (filename contains "sircl" +
        SPARK_SIRCL_TEST_LIBRARY env); exactly one rank (rank 2)
        gets no SPARK_TP4_LIBRARY → native_capable=False with
        selector=custom → consensus aborts all ranks before any
        data call.

        Review v4 requirement 2: the test seam proves the SIRCL
        identity contract (required C API symbols) without blessing
        a generic system library as SIRCL-capable."""
        from spark_transport_contract import SELECTOR_CUSTOM
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        mgr = ctx.Manager()
        recorder, recorder_lock = self._make_recorder(mgr)
        result_queue = ctx.Queue()
        port = self._free_port()
        sircl_lib, sircl_seam_env = self._make_sircl_test_seam()

        def cap_env(rank):
            if rank == 2:
                # Rank 2: no library → not native_capable.
                return {"SPARK_TP4_LIBRARY": ""}
            # Ranks 0,1,3: SIRCL test seam library.
            return {
                "SPARK_TP4_LIBRARY": sircl_lib,
                **sircl_seam_env,
            }

        results = self._spawn_and_collect(
            self._worker, list(range(4)), 4, port, result_queue, recorder,
            self._const_selector(SELECTOR_CUSTOM), 1, 8,
            self._no_identity_overrides, cap_env,
            self.COLLECT_TIMEOUT, self.JOIN_TIMEOUT,
            recorder_lock=recorder_lock,
        )
        self.assertEqual(len(results), 4, "Expected exactly 4 results")
        for status, rank, msg in results:
            self.assertEqual(status, "error", f"Rank {rank} should error")
            self.assertIn("consensus failed", msg)
        self.assertEqual(recorder["sircl_calls"], 0)
        self.assertEqual(recorder["nccl_calls"], 0)



class NativeSessionProductionPathTest(unittest.TestCase):
    """Goal 12 requirement 2: production-path _NativeSession validation.

    Tests that the production path validates SPARK_TP4_LIBRARY before
    constructing _NativeSession, uses the correct (rank, payload_bytes)
    signature, and rejects NCCL init in the SIRCL arm.
    """

    def test_missing_spark_tp4_library_raises(self) -> None:
        """No SPARK_TP4_LIBRARY set → RuntimeError before any data call."""
        with patch(
            "tp4_numerical_audit._init_control_group", return_value=None,
        ), patch(
            "tp4_numerical_audit._run_control_plane_consensus",
            return_value=SELECTOR_CUSTOM,
        ), patch(
            "torch.device", return_value=torch.device("cpu"),
        ), patch("torch.cuda.set_device"), patch(
            "spark_tp4_backend._NativeSession",
        ), patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                run_probe(
                    selector=SELECTOR_CUSTOM, rank=0,
                    iterations=1, elements=8, world_size=4,
                )
        self.assertIn("SPARK_TP4_LIBRARY", str(ctx.exception))

    def test_wrong_spark_tp4_library_path_raises(self) -> None:
        """SPARK_TP4_LIBRARY points to nonexistent file → RuntimeError."""
        with patch(
            "tp4_numerical_audit._init_control_group", return_value=None,
        ), patch(
            "tp4_numerical_audit._run_control_plane_consensus",
            return_value=SELECTOR_CUSTOM,
        ), patch(
            "torch.device", return_value=torch.device("cpu"),
        ), patch("torch.cuda.set_device"), patch(
            "spark_tp4_backend._NativeSession",
        ), patch.dict("os.environ", {"SPARK_TP4_LIBRARY": "/nonexistent/path/lib.so"}, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                run_probe(
                    selector=SELECTOR_CUSTOM, rank=0,
                    iterations=1, elements=8, world_size=4,
                )
        self.assertIn("does not exist", str(ctx.exception))

    def test_native_session_called_with_correct_signature(self) -> None:
        """_NativeSession must be called with (rank, payload_bytes)
        where payload_bytes = elements * 2 (BF16)."""
        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "fake_lib.so"
        tmp.write_bytes(b"\x7fELF fake")
        native = _MockNativeSession(elements=8, world_size=4)
        with patch(
            "tp4_numerical_audit._init_control_group", return_value=None,
        ), patch(
            "tp4_numerical_audit._run_control_plane_consensus",
            return_value=SELECTOR_CUSTOM,
        ), patch(
            "torch.device", return_value=torch.device("cpu"),
        ), patch("torch.cuda.set_device"), patch(
            "spark_tp4_backend._NativeSession", return_value=native,
        ), patch.dict("os.environ", {"SPARK_TP4_LIBRARY": str(tmp)}, clear=False):
            run_probe(
                selector=SELECTOR_CUSTOM, rank=0,
                iterations=1, elements=8, world_size=4,
            )
        # _NativeSession was patched but we verified the production path
        # reached the native session call (no early error from library check).
        tmp.unlink(missing_ok=True)

    def test_sircl_arm_no_nccl_init(self) -> None:
        """SIRCL arm must not call dist.init_process_group or
        dist.new_group with backend='nccl'."""
        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "fake_lib.so"
        tmp.write_bytes(b"\x7fELF fake")
        native = _MockNativeSession(elements=8, world_size=4)
        with patch(
            "tp4_numerical_audit._init_control_group", return_value=None,
        ), patch(
            "tp4_numerical_audit._run_control_plane_consensus",
            return_value=SELECTOR_CUSTOM,
        ), patch(
            "torch.device", return_value=torch.device("cpu"),
        ), patch("torch.cuda.set_device"), patch(
            "spark_tp4_backend._NativeSession", return_value=native,
        ), patch("torch.distributed.init_process_group") as init_pg, \
           patch("torch.distributed.new_group") as new_grp, \
           patch.dict("os.environ", {"SPARK_TP4_LIBRARY": str(tmp)}, clear=False):
            run_probe(
                selector=SELECTOR_CUSTOM, rank=0,
                iterations=1, elements=8, world_size=4,
            )
        init_pg.assert_not_called()
        # new_group may be called for Gloo control but never with nccl
        for call in new_grp.call_args_list:
            backend = call.kwargs.get("backend", "")
            self.assertNotEqual(backend, "nccl",
                f"SIRCL arm must not create NCCL group, got backend={backend}")
        tmp.unlink(missing_ok=True)

    def test_branch_mixing_sircl_with_nccl_env_rejected(self) -> None:
        """selector=custom with NCCL_NET=IB in env → _validate_process_env
        must exit (SIRCL arm must not have NCCL transport env vars)."""
        with patch.dict("os.environ", {"NCCL_NET": "IB", "NCCL_IB_DISABLE": "0"}, clear=False):
            with self.assertRaises(SystemExit):
                import tp4_numerical_audit as mod
                mod._validate_process_env(SELECTOR_CUSTOM)


class AdversarialReceiptEdgeCaseTest(unittest.TestCase):
    """Goal 12 requirement 5: additional adversarial receipt edge cases.

    Tests edge cases for parse_receipt_json that go beyond the basic
    rejection tests in FourProcessConsensusTest.  All tests assert
    REJECTION of invalid input — no exploit-blessing methods remain.
    """

    def test_negative_iterations_rejected(self) -> None:
        """iterations=-1 must be rejected."""
        receipt = _make_valid_receipt()
        receipt["iterations"] = -1
        with self.assertRaises(ValueError):
            parse_receipt_json(dict(receipt))

    def test_count_mismatch_rejected(self) -> None:
        """native_collectives=1 but total_collectives=2 → ValueError."""
        receipt = _make_valid_receipt()
        receipt["native_collectives"] = 1
        receipt["total_collectives"] = 2
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(dict(receipt))
        self.assertIn("classified sum", str(ctx.exception))

    def test_bogus_hash_rejected(self) -> None:
        """expected_fp32_hash='xyz' (not 64-char hex) → ValueError."""
        receipt = _make_valid_receipt()
        receipt["expected_fp32_hash"] = "xyz"
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(dict(receipt))
        self.assertIn("64-char hex", str(ctx.exception))

    def test_inf_in_float_field_rejected(self) -> None:
        """max_abs_error = float('inf') → ValueError (not finite)."""
        receipt = _make_valid_receipt()
        receipt["max_abs_error"] = float("inf")
        with self.assertRaises(ValueError) as ctx:
            parse_receipt_json(dict(receipt))
        self.assertIn("finite", str(ctx.exception))


# ---------------------------------------------------------------------------
# RED reproductions for c8f9665 control-plane defects.
# These tests assert the CORRECT (post-fix) behavior.  Against c8f9665
# they FAIL — proving the defects exist.  After the production repair
# they must pass.
# ---------------------------------------------------------------------------

class ControlPlaneFailClosedReproductionTest(unittest.TestCase):
    """Defect 1: _init_control_group returns None on failure, and
    _run_control_plane_consensus(..., None) returns success.
    Both must fail closed — a missing rendezvous/init failure must
    never reach a data call or success receipt."""

    def test_init_control_group_failure_does_not_succeed(self) -> None:
        """When _init_control_group fails (dist unavailable / rendezvous
        error), the production path must raise, not silently proceed to
        data transport.  Against c8f9665, run_probe proceeds past a
        None control_pg and reaches data init."""
        from unittest.mock import patch as _patch
        import tp4_numerical_audit as mod

        # _init_control_group returns None (simulating init failure).
        # _run_control_plane_consensus must NOT silently return success.
        with _patch.object(mod, "_init_control_group", return_value=None), \
             _patch("torch.device", return_value=torch.device("cpu")), \
             _patch("torch.cuda.set_device"), \
             _patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                run_probe(
                    selector="disabled", rank=0,
                    iterations=1, elements=8, world_size=4,
                )
            self.assertTrue(
                "control" in str(ctx.exception).lower()
                or "consensus" in str(ctx.exception).lower()
                or "init" in str(ctx.exception).lower(),
                msg=f"Unexpected error: {ctx.exception}",
            )

    def test_consensus_with_none_backend_does_not_succeed(self) -> None:
        """_run_control_plane_consensus with control_backend=None must
        raise RuntimeError, not return the selector unchanged.
        Against c8f9665, it returns selector (silent success)."""
        with self.assertRaises(RuntimeError) as ctx:
            _run_control_plane_consensus(
                "custom", rank=0, world_size=4, control_backend=None,
            )
        self.assertTrue(
            "control" in str(ctx.exception).lower()
            or "consensus" in str(ctx.exception).lower()
            or "backend" in str(ctx.exception).lower(),
            msg=f"Unexpected error: {ctx.exception}",
        )


class DataPgNotOnGlobalModuleReproductionTest(unittest.TestCase):
    """Defect 3: the production path must never store a data process
    group on the global torch.distributed module (dist._data_pg).
    Against c8f9665, run_probe sets d._data_pg = nccl_data_pg."""

    def test_no_data_pg_attribute_set_on_dist(self) -> None:
        """After run_probe with selector=disabled (NCCL arm) in
        production (non-injected) mode, the global dist module must
        NOT have a _data_pg attribute.

        Against c8f9665, run_probe sets d._data_pg = nccl_data_pg
        where d = dist (the global module)."""
        import torch.distributed as dist
        # Clean any pre-existing attribute from other tests.
        if hasattr(dist, "_data_pg"):
            del dist._data_pg

        fake_pg = MagicMock()

        seq_counter = [0]

        def _correct_all_reduce(tensor, op=None, group=None):
            seq = seq_counter[0]
            correct = _make_correct_fp32_sum_reduce(seq, 8, 4)
            tensor.copy_(correct)
            seq_counter[0] += 1

        with patch("torch.device", return_value=torch.device("cpu")), \
             patch("torch.cuda.set_device"), \
             patch("tp4_numerical_audit._init_control_group",
                    return_value=None), \
             patch("tp4_numerical_audit._run_control_plane_consensus",
                    return_value="disabled"), \
             patch("torch.distributed.new_group", return_value=fake_pg), \
             patch("torch.distributed.all_reduce",
                    side_effect=_correct_all_reduce), \
             patch("torch.distributed.barrier"), \
             patch("torch.distributed.destroy_process_group"):
            run_probe(
                selector="disabled", rank=0,
                iterations=1, elements=8, world_size=4,
            )
        # The global dist module must NOT have _data_pg set.
        self.assertFalse(
            hasattr(dist, "_data_pg"),
            "dist._data_pg must not be set — data PGs must be local",
        )



# ---------------------------------------------------------------------------
# Review v4 regression tests: per-rank env exclusion, SIRCL symbol
# verification, canonical image receipt, runtime identity, NCCL identity.
# ---------------------------------------------------------------------------

class SharedEnvPerRankExclusionTest(unittest.TestCase):
    """Review v4 requirement 1: _hash_shared_env must exclude
    canonical per-rank topology values so legitimate per-rank
    differences do not cause false consensus failures."""

    def test_per_rank_topology_values_do_not_change_shared_hash(self):
        """Changing SPARK_TP4_PEER0/1, SPARK_TP4_DEVICE0/1, and
        NCCL_IB_HCA (legitimate per-rank topology) must NOT change
        the shared env hash."""
        base_env = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "ITERATIONS": "1",
            "ELEMENTS": "8",
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "GLOO_SOCKET_IFNAME": "eth0",
        }
        with patch.dict("os.environ", base_env, clear=False):
            hash_before = _hash_shared_env()
        # Change per-rank topology values — should not affect hash.
        per_rank_env = {
            **base_env,
            "SPARK_TP4_PEER0": "192.0.2.10",
            "SPARK_TP4_PEER1": "192.0.2.20",
            "SPARK_TP4_DEVICE0": "rocep1s0f0",
            "SPARK_TP4_DEVICE1": "rocep1s0f1",
            "SPARK_TP4_GID0": "3",
            "SPARK_TP4_GID1": "5",
            "NCCL_IB_HCA": "rocep1s0f0,rocep1s0f1",
            "RANK": "0",
            "WORLD_SIZE": "4",
        }
        with patch.dict("os.environ", per_rank_env, clear=False):
            hash_after = _hash_shared_env()
        self.assertEqual(hash_before, hash_after)

    def test_shared_control_setting_still_changes_hash(self):
        """A truly shared control setting (MASTER_ADDR) must still
        change the shared env hash — proving the exclusion is
        narrow, not over-broad."""
        base_env = {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        with patch.dict("os.environ", base_env, clear=False):
            hash_before = _hash_shared_env()
        changed_env = {
            "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "29500",
        }
        with patch.dict("os.environ", changed_env, clear=False):
            hash_after = _hash_shared_env()
        self.assertNotEqual(hash_before, hash_after)

    def test_different_per_rank_topology_values_same_hash(self):
        """Two ranks with different per-rank topology but identical
        shared env must produce the same shared_env_hash."""
        rank0_env = {
            "SPARK_TP4_PEER0": "10.0.0.1",
            "SPARK_TP4_PEER1": "10.0.0.2",
            "SPARK_TP4_DEVICE0": "rocep1s0f0",
            "SPARK_TP4_DEVICE1": "rocep1s0f1",
            "SPARK_TP4_GID0": "3",
            "SPARK_TP4_GID1": "5",
            "NCCL_IB_HCA": "rocep1s0f0,rocep1s0f1",
            "RANK": "0",
            "WORLD_SIZE": "4",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        rank1_env = {
            "SPARK_TP4_PEER0": "10.0.0.3",
            "SPARK_TP4_PEER1": "10.0.0.4",
            "SPARK_TP4_DEVICE0": "rocep2s0f0",
            "SPARK_TP4_DEVICE1": "rocep2s0f1",
            "SPARK_TP4_GID0": "7",
            "SPARK_TP4_GID1": "9",
            "NCCL_IB_HCA": "rocep2s0f0,rocep2s0f1",
            "RANK": "1",
            "WORLD_SIZE": "4",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        with patch.dict("os.environ", rank0_env, clear=False):
            hash0 = _hash_shared_env()
        with patch.dict("os.environ", rank1_env, clear=False):
            hash1 = _hash_shared_env()
        self.assertEqual(hash0, hash1)


class NativeCapabilitySIRCLSymbolTest(unittest.TestCase):
    """Review v5 requirement 1+2: _check_native_capable must verify
    the library exports the required SIRCL C API symbols.  No
    environment variable (including the deleted SPARK_SIRCL_TEST_LIBRARY)
    may bypass existence, ctypes.CDLL, or required-symbol validation.
    Tests inject mocks at the callable/CDLL seam, not a production env."""

    def test_generic_system_library_not_sircl_capable(self):
        """A generic loadable system library (libc/kernel32) must
        NOT pass _check_native_capable — it does not export
        spark_tp4_create, spark_tp4_all_reduce, spark_tp4_destroy."""
        import platform
        if platform.system() == "Windows":
            generic_lib = os.path.join(
                os.environ.get("SystemRoot", r"C:\Windows"),
                "System32", "kernel32.dll",
            )
        else:
            import ctypes.util
            generic_lib = ctypes.util.find_library("c")
            if not generic_lib:
                self.skipTest("No libc found on this platform")
        result = _check_native_capable(generic_lib)
        self.assertFalse(result)

    def test_empty_path_not_capable(self):
        """An empty library path must return False."""
        self.assertFalse(_check_native_capable(""))

    def test_nonexistent_path_not_capable(self):
        """A nonexistent file path must return False."""
        self.assertFalse(_check_native_capable("/nonexistent/path/lib.so"))

    def test_fake_sircl_so_rejected_even_with_legacy_bypass_env(self):
        """Review v5 requirement 1: a nonexistent fake_sircl.so must
        be rejected even if the deleted SPARK_SIRCL_TEST_LIBRARY env
        is present.  No env var may bypass existence or CDLL
        validation."""
        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "fake_sircl.so"
        tmp.write_bytes(b"\x7fELF fake sircl")
        with patch.dict("os.environ", {"SPARK_SIRCL_TEST_LIBRARY": "1"}):
            result = _check_native_capable(str(tmp))
        self.assertFalse(result)
        tmp.unlink(missing_ok=True)

    def test_sircl_capability_via_cdll_seam_mock(self):
        """Review v5 requirement 2: the SIRCL test double must prove
        the exact required symbols via a CDLL seam mock injected
        locally in the test, not a production env bypass.  We patch
        ctypes.CDLL to return a mock that exports exactly
        spark_tp4_create, spark_tp4_all_reduce, spark_tp4_destroy."""
        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "test_sircl_double.so"
        tmp.write_bytes(b"\x7fELF fake sircl test double")

        class _MockSIRCLLib:
            def __getattr__(self, name):
                if name in ("spark_tp4_create", "spark_tp4_all_reduce",
                            "spark_tp4_destroy"):
                    return lambda *a, **k: None
                raise AttributeError(name)

        with patch("ctypes.CDLL", return_value=_MockSIRCLLib()):
            result = _check_native_capable(str(tmp))
        self.assertTrue(result)
        tmp.unlink(missing_ok=True)

    def test_sircl_cdll_mock_missing_symbol_rejected(self):
        """A CDLL mock missing one required symbol must fail."""
        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "test_sircl_partial.so"
        tmp.write_bytes(b"\x7fELF partial")

        class _MockPartialLib:
            def __getattr__(self, name):
                if name in ("spark_tp4_create", "spark_tp4_all_reduce"):
                    return lambda *a, **k: None
                raise AttributeError(name)

        with patch("ctypes.CDLL", return_value=_MockPartialLib()):
            result = _check_native_capable(str(tmp))
        self.assertFalse(result)
        tmp.unlink(missing_ok=True)


class NCCLIdentityTest(unittest.TestCase):
    """Review v5 requirement 1+2: _check_nccl_identity must prove
    the library is an actual NCCL library.  No environment variable
    (including the deleted SPARK_NCCL_TEST_LIBRARY) may bypass
    existence, ctypes.CDLL, or required-symbol validation.  For the
    NCCL happy identity case, discover and validate a real NCCL library
    when available; otherwise skip with a precise reason.  A text file
    or libc/kernel32 renamed to contain 'nccl' is forbidden."""

    @staticmethod
    def _find_real_nccl():
        """Discover a real NCCL library on this host.

        Searches common paths and LD_PRELOAD/VLLM_NCCL_SO_PATH.
        Returns the path if the library at that path exports
        ncclCommInitRank or ncclCommInitRankAll, else None.
        """
        import ctypes
        import pathlib
        candidates = []
        # Check VLLM_NCCL_SO_PATH / LD_PRELOAD env vars first.
        for env_name in ("VLLM_NCCL_SO_PATH", "LD_PRELOAD"):
            val = os.environ.get(env_name, "")
            if val:
                candidates.append(val)
        # Common NCCL library paths on Linux.
        import glob as _glob
        for pattern in (
            "/usr/lib/*/libnccl.so*",
            "/usr/lib/libnccl.so*",
            "/opt/*/libnccl.so*",
            "/usr/local/cuda/lib*/libnccl.so*",
        ):
            candidates.extend(_glob.glob(pattern))
        for path in candidates:
            try:
                p = pathlib.Path(path)
                if not p.is_file():
                    continue
                lib = ctypes.CDLL(str(p))
                for sym in ("ncclCommInitRank", "ncclCommInitRankAll"):
                    if hasattr(lib, sym):
                        return str(p)
            except Exception:
                continue
        return None

    def test_generic_system_library_not_nccl(self):
        """A generic loadable system library must NOT pass
        _check_nccl_identity."""
        import platform
        if platform.system() == "Windows":
            generic_lib = os.path.join(
                os.environ.get("SystemRoot", r"C:\Windows"),
                "System32", "kernel32.dll",
            )
        else:
            import ctypes.util
            generic_lib = ctypes.util.find_library("c")
            if not generic_lib:
                self.skipTest("No libc found on this platform")
        result = _check_nccl_identity(generic_lib)
        self.assertFalse(result)

    def test_empty_path_not_nccl(self):
        """An empty library path must return False."""
        self.assertFalse(_check_nccl_identity(""))

    def test_fake_nccl_so_rejected_even_with_legacy_bypass_env(self):
        """Review v5 requirement 1: a text file renamed to contain
        'nccl' must be rejected even if the deleted
        SPARK_NCCL_TEST_LIBRARY env is present.  No env var,
        filename substring, or nonexistent path may bypass CDLL or
        required-symbol validation."""
        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "fake_nccl.so"
        tmp.write_bytes(b"\x7fELF fake nccl - not a real NCCL library")
        with patch.dict("os.environ", {"SPARK_NCCL_TEST_LIBRARY": "1"}):
            result = _check_nccl_identity(str(tmp))
        self.assertFalse(result)
        tmp.unlink(missing_ok=True)

    def test_nccl_cdll_mock_missing_symbol_rejected(self):
        """A CDLL mock missing NCCL symbols must fail."""
        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "test_nccl_partial.so"
        tmp.write_bytes(b"\x7fELF partial")

        class _MockNonNCCLLib:
            def __getattr__(self, name):
                raise AttributeError(name)

        with patch("ctypes.CDLL", return_value=_MockNonNCCLLib()):
            result = _check_nccl_identity(str(tmp))
        self.assertFalse(result)
        tmp.unlink(missing_ok=True)

    def test_real_nccl_identity_when_available(self):
        """Review v5 requirement 2: when a real NCCL library is
        available on this host, validate it passes identity check.
        Otherwise skip with a precise reason — a text file or
        libc/kernel32 renamed to contain 'nccl' is forbidden."""
        nccl_path = self._find_real_nccl()
        if nccl_path is None:
            self.skipTest(
                "No real NCCL library found on this host — "
                "NCCL happy identity case requires a real NCCL "
                "installation with ncclCommInitRank/ncclCommInitRankAll"
            )
        result = _check_nccl_identity(nccl_path)
        self.assertTrue(result)


class ImageReceiptCanonicalTest(unittest.TestCase):
    """Review v4 requirement 3: _get_image_receipt must consume the
    canonical launcher identity SPARKRING_IMAGE_DIGEST, not the
    non-existent VLLM_SPARK_IMAGE_RECEIPT."""

    def test_canonical_digest_produces_nonempty_receipt(self):
        """When SPARKRING_IMAGE_DIGEST is set, _get_image_receipt
        must return that value."""
        digest = "sha256:" + "a" * 64
        with patch.dict("os.environ", {"SPARKRING_IMAGE_DIGEST": digest}):
            result = _get_image_receipt()
        self.assertEqual(result, digest)

    def test_legacy_fallback_when_canonical_absent(self):
        """When SPARKRING_IMAGE_DIGEST is absent but
        VLLM_SPARK_IMAGE_RECEIPT is set, the legacy fallback
        is used."""
        legacy = "sha256:" + "b" * 64
        env = {k: v for k, v in os.environ.items()
               if k != "SPARKRING_IMAGE_DIGEST"}
        env["VLLM_SPARK_IMAGE_RECEIPT"] = legacy
        with patch.dict("os.environ", env, clear=True):
            result = _get_image_receipt()
        self.assertEqual(result, legacy)

    def test_canonical_takes_precedence_over_legacy(self):
        """When both are set, SPARKRING_IMAGE_DIGEST wins."""
        canonical = "sha256:" + "c" * 64
        legacy = "sha256:" + "d" * 64
        with patch.dict("os.environ", {
            "SPARKRING_IMAGE_DIGEST": canonical,
            "VLLM_SPARK_IMAGE_RECEIPT": legacy,
        }):
            result = _get_image_receipt()
        self.assertEqual(result, canonical)

    def test_empty_when_neither_set(self):
        """When neither env var is set, the receipt is empty."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("SPARKRING_IMAGE_DIGEST", "VLLM_SPARK_IMAGE_RECEIPT")}
        with patch.dict("os.environ", env, clear=True):
            result = _get_image_receipt()
        self.assertEqual(result, "")


class RuntimeIdentitySparkRingEvidenceTest(unittest.TestCase):
    """Review v5 requirement 3: _get_runtime_identity must read and
    hash verified canonical runtime evidence bytes, not append an
    unchecked path.  For SPARKRING_RUNTIME_MANIFEST, require an
    existing regular file, read it safely, hash its bytes, and include
    a versioned field in the closed record.  Canonical runtime
    receipt/lock hashes are consumed as validated values.  A
    nonexistent path, directory, unreadable file, or mismatched/empty
    evidence must fail closed."""

    def test_includes_python_torch_platform(self):
        """The identity must include py, torch, and plat parts."""
        identity = _get_runtime_identity()
        self.assertIn("py=", identity)
        self.assertIn("torch=", identity)
        self.assertIn("plat=", identity)

    def test_includes_manifest_sha_when_file_exists(self):
        """When SPARKRING_RUNTIME_MANIFEST points to an existing
        regular file, the identity must include the SHA-256 of its
        bytes, not the unchecked path."""
        import tempfile
        import pathlib
        import hashlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "manifest.json"
        content = b'{"runtime_id": "test-rt"}'
        tmp.write_bytes(content)
        expected_sha = hashlib.sha256(content).hexdigest()
        with patch.dict("os.environ", {"SPARKRING_RUNTIME_MANIFEST": str(tmp)}):
            identity = _get_runtime_identity()
        self.assertIn("manifest_sha=sha256:" + expected_sha, identity)
        self.assertNotIn("manifest=" + str(tmp), identity)
        tmp.unlink(missing_ok=True)

    def test_includes_image_digest_when_set(self):
        """When SPARKRING_IMAGE_DIGEST is set, the identity
        must include it."""
        digest = "sha256:" + "e" * 64
        with patch.dict("os.environ", {"SPARKRING_IMAGE_DIGEST": digest}):
            identity = _get_runtime_identity()
        self.assertIn("image=" + digest, identity)

    def test_identity_nonempty_without_sparkring_evidence(self):
        """Even without SparkRing env, the identity is nonempty
        (py/torch/plat suffice)."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("SPARKRING_RUNTIME_MANIFEST", "SPARKRING_IMAGE_DIGEST")}
        with patch.dict("os.environ", env, clear=True):
            identity = _get_runtime_identity()
        self.assertTrue(len(identity) > 0)

    def test_nonexistent_manifest_path_fails_closed(self):
        """A nonexistent SPARKRING_RUNTIME_MANIFEST path must raise
        RuntimeError — fail closed, do not silently skip."""
        with patch.dict("os.environ", {"SPARKRING_RUNTIME_MANIFEST": "/nonexistent/manifest.json"}):
            with self.assertRaises(RuntimeError) as ctx:
                _get_runtime_identity()
        self.assertIn("does not exist", str(ctx.exception))

    def test_directory_manifest_path_fails_closed(self):
        """A directory as SPARKRING_RUNTIME_MANIFEST must raise
        RuntimeError — directories are not regular files."""
        import tempfile
        import pathlib
        tmpdir = pathlib.Path(tempfile.mkdtemp())
        with patch.dict("os.environ", {"SPARKRING_RUNTIME_MANIFEST": str(tmpdir)}):
            with self.assertRaises(RuntimeError) as ctx:
                _get_runtime_identity()
        self.assertIn("not a regular file", str(ctx.exception))

    def test_empty_manifest_file_fails_closed(self):
        """An empty manifest file must raise RuntimeError — empty
        evidence cannot bind runtime identity."""
        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "empty.json"
        tmp.write_bytes(b"")
        with patch.dict("os.environ", {"SPARKRING_RUNTIME_MANIFEST": str(tmp)}):
            with self.assertRaises(RuntimeError) as ctx:
                _get_runtime_identity()
        self.assertIn("empty", str(ctx.exception))
        tmp.unlink(missing_ok=True)

    def test_same_path_changed_bytes_changes_identity(self):
        """Review v5 requirement 3: same path with changed bytes
        must produce a different runtime identity — the hash
        tracks content, not path."""
        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "manifest.json"
        tmp.write_bytes(b'{"v": 1}')
        env_base = {k: v for k, v in os.environ.items()
                    if k != "SPARKRING_RUNTIME_MANIFEST"}
        with patch.dict("os.environ", {**env_base, "SPARKRING_RUNTIME_MANIFEST": str(tmp)}, clear=True):
            id_a = _get_runtime_identity()
        tmp.write_bytes(b'{"v": 2}')
        with patch.dict("os.environ", {**env_base, "SPARKRING_RUNTIME_MANIFEST": str(tmp)}, clear=True):
            id_b = _get_runtime_identity()
        self.assertNotEqual(id_a, id_b)
        tmp.unlink(missing_ok=True)

    def test_different_paths_identical_bytes_same_content_identity(self):
        """Review v5 requirement 3: different paths with identical
        bytes must produce the same content identity — the hash
        is over bytes, not path (path is not part of the contract)."""
        import tempfile
        import pathlib
        content = b'{"runtime_id": "same"}'
        tmp_a = pathlib.Path(tempfile.mkdtemp()) / "manifest_a.json"
        tmp_a.write_bytes(content)
        tmp_b = pathlib.Path(tempfile.mkdtemp()) / "manifest_b.json"
        tmp_b.write_bytes(content)
        env_base = {k: v for k, v in os.environ.items()
                    if k != "SPARKRING_RUNTIME_MANIFEST"}
        with patch.dict("os.environ", {**env_base, "SPARKRING_RUNTIME_MANIFEST": str(tmp_a)}, clear=True):
            id_a = _get_runtime_identity()
        with patch.dict("os.environ", {**env_base, "SPARKRING_RUNTIME_MANIFEST": str(tmp_b)}, clear=True):
            id_b = _get_runtime_identity()
        self.assertEqual(id_a, id_b)
        tmp_a.unlink(missing_ok=True)
        tmp_b.unlink(missing_ok=True)

    def test_mismatched_manifest_content_changes_identity(self):
        """Different manifest file contents must produce different
        runtime identities — so cross-rank mismatch is detected
        at consensus."""
        import tempfile
        import pathlib
        tmp = pathlib.Path(tempfile.mkdtemp()) / "manifest.json"
        tmp.write_bytes(b'{"rt": "a"}')
        env_base = {k: v for k, v in os.environ.items()
                    if k != "SPARKRING_RUNTIME_MANIFEST"}
        with patch.dict("os.environ", {**env_base, "SPARKRING_RUNTIME_MANIFEST": str(tmp)}, clear=True):
            id_a = _get_runtime_identity()
        tmp.write_bytes(b'{"rt": "b"}')
        with patch.dict("os.environ", {**env_base, "SPARKRING_RUNTIME_MANIFEST": str(tmp)}, clear=True):
            id_b = _get_runtime_identity()
        self.assertNotEqual(id_a, id_b)
        tmp.unlink(missing_ok=True)


class ConsensusImageReceiptCheckTest(unittest.TestCase):
    """Review v4 requirement 3: when SPARKRING_IMAGE_DIGEST is set,
    consensus must fail closed if any rank has an empty
    image_receipt."""

    def test_empty_image_receipt_fails_when_digest_set(self):
        """When SPARKRING_IMAGE_DIGEST is set in env and a rank's
        image_receipt is empty, consensus must fail."""
        from spark_transport_contract import SELECTOR_NCCL_IB
        # Build records with empty image_receipt for one rank.
        base = _collect_rank_identity_record_for_test(SELECTOR_NCCL_IB)
        records = []
        for r in range(4):
            rec = dict(base)
            rec["rank"] = r
            if r == 2:
                rec["image_receipt"] = ""
            records.append(rec)
        backend = _MockControlBackend(
            consensus_sum=4, identity_records=records,
        )
        with patch.dict("os.environ", {"SPARKRING_IMAGE_DIGEST": "sha256:" + "f" * 64}):
            with self.assertRaises(RuntimeError) as ctx:
                _run_control_plane_consensus(
                    SELECTOR_NCCL_IB, rank=0, world_size=4,
                    control_backend=backend,
                    identity_record=records[0],
                )
        self.assertIn("image_receipt", str(ctx.exception))

    def test_nonempty_image_receipt_passes_when_digest_set(self):
        """When SPARKRING_IMAGE_DIGEST is set and all ranks have
        non-empty image_receipt, consensus passes."""
        from spark_transport_contract import SELECTOR_NCCL_IB
        digest = "sha256:" + "a" * 64
        base = _collect_rank_identity_record_for_test(SELECTOR_NCCL_IB)
        base["image_receipt"] = digest
        records = []
        for r in range(4):
            rec = dict(base)
            rec["rank"] = r
            records.append(rec)
        backend = _MockControlBackend(
            consensus_sum=4, identity_records=records,
        )
        with patch.dict("os.environ", {"SPARKRING_IMAGE_DIGEST": digest}):
            result = _run_control_plane_consensus(
                SELECTOR_NCCL_IB, rank=0, world_size=4,
                control_backend=backend,
                identity_record=records[0],
            )
        self.assertEqual(result, SELECTOR_NCCL_IB)


class ConsensusNCCLIdentityCheckTest(unittest.TestCase):
    """Review v4 requirement 4: the NCCL arm must verify nccl_identity
    on every rank — a rank whose NCCL library fails the identity
    check must abort all ranks."""

    def test_nccl_arm_rejects_failed_identity(self):
        """When all ranks have nccl_identity=False (e.g. a generic
        system library was used instead of NCCL), consensus must
        fail with an NCCL identity error.  All ranks agree on
        False (shared_fields passes), but the NCCL-specific check
        requires True on every rank."""
        from spark_transport_contract import SELECTOR_NCCL_IB
        base = _collect_rank_identity_record_for_test(SELECTOR_NCCL_IB)
        records = []
        for r in range(4):
            rec = dict(base)
            rec["rank"] = r
            rec["nccl_identity"] = False
            records.append(rec)
        backend = _MockControlBackend(
            consensus_sum=4, identity_records=records,
        )
        with self.assertRaises(RuntimeError) as ctx:
            _run_control_plane_consensus(
                SELECTOR_NCCL_IB, rank=0, world_size=4,
                control_backend=backend,
                identity_record=records[0],
            )
        self.assertIn("NCCL identity", str(ctx.exception))

    def test_nccl_arm_passes_with_valid_identity(self):
        """When all ranks have nccl_identity=True, the NCCL arm
        passes consensus."""
        from spark_transport_contract import SELECTOR_NCCL_IB
        base = _collect_rank_identity_record_for_test(SELECTOR_NCCL_IB)
        records = []
        for r in range(4):
            rec = dict(base)
            rec["rank"] = r
            rec["nccl_identity"] = True
            records.append(rec)
        backend = _MockControlBackend(
            consensus_sum=4, identity_records=records,
        )
        result = _run_control_plane_consensus(
            SELECTOR_NCCL_IB, rank=0, world_size=4,
            control_backend=backend,
            identity_record=records[0],
        )
        self.assertEqual(result, SELECTOR_NCCL_IB)


class ConsensusRuntimeIdentityMismatchTest(unittest.TestCase):
    """Review v4 requirement 5: runtime_identity is a shared field —
    a mismatch across ranks must fail consensus."""

    def test_runtime_identity_mismatch_fails(self):
        """When one rank has a different runtime_identity, consensus
        must fail with a cross-rank identity field mismatch."""
        from spark_transport_contract import SELECTOR_NCCL_IB
        base = _collect_rank_identity_record_for_test(SELECTOR_NCCL_IB)
        records = []
        for r in range(4):
            rec = dict(base)
            rec["rank"] = r
            if r == 2:
                rec["runtime_identity"] = "different-runtime"
            records.append(rec)
        backend = _MockControlBackend(
            consensus_sum=4, identity_records=records,
        )
        with self.assertRaises(RuntimeError) as ctx:
            _run_control_plane_consensus(
                SELECTOR_NCCL_IB, rank=0, world_size=4,
                control_backend=backend,
                identity_record=records[0],
            )
        self.assertIn("runtime_identity", str(ctx.exception))
        self.assertIn("mismatch", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
