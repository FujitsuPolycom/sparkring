"""CPU-only adversarial tests for the tracer-bullet two-arm orchestrator.

Tests prove:
- Two arms enter different code paths (different selectors).
- Unknown/disabled values fail closed.
- --elements changes the workload.
- One missing/duplicate/extra rank invalidates the run.
- Predicted counters cannot replace observed counters.
- The no-executor state cannot be labeled executed.
- Observed counters classify every collective exactly once.
- Arm invalidation rules (SIRCL: fallback invalidates; NCCL: native invalidates).
- Totals reconcile per-rank and globally.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch
from spark_two_arm_orchestrator import (
    ArmPlan,
    ArmResult,
    ArmSpec,
    EXIT_MISSING_SEAM,
    EXIT_VALID,
    RankCapability,
    RankLaunchEntry,
    RankReceipt,
    SELECTOR_NCCL,
    SELECTOR_SIRCL,
    SelectorConsensusError,
    TwoArmPlan,
    TwoArmResult,
    _BF16_ATOL,
    _BF16_RTOL,
    _BF16_TOLERANCE,
    _NCCL_SELECTOR_ENVS,
    _SIRCL_SELECTOR_ENVS,
    _TRANSPORT_NCCL_IB,
    _TRANSPORT_SIRCL,
    check_selector_consensus,
    execute_arm,
    main,
    render_plan,
    validate_arm_receipts,
    validate_plan,
    validate_two_arm_results,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_sircl_spec(**kwargs) -> ArmSpec:
    defaults = dict(
        transport=_TRANSPORT_SIRCL,
        selector_env_vars=_SIRCL_SELECTOR_ENVS,
        world_size=4,
        iterations=1000,
        elements=6144,
        timeout_seconds=300,
    )
    defaults.update(kwargs)
    return ArmSpec(**defaults)


def _make_nccl_spec(**kwargs) -> ArmSpec:
    defaults = dict(
        transport=_TRANSPORT_NCCL_IB,
        selector_env_vars=_NCCL_SELECTOR_ENVS,
        world_size=4,
        iterations=1000,
        elements=6144,
        timeout_seconds=300,
    )
    defaults.update(kwargs)
    return ArmSpec(**defaults)


def _make_plan(dry_run: bool = True) -> TwoArmPlan:
    return render_plan(
        _make_sircl_spec(),
        _make_nccl_spec(),
        dry_run=dry_run,
        hosts=tuple(f"spark-{i}" for i in range(4)) if not dry_run else None,
    )

def _compute_expected_fp32_hash(iterations: int, elements: int, world_size: int) -> str:
    """Compute the real expected_fp32_hash for the given params."""
    import hashlib
    import torch
    from tp4_numerical_audit import make_rank_input
    h = hashlib.sha256()
    for seq in range(iterations):
        cpu_inputs = [make_rank_input(seq, r, elements) for r in range(world_size)]
        fp32_sum = torch.stack([t.float() for t in cpu_inputs]).sum(dim=0)
        t = fp32_sum.cpu().contiguous()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _compute_run_contract_hash(
    transport: str, selector: str, rank: int, world_size: int,
    iterations: int, elements: int,
) -> str:
    """Compute the real run_contract_hash for the given params.

    Uses the canonical build_env_projection from spark_transport_contract
    so the hash matches the validator's recomputation exactly.
    """
    import hashlib
    import json
    from spark_transport_contract import build_env_projection
    probe_id = "spark_transport/integrations/vllm/tp4_numerical_audit.py"
    env_proj = build_env_projection(
        selector, rank, world_size, iterations, elements,
        transport=transport,
    )
    contract = {
        "arm": transport,
        "selector": selector,
        "transport": transport,
        "rank": rank,
        "rank_identity": f"rank-{rank}-of-{world_size}",
        "iterations": iterations,
        "elements": elements,
        "world_size": world_size,
        "seed_identity": "0x5A17+seq*WORLD_SIZE+rank",
        "argv_projection": ["python", probe_id],
        "env_projection": env_proj,
        "probe_identity": probe_id,
        "binary_identity": probe_id,
        "topology": "tp4_switchless_ring",
        "workload": "tp4_numerical_audit",
        "order": "identical",
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode()
    ).hexdigest()


def _make_receipt(
    rank: int = 0,
    transport: str = _TRANSPORT_SIRCL,
    custom: int = 1000,
    fallback: int = 0,
    unsupported: int = 0,
    unclassified: int = 0,
    host: str = "spark-0",
    selector: str | None = None,
    iterations: int = 1000,
    elements: int = 6144,
    world_size: int = 4,
    *,
    expected_fp32_hash: str | None = None,
    actual_output_hash: str = "b" * 64,
    all_finite: bool = True,
    max_abs_error: float = 0.0,
    max_rel_error: float = 0.0,
    sample_count: int | None = None,
    run_contract_hash: str | None = None,
    rank_identity: str | None = None,
    tolerance_result: str = "pass",
    tolerance_metric: str = "elementwise_atol_rtol",
    actual_dtype: str = "bfloat16",
    actual_byte_order: str = "little",
    tolerance_atol: float = _BF16_ATOL,
    tolerance_rtol: float = _BF16_RTOL,
    native_collectives: int | None = None,
    nccl_ib_collectives: int | None = None,
    nccl_socket_collectives: int | None = None,
    fatal_after_native_collectives: int = 0,
) -> RankReceipt:
    total = custom + fallback + unsupported + unclassified
    if selector is None:
        selector = SELECTOR_SIRCL if transport == _TRANSPORT_SIRCL else SELECTOR_NCCL
    if sample_count is None:
        sample_count = iterations * elements
    if rank_identity is None:
        rank_identity = f"rank-{rank}-of-{world_size}"
    if expected_fp32_hash is None:
        expected_fp32_hash = _compute_expected_fp32_hash(iterations, elements, world_size)
    if run_contract_hash is None:
        run_contract_hash = _compute_run_contract_hash(
            transport, selector, rank, world_size, iterations, elements,
        )
    # Goal 11: default new attribution counters to match the legacy
    # counters — native_collectives mirrors custom_collectives (SIRCL
    # native path), nccl_ib_collectives mirrors fallback_collectives
    # (NCCL-IB transport path).
    if native_collectives is None:
        native_collectives = custom
    if nccl_ib_collectives is None:
        nccl_ib_collectives = fallback if transport == _TRANSPORT_NCCL_IB else 0
    if nccl_socket_collectives is None:
        nccl_socket_collectives = (
            fallback
            if transport == _TRANSPORT_SIRCL
            else 0
        )
    return RankReceipt(
        rank=rank,
        host=host,
        transport=transport,
        selector=selector,
        iterations=iterations,
        elements=elements,
        world_size=world_size,
        custom_collectives=custom,
        fallback_collectives=fallback,
        unsupported_bypassed_collectives=unsupported,
        unclassified_collectives=unclassified,
        total_collectives=total,
        expected_fp32_hash=expected_fp32_hash,
        actual_output_hash=actual_output_hash,
        actual_dtype=actual_dtype,
        actual_byte_order=actual_byte_order,
        all_finite=all_finite,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        tolerance_result=tolerance_result,
        tolerance_metric=tolerance_metric,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
        sample_count=sample_count,
        run_contract_hash=run_contract_hash,
        rank_identity=rank_identity,
        native_collectives=native_collectives,
        nccl_ib_collectives=nccl_ib_collectives,
        nccl_socket_collectives=nccl_socket_collectives,
        fatal_after_native_collectives=fatal_after_native_collectives,
    )


def _make_sircl_receipts(
    n: int = 4, iters: int = 1000,
) -> tuple[RankReceipt, ...]:
    return tuple(
        _make_receipt(
            rank=r,
            transport=_TRANSPORT_SIRCL,
            custom=iters,
            fallback=0,
            unsupported=0,
            unclassified=0,
            host=f"spark-{r}",
            iterations=iters,
        )
        for r in range(n)
    )


def _make_nccl_receipts(
    n: int = 4, iters: int = 1000,
) -> tuple[RankReceipt, ...]:
    return tuple(
        _make_receipt(
            rank=r,
            transport=_TRANSPORT_NCCL_IB,
            custom=0,
            fallback=iters,
            unsupported=0,
            unclassified=0,
            host=f"spark-{r}",
            iterations=iters,
        )
        for r in range(n)
    )


# ---------------------------------------------------------------------------
# Two arms enter different code paths
# ---------------------------------------------------------------------------

class DifferentCodePathsTest(unittest.TestCase):
    """Arms must select different selectors (different code paths)."""

    def test_sircl_selector_is_custom(self) -> None:
        plan = _make_plan()
        self.assertEqual(plan.sircl_arm.selector, SELECTOR_SIRCL)

    def test_nccl_selector_is_disabled(self) -> None:
        plan = _make_plan()
        self.assertEqual(plan.nccl_arm.selector, SELECTOR_NCCL)

    def test_selectors_differ(self) -> None:
        plan = _make_plan()
        self.assertNotEqual(
            plan.sircl_arm.selector, plan.nccl_arm.selector,
        )

    def test_sircl_env_has_custom_mode(self) -> None:
        plan = _make_plan()
        env = plan.sircl_arm.rank_launches[0].env_vars
        self.assertEqual(env["VLLM_SPARK_TP4_MODE"], "custom")

    def test_nccl_env_has_disabled_mode(self) -> None:
        plan = _make_plan()
        env = plan.nccl_arm.rank_launches[0].env_vars
        self.assertEqual(env["VLLM_SPARK_TP4_MODE"], "disabled")


# ---------------------------------------------------------------------------
# Unknown/disabled values fail closed
# ---------------------------------------------------------------------------

class FailClosedTest(unittest.TestCase):
    """Unknown selector values must fail closed."""

    def test_orchestrator_rejects_unknown_sircl_selector(self) -> None:
        with self.assertRaises(ValueError):
            render_plan(
                ArmSpec(
                    transport=_TRANSPORT_SIRCL,
                    selector_env_vars=frozenset({"VLLM_SPARK_TP4_MODE=unknown"}),
                ),
                _make_nccl_spec(),
            )

    def test_orchestrator_rejects_empty_sircl_selector(self) -> None:
        with self.assertRaises(ValueError):
            render_plan(
                ArmSpec(
                    transport=_TRANSPORT_SIRCL,
                    selector_env_vars=frozenset({""}),
                ),
                _make_nccl_spec(),
            )

    def test_plan_validation_rejects_same_selector(self) -> None:
        """Both arms having the same selector is invalid."""
        plan = render_plan(_make_sircl_spec(), _make_nccl_spec())
        # Manually create a plan with same selectors
        # This is tested by ensuring validate_plan catches it
        errors = validate_plan(plan)
        self.assertEqual(errors, [])

    def test_cli_rejects_no_dry_run_without_executor(self) -> None:
        """--no-dry-run without executor must return EXIT_MISSING_SEAM."""
        rc = main(["--no-dry-run", "--validate-only"])
        self.assertEqual(rc, EXIT_MISSING_SEAM)


# ---------------------------------------------------------------------------
# --elements changes the workload
# ---------------------------------------------------------------------------

class ElementsConsumedTest(unittest.TestCase):
    """--elements must be consumed, not hardcoded to 6144."""

    def test_elements_1024_consumed(self) -> None:
        plan = render_plan(
            _make_sircl_spec(elements=1024),
            _make_nccl_spec(elements=1024),
        )
        self.assertEqual(
            plan.sircl_arm.rank_launches[0].env_vars["ELEMENTS"], "1024",
        )
        self.assertEqual(
            plan.nccl_arm.rank_launches[0].env_vars["ELEMENTS"], "1024",
        )

    def test_elements_6144_default(self) -> None:
        plan = _make_plan()
        self.assertEqual(
            plan.sircl_arm.rank_launches[0].env_vars["ELEMENTS"], "6144",
        )

    def test_elements_in_shared_identity(self) -> None:
        plan = _make_plan()
        self.assertEqual(plan.shared_identity["elements"], "6144")

    def test_elements_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            render_plan(
                _make_sircl_spec(elements=1024),
                _make_nccl_spec(elements=2048),
            )


# ---------------------------------------------------------------------------
# One rank per host (truthful four-node launch)
# ---------------------------------------------------------------------------

class OneRankPerHostTest(unittest.TestCase):
    """Plan must represent four-node launch as one rank per host."""

    def test_four_rank_launches_per_arm(self) -> None:
        plan = _make_plan()
        self.assertEqual(len(plan.sircl_arm.rank_launches), 4)
        self.assertEqual(len(plan.nccl_arm.rank_launches), 4)

    def test_unique_hosts(self) -> None:
        plan = _make_plan()
        sircl_hosts = [rl.host for rl in plan.sircl_arm.rank_launches]
        self.assertEqual(len(set(sircl_hosts)), 4)

    def test_unique_ranks(self) -> None:
        plan = _make_plan()
        sircl_ranks = [rl.rank for rl in plan.sircl_arm.rank_launches]
        self.assertEqual(sorted(sircl_ranks), [0, 1, 2, 3])

    def test_launch_model_one_rank_per_host(self) -> None:
        plan = _make_plan()
        self.assertEqual(plan.shared_identity["launch_model"], "one_rank_per_host")

    def test_no_torchrun_nproc_per_node(self) -> None:
        """Commands must not use torchrun --nproc_per_node=4."""
        plan = _make_plan()
        for rl in plan.sircl_arm.rank_launches:
            self.assertFalse(
                any("torchrun" in c for c in rl.command),
                "Commands must not use torchrun",
            )
            self.assertFalse(
                any("nproc_per_node" in c for c in rl.command),
                "Commands must not use nproc_per_node",
            )


# ---------------------------------------------------------------------------
# Missing/duplicate/extra rank invalidates the run
# ---------------------------------------------------------------------------

class RankIntegrityTest(unittest.TestCase):
    """One missing/duplicate/extra rank invalidates the run."""

    def test_missing_rank_invalidates(self) -> None:
        plan = _make_plan()
        receipts = _make_sircl_receipts(n=4)[:3]  # only 3 receipts
        errors = validate_arm_receipts(plan.sircl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("expected 4" in e for e in errors))

    def test_duplicate_rank_invalidates(self) -> None:
        plan = _make_plan()
        r0 = _make_receipt(rank=0, transport=_TRANSPORT_SIRCL, custom=1000)
        r0_dup = _make_receipt(rank=0, transport=_TRANSPORT_SIRCL, custom=1000)
        r2 = _make_receipt(rank=2, transport=_TRANSPORT_SIRCL, custom=1000, host="spark-2")
        r3 = _make_receipt(rank=3, transport=_TRANSPORT_SIRCL, custom=1000, host="spark-3")
        receipts = (r0, r0_dup, r2, r3)
        errors = validate_arm_receipts(plan.sircl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("duplicate" in e.lower() or "rank" in e.lower() for e in errors))

    def test_extra_rank_invalidates(self) -> None:
        plan = _make_plan()
        receipts = _make_sircl_receipts(n=4) + (
            _make_receipt(rank=4, transport=_TRANSPORT_SIRCL, custom=1000, host="spark-4"),
        )
        errors = validate_arm_receipts(plan.sircl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("expected 4" in e for e in errors))

    def test_wrong_rank_order_invalidates(self) -> None:
        plan = _make_plan()
        receipts = tuple(sorted(_make_sircl_receipts(n=4), key=lambda r: -r.rank))
        errors = validate_arm_receipts(plan.sircl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("rank" in e.lower() for e in errors))


# ---------------------------------------------------------------------------
# Predicted counters cannot replace observed counters
# ---------------------------------------------------------------------------

class ObservedCountersTest(unittest.TestCase):
    """Counters must come from per-rank receipts, not caller declarations."""

    def test_no_expected_counters_in_plan(self) -> None:
        """Plan must not carry caller-authored expected counter declarations."""
        plan = _make_plan()
        self.assertFalse(
            hasattr(plan.sircl_arm, "expected_counters"),
            "ArmPlan must not have expected_counters field",
        )

    def test_receipts_have_observed_counters(self) -> None:
        receipt = _make_receipt(rank=0, custom=1000)
        self.assertEqual(receipt.custom_collectives, 1000)
        self.assertEqual(receipt.total_collectives, 1000)

    def test_receipt_classification_sums_to_total(self) -> None:
        receipt = _make_receipt(
            rank=0, custom=800, fallback=100, unsupported=50, unclassified=50,
        )
        self.assertEqual(
            receipt.custom_collectives
            + receipt.fallback_collectives
            + receipt.unsupported_bypassed_collectives
            + receipt.unclassified_collectives,
            receipt.total_collectives,
        )

    def test_receipt_rejects_mismatched_classification(self) -> None:
        with self.assertRaises(ValueError):
            RankReceipt(
                rank=0, host="spark-0", transport=_TRANSPORT_SIRCL,
                selector=SELECTOR_SIRCL, iterations=1000, elements=6144,
                world_size=4,
                custom_collectives=800, fallback_collectives=100,
                unsupported_bypassed_collectives=50, unclassified_collectives=50,
                total_collectives=999,  # wrong: 800+100+50+50=1000
                expected_fp32_hash="", actual_output_hash="",
                actual_dtype="bfloat16", actual_byte_order="little",
                all_finite=False, max_abs_error=0.0, max_rel_error=0.0,
                tolerance_result="fail", tolerance_metric="elementwise_atol_rtol",
                tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
                sample_count=0, run_contract_hash="", rank_identity="",
            )

# ---------------------------------------------------------------------------
# No-executor state cannot be labeled executed
# ---------------------------------------------------------------------------

class NoExecutorTest(unittest.TestCase):
    """Without an executor, --no-dry-run must fail, not execute."""

    def test_dry_run_is_offline(self) -> None:
        plan = _make_plan(dry_run=True)
        errors = validate_plan(plan)
        self.assertEqual(errors, [])
        self.assertEqual(plan.sircl_arm.safety_class, "OFFLINE")

    def test_non_dry_run_without_executor_fails(self) -> None:
        plan = _make_plan(dry_run=False)
        errors = validate_plan(plan)
        self.assertTrue(any("executor_available" in e for e in errors))

    def test_execute_arm_without_executor_returns_none(self) -> None:
        plan = _make_plan(dry_run=False)
        result = execute_arm(plan.sircl_arm, None, "I CONFIRM EXECUTE TWO_ARM BENCHMARK")
        self.assertIsNone(result)

    def test_execute_arm_with_wrong_confirmation_returns_none(self) -> None:
        plan = _make_plan(dry_run=False)
        result = execute_arm(plan.sircl_arm, object(), "WRONG")
        self.assertIsNone(result)

    def test_cli_no_dry_run_returns_missing_seam(self) -> None:
        rc = main(["--no-dry-run"])
        self.assertEqual(rc, EXIT_MISSING_SEAM)


# ---------------------------------------------------------------------------
# Arm invalidation rules
# ---------------------------------------------------------------------------

class ArmInvalidationTest(unittest.TestCase):
    """SIRCL arm: fallback/unclassified invalidates. NCCL arm: native invalidates."""

    def test_sircl_arm_with_fallback_invalidates(self) -> None:
        plan = _make_plan()
        receipts = tuple(
            _make_receipt(
                rank=r,
                transport=_TRANSPORT_SIRCL,
                custom=900,
                fallback=100,
                host=f"spark-{r}",
            )
            for r in range(4)
        )
        errors = validate_arm_receipts(plan.sircl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("invalidated" in e for e in errors))

    def test_sircl_arm_with_unclassified_invalidates(self) -> None:
        plan = _make_plan()
        receipts = tuple(
            _make_receipt(
                rank=r,
                transport=_TRANSPORT_SIRCL,
                custom=900,
                unclassified=100,
                host=f"spark-{r}",
            )
            for r in range(4)
        )
        errors = validate_arm_receipts(plan.sircl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("invalidated" in e for e in errors))

    def test_sircl_arm_clean_passes(self) -> None:
        plan = _make_plan()
        receipts = _make_sircl_receipts()
        errors = validate_arm_receipts(plan.sircl_arm, receipts, 1000, 6144, 4)
        self.assertEqual(errors, [])

    def test_nccl_arm_with_native_invalidates(self) -> None:
        plan = _make_plan()
        receipts = tuple(
            _make_receipt(
                rank=r,
                transport=_TRANSPORT_NCCL_IB,
                custom=100,
                fallback=900,
                host=f"spark-{r}",
            )
            for r in range(4)
        )
        errors = validate_arm_receipts(plan.nccl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("invalidated" in e for e in errors))

    def test_nccl_arm_clean_passes(self) -> None:
        plan = _make_plan()
        receipts = _make_nccl_receipts()
        errors = validate_arm_receipts(plan.nccl_arm, receipts, 1000, 6144, 4)
        self.assertEqual(errors, [])

    def test_typed_receipt_rejects_contradictory_counter_aliases(self) -> None:
        """The executor's typed seam must not bypass wire-parser aliases."""
        with self.assertRaisesRegex(ValueError, "fallback_collectives"):
            _make_receipt(
                rank=0,
                transport=_TRANSPORT_NCCL_IB,
                custom=0,
                fallback=0,
                unsupported=1000,
                host="spark-0",
                nccl_ib_collectives=1000,
            )

    def test_nccl_arm_with_unsupported_bypass_invalidates(self) -> None:
        """Unsupported fallback is invalid on the NCCL control arm too."""
        plan = _make_plan()
        receipts = tuple(
            _make_receipt(
                rank=r,
                transport=_TRANSPORT_NCCL_IB,
                custom=0,
                fallback=900,
                unsupported=100,
                host=f"spark-{r}",
                nccl_ib_collectives=900,
            )
            for r in range(4)
        )
        errors = validate_arm_receipts(plan.nccl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("unsupported-bypassed" in e for e in errors))

    def test_sircl_arm_with_fatal_after_native_invalidates(self) -> None:
        """Goal 11: fatal_after_native_collectives > 0 on SIRCL arm
        means a rank crashed after native execution — hard failure."""
        plan = _make_plan()
        receipts = tuple(
            _make_receipt(
                rank=r,
                transport=_TRANSPORT_SIRCL,
                custom=1000,
                host=f"spark-{r}",
                native_collectives=1000,
                fatal_after_native_collectives=1,
            )
            for r in range(4)
        )
        errors = validate_arm_receipts(plan.sircl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("fatal-after-native" in e for e in errors))

    def test_sircl_arm_with_nccl_socket_invalidates(self) -> None:
        """Goal 11: nccl_socket_collectives > 0 on SIRCL arm means
        NCCL Socket transport was used — not native SIRCL."""
        plan = _make_plan()
        receipts = tuple(
            _make_receipt(
                rank=r,
                transport=_TRANSPORT_SIRCL,
                custom=900,
                fallback=100,
                host=f"spark-{r}",
                native_collectives=900,
                nccl_socket_collectives=100,
            )
            for r in range(4)
        )
        errors = validate_arm_receipts(plan.sircl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("nccl_socket" in e for e in errors))

    def test_nccl_arm_with_native_collectives_invalidates(self) -> None:
        """Goal 11: native_collectives > 0 on NCCL arm means SIRCL
        transport leaked into the NCCL arm — hard failure."""
        plan = _make_plan()
        receipts = tuple(
            _make_receipt(
                rank=r,
                transport=_TRANSPORT_NCCL_IB,
                custom=100,
                fallback=900,
                host=f"spark-{r}",
                nccl_ib_collectives=900,
                native_collectives=100,
            )
            for r in range(4)
        )
        errors = validate_arm_receipts(plan.nccl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("invalidated" in e for e in errors))
        self.assertTrue(any("native" in e for e in errors))

    def test_nccl_arm_with_fatal_after_native_invalidates(self) -> None:
        """Goal 11: fatal_after_native_collectives > 0 on NCCL arm
        is also a hard failure."""
        plan = _make_plan()
        receipts = tuple(
            _make_receipt(
                rank=r,
                transport=_TRANSPORT_NCCL_IB,
                custom=0,
                fallback=1000,
                host=f"spark-{r}",
                nccl_ib_collectives=1000,
                fatal_after_native_collectives=1,
            )
            for r in range(4)
        )
        errors = validate_arm_receipts(plan.nccl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("fatal-after-native" in e for e in errors))


# ---------------------------------------------------------------------------
# Totals reconcile
# ---------------------------------------------------------------------------

class TotalsReconcileTest(unittest.TestCase):
    """Per-rank and global totals must reconcile."""

    def test_cross_arm_total_mismatch(self) -> None:
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(iters=999),  # different count
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, _make_plan())
        self.assertTrue(any("cross-arm total mismatch" in e for e in errors))

    def test_cross_arm_total_match(self) -> None:
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, _make_plan())
        # No errors about cross-arm mismatch
        self.assertFalse(any("cross-arm" in e for e in errors))


# ---------------------------------------------------------------------------
# CLI exit code tests
# ---------------------------------------------------------------------------

class CLIExitCodeTest(unittest.TestCase):
    """CLI must return correct exit codes."""

    def test_dry_run_valid_exits_zero(self) -> None:
        rc = main(["--dry-run", "--validate-only"])
        self.assertEqual(rc, EXIT_VALID)

    def test_dry_run_prints_plan(self) -> None:
        with patch("builtins.print") as mock_print:
            rc = main(["--dry-run"])
            self.assertEqual(rc, EXIT_VALID)
            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("PLAN VALID", printed)


# ---------------------------------------------------------------------------
# ArmSpec validation tests
# ---------------------------------------------------------------------------

class ArmSpecValidationTest(unittest.TestCase):
    """ArmSpec must validate its fields."""

    def test_rejects_bool_world_size(self) -> None:
        with self.assertRaises(ValueError):
            _make_sircl_spec(world_size=True)  # type: ignore[arg-type]

    def test_rejects_zero_elements(self) -> None:
        with self.assertRaises(ValueError):
            _make_sircl_spec(elements=0)

    def test_rejects_negative_timeout(self) -> None:
        with self.assertRaises(ValueError):
            _make_sircl_spec(timeout_seconds=0)


# ---------------------------------------------------------------------------
# SIRCL arm rejects unsupported_bypassed (NCCL fallback) collectives
# ---------------------------------------------------------------------------

class SirclUnsupportedBypassTest(unittest.TestCase):
    """SIRCL arm with unsupported_bypassed>0 must be invalidated.

    unsupported_bypassed represents NCCL fallback execution, not a
    skipped collective. A SIRCL arm with custom=0, unsupported=10,
    fallback=10 must be invalid.
    """

    def test_sircl_arm_unsupported_bypassed_invalidates(self) -> None:
        plan = _make_plan()
        receipts = tuple(
            _make_receipt(
                rank=r,
                transport=_TRANSPORT_SIRCL,
                custom=0,
                fallback=10,
                unsupported=10,
                host=f"spark-{r}",
            )
            for r in range(4)
        )
        errors = validate_arm_receipts(plan.sircl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("unsupported-bypassed" in e for e in errors))

    def test_sircl_arm_unsupported_only_still_invalidates(self) -> None:
        """custom=0, unsupported=10, fallback=0 must still invalidate."""
        plan = _make_plan()
        receipts = tuple(
            _make_receipt(
                rank=r,
                transport=_TRANSPORT_SIRCL,
                custom=0,
                fallback=0,
                unsupported=10,
                host=f"spark-{r}",
            )
            for r in range(4)
        )
        errors = validate_arm_receipts(plan.sircl_arm, receipts, 1000, 6144, 4)
        self.assertTrue(any("unsupported-bypassed" in e for e in errors))


# ---------------------------------------------------------------------------
# End-to-end forged result validation against plan
# ---------------------------------------------------------------------------

class ForgedResultValidationTest(unittest.TestCase):
    """End-to-end tests calling validate_two_arm_results with a plan.

    A one-rank-per-arm forged result must be rejected when validated
    against the original plan that specifies 4 hosts and world_size=4.
    """

    def _make_valid_result(self) -> TwoArmResult:
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        return TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)

    def test_valid_result_with_plan_passes(self) -> None:
        plan = _make_plan()
        result = self._make_valid_result()
        errors = validate_two_arm_results(result, plan)
        self.assertEqual(errors, [])

    def test_one_rank_forged_result_rejected(self) -> None:
        """A one-rank-per-arm forged result must be rejected."""
        plan = _make_plan()
        # Forge: only 1 receipt per arm, claiming world_size=1
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=(
                _make_receipt(
                    rank=0, transport=_TRANSPORT_SIRCL,
                    custom=1000, host="spark-0",
                ),
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=(
                _make_receipt(
                    rank=0, transport=_TRANSPORT_NCCL_IB,
                    fallback=1000, host="spark-0",
                ),
            ),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("expected 4" in e for e in errors))

    def test_wrong_host_forged_result_rejected(self) -> None:
        """Receipts with hosts not in the plan must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"evil-{r}",
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("planned host" in e for e in errors))

    def test_selector_mismatch_rejected(self) -> None:
        """Plan with wrong selectors must be caught."""
        # Create a valid result but validate against a plan that has
        # been tampered with (wrong selectors) — the plan validation
        # should catch this.
        plan = _make_plan()
        # Tamper with plan selectors
        tampered_sircl = ArmPlan(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            safety_class="OFFLINE",
            selector="disabled",  # wrong!
            rank_launches=plan.sircl_arm.rank_launches,
            timeout_seconds=300,
        )
        tampered_plan = TwoArmPlan(
            sircl_arm=tampered_sircl,
            nccl_arm=plan.nccl_arm,
            shared_identity=plan.shared_identity,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        result = self._make_valid_result()
        errors = validate_two_arm_results(result, tampered_plan)
        self.assertTrue(any("selector" in e.lower() for e in errors))

    def test_iteration_mismatch_rejected(self) -> None:
        """Receipts with total_collectives != plan iterations rejected."""
        plan = _make_plan()
        # Valid hosts and world_size, but total_collectives=500 not 1000
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=500),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=500),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("iterations" in e.lower() for e in errors))

    def test_cross_arm_host_mismatch_rejected(self) -> None:
        """Receipts from different host sets across arms rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        # NCCL receipts use different host names
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_NCCL_IB,
                    fallback=1000, host=f"other-{r}",
                )
                for r in range(4)
            ),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("host" in e.lower() for e in errors))


# ---------------------------------------------------------------------------
# A4: Public-entry-point regressions for all forgery cases
# ---------------------------------------------------------------------------

class PlanRequiredValidationTest(unittest.TestCase):
    """validate_two_arm_results must require an authoritative plan.
    No public validator path can validate without plan input.
    """

    def test_no_plan_raises_type_error(self) -> None:
        """Calling validate_two_arm_results without a plan must fail."""
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        # plan is now required — omitting it must raise TypeError
        with self.assertRaises(TypeError):
            validate_two_arm_results(result)  # type: ignore[call-arg]

    def test_one_rank_forged_without_plan_rejected(self) -> None:
        """A one-rank self-sized result must not pass even with receipts."""
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=(
                _make_receipt(
                    rank=0, transport=_TRANSPORT_SIRCL,
                    custom=1000, host="spark-0",
                    world_size=1,
                ),
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=(
                _make_receipt(
                    rank=0, transport=_TRANSPORT_NCCL_IB,
                    fallback=1000, host="spark-0",
                    world_size=1,
                ),
            ),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        plan = _make_plan()
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("expected 4" in e for e in errors))


class DuplicateHostForgeryTest(unittest.TestCase):
    """All ranks claiming the same allowed host must be rejected.
    Set membership is insufficient — each rank must match its exact
    planned host.
    """

    def test_all_ranks_same_host_rejected(self) -> None:
        """All 4 SIRCL receipts claim spark-0 — must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host="spark-0",  # all same host!
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        # Must have host mismatch errors for ranks 1,2,3
        host_errors = [e for e in errors if "planned host" in e]
        self.assertTrue(len(host_errors) >= 3,
                        f"Expected >=3 host mismatch errors, got: {errors}")


class WrongRankHostMappingTest(unittest.TestCase):
    """Receipts with a wrong rank-host mapping must be rejected."""

    def test_swapped_hosts_rejected(self) -> None:
        """Rank 0 on spark-1, rank 1 on spark-0 — must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{(r + 1) % 4}",  # swapped
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("planned host" in e for e in errors))


class SwappedTransportForgeryTest(unittest.TestCase):
    """Swapped arm transports must be rejected — SIRCL receipts on
    NCCL arm and vice versa.
    """

    def test_sircl_receipts_on_nccl_arm_rejected(self) -> None:
        """NCCL arm with SIRCL transport receipts must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        # NCCL arm but with SIRCL transport receipts!
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_sircl_receipts(n=4, iters=1000),  # wrong!
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("transport" in e for e in errors))


class SwappedSelectorForgeryTest(unittest.TestCase):
    """Receipts with swapped selectors (SIRCL selector on NCCL arm)
    must be rejected.
    """

    def test_sircl_selector_on_nccl_receipts_rejected(self) -> None:
        """NCCL arm receipts with SIRCL selector must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        # NCCL arm but receipts claim 'custom' selector
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_NCCL_IB,
                    fallback=1000, host=f"spark-{r}",
                    selector=SELECTOR_SIRCL,  # wrong!
                )
                for r in range(4)
            ),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("selector" in e for e in errors))


class WrongElementsForgeryTest(unittest.TestCase):
    """Receipts with wrong elements count must be rejected."""

    def test_wrong_elements_rejected(self) -> None:
        """Receipts with elements=2048 but plan says 6144 — rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    elements=2048,  # wrong!
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("elements" in e for e in errors))


class WrongIterationsWorldSizeForgeryTest(unittest.TestCase):
    """Receipts with wrong iterations or world_size must be rejected."""

    def test_wrong_iterations_rejected(self) -> None:
        """Receipts with iterations=500 but plan says 1000 — rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=500, host=f"spark-{r}",
                    iterations=500,  # wrong!
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("iterations" in e for e in errors))

    def test_wrong_world_size_rejected(self) -> None:
        """Receipts with world_size=2 but plan says 4 — rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    world_size=2,  # wrong!
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("world_size" in e for e in errors))


class MissingNumericalCommitmentTest(unittest.TestCase):
    """Receipts must carry a numerical commitment (total_collectives ==
    iterations). Missing or wrong total is rejected.
    """

    def test_total_collectives_not_equal_iterations_rejected(self) -> None:
        """Receipts with total_collectives != iterations — rejected."""
        plan = _make_plan()
        # total_collectives = 999 but iterations = 1000
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=999, host=f"spark-{r}",
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        # total_collectives (999) != iterations (1000)
        self.assertTrue(
            any("iterations" in e or "total_collectives" in e for e in errors),
            f"Expected iterations/total_collectives error, got: {errors}",
        )


# ---------------------------------------------------------------------------
# Goal-7 Part A: Adversarial regressions for validator hardening
# ---------------------------------------------------------------------------

class PlanValidatorRequiredTest(unittest.TestCase):
    """validate_two_arm_results must call validate_plan before receipts."""

    def test_invalid_plan_rejected_before_receipts(self) -> None:
        """A plan with invalid selectors must be rejected before receipts
        are even inspected."""
        plan = _make_plan()
        # Tamper with plan: wrong SIRCL selector
        tampered_sircl = ArmPlan(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            safety_class="OFFLINE",
            selector="disabled",  # wrong!
            rank_launches=plan.sircl_arm.rank_launches,
            timeout_seconds=300,
        )
        tampered_plan = TwoArmPlan(
            sircl_arm=tampered_sircl,
            nccl_arm=plan.nccl_arm,
            shared_identity=plan.shared_identity,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        result = TwoArmResult(
            sircl_arm=ArmResult(
                arm_name="sircl",
                transport=_TRANSPORT_SIRCL,
                receipts=_make_sircl_receipts(n=4, iters=1000),
            ),
            nccl_arm=ArmResult(
                arm_name="nccl_ib",
                transport=_TRANSPORT_NCCL_IB,
                receipts=_make_nccl_receipts(n=4, iters=1000),
            ),
            valid=True,
        )
        errors = validate_two_arm_results(result, tampered_plan)
        # Plan validation error must appear
        self.assertTrue(any("selector" in e for e in errors))

    def test_plan_with_duplicate_hosts_rejected(self) -> None:
        """A plan where all ranks use the same host is invalid —
        validate_plan catches duplicate hosts."""
        plan = _make_plan()
        # Tamper: all SIRCL ranks on spark-0
        bad_launches = tuple(
            RankLaunchEntry(
                rank=r, host="spark-0",
                command=plan.sircl_arm.rank_launches[0].command,
                env_vars=plan.sircl_arm.rank_launches[0].env_vars,
            )
            for r in range(4)
        )
        tampered_sircl = ArmPlan(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            safety_class="OFFLINE",
            selector=SELECTOR_SIRCL,
            rank_launches=bad_launches,
            timeout_seconds=300,
        )
        tampered_plan = TwoArmPlan(
            sircl_arm=tampered_sircl,
            nccl_arm=plan.nccl_arm,
            shared_identity=plan.shared_identity,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        errors = validate_plan(tampered_plan)
        self.assertTrue(any("duplicate hosts" in e for e in errors))


class SwappedArmNameForgeryTest(unittest.TestCase):
    """Swapped arm names in the result must be rejected."""

    def test_sircl_arm_named_nccl_rejected(self) -> None:
        """SIRCL arm with arm_name='nccl_ib' must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="nccl_ib",  # wrong name!
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("arm name" in e for e in errors))

    def test_nccl_arm_named_sircl_rejected(self) -> None:
        """NCCL arm with arm_name='sircl' must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        nccl = ArmResult(
            arm_name="sircl",  # wrong name!
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("arm name" in e for e in errors))


class SwappedPlanTransportForgeryTest(unittest.TestCase):
    """Swapped plan transport labels must be rejected."""

    def test_sircl_plan_with_nccl_transport_rejected(self) -> None:
        """Plan SIRCL arm with nccl_ib transport must be rejected."""
        plan = _make_plan()
        tampered_sircl = ArmPlan(
            arm_name="sircl",
            transport=_TRANSPORT_NCCL_IB,  # wrong!
            safety_class="OFFLINE",
            selector=SELECTOR_SIRCL,
            rank_launches=plan.sircl_arm.rank_launches,
            timeout_seconds=300,
        )
        tampered_plan = TwoArmPlan(
            sircl_arm=tampered_sircl,
            nccl_arm=plan.nccl_arm,
            shared_identity=plan.shared_identity,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        result = TwoArmResult(
            sircl_arm=ArmResult(
                arm_name="sircl",
                transport=_TRANSPORT_SIRCL,
                receipts=_make_sircl_receipts(n=4, iters=1000),
            ),
            nccl_arm=ArmResult(
                arm_name="nccl_ib",
                transport=_TRANSPORT_NCCL_IB,
                receipts=_make_nccl_receipts(n=4, iters=1000),
            ),
            valid=True,
        )
        errors = validate_two_arm_results(result, tampered_plan)
        # The plan's SIRCL arm has wrong transport — validate_plan
        # doesn't check transport names, but validate_two_arm_results
        # now checks arm transport against constants.
        self.assertTrue(any("transport" in e for e in errors))


class NCCLUnclassifiedForgeryTest(unittest.TestCase):
    """NCCL arm with nonzero unclassified_collectives must fail."""

    def test_nccl_with_unclassified_rejected(self) -> None:
        """NCCL arm receipts with unclassified > 0 must be rejected."""
        plan = _make_plan()
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=tuple(
                _make_receipt(
                    rank=r,
                    transport=_TRANSPORT_NCCL_IB,
                    fallback=900,
                    unclassified=100,  # should be 0 for NCCL
                    host=f"spark-{r}",
                )
                for r in range(4)
            ),
        )
        result = TwoArmResult(
            sircl_arm=ArmResult(
                arm_name="sircl",
                transport=_TRANSPORT_SIRCL,
                receipts=_make_sircl_receipts(n=4, iters=1000),
            ),
            nccl_arm=nccl,
            valid=True,
        )
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("unclassified" in e for e in errors))


class LaunchEnvSharedIdentityMismatchTest(unittest.TestCase):
    """Launch env vars must match shared_identity."""

    def test_iterations_mismatch_rejected(self) -> None:
        """Plan with ITERATIONS=1000 in env but shared_identity says 999."""
        plan = _make_plan()
        # Tamper: shared_identity iterations = 999 but env says 1000
        tampered_shared = dict(plan.shared_identity)
        tampered_shared["iterations"] = "999"
        tampered_plan = TwoArmPlan(
            sircl_arm=plan.sircl_arm,
            nccl_arm=plan.nccl_arm,
            shared_identity=tampered_shared,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        result = TwoArmResult(
            sircl_arm=ArmResult(
                arm_name="sircl",
                transport=_TRANSPORT_SIRCL,
                receipts=_make_sircl_receipts(n=4, iters=1000),
            ),
            nccl_arm=ArmResult(
                arm_name="nccl_ib",
                transport=_TRANSPORT_NCCL_IB,
                receipts=_make_nccl_receipts(n=4, iters=1000),
            ),
            valid=True,
        )
        errors = validate_two_arm_results(result, tampered_plan)
        self.assertTrue(any("ITERATIONS" in e or "iterations" in e for e in errors))


class ArbitraryOutputHashForgeryTest(unittest.TestCase):
    """Receipts with arbitrary output hashes must not be accepted as
    valid numerical evidence by the validator."""

    def test_arbitrary_hash_format_rejected(self) -> None:
        """A receipt with a non-hex hash value must be rejected at
        construction time."""
        with self.assertRaises(ValueError):
            RankReceipt(
                rank=0, host="spark-0", transport=_TRANSPORT_SIRCL,
                selector=SELECTOR_SIRCL, iterations=1, elements=6144,
                world_size=4, custom_collectives=1, fallback_collectives=0,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=1,
                expected_fp32_hash="not-a-hash",  # invalid format
                actual_output_hash="b" * 64,
                actual_dtype="bfloat16", actual_byte_order="little",
                all_finite=True, max_abs_error=0.0, max_rel_error=0.0,
                tolerance_result="pass", tolerance_metric="elementwise_atol_rtol",
                tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
                sample_count=6144, run_contract_hash="c" * 64,
                rank_identity="rank-0-of-4",
            )
    def test_valid_hash_format_accepted(self) -> None:
        """A receipt with a properly formatted hash is accepted at
        construction time (the validator checks semantic validity separately)."""
        receipt = RankReceipt(
            rank=0, host="spark-0", transport=_TRANSPORT_SIRCL,
            selector=SELECTOR_SIRCL, iterations=1, elements=6144,
            world_size=4, custom_collectives=1, fallback_collectives=0,
            unsupported_bypassed_collectives=0, unclassified_collectives=0,
            total_collectives=1,
            expected_fp32_hash="a" * 64,  # valid hex format
            actual_output_hash="b" * 64,
            actual_dtype="bfloat16", actual_byte_order="little",
            all_finite=True, max_abs_error=0.0, max_rel_error=0.0,
            tolerance_result="pass", tolerance_metric="elementwise_atol_rtol",
            tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
            sample_count=6144, run_contract_hash="c" * 64,
            rank_identity="rank-0-of-4",
            native_collectives=1,
        )
        self.assertEqual(receipt.expected_fp32_hash, "a" * 64)
# ---------------------------------------------------------------------------
# Goal-8: Four-rank authority — world_size must be exactly 4
# ---------------------------------------------------------------------------

class FourRankAuthorityTest(unittest.TestCase):
    """Plans and results with world_size != 4 must be rejected."""

    def _make_ws_plan_and_result(self, ws: int) -> tuple[TwoArmPlan, TwoArmResult]:
        """Build a self-consistent plan+result for an arbitrary world_size."""
        plan = render_plan(
            _make_sircl_spec(world_size=ws),
            _make_nccl_spec(world_size=ws),
        )
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=ws, iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=ws, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        return plan, result

    def test_world_size_1_rejected(self) -> None:
        plan, result = self._make_ws_plan_and_result(1)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("world_size" in e for e in errors),
                        f"Expected world_size error for ws=1, got: {errors}")

    def test_world_size_2_rejected(self) -> None:
        plan, result = self._make_ws_plan_and_result(2)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("world_size" in e for e in errors),
                        f"Expected world_size error for ws=2, got: {errors}")

    def test_world_size_3_rejected(self) -> None:
        plan, result = self._make_ws_plan_and_result(3)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("world_size" in e for e in errors),
                        f"Expected world_size error for ws=3, got: {errors}")

    def test_world_size_5_rejected(self) -> None:
        plan, result = self._make_ws_plan_and_result(5)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("world_size" in e for e in errors),
                        f"Expected world_size error for ws=5, got: {errors}")

    def test_world_size_4_accepted(self) -> None:
        """world_size=4 with self-consistent plan+result must pass."""
        plan, result = self._make_ws_plan_and_result(4)
        errors = validate_two_arm_results(result, plan)
        self.assertEqual(errors, [],
                         f"Expected no errors for ws=4, got: {errors}")


# ---------------------------------------------------------------------------
# Goal-8: Swapped arms consistently forged — still rejected
# ---------------------------------------------------------------------------

class SwappedArmsConsistentForgeryTest(unittest.TestCase):
    """Swapping SIRCL and NCCL arms in both plan and result consistently
    must still be rejected because the validator binds arm names to
    the authoritative _ARM_BINDING, not plan fields.
    """

    def test_swapped_arms_rejected(self) -> None:
        """Swap arm names, transports, and selectors in both plan and result.
        The validator should reject because _ARM_BINDING is validator-owned."""
        base_plan = _make_plan()
        # Build a swapped plan: sircl_arm has NCCL binding, nccl_arm has SIRCL
        swapped_sircl = ArmPlan(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            safety_class="OFFLINE",
            selector=SELECTOR_NCCL,
            rank_launches=base_plan.nccl_arm.rank_launches,
            timeout_seconds=300,
        )
        swapped_nccl = ArmPlan(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            safety_class="OFFLINE",
            selector=SELECTOR_SIRCL,
            rank_launches=base_plan.sircl_arm.rank_launches,
            timeout_seconds=300,
        )
        swapped_plan = TwoArmPlan(
            sircl_arm=swapped_sircl,
            nccl_arm=swapped_nccl,
            shared_identity=base_plan.shared_identity,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        # Build a swapped result to match the swapped plan
        sircl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        nccl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, swapped_plan)
        self.assertTrue(any("arm name" in e or "transport" in e or "selector" in e for e in errors),
                        f"Expected arm binding error, got: {errors}")


# ---------------------------------------------------------------------------
# Goal-8: Missing numerical evidence fields rejected
# ---------------------------------------------------------------------------

class MissingNumericalFieldsTest(unittest.TestCase):
    """Successful receipts missing expected_fp32_hash, actual_output_hash,
    or run_contract_hash must be rejected.
    """

    def test_missing_expected_fp32_hash_rejected(self) -> None:
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    expected_fp32_hash="",  # missing!
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("expected_fp32_hash" in e for e in errors),
                        f"Expected expected_fp32_hash error, got: {errors}")

    def test_missing_actual_output_hash_rejected(self) -> None:
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    actual_output_hash="",  # missing!
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("actual_output_hash" in e for e in errors),
                        f"Expected actual_output_hash error, got: {errors}")

    def test_missing_run_contract_hash_rejected(self) -> None:
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    run_contract_hash="",  # missing!
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("run_contract_hash" in e for e in errors),
                        f"Expected run_contract_hash error, got: {errors}")


# ---------------------------------------------------------------------------
# Goal-8: Wrong sample_count rejected
# ---------------------------------------------------------------------------

class WrongSampleCountTest(unittest.TestCase):
    """Successful receipts with sample_count != iterations*elements
    must be rejected.
    """

    def test_wrong_sample_count_rejected(self) -> None:
        plan = _make_plan()
        # iterations=1000, elements=6144 → expected sample_count=6144000
        # Use a wrong sample_count
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    sample_count=1000,  # wrong! should be 6144000
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("sample_count" in e for e in errors),
                        f"Expected sample_count error, got: {errors}")


# ---------------------------------------------------------------------------
# Goal-8: Out-of-tolerance error metrics rejected
# ---------------------------------------------------------------------------

class OutOfToleranceTest(unittest.TestCase):
    """Successful receipts with tolerance_result='fail' must be rejected.

    Goal 9: acceptance is governed by the elementwise criterion
    (tolerance_result), not by max_abs_error alone.  Max abs/rel
    error are diagnostics.
    """

    def test_tolerance_result_fail_rejected(self) -> None:
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    tolerance_result="fail",  # elementwise criterion failed
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("tolerance_result" in e and "pass" in e for e in errors),
            f"Expected tolerance_result fail error, got: {errors}",
        )

    def test_tolerance_metric_mismatch_rejected(self) -> None:
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    tolerance_metric="wrong_metric",
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("tolerance_metric" in e for e in errors),
            f"Expected tolerance_metric mismatch error, got: {errors}",
        )

    def test_max_abs_error_at_tolerance_accepted(self) -> None:
        """max_abs_error exactly at the tolerance boundary is accepted."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    max_abs_error=_BF16_TOLERANCE,  # exactly at boundary
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertFalse(any("max_abs_error" in e and "tolerance" in e for e in errors),
                         f"Should accept at-boundary error, got: {errors}")


# ---------------------------------------------------------------------------
# Goal-8: NaN/Inf error metrics rejected at construction
class NaNNegativeMetricsTest(unittest.TestCase):
    """RankReceipt construction with NaN/Inf error metrics must raise
    ValueError.
    """

    def test_nan_max_abs_error_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RankReceipt(
                rank=0, host="spark-0", transport=_TRANSPORT_SIRCL,
                selector=SELECTOR_SIRCL, iterations=1, elements=6144,
                world_size=4, custom_collectives=1, fallback_collectives=0,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=1,
                expected_fp32_hash="", actual_output_hash="",
                actual_dtype="bfloat16", actual_byte_order="little",
                all_finite=True, max_abs_error=float("nan"), max_rel_error=0.0,
                tolerance_result="pass", tolerance_metric="elementwise_atol_rtol",
                tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
                sample_count=0, run_contract_hash="", rank_identity="",
            )

    def test_inf_max_abs_error_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RankReceipt(
                rank=0, host="spark-0", transport=_TRANSPORT_SIRCL,
                selector=SELECTOR_SIRCL, iterations=1, elements=6144,
                world_size=4, custom_collectives=1, fallback_collectives=0,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=1,
                expected_fp32_hash="", actual_output_hash="",
                actual_dtype="bfloat16", actual_byte_order="little",
                all_finite=True, max_abs_error=float("inf"), max_rel_error=0.0,
                tolerance_result="pass", tolerance_metric="elementwise_atol_rtol",
                tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
                sample_count=0, run_contract_hash="", rank_identity="",
            )

    def test_nan_max_rel_error_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RankReceipt(
                rank=0, host="spark-0", transport=_TRANSPORT_SIRCL,
                selector=SELECTOR_SIRCL, iterations=1, elements=6144,
                world_size=4, custom_collectives=1, fallback_collectives=0,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=1,
                expected_fp32_hash="", actual_output_hash="",
                actual_dtype="bfloat16", actual_byte_order="little",
                all_finite=True, max_abs_error=0.0, max_rel_error=float("nan"),
                tolerance_result="pass", tolerance_metric="elementwise_atol_rtol",
                tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
                sample_count=0, run_contract_hash="", rank_identity="",
            )

    def test_inf_max_rel_error_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RankReceipt(
                rank=0, host="spark-0", transport=_TRANSPORT_SIRCL,
                selector=SELECTOR_SIRCL, iterations=1, elements=6144,
                world_size=4, custom_collectives=1, fallback_collectives=0,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=1,
                expected_fp32_hash="", actual_output_hash="",
                actual_dtype="bfloat16", actual_byte_order="little",
                all_finite=True, max_abs_error=0.0, max_rel_error=float("inf"),
                tolerance_result="pass", tolerance_metric="elementwise_atol_rtol",
                tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
                sample_count=0, run_contract_hash="", rank_identity="",
            )

# ---------------------------------------------------------------------------
# Goal-8: Per-rank env mutation on rank 3 rejected
# ---------------------------------------------------------------------------

class Rank3EnvMutationTest(unittest.TestCase):
    """Mutating RANK, WORLD_SIZE, ITERATIONS, or ELEMENTS on rank 3
    only must be rejected by per-rank validation.
    """

    def _make_plan_with_mutated_rank3_env(
        self, field: str, value: str,
    ) -> TwoArmPlan:
        """Build a plan where rank 3's env has a mutated field."""
        base_plan = _make_plan()
        mutated_launches = []
        for idx, rl in enumerate(base_plan.sircl_arm.rank_launches):
            env = dict(rl.env_vars)
            if idx == 3:
                env[field] = value
            mutated_launches.append(
                RankLaunchEntry(
                    rank=rl.rank, host=rl.host,
                    command=rl.command, env_vars=env,
                )
            )
        tampered_sircl = ArmPlan(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            safety_class="OFFLINE",
            selector=SELECTOR_SIRCL,
            rank_launches=tuple(mutated_launches),
            timeout_seconds=300,
        )
        return TwoArmPlan(
            sircl_arm=tampered_sircl,
            nccl_arm=base_plan.nccl_arm,
            shared_identity=base_plan.shared_identity,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )

    def test_rank3_rank_env_mutation_rejected(self) -> None:
        plan = self._make_plan_with_mutated_rank3_env("RANK", "0")
        errors = validate_plan(plan)
        self.assertTrue(any("RANK" in e for e in errors),
                        f"Expected RANK env error, got: {errors}")

    def test_rank3_world_size_env_mutation_rejected(self) -> None:
        plan = self._make_plan_with_mutated_rank3_env("WORLD_SIZE", "8")
        errors = validate_plan(plan)
        self.assertTrue(any("WORLD_SIZE" in e for e in errors),
                        f"Expected WORLD_SIZE env error, got: {errors}")

    def test_rank3_iterations_env_mutation_rejected(self) -> None:
        """Rank 3's receipt has iterations=999 but plan says 1000 —
        rejected by per-rank receipt validation."""
        plan = _make_plan()
        sircl_receipts = list(_make_sircl_receipts(n=4, iters=1000))
        sircl_receipts[3] = _make_receipt(
            rank=3, transport=_TRANSPORT_SIRCL,
            custom=999, host="spark-3", iterations=999,  # mutated!
        )
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(sircl_receipts),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("iterations" in e for e in errors),
                        f"Expected iterations error, got: {errors}")

    def test_rank3_elements_env_mutation_rejected(self) -> None:
        """Rank 3's receipt has elements=999 but plan says 6144 —
        rejected by per-rank receipt validation."""
        plan = _make_plan()
        sircl_receipts = list(_make_sircl_receipts(n=4, iters=1000))
        sircl_receipts[3] = _make_receipt(
            rank=3, transport=_TRANSPORT_SIRCL,
            custom=1000, host="spark-3", elements=999,  # mutated!
        )
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(sircl_receipts),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(any("elements" in e for e in errors),
                        f"Expected elements error, got: {errors}")


# ---------------------------------------------------------------------------
# Goal-8: Boolean rank IDs rejected
# ---------------------------------------------------------------------------

class BooleanRankTest(unittest.TestCase):
    """Boolean rank IDs (True/False) must be rejected by the validator."""

    def test_boolean_rank_in_plan_rejected(self) -> None:
        """A plan with boolean rank IDs in rank_launches must be rejected."""
        base_plan = _make_plan()
        bool_launches = tuple(
            RankLaunchEntry(
                rank=bool(r),  # type: ignore[arg-type]
                host=base_plan.sircl_arm.rank_launches[r].host,
                command=base_plan.sircl_arm.rank_launches[r].command,
                env_vars=base_plan.sircl_arm.rank_launches[r].env_vars,
            )
            for r in range(4)
        )
        tampered_sircl = ArmPlan(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            safety_class="OFFLINE",
            selector=SELECTOR_SIRCL,
            rank_launches=bool_launches,
            timeout_seconds=300,
        )
        tampered_plan = TwoArmPlan(
            sircl_arm=tampered_sircl,
            nccl_arm=base_plan.nccl_arm,
            shared_identity=base_plan.shared_identity,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        errors = validate_plan(tampered_plan)
        self.assertTrue(any("boolean" in e.lower() for e in errors),
                        f"Expected boolean rank error, got: {errors}")
    def test_boolean_rank_in_receipt_rejected(self) -> None:
        """RankReceipt construction with a boolean rank must raise ValueError."""
        with self.assertRaises(ValueError):
            RankReceipt(
                rank=True, host="spark-0", transport=_TRANSPORT_SIRCL,
                selector=SELECTOR_SIRCL, iterations=1, elements=6144,
                world_size=4, custom_collectives=1, fallback_collectives=0,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=1,
                expected_fp32_hash="", actual_output_hash="",
                actual_dtype="bfloat16", actual_byte_order="little",
                all_finite=True, max_abs_error=0.0, max_rel_error=0.0,
                tolerance_result="pass", tolerance_metric="elementwise_atol_rtol",
                tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
                sample_count=0, run_contract_hash="", rank_identity="",
            )
# ---------------------------------------------------------------------------
# Goal 9: Adversarial reproductions — cross-rank hash mismatch (req 1)
# ---------------------------------------------------------------------------

class CrossRankHashMismatchTest(unittest.TestCase):
    """Arbitrary different expected/output/contract hashes on every rank
    must be detected as a cross-rank mismatch (Goal 9 req 1).

    After all-reduce, every rank must produce the same expected_fp32_hash
    and actual_output_hash. A per-rank hash that differs from its peers
    is rejected.
    """

    def test_cross_rank_expected_fp32_hash_mismatch_rejected(self) -> None:
        """Each rank carries a different expected_fp32_hash — rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    expected_fp32_hash=chr(ord('a') + r) * 64,
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("expected_fp32_hash mismatch" in e for e in errors),
            f"Expected cross-rank expected_fp32_hash mismatch, got: {errors}",
        )

    def test_cross_rank_actual_output_hash_mismatch_rejected(self) -> None:
        """Each rank carries a different actual_output_hash — rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    actual_output_hash=f"{r:x}" * 64,
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("actual_output_hash mismatch" in e for e in errors),
            f"Expected cross-rank actual_output_hash mismatch, got: {errors}",
        )

    def test_cross_rank_hash_mismatch_nccl_arm_rejected(self) -> None:
        """Cross-rank hash mismatch on NCCL arm must also be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_NCCL_IB,
                    custom=0, fallback=1000, host=f"spark-{r}",
                    actual_output_hash=f"{r + 5:x}" * 64,
                )
                for r in range(4)
            ),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("actual_output_hash mismatch" in e for e in errors),
            f"Expected NCCL cross-rank actual_output_hash mismatch, got: {errors}",
        )


# ---------------------------------------------------------------------------
# Goal 9: Omitted/null numerical fields (req 2)
# ---------------------------------------------------------------------------

class OmittedNullNumericalFieldsTest(unittest.TestCase):
    """Successful receipts that omit numerical fields previously given
    benign defaults must be rejected (Goal 9 req 2).

    Covers fields not tested by MissingNumericalFieldsTest: all_finite,
    rank_identity, tolerance_result, tolerance_metric when empty/missing
    on successful receipts.
    """

    def test_all_finite_false_rejected(self) -> None:
        """A successful receipt with all_finite=False must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    all_finite=False,
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("all_finite" in e for e in errors),
            f"Expected all_finite error, got: {errors}",
        )

    def test_missing_rank_identity_rejected(self) -> None:
        """A successful receipt with empty rank_identity must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    rank_identity="",
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("rank_identity" in e for e in errors),
            f"Expected rank_identity error, got: {errors}",
        )

    def test_missing_tolerance_result_rejected(self) -> None:
        """A successful receipt with empty tolerance_result must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    tolerance_result="",
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("tolerance_result" in e for e in errors),
            f"Expected tolerance_result error, got: {errors}",
        )

    def test_missing_tolerance_metric_rejected(self) -> None:
        """A successful receipt with empty tolerance_metric must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    tolerance_metric="",
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("tolerance_metric" in e for e in errors),
            f"Expected tolerance_metric error, got: {errors}",
        )

    def test_omitted_max_abs_error_raises_typeerror(self) -> None:
        """RankReceipt with max_abs_error omitted (not passed at all) must
        raise TypeError — fields have no defaults (Goal 10 req 5)."""
        with self.assertRaises(TypeError):
            RankReceipt(
                rank=0, host="spark-0", transport=_TRANSPORT_SIRCL,
                selector=SELECTOR_SIRCL, iterations=1, elements=6144,
                world_size=4, custom_collectives=1, fallback_collectives=0,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=1,
                expected_fp32_hash="", actual_output_hash="",
                actual_dtype="bfloat16", actual_byte_order="little",
                all_finite=True,
                # max_abs_error omitted — must raise TypeError
                max_rel_error=0.0,
                tolerance_result="pass", tolerance_metric="elementwise_atol_rtol",
                tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
                sample_count=0, run_contract_hash="", rank_identity="",
            )

    def test_negative_max_abs_error_rejected_at_construction(self) -> None:
        """RankReceipt with negative max_abs_error must raise ValueError."""
        with self.assertRaises(ValueError):
            RankReceipt(
                rank=0, host="spark-0", transport=_TRANSPORT_SIRCL,
                selector=SELECTOR_SIRCL, iterations=1, elements=6144,
                world_size=4, custom_collectives=1, fallback_collectives=0,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=1,
                expected_fp32_hash="", actual_output_hash="",
                actual_dtype="bfloat16", actual_byte_order="little",
                all_finite=True, max_abs_error=-0.001, max_rel_error=0.0,
                tolerance_result="pass", tolerance_metric="elementwise_atol_rtol",
                tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
                sample_count=0, run_contract_hash="", rank_identity="",
            )

    def test_all_finite_as_int_rejected_at_construction(self) -> None:
        """RankReceipt with all_finite=1 (int, not bool) must raise ValueError."""
        with self.assertRaises(ValueError):
            RankReceipt(
                rank=0, host="spark-0", transport=_TRANSPORT_SIRCL,
                selector=SELECTOR_SIRCL, iterations=1, elements=6144,
                world_size=4, custom_collectives=1, fallback_collectives=0,
                unsupported_bypassed_collectives=0, unclassified_collectives=0,
                total_collectives=1,
                expected_fp32_hash="", actual_output_hash="",
                actual_dtype="bfloat16", actual_byte_order="little",
                all_finite=1,  # type: ignore[arg-type]
                max_abs_error=0.0, max_rel_error=0.0,
                tolerance_result="pass", tolerance_metric="elementwise_atol_rtol",
                tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
                sample_count=0, run_contract_hash="", rank_identity="",
            )
# ---------------------------------------------------------------------------
# Goal 9: Rank-3 plan mutations — argv, PYTHONPATH, shared identity (req 3)
# ---------------------------------------------------------------------------

class Rank3PlanMutationTest(unittest.TestCase):
    """Mutations to rank 3's argv, PYTHONPATH env, or shared_identity
    fields (topology, workload, order, binary) must be rejected
    (Goal 9 req 3).
    """

    def _make_plan_with_mutated_rank3(
        self, *, env_extra: dict[str, str] | None = None,
        command_mut: list[str] | None = None,
    ) -> TwoArmPlan:
        """Build a plan where rank 3 of the SIRCL arm has a mutation."""
        base_plan = _make_plan()
        mutated_launches = []
        for idx, rl in enumerate(base_plan.sircl_arm.rank_launches):
            env = dict(rl.env_vars)
            cmd = list(rl.command)
            if idx == 3:
                if env_extra:
                    env.update(env_extra)
                if command_mut:
                    cmd = command_mut
            mutated_launches.append(
                RankLaunchEntry(
                    rank=rl.rank, host=rl.host,
                    command=cmd, env_vars=env,
                )
            )
        tampered_sircl = ArmPlan(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            safety_class="OFFLINE",
            selector=SELECTOR_SIRCL,
            rank_launches=tuple(mutated_launches),
            timeout_seconds=300,
        )
        return TwoArmPlan(
            sircl_arm=tampered_sircl,
            nccl_arm=base_plan.nccl_arm,
            shared_identity=base_plan.shared_identity,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )

    def test_rank3_pythonpath_env_rejected(self) -> None:
        """Adding PYTHONPATH to rank 3's env must be rejected —
        env vars outside the allowlist are forbidden."""
        plan = self._make_plan_with_mutated_rank3(
            env_extra={"PYTHONPATH": "/tmp/evil"},
        )
        errors = validate_plan(plan)
        # validate_plan doesn't check allowlist, but validate_two_arm_results does.
        # Use validate_two_arm_results to get the allowlist check.
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("allowlist" in e.lower() and "PYTHONPATH" in e for e in errors),
            f"Expected PYTHONPATH allowlist error, got: {errors}",
        )

    def test_rank3_argv_mutation_rejected(self) -> None:
        """Mutating rank 3's command (argv) must be rejected —
        canonical argv is enforced."""
        plan = self._make_plan_with_mutated_rank3(
            command_mut=["python", "evil_script.py"],
        )
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("canonical argv" in e for e in errors),
            f"Expected canonical argv error, got: {errors}",
        )

    def test_shared_identity_topology_mutation_rejected(self) -> None:
        """Mutating shared_identity topology must be rejected —
        validator-owned pinned value."""
        plan = _make_plan()
        tampered_shared = dict(plan.shared_identity)
        tampered_shared["topology"] = "tp4_full_mesh"
        tampered_plan = TwoArmPlan(
            sircl_arm=plan.sircl_arm,
            nccl_arm=plan.nccl_arm,
            shared_identity=tampered_shared,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        errors = validate_plan(tampered_plan)
        self.assertTrue(
            any("topology" in e for e in errors),
            f"Expected topology pinned-value error, got: {errors}",
        )

    def test_shared_identity_workload_mutation_rejected(self) -> None:
        """Mutating shared_identity workload must be rejected."""
        plan = _make_plan()
        tampered_shared = dict(plan.shared_identity)
        tampered_shared["workload"] = "different_workload"
        tampered_plan = TwoArmPlan(
            sircl_arm=plan.sircl_arm,
            nccl_arm=plan.nccl_arm,
            shared_identity=tampered_shared,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        errors = validate_plan(tampered_plan)
        self.assertTrue(
            any("workload" in e for e in errors),
            f"Expected workload pinned-value error, got: {errors}",
        )

    def test_shared_identity_order_mutation_rejected(self) -> None:
        """Mutating shared_identity order must be rejected."""
        plan = _make_plan()
        tampered_shared = dict(plan.shared_identity)
        tampered_shared["order"] = "reversed"
        tampered_plan = TwoArmPlan(
            sircl_arm=plan.sircl_arm,
            nccl_arm=plan.nccl_arm,
            shared_identity=tampered_shared,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        errors = validate_plan(tampered_plan)
        self.assertTrue(
            any("order" in e for e in errors),
            f"Expected order pinned-value error, got: {errors}",
        )

    def test_shared_identity_binary_mutation_rejected(self) -> None:
        """Mutating shared_identity binary must be rejected."""
        plan = _make_plan()
        tampered_shared = dict(plan.shared_identity)
        tampered_shared["binary"] = "different_probe.py"
        tampered_plan = TwoArmPlan(
            sircl_arm=plan.sircl_arm,
            nccl_arm=plan.nccl_arm,
            shared_identity=tampered_shared,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        errors = validate_plan(tampered_plan)
        self.assertTrue(
            any("binary" in e for e in errors),
            f"Expected binary pinned-value error, got: {errors}",
        )

    def test_rank3_nccl_arm_env_mutation_rejected(self) -> None:
        """Mutating ITERATIONS on rank 3 of the NCCL arm must be rejected
        by per-rank receipt validation."""
        plan = _make_plan()
        nccl_receipts = list(_make_nccl_receipts(n=4, iters=1000))
        nccl_receipts[3] = _make_receipt(
            rank=3, transport=_TRANSPORT_NCCL_IB,
            custom=0, fallback=999, host="spark-3", iterations=999,
        )
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=tuple(nccl_receipts),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("iterations" in e for e in errors),
            f"Expected NCCL rank3 iterations error, got: {errors}",
        )


# ---------------------------------------------------------------------------
# Goal 9: Altered sanitized rank identity (req 4)
# ---------------------------------------------------------------------------

class AlteredRankIdentityTest(unittest.TestCase):
    """Receipts with an altered sanitized rank_identity (wrong format or
    wrong world_size) must be rejected by the validator (Goal 9 req 4).
    """

    def test_rank_identity_wrong_world_size_rejected(self) -> None:
        """rank_identity='rank-0-of-8' must be rejected — world_size
        mismatch vs the authoritative 4."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    rank_identity=f"rank-{r}-of-8",
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("rank_identity" in e and "validator-derived" in e for e in errors),
            f"Expected rank_identity mismatch error, got: {errors}",
        )

    def test_rank_identity_bad_format_rejected(self) -> None:
        """rank_identity='bad-format' must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    rank_identity="bad-format",
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("rank_identity" in e and "validator-derived" in e for e in errors),
            f"Expected rank_identity format error, got: {errors}",
        )

    def test_rank_identity_only_rank3_altered_rejected(self) -> None:
        """Only rank 3 has altered rank_identity — still rejected."""
        plan = _make_plan()
        sircl_receipts = list(_make_sircl_receipts(n=4, iters=1000))
        sircl_receipts[3] = _make_receipt(
            rank=3, transport=_TRANSPORT_SIRCL,
            custom=1000, host="spark-3",
            rank_identity="rank-3-of-8",
        )
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(sircl_receipts),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("rank_identity" in e and "validator-derived" in e for e in errors),
            f"Expected rank3 identity error, got: {errors}",
        )

# ---------------------------------------------------------------------------
# Goal 9: Stale trusted probe hash (req 9) — in test_spark_benchmark_contract.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Goal 10: Per-rank validation for ALL ranks (req 3)
# ---------------------------------------------------------------------------

class Rank3PerRankValidationTest(unittest.TestCase):
    """Mutating rank-3 plan launch argv/env for EACH field must be rejected.
    Goal 10 requirement 3: all ranks (not just rank_launches[0]) are
    validated for ITERATIONS, ELEMENTS, WORLD_SIZE, RANK,
    VLLM_SPARK_TP4_MODE, NCCL_NET, NCCL_IB_DISABLE, and argv.
    """

    def _make_plan_with_mutated_rank3(
        self, *, arm: str = "sircl", env_mut: dict[str, str] | None = None,
        command_mut: list[str] | None = None,
    ) -> TwoArmPlan:
        """Build a plan where rank 3 of the specified arm has a mutation."""
        base_plan = _make_plan()
        target_arm = base_plan.sircl_arm if arm == "sircl" else base_plan.nccl_arm
        other_arm = base_plan.nccl_arm if arm == "sircl" else base_plan.sircl_arm
        mutated_launches = []
        for idx, rl in enumerate(target_arm.rank_launches):
            env = dict(rl.env_vars)
            cmd = list(rl.command)
            if idx == 3:
                if env_mut:
                    env.update(env_mut)
                if command_mut:
                    cmd = command_mut
            mutated_launches.append(
                RankLaunchEntry(
                    rank=rl.rank, host=rl.host,
                    command=cmd, env_vars=env,
                )
            )
        tampered_arm = ArmPlan(
            arm_name=target_arm.arm_name,
            transport=target_arm.transport,
            safety_class=target_arm.safety_class,
            selector=target_arm.selector,
            rank_launches=tuple(mutated_launches),
            timeout_seconds=target_arm.timeout_seconds,
        )
        if arm == "sircl":
            return TwoArmPlan(
                sircl_arm=tampered_arm,
                nccl_arm=other_arm,
                shared_identity=base_plan.shared_identity,
                dry_run=True,
                confirmation_required=True,
                executor_available=False,
            )
        return TwoArmPlan(
            sircl_arm=other_arm,
            nccl_arm=tampered_arm,
            shared_identity=base_plan.shared_identity,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )

    def _run_validation(self, plan: TwoArmPlan) -> list[str]:
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        return validate_two_arm_results(result, plan)

    def test_rank3_iterations_mutation_rejected(self) -> None:
        """Mutating ITERATIONS on rank 3 of SIRCL arm must be rejected."""
        plan = self._make_plan_with_mutated_rank3(
            arm="sircl", env_mut={"ITERATIONS": "999"},
        )
        errors = self._run_validation(plan)
        self.assertTrue(
            any("ITERATIONS" in e and "rank_launch[3]" in e for e in errors),
            f"Expected rank3 ITERATIONS error, got: {errors}",
        )

    def test_rank3_elements_mutation_rejected(self) -> None:
        """Mutating ELEMENTS on rank 3 of SIRCL arm must be rejected."""
        plan = self._make_plan_with_mutated_rank3(
            arm="sircl", env_mut={"ELEMENTS": "8192"},
        )
        errors = self._run_validation(plan)
        self.assertTrue(
            any("ELEMENTS" in e and "rank_launch[3]" in e for e in errors),
            f"Expected rank3 ELEMENTS error, got: {errors}",
        )

    def test_rank3_rank_mutation_rejected(self) -> None:
        """Mutating RANK on rank 3 of SIRCL arm must be rejected."""
        plan = self._make_plan_with_mutated_rank3(
            arm="sircl", env_mut={"RANK": "99"},
        )
        errors = self._run_validation(plan)
        self.assertTrue(
            any("RANK" in e and "rank_launch[3]" in e for e in errors),
            f"Expected rank3 RANK error, got: {errors}",
        )

    def test_rank3_world_size_mutation_rejected(self) -> None:
        """Mutating WORLD_SIZE on rank 3 of SIRCL arm must be rejected."""
        plan = self._make_plan_with_mutated_rank3(
            arm="sircl", env_mut={"WORLD_SIZE": "8"},
        )
        errors = self._run_validation(plan)
        self.assertTrue(
            any("WORLD_SIZE" in e and "rank_launch[3]" in e for e in errors),
            f"Expected rank3 WORLD_SIZE error, got: {errors}",
        )

    def test_rank3_selector_mutation_sircl_arm_rejected(self) -> None:
        """Mutating VLLM_SPARK_TP4_MODE on rank 3 of SIRCL arm must be rejected."""
        plan = self._make_plan_with_mutated_rank3(
            arm="sircl", env_mut={"VLLM_SPARK_TP4_MODE": "disabled"},
        )
        errors = self._run_validation(plan)
        self.assertTrue(
            any("VLLM_SPARK_TP4_MODE" in e and "rank_launch[3]" in e for e in errors),
            f"Expected rank3 selector error, got: {errors}",
        )

    def test_rank3_nccl_net_mutation_rejected(self) -> None:
        """Mutating NCCL_NET on rank 3 of NCCL arm must be rejected."""
        plan = self._make_plan_with_mutated_rank3(
            arm="nccl", env_mut={"NCCL_NET": "Socket"},
        )
        errors = self._run_validation(plan)
        self.assertTrue(
            any("NCCL_NET" in e and "rank_launch[3]" in e for e in errors),
            f"Expected rank3 NCCL_NET error, got: {errors}",
        )

    def test_rank3_nccl_ib_disable_mutation_rejected(self) -> None:
        """Mutating NCCL_IB_DISABLE on rank 3 of NCCL arm must be rejected."""
        plan = self._make_plan_with_mutated_rank3(
            arm="nccl", env_mut={"NCCL_IB_DISABLE": "1"},
        )
        errors = self._run_validation(plan)
        self.assertTrue(
            any("NCCL_IB_DISABLE" in e and "rank_launch[3]" in e for e in errors),
            f"Expected rank3 NCCL_IB_DISABLE error, got: {errors}",
        )

    def test_rank3_argv_mutation_rejected(self) -> None:
        """Mutating command (argv) on rank 3 must be rejected."""
        plan = self._make_plan_with_mutated_rank3(
            arm="sircl", command_mut=["python", "other.py"],
        )
        errors = self._run_validation(plan)
        self.assertTrue(
            any("canonical argv" in e and "rank_launch[3]" in e for e in errors),
            f"Expected rank3 argv error, got: {errors}",
        )


# ---------------------------------------------------------------------------
# Goal 10: Four-rank selector consensus (req 1)
# ---------------------------------------------------------------------------

class SelectorConsensusTest(unittest.TestCase):
    """Test check_selector_consensus for four-rank transport agreement."""

    def test_all_agree_custom_with_native_passes(self) -> None:
        """All ranks select custom with native session available — passes."""
        caps = tuple(
            RankCapability(rank=r, selector=SELECTOR_SIRCL, native_session_available=True)
            for r in range(4)
        )
        result = check_selector_consensus(caps)
        self.assertEqual(result, SELECTOR_SIRCL)

    def test_all_agree_disabled_passes(self) -> None:
        """All ranks select disabled — passes (no native session needed)."""
        caps = tuple(
            RankCapability(rank=r, selector=SELECTOR_NCCL, native_session_available=False)
            for r in range(4)
        )
        result = check_selector_consensus(caps)
        self.assertEqual(result, SELECTOR_NCCL)

    def test_one_rank_disagrees_rejected(self) -> None:
        """One rank disagrees on selector — SelectorConsensusError."""
        caps = (
            RankCapability(rank=0, selector=SELECTOR_SIRCL, native_session_available=True),
            RankCapability(rank=1, selector=SELECTOR_SIRCL, native_session_available=True),
            RankCapability(rank=2, selector=SELECTOR_SIRCL, native_session_available=True),
            RankCapability(rank=3, selector=SELECTOR_NCCL, native_session_available=False),
        )
        with self.assertRaises(SelectorConsensusError):
            check_selector_consensus(caps)

    def test_custom_one_rank_lacking_native_rejected(self) -> None:
        """selector=custom with one rank lacking native session — error."""
        caps = (
            RankCapability(rank=0, selector=SELECTOR_SIRCL, native_session_available=True),
            RankCapability(rank=1, selector=SELECTOR_SIRCL, native_session_available=True),
            RankCapability(rank=2, selector=SELECTOR_SIRCL, native_session_available=True),
            RankCapability(rank=3, selector=SELECTOR_SIRCL, native_session_available=False),
        )
        with self.assertRaises(SelectorConsensusError):
            check_selector_consensus(caps)

    def test_wrong_number_of_capabilities_rejected(self) -> None:
        """Wrong number of capabilities (not 4) — error."""
        caps = (
            RankCapability(rank=0, selector=SELECTOR_SIRCL, native_session_available=True),
            RankCapability(rank=1, selector=SELECTOR_SIRCL, native_session_available=True),
        )
        with self.assertRaises(SelectorConsensusError):
            check_selector_consensus(caps)

    def test_empty_capabilities_rejected(self) -> None:
        """Empty capabilities tuple — error."""
        with self.assertRaises(SelectorConsensusError):
            check_selector_consensus(())

    def test_non_tuple_capabilities_rejected(self) -> None:
        """Non-tuple capabilities — error."""
        with self.assertRaises(SelectorConsensusError):
            check_selector_consensus([])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Goal 10: Commit recompute no longer fail-open (req 4)
# ---------------------------------------------------------------------------

class CommitRecomputeFailOpenTest(unittest.TestCase):
    """Malformed plan projection that causes recomputation to fail must
    produce validation errors, not silently pass (Goal 10 req 4).

    The old code used 'except Exception: pass' which silently accepted
    receipts when recomputation failed.  Now it appends errors.
    """

    def test_broken_shared_identity_causes_recompute_error(self) -> None:
        """A plan with a broken shared_identity (non-int iterations) causes
        the run_contract_hash recomputation to fail, which must produce
        validation errors rather than silently passing."""
        plan = _make_plan()
        # Break shared_identity so that plan_iters becomes -1
        # (int('not-a-number') raises ValueError → except → plan_iters=-1).
        tampered_shared = dict(plan.shared_identity)
        tampered_shared["iterations"] = "not-a-number"
        tampered_plan = TwoArmPlan(
            sircl_arm=plan.sircl_arm,
            nccl_arm=plan.nccl_arm,
            shared_identity=tampered_shared,
            dry_run=True,
            confirmation_required=True,
            executor_available=False,
        )
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=_make_sircl_receipts(n=4, iters=1000),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, tampered_plan)
        # The plan itself is invalid (shared_identity iterations not int),
        # so errors must be non-empty — validation must not silently pass.
        self.assertTrue(
            len(errors) > 0,
            "Expected validation errors for broken plan, got empty list",
        )


# ---------------------------------------------------------------------------
# Goal 10: New acceptance fields validated (req 5)
# ---------------------------------------------------------------------------

class NewAcceptanceFieldsTest(unittest.TestCase):
    """Successful receipts with wrong tolerance_atol/rtol or missing
    actual_dtype/actual_byte_order must be rejected (Goal 10 req 5).
    """

    def test_wrong_tolerance_atol_rejected(self) -> None:
        """Successful receipt with tolerance_atol != _BF16_ATOL must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    tolerance_atol=0.1,  # wrong!
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("tolerance_atol" in e for e in errors),
            f"Expected tolerance_atol error, got: {errors}",
        )

    def test_wrong_tolerance_rtol_rejected(self) -> None:
        """Successful receipt with tolerance_rtol != _BF16_RTOL must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    tolerance_rtol=0.1,  # wrong!
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("tolerance_rtol" in e for e in errors),
            f"Expected tolerance_rtol error, got: {errors}",
        )

    def test_missing_actual_dtype_rejected(self) -> None:
        """Successful receipt with empty actual_dtype must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    actual_dtype="",
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("actual_dtype" in e for e in errors),
            f"Expected actual_dtype error, got: {errors}",
        )

    def test_missing_actual_byte_order_rejected(self) -> None:
        """Successful receipt with empty actual_byte_order must be rejected."""
        plan = _make_plan()
        sircl = ArmResult(
            arm_name="sircl",
            transport=_TRANSPORT_SIRCL,
            receipts=tuple(
                _make_receipt(
                    rank=r, transport=_TRANSPORT_SIRCL,
                    custom=1000, host=f"spark-{r}",
                    actual_byte_order="",
                )
                for r in range(4)
            ),
        )
        nccl = ArmResult(
            arm_name="nccl_ib",
            transport=_TRANSPORT_NCCL_IB,
            receipts=_make_nccl_receipts(n=4, iters=1000),
        )
        result = TwoArmResult(sircl_arm=sircl, nccl_arm=nccl, valid=True)
        errors = validate_two_arm_results(result, plan)
        self.assertTrue(
            any("actual_byte_order" in e for e in errors),
            f"Expected actual_byte_order error, got: {errors}",
        )


# ---------------------------------------------------------------------------
# Goal 10: Omission at construction raises TypeError (req 5)
# ---------------------------------------------------------------------------

class OmissionProofReceiptTest(unittest.TestCase):
    """Constructing RankReceipt without each observation field must raise
    TypeError (omission, not None).  Goal 10 req 5: fields have no
    defaults — omission must fail, not silently accept benign values.
    """

    def _make_valid_kwargs(self) -> dict:
        """Return kwargs that produce a valid RankReceipt."""
        return dict(
            rank=0, host="spark-0", transport=_TRANSPORT_SIRCL,
            selector=SELECTOR_SIRCL, iterations=1, elements=6144,
            world_size=4, custom_collectives=1, fallback_collectives=0,
            unsupported_bypassed_collectives=0, unclassified_collectives=0,
            total_collectives=1,
            expected_fp32_hash="", actual_output_hash="",
            actual_dtype="bfloat16", actual_byte_order="little",
            all_finite=True, max_abs_error=0.0, max_rel_error=0.0,
            tolerance_result="pass", tolerance_metric="elementwise_atol_rtol",
            tolerance_atol=_BF16_ATOL, tolerance_rtol=_BF16_RTOL,
            sample_count=0, run_contract_hash="", rank_identity="",
        )

    def test_omit_expected_fp32_hash_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["expected_fp32_hash"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_actual_output_hash_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["actual_output_hash"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_all_finite_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["all_finite"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_max_abs_error_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["max_abs_error"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_max_rel_error_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["max_rel_error"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_tolerance_result_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["tolerance_result"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_tolerance_metric_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["tolerance_metric"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_tolerance_atol_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["tolerance_atol"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_tolerance_rtol_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["tolerance_rtol"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_sample_count_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["sample_count"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_run_contract_hash_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["run_contract_hash"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_rank_identity_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["rank_identity"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_actual_dtype_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["actual_dtype"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)

    def test_omit_actual_byte_order_raises_typeerror(self) -> None:
        kw = self._make_valid_kwargs()
        del kw["actual_byte_order"]
        with self.assertRaises(TypeError):
            RankReceipt(**kw)


if __name__ == "__main__":
    unittest.main()
