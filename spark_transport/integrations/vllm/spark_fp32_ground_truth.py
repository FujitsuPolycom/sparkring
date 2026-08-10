"""CPU-offline modeled BF16 reduction-order analysis.

**This module does not execute SIRCL, CUDA ``__hadd``, RDMA, or NCCL.**
It computes BF16 reductions in Python/PyTorch on the CPU using the
*same addition order* that the native CUDA kernel
(``gpu_tp4_tensor.cu``) uses, and compares them against a
correctly-rounded FP32 ground-truth sum.

It is a **modeled reduction-order analysis**, not a measurement of any
transport.  The "sequential BF16 sum" is a naive left-to-right
accumulation included for comparison only; **NCCL's actual reduction
order depends on algorithm, channel slicing, topology, and runtime
configuration and is not established or modeled here.**

The module validates two categories of property:

1. **Modeled invariant checks** (pass/fail):
   - All rank outputs of the modeled ring are identical to each other.
   - All values in every iteration are finite (not NaN/Inf).

   No other invariants are modeled.  Reported metrics (below) carry no
   pass/fail gate.

2. **Reported metrics** (no pass/fail gate on numerical quality):
   - MAE, RMSE, max absolute error vs **unrounded FP32** ground truth.
   - Exact agreement count with FP32 truth rounded once to BF16.
   - Per-element wins of ring-order vs sequential-order.
   - Candidate-vs-reference BF16 mismatches.
   - Outside-tolerance count governed by an explicit tolerance policy.

The module does **not** promote or certify native correctness.  A
modeled invariant pass means the *model* is self-consistent, not that
the native kernel or NCCL is correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

# ---------------------------------------------------------------------------
# Constants matching the native transport and live audit
# ---------------------------------------------------------------------------

ELEMENTS = 6144
BF16_BYTES = 2
Q1_BYTES = ELEMENTS * BF16_BYTES  # 12 288

VALID_PATTERNS = frozenset({"random", "uniform", "cancellation"})
VALID_DTYPES_STR = {"torch.bfloat16"}

# Representative decode shapes (query rows x 6144 width, BF16)
DECODE_SHAPES: tuple[tuple[int, int], ...] = (
    (1, ELEMENTS),
    (2, ELEMENTS),
    (3, ELEMENTS),
    (4, ELEMENTS),
    (5, ELEMENTS),
)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class AuditInputError(ValueError):
    """Stable error for invalid audit inputs (no traceback from CLI)."""


def _validate_int(value: int, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise AuditInputError(f"{name} must be an integer, got bool")
    if not isinstance(value, int):
        raise AuditInputError(f"{name} must be an integer, got {type(value).__name__}")
    if value < minimum or value > maximum:
        raise AuditInputError(
            f"{name} must be in [{minimum}, {maximum}], got {value}"
        )
    return value


def _validate_pattern(pattern: str) -> str:
    if pattern not in VALID_PATTERNS:
        raise AuditInputError(
            f"pattern must be one of {sorted(VALID_PATTERNS)}, got '{pattern}'"
        )
    return pattern


def _validate_tensor(tensor: torch.Tensor, *, name: str, rows: int, width: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise AuditInputError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != torch.bfloat16:
        raise AuditInputError(
            f"{name} dtype must be torch.bfloat16, got {tensor.dtype}"
        )
    if tensor.shape != (rows * width,):
        raise AuditInputError(
            f"{name} shape must be ({rows * width},), got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise AuditInputError(f"{name} contains nonfinite values")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuditCase:
    """One analysis case: a shape, world size, and input pattern."""

    name: str
    rows: int
    width: int
    world_size: int
    pattern: str

    def __post_init__(self) -> None:
        _validate_int(self.rows, "rows", minimum=1, maximum=512)
        _validate_int(self.width, "width", minimum=1, maximum=65536)
        _validate_int(self.world_size, "world_size", minimum=2, maximum=4)
        _validate_pattern(self.pattern)


@dataclass
class InvariantCheck:
    """Modeled invariant check result (pass/fail).

    Two invariants are modeled:
    - all_ranks_identical: all rank outputs of the modeled ring are equal.
    - all_finite: every output element is finite (not NaN/Inf).

    No other invariants are modeled — reported metrics in QualityMetrics
    are reported without any pass/fail gate.
    """

    all_ranks_identical: bool
    all_finite: bool
    failure_reason: str = ""

    @property
    def passed(self) -> bool:
        return self.all_ranks_identical and self.all_finite


@dataclass
class QualityMetrics:
    """Reported numerical quality metrics (no pass/fail gate).

    Continuous error metrics (MAE, RMSE, max_abs) are computed
    against the **unrounded FP32** ground truth.

    Mismatch counts:
    - ``candidate_truth_mismatches``: element-wise comparisons where
      the candidate output differs from the FP32 truth **rounded once
      to BF16**.  This is the exact-agreement check — must be zero
      for a correctly-rounded result.
    - ``candidate_sequential_mismatches``: element-wise comparisons
      where the candidate output differs from the sequential BF16 sum
      (naive left-to-right accumulation).  This is NOT a truth
      comparison — it shows reduction-order sensitivity.

    ``outside_tolerance_count`` is governed by an explicit
    tolerance policy (``bf16_fixed_abs_v1``: fixed absolute threshold
    of 2^{-7}).  If no policy is active, it is reported as
    ``NOT_JUDGED`` (the integer constant ``-1``).
    """

    candidate_mae: float
    candidate_rmse: float
    candidate_max_abs: float
    candidate_exact_agreement: int
    reference_mae: float
    reference_rmse: float
    reference_max_abs: float
    reference_exact_agreement: int
    candidate_closer: int
    reference_closer: int
    tied: int
    candidate_truth_mismatches: int
    candidate_sequential_mismatches: int
    outside_tolerance_count: int
    tolerance_policy: str


# Sentinel for NOT-JUDGED outside-tolerance count.
NOT_JUDGED = -1

# Default tolerance policy: elements whose absolute error exceeds
# this fixed absolute threshold relative to the FP32 truth are
# counted as outside-tolerance.  The policy is explicit and versioned.
#
# NOTE: This is a FIXED ABSOLUTE tolerance, not a magnitude-dependent
# BF16 ULP policy.  A real BF16 ULP would depend on the exponent of
# each value (zero, subnormal, normal, finite extremes).  The name
# reflects the actual policy: a fixed absolute threshold of 2^{-7},
# which equals one BF16 ULP at magnitude ~1 but is constant across
# all magnitudes.  The formula is: |observed - truth_fp32| > 2^{-7}.
# Units: absolute (unitless, matching the BF16 tensor values).
TOLERANCE_THRESHOLD_BF16 = 0.0078125  # 2^{-7}, fixed absolute threshold
TOLERANCE_POLICY_NAME = "bf16_fixed_abs_v1"


@dataclass
class AuditResult:
    """Result of one analysis case: invariants (pass/fail) + metrics (reported)."""

    case: AuditCase
    iterations: int
    compared_elements: int
    invariants: InvariantCheck
    metrics: QualityMetrics

    @property
    def passed(self) -> bool:
        return self.invariants.passed


@dataclass
class AuditSummary:
    """Aggregate summary across all cases."""

    results: list[AuditResult] = field(default_factory=list)
    total_compared: int = 0
    total_passed: int = 0
    total_failed: int = 0

    @property
    def all_invariants_passed(self) -> bool:
        return self.total_failed == 0 and self.total_passed > 0


# ---------------------------------------------------------------------------
# Deterministic input generation (mirrors tp4_numerical_audit.py)
# ---------------------------------------------------------------------------

def make_rank_input(
    sequence: int,
    rank: int,
    world_size: int,
    rows: int,
    width: int,
    pattern: str,
) -> torch.Tensor:
    """Generate deterministic BF16 inputs for one rank and sequence.

    Patterns:
    - "random": broad-scale values with exponents cycling -6..+5.
    - "uniform": all elements equal to a rank-dependent value.
    - "cancellation": shared component with alternating sign coefficients.
    """
    _validate_int(sequence, "sequence", minimum=0, maximum=10**9)
    _validate_int(rank, "rank", minimum=0, maximum=3)
    _validate_int(world_size, "world_size", minimum=2, maximum=4)
    _validate_int(rows, "rows", minimum=1, maximum=512)
    _validate_int(width, "width", minimum=1, maximum=65536)
    _validate_pattern(pattern)
    if rank >= world_size:
        raise AuditInputError(f"rank {rank} must be < world_size {world_size}")

    elements = rows * width
    if pattern == "uniform":
        value = float((rank + 1) * (sequence + 1))
        return torch.full((elements,), value, dtype=torch.bfloat16)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x5A17 + sequence * world_size + rank)
    independent = torch.randn(elements, generator=generator)

    indices = torch.arange(elements)
    exponents = ((indices + sequence) % 12) - 6
    scale = torch.pow(2.0, exponents)

    if pattern == "cancellation":
        shared_generator = torch.Generator(device="cpu")
        shared_generator.manual_seed(0xC011A + sequence)
        shared = torch.randn(elements, generator=shared_generator) * scale
        coefficient = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0][rank % 6]
        value = shared * coefficient + independent * scale * 0.001
    else:
        value = independent * scale

    result = value.to(torch.bfloat16)
    _validate_tensor(result, name=f"input(seq={sequence},rank={rank})",
                     rows=rows, width=width)
    return result


# ---------------------------------------------------------------------------
# BF16 ring reduction models (CPU, matching native kernel addition order)
# ---------------------------------------------------------------------------

def tp4_ring_reduce_all_ranks(inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Compute the TP4 BF16 all-reduce for all four ranks.

    Models the addition order in ``gpu_tp4_tensor.cu``:
    round 0 (0<->1, 2<->3): r0 = hadd(local, peer)
    round 1 (0<->3, 1<->2): out = hadd(r0, peer)

    Returns a list of 4 tensors, one per rank.
    """
    if len(inputs) != 4:
        raise AuditInputError("TP4 ring reduce requires exactly 4 inputs")
    for i, t in enumerate(inputs):
        _validate_tensor(t, name=f"tp4_input[{i}]", rows=1, width=t.numel())
    round0 = [inputs[i] + inputs[i ^ 1] for i in range(4)]
    output = [round0[i] + round0[i ^ 3] for i in range(4)]
    return output


def tp2_ring_reduce_all_ranks(inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Compute the TP2 BF16 all-reduce: single exchange + fused add.

    Returns a list of 2 tensors, one per rank.
    """
    if len(inputs) != 2:
        raise AuditInputError("TP2 ring reduce requires exactly 2 inputs")
    for i, t in enumerate(inputs):
        _validate_tensor(t, name=f"tp2_input[{i}]", rows=1, width=t.numel())
    result = inputs[0] + inputs[1]
    return [result, result.clone()]


def fp32_ground_truth(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Compute the exact FP32 sum of all rank inputs.

    Returns the **unrounded FP32** sum — the true arithmetic
    reference.  Continuous error metrics (MAE, RMSE, max error)
    MUST compare observed outputs to this unrounded value, not to a
    BF16-rounded copy.  Rounding the truth to BF16 before comparing
    hides the quantization error inherent in BF16 reduction and
    produces a falsely small MAE.

    For exact-agreement and mismatch metrics, use
    ``fp32_truth_rounded_to_dtype`` which rounds this FP32 truth
    once to the target dtype.
    """
    stacked = torch.stack([tensor.float() for tensor in inputs])
    return stacked.sum(dim=0)


def fp32_truth_rounded_to_dtype(
    fp32_truth: torch.Tensor, dtype: torch.dtype,
) -> torch.Tensor:
    """Round the FP32 truth once to the target dtype.

    This is used for exact-agreement and bit-level mismatch metrics:
    the comparison is against the FP32 truth cast to the target
    dtype exactly once, not against a separately-computed BF16 sum.
    """
    return fp32_truth.to(dtype)


def sequential_bf16_sum(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Naive left-to-right BF16 accumulation.

    This is a simple comparison baseline.  It does **not** model
    NCCL's actual reduction order, which depends on algorithm,
    channel slicing, topology, and runtime configuration.
    """
    result = inputs[0].clone()
    for t in inputs[1:]:
        result = result + t
    return result


# ---------------------------------------------------------------------------
# Audit cases
# ---------------------------------------------------------------------------

def default_cases() -> list[AuditCase]:
    """Return the default analysis case matrix."""
    cases: list[AuditCase] = []
    cases.append(AuditCase(
        name="tp4_q1_random", rows=1, width=ELEMENTS,
        world_size=4, pattern="random",
    ))
    cases.append(AuditCase(
        name="tp4_q1_uniform", rows=1, width=ELEMENTS,
        world_size=4, pattern="uniform",
    ))
    cases.append(AuditCase(
        name="tp4_q1_cancellation", rows=1, width=ELEMENTS,
        world_size=4, pattern="cancellation",
    ))
    for rows in (2, 3, 4, 5):
        cases.append(AuditCase(
            name=f"tp4_q{rows}_random", rows=rows, width=ELEMENTS,
            world_size=4, pattern="random",
        ))
    cases.append(AuditCase(
        name="tp2_q1_random", rows=1, width=ELEMENTS,
        world_size=2, pattern="random",
    ))
    cases.append(AuditCase(
        name="tp2_q1_cancellation", rows=1, width=ELEMENTS,
        world_size=2, pattern="cancellation",
    ))
    return cases


# ---------------------------------------------------------------------------
# Analysis execution
# ---------------------------------------------------------------------------

def run_case(
    case: AuditCase,
    iterations: int = 200,
) -> AuditResult:
    """Run one analysis case.

    Returns invariant checks (pass/fail) and quality metrics (reported).
    Does **not** gate on numerical quality — only on modeled invariants.
    """
    _validate_int(iterations, "iterations", minimum=1, maximum=1_000_000)
    reduce_fn = (
        tp4_ring_reduce_all_ranks if case.world_size == 4
        else tp2_ring_reduce_all_ranks
    )

    compared = 0
    cand_abs_sum = 0.0
    cand_sq_sum = 0.0
    cand_max = 0.0
    cand_exact = 0
    ref_abs_sum = 0.0
    ref_sq_sum = 0.0
    ref_max = 0.0
    ref_exact = 0
    cand_closer = 0
    ref_closer = 0
    tied = 0
    outside_tol = 0
    truth_mismatches = 0
    seq_mismatches = 0


    all_ranks_identical = True
    all_finite = True
    invariant_failures: list[str] = []

    for sequence in range(iterations):
        inputs = [
            make_rank_input(
                sequence, rank, case.world_size,
                case.rows, case.width, case.pattern,
            )
            for rank in range(case.world_size)
        ]

        outputs = reduce_fn(inputs)
        truth_fp32 = fp32_ground_truth(inputs)
        truth_bf16 = fp32_truth_rounded_to_dtype(truth_fp32, torch.bfloat16)
        reference = sequential_bf16_sum(inputs)

        # Per-iteration finiteness check (not just final iteration)
        for rank_idx, out in enumerate(outputs):
            if not bool(torch.isfinite(out).all()):
                all_finite = False
                invariant_failures.append(
                    f"seq={sequence} rank={rank_idx}: nonfinite output"
                )

        # Per-iteration all-ranks-identical check
        for rank_idx in range(1, len(outputs)):
            if not torch.equal(outputs[0], outputs[rank_idx]):
                all_ranks_identical = False
                invariant_failures.append(
                    f"seq={sequence}: rank 0 != rank {rank_idx}"
                )

        # Use rank 0 output for quality metrics
        candidate = outputs[0]

        # Continuous error metrics: compare to UNROUNDED FP32 truth
        cand_err = (candidate.float() - truth_fp32).abs()
        ref_err = (reference.float() - truth_fp32).abs()

        elements = case.rows * case.width
        compared += elements
        cand_abs_sum += float(cand_err.sum().item())
        cand_sq_sum += float(cand_err.square().sum().item())
        cand_max = max(cand_max, float(cand_err.max().item()))
        # Exact-agreement: compare candidate to FP32 truth rounded once to BF16
        cand_exact += int(torch.count_nonzero(candidate == truth_bf16).item())
        ref_abs_sum += float(ref_err.sum().item())
        ref_sq_sum += float(ref_err.square().sum().item())
        ref_max = max(ref_max, float(ref_err.max().item()))
        ref_exact += int(torch.count_nonzero(reference == truth_bf16).item())
        cand_closer += int(torch.count_nonzero(cand_err < ref_err).item())
        ref_closer += int(torch.count_nonzero(ref_err < cand_err).item())
        # Truth mismatches: candidate vs FP32 truth rounded once to BF16.
        # This is the exact-agreement check — must be zero for correct rounding.
        truth_mismatches += int(
            torch.count_nonzero(
                candidate.view(torch.int16) != truth_bf16.view(torch.int16)
            ).item()
        )
        # Sequential mismatches: candidate vs sequential BF16 sum.
        # This shows reduction-order sensitivity, NOT truth disagreement.
        seq_mismatches += int(
            torch.count_nonzero(
                candidate.view(torch.int16) != reference.view(torch.int16)
            ).item()
        )
        # Outside-tolerance: absolute error exceeds BF16 ULP threshold
        outside_tol += int(
            torch.count_nonzero(cand_err > TOLERANCE_THRESHOLD_BF16).item()
        )

    cand_mae = cand_abs_sum / compared if compared else 0.0
    cand_rmse = (cand_sq_sum / compared) ** 0.5 if compared else 0.0
    ref_mae = ref_abs_sum / compared if compared else 0.0
    ref_rmse = (ref_sq_sum / compared) ** 0.5 if compared else 0.0

    failure_reason = "; ".join(invariant_failures[:5]) if invariant_failures else ""
    if not all_finite:
        failure_reason = f"nonfinite outputs detected: {failure_reason}"
    elif not all_ranks_identical:
        failure_reason = f"rank outputs differ: {failure_reason}"

    invariants = InvariantCheck(
        all_ranks_identical=all_ranks_identical,
        all_finite=all_finite,
        failure_reason=failure_reason,
    )
    metrics = QualityMetrics(
        candidate_mae=cand_mae,
        candidate_rmse=cand_rmse,
        candidate_max_abs=cand_max,
        candidate_exact_agreement=cand_exact,
        candidate_truth_mismatches=truth_mismatches,
        candidate_sequential_mismatches=seq_mismatches,
        reference_mae=ref_mae,
        reference_rmse=ref_rmse,
        reference_max_abs=ref_max,
        reference_exact_agreement=ref_exact,
        candidate_closer=cand_closer,
        reference_closer=ref_closer,
        tied=tied,
        outside_tolerance_count=outside_tol,
        tolerance_policy=TOLERANCE_POLICY_NAME,
    )

    return AuditResult(
        case=case,
        iterations=iterations,
        compared_elements=compared,
        invariants=invariants,
        metrics=metrics,
    )


def run_audit(
    cases: Sequence[AuditCase] | None = None,
    iterations: int = 200,
) -> AuditSummary:
    """Run the full analysis suite and return a summary."""
    if cases is None:
        cases = default_cases()
    _validate_int(iterations, "iterations", minimum=1, maximum=1_000_000)
    summary = AuditSummary()
    for case in cases:
        result = run_case(case, iterations)
        summary.results.append(result)
        summary.total_compared += result.compared_elements
        if result.passed:
            summary.total_passed += 1
        else:
            summary.total_failed += 1
    return summary


def render_report(summary: AuditSummary) -> str:
    """Render a human-readable report from the analysis summary."""
    lines = [
        "MODELED_BF16_REDUCTION_ORDER_ANALYSIS",
        "NOTE: This is a CPU-only modeled reduction-order analysis.",
        "It does not execute SIRCL, CUDA, RDMA, or NCCL.",
        "Sequential BF16 sum is a naive baseline, not an NCCL model.",
        "",
        f"cases={len(summary.results)} "
        f"invariants_passed={summary.total_passed} "
        f"invariants_failed={summary.total_failed} "
        f"compared_elements={summary.total_compared}",
        "",
        "| case | world | rows | pattern | iters | "
        "ring_MAE | ring_RMSE | ring_max | ring_exact | "
        "ring_outside_tol | "
        "seq_MAE | seq_max | seq_exact | invariants |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in summary.results:
        case = result.case
        m = result.metrics
        inv = "PASS" if result.invariants.passed else (
            "FAIL: " + result.invariants.failure_reason[:60]
        )
        lines.append(
            f"| {case.name} | {case.world_size} | {case.rows} | "
            f"{case.pattern} | {result.iterations} | "
            f"{m.candidate_mae:.6g} | "
            f"{m.candidate_rmse:.6g} | "
            f"{m.candidate_max_abs:.6g} | "
            f"{m.candidate_exact_agreement} | "
            f"{m.outside_tolerance_count} | "
            f"{m.reference_mae:.6g} | "
            f"{m.reference_max_abs:.6g} | "
            f"{m.reference_exact_agreement} | "
            f"{inv} |"
        )
    return "\n".join(lines)
