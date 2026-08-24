"""Contract tests for the shared SparkRing diagnostic receipt."""

from scripts.sparkring_diagnostics import (
    SCHEMA,
    CheckStatus,
    DiagnosticCheck,
    build_receipt,
)


def check(status: CheckStatus) -> DiagnosticCheck:
    return DiagnosticCheck(
        check_id="RING.EXAMPLE",
        status=status,
        scope="fabric",
        subject="rank0:eth1",
        summary="Example check.",
        evidence="observed=true",
        node="rank0",
        source="test",
    )


def test_receipt_passes_only_when_every_check_passes():
    passed = build_receipt(
        [check(CheckStatus.PASS)],
        generated_at="2026-08-24T00:00:00Z",
        source="test",
    )
    unknown = build_receipt(
        [check(CheckStatus.PASS), check(CheckStatus.UNKNOWN)],
        generated_at="2026-08-24T00:00:00Z",
        source="test",
    )

    assert passed["schema"] == SCHEMA
    assert passed["passed"] is True
    assert passed["totals"] == {
        "pass": 1,
        "fail": 0,
        "unknown": 0,
        "skipped": 0,
    }
    assert unknown["passed"] is False
    assert unknown["totals"]["unknown"] == 1


def test_empty_receipt_is_not_positive_evidence():
    receipt = build_receipt(
        [], generated_at="2026-08-24T00:00:00Z", source="test"
    )

    assert receipt["passed"] is False
    assert receipt["checks"] == []
