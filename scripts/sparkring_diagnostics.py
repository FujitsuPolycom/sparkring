#!/usr/bin/env python3
"""Shared diagnostic receipt types for SparkRing operator tooling.

The receipt is the stable seam between check implementations and consumers.
Ring Doctor, preflight, cable qualification, text renderers, saved evidence,
and AI-assisted analysis can all describe results without sharing their probe
or evaluation implementations.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from collections import Counter
from typing import Any, Sequence

SCHEMA = "sparkring-diagnostics/v1"


class CheckStatus(str, enum.Enum):
    """One check outcome; absence of evidence is never reported as a pass."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


@dataclasses.dataclass(frozen=True)
class DiagnosticCheck:
    """One attributable result with the evidence supporting its status."""

    check_id: str
    status: CheckStatus
    scope: str
    subject: str
    summary: str
    evidence: str
    node: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["status"] = self.status.value
        return value


def build_receipt(
    checks: Sequence[DiagnosticCheck],
    *,
    generated_at: str | None = None,
    source: str,
) -> dict[str, Any]:
    """Build a versioned receipt without interpreting unknown as success."""
    counts = Counter(check.status.value for check in checks)
    passed = bool(checks) and all(check.status is CheckStatus.PASS for check in checks)
    return {
        "schema": SCHEMA,
        "generated_at": generated_at
        or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "passed": passed,
        "totals": {
            status.value: counts.get(status.value, 0) for status in CheckStatus
        },
        "checks": [check.to_dict() for check in checks],
    }
