from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RECEIPT = (
    ROOT
    / "performance"
    / "receipts"
    / "glm53-flash"
    / "scheduler-cadence-20260902"
    / "summary.json"
)


def test_scheduler_cadence_receipt_supports_interval_two_default() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema"] == "sparkring-glm53-scheduler-cadence/v1"
    assert receipt["status"] == "qualified"
    assert receipt["jit_policy"]["retained_samples_jit_free"] is True

    for name in ("interval_2", "interval_8"):
        cohort = receipt[name]
        for sample in cohort["samples"]:
            assert sample["completed"] == 8
            assert sample["errors"] == 0
        for field in (
            "aggregate_tokens_per_second",
            "average_ttft_seconds",
            "maximum_ttft_seconds",
            "p90_latency_seconds",
        ):
            measured = statistics.median(
                sample[field] for sample in cohort["samples"]
            )
            assert math.isclose(
                measured,
                cohort["median"][field],
                rel_tol=0.0,
                abs_tol=1e-9,
            )

    faster = receipt["interval_2"]["median"]
    slower = receipt["interval_8"]["median"]
    assert faster["aggregate_tokens_per_second"] > slower[
        "aggregate_tokens_per_second"
    ]
    assert faster["average_ttft_seconds"] < slower["average_ttft_seconds"]
    assert faster["maximum_ttft_seconds"] < slower["maximum_ttft_seconds"]


def test_operator_contract_defaults_to_interval_two() -> None:
    pins = json.loads(
        (
            ROOT / "runtime" / "glm53-flash-jj-r8-gb10" / "pins.json"
        ).read_text(encoding="utf-8")
    )
    base = json.loads(
        (
            ROOT / "recipes" / "glm53-flash-nvfp4-dflash2-bf16-tp4.json"
        ).read_text(encoding="utf-8")
    )
    cached = json.loads(
        (
            ROOT
            / "recipes"
            / "sparkcache"
            / "glm53-flash-nvfp4-dflash2-bf16-tp4.json"
        ).read_text(encoding="utf-8")
    )
    assert pins["defaults"]["prefill_schedule_interval"] == 2
    assert base["serving_common"]["prefill_schedule_interval"] == 2
    assert cached["serving_common"]["prefill_schedule_interval"] == 2
