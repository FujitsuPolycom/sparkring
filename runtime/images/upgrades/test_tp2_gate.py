"""GPU-free checks for composite model qualification and rollback admission."""

import copy

import pytest

from runtime.images.upgrades import tp2_gate as gate

IMAGE = "sha256:" + "a" * 64
INPUT = "b" * 64


def report():
    model = {
        "outcome": "passed",
        "subject_sha256": IMAGE,
        "input_sha256": INPUT,
        "skipped": 0,
        "assertions": 10,
        "evidence": {
            "cache": {"restored": True},
            "media": {"checked": True},
            "performance": {"passed": True},
        },
    }
    return {
        "passed": True,
        "promoted": False,
        "baseline_restored": True,
        "failure": None,
        "cleanup_errors": [],
        "models": [model, copy.deepcopy(model)],
    }


def test_accepts_two_complete_model_suites_and_restored_baseline():
    result = gate.receipt(
        report(), gate_id="tp2-models", input_sha256=INPUT, image_id=IMAGE
    )
    assert result["outcome"] == "passed"
    assert result["assertions"] == 21
    assert result["subject_sha256"] == IMAGE
    assert "baseline" not in result  # Controls belong to their model-specific receipts.


@pytest.mark.parametrize(
    "field,value",
    [
        ("promoted", True),
        ("baseline_restored", False),
        ("passed", False),
        ("cleanup_errors", ["failed"]),
        ("failure", "failed"),
    ],
)
def test_rejects_failure_or_changed_serving_deployment(field, value):
    value_report = report()
    value_report[field] = value
    assert (
        gate.receipt(
            value_report, gate_id="tp2-models", input_sha256=INPUT, image_id=IMAGE
        )["outcome"]
        == "failed"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("outcome", "failed"),
        ("subject_sha256", "sha256:" + "c" * 64),
        ("input_sha256", "d" * 64),
        ("skipped", 1),
        ("assertions", 0),
    ],
)
def test_rejects_wrong_subject_or_incomplete_nested_suite(field, value):
    value_report = report()
    value_report["models"][1][field] = value
    assert (
        gate.receipt(
            value_report, gate_id="tp2-models", input_sha256=INPUT, image_id=IMAGE
        )["outcome"]
        == "failed"
    )


@pytest.mark.parametrize("missing", ["cache", "media", "performance"])
def test_rejects_missing_required_workload(missing):
    value_report = report()
    del value_report["models"][0]["evidence"][missing]
    assert (
        gate.receipt(
            value_report, gate_id="tp2-models", input_sha256=INPUT, image_id=IMAGE
        )["outcome"]
        == "failed"
    )


def test_rejects_one_model_only():
    value_report = report()
    value_report["models"].pop()
    assert (
        gate.receipt(
            value_report, gate_id="tp2-models", input_sha256=INPUT, image_id=IMAGE
        )["outcome"]
        == "failed"
    )
