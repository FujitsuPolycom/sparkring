"""Performance qualification compares measured matched controls, not projections."""

import pytest
import json

from .contracts import Refused
from .performance_gate import compare, validate_config, matched_metadata


def test_all_three_metric_families_must_preserve_control():
    baseline = {
        "prefill-8192": [3000, 3100, 3200],
        "decode-8192-c1": [49, 50, 51],
        "decode-8192-c1-steps": [21, 22, 23],
    }
    assert compare(baseline, baseline, dict(prefill=0.03, decode=0.03, steps=0.03))[
        "passed"
    ]
    candidate = {**baseline, "decode-8192-c1-steps": [18, 19, 20]}
    value = compare(candidate, baseline, dict(prefill=0.03, decode=0.03, steps=0.03))
    assert not value["passed"]
    assert (
        next(r for r in value["comparisons"] if r["metric"].endswith("steps"))[
            "candidate_median"
        ]
        == 19
    )


def test_missing_or_unrepeated_cells_are_not_qualification():
    with pytest.raises(Refused, match="grids differ"):
        compare(
            {"prefill-8192": [1, 2, 3]},
            {"prefill-32768": [1, 2, 3]},
            dict(prefill=0.03),
        )
    with pytest.raises(Refused, match="three samples"):
        compare(
            {"prefill-8192": [1, 2]}, {"prefill-8192": [1, 2, 3]}, dict(prefill=0.03)
        )


def test_matrix_and_sampling_limits_are_explicit():
    config = dict(
        contexts=[8192],
        prefill_contexts=[8192],
        concurrency=[1, 8],
        max_tokens=512,
        duration=20,
        temperature=0.6,
        limits=dict(prefill=0.03, decode=0.03, steps=0.03),
    )
    validate_config(config)
    config["concurrency"] = [64]
    with pytest.raises(Refused, match="bounded matrix"):
        validate_config(config)


def test_comparison_rejects_changed_sampling_or_output_budget(tmp_path):
    config = dict(
        contexts=[8192],
        concurrency=[1, 8],
        duration=20,
        max_tokens=512,
        temperature=0.6,
    )
    metadata = dict(
        model="model",
        decode_mode="duration",
        duration_per_test=20,
        max_tokens=512,
        temperature=0.6,
        concurrency_levels=[1, 8],
        context_lengths=[8192],
        ignore_eos=True,
        loop_detection={"enabled": True},
    )
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"metadata": metadata}))
    matched_metadata(path, config, "model")
    config["max_tokens"] = 2048
    with pytest.raises(Refused, match="workload differs"):
        matched_metadata(path, config, "model")
