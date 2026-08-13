from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "endpoint_benchmark", HERE / "endpoint_benchmark.py"
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def completion_body(values: list[float]) -> dict:
    return {
        "choices": [
            {
                "logprobs": {
                    "tokens": [f"t{index}" for index in range(len(values))],
                    "token_logprobs": values,
                    "top_logprobs": [
                        {f"t{index}": value} for index, value in enumerate(values)
                    ],
                }
            }
        ]
    }


def test_logprob_validation_accepts_only_finite_complete_arrays() -> None:
    result = BENCHMARK.validate_completion_logprobs(
        completion_body([-0.25, -1.5]), 2
    )
    assert result["completion_tokens"] == 2
    assert result["finite_logprob_values"] == 4
    assert result["minimum_logprob"] == -1.5

    with pytest.raises(BENCHMARK.BenchmarkError, match="nonfinite"):
        BENCHMARK.validate_completion_logprobs(completion_body([-0.25, math.nan]), 2)
    with pytest.raises(BENCHMARK.BenchmarkError, match="lengths disagree"):
        BENCHMARK.validate_completion_logprobs(completion_body([-0.25]), 2)


def test_chat_canary_requires_expected_answer() -> None:
    result = BENCHMARK.validate_chat_answer(
        {
            "choices": [
                {
                    "message": {"reasoning_content": "17 + 25", "content": "42"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }
    )
    assert result["answer_present"] is True

    with pytest.raises(BENCHMARK.BenchmarkError, match="expected integer"):
        BENCHMARK.validate_chat_answer(
            {
                "choices": [{"message": {"content": "41"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1},
            }
        )


def test_stream_metrics_excludes_first_token_from_decode_rate() -> None:
    result = BENCHMARK.StreamResult(
        started=10.0,
        first_token_at=12.0,
        last_token_at=16.0,
        finished=16.5,
        usage={"prompt_tokens": 100, "completion_tokens": 9},
        finish_reason="length",
        text="answer",
        content_events=9,
    )
    metrics = BENCHMARK.stream_metrics(result, requested_decode_tokens=9)
    assert metrics["time_to_first_token_seconds"] == 2.0
    assert metrics["client_prompt_tokens_per_ttft_second"] == 50.0
    assert metrics["inter_token_decode_tokens_per_second"] == 2.0
    assert metrics["end_to_end_completion_tokens_per_second"] == pytest.approx(9 / 6.5)
