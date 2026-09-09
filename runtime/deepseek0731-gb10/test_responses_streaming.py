"""Actual patched protocol models, including the SDK event revalidation seam."""

import json

import pytest
from pydantic import ValidationError

import responses_streaming_probe as probe


@pytest.fixture(scope="module")
def models():
    return probe.load_models()


def test_replay_receipts_bind_the_packaged_patch_and_probe():
    contract = probe.overlay.load_contract(probe.HERE / "runtime-contract.json")
    source = json.loads((probe.HERE / "api-contract-source-replay.json").read_text())
    assert source["patch_sha256"] == contract["runtime_patch"]["sha256"]
    assert source["files"] == contract["runtime_patch"]["files"]
    receipt = json.loads((probe.HERE / "api-responses-streaming-replay.json").read_text())
    assert receipt["patch_sha256"] == source["patch_sha256"]
    assert receipt["source_sha256"] == probe.overlay.sha256_bytes(probe.patched_protocol())
    assert receipt["harness_sha256"] == probe.overlay.sha256_file(
        probe.HERE / "responses_streaming_probe.py")
    observed = probe.run_probe()
    for key in ("cases", "invalid_controls", "openai_version", "pydantic_version"):
        assert receipt[key] == observed[key]


@pytest.mark.parametrize("effort", probe.EFFORTS)
def test_request_response_and_all_sse_events_preserve_effort(models, effort):
    probe.check_case(models, {"effort": effort, "summary": "concise"})


def test_absent_reasoning_stays_absent(models):
    probe.check_case(models, None)


@pytest.mark.parametrize("reasoning", probe.INVALID_REASONING)
def test_invalid_reasoning_rejected_by_request_response_and_events(models, reasoning):
    probe.check_invalid(models, reasoning)


@pytest.mark.parametrize("name,event_type,status", probe.EVENTS)
def test_request_only_fix_reproduces_max_failure_at_sdk_event(name, event_type, status):
    source = probe.patched_protocol().decode()
    start = source.index("class ResponsesResponse(OpenAIBaseModel):")
    before, response = source[:start], source[start:]
    assert response.count("reasoning: ResponsesReasoning | None = None") == 1
    response = response.replace("reasoning: ResponsesReasoning | None = None",
                                "reasoning: Reasoning | None = None", 1)
    baseline = probe.load_models((before + response).encode())
    _, result = probe.response_from_request(baseline, {"effort": "max"}, status)
    with pytest.raises(ValidationError, match="literal_error"):
        getattr(baseline, name)(type=event_type, sequence_number=0,
                               response=result.model_dump(mode="json"))
    probe.check_case(baseline, {"effort": "xhigh"})
