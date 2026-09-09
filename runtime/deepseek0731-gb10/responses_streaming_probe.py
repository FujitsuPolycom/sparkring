"""Replay the patched Responses models and SSE serialization without a server.

Complete upstream protocol files are authenticated before execution. The actual
OpenAIBaseModel, ResponsesRequest, ResponsesResponse, SDK imports and event
subclasses execute unchanged. Rendering, sampling and Harmony interfaces are
fixtures; their optional request fields are not exercised. No vLLM installation,
HTTP request, tokenizer, engine or GPU is needed.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import logging
from pathlib import Path
import platform
from types import SimpleNamespace
from typing import ClassVar
from uuid import uuid4

import openai
from openai.types.chat import ChatCompletionMessageParam
import pydantic
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

import apply_runtime_overlay as overlay


HERE = Path(__file__).resolve().parent
PROTOCOL = "vllm/entrypoints/openai/responses/protocol.py"
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", None)
INVALID_REASONING = (
    {"effort": "off"}, {"effort": "unsupported"}, {"effort": 3},
    {"effort": "max", "summary": "unsupported"},
)
EVENTS = (
    ("ResponseCreatedEvent", "response.created", "in_progress"),
    ("ResponseInProgressEvent", "response.in_progress", "in_progress"),
    ("ResponseCompletedEvent", "response.completed", "completed"),
)


def source_fixture(name: str) -> bytes:
    inputs = json.loads((HERE / "upstream/sources.json").read_text())
    contract = overlay.load_contract(HERE / "runtime-contract.json")
    assert inputs["commit"] == contract["vllm_source"]["commit"]
    item = next(item for item in inputs["files"] if item["fixture"] == name)
    raw = (HERE / "upstream" / name).read_bytes()
    assert overlay.sha256_bytes(raw) == item["fixture_sha256"]
    source = gzip.decompress(raw) if name.endswith(".gz") else raw
    assert overlay.sha256_bytes(source) == item["sha256"]
    return source


def patched_protocol() -> bytes:
    record = overlay.load_contract(HERE / "runtime-contract.json")["runtime_patch"]
    patch_path = HERE / record["path"]
    assert overlay.sha256_file(patch_path) == record["sha256"]
    item = next(item for item in record["files"] if item["path"] == PROTOCOL)
    preimage = source_fixture("responses-protocol.py.gz")
    assert overlay.sha256_bytes(preimage) == item["preimage_sha256"]
    patches = overlay.parse_unified_patch(patch_path.read_text(encoding="utf-8"))
    result = overlay.apply_file_patch(preimage, patches[PROTOCOL])
    assert overlay.sha256_bytes(result) == item["result_sha256"]
    return result


def _unavailable(*args, **kwargs):
    raise AssertionError("renderer/sampling validation is outside this model probe")


def load_models(source: bytes | None = None) -> SimpleNamespace:
    """Keep model bodies intact; replace only unrelated runtime dependencies."""
    assert openai.__version__ == "2.29.0"
    assert pydantic.__version__ == "2.12.5"
    namespace = {
        "__name__": "sparkring_responses_protocol_replay",
        "BaseModel": BaseModel, "ConfigDict": ConfigDict,
        "ClassVar": ClassVar, "model_validator": model_validator,
        "logger": logging.getLogger(__name__),
        "init_logger": logging.getLogger, "random_uuid": lambda: uuid4().hex,
        "ChatCompletionMessageParam": ChatCompletionMessageParam,
        "ChatTemplateContentFormatOption": str,
        "OpenAIHarmonyMessage": BaseModel, "StructuredOutputsParams": BaseModel,
        "ModelConfig": SimpleNamespace, "SamplingParams": SimpleNamespace,
        "ChatParams": _unavailable, "TokenizeParams": _unavailable,
        "RequestOutputKind": SimpleNamespace, "merge_kwargs": _unavailable,
        "VLLMValidationError": _unavailable,
    }
    engine = ast.parse(source_fixture("engine-protocol.py.gz"))
    base, = [node for node in engine.body
             if isinstance(node, ast.ClassDef) and node.name == "OpenAIBaseModel"]
    exec(compile(ast.Module(body=[base], type_ignores=[]),
                 "vllm/entrypoints/openai/engine/protocol.py", "exec",
                 dont_inherit=True), namespace)
    protocol = ast.parse(patched_protocol() if source is None else source)
    protocol.body = [node for node in protocol.body if not (
        isinstance(node, ast.ImportFrom)
        and (node.module.startswith("vllm.") or node.module == "openai_harmony")
    )]
    exec(compile(protocol, PROTOCOL, "exec", dont_inherit=True), namespace)
    return SimpleNamespace(**namespace)


def response_from_request(models, reasoning, status="in_progress"):
    request = models.ResponsesRequest(
        input="Reply with ok.", model="fixture-model", max_output_tokens=10,
        stream=True, reasoning=reasoning,
    )
    sampling = SimpleNamespace(temperature=1.0, top_p=1.0, max_tokens=10,
                               presence_penalty=0.0, frequency_penalty=0.0,
                               logprobs=None)
    response = models.ResponsesResponse.from_request(
        request, sampling, model_name="fixture-model", created_time=0,
        output=[], status=status,
    )
    return request, response


def check_case(models, reasoning):
    frames = []
    for sequence, (name, event_type, status) in enumerate(EVENTS):
        request, response = response_from_request(models, reasoning, status)
        expected = request.reasoning.model_dump(mode="json") if reasoning else None
        assert response.model_dump(mode="json")["reasoning"] == expected
        # The serving generator dumps the initial response before SDK events
        # revalidate it; its final event receives the model instance directly.
        payload = response if status == "completed" else response.model_dump(
            mode="json", by_alias=True)
        event = getattr(models, name)(type=event_type, sequence_number=sequence,
                                     response=payload)
        dumped = event.model_dump_json(indent=None, by_alias=True)
        frame = f"event: {event.type}\ndata: {dumped}\n\n"
        data = json.loads(frame.split("\ndata: ", 1)[1])
        assert data["type"] == event_type
        assert data["sequence_number"] == sequence
        assert data["response"]["status"] == status
        assert data["response"]["reasoning"] == expected
        rebuilt = getattr(models, name).model_validate_json(dumped)
        assert rebuilt.response.model_dump(mode="json")["reasoning"] == expected
        frames.append(event_type)
    return {"reasoning": reasoning, "events": frames, "passed": True}


def check_invalid(models, reasoning):
    _, valid = response_from_request(models, {"effort": "xhigh"})
    response = valid.model_dump(mode="json")
    response["reasoning"] = reasoning
    checks = [lambda: models.ResponsesRequest(input="ok", reasoning=reasoning),
              lambda: models.ResponsesResponse.model_validate(response)]
    checks.extend(lambda name=name, event_type=event_type: getattr(models, name)(
        type=event_type, sequence_number=0, response=response)
        for name, event_type, _ in EVENTS)
    for check in checks:
        try:
            check()
        except ValidationError as exc:
            assert any("reasoning" in error["loc"] for error in exc.errors())
        else:
            raise AssertionError(f"invalid reasoning accepted: {reasoning!r}")
    return {"reasoning": reasoning, "rejections": len(checks), "passed": True}


def run_probe():
    models = load_models()
    cases = [check_case(models, {"effort": value, "summary": "concise"})
             for value in EFFORTS]
    cases.append(check_case(models, None))
    invalid = [check_invalid(models, value) for value in INVALID_REASONING]
    contract = overlay.load_contract(HERE / "runtime-contract.json")
    return {
        "schema": "sparkring-deepseek-responses-streaming-models/v1",
        "status": "passed", "python_version": platform.python_version(),
        "openai_version": openai.__version__, "pydantic_version": pydantic.__version__,
        "vllm_commit": contract["vllm_source"]["commit"],
        "patch_sha256": contract["runtime_patch"]["sha256"],
        "source": PROTOCOL, "source_sha256": overlay.sha256_bytes(patched_protocol()),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases, "invalid_controls": invalid,
        "limits": "Actual protocol models and SDK SSE validation/serialization; "
                  "renderer/sampling/Harmony interfaces are fixtures. "
                  "No HTTP, engine, tokenizer, model, image or GPU qualification.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    payload = json.dumps(run_probe(), indent=2) + "\n"
    if args.receipt is None:
        print(payload, end="")
    else:
        with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
