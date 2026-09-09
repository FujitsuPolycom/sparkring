"""Execute the named/required serialization branch from the packaged patch."""

from __future__ import annotations

import os
import textwrap
from http import HTTPStatus
from types import SimpleNamespace

import pytest

from test_runtime_overlay import HERE, _contract, _module


def _serializer(*, patched=True):
    module = _module()
    record = _contract()["runtime_patch"]
    patches = module.parse_unified_patch(
        (HERE / record["path"]).read_text(encoding="utf-8")
    )
    patch = patches["vllm/entrypoints/openai/chat_completion/serving.py"]
    hunk = next(h for h in patch.hunks if any(
        "elif is_named_tool_choice or is_required_tool_choice:" in line
        for line in h.old_lines
    ))
    lines = list(hunk.new_lines if patched else hunk.old_lines)
    start = next(i for i, line in enumerate(lines) if
                 "elif is_named_tool_choice or is_required_tool_choice:" in line)
    end = next(i for i in range(start + 1, len(lines))
               if lines[i].startswith("            # if the request"))
    branch = textwrap.dedent("".join(lines[start:end]))
    branch = branch.replace("elif ", "if ", 1)
    source = (
        "def serialize(tool_calls, is_named_tool_choice, is_required_tool_choice):\n"
        + textwrap.indent(branch, "    ")
        + "    return message\n"
    )
    scope = {
        "os": os, "HTTPStatus": HTTPStatus,
        "self": SimpleNamespace(create_error_response=lambda message, **kw:
                                SimpleNamespace(message=message, **kw)),
        "role": "assistant", "reasoning": None, "content": "",
        "ChatMessage": SimpleNamespace, "ToolCall": SimpleNamespace,
        "make_tool_call_id": lambda: "generated-id",
    }
    exec(source, scope)
    return scope["serialize"]


@pytest.mark.parametrize("named", [True, False])
@pytest.mark.parametrize("calls", [None, []])
def test_empty_required_result_is_error_only_with_opt_in(monkeypatch, named, calls):
    monkeypatch.setenv("SPARKRING_REJECT_EMPTY_REQUIRED_TOOL_CALLS", "1")
    baseline = _serializer(patched=False)(calls, named, not named)
    assert baseline.tool_calls == []
    result = _serializer()(calls, named, not named)
    assert result.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert result.err_type == "ToolChoiceContractError"


@pytest.mark.parametrize("flag", [None, "0"])
def test_unconfigured_contract_preserves_empty_result(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("SPARKRING_REJECT_EMPTY_REQUIRED_TOOL_CALLS", raising=False)
    else:
        monkeypatch.setenv("SPARKRING_REJECT_EMPTY_REQUIRED_TOOL_CALLS", flag)
    assert _serializer()([], False, True).tool_calls == []


def test_valid_parser_result_is_serialized_with_opt_in(monkeypatch):
    monkeypatch.setenv("SPARKRING_REJECT_EMPTY_REQUIRED_TOOL_CALLS", "1")
    call = SimpleNamespace(id=None, name="lookup", arguments='{"key":"value"}')
    result = _serializer()([call], True, False)
    assert result.tool_calls[0].id == "generated-id"
    assert result.tool_calls[0].function is call
