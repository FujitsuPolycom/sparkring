#!/usr/bin/env python3
"""Verify GB10 parser bytes and replay safe malformed-DSML recovery cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


HERE = Path(__file__).resolve().parent
DEFAULT_CONTRACT = HERE / "runtime-contract.json"
DSML = "｜DSML｜"


class ReplayError(RuntimeError):
    """The parser overlay or a deterministic recovery result differs."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser_results(contract: dict[str, Any]) -> dict[str, str]:
    return {
        value["path"]: value["result_sha256"]
        for value in contract["runtime_patch"]["files"]
        if value["path"].startswith("vllm/parser/")
    }


def verify_parser_files(site_root: Path, contract: dict[str, Any]) -> None:
    expected = _parser_results(contract)
    if len(expected) != 9:
        raise ReplayError("runtime contract must name all nine parser files")
    for relative, digest in expected.items():
        path = site_root / relative
        observed = _sha256(path) if path.is_file() else None
        if observed != digest:
            raise ReplayError(
                f"installed parser differs: {relative}: expected {digest}, got {observed}"
            )


def _invoke(name: str = "emit_result", *, complete: bool = True) -> str:
    text = (
        f'<{DSML}invoke name="{name}">\n'
        f'<{DSML}parameter name="value" string="false">9001</{DSML}parameter>\n'
    )
    return text + (f"</{DSML}invoke>\n" if complete else "")


def replay_cases() -> dict[str, dict[str, str]]:
    closer = f"</{DSML}tool_calls>"
    valid = f"<{DSML}tool_calls>\n" + _invoke() + closer
    missing = _invoke() + closer
    corrupted = f"<{DSML}toolcalls>\n" + _invoke() + closer
    return {
        "valid_wrapper_required": {
            "expected": "tool_call",
            "text": valid,
            "tool_choice": "required",
        },
        "missing_wrapper_required": {
            "expected": "tool_call",
            "text": missing,
            "tool_choice": "required",
        },
        "corrupted_wrapper_required": {
            "expected": "tool_call",
            "text": corrupted,
            "tool_choice": "required",
        },
        "reasoning_boundary_required": {
            "expected": "tool_call",
            "text": f"<think>Call emit_result now.</think>\n{missing}",
            "tool_choice": "required",
            "thinking": True,
        },
        "valid_wrapper_auto": {
            "expected": "tool_call",
            "text": valid,
            "tool_choice": "auto",
        },
        "undeclared_tool": {
            "expected": "content",
            "text": _invoke("not_declared") + closer,
            "tool_choice": "auto",
        },
        "tool_choice_none": {
            "expected": "content",
            "text": missing,
            "tool_choice": "none",
        },
        "truncated_invoke": {
            "expected": "content",
            "text": _invoke(complete=False) + closer,
            "tool_choice": "required",
        },
    }


def _collect_stream(deltas: list[Any]) -> tuple[str | None, str, str]:
    name: str | None = None
    args = ""
    content = ""
    for delta in deltas:
        if delta is None:
            continue
        content += delta.content or ""
        for tool_call in delta.tool_calls or []:
            function = tool_call.function
            if function is not None:
                name = function.name or name
                args += function.arguments or ""
    return name, args, content


def _stream(parser: Any, request: Any, chunks: list[str]) -> tuple[str | None, str, str]:
    previous = ""
    deltas = []
    for chunk in chunks:
        current = previous + chunk
        deltas.append(
            parser.extract_tool_calls_streaming(
                previous_text=previous,
                current_text=current,
                delta_text=chunk,
                previous_token_ids=(),
                current_token_ids=(),
                delta_token_ids=(),
                request=request,
            )
        )
        previous = current
    deltas.append(parser.finish_streaming())
    return _collect_stream(deltas)


def run(contract_path: Path) -> dict[str, Any]:
    import vllm
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionToolsParam,
    )
    from vllm.parser.deepseek_v4 import DeepSeekV4Parser

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    site_root = Path(vllm.__file__).resolve().parent.parent
    verify_parser_files(site_root, contract)

    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "emit_result",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    )
    tokenizer = MagicMock()
    tokenizer.all_special_tokens = []
    tokenizer.all_special_ids = []
    tokenizer.get_vocab.return_value = {}

    results = []
    for name, case in replay_cases().items():
        request = MagicMock(spec=ChatCompletionRequest)
        request.tools = [tool]
        request.tool_choice = case["tool_choice"]
        request.include_reasoning = True
        parser = DeepSeekV4Parser(
            tokenizer,
            tools=[tool],
            chat_template_kwargs={"thinking": case.get("thinking", False)},
        )
        parsed = parser.extract_tool_calls(case["text"], request)
        if case["expected"] == "tool_call":
            if not parsed.tools_called or len(parsed.tool_calls) != 1:
                raise ReplayError(f"{name}: expected exactly one recovered tool call")
            call = parsed.tool_calls[0].function
            if call.name != "emit_result" or json.loads(call.arguments) != {
                "value": 9001
            }:
                raise ReplayError(f"{name}: recovered tool call differs")
            visible = parsed.content or ""
            if DSML in visible or "<｜Assistant｜>" in visible:
                raise ReplayError(
                    f"{name}: protocol marker leaked into visible content: {visible!r}"
                )
        elif parsed.tools_called or parsed.content != case["text"]:
            raise ReplayError(f"{name}: unsafe recovery did not remain content")
        results.append({"case": name, "mode": "complete", "status": "pass"})

    stream_request = MagicMock(spec=ChatCompletionRequest)
    stream_request.tools = [tool]
    stream_request.tool_choice = "required"
    stream_request.include_reasoning = True
    stream_parser = DeepSeekV4Parser(
        tokenizer,
        tools=[tool],
        chat_template_kwargs={"thinking": False},
    )
    chunks = [
        "Checking.\n",
        "<｜DSML",
        '｜invoke name="emit_result">',
        f'\n<{DSML}parameter name="value" string="false">9001</{DSML}parameter>\n',
        f"</{DSML}invoke>",
        f"</{DSML}tool_calls>",
    ]
    function, arguments, content = _stream(stream_parser, stream_request, chunks)
    if function != "emit_result" or json.loads(arguments) != {"value": 9001}:
        raise ReplayError("split-marker streaming recovery differs")
    if content != "Checking.\n" or DSML in content or "<｜Assistant｜>" in content:
        raise ReplayError(f"streaming recovery content differs: {content!r}")
    results.append({"case": "split_marker", "mode": "stream", "status": "pass"})

    corrupted_stream_parser = DeepSeekV4Parser(
        tokenizer,
        tools=[tool],
        chat_template_kwargs={"thinking": False},
    )
    corrupted_chunks = [
        "<｜DSML｜tool",
        "calls>\n",
        f'<{DSML}invoke name="emit_result">',
        f'\n<{DSML}parameter name="value" string="false">9001</{DSML}parameter>\n',
        f"</{DSML}invoke>",
        f"</{DSML}tool_calls>",
    ]
    function, arguments, content = _stream(
        corrupted_stream_parser, stream_request, corrupted_chunks
    )
    if function != "emit_result" or json.loads(arguments) != {"value": 9001}:
        raise ReplayError("corrupted-wrapper streaming recovery differs")
    if content.strip() or DSML in content or "<｜Assistant｜>" in content:
        raise ReplayError(
            f"corrupted-wrapper streaming content differs: {content!r}"
        )
    results.append(
        {"case": "corrupted_wrapper", "mode": "stream", "status": "pass"}
    )

    return {
        "schema": "sparkring-deepseek-v4-gb10-parser-replay/v1",
        "status": "pass",
        "site_root": str(site_root),
        "cases": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args(argv)
    try:
        result = run(args.contract.resolve(strict=True))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, ReplayError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
