"""Replay exact admitted serving methods with fixture engine and parser results.

This executes the source methods without importing vLLM. Response classes,
tokenizer, engine, parser, and ancillary helpers are fixtures. It does not test
HTTP routing, a model, argument-schema enforcement, or GPU execution.
"""

import argparse
import ast
import asyncio
from http import HTTPStatus
import hashlib
import json
import logging
import os
from pathlib import Path
import time

from contract import wrap_full, wrap_stream
from serve import ADMITTED


class Record:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        return None

    def model_dump_json(self, **kwargs):
        return json.dumps(self, default=vars)


class NamedChoice(Record):
    pass


class Service(Record):
    def create_error_response(self, message, err_type="InternalServerError",
                              status_code=HTTPStatus.INTERNAL_SERVER_ERROR, param=None):
        return {"error": {"message": str(message), "type": err_type,
                          "code": int(status_code), "param": param}}

    def create_streaming_error_response(self, *args, **kwargs):
        return json.dumps(self.create_error_response(*args, **kwargs))


def load_methods(path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest not in ADMITTED:
        raise ValueError(f"Serving source not admitted: {digest}")
    names = {"chat_completion_full_generator", "chat_completion_stream_generator"}
    methods = [node for node in ast.walk(ast.parse(raw))
               if isinstance(node, ast.AsyncFunctionDef) and node.name in names]
    assert {node.name for node in methods} == names and len(methods) == 2
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *methods], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"asyncio": asyncio, "time": time, "os": os, "HTTPStatus": HTTPStatus,
                 "logger": logging.getLogger(__name__), "GenerationError": RuntimeError,
                 "ChatCompletionNamedToolChoiceParam": NamedChoice,
                 "make_tool_call_id": lambda: "fixture-call",
                 "maybe_filter_parallel_tool_calls": lambda choice, request: choice,
                 "_make_prompt_tokens_details": lambda *args: None,
                 "build_spec_decoding_metrics": lambda *args: None,
                 "clamp_prompt_logprobs": lambda value: value,
                 "should_include_usage": lambda *args: (False, False), "as_list": list}
    for name in ("ChatMessage", "ToolCall", "ChatCompletionResponseChoice", "ChatCompletionResponse",
                 "UsageInfo", "DeltaMessage", "ChatCompletionResponseStreamChoice", "ChatCompletionStreamResponse"):
        namespace[name] = Record
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}, digest


async def run_case(methods, *, mode, kind, checked, stream):
    arguments = '{"key":' if kind == "truncated" else '{"key":"value"}'
    name = "other" if kind == "wrong-name" else "lookup"
    calls = [] if kind == "empty" else [Record(id=None, name=name, arguments=arguments)]
    finish = "length" if kind in ("truncated", "complete-length", "empty") else "stop"
    req = Record(tool_choice=(NamedChoice(type="function", function=Record(name="lookup"))
                              if mode == "named" else "required"), n=1,
                 tools=[Record(type="function", function=Record(name="lookup")),
                        Record(type="function", function=Record(name="other"))],
                 logprobs=False, include_reasoning=True, return_token_ids=False,
                 echo=False, return_prompt_text=False, parallel_tool_calls=True)

    class Parser:
        tool_parser_cls = object

        def __init__(self, *args, **kwargs):
            pass

        def parse(self, *args, **kwargs):
            return None, "", calls

        def parse_delta(self, *args, **kwargs):
            return Record(tool_calls=[Record(index=i, function=call) for i, call in enumerate(calls)])

        def count_reasoning_tokens(self, *args):
            return 0

    service = Service(parser_cls=Parser, enable_auto_tools=True,
        get_chat_request_role=lambda req: "assistant", _raise_if_error=lambda *args: None,
        _create_chat_message=lambda **kwargs: Record(**kwargs),
        _finalize_response_message=lambda message, **kwargs: message,
        enable_prompt_tokens_details=False, enable_log_outputs=False,
        enable_per_request_metrics=False, system_fingerprint="fixture")

    async def engine():
        yield Record(outputs=[Record(index=0, text="fixture", token_ids=[1, 2],
                                     finish_reason=finish)], prompt_token_ids=[3, 4])

    method = methods["chat_completion_stream_generator" if stream else "chat_completion_full_generator"]
    if checked:
        method = wrap_stream(method) if stream else wrap_full(method)
    arguments = (service, req, engine(), "fixture-request", "fixture-model", [], object(), Record())
    if stream:
        output = [frame async for frame in method(*arguments)]
        data = [json.loads(frame[6:]) for frame in output if frame != "data: [DONE]\n\n"]
        errors = [frame["error"] for frame in data if "error" in frame]
        terminal = [choice for frame in data for choice in frame.get("choices", [])
                    if choice.get("finish_reason") is not None]
    else:
        output = await method(*arguments, parser=Parser())
        errors = [output["error"]] if isinstance(output, dict) and "error" in output else []
        terminal = [vars(choice) for choice in output.choices] if not errors else []
    expected_error = checked and (kind in ("empty", "truncated") or (mode == "named" and kind == "wrong-name"))
    assert bool(errors) == expected_error, (mode, kind, checked, stream, errors)
    if errors:
        assert errors[0]["type"] == "ToolChoiceContractError", errors
    elif checked and kind == "complete-length":
        assert terminal[0]["finish_reason"] == "length", terminal
    return {"mode": mode, "case": kind, "checked": checked, "stream": stream,
            "error": errors[0]["type"] if errors else None,
            "finish_reason": terminal[0]["finish_reason"] if terminal else None}


async def replay(source):
    methods, digest = load_methods(source)
    cases = []
    for mode in ("named", "required"):
        for kind in ("empty", "truncated", "wrong-name", "valid", "complete-length"):
            for checked in (False, True):
                for stream in (False, True):
                    cases.append(await run_case(methods, mode=mode, kind=kind, checked=checked, stream=stream))
    return {"schema": "sparkring-tool-contract-source-replay/v1", "source_sha256": digest,
            "source_family": ADMITTED[digest], "cases_passed": len(cases), "cases": cases,
            "scope": "Exact source methods; fixture engine/parser/response interfaces. No HTTP or model execution."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(replay(args.source))
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))
