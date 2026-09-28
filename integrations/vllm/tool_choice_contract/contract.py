"""Optional response validation for named and required Chat Completions tools.

An installer image derived with runtime/images/derive_tool_choice_contract.py
carries this file unchanged as vLLM's ``sparkring_tool_choice_contract``
module, so it imports only the standard library.
"""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import dataclass, field
from functools import wraps
from http import HTTPStatus
import json
import os

ENVIRONMENT = "SPARKRING_TOOL_CHOICE_CONTRACT"
MARKER = "_sparkring_tool_choice_contract"


def value(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def scope(request):
    choice = value(request, "tool_choice")
    if choice == "required":
        return "required", None
    if value(choice, "type") == "function":
        return "named", value(value(choice, "function"), "name")
    return None


def _reject_constant(text):
    raise ValueError("Non-JSON numeric constant")


def violation(request, calls, finish_reason):
    """Return a bounded diagnostic that names the engine's finish reason, or None.

    The diagnostic never includes generated arguments or text.
    """
    contract = scope(request)
    if contract is None:
        return None
    problem = _problem(request, contract, calls, finish_reason)
    return None if problem is None else f"{problem}; finish_reason: {finish_reason}."


def _problem(request, contract, calls, finish_reason):
    if finish_reason not in ("stop", "tool_calls", "length"):
        return "Generation did not finish normally; required tool output is incomplete"
    if not calls:
        return "Named or required tool_choice produced no tool calls"
    mode, name = contract
    if value(request, "parallel_tool_calls") is False and len(calls) != 1:
        return "Tool response contains parallel calls when parallel_tool_calls is false"
    declared = {value(value(tool, "function"), "name")
                for tool in value(request, "tools", []) or []
                if value(tool, "type") == "function"}
    for call in calls:
        function = value(call, "function")
        call_name = value(function, "name")
        if not call_name or call_name not in declared:
            return "Tool output names a function absent from the request's tool definitions"
        if mode == "named" and call_name != name:
            return "Tool output does not match the function selected by tool_choice"
        arguments = value(function, "arguments")
        try:
            if not isinstance(arguments, str):
                raise ValueError("Expected JSON text")
            parsed = json.loads(arguments, parse_constant=_reject_constant)
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object")
        except (ValueError, RecursionError):
            return "Tool arguments are not a complete JSON object"
    return None


def _error(service, message, *, streaming=False):
    factory = (service.create_streaming_error_response if streaming
               else service.create_error_response)
    return factory(message, err_type="ToolChoiceContractError",
                   status_code=HTTPStatus.INTERNAL_SERVER_ERROR, param="tool_choice")


async def _capture_finish(generator, reasons):
    async with aclosing(generator):
        async for result in generator:
            for output in result.outputs:
                if output.finish_reason is not None:
                    reasons[output.index] = output.finish_reason
            yield result


def wrap_full(original):
    @wraps(original)
    async def checked(self, request, result_generator, *args, **kwargs):
        if scope(request) is None:
            return await original(self, request, result_generator, *args, **kwargs)
        reasons = {}
        async with aclosing(_capture_finish(result_generator, reasons)) as captured:
            result = await original(self, request, captured, *args, **kwargs)
        choices = value(result, "choices")
        if choices is None:  # Preserve the engine/parser's own error response.
            return result
        if len(choices) != (value(request, "n") or 1):
            return _error(self, "Tool response is missing requested completion choices.")
        if {value(choice, "index") for choice in choices} != set(range(value(request, "n") or 1)):
            return _error(self, "Tool response contains invalid completion choice indices.")
        for choice in choices:
            message = violation(request, value(value(choice, "message"), "tool_calls"),
                                reasons.get(value(choice, "index")))
            if message:
                return _error(self, message)
        return result
    return checked


@dataclass
class StreamChoice:
    calls: dict = field(default_factory=dict)
    finished: bool = False

    def append(self, delta):
        for call in value(delta, "tool_calls", []) or []:
            index = value(call, "index")
            if not isinstance(index, int) or index < 0:
                return "Tool stream is missing a valid call index."
            function = value(call, "function")
            output = self.calls.setdefault(index, {"function": {"name": "", "arguments": ""}})
            for key in ("name", "arguments"):
                part = value(function, key)
                if part is not None:
                    if not isinstance(part, str):
                        return "Tool stream contains a non-text function fragment."
                    output["function"][key] += part
        return None


def wrap_stream(original):
    @wraps(original)
    async def checked(self, request, result_generator, *args, **kwargs):
        if scope(request) is None:
            async with aclosing(original(self, request, result_generator, *args, **kwargs)) as source:
                async for frame in source:
                    yield frame
            return
        reasons, choices = {}, {}
        done = False
        async with aclosing(_capture_finish(result_generator, reasons)) as captured:
            async with aclosing(original(self, request, captured, *args, **kwargs)) as source:
                async for frame in source:
                    # Admitted vLLM generators emit one complete SSE data event per yield.
                    payload = frame.removeprefix("data: ").strip()
                    message = None
                    rewrite = False
                    if payload == "[DONE]":
                        done = True
                        if (len(choices) != (value(request, "n") or 1)
                                or not all(choice.finished for choice in choices.values())):
                            message = "Tool stream ended before all requested choices completed."
                    else:
                        data = json.loads(payload)
                        if "error" in data or data.get("object") == "error":
                            # Preserve existing engine/parser errors and their terminal event.
                            yield frame
                            yield "data: [DONE]\n\n"
                            return
                        for choice in data.get("choices", []):
                            index = choice["index"]
                            if index not in range(value(request, "n") or 1):
                                message = "Tool stream contains an invalid completion choice index."
                                break
                            state = choices.setdefault(index, StreamChoice())
                            message = state.append(choice.get("delta", {}))
                            if message:
                                break
                            if choice.get("finish_reason") is not None:
                                message = violation(request, list(state.calls.values()), reasons.get(index))
                                state.finished = True
                                if message:
                                    break
                                if reasons.get(index) == "length":
                                    # vLLM's tool detection can otherwise hide the engine's limit.
                                    choice["finish_reason"] = "length"
                                    rewrite = True
                    if message:
                        yield f"data: {_error(self, message, streaming=True)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    yield f"data: {json.dumps(data)}\n\n" if rewrite else frame
                if not done:
                    yield f"data: {_error(self, 'Tool stream ended without a terminal event.', streaming=True)}\n\n"
                    yield "data: [DONE]\n\n"
    return checked


def enabled(environ):
    """Whether ``environ`` enables the policy; it is off unless the variable is 1."""
    setting = environ.get(ENVIRONMENT, "0")
    if setting not in ("0", "1"):
        raise ValueError(f"{ENVIRONMENT} must be 0 or 1.")
    return setting == "1"


def install(cls):
    """Wrap the full and streaming Chat Completions generators of ``cls`` once.

    Subclasses that inherit these generators, such as vLLM's Anthropic Messages
    handler, inherit the policy.
    """
    if getattr(cls, MARKER, False):
        return
    cls.chat_completion_full_generator = wrap_full(cls.chat_completion_full_generator)
    cls.chat_completion_stream_generator = wrap_stream(cls.chat_completion_stream_generator)
    setattr(cls, MARKER, True)


def install_from_environment(cls, environ=None):
    """Install the policy on ``cls`` when SPARKRING_TOOL_CHOICE_CONTRACT is 1.

    An image derived with runtime/images/derive_tool_choice_contract.py calls
    this when vLLM imports its Chat Completions serving module, so every Python
    API server process reads the same setting. A value other than 0 or 1
    raises, which stops the server instead of serving without the requested
    policy.
    """
    if enabled(os.environ if environ is None else environ):
        install(cls)
