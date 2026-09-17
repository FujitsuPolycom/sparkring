"""Response and SSE contracts with fixture engine/parser outputs; no model tests."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

import contract
import serve
import api_probe


def request(choice="required", **kwargs):
    selected = {"type": "function", "function": {"name": "lookup"}}
    return NS(tool_choice=selected if choice == "named" else choice,
              tools=[{"type": "function", "function": {"name": name}}
                     for name in ("lookup", "other")], n=1, **kwargs)


def call(name="lookup", arguments='{"key":"value"}'):
    return {"function": {"name": name, "arguments": arguments}}


class Service:
    def create_error_response(self, message, **kwargs):
        return {"error": {"message": message, "type": kwargs["err_type"],
                          "code": int(kwargs["status_code"]), "param": kwargs["param"]}}

    def create_streaming_error_response(self, message, **kwargs):
        return json.dumps(self.create_error_response(message, **kwargs))


def assert_error(result):
    assert result["error"]["type"] == "ToolChoiceContractError"
    assert result["error"]["code"] == 500
    assert result["error"]["param"] == "tool_choice"


@pytest.mark.parametrize("mode", ["required", "named"])
@pytest.mark.parametrize("calls", [[], None, [call(arguments='{"key":')],
                                   [call(arguments='[]')], [call(arguments='NaN')],
                                   [call(arguments='{"a":NaN}')], [call(arguments=None)],
                                   [call("undeclared")]])
def test_invalid_results(mode, calls):
    assert contract.violation(request(mode), calls, "length")


@pytest.mark.parametrize("mode", ["required", "named"])
@pytest.mark.parametrize("finish", ["stop", "tool_calls", "length"])
def test_complete_calls_allow_length_and_parallel_named_calls(mode, finish):
    assert contract.violation(request(mode), [call(), call()], finish) is None


def test_name_selection_parallel_flag_and_argument_schema_scope():
    assert contract.violation(request("named"), [call("other")], "stop")
    assert contract.violation(request("required"), [call("other")], "stop") is None
    assert contract.violation(request("named", parallel_tool_calls=False), [call(), call()], "stop")
    assert contract.violation(request("named", parallel_tool_calls=False), [call()], "stop") is None
    # The policy validates a JSON object, not the user-defined argument schema.
    assert contract.violation(request("named"), [call(arguments="{}")], "stop") is None


@pytest.mark.parametrize("choice", [None, "auto", "none"])
def test_unconstrained_requests_unchanged(choice):
    assert contract.violation(request(choice), [], "length") is None


def full_probe(calls, *, choice="required", finish="stop", **req_kwargs):
    closed = []

    async def engine():
        try:
            yield NS(outputs=[NS(index=0, finish_reason=finish)])
        finally:
            closed.append(True)

    async def original(self, req, generator, *args, **kwargs):
        assert args == ("id",) and kwargs == {"parser": "fixture"}
        async for result in generator:
            return {"choices": [{"index": 0, "finish_reason": finish,
                                 "message": {"tool_calls": calls}}]}

    async def run():
        return await contract.wrap_full(original)(Service(), request(choice, **req_kwargs),
                                                   engine(), "id", parser="fixture")

    result = asyncio.run(run())
    assert closed == [True]
    return result


@pytest.mark.parametrize("calls", [[], [call(arguments='{"incomplete":')], [call("invalid")]])
def test_full_generator_rejects_parser_contract_violations(calls):
    assert_error(full_probe(calls))


def test_full_named_and_required_success_and_legacy_auto():
    for mode in ("named", "required"):
        result = full_probe([call()], choice=mode, finish="length")
        assert result["choices"][0]["finish_reason"] == "length"
    assert full_probe([], choice="auto")["choices"][0]["message"]["tool_calls"] == []


def frame(delta=None, finish=None, index=0):
    return "data: " + json.dumps({"choices": [{"index": index,
        "delta": delta or {}, "finish_reason": finish}]}) + "\n\n"


def tool_delta(name=None, arguments=None, index=0):
    function = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    return {"tool_calls": [{"index": index, "function": function}]}


def stream_probe(frames, *, choice="required", finish="stop", **req_kwargs):
    closed = []

    async def engine():
        try:
            yield NS(outputs=[NS(index=0, finish_reason=finish)])
        finally:
            closed.append("engine")

    async def original(self, req, generator):
        try:
            async for result in generator:
                for item in frames:
                    yield item
        finally:
            closed.append("stream")

    async def run():
        return [item async for item in contract.wrap_stream(original)(
            Service(), request(choice, **req_kwargs), engine())]

    result = asyncio.run(run())
    assert sorted(closed) == ["engine", "stream"]
    return result


@pytest.mark.parametrize("delta", [{}, tool_delta("lookup", '{"a":'), tool_delta("wrong", '{}')])
def test_stream_errors_replace_invalid_terminal_chunk_and_close_generators(delta):
    result = stream_probe([frame(delta), frame(finish="tool_calls"), "data: [DONE]\n\n"],
                          finish="length")
    assert result[0] == frame(delta)  # Prior deltas cannot be retracted.
    assert_error(json.loads(result[-2][6:]))
    assert result[-1] == "data: [DONE]\n\n"
    assert len(result) == 3


def test_stream_complete_length_is_not_hidden_by_tools_finish_reason():
    frames = [frame(tool_delta("look", '{"k":')), frame(tool_delta("up", '1}')),
              frame(finish="tool_calls"), "data: [DONE]\n\n"]
    result = stream_probe(frames, choice="named", finish="length")
    assert result[:2] == frames[:2]
    assert json.loads(result[-2][6:])["choices"][0]["finish_reason"] == "length"


def test_stream_valid_parallel_calls_and_parallel_disabled_violation():
    frames = [frame(tool_delta("lookup", '{}', index=0)),
              frame(tool_delta("lookup", '{}', index=1)),
              frame(finish="tool_calls"), "data: [DONE]\n\n"]
    assert stream_probe(frames, choice="named") == frames
    result = stream_probe(frames, choice="named", parallel_tool_calls=False)
    assert_error(json.loads(result[-2][6:]))


@pytest.mark.parametrize("frames", [[frame()], [frame(), "data: [DONE]\n\n"]])
def test_stream_missing_terminal_choice_or_done_is_an_error(frames):
    result = stream_probe(frames)
    assert_error(json.loads(result[-2][6:]))
    assert result[-1] == "data: [DONE]\n\n"


def test_auto_stream_and_existing_engine_errors_pass_through():
    frames = [frame(), frame(finish="length"), "data: [DONE]\n\n"]
    assert stream_probe(frames, choice="auto", finish="length") == frames
    frames = ['data: {"error":{"message":"Engine failed","type":"InternalServerError"}}\n\n',
              "data: [DONE]\n\n"]
    assert stream_probe(frames) == frames


@pytest.mark.parametrize("args,environ", [(NS(api_server_count=2), {}),
    (NS(data_parallel_size=2), {}), (NS(grpc=True), {}),
    (NS(data_parallel_multi_port_external_lb=True), {}),
    (NS(), {"VLLM_USE_RUST_FRONTEND": "1"})])
def test_bootstrap_rejects_unpatched_frontend_processes(args, environ):
    with pytest.raises(ValueError, match="one Python API frontend"):
        serve.validate_frontend(args, environ)


def test_source_hash_guard_and_double_install(tmp_path, monkeypatch):
    path = tmp_path / "serving.py"
    path.write_text("fixture")

    class Fake:
        async def chat_completion_full_generator(self):
            pass

        async def chat_completion_stream_generator(self):
            yield None

    module = NS(__file__=path, OpenAIServingChat=Fake)
    with pytest.raises(RuntimeError, match="not admitted"):
        serve.install(module)
    digest = serve.hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(serve.ADMITTED, digest, "fixture")
    serve.install(module)
    original = Fake.chat_completion_full_generator
    serve.install(module)
    assert Fake.chat_completion_full_generator is original
    serve.validate_frontend(NS(api_server_count=1, data_parallel_size=1), {})


def test_api_probe_reassembles_stream_fragments_and_preserves_error():
    body = (frame(tool_delta("lookup", '{"key":')) + frame(tool_delta(arguments='"cedar"}'))
            + frame(finish="tool_calls") + "data: [DONE]\n\n")
    error, choices, done = api_probe.decoded_response(body, True)
    assert error is None and done
    assert choices[0]["message"]["tool_calls"] == [call(arguments='{"key":"cedar"}')]
    error_frame = 'data: {"error":{"type":"ToolChoiceContractError"}}\n\n'
    error, choices, done = api_probe.decoded_response(error_frame + "data: [DONE]\n\n", True)
    assert error["type"] == "ToolChoiceContractError" and done and not choices


def test_api_probe_one_token_budget_and_named_contract():
    query = api_probe.payload("fixture-model", "named", True, True)
    assert query["tool_choice"]["function"]["name"] == "lookup"
    assert query["max_tokens"] == 1 and query["stream"] is True
    assert query["parallel_tool_calls"] is False
