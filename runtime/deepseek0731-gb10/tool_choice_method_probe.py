"""CPU validation of the complete patched nonstreaming tool-result method.

The source method executes unchanged. Engine output, parser output, response
models, and external helper interfaces are fixtures. This does not load vLLM,
exercise HTTP routing, or qualify a serving image.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
from http import HTTPStatus
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace as Record
from unittest.mock import patch

EXPECTED_SHA256 = '4c129414f20ecbb44ea8613d774614e8a66152164a039def7c5c086587a1cc99'
FLAG = 'SPARKRING_REJECT_EMPTY_REQUIRED_TOOL_CALLS'


class NamedChoice:
    """Stand-in preserving the source method's exact named-choice type check."""


def load_method(source):
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == EXPECTED_SHA256, (digest, EXPECTED_SHA256)
    tree = ast.parse(raw)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
               and node.name == 'chat_completion_full_generator']
    assert len(matches) == 1
    method = matches[0]
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        'asyncio': asyncio, 'time': time, 'os': os, 'HTTPStatus': HTTPStatus,
        'ChatCompletionNamedToolChoiceParam': NamedChoice,
        'ChatMessage': Record, 'ToolCall': Record,
        'ChatCompletionResponseChoice': Record, 'ChatCompletionResponse': Record,
        'UsageInfo': Record, 'make_tool_call_id': lambda: 'fixture-generated-id',
        'maybe_filter_parallel_tool_calls': lambda choice, request: choice,
        '_make_prompt_tokens_details': lambda *args: None,
        'clamp_prompt_logprobs': lambda value: value,
        'as_list': list,
    }
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace[method.name], digest, method.lineno, method.end_lineno


async def run_case(method, *, choice, flag, parser_calls, expected_error,
                   parser_present=True, multiple_outputs=False):
    request = Record(tool_choice=NamedChoice() if choice == 'named' else choice,
        logprobs=False, include_reasoning=True, tools=[Record(name='lookup')],
        return_token_ids=False, echo=False, return_prompt_text=False)
    outputs = [Record(text='fixture output', token_ids=[1, 2], logprobs=None,
        finish_reason='stop', routed_experts=None, index=0, stop_reason=None)]
    if multiple_outputs:
        outputs.append(Record(**{**vars(outputs[0]), 'index': 1}))
    final = Record(outputs=outputs, prompt_token_ids=[10, 11, 12],
        encoder_prompt_token_ids=None, num_cached_tokens=0,
        prompt_logprobs=None, kv_transfer_params=None, ec_transfer_params=None)
    parse_requests = []
    values = iter(parser_calls) if multiple_outputs else None

    def parse(text, received_request, **kwargs):
        assert received_request is request
        assert kwargs['enable_auto_tools'] is True
        assert kwargs['model_output_token_ids'] == [1, 2]
        parse_requests.append(received_request)
        return 'fixture reasoning', '', next(values) if multiple_outputs else parser_calls

    errors = []

    def error(message, **kwargs):
        result = Record(message=message, **kwargs)
        errors.append(result)
        return result

    def check_finish(reason, request_id):
        assert reason == 'stop' and request_id == 'fixture-request'

    service = Record(parser_cls=Record(tool_parser_cls=object), enable_auto_tools=True,
        get_chat_request_role=lambda request: 'assistant', _raise_if_error=check_finish,
        create_error_response=error, enable_prompt_tokens_details=False,
        enable_per_request_metrics=False, enable_log_outputs=False,
        system_fingerprint='cpu-fixture')

    async def engine_results():
        yield final

    metadata = Record()
    environment = dict(os.environ)
    environment.pop(FLAG, None)
    if flag is not None:
        environment[FLAG] = flag
    with patch.dict(os.environ, environment, clear=True):
        result = await method(service, request, engine_results(), 'fixture-request',
            'fixture-model', [], None, metadata,
            parser=Record(parse=parse) if parser_present else None)
    assert len(parse_requests) == (len(outputs) if parser_present else 0)
    if expected_error:
        assert len(errors) == 1 and result is errors[0]
        assert result.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert result.err_type == 'ToolChoiceContractError'
        assert not hasattr(result, 'choices')
        assert not hasattr(metadata, 'final_usage_info')
    else:
        assert errors == []
        assert len(result.choices) == 1
        assert result.usage.completion_tokens == 2
        assert metadata.final_usage_info is result.usage
        calls = getattr(result.choices[0].message, 'tool_calls', None)
        if parser_calls:
            assert len(calls) == len(parser_calls)
            assert calls[0].function is parser_calls[0]
            assert calls[0].id == (parser_calls[0].id or 'fixture-generated-id')
        elif choice in ('named', 'required'):
            assert calls == []
        else:
            assert calls is None
    call_description = ([None if calls is None else [vars(call) for call in calls]
                         for calls in parser_calls] if multiple_outputs else
                        None if parser_calls is None else [vars(call) for call in parser_calls])
    return {'choice': choice, 'flag': flag, 'parser_present': parser_present,
            'parser_calls': call_description,
            'outputs': len(outputs), 'expected_error': expected_error, 'passed': True}


async def main(source):
    method, digest, start, end = load_method(source)
    results = []
    for choice in ('named', 'required'):
        for calls in (None, []):
            for flag in (None, '0', '1'):
                results.append(await run_case(method, choice=choice, flag=flag,
                    parser_calls=calls, expected_error=flag == '1'))
        for flag in (None, '1'):
            for call_id in (None, 'parser-id'):
                calls = [Record(id=call_id, name='lookup', arguments='{"key":"value"}')]
                results.append(await run_case(method, choice=choice, flag=flag,
                    parser_calls=calls, expected_error=False))
        results.append(await run_case(method, choice=choice, flag='1',
            parser_calls=None, parser_present=False, expected_error=True))
    for calls in (None, [Record(id='parser-id', name='lookup', arguments='{}')]):
        results.append(await run_case(method, choice='auto', flag='1',
            parser_calls=calls, expected_error=False))
    results.append(await run_case(method, choice='required', flag='1',
        parser_calls=[[Record(id='parser-id', name='lookup', arguments='{}')], []],
        multiple_outputs=True, expected_error=True))
    print(json.dumps({'status': 'passed', 'cases_passed': len(results),
        'python': platform.python_implementation(), 'python_version': platform.python_version(),
        'source': 'vllm/entrypoints/openai/chat_completion/serving.py', 'source_sha256': digest,
        'method_start_line': start, 'method_end_line': end,
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limits': 'Fake engine/parser outputs and response/helper interfaces; no HTTP, tokenizer, model, image, GPU, or streaming qualification.',
        'cases': results}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Patched vLLM chat-completion serving module')
    asyncio.run(main(parser.parse_args().source))
