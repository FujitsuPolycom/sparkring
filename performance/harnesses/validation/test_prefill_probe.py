import importlib.util
import io
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('prefill_probe', Path(__file__).with_name('prefill_probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_counts_and_unique_prefix():
    assert probe.tokens('128k') == 131072
    assert probe.tokens('1m') == 1048576
    assert probe.build_text('first', 1000) != probe.build_text('second', 1000)
    assert len(probe.build_text('first', 1000)) == 1000


def test_first_token_and_events():
    assert not probe.has_token({'choices': [{'delta': {'role': 'assistant'}}]})
    assert probe.has_token({'choices': [{'delta': {'reasoning_content': 'think'}}]})
    assert probe.has_token({'choices': [{'delta': {'content': 'OK'}}]})
    data = b': keepalive\n\ndata: {"usage":{"prompt_tokens":2}}\n\ndata: [DONE]\n'
    assert list(probe.events(io.BytesIO(data))) == [{'usage': {'prompt_tokens': 2}}]


def test_incomplete_stream_rejected():
    with pytest.raises(ValueError, match='terminator'):
        list(probe.events(io.BytesIO(b'data: {"choices":[]}\n')))


def test_stream_error_rejected():
    with pytest.raises(ValueError, match='streaming error'):
        list(probe.events(io.BytesIO(b'data: {"error":{"message":"failed"}}\ndata: [DONE]\n')))


def test_single_completion_request_cannot_merge_multiple_choices():
    raw = b'data: {"choices":[{"delta":{"content":"one"}},{"delta":{"content":"two"}}]}\ndata: [DONE]\n'
    with pytest.raises(ValueError, match='choice'):
        list(probe.events(io.BytesIO(raw)))


def test_context_guard_sends_no_completion(monkeypatch):
    calls = []
    def request(url, *args):
        calls.append(url)
        return io.BytesIO(b'{"count":1000}')
    monkeypatch.setattr(probe, 'request', request)
    with pytest.raises(ValueError, match='exceeds'):
        probe.measure('http://example', 'model', 1000, 1000, 1, '', 1)
    assert calls == ['http://example/tokenize']


def test_temperature_usage_and_cache_evidence(monkeypatch):
    payloads = []
    def request(url, payload, *args):
        payloads.append(payload)
        if url.endswith('/tokenize'):
            return io.BytesIO(b'{"count":1000}')
        return io.BytesIO(b'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"length"}]}\n'
                          b'data: {"usage":{"prompt_tokens":1000,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":0}}}\n'
                          b'data: [DONE]\n')
    monkeypatch.setattr(probe, 'request', request)
    result = probe.measure('http://example', 'model', 1000, 2048, 1, '', 1)
    assert payloads[-1]['temperature'] == 1 and payloads[-1]['max_tokens'] == 1
    assert result['valid'] and result['cold_prefix_confirmed']


@pytest.mark.parametrize('change', ['missing-finish', 'failed-finish', 'missing-completion',
                                   'bool-prompt', 'bool-cache', 'negative-cache'])
def test_incomplete_or_invalid_usage_cannot_qualify_prefill(monkeypatch, change):
    choice = {'delta': {'content': 'OK'}, 'finish_reason': 'length'}
    usage = {'prompt_tokens': 1000, 'completion_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 0}}
    if change == 'missing-finish':
        choice.pop('finish_reason')
    elif change == 'failed-finish':
        choice['finish_reason'] = 'error'
    elif change == 'missing-completion':
        usage.pop('completion_tokens')
    elif change == 'bool-prompt':
        usage['prompt_tokens'] = True
    elif change == 'bool-cache':
        usage['prompt_tokens_details']['cached_tokens'] = False
    else:
        usage['prompt_tokens_details']['cached_tokens'] = -1
    def request(url, *args):
        if url.endswith('/tokenize'):
            return io.BytesIO(b'{"count":1000}')
        return io.BytesIO(('data: ' + json.dumps({'choices': [choice], 'usage': usage})
                           + '\ndata: [DONE]\n').encode())
    monkeypatch.setattr(probe, 'request', request)
    with pytest.raises(ValueError):
        probe.measure('http://example', 'model', 1000, 2048, 1, '', 1)
