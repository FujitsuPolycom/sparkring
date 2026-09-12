"""Offline request construction and numeric evidence checks for cache probes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location("cache_probe_contract", Path(__file__).with_name("model_cache.py"))
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


@pytest.mark.parametrize("kind,phase", [("semantic", "semantic"), ("persistent", "before-restart")])
def test_generated_prompt_uses_requested_expected_text(kind, phase, tmp_path, monkeypatch, capsys):
    expected = "CUSTOM_CANARY"
    monkeypatch.setattr(sys, "argv", ["model_cache", "--endpoint", "http://unused", "--model", "fixture",
                                    "--kind", kind, "--phase", phase, "--expected-text", expected,
                                    "--output", str(tmp_path / "output")])
    monkeypatch.setattr(cache, "fetch", lambda *args: pytest.fail("Network access forbidden"))
    cache.main()
    result = json.loads(capsys.readouterr().out)
    suffix = f"Respond with exactly {expected} and no other text."
    prompt = suffix if kind == "semantic" else "benchmark " * 8192 + "\n" + suffix
    assert result["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert result["executed"] is False
    assert not (tmp_path / "output").exists()


def test_metric_overflow_is_not_serialized_as_infinity():
    assert cache.cache_metrics("prefix_hit_total 1e999\nexternal_hit_total -1e999\nprefix_valid 12.5\n") == {"prefix_valid": 12.5}


def test_metric_difference_overflow_is_omitted_from_json_evidence():
    result = cache.cache_metric_deltas({'prefix_hits': -1e308, 'prefix_valid': 5},
                                      {'prefix_hits': 1e308, 'prefix_valid': 8})
    assert json.loads(json.dumps(result, allow_nan=False)) == {'prefix_valid': 3}


@pytest.mark.parametrize("value", ["-", ".", "1e", "++2"])
def test_malformed_metric_numbers_are_ignored(value):
    assert cache.cache_metrics(f"prefix_hits {value}\n") == {}

def test_authenticated_requests_keep_credentials_out_of_artifacts(tmp_path, monkeypatch, capsys):
    import io
    from types import SimpleNamespace
    key = 'private-test-bearer'
    keyfile = tmp_path / 'keys'
    keyfile.write_text('\n' + key + '\nsecond-key\n')
    output = tmp_path / 'output'
    requests = []
    def open_request(request, timeout):
        assert timeout == 120
        assert request.get_header('Authorization') == 'Bearer ' + key
        requests.append(request)
        body = (json.dumps({'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]})
                if request.data is not None else 'prefix_hits 1\n')
        return io.BytesIO(body.encode())
    monkeypatch.setattr(cache.urllib.request, 'build_opener', lambda handler: SimpleNamespace(open=open_request))
    monkeypatch.setattr(sys, 'argv', ['probe', '--endpoint', 'http://fixture', '--model', 'fixture',
        '--kind', 'semantic', '--expected-text', 'OK', '--api-key-file', str(keyfile),
        '--output', str(output), '--execute-authorized'])
    cache.main()
    assert len(requests) == 3
    assert key not in capsys.readouterr().out
    assert all(key not in p.read_text() for p in output.iterdir())


def test_authenticated_plan_does_not_read_key_or_make_request(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cache, 'read_api_key', lambda path: pytest.fail('plan read secret'))
    monkeypatch.setattr(cache, 'fetch', lambda *args, **kwargs: pytest.fail('plan contacted endpoint'))
    monkeypatch.setattr(sys, 'argv', ['probe', '--endpoint', 'http://fixture', '--model', 'fixture',
        '--kind', 'semantic', '--api-key-file', str(tmp_path / 'absent'), '--output', str(tmp_path / 'out')])
    cache.main()
    assert json.loads(capsys.readouterr().out)['executed'] is False


@pytest.mark.parametrize('content', ['', '\n', 'a b', 'a\tb', 'nonascii-\u00e9'])
def test_malformed_key_file_rejected_without_echoing_content(tmp_path, content):
    path = tmp_path / 'keys'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError, match='API key file requires'):
        cache.read_api_key(path)


def test_authenticated_redirect_is_rejected():
    request = cache.urllib.request.Request('http://fixture/v1/chat/completions')
    with pytest.raises(cache.urllib.error.HTTPError, match='do not follow redirects'):
        cache._NoCredentialRedirect().redirect_request(request, None, 302, 'Found', {}, 'http://other/')
