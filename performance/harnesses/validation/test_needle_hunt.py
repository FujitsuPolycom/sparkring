"""CPU-only retrieval harness tests; no model or HTTP endpoint is contacted."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("validation_needle_test", HERE / "needle_hunt.py")
hunt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hunt
spec.loader.exec_module(hunt)


def config():
    return {"base_url": "http://192.0.2.1:8000", "model": "test", "contexts": [8192], "positions": [50],
            "modes": ["exact"], "seed": 20260723, "temperature": 1.0, "max_tokens": 512,
            "context_limit": 32768, "timeout": 1, "api_key": "dummy", "api_key_env": "TEST_KEY",
            "chat_template_kwargs": {}, "repetitions": 1}


@pytest.mark.parametrize("mode, expected_hash", [
    ("exact", "ae9d380420fa8fe2fca1f5ccc95efa36bb66469d8aa918b5fe4f367a31c14421"),
    ("revision", "e9888012a2d301aa14ecc7f51644e86dbea90a31ab7009d5f5229daa90b278f5"),
    ("join", "1991f9e648d1ae766e05eb74530feb9ce561516613ca2cbe7f09bd54e6ffbc52"),
])
def test_fixture_bytes_match_reference(mode, expected_hash):
    document, expected, _ = hunt.build_case_document(8192, 50, mode, 20260724)
    assert hashlib.sha256(document.encode()).hexdigest() == expected_hash
    assert document.count(expected) == 1


def fake_post(*, count=8000, finish="stop", answer=True):
    calls = []
    expected = hunt.build_case_document(8192, 50, "exact", 20260724)[1]
    def post(url, payload, key, timeout):
        calls.append((url, payload))
        if url.endswith("/tokenize"):
            return {"count": count}
        return {"choices": [{"message": {"content": expected if answer else "wrong"}, "finish_reason": finish}],
                "usage": {"prompt_tokens": count, "completion_tokens": 12}}
    return post, calls


def test_temperature_seed_and_tokenizer_guard():
    post, calls = fake_post()
    result = hunt.run_case(config(), 8192, 50, "exact", 1, 1, post=post)
    assert result["passed"]
    assert result["tokenized_prompt_tokens"] == 8000
    assert calls[1][1]["temperature"] == 1
    assert calls[1][1]["seed"] == 20260724
    assert calls[0][1]["messages"] == calls[1][1]["messages"]
    assert calls[0][1]["chat_template_kwargs"] == calls[1][1]["chat_template_kwargs"]


@pytest.mark.parametrize("count", [None, 0, -1, True, "8000", 32768])
def test_invalid_or_oversize_tokenization_blocks_chat(count):
    post, calls = fake_post(count=count)
    result = hunt.run_case(config(), 8192, 50, "exact", 1, 1, post=post)
    assert not result["passed"] and result["error"]
    assert len(calls) == 1


@pytest.mark.parametrize("finish", ["length", None, "tool_calls", "content_filter"])
def test_abnormal_finish_cannot_pass_even_with_answer(finish):
    post, _ = fake_post(finish=finish)
    result = hunt.run_case(config(), 8192, 50, "exact", 1, 1, post=post)
    assert result["answer_matches"]
    assert not result["passed"] and result["error"]
    assert result["output_budget_exhausted"] is (finish == "length")


def test_incremental_receipts_and_no_secret(tmp_path):
    post, _ = fake_post()
    path = tmp_path / "receipt.jsonl"
    assert hunt.execute(config(), path, post=post) == 0
    text = path.read_text()
    assert "dummy" not in text
    records = [json.loads(line) for line in text.splitlines()]
    assert [record["type"] for record in records] == ["start", "case", "summary"]
    assert records[-1]["success"]


def test_refuse_overwrite_before_request(tmp_path):
    path = tmp_path / "receipt.jsonl"
    path.write_text("preserve")
    def forbidden(*args):
        pytest.fail("Must not send a request before refusing an existing receipt")
    with pytest.raises(FileExistsError):
        hunt.execute(config(), path, post=forbidden)
    assert path.read_text() == "preserve"


@pytest.mark.parametrize("exception", [RuntimeError("HTTP 500"), TimeoutError("request timeout")])
def test_request_errors_fail_exit_and_remain_in_receipt(tmp_path, exception):
    def fail(*args):
        raise exception
    path = tmp_path / "receipt.jsonl"
    assert hunt.execute(config(), path, post=fail) == 2
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[-1]["errors"] == 1 and not records[-1]["success"]


def test_identical_repetition_reuses_prompt_seed(tmp_path):
    settings = {**config(), "repetitions": 2}
    post, calls = fake_post()
    path = tmp_path / "receipt.jsonl"
    assert hunt.execute(settings, path, post=post) == 0
    assert calls[1][1] == calls[3][1]
    cases = [json.loads(line) for line in path.read_text().splitlines() if json.loads(line)["type"] == "case"]
    assert cases[0]["prompt_sha256"] == cases[1]["prompt_sha256"]


@pytest.mark.parametrize("url", ["http://user:password@example.com", "http://example.com?key=x", "file:///tmp/x"])
def test_rejects_credential_urls(url):
    with pytest.raises(ValueError):
        hunt.base_url(url)
