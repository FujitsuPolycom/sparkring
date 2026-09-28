"""Readiness polling, functional scoring and correctness-screen scoring against an in-memory server."""
from __future__ import annotations

import struct
import zlib

from performance.harnesses.acceptance import checks
from performance.harnesses.acceptance.fakes import FakeModel
from performance.harnesses.acceptance.runners import HttpError

BASE = "http://node-a.test:8000/v1"
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
ALL = frozenset({"tools", "image", "reasoning"})


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def chat(model, thinking_off=OFF):
    return checks.Chat(model, BASE, model.served, thinking_off=thinking_off, thinking_on={})


def test_readiness_waits_until_the_profile_model_is_listed():
    clock = Clock()
    result = checks.wait_ready(FakeModel(fail_models=3), BASE, expected="Model-TP2", timeout=600, interval=10,
                               clock=clock, sleep=clock.sleep, log=lambda line: None)
    assert result == {"served_models": ["Model-TP2"], "expected": "Model-TP2", "seconds": 30.0, "attempts": 4,
                      "ok": True}


def test_readiness_rejects_another_model_and_times_out():
    clock = Clock()
    other = checks.wait_ready(FakeModel("Other-TP2"), BASE, expected="Model-TP2", timeout=600, interval=10,
                              clock=clock, sleep=clock.sleep, log=lambda line: None)
    assert not other["ok"] and "Other-TP2" in other["error"]
    down = checks.wait_ready(FakeModel(fail_models=100), BASE, expected="Model-TP2", timeout=60, interval=10,
                             clock=clock, sleep=clock.sleep, log=lambda line: None)
    assert not down["ok"] and "connection refused" in down["error"] and down["attempts"] == 7


def test_all_functional_checks_pass_with_the_profile_thinking_switch():
    model = FakeModel()
    result = checks.run_functional(chat(model, {"chat_template_kwargs": {"thinking": False}}), features=ALL)
    assert [r["status"] for r in result["checks"]] == ["PASS"] * 7
    assert result["ok"] and result["passed"] == 7
    bodies = [body for method, _, body in model.requests]
    assert all(b["temperature"] == 0 and b["model"] == "Model-TP2" for b in bodies)
    assert [b.get("chat_template_kwargs") for b in bodies] == [{"thinking": False}] * 6 + [None]
    assert bodies[-1]["max_tokens"] == 2048
    lines = checks.functional_text(result).splitlines()
    assert lines[0] == "PASS count: '1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20'"
    assert lines[-2].startswith("PASS thinking on: {'reasoning_chars': ") and lines[-1] == "0 failed"


def test_unsupported_and_requested_skips_send_no_request():
    model = FakeModel()
    result = checks.run_functional(chat(model), features=frozenset({"reasoning"}), skip={"code"})
    status = {r["name"]: (r["status"], r["detail"]) for r in result["checks"]}
    assert status["image"] == ("SKIP", "the profile accepts no image input")
    assert status["tool call"][0] == status["forced tool call"][0] == "SKIP"
    assert status["code"] == ("SKIP", "skipped on request")
    assert result["ok"] and result["skipped"] == 4 and len(model.requests) == 3
    assert "SKIP image: the profile accepts no image input\n" in checks.functional_text(result)


def test_wrong_replies_and_request_errors_fail_single_checks():
    model = FakeModel(overrides={
        "17*23": {"content": "The answer is 391."},
        "Paris": {"content": "It is sunny.", "tool_calls": []},
        "two colored halves": HttpError("HTTP 400: image input is not supported"),
        "is_prime": {"content": "def is_prime(n): pass"}})
    result = checks.run_functional(chat(model), features=ALL)
    status = {r["name"]: r["status"] for r in result["checks"]}
    assert status == {"count": "PASS", "arithmetic": "FAIL", "code": "PASS", "tool call": "FAIL",
                      "forced tool call": "PASS", "image": "FAIL", "thinking on": "FAIL"}
    assert not result["ok"] and result["failed"] == 4
    image = next(r for r in result["checks"] if r["name"] == "image")
    assert image["detail"] == "request failed: HTTP 400: image input is not supported"


def test_thinking_on_requires_reasoning_text():
    model = FakeModel(overrides={"17*23": {"content": "391"}})
    ok, detail = checks.check_thinking_on(chat(model))
    assert not ok and detail == {"reasoning_chars": 0, "content": "391"}


def test_image_is_a_two_color_png():
    data = checks.two_color_png(4, 2)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", data[16:24])
    idat = data.index(b"IDAT")
    size = struct.unpack(">I", data[idat - 4:idat])[0]
    assert (width, height) == (4, 2)
    assert zlib.decompress(data[idat + 4:idat + 4 + size]) == (b"\x00" + b"\xff\x00\x00" * 2 + b"\x00\x00\xff" * 2) * 2


def test_screen_requests_match_the_published_screen():
    items = checks.round_items(0)
    assert len(items) == 32 and sum(qid.startswith("L") for qid, _, _ in items) == 8
    assert [qid for qid, _, _ in items] == [qid for qid, _, _ in checks.round_items(0)]
    qid, prompt, pattern = checks.needle_question(1)
    assert qid == "L1" and "the vault access code is 17919." in prompt and pattern == r"\b17919\b"


def test_screen_counts_are_exclusive():
    clock = Clock()
    model = FakeModel(overrides={
        "capital of Australia": {"content": "Canberra " + "the " * 9},
        "2^20": HttpError("HTTP 500: engine dead"),
        "remainder when 1000": {"content": "5"}})
    summary, responses = checks.run_stress(chat(model), rounds=2, clock=clock)
    assert (summary["n"], summary["degenerate"], summary["errors"], summary["wrong"]) == (64, 2, 2, 2)
    assert summary["wrong_ids"] == ["a7"] and summary["wrong_samples"] == [["a7", "5"], ["a7", "5"]]
    assert summary["degenerate_items"][0][1] == "f1" and summary["error_samples"][0][0] == "a2"
    assert not summary["ok"] and len(responses) == 64
    assert all(body["chat_template_kwargs"] == {"enable_thinking": False} and body["max_tokens"] == 300
               for _, _, body in model.requests)


def test_clean_screen_passes():
    summary, _ = checks.run_stress(chat(FakeModel()), rounds=1)
    assert summary["ok"] and (summary["n"], summary["degenerate"], summary["wrong"], summary["errors"]) == (32, 0, 0, 0)
    assert summary["seconds"] >= 0
