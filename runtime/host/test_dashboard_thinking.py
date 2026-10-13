"""Thinking labels from shipped records and existing observations, without serving requests."""
import json

import pytest

from runtime.host import dashboard, dashboard_tui
from runtime.host.test_dashboard import collector, runtime_document


GLM = "glm53-flash-nvfp4-spark-tp2"
QWEN = "qwen38-flash-next-tp2"


def test_launch_kwargs_are_bounded_public_facts():
    from runtime.host.node_containers import launch_facts, thinking_kwargs
    kwargs = {"enable_thinking": False, "reasoning_effort": "high", "secret": "PRIVATE"}
    facts = launch_facts(["vllm", "serve", "M", "--default-chat-template-kwargs", json.dumps(kwargs)])
    assert facts["default_chat_template_kwargs"] == {"enable_thinking": False, "reasoning_effort": "high"}
    assert "PRIVATE" not in json.dumps(facts)
    assert thinking_kwargs('{"thinking": false}') == {"thinking": False}
    assert thinking_kwargs({"reasoning_effort": "x" * 100, "enable_thinking": "false"}) == {}
    for invalid in ("x" * 20000, "bad json", "[]", [], False):
        assert thinking_kwargs(invalid) is None


@pytest.mark.parametrize("profile,kwargs,expected", [
    (GLM, {"reasoning_effort": "high"}, "high (deployment default)"),
    (GLM, None, "max (model default)"),
    (GLM, {"enable_thinking": False}, "max (model default)"),
    (QWEN, {"enable_thinking": False, "reasoning_effort": "low"}, "off (deployment default)"),
    (QWEN, None, "xhigh (model default)"),
    (QWEN, {"enable_thinking": True}, "xhigh (deployment default)"),
    ("mimo-v26-flash-mopd-tp2", None, "on (model default)"),
    ("mimo-v26-flash-mopd-tp2", {"enable_thinking": True}, "on (deployment default)"),
    ("deepseek-v41-flash-tp4", {"thinking": False}, "off (deployment default)"),
    ("deepseek-v41-flash-tp4", {"enable_thinking": False}, "off (deployment default)"),
    ("deepseek-v41-flash-tp4", {"reasoning_effort": "none"}, "off (deployment default)"),
    ("deepseek-v41-flash-tp4", {"reasoning_effort": 75}, "75 (deployment default)"),
    ("unknown", {"reasoning_effort": "high"}, None),
])
@pytest.mark.parametrize("encoded", [False, True])
def test_collector_and_terminal_defaults(tmp_path, profile, kwargs, expected, encoded):
    found, api, clock = collector(tmp_path)
    path = found.sources[0].site_path
    site = json.loads(path.read_text())
    site["deployments"][0]["profile"] = profile
    path.write_text(json.dumps(site))
    row = site["deployments"][0]
    runtime = runtime_document()
    runtime["effective"]["default_chat_template_kwargs"] = {
        "state": "known", "value": json.dumps(kwargs) if encoded and kwargs is not None else kwargs}
    api.bodies[(row["api_host"], row["api_port"], "/v1/sparkring/status")] = json.dumps(runtime).encode()
    found.cycle()
    requests = list(api.requests)
    member = found.snapshot()["groups"][0]["members"][0]
    assert member["thinking"] == expected
    text = "\n".join("".join(s for s, _ in line) for line in dashboard_tui.model_details(member, clock.now))
    assert ("Thinking: " + expected in text) if expected else "Thinking:" not in text
    assert api.requests == requests


def test_unknown_checkpoint_and_saved_settings():
    assert dashboard.thinking_line({"profile": GLM, "checkpoint": "unrecorded"}, {}) is None
    row = {"profile": GLM, "serving": {"reasoning_effort": "high"}}
    assert dashboard.thinking_line(row, {}) == "high (deployment default)"
    assert dashboard.thinking_line(row, {"default_chat_template_kwargs": {}}) == "max (model default)"
    assert dashboard.thinking_line(row, {"default_chat_template_kwargs": {"reasoning_effort": "low"}}) == "low (deployment default)"
    assert dashboard.thinking_line({"profile": QWEN, "serving": {"thinking": "off"}}, {}) == "off (deployment default)"
    for invalid in ("bad json", "[]", [], False):
        assert dashboard.thinking_line({"profile": GLM}, {"default_chat_template_kwargs": invalid}) == "max (model default)"


@pytest.mark.parametrize("kwargs,expected", [
    (None, "model default"), ({}, "model default"),
    ({"reasoning_effort": "high"}, "high (deployment default)"),
    ({"enable_thinking": False}, "off (deployment default)"),
    ({"thinking": False}, "off (deployment default)"),
    ({"enable_thinking": True}, "on (deployment default)"),
])
@pytest.mark.parametrize("source", ["launch", "runtime"])
def test_unmanaged_observed_defaults(tmp_path, kwargs, expected, source):
    from runtime.host.test_dashboard_unmanaged import reports, install_reports
    found, api, clock = collector(tmp_path, rows=())
    docs = reports(found, clock)
    runtime = runtime_document()
    if source == "launch":
        docs[0]["model_containers"][0]["default_chat_template_kwargs"] = kwargs
    else:
        runtime["effective"]["default_chat_template_kwargs"] = {"state": "known", "value": kwargs}
        docs[0]["model_containers"][0]["default_chat_template_kwargs"] = {"reasoning_effort": "low"}
        if kwargs is None:
            docs[0]["model_containers"][0].pop("default_chat_template_kwargs")
    host = found.sources[0].fabric["positions"][0]["address"]
    api.bodies[(host, 9123, "/v1/sparkring/status")] = json.dumps(runtime).encode()
    install_reports(found, docs, tmp_path)
    found.cycle()
    assert found.snapshot()["groups"][0]["members"][0]["thinking"] == expected
