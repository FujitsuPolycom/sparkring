"""``sparkring check`` and its tester report; the API, the Sparks and Node A's records are simulated."""
import json
from pathlib import Path

import pytest

from runtime.common import fabric_document, installer
from runtime.common.test_transport import TP2, document, sircl_deployment
from runtime.host import check, controller, node, transport_receipts
from runtime.host.test_transport_receipts import pair_reports

VERDICT_KEYS = ("verdict", "nccl_observed", "problems", "receipts")


class Chat:
    """An OpenAI-style API that answers the functional checks correctly."""

    def __init__(self, broken=()):
        self.broken = set(broken)
        self.requests = []

    def get_json(self, url, *, timeout):
        return {"data": [{"id": "model"}]}

    def post_json(self, url, body, *, timeout):
        self.requests.append(body)
        content = body["messages"][0]["content"]
        text = json.dumps(content) if not isinstance(content, str) else content
        if "Count from 1 to 20" in text:
            reply = ", ".join(str(n) for n in range(1, 21))
        elif "17*23" in text:
            reply = "391"
        elif "is_prime" in text:
            reply = "def is_prime(n):\n    return n > 1"
        elif "colored halves" in text:
            reply = "red and blue"
        else:
            reply = "Paris"
        if any(word in text for word in self.broken):
            reply = "no"
        message = {"content": reply}
        if body.get("tools"):
            city = "Paris" if "Paris" in text else "Rome"
            message = {"content": None, "tool_calls": [{"function": {"name": "get_weather",
                                                                     "arguments": json.dumps({"city": city})}}]}
        if body.get("max_tokens") == 2048:
            message["reasoning_content"] = "17 times 23"
        return {"choices": [{"message": message}]}


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A SIRCL pair deployment on Node A with its fabric document and a passing receipt check."""
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1])
    directory = controller.STATE / "deployments" / "glm-sircl"
    directory.mkdir(parents=True)
    installer.write(directory / "deployment.lock.json", lock)
    value = document("pair", 2)
    (controller.STATE / "fabric.json").write_text(fabric_document.encoded(value))
    node.save(controller.STATE, "cluster.json", {"name": "test", "api_address": "192.0.2.80", "plan": {
        "spec": {"hosts": [{"host": f"code@192.0.2.{20 + n}", "management_address": f"192.0.2.{20 + n}"}
                           for n in range(2)]},
        "nodes": [{"hostname": row["hostname"]} for row in value["positions"]]}})
    monkeypatch.setattr(check, "deployments", lambda on=None: [(None, directory)])

    def transport(path, operation, *, cache):
        assert operation == "transport"
        reports = pair_reports()
        return transport_receipts.record(path, reports, transport_receipts.evaluate(lock, reports))
    monkeypatch.setattr(check.retained_source, "apply", transport)
    return directory, lock, value


def test_a_check_runs_the_functional_requests_and_the_receipt_verdict(recorded):
    directory, lock, _ = recorded
    chat = Chat()
    result = check.run(client=chat, say=lambda line: None)
    row = result["deployments"][0]
    assert result["schema"] == check.SCHEMA and result["ok"] and row["ok"]
    assert row["functional"]["failed"] == 0 and row["functional"]["passed"] >= 3
    assert row["transport"]["verdict"] == "as-expected" and transport_receipts.latest(directory)["verdict"] == "as-expected"
    assert all(body["temperature"] == 0 for body in chat.requests)


def test_a_failed_functional_check_or_a_differing_verdict_fails_the_check(recorded, monkeypatch):
    result = check.run(client=Chat(broken=["17*23"]), say=lambda line: None)
    assert not result["ok"] and result["deployments"][0]["functional"]["failed"] >= 1
    monkeypatch.setattr(check.retained_source, "apply",
                        lambda path, operation, *, cache: {"backend": "sircl", "verdict": "differs",
                                                           "problems": ["rank 1: PyNccl built"]})
    lines = []
    result = check.run(client=Chat(), say=lines.append)
    assert not result["ok"] and any("Transport check failed: rank 1: PyNccl built" in line for line in lines)


def test_a_prepared_deployment_has_no_receipt_check(recorded, monkeypatch):
    directory, lock, _ = recorded
    installer.write(directory / "plain.lock.json", {key: value for key, value in lock.items() if key != "transport"})
    (directory / "deployment.lock.json").unlink()
    (directory / "plain.lock.json").rename(directory / "deployment.lock.json")
    monkeypatch.setattr(check.retained_source, "apply", lambda *a, **k: pytest.fail("no receipt check expected"))
    result = check.run(client=Chat(), say=lambda line: None)
    assert result["deployments"][0]["transport"] == {"backend": "prepared"}


def test_an_nccl_deployment_has_no_receipt_check_and_its_transport_line_names_nccl(recorded, monkeypatch):
    directory, _, _ = recorded
    from runtime.common.test_transport_nccl import nccl_deployment
    lock, _ = nccl_deployment(TP2, "pair", 2, [0, 1])
    (directory / "deployment.lock.json").unlink()
    installer.write(directory / "deployment.lock.json", lock)
    monkeypatch.setattr(check.retained_source, "apply", lambda *a, **k: pytest.fail("no receipt check expected"))
    lines = []
    result = check.run(client=Chat(), say=lines.append)
    row = result["deployments"][0]
    assert row["transport"] == {"backend": "nccl"} and row["ok"]
    assert "  Transport: nccl" in lines
    assert transport_receipts.text({"backend": "nccl"}) == "Transport: nccl"


def test_the_report_replaces_private_items_and_keeps_fabric_addresses(recorded, tmp_path, monkeypatch):
    directory, lock, value = recorded
    monkeypatch.setattr(check, "environment", lambda lock=None: {"driver": "580.95", "docker": "28.3.0"})
    installer.write(directory / "install-result.json", {
        "api_url": "http://192.0.2.80:8015/v1", "deployment": str(directory), "host": "code@192.0.2.21",
        "mac": value["positions"][0]["ports"]["0"]["functions"]["primary"]["mac"],
        "fabric": value["positions"][0]["ports"]["0"]["functions"]["primary"]["address"],
        "hostname": value["positions"][1]["hostname"]})
    result = check.run(client=Chat(), say=lambda line: None)
    written = check.report(tmp_path / "out", result)
    root = Path(written["directory"])
    assert root.name.startswith("sparkring-report-") and (root / "report.json").is_file()
    assert {"check.json", "environment.json", "fabric-show.json", "status.json"} <= set(written["files"])
    assert any(name.startswith("receipts/glm-sircl/rank0-tp.json") for name in written["files"])
    assert "logs/glm-sircl/rank0.log" in written["files"]
    saved = (root / "install-result-glm-sircl.json").read_text()
    assert "192.0.2.80" not in saved and "LAN_ADDRESS" in saved and "USER@SPARK_1" in saved
    assert value["positions"][1]["hostname"] not in saved and "MAC_1" in saved
    assert value["positions"][0]["ports"]["0"]["functions"]["primary"]["address"] in saved
    for name in written["files"]:
        text = (root / name).read_text()
        assert "code@" not in text and value["positions"][0]["hostname"] not in text, name
