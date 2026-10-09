"""``sparkring install`` choosing SIRCL ring sessions; host and SSH boundaries simulated as in test_install_workflow.

A cluster recorded with a fabric document installs on SIRCL with NCCL off
when its image carries the SIRCL layer (an image lock v3); the plan prints the
transport, the deployment lock records its ``transport`` section, the result
carries the receipt verdict and the summary card its Transport line.
"""
import json
from pathlib import Path

import pytest

from runtime.common import fabric_document, installer
from runtime.common.test_image_lock import sircl_lock
from runtime.host import controller, fabric, install_workflow as flow, relays
from runtime.host.test_install_workflow import PROFILE, machine, sparks  # noqa: F401  (pytest fixtures)
from scripts import sparkring

MARKER = {"binary": relays.MARKER_BINARY, "sha256": "ab" * 32}
VERDICT = {"schema": "sparkring-transport-verdict/v1", "backend": "sircl", "nccl": "never",
           "expected": "SIRCL carries every collective; NCCL creates no communicator", "verdict": "as-expected",
           "nccl_observed": "absent", "problems": [], "receipts": "/var/lib/sparkring/controller/receipts/x",
           "groups": {"tp": {"rows": []}}, "checked_at": "2026-10-08T10:00:00Z"}


@pytest.fixture
def sircl(machine, tmp_path, monkeypatch):  # noqa: F811
    """The machine's cluster with its fabric document recorded, a SIRCL image lock and a passing receipt check."""
    events, previous, assets, operation = machine
    cluster = installer.read(controller.STATE / "cluster.json")
    document, _ = fabric.prepare(cluster["plan"], cluster="test", marker=MARKER)
    (controller.STATE / "fabric.json").write_text(fabric_document.encoded(document))
    lock = tmp_path / "sircl-image.json"
    lock.write_text(json.dumps(sircl_lock()))

    def apply(path, action, **options):
        if action == "transport":
            events.append("candidate:transport")
            return dict(VERDICT)
        return operation(path, action, **options)
    monkeypatch.setattr(flow.retained_source, "apply", apply)
    return events, lock, document


def install(lock, *extra):
    return sparkring.main(["install", "--profile", PROFILE, "--yes", "--json", "--image-lock", str(lock), *extra])


def deployment(result):
    return installer.load(result["deployment"])


def test_an_image_with_sircl_and_a_recorded_fabric_install_on_sircl_with_nccl_off(sircl, capsys):
    events, lock, document = sircl
    assert install(lock) == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    recorded = deployment(result)["transport"]
    assert recorded["backend"] == "sircl" and recorded["nccl"] == "never"
    assert recorded["fabric"]["id"] == document["id"] and recorded["group"]["name"] == "pair"
    assert result["transport"]["backend"] == "sircl" and result["transport"]["verdict"] == "as-expected"
    assert events.index("candidate:up") < events.index("candidate:transport")
    assert ("Transport: sircl on every collective, NCCL off (default table, pair: the design's settings, "
            "not measured)") in out.err
    assert "  Transport:   sircl, NCCL: absent" in out.err
    # The result stays with the deployment for sparkring check --report.
    saved = json.loads((Path(result["deployment"]) / flow.RESULT_FILE).read_text())
    assert saved["transport"]["verdict"] == "as-expected"


def test_the_prepared_transport_and_nccl_auto_are_separate_deployments(sircl, capsys):
    _, lock, _ = sircl
    assert install(lock) == 0
    default = json.loads(capsys.readouterr().out)
    assert install(lock, "--transport", "prepared") == 0
    prepared = json.loads(capsys.readouterr().out)
    assert "transport" not in deployment(prepared) and prepared["transport"] == {"backend": "prepared",
                                                                                 "reason": None}
    assert install(lock, "--nccl", "topology") == 0
    out = capsys.readouterr()
    automatic = json.loads(out.out)
    assert deployment(automatic)["transport"]["nccl"] == "auto"
    assert len({default["deployment"], prepared["deployment"], automatic["deployment"]}) == 3
    # The repeat command names the operator's choice.
    assert automatic["checkpoint"]["command"].endswith("--nccl auto")


def test_without_a_fabric_document_the_installation_runs_on_the_prepared_transport(sircl, capsys):
    _, lock, _ = sircl
    (controller.STATE / "fabric.json").unlink()
    assert install(lock) == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert "transport" not in deployment(result)
    assert "Transport: prepared, because this cluster has no fabric document" in out.err
    assert install(lock, "--transport", "sircl") == 3
    refused = json.loads(capsys.readouterr().out)
    assert refused["field"] == "transport" and "--transport sircl cannot run here" in refused["message"]


def test_the_default_image_keeps_the_prepared_transport_and_its_deployment_identity(machine, capsys):  # noqa: F811
    assert sparkring.main(["install", "--profile", PROFILE, "--yes", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "transport" not in deployment(result)
    assert result["transport"]["backend"] == "prepared" and "carries no SIRCL layer" in result["transport"]["reason"]
    assert sparkring.main(["install", "--profile", PROFILE, "--yes", "--json", "--nccl", "auto"]) == 3
    assert json.loads(capsys.readouterr().out)["field"] == "transport"


def test_the_nccl_transport_installs_on_a_pair_without_sircl(machine, capsys):  # noqa: F811
    """An nccl deployment runs on the installer's own image: the plan, the lock and the result name the transport."""
    from runtime.common import transport
    cluster = installer.read(controller.STATE / "cluster.json")
    document, _ = fabric.prepare(cluster["plan"], cluster="test", marker=MARKER)
    (controller.STATE / "fabric.json").write_text(fabric_document.encoded(document))
    assert sparkring.main(["install", "--profile", PROFILE, "--yes", "--json", "--transport", "nccl"]) == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    recorded = deployment(result)["transport"]
    assert recorded["backend"] == "nccl" and recorded["group"]["name"] == "pair"
    assert recorded["nccl"] == {"settings": {}, "reasons": {}}
    assert recorded["fabric"]["id"] == document["id"]
    assert result["transport"]["backend"] == "nccl" and result["transport"]["expected"] == transport.NCCL_EXPECTED
    assert ("Transport: nccl on every collective; SIRCL is not loaded and the RoCEnante slot is off "
            "(--transport nccl)") in out.err
    assert "  Transport:   nccl" in out.err
    # The nccl transport takes no --nccl mode.
    assert sparkring.main(["install", "--profile", PROFILE, "--yes", "--json", "--transport", "nccl",
                           "--nccl", "auto"]) == 3
    assert json.loads(capsys.readouterr().out)["field"] == "transport"


def test_a_failed_receipt_check_is_reported_and_the_model_keeps_serving(sircl, monkeypatch, capsys):
    events, lock, _ = sircl
    failing = dict(VERDICT, verdict="differs", problems=["rank 1 group tp:0: 14 calls reached NCCL (['all_gather'])"])
    original = flow.retained_source.apply
    monkeypatch.setattr(flow.retained_source, "apply",
                        lambda path, action, **options: dict(failing) if action == "transport"
                        else original(path, action, **options))
    assert install(lock) == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["state"] == "complete" and result["transport"]["verdict"] == "differs"
    assert "  Transport:   check failed: rank 1 group tp:0: 14 calls reached NCCL" in out.err


def test_status_prints_the_last_receipt_verdict(tmp_path):
    from runtime.host import transport_receipts
    value = {"backend": "sircl", "nccl": "never", "group": "pair", "positions": [0, 1], "fabric": "sha256:" + "0" * 64}
    assert controller.transport_status_line(value).startswith("Transport: sircl on pair (NCCL never); no receipt "
                                                              "check recorded yet")
    assert controller.transport_status_line({**value, **{key: VERDICT[key] for key in (
        "verdict", "nccl_observed", "checked_at", "receipts")}, "problems": []}) == (
        "Transport: sircl, NCCL: absent (checked 2026-10-08T10:00:00Z); sudo sparkring check repeats it")
    assert controller.transport_status_line({"backend": "prepared"}) == "Transport: prepared"
    lock = {"transport": {"nccl": "never", "group": {"name": "pair", "positions": [0, 1]},
                          "fabric": {"id": value["fabric"]}}}
    (tmp_path / transport_receipts.LATEST).write_text(json.dumps(VERDICT))
    assert controller.transport_view(tmp_path, lock)["verdict"] == "as-expected"


def test_an_installation_after_fabric_tune_runs_on_the_measured_table(sircl, monkeypatch, tmp_path, capsys):
    from runtime.common import transport
    from runtime.common.test_transport import measured_table, pair_table
    from runtime.host import fabric_tune
    _, lock, document = sircl
    image = json.loads(lock.read_text())
    host = tmp_path / "node-a"
    monkeypatch.setattr(fabric_tune, "HOST_ROOT", str(host))
    facts = {"gpu": "580.95.05", "kernel": "6.11.0-1016-nvidia"}
    monkeypatch.setattr(fabric_tune, "local_facts", lambda interface=None: dict(facts))
    digest, data = pair_table(image, document)
    measured = measured_table(host, document, image, rows={"pair": {"link_slots": 12, "link_slot": 1048576}},
                              tables={digest: data})
    (controller.STATE / transport.MEASURED_TUNING).write_text(transport.encoded(measured))
    assert install(lock) == 0
    out = capsys.readouterr()
    tuning = deployment(json.loads(out.out))["transport"]["tuning"]
    assert (tuning["source"], tuning["row"], tuning["row_source"]) == ("measured", "pair", "measured")
    assert tuning["sha256"] == transport.tuning_digest(measured) and tuning["measured_at"] == "2026-10-09"
    assert tuning["tables"][0]["path"] == f"{transport.HOST_TABLES}/{digest}.json"
    assert "Transport: sircl on every collective, NCCL off (measured on this fabric 2026-10-09, pair)" in out.err
    # The measured row keeps the default pair row's settings beside the measurement's own.
    assert ("SIRCL settings: large_blocks 32, link_slot 1048576, link_slots 12, oneshot_max 131072, ring_min 2097152"
            in out.err)
    assert f"Measured row pair: {transport.MEASURED_ROW_RULE}" in out.err
    # A driver update on Node A makes the measured table stale: the next installation says so and uses the
    # default table, which is another deployment.
    facts["gpu"] = "590.10.01"
    assert install(lock) == 0
    out = capsys.readouterr()
    assert deployment(json.loads(out.out))["transport"]["tuning"]["source"] == "defaults"
    assert ("Note: the measured tuning table no longer applies: the GPU driver of position 0 changed from 580.95.05 "
            "to 590.10.01") in out.err
