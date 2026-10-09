"""``sparkring install --transport libsircl``; host and SSH boundaries simulated as in test_install_workflow.

A cluster recorded with a fabric document installs a profile with vLLM's
PyNccl on libsircl when its image lock lists the libsircl layer and the
operator names the transport; the plan says that it is research-only, the
deployment lock records the libsircl section, and the result reports that the
installer does not judge libsircl's receipts.
"""
import json
from pathlib import Path

import pytest

from runtime.common import fabric_document, installer
from runtime.common.test_image_lock import libsircl_lock, sircl_lock
from runtime.host import controller, fabric, install_workflow as flow, relays, transport_receipts
from runtime.host.test_install_sircl import VERDICT
from runtime.host.test_install_workflow import PROFILE, machine, sparks  # noqa: F401  (pytest fixtures)
from scripts import sparkring

MARKER = {"binary": relays.MARKER_BINARY, "sha256": "ab" * 32}


@pytest.fixture
def libsircl(machine, tmp_path, monkeypatch):  # noqa: F811
    """The machine's cluster with its fabric document recorded, an image lock with libsircl and the receipt check."""
    events, previous, assets, operation = machine
    cluster = installer.read(controller.STATE / "cluster.json")
    document, _ = fabric.prepare(cluster["plan"], cluster="test", marker=MARKER)
    (controller.STATE / "fabric.json").write_text(fabric_document.encoded(document))
    lock = tmp_path / "libsircl-image.json"
    lock.write_text(json.dumps(libsircl_lock()))

    def apply(path, action, **options):
        if action == "transport":
            events.append("candidate:transport")
            # The deployment's own transport_receipts.check, which runs from its retained source; a SIRCL
            # deployment's receipts pass.
            recorded = installer.read(Path(path) / "deployment.lock.json")
            if recorded["transport"]["backend"] != "libsircl":
                return dict(VERDICT)
            return transport_receipts.check(path, recorded, None)
        return operation(path, action, **options)
    monkeypatch.setattr(flow.retained_source, "apply", apply)
    return events, lock, document


def install(lock, *extra):
    return sparkring.main(["install", "--profile", PROFILE, "--yes", "--json", "--image-lock", str(lock), *extra])


def test_the_plan_names_libsircl_and_that_it_is_research_only(libsircl, capsys):
    _, lock, document = libsircl
    assert install(lock, "--transport", "libsircl", "--plan") == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["state"] == "planned"
    assert result["transport"]["backend"] == "libsircl" and result["transport"]["status"] == "research-only"
    assert result["transport"]["fabric"] == document["id"] and result["transport"]["group"] == "pair"
    assert ("Transport: libsircl (research-only): vLLM's PyNccl carries its collectives on libsircl 0.6.0; SIRCL's "
            "adapter and RoCEnante are off") in out.err
    assert "  Research-only: no serving A/B has measured libsircl" in out.err
    assert "--transport libsircl" in result["checkpoint"]["command"]


def test_an_installation_on_libsircl_records_its_section_and_does_not_judge_its_receipts(libsircl, capsys):
    events, lock, _ = libsircl
    assert install(lock, "--transport", "libsircl") == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    recorded = installer.load(result["deployment"])["transport"]
    assert recorded["backend"] == "libsircl" and recorded["group"]["name"] == "pair"
    assert [row["LIBSIRCL_POSITION"] for row in recorded["routes"]] == ["0", "1"]
    assert result["transport"]["verdict"] == "unknown" and "does not judge" in result["transport"]["problems"][0]
    assert events.index("candidate:up") < events.index("candidate:transport")
    assert "  Transport:   libsircl (research-only), receipts not judged" in out.err


def test_libsircl_sircl_and_the_prepared_transport_are_separate_deployments(libsircl, capsys):
    _, lock, _ = libsircl
    deployments = []
    for extra in ((), ("--transport", "prepared"), ("--transport", "libsircl")):
        assert install(lock, *extra) == 0
        deployments.append(json.loads(capsys.readouterr().out)["deployment"])
    assert len(set(deployments)) == 3
    assert installer.load(deployments[0])["transport"]["backend"] == "sircl"


def test_libsircl_is_refused_where_it_cannot_run_and_nothing_changes(libsircl, tmp_path, capsys):
    _, lock, _ = libsircl
    other = tmp_path / "sircl-image.json"
    other.write_text(json.dumps(sircl_lock()))
    for image, extra, message in ((other, (), "carries no libsircl layer"),
                                  (lock, ("--nccl", "auto"), "--nccl applies to deployments on SIRCL")):
        assert install(image, "--transport", "libsircl", *extra) == 3
        refused = json.loads(capsys.readouterr().out)
        assert refused["field"] == "transport" and message in refused["message"]
        assert refused["message"].endswith("Nothing has been changed.")
    (controller.STATE / "fabric.json").unlink()
    assert install(lock, "--transport", "libsircl") == 3
    assert "no fabric document" in json.loads(capsys.readouterr().out)["message"]


def test_status_names_a_libsircl_deployment():
    view = {"backend": "libsircl", "status": "research-only", "group": "path-4", "positions": [0, 1, 2, 3]}
    assert controller.transport_status_line(view) == ("Transport: libsircl on path-4 (research-only); vLLM's PyNccl "
                                                      "on libsircl, receipts not judged")
