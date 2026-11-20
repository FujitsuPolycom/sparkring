"""``sparkring install`` choosing SIRCL ring sessions; host and SSH boundaries simulated as in test_install_workflow.

A cluster recorded with a fabric document installs on SIRCL with NCCL off
when its image carries the SIRCL layer (an image lock v3); the plan prints the
transport, the deployment lock records its ``transport`` section, the result
carries the receipt verdict and the summary card its Transport line.
"""
import json
from pathlib import Path
import zipfile

import pytest

from runtime.common import fabric_document, installer, serving
from runtime.common.test_image_lock import sircl_lock
from runtime.host import colour, controller, fabric, install_workflow as flow, node, relays, topology
from runtime.host.test_appliance import nodes
from runtime.host.test_hairpin_ring import document as hairpin_document
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
    assert "Transport: sircl on every collective, NCCL off (default table, pair)" in out.err
    assert f"  Profile:         {PROFILE} · TP2 · SIRCL" in out.err
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
    # The prepared transport reads as RoCEnante (b12x); `prepared` stays its identifier in the result.
    assert "Transport: RoCEnante (b12x), because this cluster has no fabric document" in out.err
    assert f"  Profile:         {PROFILE} · TP2 · RoCEnante (b12x)" in out.err.splitlines()
    assert result["transport"]["backend"] == "prepared"
    assert install(lock, "--transport", "sircl") == 3
    refused = json.loads(capsys.readouterr().out)
    assert refused["field"] == "transport" and "--transport sircl cannot run here" in refused["message"]


def test_an_image_without_sircl_keeps_the_prepared_transport_and_its_deployment_identity(machine, capsys):  # noqa: F811
    # The default image carries SIRCL; on a cluster without a fabric document it runs the prepared transport.
    assert sparkring.main(["install", "--profile", PROFILE, "--yes", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "transport" not in deployment(result)
    assert result["transport"]["backend"] == "prepared" and "no fabric document" in result["transport"]["reason"]
    # 2026.10.1's image carries no SIRCL layer.
    assert sparkring.main(["install", "--profile", PROFILE, "--image", "2026.10.1", "--yes", "--json"]) == 0
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
    assert f"  Profile:         {PROFILE} · TP2 · NCCL" in out.err
    # The nccl transport takes no --nccl mode.
    assert sparkring.main(["install", "--profile", PROFILE, "--yes", "--json", "--transport", "nccl",
                           "--nccl", "auto"]) == 3
    assert json.loads(capsys.readouterr().out)["field"] == "transport"


def test_row_owned_prefill_is_a_locked_deepseek_sircl_experiment(machine, sparks, monkeypatch, capsys, tmp_path):  # noqa: F811
    found = nodes(4)
    for rank, direction in ((0, "ccw_"), (3, "cw_")):
        closing_link = {port["netdev"] for role, port in topology.endpoints(found[rank]).items()
                        if role.startswith(direction)}
        for interface in found[rank]["facts"]["interfaces"]:
            if interface["name"] in closing_link:
                interface["operstate"] = "DOWN"
        reports = found[rank]["lldp"]["lldp"]["interface"]
        found[rank]["lldp"]["lldp"]["interface"] = [row for row in reports if closing_link.isdisjoint(row)]
    path = topology.build_spec(found, found[0]["node_id"])
    for rank in (1, 2):
        path["nodes"][rank]["hairpin"] = hairpin_document(path, rank)
    value = {"name": "test", "plan": path}
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    fabric_record, _ = fabric.prepare(value["plan"], cluster="test", marker=MARKER)
    (controller.STATE / "fabric.json").write_text(fabric_document.encoded(fabric_record))
    sparks.mesh = lambda rank: {"reference": {"site_path": "/etc/sparkring/managed-mesh/site.json",
                                              "site_sha256": "c" * 64, "plan_sha256": "d" * 64},
                                "host_ip": f"192.0.2.{110 + rank}", "interface": "eth0", "unit": "sparkring-mesh.service"}
    request = ["install", "--profile", "deepseek-v41-flash-tp4", "--checkpoint", "1017",
               "--image", serving.ROW_OWNED_PREFILL_IMAGE, "--transport", "sircl", "--nccl", "never", "--plan", "--json"]

    assert sparkring.main(request) == 0
    plain = json.loads(capsys.readouterr().out)
    assert sparkring.main([*request, "--row-owned-prefill"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["deployment"] != plain["deployment"]
    assert plan["serving"] == {"row_owned_prefill": True}
    assert "--row-owned-prefill" in plan["checkpoint"]["command"]
    saved = deployment(plan)
    assert saved["transport"]["backend"] == "sircl" and saved["transport"]["nccl"] == "never"
    assert saved["transport"]["group"]["name"] == "path-4"
    assert saved["image_runtime"]["image_id"] == serving.ROW_OWNED_PREFILL_IMAGE_ID

    for rank, (plain_spec, spec) in enumerate(zip(installer.specifications(deployment(plain)),
                                                  installer.specifications(saved), strict=True)):
        assert spec.entrypoint == plain_spec.entrypoint
        assert "DS41_ROW_SHARD_PREFILL" not in plain_spec.environment
        assert spec.environment["DS41_ROW_SHARD_PREFILL"] == "1"
        assert "DS41_ROW_SHARD_INDEXER" not in plain_spec.environment
        assert spec.environment["DS41_ROW_SHARD_INDEXER"] == "1"
        for flag in (
            "DS41_ROW_SHARD_COMPACT_ROUTES",
            "DS41_ROW_SHARD_LOCAL_QUERIES",
            "DS41_ROW_SHARD_KV_LATENTS",
            "DS41_ROW_SHARD_COMPRESSOR_LATENTS",
        ):
            assert flag not in plain_spec.environment
            assert spec.environment[flag] == "1"
        assert spec.environment["VLLM_PLUGINS"].split(",") == [
            *plain_spec.environment["VLLM_PLUGINS"].split(","), "ds41_row_shard"]
        assert spec.environment["PYTHONPATH"].split(":") == [
            "/opt/ds41_row_shard", "/opt/sparkring/toolchain/python", "/opt/sparkring/addons/python"]
        assert spec.environment["VLLM_ENABLE_ROCE_ALLREDUCE"] == "0"
        assert spec.environment["SIRCL_NCCL"] == "never"
        assert spec.environment["SIRCL_FABRIC"] == "path:4"
        assert "sircl" in spec.environment["VLLM_PLUGINS"].split(",")
        assert spec.mounts[:2] == plain_spec.mounts[:2]
        assert [(mount.target, mount.read_only) for mount in spec.mounts[:-2]] == [
            (mount.target, mount.read_only) for mount in plain_spec.mounts]
        plugin, adapter = spec.mounts[-2:]
        assert (plugin.source, plugin.target, plugin.read_only) == (
            saved["site"]["ranks"][rank]["repository"] + "/integrations/vllm/ds41_row_shard",
            "/opt/ds41_row_shard", True)
        assert (adapter.source, adapter.target, adapter.read_only) == (
            saved["site"]["ranks"][rank]["repository"] + "/spark_transport/sircl/sparkring_sircl/vllm/adapter.py",
            "/usr/local/lib/python3.12/dist-packages/sparkring_sircl/vllm/adapter.py", True)
        assert "--enforce-eager" not in spec.command and "--disable-custom-all-reduce" in spec.command
        plain_graphs = json.loads(plain_spec.command[plain_spec.command.index("--compilation-config") + 1])
        graphs = json.loads(spec.command[spec.command.index("--compilation-config") + 1])
        assert plain_graphs["cudagraph_mode"] == "FULL_AND_PIECEWISE"
        assert graphs == {**plain_graphs, "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
        assert spec.command[spec.command.index("--max-cudagraph-capture-size") + 1] == "96"
        assert ("--headless" in spec.command) == bool(rank)
        assert serving.profile_value(spec.command, "context_length") == 262144
        batch_tokens = spec.command.index("--max-num-batched-tokens") + 1
        assert spec.command[batch_tokens] == plain_spec.command[batch_tokens] == "8192"
        engram = json.loads(spec.command[spec.command.index("--engram-config") + 1])
        assert engram == {"table_memory": "disk", "disk_resident_scales": True, "projection_tp": False}
        draft = json.loads(spec.command[spec.command.index("--speculative-config") + 1])
        assert draft["num_speculative_tokens"] == 5 and draft["enable_adaptive_verification"] is True

    assert sparkring.main([*request, "--row-owned-prefill", "--chain-all-gather"]) == 0
    chain = json.loads(capsys.readouterr().out)
    assert chain["deployment"] != plan["deployment"]
    assert chain["serving"] == {"row_owned_prefill": True, "chain_all_gather": True}
    assert "--chain-all-gather" in chain["checkpoint"]["command"]
    chain_lock = deployment(chain)
    assert chain_lock["transport"] == saved["transport"]
    for ring_spec, chain_spec in zip(installer.specifications(saved), installer.specifications(chain_lock), strict=True):
        assert ring_spec.environment["SIRCL_GATHER_SCHEDULE"] == "ring"
        assert chain_spec.environment == {
            **ring_spec.environment,
            "SIRCL_GATHER_SCHEDULE": "chain",
            "SPARKRING_STATS_DEPLOYMENT": chain_lock["id"],
        }
        assert chain_spec.command == ring_spec.command

    assert sparkring.main([*request, "--chain-all-gather"]) == 3
    refused = json.loads(capsys.readouterr().out)
    assert refused["field"] == "serving" and "requires --row-owned-prefill" in refused["message"]

    # A shareable template repeats the install command that stages the plugin.
    output = tmp_path / "template.zip"
    installer.export(chain["deployment"], output, share=True)
    with zipfile.ZipFile(output) as template:
        text = template.read("README.txt").decode()
    assert "sparkring install --profile deepseek-v41-flash-tp4 --checkpoint 1017" in text
    assert "--transport sircl --nccl never --chain-all-gather --row-owned-prefill" in text

    surveyed = len(sparks.surveys)
    for extra, complaint in ((["--checkpoint", "original"], "requires --profile"),
                             (["--image", "2026.10.2"], "requires --image"),
                             (["--nccl", "auto"], "requires --transport sircl --nccl never"),
                             (["--context-length", "262145"], "up to 262144 tokens")):
        assert sparkring.main([*request, "--row-owned-prefill", *extra]) == 3
        refused = json.loads(capsys.readouterr().out)
        assert refused["field"] == "serving" and complaint in refused["message"]
    assert len(sparks.surveys) == surveyed


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
    assert f"  Profile:         {PROFILE} · TP2 · check failed: rank 1 group tp:0: 14 calls reached NCCL" in out.err


def test_status_prints_the_last_receipt_verdict(tmp_path):
    from runtime.host import transport_receipts
    value = {"backend": "sircl", "nccl": "never", "group": "pair", "positions": [0, 1], "fabric": "sha256:" + "0" * 64}
    assert controller.transport_status_line(value).startswith("Transport: sircl on pair (NCCL never); no receipt "
                                                              "check recorded yet")
    assert colour.plain(controller.transport_status_line({**value, **{key: VERDICT[key] for key in (
        "verdict", "nccl_observed", "checked_at", "receipts")}, "problems": []})) == (
        "Transport: sircl, NCCL: absent (checked 2026-10-08T10:00:00Z); sudo sparkring check repeats it")
    assert controller.transport_status_line({"backend": "prepared"}) == "Transport: RoCEnante (b12x)"
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
    assert ("SIRCL settings: gather_link_chunk 1048576, gather_schedule ring, large_schedule ring, link_slot 1048576, "
            "link_slots 12, oneshot_max 65536, reduce_link_chunk 1048576, scatter_link_chunk 1048576, "
            "scatter_schedule ring") in out.err
    assert f"Measured row pair: {transport.MEASURED_ROW_RULE}" in out.err
    # A driver update on Node A makes the measured table stale: the next installation says so and uses the
    # default table, which is another deployment.
    facts["gpu"] = "590.10.01"
    assert install(lock) == 0
    out = capsys.readouterr()
    assert deployment(json.loads(out.out))["transport"]["tuning"]["source"] == "defaults"
    assert ("Note: the measured tuning table no longer applies: the GPU driver of position 0 changed from 580.95.05 "
            "to 590.10.01") in out.err


def test_the_dcp2_variant_plans_by_its_alias_on_a_lock_that_lists_it_with_the_tp2_dcp2_prefill_build(sircl, capsys,
                                                                                                    tmp_path):
    # The decode-context-parallel variant installs on a v3 lock that lists it and pins the TP2/DCP2 prefill vLLM
    # build, which reads its CSF checkpoint; it gives the decode-context-parallel group a SIRCL session.
    from runtime.common.test_image_lock import CSF_PINS, sircl_block
    variant = "glm53-flash-nvfp4-spark-tp2-dcp2"
    value = sircl_lock(profiles=sorted({*sircl_lock()["profiles"], variant}),
                       sircl=dict(sircl_block(), vllm_pins=["sparkring-kraken-bc9ea774-tp2-dcp2-prefill"]))
    lock = tmp_path / "dcp2-image.json"
    lock.write_text(json.dumps(value))
    command = ["install", "--profile", "glm53-flash-tp2-dcp2", "--plan", "--yes", "--json", "--image-lock", str(lock)]
    assert sparkring.main(command) == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["state"] == "planned" and result["profile"] == variant
    recorded = deployment(result)
    assert recorded["selection"]["profile"] == variant and recorded["transport"]["group"]["dcp"] == 2
    # A lock whose vLLM reads CSF but is not that build is refused, though it lists the variant.
    lock.write_text(json.dumps(dict(value, sircl=dict(sircl_block(), vllm_pins=CSF_PINS))))
    assert sparkring.main(command) != 0
    assert "source-pinned vLLM build" in capsys.readouterr().out
