"""Host, boot and container identity must remain bound to observation time."""
import copy
import json
import uuid

import pytest

from runtime.common import installer
from runtime.common.test_installer import QWEN, site
from runtime.host import controller, node, topology
from runtime.host.test_appliance import nodes
from scripts import installer_host


def identity_files(root, *, node_number=1, boot_number=10):
    node_id, boot_id = str(uuid.UUID(int=node_number)), str(uuid.UUID(int=boot_number))
    node.save(root, "/etc/sparkring/node.json", {"node_id": node_id})
    boot = root / "proc/sys/kernel/random/boot_id"
    boot.parent.mkdir(parents=True, exist_ok=True)
    boot.write_text(boot_id + "\n")
    return node_id, boot_id


def test_host_identity_does_not_depend_on_network_configuration(tmp_path):
    expected_node, expected_boot = identity_files(tmp_path)
    result = node.snapshot(root=tmp_path, now=lambda: 20)
    assert result["node_id"] == expected_node and result["boot_id"] == expected_boot
    assert result["observed_at"] == 20 and result["source"] == "host-agent"
    assert result["state"] == "not-configured" and result["hardware_qualified"] is False


def test_reboot_does_not_relabel_a_cached_observation(tmp_path):
    node_id, boot_id = identity_files(tmp_path)
    recorded = node.snapshot(root=tmp_path, now=lambda: 20)
    node.save(tmp_path, "/run/sparkring/status.json", recorded)
    _, next_boot = identity_files(tmp_path, boot_number=11)
    cached = node.status(root=tmp_path, now=lambda: 120)
    assert cached["state"] == "stale"
    assert cached["observed_at"] == 20 and cached["boot_id"] == boot_id
    fresh = node.snapshot(root=tmp_path, now=lambda: 120)
    assert fresh["boot_id"] == next_boot and fresh["node_id"] == node_id


def test_missing_or_invalid_identity_is_unknown_without_repair(tmp_path):
    empty = node.observation_identity(root=tmp_path)
    assert empty["node_id"] is None and empty["boot_id"] is None
    assert set(empty["identity_errors"]) == {"node_id", "boot_id"}
    assert list(tmp_path.iterdir()) == []
    identity_files(tmp_path)
    node.save(tmp_path, "/etc/sparkring/node.json", {"node_id": "not-a-uuid"})
    result = node.observation_identity(root=tmp_path)
    assert result["node_id"] is None and result["boot_id"] is not None
    assert node.read(tmp_path, "/etc/sparkring/node.json")["node_id"] == "not-a-uuid"


def test_container_replacement_and_restart_have_distinct_observation_identities(tmp_path):
    actual_node, boot_id = identity_files(tmp_path)
    lock = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    row = {**lock["site"]["ranks"][0], "node_id": actual_node}
    info = {"Id": "first-container", "Image": lock["selection"]["image_id"], "Name": "/model",
            "State": {"Running": True, "StartedAt": "2026-09-24T18:00:00Z", "Health": {"Status": "healthy"}},
            "Config": {"Env": ["PRIVATE_API_TOKEN=must-not-be-exported"]}}
    first = installer_host.model_observation(lock, row, info, root=tmp_path, now=lambda: 100)
    changed = copy.deepcopy(info)
    changed["Id"] = "replacement-container"
    changed["State"]["StartedAt"] = "2026-09-24T19:00:00Z"
    second = installer_host.model_observation(lock, row, changed, root=tmp_path, now=lambda: 200)
    assert first["deployment_id"] == second["deployment_id"] == lock["id"]
    assert first["node_id"] == actual_node and first["boot_id"] == boot_id
    assert first["node_identity_matches"] is True
    assert first["container_id"] != second["container_id"]
    assert first["container_started_at"] != second["container_started_at"]
    assert (first["observed_at"], second["observed_at"]) == (100, 200)
    assert "PRIVATE_API_TOKEN" not in json.dumps(first)
    changed["State"]["StartedAt"] = "2026-09-24T20:00:00Z"
    restarted = installer_host.model_observation(lock, row, changed, root=tmp_path, now=lambda: 300)
    assert restarted["container_id"] == second["container_id"]
    assert restarted["container_started_at"] != second["container_started_at"]


def test_absent_container_does_not_report_desired_image_as_observed(tmp_path):
    identity_files(tmp_path)
    lock = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    result = installer_host.model_observation(lock, lock["site"]["ranks"][0], None, root=tmp_path)
    assert result["container_id"] is None and result["image_id"] is None
    assert result["expected_image_id"] == lock["selection"]["image_id"]
    assert result["node_identity_matches"] is None
    assert result["present"] is False and result["health"] is None


def test_wrong_node_is_reported_as_a_mismatch(tmp_path):
    identity_files(tmp_path, node_number=2)
    lock = installer.make_lock(QWEN, site(), "1" * 40, "2" * 64)
    row = {**lock["site"]["ranks"][0], "node_id": str(uuid.UUID(int=1))}
    result = installer_host.model_observation(lock, row, None, root=tmp_path)
    assert result["node_identity_matches"] is False


def test_discovered_node_ids_survive_controller_and_deployment_lock():
    found = nodes(2)
    cluster = {"name": "home", "plan": topology.build_spec(found, found[0]["node_id"])}
    raw = controller.model_site(cluster, QWEN)
    lock = installer.make_lock(QWEN, raw, "1" * 40, "2" * 64)
    assert [r["node_id"] for r in lock["site"]["ranks"]] == [n["node_id"] for n in found]
    assert len(installer.specifications(lock)) == 2


@pytest.mark.parametrize("change", ["duplicate", "partial", "invalid"])
def test_node_identity_bindings_cannot_alias_or_be_partial(change):
    raw = site()
    for index, row in enumerate(raw["hosts"]):
        row["node_id"] = str(uuid.UUID(int=index + 1))
    if change == "duplicate":
        raw["hosts"][1]["node_id"] = raw["hosts"][0]["node_id"]
    elif change == "partial":
        raw["hosts"][1].pop("node_id")
    else:
        raw["hosts"][0]["node_id"] = None
    with pytest.raises(ValueError):
        installer.make_lock(QWEN, raw, "1" * 40, "2" * 64)
