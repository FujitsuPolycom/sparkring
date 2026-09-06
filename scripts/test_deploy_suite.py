import copy
import json

import pytest

from scripts.deploy_suite import (
    create_spec,
    discover,
    lifecycle_capabilities,
    main,
    write_new,
)
from scripts.test_deploy_inventory import host_inventory
from scripts.test_deploy_network import network_fixture


def inventory():
    _, network = network_fixture()
    hosts = {}
    for name, row in network.items():
        observed = host_inventory()
        management = copy.deepcopy(observed["interfaces"][0])
        management["name"] = row["management"]["interface"]
        management["ipv4"] = [row["management"]["address"] + "/24"]
        observed.update(row)
        observed["interfaces"] = [management, *row["interfaces"]]
        observed["management"]["controller_address"] = "192.0.2.100"
        hosts[name] = observed
    return {
        "schema": "sparkring-deploy-inventory/v1",
        "controller_address": "192.0.2.100",
        "hosts": hosts,
    }


def configured_inventory(spec):
    result = inventory()
    for host in spec["hosts"]:
        facts = result["hosts"][host["host"]]
        for port in host["data_interfaces"]:
            interface = next(
                i for i in facts["interfaces"] if i["name"] == port["netdev"]
            )
            function = next(
                i for i in facts["rdma"] if i["device"] == port["rdma_device"]
            )
            interface["ipv4"] = [port["address"]]
            function["gid"] = "::ffff:" + port["address"].split("/")[0]
    return result


def test_create_four_rank_spec_without_contacting_hosts():
    observed = inventory()
    before = copy.deepcopy(observed)
    spec = create_spec(observed, "test-mesh", "/srv/sparkring/test-mesh")
    assert observed == before
    assert len(spec["hosts"]) == 4
    assert (
        len({p["address"] for h in spec["hosts"] for p in h["data_interfaces"]}) == 16
    )
    assert spec["site"]["model_roots"][0].endswith(
        "df116c4fb16b1d37ae43d2cfd624de26ffbc832e"
    )
    assert spec["site"]["marker_binary_sha256"]
    assert all(len(h["data_interfaces"]) == 4 for h in spec["hosts"])


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["hosts"]["spark-r0"]["platform"].update(architecture="x86_64"),
        lambda d: d["hosts"]["spark-r0"]["management"].update(interface="enp1s0f0np0"),
    ],
)
def test_unsafe_inventory_rejected(change):
    observed = inventory()
    change(observed)
    with pytest.raises(ValueError):
        create_spec(observed, "test", "/srv/sparkring/test")


def test_discovery_uses_read_only_probe_and_exact_hosts():
    facts = inventory()["hosts"]
    calls = []

    def run(host, argv, timeout):
        calls.append((host, argv))
        return {"returncode": 0, "stdout": json.dumps(facts[host]), "stderr": ""}

    nodes = [f"{host}={f['management']['address']}" for host, f in facts.items()]
    result = discover(nodes, "192.0.2.100", run)
    assert len(calls) == 4
    assert all(argv[:2] == ["python3", "-c"] for _, argv in calls)
    assert result["hosts"].keys() == facts.keys()


def test_no_overwrite_and_help_are_offline(tmp_path):
    p = tmp_path / "inventory.json"
    write_new(p, {"saved": True})
    with pytest.raises(FileExistsError):
        write_new(p, {"overwrite": True})
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0


def test_memory_capabilities_match_source_contract(tmp_path):
    for name in ("managed_service.py", "managed_cluster.py"):
        (tmp_path / name).write_text("operations=[]\n")
    assert lifecycle_capabilities(tmp_path) == []
    (tmp_path / "managed_memory.py").write_text("# implemented helper\n")
    with pytest.raises(ValueError, match="Incomplete"):
        lifecycle_capabilities(tmp_path)
    for name in ("managed_service.py", "managed_cluster.py"):
        (tmp_path / name).write_text(
            "operations=['memory-idle','memory-prepare','memory-check']\n"
        )
    assert lifecycle_capabilities(tmp_path) == [
        "memory-check",
        "memory-idle",
        "memory-prepare",
    ]
