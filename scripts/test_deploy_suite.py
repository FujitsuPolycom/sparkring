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
            interface["network_manager"]["ipv4_addresses"] = [port["address"]]
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


def test_existing_assets_require_explicit_paired_selection():
    roots = [f"/models/rank{i}/target" for i in range(4)]
    for reuse, selected in ((True, None), (False, roots), (True, roots[:3])):
        with pytest.raises(ValueError):
            create_spec(inventory(), "test-mesh", "/srv/sparkring/test-mesh",
                        reuse_existing_image=reuse, existing_model_roots=selected)
    result = create_spec(inventory(), "test-mesh", "/srv/sparkring/test-mesh",
                         reuse_existing_image=True, existing_model_roots=roots)
    assert result["existing_assets"] == {
        "schema": "sparkring-existing-assets/v1", "image": "all-ranks-preinstalled",
        "model_roots": roots,
    }
    assert result["site"]["model_roots"] == roots


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

def test_preserve_bootstrap_addresses_and_external_connections(tmp_path):
    from scripts.deploy_network import plan_network

    observed = inventory()
    # Bootstrap assigns .10/.11, rather than the fresh-site planner's .1/.2.
    for host in observed['hosts'].values():
        for interface in host['interfaces'][1:]:
            address = interface['ipv4'][0]
            prefix, last = address.split('/')[0].rsplit('.', 1)
            address = prefix + '.' + str(int(last) + 9) + '/24'
            interface['ipv4'] = [address]
            interface['network_manager']['ipv4_addresses'] = [address]
    before = copy.deepcopy(observed)
    spec = create_spec(observed, 'adopt', '/srv/sparkring/adopt', preserve_existing_network=True)
    plan = plan_network(spec, observed['hosts'])
    for host, network_host in zip(spec['hosts'], plan['hosts'], strict=True):
        facts = observed['hosts'][host['host']]
        by_name = {i['name']: i for i in facts['interfaces']}
        for port, record in zip(host['data_interfaces'], network_host['interfaces'], strict=True):
            assert port['address'] == by_name[port['netdev']]['ipv4'][0]
            assert 'replace_connection_uuid' not in port
            assert record['action'] == 'none'
            assert record['previous_connection_uuid'] == by_name[port['netdev']]['network_manager']['connection_uuid']
        assert network_host['apply'] == []
    assert observed == before
    source = tmp_path / 'inventory.json'
    source.write_text(json.dumps(observed))
    output = tmp_path / 'preparation.json'
    assert main(['plan', '--inventory', str(source), '--name', 'adopt', '--workspace', '/srv/sparkring/adopt', '--preserve-existing-network', '--output', str(output)]) == 0
    assert json.loads(output.read_text())['spec'] == spec


@pytest.mark.parametrize('addresses', [[], ['198.18.0.10/32'], ['198.18.0.10/24', '198.18.0.12/24']])
def test_preserve_network_rejects_incomplete_endpoint(addresses):
    observed = inventory()
    next(iter(observed['hosts'].values()))['interfaces'][1]['ipv4'] = addresses
    with pytest.raises(ValueError, match='exactly one|observed IPv4 /24'):
        create_spec(observed, 'adopt', '/srv/sparkring/adopt', preserve_existing_network=True)


def test_preserve_network_rejects_saved_state_replacement():
    observed = inventory()
    interface = next(iter(observed['hosts'].values()))['interfaces'][1]
    interface['network_manager']['ipv4_addresses'] = ['198.18.250.1/24']
    with pytest.raises(ValueError, match='unowned connection'):
        create_spec(observed, 'adopt', '/srv/sparkring/adopt', preserve_existing_network=True)


def test_preserve_network_rejects_broken_cable_cycle():
    observed = inventory()
    interface = next(iter(observed['hosts'].values()))['interfaces'][1]
    interface['ipv4'] = ['198.18.250.1/24']
    interface['network_manager']['ipv4_addresses'] = interface['ipv4'][:]
    with pytest.raises(ValueError):
        create_spec(observed, 'adopt', '/srv/sparkring/adopt', preserve_existing_network=True)
