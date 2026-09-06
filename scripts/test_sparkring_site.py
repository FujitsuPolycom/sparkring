#!/usr/bin/env python3
"""Tests for the SparkRing site loader and validator.

Entirely offline: no GPU, no cluster, no network.  The shipped example config
is the happy-path fixture, and every malformed case is a small mutation of it
so the table cannot drift away from the real schema.

Run with::

    python -m pytest scripts/test_sparkring_site.py -q
"""

from __future__ import annotations

import copy
import ipaddress
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sparkring_site  # noqa: E402
from sparkring_site import (  # noqa: E402
    Edge,
    SiteConfigError,
    _require_single_cycle,
    ipv4_mapped_gid,
    is_documentation_address,
    load_site,
    parse_site_yaml,
    validate_site,
)

EXAMPLE_PATH = (
    Path(__file__).resolve().parent / "config" / "exl3-r7-site.example.yaml"
)
GLM53_EXAMPLE_PATH = (
    Path(__file__).resolve().parent
    / "config"
    / "glm53-flash-tp4-site.example.yaml"
)


@pytest.fixture(scope="session")
def example_document() -> dict:
    return yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def document(example_document: dict) -> dict:
    result = copy.deepcopy(example_document)
    result["artifacts"] = [
        {
            "name": "transport-library",
            "path": "/opt/sparkring/libspark_transport_capi.so",
            "sha256": "1" * 64,
            "executable": False,
        },
        {
            "name": "transport-probe",
            "path": "/opt/sparkring/spark_tp4_probe",
            "sha256": "2" * 64,
            "executable": True,
        },
    ]
    return result


def six_ring_document(base: dict) -> dict:
    """Return a complete six-rank cycle derived from the shipped four-ring."""
    result = copy.deepcopy(base)
    size = 6
    result["site"] = {
        "name": "six-ring-test",
        "description": "Offline six-rank SparkRing validation fixture.",
    }
    result["topology"]["edges"] = [
        {
            "id": f"r{rank}-r{(rank + 1) % size}",
            "subnet": f"10.20.{rank}.0/24",
            "endpoints": [rank, (rank + 1) % size],
        }
        for rank in range(size)
    ]
    ranks = []
    for rank in range(size):
        previous = (rank - 1) % size
        following = (rank + 1) % size
        ranks.append(
            {
                "id": rank,
                "ssh_target": f"operator@192.0.2.{rank + 10}",
                "management": {
                    "interface": "eth0",
                    "address": f"192.0.2.{rank + 10}",
                },
                "ring_ports": [
                    {
                        "edge": f"r{previous}-r{rank}",
                        "interface": "eth1",
                        "address": f"10.20.{previous}.{rank + 100}",
                        "rdma_device": "mlx5_0",
                        "rdma_port": 1,
                        "roce_gid_index": 3,
                    },
                    {
                        "edge": f"r{rank}-r{following}",
                        "interface": "eth2",
                        "address": f"10.20.{rank}.{rank + 10}",
                        "rdma_device": "mlx5_1",
                        "rdma_port": 1,
                        "roce_gid_index": 3,
                    },
                ],
                "transport_peers": [
                    {"rank": previous, "address": f"192.0.2.{previous + 10}"},
                    {"rank": following, "address": f"192.0.2.{following + 10}"},
                ],
            }
        )
    result["ranks"] = ranks
    result["serving"]["tensor_parallel_size"] = size
    result["serving"]["decode_context_parallel_size"] = 1
    result["serving"]["master_rank"] = 0
    return result


# ==========================================================================
# Happy path
# ==========================================================================


def test_example_file_exists():
    assert EXAMPLE_PATH.is_file()


def test_example_loads_and_validates():
    site = load_site(EXAMPLE_PATH)
    assert site.schema_version == sparkring_site.SCHEMA_VERSION
    assert site.name
    assert site.source == str(EXAMPLE_PATH)


def test_digest_pinned_image_may_have_no_loose_host_artifacts(document):
    document["artifacts"] = []
    site = parse_site_yaml(yaml.safe_dump(document), source="<faststart>")
    assert site.artifacts == ()


def test_example_has_four_ranks_ids_zero_to_three():
    site = load_site(EXAMPLE_PATH)
    assert [rank.id for rank in site.ranks] == [0, 1, 2, 3]


def test_example_has_four_edges_each_with_two_ring_ports():
    site = load_site(EXAMPLE_PATH)
    assert len(site.topology.edges) == 4
    claims: dict[str, list[int]] = {edge.id: [] for edge in site.topology.edges}
    for rank in site.ranks:
        assert len(rank.ring_ports) == 2
        for port in rank.ring_ports:
            claims[port.edge].append(rank.id)
    for edge in site.topology.edges:
        assert sorted(claims[edge.id]) == sorted(edge.endpoints)


def test_six_rank_ring_loads_and_resolves_every_neighbour(document):
    site = validate_site(six_ring_document(document))

    assert site.topology.rank_ids == (0, 1, 2, 3, 4, 5)
    assert len(site.topology.edges) == 6
    assert len(site.ranks) == 6
    for rank in site.ranks:
        assert len(rank.ring_ports) == 2
        assert len(rank.neighbour_ranks) == 2
        assert {peer.rank for peer in rank.transport_peers} == set(
            rank.neighbour_ranks
        )
        for port in rank.ring_ports:
            assert port.peer_address is not None


def test_five_rank_ring_is_rejected_as_unsupported(document):
    candidate = six_ring_document(document)
    candidate["topology"]["edges"] = candidate["topology"]["edges"][:5]

    with pytest.raises(SiteConfigError, match="exactly 4 or 6 edges"):
        validate_site(candidate)


def test_peer_addresses_are_resolved_for_every_ring_port():
    site = load_site(EXAMPLE_PATH)
    for rank in site.ranks:
        for port in rank.ring_ports:
            assert port.peer_rank in (0, 1, 2, 3)
            assert port.peer_rank != rank.id
            peer_port = next(
                other for other in site.rank(port.peer_rank).ring_ports
                if other.edge == port.edge
            )
            assert port.peer_address == peer_port.address
            assert peer_port.peer_address == port.address


def test_every_edge_uses_a_distinct_slash24():
    site = load_site(EXAMPLE_PATH)
    subnets = [str(edge.subnet) for edge in site.topology.edges]
    assert len(set(subnets)) == 4
    for edge in site.topology.edges:
        assert edge.subnet.prefixlen == 24


def test_ring_addresses_live_inside_their_edge_subnet():
    site = load_site(EXAMPLE_PATH)
    for rank in site.ranks:
        for port in rank.ring_ports:
            edge = site.topology.by_id(port.edge)
            assert port.address in edge.subnet


def test_jumbo_payload_is_mtu_minus_ip_and_icmp_headers():
    site = load_site(EXAMPLE_PATH)
    assert site.topology.jumbo_payload_bytes == site.topology.mtu - 28


def test_remote_space_targets_cover_jit_and_context_cache():
    site = load_site(EXAMPLE_PATH)
    labels = [label for label, _path, _minimum in
              site.paths.remote_space_targets()]
    assert labels == ["jit_cache", "context_cache"]


def test_to_dict_is_json_serialisable():
    site = load_site(EXAMPLE_PATH)
    payload = json.loads(json.dumps(site.to_dict()))
    assert payload["schema"] == sparkring_site.SCHEMA_ID
    assert len(payload["ranks"]) == 4
    assert len(payload["topology"]["edges"]) == 4


def test_summary_lines_mention_every_rank_and_edge():
    site = load_site(EXAMPLE_PATH)
    text = "\n".join(site.summary_lines())
    for rank in site.ranks:
        assert rank.ssh_target in text
    for edge in site.topology.edges:
        assert edge.id in text


def test_no_private_addresses_in_shipped_example():
    """The public template must never carry a real site's addressing."""
    raw = EXAMPLE_PATH.read_text(encoding="utf-8")
    assert "192.168." not in raw
    site = load_site(EXAMPLE_PATH)
    for rank in site.ranks:
        assert is_documentation_address(rank.management.address)
        for port in rank.ring_ports:
            assert is_documentation_address(port.address)


def test_example_placeholder_warnings_are_reported():
    site = load_site(EXAMPLE_PATH)
    warnings = site.placeholder_warnings()
    assert warnings, "the shipped example is all placeholders"
    joined = "\n".join(warnings)
    assert "documentation/benchmark address" in joined
    assert "repeated-digit placeholder" in joined


# ==========================================================================
# IPv4-mapped GID derivation
# ==========================================================================


@pytest.mark.parametrize(
    "address,expected",
    [
        ("192.0.2.10", "0000:0000:0000:0000:0000:ffff:c000:020a"),
        ("198.51.100.11", "0000:0000:0000:0000:0000:ffff:c633:640b"),
        ("203.0.113.13", "0000:0000:0000:0000:0000:ffff:cb00:710d"),
        ("198.18.0.13", "0000:0000:0000:0000:0000:ffff:c612:000d"),
        ("0.0.0.0", "0000:0000:0000:0000:0000:ffff:0000:0000"),
        ("255.255.255.255", "0000:0000:0000:0000:0000:ffff:ffff:ffff"),
        ("10.1.2.3", "0000:0000:0000:0000:0000:ffff:0a01:0203"),
    ],
)
def test_ipv4_mapped_gid_known_vectors(address, expected):
    assert ipv4_mapped_gid(address) == expected


def test_ipv4_mapped_gid_accepts_address_objects():
    assert ipv4_mapped_gid(ipaddress.IPv4Address("192.0.2.10")) == (
        ipv4_mapped_gid("192.0.2.10")
    )


def test_ipv4_mapped_gid_round_trips_the_hex_back_to_the_address():
    for address in ("192.0.2.10", "198.51.100.12", "198.18.0.13"):
        gid = ipv4_mapped_gid(address)
        tail = gid.split(":")[-2] + gid.split(":")[-1]
        octets = [int(tail[index:index + 2], 16) for index in range(0, 8, 2)]
        assert ".".join(str(value) for value in octets) == address


def test_documentation_address_detection():
    assert is_documentation_address("192.0.2.1")
    assert is_documentation_address("198.51.100.1")
    assert is_documentation_address("203.0.113.1")
    assert is_documentation_address("198.18.5.7")
    assert not is_documentation_address("10.0.0.1")
    assert not is_documentation_address("not-an-address")


# ==========================================================================
# Ring topology validator
# ==========================================================================


def _edge(edge_id: str, subnet: str, a: int, b: int) -> Edge:
    return Edge(
        id=edge_id, subnet=ipaddress.IPv4Network(subnet), endpoints=(a, b)
    )


def test_single_cycle_accepts_a_ring():
    _require_single_cycle([
        _edge("a", "192.0.2.0/24", 0, 1),
        _edge("b", "198.51.100.0/24", 1, 2),
        _edge("c", "203.0.113.0/24", 2, 3),
        _edge("d", "198.18.0.0/24", 3, 0),
    ])


def test_single_cycle_accepts_a_differently_labelled_ring():
    _require_single_cycle([
        _edge("a", "192.0.2.0/24", 0, 2),
        _edge("b", "198.51.100.0/24", 2, 1),
        _edge("c", "203.0.113.0/24", 1, 3),
        _edge("d", "198.18.0.0/24", 3, 0),
    ])


def test_single_cycle_rejects_two_disjoint_two_cycles():
    with pytest.raises(SiteConfigError) as excinfo:
        _require_single_cycle([
            _edge("a", "192.0.2.0/24", 0, 1),
            _edge("b", "198.51.100.0/24", 0, 1),
            _edge("c", "203.0.113.0/24", 2, 3),
            _edge("d", "198.18.0.0/24", 2, 3),
        ])
    assert excinfo.value.field == "topology.edges"
    assert "single closed ring" in str(excinfo.value)


def test_single_cycle_rejects_a_star():
    with pytest.raises(SiteConfigError) as excinfo:
        _require_single_cycle([
            _edge("a", "192.0.2.0/24", 0, 1),
            _edge("b", "198.51.100.0/24", 0, 2),
            _edge("c", "203.0.113.0/24", 0, 3),
            _edge("d", "198.18.0.0/24", 1, 2),
        ])
    assert excinfo.value.field == "topology.edges"
    assert "must be on exactly 2" in str(excinfo.value)


def test_single_cycle_rejects_an_edge_set_that_leaves_a_rank_out():
    """Ranks 0-2 form a triangle plus a spare cable; rank 3 is never cabled."""
    with pytest.raises(SiteConfigError) as excinfo:
        _require_single_cycle([
            _edge("a", "192.0.2.0/24", 0, 1),
            _edge("b", "198.51.100.0/24", 1, 2),
            _edge("c", "203.0.113.0/24", 2, 0),
            _edge("d", "198.18.0.0/24", 0, 1),
        ])
    assert excinfo.value.field == "topology.edges"
    # The degree accounting names the first rank that is not on exactly two
    # edges; with rank 3 absent some other rank must carry three.
    assert "is on 3 edge(s)" in str(excinfo.value)
    assert "must be on exactly 2" in str(excinfo.value)


def test_single_cycle_rejects_a_rank_with_only_one_edge():
    with pytest.raises(SiteConfigError) as excinfo:
        _require_single_cycle([
            _edge("a", "192.0.2.0/24", 0, 1),
            _edge("b", "198.51.100.0/24", 1, 2),
            _edge("c", "203.0.113.0/24", 2, 3),
            _edge("d", "198.18.0.0/24", 1, 3),
        ])
    assert excinfo.value.field == "topology.edges"
    assert "rank 0 is on 1 edge(s)" in str(excinfo.value)


# ==========================================================================
# Malformed configuration table
# ==========================================================================


def _drop(container, key):
    del container[key]


CASES: list[tuple[str, object, str, str]] = [
    (
        "schema-version-wrong",
        lambda d: d.__setitem__("schema_version", 2),
        "schema_version", "unsupported schema version 2",
    ),
    (
        "schema-version-not-int",
        lambda d: d.__setitem__("schema_version", "1"),
        "schema_version", "expected integer 1",
    ),
    (
        "root-unknown-key",
        lambda d: d.__setitem__("extra_section", {}),
        "<root>", "unknown key(s): extra_section",
    ),
    (
        "root-missing-section",
        lambda d: _drop(d, "serving"),
        "<root>", "missing required key(s): serving",
    ),
    (
        "root-not-a-mapping",
        lambda d: None,
        "<root>", "expected a mapping",
    ),
    (
        "site-missing-description",
        lambda d: _drop(d["site"], "description"),
        "site", "missing required key(s): description",
    ),
    (
        "site-name-empty",
        lambda d: d["site"].__setitem__("name", "   "),
        "site.name", "must not be empty",
    ),
    # --- topology ---------------------------------------------------------
    (
        "topology-mtu-out-of-range",
        lambda d: d["topology"].__setitem__("mtu", 100),
        "topology.mtu", "out of range",
    ),
    (
        "topology-mtu-not-int",
        lambda d: d["topology"].__setitem__("mtu", "9000"),
        "topology.mtu", "expected an integer",
    ),
    (
        "topology-mtu-bool",
        lambda d: d["topology"].__setitem__("mtu", True),
        "topology.mtu", "expected an integer",
    ),
    (
        "topology-speed-out-of-range",
        lambda d: d["topology"].__setitem__("link_speed_mbps", 10),
        "topology.link_speed_mbps", "out of range",
    ),
    (
        "topology-wrong-edge-count",
        lambda d: d["topology"].__setitem__(
            "edges", d["topology"]["edges"][:3]
        ),
        "topology.edges", "exactly 4 or 6 edges",
    ),
    (
        "topology-duplicate-edge-id",
        lambda d: d["topology"]["edges"][1].__setitem__("id", "r0-r1"),
        "topology.edges[1].id", "duplicate edge id",
    ),
    (
        "topology-duplicate-subnet",
        lambda d: d["topology"]["edges"][1].__setitem__(
            "subnet", "192.0.2.0/24"
        ),
        "topology.edges[1].subnet", "every ring edge needs its own distinct",
    ),
    (
        "topology-subnet-not-slash24",
        lambda d: d["topology"]["edges"][0].__setitem__(
            "subnet", "192.0.2.0/25"
        ),
        "topology.edges[0].subnet", "must be /24",
    ),
    (
        "topology-subnet-has-host-bits",
        lambda d: d["topology"]["edges"][0].__setitem__(
            "subnet", "192.0.2.10/24"
        ),
        "topology.edges[0].subnet", "not a valid IPv4 network",
    ),
    (
        "topology-self-loop",
        lambda d: d["topology"]["edges"][0].__setitem__("endpoints", [0, 0]),
        "topology.edges[0].endpoints", "self-loop",
    ),
    (
        "topology-parallel-edges",
        lambda d: d["topology"]["edges"][2].__setitem__("endpoints", [0, 1]),
        "topology.edges[2].endpoints", "already joined by edge",
    ),
    (
        "topology-endpoint-out-of-range",
        lambda d: d["topology"]["edges"][0].__setitem__("endpoints", [0, 9]),
        "topology.edges[0].endpoints[1]", "not one of [0, 1, 2, 3]",
    ),
    (
        "topology-endpoint-count",
        lambda d: d["topology"]["edges"][0].__setitem__("endpoints", [0]),
        "topology.edges[0].endpoints", "joins exactly 2 ranks",
    ),
    (
        "topology-not-a-ring-degree",
        lambda d: d["topology"]["edges"][0].__setitem__("endpoints", [0, 2]),
        "topology.edges", "must be on exactly 2",
    ),
    # --- ranks ------------------------------------------------------------
    (
        "ranks-not-a-list",
        lambda d: d.__setitem__("ranks", {"0": {}}),
        "ranks", "expected a list",
    ),
    (
        "ranks-wrong-count",
        lambda d: d.__setitem__("ranks", d["ranks"][:3]),
        "ranks", "expected exactly 4 ranks",
    ),
    (
        "ranks-duplicate-id",
        lambda d: d["ranks"][1].__setitem__("id", 0),
        "ranks[1].id", "duplicate rank id 0",
    ),
    (
        "ranks-id-out-of-range",
        lambda d: d["ranks"][1].__setitem__("id", 7),
        "ranks[1].id", "not one of [0, 1, 2, 3]",
    ),
    (
        "ranks-missing-ssh-target",
        lambda d: _drop(d["ranks"][0], "ssh_target"),
        "ranks[0]", "missing required key(s): ssh_target",
    ),
    (
        "ranks-ssh-target-empty",
        lambda d: d["ranks"][0].__setitem__("ssh_target", ""),
        "ranks[0].ssh_target", "must not be empty",
    ),
    (
        "ranks-ssh-target-no-user",
        lambda d: d["ranks"][0].__setitem__("ssh_target", "198.18.1.10"),
        "ranks[0].ssh_target", "invalid ssh target",
    ),
    (
        "ranks-ssh-target-duplicated",
        lambda d: d["ranks"][1].__setitem__(
            "ssh_target", d["ranks"][0]["ssh_target"]
        ),
        "ranks[1].ssh_target", "already used by rank 0",
    ),
    (
        "ranks-management-interface-empty",
        lambda d: d["ranks"][0]["management"].__setitem__("interface", ""),
        "ranks[0].management.interface", "must not be empty",
    ),
    (
        "ranks-management-address-invalid",
        lambda d: d["ranks"][0]["management"].__setitem__(
            "address", "198.18.1.999"
        ),
        "ranks[0].management.address", "not a valid IPv4 address",
    ),
    (
        "ranks-management-address-loopback",
        lambda d: d["ranks"][0]["management"].__setitem__(
            "address", "127.0.0.1"
        ),
        "ranks[0].management.address", "loopback/multicast/unspecified",
    ),
    (
        "ranks-management-inside-ring-subnet",
        lambda d: d["ranks"][0]["management"].__setitem__(
            "address", "192.0.2.200"
        ),
        "ranks[0].management.address", "must be separate from every ring",
    ),
    (
        "ranks-management-collides-with-ring-address",
        lambda d: d["ranks"][3]["management"].__setitem__(
            "address", "192.0.2.10"
        ),
        "ranks[3].management.address", "already used by",
    ),
    (
        "ranks-wrong-ring-port-count",
        lambda d: d["ranks"][0].__setitem__(
            "ring_ports", d["ranks"][0]["ring_ports"][:1]
        ),
        "ranks[0].ring_ports", "exactly 2 ring ports",
    ),
    (
        "ranks-ring-port-unknown-edge",
        lambda d: d["ranks"][0]["ring_ports"][0].__setitem__("edge", "nope"),
        "ranks[0].ring_ports[0].edge", "unknown edge",
    ),
    (
        "ranks-ring-port-address-outside-subnet",
        lambda d: d["ranks"][0]["ring_ports"][0].__setitem__(
            "address", "203.0.113.99"
        ),
        "ranks[0].ring_ports[0].address", "is not inside edge",
    ),
    (
        "ranks-ring-port-network-address",
        lambda d: d["ranks"][0]["ring_ports"][0].__setitem__(
            "address", "192.0.2.0"
        ),
        "ranks[0].ring_ports[0].address", "is the network address",
    ),
    (
        "ranks-ring-port-broadcast-address",
        lambda d: d["ranks"][0]["ring_ports"][0].__setitem__(
            "address", "192.0.2.255"
        ),
        "ranks[0].ring_ports[0].address", "is the broadcast address",
    ),
    (
        "ranks-ring-port-both-ends-same-address",
        lambda d: d["ranks"][1]["ring_ports"][0].__setitem__(
            "address", "192.0.2.10"
        ),
        "ranks[1].ring_ports[0].address", "both ends of edge",
    ),
    (
        "ranks-ring-port-shared-interface",
        lambda d: d["ranks"][0]["ring_ports"][1].__setitem__(
            "interface", d["ranks"][0]["ring_ports"][0]["interface"]
        ),
        "ranks[0].ring_ports", "both ports use interface",
    ),
    (
        "ranks-ring-port-uses-management-interface",
        lambda d: d["ranks"][0]["ring_ports"][0].__setitem__(
            "interface", d["ranks"][0]["management"]["interface"]
        ),
        "ranks[0].ring_ports[0].interface", "also the management interface",
    ),
    (
        "ranks-ring-port-shared-rdma-device",
        lambda d: d["ranks"][0]["ring_ports"][1].__setitem__(
            "rdma_device", d["ranks"][0]["ring_ports"][0]["rdma_device"]
        ),
        "ranks[0].ring_ports", "both ports map to RDMA",
    ),
    (
        "ranks-ring-port-interface-bad-characters",
        lambda d: d["ranks"][0]["ring_ports"][0].__setitem__(
            "interface", "eth1; rm -rf /"
        ),
        "ranks[0].ring_ports[0].interface", "invalid interface name",
    ),
    (
        "ranks-ring-port-gid-index-out-of-range",
        lambda d: d["ranks"][0]["ring_ports"][0].__setitem__(
            "roce_gid_index", 999
        ),
        "ranks[0].ring_ports[0].roce_gid_index", "out of range",
    ),
    (
        "ranks-ring-port-rdma-port-out-of-range",
        lambda d: d["ranks"][0]["ring_ports"][0].__setitem__("rdma_port", 0),
        "ranks[0].ring_ports[0].rdma_port", "out of range",
    ),
    (
        "ranks-both-ports-same-edge",
        lambda d: d["ranks"][0]["ring_ports"][1].__setitem__("edge", "r0-r1"),
        "ranks[0].ring_ports", "both ports claim edge",
    ),
    (
        "edge-claimed-three-times",
        lambda d: d["ranks"][2]["ring_ports"][0].__setitem__("edge", "r0-r1"),
        "topology.edges[r0-r1]", "claimed by 3 ring port(s)",
    ),
    (
        "edge-claimed-by-wrong-ranks",
        lambda d: (
            d["ranks"][0]["ring_ports"].__setitem__(
                0, dict(d["ranks"][0]["ring_ports"][0], edge="r1-r2")
            ),
            d["ranks"][2]["ring_ports"].__setitem__(
                0, dict(d["ranks"][2]["ring_ports"][0], edge="r0-r1")
            ),
        ),
        "topology.edges[r0-r1]", "declares endpoints [0, 1] but is claimed by",
    ),
    # --- transport peers --------------------------------------------------
    (
        "peers-wrong-count",
        lambda d: d["ranks"][0].__setitem__(
            "transport_peers", d["ranks"][0]["transport_peers"][:1]
        ),
        "ranks[0].transport_peers", "exactly 2 control-channel peers",
    ),
    (
        "peers-self-reference",
        lambda d: d["ranks"][0]["transport_peers"][0].__setitem__("rank", 0),
        "ranks[0].transport_peers[0].rank",
        "cannot be its own control-channel peer",
    ),
    (
        "peers-not-ring-neighbours",
        lambda d: d["ranks"][0]["transport_peers"][1].__setitem__("rank", 2),
        "ranks[0].transport_peers", "ring neighbours are",
    ),
    (
        "peers-address-belongs-to-another-rank",
        lambda d: d["ranks"][0]["transport_peers"][0].__setitem__(
            "address", "198.18.1.12"
        ),
        "ranks[0].transport_peers[0].address",
        "is the management address of rank 2",
    ),
    (
        "peers-address-is-own-management",
        lambda d: d["ranks"][0]["transport_peers"][0].__setitem__(
            "address", "198.18.1.10"
        ),
        "ranks[0].transport_peers[0].address",
        "this rank's own management address",
    ),
    (
        "peers-duplicate-rank",
        lambda d: d["ranks"][0]["transport_peers"][1].__setitem__("rank", 1),
        "ranks[0].transport_peers", "duplicate peer rank 1",
    ),
    # --- runtime ----------------------------------------------------------
    (
        "runtime-digest-missing-prefix",
        lambda d: d["runtime"].__setitem__(
            "container_image_digest", "1" * 64
        ),
        "runtime.container_image_digest", "must be 'sha256:'",
    ),
    (
        "runtime-digest-uppercase",
        lambda d: d["runtime"].__setitem__(
            "container_image_digest", "sha256:" + "A" * 64
        ),
        "runtime.container_image_digest", "64 lowercase hex",
    ),
    (
        "runtime-revision-is-a-branch",
        lambda d: d["runtime"].__setitem__("model_revision", "main"),
        "runtime.model_revision", "immutable 40-character",
    ),
    (
        "runtime-checkpoint-hash-too-short",
        lambda d: d["runtime"].__setitem__("checkpoint_sha256", "abc123"),
        "runtime.checkpoint_sha256", "exactly 64 lowercase hex",
    ),
    (
        "runtime-checkpoint-hash-uppercase",
        lambda d: d["runtime"].__setitem__("checkpoint_sha256", "A" * 64),
        "runtime.checkpoint_sha256", "exactly 64 lowercase hex",
    ),
    (
        "runtime-model-path-relative",
        lambda d: d["runtime"].__setitem__("model_path", "models/thing"),
        "runtime.model_path", "invalid absolute path",
    ),
    # --- serving ----------------------------------------------------------
    (
        "serving-dcp-exceeds-tp",
        lambda d: d["serving"].__setitem__(
            "decode_context_parallel_size", 8
        ),
        "serving.decode_context_parallel_size", "exceeds tensor_parallel_size",
    ),
    (
        "serving-dcp-does-not-divide-tp",
        lambda d: (
            d["serving"].__setitem__("tensor_parallel_size", 4),
            d["serving"].__setitem__("decode_context_parallel_size", 3),
        ),
        "serving.decode_context_parallel_size", "not divisible",
    ),
    (
        "serving-unknown-mtp-mode",
        lambda d: d["serving"].__setitem__("mtp_mode", "turbo"),
        "serving.mtp_mode", "is not one of",
    ),
    (
        "serving-mtp-off-with-tokens",
        lambda d: d["serving"].__setitem__("mtp_mode", "off"),
        "serving.mtp_tokens", "must be 0 when mtp_mode is 'off'",
    ),
    (
        "serving-mtp-adaptive-without-tokens",
        lambda d: d["serving"].__setitem__("mtp_tokens", 0),
        "serving.mtp_tokens", "must be at least 1",
    ),
    (
        "serving-max-model-len-out-of-range",
        lambda d: d["serving"].__setitem__("max_model_len", 4),
        "serving.max_model_len", "out of range",
    ),
    (
        "serving-kv-bytes-out-of-range",
        lambda d: d["serving"].__setitem__("kv_cache_bytes_per_rank", 1024),
        "serving.kv_cache_bytes_per_rank", "out of range",
    ),
    (
        "serving-max-num-seqs-zero",
        lambda d: d["serving"].__setitem__("max_num_seqs", 0),
        "serving.max_num_seqs", "out of range",
    ),
    (
        "serving-ports-collide",
        lambda d: d["serving"].__setitem__("master_port", 8000),
        "serving.master_port", "must differ from serving.api_port",
    ),
    (
        "serving-api-port-out-of-range",
        lambda d: d["serving"].__setitem__("api_port", 70000),
        "serving.api_port", "out of range",
    ),
    (
        "serving-master-rank-unknown",
        lambda d: d["serving"].__setitem__("master_rank", 3),
        None, None,  # valid: rank 3 exists - see dedicated test below
    ),
    # --- paths ------------------------------------------------------------
    (
        "paths-jit-and-context-identical",
        lambda d: d["paths"].__setitem__(
            "context_cache_dir", d["paths"]["jit_cache_dir"]
        ),
        "paths.context_cache_dir", "must differ from paths.jit_cache_dir",
    ),
    (
        "paths-relative-remote-dir",
        lambda d: d["paths"].__setitem__("jit_cache_dir", "var/lib/cache"),
        "paths.jit_cache_dir", "invalid absolute path",
    ),
    (
        "paths-remote-dir-with-quote",
        lambda d: d["paths"].__setitem__("jit_cache_dir", "/var/li'b"),
        "paths.jit_cache_dir", "invalid absolute path",
    ),
    (
        "paths-min-free-missing-key",
        lambda d: _drop(d["paths"]["min_free_bytes"], "context_cache"),
        "paths.min_free_bytes", "missing required key(s): context_cache",
    ),
    (
        "paths-min-free-negative",
        lambda d: d["paths"]["min_free_bytes"].__setitem__("jit_cache", -1),
        "paths.min_free_bytes.jit_cache", "out of range",
    ),
    # --- artifacts --------------------------------------------------------
    (
        "artifacts-duplicate-path",
        lambda d: d["artifacts"][1].__setitem__(
            "path", d["artifacts"][0]["path"]
        ),
        "artifacts[1].path", "duplicate artifact path",
    ),
    (
        "artifacts-duplicate-name",
        lambda d: d["artifacts"][1].__setitem__(
            "name", d["artifacts"][0]["name"]
        ),
        "artifacts[1].name", "duplicate artifact name",
    ),
    (
        "artifacts-bad-hex",
        lambda d: d["artifacts"][0].__setitem__("sha256", "g" * 64),
        "artifacts[0].sha256", "exactly 64 lowercase hex",
    ),
    (
        "artifacts-executable-not-bool",
        lambda d: d["artifacts"][0].__setitem__("executable", "yes"),
        "artifacts[0].executable", "expected true or false",
    ),
    (
        "artifacts-path-with-shell-metacharacter",
        lambda d: d["artifacts"][0].__setitem__("path", "/opt/x $(whoami)"),
        "artifacts[0].path", "invalid absolute path",
    ),
    (
        "artifacts-unknown-key",
        lambda d: d["artifacts"][0].__setitem__("optional", True),
        "artifacts[0]", "unknown key(s): optional",
    ),
    # --- preflight --------------------------------------------------------
    (
        "preflight-timeout-out-of-range",
        lambda d: d["preflight"].__setitem__("ssh_timeout_seconds", 0),
        "preflight.ssh_timeout_seconds", "out of range",
    ),
    (
        "preflight-port-out-of-range",
        lambda d: d["preflight"].__setitem__(
            "required_free_ports", [0]
        ),
        "preflight.required_free_ports[0]", "out of range",
    ),
    (
        "preflight-duplicate-port",
        lambda d: d["preflight"].__setitem__(
            "required_free_ports", [8000, 8000]
        ),
        "preflight.required_free_ports[1]", "duplicate port 8000",
    ),
    (
        "preflight-port-not-int",
        lambda d: d["preflight"].__setitem__(
            "required_free_ports", ["8000"]
        ),
        "preflight.required_free_ports[0]", "expected a port number",
    ),
]

_FAILING_CASES = [case for case in CASES if case[2] is not None]


@pytest.mark.parametrize(
    "case_id,mutate,expected_field,expected_message",
    _FAILING_CASES,
    ids=[case[0] for case in _FAILING_CASES],
)
def test_malformed_configuration_is_rejected(
    document, case_id, mutate, expected_field, expected_message
):
    mutated = document
    result = mutate(mutated)
    if result is None and case_id == "root-not-a-mapping":
        mutated = None
    with pytest.raises(SiteConfigError) as excinfo:
        validate_site(mutated)
    assert excinfo.value.field == expected_field, (
        f"{case_id}: expected field {expected_field!r}, "
        f"got {excinfo.value.field!r} ({excinfo.value})"
    )
    assert expected_message in str(excinfo.value), (
        f"{case_id}: {excinfo.value}"
    )


def test_master_rank_three_is_accepted(document):
    document["serving"]["master_rank"] = 3
    site = validate_site(document)
    assert site.serving.master_rank == 3


def test_mtp_off_with_zero_tokens_is_accepted(document):
    document["serving"]["mtp_mode"] = "off"
    document["serving"]["mtp_tokens"] = 0
    site = validate_site(document)
    assert site.serving.mtp_mode == "off"


def test_dcp_of_one_is_accepted(document):
    document["serving"]["decode_context_parallel_size"] = 1
    site = validate_site(document)
    assert site.serving.decode_context_parallel_size == 1


def test_ranks_may_be_listed_out_of_order(document):
    document["ranks"] = list(reversed(document["ranks"]))
    site = validate_site(document)
    assert [rank.id for rank in site.ranks] == [0, 1, 2, 3]


def test_empty_required_free_ports_is_accepted(document):
    document["preflight"]["required_free_ports"] = []
    site = validate_site(document)
    assert site.preflight.required_free_ports == ()


def test_memory_launch_headroom_is_optional_and_validated(document):
    assert validate_site(document).preflight.memory is None

    document["preflight"]["memory"] = {
        "minimum_available_bytes": 103079215104,
        "contiguous_block_bytes": 33554432,
        "minimum_contiguous_blocks": 200,
    }
    memory = validate_site(document).preflight.memory

    assert memory is not None
    assert memory.minimum_available_bytes == 103079215104
    assert memory.contiguous_block_bytes == 33554432
    assert memory.minimum_contiguous_blocks == 200


def test_glm53_site_enables_memory_launch_headroom():
    site = load_site(GLM53_EXAMPLE_PATH)
    memory = site.preflight.memory

    assert memory is not None
    assert site.serving.decode_context_parallel_size == 4
    assert site.runtime.container_image.endswith(
        "@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f"
    )
    assert memory.minimum_available_bytes == 96 * (1 << 30)
    assert memory.contiguous_block_bytes == 32 * (1 << 20)
    assert memory.minimum_contiguous_blocks == 200

    document = yaml.safe_load(GLM53_EXAMPLE_PATH.read_text(encoding="utf-8"))
    document["preflight"] = json.loads(
        json.dumps(site.to_dict()["preflight"])
    )
    round_tripped = validate_site(document)
    assert round_tripped.preflight.memory == memory


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("minimum_available_bytes", 0, "out of range"),
        ("contiguous_block_bytes", 12_000_000, "power of two"),
        ("minimum_contiguous_blocks", 0, "out of range"),
    ),
)
def test_invalid_memory_launch_headroom_is_rejected(
    document, field, value, message
):
    document["preflight"]["memory"] = {
        "minimum_available_bytes": 103079215104,
        "contiguous_block_bytes": 16777216,
        "minimum_contiguous_blocks": 64,
    }
    document["preflight"]["memory"][field] = value

    with pytest.raises(SiteConfigError) as excinfo:
        validate_site(document)

    assert excinfo.value.field == f"preflight.memory.{field}"
    assert message in str(excinfo.value)


# ==========================================================================
# File / parse level behaviour
# ==========================================================================


def test_load_site_missing_file_names_the_path(tmp_path):
    target = tmp_path / "absent.yaml"
    with pytest.raises(SiteConfigError) as excinfo:
        load_site(target)
    assert excinfo.value.field == str(target)
    assert "not found" in str(excinfo.value)


def test_parse_site_yaml_rejects_broken_yaml():
    with pytest.raises(SiteConfigError) as excinfo:
        parse_site_yaml("ranks: [\n  - id: 0\n", source="broken.yaml")
    assert excinfo.value.field == "broken.yaml"
    assert "could not parse YAML" in str(excinfo.value)


def test_parse_site_yaml_rejects_empty_document():
    with pytest.raises(SiteConfigError) as excinfo:
        parse_site_yaml("# only a comment\n", source="empty.yaml")
    assert "file is empty" in str(excinfo.value)


def test_missing_pyyaml_gives_an_actionable_error(monkeypatch):
    monkeypatch.setattr(sparkring_site, "_yaml", None)
    monkeypatch.setattr(
        sparkring_site, "_YAML_IMPORT_ERROR", ImportError("no module")
    )
    with pytest.raises(SiteConfigError) as excinfo:
        parse_site_yaml("schema_version: 1\n")
    assert excinfo.value.field == "<pyyaml>"
    assert "pip install PyYAML" in str(excinfo.value)


# ==========================================================================
# CLI
# ==========================================================================


def test_cli_accepts_the_example(capsys):
    assert sparkring_site.main([str(EXAMPLE_PATH)]) == 0
    captured = capsys.readouterr()
    assert "valid SparkRing site configuration" in captured.out


def test_cli_json_output_is_parseable(capsys):
    assert sparkring_site.main([str(EXAMPLE_PATH), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == sparkring_site.SCHEMA_ID
    assert payload["placeholder_warnings"]


def test_cli_strict_placeholders_fails_on_the_example(capsys):
    assert sparkring_site.main(
        [str(EXAMPLE_PATH), "--strict-placeholders"]
    ) == 1
    assert "placeholder" in capsys.readouterr().err


def test_cli_rejects_a_broken_file(tmp_path, capsys):
    target = tmp_path / "site.yaml"
    target.write_text("schema_version: 9\n", encoding="utf-8")
    assert sparkring_site.main([str(target)]) == 1
    assert "INVALID" in capsys.readouterr().err
