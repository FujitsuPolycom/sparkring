"""Offline unit tests for the SparkRing fabric doctor."""

from __future__ import annotations

import ipaddress
import unittest

from scripts.ring_doctor import (
    FABRIC_INTERFACES,
    CommandResult,
    InterfaceState,
    NodeObservation,
    NodeSpec,
    build_plans,
    build_probe_command,
    diagnose,
    diagnose_launch_endpoints,
    discover_nodes,
    docker_user_accepts,
    infer_adjacency,
    infer_topology,
    parse_all_addresses,
    parse_brief_addresses,
    parse_route_get,
    select_route,
    validate_cycle,
)

IF0, IF1 = FABRIC_INTERFACES
RENDEZVOUS = ipaddress.ip_address("10.0.1.10")


def observation(
    name: str,
    addresses: str,
    *,
    hostname: str | None = None,
    reachable: bool = True,
    forward_policy: str | None = "DROP",
    docker_rules: tuple[str, ...] | None = (),
    socket_interfaces: tuple[str, ...] = (),
    host_addresses: str | None = None,
    route_get: str | None = None,
) -> NodeObservation:
    interfaces = parse_brief_addresses(addresses)
    interfaces = {
        interface_name: InterfaceState(
            state.name, state.addresses, state.oper_state, 9000
        )
        for interface_name, state in interfaces.items()
    }
    return NodeObservation(
        NodeSpec(name, f"operator@{name}", None, socket_interfaces),
        reachable,
        hostname=hostname,
        interfaces=interfaces,
        ip_forward=True if reachable else None,
        forward_policy=forward_policy if reachable else None,
        docker_user_rules=docker_rules if reachable else None,
        error="SSH exited 255: connection timed out" if not reachable else "",
        host_interfaces=parse_all_addresses(
            host_addresses if host_addresses is not None else addresses
        ),
        rendezvous_route=(
            parse_route_get(RENDEZVOUS, route_get) if route_get is not None else None
        ),
    )


def synthetic_cycle() -> dict[str, NodeObservation]:
    return {
        "a": observation(
            "a",
            f"{IF0} UP 10.0.1.1/24\n{IF1} UP 10.0.4.1/24",
        ),
        "b": observation(
            "b",
            f"{IF0} UP 10.0.2.1/24\n{IF1} UP 10.0.1.2/24",
        ),
        "c": observation(
            "c",
            f"{IF0} UP 10.0.3.1/24\n{IF1} UP 10.0.2.2/24",
        ),
        "d": observation(
            "d",
            f"{IF0} UP 10.0.4.2/24\n{IF1} UP 10.0.3.2/24",
        ),
    }


def probe_output(
    hostname: str,
    addresses: str,
    route_get: str | None = None,
    routes: str = "",
    docker_rules: str = "-N DOCKER-USER",
) -> str:
    rendezvous = (
        f"__RING_DOCTOR_RENDEZVOUS__\n{route_get}\n" if route_get is not None else ""
    )
    return f"""__RING_DOCTOR_HOSTNAME__
{hostname}
__RING_DOCTOR_ADDR__
{addresses}
__RING_DOCTOR_LINK__
2: {IF0}: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 state UP mode DEFAULT
3: {IF1}: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 state UP mode DEFAULT
__RING_DOCTOR_ROUTE__
{routes}
__RING_DOCTOR_FORWARD__
1
__RING_DOCTOR_FORWARD_CHAIN__
-P FORWARD DROP
__RING_DOCTOR_DOCKER_USER__
{docker_rules}
{rendezvous}__RING_DOCTOR_INTERFACES__
{IF0}
{IF1}
"""


class FakeDiscoveryRunner:
    def __init__(self, outputs: dict[str, str], down: set[str]) -> None:
        self.outputs = outputs
        self.down = down
        self.calls: list[tuple[str, str | None]] = []

    def run(
        self, target: str, command: str, proxy_jump: str | None = None
    ) -> CommandResult:
        del command
        self.calls.append((target, proxy_jump))
        if target in self.down:
            return CommandResult(False, detail="SSH exited 255: connection timed out")
        return CommandResult(True, stdout=self.outputs[target])


class AddressAndTopologyTests(unittest.TestCase):
    def test_adjacency_inference_from_real_brief_address_fixture(self) -> None:
        observations = {
            "r0": observation(
                "r0",
                f"{IF0} UP 10.0.1.10/24\n"
                f"{IF1} UP 10.0.4.12/24",
                hostname="spark-edfd",
            ),
            "r1": observation(
                "r1",
                f"{IF0} UP 10.0.2.10/24\n"
                f"{IF1} UP 10.0.1.11/24",
                hostname="spark-ebb8",
            ),
            "r2": observation(
                "r2",
                f"{IF0} UP 10.0.3.11/24\n"
                f"{IF1} UP 10.0.2.11/24",
                hostname="spark-ebee",
            ),
            "r3": observation("r3", "", reachable=False),
        }

        adjacency, subnet_nodes = infer_adjacency(observations)

        self.assertEqual(adjacency["r0"], {"r1"})
        self.assertEqual(adjacency["r1"], {"r0", "r2"})
        self.assertEqual(adjacency["r2"], {"r1"})
        self.assertNotIn("r3", adjacency)
        self.assertEqual(
            subnet_nodes[ipaddress.ip_network("10.0.1.0/24")],
            ("r0", "r1"),
        )
        self.assertEqual(
            subnet_nodes[ipaddress.ip_network("10.0.4.0/24")], ("r0",)
        )

    def test_cycle_validation_accepts_cycle_and_rejects_path(self) -> None:
        cycle_graph = {
            "r0": {"r1", "r3"},
            "r1": {"r0", "r2"},
            "r2": {"r1", "r3"},
            "r3": {"r0", "r2"},
        }
        valid, cycle, reason = validate_cycle(cycle_graph)
        self.assertTrue(valid, reason)
        self.assertEqual(set(cycle), set(cycle_graph))

        path_graph = {
            "r0": {"r1"},
            "r1": {"r0", "r2"},
            "r2": {"r1"},
        }
        valid, cycle, reason = validate_cycle(path_graph)
        self.assertFalse(valid)
        self.assertEqual(cycle, ())
        self.assertIn("two neighbors", reason)


class FirewallAndPlanningTests(unittest.TestCase):
    def test_same_interface_accepts_do_not_cover_cross_interface_relay(self) -> None:
        observations = synthetic_cycle()
        same_interface_rules = (
            f"-A DOCKER-USER -i {IF0} -o {IF0} -j ACCEPT",
            f"-A DOCKER-USER -i {IF1} -o {IF1} -j ACCEPT",
        )
        for item in observations.values():
            item.docker_user_rules = same_interface_rules
        topology = infer_topology(observations)
        plans = build_plans(observations, topology)

        findings = diagnose(
            [item.spec for item in observations.values()],
            observations,
            topology,
            plans,
        )

        self.assertFalse(docker_user_accepts(same_interface_rules, IF0, IF1))
        cross_findings = [
            finding
            for finding in findings
            if finding.code == "docker-user-cross-accept-missing"
        ]
        self.assertTrue(cross_findings)
        self.assertTrue(
            any(IF0 in finding.message and IF1 in finding.message for finding in cross_findings)
        )

    def test_next_hops_are_addresses_observed_on_direct_neighbors(self) -> None:
        observations = synthetic_cycle()
        topology = infer_topology(observations)
        destination = ipaddress.ip_network("10.0.2.0/24")

        route = select_route("a", destination, observations, topology)

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.gateway, ipaddress.ip_address("10.0.1.2"))
        self.assertEqual(route.path[:2], ("a", "b"))
        plans = build_plans(observations, topology)
        for source, plan in plans.items():
            neighbor_addresses = {
                address.ip
                for neighbor in topology.adjacency[source]
                for interface in observations[neighbor].interfaces.values()
                for address in interface.addresses
                if any(
                    address.ip in own_address.network
                    for own_interface in observations[source].interfaces.values()
                    for own_address in own_interface.addresses
                )
            }
            for proposed in plan.routes:
                self.assertIn(proposed.gateway, neighbor_addresses)


class PartialDiscoveryTests(unittest.TestCase):
    def test_one_unreachable_node_preserves_observations_and_unknown_checks(self) -> None:
        specs = (
            NodeSpec("r0", "operator@r0"),
            NodeSpec("r1", "operator@r1"),
            NodeSpec("r2", "operator@r2"),
            NodeSpec("r3", "operator@r3"),
        )
        outputs = {
            "operator@r0": probe_output(
                "spark-edfd",
                f"{IF0} UP 10.0.1.10/24\n"
                f"{IF1} UP 10.0.4.12/24",
            ),
            "operator@r1": probe_output(
                "spark-ebb8",
                f"{IF0} UP 10.0.2.10/24\n"
                f"{IF1} UP 10.0.1.11/24",
            ),
            "operator@r2": probe_output(
                "spark-ebee",
                f"{IF0} UP 10.0.3.11/24\n"
                f"{IF1} UP 10.0.2.11/24",
            ),
        }
        runner = FakeDiscoveryRunner(outputs, {"operator@r3"})

        observations = discover_nodes(specs, runner)
        topology = infer_topology(observations)
        plans = build_plans(observations, topology)
        findings = diagnose(specs, observations, topology, plans)

        self.assertFalse(observations["r3"].reachable)
        self.assertEqual(observations["r3"].interfaces, {})
        self.assertIn("attempted paths", observations["r3"].error)
        self.assertIn(("operator@r3", "operator@r0"), runner.calls)
        self.assertFalse(topology.valid_cycle)
        self.assertEqual(plans["r0"].routes, [])
        self.assertIn("node-unreachable", {finding.code for finding in findings})
        self.assertIn("topology-not-cycle", {finding.code for finding in findings})
        self.assertIn(
            "subnet-endpoints-unknown", {finding.code for finding in findings}
        )


WIFI = "wlP9s9"

ADDRESSED_WIFI_LISTING = f"""lo UNKNOWN 127.0.0.1/8
{IF0} UP 10.0.1.10/24
{IF1} UP 10.0.4.12/24
{WIFI} UP 192.0.2.21/24"""

FABRIC_ONLY_LISTING = f"""lo UNKNOWN 127.0.0.1/8
{IF0} UP 10.0.2.10/24
{IF1} UP 10.0.1.11/24
tailscale0 UNKNOWN 100.100.5.5/32"""

UNADDRESSED_WIFI_LISTING = f"""lo UNKNOWN 127.0.0.1/8
{IF0} UP 10.0.3.11/24
{IF1} UP 10.0.2.11/24
{WIFI} DOWN"""

LOCAL_ROUTE = f"local {RENDEZVOUS} dev lo src {RENDEZVOUS} uid 1000\n    cache <local>"
FABRIC_ROUTE = f"{RENDEZVOUS} dev {IF1} src 10.0.1.11 uid 1000\n    cache"
UNREACHABLE_ROUTE = "RTNETLINK answers: Network is unreachable"


class RouteLookupParsingTests(unittest.TestCase):
    def test_route_lookup_reports_selected_device_and_source(self) -> None:
        route = parse_route_get(RENDEZVOUS, FABRIC_ROUTE)

        self.assertTrue(route.routable)
        self.assertEqual(route.device, IF1)
        self.assertEqual(route.source, ipaddress.ip_address("10.0.1.11"))
        self.assertIsNone(route.gateway)

    def test_route_lookup_reports_a_gateway_when_the_kernel_selects_one(self) -> None:
        route = parse_route_get(
            RENDEZVOUS,
            f"{RENDEZVOUS} via 10.0.4.2 dev {IF0} src 10.0.4.1 uid 1000\n    cache",
        )

        self.assertTrue(route.routable)
        self.assertEqual(route.gateway, ipaddress.ip_address("10.0.4.2"))

    def test_unreachable_network_is_not_routable_and_keeps_kernel_wording(
        self,
    ) -> None:
        route = parse_route_get(RENDEZVOUS, UNREACHABLE_ROUTE)

        self.assertFalse(route.routable)
        self.assertIsNone(route.device)
        self.assertIn("Network is unreachable", route.raw)

    def test_denying_route_type_and_empty_output_are_not_routable(self) -> None:
        denied = parse_route_get(
            RENDEZVOUS, f"unreachable {RENDEZVOUS} dev lo src 10.0.1.11 uid 1000"
        )
        empty = parse_route_get(RENDEZVOUS, "")

        self.assertFalse(denied.routable)
        self.assertFalse(empty.routable)
        self.assertIn("no output", empty.raw)

    def test_route_lookup_joins_the_existing_probe_only_when_addressed(self) -> None:
        without_address = build_probe_command(FABRIC_INTERFACES)
        with_address = build_probe_command(FABRIC_INTERFACES, RENDEZVOUS)

        self.assertNotIn("route get", without_address)
        self.assertIn(f"ip -4 route get {RENDEZVOUS} 2>&1", with_address)
        self.assertEqual(with_address.count("route get"), 1)


class LaunchEndpointTests(unittest.TestCase):
    def test_routable_address_and_present_interfaces_report_affirmatively(
        self,
    ) -> None:
        observations = {
            "r0": observation(
                "r0",
                ADDRESSED_WIFI_LISTING,
                socket_interfaces=(WIFI,),
                route_get=LOCAL_ROUTE,
            ),
            "r1": observation(
                "r1",
                FABRIC_ONLY_LISTING,
                socket_interfaces=(IF1,),
                route_get=FABRIC_ROUTE,
            ),
        }

        findings = diagnose_launch_endpoints(
            [item.spec for item in observations.values()], observations, RENDEZVOUS
        )

        self.assertEqual([finding.code for finding in findings], ["launch-endpoints-observed"])
        self.assertEqual(findings[0].severity, "info")
        self.assertIn(str(RENDEZVOUS), findings[0].message)
        self.assertIn("r0", findings[0].evidence)
        self.assertIn("r1", findings[0].evidence)

    def test_check_stays_silent_when_no_launch_endpoint_is_named(self) -> None:
        observations = {"r1": observation("r1", FABRIC_ONLY_LISTING)}

        self.assertEqual(
            diagnose_launch_endpoints([observations["r1"].spec], observations), []
        )

    def test_node_without_a_route_to_the_rendezvous_address_is_reported(self) -> None:
        observations = {
            "r0": observation(
                "r0", ADDRESSED_WIFI_LISTING, route_get=LOCAL_ROUTE
            ),
            "r2": observation(
                "r2", UNADDRESSED_WIFI_LISTING, route_get=UNREACHABLE_ROUTE
            ),
        }

        findings = diagnose_launch_endpoints(
            [item.spec for item in observations.values()], observations, RENDEZVOUS
        )

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.code, "rendezvous-unroutable")
        self.assertEqual(finding.severity, "error")
        self.assertEqual(finding.node, "r2")
        self.assertIn(str(RENDEZVOUS), finding.message)
        self.assertIn("Network is unreachable", finding.evidence)

    def test_named_socket_interface_absent_from_a_node_is_reported(self) -> None:
        observations = {
            "r1": observation(
                "r1",
                FABRIC_ONLY_LISTING,
                socket_interfaces=(WIFI,),
                route_get=FABRIC_ROUTE,
            )
        }

        findings = diagnose_launch_endpoints(
            [observations["r1"].spec], observations, RENDEZVOUS
        )

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.code, "socket-interface-absent")
        self.assertEqual(finding.severity, "error")
        self.assertEqual(finding.node, "r1")
        self.assertIn(WIFI, finding.message)
        self.assertIn("tailscale0", finding.evidence)
        self.assertNotIn(WIFI, finding.evidence)

    def test_named_socket_interface_without_an_address_is_reported(self) -> None:
        observations = {
            "r2": observation(
                "r2",
                UNADDRESSED_WIFI_LISTING,
                socket_interfaces=(WIFI,),
                route_get=LOCAL_ROUTE,
            )
        }

        findings = diagnose_launch_endpoints(
            [observations["r2"].spec], observations, RENDEZVOUS
        )

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.code, "socket-interface-unaddressed")
        self.assertEqual(finding.severity, "error")
        self.assertEqual(finding.node, "r2")
        self.assertIn(WIFI, finding.message)
        self.assertIn("state=DOWN", finding.evidence)

    def test_unread_route_section_is_reported_as_unknown_not_as_a_fault(self) -> None:
        observations = {"r1": observation("r1", FABRIC_ONLY_LISTING)}

        findings = diagnose_launch_endpoints(
            [observations["r1"].spec], observations, RENDEZVOUS
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "rendezvous-route-unobserved")
        self.assertEqual(findings[0].severity, "warning")

    def test_one_probe_round_carries_route_and_interface_evidence(self) -> None:
        specs = (
            NodeSpec("r0", "operator@r0", None, (WIFI,)),
            NodeSpec("r1", "operator@r1", None, (WIFI,)),
            NodeSpec("r2", "operator@r2", None, (WIFI,)),
            NodeSpec("r3", "operator@r3", None, (WIFI,)),
        )
        outputs = {
            "operator@r0": probe_output(
                "spark-edfd", ADDRESSED_WIFI_LISTING, LOCAL_ROUTE
            ),
            "operator@r1": probe_output(
                "spark-ebb8",
                ADDRESSED_WIFI_LISTING.replace("10.0.1.10/24", "10.0.2.10/24"),
                FABRIC_ROUTE,
            ),
            "operator@r2": probe_output(
                "spark-ebee", UNADDRESSED_WIFI_LISTING, FABRIC_ROUTE
            ),
            "operator@r3": probe_output(
                "spark-e1a4", FABRIC_ONLY_LISTING, UNREACHABLE_ROUTE
            ),
        }
        runner = FakeDiscoveryRunner(outputs, set())

        observations = discover_nodes(
            specs, runner, FABRIC_INTERFACES, RENDEZVOUS
        )
        topology = infer_topology(observations)
        plans = build_plans(observations, topology)
        findings = diagnose(
            specs,
            observations,
            topology,
            plans,
            FABRIC_INTERFACES,
            RENDEZVOUS,
        )

        self.assertEqual(len(runner.calls), len(specs))
        self.assertEqual(
            observations["r0"].host_interfaces[WIFI].addresses[0],
            ipaddress.ip_interface("192.0.2.21/24"),
        )
        self.assertFalse(observations["r3"].rendezvous_route.routable)
        launch_findings = {
            (finding.code, finding.node)
            for finding in findings
            if finding.code.startswith(("rendezvous-", "socket-interface-"))
        }
        self.assertEqual(
            launch_findings,
            {
                ("rendezvous-unroutable", "r3"),
                ("socket-interface-unaddressed", "r2"),
                ("socket-interface-absent", "r3"),
            },
        )
        self.assertNotIn(
            "launch-endpoints-observed", {finding.code for finding in findings}
        )


CROSS_ACCEPT_RULES = (
    f"-N DOCKER-USER\n"
    f"-A DOCKER-USER -i {IF0} -o {IF1} -j ACCEPT\n"
    f"-A DOCKER-USER -i {IF1} -o {IF0} -j ACCEPT"
)

HEALTHY_RING = {
    "r0": (
        ("spark-edfd", (WIFI, IF0)),
        f"lo UNKNOWN 127.0.0.1/8\n{IF0} UP 10.0.1.10/24\n{IF1} UP 10.0.4.12/24\n"
        f"{WIFI} UP 192.0.2.21/24",
        f"10.0.2.0/24 via 10.0.1.11 dev {IF0}\n10.0.3.0/24 via 10.0.4.13 dev {IF1}",
        f"local {RENDEZVOUS} dev lo src {RENDEZVOUS} uid 1000",
    ),
    "r1": (
        ("spark-ebb8", (IF0,)),
        f"lo UNKNOWN 127.0.0.1/8\n{IF0} UP 10.0.2.10/24\n{IF1} UP 10.0.1.11/24",
        f"10.0.3.0/24 via 10.0.2.11 dev {IF0}\n10.0.4.0/24 via 10.0.1.10 dev {IF1}",
        f"{RENDEZVOUS} dev {IF1} src 10.0.1.11 uid 1000\n    cache",
    ),
    "r2": (
        ("spark-ebee", (IF0,)),
        f"lo UNKNOWN 127.0.0.1/8\n{IF0} UP 10.0.3.11/24\n{IF1} UP 10.0.2.11/24",
        f"10.0.1.0/24 via 10.0.2.10 dev {IF1}\n10.0.4.0/24 via 10.0.3.12 dev {IF0}",
        f"{RENDEZVOUS} via 10.0.2.10 dev {IF1} src 10.0.2.11 uid 1000\n    cache",
    ),
    "r3": (
        ("spark-e1a4", (IF0,)),
        f"lo UNKNOWN 127.0.0.1/8\n{IF0} UP 10.0.4.13/24\n{IF1} UP 10.0.3.12/24",
        f"10.0.1.0/24 via 10.0.4.12 dev {IF0}\n10.0.2.0/24 via 10.0.3.11 dev {IF1}",
        f"{RENDEZVOUS} via 10.0.4.12 dev {IF0} src 10.0.4.13 uid 1000\n    cache",
    ),
}


class HealthyRingLaunchEndpointTests(unittest.TestCase):
    def test_intact_ring_with_reachable_endpoints_reports_only_the_pass(
        self,
    ) -> None:
        specs = tuple(
            NodeSpec(name, f"operator@{name}", None, entry[0][1])
            for name, entry in HEALTHY_RING.items()
        )
        outputs = {
            f"operator@{name}": probe_output(
                entry[0][0],
                entry[1],
                entry[3],
                routes=entry[2],
                docker_rules=CROSS_ACCEPT_RULES,
            )
            for name, entry in HEALTHY_RING.items()
        }
        runner = FakeDiscoveryRunner(outputs, set())

        observations = discover_nodes(specs, runner, FABRIC_INTERFACES, RENDEZVOUS)
        topology = infer_topology(observations)
        plans = build_plans(observations, topology)
        findings = diagnose(
            specs, observations, topology, plans, FABRIC_INTERFACES, RENDEZVOUS
        )

        self.assertTrue(topology.valid_cycle, topology.reason)
        self.assertEqual(
            [(finding.severity, finding.code) for finding in findings],
            [("info", "launch-endpoints-observed")],
        )
        self.assertIn(str(RENDEZVOUS), findings[0].message)
        for name in HEALTHY_RING:
            self.assertIn(name, findings[0].evidence)


if __name__ == "__main__":
    unittest.main()
