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
    diagnose,
    discover_nodes,
    docker_user_accepts,
    infer_adjacency,
    infer_topology,
    parse_brief_addresses,
    select_route,
    validate_cycle,
)

IF0, IF1 = FABRIC_INTERFACES


def observation(
    name: str,
    addresses: str,
    *,
    hostname: str | None = None,
    reachable: bool = True,
    forward_policy: str | None = "DROP",
    docker_rules: tuple[str, ...] | None = (),
) -> NodeObservation:
    interfaces = parse_brief_addresses(addresses)
    interfaces = {
        interface_name: InterfaceState(
            state.name, state.addresses, state.oper_state, 9000
        )
        for interface_name, state in interfaces.items()
    }
    return NodeObservation(
        NodeSpec(name, f"operator@{name}"),
        reachable,
        hostname=hostname,
        interfaces=interfaces,
        ip_forward=True if reachable else None,
        forward_policy=forward_policy if reachable else None,
        docker_user_rules=docker_rules if reachable else None,
        error="SSH exited 255: connection timed out" if not reachable else "",
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


def probe_output(hostname: str, addresses: str) -> str:
    return f"""__RING_DOCTOR_HOSTNAME__
{hostname}
__RING_DOCTOR_ADDR__
{addresses}
__RING_DOCTOR_LINK__
2: {IF0}: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 state UP mode DEFAULT
3: {IF1}: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 state UP mode DEFAULT
__RING_DOCTOR_ROUTE__
__RING_DOCTOR_FORWARD__
1
__RING_DOCTOR_FORWARD_CHAIN__
-P FORWARD DROP
__RING_DOCTOR_DOCKER_USER__
-N DOCKER-USER
__RING_DOCTOR_INTERFACES__
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


if __name__ == "__main__":
    unittest.main()
