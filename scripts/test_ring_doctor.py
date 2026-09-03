"""Offline unit tests for the SparkRing fabric doctor."""

from __future__ import annotations

import ipaddress
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.ring_doctor import (
    FABRIC_INTERFACES,
    UNAVAILABLE,
    CommandResult,
    ControllerIdentity,
    InterfaceState,
    ManagementGuard,
    NodeObservation,
    NodePlan,
    NodeSpec,
    ProposedRoute,
    _build_parser,
    _load_inputs,
    _unit_program,
    build_plans,
    build_diagnostic_checks,
    build_management_guards,
    build_probe_command,
    diagnose,
    diagnose_launch_endpoints,
    diagnose_wifi_resilience,
    discover_nodes,
    discovery_sufficient,
    docker_user_accepts,
    emit_units,
    infer_adjacency,
    infer_topology,
    enforce_controller_location,
    parse_all_addresses,
    parse_brief_addresses,
    parse_route_get,
    parse_wifi_observation,
    apply_plans,
    select_route,
    validate_cycle,
    validate_plan_management_safety,
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
    wifi: str | None = None,
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
        wifi=(
            parse_wifi_observation(wifi)
            if wifi is not None and reachable
            else None
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


def synthetic_six_cycle() -> dict[str, NodeObservation]:
    size = 6
    return {
        f"r{rank}": observation(
            f"r{rank}",
            f"{IF0} UP 10.30.{rank}.{rank + 10}/24\n"
            f"{IF1} UP 10.30.{(rank - 1) % size}.{rank + 100}/24",
        )
        for rank in range(size)
    }


def probe_output(
    hostname: str,
    addresses: str,
    route_get: str | None = None,
    routes: str = "",
    docker_rules: str = "-N DOCKER-USER",
    wifi: str | None = None,
) -> str:
    rendezvous = (
        f"__RING_DOCTOR_RENDEZVOUS__\n{route_get}\n" if route_get is not None else ""
    )
    wifi_evidence = f"__RING_DOCTOR_WIFI__\n{wifi}\n" if wifi is not None else ""
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
{wifi_evidence}{rendezvous}__RING_DOCTOR_INTERFACES__
{IF0}
{IF1}
"""


class FakeDiscoveryRunner:
    def __init__(self, outputs: dict[str, str], down: set[str]) -> None:
        self.outputs = outputs
        self.down = down
        self.calls: list[tuple[str, str | None]] = []
        self.commands: list[tuple[str, str]] = []

    def run(
        self, target: str, command: str, proxy_jump: str | None = None
    ) -> CommandResult:
        self.calls.append((target, proxy_jump))
        self.commands.append((target, command))
        if target in self.down:
            return CommandResult(False, detail="SSH exited 255: connection timed out")
        return CommandResult(True, stdout=self.outputs[target])


class SiteConfigInputTests(unittest.TestCase):
    SITE = Path(__file__).parent / "config" / "exl3-r7-site.example.yaml"

    def test_site_supplies_canonical_nodes_interfaces_and_launch_endpoints(self) -> None:
        args = _build_parser().parse_args(["--site", str(self.SITE)])

        loaded = _load_inputs(args)
        specs = loaded.specs

        self.assertEqual([spec.name for spec in specs], ["rank0", "rank1", "rank2", "rank3"])
        self.assertEqual(specs[0].target, "operator@198.18.1.10")
        self.assertEqual(specs[0].fabric_interfaces, ("eth1", "eth2"))
        self.assertEqual(specs[0].socket_interfaces, ("eth0",))
        self.assertEqual(loaded.interfaces, ("eth1", "eth2"))
        self.assertEqual(loaded.timeout, 45)
        self.assertEqual(
            loaded.rendezvous_address, ipaddress.ip_address("198.18.1.10")
        )
        self.assertIsNotNone(loaded.site)

    def test_site_rejects_a_second_cluster_description(self) -> None:
        args = _build_parser().parse_args(
            ["--site", str(self.SITE), "--node", "operator@rank0"]
        )

        with self.assertRaisesRegex(ValueError, "--site cannot be combined with --node"):
            _load_inputs(args)

    def test_discovery_uses_each_nodes_site_defined_interface_names(self) -> None:
        specs = (
            NodeSpec("r0", "operator@r0", fabric_interfaces=("cx0a", "cx0b")),
            NodeSpec("r1", "operator@r1", fabric_interfaces=("cx1a", "cx1b")),
        )
        outputs = {
            "operator@r0": probe_output(
                "spark-r0", "cx0a UP 10.0.1.10/24\ncx0b UP 10.0.4.10/24"
            ).replace(IF0, "cx0a").replace(IF1, "cx0b"),
            "operator@r1": probe_output(
                "spark-r1", "cx1a UP 10.0.2.11/24\ncx1b UP 10.0.1.11/24"
            ).replace(IF0, "cx1a").replace(IF1, "cx1b"),
        }
        runner = FakeDiscoveryRunner(outputs, set())

        observations = discover_nodes(specs, runner)

        commands = dict(runner.commands)
        self.assertIn("cx0a", commands["operator@r0"])
        self.assertNotIn("cx1a", commands["operator@r0"])
        self.assertIn("cx1a", commands["operator@r1"])
        self.assertEqual(set(observations["r0"].interfaces), {"cx0a", "cx0b"})
        self.assertEqual(set(observations["r1"].interfaces), {"cx1a", "cx1b"})

    def test_controller_defaults_to_head_and_worker_requires_recovery_flag(self) -> None:
        loaded = _load_inputs(
            _build_parser().parse_args(["--site", str(self.SITE)])
        )
        head = ControllerIdentity(
            frozenset(), frozenset({ipaddress.ip_address("198.18.1.10")})
        )
        worker = ControllerIdentity(
            frozenset(), frozenset({ipaddress.ip_address("198.18.1.12")})
        )

        self.assertEqual(
            enforce_controller_location(loaded, allow_worker=False, identity=head),
            "rank0",
        )
        with self.assertRaisesRegex(ValueError, "--allow-worker-controller"):
            enforce_controller_location(loaded, allow_worker=False, identity=worker)
        self.assertEqual(
            enforce_controller_location(loaded, allow_worker=True, identity=worker),
            "rank2",
        )

    def test_worker_override_never_allows_an_external_controller(self) -> None:
        loaded = _load_inputs(
            _build_parser().parse_args(["--site", str(self.SITE)])
        )
        external = ControllerIdentity(
            frozenset({"operator-laptop"}),
            frozenset({ipaddress.ip_address("203.0.114.50")}),
        )

        with self.assertRaisesRegex(ValueError, "configured Spark"):
            enforce_controller_location(loaded, allow_worker=True, identity=external)


class DiagnosticReceiptAdapterTests(unittest.TestCase):
    def test_healthy_topology_produces_affirmative_shared_check(self) -> None:
        topology = infer_topology(synthetic_cycle())

        checks = build_diagnostic_checks(topology, [])

        self.assertEqual(checks[0].check_id, "fabric-topology-cycle")
        self.assertEqual(checks[0].status.value, "pass")

    def test_warning_is_unknown_not_a_pass(self) -> None:
        topology = infer_topology(synthetic_cycle())
        findings = diagnose_wifi_resilience(
            [NodeSpec("r0", "operator@r0")],
            {
                "r0": observation(
                    "r0",
                    f"{IF0} UP 10.0.1.10/24\n{IF1} UP 10.0.4.10/24",
                    wifi=UNAVAILABLE,
                )
            },
        )

        checks = build_diagnostic_checks(topology, findings)

        self.assertEqual(checks[-1].status.value, "unknown")


class ManagementRepairSafetyTests(unittest.TestCase):
    def guarded_observation(
        self, name: str = "r0", management_address: str = "192.0.2.10"
    ) -> NodeObservation:
        return observation(
            name,
            f"{IF0} UP 10.0.1.10/24\n{IF1} UP 10.0.4.10/24",
            socket_interfaces=("eth0",),
            host_addresses=(
                f"{IF0} UP 10.0.1.10/24\n"
                f"{IF1} UP 10.0.4.10/24\n"
                f"eth0 UP {management_address}/24"
            ),
        )

    def run_management_guard(
        self,
        route: str | None,
        *,
        client: str = "192.0.2.10",
        server: str = "192.0.2.10",
        ssh_connection: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if os.name == "nt":
            self.skipTest("behavioral shell guard tests require a native POSIX shell")
        guarded = self.guarded_observation()
        guard = ManagementGuard("r0", (guarded.host_interfaces["eth0"],))
        connection = ssh_connection or f"{client} 47254 {server} 22"
        route_case = (
            "return 1"
            if route is None
            else f"printf '%s\\n' {shlex.quote(route)}"
        )
        script = "\n".join(
            (
                "set -e",
                "ip() {",
                '  case "$*" in',
                '    "link show dev eth0") return 0 ;;',
                '    "-4 -o addr show dev eth0") '
                "printf '%s\\n' '2: eth0 inet 192.0.2.10/24 scope global eth0' ;;",
                f'    "-4 route get {client}") {route_case} ;;',
                "    *) return 1 ;;",
                "  esac",
                "}",
                f"export SSH_CONNECTION={shlex.quote(connection)}",
                guard.shell_command(),
            )
        )
        return subprocess.run(
            ["bash", "-c", script],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_management_guard_distinguishes_guarded_local_client(self) -> None:
        guarded = self.guarded_observation()
        command = ManagementGuard(
            "r0", (guarded.host_interfaces["eth0"],)
        ).shell_command()

        self.assertIn("ssh_client=$1", command)
        self.assertIn('[ "$1" = local ]', command)
        self.assertIn('case "$ssh_client" in 192.0.2.10)', command)

    def test_management_guard_accepts_self_ssh_on_guarded_address(self) -> None:
        result = self.run_management_guard(
            "local 192.0.2.10 dev lo src 192.0.2.10 uid 1000"
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_management_guard_accepts_remote_management_route(self) -> None:
        result = self.run_management_guard(
            "192.0.2.20 dev eth0 src 192.0.2.10 uid 1000",
            client="192.0.2.20",
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_management_guard_rejects_self_ssh_from_unrecognized_address(self) -> None:
        result = self.run_management_guard(
            "local 192.0.2.99 dev lo src 192.0.2.99 uid 1000",
            client="192.0.2.99",
        )

        self.assertNotEqual(result.returncode, 0)

    def test_management_guard_rejects_remote_route_over_fabric(self) -> None:
        result = self.run_management_guard(
            f"10.0.1.20 dev {IF0} src 10.0.1.10 uid 1000",
            client="10.0.1.20",
        )

        self.assertNotEqual(result.returncode, 0)

    def test_management_guard_rejects_unrecognized_server_address(self) -> None:
        result = self.run_management_guard(
            "192.0.2.20 dev eth0 src 192.0.2.10 uid 1000",
            client="192.0.2.20",
            server="192.0.2.99",
        )

        self.assertNotEqual(result.returncode, 0)

    def test_management_guard_rejects_malformed_ssh_connection(self) -> None:
        result = self.run_management_guard(
            "192.0.2.20 dev eth0 src 192.0.2.10 uid 1000",
            ssh_connection="malformed",
        )

        self.assertNotEqual(result.returncode, 0)

    def test_management_guard_rejects_missing_route_evidence(self) -> None:
        result = self.run_management_guard(None, client="192.0.2.20")

        self.assertNotEqual(result.returncode, 0)

    def test_management_guard_requires_distinct_addressed_interface(self) -> None:
        guarded = self.guarded_observation()
        guards, findings = build_management_guards([guarded.spec], {"r0": guarded})

        self.assertEqual(findings, [])
        self.assertEqual(guards["r0"].address_map(), {"eth0": ["192.0.2.10/24"]})

        unguarded = observation(
            "r0", f"{IF0} UP 10.0.1.10/24\n{IF1} UP 10.0.4.10/24"
        )
        guards, findings = build_management_guards(
            [unguarded.spec], {"r0": unguarded}
        )
        self.assertEqual(guards, {})
        self.assertEqual(findings[0].code, "management-guard-unconfigured")

    def test_plan_overlapping_management_subnet_is_rejected(self) -> None:
        guarded = self.guarded_observation()
        guard = ManagementGuard(
            "r0", (guarded.host_interfaces["eth0"],)
        )
        plan = NodePlan(
            "r0",
            routes=[
                ProposedRoute(
                    ipaddress.ip_network("192.0.2.0/24"),
                    ipaddress.ip_address("10.0.1.11"),
                    IF0,
                    ("r0", "r1"),
                )
            ],
        )

        reason = validate_plan_management_safety(guarded, plan, guard)

        self.assertIn("contains a management address", reason or "")

    def test_apply_checks_management_before_and_after_each_change(self) -> None:
        guarded = self.guarded_observation()
        guard = ManagementGuard("r0", (guarded.host_interfaces["eth0"],))
        plan = NodePlan(
            "r0",
            routes=[
                ProposedRoute(
                    ipaddress.ip_network("10.0.2.0/24"),
                    ipaddress.ip_address("10.0.1.11"),
                    IF0,
                    ("r0", "r1"),
                )
            ],
        )

        class ApplyRunner:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def run(self, target, command, proxy_jump=None):
                self.commands.append(command)
                return CommandResult(True)

        runner = ApplyRunner()
        application = apply_plans(
            {"r0": guarded}, {"r0": plan}, {"r0": guard}, runner
        )

        self.assertTrue(application.executed)
        self.assertEqual(application.findings, ())
        self.assertEqual(len(runner.commands), 2)
        self.assertEqual(runner.commands[0], "sudo -n true")
        self.assertGreaterEqual(runner.commands[1].count("192.0.2.10/24"), 2)
        self.assertIn("ip route replace 10.0.2.0/24", runner.commands[1])

    def test_apply_executes_repair_between_self_ssh_guards(self) -> None:
        if os.name == "nt":
            self.skipTest("behavioral shell guard tests require a native POSIX shell")
        guarded = self.guarded_observation()
        guard = ManagementGuard("r0", (guarded.host_interfaces["eth0"],))

        class RecordedPlan:
            routes = ()
            relay_directions = set()

            @staticmethod
            def commands():
                return ("printf 'REPAIR_EXECUTED\\n'",)

        class ShellRunner:
            def __init__(self) -> None:
                self.results: list[subprocess.CompletedProcess[str]] = []

            def run(self, _target, command, proxy_jump=None):
                del proxy_jump
                if command == "sudo -n true":
                    return CommandResult(True)
                prefix = "\n".join(
                    (
                        "ip() {",
                        '  case "$*" in',
                        '    "link show dev eth0") return 0 ;;',
                        '    "-4 -o addr show dev eth0") '
                        "printf '%s\\n' "
                        "'2: eth0 inet 192.0.2.10/24 scope global eth0' ;;",
                        '    "-4 route get 192.0.2.10") '
                        "printf '%s\\n' "
                        "'local 192.0.2.10 dev lo src 192.0.2.10 uid 1000' ;;",
                        "    *) return 1 ;;",
                        "  esac",
                        "}",
                        "export SSH_CONNECTION='192.0.2.10 47254 192.0.2.10 22'",
                    )
                )
                completed = subprocess.run(
                    ["bash", "-c", prefix + "\n" + command],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.results.append(completed)
                return CommandResult(
                    completed.returncode == 0,
                    completed.stdout,
                    completed.stderr,
                )

        runner = ShellRunner()
        application = apply_plans(
            {"r0": guarded},
            {"r0": RecordedPlan()},  # type: ignore[dict-item]
            {"r0": guard},
            runner,
        )

        self.assertTrue(application.executed)
        self.assertEqual(application.findings, ())
        self.assertEqual(len(runner.results), 1)
        self.assertEqual(runner.results[0].stdout, "REPAIR_EXECUTED\n")

    def test_apply_withholds_every_plan_when_any_node_lacks_noninteractive_sudo(
        self,
    ) -> None:
        rank0 = self.guarded_observation()
        rank1 = self.guarded_observation("r1", "192.0.2.11")
        observations = {"r0": rank0, "r1": rank1}
        plans = {
            name: NodePlan(
                name,
                routes=[
                    ProposedRoute(
                        ipaddress.ip_network("10.0.2.0/24"),
                        ipaddress.ip_address("10.0.1.11"),
                        IF0,
                        (name, "peer"),
                    )
                ],
            )
            for name in observations
        }
        guards = {
            name: ManagementGuard(name, (item.host_interfaces["eth0"],))
            for name, item in observations.items()
        }

        class PrivilegeRunner:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def run(self, target, command, proxy_jump=None):
                self.calls.append((target, command))
                if command != "sudo -n true":
                    raise AssertionError(f"repair command executed on {target}")
                if target == "operator@r0":
                    return CommandResult(False, stderr="sudo: a password is required")
                return CommandResult(True)

        runner = PrivilegeRunner()
        application = apply_plans(observations, plans, guards, runner)

        self.assertFalse(application.executed)
        self.assertEqual(
            [(finding.node, finding.code) for finding in application.findings],
            [("r0", "apply-failed")],
        )
        self.assertIn("non-interactive sudo", application.findings[0].message)
        self.assertEqual(
            runner.calls,
            [
                ("operator@r0", "sudo -n true"),
                ("operator@r1", "sudo -n true"),
            ],
        )

    def test_boot_program_checks_management_after_every_change(self) -> None:
        guarded = self.guarded_observation()
        guard = ManagementGuard("r0", (guarded.host_interfaces["eth0"],))
        plan = NodePlan(
            "r0",
            routes=[
                ProposedRoute(
                    ipaddress.ip_network("10.0.2.0/24"),
                    ipaddress.ip_address("10.0.1.11"),
                    IF0,
                    ("r0", "r1"),
                )
            ],
            relay_directions={(IF0, IF1)},
        )

        program = _unit_program(guarded, plan, guard)

        self.assertIn("EXPECTED_MANAGEMENT = {'eth0': ['192.0.2.10/24']}", program)
        self.assertGreaterEqual(program.count("require_management()"), 5)

    def test_emitted_unit_restarts_after_program_failure(self) -> None:
        guarded = self.guarded_observation()
        guard = ManagementGuard("r0", (guarded.host_interfaces["eth0"],))
        plan = NodePlan("r0")

        with tempfile.TemporaryDirectory() as directory:
            emit_units(
                Path(directory),
                {"r0": guarded},
                {"r0": plan},
                {"r0": guard},
            )
            unit = Path(directory, "ring-doctor-r0.service").read_text(
                encoding="utf-8"
            )

        self.assertIn(
            "[Service]\n"
            "Type=oneshot\n"
            "Restart=on-failure\n"
            "RestartSec=10\n"
            "ExecStart=",
            unit,
        )

    def test_emitted_unit_restores_firewall_plan_after_docker_startup(self) -> None:
        guarded = self.guarded_observation()
        guard = ManagementGuard("r0", (guarded.host_interfaces["eth0"],))
        plan = NodePlan("r0", relay_directions={(IF0, IF1)})

        with tempfile.TemporaryDirectory() as directory:
            emit_units(
                Path(directory),
                {"r0": guarded},
                {"r0": plan},
                {"r0": guard},
            )
            root = Path(directory)
            unit = root.joinpath("ring-doctor-r0.service").read_text(
                encoding="utf-8"
            )
            program = root.joinpath("ring-doctor-r0.py").read_text(
                encoding="utf-8"
            )

        self.assertIn("After=network-online.target docker.service", unit)
        self.assertIn(f"RELAY_DIRECTIONS = [['{IF0}', '{IF1}']]", program)
        self.assertIn('["iptables", "-I", "DOCKER-USER", "1", *rule]', program)


class AddressAndTopologyTests(unittest.TestCase):
    def test_six_node_cycle_infers_routes_for_every_remote_edge(self) -> None:
        observations = synthetic_six_cycle()

        topology = infer_topology(observations)
        plans = build_plans(observations, topology)

        self.assertTrue(topology.valid_cycle, topology.reason)
        self.assertEqual(len(topology.cycle), 6)
        self.assertTrue(
            discovery_sufficient(
                [observation.spec for observation in observations.values()],
                observations,
                topology,
            )
        )
        for plan in plans.values():
            self.assertEqual(len(plan.routes), 4)
            self.assertEqual(
                plan.relay_directions, {(IF0, IF1), (IF1, IF0)}
            )

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

WIFI_PROFILE_NAME = "CSmiles"

WIFI_DEVICE_ROWS = (
    f"{WIFI}:wifi:connected:{WIFI_PROFILE_NAME}\n"
    f"p2p-dev-{WIFI}:wifi-p2p:disconnected:\n"
    f"{IF0}:ethernet:connected:fabric-101\n"
    "lo:loopback:unmanaged:"
)

WIRED_DEVICE_ROWS = (
    f"{IF0}:ethernet:connected:fabric-101\n"
    f"{IF1}:ethernet:connected:fabric-102\n"
    "lo:loopback:unmanaged:"
)

DISCONNECTED_WIFI_ROWS = (
    f"{WIFI}:wifi:disconnected:\n"
    f"{IF0}:ethernet:connected:fabric-101\n"
    "lo:loopback:unmanaged:"
)


def wifi_section(
    autoconnect: str = "yes",
    retries: str = "0",
    powersave: str = "disable",
    driver: str = "off",
) -> str:
    return (
        f"{WIFI_DEVICE_ROWS}\n"
        f"PROFILE {WIFI} {WIFI_PROFILE_NAME}\n"
        f"{autoconnect}\n"
        f"{retries}\n"
        f"{powersave}\n"
        f"IW {WIFI} {driver}"
    )


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


class WifiResilienceTests(unittest.TestCase):
    @staticmethod
    def wifi_findings(sections):
        observations = {
            name: observation(name, ADDRESSED_WIFI_LISTING, wifi=section)
            for name, section in sections.items()
        }
        return diagnose_wifi_resilience(
            [item.spec for item in observations.values()], observations
        )

    def test_probe_gathers_wifi_evidence_in_the_same_contact(self) -> None:
        command = build_probe_command(FABRIC_INTERFACES)

        self.assertEqual(command.count("__RING_DOCTOR_WIFI__"), 1)
        self.assertIn("command -v nmcli", command)
        self.assertIn(
            "nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device status", command
        )
        self.assertIn(
            "nmcli -t -g connection.autoconnect,"
            "connection.autoconnect-retries,802-11-wireless.powersave "
            "connection show",
            command,
        )
        self.assertIn("get power_save", command)

    def test_sound_profiles_on_every_wifi_node_report_one_affirmative_info(
        self,
    ) -> None:
        findings = self.wifi_findings(
            {"r0": wifi_section(), "r1": wifi_section()}
        )

        self.assertEqual(
            [finding.code for finding in findings], ["wifi-resilience-observed"]
        )
        self.assertEqual(findings[0].severity, "info")
        self.assertIsNone(findings[0].node)
        self.assertIn(f"r0:{WIFI}", findings[0].evidence)
        self.assertIn(f"r1:{WIFI}", findings[0].evidence)
        self.assertIn(WIFI_PROFILE_NAME, findings[0].evidence)
        self.assertIn("retries=0", findings[0].evidence)

    def test_default_retry_limit_is_reported_as_four_attempts(self) -> None:
        findings = self.wifi_findings(
            {"r0": wifi_section(), "r1": wifi_section(retries="-1")}
        )

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(
            (finding.severity, finding.code, finding.node),
            ("warning", "wifi-retries-limited", "r1"),
        )
        self.assertIn(WIFI, finding.message)
        self.assertIn(WIFI_PROFILE_NAME, finding.message)
        self.assertIn("four", finding.message)
        self.assertIn("blocks until a NetworkManager timeout", " ".join(finding.message.split()))
        self.assertIn("off the management network", finding.message)
        self.assertIn("connection.autoconnect-retries=-1", finding.evidence)

    def test_finite_retry_cap_is_reported_with_its_count(self) -> None:
        findings = self.wifi_findings({"r0": wifi_section(retries="3")})

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.code, "wifi-retries-limited")
        self.assertIn("cap of 3 reconnection attempts", finding.message)
        self.assertIn("after 3 failed attempts", finding.message)
        self.assertIn("connection.autoconnect-retries=3", finding.evidence)

    def test_disabled_autoconnect_is_reported(self) -> None:
        findings = self.wifi_findings({"r0": wifi_section(autoconnect="no")})

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(
            (finding.severity, finding.code, finding.node),
            ("warning", "wifi-autoconnect-disabled", "r0"),
        )
        self.assertIn(WIFI, finding.message)
        self.assertIn(WIFI_PROFILE_NAME, finding.message)
        self.assertIn("connection.autoconnect=no", finding.evidence)

    def test_profile_powersave_enable_is_reported(self) -> None:
        for value in ("enable", "3"):
            with self.subTest(powersave=value):
                findings = self.wifi_findings(
                    {"r0": wifi_section(powersave=value)}
                )

                self.assertEqual(len(findings), 1)
                finding = findings[0]
                self.assertEqual(
                    (finding.severity, finding.code, finding.node),
                    ("warning", "wifi-powersave-enabled", "r0"),
                )
                self.assertIn("de-association", finding.message)
                self.assertIn(
                    f"802-11-wireless.powersave={value}", finding.evidence
                )

    def test_driver_power_save_on_is_reported_despite_a_disabling_profile(
        self,
    ) -> None:
        findings = self.wifi_findings(
            {"r0": wifi_section(powersave="disable", driver="on")}
        )

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(finding.code, "wifi-powersave-enabled")
        self.assertIn("driver reports power save on", finding.message)
        self.assertIn("iw power_save=on", finding.evidence)

    def test_absent_iw_leaves_the_profile_verdict_standing(self) -> None:
        findings = self.wifi_findings(
            {"r0": wifi_section(driver="unavailable")}
        )

        self.assertEqual(
            [finding.code for finding in findings], ["wifi-resilience-observed"]
        )
        self.assertIn("driver_power_save=unobserved", findings[0].evidence)

    def test_nodes_without_an_active_wifi_profile_contribute_nothing(
        self,
    ) -> None:
        wired_only = self.wifi_findings(
            {"r0": WIRED_DEVICE_ROWS, "r1": DISCONNECTED_WIFI_ROWS}
        )
        with_one_wifi_node = self.wifi_findings(
            {
                "r0": WIRED_DEVICE_ROWS,
                "r1": DISCONNECTED_WIFI_ROWS,
                "r2": wifi_section(),
            }
        )

        self.assertEqual(wired_only, [])
        self.assertEqual(
            [finding.code for finding in with_one_wifi_node],
            ["wifi-resilience-observed"],
        )
        self.assertIn(f"r2:{WIFI}", with_one_wifi_node[0].evidence)
        self.assertNotIn("r0", with_one_wifi_node[0].evidence)

    def test_missing_nmcli_is_reported_as_unobserved(self) -> None:
        findings = self.wifi_findings(
            {"r0": UNAVAILABLE, "r1": wifi_section()}
        )

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(
            (finding.severity, finding.code, finding.node),
            ("warning", "wifi-resilience-unobserved", "r0"),
        )
        self.assertIn("nmcli", finding.message)

    def test_unreadable_profile_settings_are_unobserved_not_sound(self) -> None:
        section = (
            f"{WIFI_DEVICE_ROWS}\n"
            f"PROFILE {WIFI} {WIFI_PROFILE_NAME}\n"
            f"{UNAVAILABLE}\n"
            f"IW {WIFI} off"
        )

        findings = self.wifi_findings({"r0": section})

        self.assertEqual(len(findings), 1)
        finding = findings[0]
        self.assertEqual(
            (finding.severity, finding.code, finding.node),
            ("warning", "wifi-resilience-unobserved", "r0"),
        )
        self.assertIn(WIFI_PROFILE_NAME, finding.message)

    def test_p2p_sub_device_is_never_treated_as_an_interface(self) -> None:
        section = (
            f"{WIFI}:wifi:connected:{WIFI_PROFILE_NAME}\n"
            f"p2p-dev-{WIFI}:wifi-p2p:connected:{WIFI_PROFILE_NAME}-p2p\n"
            "lo:loopback:unmanaged:\n"
            f"PROFILE {WIFI} {WIFI_PROFILE_NAME}\n"
            "yes\n"
            "0\n"
            "disable\n"
            f"IW {WIFI} off"
        )

        parsed = parse_wifi_observation(section)
        findings = self.wifi_findings({"r0": section})

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertTrue(parsed.nmcli_available)
        self.assertEqual(
            [interface.device for interface in parsed.interfaces], [WIFI]
        )
        self.assertEqual(
            [finding.code for finding in findings], ["wifi-resilience-observed"]
        )

    def test_unreachable_nodes_and_wifi_less_probe_output_are_skipped(
        self,
    ) -> None:
        observations = {
            "r0": observation("r0", "", reachable=False),
            "r1": observation("r1", ADDRESSED_WIFI_LISTING),
        }

        findings = diagnose_wifi_resilience(
            [item.spec for item in observations.values()], observations
        )

        self.assertEqual(findings, [])


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
                wifi=wifi_section() if name == "r0" else WIRED_DEVICE_ROWS,
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
            [
                ("info", "launch-endpoints-observed"),
                ("info", "wifi-resilience-observed"),
            ],
        )
        self.assertIn(str(RENDEZVOUS), findings[0].message)
        for name in HEALTHY_RING:
            self.assertIn(name, findings[0].evidence)
        self.assertIn(f"r0:{WIFI}", findings[1].evidence)


class WifiDiscoveryIntegrationTests(unittest.TestCase):
    def test_one_probe_round_carries_wifi_evidence_for_every_node(self) -> None:
        specs = tuple(
            NodeSpec(name, f"operator@{name}") for name in HEALTHY_RING
        )
        wifi_by_node = {
            "r0": wifi_section(),
            "r1": WIRED_DEVICE_ROWS,
            "r2": wifi_section(retries="-1"),
            "r3": UNAVAILABLE,
        }
        outputs = {
            f"operator@{name}": probe_output(
                entry[0][0],
                entry[1],
                routes=entry[2],
                docker_rules=CROSS_ACCEPT_RULES,
                wifi=wifi_by_node[name],
            )
            for name, entry in HEALTHY_RING.items()
        }
        runner = FakeDiscoveryRunner(outputs, set())

        observations = discover_nodes(specs, runner)
        topology = infer_topology(observations)
        plans = build_plans(observations, topology)
        findings = diagnose(specs, observations, topology, plans)

        self.assertEqual(len(runner.calls), len(specs))
        self.assertTrue(topology.valid_cycle, topology.reason)
        checked = observations["r0"].wifi
        self.assertIsNotNone(checked)
        assert checked is not None
        self.assertEqual(
            [interface.connection for interface in checked.interfaces],
            [WIFI_PROFILE_NAME],
        )
        self.assertEqual(checked.interfaces[0].driver_power_save, "off")
        wired = observations["r1"].wifi
        self.assertIsNotNone(wired)
        assert wired is not None
        self.assertTrue(wired.nmcli_available)
        self.assertEqual(wired.interfaces, ())
        self.assertEqual(
            [
                (finding.severity, finding.code, finding.node)
                for finding in findings
            ],
            [
                ("warning", "wifi-retries-limited", "r2"),
                ("warning", "wifi-resilience-unobserved", "r3"),
            ],
        )
        self.assertEqual(
            observations["r0"].to_dict()["wifi"]["interfaces"][0]["device"],
            WIFI,
        )


if __name__ == "__main__":
    unittest.main()
