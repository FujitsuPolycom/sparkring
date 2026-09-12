#!/usr/bin/env python3
"""Logic tests for qualify_direct_cable.py; no SSH or NIC access."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("qualify_direct_cable.py")
SPEC = importlib.util.spec_from_file_location("qualify_direct_cable", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def good_snapshot() -> dict:
    return {
        "interface": "enP7s7",
        "interface_exists": True,
        "carrier": "1",
        "operstate": "up",
        "speed": "10000",
        "mtu": "1500",
        "address": "02:00:00:00:00:01",
        "ip_addresses": ["198.51.100.1"],
        "route": {"dev": "enP7s7", "prefsrc": "198.51.100.1"},
        "ping": {"returncode": 0, "stderr": ""},
        "ping_received": 5,
        "counters": {
            "sysfs.rx_errors": 1,
            "sysfs.rx_dropped": 7,
            "ethtool.rx_mac_error": 0,
        },
    }


class QualificationLogicTest(unittest.TestCase):
    def endpoint(self) -> object:
        return MODULE.Endpoint(
            "left",
            "user@192.0.2.10",
            "enP7s7",
            "198.51.100.1",
            None,
            None,
        )

    def test_exact_interface_ip_route_speed_and_mtu_pass(self) -> None:
        checks = MODULE.evaluate_snapshot(
            good_snapshot(),
            self.endpoint(),
            tier="diagonal10",
            expected_mtu=1500,
            expected_speed_mbps=10000,
            gid_index=3,
        )
        self.assertTrue(all(item["passed"] for item in checks), checks)

    def test_wrong_route_fails_closed(self) -> None:
        snapshot = good_snapshot()
        snapshot["route"] = {
            "dev": "wlP9s9",
            "prefsrc": "192.0.2.10",
        }
        checks = MODULE.evaluate_snapshot(
            snapshot,
            self.endpoint(),
            tier="diagonal10",
            expected_mtu=1500,
            expected_speed_mbps=10000,
            gid_index=3,
        )
        direct_route = next(
            item for item in checks if item["name"] == "left.direct_route"
        )
        self.assertFalse(direct_route["passed"])
        self.assertTrue(direct_route["hard"])

    def test_explicit_source_route_matches_live_iproute_output(self) -> None:
        for source_fields, expected in [
            ({"from": "198.51.100.1"}, True),
            ({"from": "198.51.100.2"}, False),
            ({}, False),
            ({"from": "198.51.100.1", "prefsrc": "198.51.100.2"}, False),
            ({"from": "198.51.100.1", "gateway": "198.51.100.254"}, False),
        ]:
            with self.subTest(source_fields=source_fields):
                snapshot = good_snapshot()
                snapshot["route"] = {"dev": "enP7s7", **source_fields}
                checks = MODULE.evaluate_snapshot(
                    snapshot, self.endpoint(), tier="diagonal10",
                    expected_mtu=1500, expected_speed_mbps=10000, gid_index=3,
                )
                route_gate = next(c for c in checks if c["name"] == "left.direct_route")
                self.assertEqual(route_gate["passed"], expected)

    def test_counter_deltas_distinguish_phy_from_pressure(self) -> None:
        before = good_snapshot()
        after = good_snapshot()
        after["counters"] = {
            "sysfs.rx_errors": 3,
            "sysfs.rx_dropped": 12,
            "ethtool.rx_mac_error": 1,
        }
        deltas = MODULE.counter_deltas(before, after)
        self.assertEqual(deltas["phy"]["sysfs.rx_errors"], 2)
        self.assertEqual(deltas["phy"]["ethtool.rx_mac_error"], 1)
        self.assertEqual(deltas["pressure"]["sysfs.rx_dropped"], 5)

    def test_raw_crc_or_loss_is_hard_cable_gate(self) -> None:
        run = {
            "direction": "left->right",
            "payload_bytes": 12288,
            "sender_returncode": 1,
            "receiver_returncode": 1,
            "sender": {
                "valid": False,
                "p99_us": 12.0,
                "payload_crc_errors": 1,
            },
            "receiver": {"valid": False},
        }
        checks = MODULE.raw_probe_gates([run], 30.0)
        integrity = next(item for item in checks if item["name"].endswith(".integrity"))
        self.assertFalse(integrity["passed"])
        self.assertTrue(integrity["hard"])
        self.assertEqual(integrity["domain"], "cable_or_phy")

    def test_high_raw_latency_is_warning_not_bad_cable(self) -> None:
        clean = {name: 0 for name in MODULE.RAW_ERROR_FIELDS}
        run = {
            "direction": "left->right",
            "payload_bytes": 12288,
            "sender_returncode": 0,
            "receiver_returncode": 0,
            "sender": {"valid": True, "p99_us": 161.8, **clean},
            "receiver": {"valid": True, **clean},
        }
        checks = MODULE.raw_probe_gates([run], 30.0)
        result = {}
        code = MODULE.finalize(
            result, checks, probe_completed=True, strict_latency=False
        )
        self.assertEqual(code, MODULE.EXIT_QUALIFIED)
        self.assertTrue(result["cable_qualified"])
        self.assertFalse(result["latency_target_met"])
        self.assertEqual(result["status"], "cable_qualified_with_latency_warning")

    def test_strict_latency_blocks_model_path(self) -> None:
        checks = [
            MODULE.gate("integrity", True, {}, domain="cable_or_phy"),
            MODULE.gate(
                "latency",
                False,
                {"p99_us": 161.8},
                domain="software_latency",
                hard=False,
            ),
        ]
        result = {}
        code = MODULE.finalize(
            result, checks, probe_completed=True, strict_latency=True
        )
        self.assertEqual(code, MODULE.EXIT_FAILED)
        self.assertTrue(result["cable_qualified"])
        self.assertFalse(result["model_path_ready"])
        self.assertEqual(result["failure_domain"], "software_latency")

    def test_preflight_without_probe_is_incomplete(self) -> None:
        result = {}
        code = MODULE.finalize(result, [], probe_completed=False, strict_latency=False)
        self.assertEqual(code, MODULE.EXIT_INCOMPLETE)
        self.assertFalse(result["cable_qualified"])

    def test_rdma_output_parsers(self) -> None:
        result = MODULE.parse_result_line(
            "RESULT memory=host producer=cpu bytes=16384 samples=10000 "
            "p50_us=4.79 p99_us=4.91\n"
        )
        verify = MODULE.parse_verify_line(
            "VERIFY memory=host verifier=cpu correct=true\n"
        )
        self.assertEqual(result["bytes"], 16384)
        self.assertEqual(result["p99_us"], 4.91)
        self.assertEqual(verify["correct"], "true")

    def test_payloads_are_exact_glm_shapes(self) -> None:
        import argparse

        self.assertEqual(MODULE.parse_payloads("12288,16384"), (12288, 16384))
        with self.assertRaises(argparse.ArgumentTypeError):
            MODULE.parse_payloads("4096")

    def test_l2_peer_gate_requires_real_mac_addresses(self) -> None:
        left = {"lladdr": "02:00:00:00:00:02", "state": ["REACHABLE"]}
        right = {"lladdr": "02:00:00:00:00:01", "state": ["STALE"]}
        self.assertTrue(
            MODULE.l2_peers_exact(left, "02:00:00:00:00:01", right, "02:00:00:00:00:02")
        )
        # Missing evidence on both sides must not match as empty strings.
        self.assertFalse(MODULE.l2_peers_exact({}, None, {}, None))
        self.assertFalse(MODULE.l2_peers_exact({}, "", {}, ""))
        self.assertFalse(
            MODULE.l2_peers_exact(left, "02:00:00:00:00:01", {}, "02:00:00:00:00:02")
        )
        self.assertFalse(
            MODULE.l2_peers_exact(left, "02:00:00:00:00:09", right, "02:00:00:00:00:02")
        )
        failed = {"lladdr": "02:00:00:00:00:02", "state": ["FAILED"]}
        self.assertFalse(
            MODULE.l2_peers_exact(failed, "02:00:00:00:00:01", right, "02:00:00:00:00:02")
        )

    def test_vanished_counter_is_an_instrumentation_reset(self) -> None:
        before = {"counters": {"ethtool.rx_crc_errors": 5, "sysfs.rx_dropped": 1}}
        after = {"counters": {"sysfs.rx_dropped": 1, "ethtool.new_counter": 3}}
        deltas = MODULE.counter_deltas(before, after)
        self.assertEqual(deltas["reset"], {"ethtool.rx_crc_errors": -5})
        self.assertEqual(deltas["other"], {"ethtool.new_counter": 3})
        self.assertEqual(deltas["phy"], {})

    def test_postflight_counter_gates_are_warnings_without_probe_traffic(self) -> None:
        deltas = {
            "left": {"phy": {"ethtool.rx_crc_errors": 2}, "pressure": {}, "other": {}, "reset": {}},
            "right": {"phy": {}, "pressure": {}, "other": {}, "reset": {"sysfs.rx_bytes": -9}},
        }
        soft = MODULE.postflight_gates(deltas, ("left", "right"), probe_completed=False)
        hard = MODULE.postflight_gates(deltas, ("left", "right"), probe_completed=True)
        failing_soft = [g for g in soft if not g["passed"]]
        failing_hard = [g for g in hard if not g["passed"]]
        self.assertEqual([g["name"] for g in failing_soft], ["left.no_phy_error_delta", "right.no_counter_reset"])
        self.assertTrue(all(not g["hard"] for g in failing_soft))
        self.assertTrue(all(g["hard"] for g in failing_hard))
        result: dict = {}
        self.assertEqual(
            MODULE.finalize(result, soft, probe_completed=False, strict_latency=False),
            MODULE.EXIT_INCOMPLETE,
        )
        self.assertEqual(result["status"], "incomplete")
        result = {}
        self.assertEqual(
            MODULE.finalize(result, hard, probe_completed=True, strict_latency=False),
            MODULE.EXIT_FAILED,
        )
        self.assertEqual(result["failure_domain"], "cable_or_phy,instrumentation")

    def test_expected_probe_digest_must_be_hex(self) -> None:
        parser = MODULE.build_parser()
        base = [
            "--tier", "diagonal10", "--left", "user@left", "--right", "user@right",
            "--left-interface", "enP7s7", "--right-interface", "enP7s7",
            "--left-ip", "198.51.100.1", "--right-ip", "198.51.100.2",
            "--expected-mtu", "1500", "--preflight-only",
        ]
        args = parser.parse_args(base + ["--probe-binary-sha256", "zz"])
        with self.assertRaisesRegex(MODULE.QualificationError, "64 hex digits"):
            MODULE.run(args)


if __name__ == "__main__":
    unittest.main()
