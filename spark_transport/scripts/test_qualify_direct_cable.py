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
        self.assertEqual(MODULE.parse_payloads("12288,16384"), (12288, 16384))
        with self.assertRaises(Exception):
            MODULE.parse_payloads("4096")


if __name__ == "__main__":
    unittest.main()
