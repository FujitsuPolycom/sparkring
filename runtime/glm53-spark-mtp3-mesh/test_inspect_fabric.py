"""CPU-only validation of hardware-rule receipts and bounded marker leases."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mtp3_mesh_inspect_test", HERE / "inspect_fabric.py")
inspector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inspector
SPEC.loader.exec_module(inspector)


@pytest.fixture
def hardware_rule():
    rule = SimpleNamespace(preference=49300, handle=1792,
                           match_ethernet_source="02:00:00:00:00:01",
                           match_ethernet_destination="02:01:00:00:00:01",
                           rewrite_ethernet_destination="02:02:00:00:00:01",
                           egress_netdev="enp1s0f1np1")
    entries = [{"pref": rule.preference, "options": {
        "handle": rule.handle, "skip_sw": True, "in_hw": True,
        "keys": {"src_mac": rule.match_ethernet_source, "dst_mac": rule.match_ethernet_destination,
                 "eth_type": "88b5"},
        "actions": [
            {"kind": "pedit", "keys": [{"htype": "eth", "cmd": "set", "offset": 12,
                                         "val": "08000000", "mask": "0000ffff"}]},
            {"kind": "pedit", "keys": [
                {"htype": "eth", "cmd": "set", "offset": 0, "val": "02020000", "mask": "00000000"},
                {"htype": "eth", "cmd": "set", "offset": 4, "val": "00010000", "mask": "0000ffff"}]},
            {"kind": "mirred", "to_dev": rule.egress_netdev, "mirred_action": "redirect"},
        ],
    }}]
    for action in entries[0]["options"]["actions"]:
        action["stats"] = {"sw_packets": 0, "drops": 0, "packets": 500, "bytes": 8192000}
    return rule, entries


def test_valid_hardware_rule_has_zero_software_and_drop_counters(hardware_rule):
    rule, entries = hardware_rule
    assert inspector.verify_rule(rule, entries) == {
        "preference": 49300, "handle": 1792, "in_hw": True,
        "software_packet_counter_available": True,
    }


def test_hardware_rule_accepts_integer_pedit_values(hardware_rule):
    rule, entries = hardware_rule
    for action in entries[0]["options"]["actions"][:2]:
        for key in action["keys"]:
            key["val"] = int(key["val"], 16)
            key["mask"] = int(key["mask"], 16)
    entries[0]["options"]["keys"]["eth_type"] = 0x88B5
    assert inspector.verify_rule(rule, entries)["in_hw"]


@pytest.mark.parametrize("setting,value", [("skip_sw", False), ("skip_sw", 1), ("in_hw", False), ("in_hw", 1)])
def test_rule_requires_explicit_hardware_only_flags(hardware_rule, setting, value):
    rule, entries = hardware_rule
    entries[0]["options"][setting] = value
    with pytest.raises(ValueError, match="hardware-only"):
        inspector.verify_rule(rule, entries)


@pytest.mark.parametrize("setting,value", [("src_mac", "02:ff:00:00:00:01"),
                                           ("dst_mac", "02:ff:00:00:00:02"), ("eth_type", "0800")])
def test_rule_match_must_identify_the_marked_packet(hardware_rule, setting, value):
    rule, entries = hardware_rule
    entries[0]["options"]["keys"][setting] = value
    with pytest.raises(ValueError, match="match"):
        inspector.verify_rule(rule, entries)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "preference", "handle"])
def test_rule_identity_must_be_unique(hardware_rule, mutation):
    rule, entries = hardware_rule
    if mutation == "missing":
        entries.clear()
    elif mutation == "duplicate":
        entries.append(deepcopy(entries[0]))
    elif mutation == "preference":
        entries[0]["pref"] += 1
    else:
        entries[0]["options"]["handle"] += 1
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        inspector.verify_rule(rule, entries)


@pytest.mark.parametrize("action,key,field,value", [
    (0, 0, "offset", 14), (0, 0, "val", "88b50000"), (0, 0, "mask", "ffffffff"),
    (1, 0, "val", "02030000"), (1, 1, "val", "00020000"),
    (1, 1, "mask", "00000000"), (1, 0, "htype", "ip4"), (0, 0, "cmd", "add"),
])
def test_rewrite_bytes_and_header_are_exact(hardware_rule, action, key, field, value):
    rule, entries = hardware_rule
    entries[0]["options"]["actions"][action]["keys"][key][field] = value
    with pytest.raises(ValueError, match="rewrite"):
        inspector.verify_rule(rule, entries)


@pytest.mark.parametrize("field,value", [("to_dev", "enp1s0f0np0"), ("mirred_action", "mirror")])
def test_redirect_must_use_planned_egress(hardware_rule, field, value):
    rule, entries = hardware_rule
    entries[0]["options"]["actions"][2][field] = value
    with pytest.raises(ValueError, match="redirect"):
        inspector.verify_rule(rule, entries)


@pytest.mark.parametrize("action", range(3))
@pytest.mark.parametrize("counter", ["sw_packets", "drops"])
def test_nonzero_software_or_drop_counter_fails(hardware_rule, action, counter):
    rule, entries = hardware_rule
    entries[0]["options"]["actions"][action]["stats"][counter] = 1
    with pytest.raises(ValueError, match="software packets or drops"):
        inspector.verify_rule(rule, entries)


def test_missing_software_counter_is_not_claimed_as_zero(hardware_rule):
    rule, entries = hardware_rule
    del entries[0]["options"]["actions"][0]["stats"]["sw_packets"]
    receipt = inspector.verify_rule(rule, entries)
    assert receipt["software_packet_counter_available"] is False


@pytest.mark.parametrize("actions", [[], [{"kind": "mirred"}], [{"kind": "pedit"}] * 4])
def test_missing_or_reordered_action_sequence_fails(hardware_rule, actions):
    rule, entries = hardware_rule
    entries[0]["options"]["actions"] = actions
    with pytest.raises(ValueError, match="Expected"):
        inspector.verify_rule(rule, entries)


@pytest.mark.parametrize("entries", [None, {}, [None], ["not a rule"], [{"pref": 49300, "options": None}]])
def test_malformed_filter_inventory_fails_closed(hardware_rule, entries):
    rule, _ = hardware_rule
    with pytest.raises(ValueError):
        inspector.verify_rule(rule, entries)


@pytest.mark.parametrize("location,value", [("keys", None), ("actions", None), ("actions", [None])])
def test_malformed_filter_options_fail_closed(hardware_rule, location, value):
    rule, entries = hardware_rule
    entries[0]["options"][location] = value
    with pytest.raises(ValueError):
        inspector.verify_rule(rule, entries)


@pytest.mark.parametrize("mutation", ["src_mac", "dst_mac", "stats", "pedit_keys", "pedit_item"])
def test_malformed_action_or_match_fields_fail_closed(hardware_rule, mutation):
    rule, entries = hardware_rule
    options = entries[0]["options"]
    if mutation in ("src_mac", "dst_mac"):
        options["keys"][mutation] = None
    elif mutation == "stats":
        options["actions"][0]["stats"] = None
    elif mutation == "pedit_keys":
        options["actions"][0]["keys"] = None
    else:
        options["actions"][0]["keys"] = [None]
    with pytest.raises(ValueError):
        inspector.verify_rule(rule, entries)


def _stat(start_ticks=100000):
    fields = ["S"] + ["0"] * 24
    fields[19] = str(start_ticks)
    return "145 (mlx5 (marker) probe) " + " ".join(fields)


def test_lease_uses_linux_starttime_field_and_handles_parentheses():
    assert inspector.remaining_seconds(["probe", "--run-seconds", "7200"], _stat(), 1250.0, 100) == 6950.0


def test_expired_lease_is_negative():
    assert inspector.remaining_seconds(["probe", "--run-seconds", "7200"], _stat(), 8251.0, 100) == -51.0


@pytest.mark.parametrize("argv", [[], ["probe", "--run-seconds"], ["probe", "--run-seconds", "0"],
                                   ["probe", "--run-seconds", "7201"], ["probe", "--run-seconds", "-1"],
                                   ["probe", "--run-seconds", "nan"],
                                   ["probe", "--run-seconds", "7200", "--run-seconds", "3600"]])
def test_lease_lifetime_is_present_unambiguous_and_bounded(argv):
    with pytest.raises(ValueError):
        inspector.remaining_seconds(argv, _stat(), 1250.0, 100)


@pytest.mark.parametrize("uptime,ticks,start", [(float("nan"), 100, 100000), (float("inf"), 100, 100000),
                                              (1250.0, 0, 100000), (1250.0, -100, 100000),
                                              (-1.0, 100, 100000), (1250.0, 100, -1),
                                              (999.0, 100, 100000)])
def test_lease_rejects_impossible_clock_inputs(uptime, ticks, start):
    with pytest.raises(ValueError):
        inspector.remaining_seconds(["probe", "--run-seconds", "7200"], _stat(start), uptime, ticks)


@pytest.mark.parametrize("stat", ["", "145", "145 (probe) S 0", _stat("invalid")])
def test_malformed_proc_stat_fails_closed(stat):
    with pytest.raises(ValueError):
        inspector.remaining_seconds(["probe", "--run-seconds", "7200"], stat, 1250.0, 100)
