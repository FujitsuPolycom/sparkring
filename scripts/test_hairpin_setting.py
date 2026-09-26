"""The ConnectX hairpin in-effect rule, checked on recorded devlink output; no host is contacted."""

from __future__ import annotations

import copy
import json

import pytest

from scripts import deploy_network, hairpin_setting
from scripts.hairpin_setting import (
    DEFAULT,
    FAILED,
    IN_EFFECT,
    OFFLOAD_DISABLED,
    OFFLOAD_FIXED,
    OFFLOAD_OFF,
    OFFLOAD_ON,
    PARAMETERS,
    PENDING,
    UNKNOWN,
    driverinit_value,
    ethtool_offload,
    evaluate,
    function_state,
    grouped,
    offload_setting,
    reload_statistics,
    shortfall,
)

# Recorded with iproute2 6.1 on a four-Spark ring function that one live
# driver_reinit had moved to 8192/4 (read without sudo). Only the PCI address
# and counters identify the host.
DEVICE = "pci/0000:01:00.0"
RELOAD_AFTER_ONE_RESTART = (
    '{"dev":{"pci/0000:01:00.0":{"stats":{"reload":{"driver_reinit":{"unspecified":1},'
    '"fw_activate":{"unspecified":0,"no_reset":0}},"remote_reload":{"driver_reinit":'
    '{"unspecified":0},"fw_activate":{"unspecified":0,"no_reset":0}}}}}}'
)
QUEUE_SIZE_8192 = (
    '{"param":{"pci/0000:01:00.0":[{"name":"hairpin_queue_size","type":"driver-specific",'
    '"values":[{"cmode":"driverinit","value":8192}]}]}}'
)
NUM_QUEUES_4 = (
    '{"param":{"pci/0000:01:00.0":[{"name":"hairpin_num_queues","type":"driver-specific",'
    '"values":[{"cmode":"driverinit","value":4}]}]}}'
)
# Recorded on a two-Spark pair function at the driver's probe default.
QUEUE_SIZE_PROBE_DEFAULT = (
    '{"param":{"pci/0000:01:00.0":[{"name":"hairpin_queue_size","type":"driver-specific",'
    '"values":[{"cmode":"driverinit","value":1024}]}]}}'
)


def recorded_statistics():
    return json.loads(RELOAD_AFTER_ONE_RESTART)


def recorded_values(queue_size=QUEUE_SIZE_8192):
    return {
        "hairpin_queue_size": driverinit_value(
            json.loads(queue_size), DEVICE, "hairpin_queue_size"
        ),
        "hairpin_num_queues": driverinit_value(
            json.loads(NUM_QUEUES_4), DEVICE, "hairpin_num_queues"
        ),
    }


def state(values=None, statistics=None, offload=OFFLOAD_ON):
    values = recorded_values() if values is None else values
    document = recorded_statistics() if statistics is None else statistics
    return function_state(values, reload_statistics(document, DEVICE), offload)


def test_recorded_output_parses_to_values_and_counter():
    assert recorded_values() == {"hairpin_queue_size": 8192, "hairpin_num_queues": 4}
    assert driverinit_value(json.loads(QUEUE_SIZE_PROBE_DEFAULT), DEVICE, "hairpin_queue_size") == 1024
    assert reload_statistics(recorded_statistics(), DEVICE) == {"driver_reinit": 1, "failed": False}


def test_required_values_are_shared_with_the_planner():
    assert hairpin_setting.PARAMETERS == {"hairpin_num_queues": 4, "hairpin_queue_size": 8192}
    assert deploy_network.HAIRPIN_QUEUE_SIZE is hairpin_setting.HAIRPIN_QUEUE_SIZE


def test_values_in_use_after_one_restart_are_in_effect():
    assert state() == IN_EFFECT


def test_probe_default_needs_param_set_and_restart():
    assert state(recorded_values(QUEUE_SIZE_PROBE_DEFAULT)) == DEFAULT
    assert state({**recorded_values(), "hairpin_num_queues": 2}) == DEFAULT


def test_values_shown_without_a_restart_since_probe_are_pending():
    statistics = recorded_statistics()
    statistics["dev"][DEVICE]["stats"]["reload"]["driver_reinit"]["unspecified"] = 0
    assert state(statistics=statistics) == PENDING


def test_reload_failed_needs_a_restart():
    statistics = recorded_statistics()
    statistics["dev"][DEVICE]["reload_failed"] = True
    assert reload_statistics(statistics, DEVICE)["failed"] is True
    assert state(statistics=statistics) == FAILED
    # Differing values still need param set, which a plain restart would skip.
    assert state(recorded_values(QUEUE_SIZE_PROBE_DEFAULT), statistics) == DEFAULT


@pytest.mark.parametrize("level", ["dev", "device", "stats", "reload", "driver_reinit", "unspecified"])
def test_any_missing_statistics_level_is_unknown(level):
    statistics = recorded_statistics()
    entry = statistics["dev"][DEVICE]
    if level == "dev":
        del statistics["dev"]
    elif level == "device":
        del statistics["dev"][DEVICE]
    elif level == "stats":
        del entry["stats"]
    elif level == "reload":
        del entry["stats"]["reload"]
    elif level == "driver_reinit":
        del entry["stats"]["reload"]["driver_reinit"]
    else:
        del entry["stats"]["reload"]["driver_reinit"]["unspecified"]
    assert reload_statistics(statistics, DEVICE)["driver_reinit"] is None
    assert state(statistics=statistics) == UNKNOWN
    # A probe default does not turn an unreadable counter into a restart.
    assert state(recorded_values(QUEUE_SIZE_PROBE_DEFAULT), statistics) == UNKNOWN


def test_missing_device_entry_leaves_failure_flag_unknown():
    assert reload_statistics({"dev": {}}, DEVICE) == {"driver_reinit": None, "failed": None}
    assert reload_statistics(None, DEVICE) == {"driver_reinit": None, "failed": None}
    assert reload_statistics(recorded_statistics(), "pci/0000:01:00.1") == {
        "driver_reinit": None,
        "failed": None,
    }


@pytest.mark.parametrize("value", ["1", 1.0, True, -1, None])
def test_counter_must_be_a_json_integer(value):
    statistics = recorded_statistics()
    statistics["dev"][DEVICE]["stats"]["reload"]["driver_reinit"]["unspecified"] = value
    assert reload_statistics(statistics, DEVICE)["driver_reinit"] is None
    assert state(statistics=statistics) == UNKNOWN


def test_non_boolean_failure_flag_is_unknown():
    statistics = recorded_statistics()
    statistics["dev"][DEVICE]["reload_failed"] = 1
    assert reload_statistics(statistics, DEVICE) == {"driver_reinit": 1, "failed": None}
    assert state(statistics=statistics) == UNKNOWN


@pytest.mark.parametrize("remote", [0, 5])
def test_remote_reload_counter_has_no_effect(remote):
    statistics = recorded_statistics()
    statistics["dev"][DEVICE]["stats"]["remote_reload"]["driver_reinit"]["unspecified"] = remote
    assert state(statistics=statistics) == IN_EFFECT
    statistics["dev"][DEVICE]["stats"]["reload"]["driver_reinit"]["unspecified"] = 0
    assert state(statistics=statistics) == PENDING
    del statistics["dev"][DEVICE]["stats"]["remote_reload"]
    assert state(statistics=statistics) == PENDING


def test_offload_off_and_toggleable_needs_only_ethtool():
    assert state(offload=OFFLOAD_DISABLED) == OFFLOAD_OFF
    # Fixed-off offload is also not in effect; its remedy cannot succeed.
    assert state(offload=OFFLOAD_FIXED) == OFFLOAD_OFF
    # A restart need is reported ahead of offload, which a restart recreates.
    assert state(recorded_values(QUEUE_SIZE_PROBE_DEFAULT), offload=OFFLOAD_DISABLED) == DEFAULT


@pytest.mark.parametrize(
    "values,offload",
    [
        ({"hairpin_queue_size": None, "hairpin_num_queues": 4}, OFFLOAD_ON),
        ({"hairpin_queue_size": 8192}, OFFLOAD_ON),
        (None, None),
        (None, "enabled"),
    ],
)
def test_unreadable_values_or_offload_are_unknown(values, offload):
    assert state(values, offload=offload) == UNKNOWN


def test_driverinit_value_reads_only_the_driverinit_mode():
    document = json.loads(QUEUE_SIZE_8192)
    assert driverinit_value(document, DEVICE, "hairpin_num_queues") is None
    assert driverinit_value(document, "pci/0000:01:00.1", "hairpin_queue_size") is None
    runtime = copy.deepcopy(document)
    runtime["param"][DEVICE][0]["values"][0]["cmode"] = "runtime"
    assert driverinit_value(runtime, DEVICE, "hairpin_queue_size") is None
    text = copy.deepcopy(document)
    text["param"][DEVICE][0]["values"][0]["value"] = "8192"
    assert driverinit_value(text, DEVICE, "hairpin_queue_size") == 8192
    assert driverinit_value({"param": []}, DEVICE, "hairpin_queue_size") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Features for enp1s0f0np0:\nhw-tc-offload: on\n", OFFLOAD_ON),
        ("hw-tc-offload: on [fixed]", OFFLOAD_ON),
        ("hw-tc-offload: off", OFFLOAD_DISABLED),
        ("hw-tc-offload: off [fixed]", OFFLOAD_FIXED),
        ("hw-tc-offload: off [requested on]", OFFLOAD_DISABLED),
        ("rx-checksumming: on\n", None),
        ("hw-tc-offload: unknown", None),
        (None, None),
    ],
)
def test_ethtool_offload_parsing(text, expected):
    assert ethtool_offload(text) == expected


def test_inventory_offload_booleans_map_to_ethtool_values():
    assert offload_setting(True, False) == OFFLOAD_ON
    assert offload_setting(True, None) == OFFLOAD_ON
    assert offload_setting(False, False) == OFFLOAD_DISABLED
    assert offload_setting(False, True) == OFFLOAD_FIXED
    assert offload_setting(False, None) is None
    assert offload_setting(None, False) is None


def test_evaluate_row_and_shortfall_text():
    row = evaluate(recorded_values(), reload_statistics(recorded_statistics(), DEVICE), OFFLOAD_ON)
    assert row == {
        "values": {"hairpin_num_queues": 4, "hairpin_queue_size": 8192},
        "driver_reinit": 1,
        "reload_failed": False,
        "offload": OFFLOAD_ON,
        "state": IN_EFFECT,
    }
    fresh = evaluate(
        recorded_values(QUEUE_SIZE_PROBE_DEFAULT),
        {"driver_reinit": 0, "failed": False},
        OFFLOAD_ON,
    )
    assert fresh["state"] == DEFAULT
    assert shortfall(fresh) == "hairpin_queue_size 1024, required 8192"
    pending = evaluate(recorded_values(), {"driver_reinit": 0, "failed": False}, OFFLOAD_ON)
    assert pending["state"] == PENDING
    assert shortfall(pending) == "hairpin_queue_size 8192 set, applied only by a driver restart (none since boot)"
    offload = evaluate(recorded_values(), {"driver_reinit": 1, "failed": False}, OFFLOAD_DISABLED)
    assert shortfall(offload) == "hw-tc-offload off"
    assert shortfall(evaluate(recorded_values(), {"driver_reinit": 2, "failed": True}, OFFLOAD_ON)) == (
        "last driver restart failed")
    broken = evaluate({"hairpin_queue_size": 8192, "hairpin_num_queues": 2}, {"driver_reinit": 3, "failed": True}, None)
    assert broken["state"] == UNKNOWN
    assert shortfall(broken) == (
        "hairpin_num_queues 2, required 4, last driver restart failed, hw-tc-offload unavailable"
    )
    assert shortfall(evaluate({}, {}, OFFLOAD_ON)) == (
        "hairpin_num_queues unavailable, hairpin_queue_size unavailable, "
        "driver restarts since boot unavailable, restart failure flag unavailable"
    )
    assert shortfall(row) == ""


def test_grouped_names_functions_that_share_a_shortfall_once():
    fresh = {"values": {"hairpin_num_queues": 4, "hairpin_queue_size": 1024}, "driver_reinit": 0,
             "reload_failed": False, "offload": OFFLOAD_ON}
    failed = {**fresh, "values": dict(PARAMETERS), "driver_reinit": 1, "reload_failed": True}
    rows = [{**fresh, "netdev": "a"}, {**failed, "netdev": "b"}, {**fresh, "netdev": "c"}]
    assert grouped(rows) == "a, c: hairpin_queue_size 1024, required 8192; b: last driver restart failed"
    assert grouped(rows, name=lambda r: r["netdev"].upper(), describe=lambda r: "x") == "A, B, C: x"
