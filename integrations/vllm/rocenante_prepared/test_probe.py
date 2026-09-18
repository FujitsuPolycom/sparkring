"""GPU-free validation of the adaptive proxy counter ABI used by the probe."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepared_roce_probe", Path(__file__).with_name("probe.py")
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def counters(hcas, increment=0):
    return [
        dict(
            peer_rank=1,
            path_index=index,
            local_hca_index=hca,
            payload_bytes=100 + increment + index,
            completion_errors=0,
        )
        for index, hca in enumerate(hcas)
    ]


def test_two_paths_in_four_function_inventory():
    assert PROBE.path_payload_deltas(counters([0, 2]), counters([0, 2], 4096), 4) == [
        4096,
        0,
        4096,
        0,
    ]


def test_multiple_peer_paths_aggregate_shared_hcas():
    before = counters([0, 2]) + [{**row, "peer_rank": 2} for row in counters([0, 3])]
    after = [{**row, "payload_bytes": row["payload_bytes"] + 512} for row in before]
    assert PROBE.path_payload_deltas(before, after, 4) == [1024, 0, 512, 512]


@pytest.mark.parametrize(
    "defect",
    ["duplicate", "missing", "hca", "changed-hca", "error", "negative", "no-traffic"],
)
def test_invalid_counter_evidence_is_rejected(defect):
    before, after = counters([0, 2]), counters([0, 2], 100)
    if defect == "duplicate":
        after.append(after[0].copy())
    elif defect == "missing":
        after.pop()
    elif defect == "hca":
        after[0]["local_hca_index"] = 4
    elif defect == "changed-hca":
        after[0]["local_hca_index"] = 1
    elif defect == "error":
        after[0]["completion_errors"] = 1
    elif defect == "negative":
        after[0]["payload_bytes"] = -1
    else:
        after[0]["payload_bytes"] = before[0]["payload_bytes"]
    with pytest.raises(ValueError):
        PROBE.path_payload_deltas(before, after, 4)
