"""Sync the vendored runtime modules from spark_transport/integrations/vllm.

The wheel ships byte-identical copies of the flat runtime modules so the
adapter keeps its proven bare-name import layout without touching source.
Run from anywhere inside the repository:

    python sparkring_plugin/scripts/sync_vendor.py

tests/test_vendor_parity.py fails whenever the vendored copies drift from
the source tree, so forgetting to re-run this script cannot ship silently.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

# Transitive import closure of spark_tp4_backend, excluding
# spark_tp4_sparse_q42_q48_contract. That module declares which query rows
# the provider serves, belongs to the deployment runtime, and has no source
# in this repository, so the wheel cannot ship it: the deployment supplies it
# on sys.path. The preflight's runtime-contract check names it when absent.
MODULES = (
    "spark_collective_audit",
    "spark_cudagraph_replay_timing",
    "spark_graph_status_reporter",
    "spark_persistent_output_ring",
    "spark_q2r_probe_bridge",
    "spark_tp4_allgather_backend",
    "spark_tp4_backend",
    "spark_tp4_dcp_backend",
    "spark_tp4_flight_recorder",
    "spark_tp4_port_namespace",
    "spark_tp4_prefill_capacity_pool",
    "spark_tp4_query_contract",
    "spark_tp4_stock_timing",
    "spark_tp4_vocab_allgather_backend",
    "spark_true_adaptive_draft",
)


def main() -> int:
    plugin_root = pathlib.Path(__file__).resolve().parent.parent
    repo_root = plugin_root.parent
    source = repo_root / "spark_transport" / "integrations" / "vllm"
    vendor = plugin_root / "src" / "sparkring" / "_vendor"
    if not source.is_dir():
        print(f"source tree not found: {source}", file=sys.stderr)
        return 1
    vendor.mkdir(parents=True, exist_ok=True)
    for name in MODULES:
        module = source / f"{name}.py"
        if not module.is_file():
            print(f"missing source module: {module}", file=sys.stderr)
            return 1
        shutil.copyfile(module, vendor / f"{name}.py")
        print(f"synced {name}.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
