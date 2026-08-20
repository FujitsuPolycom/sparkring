"""Sync the vendored runtime modules from spark_transport/integrations/vllm.

The wheel ships byte-identical copies of the flat runtime modules so the
adapter keeps its proven bare-name import layout without touching source.
Run from anywhere inside the repository:

    python sparkring_plugin/scripts/sync_vendor.py

Copying also rewrites src/sparkring/_vendor/MANIFEST.json, which records
{"modules": {"<name>.py": "<sha256>"}} over the bytes written into the
vendor directory. A pip-installed host has no source tree to compare
against, so the manifest is what lets that host re-hash the modules it
received and show they are the bytes this script produced.

tests/test_vendor_parity.py fails whenever the vendored copies drift from
the source tree, so forgetting to re-run this script cannot ship silently.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

MANIFEST = "MANIFEST.json"

# Transitive import closure of spark_tp4_backend, and the set MANIFEST.json
# describes. Four import targets sit deliberately outside it, so the manifest
# covers what the wheel ships rather than everything an enabled mode reaches:
#
#   - The external row-policy module a deployment may name in
#     VLLM_SPARK_TP4_QUERY_ROW_PROVIDER. It is imported lazily at resolution
#     time; the wheel is complete without any provider, and the preflight
#     reports a configured provider's importability as its own check.
#   - spark_transport/experiments/adaptive_mtp_controller, imported by
#     spark_graph_status_reporter under SPARK_ADAPTIVE_MTP_CONTROL=1.
#   - spark_transport/experiments/moe_round_floor and
#     spark_transport/experiments/q2r_phase_timing, imported by
#     spark_q2r_probe_bridge under SPARK_Q2R_PROBE=1.
#
# Those three experiment packages must reach sys.path from outside the wheel
# whenever their gating variables are set.
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
    "spark_tp4_query_row_provider",
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
    digests: dict[str, str] = {}
    for name in MODULES:
        module = source / f"{name}.py"
        if not module.is_file():
            print(f"missing source module: {module}", file=sys.stderr)
            return 1
        copy = vendor / f"{name}.py"
        shutil.copyfile(module, copy)
        # Hash the destination, not the source: the manifest is a claim
        # about the bytes the wheel carries.
        digests[copy.name] = hashlib.sha256(copy.read_bytes()).hexdigest()
        print(f"synced {name}.py")
    document = json.dumps({"modules": digests}, sort_keys=True, indent=2)
    # write_bytes, not write_text: the serialization must come out identical
    # on every host, and text mode rewrites the line ending on Windows.
    (vendor / MANIFEST).write_bytes((document + "\n").encode("utf-8"))
    print(f"wrote {MANIFEST} over {len(digests)} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
