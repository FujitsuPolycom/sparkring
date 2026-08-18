"""Subprocess harness printing one adapter lineage's admission surface.

Usage: python _provider_equivalence_harness.py BACKEND_PY NAMESPACE_PY

Loads the given spark_tp4_backend / spark_tp4_port_namespace file pair
under the CURRENT process environment and prints a canonical JSON
description of every admission-relevant surface: the eager all-reduce
reservation tuple (the arena set an exact-state invariant counts),
per-row control-port resolution, shape admission, and capacity values.

A fresh subprocess per environment permutation is required because
spark_tp4_query_contract bakes VLLM_SPARK_MAX_QUERY_ROWS at import.
The directory `_private_fixtures/` (deployment-lineage modules that are
not publishable) is appended to sys.path as a fallback resolver for
`spark_tp4_sparse_q42_q48_contract`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys


def _load(name: str, path: str):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def main() -> int:
    backend_path, namespace_path = sys.argv[1], sys.argv[2]
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    fixtures = os.path.join(here, "_private_fixtures")
    if os.path.isdir(fixtures):
        sys.path.append(fixtures)

    namespace = _load("spark_tp4_port_namespace", namespace_path)
    backend = _load("spark_tp4_backend_under_test", backend_path)

    environ = dict(os.environ)
    surface: dict = {}

    reservations = namespace.validate_active_port_namespace(environ)
    surface["reservations"] = [
        [reservation.owner, list(reservation.ports)]
        for reservation in reservations
    ]
    surface["arena_count"] = len(reservations)

    ports: dict = {}
    for rows in range(1, 521):
        try:
            ports[str(rows)] = list(
                namespace.eager_allreduce_control_ports(rows, environ)
            )
        except ValueError:
            ports[str(rows)] = "rejected"
    surface["row_ports"] = ports

    admission: dict = {}
    for width in (2880, 4096, 6144):
        bits = []
        for rows in range(1, 521):
            bits.append(
                1 if backend._target_shape_eligible((rows, width)) else 0
            )
        admission[str(width)] = "".join(str(bit) for bit in bits)
    surface["target_shape_admission"] = admission

    if hasattr(backend, "_eager_shape_eligible"):
        eager: dict = {}
        for width in (2880, 4096, 6144):
            bits = []
            for rows in range(1, 521):
                bits.append(
                    1 if backend._eager_shape_eligible((rows, width)) else 0
                )
            eager[str(width)] = "".join(str(bit) for bit in bits)
        surface["eager_shape_admission"] = eager

    surface["maximum_allreduce_query_rows"] = (
        backend._maximum_allreduce_query_rows()
    )
    surface["graph_capacity_bytes"] = backend._graph_capacity_bytes()

    control_ports: dict = {}
    for rows in (1, 2, 40, 41, 42, 48, 100, 512):
        payload = rows * 12288
        try:
            control_ports[str(payload)] = list(
                backend._control_ports(payload)
            )
        except ValueError:
            control_ports[str(payload)] = "rejected"
    surface["backend_control_ports"] = control_ports

    print(json.dumps(surface, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
