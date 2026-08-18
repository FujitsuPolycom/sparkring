"""Equivalence tests against the deployment adapter lineage.

The deployment runtime carries an adapter lineage whose eager all-reduce
admission is governed by the sparse Q42/Q48 provider contract, and whose
exact-state invariant counts the eager reservation set (arenas) at
startup, refusing to serve on any drift. These tests pin the worktree
adapter pair to that lineage: under every environment permutation with
`VLLM_SPARK_TP4_EAGER_WIDTHS` unset, the complete admission surface —
reservation tuple, port resolution, shape admission, capacity values —
must be byte-identical to the deployment pair's. With widths set, the
deployment surface must be a strict prefix and every extension entry a
payload-owner reservation.

Requires `_private_fixtures/` (deployment-lineage copies fetched from the
cluster; unpublishable, therefore untracked): `deployed_backend.py`,
`deployed_namespace.py`, `spark_tp4_sparse_q42_q48_contract.py`. Tests
skip loudly when the fixtures are absent.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE / "_private_fixtures"
HARNESS = HERE / "_provider_equivalence_harness.py"
GOLDEN_BACKEND = FIXTURES / "deployed_backend.py"
GOLDEN_NAMESPACE = FIXTURES / "deployed_namespace.py"
WORKTREE_BACKEND = HERE / "spark_tp4_backend.py"
WORKTREE_NAMESPACE = HERE / "spark_tp4_port_namespace.py"

_BASE_ENV = {
    "SPARK_TP4_CONTROL_PORT0": "11100",
    "SPARK_TP4_CONTROL_PORT1": "11101",
    "VLLM_SPARK_TP4_MODE": "custom",
}

# (label, extra environment). Sparse requires MAX_QUERY_ROWS=40 and
# excludes Q512 (enforced by the provider contract module itself).
_WIDTHS_UNSET_PERMUTATIONS = [
    ("production_replay_sparse", {
        "VLLM_SPARK_TP4_SPARSE_Q42_Q48": "1",
        "VLLM_SPARK_MAX_QUERY_ROWS": "40",
    }),
    ("contiguous_q40", {
        "VLLM_SPARK_MAX_QUERY_ROWS": "40",
    }),
    ("contiguous_q512", {
        "VLLM_SPARK_MAX_QUERY_ROWS": "40",
        "VLLM_SPARK_TP4_PREFILL_Q512": "1",
    }),
    ("shadow_sparse", {
        "VLLM_SPARK_TP4_MODE": "shadow",
        "VLLM_SPARK_TP4_SPARSE_Q42_Q48": "1",
        "VLLM_SPARK_MAX_QUERY_ROWS": "40",
    }),
]

_WIDTHS_SET_PERMUTATIONS = [
    ("sparse_with_width_2880", {
        "VLLM_SPARK_TP4_SPARSE_Q42_Q48": "1",
        "VLLM_SPARK_MAX_QUERY_ROWS": "40",
        "VLLM_SPARK_TP4_EAGER_WIDTHS": "2880",
    }),
    ("contiguous_with_widths", {
        "VLLM_SPARK_MAX_QUERY_ROWS": "40",
        "VLLM_SPARK_TP4_EAGER_WIDTHS": "2880,4096",
    }),
]


def _surface(backend: pathlib.Path, namespace: pathlib.Path,
             extra_env: dict) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("VLLM_SPARK", "SPARK_TP4", "SPARK_Q42"))}
    env.update(_BASE_ENV)
    env.update(extra_env)
    completed = subprocess.run(
        [sys.executable, str(HARNESS), str(backend), str(namespace)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"harness failed for {backend.name}: {completed.stderr[-800:]}"
        )
    return json.loads(completed.stdout)


@unittest.skipUnless(
    GOLDEN_BACKEND.is_file() and GOLDEN_NAMESPACE.is_file(),
    "deployment-lineage fixtures absent; fetch them into "
    "_private_fixtures/ from the cluster runtime deployment",
)
class ProviderRowsEquivalenceTest(unittest.TestCase):
    def test_widths_unset_surfaces_are_identical(self) -> None:
        for label, extra in _WIDTHS_UNSET_PERMUTATIONS:
            with self.subTest(permutation=label):
                golden = _surface(GOLDEN_BACKEND, GOLDEN_NAMESPACE, extra)
                candidate = _surface(
                    WORKTREE_BACKEND, WORKTREE_NAMESPACE, extra
                )
                # eager_shape_admission exists only in the candidate;
                # with widths unset it must equal target admission for
                # the default width and reject every other width.
                eager = candidate.pop("eager_shape_admission", None)
                golden.pop("eager_shape_admission", None)
                self.assertEqual(golden, candidate)
                if eager is not None:
                    self.assertEqual(
                        eager["6144"],
                        candidate["target_shape_admission"]["6144"],
                    )
                    self.assertEqual(set(eager["2880"]), {"0"})
                    self.assertEqual(set(eager["4096"]), {"0"})

    def test_widths_set_deployment_surface_is_a_strict_prefix(self) -> None:
        for label, extra in _WIDTHS_SET_PERMUTATIONS:
            with self.subTest(permutation=label):
                unset_extra = {
                    k: v for k, v in extra.items()
                    if k != "VLLM_SPARK_TP4_EAGER_WIDTHS"
                }
                golden = _surface(
                    GOLDEN_BACKEND, GOLDEN_NAMESPACE, unset_extra
                )
                candidate = _surface(
                    WORKTREE_BACKEND, WORKTREE_NAMESPACE, extra
                )
                golden_reservations = golden["reservations"]
                candidate_reservations = candidate["reservations"]
                self.assertEqual(
                    candidate_reservations[: len(golden_reservations)],
                    golden_reservations,
                )
                extensions = candidate_reservations[
                    len(golden_reservations):
                ]
                self.assertTrue(extensions, "widths must add reservations")
                for owner, ports in extensions:
                    self.assertTrue(
                        owner.startswith("eager_allreduce:payload="),
                        owner,
                    )
                    # Extension ports begin past the Q512 legacy span.
                    self.assertGreaterEqual(ports[0], 11100 + 2 * 512)
                self.assertEqual(
                    golden["row_ports"], candidate["row_ports"]
                )


if __name__ == "__main__":
    unittest.main()
