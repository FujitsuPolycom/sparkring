"""GPU-free tests for the research-only tiled-prefill capacity selector."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import spark_tp4_backend
import spark_tp4_port_namespace
import spark_tp4_prefill_capacity_pool as pool


class PrefillCapacityPoolSelectorTest(unittest.TestCase):
    def test_research_pool_is_disabled_by_default(self) -> None:
        self.assertFalse(pool.capacity_pool_requested({}))

    def test_qualification_endpoints_share_one_transport_engine(self) -> None:
        expected = {
            40: (40, (12500, 12501)),
            512: (512, (12500, 12501)),
            1024: (1024, (12500, 12501)),
            4096: (4096, (12500, 12501)),
        }

        for query_rows, (capacity_rows, ports) in expected.items():
            with self.subTest(query_rows=query_rows):
                selection = pool.select_capacity_plan(query_rows, {})
                self.assertEqual(selection.query_rows, query_rows)
                self.assertEqual(
                    selection.capacity_plan.maximum_query_rows,
                    capacity_rows,
                )
                self.assertEqual(selection.transport_key.control_ports, ports)
                self.assertEqual(
                    selection.transport_key.engine_id,
                    "prefill-tile-engine",
                )
                self.assertEqual(selection.active_bytes, query_rows * 12288)
                self.assertEqual(
                    selection.capacity_plan.maximum_payload_bytes,
                    capacity_rows * 12288,
                )

    def test_audit_quantifies_exact_q_session_and_port_explosion(self) -> None:
        audit = pool.transport_engine_audit({})

        self.assertEqual(
            [
                (
                    row["maximum_query_rows"],
                    row["session_count_per_rank"],
                    row["control_port_count_per_rank"],
                    row["last_control_ports"],
                )
                for row in audit["exact_q_projection"]
            ],
            [
                (40, 40, 80, (11078, 11079)),
                (512, 512, 1024, (12022, 12023)),
                (1024, 1024, 2048, (13046, 13047)),
                (4096, 4096, 8192, (19190, 19191)),
            ],
        )
        self.assertEqual(
            audit["capacity_pool"]["physical_transport_engines_per_rank"],
            1,
        )
        self.assertEqual(
            audit["capacity_pool"]["logical_capacity_plan_count"], 4
        )
        self.assertEqual(audit["capacity_pool"]["control_port_count_per_rank"], 2)
        self.assertTrue(
            audit["capacity_pool"]["ports_disjoint_from_exact_decode_q1_q40"]
        )
        self.assertEqual(
            audit["q4096_physical_engine_reduction_factor"], 4096.0
        )

    def test_executable_cache_and_namespace_are_exact_payload_indexed(self) -> None:
        backend = spark_tp4_backend._Backend(0)
        with patch.object(
            spark_tp4_backend,
            "_NativeSession",
            side_effect=lambda rank, payload_bytes: (rank, payload_bytes),
        ) as constructor:
            q1 = backend.native_for(1 * pool.BYTES_PER_QUERY_ROW)
            q40 = backend.native_for(40 * pool.BYTES_PER_QUERY_ROW)
            q1_again = backend.native_for(1 * pool.BYTES_PER_QUERY_ROW)

        self.assertIs(q1, q1_again)
        self.assertNotEqual(q1, q40)
        self.assertEqual(constructor.call_count, 2)
        self.assertEqual(len(backend.native_sessions), 2)

        reservations = spark_tp4_port_namespace.active_port_reservations(
            {
                "VLLM_SPARK_TP4_MODE": "custom",
                "VLLM_SPARK_TP4_PREFILL_Q512": "1",
            }
        )
        exact_q = [
            reservation
            for reservation in reservations
            if reservation.owner.startswith("eager_allreduce:payload=")
        ]
        self.assertEqual(len(exact_q), 512)
        self.assertEqual(exact_q[0].ports, (11000, 11001))
        self.assertEqual(exact_q[-1].ports, (12022, 12023))

    def test_qualification_plan_is_blocked_on_exact_live_prerequisites(self) -> None:
        plan = pool.qualification_plan({})

        self.assertEqual(plan["status"], "research-only")
        self.assertFalse(plan["runnable"])
        self.assertEqual(
            [arm["query_rows"] for arm in plan["fixed_q_arms"]],
            [40, 512, 1024, 4096],
        )
        self.assertEqual(
            [arm["iterations"] for arm in plan["fixed_q_arms"]],
            [200, 50, 25, 10],
        )
        self.assertEqual(
            [item["id"] for item in plan["live_prerequisites"]],
            [
                "native-active-bytes-submission",
                "generation-tagged-tile-pool",
                "cumulative-credit-retirement",
                "capacity-port-namespace",
                "q4096-kernel-capacity",
                "four-rank-fixed-q-probe",
                "serving-engine-receipt",
            ],
        )
        self.assertTrue(
            all(
                prerequisite["satisfied"] is False
                for prerequisite in plan["live_prerequisites"]
            )
        )
        self.assertIn(513, plan["boundary_query_rows"])
        self.assertIn(1025, plan["boundary_query_rows"])
        self.assertEqual(
            plan["required_live_receipt"]["per_rank_topology"],
            {
                "physical_transport_engine_count": 1,
                "edge_count": 2,
                "qp_count_per_edge": 1,
                "registered_tile_pool_count_per_edge": 1,
                "registered_tile_storage_bytes_per_edge": 8_389_632,
                "logical_one_plane_payload_bytes_per_edge": 4 * 1024 * 1024,
                "descriptor_storage_accounting": "separate",
                "logical_capacity_plan_count": 4,
                "exact_q_prefill_session_count": 0,
            },
        )
        self.assertEqual(
            plan["required_live_receipt"]["fatal_counters"],
            {
                "unexpected_generation": 0,
                "poisoned_slot": 0,
                "descriptor_overflow": 0,
                "credit_regression": 0,
            },
        )

    def test_adapter_rejects_research_enable_before_loading_native_code(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "VLLM_SPARK_TP4_MODE": "custom",
                    pool.FEATURE_ENV: "1",
                },
                clear=True,
            ),
            patch.object(spark_tp4_backend, "_installed", False),
            patch.object(spark_tp4_backend.ctypes, "CDLL") as load_library,
            self.assertRaisesRegex(
                RuntimeError,
                "active-bytes native ABI is unavailable",
            ),
        ):
            spark_tp4_backend.install()

        load_library.assert_not_called()

    def test_boundaries_route_to_smallest_class_and_require_active_bytes(self) -> None:
        expected = {
            1: (40, True),
            39: (40, True),
            40: (40, False),
            41: (512, True),
            511: (512, True),
            512: (512, False),
            513: (1024, True),
            1024: (1024, False),
            1025: (4096, True),
            4096: (4096, False),
        }

        for query_rows, (capacity_rows, requires_active_bytes) in expected.items():
            with self.subTest(query_rows=query_rows):
                selection = pool.select_capacity_plan(query_rows, {})
                self.assertEqual(
                    selection.capacity_plan.maximum_query_rows,
                    capacity_rows,
                )
                self.assertEqual(
                    selection.requires_active_bytes_submission,
                    requires_active_bytes,
                )

    def test_python_and_cpp_capacity_maxima_are_identical(self) -> None:
        header = (
            Path(__file__).resolve().parents[2]
            / "include"
            / "spark_transport"
            / "tp4_tiled_session.hpp"
        ).read_text(encoding="utf-8")
        maxima = tuple(
            int(value)
            for value in re.findall(
                r"kTp4Tiled(?:Latency|Medium|Large|Streaming)MaximumQ = (\d+);",
                header,
            )
        )

        self.assertEqual(maxima, pool.CAPACITY_QUERY_ROWS)

    def test_port_audit_proves_pool_replaces_large_exact_q_namespace(self) -> None:
        audit = pool.transport_engine_audit(
            {
                "SPARK_TP4_CONTROL_PORT0": "11100",
                "SPARK_TP4_CONTROL_PORT1": "11101",
            }
        )

        self.assertEqual(
            audit["exact_q_projection"][1]["last_control_ports"],
            (12122, 12123),
        )
        self.assertEqual(
            audit["first_projected_exact_q_collision_with_pool"],
            701,
        )
        self.assertIn(
            "replace exact-Q sessions above Q40",
            audit["capacity_pool"]["coexistence_contract"],
        )

    def test_physical_tile_pool_is_bounded_independently_of_logical_q4096(self) -> None:
        plan = pool.qualification_plan({})
        q4096 = plan["fixed_q_arms"][-1]

        self.assertEqual(plan["tile_pool"]["tile_bytes"], 512 * 1024)
        self.assertEqual(plan["tile_pool"]["tiles_per_edge"], 8)
        self.assertEqual(plan["tile_pool"]["lanes_per_edge"], 2)
        self.assertEqual(plan["tile_pool"]["lane_payload_bytes"], 256 * 1024)
        self.assertEqual(plan["tile_pool"]["control_bytes_per_lane"], 64)
        self.assertEqual(
            plan["tile_pool"]["logical_one_plane_payload_bytes_per_edge"],
            4 * 1024 * 1024,
        )
        self.assertEqual(
            plan["tile_pool"]["registered_bytes_per_edge"],
            8_389_632,
        )
        self.assertFalse(plan["tile_pool"]["descriptors_included"])
        topology = plan["required_live_receipt"]["per_rank_topology"]
        self.assertEqual(
            topology["registered_tile_storage_bytes_per_edge"],
            8_389_632,
        )
        self.assertEqual(
            topology["logical_one_plane_payload_bytes_per_edge"],
            4 * 1024 * 1024,
        )
        self.assertEqual(topology["descriptor_storage_accounting"], "separate")
        self.assertEqual(q4096["maximum_payload_bytes"], 48 * 1024 * 1024)
        self.assertEqual(q4096["logical_payload_tiles"], 96)

    def test_invalid_feature_values_shapes_and_ports_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, pool.FEATURE_ENV):
            pool.capacity_pool_requested({pool.FEATURE_ENV: "yes"})
        for query_rows in (True, 0, -1, 4097, 1.0, "40"):
            with self.subTest(query_rows=query_rows):
                with self.assertRaisesRegex(ValueError, "query rows"):
                    pool.select_capacity_plan(query_rows, {})  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "control ports"):
            pool.shared_transport_key(
                {
                    pool.PORT0_ENV: "65536",
                    pool.PORT1_ENV: "65534",
                },
            )


if __name__ == "__main__":
    unittest.main()
