from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import spark_tp4_backend
import spark_tp4_port_namespace as namespace
import spark_tp4_vocab_allgather_backend


class Tp4PortNamespaceTest(unittest.TestCase):
    def test_all_active_default_families_are_globally_unique(self) -> None:
        environment = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "VLLM_SPARK_TP4_PREFILL_Q512": "1",
            "VLLM_SPARK_TP4_GRAPH_Q1": "1",
            "VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40": "1",
            "VLLM_SPARK_TP4_VOCAB_MODE": "custom",
        }

        reservations = namespace.validate_active_port_namespace(environment)

        self.assertEqual(len(reservations), 516)
        assigned = [
            port
            for reservation in reservations
            for port in reservation.ports
        ]
        self.assertEqual(len(assigned), len(set(assigned)))
        by_owner = {
            reservation.owner: reservation.ports
            for reservation in reservations
        }
        self.assertEqual(
            by_owner["eager_allreduce:q=1"], (11000, 11001)
        )
        self.assertEqual(
            by_owner["eager_allreduce:q=512"],
            (12022, 12023),
        )
        self.assertEqual(by_owner["graph_allreduce"], (9970, 9971))
        self.assertEqual(
            by_owner["graph_dual_port_q40_allreduce"], (9972, 9973)
        )
        self.assertEqual(by_owner["eager_vocab"], (9990, 9991))
        self.assertEqual(by_owner["graph_vocab"], (10110, 10111))


    def test_eager_allreduce_default_and_override_keep_stride_two(self) -> None:
        self.assertEqual(
            namespace.eager_allreduce_control_ports(1, {}),
            (11000, 11001),
        )
        self.assertEqual(
            namespace.eager_allreduce_control_ports(6, {}),
            (11010, 11011),
        )
        override = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "SPARK_TP4_CONTROL_PORT0": "9480",
            "SPARK_TP4_CONTROL_PORT1": "9481",
        }
        self.assertEqual(
            namespace.eager_allreduce_control_ports(6, override),
            (9490, 9491),
        )


    def test_only_selected_families_reserve_ports(self) -> None:
        environment = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "SPARK_TP4_CONTROL_PORT0": "9480",
            "SPARK_TP4_CONTROL_PORT1": "9481",
        }

        reservations = namespace.validate_active_port_namespace(environment)

        self.assertTrue(reservations)
        self.assertTrue(
            all(
                reservation.owner.startswith("eager_allreduce:")
                for reservation in reservations
            )
        )

    def test_allreduce_override_cannot_overlap_its_next_q_slot(self) -> None:
        environment = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "SPARK_TP4_CONTROL_PORT0": "12000",
            "SPARK_TP4_CONTROL_PORT1": "12002",
        }

        with self.assertRaisesRegex(
            ValueError,
            r"port 12002: eager_allreduce:q=1, eager_allreduce:q=2",
        ):
            namespace.validate_active_port_namespace(environment)

    def test_active_port_range_is_checked_before_use(self) -> None:
        environment = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "VLLM_SPARK_TP4_PREFILL_Q512": "1",
            "SPARK_TP4_CONTROL_PORT0": "65000",
            "SPARK_TP4_CONTROL_PORT1": "65001",
        }

        with self.assertRaisesRegex(ValueError, r"\[1, 65535\]"):
            namespace.validate_active_port_namespace(environment)

    def test_adapter_port_helpers_delegate_to_shared_namespace(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                spark_tp4_backend._control_ports(12288), (11000, 11001)
            )
            self.assertEqual(
                spark_tp4_backend._graph_control_ports(), (9970, 9971)
            )
            self.assertEqual(
                spark_tp4_backend._graph_dual_port_q40_control_ports(),
                (9972, 9973),
            )
            self.assertEqual(
                namespace.bidirectional_prefill_control_ports(1024),
                (19000, 19001),
            )
            self.assertEqual(
                namespace.bidirectional_prefill_control_ports(8192),
                (19006, 19007),
            )
            self.assertEqual(
                namespace.bidirectional_prefill_secondary_control_ports(1024),
                (19100, 19101),
            )
            self.assertEqual(
                spark_tp4_vocab_allgather_backend._eager_control_ports(),
                (9990, 9991),
            )
            self.assertEqual(
                spark_tp4_vocab_allgather_backend._graph_control_ports(),
                (10110, 10111),
            )

    def test_bidirectional_prefill_ports_are_reserved_and_collisions_fail(
        self,
    ) -> None:
        environment = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL": "1",
        }
        reservations = namespace.validate_active_port_namespace(environment)
        by_owner = {
            reservation.owner: reservation.ports
            for reservation in reservations
        }
        self.assertEqual(
            by_owner["bidirectional_prefill:q=1024"], (19000, 19001)
        )
        self.assertEqual(
            by_owner["bidirectional_prefill:q=8192"], (19006, 19007)
        )

        collision = {
            **environment,
            "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0": "11000",
            "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1": "11001",
        }
        with self.assertRaisesRegex(ValueError, "namespaces collide"):
            namespace.validate_active_port_namespace(collision)

        dual = {
            **environment,
            "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE": "dual",
        }
        dual_reservations = namespace.validate_active_port_namespace(dual)
        dual_by_owner = {
            reservation.owner: reservation.ports
            for reservation in dual_reservations
        }
        self.assertEqual(
            dual_by_owner["bidirectional_prefill_secondary:q=1024"],
            (19100, 19101),
        )
        dual_collision = {
            **dual,
            "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0": "19000",
            "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1": "19001",
        }
        with self.assertRaisesRegex(ValueError, "namespaces collide"):
            namespace.validate_active_port_namespace(dual_collision)

    def test_eager_admitted_widths_default_and_empty(self) -> None:
        self.assertEqual(
            namespace.eager_allreduce_admitted_widths({}), (6144,)
        )
        self.assertEqual(
            namespace.eager_allreduce_admitted_widths(
                {"VLLM_SPARK_TP4_EAGER_WIDTHS": ""}
            ),
            (6144,),
        )

    def test_eager_admitted_widths_parsing(self) -> None:
        self.assertEqual(
            namespace.eager_allreduce_admitted_widths(
                {"VLLM_SPARK_TP4_EAGER_WIDTHS": "4096,6144"}
            ),
            (4096, 6144),
        )
        self.assertEqual(
            namespace.eager_allreduce_admitted_widths(
                {"VLLM_SPARK_TP4_EAGER_WIDTHS": " 4096 , 6144 "}
            ),
            (4096, 6144),
        )

    def test_eager_admitted_widths_rejects_invalid(self) -> None:
        env = "VLLM_SPARK_TP4_EAGER_WIDTHS"
        for raw in (
            "0",
            "-4096",
            "abc",
            "4096.5",
            "6144,6144",
            "4096,,6144",
            "2000000",
        ):
            with self.assertRaisesRegex(ValueError, env):
                namespace.eager_allreduce_admitted_widths({env: raw})

    def test_eager_allreduce_legacy_identity(self) -> None:
        for rows in range(1, 7):
            payload = rows * 12288
            self.assertEqual(
                namespace.eager_allreduce_ports_for_payload(payload, {}),
                namespace.eager_allreduce_control_ports(rows, {}),
            )
        self.assertEqual(
            namespace.eager_allreduce_ports_for_payload(12288, {}),
            (11000, 11001),
        )
        self.assertEqual(
            namespace.eager_allreduce_ports_for_payload(73728, {}),
            (11010, 11011),
        )
        prefill = {"VLLM_SPARK_TP4_PREFILL_Q512": "1"}
        self.assertEqual(
            namespace.eager_allreduce_ports_for_payload(
                512 * 12288, prefill
            ),
            (12022, 12023),
        )

    def test_eager_allreduce_multi_width_dedups_payload_sizes(self) -> None:
        environment = {
            "VLLM_SPARK_TP4_EAGER_WIDTHS": "3072,6144",
            "VLLM_SPARK_TP4_PREFILL_Q512": "1",
        }
        sizes = namespace.eager_allreduce_payload_sizes(environment)
        # 2 rows * 3072 * 2 bytes == 1 row * 6144 * 2 bytes == 12288:
        # an identical byte count is one native operation, one entry.
        self.assertEqual(sizes.count(12288), 1)
        # A payload shared with the default width keeps the legacy
        # row-slot ports (row 1 -> slot 0).
        self.assertEqual(
            namespace.eager_allreduce_ports_for_payload(
                12288, environment
            ),
            (11000, 11001),
        )
        # Odd 3072-width row counts have no default-width equivalent and
        # occupy extension slots past the Q512 legacy span, ordered by
        # payload size.
        extension_sizes = sorted(
            rows * 3072 * 2 for rows in range(1, 513) if rows % 2 == 1
        )
        first_extension = extension_sizes[0]
        self.assertEqual(first_extension, 6144)
        self.assertEqual(
            namespace.eager_allreduce_ports_for_payload(
                first_extension, environment
            ),
            (11000 + 2 * 512, 11001 + 2 * 512),
        )

    def test_eager_allreduce_ports_for_payload_rejects_unknown(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"unsupported Spark TP4 eager all-reduce payload size: "
            r"999999 bytes",
        ):
            namespace.eager_allreduce_ports_for_payload(999999, {})

    def test_eager_allreduce_port_ceiling_fails_closed(self) -> None:
        environment = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "VLLM_SPARK_TP4_EAGER_WIDTHS": "4096,6144",
            "SPARK_TP4_CONTROL_PORT0": "65534",
            "SPARK_TP4_CONTROL_PORT1": "65535",
        }
        with self.assertRaisesRegex(ValueError, r"\[1, 65535\]"):
            namespace.validate_active_port_namespace(environment)

    def test_eager_allreduce_multi_width_namespace_unique(self) -> None:
        base = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "VLLM_SPARK_TP4_EAGER_WIDTHS": "4096,6144",
        }
        single_width = {
            **base,
            "VLLM_SPARK_TP4_EAGER_WIDTHS": "6144",
        }
        base_reservations = namespace.validate_active_port_namespace(base)
        single_reservations = namespace.validate_active_port_namespace(
            single_width
        )
        base_ports = [
            port
            for reservation in base_reservations
            for port in reservation.ports
        ]
        self.assertEqual(len(base_ports), len(set(base_ports)))
        self.assertGreater(
            len(base_reservations), len(single_reservations)
        )
        new_sizes = len(
            set(
                namespace.eager_allreduce_payload_sizes(base)
            )
            - set(
                namespace.eager_allreduce_payload_sizes(single_width)
            )
        )
        self.assertEqual(
            len(base_reservations) - len(single_reservations),
            new_sizes,
        )


if __name__ == "__main__":
    unittest.main()
