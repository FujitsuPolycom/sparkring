from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import spark_tp4_allgather_backend
import spark_tp4_backend
import spark_tp4_dcp_backend
import spark_tp4_port_namespace as namespace
import spark_tp4_vocab_allgather_backend


class Tp4PortNamespaceTest(unittest.TestCase):
    def test_all_active_default_families_are_globally_unique(self) -> None:
        environment = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "VLLM_SPARK_TP4_PREFILL_Q512": "1",
            "VLLM_SPARK_TP4_GRAPH_Q1": "1",
            "VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40": "1",
            "VLLM_SPARK_TP4_ALLGATHER_MODE": "custom",
            "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM": "1",
            "VLLM_SPARK_TP4_VOCAB_MODE": "custom",
            "VLLM_SPARK_TP4_DCP_MODE": "custom",
            "VLLM_SPARK_TP4_DCP_GRAPH_CUSTOM": "1",
        }

        reservations = namespace.validate_active_port_namespace(environment)

        self.assertEqual(len(reservations), 525)
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
            by_owner["eager_allreduce:payload=12288"], (11000, 11001)
        )
        self.assertEqual(
            by_owner["eager_allreduce:payload=6291456"],
            (12022, 12023),
        )
        self.assertEqual(by_owner["eager_allgather:slot=0"], (9490, 9491))
        self.assertNotIn("eager_allgather:slot=2", by_owner)
        self.assertNotIn("eager_allgather:slot=7", by_owner)
        self.assertEqual(by_owner["graph_allreduce"], (9970, 9971))
        self.assertEqual(
            by_owner["graph_dual_port_q40_allreduce"], (9972, 9973)
        )

    def test_ckv_slots_keep_durable_ids_and_reserve_only_when_enabled(
        self,
    ) -> None:
        disabled = {"VLLM_SPARK_TP4_ALLGATHER_MODE": "custom"}
        enabled = {
            **disabled,
            "SPARK_TP4_ALLGATHER_ENABLE_CKV": "1",
        }

        disabled_owners = {
            reservation.owner
            for reservation in namespace.validate_active_port_namespace(
                disabled
            )
        }
        enabled_by_owner = {
            reservation.owner: reservation.ports
            for reservation in namespace.validate_active_port_namespace(
                enabled
            )
        }

        self.assertNotIn("eager_allgather:slot=2", disabled_owners)
        self.assertNotIn("eager_allgather:slot=7", disabled_owners)
        self.assertEqual(
            enabled_by_owner["eager_allgather:slot=2"], (9510, 9511)
        )
        self.assertEqual(
            enabled_by_owner["eager_allgather:slot=7"], (9560, 9561)
        )
        self.assertEqual(
            namespace.eager_allgather_control_ports(2, disabled),
            (9510, 9511),
        )
        self.assertEqual(
            namespace.eager_allgather_control_ports(7, disabled),
            (9560, 9561),
        )

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

    def test_old_allreduce_default_collides_with_active_allgather(self) -> None:
        environment = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "SPARK_TP4_CONTROL_PORT0": "9480",
            "SPARK_TP4_CONTROL_PORT1": "9481",
            "VLLM_SPARK_TP4_ALLGATHER_MODE": "custom",
        }

        with self.assertRaisesRegex(
            ValueError,
            r"port 9490: eager_allreduce:payload=73728, "
            r"eager_allgather:slot=0",
        ):
            namespace.validate_active_port_namespace(environment)

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
            r"port 12002: eager_allreduce:payload=12288, "
            r"eager_allreduce:payload=24576",
        ):
            namespace.validate_active_port_namespace(environment)

    def test_active_static_family_collision_fails_closed(self) -> None:
        environment = {
            "VLLM_SPARK_TP4_MODE": "custom",
            "VLLM_SPARK_TP4_GRAPH_Q1": "1",
            "SPARK_TP4_GRAPH_CONTROL_PORT0": "9890",
            "SPARK_TP4_GRAPH_CONTROL_PORT1": "9891",
            "VLLM_SPARK_TP4_DCP_MODE": "custom",
        }

        with self.assertRaisesRegex(
            ValueError, r"port 9890: graph_allreduce, eager_dcp"
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
                spark_tp4_allgather_backend._allgather_control_ports(7),
                (9560, 9561),
            )
            self.assertEqual(
                spark_tp4_allgather_backend._indexer_graph_control_ports(),
                (9462, 9463),
            )
            self.assertEqual(
                spark_tp4_dcp_backend._eager_control_ports(), (9890, 9891)
            )
            self.assertEqual(
                spark_tp4_dcp_backend._graph_control_ports(), (9892, 9893)
            )
            self.assertEqual(
                spark_tp4_vocab_allgather_backend._eager_control_ports(),
                (9990, 9991),
            )
            self.assertEqual(
                spark_tp4_vocab_allgather_backend._graph_control_ports(),
                (10110, 10111),
            )

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
        # 2*3072*2 == 1*6144*2 == 12288 appears once.
        self.assertEqual(sizes.count(12288), 1)
        self.assertIn(12288, sizes)
        # Ports stride by 2 from base in sorted-size order.
        index = sizes.index(12288)
        self.assertEqual(
            namespace.eager_allreduce_ports_for_payload(
                12288, environment
            ),
            (11000 + 2 * index, 11001 + 2 * index),
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
