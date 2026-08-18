"""GPU-free tests for the generic query-row provider seam."""

from __future__ import annotations

import sys
import types
import unittest

import spark_tp4_port_namespace
import spark_tp4_query_row_provider as seam


def _synthetic_provider(name: str, rows) -> str:
    """Install a synthetic provider module under the given name."""

    module = types.ModuleType(name)
    module.provider_query_rows = lambda environ: rows
    sys.modules[name] = module
    return name


class QueryRowProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        seam._cache.clear()
        self._synthetic = [
            name for name in sys.modules if name.startswith("_synthetic_rows_")
        ]

    def tearDown(self) -> None:
        seam._cache.clear()
        for name in list(sys.modules):
            if name.startswith("_synthetic_rows_"):
                del sys.modules[name]

    def test_adapters_do_not_import_the_deployment_contract(self) -> None:
        """The generic tree must be importable from a clean checkout."""

        self.assertNotIn("spark_tp4_sparse_q42_q48_contract", sys.modules)

    def test_default_is_contiguous_public_rows(self) -> None:
        rows = seam.resolve_query_rows({})
        self.assertEqual(rows, tuple(range(1, seam.MAX_QUERY_ROWS + 1)))

    def test_q512_rows_without_provider(self) -> None:
        rows = seam.resolve_query_rows({"VLLM_SPARK_TP4_PREFILL_Q512": "1"})
        self.assertEqual(rows, tuple(range(1, 513)))

    def test_synthetic_sparse_provider_rows_are_canonicalized(self) -> None:
        name = _synthetic_provider("_synthetic_rows_sparse", [7, 3, 40, 12])
        rows = seam.resolve_query_rows(
            {"VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": name}
        )
        self.assertEqual(rows, (3, 7, 12, 40))

    def test_missing_provider_fails_with_named_error(self) -> None:
        with self.assertRaises(ValueError) as caught:
            seam.resolve_query_rows(
                {"VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": "_no_such_module_xyz"}
            )
        message = str(caught.exception)
        self.assertIn("VLLM_SPARK_TP4_QUERY_ROW_PROVIDER", message)
        self.assertIn("_no_such_module_xyz", message)

    def test_provider_without_interface_fails(self) -> None:
        name = "_synthetic_rows_no_interface"
        sys.modules[name] = types.ModuleType(name)
        with self.assertRaises(ValueError) as caught:
            seam.resolve_query_rows(
                {"VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": name}
            )
        self.assertIn("provider_query_rows", str(caught.exception))

    def test_malformed_provider_rows_fail_closed(self) -> None:
        cases = {
            "_synthetic_rows_empty": [],
            "_synthetic_rows_bool": [1, True],
            "_synthetic_rows_string": ["3"],
            "_synthetic_rows_zero": [0, 1],
            "_synthetic_rows_negative": [-1, 2],
            "_synthetic_rows_beyond_span": [1, 513],
            "_synthetic_rows_duplicate": [4, 4],
            "_synthetic_rows_not_iterable": 7,
        }
        for name, rows in cases.items():
            with self.subTest(provider=name):
                _synthetic_provider(name, rows)
                with self.assertRaises(ValueError) as caught:
                    seam.resolve_query_rows(
                        {"VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": name}
                    )
                self.assertIn(
                    "VLLM_SPARK_TP4_QUERY_ROW_PROVIDER", str(caught.exception)
                )

    def test_q512_and_provider_are_mutually_exclusive(self) -> None:
        name = _synthetic_provider("_synthetic_rows_conflict", [1, 2])
        with self.assertRaises(ValueError) as caught:
            seam.resolve_query_rows({
                "VLLM_SPARK_TP4_PREFILL_Q512": "1",
                "VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": name,
            })
        self.assertIn("mutually exclusive", str(caught.exception))

    def test_provider_identity_changes_reservations_deterministically(
        self,
    ) -> None:
        base_env = {
            "SPARK_TP4_CONTROL_PORT0": "11100",
            "SPARK_TP4_CONTROL_PORT1": "11101",
            "VLLM_SPARK_TP4_MODE": "custom",
        }
        name = _synthetic_provider("_synthetic_rows_identity", [2, 5])
        default = spark_tp4_port_namespace.active_port_reservations(base_env)
        with_provider = spark_tp4_port_namespace.active_port_reservations(
            {**base_env, "VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": name}
        )
        again = spark_tp4_port_namespace.active_port_reservations(
            {**base_env, "VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": name}
        )
        self.assertEqual(with_provider, again)
        self.assertNotEqual(default, with_provider)
        eager = [
            reservation for reservation in with_provider
            if reservation.owner.startswith("eager_allreduce:")
        ]
        self.assertEqual(len(eager), 2)
        # Row-slot formula: slot = row - 1.
        self.assertEqual(
            [reservation.ports for reservation in eager],
            [(11100 + 2 * 1, 11101 + 2 * 1), (11100 + 2 * 4, 11101 + 2 * 4)],
        )

    def test_width_unset_and_default_width_are_equivalent(self) -> None:
        unset = spark_tp4_port_namespace.eager_allreduce_payload_sizes({})
        explicit = spark_tp4_port_namespace.eager_allreduce_payload_sizes(
            {"VLLM_SPARK_TP4_EAGER_WIDTHS": "6144"}
        )
        self.assertEqual(unset, explicit)

    def test_provider_rows_and_extension_widths_compose(self) -> None:
        name = _synthetic_provider(
            "_synthetic_rows_composed", [1, 2, 3, 4, 5, 6, 9]
        )
        env = {
            "SPARK_TP4_CONTROL_PORT0": "11100",
            "SPARK_TP4_CONTROL_PORT1": "11101",
            "VLLM_SPARK_TP4_MODE": "custom",
            "VLLM_SPARK_TP4_QUERY_ROW_PROVIDER": name,
            "VLLM_SPARK_TP4_EAGER_WIDTHS": "2880,6144",
        }
        reservations = spark_tp4_port_namespace.active_port_reservations(env)
        ports = [
            port
            for reservation in reservations
            for port in reservation.ports
        ]
        self.assertEqual(len(ports), len(set(ports)), "port collision")
        extension_ports = [
            reservation.ports
            for reservation in reservations
            if reservation.owner.startswith("eager_allreduce:payload=")
        ]
        self.assertTrue(extension_ports)
        # Extension slots begin at 512: base + 2 * 512.
        for port0, port1 in extension_ports:
            self.assertGreaterEqual(port0, 11100 + 2 * 512)
            self.assertGreaterEqual(port1, 11101 + 2 * 512)


if __name__ == "__main__":
    unittest.main()
