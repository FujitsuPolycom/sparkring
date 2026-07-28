import json
import tempfile
import unittest
from pathlib import Path

from spark_transport.experiments.moe_round_floor.route_reuse import (
    SCHEMA,
    analyze_round,
    canonical_route_digest,
    classify,
    load_jsonl,
    summarize,
)


def record(routes: list[list[list[int]]]) -> dict:
    return {
        "schema": SCHEMA,
        "request_key": "fixture",
        "round": 7,
        "layers": [
            {
                "layer": layer,
                "positions": [{"expert_ids": experts} for experts in positions],
            }
            for layer, positions in enumerate(routes)
        ],
    }


class RouteReuseTest(unittest.TestCase):
    def test_identical_q5_routes_have_five_x_reuse(self) -> None:
        fixture = record(
            [
                [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
                [[7, 8], [7, 8], [7, 8], [7, 8], [7, 8]],
            ]
        )
        summary, layers = analyze_round(fixture)
        self.assertEqual(summary.assignments, 20)
        self.assertEqual(summary.unique_expert_layer_pairs, 4)
        self.assertEqual(summary.aggregate_reuse_factor, 5.0)
        self.assertTrue(all(layer.reuse_factor == 5.0 for layer in layers))
        self.assertEqual(summary.route_order_runs, 20)
        self.assertEqual(summary.schedule_compaction_factor, 5.0)

    def test_disjoint_routes_have_no_reuse(self) -> None:
        fixture = record(
            [[
                [0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9],
            ]]
        )
        summary, _ = analyze_round(fixture)
        self.assertEqual(summary.aggregate_reuse_factor, 1.0)
        self.assertEqual(summary.duplicate_fraction, 0.0)
        self.assertEqual(summary.schedule_compaction_factor, 1.0)
        self.assertIn("NO-GO", classify(summary.aggregate_reuse_factor))

    def test_schedule_compaction_counts_token_major_expert_runs(self) -> None:
        fixture = record(
            [[
                [1, 2],
                [2, 1],
                [1, 2],
                [2, 1],
                [1, 2],
            ]]
        )
        summary, layers = analyze_round(fixture)
        self.assertEqual(layers[0].route_order_runs, 6)
        self.assertEqual(layers[0].unique_experts, 2)
        self.assertEqual(layers[0].schedule_compaction_factor, 3.0)
        self.assertEqual(summary.schedule_compaction_factor, 3.0)

    def test_summary_uses_p10_as_conservative_gate(self) -> None:
        fixtures = []
        for round_index in range(9):
            weak = record([[[1], [2], [3], [4], [5]]])
            weak["round"] = round_index
            fixtures.append(weak)
        strong = record([[[1], [1], [1], [1], [1]]])
        strong["round"] = 9
        result = summarize([*fixtures, strong])
        self.assertEqual(result["rounds"], 10)
        self.assertIn("NO-GO", result["decision"])
        self.assertIn("logical_schedule_compaction", result)

    def test_summary_reports_positionwise_expert_expansion(self) -> None:
        fixture = record(
            [
                [[1, 2], [2, 3], [4, 5]],
                [[7, 8], [7, 8], [7, 8]],
            ]
        )
        second_round = record(
            [
                [[10, 11], [12, 13], [14, 15]],
                [[17, 18], [17, 18], [17, 18]],
            ]
        )
        second_round["round"] = 8
        result = summarize([fixture, second_round], width=3)
        expansion = result["expert_expansion"]
        self.assertEqual(expansion["observations"], 4)
        self.assertEqual(
            [point["k"] for point in expansion["curve"]],
            [1, 2, 3],
        )
        self.assertEqual(expansion["curve"][0]["median"], 1.0)
        self.assertEqual(expansion["curve"][1]["median"], 1.25)
        self.assertEqual(expansion["curve"][2]["median"], 1.75)

    def test_summary_reports_only_same_request_adjacent_round_reuse(self) -> None:
        request_a_round_0 = record([[[1, 2], [2, 3]]])
        request_a_round_0.update(request_key="request-a", round=0)
        request_a_round_1 = record([[[2, 3], [3, 4]]])
        request_a_round_1.update(request_key="request-a", round=1)
        request_a_round_3 = record([[[1, 9], [9, 10]]])
        request_a_round_3.update(request_key="request-a", round=3)
        request_b_round_0 = record([[[10], [11]]])
        request_b_round_0.update(request_key="request-b", round=0)
        request_b_round_1 = record([[[10], [11]]])
        request_b_round_1.update(request_key="request-b", round=1)

        result = summarize(
            [
                request_a_round_3,
                request_b_round_1,
                request_a_round_1,
                request_b_round_0,
                request_a_round_0,
            ],
            width=2,
        )
        adjacent = result["adjacent_round_reuse"]
        self.assertEqual(adjacent["round_pairs"], 2)
        self.assertEqual(adjacent["layer_observations"], 2)
        self.assertAlmostEqual(
            adjacent["next_round_coverage"]["median"],
            (2 / 3 + 1.0) / 2,
        )
        self.assertEqual(adjacent["jaccard"]["median"], 0.75)
        self.assertEqual(adjacent["intersection_experts"]["median"], 2.0)

    def test_adjacent_reuse_preserves_legacy_trace_compatibility(self) -> None:
        missing_request_key = record([[[1], [2]]])
        del missing_request_key["request_key"]
        duplicate_round_a = record([[[3], [4]]])
        duplicate_round_b = record([[[5], [6]]])

        result = summarize(
            [missing_request_key, duplicate_round_a, duplicate_round_b],
            width=2,
        )

        adjacent = result["adjacent_round_reuse"]
        self.assertIs(adjacent["available"], False)
        self.assertEqual(adjacent["round_pairs"], 0)
        self.assertEqual(adjacent["skipped_missing_request_key"], 1)
        self.assertEqual(adjacent["skipped_duplicate_round_records"], 2)

    def test_canonical_digest_ignores_provenance_and_input_order(self) -> None:
        rank_0_round_0 = record([[[1], [2]]])
        rank_0_round_0.update(round=0, provenance={"rank": 0, "image": "a"})
        rank_0_round_1 = record([[[2], [3]]])
        rank_0_round_1.update(round=1, provenance={"rank": 0, "image": "a"})

        rank_1_round_0 = json.loads(json.dumps(rank_0_round_0))
        rank_1_round_0["provenance"] = {"rank": 1, "image": "different"}
        rank_1_round_1 = json.loads(json.dumps(rank_0_round_1))
        rank_1_round_1["provenance"] = {"rank": 1, "image": "different"}

        rank_0_digest = canonical_route_digest(
            [rank_0_round_0, rank_0_round_1]
        )
        rank_1_digest = canonical_route_digest(
            [rank_1_round_1, rank_1_round_0]
        )
        self.assertEqual(rank_0_digest, rank_1_digest)

        rank_1_round_1["layers"][0]["positions"][1]["expert_ids"] = [9]
        self.assertNotEqual(
            rank_0_digest,
            canonical_route_digest([rank_1_round_0, rank_1_round_1]),
        )

    def test_rejected_route_waste_is_explicitly_unavailable(self) -> None:
        result = summarize([record([[[1], [1], [2]]])], width=3)
        waste = result["rejected_route_waste"]
        self.assertIs(waste["available"], False)
        self.assertIn("acceptance", waste["reason"])
        self.assertNotIn("estimate", waste)

    def test_rejected_route_waste_is_exact_per_layer_and_round(self) -> None:
        fixture = record(
            [
                [[1, 2], [2, 3], [3, 4]],
                [[7], [7], [7]],
            ]
        )
        fixture.update(accepted_prefix_tokens=1, rejected_tokens=1)

        waste = summarize([fixture], width=3)["rejected_route_waste"]

        self.assertIs(waste["available"], True)
        self.assertIs(waste["exact"], True)
        self.assertEqual(waste["rounds"], 1)
        self.assertEqual(waste["layer_round_observations"], 2)
        first_layer, second_layer = waste["layer_round_details"]
        self.assertEqual(first_layer["retained_positions"], 2)
        self.assertEqual(first_layer["rejected_positions"], 1)
        self.assertEqual(first_layer["retained_unique_experts"], 3)
        self.assertEqual(first_layer["rejected_unique_experts"], 2)
        self.assertEqual(first_layer["rejected_only_unique_experts"], 1)
        self.assertEqual(
            first_layer["rejected_only_fraction_of_all_unique_experts"],
            0.25,
        )
        self.assertEqual(second_layer["rejected_only_unique_experts"], 0)

        aggregate = waste["aggregate"]
        self.assertEqual(
            aggregate["all_unique_expert_layer_pair_observations"],
            5,
        )
        self.assertEqual(
            aggregate["rejected_only_unique_expert_layer_pair_observations"],
            1,
        )
        self.assertEqual(
            aggregate[
                "rejected_only_fraction_of_all_unique_expert_layer_pair_observations"
            ],
            0.2,
        )
        self.assertEqual(
            waste["per_layer_round"]["rejected_only_unique_experts"]["median"],
            0.5,
        )
        self.assertEqual(
            waste["per_round"][
                "rejected_only_unique_expert_layer_pairs"
            ]["median"],
            1,
        )

    def test_zero_rejected_tokens_reports_exact_zero_waste(self) -> None:
        fixture = record([[[1], [2], [3]]])
        fixture.update(accepted_prefix_tokens=2, rejected_tokens=0)

        waste = summarize([fixture], width=3)["rejected_route_waste"]

        self.assertIs(waste["available"], True)
        detail = waste["layer_round_details"][0]
        self.assertEqual(detail["rejected_positions"], 0)
        self.assertEqual(detail["rejected_unique_experts"], 0)
        self.assertEqual(detail["rejected_only_unique_experts"], 0)
        self.assertEqual(
            waste["aggregate"][
                "rejected_only_unique_expert_layer_pair_observations"
            ],
            0,
        )

    def test_rejected_route_waste_rejects_mixed_metadata_availability(
        self,
    ) -> None:
        exact = record([[[1], [2], [3]]])
        exact.update(accepted_prefix_tokens=1, rejected_tokens=1)
        legacy = record([[[1], [2], [3]]])
        legacy["round"] = 8

        with self.assertRaisesRegex(ValueError, "mixed exact acceptance"):
            summarize([exact, legacy], width=3)

    def test_rejected_route_waste_rejects_malformed_metadata(self) -> None:
        cases = [
            ({"accepted_prefix_tokens": 1}, "both be present"),
            (
                {
                    "accepted_prefix_tokens": "1",
                    "rejected_tokens": 1,
                },
                "must be integers",
            ),
            (
                {
                    "accepted_prefix_tokens": True,
                    "rejected_tokens": 1,
                },
                "must be integers",
            ),
            (
                {
                    "accepted_prefix_tokens": -1,
                    "rejected_tokens": 3,
                },
                "must be non-negative",
            ),
            (
                {
                    "width": 4,
                    "accepted_prefix_tokens": 1,
                    "rejected_tokens": 1,
                },
                "does not match analysis width",
            ),
            (
                {
                    "accepted_prefix_tokens": 1,
                    "rejected_tokens": 0,
                },
                "must equal width - 1",
            ),
        ]
        for metadata, message in cases:
            with self.subTest(metadata=metadata):
                fixture = record([[[1], [2], [3]]])
                fixture.update(metadata)
                with self.assertRaisesRegex(ValueError, message):
                    summarize([fixture], width=3)

    def test_rejects_duplicate_expert_within_position(self) -> None:
        fixture = record([[[1, 1], [1], [1], [1], [1]]])
        with self.assertRaisesRegex(ValueError, "duplicate expert"):
            analyze_round(fixture)

    def test_jsonl_loader_reports_empty_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no records"):
                load_jsonl(path)

    def test_jsonl_round_trip(self) -> None:
        fixture = record([[[1], [1], [2], [2], [2]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
            loaded = load_jsonl(path)
        self.assertEqual(loaded, [fixture])


if __name__ == "__main__":
    unittest.main()
