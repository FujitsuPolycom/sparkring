"""GPU-free tests for the quantized-checkpoint bit-width census.

Every fixture is a safetensors container written byte by byte here — an
8-byte little-endian header length, a JSON header, then payload — so the tests
exercise the same container parsing the census performs against a real
checkpoint, without a model, a GPU, or the safetensors package.
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import expert_bitwidth_census as census

# One EXL3 trellis of this shape covers a [64, 128] logical matrix: the
# packing is [K/16, N/16, 16*bits], so the tier is the last dimension over 16.
TRELLIS_SHAPE0 = 4
TRELLIS_SHAPE1 = 8
LOGICAL_WEIGHTS = TRELLIS_SHAPE0 * 16 * TRELLIS_SHAPE1 * 16


def trellis(bits: int) -> tuple[str, list[int]]:
    return "I16", [TRELLIS_SHAPE0, TRELLIS_SHAPE1, 16 * bits]


def _stored_bytes(dtype: str, shape: list[int]) -> int:
    elements = 1
    for extent in shape:
        elements *= extent
    return elements * census.DTYPE_BITS[dtype] // 8


def write_shard(
    path: Path,
    tensors: dict[str, tuple[str, list[int]]],
    *,
    metadata: dict[str, str] | None = None,
    override_bytes: dict[str, int] | None = None,
) -> None:
    """Write a safetensors container holding zeroed payload of the right size."""

    override_bytes = override_bytes or {}
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    total = 0
    for name, (dtype, shape) in tensors.items():
        size = override_bytes.get(name, _stored_bytes(dtype, shape))
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
        total += size
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(total))


def build_model(root: Path) -> None:
    """A two-shard MoE checkpoint with a 3/4/5-bit routed-expert tier mix.

    Routed experts: four (layer, expert) instances over a [64, 128] matrix
    each, at 3, 4, 5, and 3 bits. Every instance carries the float16 rotation
    vectors an EXL3 matrix cannot be decoded without.
    """

    write_shard(
        root / "model-00001-of-00002.safetensors",
        {
            "model.layers.3.mlp.experts.0.down_proj.trellis": trellis(3),
            "model.layers.3.mlp.experts.0.down_proj.suh": ("F16", [64]),
            "model.layers.3.mlp.experts.0.down_proj.svh": ("F16", [128]),
            "model.layers.3.mlp.experts.1.down_proj.trellis": trellis(4),
            "model.layers.3.mlp.experts.1.down_proj.suh": ("F16", [64]),
            "model.layers.3.mlp.experts.1.down_proj.svh": ("F16", [128]),
            "model.layers.3.self_attn.q_proj.weight": ("F16", [128, 64]),
            "model.layers.3.mlp.shared_experts.down_proj.weight": ("F16", [64, 32]),
            "model.layers.3.mlp.gate.weight": ("F16", [8, 64]),
        },
        metadata={"format": "pt"},
    )
    write_shard(
        root / "model-00002-of-00002.safetensors",
        {
            "model.layers.4.mlp.experts.0.down_proj.trellis": trellis(5),
            "model.layers.4.mlp.experts.0.down_proj.suh": ("F16", [64]),
            "model.layers.4.mlp.experts.0.down_proj.svh": ("F16", [128]),
            "model.layers.4.mlp.experts.1.down_proj.trellis": trellis(3),
            "model.layers.4.mlp.experts.1.down_proj.suh": ("F16", [64]),
            "model.layers.4.mlp.experts.1.down_proj.svh": ("F16", [128]),
            "model.embed_tokens.weight": ("F16", [256, 64]),
            "model.rotary_emb.inv_freq": ("F32", [32]),
        },
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm4_moe",
                "quantization_config": {
                    "quant_method": "exl3",
                    "bits": 3.5,
                    "head_bits": 6,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "tier_bitmap.json").write_text(
        json.dumps({"3": {"k": [3, 4, 3, 4, 3, 4, 3, 4]}}),
        encoding="utf-8",
    )


class HeaderParsingTest(unittest.TestCase):
    def test_reads_dtype_shape_and_offsets_without_payload(self) -> None:
        with TemporaryDirectory() as raw:
            shard = Path(raw) / "one.safetensors"
            write_shard(shard, {"a.weight": ("F16", [4, 8])})

            header, payload_start = census.read_safetensors_header(shard)

            self.assertEqual(header["a.weight"]["dtype"], "F16")
            self.assertEqual(header["a.weight"]["shape"], [4, 8])
            self.assertEqual(header["a.weight"]["data_offsets"], [0, 64])
            self.assertEqual(payload_start, shard.stat().st_size - 64)

    def test_metadata_block_is_not_a_tensor(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            write_shard(
                root / "one.safetensors",
                {"a.weight": ("F16", [4, 8])},
                metadata={"format": "pt"},
            )

            records, findings = census.read_tensor_records(root)

            self.assertEqual([record.name for record in records], ["a.weight"])
            self.assertEqual(findings, [])

    def test_a_file_shorter_than_the_length_prefix_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            shard = Path(raw) / "short.safetensors"
            shard.write_bytes(b"\x01\x02\x03")

            with self.assertRaises(census.CensusError) as caught:
                census.read_safetensors_header(shard)

            self.assertIn("shorter than", str(caught.exception))

    def test_a_zero_length_header_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            shard = Path(raw) / "zero.safetensors"
            shard.write_bytes((0).to_bytes(8, "little") + b"payload")

            with self.assertRaises(census.CensusError) as caught:
                census.read_safetensors_header(shard)

            self.assertIn("zero-length header", str(caught.exception))

    def test_a_header_longer_than_the_file_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            shard = Path(raw) / "over.safetensors"
            shard.write_bytes((4096).to_bytes(8, "little") + b"{}")

            with self.assertRaises(census.CensusError) as caught:
                census.read_safetensors_header(shard)

            self.assertIn("bytes in total", str(caught.exception))

    def test_an_implausible_header_length_is_rejected_before_reading(self) -> None:
        with TemporaryDirectory() as raw:
            shard = Path(raw) / "huge.safetensors"
            shard.write_bytes((2**60).to_bytes(8, "little") + b"{}")

            with self.assertRaises(census.CensusError) as caught:
                census.read_safetensors_header(shard)

            self.assertIn("beyond", str(caught.exception))

    def test_a_non_json_header_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            shard = Path(raw) / "junk.safetensors"
            body = b"not json at all"
            shard.write_bytes(len(body).to_bytes(8, "little") + body)

            with self.assertRaises(census.CensusError) as caught:
                census.read_safetensors_header(shard)

            self.assertIn("not JSON", str(caught.exception))

    def test_a_json_array_header_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            shard = Path(raw) / "array.safetensors"
            body = b"[1, 2, 3]"
            shard.write_bytes(len(body).to_bytes(8, "little") + body)

            with self.assertRaises(census.CensusError) as caught:
                census.read_safetensors_header(shard)

            self.assertIn("not a JSON object", str(caught.exception))


class GeometryTest(unittest.TestCase):
    def test_trellis_tier_comes_from_the_packed_dimension(self) -> None:
        for bits in (3, 4, 5):
            dtype, shape = trellis(bits)
            stored = _stored_bytes(dtype, shape)

            method, logical, bpw, note = census.derive_geometry(
                "model.layers.0.mlp.experts.0.down_proj.trellis", dtype, shape, stored
            )

            self.assertEqual(method, census.DERIVATION_TRELLIS)
            self.assertEqual(logical, LOGICAL_WEIGHTS)
            self.assertAlmostEqual(bpw, float(bits))
            self.assertEqual(note, "logical shape [64, 128]")

    def test_a_float_tensor_measures_its_own_dtype_width(self) -> None:
        method, logical, bpw, note = census.derive_geometry(
            "model.layers.0.self_attn.q_proj.weight", "BF16", [16, 32], 1024
        )

        self.assertEqual(method, census.DERIVATION_UNPACKED)
        self.assertEqual(logical, 512)
        self.assertAlmostEqual(bpw, 16.0)
        self.assertIsNone(note)

    def test_packed_integer_storage_is_undetermined_not_guessed(self) -> None:
        method, logical, bpw, note = census.derive_geometry(
            "model.layers.0.mlp.experts.0.down_proj.qweight", "U8", [64, 32], 2048
        )

        self.assertEqual(method, census.DERIVATION_UNDETERMINED)
        self.assertIsNone(logical)
        self.assertIsNone(bpw)
        self.assertIn("no recognized packing", note)

    def test_a_trellis_of_the_wrong_dtype_is_undetermined(self) -> None:
        method, _, _, note = census.derive_geometry(
            "a.trellis", "U8", [4, 8, 48], 1536
        )

        self.assertEqual(method, census.DERIVATION_UNDETERMINED)
        self.assertIn("requires rank-3 I16", note)

    def test_a_trellis_with_a_partial_block_is_undetermined(self) -> None:
        method, _, _, note = census.derive_geometry(
            "a.trellis", "I16", [4, 8, 50], 3200
        )

        self.assertEqual(method, census.DERIVATION_UNDETERMINED)
        self.assertIn("not a multiple of 16", note)

    def test_a_trellis_beyond_the_format_tier_range_is_undetermined(self) -> None:
        method, _, _, note = census.derive_geometry(
            "a.trellis", "I16", [4, 8, 16 * 9], 9216
        )

        self.assertEqual(method, census.DERIVATION_UNDETERMINED)
        self.assertIn("outside the 1..8 range", note)

    def test_a_byte_range_disagreeing_with_the_shape_is_undetermined(self) -> None:
        method, _, _, note = census.derive_geometry(
            "a.weight", "F16", [16, 32], 999
        )

        self.assertEqual(method, census.DERIVATION_UNDETERMINED)
        self.assertIn("occupies 1024", note)

    def test_an_unknown_dtype_is_undetermined(self) -> None:
        method, _, _, note = census.derive_geometry("a.weight", "F4", [16, 32], 256)

        self.assertEqual(method, census.DERIVATION_UNDETERMINED)
        self.assertIn("not a known safetensors dtype", note)


class ClassificationTest(unittest.TestCase):
    def test_routed_expert_requires_a_numeric_index(self) -> None:
        self.assertEqual(
            census.classify_name("model.layers.3.mlp.experts.7.down_proj.trellis"),
            census.CLASS_EXPERT,
        )

    def test_shared_experts_are_not_routed_experts(self) -> None:
        for name in (
            "model.layers.3.mlp.shared_experts.down_proj.weight",
            "model.layers.3.mlp.shared_expert.gate_proj.weight",
            "model.layers.3.mlp.shared_experts.0.up_proj.weight",
        ):
            self.assertEqual(census.classify_name(name), census.CLASS_SHARED_DENSE)

    def test_attention_dense_embedding_and_other(self) -> None:
        cases = {
            "model.layers.3.self_attn.q_proj.weight": census.CLASS_ATTENTION,
            "model.layers.3.attn.kv_a_proj.weight": census.CLASS_ATTENTION,
            "model.layers.3.mlp.gate.weight": census.CLASS_SHARED_DENSE,
            "model.layers.3.input_layernorm.weight": census.CLASS_SHARED_DENSE,
            "model.embed_tokens.weight": census.CLASS_EMBEDDING,
            "lm_head.trellis": census.CLASS_EMBEDDING,
            "model.rotary_emb.inv_freq": census.CLASS_OTHER,
        }
        for name, expected in cases.items():
            self.assertEqual(census.classify_name(name), expected, name)

    def test_sidecar_suffixes_are_not_counted_as_weights(self) -> None:
        self.assertEqual(census.tensor_role("a.b.suh"), "sidecar")
        self.assertEqual(census.tensor_role("a.b.weight_scale_inv"), "sidecar")
        self.assertEqual(census.tensor_role("a.b.trellis"), "weight")
        self.assertEqual(census.tensor_role("a.b.weight"), "weight")


class CensusTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        build_model(self.root)
        self.report = census.census(self.root, 0.01)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_totals_cover_every_tensor_in_both_shards(self) -> None:
        self.assertEqual(len(self.report["safetensors_files"]), 2)
        self.assertEqual(self.report["totals"]["tensor_count"], 17)
        self.assertEqual(self.report["totals"]["undetermined_tensor_count"], 0)

    def test_expert_class_counts_and_bytes(self) -> None:
        expert = self.report["classes"][census.CLASS_EXPERT]

        self.assertEqual(expert["tensor_count"], 12)
        self.assertEqual(expert["weight_tensor_count"], 4)
        self.assertEqual(expert["sidecar_tensor_count"], 8)
        self.assertEqual(expert["weight_bytes"], 15360)
        self.assertEqual(expert["sidecar_bytes"], 1536)
        self.assertEqual(expert["stored_bytes"], 16896)
        self.assertEqual(expert["logical_weights"], 4 * LOGICAL_WEIGHTS)

    def test_expert_bit_width_distribution_is_the_stored_tier_mix(self) -> None:
        spread = self.report["classes"][census.CLASS_EXPERT]["bits_per_weight"]

        self.assertEqual(spread["min"], 3.0)
        self.assertEqual(spread["median"], 3.5)
        self.assertEqual(spread["max"], 5.0)
        self.assertEqual(
            [(bucket["bits_per_weight"], bucket["tensor_count"]) for bucket in spread["histogram"]],
            [(3.0, 2), (4.0, 1), (5.0, 1)],
        )
        self.assertAlmostEqual(spread["histogram"][0]["share_of_logical_weights"], 0.5)

    def test_expert_aggregate_is_the_measured_replacement_for_a_uniform_rate(
        self,
    ) -> None:
        aggregate = self.report["expert_aggregate"]

        self.assertEqual(aggregate["logical_weights"], 32768)
        self.assertEqual(aggregate["payload_bytes"], 15360)
        self.assertEqual(aggregate["sidecar_bytes"], 1536)
        self.assertEqual(aggregate["average_bits_per_weight_payload"], 3.75)
        self.assertEqual(aggregate["average_bits_per_weight_with_sidecars"], 4.125)
        self.assertTrue(aggregate["covers_every_expert_tensor"])

    def test_per_expert_instance_spread_is_reported(self) -> None:
        instances = self.report["expert_instances"]

        self.assertEqual(instances["count"], 4)
        self.assertEqual(instances["stored_bytes_min"], 3456)
        self.assertEqual(instances["stored_bytes_median"], 3968)
        self.assertEqual(instances["stored_bytes_max"], 5504)
        self.assertEqual(instances["bits_per_weight_with_sidecars_min"], 3.375)
        self.assertEqual(instances["bits_per_weight_with_sidecars_median"], 3.875)
        self.assertEqual(instances["bits_per_weight_with_sidecars_max"], 5.375)

    def test_other_classes_measure_their_stored_width(self) -> None:
        classes = self.report["classes"]

        self.assertEqual(classes[census.CLASS_ATTENTION]["tensor_count"], 1)
        self.assertEqual(classes[census.CLASS_ATTENTION]["stored_bytes"], 16384)
        self.assertEqual(
            classes[census.CLASS_ATTENTION]["bits_per_weight"]["median"], 16.0
        )
        self.assertEqual(classes[census.CLASS_SHARED_DENSE]["tensor_count"], 2)
        self.assertEqual(classes[census.CLASS_EMBEDDING]["tensor_count"], 1)
        self.assertEqual(classes[census.CLASS_EMBEDDING]["logical_weights"], 16384)
        self.assertEqual(classes[census.CLASS_OTHER]["tensor_count"], 1)

    def test_declared_metadata_is_read_and_attributed(self) -> None:
        declared = self.report["declared"]

        self.assertEqual(declared["declared_average_bits_per_weight"], 3.5)
        self.assertEqual(
            declared["declared_average_source"],
            "config.json:quantization_config.bits",
        )
        self.assertIn(
            {
                "file": "config.json",
                "path": "quantization_config.head_bits",
                "value": 6,
            },
            declared["bit_rate_fields"],
        )
        self.assertEqual(
            declared["tier_vector_histogram"]["counts"], {"3": 4, "4": 4}
        )
        self.assertEqual(declared["tier_vector_histogram"]["mean_declared_tier"], 3.5)

    def test_a_declared_measured_disagreement_is_the_headline_finding(self) -> None:
        comparison = self.report["comparison"]

        self.assertTrue(comparison["comparable"])
        self.assertFalse(comparison["agrees"])
        self.assertEqual(comparison["declared_average_bits_per_weight"], 3.5)
        self.assertEqual(comparison["measured_average_bits_per_weight"], 3.75)
        self.assertEqual(comparison["difference_bits_per_weight"], 0.25)
        self.assertTrue(
            any("disagree" in finding for finding in self.report["findings"])
        )

    def test_the_report_states_the_rules_it_applied(self) -> None:
        classes = [rule["class"] for rule in self.report["classification_rules"]]

        self.assertEqual(classes[0], census.CLASS_EMBEDDING)
        self.assertEqual(classes[1], census.CLASS_SHARED_DENSE)
        self.assertEqual(classes[2], census.CLASS_EXPERT)
        self.assertIn(census.DERIVATION_TRELLIS, self.report["derivation_methods"])

    def test_the_rendered_report_names_the_measured_aggregate(self) -> None:
        text = census.render(self.report)

        self.assertIn("classification rules, first match wins:", text)
        self.assertIn("average bpw, payload      3.75", text)
        self.assertIn("average bpw, incl sidecar 4.125", text)
        self.assertIn("DISAGREE", text)


class UndeterminedTest(unittest.TestCase):
    def test_an_underivable_expert_tensor_is_excluded_and_reported(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            write_shard(
                root / "model.safetensors",
                {
                    "model.layers.0.mlp.experts.0.down_proj.trellis": trellis(3),
                    "model.layers.0.mlp.experts.1.down_proj.qweight": ("U8", [64, 32]),
                },
            )

            report = census.census(root, 0.01)

            aggregate = report["expert_aggregate"]
            self.assertEqual(aggregate["logical_weights"], LOGICAL_WEIGHTS)
            self.assertEqual(aggregate["payload_bytes"], 3072)
            self.assertEqual(aggregate["average_bits_per_weight_payload"], 3.0)
            self.assertEqual(aggregate["undetermined_tensor_count"], 1)
            self.assertEqual(aggregate["undetermined_bytes"], 2048)
            self.assertFalse(aggregate["covers_every_expert_tensor"])
            self.assertEqual(len(report["undetermined"]), 1)
            self.assertEqual(
                report["undetermined"][0]["name"],
                "model.layers.0.mlp.experts.1.down_proj.qweight",
            )
            self.assertTrue(
                any("no derivable logical shape" in item for item in report["findings"])
            )

    def test_a_tensor_whose_byte_range_leaves_the_shard_is_reported(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            shard = root / "model.safetensors"
            header = {
                "a.weight": {
                    "dtype": "F16",
                    "shape": [4, 8],
                    "data_offsets": [0, 64],
                },
                "b.weight": {
                    "dtype": "F16",
                    "shape": [4, 8],
                    "data_offsets": [64, 1024],
                },
            }
            body = json.dumps(header).encode("utf-8")
            shard.write_bytes(len(body).to_bytes(8, "little") + body + bytes(64))

            report = census.census(root, 0.01)

            self.assertEqual(report["totals"]["tensor_count"], 1)
            self.assertTrue(
                any("fall" in finding for finding in report["findings"])
            )

    def test_a_name_present_in_two_shards_is_counted_once(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            for name in ("a.safetensors", "b.safetensors"):
                write_shard(root / name, {"model.embed_tokens.weight": ("F16", [4, 8])})

            report = census.census(root, 0.01)

            self.assertEqual(report["totals"]["tensor_count"], 1)
            self.assertTrue(
                any("appears in both" in finding for finding in report["findings"])
            )

    def test_one_unreadable_shard_does_not_suppress_the_others(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            write_shard(root / "good.safetensors", {"a.weight": ("F16", [4, 8])})
            (root / "bad.safetensors").write_bytes(b"\x00" * 4)

            report = census.census(root, 0.01)

            self.assertEqual(report["totals"]["tensor_count"], 1)
            self.assertTrue(
                any("unreadable shard" in finding for finding in report["findings"])
            )


class ComparisonTest(unittest.TestCase):
    def test_a_checkpoint_without_declared_metadata_is_not_comparable(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            write_shard(
                root / "model.safetensors",
                {"model.layers.0.mlp.experts.0.down_proj.trellis": trellis(4)},
            )

            report = census.census(root, 0.01)

            self.assertFalse(report["comparison"]["comparable"])
            self.assertIsNone(report["comparison"]["agrees"])
            self.assertIn("no declared scalar", report["comparison"]["note"])

    def test_a_matching_declaration_agrees_within_tolerance(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            write_shard(
                root / "model.safetensors",
                {"model.layers.0.mlp.experts.0.down_proj.trellis": trellis(4)},
            )
            (root / "quantization_config.json").write_text(
                json.dumps({"bits": 4.0, "codebook": "mcg"}), encoding="utf-8"
            )

            report = census.census(root, 0.01)

            self.assertTrue(report["comparison"]["agrees"])
            self.assertEqual(report["findings"], [])


class MainTest(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str]:
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = census.main(argv)
        return code, stream.getvalue()

    def test_an_absent_path_fails_with_an_informative_message(self) -> None:
        with TemporaryDirectory() as raw:
            code, output = self._run([str(Path(raw) / "missing")])

        self.assertEqual(code, 1)
        self.assertIn("no such path", output)

    def test_a_file_instead_of_a_directory_fails(self) -> None:
        with TemporaryDirectory() as raw:
            target = Path(raw) / "model.safetensors"
            write_shard(target, {"a.weight": ("F16", [4, 8])})

            code, output = self._run([str(target)])

        self.assertEqual(code, 1)
        self.assertIn("not a directory", output)

    def test_a_directory_without_shards_fails(self) -> None:
        with TemporaryDirectory() as raw:
            (Path(raw) / "config.json").write_text("{}", encoding="utf-8")

            code, output = self._run([raw])

        self.assertEqual(code, 1)
        self.assertIn("holds no *.safetensors files", output)

    def test_json_output_carries_the_schema_and_every_section(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            build_model(root)

            code, output = self._run([str(root), "--json"])

        self.assertEqual(code, 0)
        document = json.loads(output)
        self.assertEqual(document["schema"], census.SCHEMA)
        for key in (
            "model_path",
            "safetensors_files",
            "classification_rules",
            "derivation_methods",
            "sidecar_suffixes",
            "totals",
            "classes",
            "expert_aggregate",
            "expert_instances",
            "declared",
            "comparison",
            "undetermined",
            "findings",
        ):
            self.assertIn(key, document)
        self.assertEqual(
            document["expert_aggregate"]["average_bits_per_weight_payload"], 3.75
        )

    def test_require_agreement_gates_the_exit_code(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            build_model(root)

            reported, _ = self._run([str(root)])
            gated, output = self._run([str(root), "--require-agreement"])

        self.assertEqual(reported, 0)
        self.assertEqual(gated, 1)
        self.assertIn("DISAGREE", output)

    def test_a_wide_tolerance_admits_the_measured_difference(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            build_model(root)

            code, output = self._run(
                [str(root), "--require-agreement", "--tolerance-bpw", "0.5"]
            )

        self.assertEqual(code, 0)
        self.assertIn("AGREE", output)


if __name__ == "__main__":
    unittest.main()
