import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from performance.harnesses.moe_round_floor.b12x_floor_benchmark import (
    CASES,
    PINNED_SOURCES,
    GateError,
    _child_command,
    audit_sources,
    build_dry_run,
    deterministic_routes,
    ensure_coherent_cases_blocked,
    parse_args,
    require_pinned_sources,
    synthetic_weight_bytes,
    validate_routes,
)


class B12xFloorPlanTest(unittest.TestCase):
    def test_required_case_matrix_is_present(self) -> None:
        names = {case.name for case in CASES}
        required = {
            "5xQ1-eager",
            "Q5-direct-micro-eager",
            "Q5-direct-micro-graph",
            "Q6-direct-micro-eager",
            "Q6-direct-micro-graph",
            "Q5-forced-dynamic-eager",
            "Q5-forced-dynamic-graph",
            "Q6-forced-dynamic-eager",
            "Q6-forced-dynamic-graph",
            "Q5-identical-route-graph",
            "Q6-identical-route-graph",
            "Q5-coherent-micro-graph",
            "Q6-coherent-micro-graph",
        }
        self.assertTrue(required.issubset(names))

    def test_coherent_cases_are_fail_closed(self) -> None:
        ensure_coherent_cases_blocked()
        broken = [
            case.__class__(
                **{
                    **case.__dict__,
                    "admission": (
                        "enabled" if case.backend == "coherent-micro" else case.admission
                    ),
                }
            )
            for case in CASES
        ]
        with self.assertRaisesRegex(GateError, "no implementation"):
            ensure_coherent_cases_blocked(broken)

    def test_routes_are_deterministic_valid_and_nontrivial(self) -> None:
        q5 = deterministic_routes(5)
        self.assertEqual(q5, deterministic_routes(5))
        validate_routes(q5, width=5, experts=256, topk=8)
        unique = len({expert for route in q5 for expert in route})
        self.assertGreater(unique, 8)
        self.assertLess(unique, 40)

    def test_identical_routes_reuse_one_topk_set(self) -> None:
        routes = deterministic_routes(6, style="identical")
        self.assertTrue(all(route == routes[0] for route in routes))
        self.assertEqual(len(set(routes[0])), 8)

    def test_weight_size_uses_tp4_local_intermediate(self) -> None:
        sizes = synthetic_weight_bytes()
        # FP4 packs two weights per byte. FC1 combines gate/up outputs; FC2
        # maps the local 512-wide intermediate back to the 6144-wide hidden state.
        self.assertEqual(sizes["w1_fp4"], 256 * 1024 * 3072)
        self.assertEqual(sizes["w2_fp4"], 256 * 6144 * 256)
        self.assertEqual(sizes["total"], sum(v for k, v in sizes.items() if k != "total"))

    def test_dry_run_is_machine_serializable_without_cuda(self) -> None:
        report = build_dry_run()
        encoded = json.dumps(report)
        self.assertIn("Q5-variable", encoded)
        self.assertFalse(report["validation"]["live_cuda_work_attempted"])

    def test_source_audit_matches_exact_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {}
            for name, (relative, _) in PINNED_SOURCES.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"{name}\n".encode()
                path.write_bytes(payload)
                expected[name] = hashlib.sha256(payload).hexdigest()

            audit = audit_sources(str(root))
            self.assertTrue(all(entry["state"] == "mismatch" for entry in audit.values()))
            with self.assertRaisesRegex(GateError, "requires every pinned"):
                require_pinned_sources(audit)
            for name, digest in expected.items():
                self.assertEqual(audit[name]["actual_sha256"], digest)

    def test_dynamic_child_forces_cutover_before_import(self) -> None:
        arguments = parse_args(
            ["--mode", "live", "--warmup", "2", "--iterations", "3"]
        )
        command, environment = _child_command(
            arguments, backend="dynamic", output=Path("out.json")
        )
        self.assertIn("live-child", command)
        self.assertEqual(environment["B12X_STATIC_COMPACT_CUTOVER_PAIRS"], "1")
        self.assertEqual(environment["PYTHONHASHSEED"], "0")


if __name__ == "__main__":
    unittest.main()


def test_live_source_binding_rejects_a_different_imported_tree(tmp_path, monkeypatch):
    import pytest
    from types import SimpleNamespace
    from performance.harnesses.moe_round_floor import b12x_floor_benchmark as bench
    payload = b"pinned source"
    audited = tmp_path / "audited" / "module.py"
    imported = tmp_path / "imported" / "module.py"
    for path in (audited, imported):
        path.parent.mkdir()
        path.write_bytes(payload)
    monkeypatch.setattr(bench, "PINNED_SOURCES", {"fixture": ("b12x/module.py", hashlib.sha256(payload).hexdigest())})
    audit = {"fixture": {"path": str(audited)}}
    with pytest.raises(GateError, match="Imported B12X source differs"):
        bench.require_imported_b12x_sources(audit, importer=lambda name: SimpleNamespace(__file__=str(imported)))
    bench.require_imported_b12x_sources(audit, importer=lambda name: SimpleNamespace(__file__=str(audited)))
    audited.write_bytes(b"modified after audit")
    with pytest.raises(GateError, match="Imported B12X source differs"):
        bench.require_imported_b12x_sources(audit, importer=lambda name: SimpleNamespace(__file__=str(audited)))


def test_indexer_correctness_gates_survive_optimized_python():
    import ast
    import pytest
    source = Path(__file__).parents[1] / "indexer_barrier" / "gpu_barrier_probe.py"
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    # Execute the actual final gate statements without importing CUDA dependencies.
    gates = ast.Module(body=main.body[-3:], type_ignores=[])
    code = compile(ast.fix_missing_locations(gates), str(source), "exec", optimize=2)
    for original, fixed in ((0, 0), (1, 1)):
        with pytest.raises(RuntimeError):
            exec(code, {"results": [{"incomplete_reads": original}, {"incomplete_reads": fixed}], "print": lambda *a, **k: None})
    exec(code, {"results": [{"incomplete_reads": 1}, {"incomplete_reads": 0}], "print": lambda *a, **k: None})
