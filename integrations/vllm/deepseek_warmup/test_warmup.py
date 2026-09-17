import ast
import difflib
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from collections import namedtuple

import numpy as np

import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sparkring_dspark_warmup_apply", ROOT / "apply.py")
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)
RELATIVE = implementation.RELATIVE
WARMUP_RELATIVE = implementation.WARMUP_RELATIVE
build = implementation.build
build_warmup = implementation.build_warmup
prepare_source = implementation.prepare_source
MANIFEST = json.loads((ROOT / "fixtures/source.json").read_text(encoding="utf-8"))


def fixture_source(relative):
    item = next(row for row in MANIFEST["files"] + MANIFEST.get("supporting_files", []) if row["path"] == relative)
    return gzip.decompress((ROOT / "fixtures" / item["fixture"]).read_bytes())


def planner(source):
    module = ast.parse(source)
    method = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)
                  and node.name == "_warmup_prepare_inputs_kernel")
    body = []
    for node in method.body[1:]:
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "target_query_len":
            break
        body.append(node)
    method.body = body + [ast.Return(value=ast.Name(id="target_query_lens", ctx=ast.Load()))]
    namespace = {}
    ast.fix_missing_locations(method)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<installed startup planner>", "exec"), namespace)
    return namespace[method.name]


def block(length, query):
    return min(256, 1 << (length + query - 1).bit_length())


def test_pinned_k5_baseline_exposes_reachable_bucket_gap():
    source = fixture_source(RELATIVE)
    settings = SimpleNamespace(_speculator_name="DSpark", num_query_per_req=5, dynamic_physical_depth=False, max_num_tokens=4096, max_num_reqs=32)
    lengths = planner(source)(settings)
    covered = {block(length, 5) for length in lengths}
    reachable = {block(length, 5) for length in range(1, 4097)}
    assert reachable - covered == {8, 32, 128}


@pytest.mark.parametrize("queries", [1, 2, 3, 5, 6, 7, 16, 32, 129, 256])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("budget", [64, 192, 4096])
def test_candidate_covers_feasible_tiles_with_positive_bounded_contexts(queries, dynamic, budget):
    source = build(fixture_source(RELATIVE))
    settings = SimpleNamespace(_speculator_name="DSpark", num_query_per_req=queries, dynamic_physical_depth=dynamic, max_num_tokens=budget, max_num_reqs=32)
    lengths = planner(source)(settings)
    assert lengths and all(1 <= length <= budget for length in lengths)
    for query in range(1, queries + 1) if dynamic else [queries]:
        assert {block(length, query) for length in lengths} == {block(length, query) for length in range(1, budget + 1)}


def test_k5_candidate_includes_short_and_nonaligned_representatives():
    settings = SimpleNamespace(_speculator_name="DSpark", num_query_per_req=5, dynamic_physical_depth=False, max_num_tokens=4096, max_num_reqs=32)
    lengths = planner(build(fixture_source(RELATIVE)))(settings)
    assert {1, 2, 12, 28, 60, 124}.issubset(lengths)


def test_runtime_kernel_is_unchanged():
    original = fixture_source(RELATIVE)
    def kernel(source):
        return ast.dump(next(node for node in ast.walk(ast.parse(source))
                             if isinstance(node, ast.FunctionDef) and node.name == "_prepare_dflash_inputs_kernel"))
    assert kernel(original) == kernel(build(original))


def test_source_drift_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        build(fixture_source(RELATIVE) + b"\n")


def test_dflash_planning_is_unchanged():
    source = fixture_source(RELATIVE)
    settings = SimpleNamespace(_speculator_name="DFlash", num_query_per_req=6, dynamic_physical_depth=False, max_num_tokens=4096)
    assert planner(source)(settings) == planner(build(source))(settings)


def startup_fragment(source):
    tree = ast.parse(source)
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "model_runner.verification_capacity_manager is not None"
                  and any(isinstance(child, ast.Attribute) and child.attr == "warmup_capacity_kernels" for child in ast.walk(node)))
    return compile(ast.Module(body=[branch], type_ignores=[]), "<installed startup branch>", "exec")


@pytest.mark.parametrize("method,capacity", [("dspark", False), ("dspark", True), ("mtp", False), ("dflash", False), (None, False)])
def test_startup_call_replay_is_dspark_only_and_uses_scratch_lane(method, capacity):
    source = fixture_source(WARMUP_RELATIVE)
    events = []
    @contextmanager
    def lane(index):
        events.append(("enter_lane", index))
        yield
        events.append(("exit_lane", index))
    speculator = SimpleNamespace(warmup_capacity_kernels=lambda: events.append("speculator_warmup"), wants_auto_sps_curve=False)
    manager = SimpleNamespace(warmup=lambda buffers: events.append("capacity_warmup")) if capacity else None
    runner = SimpleNamespace(verification_capacity_manager=manager, speculator=speculator if method else None,
                             speculative_config=SimpleNamespace(method=method) if method else None, input_buffers=object())
    namespace = {"model_runner": runner, "use_workspace_lane": lane}
    exec(startup_fragment(source), namespace)
    baseline = list(events)
    events.clear()
    exec(startup_fragment(build_warmup(source)), namespace)
    if method == "dspark" and not capacity:
        assert baseline == []
        assert events == [("enter_lane", 1), "speculator_warmup", ("exit_lane", 1)]
    else:
        assert events == baseline


def test_startup_source_drift_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        build_warmup(fixture_source(WARMUP_RELATIVE) + b"\n")


def test_fixture_hashes_and_patch_reproduce_source():
    patch = []
    for item, builder in zip(MANIFEST["files"], (build, build_warmup)):
        fixture = (ROOT / "fixtures" / item["fixture"]).read_bytes()
        assert hashlib.sha256(fixture).hexdigest() == item["fixture_sha256"]
        original = gzip.decompress(fixture)
        assert hashlib.sha256(original).hexdigest() == item["preimage_sha256"]
        result = builder(original)
        assert hashlib.sha256(result).hexdigest() == item["candidate_sha256"]
        patch.extend(difflib.unified_diff(original.decode().splitlines(keepends=True), result.decode().splitlines(keepends=True),
                                         fromfile="a/" + item["path"], tofile="b/" + item["path"], n=0))
    assert "".join(patch) == (ROOT / "dspark-prepare-warmup.patch").read_text(encoding="utf-8")


def install_fixture_tree(root):
    for item in MANIFEST["files"] + MANIFEST.get("supporting_files", []):
        target = root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fixture_source(item["path"]))


def test_source_preparation_does_not_mutate_input(tmp_path):
    source_root, output_root = tmp_path / "source", tmp_path / "output"
    install_fixture_tree(source_root)
    result = prepare_source(source_root, output_root)
    assert result["status"] == "research-only"
    for item in MANIFEST["files"]:
        assert (source_root / item["path"]).read_bytes() == fixture_source(item["path"])
        assert hashlib.sha256((output_root / item["path"]).read_bytes()).hexdigest() == item["candidate_sha256"]


def test_all_sources_are_checked_before_output_is_created(tmp_path):
    source_root, output_root = tmp_path / "source", tmp_path / "output"
    install_fixture_tree(source_root)
    (source_root / WARMUP_RELATIVE).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="does not match"):
        prepare_source(source_root, output_root)
    assert not output_root.exists()


def test_existing_output_is_preserved(tmp_path):
    source_root, output_root = tmp_path / "source", tmp_path / "output"
    install_fixture_tree(source_root)
    output_root.mkdir()
    sentinel = output_root / "existing.txt"
    sentinel.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_source(source_root, output_root)
    assert sentinel.read_text() == "preserve me"


def replay_prepare_helper(source, *, name="DSpark", max_reqs=32, max_tokens=4096, query_rows=5):
    """Execute the complete installed warmup method with CPU tensor stand-ins."""
    method = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)
                  and node.name == "_warmup_prepare_inputs_kernel")
    calls = []
    torch = SimpleNamespace(int32=np.int32, int64=np.int64,
                            arange=lambda *args, dtype, device: np.arange(*args, dtype=dtype),
                            zeros=lambda *shape, dtype, device: np.zeros(shape, dtype=dtype),
                            zeros_like=np.zeros_like)
    batch_type = namedtuple("InputBatch", "num_reqs num_scheduled_tokens positions query_start_loc idx_mapping")
    def capture(*args):
        batch = args[7]
        calls.append({"requests": batch.num_reqs, "target_length": int(batch.num_scheduled_tokens.max()),
                      "context_tokens": int(batch.num_scheduled_tokens.sum()), "query_rows": args[16],
                      "last_sampled_dtype": str(args[10].dtype), "next_prefill_dtype": str(args[11].dtype)})
    namespace = {"torch": torch, "np": np, "_DFlashInputBatch": batch_type, "prepare_dflash_inputs": capture}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<installed complete preparation warmup>", "exec"), namespace)
    settings = SimpleNamespace(_speculator_name=name, num_query_per_req=query_rows, dynamic_physical_depth=False,
                               draft_kv_cache_group_id=0, draft_kv_cache_group_ids=[0], max_num_reqs=max_reqs,
                               max_num_tokens=max_tokens, max_model_len=1048576, device="cpu", input_buffers=None,
                               block_tables=SimpleNamespace(slot_mappings=[None], input_block_tables=[None], kernel_block_sizes=[256]),
                               context_positions=None, _context_slot_mappings=[None], sample_indices=None, sample_pos=None,
                               sample_idx_mapping=None, num_cached_tokens=None, parallel_drafting_token_id=128799,
                               sample_from_anchor=name == "DSpark",
                               _speculative_steps_for_query_len=lambda rows: rows if name == "DSpark" else rows - 1)
    namespace[method.name](settings)
    return calls


def scalar_class(value):
    return "one" if value == 1 else "aligned16" if value % 16 == 0 else "ordinary"


def test_complete_helper_matches_runtime_sample_buffer_dtypes():
    original = fixture_source(RELATIVE)
    assert {row["last_sampled_dtype"] for row in replay_prepare_helper(original)} == {"int32"}
    calls = replay_prepare_helper(build(original))
    assert {row["last_sampled_dtype"] for row in calls} == {"int64"}
    assert {row["next_prefill_dtype"] for row in calls} == {"int32"}


def test_request_state_fixture_proves_runtime_dtypes():
    item = MANIFEST["supporting_files"][0]
    packed = (ROOT / "fixtures" / item["fixture"]).read_bytes()
    assert hashlib.sha256(packed).hexdigest() == item["fixture_sha256"]
    raw = gzip.decompress(packed)
    assert hashlib.sha256(raw).hexdigest() == implementation.REQ_STATES_SHA256
    assignments = {node.targets[0].attr: node.value for node in ast.walk(ast.parse(raw))
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)
                   and node.targets[0].attr in ("last_sampled_tokens", "next_prefill_tokens")}
    dtypes = {name: ast.unparse(next(keyword.value for keyword in value.keywords if keyword.arg == "dtype"))
              for name, value in assignments.items()}
    assert dtypes == {"last_sampled_tokens": "torch.int64", "next_prefill_tokens": "torch.int32"}


def test_request_state_drift_prevents_output(tmp_path):
    root, out = tmp_path / "source", tmp_path / "output"
    install_fixture_tree(root)
    (root / implementation.REQ_STATES_RELATIVE).write_bytes(b"changed buffer contract")
    with pytest.raises(ValueError, match="request-state source"):
        prepare_source(root, out)
    assert not out.exists()


@pytest.mark.parametrize("max_reqs", [1, 2, 5, 32])
@pytest.mark.parametrize("query_rows", [1, 5, 6, 7])
def test_complete_helper_covers_feasible_tiles_and_scalar_alignment(max_reqs, query_rows):
    budget = 192
    calls = replay_prepare_helper(build(fixture_source(RELATIVE)), max_reqs=max_reqs,
                                  max_tokens=budget, query_rows=query_rows)
    observed = {(block(row["target_length"], query_rows), scalar_class(row["context_tokens"])) for row in calls}
    feasible = {(block(length, query_rows), scalar_class(count * length))
                for length in range(1, budget + 1) for count in range(1, min(max_reqs, budget // length) + 1)}
    assert observed == feasible
    assert all(1 <= row["context_tokens"] <= budget for row in calls)


def test_complete_dflash_helper_calls_are_unchanged():
    original = fixture_source(RELATIVE)
    assert replay_prepare_helper(original, name="DFlash", query_rows=6) == replay_prepare_helper(build(original), name="DFlash", query_rows=6)
