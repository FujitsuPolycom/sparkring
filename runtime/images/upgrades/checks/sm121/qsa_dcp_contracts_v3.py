"""QSA DCP source admission: baseline-derived safety plus executed CPU geometry.

The immutable 09.3 baseline and frozen v2 oracle remain inputs. Reviewed DCP
transformations are expressed here independently of candidate bytes. Native
arithmetic, CUDA compilation, memory ordering and serving need separate gates.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

CHECKS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "_frozen_qsa_v2", CHECKS / "kraken/qsa_release3_contracts_v2.py"
)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
ROOT, BASE = legacy.ROOT, legacy.BASE

# Execute the frozen adversarial row-load and native-program ownership tests.
test_production_raw_ring_guards = legacy.test_production_raw_ring_guards
test_raw_ring_behavior_rejects_released_unchecked_loads = (
    legacy.test_raw_ring_behavior_rejects_released_unchecked_loads
)
test_support_keys_distinguish_shared_pool_variant = (
    legacy.test_support_keys_distinguish_shared_pool_variant
)
test_released_support_keys_alias_the_shared_pool_variant = (
    legacy.test_released_support_keys_alias_the_shared_pool_variant
)
test_selector_program_ownership = legacy.test_selector_program_ownership
test_selector_ownership_mutations_fail = legacy.test_selector_ownership_mutations_fail


def nodes(text):
    return {
        n.name: n
        for n in ast.parse(text).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }


def statements(text):
    return ast.parse(text).body


def replace_statement(function, target, replacement, *, occurrence=0):
    """Replace one baseline assignment, rejecting absent/ambiguous migrations."""
    found = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == target for t in n.targets)
    ]
    assert len(found) > occurrence, (function.name, target)
    original = found[occurrence]

    class Replace(ast.NodeTransformer):
        def visit_Assign(self, node):
            return (
                statements(replacement)
                if node is original
                else self.generic_visit(node)
            )

    Replace().visit(function)


def add_dcp_args(function, before):
    index = next(i for i, arg in enumerate(function.args.args) if arg.arg == before)
    function.args.args[index:index] = [
        ast.arg(
            arg=name,
            annotation=ast.Attribute(
                value=ast.Name(id="tl", ctx=ast.Load()),
                attr="constexpr",
                ctx=ast.Load(),
            ),
        )
        for name in ("DCP_SIZE", "DCP_RANK", "CP_INTERLEAVE")
    ]


def reviewed_kernels(baseline):
    """Retain every baseline kernel; admit only stated DCP ownership changes."""
    funcs = legacy.definitions(legacy.reviewed_kernel_tree(ast.parse(baseline)))
    page = funcs["_validate_page_tables_kernel"]
    add_dcp_args(page, "BLOCK_BATCH")
    replace_statement(
        page,
        "main_pages",
        """
local_tokens = _local_dcp_count(sequence_length, DCP_SIZE, DCP_RANK, CP_INTERLEAVE)
main_pages = tl.cdiv(local_tokens, MAIN_PAGE_SIZE)
if (page_block == 0) & real_request & (main_pages > MAIN_TABLE_WIDTH):
    tl.atomic_or(state_errors + row, 2048)
""",
    )
    replace_statement(
        page,
        "main_active",
        "main_active = real_request & (pages < main_pages) & (pages < MAIN_TABLE_WIDTH)",
    )
    replace_statement(
        page,
        "completed_groups",
        """
completed_groups = sequence_length // COMPRESS_RATIO
completed_groups = _local_dcp_count(completed_groups, DCP_SIZE, DCP_RANK, CP_INTERLEAVE // COMPRESS_RATIO)
""",
    )
    replace_statement(
        page,
        "compressed_pages",
        """
compressed_pages = tl.cdiv(completed_groups, COMPRESSED_PAGE_SIZE)
if (page_block == 0) & real_request & (compressed_pages > COMPRESSED_TABLE_WIDTH):
    tl.atomic_or(state_errors + row, 1024)
""",
    )
    replace_statement(
        page,
        "compressed_active",
        "compressed_active = real_request & (pages < compressed_pages) & (pages < COMPRESSED_TABLE_WIDTH)",
    )
    mark = funcs["_mark_live_compressed_pages_kernel"]
    add_dcp_args(mark, "BLOCK_P")
    replace_statement(
        mark,
        "completed_groups",
        """
completed_groups = tl.where(valid_length, sequence_length // COMPRESS_RATIO, 0)
completed_groups = _local_dcp_count(completed_groups, DCP_SIZE, DCP_RANK, CP_INTERLEAVE // COMPRESS_RATIO)
""",
    )
    replace_statement(
        mark,
        "live_pages",
        """
live_pages = tl.cdiv(completed_groups, COMPRESSED_PAGE_SIZE)
if (page_block == 0) & valid_length & (live_pages > COMPRESSED_TABLE_WIDTH):
    tl.atomic_or(occupancy + table_error_offset, 1)
""",
    )
    compress = funcs["_compress_completed_groups_kernel"]
    add_dcp_args(compress, "POSITION_AXES")
    replace_statement(
        compress,
        "complete",
        """
group_id = position // COMPRESS_RATIO
if DCP_SIZE == 1:
    complete = ((position + 1) % COMPRESS_RATIO) == 0
else:
    group_interleave = CP_INTERLEAVE // COMPRESS_RATIO
    dcp_round = DCP_SIZE * group_interleave
    owner = (group_id // group_interleave) % DCP_SIZE
    complete = (((position + 1) % COMPRESS_RATIO) == 0) & (owner == DCP_RANK)
""",
    )
    replace_statement(
        compress,
        "group_id",
        """
if DCP_SIZE == 1:
    local_group = group_id
else:
    local_group = ((group_id // dcp_round) * group_interleave + group_id % group_interleave)
""",
        occurrence=1,
    )
    replace_statement(
        compress, "logical_page", "logical_page = local_group // COMPRESSED_PAGE_SIZE"
    )
    replace_statement(
        compress, "page_offset", "page_offset = local_group % COMPRESSED_PAGE_SIZE"
    )
    score = funcs["_score_representatives_kernel"]
    add_dcp_args(score, "BLOCK_G")
    replace_statement(
        score,
        "eligible",
        """
global_eligible = tl.minimum((position + 1) // COMPRESS_RATIO, sequence_length // COMPRESS_RATIO)
if DCP_SIZE == 1:
    eligible = global_eligible
else:
    group_interleave = CP_INTERLEAVE // COMPRESS_RATIO
    dcp_round = DCP_SIZE * group_interleave
    complete_rounds = global_eligible // dcp_round
    remainder = global_eligible - complete_rounds * dcp_round
    rank_remainder = tl.minimum(tl.maximum(remainder - DCP_RANK * group_interleave, 0), group_interleave)
    eligible = complete_rounds * group_interleave + rank_remainder
""",
    )
    expand = funcs["_expand_selected_groups_kernel"]
    add_dcp_args(expand, "BLOCK_W")
    replace_statement(
        expand,
        "tail_length",
        """
if DCP_SIZE == 1:
    local_tail_start = tail_start
    tail_length = position + 1 - tail_start
else:
    global_tail_group = tail_start // COMPRESS_RATIO
    group_interleave = CP_INTERLEAVE // COMPRESS_RATIO
    dcp_round = DCP_SIZE * group_interleave
    tail_owner = (global_tail_group // group_interleave) % DCP_SIZE
    local_tail_group = (global_tail_group // dcp_round) * group_interleave + global_tail_group % group_interleave
    local_tail_start = local_tail_group * COMPRESS_RATIO
    tail_length = tl.where(tail_owner == DCP_RANK, position + 1 - tail_start, 0)
""",
    )
    replace_statement(
        expand,
        "result",
        "result = tl.where(columns < expanded_count, expanded, tl.where(in_tail, local_tail_start + tail_column, -1))",
    )
    return funcs


def verify(candidate, baseline):
    for name, digest in legacy.BASE_HASHES.items():
        assert hashlib.sha256(baseline[name]).hexdigest() == digest, (
            "baseline identity",
            name,
        )
    # The admitted Windows export changes only LF to CRLF in this untouched file.
    # Keep the immutable LF baseline hash and require byte equality after only
    # that one transport normalization; no code or whitespace rewrite is allowed.
    assert (
        candidate["_stable_select_cute.py"].replace(b"\r\n", b"\n")
        == baseline["_stable_select_cute.py"]
    ), "selector identity"
    expected = reviewed_kernels(baseline["_kernels.py"])
    actual = nodes(candidate["_kernels.py"])
    assert {name for name in actual if name.endswith("_kernel")} == legacy.KERNELS | {
        "_expand_global_selected_groups_kernel"
    }
    for name in legacy.KERNELS:
        assert ast.dump(actual[name]) == ast.dump(expected[name]), (
            "unreviewed safety kernel",
            name,
        )
    for name in (
        "launch_validate_page_tables",
        "launch_validate_shared_pool_ownership",
        "launch_compress_completed_groups",
        "launch_score_representatives",
        "launch_expand_selected_groups",
    ):
        launcher = copy.deepcopy(actual[name])
        calls = [
            n
            for n in ast.walk(launcher)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_launch_triton"
        ]
        forwarded = 0
        for call in calls:
            added = {
                k.arg: ast.unparse(k.value)
                for k in call.keywords
                if k.arg in {"DCP_SIZE", "DCP_RANK", "CP_INTERLEAVE"}
            }
            if added:
                assert added == {
                    "DCP_SIZE": "int(getattr(caps, 'dcp_size', 1))",
                    "DCP_RANK": "int(getattr(caps, 'dcp_rank', 0))",
                    "CP_INTERLEAVE": "int(getattr(caps, 'cp_kv_cache_interleave_size', 1))",
                }
                forwarded += 1
                call.keywords = [k for k in call.keywords if k.arg not in added]
        assert forwarded == 1, name
        assert ast.dump(launcher) == ast.dump(expected[name]), (
            "unreviewed support launch",
            name,
        )
    assert ast.dump(actual["_support_kernel_key"]) == ast.dump(
        expected["_support_kernel_key"]
    ), "support key identity"
    contract = nodes(candidate["_contract.py"])
    compiler = contract["compile_qsa"]
    block = legacy.support_context_block(compiler)
    for reference in statements(legacy.SHARED_COMPILE_LAUNCHES):
        assert any(ast.dump(n) == ast.dump(reference) for n in block.body), (
            "shared pool prepared program missing/changed"
        )
    decode = contract["_qsa_decode_impl"]
    calls = sorted(
        (n.lineno, n)
        for n in ast.walk(decode)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id.startswith("launch_")
    )
    assert [n.func.id for _, n in calls[:5]] == [
        "launch_validate_rows",
        "launch_validate_page_tables",
        "launch_validate_shared_pool_ownership",
        "launch_validate_completed_groups",
        "launch_propagate_request_errors",
    ]
    assert calls[-1][1].func.id == "launch_poison_failed_rows"
    for _, call in calls:
        assert any(k.arg == "_prepared" for k in call.keywords), call.func.id
    # Atomic validation and alias checks must still precede the opaque mutation.
    for name in ("_qsa_decode_op", "_qsa_decode_shared_op"):
        text = ast.unparse(contract[name])
        assert text.index("_require_runtime_abi(") < text.index("_qsa_decode_impl(")
        assert text.index("_require_mutation_alias_contract(") < text.index(
            "_qsa_decode_impl("
        )


def test_baseline_derived_safety_and_shared_preparation():
    verify(legacy.inputs(ROOT), legacy.inputs(BASE))


@pytest.mark.parametrize(
    "mutation",
    [
        "remove-kernel",
        "change-kernel",
        "extra-kernel",
        "raw-ring",
        "table-mask",
        "local-count",
        "shared-key",
        "selector",
        "transaction",
        "prepared",
        "shared-compile",
        "baseline",
    ],
)
def test_effective_safety_mutations_are_rejected(mutation):
    candidate, baseline = legacy.inputs(ROOT), legacy.inputs(BASE)
    tree = ast.parse(candidate["_kernels.py"])
    funcs = legacy.definitions(tree)
    if mutation == "remove-kernel":
        tree.body.remove(funcs["_copy_stable_topk_kernel"])
    elif mutation == "extra-kernel":
        tree.body += statements("def _extra_kernel(): pass")
    elif mutation == "change-kernel":
        funcs["_poison_failed_rows_kernel"].body.append(ast.Pass())
    elif mutation == "raw-ring":
        replace_statement(
            funcs["_commit_raw_ring_kernel"],
            "active",
            "active = suffix_offset < suffix_length",
        )
    elif mutation == "table-mask":
        replace_statement(
            funcs["_validate_page_tables_kernel"],
            "main_active",
            "main_active = real_request & (pages < main_pages)",
        )
    elif mutation == "local-count":
        replace_statement(
            funcs["_validate_page_tables_kernel"],
            "local_tokens",
            "local_tokens = sequence_length",
        )
    elif mutation == "shared-key":
        legacy.without_shared_key_suffix(tree)
    elif mutation == "selector":
        candidate["_stable_select_cute.py"] = b""
    elif mutation == "baseline":
        baseline["_kernels.py"] += b"\n# changed\n"
    else:
        contract = ast.parse(candidate["_contract.py"])
        name = {
            "transaction": "launch_validate_rows",
            "prepared": "launch_compress_completed_groups",
            "shared-compile": "launch_validate_shared_pool_ownership",
        }[mutation]
        scope = nodes(candidate["_contract.py"])
        owner = scope[
            "compile_qsa" if mutation == "shared-compile" else "_qsa_decode_impl"
        ]
        # Select the corresponding real node from the mutable module.
        owner = next(n for n in contract.body if getattr(n, "name", None) == owner.name)
        call = next(
            n
            for n in ast.walk(owner)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == name
        )
        if mutation == "prepared":
            call.keywords = [k for k in call.keywords if k.arg != "_prepared"]
        else:
            call.func.id = "removed_validation"
        candidate["_contract.py"] = ast.unparse(contract).encode()
    candidate["_kernels.py"] = ast.unparse(tree).encode()
    with pytest.raises(AssertionError):
        verify(candidate, baseline)


class HostCasts(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        return (
            node.func.value
            if isinstance(node.func, ast.Attribute) and node.func.attr == "to"
            else node
        )


class Pointer:
    def __init__(self, values, offset=0):
        self.values = np.asarray(values).reshape(-1)
        self.offset = np.asarray(offset, dtype=np.int64)

    def __add__(self, offset):
        return Pointer(self.values, self.offset + offset)


class HostTL:
    """Bounds-check every active read/write while executing source kernel bodies."""

    int64 = np.int64
    int32 = np.int32

    def __init__(self, pid=(0, 0)):
        self.pid = pid

    def program_id(self, axis):
        return np.int64(self.pid[axis])

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets, masks = np.broadcast_arrays(
            pointer.offset, np.asarray(mask, dtype=bool)
        )
        active = offsets[masks]
        assert np.all((active >= 0) & (active < pointer.values.size)), (
            "out-of-bounds read"
        )
        out = np.full(offsets.shape, other)
        out[masks] = pointer.values[active]
        return out[()] if out.ndim == 0 else out

    @staticmethod
    def store(pointer, value, mask=True):
        offsets, values, masks = np.broadcast_arrays(
            pointer.offset, value, np.asarray(mask, dtype=bool)
        )
        active = offsets[masks]
        assert np.all((active >= 0) & (active < pointer.values.size)), (
            "out-of-bounds write"
        )
        pointer.values[active] = values[masks]

    @staticmethod
    def atomic_or(pointer, value):
        assert np.all((pointer.offset >= 0) & (pointer.offset < pointer.values.size))
        np.bitwise_or.at(pointer.values, pointer.offset, value)

    @staticmethod
    def atomic_xchg(pointer, value, mask=True):
        HostTL.store(pointer, value, mask)

    arange = staticmethod(np.arange)
    minimum = staticmethod(np.minimum)
    maximum = staticmethod(np.maximum)
    sum = staticmethod(np.sum)
    where = staticmethod(np.where)
    full = staticmethod(np.full)
    cdiv = staticmethod(lambda a, b: (a + b - 1) // b)


def host_functions(filename, names, tl):
    functions = [
        copy.deepcopy(nodes((ROOT / filename).read_text())[name]) for name in names
    ]
    for f in functions:
        f.decorator_list = []
        for arg in f.args.args:
            arg.annotation = None
    tree = HostCasts().visit(ast.Module(body=functions, type_ignores=[]))
    env = {"tl": tl}
    exec(compile(ast.fix_missing_locations(tree), filename, "exec"), env)
    return env


def reference_count(length, size, rank, interleave):
    return sum((p // interleave) % size == rank for p in range(length))


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("interleave", [4, 8, 32])
def test_local_counts_match_independent_stripe_enumeration(size, interleave):
    env = host_functions("_kernels.py", ["_local_dcp_count"], HostTL())
    for rank in range(size):
        for count in range(0, 260):
            assert env["_local_dcp_count"](
                count, size, rank, interleave
            ) == reference_count(count, size, rank, interleave)


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize(
    "case",
    ["valid", "bad-main", "bad-compressed", "undersized", "alias", "prior-error"],
)
def test_actual_page_validator_bounds_and_shared_pool(size, case):
    for rank in range(size):
        for length in (1, 4, 31, 64, 129, 259):
            local = reference_count(length, size, rank, 4)
            groups = reference_count(length // 4, size, rank, 1)
            main_count, compressed_count = math.ceil(local / 16), math.ceil(groups / 4)
            main_width, compressed_width = max(1, main_count), max(1, compressed_count)
            if case == "undersized":
                main_width = max(1, main_width - 1)
                compressed_width = max(1, compressed_width - 1)
            main = Pointer(np.full(main_width, -1, dtype=np.int64))
            compressed = Pointer(np.full(compressed_width, -1, dtype=np.int64))
            main.values[:main_count] = 0
            compressed.values[:compressed_count] = 0
            if case == "bad-main" and main_count:
                main.values[0] = 2
            if case == "bad-compressed" and compressed_count:
                compressed.values[0] = 2
            errors = Pointer(np.array([8 if case == "prior-error" else 0]))
            tl = HostTL()
            env = host_functions(
                "_kernels.py", ["_local_dcp_count", "_validate_page_tables_kernel"], tl
            )
            for page in range(math.ceil(max(main_width, compressed_width) / 32)):
                tl.pid = (0, page)
                env["_validate_page_tables_kernel"](
                    Pointer([0]),
                    Pointer([length]),
                    main,
                    compressed,
                    Pointer([0 if case == "alias" else 1]),
                    errors,
                    main_width,
                    compressed_width,
                    1,
                    2,
                    2,
                    main_width,
                    compressed_width,
                    16,
                    4,
                    4,
                    1,
                    2,
                    True,
                    size,
                    rank,
                    4,
                    1,
                    32,
                )
            expected = 8 if case == "prior-error" else 0
            if case == "bad-main" and main_count:
                expected = 2048
            if case == "bad-compressed" and compressed_count:
                expected = 1024
            if case == "undersized":
                expected = (2048 if main_count > main_width else 0) | (
                    1024 if compressed_count > compressed_width else 0
                )
            if case == "alias" and compressed_count:
                expected = 4096
            assert errors.values.tolist() == [expected], (size, rank, length, case)


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize(
    "case",
    ["valid", "invalid-page", "undersized", "negative-length", "oversized-length"],
)
def test_idle_live_owner_occupancy_uses_local_completed_groups(size, case):
    for rank in range(size):
        length = {"negative-length": -1, "oversized-length": 1025}.get(case, 259)
        count = math.ceil(
            reference_count(max(0, min(length, 1024)) // 4, size, rank, 1) / 4
        )
        width = max(1, count - int(case == "undersized"))
        table = Pointer(np.full(width, -1, dtype=np.int64))
        table.values[:count] = 0
        if case == "invalid-page" and count:
            table.values[0] = 2
        occupancy = Pointer([0, 0, 0])
        tl = HostTL()
        env = host_functions(
            "_kernels.py",
            ["_local_dcp_count", "_mark_live_compressed_pages_kernel"],
            tl,
        )
        for page in range(math.ceil(width / 32)):
            tl.pid = (0, page)
            env["_mark_live_compressed_pages_kernel"](
                Pointer([length]),
                table,
                occupancy,
                2,
                width,
                2,
                1024,
                width,
                4,
                4,
                size,
                rank,
                4,
                32,
            )
        expected_error = int(case != "valid")
        assert int(occupancy.values[2]) == expected_error
        assert int(occupancy.values[1]) == 0
        if case == "valid":
            assert int(occupancy.values[0]) == int(count > 0)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("interleave", [4, 8, 32])
def test_global_selection_expands_only_owned_groups_and_causal_tail(size, interleave):
    for rank in range(size):
        for position in (0, 3, 4, 7, 31, 32, 66, 257):
            groups = [0, 3, -1, 8]
            width = 4 * len(groups) + 3
            selected = Pointer(np.full(width, -99, dtype=np.int64))
            env = host_functions(
                "_kernels.py", ["_expand_global_selected_groups_kernel"], HostTL()
            )
            env["_expand_global_selected_groups_kernel"](
                Pointer(groups),
                Pointer([position]),
                selected,
                len(groups),
                width,
                len(groups),
                4,
                width,
                size,
                rank,
                interleave,
                32,
            )
            expected = []
            for group in groups:
                for token in range(group * 4, group * 4 + 4):
                    owned = token >= 0 and (token // interleave) % size == rank
                    local = (
                        token // (interleave * size)
                    ) * interleave + token % interleave
                    expected.append(local if owned else -1)
            start = (position + 1) // 4 * 4
            for token in range(start, start + 3):
                owned = token <= position and (token // interleave) % size == rank
                local = (token // (interleave * size)) * interleave + token % interleave
                expected.append(local if owned else -1)
            assert selected.values.tolist() == expected


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize(
    "source,position,error",
    [(0, 8, 0), (0, 10, 0), (0, 11, 0), (-1, 8, 0), (2, 8, 0), (0, 8, 512)],
)
def test_draft_selection_bounds_errors_and_rank_local_tail(
    size, source, position, error
):
    for rank in range(size):
        width, tail = 4, 3
        selected, errors = Pointer(np.zeros(width + tail, dtype=np.int64)), Pointer([0])
        env = host_functions("_draft_selection.py", ["_prepare_kernel"], HostTL())
        env["_prepare_kernel"](
            Pointer([7]),
            Pointer([error]),
            Pointer([0, 1, -1, -1]),
            Pointer([source]),
            Pointer([1]),
            Pointer([0]),
            Pointer([position]),
            selected,
            errors,
            1,
            1,
            1,
            width,
            tail,
            size,
            rank,
            4,
            8,
        )
        valid = source == 0 and 8 <= position < 11
        wanted = [0, 1, -1, -1] if valid else [-1] * width
        for token in range(8, 11):
            owned = size == 1 or (token // 4) % size == rank
            local = (token // (4 * size)) * 4 + token % 4
            wanted.append(local if valid and token <= position and owned else -1)
        assert selected.values.tolist() == wanted
        assert errors.values.tolist() == [
            (error if source == 0 else 1) | (0 if valid else 1)
        ]


def test_both_draft_reuse_entrypoints_preserve_plan_and_error_offsets():
    contract = nodes((ROOT / "_contract.py").read_text())
    for name in ("run", "attend_reuse"):
        calls = [
            n
            for n in ast.walk(contract[name])
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "prepare_selection"
        ]
        assert len(calls) == 1
        assert ast.unparse(calls[0].args[0]) == "binding.plan.handle"
        assert ast.unparse(calls[0].args[9]) == "0"
        assert [ast.unparse(a) for a in calls[0].args[-3:]] == [
            "caps.dcp_size",
            "caps.dcp_rank",
            "caps.cp_kv_cache_interleave_size",
        ]


def test_score_and_draft_preserve_baseline_math_with_dcp_geometry():
    """Only stripe ownership changes; error gates, dot products and loads remain."""
    old = nodes((BASE / "_score_cute.py").read_text())
    new = nodes((ROOT / "_score_cute.py").read_text())
    assert ast.dump(old["_or_error"]) == ast.dump(new["_or_error"])
    assert (ROOT / "_score_pair_cute.py").read_text() == (
        BASE / "_score_pair_cute.py"
    ).read_text()
    expected = copy.deepcopy(old["_RepresentativeScoreKernel"])
    constructor = next(
        n for n in expected.body if getattr(n, "name", None) == "__init__"
    )
    constructor.args.args += [
        ast.arg(arg=name) for name in ("dcp_size", "dcp_rank", "cp_interleave")
    ]
    index = next(
        i
        for i, n in enumerate(constructor.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.lane_values"
    )
    constructor.body[index:index] = statements(
        "self.dcp_size = dcp_size\nself.dcp_rank = dcp_rank\nself.group_interleave = cp_interleave // ratio"
    )
    kernel = next(n for n in expected.body if getattr(n, "name", None) == "kernel")
    replace_statement(
        kernel,
        "eligible",
        """
global_eligible = cutlass.min(position_groups, sequence_groups).to(Int32)
eligible = global_eligible
if cutlass.const_expr(self.dcp_size > 1):
    round_width = Int32(self.dcp_size * self.group_interleave)
    complete_rounds = global_eligible // round_width
    remainder = global_eligible - complete_rounds * round_width
    rank_remainder = cutlass.min(cutlass.max(remainder - Int32(self.dcp_rank * self.group_interleave), Int32(0)), Int32(self.group_interleave))
    eligible = complete_rounds * Int32(self.group_interleave) + rank_remainder
eligible = cutlass.min(eligible, Int32(self.max_groups))
""",
        occurrence=1,
    )
    assert ast.dump(new["_RepresentativeScoreKernel"]) == ast.dump(expected)
    old = nodes((BASE / "_draft_selection.py").read_text())
    new = nodes((ROOT / "_draft_selection.py").read_text())
    assert ast.dump(old["_record_kernel"]) == ast.dump(new["_record_kernel"])
    expected = copy.deepcopy(old["_prepare_kernel"])
    add_dcp_args(expected, "BLOCK")
    replace_statement(
        expected,
        "value",
        """
if DCP_SIZE == 1:
    value = tl.where(column < WIDTH, original, tl.where(valid & (tail <= position), tail, -1))
else:
    stripe = tail // CP_INTERLEAVE
    owner = stripe % DCP_SIZE
    local_tail = (stripe // DCP_SIZE) * CP_INTERLEAVE + tail % CP_INTERLEAVE
    value = tl.where(column < WIDTH, original, tl.where(valid & (tail <= position) & (owner == DCP_RANK), local_tail, -1))
""",
    )
    assert ast.dump(new["_prepare_kernel"]) == ast.dump(expected)


def test_score_cache_separates_dcp_rank_geometry_and_gates_paired_kernel(monkeypatch):
    from contextlib import nullcontext
    import sys
    import types

    class Number:
        width = 16

        def __new__(cls, value):
            return value

    class Scalar:
        def __init__(self, *geometry):
            self.geometry = geometry

    class Paired(Scalar):
        pass

    module = types.ModuleType("score_oracle._score_pair_cute")
    module.PairedRepresentativeScoreKernel = Paired
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []
    env = dict(
        __package__="score_oracle",
        torch=NS(
            bfloat16="bf16",
            float32="f32",
            int32="i32",
            int64="i64",
            cuda=NS(device=lambda _: nullcontext()),
        ),
        BFloat16=Number,
        Float32=Number,
        Int32=Number,
        Int64=Number,
        _CACHE={},
        _RepresentativeScoreKernel=Scalar,
        raise_if_kernel_resolution_frozen=lambda *a, **k: None,
        make_ptr=lambda *a, **k: "pointer",
        cute=NS(AddressSpace=NS(gmem=1)),
        current_cuda_stream=lambda: 1,
        KernelCompileSpec=NS(from_key=lambda *args: args),
        b12x_compile=lambda *a, **k: calls.append((a, k)) or object(),
    )
    function = nodes((ROOT / "_score_cute.py").read_text())[
        "compile_score_representatives"
    ]
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "source_score_compiler",
            "exec",
        ),
        env,
    )
    args = {
        name: NS(dtype=dtype, device=NS(index=0))
        for name, dtype in (
            ("prepared_query", "bf16"),
            ("query_positions", "i64"),
            ("request_ids", "i32"),
            ("sequence_lengths", "i32"),
            ("compressed_cache", "bf16"),
            ("compressed_block_table", "i32"),
            ("state_errors", "i32"),
            ("scores", "f32"),
            ("eligible_counts", "i32"),
            ("merge_lengths", "i32"),
        )
    }
    caps = NS(
        index_heads=4,
        index_head_dim=128,
        compress_ratio=4,
        compressed_page_size=128,
        max_groups=65536,
        group_budget=512,
    )
    carriers = []
    for size, rank, interleave in ((1, 0, 1), (2, 0, 4), (2, 1, 4), (4, 3, 8)):
        caps.dcp_size, caps.dcp_rank, caps.cp_kv_cache_interleave_size = (
            size,
            rank,
            interleave,
        )
        first = env[function.name](**args, caps=caps)
        args["prepared_query"].shape = (8192, 4, 128)
        assert env[function.name](**args, caps=caps) is first
        assert first not in carriers
        carriers.append(first)
        kernel = calls[-1][0][0]
        assert kernel.geometry[-3:] == (size, rank, interleave)
        assert isinstance(kernel, Paired) is (size == 1)
    assert len(calls) == 4
