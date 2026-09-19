"""Release3 QSA CPU preservation oracle, version 1; GPU qualification is separate.

Inputs are B12X checkout roots in SPARKRING_B12X_SOURCE_ROOT and
SPARKRING_ORACLE_BASELINE. Baseline byte identities remain immutable. The sole
admitted kernel change guards raw-ring row loads and passes the live row count
without specialization; all other module AST, including launch ownership,
must be preserved. This is not a replacement for the retained legacy oracle.
"""

import ast
import contextvars
import copy
import hashlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(os.environ["SPARKRING_B12X_SOURCE_ROOT"]) / "b12x/attention/qsa"
BASE = Path(os.environ["SPARKRING_ORACLE_BASELINE"]) / "b12x/attention/qsa"
BASE_HASHES = {
    "_kernels.py": "b035d2ef7a073720aec67f9e2da821ae5f6e28c07a94a95f817609526b96dde1",
    "_contract.py": "337e89aa53b4976bd4a6e328ba351d81d882de6c41bed7b6cc6cf9216e7937f4",
    "_stable_select_cute.py": "90de48e9fd3d69286bed26b593a09951f3ec1b2c7da8d2ce81c0cdd97bc68c8c",
}
KERNELS = {
    "_validate_packed_boundaries_kernel", "_validate_packed_requests_kernel",
    "_materialize_packed_row_errors_kernel", "_validate_page_tables_kernel",
    "_clear_shared_page_occupancy_kernel", "_mark_live_compressed_pages_kernel",
    "_validate_active_raw_slots_kernel", "_prepare_index_query_kernel",
    "_validate_completed_groups_kernel", "_accumulate_request_errors_kernel",
    "_broadcast_request_errors_kernel", "_compress_completed_groups_kernel",
    "_commit_raw_ring_kernel", "_score_representatives_kernel",
    "_stage_topk_carry_kernel", "_remap_topk_group_ids_kernel",
    "_copy_stable_topk_kernel", "_expand_selected_groups_kernel",
    "_poison_failed_rows_kernel",
}


def definitions(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def reviewed_kernel_tree(baseline):
    """Express the exact reviewed patch independently of the candidate bytes."""
    tree = copy.deepcopy(baseline)
    funcs = definitions(tree)
    kernel = funcs["_commit_raw_ring_kernel"]
    decorator = kernel.decorator_list[0]
    assert ast.unparse(decorator) == "triton.jit(do_not_specialize=['raw_key_row_stride'])"
    decorator.keywords[0].value.elts.append(ast.Constant("rows"))
    index = next(i for i, arg in enumerate(kernel.args.args) if arg.arg == "state_errors")
    kernel.args.args.insert(index + 1, ast.arg(arg="rows"))
    replacements = {
        "active": "(request_start >= 0) & (request_end <= rows) & (query_length > 0) & (suffix_offset < suffix_length)",
        "observed_request": "tl.load(request_ids + row, mask=active & (status == 0), other=-1).to(tl.int64)",
    }
    for name, expression in replacements.items():
        statements = [n for n in kernel.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert len(statements) == 1
        statements[0].value = ast.parse(expression, mode="eval").body
    launcher = funcs["launch_commit_raw_ring"]
    calls = [n for n in ast.walk(launcher) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_launch_triton"]
    assert len(calls) == 1
    call = calls[0]
    index = next(i for i, arg in enumerate(call.args)
                 if isinstance(arg, ast.Name) and arg.id == "state_errors")
    call.args.insert(index + 1, ast.parse("int(request_ids.shape[0])", mode="eval").body)
    return tree


def verify(candidate, baseline):
    for name, digest in BASE_HASHES.items():
        assert hashlib.sha256(baseline[name]).hexdigest() == digest, ("baseline identity", name)
    assert hashlib.sha256(candidate["_stable_select_cute.py"]).hexdigest() == BASE_HASHES["_stable_select_cute.py"], "selector identity"
    old = ast.parse(baseline["_kernels.py"])
    new = ast.parse(candidate["_kernels.py"])
    for tree in (old, new):
        assert {n for n in definitions(tree) if n.endswith("_kernel")} == KERNELS
    assert ast.dump(new) == ast.dump(reviewed_kernel_tree(old)), "unreviewed kernel or launcher change"
    assert ast.dump(ast.parse(candidate["_contract.py"])) == ast.dump(ast.parse(baseline["_contract.py"])), "contract changed"


def inputs(root):
    return {name: (root / name).read_bytes() for name in BASE_HASHES}


def test_release3_preservation_and_selector_identity():
    verify(inputs(ROOT), inputs(BASE))


@pytest.mark.parametrize("mutation", ["remove-kernel", "change-kernel", "extra-kernel", "revert-raw-ring", "remove-selector", "change-selector", "change-contract", "change-baseline"])
def test_preservation_rejects_effective_mutations(mutation):
    candidate, baseline = inputs(ROOT), inputs(BASE)
    before = (candidate.copy(), baseline.copy())
    tree = ast.parse(candidate["_kernels.py"])
    funcs = definitions(tree)
    if mutation == "remove-kernel":
        tree.body.remove(funcs["_copy_stable_topk_kernel"])
    elif mutation == "change-kernel":
        funcs["_poison_failed_rows_kernel"].body.append(ast.Pass())
    elif mutation == "extra-kernel":
        tree.body.extend(ast.parse("def _unexpected_kernel(): pass").body)
    elif mutation == "revert-raw-ring":
        tree = ast.parse(baseline["_kernels.py"])
    elif mutation == "remove-selector":
        candidate["_stable_select_cute.py"] = b""
    elif mutation == "change-selector":
        candidate["_stable_select_cute.py"] += b"\n# mutation\n"
    elif mutation == "change-contract":
        candidate["_contract.py"] += b"\nunreviewed = True\n"
    else:
        baseline["_kernels.py"] += b"\n# mutation\n"
    if mutation in {"remove-kernel", "change-kernel", "extra-kernel", "revert-raw-ring"}:
        candidate["_kernels.py"] = ast.unparse(tree).encode()
    assert (candidate, baseline) != before, "mutation must change an existing input"
    with pytest.raises(AssertionError):
        verify(candidate, baseline)


def guard_result(tree, starts, rows, suffix, status):
    kernel = definitions(tree)["_commit_raw_ring_kernel"]
    prefix = []
    for statement in kernel.body:
        prefix.append(copy.deepcopy(statement))
        if isinstance(statement, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "real_request" for t in statement.targets
        ):
            break
    else:
        raise AssertionError("production load guard missing")

    class HostCasts(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            return node.func.value if isinstance(node.func, ast.Attribute) and node.func.attr == "to" else node

    class Pointer:
        def __init__(self, values, kind, offset=0):
            self.values, self.kind, self.offset = values, kind, offset

        def __add__(self, offset):
            return Pointer(self.values, self.kind, self.offset + offset)

    def load(pointer, mask=True, other=None):
        if not mask:
            return other
        assert 0 <= pointer.offset < len(pointer.values), "out-of-bounds row load"
        if pointer.kind == "ids":
            assert status == 0, "errored request dereferenced"
        return pointer.values[pointer.offset]

    env = dict(tl=NS(program_id=lambda axis: 0 if axis == 0 else suffix, load=load, minimum=min),
               query_start_loc=Pointer(starts, "bounds"), state_errors=Pointer([status] * rows, "status"),
               request_ids=Pointer([0] * rows, "ids"), rows=rows, RING_CAPACITY=8)
    module = HostCasts().visit(ast.Module(body=prefix, type_ignores=[]))
    exec(compile(ast.fix_missing_locations(module), "production_raw_ring_guard", "exec"), env)
    return bool(env["real_request"])


@pytest.mark.parametrize("starts,rows,suffix,status,expected", [
    ((0, 3), 2, 2, 1, False), ((3, 4), 2, 0, 1, False),
    ((-1, 1), 2, 0, 1, False), ((1, 0), 2, 0, 1, False),
    ((0, 0), 2, 0, 0, False), ((0, 2), 2, 0, 1, False),
    ((0, 2), 2, 0, 0, True), ((0, 1), 1, 0, 0, True),
    ((0, 0), 0, 0, 0, False), ((0, 3), 2, 2, 0, False),
    ((0, 16), 16, 7, 0, True), ((0, 16), 16, 8, 0, False),
])
def test_production_raw_ring_guards(starts, rows, suffix, status, expected):
    assert guard_result(ast.parse((ROOT / "_kernels.py").read_text()), starts, rows, suffix, status) is expected


@pytest.mark.parametrize("starts,rows,suffix,status", [((3, 4), 2, 0, 1), ((-1, 1), 2, 0, 1), ((0, 2), 2, 0, 1)])
def test_raw_ring_behavior_rejects_released_unchecked_loads(starts, rows, suffix, status):
    with pytest.raises(AssertionError):
        guard_result(ast.parse((BASE / "_kernels.py").read_text()), starts, rows, suffix, status)


def exercise_ownership(monkeypatch, mode, mutate=False):
    context = contextvars.ContextVar("selection", default=None)
    programs, calls, raw = {}, [], object()
    if mode == "compile":
        context.set((programs, True))
    else:
        programs["stable_selection"] = raw
        context.set((programs, False))
    module = types.ModuleType("qsa_oracle._stable_select_cute")
    module.launch_stable_selection = lambda **kw: calls.append(kw) or raw
    monkeypatch.setitem(sys.modules, module.__name__, module)
    function = definitions(ast.parse((ROOT / "_kernels.py").read_text()))["launch_stabilize_topk"]
    if mutate:
        class RemoveOwnership(ast.NodeTransformer):
            def visit_Assign(self, node):
                if mode == "compile" and ast.unparse(node.targets[0]) == "context[0]['stable_selection']":
                    return ast.Pass()
                if mode == "prepared" and ast.unparse(node.targets[0]) == "prepared":
                    node.value = ast.Constant(None)
                return node
        before = ast.dump(function)
        function = RemoveOwnership().visit(function)
        if ast.dump(function) == before:
            raise ValueError("ownership mutation was ineffective")
    env = dict(__package__="qsa_oracle", _support_launch_context=context,
               torch=NS(Tensor=object), triton=NS(next_power_of_2=lambda n: n),
               _launch_triton=lambda *a, **kw: calls.append((a, kw)), _copy_stable_topk_kernel=object())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "selector_ownership", "exec"), env)
    kwargs = {a.arg: NS(shape=(9, 4608)) for a in function.args.kwonlyargs}
    kwargs.update(group_budget=512, group_offset=4096)
    env[function.name](**kwargs)
    assert len(calls) == 2
    assert calls[0]["prepared"] is (raw if mode == "prepared" else None)
    assert programs["stable_selection"] is raw


@pytest.mark.parametrize("mode", ["compile", "prepared"])
def test_selector_program_ownership(monkeypatch, mode):
    exercise_ownership(monkeypatch, mode)


@pytest.mark.parametrize("mode", ["compile", "prepared"])
def test_selector_ownership_mutations_fail(monkeypatch, mode):
    with pytest.raises((AssertionError, KeyError)):
        exercise_ownership(monkeypatch, mode, mutate=True)
