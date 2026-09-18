"""Compiler CPU placement changes only the spawned compiler's current thread."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def source():
    return Path(os.environ["SPARKRING_B12X_SOURCE_ROOT"]) / "b12x/_lib/compile_pool.py"


def functions(fake_os):
    path = source()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"_parse_compiler_cpu_affinity", "_configure_compiler_cpu_affinity"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    env = {"os": fake_os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env


@pytest.mark.parametrize("value,expected", [("12-19", set(range(12, 20))), ("0,2-4,9", {0, 2, 3, 4, 9}), (" 2, 4-5 ", {2, 4, 5}), ("2,2-3", {2, 3})])
def test_parse_explicit_cpu_list(value, expected):
    assert functions(NS())["_parse_compiler_cpu_affinity"](value) == expected


@pytest.mark.parametrize("value", ["", " ", "-1", "3-1", "1-", "1,,2", "1-2-3", "1.0", "x", "1,+2", "١"])
def test_reject_malformed_cpu_lists(value):
    with pytest.raises(ValueError, match="B12X_COMPILE_CPU_AFFINITY"):
        functions(NS())["_parse_compiler_cpu_affinity"](value)


def test_unset_does_not_require_or_touch_affinity_support():
    functions(NS(environ={}))["_configure_compiler_cpu_affinity"]()


def test_explicit_setting_requires_linux_affinity_support():
    with pytest.raises(RuntimeError, match="affinity"):
        functions(NS(environ={"B12X_COMPILE_CPU_AFFINITY": "12-19"}))["_configure_compiler_cpu_affinity"]()


@pytest.mark.parametrize("allowed", [{10, 12, 13}, {10}])
def test_constraint_failure_restores_previous_mask(allowed):
    current = {10}
    calls = []

    def set_affinity(pid, requested):
        nonlocal current
        assert pid == 0
        calls.append(set(requested))
        accepted = requested & allowed
        if not accepted:
            raise OSError("empty CPU intersection")
        current = accepted

    fake = NS(environ={"B12X_COMPILE_CPU_AFFINITY": "12-19"},
              sched_getaffinity=lambda pid: set(current), sched_setaffinity=set_affinity)
    with pytest.raises(RuntimeError, match="B12X_COMPILE_CPU_AFFINITY"):
        functions(fake)["_configure_compiler_cpu_affinity"]()
    assert current == {10}
    assert calls == [set(range(12, 20)), {10}]


def test_success_changes_only_current_compiler_thread():
    current = {10}
    parent = {10}
    calls = []

    def set_affinity(pid, requested):
        nonlocal current
        assert pid == 0
        calls.append(set(requested))
        current = set(requested)

    fake = NS(environ={"B12X_COMPILE_CPU_AFFINITY": "12-19"},
              sched_getaffinity=lambda pid: set(current), sched_setaffinity=set_affinity)
    functions(fake)["_configure_compiler_cpu_affinity"]()
    assert calls == [set(range(12, 20))]
    assert current == set(range(12, 20)) and parent == {10}


def test_affinity_configuration_is_owned_by_worker_initializer():
    tree = ast.parse(source().read_text(encoding="utf-8"))
    callers = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                       and call.func.id == "_configure_compiler_cpu_affinity"
                       for statement in node.body for call in ast.walk(statement))]
    assert [node.name for node in callers] == ["_initialize_worker"]
    worker = callers[0]
    configure = next(i for i, node in enumerate(worker.body)
                     if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Name)
                     and node.value.func.id == "_configure_compiler_cpu_affinity")
    torch_import = next(i for i, node in enumerate(worker.body)
                        if isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names))
    assert configure < torch_import
    assert any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
               and isinstance(call.func.value, ast.Name) and call.func.value.id == "torch"
               and call.func.attr == "set_num_threads" and [ast.literal_eval(arg) for arg in call.args] == [1]
               for call in ast.walk(worker))
