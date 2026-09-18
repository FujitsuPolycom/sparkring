"""Protected node selection does not permit arbitrary pytest arguments."""

import pytest

from .pytest_gate import protected_test_path
from .pytest_gate import oracle_environment
from .pytest_gate import bind_test_roots


def test_protected_tests_inspect_selected_runtime_not_reference(monkeypatch):
    monkeypatch.setenv('SPARKRING_TEST_SOURCE_ROOT', '/wrong-runtime')
    env = oracle_environment('/candidate', '/overlay', '/protected-baseline')
    assert env['SPARKRING_TEST_SOURCE_ROOT'] == '/candidate'
    assert '/protected-baseline' in env['PYTHONPATH']
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'


def test_ast_probe_uses_candidate_file_instead_of_its_own_baseline(tmp_path):
    from types import SimpleNamespace as NS
    baseline, candidate = tmp_path/'baseline', tmp_path/'candidate'
    baseline.mkdir()
    candidate.mkdir()
    (baseline/'behavior.py').write_text('baseline')
    (candidate/'behavior.py').write_text('candidate')
    module = NS(__file__=str(baseline/'test_behavior.py'), ROOT=baseline)
    bind_test_roots([NS(module=module)], [module.__file__], baseline, candidate)
    assert (module.ROOT/'behavior.py').read_text() == 'candidate'


def test_ast_probe_refuses_missing_module_or_unexpected_root(tmp_path):
    from types import SimpleNamespace as NS
    path = str(tmp_path/'test_behavior.py')
    with pytest.raises(ValueError, match='baseline ROOT'):
        bind_test_roots([], [path], tmp_path, tmp_path/'candidate')
    module = NS(__file__=path, ROOT=tmp_path/'elsewhere')
    with pytest.raises(ValueError, match='baseline ROOT'):
        bind_test_roots([NS(module=module)], [path], tmp_path, tmp_path/'candidate')


def test_exact_cpu_test_can_be_selected_from_mixed_hardware_module(tmp_path):
    path = tmp_path / "test_graph.py"
    path.write_text("def test_cpu(): pass\n")
    assert (
        protected_test_path(tmp_path, "test_graph.py::test_cpu")
        == str(path) + "::test_cpu"
    )


@pytest.mark.parametrize(
    "value", ["../test.py", "test_graph.py::", "test_graph.py::-k anything"]
)
def test_uncontained_or_commandlike_node_ids_are_rejected(tmp_path, value):
    (tmp_path / "test_graph.py").write_text("")
    with pytest.raises(ValueError):
        protected_test_path(tmp_path, value)
