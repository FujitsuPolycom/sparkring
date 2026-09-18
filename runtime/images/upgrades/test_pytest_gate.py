"""Protected node selection does not permit arbitrary pytest arguments."""

import pytest

from .pytest_gate import protected_test_path
from .pytest_gate import oracle_environment


def test_protected_tests_inspect_selected_runtime_not_reference(monkeypatch):
    monkeypatch.setenv('SPARKRING_TEST_SOURCE_ROOT', '/wrong-runtime')
    env = oracle_environment('/candidate', '/overlay', '/protected-baseline')
    assert env['SPARKRING_TEST_SOURCE_ROOT'] == '/candidate'
    assert '/protected-baseline' in env['PYTHONPATH']
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'


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
