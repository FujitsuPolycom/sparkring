"""CPU-only checks for the FlashInfer prewarm exit status."""
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize('failure', ['sparse', 'gemm-missing', 'success'])
def test_prewarm_requires_sparse_success_and_compiled_gemm(monkeypatch, failure):
    calls = []
    def sparse():
        calls.append('sparse')
        if failure == 'sparse':
            raise RuntimeError('compiler failed')
        return object()
    def gemm():
        calls.append('gemm')
        return SimpleNamespace(is_compiled=lambda: failure != 'gemm-missing')
    for name in ('flashinfer', 'flashinfer.mla', 'flashinfer.jit'):
        stub = ModuleType(name)
        stub.__path__ = []
        monkeypatch.setitem(sys.modules, name, stub)
    sparse_module = ModuleType('flashinfer.mla._sparse_mla_sm120')
    sparse_module.get_sparse_mla_sm120_module = sparse
    gemm_module = ModuleType('flashinfer.jit.gemm')
    gemm_module.gen_gemm_sm120_module_cutlass_mxfp8 = gemm
    monkeypatch.setitem(sys.modules, sparse_module.__name__, sparse_module)
    monkeypatch.setitem(sys.modules, gemm_module.__name__, gemm_module)
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(Path(__file__).with_name('prewarm5.py')), run_name='__main__')
    assert result.value.code == (0 if failure == 'success' else 1)
    assert calls == (['sparse'] if failure == 'sparse' else ['sparse', 'gemm'])
