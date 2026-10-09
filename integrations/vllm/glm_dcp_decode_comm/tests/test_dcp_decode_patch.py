"""The code edits of the image's attention methods, their installation and registration.

The image's ``attention.py`` is compiled, not imported (it needs vLLM): a stand-in module holds functions
built from the compiled original methods, which is what ``install`` compares the loaded code with.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from types import FunctionType, ModuleType

import pytest

import glm_dcp_decode_comm as pkg

FORWARD, SPARSE = (patch.qualname for patch in pkg.PATCHES)


@pytest.fixture
def prepared(vllm_root):
    return tuple(pkg.prepare(patch, vllm_root) for patch in pkg.PATCHES)


_DECORATOR = textwrap.dedent("""
    import functools


    def eager_break_during_capture(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
""")


def attention_module(prepared, vllm_root, *, wrapped: bool = True) -> ModuleType:
    """A stand-in of ``vllm.models.deepseek_v32.attention`` whose two methods are the image's compiled code."""
    module = ModuleType(pkg.ATTENTION_MODULE)
    module.__file__ = str(vllm_root / pkg.ATTENTION_PATH)
    exec(_DECORATOR, module.__dict__)
    forward = FunctionType(prepared[0].original, module.__dict__, "forward")
    sparse = FunctionType(prepared[1].original, module.__dict__, "_sparse_indexer_and_attn")
    if wrapped:
        sparse = module.eager_break_during_capture(sparse)
    module.DeepseekV32Attention = type("DeepseekV32Attention", (), {"forward": forward,
                                                                    "_sparse_indexer_and_attn": sparse})
    return module


def test_every_edit_matches_the_image_source_once_and_keeps_its_lines(prepared):
    for item in prepared:
        assert item.original.co_qualname == item.edited.co_qualname == item.patch.qualname
        assert item.original != item.edited
        assert item.original.co_firstlineno == item.edited.co_firstlineno
    helpers = set(prepared[0].edited.co_names) | set(prepared[1].edited.co_names)
    assert set(pkg.HELPER_NAMES) <= helpers
    assert not set(pkg.HELPER_NAMES) & (set(prepared[0].original.co_names) | set(prepared[1].original.co_names))


@pytest.mark.parametrize("wrapped", (True, False), ids=("breakable-graphs", "plain"))
def test_install_edits_the_methods_once(prepared, vllm_root, wrapped):
    module = attention_module(prepared, vllm_root, wrapped=wrapped)
    cls = module.DeepseekV32Attention
    wrapper = cls.__dict__["_sparse_indexer_and_attn"]
    assert pkg.install(module, prepared) is True
    inner = getattr(wrapper, "__wrapped__", wrapper)
    assert cls.__dict__["_sparse_indexer_and_attn"] is wrapper           # a wrapper stays in place
    assert getattr(inner, pkg.MARKER) == SPARSE and getattr(cls.forward, pkg.MARKER) == FORWARD
    assert cls.forward.__code__ == prepared[0].edited.replace(co_filename=cls.forward.__code__.co_filename)
    assert inner.__code__ == prepared[1].edited.replace(co_filename=inner.__code__.co_filename)
    for name in pkg.HELPER_NAMES:
        assert module.__dict__[name] is pkg.HELPERS[name]
    assert pkg.install(module, prepared) is False


def test_another_wrapper_refuses(prepared, vllm_root):
    module = attention_module(prepared, vllm_root, wrapped=False)
    original = module.DeepseekV32Attention._sparse_indexer_and_attn

    def foreign(*args, **kwargs):
        return original(*args, **kwargs)

    foreign.__wrapped__ = original
    module.DeepseekV32Attention._sparse_indexer_and_attn = foreign
    with pytest.raises(pkg.PatchRefused, match="not by vLLM's eager_break_during_capture"):
        pkg.install(module, prepared)


def test_code_that_differs_from_the_verified_source_refuses(prepared, vllm_root):
    module = attention_module(prepared, vllm_root)
    module.DeepseekV32Attention.forward = FunctionType(prepared[0].edited, module.__dict__, "forward")
    with pytest.raises(pkg.PatchRefused, match="differs from the verified source"):
        pkg.install(module, prepared)


def test_a_helper_name_taken_by_another_object_refuses(prepared, vllm_root):
    module = attention_module(prepared, vllm_root)
    setattr(module, pkg.HELPER_NAMES[0], object())
    with pytest.raises(pkg.PatchRefused, match="already exists"):
        pkg.install(module, prepared)


def test_a_module_loaded_from_another_file_refuses(prepared, vllm_root, tmp_path):
    module = attention_module(prepared, vllm_root)
    module.__file__ = str(tmp_path / "attention.py")
    with pytest.raises(pkg.PatchRefused, match="not from the verified file"):
        pkg.install(module, prepared)


def test_a_changed_image_source_refuses(vllm_root, tmp_path):
    copy = tmp_path / pkg.ATTENTION_PATH
    copy.parent.mkdir(parents=True)
    copy.write_bytes((vllm_root / pkg.ATTENTION_PATH).read_bytes() + b"\n")
    with pytest.raises(pkg.PatchRefused, match="refusing to patch"):
        pkg.prepare(pkg.PATCHES[0], tmp_path)


@pytest.fixture
def clean_registration(monkeypatch):
    for name, value in (("_REGISTERED", None), ("_SETTINGS", None), ("_FINDER", None), ("_RUNTIME", None)):
        monkeypatch.setattr(pkg, name, value)
    monkeypatch.setattr(pkg, "_ROOTS", {})
    for name in (*pkg.FLAGS.values(), *pkg.UNSUPPORTED_SETTINGS, pkg.AUDIT, pkg.COMM_PRIORITY):
        monkeypatch.delenv(name, raising=False)
    saved = list(sys.meta_path)
    yield
    sys.meta_path[:] = saved


def test_register_patches_the_loaded_attention_module_and_waits_for_sircls_communicator(
        clean_registration, monkeypatch, prepared, vllm_root, b12x_root, sircl_root):
    roots = {"vllm": vllm_root, "b12x": b12x_root, pkg.SIRCL_PACKAGE: sircl_root}
    monkeypatch.setattr(pkg, "_package_roots", lambda names: {name: roots[name] for name in names})
    module = attention_module(prepared, vllm_root)
    monkeypatch.setitem(sys.modules, pkg.ATTENTION_MODULE, module)
    monkeypatch.delitem(sys.modules, pkg.COMMUNICATOR_MODULE, raising=False)
    monkeypatch.setenv("GLM_DCP_DECODE_OVERLAP", "1")
    monkeypatch.setenv("GLM_DCP_DECODE_A2A_FUSED", "1")
    pkg.register()
    assert getattr(module.DeepseekV32Attention.forward, pkg.MARKER) == FORWARD
    assert pkg.status()["pending_modules"] == [pkg.COMMUNICATOR_MODULE]
    assert pkg._ROOTS[pkg.SIRCL_PACKAGE] == sircl_root
    pkg.register()                                                       # idempotent

    # The communicator module, when it loads: wrapped from the verified file, refused from another.
    from glm_dcp_decode_comm import runtime

    class SirclCudaCommunicator:
        pass

    for name in runtime._WRAPPED_METHODS:
        setattr(SirclCudaCommunicator, name, lambda self, *args: None)
    communicator = ModuleType(pkg.COMMUNICATOR_MODULE)
    communicator.SirclCudaCommunicator = SirclCudaCommunicator
    communicator.__file__ = str(sircl_root / "vllm" / "communicator.py")
    assert pkg.install_comm_stream(communicator) is True
    assert all(getattr(getattr(SirclCudaCommunicator, name), runtime.WRAP_MARKER, False)
               for name in runtime._WRAPPED_METHODS)
    communicator.__file__ = str(sircl_root / "communicator.py")
    with pytest.raises(pkg.PatchRefused, match="not from the verified file"):
        pkg.install_comm_stream(communicator)


def test_register_refuses_a_sircl_tree_it_was_not_built_against(
        clean_registration, monkeypatch, vllm_root, b12x_root, sircl_root, tmp_path):
    import shutil

    copy = tmp_path / "sparkring_sircl"
    shutil.copytree(sircl_root, copy, ignore=shutil.ignore_patterns("__pycache__"))
    with (copy / "vllm" / "dcp_collectives.py").open("a", encoding="utf-8") as handle:
        handle.write("# changed\n")
    roots = {"vllm": vllm_root, "b12x": b12x_root, pkg.SIRCL_PACKAGE: copy}
    monkeypatch.setattr(pkg, "_package_roots", lambda names: {name: roots[name] for name in names})
    monkeypatch.setenv("GLM_DCP_DECODE_QUERY_PACK", "1")
    with pytest.raises(pkg.PatchRefused, match="SirclDcpCollectives keeps _runtime"):
        pkg.register()
    assert pkg._REGISTERED is None and pkg._SETTINGS is None


def test_patch_on_import_runs_its_callback_after_the_module_executes(tmp_path, monkeypatch):
    (tmp_path / "dd_probe_module.py").write_text("VALUE = 41\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    finder = pkg.PatchOnImport({"dd_probe_module": lambda module: seen.append(module.VALUE + 1)})
    sys.meta_path.insert(0, finder)
    try:
        import dd_probe_module  # noqa: F401

        assert seen == [42] and finder.pending() == () and finder not in sys.meta_path
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        sys.modules.pop("dd_probe_module", None)


SPEEDUPS = Path(__file__).resolve().parents[2] / "glm53full_speedups"


@pytest.mark.parametrize("ours_first", (True, False), ids=("ours-first", "speedups-first"))
def test_patch_on_import_shares_a_module_with_the_speedups_plugins_finder(tmp_path, monkeypatch, ours_first):
    """glm53full_speedups also patches the attention module on its first import, and profile glm53-nvfp4-tp8
    loads both plugins. Each finder asks every other finder for the module's spec; whichever is first on
    ``sys.meta_path``, the module executes once and each callback runs once."""
    if not (SPEEDUPS / "glm53full_speedups" / "__init__.py").is_file():
        pytest.skip("integrations/vllm/glm53full_speedups is not beside this plugin")
    monkeypatch.syspath_prepend(str(SPEEDUPS))
    import glm53full_speedups

    name = f"dd_shared_probe_{'ours' if ours_first else 'theirs'}"
    (tmp_path / f"{name}.py").write_text("EXECUTED = []\nEXECUTED.append(1)\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    ours = pkg.PatchOnImport({name: lambda module: seen.append(("ours", len(module.EXECUTED)))})
    theirs = glm53full_speedups.PatchOnImport({name: lambda module: seen.append(("speedups", len(module.EXECUTED)))})
    finders = (ours, theirs) if ours_first else (theirs, ours)
    for finder in reversed(finders):
        sys.meta_path.insert(0, finder)
    try:
        module = __import__(name)
        assert module.EXECUTED == [1]
        assert sorted(seen) == [("ours", 1), ("speedups", 1)]
        assert ours not in sys.meta_path and theirs not in sys.meta_path
    finally:
        for finder in finders:
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)
        sys.modules.pop(name, None)
