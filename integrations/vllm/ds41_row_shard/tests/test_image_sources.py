"""Pins, fixture copies and the ``_o_proj`` edit against an extracted image tree.

Run with ``SPARKRING_DS41_IMAGE_SOURCES`` pointing at an extracted image (a
directory containing ``vllm/models/deepseek_v4_1``). Only files are read; vLLM
is not imported.
"""

from __future__ import annotations

import ast
import functools
import os
import re
import textwrap
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

import pytest

import ds41_row_shard
from ds41_row_shard import source_edit

IMAGE_ROOT = os.environ.get("SPARKRING_DS41_IMAGE_SOURCES")
TREE = Path(IMAGE_ROOT) if IMAGE_ROOT else None

pytestmark = pytest.mark.skipif(
    TREE is None or not (TREE / "vllm" / "models" / "deepseek_v4_1").is_dir(),
    reason="SPARKRING_DS41_IMAGE_SOURCES is not an extracted vLLM tree",
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _image(relative: str) -> Path:
    return TREE / relative


def _class_method(path: Path, class_name: str, method_name: str):
    """Return the method's source at its file indentation and its AST node."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    node = next(
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef) and item.name == method_name
    )
    return "".join(lines[node.lineno - 1 : node.end_lineno]), node


def _fixture_forward(path: Path):
    """Return the fixture header and its forward, dedented, to end of file."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start = next(
        index for index, line in enumerate(lines) if line.startswith("def forward")
    )
    return "".join(lines[:start]), textwrap.dedent("".join(lines[start:]))


@pytest.mark.parametrize(
    "module,sha256",
    [(name, pin[0]) for name, pin in ds41_row_shard.PINS.items()],
    ids=list(ds41_row_shard.PINS),
)
def test_pin_matches_image_file(module, sha256):
    path = _image(module.replace(".", "/") + ".py")
    assert path.is_file(), f"{module}: {path} not found"
    assert source_edit.file_sha256(str(path)) == sha256


@pytest.mark.parametrize(
    "fixture,relative,class_name",
    [
        (
            "decoder_layer_forward.py",
            "vllm/models/deepseek_v4_1/nvidia/model.py",
            "DeepseekV4DecoderLayer",
        ),
        (
            "engram_forward.py",
            "vllm/models/deepseek_v4_1/common/engram.py",
            "Engram",
        ),
    ],
)
def test_fixture_matches_image_forward(fixture, relative, class_name):
    image_source, node = _class_method(_image(relative), class_name, "forward")
    header, fixture_source = _fixture_forward(FIXTURES / fixture)
    assert fixture_source == textwrap.dedent(image_source)
    match = re.search(r"lines (\d+)-(\d+)", header)
    assert match is not None, header
    assert (int(match.group(1)), int(match.group(2))) == (
        node.lineno,
        node.end_lineno,
    )


def test_o_proj_edit_applies_and_compiles():
    path = _image("vllm/models/deepseek_v4_1/attention.py")
    compiled = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    cls_code = next(
        code for code in compiled.co_consts
        if isinstance(code, CodeType) and code.co_name == "DeepseekV4Attention"
    )
    original = next(
        code for code in cls_code.co_consts
        if isinstance(code, CodeType) and code.co_name == "_o_proj"
    )
    module = ModuleType("test_pinned_attention")
    module.__file__ = str(path)
    method = FunctionType(original, module.__dict__)
    module.DeepseekV4Attention = type("DeepseekV4Attention", (), {"_o_proj": method})

    ds41_row_shard.patch_attention(module)

    assert getattr(method, ds41_row_shard.MARKER)
    assert method.__code__.co_firstlineno == original.co_firstlineno
    assert "_ds41_row_shard_attention_reduce" in method.__code__.co_names


def test_indexer_edit_keeps_capture_wrapper(monkeypatch):
    monkeypatch.setenv(ds41_row_shard.INDEXER_FLAG, "1")
    path = _image("vllm/models/deepseek_v4_1/attention.py")
    class_name = "DeepseekV4Attention"
    module = ModuleType("test_pinned_indexer")
    module.__file__ = str(path)
    source = path.read_text(encoding="utf-8")
    o_proj_code = source_edit._compile_function(source, f"{class_name}._o_proj", str(path))
    mqa_code = source_edit._compile_function(source, f"{class_name}.forward_mqa", str(path))
    o_proj = FunctionType(o_proj_code, module.__dict__)
    mqa = FunctionType(mqa_code, module.__dict__)

    @functools.wraps(mqa)
    def capture_wrapper(*args, **kwargs):
        return mqa(*args, **kwargs)

    module.DeepseekV4Attention = type(
        class_name, (), {"_o_proj": o_proj, "forward_mqa": capture_wrapper}
    )
    ds41_row_shard.patch_attention(module)
    assert module.DeepseekV4Attention.forward_mqa is capture_wrapper
    assert mqa.__code__.co_firstlineno == mqa_code.co_firstlineno
    assert "_ds41_indexer_score_offsets" in mqa.__code__.co_names
    assert "_ds41_publish_index_rows" in mqa.__code__.co_names


def test_compact_route_edits_apply_to_pinned_runner(monkeypatch):
    monkeypatch.setenv(ds41_row_shard.ROUTES_FLAG, "1")
    path = _image(ds41_row_shard.MOE_RUNNER_MODULE.replace(".", "/") + ".py")
    source = path.read_text(encoding="utf-8")
    module = ModuleType("test_pinned_moe_runner")
    module.__file__ = str(path)
    methods = {
        name: FunctionType(
            source_edit._compile_function(source, f"MoERunner.{name}", str(path)),
            module.__dict__,
        )
        for name in ("_forward_impl", "_apply_quant_method")
    }
    module.MoERunner = type("MoERunner", (), methods)
    ds41_row_shard.patch_moe_runner(module)
    assert "_ds41_gate_owned_routes" in methods["_forward_impl"].__code__.co_names
    assert "_ds41_select_owned_routes" in methods["_apply_quant_method"].__code__.co_names


def test_local_queries_compose_with_sharded_indexer_on_pinned_source(monkeypatch):
    monkeypatch.setenv(ds41_row_shard.LOCAL_QUERIES_FLAG, "1")
    monkeypatch.setenv(ds41_row_shard.INDEXER_FLAG, "1")
    path = _image("vllm/models/deepseek_v4_1/attention.py")
    source = path.read_text(encoding="utf-8")
    module = ModuleType("test_pinned_local_queries")
    module.__file__ = str(path)
    methods = {
        name: FunctionType(
            source_edit._compile_function(source, f"DeepseekV4Attention.{name}", str(path)),
            module.__dict__,
        )
        for name in ("_o_proj", "_forward", "forward_mqa", "_write_decode_metadata", "setup_wo_projection",
                     "get_b12x_preparation_units")
    }
    module.DeepseekV4Attention = type("DeepseekV4Attention", (), methods)
    ds41_row_shard.patch_attention(module)
    assert "_ds41_project_query" in methods["_forward"].__code__.co_names
    assert "_ds41_attend_owned" in methods["forward_mqa"].__code__.co_names
    assert "_ds41_indexer_score_offsets" in methods["forward_mqa"].__code__.co_names
    assert methods["_write_decode_metadata"].__defaults__[-1] == 0


def test_compressor_hooks_match_pinned_cache_write_methods():
    from ds41_row_shard import local_query

    modules = []
    for name, cls, method in (
        (ds41_row_shard.ATTENTION_MODULE, "DeepseekV4Attention", "_prepare_global_kv"),
        (ds41_row_shard.COMPRESSOR_MODULE, "DeepseekCompressor", "forward"),
    ):
        path = _image(name.replace(".", "/") + ".py")
        module = ModuleType(name)
        module.__file__ = str(path)
        raw = source_edit._compile_function(path.read_text(encoding="utf-8"), f"{cls}.{method}", str(path))
        fn = FunctionType(raw, module.__dict__)
        setattr(module, cls, type(cls, (), {method: fn}))
        modules.append(module)

    attention, compressor = modules
    local_query.install_compressor(
        attention, compressor, ds41_row_shard.PINS[attention.__name__][0],
        ds41_row_shard.PINS[compressor.__name__][0], ds41_row_shard.MARKER,
    )
    assert "_ds41_global_rows" in attention.DeepseekV4Attention._prepare_global_kv.__code__.co_names
    assert "_ds41_compress_rows" in compressor.DeepseekCompressor.forward.__code__.co_names
