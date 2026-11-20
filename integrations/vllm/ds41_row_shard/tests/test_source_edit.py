"""The function edit lands on the pinned source, or it is refused."""

from __future__ import annotations

import importlib.util
import sys

import pytest

from ds41_row_shard import source_edit

MARKER = "__ds41_test_edit__"

MODULE_SOURCE = '''\
def reduce(v):
    return v * 10


class C:
    def f(self, x):
        y = x + 1
        return reduce(y)
'''

TWICE_SOURCE = '''\
def reduce(v):
    return v * 10


class C:
    def f(self, x):
        y = x + 1
        if x:
            return reduce(y)
        return reduce(y)
'''

DIVERGED_SOURCE = '''\
def reduce(v):
    return v * 10


class C:
    def f(self, x):
        y = x + 2
        return reduce(y)
'''

HELPER_SOURCE = '''\
def reduce(v):
    return v * 10


_helper = 0


class C:
    def f(self, x):
        y = x + 1
        return reduce(y)
'''

DECORATED_SOURCE = '''\
def identity(fn):
    return fn


def reduce(v):
    return v * 10


class C:
    @identity
    def f(self, x):
        y = x + 1
        return reduce(y)
'''


@pytest.fixture
def loaded(tmp_path):
    """Write a module to tmp_path, import it and remove it again afterwards."""
    names: list[str] = []

    def load(source: str = MODULE_SOURCE):
        path = tmp_path / "sample_module.py"
        path.write_text(source, encoding="utf-8")
        name = f"_ds41_source_edit_{len(names)}"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        names.append(name)
        spec.loader.exec_module(module)
        return module, path

    yield load
    for name in names:
        sys.modules.pop(name, None)


def _edit(module, path, replacements, helpers, sha256=None):
    return source_edit.edit_function(
        module,
        "C.f",
        source_edit.file_sha256(str(path)) if sha256 is None else sha256,
        replacements,
        helpers,
        MARKER,
    )


def test_edit_installs_helper_call_once(loaded):
    module, path = loaded()
    assert module.C().f(1) == 20
    edit = (("return reduce(y)", "return _helper(y)"),)
    helpers = {"_helper": lambda v: v - 1}
    assert _edit(module, path, edit, helpers) is True
    assert module.C().f(1) == 1
    assert getattr(module.C.f, MARKER, False) is True
    code = module.C.f.__code__
    # A second call sees the marker and does not edit again.
    assert _edit(module, path, edit, helpers) is False
    assert module.C.f.__code__ is code
    assert module.C().f(1) == 1


def test_wrong_sha_is_refused(loaded):
    module, path = loaded()
    good = source_edit.file_sha256(str(path))
    wrong = "0" * 64 if good != "0" * 64 else "1" * 64
    with pytest.raises(source_edit.PatchRefused, match="SHA-256"):
        _edit(
            module,
            path,
            (("return reduce(y)", "return _helper(y)"),),
            {"_helper": lambda v: v - 1},
            sha256=wrong,
        )
    assert not getattr(module.C.f, MARKER, False)
    assert module.C().f(1) == 20


@pytest.mark.parametrize(
    "source,old,message",
    [
        (MODULE_SOURCE, "return reduce(y + 1)", "found 0 times"),
        (TWICE_SOURCE, "return reduce(y)", "found 2 times"),
    ],
)
def test_anchor_must_occur_exactly_once(loaded, source, old, message):
    module, path = loaded(source)
    with pytest.raises(source_edit.PatchRefused, match=message):
        _edit(module, path, ((old, "return _helper(y)"),), {"_helper": lambda v: v - 1})
    assert not getattr(module.C.f, MARKER, False)


def test_line_count_changing_edit_is_refused(loaded):
    module, path = loaded()
    edit = (("        return reduce(y)", "        return reduce(\n            y)"),)
    with pytest.raises(source_edit.PatchRefused, match="line count"):
        _edit(module, path, edit, {"_helper": lambda v: v - 1})
    assert module.C().f(1) == 20


def test_loaded_code_must_match_the_file(loaded):
    module, path = loaded()
    # The module was imported from MODULE_SOURCE; the file now says something
    # else, and the new file's own hash is passed.
    path.write_text(DIVERGED_SOURCE, encoding="utf-8")
    with pytest.raises(source_edit.PatchRefused, match="differs"):
        _edit(
            module,
            path,
            (("return reduce(y)", "return _helper(y)"),),
            {"_helper": lambda v: v - 1},
        )
    assert module.C().f(1) == 20
    assert not getattr(module.C.f, MARKER, False)


def test_helper_collision_is_refused(loaded):
    module, path = loaded(HELPER_SOURCE)
    with pytest.raises(source_edit.PatchRefused, match="already exists"):
        _edit(module, path, (("return reduce(y)", "return _helper(y)"),),
              {"_helper": lambda v: v - 1})
    assert module.C().f(1) == 20
    assert not getattr(module.C.f, MARKER, False)


def test_decorated_function_is_refused(loaded):
    module, path = loaded(DECORATED_SOURCE)
    with pytest.raises(source_edit.PatchRefused, match="decorated"):
        _edit(
            module,
            path,
            (("return reduce(y)", "return _helper(y)"),),
            {"_helper": lambda v: v - 1},
        )
    assert module.C().f(1) == 20


def test_edit_preserves_traceback_line_numbers(loaded):
    module, path = loaded()

    def boom(value):
        raise ValueError(f"boom {value}")

    return_line = next(
        number
        for number, line in enumerate(MODULE_SOURCE.splitlines(), 1)
        if line.lstrip().startswith("return reduce(y)")
    )
    assert _edit(
        module,
        path,
        (("return reduce(y)", "return _boom(y)"),),
        {"_boom": boom},
    ) is True
    with pytest.raises(ValueError) as info:
        module.C().f(1)
    frames = []
    tb = info.tb
    while tb is not None:
        frames.append(tb)
        tb = tb.tb_next
    inside = [frame for frame in frames if frame.tb_frame.f_code.co_name == "f"]
    assert len(inside) == 1
    assert inside[0].tb_frame.f_code.co_filename == str(path)
    assert inside[0].tb_lineno == return_line
    # The file is not rewritten; the line number points at the original return.
    assert "return reduce(y)" in path.read_text(encoding="utf-8").splitlines()[return_line - 1]
