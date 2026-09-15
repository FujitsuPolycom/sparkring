"""Native reuse includes build metadata and native-language files outside csrc."""

from .sources import native_digest


def test_python_edits_do_not_alias_changed_native_inputs(tmp_path):
    (tmp_path / "csrc").mkdir()
    (tmp_path / "vllm").mkdir()
    (tmp_path / "csrc/op.cu").write_text("native")
    (tmp_path / "setup.py").write_text("build")
    (tmp_path / "vllm/model.py").write_text("model")
    before = native_digest(tmp_path, ["csrc", "setup.py"])
    (tmp_path / "vllm/model.py").write_text("updated Python")
    assert native_digest(tmp_path, ["csrc", "setup.py"]) == before
    (tmp_path / "vllm/helper.cpp").write_text("outside declared native directory")
    assert native_digest(tmp_path, ["csrc", "setup.py"]) != before


def test_build_metadata_changes_require_recompilation(tmp_path):
    (tmp_path / "setup.py").write_text("one")
    before = native_digest(tmp_path, ["setup.py"])
    (tmp_path / "setup.py").write_text("two")
    assert native_digest(tmp_path, ["setup.py"]) != before
