"""Every pin the plugin records, replayed against the target image's own sources.

The image sources come from ``SPARKRING_GLM53_IMAGE_SOURCES`` (see
``conftest.py``); without them these tests fail with the setup instructions,
like a hardware test without the hardware.
"""

from __future__ import annotations

import shutil

import pytest
from glm_dsa_indexer_split import (
    FILE_CHECKS,
    PACKAGE_SHA256,
    PLUGIN_VERSION,
    PatchRefused,
    compile_originals,
    settings_from_env,
    verify_file,
    verify_package,
)


def test_package_pins_match_the_installed_files():
    verify_package()


def test_every_file_check_matches_the_image_sources(image_sources):
    roots = {"vllm": image_sources / "vllm", "b12x": image_sources / "b12x"}
    for check in FILE_CHECKS:
        verify_file(check, roots)


def test_indexer_class_compiles_from_the_pinned_source(image_sources):
    original = compile_originals(image_sources / "vllm")
    module = __import__("vllm.v1.attention.backends.mla.b12x_indexer",
                        fromlist=["B12xSparseIndexer"])
    for name, code in original.code.items():
        # conftest.py has registered the plugin: the installed attribute is
        # this plugin's wrapper of the image's method.
        loaded = getattr(vars(module.B12xSparseIndexer)[name], "__wrapped__", None)
        assert loaded is not None and loaded.__code__ == code, name


def test_tampered_indexer_source_refuses(image_sources, tmp_path):
    copy = tmp_path / "vllm"
    shutil.copytree(image_sources / "vllm", copy)
    path = copy / "v1/attention/backends/mla/b12x_indexer.py"
    data = bytearray(path.read_bytes())
    data[0] = data[0] ^ 0x20  # flip one byte inside the first comment
    path.write_bytes(bytes(data))
    with pytest.raises(PatchRefused, match="SHA-256"):
        compile_originals(copy)


def test_missing_dependency_file_refuses(image_sources, tmp_path):
    roots = {"vllm": image_sources / "vllm", "b12x": tmp_path / "absent"}
    check = next(check for check in FILE_CHECKS if check.package == "b12x")
    with pytest.raises(PatchRefused, match="is missing"):
        verify_file(check, roots)


def test_changed_dependency_file_refuses(image_sources, tmp_path):
    roots = {"vllm": tmp_path / "vllm", "b12x": image_sources / "b12x"}
    shutil.copytree(image_sources / "vllm", roots["vllm"])
    check = next(check for check in FILE_CHECKS
                 if check.package == "vllm" and check.path != "v1/attention/backends/mla/b12x_indexer.py")
    path = roots["vllm"] / check.path
    data = bytearray(path.read_bytes())
    data[0] = data[0] ^ 0x20
    path.write_bytes(bytes(data))
    with pytest.raises(PatchRefused, match="SHA-256"):
        verify_file(check, roots)


def test_settings_reject_malformed_values(monkeypatch):
    monkeypatch.setenv("GLM_DSA_INDEXER_SPLIT", "1")
    settings = settings_from_env()
    assert settings["enabled"] and settings["min_rows"] >= 1
    monkeypatch.setenv("GLM_DSA_INDEXER_SPLIT", "true")
    with pytest.raises(PatchRefused, match="GLM_DSA_INDEXER_SPLIT"):
        settings_from_env()
    monkeypatch.setenv("GLM_DSA_INDEXER_SPLIT", "1")
    monkeypatch.setenv("GLM_DSA_INDEXER_SPLIT_MIN_ROWS", "0")
    with pytest.raises(PatchRefused, match="MIN_ROWS"):
        settings_from_env()
    monkeypatch.delenv("GLM_DSA_INDEXER_SPLIT_MIN_ROWS")
    monkeypatch.setenv("GLM_DSA_INDEXER_SPLIT_FULL_LAUNCHES", "1")
    monkeypatch.setenv("GLM_DSA_INDEXER_SPLIT", "0")
    with pytest.raises(PatchRefused, match="FULL_LAUNCHES set without"):
        settings_from_env()


def test_version_names_the_ported_plugin():
    assert PLUGIN_VERSION == "1.1.0"
    assert set(PACKAGE_SHA256) == {"layout.py", "runtime.py"}
