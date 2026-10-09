"""Pins, patch mechanics and refusal paths, against the target image's own sources.

The image sources come from ``SPARKRING_GLM53_IMAGE_SOURCES``; without them
these tests fail with the setup instructions, like a hardware test without
the hardware. The two ``FileCheck`` pins for ``linear.py`` and ``parameter.py``
are replayed by every test session (``_env.py``).
"""

from __future__ import annotations

import shutil
import types
from types import FunctionType

import pytest
import glm53full_speedups as plugin
from glm53full_speedups import (
    FILE_CHECKS,
    PATCHES,
    PLUGIN_VERSION,
    PatchRefused,
    prepare,
    verify_file,
)


def test_every_file_check_matches_the_image_sources(image_sources):
    roots = {"vllm": image_sources / "vllm", "b12x": image_sources / "b12x"}
    for check in FILE_CHECKS:
        verify_file(check, roots)


@pytest.mark.parametrize("index", range(len(PATCHES)))
def test_every_patch_prepares_against_the_pinned_source(index, image_sources):
    """The edit applies exactly once, keeps the line count, and the original compiles.

    The compiled original must equal the function compiled from the whole
    module file: the same source lines, line numbers, ``__future__`` flags,
    module imports and class scope.
    """
    patch = PATCHES[index]
    prepared = prepare(patch, image_sources / "vllm")
    whole = compile((image_sources / "vllm" / patch.path).read_text(encoding="utf-8"),
                    str(image_sources / "vllm" / patch.path), "exec",
                    flags=prepared.original.co_flags & 0, dont_inherit=True)
    assert plugin._find_code(whole, patch.qualname) == prepared.original


def test_tampered_patch_target_refuses(image_sources, tmp_path):
    copy = tmp_path / "vllm"
    shutil.copytree(image_sources / "vllm", copy)
    patch = PATCHES[0]
    path = copy / patch.path
    data = bytearray(path.read_bytes())
    data[0] = data[0] ^ 0x20
    path.write_bytes(bytes(data))
    with pytest.raises(PatchRefused, match="SHA-256"):
        prepare(patch, copy)


def test_install_places_the_helper_and_is_idempotent(image_sources):
    patch = PATCHES[0]
    prepared = prepare(patch, image_sources / "vllm")
    module = types.ModuleType(patch.module)
    module.__file__ = str(image_sources / "vllm" / patch.path)
    # The image's method holds a __class__ cell; a stand-in cell keeps it callable.
    function = FunctionType(prepared.original, module.__dict__, "__init__", None,
                            (types.CellType(None),))
    module.DeepseekV32Attention = type("DeepseekV32Attention", (), {"__init__": function})
    assert plugin.install(module, prepared) is True
    assert getattr(function, plugin.MARKER) == patch.qualname
    assert function.__code__ == prepared.edited
    assert vars(module)[plugin.LATENT_HELPER] is plugin.HELPERS[plugin.LATENT_HELPER]
    assert plugin.install(module, prepared) is False


def test_install_refuses_a_foreign_module(image_sources):
    patch = PATCHES[0]
    prepared = prepare(patch, image_sources / "vllm")
    module = types.ModuleType(patch.module)
    module.__file__ = str(image_sources / "vllm" / "elsewhere.py")
    with pytest.raises(PatchRefused, match="was loaded from"):
        plugin.install(module, prepared)


def test_install_refuses_a_changed_function(image_sources):
    patch = PATCHES[1]
    prepared = prepare(patch, image_sources / "vllm")
    module = types.ModuleType(patch.module)
    module.__file__ = str(image_sources / "vllm" / patch.path)
    module.DeepseekV32MultiTokenPredictorLayer = type(
        "DeepseekV32MultiTokenPredictorLayer", (),
        {"__init__": lambda self, *args, **kwargs: None})
    with pytest.raises(PatchRefused, match="refusing to patch"):
        plugin.install(module, prepared)


def test_version_names_the_ported_plugin():
    assert PLUGIN_VERSION == "1.1.0"
