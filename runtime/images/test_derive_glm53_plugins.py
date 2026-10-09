"""The GLM-5.3 plugins layer: its files, pins and entry points."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.images import derive_glm53_plugins
from runtime.images.derived_layer import SITE, _python_file, sha


def test_pins_match_the_repository_files():
    for target, (inherited, resulting) in derive_glm53_plugins.LAYER.pins.items():
        source = derive_glm53_plugins.ROOT / derive_glm53_plugins.ADDED[target]
        assert inherited is None, target  # every file is an addition
        assert resulting == sha(source.read_bytes()), target
    assert set(derive_glm53_plugins.LAYER.pins) == set(derive_glm53_plugins.ADDED)


def test_every_added_file_is_site_packages_python_or_dist_info():
    for target in derive_glm53_plugins.ADDED:
        if ".dist-info/" in target:
            assert target.startswith(SITE) and target.endswith(("METADATA", "entry_points.txt",
                                                                "top_level.txt")), target
        else:
            _python_file(target)


def test_replace_returns_only_the_pinned_files():
    replaced = derive_glm53_plugins.LAYER.replace(lambda path: b"", {})
    assert set(replaced) == set(derive_glm53_plugins.LAYER.pins)
    for target, data in replaced.items():
        source = derive_glm53_plugins.ROOT / derive_glm53_plugins.ADDED[target]
        assert data == source.read_bytes(), target


@pytest.mark.parametrize("plugin,package", [("glm_dsa_indexer_split", "glm-dsa-indexer-split"),
                                            ("glm53full_speedups", "glm53full-speedups")])
def test_dist_info_registers_the_plugin_in_vllm_general_plugins(plugin, package):
    root = Path(derive_glm53_plugins.ROOT)
    metadata = (root / "integrations/vllm" / plugin / "dist-info"
                / f"{plugin}-1.1.0.dist-info/METADATA").read_text()
    entry_points = (root / "integrations/vllm" / plugin / "dist-info"
                    / f"{plugin}-1.1.0.dist-info/entry_points.txt").read_text()
    assert f"Name: {package}" in metadata and "Version: 1.1.0" in metadata
    assert "[vllm.general_plugins]" in entry_points
    assert f"{plugin} = {plugin}:register" in entry_points


def test_the_plugins_pin_the_image_files_they_wrap():
    """Each plugin records the image's own SHA-256 values; the layer ships no vllm or b12x file."""
    for plugin in ("glm_dsa_indexer_split", "glm53full_speedups"):
        module = __import__(f"integrations.vllm.{plugin}.{plugin}", fromlist=[plugin])
        checks = list(module.FILE_CHECKS)
        if hasattr(module, "PATCHES"):
            checks += [type("P", (), {"package": "vllm", "path": p.path, "sha256": p.sha256})
                       for p in module.PATCHES]
        assert checks
        for check in checks:
            assert re_fullmatch(check.sha256), (plugin, check.path)
        for target in derive_glm53_plugins.ADDED:
            assert "/vllm/" not in target or "dist-info" in target


def re_fullmatch(value: str) -> bool:
    import re

    return re.fullmatch(r"[0-9a-f]{64}", value) is not None
