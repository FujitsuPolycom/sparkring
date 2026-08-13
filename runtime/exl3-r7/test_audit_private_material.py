"""Audit tests: no private paths, site identities, credentials, or model weights
in the seeded exl3-r7 runtime files."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

# Patterns that indicate private site material, not container-internal conventions.
# Container-internal paths like /opt/, /cache/, /models/ are expected in a
# Containerfile and are NOT private site identities.
PRIVATE_PATTERNS = [
    # RFC1918 private site addressing (not 10.x which is used as examples)
    re.compile(r"192\.168\.\d{1,3}\.\d{1,3}"),
    re.compile(r"(?<![0-9.])172\.(1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}"),
    # SSH account forms with private IPs
    re.compile(r"[A-Za-z0-9_.-]+@(192\.168\.|10\.\d|172\.(1[6-9]|2[0-9]|3[01])\.)"),
    # Private key material
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"ssh-(rsa|ed25519|dss) AAAA"),
    # Credential assignment shapes
    re.compile(
        r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token"
        r"|client[_-]?secret|bearer)"
        r"['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9/+_.-]{12,}"
    ),
    # Private host paths (Windows)
    re.compile(r"Documents[\\/]sparkring"),
    # Site config file references
    re.compile(r"site\.yaml|launch\.json|gate\.json"),
]


def _python_files() -> list[Path]:
    return sorted(HERE.glob("*.py"))


def _all_text_files() -> list[Path]:
    files = list(HERE.glob("*.py"))
    files.extend(HERE.glob("*.sh"))
    files.extend(HERE.glob("*.json"))
    return sorted(files)


@pytest.mark.parametrize("filepath", _all_text_files(), ids=lambda p: p.name)
def test_no_private_site_identifiers(filepath: Path) -> None:
    """No tracked exl3-r7 file may contain private site addresses, credentials,
    or private host paths."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    for pattern in PRIVATE_PATTERNS:
        matches = pattern.findall(text)
        assert not matches, (
            f"{filepath.name}: private identifier pattern {pattern.pattern!r} "
            f"matched {len(matches)} time(s)"
        )


@pytest.mark.parametrize("filepath", _python_files(), ids=lambda p: p.name)
def test_no_model_weight_references(filepath: Path) -> None:
    """Python files should not contain hardcoded model weight file paths as
    defaults. Container-internal /models/ paths in probe scripts are allowed
    only when they are argparse defaults for diagnostic tools, not build inputs."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    # Model weight files as hardcoded build inputs (not argparse defaults) would
    # be a leak. The sm121 probes use them as diagnostic argparse defaults,
    # which is acceptable for offline diagnostic tools.
    # This test guards against weight references in build/prepare scripts.
    if filepath.name in ("prepare_context.py", "prepare_build_deps.py", "build-image.sh"):
        assert ".safetensors" not in text
        assert ".gguf" not in text
        assert ".bin" not in text or "binary" in text.lower()


def test_pins_json_contains_only_public_repositories() -> None:
    """pins.json must reference only public GitHub repositories, not private
    hosts or file:// URLs."""
    import json

    pins = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
    all_text = json.dumps(pins)
    assert "file://" not in all_text
    assert "192.168" not in all_text
    assert "10.0.0" not in all_text
    # All repositories must be HTTPS GitHub URLs
    for key in ("release",):
        assert pins[key]["repository"].startswith("https://github.com/")
    for name, spec in pins["components"].items():
        assert spec["repository"].startswith("https://github.com/")
