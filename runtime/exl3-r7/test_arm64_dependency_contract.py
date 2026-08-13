from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_build_verifier_can_resolve_inherited_sircl_hooks() -> None:
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")

    assert "rm -f /opt/venv/lib/python3.12/site-packages/sparkring_nf3_hybrid.pth" in containerfile
    assert (
        "PYTHONPATH=/opt/spark-vllm \\\n"
        "      /opt/venv/bin/python /opt/sparkring-r7/verify_runtime.py --installed-only"
        in containerfile
    )


def test_arm64_native_dependencies_are_receipt_gated() -> None:
    containerfile = (HERE / "Containerfile").read_text(encoding="utf-8")

    assert 'test "$(git -C /opt/exllamav3-python write-tree)" = "${EXLLAMAV3_ARM64_TREE}"' in containerfile
    assert "git -C /opt/instanttensor-src submodule update --init --recursive" in containerfile
    assert 'test "${TARGETARCH}" = arm64' in containerfile
