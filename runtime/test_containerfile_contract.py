"""Contract tests for fail-closed runtime build stages."""

from pathlib import Path


CONTAINERFILE = (Path(__file__).resolve().parent / "Containerfile").read_text(
    encoding="utf-8"
)


def test_builder_closes_freeze_before_installing_requirements():
    assert "prepare-public-requirements.py" in CONTAINERFILE
    assert "-r /build/public-requirements.txt" in CONTAINERFILE
    assert "pip install --no-cache-dir -r /build/pip-freeze.txt" not in CONTAINERFILE


def test_builder_enforces_final_packages_and_transport_ctest():
    assert "verify-frozen-packages.py --freeze /build/pip-freeze.txt" in CONTAINERFILE
    assert "ctest --test-dir /build/spark_transport_build --output-on-failure" in (
        CONTAINERFILE
    )


def test_source_pins_determine_nccl_and_deepgemm_checkouts():
    assert 'git checkout "${NCCL_COMMIT}"' in CONTAINERFILE
    assert 'git checkout "${DEEPGEMM_COMMIT}"' in CONTAINERFILE
    assert 'test "$(git rev-parse HEAD)" = "${DEEPGEMM_COMMIT}"' in CONTAINERFILE


def test_source_builds_initialize_pinned_submodules():
    flashinfer_checkout = CONTAINERFILE.index('git checkout "${FLASHINFER_COMMIT}"')
    deepgemm_checkout = CONTAINERFILE.index('git checkout "${DEEPGEMM_COMMIT}"')
    assert "git submodule update --init --recursive" in CONTAINERFILE[
        flashinfer_checkout:deepgemm_checkout
    ]
    assert "git submodule update --init --recursive" in CONTAINERFILE[
        deepgemm_checkout:
    ]
