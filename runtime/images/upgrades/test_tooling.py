"""Compiler recipe dependencies cover the DeepGEMM diagnostic include."""

from pathlib import Path


def test_native_system_stage_pins_headers_and_compiles_the_include_probe():
    recipe = Path(__file__).with_name("Dockerfile.tooling").read_text()
    native, full = recipe.split("FROM native-system-tools AS tooling", 1)
    assert "FROM ${PARENT} AS native-system-tools" in native
    assert "ARG ELFUTILS_VERSION=0.190-1.1ubuntu0.1" in native
    for package in ("libdw-dev", "libelf-dev", "libdw1t64", "libelf1t64"):
        assert package + "=${ELFUTILS_VERSION}" in native
    assert "-fsyntax-only -x c++ -include elfutils/libdwfl.h /dev/null" in native
    assert "elfutils-packages.txt" in native
    assert "COPY inputs/" not in native
    assert "COPY inputs/" in full
