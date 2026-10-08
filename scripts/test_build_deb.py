"""The relay marker that the Debian package build compiles; no compiler runs here."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_deb

ROOT = Path(__file__).resolve().parents[1]


def payload_with_source(tmp_path):
    payload = tmp_path / "payload"
    source = payload / build_deb.MARKER_SOURCE
    source.parent.mkdir(parents=True)
    source.write_bytes((ROOT / build_deb.MARKER_SOURCE).read_bytes())
    return payload, source


def test_without_a_compiler_or_rdma_headers_the_package_has_no_marker(tmp_path):
    payload, _ = payload_with_source(tmp_path)
    header = tmp_path / "mlx5dv.h"
    notes = []
    assert build_deb.relay_marker(payload, which=lambda name: None, header=header, say=notes.append) is None
    assert "a C compiler (gcc) and the rdma-core headers (libibverbs-dev) missing" in notes[0]
    assert "installs no relay table" in notes[0]
    with pytest.raises(ValueError, match="Install gcc and libibverbs-dev"):
        build_deb.relay_marker(payload, mode="require", which=lambda name: None, header=header)
    assert build_deb.relay_marker(payload, mode="skip") is None
    with pytest.raises(ValueError, match="auto, require or skip"):
        build_deb.relay_marker(payload, mode="always")


def test_the_build_compiles_the_marker_and_records_both_digests(tmp_path):
    payload, source = payload_with_source(tmp_path)
    header = tmp_path / "mlx5dv.h"
    header.write_text("")
    commands = []

    def run(argv, check):
        commands.append(argv)
        Path(argv[argv.index("-o") + 1]).write_bytes(b"\x7fELF marker")
        return SimpleNamespace(returncode=0)

    built = build_deb.relay_marker(payload, run=run, which=lambda name: "/usr/bin/cc", header=header)
    assert commands == [["/usr/bin/cc", "-O2", "-Wall", "-Wextra", "-o", str(payload / build_deb.MARKER_PATH),
                         str(source), "-libverbs", "-lmlx5"]]
    assert built == {"path": "bin/sparkring-relay-marker", "sha256": hashlib.sha256(b"\x7fELF marker").hexdigest(),
                     "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def test_a_compiler_that_writes_nothing_stops_the_build(tmp_path):
    payload, _ = payload_with_source(tmp_path)
    header = tmp_path / "mlx5dv.h"
    header.write_text("")
    with pytest.raises(ValueError, match="produced no binary"):
        build_deb.relay_marker(payload, run=lambda argv, check: None, which=lambda name: "/usr/bin/cc", header=header)


def test_the_marker_keeps_the_destination_command_line_and_adds_the_source_port_mode():
    text = (ROOT / build_deb.MARKER_SOURCE).read_text(encoding="utf-8")
    for option in ('{"device", required_argument', '{"rule", required_argument', '{"source-port", required_argument',
                   '{"run-seconds", required_argument', '{"managed", no_argument'):
        assert option in text
    # A destination rule outranks the source-port rule when both match a packet.
    assert "create_matcher(context, mask, 0)" in text and "create_matcher(context, source_mask, 1)" in text
