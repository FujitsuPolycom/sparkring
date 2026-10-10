"""The prebuilt relay marker that the Debian package ships; no compiler runs and nothing is downloaded here."""
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts import build_deb

ROOT = Path(__file__).resolve().parents[1]
# The first 20 bytes of a 64-bit little-endian AArch64 executable, then a body.
MARKER = b"\x7fELF\x02\x01\x01" + bytes(9) + (2).to_bytes(2, "little") + (183).to_bytes(2, "little") + b"marker"
X86 = MARKER[:18] + (62).to_bytes(2, "little") + MARKER[20:]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def payload_with_source(tmp_path, **record):
    """An extracted source tree with the marker source and its record, the record's fields replaced by ``record``."""
    payload = tmp_path / "payload"
    for relative in (build_deb.MARKER_SOURCE, build_deb.MARKER_RECORD):
        (payload / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, payload / relative)
    path = payload / build_deb.MARKER_RECORD
    # The binary is unpublished unless ``record`` names one, whichever binary the committed record names.
    fields = {"binary_sha256": None, "download_url": None, **record}
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), **fields}), encoding="utf-8")
    return payload, payload / build_deb.MARKER_SOURCE


def published(data=MARKER):
    return {"binary_sha256": sha(data), "download_url": "https://example.invalid/sparkring-relay-marker"}


def compiler(data=MARKER):
    calls = []

    def run(argv, check, cwd):
        calls.append((argv, cwd))
        Path(argv[argv.index("-o") + 1]).write_bytes(data)
        return SimpleNamespace(returncode=0)
    return run, calls


def test_the_committed_record_names_the_committed_marker_source():
    record = build_deb.marker_record(ROOT)
    assert record["source_sha256"] == sha((ROOT / build_deb.MARKER_SOURCE).read_bytes())
    assert (record["binary_sha256"] is None) == (record["download_url"] is None)


def test_the_release_build_compiles_on_an_arm64_host_with_gcc_and_the_rdma_headers(tmp_path):
    payload, source = payload_with_source(tmp_path)
    header = tmp_path / "mlx5dv.h"
    header.write_text("")
    run, calls = compiler()
    notes = []
    built = build_deb.relay_marker(payload, mode="require", run=run, which=lambda name: "/usr/bin/cc",
                                   header=header, machine=lambda: "aarch64", say=notes.append)
    # The relative source path keeps the build directory's name out of the binary.
    assert calls == [(["/usr/bin/cc", "-O2", "-Wall", "-Wextra", "-o", str(payload / build_deb.MARKER_PATH),
                       build_deb.MARKER_SOURCE, "-libverbs", "-lmlx5"], str(payload))]
    assert built == {"path": "bin/sparkring-relay-marker", "sha256": sha(MARKER),
                     "source_sha256": sha(source.read_bytes()), "origin": "compiled"}
    # The record names no published binary, so the build says to publish this one.
    assert "Publish it and record both digests" in notes[0]


def test_a_release_build_without_its_toolchain_or_on_another_architecture_is_an_error(tmp_path):
    payload, _ = payload_with_source(tmp_path)
    header = tmp_path / "mlx5dv.h"
    with pytest.raises(ValueError, match="Release build: .*a C compiler \\(gcc\\) and the rdma-core headers"):
        build_deb.relay_marker(payload, mode="require", which=lambda name: None, header=header,
                               machine=lambda: "aarch64")
    header.write_text("")
    with pytest.raises(ValueError, match="compiled for arm64 on an arm64 host, and this host is x86_64"):
        build_deb.relay_marker(payload, mode="require", which=lambda name: "/usr/bin/cc", header=header,
                               machine=lambda: "x86_64")
    with pytest.raises(ValueError, match="produced no binary"):
        build_deb.relay_marker(payload, mode="require", run=lambda argv, check, cwd: None,
                               which=lambda name: "/usr/bin/cc", header=header, machine=lambda: "aarch64")
    with pytest.raises(ValueError, match="not a 64-bit ARM"):
        build_deb.relay_marker(payload, mode="require", run=compiler(X86)[0], which=lambda name: "/usr/bin/cc",
                               header=header, machine=lambda: "aarch64")


def test_a_release_build_keeps_the_published_binary_of_its_source(tmp_path):
    payload, _ = payload_with_source(tmp_path, **published())
    header = tmp_path / "mlx5dv.h"
    header.write_text("")
    notes = []
    built = build_deb.relay_marker(payload, mode="require", run=compiler()[0], which=lambda name: "/usr/bin/cc",
                                   header=header, machine=lambda: "aarch64", say=notes.append)
    assert built["origin"] == "compiled" and built["sha256"] == sha(MARKER) and notes == []
    with pytest.raises(ValueError, match="differs from the binary .* publishes for this source"):
        build_deb.relay_marker(payload, mode="require", run=compiler(MARKER + b"2")[0],
                               which=lambda name: "/usr/bin/cc", header=header, machine=lambda: "aarch64")


def test_the_default_build_downloads_the_published_binary_and_checks_it(tmp_path):
    payload, source = payload_with_source(tmp_path, **published())
    seen = []

    def download(url):
        seen.append(url)
        return MARKER
    built = build_deb.relay_marker(payload, download=download, run=lambda *a, **k: pytest.fail("compiled"))
    assert seen == ["https://example.invalid/sparkring-relay-marker"]
    assert built == {"path": "bin/sparkring-relay-marker", "sha256": sha(MARKER),
                     "source_sha256": sha(source.read_bytes()), "origin": "published"}
    assert (payload / build_deb.MARKER_PATH).read_bytes() == MARKER
    with pytest.raises(ValueError, match="has sha256 .*, not the published"):
        build_deb.relay_marker(payload, download=lambda url: MARKER + b"x")
    with pytest.raises(ValueError, match="Could not download the published relay marker .*--relay-marker-binary"):
        build_deb.relay_marker(payload, download=lambda url: (_ for _ in ()).throw(OSError("offline")))


def test_the_default_build_without_a_published_binary_for_its_source_is_an_error(tmp_path):
    payload, _ = payload_with_source(tmp_path)
    with pytest.raises(ValueError, match="names no published binary.*--relay-marker require.*--relay-marker-binary"):
        build_deb.relay_marker(payload, download=lambda url: pytest.fail("downloaded"))
    payload, _ = payload_with_source(tmp_path / "changed", source_sha256="0" * 64, **published())
    with pytest.raises(ValueError, match="names the binary of another marker source"):
        build_deb.relay_marker(payload, download=lambda url: pytest.fail("downloaded"))


def test_a_named_binary_must_be_the_published_one_when_the_record_names_one(tmp_path):
    binary = tmp_path / "sparkring-relay-marker"
    binary.write_bytes(MARKER)
    payload, _ = payload_with_source(tmp_path / "unpublished")
    notes = []
    built = build_deb.relay_marker(payload, binary=binary, say=notes.append)
    assert built["origin"] == "supplied" and "Publish it" in notes[0]
    payload, _ = payload_with_source(tmp_path / "published", **published())
    assert build_deb.relay_marker(payload, binary=binary)["origin"] == "published"
    payload, _ = payload_with_source(tmp_path / "other", **published(MARKER + b"2"))
    with pytest.raises(ValueError, match="is not the relay marker .* publishes"):
        build_deb.relay_marker(payload, binary=binary)
    binary.write_bytes(X86)
    payload, _ = payload_with_source(tmp_path / "x86")
    with pytest.raises(ValueError, match="not a 64-bit ARM"):
        build_deb.relay_marker(payload, binary=binary)


def test_skip_builds_no_marker_and_other_modes_are_refused(tmp_path):
    payload, _ = payload_with_source(tmp_path)
    assert build_deb.relay_marker(payload, mode="skip") is None
    with pytest.raises(ValueError, match="leave out --relay-marker-binary"):
        build_deb.relay_marker(payload, mode="skip", binary=tmp_path / "marker")
    with pytest.raises(ValueError, match="prebuilt, require or skip"):
        build_deb.relay_marker(payload, mode="auto")
    record = payload / build_deb.MARKER_RECORD
    record.write_text(json.dumps({**json.loads(record.read_text()), "binary_sha256": "ab" * 32}))
    with pytest.raises(ValueError, match="both set or both null"):
        build_deb.relay_marker(payload, download=lambda url: MARKER)


def test_the_package_records_the_marker_and_the_build_writes_it_beside_the_package(tmp_path, monkeypatch):
    import tarfile
    source = tmp_path / "source.tar"
    # The record with an unpublished binary, whichever binary the committed record names.
    unpublished = tmp_path / "relay-marker-artifact.json"
    unpublished.write_text(json.dumps({**json.loads((ROOT / build_deb.MARKER_RECORD).read_text(encoding="utf-8")),
                                       "binary_sha256": None, "download_url": None}), encoding="utf-8")
    with tarfile.open(source, "w") as tar:
        for relative in ("packaging/debian", build_deb.MARKER_SOURCE):
            tar.add(ROOT / relative, arcname=relative)
        tar.add(unpublished, arcname=build_deb.MARKER_RECORD)
    binary = tmp_path / "marker"
    binary.write_bytes(MARKER)
    seen = {}

    def run(argv, **kwargs):
        if argv[:2] == ["git", "archive"]:
            shutil.copyfile(source, argv[argv.index("-o") + 1])
        elif "bundle" in argv:
            Path(argv[argv.index("create") + 1]).write_bytes(b"bundle")
        elif argv[0] == "dpkg-deb":
            payload = Path(argv[-2]) / "usr/lib/sparkring"
            seen["record"] = json.loads((payload / "distribution.json").read_text())
            seen["binary"] = (payload / build_deb.MARKER_PATH).read_bytes()
            Path(argv[-1]).write_bytes(b"deb")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(build_deb.distribution, "identity", lambda root: "a" * 40)
    monkeypatch.setattr(build_deb, "subprocess", SimpleNamespace(run=run, check_output=lambda *a, **k: "1790000000\n"))
    result = build_deb.build(tmp_path, tmp_path / "dist", version="1.0", marker_binary=binary)
    source_sha256 = sha((ROOT / build_deb.MARKER_SOURCE).read_bytes())
    assert seen["binary"] == MARKER
    assert seen["record"]["relay_marker"] == {"path": "bin/sparkring-relay-marker", "sha256": sha(MARKER),
                                              "source_sha256": source_sha256, "origin": "supplied"}
    # The marker is not a source file: distribution.installed verifies the source tree only.
    assert build_deb.MARKER_PATH not in seen["record"]["files"]
    asset = tmp_path / "dist" / f"sparkring-relay-marker-{source_sha256[:12]}-arm64"
    assert result["relay_marker"]["asset"] == str(asset) and asset.read_bytes() == MARKER
    assert asset.with_name(asset.name + ".sha256").read_text() == f"{sha(MARKER)}  {asset.name}\n"


def test_the_marker_keeps_the_destination_command_line_and_adds_the_source_port_mode():
    text = (ROOT / build_deb.MARKER_SOURCE).read_text(encoding="utf-8")
    for option in ('{"device", required_argument', '{"rule", required_argument', '{"source-port", required_argument',
                   '{"run-seconds", required_argument', '{"managed", no_argument'):
        assert option in text
    # A destination rule outranks the source-port rule when both match a packet.
    assert "create_matcher(context, mask, 0)" in text and "create_matcher(context, source_mask, 1)" in text
