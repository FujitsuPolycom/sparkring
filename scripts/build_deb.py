#!/usr/bin/env python3
"""Build a reproducible Linux ARM64 control-plane package from a clean commit.

The package carries the commit's source tree, its history as a Git bundle and
one program built from C: the relay marker (``sparkring-relay-marker``, from
``spark_transport/fabric/relay_marker.c``), which the fabric relay table runs
on every Spark. The marker ships prebuilt, so no Spark compiles it:

- ``--relay-marker prebuilt`` (the default, and what ``install.sh`` runs)
  takes the published binary that the marker record
  (``spark_transport/fabric/relay-marker-artifact.json``) names for this
  source, downloads it and checks its SHA-256, or takes the file
  ``--relay-marker-binary`` names. It never runs a compiler.
- ``--relay-marker require`` is the release build: it compiles the marker on
  an arm64 Linux host with a C compiler and the rdma-core development headers
  (Debian packages ``gcc`` and ``libibverbs-dev``) and stops when any of them
  is missing.
- ``--relay-marker skip`` builds a package without the marker, for package
  tests; setup with that package installs no relay table.

Every mode except ``skip`` also writes the binary and its ``.sha256`` file
beside the package, for publication. ``distribution.json`` records the
binary's SHA-256 (``relay_marker``); the package's installation checks the
installed binary against it, and every Spark checks it again before it runs
the marker.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import distribution  # noqa: E402

MARKER_SOURCE = "spark_transport/fabric/relay_marker.c"
# The published binary for the marker source: sparkring-relay-marker-artifact/v1.
MARKER_RECORD = "spark_transport/fabric/relay-marker-artifact.json"
MARKER_RECORD_SCHEMA = "sparkring-relay-marker-artifact/v1"
# Installed as /usr/lib/sparkring/bin/sparkring-relay-marker (runtime/host/relays.MARKER_BINARY).
MARKER_PATH = "bin/sparkring-relay-marker"
MARKER_HEADER = Path("/usr/include/infiniband/mlx5dv.h")
MARKER_MODES = ("prebuilt", "require", "skip")
# The marker is a small program; a larger download is not the published binary.
MARKER_MAX_BYTES = 1 << 20
# ELF header: 64-bit, little-endian, machine 183 (AArch64).
AARCH64 = 183


def marker_record(root):
    """The marker record of a source tree, checked for shape; raises ValueError."""
    path = Path(root) / MARKER_RECORD
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"The relay marker record {MARKER_RECORD} is unreadable: {error}") from error
    if not isinstance(record, dict) or record.get("schema") != MARKER_RECORD_SCHEMA:
        raise ValueError(f"{MARKER_RECORD} is not a {MARKER_RECORD_SCHEMA} record")
    if record.get("source") != MARKER_SOURCE or record.get("architecture") != "arm64":
        raise ValueError(f"{MARKER_RECORD} names another source or architecture")
    if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("source_sha256"))):
        raise ValueError(f"{MARKER_RECORD} has no source_sha256")
    binary, url = record.get("binary_sha256"), record.get("download_url")
    if (binary is None) != (url is None):
        raise ValueError(f"{MARKER_RECORD}: binary_sha256 and download_url are both set or both null")
    if binary is not None and (not re.fullmatch(r"[0-9a-f]{64}", str(binary))
                               or not str(url).startswith("https://")):
        raise ValueError(f"{MARKER_RECORD}: binary_sha256 is a SHA-256 and download_url an https URL")
    return record


def aarch64_executable(data):
    """Whether ``data`` starts with the ELF header of a 64-bit little-endian AArch64 program."""
    return (len(data) >= 20 and data[:4] == b"\x7fELF" and data[4] == 2 and data[5] == 1
            and int.from_bytes(data[18:20], "little") == AARCH64)


def fetch(url, *, limit=MARKER_MAX_BYTES, opener=urllib.request.urlopen):
    """The bytes at ``url``; raises ValueError above ``limit`` bytes."""
    with opener(url, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"The relay marker download exceeds {limit} bytes")
    return data


def relay_marker(payload, *, mode="prebuilt", binary=None, run=None, which=shutil.which, header=MARKER_HEADER,
                 machine=platform.machine, download=fetch, say=None):
    """Place the relay marker in ``payload``; ``{"path", "sha256", "source_sha256", "origin"}`` or None.

    ``payload`` is the extracted source tree. ``origin`` is ``published``
    (the record's binary), ``supplied`` (``binary``, a file the builder
    names, when the record names no binary for this source) or ``compiled``
    (``require``). Raises ValueError whenever the marker cannot be placed;
    only ``skip`` returns None.
    """
    run = run or subprocess.run
    say = say or (lambda text: print(text, file=sys.stderr))
    if mode not in MARKER_MODES:
        raise ValueError("--relay-marker is prebuilt, require or skip")
    if mode == "skip":
        if binary is not None:
            raise ValueError("--relay-marker skip builds no marker; leave out --relay-marker-binary")
        return None
    payload = Path(payload)
    source = payload / MARKER_SOURCE
    if not source.is_file():
        raise ValueError(f"The relay marker source {MARKER_SOURCE} is missing")
    source_sha256 = distribution.digest(source)
    record = marker_record(payload)
    # The record's binary is the published build of exactly this source.
    published = record["binary_sha256"] if record["source_sha256"] == source_sha256 else None
    target = payload / MARKER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "require":
        if binary is not None:
            raise ValueError("--relay-marker require compiles the marker; leave out --relay-marker-binary")
        if machine() not in ("aarch64", "arm64"):
            raise ValueError(f"Release build: the relay marker is compiled for arm64 on an arm64 host, and this host "
                             f"is {machine()}. Build on a DGX Spark or another arm64 Linux host.")
        compiler = which("cc")
        missing = [name for name, present in (("a C compiler (gcc)", compiler),
                                              ("the rdma-core headers (libibverbs-dev)", Path(header).is_file()))
                   if not present]
        if missing:
            raise ValueError("Release build: the relay marker cannot be compiled: " + " and ".join(missing)
                             + " missing. Install gcc and libibverbs-dev on the build host and build again.")
        # The relative source path keeps the build directory's name out of the binary.
        run([compiler, "-O2", "-Wall", "-Wextra", "-o", str(target), MARKER_SOURCE, "-libverbs", "-lmlx5"],
            check=True, cwd=str(payload))
        if not target.is_file():
            raise ValueError("Release build: the relay marker build produced no binary")
        data, origin = target.read_bytes(), "compiled"
        digest = hashlib.sha256(data).hexdigest()
        if published is not None and digest != published:
            raise ValueError(f"Release build: the compiled relay marker (sha256 {digest}) differs from the binary "
                             f"{MARKER_RECORD} publishes for this source ({published}). Every package of one source "
                             "carries one binary: build with the toolchain that built the published one, or publish "
                             "this binary and record it.")
    elif binary is not None:
        data = Path(binary).read_bytes()
        if len(data) > MARKER_MAX_BYTES:
            raise ValueError(f"{binary} exceeds {MARKER_MAX_BYTES} bytes; it is not a relay marker")
        digest = hashlib.sha256(data).hexdigest()
        if published is not None and digest != published:
            raise ValueError(f"{binary} (sha256 {digest}) is not the relay marker {MARKER_RECORD} publishes for "
                             f"this source ({published})")
        origin = "published" if published is not None else "supplied"
    else:
        if published is None:
            reason = ("names no published binary" if record["source_sha256"] == source_sha256
                      else "names the binary of another marker source")
            raise ValueError(f"{MARKER_RECORD} {reason}, so this package build has no relay marker to ship. "
                             "Build the release on an arm64 host with gcc and libibverbs-dev "
                             "(--relay-marker require), or pass a relay marker built from this source with "
                             "--relay-marker-binary PATH.")
        try:
            data = download(record["download_url"])
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not download the published relay marker {record['download_url']} ({error}); "
                             "pass a copy with --relay-marker-binary PATH") from error
        digest = hashlib.sha256(data).hexdigest()
        if digest != published:
            raise ValueError(f"The downloaded relay marker has sha256 {digest}, not the published {published}")
        origin = "published"
    if not aarch64_executable(data):
        raise ValueError("The relay marker is not a 64-bit ARM (AArch64) program")
    if origin != "compiled":
        target.write_bytes(data)
    if origin in ("compiled", "supplied") and published is None:
        say(f"Relay marker {origin} for source sha256 {source_sha256}: sha256 {digest}. Publish it and record both "
            f"digests and its URL in {MARKER_RECORD}, so that package builds without a compiler take it.")
    return {"path": MARKER_PATH, "sha256": digest, "source_sha256": source_sha256, "origin": origin}


def build(root, output, *, version=None, marker="prebuilt", marker_binary=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    revision = distribution.identity(root)
    epoch = int(subprocess.check_output(["git", "show", "-s", "--format=%ct", revision], cwd=root, text=True))
    version = version or f"0.1.0~dev.{epoch}+git" + revision[:12]
    if not re.fullmatch(r"[0-9][A-Za-z0-9.+~]*", version):
        raise ValueError("Use a Debian upstream version without a revision suffix")
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / f"sparkring_{version}_arm64.deb"
    if artifact.exists():
        raise ValueError("Package already exists; choose a separate output directory")
    with tempfile.TemporaryDirectory(prefix="sparkring-deb-") as temp:
        work = Path(temp)
        package = work / "package"
        payload = package / "usr/lib/sparkring"
        payload.mkdir(parents=True)
        archive = work / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), revision], cwd=root, check=True)
        with tarfile.open(archive) as tar:
            # Files shipped by the package must be regular source files.
            if any(not m.isfile() and not m.isdir() for m in tar.getmembers()):
                raise ValueError("Distribution source contains nonregular entries")
            tar.extractall(payload, filter="data")
        files = {p.relative_to(payload).as_posix(): distribution.digest(p) for p in sorted(payload.rglob("*")) if p.is_file()}
        # Built after the source files are recorded: distribution.installed
        # verifies the source tree, and relays.marker_artifact the binary.
        built = relay_marker(payload, mode=marker, binary=marker_binary)
        bundle = payload / "source.bundle"
        # A single-tip, deterministic pack avoids reflog/other-branch identities.
        subprocess.run(["git", "-c", "pack.threads=1", "bundle", "create", str(bundle), "HEAD"], cwd=root, check=True, capture_output=True)
        record = {"schema": "sparkring-distribution/v1", "revision": revision, "version": version,
                  "source_date_epoch": epoch, "architecture": "arm64", "files": files,
                  "bundle_sha256": distribution.digest(bundle)}
        if built is not None:
            record["relay_marker"] = built
        (payload / "distribution.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        bin_dir = package / "usr/bin"
        units = package / "usr/lib/systemd/system"
        generators = package / "usr/lib/systemd/system-generators"
        control = package / "DEBIAN"
        for directory in (bin_dir, units, generators, control):
            directory.mkdir(parents=True)
        templates = payload / "packaging/debian"
        shutil.copyfile(templates / "sparkring", bin_dir / "sparkring")
        # systemd runs this generator at boot and on every daemon-reload; it adds
        # the ConnectX hairpin start check to the host's mesh units.
        generator = generators / "sparkring-hairpin-mesh-check"
        shutil.copyfile(templates / "sparkring-hairpin-mesh-check", generator)
        for unit in [*templates.glob("*.service"), *templates.glob("*.timer")]:
            shutil.copyfile(unit, units / unit.name)
        for name in ("postinst", "prerm", "postrm"):
            shutil.copyfile(templates / name, control / name)
        size = sum(p.stat().st_size for p in package.rglob("*") if p.is_file()) // 1024
        (control / "control").write_text(f"""Package: sparkring
Version: {version}
Section: admin
Priority: optional
Architecture: arm64
Maintainer: SparkRing contributors
Depends: python3 (>= 3.12), python3-yaml, git, openssh-client, openssh-server, sudo, iproute2, iputils-ping, rdma-core, ibverbs-utils, ibverbs-providers, perftest, pciutils, ethtool, avahi-daemon, avahi-utils, lldpd, iptables, systemd, systemd-resolved, wireguard-tools, dnsmasq-base, dpkg-repack, rsync
Installed-Size: {size}
Description: SparkRing host setup and profile deployment controller
 Discovers and configures pairs, lines and rings of up to eight DGX Sparks.
 Uses existing Docker, NVIDIA drivers and NetworkManager installations.
 Images and model weights are selected and admitted separately.
""", encoding="utf-8")
        # Do not ship machine-specific state or conffiles. initialize generates
        # per-node identity once; approved fabric state is retained on removal.
        executables = {bin_dir / "sparkring", generator, *(control / n for n in ("postinst", "prerm", "postrm"))}
        marker_bytes = None
        if built is not None:
            executables.add(payload / MARKER_PATH)
            marker_bytes = (payload / MARKER_PATH).read_bytes()
        for path in [package, *package.rglob("*")]:
            if path.is_dir():
                path.chmod(0o755)
            elif path in executables:
                path.chmod(0o755)
            else:
                # Preserve executable source entrypoints from the Git archive.
                path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
            os.utime(path, (epoch, epoch))
        environment = {**os.environ, "SOURCE_DATE_EPOCH": str(epoch), "TZ": "UTC", "LC_ALL": "C"}
        # dpkg-deb reports progress on stdout; keep stdout for the JSON result.
        subprocess.run(["dpkg-deb", "--root-owner-group", "-Zxz", "-z6", "--build", str(package), str(artifact)],
                       check=True, env=environment, stdout=sys.stderr)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (artifact.with_suffix(".deb.sha256")).write_text(digest + "  " + artifact.name + "\n", encoding="utf-8")
    if built is not None:
        # The binary beside the package is the release asset that the marker record names.
        asset = output / f"sparkring-relay-marker-{built['source_sha256'][:12]}-arm64"
        asset.write_bytes(marker_bytes)
        asset.with_name(asset.name + ".sha256").write_text(built["sha256"] + "  " + asset.name + "\n",
                                                            encoding="utf-8")
        built = dict(built, asset=str(asset))
    return {"path": str(artifact), "sha256": digest, "revision": revision, "version": version,
            "relay_marker": built}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".sparkring/dist")
    parser.add_argument("--version")
    parser.add_argument("--relay-marker", choices=MARKER_MODES, default="prebuilt",
                        help="the relay marker: the published binary, downloaded and checked (prebuilt, the "
                             "default); compiled on this arm64 host, the release build (require); or none, for "
                             "package tests (skip)")
    parser.add_argument("--relay-marker-binary", type=Path, metavar="PATH",
                        help="with prebuilt: ship this relay marker binary instead of downloading the published one; "
                             "it must equal the published binary when the marker record names one")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(ROOT, args.output, version=args.version, marker=args.relay_marker,
                               marker_binary=args.relay_marker_binary), indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print("Package build: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
