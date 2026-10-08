#!/usr/bin/env python3
"""Build a reproducible Linux ARM64 control-plane package from a clean commit.

The package carries the commit's source tree, its history as a Git bundle and
one compiled program: the relay marker (``sparkring-relay-marker``, from
``spark_transport/fabric/relay_marker.c``), which the fabric relay table runs
on every Spark. Compiling it needs a C compiler and the rdma-core development
headers (Debian packages ``gcc`` and ``libibverbs-dev``). ``--relay-marker
auto`` (the default) builds it when both are present and otherwise builds a
package without it, whose setup then installs no relay table; ``require``
stops instead, and ``skip`` never builds it. The package records the
binary's SHA-256 in ``distribution.json`` (``relay_marker``), which every
Spark checks before it runs the binary.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.common import distribution  # noqa: E402

MARKER_SOURCE = "spark_transport/fabric/relay_marker.c"
# Installed as /usr/lib/sparkring/bin/sparkring-relay-marker (runtime/host/relays.MARKER_BINARY).
MARKER_PATH = "bin/sparkring-relay-marker"
MARKER_HEADER = Path("/usr/include/infiniband/mlx5dv.h")
MARKER_MODES = ("auto", "require", "skip")


def relay_marker(payload, *, mode="auto", run=None, which=shutil.which, header=MARKER_HEADER, say=None):
    """Compile the relay marker into ``payload``; ``{"path", "sha256", "source_sha256"}`` or None.

    ``auto`` returns None, with a note on standard error, when the compiler
    or the rdma-core headers are missing; ``require`` raises ValueError then.
    """
    run = run or subprocess.run
    say = say or (lambda text: print(text, file=sys.stderr))
    if mode not in MARKER_MODES:
        raise ValueError("--relay-marker is auto, require or skip")
    source = Path(payload) / MARKER_SOURCE
    if mode == "skip":
        return None
    compiler = which("cc")
    missing = [name for name, present in (("a C compiler (gcc)", compiler), ("the rdma-core headers (libibverbs-dev)",
                                                                             Path(header).is_file()),
                                          ("the relay marker source", source.is_file())) if not present]
    if missing:
        text = "The relay marker was not built: " + " and ".join(missing) + " missing."
        if mode == "require":
            raise ValueError(text + " Install gcc and libibverbs-dev, then build again.")
        say(text + " Setup with this package installs no relay table; install gcc and libibverbs-dev and build "
            "again to include it.")
        return None
    target = Path(payload) / MARKER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    run([compiler, "-O2", "-Wall", "-Wextra", "-o", str(target), str(source), "-libverbs", "-lmlx5"], check=True)
    if not target.is_file():
        raise ValueError("The relay marker build produced no binary")
    return {"path": MARKER_PATH, "sha256": distribution.digest(target), "source_sha256": distribution.digest(source)}


def build(root, output, *, version=None, marker="auto"):
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
        built = relay_marker(payload, mode=marker)
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
        if built is not None:
            executables.add(payload / MARKER_PATH)
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
    return {"path": str(artifact), "sha256": digest, "revision": revision, "version": version,
            "relay_marker": built}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".sparkring/dist")
    parser.add_argument("--version")
    parser.add_argument("--relay-marker", choices=MARKER_MODES, default="auto",
                        help="compile the relay marker: when the compiler and rdma-core headers are present (auto), "
                             "always (require), or never (skip)")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(ROOT, args.output, version=args.version, marker=args.relay_marker), indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print("Package build: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
