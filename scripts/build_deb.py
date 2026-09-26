#!/usr/bin/env python3
"""Build a reproducible Linux ARM64 control-plane package from a clean commit."""
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


def build(root, output, *, version=None):
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
        bundle = payload / "source.bundle"
        # A single-tip, deterministic pack avoids reflog/other-branch identities.
        subprocess.run(["git", "-c", "pack.threads=1", "bundle", "create", str(bundle), "HEAD"], cwd=root, check=True, capture_output=True)
        record = {"schema": "sparkring-distribution/v1", "revision": revision, "version": version,
                  "source_date_epoch": epoch, "architecture": "arm64", "files": files,
                  "bundle_sha256": distribution.digest(bundle)}
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
Depends: python3 (>= 3.12), python3-yaml, git, openssh-client, openssh-server, sudo, iproute2, iputils-ping, rdma-core, ibverbs-utils, pciutils, ethtool, avahi-daemon, avahi-utils, lldpd, iptables, systemd, systemd-resolved, wireguard-tools, dnsmasq-base, dpkg-repack, rsync
Installed-Size: {size}
Description: SparkRing host setup and profile deployment controller
 Discovers and configures supported pairs and four-node DGX Spark rings.
 Uses existing Docker, NVIDIA drivers and NetworkManager installations.
 Images and model weights are selected and admitted separately.
""", encoding="utf-8")
        # Do not ship machine-specific state or conffiles. initialize generates
        # per-node identity once; approved fabric state is retained on removal.
        executables = {bin_dir / "sparkring", generator, *(control / n for n in ("postinst", "prerm", "postrm"))}
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
    return {"path": str(artifact), "sha256": digest, "revision": revision, "version": version}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".sparkring/dist")
    parser.add_argument("--version")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(ROOT, args.output, version=args.version), indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print("Package build: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
