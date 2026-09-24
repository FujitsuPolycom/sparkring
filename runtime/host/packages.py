"""Create one offline Debian dependency bundle on the Internet-connected head."""
import inspect
import json
from pathlib import Path
import re
import subprocess
import tarfile

from runtime.common import distribution
from runtime.host import node


def install(directory, *, apply=False):
    """Self-contained worker installer. Its source travels with the package bundle."""
    import hashlib
    import json
    import os
    from pathlib import Path
    import platform
    import subprocess

    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    if (platform.machine() not in ("aarch64", "arm64") or
            any(release.get(k) != manifest["os"].get(k) for k in ("ID", "VERSION_ID"))):
        raise ValueError("Worker must match Node A's Linux distribution/version and ARM64 architecture")
    paths = []
    for name, digest in manifest["files"].items():
        path = root / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise ValueError("Unsafe package bundle entry")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("Package bundle checksum differs: " + name)
        if name.endswith(".deb"):
            paths.append(str(path))
    if not paths:
        raise ValueError("Bundle contains no Debian packages")
    applications = [path for path in paths if subprocess.check_output(["dpkg-deb", "-f", path, "Package"], text=True).strip() == "sparkring"]
    if len(applications) != 1 or "Packages" not in manifest["files"]:
        raise ValueError("Bundle must contain one SparkRing package and its local repository index")
    sources = root / "bundle.sources.list"
    sources.write_text("deb [trusted=yes] " + root.as_uri() + " ./\n")
    for directory in (root / "lists/partial", root / "archives/partial"):
        directory.mkdir(parents=True, exist_ok=True)
    options = ["-o", "Dir::Etc::sourcelist=" + str(sources), "-o", "Dir::Etc::sourceparts=-",
               "-o", "Dir::State::lists=" + str(root / "lists"), "-o", "Dir::Cache::archives=" + str(root / "archives"),
               "-o", "APT::Sandbox::User=root", "-o", "Acquire::Languages=none"]
    environment = {**os.environ, "DEBIAN_FRONTEND": "noninteractive", "NEEDRESTART_MODE": "l"}
    # The only repository is the verified local bundle. Install the application;
    # apt reuses satisfying dependencies instead of reinstalling Node A's OS.
    subprocess.run(["apt-get", *options, "update"], check=True, env=environment)
    command = ["apt-get", *options, "--no-install-recommends", "--no-remove", "-o", "Dpkg::Options::=--force-confdef", "-o", "Dpkg::Options::=--force-confold"]
    command += ["--yes"] if apply else ["--simulate"]
    subprocess.run([*command, "install", applications[0]], check=True, env=environment)


def repository_index(directory, *, run=subprocess.run):
    """Create a flat local apt repository from archive metadata and complete hashes."""
    directory = Path(directory).resolve()
    records = []
    for path in sorted(directory.glob("*.deb")):
        metadata = run(["dpkg-deb", "-f", str(path)], capture_output=True, text=True, check=True).stdout.strip()
        package = next(line.removeprefix("Package: ") for line in metadata.splitlines() if line.startswith("Package: "))
        if not re.fullmatch(r"[a-z0-9][a-z0-9+.-]*", package):
            raise ValueError("Invalid Debian package name")
        checksum = distribution.digest(path)
        destination = directory / (package + "_" + checksum[:16] + ".deb")
        if path.resolve().parent != directory or destination.resolve().parent != directory:
            raise ValueError("Debian archive must stay in the bundle directory")
        if destination != path:
            if destination.exists():
                raise ValueError("Duplicate package archive")
            path.rename(destination)
        records.append(metadata + f"\nFilename: ./{destination.name}\nSize: {destination.stat().st_size}\nSHA256: {checksum}\n")
    (directory / "Packages").write_text("\n".join(records), encoding="utf-8")


def build(directory, public_key, *, run=subprocess.run):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    if not distribution.installed(node.ROOT):
        raise ValueError("Worker bundles are built from an installed SparkRing package on Node A")
    result = run(["apt-cache", "depends", "--recurse", "--no-recommends", "--no-suggests",
                  "--no-conflicts", "--no-breaks", "--no-replaces", "--no-enhances", "sparkring"],
                 capture_output=True, text=True, check=True)
    names = sorted({line for line in result.stdout.splitlines() if re.fullmatch(r"[a-z0-9][a-z0-9+.-]*(?::[a-z0-9]+)?", line)})
    if "sparkring" not in names:
        raise ValueError("Installed package dependency closure is unavailable")
    run(["apt-get", "download", *[n for n in names if n != "sparkring"]], cwd=directory, check=True)
    run(["dpkg-repack", "sparkring"], cwd=directory, check=True)
    repository_index(directory, run=run)
    (directory / "controller.pub").write_text(public_key.strip() + "\n", encoding="utf-8")
    source = inspect.getsource(install)
    source += '''
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    install(directory, apply=args.apply)
    if args.prepare:
        if not args.apply:
            raise SystemExit("Worker preparation requires --apply")
        subprocess.run(["/usr/bin/sparkring", "node", "seed", "--key-file", str(directory / "controller.pub")], check=True)
'''
    (directory / "install.py").write_text(source, encoding="utf-8")
    release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    files = {p.name: distribution.digest(p) for p in directory.iterdir() if p.is_file()}
    (directory / "manifest.json").write_text(json.dumps({"schema": "sparkring-worker-bundle/v1", "os": release, "files": files}, indent=2) + "\n")
    archive = directory.with_suffix(".tar")
    with tarfile.open(archive, "w") as tar:
        for path in sorted(directory.iterdir()):
            tar.add(path, arcname=path.name, recursive=False)
    archive.chmod(0o600)
    return archive


def receive(path, digest):
    """Self-contained transfer receiver, running as the existing SSH login."""
    import hashlib
    from pathlib import Path
    import sys
    import tarfile
    root = Path(path)
    if not root.is_absolute() or root.parent != Path("/var/tmp") or not root.name.startswith("sparkring-enroll-"):
        raise ValueError("Invalid bootstrap staging directory")
    root.mkdir(mode=0o700)
    archive = root / "bundle.tar"
    checksum = hashlib.sha256()
    with archive.open("xb") as stream:
        while data := sys.stdin.buffer.read(1024 * 1024):
            checksum.update(data)
            stream.write(data)
    if checksum.hexdigest() != digest:
        raise ValueError("Worker bundle transfer checksum differs")
    with tarfile.open(archive) as tar:
        if any(not m.isfile() or Path(m.name).name != m.name for m in tar.getmembers()):
            raise ValueError("Invalid worker bundle archive")
        tar.extractall(root, filter="data")
    print(str(root))


def transfer(transport, route, archive, target):
    code = inspect.getsource(receive) + "\nreceive(" + repr(target) + ", " + repr(distribution.digest(archive)) + ")\n"
    # Stream bytes; command text contains only the destination and checksum.
    with Path(archive).open("rb") as stream:
        result = subprocess.run([*transport.argv(route), "python3 -I -c " + __import__("shlex").quote(code)],
                                stdin=stream, capture_output=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return target
