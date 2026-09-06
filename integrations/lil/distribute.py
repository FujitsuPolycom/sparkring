"""Download artifacts once and copy verified files to explicit destinations."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
import urllib.request


def digest(filename):
    with open(filename, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(manifest):
    if (
        set(manifest) != {"schema", "artifacts", "destinations"}
        or manifest["schema"] != "sparkring-artifacts/v1"
    ):
        raise ValueError("invalid artifact manifest")
    if not manifest["artifacts"] or not manifest["destinations"]:
        raise ValueError("artifacts and destinations are required")
    names = set()
    for a in manifest["artifacts"]:
        if set(a) != {"path", "source", "sha256"}:
            raise ValueError("artifact requires path, source, sha256")
        p = PurePosixPath(a["path"])
        if (
            p.is_absolute()
            or ".." in p.parts
            or str(p) != a["path"]
            or not re.fullmatch(r"[A-Za-z0-9_./-]+", a["path"])
            or a["path"] in names
            or str(p) == "."
        ):
            raise ValueError("artifact paths must be unique relative file paths")
        names.add(a["path"])
        if not re.fullmatch(r"[0-9a-f]{64}", a["sha256"]):
            raise ValueError("artifact requires SHA-256")
        if (
            not a["source"].startswith("https://")
            and not Path(a["source"]).is_absolute()
        ):
            raise ValueError("source must be HTTPS or an absolute local file path")
    seen = set()
    for d in manifest["destinations"]:
        if set(d) != {"host", "root"} or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9.-]*", d["host"]
        ):
            raise ValueError("destination requires host and root")
        p = PurePosixPath(d["root"])
        if (
            not p.is_absolute()
            or str(p) != d["root"]
            or ".." in p.parts
            or str(p) == "/"
            or not re.fullmatch(r"[A-Za-z0-9_./-]+", str(p))
        ):
            raise ValueError(
                "destination root must be a normalized absolute Linux path"
            )
        if d["host"] in seen:
            raise ValueError("duplicate destination host")
        seen.add(d["host"])


def acquire(artifact, cache):
    cache.mkdir(parents=True, exist_ok=True)
    stored = cache / artifact["sha256"]
    if stored.exists():
        if digest(stored) != artifact["sha256"]:
            raise ValueError("download cache checksum mismatch")
        return stored
    fd, staging = tempfile.mkstemp(prefix="download-", dir=cache)
    try:
        with os.fdopen(fd, "wb") as output:
            if artifact["source"].startswith("https://"):
                with urllib.request.urlopen(artifact["source"], timeout=60) as source:
                    if not source.url.startswith("https://"):
                        raise ValueError("HTTPS source redirected to an insecure URL")
                    shutil.copyfileobj(source, output)
            else:
                with open(artifact["source"], "rb") as source:
                    shutil.copyfileobj(source, output)
        if digest(staging) != artifact["sha256"]:
            raise ValueError("download checksum mismatch")
        try:
            os.link(staging, stored)
        except FileExistsError:
            if digest(stored) != artifact["sha256"]:
                raise ValueError("concurrent download checksum mismatch")
        return stored
    finally:
        os.unlink(staging)


class SSHCopy:
    def run(self, host, argv, missing_ok=False):
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                host,
                shlex.join(argv),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode and not (missing_ok and result.returncode == 1):
            raise RuntimeError(f"{host}: {result.stderr.strip()}")
        return result

    def put(self, source, destination, relative, checksum):
        host = destination["host"]
        final = str(PurePosixPath(destination["root"]) / relative)
        present = self.run(host, ["test", "-e", final], missing_ok=True)
        if present.returncode == 0:
            if self.run(host, ["sha256sum", "--", final]).stdout.split()[0] != checksum:
                raise ValueError(
                    f"{host}: destination differs; refusing overwrite: {final}"
                )
            return "reused"
        parent = str(PurePosixPath(final).parent)
        self.run(host, ["mkdir", "-p", "--", parent])
        staging = self.run(
            host, ["mktemp", "-p", parent, ".lil-copy-XXXXXXXX"]
        ).stdout.strip()
        if str(PurePosixPath(staging).parent) != parent or not PurePosixPath(
            staging
        ).name.startswith(".lil-copy-"):
            raise ValueError("unexpected remote staging path")
        try:
            subprocess.run(
                [
                    "scp",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    str(source),
                    f"{host}:{staging}",
                ],
                check=True,
                timeout=3600,
            )
            if (
                self.run(host, ["sha256sum", "--", staging]).stdout.split()[0]
                != checksum
            ):
                raise ValueError(f"{host}: transferred checksum mismatch")
            self.run(host, ["ln", "--", staging, final])
        finally:
            self.run(host, ["rm", "--", staging])
        return "copied"


def distribute(manifest, cache, transport):
    validate(manifest)
    # Verify every source before writing any destination.
    sources = [(a, acquire(a, cache)) for a in manifest["artifacts"]]
    return [
        {
            "host": d["host"],
            "path": a["path"],
            "outcome": transport.put(source, d, a["path"], a["sha256"]),
        }
        for d in manifest["destinations"]
        for a, source in sources
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument(
        "--execute",
        action="store_true",
        help="download and copy files to the manifest's explicit hosts",
    )
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    validate(manifest)
    if not a.execute:
        print(json.dumps({"execute": False, "manifest": manifest}, indent=2))
    else:
        print(json.dumps(distribute(manifest, a.cache, SSHCopy()), indent=2))


if __name__ == "__main__":
    main()
