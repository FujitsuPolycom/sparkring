"""Copy a verified file between ranks using an explicit authenticated SSH route."""

import re
from pathlib import PurePosixPath


def copy_edge(
    run,
    source_host,
    destination_host,
    peer,
    source,
    destination,
    checksum,
    *,
    pull=False,
):
    """run(host, argv) executes on a management host; payload uses the peer address.

    The receiver publishes with a hard link, refusing to overwrite existing data.
    Both push and pull support sites with asymmetric SSH authorization.
    """
    for host in (source_host, destination_host, peer):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host):
            raise ValueError("invalid SSH host")
    for filename in (source, destination):
        p = PurePosixPath(filename)
        if (
            not p.is_absolute()
            or str(p) != filename
            or ".." in p.parts
            or str(p) == "/"
            or not re.fullmatch(r"[A-Za-z0-9_./-]+", filename)
        ):
            raise ValueError("invalid artifact path")
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("SHA-256 required")
    if run(source_host, ["sha256sum", "--", source]).split()[0] != checksum:
        raise ValueError("source checksum mismatch")
    parent = str(PurePosixPath(destination).parent)
    run(destination_host, ["mkdir", "-p", "--", parent])
    stage = run(
        destination_host, ["mktemp", "-p", parent, ".lil-fanout-XXXXXXXX"]
    ).strip()
    if str(PurePosixPath(stage).parent) != parent or not PurePosixPath(
        stage
    ).name.startswith(".lil-fanout-"):
        raise ValueError("invalid staging path")
    try:
        options = [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=yes",
        ]
        if pull:
            run(destination_host, options + [peer + ":" + source, stage])
        else:
            run(source_host, options + [source, peer + ":" + stage])
        if run(destination_host, ["sha256sum", "--", stage]).split()[0] != checksum:
            raise ValueError("destination checksum mismatch")
        run(destination_host, ["ln", "--", stage, destination])
    finally:
        run(destination_host, ["rm", "--", stage])
    return {
        "source_host": source_host,
        "destination_host": destination_host,
        "peer": peer,
        "pull": pull,
        "sha256": checksum,
    }
