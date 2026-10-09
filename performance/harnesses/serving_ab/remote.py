"""The site file, SSH to its Sparks, and per-Spark checkpoint discovery.

The site file is the ring harness's ``sircl-ring-site/v1`` (spark_transport/sircl/RUNBOOK.md, "Site file"):
every Spark's ``ssh`` target, ``lan_address`` and optional ``docker`` command come from it, so no host
detail lives in this package. Every command on a Spark runs through ``sudo -n`` (file reads included),
because checkpoints and caches are root-owned on these hosts and a non-sudo search reports nothing.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# SERVING_AB_SSH names another OpenSSH client, for example the Windows one from WSL
# (/mnt/c/Windows/System32/OpenSSH/ssh.exe) when the keys for the Sparks live on the Windows side.
SSH = (os.environ.get("SERVING_AB_SSH", "ssh"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
# Where a Spark keeps checkpoints, in the order searched: the installer's per-cluster stores, the shared
# model directory and the scratch copies. {owner}--{name} and {name} come from the profile's repository.
CHECKPOINT_PATTERNS = ("/srv/sparkring/*/checkpoints/{owner}--{name}/{revision}",
                       "/srv/models/{name}/{revision}",
                       "/var/tmp/models/{owner}--{name}/{revision}")


class RemoteError(RuntimeError):
    pass


@dataclass(frozen=True)
class Spark:
    position: int
    name: str
    ssh: str
    lan_address: str
    docker: tuple[str, ...]


@dataclass(frozen=True)
class Site:
    path: str
    lan_interface: str
    remote_dir: str
    sparks: tuple[Spark, ...]

    @property
    def size(self) -> int:
        return len(self.sparks)


def load_site(path: str | Path) -> Site:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != "sircl-ring-site/v1":
        raise RemoteError(f"{path}: not a sircl-ring-site/v1 site file")
    default = tuple(shlex.split(document.get("docker", "docker")))
    sparks = tuple(Spark(i, entry["name"], entry["ssh"], entry["lan_address"],
                         tuple(shlex.split(entry["docker"])) if entry.get("docker") else default)
                   for i, entry in enumerate(document["ring"]))
    return Site(str(path), document["lan_interface"], document.get("remote_dir", "/tmp/sircl-ring"), sparks)


def run(spark: Spark, command: str, *, check: bool = True, stdin: bytes | None = None, timeout: float = 300) -> str:
    result = subprocess.run([*SSH, spark.ssh, command], input=stdin, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RemoteError(f"{spark.name}: {command[:120]!r} exited {result.returncode}: "
                          f"{result.stderr.decode(errors='replace').strip()[-400:]}")
    return result.stdout.decode(errors="replace")


def sudo(command: str) -> str:
    return "sudo -n sh -c " + shlex.quote(command)


def find_checkpoint(spark: Spark, model: Mapping[str, str], patterns: Sequence[str] = CHECKPOINT_PATTERNS) -> dict:
    """The first copy of the profile's checkpoint on ``spark`` whose config and index digests match the pins."""
    owner, name = model["repository"].split("/", 1)
    candidates = [p.format(owner=owner, name=name, revision=model["revision"]) for p in patterns]
    script = "; ".join(
        f"for d in {pattern}; do [ -d \"$d\" ] || continue; "
        f"c=$(sha256sum \"$d/config.json\" 2>/dev/null | cut -d' ' -f1); "
        f"i=$(sha256sum \"$d/model.safetensors.index.json\" 2>/dev/null | cut -d' ' -f1); "
        f"n=$(find -L \"$d\" -name '*.safetensors' -type f | wc -l); "
        f"b=$(find -L \"$d\" -name '*.safetensors' -type f -printf '%s\\n' | awk '{{s+=$1}} END {{print s+0}}'); "
        f"echo \"$d $c $i $n $b\"; done" for pattern in candidates)
    found = []
    for line in run(spark, sudo(script), check=False).splitlines():
        fields = line.split()
        if len(fields) == 5:
            found.append({"path": fields[0], "config_sha256": fields[1], "index_sha256": fields[2],
                          "shards": int(fields[3]), "shard_bytes": int(fields[4])})
    for entry in found:
        entry["matches"] = (entry["config_sha256"] == model["config_sha256"]
                            and entry["index_sha256"] == model["index_sha256"])
    chosen = next((entry for entry in found if entry["matches"]), None)
    return {"spark": spark.name, "position": spark.position, "chosen": chosen, "candidates": found}
