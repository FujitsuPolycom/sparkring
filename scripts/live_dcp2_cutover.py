#!/usr/bin/env python3
"""Clone a live four-rank NF3 DCP4 deployment into a fail-closed DCP2
SparkCache trial.

This is intentionally an operator tool, not a general SparkRing launcher.
It reads the live container definitions, normalizes the real
``/usr/bin/env -u ... /opt/venv/bin/vllm`` wrapper into a clean vLLM
argv, changes exactly five DCP/capacity fields, removes known-invalid
inherited variables, applies the SparkCache config generator to produce
a cache-enabled argv/env, pre-creates the replacement containers
(including a read-only connector-staging bind mount and PYTHONPATH), and
retains the stopped sources for rollback.

Safety classes:
  - plan, status: READ-ONLY REMOTE (docker inspect only)
  - prepare: MUTATES HOST (creates containers), confirmation-gated
  - cutover: STOPS SERVING (stops sources, starts targets), confirmation-gated
  - rollback: STOPS SERVING (stops targets, restarts sources), confirmation-gated

The script never deletes evidence-bearing containers (the stopped DCP2
candidates or the DCP4 sources) except for cleanup of containers newly
created by a failed prepare in the same invocation.

Connector staging (future MUTATES HOST step, documented but unexecuted):
  The deployed image does NOT contain ``spark_context_cache_connector``
  (``find_spec`` returns ``None``).  The operator must stage the
  connector source on the host at an explicit ``--connector-staging``
  directory.  ``prepare`` bind-mounts that directory read-only into
  the target at ``/opt/sparkcache-staging`` and sets
  ``PYTHONPATH=/opt/sparkcache-staging:/opt/sparkcache-staging/sparkcache``
  so that both top-level ``spark_context_cache_{connector,codec,store}``
  and the ``sparkcache.streaming`` package are importable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import enum

import json
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Import the pure config generator for cache-enabled argv/env transformation.
from sparkcache_config_generator import (
    generate_sparkcache_argv,
    verify_sparkcache_argv,
)

# Connector bundle identity: pins the exact connector code staged on each host.
# The operator generates the identity offline with connector_bundle_manifest.py
# and supplies it via --connector-bundle-identity.  plan/prepare/cutover
# verify the staged files match this identity on every host before proceeding.
# REQUIRED_FILES is the single canonical allowlist shared by the manifest
# builder and the remote verifier — no duplicated file lists.
from connector_bundle_manifest import REQUIRED_FILES as _BUNDLE_REQUIRED_FILES
from connector_bundle_manifest import BUNDLE_DOMAIN_SEPARATOR as _BUNDLE_DOMAIN

BUNDLE_IDENTITY_LABEL = "sparkcache-bundle-identity"
# ---------------------------------------------------------------------------
# Remote bundle verifier script
#
# Sent to the remote host via
# ``python3 -c <script> <staging-root> <required-files-json> <domain-separator>``.
# Applies the same fail-closed semantics as ``inventory_staging_root``:
# lstat (no symlink dereference), reject non-regular files via stat.S_ISREG,
# reject missing/extra files, hash in 1 MiB chunks, compare stable stat metadata
# before/after every read, compute the domain-separated identity using the
# imported BUNDLE_DOMAIN_SEPARATOR (not a hard-coded constant), and
# emit a single JSON line with either {"identity": "..."} or {"error": "..."}.
# ---------------------------------------------------------------------------
_REMOTE_VERIFIER_SCRIPT = r"""
import hashlib, json, os, stat, sys, unicodedata
from pathlib import PurePosixPath

CHUNK = 1 << 20

def fail(msg):
    print(json.dumps({"error": msg}))
    sys.exit(0)

def lstat_check(path):
    try:
        return os.lstat(path)
    except OSError as e:
        fail(f"cannot lstat {path}: {e}")

def hash_file(path, st_before):
    h = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(CHUNK)
                if not chunk:
                    break
                h.update(chunk)
                size += len(chunk)
    except OSError as e:
        fail(f"cannot read {path}: {e}")
    st_after = os.lstat(path)
    if (st_before.st_size != st_after.st_size or
        st_before.st_ino != st_after.st_ino or
        st_before.st_dev != st_after.st_dev or
        st_before.st_mtime_ns != st_after.st_mtime_ns or
        st_before.st_ctime_ns != st_after.st_ctime_ns):
        fail(f"file changed during read: {path}")
    if size != st_before.st_size:
        fail(
            f"bytes read differ from pre-read size for {path}: "
            f"read {size}, expected {st_before.st_size}"
        )
    return h.hexdigest(), size

def normalize_rel(root, abs_path):
    rel = os.path.relpath(abs_path, root)
    posix = PurePosixPath(*rel.split(os.sep)).as_posix()
    return unicodedata.normalize("NFC", posix)

def walk_fail(error):
    fail(f"walk error: {error}")

root = sys.argv[1]
required = set(json.loads(sys.argv[2]))
domain = sys.argv[3]

# Root validation: lstat, reject symlink, require directory.
root_st = lstat_check(root)
if stat.S_ISLNK(root_st.st_mode):
    fail(f"staging root is a symlink: {root}")
if not stat.S_ISDIR(root_st.st_mode):
    fail(f"staging root is not a directory: {root}")

entries = []
seen = set()
for rel in sorted(required):
    abs_path = os.path.join(root, rel)
    st = lstat_check(abs_path)
    if stat.S_ISLNK(st.st_mode):
        fail(f"required file is a symlink: {rel}")
    if not stat.S_ISREG(st.st_mode):
        fail(f"required file missing or not regular: {rel}")
    norm = normalize_rel(root, abs_path)
    if norm in seen:
        fail(f"duplicate normalized path: {norm}")
    seen.add(norm)
    content_sha, size = hash_file(abs_path, st)
    entries.append((norm, size, content_sha))

# Walk tree to find extra files (fail-closed on walk errors).
for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=walk_fail):
    for dirname in dirnames:
        full = os.path.join(dirpath, dirname)
        dst = lstat_check(full)
        if stat.S_ISLNK(dst.st_mode):
            fail(f"symlink directory found: {full}")
    for filename in filenames:
        abs_file = os.path.join(dirpath, filename)
        rel = normalize_rel(root, abs_file)
        if rel not in required:
            fail(f"extra file not in allowlist: {rel}")

h = hashlib.sha256()
h.update(domain.encode("utf-8"))
for rel, size, content_sha in entries:
    h.update(b"\x00")
    h.update(rel.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(size).encode("utf-8"))
    h.update(b"\x00")
    h.update(content_sha.encode("utf-8"))
print(json.dumps({"identity": h.hexdigest()}))
"""

# ---------------------------------------------------------------------------
# Container naming
# ---------------------------------------------------------------------------

SOURCE_PATTERN = "glm52-sparkring-nf3-dcp4-fixedk2-r{rank}"
BASELINE_DCP2_PATTERN = "glm52-sparkring-nf3-dcp2-r{rank}"
TARGET_PATTERN = "glm52-sparkring-nf3-dcp2-sparkcache-r{rank}"

PREPARE_CONFIRMATION = "PREPARE-NF3-DCP2-SPARKCACHE-ALL-FOUR"
CUTOVER_CONFIRMATION = "STOP-DCP4-START-DCP2-SPARKCACHE"
ROLLBACK_CONFIRMATION = "STOP-DCP2-SPARKCACHE-RESTORE-DCP4"

# Label applied to every target created by a prepare invocation so that
# cleanup can identify exactly the containers from that invocation, even
# if the create response was lost.
PREPARE_LABEL = "sparkcache-prepare"

# DCP2 rewrites applied before the config generator sees the command.
ARG_REWRITES = {
    "--decode-context-parallel-size": ("4", "2"),
    "--max-model-len": ("1048576", "524288"),
}
ENV_REWRITES = {
    "VLLM_SPARK_DCP_SIZE": ("4", "2"),
    "VLLM_SPARK_MAX_MODEL_LEN": ("1048576", "524288"),
    "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": ("1048576", "524288"),
}
RUNTIME_UNSET_ENVIRONMENT = {
    "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": "",
}

# Connector staging mount point inside the target container.
STAGING_DESTINATION = "/opt/sparkcache-staging"
# PYTHONPATH covering both top-level modules and the sparkcache package.
STAGING_PYTHONPATH = f"{STAGING_DESTINATION}:{STAGING_DESTINATION}/sparkcache"

# Health polling defaults.
HEALTH_TIMEOUT_S = 120
HEALTH_INTERVAL_S = 2
SSH_CONNECT_TIMEOUT_S = 10
SSH_MAX_TIMEOUT_S = 300

# The real entrypoint of the deployed image.
_WRAPPER_ENTRYPOINT = "/usr/bin/env"
_VLLM_BINARY = "/opt/venv/bin/vllm"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CutoverError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    rank: int
    ssh_target: str

    @property
    def source_name(self) -> str:
        return SOURCE_PATTERN.format(rank=self.rank)

    @property
    def target_name(self) -> str:
        return TARGET_PATTERN.format(rank=self.rank)

    @property
    def baseline_name(self) -> str:
        return BASELINE_DCP2_PATTERN.format(rank=self.rank)


# ---------------------------------------------------------------------------
# Existence semantics (tri-state, fail-closed)
# ---------------------------------------------------------------------------


class Existence(enum.Enum):
    """Tri-state existence check for remote containers.

    - ``PROVEN_ABSENT``: Docker confirmed "No such object" (rc != 0
      AND stderr contains the Docker no-such-object sentinel).
    - ``PROVEN_PRESENT``: Docker inspect returned valid JSON (rc == 0).
    - ``UNKNOWN``: Transport/auth/daemon failure (rc 255, timeout, JSON
      decode error, etc.).  Callers MUST treat this as failure, never
      as absence.
    """

    PROVEN_ABSENT = "absent"
    PROVEN_PRESENT = "present"
    UNKNOWN = "unknown"


# Docker's "No such object" error sentinel (appears in stderr).
_DOCKER_NO_SUCH_OBJECT_SENTINELS = (
    "No such container",
    "no such object",
    "No such object",
)


def _check_exists(node: Node, name: str) -> Existence:
    """Determine container existence with fail-closed semantics.

    Returns ``Existence.PROVEN_ABSENT`` only when Docker itself reports
    "No such object/No such container" with return code 1.  SSH
    failures (rc 255), daemon errors, other nonzero codes, JSON decode
    errors, etc. return ``Existence.UNKNOWN`` — callers must never
    treat ``UNKNOWN`` as absence.  A nonzero rc other than 1 is a
    transport/daemon failure even if stderr happens to contain the
    sentinel text.
    """
    result = _remote_result(node, ("docker", "inspect", name))
    if result.returncode == 0:
        try:
            doc = json.loads(result.stdout)
            if isinstance(doc, list) and len(doc) >= 1:
                return Existence.PROVEN_PRESENT
        except (json.JSONDecodeError, TypeError):
            return Existence.UNKNOWN
        return Existence.UNKNOWN
    # Only Docker rc=1 with the exact no-such-object sentinel proves
    # absence.  rc=255 (SSH), rc=125 (daemon), or any other code is a
    # transport/infrastructure failure even if stderr contains the
    # sentinel text — fail-closed as UNKNOWN.
    if result.returncode == 1:
        stderr_lower = (result.stderr or "").lower()
        if any(s.lower() in stderr_lower for s in _DOCKER_NO_SUCH_OBJECT_SENTINELS):
            return Existence.PROVEN_ABSENT
    return Existence.UNKNOWN


# ---------------------------------------------------------------------------
# Remote execution
# ---------------------------------------------------------------------------


def _ssh_base() -> list[str]:
    """Return SSH options applied to every remote invocation."""
    return [
        "ssh",
        "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
        "-o", "BatchMode=yes",
    ]


def _remote_result(node: Node, argv: Iterable[str]) -> subprocess.CompletedProcess[str]:
    """Run a remote command, returning the full CompletedProcess.

    Never raises on non-zero exit; caller inspects returncode/stdout/stderr.
    Raises CutoverError only on subprocess.TimeoutExpired.
    """
    command = shlex.join(tuple(argv))
    try:
        result = subprocess.run(
            [*_ssh_base(), node.ssh_target, command],
            check=False,
            capture_output=True,
            text=True,
            timeout=SSH_MAX_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise CutoverError(
            f"rank {node.rank} SSH timeout after {SSH_MAX_TIMEOUT_S}s: {command}"
        ) from exc
    return result


def _remote(node: Node, argv: Iterable[str], *, check: bool = True) -> str:
    """Run a remote command, returning stdout. Raises on non-zero if check."""
    result = _remote_result(node, argv)
    if check and result.returncode != 0:
        raise CutoverError(
            f"rank {node.rank} remote command failed ({result.returncode}): "
            f"{shlex.join(tuple(argv))}\nstdout={result.stdout.strip()}\n"
            f"stderr={result.stderr.strip()}"
        )
    return result.stdout


def _inspect(node: Node, name: str) -> dict[str, Any]:
    raw = _remote(node, ("docker", "inspect", name))
    document = json.loads(raw)
    if len(document) != 1:
        raise CutoverError(
            f"rank {node.rank} docker inspect returned {len(document)} records"
        )
    return document[0]


def _label_for(node: Node, token: str) -> str:
    """Return the label string applied to targets from a prepare invocation."""
    return f"{PREPARE_LABEL}={token}"


def _has_label(document: dict[str, Any], label_key: str, label_value: str) -> bool:
    """Check if a container document has the given label."""
    labels = document.get("Config", {}).get("Labels") or {}
    return labels.get(label_key) == label_value


# ---------------------------------------------------------------------------
# Wrapper normalization
# ---------------------------------------------------------------------------


def _extract_unset_vars(cmd: list[str]) -> list[str]:
    """Extract the ``-u NAME`` unset variables from a ``/usr/bin/env`` wrapper.

    The deployed entrypoint is ``/usr/bin/env`` with a ``Cmd`` like::

        ["-u", "VLLM_PREFIX_CACHE_RETENTION_INTERVAL",
         "/opt/venv/bin/vllm", "--model", ...]

    Returns the list of variable names that are unset before the vLLM
    binary, in their original order.  Raises ``CutoverError`` if the
    cmd does not start with at least one ``-u NAME`` pair, if any name
    is empty or duplicated, or if there is junk before the binary.
    """
    unset_vars: list[str] = []
    seen: set[str] = set()
    index = 0
    while index + 1 < len(cmd) and cmd[index] == "-u":
        name = cmd[index + 1]
        if not name:
            raise CutoverError(
                "wrapper has empty -u variable name"
            )
        if name in seen:
            raise CutoverError(
                f"wrapper has duplicate -u variable: {name!r}"
            )
        seen.add(name)
        unset_vars.append(name)
        index += 2
    if not unset_vars:
        raise CutoverError(
            "wrapper Cmd has no -u unset variable pairs"
        )
    return unset_vars


def _strip_wrapper(cmd: list[str]) -> list[str]:
    """Strip the ``-u NAME ... /opt/venv/bin/vllm`` prefix from a wrapper Cmd.

    Returns the effective vLLM argv (everything after the binary).
    Raises ``CutoverError`` if the binary is missing or there is junk
    between the last ``-u NAME`` pair and the binary.
    """
    index = 0
    has_unset = False
    while index + 1 < len(cmd) and cmd[index] == "-u":
        has_unset = True
        index += 2
    if not has_unset:
        raise CutoverError(
            "wrapper Cmd has no -u unset variable pairs"
        )
    if index >= len(cmd) or cmd[index] != _VLLM_BINARY:
        raise CutoverError(
            f"wrapper Cmd does not contain {_VLLM_BINARY!r} "
            f"after -u unset pairs (found {cmd[index]!r} at index {index})"
        )
    return list(cmd[index + 1:])


def _normalize_source(source: dict[str, Any], node: Node) -> dict[str, Any]:
    """Normalize a real deployed source inspect document.

    The real deployment uses ``Entrypoint=["/usr/bin/env"]`` and
    ``Path="/usr/bin/env"``, with a ``Cmd`` of
    ``["-u", NAME, ..., "/opt/venv/bin/vllm", "--model", ...]``.

    This function validates the wrapper shape, extracts the unset vars,
    and returns a normalized dict with:
      - ``Config.Entrypoint`` set to ``[_VLLM_BINARY]``
      - ``Config.Cmd`` set to the stripped vLLM argv
      - ``_unset_vars`` key containing the extracted unset variable names
      - ``_normalized`` key set to True

    Raises CutoverError if the entrypoint or wrapper shape is unexpected.
    For the deployed lane, both ``Entrypoint==[_WRAPPER_ENTRYPOINT]``
    and ``Path==_WRAPPER_ENTRYPOINT`` are required.  A synthetic
    ``/opt/venv/bin/vllm`` entrypoint is accepted only when explicitly
    marked with ``_normalized`` (pre-normalized documents from tests).
    """
    config = source.get("Config", {})
    entrypoint = config.get("Entrypoint")
    path = source.get("Path")

    if entrypoint == [_WRAPPER_ENTRYPOINT] and path == _WRAPPER_ENTRYPOINT:
        cmd = list(config.get("Cmd", []))
        unset_vars = _extract_unset_vars(cmd)
        vllm_argv = _strip_wrapper(cmd)
        if not vllm_argv:
            raise CutoverError(
                f"rank {node.rank} /usr/bin/env wrapper has no vLLM argv "
                f"after stripping -u vars and binary"
            )
        normalized = dict(source)
        normalized["Config"] = dict(config)
        normalized["Config"]["Entrypoint"] = [_VLLM_BINARY]
        normalized["Config"]["Cmd"] = vllm_argv
        normalized["_unset_vars"] = unset_vars
        normalized["_normalized"] = True
        return normalized

    if entrypoint == [_VLLM_BINARY] and source.get("_normalized"):
        # Already normalized (e.g. from a prior call or test fixture).
        normalized = dict(source)
        normalized["_unset_vars"] = list(source.get("_unset_vars", []))
        normalized["_normalized"] = True
        return normalized

    raise CutoverError(
        f"rank {node.rank} unexpected entrypoint: {entrypoint!r} "
        f"(path={path!r}). Expected Entrypoint=[{_WRAPPER_ENTRYPOINT!r}] "
        f"with Path={_WRAPPER_ENTRYPOINT!r}."
    )


# ---------------------------------------------------------------------------
# Container inspection helpers
# ---------------------------------------------------------------------------


def _env_map(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        name, _, value = item.partition("=")
        result[name] = value
    return result

def _env_list_sorted(mapping: dict[str, str]) -> list[str]:
    """Convert an env map to a sorted list of ``NAME=value`` strings."""
    return [f"{name}={value}" for name, value in sorted(mapping.items())]


def _rewrite_command(values: Iterable[str]) -> list[str]:
    command = list(values)
    for option, (before, after) in ARG_REWRITES.items():
        positions = [index for index, value in enumerate(command) if value == option]
        if len(positions) != 1:
            raise CutoverError(
                f"expected exactly one {option}, found {len(positions)}"
            )
        value_index = positions[0] + 1
        if value_index >= len(command) or command[value_index] != before:
            actual = command[value_index] if value_index < len(command) else None
            raise CutoverError(
                f"{option} precondition failed: expected {before!r}, got {actual!r}"
            )
        command[value_index] = after
    return command


def _rewrite_environment(values: Iterable[str]) -> list[str]:
    environment = _env_map(values)
    for name, (before, after) in ENV_REWRITES.items():
        actual = environment.get(name)
        if actual != before:
            raise CutoverError(
                f"{name} precondition failed: expected {before!r}, got {actual!r}"
            )
        environment[name] = after
    for name, expected in RUNTIME_UNSET_ENVIRONMENT.items():
        actual = environment.get(name)
        if actual != expected:
            raise CutoverError(
                f"{name} runtime-unset precondition failed: "
                f"expected {expected!r}, got {actual!r}"
            )
    return [f"{name}={value}" for name, value in sorted(environment.items())]


def _mount_signature(document: dict[str, Any]) -> list[tuple[str, str, bool]]:
    return sorted(
        (
            mount["Source"],
            mount["Destination"],
            not bool(mount["RW"]),
        )
        for mount in document["Mounts"]
        if mount["Type"] == "bind"
    )


def _find_writable_bind_mount(
    document: dict[str, Any], destination: str
) -> str | None:
    """Find the source path of a writable bind mount at ``destination``.

    Returns the host source path, or ``None`` if no writable bind mount
    exists at that destination.
    """
    for mount in document.get("Mounts", []):
        if (
            mount.get("Type") == "bind"
            and mount.get("Destination") == destination
            and bool(mount.get("RW"))
        ):
            return mount["Source"]
    return None


# ---------------------------------------------------------------------------
# Cache-enabled argv/env generation
# ---------------------------------------------------------------------------


def _apply_cache_config(
    source_cmd: list[str],
    source_env: list[str],
    target_checkpoint: str,
    draft_checkpoint: str,
    cache_root: str,
) -> tuple[list[str], list[str]]:
    """Apply DCP2 rewrites, then run the config generator for cache-enabled output."""
    dcp2_cmd = _rewrite_command(source_cmd)
    dcp2_env = _rewrite_environment(source_env)
    cmd, env = generate_sparkcache_argv(
        dcp2_cmd,
        dcp2_env,
        target_checkpoint=target_checkpoint,
        draft_checkpoint=draft_checkpoint,
        enabled=True,
        cache_root=cache_root,
    )
    verify_sparkcache_argv(cmd, env, expect_enabled=True)
    return cmd, env


# ---------------------------------------------------------------------------
# Container creation
# ---------------------------------------------------------------------------


def _create_argv(
    source: dict[str, Any],
    node: Node,
    cache_cmd: list[str],
    cache_env: list[str],
    prepare_token: str,
    unset_vars: list[str],
    connector_staging: str,
    cache_root_destination: str,
    connector_bundle_identity: str,
) -> list[str]:
    """Build the ``docker create`` argv for a cache-enabled target.

    ``source`` must be the **normalized** source document (via
    ``_normalize_source``).
    """
    config = source["Config"]
    host = source["HostConfig"]
    if source["State"]["Status"] not in {"running", "exited"}:
        raise CutoverError(
            f"rank {node.rank} source container has unusable state: "
            f"{source['State']['Status']!r}"
        )
    # After normalization, Entrypoint is always [_VLLM_BINARY].
    if config.get("Entrypoint") != [_VLLM_BINARY]:
        raise CutoverError(
            f"rank {node.rank} normalized source has unexpected entrypoint: "
            f"{config.get('Entrypoint')!r}"
        )
    if host.get("NetworkMode") != "host" or host.get("IpcMode") != "host":
        raise CutoverError(f"rank {node.rank} source is not host network/IPC")
    if host.get("Privileged") is not False:
        raise CutoverError(f"rank {node.rank} source unexpectedly privileged")

    argv = [
        "docker",
        "create",
        "--name",
        node.target_name,
        "--label", _label_for(node, prepare_token),
        "--label", f"{BUNDLE_IDENTITY_LABEL}={connector_bundle_identity}",
        "--network",
        "host",
        "--ipc",
        "host",
        "--gpus",
        "all",
        "--shm-size",
        str(host["ShmSize"]),
        "--ulimit",
        "memlock=-1:-1",
        "--cap-add",
        "IPC_LOCK",
        "--device",
        "/dev/infiniband:/dev/infiniband:rwm",
        "--security-opt",
        "label=disable",
        "--restart",
        "no",
        "--entrypoint",
        _WRAPPER_ENTRYPOINT,
    ]
    # Replicate all source bind mounts.
    for source_path, destination, read_only in _mount_signature(source):
        specification = f"type=bind,src={source_path},dst={destination}"
        if read_only:
            specification += ",readonly"
        argv.extend(("--mount", specification))
    argv.extend((
        "--mount",
        f"type=bind,src={connector_staging},dst={STAGING_DESTINATION},readonly",
    ))
    # Build a single deterministic PYTHONPATH: staging paths first, then
    # the original source PYTHONPATH (e.g. /opt/spark-vllm).  This
    # avoids duplicate --env PYTHONPATH=... options (ambiguous/last-wins).
    source_env_map = _env_map(source["Config"]["Env"])
    source_pythonpath = source_env_map.get("PYTHONPATH", "")
    merged_pythonpath = STAGING_PYTHONPATH
    if source_pythonpath:
        merged_pythonpath = f"{STAGING_PYTHONPATH}:{source_pythonpath}"
    # Emit cache_env, but replace PYTHONPATH with the merged value.
    # cache_env already incorporates _rewrite_environment rewrites, so its
    # values are the correct target values — no need to compare against
    # the original source env (which would false-positive on DCP2 rewrites).
    cache_env_map = _env_map(cache_env)
    cache_env_map["PYTHONPATH"] = merged_pythonpath
    for item in _env_list_sorted(cache_env_map):
        argv.extend(("--env", item))
    # Unset variables that the wrapper removes before vLLM import.
    for name in sorted(unset_vars):
        argv.extend(("-u", name))
    argv.append(_VLLM_BINARY)
    argv.extend(cache_cmd)
    return argv


def _selected_host_contract(document: dict[str, Any]) -> dict[str, Any]:
    host = document["HostConfig"]
    return {
        "network": host.get("NetworkMode"),
        "ipc": host.get("IpcMode"),
        "privileged": host.get("Privileged"),
        "shm": host.get("ShmSize"),
        "cap_add": sorted(host.get("CapAdd") or []),
        "devices": sorted(
            (
                item.get("PathOnHost"),
                item.get("PathInContainer"),
                item.get("CgroupPermissions"),
            )
            for item in (host.get("Devices") or [])
        ),
        "ulimits": sorted(
            (item.get("Name"), item.get("Soft"), item.get("Hard"))
            for item in (host.get("Ulimits") or [])
        ),
    }


def _effective_cmd_env(target: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Extract the effective vllm command/env from a target container.

    The target uses ``/usr/bin/env -u NAME … /opt/venv/bin/vllm ARGS``.
    Strip the ``-u`` unsets and the ``/opt/venv/bin/vllm`` binary to get
    the effective argv that the config generator would produce.

    Returns ``(vllm_argv, env_list, unset_vars)``.
    """
    raw_cmd = list(target["Config"]["Cmd"])
    unset_vars = _extract_unset_vars(raw_cmd)
    vllm_argv = _strip_wrapper(raw_cmd)
    return vllm_argv, list(target["Config"]["Env"]), unset_vars


def _verify_clone(
    source: dict[str, Any],
    target: dict[str, Any],
    node: Node,
    target_checkpoint: str,
    draft_checkpoint: str,
    cache_root: str,
    connector_staging: str,
) -> None:
    """Verify a target container is an exact clone of the source with cache config.

    ``source`` must be the **normalized** source document.
    ``connector_staging`` is the host-side staging directory path, used
    to verify the target's extra read-only mount.
    """
    if source["Image"] != target["Image"]:
        raise CutoverError(f"rank {node.rank} image ID changed")
    if target["Config"].get("Entrypoint") != [_WRAPPER_ENTRYPOINT]:
        raise CutoverError(
            f"rank {node.rank} target entrypoint is not [{_WRAPPER_ENTRYPOINT!r}]"
        )

    # The source has been normalized so its mount signature does NOT
    # include the staging mount.  The target has the staging mount as
    # an additional bind.  Verify source mounts are a subset of target.
    source_mounts = _mount_signature(source)
    target_mounts = _mount_signature(target)
    for sm in source_mounts:
        if sm not in target_mounts:
            raise CutoverError(
                f"rank {node.rank} source mount missing from target: {sm}"
            )
    # The target should have exactly one extra mount: the staging mount.
    extra_mounts = [m for m in target_mounts if m not in source_mounts]
    # Verify there is exactly one extra read-only mount at the staging
    # destination, and its host path matches the connector staging dir.
    staging_mounts = [
        m for m in extra_mounts
        if m[0] == connector_staging and m[1] == STAGING_DESTINATION and m[2]
    ]
    if len(staging_mounts) != 1:
        raise CutoverError(
            f"rank {node.rank} expected exactly one read-only staging mount "
            f"from {connector_staging!r} at {STAGING_DESTINATION}, "
            f"found {len(staging_mounts)}"
        )
    non_staging_extras = [m for m in extra_mounts if m[1] != STAGING_DESTINATION]
    if non_staging_extras:
        raise CutoverError(
            f"rank {node.rank} unexpected extra mounts on target: {non_staging_extras}"
        )

    if _selected_host_contract(source) != _selected_host_contract(target):
        raise CutoverError(f"rank {node.rank} host contract changed")

    # Extract the effective cmd/env from the target's wrapper.
    actual_cmd, actual_env, actual_unset_vars = _effective_cmd_env(target)

    # Verify the target's unset vars match the source's.
    expected_unset_vars = source.get("_unset_vars", [])
    if sorted(actual_unset_vars) != sorted(expected_unset_vars):
        raise CutoverError(
            f"rank {node.rank} unset vars drift: "
            f"expected {sorted(expected_unset_vars)}, got {sorted(actual_unset_vars)}"
        )

    # Verify PYTHONPATH is the merged value: staging paths + source PYTHONPATH.
    source_env_map = _env_map(source["Config"]["Env"])
    source_pythonpath = source_env_map.get("PYTHONPATH", "")
    expected_pythonpath = STAGING_PYTHONPATH
    if source_pythonpath:
        expected_pythonpath = f"{STAGING_PYTHONPATH}:{source_pythonpath}"
    target_env_map = _env_map(actual_env)
    if target_env_map.get("PYTHONPATH") != expected_pythonpath:
        raise CutoverError(
            f"rank {node.rank} PYTHONPATH missing or wrong: "
            f"expected {expected_pythonpath!r}, got "
            f"{target_env_map.get('PYTHONPATH')!r}"
        )

    # Check checkpoint identities FIRST so mismatch tests get the right
    # diagnostic, not a generic cmd-drift message.
    kvtc_positions = [
        i for i, v in enumerate(actual_cmd) if v == "--kv-transfer-config"
    ]
    if len(kvtc_positions) != 1:
        raise CutoverError(
            f"rank {node.rank} expected exactly one --kv-transfer-config"
        )
    config = json.loads(actual_cmd[kvtc_positions[0] + 1])
    extra = config["kv_connector_extra_config"]
    if extra.get("spark_cache_target_checkpoint_sha256") != target_checkpoint:
        raise CutoverError(
            f"rank {node.rank} target checkpoint identity mismatch"
        )
    if extra.get("spark_cache_draft_checkpoint_sha256") != draft_checkpoint:
        raise CutoverError(
            f"rank {node.rank} draft checkpoint identity mismatch"
        )
    if extra.get("spark_cache_root") != cache_root:
        raise CutoverError(
            f"rank {node.rank} cache root mismatch: "
            f"expected {cache_root!r}, got {extra.get('spark_cache_root')!r}"
        )

    # Now verify the full cmd/env exactly.
    # Use the normalized source Cmd (the stripped vLLM argv) for generation.
    expected_cmd, expected_env = _apply_cache_config(
        list(source["Config"]["Cmd"]),
        list(source["Config"]["Env"]),
        target_checkpoint,
        draft_checkpoint,
        cache_root,
    )
    if actual_cmd != expected_cmd:
        raise CutoverError(
            f"rank {node.rank} target cmd drift: "
            f"expected {expected_cmd}, got {actual_cmd}"
        )
    # Compare all env values exactly, including PYTHONPATH (merged value).
    actual_env_full = _env_map(actual_env)
    expected_env_full = _env_map(expected_env)
    expected_env_full["PYTHONPATH"] = expected_pythonpath
    if actual_env_full != expected_env_full:
        raise CutoverError(
            f"rank {node.rank} target env drift: "
            f"expected {expected_env_full}, got {actual_env_full}"
        )

    if target["State"]["Running"]:
        raise CutoverError(f"rank {node.rank} prepared target unexpectedly running")




def _source_summary(
    source: dict[str, Any],
    node: Node,
    target_checkpoint: str | None,
    draft_checkpoint: str | None,
) -> dict[str, Any]:
    command = source["Config"]["Cmd"]
    environment = _env_map(source["Config"]["Env"])

    def option(name: str) -> str:
        index = command.index(name)
        return command[index + 1]

    summary: dict[str, Any] = {
        "rank": node.rank,
        "source": node.source_name,
        "target": node.target_name,
        "baseline_dcp2": node.baseline_name,
        "source_running": source["State"]["Running"],
        "image_id": source["Image"],
        "entrypoint": source["Config"].get("Entrypoint"),
        "dcp": f"{option('--decode-context-parallel-size')} -> 2",
        "max_model_len": f"{option('--max-model-len')} -> 524288",
        "cache_enabled": True,
        "streaming_snapshots": False,
        "draft_policy": "separate",
        "connector": "SparkContextCacheConnector",
        "kv_load_failure_policy": "recompute",
        "kv_bytes_per_rank": option("--kv-cache-memory-bytes"),
        "max_num_seqs": option("--max-num-seqs"),
        "max_num_batched_tokens": option("--max-num-batched-tokens"),
        "environment_rewrites": {
            name: f"{environment[name]} -> {after}"
            for name, (_, after) in ENV_REWRITES.items()
        },
        "runtime_environment_unsets": {
            name: "removed before vLLM import"
            for name in RUNTIME_UNSET_ENVIRONMENT
        },
    }
    if target_checkpoint:
        summary["target_checkpoint_id"] = target_checkpoint[:12] + "..."
    if draft_checkpoint:
        summary["draft_checkpoint_id"] = draft_checkpoint[:12] + "..."
    return summary


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


def _parallel(nodes: list[Node], operation) -> list[Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        futures = {pool.submit(operation, node): node for node in nodes}
        results: dict[Node, Any] = {}
        for future in concurrent.futures.as_completed(futures):
            node = futures[future]
            try:
                results[node] = future.result()
            except Exception as exc:
                raise CutoverError(f"rank {node.rank}: {exc}") from exc
        return [results[node] for node in nodes]


def _load_sources(nodes: list[Node]) -> dict[int, dict[str, Any]]:
    pairs = _parallel(
        nodes,
        lambda node: (node.rank, _normalize_source(
            _inspect(node, node.source_name), node,
        )),
    )
    sources = dict(pairs)
    images = {document["Image"] for document in sources.values()}
    if len(images) != 1:
        raise CutoverError(f"source image IDs differ across ranks: {sorted(images)}")
    return sources


# ---------------------------------------------------------------------------
# Rollback helpers
# ---------------------------------------------------------------------------


def _stop_all_targets(
    nodes: list[Node], diagnostics: list[str] | None = None
) -> list[bool]:
    """Stop all cache target containers. Returns per-rank success flags.

    Success = docker stop returned 0, or target is proven absent.
    SSH failure or UNKNOWN existence is failure (fail-closed).
    """
    def stop_one(node: Node) -> bool:
        result = _remote_result(node, ("docker", "stop", "--time", "10", node.target_name))
        if result.returncode == 0:
            return True
        existence = _check_exists(node, node.target_name)
        if existence is Existence.PROVEN_ABSENT:
            return True  # Proven missing = safely stopped.
        # Unknown or proven-present: failure.
        if diagnostics is not None:
            diagnostics.append(
                f"rank {node.rank} target stop failed (rc={result.returncode}, "
                f"existence={existence.value}): {result.stderr.strip()}"
            )
        return False

    return _parallel(nodes, stop_one)


def _start_all_sources(
    nodes: list[Node], diagnostics: list[str] | None = None
) -> list[bool]:
    """Start all DCP4 source containers. Returns per-rank success flags.

    Success = docker start returned 0 AND inspect confirms Running=True.
    """
    def start_one(node: Node) -> bool:
        result = _remote_result(node, ("docker", "start", node.source_name))
        if result.returncode != 0:
            if diagnostics is not None:
                diagnostics.append(
                    f"rank {node.rank} source start failed (rc={result.returncode}): "
                    f"{result.stderr.strip()}"
                )
            return False
        try:
            doc = _inspect(node, node.source_name)
            return bool(doc["State"]["Running"])
        except CutoverError as exc:
            if diagnostics is not None:
                diagnostics.append(f"rank {node.rank} source start inspect failed: {exc}")
            return False

    return _parallel(nodes, start_one)


def _confirm_targets_stopped(nodes: list[Node]) -> list[bool]:
    """Read-only inspect confirmation: target is stopped or proven absent."""
    def check(node: Node) -> bool:
        existence = _check_exists(node, node.target_name)
        if existence is Existence.PROVEN_ABSENT:
            return True  # Proven missing = safely stopped.
        if existence is Existence.UNKNOWN:
            return False  # Can't confirm — fail-closed.
        try:
            doc = _inspect(node, node.target_name)
            return not doc["State"]["Running"]
        except CutoverError:
            return False

    return _parallel(nodes, check)


def _confirm_sources_running(nodes: list[Node]) -> list[bool]:
    """Read-only inspect confirmation: source exists and is running."""
    def check(node: Node) -> bool:
        existence = _check_exists(node, node.source_name)
        if existence is not Existence.PROVEN_PRESENT:
            return False
        try:
            doc = _inspect(node, node.source_name)
            return bool(doc["State"]["Running"])
        except CutoverError:
            return False

    return _parallel(nodes, check)


def _rollback_to_sources(
    nodes: list[Node], diagnostics: list[str]
) -> tuple[list[bool], list[bool]]:
    """Idempotent rollback: stop all cache targets, restart all DCP4 sources.

    Returns ``(targets_confirmed_stopped, sources_confirmed_running)`` —
    per-rank booleans verified by post-command inspect.  Command/transport
    failures are recorded in ``diagnostics`` but post-inspect is
    authoritative for the return value.

    Each phase (stop targets, start sources, confirm) is exception-isolated:
    a failure in one phase never prevents the next phase from running.
    A per-rank exception in stop is recorded as a diagnostic and the rank
    is marked as stop-failure, but source start is still attempted on all
    four ranks.
    """
    # Phase 1: Stop all targets.  Isolate exceptions per-rank so one
    # rank's timeout doesn't prevent the others from stopping.
    stop_results: list[bool] = []
    for node in nodes:
        try:
            result = _remote_result(
                node, ("docker", "stop", "--time", "10", node.target_name),
            )
            if result.returncode == 0:
                stop_results.append(True)
            else:
                existence = _check_exists(node, node.target_name)
                if existence is Existence.PROVEN_ABSENT:
                    stop_results.append(True)
                else:
                    diagnostics.append(
                        f"rank {node.rank} target stop failed "
                        f"(rc={result.returncode}, "
                        f"existence={existence.value}): "
                        f"{result.stderr.strip()}"
                    )
                    stop_results.append(False)
        except Exception as exc:
            diagnostics.append(f"rank {node.rank} target stop exception: {exc}")
            stop_results.append(False)

    # Phase 2: Start all sources — always attempted, regardless of
    # stop results.
    start_results: list[bool] = []
    for node in nodes:
        try:
            result = _remote_result(
                node, ("docker", "start", node.source_name),
            )
            if result.returncode != 0:
                diagnostics.append(
                    f"rank {node.rank} source start failed "
                    f"(rc={result.returncode}): {result.stderr.strip()}"
                )
                start_results.append(False)
                continue
            doc = _inspect(node, node.source_name)
            start_results.append(bool(doc["State"]["Running"]))
        except Exception as exc:
            diagnostics.append(f"rank {node.rank} source start exception: {exc}")
            start_results.append(False)

    # Phase 3: Confirm via read-only inspect.  Isolate exceptions.
    targets_stopped: list[bool] = []
    for node in nodes:
        try:
            existence = _check_exists(node, node.target_name)
            if existence is Existence.PROVEN_ABSENT:
                targets_stopped.append(True)
            elif existence is Existence.UNKNOWN:
                targets_stopped.append(False)
            else:
                doc = _inspect(node, node.target_name)
                targets_stopped.append(not doc["State"]["Running"])
        except Exception as exc:
            diagnostics.append(
                f"rank {node.rank} target confirm exception: {exc}"
            )
            targets_stopped.append(False)

    sources_running: list[bool] = []
    for node in nodes:
        try:
            existence = _check_exists(node, node.source_name)
            if existence is not Existence.PROVEN_PRESENT:
                sources_running.append(False)
                continue
            doc = _inspect(node, node.source_name)
            sources_running.append(bool(doc["State"]["Running"]))
        except Exception as exc:
            diagnostics.append(
                f"rank {node.rank} source confirm exception: {exc}"
            )
            sources_running.append(False)

    for i, node in enumerate(nodes):
        if not targets_stopped[i]:
            diagnostics.append(
                f"ROLLBACK WARNING: rank {node.rank} cache target "
                f"not confirmed stopped"
            )
        if not sources_running[i]:
            diagnostics.append(
                f"ROLLBACK WARNING: rank {node.rank} DCP4 source "
                f"not confirmed running"
            )
    return targets_stopped, sources_running


# ---------------------------------------------------------------------------
# Health polling
# ---------------------------------------------------------------------------


def _check_container_health(node: Node) -> dict[str, Any]:
    """Check a single target container's state and rank-0 health endpoint.

    Returns dict with keys: running, oom_killed, exit_code, health_ok, status.
    """
    doc = _inspect(node, node.target_name)
    state = doc["State"]
    result: dict[str, Any] = {
        "running": state["Running"],
        "oom_killed": state["OOMKilled"],
        "exit_code": state["ExitCode"],
        "health_ok": False,
        "status": state.get("Status", ""),
    }
    if not state["Running"]:
        return result
    if state["OOMKilled"]:
        return result
    if state["ExitCode"] != 0:
        return result
    if node.rank != 0:
        result["health_ok"] = True
        return result
    health_result = subprocess.run(
        [
            *_ssh_base(), node.ssh_target,
            "curl", "-sf", "--max-time", "5",
            "http://127.0.0.1:8000/health",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=SSH_MAX_TIMEOUT_S,
    )
    result["health_ok"] = health_result.returncode == 0
    return result


# ---------------------------------------------------------------------------
# Preflight helpers
# ---------------------------------------------------------------------------


def _verify_host_staging(node: Node, connector_staging: str) -> dict[str, Any]:
    """Lightweight diagnostic: check if the staging directory exists on host.

    This is NOT the authoritative verifier — ``_verify_bundle_identity``
    performs the complete fail-closed inventory.  This quick check only
    confirms the directory exists and is non-empty, for pre-flight
    diagnostics that tell the operator *what* is missing before the
    full identity check runs.
    """
    result = _remote_result(node, ("ls", connector_staging))
    exists = result.returncode == 0 and bool(result.stdout.strip())
    return {
        "exists": exists,
        "stderr": result.stderr.strip() if result.stderr else "",
    }


def _verify_bundle_identity(
    node: Node, connector_staging: str, expected_identity: str
) -> bool:
    """Read-only fail-closed verifier: inventory the remote staging tree,
    compute the domain-separated identity, and compare to expected.

    Sends an inline Python script to the remote host via
    ``python3 -c`` through ``_remote_result`` (no shell).  The script
    applies the same semantics as ``inventory_staging_root``: lstat
    (no symlink dereference), reject non-regular files, reject missing
    and extra files, hash in 1 MiB chunks, compare stable stat metadata
    before/after every read, and compute the same domain-separated
    identity as the offline builder.

    Returns True if the identity matches, False on any failure
    (missing files, extra files, symlinks, non-regular files,
    changed-during-read, parse error, identity mismatch, SSH failure).
    """
    required_json = json.dumps(sorted(_BUNDLE_REQUIRED_FILES))
    result = _remote_result(
        node,
        ("python3", "-c", _REMOTE_VERIFIER_SCRIPT, connector_staging, required_json, _BUNDLE_DOMAIN),
    )
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return False
    if "error" in payload:
        return False
    return payload.get("identity") == expected_identity


def _check_cache_mount(
    source: dict[str, Any], node: Node, cache_root: str
) -> str | None:
    """Verify the cache_root is a writable bind mount on the source.

    Returns the host source path of the writable bind mount at
    ``cache_root``, or ``None`` if not found.
    """
    host_path = _find_writable_bind_mount(source, cache_root)
    return host_path


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def command_plan(
    nodes: list[Node],
    target_checkpoint: str | None,
    draft_checkpoint: str | None,
    cache_root: str | None = None,
    connector_staging: str | None = None,
    connector_bundle_identity: str | None = None,
) -> int:
    """Read-only preflight: validate all preconditions and report blockers.

    Returns 0 if ready, 1 if blockers exist.  Never mutates.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    summaries: list[dict[str, Any]] = []

    # Load and normalize sources.
    try:
        sources = _load_sources(nodes)
    except CutoverError as exc:
        blockers.append(f"source inspection failed: {exc}")
        print(json.dumps({"status": "BLOCKED", "blockers": blockers}, indent=2))
        return 1

    # Validate source state/image consistency.
    for node in nodes:
        src = sources[node.rank]
        if src["State"]["Running"]:
            warnings.append(f"rank {node.rank} source is running (expected for active service)")
        else:
            blockers.append(f"rank {node.rank} source is not running — cannot inspect live state")

    # Validate cache root: must be a writable bind mount on every rank.
    resolved_cache_root: str | None = None
    for node in nodes:
        src = sources[node.rank]
        # Try the provided cache_root, or auto-detect from source mounts.
        if cache_root:
            cr = cache_root
        else:
            # Auto-detect: look for known cache mount destinations.
            for candidate in ("/var/tmp/sparkring-public-validation/context-cache",):
                if _find_writable_bind_mount(src, candidate):
                    cr = candidate
                    break
            else:
                blockers.append(
                    f"rank {node.rank} no writable cache bind mount found; "
                    f"use --cache-root to specify the destination"
                )
                continue
        host_path = _check_cache_mount(src, node, cr)
        if host_path is None:
            blockers.append(
                f"rank {node.rank} cache root {cr!r} is not a writable bind mount "
                f"on source container"
            )
        else:
            if resolved_cache_root is None:
                resolved_cache_root = cr
            elif resolved_cache_root != cr:
                blockers.append(
                    f"rank {node.rank} cache root {cr!r} differs from "
                    f"rank 0 {resolved_cache_root!r}"
                )

    # Validate connector staging (read-only host check).
    if connector_staging:
        for node in nodes:
            staging_info = _verify_host_staging(node, connector_staging)
            if not staging_info["exists"]:
                blockers.append(
                    f"rank {node.rank} connector staging directory "
                    f"{connector_staging!r} missing required files on host"
                )
        # Verify connector bundle identity on every host if provided.
        if connector_bundle_identity:
            for node in nodes:
                if not _verify_bundle_identity(node, connector_staging, connector_bundle_identity):
                    blockers.append(
                        f"rank {node.rank} connector bundle identity mismatch "
                        f"or verification failed on host"
                    )
        else:
            blockers.append(
                "B6: connector bundle identity not provided — "
                "use --connector-bundle-identity to specify the expected "
                "64-hex SHA-256 from connector_bundle_manifest.py"
            )
    else:
        blockers.append(
            "connector staging root not provided; use --connector-staging to specify "
            "the host directory containing spark_context_cache_connector.py, "
            "spark_context_cache_codec.py, spark_context_cache_store.py, and "
            "sparkcache/streaming/"
        )

    # Validate target state.  Absent targets are prepare-ready.
    # Present targets must be exact clones (verify_clone) to be
    # cutover-ready; running targets or clone mismatches are blockers.
    for node in nodes:
        existence = _check_exists(node, node.target_name)
        if existence is Existence.PROVEN_ABSENT:
            warnings.append(
                f"rank {node.rank} target absent — prepare will create it"
            )
        elif existence is Existence.UNKNOWN:
            blockers.append(
                f"rank {node.rank} cannot determine if target exists (SSH/daemon failure)"
            )
        elif existence is Existence.PROVEN_PRESENT:
            try:
                tgt = _inspect(node, node.target_name)
                if tgt["State"]["Running"]:
                    blockers.append(
                        f"rank {node.rank} target {node.target_name!r} "
                        f"already exists and is running — cannot cutover"
                    )
                else:
                    # Stopped target: verify it's an exact clone.
                    if target_checkpoint and draft_checkpoint and resolved_cache_root and connector_staging:
                        try:
                            _verify_clone(
                                sources[node.rank], tgt, node,
                                target_checkpoint, draft_checkpoint,
                                resolved_cache_root, connector_staging,
                            )
                            warnings.append(
                                f"rank {node.rank} target {node.target_name!r} "
                                f"exists and is an exact clone — cutover-ready"
                            )
                        except CutoverError as exc:
                            blockers.append(
                                f"rank {node.rank} target {node.target_name!r} "
                                f"exists but is not an exact clone: {exc}"
                            )
                    else:
                        blockers.append(
                            f"rank {node.rank} target {node.target_name!r} "
                            f"exists but checkpoints/staging not provided "
                            f"for clone verification"
                        )
            except CutoverError as exc:
                blockers.append(
                    f"rank {node.rank} target exists but inspect failed: {exc}"
                )

    # Generate/verify target config (if checkpoints provided).
    if target_checkpoint and draft_checkpoint and resolved_cache_root:
        for node in nodes:
            src = sources[node.rank]
            try:
                expected_cmd, expected_env = _apply_cache_config(
                    list(src["Config"]["Cmd"]),
                    list(src["Config"]["Env"]),
                    target_checkpoint,
                    draft_checkpoint,
                    resolved_cache_root,
                )
                verify_sparkcache_argv(expected_cmd, expected_env, expect_enabled=True)
            except CutoverError as exc:
                blockers.append(f"rank {node.rank} config generation failed: {exc}")
    elif target_checkpoint or draft_checkpoint:
        blockers.append("both --target-checkpoint and --draft-checkpoint are required for config validation")

    # Build summaries.
    for node in nodes:
        src = sources[node.rank]
        try:
            summary = _source_summary(src, node, target_checkpoint, draft_checkpoint)
            summary["cache_root"] = resolved_cache_root
            summary["connector_staging"] = connector_staging
            summary["unset_vars"] = src.get("_unset_vars", [])
            summaries.append(summary)
        except (CutoverError, ValueError) as exc:
            blockers.append(f"rank {node.rank} summary failed: {exc}")

    # Report B2 blocker explicitly.
    if not target_checkpoint or not draft_checkpoint:
        blockers.append(
            "B2: checkpoint identity not attested — "
            "manifest generator not yet run against deployed mounts"
        )

    output: dict[str, Any] = {
        "ranks": summaries,
        "cache_root": resolved_cache_root,
        "connector_staging": connector_staging,
    }
    if blockers:
        output["status"] = "BLOCKED"
        output["blockers"] = blockers
        if warnings:
            output["warnings"] = warnings
        print(json.dumps(output, indent=2))
        return 1
    output["status"] = "READY"
    if warnings:
        output["warnings"] = warnings
    print(json.dumps(output, indent=2))
    return 0


def command_prepare(
    nodes: list[Node],
    target_checkpoint: str,
    draft_checkpoint: str,
    cache_root: str,
    connector_staging: str,
    connector_bundle_identity: str,
) -> None:
    sources = _load_sources(nodes)

    for node in nodes:
        existence = _check_exists(node, node.target_name)
        if existence is Existence.PROVEN_PRESENT:
            raise CutoverError(
                f"rank {node.rank} target {node.target_name!r} already exists"
            )
        if existence is Existence.UNKNOWN:
            raise CutoverError(
                f"rank {node.rank} cannot determine if target exists "
                f"(SSH/daemon failure) — refusing to proceed"
            )

    # Verify cache root is a writable bind mount on every rank.
    for node in nodes:
        src = sources[node.rank]
        if _check_cache_mount(src, node, cache_root) is None:
            raise CutoverError(
                f"rank {node.rank} cache root {cache_root!r} is not a "
                f"writable bind mount on source — refusing to proceed"
            )

    # Verify connector staging exists on every host.
    for node in nodes:
        staging_info = _verify_host_staging(node, connector_staging)
        if not staging_info["exists"]:
            raise CutoverError(
                f"rank {node.rank} connector staging {connector_staging!r} "
                f"missing required files on host — refusing to proceed"
            )

    # Verify connector bundle identity on every host.
    for node in nodes:
        if not _verify_bundle_identity(node, connector_staging, connector_bundle_identity):
            raise CutoverError(
                f"rank {node.rank} connector bundle identity mismatch "
                f"or verification failed — refusing to proceed"
            )

    prepare_token = uuid.uuid4().hex
    created: list[Node] = []
    created_lock = threading.Lock()

    def create(node: Node) -> str:
        cache_cmd, cache_env = _apply_cache_config(
            list(sources[node.rank]["Config"]["Cmd"]),
            list(sources[node.rank]["Config"]["Env"]),
            target_checkpoint,
            draft_checkpoint,
            cache_root,
        )
        unset_vars = sources[node.rank].get("_unset_vars", [])
        output = _remote(
            node, _create_argv(
                sources[node.rank], node, cache_cmd, cache_env, prepare_token,
                unset_vars, connector_staging, cache_root,
                connector_bundle_identity,
            ),
        )
        container_id = output.strip()
        with created_lock:
            created.append(node)
        return container_id

    try:
        container_ids = _parallel(nodes, create)
        for node, container_id in zip(nodes, container_ids):
            target = _inspect(node, node.target_name)
            # Verify the prepare label is present on the created container.
            if not _has_label(target, PREPARE_LABEL, prepare_token):
                raise CutoverError(
                    f"rank {node.rank} target missing prepare label"
                )
            _verify_clone(
                sources[node.rank], target, node,
                target_checkpoint, draft_checkpoint, cache_root, connector_staging,
            )
            print(f"rank {node.rank}: prepared and exact-diff verified {container_id[:12]}")
    except Exception:
        with created_lock:
            to_remove = list(created)
        # Also check for targets bearing our token that were created but
        # whose response was lost (not recorded in `created`).
        for node in nodes:
            if node in to_remove:
                continue
            existence = _check_exists(node, node.target_name)
            if existence is Existence.PROVEN_PRESENT:
                try:
                    doc = _inspect(node, node.target_name)
                    if _has_label(doc, PREPARE_LABEL, prepare_token):
                        to_remove.append(node)
                except CutoverError:
                    pass  # Can't inspect → skip, don't risk deleting foreign container.
            # UNKNOWN: don't remove — fail-closed.
        for node in to_remove:
            _remote(
                node, ("docker", "rm", "--force", node.target_name),
                check=False,
            )
        raise


def command_cutover(
    nodes: list[Node],
    target_checkpoint: str,
    draft_checkpoint: str,
    cache_root: str,
    connector_staging: str,
    connector_bundle_identity: str,
    health_timeout: int = HEALTH_TIMEOUT_S,
) -> None:
    if health_timeout <= 0:
        raise CutoverError(f"health_timeout must be positive, got {health_timeout}")
    sources = _load_sources(nodes)
    for node in nodes:
        existence = _check_exists(node, node.target_name)
        if existence is not Existence.PROVEN_PRESENT:
            raise CutoverError(
                f"rank {node.rank} prepared target does not exist or existence unknown"
            )

    for node in nodes:
        target = _inspect(node, node.target_name)
        if target["State"]["Running"]:
            raise CutoverError(
                f"rank {node.rank} target already running before cutover"
            )
        _verify_clone(
            sources[node.rank], target, node,
            target_checkpoint, draft_checkpoint, cache_root, connector_staging,
        )

    # Re-verify connector bundle identity on every host before stopping sources.
    # This catches drift between prepare and cutover.
    for node in nodes:
        if not _verify_bundle_identity(node, connector_staging, connector_bundle_identity):
            raise CutoverError(
                f"rank {node.rank} connector bundle identity changed "
                f"since prepare — refusing to proceed with cutover"
            )

    diagnostics: list[str] = []
    transition_started = False
    rollback_complete = False

    try:
        # --- Stop sources ---
        transition_started = True
        _parallel(
            nodes,
            lambda node: _remote(
                node, ("docker", "stop", "--time", "30", node.source_name),
            ),
        )

        # --- Start targets ---
        _parallel(
            nodes,
            lambda node: _remote(
                node, ("docker", "start", node.target_name),
            ),
        )

        # --- Health polling ---
        deadline = time.monotonic() + health_timeout
        ready = False
        last_health_results: list[dict[str, Any]] | None = None
        while time.monotonic() < deadline:
            time.sleep(HEALTH_INTERVAL_S)
            health_results = _parallel(nodes, _check_container_health)
            last_health_results = health_results
            all_running = all(h["running"] for h in health_results)
            any_oom = any(h["oom_killed"] for h in health_results)
            any_exited = any(not h["running"] for h in health_results)
            any_bad_exit = any(h["exit_code"] != 0 for h in health_results)
            rank0_health = next(
                (h["health_ok"] for h, n in zip(health_results, nodes) if n.rank == 0),
                False,
            )
            if all_running and not any_oom and not any_exited and not any_bad_exit and rank0_health:
                ready = True
                break
            if any_oom:
                diagnostics.append("OOM detected on one or more targets")
                break
            if any_exited:
                exit_codes = [h["exit_code"] for h in health_results]
                diagnostics.append(
                    f"immediate exit detected: exit codes = {exit_codes}"
                )
                break

        if not ready:
            running_info = (
                [h["running"] for h in last_health_results]
                if last_health_results is not None
                else "unknown"
            )
            diagnostics.append(
                f"readiness not achieved within {health_timeout}s "
                f"(running={running_info})"
            )
            raise CutoverError(
                f"cutover failed readiness check. "
                f"Diagnostics: {'; '.join(diagnostics)}"
            )

        print(
            "DCP2 SparkCache targets started on all four ranks; "
            "DCP4 sources remain stopped for rollback"
        )
        rollback_complete = True

    except Exception as exc:
        # Preserve the original error.
        original_error = exc
        if transition_started and not rollback_complete:
            # Attempt rollback for ANY exception once transition has started.
            # This is the single rollback path — source-stop failure,
            # target-start failure, health-poll exception, and
            # unexpected exceptions all flow through here.
            try:
                rb_targets_stopped, rb_sources_running = _rollback_to_sources(
                    nodes, diagnostics
                )
                incomplete_ranks = [
                    nodes[i].rank for i in range(len(nodes))
                    if not rb_targets_stopped[i] or not rb_sources_running[i]
                ]
                if incomplete_ranks:
                    diagnostics.append(
                        f"ROLLBACK INCOMPLETE: ranks {sorted(incomplete_ranks)} "
                        f"not confirmed restored"
                    )
                    raise CutoverError(
                        f"{original_error}. ROLLBACK INCOMPLETE: ranks "
                        f"{sorted(incomplete_ranks)} not confirmed restored. "
                        f"Diagnostics: {'; '.join(diagnostics)}"
                    ) from original_error
                else:
                    raise CutoverError(
                        f"{original_error}. ROLLBACK COMPLETE. "
                        f"Diagnostics: {'; '.join(diagnostics)}"
                    ) from original_error
            except CutoverError as rb_ce:
                # A CutoverError from rollback itself must NOT erase
                # the original error.  Combine both.
                diagnostics.append(f"rollback CutoverError: {rb_ce}")
                raise CutoverError(
                    f"{original_error}. ROLLBACK INCOMPLETE (rollback "
                    f"error: {rb_ce}). Diagnostics: "
                    f"{'; '.join(diagnostics)}"
                ) from original_error
            except Exception as rb_exc:
                # Rollback itself failed — combine both errors.
                diagnostics.append(f"rollback attempt failed: {rb_exc}")
                raise CutoverError(
                    f"{original_error}. ROLLBACK INCOMPLETE (rollback error: "
                    f"{rb_exc}). Diagnostics: {'; '.join(diagnostics)}"
                ) from original_error
        else:
            # Transition not started — no rollback needed.
            raise

    if diagnostics:
        for d in diagnostics:
            print(f"WARNING: {d}", file=sys.stderr)


def command_rollback(nodes: list[Node]) -> None:
    diagnostics: list[str] = []
    targets_stopped, sources_running = _rollback_to_sources(nodes, diagnostics)
    failed_targets = [
        nodes[i].rank for i, ok in enumerate(targets_stopped) if not ok
    ]
    failed_sources = [
        nodes[i].rank for i, ok in enumerate(sources_running) if not ok
    ]
    if failed_targets or failed_sources:
        for d in diagnostics:
            print(f"WARNING: {d}", file=sys.stderr)
        parts = []
        if failed_targets:
            parts.append(f"targets not stopped: {sorted(failed_targets)}")
        if failed_sources:
            parts.append(f"sources not running: {sorted(failed_sources)}")
        raise CutoverError(
            f"ROLLBACK INCOMPLETE: {'; '.join(parts)}. "
            f"Manual intervention required."
        )
    print("Rollback complete: DCP4 sources restarted, cache targets stopped")
    for d in diagnostics:
        print(f"WARNING: {d}", file=sys.stderr)


def command_status(nodes: list[Node]) -> None:
    rows = []
    for node in nodes:
        row: dict[str, Any] = {"rank": node.rank}
        for role, name in (
            ("source", node.source_name),
            ("target", node.target_name),
            ("baseline_dcp2", node.baseline_name),
        ):
            existence = _check_exists(node, name)
            if existence is Existence.PROVEN_ABSENT:
                row[role] = {"exists": False}
                continue
            if existence is Existence.UNKNOWN:
                row[role] = {"exists": "unknown"}
                continue
            document = _inspect(node, name)
            row[role] = {
                "exists": True,
                "running": document["State"]["Running"],
                "status": document["State"]["Status"],
                "oom_killed": document["State"]["OOMKilled"],
                "exit_code": document["State"]["ExitCode"],
                "image_id": document["Image"],
            }
        rows.append(row)
    print(json.dumps(rows, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_node(value: str) -> Node:
    rank_text, separator, target = value.partition("=")
    if not separator or not target:
        raise argparse.ArgumentTypeError("node must be RANK=SSH_TARGET")
    try:
        rank = int(rank_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("node rank must be an integer") from exc
    if rank not in range(4):
        raise argparse.ArgumentTypeError("node rank must be 0, 1, 2, or 3")
    return Node(rank=rank, ssh_target=target)


def _validate_hex64(value: str) -> str:
    if len(value) != 64 or not all(c in "0123456789abcdef" for c in value):
        raise argparse.ArgumentTypeError(
            "must be 64-character lowercase hex SHA-256"
        )
    return value


def _validate_path(value: str) -> str:
    """Validate an operator-supplied path for cache-root or connector-staging.

    Must be an absolute POSIX path (starts with /), normalized (no . or ..),
    not root /, and contain no NUL or newline characters.
    """
    if not value.startswith("/"):
        raise argparse.ArgumentTypeError(
            f"path must be absolute (start with /): {value!r}"
        )
    if "\x00" in value or "\n" in value:
        raise argparse.ArgumentTypeError(
            "path must not contain NUL or newline characters"
        )
    # Normalize and check for . or .. components.
    parts = value.split("/")
    if any(part in (".", "..") for part in parts):
        raise argparse.ArgumentTypeError(
            f"path must not contain . or .. components: {value!r}"
        )
    if value == "/":
        raise argparse.ArgumentTypeError("path must not be root /")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("plan", "prepare", "cutover", "rollback", "status"),
    )
    parser.add_argument("--node", action="append", type=_parse_node, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation")
    parser.add_argument(
        "--target-checkpoint",
        type=_validate_hex64,
        help="64-char hex SHA-256 target checkpoint identity (required for prepare/cutover)",
    )
    parser.add_argument(
        "--draft-checkpoint",
        type=_validate_hex64,
        help="64-char hex SHA-256 draft checkpoint identity (required for prepare/cutover)",
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=HEALTH_TIMEOUT_S,
        help=f"Health readiness timeout in seconds (default {HEALTH_TIMEOUT_S})",
    )
    parser.add_argument(
        "--cache-root",
        type=_validate_path,
        default=None,
        help="Absolute destination path of the writable cache bind mount "
             "(required for prepare/cutover; optional for plan)",
    )
    parser.add_argument(
        "--connector-staging",
        type=_validate_path,
        default=None,
        help="Host directory containing spark_context_cache_connector.py, "
             "spark_context_cache_codec.py, spark_context_cache_store.py, and "
             "sparkcache/streaming/ (required for prepare/cutover; optional for plan)",
    )
    parser.add_argument(
        "--connector-bundle-identity",
        type=_validate_hex64,
        default=None,
        help="64-char hex SHA-256 connector bundle identity (required for "
             "prepare/cutover; optional for plan). Generated offline by "
             "connector_bundle_manifest.py.",
    )
    args = parser.parse_args(argv)
    nodes = sorted(args.node, key=lambda item: item.rank)
    if [node.rank for node in nodes] != [0, 1, 2, 3]:
        raise CutoverError("exactly one --node is required for each rank 0,1,2,3")

    if args.operation in ("prepare", "cutover"):
        if not args.target_checkpoint or not args.draft_checkpoint:
            raise CutoverError(
                f"{args.operation} requires --target-checkpoint and"
                " --draft-checkpoint (64-char hex SHA-256 from the manifest"
                " generator). B2 is not resolved until real identities exist."
            )
        if not args.cache_root:
            raise CutoverError(
                f"{args.operation} requires --cache-root (absolute destination "
                f"path of the writable cache bind mount)"
            )
        if not args.connector_staging:
            raise CutoverError(
                f"{args.operation} requires --connector-staging (host directory "
                f"containing the connector source modules)"
            )
        if not args.connector_bundle_identity:
            raise CutoverError(
                f"{args.operation} requires --connector-bundle-identity "
                f"(64-char hex SHA-256 from connector_bundle_manifest.py)"
            )

    confirmations = {
        "prepare": PREPARE_CONFIRMATION,
        "cutover": CUTOVER_CONFIRMATION,
        "rollback": ROLLBACK_CONFIRMATION,
    }
    expected = confirmations.get(args.operation)
    if expected is not None and (
        not args.execute or args.confirmation != expected
    ):
        raise CutoverError(
            f"{args.operation} requires --execute --confirmation {expected}"
        )

    if args.operation == "plan":
        return command_plan(
            nodes, args.target_checkpoint, args.draft_checkpoint,
            args.cache_root, args.connector_staging,
            args.connector_bundle_identity,
        )
    elif args.operation == "prepare":
        command_prepare(
            nodes, args.target_checkpoint, args.draft_checkpoint,
            args.cache_root, args.connector_staging,
            args.connector_bundle_identity,
        )
    elif args.operation == "cutover":
        command_cutover(
            nodes,
            target_checkpoint=args.target_checkpoint,
            draft_checkpoint=args.draft_checkpoint,
            cache_root=args.cache_root,
            connector_staging=args.connector_staging,
            connector_bundle_identity=args.connector_bundle_identity,
            health_timeout=args.health_timeout,
        )
    elif args.operation == "rollback":
        command_rollback(nodes)
    elif args.operation == "status":
        command_status(nodes)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CutoverError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
