#!/usr/bin/env python3
"""SparkRing public-functional acceptance gate.

Deterministic, ordered, fail-closed. Runs the eight ordered stages defined in
``docs/PUBLIC_FUNCTIONAL_TARGET.md`` against the matrix declared by the gate
profile, aborts at the first functional failure, and emits a single evidence
bundle with a top-level ``result.json``. The accepted NF3 matrix remains the
default; a profile may declare a different, immutable candidate matrix such as
the default EXL3+LMCache configuration.

The gate proves things by delegating to the tools that already exist in this
repository -- ``runtime/verify-runtime.py`` for runtime and artifact
attestation, ``scripts/preflight.py`` for the read-only cluster preflight,
``spark_transport/scripts/qualify_direct_cable.py`` for per-edge fabric
qualification, the repo's model-down collective probe entrypoint, and the
site's own launcher for start/stop. It re-implements none of them.

Two verdicts are computed and reported separately and are never merged:

  functional_verdict  PASS | FAIL | BASELINE-RECORDED | NOT-RUN
  performance_verdict IN-BAND | OUT-OF-BAND | BASELINE-RECORDED | NOT-MEASURED

Exit codes:

  0  functional PASS, performance IN-BAND or NOT-MEASURED
  2  functional FAIL
  3  configuration or plan error (nothing was executed)
  4  functional BASELINE-RECORDED (an expected-output file was emitted; this
     is NOT a pass)
  5  functional PASS but performance OUT-OF-BAND or BASELINE-RECORDED

``--dry-run`` is the DEFAULT and the safe path: it validates the configuration,
prints the ordered plan, and touches nothing -- no bundle directory, no files,
no connections, nothing started or stopped. ``--execute`` requires an explicit
confirmation token.

NEVER point this gate at a cluster that is serving production traffic: stages
3 and 8 start and stop the serving stack. After a stage-3 start attempt, stage
8 is still attempted when any intervening stage fails.

Configuration comes from two files:

1. **The site config** (``--site``), owned and validated by
   ``scripts/sparkring_site.py``. It describes the cluster: ranks, ssh targets,
   the four ring edges and their /24s, management addresses, the pinned runtime
   image and model identity, the serving shape, paths and artifact hashes. That
   module's schema is authoritative; this gate consumes its normalised form and
   adds only matrix checks of its own.
2. **The gate config** (``--gate-config``), owned by this script. It describes
   how to *run the gate* at your site -- things the site schema deliberately
   does not carry. Defaults are shown; entries with an empty argv are required
   before the gate can plan or run::

    {
      "schema": "sparkring-acceptance-gate-config/v1",
      "ssh":     {"command": ["ssh", "-o", "BatchMode=yes"]},
      "runtime": {
        "lock_path": "recipes/glm52-nf3-hybrid.json",
        "verify_script": "/opt/sparkring/verify-runtime.py",
        "manifest_path": "/opt/sparkring/runtime-manifest.json",
        "expect_runtime_id": null,
        "exec_prefix": ["docker", "exec", "<your-container>"],
        "model_identity": {
          "config_path": "/hybridmodel/config.json",
          "repository_path": "/hybridmodel/.sparkring-model-repository",
          "revision_path": "/hybridmodel/.sparkring-model-revision",
          "in_container": true
        }
      },
      "fabric": {
        "qualifier": "spark_transport/scripts/qualify_direct_cable.py",
        "probe_binary": "/tmp/spark_transport_probe",
        "iterations": 10000,
        "model_down_probe": {"command": [], "per_rank": true}
      },
      "launch": {
        "start_command": [], "stop_command": [],
        "rollback_verify_command": [], "rank_status_command": [],
        "ready_timeout_seconds": 3600, "stop_timeout_seconds": 900
      },
      "api": {"scheme": "http", "rank_bases": {}},
      "acceptance": {"expected_generation_path": null, "prompt": null,
                     "seed": 20260729, "max_tokens": 64},
      "performance": {"cells": [{"concurrency": 1, "max_tokens": 128},
                                {"concurrency": 8, "max_tokens": 128}],
                      "band": null},
      "preflight": {"result_path": null}
    }
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import ipaddress
import json
import math
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

RESULT_SCHEMA = "sparkring-acceptance-result/v1"
PLAN_SCHEMA = "sparkring-acceptance-plan/v1"
EXPECTED_SCHEMA = "sparkring-acceptance-expected-generation/v1"
GATE_CONFIG_SCHEMA = "sparkring-acceptance-gate-config/v1"
SITE_SCHEMA = "sparkring-site/v1"
PREFLIGHT_SCHEMA = "sparkring-preflight/v1"
GATE_VERSION = "2"

CONFIRM_TOKEN = "RUN-PUBLIC-ACCEPTANCE-GATE"
TARGET_DOC = "docs/PUBLIC_FUNCTIONAL_TARGET.md"

# ---------------------------------------------------------------------------
# The matrix. These constants ARE the supported configuration
# (docs/PUBLIC_FUNCTIONAL_TARGET.md section 2). Changing one changes what
# "public-functional acceptance" means and invalidates prior results.
# ---------------------------------------------------------------------------
REQUIRED_RANKS = 4
REQUIRED_EDGES = 4
REQUIRED_MTU = 9000
REQUIRED_LINK_SPEED_MBPS = 200000
REQUIRED_TP = 4
REQUIRED_DCP = 4
REQUIRED_MTP_MODE = "adaptive"
REQUIRED_MTP_TOKENS = 4
REQUIRED_MAX_MODEL_LEN = 262144
REQUIRED_KV_BYTES_PER_RANK = 7_000_000_000
REQUIRED_MAX_NUM_SEQS = 8
REQUIRED_CONCURRENCIES = (1, 8)

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_BASELINE_RECORDED = "BASELINE-RECORDED"
STATUS_PERFORMANCE_OUT_OF_BAND = "PERFORMANCE-OUT-OF-BAND"
STATUS_SKIPPED = "SKIPPED"

FUNCTIONAL_PASS = "PASS"
FUNCTIONAL_FAIL = "FAIL"
FUNCTIONAL_BASELINE_RECORDED = "BASELINE-RECORDED"
FUNCTIONAL_NOT_RUN = "NOT-RUN"

PERFORMANCE_IN_BAND = "IN-BAND"
PERFORMANCE_OUT_OF_BAND = "OUT-OF-BAND"
PERFORMANCE_BASELINE_RECORDED = "BASELINE-RECORDED"
PERFORMANCE_NOT_MEASURED = "NOT-MEASURED"

EXIT_OK = 0
EXIT_FUNCTIONAL_FAIL = 2
EXIT_CONFIG_ERROR = 3
EXIT_BASELINE_RECORDED = 4
EXIT_PERFORMANCE_NOT_IN_BAND = 5

# The fixed prompt is part of the gate, not of the site. Changing it
# invalidates every recorded expected-output file, so it is versioned.
DEFAULT_PROMPT_ID = "fixed-prompt/v1"
DEFAULT_PROMPT = (
    "Answer with a single short paragraph and nothing else. "
    "Describe what a ring topology is in computer networking, "
    "in exactly three sentences."
)
DEFAULT_SEED = 20260729
DEFAULT_MAX_TOKENS = 64
DEFAULT_BENCH_MAX_TOKENS = 128

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/+@:-]*[A-Za-z0-9._+@:-]$")
_ANGLE_PLACEHOLDER = re.compile(r"<[A-Z][A-Z0-9_-]*>")

DEFAULT_GATE_CONFIG: dict = {
    "schema": GATE_CONFIG_SCHEMA,
    "ssh": {"command": ["ssh", "-o", "BatchMode=yes"]},
    "runtime": {
        "lock_path": "recipes/glm52-nf3-hybrid.json",
        "verify_script": "/opt/sparkring/verify-runtime.py",
        "manifest_path": "/opt/sparkring/runtime-manifest.json",
        "expect_runtime_id": None,
        "exec_prefix": [],
        "attestation_commands": [],
        "timeout_seconds": 600,
        "model_identity": {
            "config_path": None,
            "repository_path": None,
            "revision_path": None,
            "in_container": True,
        },
    },
    "fabric": {
        "qualifier": "spark_transport/scripts/qualify_direct_cable.py",
        "probe_binary": "/tmp/spark_transport_probe",
        "iterations": 10000,
        "timeout_seconds": 1800,
        "model_down_probe": {
            "command": [],
            "per_rank": True,
            "timeout_seconds": 1800,
        },
    },
    "launch": {
        "start_command": [],
        "stop_command": [],
        "rollback_verify_command": [],
        "rank_status_command": [],
        "ready_timeout_seconds": 3600,
        "stop_timeout_seconds": 900,
    },
    "api": {"scheme": "http", "rank_bases": {}},
    "acceptance": {
        "expected_generation_path": None,
        "prompt": None,
        "seed": DEFAULT_SEED,
        "max_tokens": DEFAULT_MAX_TOKENS,
    },
    "performance": {
        "cells": [
            {"concurrency": 1, "max_tokens": DEFAULT_BENCH_MAX_TOKENS},
            {"concurrency": 8, "max_tokens": DEFAULT_BENCH_MAX_TOKENS},
        ],
        "band": None,
    },
    "preflight": {"result_path": None},
    "extensions": {"pre_shutdown": []},
    "matrix": {
        "serving": {
            "tensor_parallel_size": REQUIRED_TP,
            "decode_context_parallel_size": REQUIRED_DCP,
            "mtp_mode": REQUIRED_MTP_MODE,
            "mtp_tokens": REQUIRED_MTP_TOKENS,
            "max_model_len": REQUIRED_MAX_MODEL_LEN,
            "kv_cache_bytes_per_rank": REQUIRED_KV_BYTES_PER_RANK,
            "max_num_seqs": REQUIRED_MAX_NUM_SEQS,
        },
        "required_concurrencies": list(REQUIRED_CONCURRENCIES),
    },
}


class GateConfigError(Exception):
    """Configuration or matrix violation. Nothing is executed."""


class StageFailure(Exception):
    """A functional stage failed. Aborts the run at that stage."""

    def __init__(
        self,
        message: str,
        details: dict | None = None,
        artifacts: Sequence[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.artifacts = list(artifacts or ())


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def percentile(values: Sequence[float], q: float) -> float | None:
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    index = max(0, math.ceil(q / 100.0 * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def deep_merge(base: dict, overlay: Any) -> dict:
    merged = copy.deepcopy(base)
    if not isinstance(overlay, dict):
        return merged
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


_MISSING = object()


def dig(node: Any, dotted: str, default: Any = _MISSING) -> Any:
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise GateConfigError(f"missing required key {dotted!r}")
            return default
        node = node[part]
    return node


def as_argv(value: Any, what: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise GateConfigError(
            f"gate config: {what} must be a list of strings (argv form), "
            f"got {value!r}"
        )
    return list(value)


# ---------------------------------------------------------------------------
# Execution surfaces (injectable so the gate is testable with no cluster)
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float


@dataclass
class StreamSample:
    ttft_seconds: float | None
    total_seconds: float
    tokens: int
    text: str
    error: str | None = None
    token_count_source: str = "stream-events"


class SubprocessExecutor:
    """Real command execution. Only ever constructed for --execute runs."""

    def run(self, argv: Sequence[str], timeout: float | None = None) -> CommandResult:
        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 - argv form, no shell
                list(argv), capture_output=True, text=True, timeout=timeout
            )
            return CommandResult(
                argv=list(argv),
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_seconds=time.monotonic() - started,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                argv=list(argv),
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=f"timeout after {timeout}s",
                duration_seconds=time.monotonic() - started,
            )
        except OSError as exc:
            return CommandResult(
                argv=list(argv),
                exit_code=127,
                stdout="",
                stderr=f"cannot execute: {exc}",
                duration_seconds=time.monotonic() - started,
            )


class UrllibHttpClient:
    """Real HTTP. Transport errors are reported as status 0, never raised."""

    def get_json(self, url: str, timeout: float = 30.0) -> tuple[int, Any]:
        return self._send(urllib.request.Request(url, method="GET"), timeout)

    def post_json(
        self, url: str, payload: dict, timeout: float = 300.0
    ) -> tuple[int, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(request, timeout)

    def stream_completion(
        self, url: str, payload: dict, timeout: float = 1800.0
    ) -> StreamSample:
        body = dict(payload)
        body["stream"] = True
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        first_at: float | None = None
        chunks: list[str] = []
        stream_events = 0
        usage_tokens: int | None = None
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload_text = line[5:].strip()
                    if payload_text == "[DONE]":
                        break
                    try:
                        event = json.loads(payload_text)
                    except json.JSONDecodeError:
                        continue
                    usage = event.get("usage")
                    if isinstance(usage, dict) and isinstance(
                        usage.get("completion_tokens"), int
                    ):
                        usage_tokens = int(usage["completion_tokens"])
                    choices = event.get("choices") or [{}]
                    text = choices[0].get("text") or ""
                    if not text:
                        text = (choices[0].get("delta") or {}).get("content") or ""
                    if text:
                        if first_at is None:
                            first_at = time.monotonic()
                        stream_events += 1
                        chunks.append(text)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return StreamSample(
                ttft_seconds=None,
                total_seconds=time.monotonic() - started,
                tokens=usage_tokens if usage_tokens is not None else stream_events,
                text="".join(chunks),
                error=str(exc),
                token_count_source=(
                    "usage.completion_tokens"
                    if usage_tokens is not None
                    else "stream-events"
                ),
            )
        finished = time.monotonic()
        return StreamSample(
            ttft_seconds=None if first_at is None else first_at - started,
            total_seconds=finished - started,
            tokens=usage_tokens if usage_tokens is not None else stream_events,
            text="".join(chunks),
            token_count_source=(
                "usage.completion_tokens"
                if usage_tokens is not None
                else "stream-events"
            ),
        )

    def _send(self, request: Any, timeout: float) -> tuple[int, Any]:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", "replace")
                status = int(getattr(response, "status", 0) or 0)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
            status = int(exc.code)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return 0, str(exc)
        try:
            return status, json.loads(raw)
        except json.JSONDecodeError:
            return status, raw


class RefusingExecutor:
    """Dry-run guard: any attempt to execute or connect is a bug."""

    def _refuse(self, what: str) -> None:
        raise RuntimeError(
            f"dry-run attempted to {what}; --dry-run must touch nothing "
            "(this is a gate bug, not a config error)"
        )

    def run(self, argv: Sequence[str], timeout: float | None = None) -> CommandResult:
        self._refuse(f"execute {list(argv)!r}")
        raise AssertionError("unreachable")

    def get_json(self, url: str, timeout: float = 30.0) -> tuple[int, Any]:
        self._refuse(f"GET {url}")
        raise AssertionError("unreachable")

    def post_json(
        self, url: str, payload: dict, timeout: float = 300.0
    ) -> tuple[int, Any]:
        self._refuse(f"POST {url}")
        raise AssertionError("unreachable")

    def stream_completion(
        self, url: str, payload: dict, timeout: float = 1800.0
    ) -> StreamSample:
        self._refuse(f"stream {url}")
        raise AssertionError("unreachable")


class Bundle:
    """Evidence bundle writer. Disabled (and refusing) during a dry run."""

    def __init__(self, root: Path | None) -> None:
        self.root = root

    def _stage_dir(self, order: int, stage_id: str) -> Path:
        assert self.root is not None
        directory = self.root / "stages" / f"{order:02d}-{stage_id}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_text(self, order: int, stage_id: str, name: str, text: str) -> str:
        if self.root is None:
            raise RuntimeError(
                "dry-run attempted to write an artifact; --dry-run must touch nothing"
            )
        path = self._stage_dir(order, stage_id) / name
        path.write_text(text, encoding="utf-8")
        return str(path.relative_to(self.root)).replace("\\", "/")

    def write_json(self, order: int, stage_id: str, name: str, obj: Any) -> str:
        return self.write_text(
            order, stage_id, name, json.dumps(obj, indent=2, sort_keys=True) + "\n"
        )

    def write_root_json(self, name: str, obj: Any) -> str:
        if self.root is None:
            raise RuntimeError(
                "dry-run attempted to write a bundle file; --dry-run must touch nothing"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / name).write_text(
            json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return name


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class GateContext:
    site: dict
    gate: dict
    site_path: Path
    site_sha256: str
    gate_config_sha256: str
    repo_root: Path
    lock: dict
    executor: Any
    http: Any
    bundle: Bundle
    args: argparse.Namespace
    placeholder_warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stage_order: int = 0
    stage_id: str = ""

    # -- site accessors -----------------------------------------------------

    def ranks(self) -> list[dict]:
        return sorted(dig(self.site, "ranks"), key=lambda entry: int(entry["id"]))

    def rank(self, rank_id: int) -> dict:
        for entry in self.ranks():
            if int(entry["id"]) == int(rank_id):
                return entry
        raise GateConfigError(f"site config has no rank {rank_id}")

    def master_rank_id(self) -> int:
        return int(dig(self.site, "serving.master_rank"))

    def api_bases(self) -> dict[int, str]:
        """Rank -> API base URL. The site schema serves the API on one rank."""
        overrides = dig(self.gate, "api.rank_bases", {}) or {}
        scheme = dig(self.gate, "api.scheme", "http")
        bases: dict[int, str] = {}
        for key, value in overrides.items():
            bases[int(key)] = str(value).rstrip("/")
        if not bases:
            master = self.rank(self.master_rank_id())
            address = dig(master, "management.address")
            port = int(dig(self.site, "serving.api_port"))
            bases[self.master_rank_id()] = f"{scheme}://{address}:{port}"
        return bases

    def primary_api_base(self) -> str:
        bases = self.api_bases()
        master = self.master_rank_id()
        if master in bases:
            return bases[master]
        return bases[sorted(bases)[0]]

    def ssh_prefix(self, rank: dict) -> list[str]:
        command = as_argv(dig(self.gate, "ssh.command"), "ssh.command")
        return command + [str(rank["ssh_target"])]

    def rank_argv(
        self, rank: dict, argv: Sequence[str], in_container: bool = False
    ) -> list[str]:
        prefix = self.ssh_prefix(rank)
        if in_container:
            prefix = prefix + as_argv(
                dig(self.gate, "runtime.exec_prefix", []), "runtime.exec_prefix"
            )
        replacements = {
            "{rank}": str(rank["id"]),
            "{ssh_target}": str(rank["ssh_target"]),
        }
        expanded = []
        for token in prefix + list(argv):
            value = str(token)
            for marker, replacement in replacements.items():
                value = value.replace(marker, replacement)
            expanded.append(value)
        return expanded

    def edge_endpoints(self, edge: dict) -> tuple[dict, dict]:
        """Return (left, right) endpoint descriptors for one cable edge."""
        endpoints = list(edge.get("endpoints") or ())
        if len(endpoints) != 2:
            raise GateConfigError(
                f"topology edge {edge.get('id')!r} does not have two endpoints"
            )
        descriptors = []
        for rank_id in endpoints:
            rank = self.rank(int(rank_id))
            port = next(
                (
                    entry
                    for entry in rank.get("ring_ports", [])
                    if entry.get("edge") == edge.get("id")
                ),
                None,
            )
            if port is None:
                raise GateConfigError(
                    f"rank {rank_id} has no ring port on edge {edge.get('id')!r}"
                )
            descriptors.append(
                {
                    "rank": int(rank_id),
                    "ssh_target": str(rank["ssh_target"]),
                    "interface": str(port["interface"]),
                    "address": str(port["address"]),
                    "gid_index": int(port["roce_gid_index"]),
                }
            )
        return descriptors[0], descriptors[1]

    def repo_path(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.repo_root / candidate

    # -- artifact helpers ---------------------------------------------------

    def artifact_json(self, name: str, obj: Any) -> str:
        return self.bundle.write_json(self.stage_order, self.stage_id, name, obj)

    def capture_command(self, name: str, result: CommandResult) -> str:
        return self.artifact_json(
            name,
            {
                "argv": result.argv,
                "exit_code": result.exit_code,
                "duration_seconds": round(result.duration_seconds, 6),
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )


@dataclass
class StageOutcome:
    status: str
    message: str
    details: dict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)


@dataclass
class StageResult:
    id: str
    order: int
    title: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    message: str
    artifacts: list[str]
    details: dict

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "order": self.order,
            "title": self.title,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
            "artifacts": self.artifacts,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_site_config(path: Path) -> tuple[dict, str, list[str]]:
    """Load the site config through scripts/sparkring_site.py.

    That module owns the schema. When it is unavailable the gate degrades to
    reading an already-normalised JSON document (the shape
    ``SiteConfig.to_dict()`` produces) and says so loudly -- schema validation
    is then limited to the matrix checks this gate performs itself.
    """
    if not path.is_file():
        raise GateConfigError(f"site config not found: {path}")
    raw = path.read_bytes()
    digest = sha256_hex(raw)

    loader = None
    site_error = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import sparkring_site  # type: ignore

        loader = sparkring_site.load_site
        site_error = getattr(sparkring_site, "SiteConfigError", Exception)
    except Exception:  # noqa: BLE001 - absent or broken module must degrade
        loader = None

    if loader is None:
        print(
            "acceptance-gate: NOTE: scripts/sparkring_site.py is unavailable; "
            "reading the site config as pre-normalised JSON. Schema validation "
            "is limited to this gate's matrix checks "
            f"(see {TARGET_DOC} section 7).",
            file=sys.stderr,
        )
        try:
            site = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GateConfigError(
                f"site config {path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(site, dict):
            raise GateConfigError(f"site config {path} must be a JSON object")
        return site, digest, []

    try:
        config = loader(str(path))
    except Exception as exc:  # noqa: BLE001 - surface loader errors verbatim
        if site_error is not None and isinstance(exc, site_error):
            raise GateConfigError(f"site config is invalid: {exc}") from exc
        raise GateConfigError(f"sparkring_site.load_site({path}) failed: {exc}") from exc

    warnings: list[str] = []
    if hasattr(config, "placeholder_warnings"):
        try:
            warnings = list(config.placeholder_warnings())
        except Exception:  # noqa: BLE001 - warnings are advisory
            warnings = []
    return config.to_dict(), digest, warnings


def load_gate_config(path: Path | None) -> tuple[dict, str]:
    if path is None:
        return copy.deepcopy(DEFAULT_GATE_CONFIG), sha256_hex(
            canonical_json(DEFAULT_GATE_CONFIG).encode("utf-8")
        )
    if not path.is_file():
        raise GateConfigError(f"gate config not found: {path}")
    raw = path.read_bytes()
    try:
        overlay = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GateConfigError(f"gate config {path} is not valid JSON: {exc}") from exc
    if not isinstance(overlay, dict):
        raise GateConfigError(f"gate config {path} must be a JSON object")
    schema = overlay.get("schema")
    if schema is not None and schema != GATE_CONFIG_SCHEMA:
        raise GateConfigError(
            f"gate config schema is {schema!r}, expected {GATE_CONFIG_SCHEMA!r}"
        )
    return deep_merge(DEFAULT_GATE_CONFIG, overlay), sha256_hex(raw)


def gate_placeholder_warnings(document: Any, path: str = "gate") -> list[str]:
    """Find shipped placeholders while still allowing useful dry-run plans."""
    warnings: list[str] = []
    if isinstance(document, dict):
        for key, value in document.items():
            warnings.extend(gate_placeholder_warnings(value, f"{path}.{key}"))
    elif isinstance(document, list):
        for index, value in enumerate(document):
            warnings.extend(gate_placeholder_warnings(value, f"{path}[{index}]"))
    elif isinstance(document, str):
        for item in sorted(set(_ANGLE_PLACEHOLDER.findall(document))):
            warnings.append(f"{path} contains unresolved {item}")
    return warnings


def load_runtime_lock(path: Path) -> dict:
    if not path.is_file():
        raise GateConfigError(
            f"runtime lock not found: {path} (gate config runtime.lock_path)"
        )
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GateConfigError(f"runtime lock {path} is not valid JSON: {exc}") from exc
    if not isinstance(lock, dict):
        raise GateConfigError(f"runtime lock {path} must be a JSON object")
    return lock


# ---------------------------------------------------------------------------
# Matrix validation
# ---------------------------------------------------------------------------


def check_model_pin(lock: dict, site: dict | None = None) -> tuple[bool, str, dict]:
    """The model must be an immutable revision with the pinned config hash.

    ``verify-runtime.py`` verifies the installed runtime against its manifest;
    it does not decide whether the *lock's* model reference is pinnable. That
    is this check, and it is deliberately fail-closed: a mutable reference is
    not a pinned configuration, and a result measured against one is not
    reportable.
    """
    model = lock.get("model")
    if not isinstance(model, dict):
        return False, "runtime lock has no 'model' block", {}
    repository = str(model.get("repository") or "")
    revision = str(model.get("revision") or "")
    config_sha = str(model.get("config_sha256") or "")
    details = {
        "repository": repository,
        "revision": revision,
        "config_sha256": config_sha,
    }
    if site is not None:
        details["site_model_repo"] = str(dig(site, "runtime.model_repo", ""))
        details["site_model_revision"] = str(dig(site, "runtime.model_revision", ""))
    if not repository:
        return False, "runtime lock model.repository is empty", details
    if not _IMMUTABLE_REVISION.match(revision.lower()):
        return (
            False,
            (
                f"runtime lock model.revision is {revision!r}, which is not an "
                "immutable 40-hex commit hash. Mutable references (branch "
                "names, tags, 'latest', 'pending', empty) are rejected: pin the "
                "revision in the runtime lock before running acceptance "
                f"({TARGET_DOC} section 2.2)."
            ),
            details,
        )
    if not _SHA256_HEX.match(config_sha.lower()):
        return (
            False,
            (
                f"runtime lock model.config_sha256 is {config_sha!r}, which is "
                "not a 64-hex sha256; the model identity gate cannot run"
            ),
            details,
        )
    if site is not None:
        site_repo = str(dig(site, "runtime.model_repo", ""))
        site_revision = str(dig(site, "runtime.model_revision", "")).lower()
        if site_repo and site_repo != repository:
            return (
                False,
                (
                    f"site runtime.model_repo {site_repo!r} does not match the "
                    f"pinned lock model.repository {repository!r}; the matrix "
                    f"pins one checkpoint ({TARGET_DOC} section 2.3)"
                ),
                details,
            )
        if site_revision and site_revision != revision.lower():
            return (
                False,
                (
                    f"site runtime.model_revision {site_revision!r} does not "
                    f"match the pinned lock model.revision {revision!r}"
                ),
                details,
            )
    return True, f"model pinned to {repository}@{revision[:12]}", details


def _validate_topology(site: dict, problems: list[str]) -> None:
    topology = site.get("topology")
    if not isinstance(topology, dict):
        problems.append("site config: 'topology' block is required")
        return
    if topology.get("mtu") != REQUIRED_MTU:
        problems.append(
            f"topology.mtu must be {REQUIRED_MTU} (matrix requirement), got "
            f"{topology.get('mtu')!r}"
        )
    if topology.get("link_speed_mbps") != REQUIRED_LINK_SPEED_MBPS:
        problems.append(
            f"topology.link_speed_mbps must be {REQUIRED_LINK_SPEED_MBPS} "
            f"(200 GbE), got {topology.get('link_speed_mbps')!r}"
        )
    edges = topology.get("edges")
    if not isinstance(edges, list) or len(edges) != REQUIRED_EDGES:
        problems.append(
            f"topology.edges must list exactly {REQUIRED_EDGES} cables (the "
            "4-cycle); there is no switched or partial-mesh variant of this "
            "matrix"
        )
        return
    degree: dict[int, int] = {}
    subnets: list[str] = []
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            problems.append(f"topology.edges[{index}] must be an object")
            continue
        endpoints = edge.get("endpoints") or []
        if len(endpoints) != 2 or endpoints[0] == endpoints[1]:
            problems.append(
                f"topology.edges[{index}] must join two distinct ranks, got "
                f"{endpoints!r}"
            )
            continue
        for rank_id in endpoints:
            degree[int(rank_id)] = degree.get(int(rank_id), 0) + 1
        subnet = edge.get("subnet")
        try:
            network = ipaddress.ip_network(str(subnet), strict=False)
        except ValueError as exc:
            problems.append(f"topology.edges[{index}].subnet invalid: {exc}")
            continue
        if network.prefixlen != 24:
            problems.append(
                f"topology.edges[{index}].subnet {subnet} must be a /24 "
                "(one dedicated /24 per physical cable)"
            )
        subnets.append(str(network))
    for rank_id in range(REQUIRED_RANKS):
        if degree.get(rank_id, 0) != 2:
            problems.append(
                f"topology: rank {rank_id} has degree {degree.get(rank_id, 0)}, "
                "expected 2 (a 4-cycle: every node has exactly two cables)"
            )
    if len(set(subnets)) != len(subnets):
        problems.append(
            f"topology.edges: the four cables must use four DISTINCT /24 "
            f"subnets; got {subnets}"
        )


def _validate_serving(site: dict, gate: dict, problems: list[str]) -> None:
    serving = site.get("serving")
    if not isinstance(serving, dict):
        problems.append("site config: 'serving' block is required")
        return
    expectations = dig(gate, "matrix.serving", {})
    if not isinstance(expectations, dict) or not expectations:
        problems.append("gate config: matrix.serving must be a non-empty object")
        return
    for key, expected in expectations.items():
        if serving.get(key) != expected:
            problems.append(
                f"serving.{key} must be {expected!r} for the supported matrix, "
                f"got {serving.get(key)!r} ({TARGET_DOC} section 2.4)"
            )
    master = serving.get("master_rank")
    if not isinstance(master, int) or master not in range(REQUIRED_RANKS):
        problems.append(
            f"serving.master_rank must be a rank id in 0..{REQUIRED_RANKS - 1}, "
            f"got {master!r}"
        )


def _validate_gate_config(gate: dict, problems: list[str]) -> None:
    for key in ("launch.start_command", "launch.stop_command"):
        value = dig(gate, key, [])
        if not isinstance(value, list) or not value:
            problems.append(
                f"gate config: {key} must be a non-empty argv list -- the gate "
                "never implements launching, it invokes your launcher"
            )
    probe = dig(gate, "fabric.model_down_probe.command", [])
    if not isinstance(probe, list) or not probe:
        problems.append(
            "gate config: fabric.model_down_probe.command must be a non-empty "
            "argv list (the repo's model-down four-rank collective probe "
            "entrypoint); the gate does not reimplement it"
        )
    if not dig(gate, "fabric.qualifier", ""):
        problems.append(
            "gate config: fabric.qualifier must point at "
            "spark_transport/scripts/qualify_direct_cable.py"
        )
    delegated_attestation = dig(gate, "runtime.attestation_commands", []) or []
    if delegated_attestation:
        if not isinstance(delegated_attestation, list) or any(
            not isinstance(command, list) or not command
            for command in delegated_attestation
        ):
            problems.append(
                "gate config: runtime.attestation_commands must be a list of "
                "non-empty argv lists"
            )
    else:
        for key in ("runtime.verify_script", "runtime.manifest_path"):
            if not dig(gate, key, ""):
                problems.append(f"gate config: {key} is required")
        for key in (
            "runtime.model_identity.config_path",
            "runtime.model_identity.repository_path",
            "runtime.model_identity.revision_path",
        ):
            value = dig(gate, key, None)
            if not isinstance(value, str) or not _REMOTE_PATH.fullmatch(value):
                problems.append(
                    f"gate config: {key} must be an explicit, shell-safe absolute "
                    "path on every rank"
                )
    in_container = dig(gate, "runtime.model_identity.in_container", True)
    if not isinstance(in_container, bool):
        problems.append(
            "gate config: runtime.model_identity.in_container must be true or false"
        )
    cells = dig(gate, "performance.cells", []) or []
    concurrencies = {c.get("concurrency") for c in cells if isinstance(c, dict)}
    for index, cell in enumerate(cells):
        prefix = f"gate config: performance.cells[{index}]"
        if not isinstance(cell, dict):
            problems.append(f"{prefix} must be an object")
            continue
        for cell_field in ("concurrency", "max_tokens", "repetitions"):
            if cell_field in cell and (
                not isinstance(cell[cell_field], int) or cell[cell_field] <= 0
            ):
                problems.append(
                    f"{prefix}.{cell_field} must be a positive integer"
                )
        minimum = cell.get("minimum_tokens")
        if minimum is not None and (
            not isinstance(minimum, int)
            or minimum <= 0
            or minimum > cell.get("max_tokens", DEFAULT_BENCH_MAX_TOKENS)
        ):
            problems.append(
                f"{prefix}.minimum_tokens must be positive and no greater "
                "than max_tokens"
            )
        if "ignore_eos" in cell and not isinstance(cell["ignore_eos"], bool):
            problems.append(f"{prefix}.ignore_eos must be true or false")
    required = dig(gate, "matrix.required_concurrencies", [])
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(value, int) or value <= 0 for value in required)
    ):
        problems.append(
            "gate config: matrix.required_concurrencies must be a non-empty "
            "list of positive integers"
        )
        required = []
    missing = [c for c in required if c not in concurrencies]
    if missing:
        problems.append(
            "gate config: performance.cells must include every concurrency in "
            f"matrix.required_concurrencies; missing {missing}"
        )
    extensions = dig(gate, "extensions.pre_shutdown", []) or []
    if not isinstance(extensions, list):
        problems.append("gate config: extensions.pre_shutdown must be a list")
    else:
        seen: set[str] = set()
        for index, extension in enumerate(extensions):
            prefix = f"gate config: extensions.pre_shutdown[{index}]"
            if not isinstance(extension, dict):
                problems.append(f"{prefix} must be an object")
                continue
            identifier = extension.get("id")
            if not isinstance(identifier, str) or not re.fullmatch(
                r"[a-z0-9]+(?:-[a-z0-9]+)*", identifier
            ):
                problems.append(f"{prefix}.id must be a lowercase kebab-case id")
            elif identifier in seen:
                problems.append(f"{prefix}.id duplicates {identifier!r}")
            else:
                seen.add(identifier)
            command = extension.get("command")
            if not isinstance(command, list) or not command:
                problems.append(f"{prefix}.command must be a non-empty argv list")
            timeout = extension.get("timeout_seconds", 3600)
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                problems.append(f"{prefix}.timeout_seconds must be positive")
            mutates = extension.get("mutates_remote", False)
            if not isinstance(mutates, bool):
                problems.append(f"{prefix}.mutates_remote must be true or false")
            require_json = extension.get("require_json_status", False)
            if not isinstance(require_json, bool):
                problems.append(
                    f"{prefix}.require_json_status must be true or false"
                )


def validate_configuration(site: dict, gate: dict, lock: dict) -> list[str]:
    """Return matrix/config problems. Empty means the plan is valid."""
    problems: list[str] = []

    ranks = site.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != REQUIRED_RANKS:
        problems.append(
            f"site config: 'ranks' must list exactly {REQUIRED_RANKS} ranks "
            "(the matrix is 4x DGX Spark; there is no other node count)"
        )
    else:
        ids = sorted(int(entry.get("id", -1)) for entry in ranks)
        if ids != list(range(REQUIRED_RANKS)):
            problems.append(
                f"site config: rank ids must be exactly 0..{REQUIRED_RANKS - 1}, "
                f"got {ids}"
            )
        for entry in ranks:
            if not entry.get("ssh_target"):
                problems.append(f"rank {entry.get('id')!r} has no ssh_target")
            ports = entry.get("ring_ports") or []
            if len(ports) != 2:
                problems.append(
                    f"rank {entry.get('id')!r} must have exactly 2 ring ports "
                    f"(one per cable), got {len(ports)}"
                )

    _validate_topology(site, problems)
    _validate_serving(site, gate, problems)
    _validate_gate_config(gate, problems)

    ok, message, _ = check_model_pin(lock, site if isinstance(ranks, list) else None)
    if not ok:
        problems.append(message)

    return problems


# ---------------------------------------------------------------------------
# Parsing helpers for delegated tools
# ---------------------------------------------------------------------------


def parse_verify_runtime_stdout(text: str) -> tuple[dict | None, dict | None]:
    """Split ``verify-runtime.py --json --emit-attestation`` stdout.

    Returns (checks_object, attestation_object); either may be None when the
    output does not contain it. The caller decides whether that is fatal.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    attestation: dict | None = None
    if lines:
        try:
            candidate = json.loads(lines[-1])
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict) and "manifest_self_hash" in candidate:
            attestation = candidate
            lines = lines[:-1]
    checks: dict | None = None
    if lines:
        try:
            parsed = json.loads("\n".join(lines))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            checks = parsed
    return checks, attestation


def parse_qualifier_stdout(text: str) -> dict | None:
    """``qualify_direct_cable.py`` writes one JSON object to stdout."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        for line in reversed(stripped.splitlines()):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            return parsed if isinstance(parsed, dict) else None
        return None
    return parsed if isinstance(parsed, dict) else None


def evaluate_band(observed: dict, band: dict) -> tuple[list[dict], list[str]]:
    """Compare observed cell metrics against the documented tolerance band.

    ``band := {"C1": {"metric": {"min": x} | {"max": y}
                                | {"expected": e, "tolerance_pct": p}}}``
    Returns (comparisons, violation messages).
    """
    comparisons: list[dict] = []
    violations: list[str] = []
    for cell_label, metrics in sorted(band.items()):
        if not isinstance(metrics, dict):
            violations.append(f"band[{cell_label}] must be an object")
            continue
        cell = observed.get(cell_label)
        if not isinstance(cell, dict):
            violations.append(
                f"band[{cell_label}] is documented but cell {cell_label} was "
                "not measured"
            )
            continue
        for metric, bounds in sorted(metrics.items()):
            value = cell.get(metric)
            low = bounds.get("min") if isinstance(bounds, dict) else None
            high = bounds.get("max") if isinstance(bounds, dict) else None
            if isinstance(bounds, dict) and "expected" in bounds:
                expected = float(bounds["expected"])
                tolerance = float(bounds.get("tolerance_pct", 0.0))
                low = expected * (1.0 - tolerance / 100.0)
                high = expected * (1.0 + tolerance / 100.0)
            entry = {
                "cell": cell_label,
                "metric": metric,
                "observed": value,
                "min": low,
                "max": high,
            }
            if value is None:
                entry["in_band"] = False
                violations.append(f"{cell_label}.{metric}: banded but not measured")
            else:
                in_band = True
                if low is not None and float(value) < float(low):
                    in_band = False
                if high is not None and float(value) > float(high):
                    in_band = False
                entry["in_band"] = in_band
                if not in_band:
                    violations.append(
                        f"{cell_label}.{metric}: observed {value} outside "
                        f"[{low}, {high}]"
                    )
            comparisons.append(entry)
    return comparisons, violations


# ---------------------------------------------------------------------------
# Stage 1 - runtime image + artifact attestation
# ---------------------------------------------------------------------------


def _verify_runtime_argv(ctx: GateContext, rank: dict) -> list[str]:
    expect = dig(ctx.gate, "runtime.expect_runtime_id", None) or ctx.lock.get(
        "runtime_id"
    )
    argv = [
        "python3",
        str(dig(ctx.gate, "runtime.verify_script")),
        "--json",
        "--emit-attestation",
        "--manifest",
        str(dig(ctx.gate, "runtime.manifest_path")),
    ]
    if expect:
        argv += ["--expect-runtime-id", str(expect)]
    return ctx.rank_argv(rank, argv, in_container=True)


def _preflight_path(ctx: GateContext) -> Path | None:
    configured = ctx.args.preflight or dig(ctx.gate, "preflight.result_path", None)
    return Path(configured) if configured else None


def _model_identity_argv(
    ctx: GateContext, rank: dict, field: str
) -> list[str]:
    in_container = bool(
        dig(ctx.gate, "runtime.model_identity.in_container", True)
    )
    path = str(dig(ctx.gate, f"runtime.model_identity.{field}_path"))
    command = ["sha256sum", "--", path] if field == "config" else ["cat", "--", path]
    return ctx.rank_argv(rank, command, in_container=in_container)


def _load_required_preflight(ctx: GateContext) -> tuple[dict, str]:
    path = _preflight_path(ctx)
    if path is None:
        raise StageFailure(
            "preflight evidence is required for public acceptance; run "
            "scripts/preflight.py and supply its successful JSON with --preflight"
        )
    if not path.is_file():
        raise StageFailure(f"preflight evidence not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise StageFailure(f"preflight evidence {path} is unreadable: {exc}") from exc

    problems: list[str] = []
    if document.get("schema") != PREFLIGHT_SCHEMA:
        problems.append(
            f"schema {document.get('schema')!r} != expected {PREFLIGHT_SCHEMA!r}"
        )
    if document.get("read_only") is not True:
        problems.append("read_only is not true")
    if document.get("passed") is not True:
        problems.append(
            f"passed is not true (failed checks: {document.get('failed_check_ids')})"
        )
    totals = document.get("totals")
    if not isinstance(totals, dict):
        problems.append("totals is missing")
    else:
        checks = totals.get("checks")
        failed = totals.get("failed")
        if not isinstance(checks, int) or isinstance(checks, bool) or checks <= 0:
            problems.append("totals.checks must be a positive integer")
        if failed != 0:
            problems.append(f"totals.failed is {failed!r}, expected 0")
    rank_entries = document.get("ranks")
    observed_ranks = (
        {
            entry.get("rank")
            for entry in rank_entries
            if isinstance(entry, dict)
        }
        if isinstance(rank_entries, list)
        else set()
    )
    expected_ranks = set(range(REQUIRED_RANKS))
    if observed_ranks != expected_ranks:
        problems.append(
            f"rank evidence is {sorted(observed_ranks, key=str)}, "
            f"expected {sorted(expected_ranks)}"
        )
    if document.get("placeholder_warnings"):
        problems.append("placeholder_warnings is not empty")
    if problems:
        raise StageFailure(
            "preflight evidence is not an admissible successful full-cluster "
            "run: " + "; ".join(problems)
        )

    summary = {
        "schema": document.get("schema"),
        "generated_at": document.get("generated_at"),
        "read_only": document.get("read_only"),
        "passed": document.get("passed"),
        "totals": totals,
        "failed_check_ids": document.get("failed_check_ids"),
        "failed_ranks": document.get("failed_ranks"),
        "ranks": sorted(observed_ranks),
    }
    return summary, str(path)


def _single_identity_line(result: CommandResult, label: str) -> str:
    if result.exit_code != 0:
        raise ValueError(
            f"{label} command exited {result.exit_code}: "
            f"{result.stderr.strip()[:300]}"
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{label} command must emit exactly one non-empty line")
    return lines[0]


def _attest_deployed_model(
    ctx: GateContext, expected: dict, artifacts: list[str]
) -> dict[str, dict]:
    observed: dict[str, dict] = {}
    failures: list[str] = []
    timeout = float(dig(ctx.gate, "runtime.timeout_seconds", 600))
    for rank in ctx.ranks():
        number = int(rank["id"])
        rank_observed: dict[str, str] = {}
        for identity_field in ("repository", "revision", "config"):
            result = ctx.executor.run(
                _model_identity_argv(ctx, rank, identity_field), timeout=timeout
            )
            artifacts.append(
                ctx.capture_command(
                    f"rank{number}-model-{identity_field}.txt", result
                )
            )
            try:
                value = _single_identity_line(
                    result, f"rank {number} model {identity_field}"
                )
                if identity_field == "config":
                    value = value.split()[0].lower()
                    if not _SHA256_HEX.fullmatch(value):
                        raise ValueError(
                            f"rank {number} model config command did not emit "
                            "a sha256sum-compatible digest"
                        )
                rank_observed[identity_field] = value
            except ValueError as exc:
                failures.append(str(exc))
        observed[str(number)] = rank_observed
        comparisons = {
            "repository": expected["repository"],
            "revision": expected["revision"].lower(),
            "config": expected["config_sha256"].lower(),
        }
        labels = {
            "repository": "repository",
            "revision": "revision",
            "config": "config sha256",
        }
        for identity_field, wanted in comparisons.items():
            actual = rank_observed.get(identity_field)
            if actual is not None and actual != wanted:
                failures.append(
                    f"rank {number} deployed model {labels[identity_field]} "
                    f"{actual!r} != lock {wanted!r}"
                )
    if failures:
        raise StageFailure(
            "deployed model identity verification failed: " + " | ".join(failures),
            details={"model_identity": observed, "failures": failures},
            artifacts=artifacts,
        )
    return observed


def plan_runtime_attestation(ctx: GateContext) -> list[dict]:
    preflight = _preflight_path(ctx)
    actions = [
        {
            "kind": "local-check",
            "what": "require successful full-cluster scripts/preflight.py evidence",
            "path": str(preflight) if preflight else None,
            "on_missing": "functional failure before any remote command",
        }
    ]
    delegated = dig(ctx.gate, "runtime.attestation_commands", []) or []
    if delegated:
        actions += [
            {
                "kind": "command",
                "what": f"run delegated all-rank attestation {index}",
                "argv": as_argv(
                    command, f"runtime.attestation_commands[{index}]"
                ),
            }
            for index, command in enumerate(delegated, start=1)
        ]
        actions.append(
            {
                "kind": "local-check",
                "what": "model identity pin (immutable revision, config sha256, "
                "site/lock agreement)",
                "source": str(dig(ctx.gate, "runtime.lock_path")),
            }
        )
        return actions
    actions += [
        {
            "kind": "command",
            "what": f"attest runtime on rank {rank['id']}",
            "argv": _verify_runtime_argv(ctx, rank),
        }
        for rank in ctx.ranks()
    ]
    for rank in ctx.ranks():
        for identity_field in ("repository", "revision", "config"):
            actions.append(
                {
                    "kind": "command",
                    "what": (
                        "verify deployed model "
                        f"{identity_field} on rank {rank['id']}"
                    ),
                    "argv": _model_identity_argv(
                        ctx, rank, identity_field
                    ),
                }
            )
    actions.append(
        {
            "kind": "local-check",
            "what": "model identity pin (immutable revision, config sha256, "
            "site/lock agreement)",
            "source": str(dig(ctx.gate, "runtime.lock_path")),
        }
    )
    actions.append(
        {
            "kind": "local-check",
            "what": "cross-rank identity: same runtime_id and manifest self-hash",
        }
    )
    return actions


def run_runtime_attestation(ctx: GateContext) -> StageOutcome:
    artifacts: list[str] = []
    attestations: dict[str, dict] = {}
    failures: list[str] = []
    timeout = float(dig(ctx.gate, "runtime.timeout_seconds", 600))

    preflight_summary, preflight_source = _load_required_preflight(ctx)
    artifacts.append(ctx.artifact_json("preflight-summary.json", preflight_summary))

    delegated = dig(ctx.gate, "runtime.attestation_commands", []) or []
    if delegated:
        for index, command in enumerate(delegated, start=1):
            result = ctx.executor.run(
                as_argv(command, f"runtime.attestation_commands[{index}]"),
                timeout=timeout,
            )
            artifacts.append(
                ctx.capture_command(f"delegated-attestation-{index}.json", result)
            )
            if result.exit_code != 0:
                failures.append(
                    f"delegated attestation {index} exited {result.exit_code}"
                    + (
                        f" stderr={result.stderr.strip()[:400]}"
                        if result.stderr
                        else ""
                    )
                )
        ok, message, model_details = check_model_pin(ctx.lock, ctx.site)
        if not ok:
            failures.append(message)
        if failures:
            raise StageFailure(
                "delegated runtime/model attestation failed: "
                + " | ".join(failures),
                details={"failures": failures, "model": model_details},
                artifacts=artifacts,
            )
        artifacts.append(
            ctx.artifact_json(
                "attestation-summary.json",
                {
                    "delegated_commands": len(delegated),
                    "model": model_details,
                    "preflight": preflight_summary,
                    "preflight_source": preflight_source,
                },
            )
        )
        return StageOutcome(
            status=STATUS_PASS,
            message=(
                f"{len(delegated)} delegated all-rank attestation commands "
                f"passed; {message}"
            ),
            details={
                "delegated_commands": len(delegated),
                "model": model_details,
                "preflight": preflight_summary,
            },
            artifacts=artifacts,
        )

    for rank in ctx.ranks():
        number = int(rank["id"])
        result = ctx.executor.run(_verify_runtime_argv(ctx, rank), timeout=timeout)
        artifacts.append(ctx.capture_command(f"rank{number}-verify-runtime.json", result))
        checks, attestation = parse_verify_runtime_stdout(result.stdout)
        if result.exit_code != 0:
            failing = []
            if isinstance(checks, dict):
                failing = [
                    f"{c.get('name')}: {c.get('detail')}"
                    for c in checks.get("checks", [])
                    if c.get("status") == "fail"
                ]
            failures.append(
                f"rank {number}: verify-runtime.py exited {result.exit_code}"
                + (f" ({'; '.join(failing)})" if failing else "")
                + (f" stderr={result.stderr.strip()[:400]}" if result.stderr else "")
            )
            continue
        if not isinstance(checks, dict) or not checks.get("ok"):
            failures.append(
                f"rank {number}: verify-runtime.py did not report ok=true "
                "(unparseable or failing check output)"
            )
            continue
        if not isinstance(attestation, dict):
            failures.append(
                f"rank {number}: no attestation line emitted; runtime identity "
                "cannot be compared across ranks"
            )
            continue
        attestations[str(number)] = attestation
        image_checks = [
            check
            for check in checks.get("checks", [])
            if check.get("name") == "image_digest"
        ]
        if len(image_checks) != 1 or image_checks[0].get("status") != "pass":
            detail = (
                image_checks[0].get("detail")
                if image_checks
                else "image_digest check missing"
            )
            failures.append(
                f"rank {number}: image identity was not verified ({detail}); "
                "public acceptance requires a passing digest check"
            )
        for check in checks.get("checks", []):
            if check.get("status") == "skip":
                ctx.notes.append(
                    f"rank {number} attestation check skipped -- "
                    f"{check.get('name')}: {check.get('detail')}"
                )

    if failures:
        raise StageFailure(
            "runtime attestation failed: " + " | ".join(failures),
            details={"failures": failures},
            artifacts=artifacts,
        )

    runtime_ids = {a.get("runtime_id") for a in attestations.values()}
    self_hashes = {a.get("manifest_self_hash") for a in attestations.values()}
    if len(runtime_ids) != 1 or len(self_hashes) != 1:
        raise StageFailure(
            "ranks are not running the same frozen runtime: "
            f"runtime_ids={sorted(str(r) for r in runtime_ids)} "
            f"manifest_self_hashes={sorted(str(h) for h in self_hashes)}",
            details={"attestations": attestations},
            artifacts=artifacts,
        )

    ok, message, model_details = check_model_pin(ctx.lock, ctx.site)
    if not ok:
        raise StageFailure(message, details={"model": model_details}, artifacts=artifacts)
    deployed_model = _attest_deployed_model(ctx, model_details, artifacts)

    artifacts.append(
        ctx.artifact_json(
            "attestation-summary.json",
            {
                "attestations": attestations,
                "model": model_details,
                "deployed_model": deployed_model,
                "preflight": preflight_summary,
                "preflight_source": preflight_source,
            },
        )
    )
    runtime_id = next(iter(runtime_ids))
    return StageOutcome(
        status=STATUS_PASS,
        message=(
            f"all {len(attestations)} ranks attest runtime_id={runtime_id} with "
            f"an identical manifest self-hash; {message}"
        ),
        details={
            "runtime_id": runtime_id,
            "manifest_self_hash": next(iter(self_hashes)),
            "model": model_details,
            "deployed_model": deployed_model,
            "preflight": preflight_summary,
        },
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Stage 2 - fabric qualification + model-down transport probe
# ---------------------------------------------------------------------------


def _qualifier_argv(ctx: GateContext, edge: dict) -> list[str]:
    left, right = ctx.edge_endpoints(edge)
    argv = [
        "python3",
        str(dig(ctx.gate, "fabric.qualifier")),
        "--tier",
        "roce200",
        "--left",
        left["ssh_target"],
        "--right",
        right["ssh_target"],
        "--left-interface",
        left["interface"],
        "--right-interface",
        right["interface"],
        "--left-ip",
        left["address"],
        "--right-ip",
        right["address"],
        "--expected-mtu",
        str(dig(ctx.site, "topology.mtu")),
        "--gid-index",
        str(left["gid_index"]),
        "--strict-latency",
    ]
    probe = dig(ctx.gate, "fabric.probe_binary", None)
    if probe:
        argv += ["--probe-binary", str(probe)]
    iterations = dig(ctx.gate, "fabric.iterations", None)
    if iterations:
        argv += ["--iterations", str(iterations)]
    return argv


def _model_down_probe_argv(ctx: GateContext, rank: dict) -> list[str]:
    command = as_argv(
        dig(ctx.gate, "fabric.model_down_probe.command"),
        "fabric.model_down_probe.command",
    )
    return ctx.rank_argv(rank, command)


def _probe_targets(ctx: GateContext) -> list[dict]:
    per_rank = bool(dig(ctx.gate, "fabric.model_down_probe.per_rank", True))
    return ctx.ranks() if per_rank else ctx.ranks()[:1]


def plan_fabric_transport(ctx: GateContext) -> list[dict]:
    actions = [
        {
            "kind": "command",
            "what": f"qualify cable edge {edge.get('id')}",
            "argv": _qualifier_argv(ctx, edge),
        }
        for edge in dig(ctx.site, "topology.edges")
    ]
    actions += [
        {
            "kind": "command",
            "what": f"model-down collective probe on rank {rank['id']}",
            "argv": _model_down_probe_argv(ctx, rank),
        }
        for rank in _probe_targets(ctx)
    ]
    return actions


def run_fabric_transport(ctx: GateContext) -> StageOutcome:
    artifacts: list[str] = []
    failures: list[str] = []
    edge_status: dict[str, str] = {}
    timeout = float(dig(ctx.gate, "fabric.timeout_seconds", 1800))

    for edge in dig(ctx.site, "topology.edges"):
        label = str(edge.get("id"))
        result = ctx.executor.run(_qualifier_argv(ctx, edge), timeout=timeout)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
        artifacts.append(ctx.capture_command(f"edge-{safe}-qualify.json", result))
        parsed = parse_qualifier_stdout(result.stdout)
        status = str((parsed or {}).get("status", "unparseable"))
        edge_status[label] = status
        if result.exit_code != 0:
            failures.append(
                f"edge {label}: qualifier exited {result.exit_code} "
                f"(status={status}); re-run qualify_direct_cable.py for this "
                "edge and read its JSON"
            )
        elif status not in ("qualified", "cable_qualified_with_latency_warning"):
            failures.append(f"edge {label}: qualifier status {status!r}")
        elif status == "cable_qualified_with_latency_warning":
            ctx.notes.append(f"edge {label} qualified with a latency warning")

    probe_timeout = float(dig(ctx.gate, "fabric.model_down_probe.timeout_seconds", 1800))
    targets = _probe_targets(ctx)
    probe_results: dict[int, CommandResult] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(targets)),
        thread_name_prefix="sparkring-model-down-probe",
    ) as pool:
        pending = {
            int(rank["id"]): pool.submit(
                ctx.executor.run,
                _model_down_probe_argv(ctx, rank),
                probe_timeout,
            )
            for rank in targets
        }
        for number, future in pending.items():
            try:
                probe_results[number] = future.result()
            except Exception as exc:  # defensive: injected/site executors may raise
                probe_results[number] = CommandResult(
                    argv=_model_down_probe_argv(ctx, ctx.rank(number)),
                    exit_code=127,
                    stdout="",
                    stderr=f"probe executor raised: {exc}",
                    duration_seconds=0.0,
                )

    for rank in targets:
        number = int(rank["id"])
        result = probe_results[number]
        artifacts.append(ctx.capture_command(f"rank{number}-model-down-probe.json", result))
        if result.exit_code != 0:
            failures.append(
                f"rank {number}: model-down transport probe exited "
                f"{result.exit_code}; the four-rank collectives are not proven "
                "byte-correct, so no model may be started"
            )

    if failures:
        raise StageFailure(
            "fabric/transport qualification failed: " + " | ".join(failures),
            details={"edges": edge_status, "failures": failures},
            artifacts=artifacts,
        )
    return StageOutcome(
        status=STATUS_PASS,
        message=(
            f"{len(edge_status)} cable edges qualified and the model-down "
            f"collective probe passed on {len(targets)} rank(s)"
        ),
        details={"edges": edge_status},
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Stage 3 - all-four-rank startup
# ---------------------------------------------------------------------------


def plan_rank_startup(ctx: GateContext) -> list[dict]:
    actions = [
        {
            "kind": "command",
            "what": "start all four ranks (site launcher)",
            "argv": as_argv(
                dig(ctx.gate, "launch.start_command"), "launch.start_command"
            ),
            "mutates_remote": True,
        }
    ]
    status_command = dig(ctx.gate, "launch.rank_status_command", []) or []
    for rank in ctx.ranks():
        if status_command:
            actions.append(
                {
                    "kind": "command",
                    "what": f"confirm rank {rank['id']} is running",
                    "argv": ctx.rank_argv(
                        rank, as_argv(status_command, "launch.rank_status_command")
                    ),
                }
            )
    return actions


def run_rank_startup(ctx: GateContext) -> StageOutcome:
    artifacts: list[str] = []
    timeout = float(dig(ctx.gate, "launch.ready_timeout_seconds", 3600))
    result = ctx.executor.run(
        as_argv(dig(ctx.gate, "launch.start_command"), "launch.start_command"),
        timeout=timeout,
    )
    artifacts.append(ctx.capture_command("launch-start.json", result))
    if result.exit_code != 0:
        raise StageFailure(
            f"launcher exited {result.exit_code}; the four ranks did not start. "
            f"stderr={result.stderr.strip()[:600]}",
            details={"exit_code": result.exit_code},
            artifacts=artifacts,
        )

    status_command = dig(ctx.gate, "launch.rank_status_command", []) or []
    started: dict[str, bool] = {}
    failures: list[str] = []
    if not status_command:
        ctx.notes.append(
            "no launch.rank_status_command configured; per-rank startup was "
            "inferred from the launcher exit code only"
        )
    for rank in ctx.ranks():
        number = int(rank["id"])
        if not status_command:
            started[str(number)] = True
            continue
        status = ctx.executor.run(
            ctx.rank_argv(rank, as_argv(status_command, "launch.rank_status_command")),
            timeout=timeout,
        )
        artifacts.append(ctx.capture_command(f"rank{number}-status.json", status))
        started[str(number)] = status.exit_code == 0
        if status.exit_code != 0:
            failures.append(
                f"rank {number}: rank_status_command exited {status.exit_code}"
            )

    if failures:
        raise StageFailure(
            "not all four ranks started: " + " | ".join(failures),
            details={"ranks": started},
            artifacts=artifacts,
        )
    return StageOutcome(
        status=STATUS_PASS,
        message=f"all {len(started)} ranks started",
        details={"ranks": started},
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Stage 4 - API liveness
# ---------------------------------------------------------------------------


def plan_api_liveness(ctx: GateContext) -> list[dict]:
    bases = ctx.api_bases()
    actions: list[dict] = []
    for rank in ctx.ranks():
        number = int(rank["id"])
        base = bases.get(number)
        if base is None:
            actions.append(
                {
                    "kind": "declared",
                    "what": (
                        f"rank {number} serves no API (headless; the site "
                        "config serves the API on master_rank only)"
                    ),
                }
            )
            continue
        actions.append(
            {"kind": "http", "what": f"rank {number} GET /health", "url": f"{base}/health"}
        )
        actions.append(
            {
                "kind": "http",
                "what": f"rank {number} GET /v1/models",
                "url": f"{base}/v1/models",
            }
        )
    return actions


def run_api_liveness(ctx: GateContext) -> StageOutcome:
    artifacts: list[str] = []
    failures: list[str] = []
    observed: dict[str, dict] = {}
    bases = ctx.api_bases()
    timeout = float(dig(ctx.gate, "launch.ready_timeout_seconds", 1200))
    if not bases:
        raise StageFailure("no API base could be derived from the site config")

    for rank in ctx.ranks():
        number = int(rank["id"])
        base = bases.get(number)
        if base is None:
            observed[str(number)] = {"api": "headless"}
            continue
        health_status, health_body = ctx.http.get_json(f"{base}/health", timeout=timeout)
        models_status, models_body = ctx.http.get_json(
            f"{base}/v1/models", timeout=timeout
        )
        served = []
        if isinstance(models_body, dict):
            served = [
                str(item.get("id"))
                for item in models_body.get("data", [])
                if isinstance(item, dict)
            ]
        observed[str(number)] = {
            "health_status": health_status,
            "models_status": models_status,
            "served_models": served,
        }
        artifacts.append(
            ctx.artifact_json(
                f"rank{number}-api-liveness.json",
                {
                    "health": {"status": health_status, "body": health_body},
                    "models": {"status": models_status, "body": models_body},
                },
            )
        )
        if health_status != 200:
            failures.append(f"rank {number}: /health returned {health_status}")
        if models_status != 200:
            failures.append(f"rank {number}: /v1/models returned {models_status}")
        elif not served:
            failures.append(
                f"rank {number}: /v1/models listed no served model; the pinned "
                "checkpoint is not being served"
            )

    if failures:
        raise StageFailure(
            "API liveness failed: " + " | ".join(failures),
            details={"ranks": observed},
            artifacts=artifacts,
        )
    live = [key for key, value in observed.items() if "health_status" in value]
    return StageOutcome(
        status=STATUS_PASS,
        message=(
            f"{len(live)} rank(s) serving the API answered /health and "
            f"/v1/models; {len(observed) - len(live)} rank(s) headless"
        ),
        details={"ranks": observed},
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Stage 5 - deterministic fixed-prompt generation
# ---------------------------------------------------------------------------


def served_model_name(ctx: GateContext) -> str:
    """The model name to send to the API.

    The site schema pins the model by *path inside the container*, which is
    what vLLM reports as the served model name unless the operator overrides
    it. ``acceptance.served_model_name`` overrides it here.
    """
    override = dig(ctx.gate, "acceptance.served_model_name", None)
    if override:
        return str(override)
    return str(dig(ctx.site, "runtime.model_path"))


def generation_params(ctx: GateContext) -> dict:
    prompt_override = dig(ctx.gate, "acceptance.prompt", None)
    prompt = str(prompt_override or DEFAULT_PROMPT)
    return {
        "prompt_id": "site-override" if prompt_override else DEFAULT_PROMPT_ID,
        "prompt_sha256": sha256_hex(prompt.encode("utf-8")),
        "model": served_model_name(ctx),
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "seed": int(dig(ctx.gate, "acceptance.seed", DEFAULT_SEED)),
        "max_tokens": int(dig(ctx.gate, "acceptance.max_tokens", DEFAULT_MAX_TOKENS)),
    }


def expected_generation_path(ctx: GateContext) -> Path | None:
    configured = dig(ctx.gate, "acceptance.expected_generation_path", None)
    return ctx.repo_path(str(configured)) if configured else None


def plan_deterministic_generation(ctx: GateContext) -> list[dict]:
    base = ctx.primary_api_base()
    expected = expected_generation_path(ctx)
    return [
        {
            "kind": "http",
            "what": "greedy fixed-prompt completion",
            "url": f"{base}/v1/completions",
            "params": generation_params(ctx),
        },
        {
            "kind": "http",
            "what": "recover output token ids",
            "url": f"{base}/tokenize",
        },
        {
            "kind": "local-check",
            "what": "compare the token-id sha256 against the expected-value file",
            "expected_output_file": str(expected) if expected else None,
            "on_missing": (
                "record the observed ids as a candidate baseline and report "
                "BASELINE-RECORDED (never PASS)"
            ),
        },
    ]


def run_deterministic_generation(ctx: GateContext) -> StageOutcome:
    base = ctx.primary_api_base()
    prompt = str(dig(ctx.gate, "acceptance.prompt", None) or DEFAULT_PROMPT)
    params = generation_params(ctx)
    artifacts: list[str] = []

    payload = {
        "model": params["model"],
        "prompt": prompt,
        "max_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "n": params["n"],
        "seed": params["seed"],
        "stream": False,
    }
    status, body = ctx.http.post_json(f"{base}/v1/completions", payload, timeout=1800.0)
    artifacts.append(
        ctx.artifact_json(
            "completion.json", {"request": payload, "status": status, "response": body}
        )
    )
    if status != 200 or not isinstance(body, dict):
        raise StageFailure(
            f"/v1/completions returned {status}; deterministic generation could "
            "not be measured",
            details={"status": status},
            artifacts=artifacts,
        )
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise StageFailure("/v1/completions returned no choices", artifacts=artifacts)
    text = str(choices[0].get("text") or "")

    token_status, token_body = ctx.http.post_json(
        f"{base}/tokenize",
        {"model": params["model"], "prompt": text, "add_special_tokens": False},
        timeout=300.0,
    )
    artifacts.append(
        ctx.artifact_json("tokenize.json", {"status": token_status, "response": token_body})
    )
    token_ids = None
    if token_status == 200 and isinstance(token_body, dict):
        candidate = token_body.get("tokens")
        if isinstance(candidate, list) and all(isinstance(v, int) for v in candidate):
            token_ids = list(candidate)
    if token_ids is None:
        raise StageFailure(
            f"could not recover output token ids from {base}/tokenize "
            f"(status {token_status}). The gate deliberately does NOT fall back "
            "to hashing raw text: a text hash is not a token-id hash and would "
            f"silently weaken the acceptance criterion ({TARGET_DOC} TBD-11).",
            details={"tokenize_status": token_status},
            artifacts=artifacts,
        )

    token_ids_sha256 = sha256_hex(canonical_json(token_ids).encode("utf-8"))
    observed = {
        "schema": EXPECTED_SCHEMA,
        "recorded_at": utc_now_iso(),
        "params": params,
        "prompt": prompt,
        "token_ids": token_ids,
        "token_ids_sha256": token_ids_sha256,
        "completion_text": text,
        "serving": {
            "mtp_mode": dig(ctx.site, "serving.mtp_mode", None),
            "mtp_tokens": dig(ctx.site, "serving.mtp_tokens", None),
            "max_model_len": dig(ctx.site, "serving.max_model_len", None),
        },
    }
    artifacts.append(ctx.artifact_json("observed-generation.json", observed))

    expected_path = expected_generation_path(ctx)
    if expected_path is None or not expected_path.is_file():
        baseline_rel = "expected/generation-baseline.json"
        if ctx.bundle.root is not None:
            target = ctx.bundle.root / baseline_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(observed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            artifacts.append(baseline_rel)
        where = str(expected_path) if expected_path else "(unset)"
        return StageOutcome(
            status=STATUS_BASELINE_RECORDED,
            message=(
                f"no expected-output file at {where}: recorded {len(token_ids)} "
                f"output token ids (sha256 {token_ids_sha256}) as a CANDIDATE "
                f"baseline at {baseline_rel}. This is NOT a pass. Review the "
                "completion, commit the baseline to the path in "
                "acceptance.expected_generation_path, and re-run."
            ),
            details={
                "token_count": len(token_ids),
                "token_ids_sha256": token_ids_sha256,
                "expected_output_file": where,
                "baseline_written_to": baseline_rel,
            },
            artifacts=artifacts,
        )

    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise StageFailure(
            f"expected-output file {expected_path} is unreadable: {exc}",
            artifacts=artifacts,
        ) from exc

    expected_params = expected.get("params", {})
    drifted = {
        key: (expected_params.get(key), params.get(key))
        for key in (
            "model",
            "temperature",
            "top_p",
            "seed",
            "max_tokens",
            "prompt_sha256",
        )
        if expected_params.get(key) != params.get(key)
    }
    if drifted:
        raise StageFailure(
            "generation parameters drifted from the expected-output file, so "
            f"the comparison is invalid: {drifted}. Either restore the "
            "parameters or record a new baseline deliberately.",
            details={"drift": drifted},
            artifacts=artifacts,
        )

    expected_sha = str(expected.get("token_ids_sha256") or "")
    if expected_sha != token_ids_sha256:
        expected_ids = expected.get("token_ids") or []
        divergence = next(
            (i for i, (a, b) in enumerate(zip(expected_ids, token_ids)) if a != b),
            min(len(expected_ids), len(token_ids)),
        )
        raise StageFailure(
            "deterministic generation mismatch: token-id sha256 "
            f"{token_ids_sha256} != expected {expected_sha} (first divergence "
            f"at index {divergence}; expected {len(expected_ids)} ids, observed "
            f"{len(token_ids)})",
            details={
                "expected_token_ids_sha256": expected_sha,
                "observed_token_ids_sha256": token_ids_sha256,
                "first_divergence_index": divergence,
                "expected_token_count": len(expected_ids),
                "observed_token_count": len(token_ids),
            },
            artifacts=artifacts,
        )

    return StageOutcome(
        status=STATUS_PASS,
        message=(
            f"{len(token_ids)} output token ids match the expected-value file "
            f"(sha256 {token_ids_sha256})"
        ),
        details={
            "token_count": len(token_ids),
            "token_ids_sha256": token_ids_sha256,
            "expected_output_file": str(expected_path),
        },
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Stage 6 - profile performance matrix (reported, never merged into functional)
# ---------------------------------------------------------------------------


def performance_cells(ctx: GateContext) -> list[dict]:
    configured = dig(ctx.gate, "performance.cells", None)
    if configured:
        return [dict(cell) for cell in configured]
    return [
        {"concurrency": c, "max_tokens": DEFAULT_BENCH_MAX_TOKENS}
        for c in dig(ctx.gate, "matrix.required_concurrencies", REQUIRED_CONCURRENCIES)
    ]


def plan_performance_matrix(ctx: GateContext) -> list[dict]:
    base = ctx.primary_api_base()
    band = dig(ctx.gate, "performance.band", None)
    actions = [
        {
            "kind": "http",
            "what": (
                f"C{cell['concurrency']} cell "
                f"({cell['concurrency']} concurrent streams)"
            ),
            "url": f"{base}/v1/completions",
            "max_tokens": cell.get("max_tokens", DEFAULT_BENCH_MAX_TOKENS),
            "repetitions": cell.get("repetitions", 1),
            "ignore_eos": cell.get("ignore_eos", False),
        }
        for cell in performance_cells(ctx)
    ]
    actions.append(
        {
            "kind": "local-check",
            "what": "compare against the documented tolerance band",
            "band_configured": bool(band),
            "on_missing_band": (
                "record the observed numbers as a candidate band and report "
                "performance BASELINE-RECORDED; reference-lane numbers are "
                "never used as a band"
            ),
        }
    )
    return actions


def _run_cell(ctx: GateContext, cell: dict) -> dict:
    base = ctx.primary_api_base()
    concurrency = max(1, int(cell["concurrency"]))
    max_tokens = int(cell.get("max_tokens", DEFAULT_BENCH_MAX_TOKENS))
    repetitions = int(cell.get("repetitions", 1))
    minimum_tokens = int(cell.get("minimum_tokens", 1))
    prompt = str(cell.get("prompt") or DEFAULT_PROMPT)
    payload = {
        "model": served_model_name(ctx),
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": int(cell.get("seed", DEFAULT_SEED)),
        "ignore_eos": bool(cell.get("ignore_eos", False)),
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
    }
    url = f"{base}/v1/completions"
    started = time.monotonic()
    indexed_samples = []
    for repetition in range(repetitions):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(ctx.http.stream_completion, url, dict(payload), 1800.0)
                for _ in range(concurrency)
            ]
            indexed_samples.extend(
                (repetition, stream, future.result())
                for stream, future in enumerate(futures)
            )
    wall = time.monotonic() - started
    samples = [sample for _, _, sample in indexed_samples]

    errors = [s.error for s in samples if s.error]
    short_streams = [
        {"repetition": repetition, "stream": stream, "tokens": sample.tokens}
        for repetition, stream, sample in indexed_samples
        if not sample.error and sample.tokens < minimum_tokens
    ]
    tokens_total = sum(s.tokens for s in samples)
    ttfts = [s.ttft_seconds for s in samples if s.ttft_seconds is not None]
    per_stream = [
        s.tokens / s.total_seconds for s in samples if s.total_seconds and s.total_seconds > 0
    ]
    return {
        "cell": f"C{concurrency}",
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "repetitions": repetitions,
        "minimum_tokens": minimum_tokens,
        "ignore_eos": payload["ignore_eos"],
        "streams": [
            {
                "repetition": repetition,
                "stream": stream,
                "ttft_seconds": s.ttft_seconds,
                "total_seconds": s.total_seconds,
                "tokens": s.tokens,
                "token_count_source": s.token_count_source,
                "error": s.error,
            }
            for repetition, stream, s in indexed_samples
        ],
        "errors": errors,
        "short_streams": short_streams,
        "tokens_total": tokens_total,
        "cell_wall_seconds": wall,
        "aggregate_tokens_per_second": (tokens_total / wall) if wall > 0 else None,
        "per_stream_tokens_per_second_p50": percentile(per_stream, 50),
        "ttft_seconds_p50": percentile(ttfts, 50),
        "ttft_seconds_p99": percentile(ttfts, 99),
    }


def run_performance_matrix(ctx: GateContext) -> StageOutcome:
    artifacts: list[str] = []
    observed: dict[str, dict] = {}
    request_errors: list[str] = []

    for cell in performance_cells(ctx):
        raw = _run_cell(ctx, cell)
        label = raw["cell"]
        observed[label] = raw
        artifacts.append(ctx.artifact_json(f"cell-{label}.json", raw))
        if raw["errors"]:
            request_errors.append(f"{label}: {len(raw['errors'])} stream error(s)")
        if raw["short_streams"]:
            request_errors.append(
                f"{label}: {len(raw['short_streams'])} stream(s) returned fewer "
                f"than {raw['minimum_tokens']} tokens"
            )

    artifacts.append(ctx.artifact_json("cells.json", observed))

    if request_errors:
        # Failed requests are a functional problem that happens to surface in
        # the performance stage; they are NOT a performance miss. A miss is
        # PERFORMANCE-OUT-OF-BAND (below) and never aborts.
        raise StageFailure(
            "benchmark requests failed: " + " | ".join(request_errors),
            details={"cells": {k: v["errors"] for k, v in observed.items()}},
            artifacts=artifacts,
        )

    band = dig(ctx.gate, "performance.band", None)
    if not band:
        candidate = {
            label: {
                metric: raw.get(metric)
                for metric in (
                    "aggregate_tokens_per_second",
                    "per_stream_tokens_per_second_p50",
                    "ttft_seconds_p50",
                    "ttft_seconds_p99",
                )
            }
            for label, raw in observed.items()
        }
        artifacts.append(ctx.artifact_json("candidate-band.json", candidate))
        return StageOutcome(
            status=STATUS_BASELINE_RECORDED,
            message=(
                "no performance band is configured, so the observed profile "
                "numbers were recorded as a CANDIDATE band. No public-lane band "
                f"exists yet ({TARGET_DOC} TBD-10) and reference-lane numbers "
                "must never be used as one."
            ),
            details={"observed": candidate, "band_configured": False},
            artifacts=artifacts,
        )

    comparisons, violations = evaluate_band(observed, band)
    artifacts.append(
        ctx.artifact_json("band-comparison.json", {"band": band, "comparisons": comparisons})
    )
    if violations:
        return StageOutcome(
            status=STATUS_PERFORMANCE_OUT_OF_BAND,
            message=(
                "performance outside the documented band: "
                + " | ".join(violations)
                + ". This is a PERFORMANCE result, not a functional failure."
            ),
            details={"comparisons": comparisons, "violations": violations},
            artifacts=artifacts,
        )
    return StageOutcome(
        status=STATUS_PASS,
        message=f"all {len(comparisons)} banded metrics are in band",
        details={"comparisons": comparisons},
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Stage 7 - optional profile-specific acceptance extensions
# ---------------------------------------------------------------------------


def profile_extensions(ctx: GateContext) -> list[dict]:
    configured = dig(ctx.gate, "extensions.pre_shutdown", []) or []
    return list(configured) if isinstance(configured, list) else []


def plan_profile_extensions(ctx: GateContext) -> list[dict]:
    actions = []
    for index, extension in enumerate(profile_extensions(ctx)):
        actions.append(
            {
                "kind": "command",
                "what": extension.get("title") or extension["id"],
                "id": extension["id"],
                "argv": as_argv(
                    extension["command"],
                    f"extensions.pre_shutdown[{index}].command",
                ),
                "timeout_seconds": extension.get("timeout_seconds", 3600),
                "mutates_remote": extension.get("mutates_remote", False),
                "require_json_status": extension.get(
                    "require_json_status", False
                ),
            }
        )
    return actions or [
        {
            "kind": "local-check",
            "what": "no profile-specific acceptance extensions configured",
            "mutates_remote": False,
        }
    ]


def run_profile_extensions(ctx: GateContext) -> StageOutcome:
    configured = profile_extensions(ctx)
    if not configured:
        return StageOutcome(
            status=STATUS_PASS,
            message="no profile-specific acceptance extensions configured",
            details={"extensions": []},
        )
    artifacts: list[str] = []
    observed: list[dict] = []
    failures: list[str] = []
    baselines: list[str] = []
    for index, extension in enumerate(configured):
        identifier = extension["id"]
        result = ctx.executor.run(
            as_argv(
                extension["command"],
                f"extensions.pre_shutdown[{index}].command",
            ),
            timeout=float(extension.get("timeout_seconds", 3600)),
        )
        artifacts.append(
            ctx.capture_command(f"extension-{identifier}.json", result)
        )
        parsed = None
        try:
            candidate = json.loads(result.stdout)
            if isinstance(candidate, dict):
                parsed = candidate
        except json.JSONDecodeError:
            parsed = None
        entry = {
            "id": identifier,
            "exit_code": result.exit_code,
            "mutates_remote": extension.get("mutates_remote", False),
            "require_json_status": extension.get("require_json_status", False),
            "report": parsed,
        }
        observed.append(entry)
        if extension.get("require_json_status", False) and parsed is None:
            failures.append(f"{identifier} emitted no JSON status report")
        elif (
            result.exit_code == EXIT_BASELINE_RECORDED
            and parsed is not None
            and parsed.get("status") == "baseline-recorded"
        ):
            baselines.append(identifier)
        elif result.exit_code != 0:
            failures.append(f"{identifier} exited {result.exit_code}")
        elif parsed is not None and parsed.get("status") not in (None, "pass"):
            failures.append(
                f"{identifier} reported status={parsed.get('status')!r}"
            )
    if failures:
        raise StageFailure(
            "profile-specific acceptance extension failed: "
            + " | ".join(failures),
            details={"extensions": observed, "failures": failures},
            artifacts=artifacts,
        )
    if baselines:
        return StageOutcome(
            status=STATUS_BASELINE_RECORDED,
            message=(
                "profile-specific baseline recorded for review: "
                + ", ".join(baselines)
                + "; this is not a pass"
            ),
            details={"extensions": observed, "baselines": baselines},
            artifacts=artifacts,
        )
    return StageOutcome(
        status=STATUS_PASS,
        message=f"{len(observed)} profile-specific acceptance extension(s) passed",
        details={"extensions": observed},
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Stage 8 - shutdown / rollback verification
# ---------------------------------------------------------------------------


def plan_shutdown_rollback(ctx: GateContext) -> list[dict]:
    actions: list[dict] = []
    status_command = dig(ctx.gate, "launch.rank_status_command", []) or []
    if status_command:
        for rank in ctx.ranks():
            actions.append(
                {
                    "kind": "command",
                    "what": (
                        f"capture final restart/OOM/resource status on rank "
                        f"{rank['id']} before stop"
                    ),
                    "argv": ctx.rank_argv(
                        rank,
                        as_argv(
                            status_command,
                            "launch.rank_status_command",
                        ),
                    ),
                    "mutates_remote": False,
                }
            )
    actions.append(
        {
            "kind": "command",
            "what": "stop all four ranks (site launcher)",
            "argv": as_argv(dig(ctx.gate, "launch.stop_command"), "launch.stop_command"),
            "mutates_remote": True,
        }
    )
    for rank_id, base in sorted(ctx.api_bases().items()):
        actions.append(
            {
                "kind": "http",
                "what": f"confirm rank {rank_id} API stopped answering",
                "url": f"{base}/health",
            }
        )
    rollback = dig(ctx.gate, "launch.rollback_verify_command", []) or []
    if rollback:
        actions.append(
            {
                "kind": "command",
                "what": "verify rollback",
                "argv": as_argv(rollback, "launch.rollback_verify_command"),
            }
        )
    return actions


def run_shutdown_rollback(ctx: GateContext) -> StageOutcome:
    artifacts: list[str] = []
    failures: list[str] = []
    timeout = float(dig(ctx.gate, "launch.stop_timeout_seconds", 900))

    status_command = dig(ctx.gate, "launch.rank_status_command", []) or []
    if status_command:
        for rank in ctx.ranks():
            number = int(rank["id"])
            status = ctx.executor.run(
                ctx.rank_argv(
                    rank,
                    as_argv(status_command, "launch.rank_status_command"),
                ),
                timeout=timeout,
            )
            artifacts.append(
                ctx.capture_command(f"rank{number}-pre-stop-status.json", status)
            )
            if status.exit_code != 0:
                failures.append(
                    f"rank {number} final rank_status_command exited "
                    f"{status.exit_code}"
                )

    stop = ctx.executor.run(
        as_argv(dig(ctx.gate, "launch.stop_command"), "launch.stop_command"),
        timeout=timeout,
    )
    artifacts.append(ctx.capture_command("launch-stop.json", stop))
    if stop.exit_code != 0:
        failures.append(f"stop_command exited {stop.exit_code}")

    still_up: list[str] = []
    for rank_id, base in sorted(ctx.api_bases().items()):
        status, body = ctx.http.get_json(f"{base}/health", timeout=30.0)
        artifacts.append(
            ctx.artifact_json(
                f"rank{rank_id}-post-stop-health.json",
                {"status": status, "body": body},
            )
        )
        if status == 200:
            still_up.append(f"rank {rank_id}")
    if still_up:
        failures.append("API still answering after stop on " + ", ".join(still_up))

    rollback = dig(ctx.gate, "launch.rollback_verify_command", []) or []
    if rollback:
        result = ctx.executor.run(
            as_argv(rollback, "launch.rollback_verify_command"), timeout=timeout
        )
        artifacts.append(ctx.capture_command("rollback-verify.json", result))
        if result.exit_code != 0:
            failures.append(f"rollback_verify_command exited {result.exit_code}")
    else:
        ctx.notes.append(
            "no launch.rollback_verify_command configured; rollback was not "
            "independently verified"
        )

    if failures:
        raise StageFailure(
            "shutdown/rollback verification failed: " + " | ".join(failures),
            details={"failures": failures},
            artifacts=artifacts,
        )
    return StageOutcome(
        status=STATUS_PASS,
        message="stack stopped, API no longer answering, rollback verified",
        details={"rollback_verified": bool(rollback)},
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    id: str
    order: int
    title: str
    purpose: str
    plan: Callable[[GateContext], list[dict]]
    run: Callable[[GateContext], StageOutcome]


STAGES: tuple[Stage, ...] = (
    Stage(
        id="runtime_attestation",
        order=1,
        title="Runtime image and artifact attestation",
        purpose=(
            "every rank runs the same attested image, artifacts and model pin "
            "(delegates to runtime/verify-runtime.py and scripts/preflight.py)"
        ),
        plan=plan_runtime_attestation,
        run=run_runtime_attestation,
    ),
    Stage(
        id="fabric_transport_qualification",
        order=2,
        title="Fabric qualification and model-down transport probe",
        purpose=(
            "all four cable edges qualify and the four-rank collectives are "
            "byte-correct before any model is loaded"
        ),
        plan=plan_fabric_transport,
        run=run_fabric_transport,
    ),
    Stage(
        id="rank_startup",
        order=3,
        title="All-four-rank startup",
        purpose="the site launcher brings up all four ranks",
        plan=plan_rank_startup,
        run=run_rank_startup,
    ),
    Stage(
        id="api_liveness",
        order=4,
        title="API liveness across ranks",
        purpose="/health and /v1/models answer on every API-serving rank",
        plan=plan_api_liveness,
        run=run_api_liveness,
    ),
    Stage(
        id="deterministic_generation",
        order=5,
        title="Deterministic fixed-prompt generation",
        purpose=(
            "greedy, seeded, fixed-budget generation reproduces the expected "
            "output token ids by SHA-256"
        ),
        plan=plan_deterministic_generation,
        run=run_deterministic_generation,
    ),
    Stage(
        id="performance_matrix",
        order=6,
        title="Profile performance matrix",
        purpose=(
            "measured throughput and TTFT reported against the documented "
            "tolerance band; never merged into the functional verdict"
        ),
        plan=plan_performance_matrix,
        run=run_performance_matrix,
    ),
    Stage(
        id="profile_extensions",
        order=7,
        title="Profile-specific acceptance extensions",
        purpose=(
            "optional model/runtime-specific gates run after the common API "
            "and performance checks and before guaranteed cleanup"
        ),
        plan=plan_profile_extensions,
        run=run_profile_extensions,
    ),
    Stage(
        id="shutdown_rollback",
        order=8,
        title="Shutdown and rollback verification",
        purpose="the stack stops cleanly and rollback is verified",
        plan=plan_shutdown_rollback,
        run=run_shutdown_rollback,
    ),
)

FUNCTIONAL_STAGE_IDS = tuple(
    stage.id for stage in STAGES if stage.id != "performance_matrix"
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _git_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _safe_file_sha256(path: Path) -> str:
    try:
        return sha256_file(path)
    except OSError:
        return "unknown"


def environment_fingerprint(ctx: GateContext) -> dict:
    """Non-identifying run fingerprint: no addresses, users, or hostnames."""
    lock = ctx.lock
    model = lock.get("model") if isinstance(lock.get("model"), dict) else {}
    toolchain = lock.get("toolchain") if isinstance(lock.get("toolchain"), dict) else {}
    serving = dig(ctx.site, "serving", {})
    topology = dig(ctx.site, "topology", {})
    return {
        "gate": {
            "version": GATE_VERSION,
            "schema": RESULT_SCHEMA,
            "target_document": TARGET_DOC,
        },
        "gate_host": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "repo": {
            "git_commit": _git_commit(ctx.repo_root),
            "runtime_lock_sha256": _safe_file_sha256(
                ctx.repo_path(str(dig(ctx.gate, "runtime.lock_path")))
            ),
        },
        "runtime_lock": {
            "runtime_id": lock.get("runtime_id"),
            "vllm_commit": (lock.get("vllm") or {}).get("commit"),
            "nccl_tag": (lock.get("nccl") or {}).get("tag"),
            "sparkinfer_commit": (lock.get("sparkinfer") or {}).get("commit"),
            "flashinfer_commit": (lock.get("flashinfer") or {}).get("commit"),
            "cuda_version": toolchain.get("cuda_version"),
            "torch_version": toolchain.get("torch_version"),
            "python_version": toolchain.get("python_version"),
            "target_platform": toolchain.get("target_platform"),
        },
        "model": {
            "repository": model.get("repository"),
            "revision": model.get("revision"),
            "config_sha256": model.get("config_sha256"),
        },
        "serving": {
            "tensor_parallel_size": serving.get("tensor_parallel_size"),
            "decode_context_parallel_size": serving.get("decode_context_parallel_size"),
            "mtp_mode": serving.get("mtp_mode"),
            "mtp_tokens": serving.get("mtp_tokens"),
            "max_model_len": serving.get("max_model_len"),
            "kv_cache_bytes_per_rank": serving.get("kv_cache_bytes_per_rank"),
            "max_num_seqs": serving.get("max_num_seqs"),
        },
        "topology": {
            "ranks": len(dig(ctx.site, "ranks", [])),
            "edges": len(topology.get("edges", [])),
            "mtu": topology.get("mtu"),
            "link_speed_mbps": topology.get("link_speed_mbps"),
        },
        "config": {
            "site_schema_version": ctx.site.get("schema_version"),
            "site_config_sha256": ctx.site_sha256,
            "gate_config_sha256": ctx.gate_config_sha256,
            "placeholder_warnings": len(ctx.placeholder_warnings),
        },
    }


def build_plan(ctx: GateContext) -> dict:
    return {
        "schema": PLAN_SCHEMA,
        "gate_version": GATE_VERSION,
        "generated_at": utc_now_iso(),
        "mode": "dry-run" if ctx.args.dry_run else "execute",
        "mutates_remote": True,
        "safety_class": "STOPS SERVING",
        "target_document": TARGET_DOC,
        "environment_fingerprint": environment_fingerprint(ctx),
        "placeholder_warnings": list(ctx.placeholder_warnings),
        "stages": [
            {
                "id": stage.id,
                "order": stage.order,
                "title": stage.title,
                "purpose": stage.purpose,
                "actions": stage.plan(ctx),
            }
            for stage in STAGES
        ],
    }


def _skipped(stage: Stage, reason: str, now: str) -> StageResult:
    return StageResult(
        id=stage.id,
        order=stage.order,
        title=stage.title,
        status=STATUS_SKIPPED,
        started_at=now,
        finished_at=now,
        duration_seconds=0.0,
        message=reason,
        artifacts=[],
        details={},
    )


def compute_verdicts(stage_results: Sequence[StageResult]) -> tuple[str, str, int]:
    by_id = {result.id: result for result in stage_results}
    saw_failure = any(result.status == STATUS_FAIL for result in stage_results)

    functional = FUNCTIONAL_PASS
    for stage_id in FUNCTIONAL_STAGE_IDS:
        result = by_id.get(stage_id)
        if result is None or result.status == STATUS_SKIPPED:
            functional = FUNCTIONAL_FAIL if saw_failure else FUNCTIONAL_NOT_RUN
            break
        if result.status == STATUS_FAIL:
            functional = FUNCTIONAL_FAIL
            break
        if result.status == STATUS_BASELINE_RECORDED:
            functional = FUNCTIONAL_BASELINE_RECORDED

    performance_result = by_id.get("performance_matrix")
    if performance_result is None or performance_result.status in (
        STATUS_SKIPPED,
        STATUS_FAIL,
    ):
        performance = PERFORMANCE_NOT_MEASURED
    elif performance_result.status == STATUS_PERFORMANCE_OUT_OF_BAND:
        performance = PERFORMANCE_OUT_OF_BAND
    elif performance_result.status == STATUS_BASELINE_RECORDED:
        performance = PERFORMANCE_BASELINE_RECORDED
    else:
        performance = PERFORMANCE_IN_BAND

    if functional in (FUNCTIONAL_FAIL, FUNCTIONAL_NOT_RUN):
        exit_code = EXIT_FUNCTIONAL_FAIL
    elif functional == FUNCTIONAL_BASELINE_RECORDED:
        exit_code = EXIT_BASELINE_RECORDED
    elif performance in (PERFORMANCE_OUT_OF_BAND, PERFORMANCE_BASELINE_RECORDED):
        exit_code = EXIT_PERFORMANCE_NOT_IN_BAND
    else:
        exit_code = EXIT_OK
    return functional, performance, exit_code


def execute_stages(ctx: GateContext) -> list[StageResult]:
    results: list[StageResult] = []
    aborted_after: str | None = None
    stack_start_attempted = False

    for stage in STAGES:
        cleanup_after_failure = (
            aborted_after is not None
            and stack_start_attempted
            and stage.id == "shutdown_rollback"
        )
        if aborted_after is not None and not cleanup_after_failure:
            results.append(_skipped(stage, f"aborted-after: {aborted_after}", utc_now_iso()))
            continue
        ctx.stage_order = stage.order
        ctx.stage_id = stage.id
        started_at = utc_now_iso()
        started = time.monotonic()
        if stage.id == "rank_startup":
            stack_start_attempted = True
        try:
            outcome = stage.run(ctx)
            status, message = outcome.status, outcome.message
            details, artifacts = outcome.details, outcome.artifacts
        except StageFailure as failure:
            status = STATUS_FAIL
            message = failure.message
            details = failure.details
            artifacts = failure.artifacts
        except Exception as exc:  # noqa: BLE001 - any error is a stage failure
            status = STATUS_FAIL
            message = f"unexpected error in stage {stage.id}: {exc!r}"
            details = {"exception": repr(exc)}
            artifacts = []
        results.append(
            StageResult(
                id=stage.id,
                order=stage.order,
                title=stage.title,
                status=status,
                started_at=started_at,
                finished_at=utc_now_iso(),
                duration_seconds=round(time.monotonic() - started, 6),
                message=message,
                artifacts=list(artifacts),
                details=details,
            )
        )
        print(
            f"acceptance-gate: [{status}] {stage.order}/{len(STAGES)} "
            f"{stage.id}: {message}",
            file=sys.stderr if status == STATUS_FAIL else sys.stdout,
        )
        if status == STATUS_FAIL:
            aborted_after = stage.id

    return results


def build_result(
    ctx: GateContext, stage_results: Sequence[StageResult], started_at: str
) -> dict:
    functional, performance, exit_code = compute_verdicts(stage_results)
    return {
        "schema": RESULT_SCHEMA,
        "gate_version": GATE_VERSION,
        "target_document": TARGET_DOC,
        "run_id": ctx.args.run_id,
        "mode": "execute",
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "environment_fingerprint": environment_fingerprint(ctx),
        "stages": [result.as_dict() for result in stage_results],
        "functional_verdict": functional,
        "performance_verdict": performance,
        "exit_code": exit_code,
        "notes": list(ctx.notes),
        "placeholder_warnings": list(ctx.placeholder_warnings),
        "disclaimer": (
            "Functional acceptance and performance are separate results. "
            "Reference-lane historical throughput numbers are never public-lane "
            "results; see docs/PUBLIC_FUNCTIONAL_TARGET.md section 4.3."
        ),
    }


def run_gate(args: argparse.Namespace, executor: Any = None, http: Any = None) -> int:
    site_path = Path(args.site).resolve()
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parent.parent
    )
    site, site_sha, placeholder_warnings = load_site_config(site_path)
    gate_config, gate_sha = load_gate_config(
        Path(args.gate_config) if args.gate_config else None
    )
    placeholder_warnings.extend(gate_placeholder_warnings(gate_config))

    lock_path = Path(str(dig(gate_config, "runtime.lock_path")))
    if not lock_path.is_absolute():
        lock_path = repo_root / lock_path
    lock = load_runtime_lock(lock_path)

    problems = validate_configuration(site, gate_config, lock)
    if problems:
        raise GateConfigError(
            "the configuration does not describe the supported matrix:\n  - "
            + "\n  - ".join(problems)
        )

    ctx = GateContext(
        site=site,
        gate=gate_config,
        site_path=site_path,
        site_sha256=site_sha,
        gate_config_sha256=gate_sha,
        repo_root=repo_root,
        lock=lock,
        executor=RefusingExecutor(),
        http=RefusingExecutor(),
        bundle=Bundle(None),
        args=args,
        placeholder_warnings=list(placeholder_warnings),
    )

    if args.dry_run:
        plan = build_plan(ctx)
        text = json.dumps(plan, indent=2, sort_keys=True)
        if args.plan_out:
            Path(args.plan_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.plan_out).write_text(text + "\n", encoding="utf-8")
            print(f"acceptance-gate: plan written to {args.plan_out}")
        else:
            print(text)
        if placeholder_warnings:
            print(
                "acceptance-gate: WARNING: the site or gate config contains "
                f"{len(placeholder_warnings)} example placeholder(s); --execute "
                "will refuse until they are replaced.",
                file=sys.stderr,
            )
        print(
            "acceptance-gate: DRY RUN -- configuration validated and plan "
            "produced. Nothing was executed, nothing was written to the "
            "cluster, no bundle was created. Re-run with --execute --confirm "
            f"{CONFIRM_TOKEN} to run it for real.",
            file=sys.stderr,
        )
        return EXIT_OK

    if args.confirm != CONFIRM_TOKEN:
        raise GateConfigError(
            f"--execute requires --confirm {CONFIRM_TOKEN}. This gate starts and "
            "stops a serving stack; never point it at a cluster serving "
            "production traffic."
        )
    if placeholder_warnings:
        raise GateConfigError(
            "the site or gate config still contains example placeholders, so "
            "it does not describe an executable acceptance plan:\n  - "
            + "\n  - ".join(placeholder_warnings[:10])
            + ("\n  - ..." if len(placeholder_warnings) > 10 else "")
        )

    bundle_root = Path(args.bundle_dir) / args.run_id
    ctx.bundle = Bundle(bundle_root)
    ctx.executor = executor if executor is not None else SubprocessExecutor()
    ctx.http = http if http is not None else UrllibHttpClient()

    started_at = utc_now_iso()
    ctx.bundle.write_root_json("plan.json", build_plan(ctx))
    stage_results = execute_stages(ctx)
    result = build_result(ctx, stage_results, started_at)
    ctx.bundle.write_root_json("result.json", result)

    print(
        "acceptance-gate: functional_verdict="
        f"{result['functional_verdict']} performance_verdict="
        f"{result['performance_verdict']} exit={result['exit_code']} "
        f"bundle={bundle_root}"
    )
    return int(result["exit_code"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acceptance_gate",
        description=(
            "SparkRing public-functional acceptance gate. Dry-run by default; "
            "--execute starts and stops a serving stack."
        ),
    )
    parser.add_argument(
        "--site", required=True, help="site config (scripts/sparkring_site.py schema)"
    )
    parser.add_argument(
        "--gate-config", default=None, help="acceptance gate config JSON"
    )
    parser.add_argument(
        "--preflight", default=None, help="scripts/preflight.py evidence JSON"
    )
    parser.add_argument(
        "--repo-root", default=None, help="repository root (default: parent of scripts/)"
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="validate config and print the plan; touch nothing (DEFAULT)",
    )
    parser.add_argument(
        "--execute",
        dest="dry_run",
        action="store_false",
        help=f"actually run the gate (requires --confirm {CONFIRM_TOKEN})",
    )
    parser.add_argument(
        "--confirm", default="", help=f"confirmation token for --execute: {CONFIRM_TOKEN}"
    )
    parser.add_argument(
        "--bundle-dir",
        default="evidence/acceptance",
        help="parent directory for the evidence bundle (execute mode only)",
    )
    parser.add_argument(
        "--run-id", default=None, help="bundle name (default: acceptance-<UTC stamp>)"
    )
    parser.add_argument(
        "--plan-out", default=None, help="dry-run only: write the plan JSON here"
    )
    return parser


def main(
    argv: Sequence[str] | None = None, executor: Any = None, http: Any = None
) -> int:
    args = build_parser().parse_args(argv)
    if not args.run_id:
        args.run_id = "acceptance-" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
    try:
        return run_gate(args, executor=executor, http=http)
    except GateConfigError as exc:
        print(f"acceptance-gate: CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
