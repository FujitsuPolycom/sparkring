#!/usr/bin/env python3
"""Multi-service runtime bundle primitives for SparkRing.

A versioned, fail-closed static bundle schema
(``sparkring-runtime-bundle/v1``) representing one serving/model
service plus zero or more cache/sidecar services with deterministic
dependency ordering, structured readiness probes, reverse-order
rollback, and semantic offline validation/explanation/diff/planning.

The bundle module reuses ``sparkring_runtime.RemoteAction`` and the
canonical family builders. It does not reimplement model hashing,
image checks, privilege flags, connector JSON, health loops, readiness
loops, labels, names, timeouts, or rollback guards.

Safety:
* ``plan`` is always offline and prints a deterministic JSON document.
* ``start``/``stop``/``rollback`` require ``--execute`` and a
  confirmation token.
* ``status``/``verify-rollback`` are READ-ONLY and require
  ``--execute`` but not a confirmation token.
* Native stop/rollback actions are five-label-guarded so a foreign
  same-named container is never removed.
* The EXL3+LMCache bridge is plan-only (``execution_supported: false``).
* Source paths are confined to the resolved bundle directory.
* Shell entrypoints (sh/bash/pwsh/cmd) are rejected for structured
  containers; ``argv[0]`` is emitted as Docker ``--entrypoint`` so
  the image's inherited ENTRYPOINT cannot override it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(_ROOT / "scripts"))

import sparkring_runtime as runtime  # noqa: E402
import sparkring_generic_launcher as generic  # noqa: E402


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

BUNDLE_SCHEMA = "sparkring-runtime-bundle/v1"
BUNDLE_PLAN_SCHEMA = "sparkring-runtime-bundle-plan/v1"

MAX_SERVICES = 16

ROLE_SERVING = "serving"
ROLE_CACHE = "cache"
ROLE_SIDECAR = "sidecar"
VALID_ROLES = frozenset({ROLE_SERVING, ROLE_CACHE, ROLE_SIDECAR})

# Source kinds — closed enum, no mixing across families.
SOURCE_RUNTIME_PROFILE = "runtime-profile"
SOURCE_STRUCTURED_CONTAINER = "structured-container"
SOURCE_EXL3_LMCACHE = "canonical-exl3-lmcache-cs512"
VALID_SOURCE_KINDS = frozenset({
    SOURCE_RUNTIME_PROFILE, SOURCE_STRUCTURED_CONTAINER, SOURCE_EXL3_LMCACHE,
})

# Only the EXL3+LMCache canonical bridge is plan-only.
PLAN_ONLY_SOURCE_KINDS = frozenset({SOURCE_EXL3_LMCACHE})

READINESS_CONTAINER_RUNNING = "container-running"
READINESS_HTTP_GET = "http-get"
VALID_READINESS_KINDS = frozenset({READINESS_CONTAINER_RUNNING, READINESS_HTTP_GET})

READINESS_RANK_SCOPE_ALL = "all"
READINESS_RANK_SCOPE_RANK0 = "rank0"
VALID_RANK_SCOPES = frozenset({READINESS_RANK_SCOPE_ALL, READINESS_RANK_SCOPE_RANK0})

READINESS_PORT_SITE_API = "site-api"
VALID_PORT_REFERENCES = frozenset({READINESS_PORT_SITE_API})

_PATH_CHARS = re.compile(r"^/[A-Za-z0-9._/+-]*$")
_SVC_ID = re.compile(r"^[a-z][a-z0-9-]{0,62}$")

# Shell entrypoints rejected for structured containers.
_SHELL_ENTRYPOINTS = frozenset({
    "sh", "bash", "ash", "zsh", "dash", "fish", "csh", "tcsh",
    "pwsh", "pwsh.exe", "powershell", "powershell.exe",
    "cmd", "cmd.exe",
})

# Exit codes for stop/rollback guards.
EXIT_ABSENT = 0
EXIT_FOREIGN = 73
EXIT_DAEMON_ERROR = 74


class BundleError(ValueError):
    """A runtime bundle is structurally invalid or violates its contract."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadinessProbe:
    """A structured readiness probe (not a free-form argv)."""
    kind: str
    rank_scope: str = READINESS_RANK_SCOPE_ALL
    port: str = ""
    path: str = ""
    timeout_seconds: int = 30
    interval_seconds: int = 5


@dataclass(frozen=True)
class StructuredContainer:
    """A bounded structured container definition for cache/sidecar roles.

    Unlike ``runtime-profile`` (which appends vLLM ``serve`` flags),
    a structured container runs ``argv[0]`` as the direct executable via
    Docker's ``--entrypoint`` flag, passing ``argv[1:]`` as arguments.
    The image's inherited ENTRYPOINT is never used.
    """
    image: str
    image_id: str
    container_name: str
    argv: tuple[str, ...]
    port: int = 0
    environment: dict[str, str] = field(default_factory=dict)
    volumes: tuple[tuple[str, str, str], ...] = ()
    privileged: bool = False
    shm_size: str = "1g"
    startup_timeout_seconds: int = 120


@dataclass(frozen=True)
class BundleService:
    """A validated service descriptor within a runtime bundle."""
    service_id: str
    role: str
    depends_on: frozenset[str]
    source_kind: str
    source_path: str  # logical name as written in the bundle
    profile: runtime.RuntimeProfile | None  # None for structured/bridge
    structured: StructuredContainer | None  # None for runtime-profile/bridge
    readiness: ReadinessProbe | None = None
    # Optional per-rank constraint: None = all site ranks (backward
    # compatible); a frozenset limits start/stop/status/readiness/
    # verify-rollback to the listed rank IDs only.
    ranks: frozenset[int] | None = None
    document: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeBundle:
    """A validated multi-service runtime bundle."""
    bundle_id: str
    confirmation: str | None
    services: tuple[BundleService, ...]
    bundle_dir: Path | None = None
    document: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def _require_string(
    value: Any, where: str,
    pattern: re.Pattern[str] | None = None,
    what: str = "value",
) -> str:
    try:
        return runtime._require_string(value, where, pattern, what)  # noqa: SLF001
    except runtime.ProfileError as exc:
        raise BundleError(str(exc)) from exc


def _validate_readiness(doc: Any, where: str) -> ReadinessProbe | None:
    """Validate a structured readiness probe."""
    if doc is None:
        return None
    if not isinstance(doc, dict):
        raise BundleError(f"{where}: must be an object or null")
    allowed = {"kind", "rank_scope", "port", "path",
               "timeout_seconds", "interval_seconds"}
    unknown = sorted(set(doc) - allowed)
    if unknown:
        raise BundleError(f"{where}: unknown key {unknown[0]!r}")
    kind = _require_string(doc.get("kind"), f"{where}.kind")
    if kind not in VALID_READINESS_KINDS:
        raise BundleError(
            f"{where}.kind: must be one of {sorted(VALID_READINESS_KINDS)}"
        )
    rank_scope = doc.get("rank_scope", READINESS_RANK_SCOPE_ALL)
    if rank_scope not in VALID_RANK_SCOPES:
        raise BundleError(
            f"{where}.rank_scope: must be one of {sorted(VALID_RANK_SCOPES)}"
        )
    port = ""
    path = ""
    timeout = 30
    interval = 5
    if kind == READINESS_HTTP_GET:
        raw_port = doc.get("port", "")
        if isinstance(raw_port, int) and not isinstance(raw_port, bool):
            if not 1 <= raw_port <= 65535:
                raise BundleError(
                    f"{where}.port: must be 'site-api' or integer 1-65535"
                )
            port = str(raw_port)
        elif isinstance(raw_port, str):
            port = raw_port
        else:
            raise BundleError(
                f"{where}.port: must be 'site-api' or integer 1-65535"
            )
        if port not in VALID_PORT_REFERENCES:
            try:
                port_num = int(port)
                if not 1 <= port_num <= 65535:
                    raise ValueError
            except (ValueError, TypeError):
                raise BundleError(
                    f"{where}.port: must be 'site-api' or integer 1-65535"
                )
        path = _require_string(doc.get("path", "/"), f"{where}.path")
        if not _PATH_CHARS.fullmatch(path):
            raise BundleError(
                f"{where}.path: must start with / and contain only "
                "alphanumerics, dots, slashes, plus, or hyphens"
            )
        timeout = doc.get("timeout_seconds", 30)
        if not isinstance(timeout, int) or isinstance(timeout, bool):
            raise BundleError(f"{where}.timeout_seconds: must be an integer")
        if not 1 <= timeout <= 120:
            raise BundleError(f"{where}.timeout_seconds: must be 1-120")
        interval = doc.get("interval_seconds", 5)
        if not isinstance(interval, int) or isinstance(interval, bool):
            raise BundleError(f"{where}.interval_seconds: must be an integer")
        if not 1 <= interval <= 60:
            raise BundleError(f"{where}.interval_seconds: must be 1-60")
    return ReadinessProbe(
        kind=kind, rank_scope=rank_scope, port=str(port), path=path,
        timeout_seconds=timeout, interval_seconds=interval,
    )


def _check_native_profile_restrictions(
    profile: runtime.RuntimeProfile, where: str,
) -> None:
    """Reject health_check, attestation_hook, and entrypoint in bundle v1."""
    if profile.health_check:
        raise BundleError(
            f"{where}: referenced profile must not contain health_check "
            "(bundle v1 excludes profile-supplied container commands)"
        )
    if profile.attestation_hook:
        raise BundleError(
            f"{where}: referenced profile must not contain attestation_hook "
            "(bundle v1 excludes profile-supplied container commands)"
        )
    if profile.entrypoint is not None:
        raise BundleError(
            f"{where}: referenced profile must not declare entrypoint "
            "(bundle v1 excludes profile-supplied entrypoints)"
        )


def _is_native_generic_profile(profile: runtime.RuntimeProfile) -> bool:
    """Return True only if the profile was loaded from the native generic schema."""
    return profile.source_schema == runtime.SCHEMA


# ---------------------------------------------------------------------------
# Bundle-source path containment
# ---------------------------------------------------------------------------


def _validate_source_path(
    src_path: str, bundle_dir: Path, where: str,
) -> Path:
    """Validate and resolve a source path confined to the bundle directory.

    Rejects absolute, drive-relative, UNC, ``..``, placeholder, and
    symlink-escape paths.  Returns the resolved path.
    """
    if not src_path:
        raise BundleError(f"{where}: source path must be non-empty")
    p = Path(src_path)
    if p.is_absolute():
        raise BundleError(f"{where}: source path must be relative, not absolute")
    # Reject drive-relative (C:foo) and UNC (\\\\host\\share) on Windows
    if re.match(r"^[A-Za-z]:", src_path) or src_path.startswith("\\\\"):
        raise BundleError(f"{where}: source path must not be drive-relative or UNC")
    # Reject placeholders
    if "{" in src_path and "}" in src_path:
        raise BundleError(f"{where}: source path must not contain placeholders")
    # Resolve against bundle_dir and check containment
    bundle_root = bundle_dir.resolve()
    candidate = (bundle_dir / p)
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise BundleError(f"{where}: cannot resolve source path: {exc}") from exc
    try:
        resolved.relative_to(bundle_root)
    except ValueError:
        raise BundleError(
            f"{where}: source path {src_path!r} escapes bundle directory"
        )
    return resolved


# ---------------------------------------------------------------------------
# Structured-container validation
# ---------------------------------------------------------------------------


def _validate_structured_container(
    doc: Any, where: str,
) -> StructuredContainer:
    """Validate a structured container source for cache/sidecar roles."""
    if not isinstance(doc, dict):
        raise BundleError(f"{where}: structured container must be an object")
    allowed = {
        "schema", "image", "image_id", "container_name", "argv",
        "port", "environment", "volumes", "privileged",
        "shm_size", "startup_timeout_seconds",
    }
    unknown = sorted(set(doc) - allowed)
    if unknown:
        raise BundleError(f"{where}: unknown key {unknown[0]!r}")

    # Require explicit schema
    sc_schema = doc.get("schema")
    if sc_schema != "sparkring-structured-container/v1":
        raise BundleError(
            f"{where}: schema must be 'sparkring-structured-container/v1'"
        )

    image = _require_string(doc.get("image"), f"{where}.image")
    image_id = _require_string(
        doc.get("image_id"), f"{where}.image_id",
        runtime._IMAGE_ID, "image id",  # noqa: SLF001
    )
    container_name = _require_string(
        doc.get("container_name"), f"{where}.container_name",
        runtime._NAME, "container name",  # noqa: SLF001
    )

    raw_argv = doc.get("argv")
    if not isinstance(raw_argv, list) or not raw_argv:
        raise BundleError(f"{where}.argv: must be a non-empty list")
    argv: list[str] = []
    for i, arg in enumerate(raw_argv):
        if not isinstance(arg, str) or not arg or "\x00" in arg or "\n" in arg:
            raise BundleError(
                f"{where}.argv[{i}]: must be a non-empty one-line string"
            )
        argv.append(arg)

    # Reject shell entrypoints (case-insensitive on every platform)
    first = argv[0]
    base = Path(first).name
    if base.lower() in _SHELL_ENTRYPOINTS:
        raise BundleError(
            f"{where}.argv: shell entrypoint {first!r} is not allowed; "
            "use a direct binary path"
        )

    port = doc.get("port", 0)
    if not isinstance(port, int) or isinstance(port, bool):
        raise BundleError(f"{where}.port: must be an integer")
    if not 0 <= port <= 65535:
        raise BundleError(f"{where}.port: must be 0-65535")

    env: dict[str, str] = {}
    raw_env = doc.get("environment", {})
    if raw_env is None:
        raw_env = {}
    if not isinstance(raw_env, dict):
        raise BundleError(f"{where}.environment: must be an object")
    for k, v in raw_env.items():
        if not isinstance(k, str) or not runtime._ENV_KEY.fullmatch(k):  # noqa: SLF001
            raise BundleError(f"{where}.environment: invalid key {k!r}")
        if not isinstance(v, str) or "\x00" in v or "\n" in v:
            raise BundleError(f"{where}.environment.{k}: must be one-line string")
        env[k] = v

    raw_volumes = doc.get("volumes", [])
    if raw_volumes is None:
        raw_volumes = []
    if not isinstance(raw_volumes, list):
        raise BundleError(f"{where}.volumes: must be a list")
    volumes: list[tuple[str, str, str]] = []
    for i, vol in enumerate(raw_volumes):
        if not isinstance(vol, dict) or set(vol) != {"host", "container", "mode"}:
            raise BundleError(f"{where}.volumes[{i}]: must have host/container/mode")
        h = _require_string(
            vol["host"], f"{where}.volumes[{i}].host",
            runtime._ABS_PATH, "absolute path",  # noqa: SLF001
        )
        c = _require_string(
            vol["container"], f"{where}.volumes[{i}].container",
            runtime._ABS_PATH, "absolute path",  # noqa: SLF001
        )
        m = _require_string(vol["mode"], f"{where}.volumes[{i}].mode")
        if m not in ("ro", "rw"):
            raise BundleError(f"{where}.volumes[{i}].mode: must be ro or rw")
        if ".." in Path(h).parts or ".." in Path(c).parts:
            raise BundleError(
                f"{where}.volumes[{i}]: must not contain '..' path components"
            )
        volumes.append((h, c, m))

    privileged = doc.get("privileged", False)
    if not isinstance(privileged, bool):
        raise BundleError(f"{where}.privileged: must be a boolean")

    shm_size = doc.get("shm_size", "1g")
    shm_size = _require_string(shm_size, f"{where}.shm_size")

    startup_timeout = doc.get("startup_timeout_seconds", 120)
    if not isinstance(startup_timeout, int) or isinstance(startup_timeout, bool):
        raise BundleError(f"{where}.startup_timeout_seconds: must be an integer")
    if not 10 <= startup_timeout <= 3600:
        raise BundleError(f"{where}.startup_timeout_seconds: must be 10-3600")

    return StructuredContainer(
        image=image, image_id=image_id, container_name=container_name,
        argv=tuple(argv), port=port, environment=env,
        volumes=tuple(volumes), privileged=privileged,
        shm_size=shm_size, startup_timeout_seconds=startup_timeout,
    )


# ---------------------------------------------------------------------------
# Closed bridge-shape validation
# ---------------------------------------------------------------------------


def _validate_exl3_bridge_shape(
    services: list[BundleService], where: str,
) -> None:
    """Enforce exact EXL3+LMCache bridge shape.

    Exactly 2 services: one cache, one serving.  Serving depends on cache.
    Both use the same normalized tracked profile source.  No mixed sources.
    """
    exl3_services = [
        s for s in services if s.source_kind == SOURCE_EXL3_LMCACHE
    ]
    if not exl3_services:
        return

    non_exl3 = [s for s in services if s.source_kind != SOURCE_EXL3_LMCACHE]
    if non_exl3:
        raise BundleError(
            f"{where}: canonical-exl3-lmcache-cs512 bridge must not mix "
            f"source kinds; found {non_exl3[0].source_kind!r}"
        )

    if len(services) != 2:
        raise BundleError(
            f"{where}: EXL3+LMCache bridge requires exactly 2 services, "
            f"got {len(services)}"
        )

    cache_svcs = [s for s in services if s.role == ROLE_CACHE]
    serving_svcs = [s for s in services if s.role == ROLE_SERVING]
    if len(cache_svcs) != 1 or len(serving_svcs) != 1:
        raise BundleError(
            f"{where}: EXL3+LMCache bridge requires exactly one cache and "
            "one serving service"
        )

    cache = cache_svcs[0]
    serving = serving_svcs[0]
    if serving.depends_on != frozenset({cache.service_id}):
        raise BundleError(
            f"{where}: EXL3+LMCache bridge serving service must depend on "
            f"the cache service ({cache.service_id!r})"
        )

    if cache.source_path != serving.source_path:
        raise BundleError(
            f"{where}: EXL3+LMCache bridge services must use the same "
            "normalized tracked profile source path"
        )


# ---------------------------------------------------------------------------
# Parse bundle
# ---------------------------------------------------------------------------


def parse_bundle(
    document: Any, source: str | None = None,
    bundle_dir: Path | None = None,
) -> RuntimeBundle:
    """Structurally validate a parsed bundle document."""
    where = source or "<inline>"
    if not isinstance(document, dict):
        raise BundleError(f"{where}: root must be an object")

    required = {"schema", "bundle_id", "services"}
    optional = {"confirmation"}
    unknown = sorted(set(document) - required - optional)
    missing = sorted(required - set(document))
    if unknown:
        raise BundleError(f"{where}: unknown key {unknown[0]!r}")
    if missing:
        raise BundleError(f"{where}: missing key {missing[0]!r}")

    if document["schema"] != BUNDLE_SCHEMA:
        raise BundleError(
            f"{where}: unsupported schema {document['schema']!r}; "
            f"expected {BUNDLE_SCHEMA}"
        )

    bundle_id = _require_string(
        document["bundle_id"], f"{where}.bundle_id",
        re.compile(r"^[a-z][a-z0-9-]{0,62}$"), "bundle id",
    )

    confirmation = document.get("confirmation")
    if confirmation is not None:
        confirmation = _require_string(
            confirmation, f"{where}.confirmation",
            runtime._CONFIRMATION, "confirmation token",  # noqa: SLF001
        )

    raw_services = document["services"]
    if not isinstance(raw_services, list) or not raw_services:
        raise BundleError(f"{where}.services: must be a non-empty list")
    if len(raw_services) > MAX_SERVICES:
        raise BundleError(
            f"{where}.services: exceeds maximum of {MAX_SERVICES} services"
        )

    effective_dir = bundle_dir or _ROOT
    services: list[BundleService] = []
    service_ids: set[str] = set()
    serving_count = 0
    has_executable = False

    for index, svc_doc in enumerate(raw_services):
        if not isinstance(svc_doc, dict):
            raise BundleError(f"{where}.services[{index}]: must be an object")

        svc_required = {"service_id", "role", "source"}
        svc_optional = {"depends_on", "readiness", "ranks"}
        svc_unknown = sorted(set(svc_doc) - svc_required - svc_optional)
        if svc_unknown:
            raise BundleError(
                f"{where}.services[{index}]: unknown key {svc_unknown[0]!r}"
            )
        svc_missing = sorted(svc_required - set(svc_doc))
        if svc_missing:
            raise BundleError(
                f"{where}.services[{index}]: missing key {svc_missing[0]!r}"
            )

        svc_id = _require_string(
            svc_doc["service_id"],
            f"{where}.services[{index}].service_id",
            _SVC_ID, "service id",
        )
        if svc_id in service_ids:
            raise BundleError(f"{where}: duplicate service id {svc_id!r}")
        service_ids.add(svc_id)

        role = _require_string(
            svc_doc["role"], f"{where}.services[{index}].role",
        )
        if role not in VALID_ROLES:
            raise BundleError(
                f"{where}.services[{index}].role: must be one of "
                f"{sorted(VALID_ROLES)}"
            )
        if role == ROLE_SERVING:
            serving_count += 1

        # Dependencies are an order-independent set.
        depends_on: frozenset[str] = frozenset()
        if "depends_on" in svc_doc:
            raw_deps = svc_doc["depends_on"]
            if not isinstance(raw_deps, list):
                raise BundleError(
                    f"{where}.services[{index}].depends_on: must be a list"
                )
            seen_deps: set[str] = set()
            for dep in raw_deps:
                if not isinstance(dep, str) or not _SVC_ID.fullmatch(dep):
                    raise BundleError(
                        f"{where}.services[{index}].depends_on: "
                        f"invalid dependency {dep!r}"
                    )
                # Forward references validated by _validate_graph after all services collected
                if dep in seen_deps:
                    raise BundleError(
                        f"{where}.services[{index}].depends_on: "
                        f"duplicate dependency {dep!r}"
                    )
                seen_deps.add(dep)
            depends_on = frozenset(raw_deps)

        # Optional per-rank constraint: None = all site ranks.
        # Permitted only for structured-container cache/sidecar services.
        svc_ranks: frozenset[int] | None = None
        if "ranks" in svc_doc:
            raw_ranks = svc_doc["ranks"]
            if not isinstance(raw_ranks, list):
                raise BundleError(
                    f"{where}.services[{index}].ranks: must be a list"
                )
            if not raw_ranks:
                raise BundleError(
                    f"{where}.services[{index}].ranks: must be non-empty"
                )
            seen_ranks: set[int] = set()
            for r in raw_ranks:
                if not isinstance(r, int) or isinstance(r, bool) or r < 0:
                    raise BundleError(
                        f"{where}.services[{index}].ranks: "
                        f"invalid rank id {r!r}"
                    )
                if r in seen_ranks:
                    raise BundleError(
                        f"{where}.services[{index}].ranks: "
                        f"duplicate rank id {r}"
                    )
                seen_ranks.add(r)
            svc_ranks = frozenset(raw_ranks)

        readiness = _validate_readiness(
            svc_doc.get("readiness"),
            f"{where}.services[{index}].readiness",
        )

        source = svc_doc["source"]
        if not isinstance(source, dict):
            raise BundleError(
                f"{where}.services[{index}].source: must be an object"
            )
        src_allowed = {"kind", "path"}
        src_unknown = sorted(set(source) - src_allowed)
        if src_unknown:
            raise BundleError(
                f"{where}.services[{index}].source: "
                f"unknown key {src_unknown[0]!r}"
            )
        src_kind = _require_string(
            source.get("kind"),
            f"{where}.services[{index}].source.kind",
        )
        if src_kind not in VALID_SOURCE_KINDS:
            raise BundleError(
                f"{where}.services[{index}].source.kind: must be one of "
                f"{sorted(VALID_SOURCE_KINDS)}"
            )
        src_path = _require_string(
            source.get("path", ""),
            f"{where}.services[{index}].source.path",
        )

        # Bundle sources must remain inside the bundle directory.
        resolved_path = _validate_source_path(
            src_path, effective_dir,
            f"{where}.services[{index}].source.path",
        )

        # Rank scoping is only safe for structured-container cache/sidecar
        # services.  runtime-profile serving must always target all site
        # ranks because its generated --nnodes, TP, and DCP contract
        # still describes the whole site.  Canonical bridge services are
        # plan-only and delegate to the canonical launcher which has no
        # rank-scope concept.
        if svc_ranks is not None and src_kind != SOURCE_STRUCTURED_CONTAINER:
            raise BundleError(
                f"{where}.services[{index}]: 'ranks' is only permitted "
                f"for structured-container cache/sidecar services; "
                f"source kind {src_kind!r} must target all site ranks"
            )

        profile: runtime.RuntimeProfile | None = None
        structured: StructuredContainer | None = None

        if src_kind == SOURCE_RUNTIME_PROFILE:
            # Only serving role can use runtime-profile (vLLM serving)
            if role != ROLE_SERVING:
                raise BundleError(
                    f"{where}.services[{index}]: runtime-profile source "
                    "is only valid for serving role"
                )
            try:
                profile = generic.load_profile(resolved_path)
                _check_native_profile_restrictions(
                    profile, f"{where}.services[{index}]",
                )
            except runtime.ProfileError as exc:
                raise BundleError(str(exc)) from exc
            if not _is_native_generic_profile(profile):
                raise BundleError(
                    f"{where}.services[{index}]: source kind "
                    f"'runtime-profile' requires a native generic "
                    f"profile (schema {runtime.SCHEMA}); got "
                    f"{profile.source_schema}."
                )
            has_executable = True

        elif src_kind == SOURCE_STRUCTURED_CONTAINER:
            # Only cache/sidecar roles can use structured-container
            if role == ROLE_SERVING:
                raise BundleError(
                    f"{where}.services[{index}]: structured-container "
                    "source is not valid for serving role; use runtime-profile"
                )
            try:
                structured = _validate_structured_container(
                    json.loads(resolved_path.read_text(encoding="utf-8")),
                    f"{where}.services[{index}].source",
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise BundleError(str(exc)) from exc
            has_executable = True

        # Validate bridge inputs during structural validation.
        if src_kind in PLAN_ONLY_SOURCE_KINDS:
            if not resolved_path.is_file():
                raise BundleError(
                    f"{where}.services[{index}].source.path: "
                    f"bridge source {src_path!r} not found or not a file"
                )


        services.append(BundleService(
            service_id=svc_id, role=role, depends_on=depends_on,
            source_kind=src_kind, source_path=src_path,
            profile=profile, structured=structured, readiness=readiness,
            ranks=svc_ranks,
            document=svc_doc,
        ))

    if serving_count == 0:
        raise BundleError(
            f"{where}: bundle must have exactly one serving service"
        )
    if serving_count > 1:
        raise BundleError(
            f"{where}: bundle must have at most one serving service"
        )

    _validate_graph(services, where)

    # Bridge adapters accept only their exact declared service shape.
    _validate_exl3_bridge_shape(services, where)

    # B2: Native executable bundles require a non-null confirmation token
    if has_executable and not confirmation:
        raise BundleError(
            f"{where}: executable native bundle requires a non-null "
            "confirmation token; set 'confirmation' or use a "
            "plan-only canonical bridge source kind"
        )

    return RuntimeBundle(
        bundle_id=bundle_id,
        confirmation=confirmation,
        services=tuple(services),
        bundle_dir=effective_dir,
        document=document,
    )


def load_bundle(path: Path) -> RuntimeBundle:
    """Load and structurally validate a bundle JSON file."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return parse_bundle(document, source=str(path), bundle_dir=path.parent)


# ---------------------------------------------------------------------------
# Graph validation and ordering
# ---------------------------------------------------------------------------


def _validate_graph(services: list[BundleService], where: str) -> None:
    ids = {svc.service_id for svc in services}
    for svc in services:
        for dep in svc.depends_on:
            if dep not in ids:
                raise BundleError(
                    f"{where}: service {svc.service_id!r} depends on "
                    f"unknown service {dep!r}"
                )
    _detect_cycles(services, where)


def _detect_cycles(services: list[BundleService], where: str) -> None:
    by_id = {svc.service_id: svc for svc in services}
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {svc.service_id: WHITE for svc in services}

    def _visit(node_id: str, path: list[str]) -> None:
        color[node_id] = GRAY
        for dep in by_id[node_id].depends_on:
            if color[dep] == GRAY:
                cycle = " -> ".join(path + [dep])
                raise BundleError(
                    f"{where}: dependency cycle detected: {cycle}"
                )
            if color[dep] == WHITE:
                _visit(dep, path + [dep])
        color[node_id] = BLACK

    for svc in services:
        if color[svc.service_id] == WHITE:
            _visit(svc.service_id, [svc.service_id])


def topological_order(
    services: tuple[BundleService, ...],
) -> list[BundleService]:
    """Return services in stable topological order (ties by service_id)."""
    import heapq
    by_id = {svc.service_id: svc for svc in services}
    in_degree: dict[str, int] = {svc.service_id: 0 for svc in services}
    adj: dict[str, list[str]] = {svc.service_id: [] for svc in services}
    for svc in services:
        for dep in svc.depends_on:
            adj[dep].append(svc.service_id)
            in_degree[svc.service_id] += 1

    heap = [sid for sid in sorted(in_degree) if in_degree[sid] == 0]
    heapq.heapify(heap)
    result: list[BundleService] = []
    while heap:
        sid = heapq.heappop(heap)
        result.append(by_id[sid])
        for child in sorted(adj[sid]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(heap, child)
    if len(result) != len(services):
        raise BundleError("dependency cycle detected in topological_order")
    return result


def reverse_order(
    services: tuple[BundleService, ...],
) -> list[BundleService]:
    """Return services in reverse dependency order for stop/rollback."""
    return list(reversed(topological_order(services)))


# ---------------------------------------------------------------------------
# Per-rank expansion constraint
# ---------------------------------------------------------------------------


def _service_ranks(svc: BundleService, site: Any) -> frozenset[int]:
    """Return the effective rank set for a service.

    If the service declares ``ranks``, only those rank IDs that exist
    in the site are returned.  Otherwise all site rank IDs are returned
    (backward-compatible default).
    """
    site_ranks = {r.id for r in site.ranks}
    if svc.ranks is None:
        return frozenset(site_ranks)
    return frozenset(svc.ranks) & site_ranks


def validate_service_ranks(
    bundle: RuntimeBundle, site: Any,
) -> None:
    """Fail-closed validation that all declared rank IDs exist in the site."""
    site_ranks = {r.id for r in site.ranks}
    for svc in bundle.services:
        if svc.ranks is None:
            continue
        unknown = sorted(svc.ranks - site_ranks)
        if unknown:
            raise BundleError(
                f"service {svc.service_id!r}: rank(s) {unknown} not "
                f"found in site (available: {sorted(site_ranks)})"
            )
        if (
            svc.readiness is not None
            and svc.readiness.rank_scope == READINESS_RANK_SCOPE_RANK0
            and 0 not in svc.ranks
        ):
            raise BundleError(
                f"service {svc.service_id!r}: readiness rank_scope "
                "'rank0' requires rank 0 in the service's ranks"
            )


# ---------------------------------------------------------------------------
# Container name collision detection
# ---------------------------------------------------------------------------


def check_container_name_collisions(
    bundle: RuntimeBundle, site: Any,
) -> None:
    """Reject per-rank container name collisions across services."""
    names: dict[str, str] = {}
    for svc in bundle.services:
        for rank_id in sorted(_service_ranks(svc, site)):
            name = _container_name(svc, rank_id)
            if name in names and names[name] != svc.service_id:
                raise BundleError(
                    f"container name collision: {name!r} used by "
                    f"service {names[name]!r} and {svc.service_id!r}"
                )
            names[name] = svc.service_id


def _container_name(svc: BundleService, rank: int) -> str:
    """Get the container name for a service at a rank."""
    if svc.profile is not None:
        return runtime.container_name(svc.profile, rank)
    if svc.structured is not None:
        return f"{svc.structured.container_name}-r{rank}"
    return f"{svc.service_id}-r{rank}"


# ---------------------------------------------------------------------------
# Shared ownership guard for both readiness kinds
# ---------------------------------------------------------------------------


def _ownership_for(bundle: RuntimeBundle, svc: BundleService) -> runtime.Ownership:
    """Build the ownership tuple for a native bundle service."""
    return runtime.Ownership(
        bundle_id=bundle.bundle_id,
        service_id=svc.service_id,
        profile_id=(
            svc.profile.profile_id if svc.profile
            else (svc.structured.image_id if svc.structured else "")
        ),
    )


def _ownership_guard_script(
    engine: str, name: str, ownership: runtime.Ownership,
    profile_id: str, *,
    inspect_error: int = 1, mismatch_error: int = 1,
) -> str:
    """Build the shared five-label ownership guard script.

    Checks all five labels: managed, profile (legacy), bundle, service,
    source-profile.  Used by both readiness kinds.

    inspect_error: exit code when inspect itself fails (daemon/API issue).
    mismatch_error: exit code when label value doesn't match (foreign).
    """
    labels = [
        (runtime.MANAGED_LABEL, "true"),
        (runtime.PROFILE_LABEL, profile_id),
        (runtime.BUNDLE_LABEL, ownership.bundle_id),
        (runtime.SERVICE_LABEL, ownership.service_id),
        (runtime.SOURCE_PROFILE_LABEL, ownership.profile_id),
    ]
    parts: list[str] = []
    for label_key, expected in labels:
        parts.append(
            f'val=$({engine} inspect --format '
            f"'{{{{index .Config.Labels \"{label_key}\"}}}}' "
            f"{shlex.quote(name)} 2>/dev/null) || exit {inspect_error}; "
            f'[ "$val" = {shlex.quote(expected)} ] || exit {mismatch_error}; '
        )
    return "".join(parts)


def _readiness_actions(
    site: Any, svc: BundleService, bundle: RuntimeBundle,
) -> list[runtime.RemoteAction]:
    """Generate readiness probe actions with shared ownership guards.

    Both readiness kinds use the same ownership preamble.
    HTTP readiness verifies all labels before curl.
    The final attempt does not sleep, so the wall-clock budget fits the timeout.
    """
    if svc.readiness is None:
        return []
    if svc.profile is None and svc.structured is None:
        return []
    actions: list[runtime.RemoteAction] = []
    probe = svc.readiness
    ownership = _ownership_for(bundle, svc)
    engine = (
        svc.profile.engine if svc.profile
        else "docker"
    )
    profile_id = (
        svc.profile.profile_id if svc.profile
        else svc.structured.image_id if svc.structured else ""
    )

    effective_ranks = _service_ranks(svc, site)
    for rank in sorted(site.ranks, key=lambda r: r.id):
        if rank.id not in effective_ranks:
            continue
        if probe.rank_scope == READINESS_RANK_SCOPE_RANK0 and rank.id != 0:
            continue
        name = _container_name(svc, rank.id)
        guard = _ownership_guard_script(
            engine, name, ownership, profile_id,
        )
        if probe.kind == READINESS_CONTAINER_RUNNING:
            script = guard + (
                f'test "$({engine} inspect -f '
                f"'{{{{.State.Running}}}}' {name} 2>/dev/null)\" = true"
            )
        elif probe.kind == READINESS_HTTP_GET:
            if probe.port == READINESS_PORT_SITE_API:
                port = str(site.serving.api_port)
            else:
                port = str(probe.port)
            # Budget: total curl+sleep never exceeds timeout_seconds.
            # Each full slot = curl_timeout + sleep_time.  The final
            # attempt has no sleep and may use a shorter curl_timeout
            # so the grand total fits exactly within timeout_seconds.
            slot = min(probe.interval_seconds, probe.timeout_seconds)
            curl_timeout = slot // 2
            if curl_timeout < 1:
                curl_timeout = 1
            sleep_time = slot - curl_timeout
            # Number of full-slot attempts (each with trailing sleep)
            full_attempts = probe.timeout_seconds // slot
            remaining = probe.timeout_seconds - full_attempts * slot
            if remaining > 0:
                # Final partial attempt with no trailing sleep
                final_curl = min(curl_timeout, remaining)
                if final_curl < 1:
                    final_curl = 1
                attempts = full_attempts + 1
            else:
                final_curl = curl_timeout
                attempts = full_attempts if full_attempts > 0 else 1
            script = (
                f"{guard}"
                f'test "$({engine} inspect -f '
                f"'{{{{.State.Running}}}}' {name} 2>/dev/null)\" = true"
                f" || exit 1; "
                f"ok=0; "
                f"for i in $(seq 1 {attempts}); do "
                f"curl_timeout={curl_timeout}; "
                f'if [ "$i" -eq {attempts} ]; then curl_timeout={final_curl}; fi; '
                f"curl -fsS --max-time $curl_timeout "
                f"http://127.0.0.1:{port}{probe.path} >/dev/null 2>&1"
                f" && ok=1 && break; "
                f'if [ "$i" -lt {attempts} ] && [ {sleep_time} -gt 0 ]; then '
                f"sleep {sleep_time}; fi; "
                f"done; "
                f'test "$ok" = 1 || exit 1'
            )
        else:
            continue
        actions.append(runtime.RemoteAction(
            rank.id, rank.ssh_target, ("sh", "-c", script),
        ))
    return actions


# ---------------------------------------------------------------------------
# Native bundle phase/action builders
# ---------------------------------------------------------------------------


def _native_start_actions(
    site: Any, svc: BundleService, bundle: RuntimeBundle,
) -> list[runtime.RemoteAction]:
    """Build native start actions with ownership labels, sorted by rank."""
    if svc.profile is not None:
        actions = runtime.start_actions(
            site, svc.profile, ownership=_ownership_for(bundle, svc),
        )
        effective = _service_ranks(svc, site)
        return sorted(
            (a for a in actions if a.rank in effective),
            key=lambda a: a.rank,
        )
    if svc.structured is not None:
        return _structured_start_actions(
            site, svc, bundle,
        )
    return []


def _structured_start_actions(
    site: Any, svc: BundleService, bundle: RuntimeBundle,
) -> list[runtime.RemoteAction]:
    """Build direct-argv start actions for a structured container.

    Runs the declared argv directly without adding vLLM serve, TP, or DCP flags.
    Uses exact image identity, ownership labels, deterministic env/volumes.
    """
    sc = svc.structured
    ownership = _ownership_for(bundle, svc)
    actions: list[runtime.RemoteAction] = []
    effective_ranks = _service_ranks(svc, site)
    for rank in sorted(site.ranks, key=lambda r: r.id):
        if rank.id not in effective_ranks:
            continue
        name = f"{sc.container_name}-r{rank.id}"
        command: list[str] = [
            "docker", "run", "--detach", "--name", name,
            "--label", f"{runtime.MANAGED_LABEL}=true",
            "--label", f"{runtime.PROFILE_LABEL}={sc.image_id}",
            "--label", f"{runtime.BUNDLE_LABEL}={ownership.bundle_id}",
            "--label", f"{runtime.SERVICE_LABEL}={ownership.service_id}",
            "--label", f"{runtime.SOURCE_PROFILE_LABEL}={ownership.profile_id}",
            "--network", "host",
            "--ipc", "host",
            "--shm-size", sc.shm_size,
            "--ulimit", "memlock=-1:-1",
        ]
        if sc.privileged:
            command.append("--privileged")
        for host_path, container_path, mode in sc.volumes:
            command.extend(("--volume", f"{host_path}:{container_path}:{mode}"))
        for env_key in sorted(sc.environment):
            command.extend(("--env", f"{env_key}={sc.environment[env_key]}"))
        command.extend(["--entrypoint", sc.argv[0]])
        command.append(sc.image_id)
        command.extend(sc.argv[1:])

        # Image verify guard (fail-closed on identity drift)
        guard = (
            f'test "$(docker image inspect --format '
            f"'{{{{.Id}}}}' {shlex.quote(sc.image)})\" = "
            f"{shlex.quote(sc.image_id)}"
        )
        guard += f" && exec {shlex.join(command)}"
        actions.append(runtime.RemoteAction(
            rank.id, rank.ssh_target, ("sh", "-lc", guard),
        ))
    return actions


def _native_stop_actions(
    site: Any, svc: BundleService, bundle: RuntimeBundle,
) -> list[runtime.RemoteAction]:
    """Build native stop actions with ownership guards, sorted by rank.

    Probes the daemon before enumerating the exact container name.
    Enumeration failure => 74; empty => proven absence (exit 0);
    exact name => proceed to label checks; unexpected output => 74.
    Label inspect failure on proven-present => 74 (unknown), not 73.
    Present foreign label => 73.
    """
    actions: list[runtime.RemoteAction] = []
    ownership = _ownership_for(bundle, svc)
    engine = svc.profile.engine if svc.profile else "docker"
    profile_id = (
        svc.profile.profile_id if svc.profile
        else svc.structured.image_id if svc.structured else ""
    )
    effective_ranks = _service_ranks(svc, site)
    for rank in sorted(site.ranks, key=lambda r: r.id):
        if rank.id not in effective_ranks:
            continue
        name = _container_name(svc, rank.id)
        script = (
            # Daemon probe
            f"{engine} info >/dev/null 2>&1 || exit {EXIT_DAEMON_ERROR}; "
            # Exact-name enumeration: failure => unknown, empty => absence
            f"listing=$({engine} ps -a --filter name=^/{shlex.quote(name)}$ "
            f"--format '{{{{.Names}}}}' 2>&1) || exit {EXIT_DAEMON_ERROR}; "
            f'if [ -z "$listing" ]; then exit {EXIT_ABSENT}; fi; '
            f'if [ "$listing" != "{shlex.quote(name)}" ]; then exit {EXIT_DAEMON_ERROR}; fi; '
            # Object is proven present: label inspect failure => unknown
            + _ownership_guard_script(
                engine, name, ownership, profile_id,
                inspect_error=EXIT_DAEMON_ERROR, mismatch_error=EXIT_FOREIGN,
            )
            + f"exec {engine} rm --force {shlex.quote(name)}"
        )
        actions.append(runtime.RemoteAction(
            rank.id, rank.ssh_target, ("sh", "-c", script),
        ))
    return actions


def _native_status_actions(
    site: Any, svc: BundleService, bundle: RuntimeBundle,
) -> list[runtime.RemoteAction]:
    """Build status actions with ownership guards, sorted by rank."""
    actions: list[runtime.RemoteAction] = []
    ownership = _ownership_for(bundle, svc)
    engine = svc.profile.engine if svc.profile else "docker"
    profile_id = (
        svc.profile.profile_id if svc.profile
        else svc.structured.image_id if svc.structured else ""
    )
    effective_ranks = _service_ranks(svc, site)
    for rank in sorted(site.ranks, key=lambda r: r.id):
        if rank.id not in effective_ranks:
            continue
        name = _container_name(svc, rank.id)
        script = (
            f"{engine} info >/dev/null 2>&1 || exit {EXIT_DAEMON_ERROR}; "
            + _ownership_guard_script(
                engine, name, ownership, profile_id,
                inspect_error=EXIT_DAEMON_ERROR, mismatch_error=1,
            )
            + f"exec {engine} inspect --format "
            f"'{{{{.State.Status}}}}' {name}"
        )
        actions.append(runtime.RemoteAction(
            rank.id, rank.ssh_target, ("sh", "-c", script),
        ))
    return actions


def _native_verify_rollback_actions(
    site: Any, svc: BundleService, bundle: RuntimeBundle,
) -> list[runtime.RemoteAction]:
    """Build verify-rollback actions with ownership guards.

    Probes the daemon before enumerating the exact container name.
    Enumeration failure => 2; empty => proven absence (exit 0);
    exact name => present (exit 1); unexpected output => 2.
    Label inspect failure on proven-present => 2 (unknown).
    """
    actions: list[runtime.RemoteAction] = []
    ownership = _ownership_for(bundle, svc)
    engine = svc.profile.engine if svc.profile else "docker"
    profile_id = (
        svc.profile.profile_id if svc.profile
        else svc.structured.image_id if svc.structured else ""
    )
    effective_ranks = _service_ranks(svc, site)
    for rank in sorted(site.ranks, key=lambda r: r.id):
        if rank.id not in effective_ranks:
            continue
        name = _container_name(svc, rank.id)
        script = (
            # Daemon probe
            f"{engine} info >/dev/null 2>&1 || exit 2; "
            # Exact-name enumeration
            f"listing=$({engine} ps -a --filter name=^/{shlex.quote(name)}$ "
            f"--format '{{{{.Names}}}}' 2>&1) || exit 2; "
            # Empty => proven absence
            f'if [ -z "$listing" ]; then exit 0; fi; '
            # Unexpected output => unknown
            f'if [ "$listing" != "{shlex.quote(name)}" ]; then exit 2; fi; '
            # Exact name present: any present => exit 1
            # Label checks for informational purposes (but present regardless)
            + _ownership_guard_script(
                engine, name, ownership, profile_id,
                inspect_error=2, mismatch_error=1,
            )
            + "exit 1"
        )
        actions.append(runtime.RemoteAction(
            rank.id, rank.ssh_target, ("sh", "-c", script),
        ))
    return actions


# ---------------------------------------------------------------------------
# EXL3+LMCache bridge delegated to the canonical launcher
# ---------------------------------------------------------------------------


def _load_exl3_lmcache_bridge(
    site: Any, bundle: RuntimeBundle,
) -> dict[str, list[runtime.RemoteAction]]:
    """Build all 8 canonical phases by calling the canonical launcher.

    Consumes ``build_phases()`` and ``lifecycle_sequence()`` exported from the
    canonical launcher.
    """
    import sparkring_exl3_launcher as exl3
    import sparkring_exl3_lmcache_launcher as lmcache

    serving_svc = next(
        svc for svc in bundle.services if svc.role == ROLE_SERVING
    )
    profile_path = _validate_source_path(
        serving_svc.source_path,
        bundle.bundle_dir or _ROOT,
        "bridge source path",
    )
    profile = exl3.load_profile(profile_path)

    # Use the canonical launcher's exported phase builder.
    return lmcache.build_phases(site, profile)


# ---------------------------------------------------------------------------
# Canonical lifecycle sequences exported by the canonical launcher
# ---------------------------------------------------------------------------


def lifecycle_sequence(
    command: str, profile: Any,
) -> list[dict[str, Any]]:
    """Return the canonical lifecycle phase sequence for a command.

    Delegates to the canonical launcher's exported ``lifecycle_sequence()``
    function, which is also consumed by canonical ``main()``. Includes
    command-appropriate ``on_failure`` semantics and a timeout for every phase.
    """
    import sparkring_exl3_lmcache_launcher as lmcache
    return lmcache.lifecycle_sequence(command, profile)


# ---------------------------------------------------------------------------
# Plan document
# ---------------------------------------------------------------------------


def _render_actions(actions: list[runtime.RemoteAction]) -> list[dict]:
    return [
        {
            "rank": action.rank,
            "ssh_target": action.ssh_target,
            "remote_command": action.shell_command,
        }
        for action in sorted(actions, key=lambda a: a.rank)
    ]


def _plan_identity_hash(plan: dict[str, Any]) -> str:
    """Compute a deterministic semantic content hash for a plan document."""
    canonical = json.dumps(plan, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bundle_plan(
    bundle: RuntimeBundle, site: Any,
) -> dict[str, Any]:
    """Build a complete ordered phase plan document for a bundle."""
    is_bridge = any(
        svc.source_kind in PLAN_ONLY_SOURCE_KINDS for svc in bundle.services
    )

    if is_bridge:
        return _bridge_plan(bundle, site)

    ordered = topological_order(bundle.services)
    reversed_ = reverse_order(bundle.services)
    validate_service_ranks(bundle, site)
    check_container_name_collisions(bundle, site)

    # Emit every ordered start and readiness phase.
    phases: list[dict[str, Any]] = []
    for svc in ordered:
        phases.append({
            "phase": "start",
            "service_id": svc.service_id,
            "role": svc.role,
            "depends_on": sorted(svc.depends_on),
            "safety_class": ["MUTATES HOST"],
            "timeout": (
                svc.profile.startup_timeout_seconds if svc.profile
                else svc.structured.startup_timeout_seconds
            ),
            "actions": _render_actions(_native_start_actions(site, svc, bundle)),
        })
        if svc.readiness:
            ready_actions = _readiness_actions(site, svc, bundle)
            if ready_actions:
                phases.append({
                    "phase": "readiness",
                    "service_id": svc.service_id,
                    "role": svc.role,
                    "depends_on": sorted(svc.depends_on),
                    "safety_class": ["READ-ONLY REMOTE"],
                    "timeout": svc.readiness.timeout_seconds,
                    "actions": _render_actions(ready_actions),
                })

    # Emit read-only status phases after startup readiness.
    for svc in ordered:
        phases.append({
            "phase": "status",
            "service_id": svc.service_id,
            "role": svc.role,
            "depends_on": sorted(svc.depends_on),
            "safety_class": ["READ-ONLY REMOTE"],
            "timeout": 30,
            "actions": _render_actions(
                _native_status_actions(site, svc, bundle),
            ),
        })

    # Roll back each service in reverse topological order
    # (not a flattened rank-sorted bag)
    for svc in reversed_:
        phases.append({
            "phase": "stop",
            "service_id": svc.service_id,
            "role": svc.role,
            "depends_on": sorted(svc.depends_on),
            "safety_class": ["MUTATES HOST", "STOPS SERVING"],
            "timeout": (
                svc.profile.startup_timeout_seconds if svc.profile
                else svc.structured.startup_timeout_seconds
            ),
            "actions": _render_actions(_native_stop_actions(site, svc, bundle)),
        })

    for svc in reversed_:
        phases.append({
            "phase": "rollback",
            "service_id": svc.service_id,
            "role": svc.role,
            "depends_on": sorted(svc.depends_on),
            "safety_class": ["MUTATES HOST", "STOPS SERVING"],
            "timeout": (
                svc.profile.startup_timeout_seconds if svc.profile
                else svc.structured.startup_timeout_seconds
            ),
            "actions": _render_actions(_native_stop_actions(site, svc, bundle)),
        })

    for svc in reversed_:
        phases.append({
            "phase": "verify_rollback",
            "service_id": svc.service_id,
            "role": svc.role,
            "depends_on": sorted(svc.depends_on),
            "safety_class": ["READ-ONLY REMOTE"],
            "timeout": 30,
            "actions": _render_actions(
                _native_verify_rollback_actions(site, svc, bundle),
            ),
        })

    # Publish evidence scope, identity scope, and ownership labels.
    plan: dict[str, Any] = {
        "schema": BUNDLE_PLAN_SCHEMA,
        "bundle_id": bundle.bundle_id,
        "execution_supported": True,
        "plan_identity": "",  # filled below
        "evidence_scope": (
            "offline-validated: plan structure and generated actions; "
            "no live execution, no model correctness, no acceptance"
        ),
        "services": [
            {
                "service_id": svc.service_id,
                "role": svc.role,
                "depends_on": sorted(svc.depends_on),
                "source_kind": svc.source_kind,
                "source_path": svc.source_path,
                "profile_id": (
                    svc.profile.profile_id if svc.profile
                    else svc.structured.image_id if svc.structured
                    else None
                ),
                "identity_scope": (
                    generic._identity_scope(svc.profile)  # noqa: SLF001
                    if svc.profile
                    else f"structured-container:{svc.structured.image_id}"
                    if svc.structured
                    else None
                ),
                "ownership_labels": {
                    "managed": "true",
                    "profile": (
                        svc.profile.profile_id if svc.profile
                        else svc.structured.image_id if svc.structured
                        else None
                    ),
                    "bundle": bundle.bundle_id,
                    "service": svc.service_id,
                    "source_profile": (
                        svc.profile.profile_id if svc.profile
                        else svc.structured.image_id if svc.structured
                        else None
                    ),
                },
                "ranks": (
                    sorted(svc.ranks) if svc.ranks is not None
                    else sorted({r.id for r in site.ranks})
                ),
            }
            for svc in ordered
        ],
        "ordering": {
            "start": [svc.service_id for svc in ordered],
            "stop": [svc.service_id for svc in reversed_],
        },
        "graph_order": [svc.service_id for svc in ordered],
        "plan_generation": "OFFLINE",
        "operation_capabilities": {
            "plan": {
                "safety_class": ["OFFLINE"],
                "requires_execute": False,
            },
            "start": {
                "safety_class": ["MUTATES HOST"],
                "requires_execute": True,
                "requires_confirmation": True,
            },
            "readiness": {
                "safety_class": ["READ-ONLY REMOTE"],
                "requires_execute": True,
            },
            "status": {
                "safety_class": ["READ-ONLY REMOTE"],
                "requires_execute": True,
                "requires_confirmation": False,
            },
            "stop": {
                "safety_class": ["MUTATES HOST", "STOPS SERVING"],
                "requires_execute": True,
                "requires_confirmation": True,
            },
            "rollback": {
                "safety_class": ["MUTATES HOST", "STOPS SERVING"],
                "requires_execute": True,
                "requires_confirmation": True,
            },
            "verify_rollback": {
                "safety_class": ["READ-ONLY REMOTE"],
                "requires_execute": True,
                "requires_confirmation": False,
            },
        },
        "phases": phases,
    }
    plan["plan_identity"] = _plan_identity_hash(
        {k: v for k, v in plan.items() if k != "plan_identity"}
    )
    return plan


def _bridge_plan(
    bundle: RuntimeBundle, site: Any,
) -> dict[str, Any]:
    """Build a plan-only projection for a canonical EXL3+LMCache bridge.

    ``parse_bundle`` enforces the exact bridge shape. The projection consumes
    canonical phases and lifecycle sequences, and includes evidence scope,
    identities, labels, mounts, probes, argv, ordering, and topology.
    """
    import sparkring_exl3_launcher as exl3

    canonical_phases = _load_exl3_lmcache_bridge(site, bundle)
    serving_svc = next(
        svc for svc in bundle.services if svc.role == ROLE_SERVING
    )
    profile_path = _validate_source_path(
        serving_svc.source_path,
        bundle.bundle_dir or _ROOT,
        "bridge source path",
    )
    profile = exl3.load_profile(profile_path)

    sequences: dict[str, list[dict[str, Any]]] = {}
    for cmd in ("start", "status", "restart-engines",
                "restart-stack", "rollback", "verify-rollback"):
        seq = lifecycle_sequence(cmd, profile)
        sequences[cmd] = seq

    ordered = topological_order(bundle.services)
    reversed_ = reverse_order(bundle.services)

    plan: dict[str, Any] = {
        "schema": BUNDLE_PLAN_SCHEMA,
        "bundle_id": bundle.bundle_id,
        "execution_supported": False,
        "execution_limitation": (
            "The EXL3+LMCache bridge is plan-only. Canonical rollback "
            "is whole-stack, not invocation-ledgered, and canonical "
            "labels do not carry the bundle/service/source-profile "
            "ownership tuple. Use the canonical launcher for execution."
        ),
        "evidence_scope": (
            "offline-validated: canonical phase actions and lifecycle "
            "sequences; no live execution, no model correctness, "
            "no acceptance"
        ),
        "services": [
            {
                "service_id": svc.service_id,
                "role": svc.role,
                "depends_on": sorted(svc.depends_on),
                "source_kind": svc.source_kind,
                "source_path": svc.source_path,
                "identity_scope": (
                    f"canonical-exl3-lmcache:{profile.profile_id}"
                ),
            }
            for svc in ordered
        ],
        "ordering": {
            "start": [svc.service_id for svc in ordered],
            "stop": [svc.service_id for svc in reversed_],
        },
        "graph_order": [svc.service_id for svc in ordered],
        "canonical_phases": {
            name: _render_actions(actions)
            for name, actions in canonical_phases.items()
        },
        "lifecycle_sequences": sequences,
        "plan_generation": "OFFLINE",
        "operation_capabilities": {
            "plan": {
                "safety_class": ["OFFLINE"],
                "requires_execute": False,
            },
        },
    }
    plan["plan_identity"] = _plan_identity_hash(
        {k: v for k, v in plan.items() if k != "plan_identity"}
    )
    return plan


# ---------------------------------------------------------------------------
# Semantic projections for diff (items 14)
# ---------------------------------------------------------------------------


def _service_projection(svc: BundleService) -> dict[str, Any]:
    proj: dict[str, Any] = {
        "service_id": svc.service_id,
        "role": svc.role,
        "depends_on": sorted(svc.depends_on),
        "source_kind": svc.source_kind,
        "source_path": svc.source_path,
    }
    if svc.profile is not None:
        proj["profile"] = generic._profile_projection(svc.profile)  # noqa: SLF001
    if svc.structured is not None:
        proj["structured_container"] = {
            "image": svc.structured.image,
            "image_id": svc.structured.image_id,
            "container_name": svc.structured.container_name,
            "argv": list(svc.structured.argv),
            "port": svc.structured.port,
            "environment": dict(svc.structured.environment),
            "volumes": [list(v) for v in svc.structured.volumes],
            "privileged": svc.structured.privileged,
            "shm_size": svc.structured.shm_size,
        }
    if svc.readiness is not None:
        proj["readiness"] = {
            "kind": svc.readiness.kind,
            "rank_scope": svc.readiness.rank_scope,
            "port": svc.readiness.port,
            "path": svc.readiness.path,
            "timeout_seconds": svc.readiness.timeout_seconds,
            "interval_seconds": svc.readiness.interval_seconds,
        }
    if svc.ranks is not None:
        proj["ranks"] = sorted(svc.ranks)
    return proj


def bundle_projection(bundle: RuntimeBundle) -> dict[str, Any]:
    """Stable semantic projection of a bundle for diff comparison."""
    ordered = topological_order(bundle.services)
    return {
        "schema": BUNDLE_SCHEMA,
        "bundle_id": bundle.bundle_id,
        "confirmation": bundle.confirmation,
        "ordering": {
            "start": [svc.service_id for svc in ordered],
            "stop": [svc.service_id for svc in reverse_order(bundle.services)],
        },
        "services": {
            svc.service_id: _service_projection(svc) for svc in ordered
        },
    }


def plan_projection(
    plan: dict[str, Any], bundle: RuntimeBundle, site: Any,
) -> dict[str, Any]:
    """Stable semantic projection of a resolved plan for diff comparison.

    Includes native and canonical phases, lifecycle sequences,
    source/profile/image identity, labels, mounts, probes, argv/commands,
    graph/action order, and topology.
    """
    proj = bundle_projection(bundle)
    if "phases" in plan:
        proj["phases"] = plan["phases"]
    if "canonical_phases" in plan:
        proj["canonical_phases"] = plan["canonical_phases"]
    if "lifecycle_sequences" in plan:
        proj["lifecycle_sequences"] = plan["lifecycle_sequences"]
    if "execution_supported" in plan:
        proj["execution_supported"] = plan["execution_supported"]
    if "plan_identity" in plan:
        proj["plan_identity"] = plan["plan_identity"]
    if "evidence_scope" in plan:
        proj["evidence_scope"] = plan["evidence_scope"]
    if "operation_capabilities" in plan:
        proj["operation_capabilities"] = plan["operation_capabilities"]
    if site is not None:
        proj["topology"] = generic._target_topology(site)  # noqa: SLF001
    return proj


# ---------------------------------------------------------------------------
# Recursive diff
# ---------------------------------------------------------------------------

_MISSING = object()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if value is _MISSING:
        return None
    return value


def recursive_diff(
    left: Any, right: Any, prefix: str = "",
) -> list[dict[str, Any]]:
    """Recursively compare two values, returning exact JSON-path deltas."""
    diffs: list[dict[str, Any]] = []
    if type(left) is not type(right):
        diffs.append({
            "field": prefix or "<root>",
            "a": _jsonable(left), "b": _jsonable(right),
        })
        return diffs
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            lv = left.get(key, _MISSING)
            rv = right.get(key, _MISSING)
            if lv is _MISSING:
                diffs.append({"field": path, "a": None, "b": _jsonable(rv)})
            elif rv is _MISSING:
                diffs.append({"field": path, "a": _jsonable(lv), "b": None})
            else:
                diffs.extend(recursive_diff(lv, rv, path))
    elif isinstance(left, list):
        for i in range(max(len(left), len(right))):
            path = f"{prefix}[{i}]"
            if i >= len(left):
                diffs.append({
                    "field": path, "a": None, "b": _jsonable(right[i]),
                })
            elif i >= len(right):
                diffs.append({
                    "field": path, "a": _jsonable(left[i]), "b": None,
                })
            else:
                diffs.extend(recursive_diff(left[i], right[i], path))
    elif left != right:
        diffs.append({
            "field": prefix or "<root>",
            "a": _jsonable(left), "b": _jsonable(right),
        })
    return diffs


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


def bundle_explain(bundle: RuntimeBundle, site: Any = None) -> dict[str, Any]:
    """Produce an explanation of a bundle's structure and safety."""
    ordered = topological_order(bundle.services)
    is_bridge = any(
        svc.source_kind in PLAN_ONLY_SOURCE_KINDS for svc in bundle.services
    )
    return {
        "schema": BUNDLE_SCHEMA,
        "bundle_id": bundle.bundle_id,
        "services": [
            {
                "service_id": svc.service_id,
                "role": svc.role,
                "depends_on": sorted(svc.depends_on),
                "source_kind": svc.source_kind,
                "source_path": svc.source_path,
                "profile_id": (
                    svc.profile.profile_id if svc.profile
                    else svc.structured.image_id if svc.structured
                    else None
                ),
                "identity_scope": (
                    generic._identity_scope(svc.profile)  # noqa: SLF001
                    if svc.profile
                    else (
                        f"structured-container:{svc.structured.image_id}"
                        if svc.structured
                        else "canonical-exl3-lmcache"
                    )
                ),
                "readiness": (
                    {
                        "kind": svc.readiness.kind,
                        "rank_scope": svc.readiness.rank_scope,
                    }
                    if svc.readiness else None
                ),
                "ranks": (
                    sorted(svc.ranks) if svc.ranks is not None
                    else None
                ),
            }
            for svc in ordered
        ],
        "ordering": {
            "start": [svc.service_id for svc in ordered],
            "stop": [svc.service_id for svc in reverse_order(bundle.services)],
        },
        "execution_supported": not is_bridge,
        "execution_limitation": (
            "Canonical bridge is plan-only. Use the canonical "
            "launcher for execution."
            if is_bridge else None
        ),
        "safety_classes": {
            "plan": ["OFFLINE"],
            "start": ["MUTATES HOST"],
            "stop": ["MUTATES HOST", "STOPS SERVING"],
            "status": ["READ-ONLY REMOTE"],
            "rollback": ["MUTATES HOST", "STOPS SERVING"],
            "verify-rollback": ["READ-ONLY REMOTE"],
        },
        "confirmation_required": bundle.confirmation is not None,
        "confirmation_commands": ["start", "stop", "rollback"],
        "read_only_commands": ["status", "verify-rollback"],
        "target_topology": generic._target_topology(site),  # noqa: SLF001
        "claim_disclaimer": (
            "explain describes structure only; it does not claim "
            "model correctness, live validation, or acceptance."
        ),
    }


# ---------------------------------------------------------------------------
# Native execution with an invocation-local ledger
# ---------------------------------------------------------------------------


def execute_native_start(
    bundle: RuntimeBundle, site: Any, *,
    confirmation: str | None = None,
) -> dict[str, Any]:
    """Execute native bundle start with invocation-local ledger.

    Executor exceptions produce a result for every rank. Rollback continues
    after exceptions and reports attempted and missing ranks.
    """
    if any(
        svc.source_kind in PLAN_ONLY_SOURCE_KINDS for svc in bundle.services
    ):
        raise BundleError(
            "Canonical bridge execution is not supported; "
            "use the canonical launcher"
        )
    if bundle.confirmation and confirmation != bundle.confirmation:
        raise BundleError(
            f"confirmation token mismatch: expected {bundle.confirmation}"
        )
    if not bundle.confirmation:
        raise BundleError(
            "executable native bundle requires a non-null confirmation token"
        )

    ordered = topological_order(bundle.services)
    reversed_ = reverse_order(bundle.services)
    validate_service_ranks(bundle, site)
    check_container_name_collisions(bundle, site)

    ledger: set[tuple[str, int]] = set()
    results: dict[str, Any] = {"phases": [], "rollback": None}

    for svc in ordered:
        start_actions = _native_start_actions(site, svc, bundle)
        timeout = (
            svc.profile.startup_timeout_seconds if svc.profile
            else svc.structured.startup_timeout_seconds
        )
        # Every scheduled rank receives a result, including executor failures.
        start_results = _safe_execute(start_actions, timeout)
        failed_ranks = runtime.check_results(
            "start", start_results,
            runtime.execution_mode(svc.profile) if svc.profile else "generic",
        )
        started_ranks = {
            rank for rank, res in start_results.items()
            if runtime.action_succeeded("start", res)
        }
        for rank in started_ranks:
            ledger.add((svc.service_id, rank))

        phase_result = {
            "phase": "start",
            "service_id": svc.service_id,
            "results": start_results,
            "failed_ranks": sorted(failed_ranks),
        }
        results["phases"].append(phase_result)

        if failed_ranks:
            results["rollback"] = _execute_rollback(
                bundle, site, ledger, reversed_,
            )
            return results

        # Run readiness
        if svc.readiness:
            ready_actions = _readiness_actions(site, svc, bundle)
            if ready_actions:
                ready_results = _safe_execute(
                    ready_actions, svc.readiness.timeout_seconds,
                )
                ready_failed = [
                    rank for rank, res in ready_results.items()
                    if res.get("exit_code", 1) != 0
                ]
                phase_result = {
                    "phase": "readiness",
                    "service_id": svc.service_id,
                    "results": ready_results,
                    "failed_ranks": sorted(ready_failed),
                }
                results["phases"].append(phase_result)
                if ready_failed:
                    results["rollback"] = _execute_rollback(
                        bundle, site, ledger, reversed_,
                    )
                    return results

    return results


def _safe_execute(
    actions: list[runtime.RemoteAction], timeout: int,
) -> dict[int, dict]:
    """Execute actions, returning a result for every scheduled rank.

    A whole-executor exception gives every scheduled rank reserved failure code
    125. After a normal return, missing ranks receive individual failure results.
    """
    expected_ranks = {a.rank for a in actions}
    try:
        results = runtime.execute(actions, timeout)
    except Exception as exc:
        # Executor failures use reserved code 125 and include an error type.
        return {
            rank: {
                "exit_code": 125,
                "stdout": "",
                "stderr": f"executor exception: {type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
            }
            for rank in expected_ranks
        }
    # Ensure every expected rank has a result (missing = failure)
    for rank in expected_ranks:
        if rank not in results:
            results[rank] = {
                "exit_code": 125,
                "stdout": "",
                "stderr": "missing rank result",
                "error_type": "MissingResult",
            }
    # Detect and flag unexpected extra ranks (malformed executor output)
    extra_ranks = set(results) - expected_ranks
    if extra_ranks:
        for rank in extra_ranks:
            results[rank]["exit_code"] = 125
            results[rank]["stderr"] = (
                f"unexpected extra rank {rank} not in scheduled ranks"
            )
            results[rank]["error_type"] = "ExtraRank"
    return results


def _execute_rollback(
    bundle: RuntimeBundle, site: Any,
    ledger: set[tuple[str, int]],
    reversed_services: list[BundleService],
) -> dict[str, Any]:
    """Roll back only ledgered (service_id, rank) entries.

    Catches executor exceptions and continues to later services. Reports
    attempted, reported, and missing ranks.
    """
    rollback_results: dict[str, Any] = {
        "phases": [], "rollback_status": "success",
    }
    any_failed = False
    for svc in reversed_services:
        ledger_ranks = sorted(
            rank for (sid, rank) in ledger if sid == svc.service_id
        )
        if not ledger_ranks:
            continue
        stop_actions = [
            action for action in _native_stop_actions(site, svc, bundle)
            if action.rank in ledger_ranks
        ]
        timeout = (
            svc.profile.startup_timeout_seconds if svc.profile
            else svc.structured.startup_timeout_seconds
        )
        # Executor failures do not prevent later rollback phases.
        results = _safe_execute(stop_actions, timeout)
        reported_ranks = set(results)
        missing_ranks = sorted(set(ledger_ranks) - reported_ranks)
        extra_ranks = sorted(reported_ranks - set(ledger_ranks))
        phase_failed = any(
            res.get("exit_code", 0) != 0
            for res in results.values()
        )
        # Missing or unexpected ranks make rollback fail closed.
        if missing_ranks or extra_ranks:
            phase_failed = True
        if phase_failed:
            any_failed = True
        rollback_results["phases"].append({
            "service_id": svc.service_id,
            "attempted_ranks": ledger_ranks,
            "reported_ranks": sorted(reported_ranks),
            "missing_ranks": missing_ranks,
            "extra_ranks": extra_ranks,
            "results": results,
            "status": "failed" if phase_failed else "success",
        })
    if any_failed:
        rollback_results["rollback_status"] = "failed"
    return rollback_results


def execute_native_status(
    bundle: RuntimeBundle, site: Any,
) -> dict[str, Any]:
    """Execute status checks and aggregate failures in the returned status."""
    ordered = topological_order(bundle.services)
    validate_service_ranks(bundle, site)
    results: dict[str, Any] = {"phases": []}
    any_failed = False
    for svc in ordered:
        if svc.profile is None and svc.structured is None:
            continue
        actions = _native_status_actions(site, svc, bundle)
        res = _safe_execute(actions, 30)
        failed = [
            rank for rank, r in res.items()
            if r.get("exit_code", 0) != 0
        ]
        if failed:
            any_failed = True
        results["phases"].append({
            "service_id": svc.service_id,
            "results": res,
            "failed_ranks": sorted(failed),
        })
    results["status"] = "failed" if any_failed else "ok"
    return results


def execute_native_verify_rollback(
    bundle: RuntimeBundle, site: Any,
) -> dict[str, Any]:
    """Execute native verify-rollback checks.

    Exit 0 = absent; exit 1 = present; any other nonzero (2, 73, 74,
    125, missing, extra) = unknown/conservative failure.  Never classify
    a nonzero observation as proven absent.
    """
    reversed_ = reverse_order(bundle.services)
    validate_service_ranks(bundle, site)
    results: dict[str, Any] = {"phases": []}
    any_present = False
    any_unknown = False
    for svc in reversed_:
        if svc.profile is None and svc.structured is None:
            continue
        actions = _native_verify_rollback_actions(site, svc, bundle)
        res = _safe_execute(actions, 30)
        for rank, r in res.items():
            code = r.get("exit_code", 2)
            if code == 0:
                pass  # proven absent
            elif code == 1:
                any_present = True
            else:
                # 2 (unknown), 73 (ownership mismatch), 74 (daemon
                # error), 125 (executor exception), missing/extra —
                # all conservative unknown, never absent
                any_unknown = True
        results["phases"].append({
            "service_id": svc.service_id,
            "results": res,
        })
    if any_present:
        results["status"] = "present"
    elif any_unknown:
        results["status"] = "unknown"
    else:
        results["status"] = "absent"
    return results
