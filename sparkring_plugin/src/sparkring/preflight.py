"""Read-only operator preflight for the SparkRing transport.

Run on one rank before enabling any SparkRing mode. Validates
configuration, vendored-module integrity, and host state without
touching GPUs, RDMA queues, or the network; the single network-touching
check (control-peer TCP reachability) is opt-in via ``--connect``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from sparkring.plugin import _NATIVE_LIBRARY, _VENDOR

_SYSFS_INFINIBAND = "/sys/class/infiniband"
_VALID_MODES = ("shadow", "custom", "disabled")
_DEFAULT_CONTROL_PORTS = (11000, 11001)
_CONNECT_TIMEOUT_SECONDS = 2.0
_VENDOR_MANIFEST = os.path.join(_VENDOR, "MANIFEST.json")
_HASH_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str
    severity: str  # "required" | "info"


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _vendored():
    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    import spark_tp4_port_namespace

    return spark_tp4_port_namespace


def _raised(name: str, context: str, error: Exception) -> Check:
    """Turn an exception a handler does not model into a failed Check.

    run_preflight is the sole producer of the report, so an exception
    that escapes one handler erases every other result: --json then
    prints no Check array at all. Every handler that calls into
    configuration-derived or third-party code routes what it cannot
    model here, naming the configuration it was resolving. Only
    Exception is converted; a KeyboardInterrupt still ends the run.
    """

    return Check(
        name, False, f"{context}: {type(error).__name__}: {error}", "required"
    )


def _sha256_file(path: str) -> str:
    """Hash the file's bytes as written; no newline or encoding translation."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_HASH_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _check_vendor_integrity() -> Check:
    """Compare the vendored backend modules against their manifest.

    scripts/sync_vendor.py writes MANIFEST.json beside the copies it
    makes, as ``{"modules": {"<stem>.py": "<sha256 hex>"}}``, and every
    entry is resolved relative to that file. A checkout that has not
    generated a manifest reports info: absence is not evidence of
    tampering. A manifest that is present is authoritative, so a listed
    module that is missing, unreadable, or altered is a required
    failure — the vendored copies are what every other check imports.
    """

    if not os.path.isfile(_VENDOR_MANIFEST):
        return Check(
            "vendor-integrity",
            True,
            f"no manifest at {_VENDOR_MANIFEST}",
            "info",
        )
    try:
        with open(_VENDOR_MANIFEST, "rb") as handle:
            manifest = json.loads(handle.read())
    except (OSError, ValueError) as error:
        return _raised("vendor-integrity", _VENDOR_MANIFEST, error)
    modules = manifest.get("modules") if isinstance(manifest, dict) else None
    if not isinstance(modules, dict):
        return Check(
            "vendor-integrity",
            False,
            f"{_VENDOR_MANIFEST}: no 'modules' object",
            "required",
        )
    root = os.path.dirname(_VENDOR_MANIFEST)
    problems: list[str] = []
    for name in sorted(modules):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            problems.append(f"{name}: missing")
            continue
        try:
            actual = _sha256_file(path)
        except OSError as error:
            problems.append(f"{name}: unreadable: {error}")
            continue
        if actual != modules[name]:
            problems.append(
                f"{name}: sha256 {actual} != manifest {modules[name]}"
            )
    if problems:
        return Check(
            "vendor-integrity", False, "; ".join(problems), "required"
        )
    return Check(
        "vendor-integrity",
        True,
        f"{len(modules)} vendored modules match the manifest",
        "required",
    )


def _check_query_row_provider(environ: Mapping[str, str]) -> Check:
    """Report the resolved query-row policy.

    With VLLM_SPARK_TP4_QUERY_ROW_PROVIDER unset, the generic contiguous
    policy applies and the check passes. With it set, the named module
    must import and return a valid row set; any failure is a required
    diagnostic naming the variable and module.
    """

    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    from spark_tp4_query_row_provider import (
        PROVIDER_ENV,
        provider_module_name,
        resolve_query_rows,
    )

    module_name = provider_module_name(environ)
    configured = (
        f"{PROVIDER_ENV}={module_name}"
        if module_name is not None
        else f"{PROVIDER_ENV} unset"
    )
    try:
        rows = resolve_query_rows(environ)
    except ValueError as error:
        return Check("query-row-provider", False, str(error), "required")
    except Exception as error:
        # A configured provider is third-party code, and the resolver
        # models only its ValueError contract. Whatever else it raises
        # at call time is this rank's diagnostic, not a crash.
        return _raised("query-row-provider", configured, error)
    if module_name is None:
        return Check(
            "query-row-provider",
            True,
            f"not configured; generic contiguous rows 1..{rows[-1]}",
            "required",
        )
    return Check(
        "query-row-provider",
        True,
        f"{module_name}: {len(rows)} rows, max {rows[-1]}",
        "required",
    )


def _check_widths(environ: Mapping[str, str]) -> Check:
    namespace = _vendored()
    try:
        widths = namespace.eager_allreduce_admitted_widths(environ)
    except ValueError as error:
        return Check("env-widths", False, str(error), "required")
    except Exception as error:
        return _raised("env-widths", "VLLM_SPARK_TP4_EAGER_WIDTHS", error)
    return Check(
        "env-widths",
        True,
        f"admitted widths: {', '.join(str(w) for w in widths)}",
        "required",
    )


def _check_mode(environ: Mapping[str, str]) -> Check:
    value = environ.get("VLLM_SPARK_TP4_MODE", "")
    if not value:
        return Check(
            "env-mode", True, "transport disabled (mode unset)", "required"
        )
    if value in _VALID_MODES:
        return Check("env-mode", True, f"mode: {value}", "required")
    return Check(
        "env-mode",
        False,
        f"VLLM_SPARK_TP4_MODE must be one of {_VALID_MODES}: {value!r}",
        "required",
    )


def _check_port_namespace(environ: Mapping[str, str]) -> Check:
    namespace = _vendored()
    try:
        reservations = namespace.validate_active_port_namespace(environ)
    except ValueError as error:
        return Check("port-namespace", False, str(error), "required")
    except Exception as error:
        # The reservation plan resolves the query-row policy, so a
        # provider that raises anything but ValueError reaches here too.
        return _raised(
            "port-namespace", "active control-port reservations", error
        )
    return Check(
        "port-namespace",
        True,
        f"{len(reservations)} reservations, no overlap",
        "required",
    )


def _check_hca_devices(environ: Mapping[str, str]) -> Check:
    devices = [
        (environ.get(f"SPARK_TP4_DEVICE{index}"), environ.get(
            f"SPARK_TP4_GID{index}", "3"
        ))
        for index in (0, 1)
    ]
    configured = [(d, g) for d, g in devices if d]
    if not configured:
        return Check(
            "hca-devices", True, "no SPARK_TP4_DEVICE* configured", "info"
        )
    if not os.path.isdir(_SYSFS_INFINIBAND):
        return Check(
            "hca-devices", True, "no RDMA sysfs on this host", "info"
        )
    problems: list[str] = []
    for device, gid in configured:
        device_dir = os.path.join(_SYSFS_INFINIBAND, device)
        if not os.path.isdir(device_dir):
            problems.append(f"{device}: not present")
            continue
        gid_file = os.path.join(device_dir, "ports", "1", "gids", gid)
        if not os.path.isfile(gid_file):
            problems.append(f"{device}: GID index {gid} missing")
    if problems:
        return Check("hca-devices", False, "; ".join(problems), "required")
    return Check(
        "hca-devices",
        True,
        ", ".join(f"{d} gid {g}" for d, g in configured),
        "required",
    )


def _check_native_library(environ: Mapping[str, str]) -> Check:
    path = environ.get("SPARK_TP4_LIBRARY")
    if path:
        if os.path.isfile(path):
            return Check("native-library", True, path, "required")
        return Check(
            "native-library",
            False,
            f"SPARK_TP4_LIBRARY does not exist: {path}",
            "required",
        )
    if os.path.isfile(_NATIVE_LIBRARY):
        return Check(
            "native-library",
            True,
            f"packaged: {_NATIVE_LIBRARY}",
            "required",
        )
    return Check(
        "native-library",
        False,
        "SPARK_TP4_LIBRARY unset and no packaged "
        "libspark_transport_capi.so in this wheel",
        "required",
    )


def _check_vllm_compat() -> Check:
    try:
        from sparkring._compat import compat_report
    except ImportError:
        return Check(
            "vllm-compat", True, "compat module unavailable", "info"
        )
    try:
        report = compat_report()
        detail = (
            f"vLLM {report['vllm_version']!r}: {report['detail']}"
        )
        passed = bool(report["ok"])
    except Exception as error:
        return _raised("vllm-compat", "sparkring._compat", error)
    return Check("vllm-compat", passed, detail, "required")


def _check_peers(
    environ: Mapping[str, str], *, connect: bool
) -> Check:
    if not connect:
        return Check(
            "peer-reachability", True, "skipped (--connect)", "info"
        )
    try:
        peers = [
            (
                environ.get(f"SPARK_TP4_PEER{index}"),
                int(
                    environ.get(
                        f"SPARK_TP4_CONTROL_PORT{index}",
                        str(_DEFAULT_CONTROL_PORTS[index]),
                    )
                ),
            )
            for index in (0, 1)
        ]
    except Exception as error:
        return _raised(
            "peer-reachability", "SPARK_TP4_CONTROL_PORT0/1", error
        )
    configured = [(p, port) for p, port in peers if p]
    if not configured:
        return Check(
            "peer-reachability", True, "no SPARK_TP4_PEER* configured",
            "info",
        )
    problems: list[str] = []
    for peer, port in configured:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(_CONNECT_TIMEOUT_SECONDS)
        try:
            sock.connect((peer, port))
        except ConnectionRefusedError:
            pass  # refused proves routability
        except OSError as error:
            problems.append(f"{peer}:{port}: {error}")
        finally:
            sock.close()
    if problems:
        return Check(
            "peer-reachability", False, "; ".join(problems), "required"
        )
    return Check(
        "peer-reachability",
        True,
        ", ".join(f"{p}:{port}" for p, port in configured),
        "required",
    )


def run_preflight(
    environ: Mapping[str, str] | None = None, *, connect: bool = False
) -> list[Check]:
    env = _environment(environ)
    return [
        _check_vendor_integrity(),
        _check_query_row_provider(env),
        _check_widths(env),
        _check_mode(env),
        _check_port_namespace(env),
        _check_hca_devices(env),
        _check_native_library(env),
        _check_vllm_compat(),
        _check_peers(env, connect=connect),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sparkring-preflight",
        description="Read-only SparkRing transport preflight.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--connect",
        action="store_true",
        help="enable the control-peer TCP reachability check",
    )
    arguments = parser.parse_args(argv)
    # Collecting the checks imports vLLM, which writes progress lines to
    # stdout. Under --json stdout carries one document and nothing else, so
    # the checks run with stdout diverted to stderr; the diagnostics stay
    # visible to a person and stay out of the parsed stream.
    if arguments.as_json:
        with contextlib.redirect_stdout(sys.stderr):
            checks = run_preflight(connect=arguments.connect)
    else:
        checks = run_preflight(connect=arguments.connect)
    if arguments.as_json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        for check in checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"{status} {check.name} — {check.detail}")
    failed = [
        check
        for check in checks
        if check.severity == "required" and not check.passed
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
