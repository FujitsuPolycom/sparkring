"""Read-only operator preflight for the SparkRing transport.

Run on one rank before enabling any SparkRing mode. Validates
configuration and host state without touching GPUs, RDMA queues, or the
network; the single network-touching check (control-peer TCP
reachability) is opt-in via ``--connect``.
"""

from __future__ import annotations

import argparse
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


def _check_runtime_contract() -> Check:
    """Report whether the deployment-supplied query contract is importable.

    ``spark_tp4_sparse_q42_q48_contract`` declares which query rows the
    provider serves. It is part of the deployment runtime rather than this
    package, so the wheel does not vendor it and the adapter modules cannot
    be imported without it. Naming it here keeps a missing deployment
    runtime from surfacing as an unattributed import error.
    """

    if _VENDOR not in sys.path:
        sys.path.insert(0, _VENDOR)
    try:
        import spark_tp4_sparse_q42_q48_contract as contract
    except ImportError as error:
        return Check(
            "runtime-contract",
            False,
            "spark_tp4_sparse_q42_q48_contract is not importable; it is "
            "supplied by the deployment runtime, not by this package "
            f"({error})",
            "required",
        )
    return Check(
        "runtime-contract",
        True,
        f"query contract from {getattr(contract, '__file__', 'unknown')}",
        "required",
    )


def _check_widths(environ: Mapping[str, str]) -> Check:
    namespace = _vendored()
    try:
        widths = namespace.eager_allreduce_admitted_widths(environ)
    except ValueError as error:
        return Check("env-widths", False, str(error), "required")
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
    report = compat_report()
    detail = (
        f"vLLM {report['vllm_version']!r}: {report['detail']}"
    )
    return Check("vllm-compat", bool(report["ok"]), detail, "required")


def _check_peers(
    environ: Mapping[str, str], *, connect: bool
) -> Check:
    if not connect:
        return Check(
            "peer-reachability", True, "skipped (--connect)", "info"
        )
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
    contract = _check_runtime_contract()
    checks = [contract]
    # env-widths and port-namespace import the adapter modules, which import
    # the query contract. Without it they would raise instead of reporting,
    # so they are reported as blocked rather than run.
    if contract.passed:
        checks += [_check_widths(env), _check_mode(env),
                   _check_port_namespace(env)]
    else:
        checks += [
            Check(name, False, "blocked: runtime-contract failed", "required")
            for name in ("env-widths", "port-namespace")
        ]
        checks.append(_check_mode(env))
    checks += [
        _check_hca_devices(env),
        _check_native_library(env),
        _check_vllm_compat(),
        _check_peers(env, connect=connect),
    ]
    return checks


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
