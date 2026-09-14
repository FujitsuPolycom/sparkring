"""Read-only admission of an explicitly pinned TP4 mesh network snapshot."""

from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "runtime/common/qwen_mesh.py",
    "runtime/common/__init__.py",
    "runtime/common/candidate.py",
    "runtime/common/r35.py",
    "runtime/glm53-spark-mtp3-mesh/managed_network.py",
    "runtime/glm53-spark-mtp3-mesh/inspect_fabric.py",
    "runtime/glm53-spark-mtp3-mesh/profile.py",
    "runtime/glm53-spark-mtp3-mesh/pins.json",
    "runtime/glm53-flash-jj-r8-gb10/pins.json",
    "spark_transport/fabric/cx7_hairpin_diagonal/__init__.py",
    "spark_transport/fabric/cx7_hairpin_diagonal/fabric.py",
    "integrations/vllm/rocenante/build_bundle.py",
)


def validate_site_reference(reference):
    """Validate host-path syntax and hashes without reading the private site."""
    if not isinstance(reference, dict) or set(reference) != {
        "site_path",
        "site_sha256",
        "plan_sha256",
    }:
        raise ValueError(
            "Fabric reference requires only site_path, site_sha256 and plan_sha256"
        )
    value = reference["site_path"]
    if (
        not isinstance(value, str)
        or not PurePosixPath(value).is_absolute()
        or value == "/"
        or str(PurePosixPath(value)) != value
        or ".." in PurePosixPath(value).parts
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError(
            "Fabric site_path must be a normalized absolute Linux file path"
        )
    if any(
        not isinstance(reference[key], str)
        or not re.fullmatch("[0-9a-f]{64}", reference[key])
        for key in ("site_sha256", "plan_sha256")
    ):
        raise ValueError(
            "Fabric site and canonical plan require exact lowercase SHA-256 hashes"
        )
    return dict(reference)


@lru_cache(maxsize=1)
def _network():
    path = ROOT / "runtime/glm53-spark-mtp3-mesh/managed_network.py"
    spec = importlib.util.spec_from_file_location("qwen_mesh_managed_network", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Host:
    """Local read operations; the injected test host implements this interface."""

    def run(self, argv):
        return _network().command(argv)

    def read_bytes(self, path):
        return Path(path).read_bytes()

    def read_text(self, path):
        return self.read_bytes(path).decode()

    def process_ids(self):
        return sorted(
            int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit()
        )

    def readiness(self, pid):
        # Native managed markers flush their attachment record to stdout once.
        # Read the live descriptor, not a guessed supervisor state directory.
        # Nonblocking open plus fstat refuses pipes without consuming output.
        path = Path("/proc") / str(pid) / "fd/1"
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError(
                    "Managed marker requires a readable regular stdout readiness log"
                )
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError("Managed marker readiness log exceeds the supported size")
        return json.loads(raw)


def _argv(host, pid):
    return (
        host.read_bytes(Path("/proc") / str(pid) / "cmdline")
        .decode()
        .rstrip("\0")
        .split("\0")
    )


def _start_ticks(host, pid):
    text = host.read_text(Path("/proc") / str(pid) / "stat")
    fields = text.rsplit(")", 1)[-1].split()
    if ")" not in text or len(fields) < 20 or fields[0] not in {"R", "S", "D", "I"}:
        raise ValueError("Managed marker is absent, stopped or exited")
    ticks = int(fields[19])
    if ticks < 0:
        raise ValueError("Managed marker process start identity is invalid")
    return ticks


def _markers(host, site, plan, rank):
    binary = site["marker_binary"]
    expected_hash = site["marker_binary_sha256"]
    if hashlib.sha256(host.read_bytes(Path(binary))).hexdigest() != expected_hash:
        raise ValueError("Native marker binary digest differs from the pinned site")
    expected = {
        marker.rdma_device for marker in plan.markers if marker.source_rank == rank
    }
    if len(expected) != 2:
        raise ValueError("TP4 mesh requires exactly two planned managed source markers")
    observed = []
    for pid in host.process_ids():
        try:
            argv = _argv(host, pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeError):
            continue
        device = (
            argv[argv.index("--device") + 1]
            if "--device" in argv and argv.index("--device") + 1 < len(argv)
            else None
        )
        if device not in expected:
            continue
        reserved_source = (
            "--source-port" in argv
            and argv.index("--source-port") + 1 < len(argv)
            and argv[argv.index("--source-port") + 1] == "65535"
        )
        if argv[0] != binary and not ("--attach" in argv and reserved_source):
            continue
        wanted = [
            binary,
            "--device",
            device,
            "--source-port",
            "65535",
            "--replacement-ethertype",
            "0x88b5",
            "--attach",
            "--managed",
        ]
        if argv != wanted:
            raise ValueError(
                "Source marker requires the exact pinned persistent managed arguments"
            )
        start = _start_ticks(host, pid)
        executable = host.read_bytes(Path("/proc") / str(pid) / "exe")
        if hashlib.sha256(executable).hexdigest() != expected_hash:
            raise ValueError(
                "Live source marker executable differs from the pinned binary"
            )
        ready = host.readiness(pid)
        if (
            not isinstance(ready, dict)
            or ready.get("device") != device
            or ready.get("attached") is not True
            or ready.get("managed") is not True
            or ready.get("rdma_tx_matcher") is not True
            or ready.get("lifetime_seconds", "missing") is not None
            or type(ready.get("source_port")) is not int
            or ready.get("source_port") != 65535
            or ready.get("action") != "ethertype_rewrite"
            or type(ready.get("requested_run_seconds")) is not int
            or ready.get("requested_run_seconds") != 0
        ):
            raise ValueError(
                "Native marker has not confirmed the planned persistent attachment"
            )
        if _argv(host, pid) != argv or _start_ticks(host, pid) != start:
            raise ValueError(
                "Managed marker process identity changed during inspection"
            )
        observed.append(
            {"pid": pid, "start_ticks": start, "device": device, "attached": True}
        )
    if (
        len(observed) != len(expected)
        or {item["device"] for item in observed} != expected
    ):
        raise ValueError(
            "Expected exactly one live attached managed marker on each planned device"
        )
    return observed


def check(reference, rank, hcas, gid, host_ip, *, host=None):
    """Check local fabric and persistent attachment identities without changing state.

    hcas must list primary clockwise/counterclockwise then secondary
    clockwise/counterclockwise. The pinned site's management address is also
    the requested rank bootstrap address. The caller checks all four hosts.
    This snapshot does not attest end-to-end RC traffic, ongoing supervision,
    or compatibility with a separately running model's lifecycle owner.
    """
    reference = validate_site_reference(reference)
    if type(rank) is not int or rank not in range(4):
        raise ValueError("TP4 mesh rank must be an integer from zero through three")
    host = Host() if host is None else host
    site_path = Path(reference["site_path"])
    if (
        hashlib.sha256(host.read_bytes(site_path)).hexdigest()
        != reference["site_sha256"]
    ):
        raise ValueError("Fabric site differs from its pinned SHA-256")
    network = _network()
    # __init__ and check do not acquire the mutation lock or create a journal.
    manager = network.NetworkManager(
        site_path,
        rank,
        "/run/sparkring-qwen-mesh-read-only",
        runner=host.run,
        read_text=host.read_text,
        require_root=False,
    )
    if manager.plan.sha256 != reference["plan_sha256"]:
        raise ValueError("Fabric canonical plan differs from its pinned SHA-256")
    if (
        hashlib.sha256(host.read_bytes(site_path)).hexdigest()
        != reference["site_sha256"]
    ):
        raise ValueError("Fabric site changed while loading the canonical plan")
    expected_hcas = [
        manager.local.port(direction, function).rdma_device
        for function in (0, 1)
        for direction in ("clockwise", "counter_clockwise")
    ]
    if not isinstance(hcas, list) or hcas != expected_hcas:
        raise ValueError(
            "TP4 HCA order differs from the complete dual-domain fabric inventory"
        )
    if type(gid) is not int or gid != manager.plan.roce_gid_index:
        raise ValueError("TP4 GID index differs from the pinned fabric")
    if host_ip != manager.site["management_addresses"][rank]:
        raise ValueError(
            "Rank bootstrap address differs from the fabric management identity"
        )
    network_result = manager.check()
    markers = _markers(host, manager.site, manager.plan, rank)
    return {
        "schema": "sparkring-qwen-mesh-snapshot/v1",
        "ready": True,
        "rank": rank,
        "site_sha256": reference["site_sha256"],
        "plan_sha256": manager.plan.sha256,
        "hcas": list(hcas),
        "gid": gid,
        "host_ip": host_ip,
        "network_objects": network_result["objects"],
        "markers": markers,
        "end_to_end_rc_qualified": False,
        "supervision_verified": False,
        "scope": "Read-only local network and live attachment snapshot; no end-to-end RC or ongoing supervision qualification.",
    }
