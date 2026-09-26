"""Discover shared native fabrics or prepare a model-bound managed TP4 fabric.

The existing ASIC planner, marker attestation and managed supervisor remain the
implementation owners. This module connects them to the profile installer.
"""
import base64
import copy
import hashlib
import importlib
import ipaddress
import json
from pathlib import Path
import re
import sys
import tempfile
import time

from runtime.common import compose, managed_deployment, profiles, qwen_mesh, setup
from runtime.host import discovery, node

ROOT = profiles.ROOT
COMPONENT = ROOT / "runtime/glm53-spark-mtp3-mesh"
MESH_UNITS = ("sparkring-mesh.service", "sparkring-*-mesh.service")
# A started mesh needs all four ranks up before its markers attach.
RING_READY_SECONDS = 240
# A ring check fails with ValueError on a fabric difference and with
# RuntimeError or OSError when a command or a /proc or /sys read fails.
CHECK_FAILURES = (ValueError, RuntimeError, OSError)


def modules():
    path = str(COMPONENT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("managed_install"), importlib.import_module("managed_units")


def _service_configs():
    """Service configurations of the default and every named managed mesh layout."""
    return [Path("/etc/sparkring/managed-mesh/service.json"),
            *sorted(Path("/etc/sparkring/deployments").glob("*/service.json"))]


def inspect_local(rank):
    """The SparkRing mesh installed on this Spark as ``{"mesh": ...}``, or ``{"mesh": None}``.

    An active mesh unit is reported with ``active`` true and its ring check:
    ``snapshot`` when the check passes, else ``problem``. Without an active
    unit, an enabled one is reported with ``active`` false, as on the Sparks
    whose mesh failed when a cabled neighbor restarted. The ring step of a
    model installation starts, restarts or repairs a reported mesh.
    """
    active, enabled = [], []
    for path in _service_configs():
        if not path.is_file():
            continue
        config = json.loads(path.read_text())
        unit = managed_deployment.validate_config_paths(config)["mesh_unit"]
        if node.call(["systemctl", "is-active", unit], accepted=(0, 3, 4)).returncode == 0:
            active.append((path, config, unit))
        elif node.call(["systemctl", "is-enabled", unit], accepted=(0, 1, 4)).stdout.strip() == "enabled":
            enabled.append((path, config, unit))
    if len(active) > 1:
        raise ValueError("Multiple active mesh owners found; inspect before selecting one")
    if not active and len(enabled) > 1:
        raise ValueError("Several mesh services are enabled and none runs; inspect before selecting one")
    if not (active or enabled):
        return {"mesh": None}
    path, config, unit = (active or enabled)[0]
    if config["rank"] != rank:
        raise ValueError("Installed mesh rank order differs from the selected head/cabling; prepare a reviewed replacement")
    site_path = Path(config["site_path"])
    manager = qwen_mesh._network().NetworkManager(site_path, rank, "/run/sparkring-mesh-read-only", require_root=False)
    reference = {"site_path": str(site_path), "site_sha256": hashlib.sha256(site_path.read_bytes()).hexdigest(),
                 "plan_sha256": manager.plan.sha256}
    address = manager.site["management_addresses"][rank]
    mesh = {"reference": reference, "host_ip": address, "interface": manager.local.management_netdev,
            "unit": unit, "config": str(path), "active": bool(active), "snapshot": None}
    if active:
        hcas = [manager.local.port(d, f).rdma_device for f in (0, 1) for d in ("clockwise", "counter_clockwise")]
        try:
            mesh["snapshot"] = qwen_mesh.check(reference, rank, hcas, manager.plan.roce_gid_index, address)
        except CHECK_FAILURES as error:
            mesh["problem"] = str(error)
    return {"mesh": mesh}


def definitions(raw_site, cluster, profile):
    """Return deterministic site/topology files and their future admission hashes."""
    card = setup.selection(profile)
    if card["nodes"] != 4:
        raise ValueError("Native managed fabric requires a four-node profile")
    selected = managed_deployment.layout(raw_site["name"])
    fabric = profiles.read_json(COMPONENT / "fabric.example.json")
    hosts = cluster["plan"]["spec"]["hosts"]
    if len(hosts) != 4:
        raise ValueError("Native fabric requires four authenticated hosts")
    for rank, host in enumerate(hosts):
        target = fabric["ranks"][rank]
        target.update(ssh_alias=host["host"], management_netdev=host["management_netdev"])
        for direction in ("clockwise", "counter_clockwise"):
            for function in (0, 1):
                role = ("cw_" if direction == "clockwise" else "ccw_") + ("secondary" if function else "primary")
                port = next(p for p in host["data_interfaces"] if p["role"] == role)
                target["ports"][direction][function].update(
                    netdev=port["netdev"], rdma_device=port["rdma_device"], mac=port["mac"],
                    ipv4_cidr=str(ipaddress.IPv4Interface(port["address"]).ip) + "/32")
    workspace = raw_site["workspace"]
    marker = profiles.read_json(COMPONENT / "host-marker-artifact.json")
    site = {"schema": "sparkring-glm53-mtp3-mesh-site/v1", "topology_file": "fabric.json",
            "management_addresses": [h["management_ip"] for h in raw_site["hosts"]],
            "model_roots": [h.get("model", workspace + "/models/" + card["model_revision"]) for h in raw_site["hosts"]],
            "cache_roots": [h.get("cache", workspace + "/cache") for h in raw_site["hosts"]],
            "bundle_root": workspace + "/mesh/artifacts", "container_prefix": "sr-" + raw_site["name"],
            "marker_binary": workspace + "/mesh/artifacts/mlx5-rdma-tx-marker",
            "marker_binary_sha256": marker["binary_sha256"], "state_root": selected["state_dir"]}
    with tempfile.TemporaryDirectory(prefix="sparkring-mesh-plan-") as directory:
        root = Path(directory)
        (root / "site.json").write_text(compose.encoded(site))
        (root / "fabric.json").write_text(compose.encoded(fabric))
        _, _, plan = qwen_mesh._network().profile.load_site(root / "site.json")
    reference = {"site_path": selected["config_dir"] + "/site.json", "site_sha256": compose.digest(compose.encoded(site)),
                 "plan_sha256": plan.sha256}
    return {"schema": "sparkring-installer-native-mesh/v1", "mode": "create", "name": raw_site["name"],
            "site": site, "topology": fabric, "reference": reference, "health_port": 9976, "replaces": []}


def select(raw_site, cluster, profile, *, fresh=False, existing_only=False, invoke=discovery.ssh):
    result = copy.deepcopy(raw_site)
    observed = [json.loads(invoke(row["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "native-mesh", "--rank", str(rank)]))["mesh"]
                for rank, row in enumerate(result["hosts"])]
    # A reported mesh may be stopped or fail its ring check; the ring step of
    # the installation starts, restarts or repairs it.
    if all(observed) and not fresh:
        references = [o["reference"] for o in observed]
        if len({(r["site_sha256"], r["plan_sha256"]) for r in references}) != 1:
            raise ValueError("Ranks disagree about their native mesh; inspect before replacing it")
        for row, mesh in zip(result["hosts"], observed, strict=True):
            row.update(fabric=mesh["reference"], fabric_ip=mesh["host_ip"], interface=mesh["interface"])
        return result
    if any(observed) and not fresh:
        raise ValueError("Only part of the native mesh is enabled or running. Use a reviewed --fresh-mesh "
                         "replacement after inspection.")
    if existing_only:
        return result
    planned = definitions(result, cluster, profile)
    if fresh:
        planned["replaces"] = [{"rank": i, "unit": o["unit"], "reference": o["reference"]} for i, o in enumerate(observed) if o]
    for row, host in zip(result["hosts"], cluster["plan"]["spec"]["hosts"], strict=True):
        row.update(fabric=planned["reference"], fabric_ip=row["management_ip"], interface=host["management_netdev"])
    result["native_mesh"] = planned
    validate(planned, result)
    return result


def validate(value, raw_site):
    if (not isinstance(value, dict) or set(value) != {"schema", "mode", "name", "site", "topology", "reference", "health_port", "replaces"}
            or value["schema"] != "sparkring-installer-native-mesh/v1" or value["mode"] != "create"
            or value["name"] != raw_site["name"] or value["health_port"] != 9976):
        raise ValueError("Invalid native mesh installation plan")
    selected = managed_deployment.layout(value["name"])
    if value["site"].get("topology_file") != "fabric.json" or value["site"].get("state_root") != selected["state_dir"]:
        raise ValueError("Native topology/state paths differ from the named installation")
    if value["reference"]["site_path"] != selected["config_dir"] + "/site.json":
        raise ValueError("Native mesh site must use its named installation path")
    if value["site"]["container_prefix"] != "sr-" + raw_site["name"]:
        raise ValueError("Native mesh must bind this deployment's containers")
    if value["site"]["marker_binary"] != raw_site["workspace"] + "/mesh/artifacts/mlx5-rdma-tx-marker":
        raise ValueError("Native marker must remain inside the deployment workspace")
    marker = profiles.read_json(COMPONENT / "host-marker-artifact.json")
    if value["site"]["marker_binary_sha256"] != marker["binary_sha256"]:
        raise ValueError("Native marker differs from the published host artifact")
    with tempfile.TemporaryDirectory(prefix="sparkring-mesh-validate-") as directory:
        path = Path(directory)
        (path / "site.json").write_text(compose.encoded(value["site"]))
        (path / "fabric.json").write_text(compose.encoded(value["topology"]))
        _, _, plan = qwen_mesh._network().profile.load_site(path / "site.json")
    if compose.digest(compose.encoded(value["site"])) != value["reference"]["site_sha256"] or plan.sha256 != value["reference"]["plan_sha256"]:
        raise ValueError("Native mesh reference differs from its planned files")
    if value["site"]["management_addresses"] != [h["fabric_ip"] for h in raw_site["hosts"]]:
        raise ValueError("Native mesh bootstrap addresses differ from model ranks")
    for old in value["replaces"]:
        if set(old) != {"rank", "unit", "reference"} or old["rank"] not in range(4) or not re.fullmatch(r"sparkring[-a-z0-9]*-mesh\.service|sparkring-mesh\.service", old["unit"]):
            raise ValueError("Replacement must identify a reviewed SparkRing mesh service")
        qwen_mesh.validate_site_reference(old["reference"])
    return value


def prepare_local(lock, rank):
    """Download/verify the host artifact and save frozen config, without attaching."""
    from scripts.deploy_stage import download_host_marker
    value = validate(lock["site_input"]["native_mesh"], lock["site_input"])
    marker = Path(value["site"]["marker_binary"])
    if any(p.is_symlink() for p in (marker, *marker.parents)):
        raise ValueError("Native artifact path contains a symlink")
    marker.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    if not marker.exists():
        contract = profiles.read_json(ROOT / "runtime/sparkring/jovian-r33/mesh-host-contract.json")
        download_host_marker(contract["marker_download_url"], marker, value["site"]["marker_binary_sha256"])
    owner, _ = modules()
    owner.external_marker_attestation(binary=marker)
    return {"ok": True}


def install_local(lock, rank, payload):
    """Install canonical units around exact stopped, admitted profile containers."""
    from scripts import installer_host
    from runtime.common import installer
    owner, units = modules()
    value = validate(lock["site_input"]["native_mesh"], lock["site_input"])
    selected = managed_deployment.layout(value["name"])
    spec = installer.specifications(lock, only_rank=rank)[0]
    actual = installer_host.owned(spec, installer_host.container(spec), installer_host.image_info(lock))
    containers = payload["containers"]
    if len(containers) != 4 or containers[rank]["Id"] != actual["Id"] or actual["State"].get("Running"):
        raise ValueError("Native mesh installation requires the exact stopped model container")
    key = base64.b64decode(payload["key"], validate=True)
    if len(key) != 32 or not re.fullmatch(r"[0-9a-f]{32}", payload["epoch"]):
        raise ValueError("Invalid native mesh authentication material")
    config_root, code_root = Path(selected["config_dir"]), Path(selected["code_dir"])
    receipt_path = config_root / "installer-owner.json"
    expected = {"deployment": lock["id"], "containers": [c["Id"] for c in containers]}
    if receipt_path.exists():
        if profiles.read_json(receipt_path) != expected:
            raise ValueError("Native mesh directory belongs to another installation")
        return {"ok": True}
    if config_root.exists() or code_root.exists() or any(p.is_symlink() for p in (config_root, *config_root.parents, code_root, *code_root.parents)):
        raise ValueError("Native installation path exists; inspect incomplete installation before recovery")
    prepare_local(lock, rank)
    with tempfile.TemporaryDirectory(prefix="sparkring-native-units-") as temporary:
        root = Path(temporary)
        (root / "site.json").write_text(compose.encoded(value["site"]))
        (root / "fabric.json").write_text(compose.encoded(value["topology"]))
        units.render(root / "site.json", containers, root / "units", str(code_root), str(config_root),
                     payload["epoch"], value["health_port"], deployment_name=value["name"])
        hashes = owner.install_code(owner.source_payloads(), code_root)
        config_root.mkdir(parents=True, mode=0o700)
        for name in ("site.json", "fabric.json"):
            (config_root / name).write_bytes((root / name).read_bytes())
        (config_root / "health.key").write_bytes(key)
        (config_root / "health.key").chmod(0o600)
        (config_root / "service.json").write_bytes((root / f"units/rank{rank}/service.json").read_bytes())
        for path in (root / f"units/rank{rank}").glob("*.service"):
            destination = Path(selected["unit_dir"]) / path.name
            if destination.exists() or destination.is_symlink():
                raise ValueError("Another native unit already exists: " + path.name)
            destination.write_bytes(path.read_bytes())
            destination.chmod(0o644)
        node.save(config_root, "source-hashes.json", hashes, mode=0o600)
        node.save(config_root, "installer-owner.json", expected, mode=0o600)
    node.call(["systemctl", "daemon-reload"])
    node.call(["systemctl", "enable", selected["mesh_unit"]])
    return {"ok": True}


def mesh_unit(site_path):
    """The systemd unit of the managed mesh whose configuration directory holds site_path."""
    default = managed_deployment.layout()
    directory = site_path.rsplit("/", 1)[0]
    if directory == default["config_dir"]:
        return default["mesh_unit"]
    prefix = "/etc/sparkring/deployments/"
    if directory.startswith(prefix):
        return managed_deployment.layout(directory[len(prefix):])["mesh_unit"]
    raise ValueError(f"{site_path} is not the site of a SparkRing managed mesh")


def _active_mesh_units(call):
    listing = call(["systemctl", "list-units", "--type=service", "--state=active", "--no-legend", "--plain",
                    *MESH_UNITS]).stdout
    return {line.split()[0] for line in listing.splitlines() if line.split()}


def _readd_address(netdev, ipv4, call):
    """Delete and add one IPv4 address with its prefix and flags, re-registering its RoCE GIDs."""
    links = json.loads(call(["ip", "-j", "-4", "addr", "show", "dev", netdev]).stdout)
    entries = [entry for link in links for entry in link.get("addr_info", []) if entry.get("local") == ipv4]
    if len(entries) != 1:
        raise ValueError(f"{netdev} does not hold its fabric address {ipv4}")
    address = f"{ipv4}/{entries[0]['prefixlen']}"
    extra = (["broadcast", entries[0]["broadcast"]] if entries[0].get("broadcast") else []) +         (["noprefixroute"] if entries[0].get("noprefixroute") else [])
    call(["ip", "addr", "del", address, "dev", netdev])
    call(["ip", "addr", "add", address, *extra, "dev", netdev])


def serve_ring(reference, rank, hcas, gid, host_ip, *, call=node.call, check=qwen_mesh.check,
               stale=qwen_mesh.stale_gid_ports, sleep=time.sleep, clock=time.monotonic):
    """Serve the pinned four-Spark mesh on this Spark and wait until its ring check passes.

    Every rank runs this at the same time before the ring check of an
    installation that reuses an existing mesh. The mesh unit is enabled, so it
    returns at each boot. An inactive unit starts. A running unit whose ring
    check fails restarts, which rebuilds routes and rules lost when a cabled
    neighbor restarted. When a port's pinned RoCE GID slot lacks its IPv4
    address, as on the neighbors of a restarted Spark, the mesh stops, the
    address is deleted and added again, and the mesh starts. A Spark on which
    another SparkRing mesh unit is active is left unchanged; the ring check
    then reports it.
    """
    unit = mesh_unit(qwen_mesh.validate_site_reference(reference)["site_path"])
    active = _active_mesh_units(call)
    others = sorted(active - {unit})
    if others:
        return {"ok": True, "unit": unit, "action": "none", "active": others}
    repaired = stale(reference, rank)
    if repaired:
        call(["systemctl", "stop", unit])
        for netdev, ipv4 in repaired:
            _readd_address(netdev, ipv4, call)
        action = "repaired"
    elif unit in active:
        try:
            check(reference, rank, hcas, gid, host_ip)
            call(["systemctl", "enable", unit])
            return {"ok": True, "unit": unit, "action": "checked", "repaired": []}
        except CHECK_FAILURES:
            call(["systemctl", "restart", unit])
            action = "restarted"
    else:
        action = "started"
    call(["systemctl", "enable", "--now", unit])
    deadline = clock() + RING_READY_SECONDS
    while True:
        try:
            check(reference, rank, hcas, gid, host_ip)
            break
        except CHECK_FAILURES as error:
            if clock() >= deadline:
                raise ValueError(f"{unit} is running, but the ring check still fails after "
                                 f"{RING_READY_SECONDS} s: {error}") from None
            sleep(3)
    return {"ok": True, "unit": unit, "action": action, "repaired": [netdev for netdev, _ in repaired]}


def operate_local(lock, rank, operation):
    value = validate(lock["site_input"]["native_mesh"], lock["site_input"])
    selected = managed_deployment.layout(value["name"])
    config = Path(selected["config_dir"]) / "service.json"
    owner = profiles.read_json(config.parent / "installer-owner.json")
    if owner["deployment"] != lock["id"]:
        raise ValueError("Native mesh owner differs")
    if operation == "mesh-installed-local":
        hashes = profiles.read_json(config.parent / "source-hashes.json")
        for name, expected in hashes.items():
            path = Path(selected["code_dir"]) / name
            if path.is_symlink() or compose.digest(path.read_bytes()) != expected:
                raise ValueError("Installed mesh source changed: " + name)
        if compose.digest((config.parent / "site.json").read_bytes()) != value["reference"]["site_sha256"]:
            raise ValueError("Installed mesh site differs")
    elif operation == "mesh-up-check":
        node.call(["systemctl", "is-active", selected["mesh_unit"]])
    elif operation == "mesh-replaced":
        for prior in value["replaces"]:
            if prior["rank"] == rank and node.call(["systemctl", "is-active", prior["unit"]], accepted=(0, 3, 4)).returncode == 0:
                raise ValueError("Previous mesh remains active")
    elif operation == "mesh-up":
        # A mesh that this deployment created is served like a reused one:
        # enabled for every boot, started or restarted, and with an address
        # re-added whose RoCE GID left the pinned index.
        row = lock["site"]["ranks"][rank]
        return serve_ring(row["fabric"], rank, row["hcas"], row["gid"], row["host_ip"])
    elif operation == "mesh-gate":
        runner = selected["code_dir"] + "/runtime/glm53-spark-mtp3-mesh/managed_service.py"
        node.call(["python3", runner, "gate", "--config", str(config), "--timeout", "60"])
    elif operation == "mesh-replace":
        for prior in value["replaces"]:
            if prior["rank"] == rank:
                if compose.digest(Path(prior["reference"]["site_path"]).read_bytes()) != prior["reference"]["site_sha256"]:
                    raise ValueError("Previous mesh site changed since replacement review")
                node.call(["systemctl", "disable", "--now", prior["unit"]])
    else:
        raise ValueError("Unsupported native mesh operation")
    return {"ok": True}
