"""Discover shared native fabrics or prepare a model-bound managed TP4 fabric.

The existing ASIC planner, marker attestation and managed supervisor remain the
implementation owners. This module connects them to the profile installer, and
installs a deployment's mesh code over a stopped mesh whose installed code
differs (update_code).

On a four-Spark cycle whose fabric document records a relay table that boot
units restore (``relays.persistent_reference``), that table carries the
prepared transport's two-hop paths: ``select`` gives each rank a reference to
the fabric document instead of a mesh site and plans no mesh service, and the
ring operations (``stop_ring``, ``ring_stopped``, ``serve_ring``) check the
table instead of a mesh unit. A cluster without such a document keeps the
per-deployment mesh service.
"""
import base64
import concurrent.futures
import copy
import hashlib
import http.client
import importlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import time

from runtime.common import compose, managed_deployment, profiles, qwen_mesh, setup
from runtime.host import discovery, node, relays, roce_gid

ROOT = profiles.ROOT
COMPONENT = ROOT / "runtime/glm53-spark-mtp3-mesh"
MESH_UNITS = ("sparkring-mesh.service", "sparkring-*-mesh.service")
# Files of mesh installations that a replacement of the same name took over.
REPLACED_ROOT = Path("/var/lib/sparkring/replaced-meshes")
# A started mesh needs all four ranks up before its markers attach.
RING_READY_SECONDS = 240
# A ring check fails with ValueError on a fabric difference and with
# RuntimeError or OSError when a command or a /proc or /sys read fails.
CHECK_FAILURES = (ValueError, RuntimeError, OSError)
# Written to a mesh's configuration directory each time a deployment serves
# that mesh: the digests of the mesh code and unit files that the deployment's
# SparkRing source provides (code_warnings compares them with the installed ones).
DEPLOYMENT_SOURCE = "deployment-source.json"
DEPLOYMENT_SOURCE_SCHEMA = "sparkring-mesh-deployment-source/v1"
# Receipts of installed mesh code: installed_source_sha256 of installation.json
# (written by managed_install.py) and source-hashes.json (written by install_local).
INSTALLATION_RECEIPT = "installation.json"
SOURCE_HASHES = "source-hashes.json"
# systemctl is-active states of a unit with no process.
STOPPED_STATES = ("inactive", "failed")
# The mesh services that park_local stopped and disabled on this Spark while two-Spark
# models serve on the ring's halves. inspect_local reports them as enabled, so a
# four-rank deployment reuses them, and serve_ring enables them again and removes
# them from this record.
PARKED = "/etc/sparkring/mesh-parked.json"
PARKED_SCHEMA = "sparkring-parked-mesh/v1"


def modules():
    path = str(COMPONENT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("managed_install"), importlib.import_module("managed_units")


def _service_configs():
    """Service configurations of the default and every named managed mesh layout."""
    return [Path("/etc/sparkring/managed-mesh/service.json"),
            *sorted(Path("/etc/sparkring/deployments").glob("*/service.json"))]


def parked_units(*, root="/"):
    """The mesh units this Spark's park record (PARKED) lists; empty without a record."""
    path = node.location(root, PARKED)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as error:
        raise ValueError(f"{path} cannot be read: {error}") from None
    if not isinstance(value, dict) or value.get("schema") != PARKED_SCHEMA or not isinstance(value.get("units"), list):
        raise ValueError(f"{path} is not a {PARKED_SCHEMA} record")
    return [unit for unit in value["units"] if isinstance(unit, str)]


def _save_parked(units, *, root="/"):
    path = node.location(root, PARKED)
    if units:
        node.save(path.parent, path.name, {"schema": PARKED_SCHEMA, "units": sorted(set(units))})
    else:
        path.unlink(missing_ok=True)


def _unpark(unit, *, root="/"):
    """Remove ``unit`` from the park record; a unit it does not list leaves the record unchanged."""
    units = parked_units(root=root)
    if unit in units:
        _save_parked([name for name in units if name != unit], root=root)


def _mesh_units():
    """The mesh unit of every managed mesh configuration installed on this Spark."""
    units = []
    for path in _service_configs():
        if path.is_file():
            config = json.loads(path.read_text())
            units.append(managed_deployment.validate_config_paths(config)["mesh_unit"])
    return units


def park_local(*, call=node.call, root="/"):
    """Stop and disable every SparkRing mesh service on this Spark, recording each one in PARKED.

    A two-Spark model on half of a four-Spark ring uses ports on which the
    ring's mesh forwards and rewrites RDMA traffic, so no mesh may run on a
    Spark of either half. The record is written before any unit stops, so an
    interrupted run leaves every stopped unit recorded. A unit that failed
    earlier keeps no failed state, so status and automatic recovery do not
    report a mesh failure while the ring serves two-Spark models. The
    fabric's addresses, routes and forwarding and the ConnectX hairpin setting
    are not changed. Repeating it changes nothing.
    """
    units = [unit for unit in _mesh_units()
             if call(["systemctl", "is-active", unit], accepted=(0, 3, 4)).returncode == 0
             or call(["systemctl", "is-enabled", unit], accepted=(0, 1, 4)).stdout.strip() == "enabled"]
    recorded = sorted(set(parked_units(root=root)) | set(units))
    _save_parked(recorded, root=root)
    if units:
        call(["systemctl", "disable", "--now", *units])
    for unit in recorded:
        call(["systemctl", "reset-failed", unit], accepted=(0, 1, 5))
    return {"ok": True, "parked": units, "recorded": recorded}


def parked_local(*, call=node.call):
    """Verify park_local without changing anything: no SparkRing mesh service runs or is enabled here."""
    running = sorted(_active_mesh_units(call))
    if running:
        raise ValueError("Mesh services still run on this Spark: " + ", ".join(running))
    enabled = [unit for unit in _mesh_units()
               if call(["systemctl", "is-enabled", unit], accepted=(0, 1, 4)).stdout.strip() == "enabled"]
    if enabled:
        raise ValueError("Mesh services stay enabled on this Spark: " + ", ".join(enabled))
    return {"ok": True}


def inspect_local(rank):
    """The SparkRing mesh installed on this Spark as ``{"mesh": ...}``, or ``{"mesh": None}``.

    An active mesh unit is reported with ``active`` true and its ring check:
    ``snapshot`` when the check passes, else ``problem``. Without an active
    unit, an enabled one is reported with ``active`` false, as on the Sparks
    whose mesh failed when a cabled neighbor restarted. A unit that
    park_local stopped and disabled counts as enabled and is reported with
    ``parked`` true. The ring step of a model installation starts, restarts
    or repairs a reported mesh.
    """
    active, enabled = [], []
    parked = parked_units()
    for path in _service_configs():
        if not path.is_file():
            continue
        config = json.loads(path.read_text())
        unit = managed_deployment.validate_config_paths(config)["mesh_unit"]
        if node.call(["systemctl", "is-active", unit], accepted=(0, 3, 4)).returncode == 0:
            active.append((path, config, unit))
        elif (node.call(["systemctl", "is-enabled", unit], accepted=(0, 1, 4)).stdout.strip() == "enabled"
              or unit in parked):
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
    if not active and unit in parked:
        mesh["parked"] = True
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


def select(raw_site, cluster, profile, *, fresh=False, existing_only=False, invoke=discovery.ssh, state=None):
    """The deployment site of a four-Spark profile with each rank's ``fabric`` reference.

    When the cluster's relay table is persistent (``relays.persistent_reference``
    on Node A's state directory ``state``), every rank refers to the fabric
    document, its bootstrap address is its management address as with a
    mesh, and no mesh service is planned. ``existing_only`` (the managed GLM
    backend, which owns its own mesh) skips that. Otherwise an existing mesh
    is reused, or with ``fresh`` replaced, or a mesh service is planned.
    """
    result = copy.deepcopy(raw_site)
    if not existing_only:
        from runtime.host import controller
        reference = relays.persistent_reference(controller.STATE if state is None else state, cluster)
        if reference is not None:
            for row, host in zip(result["hosts"], cluster["plan"]["spec"]["hosts"], strict=True):
                row.update(fabric=dict(reference), fabric_ip=row["management_ip"], interface=host["management_netdev"])
            return result
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


def _install_context(lock, rank, payload):
    """The validated plan, layout, key and expected receipt of this Spark's mesh installation."""
    from scripts import installer_host
    from runtime.common import installer
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
    expected = {"deployment": lock["id"], "containers": [c["Id"] for c in containers]}
    return value, selected, key, expected


def _container_running(container_id):
    """Whether a container with this ID exists on this Spark and runs."""
    result = node.call(["docker", "inspect", "--format", "{{.State.Running}}", container_id], accepted=(0, 1))
    return result.returncode == 0 and result.stdout.strip() == "true"


def takeover_problem(value, selected, rank, receipt, deployment):
    """Why this Spark's installed mesh of the same name may not be replaced, or None.

    A new deployment with an earlier one's site name (the same cluster,
    profile and instance) installs its mesh under the same name. It takes over
    the earlier installation only when its reviewed plan lists this Spark's
    unit (``--fresh-mesh`` records every observed mesh unit in ``replaces``),
    the earlier site is unchanged since that review, and the earlier model
    container on this Spark is stopped (by down) or removed.

    ``deployment`` is this deployment's lock ID. An installation it owns whose
    receipt names other containers, which were created again for example
    after docker container prune, is installed again when its site is this
    deployment's planned site and the named container on this Spark is
    stopped or removed.
    """
    if isinstance(receipt, dict) and receipt.get("deployment") == deployment:
        reference = value["reference"]
        changed = "The installed mesh site differs from this deployment's plan"
    else:
        reviewed = [prior for prior in value["replaces"] if prior["rank"] == rank and prior["unit"] == selected["mesh_unit"]]
        if not reviewed or reviewed[0]["reference"]["site_path"] != selected["config_dir"] + "/site.json":
            return ("Native mesh directory belongs to another installation. Plan its replacement with "
                    "sparkring up PROFILE --fresh-mesh and review it; a replacement takes over an installation "
                    "whose model container is stopped or removed.")
        reference = reviewed[0]["reference"]
        changed = "Previous mesh site changed since replacement review"
    site = Path(reference["site_path"])
    if site.is_symlink() or not site.is_file() or compose.digest(site.read_bytes()) != reference["site_sha256"]:
        return changed
    containers = receipt.get("containers") if isinstance(receipt, dict) else None
    if not isinstance(containers, list) or len(containers) != 4 or not all(isinstance(c, str) for c in containers):
        return "The installed mesh's receipt is incomplete; inspect it before replacing the installation"
    if _container_running(containers[rank]):
        return ("The model container of the installation being replaced still runs on this Spark. "
                "Stop that deployment with sparkring down first.")
    return None


def set_aside(selected, name):
    """Stop a replaced installation's units and move its files out of the installed tree.

    The files are moved, not deleted, to a directory under REPLACED_ROOT.
    They leave /etc/sparkring/deployments because inspect_local reads every
    configuration there, and the unit directory because the new installation
    writes units with the same names.
    """
    units = [selected[key] for key in ("model_unit", "liveness_unit", "mesh_unit")]
    paths = [Path(selected["unit_dir"]) / unit for unit in units]
    present = [unit for unit, path in zip(units, paths) if path.exists()]
    if present:
        node.call(["systemctl", "disable", "--now", *present])
    _unpark(selected["mesh_unit"])
    target = REPLACED_ROOT / f"{name}-{time.time_ns()}"
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError("The holding path for replaced meshes contains a symlink")
    (target / "units").mkdir(parents=True, mode=0o700)
    for source, destination in ((Path(selected["config_dir"]), target / "config"), (Path(selected["code_dir"]), target / "code")):
        if source.is_symlink():
            raise ValueError("Native installation path is a symlink: " + str(source))
        if source.exists():
            shutil.move(str(source), str(destination))
    for path in paths:
        if path.exists() or path.is_symlink():
            shutil.move(str(path), str(target / "units" / path.name))
    node.call(["systemctl", "daemon-reload"])
    return target


def install_check_local(lock, rank, payload):
    """Whether install_local takes over an installation on this Spark; raises when it may not.

    The installer runs this on every rank before installing on any, so a
    refusal changes nothing.
    """
    value, selected, _, expected = _install_context(lock, rank, payload)
    receipt_path = Path(selected["config_dir"]) / "installer-owner.json"
    if not receipt_path.exists():
        return {"ok": True, "takeover": False}
    receipt = profiles.read_json(receipt_path)
    if receipt == expected:
        return {"ok": True, "takeover": False}
    problem = takeover_problem(value, selected, rank, receipt, expected["deployment"])
    if problem:
        raise ValueError(problem)
    return {"ok": True, "takeover": True}


def install_local(lock, rank, payload):
    """Install canonical units around exact stopped, admitted profile containers."""
    owner, units = modules()
    value, selected, key, expected = _install_context(lock, rank, payload)
    containers = payload["containers"]
    config_root, code_root = Path(selected["config_dir"]), Path(selected["code_dir"])
    receipt_path = config_root / "installer-owner.json"
    if receipt_path.exists():
        receipt = profiles.read_json(receipt_path)
        if receipt == expected:
            return {"ok": True}
        problem = takeover_problem(value, selected, rank, receipt, expected["deployment"])
        if problem:
            raise ValueError(problem)
        set_aside(selected, value["name"])
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


def mesh_layout(site_path):
    """The managed layout (managed_deployment.layout) whose configuration directory holds site_path."""
    default = managed_deployment.layout()
    directory = site_path.rsplit("/", 1)[0]
    if directory == default["config_dir"]:
        return default
    prefix = "/etc/sparkring/deployments/"
    if directory.startswith(prefix):
        return managed_deployment.layout(directory[len(prefix):])
    raise ValueError(f"{site_path} is not the site of a SparkRing managed mesh")


def mesh_unit(site_path):
    """The systemd unit of the managed mesh whose configuration directory holds site_path."""
    return mesh_layout(site_path)["mesh_unit"]


def _active_mesh_units(call):
    listing = call(["systemctl", "list-units", "--type=service", "--state=active", "--no-legend", "--plain",
                    *MESH_UNITS]).stdout
    return {line.split()[0] for line in listing.splitlines() if line.split()}


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _file_digest(path):
    """The SHA-256 of a regular file that is not a symlink, else None."""
    return _digest(path.read_bytes()) if path.is_file() and not path.is_symlink() else None


def _deployment_files(selected):
    """This SparkRing source's mesh files for the mesh installed at layout ``selected``.

    Returns (owner, service, code, units): ``owner`` and ``service`` are this
    source's managed_install and managed_service modules, ``code`` maps each
    path of the source allowlist (managed_install.SOURCE_FILES) to the bytes
    installed under the code directory, and ``units`` maps each unit file name
    to the text this source renders for the installed configuration: the same
    container, and a liveness unit only where one is installed.
    """
    config = json.loads((Path(selected["config_dir"]) / "service.json").read_text())
    owner, units = modules()
    liveness = (Path(selected["unit_dir"]) / selected["liveness_unit"]).is_file()
    rendered = units.unit_text(selected["code_dir"], selected["config_dir"], config["container_id"],
                               host_liveness=liveness, deployment_name=config.get("deployment_name"))
    return owner, units.service, owner.source_payloads(), {name: text.encode() for name, text in rendered.items()}


def _relative(name):
    """Whether a receipt entry names a path inside the code directory."""
    if not isinstance(name, str) or "\\" in name:
        return False
    path = PurePosixPath(name)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts


def _differences(selected, code, units):
    """(changed code paths, obsolete code paths, changed unit names) of the mesh at ``selected``.

    A code path is obsolete when an installation receipt lists it and this
    source's allowlist does not. Other files under the code directory are not
    SparkRing's and are not compared.
    """
    root, config, unit_dir = Path(selected["code_dir"]), Path(selected["config_dir"]), Path(selected["unit_dir"])
    listed = set()
    if (config / SOURCE_HASHES).is_file():
        listed |= set(json.loads((config / SOURCE_HASHES).read_text()))
    if (config / INSTALLATION_RECEIPT).is_file():
        listed |= set(json.loads((config / INSTALLATION_RECEIPT).read_text()).get("installed_source_sha256") or {})
    changed = [name for name, content in code.items() if _file_digest(root / name) != _digest(content)]
    obsolete = sorted(name for name in listed - set(code) if _relative(name) and (root / name).is_file())
    unit_names = [name for name, content in units.items() if _file_digest(unit_dir / name) != _digest(content)]
    return changed, obsolete, unit_names


def _record_source(selected, owner, code, units):
    """Save DEPLOYMENT_SOURCE for the mesh at ``selected`` unless it already holds these digests."""
    record = {"schema": DEPLOYMENT_SOURCE_SCHEMA, "source_root": str(owner.ROOT),
              "code_sha256": owner.source_hashes(code),
              "unit_sha256": {name: _digest(content) for name, content in units.items()}}
    path = Path(selected["config_dir"]) / DEPLOYMENT_SOURCE
    try:
        if json.loads(path.read_text()) == record:
            return
    except (OSError, ValueError):
        pass
    node.save(selected["config_dir"], DEPLOYMENT_SOURCE, record, mode=0o600)


def _stopped(selected, call):
    """Whether the installed mesh, model and liveness units of ``selected`` all have no process.

    The mesh unit is always asked about; the model and liveness units only
    where their unit files are installed (a liveness unit runs on rank 0 only).
    """
    names = [selected["mesh_unit"]] + [selected[key] for key in ("model_unit", "liveness_unit")
                                       if (Path(selected["unit_dir"]) / selected[key]).exists()]
    states = call(["systemctl", "is-active", *names], accepted=(0, 3, 4)).stdout.split()
    return len(states) == len(names) and all(state in STOPPED_STATES for state in states)


def peer_states(service, config, site, rank):
    """The mesh identity that each other rank's running supervisor reports, by rank.

    The value is None where nothing listens on the rank's health port (the
    connection is refused), and "unknown" where the rank does not answer or
    its answer is not an authenticated health body of this installation.
    """
    key = service.read_key(config["key_file"])

    def observe(peer):
        try:
            return service.peer_body(site["management_addresses"][peer], config["health_port"], key, peer,
                                     config["epoch"])["identity"]
        except ConnectionRefusedError:
            return None
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, http.client.HTTPException):
            return "unknown"
    peers = [peer for peer in range(4) if peer != rank]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(peers)) as pool:
        return dict(zip(peers, pool.map(observe, peers)))


def _install_files(selected, owner, code, units, changed, obsolete, unit_names, call):
    """Replace the differing code and unit files of a stopped mesh and update its receipts.

    Each replacement file is first written beside its target and renamed over it,
    so an error while writing leaves every installed file unchanged.
    """
    code_root, config_root, unit_dir = (Path(selected[key]) for key in ("code_dir", "config_dir", "unit_dir"))
    targets = ([(code_root / name, code[name]) for name in changed]
               + [(unit_dir / name, units[name]) for name in unit_names])
    for path in [target for target, _ in targets] + [code_root / name for name in obsolete]:
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Mesh file path contains a symlink: " + str(path))
    staged = []
    try:
        for target, content in targets:
            created = [directory for directory in reversed(target.parents) if not directory.exists()]
            target.parent.mkdir(parents=True, exist_ok=True)
            for directory in created:
                directory.chmod(0o755)
            temporary = target.with_name("." + target.name + ".sparkring-new")
            temporary.unlink(missing_ok=True)
            with temporary.open("xb") as stream:
                staged.append((temporary, target))
                stream.write(content)
            temporary.chmod(0o644)
    except BaseException:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    for temporary, target in staged:
        os.replace(temporary, target)
    for name in obsolete:
        (code_root / name).unlink()
    hashes = owner.source_hashes(code)
    if (config_root / SOURCE_HASHES).is_file():
        node.save(selected["config_dir"], SOURCE_HASHES, hashes, mode=0o600)
    if (config_root / INSTALLATION_RECEIPT).is_file():
        receipt = json.loads((config_root / INSTALLATION_RECEIPT).read_text())
        receipt.update(source_files=list(code), source_hashes=hashes, installed_source_sha256=hashes,
                       source_root=str(owner.ROOT), units={name: content.decode() for name, content in units.items()},
                       unit_hashes={name: _digest(content) for name, content in units.items()})
        node.save(selected["config_dir"], INSTALLATION_RECEIPT, receipt, mode=0o600)
    if unit_names:
        call(["systemctl", "daemon-reload"])
        # reenable recreates the mesh unit's [Install] links from the file just written.
        call(["systemctl", "reenable", selected["mesh_unit"]])
    remaining = [name for part in _differences(selected, code, units) for name in part]
    if remaining:
        raise ValueError("Mesh files still differ after their update: " + ", ".join(remaining))


def update_code(selected, rank, *, call=node.call, peers=peer_states):
    """Record this deployment's mesh files and install them over a stopped mesh whose files differ.

    ``selected`` is the mesh's managed layout; this SparkRing source is the
    deployment's own. The digests of this source's mesh code and unit files
    are saved as DEPLOYMENT_SOURCE in the mesh's configuration directory,
    where node status compares them with the installed files. Differing files
    are installed only when all of these hold:

    - systemctl reports the installed mesh, model and liveness units inactive
      or failed, so no supervisor, gate or liveness process runs the installed
      code;
    - this source's managed_service accepts the installed configuration;
    - every other rank's supervisor that runs reports the mesh identity that
      this source gives this configuration. The identity covers the
      supervisor's source files, and ranks with different identities never
      form the four-rank group, so a Spark keeps its installed code while
      another Spark runs other code.

    The code files are the source allowlist; installed files that the
    receipts list but the allowlist does not are removed, and other files in
    the code directory are left alone. Unit files are rendered by this
    source's managed_units for the installed container, then systemd reloads
    them. The receipts that record installed code (source-hashes.json, which
    the mesh-installed-local check verifies, and installation.json) are
    updated with the digests of the installed files.
    """
    try:
        owner, service, code, units = _deployment_files(selected)
        _record_source(selected, owner, code, units)
        changed, obsolete, unit_names = _differences(selected, code, units)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"refreshed": False, "reason": "mesh files not compared: " + str(error)}
    if not (changed or obsolete or unit_names):
        return {"refreshed": False, "reason": "unchanged"}
    if not _stopped(selected, call):
        return {"refreshed": False, "reason": "running"}
    try:
        config, site, _, _, identity = service.load_config(Path(selected["config_dir"]) / "service.json")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"refreshed": False, "reason": "this source does not accept the installed configuration: " + str(error)}
    try:
        others = sorted(peer for peer, state in peers(service, config, site, rank).items()
                        if state is not None and state != identity)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"refreshed": False, "reason": "other ranks' mesh code not observed: " + str(error)}
    if others:
        return {"refreshed": False, "reason": "other ranks run other mesh code", "ranks": others}
    # Checked again just before writing: the peer probe can take seconds.
    if not _stopped(selected, call):
        return {"refreshed": False, "reason": "running"}
    _install_files(selected, owner, code, units, changed, obsolete, unit_names, call)
    return {"refreshed": True, "files": changed, "removed": obsolete, "units": unit_names}


def code_warnings(*, root="/"):
    """Status warnings for the managed meshes on this Spark whose files differ from their deployment's.

    update_code saves, in each mesh's configuration directory, the digests of
    the mesh code and unit files that the SparkRing source of the deployment
    serving it provides (DEPLOYMENT_SOURCE). A mesh without that record, such
    as one that no deployment has served since its installation, is not
    compared. The date is that of the installed supervisor file.
    """
    base = Path(root)
    layouts = [managed_deployment.layout()]
    named = base / "etc/sparkring/deployments"
    if named.is_dir():
        for directory in sorted(named.iterdir()):
            try:
                layouts.append(managed_deployment.layout(directory.name))
            except ValueError:
                continue
    warnings = []
    for selected in layouts:
        code_root, config_root, unit_dir = (base / selected[key].lstrip("/")
                                            for key in ("code_dir", "config_dir", "unit_dir"))
        record = config_root / DEPLOYMENT_SOURCE
        if not record.is_file():
            continue
        unit = selected["mesh_unit"]
        try:
            expected = json.loads(record.read_text())
            if (any(_file_digest(code_root / name) != digest for name, digest in expected["code_sha256"].items())
                    or any(_file_digest(unit_dir / name) != digest for name, digest in expected["unit_sha256"].items())):
                supervisor = code_root / "runtime/glm53-spark-mtp3-mesh/managed_service.py"
                installed = time.strftime("%Y-%m-%d", time.localtime(
                    (supervisor if supervisor.is_file() else code_root).stat().st_mtime))
                warnings.append(f"mesh code of {unit} installed {installed} differs from this deployment's; "
                                "it refreshes when sparkring up next starts the mesh on all four Sparks")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            warnings.append(f"mesh code of {unit} not compared with its deployment's: {error}")
    return warnings


def _settle(selected, reference, rank, hcas, gid, host_ip, call, check, stale):
    """Stop this Spark's mesh where it must start again; returns (state, repaired ports, other active units).

    The state is "other" when another SparkRing mesh unit is active (nothing
    changes), "repaired" after the mesh stopped and stale RoCE GID addresses
    were added again, "stopped" when a running mesh failed its ring check and
    stopped, "running" when it passed, and "inactive" when it was not running.
    """
    unit = selected["mesh_unit"]
    active = _active_mesh_units(call)
    others = sorted(active - {unit})
    if others:
        return "other", [], others
    repaired = stale(reference, rank)
    if repaired:
        call(["systemctl", "stop", unit])
        for netdev, ipv4 in repaired:
            roce_gid.readd_address(netdev, ipv4, call)
        return "repaired", repaired, []
    if unit not in active:
        return "inactive", [], []
    try:
        check(reference, rank, hcas, gid, host_ip)
    except CHECK_FAILURES:
        # The supervisor stops in order: it stops its markers and removes its
        # network state. update_code confirms the stop before changing files.
        call(["systemctl", "stop", unit])
        return "stopped", [], []
    return "running", [], []


def stop_ring(reference, rank, hcas, gid, host_ip, *, call=node.call, check=qwen_mesh.check,
              stale=qwen_mesh.stale_gid_ports):
    """Stop this Spark's mesh where serve_ring starts it again.

    Every rank runs this, and completes it, before any rank runs serve_ring.
    The meshes that still run during serve_ring are then those that passed
    their ring check here, so every rank's update_code sees the same running
    supervisors when it decides whether to install mesh code. A mesh with a
    stale RoCE GID slot stops and the port's address is added again; a running
    mesh whose ring check fails stops. A Spark on which another SparkRing mesh
    unit is active is left unchanged.
    """
    if relays.is_reference(reference):
        # The relay table is fabric-level: nothing stops with a deployment.
        return {"ok": True, "unit": None, "action": "none", "relays": "fabric", "repaired": []}
    selected = mesh_layout(qwen_mesh.validate_site_reference(reference)["site_path"])
    state, repaired, others = _settle(selected, reference, rank, hcas, gid, host_ip, call, check, stale)
    result = {"ok": True, "unit": selected["mesh_unit"], "action": state,
              "repaired": [netdev for netdev, _ in repaired]}
    if others:
        result["active"] = others
    return result


def ring_stopped(reference, rank, hcas, gid, host_ip, *, call=node.call, check=qwen_mesh.check):
    """Verify stop_ring without changing anything: this Spark's mesh is stopped or passes its ring check.

    Another active SparkRing mesh unit also passes: stop_ring leaves it, and
    the ring check reports it. A relay table reference passes: nothing stops.
    """
    if relays.is_reference(reference):
        return {"ok": True}
    unit = mesh_unit(qwen_mesh.validate_site_reference(reference)["site_path"])
    if unit in _active_mesh_units(call):
        check(reference, rank, hcas, gid, host_ip)
    return {"ok": True}


def serve_ring(reference, rank, hcas, gid, host_ip, *, call=node.call, check=qwen_mesh.check,
               stale=qwen_mesh.stale_gid_ports, sleep=time.sleep, clock=time.monotonic, code=update_code,
               unpark=_unpark):
    """Serve the pinned four-Spark mesh on this Spark and wait until its ring check passes.

    Every rank runs this at the same time, after stop_ring, before the ring
    check of an installation. The mesh unit is enabled, so it returns at each
    boot, and ``unpark`` removes it from the park record (PARKED) that a switch
    to two-Spark models on the ring's halves left. An inactive unit starts. A running unit whose ring check fails
    stops and starts again, which rebuilds routes and rules lost when a cabled
    neighbor restarted. When a port's pinned RoCE GID slot lacks its IPv4
    address, as on the neighbors of a restarted Spark, the mesh stops, the
    address is deleted and added again, and the mesh starts. Before a mesh
    starts, ``code`` (update_code) installs this deployment's mesh code where
    the installed code differs; a running mesh keeps the code it started with.
    A Spark on which another SparkRing mesh unit is active is left unchanged;
    the ring check then reports it.

    A reference to the fabric document serves the relay table instead
    (``serve_relays``). A mesh is not started on a Spark whose fabric record
    holds relay markers: they and the mesh's markers would claim the same
    RDMA transmit packets, so such a deployment must be installed again.
    """
    if relays.is_reference(reference):
        return serve_relays(reference, rank, hcas, gid, host_ip, call=call, sleep=sleep, clock=clock)
    if relay_markers_recorded():
        raise ValueError("This Spark's fabric relay table carries the four-Spark two-hop paths, and this "
                         "deployment's mesh service would conflict with its markers. Install the model again with "
                         "sudo sparkring install so that it uses the relay table.")
    selected = mesh_layout(qwen_mesh.validate_site_reference(reference)["site_path"])
    unit = selected["mesh_unit"]
    state, repaired, others = _settle(selected, reference, rank, hcas, gid, host_ip, call, check, stale)
    if state == "other":
        return {"ok": True, "unit": unit, "action": "none", "active": others}
    if state == "running":
        refreshed = code(selected, rank, call=call)
        call(["systemctl", "enable", unit])
        unpark(unit)
        return {"ok": True, "unit": unit, "action": "checked", "repaired": [], "code": refreshed}
    action = {"repaired": "repaired", "stopped": "restarted", "inactive": "started"}[state]
    refreshed = code(selected, rank, call=call)
    call(["systemctl", "enable", "--now", unit])
    unpark(unit)
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
    return {"ok": True, "unit": unit, "action": action, "repaired": [netdev for netdev, _ in repaired],
            "code": refreshed}


def relay_markers_recorded(*, root="/"):
    """Whether this Spark's fabric record holds relay markers (``runtime/host/relays.py``)."""
    try:
        record = node.read(root, "/etc/sparkring/fabric.json")
    except (OSError, ValueError):
        return False
    return bool((record.get("relays") or {}).get("markers")) if isinstance(record, dict) else False


def check_ring(reference, rank, hcas, gid, host_ip):
    """The read-only ring check of a deployment's ``fabric`` reference: the relay table or a mesh."""
    if relays.is_reference(reference):
        return relays.check_reference(reference, rank, hcas, gid, host_ip)
    return qwen_mesh.check(reference, rank, hcas, gid, host_ip)


def serve_relays(reference, rank, hcas, gid, host_ip, *, call=node.call, check=relays.check_reference,
                 sleep=time.sleep, clock=time.monotonic, root="/"):
    """Prepare this Spark's relay table for a four-Spark deployment and wait until its ring check passes.

    Ports whose RoCE GID index 3 lost the IPv4 address, as on the neighbors
    of a restarted Spark, get it again (``roce_gid.serve``); that removes the
    relay routes through them, so the record's missing relay objects are then
    added (``relays.restore``) before the check, which waits for the markers
    up to RING_READY_SECONDS.
    """
    repaired = roce_gid.serve(hcas, gid, call=call)
    table = node.read(root, "/etc/sparkring/fabric.json").get("relays") or {}
    rows = relays.restore(table, call=call, root=root)
    deadline = clock() + RING_READY_SECONDS
    while True:
        try:
            check(reference, rank, hcas, gid, host_ip)
            break
        except CHECK_FAILURES as error:
            if clock() >= deadline:
                raise ValueError(f"The relay table's ring check still fails after {RING_READY_SECONDS} s: {error}") \
                    from None
            sleep(3)
    return {"ok": True, "unit": None, "action": "checked", "relays": "fabric",
            "repaired": repaired.get("repaired", []) if isinstance(repaired, dict) else [],
            "restored": sum(row["state"] == "restored" for row in rows)}


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
            if (prior["rank"] == rank and prior["unit"] != selected["mesh_unit"]
                    and node.call(["systemctl", "is-active", prior["unit"]], accepted=(0, 3, 4)).returncode == 0):
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
            # A unit with this deployment's own name was taken over by mesh-install.
            if prior["rank"] == rank and prior["unit"] != selected["mesh_unit"]:
                if compose.digest(Path(prior["reference"]["site_path"]).read_bytes()) != prior["reference"]["site_sha256"]:
                    raise ValueError("Previous mesh site changed since replacement review")
                node.call(["systemctl", "disable", "--now", prior["unit"]])
                _unpark(prior["unit"])
    else:
        raise ValueError("Unsupported native mesh operation")
    return {"ok": True}
