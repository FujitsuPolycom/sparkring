"""Device names from a SparkRing fabric document (``sparkring-fabric/v1``).

Route maps name RDMA devices. Without ``SIRCL_FABRIC_DOCUMENT`` the four
functions carry the DGX OS names; with it, the names come from the document's
port functions, which every Spark of the fabric must name alike.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sparkring_sircl import routes

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_NAMES = {(role.port, role.secondary): (role.device, role.netdev) for role in routes.DEFAULT_ROLES}
OTHER_NAMES ={(0, False): ("mlx5_0", "eth0"), (0, True): ("mlx5_1", "eth1"),
               (1, False): ("mlx5_2", "eth2"), (1, True): ("mlx5_3", "eth3")}


def _document(size: int = 4, names=None, *, override=None) -> dict:
    """A fabric document of ``size`` positions; ``names`` maps (port, secondary) to (rdma, netdev)."""
    names = names or DEFAULT_NAMES
    positions = []
    for position in range(size):
        ports = {}
        for port in (0, 1):
            functions = {}
            for secondary, kind in ((False, "primary"), (True, "secondary")):
                rdma, netdev = (override or {}).get((position, port, secondary), names[(port, secondary)])
                functions[kind] = {"netdev": netdev, "rdma": rdma, "mac": "-", "address": "-"}
            ports[str(port)] = {"cable": position, "functions": functions}
        positions.append({"position": position, "hostname": f"spark-{position}", "ports": ports})
    return {"schema": "sparkring-fabric/v1", "shape": "cycle", "size": size, "positions": positions}


def _write(tmp_path: Path, document: dict) -> str:
    path = tmp_path / "fabric.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def test_without_a_document_the_roles_carry_the_dgx_os_names():
    assert routes.load_roles({}) == routes.DEFAULT_ROLES
    assert routes.load_roles({"SIRCL_FABRIC_DOCUMENT": " "}) == routes.DEFAULT_ROLES
    assert [role.device for role in routes.DEFAULT_ROLES] == ["rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1",
                                                              "roceP2p1s0f1"]
    if not os.environ.get("SIRCL_FABRIC_DOCUMENT"):
        assert routes.ROLES == routes.DEFAULT_ROLES


def test_a_document_with_the_dgx_os_names_gives_the_default_roles(tmp_path):
    path = _write(tmp_path, _document())
    assert routes.load_roles({"SIRCL_FABRIC_DOCUMENT": path}) == routes.DEFAULT_ROLES


def test_a_document_with_other_names_replaces_devices_and_interfaces(tmp_path):
    roles = routes.load_roles({"SIRCL_FABRIC_DOCUMENT": _write(tmp_path, _document(names=OTHER_NAMES))})
    assert [(role.name, role.device, role.netdev, role.port, role.secondary) for role in roles] == [
        ("cw_primary", "mlx5_0", "eth0", 0, False), ("cw_secondary", "mlx5_1", "eth1", 0, True),
        ("ccw_primary", "mlx5_2", "eth2", 1, False), ("ccw_secondary", "mlx5_3", "eth3", 1, True)]


@pytest.mark.parametrize(("document", "message"), [
    ({"schema": "sparkring-fabric/v2", "positions": []}, "schema is not sparkring-fabric/v1"),
    ({"schema": "sparkring-fabric/v1", "positions": []}, "lists no positions"),
    (_document(override={(2, 0, True): ("roceP9p1s0f0", "enP9p1s0f0np0")}),
     "position 2 of the fabric document names port 0 secondary differently"),
    (_document(names={**DEFAULT_NAMES, (1, False): ("rocep1s0f0", "enp1s0f1np1")}), "one device or interface for two"),
    (_document(override={(1, 1, True): ("", "enP2p1s0f1np1")}), "position 1 of the fabric document names an empty"),
])
def test_documents_that_do_not_name_every_function_alike_are_refused(document, message):
    with pytest.raises(routes.RouteError, match=message):
        routes.roles_from_fabric_document(document)


def test_a_position_without_a_port_function_is_refused():
    document = _document()
    del document["positions"][3]["ports"]["1"]["functions"]["secondary"]
    with pytest.raises(routes.RouteError, match="position 3 of the fabric document does not name .* port 1 secondary"):
        routes.roles_from_fabric_document(document)


def test_an_unreadable_document_names_the_variable(tmp_path):
    with pytest.raises(routes.RouteError, match="SIRCL_FABRIC_DOCUMENT="):
        routes.load_roles({"SIRCL_FABRIC_DOCUMENT": str(tmp_path / "missing.json")})


def test_route_maps_name_the_documents_devices(tmp_path):
    """The roles are read once per process at import, so a fresh interpreter derives the route maps."""
    path = _write(tmp_path, _document(names=OTHER_NAMES))
    code = ("from sparkring_sircl import routes\n"
            "group = routes.derive_routes(routes.Layout.parse('ring:4'), 2)\n"
            "print(group.route_text(0))\n"
            "print(getattr(routes.role_of('mlx5_2'), 'name', None), routes.role_of('rocep1s0f0'))\n")
    environment = {**os.environ, "SIRCL_FABRIC_DOCUMENT": path,
                   "PYTHONPATH": os.pathsep.join(filter(None, (str(PROJECT), os.environ.get("PYTHONPATH"))))}
    result = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True,
                            check=True)
    assert result.stdout.splitlines() == ["1=mlx5_0/mlx5_1,2=mlx5_0/mlx5_3,3=mlx5_2/mlx5_3", "ccw_primary None"]
    environment.pop("SIRCL_FABRIC_DOCUMENT")
    default = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True,
                             check=True)
    assert default.stdout.splitlines() == [
        routes.derive_routes(routes.Layout.parse("ring:4"), 2).route_text(0), "None " + repr(routes.DEFAULT_ROLES[0])]
