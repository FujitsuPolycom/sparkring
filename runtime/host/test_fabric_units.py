"""Boot units, the packaged relay marker, the prepared transport over the relay table, and the survey bounds.

No host is contacted; the package build and the deployment ring operations
run with fakes.
"""
import json
from pathlib import Path

import pytest

from runtime.common import fabric_document, fabric_layout, installer
from runtime.host import bootstrap, control, controller, fabric, native_mesh, node, relays, settings, survey, topology
from runtime.host.test_fabric_layouts import plan_of
from runtime.host.test_relays import MARKER, Kernel

ROOT = Path(__file__).resolve().parents[2]
PACKAGING = ROOT / "packaging/debian"
PROFILE = "qwen38-flash-next-qad-tp4"


def unit(name):
    """``{section: {key: [values]}}`` of a systemd unit file."""
    sections, current = {}, None
    for line in (PACKAGING / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
        elif line and not line.startswith("#") and current is not None:
            key, _, value = line.partition("=")
            current.setdefault(key, []).append(value)
    return sections


def test_the_marker_unit_runs_the_records_markers_after_the_fabric_restore():
    marker = unit("sparkring-relay-marker.service")
    assert marker["Service"]["ExecStart"] == ["/usr/bin/sparkring node relay-markers"]
    assert marker["Service"]["Restart"] == ["on-failure"] and marker["Unit"]["StartLimitIntervalSec"] == ["0"]
    assert set(marker["Unit"]["After"][0].split()) == {"sparkring-hairpin.service", "sparkring-fabric.service"}
    assert marker["Unit"]["ConditionPathExists"] == ["/etc/sparkring/fabric.json"]
    assert marker["Install"]["WantedBy"] == ["multi-user.target"]
    restore = unit("sparkring-fabric.service")
    assert restore["Service"]["ExecStart"] == ["/usr/bin/sparkring node restore"]
    assert "sparkring-relay-marker.service" in restore["Unit"]["Before"][0].split()
    assert "sparkring-hairpin.service" in restore["Unit"]["After"][0].split()
    hairpin = unit("sparkring-hairpin.service")
    assert "sparkring-fabric.service" in hairpin["Unit"]["Before"][0].split()


def test_package_scripts_keep_the_marker_unit_enabled_across_reinstallation_without_starting_it():
    prerm = (PACKAGING / "prerm").read_text(encoding="utf-8")
    postinst = (PACKAGING / "postinst").read_text(encoding="utf-8")
    assert "sparkring-relay-marker.service" in prerm.split("UNITS=")[1].splitlines()[0]
    assert ('sparkring-fabric.service|sparkring-hairpin.service|sparkring-relay-marker.service) systemctl enable '
            '"$unit" ;;') in postinst


def test_setup_reads_the_marker_digest_the_package_recorded(tmp_path):
    built = {"path": "bin/sparkring-relay-marker", "sha256": "ab" * 32, "source_sha256": "cd" * 32}
    (tmp_path / "distribution.json").write_text(json.dumps({"relay_marker": built}))
    assert relays.marker_artifact(tmp_path) == {"binary": relays.MARKER_BINARY, "sha256": "ab" * 32}
    (tmp_path / "distribution.json").write_text(json.dumps({}))
    assert relays.marker_artifact(tmp_path) is None


def test_the_marker_binary_path_is_inside_the_package_payload():
    from scripts import build_deb
    assert relays.MARKER_BINARY == "/usr/lib/sparkring/" + build_deb.MARKER_PATH


# The prepared transport over the relay table.

def recorded_four_cycle(tmp_path):
    plan = plan_of(fabric_layout.layout("cycle", 4))
    document, relay_plan = fabric.prepare(plan, cluster="home", marker=MARKER)
    (tmp_path / "fabric.json").write_text(fabric_document.encoded(document), encoding="utf-8", newline="\n")
    return {"name": "home", "plan": plan}, document, relay_plan


def test_select_uses_the_table_and_plans_no_mesh_service_when_it_is_persistent(tmp_path):
    cluster, document, relay_plan = recorded_four_cycle(tmp_path)
    asked = []
    site = native_mesh.select(controller.model_site(cluster, PROFILE), cluster, PROFILE, state=tmp_path,
                              invoke=lambda *a, **k: asked.append(a) or '{"mesh":null}')
    assert asked == [] and "native_mesh" not in site
    reference = relays.persistent_reference(tmp_path, cluster)
    for row, host in zip(site["hosts"], cluster["plan"]["spec"]["hosts"], strict=True):
        assert row["fabric"] == reference and row["fabric_ip"] == row["management_ip"]
        assert row["interface"] == host["management_netdev"]
    lock = installer.make_lock(PROFILE, site, "a" * 40, "b" * 64)
    phases = [phase["id"] for phase in installer.operation_plan(lock, "up")["phases"]]
    assert "mesh-install" not in phases and phases.index("ring-serve") < phases.index("preflight")
    # The managed GLM backend keeps its own mesh.
    kept = native_mesh.select(controller.model_site(cluster, PROFILE), cluster, PROFILE, state=tmp_path,
                              existing_only=True, invoke=lambda *a, **k: '{"mesh":null}')
    assert all("fabric" not in row for row in kept["hosts"])


def test_without_a_persistent_table_select_plans_the_mesh_service(tmp_path):
    cluster, document, _ = recorded_four_cycle(tmp_path)
    document["relays"] = None
    document["transports"] = ["prepared"]
    (tmp_path / "fabric.json").write_text(fabric_document.encoded(document), encoding="utf-8", newline="\n")
    site = native_mesh.select(controller.model_site(cluster, PROFILE), cluster, PROFILE, state=tmp_path,
                              invoke=lambda *a, **k: '{"mesh":null}')
    assert site["native_mesh"]["mode"] == "create"


def test_ring_operations_over_the_table_stop_nothing_and_serve_by_restoring_and_checking(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    cluster, document, relay_plan = recorded_four_cycle(state)
    reference = relays.persistent_reference(state, cluster)
    rank = 0
    assert native_mesh.stop_ring(reference, rank, [], 3, "192.0.2.10") == {
        "ok": True, "unit": None, "action": "none", "relays": "fabric", "repaired": []}
    assert native_mesh.ring_stopped(reference, rank, [], 3, "192.0.2.10") == {"ok": True}
    root = tmp_path / "spark"
    config = topology.persistent_config(cluster["plan"], rank, relays=relays.section(relay_plan, rank))
    node.save(root, "/etc/sparkring/fabric.json", config)
    section = config["relays"]
    kernel = Kernel(root, sorted({r["dev"] for r in section["routes"]} | {f["dev"] for f in section["filters"]}))
    monkeypatch.setattr(native_mesh.roce_gid, "serve", lambda hcas, gid, call: {"ok": True, "repaired": ["enp1s0f0np0"]})
    checks = []

    def check(*args):
        checks.append(args)
        if len(checks) == 1:
            raise ValueError("marker missing")

    result = native_mesh.serve_relays(reference, rank, ["h"], 3, "192.0.2.10", call=kernel, check=check,
                                      sleep=lambda seconds: None, clock=lambda: 0, root=root)
    assert result == {"ok": True, "unit": None, "action": "checked", "relays": "fabric", "repaired": ["enp1s0f0np0"],
                      "restored": len(section["routes"]) * 2 + len(section["filters"]) + 4}
    assert len(checks) == 2


def test_a_mesh_service_is_not_started_beside_relay_markers(tmp_path, monkeypatch):
    cluster, document, relay_plan = recorded_four_cycle(tmp_path)
    config = topology.persistent_config(cluster["plan"], 1, relays=relays.section(relay_plan, 1))
    monkeypatch.setattr(native_mesh, "relay_markers_recorded", lambda **k: bool(config["relays"]["markers"]))
    reference = {"site_path": "/etc/sparkring/managed-mesh/site.json", "site_sha256": "c" * 64, "plan_sha256": "d" * 64}
    with pytest.raises(ValueError, match="Install the model again with sudo sparkring install"):
        native_mesh.serve_ring(reference, 1, [], 3, "192.0.2.11")


def test_the_ring_check_dispatches_on_the_reference(monkeypatch):
    seen = []
    monkeypatch.setattr(native_mesh.relays, "check_reference", lambda *a: seen.append(("relays", a)))
    monkeypatch.setattr(native_mesh.qwen_mesh, "check", lambda *a: seen.append(("mesh", a)))
    native_mesh.check_ring({"site_path": fabric_document.HOST_PATH, "site_sha256": "a" * 64, "plan_sha256": "b" * 64},
                           0, [], 3, "192.0.2.10")
    native_mesh.check_ring({"site_path": "/etc/sparkring/managed-mesh/site.json", "site_sha256": "a" * 64,
                            "plan_sha256": "b" * 64}, 0, [], 3, "192.0.2.10")
    assert [kind for kind, _ in seen] == ["relays", "mesh"]


# Survey depth and limit, discovery hops and the administration subnet.

@pytest.mark.parametrize("size, expected", [(None, (7, 9, 9)), (2, (1, 3, 3)), (4, (3, 5, 5)), (8, (7, 9, 9))])
def test_survey_bounds_follow_the_spark_count(size, expected):
    assert survey.bounds(size) == expected
    with pytest.raises(ValueError):
        survey.bounds(9)


def test_survey_stops_one_spark_beyond_the_expected_count_and_names_the_rest():
    from runtime.host.test_survey import scenario, run
    documents, table = scenario()
    found, _, _ = run(documents, table, size=2)
    assert len(found["sparks"]) == 3 and found["bounds"] == {"hops": 1, "limit": 3}
    assert any("The survey stopped at 3 Sparks" in note for note in found["notes"])


def test_discovery_routes_reach_the_far_end_of_an_eight_spark_line(tmp_path):
    route = [{"user": "root", "address": f"fe80::{hop + 1}", "interface": "enp1s0f0np0", "port": 22}
             for hop in range(bootstrap.MAX_HOPS)]
    assert bootstrap.MAX_HOPS == 7
    assert bootstrap.ssh_argv(route, tmp_path)[-1] == "root@fe80::7%enp1s0f0np0"
    with pytest.raises(ValueError, match="one through 7 hops"):
        bootstrap.ssh_argv(route + route[:1], tmp_path)


def test_the_administration_subnet_widens_for_seven_or_eight_sparks():
    assert control.subnet_for(control.DEFAULT_SUBNET, 6) == "10.253.255.0/29"
    assert control.subnet_for(control.DEFAULT_SUBNET, 8) == "10.253.255.0/28"
    assert control.subnet_for(control.DEFAULT_SUBNET, 8, explicit=True) == "10.253.255.0/29"
    with pytest.raises(ValueError, match="holds 6 Sparks; 8 need a /28"):
        control.plan([{"id": str(n)} for n in range(8)], [], "0", subnet=control.DEFAULT_SUBNET)


def test_settings_accept_a_wide_control_subnet_and_name_the_keys_a_file_sets(tmp_path):
    path = tmp_path / "sparkring.env"
    path.write_text("SPARKRING_CONTROL_CIDR=10.253.255.0/28\nSPARKRING_FABRIC_CIDR=198.18.0.0/20\n")
    assert settings.load(path)["SPARKRING_CONTROL_CIDR"] == "10.253.255.0/28"
    assert settings.explicit(path) == {"SPARKRING_CONTROL_CIDR", "SPARKRING_FABRIC_CIDR"}
    assert settings.explicit(None) == set()


def test_setup_leaves_the_fabric_supernet_to_the_layout_unless_it_is_named(tmp_path):
    from runtime.host import single_uplink
    args, _ = single_uplink._arguments([])
    assert args.fabric_cidr is None and not args.control_explicit
    path = tmp_path / "sparkring.env"
    path.write_text("SPARKRING_FABRIC_CIDR=198.18.0.0/21\n")
    args, _ = single_uplink._arguments(["--env", str(path), "--control-cidr=10.253.255.0/29"])
    assert args.fabric_cidr == "198.18.0.0/21" and args.control_explicit


def test_installer_profiles_are_refused_on_layouts_they_do_not_serve():
    from runtime.host import placement
    for layout in (fabric_layout.layout("pair", 2), fabric_layout.layout("cycle", 4)):
        assert placement.require_layout({"plan": plan_of(layout)}) == layout
    with pytest.raises(ValueError, match=r"This fabric is a path-4; the installer's profiles run on a pair or a "
                                         r"four-Spark ring \(cycle-4\)"):
        placement.require_layout({"plan": plan_of(fabric_layout.layout("path", 4))})
    with pytest.raises(ValueError, match="This fabric is a cycle-8"):
        placement.require_layout({"plan": plan_of(fabric_layout.layout("cycle", 8))})
