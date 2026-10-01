"""A mesh's installed code follows its deployment's SparkRing source only while no supervisor runs it.

The mesh layout lives under tmp_path; the deployment's managed_install,
managed_units and managed_service are small fakes, and systemctl is simulated.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from runtime.common import managed_deployment
from runtime.host import native_mesh, node

NAME = "home"
REFERENCE = {"site_path": f"/etc/sparkring/deployments/{NAME}/site.json", "site_sha256": "c" * 64,
             "plan_sha256": "d" * 64}
SUPERVISOR = "runtime/glm53-spark-mtp3-mesh/managed_service.py"
COMMON = "runtime/common/compose.py"
CONTAINER = "a" * 64


def sha(content):
    return hashlib.sha256(content).hexdigest()


class Service:
    """The deployment source's managed_service: its configuration check and the identity it computes."""

    identity = "identity-of-this-source"

    @staticmethod
    def load_config(path):
        config = json.loads(Path(path).read_text())
        return config, {"management_addresses": [f"192.0.2.{110 + rank}" for rank in range(4)]}, None, None, \
            Service.identity


class Owner:
    """The deployment source's managed_install: its source allowlist and digests."""

    ROOT = Path("/srv/sparkring/home/source")
    files = {SUPERVISOR: b"supervisor that awaits the management address\n", COMMON: b"compose\n"}

    @classmethod
    def source_payloads(cls):
        return dict(cls.files)

    @staticmethod
    def source_hashes(payloads):
        return {name: sha(content) for name, content in payloads.items()}


class Units:
    """The deployment source's managed_units."""

    service = Service
    model = "[Service]\nExecStart=/usr/bin/docker start --attach {container}\n"

    @staticmethod
    def unit_text(code_root, config_root, container_id, *, host_liveness=False, deployment_name=None):
        prefix = "sparkring-" + deployment_name if deployment_name else "sparkring"
        units = {prefix + "-mesh.service": f"[Service]\nExecStart=/usr/bin/python3 {code_root}/{SUPERVISOR} run\n",
                 prefix + "-model.service" if deployment_name else "sparkring-mesh-model.service":
                     Units.model.format(container=container_id)}
        if host_liveness:
            units[prefix + "-scheduler-liveness.service"] = "[Service]\n"
        return units


class Systemd:
    """systemctl of one Spark: the active mesh units, each unit's is-active state, and every command."""

    def __init__(self, layout, *, active=(), states=None, peek=None):
        self.layout, self.active, self.states, self.peek = layout, set(active), dict(states or {}), peek
        self.commands, self.seen = [], []

    def call(self, argv, **options):
        self.commands.append(list(argv))
        if argv[:2] == ["systemctl", "list-units"]:
            return SimpleNamespace(returncode=0, stdout="".join(f"{u} loaded active running mesh\n" for u in sorted(self.active)))
        if argv[:2] == ["systemctl", "is-active"]:
            states = [self.states.get(unit, "active" if unit in self.active else "inactive") for unit in argv[2:]]
            return SimpleNamespace(returncode=0 if "active" in states else 3, stdout="\n".join(states) + "\n")
        if argv[:2] == ["systemctl", "stop"]:
            self.active.discard(argv[-1])
        if argv[:3] == ["systemctl", "enable", "--now"] and self.peek:
            # What the unit starts with.
            self.seen.append(self.peek())
        return SimpleNamespace(returncode=0, stdout="")

    def changes(self):
        return [argv for argv in self.commands if argv[:2] not in (["systemctl", "list-units"], ["systemctl", "is-active"])]


@pytest.fixture
def mesh(tmp_path, monkeypatch):
    """A named mesh installed on this Spark from other SparkRing source, whose supervisor does not wait.

    Its files match Owner and Units except the supervisor. ``receipt`` writes
    the installation receipt of either installer: source-hashes.json
    (install_local) or installation.json (managed_install.py).
    """
    real = managed_deployment.layout

    def layout(name=None):
        selected = real(name)
        if name is None:
            return selected
        return {**selected, "code_dir": str(tmp_path / "opt" / name), "config_dir": str(tmp_path / "etc" / name),
                "unit_dir": str(tmp_path / "units")}
    monkeypatch.setattr(managed_deployment, "layout", layout)
    monkeypatch.setattr(native_mesh, "modules", lambda: (Owner, Units))
    selected = layout(NAME)
    code, config, units = (Path(selected[key]) for key in ("code_dir", "config_dir", "unit_dir"))
    installed = {SUPERVISOR: b"supervisor that fails without the management address\n", COMMON: Owner.files[COMMON]}
    for name, content in installed.items():
        (code / name).parent.mkdir(parents=True, exist_ok=True)
        (code / name).write_bytes(content)
    config.mkdir(parents=True)
    service = {"rank": 1, "container_id": CONTAINER, "deployment_name": NAME, "key_file": str(config / "health.key"),
               "health_port": 9976, "epoch": "e" * 32}
    (config / "service.json").write_text(json.dumps(service))
    units.mkdir()
    rendered = Units.unit_text(selected["code_dir"], selected["config_dir"], CONTAINER, deployment_name=NAME)
    for name, text in rendered.items():
        (units / name).write_text(text, newline="\n")

    def receipt(kind):
        hashes = {name: sha(content) for name, content in installed.items()}
        if kind == native_mesh.SOURCE_HASHES:
            node.save(config, kind, hashes)
        else:
            node.save(config, kind, {"schema": "sparkring-managed-install/v1", "rank": 1, "source_files": list(installed),
                                     "source_hashes": hashes, "installed_source_sha256": hashes,
                                     "units": rendered, "unit_hashes": {n: sha(t.encode()) for n, t in rendered.items()},
                                     "source_root": "/srv/sparkring/other/source", "applied": True})
    peers = []

    def probe(service_module, config_document, site, rank):
        peers.append(rank)
        return {peer: None for peer in range(4) if peer != rank}
    monkeypatch.setattr(native_mesh, "peer_states", probe)
    return SimpleNamespace(selected=selected, code=code, config=config, units=units, installed=installed, receipt=receipt,
                           peers=peers, unit=selected["mesh_unit"])


def serve(mesh, systemd, *, check=lambda *a: None, clock=lambda: 0.0):
    """serve_ring of rank 1 with update_code, whose peer probe is the one the test installs."""
    return native_mesh.serve_ring(REFERENCE, 1, ["mlx5_0"], 3, "192.0.2.111", call=systemd.call, check=check,
                                  stale=lambda reference, rank: [], sleep=lambda seconds: None, clock=clock,
                                  code=lambda selected, rank, call: native_mesh.update_code(
                                      selected, rank, call=call, peers=native_mesh.peer_states))


def supervisor(mesh):
    return (mesh.code / SUPERVISOR).read_bytes()


@pytest.mark.parametrize("kind", [native_mesh.SOURCE_HASHES, native_mesh.INSTALLATION_RECEIPT])
def test_a_stopped_mesh_starts_with_its_deployments_code_and_receipt(mesh, kind):
    mesh.receipt(kind)
    systemd = Systemd(mesh.selected, states={mesh.unit: "failed"}, peek=lambda: supervisor(mesh))
    result = serve(mesh, systemd)
    assert result["action"] == "started"
    assert result["code"] == {"refreshed": True, "files": [SUPERVISOR], "removed": [], "units": []}
    # The supervisor file was replaced before the unit started.
    assert systemd.seen == [Owner.files[SUPERVISOR]]
    assert systemd.changes() == [["systemctl", "enable", "--now", mesh.unit]]
    assert mesh.peers == [1]
    # The receipt names the installed files, so the mesh-installed-local check passes.
    hashes = Owner.source_hashes(Owner.files)
    document = json.loads((mesh.config / kind).read_text())
    if kind == native_mesh.SOURCE_HASHES:
        assert document == hashes
    else:
        assert document["installed_source_sha256"] == document["source_hashes"] == hashes
        assert document["source_files"] == list(Owner.files) and document["source_root"] == str(Owner.ROOT)
        assert document["applied"] is True and document["rank"] == 1
    assert all(sha((mesh.code / name).read_bytes()) == digest for name, digest in hashes.items())
    record = json.loads((mesh.config / native_mesh.DEPLOYMENT_SOURCE).read_text())
    assert record["code_sha256"] == hashes and record["source_root"] == str(Owner.ROOT)


def test_receipt_files_outside_the_allowlist_go_and_unknown_files_stay(mesh):
    mesh.receipt(native_mesh.SOURCE_HASHES)
    dropped = mesh.code / "runtime/common/retired.py"
    dropped.write_bytes(b"retired\n")
    hashes = json.loads((mesh.config / native_mesh.SOURCE_HASHES).read_text())
    node.save(mesh.config, native_mesh.SOURCE_HASHES, {**hashes, "runtime/common/retired.py": sha(b"retired\n"),
                                                       "../outside.py": "0" * 64})
    unknown = mesh.code / "notes.txt"
    unknown.write_text("operator notes\n")
    result = serve(mesh, Systemd(mesh.selected))
    assert result["code"]["removed"] == ["runtime/common/retired.py"]
    assert not dropped.exists() and unknown.read_text() == "operator notes\n"


@pytest.mark.parametrize("states", [
    {"sparkring-home-mesh.service": "activating"},
    {"sparkring-home-model.service": "deactivating"},
    {"sparkring-home-scheduler-liveness.service": "active"},
])
def test_a_mesh_that_systemd_does_not_report_stopped_keeps_its_code(mesh, states):
    """serve_ring starts a unit that is not active; an activating supervisor already runs the installed code."""
    # Rank 0's liveness unit, rendered as this source renders it.
    (mesh.units / "sparkring-home-scheduler-liveness.service").write_text("[Service]\n", newline="\n")
    mesh.receipt(native_mesh.SOURCE_HASHES)
    before = json.loads((mesh.config / native_mesh.SOURCE_HASHES).read_text())
    result = serve(mesh, Systemd(mesh.selected, states=states))
    assert result["code"] == {"refreshed": False, "reason": "running"}
    assert supervisor(mesh) == mesh.installed[SUPERVISOR] and mesh.peers == []
    assert json.loads((mesh.config / native_mesh.SOURCE_HASHES).read_text()) == before


def test_a_running_mesh_is_never_touched(mesh):
    mesh.receipt(native_mesh.SOURCE_HASHES)
    stamp = (mesh.code / SUPERVISOR).stat().st_mtime_ns
    systemd = Systemd(mesh.selected, active=[mesh.unit])
    result = serve(mesh, systemd)
    assert result["action"] == "checked" and result["code"] == {"refreshed": False, "reason": "running"}
    assert systemd.changes() == [["systemctl", "enable", mesh.unit]]
    assert supervisor(mesh) == mesh.installed[SUPERVISOR] and (mesh.code / SUPERVISOR).stat().st_mtime_ns == stamp
    # The deployment's digests are recorded, so node status reports the difference.
    record = json.loads((mesh.config / native_mesh.DEPLOYMENT_SOURCE).read_text())
    assert record["code_sha256"][SUPERVISOR] == sha(Owner.files[SUPERVISOR])


def test_a_restart_whose_stop_is_not_confirmed_keeps_the_code(mesh):
    """A failing running mesh stops; systemd still reports it deactivating, so its files stay."""
    mesh.receipt(native_mesh.SOURCE_HASHES)

    def check(*args):
        raise ValueError("Missing mesh network objects: ['route:rank1-to-rank3']")
    systemd = Systemd(mesh.selected, active=[mesh.unit], states={mesh.unit: "deactivating"})
    times = iter([0.0, 250.0])
    with pytest.raises(ValueError, match="still fails"):
        serve(mesh, systemd, check=check, clock=lambda: next(times))
    assert ["systemctl", "stop", mesh.unit] in systemd.commands
    assert supervisor(mesh) == mesh.installed[SUPERVISOR]


def test_a_restart_whose_stop_completed_updates_the_code_between_stop_and_start(mesh):
    mesh.receipt(native_mesh.SOURCE_HASHES)
    checks = iter([ValueError("Missing mesh network objects"), None])

    def check(*args):
        error = next(checks)
        if error:
            raise error
    systemd = Systemd(mesh.selected, active=[mesh.unit], peek=lambda: supervisor(mesh))
    result = serve(mesh, systemd, check=check)
    assert result["action"] == "restarted" and result["code"]["refreshed"] is True
    assert systemd.changes() == [["systemctl", "stop", mesh.unit], ["systemctl", "enable", "--now", mesh.unit]]
    assert systemd.seen == [Owner.files[SUPERVISOR]]


def test_unchanged_code_is_not_rewritten(mesh):
    for name, content in Owner.files.items():
        (mesh.code / name).write_bytes(content)
    node.save(mesh.config, native_mesh.SOURCE_HASHES, Owner.source_hashes(Owner.files))
    before = {path: path.stat().st_mtime_ns for path in [*mesh.code.rglob("*"), *mesh.units.iterdir(),
                                                         mesh.config / native_mesh.SOURCE_HASHES]}
    systemd = Systemd(mesh.selected)
    result = serve(mesh, systemd)
    assert result["code"] == {"refreshed": False, "reason": "unchanged"}
    assert systemd.changes() == [["systemctl", "enable", "--now", mesh.unit]]
    assert not any(argv[:2] == ["systemctl", "is-active"] for argv in systemd.commands) and mesh.peers == []
    assert {path: path.stat().st_mtime_ns for path in before} == before
    assert sorted(path.name for path in mesh.code.rglob("*") if path.is_file()) == ["compose.py", "managed_service.py"]


def test_a_peer_running_other_mesh_code_keeps_this_sparks_code(mesh, monkeypatch):
    """Ranks whose supervisors report different identities never form the four-rank group."""
    mesh.receipt(native_mesh.SOURCE_HASHES)
    monkeypatch.setattr(native_mesh, "peer_states", lambda *a: {0: "identity-of-other-code", 2: None,
                                                                3: Service.identity})
    result = serve(mesh, Systemd(mesh.selected))
    assert result["code"] == {"refreshed": False, "reason": "other ranks run other mesh code", "ranks": [0]}
    assert supervisor(mesh) == mesh.installed[SUPERVISOR]
    monkeypatch.setattr(native_mesh, "peer_states", lambda *a: {0: "unknown", 2: None, 3: None})
    assert serve(mesh, Systemd(mesh.selected))["code"]["ranks"] == [0]
    # Supervisors that already run this source's code, or none at all, let it update.
    monkeypatch.setattr(native_mesh, "peer_states", lambda *a: {0: Service.identity, 2: None, 3: None})
    assert serve(mesh, Systemd(mesh.selected))["code"]["refreshed"] is True


def test_a_configuration_that_this_source_rejects_keeps_the_code(mesh, monkeypatch):
    mesh.receipt(native_mesh.SOURCE_HASHES)
    monkeypatch.setattr(Service, "load_config", staticmethod(lambda path: (_ for _ in ()).throw(
        ValueError("Unsupported managed mesh configuration"))))
    result = serve(mesh, Systemd(mesh.selected))
    assert result["code"]["refreshed"] is False and "Unsupported managed mesh configuration" in result["code"]["reason"]
    assert supervisor(mesh) == mesh.installed[SUPERVISOR]


def test_differing_units_are_written_and_reloaded_before_the_start(mesh, monkeypatch):
    for name, content in Owner.files.items():
        (mesh.code / name).write_bytes(content)
    mesh.receipt(native_mesh.INSTALLATION_RECEIPT)
    monkeypatch.setattr(Units, "model", "[Service]\nTimeoutStartSec=120s\nExecStart=/usr/bin/docker start --attach {container}\n")
    model = Path(mesh.selected["unit_dir"]) / mesh.selected["model_unit"]
    systemd = Systemd(mesh.selected, peek=lambda: model.read_text())
    result = serve(mesh, systemd)
    assert result["code"] == {"refreshed": True, "files": [], "removed": [], "units": [mesh.selected["model_unit"]]}
    assert systemd.seen == [Units.model.format(container=CONTAINER)]
    assert systemd.changes() == [["systemctl", "daemon-reload"], ["systemctl", "reenable", mesh.unit],
                                 ["systemctl", "enable", "--now", mesh.unit]]
    receipt = json.loads((mesh.config / native_mesh.INSTALLATION_RECEIPT).read_text())
    assert receipt["unit_hashes"][mesh.selected["model_unit"]] == sha(model.read_bytes())
    assert receipt["units"][mesh.selected["model_unit"]] == model.read_text()
    assert not any(path.name.endswith(".sparkring-new") for path in Path(mesh.selected["unit_dir"]).iterdir())


def test_a_mesh_without_a_configuration_is_served_without_comparison(tmp_path, monkeypatch):
    selected = {**managed_deployment.layout(NAME), "code_dir": str(tmp_path / "opt"),
                "config_dir": str(tmp_path / "missing"), "unit_dir": str(tmp_path / "units")}
    calls = []
    result = native_mesh.update_code(selected, 1, call=lambda argv, **k: calls.append(argv))
    assert result["refreshed"] is False and result["reason"].startswith("mesh files not compared")
    assert calls == [] and not (tmp_path / "missing").exists()


def status_root(tmp_path, *, installed, recorded):
    """A default mesh under a status root: its installed supervisor and its deployment's record."""
    selected = managed_deployment.layout()
    code = tmp_path / selected["code_dir"].lstrip("/")
    config = tmp_path / selected["config_dir"].lstrip("/")
    (code / SUPERVISOR).parent.mkdir(parents=True)
    (code / SUPERVISOR).write_bytes(installed)
    stamp = time.mktime((2026, 9, 21, 12, 0, 0, 0, 0, -1))
    os.utime(code / SUPERVISOR, (stamp, stamp))
    config.mkdir(parents=True)
    node.save(config, native_mesh.DEPLOYMENT_SOURCE, {"schema": native_mesh.DEPLOYMENT_SOURCE_SCHEMA,
                                                       "source_root": str(Owner.ROOT),
                                                       "code_sha256": {SUPERVISOR: sha(recorded)}, "unit_sha256": {}})


def test_status_reports_mesh_code_that_differs_from_its_deployments(tmp_path):
    status_root(tmp_path, installed=b"supervisor of 2026-09-21\n", recorded=Owner.files[SUPERVISOR])
    assert native_mesh.code_warnings(root=tmp_path) == [
        "mesh code of sparkring-mesh.service installed 2026-09-21 differs from this deployment's; "
        "it refreshes when sparkring up next starts the mesh on all four Sparks"]


def test_status_is_quiet_for_matching_or_unrecorded_mesh_code(tmp_path):
    assert native_mesh.code_warnings(root=tmp_path) == []
    status_root(tmp_path, installed=Owner.files[SUPERVISOR], recorded=Owner.files[SUPERVISOR])
    assert native_mesh.code_warnings(root=tmp_path) == []


def test_peer_states_reads_each_other_ranks_identity_with_the_real_supervisor_protocol(monkeypatch):
    service = native_mesh.modules()[1].service
    key = b"k" * 32
    monkeypatch.setattr(service, "read_key", lambda path: key)

    class Connection:
        def __init__(self, address, port, timeout):
            self.address = address

        def request(self, method, path, headers):
            if self.address.endswith(".110"):
                raise ConnectionRefusedError("nothing listens")
            if self.address.endswith(".113"):
                raise TimeoutError("no answer")
            self.nonce = path[-32:]

        def getresponse(self):
            body = {"protocol": service.PROTOCOL, "nonce": self.nonce, "rank": 2, "identity": "identity-of-rank-2",
                    "epoch": "e" * 32, "generation": "b" * 32, "local_ready": True}
            raw = service.canonical({"body": body, "signature": service.sign(key, body)})
            return SimpleNamespace(status=200, read=lambda limit: raw)

        def close(self):
            pass
    monkeypatch.setattr(service.http.client, "HTTPConnection", Connection)
    config = {"key_file": "/etc/sparkring/managed-mesh/health.key", "health_port": 9975, "epoch": "e" * 32}
    site = {"management_addresses": [f"192.0.2.{110 + rank}" for rank in range(4)]}
    assert native_mesh.peer_states(service, config, site, 1) == {0: None, 2: "identity-of-rank-2", 3: "unknown"}


@pytest.mark.skipif(os.name != "posix", reason="systemd unit paths are absolute Linux paths")
def test_the_source_allowlist_and_rendered_units_refresh_an_installed_mesh(tmp_path, monkeypatch):
    """The real managed_install and managed_units: one stale supervisor file is replaced, units are unchanged."""
    real = managed_deployment.layout
    monkeypatch.setattr(managed_deployment, "layout", lambda name=None: real(name) if name is None else {
        **real(name), "code_dir": str(tmp_path / "opt" / name), "config_dir": str(tmp_path / "etc" / name),
        "unit_dir": str(tmp_path / "units")})
    owner, units = native_mesh.modules()
    selected = managed_deployment.layout(NAME)
    code, config, unit_dir = (Path(selected[key]) for key in ("code_dir", "config_dir", "unit_dir"))
    payloads = owner.source_payloads()
    owner.install_code(payloads, code)
    config.mkdir(parents=True)
    service = {"rank": 1, "container_id": CONTAINER, "deployment_name": NAME}
    (config / "service.json").write_text(json.dumps(service))
    node.save(config, native_mesh.SOURCE_HASHES, owner.source_hashes(payloads))
    unit_dir.mkdir()
    for name, text in units.unit_text(selected["code_dir"], selected["config_dir"], CONTAINER,
                                      deployment_name=NAME).items():
        (unit_dir / name).write_text(text)
    (code / SUPERVISOR).write_bytes(b"# supervisor of other code\n")
    monkeypatch.setattr(units.service, "load_config", lambda path: (service, {}, None, None, "identity"))
    result = native_mesh.update_code(selected, 1, call=Systemd(selected).call, peers=lambda *a: {0: None, 2: None, 3: None})
    assert result == {"refreshed": True, "files": [SUPERVISOR], "removed": [], "units": []}
    assert (code / SUPERVISOR).read_bytes() == payloads[SUPERVISOR]
    hashes = json.loads((config / native_mesh.SOURCE_HASHES).read_text())
    assert hashes == owner.source_hashes(payloads)
    assert all(sha((code / name).read_bytes()) == digest for name, digest in hashes.items())
