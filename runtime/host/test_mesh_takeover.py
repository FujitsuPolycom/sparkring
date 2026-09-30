"""A new deployment with an earlier one's name takes over its mesh installation once that one stopped."""
import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.common import compose, installer, managed_deployment
from runtime.host import controller, native_mesh
from runtime.host.test_native_mesh import PROFILE, cluster

pytestmark = pytest.mark.skipif(os.name != "posix", reason="mesh site paths are absolute Linux paths")

EARLIER = "e" * 64


@pytest.fixture
def mesh(tmp_path, monkeypatch):
    """A four-Spark deployment whose mesh name an earlier, stopped installation holds on this Spark.

    The named mesh layout lives under tmp_path. Docker and systemd calls are
    recorded; ``running`` maps container IDs to their running state, and an
    absent ID is a removed container.
    """
    from scripts import installer_host
    real = managed_deployment.layout

    def layout(name=None):
        selected = real(name)
        if name is None:
            return selected
        return {**selected, "code_dir": str(tmp_path / "opt" / name), "config_dir": str(tmp_path / "etc" / name),
                "unit_dir": str(tmp_path / "units")}
    monkeypatch.setattr(managed_deployment, "layout", layout)
    monkeypatch.setattr(native_mesh, "REPLACED_ROOT", tmp_path / "replaced")
    value = cluster()
    raw = controller.model_site(value, PROFILE)
    selected = layout(raw["name"])
    config, code, units = Path(selected["config_dir"]), Path(selected["code_dir"]), Path(selected["unit_dir"])
    names = [selected["model_unit"], selected["liveness_unit"], selected["mesh_unit"]]
    config.mkdir(parents=True)
    (config / "site.json").write_text('{"earlier": true}')
    (config / "installer-owner.json").write_text(json.dumps({"deployment": EARLIER, "containers": ["old0", "old1", "old2", "old3"]}))
    (code / "runtime").mkdir(parents=True)
    units.mkdir()
    for name in names:
        (units / name).write_text("[Service]\n# earlier\n")
    reference = {"site_path": str(config / "site.json"), "site_sha256": compose.digest((config / "site.json").read_bytes()),
                 "plan_sha256": "f" * 64}
    # --fresh-mesh records the observed unit of every rank as replaced.
    site = native_mesh.select(raw, value, PROFILE, fresh=True,
                              invoke=lambda host, argv: json.dumps({"mesh": {"unit": selected["mesh_unit"], "reference": reference}}))
    running, calls = {}, []

    def call(argv, accepted=(0,), **options):
        calls.append(argv)
        if argv[:2] == ["docker", "inspect"]:
            state = running.get(argv[-1])
            return SimpleNamespace(returncode=1 if state is None else 0, stdout="true\n" if state else "false\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    class Owner:
        @staticmethod
        def source_payloads():
            return {}

        @staticmethod
        def install_code(payloads, root):
            (Path(root) / "runtime").mkdir(parents=True)
            return {"runtime/managed_service.py": "0" * 64}

    class Units:
        @staticmethod
        def render(site_path, containers, output, code_root, config_root, epoch, port, deployment_name):
            for rank in range(4):
                folder = Path(output) / f"rank{rank}"
                folder.mkdir(parents=True)
                (folder / "service.json").write_text("{}")
                for name in names:
                    (folder / name).write_text("[Service]\n# replacement\n")
    monkeypatch.setattr(native_mesh.node, "call", call)
    monkeypatch.setattr(native_mesh, "modules", lambda: (Owner, Units))
    monkeypatch.setattr(native_mesh, "prepare_local", lambda lock, rank: {"ok": True})
    monkeypatch.setattr(installer, "specifications", lambda lock, only_rank: [SimpleNamespace(name=f"sr-mesh-r{only_rank}")])
    monkeypatch.setattr(installer_host, "container", lambda spec: {"Id": "new" + spec.name[-1], "State": {"Running": False}})
    monkeypatch.setattr(installer_host, "owned", lambda spec, info, image: info)
    monkeypatch.setattr(installer_host, "image_info", lambda lock: None)
    payload = {"epoch": "0" * 32, "key": base64.b64encode(b"k" * 32).decode(), "containers": [{"Id": f"new{r}"} for r in range(4)]}
    return SimpleNamespace(lock=installer.make_lock(PROFILE, site, "a" * 40, "b" * 64), payload=payload, raw=raw, value=value,
                           selected=selected, config=config, code=code, units=units, names=names, running=running,
                           calls=calls, held=tmp_path / "replaced")


def test_a_fresh_mesh_takes_over_an_earlier_installation_whose_container_is_gone(mesh):
    assert native_mesh.install_check_local(mesh.lock, 0, mesh.payload) == {"ok": True, "takeover": True}
    assert not mesh.held.exists()
    assert native_mesh.install_local(mesh.lock, 0, mesh.payload) == {"ok": True}
    assert json.loads((mesh.config / "installer-owner.json").read_text()) == {
        "deployment": mesh.lock["id"], "containers": ["new0", "new1", "new2", "new3"]}
    assert all("# replacement" in (mesh.units / name).read_text() for name in mesh.names)
    # The earlier installation stopped and moved aside, not deleted.
    [held] = list(mesh.held.iterdir())
    assert json.loads((held / "config" / "installer-owner.json").read_text())["deployment"] == EARLIER
    assert (held / "code" / "runtime").is_dir()
    assert sorted(path.name for path in (held / "units").iterdir()) == sorted(mesh.names)
    disable = ["systemctl", "disable", "--now", *mesh.names]
    assert mesh.calls.index(disable) < mesh.calls.index(["systemctl", "daemon-reload"])
    assert mesh.calls[-1] == ["systemctl", "enable", mesh.selected["mesh_unit"]]
    # The replace step leaves the unit that install took over in place.
    mesh.calls.clear()
    assert native_mesh.operate_local(mesh.lock, 0, "mesh-replace") == {"ok": True}
    assert native_mesh.operate_local(mesh.lock, 0, "mesh-replaced") == {"ok": True}
    assert mesh.calls == []
    # Repeating the install recognizes its own receipt.
    assert native_mesh.install_local(mesh.lock, 0, mesh.payload) == {"ok": True}


def test_a_takeover_needs_a_stopped_owner_and_an_unchanged_site(mesh):
    mesh.running["old0"] = True
    for check in (native_mesh.install_check_local, native_mesh.install_local):
        with pytest.raises(ValueError, match="still runs on this Spark"):
            check(mesh.lock, 0, mesh.payload)
    # A container that sparkring down stopped may be taken over.
    mesh.running["old0"] = False
    assert native_mesh.install_check_local(mesh.lock, 0, mesh.payload)["takeover"]
    (mesh.config / "site.json").write_text('{"earlier": "edited"}')
    with pytest.raises(ValueError, match="changed since replacement review"):
        native_mesh.install_local(mesh.lock, 0, mesh.payload)
    assert not mesh.held.exists() and (mesh.config / "installer-owner.json").exists()


def test_a_plan_that_did_not_review_the_replacement_names_fresh_mesh(mesh):
    site = native_mesh.select(mesh.raw, mesh.value, PROFILE, invoke=lambda host, argv: '{"mesh": null}')
    lock = installer.make_lock(PROFILE, site, "a" * 40, "b" * 64)
    assert lock["site_input"]["native_mesh"]["replaces"] == []
    with pytest.raises(ValueError, match="belongs to another installation.*--fresh-mesh"):
        native_mesh.install_check_local(lock, 0, mesh.payload)
    assert not mesh.held.exists()


def test_the_installer_checks_every_rank_before_it_installs_on_any(tmp_path, monkeypatch):
    from scripts import deploy_stage, installer_runner
    calls = []

    def remote(number, operation, *, data=None):
        calls.append((operation, number))
        if operation == "mesh-install-check-local" and number == 2:
            raise ValueError("refused on rank 2")
        return {"Id": f"new{number}"} if operation == "container-record" else {"ok": True}

    def secrets(directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "health.key").write_bytes(b"k" * 32)
        return "0" * 32
    monkeypatch.setattr(deploy_stage, "prepare_secrets", secrets)
    runner = SimpleNamespace(lock={"id": "d" * 64}, directory=tmp_path, remote=remote)
    with pytest.raises(ValueError, match="rank 2"):
        installer_runner.Runner.native_mesh(runner, "mesh-install")
    assert [number for operation, number in calls if operation == "mesh-install-check-local"] == [0, 1, 2]
    assert not any(operation == "mesh-install-local" for operation, _ in calls)


def test_an_installation_whose_containers_were_created_again_is_installed_for_the_new_ones(mesh):
    # This deployment's own installation; docker container prune removed the
    # containers its units name, and create made new ones.
    planned = mesh.lock["site_input"]["native_mesh"]["site"]
    (mesh.config / "site.json").write_text(compose.encoded(planned))
    (mesh.config / "installer-owner.json").write_text(json.dumps({"deployment": mesh.lock["id"], "containers": ["old0", "old1", "old2", "old3"]}))
    # No reviewed replacement is needed for the deployment's own installation.
    own = dict(mesh.lock, site_input=dict(mesh.lock["site_input"], native_mesh=dict(mesh.lock["site_input"]["native_mesh"], replaces=[])))
    assert native_mesh.install_check_local(own, 0, mesh.payload) == {"ok": True, "takeover": True}
    assert native_mesh.install_local(own, 0, mesh.payload) == {"ok": True}
    assert json.loads((mesh.config / "installer-owner.json").read_text())["containers"] == ["new0", "new1", "new2", "new3"]
    # A site that differs from the deployment's plan is not installed over.
    (mesh.config / "installer-owner.json").write_text(json.dumps({"deployment": mesh.lock["id"], "containers": ["x0", "x1", "x2", "x3"]}))
    (mesh.config / "site.json").write_text('{"edited": true}')
    with pytest.raises(ValueError, match="differs from this deployment's plan"):
        native_mesh.install_check_local(own, 0, mesh.payload)


@pytest.mark.parametrize("recorded,reinstalled", [(["new0", "new1", "new2", "new3"], False), (["old0", "old1", "old2", "old3"], True)])
def test_the_installer_reinstalls_units_only_for_containers_created_again(tmp_path, monkeypatch, recorded, reinstalled):
    from scripts import deploy_stage, installer_runner
    calls = []

    def remote(number, operation, *, data=None):
        calls.append(operation)
        return {"Id": f"new{number}"} if operation == "container-record" else {"ok": True}

    def secrets(directory):
        (directory / "health.key").write_bytes(b"k" * 32)
        return "0" * 32
    monkeypatch.setattr(deploy_stage, "prepare_secrets", secrets)
    record = tmp_path / "native-mesh" / "installation.json"
    installer.write(record, {"deployment": "d" * 64, "container_ids": recorded})
    runner = SimpleNamespace(lock={"id": "d" * 64}, directory=tmp_path, remote=remote)
    runner.native_mesh = lambda operation: installer_runner.Runner.native_mesh(runner, operation)
    assert runner.native_mesh("mesh-install") == {"ok": True}
    assert ("mesh-install-local" in calls) is reinstalled
    assert ("mesh-installed-local" in calls) is not reinstalled
    assert json.loads(record.read_text())["container_ids"] == ["new0", "new1", "new2", "new3"]
    assert len(list(record.parent.glob("installation-*.json"))) == int(reinstalled)
