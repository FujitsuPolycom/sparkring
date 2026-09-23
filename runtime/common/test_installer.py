"""Offline installer barriers, immutable inputs, Compose generation and sharing."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from runtime.common import compose, glm_native_candidate, installer, tp2
from scripts import sparkring, sparkring_installer

GLM = installer.DEFAULTS["glm53", 2]
QWEN = installer.DEFAULTS["qwen38", 2]


def site(nodes=2):
    return {"schema": "sparkring-install-site/v1", "name": "local-test", "hosts": [
        {"host": f"private-spark{n}", "management_ip": f"192.0.2.{20+n}", "fabric_ip": f"198.18.20.{n+1}",
         "interface": "enp1s0f0np0", "model": "/srv/private-weights/model", "reuse_verified_model": True}
        for n in range(nodes)]}


@pytest.fixture
def deployment(tmp_path):
    data = b"offline source bundle fixture"
    lock = installer.make_lock(GLM, site(), "1" * 40, hashlib.sha256(data).hexdigest())
    installer.write(tmp_path / "deployment.lock.json", lock)
    (tmp_path / "source.bundle").write_bytes(data)
    return tmp_path, lock


class Hosts:
    def __init__(self, fail=None, uncertain=False):
        self.events = []
        self.fail, self.uncertain = fail, uncertain

    def __call__(self, host, argv, timeout):
        event = (argv[1], int(argv[2]))
        self.events.append(event)
        failure = event == self.fail
        return {"returncode": int(failure), "stdout": "ok", "stderr": "injected failure" if failure else "",
                "uncertain": failure and self.uncertain}


@pytest.mark.parametrize("profile", [GLM, QWEN, "qwen38-flash-next-tp2-sparkcache"])
def test_compose_generated_offline_without_docker_or_ssh(profile, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected external call"))
    lock = installer.make_lock(profile, site(), "1" * 40, "2" * 64)
    specs = installer.specifications(lock)
    files = installer.rendered(lock)
    assert len(specs) == 2
    assert len(files) == 4
    for number, spec in enumerate(specs):
        assert spec.labels[compose.LABEL] == lock["id"]
        assert spec.image_id == lock["selection"]["image_id"]
        assert lock["selection"]["image_reference"] in files[f"rank{number}/compose.yaml"]
        assert spec.restart_policy == "no"


def test_pair_start_waits_for_both_hosts_and_starts_worker_first(deployment):
    directory, _ = deployment
    hosts = Hosts()
    assert installer.apply(directory, "up", runner=hosts, execute=True)["complete"]
    assert max(hosts.events.index(("created", rank)) for rank in (0, 1)) < hosts.events.index(("start", 1))
    assert hosts.events.index(("running", 1)) < hosts.events.index(("start", 0))
    assert hosts.events.index(("running", 0)) < hosts.events.index(("smoke", 0))


def test_default_up_and_status_do_not_contact_hosts(deployment):
    directory, _ = deployment
    result = installer.apply(directory, "up", runner=lambda *a: pytest.fail("SSH"))
    assert not result["executed"]
    assert not (directory / "state.json").exists()
    assert installer.status(directory)["live_observed"] is False


def test_read_only_failure_blocks_mutation_and_can_resume(deployment):
    directory, _ = deployment
    hosts = Hosts(("prerequisites", 1))
    with pytest.raises(RuntimeError):
        installer.apply(directory, "up", runner=hosts, execute=True)
    assert not any(event[0] == "source" for event in hosts.events)
    assert installer.apply(directory, "up", runner=Hosts(), execute=True)["complete"]


def test_unknown_mutation_outcome_cannot_retry_or_change_direction(deployment):
    directory, _ = deployment
    with pytest.raises(RuntimeError):
        installer.apply(directory, "up", runner=Hosts(("create", 1), uncertain=True), execute=True)
    second = Hosts()
    with pytest.raises(ValueError, match="uncertain"):
        installer.apply(directory, "up", runner=second, execute=True)
    assert not any(event[0] == "create" for event in second.events)
    with pytest.raises(ValueError, match="uncertain"):
        installer.apply(directory, "down", runner=Hosts(), execute=True)


def test_down_after_readiness_failure_retains_receipts_and_allows_next_up(deployment):
    directory, _ = deployment
    with pytest.raises(RuntimeError):
        installer.apply(directory, "up", runner=Hosts(("ready", 0)), execute=True)
    assert installer.apply(directory, "down", runner=Hosts(), execute=True)["complete"]
    assert installer.apply(directory, "up", runner=Hosts(), execute=True)["complete"]
    assert len(list((directory / "operations").glob("*.json"))) == 3


def test_repeat_up_rechecks_completed_actions_without_creating_again(deployment):
    directory, _ = deployment
    installer.apply(directory, "up", runner=Hosts(), execute=True)
    second = Hosts()
    installer.apply(directory, "up", runner=second, execute=True)
    assert ("created", 0) in second.events
    assert not any(event[0] in ("create", "start", "image", "model", "source") for event in second.events)


def test_concurrent_up_and_down_are_excluded(deployment):
    directory, _ = deployment
    (directory / "operation.lock").write_text("other process")
    with pytest.raises(ValueError, match="Another operation"):
        installer.apply(directory, "down", runner=Hosts(), execute=True)


@pytest.mark.parametrize("change", ["site", "image", "bundle"])
def test_changed_lock_or_source_refuses_execution(deployment, change):
    directory, lock = deployment
    if change == "bundle":
        (directory / "source.bundle").write_bytes(b"different")
    else:
        if change == "site":
            lock["site"]["ranks"][0]["host_ip"] = "198.18.20.99"
        else:
            lock["selection"]["image_id"] = "sha256:" + "f" * 64
        (directory / "deployment.lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        installer.apply(directory, "up", runner=lambda *a: pytest.fail("Must not execute"), execute=True)


@pytest.mark.parametrize("field,value", [("host", "-oProxyCommand=bad"), ("model", "/"),
                                        ("reuse_verified_model", "yes"), ("password", "secret")])
def test_site_rejects_unsafe_or_unknown_inputs(field, value):
    raw = site()
    raw["hosts"][0][field] = value
    with pytest.raises(ValueError):
        installer.make_lock(GLM, raw, "1" * 40, "2" * 64)


def test_shared_export_regenerates_examples_instead_of_redacting_private_files(deployment, tmp_path):
    directory, _ = deployment
    for private in (False, True):
        output = tmp_path / ("private.zip" if private else "share.zip")
        installer.export(directory, output, share=not private)
        with zipfile.ZipFile(output) as archive:
            text = "\n".join(archive.read(name).decode() for name in archive.namelist() if name != "source.bundle")
            assert ("private-spark0" in text) is private
            assert ("/srv/private-weights" in text) is private
            assert ("source.bundle" in archive.namelist()) is private
            assert "rank0/compose.yaml" in archive.namelist()
            assert "rank1/compose.yaml" in archive.namelist()
            assert ("deployment.lock.json" in archive.namelist()) is private


def test_managed_glm_uses_managed_native_checks_and_lifecycle():
    lock = installer.make_lock(installer.DEFAULTS["glm53", 4], site(4), "1" * 40, "2" * 64)
    phases = [entry["id"] for entry in installer.operation_plan(lock, "up")["phases"]]
    assert phases.index("managed-native-check") < phases.index("managed-start")
    assert "managed-install" in phases
    assert "start-api" not in phases
    assert all(r["cache"] == lock["site"]["workspace"] + "/managed/cache" for r in lock["site"]["ranks"])


def test_managed_glm_rejects_cache_paths_its_stager_cannot_honor():
    raw = site(4)
    raw["hosts"][0]["cache"] = "/srv/external-cache"
    with pytest.raises(ValueError, match="Managed GLM cache"):
        installer.make_lock(installer.DEFAULTS["glm53", 4], raw, "1" * 40, "2" * 64)


def test_cli_offline_init_never_discovers_hosts(tmp_path, monkeypatch, capsys):
    path = tmp_path / "site.json"
    path.write_text(json.dumps(site()))
    calls = []
    monkeypatch.setattr(sparkring_installer, "discover", lambda *a: pytest.fail("No SSH"))
    monkeypatch.setattr(installer, "init", lambda *a, **k: calls.append((a, k)))
    assert sparkring.main(["init", "--model", "glm53", "--site", str(path)]) == 0
    assert calls[0][0][1] == GLM
    assert "No hosts changed" in capsys.readouterr().out


def test_offline_glm_plan_cannot_be_executed_directly():
    card = installer.setup.selection(GLM)
    plan = tp2.render(0, "198.18.20.1", Path("/srv/models/test").as_posix(), Path("/srv/cache/test").as_posix(), None,
                      card["image_id"], planning_release=card["release"], r33_sparkcache=True,
                      site_values={"VLLM_HOST_IP": "198.18.20.1", "NCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0"})
    with pytest.raises(ValueError, match="Offline plans"):
        tp2.execute(plan, "create", {}, run=lambda *a, **k: pytest.fail("Docker"))


@pytest.mark.parametrize("variant", ["nvfp4-spark", "nvfp4-qad"])
def test_planned_glm_compose_matches_admitted_adapter_settings(tmp_path, monkeypatch, variant):
    """Compare flag/envelope composition; fake bytes are not image qualification."""
    card = installer.setup.selection(GLM, variant)
    publication = glm_native_candidate.native.publication(card["release"])
    contract = installer.read(installer.ROOT / "runtime/releases" / card["release"] / "glm-profile-contract.json")
    cache = contract["sparkcache_native"]
    installed = {"schema": "sparkring-glm-native-profile-view/v1", "files": {
        cache["placement_path"]: cache["placement_sha256"], cache["snapshot_path"]: cache["snapshot_sha256"]},
        "compiler": {"source_trees": contract["source_trees"]}, "active_contracts": [cache["lease_contract"]]}
    receipt = {**publication, "schema": glm_native_candidate.SCHEMA, "installed": installed}
    monkeypatch.setattr(glm_native_candidate, "validate_receipt", lambda value: value)
    model, scratch = tmp_path / "model", tmp_path / "cache"
    model.mkdir()
    scratch.mkdir()
    (model / "config.json").write_text("{}")
    base = tp2.render(0, "198.18.20.1", model, scratch, None, card["image_id"],
                      site_values={"VLLM_HOST_IP": "198.18.20.1", "NCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0"})
    planned = tp2.adapt_release_plan(copy.deepcopy(base), publication, sparkcache=True,
                                    target_model_variant=variant, planning_contract=contract)
    admitted = tp2.adapt_release_plan(copy.deepcopy(base), receipt, sparkcache=True, target_model_variant=variant)
    assert tp2.container_spec(planned) == tp2.container_spec(admitted)


def test_source_bundle_initializes_a_real_git_checkout(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args, cwd=repo):
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Offline test")
    git("config", "user.email", "offline@example.invalid")
    (repo / "tracked").write_text("fixture")
    git("add", "tracked")
    git("commit", "-qm", "fixture")
    monkeypatch.setattr(installer, "ROOT", repo)
    output = tmp_path / "deployment"
    lock = installer.init(output, QWEN, site())
    checkout = tmp_path / "received"
    git("clone", "-q", str(output / "source.bundle"), str(checkout))
    assert git("rev-parse", "HEAD", cwd=checkout) == lock["source_revision"]
    assert installer.load(output) == lock
