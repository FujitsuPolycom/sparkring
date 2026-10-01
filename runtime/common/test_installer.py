"""Offline installer barriers, immutable inputs, Compose generation and sharing."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

import pytest
import yaml

from runtime.common import compose, glm_native_candidate, installer, process_lock, tp2
from scripts import sparkring, sparkring_installer

GLM = installer.GLM_LEGACY[2]
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


def test_asset_preparation_has_no_stop_start_or_fabric_mutations(deployment):
    directory, _ = deployment
    hosts = Hosts()
    assert installer.apply(directory, "prepare", runner=hosts, execute=True)["complete"]
    assert {name for name, rank in hosts.events} == {
        "prepare-prerequisites", "source", "source-check", "image", "image-check", "model", "model-check"}
    assert installer.apply(directory, "up", runner=hosts, execute=True)["complete"]


def test_default_up_and_status_do_not_contact_hosts(deployment):
    directory, _ = deployment
    result = installer.apply(directory, "up", runner=lambda *a: pytest.fail("SSH"))
    assert not result["executed"]
    assert not (directory / "state.json").exists()
    saved = installer.status(directory)
    assert saved["live_observed"] is False
    card = installer.load(directory)["selection"]
    assert (saved["checkpoint"], saved["model_revision"], saved["image_release"]) == (
        card["target_variant"], card["model_revision"], card["release"])


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
    # Stopping remains possible: it checks ownership and only stops this
    # deployment's running containers, so recovery can always clear a failed start.
    assert installer.apply(directory, "down", runner=Hosts(), execute=True)["complete"]


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


class Stopped(Hosts):
    """Ranks whose model containers stopped, as after the Sparks restarted; starting a rank runs it again."""

    def __init__(self, stopped, silent=()):
        super().__init__()
        self.stopped, self.silent = set(stopped), set(silent)

    def __call__(self, host, argv, timeout):
        event = (argv[1], int(argv[2]))
        if event[0] == "start":
            self.stopped.discard(event[1])
        if event[0] == "running" and event[1] in self.stopped:
            self.events.append(event)
            return {"returncode": 1, "stdout": "", "stderr": "ValueError: Rank is not running", "uncertain": False}
        if event[0] == "running" and event[1] in self.silent:
            self.events.append(event)
            return {"returncode": 1, "stdout": "", "stderr": "ssh: connect to host 192.0.2.21 port 22: Connection timed out",
                    "uncertain": False}
        return super().__call__(host, argv, timeout)


def test_repeat_up_after_every_container_stopped_runs_every_phase_again(deployment):
    directory, _ = deployment
    installer.apply(directory, "up", runner=Hosts(), execute=True)
    restarted = Stopped({0, 1})
    result = installer.apply(directory, "up", runner=restarted, execute=True)
    assert result["complete"] and result["generation"] == 2
    phases = [event[0] for event in restarted.events]
    assert "gid-serve" in phases and phases.index("gid-serve") < phases.index("start")
    assert ("start", 1) in restarted.events and ("start", 0) in restarted.events and not restarted.stopped


def test_repeat_up_refuses_while_the_model_runs_on_some_sparks_only(deployment):
    directory, _ = deployment
    installer.apply(directory, "up", runner=Hosts(), execute=True)
    partial = Stopped({1})
    with pytest.raises(ValueError, match="runs on some Sparks but not on private-spark1. Stop it everywhere"):
        installer.apply(directory, "up", runner=partial, execute=True)
    assert {event[0] for event in partial.events} == {"running"}
    assert installer.status(directory)["state"]["complete"]


def test_repeat_up_names_a_spark_that_does_not_answer_instead_of_counting_it_stopped(deployment):
    directory, _ = deployment
    installer.apply(directory, "up", runner=Hosts(), execute=True)
    silent = Stopped({0}, silent={1})
    with pytest.raises(ValueError, match="private-spark1 did not report whether its model container runs: .*Connection timed out"):
        installer.apply(directory, "up", runner=silent, execute=True)
    assert not any(event[0] in ("start", "gid-serve") for event in silent.events)


def test_concurrent_up_and_down_are_excluded(deployment):
    directory, _ = deployment
    with process_lock.hold(directory / "operation.lock"):
        with pytest.raises(ValueError, match="Another operation"):
            installer.apply(directory, "down", runner=Hosts(), execute=True)


def test_dead_process_lock_does_not_require_manual_file_removal(deployment):
    directory, _ = deployment
    (directory / "operation.lock").write_text("interrupted process")
    assert installer.apply(directory, "down", runner=Hosts(), execute=True)["complete"]


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


SETTINGS = {"context_length": 65536, "kv_cache_gib": 16, "max_concurrency": 4, "max_images": 1}


def tuned_deployment(root):
    """A saved Qwen pair deployment on the shared installer image with serving settings SETTINGS."""
    data = b"offline source bundle fixture"
    lock = installer.make_lock(QWEN, site(), "1" * 40, hashlib.sha256(data).hexdigest(),
                               image_runtime=installer.installer_image.default_lock(), settings=SETTINGS)
    installer.write(root / "deployment.lock.json", lock)
    (root / "source.bundle").write_bytes(data)
    return root


def exported(root, *, share):
    """The files of an export of tuned_deployment, except its source bundle, by archive name."""
    output = root / "export.zip"
    installer.export(tuned_deployment(root / "deployment"), output, share=share)
    with zipfile.ZipFile(output) as archive:
        return {name: archive.read(name).decode() for name in archive.namelist() if name != "source.bundle"}


@pytest.mark.parametrize("share", [False, True])
def test_exports_render_the_deployments_serving_settings(tmp_path, share):
    from runtime.common import serving
    files = exported(tmp_path, share=share)
    for rank in (0, 1):
        container = json.loads(files[f"rank{rank}/container.json"])["command"]
        composed = yaml.safe_load(files[f"rank{rank}/compose.yaml"])["services"]["model"]["command"]
        for command in (container, composed):
            assert {name: serving.profile_value(command, name) for name in SETTINGS} == SETTINGS
    text = "\n".join(files.values())
    assert ("private-spark0" in text) is not share and ("/srv/private-weights" in text) is not share
    if not share:
        assert json.loads(files["deployment.lock.json"])["serving"] == SETTINGS


def test_the_shared_templates_init_command_saves_the_deployments_serving_settings(tmp_path, monkeypatch):
    files = exported(tmp_path, share=True)
    line = next(line for line in files["README.txt"].splitlines() if "then run sparkring init" in line)
    argv = line.split("then run sparkring ", 1)[1].rstrip(".").split()
    assert argv[-8:] == ["--context-length", "65536", "--kv-cache-gib", "16", "--max-concurrency", "4",
                         "--max-images", "1"]
    for name in ("site.example.json", "image-lock.json"):
        (tmp_path / name).write_text(files[name])
    calls = []
    monkeypatch.setattr(installer, "init", lambda *a, **k: calls.append((a, k)))
    monkeypatch.chdir(tmp_path)
    assert sparkring.main(argv) == 0
    assert calls[0][0][1] == QWEN and calls[0][1]["settings"] == SETTINGS


def test_a_shared_export_carries_the_save_cpu_switch_as_a_flag_and_a_container_variable(tmp_path, monkeypatch):
    data = b"offline source bundle fixture"
    settings = {"max_images": 1, "save_cpu": True}
    lock = installer.make_lock(QWEN, site(), "1" * 40, hashlib.sha256(data).hexdigest(),
                               image_runtime=installer.installer_image.default_lock(), settings=settings)
    (tmp_path / "deployment").mkdir()
    installer.write(tmp_path / "deployment" / "deployment.lock.json", lock)
    (tmp_path / "deployment" / "source.bundle").write_bytes(data)
    installer.export(tmp_path / "deployment", tmp_path / "export.zip", share=True)
    with zipfile.ZipFile(tmp_path / "export.zip") as archive:
        files = {name: archive.read(name).decode() for name in archive.namelist() if name != "source.bundle"}
    for rank in (0, 1):
        environment = yaml.safe_load(files[f"rank{rank}/compose.yaml"])["services"]["model"]["environment"]
        assert environment["SPARKRING_SHM_BUSY_LOOP_S"] == "0.002"
    line = next(line for line in files["README.txt"].splitlines() if "then run sparkring init" in line)
    argv = line.split("then run sparkring ", 1)[1].rstrip(".").split()
    assert argv[-3:] == ["--max-images", "1", "--save-cpu"]
    for name in ("site.example.json", "image-lock.json"):
        (tmp_path / name).write_text(files[name])
    calls = []
    monkeypatch.setattr(installer, "init", lambda *a, **k: calls.append((a, k)))
    monkeypatch.chdir(tmp_path)
    assert sparkring.main(argv) == 0
    assert calls[0][1]["settings"] == settings


def test_standalone_compose_export_refuses_a_deployment_with_serving_settings(tmp_path):
    data = b"offline source bundle fixture"
    lock = installer.make_lock(QWEN, site(), "1" * 40, hashlib.sha256(data).hexdigest(), settings={"max_images": 1})
    installer.write(tmp_path / "deployment" / "deployment.lock.json", lock)
    (tmp_path / "deployment" / "source.bundle").write_bytes(data)
    assert sparkring_installer.main(["export", "--format", "compose", "--deployment", str(tmp_path / "deployment"),
                                     "--output", str(tmp_path / "compose.yaml")]) == 2
    assert not (tmp_path / "compose.yaml").exists()


def test_managed_glm_uses_managed_native_checks_and_lifecycle():
    lock = installer.make_lock(installer.GLM_LEGACY[4], site(4), "1" * 40, "2" * 64)
    phases = [entry["id"] for entry in installer.operation_plan(lock, "up")["phases"]]
    assert phases.index("managed-native-check") < phases.index("managed-start")
    assert "managed-install" in phases
    assert "start-api" not in phases
    assert all(r["cache"] == installer.managed_workspace(lock["site"]["name"]) + "/cache" for r in lock["site"]["ranks"])


def test_a_pair_restores_roce_gid_index_3_before_its_preflight():
    from scripts import deploy_engine
    lock = installer.make_lock("qwen38-flash-next-tp2", site(2), "1" * 40, "2" * 64)
    plan = installer.operation_plan(lock, "up")
    deploy_engine.validate_plan(plan)
    phases = [entry["id"] for entry in plan["phases"]]
    assert phases.index("image") < phases.index("gid-serve") < phases.index("preflight") < phases.index("create")
    serve = next(entry for entry in plan["phases"] if entry["id"] == "gid-serve")
    assert [action["verify"]["argv"][1] for action in serve["actions"]] == ["gid-check", "gid-check"]
    assert "ring-serve" not in phases


def test_managed_glm_rejects_cache_paths_its_stager_cannot_honor():
    raw = site(4)
    raw["hosts"][0]["cache"] = "/srv/external-cache"
    with pytest.raises(ValueError, match="Managed GLM cache"):
        installer.make_lock(installer.GLM_LEGACY[4], raw, "1" * 40, "2" * 64)


def test_profile_refusals_state_the_condition_for_every_model():
    with pytest.raises(ValueError, match="does not deploy deepseek-v41-flash-cycle; 'sparkring models'"):
        installer.make_lock("deepseek-v41-flash-cycle", site(4), "1" * 40, "2" * 64)
    with pytest.raises(ValueError, match="This four-Spark profile needs the prepared mesh fabric reference"):
        installer.make_lock("deepseek-v41-flash-tp4", site(4), "1" * 40, "2" * 64)
    with pytest.raises(ValueError, match="native-mesh plan requires a four-Spark Compose profile"):
        installer.make_lock("mimo-v26-flash-mopd-tp2", {**site(2), "native_mesh": {}}, "1" * 40, "2" * 64)


def test_cli_offline_init_never_discovers_hosts(tmp_path, monkeypatch, capsys):
    path = tmp_path / "site.json"
    path.write_text(json.dumps(site()))
    calls = []
    monkeypatch.setattr(sparkring_installer, "discover", lambda *a: pytest.fail("No SSH"))
    monkeypatch.setattr(installer, "init", lambda *a, **k: calls.append((a, k)))
    assert sparkring.main(["init", "--model", "glm53", "--site", str(path)]) == 0
    assert calls[0][0][1] == installer.DEFAULTS["glm53", 2]
    assert calls[0][1]["image_runtime"] == installer.installer_image.default_lock()
    assert "No hosts changed" in capsys.readouterr().out


def test_offline_glm_plan_cannot_be_executed_directly():
    card = installer.setup.selection(GLM)
    plan = tp2.render(0, "198.18.20.1", Path("/srv/models/test").as_posix(), Path("/srv/cache/test").as_posix(), None,
                      card["image_id"], planning_release=card["release"], r33_sparkcache=True,
                      site_values={"VLLM_HOST_IP": "198.18.20.1", "NCCL_SOCKET_IFNAME": "eth0", "GLOO_SOCKET_IFNAME": "eth0"})
    with pytest.raises(ValueError, match="Offline plans"):
        tp2.execute(plan, "create", {}, run=lambda *a, **k: pytest.fail("Docker"))


@pytest.mark.parametrize("variant", ["nvfp4-spark", "nvfp4-qad"])
@pytest.mark.parametrize("enabled", [False, True])
def test_planned_glm_compose_matches_admitted_adapter_settings(tmp_path, monkeypatch, variant, enabled):
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
    planned = tp2.adapt_release_plan(copy.deepcopy(base), publication, sparkcache=enabled,
                                    target_model_variant=variant, planning_contract=contract)
    admitted = tp2.adapt_release_plan(copy.deepcopy(base), receipt, sparkcache=enabled, target_model_variant=variant)
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


def test_a_listed_checkpoint_selects_its_revision_and_pins():
    card = installer.setup.selection(QWEN, "qad-step-4000")
    assert card["target_variant"] == "qad-step-4000"
    assert card["model_revision"] == "629bc3218833a38b475b719f34aa571666f4a03e"
    pins = installer.checkpoint_pins(card)
    contract = installer.checkpoint_contract(card)
    assert pins["files"]["config.json"]["sha256"] == contract["config_sha256"]
    assert pins["files"][pins["index"]]["sha256"] == contract["index_sha256"]
    assert installer.setup.selection(QWEN)["target_variant"] == "qad-step5500-ple1000"
    with pytest.raises(ValueError, match="lists"):
        installer.setup.selection(QWEN, "main")
    with pytest.raises(ValueError, match="does not accept"):
        installer.setup.selection("mimo-v26-flash-mopd-tp2", "qad-step-4000")


GLM_RING = "glm53-flash-nvfp4-spark-tp4"


def test_a_listed_checkpoint_of_another_repository_selects_its_own_pins_and_directory():
    default = installer.setup.selection(GLM_RING)
    assert (default["target_variant"], default["model_repository"]) == (
        "nvfp4-spark", "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark")
    directories = {installer.checkpoint_directory("ring", default)}
    for name, repository, revision in (
            ("nvfp4-qad", "local-inference-lab/GLM-5.3-Flash-NVFP4", "175ae8ce3b5af842b0d0140dbeb43e9cfc557c49"),
            ("nvidia-nvfp4", "nvidia/GLM-5.3-Flash-NVFP4", "da920bb0b9f4a06727223a349e55468e38352348")):
        card = installer.setup.selection(GLM_RING, name)
        assert (card["target_variant"], card["model_repository"], card["model_revision"]) == (name, repository, revision)
        assert card["image_id"] == default["image_id"] and card["nodes"] == 4
        pins = installer.checkpoint_pins(card)
        contract = installer.checkpoint_contract(card)
        assert (pins["repository"], pins["revision"]) == (repository, revision)
        assert pins["files"]["config.json"]["sha256"] == contract["config_sha256"]
        assert pins["files"][pins["index"]]["sha256"] == contract["index_sha256"]
        directory = installer.checkpoint_directory("ring", card)
        assert directory == f"/srv/sparkring/ring/checkpoints/{repository.replace('/', '--')}/{revision}"
        directories.add(directory)
    assert len(directories) == 3
    # The pair lists the QAD checkpoint, in the same directory as the ring's, and not NVIDIA's.
    pair = installer.setup.selection("glm53-flash-nvfp4-spark-tp2", "nvfp4-qad")
    assert (pair["model_repository"], pair["nodes"]) == ("local-inference-lab/GLM-5.3-Flash-NVFP4", 2)
    assert installer.checkpoint_directory("ring", pair) == installer.checkpoint_directory(
        "ring", installer.setup.selection(GLM_RING, "nvfp4-qad"))
    assert installer.checkpoint_pins(pair) == installer.checkpoint_pins(installer.setup.selection(GLM_RING, "nvfp4-qad"))
    with pytest.raises(ValueError, match="lists: nvfp4-qad, nvfp4-spark$"):
        installer.setup.selection("glm53-flash-nvfp4-spark-tp2", "nvidia-nvfp4")
    with pytest.raises(ValueError, match="lists: nvfp4-qad, nvfp4-spark, nvidia-nvfp4"):
        installer.setup.selection(GLM_RING, "qad-step-4000")


def test_a_listed_checkpoint_changes_the_deployment_lock_and_served_model():
    from runtime.common.test_compose_installer import install_site
    image = installer.installer_image.for_profile(GLM_RING)
    locks = {name: installer.make_lock(GLM_RING, install_site(4), "1" * 40, "2" * 64, name, image_runtime=image)
             for name in (None, "nvfp4-spark", "nvfp4-qad", "nvidia-nvfp4")}
    # Naming the default selects the same deployment as no name.
    assert locks[None] == locks["nvfp4-spark"]
    assert len({lock["id"] for lock in locks.values()}) == 3
    for name, served in (("nvfp4-spark", "GLM-5.3-Flash-NVFP4-Spark-TP4"), ("nvfp4-qad", "GLM-5.3-Flash-NVFP4-QAD-TP4"),
                         ("nvidia-nvfp4", "GLM-5.3-Flash-NVFP4-NVIDIA-TP4")):
        assert installer.validate(locks[name]) == locks[name]
        assert installer.connection(locks[name])["model"] == served
        assert installer.identity(locks[name])["checkpoint"] == name


QWEN_PINS = ("profiles/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/"
             "60215d26cf5e42c2db6128774032d57fc62678da.json")


def test_checkpoint_pins_agree_with_every_installer_profile():
    for profile in sorted(installer.INSTALLABLE):
        card = installer.setup.selection(profile)
        pins = installer.checkpoint_pins(card)
        contract = installer.checkpoint_contract(card)
        files = pins["files"]
        assert files["config.json"]["sha256"] == contract["config_sha256"]
        assert files[pins["index"]]["sha256"] == contract["index_sha256"]
        required = sorted(set(files) - set(pins["optional"]))
        sums = (installer.ROOT / "profiles" / profile / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        assert sums == [f"{files[name]['sha256']}  {name}" for name in required]
        # Every other checkpoint the profile lists has a pin manifest that agrees with its entry.
        configuration = installer.read(installer.ROOT / card["configuration"])
        for name in sorted(set(configuration.get("checkpoints", {})) - {card["target_variant"]}):
            other = installer.setup.selection(profile, name)
            pins = installer.checkpoint_pins(other)
            assert pins["files"]["config.json"]["sha256"] == configuration["checkpoints"][name]["model"]["config_sha256"]
            assert pins["files"][pins["index"]]["sha256"] == configuration["checkpoints"][name]["model"]["index_sha256"]
    # The Qwen revision: 56 files, of which 53 are served (41 weight files).
    qwen = installer.checkpoint_pins(installer.setup.selection(QWEN))
    required = set(qwen["files"]) - set(qwen["optional"])
    assert (len(qwen["files"]), len(required), len(qwen["weights"])) == (56, 53, 41)
    assert qwen["optional"] == [".gitattributes", "LICENSE", "README.md"]
    assert sum(qwen["files"][name]["size"] for name in qwen["weights"]) == 110_131_860_580
    assert sum(qwen["files"][name]["size"] for name in required - set(qwen["weights"])) == 56_080_100
    with pytest.raises(ValueError, match="No pin manifest"):
        installer.checkpoint_pins({**installer.setup.selection(QWEN), "model_revision": "0" * 40})


def _extra(name):
    return lambda pins: pins["files"].__setitem__(name, dict(pins["files"]["vocab.json"]))


def _entry(name, key, value):
    return lambda pins: pins["files"][name].__setitem__(key, value)


UNSAFE_PINS = {
    "schema": lambda pins: pins.update(schema="sparkring-checkpoint-pins/v0"),
    "unknown-key": lambda pins: pins.update(note="unexpected"),
    "repository": lambda pins: pins.update(repository="huginnfork/Qwen3.8-Flash-Next-NVFP4-Abliterated"),
    "revision": lambda pins: pins.update(revision="7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd"),
    "config-hash": _entry("config.json", "sha256", "b1ecb2697178111fb10b73354a681d10884c19ecd830af1073ea93d2a09a5358"),
    "index-hash": _entry("model.safetensors.index.json", "sha256", "0" * 64),
    "parent": _extra("../vocab.json"),
    "nested-parent": _extra("tokenizer/../../vocab.json"),
    "absolute": _extra("/etc/vocab.json"),
    "cache": _extra(".cache/huggingface/download/vocab.json.metadata"),
    "git": _extra("sub/.git/config"),
    "nul": _extra("vocab\0.json"),
    "cr": _extra("vocab\r.json"),
    "lf": _extra("vocab\n.json"),
    "backslash": _extra("sub\\vocab.json"),
    "double-slash": _extra("sub//vocab.json"),
    "dot": _extra("./vocab2.json"),
    "trailing-slash": _extra("sub/"),
    "empty": _extra(""),
    "file-and-directory": _extra("config.json/extra.json"),
    "size-zero": _entry("vocab.json", "size", 0),
    "size-negative": _entry("vocab.json", "size", -1),
    "size-text": _entry("vocab.json", "size", "6722759"),
    "size-bool": _entry("vocab.json", "size", True),
    "size-float": _entry("vocab.json", "size", 6722759.0),
    "sha256-upper": _entry("vocab.json", "sha256", "CE99B4CB2983D118806CE0A8B777A35B093E2000A503EBDE25853284C9DFA003"),
    "sha256-short": _entry("vocab.json", "sha256", "ce99b4cb"),
    "git-blob-short": _entry("vocab.json", "git_blob", "0aa0ce0658d60ac4a5d609f4eadb0e8e4351417"),
    "lfs-false": _entry("vocab.json", "lfs", False),
    "xet-hash": _entry("model-00001-of-00041.safetensors", "xet_hash", "x"),
    "entry-key": _entry("vocab.json", "sha265", "0" * 64),
    "entry-missing-sha256": lambda pins: pins["files"]["vocab.json"].pop("sha256"),
    "index-optional": lambda pins: pins["optional"].append("model.safetensors.index.json"),
    "index-unpinned": lambda pins: pins.update(index="model.index.json"),
    "config-optional": lambda pins: pins["optional"].append("config.json"),
    "weight-unpinned": lambda pins: pins.update(weights=sorted(pins["weights"] + ["model-00042-of-00041.safetensors"])),
    "weight-optional": lambda pins: pins["optional"].append("model-00001-of-00041.safetensors"),
    "weights-unsorted": lambda pins: pins["weights"].reverse(),
    "weights-duplicate": lambda pins: pins["weights"].insert(0, pins["weights"][0]),
    "weights-empty": lambda pins: pins.update(weights=[]),
    "optional-unpinned": lambda pins: pins["optional"].append("LICENSE"),
}


@pytest.mark.parametrize("change", UNSAFE_PINS.values(), ids=UNSAFE_PINS.keys())
def test_checkpoint_pins_reject_unsafe_or_malformed_entries(tmp_path, change):
    card = installer.setup.selection(QWEN)
    pins = installer.checkpoint_pins(card)
    path = tmp_path / QWEN_PINS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(pins), encoding="utf-8")
    assert installer.checkpoint_pins(card, root=tmp_path) == pins
    changed = copy.deepcopy(pins)
    change(changed)
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        installer.checkpoint_pins(card, root=tmp_path)


def test_checkpoint_directory_is_per_cluster_and_revision_and_disjoint():
    cards = {profile: installer.setup.selection(profile) for profile in sorted(installer.INSTALLABLE)}
    qwen = installer.checkpoint_directory("tp2", cards[QWEN])
    assert qwen == ("/srv/sparkring/tp2/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/"
                    "60215d26cf5e42c2db6128774032d57fc62678da")
    # One directory per cluster and revision, whatever the profile or node count.
    assert installer.checkpoint_directory({"name": "tp2"}, cards["qwen38-flash-next-qad-tp4"]) == qwen
    assert installer.checkpoint_directory("tp4-installer", cards[QWEN]) != qwen
    main = {**cards[QWEN], "model_revision": "7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd"}
    assert installer.checkpoint_directory("tp2", main) == qwen.rsplit("/", 1)[0] + "/" + main["model_revision"]
    assert len({installer.checkpoint_directory("tp2", card) for card in cards.values()}) == 5
    # Deployment workspaces are named after profiles, so none is the checkpoints or cache directory.
    assert not {"checkpoints", "cache"} & set(installer.profiles.catalog())
    for profile, card in cards.items():
        model = installer.checkpoint_directory("tp2", card)
        raw = {"schema": "sparkring-install-site/v1", "name": "tp2-site", "workspace": "/srv/sparkring/tp2/" + profile,
               "hosts": [{"host": f"spark{n}", "management_ip": f"192.0.2.{20 + n}", "fabric_ip": f"198.18.20.{n + 1}",
                          "interface": "enp1s0f0np0", "model": model, "cache": "/srv/sparkring/tp2/cache"}
                         for n in range(card["nodes"])]}
        if profile in compose.TP4_PROFILES:
            for row in raw["hosts"]:
                row["fabric"] = {"site_path": "/srv/sparkring/mesh-site.json", "site_sha256": "0" * 64, "plan_sha256": "0" * 64}
        for row in installer.site_document(raw, card, "1" * 40)["ranks"]:
            assert row["model"] == model
            for key in ("cache", "repository", "deployment_root"):
                other = PurePosixPath(row[key])
                assert not PurePosixPath(model).is_relative_to(other) and not other.is_relative_to(model)
    for name in ("", "TP2", "tp2/../x", "../tp2", "tp2/", "-tp2", "a" * 41, None, {"name": "../tp2"}):
        with pytest.raises(ValueError):
            installer.checkpoint_directory(name, cards[QWEN])
    for field, value in (("model_revision", "qad-step-4000"), ("model_repository", "../etc"),
                         ("model_repository", "owner--name/model"), ("model_repository", "owner/na..me")):
        with pytest.raises(ValueError):
            installer.checkpoint_directory("tp2", {**cards[QWEN], field: value})
