"""Offline installer barriers, immutable inputs, Compose generation and sharing."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

import pytest

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


def test_managed_glm_uses_managed_native_checks_and_lifecycle():
    lock = installer.make_lock(installer.GLM_LEGACY[4], site(4), "1" * 40, "2" * 64)
    phases = [entry["id"] for entry in installer.operation_plan(lock, "up")["phases"]]
    assert phases.index("managed-native-check") < phases.index("managed-start")
    assert "managed-install" in phases
    assert "start-api" not in phases
    assert all(r["cache"] == installer.managed_workspace(lock["site"]["name"]) + "/cache" for r in lock["site"]["ranks"])


def test_managed_glm_rejects_cache_paths_its_stager_cannot_honor():
    raw = site(4)
    raw["hosts"][0]["cache"] = "/srv/external-cache"
    with pytest.raises(ValueError, match="Managed GLM cache"):
        installer.make_lock(installer.GLM_LEGACY[4], raw, "1" * 40, "2" * 64)


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
    assert len({installer.checkpoint_directory("tp2", card) for card in cards.values()}) == 3
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
