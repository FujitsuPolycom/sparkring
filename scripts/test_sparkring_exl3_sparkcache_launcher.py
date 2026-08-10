from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bootstrap_exl3  # noqa: E402
import exl3_sparkcache_config as candidate_config  # noqa: E402
import sparkring_exl3_launcher as exl3  # noqa: E402
import sparkring_exl3_lmcache_launcher as lmcache  # noqa: E402
import sparkring_exl3_sparkcache_launcher as launcher  # noqa: E402
from checkpoint_manifest_generator import FileEntry, build_receipt, compute_identity  # noqa: E402
from sparkring_site import load_site  # noqa: E402


IMAGE_ID = "sha256:" + "a" * 64
BUNDLE = "d" * 64


def checkpoint_receipt():
    receipt = build_receipt(
        [FileEntry("model.safetensors", 7, "c" * 64)],
        artifact_root_name="glm52-exl3",
    )
    receipt["checkpoint_identity_sha256"] = compute_identity(receipt)
    return receipt


TARGET = checkpoint_receipt()["checkpoint_identity_sha256"]


def generated(tmp_path):
    site_path = tmp_path / "site.yaml"
    profile_path = tmp_path / "launch.json"
    candidate_path = tmp_path / "candidate.json"
    bootstrap_exl3.write_generated_site(
        ROOT / "scripts/config/site.example.yaml",
        site_path,
        "sparkring/exl3:test",
        IMAGE_ID,
    )
    bootstrap_exl3.write_generated_profile(
        profile_path,
        "sparkring/exl3:test",
        IMAGE_ID,
        "/srv/models/exl3",
        "/srv/jit",
    )
    baseline = exl3.load_profile(profile_path)
    candidate = candidate_config.build_candidate(
        baseline.document,
        checkpoint_receipt=checkpoint_receipt(),
        connector_bundle_identity=BUNDLE,
        connector_staging_host="/srv/sparkcache/staging",
        cache_root_host="/srv/sparkcache/context",
    )
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    return site_path, profile_path, candidate_path, candidate


def args(site, profile, candidate, command="plan"):
    return [
        "--site",
        str(site),
        "--baseline-profile",
        str(profile),
        "--candidate",
        str(candidate),
        command,
    ]


def test_plan_is_dry_run_rank_complete_and_discloses_rollback(tmp_path, capsys):
    site, profile, candidate, _ = generated(tmp_path)
    assert launcher.main(args(site, profile, candidate)) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["command"] == "plan"
    assert plan["rank_completeness"]["all_phases_have_four_ranks"] is True
    assert plan["lifecycle"] == launcher.lifecycle("cutover")
    assert plan["rollback_target"].startswith("canonical EXL3+LMCache")
    assert set(plan["phases"]) == {
        "candidate_precheck",
        "candidate_absent",
        "candidate_absent_final",
        "checkpoint_quiescent",
        "checkpoint_full_attest",
        "checkpoint_helper_absent",
        "checkpoint_helper_remove",
        "candidate_start",
        "candidate_api_barrier",
        "candidate_final_status",
        "candidate_restart",
        "candidate_remove",
        "baseline_server_health",
        "baseline_ready",
        "baseline_remove_engines",
        "baseline_remove_servers",
        "baseline_start_servers",
            "baseline_start_engines",
            "baseline_final_status",
            "verified_start_prepare",
    }


def test_output_writes_exclusive_receipt_and_prints_digest(tmp_path, capsys):
    site, profile, candidate, _ = generated(tmp_path)
    output = tmp_path / "receipt.json"
    argv = args(site, profile, candidate) + ["--output", str(output)]

    assert launcher.main(argv) == 0

    summary = json.loads(capsys.readouterr().out)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema"] == launcher.PLAN_SCHEMA
    assert summary["output"] == str(output)
    assert len(summary["sha256"]) == 64
    with pytest.raises(SystemExit):
        launcher.main(argv)


def test_candidate_actions_pin_bundle_mounts_image_model_and_patch_attestation(tmp_path):
    site_path, profile_path, _candidate_path, candidate = generated(tmp_path)
    site = load_site(site_path)
    baseline = exl3.load_profile(profile_path)
    state = launcher.validate_candidate(candidate, baseline)
    starts = launcher.candidate_start_actions(site, state)
    attest = launcher.precheck_actions(site, baseline, state)
    full_attest = launcher.full_checkpoint_attestation_actions(site, state)
    quiescent = launcher.checkpoint_quiescent_actions(site, baseline, state)
    barrier = launcher.candidate_api_barrier_actions(site, state)
    status = launcher.candidate_final_status_actions(site, baseline, state)
    assert len(starts) == len(attest) == len(full_attest) == len(quiescent) == len(barrier) == len(status) == 4
    for action in starts:
        command = action.shell_command
        assert IMAGE_ID in command
        assert "org.sparkring.component=sparkcache-engine" in command
        assert "--privileged" in command
        assert BUNDLE in command
        assert "/srv/sparkcache/staging:/opt/sparkcache-staging:ro" in command
        assert "/srv/sparkcache/context:/cache/context" in command
        assert "/srv/models/exl3:/models/glm52-exl3-tr3-3.25bpw:ro" in command
        assert "SparkContextCacheConnector" in command
        assert "verify_exl3_model.py" not in command
        assert launcher.verified_start.REMOTE_OUTER_VERIFIED_ENTRYPOINT in command
        assert "SPARKRING_OUTER_MODEL_VERIFIED=1" in command
        assert "echo 3 > /proc/sys/vm/drop_caches" in command
    for action in attest:
        command = action.shell_command
        assert "async_tokens_to_discard" in command
        assert "num_output_placeholders" in command
        assert "_validate_kv_transfer_vmm" in command
        assert "SparkContextCacheConnector" in command
        assert BUNDLE in command
        assert "docker run --rm" not in command
        assert "docker exec" in command
        assert "checkpoint_identity_sha256" in command
        assert TARGET in command
        assert "onerror=walk_error" in command
        assert "with path.open('rb')" not in command
        assert "verify_exl3_model.py" not in command
        # Both the candidate and the live canonical baseline are started via
        # the shared outer-verification/page-cache-reclaim contract.
        assert command.count(
            launcher.verified_start.REMOTE_OUTER_VERIFIED_ENTRYPOINT
        ) >= 2
    for action in quiescent:
        command = action.shell_command
        assert "nvidia-smi --query-compute-apps=pid" in command
        assert "/srv/models/exl3" in command
        assert "subprocess.check_output" in command
        assert "'ps'" in command and "'-q'" in command
    for action in full_attest:
        command = action.shell_command
        assert "docker run --rm" in command
        assert "--network none" in command
        assert "--gpus" not in command
        assert IMAGE_ID in command
        assert "/srv/models/exl3:/models/glm52-exl3-tr3-3.25bpw:ro" in command
        assert "sparkcache-checkpoint-manifest-v2" in command
        assert TARGET in command
    assert "for i in $(seq 1 720)" in barrier[0].shell_command
    assert "/v1/models" in barrier[0].shell_command
    for action in status:
        command = action.shell_command
        assert "org.sparkring.sparkcache-candidate-sha256" in command
        assert "--disable-hybrid-kv-cache-manager" in command
        assert "/cache/context" in command
        assert "for i in $(seq" not in command
        assert "contract_json" in command
        assert "environment_overrides" in command
        assert "sparkcache-checkpoint-manifest-v2" not in command
        assert "verify_exl3_model.py" not in command
        assert "async_failed_req_ids" in command
        assert command.count("onerror=walk_error") >= 2
        for port in site.rank(action.rank).ring_ports:
            assert f"/sys/class/net/{port.interface}/carrier" in command
            assert f"/sys/class/net/{port.interface}/operstate" in command
        if action.rank == 0:
            assert "/v1/models" in command

    receipts = launcher.baseline_final_status_actions(site, baseline, state)
    assert len(receipts) == 4
    for action in receipts:
        for port in site.rank(action.rank).ring_ports:
            assert f"/sys/class/net/{port.interface}/carrier" in action.shell_command
            assert f"/sys/class/net/{port.interface}/operstate" in action.shell_command


def test_live_layout_precheck_reads_metadata_not_checkpoint_contents(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    shard = model / "model.safetensors"
    shard.write_bytes(b"before!")
    receipt = {
        "checkpoint_identity_sha256": TARGET,
        "file_count": 1,
        "files": [
            {
                "rel_path": "model.safetensors",
                "byte_size": 7,
                "content_sha256": "c" * 64,
            }
        ],
    }
    monkeypatch.setattr(
        sys,
        "argv",
        ["precheck", str(model), json.dumps(receipt), TARGET],
    )
    exec(launcher._CHECKPOINT_LAYOUT_PRECHECK, {})
    shard.write_bytes(b"changed")
    exec(launcher._CHECKPOINT_LAYOUT_PRECHECK, {})
    shard.write_bytes(b"wrong-size")
    with pytest.raises(AssertionError):
        exec(launcher._CHECKPOINT_LAYOUT_PRECHECK, {})


@pytest.mark.parametrize(
    "source",
    ("/srv/models/exl3", "/srv/models", "/srv/models/exl3/shard"),
)
def test_quiescence_attest_rejects_any_overlapping_model_mount(monkeypatch, source):
    document = {
        "Name": "/unrelated-name",
        "Mounts": [{"Source": source, "Destination": "/model"}],
    }

    def check_output(command, text=False):
        if command == ["docker", "ps", "-q"]:
            return "abc\n"
        payload = json.dumps([document])
        return payload if text else payload.encode()

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(
        sys,
        "argv",
        ["quiescent", "/srv/models/exl3", "expected-source-engine"],
    )
    with pytest.raises(AssertionError):
        exec(launcher._NO_MODEL_CONTAINERS_ATTEST, {})
    document["Mounts"][0]["Source"] = "/srv/other"
    exec(launcher._NO_MODEL_CONTAINERS_ATTEST, {})
    document["Name"] = "/expected-source-engine"
    with pytest.raises(AssertionError):
        exec(launcher._NO_MODEL_CONTAINERS_ATTEST, {})


def test_candidate_orchestrator_rejects_podman_before_plan_composition(tmp_path):
    _site, profile_path, _candidate_path, candidate = generated(tmp_path)
    document = json.loads(profile_path.read_text(encoding="utf-8"))
    document["engine"] = "podman"
    baseline = exl3.Profile(document)
    with pytest.raises(launcher.LauncherError, match="requires engine=docker"):
        launcher.validate_candidate(candidate, baseline)


def test_owned_remove_and_restart_are_exact_label_guarded(tmp_path):
    site_path, profile_path, _candidate_path, candidate = generated(tmp_path)
    site = load_site(site_path)
    state = launcher.validate_candidate(candidate, exl3.load_profile(profile_path))
    for operation in ("remove", "restart"):
        actions = launcher._owned_action(site, state, operation)
        for action in actions:
            command = action.shell_command
            assert "exit 73" in command
            assert "exit 74" in command
            assert "exit 75" in command
            assert "exit 76" in command
            assert "exit 78" in command
            assert BUNDLE in command
            assert state["candidate_sha256"] in command
            assert profile_path.parent.as_posix() not in command
            assert "/srv/models/exl3" in command
            assert "/srv/sparkcache/staging" in command
            assert "/srv/sparkcache/context" in command
            if operation == "remove":
                assert "docker info" in command
                assert "exit 0" in command
                assert "exit 77" in command


def status_fixture(tmp_path):
    site_path, profile_path, _candidate_path, candidate = generated(tmp_path)
    site = load_site(site_path)
    baseline = exl3.load_profile(profile_path)
    state = launcher.validate_candidate(candidate, baseline)
    start = launcher.candidate_start_actions(site, state)[0]
    contract = launcher.docker_run_contract(start, IMAGE_ID)
    image_doc = {
        "Config": {"Env": [], "Entrypoint": ["/entry"], "Labels": {}, "Volumes": None}
    }
    labels = dict(contract["labels"])
    mounts = [
        {
            "Destination": destination,
            "Source": item["source"],
            "RW": not item["read_only"],
        }
        for destination, item in contract["mounts"].items()
    ]
    host = contract["host_config"]
    document = {
        "State": {"Running": True, "OOMKilled": False},
        "RestartCount": 0,
        "Image": IMAGE_ID,
        "Config": {
            "Labels": labels,
            "Cmd": contract["cmd"],
            "Env": [f"{key}={value}" for key, value in contract["environment_overrides"].items()],
            "Entrypoint": (
                [contract["entrypoint_override"]]
                if contract["entrypoint_override"] is not None
                else ["/entry"]
            ),
        },
        "Mounts": mounts,
        "HostConfig": json.loads(json.dumps(host)),
    }
    args = [
        "attest",
        exl3.container_name(state["profile"], 0),
        IMAGE_ID,
        state["profile"].profile_id,
        BUNDLE,
        state["candidate_sha256"],
        state["profile"].model_host_path,
        state["profile"].model_container_path,
        state["staging"],
        candidate_config.STAGING_DESTINATION,
        state["cache_root"],
        candidate_config.CACHE_DESTINATION,
        state["profile"].extra_vllm_args[
            state["profile"].extra_vllm_args.index("--kv-transfer-config") + 1
        ],
        json.dumps(contract),
    ]
    return document, image_doc, args


def run_status_script(monkeypatch, document, image_doc, args):
    def check_output(command):
        selected = image_doc if command[1:3] == ["image", "inspect"] else document
        return json.dumps([selected]).encode()

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(sys, "argv", args)
    exec(launcher._STATUS_ATTEST, {})


def baseline_removal_fixture(tmp_path, component="lmcache-server"):
    site_path, profile_path, _candidate_path, _candidate = generated(tmp_path)
    site = load_site(site_path)
    baseline = exl3.load_profile(profile_path)
    if component == "lmcache-server":
        action = lmcache.server_start_actions(site, baseline)[0]
        name = lmcache.server_name(0)
    else:
        action = lmcache.engine_start_actions(site, baseline)[0]
        name = exl3.container_name(baseline, 0)
    contract = launcher.docker_run_contract(action, IMAGE_ID)
    image_doc = {
        "Config": {"Env": [], "Entrypoint": ["/entry"], "Labels": {}, "Volumes": None}
    }
    mounts = [
        {
            "Destination": destination,
            "Source": item["source"],
            "RW": not item["read_only"],
        }
        for destination, item in contract["mounts"].items()
    ]
    document = {
        "Id": "container-id-0",
        "Name": f"/{name}",
        "Image": IMAGE_ID,
        "Config": {
            "Labels": dict(contract["labels"]),
            "Cmd": contract["cmd"],
            "Env": [
                f"{key}={value}"
                for key, value in contract["environment_overrides"].items()
            ],
            "Entrypoint": (
                [contract["entrypoint_override"]]
                if contract["entrypoint_override"] is not None
                else ["/entry"]
            ),
        },
        "Mounts": mounts,
        "HostConfig": json.loads(json.dumps(contract["host_config"])),
    }
    return site, baseline, name, contract, document, image_doc


def run_removal_attest(monkeypatch, name, contract, document, image_doc):
    def check_output(command):
        selected = image_doc if command[1:3] == ["image", "inspect"] else document
        return json.dumps([selected]).encode()

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(
        sys,
        "argv",
        ["attest", name, IMAGE_ID, json.dumps(contract)],
    )
    exec(launcher._EXACT_REMOVAL_ATTEST, {})


def test_baseline_removal_rechecks_full_contract_and_removes_attested_id(
    monkeypatch, tmp_path
):
    site, baseline, name, contract, document, image_doc = baseline_removal_fixture(
        tmp_path
    )
    run_removal_attest(monkeypatch, name, contract, document, image_doc)
    actions = launcher.baseline_exact_remove_actions(
        site, baseline, component="lmcache-server"
    )
    command = actions[0].shell_command
    assert "ident=$(python3 -c" in command
    assert 'docker rm --force "$ident"' in command
    assert IMAGE_ID in command
    assert "contract_json" in command


def apply_live_docker_hostconfig_defaults(document):
    host = document["HostConfig"]
    for key in ("Binds", "CapAdd", "Dns", "MaskedPaths", "ReadonlyPaths"):
        if host[key] == []:
            host[key] = None
    for key in ("Init", "Mounts", "StorageOpt", "Sysctls", "Tmpfs"):
        host.pop(key)
    assert host["Privileged"] is True and host["SecurityOpt"] is None
    host["SecurityOpt"] = ["label=disable"]


def test_baseline_removal_accepts_live_docker_normalized_semantic_defaults(
    monkeypatch, tmp_path
):
    _site, _baseline, name, contract, document, image_doc = baseline_removal_fixture(
        tmp_path
    )
    apply_live_docker_hostconfig_defaults(document)
    run_removal_attest(monkeypatch, name, contract, document, image_doc)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("Binds", ["/unexpected:/unexpected:rw"]),
        ("CapAdd", ["SYS_ADMIN"]),
        ("Dns", ["192.0.2.53"]),
        ("MaskedPaths", ["/proc/kcore"]),
        ("ReadonlyPaths", ["/proc/sys"]),
        ("StorageOpt", {"size": "1G"}),
        ("SecurityOpt", ["label=disable", "apparmor=unconfined"]),
    ),
)
def test_baseline_removal_rejects_nondefault_hostconfig_after_normalization(
    monkeypatch, tmp_path, field, value
):
    _site, _baseline, name, contract, document, image_doc = baseline_removal_fixture(
        tmp_path
    )
    apply_live_docker_hostconfig_defaults(document)
    document["HostConfig"][field] = value
    with pytest.raises(AssertionError):
        run_removal_attest(monkeypatch, name, contract, document, image_doc)


def test_label_disable_is_equivalent_only_for_privileged_expected_contract():
    expected = {"Privileged": False, "SecurityOpt": None}
    actual = {"Privileged": False, "SecurityOpt": ["label=disable"]}
    assert launcher.normalize_docker_host_config(actual, expected) != expected


def test_docker_capadd_prefix_is_semantically_equivalent():
    expected = {"CapAdd": ["IPC_LOCK"]}
    actual = {"CapAdd": ["CAP_IPC_LOCK"]}
    assert launcher.normalize_docker_host_config(actual, expected) == expected


def test_docker_bind_order_is_semantically_equivalent():
    expected = {
        "Binds": ["/model:/model:ro", "/cache:/cache:rw", "/entry:/entry:ro"]
    }
    actual = {
        "Binds": ["/cache:/cache:rw", "/entry:/entry:ro", "/model:/model:ro"]
    }
    assert launcher.normalize_docker_host_config(actual, expected) == expected


def test_docker_bind_reordering_does_not_hide_mount_drift():
    expected = {"Binds": ["/model:/model:ro", "/cache:/cache:rw"]}
    actual = {"Binds": ["/cache:/cache:rw", "/other:/model:ro"]}
    assert launcher.normalize_docker_host_config(actual, expected) != expected


@pytest.mark.parametrize(
    "actual",
    (
        ["CAP_SYS_ADMIN"],
        ["CAP_IPC_LOCK", "CAP_SYS_ADMIN"],
        ["CAP_SYS_ADMIN", "CAP_IPC_LOCK"],
    ),
)
def test_docker_capadd_prefix_does_not_hide_capability_drift(actual):
    expected = {"CapAdd": ["IPC_LOCK"]}
    assert launcher.normalize_docker_host_config({"CapAdd": actual}, expected) != expected


@pytest.mark.parametrize(
    ("field", "meaningful"),
    (
        ("Init", True),
        ("Mounts", [{"Type": "tmpfs", "Target": "/tmp"}]),
        ("StorageOpt", {"size": "1G"}),
        ("Sysctls", {"net.ipv4.ip_forward": "1"}),
        ("Tmpfs", {"/tmp": "rw"}),
    ),
)
def test_missing_hostconfig_key_is_allowed_only_for_generated_inert_default(
    tmp_path, field, meaningful
):
    _site, _baseline, _name, contract, document, _image_doc = (
        baseline_removal_fixture(tmp_path)
    )
    expected = contract["host_config"]
    document["HostConfig"].pop(field)
    assert launcher.normalize_docker_host_config(document["HostConfig"], expected) == expected
    meaningful_expected = dict(expected)
    meaningful_expected[field] = meaningful
    with pytest.raises(AssertionError, match="missing non-inert field"):
        launcher.normalize_docker_host_config(
            document["HostConfig"], meaningful_expected
        )


@pytest.mark.parametrize(
    ("field", "meaningful"),
    (
        ("Init", True),
        ("Mounts", [{"Type": "tmpfs", "Target": "/tmp"}]),
        ("StorageOpt", {"size": "1G"}),
        ("Sysctls", {"net.ipv4.ip_forward": "1"}),
        ("Tmpfs", {"/tmp": "rw"}),
    ),
)
def test_present_nondefault_hostconfig_value_remains_rejected(
    tmp_path, field, meaningful
):
    _site, _baseline, _name, contract, document, _image_doc = (
        baseline_removal_fixture(tmp_path)
    )
    document["HostConfig"][field] = meaningful
    assert (
        launcher.normalize_docker_host_config(
            document["HostConfig"], contract["host_config"]
        )
        != contract["host_config"]
    )


@pytest.mark.parametrize(
    "mutation", ("image", "label", "mount", "runtime", "command")
)
def test_baseline_removal_rejects_contract_drift(
    monkeypatch, tmp_path, mutation
):
    _site, _baseline, name, contract, document, image_doc = baseline_removal_fixture(
        tmp_path
    )
    if mutation == "image":
        document["Image"] = "sha256:" + "b" * 64
    elif mutation == "label":
        document["Config"]["Labels"]["unexpected"] = "x"
    elif mutation == "mount":
        document["Mounts"].append(
            {"Destination": "/unexpected", "Source": "/tmp/x", "RW": True}
        )
    elif mutation == "runtime":
        document["HostConfig"]["Runtime"] = "nvidia"
    elif mutation == "command":
        document["Config"]["Cmd"] = ["wrong"]
    else:
        raise AssertionError(mutation)
    with pytest.raises(AssertionError):
        run_removal_attest(monkeypatch, name, contract, document, image_doc)


def test_exact_container_contract_accepts_only_canonical_envelope(monkeypatch, tmp_path):
    document, image_doc, args = status_fixture(tmp_path)
    run_status_script(monkeypatch, document, image_doc, args)


def test_candidate_status_accepts_live_docker_normalized_semantic_defaults(
    monkeypatch, tmp_path
):
    document, image_doc, args = status_fixture(tmp_path)
    apply_live_docker_hostconfig_defaults(document)
    run_status_script(monkeypatch, document, image_doc, args)


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_mount", "extra_label", "network", "ipc", "privileged",
        "shm", "caps", "devices", "gpu", "restart", "autoremove",
        "readonly", "security", "pid_namespace", "user_namespace",
        "memory", "cpus", "pids", "runtime",
    ),
)
def test_exact_container_contract_rejects_extra_or_hostconfig_drift(
    monkeypatch, tmp_path, mutation
):
    document, image_doc, args = status_fixture(tmp_path)
    if mutation == "extra_mount":
        document["Mounts"].append(
            {"Destination": "/unexpected", "Source": "/tmp/x", "RW": True}
        )
    elif mutation == "extra_label":
        document["Config"]["Labels"]["unexpected"] = "x"
    elif mutation == "network":
        document["HostConfig"]["NetworkMode"] = "bridge"
    elif mutation == "ipc":
        document["HostConfig"]["IpcMode"] = "private"
    elif mutation == "privileged":
        document["HostConfig"]["Privileged"] = False
    elif mutation == "shm":
        document["HostConfig"]["ShmSize"] += 1
    elif mutation == "caps":
        document["HostConfig"]["CapAdd"] = []
    elif mutation == "devices":
        document["HostConfig"]["Devices"] = []
    elif mutation == "gpu":
        document["HostConfig"]["DeviceRequests"] = []
    elif mutation == "restart":
        document["HostConfig"]["RestartPolicy"] = {
            "Name": "always", "MaximumRetryCount": 0,
        }
    elif mutation == "autoremove":
        document["HostConfig"]["AutoRemove"] = True
    elif mutation == "readonly":
        document["HostConfig"]["ReadonlyRootfs"] = True
    elif mutation == "security":
        document["HostConfig"]["SecurityOpt"] = ["no-new-privileges"]
    elif mutation == "pid_namespace":
        document["HostConfig"]["PidMode"] = "host"
    elif mutation == "user_namespace":
        document["HostConfig"]["UsernsMode"] = "host"
    elif mutation == "memory":
        document["HostConfig"]["Memory"] = 1024
    elif mutation == "cpus":
        document["HostConfig"]["NanoCpus"] = 1_000_000_000
    elif mutation == "pids":
        document["HostConfig"]["PidsLimit"] = 128
    elif mutation == "runtime":
        document["HostConfig"]["Runtime"] = "nvidia"
    else:
        raise AssertionError(mutation)
    with pytest.raises(AssertionError):
        run_status_script(monkeypatch, document, image_doc, args)


class FakeExecutor:
    def __init__(self, fail_call=None, fail_calls=()):
        self.fail_calls = set(fail_calls)
        if fail_call is not None:
            self.fail_calls.add(fail_call)
        self.calls = []

    def __call__(self, actions, timeout):
        self.calls.append((actions, timeout))
        failed = len(self.calls) in self.fail_calls
        return {
            action.rank: {
                "exit_code": 1 if failed and action.rank == 0 else 0,
                "stdout": "",
                "stderr": "synthetic" if failed and action.rank == 0 else "",
            }
            for action in actions
        }


class AdversarialExecutor(FakeExecutor):
    def __init__(self, *, raise_calls=(), missing_calls=(), fail_calls=()):
        super().__init__(fail_calls=fail_calls)
        self.raise_calls = set(raise_calls)
        self.missing_calls = set(missing_calls)

    def __call__(self, actions, timeout):
        call = len(self.calls) + 1
        if call in self.raise_calls:
            self.calls.append((actions, timeout))
            raise OSError("synthetic executor failure")
        if call in self.missing_calls:
            self.calls.append((actions, timeout))
            return {}
        return super().__call__(actions, timeout)


def test_status_execute_requires_confirmation_and_uses_fake_executor(tmp_path, capsys):
    site, profile, candidate, _ = generated(tmp_path)
    fake = FakeExecutor()
    with pytest.raises(SystemExit):
        launcher.main(
            args(site, profile, candidate, "status") + ["--execute"],
            executor=fake,
        )
    assert fake.calls == []
    argv = args(site, profile, candidate, "status")
    argv[0:0] = [
        "--execute",
        "--confirmation",
        launcher.CONFIRMATIONS["status"],
    ]
    assert launcher.main(argv, executor=fake) == 0
    assert len(fake.calls) == 3
    capsys.readouterr()


def test_precheck_failure_before_mutation_does_not_disrupt_baseline(tmp_path):
    site, profile, candidate, _ = generated(tmp_path)
    fake = FakeExecutor(fail_call=1)
    argv = args(site, profile, candidate, "cutover")
    argv[0:0] = [
        "--execute",
        "--confirmation",
        launcher.CONFIRMATIONS["cutover"],
    ]
    assert launcher.main(argv, executor=fake) == 1
    assert len(fake.calls) == 1


def test_failed_candidate_start_restores_canonical_lmcache_stack(tmp_path):
    site, profile, candidate, _ = generated(tmp_path)
    candidate_start_call = launcher.lifecycle("cutover").index("candidate_start") + 1
    fake = FakeExecutor(fail_call=candidate_start_call)
    argv = args(site, profile, candidate, "cutover")
    argv[0:0] = [
        "--execute",
        "--confirmation",
        launcher.CONFIRMATIONS["cutover"],
    ]
    assert launcher.main(argv, executor=fake) == 1
    assert len(fake.calls) == candidate_start_call + len(launcher.rollback_lifecycle())
    rollback_commands = "\n".join(
        action.shell_command for actions, _timeout in fake.calls[candidate_start_call:] for action in actions
    )
    assert "glm52-sparkring-lmcache-cs512-server-r0" in rollback_commands
    assert "LMCacheMPConnector" in rollback_commands
    assert "org.sparkring.component=engine" in rollback_commands
    assert "sparkcache-checkpoint-manifest-v2" not in rollback_commands
    rollback = launcher.rollback_lifecycle()
    assert rollback.index("checkpoint_quiescent") < rollback.index("baseline_start_servers")
    assert rollback.index("checkpoint_quiescent") < rollback.index("baseline_start_engines")


def test_full_hash_failure_after_source_stop_rolls_back_without_live_rehash(tmp_path):
    site, profile, candidate, _ = generated(tmp_path)
    full_hash_call = launcher.lifecycle("cutover").index("checkpoint_full_attest") + 1
    fake = FakeExecutor(fail_call=full_hash_call)
    argv = args(site, profile, candidate, "cutover")
    argv[0:0] = [
        "--execute",
        "--confirmation",
        launcher.CONFIRMATIONS["cutover"],
    ]
    assert launcher.main(argv, executor=fake) == 1
    assert len(fake.calls) == full_hash_call + len(launcher.rollback_lifecycle())
    rollback_commands = "\n".join(
        action.shell_command
        for actions, _timeout in fake.calls[full_hash_call:]
        for action in actions
    )
    assert "sparkcache-checkpoint-manifest-v2" not in rollback_commands


def test_automatic_rollback_continues_after_an_early_cleanup_failure(tmp_path):
    site_path, profile_path, _candidate_path, candidate = generated(tmp_path)
    site = load_site(site_path)
    baseline = exl3.load_profile(profile_path)
    phases = launcher.build_phases(site, baseline, launcher.validate_candidate(candidate, baseline))
    initial_failure = launcher.lifecycle("cutover").index("candidate_start") + 1
    # The first rollback phase is the non-negotiable verified-start prepare;
    # fail the following cleanup phase to prove later cleanup still runs.
    fake = FakeExecutor(fail_calls=(initial_failure, initial_failure + 2))
    code, results = launcher.execute_lifecycle("cutover", phases, fake)
    assert code == 1
    assert len(fake.calls) == initial_failure + len(launcher.rollback_lifecycle())
    assert len(results["automatic_rollback"]) == len(launcher.rollback_lifecycle())


def test_verified_start_prepare_failure_is_terminal_before_rollback_removal(tmp_path):
    site_path, profile_path, _candidate_path, candidate = generated(tmp_path)
    site = load_site(site_path)
    baseline = exl3.load_profile(profile_path)
    phases = launcher.build_phases(
        site, baseline, launcher.validate_candidate(candidate, baseline)
    )
    initial_failure = launcher.lifecycle("cutover").index("candidate_start") + 1
    fake = FakeExecutor(fail_calls=(initial_failure, initial_failure + 1))
    code, results = launcher.execute_lifecycle("cutover", phases, fake)
    assert code == 1
    assert len(fake.calls) == initial_failure + 1
    assert [item["phase"] for item in results["automatic_rollback"]] == [
        "verified_start_prepare"
    ]

    explicit = FakeExecutor(fail_call=1)
    code, results = launcher.execute_lifecycle("rollback", phases, explicit)
    assert code == 1
    assert len(explicit.calls) == 1
    assert set(results) == {"verified_start_prepare"}


def test_executor_exception_after_mutation_becomes_receipt_and_runs_rollback(tmp_path):
    site_path, profile_path, _candidate_path, candidate = generated(tmp_path)
    site = load_site(site_path)
    baseline = exl3.load_profile(profile_path)
    phases = launcher.build_phases(
        site, baseline, launcher.validate_candidate(candidate, baseline)
    )
    exception_call = launcher.lifecycle("cutover").index("baseline_remove_servers") + 1
    fake = AdversarialExecutor(raise_calls=(exception_call,))
    code, results = launcher.execute_lifecycle("cutover", phases, fake)
    assert code == 1
    assert len(fake.calls) == exception_call + len(launcher.rollback_lifecycle())
    receipt = results["baseline_remove_servers"][0]
    assert set(receipt) == {0, 1, 2, 3}
    assert {item["failure_kind"] for item in receipt.values()} == {
        "executor_exception"
    }
    assert len(results["automatic_rollback"]) == len(launcher.rollback_lifecycle())


def test_executor_exception_during_rollback_is_recorded_and_cleanup_continues(tmp_path):
    site_path, profile_path, _candidate_path, candidate = generated(tmp_path)
    site = load_site(site_path)
    baseline = exl3.load_profile(profile_path)
    phases = launcher.build_phases(
        site, baseline, launcher.validate_candidate(candidate, baseline)
    )
    initial_failure = launcher.lifecycle("cutover").index("candidate_start") + 1
    fake = AdversarialExecutor(
        fail_calls=(initial_failure,), raise_calls=(initial_failure + 2,)
    )
    code, results = launcher.execute_lifecycle("cutover", phases, fake)
    assert code == 1
    assert len(fake.calls) == initial_failure + len(launcher.rollback_lifecycle())
    failed_cleanup = results["automatic_rollback"][1]["result"]
    assert {item["failure_kind"] for item in failed_cleanup.values()} == {
        "executor_exception"
    }


class RankInt(int):
    """An integer subclass must not satisfy the exact rank-key contract."""


@pytest.mark.parametrize(
    "malformation",
    (
        "bool-rank",
        "float-rank",
        "string-rank",
        "int-subclass-rank",
        "missing-ranks",
        "extra-rank",
        "wrong-shape",
    ),
)
def test_executor_result_contract_fails_closed_with_exact_four_rank_receipts(
    tmp_path, malformation
):
    site_path, profile_path, _candidate_path, candidate = generated(tmp_path)
    site = load_site(site_path)
    baseline = exl3.load_profile(profile_path)
    phases = launcher.build_phases(
        site, baseline, launcher.validate_candidate(candidate, baseline)
    )

    def malformed(actions, timeout):
        valid = {
            action.rank: {"exit_code": 0, "stdout": "", "stderr": ""}
            for action in actions
        }
        if malformation == "bool-rank":
            valid[False] = valid.pop(0)
            return valid
        if malformation == "float-rank":
            valid[0.0] = valid.pop(0)
            return valid
        if malformation == "string-rank":
            valid["0"] = valid.pop(0)
            return valid
        if malformation == "int-subclass-rank":
            valid[RankInt(0)] = valid.pop(0)
            return valid
        if malformation == "missing-ranks":
            return {}
        if malformation == "extra-rank":
            valid[4] = {"exit_code": 0, "stdout": "", "stderr": ""}
            return valid
        return {action.rank: {"exit_code": 0} for action in actions}

    code, results = launcher.execute_lifecycle("cutover", phases, malformed)
    assert code == 1
    assert "automatic_rollback" not in results
    receipt = results["candidate_precheck"][0]
    assert set(receipt) == {0, 1, 2, 3}
    assert all(set(item) == launcher._RANK_RECEIPT_KEYS for item in receipt.values())
    expected_kind = (
        "executor_result_shape"
        if malformation == "wrong-shape"
        else "executor_rank_contract"
    )
    assert {item["failure_kind"] for item in receipt.values()} == {expected_kind}


@pytest.mark.parametrize(
    "malformation",
    (
        "bool-rank",
        "float-rank",
        "string-rank",
        "int-subclass-rank",
        "missing-rank",
        "extra-rank",
    ),
)
def test_action_rank_contract_rejects_nonexact_or_incomplete_ranks(malformation):
    ranks = [0, 1, 2, 3]
    if malformation == "bool-rank":
        ranks[0] = False
    elif malformation == "float-rank":
        ranks[0] = 0.0
    elif malformation == "string-rank":
        ranks[0] = "0"
    elif malformation == "int-subclass-rank":
        ranks[0] = RankInt(0)
    elif malformation == "missing-rank":
        ranks.pop()
    elif malformation == "extra-rank":
        ranks.append(4)
    else:
        raise AssertionError(malformation)
    actions = [exl3.RemoteAction(rank, f"rank-{rank}", ("true",)) for rank in ranks]
    executor_called = False

    def executor(_actions, timeout):
        nonlocal executor_called
        executor_called = True
        return {
            rank: {"exit_code": 0, "stdout": "", "stderr": ""}
            for rank in range(4)
        }

    receipt = launcher._execute_phase(executor, actions, timeout=1)
    assert executor_called is False
    assert set(receipt) == {0, 1, 2, 3}
    assert all(set(item) == launcher._RANK_RECEIPT_KEYS for item in receipt.values())
    assert {item["failure_kind"] for item in receipt.values()} == {
        "action_rank_contract"
    }


def test_failed_rollback_quiescence_never_starts_a_model(tmp_path):
    site_path, profile_path, _candidate_path, candidate = generated(tmp_path)
    site = load_site(site_path)
    baseline = exl3.load_profile(profile_path)
    phases = launcher.build_phases(site, baseline, launcher.validate_candidate(candidate, baseline))
    quiescence_call = launcher.rollback_lifecycle().index("checkpoint_quiescent") + 1
    fake = FakeExecutor(fail_call=quiescence_call)
    code, results = launcher.execute_lifecycle("rollback", phases, fake)
    assert code == 1
    called_commands = "\n".join(
        action.shell_command for actions, _timeout in fake.calls for action in actions
    )
    assert "&& exec docker run --detach" not in called_commands
    assert "verify_exl3_model.py" not in called_commands
    assert results["baseline_start_engines"] == [{"skipped": "quiescence_failed"}]
    assert results["baseline_start_servers"] == [{"skipped": "quiescence_failed"}]


def test_no_lifecycle_hashes_checkpoint_while_an_engine_is_live():
    cutover = launcher.lifecycle("cutover")
    assert cutover.index("baseline_remove_engines") < cutover.index("checkpoint_quiescent")
    assert cutover.index("checkpoint_quiescent") < cutover.index("checkpoint_full_attest")
    assert cutover.index("checkpoint_full_attest") < cutover.index("candidate_start")
    restart = launcher.lifecycle("restart-stack")
    assert restart.index("candidate_remove") < restart.index("checkpoint_quiescent")
    assert restart.index("checkpoint_quiescent") < restart.index("checkpoint_full_attest")
    assert restart.index("checkpoint_full_attest") < restart.index("candidate_start")
    for command in ("status", "restart-engines", "rollback"):
        assert "checkpoint_full_attest" not in launcher.lifecycle(command)


def test_candidate_drift_is_rejected_before_actions_are_built(tmp_path):
    _site, profile_path, _candidate_path, candidate = generated(tmp_path)
    candidate["profile"]["environment"]["VLLM_SPARK_DCP_SIZE"] = "2"
    with pytest.raises(launcher.LauncherError, match="canonical transformation"):
        launcher.validate_candidate(candidate, exl3.load_profile(profile_path))


def test_free_form_checkpoint_namespace_is_rejected(tmp_path):
    _site, profile_path, _candidate_path, candidate = generated(tmp_path)
    arguments = candidate["profile"]["extra_vllm_args"]
    index = arguments.index("--kv-transfer-config")
    kv = json.loads(arguments[index + 1])
    kv["kv_connector_extra_config"]["spark_cache_target_checkpoint_sha256"] = "f" * 64
    arguments[index + 1] = json.dumps(kv, separators=(",", ":"), sort_keys=True)
    with pytest.raises(launcher.LauncherError, match="embedded checkpoint receipt"):
        launcher.validate_candidate(candidate, exl3.load_profile(profile_path))


def test_api_barrier_precedes_final_all_rank_reinspection():
    for command in ("status", "cutover", "restart-engines", "restart-stack"):
        sequence = launcher.lifecycle(command)
        for index, phase in enumerate(sequence):
            if phase == "candidate_api_barrier":
                assert sequence[index + 1] == "candidate_final_status"
    assert launcher.rollback_lifecycle()[-3:] == [
        "baseline_final_status",
        "checkpoint_helper_absent",
        "candidate_absent_final",
    ]
