"""CPU contracts for NVFP4-Spark selection and guarded manual lifecycle."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "glm53_spark_tp2_launch", ROOT / "launch.py"
)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
IMAGE = "ghcr.io/fujitsupolycom/sparkring@sha256:" + "a" * 64
LOCAL_IMAGE = "sha256:" + "b" * 64


@pytest.fixture
def inputs(tmp_path):
    model = tmp_path / "model"
    cache = tmp_path / "cache"
    model.mkdir()
    cache.mkdir()
    (model / "config.json").write_text("{}")
    site = tmp_path / "rank.env"
    site.write_text(
        "VLLM_HOST_IP=rank.example\nNCCL_SOCKET_IFNAME=fabric0\nGLOO_SOCKET_IFNAME=fabric0\n"
    )
    return model, cache, site


def plan(inputs, rank=0):
    return launch.render(rank, "master.example", *inputs, IMAGE)


def receipt(value):
    return {
        "registry_digest": IMAGE,
        "profiles": {
            value["profile"]: {
                "profile_sha256": value["profile_sha256"],
                "transport_manifest_sha256": value["transport_manifest_sha256"],
                "source_compatibility": "passed",
            }
        },
    }


def stopped_container(value):
    return {
        "Config": {
            "Image": value["image"],
            "Entrypoint": [value["entrypoint"]],
            "Cmd": value["container_args"],
            "Healthcheck": {"Test": ["NONE"]},
            "Labels": value["labels"],
            "Env": [key + "=" + val for key, val in value["environment"].items()],
        },
        "HostConfig": {"RestartPolicy": {"Name": "no"}},
        "Mounts": [
            {"Destination": target, "Source": source, "RW": target != "/models/target"}
            for target, source in value["binds"].items()
        ],
    }


def r33_receipt(image=LOCAL_IMAGE):
    verifier = launch._r33_verifier()
    contract = verifier.load_contract()
    return {
        "schema": "sparkring-r33-image-receipt/v1",
        "checks_passed": True,
        "platform": "linux/arm64",
        "image_id": image,
        "image_reference": image,
        "artifact_lock_sha256": contract["image"]["artifact_lock_sha256"],
        "source_lock_sha256": "d" * 64,
        "sources": contract["image"]["required_sources"],
        "component_receipts": {
            name: "c" * 64 for name in contract["image"]["required_receipts"]
        },
        "nccl_version": "2.31.2",
        "source_locks_match": True,
        "source_lock_receipts_match": True,
        "installed_payload_bytes_match": True,
        "package_checks_passed": True,
    }


class Host:
    def __init__(self, value, *, floor=2147483648, busy=False):
        self.value = value
        self.floor = floor
        self.busy = busy
        self.container = stopped_container(value)
        self.commands = []

    def run(self, command, **kwargs):
        self.commands.append(command)
        stdout = ""
        if command[:2] == ["systemctl", "show"]:
            stdout = f"{{ argv[]=/usr/bin/python3 memory_guard.py --available-floor-bytes {self.floor} ; }}"
        elif command == ["docker", "ps", "--quiet"]:
            stdout = "running\n" if self.busy else ""
        elif command == ["docker", "inspect", "running"]:
            stdout = json.dumps(
                [{"HostConfig": {"DeviceRequests": [{"Capabilities": [["gpu"]]}]}}]
            )
        elif command == ["docker", "inspect", self.value["name"]]:
            stdout = json.dumps([self.container])
        return subprocess.CompletedProcess(command, 0, stdout, "")


def test_profile_selects_spark_kv875_settings_and_no_cache():
    profile = launch.load_profile()
    assert (
        profile["model"]["repository"]
        == "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"
    )
    assert profile["model"]["revision"] == "df116c4fb16b1d37ae43d2cfd624de26ffbc832e"
    args = profile["vllm_args"]

    def value(flag):
        return args[args.index(flag) + 1]

    for flag, expected in {
        "--tensor-parallel-size": "2",
        "--decode-context-parallel-size": "1",
        "--load-format": "b12x",
        "--kv-cache-memory-bytes": "9395240960",
        "--served-model-name": "GLM-5.3-Flash-NVFP4-Spark",
        "--max-model-len": "262144",
        "--max-num-seqs": "8",
        "--max-num-batched-tokens": "8192",
        "--prefill-schedule-interval": "8",
        "--kda-prefill-backend": "b12x",
        "--recurrent-checkpoint-policy": "aligned",
    }.items():
        assert value(flag) == expected
    assert json.loads(value("--speculative-config")) == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "moe_backend": "humming",
        "attention_backend": "B12X",
    }
    assert json.loads(value("--compilation-config")) == {
        "mode": 0,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16, 20, 24, 28, 32],
        "max_cudagraph_capture_size": 32,
    }
    assert json.loads(value("--limit-mm-per-prompt")) == {"image": 4, "video": 0}
    assert json.loads(value("--model-loader-extra-config")) == {"allocation": "managed"}
    assert "--kv-transfer-config" not in args
    assert not profile["sparkcache"]["enabled"]


@pytest.mark.parametrize("rank", [0, 1])
def test_rank_plan_maps_both_pci_functions_of_one_qsfp_cage(inputs, rank):
    value = plan(inputs, rank)
    env = value["environment"]
    assert env["B12X_ROCE_HCA"] == "rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1"
    assert env["B12X_ROCE_PEER_HCA_MAP"] == f"{1 - rank}=0/2"
    assert env["NCCL_IB_HCA"] == "=rocep1s0f0,roceP2p1s0f0"
    assert env["B12X_ROCE_PAIR_PATHS"] == "2"
    assert env["NCCL_MIN_NCHANNELS"] == env["NCCL_MAX_NCHANNELS"] == "8"
    assert env["VLLM_NCCL_SO_PATH"] == "/opt/sparkring/nccl-pci/libnccl.so.2.30.7"
    for key in (
        "NCCL_IB_EXTENDED_IPV4_GIDS",
        "NCCL_IB_PRESERVE_PCI_DOMAIN",
        "NCCL_IB_ROUTE_DIAGNOSTICS",
    ):
        assert env[key] == "1"
    assert env["SOURCE_IMAGE_PROFILE"] == value["profile"]
    assert env["VLLM_GLM53_KDA_GATE_SIDE_STREAM"] == "0"
    assert ("--headless" in value["container_args"]) is (rank == 1)
    assert "${NODE_RANK}" not in value["container_args"]
    assert value["command"][:2] == ["docker", "create"]
    assert value["command"][value["command"].index("--restart") + 1] == "no"
    assert "--no-healthcheck" in value["command"]
    assert value["labels"]["org.sparkring.memory-guard"] == "true"
    args = value["container_args"]
    assert args[:4] == [
        "-S",
        "-B",
        "/opt/sparkcache-jj-runtime/verify_sources.py",
        "--serve",
    ]
    assert json.loads(args[args.index("--model-loader-extra-config") + 1]) == {
        "allocation": "managed"
    }


def test_jit_cache_paths_match_the_mount_and_separate_ranks(inputs):
    ranks = [plan(inputs, rank) for rank in (0, 1)]
    for value in ranks:
        assert "/cache/jit" in value["binds"]
        for key in (
            "XDG_CACHE_HOME",
            "VLLM_CACHE_ROOT",
            "B12X_ROCE_CACHE_DIR",
            "B12X_COMPILE_CACHE_DIR",
        ):
            assert value["environment"][key].startswith("/cache/jit/")
            assert value["profile_sha256"] in value["environment"][key]
    assert (
        ranks[0]["environment"]["VLLM_CACHE_ROOT"]
        != ranks[1]["environment"]["VLLM_CACHE_ROOT"]
    )


def test_tp2_clears_inherited_mesh_sitecustomize(inputs, tmp_path):
    import sys

    (tmp_path / "sitecustomize.py").write_text(
        "raise SystemExit('TP4_HOOK_IMPORTED')\n"
    )
    inherited = {**os.environ, "PYTHONPATH": str(tmp_path)}
    command = [sys.executable, "-c", "print('CONSUMER_REACHED')"]
    trapped = subprocess.run(command, env=inherited, capture_output=True, text=True)
    assert trapped.returncode != 0 and "TP4_HOOK_IMPORTED" in trapped.stderr
    environment = plan(inputs)["environment"]
    assert environment["PYTHONPATH"] == ""
    selected = subprocess.run(
        command,
        env={**inherited, "PYTHONPATH": environment["PYTHONPATH"]},
        capture_output=True,
        text=True,
    )
    assert selected.returncode == 0, selected.stderr
    assert "CONSUMER_REACHED" in selected.stdout


@pytest.mark.parametrize(
    "assignment",
    [
        "B12X_ROCE_PEER_HCA_MAP=1=2/1",
        "VLLM_PLUGINS=sparkcache",
        "SPARK_CACHE_ENABLED=1",
        "VLLM_HOST_IP=duplicate",
    ],
)
def test_site_file_cannot_change_profile_or_enable_cache(inputs, assignment):
    with inputs[2].open("a") as stream:
        stream.write(assignment + "\n")
    with pytest.raises(ValueError):
        plan(inputs)


def test_no_published_image_is_selected_implicitly(inputs):
    with pytest.raises(ValueError, match="immutable"):
        launch.render(0, "master.example", *inputs, "latest")


def test_create_does_not_start_and_requires_active_floor(inputs):
    value = plan(inputs)
    host = Host(value)
    launch.execute(value, "create", receipt(value), run=host.run)
    assert host.commands[-1] == value["command"]
    assert not any(command[:2] == ["docker", "start"] for command in host.commands)
    assert host.commands[0][:3] == ["systemctl", "is-active", "--quiet"]


@pytest.mark.parametrize("floor,busy", [(1073741824, False), (2147483648, True)])
def test_guard_or_running_gpu_refuses_create(inputs, floor, busy):
    value = plan(inputs)
    host = Host(value, floor=floor, busy=busy)
    with pytest.raises(RuntimeError):
        launch.execute(value, "create", receipt(value), run=host.run)
    assert value["command"] not in host.commands


def test_start_requires_matching_stopped_container(inputs):
    value = plan(inputs)
    host = Host(value)
    launch.execute(value, "start", receipt(value), run=host.run)
    assert host.commands[-1] == ["docker", "start", value["name"]]


@pytest.mark.parametrize(
    "damage", ["restart", "image", "environment", "mount", "healthcheck"]
)
def test_start_does_not_adopt_changed_container(inputs, damage):
    value = plan(inputs)
    host = Host(value)
    if damage == "restart":
        host.container["HostConfig"]["RestartPolicy"]["Name"] = "always"
    elif damage == "image":
        host.container["Config"]["Image"] = "latest"
    elif damage == "environment":
        host.container["Config"]["Env"] = [
            item
            for item in host.container["Config"]["Env"]
            if not item.startswith("B12X_ROCE_PEER_HCA_MAP=")
        ]
    elif damage == "healthcheck":
        host.container["Config"]["Healthcheck"] = {
            "Test": ["CMD-SHELL", "test -f /tmp/sparkring-engine-ready"]
        }
    else:
        host.container["Mounts"][0]["RW"] = True
    with pytest.raises(RuntimeError, match="differs"):
        launch.execute(value, "start", receipt(value), run=host.run)
    assert ["docker", "start", value["name"]] not in host.commands


@pytest.mark.parametrize(
    "field", ["registry_digest", "source_compatibility", "transport_manifest_sha256"]
)
def test_runtime_receipt_must_match_before_host_commands(inputs, field):
    value = plan(inputs)
    runtime = copy.deepcopy(receipt(value))
    if field == "registry_digest":
        runtime[field] = "wrong"
    else:
        runtime["profiles"][value["profile"]][field] = "wrong"
    host = Host(value)
    with pytest.raises(ValueError):
        launch.execute(value, "create", runtime, run=host.run)
    assert not host.commands


def test_retired_entrypoint_uses_canonical_spark_plan_in_fresh_process(inputs):
    retired = ROOT.parent / "glm53-flash-nvfp4-tp2/launch.py"
    model, cache, site = inputs
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(retired),
            "plan",
            "--rank",
            "1",
            "--master",
            "master.example",
            "--model-dir",
            str(model),
            "--cache-dir",
            str(cache),
            "--env-file",
            str(site),
            "--image",
            IMAGE,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == plan(inputs, 1)


def test_managed_option_and_guard_distinguish_reference_trial():
    dependencies = json.loads((ROOT / "dependencies.json").read_text())
    loader = next(
        item
        for item in dependencies["shared_image_requirements"]
        if item["component"] == "B12X loader"
    )
    assert loader["model_loader_extra_config"] == {"allocation": "managed"}
    assert loader["shared_loader_default_allocation"] == "pinned_wc"
    assert "research-only" in launch.load_profile()["qualification"]["shared_image"]
    assert dependencies["reference_trial"]["memory_guards_active"] is False
    assert dependencies["reference_trial"]["kv_pool_tokens_estimate"] == 1050118
    assert launch.load_profile()["lifecycle"]["memory_guard_floor_bytes"] == 2147483648
    assert (
        launch.load_profile()["qualification"]["public_guarded_gpu_qualified"] is False
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_r33_receipt_adapts_final_tp2_docker_command(inputs, rank):
    runtime = r33_receipt()
    value = launch.render(rank, "master.example", *inputs, LOCAL_IMAGE, runtime)
    assert value["profile"] == "tp2-dcp1"
    assert value["name"] == f"sparkring-r33-tp2-dcp1-r{rank}"
    assert value["command"][value["command"].index("--name") + 1] == value["name"]
    assert value["entrypoint"] == "/opt/sparkring/bin/sparkring-r33"
    assert value["container_args"][:2] == ["serve", "/models/target"]
    assert "/opt/sparkcache-jj-runtime/verify_sources.py" not in value["command"]
    assert (
        value["container_args"][value["container_args"].index("--max-model-len") + 1]
        == "1048576"
    )
    assert (
        value["container_args"][value["container_args"].index("--load-format") + 1]
        == "instanttensor"
    )
    assert "--model-loader-extra-config" not in value["container_args"]
    environment = value["environment"]
    assert environment["SOURCE_IMAGE_PROFILE"] == "tp2-dcp1"
    assert environment["SPARKRING_PROFILE_MODE"] == "custom"
    assert (
        environment["VLLM_NCCL_SO_PATH"] == "/opt/local-inference/nccl/lib/libnccl.so.2"
    )
    assert environment["LD_PRELOAD"] == environment["VLLM_NCCL_SO_PATH"]
    assert "LD_PRELOAD=/opt/local-inference/nccl/lib/libnccl.so.2" in value["command"]
    assert (
        "LD_PRELOAD=/opt/sparkring/nccl-pci/libnccl.so.2.30.7" not in value["command"]
    )
    arguments = value["container_args"]
    assert arguments[arguments.index("--gdn-decode-kernel") + 1] == "b12x"
    assert (
        json.loads(arguments[arguments.index("--speculative-config") + 1])[
            "moe_backend"
        ]
        == "humming"
    )
    assert environment["LOAD_FORMAT"] == "instanttensor"
    assert environment["VLLM_PLUGINS"] == ""
    assert "SOURCE_IMAGE_PROFILE=tp2-dcp1" in value["command"]
    entrypoint_path = ROOT.parents[1] / "sparkring/jovian-r33/image/entrypoint.py"
    spec = importlib.util.spec_from_file_location(
        "r33_tp2_candidate_entrypoint", entrypoint_path
    )
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    with (
        mock.patch.dict(os.environ, environment, clear=True),
        mock.patch.object(entrypoint.subprocess, "run"),
    ):
        assert (
            entrypoint.validate_external_profile(launch.R33_PROFILE_ROOT)[
                "tensor_parallel_size"
            ]
            == 2
        )
    launch.validate_runtime_receipt(runtime, value)
    host = Host(value)
    launch.execute(value, "create", runtime, run=host.run)
    assert host.commands[-1] == value["command"]
    launch.execute(value, "start", runtime, run=host.run)
    assert host.commands[-1] == ["docker", "start", value["name"]]


def test_legacy_tp2_plan_keeps_its_pinned_preload(inputs):
    plan = launch.render(0, "master.example", *inputs, LOCAL_IMAGE)
    assert (
        plan["environment"]["LD_PRELOAD"] == "/opt/sparkring/nccl-pci/libnccl.so.2.30.7"
    )
    assert "LD_PRELOAD=/opt/sparkring/nccl-pci/libnccl.so.2.30.7" in plan["command"]
    assert plan["runtime_kind"] == "legacy"
    assert plan["name"] == "sparkring-glm53-tp2-r0"
    assert plan["command"][plan["command"].index("--name") + 1] == plan["name"]


def cache_capable_receipt():
    receipt = r33_receipt()
    contract = launch._r33_verifier().load_contract()
    selected = contract["profiles"]["tp2-dcp1-sparkcache"]
    document = {
        "schema": "sparkring-r33-runtime-capabilities/v1",
        "profile": "tp2-dcp1-sparkcache",
        "sources": {
            k: receipt["sources"][k]
            for k in (
                "vllm_integrated_tree",
                "vllm_tp2_continuation_port_commit",
                "b12x_tree",
                "sparkcache_tree",
            )
        },
        "evidence_kind": "source-component-tests",
        "live_qualification": "pending",
        "checks": {key: "implemented" for key in selected["required_capabilities"]},
        "evidence_sha256": {key: "e" * 64 for key in selected["required_capabilities"]},
    }
    data = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    receipt["runtime_capabilities"] = {"document": document, "sha256": digest}
    native = contract["sparkcache_native"]
    receipt["verification"] = {
        "checked_files": {
            "/opt/sparkring/profile-contract/tp2-sparkcache-capabilities.json": digest,
            native["placement_path"]: native["placement_sha256"],
            native["snapshot_path"]: native["snapshot_sha256"],
        }
    }
    return receipt, data


@pytest.mark.parametrize("rank", [0, 1])
def test_r33_cache_preserves_reference_settings_with_explicit_1m_target(
    inputs, rank, tmp_path
):
    runtime, capability = cache_capable_receipt()
    value = launch.render(
        rank, "master.example", *inputs, LOCAL_IMAGE, runtime, r33_sparkcache=True
    )
    assert value["name"] == f"sparkring-r33-tp2-dcp1-sparkcache-r{rank}"
    args = value["container_args"]

    def option(name):
        return args[args.index(name) + 1]

    assert option("--load-format") == "b12x"
    assert json.loads(option("--speculative-config")) == {
        "method": "mtp", "num_speculative_tokens": 3, "moe_backend": "humming", "attention_backend": "B12X",
        "draft_load_config": {"load_format": "b12x", "model_loader_extra_config": {}},
    }
    assert "--model-loader-extra-config" not in args
    assert option("--kv-cache-memory-bytes") == "8053063680"
    contract = launch._r33_verifier().load_contract()["profiles"]["tp2-dcp1-sparkcache"]
    assert contract["reference_kv_cache_memory_bytes"] == 7247757312
    assert json.loads(option("--speculative-config"))["draft_load_config"] == contract["draft_load_config"]
    assert option("--max-model-len") == "1048576"
    assert option("--max-num-seqs") == "8"
    assert option("--prefill-schedule-interval") == "8"
    assert json.loads(option("--limit-mm-per-prompt")) == {"image": 3, "video": 1}
    assert value["environment"]["VLLM_B12X_KDA_PREFILL_COALESCING"] == "1"
    assert value["environment"]["VLLM_PLUGINS"] == "b12x_loader"
    assert value["memory_guard_floor_bytes"] == 2147483648
    connector = json.loads(option("--kv-transfer-config"))
    assert connector["kv_load_failure_policy"] == "recompute"
    extra = connector["kv_connector_extra_config"]
    assert extra["spark_cache_cuda_restore_arena_budget_bytes"] == 268435456
    assert extra["spark_cache_async_page_capture_slot_bytes"] == 536870912
    assert extra["spark_cache_async_page_capture_slot_count"] == 2
    assert extra["spark_cache_root"].startswith("/cache/jit/sparkcache-context/")
    assert value["binds"]["/cache/jit"] == str(inputs[1].resolve())
    assert extra["spark_cache_root"] == (
        f"/cache/jit/sparkcache-context/sparkring-r33-{runtime['image_id'][7:19]}-tp2-cache"
    )
    assert value["qualification"]["gpu_qualified"] is False
    assert value["activation_blockers"] == []
    launch.validate_runtime_receipt(runtime, value)
    host = Host(value)
    launch.execute(value, "create", runtime, run=host.run)
    assert host.commands[-1] == value["command"]
    entrypoint_path = ROOT.parents[1] / "sparkring/jovian-r33/image/entrypoint.py"
    spec = importlib.util.spec_from_file_location(
        "r33_tp2_cache_entrypoint", entrypoint_path
    )
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    profile_root = tmp_path / "packaged-profiles"
    shutil.copytree(launch.R33_PROFILE_ROOT, profile_root)
    (profile_root / "tp2-sparkcache-capabilities.json").write_bytes(capability)
    with (
        mock.patch.dict(os.environ, value["environment"], clear=True),
        mock.patch.object(entrypoint.subprocess, "run"),
    ):
        assert entrypoint.validate_external_profile(profile_root)["sparkcache"] is True
        (profile_root / "tp2-sparkcache-capabilities.json").unlink()
        with pytest.raises(RuntimeError, match="capability evidence"):
            entrypoint.validate_external_profile(profile_root)


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "coalescing",
        "source",
        "file",
        "native",
        "live_claim",
        "evidence_kind",
    ],
)
def test_r33_cache_rejects_unproven_capabilities_before_host_action(inputs, failure):
    runtime, _ = cache_capable_receipt()
    if failure == "missing":
        runtime.pop("runtime_capabilities")
    elif failure == "coalescing":
        runtime["runtime_capabilities"]["document"]["checks"][
            "tp2_continuation_prefill_coalescing"
        ] = False
    elif failure == "source":
        runtime["runtime_capabilities"]["document"]["sources"][
            "vllm_integrated_tree"
        ] = "0" * 40
    elif failure == "file":
        runtime["runtime_capabilities"]["sha256"] = "0" * 64
    elif failure == "live_claim":
        runtime["runtime_capabilities"]["document"]["live_qualification"] = "qualified"
    elif failure == "evidence_kind":
        runtime["runtime_capabilities"]["document"]["evidence_kind"] = (
            "environment-flags"
        )
    else:
        runtime["verification"]["checked_files"].pop(
            launch._r33_verifier().load_contract()["sparkcache_native"]["snapshot_path"]
        )
    value = launch.render(
        0, "master.example", *inputs, LOCAL_IMAGE, runtime, r33_sparkcache=True
    )
    assert value["activation_blockers"]
    host = Host(value)
    with pytest.raises(ValueError):
        launch.execute(value, "create", runtime, run=host.run)
    assert host.commands == []


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("pin", [8053063680, 9395240960])
def test_r33_cache_explicit_pin_reaches_docker_with_guards(inputs, rank, pin):
    runtime, _ = cache_capable_receipt()
    value = launch.render(rank, "master.example", *inputs, LOCAL_IMAGE, runtime,
                          r33_sparkcache=True, r33_cache_kv_memory_bytes=pin)
    command = value["command"]
    assert command[command.index("--kv-cache-memory-bytes") + 1] == str(pin)
    assert command[command.index("--max-model-len") + 1] == "1048576"
    assert value["kv_cache_memory_bytes"] == pin
    assert value["memory_guard_floor_bytes"] == 2147483648
    assert value["qualification"]["gpu_qualified"] is False
    launch.validate_runtime_receipt(runtime, value)
    host = Host(value)
    launch.execute(value, "create", runtime, run=host.run)
    assert host.commands[-1] == command
    assert any(item[:2] == ["systemctl", "is-active"] for item in host.commands)


@pytest.mark.parametrize("cache,pin", [(False, 9395240960), (True, 0), (True, 8589934592), (True, True)])
def test_r33_cache_kv_override_rejects_wrong_scope_or_unreviewed_pin(inputs, cache, pin):
    with pytest.raises(ValueError, match="KV override"):
        launch.render(0, "master.example", *inputs, LOCAL_IMAGE, r33_receipt(),
                      r33_sparkcache=cache, r33_cache_kv_memory_bytes=pin)


def test_changed_r33_receipt_rejected_before_host_action(inputs):
    runtime = r33_receipt()
    value = launch.render(0, "master.example", *inputs, LOCAL_IMAGE, runtime)
    runtime["sources"] = dict(runtime["sources"], vllm_head="0" * 40)
    host = Host(value)
    with pytest.raises(ValueError):
        launch.execute(value, "create", runtime, run=host.run)
    assert host.commands == []


@pytest.fixture
def local_source_receipt(inputs, tmp_path, monkeypatch):
    # Use repository source-image contracts unless an explicit fixture path
    # supplies the same contract files for isolated compatibility tests.
    origin = Path(
        os.environ.get(
            "SPARKRING_TEST_COMMON_SOURCE_IMAGE", str(launch.SOURCE_IMAGE_ROOT)
        )
    )
    if not (origin / "receipt_contract.py").is_file():
        pytest.skip(
            "Common source-image recipe is required for local receipt integration tests"
        )
    value = launch.render(0, "master.example", *inputs, LOCAL_IMAGE)
    lock = json.loads((origin / "glm53-tp4-lock.json").read_bytes())
    assert (
        lock["profiles"][value["profile"]]["profile_sha256"] == value["profile_sha256"]
    )
    target = tmp_path / "common-source-image"
    target.mkdir()
    for filename in ("archive_utils.py", "native_files.py", "receipt_contract.py"):
        shutil.copyfile(origin / filename, target / filename)
    lock_path = target / "glm53-tp4-lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    manifest_path = ROOT / launch.load_profile()["transport"]["manifest"]
    files = json.loads(manifest_path.read_bytes())["files"]
    inside = {
        "checks_passed": True,
        "cuda_initialized": False,
        "model_loaded": False,
        "source_lock_sha256": lock_hash,
        "inherited_runtime": lock["runtime"]["expected_distributions"],
        "packages": {
            name: {
                "revision": row["revision"],
                "file_map_sha256": row["installed_file_map_sha256"],
                "files": row["installed_file_count"],
            }
            for name, row in lock["sources"].items()
        },
        "transport_profiles": {
            "tp2-rocenante-adaptive": {
                "manifest_sha256": value["transport_manifest_sha256"],
                "files_sha256": hashlib.sha256(
                    json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "files": len(files),
                "package": "b12x.comm.roce",
            }
        },
    }
    for field in (
        "bundle_manifest_sha256",
        "transport_sha256",
        "marker_source_sha256",
        "marker_binary_sha256",
        "nccl_sha256",
        "retained_vllm_native_sha256",
        "readiness_warmup",
    ):
        inside[field] = lock["runtime"][field]
    runtime = {
        "schema": "sparkring-source-image-receipt/v1",
        "image_id": LOCAL_IMAGE,
        "image_reference": LOCAL_IMAGE,
        "platform": "linux/arm64",
        "checks_passed": True,
        "profile": value["profile"],
        "source_lock_sha256": lock_hash,
        "inside_image": inside,
    }
    monkeypatch.setattr(launch, "SOURCE_IMAGE_ROOT", target)
    return value, runtime, lock_path


def test_local_image_requires_complete_source_and_transport_witness(
    local_source_receipt,
):
    value, runtime, _ = local_source_receipt
    assert value["image_identity_kind"] == "local_config_id"
    launch.validate_runtime_receipt(runtime, value)
    host = Host(value)
    launch.execute(value, "create", runtime, run=host.run)
    assert host.commands[-1] == value["command"]
    launch.execute(value, "start", runtime, run=host.run)
    assert host.commands[-1] == ["docker", "start", value["name"]]
    assert "registry_digest" not in runtime


@pytest.mark.parametrize(
    "damage",
    [
        "source_lock",
        "package_map",
        "transport_map",
        "transport_count",
        "image_id",
        "profile_hash",
        "world_size",
    ],
)
def test_local_witness_drift_stops_before_host_commands(local_source_receipt, damage):
    value, runtime, lock_path = local_source_receipt
    if damage == "source_lock":
        runtime["source_lock_sha256"] = "0" * 64
        runtime["inside_image"]["source_lock_sha256"] = "0" * 64
    elif damage == "package_map":
        runtime["inside_image"]["packages"]["b12x"]["file_map_sha256"] = "0" * 64
    elif damage in ("transport_map", "transport_count"):
        key = "files_sha256" if damage == "transport_map" else "files"
        runtime["inside_image"]["transport_profiles"]["tp2-rocenante-adaptive"][key] = 0
    elif damage == "image_id":
        runtime["image_id"] = runtime["image_reference"] = "sha256:" + "c" * 64
    else:
        lock = json.loads(lock_path.read_bytes())
        key, changed = (
            ("profile_sha256", "0" * 64) if damage == "profile_hash" else ("tp_size", 4)
        )
        lock["profiles"][value["profile"]][key] = changed
        lock_path.write_text(json.dumps(lock) + "\n")
        digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        runtime["source_lock_sha256"] = runtime["inside_image"][
            "source_lock_sha256"
        ] = digest
    host = Host(value)
    with pytest.raises(ValueError):
        launch.execute(value, "create", runtime, run=host.run)
    assert host.commands == []


def test_local_receipt_cannot_claim_a_registry_digest(inputs, local_source_receipt):
    _, runtime, _ = local_source_receipt
    runtime["registry_digest"] = IMAGE
    with pytest.raises(ValueError, match="not a registry"):
        launch.validate_runtime_receipt(runtime, plan(inputs))


@pytest.mark.parametrize("damage", [False, True])
def test_pinned_receipt_validates_in_fresh_process(
    local_source_receipt, tmp_path, damage
):
    value, runtime, lock_path = local_source_receipt
    contract = launch._source_receipt_contract(lock_path.parent)
    runtime["native_mode"] = runtime["inside_image"]["native_mode"] = "pinned"
    runtime["inside_image"]["native_files"] = contract.expected_record(
        json.loads(lock_path.read_bytes())
    )
    if damage:
        runtime["inside_image"]["native_files"]["archive_sha256"] = "0" * 64
    inputs_path = tmp_path / "receipt-inputs.json"
    inputs_path.write_text(json.dumps({"plan": value, "receipt": runtime}))
    script = """
import importlib.util,json,sys
from pathlib import Path
assert 'archive_utils' not in sys.modules and 'native_files' not in sys.modules
spec=importlib.util.spec_from_file_location('fresh_tp2_launcher',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.SOURCE_IMAGE_ROOT=Path(sys.argv[2])
data=json.loads(Path(sys.argv[3]).read_bytes())
module.validate_runtime_receipt(data['receipt'],data['plan'])
assert 'archive_utils' not in sys.modules and 'native_files' not in sys.modules
print('receipt valid')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(ROOT / "launch.py"),
            str(lock_path.parent),
            str(inputs_path),
        ],
        capture_output=True,
        text=True,
    )
    if damage:
        assert result.returncode != 0
        assert "native artifact witness differs" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "receipt valid"
