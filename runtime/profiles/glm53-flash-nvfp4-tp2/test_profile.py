"""CPU contracts for original-checkpoint selection and guarded manual lifecycle."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("glm53_nvfp4_tp2_launch", ROOT / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
IMAGE = "ghcr.io/fujitsupolycom/sparkring@sha256:" + "a" * 64


@pytest.fixture
def inputs(tmp_path):
    model = tmp_path / "model"
    cache = tmp_path / "cache"
    model.mkdir()
    cache.mkdir()
    (model / "config.json").write_text("{}")
    site = tmp_path / "rank.env"
    site.write_text("VLLM_HOST_IP=rank.example\nNCCL_SOCKET_IFNAME=fabric0\nGLOO_SOCKET_IFNAME=fabric0\n")
    return model, cache, site


def plan(inputs, rank=0):
    return launch.render(rank, "master.example", *inputs, IMAGE)


def receipt(value):
    return {"registry_digest": IMAGE, "profiles": {value["profile"]: {
        "profile_sha256": value["profile_sha256"],
        "transport_manifest_sha256": value["transport_manifest_sha256"],
        "source_compatibility": "passed",
    }}}


def stopped_container(value):
    return {
        "Config": {"Image": IMAGE, "Entrypoint": ["python3"], "Cmd": value["container_args"],
                   "Labels": value["labels"], "Env": [key + "=" + val for key, val in value["environment"].items()]},
        "HostConfig": {"RestartPolicy": {"Name": "no"}},
        "Mounts": [{"Destination": target, "Source": source, "RW": target != "/models/target"}
                   for target, source in value["binds"].items()],
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
            stdout = json.dumps([{"HostConfig": {"DeviceRequests": [{"Capabilities": [["gpu"]]}]}}])
        elif command == ["docker", "inspect", self.value["name"]]:
            stdout = json.dumps([self.container])
        return subprocess.CompletedProcess(command, 0, stdout, "")


def test_profile_preserves_tested_model_settings_and_no_cache():
    profile = launch.load_profile()
    assert profile["model"]["repository"] == "local-inference-lab/GLM-5.3-Flash-NVFP4"
    assert profile["model"]["revision"] == "520de24"
    args = profile["vllm_args"]
    def value(flag):
        return args[args.index(flag) + 1]
    for flag, expected in {
        "--tensor-parallel-size": "2", "--decode-context-parallel-size": "1",
        "--load-format": "b12x", "--kv-cache-memory-bytes": "7247757312",
        "--max-model-len": "262144", "--max-num-seqs": "8",
        "--max-num-batched-tokens": "8192", "--prefill-schedule-interval": "8",
        "--kda-prefill-backend": "b12x", "--recurrent-checkpoint-policy": "aligned",
    }.items():
        assert value(flag) == expected
    assert json.loads(value("--speculative-config")) == {
        "method": "mtp", "num_speculative_tokens": 3, "moe_backend": "humming", "attention_backend": "B12X",
    }
    assert json.loads(value("--compilation-config"))["max_cudagraph_capture_size"] == 32
    assert json.loads(value("--limit-mm-per-prompt")) == {"image": 4, "video": 0}
    assert "--kv-transfer-config" not in args
    assert not profile["sparkcache"]["enabled"]


@pytest.mark.parametrize("rank", [0, 1])
def test_rank_plan_maps_both_pci_functions_of_one_cage(inputs, rank):
    value = plan(inputs, rank)
    env = value["environment"]
    assert env["B12X_ROCE_HCA"] == "rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1"
    assert env["B12X_ROCE_PEER_HCA_MAP"] == f"{1-rank}=0/2"
    assert env["NCCL_IB_HCA"] == "=rocep1s0f0,roceP2p1s0f0"
    assert env["B12X_ROCE_PAIR_PATHS"] == "2"
    assert env["NCCL_MIN_NCHANNELS"] == env["NCCL_MAX_NCHANNELS"] == "8"
    assert ("--headless" in value["container_args"]) is (rank == 1)
    assert "${NODE_RANK}" not in value["container_args"]
    assert value["command"][:2] == ["docker", "create"]
    assert value["command"][value["command"].index("--restart") + 1] == "no"
    assert value["labels"]["org.sparkring.memory-guard"] == "true"


def test_jit_cache_paths_match_the_mount_and_separate_ranks(inputs):
    ranks = [plan(inputs, rank) for rank in (0, 1)]
    for value in ranks:
        assert "/cache/jit" in value["binds"]
        for key in ("XDG_CACHE_HOME", "VLLM_CACHE_ROOT", "B12X_ROCE_CACHE_DIR", "B12X_COMPILE_CACHE_DIR"):
            assert value["environment"][key].startswith("/cache/jit/")
            assert value["profile_sha256"] in value["environment"][key]
    assert ranks[0]["environment"]["VLLM_CACHE_ROOT"] != ranks[1]["environment"]["VLLM_CACHE_ROOT"]


@pytest.mark.parametrize("assignment", ["B12X_ROCE_PEER_HCA_MAP=1=2/1", "VLLM_PLUGINS=sparkcache", "SPARK_CACHE_ENABLED=1", "VLLM_HOST_IP=duplicate"])
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


@pytest.mark.parametrize("damage", ["restart", "image", "environment", "mount"])
def test_start_does_not_adopt_changed_container(inputs, damage):
    value = plan(inputs)
    host = Host(value)
    if damage == "restart":
        host.container["HostConfig"]["RestartPolicy"]["Name"] = "always"
    elif damage == "image":
        host.container["Config"]["Image"] = "latest"
    elif damage == "environment":
        host.container["Config"]["Env"] = [item for item in host.container["Config"]["Env"]
                                            if not item.startswith("B12X_ROCE_PEER_HCA_MAP=")]
    else:
        host.container["Mounts"][0]["RW"] = True
    with pytest.raises(RuntimeError, match="differs"):
        launch.execute(value, "start", receipt(value), run=host.run)
    assert ["docker", "start", value["name"]] not in host.commands


@pytest.mark.parametrize("field", ["registry_digest", "source_compatibility", "transport_manifest_sha256"])
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


def test_existing_spark_checkpoint_profile_keeps_its_identity():
    spark = json.loads((ROOT.parent / "glm53-flash-spark-tp2/profile.json").read_text())
    assert spark["model"]["repository"].endswith("-NVFP4-Spark")
    assert spark["model"]["revision"] == "df116c4fb16b1d37ae43d2cfd624de26ffbc832e"
