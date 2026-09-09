"""CPU contracts for original-checkpoint selection and guarded manual lifecycle."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("glm53_nvfp4_tp2_launch", ROOT / "launch.py")
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
        "Config": {"Image": value["image"], "Entrypoint": ["python3"], "Cmd": value["container_args"],
                   "Healthcheck": {"Test": ["NONE"]},
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
    assert json.loads(value("--model-loader-extra-config")) == {"allocation": "managed"}
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
    assert env["VLLM_NCCL_SO_PATH"] == "/opt/sparkring/nccl-pci/libnccl.so.2.30.7"
    for key in ("NCCL_IB_EXTENDED_IPV4_GIDS", "NCCL_IB_PRESERVE_PCI_DOMAIN", "NCCL_IB_ROUTE_DIAGNOSTICS"):
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
    assert args[:4] == ["-S", "-B", "/opt/sparkcache-jj-runtime/verify_sources.py", "--serve"]
    assert json.loads(args[args.index("--model-loader-extra-config") + 1]) == {"allocation": "managed"}


def test_jit_cache_paths_match_the_mount_and_separate_ranks(inputs):
    ranks = [plan(inputs, rank) for rank in (0, 1)]
    for value in ranks:
        assert "/cache/jit" in value["binds"]
        for key in ("XDG_CACHE_HOME", "VLLM_CACHE_ROOT", "B12X_ROCE_CACHE_DIR", "B12X_COMPILE_CACHE_DIR"):
            assert value["environment"][key].startswith("/cache/jit/")
            assert value["profile_sha256"] in value["environment"][key]
    assert ranks[0]["environment"]["VLLM_CACHE_ROOT"] != ranks[1]["environment"]["VLLM_CACHE_ROOT"]


def test_tp2_clears_inherited_mesh_sitecustomize(inputs, tmp_path):
    import sys

    (tmp_path / "sitecustomize.py").write_text("raise SystemExit('TP4_HOOK_IMPORTED')\n")
    inherited = {**os.environ, "PYTHONPATH": str(tmp_path)}
    command = [sys.executable, "-c", "print('CONSUMER_REACHED')"]
    trapped = subprocess.run(command, env=inherited, capture_output=True, text=True)
    assert trapped.returncode != 0 and "TP4_HOOK_IMPORTED" in trapped.stderr
    environment = plan(inputs)["environment"]
    assert environment["PYTHONPATH"] == ""
    selected = subprocess.run(command, env={**inherited, "PYTHONPATH": environment["PYTHONPATH"]},
                              capture_output=True, text=True)
    assert selected.returncode == 0, selected.stderr
    assert "CONSUMER_REACHED" in selected.stdout


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


@pytest.mark.parametrize("damage", ["restart", "image", "environment", "mount", "healthcheck"])
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
    elif damage == "healthcheck":
        host.container["Config"]["Healthcheck"] = {"Test": ["CMD-SHELL", "test -f /tmp/sparkring-engine-ready"]}
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


def test_managed_option_preserves_shared_default_and_reference_evidence():
    dependencies = json.loads((ROOT / "dependencies.json").read_text())
    loader = next(item for item in dependencies["shared_image_requirements"]
                  if item["component"] == "B12X loader")
    assert loader["model_loader_extra_config"] == {"allocation": "managed"}
    assert loader["shared_loader_default_allocation"] == "pinned_wc"
    record = json.loads((ROOT.parents[2] / "performance/records/glm53-flash/tp2-single-dac-source-20260908.json").read_text())
    assert "--model-loader-extra-config" not in record["conditions"]["serving_arguments"]
    assert "research-only" in launch.load_profile()["qualification"]["shared_image"]
    assert dependencies["reference_source"]["nccl"]["version"] == "2.30.4"
    assert dependencies["reference_source"]["nccl"]["library"] == "/opt/libnccl-local-inference.so.2.30.4"


@pytest.fixture
def local_source_receipt(inputs, tmp_path, monkeypatch):
    # The common recipe lands independently from this profile. Integrated CI
    # uses its repository copy; a development checkout can name the same files.
    origin = Path(os.environ.get("SPARKRING_TEST_COMMON_SOURCE_IMAGE", str(launch.SOURCE_IMAGE_ROOT)))
    if not (origin / "receipt_contract.py").is_file():
        pytest.skip("Common source-image recipe is required for local receipt integration tests")
    value = launch.render(0, "master.example", *inputs, LOCAL_IMAGE)
    lock = json.loads((origin / "glm53-tp4-lock.json").read_bytes())
    lock["profiles"][value["profile"]] = {
        "profile_sha256": value["profile_sha256"],
        "transport_manifest_sha256": value["transport_manifest_sha256"],
        "tp_size": 2, "dcp_size": 1,
    }
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
        "checks_passed": True, "cuda_initialized": False, "model_loaded": False,
        "source_lock_sha256": lock_hash,
        "inherited_runtime": lock["runtime"]["expected_distributions"],
        "packages": {name: {"revision": row["revision"], "file_map_sha256": row["installed_file_map_sha256"],
                            "files": row["installed_file_count"]} for name, row in lock["sources"].items()},
        "transport_profiles": {"tp2-rocenante-adaptive": {
            "manifest_sha256": value["transport_manifest_sha256"],
            "files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "files": len(files), "package": "b12x.comm.roce",
        }},
    }
    for field in ("bundle_manifest_sha256", "transport_sha256", "marker_source_sha256",
                  "marker_binary_sha256", "nccl_sha256", "retained_vllm_native_sha256", "readiness_warmup"):
        inside[field] = lock["runtime"][field]
    runtime = {
        "schema": "sparkring-source-image-receipt/v1", "image_id": LOCAL_IMAGE,
        "image_reference": LOCAL_IMAGE, "platform": "linux/arm64", "checks_passed": True,
        "profile": value["profile"], "source_lock_sha256": lock_hash, "inside_image": inside,
    }
    monkeypatch.setattr(launch, "SOURCE_IMAGE_ROOT", target)
    return value, runtime, lock_path


def test_local_image_requires_complete_source_and_transport_witness(local_source_receipt):
    value, runtime, _ = local_source_receipt
    assert value["image_identity_kind"] == "local_config_id"
    launch.validate_runtime_receipt(runtime, value)
    host = Host(value)
    launch.execute(value, "create", runtime, run=host.run)
    assert host.commands[-1] == value["command"]
    launch.execute(value, "start", runtime, run=host.run)
    assert host.commands[-1] == ["docker", "start", value["name"]]
    assert "registry_digest" not in runtime


@pytest.mark.parametrize("damage", ["source_lock", "package_map", "transport_map", "transport_count",
                                    "image_id", "profile_hash", "world_size"])
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
        key, changed = ("profile_sha256", "0" * 64) if damage == "profile_hash" else ("tp_size", 4)
        lock["profiles"][value["profile"]][key] = changed
        lock_path.write_text(json.dumps(lock) + "\n")
        digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        runtime["source_lock_sha256"] = runtime["inside_image"]["source_lock_sha256"] = digest
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
def test_pinned_receipt_validates_in_fresh_process(local_source_receipt, tmp_path, damage):
    value, runtime, lock_path = local_source_receipt
    contract = launch._source_receipt_contract(lock_path.parent)
    runtime["native_mode"] = runtime["inside_image"]["native_mode"] = "pinned"
    runtime["inside_image"]["native_files"] = contract.expected_record(json.loads(lock_path.read_bytes()))
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
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT / "launch.py"),
                             str(lock_path.parent), str(inputs_path)], capture_output=True, text=True)
    if damage:
        assert result.returncode != 0
        assert "native artifact witness differs" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "receipt valid"
