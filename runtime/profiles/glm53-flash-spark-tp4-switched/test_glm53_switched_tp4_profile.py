"""CPU contracts for switched NCCL selection and guarded manual serving."""

import copy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("switched_tp4_launch", ROOT / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
IMAGE = "ghcr.io/fujitsupolycom/sparkring@sha256:" + "a" * 64


@pytest.fixture
def inputs(tmp_path):
    model, cache = tmp_path / "model", tmp_path / "cache"
    model.mkdir()
    cache.mkdir()
    (model / "config.json").write_text("{}")
    site = tmp_path / "rank.env"
    site.write_text("VLLM_HOST_IP=rank.example\nNCCL_SOCKET_IFNAME=control0\n"
                    "GLOO_SOCKET_IFNAME=control0\nNCCL_IB_HCA==mlx5_7:1\nNCCL_IB_GID_INDEX=3\n")
    return model, cache, site


def plan(inputs, rank=0, image=IMAGE):
    return launch.render(rank, "master.example", *inputs, image)


def registry_receipt(value):
    return {"registry_digest": value["image"], "profiles": {value["profile"]: {
        key: value[key] for key in ("profile_sha256", "topology", "collective_backend")
    }}}


def receipt(value):
    result = registry_receipt(value)
    result["profiles"][value["profile"]]["source_compatibility"] = "passed"
    return result


def test_model_and_prefill_contract():
    profile = launch.load_profile()
    assert profile["model"]["repository"] == "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"
    assert profile["model"]["revision"] == "df116c4fb16b1d37ae43d2cfd624de26ffbc832e"
    args = profile["vllm_args"]
    for flag, expected in {
        "--tensor-parallel-size": "4", "--decode-context-parallel-size": "1",
        "--nnodes": "4", "--max-model-len": "1048576",
        "--kv-cache-memory-bytes": "25769803776", "--max-num-batched-tokens": "8192",
        "--max-num-seqs": "16", "--max-cudagraph-capture-size": "64",
        "--load-format": "fastsafetensors", "--recurrent-checkpoint-policy": "aligned",
        "--prefix-cache-retention-interval": "0", "--kda-prefill-backend": "b12x",
    }.items():
        assert args[args.index(flag) + 1] == expected
    speculative = json.loads(args[args.index("--speculative-config") + 1])
    assert speculative["method"] == "mtp" and speculative["num_speculative_tokens"] == 3
    assert not any(key.startswith("adaptive") for key in speculative)
    assert profile["environment"]["VLLM_B12X_KDA_PREFILL_COALESCING"] == "1"
    assert profile["environment"]["VLLM_GLM53_MHC_PREFILL_SHARD"] == "1"
    assert "--disable-custom-all-reduce" in args
    assert "--kv-transfer-config" not in args
    assert not profile["qualification"]["gpu_serving_qualified"]


@pytest.mark.parametrize("rank", range(4))
def test_all_ranks_select_ordinary_nccl_and_direct_verified_dispatch(inputs, rank):
    value = plan(inputs, rank)
    env = value["environment"]
    for key in ("PYTHONPATH", "VLLM_PLUGINS", "SPARKRING_TRANSPORT_PROFILE",
                "SPARKRING_TRANSPORT_MANIFEST_SHA256", "VLLM_SPARK_TP4_MODE", "VLLM_SPARK_TP4_VOCAB_MODE"):
        assert env[key] == ""
    for key in ("VLLM_ENABLE_ROCE_ALLREDUCE", "VLLM_ENABLE_PCIE_ALLREDUCE", "SPARK_TP4_HEALTH_GATE",
                "VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_USE_NCCL_SYMM_MEM",
                "NCCL_SWITCHLESS_RING_ONLY", "NCCL_IB_SUBNET_AWARE_ROUTING",
                "NCCL_IB_ROUTE_DIAGNOSTICS", "SPARKCACHE_ENABLED",
                "SPARKCACHE_ASYNC_PAGE_CAPTURE", "VLLM_DCP_TOPK_OWNER_MERGE", "VLLM_DCP_OWNER_FUSED_ENDPOINTS",
                "VLLM_DCP_COMPACT_INDEX_CACHE_OWNER", "VLLM_DCP_COMPACT_INDEX_TENSOR_VOTE",
                "VLLM_DCP_COMPACT_INDEX_LOCAL_WIDTHS", "VLLM_DCP_COMPACT_INDEX_PROFILE"):
        assert env[key] == "0", key
    assert env["NCCL_IB_EXTENDED_IPV4_GIDS"] == "1"
    assert env["NCCL_IB_PRESERVE_PCI_DOMAIN"] == "1"
    assert env["SOURCE_IMAGE_PROFILE"] == "glm53-flash-spark-tp4-switched-mtp3"
    assert value["container_args"][:4] == ["-S", "-B", "/opt/sparkcache-jj-runtime/verify_sources.py", "--serve"]
    assert ("--headless" in value["container_args"]) == (rank != 0)
    assert value["command"][:2] == ["docker", "create"]
    assert "--no-healthcheck" not in value["command"]
    assert value["command"][value["command"].index("--health-cmd") + 1] == launch.HEALTHCHECK_COMMAND
    assert env["PORT"] == "8000" and env["SERVED_MODEL_NAME"] == "glm-5.3-flash-spark"
    assert env["SPARKRING_NODE_RANK"] == str(rank)
    assert env["DFLASH_WARMUP"] == "1"
    assert env["DFLASH_WARMUP_CONCURRENCIES"] == "1"
    assert env["DFLASH_WARMUP_SHAPE_WORDS"] == "8"
    assert env["DFLASH_WARMUP_MAX_TOKENS"] == "16"
    assert value["command"][value["command"].index("--restart") + 1] == "no"
    owner = inputs[1].stat()
    assert value["container_user"] == f"{owner.st_uid}:{owner.st_gid}"
    assert value["command"][value["command"].index("--user") + 1] == value["container_user"]
    assert value["transport_profile"] is None
    assert not any(key.startswith("SPARK_TP4_PEER") or key == "B12X_ROCE_PEER_HCA_MAP" for key in env)


@pytest.mark.parametrize("selector", ["=mlx5_7:1", "=mlx5_7:1,mlx5_9:1", "=hcaA,hcaB,hcaC"])
def test_operator_selection_is_preserved_without_uplink_assumptions(inputs, selector):
    text = inputs[2].read_text().replace("=mlx5_7:1", selector)
    inputs[2].write_text(text)
    value = plan(inputs)
    assert value["environment"]["NCCL_IB_HCA"] == selector
    assert value["environment"]["NCCL_SOCKET_IFNAME"] == "control0"


@pytest.mark.parametrize("selector", ["mlx5_7", "=mlx5_*", "^mlx5_7", "=mlx5_7:1,mlx5_7:1", "<unresolved>"])
def test_broad_duplicate_or_unresolved_hca_selection_is_rejected(inputs, selector):
    inputs[2].write_text(inputs[2].read_text().replace("=mlx5_7:1", selector))
    with pytest.raises(ValueError):
        plan(inputs)


@pytest.mark.parametrize("override", ["NCCL_SWITCHLESS_RING_ONLY=1", "VLLM_ENABLE_ROCE_ALLREDUCE=1", "SPARKCACHE_ENABLED=1"])
def test_site_input_cannot_enable_another_profile(inputs, override):
    inputs[2].write_text(inputs[2].read_text() + override + "\n")
    with pytest.raises(ValueError):
        plan(inputs)


class Host:
    def __init__(self, value, floor=4294967296):
        self.value, self.floor, self.commands = value, floor, []
        self.container = {"Config": {"Image": value["image"], "User": value["container_user"],
            "Entrypoint": ["python3"], "Cmd": value["container_args"], "Healthcheck": {"Test": ["CMD-SHELL", launch.HEALTHCHECK_COMMAND]},
            "Labels": value["labels"], "Env": [key + "=" + item for key, item in value["environment"].items()]},
            "HostConfig": {"RestartPolicy": {"Name": "no"}}, "Mounts": [
                {"Destination": target, "Source": source, "RW": target != "/models/target"}
                for target, source in value["binds"].items()]}

    def run(self, command, **kwargs):
        self.commands.append(command)
        output = ""
        if command[:2] == ["systemctl", "show"]:
            output = f"argv[]=/usr/bin/python3 guard --available-floor-bytes {self.floor} ;"
        if command == ["docker", "inspect", self.value["name"]]:
            output = json.dumps([self.container])
        return subprocess.CompletedProcess(command, 0, output, "")


def test_manual_create_and_start_keep_guard_and_health_contract(inputs):
    value = plan(inputs)
    host = Host(value)
    launch.execute(value, "create", receipt(value), run=host.run)
    assert host.commands[-1] == value["command"]
    assert not any(command[:2] == ["docker", "start"] for command in host.commands)
    launch.execute(value, "start", receipt(value), run=host.run)
    assert host.commands[-1] == ["docker", "start", value["name"]]


@pytest.mark.parametrize("damage", ["guard", "healthcheck", "user", "backend"])
def test_invalid_manual_start_is_rejected(inputs, damage):
    value = plan(inputs)
    host = Host(value)
    if damage == "guard":
        host.floor = 2147483648
    elif damage == "healthcheck":
        host.container["Config"]["Healthcheck"]["Test"] = ["CMD", "mesh-readiness"]
    elif damage == "user":
        host.container["Config"]["User"] = "other"
    else:
        host.container["Config"]["Env"] = []
    with pytest.raises(RuntimeError):
        launch.execute(value, "start", receipt(value), run=host.run)
    assert ["docker", "start", value["name"]] not in host.commands


def test_common_local_source_receipt_checks_exact_lock_and_profile(inputs, tmp_path, monkeypatch):
    value = plan(inputs, image="sha256:" + "b" * 64)
    common = launch.SOURCE_IMAGE_ROOT
    lock = json.loads((common / "glm53-tp4-lock.json").read_bytes())
    lock["profiles"][value["profile"]] = {
        key: value[key] for key in ("profile_sha256", "topology", "collective_backend")
    } | {"tp_size": 4, "dcp_size": 1, "sparkcache": False, "transport_profile": None}
    copied = tmp_path / "source-image"
    copied.mkdir()
    for filename in ("archive_utils.py", "native_files.py", "receipt_contract.py"):
        shutil.copyfile(common / filename, copied / filename)
    path = copied / "glm53-tp4-lock.json"
    path.write_text(json.dumps(lock) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    inside = {"checks_passed": True, "cuda_initialized": False, "model_loaded": False,
              "source_lock_sha256": digest, "inherited_runtime": lock["runtime"]["expected_distributions"],
              "packages": {name: {"revision": row["revision"], "file_map_sha256": row["installed_file_map_sha256"],
                                   "files": row["installed_file_count"]} for name, row in lock["sources"].items()}}
    for field in ("bundle_manifest_sha256", "transport_sha256", "marker_source_sha256", "marker_binary_sha256",
                  "nccl_sha256", "retained_vllm_native_sha256", "readiness_warmup"):
        inside[field] = lock["runtime"][field]
    document = {"schema": "sparkring-source-image-receipt/v1", "image_id": value["image"],
                "image_reference": value["image"], "platform": "linux/arm64", "checks_passed": True,
                "profile": value["profile"], "source_lock_sha256": digest, "inside_image": inside}
    monkeypatch.setattr(launch, "SOURCE_IMAGE_ROOT", copied)
    launch.validate_runtime_receipt(document, value)
    document["native_mode"] = inside["native_mode"] = "pinned"
    inside["native_files"] = launch._source_receipt_contract(copied).expected_record(lock)
    inputs_path = tmp_path / "receipt-inputs.json"
    inputs_path.write_text(json.dumps({"plan": value, "receipt": document}))
    script = """
import importlib.util,json,sys
from pathlib import Path
assert 'archive_utils' not in sys.modules and 'native_files' not in sys.modules
spec=importlib.util.spec_from_file_location('fresh_switched_launcher',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.SOURCE_IMAGE_ROOT=Path(sys.argv[2])
data=json.loads(Path(sys.argv[3]).read_bytes())
module.validate_runtime_receipt(data['receipt'],data['plan'])
assert 'archive_utils' not in sys.modules and 'native_files' not in sys.modules
print('receipt valid')
"""
    command = [sys.executable, "-I", "-B", "-c", script, str(ROOT / "launch.py"), str(copied), str(inputs_path)]
    checked = subprocess.run(command, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert checked.stdout.strip() == "receipt valid"
    bad_native = copy.deepcopy(document)
    bad_native["inside_image"]["native_files"]["archive_sha256"] = "0" * 64
    inputs_path.write_text(json.dumps({"plan": value, "receipt": bad_native}))
    rejected = subprocess.run(command, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "native artifact witness differs" in rejected.stderr
    changed = copy.deepcopy(document)
    changed["inside_image"]["packages"]["vllm"]["file_map_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        launch.validate_runtime_receipt(changed, value)
    path.write_text(json.dumps(lock, indent=2))
    with pytest.raises(ValueError, match="exact source lock"):
        launch.validate_runtime_receipt(document, value)


def test_generic_warmup_owns_readiness_and_preserves_six_sampler_cases(tmp_path):
    source = (launch.SOURCE_IMAGE_ROOT / "startup/serve_with_warmup.py").read_text()
    tree = ast.parse(source)
    cases = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "SAMPLING_CASES" for target in node.targets))
    assert [row[0] for row in ast.literal_eval(cases)] == [
        "unfiltered", "temperature", "top-k", "top-p", "top-k-top-p", "seeded-top-k-top-p",
    ]
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "complete_readiness")
    marker = tmp_path / "ready"
    calls = []

    def completed(name):
        assert not marker.exists()
        calls.append(name)
        return {"completed": name}

    namespace = {"Path": Path, "READY_PATH": marker, "json": json,
                 "warmup_dflash": SimpleNamespace(make_deadline=lambda seconds: seconds,
                     remaining_seconds=lambda deadline: deadline,
                     wait_for_api=lambda *args: completed("api"),
                     run_warmup=lambda *args: completed("shape")),
                 "warmup_sampling": lambda *args: completed("sampling")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "generic-readiness", "exec"), namespace)
    env = launch.load_profile()["environment"]
    namespace["complete_readiness"](
        rank=0, endpoint="http://127.0.0.1:" + env["PORT"], model=env["SERVED_MODEL_NAME"],
        warmup_enabled=env["DFLASH_WARMUP"] == "1", concurrencies=(1,), shape_words=(8,),
        max_tokens=int(env["DFLASH_WARMUP_MAX_TOKENS"]),
        timeout_seconds=float(env["DFLASH_WARMUP_TIMEOUT_SECONDS"]), credential=None, ready_path=marker,
    )
    assert calls == ["api", "shape", "sampling"] and marker.exists()
