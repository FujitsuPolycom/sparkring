"""CPU-only semantic parity with the retained launcher, used solely as an oracle."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from runtime.common import candidate, glm_tp4, r35
from runtime.common.test_candidate import host_receipt
from runtime.common.test_candidate import inputs as inputs
from runtime.common.test_r35 import receipt

ROOT = Path(__file__).resolve().parents[2]
MESH = ROOT / "runtime/glm53-spark-mtp3-mesh"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


mesh = module("glm_spec_mesh_profile", MESH / "profile.py")
example = module("glm_spec_mesh_example", MESH / "make_example.py")


@pytest.fixture(params=["r35", "candidate"])
def rendered(request, tmp_path, monkeypatch, inputs):
    """Use existing receipt and topology fixtures; never claim hardware admission."""
    image = r35.validate_receipt(receipt()) if request.param == "r35" else candidate.validate_receipt(host_receipt(inputs, monkeypatch))
    adapter = r35 if request.param == "r35" else candidate
    image_path = tmp_path / "image.json"
    image_path.write_text(json.dumps(image))
    (tmp_path / "fabric.example.json").write_text(json.dumps(example.topology_example()))
    monkeypatch.setattr(mesh, "verify_bundle", lambda bundle, record: record["bundle_manifest_sha256"])
    results = {}
    for profile in sorted(glm_tp4.PROFILES):
        site = dict(example.site_example(), runtime_profile=profile)
        site_path = tmp_path / "site.json"
        site_path.write_text(json.dumps(site))
        output = tmp_path / profile
        mesh.render(site_path, tmp_path / "bundle", output, image_path)
        legacy = (ROOT / "runtime/glm53-flash-jj-r8-gb10/launch-rank.sh").read_text()
        if request.param == "candidate":
            legacy = candidate.adapt_launcher(legacy, image["installed"])
        (output / "legacy-oracle.sh").write_text(legacy, newline="\n")
        results[profile] = (output, [mesh.defaults(output / f"rank{rank}.env") for rank in range(4)])
    return image, adapter.profile_contract(image["installed"]), results


def oracle(launcher, environment, rank, tmp_path):
    """Shell is a test oracle only. Offline mode cannot inspect or create Docker."""
    bash = shutil.which("bash")
    if os.name == "nt":
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        bash = str(git_bash) if git_bash.is_file() else None
    if bash is None:
        pytest.skip("Bash is unavailable for the legacy semantic oracle")
    values = dict(environment, SPARKRING_CREATE_ONLY="1", SPARKRING_PRINT_CONTAINER_SPEC="1", SPARKRING_OFFLINE_SPEC="1")
    path = tmp_path / "oracle.env"
    path.write_text("\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n", newline="\n")
    child_env = dict(os.environ, PYTHON_FOR_ORACLE=sys.executable, MSYS_NO_PATHCONV="1")
    # Git Bash need not have a python3 executable; use this test process's Python.
    script = 'python3() { "$PYTHON_FOR_ORACLE" "$@"; }; export -f python3; source "$1" "$2" "$3"'
    result = subprocess.run([bash, "-c", script, "oracle", launcher.as_posix(), str(rank), path.as_posix()],
                            env=child_env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["argv"]


def split_oracle(argv, image_ref):
    """Interpret the test oracle's Docker envelope, retaining its ordered argv."""
    index = argv.index(image_ref)
    envelope = argv[2:index]
    options, env, labels, mounts = {}, {}, {}, []
    cursor = 0
    while cursor < len(envelope):
        key = envelope[cursor]
        cursor += 1
        if key == "--init":
            options[key] = True
            continue
        value = envelope[cursor]
        cursor += 1
        if key == "-e":
            name, content = value.split("=", 1)
            env[name] = content
        elif key == "--label":
            name, content = value.split("=", 1)
            labels[name] = content
        elif key == "-v":
            mounts.append(value)
        else:
            assert key not in options
            options[key] = value
    return options, env, labels, mounts, argv[index + 1:]


def semantic_command(argv):
    return [json.loads(value) if value.startswith("{") else value for value in argv]


def test_all_profiles_and_ranks_match_legacy_semantics(rendered, tmp_path):
    image, contract, profiles = rendered
    for profile, (output, ranks) in profiles.items():
        for rank, values in enumerate(ranks):
            before = deepcopy(values)
            spec = glm_tp4.build_spec(values, image_record=image, contract=contract)
            args = oracle(output / "legacy-oracle.sh", values, rank, tmp_path)
            options, env, labels, mounts, command = split_oracle(args, values["IMAGE_REF"])
            assert semantic_command(spec.command) == semantic_command(command), (profile, rank)
            assert spec.environment == env, (profile, rank)
            assert spec.labels == labels
            assert [f"{bind.source}:{bind.target}" + (":ro" if bind.read_only else "") for bind in spec.mounts] == mounts
            assert spec.name == options.pop("--name")
            assert spec.entrypoint == (options.pop("--entrypoint"),)
            assert spec.network_mode == options.pop("--network")
            assert spec.ipc_mode == options.pop("--ipc")
            assert spec.shm_size == 32 * 1024**3 and options.pop("--shm-size") == "32g"
            assert spec.cap_add == (options.pop("--cap-add"),)
            assert spec.security_opt == (options.pop("--security-opt"),)
            assert spec.devices == (options.pop("--device"),)
            assert spec.memlock == -1 and options.pop("--ulimit") == "memlock=-1:-1"
            assert spec.gpu_count == -1 and options.pop("--gpus") == "all"
            assert spec.init is options.pop("--init")
            assert spec.memory is None and spec.memory_swap is None
            if rank == 0:
                assert spec.effective_health_mode == "shell"
                assert spec.health_command == (options.pop("--health-cmd"),)
                for attr, option in (("health_interval", "interval"), ("health_timeout", "timeout"), ("health_start_period", "start-period")):
                    assert str(getattr(spec, attr)) + "s" == options.pop("--health-" + option)
                assert str(spec.health_retries) == options.pop("--health-retries")
            else:
                assert spec.effective_health_mode == "inherit"
            assert not options
            assert values == before


def test_normal_tuning_and_text_mode_preserve_shell_semantics(rendered, tmp_path):
    image, contract, profiles = rendered
    output, ranks = profiles["tp4-dcp1-sparkcache"]
    values = dict(ranks[0], MULTIMODAL_INPUTS="0", OMP_NUM_THREADS="1", SPARK_TP4_GRAPH_SUBMIT_CPU="8",
                  SPARK_TP4_GRAPH_PROGRESS_CPU="9", SPARK_TP4_GRAPH_DIRECT_DOORBELL="0", PORT="8115",
                  SPARKRING_LIVENESS_PORT="8116", MAX_NUM_SEQS="8", MAX_NUM_BATCHED_TOKENS="4096",
                  PREFILL_SCHEDULE_INTERVAL="8", ENABLE_PROMPT_TOKENS_DETAILS="0", JIT_MONITOR_VERBOSE="1")
    spec = glm_tp4.build_spec(values, image_record=image, contract=contract)
    _, env, labels, _, command = split_oracle(oracle(output / "legacy-oracle.sh", values, 0, tmp_path), values["IMAGE_REF"])
    assert spec.environment == env
    assert spec.labels == labels
    assert semantic_command(spec.command) == semantic_command(command)


@pytest.mark.parametrize("key,value", [
    ("SOURCE_IMAGE_PROFILE", "tp4-dcp1-mtp3-prefill"), ("SPARKRING_RUNTIME_RELEASE", "r33"),
    ("SPECULATION_METHOD", "dflash"), ("NUM_SPECULATIVE_TOKENS", "7"), ("TARGET_MODEL_VARIANT", "nvfp4"),
    ("SPARKCACHE_ACCESS_MODE", "restore-only"), ("SPARKCACHE_ASYNC_PAGE_CAPTURE", "0"),
    ("SPARKCACHE_BUFFER_BUDGET_BYTES", "0"), ("SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES", "3221225472"),
    ("SIRCL_BUNDLE_HOST_ROOT", "/tmp/source"), ("SPARKCACHE_SOURCE_OVERLAY", "/tmp/source"),
    ("VLLM_KV_METRICS_OVERLAY", "/tmp/source"), ("R33_PROFILE_CONTRACT_HOST_ROOT", "/tmp/source"),
    ("SPARK_CUDAGRAPH_REPLAY_TIMING", "1"), ("VLLM_GLM53_MHC_PREFILL_SHARD", "0"),
    ("VLLM_B12X_KDA_PREFILL_COALESCING", "0"), ("VLLM_SPARK_TP4_VOCAB_MODE", "stock"),
    ("VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE", "single"), ("NODE_RANK", "4"),
    ("SPARKRING_NODE_RANK", "1"), ("NCCL_IB_HCA", "roce0,roce1"), ("NCCL_LIBRARY_SHA256", "0" * 64),
    ("SPARKRING_DECLARED_SIRCL_NATIVE_SHA256", "0" * 64), ("SPARKCACHE_SOURCE_LEASE_CONTRACT", "/wrong.json"),
    ("MAX_CUDAGRAPH_CAPTURE_SIZE", "9223372036854775807"), ("MAX_NUM_SEQS", "016"),
    ("GPU_MEMORY_UTILIZATION", "nan"), ("TARGET_MODEL_HOST_PATH", "/model,alias"), ("SHM_SIZE", "0g"),
    ("PYTHONPATH", "/source-override"), ("VLLM_SPARK_SHARED_CAPTURE_STREAM", "0"),
])
def test_unsupported_or_unbound_settings_fail_closed(rendered, key, value):
    image, contract, profiles = rendered
    values = dict(profiles["tp4-dcp1-sparkcache"][1][0], **{key: value})
    with pytest.raises(ValueError):
        glm_tp4.build_spec(values, image_record=image, contract=contract)


def test_normalization_is_explicit_idempotent_and_input_isolated(rendered):
    _, _, profiles = rendered
    for _, ranks in profiles.values():
        for values in ranks:
            normalized = glm_tp4.normalize_environment(values)
            assert normalized == glm_tp4.normalize_environment(normalized)
            assert normalized["CP_KV_CACHE_INTERLEAVE_SIZE"] == values["DECODE_CONTEXT_PARALLEL_SIZE"]
            assert normalized["B12X_MLA_CKV_GATHER"] == str(int(values["DECODE_CONTEXT_PARALLEL_SIZE"] == "4"))
            assert normalized["SPARKCACHE_CLEAR_ONCE"] == values["SPARKCACHE_CACHE_NAMESPACE"]
            assert normalized["NCCL_DEBUG"] == "WARN"


def test_missing_secret_file_material_fails_before_emitting_spec(rendered):
    image, contract, profiles = rendered
    values = dict(profiles["tp4-dcp1"][1][0], API_KEYS_FILE="/private/api-keys")
    with pytest.raises(ValueError, match="API_KEYS_FILE"):
        glm_tp4.build_spec(values, image_record=image, contract=contract)


def test_image_and_native_contract_must_match(rendered):
    image, contract, profiles = rendered
    values = profiles["tp4-dcp4-sparkcache"][1][0]
    for bad in (dict(image, image_id="sha256:" + "0" * 64), dict(image, platform="linux/amd64")):
        with pytest.raises(ValueError):
            glm_tp4.build_spec(values, image_record=bad, contract=contract)
    changed = deepcopy(contract)
    changed["sparkcache_native"]["snapshot_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="native contract"):
        glm_tp4.build_spec(values, image_record=image, contract=changed)
