"""Execute rendered launchers against fake Docker and hash commands, without GPUs."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


HERE = Path(__file__).resolve().parent


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profile = _module("mtp3_launcher_profile_test", "profile.py")
example = _module("mtp3_launcher_example_test", "make_example.py")


def _bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    drive, tail = os.path.splitdrive(str(path))
    assert drive
    return f"/mnt/{drive[0].lower()}/" + tail.lstrip("\\/").replace("\\", "/")


def _write_executable(path, source):
    path.write_text(source, encoding="utf-8", newline="\n")
    path.chmod(0o755)


@pytest.fixture
def launch_fixture(tmp_path, monkeypatch):
    """Only the generated shell and JSON encoding run; every Docker call is intercepted."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    records = []
    for name in sorted(profile.build_bundle.REQUIRED_SIRCL_FILES):
        path = bundle / name
        path.write_text("CPU-only launcher fixture\n", encoding="utf-8")
        records.append({"path": name, "sha256": profile.sha(path)})
    manifest = bundle / "sparkring-overlay-manifest.json"
    manifest.write_text(json.dumps({"files": records}), encoding="utf-8")
    monkeypatch.setitem(profile.PINS, "canonical_bundle_manifest_sha256", profile.sha(manifest))
    site = example.site_example()
    for rank in range(4):
        target = tmp_path / f"target-r{rank}"
        target.mkdir()
        for name in ("config.json", "model.safetensors.index.json"):
            (target / name).write_text("hash-command fixture", encoding="utf-8")
        cache = tmp_path / f"cache-r{rank}"
        cache.mkdir()
        site["model_roots"][rank] = _bash_path(target)
        site["cache_roots"][rank] = _bash_path(cache)
    site["bundle_root"] = _bash_path(bundle)
    (tmp_path / "fabric.example.json").write_text(json.dumps(example.topology_example()))
    site_path = tmp_path / "site.json"
    site_path.write_text(json.dumps(site))
    output = tmp_path / "rendered"
    profile.render(site_path, bundle, output)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "docker", """#!/bin/sh
if [ "$1" = image ] && [ "$2" = inspect ]; then
  printf '%s\n' "$FIXTURE_IMAGE_ID"
elif [ "$1" = container ] && [ "$2" = inspect ]; then
  exit 1
elif [ "$1" = run ] || [ "$1" = create ]; then
  printf '%s\n' "$@" > "$FIXTURE_DOCKER_CAPTURE"
  printf '%s\n' 'cpu-only-container-fixture'
elif [ "$1" = inspect ]; then
  printf '%s\n' "${FIXTURE_HEALTH_STATE:-healthy}"
else
  printf 'Unexpected fake-Docker operation: %s\n' "$1" >&2
  exit 97
fi
""")
    _write_executable(fake_bin / "sha256sum", """#!/bin/sh
printf '%s\n' "$2" >> "$FIXTURE_HASH_CAPTURE"
case "$2" in
  */target-r*/config.json) hash="$FIXTURE_TARGET_CONFIG_SHA" ;;
  */target-r*/model.safetensors.index.json) hash="$FIXTURE_TARGET_INDEX_SHA" ;;
  */libspark_transport_capi.so) hash="$FIXTURE_NATIVE_SHA" ;;
  */sparkring-overlay-manifest.json) hash="$FIXTURE_MANIFEST_SHA" ;;
  *) printf 'Unexpected file hashed: %s\n' "$2" >&2; exit 95 ;;
esac
printf '%s  %s\n' "$hash" "$2"
""")

    def launch(rank, overrides=None, fixture_overrides=None):
        capture = tmp_path / f"docker-r{rank}.txt"
        hash_capture = tmp_path / f"hash-r{rank}.txt"
        config = output / f"test-rank{rank}.env"
        variables = {
            "FIXTURE_IMAGE_ID": profile.defaults(profile.BASE / "runtime.env.example")["IMAGE_ID"],
            "FIXTURE_DOCKER_CAPTURE": _bash_path(capture), "FIXTURE_HASH_CAPTURE": _bash_path(hash_capture),
            "FIXTURE_TARGET_CONFIG_SHA": profile.PINS["target"]["config_sha256"],
            "FIXTURE_TARGET_INDEX_SHA": profile.PINS["target"]["index_sha256"],
            "FIXTURE_NATIVE_SHA": profile.sha(bundle / "libspark_transport_capi.so"),
            "FIXTURE_MANIFEST_SHA": profile.sha(manifest),
        }
        variables.update(fixture_overrides or {})
        appended = [f"PATH={shlex.quote(_bash_path(fake_bin))}:$PATH"]
        appended += [f"export {key}={shlex.quote(value)}" for key, value in variables.items()]
        appended += [f"{key}={shlex.quote(value)}" for key, value in (overrides or {}).items()]
        config.write_text((output / f"rank{rank}.env").read_text() + "\n".join(appended) + "\n",
                          encoding="utf-8", newline="\n")
        result = subprocess.run(["bash", _bash_path(output / "launch-rank.sh"), str(rank), _bash_path(config)],
                                cwd=profile.ROOT, text=True, capture_output=True, timeout=30)
        arguments = capture.read_text().splitlines() if capture.exists() else []
        hashes = hash_capture.read_text().splitlines() if hash_capture.exists() else []
        return result, arguments, hashes

    return launch, site, output


def _option(arguments, name):
    assert arguments.count(name) == 1
    return arguments[arguments.index(name) + 1]


@pytest.mark.parametrize('rank', range(4))
def test_prepare_only_creates_stopped_container_without_readiness_wait(launch_fixture, rank):
    launch, _, _ = launch_fixture
    result, arguments, _ = launch(rank, {'SPARKRING_CREATE_ONLY': '1'},
                                  {'FIXTURE_HEALTH_STATE': 'unhealthy'})
    assert result.returncode == 0, result.stderr
    assert arguments[0] == 'create'
    assert '-d' not in arguments


def test_prepare_only_rejects_invalid_switch(launch_fixture):
    launch, _, _ = launch_fixture
    result, arguments, _ = launch(0, {'SPARKRING_CREATE_ONLY': 'unexpected'})
    assert result.returncode != 0
    assert not arguments


@pytest.mark.parametrize("rank", range(4))
def test_rendered_mtp_launcher_runs_without_external_draft(launch_fixture, rank):
    launch, site, output = launch_fixture
    result, arguments, hashes = launch(rank)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "cpu-only-container-fixture"
    assert _option(arguments, "--name") == f"{site['container_prefix']}-r{rank}"
    assert _option(arguments, "--served-model-name") == "glm-5.3-flash-spark"
    assert f"{site['model_roots'][rank]}:/models/target:ro" in arguments
    assert f"{site['cache_roots'][rank]}:/cache/jit" in arguments
    assert f"{site['bundle_root']}:/opt/spark-sircl:ro" in arguments
    assert not any("/dflash-draft" in arg for arg in arguments)
    assert not any("draft/" in path for path in hashes)
    assert sorted(Path(path).name for path in hashes) == [
        "config.json", "libspark_transport_capi.so", "model.safetensors.index.json", "sparkring-overlay-manifest.json"]
    speculation = json.loads(_option(arguments, "--speculative-config"))
    assert speculation == {
        "method": "mtp", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 4,
        "kv_cache_dtype": "auto", "draft_sample_method": "probabilistic", "rejection_sample_method": "standard",
        "draft_load_config": {"load_format": "safetensors"}, "attention_backend": "B12X",
    }
    compilation = json.loads(_option(arguments, "--compilation-config"))
    assert compilation == {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": list(range(4, 65, 4)),
                           "custom_ops": ["all"], "pass_config": {"fuse_allreduce_rms": False}}
    assert _option(arguments, "--max-cudagraph-capture-size") == "64"
    expected_options = {
        "--tensor-parallel-size": "4", "--pipeline-parallel-size": "1", "--decode-context-parallel-size": "4",
        "--node-rank": str(rank), "--nnodes": "4", "--master-addr": "192.0.2.10", "--master-port": "29775",
        "--cp-kv-cache-interleave-size": "4", "--max-model-len": "1048576", "--max-num-seqs": "16",
        "--max-num-batched-tokens": "8192", "--prefill-schedule-interval": "2",
        "--dtype": "bfloat16", "--quantization": "modelopt_mixed", "--kv-cache-dtype": "fp8",
        "--kv-cache-memory-bytes": "25769803776", "--attention-backend": "B12X", "--moe-backend": "b12x",
        "--linear-backend": "b12x", "--kda-prefill-backend": "b12x", "--load-format": "fastsafetensors",
    }
    assert {name: _option(arguments, name) for name in expected_options} == expected_options
    assert ("--headless" in arguments) == (rank != 0)
    for flag in ("--async-scheduling", "--enable-prefix-caching", "--enable-chunked-prefill",
                 "--disable-custom-all-reduce", "--no-enable-flashinfer-autotune"):
        assert flag in arguments
    connector = json.loads(_option(arguments, "--kv-transfer-config"))
    extra = connector["kv_connector_extra_config"]
    assert extra["spark_cache_target_checkpoint_sha256"] == profile.PINS["target"]["checkpoint_identity"]
    assert extra["spark_cache_draft_checkpoint_sha256"] == profile.PINS["target"]["checkpoint_identity"]
    assert extra["spark_cache_draft_policy"] == "separate"
    assert extra["spark_cache_root"] == "/cache/jit/sparkcache-context/" + profile.PINS["cache_identity"]["namespace"]
    assert extra["spark_cache_clear_once"] == profile.PINS["cache_identity"]["namespace"]
    assert extra["spark_cache_publication_schema"] == "tail-cow-v2"
    assert extra["spark_cache_async_page_capture"] is True
    assert extra["spark_cache_async_page_capture_slot_bytes"] == 3221225472
    assert extra["spark_cache_async_page_capture_slot_count"] == 2
    for value in ("VLLM_SPARK_TP4_MODE=custom", "VLLM_SPARK_SHARED_CAPTURE_STREAM=1",
                  "SPARK_TP4_GRAPH_DIRECT_DOORBELL=1", "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL=1",
                  "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE=fused",
                  "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE=dual",
                  "SPARK_TP4_CAPABILITY_VOTE=1", "SPARK_TP4_HEALTH_GATE=1", "SPARK_TP4_FLIGHT_RECORDER=0"):
        assert value in arguments
    assert "DFLASH_WARMUP_CONCURRENCIES=" + ",".join(map(str, range(1, 17))) in arguments
    assert "--max-num-scheduled-tokens" not in arguments
    assert "8200" not in arguments
    assert (output / f"rank{rank}.env").is_file()


@pytest.mark.parametrize("kind", ["config", "index"])
def test_mtp_launcher_refuses_nonmatching_spark_checkpoint_before_docker_run(launch_fixture, kind):
    launch, _, _ = launch_fixture
    fixture = {f"FIXTURE_TARGET_{kind.upper()}_SHA": "f" * 64}
    result, arguments, _ = launch(0, fixture_overrides=fixture)
    assert result.returncode == 78
    assert "identity mismatch" in result.stderr
    assert arguments == []


def test_mtp_launcher_refuses_nonmatching_image_before_docker_run(launch_fixture):
    launch, _, _ = launch_fixture
    result, arguments, _ = launch(0, fixture_overrides={"FIXTURE_IMAGE_ID": "sha256:" + "f" * 64})
    assert result.returncode == 78
    assert "image identity mismatch" in result.stderr
    assert arguments == []


@pytest.mark.parametrize("setting,value,message", [
    ("SPECULATION_METHOD", "unknown", "SPECULATION_METHOD must be dflash or mtp"),
    ("TARGET_MODEL_VARIANT", "unknown", "TARGET_MODEL_VARIANT must be nvfp4 or nvfp4-spark"),
    ("MAX_CUDAGRAPH_CAPTURE_SIZE", "63", "must be divisible"),
])
def test_mtp_launcher_fails_closed_on_invalid_contract(launch_fixture, setting, value, message):
    launch, _, _ = launch_fixture
    result, arguments, _ = launch(0, overrides={setting: value})
    assert result.returncode != 0
    assert message in result.stderr
    assert arguments == []


def test_external_dflash_still_requires_its_checkpoint(launch_fixture):
    launch, _, _ = launch_fixture
    result, arguments, _ = launch(0, overrides={"SPECULATION_METHOD": "dflash"})
    assert result.returncode != 0
    assert "set DFLASH_MODEL_HOST_PATH" in result.stderr
    assert arguments == []


def test_rendered_shell_has_valid_bash_syntax(launch_fixture):
    _, _, output = launch_fixture
    subprocess.run(["bash", "-n", _bash_path(output / "launch-rank.sh")], check=True, timeout=10)
