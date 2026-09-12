"""Offline configuration and command contracts; no Docker or GPU calls."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sglang_launch", HERE / "deepseek_v41_sglang_cycle_serve.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def environment(tmp_path, **overrides):
    cfg = {key: "/operator/" + key.lower() for key in launch.REQUIRED}
    cfg.update(NODE_RANK="0", MASTER_ADDR="192.0.2.10", HOST_IP="192.0.2.10",
               IMAGE="local/example:sglang", IMAGE_ID="sha256:" + "a" * 64,
               NCCL_SO_SHA256="b" * 64, NCCL_SOCKET_IFNAME="eth0", GLOO_SOCKET_IFNAME="eth0",
               NCCL_IB_HCA="rocep1s0f0,rocep1s0f1", NCCL_IB_GID_INDEX="3")
    cfg.update(overrides)
    path = tmp_path / "rank.env"
    path.write_text("\n".join(f"{k}={v}" for k, v in cfg.items()))
    return path


def test_default_plan_keeps_keys_in_file(tmp_path):
    cfg = launch.read_config(environment(tmp_path))
    cmd = launch.command(cfg)
    assert "--gpus" in cmd
    assert "API_KEY_FILE=/run/secrets/api-keys" in cmd
    assert not any(arg.startswith("API_KEY=") for arg in cmd)
    assert not any("LD_PRELOAD" in arg for arg in cmd)
    assert "CONTEXT_LENGTH=262144" in cmd
    assert "CHUNKED_PREFILL_SIZE=4096" in cmd
    assert "MAX_RUNNING_REQUESTS=8" in cmd
    assert any(launch.PINS["nccl_target"] + ":ro" in arg for arg in cmd)
    assert cmd[-2:] == ["/operator/entrypoint.py", "run"]


@pytest.mark.parametrize("values", [
    {"NODE_RANK": "4"}, {"IMAGE_ID": "latest"}, {"NCCL_SO_SHA256": "bad"},
    {"API_KEY_FILE": "relative/path"}, {"API_PORT": "70000"},
    {"MAX_RUNNING_REQUESTS": "0"}, {"MEM_FRACTION_STATIC": "1"},
    {"MASTER_ADDR": "<UNRESOLVED>"}, {"API_KEY": "fixture"},
])
def test_invalid_configuration_rejected(tmp_path, values):
    with pytest.raises(ValueError):
        launch.read_config(environment(tmp_path, **values))


def test_duplicate_rejected(tmp_path):
    path = environment(tmp_path)
    with path.open("a") as out:
        out.write("\nNODE_RANK=1\n")
    with pytest.raises(ValueError):
        launch.read_config(path)


def test_offline_check_never_requires_docker(tmp_path, monkeypatch):
    path = environment(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    result = subprocess.run([sys.executable, str(HERE / "deepseek_v41_sglang_cycle_serve.py"),
                             "--check", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "docker run" in result.stdout


def test_pinned_auth_patch_refuses_drift(tmp_path):
    source = tmp_path / "auth.py"
    source.write_text("# changed upstream module\n")
    result = subprocess.run([sys.executable, str(launch.RUNTIME / "patch-multikey.py"), str(source)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "refusing" in result.stderr


def test_auth_patch_splits_api_keys_but_keeps_admin_exact(tmp_path):
    import secrets
    source = tmp_path / "auth.py"
    source.write_text(
        "def authorized(parts, expected_token):\n"
        "    if len(parts) == 2:\n"
        "        return secrets.compare_digest(parts[1], expected_token)\n"
        "    return False\n"
    )
    result = subprocess.run(
        [sys.executable, str(launch.RUNTIME / "patch-multikey.py"), str(source)],
        capture_output=True, text=True, check=True,
    )
    scope = {"secrets": secrets, "api_key": "first-key,second-key", "admin_api_key": "admin-a,admin-b"}
    exec(compile(result.stdout, str(source), "exec"), scope)
    check = scope["authorized"]
    for key in ("first-key", "second-key", scope["api_key"]):
        assert check(["Bearer", key], scope["api_key"])
    assert not check(["Bearer", "first"], scope["api_key"])
    assert not check(["Bearer", "admin-a"], scope["admin_api_key"])
    assert not check(["Bearer", "first-key"], scope["admin_api_key"])
    assert check(["Bearer", scope["admin_api_key"]], scope["admin_api_key"])


def host_config(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    for i in range(1, 49):
        (model / f'model-{i:05d}-of-00048.safetensors').write_bytes(b'fixture')
    packed = tmp_path / 'packed'
    packed.mkdir()
    for layer in (1, 14):
        (packed / f'engram-l{layer}-r0of4.bin').write_bytes(b'fixture')
    keys = tmp_path / 'keys'
    keys.write_text('fixture-key\n')
    library = tmp_path / 'nccl.so'
    library.write_bytes(b'fixture-nccl')
    state = tmp_path / 'state'
    (state / 'operator').mkdir(parents=True)
    (state / 'operator/auth.py').write_text('# fixture')
    cfg = launch.read_config(environment(tmp_path))
    cfg.update(MODEL_HOST_PATH=str(model),
        ENGRAM_HOST_PATH=str(packed), API_KEY_FILE=str(keys), NCCL_SO_HOST_PATH=str(library),
        NCCL_SO_SHA256=launch.hashlib.sha256(library.read_bytes()).hexdigest(),
        STATE_HOST_PATH=str(state))
    return cfg


def test_catalog_preserves_sglang_defaults_and_runtime():
    from runtime.common.profiles import resolve
    from runtime.common.launch import plan
    resolved = resolve("deepseek-v41-flash-sglang-cycle")
    assert resolved["runtime"]["engine"] == "sglang"
    assert resolved["serving"]["max_model_len"] == 262144
    assert resolved["serving"]["expert_parallel_size"] == 4
    assert resolved["serving"]["max_num_seqs"] == 8
    assert resolved["serving"]["chunked_prefill_size"] == 4096
    command = plan("deepseek-v41-flash-sglang-cycle", ["--prepare", "/private/rank.env"])
    assert command["effect"] == "host-action"
    assert command["command"][1].endswith("deepseek_v41_sglang_cycle_serve.py")


@pytest.mark.parametrize("overrides", [
    {"CACHE_HOST_PATH": "/operator/model_host_path"},
    {"CACHE_HOST_PATH": "/operator/model_host_path/cache"},
    {"ENGRAM_HOST_PATH": "/operator/model_host_path/packed"},
    {"API_PORT": "20000"},
    {"API_PORT": "020000"},
    {"CACHE_HOST_PATH": "/operator/cache/../model_host_path"},
    {"MASTER_ADDR": "host:20000"},
    {"API_PORT": "\u00b2"},
])
def test_invalid_mounts_and_endpoints_fail_before_host_access(tmp_path, overrides):
    with pytest.raises(ValueError):
        launch.read_config(environment(tmp_path, **overrides))


def test_profile_table_keeps_both_deepseek_engines():
    from scripts.generate_profiles import profile_table
    table = profile_table(compact=True)
    rows = [line for line in table.splitlines() if line.startswith("| DeepSeek-V4.1-Flash |")]
    assert len(rows) == 2
    assert any("| SGLang |" in row and "| 262K /" in row for row in rows)
    assert any("| vLLM |" in row and "| 1M /" in row for row in rows)


def test_image_drift_rejected_before_host_actions(tmp_path, monkeypatch):
    cfg = host_config(tmp_path)
    monkeypatch.setattr(launch, 'output', lambda args: 'sha256:' + 'c' * 64)
    with pytest.raises(ValueError, match='image identity'):
        launch.verify_host(cfg)


def test_running_model_rejected_even_before_gpu_allocation(tmp_path, monkeypatch):
    cfg = host_config(tmp_path)
    def output(args):
        if args[:3] == ['docker', 'image', 'inspect']:
            return cfg['IMAGE_ID']
        if args[:2] == ['docker', 'ps']:
            return 'vllm_dsv41'
        pytest.fail('must refuse before additional host actions')
    monkeypatch.setattr(launch, 'output', output)
    with pytest.raises(ValueError, match='model container'):
        launch.verify_host(cfg)


def test_library_drift_rejected(tmp_path, monkeypatch):
    cfg = host_config(tmp_path)
    Path(cfg['NCCL_SO_HOST_PATH']).write_bytes(b'different-library')
    monkeypatch.setattr(launch, 'output', lambda args: cfg['IMAGE_ID'])
    with pytest.raises(ValueError, match='NCCL content identity'):
        launch.verify_host(cfg)
