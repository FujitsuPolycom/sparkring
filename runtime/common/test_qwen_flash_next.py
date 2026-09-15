"""Offline admission checks for the isolated Qwen Flash-Next TP2 plan."""

import json
from pathlib import Path
import pytest
from runtime.common.qwen_flash_next import ROOT, render, publication
from runtime.common import qwen_flash_next as adapter

PROFILE = (
    Path(__file__).resolve().parents[2] / "profiles/qwen38-flash-next-tp2/config.json"
)


def plan(rank=0):
    return render(
        json.loads(PROFILE.read_text()),
        rank=rank,
        master="192.0.2.1",
        host_ip=f"192.0.2.{rank + 1}",
        interface="test0",
        image=publication()["image_id"],
        model=str(ROOT / "fixture-model"),
        cache=str(ROOT / "fixture-cache"),
    )


def test_qwen_model_and_native_prefix_only():
    command = plan()
    assert (
        command[command.index("--served-model-name") + 1] == "Qwen3.8-Flash-Next-NVFP4"
    )
    assert "--kv-transfer-config" not in command
    assert "--device-ids" not in command
    assert "VLLM_SSM_CONV_STATE_LAYOUT=DS" in command
    assert "VLLM_USE_V2_MODEL_RUNNER=1" in command
    assert not any(value.startswith("VLLM_PLE_TABLE_MEMORY=") for value in command)
    assert "VLLM_PLE_CPU_OFFLOAD=0" in command
    assert "VLLM_MXFP8_LM_HEAD=0" in command
    assert (
        json.loads(command[command.index("--speculative-config") + 1])["moe_backend"]
        == "b12x"
    )
    assert command[command.index("--decode-context-parallel-size") + 1] == "1"


def test_direct_loader_avoids_fastsafetensors_staging():
    command = plan()
    assert command[command.index("--load-format") + 1] == "b12x"
    assert "VLLM_PLUGINS=b12x_loader" in command
    assert not any(value.startswith("SAFETENSORS_FAST_GPU=") for value in command)


def test_reciprocal_selected_hca_positions():
    for rank in (0, 1):
        command = plan(rank)
        assert f"B12X_ROCE_PEER_HCA_MAP={1 - rank}=0/1" in command
        assert "B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0" in command
        assert ("--headless" in command) == bool(rank)


def test_invalid_rank_rejected():
    with pytest.raises(ValueError, match="rank0/1"):
        plan(2)


def test_expanded_capacity_preserves_native_context():
    command = plan()
    assert command[command.index('--max-model-len') + 1] == '262144'
    assert command[command.index('--max-num-seqs') + 1] == '16'
    assert command[command.index('--max-num-batched-tokens') + 1] == '8192'
    assert command[command.index('--kv-cache-memory-bytes') + 1] == '25769803776'
    assert '--hf-overrides' not in command
    assert not any(value.startswith('VLLM_ALLOW_LONG_MAX_MODEL_LEN=') for value in command)
    graphs = json.loads(command[command.index('--compilation-config') + 1])
    assert all(4 * concurrency in graphs['cudagraph_capture_sizes'] for concurrency in range(1, 17))


def options():
    return dict(rank=0, master='192.0.2.1', host_ip='192.0.2.1', interface='test0',
                image=publication()['image_id'], model=str(ROOT / 'fixture-model'),
                cache=str(ROOT / 'fixture-cache'))


def test_unregistered_image_rejected():
    values = options()
    values['image'] = 'sha256:' + 'a' * 64
    with pytest.raises(ValueError, match='registered R37'):
        render(json.loads(PROFILE.read_text()), **values)


def test_profile_arguments_and_environment_cannot_bypass_canonical():
    for location in ('vllm_args', 'environment'):
        profile = json.loads(PROFILE.read_text())
        if location == 'vllm_args':
            profile[location].append('--kv-transfer-config={}')
        else:
            profile[location]['SPARKCACHE_ENABLED'] = '1'
        with pytest.raises(ValueError, match='canonical'):
            render(profile, **options())


@pytest.mark.parametrize('key,value', [('model', 'relative'), ('cache', '/tmp/cache,readonly'),
                                     ('interface', 'eth0,eth1'), ('host_ip', 'placeholder'),
                                     ('master', 'host --flag')])
def test_invalid_site_inputs(key, value):
    values = options()
    values[key] = value
    with pytest.raises(ValueError):
        render(json.loads(PROFILE.read_text()), **values)


def test_mount_overlap_rejected():
    values = options()
    values['cache'] = str(Path(values['model']) / 'cache')
    with pytest.raises(ValueError, match='disjoint'):
        render(json.loads(PROFILE.read_text()), **values)


def test_explicit_entrypoint_and_compile_identity():
    command = plan()
    assert command[command.index('--entrypoint') + 1] == '/opt/venv/bin/python'
    assert adapter.candidate.ENTRYPOINT in command
    namespace = f"qwen-flash-next-{publication()['image_id'][7:19]}-ada4da32a583"
    assert f'VLLM_CACHE_ROOT=/cache/{namespace}/vllm' in command
    assert 'VLLM_SPARK_TP4_MODE=' in command
    assert 'VLLM_SPARK_TP4_VOCAB_MODE=' in command


def test_image_check_binds_receipt_without_glm_admission(monkeypatch):
    from types import SimpleNamespace
    image = publication()['image_id']
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ['image', 'inspect']:
            return SimpleNamespace(stdout=json.dumps([{'Id':image,'Os':'linux','Architecture':'arm64',
                'Config':{'Entrypoint':['/opt/venv/bin/python',adapter.candidate.ENTRYPOINT]}}]))
        if '/bin/cat' in command:
            return SimpleNamespace(stdout=b'{"fixture":"raw receipt bytes"}\n')
        assert command[-1] == 'verify'
        return SimpleNamespace(stdout='{"fixture":"verification"}')
    def make_receipt(actual, raw, verification):
        assert actual == image and isinstance(raw, bytes)
        assert verification == {'fixture':'verification'}
        return {'admitted':True}
    monkeypatch.setattr(adapter.candidate, 'make_receipt', make_receipt)
    assert adapter.verify_image(image, run=run) == {'admitted':True}
    assert len(calls) == 3
    assert all('create' not in command and 'start' not in command for command in calls)


def test_wrong_architecture_rejected_before_verifier():
    from types import SimpleNamespace
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps([{'Id':publication()['image_id'],'Os':'linux','Architecture':'amd64'}]))
    with pytest.raises(ValueError, match='platform'):
        adapter.verify_image(publication()['image_id'], run=run)
    assert len(calls) == 1


def test_capacity_overrides_rejected():
    for flag, value in (("--max-model-len", "65536"), ("--max-num-seqs", "8"),
                        ("--max-num-batched-tokens", "4096"), ("--kv-cache-memory-bytes", "12884901888")):
        profile = json.loads(PROFILE.read_text())
        profile["vllm_args"][profile["vllm_args"].index(flag) + 1] = value
        with pytest.raises(ValueError, match="canonical"):
            render(profile, **options())


def test_catalog_exposes_native_capacity_and_no_overrides():
    from runtime.common.profiles import load, resolve
    profile, _ = load('qwen38-flash-next-tp2')
    resolved = resolve('qwen38-flash-next-tp2')
    assert profile['overrides'] == []
    assert resolved['status'] == 'research-only'
    assert resolved['topology'] == 'direct-pair-2'
    expected = {'tensor_parallel_size': 2, 'decode_context_parallel_size': 1,
                'max_model_len': 262144, 'max_num_seqs': 16,
                'max_num_batched_tokens': 8192, 'kv_cache_memory_bytes': 25769803776}
    assert all(resolved['serving'][key] == value for key, value in expected.items())


def test_evidence_uses_one_native_context_profile():
    root = Path(__file__).resolve().parents[2]
    evidence = json.loads((root / 'performance/records/qwen38-flash-next/r37-tp2.json').read_text())
    assert 'baseline' not in evidence
    assert evidence['checks']['c16_exact_json_passed'] == 16
    assert evidence['checks']['long_context']['prompt_tokens'] == 257504
    assert evidence['profile_defaults']['kv_cache_memory_bytes'] == 25769803776
    assert evidence['measurements']['profile_max_model_len'] == 262144
    assert adapter.CONFIG_NAMES == ('config.json', 'sparkcache.json')
    for name in adapter.CONFIG_NAMES:
        config = adapter.read(adapter.CONFIG_ROOT / name)
        assert config['vllm_args'][config['vllm_args'].index('--max-model-len') + 1] == '262144'


def test_sparkcache_requires_extension_and_preserves_capacity():
    profile = adapter.read(adapter.CONFIG_ROOT / 'sparkcache.json')
    values = options()
    with pytest.raises(ValueError, match='cache-extension'):
        adapter.render(profile, **values)
    values['image'] = 'sha256:' + 'a' * 64
    command = adapter.render(profile, **values)
    assert command[command.index('--name') + 1] == 'qwen-flash-next-sparkcache-tp2-r0'
    assert command[command.index('--kv-cache-memory-bytes') + 1] == '25769803776'
    connector = json.loads(command[command.index('--kv-transfer-config') + 1])
    assert connector['kv_load_failure_policy'] == 'recompute'
    assert connector['kv_connector_extra_config']['spark_cache_model_profile'] == 'qwen38-flash-next-hybrid'


def test_media_defaults_keep_fixed_capacity():
    for rank in (0, 1):
        command = plan(rank)
        assert json.loads(command[command.index('--limit-mm-per-prompt') + 1]) == {'image': 3, 'video': 1}
        assert json.loads(command[command.index('--media-io-kwargs') + 1]) == {'video': {'num_frames': 16}}
        for flag, value in (('--max-model-len', '262144'), ('--max-num-seqs', '16'),
                            ('--max-num-batched-tokens', '8192'), ('--kv-cache-memory-bytes', '25769803776')):
            assert command[command.index(flag) + 1] == value


def test_media_evidence_does_not_claim_strict_combined_json():
    path = Path(__file__).resolve().parents[2] / 'performance/records/qwen38-flash-next/r37-tp2.json'
    evidence = json.loads(path.read_text())['media_validation']
    assert evidence['concurrency'] == 1
    assert all(row['semantic_pass'] and row['http_status'] == 200 for row in evidence['results'])
    combined = next(row for row in evidence['results'] if row['test'] == 'combined')
    assert combined['strict_json_response'] is False
