"""Composition tests use bundled source fixtures; no Docker, Torch or GPU required."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

import compose


def test_composition_entrypoint_is_self_contained(tmp_path):
    output = tmp_path / 'composition'
    result = subprocess.run([sys.executable, str(Path(compose.__file__)), '--output-root', str(output)],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt['candidate_files']['vllm/v1/core/sched/scheduler.py'].startswith('75efa57e')
    assert receipt['candidate_files']['vllm/v1/core/single_type_kv_cache_manager.py'].startswith('d2e35b01')
    assert compose.verify_candidate(output / 'candidate') == receipt['candidate_files']
    with pytest.raises(FileExistsError):
        compose.compose(output)


def test_original_drift_rejected_before_output_created(tmp_path):
    output = tmp_path / 'prepared'
    compose.compose(output)
    source = output / 'original'
    (source / 'vllm/v1/core/sched/scheduler.py').write_bytes(b'not the attested source')
    rejected = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='Original source checksum'):
        compose.compose(rejected, source_root=source)
    assert not rejected.exists()


def test_candidate_drift_rejected(tmp_path):
    output = tmp_path / 'prepared'
    compose.compose(output)
    path = output / 'candidate/vllm/v1/core/sched/scheduler.py'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Candidate source checksum'):
        compose.verify_candidate(output / 'candidate')


def test_each_final_patch_remains_idempotent(tmp_path):
    output = tmp_path / 'prepared'
    compose.compose(output)
    candidate = output / 'candidate'
    assert not compose.barrier.apply_patch(candidate / 'b12x/attention/dsa_indexer/fused_indexer.py')['changed']
    assert not compose.partial.apply_patch(candidate / 'vllm/v1/core/sched/scheduler.py')['changed']
    # Earlier chain stages intentionally reject later scheduler postimages.
    with pytest.raises(RuntimeError, match='preimage'):
        compose.accounting.apply_patch(candidate / 'vllm/v1/core/sched/scheduler.py')


def test_actual_allocator_checker_accepts_final_source_composition(tmp_path):
    output = tmp_path / 'prepared'
    compose.compose(output)
    receipt = tmp_path / 'allocator.json'
    result = subprocess.run([sys.executable, str(Path(compose.__file__).with_name('check_mtp3_checkpoint_allocations.py')),
                             '--source-root', str(output / 'original/vllm'),
                             '--candidate-root', str(output / 'candidate/vllm'), '--output', str(receipt)],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    data = json.loads(receipt.read_text())
    assert data['passed']
    assert data['source_inputs']['scheduler']['sha256'] == compose.partial.AFTER_SHA256
    for name in ('fresh', 'resumed'):
        assert data['populations'][name]['summary']['cases'] == 336
        assert data['populations'][name]['summary']['stale_or_unwritten_registered_states'] == 0
