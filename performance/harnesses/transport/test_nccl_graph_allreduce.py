from __future__ import annotations

import pytest
import sys
from types import SimpleNamespace

from . import nccl_graph_allreduce as probe


def test_pynccl_constructor_failure_destroys_process_group(monkeypatch, tmp_path):
    calls = []
    dist = SimpleNamespace(init_process_group=lambda **kwargs: calls.append('init'),
                           destroy_process_group=lambda: calls.append('destroy'),
                           group=SimpleNamespace(WORLD=None))
    torch = SimpleNamespace(cuda=SimpleNamespace(set_device=lambda _: None),
                            device=lambda _: None, distributed=dist)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.distributed', dist)
    def fail(**kwargs):
        raise RuntimeError('synthetic communicator failure')
    monkeypatch.setitem(sys.modules, 'vllm.distributed.device_communicators.pynccl',
                        SimpleNamespace(PyNcclCommunicator=fail))
    monkeypatch.setattr(sys, 'argv', ['probe', '--head-ip', '192.0.2.1', '--implementation', 'pynccl',
                                    '--nccl-library', '/synthetic.so', '--output', str(tmp_path / 'receipt.json')])
    with pytest.raises(RuntimeError, match='synthetic communicator'):
        probe.main()
    assert calls == ['init', 'destroy']


def test_existing_receipt_is_refused_before_device_import(monkeypatch, tmp_path):
    output = tmp_path / 'receipt.json'
    output.write_text('preserved')
    monkeypatch.setitem(sys.modules, 'torch', None)
    monkeypatch.setattr(sys, 'argv', ['probe', '--head-ip', '192.0.2.1', '--output', str(output)])
    with pytest.raises(SystemExit, match='output'):
        probe.main()
    assert output.read_text() == 'preserved'


def test_parse_query_rows_accepts_sorted_unique_positive_values() -> None:
    assert probe.parse_query_rows("8,16,32,64,128") == (8, 16, 32, 64, 128)


@pytest.mark.parametrize("value", ["", "0", "8,8", "16,8", "8,nope"])
def test_parse_query_rows_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="query rows"):
        probe.parse_query_rows(value)


def test_nearest_rank_percentile_returns_observed_samples() -> None:
    samples = [7.0, 1.0, 5.0, 3.0]
    assert probe.nearest_rank(samples, 0.50) == 3.0
    assert probe.nearest_rank(samples, 0.95) == 7.0


@pytest.mark.parametrize('sample', [float('nan'), float('inf'), -1.0, True])
def test_invalid_latency_cannot_be_reported_as_measurement(sample):
    with pytest.raises(ValueError, match='finite'):
        probe.summarize([sample])


@pytest.mark.parametrize("fraction", [-0.1, 0.0, 1.1])
def test_nearest_rank_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        probe.nearest_rank([1.0], fraction)
