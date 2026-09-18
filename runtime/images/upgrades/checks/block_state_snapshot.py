"""Optional cache publication observes the block tables from its scheduled step."""

import ast
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def factory():
    root = Path(os.environ['SPARKRING_TEST_SOURCE_ROOT'])
    cls = next(n for n in ast.parse((root / 'vllm/v1/core/sched/output.py').read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'KVConnectorBlockState')
    fn = next(n for n in ast.parse((root / 'vllm/v1/core/sched/scheduler.py').read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == '_build_kv_connector_block_state')
    scope = dict(dataclass=dataclass, field=field, Callable=Callable, Iterable=Iterable,
                 KVCacheConfig=object, KVCacheManager=object, MambaSpec=NS)
    exec(compile(ast.Module(body=[cls, fn], type_ignores=[]), '<selected block-state methods>', 'exec'), scope)
    return scope['_build_kv_connector_block_state']


def test_scheduled_snapshot_retains_mapping_and_offers_after_allocator_advances():
    build = factory()
    tables = {'request': ([11, 12], [31])}
    calls = []

    def resolve(req):
        calls.append(req)
        return tuple(list(group) for group in tables[req])

    offers = {'request': [(1, 31, 512)]}
    state = build(NS(kv_cache_groups=[]), NS(get_block_ids=resolve), ['request'], offers, None)
    tables['request'][0][:] = [81]
    offers['request'].clear()
    assert isinstance(state.block_ids, Mapping)
    assert state.block_ids == {'request': ([11, 12], [31])}
    assert state.boundary_state_offloads == {'request': [(1, 31, 512)]}
    assert calls == ['request']
    if hasattr(state, 'get_block_ids'):
        assert state.get_block_ids('request') == state.block_ids['request']
        assert state.get_block_ids('not-offered') is None
        assert calls == ['request']


def test_nondivisible_recurrent_retention_is_rejected_before_resolving():
    def resolve(req):
        raise AssertionError('No table may be resolved before validation')

    with pytest.raises(ValueError, match='divisible'):
        factory()(NS(kv_cache_groups=[NS(kv_cache_spec=NS(block_size=2048))]),
                  NS(get_block_ids=resolve), ['request'], {}, 3072)
