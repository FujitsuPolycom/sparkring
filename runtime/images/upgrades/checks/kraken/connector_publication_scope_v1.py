"""Run unchanged publication cases against the request-scoped cache consumer."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

CASES = Path(__file__).resolve().parents[1] / 'connector_publication_retention.py'
assert hashlib.sha256(CASES.read_bytes()).hexdigest() == '3009e8528fa65d538f69e3d93ddcb3a41c65685fda16f8a13166e39208d86e1f'
spec = importlib.util.spec_from_file_location('_scoped_publication_cases_v1', CASES)
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)


def consumer_accepts(env, offers, target, required=(0,)):
    configured = os.environ.get('SPARKRING_SPARKCACHE_SOURCE_ROOT')
    if configured:
        path = Path(configured) / 'sparkcache/spark_context_cache_connector.py'
    else:
        module = importlib.util.find_spec('sparkcache')
        assert module and module.origin, 'Exact installed SparkCache source is required'
        path = Path(module.origin).with_name('spark_context_cache_connector.py')
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '0ccd5baff4076b9ac4880ec14a301e2a16cf61608928a8a25ed2e3bcff02f703'
    owner = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef)
                 and n.name == 'SparkContextCacheConnector')
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_validated_recurrent_boundary_blocks')
    # This method is identical in the released and request-scoped connectors.
    assert hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest() == '0113ab60e92ea1945cc3240da28fad49f8ac0a8ba90c7e31f8861cc410525e02'
    legacy.functions(path, {'_validated_recurrent_boundary_blocks'}, env,
                     'SparkContextCacheConnector')
    connector = legacy.NS(
        _recurrent_group_indexes=set(required),
        _group_topology=[{'reuse_policy': 'recurrent_align'}] * (max(required) + 1),
        _streaming_snapshots_enabled=False,
        _async_page_capture_enabled=True,
        _uses_capture_job_leases=lambda: True,
        counters=legacy.Counter(),
    )
    output = legacy.NS(kv_connector_block_state=legacy.NS(boundary_state_offloads={'r': offers}))
    return bool(env['_validated_recurrent_boundary_blocks'](connector, output, 'r', target))


# Reuse the hash-bound cases and their assertions, changing only the explicitly
# reviewed consumer binding. Each case calls this function through its globals.
legacy.consumer_accepts = consumer_accepts
for name, value in vars(legacy).items():
    if name.startswith('test_') and callable(value):
        globals()[name] = value
