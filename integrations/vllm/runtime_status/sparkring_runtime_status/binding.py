"""Optional installer assertions and pure identity/freshness joins.

Bindings are supplied by the installer, not attestation. A matched report
establishes agreement at the recorded times, not model or fabric qualification.
No hostname/rank-only fallback, host action, HTTP request or GPU query is made.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import re
import time
import uuid

from .collector import MISSING, fact
from .resources import read_bounded

BINDING_PATH = '/run/sparkring/runtime-binding.json'
FIELDS = ('deployment_id', 'node_id', 'container_id', 'image_id')
_configured = False
_binding = None
_boot = None
_reason = 'binding_not_configured'


def valid(field, value):
    if field in ('node_id', 'boot_id'):
        try:
            return type(value) is str and str(uuid.UUID(value)) == value
        except (ValueError, AttributeError):
            return False
    if field == 'rank':
        return type(value) is int and 0 <= value < 256
    pattern = r'sha256:[0-9a-f]{64}' if field == 'image_id' else r'[0-9a-f]{64}'
    return type(value) is str and re.fullmatch(pattern, value) is not None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate identity field')
        result[key] = value
    return result


def configure(environ, *, proc_root=Path('/proc'), read=read_bounded):
    """Read the fixed binding path once during plugin registration, before serving."""
    global _configured, _binding, _boot, _reason
    if _configured:
        return
    _configured = True
    _binding = None
    boot = read(proc_root / 'sys/kernel/random/boot_id', 64)
    _boot = boot.strip() if boot and valid('boot_id', boot.strip()) else None
    filename = environ.get('SPARKRING_RUNTIME_BINDING')
    _reason = 'binding_not_configured' if not filename else 'binding_invalid_or_unavailable'
    if filename != BINDING_PATH:
        return
    try:
        raw = read(Path(filename), 16384)
        value = json.loads(raw, object_pairs_hook=_unique) if raw else None
        if (type(value) is not dict or set(value) != {'schema', 'rank', *FIELDS}
                or value['schema'] != 'sparkring-runtime-binding/v1'
                or not all(valid(key, value[key]) for key in (*FIELDS, 'rank'))):
            return
        _binding = value
    except (ValueError, TypeError, RecursionError):
        pass


def worker_identity(rank):
    """Return cached assertions only when their rank matches the resident worker."""
    matches = _binding is not None and _binding['rank'] == rank
    result = {key: fact(_binding[key] if matches else MISSING,
                        source='installer_runtime_binding',
                        reason=None if matches else 'binding_rank_mismatch' if _binding else _reason)
              for key in FIELDS}
    result['boot_id'] = fact(_boot if _boot else MISSING, source='linux_proc_boot_id',
                             reason=None if _boot else 'boot_id_unavailable')
    return result


def _value(record, key):
    item = record.get('identity', {}).get(key, {})
    return item.get('value') if type(item) is dict and item.get('state') == 'known' else None


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def join(expected, host, model, runtime, *, now=None, host_max_age=90,
         model_max_age=90, worker_max_age=10):
    """Join one expected rank to independent host, Docker and runtime observations.

    Caller-provided observations must come from their authenticated producers.
    Freshness is evaluated independently; reception time cannot refresh a
    cached sample. The result never authorizes lifecycle actions.
    """
    def result(state, reason):
        return {'schema': 'sparkring-status-binding/v1', 'state': state, 'reason': reason,
                'scope': 'identity_and_freshness_only', 'model_ready': None,
                'hardware_qualified': False}

    now = time.time() if now is None else now
    if (not _finite(now)
            or any(not _finite(age) or age <= 0
                   for age in (host_max_age, model_max_age, worker_max_age))):
        return result('unbound', 'invalid_freshness_policy')
    if not all(type(value) is dict for value in (expected, host, model, runtime)):
        return result('unbound', 'invalid_observation_shape')
    for key in ('deployment_id', 'node_id', 'image_id', 'rank'):
        if not valid(key, expected.get(key)):
            return result('unbound', 'expected_identity_incomplete')
    if (host.get('schema') != 'sparkring-node-status/v1' or host.get('source') != 'host-agent'
            or model.get('schema') != 'sparkring-model-observation/v1'
            or model.get('source') != 'installer-docker-inspect'):
        return result('unbound', 'producer_schema_or_source_missing')
    workers = runtime.get('workers')
    if type(workers) is not dict or type(workers.get('ranks')) is not list or len(workers['ranks']) > 256:
        return result('unbound', 'worker_reports_unavailable')
    if host.get('state') == 'stale' or workers.get('stale') is True:
        return result('stale', 'producer_marks_observation_stale')
    if workers.get('state') != 'complete':
        return result('unbound', 'worker_collection_incomplete')
    rows = workers['ranks']
    if any(type(row) is not dict or type(row.get('identity')) is not dict for row in rows):
        return result('unbound', 'worker_identity_invalid')
    ranks = [_value(row, 'rank') for row in rows]
    if not all(valid('rank', rank) for rank in ranks) or len(set(ranks)) != len(ranks):
        return result('unbound', 'worker_rank_identity_ambiguous')
    selected = [row for row in rows if _value(row, 'rank') == expected['rank']]
    if len(selected) != 1 or selected[0].get('error'):
        return result('unbound', 'expected_worker_not_reported')
    worker = selected[0]
    if worker.get('schema') != 'sparkring-worker-status/v1':
        return result('unbound', 'worker_schema_unrecognized')
    for key in ('node_id', 'boot_id'):
        if not valid(key, host.get(key)) or not valid(key, model.get(key)):
            return result('unbound', 'host_identity_incomplete')
    for key in (*FIELDS, 'rank'):
        if not valid(key, model.get(key)) or not valid(key, _value(worker, key)):
            return result('unbound', 'container_or_runtime_identity_incomplete')
    if not valid('boot_id', _value(worker, 'boot_id')) or type(_value(worker, 'process_id')) is not int or _value(worker, 'process_id') <= 0:
        return result('unbound', 'worker_process_identity_incomplete')
    if host.get('identity_errors') or model.get('identity_errors'):
        return result('unbound', 'producer_identity_error')
    if model.get('node_identity_matches') is None:
        return result('unbound', 'node_comparison_unavailable')
    if model.get('node_identity_matches') is not True or model.get('present') is not True or model.get('running') is not True:
        return result('mismatch', 'container_or_node_not_matching')
    for key in ('deployment_id', 'node_id', 'image_id', 'rank'):
        if model[key] != expected[key] or _value(worker, key) != expected[key]:
            return result('mismatch', key + '_differs')
    if host['node_id'] != expected['node_id']:
        return result('mismatch', 'host_node_differs')
    if model.get('expected_node_id') != expected['node_id'] or model.get('expected_image_id') != expected['image_id']:
        return result('mismatch', 'docker_expectations_differ')
    if _value(worker, 'container_id') != model['container_id']:
        return result('mismatch', 'container_id_differs')
    if host['boot_id'] != model['boot_id'] or host['boot_id'] != _value(worker, 'boot_id'):
        return result('mismatch', 'boot_id_differs')
    ns = worker.get('collected_at_unix_ns')
    if type(ns) is not int or not 0 < ns <= 2**63 - 1:
        return result('unbound', 'worker_timestamp_unavailable')
    worker_time = ns / 1e9
    for label, stamp, limit in (('host', host.get('observed_at'), host_max_age),
                                ('model', model.get('observed_at'), model_max_age),
                                ('worker', worker_time, worker_max_age)):
        if not _finite(stamp):
            return result('unbound', label + '_timestamp_unavailable')
        if stamp > now + 5 or now - stamp > limit:
            return result('stale', label + '_timestamp_outside_window')
    try:
        started = datetime.fromisoformat(model['container_started_at'].replace('Z', '+00:00'))
        if started.tzinfo is None:
            raise ValueError('Timestamp needs timezone')
        started = started.timestamp()
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        return result('unbound', 'container_start_time_unavailable')
    if worker_time < started or model['observed_at'] < started:
        return result('stale', 'observation_predates_container_start')
    return result('matched', 'identities_and_independent_timestamps_match')
