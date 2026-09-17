"""Guarded source changes and actual ZMQ wait behavior, without model imports."""
import ast
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

ROOT = Path(__file__).resolve().parent


def load_module(name):
    spec = importlib.util.spec_from_file_location('ipc_wait_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patcher = load_module('apply')
RECORDS = json.loads((ROOT / 'sources.json').read_text())['sources']


@pytest.fixture(params=RECORDS, ids=lambda record: record['name'])
def sources(request, tmp_path):
    record = request.param
    compressed = (ROOT / record['fixture']).read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == record['fixture_sha256']
    raw = gzip.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == record['source_sha256']
    changed, selected = patcher.patch_source(raw)
    assert selected == record
    before, after = tmp_path / 'before.py', tmp_path / 'after.py'
    before.write_bytes(raw)
    after.write_bytes(changed)
    return before, after, record


def test_reject_unknown_bytes():
    with pytest.raises(ValueError, match='not an admitted'):
        patcher.patch_source(b'class SpinCondition: pass\n')


def test_hashes_and_idempotence(sources):
    before, after, record = sources
    changed = after.read_bytes()
    assert hashlib.sha256(changed).hexdigest() == record['result_sha256']
    assert patcher.patch_source(changed)[0] == changed
    with pytest.raises(ValueError, match='not an admitted'):
        patcher.patch_source(before.read_bytes() + b'\n')


def test_cli_check_then_apply(sources):
    before, after, record = sources
    command = [sys.executable, str(ROOT / 'apply.py'), '--source', str(before)]
    original = before.read_bytes()
    subprocess.run(command + ['--check'], check=True, capture_output=True)
    assert before.read_bytes() == original
    receipt = before.with_name('patch-receipt.json')
    subprocess.run(command + ['--receipt', str(receipt)], check=True, capture_output=True)
    assert before.read_bytes() == after.read_bytes()
    assert json.loads(receipt.read_text())['result_sha256'] == record['result_sha256']
    rejected = subprocess.run(command + ['--receipt', str(receipt)], capture_output=True)
    assert rejected.returncode != 0
    assert b'Refusing to overwrite' in rejected.stderr


def test_only_constructor_changes(sources):
    before, after, _ = sources
    original, changed = ast.parse(before.read_bytes()), ast.parse(after.read_bytes())
    left = next(node for node in original.body if isinstance(node, ast.ClassDef) and node.name == 'SpinCondition')
    right = next(node for node in changed.body if isinstance(node, ast.ClassDef) and node.name == 'SpinCondition')
    # All notifications, timeout/recheck logic, cancellation and queue algorithms
    # remain byte-independent AST-equivalent; only constructor selection changes.
    left.body = [node for node in left.body if not isinstance(node, ast.FunctionDef) or node.name != '__init__']
    right.body = [node for node in right.body if not isinstance(node, ast.FunctionDef) or node.name != '__init__']
    assert ast.dump(original) == ast.dump(changed)


def test_constructor_imports_environment_dependency_itself(sources, monkeypatch, tmp_path):
    before, after, record = sources
    pytest.importorskip('zmq')
    probe = load_module('probe')
    original_globals = probe.load_condition(before).__init__.__globals__
    assert ('os' in original_globals) == (record['name'] == 'r37')
    local_import = b'                import os\n\n'
    assert after.read_bytes().count(local_import) == 1
    if record['name'] == 'deepseek':
        missing_import = tmp_path / 'missing-import.py'
        missing_import.write_bytes(after.read_bytes().replace(local_import, b''))
        monkeypatch.delenv(patcher.ENVIRONMENT, raising=False)
        with pytest.raises(NameError, match="os"):
            construct(missing_import, {})


def construct(source, kwargs):
    zmq = pytest.importorskip('zmq')
    probe = load_module('probe')
    ctx = zmq.Context()
    address = 'inproc://' + str(uuid.uuid4())
    cls = probe.load_condition(source)
    writer = cls(False, ctx, address)
    try:
        condition = cls(True, ctx, address, **kwargs)
    except Exception:
        writer.local_notify_socket.close(linger=0)
        ctx.term()
        raise
    return condition, writer, ctx


def close(pair):
    condition, writer, ctx = pair
    for socket in (condition.local_notify_socket, condition.read_cancel_socket,
                   condition.write_cancel_socket, writer.local_notify_socket):
        socket.close(linger=0)
    ctx.term()


@pytest.mark.parametrize('setting,expected', [(None, 1.0), ('0.002', 0.002), ('0', 0.0), ('1', 1.0)])
def test_default_and_optional_interval(sources, monkeypatch, setting, expected):
    _, after, _ = sources
    monkeypatch.delenv(patcher.ENVIRONMENT, raising=False)
    if setting is not None:
        monkeypatch.setenv(patcher.ENVIRONMENT, setting)
    pair = construct(after, {})
    try:
        assert pair[0].busy_loop_s == expected
        assert pair[1].busy_loop_s == 0
    finally:
        close(pair)


def test_environment_override_is_absent_in_original_source(sources, monkeypatch):
    before, after, _ = sources
    monkeypatch.setenv(patcher.ENVIRONMENT, '0.002')
    for source, expected in ((before, 1), (after, 0.002)):
        pair = construct(source, {})
        try:
            assert pair[0].busy_loop_s == expected
        finally:
            close(pair)


@pytest.mark.parametrize('setting', ['', 'NaN', 'inf', '-0.1', '1.1', 'false'])
def test_bad_environment_fails_before_reader_sockets(sources, monkeypatch, setting):
    _, after, _ = sources
    monkeypatch.setenv(patcher.ENVIRONMENT, setting)
    with pytest.raises(ValueError, match='finite and between 0 and 1'):
        construct(after, {})


@pytest.mark.parametrize('value', [0, 0.002, 1, 2])
def test_explicit_caller_override_ignores_environment(sources, monkeypatch, value):
    _, after, _ = sources
    monkeypatch.setenv(patcher.ENVIRONMENT, 'invalid')
    pair = construct(after, {'busy_loop_s': value})
    try:
        assert pair[0].busy_loop_s == value
    finally:
        close(pair)


@pytest.mark.skipif(not hasattr(os, 'sched_yield'), reason='POSIX scheduler-yield measurement')
def test_real_notify_idle_burst_and_cancel(sources):
    pytest.importorskip('zmq')
    _, after, _ = sources
    output = after.with_name('observations.json')
    subprocess.run([sys.executable, str(ROOT / 'probe.py'), '--source', str(after),
                    '--output', str(output), '--repeats', '1'],
                   check=True, capture_output=True, text=True, timeout=60)
    result = json.loads(output.read_text())
    for row in result['results']:
        assert row['received'] == row['sent']
        if row['case'] == 'idle' and row['busy_loop_s'] == 0.002:
            assert row['polls'] > 0
    assert all(value < 0.5 for value in result['cancellation_seconds'].values())


def test_park_timeout_still_returns_without_notification(sources):
    _, after, _ = sources
    pair = construct(after, {'busy_loop_s': 0.002})
    try:
        pair[0].last_read = time.monotonic() - 2
        started = time.monotonic()
        pair[0].wait(timeout_ms=20)
        assert 0.005 < time.monotonic() - started < 1
    finally:
        close(pair)
