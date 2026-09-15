"""Compose manifest-verified Python fixtures for the CPU test session.

Exports MTP3 source paths and restores prior environment values at teardown.
The experiment import directory supports the tests' bare helper imports.
"""
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
_ADDED_IMPORT_PATH = str(HERE) not in sys.path
if _ADDED_IMPORT_PATH:
    sys.path.insert(0, str(HERE))
import compose  # noqa: E402


def pytest_configure(config):
    temporary = tempfile.TemporaryDirectory(prefix='mtp3-cache-reuse-tests-')
    config._mtp3_cache_reuse_temporary = temporary
    output = Path(temporary.name) / 'composition'
    compose.compose(output)
    values = {'MTP3_ORIGINAL_SOURCE': str(output / 'original/vllm'),
              'MTP3_RETENTION_SOURCE': str(output / 'candidate/vllm'),
              'MTP3_SCHEDULER_SOURCE': str(output / 'candidate/vllm/v1/core/sched/scheduler.py')}
    config._mtp3_cache_reuse_environment = {key: os.environ.get(key) for key in values}
    os.environ.update(values)


def pytest_unconfigure(config):
    for key, value in getattr(config, '_mtp3_cache_reuse_environment', {}).items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    temporary = getattr(config, '_mtp3_cache_reuse_temporary', None)
    if temporary:
        temporary.cleanup()
    if _ADDED_IMPORT_PATH and str(HERE) in sys.path:
        sys.path.remove(str(HERE))
