"""The vLLM log adapter uses connector formatting and preserves other connectors."""

from types import SimpleNamespace

import pytest

from runtime.images import sparkcache_metrics as metrics


def fixture_source():
    return ("def report(self, log_fn):\n    if True:\n        if True:\n" + metrics.ORIGINAL).encode()


def test_formatter_can_be_silent_without_reducing_or_mutating_stats():
    patched = metrics.transform(fixture_source())
    assert metrics.transform(patched) == patched
    namespace = {}
    exec(patched, namespace)
    lines = []
    state = SimpleNamespace(transfer_stats_accumulator=SimpleNamespace(format_log_lines=lambda: ()))
    namespace["report"](state, lambda *args: lines.append(args))
    assert lines == []
    state.transfer_stats_accumulator.format_log_lines = lambda: ("sparkcache: counters counter_restore_hit=1",)
    namespace["report"](state, lambda *args: lines.append(args))
    assert lines == [("%s", "sparkcache: counters counter_restore_hit=1")]
    state.transfer_stats_accumulator = SimpleNamespace(reduce=lambda: {"other": 2})
    namespace["report"](state, lambda *args: lines.append(args))
    assert lines[-1] == ("KV Transfer metrics: %s", "other=2")


def test_unknown_logger_is_rejected():
    with pytest.raises(ValueError, match="logger block differs"):
        metrics.transform(b"unrecognized logger")
