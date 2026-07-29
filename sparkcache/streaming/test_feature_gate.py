from __future__ import annotations

import pytest

from sparkcache.streaming.feature_gate import (
    EXTRA_CONFIG_KEY,
    StreamingSnapshotsUnavailable,
    is_enabled,
    require_live_integration,
)


@pytest.mark.parametrize("value", (None, False, 0, "", "0", "false", " OFF "))
def test_streaming_snapshot_gate_is_disabled_for_default_and_false_values(value) -> None:
    # An omitted optional value cannot become an opt-in.
    assert not is_enabled(value)


@pytest.mark.parametrize("value", (True, 1, "1", "true", "YES"))
def test_streaming_snapshot_gate_recognizes_explicit_opt_in(value) -> None:
    assert is_enabled(value)


def test_invalid_streaming_snapshot_gate_value_is_rejected() -> None:
    with pytest.raises(ValueError, match=EXTRA_CONFIG_KEY):
        is_enabled("maybe")


def test_legacy_direct_enable_still_fails_closed() -> None:
    with pytest.raises(StreamingSnapshotsUnavailable, match="factory"):
        require_live_integration()
