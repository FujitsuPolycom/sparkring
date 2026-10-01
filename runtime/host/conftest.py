"""Shared fixtures of the host tests: no test enables or queries a systemd unit of the machine running it."""
import pytest


@pytest.fixture(autouse=True)
def no_recovery_timer(monkeypatch):
    """Replace automatic recovery's systemd calls; a root run on a Spark would otherwise enable its timer."""
    from runtime.host import recovery
    monkeypatch.setattr(recovery, "enable_timer", lambda **kwargs: False)
    monkeypatch.setattr(recovery, "timer_enabled", lambda **kwargs: None)
