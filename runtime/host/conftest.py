"""Shared fixtures of the host tests: no test enables or queries a systemd unit of the machine running it,
no setup simulation runs the fabric bandwidth test on the machine running it or over SSH, and no
installation drops page caches or reads memory on a Spark before its start."""
import pytest


@pytest.fixture(autouse=True)
def no_recovery_timer(monkeypatch):
    """Replace automatic recovery's systemd calls; a root run on a Spark would otherwise enable its timer."""
    from runtime.host import recovery
    monkeypatch.setattr(recovery, "enable_timer", lambda **kwargs: False)
    monkeypatch.setattr(recovery, "timer_enabled", lambda **kwargs: None)


@pytest.fixture(autouse=True)
def no_bandwidth_test(monkeypatch):
    """Replace setup's final bandwidth check, which would start ib_write_bw locally and over SSH.

    ``runtime.host.test_fabric_bandwidth`` tests the check through ``REAL_AFTER_SETUP``.
    """
    from runtime.host import fabric_bandwidth
    monkeypatch.setattr(fabric_bandwidth, "after_setup", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def no_memory_settle(monkeypatch):
    """Replace the page-cache drop and memory settle before each start, which would sign in to every Spark of
    the deployment over SSH. ``runtime.host.test_memory_settle`` tests it through ``REAL_SETTLE_MEMORY``."""
    from runtime.host import install_workflow
    monkeypatch.setattr(install_workflow, "settle_memory", lambda directory, **kwargs: [])
