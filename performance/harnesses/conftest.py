"""Shared fixture of the harness tests: no test holds the quiet mode of a dashboard on the machine
running it (performance/harnesses/dashboard_quiet.py, SKIP), not even through a harness it starts
as a child process. ``test_dashboard_quiet`` removes the setting to test holding itself."""
import pytest


@pytest.fixture(autouse=True)
def no_dashboard_quiet_hold(monkeypatch):
    from performance.harnesses import dashboard_quiet
    monkeypatch.setenv(dashboard_quiet.SKIP, "skip")
    monkeypatch.delenv(dashboard_quiet.INHERITED, raising=False)
