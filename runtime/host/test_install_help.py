"""`sparkring install --help` and `sparkring up --help` print every serving setting's help as written.

argparse formats help strings with the % operator, so a bare % in a help
string either raises or, as "% s" does, prints the parser's internal
dictionary in place of the text.
"""
import pytest

from runtime.common import serving
from runtime.host import controller, install_workflow


def compact(text):
    """``text`` without whitespace, so line wrapping and hyphen breaks do not matter."""
    return "".join(text.split())


def printed_help(capsys, command, argv):
    with pytest.raises(SystemExit) as stop:
        command(argv)
    assert stop.value.code == 0
    return capsys.readouterr().out


@pytest.mark.parametrize("command, argv", [
    (install_workflow.main, ["--help"]),
    (controller.lifecycle, ["up", "--help"]),
], ids=["install", "up"])
def test_help_prints_each_serving_setting_as_written(capsys, command, argv):
    text = printed_help(capsys, command, argv)
    assert "'container':" not in text and "'prog':" not in text
    for name, row in (*serving.SETTINGS.items(), *serving.SWITCHES.items()):
        assert serving.option(name) in text
        assert compact(row[-1]) in compact(text), name


@pytest.mark.parametrize("operation", ["up", "down", "restart"])
@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_lifecycle_help_never_refreshes_or_writes_state(operation, flag, tmp_path, monkeypatch, capsys):
    from runtime.host import dashboard_admin
    from scripts import sparkring
    monkeypatch.delenv("SPARKRING_HELPER_SOCKET", raising=False)
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(controller, "STATE", tmp_path / "controller")
    writes = []

    def refresh():
        writes.append(True)
        for path in (tmp_path / "dashboard/site.json", controller.STATE / "publication.lock"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unexpected state write")
    monkeypatch.setattr(dashboard_admin, "refresh_site", refresh)
    text = printed_help(capsys, sparkring.main, [operation, "glm53", flag])
    assert "usage:" in text
    assert not writes
    assert list(tmp_path.rglob("*")) == []
