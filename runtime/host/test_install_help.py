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
