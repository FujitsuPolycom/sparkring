"""Check the deployment walkthrough without contacting hosts or running examples."""

import os
from pathlib import Path
import re
import subprocess

import pytest

from scripts.deploy_suite import main


DOC = Path(__file__).resolve().parents[1] / "docs/operations/deployment-suite.md"


@pytest.mark.parametrize(
    "command",
    [
        "discover",
        "plan",
        "network-plan",
        "network-check",
        "stage",
        "runtime-plan",
        "apply-plan",
    ],
)
def test_documented_commands_have_offline_help(command, capsys):
    with pytest.raises(SystemExit) as result:
        main([command, "--help"])
    assert result.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert command in DOC.read_text()


@pytest.mark.skipif(os.name != "posix", reason="walkthrough uses Bash on Linux/WSL")
def test_documented_shell_examples_parse_without_execution():
    examples = re.findall(r"```bash\n(.*?)\n```", DOC.read_text(), flags=re.S)
    assert len(examples) >= 6
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-n"],
        input="\n".join(examples),
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
