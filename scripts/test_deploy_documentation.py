"""Check the deployment walkthrough without contacting hosts or running examples."""

import os
from pathlib import Path
import re
import subprocess
import shutil

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


@pytest.mark.parametrize("relative", [
    "docs/operations/host-preparation.md", "docs/operations/pair-network.md",
    "docs/operations/bootstrap.md", "docs/GLM53_SPARK_MESH_HOST_SETUP.md",
    "profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md",
    "profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md",
    "profiles/qwen38-flash-next-tp2/README.md",
    "profiles/qwen38-flash-next-qad-tp4/README.md",
])
def test_setup_shell_examples_parse_without_execution(relative):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash required")
    text = (DOC.parents[2] / relative).read_text(encoding="utf-8")
    examples = re.findall(r"```bash\n(.*?)\n```", text, flags=re.S)
    assert examples
    result = subprocess.run([bash, "--noprofile", "--norc", "-n"],
                            input="\n".join(examples), text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
