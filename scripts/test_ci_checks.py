"""GPU-free checks for CI diagnostics and duplicate-heading links."""
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import check_markdown_links as markdown  # noqa: E402
import check_release_safety as safety  # noqa: E402


def test_duplicate_heading_anchors(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text("# Setup\n# Setup\n# Setup-1\n# Setup\n", encoding="utf-8")
    assert markdown.anchors(doc) == {"setup", "setup-1", "setup-1-1", "setup-2"}

def test_formatting_warns_for_list_touching_table():
    assert markdown.formatting_warnings("| value |\n* note\n* another\n") == [2]
    assert markdown.formatting_warnings("| value |\n\n* note\n") == []
    assert markdown.formatting_warnings("```\ntext\n* code\n```\n") == []


def test_fenced_headings_do_not_create_anchors(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text("# Visible\n```\n# Hidden\n```\n", encoding="utf-8")
    assert markdown.anchors(doc) == {"visible"}


def test_scanner_reports_location_without_secret(tmp_path, monkeypatch, capsys):
    secret = "synthetic" + "CredentialValue123"
    (tmp_path / "sample.txt").write_text("api_key=" + secret, encoding="utf-8")
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, "sample.txt\0", ""))
    assert safety.main(tmp_path) == 1
    output = capsys.readouterr().out
    assert secret not in output
    assert '"line": 1' in output
    assert "credential-assignment" in output


@pytest.mark.parametrize("error", [OSError("private detail"), subprocess.CalledProcessError(2, "git")])
def test_scanner_errors_fail_closed(tmp_path, monkeypatch, capsys, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(safety.subprocess, "run", fail)
    assert safety.main(tmp_path) == 2
    assert "private detail" not in capsys.readouterr().out


def test_scanner_missing_tracked_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, "missing.txt\0", ""))
    assert safety.main(tmp_path) == 2


def test_scanner_clean_file_passes(tmp_path, monkeypatch):
    (tmp_path / "sample.txt").write_text("No credentials here.", encoding="utf-8")
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, "sample.txt\0", ""))
    assert safety.main(tmp_path) == 0
