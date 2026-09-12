"""GPU-free checks for CI diagnostics and duplicate-heading links."""
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("payload", [b"\x80\x04binary", b"import hook\x00payload"])
def test_small_binary_pth_is_not_a_startup_hook(tmp_path, monkeypatch, payload):
    from scripts.check_repository_layout import validate_artifacts
    (tmp_path / "sample.pth").write_bytes(payload)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"sample.pth\0")
    with pytest.raises(ValueError, match="not a repository source"):
        validate_artifacts(tmp_path)


def test_small_text_pth_startup_hook_is_allowed(tmp_path, monkeypatch):
    from scripts.check_repository_layout import validate_artifacts
    (tmp_path / "sample.pth").write_bytes(b"import sparkring_transport\n")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: b"sample.pth\0")
    validate_artifacts(tmp_path)


def test_layout_check_protects_locked_markdown_asset(tmp_path):
    import hashlib
    import json
    from scripts.check_repository_layout import validate_locked_profile_assets

    asset = tmp_path / "runtime/example/README.md"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"# Published build input\n")
    lock = tmp_path / "runtime/sparkring/source_image/glm53-tp4-lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"profile_assets": {
        "runtime/example/README.md": {"sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}
    }}))
    assert validate_locked_profile_assets(tmp_path) == 1
    asset.write_bytes(b"# Edited documentation\n")
    with pytest.raises(ValueError, match="published profile asset changed"):
        validate_locked_profile_assets(tmp_path)

sys.path.insert(0, str(Path(__file__).parent))
import check_markdown_links as markdown  # noqa: E402
import check_release_safety as safety  # noqa: E402


def test_duplicate_heading_anchors(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text("# Setup\n# Setup\n# Setup-1\n# Setup\n", encoding="utf-8")
    assert markdown.anchors(doc) == {"setup", "setup-1", "setup-1-1", "setup-2"}


def test_links_cannot_pass_using_files_outside_repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("[Outside](../outside.md#available)\n")
    (tmp_path / "outside.md").write_text("# Available\n")
    monkeypatch.setattr(markdown.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, b"README.md\0", b""))
    monkeypatch.setattr(sys, "argv", ["check", str(root)])
    assert markdown.main() == 1


def test_link_error_escapes_annotation_control_characters(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(markdown.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, b"bad%0A\nname.md\0", b""))
    monkeypatch.setattr(markdown.Path, "read_text", lambda *a, **k: "[Missing](absent.md)\n")
    monkeypatch.setattr(sys, "argv", ["check", str(tmp_path)])
    assert markdown.main() == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("::error::bad%250A%0Aname.md:")

def test_formatting_warns_for_list_touching_table():
    assert markdown.formatting_warnings("| value |\n* note\n* another\n") == [2]
    assert markdown.formatting_warnings("| value |\n\n* note\n") == []
    assert markdown.formatting_warnings("```\ntext\n* code\n```\n") == []


def test_fenced_headings_do_not_create_anchors(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text("# Visible\n```\n# Hidden\n```\n", encoding="utf-8")
    assert markdown.anchors(doc) == {"visible"}


@pytest.mark.parametrize("inner", [chr(96) * 3, "~~~"])
def test_shorter_or_other_fences_remain_literal(tmp_path, monkeypatch, inner):
    fence = chr(96) * 4
    doc = tmp_path / "README.md"
    doc.write_text(f"# Visible\n{fence}markdown\n{inner}\n# Hidden\n[Example](missing.md)\n{fence}\n")
    assert markdown.anchors(doc) == {"visible"}
    monkeypatch.setattr(markdown.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, b"README.md\0", b""))
    monkeypatch.setattr(sys, "argv", ["check", str(tmp_path)])
    assert markdown.main() == 0


def test_scanner_reports_location_without_secret(tmp_path, monkeypatch, capsys):
    secret = "synthetic" + "CredentialValue123"
    (tmp_path / "sample.txt").write_text("api_key=" + secret, encoding="utf-8")
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, b"sample.txt\0", b""))
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
                        subprocess.CompletedProcess(a, 0, b"missing.txt\0", b""))
    assert safety.main(tmp_path) == 2


def test_scanner_clean_file_passes(tmp_path, monkeypatch):
    (tmp_path / "sample.txt").write_text("No credentials here.", encoding="utf-8")
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, b"sample.txt\0", b""))
    assert safety.main(tmp_path) == 0


@pytest.mark.parametrize("statement", ["from spark_transport import experiments", "from spark_transport import experiments as retained", "from .. import experiments"])
def test_import_boundary_rejects_alias_bypass(tmp_path, statement):
    from scripts.check_repository_layout import validate_imports
    source = tmp_path / "spark_transport/fabric/example.py"
    source.parent.mkdir(parents=True)
    source.write_text(statement + "\n")
    with pytest.raises(ValueError, match="maintained owner"):
        validate_imports(tmp_path)
    source.write_text("from spark_transport import fabric\n")
    assert validate_imports(tmp_path) == 1


def test_unicode_git_paths_and_encoded_accents(tmp_path, monkeypatch):
    name = "café.md"
    (tmp_path / name).write_text("## Café\n\n[Local](#caf%C3%A9)\n[Decomposed](#cafe%CC%81)\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("[Page](caf%C3%A9.md#caf%C3%A9)\n", encoding="utf-8")
    def tracked(*args, **kwargs):
        assert not kwargs.get("text")
        return subprocess.CompletedProcess(args, 0, (name + "\0README.md\0").encode("utf-8"), b"")
    monkeypatch.setattr(markdown.subprocess, "run", tracked)
    monkeypatch.setattr(sys, "argv", ["check", str(tmp_path)])
    assert markdown.main() == 0
    assert safety.main(tmp_path) == 0
