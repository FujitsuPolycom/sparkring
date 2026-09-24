import sys

from runtime.host import progress


def test_install_log_is_flushed_and_tailable_before_command_finishes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    with progress.run("setup"):
        with progress.step("Node A: verify existing fabric"):
            assert "Node A: verify existing fabric" in (tmp_path / "install.log").read_text()
        print("Existing checkpoint will be reused")
    text = (tmp_path / "install.log").read_text()
    assert "Done: Node A" in text and "checkpoint will be reused" in text
    assert "logs --follow" in capsys.readouterr().out


def test_verbose_child_output_is_separate_from_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    with progress.run("setup"):
        progress.command([sys.executable, "-c", "print('verbose fixture output')"], title="Prepare worker tools", check=True)
    assert "verbose fixture output" not in (tmp_path / "install.log").read_text()
    assert "verbose fixture output" in (tmp_path / "install-details.log").read_text()


def test_failed_result_never_reports_done(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path))
    with progress.run("up"):
        with progress.step("Node B: verify checkpoint") as state:
            state["failed"] = True
            progress.failure("trace details\nCheckpoint checksum differs")
    text = (tmp_path / "install.log").read_text()
    assert "Stopped: Node B" in text and "Done: Node B" not in text
    assert "trace details" not in text
    assert "trace details" in (tmp_path / "install-details.log").read_text()
