"""The pins: the image's and SIRCL's files, the package's modules, and refusals of anything else."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import glm_dcp_decode_comm as pkg

PROJECT = Path(__file__).resolve().parents[1]


def test_the_package_modules_match_their_pins():
    pkg.verify_package()
    assert set(pkg.PACKAGE_SHA256) == {path.name for path in (PROJECT / "glm_dcp_decode_comm").glob("*.py")
                                       if path.name != "__init__.py"}


def test_every_image_pin_matches_the_image(vllm_root, b12x_root):
    roots = {"vllm": vllm_root, "b12x": b12x_root}
    checks = [check for check in pkg.FILE_CHECKS if check.package in roots]
    assert len(checks) == 13
    for check in checks:
        pkg.verify_file(check, roots)
    assert pkg.digest(vllm_root / pkg.ATTENTION_PATH) == pkg.ATTENTION_SHA256


def test_every_sircl_pin_matches_the_sircl_tree(sircl_root):
    """Without GLM_DCP_DECODE_SIRCL_ROOT the tree is this repository's SIRCL source, which the image installs."""
    checks = [check for check in pkg.FILE_CHECKS if check.package == pkg.SIRCL_PACKAGE]
    assert len(checks) == 17
    for check in checks:
        pkg.verify_file(check, {pkg.SIRCL_PACKAGE: sircl_root})
    assert pkg.sircl_version(sircl_root) == pkg.SIRCL_VERSION


def test_crlf_checkouts_have_the_lf_digest(tmp_path):
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"a = 1\nb = 2\n")
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert pkg.digest(lf) == pkg.digest(crlf)
    crlf.write_bytes(b"a = 1\r\nb = 3\r\n")
    assert pkg.digest(lf) != pkg.digest(crlf)


def _copy(sircl_root: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "sparkring_sircl"
    shutil.copytree(sircl_root, copy, ignore=shutil.ignore_patterns("__pycache__"))
    return copy


def test_a_changed_or_missing_sircl_file_refuses(sircl_root, tmp_path):
    copy = _copy(sircl_root, tmp_path)
    check = next(c for c in pkg.FILE_CHECKS if c.path == "oneshot/_scatter_cute.py")
    with (copy / check.path).open("a", encoding="utf-8") as handle:
        handle.write("# one more line\n")
    with pytest.raises(pkg.PatchRefused, match="the scatter kernel whose wire protocol"):
        pkg.verify_file(check, {pkg.SIRCL_PACKAGE: copy})
    (copy / check.path).unlink()
    with pytest.raises(pkg.PatchRefused, match="is missing"):
        pkg.verify_file(check, {pkg.SIRCL_PACKAGE: copy})


def test_another_sircl_version_refuses_before_any_file_is_compared(sircl_root, vllm_root, b12x_root, tmp_path):
    copy = _copy(sircl_root, tmp_path)
    init = copy / "__init__.py"
    text = init.read_text(encoding="utf-8")
    assert f'__version__ = "{pkg.SIRCL_VERSION}"' in text
    init.write_text(text.replace(f'__version__ = "{pkg.SIRCL_VERSION}"', '__version__ = "0.2.0"'), encoding="utf-8")
    with pytest.raises(pkg.PatchRefused, match=re.escape(f"is SIRCL 0.2.0; this plugin was built against SIRCL "
                                                         f"{pkg.SIRCL_VERSION}")):
        pkg.verify({"vllm": vllm_root, "b12x": b12x_root, pkg.SIRCL_PACKAGE: copy})


def test_verify_accepts_the_trees_it_was_built_against(sircl_root, vllm_root, b12x_root):
    prepared = pkg.verify({"vllm": vllm_root, "b12x": b12x_root, pkg.SIRCL_PACKAGE: sircl_root})
    assert [item.patch.qualname for item in prepared] == [patch.qualname for patch in pkg.PATCHES]


def _refresh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PROJECT / "tools" / "refresh_pins.py"), *args],
                          capture_output=True, text=True, timeout=120)


def test_the_refresh_tool_finds_no_drift_against_the_pinned_trees(sircl_root, vllm_root, b12x_root):
    done = _refresh("--check", "--vllm", str(vllm_root), "--b12x", str(b12x_root), "--sircl", str(sircl_root))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "every pin matches" in done.stdout


def test_the_refresh_tool_reports_a_changed_sircl_file(sircl_root, vllm_root, b12x_root, tmp_path):
    copy = _copy(sircl_root, tmp_path)
    with (copy / "protocol.py").open("a", encoding="utf-8") as handle:
        handle.write("# changed\n")
    done = _refresh("--check", "--vllm", str(vllm_root), "--b12x", str(b12x_root), "--sircl", str(copy))
    assert done.returncode == 1
    assert "sparkring_sircl/protocol.py" in done.stdout and "1 pin(s) differ" in done.stdout


def test_the_sircl_only_refresh_finds_no_drift_and_needs_no_image_tree(sircl_root, tmp_path):
    done = _refresh("--check", "--sircl-only", "--sircl", str(sircl_root))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "every pin matches" in done.stdout
    copy = _copy(sircl_root, tmp_path)
    with (copy / "oneshot" / "runtime.py").open("a", encoding="utf-8") as handle:
        handle.write("# changed\n")
    done = _refresh("--check", "--sircl-only", "--sircl", str(copy))
    assert done.returncode == 1
    assert "sparkring_sircl/oneshot/runtime.py" in done.stdout and "1 pin(s) differ" in done.stdout


def test_the_dist_info_registers_this_version_in_vllm_general_plugins():
    dist_info = PROJECT / "dist-info" / f"{pkg.PLUGIN_NAME}-{pkg.PLUGIN_VERSION}.dist-info"
    metadata = (dist_info / "METADATA").read_text(encoding="utf-8").splitlines()
    assert "Name: glm-dcp-decode-comm" in metadata and f"Version: {pkg.PLUGIN_VERSION}" in metadata
    assert (dist_info / "entry_points.txt").read_text(encoding="utf-8").splitlines() == [
        "[vllm.general_plugins]", f"{pkg.PLUGIN_NAME} = {pkg.PLUGIN_NAME}:register"]
    assert (dist_info / "top_level.txt").read_text(encoding="utf-8").split() == [pkg.PLUGIN_NAME]
    assert [path.name for path in (PROJECT / "dist-info").iterdir()] == [dist_info.name]
