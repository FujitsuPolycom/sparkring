import json

import pytest

from runtime.common import installer_image
from runtime.host import images

MIMO = ("mimo-v26-flash-mopd-tp2", "mimo-v26-flash-mopd-tp4")


def test_images_lists_the_default_first_and_each_image_s_release_tag(capsys):
    assert images.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036  (2026.10.2, default)"
    assert lines[1].endswith("GiB download; transports libsircl, prepared, sircl; kraken line; "
                             "runs every installer profile")
    assert "dev-20261001-kraken-cuda1342-nccl2323-status034  (2026.10.0)" in lines
    # The images published before 2026.10.2 carry the prepared transport only and name no image line.
    rollback = lines.index(installer_image.DEFAULT_LOCK.parent.name + "  (2026.10.1)")
    assert lines[rollback + 1].endswith("GiB download; transports prepared; runs every installer profile")
    mimovision = lines.index("dev-20260927-mimovision-cuda1342-nccl2323-status032  (2026.09.5)")
    assert lines[mimovision + 1].endswith("GiB download; transports prepared; runs every installer profile")
    assert any(line.endswith("runs every installer profile except swift15-qwen38-flash-next-tp2, "
                             "swift15-qwen38-flash-next-tp4, deepseek-v41-flash-tp4") for line in lines)
    assert lines[-1] == "Install a profile on one of them: sudo sparkring install --profile PROFILE --image NAME"


def test_images_for_a_profile_lists_only_images_that_run_it(capsys):
    assert images.main(["--profile", MIMO[0], "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["default"] and all(MIMO[0] in row["profiles"] for row in listed)
    assert "dev-20260927-mimovision-cuda1342-nccl2323-status032" in [row["name"] for row in listed]
    assert not any(name.startswith("mimo-v26-flash-rl") for row in listed for name in row["profiles"])
    assert listed[0]["transports"] == ["libsircl", "prepared", "sircl"] and listed[0]["line"] == "kraken"
    assert all(row["transports"] == ["prepared"] and row["line"] is None for row in listed[1:])
    assert not any(row["archived"] for row in listed)


def test_a_v3_image_lists_its_line_and_its_transports(capsys, monkeypatch):
    from runtime.common.test_image_lock import sircl_lock
    rows = installer_image.catalog()
    value = sircl_lock()
    rows.append({"name": value["name"], "path": "v3.json", "lock": value, "tags": [], "default": False})
    monkeypatch.setattr(installer_image, "catalog", lambda: rows)
    assert images.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    shown = lines[lines.index(value["name"]) + 1]
    assert "; transports prepared, sircl; kraken line; runs " in shown


def test_images_refuses_a_name_that_is_not_an_installer_profile(capsys):
    with pytest.raises(SystemExit):
        images.main(["--profile", "qwen"])
    assert "not an installer profile" in capsys.readouterr().err


def test_images_selects_a_profile_by_its_alias(capsys):
    assert images.main(["--profile", "glm53-flash-tp4", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed and all("glm53-flash-nvfp4-spark-tp4" in row["profiles"] for row in listed)
