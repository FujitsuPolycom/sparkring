import json

import pytest

from runtime.common import installer_image
from runtime.host import images

MIMO = ("mimo-v26-flash-mopd-tp2", "mimo-v26-flash-mopd-tp4")


def test_images_lists_the_default_first_with_its_release_tag(capsys):
    assert images.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == installer_image.DEFAULT_LOCK.parent.name + "  (2026.10.0, default)"
    assert lines[1].endswith("GiB download; runs every installer profile")
    mimovision = lines.index("dev-20260927-mimovision-cuda1342-nccl2323-status032  (2026.09.5)")
    assert lines[mimovision + 1].endswith("GiB download; runs every installer profile")
    assert any(line.endswith("runs every installer profile except swift15-qwen38-flash-next-tp2, "
                             "swift15-qwen38-flash-next-tp4, deepseek-v41-flash-tp4") for line in lines)
    assert lines[-1] == "Install a profile on one of them: sudo sparkring install --profile PROFILE --image NAME"


def test_images_for_a_profile_lists_only_images_that_run_it(capsys):
    assert images.main(["--profile", MIMO[0], "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["default"] and all(MIMO[0] in row["profiles"] for row in listed)
    assert "dev-20260927-mimovision-cuda1342-nccl2323-status032" in [row["name"] for row in listed]
    assert not any(name.startswith("mimo-v26-flash-rl") for row in listed for name in row["profiles"])


def test_images_refuses_a_name_that_is_not_an_installer_profile(capsys):
    with pytest.raises(SystemExit):
        images.main(["--profile", "qwen"])
    assert "not an installer profile" in capsys.readouterr().err
