"""Repeating an incomplete asset preparation, without hosts."""
import hashlib

import pytest

from runtime.common import installer
from runtime.common.test_installer import QWEN, Hosts, site


@pytest.fixture
def directory(tmp_path):
    data = b"offline source bundle fixture"
    lock = installer.make_lock(QWEN, site(), "1" * 40, hashlib.sha256(data).hexdigest())
    installer.write(tmp_path / "deployment.lock.json", lock)
    (tmp_path / "source.bundle").write_bytes(data)
    return tmp_path


@pytest.mark.parametrize("uncertain", [False, True])
def test_incomplete_preparation_is_repeated_as_a_new_generation(directory, uncertain):
    with pytest.raises(RuntimeError):
        installer.apply(directory, "prepare", runner=Hosts(("model", 1), uncertain=uncertain), execute=True)
    repeat = Hosts()
    result = installer.apply(directory, "prepare", runner=repeat, execute=True)
    assert result["complete"] and result["generation"] == 2
    # Every action runs again rather than resuming the refused receipt.
    assert {("source", 0), ("model", 0), ("model", 1), ("image", 1)} <= set(repeat.events)
    assert sorted(p.name for p in (directory / "operations").glob("*.json")) == ["0001-prepare.json", "0002-prepare.json"]
    assert installer.apply(directory, "up", runner=Hosts(), execute=True)["complete"]


def test_complete_preparation_is_rechecked_in_its_own_generation(directory):
    installer.apply(directory, "prepare", runner=Hosts(), execute=True)
    repeat = Hosts()
    assert installer.apply(directory, "prepare", runner=repeat, execute=True)["generation"] == 1
    assert ("model-check", 0) in repeat.events and ("model", 0) not in repeat.events


def test_incomplete_preparation_still_blocks_a_start(directory):
    with pytest.raises(RuntimeError):
        installer.apply(directory, "prepare", runner=Hosts(("image", 0)), execute=True)
    start = Hosts()
    with pytest.raises(ValueError, match="incomplete or uncertain"):
        installer.apply(directory, "up", runner=start, execute=True)
    assert start.events == []
