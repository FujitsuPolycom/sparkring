"""The ``libsircl`` vLLM plugin; offline, with a file standing in for the library."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).with_name("sparkring_libsircl.py")
spec = importlib.util.spec_from_file_location("sparkring_libsircl", SOURCE)
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


def library(tmp_path, data=b"\x7fELF libsircl"):
    path = tmp_path / "libsircl.so.0.6.0"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def test_register_points_pynccl_at_the_checked_library(tmp_path, monkeypatch):
    path, digest = library(tmp_path)
    monkeypatch.setenv(plugin.LIBRARY, str(path))
    monkeypatch.setenv(plugin.DIGEST, digest)
    # The installer image's entrypoint names its NVIDIA NCCL here before vLLM starts.
    monkeypatch.setenv(plugin.TARGET, "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2")
    plugin.register()
    plugin.register()
    import os
    assert os.environ[plugin.TARGET] == str(path)


@pytest.mark.parametrize("change, message", [
    ({plugin.LIBRARY: None}, "needs"),
    ({plugin.DIGEST: "0" * 64}, "not SPARKRING_LIBSIRCL_SHA256"),
    ({plugin.LIBRARY: "relative/libsircl.so"}, "absolute path"),
    ({plugin.LIBRARY: "/nonexistent/libsircl.so.0.6.0"}, "regular file"),
])
def test_a_missing_or_different_library_fails_the_process(tmp_path, change, message):
    path, digest = library(tmp_path)
    environ = {plugin.LIBRARY: str(path), plugin.DIGEST: digest, plugin.TARGET: "/nccl"}
    for key, value in change.items():
        if value is None:
            environ.pop(key)
        else:
            environ[key] = value
    with pytest.raises(plugin.LibsirclSelectionError, match=message):
        plugin.selected(environ)


def test_a_symbolic_link_is_not_the_library(tmp_path):
    path, digest = library(tmp_path)
    link = tmp_path / "link.so"
    try:
        link.symlink_to(path)
    except OSError:
        pytest.skip("this host cannot create symbolic links")
    with pytest.raises(plugin.LibsirclSelectionError, match="regular file"):
        plugin.selected({plugin.LIBRARY: str(link), plugin.DIGEST: digest})


def test_the_entry_point_names_the_register_function():
    group, name, value = plugin.ENTRY_POINT
    assert (group, name) == ("vllm.general_plugins", "libsircl")
    module, _, function = value.partition(":")
    assert module == SOURCE.stem and callable(getattr(plugin, function))
