"""The ``libsircl`` vLLM plugin; offline, with a file and a loader standing in for the library."""
import hashlib
import importlib.util
import os
import sys
import types
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


class Library:
    """A loaded library as ctypes presents it: its exported functions as attributes."""

    def __init__(self, identity=b'{"library":"libsircl","version":"0.6.0"}', functions=("ncclAllReduce",)):
        identity_ = identity

        class Info:
            restype = None

            def __call__(self):
                return identity_
        self.sirclGetInfo = Info()
        for name in functions:
            setattr(self, name, object())


def test_register_points_pynccl_at_the_checked_library(tmp_path, monkeypatch):
    path, digest = library(tmp_path)
    monkeypatch.setenv(plugin.LIBRARY, str(path))
    monkeypatch.setenv(plugin.DIGEST, digest)
    # The installer image's entrypoint names its NVIDIA NCCL here before vLLM starts.
    monkeypatch.setenv(plugin.TARGET, "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2")
    loaded = []
    monkeypatch.setattr(plugin, "LOADER", lambda name: loaded.append(name) or Library())
    monkeypatch.setattr(plugin, "pynccl_functions", lambda: ["ncclAllReduce"])
    plugin.register()
    plugin.register()
    assert os.environ[plugin.TARGET] == str(path) and loaded == [str(path), str(path)]


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


@pytest.mark.parametrize("loaded, functions, message", [
    (Library(identity=b'{"library":"nccl"}'), [], "identifies itself as 'nccl'"),
    (Library(functions=()), ["ncclAllReduce", "ncclCommSuspend"], "lacks functions vLLM's PyNccl binds: "
                                                                   "ncclAllReduce, ncclCommSuspend"),
    (None, [], "does not load as libsircl"),
])
def test_a_library_that_is_not_libsircl_or_lacks_pyncclss_functions_fails_instead_of_falling_back(
        loaded, functions, message):
    def load(path):
        if loaded is None:
            raise OSError("undefined symbol")
        return loaded
    with pytest.raises(plugin.LibsirclSelectionError, match=message):
        plugin.check_binding("/opt/sparkring/libsircl/lib/libsircl.so.0.6.0", load=load, functions=lambda: functions)


WRAPPER = "vllm.distributed.device_communicators.pynccl_wrapper"


def test_outside_vllm_only_the_identity_is_checked(monkeypatch):
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.setitem(sys.modules, WRAPPER, None)
    monkeypatch.setattr(plugin.importlib.util, "find_spec", lambda name: None)
    assert plugin.pynccl_functions() is None
    info = plugin.check_binding("/lib", load=lambda path: Library(functions=()), functions=plugin.pynccl_functions)
    assert info["library"] == "libsircl"


def test_an_importable_vllm_whose_pynccl_wrapper_fails_to_import_is_refused(monkeypatch):
    # The functions PyNccl binds cannot be listed, so the binding check must not pass with nothing to check.
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, WRAPPER, None)
    with pytest.raises(plugin.LibsirclSelectionError, match="PyNccl wrapper is not"):
        plugin.pynccl_functions()
    with pytest.raises(plugin.LibsirclSelectionError, match="PyNccl wrapper is not"):
        plugin.check_binding("/lib", load=lambda path: Library(), functions=plugin.pynccl_functions)


def test_a_library_replaced_after_its_check_is_refused(tmp_path):
    path, digest = library(tmp_path)
    assert plugin.selected({plugin.LIBRARY: str(path), plugin.DIGEST: digest}) == str(path)
    plugin.identical(str(path))
    other = tmp_path / "other.so"
    other.write_bytes(b"ELF something else")
    os.replace(other, path)
    with pytest.raises(plugin.LibsirclSelectionError, match="replaced after its SHA-256 was checked"):
        plugin.identical(str(path))


def test_a_path_through_a_linked_directory_is_not_the_library(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    path, digest = library(real)
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("this host cannot create symbolic links")
    with pytest.raises(plugin.LibsirclSelectionError, match="not its own real path"):
        plugin.selected({plugin.LIBRARY: str(linked / path.name), plugin.DIGEST: digest})


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


class FrozenEnvs(types.ModuleType):
    """A ``vllm.envs`` whose values were read before the plugin ran."""

    def __init__(self, value):
        super().__init__("vllm.envs")
        self.VLLM_NCCL_SO_PATH = value


class LiveEnvs(types.ModuleType):
    """A ``vllm.envs`` that reads the environment at each access, as vLLM does without its cache."""

    def __init__(self):
        super().__init__("vllm.envs")

    def __getattr__(self, name):
        if name == plugin.TARGET:
            return os.environ.get(name)
        raise AttributeError(name)


@pytest.mark.parametrize("envs, message", [
    (LiveEnvs(), None),
    (FrozenEnvs("/opt/sparkring/toolchain/nccl/lib/libnccl.so.2"), "vLLM resolves VLLM_NCCL_SO_PATH to"),
    (types.ModuleType("vllm.envs"), "vllm.envs has no VLLM_NCCL_SO_PATH"),
])
def test_register_confirms_vllm_resolves_the_library(tmp_path, monkeypatch, envs, message):
    path, digest = library(tmp_path)
    monkeypatch.setenv(plugin.LIBRARY, str(path))
    monkeypatch.setenv(plugin.DIGEST, digest)
    monkeypatch.setenv(plugin.TARGET, "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2")
    monkeypatch.setattr(plugin, "LOADER", lambda name: Library())
    monkeypatch.setattr(plugin, "pynccl_functions", lambda: ["ncclAllReduce"])
    package = types.ModuleType("vllm")
    package.envs = envs
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    if message is None:
        plugin.register()
        assert plugin.resolved_target() == str(path)
    else:
        with pytest.raises(plugin.LibsirclSelectionError, match=message):
            plugin.register()
