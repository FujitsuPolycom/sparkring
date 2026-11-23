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


def test_register_invokes_opt_in_ep_extensions_only_when_enabled(tmp_path, monkeypatch):
    path, digest = library(tmp_path)
    monkeypatch.setenv(plugin.LIBRARY, str(path))
    monkeypatch.setenv(plugin.DIGEST, digest)
    monkeypatch.setattr(plugin, "LOADER", lambda name: Library())
    monkeypatch.setattr(plugin, "pynccl_functions", lambda: ["ncclAllReduce"])
    monkeypatch.setattr(plugin, "resolved_target", lambda: str(path))
    called = []
    fake_b12x = types.ModuleType("sircl_ep_b12x")
    fake_b12x.install_b12x_ep_moe = lambda: called.append("b12x")
    fake_a2a = types.ModuleType("sircl_ep_all2all")
    fake_a2a.install_sircl_ep_all2all = lambda: called.append("a2a")
    monkeypatch.setitem(sys.modules, "sircl_ep_b12x", fake_b12x)
    monkeypatch.setitem(sys.modules, "sircl_ep_all2all", fake_a2a)

    monkeypatch.delenv("SIRCL_B12X_EP_MOE", raising=False)
    monkeypatch.delenv("SIRCL_EP_ALL2ALL", raising=False)
    plugin.register()
    assert called == []

    monkeypatch.setenv("SIRCL_B12X_EP_MOE", "1")
    plugin.register()
    assert called == ["b12x"]

    monkeypatch.setenv("SIRCL_EP_ALL2ALL", "on")
    plugin.register()
    assert called == ["b12x", "b12x", "a2a"]


def test_sircl_ep_b12x_parallel_config_matrix():
    mod_path = Path(__file__).with_name("sircl_ep_b12x.py")
    spec_b12x = importlib.util.spec_from_file_location("sircl_ep_b12x_test", mod_path)
    try:
        mod = importlib.util.module_from_spec(spec_b12x)
        spec_b12x.loader.exec_module(mod)
    except ImportError:
        pytest.skip("torch not installed in local test env")

    def cfg(**kw):
        defaults = {
            "use_ep": False,
            "ep_size": 1,
            "use_all2all_kernels": False,
            "use_ag_rs_all2all_kernels": False,
            "enable_eplb": False,
        }
        defaults.update(kw)
        return types.SimpleNamespace(**defaults)

    assert mod._is_ep_parallel_supported(cfg()) is True
    assert mod._is_ep_parallel_supported(cfg(use_ep=True, ep_size=4)) is True
    assert mod._is_ep_parallel_supported(
        cfg(use_ep=True, ep_size=4, use_all2all_kernels=True, use_ag_rs_all2all_kernels=True)
    ) is True
    assert mod._is_ep_parallel_supported(
        cfg(use_ep=True, ep_size=4, use_all2all_kernels=True, use_ag_rs_all2all_kernels=False)
    ) is False
    assert mod._is_ep_parallel_supported(cfg(use_ep=True, ep_size=4, enable_eplb=True)) is False
    assert mod._is_ep_parallel_supported(cfg(use_ep=True, ep_size=1)) is False


def test_sircl_ep_b12x_remap_and_widen():
    import dataclasses

    mod_path = Path(__file__).with_name("sircl_ep_b12x.py")
    spec_b12x = importlib.util.spec_from_file_location("sircl_ep_b12x_test_w4a8", mod_path)
    try:
        import torch

        mod = importlib.util.module_from_spec(spec_b12x)
        spec_b12x.loader.exec_module(mod)
    except ImportError:
        pytest.skip("torch not installed in local test env")

    # Rank 3 of EP4 owns experts 6, 7 out of 8: negative -1 MUST stay -1, not wrap to expert_map[-1] == 1!
    emap = torch.tensor([-1, -1, -1, -1, -1, -1, 0, 1], dtype=torch.int32)
    ids = torch.tensor([[6, -1, 7, 8], [0, -5, 7, 6]], dtype=torch.int64)
    remapped = mod._remap_expert_ids(ids, emap, global_num_experts=8)
    assert remapped.dtype == torch.int32
    assert remapped.is_contiguous()
    assert remapped.tolist() == [[0, -1, 1, -1], [-1, -1, 1, 0]]

    @dataclasses.dataclass(frozen=True)
    class FakeWorkload:
        stage: str
        token_counts: tuple[int, ...]
        fixed_token_counts: tuple[int, ...]
        max_tokens: int

    wl = FakeWorkload(stage="weights", token_counts=(5, 6, 96, 8192), fixed_token_counts=(5, 6, 96), max_tokens=8192)
    widened = mod._widen_workload_for_ep(wl, ep_mult=4, cap=32768)
    assert widened.max_tokens == 32768
    assert widened.fixed_token_counts == (5, 6, 20, 24, 96, 384)
    assert set(widened.fixed_token_counts) <= set(widened.token_counts)
    assert widened.token_counts[-1] == 32768
    assert mod._widen_workload_for_ep(wl, ep_mult=1) == wl


def test_sircl_ep_b12x_resize_dp_workspace():
    mod_path = Path(__file__).with_name("sircl_ep_b12x.py")
    spec_b12x = importlib.util.spec_from_file_location("sircl_ep_b12x_test_ws", mod_path)
    try:
        import torch

        mod = importlib.util.module_from_spec(spec_b12x)
        spec_b12x.loader.exec_module(mod)
    except ImportError:
        pytest.skip("torch not installed in local test env")

    class FakeExperts:
        def __init__(self, ws2_len_fn):
            self._ws2_len_fn = ws2_len_fn
            self.output_dtype = torch.bfloat16

        def workspace_dtype(self, in_dtype):
            return in_dtype

        def workspace_shapes(self, m, n, k, topk, ge, le, meta, act):
            return (0,), (self._ws2_len_fn(m),), (m, k)

    # W4A8 case: workspace2 is already pre-sized for max_tokens (independent of M).
    # Must reuse ws13 and ws2 in place and only allocate a disjoint fused_out.
    ws13 = torch.empty((0,), dtype=torch.bfloat16)
    ws2 = torch.empty((1024,), dtype=torch.bfloat16)
    out_small = torch.empty((16, 64), dtype=torch.bfloat16)
    new_ws13, new_ws2, new_out = mod._resize_dp_workspace(
        FakeExperts(lambda _m: 1024),
        (ws13, ws2, out_small),
        in_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        m_full=64,
        n_dim=128,
        k_dim=64,
        top_k=6,
        global_num_experts=384,
        local_num_experts=96,
        expert_tokens_meta=None,
        activation="silu",
    )
    assert new_ws13 is ws13
    assert new_ws2 is ws2
    assert new_out.shape == (64, 64)
    assert new_out.dtype == torch.bfloat16
    assert new_out.data_ptr() != out_small.data_ptr()

    # W4A16 case: workspace2 grows with M. Must allocate fresh ws2 + fused_out.
    new_ws13_b, new_ws2_b, new_out_b = mod._resize_dp_workspace(
        FakeExperts(lambda m: m * 32),
        (ws13, ws2, out_small),
        in_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        m_full=64,
        n_dim=128,
        k_dim=64,
        top_k=6,
        global_num_experts=384,
        local_num_experts=96,
        expert_tokens_meta=None,
        activation="silu",
    )
    assert new_ws13_b is ws13
    assert new_ws2_b is not ws2
    assert new_ws2_b.shape == (2048,)
    assert new_out_b.shape == (64, 64)


def test_sircl_ep_b12x_dp_sync_num_reqs(monkeypatch):
    torch = pytest.importorskip("torch")
    import dataclasses
    import enum
    import types

    mod_path = Path(__file__).with_name("sircl_ep_b12x.py")
    spec_b12x = importlib.util.spec_from_file_location("sircl_ep_b12x_test_dp_sync", mod_path)
    mod = importlib.util.module_from_spec(spec_b12x)
    spec_b12x.loader.exec_module(mod)

    class FakeCGMode(enum.IntEnum):
        NONE = 0
        PIECEWISE = 1
        FULL = 2

    @dataclasses.dataclass(frozen=True)
    class FakeDesc:
        cg_mode: FakeCGMode
        num_tokens: int
        num_reqs: int | None

    @dataclasses.dataclass(frozen=True)
    class FakeDPSync:
        num_reqs: int

    class FakeDist:
        def all_reduce(self, tensor, group=None):
            # Simulate rank 0 having num_reqs=2 and rank 3 having num_reqs=1.
            tensor[5, 0] = 2
            tensor[5, 1] = 1
            tensor[5, 2] = 1
            tensor[5, 3] = 1

    fake_comp = types.ModuleType("vllm.config.compilation")
    fake_comp.CUDAGraphMode = FakeCGMode
    fake_dp = types.ModuleType("vllm.v1.worker.gpu.dp_utils")
    fake_dp.dist = FakeDist()

    def orig_sync(cm, desired_desc, num_tokens, num_reqs, uniform_tok, dp_size, dp_rank):
        tensor = torch.zeros(6, dp_size, dtype=torch.int32)
        tensor[5, dp_rank] = num_reqs
        fake_dp.dist.all_reduce(tensor)
        desc = cm.dispatch(num_reqs, 6, None)
        return desc, FakeDPSync(
            num_reqs=desc.num_reqs if desc.cg_mode == FakeCGMode.FULL else int(tensor[5].max())
        )

    def orig_dispatch(cm, num_reqs, num_tokens, uniform_tok, dp_size, dp_rank, dp_sync=None):
        desc = cm.dispatch(num_reqs, num_tokens, uniform_tok)
        return desc, dp_sync

    fake_dp.sync_cudagraph_and_dp_padding = orig_sync
    fake_dp.dispatch_cg_and_sync_dp = orig_dispatch

    for name, m in (
        ("vllm", types.ModuleType("vllm")),
        ("vllm.config", types.ModuleType("vllm.config")),
        ("vllm.config.compilation", fake_comp),
        ("vllm.v1", types.ModuleType("vllm.v1")),
        ("vllm.v1.worker", types.ModuleType("vllm.v1.worker")),
        ("vllm.v1.worker.gpu", types.ModuleType("vllm.v1.worker.gpu")),
        ("vllm.v1.worker.gpu.dp_utils", fake_dp),
    ):
        monkeypatch.setitem(sys.modules, name, m)
    sys.modules["vllm.v1.worker.gpu"].dp_utils = fake_dp

    mod._patch_dp_sync_cudagraph_num_reqs()

    seen_full = []

    class FakeFullManager:
        def dispatch(self, num_reqs, num_tokens, uniform_tok):
            seen_full.append((num_reqs, num_tokens))
            return FakeDesc(FakeCGMode.FULL, num_tokens, num_reqs)

    # Rank 3 has local num_reqs=1 while rank 0 has num_reqs=2; FULL graph must agree on 2.
    desc_r3, sync_r3 = fake_dp.sync_cudagraph_and_dp_padding(
        FakeFullManager(), FakeDesc(FakeCGMode.FULL, 6, 1), 6, 1, None, 4, 3
    )
    assert desc_r3.num_reqs == 2
    assert sync_r3.num_reqs == 2
    assert seen_full == [(2, 6)]

    # DSpark reuse path with dp_sync.num_reqs=2 must also dispatch with num_reqs=2 on rank 3.
    seen_full.clear()
    dspark_desc, _ = fake_dp.dispatch_cg_and_sync_dp(
        FakeFullManager(), 1, 10, 5, 4, 3, dp_sync=FakeDPSync(num_reqs=2)
    )
    assert dspark_desc.num_reqs == 2
    assert seen_full == [(2, 10)]

    # Eager (NONE) fallback must preserve local num_reqs=1 on the returned descriptor.
    class FakeNoneManager:
        def dispatch(self, num_reqs, num_tokens, uniform_tok):
            return FakeDesc(FakeCGMode.NONE, num_tokens, num_reqs)

    desc_none, sync_none = fake_dp.sync_cudagraph_and_dp_padding(
        FakeNoneManager(), FakeDesc(FakeCGMode.FULL, 6, 1), 6, 1, None, 4, 3
    )
    assert desc_none.num_reqs == 1
    assert sync_none.num_reqs == 2


def test_sircl_ep_b12x_dp_speculator_propose_sync(monkeypatch):
    torch = pytest.importorskip("torch")
    import types
    from typing import NamedTuple

    mod_path = Path(__file__).with_name("sircl_ep_b12x.py")
    spec_b12x = importlib.util.spec_from_file_location("sircl_ep_b12x_test_spec_sync", mod_path)
    mod = importlib.util.module_from_spec(spec_b12x)
    spec_b12x.loader.exec_module(mod)

    class FakeExecState(NamedTuple):
        num_spec_tokens_to_schedule: int
        boundary_logits_only: bool = False

    propose_calls = []
    adaptive_calls = []

    class FakeSpec:
        def propose(self, dummy_run=False):
            propose_calls.append(dummy_run)
            return torch.ones((2, 5), dtype=torch.int32)

    class FakeAdaptive:
        def record_confidences(self):
            adaptive_calls.append(True)

    class FakeRunner:
        def __init__(self, dp_size):
            self.dp_size = dp_size
            self.speculator = FakeSpec()
            self.adaptive_verification = FakeAdaptive()
            self.execute_model_state = None

        def sample_tokens(self, grammar_output):
            st = self.execute_model_state
            self.execute_model_state = None
            if st.num_spec_tokens_to_schedule > 0:
                draft = self.speculator.propose()
                if self.adaptive_verification is not None:
                    self.adaptive_verification.record_confidences()
                return draft
            return torch.empty((2, 0), dtype=torch.int32)

    fake_mr = types.ModuleType("vllm.v1.worker.gpu.model_runner")
    fake_mr.GPUModelRunner = FakeRunner
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.model_runner", fake_mr)

    mod._patch_dp_speculator_propose_sync()

    # DP=4, num_spec_tokens_to_schedule=0: must run propose(dummy_run=True), return 0 draft tokens,
    # skip adaptive_verification, and restore speculator.propose + adaptive_verification afterwards.
    r_dp4 = FakeRunner(dp_size=4)
    r_dp4.execute_model_state = FakeExecState(num_spec_tokens_to_schedule=0)
    out = r_dp4.sample_tokens(None)
    assert out.shape == (2, 0)
    assert propose_calls == [True]
    assert adaptive_calls == []
    assert r_dp4.adaptive_verification is not None

    # Subsequent normal decode step (num_spec_tokens_to_schedule=5) runs normal propose + adaptive.
    propose_calls.clear()
    r_dp4.execute_model_state = FakeExecState(num_spec_tokens_to_schedule=5)
    out2 = r_dp4.sample_tokens(None)
    assert out2.shape == (2, 5)
    assert propose_calls == [False]
    assert adaptive_calls == [True]

    # DP=1, num_spec_tokens_to_schedule=0: must NOT force propose.
    propose_calls.clear()
    r_dp1 = FakeRunner(dp_size=1)
    r_dp1.execute_model_state = FakeExecState(num_spec_tokens_to_schedule=0)
    out3 = r_dp1.sample_tokens(None)
    assert out3.shape == (2, 0)
    assert propose_calls == []


def test_sircl_ep_b12x_router_skip_padding(monkeypatch):
    torch = pytest.importorskip("torch")
    import types

    mod_path = Path(__file__).with_name("sircl_ep_b12x.py")
    spec_b12x = importlib.util.spec_from_file_location("sircl_ep_b12x_test_pad", mod_path)
    mod = importlib.util.module_from_spec(spec_b12x)
    spec_b12x.loader.exec_module(mod)

    seen_kwargs = []
    pad_tensor = torch.tensor([False, False, True, True])

    def fake_dsv4_topk(gating_output, *args, **kwargs):
        seen_kwargs.append(kwargs.copy())
        return gating_output

    fake_router = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
    )
    fake_router.dsv4_topk = fake_dsv4_topk
    fake_router._get_padding_mask = lambda n: pad_tensor[:n]

    for name, m in (
        ("vllm", types.ModuleType("vllm")),
        ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
        ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
        (
            "vllm.model_executor.layers.fused_moe",
            types.ModuleType("vllm.model_executor.layers.fused_moe"),
        ),
        (
            "vllm.model_executor.layers.fused_moe.router",
            types.ModuleType("vllm.model_executor.layers.fused_moe.router"),
        ),
        (
            "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router",
            fake_router,
        ),
    ):
        monkeypatch.setitem(sys.modules, name, m)
    sys.modules["vllm.model_executor.layers.fused_moe.router"].fused_topk_bias_router = (
        fake_router
    )

    monkeypatch.setenv("SIRCL_EP_ROUTER_SKIP_PADDING", "1")
    mod._patch_router_skip_padding()

    x = torch.zeros((4, 8))
    fake_router.dsv4_topk(x)
    assert len(seen_kwargs) == 1
    assert torch.equal(seen_kwargs[0]["is_padding"], pad_tensor)

    # Explicit caller is_padding must be preserved.
    custom = torch.tensor([True, False, False, False])
    fake_router.dsv4_topk(x, is_padding=custom)
    assert torch.equal(seen_kwargs[1]["is_padding"], custom)


def test_sircl_ep_b12x_force_grouped_routing(monkeypatch):
    pytest.importorskip("torch")
    import dataclasses
    import types

    mod_path = Path(__file__).with_name("sircl_ep_b12x.py")
    spec_b12x = importlib.util.spec_from_file_location("sircl_ep_b12x_test_grp", mod_path)
    mod = importlib.util.module_from_spec(spec_b12x)
    spec_b12x.loader.exec_module(mod)

    @dataclasses.dataclass(frozen=True)
    class FakeConfig:
        backend: str
        dynamic_route_mode: str | None = None

    @dataclasses.dataclass(frozen=True)
    class FakeSelection:
        config: FakeConfig

    @dataclasses.dataclass(frozen=True)
    class FakePlan:
        _compile_jobs: object
        _memory_requirements: object
        _materialize: object

    @dataclasses.dataclass(frozen=True)
    class FakeCompositePlan:
        variants: dict

    seen: dict[str, list[FakeConfig]] = {"compile": [], "memory": [], "materialize": []}

    def make_plan():
        leaf = FakePlan(
            _compile_jobs=lambda cfg, dev: seen["compile"].append(cfg) or cfg.dynamic_route_mode,
            _memory_requirements=lambda cfg, dev: seen["memory"].append(cfg) or cfg.dynamic_route_mode,
            _materialize=lambda sel, dev: seen["materialize"].append(sel.config) or sel.config.dynamic_route_mode,
        )
        return FakeCompositePlan(variants={5: leaf})

    fake_impl = types.ModuleType("b12x.moe.fused_moe._impl")
    fake_impl._DIRECT_ROUTING_MAX_ROUTED_ROWS = 32
    def orig_sel(*, route_mode, **_kw):
        return route_mode == "direct"

    fake_impl._dynamic_direct_routing_selected = orig_sel
    fake_impl._heuristic_dynamic_route_mode = lambda *_a, **_kw: "direct"

    fake_prep = types.ModuleType("b12x.moe.fused_moe._preparation")
    fake_prep.plan = make_plan

    fake_fm = types.ModuleType("b12x.moe.fused_moe")
    fake_fm._impl = fake_impl
    fake_fm._preparation = fake_prep
    fake_fm._canonical_plan_execution = make_plan
    fake_fm.plan_execution = lambda **kw: fake_fm._canonical_plan_execution(**kw)

    for name, m in (
        ("b12x", types.ModuleType("b12x")),
        ("b12x.moe", types.ModuleType("b12x.moe")),
        ("b12x.moe.fused_moe", fake_fm),
        ("b12x.moe.fused_moe._impl", fake_impl),
        ("b12x.moe.fused_moe._preparation", fake_prep),
    ):
        monkeypatch.setitem(sys.modules, name, m)

    monkeypatch.setenv("SIRCL_EP_FORCE_GROUPED_ROUTING", "0")
    mod._patch_ep_grouped_routing()
    assert fake_impl._dynamic_direct_routing_selected is orig_sel
    assert fake_impl._heuristic_dynamic_route_mode() == "direct"
    unpatched_leaf = fake_fm.plan_execution().variants[5]
    assert unpatched_leaf._compile_jobs(FakeConfig("dynamic", "direct"), None) == "direct"

    seen["compile"].clear()
    monkeypatch.setenv("SIRCL_EP_FORCE_GROUPED_ROUTING", "1")
    mod._patch_ep_grouped_routing()
    # _dynamic_direct_routing_selected stays unpatched so parent planning and
    # unpatched b12x compilation subprocesses agree on program keys.
    assert fake_impl._dynamic_direct_routing_selected is orig_sel
    assert fake_impl._DIRECT_ROUTING_MAX_ROUTED_ROWS == 32
    assert fake_impl._heuristic_dynamic_route_mode() == "grouped"

    leaf = fake_fm.plan_execution().variants[5]
    direct_cfg = FakeConfig("dynamic", "direct")
    assert leaf._compile_jobs(direct_cfg, None) == "grouped"
    assert leaf._memory_requirements(direct_cfg, None) == "grouped"
    assert leaf._materialize(FakeSelection(direct_cfg), None) == "grouped"
    assert [c.dynamic_route_mode for c in seen["compile"]] == ["grouped"]
    assert [c.dynamic_route_mode for c in seen["memory"]] == ["grouped"]
    assert [c.dynamic_route_mode for c in seen["materialize"]] == ["grouped"]

    # Non-dynamic configs (e.g. w4a16) pass through untouched.
    a16_cfg = FakeConfig("w4a16", None)
    assert leaf._compile_jobs(a16_cfg, None) is None


