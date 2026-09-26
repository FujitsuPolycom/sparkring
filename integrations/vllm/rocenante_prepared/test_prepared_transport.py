"""GPU-free interface and safety checks; CUDA/RDMA qualification is separate."""

from __future__ import annotations

import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parents[2] / "runtime/transport_profiles/tp2-rocenante-adaptive/roce"


def definitions(path, names, env=None):
    namespace = {} if env is None else env
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def runtime_method(name, env=None):
    tree = ast.parse((ROOT / "roce/roce_oneshot.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RoceOneshotAllReduce"
    )
    method = next(n for n in cls.body if getattr(n, "name", None) == name)
    method.decorator_list = []
    namespace = {} if env is None else env
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                method,
            ],
            type_ignores=[],
        )
    )
    exec(compile(module, name, "exec"), namespace)
    return namespace[name]


def c_definition(source, signature, terminator="\n}\n"):
    """Return one top-level C definition, from its signature through its terminator."""
    start = source.index(signature)
    end = source.index(terminator, start) + len(terminator)
    return source[start:end]


def test_legacy_wire_path_and_kernel_math_are_preserved():
    for name in (
        "_proxy.py",
        "_path_config.py",
        "_cute_intrinsics.py",
    ):
        assert (ROOT / "roce" / name).read_text().strip() == (
            LEGACY / name
        ).read_text().strip()
    # The prepared proxy paces hardware-forwarded stripes on the sending side;
    # peers still exchange the same connection blob, region layout, queue-pair
    # attributes and data/flag placement as the preserved adaptive proxy.
    prepared = (ROOT / "roce/_roce_proxy.c").read_text()
    legacy = (LEGACY / "_roce_proxy.c").read_text()
    abi = next(line for line in prepared.splitlines() if line.startswith("#define ROCE_ABI_VERSION"))
    assert abi in legacy.splitlines()
    for signature, terminator in (
        ("typedef struct {\n    uint32_t abi_version;", "} roce_blob_t;"),
        ("int roce_layout(", "\n}\n"),
        ("static int connect_qp(", "\n}\n"),
    ):
        assert c_definition(prepared, signature, terminator) == c_definition(
            legacy, signature, terminator
        )
    for name in ("_oneshot_cute.py", "_allgather_cute.py"):

        def classes(path):
            return [
                ast.dump(node)
                for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef)
            ]

        assert classes(ROOT / "roce" / name) == classes(LEGACY / name)
    for name in (
        "__init__",
        "_launcher_key",
        "_gather_launcher_key",
        "_counter_addresses",
        "_order_stream",
        "_mark_stream",
        "check_health",
    ):

        def method(path):
            cls = next(
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "RoceOneshotAllReduce"
            )
            return ast.dump(
                next(n for n in cls.body if getattr(n, "name", None) == name)
            )

        assert method(ROOT / "roce/roce_oneshot.py") == method(
            LEGACY / "roce_oneshot.py"
        ), name


def test_public_collectives_require_a_prepared_plan():
    for name in ("all_reduce", "all_gather"):
        fn = runtime_method(name)
        import inspect

        assert (
            inspect.signature(fn).parameters["plan"].default is inspect.Parameter.empty
        )
    tree = ast.parse((ROOT / "roce/roce_oneshot.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RoceOneshotAllReduce"
    )
    for name in (
        "_run_prepared_all_reduce",
        "_run_prepared_all_gather",
        "_launch_gather",
    ):
        fn = next(n for n in cls.body if getattr(n, "name", None) == name)
        assert not any(
            isinstance(n, ast.Call)
            and (
                getattr(n.func, "id", "") == "get_launcher"
                or getattr(n.func, "attr", "") == "get_launcher"
            )
            for n in ast.walk(fn)
        )


def test_reduce_waits_before_staging_and_marks_after_output_copy():
    calls = []

    class Tensor:
        shape = (128,)
        dtype = "bf16"
        device = "cuda:0"

        def __init__(self, address, label):
            self.address, self.label = address, label

        def data_ptr(self):
            return self.address

        def numel(self):
            return 128

        def element_size(self):
            return 2

        def is_contiguous(self):
            return True

        def copy_(self, other):
            calls.append(("copy", self.label, other.label))

    source, output = Tensor(17, "input"), Tensor(19, "output")
    scratch = (Tensor(32, "stage"), Tensor(48, "result"))
    fake_torch = NS(
        cuda=NS(
            device=lambda _: nullcontext(),
            is_current_stream_capturing=lambda: True,
            stream=lambda _: nullcontext(),
        )
    )
    env = definitions(
        ROOT / "roce/roce_oneshot.py", {"_grid_blocks"}, dict(PACK_BYTES=16)
    )
    env.update(torch=fake_torch, PACK_BYTES=16, _nullcontext=nullcontext)
    run = runtime_method("_run_prepared_all_reduce", env)
    runtime = NS(
        _lock=nullcontext(),
        check_health=lambda: None,
        should_allreduce=lambda _: True,
        device="cuda:0",
        _threads=128,
        _blocks=64,
        _aligned_scratch=lambda which, _: scratch[which],
        _order_stream=lambda _: calls.append(("wait",)),
        _mark_stream=lambda _: calls.append(("mark",)),
        _counter_addresses=lambda _: (100, 104),
        _recv_base=0,
        _flag_base=0,
        _send_base=0,
        _ctrl_base=0,
        _slot_bytes=0,
        _epoch_address=0,
        _poison_address=0,
        spin_limit=1,
    )
    prepared = NS(
        reduce_launchers={"bf16": lambda *args: calls.append(("launch", args))}
    )
    assert run(runtime, source, prepared=prepared, out=output) is output
    assert [event[0] for event in calls] == ["wait", "copy", "launch", "copy", "mark"]
    assert calls[2][1][-1] == 1


def test_gather_retains_legacy_cross_stream_ordering():
    fn = runtime_method("_run_prepared_all_gather")
    tree = ast.parse((ROOT / "roce/roce_oneshot.py").read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == fn.__name__
    )
    text = ast.get_source_segment((ROOT / "roce/roce_oneshot.py").read_text(), method)
    assert text.index(
        "self._order_stream(capturing)", text.index("staged, gathered")
    ) < text.index("staged[:nbytes].copy_")
    assert text.index(
        "self._mark_stream(capturing)", text.index("stacked =")
    ) > text.index("out.copy_(result)")


@pytest.mark.parametrize("paths", [2, 4])
def test_compile_uses_peer_paths_not_inventory_count(monkeypatch, paths):
    calls = []
    for name, values in {
        "test_roce._allgather_cute": {
            "get_launcher": lambda *args: calls.append(("gather", args)) or "gather"
        },
        "test_roce._oneshot_cute": {
            "get_launcher": lambda *args: calls.append(("reduce", args)) or "reduce"
        },
        "test_roce.roce_oneshot": {"_DTYPE_NAMES": {"bf16": "bfloat16"}},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    package = ModuleType("test_roce")
    package.__path__ = []
    package._allgather_cute = sys.modules["test_roce._allgather_cute"]
    monkeypatch.setitem(sys.modules, "test_roce", package)
    env = dict(
        __package__="test_roce",
        RoceQuery=lambda **kwargs: NS(**kwargs),
        torch=NS(bfloat16="bf16", cuda=NS(device=lambda _: nullcontext())),
    )
    env = definitions(ROOT / "roce/_preparation.py", {"compile_roce", "_dtypes"}, env)
    payload = dict(
        world_size=2,
        rank=0,
        call={"dtypes": ["bfloat16"]},
        setup=dict(
            threads=128, slots=2, flag_stride=4096, hca_count=4, opposite_paths=paths
        ),
    )
    result = env["compile_roce"](payload, 0)
    assert result == {"bf16": "reduce", "gather": "gather"}
    assert calls[0][1] == ("bfloat16", 2, 0, 128, 2, 4096, paths, 0)
    assert calls[1][1] == (2, 0, 128, 2, 4096, paths, 0)


def test_query_binds_exact_peer_map_and_capacity():
    env = definitions(
        ROOT / "roce/_preparation.py",
        {"_runtime_setup", "query_from_runtime"},
        dict(FrozenMapping=dict, RoceQuery=lambda **kwargs: NS(**kwargs)),
    )
    runtime = NS(
        world_size=2,
        rank=0,
        hca_names=("h0", "h1", "h2", "h3"),
        _threads=128,
        _layout=NS(slots=2, flag_stride=4096),
        _opposite_paths=2,
        _peer_hca_map=((-1, -1), (0, 2)),
        _blocks=64,
        max_size=2**20,
        max_gather_bytes=2**22,
        gid_index=3,
        spin_limit=1000,
    )
    query = env["query_from_runtime"](
        runtime,
        surface="AllReduce.all_reduce",
        call={"dtypes": ("bfloat16",)},
        topology="roce_rdma",
        peer_hosts=("r0", "r1"),
    )
    assert query.setup["peer_hca_map"] == ((-1, -1), (0, 2))
    assert query.setup["hca_count"] == 4 and query.setup["opposite_paths"] == 2
    assert query.setup["max_size"] == 2**20 and query.setup["max_blocks"] == 64


def test_native_callables_publish_program_identities():
    for name in ("_oneshot_cute.py", "_allgather_cute.py"):
        text = (ROOT / "roce" / name).read_text()
        assert "@program_cache\n" in text
        assert "return attach_programs(run, raw)" in text
        assert "comm.roce.sparkring_adaptive." in text


def test_package_stages_only_manifest_bound_files(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepared_roce_package", ROOT / "package.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.stage(tmp_path / "bundle")
    import json
    import hashlib

    manifest = json.loads((tmp_path / "bundle/manifest.json").read_text())
    files = {
        p.relative_to(tmp_path / "bundle").as_posix()
        for p in (tmp_path / "bundle").rglob("*")
        if p.is_file()
    }
    assert files == set(manifest["files"]) | {"manifest.json"}
    assert (
        result["manifest_sha256"]
        == hashlib.sha256((ROOT / "manifest.json").read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        module.stage(tmp_path / "bundle")


def _selector():
    spec = importlib.util.spec_from_file_location(
        "prepared_roce_selector", ROOT / "sparkring_transport_selector.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selector_preserves_legacy_profile_without_prepared_host_checks(monkeypatch):
    import hashlib

    selector = _selector()
    monkeypatch.setattr(selector, "_host_uses_prepared_api", lambda: False)
    monkeypatch.setattr(
        selector,
        "_verify_host_sources",
        lambda _: pytest.fail("legacy profile must not require prepared B12X"),
    )
    legacy_root = LEGACY.parent.parent
    digest = hashlib.sha256((LEGACY.parent / "manifest.json").read_bytes()).hexdigest()
    assert selector.verify_bundle(selector.PROFILE, digest, legacy_root) == LEGACY
    monkeypatch.setenv(selector.PROFILE_ENV, selector.PROFILE)
    monkeypatch.setenv(selector.DIGEST_ENV, digest)
    monkeypatch.setattr(sys, "meta_path", sys.meta_path[:])
    selector.install_from_environment(legacy_root)
    assert sys.meta_path[0].package == LEGACY


def test_selector_rejects_legacy_bundle_on_prepared_runtime(monkeypatch):
    import hashlib

    selector = _selector()
    monkeypatch.setattr(selector, "_host_uses_prepared_api", lambda: True)
    digest = hashlib.sha256((LEGACY.parent / "manifest.json").read_bytes()).hexdigest()
    with pytest.raises(
        ValueError, match="requires transport profile tp2-rocenante-adaptive-prepared"
    ):
        selector.verify_bundle(selector.PROFILE, digest, LEGACY.parent.parent)


def test_padded_pack_sized_gather_returns_independent_storage():
    import torch

    calls = []
    staging = torch.empty(64, dtype=torch.uint8)
    gathered = torch.empty(128, dtype=torch.uint8)
    proxy_torch = NS(
        cuda=NS(
            device=lambda _: nullcontext(),
            is_current_stream_capturing=lambda: False,
            stream=lambda _: nullcontext(),
        ),
        uint8=torch.uint8,
        contiguous_format=torch.contiguous_format,
    )
    env = definitions(ROOT / "roce/roce_oneshot.py", {"_align_up"}, dict(PACK_BYTES=16))
    env.update(torch=proxy_torch, PACK_BYTES=16, _nullcontext=nullcontext)
    run = runtime_method("_run_prepared_all_gather", env)

    def gather(*args):
        calls.append("launch")
        gathered[:64].copy_(staging)
        gathered[64:].copy_(staging)

    runtime = NS(
        _lock=nullcontext(),
        check_health=lambda: None,
        should_all_gather=lambda *a: True,
        _normalize_dim=lambda *a: 0,
        _direct_gather_layout=lambda *a: False,
        world_size=2,
        device="cpu",
        _gather_scratch=lambda _: (staging, gathered),
        _launch_gather=gather,
        _order_stream=lambda _: calls.append("wait"),
        _mark_stream=lambda _: calls.append("mark"),
    )
    first_input = torch.arange(33, dtype=torch.bfloat16)[1:]
    assert (
        first_input.data_ptr() % 16
        and first_input.numel() * first_input.element_size() == 64
    )
    first = run(runtime, first_input, prepared=NS(gather_launcher=object()), dim=0)
    saved = first.clone()
    second = run(runtime, first_input * 2, prepared=NS(gather_launcher=object()), dim=0)
    assert first.untyped_storage().data_ptr() != gathered.untyped_storage().data_ptr()
    torch.testing.assert_close(first, saved, rtol=0, atol=0)
    torch.testing.assert_close(second, saved * 2, rtol=0, atol=0)
    assert calls == ["wait", "launch", "mark"] * 2


@pytest.mark.parametrize("defect", ["none", "mismatch", "missing"])
def test_selector_checks_complete_prepared_host_source_contract(monkeypatch, defect):
    import hashlib

    selector = _selector()
    payload = b"controlled host source bytes"
    preimages = {
        selector.HOST_SOURCE_PREFIX + name: hashlib.sha256(payload).hexdigest()
        for name in selector.HOST_SOURCE_FILES
    }

    class Source:
        def __init__(self, name):
            self.name = name

        def is_symlink(self):
            return False

        def read_bytes(self):
            return payload if defect != "mismatch" else b"changed"

    monkeypatch.setattr(selector, "Path", Source)
    if defect == "missing":
        preimages.pop(next(iter(preimages)))
    if defect == "none":
        selector._verify_host_sources(preimages)
    else:
        with pytest.raises(ValueError):
            selector._verify_host_sources(preimages)


@pytest.mark.parametrize("valid_host", [True, False])
def test_selector_prepared_activation_is_explicit_and_fail_closed(
    tmp_path, monkeypatch, valid_host
):
    import hashlib

    package_spec = importlib.util.spec_from_file_location(
        "prepared_roce_packaging", ROOT / "package.py"
    )
    packaging = importlib.util.module_from_spec(package_spec)
    package_spec.loader.exec_module(packaging)
    selector = _selector()
    destination = tmp_path / selector.PREPARED_PROFILE
    packaging.stage(destination)
    calls = []

    def verify_host(preimages):
        calls.append(preimages)
        if not valid_host:
            raise ValueError("different prepared host runtime")

    monkeypatch.setattr(selector, "_verify_host_sources", verify_host)
    monkeypatch.setenv(selector.PROFILE_ENV, selector.PREPARED_PROFILE)
    monkeypatch.setenv(
        selector.DIGEST_ENV,
        hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest(),
    )
    original = sys.meta_path[:]
    monkeypatch.setattr(sys, "meta_path", original[:])
    if valid_host:
        selector.install_from_environment(tmp_path)
        spec = sys.meta_path[0].find_spec("b12x.comm.roce._preparation")
        assert Path(spec.origin) == destination / "roce/_preparation.py"
    else:
        with pytest.raises(SystemExit, match="activation failed"):
            selector.install_from_environment(tmp_path)
        assert sys.meta_path == original
    assert len(calls) == 1 and len(calls[0]) == 6


@pytest.mark.parametrize("world", [2, 4])
def test_transport_worlds_are_not_restricted_by_profile_name(world):
    import os

    path = ROOT / "roce/_path_config.py"
    env = definitions(
        path,
        {"peer_path_count", "peer_hca_map"},
        dict(
            os=os,
            PATH_COUNT=2,
            MAX_PATH_COUNT=4,
            PEER_HCA_MAP_ENV="SPARKRING_TEST_UNSET_PEER_MAP",
        ),
    )
    for rank in range(world):
        assert env["peer_hca_map"](world, rank, 2, 2) == tuple(
            (-1, -1) if peer == rank else (0, 1) for peer in range(world)
        )
    text = (ROOT / "roce/roce_oneshot.py").read_text()
    assert "SUPPORTED_WORLD_SIZES = tuple(range(2, 17))" in text
