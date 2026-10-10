"""Version pins, hook anchors, shim refusal, the four-rank gate and parity with the session package.

These tests run without vLLM. ``SIRCL_TEST_VLLM_ROOTS`` (paths of ``vllm``
package directories separated by the platform's path separator) additionally
checks every hook anchor and the pin status against real vLLM source trees.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import types
from pathlib import Path

import pytest
pytest.importorskip("torch")  # the general plugin imports torch

from sparkring_sircl.vllm import hooks, pins, platform, plugin, sessionapi, shims, tp4  # noqa: E402
from sparkring_sircl.vllm.fabric import Layout, describe_group  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[1] / "sparkring_sircl"
VECTORS = Path(__file__).resolve().parent / "data" / "routes.json"


def test_file_hash_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"a = 1\nb = 2\n")
    crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert pins.file_hash(lf) == pins.file_hash(crlf)


def test_every_pinned_build_names_every_hook_file():
    for build in pins.SUPPORTED:
        assert set(build.files) == set(pins.HOOK_FILES)
        # None: the build does not have the file (it must then be absent from a matching tree).
        assert all(value is None or (isinstance(value, str) and len(value) == 64)
                   for value in build.files.values())
    for name in pins.HOOK_FILES:
        assert any(build.files[name] is not None for build in pins.SUPPORTED)
    for hook in hooks.HOOKS:
        assert set(hook.builds) <= {build.name for build in pins.SUPPORTED}


def test_the_csf_overlay_build_differs_from_the_image_build_only_in_code_no_shim_relies_on():
    """The CSF overlay's vLLM (sparkring/kraken-beta-20261007 at bc9ea774) changes five hook files of the
    image's build; the functions the shims wrap or call in them are identical (pins.py), and its fused
    all-reduce + RMSNorm helper is the file the karmic-kraken-beta build pins for that shim."""
    builds = {build.name: build for build in pins.SUPPORTED}
    image, karmic = builds["lil-image-aba309e4610c"], builds["lil-karmic-kraken-beta-4a87c588"]
    csf = builds["sparkring-kraken-beta-20261007-bc9ea774"]
    assert {name for name in pins.HOOK_FILES if csf.files[name] != image.files[name]} == {
        "v1/attention/ops/dcp.py", "v1/worker/gpu_worker.py", "models/glm5next/nvidia/model.py",
        "models/common/ops/fused_allreduce_rms_norm.py", "distributed/device_communicators/b12x_pcie_all_reduce.py"}
    assert all(csf.files[name] == karmic.files[name] for name in pins.NORM_FILES)
    assert all(csf.files[name] is not None for shim in shims.SHIMS.values() for name in shim.files)
    applicable = {hook.shim for hook in hooks.applicable([csf.name]) if hook.shim}
    assert applicable == set(shims.SHIMS) - {None}


def _tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_shims_refuse_unpinned_vllm(tmp_path):
    root = _tree(tmp_path / "vllm", {"v1/attention/ops/dcp.py": "def dcp_a2a_lse_reduce(): pass\n"})
    with pytest.raises(shims.ShimRefused, match="refuses to load"):
        shims.install(["dcp_all_to_all"], root=root)
    with pytest.raises(shims.ShimRefused, match="unknown SIRCL shim"):
        shims.check(["no_such_shim"], root)
    assert "dcp_all_to_all" not in shims.installed()


def test_shims_accept_a_pinned_build(tmp_path, monkeypatch):
    text = "def dcp_a2a_lse_reduce(): pass\n"
    root = _tree(tmp_path / "vllm", {"v1/attention/ops/dcp.py": text})
    build = pins.VllmBuild("synthetic", "0", "test tree",
                           {"v1/attention/ops/dcp.py": pins.file_hash(root / "v1/attention/ops/dcp.py")})
    monkeypatch.setattr(pins, "SUPPORTED", (build,))
    assert shims.check(["dcp_all_to_all"], root) == {"dcp_all_to_all": "synthetic"}
    report = pins.check(root)
    assert report.supported and report.matches == ("synthetic",)


def test_hook_table_covers_every_collective_and_states_its_mechanism():
    collectives = " | ".join(hook.collective for hook in hooks.HOOKS)
    for name in ("Device communicator", "PyNccl construction", "TP all-reduce", "TP all-gather",
                 "latent", "reduce-scatter", "DCP all-gather", "DCP all-to-all", "Point-to-point",
                 "bypass", "Expert-parallel", "RoCE all-reduce slot", "Fused residual-add + RMSNorm",
                 "Post-step health check", "Plugin order", "mHC prefill row ownership"):
        assert name in collectives
    for hook in hooks.HOOKS:
        if hook.shim:
            assert hook.shim in shims.SHIMS and not hook.official
        assert hook.anchors and hook.status


def test_anchor_verification_on_a_synthetic_tree(tmp_path):
    anchor = hooks.Anchor("a.py", 3, 3, "needle")
    hook = hooks.Hook("x", "y", True, (anchor,), None, "", "")
    _tree(tmp_path, {"a.py": "1\n2\n3\nneedle\n"})
    assert hooks.verify(tmp_path, [hook])[0].found_at == 4
    _tree(tmp_path, {"a.py": "nothing\n"})
    assert hooks.verify(tmp_path, [hook])[0].found_at is None


def _roots():
    raw = os.environ.get("SIRCL_TEST_VLLM_ROOTS", "")
    return [Path(item) for item in raw.split(os.pathsep) if item]


@pytest.mark.skipif(not _roots(), reason="SIRCL_TEST_VLLM_ROOTS is not set")
@pytest.mark.parametrize("root", _roots(), ids=lambda path: path.parent.name)
def test_real_vllm_trees(root):
    report = pins.check(root)
    assert report.supported
    applicable = hooks.applicable(report.matches)
    missing = [(r.anchor.file, r.anchor.text) for r in hooks.verify(root, applicable) if r.found_at is None]
    assert missing == []


def test_platform_activation_is_opt_in(monkeypatch):
    monkeypatch.delenv("SIRCL_MODE", raising=False)
    assert platform.activate() is None
    plugin.register()                     # disabled: does nothing, needs no vLLM
    monkeypatch.setenv("SIRCL_MODE", "bogus")
    with pytest.raises(ValueError):
        platform.enabled()


def test_four_rank_sessions_need_a_four_spark_ring_and_the_pinned_adapter(monkeypatch):
    cycle = describe_group(Layout.ring(4), range(4))
    path = describe_group(Layout.ring(8), range(4))

    class Base:
        def all_reduce(self, tensor):
            return tensor

    assert "not installed" in tp4.unavailable_reason(cycle, kind="tp", communicator_class=Base)
    assert "four-rank tensor-parallel" in tp4.unavailable_reason(cycle, kind="dcp",
                                                                 communicator_class=Base)
    module = types.ModuleType("spark_tp4_backend")
    module.__file__ = str(Path(__file__))
    monkeypatch.setitem(sys.modules, "spark_tp4_backend", module)
    Base.all_reduce._spark_tp4_backend = True
    assert "pinned revision" in tp4.unavailable_reason(cycle, kind="tp", communicator_class=Base)
    monkeypatch.setattr(tp4, "PINNED_ADAPTER_SHA256", pins.file_hash(Path(__file__)))
    assert tp4.unavailable_reason(cycle, kind="tp", communicator_class=Base) is None
    assert "share no cable" in tp4.unavailable_reason(path, kind="tp", communicator_class=Base)
    assert tp4.conflicts(path, kind="tp", communicator_class=Base,
                         environ={"VLLM_SPARK_TP4_MODE": "custom"})
    assert not tp4.conflicts(cycle, kind="tp", communicator_class=Base,
                             environ={"VLLM_SPARK_TP4_MODE": "custom"})

    module._mode = lambda: "custom"
    module._eligible = lambda communicator, tensor, mode: tensor == "admitted"
    assert tp4.admits(object(), "admitted", capturing=False)
    assert not tp4.admits(object(), "other", capturing=False)


# -- parity with the ring-session package and its route module ----------------------------------

RUNTIME = PACKAGE / "oneshot" / "runtime.py"


@pytest.mark.skipif(not RUNTIME.is_file(), reason="the session package is not present")
def test_the_session_package_offers_every_method_and_keyword_the_adapter_uses():
    tree = ast.parse(RUNTIME.read_text(encoding="utf-8"))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    session = classes["RoceOneshotAllReduce"]
    methods = {node.name: node for node in session.body if isinstance(node, ast.FunctionDef)}
    protocol = [name for name in dir(sessionapi.RingSession) if not name.startswith("_")]
    attributes = {"rank", "world_size", "max_size", "dispatch_limit_bytes", "max_gather_bytes",
                  "lane_count", "hca_names", "scatter_available"}
    assigned = {target.attr for node in ast.walk(session) if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
                if isinstance(target, ast.Attribute)}
    for name in protocol:
        assert name in methods or name in attributes, name
    assert attributes <= assigned
    assert "links" in {arg.arg for arg in methods["prepare"].args.kwonlyargs}   # sessionapi.link_keywords
    keywords = {arg.arg for arg in methods["__init__"].args.kwonlyargs}
    assert {"exchange_group", "device", "max_size", "max_gather_bytes", "peer_routes", "algorithm",
            "layout"} <= keywords
    assert "from_exchange_group" in methods
    for protocol_class in (sessionapi.LargeMessageSession, sessionapi.RegimeSession):
        members = [name for name in dir(protocol_class) if not name.startswith("_")]
        annotated = set(getattr(protocol_class, "__annotations__", {}))
        for name in members:
            assert name in methods, name
        assert annotated <= assigned, annotated - assigned
    assert any(isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "API_VERSION" for t in node.targets)
               and getattr(node.value, "value", None) == sessionapi.REQUIRED_API_VERSION for node in tree.body)


def _core_routes():
    try:
        from sparkring_sircl import routes
    except ImportError:
        return None
    return routes if hasattr(routes, "derive_routes") and hasattr(routes, "Layout") else None


PLACEMENTS = [
    ("ring:8", [0, 1, 2, 3], None), ("ring:8", [4, 5, 6, 7], None), ("ring:8", list(range(8)), None),
    ("ring:8", [6, 7], None), ("ring:8", [7, 0], None), ("ring:8", [0, 1, 2, 3], list(range(8))),
    ("ring:8", [4, 5, 6, 7], list(range(8))), ("ring:8", [2, 3], list(range(8))),
    ("ring:8", [0, 4], list(range(8))), ("ring:8", [0, 2, 4, 6], list(range(8))),
    ("ring:4", [0, 1, 2, 3], None), ("ring:3", [0, 1, 2], None), ("pair:2", [0, 1], None),
    ("pair", [0, 1], None),
]


@pytest.mark.skipif(_core_routes() is None, reason="the core route module is not present")
@pytest.mark.parametrize("layout,members,parent", PLACEMENTS)
def test_route_maps_and_relay_load_equal_the_core_derivation(layout, members, parent):
    routes = _core_routes()
    topology = describe_group(Layout.parse(layout), members, parent=parent)
    core_layout = routes.Layout.parse(topology.session_layout())
    assert core_layout.identity()["positions"] == list(members)
    derived = routes.derive_routes(core_layout)
    for rank in range(len(members)):
        assert derived.route_map(rank) == topology.route_map(rank)
    assert routes.relay_load(derived)[1] == topology.relay_factor()


@pytest.mark.skipif(not VECTORS.is_file() or _core_routes() is None, reason="vectors or core absent")
def test_session_layout_text_round_trips_through_the_core_parser():
    routes = _core_routes()
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    vector = vectors["ring-of-8-tp4-4-7"]
    topology = describe_group(Layout.ring(8), vector["rank_positions"])
    parsed = routes.Layout.parse(topology.session_layout())
    assert parsed.identity()["cables"] == vector["cables"]
