"""The ``libsircl`` transport: its section, routing settings, containers, text and checks; offline.

The fabric documents are the simulated ones of ``test_transport``: a pair, a
path of four and an eight-Spark cycle.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from runtime.common import fabric_document, image_lock, installer, libsircl, transport
from runtime.common.test_image_lock import libsircl_lock, sircl_lock
from runtime.common.test_transport import TP2, TP4, document, install_site
from spark_transport.sircl.sparkring_sircl.vllm.serve import plan as serve_plan

TP8 = "glm53-flash-csf-tp8"
ROUTES = set(libsircl.ROUTE_VARIABLES)


def image(profile=TP2):
    if profile.endswith("tp8"):
        profiles = sorted({*installer.installer_image.default_lock()["profiles"], *installer.installer_image.SIRCL_ONLY})
        return libsircl_lock(profiles=profiles)
    return libsircl_lock()


def deployment(profile, shape, size, positions, *, value=None):
    """``(lock, section)`` of a libsircl deployment of ``profile`` on ``positions`` of a ``shape`` fabric."""
    value = value or image(profile)
    section = libsircl.section(value, document(shape, size), positions, dcp=transport.profile_dcp(profile))
    placement = None if list(positions) == list(range(size)) else positions
    lock = installer.make_lock(profile, install_site(len(positions), placement=placement), "1" * 40, "2" * 64,
                               image_runtime=image_lock.v2_view(value), transport=section)
    return lock, section


def tool(layout, *, lanes=2):
    """libsircl's tools/site_routes.py, run as its README says, with the DGX OS device names."""
    environment = {"PYTHONPATH": str(libsircl.SIRCL_ROOT), "PATH": os.environ.get("PATH", "")}
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    done = subprocess.run([sys.executable, "-B", str(libsircl.SITE_ROUTES), "--layout", layout, "--lanes", str(lanes),
                           "--json"], capture_output=True, text=True, env=environment, check=True)
    return [row["env"] for row in json.loads(done.stdout)["ranks"]]


def common_settings(spec, section):
    """Every rank's settings besides the routing settings."""
    environment, block = spec.environment, section["libsircl"]
    assert environment["SPARKRING_LIBSIRCL_LIBRARY"] == block["library"]["path"]
    assert environment["SPARKRING_LIBSIRCL_SHA256"] == block["library"]["sha256"]
    assert environment["LIBSIRCL_TRANSPORT"] == "verbs" and environment["SIRCL_GID_INDEX"] == "3"
    assert environment["LIBSIRCL_FAIL_STOP"] == "1" and environment["LIBSIRCL_NCCL_API_VERSION"] == "22705"
    assert environment["SIRCL_BOOTSTRAP_ADDR"] == environment["VLLM_HOST_IP"]
    # The collective paths that bypass PyNccl are off, PyNccl on.
    for key, setting in libsircl.INDEPENDENT_TRANSPORTS_OFF.items():
        assert environment[key] == setting
    assert list(spec.command).count("--disable-custom-all-reduce") == 1
    assert environment["LIBSIRCL_RECEIPT"] == "/run/sparkring/sircl/receipts/libsircl"
    plugins = environment["VLLM_PLUGINS"].split(",")
    assert plugins[:2] == ["b12x_loader", "sparkring_status"] and plugins[-1] == "libsircl" and "sircl" not in plugins
    for key, setting in serve_plan.DISABLED_TRANSPORTS.items():
        assert environment[key] == setting
    assert (environment["NCCL_DEBUG"], environment["NCCL_DEBUG_SUBSYS"]) == ("INFO", "INIT")
    # SIRCL's own adapter is not configured.
    assert not {key for key in environment if key.startswith("SIRCL_") and key not in ROUTES} - {
        "SIRCL_GID_INDEX", "SIRCL_BOOTSTRAP_ADDR", "SIRCL_ENABLED"}


# The environment of a pair, a path of four and a cycle of eight.

def test_a_pair_runs_pynccl_on_libsircl_with_each_ranks_route_map():
    lock, section = deployment(TP2, "pair", 2, [0, 1])
    assert section["backend"] == "libsircl" and section["status"] == "research-only"
    assert section["group"] == {"layout": "pair:1", "positions": [0, 1], "shape": "pair", "size": 2, "name": "pair",
                                "max_relays": 0, "lanes": 2, "cabling": "all"}
    specs = installer.specifications(lock)
    for rank, spec in enumerate(specs):
        environment = spec.environment
        common_settings(spec, section)
        assert {key: environment[key] for key in ROUTES if key in environment} == section["routes"][rank]
        assert environment["LIBSIRCL_POSITION"] == str(rank)
        assert environment["SIRCL_PEER_ROUTES"] == f"{1 - rank}=rocep1s0f0/roceP2p1s0f0"
        assert environment["LIBSIRCL_CHAIN_ORDER"] == "0,1" and environment["LIBSIRCL_RING_WINDOW"] == "0"
        assert "LIBSIRCL_FORWARD_WINDOWS" not in environment and "LIBSIRCL_P2P_WINDOWS" not in environment
        assert any(mount.target == transport.RECEIPT_TARGET and not mount.read_only for mount in spec.mounts)


@pytest.mark.parametrize("shape, size, positions", [("path", 4, [0, 1, 2, 3]), ("cycle", 8, [4, 5, 6, 7])])
def test_a_path_of_four_gets_forward_windows_on_its_relayed_lanes_and_the_ring_plan(shape, size, positions):
    lock, section = deployment(TP4, shape, size, positions)
    assert (section["group"]["name"], section["group"]["max_relays"], section["group"]["cabling"]) == ("path-4", 2,
                                                                                                       "none")
    specs = installer.specifications(lock)
    routes = [{key: spec.environment[key] for key in ROUTES if key in spec.environment} for spec in specs]
    assert routes == section["routes"]
    assert [row["LIBSIRCL_POSITION"] for row in routes] == ["0", "1", "2", "3"]
    assert all(row["LIBSIRCL_CHAIN_ORDER"] == "0,1,2,3" for row in routes)
    # Lanes that cross relays keep forward windows of 131,072 bytes; the ring's relayed lanes, from rank 3 to
    # rank 0 through two relays, keep 393,216 bytes.
    assert routes[0]["LIBSIRCL_FORWARD_WINDOWS"] == "2=131072/131072,3=131072/131072"
    assert routes[3]["LIBSIRCL_FORWARD_WINDOWS"] == "0=131072/131072,1=131072/131072"
    assert [row["LIBSIRCL_RING_WINDOW"] for row in routes] == ["0", "0", "0", "393216"]
    assert all(row["SIRCL_FORWARD_CHUNK_BYTES"] == "32768" for row in routes)
    # Point-to-point channels get a window on every relayed lane, after the session's forward windows: 65,536
    # bytes in 32 KiB chunks toward each peer two or three positions away.
    assert [row["LIBSIRCL_P2P_WINDOWS"] for row in routes] == [
        "2=65536/65536,3=65536/65536", "3=65536/65536", "0=65536/65536", "0=65536/65536,1=65536/65536"]
    assert all(row["SIRCL_P2P_CHUNK_BYTES"] == "32768" for row in routes)
    for spec in specs:
        common_settings(spec, section)


def test_a_cycle_of_eight_routes_every_rank_to_its_seven_peers():
    lock, section = deployment(TP8, "cycle", 8, list(range(8)))
    assert (section["group"]["name"], section["group"]["max_relays"], section["group"]["cabling"]) == ("cycle-8", 3,
                                                                                                       "ring")
    specs = installer.specifications(lock)
    assert len(specs) == 8
    for rank, spec in enumerate(specs):
        environment = spec.environment
        common_settings(spec, section)
        peers = libsircl.peer_routes(environment["SIRCL_PEER_ROUTES"])
        assert sorted(peers) == [peer for peer in range(8) if peer != rank]
        assert all(set(lanes) <= set(section["devices"][rank]) for lanes in peers.values())
        assert environment["LIBSIRCL_CHAIN_ORDER"] == "0,1,2,3,4,5,6,7"
        # Every ring edge of the whole cycle is a cable.
        assert environment["LIBSIRCL_RING_WINDOW"] == "0" and environment["LIBSIRCL_FORWARD_WINDOWS"]
        command = list(spec.command)
        assert command[command.index("--tensor-parallel-size") + 1] == "8"


@pytest.mark.parametrize("shape, size, positions, layout", [
    ("pair", 2, [0, 1], None), ("path", 4, [0, 1, 2, 3], "path:0-3"), ("cycle", 8, list(range(8)), "ring:8")])
def test_the_routing_settings_are_those_libsircls_own_tool_prints(shape, size, positions, layout):
    _, section = deployment(TP2 if size == 2 else TP4 if size == 4 else TP8, shape, size, positions)
    # The tool's own layout names for the same cabling (rank r at position r); the pair is a single cable.
    expected = tool(layout or section["planner"]["layout"])
    assert section["routes"] == expected
    assert section["planner"]["tool"] == "spark_transport/libsircl/tools/site_routes.py"


def test_the_route_maps_use_the_device_names_the_fabric_document_records():
    value = document("pair", 2)
    named = copy.deepcopy(value)
    for row in named["positions"]:
        for entry in row["ports"].values():
            for function in entry["functions"].values():
                function["rdma"] = "mlx5_" + function["rdma"]
    _, _, routes, _ = libsircl.group(named, [0, 1])
    assert routes[0]["SIRCL_PEER_ROUTES"] == "1=mlx5_rocep1s0f0/mlx5_roceP2p1s0f0"


# Where it runs, and refusals.

def test_libsircl_is_never_chosen_by_default_and_runs_only_on_an_image_that_carries_it():
    pair = document("pair", 2)
    assert transport.choose(image(), pair) == ("sircl", "never", None)
    assert transport.choose(image(), pair, backend="libsircl") == ("libsircl", None, None)
    assert "libsircl" in transport.BACKENDS
    with pytest.raises(transport.TransportError, match="carries no libsircl layer"):
        transport.choose(sircl_lock(), pair, backend="libsircl")
    with pytest.raises(transport.TransportError, match="--nccl applies to deployments on SIRCL"):
        transport.choose(image(), pair, backend="libsircl", nccl="auto")


def renamed(value, position):
    """``value`` with position ``position``'s port 0 primary device renamed."""
    value = copy.deepcopy(value)
    value["positions"][position]["ports"]["0"]["functions"]["primary"]["rdma"] = "other0"
    return value


@pytest.mark.parametrize("value, message", [
    (None, "no fabric document"),
    ({**document("cycle", 4), "transports": ["prepared"]}, "relay table is not installed"),
    (renamed(document("cycle", 4), 1), "name their fabric devices differently"),
])
def test_a_fabric_libsircl_cannot_route_is_refused(value, message):
    with pytest.raises(transport.TransportError, match=message):
        transport.choose(image(), value, backend="libsircl")


@pytest.mark.parametrize("shape, size, positions, message", [
    ("cycle", 8, [0, 1, 2, 3, 4, 5], "through 4 relays"),
    ("path", 6, list(range(6)), "through 4 relays"),
    ("cycle", 8, [0, 2], "not consecutive"),
])
def test_groups_beyond_sircls_placement_rules_are_refused(shape, size, positions, message):
    with pytest.raises(transport.TransportError, match=message):
        libsircl.section(image(), document(shape, size), positions)


def test_decode_context_parallelism_is_refused():
    with pytest.raises(transport.TransportError, match="decode-context parallelism 4"):
        deployment("glm53-nvfp4-tp8", "cycle", 8, list(range(8)))


@pytest.mark.parametrize("edit, message", [
    (lambda spec: {**spec.environment, "LIBSIRCL_POSITION": "5"}, "LIBSIRCL_POSITION"),
    (lambda spec: {**spec.environment, "LIBSIRCL_FAIL_STOP": "0"}, "LIBSIRCL_FAIL_STOP"),
    (lambda spec: {**spec.environment, "VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "1"}, "VLLM_DISTRIBUTED_USE_SPLIT_GROUP"),
])
def test_a_profile_whose_settings_bypass_pynccl_or_set_the_transports_variables_is_refused(monkeypatch, edit, message):
    lock, _ = deployment(TP2, "pair", 2, [0, 1])
    original = installer.installer_image.adapt

    def adapt(spec, value, **options):
        result = original(spec, value, **options)
        return result.__class__(**{**result.__dict__, "environment": edit(result)})
    monkeypatch.setattr(installer.installer_image, "adapt", adapt)
    with pytest.raises(transport.TransportError, match=message):
        installer.specifications(lock)


def test_the_profiles_own_switches_of_the_bypassing_paths_are_replaced(monkeypatch):
    lock, _ = deployment(TP2, "pair", 2, [0, 1])
    original = installer.installer_image.adapt

    def adapt(spec, value, **options):
        result = original(spec, value, **options)
        return result.__class__(**{**result.__dict__, "environment": {
            **result.environment, "VLLM_DISABLE_PYNCCL": "1", "VLLM_ALLREDUCE_USE_FLASHINFER": "1"},
            "command": (*result.command, "--disable-custom-all-reduce")})
    monkeypatch.setattr(installer.installer_image, "adapt", adapt)
    for spec in installer.specifications(lock):
        assert spec.environment["VLLM_DISABLE_PYNCCL"] == spec.environment["VLLM_ALLREDUCE_USE_FLASHINFER"] == "0"
        assert list(spec.command).count("--disable-custom-all-reduce") == 1


@pytest.mark.parametrize("flags, message", [
    (("--enable-sleep-mode",), "ncclCommSuspend"),
    (("--pipeline-parallel-size", "4"), "pipeline parallelism 4"),
    (("--data-parallel-size=2",), "data parallelism"),
    (("--enable-expert-parallel",), "default all-to-all backend"),
    (("--enable-expert-parallel", "--all2all-backend", "naive"), "naive all-to-all backend"),
    (("--compilation-config", '{"pass_config": {"enable_sp": true}}'), "sequence-parallelism"),
])
def test_vllm_features_libsircl_does_not_carry_are_refused(flags, message):
    problems = libsircl.capability_problems(["serve", "/model", *flags])
    assert len(problems) == 1 and message in problems[0]


def test_two_pipeline_stages_and_expert_parallelism_on_grouped_all_gathers_pass():
    assert libsircl.capability_problems(["serve", "-pp", "2", "--enable-expert-parallel", "--all2all-backend",
                                         "allgather_reducescatter"]) == []


def test_a_library_without_the_fail_stop_mode_is_refused():
    value = image()
    value["libsircl"]["fail_stop"] = False
    with pytest.raises(transport.TransportError, match=r"the image's libsircl \(snapshot [0-9a-f]{8}\) has no fail-stop mode .LIBSIRCL_FAIL_STOP."):
        transport.choose(value, document("pair", 2), backend="libsircl")
    _, section = deployment(TP2, "pair", 2, [0, 1])
    section["libsircl"]["fail_stop"] = False
    with pytest.raises(transport.TransportError, match="no fail-stop mode"):
        libsircl.environment(section, {}, ("serve",))


@pytest.mark.parametrize("flags, message", [(("--enable-eplb",), "enable-eplb"),
                                            (("--enable-batch-sharded-sampling",), "batch-sharded-sampling"),
                                            (("--enable-dbo",), "micro-batching")])
def test_vllm_arguments_whose_collectives_bypass_pynccl_are_refused(monkeypatch, flags, message):
    lock, _ = deployment(TP2, "pair", 2, [0, 1])
    original = installer.installer_image.adapt

    def adapt(spec, value, **options):
        result = original(spec, value, **options)
        return result.__class__(**{**result.__dict__, "command": (*result.command, *flags)})
    monkeypatch.setattr(installer.installer_image, "adapt", adapt)
    with pytest.raises(transport.TransportError, match=message):
        installer.specifications(lock)


# The section in the lock.

def test_the_lock_records_the_section_and_its_identity_covers_it():
    lock, section = deployment(TP2, "pair", 2, [0, 1])
    assert lock["transport"] == section and installer.validate(lock) == lock
    plain = installer.make_lock(TP2, install_site(2), "1" * 40, "2" * 64, image_runtime=image_lock.v2_view(image()))
    other = copy.deepcopy(section)
    other["libsircl"]["library"]["sha256"] = "6" * 64
    rebuilt = installer.make_lock(TP2, install_site(2), "1" * 40, "2" * 64, image_runtime=image_lock.v2_view(image()),
                                  transport=other)
    assert len({lock["id"], plain["id"], rebuilt["id"]}) == 3
    assert libsircl.request_identity(section) == {"backend": "libsircl", "fabric": section["fabric"]["id"],
                                                  "library": section["libsircl"]["library"]["sha256"]}


@pytest.mark.parametrize("edit, message", [
    (lambda section: section.update(status="qualified"), "status research-only"),
    (lambda section: section["routes"][1].update(LIBSIRCL_POSITION="0"), "routing settings"),
    (lambda section: section["routes"][0].update(SIRCL_PEER_ROUTES="1=mlx5_9/mlx5_8"), "own RDMA devices"),
    (lambda section: section["routes"][0].update(LIBSIRCL_DEBUG="1"), "routing settings"),
    (lambda section: section["group"].update(max_relays=1), "differs from its layout"),
    (lambda section: section["planner"].update(layout="path:0-1"), "explicit layout"),
    (lambda section: section["libsircl"]["library"].update(path="/tmp/libnccl.so.2"), "library is"),
])
def test_a_section_that_disagrees_with_the_lock_is_refused(edit, message):
    _, section = deployment(TP2, "pair", 2, [0, 1])
    edit(section)
    with pytest.raises(ValueError, match=message):
        installer.make_lock(TP2, install_site(2), "1" * 40, "2" * 64, image_runtime=image_lock.v2_view(image()),
                            transport=section)


# Text, admission and host checks.

def test_the_plan_says_what_carries_the_collectives_and_that_it_is_research_only():
    _, section = deployment(TP4, "cycle", 8, [4, 5, 6, 7])
    lines = transport.plan_lines(section, ["a note"])
    assert lines[0] == ("Transport: libsircl (research-only): vLLM's PyNccl carries its collectives on libsircl 0.6.0; "
                        "SIRCL's adapter and RoCEnante are off")
    assert lines[1] == "  libsircl group: path-4 at positions 4, 5, 6, 7; 2 lanes per peer, at most 2 relays on a lane"
    assert lines[2] == (f"  Library: /opt/sparkring/libsircl/lib/libsircl.so.0.6.0, SHA-256 {'7' * 12}, snapshot "
                        "bbbbbbbb; fail-stop on (LIBSIRCL_FAIL_STOP=1)")
    assert lines[3].startswith("  Off: vLLM's custom all-reduce, torch and NCCL symmetric memory")
    assert lines[4].startswith("  Research-only: no serving A/B has measured libsircl")
    # On a path, torch's NVIDIA NCCL cannot connect the ends, which share no cable.
    assert lines[5].startswith("  Note: torch.distributed's NVIDIA NCCL cannot connect") and lines[6] == "  Note: a note"
    _, pair = deployment(TP2, "pair", 2, [0, 1])
    assert not any("cannot connect" in line for line in transport.plan_lines(pair))


class Image:
    """Files of an image with the libsircl layer as isolated ``cat`` containers read them."""

    def __init__(self, section):
        block = section["libsircl"]
        layer = {"schema": libsircl.LAYER_SCHEMA, "version": block["version"], "snapshot": block["snapshot"],
                 "nccl_api_version": block["nccl_api_version"],
                 "fail_stop": block["fail_stop"], "library": {**block["library"], "soname": "libnccl.so.2"},
                 "plugin": block["plugin"],
                 "files": {block["library"]["path"]: block["library"]["sha256"],
                           block["plugin"]["path"]: block["plugin"]["sha256"]}}
        self.layer = json.dumps(layer).encode()
        block["receipt"]["sha256"] = hashlib.sha256(self.layer).hexdigest()
        self.base = json.dumps({"files": {**layer["files"],
                                          image_lock.LIBSIRCL_RECEIPT: block["receipt"]["sha256"]}}).encode()

    def __call__(self, argv, text=True):
        data = self.base if argv[-1] == installer.installer_image.PARENT_RECEIPT else self.layer
        return type("Done", (), {"stdout": data})()


def test_the_libsircl_layer_is_admitted_only_when_the_images_verification_covers_it():
    lock, section = deployment(TP2, "pair", 2, [0, 1])
    lock = copy.deepcopy(lock)
    built = Image(lock["transport"])
    lock["image_runtime"]["parent_receipt_sha256"] = hashlib.sha256(built.base).hexdigest()
    assert transport.admit_layer(lock, run=built)["files_verified"] == 2
    lock["transport"]["libsircl"]["plugin"]["sha256"] = "0" * 64
    with pytest.raises(transport.TransportError, match="vLLM plugin"):
        transport.admit_layer(lock, run=built)


def test_a_spark_must_hold_the_fabric_document_the_routes_were_planned_from(tmp_path):
    _, section = deployment(TP2, "pair", 2, [0, 1])
    path = tmp_path / fabric_document.HOST_PATH.lstrip("/")
    with pytest.raises(transport.TransportError, match="cannot be used"):
        transport.check_host_document(section, root=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(fabric_document.encoded(document("pair", 2)))
    assert transport.check_host_document(section, root=tmp_path)["ok"]
    path.write_text(fabric_document.encoded(document("cycle", 4)))
    with pytest.raises(transport.TransportError, match="Run sudo sparkring install again"):
        transport.check_host_document(section, root=tmp_path)


def test_the_receipt_check_reports_that_libsircl_receipts_are_not_judged(tmp_path):
    from runtime.host import transport_receipts
    lock, section = deployment(TP2, "pair", 2, [0, 1])
    verdict = transport_receipts.check(tmp_path, lock, runner=None, now=lambda: 1_791_000_000)
    assert verdict["verdict"] == "unknown" and verdict["backend"] == "libsircl"
    assert verdict["receipts"] == transport.receipt_directory(lock)
    assert transport_receipts.text(verdict).startswith("Transport: libsircl (research-only); receipts not judged")
    assert not (Path(tmp_path) / transport_receipts.LATEST).exists()
    assert libsircl.summary(section)["expected"] == libsircl.EXPECTED
