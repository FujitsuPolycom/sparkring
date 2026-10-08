"""The transport section of a deployment lock and the SIRCL adapter; offline.

The adapter must give every rank the SIRCL settings that the SIRCL
launcher's ``serve.plan.build_plan`` gives the same group, except where the
image layer replaces a launcher mechanism (installed package, prebuilt
libraries, receipt directory, fabric document). The parity tests build that
launcher plan from the installer's own containers and compare variable for
variable.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from runtime.common import fabric_document, fabric_layout, image_lock, installer, transport
from runtime.common.test_image_lock import sircl_lock
from runtime.host import fabric, relays
from runtime.host.test_fabric_layouts import plan_of
from spark_transport.sircl.sparkring_sircl.ring.site import Host, Site
from spark_transport.sircl.sparkring_sircl.vllm.serve import plan as serve_plan
from spark_transport.sircl.sparkring_sircl.vllm.serve import profile as serve_profile

MARKER = {"binary": relays.MARKER_BINARY, "sha256": "ab" * 32}
TP2 = "glm53-flash-nvfp4-spark-tp2"
TP4 = "glm53-flash-nvfp4-spark-tp4"
# Variables whose value the image layer, not the launcher, decides.
LAUNCHER_ONLY = {"PYTHONPATH", "SIRCL_RECEIPT_DIR", "SIRCL_BUILD_CACHE_DIR", "SIRCL_NATIVE_LIBRARY"}
ADAPTER_ONLY = {"SIRCL_RECEIPT_DIR", "SIRCL_BUILD_CACHE_DIR", "SIRCL_NATIVE_LIBRARY", "SIRCL_P2P_NATIVE_LIBRARY",
                "SIRCL_FABRIC_DOCUMENT"}


def document(shape, size, *, marker=MARKER):
    value, _ = fabric.prepare(plan_of(fabric_layout.layout(shape, size)), cluster="test", marker=marker)
    return value


def install_site(nodes, *, relay_reference=True):
    hosts = []
    for number in range(nodes):
        row = {"host": f"spark{number}", "management_ip": f"192.0.2.{20 + number}",
               "fabric_ip": f"192.0.2.{20 + number}", "interface": "enp1s0f0np0"}
        if nodes == 4:
            row["fabric"] = {"site_path": fabric_document.HOST_PATH, "site_sha256": "1" * 64,
                             "plan_sha256": "2" * 64}
        else:
            row["fabric_ip"] = f"198.18.20.{number + 1}"
        hosts.append(row)
    return {"schema": "sparkring-install-site/v1", "name": "sircltest", "hosts": hosts}


def sircl_deployment(profile, shape, size, positions, *, nccl="never", tuning=None, image=None):
    image = image or sircl_lock()
    section = transport.section(image, document(shape, size), positions, nccl=nccl,
                                tuning=tuning or transport.load_tuning())
    lock = installer.make_lock(profile, install_site(len(positions)), "1" * 40, "2" * 64,
                               image_runtime=image_lock.v2_view(image), transport=section)
    return lock, section


# The tuning table.

def test_the_default_tuning_table_is_canonical_and_chooses_among_sircl_settings_only():
    table = transport.load_tuning()
    raw = transport.TUNING_DEFAULTS.read_bytes().replace(b"\r\n", b"\n")
    assert raw.decode() == transport.encoded(table)
    assert transport.tuning_digest(table) == hashlib.sha256(raw).hexdigest()
    assert table["source"] == "defaults" and table["fabric"] is None and table["tables"] == []
    for row in table["layouts"].values():
        assert set(row["settings"]) <= set(transport.SETTINGS)
    from spark_transport.sircl.sparkring_sircl import __version__
    from spark_transport.sircl.sparkring_sircl.oneshot._proxy import ABI_VERSION
    assert table["sircl"] == {"version": __version__, "abi_version": ABI_VERSION}


@pytest.mark.parametrize("shape, size, expected", [("pair", 2, "pair"), ("path", 4, "path"), ("cycle", 8, "cycle-8"),
                                                   ("cycle", 4, "cycle"), ("cycle", 6, "cycle")])
def test_a_group_takes_its_own_row_else_the_row_of_its_shape(shape, size, expected):
    assert transport.tuning_row(transport.load_tuning(), shape, size)[0] == expected


@pytest.mark.parametrize("edit, message", [
    (lambda table: table["layouts"]["pair"]["settings"].update(nccl_bands=1), "not a SIRCL session setting"),
    (lambda table: table["layouts"]["pair"]["settings"].update(large_schedule="tree"), "not a SIRCL session"),
    (lambda table: table["layouts"].update(ring={"source": "rules", "settings": {}}), "rows are pair"),
    (lambda table: table.update(source="measured"), "measured on"),
    (lambda table: table["tables"].append({"path": "missing.json", "sha256": "0" * 64}), "missing"),
])
def test_a_malformed_tuning_table_is_refused(edit, message):
    table = transport.load_tuning()
    edit(table)
    with pytest.raises(transport.TransportError, match=message):
        transport.validate_tuning(table)


def test_a_measured_table_applies_only_to_its_own_fabric_and_image(tmp_path):
    cycle = document("cycle", 4)
    image = sircl_lock()
    measured = dict(transport.load_tuning(), source="measured", fabric=cycle["id"], image_id=image["image_id"])
    (tmp_path / transport.MEASURED_TUNING).write_text(transport.encoded(measured))
    assert transport.tuning_in_effect(tmp_path, cycle, image) == (measured, [])
    table, notes = transport.tuning_in_effect(tmp_path, cycle, dict(image, image_id="sha256:" + "9" * 64))
    assert table == transport.load_tuning() and "another fabric or image" in notes[0]


# Choosing the transport.

def test_sircl_is_the_default_wherever_the_image_and_the_fabric_carry_it():
    assert transport.choose(sircl_lock(), document("cycle", 4)) == ("sircl", "never", None)
    assert transport.choose(sircl_lock(), document("pair", 2), nccl="topology") == ("sircl", "auto", None)


@pytest.mark.parametrize("image, fabric_value, reason", [
    (lambda: installer.installer_image.default_lock(), lambda: document("cycle", 4), "carries no SIRCL layer"),
    (sircl_lock, lambda: None, "no fabric document"),
    (sircl_lock, lambda: document("cycle", 4, marker=None), "relay table is not installed"),
])
def test_without_sircl_the_prepared_transport_runs_and_says_why(image, fabric_value, reason):
    backend, nccl, why = transport.choose(image(), fabric_value())
    assert (backend, nccl) == ("prepared", None) and reason in why
    with pytest.raises(transport.TransportError, match="--transport sircl cannot run here: .*" + reason):
        transport.choose(image(), fabric_value(), backend="sircl")


def test_devices_named_differently_on_one_spark_keep_sircl_off():
    value = document("pair", 2)
    value["positions"][1]["ports"]["0"]["functions"]["primary"]["rdma"] = "mlx5_7"
    assert "name their fabric devices differently" in transport.choose(sircl_lock(), value)[2]


def test_nccl_choices_apply_to_sircl_only():
    with pytest.raises(transport.TransportError, match="--nccl applies"):
        transport.choose(sircl_lock(), document("pair", 2), backend="prepared", nccl="auto")
    with pytest.raises(transport.TransportError, match="never or auto"):
        transport.nccl_mode("ring")


# The section of the deployment lock.

@pytest.mark.parametrize("shape, size, positions, layout, name, devices", [
    ("pair", 2, [0, 1], "pair:1", "pair", [["rocep1s0f0", "roceP2p1s0f0"]] * 2),
    ("cycle", 4, [2, 3], "ring:4", "pair", [["rocep1s0f0", "roceP2p1s0f0"], ["rocep1s0f1", "roceP2p1s0f1"]]),
    ("cycle", 4, [0, 1, 2, 3], "ring:4", "cycle-4", None),
])
def test_the_section_places_the_group_on_the_recorded_fabric(shape, size, positions, layout, name, devices):
    _, section = sircl_deployment(TP2 if len(positions) == 2 else TP4, shape, size, positions)
    group = section["group"]
    assert (group["layout"], group["positions"], group["name"]) == (layout, positions, name)
    assert section["fabric"]["id"] == document(shape, size)["id"]
    assert section["sircl"] == image_lock.sircl(sircl_lock())
    if devices is not None:
        assert section["devices"] == devices
    else:
        assert all(sorted(row) == sorted(fabric_layout.DEVICES.values()) for row in section["devices"])
    expected_row = "pair" if name == "pair" else "cycle"
    assert section["tuning"]["row"] == expected_row and section["tuning"]["applies"]


def test_device_names_come_from_the_fabric_document():
    value = document("pair", 2)
    for row in value["positions"]:
        row["ports"]["0"]["functions"]["primary"]["rdma"] = "mlx5_0"
    section = transport.section(sircl_lock(), value, [0, 1], nccl="never", tuning=transport.load_tuning())
    assert section["devices"] == [["mlx5_0", "roceP2p1s0f0"]] * 2


def test_a_default_table_for_another_sircl_build_leaves_the_rules_in_charge():
    image = sircl_lock(sircl=dict(sircl_lock()["sircl"], version="0.3.0",
                                  wheel={"name": "sparkring_sircl-0.3.0-py3-none-any.whl", "sha256": "1" * 64},
                                  tuning_key={**sircl_lock()["sircl"]["tuning_key"], "sircl": "0.3.0/abi9"}))
    _, section = sircl_deployment(TP2, "pair", 2, [0, 1], image=image)
    assert section["tuning"]["applies"] is False and section["tuning"]["settings"] == {}
    assert "another SIRCL build" in transport.plan_lines(section)[0]


def test_the_lock_records_the_section_and_its_identity_covers_it():
    lock, section = sircl_deployment(TP2, "pair", 2, [0, 1])
    assert lock["transport"] == section and installer.validate(lock) == lock
    auto, _ = sircl_deployment(TP2, "pair", 2, [0, 1], nccl="auto")
    plain = installer.make_lock(TP2, install_site(2), "1" * 40, "2" * 64,
                                image_runtime=image_lock.v2_view(sircl_lock()))
    assert len({lock["id"], auto["id"], plain["id"]}) == 3 and "transport" not in plain


@pytest.mark.parametrize("edit, message", [
    (lambda section: section.update(nccl="ring"), "never or auto"),
    (lambda section: section.update(image="another-image"), "installer image"),
    (lambda section: section["group"].update(positions=[1, 2]), "outside the pair:1 layout"),
    (lambda section: section["group"].update(max_relays=1), "differs from its layout"),
    (lambda section: section["devices"].pop(), "each rank's RDMA devices"),
    (lambda section: section["tuning"]["settings"].update(oneshot_max=-16), "not passed"),
    (lambda section: section["sircl"].update(abi_version=0), "native ABI"),
])
def test_a_section_that_disagrees_with_the_lock_is_refused(edit, message):
    _, section = sircl_deployment(TP2, "pair", 2, [0, 1])
    edit(section)
    with pytest.raises(ValueError, match=message):
        installer.make_lock(TP2, install_site(2), "1" * 40, "2" * 64, image_runtime=image_lock.v2_view(sircl_lock()),
                            transport=section)


def test_a_managed_backend_cannot_run_on_sircl():
    _, section = sircl_deployment(TP2, "pair", 2, [0, 1])
    with pytest.raises(ValueError):
        installer.make_lock(installer.GLM_LEGACY[2], install_site(2), "1" * 40, "2" * 64, transport=section)


# The adapter.

def launcher_plan(lock, nccl):
    """The SIRCL launcher's plan for the same group, from the installer's adapted containers without SIRCL."""
    plain = installer.specifications({key: value for key, value in lock.items() if key != "transport"})
    section = lock["transport"]
    ranks = []
    for number, spec in enumerate(plain):
        health = serve_profile.Health(spec.health_command, "10s", "5s", "0s", 3) if spec.health_command else None
        ranks.append(serve_profile.RankContainer(
            rank=number, image_reference=spec.image_id, entrypoint=spec.entrypoint, command=spec.command,
            environment=dict(spec.environment),
            mounts=tuple(serve_profile.Mount(m.source, m.target, m.read_only) for m in spec.mounts),
            devices=spec.devices, security_opt=spec.security_opt, memory=spec.memory, memory_swap=spec.memory_swap,
            memlock=spec.memlock, platform=spec.platform, pull_policy=spec.pull_policy, restart=spec.restart_policy,
            init=spec.init, network_mode=spec.network_mode, ipc_mode=spec.ipc_mode, health=health))
    command = list(plain[0].command)
    arguments = tuple(command[command.index("--master-addr") + 2:])
    profile = serve_profile.ServingProfile(
        id=lock["selection"]["profile"], repository=installer.ROOT, tensor_parallel=len(plain),
        served_model_name="model", api_port=int(command[command.index("--port") + 1]),
        master_port=int(command[command.index("--master-port") + 1]),
        max_num_batched_tokens=int(command[command.index("--max-num-batched-tokens") + 1]),
        image_reference=plain[0].image_id, image_id=plain[0].image_id, release="test", model_repository="m/m",
        model_revision="0" * 40, config_sha256="0" * 64, index_sha256="0" * 64, checkpoint_files=(), ranks=tuple(ranks),
        recipe_environment={}, recipe_arguments=arguments, request_settings={}, topology="test", seccomp_policy=b"{}",
        sources={})
    size = section["fabric"]["size"]
    site = Site(image=plain[0].image_id, lan_interface="eth0", control_port=29600, remote_dir="/srv/sircl",
                ring=tuple(Host(f"spark{p}", f"spark{p}", f"192.0.2.{30 + p}") for p in range(size)))
    settings = section["tuning"]["settings"]
    options = serve_plan.Options(
        positions=tuple(section["group"]["positions"]), model_path="/srv/model", nccl_mode=nccl,
        require_no_nccl=nccl == "never",
        **{key: settings[key] for key in ("oneshot_max", "large_blocks", "ring_min", "chain_min") if key in settings})
    return serve_plan.build_plan(site, profile, options, staged_digest="0" * 16, library="roce_proxy-" + "0" * 16 + ".so")


@pytest.mark.parametrize("profile, shape, size, positions, nccl", [
    (TP4, "cycle", 4, [0, 1, 2, 3], "never"),
    (TP4, "cycle", 4, [0, 1, 2, 3], "auto"),
    (TP2, "cycle", 4, [2, 3], "never"),
    (TP2, "cycle", 4, [0, 1], "auto"),
    ("qwen38-flash-next-qad-tp4", "cycle", 4, [0, 1, 2, 3], "never"),
    ("mimo-v26-flash-mopd-tp2", "cycle", 4, [2, 3], "never"),
])
def test_every_rank_gets_the_settings_the_sircl_launcher_gives_the_same_group(profile, shape, size, positions, nccl):
    lock, _ = sircl_deployment(profile, shape, size, positions, nccl=nccl)
    adapted = installer.specifications(lock)
    planned = launcher_plan(lock, nccl)
    for spec, launch in zip(adapted, planned.ranks, strict=True):
        ours, theirs = spec.environment, launch.environment
        sircl = {key for key in theirs if key.startswith("SIRCL_")} - LAUNCHER_ONLY
        assert {key for key in ours if key.startswith("SIRCL_")} - ADAPTER_ONLY == sircl
        shared = sircl | set(serve_plan.DISABLED_TRANSPORTS) | {"NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_IB_HCA",
                                                               serve_plan.MHC_SHARD}
        assert {key: ours.get(key) for key in shared} == {key: theirs.get(key) for key in shared}
        assert ours["VLLM_PLUGINS"].split(",")[-1] == "sircl" and theirs["VLLM_PLUGINS"].endswith(",sircl")


def test_the_containers_load_the_images_prebuilt_libraries_and_the_sparks_fabric_document():
    lock, section = sircl_deployment(TP4, "cycle", 4, [0, 1, 2, 3])
    for spec in installer.specifications(lock):
        environment = spec.environment
        assert environment["SIRCL_NATIVE_LIBRARY"] == section["sircl"]["native"]["path"]
        assert environment["SIRCL_P2P_NATIVE_LIBRARY"] == section["sircl"]["p2p"]["path"]
        assert environment["SIRCL_BUILD_CACHE_DIR"] == image_lock.LIBRARY_DIRECTORY
        assert environment["SIRCL_FABRIC_DOCUMENT"] == fabric_document.HOST_PATH
        assert environment["SIRCL_NCCL"] == "never" and environment["SPARK_TP4_ENABLED"] == "0"
        assert environment["VLLM_ENABLE_ROCE_ALLREDUCE"] == "0" and environment["SPARKRING_TRANSPORT_PROFILE"] == ""
        assert "PYTHONPATH" not in environment
        mounts = {mount.target: mount for mount in spec.mounts}
        assert mounts[fabric_document.HOST_PATH].read_only and mounts[fabric_document.HOST_PATH].source == \
            fabric_document.HOST_PATH
        receipts = mounts[transport.RECEIPT_TARGET]
        assert not receipts.read_only and receipts.source == "/srv/sparkring/sircltest/sircl/receipts"
        assert "SIRCL_TUNING_TABLE" not in environment


def test_a_pair_tuning_row_sets_its_session_settings():
    lock, section = sircl_deployment(TP2, "pair", 2, [0, 1])
    environment = installer.specifications(lock)[0].environment
    assert section["tuning"]["settings"] == {"large_blocks": 32, "oneshot_max": 131072, "ring_min": 2097152}
    assert (environment["SIRCL_LARGE_BLOCKS"], environment["SIRCL_ONESHOT_MAX_BYTES"],
            environment["SIRCL_RING_MIN_BYTES"]) == ("32", "131072", "2097152")


def test_with_nccl_auto_on_a_pair_nccl_uses_the_devices_facing_the_partner():
    lock, _ = sircl_deployment(TP2, "cycle", 4, [0, 1], nccl="auto")
    specs = installer.specifications(lock)
    # In the style of the profile's own value, which names no ports.
    assert [spec.environment["NCCL_IB_HCA"] for spec in specs] == ["=rocep1s0f0,roceP2p1s0f0",
                                                                   "=rocep1s0f1,roceP2p1s0f1"]
    assert all("NCCL_DEBUG_SUBSYS" not in spec.environment or spec.environment["NCCL_DEBUG_SUBSYS"] != "INIT"
               for spec in specs)


def test_a_measured_table_that_matches_the_group_is_mounted_and_named(tmp_path, monkeypatch):
    from spark_transport.sircl.sparkring_sircl import tuning as sircl_tuning
    image = sircl_lock()
    cycle = document("cycle", 4)
    topology = transport.group_topology("ring:4", [0, 1, 2, 3])
    key = {**sircl_tuning.facts_for_layout(topology.session_layout(), topology.lane_count),
           **image["sircl"]["tuning_key"]}
    measured = sircl_tuning.build_document(key, [
        {"collective": "all_reduce", "mode": "eager", "bytes": 1 << 20, "choice": {"algorithm": "twoshot"},
         "p50_us": 100.0}])
    path = tmp_path / "runtime/tables/cycle4.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(measured))
    table = dict(transport.load_tuning(), tables=[{"path": "runtime/tables/cycle4.json",
                                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}])
    transport.validate_tuning(table, root=tmp_path)
    section = transport.section(image, cycle, [0, 1, 2, 3], nccl="never", tuning=table, root=tmp_path)
    digest = sircl_tuning.document_hash(measured)
    assert section["tuning"]["tables"] == [{**table["tables"][0], "hash": digest}]
    lock = installer.make_lock(TP4, install_site(4), "1" * 40, "2" * 64, image_runtime=image_lock.v2_view(image),
                               transport=section)
    spec = installer.specifications(lock)[0]
    assert spec.environment["SIRCL_TUNING_TABLE"] == f"{transport.TABLE_TARGET}/{digest}.json"
    assert any(mount.target == f"{transport.TABLE_TARGET}/{digest}.json" and mount.read_only for mount in spec.mounts)


def test_a_profile_that_sets_a_variable_the_adapter_owns_is_refused(monkeypatch):
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1])
    original = installer.installer_image.adapt

    def adapt(spec, value, **options):
        result = original(spec, value, **options)
        return result.__class__(**{**result.__dict__, "environment": {**result.environment, "SIRCL_LAYOUT": "x"}})
    monkeypatch.setattr(installer.installer_image, "adapt", adapt)
    with pytest.raises(transport.TransportError, match="SIRCL_LAYOUT"):
        installer.specifications(lock)


def test_micro_batching_is_refused_before_any_container_exists(monkeypatch):
    lock, _ = sircl_deployment(TP2, "pair", 2, [0, 1])
    original = installer.installer_image.adapt

    def adapt(spec, value, **options):
        result = original(spec, value, **options)
        return result.__class__(**{**result.__dict__, "command": (*result.command, "--enable-dbo")})
    monkeypatch.setattr(installer.installer_image, "adapt", adapt)
    with pytest.raises(transport.TransportError, match="micro-batching"):
        installer.specifications(lock)


def test_the_compose_exports_of_a_lock_without_a_section_are_unchanged():
    image = image_lock.v2_view(sircl_lock())
    plain = installer.make_lock(TP2, install_site(2), "1" * 40, "2" * 64, image_runtime=image)
    specs = installer.specifications(plain)
    assert all("SIRCL_MODE" not in spec.environment for spec in specs)
    assert all(spec.environment["VLLM_PLUGINS"].split(",")[:2] == ["b12x_loader", "sparkring_status"]
               and "sircl" not in spec.environment["VLLM_PLUGINS"].split(",") for spec in specs)


# Text, admission and host checks.

def test_the_plan_says_what_carries_the_collectives_and_where_the_settings_come_from():
    _, cycle = sircl_deployment(TP4, "cycle", 4, [0, 1, 2, 3])
    assert transport.plan_lines(cycle)[0] == ("Transport: sircl on every collective, NCCL off (default table, "
                                              "cycle-4: not measured, SIRCL's own rules apply)")
    assert "at most 1 relay on a lane" in transport.plan_lines(cycle)[1]
    _, pair = sircl_deployment(TP2, "pair", 2, [0, 1], nccl="auto")
    assert pair["group"]["cabling"] == "all"
    assert transport.plan_lines(pair)[0].startswith("Transport: sircl; NCCL may carry what the cabling allows")
    assert transport.prepared_line("image X carries no SIRCL layer", False) == (
        "Transport: prepared, because image X carries no SIRCL layer")


class Image:
    """Files of an image as isolated ``cat`` containers read them."""

    def __init__(self, lock, section):
        layer = {"schema": "sparkring-sircl-layer/v1", "version": section["sircl"]["version"],
                 "abi_version": section["sircl"]["abi_version"], "wheel": section["sircl"]["wheel"],
                 "native": section["sircl"]["native"], "p2p": section["sircl"]["p2p"],
                 "tuning_key": section["sircl"]["tuning_key"],
                 "files": {"/usr/local/lib/python3.12/dist-packages/sparkring_sircl/__init__.py": "7" * 64,
                           section["sircl"]["native"]["path"]: section["sircl"]["native"]["sha256"],
                           section["sircl"]["p2p"]["path"]: section["sircl"]["p2p"]["sha256"]}}
        self.layer = json.dumps(layer).encode()
        section["sircl"]["receipt"]["sha256"] = hashlib.sha256(self.layer).hexdigest()
        self.base = {"files": {**layer["files"], image_lock.LAYER_RECEIPT: section["sircl"]["receipt"]["sha256"]}}

    def __call__(self, argv, text=True):
        path = argv[-1]
        data = json.dumps(self.base).encode() if path == installer.installer_image.PARENT_RECEIPT else self.layer
        return type("Done", (), {"stdout": data})()


def test_the_sircl_layer_is_admitted_only_when_the_images_verification_covers_it():
    image_value = sircl_lock()
    lock, section = sircl_deployment(TP2, "pair", 2, [0, 1], image=image_value)
    image = Image(lock, section)
    lock = copy.deepcopy(lock)
    lock["image_runtime"]["parent_receipt_sha256"] = hashlib.sha256(json.dumps(image.base).encode()).hexdigest()
    lock["transport"]["sircl"]["receipt"]["sha256"] = section["sircl"]["receipt"]["sha256"]
    assert transport.admit_layer(lock, run=image)["files_verified"] == 3
    image.base["files"].pop(section["sircl"]["p2p"]["path"])
    lock["image_runtime"]["parent_receipt_sha256"] = hashlib.sha256(json.dumps(image.base).encode()).hexdigest()
    with pytest.raises(transport.TransportError, match="p2p library"):
        transport.admit_layer(lock, run=image)


def test_a_spark_must_hold_the_fabric_document_the_deployment_was_made_on(tmp_path):
    _, section = sircl_deployment(TP2, "pair", 2, [0, 1])
    path = tmp_path / fabric_document.HOST_PATH.lstrip("/")
    with pytest.raises(transport.TransportError, match="cannot be used"):
        transport.check_host_document(section, root=tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(fabric_document.encoded(document("pair", 2)))
    assert transport.check_host_document(section, root=tmp_path)["ok"]
    path.write_text(fabric_document.encoded(document("cycle", 4)))
    with pytest.raises(transport.TransportError, match="Run sudo sparkring install again"):
        transport.check_host_document(section, root=tmp_path)


def test_the_adapter_names_no_variable_the_sircl_package_does_not_read():
    from spark_transport.sircl.sparkring_sircl import env
    from spark_transport.sircl.sparkring_sircl.vllm import settings
    known = ({variable.name for variable in env.VARIABLES} | {variable.name for variable in settings.VARIABLES}
             | set(serve_plan.OPTION_VARIABLES) | set(serve_plan.REASONS) | {"SIRCL_P2P_NATIVE_LIBRARY"})
    lock, _ = sircl_deployment(TP4, "cycle", 4, [0, 1, 2, 3])
    environment = installer.specifications(lock)[0].environment
    # SIRCL_ENABLED is the profile's own name of the four-rank adapter's switch, not set by the adapter.
    assert {key for key in environment if key.startswith("SIRCL_")} - known == {"SIRCL_ENABLED"}
    assert Path(transport.TUNING_DEFAULTS).is_file()

# Every installer profile and the layouts beyond the profiles' own.

@pytest.mark.parametrize("profile", installer.installer_image.SUPPORTED)
@pytest.mark.parametrize("nccl", ["never", "auto"])
def test_every_installer_profile_renders_on_sircl_on_its_layouts(profile, nccl):
    cases = ([("cycle", 4, [0, 1, 2, 3])] if profile.endswith("-tp4") else
             [("pair", 2, [0, 1]), ("cycle", 4, [0, 1]), ("cycle", 4, [2, 3])])
    for shape, size, positions in cases:
        lock, section = sircl_deployment(profile, shape, size, positions, nccl=nccl)
        specs = installer.specifications(lock)
        assert len(specs) == len(positions)
        for spec in specs:
            environment = spec.environment
            assert (environment["SIRCL_MODE"], environment["SIRCL_NCCL"]) == ("custom", nccl)
            assert environment["SIRCL_FABRIC"] == section["group"]["layout"]
            assert environment["SIRCL_RANK_POSITIONS"] == ",".join(map(str, positions))
            assert environment["VLLM_PLUGINS"].split(",")[-1] == "sircl"
            for key, value in serve_plan.DISABLED_TRANSPORTS.items():
                assert environment[key] == value
            if nccl == "never":
                assert (environment["NCCL_DEBUG"], environment["NCCL_DEBUG_SUBSYS"]) == ("INFO", "INIT")


@pytest.mark.parametrize("shape, size, positions, layout, name, row, relays_, cabling", [
    ("cycle", 8, list(range(8)), "ring:8", "cycle-8", "cycle-8", 3, "ring"),
    ("cycle", 8, [4, 5, 6, 7], "ring:8", "path-4", "path", 2, "none"),
    ("cycle", 8, [6, 7, 0, 1], "ring:8", "path-4", "path", 2, "none"),
    ("path", 4, [0, 1, 2, 3], "path:4", "path-4", "path", 2, "none"),
])
def test_a_section_places_arcs_of_larger_fabrics(shape, size, positions, layout, name, row, relays_, cabling):
    value = document(shape, size)
    section = transport.section(sircl_lock(), value, positions, nccl="never", tuning=transport.load_tuning())
    group = section["group"]
    assert (group["layout"], group["positions"], group["name"], group["max_relays"], group["cabling"]) == (
        layout, positions, name, relays_, cabling)
    assert section["tuning"]["row"] == row
    for rank, devices in enumerate(section["devices"]):
        assert devices and set(devices) <= set(fabric_document.devices(value, positions[rank]))


def test_a_path_fabric_runs_sircl_on_its_path_and_keeps_nccl_off_across_relays():
    lock, section = sircl_deployment(TP4, "path", 4, [0, 1, 2, 3], nccl="auto")
    plain = installer.specifications({key: value for key, value in lock.items() if key != "transport"})
    for spec, before in zip(installer.specifications(lock), plain, strict=True):
        assert spec.environment["SIRCL_FABRIC"] == "path:4" and section["group"]["cabling"] == "none"
        # NCCL may not run across the path's relays, so its devices stay the profile's.
        assert spec.environment["NCCL_IB_HCA"] == before.environment["NCCL_IB_HCA"]


@pytest.mark.parametrize("edit, message", [
    (lambda table: table["layouts"]["pair"].update(source="guessed"), "measured, rules or inherited"),
    (lambda table: table["tables"].append({"path": "../outside.json", "sha256": "0" * 64}), "repository path"),
    (lambda table: table.update(sircl={"version": "0.2.0"}), "SIRCL version and ABI"),
])
def test_a_tuning_table_with_unstated_evidence_or_an_outside_table_is_refused(edit, message):
    table = transport.load_tuning()
    edit(table)
    with pytest.raises(transport.TransportError, match=message):
        transport.validate_tuning(table)


def test_a_layer_receipt_that_differs_from_the_lock_is_refused():
    lock, section = sircl_deployment(TP2, "pair", 2, [0, 1])
    image = Image(lock, section)
    lock = copy.deepcopy(lock)
    lock["image_runtime"]["parent_receipt_sha256"] = hashlib.sha256(json.dumps(image.base).encode()).hexdigest()
    lock["transport"]["sircl"]["receipt"]["sha256"] = "0" * 64
    with pytest.raises(transport.TransportError, match="layer receipt differs"):
        transport.admit_layer(lock, run=image)
