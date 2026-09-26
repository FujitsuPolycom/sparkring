"""Pinned mesh snapshots and native attachment checks using a fake local host."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from runtime.common import qwen_mesh as mesh


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def rule_entry(rule):
    mac = bytes.fromhex(rule.rewrite_ethernet_destination.replace(":", ""))

    def edit(offset, value, mask):
        return {
            "htype": "eth",
            "cmd": "set",
            "offset": offset,
            "val": value,
            "mask": mask,
        }

    return {
        "pref": rule.preference,
        "kind": "flower",
        "options": {
            "handle": rule.handle,
            "skip_sw": True,
            "in_hw": True,
            "keys": {
                "src_mac": rule.match_ethernet_source,
                "dst_mac": rule.match_ethernet_destination,
                "eth_type": "88b5",
            },
            "actions": [
                {"kind": "pedit", "keys": [edit(12, 0x08000000, 0xFFFF)]},
                {
                    "kind": "pedit",
                    "keys": [
                        edit(0, int.from_bytes(mac[:4], "big"), 0),
                        edit(4, int.from_bytes(mac[4:] + b"\0\0", "big"), 0xFFFF),
                    ],
                },
                {
                    "kind": "mirred",
                    "to_dev": rule.egress_netdev,
                    "mirred_action": "redirect",
                },
            ],
        },
    }


class FakeHost:
    def __init__(self, manager, site_raw, binary):
        self.manager = manager
        self.site_raw = site_raw
        self.binary = binary
        self.commands = []
        self.read_paths = []
        self.network = {
            key: self.network_object(kind, obj)
            for key, (kind, obj) in manager.objects.items()
        }
        self.mtu = 4096
        self.address = manager.site["management_addresses"][manager.rank]
        self.gid_override = None
        self.processes = {}
        self.recycle = False
        for index, marker in enumerate(
            m for m in manager.plan.markers if m.source_rank == manager.rank
        ):
            self.processes[100 + index] = {
                "argv": [
                    manager.site["marker_binary"],
                    "--device",
                    marker.rdma_device,
                    "--source-port",
                    "65535",
                    "--replacement-ethertype",
                    "0x88b5",
                    "--attach",
                    "--managed",
                ],
                "exe": binary,
                "start": 123 + index,
                "state": "S",
                "ready": {
                    "device": marker.rdma_device,
                    "rdma_tx_matcher": True,
                    "action": "ethertype_rewrite",
                    "source_port": 65535,
                    "replacement_port": 0,
                    "attached": True,
                    "requested_run_seconds": 0,
                    "max_run_seconds": 7200,
                    "sigint_sigterm_cleanup": True,
                    "managed": True,
                    "lifetime_seconds": None,
                },
            }

    def network_object(self, kind, obj):
        if kind == "route":
            value = {"dev": obj.source_netdev, "prefsrc": obj.source_ipv4}
            value.update(
                {"gateway": obj.gateway_ipv4} if obj.gateway_ipv4 else {"scope": "link"}
            )
            return value
        if kind == "neighbor":
            return {"lladdr": obj.next_hop_mac, "state": ["PERMANENT"]}
        if kind == "qdisc":
            return {"kind": "clsact", "handle": "ffff:"}
        return rule_entry(obj)

    def run(self, argv):
        self.commands.append(argv)
        assert not set(argv).intersection(
            {"add", "del", "replace", "flush", "up", "down"}
        )
        if argv[0] == "ibv_devinfo":
            return f"active_mtu: {self.mtu} (5)\n"
        if "addr" in argv:
            device = argv[-1]
            if device == self.manager.local.management_netdev:
                return json.dumps([{"addr_info": [{"local": self.address}]}])
            port = next(p for p in self.manager.local.ports if p.netdev == device)
            return json.dumps(
                [
                    {
                        "mtu": 9000,
                        "address": port.mac,
                        "addr_info": [{"local": port.ipv4}],
                    }
                ]
            )
        rows = []
        for key, (kind, obj) in self.manager.objects.items():
            if key not in self.network:
                continue
            if (
                (
                    "route" in argv
                    and kind == "route"
                    and argv[-1] == obj.destination_ipv4 + "/32"
                )
                or (
                    "neigh" in argv
                    and kind == "neighbor"
                    and obj.destination_ipv4 in argv
                )
                or ("qdisc" in argv and kind == "qdisc" and argv[-1] == obj)
                or (
                    "filter" in argv
                    and kind == "rule"
                    and argv[-2:] == [obj.ingress_netdev, "ingress"]
                )
            ):
                rows.append(self.network[key])
        assert argv[0] in ("ip", "tc") and "show" in argv
        return json.dumps(rows)

    def read_bytes(self, path):
        self.read_paths.append(path.as_posix())
        if path == Path("/private/mesh-site.json"):
            return self.site_raw
        if path == Path(self.manager.site["marker_binary"]):
            return self.binary
        if path.parts[-1] in ("cmdline", "exe") and "proc" in path.parts:
            record = self.processes[int(path.parts[-2])]
            return (
                record["exe"]
                if path.name == "exe"
                else "\0".join(record["argv"]).encode() + b"\0"
            )
        raise AssertionError(path)

    def read_text(self, path):
        if path.name == "stat":
            pid = int(path.parts[-2])
            record = self.processes[pid]
            return f"{pid} (marker) " + " ".join(
                [record["state"], *["0"] * 18, str(record["start"])]
            )
        port = next(p for p in self.manager.local.ports if p.rdma_device in path.parts)
        if "gids" in path.parts:
            return self.gid_override or "::ffff:" + port.ipv4
        if "ndevs" in path.parts:
            return port.netdev
        raise AssertionError(path)

    def process_ids(self):
        return sorted(self.processes)

    def readiness(self, pid):
        if self.recycle:
            self.processes[pid]["start"] += 1
        return copy.deepcopy(self.processes[pid]["ready"])


@pytest.fixture(params=range(4))
def rig(request, tmp_path, monkeypatch):
    path = mesh.ROOT / "runtime/glm53-spark-mtp3-mesh/make_example.py"
    spec = importlib.util.spec_from_file_location("qwen_mesh_fixture_examples", path)
    examples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(examples)
    binary = b"pinned native managed marker"
    site = examples.site_example()
    site["marker_binary_sha256"] = sha(binary)
    raw = json.dumps(site).encode()
    site_path = tmp_path / "site.json"
    site_path.write_bytes(raw)
    (tmp_path / "fabric.example.json").write_text(
        json.dumps(examples.topology_example())
    )
    network = mesh._network()
    loaded = network.profile.load_site(site_path)
    manager = network.NetworkManager(
        site_path, request.param, tmp_path / "uncreated-state", require_root=False
    )
    host = FakeHost(manager, raw, binary)

    def load(p):
        assert p == Path("/private/mesh-site.json")
        return loaded

    def no_lock(*args, **kwargs):
        raise AssertionError(
            "Read-only checks must not acquire the network mutation lock"
        )

    monkeypatch.setattr(network.profile, "load_site", load)
    monkeypatch.setattr(network.NetworkManager, "_lock", no_lock)
    reference = {
        "site_path": "/private/mesh-site.json",
        "site_sha256": sha(raw),
        "plan_sha256": manager.plan.sha256,
    }
    hcas = [
        manager.local.port(direction, function).rdma_device
        for function in (0, 1)
        for direction in ("clockwise", "counter_clockwise")
    ]
    return {
        "reference": reference,
        "rank": request.param,
        "hcas": hcas,
        "gid": manager.plan.roce_gid_index,
        "host_ip": site["management_addresses"][request.param],
        "host": host,
    }


def test_complete_snapshot_requires_every_rank_network_and_managed_attachment(rig):
    result = mesh.check(**rig)
    assert result["ready"] is True
    assert result["rank"] == rig["rank"]
    assert result["hcas"] == rig["hcas"]
    assert len(result["markers"]) == 2
    assert result["network_objects"] == len(rig["host"].network)
    assert result["end_to_end_rc_qualified"] is False
    assert result["supervision_verified"] is False
    assert sum(argv[0] == "ibv_devinfo" for argv in rig["host"].commands) == 4
    assert not rig["host"].manager.state_dir.exists()


def test_stale_gid_ports_name_each_port_whose_pinned_slot_lacks_its_address(rig):
    host = rig["host"]
    assert mesh.stale_gid_ports(rig["reference"], rig["rank"], host=host) == []
    # A cabled neighbor restarted: each port's IPv4 GID moved out of the pinned slot.
    host.gid_override = "::ffff:198.51.100.1"
    assert mesh.stale_gid_ports(rig["reference"], rig["rank"], host=host) == [
        (port.netdev, port.ipv4) for port in host.manager.local.ports]
    assert not any(argv[:2] == ["ip", "addr"] and "show" not in argv for argv in host.commands)


@pytest.mark.parametrize(
    "field,value",
    [
        ("site_sha256", "0" * 64),
        ("plan_sha256", "0" * 64),
    ],
)
def test_reference_identity_drift_fails_before_host_commands(rig, field, value):
    rig["reference"][field] = value
    with pytest.raises(ValueError, match="pinned SHA-256"):
        mesh.check(**rig)
    assert rig["host"].commands == []


@pytest.mark.parametrize(
    "change", ["primary_only", "reordered", "wrong_gid", "boolean_gid", "wrong_address"]
)
def test_requested_rank_configuration_must_match_the_pinned_fabric(rig, change):
    if change == "primary_only":
        rig["hcas"] = rig["hcas"][:2]
    elif change == "reordered":
        rig["hcas"] = list(reversed(rig["hcas"]))
    elif change == "wrong_gid":
        rig["gid"] += 1
    elif change == "boolean_gid":
        rig["gid"] = True
    else:
        rig["host_ip"] = "192.0.2.250"
    with pytest.raises(ValueError):
        mesh.check(**rig)
    assert rig["host"].commands == []


@pytest.mark.parametrize("kind", ["route", "qdisc", "rule"])
def test_missing_network_objects_cannot_pass_with_live_markers(rig, kind):
    host = rig["host"]
    del host.network[next(key for key in host.network if key.startswith(kind + ":"))]
    with pytest.raises(ValueError, match="Missing mesh network objects"):
        mesh.check(**rig)


@pytest.mark.parametrize(
    "change", ["hardware", "redirect", "drops", "management", "rdma_mtu", "gid"]
)
def test_network_state_is_inspected_instead_of_trusting_a_status_file(rig, change):
    host = rig["host"]
    row = host.network[next(key for key in host.network if key.startswith("rule:"))]
    if change == "hardware":
        row["options"]["in_hw"] = False
    elif change == "redirect":
        row["options"]["actions"][2]["to_dev"] = "other"
    elif change == "drops":
        row["options"]["actions"][0]["stats"] = {"drops": 1}
    elif change == "management":
        host.address = "192.0.2.250"
    elif change == "rdma_mtu":
        host.mtu = 2048
    else:
        host.gid_override = "::ffff:192.0.2.250"
    with pytest.raises(ValueError):
        mesh.check(**rig)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "bounded",
        "binary",
        "exe",
        "zombie",
        "reused_pid",
        "foreign_binary",
    ],
)
def test_persistent_marker_identity_is_bound_to_live_process(rig, change):
    host = rig["host"]
    record = host.processes[100]
    if change == "missing":
        del host.processes[100]
    elif change == "duplicate":
        host.processes[102] = copy.deepcopy(record)
    elif change == "bounded":
        record["argv"][-1:] = ["--run-seconds", "7200"]
    elif change == "binary":
        host.binary = b"changed marker on disk"
    elif change == "exe":
        record["exe"] = b"different running executable"
    elif change == "zombie":
        record["state"] = "Z"
    elif change == "reused_pid":
        host.recycle = True
    else:
        record["argv"][0] = "/other/marker"
    with pytest.raises(ValueError):
        mesh.check(**rig)


@pytest.mark.parametrize(
    "field,value",
    [
        ("attached", False),
        ("managed", False),
        ("rdma_tx_matcher", False),
        ("lifetime_seconds", 7200),
        ("device", "another-device"),
        ("source_port", 4791),
        ("source_port", True),
        ("action", "udp_destination_rewrite"),
        ("requested_run_seconds", 7200),
        ("requested_run_seconds", False),
    ],
)
def test_native_attachment_record_is_required_even_when_pids_exist(rig, field, value):
    rig["host"].processes[100]["ready"][field] = value
    with pytest.raises(ValueError, match="confirmed the planned persistent attachment"):
        mesh.check(**rig)


def test_unreadable_readiness_cannot_fall_back_to_process_presence(rig, monkeypatch):
    def unreadable(pid):
        raise PermissionError("readiness log is not readable")

    monkeypatch.setattr(rig["host"], "readiness", unreadable)
    with pytest.raises(PermissionError, match="not readable"):
        mesh.check(**rig)


@pytest.mark.parametrize(
    "site_path", ["relative.json", "/", "/a/../b", "/a//b", "/a\\b", "/a\n.json"]
)
def test_static_reference_rejects_unsafe_paths_without_loading_or_reading(
    site_path, monkeypatch
):
    def forbidden():
        raise AssertionError("Static validation must not load the mesh or private site")

    monkeypatch.setattr(mesh, "_network", forbidden)
    with pytest.raises(ValueError, match="normalized absolute"):
        mesh.validate_site_reference(
            {"site_path": site_path, "site_sha256": "a" * 64, "plan_sha256": "b" * 64}
        )


def test_static_reference_accepts_an_unavailable_host_file():
    reference = {
        "site_path": "/not-present/site.json",
        "site_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
    }
    assert mesh.validate_site_reference(reference) == reference


@pytest.mark.parametrize("change", ["extra", "missing", "uppercase_hash", "short_hash"])
def test_static_reference_requires_exact_fields_and_hashes(change):
    reference = {
        "site_path": "/site.json",
        "site_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
    }
    if change == "extra":
        reference["repair"] = True
    elif change == "missing":
        del reference["plan_sha256"]
    elif change == "uppercase_hash":
        reference["site_sha256"] = "A" * 64
    else:
        reference["plan_sha256"] = "b" * 63
    with pytest.raises(ValueError):
        mesh.validate_site_reference(reference)


def test_source_inventory_includes_loaded_mesh_code_and_import_time_pins():
    network = mesh._network()
    modules = [
        mesh,
        network,
        network.inspector,
        network.profile,
        network.fabric,
        network.profile.candidate,
        network.profile.r35,
        network.profile.build_bundle,
    ]
    for module in modules:
        assert (
            Path(module.__file__).relative_to(mesh.ROOT).as_posix() in mesh.SOURCE_FILES
        )
    assert len(mesh.SOURCE_FILES) == len(set(mesh.SOURCE_FILES))
    assert all((mesh.ROOT / name).is_file() for name in mesh.SOURCE_FILES)


@pytest.mark.parametrize(
    "mode,raw,error",
    [
        (stat.S_IFREG, b'{"attached":true}', None),
        (stat.S_IFIFO, b'{"attached":true}', "regular stdout"),
        (stat.S_IFREG, b"x" * 65537, "supported size"),
    ],
    ids=["regular", "pipe", "oversized"],
)
def test_stdout_reader_requires_a_bounded_regular_file(monkeypatch, mode, raw, error):
    stream = io.BytesIO(raw)
    stream.fileno = lambda: 77
    reads = []
    original_read = stream.read

    def read(size):
        reads.append(size)
        return original_read(size)

    stream.read = read
    monkeypatch.setattr(mesh.os, "O_NONBLOCK", 0, raising=False)

    def open_fd(path, flags):
        assert path == Path("/proc/123/fd/1")
        return 77

    monkeypatch.setattr(mesh.os, "open", open_fd)
    monkeypatch.setattr(mesh.os, "fdopen", lambda fd, mode: stream)
    monkeypatch.setattr(mesh.os, "fstat", lambda fd: SimpleNamespace(st_mode=mode))
    if error:
        with pytest.raises(ValueError, match=error):
            mesh.Host().readiness(123)
    else:
        assert mesh.Host().readiness(123) == {"attached": True}
    assert reads == ([] if mode == stat.S_IFIFO else [65537])


@pytest.mark.parametrize("rank", [True, "0", -1, 4])
def test_invalid_rank_is_rejected_before_reading_private_site(rank):
    reference = {
        "site_path": "/site.json",
        "site_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
    }
    with pytest.raises(ValueError, match="rank must be an integer"):
        mesh.check(reference, rank, [], 3, "192.0.2.10", host=object())
