"""Own only explicitly created local mesh routes, neighbors and TC rules.

Implemented ownership bookkeeping; hardware readiness remains a local snapshot,
not an end-to-end RC qualification. Marker processes belong to the supervisor.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mesh_network_inspector", HERE / "inspect_fabric.py")
inspector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inspector
SPEC.loader.exec_module(inspector)
profile = inspector.profile
fabric = profile.fabric


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(f"Network command failed: {argv!r}: {result.stderr.strip()}")
    return result.stdout


class NetworkManager:
    """Serialize local configuration changes and conservatively retain ambiguous state.

    A successful add is journaled as owned. If interruption leaves a pending
    add, an existing exact object is adopted, never retrospectively owned.
    This favors retaining an artifact over deleting somebody else's state.
    """

    def __init__(self, site_path, rank, state_dir, *, runner=command,
                 read_text=lambda path: path.read_text(), require_root=True):
        if type(rank) is not int or rank not in range(4):
            raise ValueError("Rank must be an integer in [0,3]")
        self.site, self.topology, self.plan = profile.load_site(Path(site_path))
        self.rank, self.local = rank, self.topology.rank(rank)
        self.state_dir = Path(state_dir).absolute()
        if self.state_dir == Path(self.state_dir.anchor):
            raise ValueError("State directory must not be a filesystem root")
        self.runner, self.read_text, self.require_root = runner, read_text, require_root
        self.journal_path = self.state_dir / f"network-r{rank}.json"
        self.objects = {}
        for route in self.plan.routes:
            if route.source_rank == rank:
                self.objects["route:" + route.path_name] = ("route", route)
                if route.permanent_final_neighbor:
                    self.objects["neighbor:" + route.path_name] = ("neighbor", route)
        for rule in self.plan.tc_rules:
            if rule.intermediate_rank == rank:
                self.objects.setdefault("qdisc:" + rule.ingress_netdev, ("qdisc", rule.ingress_netdev))
        for rule in self.plan.tc_rules:
            if rule.intermediate_rank == rank:
                self.objects["rule:" + rule.name] = ("rule", rule)

    def _json(self, argv):
        value = json.loads(self.runner(argv))
        if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
            raise ValueError(f"Expected an object inventory from {argv!r}")
        return value

    def _secure(self, path):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Symlink state path is forbidden: {path}")
        if self.require_root and (info.st_uid != 0 or info.st_mode & 0o022):
            raise ValueError(f"State must be root-owned and not group/world writable: {path}")

    @contextmanager
    def _lock(self):
        if self.require_root and (not hasattr(os, "geteuid") or os.geteuid() != 0):
            raise PermissionError("Managed networking requires root")
        for path in [*reversed(self.state_dir.parents), self.state_dir]:
            if path.exists() or path.is_symlink():
                # Writable sticky parents such as /tmp are acceptable only above
                # the private, root-owned state directory, never inside it.
                if path.is_symlink():
                    raise ValueError(f"Symlink state ancestor is forbidden: {path}")
                info = path.stat()
                if self.require_root and (info.st_uid != 0 or
                        (info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX)):
                    raise ValueError(f"Unsafe writable state ancestor: {path}")
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._secure(self.state_dir)
        lock_path = self.state_dir / "network.lock"
        if lock_path.exists() or lock_path.is_symlink():
            self._secure(lock_path)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            self.journal = {"schema": "sparkring-managed-network/v1", "rank": self.rank,
                            "plan_sha256": self.plan.sha256, "objects": {}}
            if self.journal_path.exists() or self.journal_path.is_symlink():
                self._secure(self.journal_path)
                self.journal = json.loads(self.journal_path.read_text())
                if (self.journal.get("schema") != "sparkring-managed-network/v1"
                        or self.journal.get("rank") != self.rank
                        or self.journal.get("plan_sha256") != self.plan.sha256
                        or not isinstance(self.journal.get("objects"), dict)
                        or any(k not in self.objects or v not in ("owned", "adopted", "pending")
                               for k, v in self.journal["objects"].items())):
                    raise ValueError("Ownership journal differs from the canonical plan")
            yield
        finally:
            os.close(fd)

    def _save(self):
        fd, temporary = tempfile.mkstemp(prefix=".network-", dir=self.state_dir)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(self.journal, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.journal_path)
            if os.name == "posix":
                directory = os.open(self.state_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _links(self, *, verify_rdma_mtu=True):
        management = self._json(["ip", "-j", "-4", "addr", "show", "dev", self.local.management_netdev])
        if self.site["management_addresses"][self.rank] not in {
                x.get("local") for link in management for x in link.get("addr_info", [])}:
            raise ValueError("Management address does not identify this rank")
        for port in self.local.ports:
            links = self._json(["ip", "-j", "addr", "show", "dev", port.netdev])
            if (len(links) != 1 or links[0].get("mtu") != self.plan.expected_ethernet_mtu
                    or links[0].get("address", "").lower() != port.mac.lower()
                    or port.ipv4 not in {x.get("local") for x in links[0].get("addr_info", [])}):
                raise ValueError(f"Link address or MTU differs: {port.netdev}")
            root = Path("/sys/class/infiniband") / port.rdma_device / "ports/1"
            gid = ipaddress.IPv6Address(self.read_text(root / f"gids/{self.plan.roce_gid_index}").strip())
            if gid.ipv4_mapped != ipaddress.IPv4Address(port.ipv4):
                raise ValueError(f"RoCE GID differs: {port.rdma_device}")
            netdev = self.read_text(root / f"gid_attrs/ndevs/{self.plan.roce_gid_index}").strip()
            if netdev != port.netdev:
                raise ValueError(f"RDMA device maps to a different netdev: {port.rdma_device}")
            if verify_rdma_mtu:
                device = self.runner(["ibv_devinfo", "-d", port.rdma_device, "-i", "1"])
                mtu = re.findall(r"^\s*active_mtu:\s+(\d+)\s+\(\d+\)\s*$", device, re.MULTILINE)
                if mtu != [str(self.plan.expected_roce_mtu)]:
                    raise ValueError(f"RoCE MTU differs: {port.rdma_device}")

    def _present(self, kind, obj, *, health=True):
        if kind == "route":
            rows = self._json(["ip", "-j", "-4", "route", "show", "exact", obj.destination_ipv4 + "/32"])
            if rows and (len(rows) != 1 or rows[0].get("dev") != obj.source_netdev
                         or rows[0].get("gateway") != obj.gateway_ipv4
                         or rows[0].get("prefsrc") != obj.source_ipv4
                         or rows[0].get("type", "unicast") != "unicast"
                         or "multipath" in rows[0]
                         or (obj.gateway_ipv4 is None and rows[0].get("scope") != "link")):
                raise ValueError(f"Conflicting route: {obj.destination_ipv4}")
            return bool(rows)
        if kind == "neighbor":
            rows = self._json(["ip", "-j", "neigh", "show", "to", obj.destination_ipv4, "dev", obj.source_netdev])
            if rows and (len(rows) != 1 or rows[0].get("lladdr", "").lower() != obj.next_hop_mac.lower()
                         or rows[0].get("state") != ["PERMANENT"]):
                raise ValueError(f"Conflicting permanent neighbor: {obj.destination_ipv4}")
            return bool(rows)
        if kind == "qdisc":
            rows = self._json(["tc", "-j", "qdisc", "show", "dev", obj])
            ingress = [x for x in rows if x.get("kind") in ("clsact", "ingress")
                       or x.get("handle") == "ffff:"]
            if ingress and (len(ingress) != 1 or ingress[0].get("kind") != "clsact"):
                raise ValueError(f"Conflicting ingress qdisc: {obj}")
            return bool(ingress)
        rows = self._json(["tc", "-j", "-s", "filter", "show", "dev", obj.ingress_netdev, "ingress"])
        # tc also returns a summary entry without options for each preference.
        candidates = [x for x in rows if x.get("pref") == obj.preference
                      or x.get("options", {}).get("handle") == obj.handle]
        material = [x for x in candidates if x.get("options")]
        if not candidates:
            return False
        if len(material) != 1:
            raise ValueError(f"Conflicting TC preference/handle: {obj.preference}/{obj.handle}")
        if not health:
            material = json.loads(json.dumps(material))
            for action in material[0].get("options", {}).get("actions", []):
                action["stats"] = {}
        inspector.verify_rule(obj, material)
        return True

    def _argv(self, kind, obj, add):
        if kind == "route":
            return fabric.route_command(obj, add=add)
        if kind == "neighbor":
            return fabric.neighbor_command(obj, add=add)
        if kind == "rule":
            return fabric.tc_rule_command(obj, add=add)
        return ["tc", "qdisc", "add" if add else "del", "dev", obj, "clsact"]

    def check(self, *, verify_rdma_mtu=True):
        """Verify local state; periodic callers may defer only the verbs MTU probe.

        Startup and periodic full checks must retain the default. Disabling
        the probe still verifies Ethernet MTU, GID/netdev identity and TC rules.
        """
        self._links(verify_rdma_mtu=verify_rdma_mtu)
        missing = [key for key, (kind, obj) in self.objects.items() if not self._present(kind, obj)]
        if missing:
            raise ValueError(f"Missing mesh network objects: {missing}")
        return {"ready": True, "rank": self.rank, "plan_sha256": self.plan.sha256,
                "objects": len(self.objects)}

    def up(self):
        with self._lock():
            self._links()
            # Detect conflicting objects before making any network changes.
            for kind, obj in self.objects.values():
                self._present(kind, obj)
            for key, (kind, obj) in self.objects.items():
                if self._present(kind, obj):
                    if self.journal["objects"].get(key) != "owned":
                        self.journal["objects"][key] = "adopted"
                    self._save()
                    continue
                self.journal["objects"][key] = "pending"
                self._save()
                self.runner(self._argv(kind, obj, True))
                self.journal["objects"][key] = "owned"
                self._save()
                if not self._present(kind, obj):
                    raise ValueError(f"Created network object is absent: {key}")
            return {**self.check(), "ownership": dict(self.journal["objects"])}

    def down(self):
        warnings = []
        with self._lock():
            for key, (kind, obj) in reversed(list(self.objects.items())):
                ownership = self.journal["objects"].get(key)
                if ownership != "owned":
                    if ownership == "pending":
                        warnings.append(f"Retained ambiguous pending object: {key}")
                    continue
                try:
                    if self._present(kind, obj, health=False):
                        if kind == "qdisc" and any(self._json(
                                ["tc", "-j", "filter", "show", "dev", obj, direction])
                                for direction in ("ingress", "egress")):
                            warnings.append(f"Retained nonempty owned qdisc: {obj}")
                            continue
                        self.runner(self._argv(kind, obj, False))
                    del self.journal["objects"][key]
                    self._save()
                except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                    warnings.append(f"Retained {key}: {error}")
            return {"rank": self.rank, "clean": not warnings, "warnings": warnings,
                    "ownership": dict(self.journal["objects"])}
