"""GPU- and RDMA-free checks of the per-port RoCE GID index.

Three layers are executed rather than inspected:

* the runtime's GID table reader and per-device selection, run against
  fake ``/sys/class/infiniband`` trees and compared with SparkRing's host
  resolver ``integrations/vllm/spark_roce_gid.py``;
* the proxy binding, which passes one index per HCA to ``roce_create``;
* the proxy's device setup and queue-pair connection, compiled with GCC
  against the stand-in ``infiniband/verbs.h`` of ``test_peer_wait.py``.

They establish index selection and its use as each HCA's source GID, not RDMA
connectivity on GB10 hardware.
"""

from __future__ import annotations

import ast
import ctypes
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parent
ROCE = ROOT / "roce"
RUNTIME = ROCE / "roce_oneshot.py"
PROXY_SOURCE = ROCE / "_roce_proxy.c"
SELECTION = {"_env_int", "_env_list", "default_gid_index", "_sysfs_text", "port_gid",
             "port_gid_indices", "discover_hcas"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


spark_roce_gid = load_module("prepared_roce_host_gid_resolver", ROOT.parent / "spark_roce_gid.py")


def selection(root: Path) -> dict:
    """The runtime's GID functions, executed from source with ``root`` as the sysfs class.

    ``roce_oneshot`` imports CUDA packages, so its functions are compiled on
    their own with the module constants they read.
    """
    tree = ast.parse(RUNTIME.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in SELECTION]
    assert {node.name for node in nodes} == SELECTION
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[]))
    namespace = dict(os=os, ipaddress=ipaddress, Path=Path, Optional=Optional, Sequence=Sequence,
                     DEFAULT_GID_INDEX=3, INFINIBAND_SYSFS=root)
    exec(compile(module, str(RUNTIME), "exec"), namespace)
    return namespace


def test_runtime_reads_the_kernel_rdma_device_class():
    text = RUNTIME.read_text()
    assert 'INFINIBAND_SYSFS = Path("/sys/class/infiniband")' in text.splitlines()
    assert "DEFAULT_GID_INDEX = 3" in text.splitlines()


# The kernel writes every GID as eight groups of four hexadecimal digits.
LINK_LOCAL = ipaddress.IPv6Address("fe80::1").exploded
EMPTY = ipaddress.IPv6Address("::").exploded


def add_device(root: Path, device: str, entries: dict, *, interfaces=("enp1s0f0np0",), state="4: ACTIVE"):
    """A fake RDMA device: ``entries`` maps a GID index to (GID text, type, owning interface)."""
    base = root / device
    (base / "device" / "net").mkdir(parents=True)
    for interface in interfaces:
        (base / "device" / "net" / interface).mkdir()
    port = base / "ports" / "1"
    for sub in ("gids", "gid_attrs/types", "gid_attrs/ndevs"):
        (port / sub).mkdir(parents=True)
    (port / "state").write_text(state + "\n")
    for index, (gid, kind, owner) in entries.items():
        (port / "gids" / str(index)).write_text(gid + "\n")
        if kind is not None:
            (port / "gid_attrs" / "types" / str(index)).write_text(kind + "\n")
        if owner is not None:
            (port / "gid_attrs" / "ndevs" / str(index)).write_text(owner + "\n")


def spark_port(address="198.18.0.1", interface="enp1s0f0np0", *, v2_index=3):
    """The table of a Spark fabric port: IPv6 link-local entries first, then the address."""
    mapped = spark_roce_gid.ipv4_mapped_gid(address)
    return {0: (LINK_LOCAL, "IB/RoCE v1", interface), 1: (LINK_LOCAL, "RoCE v2", interface),
            2: (mapped, "IB/RoCE v1", interface), v2_index: (mapped, "RoCE v2", interface)}


def neighbor_restarted():
    """The address registered again while its RoCE v2 entry at index 3 was held, then released."""
    table = spark_port(v2_index=4)
    table[3] = (EMPTY, None, None)
    return table


def two_addresses():
    table = spark_port()
    second = spark_roce_gid.ipv4_mapped_gid("198.18.8.1")
    table.update({4: (second, "IB/RoCE v1", "enp1s0f0np0"), 5: (second, "RoCE v2", "enp1s0f0np0")})
    return table


def other_interface_entry():
    table = spark_port()
    table[5] = (spark_roce_gid.ipv4_mapped_gid("198.19.0.9"), "RoCE v2", "bond0")
    return table


CASES = {
    "spark-port": (spark_port(), {}, 3),
    "neighbor-restarted": (neighbor_restarted(), {}, 4),
    "ipv6-only": ({0: (LINK_LOCAL, "IB/RoCE v1", "enp1s0f0np0"), 1: (LINK_LOCAL, "RoCE v2", "enp1s0f0np0")},
                  {}, None),
    "roce-v1-only": ({2: (spark_roce_gid.ipv4_mapped_gid("198.18.0.1"), "IB/RoCE v1", "enp1s0f0np0")},
                     {}, None),
    "two-addresses": (two_addresses(), {}, None),
    "other-interface-entry": (other_interface_entry(), {}, 3),
    "interface-unknown": (spark_port(), {"interfaces": ()}, 3),
    "interface-unknown-two-owners": (other_interface_entry(), {"interfaces": ()}, None),
    "spaced-type": ({7: (spark_roce_gid.ipv4_mapped_gid("198.18.0.1"), "  RoCE   V2 ", "enp1s0f0np0")},
                    {}, 7),
    "unparsable-entry": ({3: ("not-a-gid", "RoCE v2", "enp1s0f0np0"),
                          6: (spark_roce_gid.ipv4_mapped_gid("198.18.0.1"), "RoCE v2", "enp1s0f0np0")},
                         {}, 6),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_runtime_selects_the_entry_the_host_resolver_selects(tmp_path, case):
    table, options, expected = CASES[case]
    add_device(tmp_path, "rocep1s0f0", table, **options)
    port_gid = selection(tmp_path)["port_gid"]
    if expected is None:
        with pytest.raises(ValueError):
            port_gid("rocep1s0f0", root=tmp_path)
        with pytest.raises(ValueError):
            spark_roce_gid.resolve_device_gid_index("rocep1s0f0", root=tmp_path)
        return
    index, address, interface = port_gid("rocep1s0f0", root=tmp_path)
    assert index == expected == spark_roce_gid.resolve_device_gid_index("rocep1s0f0", root=tmp_path)
    assert ipaddress.IPv4Address(address)
    # The entry's owning interface, also when device/net names none.
    assert interface == "enp1s0f0np0"


def test_missing_gid_table_is_a_resolution_error(tmp_path):
    (tmp_path / "rocep1s0f0").mkdir()
    with pytest.raises(ValueError, match="rocep1s0f0 port 1 has no readable GID table"):
        selection(tmp_path)["port_gid"]("rocep1s0f0", root=tmp_path)


def test_errors_list_the_entries_present(tmp_path):
    add_device(tmp_path, "rocep1s0f0", two_addresses())
    with pytest.raises(ValueError) as error:
        selection(tmp_path)["port_gid"]("rocep1s0f0", root=tmp_path)
    message = str(error.value)
    assert "several RoCE v2 IPv4 GIDs owned by enp1s0f0np0" in message
    assert "index 3 (198.18.0.1, enp1s0f0np0)" in message and "index 5 (198.18.8.1" in message


@pytest.fixture
def unset_gid_environment(monkeypatch):
    for name in ("B12X_ROCE_GID_INDEX", "NCCL_IB_GID_INDEX", "B12X_ROCE_HCA", "NCCL_IB_HCA"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_each_device_uses_its_own_resolved_index(tmp_path, unset_gid_environment):
    unset_gid_environment.setenv("NCCL_IB_GID_INDEX", "3")
    add_device(tmp_path, "rocep1s0f0", spark_port("198.18.0.1", "enp1s0f0np0"),
               interfaces=("enp1s0f0np0",))
    add_device(tmp_path, "roceP2p1s0f0", {
        index: (gid, kind, "enP2p1s0f0np0" if owner else None)
        for index, (gid, kind, owner) in neighbor_restarted().items()}, interfaces=("enP2p1s0f0np0",))
    indices, lines = selection(tmp_path)["port_gid_indices"](("rocep1s0f0", "roceP2p1s0f0"), root=tmp_path)
    assert indices == (3, 4)
    assert lines == (
        "rocep1s0f0 uses RoCE GID index 3, the RoCE v2 GID of 198.18.0.1 on enp1s0f0np0",
        "roceP2p1s0f0 uses RoCE GID index 4, the RoCE v2 GID of 198.18.0.1 on enP2p1s0f0np0",
    )


@pytest.mark.parametrize("variables,configured", [
    ({"B12X_ROCE_GID_INDEX": "5", "NCCL_IB_GID_INDEX": "3"}, 5),
    ({"NCCL_IB_GID_INDEX": "2"}, 2),
    ({}, 3),
])
def test_an_unresolved_device_uses_the_configured_index(tmp_path, unset_gid_environment, variables, configured):
    for name, value in variables.items():
        unset_gid_environment.setenv(name, value)
    add_device(tmp_path, "rocep1s0f0", spark_port(v2_index=4))
    add_device(tmp_path, "rocep1s0f1", two_addresses(), interfaces=("enp1s0f0np0",))
    indices, lines = selection(tmp_path)["port_gid_indices"](("rocep1s0f0", "rocep1s0f1"), root=tmp_path)
    assert indices == (4, configured)
    assert lines[1].startswith(f"rocep1s0f1 uses the configured RoCE GID index {configured}: "
                               "rocep1s0f1 port 1 has several RoCE v2 IPv4 GIDs")


def test_an_explicit_index_applies_to_every_device_without_reading_tables(tmp_path, unset_gid_environment):
    unset_gid_environment.setenv("B12X_ROCE_GID_INDEX", "3")
    indices, lines = selection(tmp_path)["port_gid_indices"](("rocep1s0f0", "roceP2p1s0f0"), 6, root=tmp_path)
    assert indices == (6, 6)
    assert lines == ("rocep1s0f0 uses RoCE GID index 6, the caller's gid_index",
                     "roceP2p1s0f0 uses RoCE GID index 6, the caller's gid_index")


def test_discovery_admits_a_device_whose_address_moved(tmp_path, unset_gid_environment):
    add_device(tmp_path, "rocep1s0f0", neighbor_restarted())
    add_device(tmp_path, "rocep1s0f1", two_addresses())
    add_device(tmp_path, "roceP2p1s0f0", spark_port(), state="1: DOWN")
    add_device(tmp_path, "roceP2p1s0f1", {2: (LINK_LOCAL, "RoCE v2", "enp1s0f0np0")})
    discover = selection(tmp_path)["discover_hcas"]
    # Resolution admits the moved address; the configured index 3 admits the
    # port with two addresses; inactive ports and ports without either are not used.
    assert discover() == ("rocep1s0f0", "rocep1s0f1")
    # An explicit index admits exactly the ports populated there.
    assert discover(3) == ("rocep1s0f1",)
    assert discover(4) == ("rocep1s0f0", "rocep1s0f1")
    unset_gid_environment.setenv("B12X_ROCE_HCA", "roceP2p1s0f1,rocep1s0f0")
    assert discover() == ("roceP2p1s0f1", "rocep1s0f0")


def test_runtime_binds_one_index_per_device():
    """The constructor selects per device, shares the common index, logs and passes the tuple on."""
    text = RUNTIME.read_text()
    cls = next(node for node in ast.parse(text).body
               if isinstance(node, ast.ClassDef) and node.name == "RoceOneshotAllReduce")
    init = ast.unparse(next(node for node in cls.body if getattr(node, "name", None) == "__init__"))
    assert "discover_hcas(gid_index)" in init
    assert "self.gid_indices, gid_lines = port_gid_indices(self.hca_names, gid_index)" in init
    assert "self.gid_indices[0] if len(set(self.gid_indices)) == 1 else None" in init
    assert "print(f'RoCEnante rank {self.rank}: {line}', file=sys.stderr, flush=True)" in init
    assert "gid_indices=self.gid_indices" in init


# -- the proxy binding ----------------------------------------------------------


class FakeLibrary:
    def __init__(self):
        self.created = []

    def roce_create(self, *arguments):
        self.created.append(arguments)
        return 1

    def roce_destroy(self, context):
        pass


def test_binding_passes_one_gid_index_per_hca(monkeypatch):
    proxy = load_module("prepared_roce_proxy_binding", ROCE / "_proxy.py")
    library = FakeLibrary()
    monkeypatch.setattr(proxy, "load", lambda: library)
    options = dict(world_size=2, rank=0, hca_names=("rocep1s0f0", "roceP2p1s0f0"), region_ptr=4096,
                   region_bytes=1 << 20, slot_bytes=4096, peer_hca_map=((-1, -1), (0, 1)))
    created = proxy.Proxy(gid_indices=(3, 4), **options)
    arguments = library.created[0]
    assert arguments[3] == 2
    assert isinstance(arguments[4], ctypes.Array) and arguments[4]._type_ is ctypes.c_int
    assert list(arguments[4]) == [3, 4]
    assert created.gid_indices == (3, 4)
    for wrong, message in (((3,), "one RoCE GID index per HCA"), ((3, 256), "between 0 and 255"),
                           ((-1, 3), "between 0 and 255")):
        with pytest.raises(ValueError, match=message):
            proxy.Proxy(gid_indices=wrong, **options)
    assert len(library.created) == 1


def test_binding_declares_an_int_array_for_roce_create():
    tree = ast.parse((ROCE / "_proxy.py").read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and ast.unparse(node.targets[0]) == "lib.roce_create.argtypes")
    assert ast.unparse(assignment.value.elts[4]) == "ctypes.POINTER(ctypes.c_int)"


# -- the proxy's device setup and connection, compiled against stand-in verbs ----

HARNESS = r"""
#include "proxy_under_test.c"

// Stand-in verbs for two HCAs. The GID at index i of HCA h is
// ::ffff:198.18.h.i unless i is the scenario's empty index.
static struct ibv_device devices[2];
static struct ibv_device *device_list[2] = {&devices[0], &devices[1]};
static const char *device_names[2] = {"rocep1s0f0", "roceP2p1s0f0"};
static struct ibv_context contexts[2];
static struct ibv_pd pds[2];
static struct ibv_mr mrs[2] = {{1, 101}, {2, 102}};
static struct ibv_cq cqs[2] = {{0}, {1}};
static struct ibv_qp qps[2][ROCE_MAX_PEERS];
static int n_qps[2];
static int queried[2] = {-1, -1};
static int rtr_sgid[2] = {-1, -1};
static int rtr_dgid[2] = {-1, -1};
static int empty_index = -1;

struct ibv_device **ibv_get_device_list(int *num) { *num = 2; return device_list; }
const char *ibv_get_device_name(struct ibv_device *d) { return device_names[d - devices]; }
void ibv_free_device_list(struct ibv_device **l) { (void)l; }
struct ibv_context *ibv_open_device(struct ibv_device *d) { return &contexts[d - devices]; }
int ibv_query_port(struct ibv_context *c, uint8_t p, struct ibv_port_attr *a) {
    (void)c; (void)p;
    a->state = IBV_PORT_ACTIVE; a->lid = 0; a->active_mtu = IBV_MTU_4096;
    return 0;
}
int ibv_query_gid(struct ibv_context *c, uint8_t p, int i, union ibv_gid *g) {
    (void)p;
    int h = (int)(c - contexts);
    queried[h] = i;
    memset(g->raw, 0, sizeof(g->raw));
    if (i != empty_index) {
        g->raw[10] = 0xff; g->raw[11] = 0xff; g->raw[12] = 198; g->raw[13] = 18;
        g->raw[14] = (uint8_t)h; g->raw[15] = (uint8_t)i;
    }
    return 0;
}
struct ibv_pd *ibv_alloc_pd(struct ibv_context *c) { return &pds[c - contexts]; }
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *a, size_t n, int f) {
    (void)a; (void)n; (void)f;
    return &mrs[pd - pds];
}
struct ibv_cq *ibv_create_cq(struct ibv_context *c, int n, void *u, void *ch, int v) {
    (void)n; (void)u; (void)ch; (void)v;
    return &cqs[c - contexts];
}
struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *a) {
    (void)a;
    int h = (int)(pd - pds);
    struct ibv_qp *qp = &qps[h][n_qps[h]++];
    qp->qp_num = (uint32_t)(100 * (h + 1) + n_qps[h]);
    qp->cq = h;
    return qp;
}
int ibv_modify_qp(struct ibv_qp *q, struct ibv_qp_attr *a, int m) {
    (void)m;
    if (a->qp_state == IBV_QPS_RTR) {
        rtr_sgid[q->cq] = a->ah_attr.grh.sgid_index;
        rtr_dgid[q->cq] = a->ah_attr.grh.dgid.raw[15];
    }
    return 0;
}
int ibv_poll_cq(struct ibv_cq *cq, int entries, struct ibv_wc *wc) { (void)cq; (void)entries; (void)wc; return 0; }
const char *ibv_wc_status_str(enum ibv_wc_status s) { (void)s; return "success"; }
int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad) {
    (void)qp; (void)wr; (void)bad;
    return 0;
}
int ibv_destroy_qp(struct ibv_qp *q) { (void)q; return 0; }
int ibv_destroy_cq(struct ibv_cq *c) { (void)c; return 0; }
int ibv_dereg_mr(struct ibv_mr *m) { (void)m; return 0; }
int ibv_dealloc_pd(struct ibv_pd *p) { (void)p; return 0; }
int ibv_close_device(struct ibv_context *c) { (void)c; return 0; }

// Rank 0 of a pair with two HCAs; rank 1 reaches it on HCA 0 and 1 and
// publishes GIDs whose last byte is 200 + its HCA index.
int main(int argc, char **argv) {
    const char *scenario = argc > 1 ? argv[1] : "";
    int gids[2] = {3, 4};
    if (strcmp(scenario, "empty-index") == 0) {
        empty_index = 4;
    } else if (strcmp(scenario, "out-of-range") == 0) {
        gids[1] = 256;
    } else if (strcmp(scenario, "per-port") != 0) {
        return 2;
    }
    uint64_t layout[8];
    if (roce_layout(2, 4096, layout) != 0) return 3;
    void *region = calloc(1, layout[4]);
    uint8_t map[2 * ROCE_MAX_PATHS] = {255, 255, 255, 255, 0, 1, 255, 255};
    char err[512] = "";
    roce_ctx_t *c = roce_create(2, 0, device_names, 2, gids, region, layout[4], 4096, 2, map,
                                sizeof(map), err, sizeof(err));
    int published[2] = {-1, -1};
    if (c != NULL) {
        roce_blob_t blobs[2];
        if (roce_local_blob(c, &blobs[0], sizeof(blobs[0])) != 0) return 4;
        published[0] = blobs[0].gid[0][15];
        published[1] = blobs[0].gid[1][15];
        blobs[1] = blobs[0];
        blobs[1].rank = 1;
        blobs[1].region_addr = 0x100000;
        memset(blobs[1].peer_hca, UINT8_MAX, sizeof(blobs[1].peer_hca));
        for (int h = 0; h < 2; h++) {
            blobs[1].rkey[h] = 7u + (uint32_t)h;
            blobs[1].qp_num[h][0] = 500u + (uint32_t)h;
            blobs[1].qp_num[h][1] = 0;
            blobs[1].gid[h][15] = (uint8_t)(200 + h);
            blobs[1].peer_hca[0][h] = (uint8_t)h;
        }
        if (roce_connect(c, blobs, sizeof(blobs)) != 0) {
            snprintf(err, sizeof(err), "%s", c->err);
        }
        roce_destroy(c);
    }
    printf("{\"error\": \"%s\", \"queried\": [%d, %d], \"published\": [%d, %d], "
           "\"sgid\": [%d, %d], \"dgid\": [%d, %d]}\n",
           err, queried[0], queried[1], published[0], published[1],
           rtr_sgid[0], rtr_sgid[1], rtr_dgid[0], rtr_dgid[1]);
    free(region);
    return 0;
}
"""


@pytest.fixture(scope="module")
def device_setup(tmp_path_factory):
    compiler = shutil.which("gcc")
    if compiler is None or sys.platform != "linux":
        pytest.skip("GCC on Linux compiles the proxy against stand-in verbs")
    verbs = load_module("prepared_roce_peer_wait_checks", ROOT / "test_peer_wait.py").FAKE_VERBS
    directory = tmp_path_factory.mktemp("proxy-gid")
    (directory / "infiniband").mkdir()
    (directory / "infiniband/verbs.h").write_text(verbs)
    shutil.copyfile(PROXY_SOURCE, directory / "proxy_under_test.c")
    (directory / "harness.c").write_text(HARNESS)
    binary = directory / "harness"
    subprocess.run(
        [compiler, "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
         "-I", str(directory), str(directory / "harness.c"), "-o", str(binary), "-lpthread"],
        check=True, capture_output=True, text=True,
    )

    def run(scenario):
        result = subprocess.run([str(binary), scenario], capture_output=True, text=True, check=True, env={})
        return json.loads(result.stdout)
    return run


def test_each_hca_queries_publishes_and_routes_from_its_own_index(device_setup):
    result = device_setup("per-port")
    assert result["error"] == ""
    assert result["queried"] == [3, 4]
    # The published GID is the one at each HCA's own index.
    assert result["published"] == [3, 4]
    # Each queue pair's source GID index is its local HCA's; the destination
    # is the peer's published GID for the remote HCA of that path.
    assert result["sgid"] == [3, 4]
    assert result["dgid"] == [200, 201]


def test_an_empty_gid_fails_setup_with_the_device_and_index(device_setup):
    result = device_setup("empty-index")
    assert result["error"] == "RDMA device roceP2p1s0f0 port 1 has no GID at index 4"
    assert result["queried"] == [3, 4]


def test_an_index_outside_the_address_vector_field_is_refused(device_setup):
    result = device_setup("out-of-range")
    assert result["error"] == "RoCE GID index 256 of roceP2p1s0f0 is outside 0-255"
    assert result["queried"] == [-1, -1]
