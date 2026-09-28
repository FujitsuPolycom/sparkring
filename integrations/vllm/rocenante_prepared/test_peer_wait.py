"""GPU- and RDMA-free checks of the supervised peer wait.

Four layers are executed rather than inspected:

* the wait loop's PTX, run by a small interpreter for the instructions it uses;
* both collective kernels' control flow, run as Python with CPU memory;
* the proxy's peer-wait supervision, compiled with GCC against a stand-in
  ``infiniband/verbs.h`` whose posts and completions the test scripts;
* the runtime's host-side settings and health messages.

They establish protocol decisions, not CUDA compilation, memory ordering,
RDMA delivery or timing on GB10 hardware.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parent
ROCE = ROOT / "roce"
PROXY_SOURCE = ROCE / "_roce_proxy.c"
MASK = 0xFFFFFFFF


def module_constants(path: Path, names: set[str]) -> dict[str, object]:
    """Evaluate the module-level constant assignments ``names`` of ``path``.

    The modules import CUDA packages, so their constants are read from source.
    Only this repository's own constant expressions are evaluated, with no
    builtins and with earlier constants of the same module as the only names.
    """
    values: dict[str, object] = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                expression = compile(ast.Expression(node.value), str(path), "eval")
                values[target.id] = eval(expression, {"__builtins__": {}}, dict(values))
    assert set(values) == names
    return values


def c_defines(names: set[str]) -> dict[str, int]:
    source = PROXY_SOURCE.read_text()
    values = {}
    for name in names:
        match = re.search(rf"^#define {name} (\S+)$", source, re.MULTILINE)
        assert match, name
        values[name] = int(re.sub(r"[uUlL]+$", "", match.group(1)), 0)
    return values


# -- the wait loop's PTX -------------------------------------------------------


def load_intrinsics_constants():
    return module_constants(ROCE / "_cute_intrinsics.py", {"ABORT_CHECK_POLLS", "PEER_WAIT_PTX"})


def run_wait_ptx(*, arrives_at=None, abort_from_check=None, expected=7, limit=MASK):
    """Interpret the peer-wait PTX.

    The flag equals ``expected`` from load number ``arrives_at`` on; the abort
    word is nonzero from its ``abort_from_check``-th read on. Returns the
    result register, the flag loads and the abort-word reads.
    """
    ptx = load_intrinsics_constants()["PEER_WAIT_PTX"]
    program, labels = [], {}
    for line in ptx.splitlines():
        line = line.strip()
        if not line or line in "{}" or line.startswith(".reg"):
            continue
        if line.endswith(":"):
            labels[line[:-1]] = len(program)
            continue
        program.append(line.rstrip(";"))
    registers = {"$1": "flag", "$2": expected, "$3": "abort", "$4": limit}
    counts = {"flag": 0, "abort": 0}

    def value(token):
        token = token.strip()
        if token in registers:
            return registers[token]
        return int(token)

    pc = 0
    while pc < len(program):
        instruction = program[pc]
        pc += 1
        guard = None
        if instruction.startswith("@"):
            guard, instruction = instruction.split(None, 1)
        if guard is not None:
            negate = guard.startswith("@!")
            taken = bool(registers[guard.lstrip("@!")])
            if taken == negate:
                continue
        opcode, _, rest = instruction.partition(" ")
        operands = [item.strip() for item in rest.split(",")]
        if opcode == "bra":
            pc = labels[operands[0]]
        elif opcode == "mov.u32":
            registers[operands[0]] = value(operands[1]) & MASK
        elif opcode == "add.u32":
            registers[operands[0]] = (value(operands[1]) + value(operands[2])) & MASK
        elif opcode == "and.b32":
            registers[operands[0]] = value(operands[1]) & value(operands[2])
        elif opcode.startswith("setp."):
            compare = {"ne": int.__ne__, "eq": int.__eq__, "ge": int.__ge__}[opcode.split(".")[1]]
            registers[operands[0]] = compare(value(operands[1]), value(operands[2]))
        elif opcode.startswith("ld."):
            target = value(operands[1].strip("[]"))
            counts[target] += 1
            if target == "flag":
                arrived = arrives_at is not None and counts["flag"] >= arrives_at
                registers[operands[0]] = expected if arrived else (expected - 2) & MASK
            else:
                stop = abort_from_check is not None and counts["abort"] >= abort_from_check
                registers[operands[0]] = expected if stop else 0
        else:
            raise AssertionError(f"uninterpreted PTX instruction {instruction!r}")
    return registers["$0"], counts["flag"], counts["abort"]


def test_wait_returns_when_the_flag_arrives_without_reading_the_abort_word():
    assert run_wait_ptx(arrives_at=5) == (0, 5, 0)


def test_wait_reads_the_abort_word_once_per_check_interval():
    interval = load_intrinsics_constants()["ABORT_CHECK_POLLS"]
    assert interval & (interval - 1) == 0
    assert run_wait_ptx(abort_from_check=1) == (1, interval, 1)
    assert run_wait_ptx(abort_from_check=3) == (1, 3 * interval, 3)
    # A live peer that answers after several checks is waited for.
    assert run_wait_ptx(arrives_at=3 * interval + 17) == (0, 3 * interval + 17, 3)


def test_arrived_flag_wins_over_a_pending_abort_in_the_same_poll():
    interval = load_intrinsics_constants()["ABORT_CHECK_POLLS"]
    assert run_wait_ptx(arrives_at=interval, abort_from_check=1) == (0, interval, 0)


def test_device_poll_bound_stops_a_wait_without_the_host():
    assert run_wait_ptx(limit=10) == (1, 10, 0)
    assert run_wait_ptx(limit=1) == (1, 1, 0)


# -- the collective kernels ----------------------------------------------------

CTRL, EPOCH, POISON, STAGE, TAIL = 0x1000, 0x2000, 0x2100, 0x2200, 0x2300
INPUT, OUTPUT, RECV, FLAGS, SEND = 0x3000, 0x6000, 0x4000, 0x8000, 0xC000


class Uint32(int):
    def __new__(cls, value):
        return super().__new__(cls, int(value) & MASK)

    def __add__(self, other):
        return Uint32(int(self) + int(other))


def simulate_kernel(filename, *, world, rank, paths, sequence, stop, packs=1):
    """Run one waiting thread of a one-block launch of ``filename``'s kernel."""
    source = ROCE / filename
    tree = ast.parse(source.read_text())
    kernel = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "kernel"
    )
    kernel.decorator_list = []
    for argument in kernel.args.args:
        argument.annotation = None
    kernel.returns = None
    offsets = module_constants(ROCE / "_oneshot_cute.py", {"CTRL_ABORT", "CTRL_DONE", "CTRL_FAILED"})
    memory = {EPOCH: Uint32(sequence - 1), CTRL + offsets["CTRL_DONE"]: Uint32(0xDEAD)}
    waits, outputs = [], []

    def load(address):
        return Uint32(memory.get(address, 0))

    def store(address, value):
        memory[address] = Uint32(value)

    def atomic_add(address, value):
        previous = load(address)
        store(address, previous + value)
        return previous

    def wait(address, expected, abort_address, limit):
        waits.append((address, int(expected), abort_address, int(limit)))
        return Uint32(1 if stop else 0)

    def store_vector(address, *words):
        if OUTPUT <= address < OUTPUT + 0x1000:
            outputs.append(address)

    namespace = {
        "Uint32": Uint32, "Int32": int, "Int64": int, "PACK_BYTES": 16, "PATH_COUNT": 2,
        **offsets,
        "cute": NS(
            arch=NS(
                thread_idx=lambda: (0, 0, 0),
                block_idx=lambda: (0, 0, 0),
                grid_dim=lambda: (1, 1, 1),
                sync_threads=lambda: None,
            ),
            make_rmem_tensor=lambda shape, dtype: [0.0] * shape[0],
        ),
        "cutlass": NS(const_expr=bool, range_constexpr=range, Float32=float),
        "ld_relaxed_gpu_u32": load, "ld_relaxed_sys_u32": load,
        "st_relaxed_sys_u32": store, "st_release_gpu_u32": store,
        "atomic_add_relaxed_gpu_u32": atomic_add,
        "fence_sc_sys": lambda: None, "fence_sc_gpu": lambda: None,
        "ld_global_v4_u32": lambda address: (1, 2, 3, 4),
        "ld_relaxed_sys_v4_u32": lambda address: (5, 6, 7, 8),
        "st_global_v4_u32": store_vector,
        "spin_until_eq_or_abort_sys": wait,
    }
    exec(compile(ast.Module(body=[kernel], type_ignores=[]), str(source), "exec"), namespace)
    runtime = NS(
        _threads=32, _world_size=world, _rank=rank, _slots=2, _opposite_paths=paths,
        _flag_stride=128, _pack_elems=8,
        _accumulate_words=lambda accumulator, words, initialize: None,
        _store_accumulator=lambda address, accumulator: outputs.append(address),
    )
    def pointer(address):
        return NS(toint=lambda: address)

    arguments = dict(
        self=runtime, input_ptr=pointer(INPUT), output_ptr=pointer(OUTPUT), nbytes=16 * packs,
        recv_base=RECV, flag_base=FLAGS, send_base=SEND, ctrl_base=CTRL, slot_bytes=4096,
        epoch_ptr=EPOCH, stage_counter_ptr=STAGE, tail_counter_ptr=TAIL, poison_ptr=POISON,
        spin_limit=Uint32(MASK),
    )
    if filename == "_allgather_cute.py":
        arguments.update(shard_packs=packs, row_packs=1)
    else:
        arguments.update(size_packs=packs)
    namespace["kernel"](**arguments)
    assert waits, "the simulated thread must wait for a peer"
    assert all(expected == sequence & MASK for _, expected, _, _ in waits)
    assert all(abort == CTRL + offsets["CTRL_ABORT"] for _, _, abort, _ in waits)
    assert all(limit == MASK for _, _, _, limit in waits)
    return memory, outputs, offsets


KERNELS = ["_allgather_cute.py", "_oneshot_cute.py"]
TOPOLOGIES = [(2, 1, 2), (2, 1, 4), (4, 1, 2), (4, 1, 4)]


@pytest.mark.parametrize("filename", KERNELS)
@pytest.mark.parametrize("world,rank,paths", TOPOLOGIES)
@pytest.mark.parametrize("sequence", [1, 0x80000000, 0xFFFFFFFF, 0])
def test_completed_wait_advances_the_epoch_and_publishes_the_sequence(filename, world, rank, paths, sequence):
    memory, outputs, offsets = simulate_kernel(
        filename, world=world, rank=rank, paths=paths, sequence=sequence, stop=False
    )
    assert memory[EPOCH] == sequence & MASK
    assert memory[CTRL + offsets["CTRL_DONE"]] == sequence & MASK
    assert memory.get(CTRL + offsets["CTRL_FAILED"], 0) == 0
    assert memory.get(POISON, 0) == 0
    assert outputs, "a completed collective stores its output"


@pytest.mark.parametrize("filename", KERNELS)
@pytest.mark.parametrize("world,rank,paths", TOPOLOGIES)
@pytest.mark.parametrize("sequence", [1, 0x80000000, 0xFFFFFFFF, 0])
def test_stopped_wait_poisons_without_output_even_at_sequence_zero(filename, world, rank, paths, sequence):
    memory, outputs, offsets = simulate_kernel(
        filename, world=world, rank=rank, paths=paths, sequence=sequence, stop=True
    )
    assert outputs == [], "no output may derive from an incomplete collective"
    assert memory[CTRL + offsets["CTRL_FAILED"]] == 1
    assert memory[POISON] == 1
    assert memory[CTRL + 8] == sequence & MASK
    assert memory[CTRL + 12] == 0
    assert memory[EPOCH] == (sequence - 1) & MASK
    assert memory[CTRL + offsets["CTRL_DONE"]] == 0xDEAD


# -- the proxy's supervision, compiled against stand-in verbs -------------------

FAKE_VERBS = r"""
#ifndef SPARKRING_FAKE_VERBS_H
#define SPARKRING_FAKE_VERBS_H
#include <stddef.h>
#include <stdint.h>
struct ibv_device { int unused; };
struct ibv_context { int unused; };
struct ibv_pd { int unused; };
struct ibv_mr { uint32_t lkey; uint32_t rkey; };
struct ibv_cq { int index; };
struct ibv_qp { uint32_t qp_num; int cq; };
union ibv_gid { uint8_t raw[16]; };
enum ibv_mtu { IBV_MTU_4096 = 5 };
enum ibv_port_state { IBV_PORT_ACTIVE = 4 };
struct ibv_port_attr { enum ibv_port_state state; uint16_t lid; enum ibv_mtu active_mtu; };
enum ibv_qp_type { IBV_QPT_RC = 2 };
struct ibv_qp_cap { uint32_t max_send_wr, max_recv_wr, max_send_sge, max_recv_sge, max_inline_data; };
struct ibv_qp_init_attr { struct ibv_cq *send_cq, *recv_cq; struct ibv_qp_cap cap; enum ibv_qp_type qp_type; };
enum ibv_qp_state { IBV_QPS_INIT = 1, IBV_QPS_RTR = 2, IBV_QPS_RTS = 3 };
struct ibv_global_route { union ibv_gid dgid; uint32_t flow_label; uint8_t sgid_index, hop_limit, traffic_class; };
struct ibv_ah_attr { struct ibv_global_route grh; uint16_t dlid; uint8_t sl, src_path_bits, is_global, port_num; };
struct ibv_qp_attr {
    enum ibv_qp_state qp_state; enum ibv_mtu path_mtu; uint32_t dest_qp_num, rq_psn, sq_psn;
    uint16_t pkey_index; uint8_t port_num; int qp_access_flags;
    uint8_t max_dest_rd_atomic, min_rnr_timer, timeout, retry_cnt, rnr_retry, max_rd_atomic;
    struct ibv_ah_attr ah_attr;
};
enum {
    IBV_QP_STATE = 1, IBV_QP_PKEY_INDEX = 2, IBV_QP_PORT = 4, IBV_QP_ACCESS_FLAGS = 8,
    IBV_QP_AV = 16, IBV_QP_PATH_MTU = 32, IBV_QP_DEST_QPN = 64, IBV_QP_RQ_PSN = 128,
    IBV_QP_MAX_DEST_RD_ATOMIC = 256, IBV_QP_MIN_RNR_TIMER = 512, IBV_QP_TIMEOUT = 1024,
    IBV_QP_RETRY_CNT = 2048, IBV_QP_RNR_RETRY = 4096, IBV_QP_SQ_PSN = 8192,
    IBV_QP_MAX_QP_RD_ATOMIC = 16384
};
enum { IBV_ACCESS_LOCAL_WRITE = 1, IBV_ACCESS_REMOTE_WRITE = 2 };
enum ibv_wc_status { IBV_WC_SUCCESS = 0, IBV_WC_RETRY_EXC_ERR = 12 };
struct ibv_wc { uint64_t wr_id; enum ibv_wc_status status; uint32_t vendor_err; };
struct ibv_sge { uint64_t addr; uint32_t length; uint32_t lkey; };
enum ibv_wr_opcode { IBV_WR_RDMA_WRITE = 0 };
enum { IBV_SEND_SIGNALED = 2, IBV_SEND_INLINE = 8 };
struct ibv_send_wr {
    uint64_t wr_id; struct ibv_send_wr *next; struct ibv_sge *sg_list; int num_sge;
    enum ibv_wr_opcode opcode; unsigned int send_flags;
    union { struct { uint64_t remote_addr; uint32_t rkey; } rdma; } wr;
};
struct ibv_device **ibv_get_device_list(int *num);
const char *ibv_get_device_name(struct ibv_device *device);
void ibv_free_device_list(struct ibv_device **list);
struct ibv_context *ibv_open_device(struct ibv_device *device);
int ibv_query_port(struct ibv_context *context, uint8_t port, struct ibv_port_attr *attr);
int ibv_query_gid(struct ibv_context *context, uint8_t port, int index, union ibv_gid *gid);
struct ibv_pd *ibv_alloc_pd(struct ibv_context *context);
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access);
struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe, void *user, void *channel, int vector);
struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *attr);
int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int mask);
int ibv_poll_cq(struct ibv_cq *cq, int entries, struct ibv_wc *wc);
const char *ibv_wc_status_str(enum ibv_wc_status status);
int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad);
int ibv_destroy_qp(struct ibv_qp *qp);
int ibv_destroy_cq(struct ibv_cq *cq);
int ibv_dereg_mr(struct ibv_mr *mr);
int ibv_dealloc_pd(struct ibv_pd *pd);
int ibv_close_device(struct ibv_context *context);
#endif
"""

HARNESS = r"""
#include "proxy_under_test.c"

// Stand-in verbs: every signaled post completes on its queue pair's CQ with
// the status the scenario selects for notices or for payload writes.
typedef struct { uint64_t wr_id; uint64_t remote_addr; uint32_t value; int inline_value; } posted_t;
static posted_t posted[256];
static int n_posted;
static struct ibv_wc queued[2][256];
static int n_queued[2];
static enum ibv_wc_status notice_status = IBV_WC_SUCCESS;

struct ibv_device **ibv_get_device_list(int *num) { *num = 0; return NULL; }
const char *ibv_get_device_name(struct ibv_device *device) { (void)device; return ""; }
void ibv_free_device_list(struct ibv_device **list) { (void)list; }
struct ibv_context *ibv_open_device(struct ibv_device *device) { (void)device; return NULL; }
int ibv_query_port(struct ibv_context *c, uint8_t p, struct ibv_port_attr *a) { (void)c; (void)p; (void)a; return -1; }
int ibv_query_gid(struct ibv_context *c, uint8_t p, int i, union ibv_gid *g) { (void)c; (void)p; (void)i; (void)g; return -1; }
struct ibv_pd *ibv_alloc_pd(struct ibv_context *c) { (void)c; return NULL; }
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *a, size_t n, int f) { (void)pd; (void)a; (void)n; (void)f; return NULL; }
struct ibv_cq *ibv_create_cq(struct ibv_context *c, int n, void *u, void *ch, int v) { (void)c; (void)n; (void)u; (void)ch; (void)v; return NULL; }
struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *a) { (void)pd; (void)a; return NULL; }
int ibv_modify_qp(struct ibv_qp *q, struct ibv_qp_attr *a, int m) { (void)q; (void)a; (void)m; return -1; }
int ibv_destroy_qp(struct ibv_qp *q) { (void)q; return 0; }
int ibv_destroy_cq(struct ibv_cq *c) { (void)c; return 0; }
int ibv_dereg_mr(struct ibv_mr *m) { (void)m; return 0; }
int ibv_dealloc_pd(struct ibv_pd *p) { (void)p; return 0; }
int ibv_close_device(struct ibv_context *c) { (void)c; return 0; }
const char *ibv_wc_status_str(enum ibv_wc_status s) {
    return s == IBV_WC_SUCCESS ? "success" : "transport retry counter exceeded";
}

int ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad) {
    (void)bad;
    for (; wr != NULL; wr = wr->next) {
        posted_t *row = &posted[n_posted++];
        row->wr_id = wr->wr_id;
        row->remote_addr = wr->wr.rdma.remote_addr;
        row->inline_value = (wr->send_flags & IBV_SEND_INLINE) != 0;
        row->value = row->inline_value ? *(const uint32_t *)(uintptr_t)wr->sg_list[0].addr : 0;
        if (wr->send_flags & IBV_SEND_SIGNALED) {
            struct ibv_wc *wc = &queued[qp->cq][n_queued[qp->cq]++];
            wc->wr_id = wr->wr_id;
            wc->status = (wr->wr_id & ROCE_NOTICE_WR) ? notice_status : IBV_WC_SUCCESS;
            wc->vendor_err = wc->status == IBV_WC_SUCCESS ? 0 : 0x81;
        }
    }
    return 0;
}

int ibv_poll_cq(struct ibv_cq *cq, int entries, struct ibv_wc *wc) {
    int n = n_queued[cq->index] < entries ? n_queued[cq->index] : entries;
    memcpy(wc, queued[cq->index], (size_t)n * sizeof(*wc));
    memmove(queued[cq->index], queued[cq->index] + n,
            (size_t)(n_queued[cq->index] - n) * sizeof(*wc));
    n_queued[cq->index] -= n;
    return n;
}

static struct ibv_cq cqs[2] = {{0}, {1}};
static struct ibv_qp qps[2] = {{11, 0}, {12, 1}};
static roce_ctx_t *c;
static volatile uint32_t *ctrl;
static const uint64_t SECOND = 1000000000ull;
static uint64_t t0 = 1000 * 1000000000ull;
static int stopped;

static void set_flag(int peer, int path, uint32_t slot, uint32_t value) {
    uint32_t *flag = (uint32_t *)(c->region + c->flag_off +
                                  (((uint64_t)peer * ROCE_SLOTS + slot) * ROCE_LAYOUT_PATHS + path) *
                                      ROCE_FLAG_STRIDE);
    *flag = value;
}

// Rank 0 of a pair has posted ``seq`` and its kernel waits for rank 1.
static void setup(uint32_t seq, uint64_t timeout_ns) {
    uint64_t layout[8];
    c = calloc(1, sizeof(*c));
    if (roce_layout(2, 4096, layout) != 0) exit(2);
    c->world = 2; c->rank = 0; c->n_hca = 2; c->slot_bytes = 4096;
    c->region = calloc(1, layout[4]); c->region_bytes = layout[4];
    c->recv_off = layout[0]; c->flag_off = layout[1]; c->send_off = layout[2]; c->ctrl_off = layout[3];
    c->peer_path_count[1] = 2; c->peer_hca[1][0] = 0; c->peer_hca[1][1] = 1;
    c->peer_hca[0][0] = c->peer_hca[0][1] = -1;
    c->peer_addr[1] = 0x100000; c->peer_rkey[1][0] = 7; c->peer_rkey[1][1] = 8;
    for (int h = 0; h < 2; h++) { c->hca[h].cq = &cqs[h]; c->hca[h].qp[1] = &qps[h]; }
    if (roce_set_wait_timeout(c, timeout_ns) != 0) exit(3);
    ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    ctrl[0] = seq; c->last_seq = seq; ctrl[ROCE_CTRL_DONE] = seq - 1u;
    c->posted_ns = t0; c->last_tick_ns = t0;
    for (int path = 0; path < 2; path++) {
        set_flag(1, path, seq & 1u, seq == 1u ? 0u : seq - 2u);
        set_flag(1, path, (seq + 1u) & 1u, seq - 1u);
    }
}

// One proxy iteration at ``seconds`` after the post, as proxy_main runs it.
static int tick(double seconds) {
    if (stopped) return -1;
    if (proxy_tick(c, ctrl, t0 + (uint64_t)(seconds * (double)SECOND)) != 0) {
        fail_proxy(c, ctrl);
        stopped = 1;
        return -1;
    }
    return 0;
}

static void report(const char *scenario) {
    printf("{\"scenario\": \"%s\", \"failed\": %d, \"abort\": %u, \"stalls\": %llu, "
           "\"resolved\": %llu, \"longest_stall_ns\": %llu, \"notices_posted\": %llu, "
           "\"notices_received\": %llu, \"error\": \"%s\", \"posts\": [",
           scenario, atomic_load(&c->failed), ctrl[ROCE_CTRL_ABORT],
           (unsigned long long)atomic_load(&c->stalls),
           (unsigned long long)atomic_load(&c->stalls_resolved),
           (unsigned long long)atomic_load(&c->longest_stall_ns),
           (unsigned long long)atomic_load(&c->notices_posted),
           (unsigned long long)atomic_load(&c->notices_received), c->err);
    for (int i = 0; i < n_posted; i++) {
        printf("%s{\"notice\": %d, \"path\": %llu, \"offset\": %llu, \"value\": %u}", i ? ", " : "",
               (posted[i].wr_id & ROCE_NOTICE_WR) != 0,
               (unsigned long long)((posted[i].wr_id & ~ROCE_NOTICE_WR) % ROCE_MAX_PATHS),
               (unsigned long long)(posted[i].remote_addr - c->peer_addr[1]), posted[i].value);
    }
    printf("]}\n");
}

int main(int argc, char **argv) {
    const char *scenario = argc > 1 ? argv[1] : "";
    if (strcmp(scenario, "late-peer") == 0) {
        setup(5, 60 * SECOND);
        tick(1.0);                                  // below the stall report
        tick(6.0);                                  // stall: report and check
        tick(6.5);                                  // within the check interval
        tick(7.2);                                  // second check
        set_flag(1, 0, 1, 5); set_flag(1, 1, 1, 5); // the peer's writes land
        ctrl[ROCE_CTRL_DONE] = 5;                   // and the kernel completes
        tick(40.0);
    } else if (strcmp(scenario, "late-peer-at-wrap") == 0) {
        setup(0, 60 * SECOND);
        tick(6.0);
        set_flag(1, 0, 0, 0); set_flag(1, 1, 0, 0);
        ctrl[ROCE_CTRL_DONE] = 0;
        tick(8.0);
    } else if (strcmp(scenario, "next-doorbell-ends-stall") == 0) {
        setup(5, 60 * SECOND);
        tick(6.0);
        set_flag(1, 0, 1, 5); set_flag(1, 1, 1, 5);
        ctrl[ROCE_CTRL_DONE] = 5;
        ctrl[0] = 6; c->last_seq = 6; c->posted_ns = t0 + 9 * SECOND;
        set_flag(1, 0, 0, 4); set_flag(1, 1, 0, 4);
        tick(9.5);
    } else if (strcmp(scenario, "unreachable-peer") == 0) {
        setup(5, 60 * SECOND);
        notice_status = IBV_WC_RETRY_EXC_ERR;
        tick(6.0);                                  // the check is posted
        tick(6.5);                                  // and fails
    } else if (strcmp(scenario, "inconsistent-flags") == 0) {
        setup(5, 60 * SECOND);
        set_flag(1, 1, 1, 9);
        tick(6.0);
    } else if (strcmp(scenario, "timeout") == 0) {
        setup(5, 10 * SECOND);
        tick(4.0); tick(5.0); tick(9.9); tick(10.0);
    } else if (strcmp(scenario, "peer-stopped") == 0) {
        setup(5, 60 * SECOND);
        ctrl[ROCE_CTRL_NOTICE + 1] = ROCE_NOTICE_ABORT | 5u;
        tick(0.1);
    } else if (strcmp(scenario, "peer-waits-for-this-rank") == 0) {
        setup(5, 60 * SECOND);
        ctrl[ROCE_CTRL_DONE] = 5;
        ctrl[ROCE_CTRL_NOTICE + 1] = 6u;
        tick(0.1); tick(0.2);
    } else if (strcmp(scenario, "flags-in-host-memory") == 0) {
        setup(5, 10 * SECOND);
        set_flag(1, 0, 1, 5); set_flag(1, 1, 1, 5);
        tick(6.0); tick(10.0);
    } else if (strcmp(scenario, "device-bound") == 0) {
        setup(5, 60 * SECOND);
        ctrl[ROCE_CTRL_FAILED] = 1; ctrl[ROCE_CTRL_STOPPED_SEQ] = 5; ctrl[ROCE_CTRL_MISSING_PEER] = 1;
        tick(0.1);
    } else if (strcmp(scenario, "loop-gap") == 0) {
        setup(5, 60 * SECOND);
        ctrl[ROCE_CTRL_DONE] = 5;
        tick(0.1); tick(3.1);
    } else {
        return 2;
    }
    report(scenario);
    return 0;
}
"""


@pytest.fixture(scope="module")
def supervision(tmp_path_factory):
    compiler = shutil.which("gcc")
    if compiler is None or sys.platform != "linux":
        pytest.skip("GCC on Linux compiles the proxy against stand-in verbs")
    directory = tmp_path_factory.mktemp("proxy")
    (directory / "infiniband").mkdir()
    (directory / "infiniband/verbs.h").write_text(FAKE_VERBS)
    shutil.copyfile(PROXY_SOURCE, directory / "proxy_under_test.c")
    (directory / "harness.c").write_text(HARNESS)
    binary = directory / "harness"
    subprocess.run(
        [compiler, "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
         "-I", str(directory), str(directory / "harness.c"), "-o", str(binary), "-lpthread"],
        check=True, capture_output=True, text=True,
    )

    def run(scenario):
        result = subprocess.run([str(binary), scenario], capture_output=True, text=True, check=True)
        return json.loads(result.stdout), result.stderr
    return run


def ctrl_offset(world=2, slot_bytes=4096):
    """Control-record offset of roce_layout for a pair."""
    stride, slots, paths = 128, 2, 2
    return world * slots * slot_bytes + world * slots * paths * stride + slots * slot_bytes


def test_late_peer_is_waited_for_checked_and_logged(supervision):
    result, log = supervision("late-peer")
    assert result["failed"] == 0 and result["abort"] == 0 and result["error"] == ""
    assert result["stalls"] == 1 and result["resolved"] == 1
    assert 7 * 10**9 <= result["longest_stall_ns"] <= 40 * 10**9
    # One check at the stall report and one a second later, each a notice of
    # the awaited sequence in rank 1's slot of the peer's control record.
    checks = [row for row in result["posts"] if row["notice"]]
    assert result["notices_posted"] == len(checks) == 2
    # Rank 0 writes the notice word that the peer's record keeps for rank 0.
    offset = ctrl_offset() + 4 * c_defines({"ROCE_CTRL_NOTICE"})["ROCE_CTRL_NOTICE"]
    assert all(row == {"notice": 1, "path": 0, "offset": offset, "value": 5} for row in checks)
    assert "waited 6.0 s at sequence 5 for rank 1 path 0 (flag 3), rank 1 path 1 (flag 3)" in log
    assert "the wait at sequence 5 ended within 40.0 s" in log


def test_late_peer_across_the_sequence_wrap(supervision):
    result, log = supervision("late-peer-at-wrap")
    assert result["failed"] == 0 and result["stalls"] == 1 and result["resolved"] == 1
    assert "(flag 4294967294)" in log


def test_next_doorbell_ends_a_reported_stall(supervision):
    result, log = supervision("next-doorbell-ends-stall")
    assert result["failed"] == 0 and result["resolved"] == 1
    assert "the wait at sequence 5 ended within 9.0 s" in log


def test_unreachable_peer_stops_every_wait_and_notifies_on_every_path(supervision):
    result, log = supervision("unreachable-peer")
    assert result["failed"] == 1 and result["abort"] == 5
    assert "rank 1 did not acknowledge a peer check on path 0" in result["error"]
    assert "transport retry counter exceeded" in result["error"]
    stops = [row for row in result["posts"] if row["notice"] and row["value"] & 0x80000000]
    assert sorted(row["path"] for row in stops) == [0, 1]
    assert all(row["value"] == 0x80000000 | 5 for row in stops)
    assert "its runtime is poisoned" in log


def test_inconsistent_flags_stop_the_runtime(supervision):
    result, _ = supervision("inconsistent-flags")
    assert result["failed"] == 1 and result["abort"] == 5
    assert "rank 1 path 1 flags hold 9 and 4" in result["error"]
    assert "disagree about the collective sequence" in result["error"]


def test_timeout_stops_a_wait_on_a_live_peer(supervision):
    result, _ = supervision("timeout")
    assert result["failed"] == 1 and result["abort"] == 5
    assert result["error"].startswith("waited 10.0 s at sequence 5, the B12X_ROCE_PEER_TIMEOUT_S limit")
    assert "still acknowledge checks" in result["error"]


def test_peer_stop_notice_stops_this_rank(supervision):
    result, _ = supervision("peer-stopped")
    assert result["failed"] == 1 and result["abort"] == 5
    assert result["error"] == "rank 1 stopped its RoCE runtime at sequence 5 while this rank's doorbell was 5"


def test_waiting_peer_notice_is_logged_once(supervision):
    result, log = supervision("peer-waits-for-this-rank")
    assert result["failed"] == 0 and result["notices_received"] == 1
    assert log.count("rank 1 reports waiting for this rank at sequence 6") == 1
    assert "doorbell is 5, posted 5, completed 5" in log


def test_flags_the_kernel_does_not_observe_are_reported_and_bounded(supervision):
    result, log = supervision("flags-in-host-memory")
    assert "every peer flag for it is in host memory; the kernel has not observed them" in log
    assert not [row for row in result["posts"] if row["notice"] and not row["value"] & 0x80000000]
    assert result["failed"] == 1 and "although every peer flag for it is in host memory" in result["error"]


def test_device_poll_bound_is_propagated_to_the_peers(supervision):
    result, _ = supervision("device-bound")
    assert result["failed"] == 1
    assert "stopped waiting for rank 1 at sequence 5 after its device poll bound" in result["error"]
    assert any(row["value"] == 0x80000000 | 5 for row in result["posts"])


def test_proxy_loop_gap_is_logged(supervision):
    result, log = supervision("loop-gap")
    assert result["failed"] == 0
    assert "the proxy loop made no progress for 3.0 s (doorbell 5, posted 5)" in log


def test_pure_supervision_helpers_compile_on_their_own(tmp_path):
    """The decisions need no verbs: flag states and notice sequences across wraps."""
    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("GCC is unavailable")
    source = PROXY_SOURCE.read_text()
    section = source[source.index("// ROCE_SUPERVISION_BEGIN"):source.index("// ROCE_SUPERVISION_END")]
    defines = "\n".join(
        f"#define {name} {value}u" if name != "ROCE_STALL_REPORT_NS" else f"#define {name} {value}ull"
        for name, value in c_defines({"ROCE_NOTICE_SEQ", "ROCE_STALL_REPORT_NS"}).items()
    )
    cases = [
        # seq, current, other -> state (0 arrived, 1 pending, 2 inconsistent)
        (5, 5, 4, 0), (5, 5, 6, 0), (5, 3, 4, 1), (5, 3, 6, 2), (5, 7, 4, 2), (5, 5, 2, 2),
        (1, 0, 0, 1), (1, 1, 0, 0), (1, 1, 2, 0), (2, 0, 1, 1),
        (0, MASK - 1, MASK, 1), (0, 0, MASK, 0), (0, 0, 1, 0), (1, MASK, 0, 1),
        (0x80000000, 0x7FFFFFFE, 0x7FFFFFFF, 1),
    ]
    notices = [
        # low 31 bits, reference -> sequence
        (6, 5, 6), (4, 5, 4), (6, 0x80000005, 0x80000006), (0x7FFFFFFF, 0x80000000, 0x7FFFFFFF),
        (0x7FFFFFFF, 0xFFFFFFF0, MASK), (0, 0xFFFFFFFF, 0), (1, 0, 1), (0x7FFFFFFF, 0, MASK),
    ]
    program = (
        "#include <stdint.h>\n#include <stdio.h>\n" + defines + "\n" + section + "\n"
        "int main(void) {\n"
        + "".join(f"  printf(\"%d\\n\", roce_flag_state({s}u, {a}u, {b}u));\n" for s, a, b, _ in cases)
        + "".join(f"  printf(\"%u\\n\", roce_notice_seq({n}u, {r}u));\n" for n, r, _ in notices)
        + "  printf(\"%llu %llu\\n\", (unsigned long long)roce_stall_report_ns(300000000000ull),"
        " (unsigned long long)roce_stall_report_ns(4000000000ull));\n"
        "  return 0;\n}\n"
    )
    (tmp_path / "helpers.c").write_text(program)
    subprocess.run([compiler, "-std=gnu11", "-Wall", "-Wextra", "-Werror", str(tmp_path / "helpers.c"),
                    "-o", str(tmp_path / "helpers")], check=True, capture_output=True, text=True)
    lines = subprocess.run([str(tmp_path / "helpers")], check=True, capture_output=True, text=True).stdout.split()
    states = [int(value) for value in lines[: len(cases)]]
    assert states == [expected for *_, expected in cases]
    sequences = [int(value) for value in lines[len(cases): len(cases) + len(notices)]]
    assert sequences == [expected for *_, expected in notices]
    assert lines[-2:] == ["5000000000", "2000000000"]


# -- shared constants and host-side runtime --------------------------------------


def test_kernel_runtime_and_proxy_agree_on_the_control_record():
    words = c_defines({
        "ROCE_CTRL_STOPPED_SEQ", "ROCE_CTRL_MISSING_PEER", "ROCE_CTRL_ABORT", "ROCE_CTRL_DONE",
        "ROCE_CTRL_FAILED", "ROCE_ABI_VERSION",
    })
    kernel = module_constants(ROCE / "_oneshot_cute.py", {"CTRL_ABORT", "CTRL_DONE", "CTRL_FAILED"})
    assert kernel == {
        "CTRL_ABORT": 4 * words["ROCE_CTRL_ABORT"],
        "CTRL_DONE": 4 * words["ROCE_CTRL_DONE"],
        "CTRL_FAILED": 4 * words["ROCE_CTRL_FAILED"],
    }
    runtime = module_constants(ROCE / "roce_oneshot.py", {
        "_CTRL_STOPPED_SEQ", "_CTRL_MISSING_PEER", "_CTRL_ABORT", "_CTRL_DONE", "_CTRL_FAILED",
    })
    assert runtime == {
        "_CTRL_STOPPED_SEQ": words["ROCE_CTRL_STOPPED_SEQ"],
        "_CTRL_MISSING_PEER": words["ROCE_CTRL_MISSING_PEER"],
        "_CTRL_ABORT": words["ROCE_CTRL_ABORT"],
        "_CTRL_DONE": words["ROCE_CTRL_DONE"],
        "_CTRL_FAILED": words["ROCE_CTRL_FAILED"],
    }
    # Existing kernel stores of the stopped sequence and missing peer.
    for name in KERNELS:
        text = (ROCE / name).read_text()
        assert "ctrl_base + Int64(8), seq" in text and "ctrl_base + Int64(12)" in text
        assert "from ._oneshot_cute import CTRL_ABORT" in text or name == "_oneshot_cute.py"
    assert module_constants(ROCE / "_proxy.py", {"ABI_VERSION"})["ABI_VERSION"] == words["ROCE_ABI_VERSION"]


def load_proxy_module():
    spec = importlib.util.spec_from_file_location("prepared_roce_proxy_binding", ROCE / "_proxy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proxy_binding_reports_supervision_counters_and_sets_the_timeout():
    module = load_proxy_module()
    calls = []

    class Library:
        def roce_stat(self, ctx, which):
            return which * 1_000_000

        def roce_two_wave_threshold_bytes(self, ctx):
            return 0

        def roce_wave_mode(self, ctx):
            return 0

        def roce_set_wait_timeout(self, ctx, nanoseconds):
            calls.append(nanoseconds)
            return 0

    proxy = object.__new__(module.Proxy)
    proxy._lib, proxy._ctx = Library(), 1
    stats = proxy.stats()
    assert stats["peer_wait_stalls"] == 5_000_000 and stats["peer_wait_stalls_resolved"] == 6_000_000
    assert stats["longest_resolved_stall_ms"] == 7 and stats["peer_wait_timeout_ms"] == 11
    assert stats["peer_notices_posted"] == 8_000_000 and stats["peer_notices_received"] == 9_000_000
    assert stats["longest_proxy_loop_gap_ms"] == 10
    proxy.set_wait_timeout(2.5)
    assert calls == [2_500_000_000]
    with pytest.raises(RuntimeError, match="positive"):
        proxy.set_wait_timeout(0)
    proxy._ctx = None


def runtime_function(name, env):
    tree = ast.parse((ROCE / "roce_oneshot.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            break
        if isinstance(node, ast.ClassDef) and node.name == "RoceOneshotAllReduce":
            method = next((n for n in node.body if getattr(n, "name", None) == name), None)
            if method is not None:
                node = method
                node.decorator_list = []
                break
    else:
        raise AssertionError(name)
    module = ast.fix_missing_locations(ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    ))
    exec(compile(module, name, "exec"), env)
    return env[name]


def runtime_env():
    import math
    import os

    env = module_constants(ROCE / "roce_oneshot.py", {
        "DEFAULT_SPIN_LIMIT", "DEFAULT_PEER_TIMEOUT_S", "_CTRL_STOPPED_SEQ", "_CTRL_MISSING_PEER",
        "_CTRL_ABORT", "_CTRL_DONE", "_CTRL_FAILED", "_U32",
    })
    env.update(math=math, os=os)
    runtime_function("_env_int", env)
    runtime_function("_env_float", env)
    return env


def test_wait_settings_default_to_host_supervision(monkeypatch):
    env = runtime_env()
    settings = runtime_function("wait_settings", env)
    monkeypatch.delenv("B12X_ROCE_SPIN_LIMIT", raising=False)
    monkeypatch.delenv("B12X_ROCE_PEER_TIMEOUT_S", raising=False)
    assert settings() == (0xFFFFFFFF, 300.0)
    monkeypatch.setenv("B12X_ROCE_PEER_TIMEOUT_S", "45.5")
    monkeypatch.setenv("B12X_ROCE_SPIN_LIMIT", "100000000")
    assert settings() == (100_000_000, 45.5)
    for name, value in (
        ("B12X_ROCE_SPIN_LIMIT", "0"), ("B12X_ROCE_SPIN_LIMIT", str(1 << 32)),
        ("B12X_ROCE_PEER_TIMEOUT_S", "0"), ("B12X_ROCE_PEER_TIMEOUT_S", "-3"),
        ("B12X_ROCE_PEER_TIMEOUT_S", "nan"), ("B12X_ROCE_PEER_TIMEOUT_S", "inf"),
    ):
        monkeypatch.setenv("B12X_ROCE_SPIN_LIMIT", "1")
        monkeypatch.setenv("B12X_ROCE_PEER_TIMEOUT_S", "1")
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            settings()


def test_ranks_agree_on_the_peer_wait_timeout():
    tree = ast.parse((ROCE / "roce_oneshot.py").read_text())
    init = next(
        method
        for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "RoceOneshotAllReduce"
        for method in cls.body if getattr(method, "name", None) == "__init__"
    )
    config = next(
        node.value for node in ast.walk(init)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "config"
    )
    keys = {key.value for key in config.keys}
    assert {"spin_limit", "peer_timeout_s", "proxy_abi"} <= keys
    text = ast.unparse(init)
    assert text.index("self._proxy.set_wait_timeout(self.peer_timeout_s)") < text.index("self._proxy.start()")


def health_runtime(*, failed=False, error="", words=None):
    control = [0] * 32
    for index, value in (words or {}).items():
        control[index] = value
    proxy = NS(failed=lambda: failed, error=lambda: error)
    return NS(_proxy=proxy, _ctrl_np=control, rank=1)


def test_health_reports_the_proxy_verdict_first():
    env = runtime_env()
    check = runtime_function("check_health", env)
    runtime = health_runtime(failed=True, error="rank 0 stopped its RoCE runtime at sequence 9",
                             words={8: 1, 2: 9, 3: 0})
    with pytest.raises(RuntimeError) as raised:
        check(runtime)
    message = str(raised.value)
    assert message.startswith("RoCE transport on rank 1 stopped: rank 0 stopped its RoCE runtime at sequence 9")
    assert "poisoned" in message


@pytest.mark.parametrize("sequence", [5, 0x80000001 - (1 << 32), 0])
def test_health_reports_a_device_stopped_wait_at_any_sequence(sequence):
    env = runtime_env()
    check = runtime_function("check_health", env)
    poisoned = runtime_function("poisoned", env)
    runtime = health_runtime(words={8: 1, 2: sequence, 3: 0})
    assert poisoned(runtime)
    with pytest.raises(RuntimeError) as raised:
        check(runtime)
    expected = sequence & MASK
    assert f"stopped waiting for rank 0 at sequence {expected}" in str(raised.value)
    assert f"its epoch stopped at {(expected - 1) & MASK}" in str(raised.value)
    healthy = health_runtime()
    check(healthy)
    assert not poisoned(healthy)
