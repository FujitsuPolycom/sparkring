"""Environment variables of SIRCL ring sessions.

Every variable is read when a session is constructed; values that affect the
wire protocol are part of the setup agreement, so ranks with different values
fail together. ``SIRCL_COLUMN_GATHER`` is the exception: the vLLM adapter's
executor reads it when a group is set up (the ring harness's adapter path when
it first plans a call), and it must be the same on every rank. SIRCL's four-rank sessions keep their own ``SPARK_TP4_*`` and
``VLLM_SPARK_TP4_*`` names. ``python -m sparkring_sircl.env`` prints the table.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Variable:
    name: str
    meaning: str
    default: str


VARIABLES = (
    Variable("SIRCL_PEER_ROUTES", "route map of the session: <peer>=<device>[/<device>],...", "required"),
    Variable("SIRCL_LAYOUT", "layout identity for route checks: ring:<n>[:<positions>], path:<a>-<b>[:<positions>] "
             "or cables=<cable>,...;positions=<p>,...", "unset (no layout checks)"),
    Variable("SIRCL_DEVICES", "RDMA devices to open, in index order", "the route map's devices"),
    Variable("SIRCL_FABRIC_DOCUMENT", "fabric document (sparkring-fabric/v1) whose port functions name "
             "the RDMA device and network interface of every function; every Spark must name them "
             "alike", "unset: the DGX OS names rocep1s0f0, roceP2p1s0f0, rocep1s0f1, roceP2p1s0f1"),
    Variable("SIRCL_GID_INDEX", "GID index of every device, used verbatim (fallback NCCL_IB_GID_INDEX)",
             "resolved per device"),
    Variable("SIRCL_TRAFFIC_CLASS", "IP DSCP/ECN byte of every queue pair, 0-255 (fallback NCCL_IB_TC)", "0"),
    Variable("SIRCL_SPIN_LIMIT", "flag polls before a wait times out when the command ring holds no wait "
             "limit, and for kernels that count polls", "20000000"),
    Variable("SIRCL_STARTUP_WAIT_S", "flag-wait limit in seconds of the startup regime (compilation, warm-up, "
             "graph capture)", "600"),
    Variable("SIRCL_SERVING_WAIT_S", "flag-wait limit in seconds of the serving regime (enter_serving)", "20"),
    Variable("SIRCL_ALLREDUCE_DISPATCH_LIMIT_BYTES", "dispatch ceiling of should_allreduce", "the capacity"),
    Variable("SIRCL_ALLREDUCE_ALGORITHM", "auto, oneshot, twoshot or swing", "auto"),
    Variable("SIRCL_ONESHOT_MAX_BYTES", "largest auto one-shot message",
             "with a layout, the latency model's limit for it (28672 on a ring of eight farthest first, "
             "73728 on a path of four); else 131072"),
    Variable("SIRCL_LARGE_ALGORITHM", "twoshot or swing above the one-shot limit", "twoshot"),
    Variable("SIRCL_SWING_ABOVE_BYTES", "auto uses Swing above this size; 0 off", "0"),
    Variable("SIRCL_THREADS", "threads per block (multiple of 32, 32-1024; the chain all-reduce needs at "
             "least 64)", "512"),
    Variable("SIRCL_BLOCKS", "largest grid of one-shot and plain all-gather launches (power of two)", "8"),
    Variable("SIRCL_LARGE_BLOCKS", "largest grid of two-shot and large-message launches (power of two)", "32"),
    Variable("SIRCL_FAST_LAUNCH", "0: launch the one-shot, two-shot and all-gather kernels through the CuTe "
             "DSL's per-call argument conversion instead of a prebuilt argument block", "1"),
    Variable("SIRCL_CALL_PROFILE", "eager calls whose stage timestamps the session keeps "
             "(sparkring_sircl.callprofile; call_profile())", "0: off"),
    Variable("SIRCL_CALL_PROFILE_FILE", "path prefix: the call profile's summary goes to <prefix>.rank<rank>.json "
             "every SIRCL_CALL_PROFILE calls and at close", ""),
    Variable("SIRCL_CALL_PROFILE_GPU", "1: also time each profiled launch on the device (CUDA events)", "0"),
    Variable("SIRCL_TUNING_TABLE", "tuning tables (python -m sparkring_sircl.ring tune), comma-separated paths; a "
             "session uses the one whose key matches its group shape and build; none: the rules choose", ""),
    Variable("SIRCL_FLAG_POLLERS", "who polls the peers' flags in the one-shot, two-shot and all-gather "
             "kernels: one-block (block 0, which hands arrival to the other blocks in device memory) or "
             "every-block", "one-block"),
    Variable("SIRCL_LARGE_PIECE_BYTES", "op size of all_reduce_large (multiple of 16); the arena's slots hold it",
             "the larger of 4 MiB and the capacity"),
    Variable("SIRCL_LARGE_SCHEDULE", "all_reduce_large: auto (chain ops on chains of cable neighbors from "
             "SIRCL_CHAIN_MIN_BYTES), chain (always), ring (ring ops over the chain closed by its last rank's "
             "lanes to its first) or pieces (two-shot pieces)", "auto"),
    Variable("SIRCL_CHAIN_MIN_BYTES", "smallest collective auto runs as a chain op (all-reduce message, "
             "all-gather output, reduce-scatter input bytes), one size for all three",
             "all-reduce 8388608, all-gather 8388608, reduce-scatter 4194304 (Sparks 0-3, ring harness runs "
             "20261007-082738 and 20261007-082927)"),
    Variable("SIRCL_RING_MIN_BYTES", "smallest collective a ring schedule runs as a ring op (sizes as "
             "SIRCL_CHAIN_MIN_BYTES), one size for all three; below it the schedule runs as auto",
             "all-reduce 4194304, all-gather 8388608, reduce-scatter 4194304 (the same runs)"),
    Variable("SIRCL_CHAIN_CHUNK_BYTES", "chunk of a chain op (multiple of 16, at most the chain slot); "
             "set_chain_chunk_bytes changes it", "524288"),
    Variable("SIRCL_CHAIN_SLOT_BYTES", "chain ring slot (multiple of 4096): the largest chain chunk", "1048576"),
    Variable("SIRCL_CHAIN_SLOTS", "slots per chain stream ring (2-32)", "4"),
    Variable("SIRCL_CHAIN_BLOCKS", "blocks per role of the chain kernel", "4"),
    Variable("SIRCL_CHAIN_UNROLL", "16-byte packs each thread of the chain kernel moves per pass (1-8)", "4"),
    Variable("SIRCL_COLUMN_GATHER", "1: the vLLM adapter carries an all-gather along a dimension with more than one "
             "row in front of it, which the session would run on its ring or chain as a dimension-0 gather of the "
             "same shard, as that gather into a staging buffer plus one local copy "
             "(sparkring_sircl.vllm.executor.ColumnGather); 0: all_gather_large along that dimension", "1"),
    Variable("SIRCL_GATHER_SCHEDULE", "all_gather_large: auto (chain all-gathers on chains of cable neighbors "
             "when every rank's shard lands in one piece of the output, from SIRCL_CHAIN_MIN_BYTES of output), "
             "chain (whenever the shard qualifies), ring (ring all-gathers whenever the shard qualifies) or "
             "pieces (tiles)", "auto"),
    Variable("SIRCL_SCATTER_SCHEDULE", "reduce_scatter: auto (chain reduce-scatters on chains of cable "
             "neighbors from SIRCL_CHAIN_MIN_BYTES of input), chain (always on chains), ring (ring "
             "reduce-scatters) or pieces (scatter ops)", "pieces"),
    Variable("SIRCL_LINK_CHUNK_BYTES", "piece of a link collective: chain all-gather and reduce-scatter, ring "
             "all-gather, reduce-scatter and all-reduce (multiple of 16, at most the link slot); "
             "set_link_chunk_bytes changes it", "524288"),
    Variable("SIRCL_GATHER_LINK_CHUNK_BYTES", "piece of the chain and ring all-gathers (multiple of 16, at most the "
             "link slot)", "SIRCL_LINK_CHUNK_BYTES"),
    Variable("SIRCL_SCATTER_LINK_CHUNK_BYTES", "piece of the chain and ring reduce-scatters (multiple of 16, at most "
             "the link slot)", "SIRCL_LINK_CHUNK_BYTES"),
    Variable("SIRCL_REDUCE_LINK_CHUNK_BYTES", "piece of the ring all-reduce (multiple of 16, at most the link slot)",
             "SIRCL_LINK_CHUNK_BYTES"),
    Variable("SIRCL_RING_STAGGER", "rounds a ring reduce-scatter's relay leaves after the partial it extends "
             "arrived (0 to 4; needs stagger x (W - 1) + 2 link slots)", "auto: 1 when the link slots hold it, else 0"),
    Variable("SIRCL_RING_GATHER_STAGGER", "rounds a ring all-gather's forward (also in the ring all-reduce) leaves "
             "after the piece it passes on arrived (0 to 4; needs stagger x (W - 1) + 2 link slots)",
             "auto: 1 when the link slots hold it, else 0"),
    Variable("SIRCL_LINK_SLOT_BYTES", "chain link slot (multiple of 4096): the largest link piece",
             "524288, or the largest configured link piece up to 1048576"),
    Variable("SIRCL_LINK_SLOTS", "slots per chain link ring (2-32)", "8"),
    Variable("SIRCL_LINK_BLOCKS", "blocks per role of the link kernels (chain and ring collectives)", "4"),
    Variable("SIRCL_LINK_UNROLL", "16-byte packs each thread of a link kernel moves per pass (1-8)", "4"),
    Variable("SIRCL_FORWARD_WINDOW_BYTES", "largest unacknowledged bytes of a lane through relays; 0 off",
             "131072"),
    Variable("SIRCL_FORWARD_PROOF", "1: bytes of a relayed lane that the op order proves delivered leave its "
             "forward window at once; 0: only their completions free the window", "1"),
    Variable("SIRCL_FORWARD_CHUNK_BYTES", "chunk of a windowed lane's stripe (multiple of 16)", "32768"),
    Variable("SIRCL_HAIRPIN_QUEUE_BYTES", "bytes of one relay hairpin queue (windows keep 75 % of it)", "524288"),
    Variable("SIRCL_PACKS_PER_THREAD", "16-byte packs per thread when sizing the grid, 1-64", "2"),
    Variable("SIRCL_POST_ORDER", "posting order of the progress thread's lanes: rank, ring-farthest, farthest "
             "(most relays first on the session's layout) or an explicit peer list",
             "farthest with a layout, else rank"),
    Variable("SIRCL_POST_MODE", "verbs (direct is unsupported by this build)", "verbs"),
    Variable("SIRCL_TRACE", "phase tracing (unsupported by this build; must be unset or 0)", "0"),
    Variable("SIRCL_EVENT_TRACE", "records kept by the event trace of chain and link ops, native and in the "
             "chain kernel (0: off; AllReduce.event_trace_records reads them)", "0"),
    Variable("SIRCL_PROGRESS_CPU", "CPU list of the progress thread, e.g. 9 or 5-9,15-19", "unpinned"),
    Variable("SIRCL_BUILD_CACHE_DIR", "directory of the built native library", "<XDG cache home>/sircl/roce"),
    Variable("SIRCL_NATIVE_LIBRARY", "a prebuilt native library to load instead of the build cache", "unset"),
    Variable("SIRCL_MAX_RELAYS", "largest relay count of any lane; above 3 is research-only", "3"),
    Variable("SIRCL_FUSED_NORM_MAX_ROWS", "rows per fused all-reduce + RMSNorm launch, at most the GPU's "
             "multiprocessors (every CTA of a launch must be resident)", "the multiprocessor count"),
    Variable("SIRCL_TOPOLOGY", "not read except to refuse values other than direct", "unset"),
)


def describe() -> tuple[Variable, ...]:
    return VARIABLES


if __name__ == "__main__":  # pragma: no cover
    for variable in VARIABLES:
        print(f"{variable.name:40} {variable.default:28} {variable.meaning}")
