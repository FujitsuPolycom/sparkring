"""Worker-side behavior of glm_dcp_decode_comm: streams, the query path, the selection reuse and the combine.

Imported by the helpers that ``__init__.py`` places into
``vllm.models.deepseek_v32.attention`` at the first decode-layer call in a
worker. Every function here decides per layer call whether its item applies
(:func:`_active`) and otherwise runs exactly the statement the image would
have run, so a declined call costs one attribute lookup and a few checks.

A layer call applies when all of these hold: the item's flag is set; the layer
is a DSA attention layer with decode context parallelism (``dcp_manager``,
``impl.dcp_world_size > 1``, no prefill context parallelism, B12X sparse MLA
with a BF16 query, no B12X PCIe transport, no padded heads); the batch is
decode only (``num_prefills == 0`` and every token a decode token); no
torch.compile tracing and no breakable piecewise capture is active.

The session. On such a layer the DCP group's device communicator must be
SIRCL's (``sparkring_sircl.vllm.communicator.SirclCudaCommunicator``)
holding a DCP session: its group adapter (``communicator.sircl``) carries a
``SirclDcpCollectives`` (``.dcp``) whose runtime (``._runtime``, the adapter's
``.session``) is a ``sparkring_sircl.oneshot.runtime.RoceOneshotAllReduce``
with the scatter collectives, all loaded from the files whose SHA-256
``FILE_CHECKS`` records. Any other communicator or session raises
:class:`PatchRefused` at the layer's first decode call on every rank of the
group: the plugin refuses to run on a session it was not built against
(:func:`_session_check`). Model configurations that the items do not cover
leave the layer unchanged instead (:func:`_static_check`).

Rank invariance. Every decision that changes which SIRCL ops a rank issues
depends only on settings, shapes, dtypes, strides, the batch's metadata and the
session's voted limits, never on a pointer value, so all ranks of a DCP group
issue the same ops in the same order. Pointer alignment is repaired on the
rank that needs it (a contiguous copy) and never changes the plan.

Audit mode (``Settings.audit``) keeps the model on the exact paths and adds,
inside the same eager or captured decode forward, the image's computation of
each value an exact item replaces, compared word for word on the device
(``EXACT_CHECKS``). Counters live in device memory, so captured graphs keep
counting on every replay; ``audit_tick`` logs them from an eager forward at
most every ``AUDIT_INTERVAL_S`` seconds. The image's reference query gather
and all-to-all are extra DCP collectives that every rank issues in the same
order, in eager calls only, so audit mode is for qualification, not for
timing. Inside a CUDA graph capture audit mode adds only comparisons on the
capturing stream; the checks that need a reference collective
(``gathered_query``, ``combine``) count eager calls.

Side streams. Every fork onto the communication or wk stream is joined into
the forking stream before the layer's ``forward`` returns: ``wk_join`` and
``gather`` join what they consume, and ``finish`` (the edited ``return`` of
``forward``) joins and drops whatever a call left unconsumed.
"""

from __future__ import annotations

import atexit
import functools
import logging
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import PatchRefused
from . import layout as L


def _kernels():
    """The Triton kernels (imported in the worker on first use; Triton is not needed to import this module)."""
    from . import kernels

    return kernels


PLUGIN_NAME = "glm_dcp_decode_comm"
_LOG = logging.getLogger("vllm." + PLUGIN_NAME)
_LOCK = threading.Lock()
_ONCE: set[str] = set()
COUNTS: dict[str, int] = {}

# The SIRCL classes the plugin was built against (SIRCL_VERSION): (module, class name).
SIRCL_COMMUNICATOR = ("sparkring_sircl.vllm.communicator", "SirclCudaCommunicator")
SIRCL_DCP_COLLECTIVES = ("sparkring_sircl.vllm.dcp_collectives", "SirclDcpCollectives")
SIRCL_SESSION = ("sparkring_sircl.oneshot.runtime", "RoceOneshotAllReduce")
SIRCL_SHIM_MARKER = "_sircl_original"     # SIRCL's shims.MARKER on a function a shim replaced
SIRCL_UNKNOWN = "(version not configured)"  # a refusal before configure() names no SIRCL version


def _log_once(key: str, level: int, message: str, *args) -> None:
    with _LOCK:
        if key in _ONCE:
            return
        _ONCE.add(key)
    _LOG.log(level, message, *args)


def _count(key: str, n: int = 1) -> None:
    with _LOCK:
        COUNTS[key] = COUNTS.get(key, 0) + n


# ----------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class Settings:
    overlap: bool = False
    wk_overlap: bool = False
    query_pack: bool = False
    selection_reuse: bool = False
    a2a_fused: bool = False
    comm_priority: int = -1
    audit: bool = False
    # The installed SIRCL package's directory and version (``register`` verified their files).
    sircl_root: str = ""
    sircl_version: str = ""

    @property
    def query_path(self) -> bool:
        """The plugin builds the layer's query (the overlap implies the query pack)."""
        return self.query_pack or self.overlap

    @property
    def comm_stream(self) -> bool:
        """Every DCP collective runs on the DCP communication stream."""
        return self.overlap

    def describe(self) -> str:
        items = [name for name, on in (
            ("overlap", self.overlap), ("wk_overlap", self.wk_overlap), ("query_pack", self.query_path),
            ("selection_reuse", self.selection_reuse), ("a2a_fused", self.a2a_fused), ("audit", self.audit)) if on]
        return ", ".join(items) + (f"; communication stream priority {self.comm_priority}"
                                   if self.overlap else "")


SETTINGS = Settings()


def configure(settings: Settings) -> None:
    global SETTINGS
    SETTINGS = settings


# ----------------------------------------------------------------------------- streams


_STREAMS: dict[tuple[str, int], torch.cuda.Stream] = {}


def side_stream(kind: str, device: torch.device) -> torch.cuda.Stream:
    """The process's ``comm`` (DCP collectives) or ``wk`` stream for ``device``, created on first use."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (kind, index)
    stream = _STREAMS.get(key)
    if stream is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{PLUGIN_NAME}: the {kind} stream must be created before CUDA graph capture")
        priority = SETTINGS.comm_priority if kind == "comm" else 0
        stream = torch.cuda.Stream(device=torch.device("cuda", index), priority=priority)
        _STREAMS[key] = stream
        _LOG.info("%s: %s stream created on cuda:%d (priority %d)", PLUGIN_NAME, kind, index, priority)
    return stream


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 - no CUDA
        return False


def _breakable_capture() -> bool:
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        return BreakableCUDAGraphCapture.current() is not None
    except Exception:  # noqa: BLE001 - unknown capture context: treat as breakable and decline
        return True


# ----------------------------------------------------------------------------- SIRCL's DCP session


def _is(obj, identity: tuple[str, str]) -> bool:
    """Whether ``obj``'s class is, or derives from, the class ``identity`` names."""
    module, name = identity
    return any(c.__module__ == module and c.__name__ == name for c in type(obj).__mro__)


def _dcp_parts(communicator):
    """``(adapter, collectives, session)`` of a SIRCL communicator that owns a DCP session, else None."""
    adapter = getattr(communicator, "sircl", None)
    dcp = getattr(adapter, "dcp", None)
    session = getattr(adapter, "session", None)
    if dcp is None or session is None or getattr(dcp, "_runtime", None) is not session:
        return None
    return adapter, dcp, session


def _loaded_from(module_name: str, relative: str) -> str | None:
    """Why ``module_name`` is not loaded from ``relative`` inside the verified SIRCL directory, else None."""
    root = SETTINGS.sircl_root
    module = sys.modules.get(module_name)
    if not root or module is None:
        return None
    path = Path(getattr(module, "__file__", "") or "")
    expected = Path(root) / relative
    try:
        if path.resolve() == expected.resolve():
            return None
    except OSError:
        pass
    return f"{module_name} is loaded from {path or 'nowhere'}, not from the verified file {expected}"


def _session_check(communicator) -> None:
    """Refuse (:class:`PatchRefused`) a DCP group whose communicator or session this plugin was not built for."""
    if not _is(communicator, SIRCL_COMMUNICATOR):
        raise PatchRefused(f"{PLUGIN_NAME}: the DCP group's device communicator is "
                           f"{type(communicator).__module__}.{type(communicator).__qualname__}, not SIRCL's "
                           f"{'.'.join(SIRCL_COMMUNICATOR)}; this plugin runs only on SIRCL "
                           f"{SETTINGS.sircl_version or SIRCL_UNKNOWN} DCP sessions (set the GLM_DCP_DECODE_* flags to 0)")
    parts = _dcp_parts(communicator)
    if parts is None:
        raise PatchRefused(f"{PLUGIN_NAME}: the DCP group's SIRCL communicator holds no DCP session "
                           "(SIRCL_GROUPS must include dcp); set the GLM_DCP_DECODE_* flags to 0")
    _, dcp, session = parts
    problems = []
    if not _is(dcp, SIRCL_DCP_COLLECTIVES):
        problems.append(f"its DCP collectives are {type(dcp).__module__}.{type(dcp).__qualname__}")
    if not _is(session, SIRCL_SESSION):
        problems.append(f"its session is {type(session).__module__}.{type(session).__qualname__}")
    for module_name, relative in ((SIRCL_COMMUNICATOR[0], "vllm/communicator.py"),
                                  (SIRCL_DCP_COLLECTIVES[0], "vllm/dcp_collectives.py"),
                                  (SIRCL_SESSION[0], "oneshot/runtime.py")):
        problem = _loaded_from(module_name, relative)
        if problem:
            problems.append(problem)
    package = sys.modules.get("sparkring_sircl")
    version = getattr(package, "__version__", None)
    if SETTINGS.sircl_version and package is not None and version != SETTINGS.sircl_version:
        problems.append(f"sparkring_sircl {version} is loaded, the plugin was built against {SETTINGS.sircl_version}")
    if not getattr(session, "scatter_available", False):
        problems.append("the session has no scatter collectives")
    if problems:
        raise PatchRefused(f"{PLUGIN_NAME}: refusing the DCP group's SIRCL session: " + "; ".join(problems))


def _dcp_session_of(communicator):
    """The DCP session that ``communicator``'s collectives run on, or None.

    A group shares another group's session when their global ranks are equal
    (``adapter._share_session``); its collectives then run on that DCP session
    too, so they go on the communication stream with the DCP group's.
    """
    adapter = getattr(communicator, "sircl", None)
    if adapter is None:
        return None
    parts = _dcp_parts(communicator)
    if parts is not None:
        return parts[2]
    session = getattr(adapter, "session", None)
    if session is None or getattr(adapter, "shared_from", None) is None:
        return None
    try:
        from sparkring_sircl.vllm import adapter as adapter_module

        owners = adapter_module.live_adapters()
    except Exception:  # noqa: BLE001 - adapter module not loaded
        return None
    for owner in owners:
        if getattr(owner, "dcp", None) is not None and getattr(owner, "session", None) is session:
            return session
    return None


# ----------------------------------------------------------------------------- communicator wrapper


# Every collective of SIRCL's communicator that can run on a group's session. Point-to-point calls use the
# group's channels, not its session, and stay where they are.
_WRAPPED_METHODS = ("all_reduce", "all_reduce_in_place", "all_gather", "all_gatherv", "reduce_scatter",
                    "reduce_scatterv", "gather", "broadcast", "all_to_all_single")
WRAP_MARKER = "__glm_dcp_decode_comm_stream__"


def _tensors(values) -> list[torch.Tensor]:
    out = []
    for value in values:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            out.append(value)
        elif isinstance(value, (list, tuple)):
            out.extend(_tensors(value))
    return out


def _on_comm_stream(method):
    @functools.wraps(method)
    def run(self, *args, **kwargs):
        if not SETTINGS.comm_stream or torch.compiler.is_compiling() or _dcp_session_of(self) is None:
            return method(self, *args, **kwargs)
        main = torch.cuda.current_stream(self.device)
        comm = side_stream("comm", self.device)
        if main == comm:
            return method(self, *args, **kwargs)
        for tensor in _tensors(list(args) + list(kwargs.values())):
            tensor.record_stream(comm)
        comm.wait_stream(main)
        with torch.cuda.stream(comm):
            result = method(self, *args, **kwargs)
        main.wait_stream(comm)
        for tensor in _tensors([result]):
            tensor.record_stream(main)
        _count(f"comm_stream.{method.__name__}")
        return result

    setattr(run, WRAP_MARKER, True)
    return run


def install_comm_stream(communicator_cls) -> bool:
    """Run every collective of a DCP session's communicator on the DCP communication stream.

    Wraps the collective methods of SIRCL's ``SirclCudaCommunicator``; a wrapped
    method moves a call only when its communicator's group runs on a DCP session
    (:func:`_dcp_session_of`) and the overlap is on, so the tensor-parallel and
    every other group keep their streams. A SIRCL session requires every
    collective of one CUDA graph capture on one stream
    (``RoceOneshotAllReduce._order_stream``); putting all of a DCP session's
    collectives on this stream keeps that rule while the attention layer runs
    one of them concurrently with independent work. Returns False if already
    installed.
    """
    done = False
    for name in _WRAPPED_METHODS:
        current = communicator_cls.__dict__.get(name)
        if current is None:
            raise PatchRefused(f"{PLUGIN_NAME}: {communicator_cls.__qualname__}.{name} not found")
        if getattr(current, WRAP_MARKER, False):
            continue
        setattr(communicator_cls, name, _on_comm_stream(current))
        done = True
    return done


# ----------------------------------------------------------------------------- audit

# Exact items, each against the image's computation of the same value:
#   query           this rank's DCP query (rope_cat) vs fused_q + torch.cat;
#   gathered_query  the query the attention reads vs the image's query_gather of the image's query;
#   wk              the wk GEMM output joined from the wk stream vs the GEMM on the main stream;
#   selection       a reused DCP index selection (indices and counts) vs a fresh filter call;
#   combine         the fused all-to-all combine vs the image's combine (pack, exchange, combine).
EXACT_CHECKS = ("query", "gathered_query", "wk", "selection", "combine")
AUDIT_LAYERS = 128  # counter rows; GLM-5.3 has 78 target attention layers and one MTP layer
AUDIT_INTERVAL_S = 10.0
_WORD_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


class _Audit:
    """Device-resident audit counters of one GPU, allocated outside CUDA graph capture.

    ``exact[layer, check]`` holds (calls, compared words, differing words).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.exact = torch.zeros((AUDIT_LAYERS, len(EXACT_CHECKS), 3), dtype=torch.int64, device=device)
        self.slots: dict[str, int] = {}
        self.reported_calls = 0

    def slot(self, layer) -> int:
        name = str(getattr(layer, "layer_name", None) or f"layer@{id(layer):x}")
        slot = self.slots.get(name)
        if slot is None:
            slot = min(len(self.slots), AUDIT_LAYERS - 1)
            if slot == AUDIT_LAYERS - 1:
                _log_once("audit-rows", logging.WARNING, "%s: more than %d audited layers; the last counter row "
                          "is shared", PLUGIN_NAME, AUDIT_LAYERS - 1)
            self.slots[name] = slot
        return slot

    def report(self, *, force: bool = False) -> dict:
        """Log the counters (one host copy, which waits for the current stream) if they changed."""
        exact = self.exact.cpu()
        calls = int(exact[..., 0].sum())
        summary = _audit_summary(exact, self.slots)
        if calls == self.reported_calls and not force:
            return summary
        self.reported_calls = calls
        where = f"cuda:{self.device.index}"
        for name, item in summary.items():
            level = logging.WARNING if item["differing_words"] else logging.INFO
            _LOG.log(level, "%s audit %s: %s: %d calls, %d words compared, %d differ%s", PLUGIN_NAME, where, name,
                     item["calls"], item["compared_words"], item["differing_words"],
                     f" (layers {', '.join(item['layers'])})" if item["layers"] else "")
        return summary


def _audit_summary(exact: torch.Tensor, slots: dict[str, int]) -> dict:
    names: dict[int, list[str]] = {}
    for name, slot in slots.items():
        names.setdefault(slot, []).append(name)
    out: dict = {}
    for i, check in enumerate(EXACT_CHECKS):
        cells = exact[:, i]
        calls = int(cells[:, 0].sum())
        if calls == 0:
            continue
        bad_rows = [int(r) for r in torch.nonzero(cells[:, 2]).flatten()]
        out[check] = {
            "calls": calls, "compared_words": int(cells[:, 1].sum()), "differing_words": int(cells[:, 2].sum()),
            "layers": ["/".join(names.get(r, [f"row {r}"])) for r in bad_rows[:4]],
        }
    return out


_AUDITS: dict[int, _Audit] = {}
_AUDIT_NEXT = [0.0]
_AUDIT_EXIT_HOOK: list[bool] = []


def _audit_for(device: torch.device) -> _Audit | None:
    index = device.index if device.index is not None else torch.cuda.current_device()
    audit = _AUDITS.get(index)
    if audit is None:
        if _capturing():
            _log_once("audit-capture", logging.WARNING, "%s: audit counters are created at the first eager "
                      "decode forward; a graph captured before it is not audited", PLUGIN_NAME)
            return None
        audit = _Audit(torch.device("cuda", index))
        with _LOCK:
            _AUDITS[index] = audit
            hook = not _AUDIT_EXIT_HOOK
            _AUDIT_EXIT_HOOK.append(True)
        if hook:
            atexit.register(_audit_at_exit)
        _LOG.info("%s: audit counters on cuda:%d (exact: %s)", PLUGIN_NAME, index, ", ".join(EXACT_CHECKS))
    return audit


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(_WORD_DTYPES[t.element_size()]).reshape(-1)


def audit_exact(layer, check: str, got, ref) -> None:
    """Count the words in which ``got`` differs from ``ref`` (tensors or equal-length tuples of tensors)."""
    if isinstance(got, torch.Tensor):
        got, ref = (got,), (ref,)
    first = got[0]
    audit = _audit_for(first.device)
    if audit is None:
        return
    cell = audit.exact[audit.slot(layer), EXACT_CHECKS.index(check)]
    cell[0].add_(1)
    for a, b in zip(got, ref, strict=True):
        cell[1].add_(a.numel())
        if a.shape != b.shape or a.dtype != b.dtype:
            _log_once(f"audit-shape:{check}", logging.WARNING, "%s audit: %s: %s %s against %s %s", PLUGIN_NAME,
                      check, tuple(a.shape), a.dtype, tuple(b.shape), b.dtype)
            cell[2].add_(a.numel())
            continue
        cell[2].add_(torch.ne(_words(a), _words(b)).sum())


def audit_tick(*, force: bool = False) -> None:
    """Log every GPU's audit counters if ``AUDIT_INTERVAL_S`` passed (eager forwards only)."""
    if not _AUDITS or torch.compiler.is_compiling() or _capturing():
        return
    now = time.monotonic()
    if not force and now < _AUDIT_NEXT[0]:
        return
    _AUDIT_NEXT[0] = now + AUDIT_INTERVAL_S
    for audit in list(_AUDITS.values()):
        audit.report(force=force)


def _audit_at_exit() -> None:
    try:
        audit_tick(force=True)
    except Exception:  # noqa: BLE001 - the CUDA context may already be gone at exit
        pass


# ----------------------------------------------------------------------------- per-layer state


@dataclass
class _Stash:
    """The query of one layer call prepared by ``query``."""

    rows: int
    q_cat: torch.Tensor                    # [rows, H, 576] BF16
    source: int = 0                        # data_ptr of the ql_nope the query was built from
    gathered: torch.Tensor | None = None   # gather output [rows, world * H, 576] (issued early or late)
    issued: bool = False
    workspace: bool = False                # gathered is the attention workspace's query view
    keep: list = field(default_factory=list)


def _runs_indexer(layer) -> bool:
    """Whether this call of ``forward`` runs the DSA indexer: the guard the image's ``forward`` uses.

    Evaluated per call: the MTP speculator sets ``skip_topk`` on the draft
    layer between draft steps (step 0 computes the top-k, steps 1+ reuse it).
    """
    return getattr(layer, "indexer", None) is not None and not getattr(layer, "skip_topk", True)


class _Layer:
    """Static facts and per-call state of one attention layer, kept on the module."""

    def __init__(self, layer):
        self.ok, self.reason = _static_check(layer)
        self.stash: _Stash | None = None
        self.wk: tuple | None = None
        self.provider_installed = False

    @staticmethod
    def of(layer) -> "_Layer":
        state = layer.__dict__.get("_glm_dcp_decode_comm_state")
        if state is None:
            state = _Layer(layer)
            object.__setattr__(layer, "_glm_dcp_decode_comm_state", state)
            if state.ok:
                indexer = _runs_indexer(layer)
                _log_once(f"layer:{indexer}", logging.INFO,
                          "%s: %s (first: %s): %s", PLUGIN_NAME,
                          "DSA indexer layers" if indexer else "layers without the DSA indexer",
                          getattr(layer, "layer_name", "?"), SETTINGS.describe())
            else:
                _log_once(f"declined:{state.reason}", logging.INFO, "%s: %s left unchanged: %s",
                          PLUGIN_NAME, getattr(layer, "layer_name", "?"), state.reason)
        return state


def _static_check(layer) -> tuple[bool, str]:
    """Whether the items cover this layer's configuration; raises for a DCP session it was not built for."""
    impl = getattr(layer, "impl", None)
    manager = getattr(layer, "dcp_manager", None)
    if impl is None or manager is None or int(getattr(impl, "dcp_world_size", 1)) <= 1:
        return False, "no decode context parallelism"
    if getattr(layer, "use_pcp", False):
        return False, "prefill context parallelism"
    if not any(c.__name__ == "B12xMLASparseImpl" for c in type(impl).__mro__):
        return False, f"attention implementation {type(impl).__name__} is not B12X sparse MLA"
    if getattr(layer, "_fp8_query", False):
        return False, "FP8 query path"
    if getattr(manager, "b12x_transport", None) is not None:
        return False, "B12X PCIe DCP transport"
    if getattr(manager, "padded_num_heads", None) is not None:
        return False, "padded query heads"
    if int(layer.kv_lora_rank) != L.QL_NOPE_DIM or int(layer.qk_rope_head_dim) != L.ROPE_DIM:
        return False, "latent or RoPE width differs from 512 + 64"
    _session_check(getattr(manager.group, "device_communicator", None))
    return True, ""


def _metadata(layer):
    from vllm.model_executor.layers.attention.attention import get_attention_context

    return get_attention_context(layer.layer_name)


def _decode_rows(layer) -> int:
    """Rows of a decode-only batch for this layer, else 0."""
    try:
        metadata, _, _, _ = _metadata(layer)
    except Exception:  # noqa: BLE001 - no forward context
        return 0
    if metadata is None:
        return 0
    actual = int(getattr(metadata, "num_actual_tokens", 0) or 0)
    if actual <= 0 or int(getattr(metadata, "num_prefills", 1)) != 0:
        return 0
    if int(getattr(metadata, "num_decode_tokens", -1)) != actual:
        return 0
    return actual


def _active(layer) -> tuple[_Layer | None, int]:
    state = _Layer.of(layer)
    if not state.ok or torch.compiler.is_compiling():
        return None, 0
    if _capturing() and _breakable_capture():
        return None, 0
    rows = _decode_rows(layer)
    if rows <= 0:
        return None, 0
    return state, rows


def _comm(layer):
    return layer.dcp_manager.group.device_communicator


def _dcp(layer):
    """The layer's ``SirclDcpCollectives`` (``_session_check`` admitted it)."""
    return _comm(layer).sircl.dcp


# ----------------------------------------------------------------------------- wk overlap


def wk_fork(layer, hidden_states) -> None:
    """Start the indexer's wk GEMM on the wk stream (DSA indexer layers, decode batches)."""
    if not SETTINGS.wk_overlap:
        return
    state, rows = _active(layer)
    if state is None or not _runs_indexer(layer):
        return
    main = torch.cuda.current_stream(hidden_states.device)
    side = side_stream("wk", hidden_states.device)
    hidden_states.record_stream(side)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        kw = layer.indexer.wk_weights_proj(hidden_states)[0]
    state.wk = (hidden_states, kw)
    _count("wk_fork")


def wk_join(layer, hidden_states):
    """The indexer's wk output: joined from the wk stream, or computed here as the image does."""
    state = layer.__dict__.get("_glm_dcp_decode_comm_state")
    pending = state.wk if state is not None else None
    if pending is not None:
        state.wk = None
        source, kw = pending
        main = torch.cuda.current_stream(kw.device)
        main.wait_stream(side_stream("wk", kw.device))  # joined on every path
        if source is hidden_states:
            kw.record_stream(main)
            if SETTINGS.audit:
                audit_exact(layer, "wk", kw, layer.indexer.wk_weights_proj(hidden_states)[0])
            return kw
    return layer.indexer.wk_weights_proj(hidden_states)[0]


def finish(layer) -> None:
    """End of ``forward``: join side-stream work this call forked and did not consume.

    ``wk_join`` and ``gather`` normally join the wk fork and the early query
    gather. When a call leaves either unconsumed (the image's ``forward``
    skipped the statement that consumes it), this joins it into the current
    stream, so a CUDA graph capture never ends with an unjoined stream, and
    drops it, so no later call sees it.
    """
    state = layer.__dict__.get("_glm_dcp_decode_comm_state")
    if state is None or (state.wk is None and state.stash is None):
        return
    if state.wk is not None:
        _, kw = state.wk
        state.wk = None
        torch.cuda.current_stream(kw.device).wait_stream(side_stream("wk", kw.device))
        _count("wk_unconsumed")
        _log_once("wk-unconsumed", logging.WARNING, "%s: %s: a wk GEMM forked onto the wk stream was not "
                  "used by its call; joined at the end of forward", PLUGIN_NAME, getattr(layer, "layer_name", "?"))
    if state.stash is not None:
        _drop(layer, state)
        _count("query_unconsumed")
        _log_once("query-unconsumed", logging.WARNING, "%s: %s: a prepared DCP query was not used by its call; "
                  "dropped at the end of forward", PLUGIN_NAME, getattr(layer, "layer_name", "?"))


def _drop(layer, state: "_Layer") -> None:
    """Drop the prepared query, joining its early gather into the current stream if it was issued."""
    stash, state.stash = state.stash, None
    if stash is not None and stash.issued and SETTINGS.comm_stream and stash.gathered is not None:
        device = stash.gathered.device
        torch.cuda.current_stream(device).wait_stream(side_stream("comm", device))


# ----------------------------------------------------------------------------- query pack and overlap


def gather_fits_one_op(dcp, nbytes: int, row_bytes: int) -> bool:
    """Whether a decode shard of ``nbytes`` (rows of ``row_bytes``) is one SIRCL all-gather op.

    The decision ``SirclDcpCollectives.all_gather`` makes for a shard
    gathered along its last dimension (one op while it fits a sub-gather), from
    the group's voted limits and the shard's shape only.
    """
    runtime = dcp._runtime
    return (runtime is not None and nbytes <= min(int(dcp.gather_chunk), int(runtime.max_gather_bytes))
            and nbytes % L.PACK_BYTES == 0 and row_bytes % L.PACK_BYTES == 0)


def _gather_one_op(layer, src_rows: torch.Tensor, out_rows: torch.Tensor) -> bool:
    """All-gather ``src_rows`` (``[R, X]``) into ``out_rows`` (``[R, world * X]``) as one SIRCL op.

    The runtime call ``SirclDcpCollectives.all_gather`` makes for a decode shard
    (a last-dimension gather of the ``[rows, inner]`` view), with the output
    placed where the caller wants it. False when it does not fit one op.
    """
    dcp = _dcp(layer)
    nbytes = src_rows.numel() * src_rows.element_size()
    if not gather_fits_one_op(dcp, nbytes, int(src_rows.shape[-1]) * src_rows.element_size()):
        return False
    dcp._runtime.all_gather(src_rows, dim=-1, out=out_rows)
    return True


def _workspace_query(layer, rows: int, total_heads: int) -> torch.Tensor | None:
    """The attention workspace's query view ``[rows, total_heads, 576]`` (``forward_mqa``'s), or None.

    ``forward_mqa`` sizes its query buffer for ``_kernel_num_heads`` heads (the gathered heads rounded up to
    whole groups of eight) and skips its copy only for an exact alias of the first ``total_heads`` of them.
    That alias is a contiguous buffer, which the gather writes, only when no head is added; otherwise the
    gathered query gets a buffer of its own (a shape decision, the same on every rank).
    """
    impl = layer.impl
    kernel_heads = int(getattr(impl, "_kernel_num_heads", total_heads))
    if kernel_heads != total_heads:
        return None
    view = impl._borrow_workspaces(input_num_heads=kernel_heads)[0]
    if (view.dim() != 3 or int(view.shape[0]) < rows or int(view.shape[1]) != total_heads
            or int(view.shape[2]) != L.HEAD_DIM or view.dtype != torch.bfloat16):
        return None
    view = view[:rows]
    return view if view.is_contiguous() else None


def _issue_gather(layer, state: _Layer, stash: _Stash, *, early: bool) -> None:
    """Gather the stashed query on the communication stream (or the current stream without the overlap)."""
    rows = stash.rows
    world = int(layer.impl.dcp_world_size)
    heads = int(layer.num_heads)
    device = layer.W_UK_T.device
    src = stash.q_cat[:rows].view(rows, heads * L.HEAD_DIM)
    dest = _workspace_query(layer, rows, world * heads) if stash.workspace else None
    if dest is None:
        stash.workspace = False
        dest = torch.empty((rows, world * heads, L.HEAD_DIM), dtype=torch.bfloat16, device=device)
    out_rows = dest.view(rows, world * heads * L.HEAD_DIM)
    main = torch.cuda.current_stream(device)
    if SETTINGS.comm_stream:
        comm = side_stream("comm", device)
        src.record_stream(comm)
        dest.record_stream(comm)
        comm.wait_stream(main)
        context = torch.cuda.stream(comm)
    else:
        context = torch.cuda.stream(main)
    with context:
        if not _gather_one_op(layer, src, out_rows):
            # Above one op: SIRCL's communicator plans the gather (on this stream).
            gathered = _comm(layer).all_gather(src, dim=-1)
            out_rows.copy_(gathered.view(out_rows.shape))
            _count("gather_split_ops")
    stash.gathered = dest
    stash.issued = True
    stash.keep.append(src)
    _count("gather_early" if early else "gather_late")


def query(layer, positions, q_pe, ql_nope) -> None:
    """Build this layer call's DCP query right after W_UK; with the overlap, also issue its gather here."""
    settings = SETTINGS
    if settings.audit:
        audit_tick()
    if not settings.query_path:
        return
    state, rows = _active(layer)
    if state is None:
        return
    heads = int(layer.num_heads)
    q_cat = torch.empty((rows, heads, L.HEAD_DIM), dtype=torch.bfloat16, device=ql_nope.device)
    _kernels().rope_cat(positions, q_pe, layer.rotary_emb.cos_sin_cache, ql_nope, q_cat, rows)
    # The attention workspace may hold the gathered query unless the DSA
    # indexer, which borrows the same workspace, runs before the attention.
    workspace = not (settings.overlap and _runs_indexer(layer))
    stash = _Stash(rows=rows, q_cat=q_cat, source=ql_nope.data_ptr(), workspace=workspace)
    state.stash = stash
    if settings.overlap:
        _issue_gather(layer, state, stash, early=True)


def _claim(layer, mqa_q_arg) -> _Stash | None:
    """The prepared query of this attention call, or None (a stale one is dropped).

    ``query`` prepares the query in ``forward``; the attention call that
    follows passes ``(ql_nope[:rows], q_pe[:rows])``. A prepared query whose
    ``ql_nope`` or row count differs from that argument belongs to a call
    that never reached the gather (an early return) and is discarded, so this
    call runs the image's path. The identity check reads a pointer of this
    rank's own tensors; every rank reaches the same statements, so the
    outcome is the same on every rank.
    """
    state = layer.__dict__.get("_glm_dcp_decode_comm_state")
    stash = state.stash if state is not None else None
    if stash is None or mqa_q_arg is None:
        return stash
    first = mqa_q_arg[0] if isinstance(mqa_q_arg, tuple) else None
    if first is not None and first.data_ptr() == stash.source and int(first.shape[0]) == stash.rows:
        return stash
    _drop(layer, state)
    _count("query_stale")
    _log_once("query-stale", logging.WARNING, "%s: a prepared DCP query did not match its attention call "
              "(%s); that call ran the image's query path", PLUGIN_NAME, getattr(layer, "layer_name", "?"))
    return None


def has_query(layer, mqa_q_arg=None) -> bool:
    return _claim(layer, mqa_q_arg) is not None


def _skip_fused_q(stash: _Stash):
    def fused_q_replacement(positions, q_pe_arg, *args, **kwargs):
        """Layers without the DSA indexer: the query is already built; nothing to launch.

        ``fused_q``'s first two outputs belong to the indexer, which these
        layers do not run; its third, the RoPE'd ``q_pe``, only feeds the
        query concatenation, which ``has_query`` makes the attention skip.
        """
        return None, None, stash.q_cat[:, :, L.QL_NOPE_DIM:]

    return fused_q_replacement


def fused_q_fn(layer, image_fused_q):
    """The image's ``fused_q``, or a no-op for a layer without the DSA indexer whose query is built.

    Audit mode keeps the image's ``fused_q`` on every layer: its RoPE'd
    ``q_pe`` is the reference for the plugin's query.
    """
    state = layer.__dict__.get("_glm_dcp_decode_comm_state")
    if state is None or state.stash is None or _runs_indexer(layer) or SETTINGS.audit:
        return image_fused_q
    _count("fused_q_skipped")
    return _skip_fused_q(state.stash)


def _audit_query(layer, stash: _Stash, mqa_q_arg, gathered: torch.Tensor) -> None:
    """Audit mode: this rank's query and the gathered query against the image's."""
    rows = stash.rows
    if not isinstance(mqa_q_arg, tuple):
        _log_once("audit-query-arg", logging.WARNING, "%s audit: the attention's query argument is not the "
                  "(ql_nope, q_pe) pair; query checks skipped", PLUGIN_NAME)
        return
    image_q = torch.cat(mqa_q_arg, dim=-1)  # the image's statement on the image's fused_q output
    audit_exact(layer, "query", stash.q_cat[:rows], image_q[:rows])
    if not _capturing():  # the reference gather is a DCP collective: eager calls only
        audit_exact(layer, "gathered_query", gathered, layer.dcp_manager.query_gather(image_q)[:rows])


def gather(layer, mqa_q_arg):
    """The gathered query for the attention kernel; the image's ``query_gather`` when not prepared here."""
    stash = _claim(layer, mqa_q_arg)
    if stash is None:
        return layer.dcp_manager.query_gather(mqa_q_arg)
    state = layer.__dict__["_glm_dcp_decode_comm_state"]
    state.stash = None
    device = layer.W_UK_T.device
    main = torch.cuda.current_stream(device)
    if not stash.issued:
        _issue_gather(layer, state, stash, early=False)
    if SETTINGS.comm_stream:
        main.wait_stream(side_stream("comm", device))
        stash.gathered.record_stream(main)
    gathered = stash.gathered[:stash.rows]
    if SETTINGS.audit:
        _audit_query(layer, stash, mqa_q_arg, gathered)
    return gathered


# ----------------------------------------------------------------------------- selection reuse


class SelectionProvider:
    """``B12xPhysicalSelectionProvider`` for one DCP attention layer: the DCP index selection, reused.

    ``forward_mqa`` asks the provider before it filters and converts the
    global top-k indices to this rank's physical slots
    (``b12x_mla_sparse.py``, ``forward_mqa``). In a decode batch the provider
    makes that same call (``triton_filter_and_convert_dcp_index`` with the
    arguments ``forward_mqa`` would pass) on calls that run the DSA indexer,
    whose top-k is new, and on any call whose cached entry another layer did
    not make, and returns the cached result to the calls of other layers that
    follow until the next indexer call: their inputs (the shared top-k buffer,
    the batch's request ids and block table, the page size and DCP rank) are
    the same objects with the same contents. A layer never reuses its own
    entry, so draft steps of the MTP layer that reuse the top-k still convert
    it per step. ``None`` (the image computes) otherwise.
    """

    _cache: dict = {}

    def __init__(self, layer):
        self._layer = weakref.ref(layer)

    def get_b12x_physical_selection(self, *, num_tokens: int, num_prefills: int, num_decode_tokens: int):
        layer = self._layer()
        if layer is None or not SETTINGS.selection_reuse or num_prefills != 0 or num_decode_tokens != num_tokens:
            return None
        state, rows = _active(layer)
        if state is None or rows != num_tokens:
            return None
        impl = layer.impl
        metadata, _, kv_cache, _ = _metadata(layer)
        key = (id(metadata), int(num_tokens), int(impl.dcp_rank), id(impl.topk_indices_buffer))
        cache = SelectionProvider._cache
        if (not _runs_indexer(layer) and cache.get("key") == key and cache.get("metadata") is metadata
                and cache.get("layer") != id(layer)):
            _count("selection_reused")
            value = cache["value"]
            if SETTINGS.audit:
                audit_exact(layer, "selection", value, self._filter(impl, metadata, kv_cache, num_tokens))
            return value
        value = self._filter(impl, metadata, kv_cache, num_tokens)
        cache.clear()
        cache.update(key=key, metadata=metadata, value=value, layer=id(layer))
        _count("selection_computed")
        return value

    @staticmethod
    def _filter(impl, metadata, kv_cache, num_tokens: int):
        """``forward_mqa``'s DCP filter call with its arguments."""
        from vllm.v1.attention.backends.mla import b12x_mla_sparse as sparse

        topk_indices = impl.topk_indices_buffer[:num_tokens]
        return sparse.triton_filter_and_convert_dcp_index(
            metadata.req_id_per_token[:num_tokens],
            metadata.block_table,
            topk_indices,
            dcp_size=impl.dcp_world_size,
            dcp_rank=impl.dcp_rank,
            cp_kv_cache_interleave_size=metadata.cp_kv_cache_interleave_size,
            BLOCK_SIZE=metadata.block_size,
            BLOCK_STRIDE_ROWS=sparse._selected_index_block_stride_rows(kv_cache, block_size=metadata.block_size),
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )


def install_provider(layer) -> None:
    """Make the layer's B12X implementation consult a ``SelectionProvider`` (first decode call)."""
    if not SETTINGS.selection_reuse:
        return
    state, rows = _active(layer)
    if state is None or state.provider_installed:
        return
    impl = layer.impl
    current = getattr(impl, "_physical_selection_provider", None)
    if current is not None and not isinstance(current, SelectionProvider):
        _log_once("provider-taken", logging.WARNING, "%s: %s already has a physical selection provider "
                  "(%s); selection reuse is left out", PLUGIN_NAME, layer.layer_name, type(current).__name__)
        state.provider_installed = True
        return
    if _capturing():
        return  # installed at the next eager forward (warm-up precedes every capture)
    impl._physical_selection_provider = SelectionProvider(layer)
    state.provider_installed = True


# ----------------------------------------------------------------------------- fused all-to-all combine


def scatter_pack_fits(dcp, rows: int, heads: int) -> bool:
    """Whether the packed all-to-all of ``rows`` rows of ``heads`` heads per rank is one scatter op.

    The image's combine moves the same bytes as one session all-to-all exactly
    when its message fits the DCP group's scatter op limit
    (``SirclDcpCollectives._scatter_op_limit``, the relay-safe op size the
    adapter's executor splits at) and the session sends it as one piece
    (``_scatter_ops.op_bytes`` and ``piece_bytes``); the fused combine runs
    only then. Settings and shapes only: the same answer on every rank.
    """
    from sparkring_sircl.oneshot import _scatter_ops

    rt = dcp._runtime
    if rt is None or not rt.scatter_available or not L.heads_supported(heads) or rows < 1:
        return False
    world = int(rt.world_size)
    chunk = L.wire_chunk_bytes(rows, heads)
    nbytes = world * chunk
    return (chunk % L.PACK_BYTES == 0 and nbytes <= int(dcp._scatter_op_limit())
            and nbytes <= int(_scatter_ops.op_bytes(rt)) and int(_scatter_ops.piece_bytes(rt, chunk)) >= chunk)


def packed_all_to_all(rt, attn_out: torch.Tensor, lse: torch.Tensor, recv: torch.Tensor, rows: int,
                      heads: int) -> bool:
    """One packed all-to-all on session ``rt`` (one scatter op); False to decline (inside a capture, unprepared).

    ``attn_out`` ``[rows, world * heads, 512]`` BF16 and ``lse``
    ``[rows, world * heads]`` FP32 must have 16-byte aligned rows, heads and
    pointers (:func:`_aligned_inputs`); ``recv`` is ``[world, chunk bytes]``
    uint8. The launch follows the session's scatter launch
    (``_scatter_ops._launch``): its lock and health check, the tuning table's
    choice for an all-to-all of this size, the large-message grid, the
    per-grid counters and its stream rules.
    """
    from sparkring_sircl import protocol as proto

    from . import _scatter_pack_cute

    world = int(rt.world_size)
    geometry = L.WireGeometry(rows, heads)
    chunk = geometry.chunk_bytes
    nbytes = world * chunk
    key = (world, rt.rank, rt._threads, rt._layout.slots, rt._layout.flag_stride, rt.lane_count, heads,
           rt.device.index)
    if _capturing() and not _scatter_pack_cute.is_prepared(*key):
        _log_once("pack-unprepared", logging.WARNING, "%s: the packed all-to-all kernel was not compiled "
                  "before CUDA graph capture; this graph uses the image's combine", PLUGIN_NAME)
        return False
    launcher = _scatter_pack_cute.get_launcher(*key)
    # The session's all_to_all enters the tuned op (which holds the tuning lock while it is active) before
    # the session lock; the same order here.
    with rt._tuned_op("all_to_all", nbytes), rt._lock:
        rt.check_health()
        with torch.cuda.device(rt.device):
            capturing = torch.cuda.is_current_stream_capturing()
            size_packs = nbytes // L.PACK_BYTES
            blocks = getattr(rt, "_large_blocks", rt._blocks)
            grid = proto.grid_blocks(size_packs, rt._threads, blocks, rt._packs_per_thread)
            stage_counter, tail_counter = rt._counter_addresses(grid)
            rt._order_stream(capturing)
            launcher(
                attn_out.data_ptr(), lse.data_ptr(), recv.data_ptr(), size_packs, nbytes, geometry.chunk_packs,
                rows, attn_out.stride(0) * 2, attn_out.stride(1) * 2, lse.stride(0) * 4, chunk, rt._recv_base,
                rt._flag_base, rt._send_base, rt._ctrl_base, rt._slot_bytes, rt._epoch_address, stage_counter,
                tail_counter, rt._poison_address, rt.spin_limit, grid,
            )
            rt._mark_stream(capturing)
        if not capturing:
            rt.check_health()
    return True


def _image_combine_parts(layer, image_combine) -> tuple[bool, str]:
    """``(is_lse_base_on_e, "")`` when ``image_combine`` is SIRCL's all-to-all combine on this layer's group,
    else ``(False, reason)``: the fused combine replaces only that one."""
    if not isinstance(image_combine, functools.partial):
        return False, f"the DCP combine is {image_combine!r}, not vLLM's partial of dcp_a2a_lse_reduce"
    if getattr(image_combine.func, SIRCL_SHIM_MARKER, None) is None:
        return False, "the DCP combine is not SIRCL's dcp_all_to_all shim"
    if image_combine.keywords.get("cp_group") is not layer.dcp_manager.group:
        return False, "the DCP combine is bound to another group"
    return bool(image_combine.keywords.get("is_lse_base_on_e", True)), ""


def _fused_eligible(layer, attn_out, lse, seq_lens, query_start_loc) -> bool:
    """Shapes, dtypes and configuration only (never pointers): the same answer on every rank."""
    world = int(layer.impl.dcp_world_size)
    if seq_lens is not None or query_start_loc is not None or not layer.dcp_manager.use_a2a:
        return False
    if attn_out.dim() != 3 or lse.dim() != 2 or attn_out.dtype != torch.bfloat16 or lse.dtype != torch.float32:
        return False
    rows, total_heads, dim = attn_out.shape
    if dim != L.V_DIM or total_heads % world or tuple(lse.shape) != (rows, total_heads) or attn_out.device != lse.device:
        return False
    return L.heads_supported(total_heads // world) and scatter_pack_fits(_dcp(layer), int(rows), total_heads // world)


def _aligned_inputs(attn_out: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``attn_out`` and ``lse`` with 16-byte aligned pointers, row and head strides (a copy where not).

    Rank-local: the copy changes no op of the session.
    """
    if (attn_out.stride(2) != 1 or attn_out.data_ptr() % L.PACK_BYTES or (attn_out.stride(0) * 2) % L.PACK_BYTES
            or (attn_out.stride(1) * 2) % L.PACK_BYTES):
        attn_out = attn_out.clone(memory_format=torch.contiguous_format)
    if lse.stride(1) != 1 or lse.data_ptr() % L.PACK_BYTES or (lse.stride(0) * 4) % L.PACK_BYTES:
        lse = lse.clone(memory_format=torch.contiguous_format)
    return attn_out, lse


def _fused_combine(layer, image_combine, attn_out, lse, *, seq_lens=None, query_start_loc=None):
    base_e, reason = _image_combine_parts(layer, image_combine)
    if reason or not _fused_eligible(layer, attn_out, lse, seq_lens, query_start_loc):
        if reason:
            _log_once(f"combine:{reason}", logging.WARNING, "%s: %s; the image's combine runs", PLUGIN_NAME, reason)
        _count("combine_image")
        return image_combine(attn_out, lse, seq_lens=seq_lens, query_start_loc=query_start_loc)
    world = int(layer.impl.dcp_world_size)
    rows = int(attn_out.shape[0])
    heads = int(attn_out.shape[1]) // world
    rank = int(layer.dcp_manager.group.rank_in_group)
    device = attn_out.device
    own_out, own_lse = _aligned_inputs(attn_out, lse)
    recv = torch.empty((world, L.wire_chunk_bytes(rows, heads)), dtype=torch.uint8, device=device)
    main = torch.cuda.current_stream(device)
    if SETTINGS.comm_stream:
        comm = side_stream("comm", device)
        for tensor in (own_out, own_lse, recv):
            tensor.record_stream(comm)
        comm.wait_stream(main)
        context = torch.cuda.stream(comm)
    else:
        comm = None
        context = torch.cuda.stream(main)
    with context:
        launched = packed_all_to_all(_dcp(layer)._runtime, own_out, own_lse, recv, rows, heads)
    if comm is not None:
        main.wait_stream(comm)
    if not launched:
        _count("combine_image")
        return image_combine(attn_out, lse, seq_lens=seq_lens, query_start_loc=query_start_loc)
    _count("combine_fused")
    out = _kernels().wire_combine(recv, own_out, own_lse, world, rank, heads, base_e)
    if SETTINGS.audit and not _capturing():  # the reference all-to-all: eager calls only
        audit_exact(layer, "combine", out,
                    image_combine(attn_out, lse, seq_lens=seq_lens, query_start_loc=query_start_loc))
    return out


def combine_fn(layer):
    """The combine this layer call uses: the fused all-to-all, or the image's (``dcp_manager.combine``)."""
    image_combine = layer.dcp_manager.combine
    if not SETTINGS.a2a_fused:
        return image_combine
    state, rows = _active(layer)
    if state is None:
        return image_combine
    return functools.partial(_fused_combine, layer, image_combine)


def status() -> dict:
    """Settings, counters and stream names; in audit mode also the audit summary (waits for the GPU)."""
    with _LOCK:
        out = {"settings": SETTINGS.describe(), "counts": dict(COUNTS),
               "streams": sorted(f"{k}:{i}" for k, i in _STREAMS)}
        audits = dict(_AUDITS)
    if audits and not _capturing():
        out["audit"] = {f"cuda:{i}": _audit_summary(a.exact.cpu(), a.slots) for i, a in audits.items()}
    return out


__all__ = [
    "EXACT_CHECKS",
    "SETTINGS",
    "SelectionProvider",
    "Settings",
    "audit_exact",
    "audit_tick",
    "combine_fn",
    "configure",
    "finish",
    "fused_q_fn",
    "gather",
    "gather_fits_one_op",
    "has_query",
    "install_comm_stream",
    "install_provider",
    "packed_all_to_all",
    "query",
    "scatter_pack_fits",
    "side_stream",
    "status",
    "wk_fork",
    "wk_join",
]
