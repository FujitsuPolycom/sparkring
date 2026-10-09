"""Version-pinned shims for collectives no official vLLM hook reaches.

A shim replaces module or class attributes of vLLM. It installs only when every vLLM
file it touches has the SHA-256 recorded for one pinned build (:mod:`.pins`);
otherwise :func:`install` raises :class:`ShimRefused` and nothing is replaced.

Each shim keeps vLLM's original function and calls it whenever SIRCL does not
own the group, so an installed shim changes nothing for other groups:

``dcp_all_to_all``
    Replaces ``vllm.v1.attention.ops.dcp.dcp_a2a_lse_reduce``, the DCP output
    combine that vLLM's GLM configuration selects by default
    (``dcp_comm_backend="a2a"``). vLLM's version exchanges with
    ``torch.distributed.all_to_all_single`` on the DCP group's NCCL device
    group, bypassing the device communicator; on a DCP group NCCL may not run
    (four consecutive Sparks of a ring) that would hang. The replacement keeps
    vLLM's pack, buffer and unpack helpers and arithmetic and exchanges through
    the SIRCL communicator's ``all_to_all_single``, whose plan is rank-invariant.
    SIRCL's communicator installs it when it builds a DCP group of two or more
    ranks, before the attention layers bind the combine function.
``fused_allreduce_rms_norm``
    Wraps vLLM's eager helper
    ``vllm.models.common.ops.fused_allreduce_rms_norm.fused_allreduce_rms_norm``
    where it is defined and in every loaded vLLM module that imported it
    (the DeepSeek-V3.2 and GLM-5.3 decoder layers, final norm and MTP draft).
    When the tensor-parallel communicator holds bound fused kernels
    (``sircl_fused_add_rms_norm``) and the call qualifies, the all-reduce,
    residual add and RMSNorm run as one SIRCL collective; otherwise vLLM's
    helper runs unchanged (:mod:`.norm_fusion`). SIRCL's communicator installs
    it when ``SIRCL_FUSED_NORM=1`` binds the kernels; installed alone it
    changes nothing.
``roce_slot``
    Makes ``vllm.distributed.device_communicators.b12x_roce_all_reduce.B12xRoceAllReduce``
    SIRCL's slot class, so that ``VLLM_ENABLE_ROCE_ALLREDUCE=1`` builds SIRCL
    instead of another RDMA transport in the tensor-parallel slot.
    The general plugin installs it whenever SIRCL is enabled with that switch.
``mhc_prefill_shard``
    Wraps ``vllm.models.glm5next.nvidia.mhc_prefill_sharding.maybe_create``
    (and the GLM model's binding of it) so that GLM-5.3-Flash's mHC prefill
    row ownership reduce-scatters and all-gathers through SIRCL's
    communicator on a tensor-parallel group without PyNccl (:mod:`.mhc`).
    SIRCL's communicator installs it when it builds such a group with a session
    and ``VLLM_GLM53_MHC_PREFILL_SHARD`` is set, before the model is built.
``qwen_hc_prefill_shard``
    Wraps ``vllm.models.qwen4_exp.nvidia.hc_prefill.create`` so that
    Qwen3.8's hyper-connection prefill row ownership
    (``VLLM_QWEN3_8_HC_PREFILL_MODE=shard``) reduce-scatters and all-gathers
    through SIRCL's communicator on a tensor-parallel group without PyNccl
    (:mod:`.qwen_hc`). Pinned to the image's build, the only pinned build with
    the module. SIRCL's communicator installs it when it builds such a group
    with a session and the mode is ``shard``, before the model is built.
``dcp_b12x_transport``
    Wraps ``vllm.distributed.device_communicators.b12x_dcp.get_b12x_dcp_transport``
    so that it returns None for a group SIRCL's communicator owns. vLLM's MLA
    DCP manager asks it for the B12X PCIe transport when the backend is B12X
    and ``VLLM_USE_B12X_DCP_A2A=1`` (``v1/attention/ops/dcp.py:1501-1544``);
    that transport exchanges CUDA IPC handles, which work within one machine
    only. With None the manager keeps its plain paths: the a2a combine
    (``dcp_a2a_lse_reduce``, carried by the ``dcp_all_to_all`` shim) and the
    query all-gather through the group coordinator. SIRCL's communicator
    installs it with ``dcp_all_to_all``.
``worker_regimes``
    Wraps the methods of vLLM's GPU worker (``vllm.v1.worker.gpu_worker.Worker``)
    after which one rank may reach its next collective much later than its
    peers: ``compile_or_warm_up_model`` (compilation, warm-up, graph capture),
    ``determine_available_memory`` (the memory profile run), ``sleep``,
    ``wake_up``, ``reload_weights``, ``update_weights`` and ``profile``. Each
    runs in :func:`.adapter.startup_all`, so every SIRCL session of the process
    waits for peers with the startup regime's limit until the next completed
    step, and the return of ``compile_or_warm_up_model`` arms the serving
    regime. SIRCL's communicator installs it when it builds a group with a
    session; on a vLLM that matches no pinned build it logs a warning and the
    sessions keep the startup regime.
``step_health``
    Wraps the step methods of vLLM's GPU worker (``execute_model``,
    ``sample_tokens``, ``execute_dummy_batch``) so that each first runs
    :func:`.adapter.check_all_failures`: a flag wait that timed out or a
    progress thread that stopped, recorded by any SIRCL session or
    point-to-point channel set of the process, raises before the step's SIRCL
    ops launch. The check reads host memory only. SIRCL's communicator
    installs it with ``worker_regimes``; on a vLLM that matches no pinned
    build it logs a warning, and a failure raises at the post-step check or at
    the next eager SIRCL call.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import logging
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from . import pins

logger = logging.getLogger("sircl.vllm.shims")

_INSTALLED: dict[str, str] = {}
_INSTALL_LOCK = threading.Lock()     # one installer at a time: a shim reads, wraps and rebinds module attributes
MARKER = "_sircl_original"


class ShimRefused(RuntimeError):
    """A shim's target files do not match any pinned vLLM build, or its target was replaced."""


@dataclasses.dataclass(frozen=True)
class Shim:
    name: str
    files: tuple[str, ...]
    apply: Callable[[], None]


def _install_dcp_all_to_all() -> None:
    module = importlib.import_module("vllm.v1.attention.ops.dcp")
    original = module.dcp_a2a_lse_reduce
    if getattr(original, MARKER, None) is not None:
        return

    def dcp_a2a_lse_reduce(cp_attn_out, cp_attn_lse, cp_group, ctx=None, return_lse=False,
                           is_lse_base_on_e=True, seq_lens=None, query_start_loc=None):
        communicator = getattr(cp_group, "device_communicator", None)
        if getattr(communicator, "sircl", None) is None or cp_group.world_size == 1:
            return original(cp_attn_out, cp_attn_lse, cp_group, ctx, return_lse,
                            is_lse_base_on_e, seq_lens, query_start_loc)
        world_size = cp_group.world_size
        tokens, heads, head_dim = cp_attn_out.shape
        if heads % world_size != 0:
            raise ValueError(f"H={heads} must be divisible by DCP world size {world_size}.")
        heads_per_rank = heads // world_size
        lse_pack_dim = module._dcp_a2a_lse_pack_dim(cp_attn_out.dtype)
        send_buffer, recv_buffer = module._dcp_a2a_send_recv_buffers(
            (world_size, tokens, heads_per_rank, head_dim + lse_pack_dim),
            device=cp_attn_out.device, dtype=cp_attn_out.dtype)
        module._dcp_a2a_pack_send(cp_attn_out, cp_attn_lse, send_buffer, world_size, heads_per_rank,
                                  head_dim, lse_pack_dim, seq_lens=seq_lens,
                                  query_start_loc=query_start_loc)
        communicator.all_to_all_single(recv_buffer.view(-1), send_buffer.view(-1))
        return module._dcp_a2a_unpack_combine(recv_buffer, head_dim, lse_pack_dim, return_lse,
                                              is_lse_base_on_e)

    setattr(dcp_a2a_lse_reduce, MARKER, original)
    module.dcp_a2a_lse_reduce = dcp_a2a_lse_reduce


def _install_fused_allreduce_rms_norm() -> None:
    from . import norm_fusion

    norm_fusion.install_shim(MARKER)


def _install_roce_slot() -> None:
    from .tp_slot import SirclRingAllReduce

    module = importlib.import_module("vllm.distributed.device_communicators.b12x_roce_all_reduce")
    current = getattr(module, "B12xRoceAllReduce", None)
    if current is SirclRingAllReduce:
        return
    if current is None or current.__module__ != module.__name__:
        raise ShimRefused(f"vLLM's RoCE slot class is {current!r}, not vLLM's own; another plugin "
                          "replaced it, and SIRCL will not install over it")
    module.B12xRoceAllReduce = SirclRingAllReduce


def _install_mhc_prefill_shard() -> None:
    from . import mhc

    mhc.install_shim(MARKER)


def _install_qwen_hc_prefill_shard() -> None:
    from . import qwen_hc

    qwen_hc.install_shim(MARKER)


WORKER_STARTUP_METHODS = ("compile_or_warm_up_model", "determine_available_memory", "sleep", "wake_up",
                          "reload_weights", "update_weights", "profile")


def _install_dcp_b12x_transport() -> None:
    module = importlib.import_module("vllm.distributed.device_communicators.b12x_dcp")
    original = module.get_b12x_dcp_transport
    if getattr(original, MARKER, None) is not None:
        return

    @functools.wraps(original)
    def get_b12x_dcp_transport(group, *args, **kwargs):
        communicator = getattr(group, "device_communicator", None)
        if getattr(communicator, "sircl", None) is not None:
            return None
        return original(group, *args, **kwargs)

    setattr(get_b12x_dcp_transport, MARKER, original)
    module.get_b12x_dcp_transport = get_b12x_dcp_transport


WARM_UP_METHOD = "compile_or_warm_up_model"
# The worker's methods that run one step's SIRCL ops (forward pass, sampling with any draft model, a data-parallel
# rank's dummy batch).
WORKER_STEP_METHODS = ("execute_model", "sample_tokens", "execute_dummy_batch")


def _in_startup(original: Callable, label: str, then_serve: bool) -> Callable:
    from . import adapter

    @functools.wraps(original)
    def method(self, *args, **kwargs):
        with adapter.startup_all(label, then_serve=then_serve):
            return original(self, *args, **kwargs)

    setattr(method, MARKER, original)
    return method


def _install_worker_regimes() -> None:
    worker = importlib.import_module("vllm.v1.worker.gpu_worker").Worker
    missing = [name for name in WORKER_STARTUP_METHODS if not callable(worker.__dict__.get(name))]
    if missing:
        raise ShimRefused(f"vLLM's Worker defines no {', '.join(missing)}")
    for name in WORKER_STARTUP_METHODS:
        original = worker.__dict__[name]
        if getattr(original, MARKER, None) is None:
            setattr(worker, name, _in_startup(original, f"Worker.{name}", name == WARM_UP_METHOD))


def _checked_step(original: Callable) -> Callable:
    from . import adapter

    @functools.wraps(original)
    def method(self, *args, **kwargs):
        adapter.check_all_failures()
        return original(self, *args, **kwargs)

    setattr(method, MARKER, original)
    return method


def _install_step_health() -> None:
    worker = importlib.import_module("vllm.v1.worker.gpu_worker").Worker
    missing = [name for name in WORKER_STEP_METHODS if not callable(worker.__dict__.get(name))]
    if missing:
        raise ShimRefused(f"vLLM's Worker defines no {', '.join(missing)}")
    for name in WORKER_STEP_METHODS:
        original = worker.__dict__[name]
        if getattr(original, MARKER, None) is None:
            setattr(worker, name, _checked_step(original))


SHIMS: dict[str, Shim] = {
    "dcp_all_to_all": Shim("dcp_all_to_all", pins.DCP_FILES, _install_dcp_all_to_all),
    "fused_allreduce_rms_norm": Shim("fused_allreduce_rms_norm", pins.NORM_FILES,
                                     _install_fused_allreduce_rms_norm),
    "roce_slot": Shim("roce_slot", pins.SLOT_FILES, _install_roce_slot),
    "mhc_prefill_shard": Shim("mhc_prefill_shard", pins.MHC_FILES, _install_mhc_prefill_shard),
    "qwen_hc_prefill_shard": Shim("qwen_hc_prefill_shard", pins.QWEN_HC_FILES, _install_qwen_hc_prefill_shard),
    "worker_regimes": Shim("worker_regimes", pins.WORKER_FILES, _install_worker_regimes),
    "step_health": Shim("step_health", pins.WORKER_FILES, _install_step_health),
    "dcp_b12x_transport": Shim("dcp_b12x_transport", pins.DCP_TRANSPORT_FILES, _install_dcp_b12x_transport),
}


def check(names: Sequence[str], root: Path | None) -> dict[str, str]:
    """The pinned build each requested shim matches; raises :class:`ShimRefused` otherwise."""
    result = {}
    for name in names:
        shim = SHIMS.get(name)
        if shim is None:
            raise ShimRefused(f"unknown SIRCL shim {name!r}; known: {', '.join(sorted(SHIMS))}")
        if root is None:
            raise ShimRefused(f"shim {name} needs an installed vLLM")
        build = pins.matching_build(root, shim.files)
        if build is None:
            actual = pins.hashes(root, shim.files)
            raise ShimRefused(
                f"shim {name} refuses to load: vLLM at {root} matches no pinned build for "
                + ", ".join(f"{file} ({digest})" for file, digest in actual.items())
            )
        result[name] = build.name
    return result


def install(names: Sequence[str], *, root: Path | None = None) -> dict[str, str]:
    """Check every requested shim first, then install them all; returns shim -> pinned build."""
    with _INSTALL_LOCK:
        names = [name for name in names if name not in _INSTALLED]
        if not names:
            return {}
        root = pins.installed_root() if root is None else root
        builds = check(names, root)
        for name in names:
            SHIMS[name].apply()
            _INSTALLED[name] = builds[name]
            logger.info("SIRCL shim %s installed for vLLM build %s", name, builds[name])
        return builds


def installed() -> dict[str, str]:
    return dict(_INSTALLED)
