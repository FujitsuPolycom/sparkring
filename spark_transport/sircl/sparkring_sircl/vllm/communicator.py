"""SIRCL's vLLM device communicator: vLLM's CUDA communicator with SIRCL in front.

vLLM's group coordinators build their device communicator from the platform
(``current_platform.get_device_communicator_cls()``,
``distributed/parallel_state.py:612-621`` in the image's vLLM); with SIRCL's
platform (:mod:`.platform`) that is :class:`SirclCudaCommunicator`, for every
group: tensor parallel, decode context parallel, expert parallel and the
rest. For each group it

1. places the group on the fabric before vLLM's constructor runs
   (:class:`.adapter.GroupPlacement`) and, when NCCL may not connect the
   group's ranks, builds the parent class without PyNccl, whose constructor
   would otherwise all-reduce over the whole group
   (``device_communicators/pynccl.py:180-191``);
2. builds the group's SIRCL session where ``SIRCL_GROUPS`` asks for one and
   registers the group with the NCCL guard (:mod:`.guard`);
3. overrides every collective method of ``CudaCommunicator`` and its base
   (``all_reduce``, ``all_reduce_in_place``, ``all_gather``,
   ``all_gatherv``, ``reduce_scatter``, ``reduce_scatterv``, ``gather``,
   ``broadcast``, ``send``, ``recv``, ``batch_isend_irecv``) and adds
   ``all_to_all_single`` for the DCP combine, each asking the group's
   :class:`.adapter.GroupAdapter` for a rank-invariant plan;
4. on the tensor-parallel group, chains the process-wide capture context and
   health check of every SIRCL session into ``b12x_ar_comm``, the slot that
   vLLM's ``graph_capture`` (``parallel_state.py:756-790``) and the worker's
   post-step check (``v1/worker/gpu_worker.py:1402-1440``) read; after the
   worker's warm-up the post-step check also puts every session in the serving
   flag-wait regime;
5. installs the ``worker_regimes`` shim with the first group that gets a
   session, so the worker's warm-up, profiling, sleep, wake-up and weight loads
   run in the startup regime and the warm-up's return arms the serving regime
   (:func:`.adapter.startup_all`), and the ``step_health`` shim with the first
   group that gets a session or point-to-point channels, so each of the
   worker's step methods first checks every session and channel set of the
   process for a recorded failure (:func:`.adapter.check_all_failures`);
6. with ``SIRCL_FUSED_NORM=1``, binds the fused all-reduce + residual add +
   RMSNorm kernels to the tensor-parallel group's session and installs the
   ``fused_allreduce_rms_norm`` shim (:mod:`.norm_fusion`), exposing the bound
   kernels as ``sircl_fused_add_rms_norm``;
7. builds the point-to-point channels of a pipeline-parallel group and of a
   group with its own session (``SIRCL_P2P_GROUPS``, :mod:`.p2p`), so
   ``send``, ``recv`` and ``batch_isend_irecv``, and the ``torch.distributed``
   point-to-point calls of vLLM's pipeline code on the group's device group,
   run on SIRCL wherever the group has a channel to the peer.

With ``SIRCL_MODE`` unset or ``disabled`` the class is exactly vLLM's.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from typing import Any

import torch

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

from . import adapter as adapter_module
from . import groupops, guard, mhc, qwen_hc, settings, shims, tp4
from .adapter import AdapterConfig, GroupAdapter, GroupPlacement, NcclPaths, SirclSetupError
from .fabric import NcclPolicy
from .tp_slot import attach_sircl_logging

logger = logging.getLogger("sircl.vllm.communicator")

_NAIVE_ALL2ALL = ("naive", "allgather_reducescatter")


@functools.lru_cache(maxsize=None)
def _config(world: int) -> AdapterConfig:
    config = AdapterConfig.from_env(world)
    attach_sircl_logging()
    guard.configure(adapter_module.resolver_for(config))
    guard.install_tripwire()
    return config


class HubSlot:
    """``b12x_ar_comm`` of a tensor-parallel group that has no SIRCL session of its own.

    It declines every collective (vLLM's own chain continues), and gives vLLM's
    capture and post-step check every other SIRCL session of the process.
    """

    backend_name = "SIRCL"
    disabled = False

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return False

    def custom_all_reduce(self, inp: torch.Tensor) -> None:
        return None

    def should_all_gather(self, inp: torch.Tensor, dim: int) -> bool:
        return False

    def supports_fused_add_rms_norm(self) -> bool:
        return False

    def capture(self, stream: Any = None):
        return adapter_module.capture_all(stream)

    def check_health(self) -> None:
        adapter_module.check_all_health()

    def close(self) -> None:
        return None


def _chain(slot: Any) -> Any:
    """Make ``slot``'s capture and health check cover every SIRCL session of the process."""
    slot.capture = lambda stream=None: adapter_module.capture_all(stream)
    slot.check_health = adapter_module.check_all_health
    return slot


def _install_worker_regimes(unique_name: str) -> None:
    """Run the worker's start-up work in the startup flag-wait regime (pinned shim ``worker_regimes``).

    On a vLLM that matches no pinned build the serving regime is never armed:
    the sessions keep the startup limit, and a warning says so.
    """
    try:
        shims.install(["worker_regimes"])
    except shims.ShimRefused as exc:
        logger.warning("SIRCL sessions of %s keep the startup regime's flag-wait limit while serving, "
                       "because vLLM's worker cannot be wrapped: %s", unique_name, exc)


def _install_step_health(unique_name: str) -> None:
    """Check every session and channel set for a recorded failure when each worker step starts (pinned shim
    ``step_health``).

    On a vLLM that matches no pinned build a failure raises at the worker's
    post-step check or at the next eager SIRCL call instead, and a warning
    says so.
    """
    try:
        shims.install(["step_health"])
    except shims.ShimRefused as exc:
        logger.warning("SIRCL failures of %s raise at the worker's post-step check or at the next eager SIRCL "
                       "call, not when a step starts, because vLLM's worker cannot be wrapped: %s",
                       unique_name, exc)


_DIRECT_DCP = ("VLLM_USE_DIRECT_DCP_A2A", "VLLM_USE_DIRECT_DCP_Q_GATHER", "VLLM_USE_DIRECT_DCP_KV_GATHER")


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _dcp_conflicts(env, config) -> list[str]:
    """Settings under which a decode-context-parallel group would bypass SIRCL in ways nothing can carry."""
    found = [f"{name}={env.get(name)} maps peer GPU memory, which works within one machine only "
             "(v1/attention/ops/dcp.py)" for name in _DIRECT_DCP if _truthy(env.get(name))]
    if _truthy(env.get("VLLM_MLA_PREFILL_DCP_OVERLAP")):
        found.append("VLLM_MLA_PREFILL_DCP_OVERLAP=1 gathers the prefill context on PyNccl directly "
                     "(v1/attention/ops/dcp_prefetch.py:36-46)")
    parallel = getattr(config, "parallel_config", None)
    if int(getattr(parallel, "prefill_context_parallel_size", 1) or 1) > 1:
        found.append("prefill context parallelism builds groups without a SIRCL session")
    return found


def _current_vllm_config() -> Any:
    """vLLM's current config, which the worker sets while it builds groups; None outside one (tools, tests)."""
    try:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config()
    except Exception:  # noqa: BLE001 - no current config
        return None


def _micro_batching(parallel: Any) -> bool:
    """vLLM's micro-batching switch: ``ParallelConfig.use_ubatching`` (``enable_dbo`` or ``ubatch_size`` above 1)."""
    enabled = getattr(parallel, "use_ubatching", None)
    if isinstance(enabled, bool):
        return enabled
    return bool(getattr(parallel, "enable_dbo", False)) or int(getattr(parallel, "ubatch_size", 0) or 0) > 1


def _session_conflicts(placement: GroupPlacement, config: Any = None) -> list[str]:
    """Settings no SIRCL session can serve, refused on every group that gets one, whatever NCCL may do there.

    vLLM's micro-batching (dual-batch overlap, ``--enable-dbo``, or
    ``--ubatch-size`` above 1; ``config/parallel.py:595-597``) runs each
    micro-batch's forward in its own thread on its own compute and
    communication streams, handing control between threads with events
    (``v1/worker/ubatching.py``). A session's operations must be issued in
    the same order on every rank of its group, from one thread in program
    order; with two issuing threads that order rests on every hand-off, so
    the adapter does not serve it.
    """
    if not placement.session:
        return []
    config = _current_vllm_config() if config is None else config
    if _micro_batching(getattr(config, "parallel_config", None)):
        return ["vLLM's micro-batching (--enable-dbo, or --ubatch-size above 1) issues the group's collectives "
                "from one thread per micro-batch on separate CUDA streams (v1/worker/ubatching.py), and a "
                "SIRCL session needs one issuing order on every rank; serve without it"]
    return []


def _config_conflicts(placement: GroupPlacement, environ=None) -> list[str]:
    """Features that issue NCCL collectives outside the device communicator, on a group NCCL may not run.

    Each would either hang in NCCL's connection setup or find no PyNccl
    communicator, so SIRCL refuses them when the group is set up. GLM-5.3-Flash's
    mHC prefill sharding and Qwen3.8's hyper-connection prefill row ownership
    are the exceptions on a group that gets a SIRCL session: the
    ``mhc_prefill_shard`` and ``qwen_hc_prefill_shard`` shims carry them
    (:mod:`.mhc`, :mod:`.qwen_hc`). Direct
    ``torch.distributed`` gathers, exchanges, broadcasts and reductions on a
    group with a session are carried by the tripwire (:mod:`.adapter`), not
    refused here: speculative decoding's adaptive verification broadcasts its
    confidence snapshot that way (``GroupCoordinator.broadcast``). The serve
    launcher's plan applies the recipe-visible conditions before any container
    starts (``serve.plan.relay_conflicts``).
    """
    if placement.policy is not NcclPolicy.NONE or placement.kind not in ("tp", "dcp"):
        return []
    env = os.environ if environ is None else environ
    if placement.kind == "dcp":
        return _dcp_conflicts(env, _current_vllm_config())
    found = []
    if mhc.requested(env) and not placement.session:
        found.append("VLLM_GLM53_MHC_PREFILL_SHARD calls PyNccl reduce-scatter and all-gather "
                     "directly (models/glm5next/nvidia/mhc_prefill_sharding.py), and the group has no "
                     "SIRCL session to carry them (SIRCL_GROUPS)")
    if qwen_hc.requested(env) and not placement.session:
        found.append("VLLM_QWEN3_8_HC_PREFILL_MODE=shard calls PyNccl all-gather and reduce-scatter "
                     "directly (models/qwen4_exp/nvidia/hc_prefill.py), and the group has no SIRCL session "
                     "to carry them (SIRCL_GROUPS)")
    config = _current_vllm_config()
    if config is None:
        return found
    pass_config = getattr(getattr(config, "compilation_config", None), "pass_config", None)
    for name, why in (("fuse_gemm_comms", "fuses GEMMs with symmetric-memory reduce-scatter and "
                                          "all-gather outside the device communicator"),
                      ("fuse_allreduce_rms", "replaces all-reduce + RMSNorm with a fused kernel "
                                             "that has its own transport")):
        if getattr(pass_config, name, False):
            found.append(f"compilation pass_config.{name} {why}")
    if getattr(getattr(config, "parallel_config", None), "enable_batch_sharded_sampling", False):
        found.append("enable_batch_sharded_sampling calls torch all_to_all_single on the NCCL group "
                     "(v1/worker/gpu/sample/batch_shard.py)")
    if (getattr(getattr(config, "speculative_config", None), "enable_adaptive_verification", False)
            and not placement.session):
        found.append("enable_adaptive_verification broadcasts through GroupCoordinator.broadcast on the NCCL "
                     "group directly, and the group has no SIRCL session whose carrier runs it")
    return found


class SirclCudaCommunicator(CudaCommunicator):
    """``CudaCommunicator`` whose collectives go through SIRCL's rank-invariant plans."""

    sircl: GroupAdapter | None
    sircl_fused_add_rms_norm: Any = None        # norm_fusion.SirclFusedNorm on a TP group that bound it

    def __init__(
        self,
        cpu_group,
        device: torch.device | None = None,
        device_group=None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group=None,
        use_all2all: bool = False,
    ):
        self.sircl = None
        if not settings.enabled():
            super().__init__(cpu_group, device, device_group, unique_name, global_ranks,
                             global_world_size, tcp_store_group=tcp_store_group,
                             use_all2all=use_all2all)
            return
        import torch.distributed as dist

        ranks = groupops.ranks(cpu_group, global_ranks)
        world = int(global_world_size or dist.get_world_size())
        config = _config(world)
        rank = ranks.index(dist.get_rank()) if global_ranks is None else groupops.rank(cpu_group)
        kind = unique_name.split(":")[0]
        parent = None
        if kind in adapter_module.SUBGROUP_KINDS and len(ranks) > 1:
            from vllm.distributed.parallel_state import get_tp_group

            parent = list(get_tp_group().ranks)
        placement = GroupPlacement.of(unique_name, ranks, rank, config, parent_ranks=parent)
        from . import p2p as p2p_module

        shape = p2p_module.shape_of(_current_vllm_config(), world)
        problems = tp4.conflicts(placement.topology, kind=kind, communicator_class=CudaCommunicator,
                                 coordinator_class=_group_coordinator_class())
        if getattr(CudaCommunicator.all_reduce, "_rocenante_virtual_diagonal", False):
            problems.append("the RoCEnante virtual-diagonal overlay wraps CudaCommunicator; it and "
                            "SIRCL ring sessions cannot serve one run")
        problems += _config_conflicts(placement)
        problems += _session_conflicts(placement)
        if problems and len(ranks) > 1:
            raise SirclSetupError(f"SIRCL cannot set up {unique_name}: " + "; ".join(problems))
        with guard.pynccl_suppressed(placement.suppress_pynccl):
            super().__init__(cpu_group, device, device_group, unique_name, global_ranks,
                             global_world_size, tcp_store_group=tcp_store_group,
                             use_all2all=use_all2all)
        pynccl = getattr(self, "pynccl_comm", None)
        if placement.suppress_pynccl and pynccl is not None and not getattr(pynccl, "disabled", True):
            raise SirclSetupError(f"PyNccl was built for {unique_name} although NCCL may not "
                                  f"connect its ranks ({placement.reason})")
        if (placement.policy is NcclPolicy.NONE and self.use_all2all
                and getattr(self, "all2all_backend", None) not in _NAIVE_ALL2ALL):
            raise SirclSetupError(
                f"the {self.all2all_backend} all-to-all backend of {unique_name} connects every "
                "pair of ranks itself; on a group NCCL may not run use the naive backend, whose "
                "gathers SIRCL carries")
        if len(ranks) < 2:
            return
        slot = getattr(self, "b12x_ar_comm", None)
        self.sircl = GroupAdapter(
            placement, config=config, cpu_group=cpu_group, device_group=self.device_group,
            device=self.device, slot=slot, nccl=self._nccl_paths(), communicator=self,
            communicator_class=CudaCommunicator, shape=shape, parent_ranks=parent,
        )
        if self.sircl.session is not None:
            _install_worker_regimes(unique_name)
        if self.sircl.session is not None or self.sircl.p2p is not None:
            _install_step_health(unique_name)
        if kind == "tp":
            own = self.sircl.slot
            self.b12x_ar_comm = _chain(own if own is not None and not own.disabled else HubSlot())
            if placement.policy is NcclPolicy.NONE and mhc.requested():
                # GLM-5.3-Flash's mHC prefill sharding reduce-scatters and all-gathers
                # on PyNccl directly; carry it before the model is built (pinned shim).
                if self.sircl.session is None:
                    raise SirclSetupError(f"VLLM_GLM53_MHC_PREFILL_SHARD needs a SIRCL session on "
                                          f"{unique_name}, whose ranks NCCL may not connect")
                try:
                    shims.install(["mhc_prefill_shard"])
                except shims.ShimRefused as exc:
                    raise SirclSetupError(f"SIRCL cannot carry the mHC prefill sharding of {unique_name}: "
                                          f"{exc}; set VLLM_GLM53_MHC_PREFILL_SHARD=0") from exc
            if placement.policy is NcclPolicy.NONE and qwen_hc.requested():
                # Qwen3.8's hyper-connection prefill row ownership reduce-scatters and
                # all-gathers on PyNccl directly; carry it before the model is built.
                if self.sircl.session is None:
                    raise SirclSetupError(f"VLLM_QWEN3_8_HC_PREFILL_MODE=shard needs a SIRCL session on "
                                          f"{unique_name}, whose ranks NCCL may not connect")
                try:
                    shims.install(["qwen_hc_prefill_shard"])
                except shims.ShimRefused as exc:
                    raise SirclSetupError(
                        f"SIRCL cannot carry the hyper-connection prefill row ownership of {unique_name}: {exc}; "
                        "set VLLM_QWEN3_8_HC_PREFILL_MODE=off (the serve launcher's "
                        "--env VLLM_QWEN3_8_HC_PREFILL_MODE=off)") from exc
            if settings.fused_norm():
                from . import norm_fusion

                self.sircl_fused_add_rms_norm = norm_fusion.bind(self)
        if kind == "dcp" and self.sircl.session is not None:
            # vLLM's DCP combine exchanges on the NCCL group directly, and its B12X
            # PCIe transport maps peer memory within one machine; replace both
            # before the attention layers bind them (pinned shims).
            try:
                shims.install(["dcp_all_to_all", "dcp_b12x_transport"])
            except shims.ShimRefused as exc:
                raise SirclSetupError(f"SIRCL cannot carry the DCP collectives of {unique_name}: "
                                      f"{exc}") from exc

    # -- the stock paths SIRCL may hand a collective to ----------------------------------

    def _nccl_paths(self) -> NcclPaths:
        base = CudaCommunicator

        def all_to_all_single(output: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
            torch.distributed.all_to_all_single(output, input_, group=self.device_group)
            return output

        def all_reduce_in_place(input_: torch.Tensor) -> torch.Tensor:
            method = getattr(base, "all_reduce_in_place", None)
            if method is None:
                input_.copy_(base.all_reduce(self, input_))
                return input_
            return method(self, input_)

        return NcclPaths(
            all_reduce=lambda inp: base.all_reduce(self, inp),
            all_reduce_in_place=all_reduce_in_place,
            all_gather=lambda inp, dim: base.all_gather(self, inp, dim),
            all_gatherv=lambda inp, dim, sizes: base.all_gatherv(self, inp, dim, sizes),
            reduce_scatter=lambda inp, dim: base.reduce_scatter(self, inp, dim),
            reduce_scatterv=lambda inp, dim, sizes: base.reduce_scatterv(self, inp, dim, sizes),
            gather=lambda inp, dst, dim: base.gather(self, inp, dst, dim),
            broadcast=lambda tensor, src: base.broadcast(self, tensor, src),
            all_to_all_single=all_to_all_single,
        )

    # -- collectives -------------------------------------------------------------------

    def all_reduce(self, input_):
        if self.sircl is None:
            return super().all_reduce(input_)
        return self.sircl.all_reduce(input_)

    def all_reduce_in_place(self, input_: torch.Tensor) -> torch.Tensor:
        if self.sircl is None:
            return super().all_reduce_in_place(input_)
        return self.sircl.all_reduce(input_, in_place=True)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.sircl is None:
            return super().all_gather(input_, dim)
        return self.sircl.all_gather(input_, dim)

    def all_gatherv(self, input_, dim: int = 0, sizes: list[int] | None = None):
        if self.sircl is None:
            return super().all_gatherv(input_, dim, sizes)
        return self.sircl.all_gatherv(input_, dim, sizes)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        if self.sircl is None:
            return super().reduce_scatter(input_, dim)
        return self.sircl.reduce_scatter(input_, dim)

    def reduce_scatterv(self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None):
        if self.sircl is None:
            return super().reduce_scatterv(input_, dim, sizes)
        return self.sircl.reduce_scatterv(input_, dim, sizes)

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1):
        if self.sircl is None:
            return super().gather(input_, dst, dim)
        return self.sircl.gather(input_, dst, dim)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.sircl is None:
            return super().broadcast(tensor, src)
        return self.sircl.broadcast(tensor, src)

    def all_to_all_single(self, output: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
        """Equal-split all-to-all of flat buffers (the DCP combine's exchange)."""
        if self.sircl is None:
            torch.distributed.all_to_all_single(output, input_, group=self.device_group)
            return output
        return self.sircl.all_to_all_single(output, input_)

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        if self.sircl is None:
            return super().send(tensor, dst)
        peer = (self.rank_in_group + 1) % self.world_size if dst is None else dst
        return self.sircl.send(tensor, peer, lambda: super(SirclCudaCommunicator, self).send(tensor, dst))

    def recv(self, size: torch.Size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor:
        if self.sircl is None:
            return super().recv(size, dtype, src)
        peer = (self.rank_in_group - 1) % self.world_size if src is None else src
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        return self.sircl.recv(tensor, peer, lambda: super(SirclCudaCommunicator, self).recv(size, dtype, src))

    def batch_isend_irecv(self, p2p_ops: list):
        """``P2POp`` lists as vLLM builds them: ``op`` is ``torch.distributed.isend`` or ``irecv``, the peer a
        global rank (``peer``) or a group rank (``group_peer``, elastic expert parallelism)."""
        if self.sircl is None:
            return super().batch_isend_irecv(p2p_ops)
        ops = []
        for op in p2p_ops:
            name = getattr(getattr(op, "op", None), "__name__", "")
            peer = getattr(op, "group_peer", None)
            if peer is None:
                global_peer = getattr(op, "peer", None)
                if global_peer is None or global_peer not in self.ranks:
                    raise SirclSetupError(f"a batched point-to-point op names peer {global_peer}, which is not a "
                                          f"rank of {self.unique_name}")
                peer = self.ranks.index(global_peer)
            if name not in ("isend", "irecv"):
                raise SirclSetupError(f"a batched point-to-point op is {name or type(op).__name__}, not isend or "
                                      "irecv")
            ops.append(("send" if name == "isend" else "recv", op.tensor, int(peer)))
        return self.sircl.batch_isend_irecv(
            ops, lambda: super(SirclCudaCommunicator, self).batch_isend_irecv(p2p_ops))

    # -- reporting and teardown --------------------------------------------------------

    def sircl_report(self) -> dict[str, Any] | None:
        """The group's receipt with its plan counters (None when SIRCL does not own the group)."""
        return None if self.sircl is None else self.sircl.report()

    def destroy(self):
        self.sircl_fused_add_rms_norm = None
        sircl, self.sircl = self.sircl, None
        if sircl is not None:
            with contextlib.suppress(Exception):
                sircl.close()
        if isinstance(getattr(self, "b12x_ar_comm", None), HubSlot):
            self.b12x_ar_comm = None
        super().destroy()


def _group_coordinator_class() -> type | None:
    try:
        from vllm.distributed.parallel_state import GroupCoordinator
    except ImportError:
        return None
    return GroupCoordinator
