"""Where each collective enters SIRCL in vLLM, as checkable data.

The adapter carries collectives through two official extension points:

- ``vllm.platform_plugins``: SIRCL's platform is a subclass of vLLM's
  ``CudaPlatform`` (so ``is_cuda()`` and every CUDA code path stay unchanged)
  whose ``get_device_communicator_cls`` returns
  :class:`sparkring_sircl.vllm.communicator.SirclCudaCommunicator`, a
  ``CudaCommunicator`` subclass. Every group coordinator (TP, DCP, EP, ...)
  then sends its device collectives through SIRCL's plans first.
- ``vllm.general_plugins``: runs in every vLLM process before any process
  group exists, checks the NCCL environment, installs the NCCL tripwire and
  the version-pinned shims.

Each :class:`Hook` names the collective, the mechanism, the vLLM files and
line ranges it relies on in the image's build (:mod:`.pins`), an anchor text
that must appear in that range, whether a pinned shim is needed, and the
pinned builds that have the code (``builds``; empty for every pinned build).
:func:`verify` checks the anchors against a vLLM source tree; line numbers may
drift by up to ``slack`` lines between pinned builds. :func:`applicable`
selects the hooks of the builds a tree matches.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Anchor:
    file: str            # path relative to the vllm package
    first: int           # first line of the relied-on range (image build numbering)
    last: int
    text: str            # text that must appear in the range


@dataclasses.dataclass(frozen=True)
class Hook:
    collective: str
    mechanism: str
    official: bool
    anchors: tuple[Anchor, ...]
    shim: str | None
    status: str
    notes: str
    builds: tuple[str, ...] = ()        # pinned builds that have this code; empty: every one


HOOKS: tuple[Hook, ...] = (
    Hook(
        collective="Device communicator of every group",
        mechanism="platform plugin -> SirclCudaPlatform.get_device_communicator_cls -> "
                  "SirclCudaCommunicator",
        official=True,
        anchors=(
            Anchor("platforms/__init__.py", 263, 272, "activated_oot_plugins"),
            Anchor("platforms/cuda.py", 625, 628, "def get_device_communicator_cls"),
            Anchor("distributed/parallel_state.py", 612, 613,
                   "current_platform.get_device_communicator_cls()"),
        ),
        shim=None,
        status="implemented",
        notes="One out-of-tree platform plugin may activate; it is loaded through the same "
              "VLLM_PLUGINS filter as general plugins.",
    ),
    Hook(
        collective="PyNccl construction and its warm-up all-reduce",
        mechanism="VLLM_DISABLE_PYNCCL set only while SirclCudaCommunicator builds a group NCCL "
                  "may not run",
        official=True,
        anchors=(
            Anchor("distributed/device_communicators/pynccl.py", 127, 130,
                   "if self.world_size == 1 or envs.VLLM_DISABLE_PYNCCL:"),
            Anchor("distributed/device_communicators/pynccl.py", 186, 191, "self.all_reduce(data)"),
            Anchor("distributed/device_communicators/cuda_communicator.py", 165, 173,
                   "_acquire_pynccl("),
            Anchor("v1/executor/multiproc_executor.py", 846, 848, "enable_envs_cache()"),
        ),
        shim=None,
        status="implemented",
        notes="vLLM reads its environment live until the worker caches it after loading the "
              "model; groups are built before that. The guard refuses if the cache is already on.",
    ),
    Hook(
        collective="TP all-reduce (and the in-place variant the MoE runner uses)",
        mechanism="SirclCudaCommunicator.all_reduce / all_reduce_in_place",
        official=True,
        anchors=(
            Anchor("distributed/communication_op.py", 12, 14, "return get_tp_group().all_reduce(input_)"),
            Anchor("distributed/parallel_state.py", 827, 830,
                   "return self.device_communicator.all_reduce(input_)"),
            Anchor("distributed/parallel_state.py", 816, 825,
                   'reduce_in_place = getattr(self.device_communicator, "all_reduce_in_place", None)'),
        ),
        shim=None,
        status="implemented (ring session; chunked above the dispatch ceiling)",
        notes="The torch.compile custom op vllm.all_reduce resolves the group and calls the "
              "same communicator method.",
    ),
    Hook(
        collective="TP all-gather (logits, vocabulary, column-parallel gather_output, MLA latent "
                   "query gather on the TP group)",
        mechanism="SirclCudaCommunicator.all_gather (any dimension)",
        official=True,
        anchors=(
            Anchor("distributed/parallel_state.py", 848, 851,
                   "return self.device_communicator.all_gather(input_, dim)"),
            Anchor("models/deepseek_v32/attention.py", 604, 609,
                   "get_tp_group().all_gather(mqa_q_arg, dim=1)"),
        ),
        shim=None,
        status="implemented (ring session; rows or tiles above the gather op size)",
        notes="A gather along dimension k of a contiguous tensor is the last-dimension gather of "
              "its [outer, inner] view.",
    ),
    Hook(
        collective="All-gatherv, reduce-scatter, reduce-scatterv and gather",
        mechanism="SirclCudaCommunicator methods of the same names",
        official=True,
        anchors=(
            Anchor("distributed/parallel_state.py", 853, 861,
                   "return self.device_communicator.all_gatherv(input_, dim, sizes)"),
            Anchor("distributed/parallel_state.py", 879, 884,
                   "return self.device_communicator.reduce_scatterv(input_, dim, sizes)"),
            Anchor("distributed/parallel_state.py", 886, 889,
                   "return self.device_communicator.reduce_scatter(input_, dim)"),
            Anchor("distributed/parallel_state.py", 891, 905,
                   "return self.device_communicator.gather(input_, dst, dim)"),
        ),
        shim=None,
        status="implemented (session reduce-scatter where prepared, else all-reduce and slice; "
               "uneven sizes padded)",
        notes="",
    ),
    Hook(
        collective="DCP all-gather (LSE gather, query gather, indexer top-k merge)",
        mechanism="SirclCudaCommunicator.all_gather on the dcp group",
        official=True,
        anchors=(
            Anchor("v1/attention/ops/dcp.py", 455, 457, "cp_group.all_gather(cp_attn_lse, dim=0)"),
            Anchor("v1/attention/ops/dcp.py", 1705, 1708, "self.group.all_gather(query, dim=1)"),
        ),
        shim=None,
        status="implemented",
        notes="MLADCPManager uses this path when neither the fork's B12X PCIe DCP transport nor "
              "the symmetric-memory workspace is selected; both need ranks on one node.",
    ),
    Hook(
        collective="DCP all-to-all (LSE-weighted output combine, the GLM default)",
        mechanism="pinned shim dcp_all_to_all replacing vllm.v1.attention.ops.dcp."
                  "dcp_a2a_lse_reduce; exchange through SirclCudaCommunicator.all_to_all_single",
        official=False,
        anchors=(
            Anchor("v1/attention/ops/dcp.py", 951, 960, "def dcp_a2a_lse_reduce("),
            Anchor("v1/attention/ops/dcp.py", 1012, 1017, "dist.all_to_all_single("),
        ),
        shim="dcp_all_to_all",
        status="implemented; installed when a multi-rank DCP group is built",
        notes="dcp_a2a_lse_reduce calls torch.distributed.all_to_all_single on the NCCL device "
              "group directly, bypassing the device communicator, so no official hook reaches it.",
    ),
    Hook(
        collective="Point-to-point send and receive",
        mechanism="SirclCudaCommunicator.send / recv / batch_isend_irecv on the group's SIRCL "
                  "point-to-point channels; torch.distributed.isend / irecv / send / recv / "
                  "batch_isend_irecv (and broadcast on a group without a session) on the device group "
                  "reach the same channels through the NCCL tripwire's point-to-point carrier; NCCL "
                  "only between cabled ranks of a group without channels, else refused",
        official=True,
        anchors=(
            Anchor("distributed/parallel_state.py", 1447, 1461,
                   "self.device_communicator.send(tensor, dst)"),
            Anchor("distributed/parallel_state.py", 1290, 1294, "torch.distributed.isend("),
            Anchor("distributed/parallel_state.py", 1406, 1410, "torch.distributed.irecv("),
            Anchor("distributed/device_communicators/cuda_communicator.py", 1046, 1051,
                   "def batch_isend_irecv(self, p2p_ops: list):"),
        ),
        shim=None,
        status="implemented; CPU tests, simulator and GPU emulation; not run on a ring",
        notes="isend_tensor_dict / irecv_tensor_dict (pipeline-parallel activations) call "
              "torch.distributed on the device group directly, bypassing the device communicator, so "
              "the tripwire carries them. Eager only: a send or receive under CUDA graph capture is "
              "refused.",
    ),
    Hook(
        collective="GLM-5.3-Flash mHC prefill row ownership (reduce-scatter and all-gather to "
                   "PyNccl directly, VLLM_GLM53_MHC_PREFILL_SHARD=1)",
        mechanism="pinned shim mhc_prefill_shard wrapping mhc_prefill_sharding.maybe_create: "
                  "the group's PyNccl slot holds SirclPrefillComm while vLLM builds the ownership "
                  "object, so its reduce_scatter and all_gather go through SirclCudaCommunicator",
        official=False,
        anchors=(
            Anchor("models/glm5next/nvidia/mhc_prefill_sharding.py", 339, 351,
                   "self.comm.reduce_scatter(output, partial, stream=stream)"),
            Anchor("models/glm5next/nvidia/mhc_prefill_sharding.py", 353, 363,
                   "self.comm.all_gather(output, owned, stream=stream)"),
            Anchor("models/glm5next/nvidia/mhc_prefill_sharding.py", 447, 466,
                   'comm = getattr(group.device_communicator, "pynccl_comm", None)'),
            Anchor("models/glm5next/nvidia/model.py", 110, 112,
                   "maybe_create as maybe_create_mhc_prefill_ownership"),
            Anchor("models/glm5next/nvidia/model.py", 1169, 1171,
                   "mhc_owner = maybe_create_mhc_prefill_ownership(self, hidden_states, positions)"),
            Anchor("models/glm5next/nvidia/model.py", 671, 673,
                   "x = mhc_prefill_ownership.reduce_scatter(x)"),
            Anchor("models/glm5next/nvidia/model.py", 694, 698,
                   "x = mhc_prefill_ownership.all_gather(x)"),
            Anchor("models/glm5next/nvidia/model.py", 1216, 1218,
                   "hidden_states = mhc_owner.all_gather(hidden_states)"),
        ),
        shim="mhc_prefill_shard",
        status="implemented; installed when a tensor-parallel group without PyNccl is built with a "
               "session and VLLM_GLM53_MHC_PREFILL_SHARD set",
        notes="Per 8,192-token chunk at TP4 (45 layers): 90 reduce-scatters of [8192, 4096] BF16 and "
              "90 all-gathers of [2048, 4096] BF16 shards. Groups with a working PyNccl keep vLLM's "
              "path.",
        builds=("lil-image-aba309e4610c", "sparkring-kraken-beta-20261007-bc9ea774"),
    ),
    Hook(
        collective="Qwen3.8 hyper-connection prefill row ownership (reduce-scatter and all-gather to "
                   "PyNccl directly, VLLM_QWEN3_8_HC_PREFILL_MODE=shard)",
        mechanism="pinned shim qwen_hc_prefill_shard wrapping hc_prefill.create: the group's PyNccl slot "
                  "holds SirclPrefillComm while vLLM builds the ownership object, so its reduce_scatter "
                  "and all_gather go through SirclCudaCommunicator",
        official=False,
        anchors=(
            Anchor("models/qwen4_exp/nvidia/hc_prefill.py", 104, 111, "self.group.all_gather(output, source)"),
            Anchor("models/qwen4_exp/nvidia/hc_prefill.py", 113, 120,
                   "self.group.reduce_scatter(output, source)"),
            Anchor("models/qwen4_exp/nvidia/hc_prefill.py", 123, 130,
                   'comm = getattr(group.device_communicator, "pynccl_comm", None)'),
            Anchor("models/qwen4_exp/nvidia/model.py", 595, 595,
                   "hc_owner = hc_prefill.create(self, full_rows) if hc_prefill_eager else None"),
            Anchor("models/qwen4_exp/nvidia/model.py", 387, 387, "hc_owner.reduce(attn_out)"),
            Anchor("models/qwen4_exp/nvidia/model.py", 405, 405, "hc_owner.reduce(mlp_out)"),
            Anchor("models/qwen4_exp/nvidia/model.py", 374, 374, "block_input = hc_owner.gather(block_input)"),
            Anchor("models/qwen4_exp/nvidia/model.py", 785, 785,
                   "if hc_prefill.eligible(self.model, positions.shape[-1]):"),
        ),
        shim="qwen_hc_prefill_shard",
        status="implemented; installed when a tensor-parallel group without PyNccl is built with a "
               "session and VLLM_QWEN3_8_HC_PREFILL_MODE=shard",
        notes="Per admitted eager pure prefill of L layers: 2 L reduce-scatters of [rows, H] BF16 and "
              "2 L + 1 all-gathers of [rows / W, H] shards, plus one of [rows / W, hc H] per PLE layer "
              "and one for the MTP drafter. Groups with a working PyNccl keep vLLM's path.",
        builds=("lil-image-aba309e4610c", "sparkring-kraken-beta-20261007-bc9ea774"),
    ),
    Hook(
        collective="Broadcasts that bypass the device communicator",
        mechanism="NCCL tripwire on torch.distributed (refuses NCCL calls on a group NCCL may "
                  "not run)",
        official=False,
        anchors=(
            Anchor("distributed/parallel_state.py", 907, 920, "torch.distributed.broadcast("),
        ),
        shim=None,
        status="guard implemented; no SIRCL path (GroupCoordinator.broadcast and "
               "broadcast_tensor_dict are not used by tensor-parallel serving at DP=1, PP=1)",
        notes="SirclCudaCommunicator.broadcast carries device-communicator broadcasts.",
    ),
    Hook(
        collective="Expert-parallel group of mixture-of-experts models",
        mechanism="SirclCudaCommunicator without PyNccl and without a session; collectives refused",
        official=True,
        anchors=(
            Anchor("distributed/parallel_state.py", 2288, 2321,
                   "if config.model_config is None or config.model_config.is_moe:"),
        ),
        shim=None,
        status="implemented",
        notes="vLLM builds the EP group for every MoE model, even without expert parallelism; "
              "at DP=1 it carries no traffic.",
    ),
    Hook(
        collective="RoCE all-reduce slot of the tensor-parallel group",
        mechanism="pinned shim roce_slot (B12xRoceAllReduce -> SirclRingAllReduce) when "
                  "VLLM_ENABLE_ROCE_ALLREDUCE=1; otherwise SirclCudaCommunicator builds the slot",
        official=False,
        anchors=(
            Anchor("distributed/device_communicators/cuda_communicator.py", 193, 203,
                   "self.b12x_ar_comm = B12xRoceAllReduce("),
            Anchor("distributed/device_communicators/b12x_roce_all_reduce.py", 56, 59,
                   "class B12xRoceAllReduce:"),
        ),
        shim="roce_slot",
        status="implemented",
        notes="",
    ),
    Hook(
        collective="Fused residual-add + RMSNorm all-reduce (compiled graphs)",
        mechanism="not carried: on a group NCCL may not run SIRCL refuses "
                  "pass_config.fuse_allreduce_rms at setup; official route for a future carrier: a "
                  "post-grad custom pass set by the platform's check_and_update_config",
        official=False,
        anchors=(
            Anchor("compilation/passes/fusion/allreduce_rms_fusion.py", 316, 334,
                   "communicator = get_b12x_pcie_allreduce()"),
            Anchor("compilation/passes/fusion/allreduce_rms_fusion.py", 1097, 1120,
                   "if get_b12x_pcie_allreduce() is not None:"),
            Anchor("distributed/device_communicators/b12x_pcie_all_reduce.py", 1036, 1050,
                   "isinstance(communicator, B12xPcieAllReduce)"),
            Anchor("compilation/backends.py", 955, 971, "if self.pass_key in self.inductor_config:"),
        ),
        shim=None,
        status="unsupported (refused at setup on groups NCCL may not run)",
        notes="The pass fuses through FlashInfer (no SM121 size entry) or the B12X PCIe "
              "communicator (one machine); on SM121 it registers no pattern without the latter, "
              "and SIRCL's tensor-parallel slot is not one.",
    ),
    Hook(
        collective="Fused residual-add + RMSNorm all-reduce (eager model helper)",
        mechanism="pinned shim fused_allreduce_rms_norm wraps vLLM's helper where it is defined and "
                  "bound; with SIRCL_FUSED_NORM=1 the TP group's session runs the all-reduce, residual "
                  "add and RMSNorm as one kernel (norm_fusion.py); otherwise the helper's all-reduce "
                  "reaches SIRCL through the communicator and vLLM's RMSNorm follows",
        official=False,
        anchors=(
            Anchor("models/common/ops/fused_allreduce_rms_norm.py", 23, 27,
                   "def fused_allreduce_rms_norm("),
            Anchor("models/common/ops/fused_allreduce_rms_norm.py", 62, 63,
                   "return norm(reduced, residual)"),
            Anchor("compilation/passes/fusion/allreduce_rms_fusion.py", 108, 131,
                   "FI_ALLREDUCE_FUSION_MAX_SIZE_MB"),
            Anchor("model_executor/layers/layernorm.py", 87, 94,
                   "ir.ops.fused_add_rms_norm.maybe_inplace("),
            Anchor("kernels/vllm_c.py", 54, 86,
                   "torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)"),
            Anchor("ir/ops/layernorm.py", 43, 62, "def fused_add_rms_norm("),
            Anchor("platforms/cuda.py", 756, 778,
                   'default = ["native"] if using_inductor else ["vllm_c", "native"]'),
            Anchor("v1/worker/worker_base.py", 97, 99, "ir_op_priority.set_default()"),
        ),
        shim="fused_allreduce_rms_norm",
        status="shim implemented and pinned; fused kernels research-only (off unless SIRCL_FUSED_NORM=1)",
        notes="Bit-identical to SIRCL's all-reduce followed by the vllm_c fused_add_rms_norm kernel; "
              "setup refuses SIRCL_FUSED_NORM=1 when vLLM's RMSNorm dispatches to another provider "
              "(native under inductor compilation) or VLLM_BATCH_INVARIANT=1. On SM121 the helper's "
              "FlashInfer path never applies (no 12.1 size entry).",
    ),
    Hook(
        collective="Post-step health check and CUDA graph capture context",
        mechanism="the tensor-parallel communicator's b12x_ar_comm, chained to every SIRCL "
                  "session of the process",
        official=False,
        anchors=(
            Anchor("v1/worker/gpu_worker.py", 1401, 1410,
                   'comm = getattr(communicator, "b12x_ar_comm", None)'),
            Anchor("distributed/parallel_state.py", 757, 763,
                   "maybe_b12x_context = b12x_ar_comm.capture(stream=stream)"),
            Anchor("distributed/parallel_state.py", 1739, 1745, "get_tp_group().graph_capture(context)"),
        ),
        shim=None,
        status="implemented (attribute on SIRCL's own communicator; no vLLM change)",
        notes="vLLM enters only the TP, PP and DP groups' capture contexts and checks only the TP "
              "slot after a step, so DCP sessions are reached through the TP slot.",
    ),
    Hook(
        collective="DCP B12X PCIe transport (query gather and combine within one machine)",
        mechanism="get_b12x_dcp_transport returns None for a group SIRCL owns, so the MLA DCP manager keeps "
                  "the a2a combine and the coordinator's query all-gather",
        official=False,
        anchors=(
            Anchor("v1/attention/ops/dcp.py", 1521, 1535, "self.b12x_transport = get_b12x_dcp_transport("),
            Anchor("distributed/device_communicators/b12x_dcp.py", 185, 185, "def get_b12x_dcp_transport("),
        ),
        shim="dcp_b12x_transport",
        status="shim implemented and pinned",
        notes="Used only with the B12X attention backend and VLLM_USE_B12X_DCP_A2A=1 "
              "(model_executor/layers/attention/mla_attention.py:878-882).",
    ),
    Hook(
        collective="Direct torch.distributed collectives on groups with a SIRCL session",
        mechanism="the tripwire hands a refused call to the group's carrier, which runs it through the "
                  "adapter's plans",
        official=False,
        anchors=(
            Anchor("v1/attention/ops/dcp.py", 1755, 1759, "torch.distributed.all_gather_into_tensor"),
            Anchor("model_executor/layers/quantization/utils/quant_utils.py", 65, 69,
                   "group=get_ep_group().device_group"),
        ),
        shim=None,
        status="implemented (torch.distributed wrappers installed by the general plugin)",
        notes="Carries the DCP chunked-context KV gather, the online-quantization amax reductions on the EP and "
              "TP groups, and other direct all_reduce, broadcast, all_gather, reduce_scatter_tensor and "
              "all_to_all_single calls; other calls stay refused.",
    ),
    Hook(
        collective="Flag-wait regimes of every SIRCL session",
        mechanism="serving regime at the post-step health check; the worker_regimes shim runs the "
                  "worker's warm-up, profiling, sleep, wake-up, weight-load and profiler methods in "
                  "the startup regime",
        official=False,
        anchors=(
            Anchor("v1/worker/gpu_worker.py", 1400, 1400,
                   "return self._b12x_roce_guarded(self.model_runner.sample_tokens"),
            Anchor("v1/worker/gpu_worker.py", 1002, 1002, "def compile_or_warm_up_model(self)"),
            Anchor("v1/worker/gpu_worker.py", 636, 636, "def determine_available_memory(self)"),
            Anchor("v1/worker/gpu_worker.py", 278, 278, "def sleep(self, level: int = 1)"),
            Anchor("v1/worker/gpu_worker.py", 315, 315, "def wake_up(self, tags"),
            Anchor("v1/worker/gpu_worker.py", 562, 562, "def reload_weights(self"),
            Anchor("v1/worker/gpu_worker.py", 1720, 1720, "def update_weights(self"),
            Anchor("v1/worker/gpu_worker.py", 1543, 1543, "def profile(self"),
        ),
        shim="worker_regimes",
        status="shim implemented and pinned",
        notes="vLLM's warm-up issues hand-built steps through execute_model (gpu_worker.py:1044 "
              "and 1139), so post-step checks also run during compile_or_warm_up_model; the serving "
              "regime is armed only when that method returns. Without the shim (an unpinned vLLM) "
              "it is never armed and sessions keep the startup limit.",
    ),
    Hook(
        collective="Start-of-step failure check of every SIRCL session",
        mechanism="the step_health shim runs adapter.check_all_failures when each of the worker's step methods "
                  "starts",
        official=False,
        anchors=(
            Anchor("v1/worker/gpu_worker.py", 1397, 1397, "def sample_tokens("),
            Anchor("v1/worker/gpu_worker.py", 1444, 1444, "def execute_model("),
            Anchor("v1/worker/gpu_worker.py", 1612, 1612, "def execute_dummy_batch(self) -> None:"),
        ),
        shim="step_health",
        status="shim implemented and pinned",
        notes="sample_tokens runs the sampler and any draft model's forward pass, whose SIRCL ops come after the "
              "sampled tokens' copy that the post-step check waits for; execute_dummy_batch runs a data-parallel "
              "rank's dummy forward pass. The check reads the sessions' control words and native failure flags in "
              "host memory, never the device poison word.",
    ),
    Hook(
        collective="Plugin order",
        mechanism="general plugins load in every worker before init_device builds process groups",
        official=True,
        anchors=(
            Anchor("v1/worker/worker_base.py", 353, 355, "load_general_plugins()"),
            Anchor("plugins/__init__.py", 63, 70, "for plugin in discovered_plugins:"),
        ),
        shim=None,
        status="implemented",
        notes="vLLM calls general plugins in entry-point discovery order, not VLLM_PLUGINS order; "
              "SIRCL's registration is order-independent and idempotent.",
    ),
)


def applicable(build_names: Sequence[str], hooks: Sequence[Hook] = HOOKS) -> list[Hook]:
    """The hooks whose code the named pinned builds have."""
    names = set(build_names)
    return [hook for hook in hooks if not hook.builds or names & set(hook.builds)]


@dataclasses.dataclass(frozen=True)
class AnchorResult:
    anchor: Anchor
    found_at: int | None


def verify(root: Path, hooks: Sequence[Hook] = HOOKS, *, slack: int = 120) -> list[AnchorResult]:
    """Locate every anchor's text in ``root`` near its recorded range (line numbers drift)."""
    results = []
    for hook in hooks:
        for anchor in hook.anchors:
            path = root / anchor.file
            found = None
            if path.is_file():
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                start = max(anchor.first - 1 - slack, 0)
                for index in range(start, min(anchor.last + slack, len(lines))):
                    if anchor.text in lines[index]:
                        found = index + 1
                        break
            results.append(AnchorResult(anchor, found))
    return results
