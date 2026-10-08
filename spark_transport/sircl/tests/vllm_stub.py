"""A minimal stand-in vLLM package for the adapter's CPU tests.

It reproduces, as small files on disk, only the vLLM behaviour the adapter's
hooks rely on (see ``sparkring_sircl.vllm.hooks``): the platform's
communicator class name, ``CudaCommunicator`` building PyNccl through the
module function ``_acquire_pynccl`` (which reads ``VLLM_DISABLE_PYNCCL`` and
otherwise runs a warm-up all-reduce) and building the RoCE slot class when
``VLLM_ENABLE_ROCE_ALLREDUCE=1``, the stock collective methods, the group
coordinator's all-gather entry, ``in_the_same_node_as`` and the environment
cache flag. Every stock call is appended to ``EVENTS`` so a test can prove
that SIRCL's plans never reached it.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import MutableMapping
from pathlib import Path

FILES = {
    "vllm/__init__.py": "",
    "vllm/envs.py": '''
CACHE = False
VLLM_BATCH_INVARIANT = False


def _is_envs_cache_enabled():
    return CACHE
''',
    "vllm/platforms/__init__.py": '''
import importlib

# Tests set this to stand for a platform plugin that failed to load.
SKIP_OUT_OF_TREE = False


def cuda_platform_plugin():
    return "vllm.platforms.cuda.CudaPlatform"


def __getattr__(name):
    # vLLM resolves current_platform lazily: an activated out-of-tree platform
    # plugin wins over the built-in CUDA platform.
    if name != "current_platform":
        raise AttributeError(name)
    qualname = cuda_platform_plugin()
    if not SKIP_OUT_OF_TREE:
        from sparkring_sircl.vllm import platform

        qualname = platform.activate() or qualname
    module, _, cls = qualname.rpartition(".")
    return getattr(importlib.import_module(module), cls)()
''',
    "vllm/platforms/cuda.py": '''
class CudaPlatform:
    @classmethod
    def get_device_communicator_cls(cls):
        return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"
''',
    "vllm/distributed/__init__.py": "",
    "vllm/distributed/parallel_state.py": '''
import threading


class GroupCoordinator:
    def _all_gather_out_place(self, input_, dim):
        return self.device_communicator.all_gather(input_, dim)


TP_RANKS = None
_LOCAL = threading.local()


class _Tp:
    @property
    def ranks(self):
        return list(TP_RANKS)


def set_tp_group(group):
    """Tests set the calling thread's (emulated rank's) tensor-parallel group coordinator."""
    _LOCAL.group = group


def get_tp_group():
    group = getattr(_LOCAL, "group", None)
    return group if group is not None else _Tp()


def in_the_same_node_as(group, source_rank=0):
    return [False] * group.sircl_size()
''',
    "vllm/distributed/device_communicators/__init__.py": "",
    "vllm/distributed/device_communicators/b12x_roce_all_reduce.py": '''
class B12xRoceAllReduce:
    """Another RDMA transport in vLLM's RoCE slot."""

    backend_name = "B12X_ROCENANTE"

    def __init__(self, group, device_group, device, *, global_ranks=None):
        self.disabled = False

    def close(self):
        pass
''',
    "vllm/distributed/device_communicators/cuda_communicator.py": '''
import os

import torch

EVENTS = []


class FakePyNccl:
    def __init__(self, disabled):
        self.disabled = disabled
        self.available = not disabled


def _acquire_pynccl(group, device):
    disabled = os.environ.get("VLLM_DISABLE_PYNCCL", "0").lower() in ("1", "true")
    EVENTS.append(("pynccl", group.sircl_rank(), "disabled" if disabled else "warm-up all-reduce"))
    return FakePyNccl(disabled), None, True


class CudaCommunicator:
    def __init__(self, cpu_group, device=None, device_group=None, unique_name="",
                 global_ranks=None, global_world_size=None, tcp_store_group=None,
                 use_all2all=False):
        self.cpu_group = cpu_group
        self.device = torch.device("cpu")
        self.device_group = device_group
        self.unique_name = unique_name
        self.ranks = list(global_ranks)
        self.world_size = len(self.ranks)
        self.rank_in_group = cpu_group.sircl_rank()
        self.use_all2all = use_all2all
        self.all2all_backend = "allgather_reducescatter"
        self.pynccl_comm = None
        if self.world_size > 1:
            self.pynccl_comm, _, _ = _acquire_pynccl(cpu_group, self.device)
        self.b12x_ar_comm = None
        if (unique_name.split(":")[0] == "tp" and self.world_size > 1
                and os.environ.get("VLLM_ENABLE_ROCE_ALLREDUCE") == "1"):
            from .b12x_roce_all_reduce import B12xRoceAllReduce

            self.b12x_ar_comm = B12xRoceAllReduce(group=cpu_group, device_group=device_group,
                                                  device=self.device, global_ranks=self.ranks)

    def _stock(self, name, tensor):
        EVENTS.append(("stock " + name, self.unique_name))
        return tensor.clone()

    def all_reduce(self, input_):
        return self._stock("all_reduce", input_)

    def all_reduce_in_place(self, input_):
        return self._stock("all_reduce_in_place", input_)

    def all_gather(self, input_, dim=-1):
        return self._stock("all_gather", input_)

    def all_gatherv(self, input_, dim=0, sizes=None):
        return self._stock("all_gatherv", input_)

    def reduce_scatter(self, input_, dim=-1):
        return self._stock("reduce_scatter", input_)

    def reduce_scatterv(self, input_, dim=-1, sizes=None):
        return self._stock("reduce_scatterv", input_)

    def gather(self, input_, dst=0, dim=-1):
        return self._stock("gather", input_)

    def broadcast(self, tensor, src=0):
        return self._stock("broadcast", tensor)

    def send(self, tensor, dst=None):
        EVENTS.append(("stock send", self.unique_name))

    def recv(self, size, dtype, src=None):
        EVENTS.append(("stock recv", self.unique_name))
        return torch.empty(size, dtype=dtype)

    def batch_isend_irecv(self, p2p_ops):
        EVENTS.append(("stock batch_isend_irecv", self.unique_name))

    def destroy(self):
        if self.b12x_ar_comm is not None:
            self.b12x_ar_comm.close()
''',
}

# vLLM's DCP output combine as the adapter's dcp_all_to_all shim relies on it:
# a module function that packs, exchanges on the NCCL device group with
# torch.distributed.all_to_all_single and unpacks with module helpers. The
# combine here keeps the arithmetic simple (a sum over source ranks).
FILES["vllm/v1/__init__.py"] = ""
FILES["vllm/v1/attention/__init__.py"] = ""
FILES["vllm/v1/attention/ops/__init__.py"] = ""
FILES["vllm/v1/attention/ops/dcp.py"] = '''
import functools

import torch
import torch.distributed as dist


def _dcp_a2a_lse_pack_dim(output_dtype):
    return 2


def _dcp_a2a_send_recv_buffers(shape, device, dtype):
    return torch.zeros(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device)


def _dcp_a2a_pack_send(cp_attn_out, cp_attn_lse, send_buffer, world_size, heads_per_rank, head_dim,
                       lse_pack_dim, seq_lens=None, query_start_loc=None):
    for peer in range(world_size):
        send_buffer[peer, :, :, :head_dim] = cp_attn_out[:, peer * heads_per_rank:(peer + 1) * heads_per_rank]


def _dcp_a2a_unpack_combine(recv_buffer, head_dim, lse_pack_dim, return_lse, is_lse_base_on_e):
    return recv_buffer[..., :head_dim].float().sum(0).to(recv_buffer.dtype)


def dcp_a2a_lse_reduce(cp_attn_out, cp_attn_lse, cp_group, ctx=None, return_lse=False,
                       is_lse_base_on_e=True, seq_lens=None, query_start_loc=None):
    world_size = cp_group.world_size
    tokens, heads, head_dim = cp_attn_out.shape
    heads_per_rank = heads // world_size
    send_buffer, recv_buffer = _dcp_a2a_send_recv_buffers(
        (world_size, tokens, heads_per_rank, head_dim + 2), cp_attn_out.device, cp_attn_out.dtype)
    _dcp_a2a_pack_send(cp_attn_out, cp_attn_lse, send_buffer, world_size, heads_per_rank, head_dim, 2)
    dist.all_to_all_single(recv_buffer.view(-1), send_buffer.view(-1), group=cp_group.device_group)
    return _dcp_a2a_unpack_combine(recv_buffer, head_dim, 2, return_lse, is_lse_base_on_e)


class MLADCPManager:
    """vLLM's MLA DCP manager as SIRCL relies on it: transport choice, combine and KV gather."""

    def __init__(self, group, *, use_b12x=False, use_a2a=True):
        self.group = group
        self.b12x_transport = None
        if use_b12x and use_a2a:
            from vllm.distributed.device_communicators.b12x_dcp import get_b12x_dcp_transport

            self.b12x_transport = get_b12x_dcp_transport(group, "cpu", 16, 8, 576, 512, None, None)
        if self.b12x_transport is not None:
            self.combine = self.b12x_transport.combine
            self.query_gather = self.b12x_transport.gather
            return
        self.combine = functools.partial(dcp_a2a_lse_reduce, cp_group=group)
        self.query_gather = lambda query: group.all_gather(query, dim=1)

    def init_kv_gather(self, workspace, max_gathered_tokens):
        self._kv_gather = functools.partial(torch.distributed.all_gather_into_tensor,
                                            group=self.group.device_group)

    def kv_gather(self, gathered_kv, local_kv):
        return self._kv_gather(gathered_kv, local_kv)
'''
FILES["vllm/distributed/device_communicators/b12x_dcp.py"] = '''
class B12xDCPTransport:
    """The B12X PCIe DCP transport: CUDA IPC between GPUs of one machine."""

    def __init__(self, group):
        self.group = group

    def gather(self, query):
        raise RuntimeError("the B12X PCIe DCP transport maps peer memory within one machine only")

    def combine(self, output, lse, **kwargs):
        raise RuntimeError("the B12X PCIe DCP transport maps peer memory within one machine only")


def get_b12x_dcp_transport(group, device, max_tokens, num_heads, query_dim, output_dim, query_dtype,
                           output_dtype):
    return B12xDCPTransport(group)
'''

# GLM-5.3-Flash's mHC prefill row ownership as the mhc_prefill_shard shim relies on it:
# maybe_create reads the TP group's PyNccl communicator, checks its availability,
# rank and device, votes over the CPU group and returns an ownership object whose
# reduce_scatter and all_gather call that communicator directly; the model module
# binds maybe_create under its own name when it is imported, and importing the
# GLM package imports the model module.
FILES["vllm/v1/worker/__init__.py"] = ""
FILES["vllm/v1/worker/gpu_worker.py"] = '''
class Worker:
    """vLLM's GPU worker as SIRCL's regime shim sees it.

    Every method runs ``Worker.DURING(name)`` when a test sets it, standing for
    the work the method does (compilation, capture, sleeping) while SIRCL's
    sessions are in the regime the shim selected.
    """

    DURING = None

    def _run(self, name):
        if Worker.DURING is not None:
            Worker.DURING(name)
        return name

    def compile_or_warm_up_model(self):
        return self._run("compile_or_warm_up_model")

    def determine_available_memory(self):
        return self._run("determine_available_memory")

    def sleep(self, level=1):
        return self._run("sleep")

    def wake_up(self, tags=None):
        return self._run("wake_up")

    def reload_weights(self, *args, **kwargs):
        return self._run("reload_weights")

    def update_weights(self, update_info):
        return self._run("update_weights")

    def profile(self, is_start=True, profile_prefix=None):
        return self._run("profile")

    def execute_model(self, scheduler_output):
        return self._run("execute_model")
'''
FILES["vllm/models/__init__.py"] = ""
FILES["vllm/models/glm5next/__init__.py"] = """
from .nvidia.model import prefill
"""
FILES["vllm/models/glm5next/nvidia/__init__.py"] = ""
FILES["vllm/models/glm5next/nvidia/mhc_prefill_sharding.py"] = '''
from dataclasses import dataclass
from typing import Any


def current_stream():
    return None


@dataclass
class PrefillOwnership:
    comm: Any
    rank: int
    rows: int
    world_size: int
    rs_count: int = 0
    ag_count: int = 0

    def _check(self):
        if not self.comm.available or self.comm.disabled:
            raise RuntimeError("mHC PyNccl communicator became unavailable")

    def reduce_scatter(self, partial):
        self._check()
        output = partial.new_empty((self.rows // self.world_size, partial.shape[1]))
        self.comm.reduce_scatter(output, partial, stream=current_stream())
        self.rs_count += 1
        return output

    def all_gather(self, owned):
        self._check()
        output = owned.new_empty((self.rows, *owned.shape[1:]))
        self.comm.all_gather(output, owned, stream=current_stream())
        self.ag_count += 1
        return output


def maybe_create(model, hidden, positions):
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    comm = getattr(group.device_communicator, "pynccl_comm", None)
    error = None
    if comm is None or not getattr(comm, "available", False) or comm.disabled \\
            or comm.world_size != group.world_size:
        error = "mHC prefill requires the enabled TP PyNccl communicator"
    elif comm.rank != group.rank_in_group or comm.device != hidden.device:
        error = "mHC communicator rank/device ownership mismatch"
    votes = group.vote_object(error)
    errors = [vote for vote in votes if vote is not None]
    if errors:
        raise RuntimeError("mHC prefill capability vote failed: " + repr(errors))
    return PrefillOwnership(comm=comm, rank=group.rank_in_group, rows=hidden.shape[0],
                            world_size=group.world_size)
'''
FILES["vllm/models/glm5next/nvidia/model.py"] = '''
from .mhc_prefill_sharding import maybe_create as maybe_create_mhc_prefill_ownership


def prefill(hidden, partials, positions=None):
    """One eager forward with row ownership: per layer, reduce-scatter the partial, gather the owned rows."""
    owner = maybe_create_mhc_prefill_ownership(None, hidden, positions)
    outputs = []
    for partial in partials:
        owned = owner.reduce_scatter(partial)
        outputs.append(owner.all_gather(owned))
    return owner, outputs
'''

# vLLM's eager post-all-reduce norm helper and the RMSNorm path the fused_allreduce_rms_norm
# shim relies on: the helper all-reduces through the TP group and calls norm(reduced, residual);
# RMSNorm runs the IR op fused_add_rms_norm, whose dispatch names the provider the platform's
# priority selects (PROVIDER, set by tests). The arithmetic stands for the vllm_c kernel's order
# only loosely: the tests compare the fused and unfused paths of this stand-in with each other.
FILES["vllm/ir/__init__.py"] = """
from . import ops
"""
FILES["vllm/ir/op.py"] = '''
class IrOpImpl:
    """One provider's implementation of an IR op."""

    def __init__(self, provider):
        self.provider = provider
'''
FILES["vllm/ir/ops/__init__.py"] = """
from .layernorm import fused_add_rms_norm
"""
FILES["vllm/ir/ops/layernorm.py"] = '''
import torch

from ..op import IrOpImpl

PROVIDER = "vllm_c"


def add_rms_norm(x, x_residual, weight, epsilon):
    z = (x.float() + x_residual.float()).to(x.dtype)
    inverse = torch.rsqrt(z.float().pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return ((z.float() * inverse) * weight.float()).to(x.dtype), z


class _FusedAddRmsNorm:
    def dispatch(self, x, x_residual, weight, epsilon, variance_size=None):
        return IrOpImpl(PROVIDER)

    def maybe_inplace(self, x, x_residual, weight, epsilon, variance_size=None):
        normed, z = add_rms_norm(x, x_residual, weight, epsilon)
        x_residual.copy_(z)
        x.copy_(normed)
        return x, x_residual


fused_add_rms_norm = _FusedAddRmsNorm()
'''
FILES["vllm/kernels/__init__.py"] = ""
FILES["vllm/kernels/vllm_c.py"] = '''
"""vLLM's C kernels: fused_add_rms_norm runs torch.ops._C.fused_add_rms_norm (not loaded here)."""
'''
FILES["vllm/model_executor/__init__.py"] = ""
FILES["vllm/model_executor/layers/__init__.py"] = ""
FILES["vllm/model_executor/layers/layernorm.py"] = '''
import torch

from vllm import ir


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6, var_hidden_size=None, has_weight=True, dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = None if var_hidden_size == hidden_size else var_hidden_size
        self.has_weight = has_weight
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype or torch.bfloat16))
        self.pass_weight_add = has_weight

    def forward_native(self, x, residual=None):
        return ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, self.weight.data, self.variance_epsilon,
                                                       self.variance_size_override)

    def forward_cuda(self, x, residual=None):
        return self.forward_native(x, residual)

    def forward(self, x, residual=None):
        return self.forward_cuda(x, residual)
'''
FILES["vllm/models/common/__init__.py"] = ""
FILES["vllm/models/common/ops/__init__.py"] = ""
FILES["vllm/models/common/ops/fused_allreduce_rms_norm.py"] = '''
from vllm.distributed.parallel_state import get_tp_group


def fused_allreduce_rms_norm(hidden_states, residual, norm):
    """All-reduce + add residual + RMSNorm: the TP group's all-reduce, then the norm (vLLM's path on GB10)."""
    reduced = get_tp_group().device_communicator.all_reduce(hidden_states)
    return norm(reduced, residual)
'''
FILES["vllm/models/deepseek_v32/__init__.py"] = ""
FILES["vllm/models/deepseek_v32/nvidia/__init__.py"] = ""
FILES["vllm/models/deepseek_v32/nvidia/model.py"] = '''
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm


def layer_boundary(hidden_states, residual, norm):
    """The previous sublayer's un-reduced output, reduced into the next RMSNorm."""
    return fused_allreduce_rms_norm(hidden_states, residual, norm)
'''
FILES["vllm/models/deepseek_v32/nvidia/mtp.py"] = '''
from vllm.models.common.ops.fused_allreduce_rms_norm import fused_allreduce_rms_norm


def final_norm(hidden_states, residual, norm):
    hidden_states, _ = fused_allreduce_rms_norm(hidden_states, residual, norm)
    return hidden_states
'''

# Qwen3.8's hyper-connection prefill row ownership as the qwen_hc_prefill_shard shim relies on it:
# create reads the TP group's PyNccl communicator, requires it available and enabled, and returns an
# ownership object whose reduce and gather call that communicator directly with PyNccl's signatures;
# the model looks create up through its module.
FILES["vllm/models/qwen4_exp/__init__.py"] = ""
FILES["vllm/models/qwen4_exp/nvidia/__init__.py"] = ""
FILES["vllm/models/qwen4_exp/nvidia/hc_prefill.py"] = """
from dataclasses import dataclass
from typing import Any

from vllm.distributed.parallel_state import get_tp_group


@dataclass
class RowOwnership:
    rows: int
    rank: int
    group: Any
    size: int = 4
    reductions: int = 0
    gathers: int = 0

    def gather(self, tensor):
        self.gathers += 1
        source = tensor.contiguous()
        output = source.new_empty((self.rows, *source.shape[1:]))
        self.group.all_gather(output, source)
        return output

    def reduce(self, tensor):
        self.reductions += 1
        source = tensor.contiguous()
        output = source.new_empty((self.rows // self.size, *source.shape[1:]))
        self.group.reduce_scatter(output, source)
        return output


def create(model, rows):
    group = get_tp_group()
    comm = getattr(group.device_communicator, "pynccl_comm", None)
    if comm is None or not comm.available or comm.disabled:
        raise RuntimeError("Qwen HC ownership requires the TP NCCL communicator")
    return RowOwnership(rows, group.rank_in_group, comm, group.world_size)
"""
FILES["vllm/models/qwen4_exp/nvidia/model.py"] = """
from . import hc_prefill


def prefill(rows, partials):
    # One eager pure prefill with row ownership: per sublayer, reduce-scatter the partial, gather the owned rows.
    owner = hc_prefill.create(None, rows)
    outputs = []
    for partial in partials:
        owned = owner.reduce(partial)
        outputs.append(owner.gather(owned))
    return owner, outputs
"""

PURGED_PREFIXES = ("vllm", "sparkring_sircl.vllm.communicator", "sparkring_sircl.vllm.cuda_platform")


def write(root: Path) -> Path:
    for name, text in FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.lstrip("\n"), encoding="utf-8")
    return root / "vllm"


def purge() -> None:
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm.") or name in PURGED_PREFIXES[1:]:
            del sys.modules[name]


class ThreadEnviron(MutableMapping):
    """``os.environ`` with per-thread overrides, for emulated ranks that are threads of one process.

    Each rank of a real deployment is its own process with its own
    environment (for example its own ``SIRCL_PEER_ROUTES``). Writes go to the
    shared mapping.
    """

    def __init__(self, base: dict[str, str]) -> None:
        self._base = dict(base)
        self._local = threading.local()

    def override(self, values: dict[str, str]) -> None:
        self._local.values = dict(values)

    def _view(self) -> dict[str, str]:
        merged = dict(self._base)
        merged.update(getattr(self._local, "values", {}))
        return merged

    def __getitem__(self, key: str) -> str:
        return self._view()[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._base[key] = value
        getattr(self._local, "values", {}).pop(key, None)

    def __delitem__(self, key: str) -> None:
        self._base.pop(key, None)
        getattr(self._local, "values", {}).pop(key, None)

    def __iter__(self):
        return iter(self._view())

    def __len__(self) -> int:
        return len(self._view())

    def copy(self) -> dict[str, str]:
        return self._view()


def environ_snapshot() -> dict[str, str]:
    return dict(os.environ)
