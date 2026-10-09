"""vLLM general plugin: cheaper decode context parallel (DCP) collectives for GLM-5.3 DSA decode on SIRCL.

Status: research-only. CPU tests cover the settings, the pins, the code edits,
the helpers' paths for every flag combination, the wire format of the packed
all-to-all and the refusal paths (``tests/``); the GPU checks
(``tests/gpu_checks.py``) compare the Triton kernels with the image's kernels
and run the packed all-to-all on an emulated DCP group of SIRCL sessions. On
the hardware ring, audit mode (below) is the qualification step.

Scope. GLM-5.3 (``glm_moe_dsa``) on serving image 816c6d6a7e96 (the vLLM
``0.1.dev21553+gab86b7073`` wheel with the Python sources of vLLM bc9ea774 in
its CSF-sources layer, b12x 1.5.0 at B12X cc36aa6f) with B12X sparse MLA,
``--decode-context-parallel-size`` above 1 and the DCP group's collectives on a
SIRCL DCP session of the pinned SIRCL release (``SIRCL_VERSION``;
``sparkring_sircl``, SIRCL's vLLM adapter with ``SIRCL_GROUPS`` including
``dcp``). Every item acts only on decode-only batches
(CUDA-graph capture or eager) and leaves prefill and mixed batches to the image.

Items, each behind its own flag (``0`` or unset: off; ``1``: on; anything
else refuses at startup):

* ``GLM_DCP_DECODE_QUERY_PACK`` (exact). One Triton kernel builds the DCP
  query ``[ql_nope | RoPE(q_pe)]`` right after the ``W_UK`` absorption with the
  statements of the image's ``fused_q`` (``kernels.rope_cat``), replacing the
  ``torch.cat`` before the query all-gather on every layer and the whole
  ``fused_q`` launch on layers without the DSA indexer; the query is gathered
  straight into the attention workspace's query buffer, so the attention's
  copy of it is skipped (``forward_mqa`` recognizes the exact alias).
* ``GLM_DCP_DECODE_OVERLAP`` (exact; implies the query pack). Every collective
  of the DCP session runs on a dedicated communication stream, joined after
  each call, and on DSA indexer layers the query all-gather is issued right
  after ``W_UK`` and joined only before the attention, so it runs beside the
  indexer's ``wq_b`` GEMM and top-k (on other layers beside the KV-cache
  write).
* ``GLM_DCP_DECODE_WK_OVERLAP`` (exact). On calls that run the DSA indexer
  the indexer's ``wk`` GEMM runs on a side stream beside the latent
  ``q_a``/``kv_a`` projection and its all-gather.
* ``GLM_DCP_DECODE_SELECTION_REUSE`` (exact). The DCP index selection (the
  conversion of the shared top-k to this rank's physical slots) is computed
  once per DSA indexer layer and reused by the layers that follow it, which
  read the same top-k.
* ``GLM_DCP_DECODE_A2A_FUSED`` (exact). The all-to-all combine's pack runs
  inside the scatter kernel's staging (``_scatter_pack_cute.py``, one scatter
  op on the DCP session) and the combine reads the own share from the attention
  output (``kernels.wire_combine``): one Triton kernel fewer per layer. It
  applies where the image's combine is one scatter op (the message within the
  DCP group's relay-safe op limit); larger messages keep the image's combine.

Every fork onto the communication or wk stream is joined into the forking
stream before ``forward`` returns, on every path: by the statement that
consumes it, or by the edited ``return`` of ``forward`` (``finish``), so a
CUDA graph capture never ends with an unjoined stream. Whether a call runs
the indexer is read per call (``self.indexer is not None and not
self.skip_topk``), because the MTP speculator switches ``skip_topk`` on the
draft layer between draft steps.

Other settings:

* ``GLM_DCP_DECODE_COMM_PRIORITY`` [-1]: CUDA priority of the communication
  stream, -5 to 0 (lower runs first).
* ``GLM_DCP_DECODE_AUDIT`` [0] (qualification mode; needs an item flag). The
  model runs the exact paths and every decode forward, eager or captured,
  also runs the image's computation of each value an exact item replaces and
  counts differing words on the device. Logged as ``glm_dcp_decode_comm audit
  ...`` lines from an eager forward (at most every 10 s) and at exit. Eager
  calls add one reference query all-gather and, with the fused combine, one
  reference all-to-all per layer (not for timing). Inside a CUDA graph capture
  audit mode adds only comparisons on the capturing stream (``query``, ``wk``,
  ``selection``): no collective and no side-stream work.
* ``GLM_DCP_DECODE_QUERY_FP8``, ``GLM_DCP_DECODE_A2A_FP8``,
  ``GLM_DCP_DECODE_RESEARCH_LOSSY``, ``GLM_DCP_DECODE_QUERY_FP8_TILE``:
  unsupported (lossy FP8 payloads that this build does not carry); any value
  other than ``0`` or unset refuses at startup.

Mechanism. ``register`` requires the SHA-256 of every file whose behavior the
plugin relies on (``FILE_CHECKS``: image files and SIRCL files, digests
taken with CRLF line endings read as LF) and of the package's other modules
(``PACKAGE_SHA256``, so a launcher pin of this file pins the whole package)
to equal the recorded values, applies exact edits that keep the line count to
the source of ``DeepseekV32Attention.forward`` and
``._sparse_indexer_and_attn`` (``vllm/models/deepseek_v32/attention.py``),
compiles original and edited source at their original line numbers, requires
the compiled original to equal the loaded code, places helper functions in the
module namespace and installs the edited code as ``__code__`` of the existing
functions. vLLM turns breakable CUDA graphs on by default for
``GlmMoeDsaForCausalLM`` (``VLLM_USE_BREAKABLE_CUDAGRAPH=1``,
``config/vllm.py``), and then ``_sparse_indexer_and_attn`` is the closure of
``eager_break_during_capture``; the edit goes into the function that closure
calls, recognized by the closure's code object, and the closure stays in
place. Any other wrapper refuses. With the overlap it wraps the collective
methods of SIRCL's communicator class for the communication stream. Modules
not imported yet are patched right after their first import. Any mismatch
raises ``PatchRefused`` at startup. No file is written.

The session it was built against. At a DCP layer's first decode call the
worker requires the DCP group's device communicator to be SIRCL's
``SirclCudaCommunicator`` holding a ``SirclDcpCollectives`` session, every one
of them loaded from the verified SIRCL files, and raises
``PatchRefused`` otherwise (``runtime._session_check``). ``tools/refresh_pins.py``
records the pins of another vLLM, b12x or SIRCL tree; a changed file is a
different build and needs a new qualification.

Every helper runs the image's statement unless its item applies to the
current layer call; with every flag off the edited functions make the image's
calls in the image's order.

Coexistence. Another plugin that edits ``DeepseekV32Attention.forward`` or
``_sparse_indexer_and_attn`` makes the loaded code differ from the verified
source, and ``register`` refuses; plugins that wrap the B12X implementation's
methods or edit other methods are untouched. ``glm53full_speedups`` edits
``DeepseekV32Attention.__init__`` of the same module on its first import:
the two import hooks each wrap that module's loader once, in either order on
``sys.meta_path``.
"""

from __future__ import annotations

import __future__
import ast
import hashlib
import importlib.abc
import importlib.util
import logging
import os
import re
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

PLUGIN_NAME = "glm_dcp_decode_comm"
PLUGIN_VERSION = "2.0.0"
MARKER = "__glm_dcp_decode_comm__"
HELPER_PREFIX = "_glm_dcp_decode_comm_"
IMAGE = "816c6d6a7e96"
IMAGE_VLLM_VERSION = "0.1.dev21553+gab86b7073 with the vLLM bc9ea774 Python sources"
SIRCL_PACKAGE = "sparkring_sircl"
# The SIRCL build the sparkring_sircl pins below record: the package at this version, built from SparkRing's
# spark_transport/sircl/sparkring_sircl and installed by the serving image as dist-packages/sparkring_sircl
# (README, "Pins", names the build). tools/refresh_pins.py rewrites the pins and this value together.
SIRCL_VERSION = "0.3.1"

FLAGS = {
    "overlap": "GLM_DCP_DECODE_OVERLAP",
    "wk_overlap": "GLM_DCP_DECODE_WK_OVERLAP",
    "query_pack": "GLM_DCP_DECODE_QUERY_PACK",
    "selection_reuse": "GLM_DCP_DECODE_SELECTION_REUSE",
    "a2a_fused": "GLM_DCP_DECODE_A2A_FUSED",
}
COMM_PRIORITY = "GLM_DCP_DECODE_COMM_PRIORITY"
AUDIT = "GLM_DCP_DECODE_AUDIT"
UNSUPPORTED_SETTINGS = ("GLM_DCP_DECODE_QUERY_FP8", "GLM_DCP_DECODE_A2A_FP8", "GLM_DCP_DECODE_RESEARCH_LOSSY",
                        "GLM_DCP_DECODE_QUERY_FP8_TILE")

ATTENTION_MODULE = "vllm.models.deepseek_v32.attention"
COMMUNICATOR_MODULE = "sparkring_sircl.vllm.communicator"
COMMUNICATOR_CLASS = "SirclCudaCommunicator"

_LOG = logging.getLogger("vllm." + PLUGIN_NAME)


class PatchRefused(RuntimeError):
    """The loaded code or session differs from the sources this plugin was written against."""


@dataclass(frozen=True)
class Edit:
    """Replace ``old`` with ``new`` exactly once inside one function's source."""

    old: str
    new: str


@dataclass(frozen=True)
class CodePatch:
    module: str
    path: str  # relative to the vllm package directory
    sha256: str
    qualname: str
    edits: tuple[Edit, ...]
    decorators: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileCheck:
    package: str  # vllm, b12x or sparkring_sircl
    path: str
    sha256: str
    reason: str


# SHA-256 of the package's other modules; tools/refresh_pins.py rewrites them.
PACKAGE_SHA256 = {
    "_scatter_pack_cute.py": "f8837cbb5c3518b938f49be8b69b796e2527b927d475cd20d2716d5d1f2b36e1",
    "kernels.py": "b6b518fba57b90e33b722d860752ae43f818a8ed71e874eb08dcb36c73b1d1f8",
    "layout.py": "e091fe3d1de5bda5b9ec42d5e77c0b9f1adcdf9fa5fe956a03cdc6e18ff6bb7a",
    "reference.py": "dd1b3a1ec2270ec11e43986014f74791aaf6f8ec1de94fba3be443750626840f",
    "runtime.py": "2e000c24a60d001df35c045dda1ae1cba76fb0df1bbd4331db43ecc9f1359818",
}

ATTENTION_PATH = "models/deepseek_v32/attention.py"
ATTENTION_SHA256 = "c1358e677002658c848247758c45ef5543eec9c5122bc3b36e2e5c4447c68bd6"

H = HELPER_PREFIX
PATCHES = (
    CodePatch(
        ATTENTION_MODULE, ATTENTION_PATH, ATTENTION_SHA256, "DeepseekV32Attention.forward",
        (
            Edit("        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]\n",
                 f"        {H}wk_fork(self, hidden_states); qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]\n"),
            Edit("            kw = self.indexer.wk_weights_proj(hidden_states)[0]\n",
                 f"            kw = {H}wk_join(self, hidden_states)\n"),
            Edit("        ql_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)\n",
                 "        ql_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1); "
                 f"{H}query(self, positions, q_pe, ql_nope)\n"),
            Edit("        index_q_fp8, index_weights_out, mqa_q = fused_q(\n",
                 f"        index_q_fp8, index_weights_out, mqa_q = {H}fused_q(self, fused_q)(\n"),
            Edit("        return self.o_proj(output)[0]\n",
                 f"        return {H}finish(self, self.o_proj(output)[0])\n"),
        ),
    ),
    CodePatch(
        ATTENTION_MODULE, ATTENTION_PATH, ATTENTION_SHA256, "DeepseekV32Attention._sparse_indexer_and_attn",
        (
            # The DCP query gather and combine of the whole-batch route (the image's branch for batches that
            # take neither the full-CKV nor the split-CKV route), at its indentation.
            Edit("                if isinstance(mqa_q_arg, tuple):\n"
                 "                    mqa_q_arg = torch.cat(mqa_q_arg, dim=-1)\n"
                 "                assert self.dcp_manager.query_gather is not None\n"
                 "                mqa_q_arg = self.dcp_manager.query_gather(mqa_q_arg)\n",
                 f"                if isinstance(mqa_q_arg, tuple) and not {H}has_query(self, mqa_q_arg):\n"
                 "                    mqa_q_arg = torch.cat(mqa_q_arg, dim=-1)\n"
                 "                assert self.dcp_manager.query_gather is not None\n"
                 f"                mqa_q_arg = {H}gather(self, mqa_q_arg)\n"),
            Edit("                    attn_out = self.dcp_manager.combine(\n",
                 f"                    attn_out = {H}combine(self)(\n"),
        ),
        decorators=("eager_break_during_capture",),
    ),
)
HELPER_NAMES = tuple(f"{H}{name}" for name in
                     ("wk_fork", "wk_join", "query", "fused_q", "has_query", "gather", "combine", "finish"))

FILE_CHECKS: tuple[FileCheck, ...] = (
    # vLLM and b12x of serving image 816c6d6a7e96.
    FileCheck("vllm", "models/deepseek_v32/common/kernels.py",
              "c33fec19f157eb019c088f6c93f4fc444f6fc9ff24890f5684054e85f16389bd",
              "fused_q runs _fused_q_kernel on the BF16 query path, whose program 0 is the RoPE that "
              "kernels.rope_cat restates, and fused_q's MLA output is only the RoPE'd q_pe"),
    FileCheck("vllm", "models/deepseek_v32/nvidia/ops/fused_q_cutedsl.py",
              "548992905e056e4021e052609664f422c753f24a3eaaac46963957442c2ba808",
              "is_fused_q_cutedsl_supported needs quantize_mqa, so the BF16 query path runs _fused_q_kernel"),
    FileCheck("vllm", "v1/attention/ops/dcp.py",
              "5cc592b1baa5defd47336d19be4002fe0f98827134ebbd755ef9364350456a53",
              "_dcp_a2a_pack_send_kernel packs [B, H, 514] records with the FP32 LSE in two BF16 "
              "slots and _dcp_a2a_unpack_combine_kernel is the combine kernels.wire_combine restates; "
              "MLADCPManager binds combine as a partial of dcp_a2a_lse_reduce with is_lse_base_on_e and "
              "its query gather through the impl's gather_dcp_query"),
    FileCheck("vllm", "v1/attention/backends/mla/b12x_mla_sparse.py",
              "28ba243901a980b3c0ea75fd03c16d589e352dd8727fe6f7d7c879f71daf74c8",
              "forward_mqa skips its query copy when the query is an exact alias of the first "
              "input_num_heads heads of its workspace query buffer, which holds _kernel_num_heads heads "
              "(input heads rounded up to eight), consults _physical_selection_provider with the whole "
              "decode batch before triton_filter_and_convert_dcp_index, and returns the decode output and "
              "natural-log LSE of the input heads; _borrow_workspaces(input_num_heads=_kernel_num_heads) "
              "gives forward_mqa's views; decode-only batches take neither the full-CKV nor the split-CKV "
              "route; gather_dcp_query all-gathers GLM-5.3 decode queries along dim 1"),
    FileCheck("vllm", "v1/attention/backends/mla/b12x_indexer.py",
              "e1841ab03516e9f58e52400491ed5a11db7826c4a676f49c427a9e6246210ae2",
              "the DSA indexer's decode borrows the same workspace for its top-k scratch and all-gathers "
              "the DCP candidates through the dcp group; its key gather (VLLM_DCP_INDEXER_KEY_GATHER) runs "
              "in eager prefill only"),
    FileCheck("vllm", "v1/attention/backends/mla/sparse_utils.py",
              "615a849ef7673f6661fe06f798f9c9a39afe4f86ca266e42701b9ccd4e708e19",
              "triton_filter_and_convert_dcp_index is a pure function of its arguments for 2048-wide "
              "selections (one owner per row)"),
    FileCheck("vllm", "v1/worker/workspace.py",
              "18334e399b5d7749a0733d64fe6214906d0f9d276fc7c65806603de4c912449c",
              "get_simultaneous returns views at offsets fixed by the requested specs"),
    FileCheck("vllm", "v1/worker/gpu/cudagraph_utils.py",
              "dfa39d76f6c4420a514ff4748a73ac95d3b0f94f20b0bc97477769763ab3d761",
              "FULL graphs are captured by torch.cuda.graph around one eager forward, after an eager "
              "warm-up forward of the same batch, and may fork work onto auxiliary streams"),
    FileCheck("vllm", "compilation/breakable_cudagraph.py",
              "a7b53a403c776df8195881c98fdba72e0e2662c15adc047cd5c79736247d19e2",
              "eager_break_during_capture returns the function unchanged unless "
              "VLLM_USE_BREAKABLE_CUDAGRAPH=1"),
    FileCheck("vllm", "model_executor/layers/attention/mla_attention.py",
              "601f055662f71cf545fc3812dea36169e1002cce45f753ec674d7aa572e027d4",
              "MLAAttention builds its MLADCPManager with the impl's gather_dcp_query and keeps "
              "W_UK_T, num_heads, kv_lora_rank and qk_rope_head_dim"),
    FileCheck("vllm", "distributed/parallel_state.py",
              "3aba1b4376c828a343d781d23fd1dd0c4bdb40b8dafa06a927006e7596359212",
              "every all-gather, reduce-scatter and all-to-all of the dcp group reaches its device "
              "communicator"),
    FileCheck("vllm", "models/deepseek_v32/nvidia/b12x.py",
              "2412da9bec4b3597064395c4c78b0d93c1235153fe3e55df4b9dba6dcde4942d",
              "DeepseekV32B12xAttention inherits forward and _sparse_indexer_and_attn"),
    FileCheck("b12x", "attention/_shared/mla/kernel.py",
              "d4916b11ac485aa327cfa32662427ddc7a09929c09f39f07ae6974d5ad28d63c",
              "the unified decode stages an NVFP4-cache query as BF16 and returns the workspace "
              "output buffer and final LSE"),
    # SIRCL at SIRCL_VERSION.
    FileCheck("sparkring_sircl", "__init__.py",
              "db3b3e9e101f8caa13f6bbf049e36a35e939e219930ab2add6b41ce503a54793",
              "the package version SIRCL_VERSION"),
    FileCheck("sparkring_sircl", "protocol.py",
              "6ace052d19d026a4767c1accbb93b4ec33d258b22a595be468d948ea63b2d222",
              "the command-ring words (Ctrl), op code 3 (Op.SCATTER, OP_SHIFT), grid_blocks and the "
              "counter layout the packed all-to-all writes and reads"),
    FileCheck("sparkring_sircl", "scatter_plan.py",
              "bebd8e7117ae5ec462b1871a12ebde30eadbebf89a45dc4bf645fa076f2b9411",
              "the scatter piece rules that decide when an all-to-all is one op"),
    FileCheck("sparkring_sircl", "oneshot/runtime.py",
              "7e85af0d5313b85bf4ec11e4e8f60935e9232b4ffdb084d6726d32d40cad381d",
              "RoceOneshotAllReduce: all_gather(out=), the arena view, _counter_addresses, _order_stream, "
              "_mark_stream, _tuned_op, _lock, check_health and the launch geometry"),
    FileCheck("sparkring_sircl", "oneshot/_scatter_ops.py",
              "a06b0f593c3f11e25c2aaddceadc8a075552256b6d0448b1ea55085400018d6e",
              "the scatter launch protocol and op_bytes / piece_bytes, which the packed all-to-all follows"),
    FileCheck("sparkring_sircl", "oneshot/_scatter_cute.py",
              "ec1b3e9d71347ef04f267869ff8aa4a02a1f2001627de02b8eb7cb5b43fc8c9d",
              "the scatter kernel whose wire protocol _scatter_pack_cute.py restates"),
    FileCheck("sparkring_sircl", "oneshot/_roce_proxy.c",
              "09db871538819191838bc17d6aafe8bc436c2af505263b609851f6062d9f025b",
              "the progress thread's op code 3: chunk p to peer p on p's lanes, no later phase"),
    FileCheck("sparkring_sircl", "oneshot/_compile.py",
              "6b2378acc34032e19b07159a4a1d3fc034f4cea898dbd40577b0bb42891839b1",
              "compile_launcher, current_cuda_stream, make_pointer and the kernel-resolution freeze"),
    FileCheck("sparkring_sircl", "oneshot/_cute_intrinsics.py",
              "6b290461f0b382dd4fc5b2f696afad40bd66f571d59f0ab22d2a5d2a48d06c1f",
              "the device intrinsics the packed all-to-all kernel uses"),
    FileCheck("sparkring_sircl", "oneshot/_cute_batch.py",
              "62cf464dc61fe3b54afe6075965d6aa5f4ac1bf7986f0fd22d12f74d2f41c989",
              "ld_v4_u32_batch, the batched peer loads of the copy phase"),
    FileCheck("sparkring_sircl", "oneshot/_timed_wait.py",
              "fff40032371ddfec7c92ed561ffccae65fac4a588e917e69c59640a4a7298ce5",
              "spin_until_eq_timed_sys, the flag wait with the session's wait limit"),
    FileCheck("sparkring_sircl", "vllm/communicator.py",
              "282e93371b9358f1bce0ace9d839fb5bf78ce570544bbe4cae8ecdebb3c6c197",
              "SirclCudaCommunicator: its group adapter in .sircl and the collective methods the "
              "communication stream wraps; the dcp_all_to_all and dcp_b12x_transport shims installed for a "
              "DCP group with a session"),
    FileCheck("sparkring_sircl", "vllm/adapter.py",
              "5c686d3d22da72f4e9a4703b5a8ff64e0069851aab37ca6ad17124c9b176276c",
              "GroupAdapter keeps a DCP group's SirclDcpCollectives in .dcp and its runtime in .session; "
              "live_adapters"),
    FileCheck("sparkring_sircl", "vllm/dcp_collectives.py",
              "06b4cf32fa4c1fee5d61e89018f60879e81ddbe5e5db6de653d4be2101f1484e",
              "SirclDcpCollectives keeps _runtime, gather_chunk, gather_max and _scatter_op_limit, and gathers "
              "a decode shard as one last-dimension runtime all-gather of the [rows, inner] view"),
    FileCheck("sparkring_sircl", "vllm/executor.py",
              "7077498840f0339e69c42d904c3b27aee90555c52903ee066dd9a768697867b0",
              "a DCP all-to-all within the scatter op limit is one session all_to_all"),
    FileCheck("sparkring_sircl", "vllm/planner.py",
              "5e5f275251ff12be1577625bb27430cb1efa8a10fa39d2f230c8881a900da837",
              "the all-to-all plan's op count from the scatter op limit"),
    FileCheck("sparkring_sircl", "vllm/shims.py",
              "5a6480862bf87a4a89480b7f3b4e26ba48c9810b355e35a1eb0bba159f105f78",
              "dcp_all_to_all runs vLLM's pack and unpack-combine around the communicator's "
              "all_to_all_single and marks itself with _sircl_original; dcp_b12x_transport returns no B12X "
              "PCIe transport for a SIRCL group"),
)


@dataclass(frozen=True)
class Prepared:
    patch: CodePatch
    path: Path
    original: CodeType
    edited: CodeType


# --------------------------------------------------------------------------- settings


def _flag(env, name: str) -> bool:
    value = env.get(name, "0").strip() or "0"
    if value not in ("0", "1"):
        raise PatchRefused(f"{PLUGIN_NAME}: {name} must be 0 or 1, got {value!r}")
    return value == "1"


def settings_from_env(env=os.environ) -> dict:
    """Parse and check every flag; a malformed, inconsistent or unsupported setting refuses."""
    unsupported = [f"{name}={env.get(name)!r}" for name in UNSUPPORTED_SETTINGS
                   if (env.get(name, "") or "").strip() not in ("", "0")]
    if unsupported:
        raise PatchRefused(f"{PLUGIN_NAME}: {', '.join(unsupported)}: the lossy FP8 payloads are not carried by "
                           f"{PLUGIN_NAME} {PLUGIN_VERSION}; unset them")
    values: dict = {item: _flag(env, name) for item, name in FLAGS.items()}
    priority_text = env.get(COMM_PRIORITY, "-1").strip() or "-1"
    try:
        priority = int(priority_text)
    except ValueError:
        priority = 1
    if not -5 <= priority <= 0:
        raise PatchRefused(f"{PLUGIN_NAME}: {COMM_PRIORITY} must be a whole number from -5 to 0, "
                           f"got {priority_text!r}")
    audit = _flag(env, AUDIT)
    if audit and not any(values.values()):
        raise PatchRefused(f"{PLUGIN_NAME}: {AUDIT}=1 audits the items; set at least one item flag")
    values["comm_priority"] = priority
    values["audit"] = audit
    return values


def enabled(values: dict) -> bool:
    return any(values[item] for item in FLAGS)


# --------------------------------------------------------------------------- pins


def digest(path: Path) -> str:
    """SHA-256 of a file's bytes with CRLF line endings read as LF (a checkout's line endings are no build)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_package(directory: Path | None = None) -> None:
    """Require the package's other modules to be the ones whose SHA-256 this file records."""
    directory = directory or Path(__file__).resolve().parent
    for name, expected in PACKAGE_SHA256.items():
        path = directory / name
        try:
            found = digest(path)
        except FileNotFoundError:
            raise PatchRefused(f"{PLUGIN_NAME}: {path} is missing") from None
        if found != expected:
            raise PatchRefused(f"{PLUGIN_NAME}: {path} has SHA-256 {found}, expected {expected} (the package "
                               "changed after its pins were recorded)")


def verify_file(check: FileCheck, roots: dict[str, Path]) -> None:
    path = roots[check.package] / check.path
    try:
        found = digest(path)
    except FileNotFoundError:
        raise PatchRefused(f"{PLUGIN_NAME}: {path} is missing; the plugin relies on {check.reason}") from None
    if found != check.sha256:
        raise PatchRefused(f"{PLUGIN_NAME}: {path} has SHA-256 {found}, expected {check.sha256}; "
                           f"the plugin relies on {check.reason}")


def sircl_version(root: Path) -> str | None:
    """The ``__version__`` that ``root/__init__.py`` assigns, read without importing the package."""
    try:
        text = (Path(root) / "__init__.py").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


# --------------------------------------------------------------------------- patching


def _future_flags(tree: ast.Module) -> int:
    flags = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                flags |= getattr(__future__, alias.name).compiler_flag
    return flags


def _locate(tree: ast.Module, qualname: str, decorators: tuple[str, ...]):
    class_name, name = qualname.split(".")
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(classes) != 1:
        raise PatchRefused(f"{PLUGIN_NAME}: class {class_name} not found once")
    owner = classes[0]
    functions = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(functions) != 1:
        raise PatchRefused(f"{PLUGIN_NAME}: method {qualname} not found once")
    node = functions[0]
    found = tuple(ast.unparse(d) for d in node.decorator_list)
    if found != decorators:
        raise PatchRefused(f"{PLUGIN_NAME}: {qualname} has decorators {found}, expected {decorators}")
    return node, owner


def _find_code(code: CodeType, qualname: str) -> CodeType | None:
    for const in code.co_consts:
        if isinstance(const, CodeType):
            if const.co_qualname == qualname:
                return const
            found = _find_code(const, qualname)
            if found is not None:
                return found
    return None


def _module_imports(tree: ast.Module) -> str:
    """Module-scope imports (not ``__future__``): CPython 3.12 compiles ``name.attr(...)`` by binding."""
    found: list[str] = []

    def visit(body: Iterable[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                    found.append(ast.unparse(node))
            elif isinstance(node, (ast.If, ast.Try, ast.TryStar, ast.With)):
                for attr in ("body", "orelse", "finalbody"):
                    visit(getattr(node, attr, None) or [])
                for handler in getattr(node, "handlers", None) or []:
                    visit(handler.body)

    visit(tree.body)
    return "\n".join(found)


def _compile_method(lines: str, first: int, owner: ast.ClassDef, filename: str, flags: int, qualname: str,
                    imports: str) -> CodeType:
    """Compile one method's source lines (decorators included) at their original line numbers."""
    source = "\n" * (first - 2) + f"class {owner.name}:\n" + lines
    if not source.endswith("\n"):
        source += "\n"
    source += imports + "\n"
    code = _find_code(compile(source, filename, "exec", flags=flags, dont_inherit=True), qualname)
    if code is None:
        raise PatchRefused(f"{PLUGIN_NAME}: compiled code of {qualname} not found")
    return code


def prepare(patch: CodePatch, vllm_root: Path) -> Prepared:
    """Verify the source file and compile the original and the edited method."""
    path = vllm_root / patch.path
    data = path.read_bytes()
    found = digest(path)
    if found != patch.sha256:
        raise PatchRefused(f"{PLUGIN_NAME}: {path} has SHA-256 {found}, expected {patch.sha256} "
                           f"(vLLM {IMAGE_VLLM_VERSION} of image {IMAGE}); refusing to patch {patch.qualname}")
    text = data.decode("utf-8")
    tree = ast.parse(text, filename=str(path))
    node, owner = _locate(tree, patch.qualname, patch.decorators)
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = "".join(text.splitlines(keepends=True)[first - 1: node.end_lineno])
    edited = lines
    for edit in patch.edits:
        count = edited.count(edit.old)
        if count != 1:
            raise PatchRefused(f"{PLUGIN_NAME}: {patch.qualname} contains {edit.old!r} {count} times, "
                               "expected once")
        edited = edited.replace(edit.old, edit.new)
    if edited.count("\n") != lines.count("\n"):
        raise PatchRefused(f"{PLUGIN_NAME}: an edit of {patch.qualname} changes its line count")
    flags = _future_flags(tree)
    imports = _module_imports(tree)
    return Prepared(
        patch=patch, path=path,
        original=_compile_method(lines, first, owner, str(path), flags, patch.qualname, imports),
        edited=_compile_method(edited, first, owner, str(path), flags, patch.qualname, imports),
    )


def _eager_break_wrapper_code(module: ModuleType) -> CodeType | None:
    """Code of the wrapper that vLLM's ``eager_break_during_capture`` returns, or None.

    The decorator returns the method itself unless breakable CUDA graphs are
    on, and then a closure over the method (``compilation/breakable_cudagraph.py``,
    pinned in ``FILE_CHECKS``) that calls it.
    """
    decorator = module.__dict__.get("eager_break_during_capture")
    if type(decorator) is not FunctionType:
        return None
    for const in decorator.__code__.co_consts:
        if isinstance(const, CodeType) and const.co_name == "wrapper":
            return const
    return None


def _resolve(module: ModuleType, qualname: str) -> tuple[FunctionType, FunctionType]:
    """The class attribute of ``qualname`` and the function that runs its body.

    Both are the same plain function, or, when vLLM's
    ``eager_break_during_capture`` wrapped the method, the wrapper and the
    function it closes over and calls; the edit goes into the latter, so the
    wrapper keeps its breakable-graph behavior. Any other wrapper refuses.
    """
    class_name, name = qualname.split(".")
    attribute = vars(vars(module)[class_name]).get(name)
    if type(attribute) is not FunctionType:
        raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__}.{qualname} is not a plain function")
    inner = getattr(attribute, "__wrapped__", None)
    if inner is None:
        return attribute, attribute
    wrapper_code = _eager_break_wrapper_code(module)
    closed_over = [cell.cell_contents for cell in attribute.__closure__ or ()]
    if wrapper_code is None or attribute.__code__ is not wrapper_code or not any(c is inner for c in closed_over):
        code = attribute.__code__
        raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__}.{qualname} is wrapped by {code.co_qualname} "
                           f"({code.co_filename}:{code.co_firstlineno}), not by vLLM's "
                           "eager_break_during_capture; refusing to patch")
    if type(inner) is not FunctionType or getattr(inner, "__wrapped__", None) is not None:
        raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__}.{qualname}: eager_break_during_capture wraps "
                           "another wrapper; refusing to patch")
    return attribute, inner


def install(module: ModuleType, prepared: tuple[Prepared, ...]) -> bool:
    """Place the helpers and install every edited method; False if already installed."""
    loaded_from = Path(getattr(module, "__file__", "") or "")
    changed = False
    for item in prepared:
        if loaded_from.resolve() != item.path.resolve():
            raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__} was loaded from {loaded_from}, not from the "
                               f"verified file {item.path}")
        _, function = _resolve(module, item.patch.qualname)
        current = function.__code__
        if getattr(function, MARKER, None) == item.patch.qualname and current == item.edited.replace(
                co_filename=current.co_filename):
            continue
        if current != item.original or current.co_qualname != item.original.co_qualname:
            raise PatchRefused(f"{PLUGIN_NAME}: loaded {module.__name__}.{item.patch.qualname} differs from the "
                               "verified source (another patch or interpreter version); refusing to patch")
        changed = True
    if not changed:
        return False
    for name in HELPER_NAMES:
        helper = HELPERS[name]
        present = module.__dict__.get(name)
        if present is None:
            setattr(module, name, helper)
        elif present is not helper:
            raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__}.{name} already exists")
    for item in prepared:
        attribute, function = _resolve(module, item.patch.qualname)
        if getattr(function, MARKER, None) == item.patch.qualname:
            continue
        function.__code__ = item.edited.replace(co_filename=function.__code__.co_filename)
        setattr(function, MARKER, item.patch.qualname)
        _LOG.info("%s: %s.%s patched%s", PLUGIN_NAME, module.__name__, item.patch.qualname,
                  "" if attribute is function else " (inside vLLM's eager_break_during_capture wrapper)")
    return True


def install_comm_stream(module: ModuleType) -> bool:
    """Wrap SIRCL's communicator collectives for the DCP communication stream."""
    from . import runtime

    root = _ROOTS.get(SIRCL_PACKAGE)
    if root is not None:
        expected = root / "vllm" / "communicator.py"
        loaded = Path(getattr(module, "__file__", "") or "")
        if loaded.resolve() != expected.resolve():
            raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__} was loaded from {loaded}, not from the "
                               f"verified file {expected}")
    cls = getattr(module, COMMUNICATOR_CLASS, None)
    if cls is None:
        raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__}.{COMMUNICATOR_CLASS} not found")
    done = runtime.install_comm_stream(cls)
    if done:
        _LOG.info("%s: %s.%s collectives of DCP sessions run on the DCP communication stream", PLUGIN_NAME,
                  module.__name__, COMMUNICATOR_CLASS)
    return done


class PatchOnImport(importlib.abc.MetaPathFinder):
    """Run a callback right after the first import of each named module executes."""

    def __init__(self, callbacks: dict[str, Callable[[ModuleType], None]]):
        self._callbacks = dict(callbacks)
        self._lock = threading.Lock()
        self._resolving = threading.local()

    def pending(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._callbacks))

    def find_spec(self, fullname, path=None, target=None):
        with self._lock:
            callback = self._callbacks.get(fullname)
        # Another plugin's finder for the same module (glm53full_speedups patches the attention module too)
        # asks every finder, this one included. Answering None while this finder resolves the name ends that
        # recursion, so each finder wraps the module's loader once and each callback runs once.
        active = self._resolving.__dict__.setdefault("names", set())
        if callback is None or fullname in active:
            return None
        active.add(fullname)
        try:
            spec = None
            for finder in list(sys.meta_path):
                find_spec = getattr(finder, "find_spec", None)
                if finder is self or find_spec is None:
                    continue
                spec = find_spec(fullname, path, target)
                if spec is not None:
                    break
        finally:
            active.discard(fullname)
        if spec is None:
            return None
        loader = spec.loader
        exec_module = getattr(loader, "exec_module", None)
        if exec_module is None:
            raise ImportError(f"{PLUGIN_NAME}: cannot wrap the loader {loader!r} of {fullname}")

        def exec_then_patch(module, _exec=exec_module, _callback=callback, _name=fullname):
            _exec(module)
            _callback(module)
            self._finish(_name)

        loader.exec_module = exec_then_patch
        return spec

    def _finish(self, fullname: str) -> None:
        with self._lock:
            self._callbacks.pop(fullname, None)
            done = not self._callbacks
        if done and self in sys.meta_path:
            sys.meta_path.remove(self)


# --------------------------------------------------------------------------- helpers (placed in the attention module)

_SETTINGS: dict | None = None
_RUNTIME: ModuleType | None = None
_ROOTS: dict[str, Path] = {}


def _rt() -> ModuleType:
    global _RUNTIME
    if _RUNTIME is None:
        from . import runtime

        root = _ROOTS.get(SIRCL_PACKAGE)
        runtime.configure(runtime.Settings(
            overlap=_SETTINGS["overlap"], wk_overlap=_SETTINGS["wk_overlap"],
            query_pack=_SETTINGS["query_pack"], selection_reuse=_SETTINGS["selection_reuse"],
            a2a_fused=_SETTINGS["a2a_fused"], comm_priority=_SETTINGS["comm_priority"], audit=_SETTINGS["audit"],
            sircl_root=str(root) if root is not None else "", sircl_version=SIRCL_VERSION))
        _RUNTIME = runtime
    return _RUNTIME


def _on() -> bool:
    return bool(_SETTINGS) and enabled(_SETTINGS)


def wk_fork(layer, hidden_states) -> None:
    if _SETTINGS and _SETTINGS["wk_overlap"]:
        _rt().wk_fork(layer, hidden_states)


def wk_join(layer, hidden_states):
    if _SETTINGS and _SETTINGS["wk_overlap"]:
        return _rt().wk_join(layer, hidden_states)
    return layer.indexer.wk_weights_proj(hidden_states)[0]


def query(layer, positions, q_pe, ql_nope) -> None:
    if _on():
        runtime = _rt()
        runtime.install_provider(layer)
        runtime.query(layer, positions, q_pe, ql_nope)


def fused_q(layer, image_fused_q):
    if _on():
        return _rt().fused_q_fn(layer, image_fused_q)
    return image_fused_q


def has_query(layer, mqa_q_arg=None) -> bool:
    return _on() and _rt().has_query(layer, mqa_q_arg)


def gather(layer, mqa_q_arg):
    if _on():
        return _rt().gather(layer, mqa_q_arg)
    return layer.dcp_manager.query_gather(mqa_q_arg)


def combine(layer):
    if _on():
        return _rt().combine_fn(layer)
    return layer.dcp_manager.combine


def finish(layer, result):
    """``forward``'s return value, after joining any side-stream work the call left unconsumed."""
    if _on():
        _rt().finish(layer)
    return result


HELPERS: dict[str, Callable] = {
    f"{H}wk_fork": wk_fork,
    f"{H}wk_join": wk_join,
    f"{H}query": query,
    f"{H}fused_q": fused_q,
    f"{H}has_query": has_query,
    f"{H}gather": gather,
    f"{H}combine": combine,
    f"{H}finish": finish,
}


# --------------------------------------------------------------------------- registration

_STATE_LOCK = threading.Lock()
_FINDER: PatchOnImport | None = None
_REGISTERED: dict | bool | None = None


def _package_roots(packages: Iterable[str]) -> dict[str, Path]:
    roots = {}
    for name in packages:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise PatchRefused(f"{PLUGIN_NAME}: package {name} not found (the plugin needs vLLM and b12x of "
                               f"image {IMAGE} and SIRCL {SIRCL_VERSION} ({SIRCL_PACKAGE}) on the path)")
        roots[name] = Path(list(spec.submodule_search_locations)[0]).resolve()
    return roots


def verify(roots: dict[str, Path]) -> tuple[Prepared, ...]:
    """Every pin and the SIRCL version against ``roots``; the prepared code edits."""
    verify_package()
    version = sircl_version(roots[SIRCL_PACKAGE])
    if version != SIRCL_VERSION:
        raise PatchRefused(f"{PLUGIN_NAME}: {roots[SIRCL_PACKAGE]} is SIRCL {version}; this plugin was built "
                           f"against SIRCL {SIRCL_VERSION}")
    for check in FILE_CHECKS:
        verify_file(check, roots)
    return tuple(prepare(patch, roots["vllm"]) for patch in PATCHES)


def register() -> None:
    """vLLM general-plugin entry point; idempotent within a process."""
    global _FINDER, _REGISTERED, _SETTINGS
    values = settings_from_env()
    with _STATE_LOCK:
        if _REGISTERED is not None:
            if (values if enabled(values) else False) != _REGISTERED:
                raise PatchRefused(f"{PLUGIN_NAME}: settings changed after registration")
            return
        if not enabled(values):
            _REGISTERED = False
            _LOG.info("%s: no GLM_DCP_DECODE_* item flag is 1; nothing patched", PLUGIN_NAME)
            return
        roots = _package_roots(("vllm", "b12x", SIRCL_PACKAGE))
        prepared = verify(roots)
        _ROOTS.clear()
        _ROOTS.update(roots)
        _SETTINGS = values
        pending: dict[str, Callable[[ModuleType], None]] = {}
        module = sys.modules.get(ATTENTION_MODULE)
        if module is not None:
            install(module, prepared)
        else:
            pending[ATTENTION_MODULE] = lambda m, _p=prepared: install(m, _p)
        if values["overlap"]:
            comm_module = sys.modules.get(COMMUNICATOR_MODULE)
            if comm_module is not None:
                install_comm_stream(comm_module)
            else:
                pending[COMMUNICATOR_MODULE] = install_comm_stream
        if pending:
            _FINDER = PatchOnImport(pending)
            sys.meta_path.insert(0, _FINDER)
        _REGISTERED = values
    items = ", ".join(f"{FLAGS[k]}=1" for k in FLAGS if values[k]) + (f", {AUDIT}=1" if values["audit"] else "")
    _LOG.info("%s %s: enabled %s (communication stream priority %d) on SIRCL %s; %s", PLUGIN_NAME,
              PLUGIN_VERSION, items, values["comm_priority"], SIRCL_VERSION,
              "patched now" if not pending else f"patching on first import of {', '.join(sorted(pending))}")


def status() -> dict[str, object]:
    """Registration state and, in a worker that ran a decode layer, the runtime counters."""
    with _STATE_LOCK:
        out: dict[str, object] = {
            "version": PLUGIN_VERSION,
            "sircl_version": SIRCL_VERSION,
            "settings": dict(_SETTINGS) if _SETTINGS else None,
            "pending_modules": list(_FINDER.pending()) if _FINDER is not None else [],
        }
    if _RUNTIME is not None:
        out["runtime"] = _RUNTIME.status()
    return out


__all__ = [
    "FILE_CHECKS",
    "FLAGS",
    "HELPERS",
    "PACKAGE_SHA256",
    "PATCHES",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PatchRefused",
    "SIRCL_VERSION",
    "UNSUPPORTED_SETTINGS",
    "digest",
    "prepare",
    "register",
    "settings_from_env",
    "status",
    "verify",
    "verify_file",
    "verify_package",
]
