"""vLLM general plugin: GLM-5.3-NVFP4 decode speedups at tensor-parallel size 8.

Status: research-only. Offline CPU tests cover the patch mechanics, weight
loading and the gathered or reduced outputs (``tests/``); no item has run on
a GPU.

The serving image ``sparkring-dev/kraken:csf-sircl-libsircl-20261008``
(``816c6d6a7e96``; vLLM ``0.1.dev21553+gab86b7073`` with Python sources of
vLLM ``bc9ea774``, B12X sources ``cc36aa6f``) runs ``GlmMoeDsaForCausalLM``
through the CUDA DSA model in ``vllm/models/deepseek_v32``. At TP8 two of
its BF16 projections are
replicated on every rank. Each item below has its own environment flag (``1``
selects it, ``0`` or unset leaves the image's code untouched):

* ``GLM53FULL_LATENT_SHARD``: every MLA layer computes its query and key/value
  latents with one fused projection, ``fused_qkv_a_proj`` (``q_a_proj``
  2048 x 6144 stacked on ``kv_a_proj_with_mqa`` 576 x 6144, BF16, 32.2 MB),
  built with ``disable_tp=True`` (``vllm/models/deepseek_v32/attention.py``
  lines 298-303, class ``DeepSeekV2FusedQkvAProjLinear``). Every rank
  therefore streams the whole weight in each of the 78 target layers and in
  every MTP draft step. With the flag a TP8 process builds the projection
  column-parallel: rank ``r`` keeps fused rows ``[328 r, 328 r + 328)`` of
  ``[q_a ; kv_a]`` (rank 6 holds the last 80 query rows and the first 248
  key/value rows) and all-gathers the 328-column results along the last
  dimension (``ColumnParallelLinear.forward`` with ``gather_output=True``).
  Concatenating contiguous blocks in rank order reproduces the fused output's
  column order, so the split into 2048 query, 512 latent and 64 RoPE columns
  that follows is unchanged. The gather copies bytes exactly; only the GEMM's
  reduction order can differ. Online quantization that the deployment
  composes on top of the checkpoint (``--quantization-config``) applies to the
  328-row shard as it would to the full layer.

  Loading. The layer's loader copies, from each checkpoint tensor, only the
  rows that intersect the rank's block, as a view of that tensor. The b12x
  checkpoint loader (``--load-format b12x``) reads exactly the bytes of the
  view a weight loader copies (``b12x/loader/_checkpoint.py``,
  ``DirectWeightSession.__call__``), so it honours this split. A tensor that
  holds none of the rank's rows is a no-op that never inspects the parameter:
  with composed online quantization the layer is processed, and its weight
  replaced by a packed, empty parameter, as soon as the rank's rows are in,
  and a later ``load_weights`` call can hand that replacement to the loader.
* ``GLM53FULL_EH_PROJ_TP``: the MTP draft input projection ``eh_proj`` (BF16,
  12288 -> 6144, 151 MB) is a replicated ``nn.Linear``
  (``vllm/models/deepseek_v32/nvidia/mtp.py``), read in full by every rank in
  each draft step. With the flag a TP8 process builds it as
  ``RowParallelLinear(input_is_parallel=False)``: rank ``r`` keeps input
  columns ``[1536 r, 1536 (r + 1))`` (18.9 MB) and the tensor-parallel
  all-reduce sums the partial outputs. Only draft numerics change; rejection
  sampling keeps the target distribution.

Every item checks the tensor-parallel size when the patched code runs and keeps
the image's layer at any size other than 8.

Mechanism. For
each selected item the plugin requires the SHA-256 of the edited file and of
the files whose behavior the item relies on to equal the values recorded below.
It takes the function's source lines, applies one exact edit that keeps the
line count, compiles original and edited source at their original line numbers
with the module's ``__future__`` flags and imports, requires the compiled
original to equal the loaded code, places the helper in the module namespace
under a name starting with ``_glm53full_speedups_`` and installs the edited
code as ``__code__`` of the existing function. A module not imported yet is
patched right after its first import executes (a ``sys.meta_path`` finder).
Any mismatch raises ``PatchRefused`` at startup. No file is written.

Activation. ``register`` is the entry point ``glm53full_speedups`` of group
``vllm.general_plugins``; vLLM calls it in every process when ``VLLM_PLUGINS``
names it. This module imports only the standard library at import time.

Coexistence. ``glm53full_compat`` replaces ``DeepseekV32B12xAttention.__init__``
(a different class, which calls ``DeepseekV32Attention.__init__``). The
plugins touch disjoint functions and load in any order.
"""

from __future__ import annotations

import __future__
import ast
import hashlib
import importlib.abc
import logging
import os
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

PLUGIN_NAME = "glm53full_speedups"
PLUGIN_VERSION = "1.1.0"
TARGET_TP = 8
MARKER = "__glm53full_speedups__"
HELPER_PREFIX = "_glm53full_speedups_"
IMAGE = "816c6d6a7e96"
IMAGE_VLLM_SOURCES = "bc9ea774"
IMAGE_B12X_SOURCES = "cc36aa6f"

LATENT_FLAG = "GLM53FULL_LATENT_SHARD"
EH_PROJ_FLAG = "GLM53FULL_EH_PROJ_TP"
FLAGS = (LATENT_FLAG, EH_PROJ_FLAG)

_LOG = logging.getLogger("vllm." + PLUGIN_NAME)


class PatchRefused(RuntimeError):
    """The loaded vLLM differs from the source this plugin was written against."""


@dataclass(frozen=True)
class Edit:
    """Replace ``old`` with ``new`` exactly once inside one function's source."""

    old: str
    new: str


@dataclass(frozen=True)
class CodePatch:
    """Edit one function of one image source file for one flag."""

    flag: str
    module: str  # dotted module name
    path: str  # file path relative to the vllm package directory
    sha256: str  # SHA-256 of that file in image 816c6d6a7e96
    qualname: str  # module-level function or Class.method
    edits: tuple[Edit, ...]
    helpers: tuple[str, ...]


@dataclass(frozen=True)
class FileCheck:
    """A source file whose internals a flag's helper relies on."""

    flag: str
    package: str  # "vllm" or "b12x"
    path: str
    sha256: str
    reason: str


_ATTENTION = (
    "vllm.models.deepseek_v32.attention",
    "models/deepseek_v32/attention.py",
    "c1358e677002658c848247758c45ef5543eec9c5122bc3b36e2e5c4447c68bd6",
)
_MTP = (
    "vllm.models.deepseek_v32.nvidia.mtp",
    "models/deepseek_v32/nvidia/mtp.py",
    "103516ab5c596f9851172fdaf86c67cb5435d1ca2781fc80268d5d61dbcb3456",
)

LATENT_HELPER = HELPER_PREFIX + "qkv_a"
EH_PROJ_HELPER = HELPER_PREFIX + "eh_proj"

PATCHES: tuple[CodePatch, ...] = (
    CodePatch(LATENT_FLAG, *_ATTENTION, "DeepseekV32Attention.__init__", (
        Edit("        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(\n",
             f"        self.fused_qkv_a_proj = {LATENT_HELPER}(DeepSeekV2FusedQkvAProjLinear,\n"),
    ), (LATENT_HELPER,)),
    CodePatch(EH_PROJ_FLAG, *_MTP, "DeepseekV32MultiTokenPredictorLayer.__init__", (
        Edit("self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)",
             f"self.eh_proj = {EH_PROJ_HELPER}(config.hidden_size, prefix, nn.Linear)"),
    ), (EH_PROJ_HELPER,)),
)

FILE_CHECKS: tuple[FileCheck, ...] = (
    FileCheck(LATENT_FLAG, "vllm", "model_executor/layers/linear.py",
              "6dbc49ad5ff3445f3bb6102b0302139b615675583d7c7a03c8b35b54cfdf36d9",
              "MergedColumnParallelLinear partition sizes, ColumnParallelLinear.forward's "
              "gather_output all-gather, and the v1/v2 weight-loader hooks"),
    FileCheck(LATENT_FLAG, "vllm", "model_executor/parameter.py",
              "9eb09a01c877da62fce22a4b97e0628cdb651c1c4a9503537a007218a8c7f882",
              "copy_tensor_parallel_shard and ModelWeightParameter's output_dim"),
    FileCheck(LATENT_FLAG, "vllm", "model_executor/models/deepseek_v2.py",
              "112a2d92e4f80a2e9a3baa567571f0b22c491bf8c18a1dbba2e9259e9b36692a",
              "DeepSeekV2FusedQkvAProjLinear: a disable_tp MergedColumnParallelLinear whose "
              "min-latency GEMM applies only to BF16 [2112, 7168] on SM90/SM100"),
    FileCheck(LATENT_FLAG, "vllm", "models/deepseek_v32/nvidia/model.py",
              "03c85e81a225a0de3312bf62830ff836c8f0c9d894e7fee4381e5510f366cd10",
              "q_a_proj and kv_a_proj_with_mqa load as stacked shards 0 and 1 of "
              "fused_qkv_a_proj; no sequence parallelism at this launch"),
    FileCheck(LATENT_FLAG, "vllm", "distributed/device_communicators/cuda_communicator.py",
              "33d27cad52def83159c7f63a9b7bd47e61d3e1d50c453ea9927576fc808f3ad8",
              "all_gather offers each gather to b12x_ar_comm (sparknet) before NCCL"),
    FileCheck(LATENT_FLAG, "vllm", "model_executor/model_loader/reload/layerwise.py",
              "71f7022f85592ad8485b1856542fa8a23433315d6dfcbdddb01ead797575e122",
              "composed online quantization counts copied elements per layer, processes "
              "the layer once the rank's rows are complete, and replays deferred loader "
              "calls on the materialized parameter"),
    FileCheck(LATENT_FLAG, "b12x", "loader/_checkpoint.py",
              "6e1f57ba228dfccbcb0a0554bbd0ff66d42f362c1cb87c16bf8b8b8949881235",
              "DirectWeightSession.weights yields each checkpoint tensor as a meta tensor "
              "and DirectWeightSession.__call__ reads exactly the bytes of the view a "
              "weight loader copies, so the contiguous 328-row split loads unchanged "
              "under --load-format b12x"),
    FileCheck(EH_PROJ_FLAG, "vllm", "model_executor/layers/linear.py",
              "6dbc49ad5ff3445f3bb6102b0302139b615675583d7c7a03c8b35b54cfdf36d9",
              "RowParallelLinear(input_is_parallel=False, return_bias=False) and its "
              "input-dimension weight loader"),
    FileCheck(EH_PROJ_FLAG, "vllm", "models/deepseek_v32/nvidia/glm52_low_latency_gemm.py",
              "912e0d144520205c2606e62c5e1db64091a4c75b273bac8dccc6b54eaa8a262a",
              "build_glm52_plan returns None off SM103 and for the sharded [6144, 1536] "
              "weight, so mtp.py calls self.eh_proj(eh_input)"),
)


@dataclass(frozen=True)
class Prepared:
    patch: CodePatch
    path: Path
    original: CodeType
    edited: CodeType


# --------------------------------------------------------------------------- patching


def _future_flags(tree: ast.Module) -> int:
    flags = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                flags |= getattr(__future__, alias.name).compiler_flag
    return flags


def _locate(tree: ast.Module, qualname: str) -> tuple[ast.FunctionDef, ast.ClassDef | None]:
    parts = qualname.split(".")
    if len(parts) > 2:
        raise PatchRefused(f"{PLUGIN_NAME}: unsupported qualname {qualname}")
    owner = None
    scope: Iterable[ast.stmt] = tree.body
    if len(parts) == 2:
        classes = [n for n in scope if isinstance(n, ast.ClassDef) and n.name == parts[0]]
        if len(classes) != 1:
            raise PatchRefused(f"{PLUGIN_NAME}: class {parts[0]} not found once")
        owner = classes[0]
        scope = owner.body
    functions = [n for n in scope if isinstance(n, ast.FunctionDef) and n.name == parts[-1]]
    if len(functions) != 1 or functions[0].decorator_list:
        raise PatchRefused(f"{PLUGIN_NAME}: undecorated function {qualname} not found once")
    return functions[0], owner


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
    """Module-scope import statements (not ``__future__``), one per line.

    CPython 3.12 compiles ``name.attr(...)`` differently when ``name`` is bound
    by a module-level import; compiling the function with the module's imports
    keeps that choice identical to the loaded code.
    """
    found: list[str] = []

    def visit(body: Iterable[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                    found.append(ast.unparse(node))
            elif isinstance(node, (ast.If, ast.Try, ast.TryStar, ast.With)):
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(node, field, None) or [])
                for handler in getattr(node, "handlers", None) or []:
                    visit(handler.body)

    visit(tree.body)
    return "\n".join(found)


def _compile_function(lines: str, node: ast.FunctionDef, owner: ast.ClassDef | None,
                      filename: str, flags: int, qualname: str, imports: str) -> CodeType:
    """Compile one function's source lines at their original line numbers.

    A method is compiled inside a class statement of the same name on the line
    above its ``def``, so its qualified name, ``__class__`` cell, indentation
    and name mangling equal those of the loaded method.
    """
    if owner is None:
        source = "\n" * (node.lineno - 1) + lines
    else:
        source = "\n" * (node.lineno - 2) + f"class {owner.name}:\n" + lines
    if not source.endswith("\n"):
        source += "\n"
    source += imports + "\n"
    module_code = compile(source, filename, "exec", flags=flags, dont_inherit=True)
    code = _find_code(module_code, qualname)
    if code is None:
        raise PatchRefused(f"{PLUGIN_NAME}: compiled code of {qualname} not found")
    return code


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(patch: CodePatch, vllm_root: Path) -> Prepared:
    """Verify the source file and compile the original and edited function code."""
    path = vllm_root / patch.path
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != patch.sha256:
        raise PatchRefused(
            f"{PLUGIN_NAME}: {path} has SHA-256 {digest}, expected {patch.sha256} "
            f"(vLLM sources {IMAGE_VLLM_SOURCES} of image {IMAGE}); refusing to patch "
            f"{patch.qualname}"
        )
    text = data.decode("utf-8")
    tree = ast.parse(text, filename=str(path))
    node, owner = _locate(tree, patch.qualname)
    lines = "".join(text.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])
    edited = lines
    for edit in patch.edits:
        count = edited.count(edit.old)
        if count != 1:
            raise PatchRefused(
                f"{PLUGIN_NAME}: {patch.qualname} contains {edit.old!r} {count} times, "
                "expected once"
            )
        edited = edited.replace(edit.old, edit.new)
    if edited.count("\n") != lines.count("\n"):
        raise PatchRefused(f"{PLUGIN_NAME}: an edit of {patch.qualname} changes its line count")
    flags = _future_flags(tree)
    imports = _module_imports(tree)
    return Prepared(
        patch=patch,
        path=path,
        original=_compile_function(lines, node, owner, str(path), flags, patch.qualname, imports),
        edited=_compile_function(edited, node, owner, str(path), flags, patch.qualname, imports),
    )


def verify_file(check: FileCheck, roots: dict[str, Path]) -> None:
    path = roots[check.package] / check.path
    digest = _sha256(path)
    if digest != check.sha256:
        raise PatchRefused(
            f"{PLUGIN_NAME}: {path} has SHA-256 {digest}, expected {check.sha256}; "
            f"{check.flag} relies on {check.reason}"
        )


def _resolve(module: ModuleType, qualname: str) -> FunctionType:
    owner: object = module
    *classes, name = qualname.split(".")
    for class_name in classes:
        owner = vars(owner)[class_name]
    function = vars(owner).get(name)
    if type(function) is not FunctionType:
        raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__}.{qualname} is not a plain function")
    return function


def install(module: ModuleType, prepared: Prepared) -> bool:
    """Place the helpers and install the edited code; False if already installed."""
    patch = prepared.patch
    loaded_from = Path(getattr(module, "__file__", "") or "")
    if loaded_from.resolve() != prepared.path.resolve():
        raise PatchRefused(
            f"{PLUGIN_NAME}: {module.__name__} was loaded from {loaded_from}, not from the "
            f"verified file {prepared.path}"
        )
    function = _resolve(module, patch.qualname)
    current = function.__code__
    if getattr(function, MARKER, None) == patch.qualname and current == prepared.edited:
        return False
    if current != prepared.original or current.co_qualname != prepared.original.co_qualname:
        raise PatchRefused(
            f"{PLUGIN_NAME}: loaded {module.__name__}.{patch.qualname} differs from the "
            "verified source (another patch or interpreter version); refusing to patch"
        )
    for name in patch.helpers:
        helper = HELPERS[name]
        present = module.__dict__.get(name)
        if present is None:
            setattr(module, name, helper)
        elif present is not helper:
            raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__}.{name} already exists")
    function.__code__ = prepared.edited.replace(co_filename=current.co_filename)
    setattr(function, MARKER, patch.qualname)
    _LOG.info("%s: %s.%s patched for %s", PLUGIN_NAME, module.__name__, patch.qualname,
              patch.flag)
    return True


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
        # Other plugins' finders for the same module ask every finder, this one
        # included; answering None for a name being resolved ends that recursion.
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


# --------------------------------------------------------------------------- shared runtime


_ONCE: set[str] = set()
_ONCE_LOCK = threading.Lock()
_COUNTS: dict[str, int] = {}


def _log_once(key: str, level: int, message: str, *args) -> None:
    with _ONCE_LOCK:
        if key in _ONCE:
            return
        _ONCE.add(key)
    _LOG.log(level, message, *args)


def _count(key: str) -> None:
    with _ONCE_LOCK:
        _COUNTS[key] = _COUNTS.get(key, 0) + 1


def tp_world_size() -> int | None:
    """Tensor-parallel size of this process, or None before vLLM initializes it."""
    try:
        from vllm.distributed import parallel_state
    except Exception:  # noqa: BLE001 - no vLLM distributed state in this process
        return None
    try:
        if not parallel_state.model_parallel_is_initialized():
            return None
        return int(parallel_state.get_tensor_model_parallel_world_size())
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- item: latent shard


_LATENT_CLASS: type | None = None
_LATENT_LOCK = threading.Lock()


def latent_linear_class() -> type:
    """``ContiguousLatentLinear``, defined on first use so this module imports only stdlib."""
    global _LATENT_CLASS
    with _LATENT_LOCK:
        if _LATENT_CLASS is not None:
            return _LATENT_CLASS
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        copy_tensor_parallel_shard,
    )

    class ContiguousLatentLinear(MergedColumnParallelLinear):
        """Fused projection whose output rows are split into one contiguous block per rank.

        The checkpoint shards ``output_sizes`` are stacked in order; rank ``r``
        owns fused rows ``[w r, w (r + 1))`` with ``w = sum(output_sizes) / tp``.
        ``gather_output=True`` makes ``ColumnParallelLinear.forward`` all-gather
        the ranks' outputs along the last dimension, which concatenates the
        blocks in fused-row order.
        """

        def __init__(self, input_size: int, output_sizes: list[int], *, quant_config,
                     prefix: str):
            super().__init__(input_size, list(output_sizes), bias=False, gather_output=True,
                             quant_config=quant_config, prefix=prefix)
            total = sum(self.output_sizes)
            if total % self.tp_size:
                raise PatchRefused(f"{PLUGIN_NAME}: {prefix} width {total} is not divisible "
                                   f"by TP {self.tp_size}")
            self.local_width = total // self.tp_size
            self.local_start = self.tp_rank * self.local_width

        def _load(self, param, loaded_weight, loaded_shard_id) -> None:
            if isinstance(loaded_shard_id, bool) or not isinstance(loaded_shard_id, int):
                raise ValueError(f"{PLUGIN_NAME}: {self.prefix} loads one checkpoint shard at a "
                                 f"time; got shard id {loaded_shard_id!r}")
            self.validate_shard_id(loaded_shard_id)
            begin = sum(self.output_sizes[:loaded_shard_id])
            end = begin + self.output_sizes[loaded_shard_id]
            first = max(begin, self.local_start)
            last = min(end, self.local_start + self.local_width)
            if first >= last:
                # This checkpoint tensor holds none of this rank's rows: nothing
                # to copy, and ``param`` must not be inspected. Under composed
                # online quantization the layer is processed as soon as this
                # rank's rows are in (vllm/model_executor/model_loader/reload/
                # layerwise.py), which replaces ``weight`` with a packed, empty
                # Parameter that keeps this loader but no ``output_dim``.
                # AutoWeightsLoader calls the model's load_weights once per
                # contiguous run of ``model.*`` weights, so ``lm_head.weight``
                # in the file stream starts a call whose fresh parameter map
                # hands that replacement to this loader for the other tensor.
                return
            if getattr(param, "output_dim", None) != 0 or getattr(param, "is_sharded_weight",
                                                                  False):
                raise ValueError(
                    f"{PLUGIN_NAME}: {self.prefix} received fused rows [{first}, {last}) from "
                    f"checkpoint shard {loaded_shard_id} for a {type(param).__name__} without "
                    "output dimension 0: the layer was processed before all of this rank's "
                    "rows were loaded, or the checkpoint tensor is already sharded")
            unit = 1
            if isinstance(param, BlockQuantScaleParameter):
                unit = int(self.weight_block_size[0])
            for value in (begin, end, first, last):
                if value % unit:
                    raise ValueError(f"{PLUGIN_NAME}: {self.prefix} row {value} is not on a "
                                     f"{unit}-row scale block boundary")
            if int(loaded_weight.shape[0]) != (end - begin) // unit:
                raise ValueError(f"{PLUGIN_NAME}: {self.prefix} shard {loaded_shard_id} has "
                                 f"{loaded_weight.shape[0]} rows, expected {(end - begin) // unit}")
            rows = (last - first) // unit
            destination = param.data.narrow(0, (first - self.local_start) // unit, rows)
            copy_tensor_parallel_shard(destination, loaded_weight, 0, (first - begin) // unit,
                                       rows)

        def weight_loader(self, param, loaded_weight, loaded_shard_id=None) -> None:
            self._load(param, loaded_weight, loaded_shard_id)

        def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None) -> None:
            self._load(param, loaded_weight, loaded_shard_id)

        def extra_repr(self) -> str:
            return (super().extra_repr() + f", fused_rows=[{self.local_start}, "
                    f"{self.local_start + self.local_width})")

    with _LATENT_LOCK:
        if _LATENT_CLASS is None:
            _LATENT_CLASS = ContiguousLatentLinear
        return _LATENT_CLASS


def qkv_a(linear_cls: type, input_size: int, output_sizes: list[int], **kwargs):
    """``DeepseekV32Attention.fused_qkv_a_proj``: column-parallel at TP8.

    Called with the image's constructor arguments; returns the image's
    replicated layer at any other TP size or when the arguments differ from
    the image's call (``quant_config`` and ``prefix`` keywords only).
    """
    tp = tp_world_size()
    prefix = str(kwargs.get("prefix", ""))
    reason = None
    if tp != TARGET_TP:
        reason = f"tensor-parallel size {tp}"
    elif set(kwargs) - {"quant_config", "prefix"}:
        reason = f"unexpected constructor arguments {sorted(kwargs)}"
    elif sum(output_sizes) % tp:
        reason = f"output sizes {list(output_sizes)} do not split into {tp} blocks"
    if reason is not None:
        _count("replicated")
        _log_once("keep", logging.INFO, "%s: %s keeps the image's replicated fused_qkv_a_proj: %s",
                  PLUGIN_NAME, prefix, reason)
        return linear_cls(input_size, output_sizes, **kwargs)
    layer = latent_linear_class()(input_size, list(output_sizes),
                                  quant_config=kwargs.get("quant_config"), prefix=prefix)
    _count("sharded")
    _log_once("shard", logging.INFO,
              "%s: %s is column-parallel over TP%d: rank %d computes fused rows [%d, %d) of %d "
              "and all-gathers %d BF16 columns per row (%d bytes per row per rank); every MLA "
              "layer, including the MTP layer, follows the same split", PLUGIN_NAME, prefix, tp,
              layer.tp_rank, layer.local_start, layer.local_start + layer.local_width,
              sum(output_sizes), layer.local_width, 2 * layer.local_width)
    return layer


# --------------------------------------------------------------------------- item: eh_proj


def eh_proj(hidden_size: int, prefix: str, linear_cls):
    """MTP ``eh_proj``: row-parallel at TP8, the image's replicated nn.Linear otherwise."""
    tp = tp_world_size()
    if tp != TARGET_TP:
        _log_once("eh_proj.keep", logging.INFO,
                  "%s: %s at TP %s keeps the replicated eh_proj", PLUGIN_NAME, EH_PROJ_FLAG, tp)
        return linear_cls(hidden_size * 2, hidden_size, bias=False)
    from vllm.model_executor.layers.linear import RowParallelLinear

    layer = RowParallelLinear(
        hidden_size * 2,
        hidden_size,
        bias=False,
        input_is_parallel=False,
        reduce_results=True,
        quant_config=None,
        prefix=f"{prefix}.eh_proj",
        return_bias=False,
    )
    _log_once("eh_proj.row", logging.INFO,
              "%s: %s.eh_proj is RowParallelLinear(%d -> %d) over TP%d: %d input columns per "
              "rank and one all-reduce of %d values per token", PLUGIN_NAME, prefix,
              hidden_size * 2, hidden_size, tp, hidden_size * 2 // tp, hidden_size)
    return layer


HELPERS: dict[str, Callable] = {
    LATENT_HELPER: qkv_a,
    EH_PROJ_HELPER: eh_proj,
}


# --------------------------------------------------------------------------- registration


_STATE_LOCK = threading.Lock()
_FINDER: PatchOnImport | None = None
_REGISTERED: frozenset[str] | None = None


def selected_flags() -> tuple[str, ...]:
    """Flags set to 1; any value other than 0, 1 or unset is refused."""
    selected = []
    for flag in FLAGS:
        value = os.environ.get(flag, "0").strip()
        if value not in ("0", "1"):
            raise PatchRefused(f"{PLUGIN_NAME}: {flag} must be 0 or 1, got {value!r}")
        if value == "1":
            selected.append(flag)
    return tuple(selected)


def _package_roots() -> dict[str, Path]:
    import importlib.util

    roots = {}
    for name in ("vllm", "b12x"):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise PatchRefused(f"{PLUGIN_NAME}: package {name} not found")
        roots[name] = Path(list(spec.submodule_search_locations)[0]).resolve()
    return roots


def register() -> None:
    """vLLM general-plugin entry point; idempotent within a process."""
    global _FINDER, _REGISTERED
    flags = frozenset(selected_flags())
    with _STATE_LOCK:
        if _REGISTERED is not None:
            if flags != _REGISTERED:
                raise PatchRefused(f"{PLUGIN_NAME}: flags changed after registration")
            return
        if not flags:
            _REGISTERED = flags
            _LOG.info("%s: %s are 0; nothing patched", PLUGIN_NAME, ", ".join(FLAGS))
            return
        roots = _package_roots()
        for check in FILE_CHECKS:
            if check.flag in flags:
                verify_file(check, roots)
        by_module: dict[str, list[Prepared]] = {}
        for patch in PATCHES:
            if patch.flag in flags:
                by_module.setdefault(patch.module, []).append(prepare(patch, roots["vllm"]))
        pending: dict[str, Callable[[ModuleType], None]] = {}
        for name, prepared in by_module.items():

            def apply(module: ModuleType, _prepared=tuple(prepared)) -> None:
                for item in _prepared:
                    install(module, item)

            module = sys.modules.get(name)
            if module is not None:
                apply(module)
            else:
                pending[name] = apply
        if pending:
            _FINDER = PatchOnImport(pending)
            sys.meta_path.insert(0, _FINDER)
        _REGISTERED = flags
    _LOG.info("%s: enabled %s; %d function(s) patched now or on first import of: %s",
              PLUGIN_NAME, ", ".join(sorted(flags)),
              sum(len(items) for items in by_module.values()),
              ", ".join(sorted(pending)) or "none (already imported)")


def status() -> dict[str, object]:
    """Registration state of this process, for diagnostics and tests."""
    with _STATE_LOCK:
        return {
            "registered_flags": sorted(_REGISTERED) if _REGISTERED is not None else None,
            "pending_modules": list(_FINDER.pending()) if _FINDER is not None else [],
            "layers": dict(_COUNTS),
        }
