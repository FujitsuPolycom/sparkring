"""vLLM general plugin: split the GLM-5.3 DSA indexer's prefill rows over the TP group's KV copies.

Status: research-only. CPU tests cover the settings, the pins, the
installation, the row partition, the all-gather layout, the split against the
image's own ``forward`` on synthetic inputs with thread-emulated ranks and
the fallbacks (``tests/``). Nothing here has run on the ring; the verify
mode below is the qualification step there.

Scope. GLM-5.3 (``glm_moe_dsa``) on image
``sparkring-dev/kraken:csf-sircl-libsircl-20261008`` (``816c6d6a7e96``; vLLM
``0.1.dev21553+gab86b7073`` with Python sources of vLLM ``bc9ea774``, B12X
sources ``cc36aa6f``), B12X DSA indexer, TP 8 with decode context
parallelism (DCP) 1, 2 or 4. DCP 8 at TP 8 leaves one KV copy and refuses
when the first indexer is constructed.

What it does. The DSA indexer (``B12xSparseIndexer`` in
``vllm/v1/attention/backends/mla/b12x_indexer.py``) scores every prefill row
of a step on every TP rank: each of the ``8 / d`` DCP groups, each holding one
full KV copy, repeats the same work. With the plugin, group ``j`` scores and
merges only its block of the step's prefill rows, and one all-gather over the
TP group fills ``topk_indices_buffer`` for every row on every rank. Rows are
padded to a multiple of 8; each rank sends its own eighth
(``layout.py``). Each row's selection is computed by the image's
``_run_paged_topk`` with the same plan and the same per-row inputs as the
image's launch of that row, and merged by the image's ``_merge_dcp_topk``.
Decode-only calls, CUDA graph captures and every prefill step the split
cannot handle as the image does take the image's ``forward`` unchanged and
are counted (``runtime.FALLBACKS``).

Settings (read by ``register``; the same on every rank):

* ``GLM_DSA_INDEXER_SPLIT`` [0]: 1 installs the wrappers.
* ``GLM_DSA_INDEXER_SPLIT_MIN_ROWS`` [512]: smallest step, in prefill rows,
  that takes the split (whole number from 1).
* ``GLM_DSA_INDEXER_SPLIT_FULL_LAUNCHES`` [0]: 1 cuts each request's rows of
  the group's block into launches of the prepared prefill plan's row capacity
  (4,096) instead of the image's logits-budget launches (128 rows at a 1M
  context). Each row keeps its inputs, so its selection is the same.
* ``GLM_DSA_INDEXER_SPLIT_MIXED`` [0]: 1 also splits the prefill rows of steps
  that carry decode rows; the decode rows run the image's decode statements
  (the image's ``forward`` with the step's prefill part removed for that call).
  0 sends such steps to the image's path (counted as ``mixed``).
* ``GLM_DSA_INDEXER_SPLIT_VERIFY`` [0]: N > 0 computes the first N eligible
  layer-chunks with the split and twice with the image's path, keeps the
  image's result and logs, per rank, the final index sets (split against
  image, image against image) and every pre-merge candidate list of the
  rank's block, split against image, with differences among tied scores
  counted apart. Extra collectives and syncs: not for timing.
* ``GLM_DSA_INDEXER_SPLIT_VERIFY_MIN_CONTEXT`` [0]: verify only layer-chunks
  whose request context reaches this many tokens.

Mechanism. ``register`` requires the SHA-256 of every image file whose
behavior the plugin relies on (``FILE_CHECKS``) and of the package's other
modules (``PACKAGE_SHA256``, so a launcher pin of this file pins the whole
package) to equal the recorded values, and compiles ``B12xSparseIndexer.forward``
and ``.__init__`` from the verified source at their line numbers in that file.
When ``b12x_indexer`` first executes (or at once if it already has) the loaded
methods must be exactly those compiled functions; any wrapper refuses with
``PatchRefused`` naming it. The class attributes are then replaced by
wrappers that call them (``functools.wraps``). The ``__init__`` wrapper, run
for every indexer the model constructs (after every plugin registered),
refuses unless ``forward`` is this plugin's wrapper itself, and checks the TP
and DCP groups (``runtime.attach``). No file is written.

Coexistence. No other plugin replaces ``B12xSparseIndexer.forward`` or
``.__init__``. ``glm53full_compat`` wraps ``B12xIndexerMetadataBuilder.__init__``;
``glm_dsa_ckv_gather`` wraps the B12X sparse MLA implementation, which reads
``topk_indices_buffer`` after the indexer. The split path calls
``_run_paged_topk`` and ``_merge_dcp_topk`` through the image module, so a
replacement of either applies to both paths.

Prefill runs the image's gathered-key path
(``B12xSparseIndexer._run_gathered_prefill``) when the indexer was built with
``VLLM_DCP_INDEXER_KEY_GATHER=1``; that path does not score rows per DCP
group, so the split sends every such prefill step to the image's ``forward``
(counted as ``key_gather``). The default of the variable is 0.
"""

from __future__ import annotations

import __future__
import ast
import functools
import hashlib
import importlib.abc
import importlib.util
import logging
import os
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

PLUGIN_NAME = "glm_dsa_indexer_split"
PLUGIN_VERSION = "1.1.0"
MARKER = "__glm_dsa_indexer_split__"
IMAGE = "816c6d6a7e96"
IMAGE_VLLM_SOURCES = "bc9ea774"
IMAGE_B12X_SOURCES = "cc36aa6f"

ENABLE = "GLM_DSA_INDEXER_SPLIT"
MIN_ROWS = "GLM_DSA_INDEXER_SPLIT_MIN_ROWS"
FULL_LAUNCHES = "GLM_DSA_INDEXER_SPLIT_FULL_LAUNCHES"
MIXED = "GLM_DSA_INDEXER_SPLIT_MIXED"
VERIFY = "GLM_DSA_INDEXER_SPLIT_VERIFY"
VERIFY_MIN_CONTEXT = "GLM_DSA_INDEXER_SPLIT_VERIFY_MIN_CONTEXT"
DEFAULT_MIN_ROWS = 512

INDEXER_MODULE = "vllm.v1.attention.backends.mla.b12x_indexer"
INDEXER_PATH = "v1/attention/backends/mla/b12x_indexer.py"
INDEXER_SHA256 = "e1841ab03516e9f58e52400491ed5a11db7826c4a676f49c427a9e6246210ae2"
INDEXER_CLASS = "B12xSparseIndexer"
WRAPPED = ("forward", "__init__")

_LOG = logging.getLogger("vllm." + PLUGIN_NAME)


class PatchRefused(RuntimeError):
    """The loaded code or the parallel layout differs from what this plugin was written for."""


@dataclass(frozen=True)
class FileCheck:
    package: str  # vllm or b12x
    path: str  # relative to the package directory
    sha256: str
    reason: str


# SHA-256 of the package's other modules; make_patch.py --refresh-pins rewrites them.
PACKAGE_SHA256 = {
    "layout.py": "9bac5896f2223d32ec90e38e78918cf034ba58747fb516bb4320a2b75681ee1f",
    "runtime.py": "964833cfc25b3c7c47c8c9a29eb0767e77a89e7ecdfb4470803f9263af364c8f",
}

FILE_CHECKS: tuple[FileCheck, ...] = (
    FileCheck("vllm", INDEXER_PATH, INDEXER_SHA256,
              "B12xSparseIndexer.forward scores each prefill chunk with _run_paged_topk (plan _plan('prefill', rows), "
              "per-row lengths cu_seqlen_ke - cu_seqlen_ks, the request's page-table row expanded over the rows, "
              "pages from the request's local length) and merges it with _merge_dcp_topk, both looked up in the "
              "module; the decode branch is independent of the prefill branch; the gathered-key prefill path runs "
              "only when the indexer's dcp_key_gather is true; the prepared prefill plan holds 4,096 rows"),
    FileCheck("vllm", "v1/attention/backends/mla/indexer.py",
              "9519ec4511c5de03e851795ee0fa04ba64e4de2bc30c97f61bd3e591761adf9e",
              "prefill chunks are single-request runs of consecutive rows in step order; a row's "
              "cu_seqlen_ks/ke come from its own position only, so they do not depend on where a launch starts; "
              "decode rows precede prefill rows"),
    FileCheck("vllm", "models/deepseek_v32/nvidia/b12x.py",
              "2412da9bec4b3597064395c4c78b0d93c1235153fe3e55df4b9dba6dcde4942d",
              "B12xDSAIndexer.run_indexer calls the B12xSparseIndexer module and DCP 1 asks for physical slots"),
    FileCheck("vllm", "models/deepseek_v32/attention.py",
              "c1358e677002658c848247758c45ef5543eec9c5122bc3b36e2e5c4447c68bd6",
              "the attention reads topk_indices_buffer only after run_indexer returns"),
    FileCheck("vllm", "models/deepseek_v32/nvidia/model.py",
              "03c85e81a225a0de3312bf62830ff836c8f0c9d894e7fee4381e5510f366cd10",
              "topk_indices_buffer is one [max_num_batched_tokens, index_topk] int32 tensor shared by every layer "
              "and the MTP draft"),
    FileCheck("vllm", "distributed/parallel_state.py",
              "3aba1b4376c828a343d781d23fd1dd0c4bdb40b8dafa06a927006e7596359212",
              "DCP groups are runs of consecutive TP ranks and GroupCoordinator.all_gather concatenates the ranks' "
              "tensors in rank order"),
    FileCheck("vllm", "distributed/device_communicators/cuda_communicator.py",
              "33d27cad52def83159c7f63a9b7bd47e61d3e1d50c453ea9927576fc808f3ad8",
              "a dim-0 all-gather returns the rank-major concatenation (sparknet adapter, symmetric memory or PyNccl)"),
    FileCheck("vllm", "compilation/breakable_cudagraph.py",
              "a7b53a403c776df8195881c98fdba72e0e2662c15adc047cd5c79736247d19e2",
              "eager segments run after the capture segment ended, so a prefill indexer call never runs inside a "
              "capture outside FULL decode graphs"),
    FileCheck("vllm", "model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py",
              "80aac0786fb6d14055cd7ef9a2030d5a3c22cbccf61cf4244c2b06ee3d11873e",
              "the candidate merge selects each row's top-k by (score, position) keys of that row only"),
    FileCheck("vllm", "forward_context.py",
              "9f8be687de3bb5cecd5b32ef4fb92cdc83faa620534f157084cf6841e28be091",
              "the forward context's attn_metadata maps layer names to their metadata"),
    FileCheck("b12x", "attention/dsa_indexer/paged.py",
              "9e0ce066b3c1254a44860e9054cf66f6fdbfb05a403ac352237d67431e1d3f4e",
              "the prefill route gathers each supertile's keys from the page table alone, walks a number of "
              "supertiles set by the page-table width, and folds each row's carry in its own row"),
    FileCheck("b12x", "attention/dsa_indexer/contiguous_kernel.py",
              "47338c14b1c6069c606903268a15b3026cbe1b9d96b1cfe3cbdab926dd9ca855",
              "a logit depends only on its row's query and weights and on the key; a tile is skipped only when no "
              "row of the tile reaches it"),
    FileCheck("b12x", "attention/dsa_indexer/tiled_topk.py",
              "13cbc9cb2841bb9050c420b767d7f65d77fe2299159ea5688d9de00aacf4c461",
              "one CTA selects one row's top-k from that row's logits and carry"),
    FileCheck("b12x", "attention/dsa_indexer/_preparation.py",
              "6237d4a0420e8af3ea60d135f92f0cbbdbaa4dfdad00b341f32dab6ba9e47635",
              "serving runs the prepared state with allow_transient_fold_buffers=False (no row-count dependent "
              "fold mode)"),
    FileCheck("b12x", "attention/dsa_indexer/scratch.py",
              "c3705a1427c03639563a5d17283f1157e4b841bf7a9b9c6c55eaa7db4580771f",
              "prefill plans take the packed-contiguous route with a scratch layout fixed by the plan"),
)


@dataclass(frozen=True)
class Original:
    path: Path
    code: dict  # method name -> compiled code


# --------------------------------------------------------------------------- settings


def _flag(env: Mapping[str, str], name: str) -> bool:
    value = (env.get(name) or "0").strip() or "0"
    if value not in ("0", "1"):
        raise PatchRefused(f"{PLUGIN_NAME}: {name} must be 0 or 1, got {value!r}")
    return value == "1"


def _whole(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    text = (env.get(name) or "").strip()
    if not text:
        return default
    if not text.isdigit() or int(text) < minimum:
        raise PatchRefused(f"{PLUGIN_NAME}: {name} must be a whole number of at least {minimum}, got {text!r}")
    return int(text)


def settings_from_env(env: Mapping[str, str] = os.environ) -> dict:
    """Every setting, checked; a malformed value refuses."""
    values = {
        "enabled": _flag(env, ENABLE),
        "min_rows": _whole(env, MIN_ROWS, DEFAULT_MIN_ROWS, 1),
        "full_launches": _flag(env, FULL_LAUNCHES),
        "mixed": _flag(env, MIXED),
        "verify": _whole(env, VERIFY, 0, 0),
        "verify_min_context": _whole(env, VERIFY_MIN_CONTEXT, 0, 0),
    }
    if not values["enabled"]:
        named = [name for name in (MIN_ROWS, FULL_LAUNCHES, MIXED, VERIFY, VERIFY_MIN_CONTEXT) if env.get(name)]
        if named:
            raise PatchRefused(f"{PLUGIN_NAME}: {', '.join(named)} set without {ENABLE}=1")
    return values


def describe_settings(values: Mapping[str, object]) -> str:
    return (f"min rows {values['min_rows']}, full launches {'on' if values['full_launches'] else 'off'}, mixed steps "
            f"{'split' if values['mixed'] else 'image path'}, verify {values['verify']} layer-chunks"
            + (f" from {values['verify_min_context']} tokens" if values["verify_min_context"] else ""))


# --------------------------------------------------------------------------- pins and compiled originals


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_package(directory: Path | None = None) -> None:
    """Require the package's other modules to be the ones whose SHA-256 this file records."""
    directory = directory or Path(__file__).resolve().parent
    for name, expected in PACKAGE_SHA256.items():
        path = directory / name
        try:
            digest = _sha256(path)
        except FileNotFoundError:
            raise PatchRefused(f"{PLUGIN_NAME}: {path} is missing") from None
        if digest != expected:
            raise PatchRefused(f"{PLUGIN_NAME}: {path} has SHA-256 {digest}, expected {expected} (the package "
                               "changed after its pins were recorded)")


def verify_file(check: FileCheck, roots: Mapping[str, Path]) -> None:
    path = roots[check.package] / check.path
    try:
        digest = _sha256(path)
    except FileNotFoundError:
        raise PatchRefused(f"{PLUGIN_NAME}: {path} is missing; the plugin relies on: {check.reason}") from None
    if digest != check.sha256:
        raise PatchRefused(f"{PLUGIN_NAME}: {path} has SHA-256 {digest}, expected {check.sha256} (vLLM "
                           f"vLLM sources {IMAGE_VLLM_SOURCES}, B12X sources {IMAGE_B12X_SOURCES} of image {IMAGE}); the plugin relies on: "
                           f"{check.reason}")


def _future_flags(tree: ast.Module) -> int:
    flags = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                flags |= getattr(__future__, alias.name).compiler_flag
    return flags


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


def _find_code(code: CodeType, qualname: str) -> CodeType | None:
    for const in code.co_consts:
        if isinstance(const, CodeType):
            if const.co_qualname == qualname:
                return const
            found = _find_code(const, qualname)
            if found is not None:
                return found
    return None


def compile_originals(vllm_root: Path) -> Original:
    """Compile ``B12xSparseIndexer.forward`` and ``.__init__`` from the verified file at their line numbers there."""
    path = vllm_root / INDEXER_PATH
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != INDEXER_SHA256:
        raise PatchRefused(f"{PLUGIN_NAME}: {path} has SHA-256 {digest}, expected {INDEXER_SHA256} (vLLM "
                           f"vLLM sources {IMAGE_VLLM_SOURCES} of image {IMAGE}); refusing to wrap {INDEXER_CLASS}")
    text = data.decode("utf-8")
    tree = ast.parse(text, filename=str(path))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == INDEXER_CLASS]
    if len(classes) != 1:
        raise PatchRefused(f"{PLUGIN_NAME}: class {INDEXER_CLASS} not found once in {path}")
    owner = classes[0]
    flags, imports = _future_flags(tree), _module_imports(tree)
    lines = text.splitlines(keepends=True)
    code: dict[str, CodeType] = {}
    for name in WRAPPED:
        nodes = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(nodes) != 1 or nodes[0].decorator_list:
            raise PatchRefused(f"{PLUGIN_NAME}: {INDEXER_CLASS}.{name} is not one undecorated method in {path}")
        node = nodes[0]
        source = "\n" * (node.lineno - 2) + f"class {owner.name}:\n" + "".join(lines[node.lineno - 1:node.end_lineno])
        if not source.endswith("\n"):
            source += "\n"
        source += imports + "\n"
        compiled = _find_code(compile(source, str(path), "exec", flags=flags, dont_inherit=True),
                              f"{INDEXER_CLASS}.{name}")
        if compiled is None:
            raise PatchRefused(f"{PLUGIN_NAME}: compiled code of {INDEXER_CLASS}.{name} not found")
        code[name] = compiled
    return Original(path=path, code=code)


# --------------------------------------------------------------------------- installation

_INSTALLED: dict[str, FunctionType] = {}


def describe(function: object) -> str:
    code = getattr(function, "__code__", None)
    if code is None:
        return repr(function)
    return f"{code.co_qualname} ({code.co_filename}:{code.co_firstlineno})"


def _is_image_function(function: object, original: CodeType) -> bool:
    return (type(function) is FunctionType and getattr(function, "__wrapped__", None) is None
            and function.__code__ == original and function.__code__.co_qualname == original.co_qualname)


def check_forward(cls: type) -> None:
    """Refuse unless ``cls.forward`` resolves to this plugin's installed wrapper."""
    installed = _INSTALLED.get("forward")
    current = getattr(cls, "forward", None)
    if installed is None or current is installed:
        return
    chain, inner = [], current
    for _ in range(16):
        inner = getattr(inner, "__wrapped__", None)
        if inner is None:
            break
        chain.append(inner)
    how = "wraps this plugin's wrapper" if any(item is installed for item in chain) else "replaced this plugin's wrapper"
    raise PatchRefused(f"{PLUGIN_NAME}: {cls.__qualname__}.forward is {describe(current)}, which {how} after "
                       f"{PLUGIN_NAME} installed it; refusing to run with an unknown wrapper of the DSA indexer")


def _make_forward(image_forward: FunctionType) -> FunctionType:
    @functools.wraps(image_forward)
    def forward(self, hidden_states, q_quant, k, weights):
        from . import runtime

        return runtime.forward(self, image_forward, hidden_states, q_quant, k, weights)

    setattr(forward, MARKER, f"{INDEXER_CLASS}.forward")
    return forward


def _make_init(image_init: FunctionType) -> FunctionType:
    @functools.wraps(image_init)
    def __init__(self, *args, **kwargs):
        image_init(self, *args, **kwargs)
        from . import runtime

        check_forward(type(self))
        try:
            runtime.attach(self)
        except runtime.SplitRefused as exc:
            raise PatchRefused(str(exc)) from None

    setattr(__init__, MARKER, f"{INDEXER_CLASS}.__init__")
    return __init__


def install(module: ModuleType, original: Original) -> bool:
    """Wrap ``B12xSparseIndexer.forward`` and ``.__init__``; False if this plugin's wrappers are in place."""
    loaded_from = Path(getattr(module, "__file__", "") or "")
    if loaded_from.resolve() != original.path.resolve():
        raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__} was loaded from {loaded_from}, not from the verified "
                           f"file {original.path}")
    cls = vars(module).get(INDEXER_CLASS)
    if not isinstance(cls, type):
        raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__}.{INDEXER_CLASS} is not a class")
    pending = []
    for name in WRAPPED:
        current = vars(cls).get(name)
        if current is not None and current is _INSTALLED.get(name):
            continue
        if not _is_image_function(current, original.code[name]):
            raise PatchRefused(f"{PLUGIN_NAME}: {module.__name__}.{INDEXER_CLASS}.{name} is {describe(current)}, not "
                               "the image's method; refusing to wrap an unknown wrapper")
        pending.append((name, current))
    if not pending:
        return False
    from . import runtime

    runtime.configure(runtime.Settings(**{key: _SETTINGS[key] for key in
                                          ("min_rows", "full_launches", "mixed", "verify", "verify_min_context")}))
    for name, current in pending:
        wrapper = _make_forward(current) if name == "forward" else _make_init(current)
        setattr(cls, name, wrapper)
        _INSTALLED[name] = wrapper
        _LOG.info("%s: %s.%s.%s wrapped", PLUGIN_NAME, module.__name__, INDEXER_CLASS, name)
    return True


class PatchOnImport(importlib.abc.MetaPathFinder):
    """Run a callback right after the first import of each named module executes.

    The finder returns the spec the remaining finders produce, with that
    spec's loader wrapped, and removes itself after the last callback ran.
    While it resolves a name it answers ``None`` for that name, so finders of
    other plugins that ask every finder for the same module do not recurse.
    """

    def __init__(self, callbacks: dict[str, Callable[[ModuleType], object]]):
        self._callbacks = dict(callbacks)
        self._lock = threading.Lock()
        self._resolving = threading.local()

    def pending(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._callbacks))

    def find_spec(self, fullname, path=None, target=None):
        with self._lock:
            callback = self._callbacks.get(fullname)
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


# --------------------------------------------------------------------------- registration

_STATE_LOCK = threading.Lock()
_FINDER: PatchOnImport | None = None
_REGISTERED: dict | None = None
_SETTINGS: dict = {}


def _package_roots(packages: Iterable[str]) -> dict[str, Path]:
    roots = {}
    for name in packages:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            raise PatchRefused(f"{PLUGIN_NAME}: package {name} not found (the plugin needs vLLM and b12x of image "
                               f"{IMAGE})")
        roots[name] = Path(list(spec.submodule_search_locations)[0]).resolve()
    return roots


def register() -> None:
    """vLLM general-plugin entry point; idempotent within a process."""
    global _FINDER, _REGISTERED, _SETTINGS
    values = settings_from_env()
    with _STATE_LOCK:
        if _REGISTERED is not None:
            if values != _REGISTERED:
                raise PatchRefused(f"{PLUGIN_NAME}: settings changed after registration")
            return
        if not values["enabled"]:
            _REGISTERED = values
            _LOG.info("%s: %s is not 1; nothing patched", PLUGIN_NAME, ENABLE)
            return
        verify_package()
        roots = _package_roots(("vllm", "b12x"))
        for check in FILE_CHECKS:
            verify_file(check, roots)
        original = compile_originals(roots["vllm"])
        _SETTINGS = values
        module = sys.modules.get(INDEXER_MODULE)
        if module is not None:
            install(module, original)
            pending = ()
        else:
            _FINDER = PatchOnImport({INDEXER_MODULE: lambda m, _o=original: install(m, _o)})
            sys.meta_path.insert(0, _FINDER)
            pending = (INDEXER_MODULE,)
        _REGISTERED = values
    _LOG.info("%s %s: enabled (%s); %s", PLUGIN_NAME, PLUGIN_VERSION, describe_settings(values),
              "patched now" if not pending else f"patching on first import of {', '.join(pending)}")


def status() -> dict[str, object]:
    """Registration state and, in a process that imported the runtime, its counters."""
    with _STATE_LOCK:
        out: dict[str, object] = {
            "version": PLUGIN_VERSION,
            "settings": dict(_REGISTERED) if _REGISTERED else None,
            "pending_modules": list(_FINDER.pending()) if _FINDER is not None else [],
            "installed": sorted(_INSTALLED),
        }
    runtime = sys.modules.get(f"{__name__}.runtime")
    if runtime is not None:
        out["runtime"] = runtime.stats()
    return out


__all__ = [
    "FILE_CHECKS",
    "PACKAGE_SHA256",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "PatchRefused",
    "check_forward",
    "compile_originals",
    "install",
    "register",
    "settings_from_env",
    "status",
    "verify_file",
    "verify_package",
]
