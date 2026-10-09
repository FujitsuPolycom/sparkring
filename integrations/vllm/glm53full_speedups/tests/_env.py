"""The CPU import environment for the speedups tests: the image's linear layers.

``SPARKRING_GLM53_IMAGE_SOURCES`` names a directory with ``vllm/`` and
``b12x/`` subtrees holding the target image's Python sources (image
``sparkring-dev/kraken:csf-sircl-libsircl-20261008``; see the integration
README). Without it every test that needs the image's code fails with the
instructions below.

The image's ``vllm/model_executor/parameter.py`` and
``vllm/model_executor/layers/linear.py`` load as the modules
``vllm.model_executor.parameter`` and ``vllm.model_executor.layers.linear``,
so the plugin's ``ContiguousLatentLinear`` extends the image's
``MergedColumnParallelLinear`` and the tests run the image's weight loaders
and GEMM paths. Every ``vllm.*`` name the two files import is a stand-in;
the tensor-parallel collectives run on eight threads (``TPWorld``).

The loaded files' SHA-256 must equal the pins the plugin records, so every
test session replays those two pins.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

HELP = (
    "set SPARKRING_GLM53_IMAGE_SOURCES to a directory holding the target image's "
    "vllm/ and b12x/ Python sources (see integrations/vllm/glm53full_speedups/README.md)"
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(os.environ["SPARKRING_GLM53_IMAGE_SOURCES"]).resolve() \
    if os.environ.get("SPARKRING_GLM53_IMAGE_SOURCES") else None

LINEAR_PATH = "vllm/model_executor/layers/linear.py"
PARAMETER_PATH = "vllm/model_executor/parameter.py"
TP = 8

_CONTEXT = threading.local()


class TPGroup:
    """Barrier-based tensor-parallel collectives among ``size`` threads."""

    def __init__(self, size: int):
        self.size = size
        self._barrier = threading.Barrier(size, timeout=120.0)
        self._slots: list = [None] * size

    @property
    def rank(self) -> int:
        return getattr(_CONTEXT, "rank", 0)

    def all_gather(self, tensor, dim: int = -1):
        self._slots[self.rank] = tensor.detach().clone()
        self._barrier.wait()
        result = __import__("torch").cat(list(self._slots), dim=dim)
        self._barrier.wait()
        return result

    def all_reduce(self, tensor, op: str = "sum"):
        self._slots[self.rank] = tensor.detach().clone()
        self._barrier.wait()
        torch = __import__("torch")
        total = torch.zeros_like(tensor)
        for part in self._slots:
            total += part
        self._barrier.wait()
        return tensor.copy_(total)

    def wait(self):
        self._barrier.wait()

    def abort(self):
        self._barrier.abort()


_GROUP: TPGroup | None = None


def group() -> TPGroup:
    if _GROUP is None:
        raise RuntimeError("no TP group built; call build_group(size) first")
    return _GROUP


def build_group(size: int = TP) -> TPGroup:
    global _GROUP
    _GROUP = TPGroup(size)
    return _GROUP


def run_ranks(body):
    """Run ``body(rank)`` on every rank in its own thread; re-raise the first failure."""
    results: list = [None] * group().size
    errors: list = [None] * group().size

    def target(rank: int) -> None:
        _CONTEXT.rank = rank
        try:
            results[rank] = body(rank)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors[rank] = exc
            _GROUP.abort()

    threads = [threading.Thread(target=target, args=(rank,)) for rank in range(group().size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(180.0)
    for rank, error in enumerate(errors):
        if error is not None:
            raise error
    return results


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    if "__path__" in attributes:  # a stand-in package: find_spec works on it
        from importlib.machinery import ModuleSpec

        spec = ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = list(attributes["__path__"])
        module.__spec__ = spec
    sys.modules[name] = module
    return module


def stub_packages() -> None:
    """Register the image's ``vllm`` and ``b12x`` trees as importable packages.

    ``register`` resolves the package roots through them; no image module is
    imported.
    """
    if "vllm" in sys.modules:
        return
    if ROOT is None:
        raise RuntimeError(HELP)
    _module("vllm", __path__=[str(ROOT / "vllm")])
    _module("b12x", __path__=[str(ROOT / "b12x")])


def _stub_environ() -> None:
    """Register ``vllm`` and the names the two image files import."""
    if "vllm" in sys.modules:
        return
    stub_packages()
    _module("vllm.envs", VLLM_BATCH_INVARIANT=False, VLLM_XPU_FORCE_N_CONTIG_WEIGHT=False)
    _module("vllm.config", get_current_vllm_config=lambda: None,
            get_current_vllm_config_or_none=lambda: None)

    def divide(numerator: int, denominator: int) -> int:
        if numerator % denominator:
            raise ValueError(f"{numerator} is not divisible by {denominator}")
        return numerator // denominator

    distributed = _module(
        "vllm.distributed", divide=divide,
        get_tensor_model_parallel_rank=lambda: group().rank if _GROUP else 0,
        get_tensor_model_parallel_world_size=lambda: group().size if _GROUP else 1,
        split_tensor_along_last_dim=lambda tensor, num_partitions: [
            part.contiguous() for part in __import__("torch").chunk(tensor, num_partitions, dim=-1)],
        tensor_model_parallel_all_gather=lambda tensor, dim=-1: group().all_gather(tensor, dim),
        tensor_model_parallel_all_reduce=lambda tensor: group().all_reduce(tensor))
    parallel_state = _module("vllm.distributed.parallel_state",
                             model_parallel_is_initialized=lambda: True,
                             get_tensor_model_parallel_world_size=lambda: group().size if _GROUP else 1,
                             register_b12x_collective_describer=lambda *_: None)
    distributed.parallel_state = parallel_state
    _module("vllm.logger", init_logger=logging.getLogger)

    class PluggableLayer(__import__("torch").nn.Module):
        registry: dict = {}

        @classmethod
        def register(cls, name: str):
            def decorate(candidate):
                cls.registry[name] = candidate
                return candidate
            return decorate

    _module("vllm.model_executor.custom_op", PluggableLayer=PluggableLayer)
    _module("vllm.model_executor.determinism.batch_invariant",
            linear_batch_invariant=lambda *_: (_ for _ in ()).throw(AssertionError("not used on CPU")))
    _module("vllm.model_executor.layers.quantization",
            resolve_quant_method=lambda *_: (_ for _ in ()).throw(AssertionError("no quantization in tests")))
    _module("vllm.model_executor.layers.quantization.base_config",
            QuantizationConfig=type("QuantizationConfig", (), {}),
            QuantizeMethodBase=type("QuantizeMethodBase", (), {}))
    _module("vllm.model_executor.layers.utils",
            dispatch_unquantized_gemm=lambda backend: (
                lambda layer, x, weight, bias: __import__("torch").nn.functional.linear(x, weight, bias)),
            dispatch_cpu_unquantized_gemm=lambda *_: None)
    _module("vllm.model_executor.utils", set_weight_attrs=lambda obj, attrs: (
        [setattr(obj, key, value) for key, value in attrs.items()] and None))
    _module("vllm.model_executor.weight_transfer", copy_weight=lambda target, source: target.copy_(source))
    _module("vllm.platforms", current_platform=SimpleNamespace(
        is_cuda=lambda: False, is_cuda_alike=lambda: False, is_cpu=lambda: False,
        is_xpu=lambda: False, is_device_capability=lambda *_: False,
        is_device_capability_family=lambda *_: False,
        use_sync_weight_loader=lambda: False,
        make_synced_weight_loader=lambda loader: loader))
    for name in ("vllm.model_executor", "vllm.model_executor.layers"):
        _module(name)


def _load(name: str, relative: str) -> types.ModuleType:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; {HELP}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_image_modules():
    """Load the image's parameter and linear modules; refuse an unpinned file."""
    if ROOT is None:
        raise RuntimeError(HELP)
    import glm53full_speedups as plugin

    _stub_environ()
    expected = {}
    for check in plugin.FILE_CHECKS:
        if check.path in (LINEAR_PATH.removeprefix("vllm/"), PARAMETER_PATH.removeprefix("vllm/")):
            expected["vllm/" + check.path] = check.sha256
    for relative, sha256 in expected.items():
        found = _digest(ROOT / relative)
        if found != sha256:
            raise RuntimeError(f"{ROOT / relative} has SHA-256 {found}, expected {sha256} "
                               f"(the image sources are not the ones the plugin pins)")
    parameter = _load("vllm.model_executor.parameter", PARAMETER_PATH)
    linear = _load("vllm.model_executor.layers.linear", LINEAR_PATH)
    return parameter, linear


DEEPSEEK_V2_PATH = "vllm/model_executor/models/deepseek_v2.py"
FUSED_CLASS = "DeepSeekV2FusedQkvAProjLinear"


_FUSED_CLASS = None


def fused_qkv_a_proj_class():
    global _FUSED_CLASS
    if _FUSED_CLASS is not None:
        return _FUSED_CLASS
    """The image's ``DeepSeekV2FusedQkvAProjLinear``, compiled from its pinned file.

    The whole ``deepseek_v2`` module needs the model's dependencies; the class
    needs only the linear module's namespace, where its base class lives. Its
    compiled methods must equal the ones compiled from the whole file, so the
    extraction cannot drift from the pinned source.
    """
    import ast

    import glm53full_speedups as plugin

    path = ROOT / DEEPSEEK_V2_PATH
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    expected = next(check.sha256 for check in plugin.FILE_CHECKS
                    if check.path == "model_executor/models/deepseek_v2.py")
    if digest != expected:
        raise RuntimeError(f"{path} has SHA-256 {digest}, expected {expected}")
    tree = ast.parse(text, filename=str(path))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == FUSED_CLASS]
    if len(classes) != 1:
        raise RuntimeError(f"{FUSED_CLASS} not found once in {path}")
    source = ast.get_source_segment(text, classes[0])
    # The class's own line numbers, so its compiled methods equal the ones in
    # the whole-file compile (the code objects compare line tables too).
    # CPython compiles ``name.attr(...)`` differently when ``name`` is bound
    # by a module-level import, so the extraction compiles with the module's
    # own import statements (never executed).
    imports = "\n".join(ast.unparse(node) for node in tree.body
                        if isinstance(node, (ast.Import, ast.ImportFrom)))
    padded = "\n" * (classes[0].lineno - 1) + source + "\n" + imports + "\n"
    whole = compile(text, str(path), "exec")
    namespace = dict(vars(linear))
    namespace["QuantizationConfig"] = sys.modules[
        "vllm.model_executor.layers.quantization.base_config"].QuantizationConfig
    # The module's imports run with a stand-in ``__import__`` (the class body
    # binds only names), so the created class's methods compile exactly as
    # they do in the whole-file compile.
    import builtins

    class _AnyModule(types.ModuleType):
        def __getattr__(self, name):
            value = type(name, (), {})
            setattr(self, name, value)
            return value

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = sys.modules.get(name)
        if module is None:
            module = _AnyModule(name)
            sys.modules[name] = module
        parent, _, leaf = name.rpartition(".")
        if parent and parent in sys.modules and not hasattr(sys.modules[parent], leaf):
            setattr(sys.modules[parent], leaf, module)
        if not hasattr(module, leaf):  # ``import a.b as c`` reads ``a.b.b``
            setattr(module, leaf, module)
        for item in fromlist or ():
            if item != "*" and not hasattr(module, item):
                setattr(module, item, _AnyModule(f"{name}.{item}"))
        return module

    namespace["__builtins__"] = dict(vars(builtins), __import__=_fake_import)
    exec(compile(padded, str(path), "exec"), namespace)
    namespace["__builtins__"] = vars(builtins)
    extracted = namespace[FUSED_CLASS]
    for method in ("__init__", "forward"):
        qualname = f"{FUSED_CLASS}.{method}"
        assert getattr(extracted, method).__code__ == _find_code(whole, qualname), qualname
    _FUSED_CLASS = extracted
    return extracted


def _find_code(code, qualname):
    """The compiled code object of ``qualname`` inside ``code``, or None."""
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            if const.co_qualname == qualname:
                return const
            found = _find_code(const, qualname)
            if found is not None:
                return found
    return None


def load() -> None:
    """Load the image's modules (``conftest.py`` calls this once per session)."""
    global parameter, linear
    if "vllm.model_executor.layers.linear" not in sys.modules:
        parameter, linear = load_image_modules()


parameter = None
linear = None
