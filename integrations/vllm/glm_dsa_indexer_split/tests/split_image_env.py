"""The CPU import environment for the tests: the image's own indexer modules.

``SPARKRING_GLM53_IMAGE_SOURCES`` names a directory with ``vllm/`` and
``b12x/`` subtrees holding the target image's Python sources (image
``sparkring-dev/kraken:csf-sircl-libsircl-20261008``, read-only extraction;
see the integration README). Without it every test that needs the image's
code fails with the instructions below.

``vllm`` and ``vllm.distributed`` stand in as plain modules; the two files
the plugin reads and wraps load from the image copy as the modules
``vllm.v1.attention.backends.mla.indexer`` and
``vllm.v1.attention.backends.mla.b12x_indexer``. Every stub name is a value
the tests or the stand-ins replace; the image's own functions the tests run
(``B12xSparseIndexer.forward``, ``._plan``, ``_merge_dcp_topk``, the chunk
splitters and the metadata dataclasses) need no CUDA and no b12x.

The loaded files' SHA-256 must equal the pins the plugin records, so every
test session replays those two pins.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

HELP = (
    "set SPARKRING_GLM53_IMAGE_SOURCES to a directory holding the target image's "
    "vllm/ and b12x/ Python sources (see integrations/vllm/glm_dsa_indexer_split/README.md)"
)

# The plugin registers enabled in conftest; every prefill step is eligible.
# Probes (fresh interpreters) set their flags explicitly instead.
if not os.environ.get("PROBE_NO_DEFAULTS"):
    os.environ.setdefault("GLM_DSA_INDEXER_SPLIT", "1")
    if os.environ["GLM_DSA_INDEXER_SPLIT"] == "1":
        os.environ.setdefault("GLM_DSA_INDEXER_SPLIT_MIN_ROWS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(os.environ["SPARKRING_GLM53_IMAGE_SOURCES"]).resolve() \
    if os.environ.get("SPARKRING_GLM53_IMAGE_SOURCES") else None

INDEXER_PATH = "vllm/v1/attention/backends/mla/indexer.py"
B12X_INDEXER_PATH = "vllm/v1/attention/backends/mla/b12x_indexer.py"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module(name: str, **attributes) -> types.ModuleType:
    """Register a stand-in module, keeping names another suite's stand-ins set."""
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        if "__path__" in attributes:  # a stand-in package: find_spec works on it
            from importlib.machinery import ModuleSpec

            spec = ModuleSpec(name, None, is_package=True)
            spec.submodule_search_locations = list(attributes["__path__"])
            module.__spec__ = spec
        sys.modules[name] = module
    for key, value in attributes.items():
        if not hasattr(module, key):
            setattr(module, key, value)
    return module


def _triton_jit(function=None, **_kwargs):
    """Identity stand-in: the tests replace every decorated kernel."""
    return function if function is not None else (lambda f: f)


def _stub_environ() -> None:
    """Register ``vllm`` and the names the two image files import.

    Names another test suite's stand-ins already set are kept; every missing
    one is added, so the environment is order-independent.
    """
    vllm = _module("vllm", __path__=[str(ROOT / "vllm")])
    _module("b12x", __path__=[str(ROOT / "b12x")])
    envs = _module("vllm.envs", VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=4 * 2200,
                   VLLM_DCP_INDEXER_KEY_GATHER=False, VLLM_BATCH_INVARIANT=False,
                   VLLM_XPU_FORCE_N_CONTIG_WEIGHT=False)
    _module("vllm.config", VllmConfig=type("VllmConfig", (), {}))
    distributed = _module("vllm.distributed",
                          get_dcp_group=lambda: (_ for _ in ()).throw(AssertionError("no DCP group stubbed")),
                          get_pcp_group=lambda: (_ for _ in ()).throw(AssertionError("no PCP group stubbed")))
    parallel_state = _module("vllm.distributed.parallel_state",
                             model_parallel_is_initialized=lambda: False,
                             get_tensor_model_parallel_world_size=lambda: 1)
    distributed.parallel_state = parallel_state
    _module("vllm.logger", init_logger=logging.getLogger)
    _module("vllm.forward_context",
            get_forward_context=lambda: (_ for _ in ()).throw(AssertionError("no forward context stubbed")))
    _module("vllm.platforms", current_platform=SimpleNamespace(
        is_cuda=lambda: False, is_cuda_alike=lambda: False, is_cpu=lambda: False,
        is_xpu=lambda: False, is_device_capability=lambda *_: False,
        is_device_capability_family=lambda *_: False))
    _module("vllm.triton_utils", tl=SimpleNamespace(constexpr=lambda value: value),
            triton=SimpleNamespace(jit=_triton_jit, cdiv=lambda x, y: -(-int(x) // int(y))))
    _module("vllm.utils")
    _module("vllm.utils.deep_gemm", get_paged_mqa_logits_metadata=lambda *_: None,
            has_deep_gemm=lambda: False, native_next_n_supported=lambda *_: False)
    _module("vllm.utils.math_utils", round_down=lambda value, align: value // align * align)
    _module("vllm.utils.platform_utils", num_compute_units=lambda: 1)
    _module("vllm.utils.torch_utils", PIN_MEMORY=False, async_tensor_h2d=lambda *_: None)
    _module("vllm.utils.b12x", set_b12x_preparation_provider=lambda *_: None,
            B12xPreparationUnit=type("B12xPreparationUnit", (), {}),
            B12xWorkload=type("B12xWorkload", (), {}),
            get_b12x_dsa_indexer=lambda *_: None)
    attention_backend = _module(
        "vllm.v1.attention.backend",
        AttentionBackend=type("AttentionBackend", (), {}),
        AttentionCGSupport=SimpleNamespace(ALWAYS="always", NEVER="never"),
        AttentionMetadataBuilder=type("AttentionMetadataBuilder", (), {}),
        CommonAttentionMetadata=type("CommonAttentionMetadata", (), {}),
        MultipleOf=type("MultipleOf", (), {}))
    for name in ("vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends",
                 "vllm.v1.attention.backends.mla", "vllm.v1.kv_cache_interface",
                 "vllm.v1.worker", "vllm.model_executor", "vllm.model_executor.warmup",
                 "vllm.model_executor.models", "vllm.model_executor.kernels",
                 "vllm.model_executor.kernels.attention", "vllm.model_executor.kernels.attention.dsa"):
        _module(name)
    _module("vllm.v1.attention.backends.mla.compressor_utils",
            get_compressed_slot_mapping=lambda *_: None)
    _module("vllm.v1.attention.backends.mla.sparse_utils", request_row_bounds=lambda *_: None)
    _module("vllm.v1.attention.backends.utils", get_dcp_local_seq_lens=lambda *_: None,
            refresh_dcp_local_seq_lens_=lambda *_: None, split_decodes_and_prefills=lambda *_: None)
    _module("vllm.v1.kv_cache_interface", AttentionSpec=type("AttentionSpec", (), {}),
            KVCacheLayout=type("KVCacheLayout", (), {}), KVCacheSpec=type("KVCacheSpec", (), {}),
            MLAAttentionSpec=type("MLAAttentionSpec", (), {}))
    _module("vllm.v1.worker.block_table", get_block_table_width=lambda *_: 1)
    _module("vllm.v1.worker.workspace", current_workspace_manager=lambda: None)
    _module("vllm.model_executor.warmup.jit_warmup", kernel_launcher=lambda *_: None,
            zip_inputs=lambda *_: None)
    _module("vllm.model_executor.warmup.jit_warmup_triton_helper",
            LaunchSpec=type("LaunchSpec", (), {}),
            TritonPointerInputVariant=type("TritonPointerInputVariant", (), {}),
            TritonWarmupTensor=type("TritonWarmupTensor", (), {}),
            VllmTritonJitKernel=type("VllmTritonJitKernel", (),
                                    {"__getitem__": lambda self, key: type(key, (), {})})(),
            triton_scalar_specialization_rep=lambda *_: None)
    _module("vllm.model_executor.models.deepseek_v2",
            DeepseekV32IndexerCache=type("DeepseekV32IndexerCache", (), {}))
    assert envs and attention_backend and vllm


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
    """Load the image's two indexer modules; refuse a file the plugin does not pin."""
    if ROOT is None:
        raise RuntimeError(HELP)
    import glm_dsa_indexer_split as plugin

    _stub_environ()
    expected = {B12X_INDEXER_PATH: plugin.INDEXER_SHA256,
                INDEXER_PATH: next(check.sha256 for check in plugin.FILE_CHECKS
                                   if check.path == "v1/attention/backends/mla/indexer.py")}
    for relative, sha256 in expected.items():
        found = _digest(ROOT / relative)
        if found != sha256:
            raise RuntimeError(f"{ROOT / relative} has SHA-256 {found}, expected {sha256} "
                               f"(the image sources are not the ones the plugin pins)")
    indexer = _load("vllm.v1.attention.backends.mla.indexer", INDEXER_PATH)
    b12x_indexer = _load("vllm.v1.attention.backends.mla.b12x_indexer", B12X_INDEXER_PATH)
    return indexer, b12x_indexer


# Importing this module loads the image's modules; every later
# ``import vllm.v1.attention.backends.mla.…`` in the tests finds them.
indexer, b12x_indexer = load_image_modules()
