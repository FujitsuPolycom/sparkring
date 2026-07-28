"""Opt-in, fail-closed GLM-5.2 MTP index reuse for the vLLM V2 runner.

The checkpoint contract ``index_share_for_mtp_iteration=true`` means that
serial MTP step 0 computes sparse-attention top-k indices and later MTP steps
reuse that buffer.  The installed legacy proposer implements that contract,
but the V2 ``MTPSpeculator`` does not toggle ``set_skip_topk``.

Nothing is patched merely by importing this module.  Call :func:`install`
from a startup hook and set ``SPARK_GLM52_MTP_INDEX_REUSE=1``.  Unknown vLLM
versions or source fingerprints raise before any class is modified.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

_LOG = logging.getLogger("spark_glm52_mtp_index_reuse")
_ENABLE_ENV = "SPARK_GLM52_MTP_INDEX_REUSE"
_LOG_EVERY_ENV = "SPARK_GLM52_MTP_INDEX_REUSE_LOG_EVERY"
_DCP1_REMAP_ENV = "SPARK_B12X_DCP1_PHYSICAL_REMAP"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_MISSING = object()
_COMPOSITION_MARKER = "_spark_glm52_mtp_index_reuse_wrapper"


@dataclass(frozen=True)
class _RuntimeBindings:
    version: str
    mtp_speculator_cls: type
    autoregressive_cls: type
    deepseek_predictor_cls: type
    load_eagle_model: Callable[..., Any]
    mla_forward: Callable[..., Any]


@dataclass(frozen=True)
class _SourceContract:
    version: str
    fingerprints: Mapping[str, str]


@dataclass(frozen=True)
class InstallResult:
    status: str
    version: str | None
    reason: str


@dataclass
class _Counters:
    activations: int = 0
    buffer_validations: int = 0
    compute_arms: int = 0
    reuse_arms: int = 0
    proposals_completed: int = 0
    proposals_failed: int = 0
    logical_step0_compute_forwards: int = 0
    logical_reuse_forwards: int = 0
    prefills_failed: int = 0


_EXPECTED_VERSION = (
    "0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea."
    "fi25dd814.cu132.20260626"
)
_SUPPORTED_SOURCES: tuple[_SourceContract, ...] = (
    _SourceContract(
        version=_EXPECTED_VERSION,
        fingerprints={
            "MTPSpeculator": (
                "9924c6a2885bed9e856cdf27325df7bfc00930f9e71befb4c6eca7e4938760ce"
            ),
            "AutoRegressiveSpeculator._prefill": (
                "111c75e075362da8d417752d695c139b6a6a93b0e13c5caf36f2c07a149e599c"
            ),
            "AutoRegressiveSpeculator.capture": (
                "8fb3bc2a6332108b3d859688f8191afbd3fd95534fd02f7425d251622a36fffd"
            ),
            "AutoRegressiveSpeculator._multi_step_decode": (
                "0e7c45ef90ae463db0a3bd4a9142301c9a2261d11837fac1215eb538d4e1ac26"
            ),
            "AutoRegressiveSpeculator.propose": (
                "b19960593a8ee2ba33cb2a3782435fb55c7c02a29185d44cc39822c84213fcbb"
            ),
            "DeepSeekMultiTokenPredictor.set_skip_topk": (
                "a72bcc6a25e3d7a15b4fdf52da2e5f985abce9c9e251fcb474960e8d62332df5"
            ),
            "DeepSeekMultiTokenPredictor.forward": (
                "e333135529bb3dff29b230abd76d688c6b1fe1319b638b91035fd7f04ed5382d"
            ),
            "load_eagle_model": (
                "b042713de9011d397f814b3d14d9d7e348d2f8e27b33091025dfafbd99f41646"
            ),
            "MultiHeadLatentAttentionWrapper.forward": (
                "ae9b5068fc0115df2d93ef33713255662c97bd7f543cd4b67aa724d855d97392"
            ),
        },
    ),
)

_LOCK = threading.RLock()
_COUNTERS = _Counters()
_PATCH_STATE: tuple[type, dict[str, object]] | None = None


def _load_bindings() -> _RuntimeBindings:
    import vllm
    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
    from vllm.model_executor.models.deepseek_mtp import (
        DeepSeekMultiTokenPredictor,
    )
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    return _RuntimeBindings(
        version=vllm.__version__,
        mtp_speculator_cls=MTPSpeculator,
        autoregressive_cls=AutoRegressiveSpeculator,
        deepseek_predictor_cls=DeepSeekMultiTokenPredictor,
        load_eagle_model=load_eagle_model,
        mla_forward=MultiHeadLatentAttentionWrapper.forward,
    )


def _sha256_source(obj: object) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode("utf-8")).hexdigest()


def _fingerprint_bindings(bindings: _RuntimeBindings) -> dict[str, str]:
    auto = bindings.autoregressive_cls
    predictor = bindings.deepseek_predictor_cls
    return {
        "MTPSpeculator": _sha256_source(bindings.mtp_speculator_cls),
        "AutoRegressiveSpeculator._prefill": _sha256_source(auto._prefill),
        "AutoRegressiveSpeculator.capture": _sha256_source(auto.capture),
        "AutoRegressiveSpeculator._multi_step_decode": _sha256_source(
            auto._multi_step_decode
        ),
        "AutoRegressiveSpeculator.propose": _sha256_source(auto.propose),
        "DeepSeekMultiTokenPredictor.set_skip_topk": _sha256_source(
            predictor.set_skip_topk
        ),
        "DeepSeekMultiTokenPredictor.forward": _sha256_source(predictor.forward),
        "load_eagle_model": _sha256_source(bindings.load_eagle_model),
        "MultiHeadLatentAttentionWrapper.forward": _sha256_source(
            bindings.mla_forward
        ),
    }


def _matching_contract(
    bindings: _RuntimeBindings,
    contracts: Sequence[_SourceContract],
) -> tuple[_SourceContract, dict[str, str]]:
    actual = _fingerprint_bindings(bindings)
    for contract in contracts:
        if contract.version == bindings.version and dict(contract.fingerprints) == actual:
            return contract, actual

    expected_versions = sorted({contract.version for contract in contracts})
    mismatches: dict[str, dict[str, str | None]] = {}
    same_version = [
        contract for contract in contracts if contract.version == bindings.version
    ]
    if same_version:
        expected = same_version[0].fingerprints
        for name in sorted(set(expected) | set(actual)):
            if expected.get(name) != actual.get(name):
                mismatches[name] = {
                    "expected": expected.get(name),
                    "actual": actual.get(name),
                }
    raise RuntimeError(
        "unsupported vLLM source; refusing GLM-5.2 MTP index-reuse patch: "
        + json.dumps(
            {
                "actual_version": bindings.version,
                "expected_versions": expected_versions,
                "fingerprint_mismatches": mismatches,
            },
            sort_keys=True,
        )
    )


def check_compatibility() -> dict[str, Any]:
    """Return the installed source identity, raising if it is unsupported."""

    bindings = _load_bindings()
    _, actual = _matching_contract(bindings, _SUPPORTED_SOURCES)
    return {
        "status": "compatible",
        "version": bindings.version,
        "fingerprints": actual,
    }


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE_VALUES


def _architectures(config: object) -> tuple[str, ...]:
    value = getattr(config, "architectures", None)
    return tuple(value or ())


def _sharing_requested(speculator: object) -> bool:
    draft_config = getattr(
        getattr(speculator, "draft_model_config", None),
        "hf_config",
        None,
    )
    return getattr(draft_config, "index_share_for_mtp_iteration", False) is True


def _validate_runtime_contract(
    speculator: object,
    _target_model: object,
    draft_model: object,
) -> object:
    vllm_config = getattr(speculator, "vllm_config", None)
    draft_config = getattr(speculator.draft_model_config, "hf_config", None)
    target_config = getattr(
        getattr(vllm_config, "model_config", None),
        "hf_config",
        None,
    )
    speculative_config = getattr(vllm_config, "speculative_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)

    errors: list[str] = []
    if getattr(speculative_config, "method", None) != "mtp":
        errors.append("speculative method is not mtp")
    if _architectures(target_config) != ("GlmMoeDsaForCausalLM",):
        errors.append(
            f"target architecture is {_architectures(target_config)!r}, "
            "not ('GlmMoeDsaForCausalLM',)"
        )
    if _architectures(draft_config) != ("DeepSeekMTPModel",):
        errors.append(
            f"draft architecture is {_architectures(draft_config)!r}, "
            "not ('DeepSeekMTPModel',)"
        )
    if getattr(draft_config, "num_nextn_predict_layers", None) != 1:
        errors.append("num_nextn_predict_layers is not 1")
    index_topk = getattr(draft_config, "index_topk", None)
    if index_topk != 2048:
        errors.append(f"index_topk is {index_topk!r}, not 2048")
    dcp_size = getattr(parallel_config, "decode_context_parallel_size", None)
    if dcp_size not in (1, 2, 4):
        errors.append(
            f"decode_context_parallel_size is {dcp_size!r}, "
            "not 1, 2, or 4"
        )
    if dcp_size == 1 and not _env_true(_DCP1_REMAP_ENV):
        errors.append(
            f"{_DCP1_REMAP_ENV} is not enabled; DCP1 MTP reuse requires "
            "the exact-source B12X logical-to-local-physical top-k remap"
        )
    if not _env_true("VLLM_DCP_GLOBAL_TOPK"):
        errors.append("VLLM_DCP_GLOBAL_TOPK is not enabled")
    if not _env_true("VLLM_DCP_SHARD_DRAFT"):
        errors.append("VLLM_DCP_SHARD_DRAFT is not enabled")

    draft_inner = getattr(draft_model, "model", None)
    if draft_inner is None:
        errors.append("draft model has no .model")
    if draft_inner is not None and not callable(
        getattr(draft_inner, "set_skip_topk", None)
    ):
        errors.append("draft model has no callable set_skip_topk")

    buffer_holders: list[tuple[str, object]] = []
    if draft_inner is not None and hasattr(draft_inner, "named_modules"):
        for name, module in draft_inner.named_modules():
            if hasattr(module, "topk_indices_buffer"):
                buffer_holders.append(
                    (name or "<draft-root>", module.topk_indices_buffer)
                )
    if not buffer_holders:
        errors.append("draft has no topk_indices_buffer holders")
    else:
        missing = [
            name for name, buffer in buffer_holders if buffer is None
        ]
        if missing:
            errors.append(
                "draft topk_indices_buffer is None at "
                + ", ".join(missing[:8])
            )
        identities: dict[int, list[str]] = {}
        buffers_by_identity: dict[int, object] = {}
        for name, buffer in buffer_holders:
            if buffer is not None:
                identity = id(buffer)
                identities.setdefault(identity, []).append(name)
                buffers_by_identity[identity] = buffer
        if len(identities) != 1:
            groups = [
                ",".join(names[:4])
                for names in identities.values()
            ]
            errors.append(
                "draft topk_indices_buffer holders do not have one shared "
                f"identity: {groups!r}"
            )
        elif identities:
            draft_buffer = buffers_by_identity[next(iter(identities))]
            shape = getattr(draft_buffer, "shape", None)
            if shape is None:
                errors.append("draft topk_indices_buffer has no shape")
            elif len(shape) != 2 or int(shape[-1]) != index_topk:
                errors.append(
                    f"draft topk_indices_buffer shape is {tuple(shape)!r}; "
                    f"expected [tokens, {index_topk}]"
                )
            dtype = getattr(draft_buffer, "dtype", None)
            if dtype is not None and not str(dtype).endswith("int32"):
                errors.append(
                    f"draft topk_indices_buffer dtype is {dtype!r}, not int32"
                )

    if errors:
        raise RuntimeError(
            "GLM-5.2 MTP index reuse runtime contract failed: "
            + "; ".join(errors)
        )
    return draft_inner


def _skip_modules(draft_inner: object) -> list[object]:
    modules: list[object] = []
    for _, module in draft_inner.named_modules():
        if hasattr(module, "skip_topk"):
            modules.append(module)
    return modules


def _arm(speculator: object, skip: bool) -> None:
    draft_inner = getattr(speculator, "_spark_mtp_index_reuse_model", None)
    if draft_inner is None:
        raise RuntimeError("MTP index reuse was not activated for this speculator")
    draft_inner.set_skip_topk(skip)
    modules = _skip_modules(draft_inner)
    if not modules:
        raise RuntimeError("set_skip_topk reached no skip_topk-bearing modules")
    if any(bool(module.skip_topk) is not skip for module in modules):
        raise RuntimeError("set_skip_topk did not update every sparse MTP module")
    with _LOCK:
        if skip:
            _COUNTERS.reuse_arms += 1
        else:
            _COUNTERS.compute_arms += 1


def _activate_speculator(
    speculator: object,
    target_model: object,
    draft_model: object,
) -> None:
    draft_inner = _validate_runtime_contract(speculator, target_model, draft_model)
    speculator._spark_mtp_index_reuse_model = draft_inner
    speculator._spark_mtp_index_reuse_active = True
    _arm(speculator, False)
    with _LOCK:
        _COUNTERS.activations += 1
        _COUNTERS.buffer_validations += 1
    _LOG.warning(
        "Activated GLM-5.2 MTP index reuse: "
        "step 0 computes sparse top-k; serial steps 1+ reuse it"
    )


def _stats_after_proposal(num_steps: int) -> None:
    with _LOCK:
        _COUNTERS.proposals_completed += 1
        _COUNTERS.logical_step0_compute_forwards += 1
        _COUNTERS.logical_reuse_forwards += max(num_steps - 1, 0)
        completed = _COUNTERS.proposals_completed
        snapshot = asdict(_COUNTERS)
    raw_every = os.getenv(_LOG_EVERY_ENV, "0").strip()
    try:
        every = int(raw_every)
    except ValueError:
        every = 0
    if every > 0 and completed % every == 0:
        _LOG.info("GLM-5.2 MTP index reuse stats: %s", json.dumps(snapshot))


def _install(
    bindings: _RuntimeBindings,
    contracts: Sequence[_SourceContract],
) -> InstallResult:
    """Internal injectable install seam used by GPU-free tests."""

    global _PATCH_STATE
    with _LOCK:
        if _PATCH_STATE is not None:
            return InstallResult(
                status="already-installed",
                version=bindings.version,
                reason="class patch is already active",
            )

        _matching_contract(bindings, contracts)
        mtp_cls = bindings.mtp_speculator_cls
        auto_cls = bindings.autoregressive_cls
        original_load = mtp_cls.load_draft_model
        original_prefill = auto_cls._prefill
        original_propose = auto_cls.propose

        def load_draft_model(self, target_model, target_attn_layer_names):
            draft_model = original_load(
                self,
                target_model,
                target_attn_layer_names,
            )
            self._spark_mtp_index_reuse_active = False
            if _sharing_requested(self):
                _activate_speculator(self, target_model, draft_model)
            return draft_model

        def _prefill(self, *args, **kwargs):
            if not getattr(self, "_spark_mtp_index_reuse_active", False):
                return original_prefill(self, *args, **kwargs)
            _arm(self, False)
            try:
                result = original_prefill(self, *args, **kwargs)
            except BaseException:
                with _LOCK:
                    _COUNTERS.prefills_failed += 1
                _arm(self, False)
                raise
            _arm(self, True)
            return result

        @functools.wraps(original_propose)
        def propose(self, *args, **kwargs):
            active = getattr(self, "_spark_mtp_index_reuse_active", False)
            try:
                result = original_propose(self, *args, **kwargs)
            except BaseException:
                if active:
                    with _LOCK:
                        _COUNTERS.proposals_failed += 1
                    # A failure after step 0 may leave physical draft-cache
                    # indices armed. No unrelated/profile forward may consume
                    # that request-specific buffer.
                    _arm(self, False)
                raise
            if active:
                _stats_after_proposal(int(self.num_speculative_steps))
                # Reuse is scoped strictly to the serial steps inside this
                # proposal. Startup profiling, prewarming, and other direct
                # model forwards can run before the next _prefill wrapper.
                _arm(self, False)
            return result

        load_draft_model.__name__ = original_load.__name__
        load_draft_model.__doc__ = original_load.__doc__
        _prefill.__name__ = original_prefill.__name__
        _prefill.__doc__ = original_prefill.__doc__
        setattr(propose, _COMPOSITION_MARKER, True)

        previous: dict[str, object] = {
            "load_draft_model": mtp_cls.__dict__.get(
                "load_draft_model",
                _MISSING,
            ),
            "_prefill": mtp_cls.__dict__.get("_prefill", _MISSING),
            "propose": mtp_cls.__dict__.get("propose", _MISSING),
        }
        # Assignment happens only after every source fingerprint has matched.
        mtp_cls.load_draft_model = load_draft_model
        mtp_cls._prefill = _prefill
        mtp_cls.propose = propose
        _PATCH_STATE = (mtp_cls, previous)

    _LOG.warning(
        "Installed GLM-5.2 V2 MTP index reuse for exact vLLM version %s",
        bindings.version,
    )
    return InstallResult(
        status="installed",
        version=bindings.version,
        reason="exact source contract matched; runtime model gate pending",
    )


def install() -> InstallResult:
    """Install the opt-in patch after exact version/source verification."""

    if not _env_true(_ENABLE_ENV):
        return InstallResult(
            status="disabled",
            version=None,
            reason=f"set {_ENABLE_ENV}=1 to opt in",
        )
    return _install(_load_bindings(), _SUPPORTED_SOURCES)


def uninstall() -> bool:
    """Restore original class attributes; restart remains the primary rollback."""

    global _PATCH_STATE
    with _LOCK:
        if _PATCH_STATE is None:
            return False
        mtp_cls, previous = _PATCH_STATE
        for name, value in previous.items():
            if value is _MISSING:
                delattr(mtp_cls, name)
            else:
                setattr(mtp_cls, name, value)
        _PATCH_STATE = None
        return True


def get_stats() -> dict[str, int]:
    """Return process-local policy and logical-forward counters."""

    with _LOCK:
        return asdict(_COUNTERS)


def reset_stats() -> None:
    """Reset counters. Intended for tests and pre-traffic instrumentation."""

    with _LOCK:
        for name in _COUNTERS.__dataclass_fields__:
            setattr(_COUNTERS, name, 0)


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Check GLM-5.2 MTP index-reuse source compatibility"
    )
    parser.parse_args()
    print(json.dumps(check_compatibility(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
