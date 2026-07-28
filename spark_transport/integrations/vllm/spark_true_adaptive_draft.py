"""Generate only the speculative depth selected for the next scheduler step.

The deployed V2/async GLM-5.2 runtime adapts target verification depth, but its
autoregressive MTP speculator still generates the configured maximum number of
draft tokens on every step.  This opt-in adapter binds proposal generation to
``SchedulerOutput.num_spec_tokens_to_schedule``.

The scheduler field is the correct value for proposals produced by the current
worker invocation: ``AsyncScheduler._update_after_schedule`` uses the same
value to create the placeholder list consumed on the *next* target step.
The deployed V2 runner splits target ``execute_model`` from
``sample_tokens``/proposal, so the adapter carries one fail-closed pending
depth across that exact boundary and rejects an overlapping execute. A depth
change therefore cannot make the scheduler request more tokens than the
worker generated.

This module is inert unless ``VLLM_SPARK_TRUE_ADAPTIVE_DRAFT=1``.  Installation
is fail closed:

* only the source hashes audited for the deployed Eldritch/B12X runtime are
  accepted;
* adaptive window 32, ladder 2,4, and maximum depth 4 must be attested;
* adaptive proposal depths must be in ``[1, 4]``;
* scheduler K0/``None`` prefill or non-drafted work remains on the stock path;
* dummy/profile graph work retains the configured maximum depth;
* unused persistent draft columns are overwritten with ``-1``;
* the scheduler-facing draft-token handler receives only the selected prefix.

The implementation deliberately does not patch scheduler policy.  Controller
status/reset is a separate surface so both changes can be admitted and reverted
independently.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import logging
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

_ENABLE_ENV = "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT"
_MAXIMUM_ENV = "VLLM_SPARK_MTP_TOKENS"
_WINDOW_ENV = "VLLM_SPARK_MTP_ADAPTIVE_WINDOW"
_DEPTHS_ENV = "VLLM_ADAPTIVE_SPEC_DEPTHS"
_RUNNER_SELECTED_ATTR = "_spark_pending_proposal_steps"
_SELECTED_ATTR = "_spark_selected_proposal_steps"
_HANDLER_SELECTED_ATTR = "_spark_selected_handler_steps"
_PATCH_MARKER = "_spark_true_adaptive_draft"
_INDEX_REUSE_MARKER = "_spark_glm52_mtp_index_reuse_wrapper"

_EXPECTED_SOURCE_SHA256 = {
    "model_runner": "9b3c111c6544ca30c62b0ab7982318367cfbbc46a929a4cc7cfbee3040f4de59",
    "autoregressive_speculator": (
        "dad44b274e2a2fd8ab2196bcaaec5989b24359cb149cc2dbf879a25f90894cad"
    ),
    "mtp_speculator": (
        "431517c1f397433b488a2cba6d28fe99ca9ef33d8ed47aebc2c71ef860d486a8"
    ),
    "draft_utils": "e34e8ccb8ad7091fad09392d14d12190909d44dc2a553adafe058e660579fce2",
}

_EXPECTED_EXECUTE_PARAMETERS = (
    "self",
    "scheduler_output",
    "intermediate_tensors",
    "dummy_run",
    "skip_attn_for_dummy_run",
    "is_profile",
)
_EXPECTED_PROPOSE_PREFIX = (
    "self",
    "input_batch",
    "attn_metadata",
    "slot_mappings",
)
_EXPECTED_SAMPLE_PARAMETERS = ("self", "grammar_output")
_EXPECTED_HANDLER_PARAMETERS = ("self", "input_batch", "draft_tokens")

_lock = threading.Lock()
_installed = False
_execute_calls = 0
_sample_calls = 0
_proposal_calls = 0
_handler_calls = 0
_proposal_requests_by_depth: Counter[int] = Counter()
_proposal_batches_by_depth: Counter[int] = Counter()
_saved_draft_steps = 0
_failures = 0
_patched_originals: tuple[
    type[Any],
    Callable[..., Any],
    Callable[..., Any],
    type[Any],
    Callable[..., Any],
    type[Any],
    Callable[..., Any],
] | None = None


def _strict_enabled() -> bool:
    value = os.getenv(_ENABLE_ENV, "0")
    if value not in {"0", "1"}:
        raise RuntimeError(f"{_ENABLE_ENV} must be exactly 0 or 1, got {value!r}")
    return value == "1"


def _attested_contract() -> tuple[int, int, tuple[int, ...]]:
    try:
        maximum = int(os.getenv(_MAXIMUM_ENV, "0"))
        window = int(os.getenv(_WINDOW_ENV, "0"))
        depths = tuple(
            int(item.strip())
            for item in os.getenv(_DEPTHS_ENV, "").split(",")
            if item.strip()
        )
    except ValueError as error:
        raise RuntimeError("adaptive-draft contract contains a non-integer") from error
    if maximum != 4:
        raise RuntimeError(f"{_MAXIMUM_ENV} must attest 4, got {maximum}")
    if window != 32:
        raise RuntimeError(f"{_WINDOW_ENV} must attest 32, got {window}")
    if depths != (2, 4):
        raise RuntimeError(f"{_DEPTHS_ENV} must attest exactly 2,4, got {depths}")
    return maximum, window, depths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(module: Any, expected: str, label: str) -> None:
    source_path = Path(inspect.getsourcefile(module) or "")
    if not source_path.is_file():
        raise RuntimeError(f"{label} source path is unavailable: {source_path}")
    actual = _sha256(source_path)
    if actual != expected:
        raise RuntimeError(
            f"{label} source SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _require_parameters(
    function: Callable[..., Any],
    expected: tuple[str, ...],
    label: str,
    *,
    prefix: bool = False,
) -> None:
    actual = tuple(inspect.signature(function).parameters)
    matches = actual[: len(expected)] == expected if prefix else actual == expected
    if not matches:
        qualifier = "prefix " if prefix else ""
        raise RuntimeError(
            f"{label} signature mismatch: expected {qualifier}{expected}, got {actual}"
        )


def _selected_steps(owner: Any, maximum: int) -> int:
    selected = getattr(owner, _SELECTED_ATTR, maximum)
    if type(selected) is not int or not 1 <= selected <= maximum:
        raise RuntimeError(
            f"selected adaptive draft depth must be an integer in [1,{maximum}], "
            f"got {selected!r}"
        )
    return selected


def _row_count(tensor: Any) -> int:
    shape = getattr(tensor, "shape", ())
    if not shape:
        raise RuntimeError("speculator returned a tensor without a row dimension")
    rows = int(shape[0])
    if rows < 0:
        raise RuntimeError(f"speculator returned an invalid row count: {rows}")
    return rows


def _overwrite_unused_columns(tensor: Any, selected: int, maximum: int) -> None:
    if selected >= maximum:
        return
    tail = tensor[:, selected:maximum]
    fill = getattr(tail, "fill_", None)
    if callable(fill):
        fill(-1)
    else:
        tail[...] = -1


def _record_failure() -> None:
    global _failures
    with _lock:
        _failures += 1


def _patch_classes(
    model_runner_class: type[Any],
    speculator_class: type[Any],
    handler_class: type[Any],
) -> None:
    """Patch the three narrow seams; separated for CPU-only unit tests."""

    global _installed
    global _patched_originals
    with _lock:
        if _installed:
            return

    original_execute = model_runner_class.execute_model
    original_sample_tokens = model_runner_class.sample_tokens
    original_propose = speculator_class.propose
    original_set_draft_tokens = handler_class.set_draft_tokens

    for function, label in (
        (original_execute, "GPUModelRunner.execute_model"),
        (original_sample_tokens, "GPUModelRunner.sample_tokens"),
        (original_propose, "MTPSpeculator.propose"),
        (original_set_draft_tokens, "DraftTokensHandler.set_draft_tokens"),
    ):
        if getattr(function, _PATCH_MARKER, False):
            raise RuntimeError(f"{label} is already patched by adaptive drafting")

    _require_parameters(
        original_execute,
        _EXPECTED_EXECUTE_PARAMETERS,
        "GPUModelRunner.execute_model",
    )
    _require_parameters(
        original_propose,
        _EXPECTED_PROPOSE_PREFIX,
        "MTPSpeculator.propose",
        prefix=True,
    )
    _require_parameters(
        original_sample_tokens,
        _EXPECTED_SAMPLE_PARAMETERS,
        "GPUModelRunner.sample_tokens",
    )
    _require_parameters(
        original_set_draft_tokens,
        _EXPECTED_HANDLER_PARAMETERS,
        "DraftTokensHandler.set_draft_tokens",
    )

    @functools.wraps(original_execute)
    def execute_model(
        self: Any,
        scheduler_output: Any,
        intermediate_tensors: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> Any:
        global _execute_calls
        speculator = getattr(self, "speculator", None)
        if speculator is None or dummy_run or is_profile:
            return original_execute(
                self,
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )
        if int(getattr(scheduler_output, "total_num_scheduled_tokens", 1)) <= 0:
            return original_execute(
                self,
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )

        maximum = int(getattr(self, "num_speculative_steps", 0))
        if maximum != 4:
            _record_failure()
            raise RuntimeError(
                "true adaptive drafting requires model-runner maximum depth 4, "
                f"got {maximum}"
            )
        if hasattr(self, _RUNNER_SELECTED_ATTR):
            _record_failure()
            raise RuntimeError(
                "true adaptive drafting found an unconsumed execute/sample "
                "depth; overlapping runner batches are unsupported"
            )
        selected = getattr(scheduler_output, "num_spec_tokens_to_schedule", None)
        if selected is None or (type(selected) is int and selected == 0):
            # Chunked prefill and explicitly non-drafted scheduler work do not
            # own a proposal depth.  Preserve the pinned runtime's stock
            # behavior and, critically, do not leave pending state for the
            # following decode step.
            return original_execute(
                self,
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )
        if type(selected) is not int or not 1 <= selected <= maximum:
            _record_failure()
            raise RuntimeError(
                "SchedulerOutput.num_spec_tokens_to_schedule must be an integer "
                f"in [1,{maximum}], got {selected!r}"
            )

        setattr(self, _RUNNER_SELECTED_ATTR, selected)
        with _lock:
            _execute_calls += 1
        try:
            result = original_execute(
                self,
                scheduler_output,
                intermediate_tensors,
                dummy_run,
                skip_attn_for_dummy_run,
                is_profile,
            )
            if result is not None:
                delattr(self, _RUNNER_SELECTED_ATTR)
            return result
        except Exception:
            if hasattr(self, _RUNNER_SELECTED_ATTR):
                delattr(self, _RUNNER_SELECTED_ATTR)
            _record_failure()
            raise

    @functools.wraps(original_sample_tokens)
    def sample_tokens(self: Any, grammar_output: Any) -> Any:
        global _sample_calls
        selected = getattr(self, _RUNNER_SELECTED_ATTR, None)
        if selected is None:
            # Dummy/profile and non-speculative paths retain stock behavior.
            return original_sample_tokens(self, grammar_output)

        speculator = getattr(self, "speculator", None)
        handler = getattr(self, "draft_tokens_handler", None)
        if speculator is None or handler is None:
            delattr(self, _RUNNER_SELECTED_ATTR)
            _record_failure()
            raise RuntimeError(
                "true adaptive drafting lost its speculator or draft handler "
                "between execute_model and sample_tokens"
            )
        previous_speculator = getattr(speculator, _SELECTED_ATTR, None)
        previous_handler = getattr(handler, _HANDLER_SELECTED_ATTR, None)
        setattr(speculator, _SELECTED_ATTR, selected)
        setattr(handler, _HANDLER_SELECTED_ATTR, selected)
        with _lock:
            _sample_calls += 1
        try:
            return original_sample_tokens(self, grammar_output)
        except Exception:
            _record_failure()
            raise
        finally:
            delattr(self, _RUNNER_SELECTED_ATTR)
            if previous_speculator is None:
                delattr(speculator, _SELECTED_ATTR)
            else:
                setattr(speculator, _SELECTED_ATTR, previous_speculator)
            if previous_handler is None:
                delattr(handler, _HANDLER_SELECTED_ATTR)
            else:
                setattr(handler, _HANDLER_SELECTED_ATTR, previous_handler)

    @functools.wraps(original_propose)
    def propose(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _proposal_calls, _saved_draft_steps
        maximum = int(getattr(self, "num_speculative_steps", 0))
        if maximum != 4:
            _record_failure()
            raise RuntimeError(
                f"true adaptive drafting expected speculator depth 4, got {maximum}"
            )
        selected = _selected_steps(self, maximum)
        previous = self.num_speculative_steps
        try:
            self.num_speculative_steps = selected
            result = original_propose(self, *args, **kwargs)
            rows = _row_count(result)
            full = self.draft_tokens[:rows]
            _overwrite_unused_columns(full, selected, maximum)
            with _lock:
                _proposal_calls += 1
                _proposal_batches_by_depth[selected] += 1
                _proposal_requests_by_depth[selected] += rows
                _saved_draft_steps += rows * (maximum - selected)
            # GPUModelRunner currently accepts fixed-width or empty output.  The
            # scheduler-facing handler is independently sliced below.
            return full
        except Exception:
            _record_failure()
            raise
        finally:
            self.num_speculative_steps = previous

    @functools.wraps(original_set_draft_tokens)
    def set_draft_tokens(
        self: Any, input_batch: Any, draft_tokens: Any
    ) -> Any:
        global _handler_calls
        selected = getattr(self, _HANDLER_SELECTED_ATTR, None)
        if selected is None:
            return original_set_draft_tokens(self, input_batch, draft_tokens)
        maximum = int(draft_tokens.shape[1])
        if type(selected) is not int or not 1 <= selected <= maximum:
            _record_failure()
            raise RuntimeError(
                f"draft-token handler selected width is invalid: {selected!r}"
            )
        with _lock:
            _handler_calls += 1
        return original_set_draft_tokens(
            self, input_batch, draft_tokens[:, :selected]
        )

    setattr(execute_model, _PATCH_MARKER, True)
    setattr(sample_tokens, _PATCH_MARKER, True)
    setattr(propose, _PATCH_MARKER, True)
    setattr(set_draft_tokens, _PATCH_MARKER, True)
    model_runner_class.execute_model = execute_model
    model_runner_class.sample_tokens = sample_tokens
    speculator_class.propose = propose
    handler_class.set_draft_tokens = set_draft_tokens
    with _lock:
        _patched_originals = (
            model_runner_class,
            original_execute,
            original_sample_tokens,
            speculator_class,
            original_propose,
            handler_class,
            original_set_draft_tokens,
        )
        _installed = True


def install() -> bool:
    """Install the source-pinned adapter when explicitly enabled."""

    if not _strict_enabled():
        return False
    _attested_contract()

    import vllm.v1.worker.gpu.model_runner as model_runner_module
    import vllm.v1.worker.gpu.spec_decode.autoregressive.speculator as autoregressive_module
    import vllm.v1.worker.gpu.spec_decode.mtp.speculator as mtp_module
    import vllm.v1.worker.gpu.spec_decode.utils as draft_utils_module

    _verify_source(
        model_runner_module,
        _EXPECTED_SOURCE_SHA256["model_runner"],
        "GPU model runner",
    )
    _verify_source(
        autoregressive_module,
        _EXPECTED_SOURCE_SHA256["autoregressive_speculator"],
        "autoregressive speculator",
    )
    _verify_source(
        mtp_module,
        _EXPECTED_SOURCE_SHA256["mtp_speculator"],
        "MTP speculator",
    )
    _verify_source(
        draft_utils_module,
        _EXPECTED_SOURCE_SHA256["draft_utils"],
        "draft-token utilities",
    )

    if not getattr(mtp_module.MTPSpeculator.propose, _INDEX_REUSE_MARKER, False):
        raise RuntimeError(
            "true adaptive drafting requires the GLM-5.2 MTP index-reuse "
            "proposer to be installed first"
        )

    _patch_classes(
        model_runner_module.GPUModelRunner,
        mtp_module.MTPSpeculator,
        draft_utils_module.DraftTokensHandler,
    )
    LOGGER.warning(
        "Spark true adaptive drafting installed: maximum=4 depths=2,4 window=32"
    )
    return True


def true_adaptive_draft_snapshot() -> dict[str, object]:
    with _lock:
        hook_ownership = {
            "execute_model": False,
            "sample_tokens": False,
            "propose": False,
            "set_draft_tokens": False,
        }
        if _patched_originals is not None:
            (
                model_runner_class,
                _,
                _,
                speculator_class,
                _,
                handler_class,
                _,
            ) = _patched_originals
            hook_ownership = {
                "execute_model": bool(
                    getattr(
                        model_runner_class.execute_model,
                        _PATCH_MARKER,
                        False,
                    )
                ),
                "sample_tokens": bool(
                    getattr(
                        model_runner_class.sample_tokens,
                        _PATCH_MARKER,
                        False,
                    )
                ),
                "propose": bool(
                    getattr(speculator_class.propose, _PATCH_MARKER, False)
                ),
                "set_draft_tokens": bool(
                    getattr(
                        handler_class.set_draft_tokens,
                        _PATCH_MARKER,
                        False,
                    )
                ),
            }
        owns_all_hooks = all(hook_ownership.values())
        return {
            "schema": "sparkring-true-adaptive-draft/v1",
            "enabled": os.getenv(_ENABLE_ENV, "0") == "1",
            "installed": bool(_installed and owns_all_hooks),
            "hook_ownership": hook_ownership,
            "execute_calls": _execute_calls,
            "sample_calls": _sample_calls,
            "proposal_calls": _proposal_calls,
            "handler_calls": _handler_calls,
            "proposal_batches_by_depth": {
                str(depth): count
                for depth, count in sorted(_proposal_batches_by_depth.items())
            },
            "proposal_requests_by_depth": {
                str(depth): count
                for depth, count in sorted(_proposal_requests_by_depth.items())
            },
            "saved_draft_steps": _saved_draft_steps,
            "failures": _failures,
            "source_sha256": dict(_EXPECTED_SOURCE_SHA256),
        }


def _reset_for_tests() -> None:
    global _installed
    global _patched_originals
    global _execute_calls
    global _sample_calls
    global _proposal_calls
    global _handler_calls
    global _saved_draft_steps
    global _failures
    with _lock:
        if _patched_originals is not None:
            (
                model_runner_class,
                original_execute,
                original_sample_tokens,
                speculator_class,
                original_propose,
                handler_class,
                original_set_draft_tokens,
            ) = _patched_originals
            model_runner_class.execute_model = original_execute
            model_runner_class.sample_tokens = original_sample_tokens
            speculator_class.propose = original_propose
            handler_class.set_draft_tokens = original_set_draft_tokens
            _patched_originals = None
        _installed = False
        _execute_calls = 0
        _sample_calls = 0
        _proposal_calls = 0
        _handler_calls = 0
        _proposal_requests_by_depth.clear()
        _proposal_batches_by_depth.clear()
        _saved_draft_steps = 0
        _failures = 0
