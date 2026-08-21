"""Source-pinned ordinal context for real V2 draft-generation calls.

The deployed ``AutoRegressiveSpeculator._multi_step_decode`` owns the only
Python loop over draft positions.  Its two execution branches call exactly
one of:

* ``decode_cudagraph_manager.run_fullgraph``; or
* ``AutoRegressiveSpeculator._generate_draft``.

This module labels those *observed calls* with the loop ordinal without
reading ``current_draft_step`` from the GPU.  It deliberately does not call
either branch a complete draft step: attention-metadata preparation happens
earlier in the loop and has no enclosing callable seam in the deployed
source.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .phase_timing import PhaseDescriptor, PhaseKind
from .vllm_adapter import AdapterValidationError, source_sha256


def _descriptor(position: int, dispatch: str) -> PhaseDescriptor:
    return PhaseDescriptor(
        PhaseKind.DRAFT_GENERATION,
        f"position={position},dispatch={dispatch}",
    )


_UNSCOPED = PhaseDescriptor(
    PhaseKind.DRAFT_GENERATION,
    "position=unscoped,dispatch=eager",
)


@dataclass
class _ActiveLoop:
    speculator: object
    dispatch: str
    round_depth: int
    next_position: int = 1


class DraftGenerationOrdinals:
    """Bounded host-only state for one model-execution thread."""

    def __init__(
        self,
        expected_speculative_steps: int,
        *,
        attested_round_depths: tuple[int, ...] | None = None,
        adaptive_window: int | None = None,
    ) -> None:
        if (
            not isinstance(expected_speculative_steps, int)
            or isinstance(expected_speculative_steps, bool)
            or expected_speculative_steps < 2
        ):
            raise ValueError(
                "expected_speculative_steps must be an integer >= 2"
            )
        self.expected_speculative_steps = expected_speculative_steps
        if attested_round_depths is None:
            attested_round_depths = (expected_speculative_steps,)
        if (
            not attested_round_depths
            or tuple(sorted(set(attested_round_depths)))
            != attested_round_depths
            or any(
                isinstance(depth, bool)
                or not isinstance(depth, int)
                or not 2 <= depth <= expected_speculative_steps
                for depth in attested_round_depths
            )
            or expected_speculative_steps not in attested_round_depths
        ):
            raise ValueError(
                "attested_round_depths must be sorted unique depths in "
                "[2, expected_speculative_steps] and include the configured "
                "depth"
            )
        if len(attested_round_depths) > 1:
            if (
                isinstance(adaptive_window, bool)
                or not isinstance(adaptive_window, int)
                or adaptive_window < 1
            ):
                raise ValueError(
                    "adaptive depths require a positive adaptive_window"
                )
        elif adaptive_window is not None:
            raise ValueError(
                "fixed depth must not declare an adaptive_window"
            )
        self.attested_round_depths = attested_round_depths
        self.adaptive_window = adaptive_window
        self.loop_positions = tuple(range(1, expected_speculative_steps))
        self.descriptors = tuple(
            _descriptor(position, dispatch)
            for position in self.loop_positions
            for dispatch in ("full_graph", "eager")
        ) + (_UNSCOPED,)
        self._by_key = {
            (position, dispatch): _descriptor(position, dispatch)
            for position in self.loop_positions
            for dispatch in ("full_graph", "eager")
        }
        self._speculator: object | None = None
        self._decode_manager: object | None = None
        self._active: _ActiveLoop | None = None
        self._armed = False
        self._completed_rounds_by_depth = {
            depth: 0 for depth in attested_round_depths
        }

    def bind(self, speculator: object, decode_manager: object) -> None:
        """Bind the exact initialized objects before the recorder can arm."""
        if self._speculator is not None:
            if (
                self._speculator is not speculator
                or self._decode_manager is not decode_manager
            ):
                raise RuntimeError(
                    "draft-generation ordinals were rebound to new objects"
                )
            return
        actual = getattr(speculator, "num_speculative_steps", None)
        if actual != self.expected_speculative_steps:
            raise RuntimeError(
                "unexpected speculative depth: expected "
                f"{self.expected_speculative_steps}, got {actual!r}"
            )
        self._speculator = speculator
        self._decode_manager = decode_manager

    @staticmethod
    def _dispatch(batch_desc: object) -> str:
        mode = getattr(batch_desc, "cg_mode", None)
        return (
            "full_graph"
            if getattr(mode, "name", None) == "FULL"
            else "eager"
        )

    def enter(self, speculator: object, batch_desc: object) -> bool:
        """Enter the real loop; return false for pre-bind startup calls."""
        if self._active is not None:
            raise RuntimeError("nested draft-generation loop")
        round_depth = getattr(speculator, "num_speculative_steps", None)
        if round_depth not in self.attested_round_depths:
            raise RuntimeError(
                f"unattested adaptive draft depth {round_depth!r}; "
                f"expected one of {list(self.attested_round_depths)}"
            )
        if self._speculator is None:
            return False
        if self._speculator is not speculator:
            raise RuntimeError("unexpected speculator entered draft loop")
        self._active = _ActiveLoop(
            speculator=speculator,
            dispatch=self._dispatch(batch_desc),
            round_depth=round_depth,
        )
        return True

    def _claim(self, dispatch: str) -> PhaseDescriptor:
        active = self._active
        if active is None:
            if dispatch == "eager":
                # CUDA-graph capture calls _generate_draft directly before
                # initialize_kv_cache's post-call binding. It is unarmed and
                # is intentionally not assigned a fake position.
                return _UNSCOPED
            raise RuntimeError(
                "draft decode graph replay occurred outside its source-pinned "
                "loop"
            )
        if active.dispatch != dispatch:
            raise RuntimeError(
                f"draft loop selected {active.dispatch}, observed {dispatch}"
            )
        position = active.next_position
        if position not in self.loop_positions:
            raise RuntimeError("too many draft-generation calls in one loop")
        active.next_position += 1
        return self._by_key[(position, dispatch)]

    def eager_descriptor(
        self,
        speculator: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> PhaseDescriptor:
        del args, kwargs
        active = self._active
        if active is not None and active.speculator is not speculator:
            raise RuntimeError("unexpected speculator generated a draft")
        return self._claim("eager")

    def graph_descriptor(
        self, manager: object
    ) -> PhaseDescriptor | None:
        active = self._active
        if (
            active is None
            or self._decode_manager is None
            or manager is not self._decode_manager
        ):
            return None
        return self._claim("full_graph")

    def leave(self, speculator: object, *, validate: bool) -> None:
        active = self._active
        if active is None:
            return
        self._active = None
        if active.speculator is not speculator:
            raise RuntimeError("unexpected speculator left draft loop")
        if validate and active.next_position != active.round_depth:
            observed = active.next_position - 1
            raise RuntimeError(
                "draft loop completed with "
                f"{observed} generation calls; expected "
                f"{active.round_depth - 1}"
            )
        if validate and self._armed:
            self._completed_rounds_by_depth[active.round_depth] += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "configured_speculative_steps": self.expected_speculative_steps,
            "attested_round_depths": list(self.attested_round_depths),
            "adaptive_window": self.adaptive_window,
            "completed_rounds_by_depth": {
                str(depth): count
                for depth, count in self._completed_rounds_by_depth.items()
            },
            "position_zero": "draft_prefill",
            "expected_loop_positions": list(self.loop_positions),
            "complete_iteration": False,
            "excludes": ["per-position attention preparation"],
            "bound": self._speculator is not None,
        }

    def arm(self) -> None:
        if self._armed:
            raise RuntimeError("draft-generation ordinals are already armed")
        if self._active is not None:
            raise RuntimeError("cannot arm during a draft-generation loop")
        for depth in self._completed_rounds_by_depth:
            self._completed_rounds_by_depth[depth] = 0
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    def reset(self) -> None:
        if self._armed:
            raise RuntimeError("disarm draft-generation ordinals before reset")
        if self._active is not None:
            raise RuntimeError("cannot reset during a draft-generation loop")
        for depth in self._completed_rounds_by_depth:
            self._completed_rounds_by_depth[depth] = 0


class PinnedDraftLoopAdapter:
    """Wrap the loop only to establish and validate ordinal context."""

    def __init__(
        self,
        *,
        owner: type,
        expected_source_sha256: str,
        ordinals: DraftGenerationOrdinals,
    ) -> None:
        self._owner = owner
        self._expected_source_sha256 = expected_source_sha256
        self._ordinals = ordinals
        self._original: Callable[..., Any] | None = None
        self._installed = False

    def validate(self) -> Callable[..., Any]:
        original = getattr(self._owner, "_multi_step_decode", None)
        if original is None or not callable(original):
            raise AdapterValidationError(
                "AutoRegressiveSpeculator._multi_step_decode is absent"
            )
        if getattr(original, "_spark_q2r_draft_loop", False):
            raise AdapterValidationError("draft loop is already wrapped")
        actual = source_sha256(original)
        if actual != self._expected_source_sha256:
            raise AdapterValidationError(
                "source mismatch for "
                "AutoRegressiveSpeculator._multi_step_decode: expected "
                f"{self._expected_source_sha256}, got {actual}"
            )
        return original

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("draft-loop adapter is already installed")
        original = self.validate()
        ordinals = self._ordinals

        @functools.wraps(original)
        def wrapped(
            instance: Any, *args: Any, **kwargs: Any
        ) -> Any:
            if len(args) >= 3:
                batch_desc = args[2]
            elif "batch_desc" in kwargs:
                batch_desc = kwargs["batch_desc"]
            else:
                raise RuntimeError(
                    "draft loop call has no source-pinned batch_desc argument"
                )
            entered = ordinals.enter(instance, batch_desc)
            failed = True
            try:
                result = original(instance, *args, **kwargs)
                failed = False
                return result
            finally:
                if entered:
                    ordinals.leave(instance, validate=not failed)

        wrapped._spark_q2r_draft_loop = True  # type: ignore[attr-defined]
        wrapped._spark_original = original  # type: ignore[attr-defined]
        setattr(self._owner, "_multi_step_decode", wrapped)
        self._original = original
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        current = getattr(self._owner, "_multi_step_decode")
        if not getattr(current, "_spark_q2r_draft_loop", False):
            raise AdapterValidationError(
                "draft loop changed after installation"
            )
        assert self._original is not None
        setattr(self._owner, "_multi_step_decode", self._original)
        self._original = None
        self._installed = False
