"""Worker-only bootstrap and sequenced control for the Q-2R live probe.

Import is inert. ``install()`` adds one lightweight wrapper around the exact
source-pinned GPU ``Worker.initialize_from_config`` seam. CUDA events, the
route arena, and the custom CUDA op are created only when that worker method
runs; API/frontend processes never allocate probe CUDA state.

The low-rate graph-status reporter calls ``q2r_probe_snapshot()``. It consumes
at most one atomically replaced JSON command per poll and publishes the
acknowledged sequence with phase timing state. Route counters are read only by
the explicit ``drain`` command after inference has stopped.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONTROL_SCHEMA = "sparkring-q2r-control/v1"
_STATUS_SCHEMA = "sparkring-q2r-status/v1"
_EXPECTED_ROUTE_WRAPPER_MARKER = "_sparkring_target_route_installer"
_EXPECTED_BRIDGE_MARKER = "_sparkring_q2r_bootstrap"


class ProbeBridgeError(RuntimeError):
    """The live probe could not preserve its fail-closed contract."""


@dataclass(frozen=True)
class ProbeFunctions:
    phase_install: Callable[[], None]
    phase_arm: Callable[[str], None]
    phase_disarm: Callable[[], None]
    phase_drain: Callable[[], Any]
    phase_snapshot: Callable[[], Mapping[str, Any]]
    route_arm: Callable[..., None]
    route_disarm: Callable[..., None]
    route_counters: Callable[[], Mapping[str, int]]
    route_drain: Callable[..., Mapping[str, int]]
    provenance_factory: Callable[..., Any]
    dcp_graph_report: Callable[[], Mapping[str, Any]] | None = None


class ProbeControlBridge:
    """One process-local, strictly sequenced control mailbox."""

    def __init__(
        self,
        *,
        functions: ProbeFunctions,
        control_path: Path,
        route_output_path: Path,
        rank: int,
        session_id: str = "offline-test",
        source_bundle_manifest: str = "0" * 64,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if rank < 0:
            raise ValueError("rank must be non-negative")
        if not session_id:
            raise ValueError("session_id must be non-empty")
        if (
            len(source_bundle_manifest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_bundle_manifest
            )
        ):
            raise ValueError(
                "source_bundle_manifest must be lowercase SHA-256"
            )
        self._functions = functions
        self._control_path = Path(control_path)
        self._route_output_path = Path(route_output_path)
        self._rank = rank
        self._session_id = session_id
        self._source_bundle_manifest = source_bundle_manifest
        self._provenance = dict(
            provenance
            or {
                "image": "offline-test",
                "checkpoint": "offline-test",
                "config_sha256": "0" * 64,
                "source_sha256": {
                    "bundle": source_bundle_manifest,
                },
                "rank": rank,
            }
        )
        self._lock = threading.Lock()
        self._worker_ready = False
        self._last_sequence = 0
        self._last_action = ""
        self._last_result = "idle"
        self._last_error = ""
        self._route_counters: dict[str, int] | None = None
        self._route_armed = False
        self._capture_routes = True
        self._drain: dict[str, Any] = {"state": "not_run"}
        self._dcp_graph_report: dict[str, Any] = {"state": "not_run"}

    def mark_worker_ready(self) -> None:
        with self._lock:
            self._worker_ready = True

    def _read_command(self) -> dict[str, Any] | None:
        if not self._control_path.is_file():
            return None
        try:
            value = json.loads(
                self._control_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ProbeBridgeError(f"invalid control file: {error}") from error
        if not isinstance(value, dict):
            raise ProbeBridgeError("control command must be a JSON object")
        if value.get("schema") != _CONTROL_SCHEMA:
            raise ProbeBridgeError("control command schema mismatch")
        sequence = value.get("sequence")
        if (
            not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 1
        ):
            raise ProbeBridgeError(
                "control sequence must be a positive integer"
            )
        action = value.get("action")
        if action not in {
            "arm",
            "next_request",
            "disarm",
            "verify_clean",
            "drain",
            "dcp_graph_report",
        }:
            raise ProbeBridgeError(f"unsupported control action: {action!r}")
        return value

    @staticmethod
    def _request_fields(command: Mapping[str, Any]) -> tuple[int, str]:
        slot = command.get("request_slot")
        key = command.get("request_key")
        if (
            not isinstance(slot, int)
            or isinstance(slot, bool)
            or slot < 0
        ):
            raise ProbeBridgeError(
                "request_slot must be a non-negative integer"
            )
        if not isinstance(key, str) or not key:
            raise ProbeBridgeError("request_key must be non-empty")
        return slot, key

    def _execute(self, command: Mapping[str, Any]) -> None:
        action = str(command["action"])
        functions = self._functions
        if action == "arm":
            epoch = command.get("epoch")
            if not isinstance(epoch, str) or not epoch:
                raise ProbeBridgeError("arm requires a non-empty epoch")
            slot, key = self._request_fields(command)
            capture_routes = command.get("capture_routes", True)
            if not isinstance(capture_routes, bool):
                raise ProbeBridgeError("capture_routes must be a boolean")
            self._capture_routes = capture_routes
            functions.phase_arm(epoch)
            if capture_routes:
                try:
                    functions.route_arm(
                        request_slot=slot,
                        request_key=key,
                        stream_slot=0,
                    )
                except Exception:
                    functions.phase_disarm()
                    raise
                self._route_armed = True
            else:
                self._route_armed = False
            return
        if action == "next_request":
            slot, key = self._request_fields(command)
            capture_routes = command.get(
                "capture_routes", self._capture_routes
            )
            if (
                not isinstance(capture_routes, bool)
                or capture_routes != self._capture_routes
            ):
                raise ProbeBridgeError(
                    "capture_routes cannot change during an armed epoch"
                )
            if capture_routes:
                functions.route_disarm(stream_slot=0)
                functions.route_arm(
                    request_slot=slot,
                    request_key=key,
                    stream_slot=0,
                )
                self._route_armed = True
            return
        if action == "disarm":
            functions.route_disarm(stream_slot=0)
            functions.phase_disarm()
            self._route_armed = False
            return
        if action == "verify_clean":
            functions.route_disarm(stream_slot=0)
            functions.phase_disarm()
            self._route_armed = False
            self._route_counters = dict(functions.route_counters())
            dirty = {
                name: int(value)
                for name, value in self._route_counters.items()
                if int(value) != 0
            }
            if dirty:
                raise ProbeBridgeError(
                    f"pre-arm route counters are not clean: {dirty}"
                )
            return
        if action == "dcp_graph_report":
            report = functions.dcp_graph_report
            if report is None:
                raise ProbeBridgeError(
                    "DCP graph report is unavailable in this launch"
                )
            value = report()
            if not isinstance(value, Mapping):
                raise ProbeBridgeError(
                    "DCP graph report must return a mapping"
                )
            self._dcp_graph_report = {
                "state": "complete",
                **dict(value),
            }
            return

        # Drain is an explicitly model-idle operation. Reading route counters
        # performs the one permitted device synchronization. Timing events are
        # drained before route artifact validation, so an independently useful
        # phase decomposition survives any later route-format failure.
        functions.route_disarm(stream_slot=0)
        functions.phase_disarm()
        self._route_armed = False
        provenance = command.get("provenance")
        if not isinstance(provenance, dict):
            raise ProbeBridgeError("drain requires provenance")
        if provenance != self._provenance:
            raise ProbeBridgeError(
                "drain provenance does not match worker-published provenance"
            )
        source_sha256 = provenance.get("source_sha256")
        if not isinstance(source_sha256, dict):
            raise ProbeBridgeError(
                "drain provenance requires source_sha256"
            )
        provenance_value = functions.provenance_factory(
            image=provenance.get("image", ""),
            checkpoint=provenance.get("checkpoint", ""),
            config_sha256=provenance.get("config_sha256", ""),
            source_sha256=source_sha256,
            rank=self._rank,
        )
        # The route counter read is the synchronization boundary for both
        # route-capturing and phase-only epochs. It must precede polling phase
        # events so their completion state cannot race the model stream.
        self._route_counters = dict(functions.route_counters())
        drain_result = functions.phase_drain()
        if int(getattr(drain_result, "still_pending", -1)) != 0:
            raise ProbeBridgeError(
                "phase timing still has pending events after route sync"
            )
        if int(getattr(drain_result, "errors", -1)) != 0:
            raise ProbeBridgeError("phase timing drain reported errors")
        if not self._capture_routes:
            dirty = {
                name: int(value)
                for name, value in self._route_counters.items()
                if int(value) != 0
            }
            if dirty:
                raise ProbeBridgeError(
                    f"phase-only drain found dirty route counters: {dirty}"
                )
            self._drain = {
                "state": "skipped-phase-only",
                "records": 0,
                "bytes": 0,
                "sha256": "",
            }
            return
        self._route_counters = dict(
            functions.route_drain(
                self._route_output_path,
                provenance_value,
                stream_slot=0,
            )
        )
        encoded = self._route_output_path.read_bytes()
        records = sum(
            1 for line in encoded.splitlines() if line.strip()
        )
        self._drain = {
            "state": "complete",
            "records": records,
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def poll(self) -> None:
        with self._lock:
            if not self._worker_ready:
                return
            try:
                command = self._read_command()
            except Exception as error:
                self._last_result = "failed"
                self._last_error = str(error)
                return
            if command is None:
                return
            sequence = int(command["sequence"])
            if sequence < self._last_sequence:
                self._last_result = "failed"
                self._last_error = (
                    f"control sequence moved backwards: {sequence} < "
                    f"{self._last_sequence}"
                )
                return
            if sequence == self._last_sequence:
                return
            if sequence != self._last_sequence + 1:
                self._last_result = "failed"
                self._last_error = (
                    f"control sequence gap: expected "
                    f"{self._last_sequence + 1}, got {sequence}"
                )
                return
            self._last_sequence = sequence
            self._last_action = str(command["action"])
            try:
                self._execute(command)
            except Exception as error:
                self._last_result = "failed"
                self._last_error = str(error)
            else:
                self._last_result = "ok"
                self._last_error = ""

    def snapshot(self) -> dict[str, Any]:
        self.poll()
        with self._lock:
            worker_ready = self._worker_ready
            last_sequence = self._last_sequence
            last_action = self._last_action
            last_result = self._last_result
            last_error = self._last_error
            route_counters = (
                None
                if self._route_counters is None
                else dict(self._route_counters)
            )
            route_armed = self._route_armed
            capture_routes = self._capture_routes
            drain = dict(self._drain)
            dcp_graph_report = dict(self._dcp_graph_report)
        phase: Mapping[str, Any] = {"enabled": False}
        if worker_ready:
            try:
                phase = self._functions.phase_snapshot()
            except Exception as error:
                last_result = "failed"
                last_error = f"phase snapshot failed: {error}"
        return {
            "schema": _STATUS_SCHEMA,
            "enabled": True,
            "session_id": self._session_id,
            "source_bundle_manifest": self._source_bundle_manifest,
            "provenance": dict(self._provenance),
            "worker_ready": worker_ready,
            "control_path": str(self._control_path),
            "route_output_path": str(self._route_output_path),
            "last_sequence": last_sequence,
            "last_action": last_action,
            "last_result": last_result,
            "last_error": last_error,
            "route": {
                "armed": route_armed,
                "capture_enabled": capture_routes,
                "counters": route_counters,
                "drain": drain,
            },
            "dcp_graph_report": dcp_graph_report,
            **dict(phase),
        }


_install_lock = threading.Lock()
_control_bridge: ProbeControlBridge | None = None
_worker_type: type | None = None
_route_wrapped_initialize: Callable[..., Any] | None = None


def _required_path(variable: str) -> Path:
    value = os.getenv(variable)
    if not value:
        raise ProbeBridgeError(f"{variable} is required")
    return Path(value)


def _rank() -> int:
    text = os.getenv("RANK")
    if text is None:
        raise ProbeBridgeError("RANK is required")
    try:
        rank = int(text)
    except ValueError as error:
        raise ProbeBridgeError("RANK must be an integer") from error
    if rank < 0:
        raise ProbeBridgeError("RANK must be non-negative")
    return rank


def _required_text(variable: str) -> str:
    value = os.getenv(variable)
    if not value:
        raise ProbeBridgeError(f"{variable} is required")
    return value


def _required_sha256(variable: str) -> str:
    value = _required_text(variable)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ProbeBridgeError(f"{variable} must be lowercase SHA-256")
    return value


def _provenance(rank: int, bundle_manifest: str) -> dict[str, Any]:
    source_text = os.getenv("SPARK_Q2R_SOURCE_SHA256_JSON")
    if source_text:
        try:
            source_sha256 = json.loads(source_text)
        except json.JSONDecodeError as error:
            raise ProbeBridgeError(
                "SPARK_Q2R_SOURCE_SHA256_JSON is invalid"
            ) from error
        if not isinstance(source_sha256, dict):
            raise ProbeBridgeError(
                "SPARK_Q2R_SOURCE_SHA256_JSON must be an object"
            )
    else:
        source_sha256 = {"bundle": bundle_manifest}
    return {
        "image": _required_text("SPARK_Q2R_IMAGE"),
        "checkpoint": _required_text("SPARK_Q2R_CHECKPOINT"),
        "config_sha256": _required_sha256("SPARK_Q2R_CONFIG_SHA256"),
        "source_sha256": source_sha256,
        "rank": rank,
    }


def install() -> None:
    """Install lightweight source-pinned bootstrap; allocate no CUDA arena."""

    global _control_bridge, _worker_type, _route_wrapped_initialize
    if os.getenv("SPARK_Q2R_PROBE") != "1":
        raise ProbeBridgeError("SPARK_Q2R_PROBE=1 is required")
    with _install_lock:
        if _control_bridge is not None:
            raise ProbeBridgeError("Q-2R probe bridge is already installed")

        import torch
        from moe_round_floor.target_route_capture import CaptureProvenance
        from moe_round_floor.target_route_capture_live import (
            LiveDependencies,
            LiveInstallConfig,
            arm_capture_salted,
            capture_counters,
            disarm_capture,
            drain_capture,
            install_opt_in,
        )
        from performance.harnesses.q2r_phase_timing import live_installer as phase
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
            MoERunner,
        )
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu_worker import Worker

        rank = _rank()
        dcp_graph_report = None
        if os.getenv("VLLM_SPARK_TP4_DCP_GRAPH_SHADOW") == "1":
            from spark_tp4_dcp_backend import dcp_graph_shadow_report

            dcp_graph_report = dcp_graph_shadow_report
        bundle_manifest = _required_sha256(
            "SPARK_Q2R_SOURCE_BUNDLE_MANIFEST"
        )
        functions = ProbeFunctions(
            phase_install=phase.install,
            phase_arm=phase.arm,
            phase_disarm=phase.disarm,
            phase_drain=phase.drain,
            phase_snapshot=phase.snapshot,
            route_arm=arm_capture_salted,
            route_disarm=disarm_capture,
            route_counters=capture_counters,
            route_drain=drain_capture,
            provenance_factory=CaptureProvenance,
            dcp_graph_report=dcp_graph_report,
        )
        bridge = ProbeControlBridge(
            functions=functions,
            control_path=_required_path("SPARK_Q2R_CONTROL_PATH"),
            route_output_path=_required_path("SPARK_Q2R_ROUTE_OUTPUT_PATH"),
            rank=rank,
            session_id=_required_text("SPARK_Q2R_SESSION_ID"),
            source_bundle_manifest=bundle_manifest,
            provenance=_provenance(rank, bundle_manifest),
        )

        install_opt_in(
            worker_type=Worker,
            runner_type=GPUModelRunner,
            dependencies=LiveDependencies(
                torch_module=torch,
                moe_runner_type=MoERunner,
                base_router_type=BaseRouter,
            ),
            config=LiveInstallConfig(
                runner_sample_sha256=_required_sha256(
                    "SPARK_Q2R_RUNNER_SAMPLE_SHA256"
                )
            ),
        )
        route_wrapped = Worker.initialize_from_config
        if not getattr(
            route_wrapped, _EXPECTED_ROUTE_WRAPPER_MARKER, False
        ):
            raise ProbeBridgeError("route installer marker is absent")
        if getattr(route_wrapped, _EXPECTED_BRIDGE_MARKER, False):
            raise ProbeBridgeError("worker bootstrap is already wrapped")

        @functools.wraps(route_wrapped)
        def worker_bootstrap(
            worker: Any, *args: Any, **kwargs: Any
        ) -> Any:
            functions.phase_install()
            result = route_wrapped(worker, *args, **kwargs)
            bridge.mark_worker_ready()
            return result

        setattr(worker_bootstrap, _EXPECTED_BRIDGE_MARKER, True)
        worker_bootstrap._spark_original = route_wrapped  # type: ignore[attr-defined]
        Worker.initialize_from_config = worker_bootstrap
        _control_bridge = bridge
        _worker_type = Worker
        _route_wrapped_initialize = route_wrapped


def q2r_probe_snapshot() -> dict[str, Any]:
    bridge = _control_bridge
    if bridge is None:
        return {
            "schema": _STATUS_SCHEMA,
            "enabled": False,
            "session_id": "",
            "source_bundle_manifest": "",
            "provenance": {},
            "worker_ready": False,
            "last_sequence": 0,
            "last_action": "",
            "last_result": "not_installed",
            "last_error": "",
            "route": {
                "armed": False,
                "counters": None,
                "drain": {"state": "not_run"},
            },
            "dcp_graph_report": {"state": "not_run"},
            "phase_timing": {"enabled": False},
        }
    return bridge.snapshot()
