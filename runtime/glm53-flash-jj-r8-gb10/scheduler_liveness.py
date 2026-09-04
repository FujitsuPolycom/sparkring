#!/usr/bin/env python3
"""Expose scheduler liveness separately from vLLM API readiness."""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable


_METRICS = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_usage": "vllm:kv_cache_usage_perc",
    "uncertain_ranks": "vllm:sparkcache_capture_ownership_uncertain_ranks",
}


def _metric_sum(text: str, name: str, *, required: bool = True) -> float:
    pattern = re.compile(
        rf"(?m)^{re.escape(name)}(?:\{{[^\n]*\}})?\s+([0-9.eE+-]+)$"
    )
    values = [float(match.group(1)) for match in pattern.finditer(text)]
    if not values and required:
        raise ValueError(f"metrics response does not contain {name}")
    return sum(values)


class SchedulerLiveness:
    """Track sustained scheduler blockage from successive metric samples."""

    def __init__(
        self,
        *,
        blocked_timeout_seconds: float,
        idle_kv_warn_seconds: float,
        stale_sample_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (
            ("blocked timeout", blocked_timeout_seconds),
            ("idle KV warning", idle_kv_warn_seconds),
            ("stale sample timeout", stale_sample_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self._blocked_timeout = float(blocked_timeout_seconds)
        self._idle_kv_warn = float(idle_kv_warn_seconds)
        self._stale_sample = float(stale_sample_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._blocked_since: float | None = None
        self._idle_nonfall_since: float | None = None
        self._last_idle_kv: float | None = None
        self._last_success: float | None = None
        self._last_error = "metrics have not been sampled"
        self._values = {
            "running": 0.0,
            "waiting": 0.0,
            "kv_usage": 0.0,
            "uncertain_ranks": 0.0,
        }

    def observe(self, metrics_text: str) -> None:
        values = {
            "running": _metric_sum(metrics_text, _METRICS["running"]),
            "waiting": _metric_sum(metrics_text, _METRICS["waiting"]),
            "kv_usage": _metric_sum(metrics_text, _METRICS["kv_usage"]),
            "uncertain_ranks": _metric_sum(
                metrics_text,
                _METRICS["uncertain_ranks"],
                required=False,
            ),
        }
        now = self._clock()
        with self._lock:
            if values["running"] == 0 and values["waiting"] > 0:
                if self._blocked_since is None:
                    self._blocked_since = now
            else:
                self._blocked_since = None

            if values["running"] == 0 and values["kv_usage"] > 0:
                if (
                    self._last_idle_kv is None
                    or values["kv_usage"] < self._last_idle_kv - 1e-9
                ):
                    self._idle_nonfall_since = now
                self._last_idle_kv = values["kv_usage"]
            else:
                self._idle_nonfall_since = None
                self._last_idle_kv = None

            self._values = values
            self._last_success = now
            self._last_error = ""

    def observe_error(self, error: BaseException) -> None:
        with self._lock:
            self._last_error = str(error)

    def snapshot(self) -> dict[str, object]:
        now = self._clock()
        with self._lock:
            blocked_seconds = (
                max(0.0, now - self._blocked_since)
                if self._blocked_since is not None
                else 0.0
            )
            idle_kv_seconds = (
                max(0.0, now - self._idle_nonfall_since)
                if self._idle_nonfall_since is not None
                else 0.0
            )
            sample_age = (
                max(0.0, now - self._last_success)
                if self._last_success is not None
                else None
            )
            healthy = True
            reason = "ok"
            if self._values["uncertain_ranks"] > 0:
                healthy = False
                reason = "capture_ownership_uncertain"
            elif sample_age is None or sample_age > self._stale_sample:
                healthy = False
                reason = "metrics_unavailable"
            elif blocked_seconds >= self._blocked_timeout:
                healthy = False
                reason = "scheduler_capacity_stall"
            warnings = []
            if idle_kv_seconds >= self._idle_kv_warn:
                warnings.append("idle_kv_not_falling")
            return {
                "schema": "sparkring-scheduler-liveness/v1",
                "healthy": healthy,
                "reason": reason,
                "warnings": warnings,
                "running_requests": self._values["running"],
                "waiting_requests": self._values["waiting"],
                "kv_cache_usage": self._values["kv_usage"],
                "capture_ownership_uncertain_ranks": self._values[
                    "uncertain_ranks"
                ],
                "blocked_seconds": blocked_seconds,
                "idle_kv_nonfall_seconds": idle_kv_seconds,
                "sample_age_seconds": sample_age,
                "last_sample_error": self._last_error,
            }

    def http_status(self) -> int:
        return 200 if bool(self.snapshot()["healthy"]) else 503

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        values = {
            "sparkring:scheduler_liveness": int(bool(snapshot["healthy"])),
            "sparkring:scheduler_blocked_seconds": snapshot["blocked_seconds"],
            "sparkring:idle_kv_nonfall_seconds": snapshot[
                "idle_kv_nonfall_seconds"
            ],
            "sparkring:liveness_sample_age_seconds": (
                snapshot["sample_age_seconds"]
                if snapshot["sample_age_seconds"] is not None
                else self._stale_sample + 1
            ),
        }
        return "".join(f"{name} {value}\n" for name, value in values.items())


class SchedulerLivenessService:
    """Poll vLLM metrics and serve a bounded liveness API on rank zero."""

    def __init__(
        self,
        *,
        metrics_url: str,
        port: int,
        sample_interval_seconds: float,
        monitor: SchedulerLiveness,
        credential: str | None = None,
    ) -> None:
        if not 1 <= port <= 65535:
            raise ValueError("liveness port must be between 1 and 65535")
        if sample_interval_seconds <= 0:
            raise ValueError("sample interval must be positive")
        self._metrics_url = metrics_url
        self._sample_interval = float(sample_interval_seconds)
        self._monitor = monitor
        self._credential = credential
        self._stop = threading.Event()
        self._last_reported_state: tuple[object, ...] | None = None
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if self.path == "/liveness":
                    payload = json.dumps(
                        service._monitor.snapshot(),
                        separators=(",", ":"),
                    ).encode()
                    self.send_response(service._monitor.http_status())
                    self.send_header("Content-Type", "application/json")
                elif self.path == "/metrics":
                    payload = service._monitor.prometheus().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                else:
                    payload = b"not found\n"
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="sparkring-liveness-http",
            daemon=True,
        )
        self._sample_thread = threading.Thread(
            target=self._sample_loop,
            name="sparkring-liveness-sampler",
            daemon=True,
        )

    def _sample_once(self) -> None:
        request = urllib.request.Request(self._metrics_url)
        if self._credential:
            request.add_header("Authorization", f"Bearer {self._credential}")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                self._monitor.observe(response.read().decode())
        except Exception as error:  # noqa: BLE001 - liveness records fetch failure
            self._monitor.observe_error(error)
        snapshot = self._monitor.snapshot()
        state = (
            snapshot["healthy"],
            snapshot["reason"],
            tuple(snapshot["warnings"]),
        )
        if state != self._last_reported_state and (
            not bool(snapshot["healthy"]) or snapshot["warnings"]
        ):
            print(json.dumps({"scheduler_liveness": snapshot}, separators=(",", ":")), flush=True)
        self._last_reported_state = state

    def _sample_loop(self) -> None:
        while not self._stop.wait(self._sample_interval):
            self._sample_once()

    def start(self) -> None:
        self._sample_once()
        self._server_thread.start()
        self._sample_thread.start()

    def close(self) -> None:
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()
        self._server_thread.join(timeout=5)
        self._sample_thread.join(timeout=5)


def start_liveness_service(
    *,
    metrics_url: str,
    port: int,
    blocked_timeout_seconds: float,
    idle_kv_warn_seconds: float,
    stale_sample_seconds: float,
    sample_interval_seconds: float,
    credential: str | None,
) -> SchedulerLivenessService:
    monitor = SchedulerLiveness(
        blocked_timeout_seconds=blocked_timeout_seconds,
        idle_kv_warn_seconds=idle_kv_warn_seconds,
        stale_sample_seconds=stale_sample_seconds,
    )
    service = SchedulerLivenessService(
        metrics_url=metrics_url,
        port=port,
        sample_interval_seconds=sample_interval_seconds,
        monitor=monitor,
        credential=credential,
    )
    service.start()
    return service
