"""Official vLLM endpoint/general plugin pair; no execution interception."""

from __future__ import annotations

import asyncio
import copy
import base64
import hashlib
from html import escape
from importlib.resources import files
import os
import re
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .presentation import render_report, render_text
from .metrics import CounterWindow, read_counters

from .collector import (
    ARG_FIELDS, MISSING, SCHEMA, WORKER_SCHEMA, configuration, environment, fact,
    observations, path, provenance, stored, worker_snapshot,
)


METHOD = "sparkring_status_snapshot_v1"
VIEW_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer"}


def dashboard_response(doc, status_code=200):
    template = files(__package__).joinpath("dashboard.html").read_text(encoding="utf-8")
    title, report = render_report(doc)
    # Hash static assets rather than allowing arbitrary inline code or remote CDNs.
    policies = []
    for tag, directive in (("script", "script-src"), ("style", "style-src")):
        content = re.search(rf"<{tag}>(.*?)</{tag}>", template, flags=re.S).group(1)
        digest = base64.b64encode(hashlib.sha256(content.encode()).digest()).decode()
        policies.append(f"{directive} 'sha256-{digest}'")
    csp = "; ".join(["default-src 'none'", *policies, "connect-src 'self'",
                     "base-uri 'none'", "frame-ancestors 'none'", "form-action 'none'"])
    page = re.sub(r"%%(TITLE|REPORT)%%", lambda match: {"TITLE": escape(title), "REPORT": report}[match[1]], template)
    return HTMLResponse(page,
                        status_code=status_code, headers={**VIEW_HEADERS, "Content-Security-Policy": csp})


def expected_workers(config):
    parallel = stored(config, "parallel_config")
    # The torchrun-compatible executor inherits UniProcExecutor: one local
    # worker responds, even though its distributed world spans other engines.
    if stored(parallel, "distributed_executor_backend") == "external_launcher":
        return 1
    count = stored(parallel, "world_size")
    if count is MISSING:
        sizes = [stored(parallel, name) for name in (
            "tensor_parallel_size", "pipeline_parallel_size",
            "prefill_context_parallel_size")]
        if not all(type(size) is int and size > 0 for size in sizes):
            return None
        count = sizes[0] * sizes[1] * sizes[2]
    # Ordinary MP/Ray world_size already scopes TP*PP*PCP to one DP engine.
    return count if type(count) is int and 0 < count <= 256 else None


def _worker_method(self):
    try:
        return worker_snapshot(self)
    except Exception:
        # Exception messages can contain paths, request data, or credentials.
        return {"schema": WORKER_SCHEMA, "error": "snapshot_failed",
                "identity": {"rank": fact(stored(self, "rank"), source="worker_instance")}}


def register_worker_method():
    """Add only a unique read-only RPC method via the general plugin hook."""
    from vllm.v1.worker.worker_base import WorkerBase

    existing = getattr(WorkerBase, METHOD, None)
    if existing is not None and existing is not _worker_method:
        raise RuntimeError("SparkRing status worker method name is already registered")
    from .binding import configure
    configure(os.environ)
    setattr(WorkerBase, METHOD, _worker_method)


class StatusService:
    def __init__(self, config, args, engine, *, receipt=None, environ=None,
                 wait_seconds=0.25, cache_seconds=5.0):
        self.engine = engine
        self.wait_seconds = wait_seconds
        self.cache_seconds = cache_seconds
        self.configured = {
            "arguments": configuration(args, ARG_FIELDS),
            "environment": environment(os.environ if environ is None else environ),
        }
        # Parsed arguments are not the resolved VllmConfig.
        for item in self.configured["arguments"].values():
            item["source"] = "parsed_server_arguments"
        self.effective = configuration(config)
        self.provenance = provenance() if receipt is None else receipt
        self.started_ns = time.time_ns()
        # An EngineClient may address only one DP engine. Never fabricate global
        # DP rank coverage from a local executor response.
        self.expected = expected_workers(config)
        self.pending = None
        self.cached = None
        self.cached_monotonic = None
        self.last_error = None
        self.last_attempt = None
        self.metric_window = CounterWindow()
        self.metric_snapshot = None
        self.metric_time = None

    async def _collect(self):
        # Do not use executor timeout or cancellation: workers share ordered
        # response queues with serving. Late replies must still be consumed.
        return await self.engine.collective_rpc(METHOD, timeout=None)

    def _finish_pending(self):
        if self.pending is None or not self.pending.done():
            return
        try:
            rows = self.pending.result()
            if type(rows) is not list or len(rows) > 256:
                raise ValueError("unexpected RPC result")
            if any(type(row) is not dict or row.get("schema") != WORKER_SCHEMA for row in rows):
                raise ValueError("unexpected worker schema")
            self.cached = rows
            self.cached_monotonic = time.monotonic()
            self.last_error = None
        except Exception:
            self.last_error = "worker_rpc_failed"
        finally:
            self.pending = None

    async def snapshot(self):
        self._finish_pending()
        now = time.monotonic()
        refresh_after = max(self.last_attempt or 0, self.cached_monotonic or 0)
        if (self.engine is not None and self.pending is None
                and (self.last_attempt is None or now - refresh_after >= self.cache_seconds)):
            self.last_attempt = now
            self.pending = asyncio.create_task(self._collect())
        if self.pending is not None:
            # asyncio.wait leaves the RPC alive when the bounded HTTP wait ends.
            await asyncio.wait({self.pending}, timeout=self.wait_seconds)
            self._finish_pending()
        age = None if self.cached_monotonic is None else time.monotonic() - self.cached_monotonic
        ranks = copy.deepcopy(self.cached or [])
        valid = [row for row in ranks if "error" not in row]
        identities = [path(row, "identity.rank.value") for row in valid]
        identities_known = all(type(rank) is int and rank >= 0 for rank in identities)
        identities_unique = identities_known and len(set(identities)) == len(identities)
        state = "complete" if (self.expected is not None and len(valid) == self.expected
                                and identities_unique) else "partial"
        if self.pending is not None:
            state = "pending"
        elif self.engine is None:
            state = "unavailable"
        elif self.last_error:
            state = "error"
        if self.metric_time is None or now - self.metric_time >= self.cache_seconds:
            self.metric_time = now
            try:
                self.metric_snapshot = self.metric_window.snapshot(read_counters(), now)
            except Exception:
                self.metric_snapshot = {'state': 'not_observed', 'source': 'prometheus_speculation_counters',
                                        'reason': 'counter_snapshot_unavailable'}
        observed = observations()
        observed['speculative_acceptance'] = self.metric_snapshot
        return {
            "schema": SCHEMA, "generated_at_unix_ns": time.time_ns(),
            "startup_snapshot_at_unix_ns": self.started_ns,
            "configured": self.configured, "effective": self.effective,
            "observed": observed, "provenance": self.provenance,
            "workers": {"state": state, "scope": "addressed_engine_executor",
                        "expected_count": self.expected, "received_count": len(ranks),
                        "missing_count": max(0, self.expected - len(ranks)) if self.expected is not None else None,
                        "missing_rpc_slots": list(range(len(ranks), self.expected)) if self.expected is not None else [],
                        "rank_identities_unique": identities_unique,
                        "cache_age_seconds": age,
                        "stale": age is not None and age >= self.cache_seconds,
                        "rpc_pending": self.pending is not None,
                        "pending_age_seconds": now - self.last_attempt if self.pending is not None else None,
                        "error": self.last_error, "ranks": ranks},
        }


class StatusPlugin:
    name = "sparkring_status"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        async def snapshot(request):
            service = getattr(request.app.state, "sparkring_status_service", None)
            if service is None:
                return {"schema": SCHEMA, "state": "initializing"}, 503
            return await service.snapshot(), 200

        @app.get("/v1/sparkring/status", include_in_schema=True)
        async def status(request: Request):
            doc, code = await snapshot(request)
            return JSONResponse(doc, status_code=code, headers=VIEW_HEADERS)

        @app.get("/v1/sparkring/status.txt", response_class=PlainTextResponse)
        async def text_status(request: Request):
            doc, code = await snapshot(request)
            return PlainTextResponse(render_text(doc), status_code=code, headers=VIEW_HEADERS)

        @app.get("/v1/sparkring/status/view", response_class=HTMLResponse)
        async def browser_status(request: Request):
            doc, code = await snapshot(request)
            return dashboard_response(doc, code)

    async def init_state(self, engine_client, state, args) -> None:
        state.sparkring_status_service = StatusService(
            getattr(state, "vllm_config", None), args, engine_client)
