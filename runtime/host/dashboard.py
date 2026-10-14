"""``sparkring dashboard``: a read-only web page of one or more SparkRing fabrics and the models they serve.

Status: Experimental. Covered by offline tests only; no ring has run it.

The page shows each fabric's ring of Sparks with the arc every model serves
from, and per model its state, output tokens per second, requests, KV cache
use and budget, draft acceptance, first-token time, prefix-cache hits,
serving settings, collective transport, image and per-Spark memory and disk.
``sparkring dashboard tui`` (``runtime.host.dashboard_tui``) shows the same
status document in a terminal, read from this service.

Where its data comes from
-------------------------

- Host files. The fabric document and the site file are compared every
  one-second cycle by device, inode, modification time and size, and re-read
  when any of them changed, so a site file that ``export`` replaces is shown
  within a second. SIGHUP (``systemctl reload sparkring-dashboard``) re-reads
  them at the next cycle whatever their stamps say. The host-agent status and
  the recovery record are re-read every FILES_SECONDS:

  - the fabric document (``sparkring-fabric/v1``, ``runtime.common.fabric_document``)
    for the Sparks, their positions, hostnames and management addresses;
  - the site file (``sparkring-dashboard-site/v1``, SITE_PATH) that
    ``sudo sparkring dashboard export`` writes on Node A from the controller's
    deployment records: each deployment's profile, served model name, positions,
    API address and port and container names. The controller's records are
    readable by root only; the site file holds only these fields, so the
    dashboard needs no privilege;
  - this Spark's host-agent status (``sparkring-node-status/v1``,
    ``/run/sparkring/status.json``) and the recovery timer's public cache of
    every Spark's agent facts (``sparkring-dashboard-nodes/v1``, NODE_REPORTS_PATH);
  - on Node A, the recovery timer's latest finding per deployment
    (``sparkring-recovery/v1``, RECOVERY_PATH), which it writes about once a
    minute while a model is active.

- HTTP GET requests to each deployment's API, through ``ReadOnlyClient``:

  - ``/metrics`` every METRICS_VIEWED_SECONDS while a browser or the terminal
    view shows the status (it fetched ``/api/status`` within VIEWER_SECONDS;
    ``/api/status?viewer=passive`` does not count), else every
    METRICS_UNVIEWED_SECONDS (METRICS_PLUGIN_UNVIEWED_SECONDS when a fresh
    stat-logger snapshot is available). The vllm:cache_config_info gauge gives the
    KV cache budget;
  - ``/v1/models`` every MODELS_SECONDS while the model serves, for its served
    name and context length. A failed metrics read tells the states apart: a
    refused connection is a stopped model, a timeout an unreachable Spark and an
    HTTP error a model still starting; a silent API is asked every FAILED_SECONDS;
  - ``/v1/sparkring/status`` (the runtime-status plugin) once per model run.
    Each uncached request makes the plugin ask every worker for a report
    through the engine's control path, which runs between decode steps, so it
    is never polled. ``/api/refresh?model=ID`` asks for one more, at most once
    per RUNTIME_REFRESH_SECONDS per model.

Invariants
----------

- ``ReadOnlyClient.get`` is the only network request this module makes. It
  sends GET without a body, only to READ_PATHS, only to addresses the host
  files name. Inference paths such as ``/v1/chat/completions`` and
  ``/v1/completions`` are refused before a connection is opened.
- The module starts no process: no SSH, no ``docker``, no ``ping``. Log
  viewing is on demand by the operator: the page shows the command that
  prints a model's log, and the operator runs it.
- While quiet mode is held, no request is sent at all, from the next
  one-second cycle on, for measurements that must see no outside traffic. An
  operator holds it with the quiet file (QUIET_PATH, ``sparkring dashboard quiet
  on``); a measurement holds it with a holder file of its own
  (``quiet hold``/``quiet release``), so measurements that overlap, or an
  operator's hold, never end each other's. A holder lapses when its process has
  exited or after its time to live.
- The service lowers its own priority (``lower_priority``): idle CPU
  scheduling, nice 19 and, on a Spark, the CPUs outside the fastest class as
  SIRCL's classifier finds them (the Cortex-A725 cores of a GB10), so serving
  processes and SIRCL's threads on the performance cores are never displaced.
  It writes no file; ``export`` and ``quiet``, run by an operator, each write one.
- It listens on Node A's LAN address and on 127.0.0.1 (``listen_addresses``),
  never on a fabric or administration address. Loopback serves programs on
  Node A itself, such as ``curl http://127.0.0.1:8090/api/status`` and the
  terminal view, without the LAN address that setup recorded; every program
  there can reach the LAN address too, so it widens nothing.
"""
from __future__ import annotations

import argparse
import base64
import collections
from functools import lru_cache
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import shlex
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from runtime.common.transport_names import collective_name
from runtime.common import serving, thinking
from runtime.host.dashboard_rates import number, plugin_fresh

SCHEMA = "sparkring-dashboard/v1"
SITE_SCHEMA = "sparkring-dashboard-site/v1"
SITE_PATH = Path("/var/lib/sparkring/dashboard/site.json")
FABRIC_PATH = Path("/etc/sparkring/fabric/topology.json")
NODE_STATUS_PATH = Path("/run/sparkring/status.json")
NODE_REPORTS_PATH = Path("/var/lib/sparkring/dashboard/nodes.json")
RECOVERY_PATH = Path("/var/lib/sparkring/recovery.json")
QUIET_PATH = Path("/run/sparkring/dashboard.quiet")
# One file per measurement that holds quiet mode, beside QUIET_PATH (quiet_holders_directory).
QUIET_HOLDERS = "dashboard-quiet"
QUIET_SCHEMA = "sparkring-dashboard-quiet/v1"
QUIET_TTL_SECONDS = 12 * 3600
QUIET_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}")
PAGE = Path(__file__).with_name("dashboard.html")

# The only paths the collector requests: the model list, vLLM's Prometheus metrics and the
# runtime-status plugin's JSON report. Each is a GET that runs no model.
READ_PATHS = frozenset({"/metrics", "/v1/models", "/v1/sparkring/status"})
METRICS_VIEWED_SECONDS = 5.0
METRICS_UNVIEWED_SECONDS = 10.0
METRICS_PLUGIN_UNVIEWED_SECONDS = 60.0
HISTORY_SECONDS = 3 * 3600
HISTORY_SAMPLES = 2161
HISTORY_DEPLOYMENTS = 8
HISTORY_DELTA_SAMPLES = 120
HISTORY_VALUES = ("gen", "prompt", "accepted", "drafts", "draft_tokens", "steps", "ttft_sum", "ttft_count",
                  "pc_hits", "pc_queries", "step_seconds", "running", "waiting", "kv")
MODELS_SECONDS = 60.0
VIEWER_SECONDS = 60.0
FILES_SECONDS = 30.0
RUNTIME_REFRESH_SECONDS = 300.0
HTTP_TIMEOUT = 2.0
FAILED_SECONDS = 15.0         # a model API that does not answer is asked again after this long
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024
MAX_EVENTS = 200
CLOCK_STEP_SECONDS = 2.0      # a cycle whose clock reads this much earlier than the last one's saw a step back
DEFAULT_PORT = 8090
LOG_LINES = 200

COUNTERS = {
    "vllm:generation_tokens_total": "gen",
    "vllm:prompt_tokens_total": "prompt",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted",
    "vllm:spec_decode_num_drafts_total": "drafts",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
    "vllm:num_preemptions_total": "preemptions",
    "vllm:time_to_first_token_seconds_sum": "ttft_sum",
    "vllm:time_to_first_token_seconds_count": "ttft_count",
    "vllm:prefix_cache_queries_total": "pc_queries",
    "vllm:prefix_cache_hits_total": "pc_hits",
    "sglang:generation_tokens_total": "gen",
    "sglang:prompt_tokens_total": "prompt",
    "sglang:time_to_first_token_seconds_sum": "ttft_sum",
    "sglang:time_to_first_token_seconds_count": "ttft_count",
}
SETTING_KEYS = (
    "tensor_parallel_size", "decode_context_parallel_size", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
    "cache_dtype", "quantization", "speculative_method", "num_speculative_tokens",
    "reasoning_parser", "tool_call_parser", "default_chat_template_kwargs",
    "kv_cache_memory_bytes", "prefix_caching_enabled",
)
LABELS = {"busy": "Busy: API unavailable", "serving": "Serving", "not_ready": "Not ready", "stopped": "No model running",
          "offline": "Offline", "idle": "Idle", None: "Checking"}
MODEL_TAIL_WORDS = {"csf", "nvfp4", "fp8", "fp4", "mxfp8", "mxfp4", "int4", "int8", "awq", "gptq", "bf16",
                    "fp16", "qad", "spark", "attention", "gguf", "nf3", "exl3"}
LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
LABEL_ESCAPE_RE = re.compile(r"\\(.)")
CONTAINER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
HOSTNAME_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9-]{1,63})*")


# -- the one network client ---------------------------------------------------------------
class RequestRefused(ValueError):
    """A request the dashboard must never send: not a GET, not a read path, or not a known address."""


def request_url(host, port, path, allowed_hosts) -> str:
    """The URL of one permitted read; RequestRefused for anything else.

    `path` must be exactly one of READ_PATHS (no query, no other path), `host` one of
    `allowed_hosts` (the addresses the host files name) and `port` a TCP port."""
    if path not in READ_PATHS:
        raise RequestRefused(f"the dashboard reads only {', '.join(sorted(READ_PATHS))}, not {path!r}")
    if host not in allowed_hosts:
        raise RequestRefused(f"{host!r} is not an address the fabric document or site file names")
    if type(port) is not int or not 0 < port < 65536:
        raise RequestRefused(f"{port!r} is not a TCP port")
    literal = f"[{host}]" if ":" in host else host
    return f"http://{literal}:{port}{path}"


class Fetched:
    """The outcome of one read: `body` on success, else `failure` (refused, timeout, unreachable,
    http) with `code` for an HTTP error status."""

    def __init__(self, body=None, failure=None, code=None):
        self.body, self.failure, self.code = body, failure, code

    @property
    def ok(self) -> bool:
        return self.failure is None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A read endpoint cannot widen the permitted address, port or path by redirecting."""

    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


class ReadOnlyClient:
    """Sends the collector's GET requests and counts them; the module's only network egress.

    `opener` is a urllib opener; the default ignores proxy settings from the environment, so
    requests to Spark addresses never leave through a proxy."""

    def __init__(self, opener=None, timeout=HTTP_TIMEOUT):
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.timeout = timeout
        self.allowed: frozenset = frozenset()
        self.lock = threading.Lock()
        self.requests = collections.Counter()   # path -> requests sent
        self.received = 0                        # response bytes

    def allow(self, hosts) -> None:
        self.allowed = frozenset(hosts)

    def get(self, host, port, path) -> Fetched:
        url = request_url(host, port, path, self.allowed)
        request = urllib.request.Request(url, method="GET", headers={
            "User-Agent": "sparkring-dashboard", "Accept": "application/json, text/plain"})
        if request.get_method() != "GET" or request.data is not None:
            raise RequestRefused("the dashboard sends GET requests without a body only")
        with self.lock:
            self.requests[path] += 1
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            return Fetched(failure="http", code=error.code)
        except urllib.error.URLError as error:
            return Fetched(failure=_failure_kind(error.reason))
        except (TimeoutError, socket.timeout):
            return Fetched(failure="timeout")
        except OSError as error:
            return Fetched(failure=_failure_kind(error))
        with self.lock:
            self.received += len(body)
        if len(body) > MAX_RESPONSE_BYTES:
            return Fetched(failure="too_large")
        return Fetched(body=body)

    def counters(self) -> dict:
        with self.lock:
            return {"requests": dict(self.requests), "received_bytes": self.received}


def _failure_kind(reason) -> str:
    if isinstance(reason, ConnectionRefusedError):
        return "refused"
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "timeout"
    return "unreachable"


# -- parsers --------------------------------------------------------------------------------
def parse_labels(text: str) -> dict:
    """'a="x",b="y"' -> {"a": "x", "b": "y"}, with the exposition format's escapes undone."""
    return {key: LABEL_ESCAPE_RE.sub(lambda m: "\n" if m.group(1) == "n" else m.group(1), value)
            for key, value in LABEL_RE.findall(text)}


def parse_metrics(text: str) -> dict:
    """Serving-engine series the page uses, summed across engines; KV use takes the maximum.

    `cache_info` holds the first engine's vllm:cache_config_info labels; `model_name` the
    model_name label of the first vLLM series; `started` the API process's
    process_start_time_seconds, which identifies one model run."""
    out: dict = {"finished": {}, "cache_info": None, "model_name": None, "started": None}
    kv = None
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        brace = line.find("{")
        if brace != -1:
            end = line.rfind("}")
            if end < brace:
                continue
            name, labels, rest = line[:brace], line[brace + 1:end], line[end + 1:]
        else:
            name, _, rest = line.partition(" ")
            labels = ""
        if name == "vllm:cache_config_info":
            if out["cache_info"] is None:
                out["cache_info"] = parse_labels(labels)
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        if out["model_name"] is None and name.startswith(("vllm:", "sglang:")) and 'model_name="' in labels:
            out["model_name"] = parse_labels(labels).get("model_name")
        if name in COUNTERS:
            key = COUNTERS[name]
            out[key] = out.get(key, 0.0) + value
        elif name in ("vllm:num_requests_running", "sglang:num_running_reqs"):
            out["running"] = out.get("running", 0.0) + value
        elif name in ("vllm:num_requests_waiting", "sglang:num_queue_reqs"):
            out["waiting"] = out.get("waiting", 0.0) + value
        elif name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "sglang:token_usage"):
            kv = value if kv is None else max(kv, value)
        elif name == "vllm:request_success_total":
            reason = parse_labels(labels).get("finished_reason") or "other"
            out["finished"][reason] = out["finished"].get(reason, 0.0) + value
        elif name == "process_start_time_seconds":
            out["started"] = value
    out["kv"] = kv
    return out


def capacity_from_cache_info(info: dict | None) -> dict | None:
    """The KV cache budget from vllm:cache_config_info, or None when the gauge lacks it.

    The gauge carries kv_cache_size_tokens, kv_cache_max_concurrency and kv_cache_memory_bytes,
    the figures of the engine's "GPU KV cache size" startup line; no log needs to be read."""
    if not info:
        return None

    def number(key, cast):
        try:
            return cast(info[key])
        except (KeyError, TypeError, ValueError):
            return None  # absent, or vLLM's "None" for an unset value

    tokens = number("kv_cache_size_tokens", int)
    if tokens is None:
        return None
    return {"kv_tokens": tokens, "concurrency": number("kv_cache_max_concurrency", float),
            "kv_bytes": number("kv_cache_memory_bytes", int),
            "gpu_memory_utilization": number("gpu_memory_utilization", float),
            "block_size": number("block_size", int), "max_num_seqs": None, "max_model_len": None,
            "mm": None, "source": "metrics"}


def known(field):
    return field.get("value") if isinstance(field, dict) and field.get("state") == "known" else None


def read_transport(doc: dict) -> dict | None:
    """The transport of the TP group's collectives and the b12x bundle the image carries.

    runtime-status 0.3.6 and later report `tp_collective_transport` per worker; for older
    images the label comes from the TP group's resident backend facts. `label` is the
    transport's display name (``runtime.common.transport_names``).

    `bundle` (and its older alias `profile`) names the b12x RoCEnante bundle in the image,
    whichever transport runs; `bundle_in_use` is True when RoCEnante carries the
    collectives, False when another transport does and None when that is not known. The
    page and the terminal view show the transport, never the bundle name."""
    backend = rocenante = nccl_available = nccl_disabled = collective = sircl_version = None
    for rank in (doc.get("workers") or {}).get("ranks") or []:
        transport = rank.get("transport") or {}
        tp_group = (transport.get("groups") or {}).get("TP") or {}
        backend = backend or known(tp_group.get("b12x_backend"))
        if rocenante is None:
            rocenante = known((tp_group.get("rocenante") or {}).get("enabled"))
        nccl = tp_group.get("nccl") or {}
        if nccl_available is None:
            nccl_available = known(nccl.get("available"))
        if nccl_disabled is None:
            nccl_disabled = known(nccl.get("disabled"))
        for effective in (rank.get("effective") or {}, transport.get("effective") or {}):
            collective = collective or known(effective.get("tp_collective_transport"))
            sircl_version = sircl_version or known(effective.get("tp_sircl_version"))
    provenance = doc.get("provenance") or {}
    bundle = known((provenance.get("b12x_comm_bundle") or {}).get("name")) or known(provenance.get("transport_profile"))
    running = collective
    if collective is None:
        if isinstance(backend, str) and "sircl" in backend.lower():
            running = "sircl"
        elif rocenante is True:
            running = "rocenante"
        elif not backend and nccl_available and not nccl_disabled:
            running = "nccl"
    label = collective_name(running) if running is not None else (str(backend) if backend else None)
    if label is None and bundle is None:
        return None
    nccl_on = None if nccl_available is None and nccl_disabled is None else bool(nccl_available) and not nccl_disabled
    in_use = None if running is None else running == "rocenante"
    return {"label": label, "collective": collective, "sircl_version": sircl_version if collective == "sircl" else None,
            "nccl": nccl_on, "bundle": bundle, "profile": bundle, "bundle_in_use": in_use, "source": "runtime"}


def read_runtime_status(doc: dict) -> dict:
    """The fields of a sparkring-runtime-status/v1 document the page shows."""
    effective = doc.get("effective") or {}
    arguments = (doc.get("configured") or {}).get("arguments") or {}

    def value(key):
        found = known(effective.get(key))
        return known(arguments.get(key)) if found is None else found

    workers = doc.get("workers") or {}
    nodes, image = [], None
    for rank in workers.get("ranks") or []:
        identity = rank.get("identity") or {}
        image = image or known(identity.get("image_id"))
        resources = rank.get("resources") or {}
        memory = resources.get("memory") or {}
        root = next((fs for fs in resources.get("filesystems") or [] if "/" in (fs.get("paths") or [])), {})
        nodes.append({"rank": known(identity.get("rank")), "node": resources.get("node"),
                      "memory_total": memory.get("total_bytes"), "memory_available": memory.get("available_bytes"),
                      "disk_total": root.get("total_bytes"),
                      "disk_available": root.get("available_bytes", root.get("free_bytes"))})
    started = doc.get("startup_snapshot_at_unix_ns")
    return {"settings": {key: value(key) for key in SETTING_KEYS}, "nodes": nodes,
            "workers": {"expected": workers.get("expected_count"), "received": workers.get("received_count")},
            "image": image.split(":")[-1][:12] if isinstance(image, str) else None,
            "started_at": started / 1e9 if isinstance(started, (int, float)) else None,
            "tp": value("tensor_parallel_size"), "dp": value("data_parallel_size"),
            "pp": value("pipeline_parallel_size"), "transport": read_transport(doc)}


def short_model(model) -> str | None:
    """Model family without its packaging tail: 'GLM-5.3-Flash-NVFP4-Spark-TP2' -> 'GLM-5.3-Flash'."""
    if not model:
        return None
    kept = []
    for part in str(model).split("/")[-1].split("-"):
        low = part.lower()
        if kept and (low in MODEL_TAIL_WORDS or re.fullmatch(r"(tp|dcp)\d+", low) or re.fullmatch(r"[\d.]+bpw", low)):
            break
        kept.append(part)
    return "-".join(kept) or str(model)


def layout_label(size) -> str | None:
    if not isinstance(size, int) or size < 1:
        return None
    return f"TP{size} {'ring' if size >= 4 else 'pair'}"


def run_name(positions) -> str:
    """'tp4-7-2' for the arc 7, 0, 1, 2; the run naming the lab tooling and SIRCL use."""
    return f"tp{len(positions)}-{positions[0]}-{positions[-1]}"


# -- host files ------------------------------------------------------------------------------
def read_json(path: Path):
    """A small JSON host file, or raises OSError/ValueError."""
    with open(path, "rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{path} is larger than {MAX_FILE_BYTES} bytes")
    return json.loads(data.decode("utf-8"))


def fabric_from_document(document: dict) -> dict:
    """The parts of a sparkring-fabric/v1 document the dashboard uses; ValueError when unusable.

    The full validator (runtime.common.fabric_document.validate) also checks every port and
    cable. The dashboard retains cable address ownership to resolve repeated rendezvous
    addresses without assuming that neighbouring containers belong to one model."""
    if not isinstance(document, dict) or document.get("schema") != "sparkring-fabric/v1":
        raise ValueError("not a sparkring-fabric/v1 document")
    size, positions, shape = document.get("size"), document.get("positions"), document.get("shape")
    if shape not in ("cycle", "path", "pair") or not isinstance(size, int) or size < 2 \
            or not isinstance(positions, list) or len(positions) != size:
        raise ValueError("the fabric document's shape, size and positions do not agree")
    rows = []
    for index, row in enumerate(positions):
        if not isinstance(row, dict) or row.get("position") != index:
            raise ValueError(f"position {index} is missing or out of order")
        hostname = row.get("hostname")
        if not isinstance(hostname, str) or not HOSTNAME_RE.fullmatch(hostname):
            raise ValueError(f"position {index} has no usable hostname")
        address = (row.get("management") or {}).get("address")
        addresses = [address, row.get("lan_address")]
        for port in (row.get("ports") or {}).values():
            for function in (port.get("functions") or {}).values():
                addresses.append(str(function.get("address") or "").split("/")[0])
        neighbors = []
        for port in (row.get("ports") or {}).values():
            peer = port.get("peer") or {}
            remote_position = peer.get("position")
            if type(remote_position) is not int or not 0 <= remote_position < size:
                continue
            remote_port = (positions[remote_position].get("ports") or {}).get(str(peer.get("port")), {})
            for function in (remote_port.get("functions") or {}).values():
                remote_address = str(function.get("address") or "").split("/")[0]
                if _ip(remote_address):
                    neighbors.append({"position": remote_position, "address": remote_address})
        rows.append({"position": index, "hostname": hostname, "node_id": row.get("node_id"),
                     "neighbors": neighbors,
                     "addresses": [a for a in addresses if _ip(a)],
                     "address": address if _ip(address) else None,
                     "lan_address": row.get("lan_address") if _ip(row.get("lan_address")) else None})
    verified = document.get("verified")
    return {"id": document.get("id") if isinstance(document.get("id"), str) else None,
            "cluster": document.get("cluster") if isinstance(document.get("cluster"), str) else None,
            "shape": shape, "closed": shape == "cycle", "size": size, "positions": rows,
            "verified": verified.get("result") if isinstance(verified, dict) else None}


def _ip(value) -> bool:
    try:
        ipaddress.ip_address(str(value))
    except ValueError:
        return False
    return True


def deployments_from_site(site: dict, size: int) -> list:
    """The deployments of a sparkring-dashboard-site/v1 file, checked against the fabric's size.

    A row that names positions outside the fabric, an unusable API address or port, or a
    container name with characters a shell would interpret is left out."""
    if not isinstance(site, dict) or site.get("schema") != SITE_SCHEMA:
        raise ValueError(f"not a {SITE_SCHEMA} file")
    rows = []
    for row in site.get("deployments") or []:
        if not isinstance(row, dict):
            continue
        positions = row.get("positions")
        port = row.get("api_port")
        if not (isinstance(positions, list) and positions and all(type(p) is int and 0 <= p < size for p in positions)
                and len(set(positions)) == len(positions)):
            continue
        if not _ip(row.get("api_host")) or type(port) is not int or not 0 < port < 65536:
            continue
        containers = [c for c in row.get("containers") or [] if isinstance(c, dict)
                      and type(c.get("position")) is int and isinstance(c.get("name"), str)
                      and CONTAINER_RE.fullmatch(c["name"])]
        rows.append({"id": str(row.get("id") or run_name(positions)), "deployment_id": row.get("deployment_id"), "profile": row.get("profile"),
                     "checkpoint": row.get("checkpoint"), "serving": row.get("serving"),
                     "settings": row.get("settings"),
                     "model": row.get("model"), "positions": positions, "api_host": row["api_host"],
                     "api_port": port, "display_url": row.get("display_url"),
                     "stopped": bool(row.get("stopped")), "containers": containers})
    return rows


@lru_cache(maxsize=128)
def thinking_record(profile, checkpoint):
    """Resolve shipped thinking records once per profile/checkpoint, without host access."""
    try:
        return thinking.of(profile, checkpoint)
    except (OSError, ValueError, KeyError):
        return None


def chat_defaults(row, settings):
    """Observed runtime kwargs take precedence over launch facts and saved settings, even when empty."""
    for source in (settings or {}, row.get("launch") or {}, row.get("settings") or {}):
        value = source.get("default_chat_template_kwargs")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                continue
        if isinstance(value, dict):
            return value
    return None


def thinking_state(row, settings):
    """Effective thinking and provenance from observations, then saved/profile defaults."""
    unmanaged = row.get("unmanaged")
    profile, checkpoint = row.get("profile"), row.get("checkpoint")
    record = None if unmanaged or not isinstance(profile, str) or (
        checkpoint is not None and not isinstance(checkpoint, str)) else thinking_record(profile, checkpoint)
    if not unmanaged and record is None:
        return None
    kwargs = chat_defaults(row, settings)
    chosen = {}
    enabled = False
    if kwargs is None:
        chosen = serving.effective(profile, row.get("serving"), checkpoint) if not unmanaged else {}
    else:
        off = record and record["off"]
        enabled = bool(off) and all(kwargs.get(k) is True for k in off)
        if (off and all(kwargs.get(k) == v for k, v in off.items())
                or (unmanaged or record["name"] == "deepseek-v41-encoder") and (
                    kwargs.get("enable_thinking") is False or kwargs.get("thinking") is False
                    or kwargs.get("reasoning_effort") == "none")):
            chosen = {"thinking": "off"}
        elif kwargs.get("reasoning_effort") is not None and (
                unmanaged or thinking.accepts(record, kwargs["reasoning_effort"])):
            chosen = {"reasoning_effort": kwargs["reasoning_effort"]}
        elif unmanaged and (kwargs.get("enable_thinking") is True or kwargs.get("thinking") is True):
            chosen = {"thinking": "on"}
    if unmanaged:
        return {"thinking": chosen.get("thinking", "on" if "reasoning_effort" in chosen else "unknown"),
                "level": chosen.get("reasoning_effort"), "chosen": bool(chosen), "levels": [],
                "model": {"thinking": "unknown", "level": None}}
    state = thinking.deployment_default(record, chosen)
    if enabled and state["thinking"] != "off":
        state.update(thinking="on", chosen=True)
    return state


def thinking_line(row, settings):
    """Shared display text; unknown installed checkpoints have no line."""
    state = thinking_state(row, settings)
    return thinking.deployment_text(state) if state else None


def rendezvous_head(fabric, container, observed):
    """Resolve a master to a Spark, preferring ownership on the worker's cable.

    Unique fabric/host addresses also identify masters reached through relays.
    Legacy reports without address ownership may use a single observed rank zero;
    multiple possible owners are never combined into an invented distributed run.
    """
    master = container.get("master_addr")
    positions = {p["position"]: p for p in fabric["positions"]}
    local = positions[container["position"]]
    owners = {p["position"] for p in positions.values()
              if master in p.get("addresses", [p.get("address")])
              or master == p.get("hostname")}
    if container.get("node_rank") == 0:
        return container["position"] if not owners or container["position"] in owners else None
    neighbors = {n["position"] for n in local.get("neighbors", []) if n["address"] == master}
    if len(neighbors) == 1:
        return next(iter(neighbors))
    if len(owners) == 1:
        return next(iter(owners))
    if owners or neighbors:
        return None
    heads = {c["position"] for c in observed if c.get("node_rank") == 0
             and (c.get("engine"), c.get("master_addr"), c.get("master_port")) ==
                 (container.get("engine"), master, container.get("master_port"))}
    return next(iter(heads)) if len(heads) == 1 else None


def parallelism(facts):
    """Display only parallel sizes present in the public launch facts."""
    return " / ".join(f"{k.upper()}{facts[k]}" for k in ("tp", "dcp", "pp", "dp") if k in facts)


def unmanaged_groups(fabric, documents, deployments, now):
    """Join fresh launch facts by head Spark, or GPU jobs by image and rendezvous."""
    from runtime.host.node_containers import MAX_CONTAINERS, thinking_kwargs
    groups, observed = {}, []
    known = {(c["position"], c["name"]) for row in deployments if not row["stopped"] for c in row["containers"]}
    for position in fabric["positions"]:
        candidates = [d for d in documents if isinstance(d, dict) and d.get("node_id") == position.get("node_id")
                      and position.get("node_id")]
        candidates = [d for d in candidates if number(d.get("observed_at"))]
        if not candidates:
            continue
        doc = max(candidates, key=lambda d: d["observed_at"])
        interval = doc.get("interval_seconds")
        interval = min(3600, max(30, interval)) if number(interval) else 30
        if not -5 <= now - doc["observed_at"] <= max(90, interval * 2 + 30):
            continue
        rows = doc.get("model_containers")
        for container in rows[:MAX_CONTAINERS] if isinstance(rows, list) else []:
            if not isinstance(container, dict) or container.get("managed") is not False:
                continue
            name = container.get("name")
            if not isinstance(name, str) or not CONTAINER_RE.fullmatch(name) or (position["position"], name) in known:
                continue
            # Revalidate the small public fact schema, including numeric types.
            c = {k: container[k][:512] for k in ("image", "image_id", "engine", "served_model", "master_addr", "api_bind")
                 if isinstance(container.get(k), str)}
            c.update({k: container[k] for k in ("tp", "pp", "dp", "dcp", "nnodes", "node_rank", "master_port", "api_port")
                      if type(container.get(k)) is int and (0 if k == "node_rank" else 1) <= container[k] <= 65535})
            c.update(name=name, position=position["position"], headless=container.get("headless") is True)
            kwargs = thinking_kwargs(container.get("default_chat_template_kwargs"))
            if kwargs is not None:
                c["default_chat_template_kwargs"] = kwargs
            master = c.get("master_addr")
            if master:
                try:
                    parsed = ipaddress.ip_address(master)
                    master = None if parsed.is_loopback or parsed.is_unspecified else str(parsed)
                except ValueError:
                    master = master.lower().rstrip(".")
                    if master == "localhost":
                        master = None
            c["master_addr"] = master
            observed.append(c)
    for c in observed:
        head = rendezvous_head(fabric, c, observed) if c.get("master_addr") and c.get("master_port") else None
        key = (("gpu-job", c.get("image_id") or c.get("image"), c["master_addr"], c["master_port"])
               if c.get("engine") == "unknown" and c.get("master_addr") and c.get("master_port") else
               (c.get("engine"), "head", head, c["master_port"]) if head is not None else
               ("single", c["position"], c["name"]))
        groups.setdefault(key, []).append(c)
    result = []
    for key, containers in groups.items():
        containers.sort(key=lambda c: (c.get("node_rank", 65536), c["position"], c["name"]))
        positions = sorted({c["position"] for c in containers})
        ranks = {c["node_rank"] for c in containers if "node_rank" in c}
        expected = max([c.get("nnodes", 1) for c in containers] + [max(ranks, default=0) + 1])
        heads = [c for c in containers if c.get("engine") in ("vllm", "sglang")
                 and c.get("node_rank", 0 if expected == 1 else -1) == 0 and not c["headless"]]
        head = heads[0] if len(heads) == 1 else None
        info = head or containers[0]
        head_position = head["position"] if head else None
        head_spark = next((p for p in fabric["positions"] if p["position"] == head_position), {})
        bind = (head or {}).get("api_bind")
        host = (head_spark.get("address") if not bind or bind in ("0.0.0.0", "::") else
                bind if bind in head_spark.get("addresses", [head_spark.get("address")]) else None)
        complete = len(ranks) == expected and len(positions) >= expected
        if expected == 1:
            complete = True
        summary = None if complete else f"incomplete: {len(positions)} of {expected} Sparks reporting"
        digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
        result.append({"id": "unmanaged-" + digest, "unmanaged": True, "model": head.get("served_model") if head else None,
                       "positions": positions, "head_position": head_position, "api_host": host,
                       "api_port": head.get("api_port") if head and host else None,
                       "stopped": False, "containers": containers, "launch": info,
                       "expected_sparks": expected, "group_note": summary,
                       "stopped_records": [r for r in deployments if r["stopped"] and set(r["positions"]) & set(positions)]})
    return result


class Source:
    """One fabric's host files: its fabric document and its site file, re-read when they change."""

    def __init__(self, fabric_path: Path, site_path: Path):
        self.fabric_path, self.site_path = Path(fabric_path), Path(site_path)
        self.stamps = None
        self.fabric = None
        self.deployments: list = []
        self.problem = None
        self.display_name = None

    def _stamp(self, path):
        """(device, inode, modification time, size): a file replaced by rename gets another inode
        even when its time and size repeat."""
        try:
            stat = path.stat()
            return (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        except OSError:
            return None

    def refresh(self, force: bool = False) -> bool:
        """Re-read both files when either one's stamp changed, or when `force`; True when they were
        re-read."""
        stamps = (self._stamp(self.fabric_path), self._stamp(self.site_path))
        if stamps == self.stamps and not force:
            return False
        self.stamps = stamps
        self.display_name = None
        try:
            self.fabric = fabric_from_document(read_json(self.fabric_path))
        except (OSError, ValueError) as error:
            self.fabric, self.deployments = None, []
            self.problem = f"fabric document {self.fabric_path}: {error}"
            return True
        try:
            site = read_json(self.site_path)
            self.deployments = deployments_from_site(site, self.fabric["size"])
            from runtime.host import ring_name
            if site.get("display_name") is not None:
                self.display_name = ring_name.validate(site["display_name"])
            self.problem = None
        except FileNotFoundError:
            self.deployments = []
            self.problem = (f"no site file at {self.site_path}; on Node A, sudo sparkring dashboard export "
                            "writes it from the deployment records")
        except (OSError, ValueError) as error:
            self.deployments = []
            self.problem = f"site file {self.site_path}: {error}"
        return True

    def ring_id(self) -> str:
        cluster = (self.fabric or {}).get("cluster") or "fabric"
        return "ring-" + re.sub(r"[^A-Za-z0-9_.-]", "-", cluster)[:40]


# -- one model --------------------------------------------------------------------------------
class Model:
    """One deployment's API endpoint: raw observations and state."""

    def __init__(self, source: Source, row: dict):
        self.source = source
        self.row = row
        self.key = (source.ring_id(), row["id"])
        self.id = f"{source.ring_id()}:{row['id']}"
        self.state = None
        self.state_since = None
        self.model = row.get("model")
        self.max_model_len = None
        self.metrics = None
        self.capacity = None
        self.runtime = None
        self.run_key = None           # identifies the model run whose runtime status was read
        self.runtime_due = True
        self.runtime_requested_at = None
        self.next_metrics = 0.0
        self.next_models = 0.0
        self.last_ok = None
        self.models_ok_at = None
        self.metrics_at = None

    def observe_metrics(self, metrics: dict, now: float) -> None:
        self.metrics = metrics
        self.metrics_at = now
        self.capacity = capacity_from_cache_info(metrics.get("cache_info")) or self.capacity
        if metrics.get("model_name"):
            self.model = metrics["model_name"]
        run_key = (self.model, metrics.get("started"))
        if run_key != self.run_key:
            self.run_key, self.runtime_due = run_key, True
        self.last_ok = now


# -- the collector ----------------------------------------------------------------------------
class Collector:
    """Decides which reads are due, sends them through the client and keeps the results."""

    def __init__(self, sources, client: ReadOnlyClient, *, quiet_path: Path = QUIET_PATH,
                 node_status_path: Path | None = NODE_STATUS_PATH, recovery_path: Path | None = RECOVERY_PATH,
                 node_reports_path: Path | None = NODE_REPORTS_PATH,
                 clock=time.time, cpu_clock=time.thread_time):
        self.sources = list(sources)
        self.client = client
        self.quiet_path = Path(quiet_path) if quiet_path else None
        self.node_status_path = Path(node_status_path) if node_status_path else None
        self.recovery_path = Path(recovery_path) if recovery_path else None
        self.node_reports_path = Path(node_reports_path) if node_reports_path else None
        self.node_reports = []
        self.recovery: dict = {}      # deployment directory name -> the recovery timer's latest finding
        self.clock, self.cpu_clock = clock, cpu_clock
        self.lock = threading.Lock()
        self.models: dict = {}
        self.events: collections.deque = collections.deque(maxlen=MAX_EVENTS)
        self.started = clock()
        self.last_view = None
        self.next_files = 0.0
        self.reload_requested = threading.Event()   # SIGHUP: re-read the host files at the next cycle
        self.last_cycle = None
        self.quiet = False
        self.node_status = None
        self.cycles = 0
        self.cycle_cpu = collections.deque(maxlen=120)   # CPU seconds of recent cycles that sent requests
        self.history = {}
        self.history_seq = 0
        self.history_gaps = set()

    # -- inputs ----------------------------------------------------------------------------
    def viewed(self, now) -> bool:
        return self.last_view is not None and now - self.last_view <= VIEWER_SECONDS

    def mark_viewed(self, now=None) -> None:
        """A browser fetched the status: poll at the viewed cadence from now on."""
        now = self.clock() if now is None else now
        with self.lock:
            was = self.viewed(now)
            self.last_view = now
            if not was:
                for model in self.models.values():
                    model.next_metrics = min(model.next_metrics, now)

    def request_runtime(self, model_id: str, now=None) -> bool:
        """Read one model's runtime status again at the next cycle; at most once per RUNTIME_REFRESH_SECONDS."""
        now = self.clock() if now is None else now
        with self.lock:
            model = self.models.get(model_id)
            if model is None or (model.runtime_requested_at is not None
                                 and now - model.runtime_requested_at < RUNTIME_REFRESH_SECONDS):
                return False
            model.runtime_requested_at, model.runtime_due = now, True
            return True

    def request_reload(self) -> None:
        """Re-read every host file at the next cycle (the SIGHUP handler; systemctl reload)."""
        self.reload_requested.set()

    def read_status_files(self) -> None:
        self.node_reports = []
        if self.node_reports_path is not None:
            try:
                doc = read_json(self.node_reports_path)
                if doc.get("schema") == "sparkring-dashboard-nodes/v1" and isinstance(doc.get("nodes"), list):
                    self.node_reports = doc["nodes"][:256]
            except (OSError, ValueError, AttributeError):
                pass
        if self.node_status_path is not None:
            try:
                self.node_status = read_json(self.node_status_path)
            except (OSError, ValueError):
                self.node_status = None
        if self.recovery_path is not None:
            self.recovery = recovery_findings(self.recovery_path)

    def reload_sources(self, force: bool = False) -> None:
        """Re-read each fabric's document and site file whose stamp changed (all of them when
        `force`), and follow the deployments they list: a model per row, kept across re-reads."""
        for source in self.sources:
            source.refresh(force)
        hosts, wanted = set(), {}
        for source in self.sources:
            unmanaged = unmanaged_groups(source.fabric, [*self.node_reports, self.node_status], source.deployments,
                                         self.last_cycle if self.last_cycle is not None else self.clock()) if source.fabric else []
            hidden = {r["id"] for group in unmanaged for r in group["stopped_records"]}
            for row in [r for r in source.deployments if r["id"] not in hidden] + unmanaged:
                if row["api_host"]:
                    hosts.add(row["api_host"])
                wanted[(source.ring_id(), row["id"])] = (source, row)
        self.client.allow(hosts)
        with self.lock:
            for model_id in [m for m, model in self.models.items() if model.key not in wanted]:
                del self.models[model_id]
            known = {model.key: model for model in self.models.values()}
            for key, (source, row) in wanted.items():
                if key in known:
                    if (known[key].row.get("api_host"), known[key].row.get("api_port"), known[key].row.get("containers")) != (
                            row.get("api_host"), row.get("api_port"), row.get("containers")) and row.get("unmanaged"):
                        fresh = Model(source, row)
                        self.models[fresh.id] = fresh
                        self.history_gaps.add(fresh.id)
                    known[key].row = row
                else:
                    model = Model(source, row)
                    self.models[model.id] = model

    # -- one cycle -------------------------------------------------------------------------
    def clock_went_back(self, now: float) -> None:
        """The wall clock stepped back, as an NTP correction after boot or a manual date does.

        Every schedule here is kept in that clock, so each read would otherwise wait out the whole
        step: the reads fall due now. Raw observations from the clock's future are invalidated,
        and a viewer or a state that the old clock dated later than now is dated now."""
        print(f"sparkring dashboard: the clock went back by {self.last_cycle - now:.0f} s; reads rescheduled",
              file=sys.stderr, flush=True)
        self.next_files = now
        with self.lock:
            self.history_gaps.update(self.models)
            if self.last_view is not None and self.last_view > now:
                self.last_view = now
            for model in self.models.values():
                model.next_metrics = min(model.next_metrics, now)
                model.next_models = min(model.next_models, now)
                if model.runtime_requested_at is not None and model.runtime_requested_at > now:
                    model.runtime_requested_at = None
                if model.state_since is not None and model.state_since > now:
                    model.state_since = now
                model.metrics_at = None

    def cycle(self, now=None) -> None:
        now = self.clock() if now is None else now
        if self.last_cycle is not None and now < self.last_cycle - CLOCK_STEP_SECONDS:
            self.clock_went_back(now)
        self.last_cycle = now
        # Host files first, in quiet mode too: two stats a second, and no request to any Spark.
        forced = self.reload_requested.is_set()
        if forced:
            self.reload_requested.clear()
        if forced or now >= self.next_files:
            self.read_status_files()
            self.next_files = now + FILES_SECONDS
        self.reload_sources(forced)
        # Quiet mode is checked on every cycle (a stat and a small directory listing a second), so a
        # hold takes effect before a measurement that starts right after it.
        self.quiet = bool(self.quiet_path and quiet_state(self.quiet_path, now)["quiet"])
        with self.lock:
            self.prune_history(now)
        if self.quiet:
            with self.lock:
                self.history_gaps.update(self.models)
            return
        before = self.cpu_clock()
        sent = False
        with self.lock:
            models = list(self.models.values())
            viewed = self.viewed(now)
        for model in models:
            sent = self.poll(model, now, viewed) or sent
            with self.lock:
                self.remember(model, now)
        if sent:
            self.cycle_cpu.append(self.cpu_clock() - before)
            self.cycles += 1

    def poll(self, model: Model, now: float, viewed: bool) -> bool:
        """Send the reads that are due for one model; True when any request was sent.

        A model whose /metrics answer is serving: vLLM's API server listens only once its engine
        is ready. A refused connection is a stopped model, a timeout an unreachable Spark and an
        HTTP error a model still starting. A /metrics answer that names a different model means
        another deployment of an overlapping slot holds the port, and this one is idle. A deployment
        whose last recorded operation stopped it is stopped whatever its port does."""
        row = model.row
        host, port = row["api_host"], row["api_port"]
        if row.get("unmanaged") and (host is None or port is None):
            self.set_state(model, "busy", now)
            return False
        sent = False
        plugin = self.plugin_snapshot(model, now)
        cadence = (METRICS_VIEWED_SECONDS if viewed and not row["stopped"] else
                   METRICS_PLUGIN_UNVIEWED_SECONDS if plugin or row["stopped"] else METRICS_UNVIEWED_SECONDS)
        if model.state == "serving" and model.metrics_at is not None:
            model.next_metrics = min(model.next_metrics, model.metrics_at + cadence)
        if now >= model.next_metrics:
            sent = True
            fetched = self.client.get(host, port, "/metrics")
            if fetched.ok:
                metrics = parse_metrics(fetched.body.decode("utf-8", "replace"))
                if not row.get("unmanaged") and row.get("model") and metrics.get("model_name") and metrics["model_name"] != row["model"]:
                    state = "idle"
                    with self.lock:
                        model.metrics = None
                        model.metrics_at = None
                else:
                    state = "serving"
                    with self.lock:
                        model.observe_metrics(metrics, now)
                model.next_metrics = now + cadence
            else:
                state = {"refused": "stopped", "timeout": "offline", "unreachable": "offline"}.get(
                    fetched.failure, "not_ready")
                if row["stopped"]:
                    state = "stopped"  # the records say its last operation stopped it
                if row.get("unmanaged"):
                    state = ("serving" if fetched.failure == "http" and model.models_ok_at is not None
                             and 0 <= now - model.models_ok_at < MODELS_SECONDS else "busy")
                    # SGLang can serve without --enable-metrics. Retain the same
                    # read paths and cadence even when its counters are absent.
                    if now >= model.next_models:
                        model.next_models = now + MODELS_SECONDS
                        listed = self.models_listed(self.client.get(host, port, "/v1/models"))
                        if listed:
                            state = "serving"
                            with self.lock:
                                model.models_ok_at = now
                                model.model = listed.get("id") or model.model
                                model.max_model_len = listed.get("max_model_len") or model.max_model_len
                with self.lock:
                    model.metrics = None
                    model.metrics_at = None
                model.next_metrics = now + max(cadence, FAILED_SECONDS)
            self.set_state(model, state, now)
        if model.state == "serving" and now >= model.next_models:
            sent = True
            model.next_models = now + MODELS_SECONDS
            listed = self.models_listed(self.client.get(host, port, "/v1/models"))
            if listed:
                with self.lock:
                    model.models_ok_at = now
                    model.model = listed.get("id") or model.model
                    model.max_model_len = listed.get("max_model_len") or model.max_model_len
        if model.state == "serving" and model.runtime_due:
            sent = True
            model.runtime_due = False
            fetched = self.client.get(host, port, "/v1/sparkring/status")
            runtime = None
            if fetched.ok:
                try:
                    runtime = read_runtime_status(json.loads(fetched.body))
                except (ValueError, AttributeError, TypeError):
                    runtime = None
            with self.lock:
                model.runtime = runtime
        return sent

    @staticmethod
    def models_listed(fetched: Fetched) -> dict | None:
        if not fetched.ok:
            return None
        try:
            data = json.loads(fetched.body).get("data") or []
        except (ValueError, AttributeError):
            return None
        return data[0] if data and isinstance(data[0], dict) else None

    def set_state(self, model: Model, state: str, now: float) -> None:
        with self.lock:
            if state == model.state:
                return
            previous, model.state, model.state_since = model.state, state, now
            if state != "serving":
                model.runtime = None if state != "not_ready" else model.runtime
            text = LABELS[state] + (f": {model.model}" if state == "serving" and model.model else "")
            if previous is not None or state == "serving":
                self.events.append({"at": now, "stack": run_name(model.row["positions"]), "text": text,
                                    "kind": state})

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.cycle()
            except Exception as error:  # one bad response must not stop the page
                print(f"sparkring dashboard: collection failed: {error!r}", file=sys.stderr, flush=True)
            stop.wait(1.0)

    # -- the status document -----------------------------------------------------------------
    def plugin_snapshot(self, model, now):
        fallback = {"run_id": (model.metrics or {}).get("started")}
        deployment = model.row.get("deployment_id")
        # The recovery timer also exports node reports with recovery disabled.
        # Read that local cache only; quiet mode must never trigger a remote read.
        reports = [self.node_status, *self.node_reports]
        candidates = [report["telemetry"].get(deployment) for report in reports
                      if isinstance(report, dict) and isinstance(report.get("telemetry"), dict)]
        candidates.append((self.recovery.get(model.row["id"]) or {}).get("telemetry"))
        return next((p for p in candidates if plugin_fresh(p, fallback, now)), None)

    def prune_history(self, now):
        for model_id in list(self.history):
            if model_id not in self.models:
                del self.history[model_id]
                continue
            rows = self.history[model_id]
            while rows and rows[0]["counters"]["at"] < now - HISTORY_SECONDS:
                rows.popleft()
            while rows and rows[-1]["observed_at"] > now:
                rows.pop()
        self.history_gaps.intersection_update(self.models)

    def remember(self, model, now):
        """Retain only bounded raw numeric fields, never rates or mutable source documents.

        Each sample carries exactly one source. A fresh stat-logger snapshot is
        stored as the plugin payload alone; only a cycle without one stores the
        /metrics read, so the label and the values can never diverge and no
        sample mixes plugin and /metrics counters.
        """
        if model.id not in self.history:
            if len(self.history) >= HISTORY_DEPLOYMENTS:
                return
            self.history[model.id] = collections.deque(maxlen=HISTORY_SAMPLES)
        rows = self.history[model.id]
        plugin = self.plugin_snapshot(model, now)
        if not plugin and (model.state != "serving" or not number(model.metrics_at)):
            self.history_gaps.add(model.id)
            return
        metrics = model.metrics or {}
        if plugin:
            sample = {k: plugin[k] for k in (*HISTORY_VALUES, "at", "monotonic") if number(plugin.get(k))} | {
                "source": "plugin", "schema": "sparkring-stats/v1", "run_id": plugin["run_id"],
                "step_timing": bool(plugin.get("step_timing"))}
        else:
            sample = {"source": "metrics", "at": model.metrics_at, "run_id": metrics.get("started"),
                      "values": {k: metrics[k] for k in HISTORY_VALUES if number(metrics.get(k))}}
        at = sample["at"]
        if not number(at) or now - at > (180 if plugin else 125):
            self.history_gaps.add(model.id)
            return
        # Snapshot identity is independent of the collector's read time and fallback reads.
        previous = rows[-1]["counters"] if rows else None
        if previous and (rows[-1]["source"], previous["run_id"],
                         previous.get("monotonic", previous["at"])) == (
                             sample["source"], sample["run_id"], sample.get("monotonic", at)):
            return
        self.history_seq += 1
        rows.append({"seq": self.history_seq, "observed_at": now, "source": sample["source"],
                     "counters": sample, "boundary": model.id in self.history_gaps})
        self.history_gaps.discard(model.id)

    def history_snapshot(self, now=None):
        now = self.clock() if now is None else now
        with self.lock:
            self.prune_history(now)
            return {"generated_at": now, "service_started": self.started, "history_cursor": self.history_seq,
                    "counter_history": {key: list(rows) for key, rows in self.history.items()}}

    def snapshot(self, now=None, since=None) -> dict:
        """The sparkring-dashboard/v1 document: the fields the page reads, in the shape it reads them."""
        now = self.clock() if now is None else now
        with self.lock:
            models = list(self.models.values())
            groups, rings = [], []
            color = 0
            for source in self.sources:
                fabric = source.fabric
                members = []
                for model in sorted(models, key=lambda m: (min(m.row["positions"]), m.id)):
                    if model.source is source:
                        member = self.member(model, now, color)
                        rows = self.history.get(model.id, ())
                        delta = list(rows)[-1:] if since is None else [r for r in rows if r["seq"] > since]
                        if len(delta) > HISTORY_DELTA_SAMPLES:
                            delta = delta[-HISTORY_DELTA_SAMPLES:]
                            delta[0] = {**delta[0], "boundary": True}
                        member["counter_samples"] = delta
                        members.append(member)
                        color += 1
                members.sort(key=lambda m: m["arc"]["first"])
                groups.append({"id": source.ring_id(), "name": self.ring_name(source), "kind": "ring",
                               "members": members})
                if fabric:
                    rings.append(self.ring(source, members, now))
            events = list(self.events)[::-1][:60]
            quiet, viewed = self.quiet, self.viewed(now)
            cpu = list(self.cycle_cpu)
            cursor = self.history_seq
        problems = [s.problem for s in self.sources if s.problem]
        topology = {"source": "fabric" if rings else "none", "detail": "; ".join(problems) or None,
                    "rings": len(rings)}
        counters = self.client.counters()
        return {"schema": SCHEMA, "title": rings[0]["name"] if len(rings) == 1 else "SparkRing",
                "generated_at": now, "service_started": self.started,
                "history_cursor": cursor,
                "poll_seconds": METRICS_VIEWED_SECONDS, "groups": groups, "rings": rings,
                "ring": rings[0] if len(rings) == 1 else None, "topology": topology, "events": events,
                "node": self.node_summary(),
                "collector": {"quiet": quiet, "viewed": viewed, "cycles": self.cycles,
                              "metrics_seconds": METRICS_VIEWED_SECONDS if viewed else METRICS_UNVIEWED_SECONDS,
                              "models_seconds": MODELS_SECONDS,
                              "cpu_ms_per_cycle": round(sum(cpu) / len(cpu) * 1000, 2) if cpu else None,
                              "requests": counters["requests"], "received_bytes": counters["received_bytes"],
                              "uptime_seconds": now - self.started}}

    def ring_name(self, source) -> str:
        fabric = source.fabric
        if not fabric:
            return "fabric"
        return source.display_name or fabric.get("cluster") or "sparkring"

    def member(self, model: Model, now: float, color: int) -> dict:
        row, fabric = model.row, model.source.fabric or {"positions": []}
        hostnames = {p["position"]: p for p in fabric["positions"]}
        positions = row["positions"]
        runtime = model.runtime or {}
        metrics = model.metrics or {}
        serving = model.state == "serving"
        nodes = runtime.get("nodes") or []
        head_position = row.get("head_position", positions[0])
        head = hostnames.get(head_position, {})
        port = row["api_port"]
        endpoint = row.get("display_url") or f"http://{row['api_host']}:{port}/v1"
        if not row["api_host"] or not port:
            endpoint = None
        since = runtime.get("started_at") if serving and runtime.get("started_at") else model.state_since
        capacity = dict(model.capacity) if model.capacity else None
        if capacity and runtime.get("settings"):
            capacity["max_num_seqs"] = runtime["settings"].get("max_num_seqs")
        container = next((c["name"] for c in row["containers"] if c["position"] == head_position), None)
        launch = row.get("launch") or {}
        gpu_job = bool(row.get("unmanaged") and launch.get("engine") == "unknown")
        sparks = []
        for position in positions:
            spark = hostnames.get(position, {})
            up = True if serving else (False if model.state == "offline" and position == positions[0] else None)
            sparks.append({"name": spark.get("hostname", str(position)), "address": spark.get("address"),
                           "lan_address": spark.get("lan_address"), "up": up, "position": position})
        thought = None if gpu_job else thinking_state(row, runtime.get("settings"))
        return {
            "id": model.id, "name": ("Spark " if len(positions) == 1 else "Sparks ") + ", ".join(map(str, positions))
            if row.get("unmanaged") else run_name(positions),
            "profile": row.get("profile"),
            "start_command": start_command(row),
            "unmanaged": bool(row.get("unmanaged")), "gpu_job": gpu_job, "launch": launch,
            "ownership_label": "Not installed by SparkRing: no automatic recovery, no install record" if row.get("unmanaged") else None,
            "group_note": row.get("group_note"), "containers": row["containers"] if row.get("unmanaged") else [],
            "stopped_records": row.get("stopped_records", []),
            "state": model.state, "state_label": "GPU in use" if gpu_job else LABELS.get(model.state, "Checking"), "state_since": since,
            "model": model.model, "model_short": short_model(model.model), "max_model_len": model.max_model_len,
            "endpoint": endpoint,
            # The runtime-status view asks the workers for a report every 5 s while it is open.
            "details_url": f"http://{row['api_host']}:{port}/v1/sparkring/status/view" if model.runtime else None,
            "layout": {"label": parallelism(launch)
                       if row.get("unmanaged") else layout_label(len(positions)) if serving else None, "count": len(positions),
                       "tp": launch.get("tp") if row.get("unmanaged") else runtime.get("tp")},
            "live": {key: metrics.get(key) for key in ("running", "waiting", "kv")} if model.metrics else None,
            "counters": {"source": "metrics", "at": model.metrics_at, "run_id": metrics.get("started"),
                         "values": {key: value for key, value in metrics.items()
                                    if isinstance(value, (int, float))}} if model.metrics else None,
            "plugin_counters": self.plugin_snapshot(model, now),
            "totals": {"finished": metrics.get("finished"), "preemptions": metrics.get("preemptions"),
                       "output_tokens": metrics.get("gen"), "prompt_tokens": metrics.get("prompt")}
            if model.metrics else None,
            "thinking": thinking.deployment_text(thought) if thought else None,
            "thinking_default": thought, "thinking_provenance": thinking.provenance_text(thought),
            "settings": runtime.get("settings"), "nodes": nodes, "capacity": capacity if serving else None,
            "workers": runtime.get("workers"), "image": runtime.get("image"), "transport": runtime.get("transport"),
            "sparks": sparks,
            "history": [],
            "arc": {"positions": positions, "first": positions[0], "last": positions[-1], "size": len(positions),
                    "run": run_name(positions), "source": "agent" if row.get("unmanaged") else "deployment",
                    "source_label": "agent launch facts" if row.get("unmanaged") else "deployment record",
                    "inferred": False, "detail": None, "color": color},
            "ring": model.source.ring_id(),
            "head": {"position": head_position, "name": head.get("hostname"), "address": row["api_host"], "port": port,
                     "container": container},
            "logs": log_commands(head.get("hostname"), container),
            "runtime_read": model.runtime is not None,
            "recovery": {k: v for k, v in self.recovery[row["id"]].items() if k != "telemetry"}
            if row["id"] in self.recovery else None,
        }

    def ring(self, source: Source, members: list, now: float) -> dict:
        fabric = source.fabric
        owners: dict = {}
        for member in members:
            for position in member["arc"]["positions"]:
                owners.setdefault(position, []).append(member["id"])
        local = socket.gethostname()
        positions = []
        for row in fabric["positions"]:
            busy = owners.get(row["position"]) or []
            serving = any(m["state"] == "serving" and m["id"] in busy for m in members)
            up = True if serving or row["hostname"] == local else None
            positions.append({"position": row["position"], "name": row["hostname"],
                              "short": row["hostname"].removeprefix("spark-"), "address": row["address"],
                              "up": up, "since": None, "member": busy[0] if busy else None, "members": busy,
                              "run": None})
        first = fabric["positions"][0]["hostname"]
        return {"id": source.ring_id(), "name": self.ring_name(source), "node_a": first,
                "short": first.removeprefix("spark-"), "size": fabric["size"], "closed": fabric["closed"],
                "source": "fabric",
                "fabric": {"id": fabric["id"], "cluster": fabric["cluster"], "shape": fabric["shape"],
                           "verified": fabric["verified"], "read_from": [str(source.fabric_path)]},
                "positions": positions, "runs": [],
                "discovery": {"ports": sorted({m["head"]["port"] for m in members if m["head"]["port"]}),
                              "every_seconds": METRICS_VIEWED_SECONDS if self.last_view and self.viewed(now)
                              else METRICS_UNVIEWED_SECONDS, "at": now}}

    def node_summary(self) -> dict | None:
        status = self.node_status
        if not isinstance(status, dict):
            return None
        return {key: status.get(key) for key in ("hostname", "state", "next_action", "observed_at", "warnings", "error")}


def recovery_findings(path: Path) -> dict:
    """{deployment directory name: {state, summary, next_action, checked_at, enabled}} from the
    recovery timer's sparkring-recovery/v1 record; empty when it is absent or unreadable."""
    try:
        document = read_json(path)
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict) or document.get("schema") != "sparkring-recovery/v1":
        return {}
    found = {}
    for directory, record in (document.get("deployments") or {}).items():
        if not isinstance(record, dict):
            continue
        finding = record.get("finding") if isinstance(record.get("finding"), dict) else {}
        found[Path(str(directory)).name] = {
            "state": finding.get("state"), "summary": finding.get("summary"),
            "next_action": finding.get("next_action"), "checked_at": record.get("checked_at"),
            "enabled": record.get("enabled", True), "stopped": bool(record.get("stopped")),
            "telemetry": finding.get("telemetry")}
    return found


def start_command(row):
    """Name the saved deployment, including inactive records that share a slice."""
    if row.get("unmanaged"):
        return None
    profile, identity = row.get("profile"), row.get("id", "")
    if isinstance(profile, str) and profile and (identity == profile or identity.startswith(profile + "-")):
        command = ["sudo", "sparkring", "up", profile]
        instance = identity[len(profile):].removeprefix("-")
        if instance:
            command += ["--instance", instance]
    else:
        command = ["sudo", "sparkring", "up", "--on", ",".join(map(str, row["positions"]))]
    return shlex.join(command)


def log_commands(hostname, container) -> dict:
    """How the operator prints a model's log on demand: the commands, not their output.

    The dashboard holds no SSH key and no Docker access; the operator runs these on the
    head Spark with their own login, so nothing reads a log unless someone asks."""
    if not container:
        return {"enabled": False, "host": hostname, "commands": []}
    return {"enabled": True, "host": hostname, "commands": [f"sudo docker logs --tail {LOG_LINES} {container}"]}


# -- process placement -------------------------------------------------------------------------
def efficiency_cpus(root: Path | None = None, cpuinfo: Path | None = None) -> set | None:
    """The CPUs outside the fastest class (on GB10 the Cortex-A725 cores), or None on a host whose
    cores are all of one class: runtime.host.host_load.efficiency_cpus, which the host agent's and
    the recovery timer's placement drop-ins use too."""
    from runtime.host import host_load
    return host_load.efficiency_cpus(root, cpuinfo)


def lower_priority(cpu_list: str | None = None) -> list:
    """Move this thread, and the threads it starts later, to idle CPU scheduling, nice 19 and the
    efficiency cores (or `cpu_list`); returns what changed.

    Call it before any other thread starts: Linux applies each setting to the calling thread and
    new threads inherit it. Each step is skipped where the platform lacks it. The systemd unit
    (packaging/debian/sparkring-dashboard.service) sets the same limits; doing it here as well
    covers a dashboard started by hand or on another host."""
    done = []
    if hasattr(os, "sched_setscheduler") and hasattr(os, "SCHED_IDLE"):
        try:
            os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
            done.append("idle scheduling")
        except OSError:
            pass
    if hasattr(os, "nice"):
        try:
            os.nice(max(0, 19 - os.nice(0)))
            done.append("nice 19")
        except OSError:
            pass
    if hasattr(os, "sched_setaffinity"):
        from spark_transport.sircl.sparkring_sircl import cpus
        chosen = set(cpus.parse_cpu_list(cpu_list)) if cpu_list else efficiency_cpus()
        if chosen:
            try:
                os.sched_setaffinity(0, chosen)
                done.append("CPUs " + cpus.format_cpu_list(chosen))
            except OSError:
                pass
    return done


# -- the web server ----------------------------------------------------------------------------
def page_bytes(path: Path = PAGE) -> tuple[bytes, str]:
    """The page and its Content-Security-Policy: its inline script and style admitted by hash only."""
    text = Path(path).read_text(encoding="utf-8")
    text = text.replace("// VIEWER_RATES", Path(__file__).with_name("dashboard_rates.js").read_text(encoding="utf-8"))
    policies = []
    for tag, directive in (("script", "script-src"), ("style", "style-src")):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
        digest = base64.b64encode(hashlib.sha256(match.group(1).encode()).digest()).decode()
        policies.append(f"{directive} 'sha256-{digest}'")
    csp = "; ".join(["default-src 'none'", *policies, "style-src-attr 'unsafe-inline'", "connect-src 'self'",
                     "img-src 'self' data:", "base-uri 'none'", "frame-ancestors 'none'", "form-action 'none'"])
    return text.encode("utf-8"), csp


def host_allowed(header: str | None, names) -> bool:
    """Whether a request's Host header names this server: an IP literal, `localhost` or one of `names`.

    A browser on the LAN that a hostile web page points at the dashboard through DNS
    rebinding sends that page's host name, which is refused. `localhost` is no page's name:
    browsers resolve it to loopback themselves."""
    if not header:
        return False
    host = header.strip()
    if host.startswith("["):
        host = host[1:host.find("]")] if "]" in host else host
    else:
        host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return _ip(host) or host.lower() == "localhost" or host.lower() in names


def make_handler(collector: Collector, page: bytes, csp: str, names=frozenset(), password: str | None = None,
                 manager=None):
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
               "X-Frame-Options": "DENY"}
    expected = None if password is None else "Basic " + base64.b64encode(f"sparkring:{password}".encode()).decode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "sparkring-dashboard"

        def do_GET(self):
            if not host_allowed(self.headers.get("Host"), names):
                return self._send(421, b"This dashboard answers to its IP address or a name given with --name.\n",
                                  "text/plain; charset=utf-8")
            if (expected is not None and self.headers.get("Authorization") != expected
                    and not (manager and manager.authenticated(self.headers))):
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="SparkRing dashboard", charset="UTF-8"')
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            path, _, query = self.path.partition("?")
            params = {k: v[-1] for k, v in urllib.parse.parse_qs(query).items()}
            if path == "/":
                return self._send(200, page, "text/html; charset=utf-8", {"Content-Security-Policy": csp})
            if path == "/api/status":
                if params.get("viewer") != "passive":   # sparkring dashboard tui --passive
                    collector.mark_viewed()
                try:
                    since = int(params["since"]) if "since" in params else None
                except ValueError:
                    return self._json(400, {"error": "Invalid history cursor"})
                snapshot = collector.snapshot(since=since)
                return self._json(200, {**snapshot, "manage": True} if manager else snapshot)
            if path == "/api/history":
                body = json.dumps(collector.history_snapshot(), separators=(",", ":")).encode()
                extra = {"Vary": "Accept-Encoding"}
                if "gzip" in self.headers.get("Accept-Encoding", ""):
                    body = gzip.compress(body, compresslevel=1, mtime=0)
                    extra["Content-Encoding"] = "gzip"
                return self._send(200, body, "application/json", extra)
            if path == "/api/manage" and manager:
                try:
                    return self._json(200, manager.session(self.headers))
                except PermissionError as error:
                    return self._json(401, {"error": str(error)})
            if path == "/api/refresh":
                accepted = collector.request_runtime(params.get("model", ""))
                return self._json(202 if accepted else 429, {"accepted": accepted})
            if path == "/healthz":
                return self._send(200, b"ok\n", "text/plain")
            return self._send(404, b"not found\n", "text/plain")

        def _refuse(self):
            self._send(405, b"read-only\n", "text/plain", {"Allow": "GET"})

        do_PUT = do_DELETE = do_PATCH = _refuse

        def do_POST(self):
            if manager is None:
                return self._refuse()
            if not host_allowed(self.headers.get("Host"), names):
                return self._json(421, {"error": "Host refused"})
            if not manager.authenticated(self.headers):
                return self._json(401, {"error": "Authentication required"})
            if self.headers.get("Content-Type") != "application/json" or self.headers.get("Transfer-Encoding"):
                return self._json(415, {"error": "Use a bounded JSON request"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    return self._json(413, {"error": "Action request is too large or empty"})
                self.connection.settimeout(15)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request")
                result = manager.dispatch(self.path, self.headers, json.loads(raw))
                return self._json(200, result)
            except PermissionError as error:
                return self._json(403, {"error": str(error)})
            except (ValueError, KeyError, TypeError) as error:
                return self._json(400, {"error": str(error)})
            except OSError as error:
                return self._json(503, {"error": "Helper unavailable: " + str(error)})

        def _json(self, code, value):
            self._send(code, json.dumps(value, separators=(",", ":")).encode(), "application/json")

        def _send(self, code, body, kind, extra=None):
            self.send_response(code)
            for name, value in {**headers, **(extra or {}), "Content-Type": kind,
                                "Content-Length": str(len(body))}.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def default_bind(source: Source) -> str:
    """On Node A, its LAN address from the fabric document (where its model API serves; else its
    management address); on any other host, loopback. ``listen_addresses`` adds loopback to it."""
    rows = (source.fabric or {}).get("positions") or []
    if rows and socket.gethostname() == rows[0]["hostname"]:
        return rows[0].get("lan_address") or rows[0].get("address") or "127.0.0.1"
    return "127.0.0.1"


def service_url(fabric_path: Path = FABRIC_PATH) -> str:
    """The address of the dashboard service on this Spark, by the service's default bind rule
    (``default_bind``): Node A's LAN address, else loopback. A ``--bind`` given to the service
    is not visible here."""
    source = Source(fabric_path, SITE_PATH)
    source.refresh()
    bind = default_bind(source)
    literal = f"[{bind}]" if ":" in bind else bind
    return f"http://{literal}:{DEFAULT_PORT}"


def listen_addresses(bind: str) -> list:
    """The addresses to listen on: `bind`, then 127.0.0.1 unless `bind` is a loopback address or a
    wildcard (0.0.0.0, ::), which already admits loopback connections."""
    try:
        literal = ipaddress.ip_address(bind)
    except ValueError:
        return [bind, "127.0.0.1"]           # a host name: its address is not known here
    if literal.is_loopback or literal.is_unspecified:
        return [bind]
    return [bind, "127.0.0.1"]


def open_servers(addresses, port: int, handler, *, log=None) -> list:
    """One HTTP server per address. The first address is the configured one and its failure raises;
    a later one that cannot be bound (port taken, no such address) costs one line on stderr."""
    servers = []
    for index, address in enumerate(addresses):
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        server_class = type("DashboardServer", (ThreadingHTTPServer,), {"address_family": family})
        try:
            server = server_class((address, port), handler)
        except OSError as error:
            if index == 0:
                raise
            (log or (lambda line: print(line, file=sys.stderr, flush=True)))(
                f"sparkring dashboard: not listening on {address}:{port} ({error}); "
                f"{addresses[0]}:{port} serves the page")
            continue
        server.daemon_threads = True
        servers.append(server)
    return servers


def url_of(address: str, port: int) -> str:
    return f"http://[{address}]:{port}/" if ":" in address else f"http://{address}:{port}/"


def reload_on_hangup(collector: Collector, signals=signal) -> bool:
    """Make SIGHUP (systemctl reload) re-read the host files at the collector's next cycle; False on
    a platform without SIGHUP. Python's default for SIGHUP ends the process, and systemd counts an
    end by SIGHUP as clean, so a reload before this ran would leave the service stopped: serve
    installs it before anything else."""
    if not hasattr(signals, "SIGHUP"):
        return False
    signals.signal(signals.SIGHUP, lambda number, frame: collector.request_reload())
    return True


def serve(args) -> int:
    manager = None
    if getattr(args, "manage", False):
        if not hasattr(os, "geteuid") or os.geteuid() == 0:
            raise ValueError("Dashboard manage mode must run as an unprivileged user")
        if not args.manage_token_file or args.fabric or args.site:
            raise ValueError("Manage mode requires --manage-token-file and this controller's default fabric/site")
        from runtime.host.dashboard_actions import Manager
        manager = Manager(Path(args.manage_token_file).read_text(encoding="utf-8").strip())
    pairs = list(zip(args.fabric, args.site)) if args.fabric else [(FABRIC_PATH, SITE_PATH)]
    sources = [Source(Path(f), Path(s)) for f, s in pairs]
    client = ReadOnlyClient()
    collector = Collector(sources, client, quiet_path=Path(args.quiet_file),
                          node_status_path=None if args.no_node_status else NODE_STATUS_PATH,
                          recovery_path=RECOVERY_PATH)
    reload_on_hangup(collector)
    placement = lower_priority(args.cpus)
    collector.cycle()
    password = None
    if args.password_file:
        password = Path(args.password_file).read_text(encoding="utf-8").strip()
        if not password:
            raise ValueError(f"{args.password_file} is empty")
    bind = args.bind or default_bind(sources[0])
    page, csp = page_bytes()
    names = frozenset(name.lower() for name in args.name or ())
    servers = open_servers(listen_addresses(bind), args.port, make_handler(collector, page, csp, names, password, manager))
    stop = threading.Event()
    threading.Thread(target=collector.run, args=(stop,), daemon=True, name="collector").start()
    for server in servers[1:]:
        threading.Thread(target=server.serve_forever, daemon=True, name="listener").start()
    where = " and ".join(url_of(server.server_address[0], server.server_address[1]) for server in servers)
    print(f"SparkRing dashboard on {where} ({', '.join(placement) or 'default priority'}); "
          f"{len(collector.models)} model API(s) from {len(sources)} fabric(s)", flush=True)
    try:
        servers[0].serve_forever()
    finally:
        stop.set()
    return 0


def quiet_holders_directory(quiet_path: Path) -> Path:
    return Path(quiet_path).parent / QUIET_HOLDERS


def _process_alive(pid) -> bool:
    """Whether process `pid` of this host runs. Only POSIX can ask without side effects; elsewhere
    a holder's process counts as running and its time to live ends it."""
    if os.name != "posix":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True        # another user's process, such as a root harness seen by the dashboard's user
    return True


def quiet_state(quiet_path: Path, now: float, *, alive=_process_alive) -> dict:
    """{"quiet", "operator", "holders"}: whether quiet mode is held, by the operator's quiet file and
    by which live measurement holders."""
    operator = Path(quiet_path).exists()
    holders = []
    try:
        entries = sorted(quiet_holders_directory(quiet_path).glob("*.json"))
    except OSError:
        entries = []
    for entry in entries:
        try:
            holder = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(holder, dict) or holder.get("schema") != QUIET_SCHEMA:
            continue
        until, pid = holder.get("until"), holder.get("pid")
        if not isinstance(until, (int, float)) or until <= now:
            continue
        if type(pid) is int and pid > 0 and holder.get("host") == socket.gethostname() and not alive(pid):
            continue
        holders.append({"id": entry.stem, "label": holder.get("label"), "pid": pid, "since": holder.get("since"),
                        "until": until})
    return {"quiet": operator or bool(holders), "operator": operator, "holders": holders}


def quiet_hold(quiet_path: Path, label: str, *, pid=None, ttl=QUIET_TTL_SECONDS, now=None) -> dict:
    """Hold quiet mode for one measurement; returns its holder id and whether quiet mode was already held."""
    if not QUIET_LABEL_RE.fullmatch(label or ""):
        raise ValueError(f"a holder label is 1 to 40 letters, digits, '.', '_' or '-', not {label!r}")
    if not (type(ttl) is int and 1 <= ttl <= 7 * 24 * 3600):
        raise ValueError("a holder's time to live is 1 s to 7 days")
    now = time.time() if now is None else now
    before = quiet_state(quiet_path, now)
    directory = quiet_holders_directory(quiet_path)
    directory.mkdir(parents=True, exist_ok=True)
    holder_id = f"{label}-{pid or 0}-{os.urandom(4).hex()}"
    path = directory / f"{holder_id}.json"
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps({"schema": QUIET_SCHEMA, "label": label, "pid": pid, "host": socket.gethostname(),
                                     "since": now, "until": now + ttl}) + "\n", encoding="utf-8")
    temporary.replace(path)
    return {"holder": holder_id, "was_quiet": before["quiet"], "until": now + ttl}


def quiet_release(quiet_path: Path, holder_id: str) -> dict:
    """End one measurement's hold; quiet mode stays on while the operator or another holder holds it."""
    if not QUIET_LABEL_RE.fullmatch(holder_id.split("-", 1)[0]) or "/" in holder_id or "\\" in holder_id:
        raise ValueError(f"not a quiet holder id: {holder_id!r}")
    (quiet_holders_directory(quiet_path) / f"{holder_id}.json").unlink(missing_ok=True)
    return {"released": holder_id, "still_quiet": quiet_state(quiet_path, time.time())["quiet"]}


def quiet(args) -> int:
    path = Path(args.quiet_file)
    if args.state == "on":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("The SparkRing dashboard sends no request while this file exists.\n", encoding="utf-8")
        print(f"Dashboard paused: {path} exists; it sends no request until sparkring dashboard quiet off.")
    elif args.state == "off":
        path.unlink(missing_ok=True)
        holders = quiet_state(path, time.time())["holders"]
        print("Dashboard collecting again." if not holders else
              "The operator's pause is off; the dashboard stays quiet while these measurements hold it: "
              + ", ".join(f"{h['label']} (pid {h['pid']})" for h in holders))
    elif args.state == "status":
        print(json.dumps(quiet_state(path, time.time()), indent=2))
    elif args.state == "hold":
        print(json.dumps(quiet_hold(path, args.label, pid=args.pid, ttl=args.ttl)))
    else:
        print(json.dumps(quiet_release(path, args.holder)))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sparkring dashboard", description=(
        "A read-only web page of the fabric's Sparks and the models they serve. It reads host files and "
        "sends GET requests for /metrics, /v1/models and /v1/sparkring/status only: no inference request, "
        "no SSH."))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("serve", help="collect and serve the page")
    run.add_argument("--bind", help="listen address (default: Node A's LAN address, else 127.0.0.1); 127.0.0.1 "
                                    "is added unless the address is loopback or a wildcard")
    run.add_argument("--port", type=int, default=DEFAULT_PORT)
    run.add_argument("--fabric", action="append", type=Path, help="fabric document (repeat with --site per fabric)")
    run.add_argument("--site", action="append", type=Path, help="site file written by export")
    run.add_argument("--name", action="append", help="a host name browsers may use for this page")
    run.add_argument("--password-file", help="require HTTP Basic authentication (user sparkring) with this password")
    run.add_argument("--manage", action="store_true", help="opt in to authenticated actions through the privileged helper")
    run.add_argument("--manage-token-file", help="file containing the management bearer token (at least 32 characters)")
    run.add_argument("--cpus", help="CPU list to run on, such as 0-4,10-14 (default: the efficiency cores)")
    run.add_argument("--quiet-file", default=str(QUIET_PATH))
    run.add_argument("--no-node-status", action="store_true", help="do not read this host's sparkring-agent status")
    out = commands.add_parser("export", help="write the site file from the deployment records (Node A, sudo)")
    out.add_argument("--output", default=str(SITE_PATH))
    hush = commands.add_parser("quiet", help="pause or resume all requests, for example around a measurement: "
                                             "on/off for an operator; hold/release for a measurement; status")
    hush.add_argument("state", choices=("on", "off", "status", "hold", "release"))
    hush.add_argument("holder", nargs="?", help="release: the holder id that hold printed")
    hush.add_argument("--label", help="hold: the measurement's name, such as serving_ab")
    hush.add_argument("--pid", type=int, help="hold: the measurement's process; the hold lapses when it exits")
    hush.add_argument("--ttl", type=int, default=QUIET_TTL_SECONDS, help="hold: seconds after which the hold lapses")
    hush.add_argument("--quiet-file", default=str(QUIET_PATH))
    commands.add_parser("disable", help="stop the service and keep it off across reboots, setups and upgrades (sudo)")
    commands.add_parser("enable", help="turn the service back on (sudo)")
    from runtime.host import dashboard_tui     # a reader of this service's /api/status; collects nothing
    dashboard_tui.add_parser(commands)
    args = parser.parse_args(argv)
    try:
        if args.command == "tui":
            return dashboard_tui.main(args)
        if args.command == "serve":
            if bool(args.fabric) != bool(args.site) or len(args.fabric or ()) != len(args.site or ()):
                raise ValueError("give --fabric and --site together, once per fabric")
            return serve(args)
        if args.command == "quiet" and args.state == "hold" and not args.label:
            raise ValueError("quiet hold needs --label")
        if args.command == "quiet" and args.state == "release" and not args.holder:
            raise ValueError("quiet release needs the holder id that quiet hold printed")
        if args.command in ("export", "enable", "disable"):
            # Root actions; their module may start systemctl, which the collector never does.
            from runtime.host import dashboard_admin
            return (dashboard_admin.export(args.output) if args.command == "export"
                    else dashboard_admin.switch(args.command))
        return quiet(args)
    except (ValueError, OSError, RuntimeError) as error:
        print("SparkRing: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
