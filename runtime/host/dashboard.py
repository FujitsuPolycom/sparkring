"""``sparkring dashboard``: a read-only web page of one or more SparkRing fabrics and the models they serve.

Status: Experimental. Covered by offline tests only; no ring has run it.

The page shows each fabric's ring of Sparks with the arc every model serves
from, and per model its state, output tokens per second, requests, KV cache
use and budget, draft acceptance, first-token time, prefix-cache hits,
serving settings, collective transport, image and per-Spark memory and disk.

Where its data comes from
-------------------------

- Host files, read when they change (``stat`` every FILES_SECONDS):

  - the fabric document (``sparkring-fabric/v1``, ``runtime.common.fabric_document``)
    for the Sparks, their positions, hostnames and management addresses;
  - the site file (``sparkring-dashboard-site/v1``, SITE_PATH) that
    ``sudo sparkring dashboard export`` writes on Node A from the controller's
    deployment records: each deployment's profile, served model name, positions,
    API address and port and container names. The controller's records are
    readable by root only; the site file holds only these fields, so the
    dashboard needs no privilege;
  - this Spark's host-agent status (``sparkring-node-status/v1``,
    ``/run/sparkring/status.json``) when it exists;
  - on Node A, the recovery timer's latest finding per deployment
    (``sparkring-recovery/v1``, RECOVERY_PATH), which it writes about once a
    minute while a model is active.

- HTTP GET requests to each deployment's API, through ``ReadOnlyClient``:

  - ``/metrics`` every METRICS_VIEWED_SECONDS while a browser shows the page
    (it fetched ``/api/status`` within VIEWER_SECONDS), else every
    METRICS_UNVIEWED_SECONDS. The vllm:cache_config_info gauge in it gives the
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
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

SCHEMA = "sparkring-dashboard/v1"
SITE_SCHEMA = "sparkring-dashboard-site/v1"
SITE_PATH = Path("/var/lib/sparkring/dashboard/site.json")
FABRIC_PATH = Path("/etc/sparkring/fabric/topology.json")
NODE_STATUS_PATH = Path("/run/sparkring/status.json")
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
METRICS_UNVIEWED_SECONDS = 60.0
MODELS_SECONDS = 60.0
VIEWER_SECONDS = 60.0
FILES_SECONDS = 30.0
RUNTIME_REFRESH_SECONDS = 300.0
HTTP_TIMEOUT = 2.0
FAILED_SECONDS = 15.0         # a model API that does not answer is asked again after this long
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024
HISTORY_SECONDS = 3600.0
SAMPLE_SECONDS = 330.0          # counter samples kept for the 5-minute rate windows
MAX_EVENTS = 200
DEFAULT_PORT = 8090
LOG_LINES = 200

COUNTERS = {
    "vllm:generation_tokens_total": "gen",
    "vllm:prompt_tokens_total": "prompt",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted",
    "vllm:spec_decode_num_drafts_total": "drafts",
    "vllm:num_preemptions_total": "preemptions",
    "vllm:time_to_first_token_seconds_sum": "ttft_sum",
    "vllm:time_to_first_token_seconds_count": "ttft_count",
    "vllm:prefix_cache_queries_total": "pc_queries",
    "vllm:prefix_cache_hits_total": "pc_hits",
}
SETTING_KEYS = (
    "tensor_parallel_size", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
    "cache_dtype", "quantization", "speculative_method", "num_speculative_tokens",
    "reasoning_parser", "tool_call_parser", "default_chat_template_kwargs",
    "kv_cache_memory_bytes", "prefix_caching_enabled",
)
LABELS = {"serving": "Serving", "not_ready": "Not ready", "stopped": "No model running",
          "offline": "Offline", "idle": "Idle", None: "Checking"}
COLLECTIVE_LABELS = {"sircl": "SIRCL", "rocenante": "RoCEnante", "nccl": "NCCL"}
MODEL_TAIL_WORDS = {"nvfp4", "fp8", "fp4", "mxfp8", "mxfp4", "int4", "int8", "awq", "gptq", "bf16",
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


class ReadOnlyClient:
    """Sends the collector's GET requests and counts them; the module's only network egress.

    `opener` is a urllib opener; the default ignores proxy settings from the environment, so
    requests to Spark addresses never leave through a proxy."""

    def __init__(self, opener=None, timeout=HTTP_TIMEOUT):
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
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
    """The vLLM series the page uses, summed across engines; KV cache use takes the maximum.

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
        if out["model_name"] is None and name.startswith("vllm:") and 'model_name="' in labels:
            out["model_name"] = parse_labels(labels).get("model_name")
        if name in COUNTERS:
            key = COUNTERS[name]
            out[key] = out.get(key, 0.0) + value
        elif name == "vllm:num_requests_running":
            out["running"] = out.get("running", 0.0) + value
        elif name == "vllm:num_requests_waiting":
            out["waiting"] = out.get("waiting", 0.0) + value
        elif name in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
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
    images the label comes from the TP group's resident backend facts."""
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
    if collective is not None:
        label = COLLECTIVE_LABELS.get(str(collective), str(collective))
    elif isinstance(backend, str) and "sircl" in backend.lower():
        label = "SIRCL"
    elif rocenante is True:
        label = "RoCEnante"
    elif backend:
        label = str(backend)
    elif nccl_available and not nccl_disabled:
        label = "NCCL"
    else:
        label = None
    if label is None and bundle is None:
        return None
    nccl_on = None if nccl_available is None and nccl_disabled is None else bool(nccl_available) and not nccl_disabled
    return {"label": label, "collective": collective, "sircl_version": sircl_version if collective == "sircl" else None,
            "nccl": nccl_on, "bundle": bundle, "profile": bundle, "source": "runtime"}


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
        if kept and (low in MODEL_TAIL_WORDS or re.fullmatch(r"tp\d+", low) or re.fullmatch(r"[\d.]+bpw", low)):
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
    cable; the dashboard needs positions, hostnames and management addresses only, so it
    shows a document that a later setup schema extends."""
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
        rows.append({"position": index, "hostname": hostname, "node_id": row.get("node_id"),
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
        rows.append({"id": str(row.get("id") or run_name(positions)), "profile": row.get("profile"),
                     "model": row.get("model"), "positions": positions, "api_host": row["api_host"],
                     "api_port": port, "display_url": row.get("display_url"),
                     "stopped": bool(row.get("stopped")), "containers": containers})
    return rows


class Source:
    """One fabric's host files: its fabric document and its site file, re-read when they change."""

    def __init__(self, fabric_path: Path, site_path: Path):
        self.fabric_path, self.site_path = Path(fabric_path), Path(site_path)
        self.stamps = None
        self.fabric = None
        self.deployments: list = []
        self.problem = None

    def _stamp(self, path):
        try:
            stat = path.stat()
            return (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return None

    def refresh(self) -> bool:
        """Re-read the files when their size or time changed; True when anything was re-read."""
        stamps = (self._stamp(self.fabric_path), self._stamp(self.site_path))
        if stamps == self.stamps:
            return False
        self.stamps = stamps
        try:
            self.fabric = fabric_from_document(read_json(self.fabric_path))
        except (OSError, ValueError) as error:
            self.fabric, self.deployments = None, []
            self.problem = f"fabric document {self.fabric_path}: {error}"
            return True
        try:
            self.deployments = deployments_from_site(read_json(self.site_path), self.fabric["size"])
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
    """One deployment's API endpoint: its observations, rates, history and state."""

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
        self.samples: collections.deque = collections.deque()
        self.history: collections.deque = collections.deque()

    def window(self, key, seconds, now):
        if len(self.samples) < 2:
            return None
        newest_t, newest = self.samples[-1]
        base = next(((t, m) for t, m in self.samples if t >= now - seconds - 0.5), None)
        if base is None or base[0] >= newest_t or key not in newest or key not in base[1]:
            return None
        delta = newest[key] - base[1][key]
        return (delta, newest_t - base[0]) if delta >= 0 else None

    def rate(self, key, seconds, now):
        found = self.window(key, seconds, now)
        return round(found[0] / found[1], 2) if found and found[1] >= 1 else None

    def ratio(self, top, bottom, seconds, now):
        a, b = self.window(top, seconds, now), self.window(bottom, seconds, now)
        return a[0] / b[0] if a and b and b[0] > 0 else None

    def observe_metrics(self, metrics: dict, now: float) -> None:
        if self.samples and metrics.get("gen", 0) < self.samples[-1][1].get("gen", 0):
            self.samples.clear()  # counters went back: the server restarted
        self.metrics = metrics
        # Rate windows need the counters only; the gauge labels stay out of the kept samples.
        self.samples.append((now, {key: value for key, value in metrics.items() if isinstance(value, float)}))
        while self.samples and now - self.samples[0][0] > SAMPLE_SECONDS:
            self.samples.popleft()
        self.history.append([round(now, 1), self.rate("gen", 15, now), metrics.get("running"),
                             metrics.get("waiting"), metrics.get("kv")])
        while self.history and now - self.history[0][0] > HISTORY_SECONDS:
            self.history.popleft()
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
                 clock=time.time, cpu_clock=time.thread_time):
        self.sources = list(sources)
        self.client = client
        self.quiet_path = Path(quiet_path) if quiet_path else None
        self.node_status_path = Path(node_status_path) if node_status_path else None
        self.recovery_path = Path(recovery_path) if recovery_path else None
        self.recovery: dict = {}      # deployment directory name -> the recovery timer's latest finding
        self.clock, self.cpu_clock = clock, cpu_clock
        self.lock = threading.Lock()
        self.models: dict = {}
        self.events: collections.deque = collections.deque(maxlen=MAX_EVENTS)
        self.started = clock()
        self.last_view = None
        self.next_files = 0.0
        self.quiet = False
        self.node_status = None
        self.cycles = 0
        self.cycle_cpu = collections.deque(maxlen=120)   # CPU seconds of recent cycles that sent requests

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

    def reload_files(self, now) -> None:
        changed = False
        for source in self.sources:
            changed = source.refresh() or changed
        if self.node_status_path is not None:
            try:
                self.node_status = read_json(self.node_status_path)
            except (OSError, ValueError):
                self.node_status = None
        if self.recovery_path is not None:
            self.recovery = recovery_findings(self.recovery_path)
        if not changed:
            return
        hosts, wanted = set(), {}
        for source in self.sources:
            for row in source.deployments:
                hosts.add(row["api_host"])
                wanted[(source.ring_id(), row["id"])] = (source, row)
        self.client.allow(hosts)
        with self.lock:
            for model_id in [m for m, model in self.models.items() if model.key not in wanted]:
                del self.models[model_id]
            known = {model.key: model for model in self.models.values()}
            for key, (source, row) in wanted.items():
                if key in known:
                    known[key].row = row
                else:
                    model = Model(source, row)
                    self.models[model.id] = model

    # -- one cycle -------------------------------------------------------------------------
    def cycle(self, now=None) -> None:
        now = self.clock() if now is None else now
        # Quiet mode is checked on every cycle (a stat and a small directory listing a second), so a
        # hold takes effect before a measurement that starts right after it.
        self.quiet = bool(self.quiet_path and quiet_state(self.quiet_path, now)["quiet"])
        if self.quiet:
            return
        if now >= self.next_files:
            self.reload_files(now)
            self.next_files = now + FILES_SECONDS
        before = self.cpu_clock()
        sent = False
        with self.lock:
            models = list(self.models.values())
            viewed = self.viewed(now)
        for model in models:
            sent = self.poll(model, now, viewed) or sent
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
        sent = False
        if now >= model.next_metrics:
            sent = True
            cadence = METRICS_VIEWED_SECONDS if viewed and not row["stopped"] else METRICS_UNVIEWED_SECONDS
            fetched = self.client.get(host, port, "/metrics")
            if fetched.ok:
                metrics = parse_metrics(fetched.body.decode("utf-8", "replace"))
                if row.get("model") and metrics.get("model_name") and metrics["model_name"] != row["model"]:
                    state = "idle"
                    with self.lock:
                        model.metrics = None
                        model.samples.clear()
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
                with self.lock:
                    model.metrics = None
                    model.samples.clear()
                model.next_metrics = now + max(cadence, FAILED_SECONDS)
            self.set_state(model, state, now)
        if model.state == "serving" and now >= model.next_models:
            sent = True
            model.next_models = now + MODELS_SECONDS
            listed = self.models_listed(self.client.get(host, port, "/v1/models"))
            if listed:
                with self.lock:
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
    def snapshot(self, now=None) -> dict:
        """The sparkring-dashboard/v1 document: the fields the page reads, in the shape it reads them."""
        now = self.clock() if now is None else now
        with self.lock:
            models = list(self.models.values())
            groups, rings = [], []
            color = 0
            for source in self.sources:
                fabric = source.fabric
                members = []
                for model in models:
                    if model.source is source:
                        members.append(self.member(model, now, color))
                        color += 1
                members.sort(key=lambda m: m["arc"]["first"])
                groups.append({"id": source.ring_id(), "name": self.ring_name(source), "kind": "ring",
                               "members": members})
                if fabric:
                    rings.append(self.ring(source, members, now))
            events = list(self.events)[::-1][:60]
            quiet, viewed = self.quiet, self.viewed(now)
            cpu = list(self.cycle_cpu)
        problems = [s.problem for s in self.sources if s.problem]
        topology = {"source": "fabric" if rings else "none", "detail": "; ".join(problems) or None,
                    "rings": len(rings)}
        counters = self.client.counters()
        return {"schema": SCHEMA, "title": "SparkRing", "generated_at": now, "service_started": self.started,
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
        return f"{fabric['positions'][0]['hostname']} {'ring' if fabric['closed'] else fabric['shape']}"

    def member(self, model: Model, now: float, color: int) -> dict:
        row, fabric = model.row, model.source.fabric or {"positions": []}
        hostnames = {p["position"]: p for p in fabric["positions"]}
        positions = row["positions"]
        runtime = model.runtime or {}
        metrics = model.metrics or {}
        serving = model.state == "serving"
        accept = model.ratio("accepted", "drafts", 300, now)
        nodes = runtime.get("nodes") or []
        head = hostnames.get(positions[0], {})
        port = row["api_port"]
        endpoint = row.get("display_url") or f"http://{row['api_host']}:{port}/v1"
        since = runtime.get("started_at") if serving and runtime.get("started_at") else model.state_since
        capacity = dict(model.capacity) if model.capacity else None
        if capacity and runtime.get("settings"):
            capacity["max_num_seqs"] = runtime["settings"].get("max_num_seqs")
        container = next((c["name"] for c in row["containers"] if c["position"] == positions[0]), None)
        sparks = []
        for position in positions:
            spark = hostnames.get(position, {})
            up = True if serving else (False if model.state == "offline" and position == positions[0] else None)
            sparks.append({"name": spark.get("hostname", str(position)), "address": spark.get("address"),
                           "up": up, "position": position})
        return {
            "id": model.id, "name": run_name(positions), "profile": row.get("profile"),
            "state": model.state, "state_label": LABELS.get(model.state, "Checking"), "state_since": since,
            "model": model.model, "model_short": short_model(model.model), "max_model_len": model.max_model_len,
            "endpoint": endpoint,
            # The runtime-status view asks the workers for a report every 5 s while it is open.
            "details_url": f"http://{row['api_host']}:{port}/v1/sparkring/status/view" if model.runtime else None,
            "layout": {"label": layout_label(len(positions)) if serving else None, "count": len(positions),
                       "tp": runtime.get("tp")},
            "live": {"running": metrics.get("running"), "waiting": metrics.get("waiting"), "kv": metrics.get("kv"),
                     "output_tps": model.rate("gen", 15, now), "prompt_tps": model.rate("prompt", 15, now),
                     "output_tps_5m": model.rate("gen", 300, now),
                     "accept_length_5m": round(1 + accept, 2) if accept is not None else None,
                     "ttft_5m": model.ratio("ttft_sum", "ttft_count", 300, now),
                     "prefix_hit_5m": model.ratio("pc_hits", "pc_queries", 300, now)} if model.metrics else None,
            "totals": {"finished": metrics.get("finished"), "preemptions": metrics.get("preemptions"),
                       "output_tokens": metrics.get("gen"), "prompt_tokens": metrics.get("prompt")}
            if model.metrics else None,
            "settings": runtime.get("settings"), "nodes": nodes, "capacity": capacity if serving else None,
            "workers": runtime.get("workers"), "image": runtime.get("image"), "transport": runtime.get("transport"),
            "sparks": sparks,
            "history": [h for h in model.history if h[0] >= now - HISTORY_SECONDS],
            "arc": {"positions": positions, "first": positions[0], "last": positions[-1], "size": len(positions),
                    "run": run_name(positions), "source": "deployment", "source_label": "deployment record",
                    "inferred": False, "detail": None, "color": color},
            "ring": model.source.ring_id(),
            "head": {"position": positions[0], "name": head.get("hostname"), "address": row["api_host"], "port": port,
                     "container": container},
            "logs": log_commands(head.get("hostname"), container),
            "runtime_read": model.runtime is not None,
            "recovery": self.recovery.get(row["id"]),
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
                "discovery": {"ports": sorted({m["head"]["port"] for m in members}),
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
            "enabled": record.get("enabled", True), "stopped": bool(record.get("stopped"))}
    return found


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
    policies = []
    for tag, directive in (("script", "script-src"), ("style", "style-src")):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.S)
        digest = base64.b64encode(hashlib.sha256(match.group(1).encode()).digest()).decode()
        policies.append(f"{directive} 'sha256-{digest}'")
    csp = "; ".join(["default-src 'none'", *policies, "style-src-attr 'unsafe-inline'", "connect-src 'self'",
                     "img-src 'self' data:", "base-uri 'none'", "frame-ancestors 'none'", "form-action 'none'"])
    return text.encode("utf-8"), csp


def host_allowed(header: str | None, names) -> bool:
    """Whether a request's Host header names this server: an IP literal or one of `names`.

    A browser on the LAN that a hostile web page points at the dashboard through DNS
    rebinding sends that page's host name, which is refused."""
    if not header:
        return False
    host = header.strip()
    if host.startswith("["):
        host = host[1:host.find("]")] if "]" in host else host
    else:
        host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return _ip(host) or host.lower() in names


def make_handler(collector: Collector, page: bytes, csp: str, names=frozenset(), password: str | None = None):
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
               "X-Frame-Options": "DENY"}
    expected = None if password is None else "Basic " + base64.b64encode(f"sparkring:{password}".encode()).decode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "sparkring-dashboard"

        def do_GET(self):
            if not host_allowed(self.headers.get("Host"), names):
                return self._send(421, b"This dashboard answers to its IP address or a name given with --name.\n",
                                  "text/plain; charset=utf-8")
            if expected is not None and self.headers.get("Authorization") != expected:
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
                collector.mark_viewed()
                return self._json(200, collector.snapshot())
            if path == "/api/refresh":
                accepted = collector.request_runtime(params.get("model", ""))
                return self._json(202 if accepted else 429, {"accepted": accepted})
            if path == "/healthz":
                return self._send(200, b"ok\n", "text/plain")
            return self._send(404, b"not found\n", "text/plain")

        def _refuse(self):
            self._send(405, b"read-only\n", "text/plain", {"Allow": "GET"})

        do_POST = do_PUT = do_DELETE = do_PATCH = _refuse

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
    management address); on any other host, loopback."""
    rows = (source.fabric or {}).get("positions") or []
    if rows and socket.gethostname() == rows[0]["hostname"]:
        return rows[0].get("lan_address") or rows[0].get("address") or "127.0.0.1"
    return "127.0.0.1"


def serve(args) -> int:
    pairs = list(zip(args.fabric, args.site)) if args.fabric else [(FABRIC_PATH, SITE_PATH)]
    sources = [Source(Path(f), Path(s)) for f, s in pairs]
    client = ReadOnlyClient()
    collector = Collector(sources, client, quiet_path=Path(args.quiet_file),
                          node_status_path=None if args.no_node_status else NODE_STATUS_PATH,
                          recovery_path=RECOVERY_PATH)
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
    server = ThreadingHTTPServer((bind, args.port), make_handler(collector, page, csp, names, password))
    server.daemon_threads = True
    stop = threading.Event()
    threading.Thread(target=collector.run, args=(stop,), daemon=True, name="collector").start()
    print(f"SparkRing dashboard on http://{bind}:{args.port}/ ({', '.join(placement) or 'default priority'}); "
          f"{len(collector.models)} model API(s) from {len(sources)} fabric(s)", flush=True)
    try:
        server.serve_forever()
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
    run.add_argument("--bind", help="listen address (default: Node A's LAN address, else 127.0.0.1)")
    run.add_argument("--port", type=int, default=DEFAULT_PORT)
    run.add_argument("--fabric", action="append", type=Path, help="fabric document (repeat with --site per fabric)")
    run.add_argument("--site", action="append", type=Path, help="site file written by export")
    run.add_argument("--name", action="append", help="a host name browsers may use for this page")
    run.add_argument("--password-file", help="require HTTP Basic authentication (user sparkring) with this password")
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
    args = parser.parse_args(argv)
    try:
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
