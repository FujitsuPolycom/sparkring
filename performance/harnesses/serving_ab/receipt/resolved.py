"""Build a run receipt's ``resolved.json``: what a deployment actually serves, per rank, from collected evidence.

The input of :func:`build` is what :mod:`.collect` reads from the ring (or what a campaign saved): per rank
the container's selected log lines, ``docker inspect``, the vLLM process's environment and command line,
its SIRCL receipt files and GPU clocks; per deployment Node A's ``sparkring status --json`` entry and the
API's ``/metrics`` text. The output separates three kinds of values:

``settings``
    Serving-relevant settings as the server resolved them, flattened to dotted keys: ``target.*`` (KV cache
    dtype and block size), ``speculative.*`` (the speculative configuration, including the draft's
    ``kv_cache_dtype``), ``compilation.*`` (the engine's resolved ``compilation_config``), ``scheduler.*``,
    ``engram.*``, ``model.*``, ``backends.*``, ``kernel.*``, ``parallel.*``, ``args.*`` (every other explicit
    vLLM argument), ``env.*`` (the vLLM process's ``VLLM_*``, ``B12X_*``, ``SIRCL_*``, ``OMP_*``, ``NCCL_*``
    and related variables), ``sircl.*`` (the SIRCL session receipts) and ``image.id`` and ``deployment.*``.
    A comparison fails on any undeclared difference here.
``derived``
    Values that follow from settings by vLLM's documented rules: the effective target and draft KV dtypes
    (``auto`` is the model dtype; a draft without ``kv_cache_dtype`` inherits the target's).
``observed``
    Outcomes and conditions: KV capacity in tokens, the resolved attention block size, the kernels chosen,
    custom fusions, variables vLLM set itself, GPU clocks. Reported, never failed on.

Per-rank identity is excluded from ``settings`` so that two deployments on different Sparks compare:
addresses, ports, rank numbers, RDMA device names and cache or library paths (:data:`IDENTITY_ENV`,
:data:`IDENTITY_ARGS`). ``digest`` is the SHA-256 of the canonical JSON of rank 0's ``settings``.

Every value's origin is kept in ``sources``. Where vLLM logs no resolved value, the value is the explicit
argument, or vLLM's default stated as ``None`` (the server chooses), and ``scheduler.async_scheduling`` is
inferred from vLLM's rule: unset, it is on for EAGLE/MTP/DFlash/DSpark/draft-model speculation or no
speculation, with the ``mp`` or ``uni`` executor (``vllm/config/vllm.py`` of the image's vLLM).
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from collections.abc import Mapping

from . import pyrepr, vllm_log

SCHEMA = "serving-ab-resolved/v1"
ENV_PREFIXES = ("VLLM_", "B12X_", "SIRCL_", "OMP_", "NCCL_", "SPARKRING_", "SPARK_", "QWEN_", "GLM", "DEEPSEEK_",
                "DSPARK_", "TORCH_", "PYTORCH_", "CUDA_", "CUTE_", "TRITON_", "FLASHINFER_", "MKL_", "UCX_")
"""Environment variable prefixes that can change what or how a rank serves (kept in ``env.*``)."""
IDENTITY_ENV = re.compile(r"(_HOST_IP|_ADDR|_ADDRESS|_PORT|RANK|_HCA|HCA_MAP|_NODE|NODE_|_DIR|_ROOT|_PATH|"
                          r"_HOME|_VISIBLE_DEVICES|DEPLOYMENT_ID|_PID|_IFNAME|_BASE)$|^SIRCL_RANK|^VLLM_HOST_IP$")
"""Variables that name the rank, its host, devices or cache paths rather than a serving choice."""
IDENTITY_ARGS = frozenset({"model_tag", "model", "host", "port", "master_addr", "master_port", "node_rank",
                           "data_parallel_address", "data_parallel_rpc_port", "data_parallel_rank", "headless",
                           "api_key", "hf_token", "uvicorn_log_level", "disable_log_requests",
                           "enable_log_requests", "download_dir"})
EXCLUDED_COMPILATION = frozenset({"debug_dump_path", "cache_dir", "local_cache_dir"})
ARG_DEFAULTS = {"kv_cache_dtype": "auto"}
ASYNC_SPECULATION = frozenset({"eagle", "eagle3", "extract_hidden_states", "mtp", "dflash", "dspark", "ngram_gpu",
                               "draft_model"})
RECEIPT_KEYS = ("nccl", "nccl_mode", "pynccl", "session", "lanes", "capacity", "dispatch", "oneshot_max", "gather",
                "gather_piece", "large_piece", "schedules", "chain_min", "ring_min", "links", "column_gather",
                "fused_norm", "mhc", "op_per_peer", "four_rank_sessions", "relay_factor", "layout", "tuning", "p2p")
"""SIRCL receipt fields that are session settings (counters, devices and ranks are left out)."""
KEY_SETTINGS = ("target.kv_cache_dtype", "target.block_size", "speculative.method",
                "speculative.num_speculative_tokens", "speculative.kv_cache_dtype", "scheduler.async_scheduling",
                "compilation.mode", "compilation.cudagraph_mode", "scheduler.max_num_seqs",
                "scheduler.max_num_batched_tokens", "scheduler.prefill_schedule_interval", "model.quantization")
"""The settings a report shows beside each arm."""


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(settings: Mapping) -> str:
    return "sha256:" + hashlib.sha256(canonical(settings).encode()).hexdigest()


def _leaf(value) -> bool:
    """Scalars, lists, empty dicts, enum members and opaque ``<...>`` reprs are compared whole."""
    if not isinstance(value, dict) or not value:
        return True
    return ("enum" in value and set(value) <= {"enum", "value"}) or set(value) == {"repr"}


def flatten(prefix: str, value, out: dict, *, skip=frozenset()) -> None:
    """``out[prefix.key.sub] = leaf`` for every leaf of nested dicts; enums, lists and scalars are leaves."""
    if _leaf(value):
        out[prefix] = value
        return
    for key, item in value.items():
        if key in skip or key == "__type__":
            continue
        flatten(f"{prefix}.{key}", item, out)


def serving_env(env: Mapping[str, str]) -> dict:
    """The variables of ``env`` that can change serving, without the rank's identity and paths."""
    return {k: v for k, v in sorted(env.items())
            if k.startswith(ENV_PREFIXES) and not IDENTITY_ENV.search(k)}


def parse_env(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "=" in line and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", line):
            key, value = line.split("=", 1)
            out[key] = value
    return out


def args_from_command(tokens: list[str]) -> dict:
    """vLLM's explicit arguments from a command line: ``--flag value``, ``--flag=value`` and bare flags.

    Names become argument names (``--kv-cache-dtype`` -> ``kv_cache_dtype``), JSON values are parsed, and
    ``--no-flag`` gives ``flag: False``. Used where no ``non-default args`` line was captured.
    """
    out: dict = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        name, value = (token[2:].split("=", 1) + [None])[:2]
        if value is None:
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                value, i = tokens[i + 1], i + 1
            else:
                value = True
        key = name.replace("-", "_")
        if value is True and key.startswith("no_"):
            key, value = key[3:], False
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        out[key] = value
        i += 1
    return out


def parse_metrics_info(text: str) -> dict:
    """The labels of every ``*_info`` gauge of a Prometheus text (``vllm:cache_config_info`` holds the
    resolved cache configuration: ``cache_dtype``, ``block_size``, ``mamba_block_size``, ``num_gpu_blocks``)."""
    out = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*_info)\{(.*)\}\s+\S+\s*$", line)
        if match:
            out[match[1]] = {k: bytes(v, "utf-8").decode("unicode_escape")
                             for k, v in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"', match[2])}
    return out


def _number(text):
    if isinstance(text, str):
        for convert in (int, float):
            try:
                return convert(text)
            except ValueError:
                pass
        return {"True": True, "False": False, "None": None}.get(text, text)
    return text


def _dtype_name(dtype) -> str | None:
    return dtype.split(".")[-1] if isinstance(dtype, str) else None


def async_rule(explicit, method, executor, disabled: bool) -> tuple[bool | None, str]:
    """The resolved async-scheduling value and how it is known (see the module docstring)."""
    if disabled:
        return False, "log: vLLM disabled async scheduling"
    if explicit is not None:
        return bool(explicit), "argument"
    if method and method not in ASYNC_SPECULATION and not str(method).endswith("_mtp"):
        return False, f"inferred: vLLM leaves async scheduling off for {method} speculation"
    if executor in ("mp", "uni"):
        return True, f"inferred: vLLM turns unset async scheduling on with the {executor} executor"
    return None, "unknown: unset, and the executor is not recorded"


def rank_settings(scan: Mapping, *, env: Mapping[str, str] | None = None, command_args: Mapping | None = None,
                  receipts: Mapping[str, Mapping] | None = None, metrics: Mapping | None = None,
                  image_id: str | None = None, deployment: Mapping | None = None) -> dict:
    """``settings``, ``derived``, ``observed`` and ``sources`` of one rank from its parsed evidence."""
    settings: dict = {}
    derived: dict = {}
    observed: dict = {}
    sources: dict = {}
    args = dict(scan.get("non_default_args") or {})
    args_complete = "non_default_args" in scan and "non_default_args" not in scan.get("truncated", [])
    args_source = "non-default args line"
    if not args and command_args:
        args, args_complete, args_source = dict(command_args), True, "container command line"
    engine = scan.get("engine_config") or {}
    cache = (metrics or {}).get("vllm:cache_config_info") or {}

    def put(key, value, source, into=settings):
        into[key] = value
        sources[key] = source

    def explicit(name):
        """(value, source) of an explicit argument, vLLM's default when the argument list is complete."""
        if name in args:
            return args[name], args_source
        if args_complete:
            return ARG_DEFAULTS.get(name), "vLLM default (not set)"
        return None, "not recorded"

    # Target model and KV cache.
    if "kv_cache_dtype" in engine:
        put("target.kv_cache_dtype", engine["kv_cache_dtype"], "engine config line")
    elif cache.get("cache_dtype"):
        put("target.kv_cache_dtype", cache["cache_dtype"], "/metrics vllm:cache_config_info")
    else:
        put("target.kv_cache_dtype", *explicit("kv_cache_dtype"))
    put("target.block_size", *explicit("block_size"))
    for key, name in (("gpu_memory_utilization", "gpu_memory_utilization"),
                      ("kv_cache_memory_bytes", "kv_cache_memory_bytes")):
        put(f"target.{key}", *explicit(name))
    if "enable_prefix_caching" in engine:
        put("target.enable_prefix_caching", engine["enable_prefix_caching"], "engine config line")
    model_dtype = _dtype_name(engine.get("dtype")) or (args.get("dtype") if args.get("dtype") != "auto" else None)
    for key, name in (("dtype", "dtype"), ("quantization", "quantization"), ("load_format", "load_format"),
                      ("max_model_len", "max_seq_len")):
        if name in engine:
            put(f"model.{key}", engine[name], "engine config line")
        else:
            put(f"model.{key}", *explicit(key))
    if "served_model_name" in args:
        put("model.served_model_name", args["served_model_name"], args_source)

    # Speculative decoding: the explicit configuration, then the engine's method and depth.
    spec_args = args.get("speculative_config")
    spec_engine = engine.get("speculative_config")
    if isinstance(spec_args, dict) or isinstance(spec_engine, dict):
        if isinstance(spec_args, dict):
            for key, value in spec_args.items():
                put(f"speculative.{key}", value, args_source)
            if "kv_cache_dtype" not in spec_args:
                put("speculative.kv_cache_dtype", None, f"{args_source}: not set (the draft inherits the target's)")
        else:
            put("speculative.kv_cache_dtype", None,
                "vLLM default (not set)" if args_complete else "not recorded")
        if isinstance(spec_engine, dict):
            if "method" in spec_engine:
                put("speculative.method", spec_engine["method"], "engine config line")
            if "num_spec_tokens" in spec_engine:
                put("speculative.num_speculative_tokens", spec_engine["num_spec_tokens"], "engine config line")
    elif "speculative_config" in engine or args_complete:
        put("speculative.method", None, "engine config line" if engine else "vLLM default (not set)")

    # Compilation: the engine's resolved configuration, else the explicit one.
    compilation = engine.get("compilation_config")
    if isinstance(compilation, dict):
        flat: dict = {}
        flatten("compilation", compilation, flat, skip=EXCLUDED_COMPILATION)
        for key, value in flat.items():
            put(key, value, "engine config line")
    elif isinstance(args.get("compilation_config"), dict):
        flat = {}
        flatten("compilation", args["compilation_config"], flat, skip=EXCLUDED_COMPILATION)
        for key, value in flat.items():
            put(key, value, f"{args_source} (explicit fields only; not resolved)")

    # Scheduler.
    method = settings.get("speculative.method")
    executor = args.get("distributed_executor_backend")
    value, source = async_rule(args.get("async_scheduling"), method, executor, bool(scan.get("async_scheduling_disabled")))
    if args.get("async_scheduling") is None and not args_complete and not scan.get("async_scheduling_disabled"):
        value, source = None, "not recorded"
    put("scheduler.async_scheduling", value, source)
    put("scheduler.max_num_seqs", *explicit("max_num_seqs"))
    if "max_num_batched_tokens" in args:
        put("scheduler.max_num_batched_tokens", args["max_num_batched_tokens"], args_source)
    elif "chunked_prefill_tokens" in scan:
        put("scheduler.max_num_batched_tokens", scan["chunked_prefill_tokens"], "log: chunked prefill budget")
    else:
        put("scheduler.max_num_batched_tokens", *explicit("max_num_batched_tokens"))
    put("scheduler.prefill_schedule_interval", *explicit("prefill_schedule_interval"))
    if "enable_chunked_prefill" in engine:
        put("scheduler.enable_chunked_prefill", engine["enable_chunked_prefill"], "engine config line")

    # Engram (DeepSeek-V4.1): the resolved configuration, else the explicit one.
    if isinstance(scan.get("engram"), dict):
        flat = {}
        flatten("engram", scan["engram"], flat)
        for key, value in flat.items():
            put(key, value, "log: resolved Engram configuration")
    elif isinstance(args.get("engram_config"), dict):
        flat = {}
        flatten("engram", args["engram_config"], flat)
        for key, value in flat.items():
            put(key, value, args_source)

    # Backends and kernel configuration.
    kernel = engine.get("kernel_config")
    if isinstance(kernel, dict):
        flat = {}
        flatten("kernel", kernel, flat)
        for key, value in flat.items():
            put(key, value, "engine config line")
    for name in ("moe_backend", "linear_backend", "attention_backend", "gdn_decode_kernel"):
        if name in args:
            put(f"backends.{name}", args[name], args_source)
    for key, name in (("tensor_parallel_size", "tensor_parallel_size"),
                      ("pipeline_parallel_size", "pipeline_parallel_size"),
                      ("data_parallel_size", "data_parallel_size"),
                      ("decode_context_parallel_size", "decode_context_parallel_size"),
                      ("dcp_comm_backend", "dcp_comm_backend"),
                      ("disable_custom_all_reduce", "disable_custom_all_reduce"),
                      ("enforce_eager", "enforce_eager")):
        if name in engine:
            put(f"parallel.{key}" if key != "enforce_eager" else "model.enforce_eager", engine[name],
                "engine config line")

    # Every other explicit argument.
    covered = {"kv_cache_dtype", "block_size", "gpu_memory_utilization", "kv_cache_memory_bytes", "dtype",
               "quantization", "load_format", "max_model_len", "served_model_name", "speculative_config",
               "compilation_config", "async_scheduling", "max_num_seqs", "max_num_batched_tokens",
               "prefill_schedule_interval", "engram_config", "moe_backend", "linear_backend", "attention_backend",
               "gdn_decode_kernel", "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
               "decode_context_parallel_size", "enable_prefix_caching", "enable_chunked_prefill", "enforce_eager"}
    for name, value in sorted(args.items()):
        if name not in covered and name not in IDENTITY_ARGS:
            put(f"args.{name}", value, args_source)

    # Environment of the vLLM process.
    for name, value in serving_env(env or {}).items():
        put(f"env.{name}", value, "vLLM process environment")

    # SIRCL session settings: the log's receipt lines (every capture has them, so two records compare in
    # one representation), else the receipt files.
    groups = {}
    for group, fields in (scan.get("sircl_receipt_lines") or {}).items():
        groups[group.split(":")[0]] = ({k: None if v == "-" else _number(v) for k, v in fields.items()},
                                       "log: SIRCL receipt line")
    for name, receipt in sorted((receipts or {}).items()):
        if isinstance(receipt, Mapping) and receipt.get("group"):
            groups.setdefault(str(receipt["group"]).split(":")[0], (dict(receipt), f"SIRCL receipt file {name}"))
    for group, (receipt, source) in sorted(groups.items()):
        for key in RECEIPT_KEYS:
            if key in receipt:
                value = receipt[key]
                if key == "p2p" and isinstance(value, str):
                    # The peer count is the rank's place in the group (an end of a path has one neighbour,
                    # a middle rank two); the slots and windows are the session's settings.
                    value = re.sub(r"(^|,)peers=\d+", "", value).strip(",")
                put(f"sircl.{group}.{key}", value, source)

    if image_id:
        put("image.id", image_id, "docker inspect")
    for key in ("profile", "checkpoint", "model_repository", "model_revision", "image_release"):
        if deployment and deployment.get(key) is not None:
            put(f"deployment.{key}", deployment[key], "sparkring status --json")
    transport = (deployment or {}).get("transport") or {}
    for key in ("backend", "nccl"):
        if transport.get(key) is not None:
            put(f"deployment.transport.{key}", transport[key], "sparkring status --json")

    # Derived: effective KV dtypes by vLLM's rules.
    target = settings.get("target.kv_cache_dtype")
    if target is not None:
        effective = model_dtype if target == "auto" else target
        put("target.kv_cache_dtype_effective", effective,
            "derived: auto is the model dtype" if target == "auto" else "derived: as set", into=derived)
        if settings.get("speculative.method"):
            draft = settings.get("speculative.kv_cache_dtype")
            if draft is None:
                put("speculative.kv_cache_dtype_effective", effective, "derived: unset, the draft inherits the target's",
                    into=derived)
            else:
                put("speculative.kv_cache_dtype_effective", model_dtype if draft == "auto" else draft,
                    "derived: auto is the model dtype" if draft == "auto" else "derived: as set", into=derived)

    # Observed outcomes.
    if cache.get("block_size"):
        put("kv.block_size", _number(cache["block_size"]), "/metrics vllm:cache_config_info", into=observed)
    elif "attention_block_size" in scan:
        put("kv.block_size", scan["attention_block_size"], "log: attention block size", into=observed)
    elif "kv_block_size" in scan:
        put("kv.block_size", scan["kv_block_size"]["tokens"], "log: KV cache block size", into=observed)
    for key, label in (("mamba_block_size", "mamba_block_size"), ("num_gpu_blocks", "num_gpu_blocks"),
                       ("kv_cache_size_tokens", "kv_cache_size_tokens")):
        if cache.get(label) not in (None, "None"):
            put(f"kv.{key}", _number(cache[label]), "/metrics vllm:cache_config_info", into=observed)
    if "kv_cache_tokens" in scan:
        put("kv.cache_tokens", scan["kv_cache_tokens"], "log: GPU KV cache size", into=observed)
    for name in ("moe_kernel", "gemm_kernel", "auto_env", "allreduce_backends", "startup_audit"):
        for subject, value in sorted((scan.get(name) or {}).items()):
            put(f"{name}.{subject}", value, f"log: {name.replace('_', ' ')}", into=observed)
    for name in ("custom_fusions", "lm_heads", "model_runner_v2", "breakable_cudagraph", "kv_cache_layout",
                 "architecture"):
        if name in scan:
            put(name, scan[name], f"log: {name.replace('_', ' ')}", into=observed)
    if "engine_config" in scan:
        put("vllm.version", scan["engine_config"].get("version"), "engine config line", into=observed)
    return {"settings": settings, "derived": derived, "observed": observed, "sources": sources}


def parse_clocks(text: str) -> list[dict]:
    out = []
    names = ("sm_mhz", "max_sm_mhz", "event_reasons", "applications_mhz", "temperature_c", "power_w")
    for line in text.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) >= 3 and fields[0][:1].isdigit():
            out.append({name: _number(re.sub(r"\s*(MHz|W|C)$", "", value)) for name, value in zip(names, fields)})
    return out


def build_rank(raw: Mapping, *, metrics: Mapping | None = None, deployment: Mapping | None = None) -> dict:
    """One rank's record from :mod:`.collect`'s raw sections (see :func:`build`)."""
    inspect = raw.get("inspect") or {}
    config = inspect.get("Config") or {}
    command = list(raw.get("process_cmdline") or []) or [inspect.get("Path", "")] + list(inspect.get("Args") or [])
    env = raw.get("process_env")
    env_source = "vLLM process environment"
    if env is None:
        env = raw.get("container_env") or parse_env("\n".join(config.get("Env") or []))
        env_source = "container environment"
    scan = vllm_log.scan(raw.get("log") or "")
    record = rank_settings(scan, env=env, command_args=args_from_command(command), receipts=raw.get("receipts"),
                           metrics=metrics, image_id=inspect.get("Image") or raw.get("image_id"),
                           deployment=deployment)
    if env_source != "vLLM process environment":
        record["sources"] = {k: (env_source if v == "vLLM process environment" else v)
                             for k, v in record["sources"].items()}
    clocks = parse_clocks(raw.get("clocks") or "")
    if clocks:
        record["observed"]["clocks"] = clocks
    container_env = serving_env(raw.get("container_env") or {})
    process_env = serving_env(raw.get("process_env") or {})
    added = {k: v for k, v in process_env.items() if container_env.get(k) != v} if raw.get("process_env") else {}
    receipt_files = {name: {k: receipt.get(k) for k in ("group", *RECEIPT_KEYS) if k in receipt}
                     for name, receipt in sorted((raw.get("receipts") or {}).items()) if isinstance(receipt, Mapping)}
    return {
        "rank": raw.get("rank"), "position": raw.get("position"), "host": raw.get("host"),
        "container": raw.get("container"), "image_id": inspect.get("Image") or raw.get("image_id"),
        "started_at": (inspect.get("State") or {}).get("StartedAt"), "restart_count": inspect.get("RestartCount"),
        "command": command, "environment": serving_env(env), "environment_set_at_start": added,
        "receipt_files": receipt_files,
        "log_lines": scan["raw"], "log_truncated": scan.get("truncated", []), **record,
        "digest": digest(record["settings"]),
    }


def build(ranks: list[Mapping], *, deployment: Mapping | None = None, metrics_text: str | None = None,
          models: Mapping | None = None, context: Mapping | None = None) -> dict:
    """The ``resolved.json`` document of one deployment.

    ``ranks``: per rank ``{rank, position, host, container, inspect, process_env, container_env,
    process_cmdline, log, receipts, clocks}`` (missing parts are allowed); ``deployment``: the
    ``sparkring status --json`` slot's ``deployment`` object; ``metrics_text``: rank 0's ``/metrics``;
    ``models``: its ``/v1/models``; ``context``: how and where it was collected.
    """
    metrics = parse_metrics_info(metrics_text or "")
    built = [build_rank(r, metrics=metrics if r.get("rank", i) == 0 else None, deployment=deployment)
             for i, r in enumerate(ranks)]
    head = built[0] if built else {"settings": {}, "derived": {}, "observed": {}, "sources": {}}
    mismatches = []
    for record in built[1:]:
        for key, value in record["settings"].items():
            # Only values read the same way compare: a worker rank logs no argument or engine line, so its
            # arguments come from its command line, which states fewer values in another form.
            if key not in head["settings"] or head["sources"].get(key) != record["sources"].get(key):
                continue
            if canonical(head["settings"][key]) != canonical(value):
                mismatches.append({"rank": record["rank"], "key": key, "rank0": head["settings"][key], "value": value})
    serving = dict((deployment or {}).get("serving") or {})
    return {
        "schema": SCHEMA,
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **({"context": dict(context)} if context else {}),
        "deployment": {k: (deployment or {}).get(k) for k in (
            "profile", "checkpoint", "model_repository", "model_revision", "image_id", "image_release", "api_url",
            "model", "deployment_id", "source_revision")} | {"serving": serving,
                                                             "transport": (deployment or {}).get("transport")},
        "served_models": [m.get("id") for m in (models or {}).get("data", [])] if models else None,
        "settings": head["settings"], "derived": head["derived"], "observed": head["observed"],
        "sources": head["sources"], "digest": digest(head["settings"]),
        "rank_mismatches": mismatches,
        "ranks": built,
    }


def from_runner_start(directory) -> dict:
    """``resolved.json`` of a serving A/B runner start from the files it saved (``log-r<N>.txt``,
    ``inspect-r<N>.json``); the environment is the container's, as no process environment was read."""
    from pathlib import Path
    directory = Path(directory)
    ranks = []
    for log in sorted(directory.glob("log-r*.txt"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        r = int(re.sub(r"\D", "", log.stem))
        inspect_file = directory / f"inspect-r{r}.json"
        inspect = None
        if inspect_file.exists():
            try:
                loaded = json.loads(inspect_file.read_text(encoding="utf-8") or "null")
                inspect = loaded[0] if isinstance(loaded, list) and loaded else loaded
            except ValueError:
                inspect = None
        ranks.append({"rank": r, "inspect": inspect, "log": log.read_text(encoding="utf-8", errors="replace")})
    return build(ranks, context={"source": f"serving A/B runner start {directory.name}"})


def command_tokens(text: str) -> list[str]:
    """A command line from ``docker inspect`` args (a JSON list) or shell text."""
    text = text.strip()
    if text.startswith("["):
        return [str(t) for t in json.loads(text)]
    return shlex.split(text)


__all__ = ["SCHEMA", "build", "build_rank", "rank_settings", "digest", "flatten", "serving_env", "args_from_command",
           "parse_metrics_info", "from_runner_start", "command_tokens", "pyrepr"]
