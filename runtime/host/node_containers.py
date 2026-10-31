"""Bounded local model observations; never publish argv or environment.

Docker inspect uses its local Unix socket. GPU PIDs already collected by the
agent identify docker-exec servers whose launch differs from the entrypoint,
and bare ``flare run`` processes that the Orbit agent or an operator started.
No command is executed and no environment is read from /proc.
"""
import hashlib
import http.client
import ipaddress
import json
from pathlib import Path
import re
import shlex
import socket
import time

MAX_CONTAINERS = 32
MAX_FLARE_PROCESSES = 16
MAX_ARGV_BYTES = 16384
MAX_ARGS = 256
MAX_GPU_PIDS = 256
MAX_RESPONSE = 256 * 1024
MAX_INSPECT_SECONDS = 1.0
MANAGED_NAME = re.compile(r"(?:sr-)?[A-Za-z0-9][A-Za-z0-9_.-]*-[0-9a-f]{6}-r[0-9]+")
MANAGED_LABELS = ("io.sparkring.deployment", "org.sparkring.profile", "io.sparkring.stock-plan")
FLARE_BINARY = re.compile(r"flare(?:-[0-9a-f]{8,64})?")
INTEGER_FLAGS = {
    "tp": ("--tensor-parallel-size", "--tp-size", "--tp", "-tp"),
    "pp": ("--pipeline-parallel-size", "--pp-size", "--pp", "-pp"),
    "dp": ("--data-parallel-size", "--dp-size", "--dp", "-dp"),
    "dcp": ("--decode-context-parallel-size", "--dcp-size", "-dcp"),
    "nnodes": ("--nnodes", "-n"), "node_rank": ("--node-rank", "-r"),
    "api_port": ("--port",), "master_port": ("--master-port",),
    "dp_port": ("--data-parallel-rpc-port", "-dpp"),
    "dp_rank": ("--data-parallel-rank", "-dpn"),
}
TEXT_FLAGS = {"master_addr": ("--master-addr",), "api_bind": ("--host",),
              "dp_addr": ("--data-parallel-address", "-dpa"),
              "dist_addr": ("--dist-init-addr", "--nccl-init-addr"),
              "served_model": ("--served-model-name",),
              "model": ("--model", "--model-path")}


def bounded_argv(argv, depth=0):
    if depth > 2 or not isinstance(argv, list) or len(argv) > MAX_ARGS:
        return []
    total = 0
    for arg in argv:
        if not isinstance(arg, str) or len(arg) > MAX_ARGV_BYTES:
            return []
        total += len(arg.encode("utf-8")) + 1
        if total > MAX_ARGV_BYTES:
            return []
    # Shell entrypoints are parsed as text only; substitutions are never expanded.
    if len(argv) >= 3 and Path(argv[0]).name in ("sh", "bash", "dash") and argv[1] in ("-c", "-lc"):
        try:
            return bounded_argv(shlex.split(argv[2]), depth + 1)
        except ValueError:
            return []
    return argv


def flare_run_index(argv):
    """The index of ``run`` in a ``flare run`` invocation, or None.

    The binary is ``flare`` or a content-addressed ``flare-<hex>``; ``flare
    agent``, ``flare orbit ...`` and other subcommands are not serving ranks.
    """
    for index, arg in enumerate(argv[:4]):
        if FLARE_BINARY.fullmatch(Path(arg).name) and index + 1 < len(argv) and argv[index + 1] == "run":
            return index + 1
    return None


def argument_values(argv, start):
    """Literal ``--flag value`` and ``--flag=value`` pairs from ``start``; variables stay unknown."""
    values: dict = {}
    index = start
    while index < len(argv):
        arg = argv[index]
        key, separator, value = arg.partition("=")
        if not separator:
            following = argv[index + 1] if index + 1 < len(argv) else ""
            # A negative memory limit is a value, not the next option.
            if following and (not following.startswith("-") or re.fullmatch(r"-[0-9][0-9A-Za-z.]*", following)):
                value = following
                index += 1
            else:
                value = ""
        values[key] = value
        index += 1
    return values


def bounded_text(value):
    return isinstance(value, str) and 0 < len(value) <= 512 and not any(c in value for c in "$`\n\r")


def flare_launch_facts(argv):
    """Public launch facts of one ``flare run`` process, or None for another command.

    Only the recognized serving options are published: rank, rendezvous,
    HTTP bind and port, and the served alias. Package paths, trust stores,
    publisher keys, API keys and the rest of argv stay private.
    """
    run = flare_run_index(argv)
    if run is None:
        return None
    values = argument_values(argv, run + 1)
    rank = values.get("--rank", "")
    if rank and not re.fullmatch(r"[0-9]{1,5}", rank):
        return {"engine": "unknown"}
    rank = int(rank) if rank else 0
    if rank > 65535:
        return {"engine": "unknown"}
    facts: dict = {"engine": "flare", "node_rank": rank, "headless": rank != 0}
    rendezvous = values.get("--rendezvous")
    if isinstance(rendezvous, str) and bounded_text(rendezvous):
        host, separator, port = rendezvous.rpartition(":")
        host = host.strip("[]")
        try:
            ipaddress.ip_address(host)
            valid_host = True
        except ValueError:
            valid_host = bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}", host))
        if separator and port.isdigit() and 0 < int(port) < 65536 and valid_host:
            facts.update(master_addr=host, master_port=int(port))
    if rank == 0:
        # Flare's --host is an IpAddr with a 127.0.0.1 default. An unparsable
        # value is not published as loopback, which would claim reachability
        # the process never had.
        provided = values.get("--host")
        bind = (provided if provided is not None else "127.0.0.1").removeprefix("[").removesuffix("]")
        port = values.get("--port") or "8000"
        try:
            ipaddress.ip_address(bind)
        except ValueError:
            bind = None
        if bind is not None and re.fullmatch(r"[0-9]{1,5}", port) and 0 < int(port) < 65536:
            facts.update(api_bind=bind, api_port=int(port))
    served = values.get("--served-model-name")
    if isinstance(served, str) and bounded_text(served):
        facts["served_model"] = served
    return facts


def flare_row_name(facts):
    """A stable row name for one bare process, independent of its PID."""
    key = [facts.get("master_addr"), facts.get("master_port"), facts.get("node_rank", 0),
           facts.get("api_bind"), facts.get("api_port")]
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    return f"flare-r{facts.get('node_rank', 0)}-{digest}"


def thinking_kwargs(value):
    """Only bounded thinking arguments are public launch facts; arbitrary template data stays private."""
    if isinstance(value, str):
        if len(value) > MAX_ARGV_BYTES:
            return None
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, dict):
        return None
    return {k: v for k, v in value.items()
            if (k in ("enable_thinking", "thinking") and type(v) is bool
                or k == "reasoning_effort" and (type(v) is int and 1 <= v <= 100
                                               or isinstance(v, str) and re.fullmatch(r"[a-z]{1,16}", v)))}


def launch_facts(argv):
    argv = bounded_argv(argv)
    flare = flare_launch_facts(argv)
    if flare is not None:
        return flare
    engine = next(("vllm" if "vllm" in a else "sglang" for a in argv
                   if a in ("vllm", "sglang", "sglang.launch_server", "vllm.entrypoints.openai.api_server",
                            "vllm.entrypoints.cli.main")
                   or a.endswith(("/vllm", "/sglang"))), "unknown")
    # SparkRing's serving wrapper execs the vLLM CLI with these same arguments.
    wrapper = "/opt/sparkring/toolchain/toolchain.py"
    if any(a == wrapper and argv[i + 1:i + 2] == ["serve"] for i, a in enumerate(argv)):
        engine = "vllm"
    facts: dict = {"engine": engine}
    if engine == "unknown" or any(a in argv for a in ("bench", "preload", "--help", "--version")):
        return {"engine": "unknown"}
    # Keep only literal, recognized values. Shell variables are unknown, not facts.
    values: dict = {}
    for i, arg in enumerate(argv):
        key, sep, value = arg.partition("=")
        if not sep:
            value = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("-") else ""
        values[key] = value
    for key, flags in INTEGER_FLAGS.items():
        value = next((values[f] for f in flags if f in values), None)
        if value is not None and re.fullmatch(r"[0-9]{1,5}", value):
            number = int(value)
            if (0 <= number <= 65535 if key.endswith("rank") else 0 < number <= 65535):
                facts[key] = number
    for key, flags in TEXT_FLAGS.items():
        value = next((values[f] for f in flags if f in values), None)
        if value and len(value) <= 512 and not any(c in value for c in "$`\n\r"):
            facts[key] = value
    kwargs = thinking_kwargs(values.get("--default-chat-template-kwargs"))
    if kwargs is not None:
        facts["default_chat_template_kwargs"] = kwargs
    for key in ("tp", "pp", "dp", "dcp"):
        if "--config" not in values and not any(f in values for f in INTEGER_FLAGS[key]):
            facts.setdefault(key, 1)
    facts["headless"] = "--headless" in values
    if "dist_addr" in facts:
        address, sep, port = facts.pop("dist_addr").rpartition(":")
        if sep and port.isdigit() and 0 < int(port) < 65536:
            facts.update(master_addr=address.strip("[]"), master_port=int(port))
    if "master_addr" not in facts and "dp_addr" in facts:
        facts["master_addr"] = facts["dp_addr"]
        if "dp_port" in facts:
            facts["master_port"] = facts["dp_port"]
        if "dp_rank" in facts:
            facts.setdefault("node_rank", facts["dp_rank"])
            if "dp" in facts:
                facts.setdefault("nnodes", facts["dp"])
    elif engine == "vllm" and "master_addr" in facts and "--master-port" not in values and "--config" not in values:
        facts.setdefault("master_port", 29501)
    if "model" not in facts:
        for i, arg in enumerate(argv[:-2]):
            if (Path(arg).name == "vllm" or arg in (wrapper, "vllm.entrypoints.cli.main")) \
                    and argv[i + 1] == "serve" and not argv[i + 2].startswith("-"):
                value = argv[i + 2]
                if len(value) <= 512 and not any(c in value for c in "$`\n\r"):
                    facts["model"] = value
    facts["served_model"] = facts.get("served_model") or facts.pop("model", None)
    facts.pop("model", None)
    if not facts["headless"]:
        # Documented HTTP defaults for a recognized server invocation, never a port scan.
        if "--port" not in values and "--config" not in values:
            facts.setdefault("api_port", 8000 if engine == "vllm" else 30000)
    else:
        facts.pop("api_port", None)
    if "--node-rank" not in values and "-r" not in values and "--config" not in values:
        facts.setdefault("node_rank", 0)
    if "--nnodes" not in values and "-n" not in values and "--config" not in values:
        facts.setdefault("nnodes", 1)
    return facts


def inspect_local(container_id, root="/"):
    if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
        return {}
    connection = http.client.HTTPConnection("localhost", timeout=.25)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(.25)
        sock.connect(str(Path(root) / "var/run/docker.sock"))
        connection.sock = sock
        connection.request("GET", f"/containers/{container_id}/json")
        response = connection.getresponse()
        payload = response.read(MAX_RESPONSE + 1)
        return json.loads(payload) if response.status == 200 and len(payload) <= MAX_RESPONSE else {}
    finally:
        connection.close()
        sock.close()


def observations(facts, *, root="/", inspect=inspect_local):
    """At most 32 running containers, 256 existing GPU PIDs and 16 KiB per argv."""
    rows = (facts.get("docker") or {}).get("containers") or []
    candidates = []
    for row in rows:
        if row.get("state") == "running":
            candidates.append(row)
            if len(candidates) == MAX_CONTAINERS:
                break
    gpu_argv, gpu_parsed, gpu_container = {}, {}, {}
    for gpu_pid in ((facts.get("gpu") or {}).get("compute_processes") or [])[:MAX_GPU_PIDS]:
        if type(gpu_pid) is not int or gpu_pid <= 0:
            continue
        pid, parsed, container = gpu_pid, {"engine": "unknown"}, None
        try:
            for _ in range(8):
                proc = Path(root) / "proc" / str(pid)
                with (proc / "cgroup").open("rb") as stream:
                    groups = stream.read(8192).decode("utf-8", "replace")
                found = next((r.get("id") for r in candidates if r.get("id") and r["id"] in groups), None)
                if found:
                    container = found
                # Read every hop, not only container members: a bare `flare run`
                # process has no container cgroup to match.
                with (proc / "cmdline").open("rb") as stream:
                    raw = stream.read(MAX_ARGV_BYTES + 1)
                here = (launch_facts(raw.rstrip(b"\0").decode("utf-8", "replace").split("\0"))
                        if len(raw) <= MAX_ARGV_BYTES else {"engine": "unknown"})
                if parsed["engine"] == "unknown" or here["engine"] != "unknown":
                    parsed = here
                if parsed["engine"] != "unknown":
                    break
                with (proc / "status").open("rb") as stream:
                    parent = re.search(rb"^PPid:\s+(\d+)", stream.read(8192), re.M)
                if not parent or int(parent[1]) <= 1 or int(parent[1]) == pid:
                    break
                pid = int(parent[1])
        except OSError:
            continue
        if parsed["engine"] != "unknown":
            gpu_parsed[gpu_pid] = parsed
        if container:
            gpu_container[gpu_pid] = container
            if container not in gpu_argv or parsed["engine"] != "unknown":
                gpu_argv[container] = parsed
    result = []
    deadline = time.monotonic() + MAX_INSPECT_SECONDS
    for row in candidates:
        try:
            doc = inspect(row.get("id", ""), root=root) if time.monotonic() < deadline else {}
        except (OSError, ValueError, http.client.HTTPException, AttributeError):
            doc = {}
        config = doc.get("Config") or {}
        if doc.get("State", {}).get("Running") is False:
            continue
        parsed = launch_facts([doc["Path"], *(doc.get("Args") or [])]) if doc.get("Path") else launch_facts(
            (config.get("Entrypoint") or []) + (config.get("Cmd") or []))
        gpu = gpu_argv.get(row.get("id"))
        if gpu and gpu["engine"] != "unknown":
            parsed = gpu
        if parsed["engine"] == "unknown" and gpu is None:
            continue
        name = str(row.get("name") or "")[:128]
        labels = config.get("Labels") or {}
        result.append({"name": name, "image": str(config.get("Image") or row.get("image") or "")[:512],
                       "image_id": str(doc.get("Image") or "").removeprefix("sha256:")[:12],
                       "managed": bool(MANAGED_NAME.fullmatch(name) or any(labels.get(k) for k in MANAGED_LABELS)),
                       **parsed})
    bare = 0
    for pid, parsed in gpu_parsed.items():
        if bare >= MAX_FLARE_PROCESSES:
            break
        if pid in gpu_container or parsed.get("engine") != "flare":
            continue
        result.append({"name": flare_row_name(parsed), "image": "", "image_id": "", "managed": False, **parsed})
        bare += 1
    return result
