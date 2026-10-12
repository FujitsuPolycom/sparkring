"""Read what a deployment on the ring serves, read-only, through Node A.

Node A is reached by SSH (``--node-a``); every other read runs on Node A: ``sudo -n sparkring status --json``
names the deployment on the requested positions, its containers and their hosts, and each rank's Spark is
read with ``sudo -n ssh HOST sh -s``, the read script on standard input. Per rank the script prints
``docker inspect``, the serving subset of the container's environment and of the vLLM process's
environment (``/proc/<pid>/environ`` of the ``vllm`` serve process, which holds variables the toolchain
entrypoint adds), the process's command line, the log lines :mod:`.vllm_log` recognises, the SIRCL receipt
files under ``/run/sparkring/sircl/receipts`` and ``nvidia-smi`` clocks. Rank 0's ``/metrics`` and
``/v1/models`` are read over HTTP from this machine (through Node A's ``curl`` when this machine cannot
reach the API). Nothing is started, stopped, written or locked on the ring; no generation request is sent.

Environment values leave a Spark only for the variable prefixes of :data:`.resolved.ENV_PREFIXES`, so tokens
in a container's environment are not read.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from . import resolved, vllm_log

SSH = (os.environ.get("SERVING_AB_SSH", "ssh"), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
MARK = "@@SERVING-AB-RECEIPT@@"
RECEIPT_DIR = "/run/sparkring/sircl/receipts"
CLOCK_QUERY = "clocks.sm,clocks.max.sm,clocks_event_reasons.active,clocks.applications.graphics,temperature.gpu,power.draw"


class CollectError(RuntimeError):
    pass


class NodeA:
    """Commands on Node A over SSH and, through it, read scripts on a Spark's host as root."""

    def __init__(self, target: str, ssh: Sequence[str] = SSH, timeout: float = 300):
        self.target, self.ssh, self.timeout = target, tuple(ssh), timeout

    def run(self, command: str, *, stdin: str | None = None, check: bool = True) -> str:
        # Bytes, not text mode: a Windows text-mode pipe would turn the script's newlines into CRLF.
        result = subprocess.run([*self.ssh, self.target, command], capture_output=True, timeout=self.timeout,
                                input=None if stdin is None else stdin.encode("utf-8"))
        stdout = result.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
        if check and result.returncode:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise CollectError(f"{self.target}: {command[:120]!r} exited {result.returncode}: {stderr[-400:]}")
        return stdout

    def spark(self, host: str, script: str) -> str:
        """Run ``script`` with ``sh`` on ``host``, the SSH target Node A reaches the Spark by (``root@ADDRESS``)."""
        command = f"sudo -n ssh -o BatchMode=yes -o ConnectTimeout=10 {shlex.quote(host)} sh -s"
        return self.run(command, stdin=script)


def env_filter() -> str:
    return "^(" + "|".join(p.rstrip("_") + ("_" if p.endswith("_") else "") for p in resolved.ENV_PREFIXES) + ")"


# docker inspect fields read, without Config.Env (the environment is read through env_filter instead).
INSPECT_FORMAT = ('{"Name":{{json .Name}},"Image":{{json .Image}},"Created":{{json .Created}},'
                  '"Path":{{json .Path}},"Args":{{json .Args}},"State":{{json .State}},'
                  '"RestartCount":{{json .RestartCount}},"Config":{"Image":{{json .Config.Image}},'
                  '"Entrypoint":{{json .Config.Entrypoint}},"Cmd":{{json .Config.Cmd}},'
                  '"Labels":{{json .Config.Labels}}}}')


def rank_script(container: str) -> str:
    """The read-only script that prints one rank's sections (see the module docstring)."""
    grep = shlex.quote(vllm_log.GREP_PATTERN)
    keep = shlex.quote(env_filter())
    inspect = shlex.quote(INSPECT_FORMAT)
    # [v]llm: the pattern does not match the pgrep's own shell, whose command line holds "[v]llm".
    return f"""C={shlex.quote(container)}
m() {{ printf '\\n{MARK} %s\\n' "$1"; }}
m inspect; docker inspect --format {inspect} "$C"
m container_env; docker exec "$C" env | grep -E {keep}
P=$(docker exec "$C" sh -c 'pgrep -o -f "[v]llm.entrypoints|[v]llm serve" 2>/dev/null || echo 1')
m process_pid; echo "$P"
m process_cmdline; docker exec "$C" sh -c "tr '\\\\0' '\\\\n' < /proc/$P/cmdline"
m process_env; docker exec "$C" sh -c "tr '\\\\0' '\\\\n' < /proc/$P/environ" | grep -E {keep}
m log; docker logs "$C" 2>&1 | grep -E {grep} | grep -v -E '"(GET|POST|PUT|DELETE) /' | head -n 3000
for f in $(docker exec "$C" sh -c 'ls {RECEIPT_DIR}/*.json 2>/dev/null'); do m "receipt $(basename "$f")"; docker exec "$C" cat "$f"; done
m clocks; nvidia-smi --query-gpu={CLOCK_QUERY} --format=csv,noheader 2>&1
m end
"""


def sections(text: str) -> dict:
    """``{name: text}`` of a rank script's output; receipt files under ``receipts``."""
    out: dict = {"receipts": {}}
    name, lines = None, []

    def close():
        body = "\n".join(lines).strip("\n")
        if name and name.startswith("receipt "):
            out["receipts"][name.split(" ", 1)[1]] = body
        elif name:
            out[name] = body

    for line in text.splitlines():
        if line.startswith(MARK + " "):
            close()
            name, lines = line[len(MARK) + 1:].strip(), []
        elif name:
            lines.append(line)
    close()
    return out


def raw_rank(text: str, *, rank: int, position: int | None, host: str, container: str) -> dict:
    """:func:`.resolved.build`'s per-rank input from a rank script's output."""
    parts = sections(text)
    if "end" not in parts:
        raise CollectError(f"{host}: the read of {container} ended early")
    try:
        loaded = json.loads(parts.get("inspect") or "null")
        inspect = loaded[0] if isinstance(loaded, list) and loaded else loaded if isinstance(loaded, dict) else None
    except ValueError:
        inspect = None
    receipts = {}
    for name, body in parts["receipts"].items():
        try:
            receipts[name] = json.loads(body)
        except ValueError:
            receipts[name] = {"unreadable": body[:200]}
    cmdline = [t for t in (parts.get("process_cmdline") or "").splitlines() if t]
    return {"rank": rank, "position": position, "host": host, "container": container, "inspect": inspect,
            "container_env": resolved.parse_env(parts.get("container_env") or ""),
            "process_env": resolved.parse_env(parts["process_env"]) if parts.get("process_env") else None,
            "process_cmdline": cmdline, "log": parts.get("log") or "", "receipts": receipts,
            "clocks": parts.get("clocks") or ""}


def slot_for(status: dict, positions: Sequence[int]) -> dict:
    """The ``slots`` entry of ``sparkring status --json`` whose placement is ``positions``."""
    for slot in status.get("slots") or []:
        if list(slot.get("placement") or []) == list(positions) or sorted(slot.get("placement") or []) == sorted(positions):
            return slot
    found = [slot.get("placement") for slot in status.get("slots") or []]
    raise CollectError(f"no deployment slot on positions {list(positions)}; status lists {found}")


def http_get(url: str, node_a: NodeA | None = None, timeout: float = 20) -> str | None:
    """GET ``url`` from this machine, else with ``curl`` on Node A; None when neither answers."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError):
        if node_a is None:
            return None
        text = node_a.run(f"curl -s -m {int(timeout)} {shlex.quote(url)}", check=False)
        return text or None


def api_base(api_url: str) -> str:
    return api_url[:-3] if api_url.rstrip("/").endswith("/v1") else api_url.rstrip("/")


def collect(node_a: NodeA, positions: Sequence[int], *, sparks: dict[int, str] | None = None,
            containers: dict[int, str] | None = None, api: str | None = None) -> tuple[dict, dict]:
    """``(resolved document, raw reads by file name)`` of the deployment on ``positions``.

    ``sparks`` and ``containers`` (position -> host, position -> container name) replace what
    ``sparkring status --json`` reports, for a deployment the status does not list; ``api`` replaces its API
    URL.
    """
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status_text = node_a.run("sudo -n sparkring status --json", check=False)
    try:
        status = json.loads(status_text)
    except ValueError:
        status = {}
    deployment: dict = {}
    listed: dict[int, tuple[str, str]] = {}
    try:
        slot = slot_for(status, positions)
        deployment = dict(slot.get("deployment") or {})
        placement = list(slot.get("placement") or positions)
        for item in deployment.get("containers") or []:
            listed[placement[item["rank"]]] = (item["host"], item["name"])
    except CollectError:
        if not sparks:
            raise
    ranks, texts = [], {}
    for rank, position in enumerate(positions):
        host = (sparks or {}).get(position) or listed.get(position, (None, None))[0]
        name = (containers or {}).get(position) or listed.get(position, (None, None))[1]
        if not host:
            raise CollectError(f"position {position}: no host (status lists none; give --spark {position}=HOST)")
        if not name:
            found = node_a.spark(host, "docker ps --format '{{.Names}}'").split()
            name = next((n for n in found if n.endswith(f"-r{rank}")), None)
            if not name:
                raise CollectError(f"position {position}: no container ending -r{rank} among {found}")
        text = node_a.spark(host, rank_script(name))
        texts[rank] = text
        ranks.append(raw_rank(text, rank=rank, position=position, host=host, container=name))
    url = api or deployment.get("api_url")
    metrics = models = None
    if url:
        metrics = http_get(api_base(url) + "/metrics", node_a)
        listed_models = http_get(api_base(url) + "/v1/models", node_a)
        try:
            models = json.loads(listed_models) if listed_models else None
        except ValueError:
            models = None
    context = {"collected_at": started, "node_a": node_a.target, "positions": list(positions), "api": url,
               "method": "read-only: sparkring status --json, docker inspect/exec env/logs, receipt files, "
                         "nvidia-smi, GET /metrics and /v1/models"}
    document = resolved.build(ranks, deployment=deployment, metrics_text=metrics, models=models, context=context)
    raw = {f"rank{rank}.txt": text for rank, text in texts.items()}
    raw.update({"deployment.json": json.dumps(deployment, indent=1), "metrics.txt": metrics or "",
                "models.json": json.dumps(models), "context.json": json.dumps(context, indent=1)})
    return document, raw


def rebuild(raw_dir: Path) -> dict:
    """``resolved.json`` again from a collection's ``raw`` directory, offline (after a parser change)."""
    raw_dir = Path(raw_dir)
    deployment = json.loads((raw_dir / "deployment.json").read_text(encoding="utf-8"))
    placement = {item["rank"]: (item["host"], item["name"]) for item in deployment.get("containers") or []}
    context = json.loads((raw_dir / "context.json").read_text(encoding="utf-8"))
    context.setdefault("collected_at", time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime((raw_dir / "rank0.txt").stat().st_mtime)))
    ranks = []
    for path in sorted(raw_dir.glob("rank*.txt"), key=lambda p: int(p.stem[4:])):
        rank = int(path.stem[4:])
        host, name = placement.get(rank, (None, None))
        positions = context.get("positions") or []
        ranks.append(raw_rank(path.read_text(encoding="utf-8"), rank=rank, host=host, container=name,
                              position=positions[rank] if rank < len(positions) else None))
    models = json.loads((raw_dir / "models.json").read_text(encoding="utf-8"))
    return resolved.build(ranks, deployment=deployment, metrics_text=(raw_dir / "metrics.txt").read_text(encoding="utf-8"),
                          models=models, context=context)


def write(document: dict, raw: dict, output: Path) -> Path:
    """``output/resolved.json`` and the raw reads under ``output/raw/`` (each rank's read, the status entry,
    ``/metrics`` and ``/v1/models``), from which :func:`rebuild` recomputes the receipt."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    for name, text in raw.items():
        (output / "raw" / name).write_text(text, encoding="utf-8")
    path = output / "resolved.json"
    path.write_text(json.dumps(document, indent=1, default=str) + "\n", encoding="utf-8")
    return path
