"""SSH and container commands of the ring harness (run on the operator's machine).

Every remote action is one ``ssh`` invocation with batch mode and a deadline.
Commands are built from validated site fields and plan values only, and every
interpolated value is shell-quoted. Containers of the harness carry the label
``sircl-ring=<run id>``, so cleanup finds exactly them and nothing else.

Every Docker invocation starts with the Spark's Docker command from the site
file (``docker`` by default, for example ``sudo -n docker`` on a Spark whose
SSH user is not in the docker group); each builder takes it as ``docker``.
"""

from __future__ import annotations

import dataclasses
import shlex
import subprocess

from . import nccl as nccl_mod
from .plan import CONTAINER_PREFIX, ConfigurationPlan, RankPlan
from .site import DEFAULT_DOCKER, docker_problem

SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10",
               "-o", "ServerAliveCountMax=3")
LABEL = "sircl-ring"
CANONICAL_DEVICES = ("rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1")


@dataclasses.dataclass(frozen=True)
class Result:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def ssh(target: str, command: str, *, timeout: float = 60, input_bytes: bytes | None = None,
        binary: str = "ssh") -> Result:
    """Run ``command`` on ``target`` through ssh; a deadline returns code 124."""
    argv = [binary, *SSH_OPTIONS, target, command]
    try:
        process = subprocess.run(argv, input=input_bytes, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return Result(124, "", f"ssh {target} timed out after {timeout:g} s")
    except FileNotFoundError:
        return Result(127, "", f"{binary} is not installed on this machine")
    return Result(process.returncode, process.stdout.decode(errors="replace"),
                  process.stderr.decode(errors="replace"))


def q(value: object) -> str:
    return shlex.quote(str(value))


def docker_command(docker: str = DEFAULT_DOCKER) -> str:
    """The shell words that invoke Docker for the prefix ``docker`` (validated, each word quoted)."""
    problem = docker_problem(docker)
    if problem:
        raise ValueError(f"Docker command {docker!r} {problem}")
    return " ".join(q(word) for word in docker.split())


def worker_command(plan: ConfigurationPlan, rank: RankPlan) -> str:
    run = f"/sircl/runs/{plan.run_id}/{plan.name}"
    return (f"exec python3 -m sparkring_sircl.ring.worker --plan {run}/plan.json "
            f"--global-rank {rank.global_rank} > {run}/rank-{rank.global_rank}.log 2>&1")


def docker_run(plan: ConfigurationPlan, rank: RankPlan) -> str:
    """Start one rank's container (detached; removed by cleanup)."""
    environment = {
        "PYTHONPATH": f"/sircl/src/{plan.source_digest}",
        "SIRCL_BUILD_CACHE_DIR": "/sircl/build-cache",
        "CUTE_DSL_CACHE_DIR": "/sircl/cute-cache",
        "GLOO_SOCKET_IFNAME": plan.lan_interface,
        "CUDA_VISIBLE_DEVICES": "0",
    }
    if plan.tuning_tables:
        # The run writes every table beside plan.json; each session takes the one whose key matches it.
        environment["SIRCL_TUNING_TABLE"] = ",".join(f"/sircl/runs/{plan.run_id}/{plan.name}/tuning-{digest}.json"
                                                     for digest, _ in plan.tuning_tables)
    if plan.options.baseline == "nccl":
        # Set at container start, so the image's NCCL is preloaded before torch loads its own and NCCL logs
        # its initialization and network setup to a file the worker reads.
        environment.update(nccl_mod.DEBUG_SETTINGS)
        environment["NCCL_DEBUG_FILE"] = f"/sircl/runs/{plan.run_id}/{plan.name}/nccl-rank-{rank.global_rank}.log"
        environment.update(nccl_mod.environment(plan.lan_interface, plan.gid_index, plan.options.nccl_library,
                                                plan.options.nccl_env))
    parts = [docker_command(rank.docker), "run", "-d", "--name", q(rank.container),
             "--label", q(f"{LABEL}={plan.run_id}"),
             "--privileged", "--gpus", q("device=0"), "--network", "host", "--ipc", "host",
             "--ulimit", "memlock=-1", "--entrypoint", "bash", "-v", q(f"{plan.remote_dir}:/sircl")]
    for name, value in environment.items():
        parts += ["-e", q(f"{name}={value}")]
    parts += [q(plan.image), "-lc", q(worker_command(plan, rank))]
    return " ".join(parts)


def prepare_command(image: str, remote_dir: str, run_id: str, source_digest: str,
                    docker: str = DEFAULT_DOCKER) -> str:
    """Build the native library inside the image into the shared build cache."""
    return " ".join([
        docker_command(docker), "run", "--rm", "--name", q(f"{CONTAINER_PREFIX}-{run_id}-prepare"),
        "--label", q(f"{LABEL}={run_id}"), "--entrypoint", "bash", "-v", q(f"{remote_dir}:/sircl"),
        "-e", q(f"PYTHONPATH=/sircl/src/{source_digest}"), "-e", "SIRCL_BUILD_CACHE_DIR=/sircl/build-cache",
        q(image), "-lc",
        q("python3 -m sparkring_sircl.build && python3 -c 'import torch, cutlass, cuda.bindings; "
          "print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda)'"),
    ])


def container_state(name: str, docker: str = DEFAULT_DOCKER) -> str:
    return (f"{docker_command(docker)} inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {q(name)} "
            "2>/dev/null || echo missing")


def remove_harness_containers(run_id: str | None = None, docker: str = DEFAULT_DOCKER) -> str:
    """Remove the harness's containers (of one run, or all) and print how many remain.

    Exits non-zero when Docker cannot be reached, so a refused Docker command
    never reads as zero containers left.
    """
    d = docker_command(docker)
    label = q(LABEL if run_id is None else f"{LABEL}={run_id}")
    return (f"ids=$({d} ps -aq --filter label={label}) || exit 1; "
            f"if [ -n \"$ids\" ]; then {d} rm -f $ids >/dev/null || exit 1; fi; "
            f"left=$({d} ps -aq --filter label={label}) || exit 1; "
            f"printf '%s\\n' \"$left\" | awk 'NF' | wc -l")


def running_containers(docker: str = DEFAULT_DOCKER) -> str:
    """Running containers with their harness label (empty for non-harness containers)."""
    return (docker_command(docker)
            + " ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Label \"" + LABEL + "\"}}'")


def preflight_script(image: str, lan_interface: str, docker: str = DEFAULT_DOCKER) -> str:
    """Read-only facts of one Spark, one ``key<TAB>value`` line each.

    The ``docker`` line holds the server version, or ``unusable: <last error
    line>`` when the Spark's Docker command cannot reach the daemon.
    """
    devices = " ".join(CANONICAL_DEVICES)
    d = docker_command(docker)
    return "; ".join([
        "echo \"hostname\t$(hostname)\"",
        f"if v=$({d} info --format '{{{{.ServerVersion}}}}' 2>/dev/null); then echo \"docker\t$v\"; "
        f"else echo \"docker\tunusable: $({d} info 2>&1 >/dev/null | tail -n 1)\"; fi",
        f"echo \"image\t$({d} image inspect --format '{{{{.Id}}}}' {q(image)} 2>/dev/null || echo missing)\"",
        "echo \"gpu\t$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | head -1 || echo missing)\"",
        f"for d in {devices}; do echo \"device:$d\t$(cat /sys/class/infiniband/$d/ports/1/state 2>/dev/null || echo missing)"
        f" $(ls /sys/class/infiniband/$d/device/net 2>/dev/null | head -1)\"; done",
        f"echo \"lan\t$(ip -o -4 addr show dev {q(lan_interface)} 2>/dev/null | awk '{{print $4}}' | head -1)\"",
        "ip -o -4 addr show | awk '{print \"address:\" $2 \"\\t\" $4}'",
        "for f in /sys/class/infiniband/*/ports/1/hw_counters/out_of_sequence; do "
        "[ -e \"$f\" ] && echo \"counters\tpresent\" && break; done",
    ])


def route_get(destinations: list[str]) -> str:
    return "; ".join(f"echo \"route:{d}\t$(ip -o route get {q(d)} 2>/dev/null | head -1)\"" for d in destinations)


def write_file(path: str) -> str:
    return f"mkdir -p {q(path.rsplit('/', 1)[0])} && cat > {q(path)}"


def read_file(path: str) -> str:
    return f"cat {q(path)} 2>/dev/null"


def extract_tar(directory: str) -> str:
    """Unpack a tar stream into ``directory`` (a source-digest directory, written once)."""
    return f"mkdir -p {q(directory)} && tar -C {q(directory)} -xf -"
