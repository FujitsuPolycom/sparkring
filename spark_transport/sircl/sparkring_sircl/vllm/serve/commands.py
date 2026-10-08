"""Shell commands the serve launcher runs on a Spark (one ``ssh`` invocation each).

Every builder returns one command line for the Spark's shell, built only from
validated plan values with every interpolated value quoted. Docker
invocations start with the Spark's Docker command from the site file
(:func:`sparkring_sircl.ring.remote.docker_command`). Commands that print
facts print one ``key<TAB>value`` line per fact.

Host-side file operations outside the launcher's remote directory, creating
the cache and run directories and reading the model and cache directories,
run through the Spark's ``sudo`` prefix when it has one (:mod:`.sitefile`):
``<prefix> sh -c '<script>'``. Staging under the remote directory runs as
the SSH user.

Containers of the launcher carry the label ``sircl-serve=<run id>``; the
removal command filters on that label, so it never touches another
deployment's containers.
"""

from __future__ import annotations

import posixpath
import shlex
from collections.abc import Sequence

from ...ring.remote import docker_command
from .plan import (BUILD_TARGET, CONTAINER_PREFIX, LABEL, OVERLAY_FILES, OVERLAY_RECORD, OVERLAY_TARGET,
                   SOURCE_TARGET, Mount, RankLaunch, ServePlan)
from .profile import ServingProfile

STAGED_MARKER = ".sircl-staged"
LABELS = (LABEL, "sircl-ring", "io.sparkring.deployment")


def q(value: object) -> str:
    return shlex.quote(str(value))


def privileged(prefix: str, script: str) -> str:
    """``script`` run plainly, or through ``prefix`` (for example ``sudo -n``) as ``sh -c``."""
    words = prefix.split()
    if not words:
        return script
    return " ".join([*(q(word) for word in words), "sh", "-c", q(script)])


def running_containers(docker: str) -> str:
    """Running containers: name, image, status and the labels of SIRCL and SparkRing deployments."""
    fields = "\t".join(["{{.Names}}", "{{.Image}}", "{{.Status}}",
                        *("{{.Label " + f'"{label}"' + "}}" for label in LABELS)])
    return f"{docker_command(docker)} ps --format {q(fields)}"


def run_containers(run_id: str, docker: str) -> str:
    """Containers of one run in any state: name and state."""
    return (f"{docker_command(docker)} ps -a --filter {q('label=' + LABEL + '=' + run_id)} "
            f"--format {q('{{.Names}}' + chr(9) + '{{.State}}')}")


def remove_containers(run_id: str | None, docker: str) -> str:
    """Remove the launcher's containers (of one run, or of every run) and print how many remain.

    Exits non-zero when Docker cannot be reached, so a refused Docker command
    never reads as zero containers left.
    """
    d = docker_command(docker)
    label = q(LABEL if run_id is None else f"{LABEL}={run_id}")
    return (f"ids=$({d} ps -aq --filter label={label}) || exit 1; "
            f"if [ -n \"$ids\" ]; then {d} rm -f $ids >/dev/null || exit 1; fi; "
            f"left=$({d} ps -aq --filter label={label}) || exit 1; "
            f"printf '%s\\n' \"$left\" | awk 'NF' | wc -l")


def container_state(container: str, docker: str) -> str:
    """``<status> <exit code> <health>``, or ``missing``."""
    template = "{{.State.Status}} {{.State.ExitCode}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}"
    return f"{docker_command(docker)} inspect -f {q(template)} {q(container)} 2>/dev/null || echo missing"


def container_logs(container: str, docker: str, tail: int | None = None) -> str:
    tail_option = f" --tail {int(tail)}" if tail is not None else ""
    return f"{docker_command(docker)} logs{tail_option} {q(container)} 2>&1"


def log_lines(container: str, docker: str, patterns: Sequence[str]) -> str:
    """Lines of the container's whole log holding any of ``patterns`` (fixed strings)."""
    expressions = " ".join(f"-e {q(pattern)}" for pattern in patterns)
    return f"{docker_command(docker)} logs {q(container)} 2>&1 | grep -F {expressions} || true"


def receipts(receipt_dir: str, global_rank: int) -> str:
    """``<file name><TAB><JSON on one line>`` for every receipt of one rank."""
    return (f"cd {q(receipt_dir)} 2>/dev/null || exit 0; "
            f"for f in rank{int(global_rank)}-*.json; do [ -f \"$f\" ] || continue; "
            "printf '%s\\t' \"$f\"; tr -d '\\n' < \"$f\"; echo; done")


def api_get(port: int, path: str, timeout: float = 10) -> str:
    return f"curl -sS -m {timeout:g} {q(f'http://127.0.0.1:{int(port)}{path}')}"


def api_post(port: int, path: str, timeout: float) -> str:
    """POST the JSON body on stdin; the last output line is curl's total time in seconds."""
    return (f"curl -sS -m {timeout:g} -H 'Content-Type: application/json' --data-binary @- "
            f"-w '\\n%{{time_total}}\\n' {q(f'http://127.0.0.1:{int(port)}{path}')}")


def make_directories(paths: Sequence[str], sudo: str = "") -> str:
    return privileged(sudo, "mkdir -p " + " ".join(q(path) for path in paths))


def stage_tree(directory: str, digest: str) -> str:
    """Unpack the staged tree from stdin into ``directory`` once (content-addressed, never rewritten).

    An existing complete directory is kept and stdin is drained. A directory
    without the completion marker is left in place and reported, for the
    operator to inspect and remove.
    """
    parent = posixpath.dirname(directory)
    marker = f"{directory}/{STAGED_MARKER}"
    problem = f"incomplete: {directory} exists without {STAGED_MARKER}; inspect and remove it, then stage again"
    return (f"if [ -f {q(marker)} ]; then cat >/dev/null; echo present; "
            f"elif [ -e {q(directory)} ]; then cat >/dev/null; echo {q(problem)} >&2; exit 1; "
            f"else mkdir -p {q(parent)} && tmp={q(directory)}.partial.$$ && mkdir \"$tmp\" && "
            f"tar -C \"$tmp\" -xf - && printf '%s\\n' {q(digest)} > \"$tmp\"/{STAGED_MARKER} && "
            f"mv -T \"$tmp\" {q(directory)} && echo staged; fi")


def write_once(path: str, sha256: str) -> str:
    """Write stdin to ``path`` unless it already holds content with SHA-256 ``sha256``."""
    directory = posixpath.dirname(path)
    return (f"if [ -f {q(path)} ] && [ \"$(sha256sum {q(path)} | cut -c1-64)\" = {q(sha256)} ]; then "
            f"cat >/dev/null; echo present; "
            f"else mkdir -p {q(directory)} && cat > {q(path)}.tmp.$$ && mv {q(path)}.tmp.$$ {q(path)} && "
            f"[ \"$(sha256sum {q(path)} | cut -c1-64)\" = {q(sha256)} ] && echo written; fi")


def overlay_mount(launch: object) -> Mount | None:
    """The source overlay mount of a rank (a serve launch or a bundle rank), or None."""
    return next((mount for mount in getattr(launch, "mounts", ()) if mount.target == OVERLAY_TARGET), None)


def stage_container(plan: ServePlan, launch: RankLaunch, *, image: str | None = None) -> str:
    """Build the native library inside the serving image (the profile's unless given) and probe its imports.

    With a source overlay the container mounts it as serving does, first on
    ``PYTHONPATH``, so the probe reports the vLLM and B12X serving imports.
    """
    name = f"{CONTAINER_PREFIX}-{plan.run_id}-stage"
    overlay = overlay_mount(launch)
    mounts = ["--mount", q(f"type=bind,src={overlay.source},dst={OVERLAY_TARGET},readonly")] if overlay else []
    path = f"{OVERLAY_TARGET}:{SOURCE_TARGET}" if overlay else SOURCE_TARGET
    return " ".join([
        "mkdir -p", q(plan.build_dir), "&&",
        docker_command(launch.docker), "run", "--rm", "--name", q(name),
        "--label", q(f"{LABEL}={plan.run_id}"), "--label", q(f"{LABEL}.role=stage"),
        "--network", "none", "--pull", "never", "--entrypoint", "python3",
        "--mount", q(f"type=bind,src={plan.source_dir},dst={SOURCE_TARGET},readonly"),
        "--mount", q(f"type=bind,src={plan.build_dir},dst={BUILD_TARGET}"), *mounts,
        "--env", q(f"PYTHONPATH={path}"), "--env", q(f"SIRCL_BUILD_CACHE_DIR={BUILD_TARGET}"),
        q(image or plan.profile.image_id), "-m", "sparkring_sircl.vllm.serve.probe", "--build",
    ])


# The largest OVERLAY.json preflight reads back; a larger one is identified by its SHA-256 alone.
OVERLAY_RECORD_LIMIT = 65536


def overlay_facts(directory: str, sudo: str = "") -> str:
    """Whether ``directory`` is a source overlay: the directory, each required file, ``OVERLAY.json`` (its
    SHA-256 and base64 content) and any distribution metadata at its top level, read-only.

    Runs through the Spark's ``sudo`` prefix, like the model directory's checks.
    """
    record = f"{directory}/{OVERLAY_RECORD}"
    return privileged(sudo, "; ".join([
        f"if [ -d {q(directory)} ]; then echo \"overlay\tpresent\"; else echo \"overlay\tmissing\"; fi",
        f"for f in {' '.join(q(name) for name in OVERLAY_FILES)}; do if [ -f {q(directory)}/\"$f\" ]; then "
        "echo \"file:$f\tpresent\"; else echo \"file:$f\tmissing\"; fi; done",
        f"echo \"record_sha256\t$(sha256sum {q(record)} 2>/dev/null | cut -c1-64)\"",
        f"echo \"record\t$(head -c {OVERLAY_RECORD_LIMIT} {q(record)} 2>/dev/null | base64 -w0)\"",
        f"echo \"metadata\t$(cd {q(directory)} 2>/dev/null && ls -d -- *.dist-info *.egg-info 2>/dev/null | "
        "tr '\\n' ' ')\"",
    ]))


def staged_state(plan: ServePlan) -> str:
    """Whether this plan's package tree, native library and seccomp policy are on the Spark."""
    marker = f"{plan.source_dir}/{STAGED_MARKER}"
    library = f"{plan.build_dir}/{plan.library}"
    return "; ".join([
        f"if [ -f {q(marker)} ]; then echo \"source\tpresent\"; else echo \"source\tmissing\"; fi",
        f"if [ -f {q(library)} ]; then echo \"library\tpresent\"; else echo \"library\tmissing\"; fi",
        f"echo \"seccomp\t$(sha256sum {q(plan.seccomp_path)} 2>/dev/null | cut -c1-64 || true)\"",
    ])


def listening_ports(ports: Sequence[int]) -> str:
    """``port:<n><TAB>in use|free`` for each TCP port (listening sockets on any address)."""
    checks = []
    for port in ports:
        checks.append(f"if ss -Hltn {q(f'sport = :{int(port)}')} 2>/dev/null | grep -q .; then "
                      f"echo \"port:{int(port)}\tin use\"; else echo \"port:{int(port)}\tfree\"; fi")
    return "; ".join(checks)


def host_facts(plan: ServePlan, launch: RankLaunch, profile: ServingProfile) -> str:
    """Memory, the model directory's files and digests, and the cache directory, read-only.

    For the profile's checkpoint it reads the size of every file of the
    profile's manifest; for another checkpoint (``--checkpoint-id``) it counts
    the directory's files and bytes instead. Runs through the Spark's ``sudo``
    prefix, because checkpoint and cache directories such as
    ``/srv/sparkring/<cluster>`` may be readable by root only.
    """
    model = launch.mount("/models/target").source
    cache = launch.mount("/cache").source
    if plan.foreign_checkpoint:
        files = (f"find -L {q(model)} -type f -printf '%s\\n' 2>/dev/null | "
                 "awk '{n++; s+=$1} END {printf \"files\\t%d %.0f\\n\", n, s}'")
    else:
        names = " ".join(q(item.name) for item in profile.checkpoint_files)
        files = (f"for f in {names}; do p={q(model)}/\"$f\"; if [ -f \"$p\" ]; then "
                 "echo \"size:$f\t$(stat -L -c %s \"$p\")\"; else echo \"size:$f\tmissing\"; fi; done")
    return privileged(launch.sudo, "; ".join([
        "awk '/^MemTotal:|^MemAvailable:/ {sub(\":\", \"\", $1); print \"mem:\" $1 \"\\t\" $2}' /proc/meminfo",
        # The launcher sends its API requests with curl on rank 0's Spark.
        "echo \"curl\t$(command -v curl || echo missing)\"",
        f"if [ -d {q(model)} ]; then echo \"model\tpresent\"; else echo \"model\tmissing\"; fi",
        files,
        f"for f in config.json model.safetensors.index.json; do p={q(model)}/\"$f\"; "
        "echo \"sha256:$f\t$(sha256sum \"$p\" 2>/dev/null | cut -c1-64)\"; done",
        f"if [ -d {q(cache)} ]; then echo \"cache\tpresent\"; else echo \"cache\tabsent\"; fi",
        # Free space of the file system that holds (or will hold) the cache directory.
        f"d={q(cache)}; while [ ! -d \"$d\" ]; do d=$(dirname \"$d\"); done; "
        "echo \"space_kib\t$(df -Pk \"$d\" 2>/dev/null | awk 'NR==2 {print $4}')\"",
    ]))


def preview(plans: ServePlan | Sequence[ServePlan]) -> list[str]:
    """Every remote command of stage and start for ``--print``, one ``<ssh target>: <command>`` per line."""
    plans = [plans] if isinstance(plans, ServePlan) else list(plans)
    lines = []
    for plan in plans:
        for launch in plan.ranks:
            lines.append(f"{launch.ssh}: {stage_tree(plan.source_dir, plan.staged_digest)}  < staged tree")
            lines.append(f"{launch.ssh}: {write_once(plan.seccomp_path, plan.profile.seccomp_sha256)}"
                         "  < seccomp policy")
            lines.append(f"{launch.ssh}: {stage_container(plan, launch)}")
    for plan in plans:
        for launch in plan.ranks:
            lines.append(f"{launch.ssh}: {make_directories(launch.directories, launch.sudo)}")
            for table in plan.tuning.tables:
                command = privileged(launch.sudo, write_once(table.host_path(plan.run_dir), table.sha256))
                lines.append(f"{launch.ssh}: {command}  < tuning table {table.hash}")
            lines.append(f"{launch.ssh}: {launch.shell()}")
    return lines
