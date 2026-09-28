"""Run the one-line installer on Node A so that an SSH disconnect cannot stop it.

The installer runs detached (`nohup setsid`) in its own directory under the
SSH user's home, `~/sparkring-acceptance/<run>/`, writing `stdout.json`,
`stderr.log` and, when it exits, `exit_code`. The harness polls for
`exit_code` and then copies the two outputs back. A run directory that already
exists is never reused, and nothing on Node A is removed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
import shlex

PUBLISHED_SCRIPT = "https://raw.githubusercontent.com/FujitsuPolycom/sparkring/{ref}/install.sh"
RESULT_SCHEMA = "sparkring-install-result/v1"
REMOTE_ROOT = "sparkring-acceptance"
REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
HEREDOC = "SPARKRING_ACCEPTANCE_COMMAND"
SOURCE_REVISION = re.compile(r"^Source revision: ([0-9a-f]{40})\s*$", re.M)
API_READINESS = re.compile(r"^Done: Node 0: Wait for API readiness \(([0-9.]+)s\)\s*$", re.M)


class InstallError(Exception):
    """The installer could not be started, followed or read."""


@dataclass(frozen=True)
class Source:
    """Where install.sh comes from: the published script at a ref, or a Git bundle on Node A."""
    kind: str
    ref: str
    bundle: str | None = None

    def describe(self):
        if self.kind == "bundle":
            return f"a Git bundle on Node A, ref `{self.ref}`"
        return f"the published one-line command at `{self.ref}`"


def parse_source(value):
    """Parse `published`, `published:REF` or `bundle:PATH:REF`."""
    if value == "published":
        return Source("published", "main")
    kind, _, rest = value.partition(":")
    if kind == "published" and REF.fullmatch(rest):
        return Source("published", rest)
    if kind == "bundle":
        # Git forbids ':' in ref names, so the last ':' ends the path.
        path, _, ref = rest.rpartition(":")
        if path and REF.fullmatch(ref):
            _remote_path(path)
            return Source("bundle", ref, path)
    raise ValueError("--install takes published, published:REF or bundle:PATH_ON_NODE_A:REF")


def _remote_path(path):
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    if path.startswith("/"):
        return shlex.quote(path)
    raise ValueError("The bundle path on Node A must be absolute or start with ~/")


def install_command(source, profile, extra=()):
    """The shell command a person would type on Node A for this source.

    ``extra`` holds further `sparkring install` arguments, for example
    ``--image-lock PATH`` for a development image lock on Node A.
    """
    flags = " ".join([f"--profile {shlex.quote(profile)} --yes --json", *(shlex.quote(arg) for arg in extra)])
    if source.kind == "published":
        url = PUBLISHED_SCRIPT.format(ref=source.ref)
        ref = "" if source.ref == "main" else f"--ref {shlex.quote(source.ref)} "
        return f"curl -fsSL {shlex.quote(url)} | bash -s -- {ref}{flags}"
    bundle, ref = _remote_path(source.bundle), shlex.quote(source.ref)
    return f"git clone -q --branch {ref} {bundle} src && bash src/install.sh --repository {bundle} --ref {ref} {flags}"


def _directory(run_id):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError(f"Invalid run ID: {run_id!r}")
    return f'"$HOME"/{REMOTE_ROOT}/{run_id}'


def launch_script(run_id, command):
    return f"""set -eu
dir={_directory(run_id)}
if [ -e "$dir" ]; then echo "$dir already exists; start another run" >&2; exit 17; fi
mkdir -p "$dir"
cd "$dir"
cat > command.sh <<'{HEREDOC}'
set -o pipefail
{command}
{HEREDOC}
nohup setsid bash -c 'bash command.sh > stdout.json 2> stderr.log; echo $? > exit_code.part; mv exit_code.part exit_code' < /dev/null > /dev/null 2>&1 &
echo launched
"""


def poll_script(run_id):
    return f"""dir={_directory(run_id)}
if [ ! -d "$dir" ]; then echo missing; exit 0; fi
if [ -f "$dir/exit_code" ]; then echo "exit $(cat "$dir/exit_code")"; else echo running; fi
tail -n 1 "$dir/stderr.log" 2>/dev/null || true
"""


def read_script(run_id, name):
    return f'cat {_directory(run_id)}/{shlex.quote(name)}\n'


def launch(ssh, run_id, command, *, timeout=120):
    result = ssh.run(launch_script(run_id, command), timeout=timeout)
    if result.returncode != 0 or "launched" not in result.stdout:
        raise InstallError(f"Could not start the installer on Node A (exit {result.returncode}): {result.stderr.strip()}")


def wait(ssh, run_id, *, timeout, interval, clock, sleep, log):
    """Poll until the installer exits; return its exit status.

    Failed SSH connections are retried until the deadline: the installer
    keeps running on Node A regardless.
    """
    deadline = clock() + timeout
    last = None
    while True:
        result = ssh.run(poll_script(run_id), timeout=60)
        lines = result.stdout.splitlines()
        if result.returncode == 0 and lines:
            status, progress = lines[0].strip(), (lines[1].strip() if len(lines) > 1 else "")
            if status == "missing":
                raise InstallError(f"Node A has no run directory for {run_id}; the launch did not happen. "
                                   "Start a separate installation with --redo install.")
            if status.startswith("exit "):
                try:
                    return int(status.split()[1])
                except (IndexError, ValueError):
                    raise InstallError(f"Unreadable installer exit status: {status!r}") from None
            if progress and progress != last:
                log(f"install: {progress}")
                last = progress
        else:
            log(f"install: Node A unreachable (ssh exit {result.returncode}); still polling")
        if clock() >= deadline:
            raise InstallError(f"The installer had not exited after {timeout} s; it may still be running on Node A. "
                               "Run the harness again to keep waiting.")
        sleep(interval)


def fetch(ssh, run_id, name):
    result = ssh.run(read_script(run_id, name), timeout=120)
    if result.returncode != 0:
        raise InstallError(f"Could not read {name} from Node A (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout


def result_document(stdout):
    """Return (document, whether standard output held nothing else)."""
    text = stdout.strip()
    try:
        document, only = json.loads(text), True
    except ValueError:
        document, only = None, False
        decoder = json.JSONDecoder()
        for match in re.finditer(r"^\{", text, re.M):
            try:
                candidate = decoder.raw_decode(text, match.start())[0]
            except ValueError:
                continue
            if isinstance(candidate, dict) and candidate.get("schema") == RESULT_SCHEMA:
                document = candidate
    if not isinstance(document, dict) or document.get("schema") != RESULT_SCHEMA:
        raise InstallError(f"The installer's standard output holds no {RESULT_SCHEMA} document")
    return document, only


def image_release(document):
    """The installer image release the installation verified, or None.

    `sparkring install --image-lock` can install an image other than the one
    the profile's release selects, so the record names this one.
    """
    verification = (document.get("transaction") or {}).get("verification") or {}
    return verification.get("image_release")


def summarize(stdout, stderr, exit_code, *, profile):
    """Condense the installer's outputs into the fields the record uses.

    The summary omits the result's API URL, deployment path and log path,
    which name Node A's address and directories.
    """
    document, only = result_document(stdout)
    state = document.get("state")
    revision = SOURCE_REVISION.findall(stderr)
    readiness = API_READINESS.findall(stderr)
    summary = {
        "state": state, "exit_code": exit_code, "only_document": only,
        "message": document.get("message"), "stage": document.get("stage"), "field": document.get("field"),
        "profile": document.get("profile"), "image_id": document.get("image_id"),
        "image_release": image_release(document), "nodes": document.get("nodes"),
        "source_revision": revision[-1] if revision else None,
        "api_ready_seconds": float(readiness[-1]) if readiness else None,
    }
    problems = []
    if state != "complete":
        problems.append(f"installer state {state}" + (f": {summary['message']}" if summary["message"] else ""))
    if exit_code != 0:
        problems.append(f"exit status {exit_code}")
    if summary["profile"] not in (None, profile):
        problems.append(f"installed profile {summary['profile']}, requested {profile}")
    summary["ok"] = not problems
    summary["problems"] = problems
    return summary


def phases(stderr):
    """The `sparkring install` progress lines, without install.sh's build and apt output."""
    lines = stderr.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("SparkRing install."):
            return "\n".join(lines[index:]) + "\n"
    return stderr if stderr.endswith("\n") or not stderr else stderr + "\n"
