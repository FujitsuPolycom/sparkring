"""Remote and local process access for the acceptance harness.

Every SSH command, HTTP request and benchmark subprocess goes through one of
these small objects so tests can replace them with fakes. None of them retries
on its own; callers decide which failures are transient.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")
# ssh reports its own connection failures with exit status 255.
SSH_FAILURE = 255


@dataclass(frozen=True)
class Result:
    returncode: int
    stdout: str
    stderr: str


class SshRunner:
    """Run a Bash script on one host through the local `ssh` client.

    The script travels on standard input to `bash -s`, so it needs no shell
    quoting beyond what the script itself does. BatchMode refuses password
    prompts: the host must accept the operator's key.
    """

    def __init__(self, target, *, ssh=("ssh", *SSH_OPTIONS)):
        if not target or target.startswith("-"):
            raise ValueError("SSH target must be a host or user@host")
        self.target = target
        self.ssh = tuple(ssh)

    def run(self, script, *, timeout):
        # Bytes, not text mode: text mode on Windows would send CRLF line
        # endings, which the remote Bash reads as part of each command.
        try:
            completed = subprocess.run([*self.ssh, self.target, "bash", "-s"], input=script.encode("utf-8"),
                                       capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Result(SSH_FAILURE, "", f"ssh {self.target}: no result within {timeout} s")
        except OSError as error:
            return Result(SSH_FAILURE, "", f"ssh {self.target}: {error}")
        return Result(completed.returncode, completed.stdout.decode("utf-8", "replace"),
                      completed.stderr.decode("utf-8", "replace"))


class HttpError(Exception):
    """An HTTP request that returned no usable JSON document."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class HttpClient:
    """JSON over HTTP with environment proxies and redirects disabled.

    The installer's API has no authentication, so requests carry no key.
    """

    def __init__(self):
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _send(self, request, timeout):
        try:
            with self._opener.open(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            body = error.read(300).decode("utf-8", "replace")
            raise HttpError(f"HTTP {error.code} from {request.full_url}: {body}") from None
        except (urllib.error.URLError, OSError) as error:
            raise HttpError(f"{request.full_url}: {getattr(error, 'reason', error)}") from None
        try:
            return json.loads(payload)
        except ValueError:
            raise HttpError(f"{request.full_url}: response is not JSON") from None

    def get_json(self, url, *, timeout):
        return self._send(urllib.request.Request(url), timeout)

    def post_json(self, url, body, *, timeout):
        request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        return self._send(request, timeout)


def run_process(argv, *, log_path, cwd=None, timeout=None):
    """Run a local program with standard output and error appended to one log file."""
    with Path(log_path).open("ab") as log:
        try:
            return subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nNo exit within {timeout} s; the process was stopped.\n".encode())
            return -1
