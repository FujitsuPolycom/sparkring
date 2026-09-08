"""Verify packaged startup modules agree with their installed liveness contract."""

import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepared_liveness_is_installed_attested_and_compatible(tmp_path, monkeypatch):
    prepare = load("liveness_prepare", HERE / "prepare.py")
    native = tmp_path / "native.so"
    native.write_bytes(b"offline-native-fixture")
    digest = hashlib.sha256(native.read_bytes()).hexdigest()
    monkeypatch.setattr(prepare, "PLACEMENT", digest)
    monkeypatch.setattr(prepare, "TRANSPORT", digest)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        content = b"# fixture\n"
        member = tarfile.TarInfo("sparkcache/__init__.py")
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))

    def checkout(command, **kwargs):
        if "rev-parse" in command:
            return prepare.CACHE_COMMIT + "\n"
        if "status" in command:
            return b""
        assert "archive" in command
        return archive.getvalue()

    def bundle(command, **kwargs):
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir()
        (destination / "native.so").write_bytes(native.read_bytes())

    # External cache/native inputs are fixtures; preparation and startup source
    # selection execute unchanged, without a registry, GPU, or Docker daemon.
    monkeypatch.setattr(prepare.subprocess, "check_output", checkout)
    with monkeypatch.context() as context:
        context.setattr(prepare.subprocess, "run", bundle)
        output = tmp_path / "context"
        receipt = prepare.prepare(tmp_path / "cache", native, native, output)
    helper = output / "startup/scheduler_liveness.py"
    assert receipt["files"]["startup/scheduler_liveness.py"] == hashlib.sha256(helper.read_bytes()).hexdigest()

    installation = tmp_path / "installed"
    installation.mkdir()

    def installed_path(value):
        absolute = Path(value)
        if str(value).startswith("/opt/sparkring/bin"):
            return installation / absolute.name if absolute.name != "bin" else installation
        # The second inventory entry is a native placement dependency.
        target = tmp_path / "other" / absolute.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"fixture")
        return target

    tree = ast.parse((HERE / "install.py").read_text())
    copies = next(node for node in tree.body if isinstance(node, ast.For)
                  and ast.unparse(node.target) == "(name, target)")
    inventory = next(node for node in tree.body if isinstance(node, ast.For)
                     and ast.unparse(node.target) == "path" and isinstance(node.iter, ast.Tuple))
    namespace = dict(SOURCE=output, Path=installed_path, shutil=shutil, hashlib=hashlib, files={})
    exec(compile(ast.Module(body=[copies, inventory], type_ignores=[]), "installed-startup", "exec"), namespace)
    installed = installation / "scheduler_liveness.py"
    assert installed.read_bytes() == helper.read_bytes()
    assert namespace["files"][str(installed)] == hashlib.sha256(installed.read_bytes()).hexdigest()

    # A subprocess prevents module import caches from hiding an absent or stale
    # sibling. The actual installed wrapper forwards the configured timeout.
    program = '''
import importlib.util
import os
import sys
sys.path.insert(0, sys.argv[1])
import scheduler_liveness
spec = importlib.util.spec_from_file_location("wrapper", sys.argv[1] + "/serve-with-warmup.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)
assert wrapper.scheduler_liveness is scheduler_liveness
observed = {}
scheduler_liveness.start_liveness_service = lambda **kwargs: observed.update(kwargs)
os.environ["SPARKRING_LIVENESS_ENABLED"] = "1"
os.environ["SPARKRING_LIVENESS_OUTPUT_SECONDS"] = "901"
wrapper.start_rank_liveness(rank=0, endpoint="http://localhost:8015", credential="test-key")
assert observed["output_timeout_seconds"] == 901
assert observed["credential"] == "test-key"
'''
    subprocess.run([sys.executable, "-S", "-B", "-c", program, str(installation)], check=True)

    module = load("packaged_liveness", installed)
    clock = [0.0]
    monitor = module.SchedulerLiveness(blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, output_timeout_seconds=300, clock=lambda: clock[0])
    metrics = ('vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n'
               'vllm:kv_cache_usage_perc {usage}\nvllm:iteration_tokens_total_count 7\n')
    monitor.observe(metrics.format(usage=0.1))
    clock[0] = 301
    monitor.observe(metrics.format(usage=0.2))
    snapshot = monitor.snapshot()
    assert snapshot["output_iterations"] == 7
    assert snapshot["output_stalled_seconds"] == 301
    assert snapshot["reason"] == "engine_output_stall"
    assert snapshot["healthy"] is False
    # Growing allocated KV is not treated as verified execution progress.
    assert snapshot["kv_cache_usage"] == 0.2
    assert json.loads((output / "context.json").read_text())["files"] == receipt["files"]
