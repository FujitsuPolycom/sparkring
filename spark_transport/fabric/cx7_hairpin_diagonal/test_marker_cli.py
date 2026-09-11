"""CPU-only CLI and signal tests; no RDMA device is opened or modified."""

import os
from pathlib import Path
import shutil
import signal
import subprocess

import pytest


SOURCE = Path(__file__).parent / "native" / "mlx5_rdma_tx_rewrite_probe.c"
BASE = ["--device", "test-device", "--source-port", "65535",
        "--replacement-ethertype", "0x88b5"]


@pytest.fixture(scope="module")
def parser_binary(tmp_path_factory):
    """Compile the actual parser and waits without linking any RDMA code."""
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None or os.name == "nt":
        pytest.skip("POSIX C compiler required for native parser tests")
    prefix = SOURCE.read_text().split("static struct ibv_context *open_context", 1)[0]
    prefix = "\n".join(line for line in prefix.splitlines()
                       if not line.startswith("#include <infiniband/"))
    harness = r'''
int main(int argc, char **argv) {
    struct options options;
    int rc = parse_options(argc, argv, &options);
    if (rc != 0) return rc > 0 ? 0 : 1;
    if (getenv("MARKER_TEST_WAIT") != NULL) {
        if (install_stop_handlers() != 0) return 2;
        puts("ready");
        fflush(stdout);
        return options.managed ? wait_until_stop()
                               : wait_until_timeout_or_stop(options.run_seconds);
    }
    printf("managed=%d seconds=%u attach=%d\n", options.managed,
           options.run_seconds, options.attach);
    return 0;
}
'''
    directory = tmp_path_factory.mktemp("marker-parser")
    source = directory / "parser.c"
    source.write_text(prefix + harness)
    binary = directory / "parser"
    subprocess.run([compiler, "-std=c11", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(binary)], check=True, capture_output=True)
    return binary


@pytest.mark.parametrize("args, expected", [
    (["--attach", "--managed"], "managed=1 seconds=0 attach=1"),
    (["--attach", "--run-seconds", "1"], "managed=0 seconds=1 attach=1"),
    (["--attach", "--run-seconds", "7200"], "managed=0 seconds=7200 attach=1"),
    ([], "managed=0 seconds=0 attach=0"),
])
def test_valid_modes(parser_binary, args, expected):
    result = subprocess.run([str(parser_binary), *BASE, *args],
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("args, message", [
    (["--managed"], "valid only with --attach"),
    (["--attach"], "requires --run-seconds or --managed"),
    (["--run-seconds", "1"], "valid only with --attach"),
    (["--attach", "--managed", "--run-seconds", "1"], "mutually exclusive"),
    (["--attach", "--run-seconds", "1", "--managed"], "mutually exclusive"),
    (["--attach", "--run-seconds", "0"], "must be from 1 to 7200"),
    (["--attach", "--run-seconds", "7201"], "must be from 1 to 7200"),
    (["--attach", "--run-seconds", "-1"], "must be from 1 to 7200"),
    (["--attach", "--run-seconds", "inf"], "must be from 1 to 7200"),
])
def test_invalid_modes(parser_binary, args, message):
    result = subprocess.run([str(parser_binary), *BASE, *args],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert message in result.stderr


def test_help(parser_binary):
    result = subprocess.run([str(parser_binary), "--help"],
                            capture_output=True, text=True, check=True)
    assert "--managed" in result.stderr
    assert "--run-seconds" in result.stderr


def test_full_binary_rejects_modes_before_device_open():
    binary = os.environ.get("MARKER_TEST_BINARY")
    if binary is None:
        pytest.skip("MARKER_TEST_BINARY must name a compiled native helper")
    for args in (["--managed"], ["--attach"],
                 ["--attach", "--managed", "--run-seconds", "1"],
                 ["--attach", "--run-seconds", "7201"]):
        result = subprocess.run([binary, *BASE, *args],
                                capture_output=True, text=True, timeout=3)
        assert result.returncode == 1
        assert "cannot open RDMA device" not in result.stderr
        assert "cannot enumerate RDMA devices" not in result.stderr
        assert "--" in result.stderr


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_managed_wait_exits_on_signal(parser_binary, stop_signal):
    process = subprocess.Popen([str(parser_binary), *BASE, "--attach", "--managed"],
                               stdout=subprocess.PIPE, text=True,
                               env={**os.environ, "MARKER_TEST_WAIT": "1"})
    try:
        assert process.stdout.readline().strip() == "ready"
        # A managed attachment must not return because its run_seconds is zero.
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.2)
        process.send_signal(stop_signal)
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)


def test_bounded_wait_expires(parser_binary):
    result = subprocess.run([str(parser_binary), *BASE, "--attach", "--run-seconds", "1"],
                            env={**os.environ, "MARKER_TEST_WAIT": "1"},
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 0
    assert result.stdout.strip() == "ready"
