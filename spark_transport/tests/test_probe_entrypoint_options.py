"""Compile actual probe option parsers with protocol constants, without CUDA/RDMA."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_integer_overflow_diagnostic_names_the_option(tmp_path):
    command = compile_cpu(tmp_path, r'''
#include "probe_options.hpp"
int main() {
  for (const char* value : {"18446744073709551616", "65536"}) {
    try {
      spark_transport::probe::unsigned_value<std::uint16_t>(value, "port");
      return 2;
    } catch (const std::out_of_range& error) {
      if (std::string(error.what()) != "port exceeds its integer range") return 1;
    }
  }
  return 0;
}
''')
    assert subprocess.run(command, capture_output=True, timeout=5).returncode == 0


def linux_path(path):
    text = path.resolve().as_posix()
    return "/mnt/" + text[0].lower() + text[2:] if os.name == "nt" else text


def compile_cpu(tmp_path, source):
    path = tmp_path / "probe.cpp"
    path.write_text(source)
    binary = tmp_path / "probe"
    if os.name == "nt":
        if not shutil.which("wsl"):
            pytest.skip("C++ compiler unavailable")
        probe = subprocess.run(
            ["wsl", "--", "sh", "-c", "command -v g++"], capture_output=True
        )
        if probe.returncode:
            pytest.skip("WSL C++ compiler unavailable")
        prefix = ["wsl", "--"]
        compiler = "g++"
    else:
        prefix = []
        compiler = shutil.which("g++")
        if not compiler:
            pytest.skip("C++ compiler unavailable")
    command = prefix + [
        compiler,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        linux_path(ROOT / "app"),
        "-I",
        linux_path(ROOT / "include"),
        linux_path(path),
        "-o",
        linux_path(binary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return prefix + [linux_path(binary)]


@pytest.fixture(
    scope="module",
    params=[
        "tp4_tiled_prefill_probe.cu",
        "tp4_vocab_allgather_probe.cu",
        "tp4_vocab_graph_probe.cu",
        "transport_probe.cpp",
    ],
)
def parser(request, tmp_path_factory):
    name = request.param
    text = (ROOT / "app" / name).read_text()
    end = (
        "double now_microseconds()"
        if name == "transport_probe.cpp"
        else "void check_cuda("
    )
    options = text[text.index("namespace {") + len("namespace {") : text.index(end)]
    support = r"""#include "probe_options.hpp"
#include "spark_transport/gpu_doorbell.hpp"
#include "spark_transport/memory_buffer.hpp"
#include "spark_transport/gpu_tp4_vocab_allgather.hpp"
#include "spark_transport/tp4_tiled_session.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
namespace spark_transport {
MemoryKind parse_memory_kind(std::string_view value) {
 if (value == "cuda-managed") return MemoryKind::kCudaManaged;
 throw std::invalid_argument("test stub only supports cuda-managed");
}
std::size_t aligned_control_offset(std::size_t bytes) {
 constexpr auto alignment = alignof(DoorbellControl);
 return (bytes + alignment - 1) & ~(alignment - 1);
}
}
"""
    extra = ""
    branch = ""
    if name == "tp4_tiled_prefill_probe.cu":
        extra = text[
            text.index("std::string json_string(") : text.index(
                "std::string make_receipt("
            )
        ]
        branch = 'if(argc==2 && std::string(argv[1])=="--emit-json") { std::string text; for(int i=0;i<32;++i) text+=char(i); text+=char(34); text+=char(92); std::cout<<json_string(text); return 0; }'
    driver = (
        "int main(int argc,char**argv) { try { "
        + branch
        + " (void)parse_options(argc,argv); return 0; } catch(const std::exception&) {return 2;} }"
    )
    binary = compile_cpu(
        tmp_path_factory.mktemp("probe-options"), support + options + extra + driver
    )
    base = (
        ["--server", "--device", "hca"]
        if name == "transport_probe.cpp"
        else ["--rank", "0", "--peer0", "a", "--peer1", "b"]
    )
    if name == "tp4_tiled_prefill_probe.cu":
        base += [
            "--arm-id",
            "fixture",
            "--query-rows",
            "1",
            "--measured-operations",
            "1",
            "--receipt-schema",
            "sparkring-tp4-tiled-prefill-probe/v2",
            "--receipt-prefix",
            "TP4_TILED_PREFILL_RECEIPT",
        ]
    return name, binary, base


def test_valid_defaults_remain_accepted(parser):
    _, binary, base = parser
    assert (
        subprocess.run(binary + base, capture_output=True, timeout=10).returncode == 0
    )


def test_vocab_query_limit_matches_protocol(parser):
    name, binary, base = parser
    if name != "tp4_vocab_allgather_probe.cu":
        pytest.skip("Only vocabulary allgather accepts an explicit Q")
    for rows, expected in ((40, 0), (41, 2)):
        assert subprocess.run(
            binary + base + ["--q", str(rows)], capture_output=True, timeout=10
        ).returncode == expected


@pytest.mark.parametrize(
    "kind,value",
    [
        ("gid", "256"),
        ("gid", "-255"),
        ("port", "65536"),
        ("port", "0"),
        ("warmup", "4294967296"),
    ],
)
def test_entrypoint_rejects_invalid_values(parser, kind, value):
    name, binary, base = parser
    option = {
        "gid": "--gid" if name == "transport_probe.cpp" else "--gid0",
        "port": "--control-port"
        if name == "transport_probe.cpp"
        else "--control-port0",
        "warmup": "--warmup-operations"
        if name == "tp4_tiled_prefill_probe.cu"
        else "--warmup",
    }[kind]
    assert (
        subprocess.run(
            binary + base + [option, value], capture_output=True, timeout=10
        ).returncode
        == 2
    )


def test_tiled_json_escapes_every_control_character(parser):
    name, binary, _ = parser
    if name != "tp4_tiled_prefill_probe.cu":
        pytest.skip("Only tiled probe emits JSON receipts")
    result = subprocess.run(
        binary + ["--emit-json"], capture_output=True, text=True, timeout=10, check=True
    )
    assert json.loads(result.stdout) == "".join(chr(i) for i in range(32)) + chr(
        34
    ) + chr(92)


@pytest.fixture(scope="module")
def transport_verdict(tmp_path_factory):
    text = (ROOT / "app/transport_probe.cpp").read_text()
    start = text.rindex("    bool local_correct = true;")
    end = text.index("  } catch", start)
    tail = text[start:end]
    source = (
        r"""#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
namespace spark_transport {
enum class MemoryKind { kHost };
const char* memory_kind_name(MemoryKind) { return "fixture"; }
}
struct Options {bool server, gpu_verifier; int bytes; spark_transport::MemoryKind memory;};
struct Buffer {
 bool matches;
 bool verify_on_gpu(std::uint8_t,int) {return matches;}
 bool verify_on_cpu(std::uint8_t,int) {return matches;}
};
struct Channel {std::uint32_t remote; std::uint32_t exchange(std::uint32_t){return remote;}};
int main(int argc,char**argv) {
 if(argc!=4) return 2;
 try {
 Options options{std::string(argv[1])=="server",false,1,spark_transport::MemoryKind::kHost};
 Buffer storage{std::string(argv[2])=="true"}; auto* buffer=&storage;
 Channel channel{std::string(argv[3])=="true" ? 1U:0U};
 constexpr std::uint8_t pattern=0xa5;
"""
        + tail
        + '} catch(const std::exception& error) {return std::string(error.what()) == '
        '"data verification failed on at least one endpoint" ? 1 : 3;} }'
    )
    return compile_cpu(tmp_path_factory.mktemp("transport-verdict"), source)


@pytest.mark.parametrize(
    "role,local,remote,expected",
    [
        ("server", "false", "true", 1),
        ("server", "true", "true", 0),
        ("server", "true", "false", 1),
        ("client", "true", "false", 1),
        ("client", "true", "true", 0),
    ],
)
def test_transport_verdict_checks_both_endpoints(
    transport_verdict, role, local, remote, expected
):
    result = subprocess.run(
        transport_verdict + [role, local, remote], capture_output=True, timeout=10
    )
    assert result.returncode == expected
