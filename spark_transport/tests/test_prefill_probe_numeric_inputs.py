"""Compile probe numeric parsers without CUDA or RDMA headers/libraries."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def parser_binary(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("probe-parser")
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler required for standalone numeric parser")
    source = tmp_path / "parser.cpp"
    source.write_text('#include "probe_options.hpp"\n#include <iostream>\n'
                      'using spark_transport::probe::unsigned_value;\n' + r'''
int main(int argc, char** argv) {
  if (argc != 3) return 3;
  try {
    std::uint64_t value;
    if (std::string(argv[1]) == "8") value = unsigned_value<std::uint8_t>(argv[2], "gid");
    else if (std::string(argv[1]) == "16") value = unsigned_value<std::uint16_t>(argv[2], "port");
    else value = unsigned_value<std::uint32_t>(argv[2], "count");
    std::cout << value;
    return 0;
  } catch (const std::exception&) { return 2; }
}
''')
    binary = tmp_path / "parser"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "app"), str(source), "-o", str(binary)], check=True, capture_output=True)
    return binary


@pytest.mark.parametrize("width,value,accepted", [
    (8, "255", True), (8, "256", False),
    (16, "65535", True), (16, "65537", False),
    (32, "4294967295", True), (32, "4294967296", False),
    (32, "-4294967296", False), (32, "+1", False),
    (32, " 1", False), (32, "1junk", False),
])
def test_no_truncation_or_signed_unsigned_input(parser_binary, width, value, accepted):
    result = subprocess.run([str(parser_binary), str(width), value], capture_output=True, text=True)
    assert result.returncode == (0 if accepted else 2)
    if accepted:
        assert result.stdout == value


def test_tensor_probe_retires_both_streams_before_reading_shared_counter():
    source = (ROOT / "app/tp4_tensor_probe.cu").read_text()
    first = source.index('cudaStreamSynchronize(stream0)')
    second = source.index('cudaStreamSynchronize(stream1)')
    readback = source.index('cudaMemcpy(&host_mismatches')
    assert first < second < readback
    assert 'if (stream1 != nullptr)' in source[first:second]
