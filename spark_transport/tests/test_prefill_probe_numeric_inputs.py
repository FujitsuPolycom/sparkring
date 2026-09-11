"""Compile probe numeric parsers without CUDA or RDMA headers/libraries."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", params=["tp4_bidirectional_prefill_probe.cu", "tp4_fused_prefill_probe.cu"])
def parser_binary(request, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("probe-parser")
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler required for standalone numeric parser")
    text = (ROOT / "app" / request.param).read_text()
    helper = text.split("template <typename Integer>", 1)[1].split("Options parse_options", 1)[0]
    source = tmp_path / "parser.cpp"
    source.write_text("#include <algorithm>\n#include <cstdint>\n#include <limits>\n"
                      "#include <string>\n#include <stdexcept>\n#include <iostream>\n"
                      "template <typename Integer>" + helper + r'''
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
                    str(source), "-o", str(binary)], check=True, capture_output=True)
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
