"""A stock vLLM image with libsircl as its NCCL: the probe, the preflight and each rank's container; offline.

The image is represented by its ``docker image inspect`` record and the
probe's facts; a small ELF file built here stands in for libraries.
"""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import yaml

from runtime.common import libsircl, stock_image, stock_image_probe
from runtime.common.test_transport import document

LIBRARY = "/var/lib/sparkring/libsircl/" + "a" * 64 + "/libsircl.so.0.6.0"
TARGET = "/opt/sparkring/libsircl/lib/libsircl.so.0.6.0"
REFERENCE = "ghcr.io/spark-arena/dgx-vllm-eugr-nightly@sha256:" + "5" * 64
# The functions PyNccl's NCCLLibrary binds in Local Inference Lab's integration/karmic-kraken-beta at 57a80980bb.
FUNCTIONS = ["ncclAllGather", "ncclAllReduce", "ncclBroadcast", "ncclCommAbort", "ncclCommDestroy",
             "ncclCommInitRank", "ncclCommQueryProperties", "ncclCommResume", "ncclCommSuspend",
             "ncclCommWindowDeregister", "ncclCommWindowRegister", "ncclGetErrorString", "ncclGetUniqueId",
             "ncclGetVersion", "ncclGroupEnd", "ncclGroupStart", "ncclRecv", "ncclReduce", "ncclReduceScatter",
             "ncclSend"]


def build_elf(path, *, needed=("libc.so.6",), soname="libnccl.so.2", versions=("GLIBC_2.17", "GLIBC_2.34"),
              extra=b""):
    """A minimal ELF64 shared object with .dynstr, .dynamic, .gnu.version_r and .shstrtab sections."""
    strings, offsets = b"\0", {}
    for name in (*needed, *([soname] if soname else []), *versions):
        offsets[name] = len(strings)
        strings += name.encode() + b"\0"
    dynamic = b"".join(struct.pack("<qQ", 1, offsets[name]) for name in needed)
    if soname:
        dynamic += struct.pack("<qQ", 14, offsets[soname])
    dynamic += struct.pack("<qQ", 0, 0)
    verneed = struct.pack("<HHIII", 1, len(versions), offsets[needed[0]], 16, 0)
    for index, name in enumerate(versions):
        verneed += struct.pack("<IHHII", 0, 0, 2 + index, offsets[name], 16 if index + 1 < len(versions) else 0)
    names = b"\0.dynstr\0.dynamic\0.gnu.version_r\0.shstrtab\0"
    body, sections = b"", []
    for name, data in ((b".dynstr", strings), (b".dynamic", dynamic), (b".gnu.version_r", verneed),
                       (b".shstrtab", names)):
        sections.append((names.index(name + b"\0"), 64 + len(body), len(data)))
        body += data
    body += extra
    shoff = 64 + len(body)
    header = (b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)
              + struct.pack("<HHIQQQIHHHHHH", 3, 183, 1, 0, 0, shoff, 0, 64, 0, 0, 64, len(sections) + 1,
                            len(sections)))
    table = struct.pack("<IIQQQQIIQQ", *([0] * 10))
    for name, offset, size in sections:
        table += struct.pack("<IIQQQQIIQQ", name, 1, 0, 0, offset, size, 0, 0, 1, 0)
    Path(path).write_bytes(header + body + table)
    return path


def image(entrypoint=("/opt/nvidia/nvidia_entrypoint.sh",), preload="/usr/lib/aarch64-linux-gnu/libnccl.so.2"):
    environment = ["PATH=/usr/local/bin:/usr/bin", "CUDA_VERSION=13.0.2"] + ([f"LD_PRELOAD={preload}"] if preload else [])
    return {"Id": "sha256:" + "5" * 64, "Architecture": "arm64",
            "Config": {"Entrypoint": list(entrypoint), "Env": environment}}


def facts(**changes):
    """The probe's facts of a stock image that libsircl can run on."""
    value = {"machine": "aarch64", "python": "3.12.3", "glibc": "2.39",
             "library": {"path": TARGET, "soname": "libnccl.so.2", "needed": ["libc.so.6"], "glibc_required": "2.34",
                         "fail_stop": True, "load": "ok", "nccl_version": 22705,
                         "identity": {"library": "libsircl", "version": "0.6.0"}, "missing": []},
             "cuda": {"driver_api": 13000, "compute_capabilities": ["12.1"]},
             "vllm": {"version": "0.30.1rc1.dev723+g6bbad6acd", "root": "/usr/local/lib/python3.12/dist-packages/vllm",
                      "command": "/usr/local/bin/vllm", "nccl_so_path": True, "find_reads_nccl_so_path": True,
                      "pynccl_functions": list(FUNCTIONS), "multinode": True,
                      "switches": ["VLLM_DISABLE_PYNCCL", "VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_USE_NCCL_SYMM_MEM",
                                   "VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC"],
                      "fusions": ["fuse_allreduce_rms", "fuse_gemm_comms", "enable_sp"], "general_plugins": []},
             "torch": {"version": "2.13.0+cu130", "nccl": "dynamic"}, "loader": {"fastsafetensors": False}}
    for path, setting in changes.items():
        target = value
        *parents, last = path.split(".")
        for key in parents:
            target = target[key]
        target[last] = setting
    return value


def binding(**changes):
    return {"vllm_nccl_so_path": TARGET, "pynccl_provider": TARGET, "global_provider": TARGET,
            "torch_nccl_version": [2, 27, 5], "mapped": [TARGET], **changes}


# The probe.

def test_the_elf_reader_finds_the_soname_dependencies_and_newest_glibc_version(tmp_path):
    path = build_elf(tmp_path / "libsircl.so.0.6.0", needed=("libc.so.6", "libdl.so.2"),
                     versions=("GLIBC_2.17", "GLIBC_2.34", "GLIBC_2.28"))
    assert stock_image_probe.elf_facts(path) == {"soname": "libnccl.so.2", "needed": ["libc.so.6", "libdl.so.2"],
                                                 "glibc_required": "2.34"}
    plain = build_elf(tmp_path / "libtorch_cuda.so", soname=None, needed=("libnccl.so.2",), versions=())
    assert stock_image_probe.elf_facts(plain) == {"soname": None, "needed": ["libnccl.so.2"], "glibc_required": None}
    (tmp_path / "text").write_text("not elf")
    with pytest.raises(ValueError, match="not a 64-bit"):
        stock_image_probe.elf_facts(tmp_path / "text")


def test_the_probe_reads_pyncclss_bound_functions_and_the_library_selector_from_vllms_sources(tmp_path, monkeypatch):
    root = tmp_path / "site" / "vllm"
    for relative, text in (
            ("__init__.py", "raise RuntimeError('the probe never imports vLLM')\n"),
            ("envs.py", 'environment_variables = {"VLLM_NCCL_SO_PATH": None, "VLLM_USE_NCCL_SYMM_MEM": None}\n'),
            ("utils/nccl.py", "def find_nccl_library():\n    return envs.VLLM_NCCL_SO_PATH\n"),
            ("distributed/device_communicators/pynccl_wrapper.py",
             'class NCCLLibrary:\n    exported_functions = [Function("ncclAllReduce", 0, []), '
             'Function("ncclCommSuspend", 0, [])]\n'),
            ("engine/arg_utils.py", "nnodes node_rank master_addr master_port\n"),
            ("entrypoints/cli/serve.py", "headless\n"),
            ("config/compilation.py", "class PassConfig:\n    fuse_allreduce_rms: bool = False\n    enable_sp: bool\n")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    found = stock_image_probe.vllm_facts()
    assert found["nccl_so_path"] and found["find_reads_nccl_so_path"] and found["multinode"]
    assert found["pynccl_functions"] == ["ncclAllReduce", "ncclCommSuspend"]
    assert found["switches"] == ["VLLM_USE_NCCL_SYMM_MEM"] and found["fusions"] == ["fuse_allreduce_rms", "enable_sp"]


def test_the_probe_runs_with_the_standard_library_alone_and_prints_one_record(tmp_path):
    library = build_elf(tmp_path / "libsircl.so.0.6.0", extra=b"LIBSIRCL_FAIL_STOP\0")
    done = subprocess.run([sys.executable, "-I", "-", str(library)], input=stock_image.PROBE_SOURCE,
                          capture_output=True, text=True, check=True, cwd=tmp_path)
    record = stock_image.parse(done.stdout)
    assert record["library"]["soname"] == "libnccl.so.2" and record["library"]["fail_stop"] is True
    # A stand-in library does not load; the preflight names the loader's error.
    assert record["library"]["load"] != "ok"


# The preflight.

def test_an_image_libsircl_can_run_passes_the_preflight():
    assert stock_image.preflight(image(), facts(), ["--served-model-name", "m"]) == []
    assert stock_image.binding_problems(binding(), "0.6.0") == []


@pytest.mark.parametrize("changes, arguments, message", [
    ({"machine": "x86_64"}, [], "not arm64/aarch64"),
    ({"glibc": "2.31"}, [], "needs glibc 2.34, and the image has 2.31"),
    ({"library.load": "GLIBC_2.34 not found"}, [], "does not load in the image: GLIBC_2.34 not found"),
    ({"library.fail_stop": False}, [], "no fail-stop mode"),
    ({"library.nccl_version": 23203}, [], "NCCL API level 23203"),
    ({"library.identity": {"library": "nccl"}}, [], "does not identify itself as libsircl"),
    ({"library.missing": ["ncclCommShrink"]}, [], "binds ncclCommShrink, which the host libsircl does not export"),
    ({"cuda": {"driver_api": 12090, "compute_capabilities": ["12.1"]}}, [], "CUDA driver API is 12090"),
    ({"cuda": {"error": "libcuda.so.1: cannot open"}}, [], "finds no CUDA driver"),
    ({"cuda.compute_capabilities": ["9.0"]}, [], "compute capability 9.0"),
    ({"vllm.nccl_so_path": False}, [], "does not select PyNccl's library with VLLM_NCCL_SO_PATH"),
    ({"vllm.pynccl_functions": [*FUNCTIONS, "ncclRedOpCreatePreMulSum"]}, [], "binds ncclRedOpCreatePreMulSum, which "
                                                                             "libsircl exports only as a refusal"),
    ({"vllm.multinode": False}, [], "lacks --nnodes"),
    ({"torch.nccl": "static"}, [], "torch NCCL is static"),
    ({}, ["--tensor-parallel-size", "4"], "the vLLM arguments set --tensor-parallel-size"),
    ({}, ["--enable-sleep-mode"], "ncclCommSuspend and ncclCommResume"),
    ({}, ["--pipeline-parallel-size=4"], "pipeline parallelism 4"),
    ({}, ["--enable-expert-parallel"], "expert parallelism"),
    ({}, ["--enable-eplb"], "enable-eplb"),
    ({}, ["--compilation-config", '{"pass_config": {"fuse_allreduce_rms": true}}'], "fuse_allreduce_rms"),
])
def test_an_image_or_arguments_libsircl_cannot_serve_are_refused_with_the_reason(changes, arguments, message):
    problems = stock_image.preflight(image(), facts(**copy.deepcopy(changes)), arguments)
    assert any(message in problem for problem in problems), problems


def test_the_architecture_of_the_image_record_counts_too():
    assert "not arm64/aarch64" in stock_image.preflight(dict(image(), Architecture="amd64"), facts(), [])[0]


@pytest.mark.parametrize("changes, message", [
    ({"pynccl_provider": "/usr/lib/aarch64-linux-gnu/libnccl.so.2"}, "PyNccl's load"),
    ({"global_provider": "/usr/lib/aarch64-linux-gnu/libnccl.so.2"}, "global scope"),
    ({"torch_nccl_version": [2, 29, 7]}, "torch reports NCCL [2, 29, 7]"),
    ({"mapped": [TARGET, "/usr/lib/aarch64-linux-gnu/libnccl.so.2"]}, "not libsircl alone"),
])
def test_a_caller_bound_to_another_nccl_is_refused(changes, message):
    problems = stock_image.binding_problems(binding(**changes), "0.6.0")
    assert len(problems) == 1 and message in problems[0]


# Each rank's container.

def test_libsircl_comes_first_in_ld_preload_and_other_nccl_preloads_are_dropped():
    value = image(preload="/usr/lib/aarch64-linux-gnu/libnccl.so.2:/usr/lib/libjemalloc.so.2")
    assert stock_image.preloads(value, "0.6.0") == f"{TARGET}:/usr/lib/libjemalloc.so.2"
    assert stock_image.preloads(image(preload=None), "0.6.0") == TARGET


@pytest.mark.parametrize("entrypoint, expected", [
    (("/opt/nvidia/nvidia_entrypoint.sh",), ["vllm", "serve", "/model"]),
    ((), ["vllm", "serve", "/model"]),
    (("vllm", "serve"), ["/model"]),
    (("/usr/local/bin/vllm",), ["serve", "/model"]),
])
def test_the_command_runs_vllm_serve_after_the_images_own_entrypoint(entrypoint, expected):
    assert stock_image.serve_command(entrypoint, ["/model"]) == expected


def specs(shape, size, positions, *, value=None, found=None, arguments=()):
    value, found = value or image(), found or facts()
    group, devices, routes, _ = libsircl.group(document(shape, size), positions)
    rows = [{"host": f"spark{position}", "fabric_ip": f"192.0.2.{20 + position}", "interface": "enP7s7"}
            for position in positions]
    return [stock_image.rank_spec(value, found, list(arguments), name="sr-stock-0123456789ab", reference=value["Id"],
                                  rank=rank, row=row, master=rows[0]["fabric_ip"], size=len(rows),
                                  routes=routes[rank], host_library=LIBRARY, version="0.6.0",
                                  model_path="/srv/models/m", directory="/var/lib/sparkring/stock/sr-stock-0123456789ab")
            for rank, row in enumerate(rows)], routes


@pytest.mark.parametrize("shape, size, positions", [("pair", 2, [0, 1]), ("cycle", 8, list(range(8)))])
def test_every_rank_runs_the_image_with_libsircl_as_the_nccl_of_pynccl_and_torch(shape, size, positions):
    ranks, routes = specs(shape, size, positions, arguments=["--served-model-name", "m"])
    assert len(ranks) == size
    for rank, spec in enumerate(ranks):
        environment = spec.environment
        assert environment["VLLM_NCCL_SO_PATH"] == TARGET and environment["LD_PRELOAD"] == TARGET
        assert environment["LIBSIRCL_FAIL_STOP"] == "1" and environment["LIBSIRCL_NCCL_API_VERSION"] == "22705"
        assert environment["LIBSIRCL_TRANSPORT"] == "verbs" and environment["SIRCL_GID_INDEX"] == "3"
        assert {key: environment[key] for key in libsircl.ROUTE_VARIABLES if key in environment} == routes[rank]
        assert environment["SIRCL_BOOTSTRAP_ADDR"] == environment["VLLM_HOST_IP"] == f"192.0.2.{20 + positions[rank]}"
        # The switches the image's vLLM defines are off; the others are not set.
        for key in facts()["vllm"]["switches"]:
            assert environment[key] == "0"
        assert "VLLM_ENABLE_PCIE_ALLREDUCE" not in environment and environment["CUDA_VERSION"] == "13.0.2"
        assert spec.entrypoint == ("/opt/nvidia/nvidia_entrypoint.sh",)
        command = list(spec.command)
        assert command[:3] == ["vllm", "serve", "/model"]
        assert command[command.index("--tensor-parallel-size") + 1] == command[command.index("--nnodes") + 1] == str(size)
        assert command[command.index("--node-rank") + 1] == str(rank)
        assert command[command.index("--master-addr") + 1] == f"192.0.2.{20 + positions[0]}"
        assert ("--headless" in command) == (rank > 0) and "--disable-custom-all-reduce" in command
        config = json.loads(command[command.index("--compilation-config") + 1])
        assert config == {"pass_config": {"fuse_allreduce_rms": False, "fuse_gemm_comms": False, "enable_sp": False}}
        mounts = {mount.target: (mount.source, mount.read_only) for mount in spec.mounts}
        assert mounts[TARGET] == (LIBRARY, True) and mounts["/model"] == ("/srv/models/m", True)
        assert mounts["/run/sparkring/sircl/receipts"][1] is False
        assert spec.cap_add == ("IPC_LOCK",) and spec.security_opt == () and spec.devices == ("/dev/infiniband",)
        assert spec.memlock == -1 and spec.network_mode == "host" and spec.ipc_mode == "host"
    rendered = yaml.safe_load(stock_image.compose_text(ranks[0], REFERENCE))
    service = rendered["services"]["model"]
    assert service["image"] == REFERENCE and service["pull_policy"] == "never"
    assert service["environment"]["LD_PRELOAD"] == TARGET


def test_the_loader_seccomp_policy_is_added_only_when_the_images_loader_needs_io_uring():
    plain, _ = specs("pair", 2, [0, 1])
    assert all(spec.security_opt == () for spec in plain)
    loaded, _ = specs("pair", 2, [0, 1], arguments=["--load-format", "fastsafetensors"])
    assert loaded[0].security_opt == ("seccomp=/var/lib/sparkring/stock/sr-stock-0123456789ab/loader-seccomp.json",)
    b12x, _ = specs("pair", 2, [0, 1], found=facts(**{"vllm.general_plugins": ["b12x_loader"]}))
    assert b12x[1].security_opt == loaded[0].security_opt


def test_the_arguments_compilation_config_is_kept_and_none_is_added():
    ranks, _ = specs("pair", 2, [0, 1], arguments=["--compilation-config", '{"cudagraph_mode": "NONE"}'])
    command = list(ranks[0].command)
    assert command.count("--compilation-config") == 1
    assert command[command.index("--compilation-config") + 1] == '{"cudagraph_mode": "NONE"}'


def test_a_registry_reference_or_image_id_is_a_stock_image_and_a_release_name_is_not():
    assert stock_image.is_reference(REFERENCE) and stock_image.is_reference("eugr/spark-vllm-b12x:nightly-20261001")
    assert stock_image.is_reference("sha256:" + "5" * 64)
    assert not stock_image.is_reference("2026.10.1") and not stock_image.is_reference("statusrows")


def test_the_api_manifest_marks_the_functions_pynccl_calls_implemented():
    manifest = stock_image.api_manifest()
    assert all(manifest[name] for name in stock_image.CORE_FUNCTIONS)
    assert manifest["ncclCommSuspend"] is False and manifest["ncclCommResume"] is False
    assert stock_image.architectures() == ["sm_120", "sm_121"]
