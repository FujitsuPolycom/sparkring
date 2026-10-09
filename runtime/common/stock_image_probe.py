"""The facts of a vLLM image that libsircl's stock-image option checks, read inside a container of the image.

This program uses only the Python standard library, so it runs in any image
with Python 3.8 or later: ``runtime/common/stock_image.py`` sends its source
to ``python3 -I -`` in a network-less, read-only container of the image, with
the GPUs and the host build of libsircl mounted as serving mounts them and
the library's container path as the first argument. It prints one line,
``STOCK-IMAGE-PROBE <json>``.

Without ``--binding`` it reads files and imports neither torch nor vLLM:

- ``machine``, ``python`` and ``glibc``: ``platform.machine()``, the
  interpreter version and ``os.confstr("CS_GNU_LIBC_VERSION")``;
- ``library``: the mounted libsircl's ELF facts (``elf_facts``), whether its
  bytes name ``LIBSIRCL_FAIL_STOP`` (the fail-stop mode), whether it loads
  here (``load``: ``ok``, or the loader's error such as a missing
  ``GLIBC_x.y`` version), ``ncclGetVersion``'s value, ``sirclGetInfo``'s
  library name and version, and the functions vLLM's PyNccl binds that it
  lacks;
- ``cuda``: ``cuDriverGetVersion`` through the container's ``libcuda.so.1``
  and the compute capability of every visible GPU, or the error;
- ``vllm``: its version and directory, the ``vllm`` command, whether
  ``vllm/envs.py`` defines ``VLLM_NCCL_SO_PATH`` and ``find_nccl_library``
  reads it, the functions PyNccl's ``NCCLLibrary`` binds (``Function("<name>",
  ...)`` entries of ``pynccl_wrapper.py``), whether the CLI accepts the
  multi-node options, which switches of collective paths ``vllm/envs.py``
  defines, which communication-fusion fields the compilation configuration
  has, and the ``vllm.general_plugins`` entry-point names;
- ``torch``: its version and whether ``libtorch_cuda.so`` needs
  ``libnccl.so.2`` (``dynamic``), carries NCCL itself (``static``) or neither
  (``absent``);
- ``loader``: whether ``fastsafetensors`` is installed.

With ``--binding``, run with the planned ``VLLM_NCCL_SO_PATH`` and
``LD_PRELOAD``, it imports torch and reports which library each caller bound
(``binding``): the file that provides ``ncclGetVersion`` to a ``ctypes``
load of ``VLLM_NCCL_SO_PATH`` (PyNccl's load) and to the process's global
scope, ``torch.cuda.nccl.version()``, and every mapped file whose SONAME is
``libnccl.so.2``.
"""
import ast
import ctypes
import importlib.util
import json
import os
import platform
import re
import shutil
import struct
import sys

MARK = "STOCK-IMAGE-PROBE "
DT_NEEDED, DT_SONAME = 1, 14
FAIL_STOP = b"LIBSIRCL_FAIL_STOP\x00"
SWITCHES = ("VLLM_DISABLE_PYNCCL", "VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_USE_NCCL_SYMM_MEM",
            "VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "VLLM_ENABLE_PCIE_ALLREDUCE",
            "VLLM_ENABLE_ROCE_ALLREDUCE")
FUSIONS = ("fuse_allreduce_rms", "fuse_gemm_comms", "enable_sp")
MULTINODE = ("nnodes", "node_rank", "master_addr", "master_port", "headless")


def elf_facts(path):
    """``{"soname", "needed", "glibc_required"}`` of an ELF64 little-endian shared object, from its section headers.

    ``glibc_required`` is the newest ``GLIBC_x.y`` version its
    ``.gnu.version_r`` section needs, as ``"x.y"``, or None.
    """
    with open(path, "rb") as stream:
        data = stream.read()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError(f"{path} is not a 64-bit little-endian ELF file")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    sections = []
    for index in range(shnum):
        name, kind, _, _, offset, size, link, _, _, entsize = struct.unpack_from("<IIQQQQIIQQ", data,
                                                                                 shoff + index * shentsize)
        sections.append({"name": name, "offset": offset, "size": size})
    names = sections[shstrndx]

    def string(table, offset):
        start = table["offset"] + offset
        return data[start:data.index(b"\x00", start)].decode()
    by_name = {string(names, section["name"]): section for section in sections}
    dynamic, strings = by_name.get(".dynamic"), by_name.get(".dynstr")
    needed, soname = [], None
    if dynamic is not None and strings is not None:
        for offset in range(dynamic["offset"], dynamic["offset"] + dynamic["size"], 16):
            tag, value = struct.unpack_from("<qQ", data, offset)
            if tag == 0:
                break
            if tag == DT_NEEDED:
                needed.append(string(strings, value))
            elif tag == DT_SONAME:
                soname = string(strings, value)
    versions = []
    requirements = by_name.get(".gnu.version_r")
    if requirements is not None and strings is not None and requirements["size"]:
        offset = requirements["offset"]
        while True:
            _, count, _, aux, following = struct.unpack_from("<HHIII", data, offset)
            entry = offset + aux
            for _ in range(count):
                _, _, _, name, next_aux = struct.unpack_from("<IHHII", data, entry)
                found = re.fullmatch(r"GLIBC_([0-9]+)\.([0-9]+)(?:\.([0-9]+))?", string(strings, name))
                if found:
                    versions.append(tuple(int(part or 0) for part in found.groups()))
                entry += next_aux
            if not following:
                break
            offset += following
    newest = max(versions) if versions else None
    required = None if newest is None else ".".join(str(part) for part in (newest if newest[2] else newest[:2]))
    return {"soname": soname, "needed": needed, "glibc_required": required}


def _read(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return stream.read()
    except OSError:
        return ""


def _package(name):
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def _reads_variable(text, function, variable):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == function
               and variable in (ast.get_source_segment(text, node) or "") for node in ast.walk(tree))


def bound_functions(text):
    """The names of ``Function("<name>", ...)`` calls in a ``pynccl_wrapper.py`` source."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return sorted({node.args[0].value for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Function" and node.args
                   and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)})


def _metadata_version(name):
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return None


def _general_plugins():
    try:
        from importlib import metadata
        points = metadata.entry_points()
        group = (points.select(group="vllm.general_plugins") if hasattr(points, "select")
                 else points.get("vllm.general_plugins", ()))
        return sorted({point.name for point in group})
    except Exception:
        return []


def vllm_facts():
    root = _package("vllm")
    if root is None:
        return None
    envs = _read(os.path.join(root, "envs.py"))
    finders = [_read(os.path.join(root, "utils", "nccl.py")), _read(os.path.join(root, "utils", "__init__.py"))]
    wrapper = _read(os.path.join(root, "distributed", "device_communicators", "pynccl_wrapper.py"))
    cli = _read(os.path.join(root, "engine", "arg_utils.py")) + _read(os.path.join(root, "entrypoints", "cli",
                                                                                   "serve.py"))
    compilation = _read(os.path.join(root, "config", "compilation.py"))
    return {"version": _metadata_version("vllm"), "root": root, "command": shutil.which("vllm"),
            "nccl_so_path": '"VLLM_NCCL_SO_PATH"' in envs,
            "find_reads_nccl_so_path": any(_reads_variable(text, "find_nccl_library", "VLLM_NCCL_SO_PATH")
                                           for text in finders),
            "pynccl_functions": bound_functions(wrapper),
            "multinode": all(word in cli for word in MULTINODE),
            "switches": [name for name in SWITCHES if f'"{name}"' in envs],
            "fusions": [name for name in FUSIONS if re.search(rf"\b{name}\s*:", compilation)],
            "general_plugins": _general_plugins()}


def library_facts(path, functions):
    """The mounted library's facts; ``functions`` are the names vLLM's PyNccl binds."""
    result = {"path": path}
    try:
        result.update(elf_facts(path))
        with open(path, "rb") as stream:
            result["fail_stop"] = FAIL_STOP in stream.read()
    except (OSError, ValueError, struct.error) as error:
        result["error"] = str(error)
        return result
    try:
        handle = ctypes.CDLL(path)
    except OSError as error:
        result["load"] = str(error)
        return result
    result["load"] = "ok"
    version = ctypes.c_int()
    result["nccl_version"] = version.value if handle.ncclGetVersion(ctypes.byref(version)) == 0 else None
    try:
        handle.sirclGetInfo.restype = ctypes.c_char_p
        info = json.loads(handle.sirclGetInfo().decode())
        result["identity"] = {"library": info.get("library"), "version": info.get("version")}
    except (AttributeError, ValueError):
        result["identity"] = None
    result["missing"] = [name for name in functions if not hasattr(handle, name)]
    return result


def cuda_facts():
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        return {"error": str(error)}
    version = ctypes.c_int()
    if driver.cuDriverGetVersion(ctypes.byref(version)) != 0:
        return {"error": "cuDriverGetVersion failed"}
    result = {"driver_api": version.value, "compute_capabilities": []}
    if driver.cuInit(0) != 0:
        result["error"] = "cuInit failed: no visible GPU"
        return result
    count = ctypes.c_int()
    driver.cuDeviceGetCount(ctypes.byref(count))
    for index in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        driver.cuDeviceGet(ctypes.byref(device), index)
        driver.cuDeviceGetAttribute(ctypes.byref(major), 75, device)
        driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)
        result["compute_capabilities"].append(f"{major.value}.{minor.value}")
    return result


def torch_facts():
    root = _package("torch")
    if root is None:
        return None
    library = os.path.join(root, "lib", "libtorch_cuda.so")
    result = {"version": _metadata_version("torch"), "libtorch_cuda": library}
    if not os.path.isfile(library):
        result["nccl"] = "absent"
        return result
    try:
        needed = elf_facts(library)["needed"]
    except (OSError, ValueError, struct.error) as error:
        result.update(nccl="unknown", error=str(error))
        return result
    if any(name.startswith("libnccl.so") for name in needed):
        result["nccl"] = "dynamic"
    else:
        with open(library, "rb") as stream:
            result["nccl"] = "static" if b"ncclCommInitRank" in stream.read() else "absent"
    return result


def glibc():
    try:
        return (os.confstr("CS_GNU_LIBC_VERSION") or "").replace("glibc ", "") or None
    except (AttributeError, ValueError, OSError):
        return None


def probe(library=None):
    vllm = vllm_facts()
    return {"machine": platform.machine(), "python": platform.python_version(), "glibc": glibc(),
            "library": library_facts(library, (vllm or {}).get("pynccl_functions") or []) if library else None,
            "cuda": cuda_facts(), "vllm": vllm, "torch": torch_facts(),
            "loader": {"fastsafetensors": _package("fastsafetensors") is not None
                       or importlib.util.find_spec("fastsafetensors") is not None}}


class _DlInfo(ctypes.Structure):
    _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p), ("dli_sname", ctypes.c_char_p),
                ("dli_saddr", ctypes.c_void_p)]


def provider(handle, name="ncclGetVersion"):
    """The real path of the file that provides ``name`` to ``handle``, or None."""
    try:
        address = ctypes.cast(getattr(handle, name), ctypes.c_void_p).value
    except AttributeError:
        return None
    for library in (None, "libdl.so.2"):
        try:
            dladdr = ctypes.CDLL(library).dladdr
            break
        except (OSError, AttributeError):
            continue
    else:
        return None
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    info = _DlInfo()
    if not dladdr(address, ctypes.byref(info)) or not info.dli_fname:
        return None
    return os.path.realpath(info.dli_fname.decode())


def mapped_nccl():
    """Every mapped file whose ELF SONAME is ``libnccl.so.2``."""
    paths = set()
    try:
        with open("/proc/self/maps") as stream:
            for line in stream:
                parts = line.split()
                if len(parts) >= 6 and parts[-1].startswith("/"):
                    paths.add(parts[-1])
    except OSError:
        return None
    found = []
    for path in sorted(paths):
        try:
            if elf_facts(path)["soname"] == "libnccl.so.2":
                found.append(os.path.realpath(path))
        except (OSError, ValueError, struct.error, IndexError):
            continue
    return sorted(set(found))


def binding():
    """Which library PyNccl's load and torch bound, in a process started with the planned loader settings."""
    result = {"vllm_nccl_so_path": os.environ.get("VLLM_NCCL_SO_PATH"), "ld_preload": os.environ.get("LD_PRELOAD")}
    try:
        result["pynccl_provider"] = provider(ctypes.CDLL(result["vllm_nccl_so_path"] or "libnccl.so.2"))
    except OSError as error:
        result["pynccl_provider"] = None
        result["pynccl_error"] = str(error)
    try:
        import torch
        result["torch_nccl_version"] = list(torch.cuda.nccl.version())
    except Exception as error:
        result["torch_nccl_version"] = None
        result["torch_error"] = repr(error)
    result["global_provider"] = provider(ctypes.CDLL(None))
    result["mapped"] = mapped_nccl()
    return result


if __name__ == "__main__":
    arguments = [item for item in sys.argv[1:] if item != "--binding"]
    value = binding() if "--binding" in sys.argv[1:] else probe(arguments[0] if arguments else None)
    print(MARK + json.dumps(value, sort_keys=True))
