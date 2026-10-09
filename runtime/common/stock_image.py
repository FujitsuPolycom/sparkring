"""A stock ARM64 vLLM image with libsircl as its NCCL: the preflight and each rank's container.

``sparkring install --image REF --transport libsircl --plan`` (the host flow
is ``runtime/host/stock_install.py``) plans a deployment of an unmodified
vLLM image, such as one of the community images built from
``eugr/spark-vllm-docker``, with libsircl in place of NCCL for both vLLM's
PyNccl and torch's ProcessGroupNCCL. The image is not an installer image:
it carries no SparkRing receipts, and SparkRing does not pin or pull it.
Status: research-only. docs/architecture/libsircl.md describes the option.

libsircl comes from the host: a build of the vendored source
(``libsircl_layer.py host-library``) at ``libsircl.host_library_path``,
mounted read-only at the path the image layer uses. ``preflight`` refuses,
naming each reason, unless the image's facts (``stock_image_probe.probe``,
read in a network-less, read-only container of the image with the GPUs and
the library mounted) show:

- an ``arm64`` image whose interpreter runs on ``aarch64``;
- a glibc at least as new as the newest ``GLIBC_x.y`` version the library
  needs, and a library that loads there, reports NCCL API level 22705,
  identifies itself as libsircl and has the fail-stop mode;
- a CUDA driver API of 13000 or later, and GPUs whose compute capability the
  kernel packs carry (``sm_120``, ``sm_121``);
- a vLLM that reads ``VLLM_NCCL_SO_PATH``, whose PyNccl binds only functions
  the library exports and that libsircl implements (``tests/api_manifest.json``;
  ``ncclCommSuspend`` and ``ncclCommResume`` are refused through
  ``--enable-sleep-mode`` instead), and whose CLI takes the multi-node
  options;
- a torch whose ``libtorch_cuda.so`` needs ``libnccl.so.2``, so that
  ``LD_PRELOAD`` points its ProcessGroupNCCL at libsircl;
- vLLM arguments that set none of the options the plan owns and pass the
  installer transport's refusals (``libsircl.refusals``).

``binding_problems`` then checks a second probe, run with the planned
``VLLM_NCCL_SO_PATH`` and ``LD_PRELOAD``: PyNccl's load and the process's
global scope must resolve ``ncclGetVersion`` in libsircl, torch must report
NCCL 2.27.5 (libsircl's API level 22705), and libsircl must be the only
mapped file with the SONAME ``libnccl.so.2``.
"""
import json
from pathlib import Path, PurePosixPath
import re

from runtime.common import libsircl, stock_image_probe, transport
from runtime.common.container_spec import Bind, ContainerSpec

ROOT = Path(__file__).resolve().parents[2]
TREE = ROOT / "spark_transport" / "libsircl"
API_MANIFEST = TREE / "tests" / "api_manifest.json"
PROBE_SOURCE = Path(stock_image_probe.__file__).read_text(encoding="utf-8")
SCHEMA = "sparkring-stock-plan/v1"
REQUIRED_DRIVER_API = 13000
# Functions every PyNccl communicator calls; libsircl must implement each of them.
CORE_FUNCTIONS = ("ncclGetErrorString", "ncclGetVersion", "ncclGetUniqueId", "ncclCommInitRank", "ncclAllReduce",
                  "ncclAllGather", "ncclReduceScatter", "ncclReduce", "ncclBroadcast", "ncclGroupStart",
                  "ncclGroupEnd", "ncclCommDestroy", "ncclCommAbort")
# Functions PyNccl calls only in a mode the plan refuses by its own rule: sleep mode (capability_problems).
CONDITIONAL_FUNCTIONS = ("ncclCommSuspend", "ncclCommResume")
# Options the plan sets on every rank's command; VLLM_ARGUMENTS may not set them.
OWNED_ARGUMENTS = ("--model", "--tensor-parallel-size", "-tp", "--nnodes", "--node-rank", "--master-addr",
                   "--master-port", "--headless", "--host", "--port", "--distributed-executor-backend",
                   libsircl.CUSTOM_ALL_REDUCE_OFF)
# Load formats whose loader uses io_uring, which SparkRing's loader seccomp policy admits.
IO_URING_FORMATS = ("fastsafetensors", "instanttensor")
MODEL_TARGET = "/model"
API_PORT = 8000
MASTER_PORT = 29511
RECEIPTS = "receipts"


class StockImageError(libsircl.LibsirclError):
    """A stock image cannot run with libsircl, or an input of its plan is malformed."""


def _require(condition, text):
    if not condition:
        raise StockImageError(text)


def is_reference(text):
    """Whether ``--image`` names an image by registry reference or local image ID, not an installer release."""
    return isinstance(text, str) and ("/" in text or ":" in text)


def api_manifest(path=API_MANIFEST):
    """``{function: implemented}`` of libsircl's API manifest."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["name"]: bool(row["implemented"]) for row in document["functions"]}


def architectures(tree=TREE):
    return libsircl.pack_architectures((Path(tree) / "Makefile").read_text(encoding="utf-8"))


def library_target(version):
    """Where a container sees the host library: the path the image layer installs it at."""
    return libsircl.library_path(version)


def probe_command(reference, host_library, version, *, environment=None, binding=False):
    """``docker run`` of the probe in a network-less, read-only container of ``reference``; the program is stdin."""
    target = library_target(version)
    command = ["sudo", "-n", "docker", "run", "--rm", "-i", "--pull", "never", "--network", "none", "--read-only",
               "--tmpfs", "/tmp", "--gpus", "all",
               "--mount", f"type=bind,src={host_library},dst={target},readonly"]
    for key, value in sorted((environment or {}).items()):
        command += ["--env", f"{key}={value}"]
    command += ["--entrypoint", "python3", reference, "-I", "-", target]
    return command + (["--binding"] if binding else [])


def parse(text):
    lines = [line[len(stock_image_probe.MARK):] for line in text.splitlines()
             if line.startswith(stock_image_probe.MARK)]
    _require(len(lines) == 1, "the stock-image probe printed no record")
    return json.loads(lines[0])


def _version(text):
    return tuple(int(part) for part in re.findall(r"[0-9]+", text or ""))


def image_environment(image):
    """``{name: value}`` of a ``docker image inspect`` record's ``Config.Env``."""
    return dict(item.partition("=")[::2] for item in (image.get("Config") or {}).get("Env") or [])


def preloads(image, version):
    """``LD_PRELOAD``: libsircl first, then the image's own entries except other NCCL libraries."""
    own = [item for item in re.split(r"[:\s]+", image_environment(image).get("LD_PRELOAD", "")) if item]
    return ":".join([library_target(version), *(item for item in own if "libnccl.so" not in PurePosixPath(item).name)])


def owned_arguments(arguments):
    return sorted({flag for flag in OWNED_ARGUMENTS for item in arguments if item == flag or item.startswith(flag + "=")})


def preflight(image, facts, arguments, *, packs=None, manifest=None):
    """What stops ``image`` with ``facts`` (``stock_image_probe.probe``) from running ``arguments`` on libsircl;
    empty when nothing does."""
    packs = architectures() if packs is None else packs
    manifest = api_manifest() if manifest is None else manifest
    problems = []
    if image.get("Architecture") != "arm64" or facts.get("machine") != "aarch64":
        problems.append(f"the image is {image.get('Architecture')}/{facts.get('machine')}, not arm64/aarch64")
    library = facts.get("library") or {}
    if library.get("error") or library.get("load") != "ok":
        problems.append(f"the host libsircl does not load in the image: {library.get('error') or library.get('load')}")
    required, present = library.get("glibc_required"), facts.get("glibc")
    if required and (not present or _version(present) < _version(required)):
        problems.append(f"the host libsircl needs glibc {required}, and the image has {present or 'none'}")
    if library.get("soname") != libsircl.SONAME:
        problems.append(f"the host library's SONAME is {library.get('soname')}, not {libsircl.SONAME}")
    if library.get("load") == "ok":
        if library.get("nccl_version") != libsircl.NCCL_API_VERSION:
            problems.append(f"the host library reports NCCL API level {library.get('nccl_version')}, not "
                            f"{libsircl.NCCL_API_VERSION}")
        if (library.get("identity") or {}).get("library") != libsircl.PLUGIN_NAME:
            problems.append("the host library does not identify itself as libsircl (sirclGetInfo)")
    if not library.get("fail_stop"):
        problems.append(libsircl.fail_stop_missing("the host libsircl build"))
    cuda = facts.get("cuda") or {}
    if not cuda.get("driver_api"):
        problems.append(f"the image finds no CUDA driver: {cuda.get('error')}")
    elif cuda["driver_api"] < REQUIRED_DRIVER_API:
        problems.append(f"the CUDA driver API is {cuda['driver_api']}; libsircl's kernel packs, sm_120 and sm_121 code "
                        f"built by nvcc 13.3 without PTX, need {REQUIRED_DRIVER_API} or later")
    capabilities = cuda.get("compute_capabilities") or []
    if cuda.get("driver_api") and not capabilities:
        problems.append(f"the container sees no GPU: {cuda.get('error') or 'no device'}")
    for capability in capabilities:
        if "sm_" + capability.replace(".", "") not in packs:
            problems.append(f"a GPU of compute capability {capability} runs none of the kernel packs' "
                            f"architectures ({', '.join(packs)})")
    vllm = facts.get("vllm")
    if not vllm:
        problems.append("the image has no vllm package")
    else:
        if not (vllm.get("nccl_so_path") and vllm.get("find_reads_nccl_so_path")):
            problems.append(f"the image's vLLM {vllm.get('version')} does not select PyNccl's library with "
                            "VLLM_NCCL_SO_PATH")
        bound = vllm.get("pynccl_functions") or []
        if not bound:
            problems.append("the image's vLLM has no PyNccl function list (pynccl_wrapper.py)")
        missing = library.get("missing") or []
        if missing:
            problems.append(f"vLLM's PyNccl binds {', '.join(missing)}, which the host libsircl does not export")
        for name in CORE_FUNCTIONS:
            if name in bound and manifest.get(name) is False:
                problems.append(f"vLLM's PyNccl calls {name}, which libsircl exports only as a refusal")
        for name in bound:
            if manifest.get(name) is False and name not in CORE_FUNCTIONS and name not in CONDITIONAL_FUNCTIONS:
                problems.append(f"vLLM's PyNccl binds {name}, which libsircl exports only as a refusal")
        if not vllm.get("multinode"):
            problems.append("the image's vLLM CLI lacks --nnodes, --node-rank, --master-addr or --headless")
        if not vllm.get("command"):
            problems.append("the image has no vllm command")
    torch = facts.get("torch")
    if not torch or torch.get("nccl") != "dynamic":
        problems.append(f"the image's torch NCCL is {(torch or {}).get('nccl', 'missing')}: only a torch whose "
                        "libtorch_cuda.so needs libnccl.so.2 can be pointed at libsircl, and torch's own collectives "
                        "would otherwise reach another NCCL")
    owned = owned_arguments(arguments)
    if owned:
        problems.append(f"the vLLM arguments set {', '.join(owned)}, which the plan sets on every rank")
    problems += libsircl.refusals(image_environment(image), arguments)
    return problems


def binding_environment(image, version):
    return {"VLLM_NCCL_SO_PATH": library_target(version), "LD_PRELOAD": preloads(image, version),
            "LIBSIRCL_NCCL_API_VERSION": str(libsircl.NCCL_API_VERSION)}


def binding_problems(found, version):
    """What the binding probe (``stock_image_probe.binding``) found bound to another library than libsircl."""
    target = library_target(version)
    problems = []
    if found.get("pynccl_provider") != target:
        problems.append(f"PyNccl's load of VLLM_NCCL_SO_PATH resolves ncclGetVersion in "
                        f"{found.get('pynccl_provider') or found.get('pynccl_error')}, not {target}")
    if found.get("global_provider") != target:
        problems.append(f"the process's global scope resolves ncclGetVersion in {found.get('global_provider')}, "
                        f"not {target}")
    if found.get("torch_nccl_version") != [2, 27, 5]:
        problems.append(f"torch reports NCCL {found.get('torch_nccl_version') or found.get('torch_error')}, not "
                        "libsircl's 2.27.5")
    if found.get("mapped") != [target]:
        problems.append(f"the files mapped with the SONAME libnccl.so.2 are {found.get('mapped')}, not libsircl alone")
    return problems


def serve_command(entrypoint, arguments):
    """The command after the image's own entrypoint that runs ``vllm serve`` with ``arguments``."""
    words = [PurePosixPath(word).name for word in entrypoint or ()]
    if words[-2:] == ["vllm", "serve"]:
        return list(arguments)
    if words[-1:] == ["vllm"]:
        return ["serve", *arguments]
    return ["vllm", "serve", *arguments]


def compilation_arguments(arguments, facts):
    """``--compilation-config`` turning off the communication fusions the image's vLLM has, unless the arguments
    give their own (which ``libsircl.refusals`` checks)."""
    if libsircl.argument(arguments, "--compilation-config", "-cc") is not None:
        return []
    fusions = (facts.get("vllm") or {}).get("fusions") or []
    if not fusions:
        return []
    return ["--compilation-config", json.dumps({"pass_config": {name: False for name in fusions}},
                                               separators=(",", ":"))]


def rank_command(image, facts, arguments, *, rank, size, master):
    distributed = ["--tensor-parallel-size", str(size), "--nnodes", str(size), "--node-rank", str(rank),
                   "--master-addr", master, "--master-port", str(MASTER_PORT), libsircl.CUSTOM_ALL_REDUCE_OFF]
    distributed += ["--headless"] if rank else ["--host", "0.0.0.0", "--port", str(API_PORT)]
    entrypoint = (image.get("Config") or {}).get("Entrypoint") or []
    return serve_command(entrypoint, [MODEL_TARGET, *arguments, *distributed,
                                      *compilation_arguments(arguments, facts)])


def loader_policy_needed(facts, arguments):
    """Whether the image's loader needs io_uring, which SparkRing's loader seccomp policy admits."""
    load_format = libsircl.argument(arguments, "--load-format")
    plugins = (facts.get("vllm") or {}).get("general_plugins") or []
    return load_format in IO_URING_FORMATS or "b12x_loader" in plugins


def rank_spec(image, facts, arguments, *, name, reference, rank, row, master, size, routes, host_library, version,
              model_path, directory, gid=3):
    """Rank ``rank``'s container: the image's own entrypoint with libsircl as its NCCL.

    ``row`` is the rank's site row (``host``, ``fabric_ip``, ``interface``);
    ``directory`` the plan's directory on every Spark, which holds the
    receipts and, when the loader needs it, the seccomp policy.
    """
    switches = (facts.get("vllm") or {}).get("switches") or []
    off = {key: value for key, value in libsircl.INDEPENDENT_TRANSPORTS_OFF.items() if key in switches}
    if "VLLM_ENABLE_ROCE_ALLREDUCE" in switches:
        off["VLLM_ENABLE_ROCE_ALLREDUCE"] = "0"
    library = {libsircl.FAIL_STOP_VARIABLE: "1", "LIBSIRCL_NCCL_API_VERSION": str(libsircl.NCCL_API_VERSION),
               "LIBSIRCL_TRANSPORT": "verbs", "LIBSIRCL_RECEIPT": libsircl.RECEIPT_PREFIX}
    environment = {
        **image_environment(image), **library, **off, **routes,
        "VLLM_NCCL_SO_PATH": library_target(version), "LD_PRELOAD": preloads(image, version),
        "SIRCL_BOOTSTRAP_ADDR": row["fabric_ip"], "SIRCL_GID_INDEX": str(gid),
        "VLLM_HOST_IP": row["fabric_ip"], "GLOO_SOCKET_IFNAME": row["interface"], "NCCL_SOCKET_IFNAME": row["interface"],
        "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT",
    }
    mounts = (Bind(host_library, library_target(version), True), Bind(model_path, MODEL_TARGET, True),
              Bind(str(PurePosixPath(directory) / RECEIPTS), transport.RECEIPT_TARGET, False))
    security = ()
    if loader_policy_needed(facts, arguments):
        security = ("seccomp=" + str(PurePosixPath(directory) / "loader-seccomp.json"),)
    return ContainerSpec(name=f"{name}-r{rank}", image_id=reference,
                         entrypoint=tuple((image.get("Config") or {}).get("Entrypoint") or ()),
                         command=tuple(rank_command(image, facts, arguments, rank=rank, size=size, master=master)),
                         environment=environment, mounts=mounts, cap_add=("IPC_LOCK",), security_opt=security,
                         labels={"io.sparkring.stock-plan": name, "io.sparkring.rank": str(rank),
                                 "io.sparkring.status": libsircl.STATUS})


def compose_text(spec, reference):
    from runtime.common import compose
    text = compose.compose_text(spec, reference)
    first, _, rest = text.partition("\n")
    return ("# Generated by sparkring install --image REF --transport libsircl --plan; research-only.\n" + rest)
