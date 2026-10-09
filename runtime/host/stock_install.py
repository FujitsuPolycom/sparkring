"""``sparkring install --image REF --transport libsircl --plan -- VLLM_ARGUMENTS``: a stock vLLM image on libsircl.

The option plans a deployment of an unmodified ARM64 vLLM image with
libsircl, SIRCL's NCCL-compatible C library, as the NCCL of both vLLM's
PyNccl and torch's ProcessGroupNCCL (``runtime/common/stock_image.py``).
Status: research-only. It plans and checks only:

1. It places the group on the recorded fabric as ``--transport libsircl``
   does (``libsircl.group``: pairs, paths and whole cycles of two to eight
   Sparks) and takes each rank's routing settings from libsircl's
   ``tools/site_routes.py``.
2. On every Spark of the group, over SSH and read-only, it inspects the image
   (``docker image inspect``; it never pulls), hashes the host build of
   libsircl named by ``--libsircl-library``, and runs the stock-image probe
   twice in network-less, read-only containers of the image: once for the
   image's facts and once with the planned ``VLLM_NCCL_SO_PATH`` and
   ``LD_PRELOAD`` for the library each caller binds. Every Spark must hold
   the same image and library bytes.
3. It refuses with every reason ``stock_image.preflight`` and
   ``stock_image.binding_problems`` give, or writes each rank's Compose file
   (and SparkRing's loader seccomp policy where the image's loader needs it)
   to ``<state>/stock/<name>/`` on Node A, with the commands that copy them
   to ``/var/lib/sparkring/stock/<name>/`` on each Spark and start and stop
   the containers.

The installer does not start, stop or switch a stock deployment, and
``sparkring status`` does not list it. Without ``--plan`` the option refuses.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex

from runtime.common import installer, libsircl, loader_policy, stock_image, transport
from runtime.host.install_errors import NeedsInput

SPARK_DIRECTORY = "/var/lib/sparkring/stock"
_LIBRARY = re.compile(r"/[^\0]*/libsircl\.so\.([0-9]+\.[0-9]+\.[0-9]+)")


def _needs(message, field, details=None):
    raise NeedsInput(f"{message}. Nothing has been changed.", field=field, details=details)


def request(args):
    """``(reference, library path, version, model path)`` of a stock-image request, refusing what it cannot plan."""
    if args.transport != libsircl.BACKEND:
        _needs(f"--image {args.image} names a stock vLLM image, which runs only with --transport libsircl", "transport")
    if args.nccl is not None:
        _needs("--nccl applies to deployments on SIRCL; a stock image runs every collective on libsircl", "transport")
    if args.profile:
        _needs("--image REF runs the image's own vLLM with the arguments after --; it takes no --profile", "profile")
    if not args.plan:
        _needs("The stock-image option is research-only and plans only: add --plan. It writes each rank's Compose "
               "file and the commands that start them", "plan")
    library = getattr(args, "libsircl_library", None)
    found = _LIBRARY.fullmatch(library or "")
    if not found:
        _needs("--libsircl-library names the host build of libsircl every Spark holds, an absolute path ending in "
               "libsircl.so.<version> (runtime/images/libsircl_layer.py host-library)", "libsircl_library")
    paths = args.model_path or []
    if len(paths) != 1 or "=" in paths[0] or not PurePosixPath(paths[0]).is_absolute():
        _needs("--model-path names one absolute directory that holds the model on every Spark", "model_path")
    return args.image, library, found.group(1), paths[0]


def _ssh(invoke, host, command, data=None):
    try:
        return invoke(host, command, data=data) if data is not None else invoke(host, command)
    except (RuntimeError, OSError) as error:
        raise NeedsInput(f"Spark {host} could not run {' '.join(command[:4])}: {error}. Nothing has been changed.",
                         field="image") from None


def survey(invoke, rows, reference, library, version, arguments):
    """Each Spark's image record, library digest, facts and binding, and the problems ``preflight`` finds."""
    reports, problems = [], {}
    for rank, row in enumerate(rows):
        host = row["host"]
        image = json.loads(_ssh(invoke, host, ["sudo", "-n", "docker", "image", "inspect", reference]))[0]
        digest = _ssh(invoke, host, ["sha256sum", library]).split()[0]
        facts = stock_image.parse(_ssh(invoke, host, stock_image.probe_command(reference, library, version),
                                       data=stock_image.PROBE_SOURCE))
        found = stock_image.preflight(image, facts, arguments)
        bound = None
        if not found:
            bound = stock_image.parse(_ssh(invoke, host, stock_image.probe_command(
                reference, library, version, environment=stock_image.binding_environment(image, version),
                binding=True), data=stock_image.PROBE_SOURCE))
            found = stock_image.binding_problems(bound, version)
        if found:
            problems[f"rank {rank} ({host})"] = found
        reports.append({"host": host, "image": image, "library_sha256": digest, "facts": facts, "binding": bound})
    if len({report["image"]["Id"] for report in reports}) > 1:
        problems["image"] = ["the Sparks hold different images under " + reference]
    if len({report["library_sha256"] for report in reports}) > 1:
        problems["library"] = [f"the Sparks hold different bytes at {library}"]
    return reports, problems


def plan(args, arguments, *, state, invoke):
    """Check a stock-image request on every Spark of its group and write each rank's Compose file."""
    reference, library, version, model = request(args)
    from runtime.host import controller, install_workflow, placement as placements, relays
    install_workflow.require_head()
    try:
        cluster = installer.read(Path(state) / "cluster.json")
    except OSError:
        _needs("Run sudo sparkring setup first: a stock image runs on the recorded fabric", "setup")
    document = install_workflow.recorded_fabric(state, cluster)
    reason = libsircl.fabric_unavailable(document)
    if reason:
        _needs(f"--transport libsircl cannot run here: {reason}", "transport")
    layout = placements.layout_of(cluster)
    try:
        placement = placements.parse(args.on, layout) if args.on else None
    except ValueError as error:
        _needs(str(error), "placement")
    positions = list(placement) if placement is not None else list(range(len(cluster["plan"]["spec"]["hosts"])))
    try:
        group, devices, routes, planner = libsircl.group(document, positions)
    except transport.TransportError as error:
        _needs(f"--transport libsircl cannot run on positions {positions}: {error}", "placement")
    reference_document = relays.group_reference(state, cluster) if placements.relayed(layout, placement) else None
    site = controller.model_site(cluster, "stock-libsircl", "main", placement, fabric=reference_document)
    rows = site["hosts"]
    reports, problems = survey(invoke, rows, reference, library, version, list(arguments))
    if problems:
        lines = [f"{where}: {problem}" for where, found in problems.items() for problem in found]
        _needs("The stock image cannot run with libsircl here: " + "; ".join(lines), "image",
               details={"problems": problems})
    image = reports[0]["image"]
    digest = reports[0]["library_sha256"]
    identity = {"image": image["Id"], "library": digest, "positions": positions, "model": model,
                "arguments": list(arguments), "fabric": document["id"]}
    name = "sr-stock-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    spark_directory = f"{SPARK_DIRECTORY}/{name}"
    output = Path(state) / "stock" / name
    output.mkdir(parents=True, exist_ok=True)
    master = rows[0]["fabric_ip"]
    files, commands = {}, {"prepare": [], "start": [], "stop": []}
    for rank, (row, report) in enumerate(zip(rows, reports, strict=True)):
        spec = stock_image.rank_spec(image, report["facts"], list(arguments), name=name, reference=image["Id"],
                                     rank=rank, row=row, master=master, size=len(rows), routes=routes[rank],
                                     host_library=library, version=version, model_path=model,
                                     directory=spark_directory)
        compose = f"compose.rank{rank}.yaml"
        files[compose] = stock_image.compose_text(spec, image["Id"])
        host = shlex.quote(row["host"])
        remote = f"{spark_directory}/{compose}"
        commands["prepare"].append(f"ssh {host} 'sudo install -d -m 0755 {spark_directory}/{stock_image.RECEIPTS}' && "
                                   f"ssh {host} 'sudo tee {remote} >/dev/null' < {output / compose}")
        if spec.security_opt:
            files["loader-seccomp.json"] = loader_policy.PROFILE.read_text(encoding="utf-8")
            commands["prepare"].append(f"ssh {host} 'sudo tee {spark_directory}/loader-seccomp.json >/dev/null' "
                                       f"< {output / 'loader-seccomp.json'}")
        commands["start"].append(f"ssh {host} 'sudo docker compose -f {remote} up -d'")
        commands["stop"].append(f"ssh {host} 'sudo docker compose -f {remote} down'")
    for file_name, text in files.items():
        (output / file_name).write_text(text, encoding="utf-8", newline="\n")
    result = {"schema": stock_image.SCHEMA, "state": "planned", "status": libsircl.STATUS, "name": name,
              "image": {"reference": reference, "id": image["Id"]}, "library": {"path": library, "sha256": digest},
              "group": group, "positions": positions, "routes": routes, "planner": planner,
              "directory": {"node_a": str(output), "sparks": spark_directory}, "files": sorted(files),
              "commands": commands, "api_url": f"http://{rows[0]['fabric_ip']}:{stock_image.API_PORT}/v1",
              "preflight": [{"host": report["host"], "glibc": report["facts"].get("glibc"),
                             "cuda_driver_api": (report["facts"].get("cuda") or {}).get("driver_api"),
                             "vllm": (report["facts"].get("vllm") or {}).get("version"),
                             "torch": (report["facts"].get("torch") or {}).get("version"),
                             "torch_nccl": (report["binding"] or {}).get("torch_nccl_version")}
                            for report in reports]}
    (output / "plan.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                      newline="\n")
    return result


def lines(result):
    """What the plan prints."""
    group = result["group"]
    printed = [f"Stock image {result['image']['reference']} ({result['image']['id'][7:19]}) on libsircl "
               f"({result['status']}): vLLM's PyNccl and torch's ProcessGroupNCCL on {result['library']['path']}",
               f"  libsircl group: {group['name']} at positions {', '.join(map(str, group['positions']))}; "
               f"{group['lanes']} lanes per peer, {libsircl.relays_text(group)}",
               f"  Off: {libsircl.OFF_TEXT}; fail-stop on ({libsircl.FAIL_STOP_VARIABLE}=1)",
               f"  Files: {result['directory']['node_a']}; SparkRing does not start, stop or switch this deployment",
               "  Copy them to every Spark:"]
    printed += ["    " + command for command in result["commands"]["prepare"]]
    printed += ["  Start every rank:"] + ["    " + command for command in result["commands"]["start"]]
    printed += ["  Stop:"] + ["    " + command for command in result["commands"]["stop"]]
    return printed
