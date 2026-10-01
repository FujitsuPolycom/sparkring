"""One Linux entrypoint for cluster setup, asset preparation and model replacement.

Before approval, ``sparkring install`` surveys every Spark for copies of the
profile's pinned checkpoint and prints a checkpoint plan
(``runtime.host.checkpoint_plan``): what each Spark links, copies, receives or
downloads, from which folders, and the free space it needs. The plan is saved
as ``<deployment>/checkpoint-plan.json``. A plan saved by ``--plan``, or
approved at a terminal prompt, is marked reviewed and bounds every later
``--yes`` run until the next ``--plan`` run or terminal approval replaces it; a
``--yes`` run whose fresh plan leaves it is refused, and that plan is saved
beside it as ``checkpoint-plan.refused.json``. The approved plan bounds what
the checkpoint preparation may download and write. A new deployment whose plan
has problems is not recorded, so repeating the command plans it again.

On an installed four-Spark ring the installation also applies the ConnectX
hairpin setting (``runtime/host/hairpin_ring.py``) where a Spark lacks it or its
boot record, after the one approval and before the model transaction.
"""
import argparse
import concurrent.futures
import contextlib
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

from runtime.common import distribution, installer, installer_image, process_lock, profiles
from runtime.common import serving as serving_settings
from runtime.host import (checkpoint_plan, checkpoint_search, controller, discovery, fabric_ssh, hairpin_ring,
                          install_assets, install_space, models, native_mesh, node, progress, recovery,
                          retained_source, retention, rollout, settings, topology)
from runtime.host.install_errors import NeedsInput
from scripts import deploy_network

# The survey runs as root at the lowest CPU and I/O priority and reads its source
# on stdin. Its own time and entry budgets bound its work; the SSH timeout bounds
# a survey that does not return.
SURVEY_COMMAND = ["sudo", "-n", "nice", "-n", "19", "ionice", "-c", "3", "python3", "-I", "-B", "-"]
SURVEY_TIMEOUT = 150
PLAN_FILE = "checkpoint-plan.json"
# A plan that a --yes run could not use because it leaves the reviewed plan, kept for inspection.
REFUSED_FILE = "checkpoint-plan.refused.json"


def require_head(cluster=None, *, command="install"):
    if sys.platform != "linux" or not distribution.installed(installer.ROOT):
        raise NeedsInput(f"sparkring {command} runs only from the installed ARM64 Debian package. Download the "
                         "sparkring_*_arm64.deb asset of a prerelease at https://github.com/FujitsuPolycom/sparkring/releases "
                         "or build it from a full clone (see \"Get the package\" in docs/operations/install-reference.md), "
                         f"install it on Node A with sudo apt install, then run sudo sparkring {command}.", field="node_a")
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise NeedsInput(f"Run sudo sparkring {command} on Node A.", field="node_a")
    identity = node.read("/", "/etc/sparkring/node.json")["node_id"]
    if cluster is not None and cluster["plan"]["spec"]["hosts"][0]["node_id"] != identity:
        raise ValueError("This is not the enrolled Node A; image/model traffic must originate on that Spark")
    return identity


def check_access(cluster, *, invoke=None):
    """Confirm noninteractive SSH and sudo on every enrolled Spark before any change.

    Passwords never pass through SparkRing. A missing grant stops with the exact
    one-time command a person runs on that Spark (it prompts for their password).
    """
    invoke = invoke or discovery.ssh
    missing = []
    for rank, row in enumerate(cluster["plan"]["spec"]["hosts"]):
        try:
            invoke(row["host"], ["sudo", "-n", "true"])
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            user = row["host"].split("@", 1)[0] if "@" in row["host"] else "USER"
            missing.append({"rank": rank, "host": row["host"], "error": str(error).splitlines()[-1][:200],
                            "fix": f"ssh -t {row['host']} \"echo '{user} ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/{user} "
                                   f"&& sudo chmod 440 /etc/sudoers.d/{user} && sudo visudo -cf /etc/sudoers.d/{user}\""})
    if missing:
        raise NeedsInput("Some Sparks need noninteractive SSH and sudo for the installer. Run each listed fix once "
                         "(it asks for that Spark's password), then repeat the install. Nothing has been changed.",
                         field="access", details={"hosts": missing})


def rebuild(cluster, found):
    """The plan of the recorded cluster from fresh inspect documents; Node A stays rank 0."""
    return topology.build_spec(found, cluster["plan"]["spec"]["hosts"][0]["node_id"], name=cluster["name"],
                               fabric_cidr=cluster["plan"].get("fabric_cidr", "198.18.0.0/21"),
                               preserve_control=cluster["plan"]["spec"].get("preserve_control_ipv6", False))


def refresh_cluster(cluster):
    """Re-observe cables without reconfiguring the adopted fabric.

    The ConnectX hairpin setting is not verified here: workers that run an
    older SparkRing, or a ring rebooted before its Sparks were armed, are
    handled by the hairpin step (``hairpin_ring.requirement`` and ``ensure``).
    A port whose RoCE GID index 3 lacks its address passes this check: the
    model installation re-adds the address.
    """
    hosts = cluster["plan"]["spec"]["hosts"]
    found = controller.collect([h["host"] for h in hosts])
    require_head(cluster)
    plan = rebuild(cluster, found)
    if [h["node_id"] for h in plan["spec"]["hosts"]] != [h["node_id"] for h in hosts]:
        raise NeedsInput("Cable order changed. Run sparkring setup to review the new fabric first.", field="fabric")
    deploy_network.verify_network(plan["spec"], plan["inventory"]["hosts"], hairpin=False, stale_gids=True)
    return {**cluster, "plan": plan}


def planned_head(cluster):
    """The refreshed cluster with Node A at this controller's revision, for ``--plan``.

    Node A's inspection reports the package installed there. ``install.sh
    --plan`` runs this controller from the package it built, extracted and not
    installed, when Node A has another version; the installation that such a
    plan describes runs after that package is installed on Node A. The ConnectX
    hairpin step compares every Spark with Node A's revision, so the plan uses
    this controller's. When Node A runs this controller's package, the cluster
    is returned unchanged.
    """
    revision = distribution.identity(installer.ROOT)
    nodes = cluster["plan"]["nodes"]
    if nodes[0].get("revision") == revision:
        return cluster
    return {**cluster, "plan": {**cluster["plan"], "nodes": [{**nodes[0], "revision": revision}, *nodes[1:]]}}


def choose_profile(value, count, interactive):
    if not value:
        choices = [r for r in models.catalog() if r["automated"] and r["nodes"] == count]
        if not interactive:
            raise NeedsInput("Select an exact profile with --profile.", field="profile", details={"choices": [r["profile"] for r in choices]})
        for number, choice in enumerate(choices, 1):
            print(f"{number}. {choice['profile']}")
        selected = input("Model profile number: ").strip()
        if not selected.isdigit() or not 1 <= int(selected) <= len(choices):
            raise NeedsInput("Choose one of the listed profiles.", field="profile")
        value = choices[int(selected) - 1]["profile"]
    return models.select(value, count)


def _error_text(error):
    """The last non-empty line of an error, which names its cause, at most 300 characters."""
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    return (lines[-1] if lines else type(error).__name__)[:300]


def survey_cluster(rows, pins, *, named, ignore_local, operator, owned=None, invoke=None):
    """Survey every Spark for copies of the pinned checkpoint, all ranks at once.

    ``rows`` are the deployment rows in rank order (``host``, ``model``,
    ``cache`` and ``reuse_verified_model``). Each rank's survey reports the
    state of the SparkRing checkpoint directory that rank uses (the row's
    ``model``, or ``owned`` for a row that serves a named copy in place) and
    the device of the row's compile cache, and classifies the named paths that
    apply to that rank: the ``--model-path`` values given for every Spark or
    for that rank, and the copy a row serves in place. The survey source
    carries Node A's pins, so a worker whose installed package differs looks
    for the same files.

    Returns one entry per rank: the ``sparkring-checkpoint-survey/v1`` document,
    or ``{"error": text}`` when SSH failed, timed out or returned anything else
    than a survey of this checkpoint. A failure on one rank never stops the
    others; the plan treats that rank as holding no copy.
    """
    invoke = invoke or discovery.ssh
    entries = checkpoint_plan.named_paths(named, len(rows))

    def one(rank):
        row = rows[rank]
        paths = [entry["path"] for entry in entries if entry["rank"] in (None, rank)]
        if row.get("reuse_verified_model") and row["model"] not in paths:
            paths.append(row["model"])
        target = owned if row.get("reuse_verified_model") and owned else row["model"]
        source = checkpoint_search.probe_source(pins, checkpoint_search.options(
            owned=target, operator=operator, named=paths, ignore_local=ignore_local, cache=row.get("cache")))
        try:
            document = json.loads(invoke(row["host"], list(SURVEY_COMMAND), data=source, timeout=SURVEY_TIMEOUT))
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
            return {"error": _error_text(error)}
        if (not isinstance(document, dict) or document.get("schema") != checkpoint_plan.SURVEY_SCHEMA
                or (document.get("repository"), document.get("revision")) != (pins["repository"], pins["revision"])):
            return {"error": "invalid survey output"}
        return document

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:
        return list(pool.map(one, range(len(rows))))


def inspect_sparks(rows, image_ids, caches, *, invoke=None):
    """Each Spark's nearest present serving image and its compile cache directories, all ranks at once.

    ``image_ids`` lists the lock's image and the images it derives from,
    nearest first (``install_space.lineage``); ``caches`` holds, per rank,
    the paths of the deployment's compile cache directories on that Spark.
    Returns one ``install_space.inspect_spark`` result per rank, or ``{"error":
    text}`` when SSH failed or returned anything else; the plan then reserves
    the whole image and the cache allowance on that Spark
    (``install_space.image_need`` and ``cache_need``).
    """
    invoke = invoke or discovery.ssh

    def one(rank):
        source = install_space.probe_source(image_ids, caches[rank])
        try:
            document = json.loads(invoke(rows[rank]["host"], list(install_space.PROBE_COMMAND), data=source,
                                         timeout=install_space.PROBE_TIMEOUT))
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
            return {"error": _error_text(error)}
        if not isinstance(document, dict) or not isinstance(document.get("caches", {}), dict):
            return {"error": "invalid inspection output"}
        return document

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:
        return list(pool.map(one, range(len(rows))))


def install_needs(lineage, found, policy):
    """``(images, caches)``: each rank's image and compile cache needs for ``checkpoint_plan.plan``.

    ``lineage`` is ``install_space.lineage`` of the image lock, ``found`` the
    ``inspect_sparks`` results and ``policy`` ``checkpoint_plan.storage_policy()``.
    """
    images = [install_space.image_need(lineage, item, policy["image_bytes"]) for item in found]
    caches = [install_space.cache_need({} if item.get("error") else item.get("caches") or {}, policy["cache_bytes"])
              for item in found]
    return images, caches


def cache_paths(rows, card, image_id):
    """The deployment's compile cache directories on each rank; empty where they cannot be named."""
    try:
        names = install_space.cache_names(card, image_id)
    except (OSError, ValueError, KeyError, TypeError):
        names = []
    return [[str(PurePosixPath(row["cache"]) / name) for name in names] if row.get("cache") else [] for row in rows]


def _device(path):
    """``st_dev`` of the filesystem that holds ``path`` or its nearest existing ancestor, or None."""
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return os.stat(path).st_dev
    except OSError:
        return None


def retained_deployments(state_root, candidate, rows):
    """The other deployments recorded on Node A, as the checkpoint plan and adoption use them.

    Returns ``(models, receipts)``. ``models`` maps each other deployment's
    directory name to the model path it uses on each Spark of ``rows``, or None
    where it has no rank on that Spark; the plan names the deployments whose
    receipts linking will refresh. ``receipts`` maps each rank of ``rows`` to
    ``{"path", "deployment"}``: the model receipt
    ``<workspace>/installer/model.json`` and the ID of every other deployment
    with a rank on that Spark. Adoption on that Spark refreshes the entries of
    those receipts that record an inode it links; it accepts only regular files
    inside a workspace whose owner record names that deployment. A lock without
    an ID is skipped, because its receipt cannot be bound to it.
    """
    models, receipts = {}, {rank: [] for rank in range(len(rows))}
    base = Path(state_root) / "deployments"
    candidate = Path(candidate).resolve()
    for directory in sorted(base.iterdir()) if base.is_dir() else ():
        if not directory.is_dir() or directory.resolve() == candidate:
            continue
        try:
            lock = installer.read(directory / "deployment.lock.json")
            by_host = {row["host"]: row["model"] for row in lock["site"]["ranks"]}
            receipt = {"path": str(PurePosixPath(lock["site"]["workspace"]) / "installer" / "model.json"),
                       "deployment": lock["id"]}
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not isinstance(receipt["deployment"], str):
            continue
        paths = [by_host.get(row["host"]) for row in rows]
        if any(path is not None for path in paths):
            models[directory.name] = paths
            for rank, path in enumerate(paths):
                if path is not None:
                    receipts[rank].append(receipt)
    return models, receipts


def _select_mesh(site, cluster, profile, hint, **options):
    """``native_mesh.select``; ``hint`` is appended to its refusals (why mesh services may not run)."""
    try:
        return native_mesh.select(site, cluster, profile, invoke=discovery.ssh, **options)
    except NeedsInput:
        raise
    except ValueError as error:
        if hint:
            raise ValueError(str(error) + hint) from error
        raise


def select_deployment(args, cluster, state_root, *, mesh_hint=""):
    """Choose the deployment for this request, survey every Spark and plan its checkpoint.

    Returns ``(directory, lock, plan)``, where ``plan`` is the
    ``sparkring-checkpoint-plan/v1`` document, which the caller prints; this
    function prints only the profile choices in a terminal and the line
    announcing the survey. The survey runs on every call, also for a
    deployment that already exists, which keeps each rank's mode as its lock
    records it. A new deployment uses the cluster's SparkRing checkpoint
    directory on every rank, except where a named path holds exactly the pinned
    files on another filesystem than that directory: such a copy is served in
    place and never written. Named paths are part of the deployment request;
    ``--ignore-local-copies`` is not. A new deployment whose plan has problems
    is not recorded and ``lock`` is None, so that repeating the command, for
    example after the operator completed a named copy, plans it again.

    ``mesh_hint`` is appended to a native-mesh refusal, for example when Sparks
    lack the ConnectX hairpin setting, so their mesh services cannot start.
    """
    count = len(cluster["plan"]["nodes"])
    profile = choose_profile(args.profile, count, not args.json and sys.stdin.isatty())
    image = installer_image.for_profile(profile, installer.read(args.image_lock) if args.image_lock else None)
    try:
        named = checkpoint_plan.named_paths(args.model_path, count)
    except ValueError as error:
        raise NeedsInput(str(error) + ". Nothing has been changed.", field="model_path") from None
    try:
        card = installer.setup.selection(profile, args.checkpoint)
    except ValueError as error:
        raise NeedsInput(str(error) + ". Nothing has been changed.", field="checkpoint_name") from None
    # A checkpoint is requested by its listed name: an alias requests the same
    # deployment as that name, and the profile's default the same as no flag.
    checkpoint = (card["target_variant"] if args.checkpoint is not None
                  and card["target_variant"] != installer.setup.selection(profile)["target_variant"] else None)
    request = {"profile": profile, "image_runtime": image, "source": distribution.identity(installer.ROOT),
               "model_path": named or None, "cache_path": args.cache_path,
               "nodes": cluster["plan"]["spec"]["hosts"], "api_address": cluster.get("api_address")}
    if checkpoint is not None:
        request["checkpoint"] = checkpoint
    # Serving settings are part of the request, so other settings install
    # another deployment; without any, the request is unchanged.
    requested = serving_settings.from_arguments(args)
    if requested:
        # A setting whose vLLM flag the profile does not set, or a value above
        # the selected checkpoint's limit, is refused before any Spark is surveyed.
        from runtime.common import qwen_flash_next
        configuration = profiles.read_json(installer.ROOT / card["configuration"])
        if "checkpoints" in configuration:
            configuration = qwen_flash_next.checkpoint_settings(configuration, card["target_variant"])
        serving_settings.apply(configuration.get("vllm_args", []), requested)
        request["serving"] = requested
    instance = "i" + hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
    directory = state_root / "deployments" / (profile + "-" + instance)
    pins = installer.checkpoint_pins(card)
    owned = installer.checkpoint_directory(cluster, card)
    locked = directory.exists()
    if locked:
        lock = installer.load(directory)
        rows = lock["site"]["ranks"]
        selection = lock["selection"]
        image_id = selection["image_id"]
        image_lock = lock.get("image_runtime") or {"name": selection["release"], "image_id": image_id}
    else:
        site = controller.model_site(cluster, profile, instance)
        for row in site["hosts"]:
            row["model"] = owned
            # One compile/tuning cache per cluster. Containers use a subdirectory
            # keyed by model family, image and checkpoint revision, so repeated or
            # alternating installs reuse earlier kernel tuning.
            row["cache"] = args.cache_path or "/srv/sparkring/" + cluster["name"] + "/cache"
        rows = [{"rank": rank, "host": row["host"], "model": owned, "cache": row["cache"],
                 "reuse_verified_model": False} for rank, row in enumerate(site["hosts"])]
        selection, image_lock, image_id = card, image, image["image_id"]
    operator = os.environ.get("SUDO_USER") or "root"
    print(checkpoint_plan.announce(pins, len(rows)))
    lineage = install_space.lineage(image_lock)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        found = pool.submit(inspect_sparks, rows, [entry["image_id"] for entry in lineage],
                            cache_paths(rows, selection, image_id))
        surveys = survey_cluster(rows, pins, named=named, ignore_local=args.ignore_local_copies,
                                 operator=operator, owned=owned)
        found = found.result()
    images, caches = install_needs(lineage, found, checkpoint_plan.storage_policy())
    retained, _ = retained_deployments(state_root, directory, rows)
    plan = checkpoint_plan.plan(pins, surveys, rows, named=named, ignore_local=args.ignore_local_copies,
                                operator=operator, images=images, caches=caches, relay_device=_device(directory),
                                retained=retained, locked=locked,
                                request={"profile": profile, "checkpoint": checkpoint, "cache_path": args.cache_path,
                                         "image_lock": str(args.image_lock) if args.image_lock else None})
    if not locked:
        if plan["problems"]:
            return directory, None, plan
        for row, entry in zip(site["hosts"], plan["nodes"], strict=True):
            if entry["mode"] == "in-place":
                row.update(model=entry["path"], reuse_verified_model=True)
        if profile in installer.compose.TP4_PROFILES:
            site = _select_mesh(site, cluster, profile, mesh_hint)
        elif installer.backend({"profile": profile}) == "glm-managed":
            site = _select_mesh(site, cluster, profile, mesh_hint, existing_only=True)
        lock = installer.init(directory, profile, site, variant=checkpoint, image_runtime=image, settings=requested)
    return directory, lock, plan


def saved_plan(directory):
    """The checkpoint plan saved with the deployment, or None."""
    try:
        value = installer.read(Path(directory) / PLAN_FILE)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and value.get("schema") == checkpoint_plan.SCHEMA else None


def save_plan(directory, plan):
    node.save(directory, PLAN_FILE, plan, mode=0o600)


def save_refused(directory, plan):
    """Keep a plan that the reviewed plan's bound refused, beside the reviewed plan, which stays the bound."""
    node.save(directory, REFUSED_FILE, plan, mode=0o600)


def forget_refused(directory):
    try:
        (Path(directory) / REFUSED_FILE).unlink()
    except FileNotFoundError:
        pass


def bounded(fresh, reviewed):
    """The fresh plan, bounded by the downloads and per-Spark writes of the reviewed plan.

    Adoption follows the fresh plan, whose sources and file identities are the
    ones just surveyed. The guard that stops unplanned downloads and writes
    compares with the reviewed plan's ``hub_files`` and each Spark's
    ``write_bytes``; ``envelope`` has confirmed that the fresh plan stays within
    them.
    """
    result = copy.deepcopy(fresh)
    result.update(hub_files=list(reviewed["hub_files"]), hub_bytes=reviewed["hub_bytes"])
    for mine, theirs in zip(result["nodes"], reviewed["nodes"], strict=True):
        mine["write_bytes"] = theirs["write_bytes"]
    return result


def approve(fresh, reviewed, *, command_line, setup_only, interactive, request, question="Apply this installation?",
            default=True, refusal=None):
    """Approve the checkpoint plan printed in this run; returns ``(approved plan, approval)``.

    ``approval`` is ``command-line`` for a ``--yes`` typed in this run,
    ``reviewed-plan`` when such a run stays within the reviewed plan
    (``reviewed``: saved by ``--plan`` or approved at a terminal), ``setup``
    when setup's approval of a first installation covers it, and ``prompt`` for
    a confirmation in a terminal.
    Setup's approval does not cover a plan with attention items (a download
    larger than 1 GiB, or a Spark whose search failed or stopped at its limit);
    that plan gets its own prompt, which Enter cancels. ``NeedsInput`` is
    raised when nothing approves the plan. Messages suggest the plan's
    ``command``, which repeats the deployment request.
    ``question`` and ``default`` are the terminal prompt of an installation
    that setup did not approve, and ``refusal`` replaces the message raised
    without a terminal, for example when the installation also applies the
    ConnectX hairpin setting.
    """
    command = fresh.get("command") or checkpoint_plan.COMMAND
    if command_line and reviewed is not None:
        items = checkpoint_plan.envelope(reviewed, fresh)
        if items:
            raise NeedsInput(checkpoint_plan.envelope_message(items, command), field="checkpoint",
                             details={"differences": items})
        return bounded(fresh, reviewed), "reviewed-plan"
    if command_line:
        return fresh, "command-line"
    if setup_only:
        items = checkpoint_plan.attention(fresh)
        if not items:
            return fresh, "setup"
        if not interactive:
            raise NeedsInput(checkpoint_plan.attention_message(items, command), field="checkpoint",
                             details={"attention": items})
        try:
            controller.confirm("Proceed with this checkpoint plan?")
        except ValueError:
            raise ValueError(f"Cancelled before any checkpoint or model change; setup is complete. Repeat {command} to "
                             "review the checkpoint plan again.") from None
        return fresh, "prompt"
    if not interactive:
        raise NeedsInput(refusal or f"Approve this installation with {command} --yes, or review its plan first with "
                         f"{command} --plan. Nothing has been changed.", field="approval", details=request)
    controller.confirm(question, default=default)
    return fresh, "prompt"


def check_workloads(directory, previous, *, stop=None):
    """Reject unrelated GPU work before any planned service interruption.

    ``stop(host, names)`` approves stopping unrelated GPU containers; they are
    stopped, never removed. GPU processes outside containers always need input.
    """
    from scripts.installer_runner import PROBE, check_facts, check_workloads, ssh
    lock = installer.load(directory)
    allowed = installer.read(previous / "deployment.lock.json") if previous else lock
    managed = bool(previous and (previous / "managed/runtime/prepared.json").exists())
    for row in lock["site"]["ranks"]:
        for attempt in (0, 1):
            facts = json.loads(ssh(row["host"], ["python3", "-I", "-B", "-c", PROBE], timeout=120))
            check_facts(facts, row)
            if any(_permits(check_workloads, facts, permitted, row["rank"], managed) for permitted in (allowed, lock)):
                break
            names = [entry["name"] for entry in facts["gpu_containers"]
                     if entry["labels"].get("io.sparkring.deployment") not in (allowed["id"], lock["id"])]
            if attempt or stop is None or not names:
                raise NeedsInput("An unrelated GPU workload is running. Stop it, or repeat with --stop-workloads to stop "
                                 "(not remove) the listed containers.", field="workload",
                                 details={"rank": row["rank"], "host": row["host"], "containers": names})
            stop(row["host"], names)
            for name in names:
                ssh(row["host"], ["docker", "stop", "--time", "60", name], timeout=180)


def _permits(check, facts, permitted, rank, managed):
    try:
        check(facts, permitted, rank, managed_prepared=managed)
        return True
    except ValueError:
        return False


def check_managed_namespace(lock):
    if lock["backend"] != "glm-managed":
        return
    from runtime.host.managed_slot import inspect_slot
    for row in lock["site"]["ranks"]:
        code = inspect.getsource(inspect_slot) + "\nimport json\nprint(json.dumps(inspect_slot(" + repr(lock["site"]["name"]) + "," + repr(lock["selection"]["image_id"]) + "," + str(row["rank"]) + ")))\n"
        observed = json.loads(discovery.ssh(row["host"], ["sudo", "-n", "python3", "-I", "-c", code]))
        if not observed["available"]:
            raise NeedsInput("GLM's managed-service paths belong to another deployment. A reviewed mesh migration or existing-mesh adapter is required; the current model and fabric have not been changed.",
                             field="fabric", details={"rank": row["rank"], "occupied": observed["occupied"]})


def serving(directory, *, runner=None):
    """Whether the deployment in ``directory`` runs on every Spark and, on four Sparks, passes each ring check.

    A pair checks RoCE GID index 3 instead of a ring. Read-only; a check that
    cannot run counts as not serving. An installation of the active deployment
    stops it on every Spark first when it does not serve: after one Spark
    restarted, the others keep a model that waits for it and that holds the
    RoCE GID entries which the installation then re-adds. Managed GLM
    deployments keep their own lifecycle and count as serving.
    """
    if runner is None:
        from scripts.installer_runner import Runner
        runner = Runner(directory)
    lock = runner.lock
    if lock["backend"] == "glm-managed":
        return True
    ranks = lock["site"]["ranks"]
    checks = {2: "gid-check", 4: "ring-check"} if lock["backend"] == "compose" else {}
    operations = ["running"] + ([checks[len(ranks)]] if len(ranks) in checks else [])
    for row in ranks:
        for operation in operations:
            try:
                runner.remote(row["rank"], operation)
            except (RuntimeError, ValueError, OSError, subprocess.SubprocessError):
                return False
    return True


def hairpin_step(cluster, assets, state_root, record, *, restart_approved=True):
    """Apply the ConnectX hairpin setting on the approved ring, then verify the whole network.

    Returns the cluster with the resulting plan; ``record`` receives the
    receipt. Runs before the model transaction, so a refusal or failure
    creates no transaction record. ``restart_approved`` is False when the
    approval text named no driver restart. Mesh units that the start check
    refused are not started here: the model installation that follows owns
    the mesh.
    """
    directory = state_root / "hairpin" / str(time.time_ns())
    try:
        plan = hairpin_ring.ensure(cluster["plan"], approved=True, restart_approved=restart_approved,
                                   inspect=controller.collect, rebuild=lambda found: rebuild(cluster, found),
                                   update_workers=assets.sync_packages, directory=directory, record=record)
    except NeedsInput as error:
        if record.get("path"):
            error.details = {**error.details, "receipt": record["path"]}
        raise
    deploy_network.verify_network(plan["spec"], plan["inventory"]["hosts"], stale_gids=True)
    return {**cluster, "plan": plan}


def download_limit(args):
    """Bytes per second for checkpoint downloads: ``--download-limit``, else the ``--env`` file's value; None for none.

    The ``--env`` file's ``SPARKRING_DOWNLOAD_LIMIT`` is read on every run
    that names the file, unlike its setup keys.
    """
    text = getattr(args, "download_limit", None)
    try:
        if text is None and getattr(args, "env", None):
            text = settings.load(args.env)["SPARKRING_DOWNLOAD_LIMIT"]
        return settings.download_limit(text or "none")
    except (OSError, ValueError) as error:
        raise NeedsInput(f"{error}. Nothing has been changed.", field="download_limit") from None


def retain_preference(args):
    """The ``--env`` file's ``SPARKRING_RETAIN_DEPLOYMENTS``, or None when the run names no file or the file does not set it.

    Automatic release (``runtime/host/retention.py``) saves the value after a
    successful installation, so later installations and ``sparkring up`` keep it.
    """
    if not getattr(args, "env", None):
        return None
    try:
        text = settings.load(args.env)["SPARKRING_RETAIN_DEPLOYMENTS"]
        if text:
            settings.retain_deployments(text)
    except (OSError, ValueError) as error:
        raise NeedsInput(f"{error}. Nothing has been changed.", field="retain_deployments") from None
    return text or None


DASHBOARD = "/v1/sparkring/status/view"
# The summary card's Recovery line for each value of the result's "recovery".
RECOVERY_TEXT = {"on": "restarts the model if a Spark stops serving; turn off: sudo sparkring recover off",
                 "off": "off; turn on: sudo sparkring recover on",
                 "unsupported": "not available for this model's backend; restart it by hand",
                 "unrecorded": "not set up (see the warning above); sudo sparkring recover on retries"}
STOP_COMMAND = "sudo sparkring down --execute"
REMOVE_COMMAND = "sudo apt remove sparkring"


def switch_back_command(previous):
    """The command that reinstalls the deployment in ``previous``, or None when it cannot be read.

    A profile that the catalog replaced (``profiles.REPLACED``) is named by its
    replacement, because the installer refuses the replaced ID.
    """
    if not previous:
        return None
    plan = saved_plan(previous)
    if plan and plan.get("command"):
        command = plan["command"]
    else:
        try:
            profile = installer.read(Path(previous) / "deployment.lock.json")["selection"]["profile"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        command = checkpoint_plan.install_command({"profile": profile})
    for old, new in profiles.REPLACED.items():
        command = command.replace(f"--profile {old}", f"--profile {new}")
    return command


def summary(lock, connection, previous):
    """The additive result fields that tell the operator how to use and leave the model.

    ``connection`` holds ``api_url`` and ``model`` (``installer.connection``);
    ``previous`` is the deployment directory the installation replaced, or None.

    ``dashboard_url`` is null for a model whose image has no status dashboard.
    ``commands.switch_back`` is null when no other model ran before.
    """
    base = connection["api_url"].removesuffix("/v1")
    body = json.dumps({"model": connection["model"], "messages": [{"role": "user", "content": "Hello"}]})
    return {"dashboard_url": base + DASHBOARD if "image_runtime" in lock else None,
            "example_request": (f"curl {connection['api_url']}/chat/completions "
                                f"-H 'Content-Type: application/json' -d '{body}'"),
            "commands": {"switch_back": switch_back_command(previous), "stop": STOP_COMMAND,
                         "remove": REMOVE_COMMAND}}


def summary_lines(result):
    """The card printed after ``Model ready:``; empty when the result lacks the summary fields."""
    if "example_request" not in result:
        return []
    commands = result.get("commands") or {}
    rows = [("Model", result.get("model")), ("API", result.get("api_url")), ("Dashboard", result.get("dashboard_url")),
            ("Try it", result["example_request"]), ("Switch back", commands.get("switch_back")),
            ("Stop", commands.get("stop")),
            ("Recovery", RECOVERY_TEXT.get(result.get("recovery"))),
            ("Uninstall", f"stop the model, then {commands['remove']} on each Spark" if commands.get("remove") else None)]
    return [f"  {name + ':':<13}{value}" for name, value in rows if value]


def execute(args):
    state_root = controller.STATE
    require_head()
    limit = download_limit(args)
    retain = retain_preference(args)
    interactive = not args.json and sys.stdin.isatty()
    if args.allow_driver_reload:
        print("--allow-driver-reload: " + controller.ALLOW_DRIVER_RELOAD)
    # Checked before setup changes anything; the node numbers are checked
    # against the cluster once it is known.
    try:
        checkpoint_plan.named_paths(args.model_path)
    except ValueError as error:
        raise NeedsInput(str(error) + ". Nothing has been changed.", field="model_path") from None
    command_line = args.yes
    with process_lock.hold(state_root / "install.lock"):
        if not (state_root / "cluster.json").exists():
            if args.plan:
                raise NeedsInput("Run sparkring setup --plan to discover this unconfigured cluster.", field="setup")
            from runtime.host import single_uplink
            options = ((["--env", str(args.env)] if args.env else []) + (["--yes"] if args.yes else [])
                       + (["--stop-workloads"] if args.stop_workloads else []))
            follow = "then install " + (args.profile or "the model profile you choose") + " and start it"
            if not args.yes and not interactive:
                raise NeedsInput("First installation requires setup approval. Review sparkring setup --plan, then use --yes.",
                                 field="approval", details={"scope": single_uplink.scope(options, follow=follow)})
            if single_uplink.main(options, follow=follow):
                raise ValueError("Cluster setup did not complete")
            # The setup approval listed this installation as its final step.
            # It did not show the checkpoint plan, which approve() checks.
            args.yes = True
        setup_only = args.yes and not command_line
        cluster = installer.read(state_root / "cluster.json")
        check_access(cluster)
        cluster = refresh_cluster(cluster)
        if args.plan:
            cluster = planned_head(cluster)
        hairpin = hairpin_ring.requirement(cluster["plan"])
        needs_hairpin = hairpin_ring.required(hairpin)
        # Printed before the deployment is selected, so that a native-mesh
        # refusal caused by Sparks without the setting follows its listing.
        for line in hairpin_ring.consent_lines(cluster["plan"], hairpin):
            print(line)
        mesh_hint = hairpin_ring.mesh_hint(hairpin)
        directory, lock, checkpoint = select_deployment(args, cluster, state_root, mesh_hint=mesh_hint)
        if lock is not None:
            check_managed_namespace(lock)
        previous = rollout.active(state_root)
        replaces = str(previous) if previous and previous != directory else None
        print(f"Install {checkpoint['profile']} on {len(checkpoint['nodes'])} Sparks.")
        print("Update workers and prepare assets; then " + ("replace the current model." if replaces else "start the selected model."))
        if lock is not None and lock.get("serving"):
            base = installer.specifications(dict(lock, serving={}), only_rank=0)[0].command
            print("Serving settings: " + "; ".join(serving_settings.describe(lock["serving"], base)))
            for line in serving_settings.warnings(lock["serving"], base):
                print("Warning: " + line)
        if lock is not None and "native_mesh" in lock["site_input"]:
            if previous:
                raise NeedsInput("The replacement needs native fabric configuration. Review sparkring setup before "
                                 "replacing a running deployment." + mesh_hint, field="fabric")
            print("Configure and start the profile's supervised native fabric.")
        if limit and checkpoint["hub_files"]:
            print(f"Downloads from huggingface.co are limited to {progress.rate_text(limit)}.")
        # The plan's last line says what is downloaded, so it stays directly
        # above any prompt.
        for line in checkpoint_plan.describe(checkpoint):
            print(line)
        # A reviewed plan (saved by --plan or approved at a terminal) bounds
        # every --yes run until the next --plan run or terminal approval.
        previous_plan = saved_plan(directory) if lock is not None else None
        reviewed = previous_plan if previous_plan and previous_plan.get("reviewed") else None
        problems = checkpoint["problems"]
        if problems:
            if lock is not None:
                if args.plan:
                    save_plan(directory, {**checkpoint, "reviewed": True})
                    forget_refused(directory)
                elif reviewed is None:
                    save_plan(directory, checkpoint)
                else:
                    save_refused(directory, checkpoint)
            raise NeedsInput("\n".join(problem["message"] for problem in problems), field=problems[0]["field"],
                             details={"problems": problems})
        steps = ["verify-fabric", "update-workers", "prepare-images-and-checkpoints", "switch-model", "verify-serving"]
        if needs_hairpin:
            steps.insert(steps.index("update-workers") + 1, "apply-hairpin-setting")
        plan = {"schema": "sparkring-install-result/v1", "state": "planned", "deployment": str(directory),
                "profile": lock["selection"]["profile"], "image_id": lock["selection"]["image_id"],
                "nodes": len(lock["site"]["ranks"]), "replaces": replaces, "steps": steps,
                "serving": lock.get("serving") or {}, **installer.connection(lock)}
        if hairpin:
            plan["hairpin"] = {"required": needs_hairpin, "ranks": hairpin_ring.rank_rows(hairpin, cluster["plan"])}
        if limit:
            plan["download_limit_bps"] = limit * 8
        plan["checkpoint"] = checkpoint_plan.summary(checkpoint)
        if args.plan:
            save_plan(directory, {**checkpoint, "reviewed": True})
            forget_refused(directory)
            plan["checkpoint"]["reviewed"] = True
            return plan
        consent = {}
        if needs_hairpin:
            # Sparks whose reload statistics are unavailable stop the step
            # before the question, not after the answer.
            hairpin_ring.refuse_unknown(hairpin)
            consent = {"question": "Apply the ConnectX hairpin setting and this installation?",
                       "default": hairpin_ring.consent_default(cluster["plan"], hairpin),
                       "refusal": hairpin_ring.m7(hairpin)}
        try:
            approved, approval = approve(checkpoint, reviewed, command_line=command_line, setup_only=setup_only,
                                         interactive=interactive, request=plan, **consent)
        except NeedsInput as refusal:
            # The reviewed plan stays the bound: a plan that leaves it is kept
            # beside it, so a repeated --yes is refused again until --plan.
            if reviewed is None:
                save_plan(directory, checkpoint)
            elif refusal.field == "checkpoint":
                save_refused(directory, checkpoint)
            raise
        checkpoint["approval"] = approved["approval"] = approval
        plan["checkpoint"]["approval"] = approval
        forget_refused(directory)
        if approval == "prompt":
            # The operator saw this plan in full and approved it at the terminal.
            save_plan(directory, {**checkpoint, "reviewed": True})
            plan["checkpoint"]["reviewed"] = True
        elif approval != "reviewed-plan":
            save_plan(directory, checkpoint)
        _, receipts = retained_deployments(state_root, directory, lock["site"]["ranks"])

        def approve_stop(host, names):
            print(f"{host}: stopping unrelated GPU containers (not removing them): " + ", ".join(names))
            if not args.stop_workloads:
                controller.confirm("Stop these containers before installing?")
        check_workloads(directory, previous, stop=approve_stop if args.stop_workloads or interactive else None)
        transport = fabric_ssh.Transport(cluster, state_root / "bulk-ssh")
        plan["transfer"] = transport.verify()
        assets = install_assets.Assets(transport, directory / "assets", download_limit=limit)
        if needs_hairpin:
            record = {}
            cluster = hairpin_step(cluster, assets, state_root, record,
                                   restart_approved=hairpin_ring.restart_expected(hairpin))
            plan["hairpin"].update(ranks=hairpin_ring.result_ranks(hairpin, cluster["plan"], record),
                                   receipt=record.get("path"))
        cache = state_root / "retained-sources"

        def apply(path, operation):
            return retained_source.apply(path, operation, cache=cache)

        def prepare(path):
            if previous and previous != path:
                # Verify the rollback controller bundle before any downtime.
                retained_source.checkout(previous, cache)
            assets.sync_packages()
            # Image distribution starts at once and overlaps the prerequisite
            # and source phases. The checkpoint phase waits for it: a download
            # runs inside the serving image, and every checkpoint write is
            # checked against free space once the image is in place.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                images = pool.submit(assets.images, lock["selection"])
                runner = assets.runner(path, previous, images, plan=approved, receipts=receipts)
                request = failure = None
                try:
                    installer.apply(path, "prepare", runner=runner, execute=True)
                except RuntimeError as error:
                    # The checkpoint phase reports a request for input as a
                    # failed action; the runner keeps the request itself.
                    request = getattr(runner, "needs_input", None)
                    failure = error
                if request is not None:
                    # An image failure never replaces the request; it is added to it.
                    try:
                        images.result()
                    except Exception as error:  # noqa: BLE001 - reported with the request
                        request.details["image_error"] = str(error)
                    raise request
                # An image failure is the cause when the checkpoint phase waited for it.
                images.result()
                if failure is not None:
                    # The phase failure names only the phase; the checkpoint
                    # preparation's own error names the Spark and the cause.
                    cause = getattr(runner, "models_error", None)
                    if cause:
                        raise RuntimeError(cause) from failure
                    raise failure
            check_workloads(path, previous)

        # check_workloads has confirmed that only the active or candidate
        # deployment uses the GPUs, so an abandoned failed switch can be replaced.
        # A state that recovery's own unfinished attempt left is not recovery's
        # once this installation changes it.
        for path in {previous, directory} - {None}:
            recovery.forget_attempt(path)
        # A candidate whose own start did not complete, for example a first
        # installation that failed its readiness check, stops before it is
        # prepared again.
        result = rollout.execute(directory, previous, state_root=state_root, prepare=prepare, apply=apply,
                                 verify=lambda path: apply(path, "verify"), supersede=True, serving=serving,
                                 unfinished=installer.unfinished)
        # Records the deployment's choice (on unless --no-auto-recover, also
        # after sparkring recover off), resets failures and restarts, records
        # the generation's boots and enables the timer.
        record = recovery.started(directory, enabled=not args.no_auto_recover)
        plan["recovery"] = ("unrecorded" if record is None else "unsupported" if not record.get("supported", True)
                            else "on" if record["enabled"] else "off")
        plan["auto_recover"] = plan["recovery"] == "on"
        # Releases what older deployments hold on the Sparks; never fails the installation.
        plan["retention"] = retention.after_operation(state_root, discovery.ssh, preference=retain)
        try:
            plan["checkpoint"]["result"] = installer.read(directory / "assets/checkpoint-result.json")
        except (OSError, ValueError):
            pass
        return {**plan, "state": "complete", "transaction": result, "log": str(progress.directory() / "install.log"),
                **summary(lock, plan, replaces)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring install", description="Set up this Spark ring and deploy one exact model profile.")
    parser.add_argument("--profile", help="exact profile from sparkring models; prompted in a terminal")
    parser.add_argument("--image-lock", type=Path, help="development image lock replacing the shared installer image")
    parser.add_argument("--model-path", action="append", metavar="[N=]PATH",
                        help="a local copy of the checkpoint; PATH for every Spark or N=PATH for Node N (repeatable); "
                             "SparkRing links or copies its files into its own directory, or serves an exact copy on "
                             "another filesystem read-only, and never writes to it")
    parser.add_argument("--checkpoint", metavar="NAME",
                        help="another checkpoint the profile lists, with the settings it needs; default: the profile's own")
    parser.add_argument("--ignore-local-copies", action="store_true",
                        help="use only SparkRing's own checkpoint directories and named copies")
    parser.add_argument("--cache-path", help="optional local writable cache path on each Spark")
    parser.add_argument("--env", type=Path, help="optional literal preferences: setup keys on first installation, the "
                                                 "download limit and retained deployments on every run")
    parser.add_argument("--plan", action="store_true",
                        help="print and save the setup, checkpoint and model plan without changing any Spark; a later "
                             "--yes stays within a saved plan")
    parser.add_argument("--yes", action="store_true",
                        help="approve the displayed setup, checkpoint plan, the listed ConnectX driver restarts on an "
                             "idle ring, and model replacement; SSH trust is still required")
    parser.add_argument("--json", action="store_true", help="emit one JSON result on stdout; progress stays on stderr")
    parser.add_argument("--stop-workloads", action="store_true",
                        help="stop (never remove) running GPU containers that are not SparkRing's current deployment")
    parser.add_argument("--allow-driver-reload", action="store_true", help=controller.ALLOW_DRIVER_RELOAD)
    parser.add_argument("--download-limit", metavar="RATE",
                        help="cap checkpoint downloads from huggingface.co, in bits per second: 850Mbit, 2Gbit or none; "
                             "default: SPARKRING_DOWNLOAD_LIMIT of the --env file, else none")
    parser.add_argument("--events", type=Path, metavar="FILE",
                        help="write one JSON progress event per line to FILE, replacing it; stdout is unchanged")
    parser.add_argument("--no-auto-recover", action="store_true",
                        help="do not restart this model automatically when a Spark stops serving; "
                             "sudo sparkring recover on turns it on later")
    serving_settings.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.events is not None and not args.events.parent.is_dir():
        parser.error(f"--events: the directory {args.events.parent} does not exist")
    output = sys.stdout
    code = 0
    with contextlib.redirect_stdout(sys.stderr), progress.run("install", events=args.events):
        transaction = controller.STATE / "transaction.json"
        try:
            previous_transaction = transaction.read_bytes()
        except OSError:
            previous_transaction = None
        try:
            result = execute(args)
        except NeedsInput as error:
            result, code = {"schema": "sparkring-install-result/v1", **error.document()}, 3
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
            result, code = {"schema": "sparkring-install-result/v1", "state": "failed", "message": str(error)}, 2
            if transaction.exists() and transaction.read_bytes() != previous_transaction:
                result["transaction"] = installer.read(transaction)
            progress.failure(str(error))
        if result["state"] == "complete":
            if (result.get("hairpin") or {}).get("required"):
                print(hairpin_ring.COMPLETE)
            print("Model ready: " + result["api_url"])
            for line in summary_lines(result):
                print(line)
        elif result["state"] == "needs_input":
            print(result["message"])
            # A checkpoint message already names every file, path and size its
            # details hold; --json keeps the details for scripts.
            if result.get("field") != "checkpoint":
                for line in controller.detail_lines(result.get("details")):
                    print("  " + line)
        elif result["state"] == "planned":
            command = (result.get("checkpoint") or {}).get("command") or checkpoint_plan.COMMAND
            print(f"Plan saved. Install it with {command} --yes.")
        progress.emit(result["state"], "SparkRing install", phase="install", message=result.get("message"),
                      field=result.get("field"), api_url=result.get("api_url") if result["state"] == "complete" else None)
    if args.json:
        print(json.dumps(result, indent=2), file=output)
    return code
