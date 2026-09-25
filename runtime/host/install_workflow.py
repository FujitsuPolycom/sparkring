"""One Linux entrypoint for cluster setup, asset preparation and model replacement."""
import argparse
import concurrent.futures
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

from runtime.common import distribution, installer, installer_image, process_lock
from runtime.host import controller, discovery, fabric_ssh, install_assets, models, native_mesh, node, progress, retained_source, rollout, topology
from runtime.host.install_errors import NeedsInput
from scripts import deploy_network


def require_head(cluster=None):
    if sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() != 0 or not distribution.installed(installer.ROOT):
        raise NeedsInput("Install the ARM64 package, then run sudo sparkring install on Node A.", field="node_a")
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


def refresh_cluster(cluster):
    """Re-observe cables without reconfiguring the adopted fabric."""
    hosts = cluster["plan"]["spec"]["hosts"]
    found = controller.collect([h["host"] for h in hosts])
    plan = topology.build_spec(found, require_head(cluster), name=cluster["name"],
                               fabric_cidr=cluster["plan"].get("fabric_cidr", "198.18.0.0/21"),
                               preserve_control=cluster["plan"]["spec"].get("preserve_control_ipv6", False))
    if [h["node_id"] for h in plan["spec"]["hosts"]] != [h["node_id"] for h in hosts]:
        raise NeedsInput("Cable order changed. Run sparkring setup to review the new fabric first.", field="fabric")
    deploy_network.verify_network(plan["spec"], plan["inventory"]["hosts"])
    return {**cluster, "plan": plan}


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


def select_deployment(args, cluster, state_root):
    profile = choose_profile(args.profile, len(cluster["plan"]["nodes"]), not args.json and sys.stdin.isatty())
    image = installer_image.for_profile(profile, installer.read(args.image_lock) if args.image_lock else None)
    request = {"profile": profile, "image_runtime": image, "source": distribution.identity(installer.ROOT),
               "model_path": args.model_path, "cache_path": args.cache_path,
               "nodes": cluster["plan"]["spec"]["hosts"], "api_address": cluster.get("api_address")}
    instance = "i" + hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
    directory = state_root / "deployments" / (profile + "-" + instance)
    if directory.exists():
        return directory, installer.load(directory)
    site = controller.model_site(cluster, profile, instance)
    for rank, row in enumerate(site["hosts"]):
        if args.model_path:
            row.update(model=args.model_path, reuse_verified_model=True)
        else:
            from runtime.host import assets as asset_discovery
            card = installer.setup.selection(profile)
            code = asset_discovery.probe_code(card, installer.checkpoint_contract(card))
            found = json.loads(discovery.ssh(row["host"], ["sudo", "-n", "python3", "-I", "-c", code]))
            if found["model_path"]:
                row.update(model=found["model_path"], reuse_verified_model=True)
            print(f"Node {rank}: " + ("Cached checkpoint found; verify before launch" if found["model_path"] else "Checkpoint will be copied or downloaded"))
        # One compile/tuning cache per cluster. Containers use a subdirectory
        # keyed by model family, image and checkpoint revision, so repeated or
        # alternating installs reuse earlier kernel tuning.
        row["cache"] = args.cache_path or "/srv/sparkring/" + cluster["name"] + "/cache"
    if profile in installer.compose.TP4_PROFILES:
        site = native_mesh.select(site, cluster, profile, invoke=discovery.ssh)
    elif installer.backend({"profile": profile}) == "glm-managed":
        site = native_mesh.select(site, cluster, profile, existing_only=True, invoke=discovery.ssh)
    lock = installer.init(directory, profile, site, image_runtime=image)
    return directory, lock


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


def execute(args):
    state_root = controller.STATE
    require_head()
    interactive = not args.json and sys.stdin.isatty()
    with process_lock.hold(state_root / "install.lock"):
        if not (state_root / "cluster.json").exists():
            if args.plan:
                raise NeedsInput("Run sparkring setup --plan to discover this unconfigured cluster.", field="setup")
            if not args.yes and not interactive:
                raise NeedsInput("First installation requires setup approval. Review sparkring setup --plan, then use --yes.", field="approval")
            from runtime.host import single_uplink
            options = ((["--env", str(args.env)] if args.env else []) + (["--yes"] if args.yes else [])
                       + (["--stop-workloads"] if args.stop_workloads else []))
            if single_uplink.main(options):
                raise ValueError("Cluster setup did not complete")
        cluster = installer.read(state_root / "cluster.json")
        check_access(cluster)
        cluster = refresh_cluster(cluster)
        directory, lock = select_deployment(args, cluster, state_root)
        check_managed_namespace(lock)
        previous = rollout.active(state_root)
        plan = {"schema": "sparkring-install-result/v1", "state": "planned", "deployment": str(directory),
                "profile": lock["selection"]["profile"], "image_id": lock["selection"]["image_id"],
                "nodes": len(lock["site"]["ranks"]), "replaces": str(previous) if previous and previous != directory else None,
                "steps": ["verify-fabric", "update-workers", "prepare-images-and-checkpoints", "switch-model", "verify-serving"],
                **installer.connection(lock)}
        print(f"Install {plan['profile']} on {plan['nodes']} Sparks.")
        print("Update workers and prepare assets; then " + ("replace the current model." if plan["replaces"] else "start the selected model."))
        if "native_mesh" in lock["site_input"]:
            if previous:
                raise NeedsInput("The replacement needs native fabric configuration. Review sparkring setup before replacing a running deployment.", field="fabric")
            print("Configure and start the profile's supervised native fabric.")
        if args.plan:
            return plan
        if not args.yes:
            if not interactive:
                raise NeedsInput("Review with --plan; add --yes to apply these changes.", field="approval", details=plan)
            controller.confirm("Apply this installation?")
        def approve_stop(host, names):
            print(f"{host}: stopping unrelated GPU containers (not removing them): " + ", ".join(names))
            if not args.stop_workloads:
                controller.confirm("Stop these containers before installing?")
        check_workloads(directory, previous, stop=approve_stop if args.stop_workloads or interactive else None)
        transport = fabric_ssh.Transport(cluster, state_root / "bulk-ssh")
        plan["transfer"] = transport.verify()
        assets = install_assets.Assets(transport, directory / "assets")
        cache = state_root / "retained-sources"

        def apply(path, operation):
            return retained_source.apply(path, operation, cache=cache)

        def prepare(path):
            if previous and previous != path:
                # Verify the rollback controller bundle before any downtime.
                retained_source.checkout(previous, cache)
            assets.sync_packages()
            # Image distribution and checkpoint preparation use different
            # resources (registry and fabric versus Hugging Face and disk), so
            # they run together; image admission waits for the distribution.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                images = pool.submit(assets.images, lock["selection"])
                try:
                    installer.apply(path, "prepare", runner=assets.runner(path, previous, images), execute=True)
                finally:
                    images.result()
            check_workloads(path, previous)

        # check_workloads has confirmed that only the active or candidate
        # deployment uses the GPUs, so an abandoned failed switch can be replaced.
        result = rollout.execute(directory, previous, state_root=state_root, prepare=prepare, apply=apply,
                                 verify=lambda path: apply(path, "verify"), supersede=True)
        return {**plan, "state": "complete", "transaction": result, "log": str(progress.directory() / "install.log")}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring install", description="Set up this Spark ring and deploy one exact model profile.")
    parser.add_argument("--profile", help="exact profile from sparkring models; prompted in a terminal")
    parser.add_argument("--image-lock", type=Path, help="development image lock replacing the shared installer image")
    parser.add_argument("--model-path", help="existing complete checkpoint path on each Spark")
    parser.add_argument("--cache-path", help="optional local writable cache path on each Spark")
    parser.add_argument("--env", type=Path, help="optional literal setup preferences, read on first installation")
    parser.add_argument("--plan", action="store_true", help="inspect and save the plan without updating workers or models")
    parser.add_argument("--yes", action="store_true", help="approve the displayed setup and model replacement; SSH trust is still required")
    parser.add_argument("--json", action="store_true", help="emit one JSON result on stdout; progress stays on stderr")
    parser.add_argument("--stop-workloads", action="store_true",
                        help="stop (never remove) running GPU containers that are not SparkRing's current deployment")
    args = parser.parse_args(argv)
    output = sys.stdout
    code = 0
    with contextlib.redirect_stdout(sys.stderr), progress.run("install"):
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
            print("Model ready: " + result["api_url"])
        elif result["state"] == "needs_input":
            print(result["message"])
        elif result["state"] == "planned":
            print("Plan saved. Repeat with --yes to install.")
    if args.json:
        print(json.dumps(result, indent=2), file=output)
    return code
