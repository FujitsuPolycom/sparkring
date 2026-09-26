"""Locked, profile-driven installations and portable Compose exports.

This module plans without host access. The controller executes through the
existing deployment engine; serving settings remain owned by profile adapters.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import ipaddress
from pathlib import Path, PurePosixPath
import re
import uuid
import zipfile

from runtime.common import compose, distribution, installer_image, process_lock, profiles, setup, tp2
from scripts import deploy_engine

ROOT = profiles.ROOT
# Profiles offered by `sparkring install`, `sparkring models` and `init --model`.
# Every one runs on the shared image selected by installer_image.DEFAULT_LOCK.
DEFAULTS = {
    ("glm53", 2): "glm53-flash-nvfp4-spark-tp2",
    ("glm53", 4): "glm53-flash-nvfp4-spark-tp4",
    ("mimo26", 2): "mimo-v26-flash-rl-tp2",
    ("mimo26", 4): "mimo-v26-flash-rl-tp4",
    ("qwen38", 2): "qwen38-flash-next-tp2",
    ("qwen38", 4): "qwen38-flash-next-qad-tp4",
}
INSTALLABLE = frozenset(installer_image.SUPPORTED)
# Profiles on their published per-release images. Saved deployments of these
# still validate and roll back, but new installations use INSTALLABLE only.
GLM_LEGACY = {2: "glm53-flash-spark-tp2-dcp1-sparkcache", 4: "glm53-flash-spark-tp4-dcp1-sparkcache"}
GLM_NO_CACHE = {2: "glm53-flash-spark-tp2-dcp1-nocache", 4: "glm53-flash-spark-tp4-dcp1-nocache"}
SUPPORTED = frozenset((*INSTALLABLE, *GLM_LEGACY.values(), *GLM_NO_CACHE.values(), *compose.SUPPORTED))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value if isinstance(value, str) else compose.encoded(value))
    path.chmod(0o600)


def read(path):
    return profiles.read_json(path)


def host(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,253}", value) or value == "controller":
        raise ValueError("Use one SSH alias or user@host, without options")
    return value


def address(value):
    parsed = ipaddress.IPv4Address(value)
    if parsed.is_unspecified or parsed.is_multicast or parsed.is_loopback:
        raise ValueError("Use an actual host IPv4 address")
    return str(parsed)


def checkpoint_contract(card):
    source, _ = profiles.load(card["profile"])
    model = profiles.resolve(card["profile"])["model"]
    configuration = profiles.read_json(profiles.local_path(source["configuration"]["path"]))
    if source["configuration"]["format"] == "release-profile":
        model = configuration["target_variants"][card["target_variant"]]
    elif card.get("target_variant") and "checkpoints" in configuration:
        model = configuration["checkpoints"][card["target_variant"]]["model"]
    return model


PIN_SCHEMA = "sparkring-checkpoint-pins/v1"
PIN_KEYS = {"schema", "repository", "revision", "index", "weights", "optional", "files"}
PIN_ENTRY_KEYS = {"size", "sha256", "git_blob", "lfs", "xet_hash"}


def _checkpoint_slug(card):
    """`<owner>--<name>` of the card's repository, after checking repository and revision.

    Hub repository names never contain "--" or "..", so the slug names exactly
    one repository, as the Hub cache's `models--<owner>--<name>` does.
    """
    repository, revision = card["model_repository"], card["model_revision"]
    part = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"
    if (not isinstance(repository, str) or not re.fullmatch(part + "/" + part, repository)
            or "--" in repository or ".." in repository):
        raise ValueError("Checkpoint repository must be a Hugging Face owner/name")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Checkpoint revision must be a full 40-character commit id")
    return repository.replace("/", "--")


def _pinned_name(name):
    """A normalized relative POSIX path that cannot leave a checkpoint directory."""
    return (isinstance(name, str) and bool(name) and not name.startswith("/")
            and not any(character in name for character in "\0\r\n\\")
            and PurePosixPath(name).as_posix() == name
            and not {"", ".", "..", ".cache", ".git"} & set(name.split("/")))


def checkpoint_pins(card, *, root=None):
    """The validated pin manifest of the card's checkpoint revision.

    `profiles/checkpoints/<owner>--<name>/<revision>.json`, written by
    `scripts/pin_checkpoint.py`, pins every file of the revision in `files`
    (size, SHA-256, Git blob id, and LFS and Xet identities). `index` names the
    weight index, `weights` is the sorted set of the index's weight files, and
    `optional` lists documentation and repository metadata that no serving
    component reads. The required files are `files` minus `optional`.

    The manifest is refused unless it names the card's repository and revision,
    its `config.json` and index SHA-256 equal `checkpoint_contract(card)`, every
    name is a normalized relative path without `..`, `.cache` or `.git`
    components, NUL, CR, LF or backslash, every size is a positive integer and
    every digest well formed, the index and weights are required files, and
    `optional` names only pinned files. `root` selects another checkout.
    """
    relative = f"profiles/checkpoints/{_checkpoint_slug(card)}/{card['model_revision']}.json"
    path = Path(root if root is not None else ROOT) / relative
    if not path.is_file():
        raise ValueError(f"No pin manifest {relative}; generate it with scripts/pin_checkpoint.py")
    pins = profiles.read_json(path)

    def check(condition, reason):
        if not condition:
            raise ValueError(f"{relative}: {reason}")

    check(isinstance(pins, dict) and set(pins) == PIN_KEYS, "expected exactly " + ", ".join(sorted(PIN_KEYS)))
    check(pins["schema"] == PIN_SCHEMA, "expected " + PIN_SCHEMA)
    check((pins["repository"], pins["revision"]) == (card["model_repository"], card["model_revision"]),
          "pins another repository or revision than the profile")
    files = pins["files"]
    check(isinstance(files, dict) and files, "files must map each file name to its pins")
    for name, entry in files.items():
        check(_pinned_name(name), f"unsafe file name {name!r}")
        check(isinstance(entry, dict) and {"size", "sha256", "git_blob"} <= set(entry) <= PIN_ENTRY_KEYS,
              f"{name}: expected size, sha256, git_blob and optionally lfs and xet_hash")
        check(type(entry["size"]) is int and entry["size"] > 0, f"{name}: size must be a positive integer")
        check(isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]),
              f"{name}: sha256 must be 64 lowercase hexadecimal digits")
        check(isinstance(entry["git_blob"], str) and re.fullmatch(r"[0-9a-f]{40}", entry["git_blob"]),
              f"{name}: git_blob must be 40 lowercase hexadecimal digits")
        check(entry.get("lfs", True) is True, f"{name}: lfs, when present, must be true")
        check("xet_hash" not in entry or (isinstance(entry["xet_hash"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["xet_hash"])),
              f"{name}: xet_hash must be 64 lowercase hexadecimal digits")
    directories = {parent for name in files for parent in PurePosixPath(name).parents}
    check(not any(PurePosixPath(name) in directories for name in files), "a file name is also a directory of another file")
    for key in ("weights", "optional"):
        values = pins[key]
        check(isinstance(values, list) and all(isinstance(value, str) for value in values)
              and len(set(values)) == len(values), f"{key} must be a list of distinct file names")
    check(pins["weights"] == sorted(pins["weights"]), "weights must be sorted")
    check(set(pins["optional"]) <= set(files), "optional names a file that is not pinned")
    required = set(files) - set(pins["optional"])
    check(isinstance(pins["index"], str) and pins["index"] in required, "index must be a required file")
    check(pins["weights"] and set(pins["weights"]) <= required, "weights must name at least one file, all required")
    check("config.json" in required, "config.json must be a required file")
    contract = checkpoint_contract(card)
    check(files["config.json"]["sha256"] == contract.get("config_sha256"),
          "config.json differs from the profile's checkpoint contract")
    check(files[pins["index"]]["sha256"] == contract.get("index_sha256"),
          "the index differs from the profile's checkpoint contract")
    return pins


def checkpoint_directory(cluster, card):
    """SparkRing's checkpoint directory for the card's revision on one cluster.

    `/srv/sparkring/<cluster>/checkpoints/<owner>--<name>/<revision>` is shared
    by every deployment of that revision on the cluster, whatever its profile or
    node count. It is disjoint from the cluster cache
    `/srv/sparkring/<cluster>/cache` and from deployment workspaces
    `/srv/sparkring/<cluster>/<profile>[-<instance>]`, whose names start with a
    profile ID. `cluster` is the cluster record, whose `name` is used, or that
    name itself.
    """
    name = cluster.get("name") if isinstance(cluster, dict) else cluster
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", name):
        raise ValueError("Choose a lowercase cluster name")
    return f"/srv/sparkring/{name}/checkpoints/{_checkpoint_slug(card)}/{card['model_revision']}"


def managed_workspace(name):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", name):
        raise ValueError("Invalid managed deployment name")
    return "/srv/sparkring/" + name + "-managed"


def site_document(raw, card, revision):
    if not isinstance(raw, dict) or set(raw) - {"schema", "name", "workspace", "hosts", "controller_address", "api_address", "native_mesh"}:
        raise ValueError("Unknown site setting")
    if raw.get("schema") != "sparkring-install-site/v1":
        raise ValueError("Expected sparkring-install-site/v1")
    name = raw.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", name):
        raise ValueError("Choose a lowercase deployment name, at most 40 characters")
    rows = raw.get("hosts")
    if not isinstance(rows, list) or len(rows) != card["nodes"]:
        raise ValueError(f"The selected profile needs exactly {card['nodes']} hosts in rank order")
    workspace = str(compose.linux_path(raw.get("workspace", "/srv/sparkring/" + name)))
    if len(PurePosixPath(workspace).parts) < 4:
        raise ValueError("Use a dedicated workspace below an operator-owned parent")
    ranks = []
    cache_default = managed_workspace(name) + "/cache" if backend(card) == "glm-managed" else workspace + "/cache"
    for number, row in enumerate(rows):
        allowed = {"host", "management_ip", "fabric_ip", "interface", "model", "cache", "reuse_verified_model", "fabric", "node_id"}
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError("Unknown host field; passwords and runtime overrides are not site inputs")
        interface = row.get("interface")
        if not isinstance(interface, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", interface):
            raise ValueError("Supply the discovered fabric interface")
        reuse = row.get("reuse_verified_model", False)
        if type(reuse) is not bool:
            raise ValueError("reuse_verified_model must be a boolean operator declaration")
        item = {
            "rank": number, "host": host(row["host"]),
            "management_ip": address(row["management_ip"]), "host_ip": address(row["fabric_ip"]),
            "interface": interface, "gid": 3,
            "hcas": ["rocep1s0f0", "roceP2p1s0f0"] if card["nodes"] == 2 else
                    ["rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1"],
            "model": str(compose.linux_path(row.get("model", workspace + "/models/" + card["model_revision"]))),
            "cache": str(compose.linux_path(row.get("cache", cache_default))),
            "repository": workspace + "/source-" + revision[:12],
            "deployment_root": workspace + "/containers", "reuse_verified_model": reuse,
        }
        if "node_id" in row:
            if not isinstance(row["node_id"], str):
                raise ValueError("Persistent node ID must be a UUID string")
            item["node_id"] = str(uuid.UUID(row["node_id"]))
        if backend(card) == "glm-managed" and item["cache"] != cache_default:
            raise ValueError("Managed GLM cache must remain under its dedicated backend workspace")
        if item["host_ip"] == item["management_ip"] and not (card["nodes"] == 4 and "fabric" in row):
            raise ValueError("Management and data fabric addresses must be distinct")
        paths = [PurePosixPath(item[key]) for key in ("model", "cache", "repository", "deployment_root")]
        for i, path in enumerate(paths):
            if any(path.is_relative_to(other) or other.is_relative_to(path) for other in paths[i + 1:]):
                raise ValueError("Model, cache, source and container paths must be disjoint")
        if "fabric" in row:
            from runtime.common import qwen_mesh
            qwen_mesh.validate_site_reference(row["fabric"])
            item["fabric"] = row["fabric"]
        if card["profile"] in compose.TP4_PROFILES and "fabric" not in item:
            raise ValueError("Qwen TP4 needs the prepared mesh fabric reference in each host; import the existing site")
        ranks.append(item)
    for field in ("host", "management_ip", "host_ip"):
        if len({r[field] for r in ranks}) != len(ranks):
            raise ValueError("Hosts and rank addresses must be distinct")
    identities = [row.get("node_id") for row in ranks]
    if any(identities) and (not all(identities) or len(set(identities)) != len(ranks)):
        raise ValueError("Supply distinct persistent node IDs for every rank, or omit all of them")
    result = {"name": name, "workspace": workspace, "ranks": ranks,
              "controller_address": address(raw.get("controller_address", ranks[0]["management_ip"]))}
    if "api_address" in raw:
        result["api_address"] = address(raw["api_address"])
    if "native_mesh" in raw:
        if card["profile"] not in compose.TP4_PROFILES:
            raise ValueError("This native-mesh plan requires a Qwen TP4 profile")
        from runtime.host import native_mesh
        native_mesh.validate(raw["native_mesh"], raw)
    return result


def backend(card):
    return "glm-managed" if card["profile"] in (GLM_LEGACY[4], GLM_NO_CACHE[4]) else "compose"


def make_lock(profile, raw_site, revision, bundle_sha256, variant=None, *, image_runtime=None):
    if profile not in SUPPORTED:
        raise ValueError("Installer supports the shared GLM/Qwen pair/ring profiles; other profiles retain their own guides")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"[0-9a-f]{64}", bundle_sha256):
        raise ValueError("Lock requires the exact source commit and bundle checksum")
    card = setup.selection(profile, variant)
    if image_runtime is not None:
        from runtime.common import installer_image
        card = installer_image.selection(card, image_runtime)
    site = site_document(raw_site, card, revision)
    selected_backend = backend(card)
    if selected_backend == "glm-managed" and all("fabric" in row for row in site["ranks"]):
        selected_backend = "glm-existing-mesh"
    value = {"schema": "sparkring-install-lock/v1", "selection": card, "site": site,
             "site_input": raw_site, "source_revision": revision, "bundle_sha256": bundle_sha256,
             "backend": selected_backend}
    if image_runtime is not None:
        value["image_runtime"] = image_runtime
    value["id"] = compose.digest(compose.encoded(value))
    return value


def validate(lock):
    expected = make_lock(lock["selection"]["profile"], lock["site_input"], lock["source_revision"],
                         lock["bundle_sha256"], lock["selection"]["target_variant"], image_runtime=lock.get("image_runtime"))
    if expected != lock:
        raise ValueError("Deployment lock or its profile/release inputs changed; initialize a new deployment")
    return lock


def load(directory):
    directory = Path(directory).resolve()
    lock = validate(read(directory / "deployment.lock.json"))
    if hashlib.sha256((directory / "source.bundle").read_bytes()).hexdigest() != lock["bundle_sha256"]:
        raise ValueError("Source bundle differs from the deployment lock")
    return lock


def init(directory, profile, raw_site, *, variant=None, image_runtime=None):
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("Deployment directory already exists; use up/status or choose a new directory")
    # Validate user inputs before creating artifacts.
    revision = distribution.identity(ROOT)
    make_lock(profile, raw_site, revision, "0" * 64, variant, image_runtime=image_runtime)
    if directory.is_relative_to(ROOT) and not directory.is_relative_to(ROOT / ".sparkring"):
        raise ValueError("Private deployments inside the checkout belong under .sparkring/")
    directory.mkdir(parents=True, mode=0o700)
    bundle = directory / "source.bundle"
    distribution.bundle(ROOT, bundle)
    lock = make_lock(profile, raw_site, revision, hashlib.sha256(bundle.read_bytes()).hexdigest(), variant, image_runtime=image_runtime)
    write(directory / "site.json", raw_site)
    write(directory / "deployment.lock.json", lock)
    if lock["backend"] == "compose":
        for name, text in rendered(lock).items():
            write(directory / name, text)
    return lock


def compose_site(lock):
    ranks = [{k: v for k, v in row.items() if k not in ("management_ip", "reuse_verified_model", "node_id")}
             for row in lock["site"]["ranks"]]
    return {"schema": "sparkring-compose-site/v1", "name": lock["site"]["name"],
            "master": ranks[0]["host_ip"], "ranks": ranks}


def specifications(lock, *, receipt=None, local=False, only_rank=None):
    card, site = lock["selection"], lock["site"]
    if lock["backend"] != "compose":
        raise ValueError("Managed GLM Compose files are produced by its existing staging lifecycle")
    if card["profile"] in compose.SUPPORTED:
        specs, _ = compose.specifications(card["profile"], compose_site(lock), checkpoint=card["target_variant"])
        if "image_runtime" in lock:
            from runtime.common import installer_image
            specs = [installer_image.adapt(spec, lock["image_runtime"], binding=installer_image.binding_path(lock, row),
                                           source_root=row["repository"], profile=card["profile"])
                     for spec, row in zip(specs, site["ranks"], strict=True)]
    else:
        specs = []
        for row in site["ranks"]:
            if only_rank is not None and row["rank"] != only_rank:
                continue
            # Real admission occurs independently on each host before create/start.
            if local and receipt is not None:
                model, cache = Path(row["model"]), Path(row["cache"])
                options = {"r33_receipt": receipt}
            else:
                model, cache = PurePosixPath(row["model"]), PurePosixPath(row["cache"])
                options = {"planning_release": card["release"]}
            plan = tp2.render(row["rank"], site["ranks"][0]["host_ip"], model, cache, None,
                              card["image_id"], r33_sparkcache=card["sparkcache"],
                              target_model_variant=card["target_variant"],
                              site_values={"VLLM_HOST_IP": row["host_ip"], "NCCL_SOCKET_IFNAME": row["interface"],
                                           "GLOO_SOCKET_IFNAME": row["interface"]}, **options)
            specs.append(tp2.container_spec(plan))
    if only_rank is not None and card["profile"] in compose.SUPPORTED:
        specs = [specs[only_rank]]
    return [replace(spec, name=f"sr-{site['name']}-r{only_rank if only_rank is not None else number}",
                    labels={**spec.labels, compose.LABEL: lock["id"], "io.sparkring.rank": str(only_rank if only_rank is not None else number)})
            for number, spec in enumerate(specs)]


def rendered(lock):
    files = {}
    for number, spec in enumerate(specifications(lock)):
        files[f"rank{number}/compose.yaml"] = compose.compose_text(spec, lock["selection"]["image_reference"])
        files[f"rank{number}/container.json"] = compose.encoded(spec.document())
    return files


def connection(lock):
    if lock["backend"] in ("glm-managed", "glm-existing-mesh"):
        from runtime.common import glm_tp4
        port = int(glm_tp4.DEFAULTS["PORT"])
        model = "GLM-5.3-Flash-NVFP4-" + ("QAD" if lock["selection"]["target_variant"] == "nvfp4-qad" else "Spark") + "-TP4"
    else:
        args = specifications(lock, only_rank=0)[0].command
        port = int(args[args.index("--port") + 1])
        model = args[args.index("--served-model-name") + 1]
    host_ip = lock["site"].get("api_address", lock["site"]["ranks"][0]["management_ip"])
    return {"api_url": f"http://{host_ip}:{port}/v1", "model": model, "port": port}


def operation_plan(lock, action):
    if action not in ("prepare", "up", "down", "status"):
        raise ValueError("Unsupported installation operation")
    ranks = lock["site"]["ranks"]

    def phase(name, rows, risk="read-only", verify=None):
        actions = []
        for row in rows:
            command = ["installer", name, str(row["rank"])]
            item = {"host": row["host"], "argv": command, "risk": risk, "timeout": 7200}
            if verify:
                item["verify"] = {"argv": ["installer", verify, str(row["rank"])], "stdout": "ok", "timeout": 7200}
            actions.append(item)
        return {"id": name, "actions": actions}

    if action == "status":
        phases = [phase("status", ranks)]
    elif action == "down":
        if lock["backend"] == "glm-managed":
            phases = [phase("managed-stop", ranks[:1], "stops-model", "managed-stopped")]
        else:
            phases = [phase("owned", ranks), phase("stop", ranks, "stops-model", "stopped")]
    else:
        # Checkpoint preparation precedes image admission, but a checkpoint
        # download runs the serving image's Hugging Face client, so the image
        # must already be on the node. sparkring install distributes it before
        # this phase starts (install_assets.Assets.runner). Callers that run
        # this plan with the plain installer_runner.Runner, such as the
        # lower-level sparkring up, need the image on every node beforehand;
        # that path does not provide it.
        phases = [phase("prepare-prerequisites" if action == "prepare" else "prerequisites", ranks), phase("source", ranks, "mutates-host", "source-check"),
                  phase("model", ranks, "mutates-host", "model-check"),
                  phase("image", ranks, "mutates-host", "image-check")]
        if action == "prepare":
            if lock["backend"] in ("glm-managed", "glm-existing-mesh"):
                phases += [phase("managed-prepare", ranks[:1], "mutates-host", "managed-prepared")]
            return deploy_engine.seal_plan({"schema": "sparkring-deploy-plan/v1", "deployment": lock["id"],
                                           "operation": action, "phases": phases})
        if lock["backend"] == "glm-managed":
            phases += [phase("managed-prepare", ranks[:1], "mutates-host", "managed-prepared"),
                       phase("managed-create", ranks[:1], "starts-model", "managed-created"),
                       phase("managed-install", ranks[:1], "mutates-host", "managed-installed"),
                       phase("managed-up", ranks[:1], "mutates-host", "managed-up-check"),
                       phase("managed-native-check", ranks[:1], "hardware-test", "managed-native-checked"),
                       phase("managed-start", ranks[:1], "starts-model", "managed-running"),
                       phase("managed-ready", ranks[:1])]
        else:
            if lock["backend"] == "glm-existing-mesh":
                phases += [phase("managed-prepare", ranks[:1], "mutates-host", "managed-prepared")]
            if "native_mesh" in lock["site_input"]:
                phases += [phase("mesh-prepare", ranks, "mutates-host", "mesh-prepared"),
                           phase("create", ranks, "starts-model", "created"),
                           phase("mesh-install", ranks[:1], "mutates-host", "mesh-installed"),
                           phase("mesh-replace", ranks, "mutates-host", "mesh-replaced"),
                           phase("mesh-up", ranks, "mutates-host", "mesh-up-check"),
                           phase("mesh-gate", ranks), phase("preflight", ranks)]
            else:
                if len(ranks) == 4 and lock["backend"] != "glm-existing-mesh":
                    # A reused mesh is started, repaired and awaited on all
                    # four ranks before the read-only ring check.
                    phases += [phase("ring-serve", ranks, "mutates-host", "ring-check")]
                phases += [phase("preflight", ranks), phase("create", ranks, "starts-model", "created")]
            phases += [
                       phase("start", ranks[1:], "starts-model", "running"),
                       phase("start", ranks[:1], "starts-model", "running"), phase("ready", ranks)]
            # Phase IDs are receipt keys, so API/worker barriers need distinct IDs.
            phases[-3]["id"], phases[-2]["id"] = "start-workers", "start-api"
        # Checkpoint files verified before start must still be unchanged once
        # every rank has loaded the model; a change fails the switch, so the
        # previous deployment is restored.
        phases += [phase("smoke", ranks[:1]), phase("model-settled", ranks)]
    return deploy_engine.seal_plan({"schema": "sparkring-deploy-plan/v1", "deployment": lock["id"],
                                    "operation": action, "phases": phases})


def apply(directory, action, *, runner, execute=False):
    directory = Path(directory).resolve()
    lock = load(directory)
    plan = operation_plan(lock, action)
    if not execute:
        result = {"executed": False, "deployment": lock["id"], "operation": action,
                  "profile": lock["selection"]["profile"], "image": lock["selection"]["image_reference"],
                  "hosts": [r["host"] for r in lock["site"]["ranks"]],
                  "phases": [p["id"] for p in plan["phases"]]}
        if "native_mesh" in lock["site_input"]:
            result["native_mesh"] = {"mode": "create", "replaces": lock["site_input"]["native_mesh"]["replaces"]}
        return result
    with process_lock.hold(directory / "operation.lock"):
        state_path = directory / "state.json"
        state = read(state_path) if state_path.exists() else {"generation": 0, "operation": None, "complete": True}
        # Asset preparation never creates, starts or stops a model container.
        # Its actions check hosts or verify and complete the deployment's
        # source, checkpoint and image, and a repeated action re-verifies what
        # an earlier one left. An incomplete preparation is therefore repeated
        # as a new generation; resuming its receipt would refuse the actions
        # that a failure or interruption left uncertain or running.
        retry = action == "prepare" and state["operation"] == action and not state["complete"]
        if state["operation"] != action or retry:
            # Stopping is always permitted after an incomplete operation: it
            # verifies ownership labels, stops only this deployment's running
            # containers and ignores ranks whose container was never created.
            if not state["complete"] and action != "down" and not retry:
                raise ValueError("Previous operation is incomplete or uncertain; inspect its receipts before changing direction")
            state = {"generation": state["generation"] + 1, "operation": action, "complete": False}
        receipt_path = directory / "operations" / f"{state['generation']:04}-{action}.json"
        state.update(complete=False, receipt=str(receipt_path.relative_to(directory)))
        deploy_engine.save_receipt(state_path, state)
        result = deploy_engine.execute_plan(plan, receipt_path, plan["sha256"], runner=runner,
                                             resume=receipt_path.exists(), allow_model_actions=True,
                                             allow_hardware_tests=lock["backend"] == "glm-managed")
        state["complete"] = result["complete"]
        deploy_engine.save_receipt(state_path, state)
        return {"executed": True, **state, **connection(lock)}


def status(directory):
    directory = Path(directory)
    lock = load(directory)
    state = read(directory / "state.json") if (directory / "state.json").exists() else {"operation": "not started", "complete": False}
    result = {"profile": lock["selection"]["profile"], "state": state,
              "deployment_id": lock["id"], "source_revision": lock["source_revision"],
              "image_id": lock["selection"]["image_id"], "image_reference": lock["selection"]["image_reference"],
              "live_observed": False, "hosts": [r["host"] for r in lock["site"]["ranks"]], **connection(lock)}
    if state.get("receipt"):
        receipt = read(directory / state["receipt"])
        result["actions"] = {key: value["state"] for key, value in receipt["actions"].items()}
        result["problems"] = {key: value.get("result", {}).get("stderr", "Inspect the incomplete operation")
                              for key, value in receipt["actions"].items() if value["state"] != "succeeded"}
        if "failure" in receipt:
            result["failure"] = receipt["failure"]
    return result


def export(directory, output, *, share=False):
    lock = load(directory)
    output = Path(output)
    if output.exists():
        raise ValueError("Export already exists; choose a new output")
    files = {}
    if share:
        count = lock["selection"]["nodes"]
        example = {"schema": "sparkring-install-site/v1", "name": "example", "hosts": [
            {"host": f"spark{n}", "management_ip": f"192.0.2.{10+n}",
             "fabric_ip": f"198.18.20.{n+1}", "interface": "FABRIC_NETDEV"}
            for n in range(count)]}
        if count == 4 and lock["selection"]["profile"] in compose.TP4_PROFILES:
            for row in example["hosts"]:
                row["fabric"] = {"site_path": "/srv/sparkring/mesh-site.json", "site_sha256": "0" * 64, "plan_sha256": "0" * 64}
        runtime = lock.get("image_runtime")
        if runtime is not None:
            runtime = {**runtime, "image_reference": runtime["image_id"]}
            files["image-lock.json"] = compose.encoded(runtime)
        portable = make_lock(lock["selection"]["profile"], example, lock["source_revision"],
                             lock["bundle_sha256"], lock["selection"]["target_variant"], image_runtime=runtime)
        files["site.example.json"] = compose.encoded(example)
        files["profile.json"] = compose.encoded(lock["selection"])
        files["README.txt"] = ("Portable profile template, not a configured deployment. Fill site.example.json, then run "
                               "sparkring init --profile " + lock["selection"]["profile"] + " --site site.example.json"
                               + (" --image-lock image-lock.json" if runtime else "") + ".\n"
                               "Compose files below contain example hosts/paths. Re-render for your site. A separate "
                               "Compose project runs on each rank; Compose alone does not configure RDMA or coordinate hosts.\n")
        if runtime:
            files["profile.json"] = compose.encoded(portable["selection"])
            files["README.txt"] += "Preload the pinned candidate image on every rank; its private registry location is excluded.\n"
        if portable["backend"] == "compose":
            files.update(rendered(portable))
    else:
        files["deployment.lock.json"] = compose.encoded(lock)
        files["site.json"] = compose.encoded(lock["site_input"])
        files["README.txt"] = "PRIVATE: contains this site's addresses and storage paths. Use export --share for a portable template.\n"
        if lock["backend"] == "compose":
            files.update(rendered(lock))
        else:
            files["README.txt"] += "Managed GLM exports Compose at staging; retain its managed lifecycle for start/stop.\n"
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)
        if not share:
            archive.write(Path(directory) / "source.bundle", "source.bundle")
            files["source.bundle"] = ""
    output.chmod(0o600)
    return {"path": str(output), "shareable_template": share, "files": sorted(files)}
