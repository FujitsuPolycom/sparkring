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

from runtime.common import (compose, derived_checkpoint, distribution, installer_image, process_lock, profiles, serving,
                            setup, tp2)
from scripts import deploy_engine

ROOT = profiles.ROOT
# Profiles offered by `sparkring install`, `sparkring models` and `init --model`.
# Every one runs on the shared image selected by installer_image.DEFAULT_LOCK.
DEFAULTS = {
    ("glm53", 2): "glm53-flash-nvfp4-spark-tp2",
    ("glm53", 4): "glm53-flash-nvfp4-spark-tp4",
    ("mimo26", 2): "mimo-v26-flash-mopd-tp2",
    ("mimo26", 4): "mimo-v26-flash-mopd-tp4",
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
    """The pinned model (repository, revision, config.json and index SHA-256) of the card's checkpoint.

    A derived checkpoint's entry (runtime/common/derived_checkpoint.py) has two:
    a selection naming the base's repository and revision, which the installer
    acquires first, gets the base entry's model, and the derived checkpoint's
    own card (derived_checkpoint.view) the entry's model.
    """
    source, _ = profiles.load(card["profile"])
    model = profiles.resolve(card["profile"])["model"]
    configuration = profiles.read_json(profiles.local_path(source["configuration"]["path"]))
    if source["configuration"]["format"] == "release-profile":
        model = configuration["target_variants"][card["target_variant"]]
    elif card.get("target_variant") and "checkpoints" in configuration:
        entry = configuration["checkpoints"][card["target_variant"]]
        model = entry["model"]
        if "derived" in entry and (card["model_repository"], card["model_revision"]) != (
                model["repository"], model["revision"]):
            model = configuration["checkpoints"][entry["derived"]["base"]]["model"]
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


# The ring halves a two-rank deployment may occupy on a four-Spark ring (runtime/host/placement.py).
HALVES = ((0, 1), (2, 3))
# The RDMA devices of a pair's rank: the primary and secondary functions of ConnectX port 0.
PAIR_HCAS = ["rocep1s0f0", "roceP2p1s0f0"]


def site_document(raw, card, revision):
    """The normalized site of a deployment lock from its raw site input.

    ``placement``, on a two-rank card only, names the half of a four-Spark
    ring (``[0, 1]`` or ``[2, 3]``) whose two Sparks the rows list in rank
    order. A row's ``hcas``, on a two-rank card only, names the rank's
    primary and secondary RDMA devices facing its partner; a row without
    ``hcas`` uses port 0's functions (``PAIR_HCAS``), as on a pair.
    """
    if not isinstance(raw, dict) or set(raw) - {"schema", "name", "workspace", "hosts", "controller_address",
                                                "api_address", "native_mesh", "placement"}:
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
    if "placement" in raw:
        placement = raw["placement"]
        if card["nodes"] != 2 or not (isinstance(placement, list) and len(placement) == 2
                                      and all(type(item) is int for item in placement)
                                      and tuple(placement) in HALVES):
            raise ValueError("A placement names one half of a four-Spark ring, [0, 1] or [2, 3], for a two-Spark "
                             "profile")
    for number, row in enumerate(rows):
        allowed = {"host", "management_ip", "fabric_ip", "interface", "model", "cache", "reuse_verified_model",
                   "fabric", "node_id", "hcas"}
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError("Unknown host field; passwords and runtime overrides are not site inputs")
        interface = row.get("interface")
        if not isinstance(interface, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", interface):
            raise ValueError("Supply the discovered fabric interface")
        reuse = row.get("reuse_verified_model", False)
        if type(reuse) is not bool:
            raise ValueError("reuse_verified_model must be a boolean operator declaration")
        hcas = row.get("hcas", PAIR_HCAS)
        if "hcas" in row and (card["nodes"] != 2 or not isinstance(hcas, list) or len(hcas) != 2
                              or len(set(hcas)) != 2
                              or not all(isinstance(hca, str) and re.fullmatch(r"[A-Za-z0-9_]{1,64}", hca)
                                         for hca in hcas)):
            raise ValueError("A two-Spark rank's hcas name its two distinct RDMA devices that face its partner")
        item = {
            "rank": number, "host": host(row["host"]),
            "management_ip": address(row["management_ip"]), "host_ip": address(row["fabric_ip"]),
            "interface": interface, "gid": 3,
            "hcas": list(hcas) if card["nodes"] == 2 else
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
            raise ValueError("This four-Spark profile needs the prepared mesh fabric reference in each host; import the existing site")
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
    if "placement" in raw:
        result["placement"] = list(raw["placement"])
    if "native_mesh" in raw:
        if card["profile"] not in compose.TP4_PROFILES:
            raise ValueError("A native-mesh plan requires a four-Spark Compose profile")
        from runtime.host import native_mesh
        native_mesh.validate(raw["native_mesh"], raw)
    return result


def backend(card):
    return "glm-managed" if card["profile"] in (GLM_LEGACY[4], GLM_NO_CACHE[4]) else "compose"


def make_lock(profile, raw_site, revision, bundle_sha256, variant=None, *, image_runtime=None, settings=None):
    if profile not in SUPPORTED:
        raise ValueError(f"The installer does not deploy {profile}; 'sparkring models' marks the profiles it installs, "
                         "and other profiles use their own guides")
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
    settings = serving.normalized(settings)
    if settings and selected_backend != "compose":
        raise ValueError("Serving settings apply to Compose deployments; this profile's backend is " + selected_backend)
    value = {"schema": "sparkring-install-lock/v1", "selection": card, "site": site,
             "site_input": raw_site, "source_revision": revision, "bundle_sha256": bundle_sha256,
             "backend": selected_backend}
    if image_runtime is not None:
        value["image_runtime"] = image_runtime
    if settings:
        value["serving"] = settings
    value["id"] = compose.digest(compose.encoded(value))
    return value


def validate(lock):
    expected = make_lock(lock["selection"]["profile"], lock["site_input"], lock["source_revision"],
                         lock["bundle_sha256"], lock["selection"]["target_variant"], image_runtime=lock.get("image_runtime"),
                         settings=lock.get("serving"))
    if expected != lock:
        raise ValueError("Deployment lock or its profile/release inputs changed; initialize a new deployment")
    return lock


def load(directory):
    directory = Path(directory).resolve()
    lock = validate(read(directory / "deployment.lock.json"))
    if hashlib.sha256((directory / "source.bundle").read_bytes()).hexdigest() != lock["bundle_sha256"]:
        raise ValueError("Source bundle differs from the deployment lock")
    return lock


def init(directory, profile, raw_site, *, variant=None, image_runtime=None, settings=None):
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("Deployment directory already exists; use up/status or choose a new directory")
    # Validate user inputs before creating artifacts.
    revision = distribution.identity(ROOT)
    provisional = make_lock(profile, raw_site, revision, "0" * 64, variant, image_runtime=image_runtime, settings=settings)
    if settings:
        # Refuses a switch that the deployment's image cannot apply, and a
        # setting whose vLLM flag the profile does not set. A recorded
        # deployment's lock is not checked again, so it stays usable.
        release = provisional["selection"]["release"]
        serving.check_image(serving.normalized(settings), release, installer_image.capabilities(release))
        specifications(provisional)
    if directory.is_relative_to(ROOT) and not directory.is_relative_to(ROOT / ".sparkring"):
        raise ValueError("Private deployments inside the checkout belong under .sparkring/")
    directory.mkdir(parents=True, mode=0o700)
    bundle = directory / "source.bundle"
    distribution.bundle(ROOT, bundle)
    lock = make_lock(profile, raw_site, revision, hashlib.sha256(bundle.read_bytes()).hexdigest(), variant,
                     image_runtime=image_runtime, settings=settings)
    write(directory / "site.json", raw_site)
    write(directory / "deployment.lock.json", lock)
    if lock["backend"] == "compose":
        for name, text in rendered(lock).items():
            write(directory / name, text)
    return lock


def served_model(lock, row):
    """The checkpoint directory that rank ``row``'s container mounts.

    It is the row's ``model``, except for a derived checkpoint, whose row names
    the base's SparkRing checkpoint directory; the container then mounts the
    derived checkpoint's directory beside it (derived_checkpoint.directory).
    """
    derived = derived_checkpoint.model_of(lock["selection"])
    return row["model"] if derived is None else derived_checkpoint.directory(row["model"], derived)


def compose_site(lock):
    ranks = [{k: v for k, v in row.items() if k not in ("management_ip", "reuse_verified_model", "node_id")}
             for row in lock["site"]["ranks"]]
    for rank, row in zip(ranks, lock["site"]["ranks"], strict=True):
        rank["model"] = served_model(lock, row)
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
    settings = lock.get("serving") or {}
    ranks = [only_rank] if only_rank is not None else range(len(specs))
    selected = (card["profile"], card["target_variant"])
    return [replace(serving.container(spec, settings, rank=rank, model=selected), name=container_name(lock, rank),
                    labels={**spec.labels, **container_labels(lock, rank)})
            for rank, spec in zip(ranks, specs, strict=True)]


def container_name(lock, rank):
    """A rank's model container name: ``sr-<site>-r<rank>``, or ``<site>-r<rank>`` for managed GLM."""
    return ("sr-" if lock["backend"] == "compose" else "") + f"{lock['site']['name']}-r{rank}"


def container_labels(lock, rank):
    """Labels on a Compose deployment's containers: its lock ID and the rank."""
    return {compose.LABEL: lock["id"], "io.sparkring.rank": str(rank)}


def containers(lock):
    """Each rank's host, model container name and, for Compose deployments, container labels."""
    return [{"rank": row["rank"], "host": row["host"], "name": container_name(lock, row["rank"]),
             **({"labels": container_labels(lock, row["rank"])} if lock["backend"] == "compose" else {})}
            for row in lock["site"]["ranks"]]


def rendered(lock):
    files = {}
    for number, spec in enumerate(specifications(lock)):
        files[f"rank{number}/compose.yaml"] = compose.compose_text(spec, lock["selection"]["image_reference"])
        files[f"rank{number}/container.json"] = compose.encoded(spec.document())
    return files


def connection(lock):
    """The API of the deployment: ``api_url``, the served ``model`` and the ``port``.

    The API answers on the API rank at the deployment's port (serving setting
    ``api_port``, else the profile's). Its address is the listen address
    (serving setting ``api_bind``), else the site's ``api_address`` (Node A's
    LAN address, or Spark 2's for a half on Sparks 2 and 3), else the API
    rank's management address. SparkRing's own checks use this URL; an address
    named for display (runtime.host.api_endpoint) never replaces it here.
    """
    if lock["backend"] in ("glm-managed", "glm-existing-mesh"):
        from runtime.common import glm_tp4
        port = int(glm_tp4.DEFAULTS["PORT"])
        model = "GLM-5.3-Flash-NVFP4-" + ("QAD" if lock["selection"]["target_variant"] == "nvfp4-qad" else "Spark") + "-TP4"
    else:
        args = specifications(lock, only_rank=0)[0].command
        port = int(args[args.index("--port") + 1])
        model = args[args.index("--served-model-name") + 1]
    host_ip = ((lock.get("serving") or {}).get("api_bind")
               or lock["site"].get("api_address", lock["site"]["ranks"][0]["management_ip"]))
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
        if derived_checkpoint.model_of(lock["selection"]) is not None:
            # A derived checkpoint is written from the verified base inside the
            # serving image, so its phase follows both; sparkring install runs
            # the derivation (install_assets.Assets.derive) when the first rank
            # reaches it, and every rank then verifies its own directory.
            phases.append(phase("derive", ranks, "mutates-host", "derive-check"))
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
            # On a four-Spark ring, ring-stop first stops, on every rank, each
            # mesh that the mesh start (mesh-up or ring-serve) starts again.
            # Every rank's mesh start then sees the same running meshes when it
            # decides whether to install this deployment's mesh code
            # (native_mesh.update_code).
            if "native_mesh" in lock["site_input"]:
                phases += [phase("mesh-prepare", ranks, "mutates-host", "mesh-prepared"),
                           phase("create", ranks, "starts-model", "created"),
                           phase("mesh-install", ranks[:1], "mutates-host", "mesh-installed"),
                           phase("mesh-replace", ranks, "mutates-host", "mesh-replaced"),
                           phase("ring-stop", ranks, "mutates-host", "ring-stopped"),
                           phase("mesh-up", ranks, "mutates-host", "mesh-up-check"),
                           phase("mesh-gate", ranks), phase("preflight", ranks)]
            else:
                if len(ranks) == 4 and lock["backend"] != "glm-existing-mesh":
                    # A reused mesh is started, repaired and awaited on all
                    # four ranks before the read-only ring check.
                    phases += [phase("ring-stop", ranks, "mutates-host", "ring-stopped"),
                               phase("ring-serve", ranks, "mutates-host", "ring-check")]
                elif len(ranks) == 2:
                    if lock["site"].get("placement"):
                        # A half of a four-Spark ring uses ports that the
                        # ring's four-rank mesh also forwards on. Its two
                        # Sparks stop and disable their mesh services first;
                        # a four-rank deployment's ring-serve enables them
                        # again (native_mesh.park_local, serve_ring).
                        phases += [phase("ring-park", ranks, "mutates-host", "ring-parked")]
                    # A pair's fabric addresses return to RoCE GID index 3,
                    # which they leave when the cabled Spark restarts while
                    # the model runs.
                    phases += [phase("gid-serve", ranks, "mutates-host", "gid-check")]
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
        restart = False
        if action == "up" and state["operation"] == action and state["complete"] and lock["backend"] == "compose":
            # A repeated up resumes its completed receipt, which only
            # re-verifies each action. When no rank runs the model any more,
            # for example after the Sparks restarted, a new generation runs
            # every phase instead: it restores the RoCE GIDs and the CDI spec
            # and starts the containers. A rank that still runs the model holds
            # the GID entries those phases repair (roce_gid.serve), so the model
            # must stop everywhere first. Only the host's own "not running"
            # answer counts as stopped; a Spark that does not answer stops up.
            stopped, total = [], 0
            for step in plan["phases"]:
                if step["id"] not in ("start-workers", "start-api"):
                    continue
                for item in step["actions"]:
                    total += 1
                    result = runner(item["host"], item["verify"]["argv"], item["verify"]["timeout"])
                    if deploy_engine.verified(result, item["verify"]):
                        continue
                    if "Rank is not running" not in result["stderr"]:
                        detail = (result["stderr"].strip().splitlines() or ["no answer"])[-1]
                        raise ValueError(f"{item['host']} did not report whether its model container runs: {detail}")
                    stopped.append(item["host"])
            if stopped and len(stopped) < total:
                raise ValueError("The model runs on some Sparks but not on " + ", ".join(stopped) + ". Stop it everywhere "
                                 "with sparkring down --execute, then start it with sparkring up --execute.")
            restart = bool(stopped)
        if state["operation"] != action or retry or restart:
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


def unfinished(directory):
    """The deployment's last operation (``prepare``, ``up`` or ``down``) when it did not complete, else None.

    After such an operation ``apply`` accepts ``down`` and the same operation
    again: a preparation repeats as a new generation, and a start or stop
    resumes its receipt, whose uncertain actions it refuses.
    """
    path = Path(directory) / "state.json"
    if not path.exists():
        return None
    state = read(path)
    return None if state["complete"] else state["operation"]


def identity(lock):
    """The checkpoint and image release that a deployment lock selects.

    For a derived checkpoint, ``model_repository`` and ``model_revision`` name
    its base and ``derived`` the derived checkpoint's own repository and revision.
    """
    card = lock["selection"]
    result = {"checkpoint": card["target_variant"], "model_repository": card["model_repository"],
              "model_revision": card["model_revision"], "image_release": card["release"]}
    derived = derived_checkpoint.model_of(card)
    if derived is not None:
        result["derived"] = {"repository": derived["repository"], "revision": derived["revision"]}
    return result


def status(directory):
    directory = Path(directory)
    lock = load(directory)
    state = read(directory / "state.json") if (directory / "state.json").exists() else {"operation": "not started", "complete": False}
    result = {"profile": lock["selection"]["profile"], "state": state,
              "deployment_id": lock["id"], "source_revision": lock["source_revision"],
              "image_id": lock["selection"]["image_id"], "image_reference": lock["selection"]["image_reference"],
              "live_observed": False, "hosts": [r["host"] for r in lock["site"]["ranks"]],
              **identity(lock), **connection(lock)}
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
        if derived_checkpoint.model_of(lock["selection"]) is not None:
            # A derived checkpoint is written beside its base's SparkRing checkpoint directory.
            for row in example["hosts"]:
                row["model"] = checkpoint_directory("example", lock["selection"])
        runtime = lock.get("image_runtime")
        if runtime is not None:
            runtime = {**runtime, "image_reference": runtime["image_id"]}
            files["image-lock.json"] = compose.encoded(runtime)
        # Serving settings are part of the deployment, not of its site, so the
        # template renders them and its init command names them.
        settings = lock.get("serving") or {}
        portable = make_lock(lock["selection"]["profile"], example, lock["source_revision"],
                             lock["bundle_sha256"], lock["selection"]["target_variant"], image_runtime=runtime,
                             settings=settings)
        flags = "".join(" " + serving.label(name, value) for name, value in sorted(settings.items()))
        files["site.example.json"] = compose.encoded(example)
        files["profile.json"] = compose.encoded(lock["selection"])
        files["README.txt"] = ("Portable profile template, not a configured deployment. Fill site.example.json, then run "
                               "sparkring init --profile " + lock["selection"]["profile"] + " --site site.example.json"
                               + (" --image-lock image-lock.json" if runtime else "") + flags + ".\n"
                               "Compose files below contain example hosts/paths. Re-render for your site. A separate "
                               "Compose project runs on each rank; Compose alone does not configure RDMA or coordinate hosts.\n")
        if settings:
            files["README.txt"] += "Serving settings, applied in the Compose files in place of the profile's values:" + flags + ".\n"
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
