"""Data and page of the Install Builder.

The page writes the `sparkring install` command for a profile, checkpoint and
serving settings, and the per-rank Compose files that `sparkring compose
render` writes for a site. It runs no Python. For every profile, checkpoint and
save-CPU switch state, export() renders the deployment with compose.build for a
sentinel site whose values are unique tokens; engine.js substitutes a site's
values for those tokens line by line. export() refuses to produce data unless
every token appears only on the lines the engine rewrites and the site fields
that never reach a container file are absent from them. verify.py compares the
engine with compose.build for random sites.

Each profile names its image's capabilities (installer_image.capabilities):
the page offers the save-CPU switch, and export() renders its "on" variant,
only on an image whose vLLM reads the switch's variable.

The page's command pack offers what the source's commands accept: export()
records in ``features`` whether `sparkring install` defines ``--on`` (a
two-Spark model on half of a four-Spark ring) and ``--api-address`` (the
address shown for the model) and whether `sparkring cabling` defines
``--bandwidth`` (a measurement of every cable), read from the argument
parsers in the source files, and whether the source's serving settings
include the API endpoint's port and listen address (``api_port``,
``api_bind``), read from runtime/common/serving.py.

Each checkpoint carries ``capacity``: the engine-reported KV pool of
performance/profile-capacity.json that the page scales to the chosen KV cache
size for its token estimate, with the measurement's record, conditions and
KV-size evidence, which the page names beside the estimate
(kv_measurement), or None when the profile has no usable measurement.

Each profile and checkpoint carries ``status`` and ``purpose``, which the page
shows beside it. A profile's status is its profile.json ``status``; its
purpose, and each other checkpoint's status, purpose and evidence, come from
profiles/labels.json (labels()). They live there rather than in the profile's
configuration because the deployment identity covers each checkpoint entry
of the configuration (compose.identity_inventory), so a label written there
would make every deployment of the profile another deployment.
"""
import ast
import json
from pathlib import Path
import re
import subprocess

from runtime.common import compose, installer_image, profiles, qwen_flash_next
from runtime.common import serving as serving_settings

HERE = Path(__file__).resolve().parent
ROOT = compose.ROOT
SCHEMA = "sparkring-compose-builder-data/v1"
REPOSITORY = "FujitsuPolycom/sparkring"
# The status values of profiles and checkpoints (docs/development/writing.md).
STATUSES = ("qualified", "implemented", "research-only", "unsupported")
# Compose-supported profiles the builder does not list.
EXCLUDED = frozenset({"qwen38-flash-next-tp2-sparkcache", "qwen38-flash-next-qad-tp4-sparkcache"})

# Lines on which a substituted site value may appear: these YAML keys, or a list item ("-").
SITE_KEYS = {"name", "container_name", "VLLM_HOST_IP", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME",
             "B12X_ROCE_HCA", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "B12X_ROCE_GID_INDEX", "source",
             compose.LABEL, "-"}
GID_KEYS = {"NCCL_IB_GID_INDEX", "B12X_ROCE_GID_INDEX"}
# Site fields that no container file contains; their sentinels must not appear.
ABSENT = ("zqhost", "zqdeploy", "zqfabric")
# The command options the page's command pack uses, by feature: the source file whose
# argument parser defines the option, and the option. A feature is offered only when its
# file defines the option.
FEATURES = {
    # sudo sparkring install --on 0,1 | 2,3: a two-Spark model on one half of a four-Spark ring.
    "ring_halves": ("runtime/host/install_workflow.py", "--on"),
    # sudo sparkring install ... --on 0,1 --and ... --on 2,3: both halves' models in one run.
    "joined_halves": ("runtime/host/install_workflow.py", "--and"),
    # sudo sparkring cabling --bandwidth: measure the bandwidth of every cable.
    "cable_check": ("runtime/host/cabling.py", "--bandwidth"),
    # sudo sparkring install --api-address ADDRESS: the address shown for the model.
    "api_address": ("runtime/host/install_workflow.py", "--api-address"),
}
# The serving settings the page's API endpoint option uses, by feature: the
# source file that defines the serving settings, and the setting's name, a key
# of its SETTINGS. `sparkring install` takes each as an option (--api-port,
# --api-bind) and, in a terminal without --yes, asks for both.
SETTING_FEATURES = {
    "api_port": ("runtime/common/serving.py", "api_port"),
    "api_bind": ("runtime/common/serving.py", "api_bind"),
}


def profile_ids():
    """The profiles the builder lists, in compose.SUPPORTED order."""
    return [profile_id for profile_id in compose.SUPPORTED if profile_id not in EXCLUDED]


def example_site(profile_id):
    return compose.read_site(ROOT / "profiles" / profile_id / "compose" / "site.example.yaml")


def _argument(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def _setting_rows(command):
    """The serving settings a command sets, with the command's values and their limits.

    A switch that needs an image capability (serving.NEEDS) names it in
    ``needs``; the profile's ``image_capabilities`` say whether its image has it.

    ``maximum`` is the ceiling of an ABOVE_PROFILE setting and ``largest``
    the largest whole number a setting takes (serving.MAXIMUM). The API
    endpoint's settings, which the page offers in its API endpoint option,
    carry ``endpoint``; ``api_port`` lists the ports it may not take
    (serving.reserved_ports) in ``reserved``, and a listen address
    (serving.ADDRESSES) carries ``address``.
    """
    rows = []
    for name, (flag, key, _, minimum, text) in serving_settings.SETTINGS.items():
        value = serving_settings.profile_value(command, name)
        if value is None:
            continue
        row = {"name": name, "option": serving_settings.option(name), "flag": flag + (f" ({key})" if key else ""),
               "profile": value, "minimum": minimum,
               "maximum": serving_settings.ceiling(value) if name in serving_settings.ABOVE_PROFILE else None,
               "help": text}
        if name in serving_settings.MAXIMUM:
            row["largest"] = serving_settings.MAXIMUM[name]
        if name == "api_port":
            row.update(endpoint=True, reserved=[list(item) for item in serving_settings.reserved_ports(command)])
        if name in serving_settings.ADDRESSES:
            row.update(endpoint=True, address=True)
        rows.append(row)
    for name, (variable, value, text) in serving_settings.SWITCHES.items():
        row = {"name": name, "option": serving_settings.option(name), "flag": variable, "switch": True,
               "value": value, "help": text}
        if name in serving_settings.NEEDS:
            row["needs"] = serving_settings.NEEDS[name]
        rows.append(row)
    return rows


def _changes(entry):
    """What a checkpoint entry changes relative to the profile's default, as short lines."""
    lines = []
    if "derived" in entry:
        lines.append(f"derived on the Sparks from {entry['derived']['base']} and {entry['derived']['donor']}")
    if "served_model_name" in entry:
        lines.append("served as " + entry["served_model_name"])
    lines.extend(f"{flag} {value}" for flag, value in (entry.get("arguments") or {}).items())
    lines.extend(f"{name}={value}" for name, value in (entry.get("environment") or {}).items())
    lines.extend(f"speculative {key}: {value}" for key, value in (entry.get("speculative") or {}).items())
    return lines


def _checkpoints(profile_id, configuration, site, options):
    """The default checkpoint first, then every other listed one, with the command facts each selects."""
    default, names = qwen_flash_next.checkpoint_names(configuration)
    aliases = configuration.get("checkpoint_aliases") or {}
    rows = []
    for name in ([default, *sorted(n for n in names if n != default)] if default else [None]):
        model = qwen_flash_next.checkpoint_settings(configuration, name).get("model") or {}
        specs, _ = compose.specifications(profile_id, site, checkpoint=None if name == default else name, **options)
        command = list(specs[0].command)
        entry = (configuration.get("checkpoints") or {}).get(name, {})
        rows.append({
            "name": name,
            "default": name == default,
            "aliases": sorted(alias for alias, target in aliases.items() if target == name),
            "derived": "derived" in entry,
            "model_repository": model.get("repository"),
            "model_revision": model.get("revision"),
            "served_model_name": _argument(command, "--served-model-name"),
            "port": _argument(command, "--port"),
            "settings": _setting_rows(command),
            "changes": [] if name == default else _changes(entry),
        })
    return rows


def labels(root=ROOT):
    """profiles/labels.json (sparkring-profile-labels/v1), validated: a purpose by profile, and by checkpoint name.

    ``profiles`` maps a profile ID to ``purpose``, one line, and
    ``checkpoints``, which maps each checkpoint name the profile lists to its
    ``purpose``. A checkpoint other than the profile's default also has
    ``status``, one of STATUSES, and ``evidence``, the repository file (with
    an optional ``#`` anchor) that establishes it; the default checkpoint has
    the profile's own status.
    """
    document = json.loads((Path(root) / "profiles" / "labels.json").read_text(encoding="utf-8"))
    if set(document) != {"schema", "profiles"} or document["schema"] != "sparkring-profile-labels/v1":
        raise ValueError("profiles/labels.json: expected sparkring-profile-labels/v1 with profiles")
    for profile_id, row in document["profiles"].items():
        if (not isinstance(row, dict) or set(row) != {"purpose", "checkpoints"} or not isinstance(row["purpose"], str)
                or not row["purpose"] or not isinstance(row["checkpoints"], dict)):
            raise ValueError(f"profiles/labels.json: {profile_id} needs a purpose and its checkpoints")
        for name, entry in row["checkpoints"].items():
            keys = set(entry) if isinstance(entry, dict) else set()
            if keys not in ({"purpose"}, {"purpose", "status", "evidence"}) or not entry["purpose"]:
                raise ValueError(f"profiles/labels.json: {profile_id} {name} needs a purpose, and a status and evidence or neither")
            if "status" in entry and (entry["status"] not in STATUSES
                                      or not (Path(root) / entry["evidence"].split("#", 1)[0]).is_file()):
                raise ValueError(f"profiles/labels.json: {profile_id} {name} needs a status of {', '.join(STATUSES)} "
                                 "and an evidence file of this repository")
    return document


def _labelled(profile_id, metadata, checkpoints, document):
    """The profile's status and purpose, with each checkpoint's in place; refuses a profile or checkpoint without labels."""
    row = document["profiles"].get(profile_id)
    names = [checkpoint["name"] for checkpoint in checkpoints if checkpoint["name"] is not None]
    if row is None or sorted(row["checkpoints"]) != sorted(names):
        raise ValueError(f"profiles/labels.json must label {profile_id} and exactly its checkpoints: {', '.join(names) or 'none'}")
    for checkpoint in checkpoints:
        entry = row["checkpoints"].get(checkpoint["name"], {})
        if checkpoint["default"] == ("status" in entry):
            raise ValueError(f"profiles/labels.json: {profile_id} {checkpoint['name']}: only a checkpoint other than the "
                             "default has its own status")
        checkpoint.update(status=entry.get("status", metadata["status"]), purpose=entry.get("purpose"),
                          evidence=entry.get("evidence"))
    return {"status": metadata["status"], "purpose": row["purpose"]}


def capacity_records():
    """performance/profile-capacity.json's records by profile ID: the engine-reported KV pools."""
    return json.loads((ROOT / "performance" / "profile-capacity.json").read_text(encoding="utf-8"))["profiles"]


def kv_measurement(record, checkpoint, default):
    """The engine-reported KV pool that sizes a checkpoint of a profile, from the profile's capacity record.

    ``checkpoint`` is the checkpoint's name and ``default`` the profile's
    default checkpoint name (None for a profile without named checkpoints).
    A measurement of that checkpoint wins: one in the record's
    ``checkpoints``, or the record's own when it measured that checkpoint
    (its ``checkpoint``, else the profile's default). Otherwise the record's
    own measurement, of another checkpoint of the same profile, sizes it. A
    measurement without ``kv_bytes_per_rank`` cannot be scaled to another KV
    size and sizes nothing. Returns {tokens, kv_bytes_per_rank, checkpoint,
    source, conditions, kv_evidence} or None: ``source`` is the repository
    file that records the measurement, ``conditions`` how it was measured
    (image, configuration and what the figure does not prove) and
    ``kv_evidence`` where its KV size comes from, each None when the
    measurement does not name it. A record is for one profile, so a
    two-Spark measurement never sizes a four-Spark profile.
    """
    if not record:
        return None
    other = (record.get("checkpoints") or {}).get(checkpoint)
    measured, name = (other, checkpoint) if other else (record, record.get("checkpoint") or default)
    if not measured.get("kv_bytes_per_rank"):
        return None
    return {"tokens": measured["tokens"], "kv_bytes_per_rank": measured["kv_bytes_per_rank"], "checkpoint": name,
            **{key: measured.get(key) for key in ("source", "conditions", "kv_evidence")}}


def sentinel_site(example):
    """The example site with every host value replaced by a unique token."""
    site = json.loads(json.dumps(example))
    site["name"] = "zqsite"
    for row in site["ranks"]:
        n = row["rank"]
        row.update(host=f"zqhost{n}", host_ip=f"198.18.{n}.77", interface=f"zqif{n}",
                   hcas=[f"zqhca{n}{c}" for c in "abcd"[:len(row["hcas"])]], gid=201 + n,
                   model=f"/zqmodel{n}/m", cache=f"/zqcache{n}/c", repository=f"/zqrepo{n}/r",
                   deployment_root=f"/zqdeploy{n}/d")
        if "fabric" in row:
            row["fabric"] = {"site_path": f"/zqfabric{n}/site.json", "site_sha256": "c" * 63 + str(n),
                             "plan_sha256": "d" * 63 + str(n)}
    site["master"] = site["ranks"][0]["host_ip"]
    return site


def _tokens(site, n, identity):
    row = site["ranks"][n]
    tokens = ["zqsite", site["master"], row["host_ip"], row["interface"], row["model"], row["cache"],
              row["repository"], identity, *row["hcas"]]
    return [token for token in tokens if token]


def _yaml_key(line):
    stripped = line.strip()
    return "-" if stripped.startswith("- ") else stripped.split(":", 1)[0]


def _json_key(line):
    match = re.match(r'\s*"([^"]+)":', line)
    return match.group(1) if match else "-"


def _check_template(text, site, n, identity, key_of, label):
    tokens = _tokens(site, n, identity)
    gid = str(site["ranks"][n]["gid"])
    for line in text.splitlines():
        for absent in ABSENT:
            if absent in line:
                raise ValueError(f"{label}: site field {absent} reached a container file: {line.strip()}")
        hit = any(token in line for token in tokens)
        key = key_of(line)
        if re.search(r"zq|198\.18\.", line) and not hit:
            raise ValueError(f"{label}: unrecognized sentinel on line: {line.strip()}")
        if hit and key not in SITE_KEYS:
            raise ValueError(f"{label}: a site value reached key {key}: {line.strip()}")
        if key in GID_KEYS and gid not in line:
            raise ValueError(f"{label}: GID line without the GID sentinel: {line.strip()}")


def profile_data(profile_id, image_runtime=None, image_option=None):
    """One profile's facts, checkpoints and sentinel templates, on the default or the given installer image.

    ``image_option`` is the `--image` value that selects ``image_runtime``; the
    page's commands pass it when it is set.
    """
    metadata, _ = profiles.load(profile_id)
    configuration = qwen_flash_next.read(ROOT / metadata["configuration"]["path"])
    example = example_site(profile_id)
    runtime = image_runtime or compose.installer_image_runtime(profile_id)
    options = {"image_runtime": runtime} if runtime is not None else {}
    specs, image = compose.specifications(profile_id, example, **options)
    checkpoints = _checkpoints(profile_id, configuration, example, options)
    record = capacity_records().get(profile_id)
    for checkpoint in checkpoints:
        checkpoint["capacity"] = kv_measurement(record, checkpoint["name"], checkpoints[0]["name"])
    status = _labelled(profile_id, metadata, checkpoints, labels())
    site = sentinel_site(example)
    release = (runtime or {}).get("name") or Path(metadata["release"]).parent.name
    capabilities = list(installer_image.capabilities(release))
    # The save-CPU switch renders only on an image that reads its variable; elsewhere it is refused.
    renders = [("off", None)] + ([("on", {"save_cpu": True})] if serving_settings.NEEDS["save_cpu"] in capabilities else [])
    for checkpoint in checkpoints:
        name = None if checkpoint["default"] else checkpoint["name"]
        variants = {}
        for variant, serving in renders:
            manifest, files = compose.build(profile_id, site, checkpoint=name, serving=serving, image_runtime=runtime)
            ranks = []
            for n in range(len(site["ranks"])):
                label = f"{profile_id} {checkpoint['name']} save-cpu {variant} rank{n}"
                _check_template(files[f"rank{n}/compose.yaml"], site, n, manifest["id"], _yaml_key, label + " compose.yaml")
                _check_template(files[f"rank{n}/container.json"], site, n, manifest["id"], _json_key, label + " container.json")
                ranks.append({"compose": files[f"rank{n}/compose.yaml"], "container": files[f"rank{n}/container.json"]})
            variants[variant] = {"identity": manifest["id"], "ranks": ranks}
        checkpoint["variants"] = variants
    inputs = compose.source_inventory(profile_id)
    # The model's display name, from the default checkpoint's repository; the profile title names that checkpoint too.
    names = json.loads((ROOT / "profiles" / "model-names.json").read_text(encoding="utf-8"))["models"]
    return {
        "id": profile_id,
        "title": metadata.get("title", profile_id),
        "model_name": names.get(checkpoints[0]["model_repository"], metadata.get("title", profile_id)),
        **status,
        "nodes": len(example["ranks"]),
        "installable": profile_id in (runtime or {}).get("profiles", []),
        "image": image,
        "image_id": specs[0].image_id,
        "image_release": release,
        "image_capabilities": capabilities,
        "image_option": image_option,
        "checkpoints": checkpoints,
        "example_site": example,
        "inputs": inputs,
        "identity_inventory": compose.identity_inventory(inputs),
        "options": options,
    }


def command_options(path):
    """The option strings that the ``add_argument`` calls of a Python source file define, such as ``--on``."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            found.update(arg.value for arg in node.args
                         if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"))
    return found


def setting_names(path):
    """The keys of the ``SETTINGS`` dictionary that a Python source file assigns, such as ``api_port``."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SETTINGS"
                                                 for target in node.targets) and isinstance(node.value, ast.Dict)):
            return {key.value for key in node.value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    return set()


def features(root=ROOT):
    """``{feature: bool}`` for every FEATURES and SETTING_FEATURES entry: whether the source at ``root`` defines it.

    A missing source file offers nothing.
    """
    result = {}
    for name, (relative, option) in FEATURES.items():
        path = Path(root) / relative
        result[name] = path.is_file() and option in command_options(path)
    for name, (relative, setting) in SETTING_FEATURES.items():
        path = Path(root) / relative
        result[name] = path.is_file() and setting in setting_names(path)
    return result


def _git(*argv):
    result = subprocess.run(["git", "-C", str(ROOT), *argv], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_identity():
    """(tag, commit) of the checkout: its exact release tag or None, and its commit or None outside Git."""
    return _git("describe", "--tags", "--exact-match", "HEAD"), _git("rev-parse", "HEAD")


def release_distance(description):
    """(tag, commits) from `git describe --tags --long` output such as ``2026.10.0-24-g66f04d51``.

    The page names a checkout between releases by the release before it and the
    number of commits since; (None, None) when there is no earlier tag.
    """
    match = re.fullmatch(r"(.+)-(\d+)-g[0-9a-f]+", description or "")
    return (match.group(1), int(match.group(2))) if match else (None, None)


def export(*, tag=None, commit=None, repository=REPOSITORY):
    """The page's data: every listed profile, the source the install command pins and its command features.

    ``ref`` is the tag when the checkout is a release, else the commit: the
    install command fetches install.sh from it and passes it as --ref. For a
    checkout between releases, ``since_tag`` and ``commits_since`` name the
    release before it and how many commits it is ahead. ``features`` is
    features() of the checkout.
    """
    since_tag = commits_since = None
    if commit is None:
        found_tag, commit = source_identity()
        tag = tag or found_tag
        if tag is None:
            since_tag, commits_since = release_distance(_git("describe", "--tags", "--long", "HEAD"))
    if not commit:
        raise ValueError("The builder needs the checkout's commit; pass --commit outside Git")
    return {"schema": SCHEMA, "repository": repository, "tag": tag, "commit": commit, "ref": tag or commit,
            "since_tag": since_tag, "commits_since": commits_since, "features": features(),
            "image": None, "images": image_catalog(),
            "profiles": [profile_data(profile_id) for profile_id in profile_ids()]}


def image_catalog():
    """The installer images `sparkring images` lists, default first, with the listed profiles each runs.

    ``option`` is the value `--image` takes: the release tag that published the
    image when there is one, else the first part of its release name that
    contains a letter and names no other image (``statusrows``), else its
    release name; installer_image.lock_path resolves each form. ``file`` is the
    data file of a non-default image, which the page loads when the image is
    selected.
    """
    listed = set(profile_ids())
    catalog = installer_image.catalog()
    names = [row["name"] for row in catalog]
    rows = []
    for row in catalog:
        runs = [profile_id for profile_id in installer_image.profiles_of(row["lock"]) if profile_id in listed]
        if not runs:
            continue
        rows.append({"name": row["name"], "default": row["default"], "tags": row["tags"],
                     "option": _image_option(row, names),
                     "download_bytes": row["lock"].get("download_bytes"), "profiles": runs,
                     "file": None if row["default"] else f"images/{row['name']}.json"})
    return rows


def _image_option(row, names):
    if row["tags"]:
        return row["tags"][-1]
    for part in row["name"].split("-"):
        if part != "dev" and re.search("[a-z]", part) and sum(f"-{part}-" in f"-{name}-" for name in names) == 1:
            return part
    return row["name"]


def image_data(name):
    """The profiles a non-default installer image runs, rendered on it, for its data file."""
    catalog = installer_image.catalog()
    row = next(row for row in catalog if row["name"] == name)
    option = _image_option(row, [row["name"] for row in catalog])
    runs = [profile_id for profile_id in profile_ids() if profile_id in installer_image.profiles_of(row["lock"])]
    return {"schema": SCHEMA, "image": name,
            "profiles": [profile_data(profile_id, row["lock"], option) for profile_id in runs]}


def engine_source():
    """engine.js without its Node export, for the page's inline script."""
    text = (HERE / "engine.js").read_text(encoding="utf-8")
    text = text.replace("if (typeof module !== 'undefined') module.exports = SparkRingEngine;\n", "")
    if "</script" in text.lower():
        raise ValueError("engine.js must not contain a closing script tag")
    return text


def page(data, verification=None, *, standalone=True):
    """index.html: page.html with the data, the verification summary and the engine inlined.

    ``standalone`` makes a complete document, with the charset, viewport, title
    and style in its head, for a static host such as GitHub Pages. Without it
    the result is page.html's content alone, for a host that supplies the
    document around it, such as a claude.ai artifact.
    """
    template = (HERE / "page.html").read_text(encoding="utf-8")
    if template.count("__DATA__") != 1 or template.count("__ENGINE__") != 1:
        raise ValueError("page.html needs exactly one __DATA__ and one __ENGINE__ placeholder")
    document = {**data, "verification": verification}
    # JSON inside <script type="application/json">: escaping "<" keeps "</script" out of it.
    text = json.dumps(document, separators=(",", ":")).replace("<", "\\u003c")
    content = template.replace("__DATA__", text).replace("__ENGINE__", engine_source())
    if not standalone:
        return content
    head, body = content.split("</style>", 1)
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + "</style>\n</head>\n<body>" + body + "</body>\n</html>\n")
