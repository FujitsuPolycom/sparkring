"""Data and page of the Compose builder.

The page writes the `sparkring install` command for a profile, checkpoint and
serving settings, and the per-rank Compose files that `sparkring compose
render` writes for a site. It runs no Python. For every profile, checkpoint and
save-CPU switch state, export() renders the deployment with compose.build for a
sentinel site whose values are unique tokens; engine.js substitutes a site's
values for those tokens line by line. export() refuses to produce data unless
every token appears only on the lines the engine rewrites and the site fields
that never reach a container file are absent from them. verify.py compares the
engine with compose.build for random sites.
"""
import json
from pathlib import Path
import re
import subprocess

from runtime.common import compose, profiles, qwen_flash_next
from runtime.common import serving as serving_settings

HERE = Path(__file__).resolve().parent
ROOT = compose.ROOT
SCHEMA = "sparkring-compose-builder-data/v1"
REPOSITORY = "FujitsuPolycom/sparkring"
# Compose-supported profiles the builder does not list.
EXCLUDED = frozenset({"qwen38-flash-next-qad-tp4-sparkcache"})

# Lines on which a substituted site value may appear: these YAML keys, or a list item ("-").
SITE_KEYS = {"name", "container_name", "VLLM_HOST_IP", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME",
             "B12X_ROCE_HCA", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "B12X_ROCE_GID_INDEX", "source",
             compose.LABEL, "-"}
GID_KEYS = {"NCCL_IB_GID_INDEX", "B12X_ROCE_GID_INDEX"}
# Site fields that no container file contains; their sentinels must not appear.
ABSENT = ("zqhost", "zqdeploy", "zqfabric")


def profile_ids():
    """The profiles the builder lists, in compose.SUPPORTED order."""
    return [profile_id for profile_id in compose.SUPPORTED if profile_id not in EXCLUDED]


def example_site(profile_id):
    owner = profile_id.removesuffix("-sparkcache")
    return compose.read_site(ROOT / "profiles" / owner / "compose" / "site.example.yaml")


def _argument(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


def _setting_rows(command):
    """The serving settings a command sets, with the command's values and their limits."""
    rows = []
    for name, (flag, key, _, minimum, text) in serving_settings.SETTINGS.items():
        value = serving_settings.profile_value(command, name)
        if value is None:
            continue
        rows.append({"name": name, "option": serving_settings.option(name), "flag": flag + (f" ({key})" if key else ""),
                     "profile": value, "minimum": minimum,
                     "maximum": serving_settings.ceiling(value) if name in serving_settings.ABOVE_PROFILE else None,
                     "help": text})
    for name, (variable, value, text) in serving_settings.SWITCHES.items():
        rows.append({"name": name, "option": serving_settings.option(name), "flag": variable, "switch": True,
                     "value": value, "help": text})
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


def profile_data(profile_id):
    """One profile's facts, checkpoints and sentinel templates."""
    metadata, _ = profiles.load(profile_id)
    configuration = qwen_flash_next.read(ROOT / metadata["configuration"]["path"])
    example = example_site(profile_id)
    runtime = compose.installer_image_runtime(profile_id)
    options = {"image_runtime": runtime} if runtime is not None else {}
    specs, image = compose.specifications(profile_id, example, **options)
    checkpoints = _checkpoints(profile_id, configuration, example, options)
    site = sentinel_site(example)
    for checkpoint in checkpoints:
        name = None if checkpoint["default"] else checkpoint["name"]
        variants = {}
        for variant, serving in (("off", None), ("on", {"save_cpu": True})):
            manifest, files = compose.build(profile_id, site, checkpoint=name, serving=serving)
            ranks = []
            for n in range(len(site["ranks"])):
                label = f"{profile_id} {checkpoint['name']} save-cpu {variant} rank{n}"
                _check_template(files[f"rank{n}/compose.yaml"], site, n, manifest["id"], _yaml_key, label + " compose.yaml")
                _check_template(files[f"rank{n}/container.json"], site, n, manifest["id"], _json_key, label + " container.json")
                ranks.append({"compose": files[f"rank{n}/compose.yaml"], "container": files[f"rank{n}/container.json"]})
            variants[variant] = {"identity": manifest["id"], "ranks": ranks}
        checkpoint["variants"] = variants
    inputs = compose.source_inventory(profile_id)
    return {
        "id": profile_id,
        "title": metadata.get("title", profile_id),
        "nodes": len(example["ranks"]),
        "sparkcache": profile_id.endswith("-sparkcache"),
        "installable": profile_id in (runtime or {}).get("profiles", []),
        "image": image,
        "image_id": specs[0].image_id,
        "image_release": (runtime or {}).get("name") or Path(metadata["release"]).parent.name,
        "checkpoints": checkpoints,
        "example_site": example,
        "inputs": inputs,
        "identity_inventory": compose.identity_inventory(inputs),
        "options": options,
    }


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
    """The page's data: every listed profile, and the source the install command pins.

    ``ref`` is the tag when the checkout is a release, else the commit: the
    install command fetches install.sh from it and passes it as --ref. For a
    checkout between releases, ``since_tag`` and ``commits_since`` name the
    release before it and how many commits it is ahead.
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
            "since_tag": since_tag, "commits_since": commits_since,
            "profiles": [profile_data(profile_id) for profile_id in profile_ids()]}


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
