"""Resolve versioned deployment profiles without host access or environment leakage."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
STATUS = {"implemented", "qualified", "research-only", "unsupported"}
OVERRIDES = {"max_model_len", "max_num_seqs", "max_num_batched_tokens"}
PROFILE_FIELDS = {"schema", "id", "title", "recommendation", "status", "configuration", "release", "guide", "evidence_scope", "overrides", "launcher"}
COMMON = {"decode_context_parallel_size": 1, "pipeline_parallel_size": 1}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate key {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)


def local_path(value, root=ROOT):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Repository references must be nonempty POSIX relative paths")
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping repository file: {value}")
    return path


def catalog(root=ROOT):
    data = read_json(root / "profiles/catalog.json")
    if set(data) != {"schema", "profiles"} or data["schema"] != "sparkring-catalog/v1":
        raise ValueError("profiles/catalog.json: expected sparkring-catalog/v1")
    result = {}
    for row in data["profiles"]:
        if set(row) != {"id", "path"} or row["id"] in result:
            raise ValueError("Catalog entries require unique id and path")
        result[row["id"]] = local_path(row["path"], root)
    return result


def load(profile_id, root=ROOT):
    entries = catalog(root)
    if profile_id not in entries:
        raise ValueError(f"Unknown profile: {profile_id}; use list to discover IDs")
    p = read_json(entries[profile_id])
    if set(p) != PROFILE_FIELDS or p["schema"] != "sparkring-deployment/v1":
        raise ValueError(f"{profile_id}: expected exact sparkring-deployment/v1 fields")
    if p["id"] != profile_id or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", profile_id):
        raise ValueError(f"{profile_id}: invalid or mismatched profile id")
    if p["status"] not in STATUS or p["recommendation"] not in {"recommended", "alternative", "retired"}:
        raise ValueError(f"{profile_id}: invalid evidence status or recommendation")
    if not p["evidence_scope"] or not isinstance(p["overrides"], list) or not set(p["overrides"]) <= OVERRIDES:
        raise ValueError(f"{profile_id}: specify evidence scope and supported overrides")
    adapter = p["launcher"]
    if adapter.get("kind") not in {"guide", "python", "bash"}:
        raise ValueError(f"{profile_id}: unsupported launcher")
    if adapter["kind"] != "guide":
        local_path(adapter["path"], root)
        if adapter.get("receipt"):
            local_path(adapter["receipt"], root)
    local_path(p["guide"], root)
    source = p["configuration"]
    allowed = {"format", "path", "key"} if source.get("format") == "release-profile" else {"format", "path"}
    if set(source) != allowed or source["format"] not in {"recipe", "release-profile", "serving-profile"}:
        raise ValueError(f"{profile_id}: unsupported configuration format or fields")
    local_path(source["path"], root)
    release = read_json(local_path(p["release"], root))
    if set(release) != {"schema", "id", "selection", "inputs", "image"} or release["schema"] != "sparkring-release-selection/v1":
        raise ValueError(f"{profile_id}: unsupported release selection")
    if isinstance(release["image"], dict) and "configuration_runtime" in release["image"]:
        if release["image"]["configuration_runtime"] != source["path"]:
            raise ValueError(f"{profile_id}: release runtime reference must identify its authoritative recipe")
        local_path(release["image"]["configuration_runtime"], root)
    for item in release["inputs"]:
        if set(item) != {"path", "sha256"} or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError(f"{profile_id}: invalid release input digest")
        if hashlib.sha256(local_path(item["path"], root).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"{item['path']}: release input changed; select a distinct release instead of relabeling evidence")
    return p, release


def configuration(p, root=ROOT):
    source = p["configuration"]
    data = read_json(local_path(source["path"], root))
    if source["format"] == "recipe":
        if data.get("schema") not in {"sparkring-recipe/v1", "sparkring-sparkcache-composition/v1"}:
            raise ValueError(f"{p['id']}: unsupported recipe schema {data.get('schema')}")
        if data.get("base_recipe"):
            base = read_json(local_path(data["base_recipe"], root))
            if base.get("base_recipe"):
                raise ValueError("Composition recipes reference one base recipe; nested inheritance is unsupported")
            for key in ("repository", "revision"):
                if key in data["model"] and key in base["model"] and data["model"][key] != base["model"][key]:
                    raise ValueError(f"{p['id']}: composition model {key} differs from its base recipe")
        serving = copy.deepcopy(data.get("serving", data.get("serving_common", {})))
        if data.get("profiles"):
            selected = data.get("preferred_profile")
            if selected not in data["profiles"]:
                raise ValueError(f"{p['id']}: select a defined preferred recipe profile")
            serving.update(copy.deepcopy(data["profiles"][selected]))
        serving.setdefault("tensor_parallel_size", data["hardware"]["ranks"])
        serving.setdefault("node_count", data["hardware"]["ranks"])
        return data["model"], serving, data["hardware"]["topology"], data.get("evidence", data.get("publication", {})), data.get("runtime", {})
    if source["format"] == "release-profile":
        if data.get("schema") != "sparkring-r33-profile-contract/v1":
            raise ValueError("Unsupported release-profile schema")
        selected = data["profiles"][source["key"]]
        serving = {k: selected[k] for k in ("tensor_parallel_size", "decode_context_parallel_size", "node_count", "kv_cache_memory_bytes")}
        serving.update(selected.get("serving", {}))
        serving["max_model_len"] = data["model"]["max_model_len"]
        serving["sparkcache"] = selected["sparkcache"]
        return data["model"], serving, selected["transport"], p["evidence_scope"], {"release_profile": source["key"], "template": str(Path(source["path"]).parent / selected["template"]).replace("\\", "/")}
    if data.get("schema") != "sparkring-serving-profile/v1":
        raise ValueError("Unsupported serving-profile schema")
    args = data["vllm_args"]
    serving = {}
    for key in ("tensor_parallel_size", "decode_context_parallel_size", "max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        flag = "--" + key.replace("_", "-")
        if flag in args:
            serving[key] = int(args[args.index(flag) + 1])
    serving["node_count"] = serving["tensor_parallel_size"]
    return data["model"], serving, "switched", data.get("qualification", {}), {"vllm_args": args, "environment": data["environment"]}


def resolve(profile_id, overrides=None, site=None, root=ROOT):
    p, release = load(profile_id, root)
    model, defaults, topology, evidence, runtime = configuration(p, root)
    values = {**COMMON, **defaults}
    origin = {key: ("profile" if key in defaults else "common") for key in values}
    overrides = overrides or {}
    if not set(overrides) <= set(p["overrides"]):
        raise ValueError(f"{profile_id}: unsupported override; model, topology and release identities are immutable")
    for key, value in overrides.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key}: expected a positive integer")
        values[key] = value
        origin[key] = "explicit"
    for key in ("tensor_parallel_size", "decode_context_parallel_size", "pipeline_parallel_size", "node_count", *OVERRIDES):
        if key in values and (type(values[key]) is not int or values[key] <= 0):
            raise ValueError(f"{key}: expected a positive integer")
    if values["tensor_parallel_size"] % values["decode_context_parallel_size"]:
        raise ValueError("decode_context_parallel_size must divide tensor_parallel_size")
    if values["node_count"] != values["tensor_parallel_size"]:
        raise ValueError("These GB10 profiles require one tensor-parallel rank per node")
    site = copy.deepcopy(site or {})
    if set(site) - {"schema", "nodes", "model_dir", "cache_dir"} or (site and site.get("schema") != "sparkring-site/v1"):
        raise ValueError("Site configuration accepts only schema, nodes, model_dir and cache_dir")
    if site:
        if set(site) != {"schema", "nodes", "model_dir", "cache_dir"} or len(site["nodes"]) != values["node_count"]:
            raise ValueError("Site requires model_dir, cache_dir and one node per rank")
        for rank, node in enumerate(site["nodes"]):
            if set(node) != {"rank", "address"} or type(node["rank"]) is not int or node["rank"] != rank:
                raise ValueError("Site nodes must be ordered by unique contiguous rank")
            if not isinstance(node["address"], str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", node["address"]):
                raise ValueError("Each site node requires a resolved address")
        for key in ("model_dir", "cache_dir"):
            if not isinstance(site[key], str) or not site[key].startswith("/") or any(c in site[key] for c in "<>\n\r,"):
                raise ValueError(f"{key}: use an absolute resolved host path without placeholders or commas")
        if site["model_dir"] == site["cache_dir"]:
            raise ValueError("Model and writable cache directories must be distinct")
    changed = any(values[k] != {**COMMON, **defaults}.get(k) for k in overrides)
    return {"schema": "sparkring-resolved/v1", "profile": profile_id, "model": model,
            "topology": topology, "serving": values, "origins": origin, "release": release,
            "runtime": runtime, "site": site, "status": "research-only" if changed else p["status"],
            "recommendation": p["recommendation"], "evidence_scope": p["evidence_scope"],
            "evidence": evidence, "modified_defaults": changed, "guide": p["guide"]}


def legacy_recipe_bytes(source, destination, root=ROOT):
    """Export repository-root recipe references in the historical relative format."""
    text = local_path(source, root).read_text(encoding="utf-8-sig")
    data = json.loads(text)
    if data.get("base_recipe"):
        rows = read_json(root / "profiles/compatibility.json")["mirrors"]
        locations = {row["source"]: row["destination"] for row in rows if row["kind"] == "recipe"}
        if data["base_recipe"] not in locations:
            raise ValueError("Base recipe has no compatible public export")
        relative = os.path.relpath(root / locations[data["base_recipe"]], (root / destination).parent).replace("\\", "/")
        text = text.replace(json.dumps(data["base_recipe"]), json.dumps(relative))
    return text.encode("utf-8")


def legacy_profile(path, root=ROOT):
    supplied = read_json(path)
    matches = []
    for id in catalog(root):
        p, _ = load(id, root)
        if p["configuration"]["format"] == "recipe":
            source = p["configuration"]["path"]
            canonical = read_json(local_path(source, root))
            if canonical == supplied:
                matches.append(id)
                continue
            rows = read_json(root / "profiles/compatibility.json")["mirrors"]
            exports = [row for row in rows if row["source"] == source and row["kind"] == "recipe"]
            if len(exports) == 1 and json.loads(legacy_recipe_bytes(source, exports[0]["destination"], root)) == supplied:
                matches.append(id)
    if len(matches) != 1:
        raise ValueError("Legacy recipe must match exactly one catalog definition; migrate edits to its authoritative recipe")
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "resolve", "validate", "render-env"))
    parser.add_argument("profile", nargs="?")
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--site", type=Path)
    parser.add_argument("--site-values", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=INTEGER")
    args = parser.parse_args(argv)
    try:
        if args.action == "list":
            for id in catalog():
                p, _ = load(id)
                print(f"{id}\t{p['status']}\t{p['recommendation']}")
            return
        if args.action == "validate":
            for id in catalog():
                resolve(id)
            print(f"Validated {len(catalog())} deployment profiles")
            return
        if bool(args.profile) == bool(args.legacy):
            raise ValueError("Select either a profile ID or --legacy recipe.json")
        overrides = {}
        for assignment in args.set:
            key, separator, value = assignment.partition("=")
            if not separator or key in overrides:
                raise ValueError("Overrides require unique KEY=INTEGER assignments")
            overrides[key] = int(value)
        profile_id = args.profile or legacy_profile(args.legacy)
        if args.action == 'render-env':
            from runtime.common.environment import render_environment
            if args.site or not args.site_values:
                raise ValueError('render-env requires --site-values with per-rank placeholder values')
            print(render_environment(profile_id, read_json(args.site_values), overrides), end='')
            return
        if args.site_values:
            raise ValueError('--site-values applies to render-env only')
        result = resolve(profile_id, overrides, read_json(args.site) if args.site else None)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
