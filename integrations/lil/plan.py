"""Validate and render a SparkRing image plan without contacting hosts."""

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
DIGEST = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict), "expected a JSON object")
    require(
        set(required) <= value.keys(),
        f"missing fields: {sorted(set(required) - value.keys())}",
    )
    require(
        value.keys() <= set(required) | set(optional), "unknown configuration fields"
    )


def positive(value, name):
    require(type(value) is int and value > 0, f"{name} must be a positive integer")


def path(value):
    require(isinstance(value, str), "mount path must be a string")
    p = PurePosixPath(value)
    require(
        p.is_absolute() and str(p) == value and value != "/",
        "mount paths must be normalized absolute Linux paths below /",
    )
    require(
        ".." not in p.parts and not any(c in value for c in "\n\r\x00,:"),
        "invalid mount path",
    )
    return p


def read_json(filename):
    return json.loads(Path(filename).read_text(encoding="utf-8"))


def checked_source(relative, sha256):
    require(isinstance(relative, str), "source path must be a string")
    require(
        isinstance(sha256, str) and DIGEST.fullmatch(sha256),
        "source requires a SHA-256 identity",
    )
    resolved = (ROOT / relative).resolve()
    require(resolved.is_relative_to(ROOT), "source must be inside this checkout")
    raw = resolved.read_bytes()
    # Hash logical UTF-8 source so Windows checkout line endings do not change it.
    normalized = raw.replace(b"\r\n", b"\n")
    require(
        hashlib.sha256(normalized).hexdigest() == sha256,
        f"source changed: {relative}; review and refresh descriptor",
    )
    return json.loads(normalized)


def render(descriptor, site):
    keys(
        descriptor,
        (
            "schema",
            "profile",
            "sources",
            "defaults",
            "supported",
            "lil_revision",
            "transport",
        ),
    )
    require(
        descriptor["schema"] == "sparkring-lil-descriptor/v1",
        "unsupported descriptor schema",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", descriptor["lil_revision"]) is not None,
        "invalid lil revision",
    )
    keys(descriptor["sources"], ("runtime", "models"), ("mesh", "public_image"))
    sources = {}
    for role, ref in descriptor["sources"].items():
        keys(ref, ("path", "sha256"))
        sources[role] = checked_source(ref["path"], ref["sha256"])
    keys(descriptor["transport"], ("provider", "rail_mode"))
    require(
        descriptor["transport"] == {"provider": "sircl", "rail_mode": "dual"},
        "unsupported transport mapping",
    )
    runtime = sources["runtime"]
    models = sources["models"]
    mtp = "mesh" in sources
    if mtp:
        require(
            "public_image" in sources, "MTP mesh requires its published image record"
        )
        public = sources["public_image"]
        require(
            public["checks_passed"] is True,
            "published image record has not passed verification",
        )
        runtime["operator_image"] = {
            "reference": public["public_reference"],
            "image_id": public["config_image_id"],
            "platform": public["platform"],
        }
        runtime["sircl"]["overlay_manifest_sha256"] = sources["mesh"][
            "canonical_bundle_manifest_sha256"
        ]
        target = sources["mesh"]["target"]
        models = {
            "target_model": {**target, "weight_index_sha256": target["index_sha256"]}
        }
    image = runtime["operator_image"]["reference"]
    require(
        re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image) is not None,
        "image must use an immutable registry digest",
    )
    require(
        runtime["operator_image"]["platform"] == "linux/arm64",
        "profile requires linux/arm64",
    )
    checkpoints = {}
    for role, key in (
        (("target", "target_model"),)
        if mtp
        else (("target", "target_model"), ("draft", "draft_model"))
    ):
        model = models[key]
        require(
            re.fullmatch(r"[0-9a-f]{40}", model["revision"]),
            "checkpoint revision must be immutable",
        )
        checkpoints[role] = {
            "repository": model["repository"],
            "revision": model["revision"],
            "config_sha256": model["config_sha256"],
        }
        weight_key = "weight_index_sha256" if role == "target" else "weights_sha256"
        checkpoints[role][weight_key] = model[weight_key]
        for digest_key in ("config_sha256", weight_key):
            require(
                DIGEST.fullmatch(checkpoints[role][digest_key]),
                "checkpoint requires SHA-256 identities",
            )
    keys(site, ("schema", "nodes", "storage", "cache"), ("settings",))
    require(site["schema"] == "sparkring-lil-site/v1", "unsupported site schema")
    settings = copy.deepcopy(descriptor["defaults"])
    overrides = site.get("settings", {})
    keys(overrides, (), settings.keys())
    settings.update(overrides)
    require(
        settings["speculator"] == ("mtp" if mtp else "dflash"),
        "speculator must match the pinned profile",
    )
    expected = (
        "tp",
        "dcp",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "kv_cache_memory_bytes",
        "speculator",
        "speculative_tokens",
        "port",
    )
    keys(settings, expected)
    for name in expected:
        if name != "speculator":
            positive(settings[name], name)
    require(settings["port"] <= 65535, "port must be at most 65535")
    supported = descriptor["supported"]
    keys(
        supported,
        (
            "tp",
            "dcp",
            "speculator",
            "max_num_seqs",
            "max_model_len",
            "speculative_tokens",
        ),
    )
    for name in ("tp", "dcp", "speculator", "speculative_tokens"):
        require(
            settings[name] in supported[name],
            f"unsupported {name} for this image profile",
        )
    for name in ("max_num_seqs", "max_model_len"):
        require(settings[name] <= supported[name], f"{name} exceeds profile limit")
    require(settings["tp"] % settings["dcp"] == 0, "DCP must divide TP")
    nodes = site["nodes"]
    require(
        isinstance(nodes, list) and len(nodes) == settings["tp"],
        "one node required per physical TP rank",
    )
    for node in nodes:
        keys(node, ("rank", "host"))
        require(type(node["rank"]) is int, "rank must be an integer")
        require(
            isinstance(node["host"], str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", node["host"]),
            "invalid host label",
        )
    require(
        sorted(n["rank"] for n in nodes) == list(range(settings["tp"])),
        "physical ranks must be unique and contiguous from zero",
    )
    require(
        len({n["host"] for n in nodes}) == len(nodes),
        "one distinct host required per rank",
    )
    storage = site["storage"]
    keys(
        storage,
        (("target", "jit", "cache") if mtp else ("target", "draft", "jit", "cache")),
    )
    paths = [path(v) for v in storage.values()]
    for i, left in enumerate(paths):
        for right in paths[i + 1 :]:
            require(
                not (left == right or left in right.parents or right in left.parents),
                "storage directories must not overlap",
            )
    cache = site["cache"]
    keys(cache, ("enabled",), ("namespace", "access_mode"))
    require(type(cache["enabled"]) is bool, "cache.enabled must be boolean")
    cache_plan = None
    if cache["enabled"]:
        namespace = cache.get("namespace", "")
        require(
            isinstance(namespace, str)
            and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", namespace),
            "cache namespace is required and must be a simple directory name",
        )
        mode = cache.get("access_mode", "read-write")
        require(
            mode in ("read-write", "restore-only", "store-only"),
            "unsupported cache access mode",
        )
        # Bind filesystem segregation to reviewed model and topology inputs.
        identity = {
            "profile": descriptor["profile"],
            "model_sources": descriptor["sources"],
            "image": image,
            "tp": settings["tp"],
            "dcp": settings["dcp"],
            "speculator": settings["speculator"],
            "speculative_tokens": settings["speculative_tokens"],
        }
        suffix = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()[:16]
        cache_plan = {
            "access_mode": mode,
            "namespace": f"{namespace}-{suffix}",
            "runtime_identity_required": True,
            "publication_policy_source": descriptor["sources"]["runtime"]["path"],
        }
    elif set(cache) != {"enabled"}:
        raise ValueError("disabled cache must not include ignored cache settings")
    plans = []
    for node in sorted(nodes, key=lambda n: n["rank"]):
        mounts = [
            {
                "role": "target",
                "source": storage["target"],
                "target": "/models/target",
                "read_only": True,
            },
            {
                "role": "draft",
                "source": storage.get("draft", ""),
                "target": "/dflash-draft",
                "read_only": True,
            },
            {
                "role": "jit",
                "source": storage["jit"],
                "target": "/cache/jit",
                "read_only": False,
            },
        ]
        if mtp:
            mounts = [mount for mount in mounts if mount["role"] != "draft"]
        if cache_plan:
            mounts.append(
                {
                    "role": "sparkcache",
                    "source": str(
                        PurePosixPath(storage["cache"])
                        / cache_plan["namespace"]
                        / f"rank-{node['rank']}"
                    ),
                    "target": "/cache/sparkcache",
                    "read_only": False,
                }
            )
        plans.append(
            {
                **node,
                "image": image,
                "mounts": mounts,
                "cache": cache_plan,
                "settings": settings,
            }
        )
    return {
        "schema": "sparkring-lil-plan/v1",
        "status": "research-only",
        "executable": False,
        "profile": descriptor["profile"],
        "lil_revision": descriptor["lil_revision"],
        "sources": descriptor["sources"],
        "ranks": plans,
        "checkpoints": checkpoints,
        "resolved_runtime": runtime,
        "transport": {
            **descriptor["transport"],
            "configuration_resolved": False,
        },
        "native_identities": {
            "nccl": runtime["transport"]["nccl_sha256"],
            "sircl": runtime["sircl"]["native_sha256"],
            "cache_placement": runtime["sparkcache"]["cuda_placement_sha256"],
            "cache_capture": runtime["sparkcache"]["cuda_snapshot_sha256"],
        },
        "distribution": {
            "mode": "download-once-then-fanout",
            "execute": False,
            "verify_each_rank": True,
        },
        "remaining_integration": [
            "lil image-owned code validation and startup wrapper",
            "per-rank SIRCL device/peer configuration and fabric validation",
            "translate reviewed SparkCache profile into kv-transfer-config",
            "artifact distribution over selected fabric",
            "live image and topology validation",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = render(read_json(args.descriptor), read_json(args.site))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Invalid integration plan: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
