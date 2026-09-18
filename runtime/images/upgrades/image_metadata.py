"""Create a labels-only OCI child in an operator-owned loopback registry."""

import copy
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import re
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:19555"
REPOSITORY = "sparkring/native"
RECEIPT = "/opt/sparkring/receipts/native-installed.json"


def require(condition, message):
    """Keep admission checks active when Python optimization is enabled."""
    if not condition:
        raise ValueError(message)


def registry_url(path):
    url = urllib.parse.urljoin(BASE, path)
    parsed = urllib.parse.urlparse(url)
    require(
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and parsed.port == 19555
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
        and parsed.path.startswith(f"/v2/{REPOSITORY}/")
        and ".." not in urllib.parse.unquote(parsed.path).split("/"),
        "Registry operation escaped the owned loopback repository",
    )
    return url


class NoRedirects(urllib.request.HTTPRedirectHandler):
    """A registry response cannot redirect a write outside the owned endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Registry redirects are unsupported")


def sha(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def request(path, *, method="GET", data=None, content_type=None):
    url = registry_url(path)
    headers = {
        "Accept": "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"
    }
    if content_type:
        headers["Content-Type"] = content_type
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirects())
    with opener.open(
        urllib.request.Request(url, method=method, data=data, headers=headers),
        timeout=120,
    ) as response:
        return response.status, response.read(), dict(response.headers)


def docker(*args):
    return subprocess.check_output(
        ["docker", "--host", "unix:///var/run/docker.sock", *args]
    )


def metadata_inputs(receipt, manifest, catalog, source_index):
    require(
        manifest["schema"] == "sparkring-release-source-inputs/v1",
        "Unsupported source manifest",
    )
    release = manifest["release_candidate"]
    require(re.fullmatch(r"[a-zA-Z0-9_.-]+", release), "Invalid release identifier")
    require(
        source_index["sources"] == manifest and source_index["version"] == release,
        "Source manifest differs from installed source index",
    )
    components = {item["id"]: item for item in manifest["components"]}
    require(
        len(manifest["components"]) == 2 and set(components) == {"vllm", "b12x"},
        "Expected one vLLM and one B12X source",
    )
    for name, component in components.items():
        require(
            re.fullmatch(r"[0-9a-f]{40}", component["upstream_commit"]),
            "Invalid upstream commit",
        )
        require(
            component["accepted_snapshot_sha256"]
            == receipt["compiler"]["source_trees"][name],
            "Source snapshot differs from installed compiler receipt",
        )
    update = receipt["feature_update"]
    profile = source_index["transport_profile"]
    transport = update["transport_bundles"][profile]
    require(
        transport["manifest_sha256"] == source_index["transport_manifest_sha256"],
        "Source index transport differs from installed receipt",
    )
    require(
        catalog["transport_profiles"][profile]["manifest_sha256"]
        == transport["manifest_sha256"],
        "Feature catalog transport differs from installed receipt",
    )
    return {
        "release": release,
        "components": components,
        "features": sorted(catalog["features"]),
        "feature_update_sha256": update["descriptor_sha256"],
        "feature_catalog_sha256": update["catalog_sha256"],
        "transport_profile": profile,
        "transport_manifest_sha256": transport["manifest_sha256"],
    }


def labels_for_parent(old, parent, metadata):
    labels = copy.deepcopy(old)
    inherited = {
        key
        for key in old
        if key.startswith("com.nvidia.") and key != "com.nvidia.volumes.needed"
    }
    inherited.update(
        {
            "org.opencontainers.image.version",
            "org.sparkring.release",
            "org.sparkring.features",
            "org.sparkring.feature-descriptor-sha256",
            "org.sparkring.source-extension",
            "org.sparkring.vllm.tree",
        }
    )
    for key in sorted(inherited):
        if key in labels:
            labels["org.sparkring.inherited." + key] = labels.pop(key)
    labels.pop("org.opencontainers.image.revision", None)
    labels.pop("org.opencontainers.image.licenses", None)
    labels.update(
        {
            "org.opencontainers.image.title": "SparkRing shared ARM64 inference runtime",
            "org.opencontainers.image.version": metadata["release"],
            "org.opencontainers.image.source": "https://github.com/FujitsuPolycom/sparkring",
            "org.opencontainers.image.description": "GB10/SM121 candidate with vLLM and isolated SGLang. Feature selection and qualification are profile-specific; component licenses remain separate.",
            "org.sparkring.release": metadata["release"],
            "org.sparkring.status": "implemented; profile qualification pending",
            "org.sparkring.features": ",".join(metadata["features"]),
            "org.sparkring.component-license-index": "/opt/sparkring/licenses/components.md",
            "org.sparkring.source-index": "/opt/sparkring/releases/shared/"
            + metadata["release"]
            + ".json",
            "org.sparkring.installed-receipt": RECEIPT,
            "org.sparkring.metadata-only-parent": parent,
            "org.sparkring.feature-update-sha256": metadata["feature_update_sha256"],
            "org.sparkring.feature-catalog-sha256": metadata["feature_catalog_sha256"],
            "org.sparkring.transport-profile": metadata["transport_profile"],
            "org.sparkring.transport-manifest-sha256": metadata[
                "transport_manifest_sha256"
            ],
            "org.sparkring.qualification": "Candidate; consult exact-image profile evidence. No universal serving qualification.",
        }
    )
    for name, component in metadata["components"].items():
        labels["org.sparkring." + name + ".upstream-commit"] = component[
            "upstream_commit"
        ]
        labels["org.sparkring." + name + ".snapshot-sha256"] = component[
            "accepted_snapshot_sha256"
        ]
    return labels


def without_labels(config):
    value = copy.deepcopy(config)
    value["config"].pop("Labels", None)
    return value


def child_configuration(original, parent, metadata):
    """Retain JSON key order because the serialized configuration is its identity."""
    child = copy.deepcopy(original)
    child["config"]["Labels"] = labels_for_parent(
        original["config"].get("Labels") or {}, parent, metadata
    )
    require(
        without_labels(original) == without_labels(child),
        "Non-label OCI configuration changed",
    )
    return encoded(child)


def verify_equivalence(before, after, child_id):
    require(
        after["Id"] == child_id and after["RootFS"] == before["RootFS"],
        "Child image identity or filesystem layers differ",
    )
    require(
        after["Architecture"] == before["Architecture"] and after["Os"] == before["Os"],
        "Child platform differs",
    )
    left, right = copy.deepcopy(before["Config"]), copy.deepcopy(after["Config"])
    left.pop("Labels", None)
    right.pop("Labels", None)
    require(left == right, "Non-label Docker runtime configuration changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--installed-receipt", type=Path)
    args = parser.parse_args(argv)
    PARENT, OUTPUT = args.parent_image, args.output
    require(
        re.fullmatch(r"sha256:[0-9a-f]{64}", PARENT),
        "Parent must be an immutable Docker image ID",
    )
    OUTPUT.mkdir(exist_ok=False)
    before = json.loads(docker("image", "inspect", PARENT))[0]
    require(
        before["Id"] == PARENT
        and before["Os"] == "linux"
        and before["Architecture"] == "arm64",
        "Parent must be the selected Linux ARM64 image",
    )
    source_raw = args.source_manifest.read_bytes()
    source_manifest = json.loads(source_raw)
    release = source_manifest["release_candidate"]
    require(re.fullmatch(r"[a-zA-Z0-9_.-]+", release), "Invalid release identifier")
    # Never start this container: it exists only to extract immutable image files.
    container = (
        docker("create", "--network", "none", "--entrypoint", "/bin/false", PARENT)
        .decode()
        .strip()
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", container),
        "Docker did not return one owned container ID",
    )
    try:
        docker("cp", container + ":" + RECEIPT, str(OUTPUT / "native-installed.json"))
        receipt_raw = (OUTPUT / "native-installed.json").read_bytes()
        if args.installed_receipt:
            require(
                args.installed_receipt.read_bytes() == receipt_raw,
                "Supplied receipt differs from parent image",
            )
        receipt = json.loads(receipt_raw)
        extracted = {}

        def verified_file(path, name):
            destination = OUTPUT / name
            docker("cp", container + ":" + path, str(destination))
            data = destination.read_bytes()
            require(
                sha(data)[7:] == receipt["files"][path],
                "Installed metadata differs: " + path,
            )
            extracted[path] = sha(data)[7:]
            return data

        catalog_raw = verified_file(
            receipt["feature_update"]["catalog"], "capabilities.json"
        )
        require(
            sha(catalog_raw)[7:] == receipt["feature_update"]["catalog_sha256"],
            "Installed catalog hash differs",
        )
        index_raw = verified_file(
            "/opt/sparkring/releases/shared/" + release + ".json", "source-index.json"
        )
        verified_file("/opt/sparkring/licenses/components.md", "components.md")
        metadata = metadata_inputs(
            receipt, source_manifest, json.loads(catalog_raw), json.loads(index_raw)
        )
        transport = receipt["feature_update"]["transport_bundles"][
            metadata["transport_profile"]
        ]
        transport_raw = verified_file(transport["manifest"], "transport-manifest.json")
        require(
            sha(transport_raw)[7:] == metadata["transport_manifest_sha256"],
            "Installed transport manifest hash differs",
        )
    finally:
        docker("rm", container)
    _, manifest_raw, manifest_headers = request(
        f"/v2/{REPOSITORY}/manifests/{PARENT[7:]}"
    )
    manifest = json.loads(manifest_raw)
    require(
        manifest["config"]["digest"] == PARENT, "Registry manifest names another parent"
    )
    _, original_bytes, _ = request(f"/v2/{REPOSITORY}/blobs/{PARENT}")
    require(
        sha(original_bytes) == PARENT,
        "Registry configuration digest differs from parent",
    )
    original = json.loads(original_bytes)
    data = child_configuration(original, PARENT, metadata)
    child_id = sha(data)
    (OUTPUT / "parent-config.json").write_bytes(original_bytes)
    (OUTPUT / "child-config.json").write_bytes(data)
    status, _, headers = request(
        f"/v2/{REPOSITORY}/blobs/uploads/", method="POST", data=b""
    )
    require(status == 202, "Registry did not open a configuration upload")
    location = headers.get("Location") or headers.get("location")
    require(
        isinstance(location, str) and location, "Registry omitted its upload location"
    )
    join = "&" if "?" in location else "?"
    status, _, _ = request(
        location + join + urllib.parse.urlencode({"digest": child_id}),
        method="PUT",
        data=data,
        content_type="application/octet-stream",
    )
    require(status == 201, "Registry did not accept the child configuration")
    child_manifest = copy.deepcopy(manifest)
    child_manifest["config"]["digest"] = child_id
    child_manifest["config"]["size"] = len(data)
    manifest_bytes = encoded(child_manifest)
    status, _, child_headers = request(
        f"/v2/{REPOSITORY}/manifests/{child_id[7:]}",
        method="PUT",
        data=manifest_bytes,
        content_type=manifest["mediaType"],
    )
    require(status == 201, "Registry did not accept the child manifest")
    reference = f"127.0.0.1:19555/{REPOSITORY}:{child_id[7:]}"
    pull = docker("pull", reference)
    (OUTPUT / "pull.log").write_bytes(pull)
    after = json.loads(docker("image", "inspect", child_id))[0]
    verify_equivalence(before, after, child_id)
    proof = {
        "schema": "sparkring-image-label-equivalence/v1",
        "before": before,
        "after": after,
        "parent_image_id": PARENT,
        "image_id": child_id,
        "raw_config_except_labels_identical": True,
        "runtime_config_except_labels_identical": True,
        "rootfs_identical": True,
        "rootfs_layers": len(before["RootFS"]["Layers"]),
        "unchanged_raw_config_sha256": sha(encoded(without_labels(original))),
        "local_registry_manifest": child_headers.get("Docker-Content-Digest")
        or sha(manifest_bytes),
        "parent_local_registry_manifest": manifest_headers.get("Docker-Content-Digest"),
        "installed_payload_unchanged": True,
        "external_publication": False,
        "source_commit_label": None,
        "serving_qualification_added": False,
    }
    proof["metadata_inputs"] = {
        "installed_receipt_sha256": sha(receipt_raw)[7:],
        "source_manifest_sha256": sha(source_raw)[7:],
        "verified_installed_files": extracted,
        "derived": metadata,
    }
    (OUTPUT / "equivalence.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps(
            {
                "image_id": child_id,
                "proof": str(OUTPUT / "equivalence.json"),
                "proof_sha256": sha((OUTPUT / "equivalence.json").read_bytes()),
                "rootfs_layers_unchanged": proof["rootfs_layers"],
                "external_publication": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
