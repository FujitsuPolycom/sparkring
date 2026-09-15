"""Run bounded native-image checks on an exclusively reserved two-node stack.

The caller reserves the stack and preserves its production rollback before this
command. This command owns only its named test containers and marked cache roots.
Passing a suite is not permission for unattended production promotion.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.images.upgrades.contracts import beneath, load_policy, read, require  # noqa: E402
from runtime.images.upgrades.hardware import Pair, isolated_spec, wait_ready  # noqa: E402
from runtime.images.upgrades.io import write_json  # noqa: E402
from runtime.images.upgrades.serving_checks import chat, needle_fixture, verify_needle  # noqa: E402
from runtime.images.upgrades import performance_gate  # noqa: E402
from runtime.images.upgrades import media_check  # noqa: E402


def option(args, name, value):
    result = list(args)
    require(result.count(name) == 1, "Serving option must occur once: " + name)
    result[result.index(name) + 1] = str(value)
    return result


def native_arguments(args, metadata):
    result = list(args)
    changes = []
    if "--kv-transfer-config" not in result:
        return result, changes
    config = json.loads(result[result.index("--kv-transfer-config") + 1])
    extra = config["kv_connector_extra_config"]
    for prefix, path in (
        (
            "spark_cache_cuda_placement_library",
            "/opt/sparkring/sparkcache/lib/libspark_cache_placement.so",
        ),
        (
            "spark_cache_async_page_capture_library",
            "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so",
        ),
    ):
        if prefix in extra:
            require(
                path in metadata["files"], "Native image omits a selected cache library"
            )
            extra[prefix] = path
            extra[prefix + "_sha256"] = metadata["files"][path]
            changes.append(prefix)
    contracts = [
        p for p in metadata.get("active_contracts", []) if "vllm-connector-jobs-" in p
    ]
    require(
        len(contracts) == 1,
        "Cache qualification needs one explicit active source-lease contract",
    )
    extra["spark_cache_async_page_capture_lease_contract"] = contracts[0]
    extra["spark_cache_async_page_capture_vllm_root"] = (
        "/opt/venv/lib/python3.12/site-packages"
    )
    changes.append("source-lease-contract")
    if config["kv_connector"] == "SparkBoundaryCacheConnector":
        boundary = metadata.get("boundary_runtime")
        require(
            boundary and boundary["path"] in metadata["files"],
            "Native image has no boundary-runtime attestation",
        )
        require(
            metadata["files"][boundary["path"]] == boundary["sha256"],
            "Boundary identity differs from installed inventory",
        )
        extra["spark_cache_boundary_contract"] = boundary["path"]
        extra["spark_cache_boundary_contract_sha256"] = boundary["sha256"]
        changes.append("boundary-runtime-identity")
    return option(
        result, "--kv-transfer-config", json.dumps(config, separators=(",", ":"))
    ), changes


def load_site(path):
    path = Path(path).resolve()
    site = read(path)
    require(
        site.get("schema") == "sparkring-tp2-qualification/v1",
        "Unknown qualification site schema",
    )
    require(
        re.fullmatch(r"[a-z][a-z0-9-]{0,30}", site["name"]), "Invalid model trial name"
    )
    require(
        len(site["hosts"]) == len(site["snapshots"]) == len(site["hostnames"]) == 2,
        "Qualification needs two ordered ranks",
    )
    root = PurePosixPath(site["cache_parent"])
    require(
        root.is_absolute()
        and ".." not in root.parts
        and root.name.startswith("sparkring-upgrade-"),
        "Use a dedicated qualification cache parent",
    )
    for field in ("port", "master_port"):
        require(
            type(site[field]) is int and 1024 <= site[field] <= 65535,
            "Invalid test port",
        )
    ipaddress.ip_address(site["api_host"])
    require(
        site["port"] != site["master_port"], "Serving and rendezvous ports must differ"
    )
    snapshots = [read(beneath(path.parent, name)) for name in site["snapshots"]]
    return site, snapshots


def prepare_specs(site, snapshots, metadata, image_id, run_id):
    specs = []
    locations = []
    changes = []
    for rank, snapshot in enumerate(snapshots):
        root = (
            PurePosixPath(site["cache_parent"])
            / run_id
            / (site["name"] + "-r" + str(rank))
        )
        mappings = {}
        for index, mount in enumerate(snapshot["Mounts"]):
            if mount["RW"]:
                require(
                    mount["Destination"].startswith("/cache"),
                    "Only cache mounts may be redirected by this adapter",
                )
                mappings[mount["Destination"]] = str(root / ("mount-" + str(index)))
        args = option(snapshot["Config"]["Cmd"], "--port", site["port"])
        args = option(args, "--master-port", site["master_port"])
        args, updated = native_arguments(args, metadata)
        spec = isolated_spec(
            snapshot,
            run_id=run_id,
            role="candidate",
            rank=rank,
            cache_mappings=mappings,
            image_id=image_id,
            native=True,
            command=args,
        )
        environment = dict(spec.environment)
        # Installation is wholly inside the image; a former source-overlay path
        # must not shadow its rebuilt packages.
        environment["PYTHONPATH"] = ""
        spec = replace(spec, environment=environment)
        persistent = None
        if "--kv-transfer-config" in args:
            extra = json.loads(args[args.index("--kv-transfer-config") + 1])[
                "kv_connector_extra_config"
            ]
            target = PurePosixPath(extra["spark_cache_root"])
            matches = [
                m for m in spec.mounts if target.is_relative_to(PurePosixPath(m.target))
            ]
            require(matches, "Persistent cache path is not inside an isolated bind")
            mount = max(matches, key=lambda m: len(m.target))
            physical = PurePosixPath(mount.source) / target.relative_to(
                PurePosixPath(mount.target)
            )
            persistent = str(physical.relative_to(root))
        specs.append(spec)
        locations.append(
            {
                "root": str(root),
                "mounts": list(mappings.values()),
                "persistent": persistent,
            }
        )
        changes.append(updated)
    return specs, locations, changes


def remote_cache_roots(pair, rank, location):
    script = """import pathlib,sys
root=pathlib.Path(sys.argv[1]); owner=sys.argv[2]
if root!=root.resolve() or any(p.is_symlink() for p in (root,*root.parents)): raise ValueError("Cache path traverses a symlink")
root.mkdir(parents=True,exist_ok=True)
marker=root/".sparkring-test-cache-owner"
if marker.exists():
 if marker.read_text()!=owner: raise ValueError("Cache root has another owner")
elif any(root.iterdir()): raise ValueError("Unmarked cache root is not empty")
else: marker.write_text(owner)
for item in sys.argv[3:]:
 path=pathlib.Path(item)
 if not path.resolve().is_relative_to(root): raise ValueError("Cache mount escapes root")
 path.mkdir(parents=True,exist_ok=True)
"""
    pair.call(
        rank,
        ["python3", "-c", script, location["root"], pair.run_id, *location["mounts"]],
    )


def fault(pair, rank, location, action):
    argv = [
        "python3",
        "-",
        action,
        "--root",
        location["root"],
        "--owner",
        pair.run_id,
        "--persistent",
        location["persistent"],
    ]
    if action in ("corrupt", "repair"):
        argv.append("--workers-stopped")
    return json.loads(
        pair.call(
            rank,
            argv,
            seconds=300,
            input_bytes=Path(__file__).with_name("cache_fault.py").read_bytes(),
        )
    )


def wait_publication(pair, locations, *, seconds=120):
    deadline = time.monotonic() + seconds
    previous = None
    while time.monotonic() < deadline:
        rows = [
            fault(pair, rank, location, "inventory")
            for rank, location in enumerate(locations)
        ]
        if all(rows) and rows == previous:
            return rows
        previous = rows
        time.sleep(3)
    raise ValueError("Persistent chunks did not become stable on both ranks")


def restart(pair, specs, base, model, timeout):
    before = [
        pair.owned(rank, spec.name)["State"]["StartedAt"]
        for rank, spec in enumerate(specs)
    ]
    for rank, spec in enumerate(specs):
        pair.stop(rank, spec.name)
    for rank in (1, 0):
        print(pair.start(rank, specs[rank].name), flush=True)
    wait_ready(base, model, seconds=timeout)
    after = [
        pair.owned(rank, spec.name)["State"]["StartedAt"]
        for rank, spec in enumerate(specs)
    ]
    require(
        all(a != b for a, b in zip(before, after)),
        "Worker restart did not replace both process lifetimes",
    )
    return {"before": before, "after": after}


def run(
    site_path,
    policy_path,
    lease_path,
    image_id,
    output,
    *,
    run_id,
    input_sha256,
    gate_id,
    leave_running=False,
):
    policy = load_policy(policy_path)
    site, snapshots = load_site(site_path)
    pair = Pair(
        policy,
        lease_path,
        site["hosts"],
        run_id=run_id,
        gate_id=gate_id,
        hostnames=site["hostnames"],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    require(
        re.fullmatch(r"sha256:[0-9a-f]{64}", image_id),
        "Native test image must be immutable",
    )
    for rank in (0, 1):
        running = pair.call(rank, ["docker", "ps", "-q"]).decode().split()
        if running:
            active = json.loads(pair.call(rank, ["docker", "inspect", *running]))
            require(
                not any(c["HostConfig"].get("DeviceRequests") for c in active),
                "Another GPU container is active; do not interrupt it",
            )
        verify = json.loads(
            pair.call(
                rank,
                [
                    "docker",
                    "run",
                    "--rm",
                    "--runtime",
                    "runc",
                    "--network",
                    "none",
                    "--read-only",
                    "--env",
                    "NVIDIA_VISIBLE_DEVICES=void",
                    "--env",
                    "CUDA_VISIBLE_DEVICES=",
                    image_id,
                    "verify",
                ],
                seconds=300,
            )
        )
        require(
            verify.get("schema") == "sparkring-native-verification/v1"
            and verify.get("files_verified", 0) > 0,
            "Native image verification did not pass",
        )
    raw = pair.call(
        0,
        [
            "docker",
            "run",
            "--rm",
            "--runtime",
            "runc",
            "--network",
            "none",
            "--read-only",
            "--entrypoint",
            "/bin/cat",
            image_id,
            "/opt/sparkring/receipts/native-installed.json",
        ],
        seconds=120,
    )
    metadata = json.loads(raw)
    specs, locations, changes = prepare_specs(
        site, snapshots, metadata, image_id, run_id
    )
    write_json(
        output / "plan.json",
        {
            "specs": [s.document() for s in specs],
            "caches": locations,
            "native_adaptations": changes,
            "image_id": image_id,
        },
    )
    base = f"http://{site['api_host']}:{site['port']}"
    timeout = site.get("startup_seconds", 900)
    created = []
    success = False
    assertions = 0
    evidence = {}
    try:
        for rank in (0, 1):
            remote_cache_roots(pair, rank, locations[rank])
            created.append(rank)
            pair.create(rank, specs[rank])
        for rank in (1, 0):
            print(pair.start(rank, specs[rank].name), flush=True)
        wait_ready(base, site["model"], seconds=timeout)
        assertions += 1
        answer = chat(
            base,
            site["model"],
            [
                {
                    "role": "user",
                    "content": "What is 19 + 23? Reply with the integer only.",
                }
            ],
        )
        require(re.search(r"\b42\b", answer["content"]), "Text arithmetic smoke failed")
        assertions += 1
        evidence["text"] = answer
        if site.get("cache_checks"):
            require(
                all(item["persistent"] for item in locations),
                "Cache checks need a selected connector",
            )
            fixture = needle_fixture(run_id)
            cold = verify_needle(
                chat(base, site["model"], fixture["messages"]), fixture
            )
            assertions += 1
            publication = wait_publication(pair, locations)
            warm = verify_needle(
                chat(base, site["model"], fixture["messages"]),
                fixture,
                minimum_cached=4096,
            )
            assertions += 1
            restarted = restart(pair, specs, base, site["model"], timeout)
            restored = verify_needle(
                chat(base, site["model"], fixture["messages"]),
                fixture,
                minimum_cached=4096,
            )
            assertions += 1
            for rank in (0, 1):
                pair.stop(rank, specs[rank].name)
            faults = [
                fault(pair, rank, item, "corrupt")
                for rank, item in enumerate(locations)
            ]
            for rank in (1, 0):
                print(pair.start(rank, specs[rank].name), flush=True)
            wait_ready(base, site["model"], seconds=timeout)
            recomputed = verify_needle(
                chat(base, site["model"], fixture["messages"]), fixture
            )
            require(
                recomputed["usage"]["prompt_tokens_details"]["cached_tokens"] == 0,
                "Corrupted test store received cache credit",
            )
            assertions += 1
            evidence["cache"] = {
                "fixture_sha256": fixture["sha256"],
                "cold": cold,
                "warm": warm,
                "restart": restarted,
                "restored": restored,
                "publication": publication,
                "faults": faults,
                "recomputed": recomputed,
            }
            for rank in (0, 1):
                pair.stop(rank, specs[rank].name)
            evidence["cache"]["repair"] = [
                fault(pair, rank, item, "repair") for rank, item in enumerate(locations)
            ]
            if leave_running or site.get("performance") or site.get("media_checks"):
                for rank in (1, 0):
                    print(pair.start(rank, specs[rank].name), flush=True)
                wait_ready(base, site["model"], seconds=timeout)
        if site.get("media_checks"):
            evidence["media"] = media_check.run(pair, specs[0], base, site["model"])
            assertions += 4
        if site.get("performance"):
            comparison = performance_gate.run(
                site["performance"],
                host=site["api_host"],
                port=site["port"],
                model=site["model"],
                output=output / "performance",
                policy_root=Path(site_path).resolve().parent,
            )
            evidence["performance"] = comparison
            assertions += len(comparison["comparisons"])
            require(
                comparison["passed"],
                "Candidate failed the protected performance comparison",
            )
        success = True
    finally:
        if not (success and leave_running):
            for rank in created:
                try:
                    pair.stop(rank, specs[rank].name)
                except Exception as error:
                    evidence.setdefault("cleanup_errors", []).append(str(error))
        receipt = {
            "schema": "sparkring-upgrade-gate/v1",
            "gate": gate_id,
            "input_sha256": input_sha256,
            "subject_sha256": image_id,
            "variant": "image",
            "outcome": "passed"
            if success and not evidence.get("cleanup_errors")
            else "failed",
            "assertions": assertions,
            "skipped": 0,
            "evidence": evidence,
            "left_running": success and leave_running,
            "scope": "Bounded TP2 text and optional persistent-cache integrity checks; no performance or general media claim.",
        }
        if evidence.get("performance"):
            receipt["measurements"] = evidence["performance"]["candidate"][
                "measurements"
            ]
            receipt["baseline"] = {
                "schema": "sparkring-upgrade-gate/v1",
                "gate": gate_id,
                "input_sha256": input_sha256,
                "subject_sha256": site["performance"]["baseline_image_id"],
                "variant": "control",
                "outcome": "passed",
                "assertions": len(evidence["performance"]["comparisons"]),
                "skipped": 0,
                "measurements": evidence["performance"]["baseline"]["measurements"],
            }
            receipt["scope"] = (
                "Bounded TP2 text, selected cache checks and matched measured throughput; no general media claim."
            )
        write_json(output / "result.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("site", "policy", "lease", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("image-id", "run-id", "input-sha256", "gate-id"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--leave-running", action="store_true")
    args = parser.parse_args()
    require(
        args.execute,
        "Qualification mutates only reserved test resources; pass --execute with the exact lease",
    )
    print(
        json.dumps(
            run(
                args.site,
                args.policy,
                args.lease,
                args.image_id,
                args.output,
                run_id=args.run_id,
                input_sha256=args.input_sha256,
                gate_id=args.gate_id,
                leave_running=args.leave_running,
            )
        ),
        flush=True,
    )
