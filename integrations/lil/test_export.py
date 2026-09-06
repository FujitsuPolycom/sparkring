import importlib.util
import os
from pathlib import Path
import sys
import subprocess
import json

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("bundle_export", HERE / "export.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fabric():
    rows = []
    for i in range(4):
        row = {key: "3" for key in module.FABRIC_KEYS}
        row.update(
            {
                "HOST_IP": f"192.0.2.{i + 1}",
                "MASTER_ADDR": "192.0.2.1",
                "SOCKET_IFNAME": "eth0",
                "NCCL_IB_HCA": "roce0,roce1",
                "SPARK_TP4_PEER0": "192.0.2.11",
                "SPARK_TP4_PEER1": "192.0.2.12",
                "SPARK_TP4_DEVICE0": "roce0",
                "SPARK_TP4_DEVICE1": "roce1",
                "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0": "198.51.100.11",
                "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1": "198.51.100.12",
                "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0": "roce2",
                "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1": "roce3",
            }
        )
        rows.append(row)
    return {"ranks": rows}


@pytest.mark.skipif(
    os.name != "posix", reason="canonical launcher runs in Bash on Linux/WSL"
)
@pytest.mark.parametrize("cache", [True, False])
def test_canonical_export_without_docker(cache):
    d = module.read_json(HERE / "glm53.json")
    s = module.read_json(HERE / "site.example.json")
    if not cache:
        s["cache"] = {"enabled": False}
    result = module.export(d, s, fabric(), "example-model")
    assert result["schema"] == "lil-image-bundle/v1"
    for i, r in enumerate(result["ranks"]):
        a = r["argv"]
        assert a[:2] == ["docker", "run"]
        assert a[a.index("--node-rank") + 1] == str(i)
        assert ("--headless" in a) == (i > 0)
        assert ("--kv-transfer-config" in a) == cache
        assert "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE=dual" in a
        assert r["checks"]
        import json

        graph = json.loads(a[a.index("--compilation-config") + 1])
        assert graph["cudagraph_capture_sizes"] == list(range(8, 129, 8))


@pytest.mark.skipif(os.name != "posix", reason="canonical Bash launcher")
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "descriptor_file,site_file",
    [
        ("glm53.json", "site.example.json"),
        ("glm53-mtp3.json", "site-mtp3.example.json"),
    ],
)
def test_fabric_gid_indices_reach_container_environment(
    explicit, descriptor_file, site_file
):
    descriptor = module.read_json(HERE / descriptor_file)
    site = module.read_json(HERE / site_file)
    network = fabric()
    for index, row in enumerate(network["ranks"]):
        if explicit:
            row.update(
                {
                    "NCCL_IB_GID_INDEX": str(index + 1),
                    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0": str(index + 4),
                    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1": str(index + 8),
                }
            )
    bundle = module.export(descriptor, site, network, "gid-fixture")
    for row, rank in zip(network["ranks"], bundle["ranks"], strict=True):
        for key in module.OPTIONAL_FABRIC_KEYS:
            assert f"{key}={row.get(key, '3')}" in rank["argv"]


@pytest.mark.skipif(
    not os.environ.get("LIL_TEST_BINARY"),
    reason="set LIL_TEST_BINARY to the locally built companion CLI",
)
@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize(
    "descriptor_file,site_file",
    [
        ("glm53.json", "site.example.json"),
        ("glm53-mtp3.json", "site-mtp3.example.json"),
    ],
)
def test_export_consumed_by_lil_cli(tmp_path, cache, descriptor_file, site_file):
    descriptor = module.read_json(HERE / descriptor_file)
    site = module.read_json(HERE / site_file)
    if not cache:
        site["cache"] = {"enabled": False}
    bundle = module.export(descriptor, site, fabric(), "handoff-fixture")
    filename = tmp_path / "bundle.json"
    filename.write_text(json.dumps(bundle))
    for action in ("validate", "render"):
        result = subprocess.run(
            [os.environ["LIL_TEST_BINARY"], "image", action, str(filename)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        if action == "render":
            assert json.loads(result.stdout) == bundle


def test_companion_revision_matches_instructions_and_ci():
    revision = module.read_json(HERE / "glm53.json")["lil_revision"]
    assert module.read_json(HERE / "glm53-mtp3.json")["lil_revision"] == revision
    for path in (
        HERE / "README.md",
        HERE / "OWNERSHIP.md",
        module.ROOT / ".github/workflows/ci.yml",
    ):
        assert revision in path.read_text(), str(path)


@pytest.mark.skipif(
    os.name != "posix" or not os.environ.get("LIL_TEST_BINARY"),
    reason="requires Linux/WSL and the built companion CLI",
)
@pytest.mark.parametrize("action", ["status", "logs", "stop"])
@pytest.mark.parametrize(
    "mode,allow,expected_actions",
    [
        ("legacy", False, 0),
        ("legacy", True, 4),
        ("conflict", True, 3),
        ("unreachable", True, 3),
    ],
)
def test_exported_bundle_lifecycle_with_fake_ssh(
    tmp_path, action, mode, allow, expected_actions
):
    bundle = module.export(
        module.read_json(HERE / "glm53-mtp3.json"),
        module.read_json(HERE / "site-mtp3.example.json"),
        fabric(),
        "lifecycle-fixture",
    )
    filename = tmp_path / "bundle.json"
    filename.write_text(json.dumps(bundle))
    calls = tmp_path / "calls.jsonl"
    fake = tmp_path / "ssh"
    fake.write_text(
        "#!"
        + sys.executable
        + "\n"
        + r"""
import json, os, shlex, sys
from pathlib import Path
bundle = json.loads(Path(os.environ["FAKE_BUNDLE"]).read_text())
rank = next(r for r in bundle["ranks"] if r["host"] == sys.argv[-2])
args = shlex.split(sys.argv[-1])
mode = os.environ["FAKE_MODE"]
if mode == "unreachable" and rank["rank"] == 1:
    sys.exit(255)
if args[1] == "inspect" and args[-1] == rank["name"]:
    labels = {"lil.image_bundle": bundle["id"]}
    if mode == "conflict" and rank["rank"] == 1:
        labels["lil.image_bundle_sha256"] = "f" * 64
    print(json.dumps({"Id": str(rank["rank"] + 1) * 64, "Config": {"Labels": labels}}))
else:
    assert args[-1] == str(rank["rank"] + 1) * 64, args
    with open(os.environ["FAKE_CALLS"], "a") as stream:
        stream.write(json.dumps(args) + "\n")
    print("ok")
"""
    )
    fake.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        FAKE_BUNDLE=str(filename),
        FAKE_CALLS=str(calls),
        FAKE_MODE=mode,
    )
    argv = [os.environ["LIL_TEST_BINARY"], "image", action]
    if allow:
        argv.append("--allow-legacy-owner")
    result = subprocess.run(
        [*argv, str(filename)], env=env, capture_output=True, text=True, timeout=20
    )
    assert (result.returncode == 0) == (mode == "legacy" and allow), result.stderr
    observed = calls.read_text().splitlines() if calls.exists() else []
    assert len(observed) == expected_actions, result.stderr


@pytest.mark.skipif(os.name != "posix", reason="canonical Bash launcher")
def test_mtp3_uses_published_mesh_image_and_target_identity():
    descriptor = module.read_json(HERE / "glm53-mtp3.json")
    site = module.read_json(HERE / "site-mtp3.example.json")
    bundle = module.export(descriptor, site, fabric(), "mtp-fixture")
    for r in bundle["ranks"]:
        argv = r["argv"]
        assert any("sha256:23f00af8" in a for a in argv)
        assert not any("/dflash-draft" in a for a in argv)
        speculation = json.loads(argv[argv.index("--speculative-config") + 1])
        assert (
            speculation["method"] == "mtp"
            and speculation["num_speculative_tokens"] == 3
        )
        graph = json.loads(argv[argv.index("--compilation-config") + 1])
        assert graph["cudagraph_capture_sizes"] == list(range(4, 65, 4))
        assert any(
            "verify_installed_mesh" in " ".join(c["argv"])
            and "supervised trial only" in c["expected"]
            for c in r["checks"]
        )
        cache = json.loads(argv[argv.index("--kv-transfer-config") + 1])[
            "kv_connector_extra_config"
        ]
        assert (
            cache["spark_cache_target_checkpoint_sha256"]
            == cache["spark_cache_draft_checkpoint_sha256"]
        )
