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


@pytest.mark.skipif(
    not os.environ.get("LIL_TEST_BINARY"),
    reason="set LIL_TEST_BINARY to the locally built companion CLI",
)
def test_export_consumed_by_lil_cli(tmp_path):
    descriptor = module.read_json(HERE / "glm53.json")
    site = module.read_json(HERE / "site.example.json")
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
            "managed_service.py gate" in " ".join(c["argv"]) for c in r["checks"]
        )
        cache = json.loads(argv[argv.index("--kv-transfer-config") + 1])[
            "kv_connector_extra_config"
        ]
        assert (
            cache["spark_cache_target_checkpoint_sha256"]
            == cache["spark_cache_draft_checkpoint_sha256"]
        )
