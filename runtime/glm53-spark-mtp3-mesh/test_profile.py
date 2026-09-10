"""GPU-free contracts for native-MTP3 configuration and mesh source composition."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


make_example = _module("mtp3_mesh_example_test", "make_example.py")
mesh_profile = _module("mtp3_mesh_profile_test", "profile.py")


def _site(tmp_path):
    (tmp_path / "fabric.example.json").write_text(json.dumps(make_example.topology_example()))
    path = tmp_path / "site.json"
    path.write_text(json.dumps(make_example.site_example()))
    return path


def test_example_has_only_benchmark_or_documentation_addresses(tmp_path):
    site, topology, plan = mesh_profile.load_site(_site(tmp_path))
    assert site["management_addresses"] == [f"192.0.2.{10+i}" for i in range(4)]
    assert all(port.ipv4.startswith("198.18.") for rank in topology.ranks for port in rank.ports)
    assert len(plan.tc_rules) == len(plan.markers) == len(plan.routes) == 8
    assert {m.udp_source_port for m in plan.markers} == {65535}


@pytest.mark.parametrize("path", ["/", "relative", "/tmp/../etc", "/tmp/a:b", "/tmp/REPLACE", "/tmp/a\nexec"])
def test_path_guard(path):
    with pytest.raises(ValueError):
        mesh_profile.absolute(path, "fixture")


def test_source_composition_renders_dcp1_dual_domains_and_disables_cache(
    tmp_path, monkeypatch, manifest_bundle
):
    from scripts.test_deploy_selection import local_receipt

    lock = json.loads(mesh_profile.SOURCE_LOCK.read_text())
    lock["runtime"]["bundle_manifest_sha256"] = mesh_profile.sha(
        manifest_bundle / "sparkring-overlay-manifest.json")
    lock_path = tmp_path / "source-lock.json"
    lock_path.write_text(json.dumps(lock))
    document = local_receipt()
    monkeypatch.setattr(mesh_profile, "SOURCE_LOCK", lock_path)
    document["source_lock_sha256"] = document["inside_image"]["source_lock_sha256"] = mesh_profile.sha(lock_path)
    document["inside_image"]["bundle_manifest_sha256"] = lock["runtime"]["bundle_manifest_sha256"]
    receipt = tmp_path / "image.json"
    receipt.write_text(json.dumps(document))
    site = _site(tmp_path)
    data = json.loads(site.read_text())
    data["runtime_profile"] = document["profile"]
    site.write_text(json.dumps(data))
    output = tmp_path / "rendered"
    result = mesh_profile.render(site, manifest_bundle, output, receipt)
    for rank in range(4):
        env = mesh_profile.defaults(output / f"rank{rank}.env")
        assert env["DECODE_CONTEXT_PARALLEL_SIZE"] == "1"
        assert env["SPARKCACHE_ENABLED"] == env["SPARKCACHE_ASYNC_PAGE_CAPTURE"] == "0"
        assert env["VLLM_B12X_KDA_PREFILL_COALESCING"] == env["VLLM_GLM53_MHC_PREFILL_SHARD"] == "1"
        assert env["NCCL_IB_HCA"].startswith("=") and len(env["NCCL_IB_HCA"].split(",")) == 4
        assert env["NCCL_LIBRARY_PATH"] == lock["runtime"]["nccl_path"]
        assert env["NCCL_LIBRARY_SHA256"] == lock["runtime"]["nccl_sha256"]
    assert result["image_receipt_sha256"] == mesh_profile.sha(receipt)


@pytest.mark.parametrize("field,value", [("container_prefix", "a;rm"), ("marker_binary_sha256", "x"),
                                        ("management_addresses", ["192.0.2.1"]*4), ("model_roots", [])])
def test_site_rejects_ambiguous_identity(tmp_path, field, value):
    path = _site(tmp_path)
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        mesh_profile.load_site(path)


@pytest.mark.parametrize("mutation", ["missing", "extra", "schema"])
def test_site_schema_is_exact(tmp_path, mutation):
    path = _site(tmp_path)
    data = json.loads(path.read_text())
    if mutation == "missing":
        del data["model_roots"]
    elif mutation == "extra":
        data["draft_model_roots"] = ["/srv/draft"] * 4
    else:
        data["schema"] = "unrecognized/v1"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Site fields"):
        mesh_profile.load_site(path)


@pytest.mark.parametrize("mutation", ["renamed", "direction", "function"])
def test_topology_rejects_hca_placement_incompatible_with_rocenante(tmp_path, mutation):
    path = _site(tmp_path)
    topology_path = tmp_path / "fabric.example.json"
    topology = json.loads(topology_path.read_text())
    ports = topology["ranks"][0]["ports"]
    if mutation == "renamed":
        ports["clockwise"][0]["rdma_device"] = "mlx5_0"
    elif mutation == "direction":
        for function in (0, 1):
            clockwise = ports["clockwise"][function]
            counter = ports["counter_clockwise"][function]
            clockwise["rdma_device"], counter["rdma_device"] = counter["rdma_device"], clockwise["rdma_device"]
    else:
        first, second = ports["clockwise"]
        first["rdma_device"], second["rdma_device"] = second["rdma_device"], first["rdma_device"]
    topology_path.write_text(json.dumps(topology))
    with pytest.raises(ValueError, match="HCA order and peer map"):
        mesh_profile.load_site(path)


@pytest.mark.parametrize("field,value", [("bounded_runtime_seconds", 3600),
                                        ("shared_diagonal_flow_label", False)])
def test_profile_requires_declared_marker_lifetime_and_shared_source_port(tmp_path, field, value):
    path = _site(tmp_path)
    topology_path = tmp_path / "fabric.example.json"
    topology = json.loads(topology_path.read_text())
    topology[field] = value
    topology_path.write_text(json.dumps(topology))
    with pytest.raises(ValueError):
        mesh_profile.load_site(path)


def test_only_mtp3_profile_is_defined():
    pins = mesh_profile.PINS
    assert pins["speculation"]["method"] == "mtp"
    assert pins["speculation"]["num_speculative_tokens"] == 3
    assert pins["speculation"]["attention_backend"] == "B12X"
    assert pins["capture_sizes"] == list(range(4, 65, 4))
    assert pins["captured_sircl_query_rows"] == [16, 20, 24, 28, 32]


def test_vendor_is_complete_and_content_bound():
    vendor = mesh_profile.ROOT / "third_party/b12x_roce"
    provenance = json.loads((vendor / "provenance.json").read_text())
    actual, files = mesh_profile.build_bundle._canonical_tree(vendor / "b12x/comm/roce")
    assert actual == provenance["roce_tree_sha256"]
    assert len(files) == 9
    assert (vendor / "LICENSE").is_file()


def test_marker_source_matches_its_declared_pin():
    source = HERE / mesh_profile.PINS["marker"]["source"]
    assert mesh_profile.sha(source) == mesh_profile.PINS["marker"]["source_sha256"]


def test_native_mtp_cache_identity_is_not_external_dflash():
    source = (mesh_profile.BASE / "launch-rank.sh").read_text()
    assert 'DRAFT_CHECKPOINT_FINGERPRINT="${TARGET_CHECKPOINT_FINGERPRINT}"' in source
    assert '"spark_cache_draft_checkpoint_sha256": os.environ["DRAFT_CHECKPOINT_FINGERPRINT"]' in source
    assert mesh_profile.PINS["cache_identity"]["draft_checkpoint_source"] == "target.checkpoint_identity"


def test_speculation_payloads_use_real_launcher_code(monkeypatch):
    source = (mesh_profile.BASE / "launch-rank.sh").read_text()
    begin = source.index('speculative_config="$(python3 - <<\'PY\'')
    python_source = source[begin:].split("\n", 1)[1].split("\nPY\n", 1)[0]
    common = {"NUM_SPECULATIVE_TOKENS":"3", "DRAFT_TENSOR_PARALLEL_SIZE":"4", "DRAFT_KV_CACHE_DTYPE":"auto",
              "DRAFT_SAMPLE_METHOD":"probabilistic", "REJECTION_SAMPLE_METHOD":"standard"}
    for key, value in common.items():
        monkeypatch.setenv(key, value)
    for method in ("mtp", "dflash"):
        monkeypatch.setenv("SPECULATION_METHOD", method)
        namespace = {}
        exec(compile(python_source, "launcher-speculation", "exec"), namespace)
        config = namespace["config"]
        assert config["method"] == method
        if method == "mtp":
            assert "model" not in config
            assert config["attention_backend"] == "B12X"
        else:
            assert config["model"] == "/dflash-draft"
            assert "attention_backend" not in config


@pytest.fixture
def manifest_bundle(tmp_path, monkeypatch):
    """Use synthetic bytes to exercise renderer integrity checks without CUDA artifacts."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    records = []
    for name in sorted(mesh_profile.build_bundle.REQUIRED_SIRCL_FILES):
        path = bundle / name
        path.write_text("CPU-only renderer fixture\n", encoding="utf-8")
        records.append({"path": name, "sha256": mesh_profile.sha(path)})
    manifest = bundle / "sparkring-overlay-manifest.json"
    manifest.write_text(json.dumps({"files": records}), encoding="utf-8")
    monkeypatch.setitem(mesh_profile.PINS, "canonical_bundle_manifest_sha256", mesh_profile.sha(manifest))
    return bundle


def test_render_preserves_mtp_kernel_scheduler_cache_contract(tmp_path, manifest_bundle):
    site = _site(tmp_path)
    output = tmp_path / "rendered"
    receipt = mesh_profile.render(site, manifest_bundle, output)
    assert receipt["execution_authorized"] is False
    assert receipt["status"] == "research-only"
    assert len(receipt["ranks"]) == 4
    expected = {
        "TARGET_MODEL_VARIANT": "nvfp4-spark", "SPECULATION_METHOD": "mtp",
        "NUM_SPECULATIVE_TOKENS": "3", "DRAFT_TENSOR_PARALLEL_SIZE": "4",
        "MAX_CUDAGRAPH_CAPTURE_SIZE": "64", "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
        "TENSOR_PARALLEL_SIZE": "4", "DECODE_CONTEXT_PARALLEL_SIZE": "4",
        "PIPELINE_PARALLEL_SIZE": "1", "MAX_NUM_BATCHED_TOKENS": "8192",
        "MAX_NUM_SEQS": "16", "MAX_MODEL_LEN": "1048576", "PREFILL_SCHEDULE_INTERVAL": "2",
        "ATTENTION_BACKEND": "B12X", "MOE_BACKEND": "b12x", "LINEAR_BACKEND": "b12x",
        "KDA_PREFILL_BACKEND": "b12x", "LOAD_FORMAT": "fastsafetensors",
        "KV_CACHE_DTYPE": "fp8", "KV_CACHE_MEMORY_BYTES": "25769803776",
        "SIRCL_ENABLED": "1", "SPARK_TP4_GRAPH_DIRECT_DOORBELL": "1",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL": "1",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE": "fused",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE": "dual",
        "SPARKCACHE_ASYNC_PAGE_CAPTURE": "1", "SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES": "3221225472",
        "SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT": "2", "SPARKCACHE_PUBLICATION_SCHEMA": "tail-cow-v2",
        "SPARKCACHE_CACHE_NAMESPACE": mesh_profile.PINS["cache_identity"]["namespace"],
        "SPARK_TP4_GRAPH_SUBMIT_CPU": "10", "SPARK_TP4_GRAPH_PROGRESS_CPU": "11",
        "SPARK_TP4_MAX_INFLIGHT": "64", "SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS": "10",
        "SPARKRING_WARMUP_TEMPERATURE": "0",
    }
    for rank in range(4):
        values = mesh_profile.defaults(output / f"rank{rank}.env")
        assert {key: values[key] for key in expected} == expected
        assert values["HOST_IP"] == f"192.0.2.{rank+10}"
        assert values["MASTER_ADDR"] == "192.0.2.10"
        assert "DFLASH_MODEL_HOST_PATH" not in values
        assert not any("REPLACE" in value for value in values.values())
    assert (output / "launch-rank.sh").read_bytes() == (mesh_profile.BASE / "launch-rank.sh").read_bytes()
    for name, digest in receipt["files"].items():
        assert mesh_profile.sha(output / name) == digest
    assert json.loads((output / "fabric-plan.json").read_text()) == receipt
    copied_site = json.loads((output / "site.json").read_text())
    assert copied_site["topology_file"] == "fabric.json"
    mesh_profile.load_site(output / "site.json")


def test_render_peer_devices_follow_topology(tmp_path, manifest_bundle):
    site_path = _site(tmp_path)
    topology_path = tmp_path / "fabric.example.json"
    source = json.loads(topology_path.read_text())
    for rank in source["ranks"]:
        rank["management_netdev"] = f"mgmt{rank['rank']}"
    topology_path.write_text(json.dumps(source))
    site, topology, plan = mesh_profile.load_site(site_path)
    output = tmp_path / "rendered"
    receipt = mesh_profile.render(site_path, manifest_bundle, output)
    for rank in range(4):
        env = mesh_profile.defaults(output / f"rank{rank}.env")
        assert env["SOCKET_IFNAME"] == f"mgmt{rank}"
        directions = ("clockwise", "counter_clockwise") if rank % 2 == 0 else ("counter_clockwise", "clockwise")
        for slot, direction in enumerate(directions):
            for function in (0, 1):
                local = topology.rank(rank).port(direction, function)
                peer = topology.rank(local.peer_rank).port(local.peer_direction, local.peer_function)
                assert local.peer_rank == rank ^ (1 if slot == 0 else 3)
                prefix = "SPARK_TP4_" if function == 0 else "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_"
                assert env[prefix + f"DEVICE{slot}"] == local.rdma_device
                assert env[prefix + f"PEER{slot}"] == peer.ipv4
                assert env[prefix + f"GID{slot}"] == "3"
        expected_hcas = [topology.rank(rank).port(direction, 0).rdma_device
                         for direction in ("clockwise", "counter_clockwise")]
        assert set(env["NCCL_IB_HCA"].split(",")) == set(expected_hcas)
        local_plan = receipt["ranks"][rank]
        assert local_plan["routes"] == [mesh_profile.fabric.route_command(route, add=True)
                                        for route in plan.routes if route.source_rank == rank]
        assert len(local_plan["markers"]) == len(local_plan["tc_rules"]) == 2
        assert {marker["device"] for marker in local_plan["markers"]} == {
            marker.rdma_device for marker in plan.markers if marker.source_rank == rank}
        for marker in local_plan["markers"]:
            assert marker["argv"] == [site["marker_binary"], "--device", marker["device"],
                                      "--source-port", "65535", "--replacement-ethertype", "0x88b5",
                                      "--attach", "--run-seconds", "7200"]


def test_render_refuses_existing_output_without_changing_it(tmp_path, manifest_bundle):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "operator-file"
    sentinel.write_text("preserve this content")
    with pytest.raises(ValueError, match="exists"):
        mesh_profile.render(_site(tmp_path), manifest_bundle, output)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "preserve this content"


@pytest.mark.parametrize("mutation", ["manifest", "payload", "missing"])
def test_render_refuses_changed_bundle_before_output(tmp_path, manifest_bundle, mutation):
    if mutation == "manifest":
        path = manifest_bundle / "sparkring-overlay-manifest.json"
        path.write_text(path.read_text() + "\n")
    elif mutation == "payload":
        (manifest_bundle / "sitecustomize.py").write_text("altered executable fixture")
    else:
        (manifest_bundle / "sitecustomize.py").unlink()
    output = tmp_path / "rendered"
    with pytest.raises((ValueError, FileNotFoundError)):
        mesh_profile.render(_site(tmp_path), manifest_bundle, output)
    assert not output.exists()


def test_sampling_warmup_requires_source_bound_image_receipt(tmp_path, manifest_bundle):
    document = _image_receipt_document()
    document["inside_image"]["readiness_warmup"] = {
        "environment": "SPARKRING_WARMUP_TEMPERATURE", "temperature": 1.0,
        "helper_sha256": mesh_profile.sha(mesh_profile.BASE / "warmup_dflash.py"),
    }
    path = tmp_path / "sampling-image.json"
    path.write_text(json.dumps(document))
    output = tmp_path / "sampling-render"
    mesh_profile.render(_site(tmp_path), manifest_bundle, output, path)
    for rank in range(4):
        assert mesh_profile.defaults(output / f"rank{rank}.env")["SPARKRING_WARMUP_TEMPERATURE"] == "1"


@pytest.mark.parametrize("field,value", [("temperature", 0), ("helper_sha256", "0" * 64), ("environment", "OTHER")])
def test_sampling_warmup_receipt_rejects_mismatch(tmp_path, field, value):
    document = _image_receipt_document()
    warmup = {"environment": "SPARKRING_WARMUP_TEMPERATURE", "temperature": 1.0,
              "helper_sha256": mesh_profile.sha(mesh_profile.BASE / "warmup_dflash.py")}
    warmup[field] = value
    document["inside_image"]["readiness_warmup"] = warmup
    path = tmp_path / "sampling-image.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="sampling warmup helper"):
        mesh_profile.load_image_receipt(path)


@pytest.mark.parametrize("unsafe", ["../outside", "/outside", "nested/../../outside", "C:/outside", "..\\outside"])
def test_render_rejects_unsafe_manifest_paths(tmp_path, manifest_bundle, monkeypatch, unsafe):
    manifest = manifest_bundle / "sparkring-overlay-manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": unsafe, "sha256": "a" * 64}]}))
    monkeypatch.setitem(mesh_profile.PINS, "canonical_bundle_manifest_sha256", mesh_profile.sha(manifest))
    with pytest.raises(ValueError):
        mesh_profile.render(_site(tmp_path), manifest_bundle, tmp_path / "rendered")


def test_render_canonical_bundle_when_locally_available(tmp_path):
    bundle = mesh_profile.ROOT.parent / "rebuilt-bundle"
    if not bundle.is_dir():
        pytest.skip("Optional canonical native bundle is absent; portable synthetic-byte coverage still runs")
    receipt = mesh_profile.render(_site(tmp_path), bundle, tmp_path / "rendered")
    assert receipt["bundle_manifest_sha256"] == mesh_profile.PINS["canonical_bundle_manifest_sha256"]


def _image_receipt_document():
    bundle_sha = mesh_profile.PINS["canonical_bundle_manifest_sha256"]
    source_sha = "b" * 64
    return {
        "schema": "sparkring-mtp3-mesh-image-receipt/v1", "checks_passed": True,
        "platform": "linux/arm64", "parent_image_id": mesh_profile.IMAGE["operator_image"]["image_id"],
        "image_id": "sha256:" + "a" * 64, "image_reference": "sha256:" + "a" * 64,
        "bundle_manifest_sha256": bundle_sha, "source_receipt_sha256": source_sha,
        "inside_image": {"checks_passed": True, "bundle_manifest_sha256": bundle_sha,
                         "source_receipt_sha256": source_sha, "cuda_initialized": False, "model_loaded": False,
                         "compute": json.loads((mesh_profile.HERE / "image-receipt.json").read_text())["inside_image"]["compute"]},
    }


def _r33_image_receipt_document(bundle_sha):
    verifier = mesh_profile._r33_profile_verifier()
    contract = verifier.load_contract()
    nccl_sha = "84a4b8d83fb5fa1f0d640d311ad38b45140672dae9889775fe1e4a3990479e47"
    return {
        "schema": "sparkring-r33-image-receipt/v1", "checks_passed": True,
        "platform": "linux/arm64", "image_id": "sha256:" + "d" * 64,
        "image_reference": "sha256:" + "d" * 64,
        "artifact_lock_sha256": contract["image"]["artifact_lock_sha256"],
        "source_lock_sha256": "e" * 64, "sources": contract["image"]["required_sources"],
        "component_receipts": {name: "f" * 64 for name in contract["image"]["required_receipts"]},
        "nccl_version": "2.31.2", "source_locks_match": True,
        "source_lock_receipts_match": True, "installed_payload_bytes_match": True,
        "package_checks_passed": True, "bundle_manifest_sha256": bundle_sha,
        "verification": {"checked_files": {
            "/opt/sparkring/sircl/python/sparkring-overlay-manifest.json": bundle_sha,
            "/opt/local-inference/nccl/lib/libnccl.so.2.31.2": nccl_sha,
            "/opt/sparkring/sircl/libspark_transport_capi.so": "bea00f2ba6051c2c0bcd2853aae894672aa7f1fe5a1d905edaa9120aabf74246",
            "/opt/sparkring/sparkcache/lib/libspark_cache_placement.so": "d89c9fdae8dc99ae3f7a151cc3dd9e92fdc8fd0b994069fc263027fd4d056c93",
            "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so": "7da9e72f096ae679906ba71336c16e7894a247eb5b0d217aaccd115b85058953",
        }},
    }


def test_r33_receipt_accepts_receipt_bound_mesh_manifest():
    bundle_sha = "c" * 64
    document = _r33_image_receipt_document(bundle_sha)
    result = mesh_profile.validate_image_receipt(document)
    assert result["bundle_manifest_sha256"] == bundle_sha
    assert result["r33_candidate"] is True


@pytest.mark.parametrize("failure", ["missing", "mismatch", "malformed", "invalid-verification"])
def test_r33_receipt_rejects_unverified_mesh_manifest(failure):
    bundle_sha = "c" * 64
    document = _r33_image_receipt_document(bundle_sha)
    manifest_path = "/opt/sparkring/sircl/python/sparkring-overlay-manifest.json"
    if failure == "missing":
        del document["verification"]["checked_files"][manifest_path]
    elif failure == "mismatch":
        document["verification"]["checked_files"][manifest_path] = "d" * 64
    elif failure == "malformed":
        document["bundle_manifest_sha256"] = "not-a-sha256"
    else:
        document["verification"] = None
    with pytest.raises(ValueError, match="does not bind its mesh manifest"):
        mesh_profile.validate_image_receipt(document)


def test_r33_receipt_renders_canonical_managed_tp4_environment(tmp_path, manifest_bundle, monkeypatch):
    manifest_sha = mesh_profile.sha(manifest_bundle / "sparkring-overlay-manifest.json")
    monkeypatch.setitem(mesh_profile.PINS, "canonical_bundle_manifest_sha256", manifest_sha)
    document = _r33_image_receipt_document(manifest_sha)
    receipt = tmp_path / "r33-image.json"
    receipt.write_text(json.dumps(document))
    site = _site(tmp_path)
    data = json.loads(site.read_text())
    data["runtime_profile"] = "tp4-dcp1"
    site.write_text(json.dumps(data))
    output = tmp_path / "r33-rendered"
    mesh_profile.render(site, manifest_bundle, output, receipt)
    for rank in range(4):
        env = mesh_profile.defaults(output / f"rank{rank}.env")
        assert env["SOURCE_IMAGE_PROFILE"] == "tp4-dcp1"
        assert env["SPARKCACHE_ENABLED"] == env["SPARKCACHE_ASYNC_PAGE_CAPTURE"] == "0"
        assert env["SPARKRING_PROFILE_MODE"] == "custom"
        assert env["SPARKRING_MANAGED_MESH_RENDERED"] == "1"
        assert env["VLLM_SPARK_TP4_MODE"] == env["VLLM_SPARK_TP4_VOCAB_MODE"] == "custom"
        assert env["SPARK_TP4_LIBRARY"] == "/opt/sparkring/sircl/libspark_transport_capi.so"
        assert env["VLLM_NCCL_SO_PATH"] == env["NCCL_LOCAL_INFERENCE_PATH"] == env["LD_PRELOAD"] == "/opt/local-inference/nccl/lib/libnccl.so.2"
        assert env["NODE_RANK"] == str(rank)
        assert env["SPARK_TP4_CONTROL_PORT0"] == env["SPARK_TP4_GRAPH_CONTROL_PORT0"]


def test_r33_sparkcache_renders_receipt_bound_native_libraries(tmp_path, manifest_bundle, monkeypatch):
    manifest_sha = mesh_profile.sha(manifest_bundle / "sparkring-overlay-manifest.json")
    monkeypatch.setitem(mesh_profile.PINS, "canonical_bundle_manifest_sha256", manifest_sha)
    receipt = tmp_path / "r33-image.json"
    receipt.write_text(json.dumps(_r33_image_receipt_document(manifest_sha)))
    site = _site(tmp_path)
    data = json.loads(site.read_text())
    data["runtime_profile"] = "tp4-dcp1-sparkcache"
    site.write_text(json.dumps(data))
    output = tmp_path / "r33-cache"
    mesh_profile.render(site, manifest_bundle, output, receipt)
    for rank in range(4):
        env = mesh_profile.defaults(output / f"rank{rank}.env")
        assert env["SOURCE_IMAGE_PROFILE"] == "tp4-dcp1-sparkcache"
        assert env["SPARKCACHE_ENABLED"] == env["SPARKCACHE_ASYNC_PAGE_CAPTURE"] == "1"
        assert env["SPARKCACHE_PLACEMENT_LIBRARY_PATH"] == "/opt/sparkring/sparkcache/lib/libspark_cache_placement.so"
        assert env["SPARKCACHE_PLACEMENT_LIBRARY_SHA256"] == "d89c9fdae8dc99ae3f7a151cc3dd9e92fdc8fd0b994069fc263027fd4d056c93"
        assert env["SPARKCACHE_SNAPSHOT_LIBRARY_PATH"] == "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so"
        assert env["SPARKCACHE_SNAPSHOT_LIBRARY_SHA256"] == "7da9e72f096ae679906ba71336c16e7894a247eb5b0d217aaccd115b85058953"
        assert env["SPARKCACHE_SOURCE_LEASE_CONTRACT"].startswith("/opt/venv/")


def test_r33_sparkcache_rejects_unbound_native_library(tmp_path, manifest_bundle, monkeypatch):
    manifest_sha = mesh_profile.sha(manifest_bundle / "sparkring-overlay-manifest.json")
    monkeypatch.setitem(mesh_profile.PINS, "canonical_bundle_manifest_sha256", manifest_sha)
    document = _r33_image_receipt_document(manifest_sha)
    document["verification"]["checked_files"]["/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so"] = "0" * 64
    receipt = tmp_path / "r33-image.json"
    receipt.write_text(json.dumps(document))
    site = _site(tmp_path)
    data = json.loads(site.read_text())
    data["runtime_profile"] = "tp4-dcp1-sparkcache"
    site.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not bind"):
        mesh_profile.render(site, manifest_bundle, tmp_path / "rejected", receipt)


@pytest.mark.parametrize('field', ['compute', 'source_lock_sha256', 'b12x_revision', 'environment', 'proposal_head_nvfp4', 'target_head_quantization'])
def test_receipt_requires_profile_compute(tmp_path, field):
    document = _image_receipt_document()
    if field == 'compute':
        del document['inside_image']['compute']
    else:
        document['inside_image']['compute'][field] = None
    path = tmp_path / 'wrong-compute.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        mesh_profile.load_image_receipt(path)


def test_verified_image_receipt_changes_only_image_selection(tmp_path, manifest_bundle):
    site_path = _site(tmp_path)
    document = _image_receipt_document()
    receipt_path = tmp_path / "image-receipt.json"
    receipt_path.write_text(json.dumps(document))
    parent_output, child_output = tmp_path / "parent", tmp_path / "child"
    parent = mesh_profile.render(site_path, manifest_bundle, parent_output)
    child = mesh_profile.render(site_path, manifest_bundle, child_output, receipt_path)
    assert parent["image"] == mesh_profile.IMAGE["operator_image"]
    assert "image_receipt_sha256" not in parent
    assert child["image"] == document
    assert child["image_receipt_sha256"] == mesh_profile.sha(receipt_path)
    defaults = mesh_profile.defaults(mesh_profile.BASE / "runtime.env.example")
    for rank in range(4):
        parent_env = mesh_profile.defaults(parent_output / f"rank{rank}.env")
        child_env = mesh_profile.defaults(child_output / f"rank{rank}.env")
        assert parent_env["IMAGE_ID"] == defaults["IMAGE_ID"]
        assert parent_env["IMAGE_REF"] == defaults["IMAGE_REF"]
        assert child_env["IMAGE_ID"] == child_env["IMAGE_REF"] == document["image_id"]
        assert {key: value for key, value in child_env.items() if key not in {"IMAGE_ID", "IMAGE_REF"}} == {
            key: value for key, value in parent_env.items() if key not in {"IMAGE_ID", "IMAGE_REF"}}


@pytest.mark.parametrize("scope,field,value", [
    ("outer", "schema", "unrecognized/v1"), ("outer", "checks_passed", False),
    ("outer", "checks_passed", 1), ("outer", "platform", "linux/amd64"),
    ("outer", "parent_image_id", "sha256:" + "c" * 64),
    ("outer", "image_id", "mutable:tag"), ("outer", "image_reference", "mutable:tag"),
    ("outer", "bundle_manifest_sha256", "c" * 64), ("outer", "source_receipt_sha256", "short"),
    ("outer", "inside_image", None), ("inside", "checks_passed", False),
    ("inside", "checks_passed", 1), ("inside", "bundle_manifest_sha256", "c" * 64),
    ("inside", "source_receipt_sha256", "c" * 64), ("inside", "cuda_initialized", True),
    ("inside", "cuda_initialized", 0), ("inside", "model_loaded", True), ("inside", "model_loaded", 0),
])
def test_image_receipt_rejects_unverified_or_incompatible_image(tmp_path, scope, field, value):
    document = _image_receipt_document()
    target = document if scope == "outer" else document["inside_image"]
    target[field] = value
    path = tmp_path / "image-receipt.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Image receipt"):
        mesh_profile.load_image_receipt(path)


def test_invalid_image_receipt_creates_no_rendered_output(tmp_path, manifest_bundle):
    document = _image_receipt_document()
    document["checks_passed"] = False
    path = tmp_path / "image-receipt.json"
    path.write_text(json.dumps(document))
    output = tmp_path / "rendered"
    with pytest.raises(ValueError, match="Image receipt"):
        mesh_profile.render(_site(tmp_path), manifest_bundle, output, path)
    assert not output.exists()


def _site_with_api_keys(tmp_path, value="/srv/sparkring/private/api-keys"):
    path = _site(tmp_path)
    data = json.loads(path.read_text())
    data["api_keys_file"] = value
    path.write_text(json.dumps(data))
    return path


def test_site_accepts_absolute_api_keys_file(tmp_path):
    site, _, _ = mesh_profile.load_site(_site_with_api_keys(tmp_path))
    assert site["api_keys_file"] == "/srv/sparkring/private/api-keys"


@pytest.mark.parametrize("value", ["relative/keys", "/srv/../etc/keys", "/srv/a:b", "/srv/a\nexec", "/"])
def test_site_rejects_unsafe_api_keys_file(tmp_path, value):
    with pytest.raises(ValueError):
        mesh_profile.load_site(_site_with_api_keys(tmp_path, value))


def test_render_omits_api_keys_file_when_site_is_silent(tmp_path, manifest_bundle):
    output = tmp_path / "rendered-default"
    mesh_profile.render(_site(tmp_path), manifest_bundle, output)
    for rank in range(4):
        assert "API_KEYS_FILE" not in mesh_profile.defaults(output / f"rank{rank}.env")


def test_render_propagates_api_keys_file_to_every_rank(tmp_path, manifest_bundle):
    path = _site_with_api_keys(tmp_path)
    output = tmp_path / "rendered-keys"
    mesh_profile.render(path, manifest_bundle, output)
    for rank in range(4):
        values = mesh_profile.defaults(output / f"rank{rank}.env")
        assert values["API_KEYS_FILE"] == "/srv/sparkring/private/api-keys"
    mesh_profile.load_site(output / "site.json")


@pytest.mark.parametrize("value", [True, False, None, 0, -1, 0.5, 900.0, "900", float("nan"), float("inf"), 2147483648])
def test_site_rejects_invalid_liveness_output_seconds(tmp_path, value):
    path = _site(tmp_path)
    data = json.loads(path.read_text())
    data["liveness_output_seconds"] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="liveness_output_seconds"):
        mesh_profile.load_site(path)


def test_liveness_override_preserves_default_and_canonical_regeneration(tmp_path, manifest_bundle):
    path = _site(tmp_path)
    default = tmp_path / "default"
    mesh_profile.render(path, manifest_bundle, default)
    data = json.loads(path.read_text())
    data["liveness_output_seconds"] = 900
    data["api_keys_file"] = "/srv/sparkring/private/api-keys"
    path.write_text(json.dumps(data))
    selected = tmp_path / "selected"
    repeated = tmp_path / "repeated"
    mesh_profile.render(path, manifest_bundle, selected)
    mesh_profile.render(selected / "site.json", manifest_bundle, repeated)
    for rank in range(4):
        name = f"rank{rank}.env"
        baseline = mesh_profile.defaults(default / name)
        actual = mesh_profile.defaults(selected / name)
        assert baseline["SPARKRING_LIVENESS_OUTPUT_SECONDS"] == "300"
        assert actual["SPARKRING_LIVENESS_OUTPUT_SECONDS"] == "900"
        assert actual["API_KEYS_FILE"] == data["api_keys_file"]
        assert (selected / name).read_bytes() == (repeated / name).read_bytes()
        del actual["API_KEYS_FILE"]
        actual["SPARKRING_LIVENESS_OUTPUT_SECONDS"] = "300"
        assert actual == baseline
