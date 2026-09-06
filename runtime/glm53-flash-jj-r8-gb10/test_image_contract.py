from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_module("jj_r8_build_image", HERE / "build_image.py")
install = load_module("jj_r8_install_overlay", HERE / "install_overlay.py")
verify = load_module("jj_r8_verify_image", HERE / "verify_image.py")
metrics_patch = load_module(
    "jj_r8_metrics_patch",
    HERE / "patch_kv_metrics_logging.py",
)


def test_metrics_patch_uses_connector_owned_compact_lines(tmp_path: Path) -> None:
    target = tmp_path / "metrics.py"
    target.write_text(
        "            xfer_metrics = self.transfer_stats_accumulator.reduce()\n"
        "            xfer_metrics_str = \", \".join(f\"{k}={v}\" for k, v in xfer_metrics.items())\n"
        "            log_fn(\"KV Transfer metrics: %s\", xfer_metrics_str)\n",
        encoding="utf-8",
    )

    metrics_patch.apply_patch(target)

    result = target.read_text(encoding="utf-8")
    assert "format_log_lines" in result
    assert 'log_fn("%s", line)' in result
    assert 'log_fn("KV Transfer metrics: %s", xfer_metrics_str)' in result


def test_image_packages_dflash_warmup_and_rank_zero_waits_for_it() -> None:
    recipe = (HERE / "Dockerfile").read_text(encoding="utf-8")
    builder = (HERE / "build_image.py").read_text(encoding="utf-8")
    launcher = (HERE / "launch-rank.sh").read_text(encoding="utf-8")

    assert "COPY warmup_dflash.py /opt/sparkring/bin/warmup_dflash.py" in recipe
    assert "COPY serve_with_warmup.py /opt/sparkring/bin/serve-with-warmup.py" in recipe
    assert (
        "COPY scheduler_liveness.py /opt/sparkring/bin/scheduler_liveness.py"
        in recipe
    )
    assert "chmod 0755 /opt/sparkring/bin/serve-with-warmup.py" in recipe
    assert '"warmup_dflash.py"' in builder
    assert '"serve_with_warmup.py"' in builder
    assert '"scheduler_liveness.py"' in builder
    assert '0) container_action=(run -d)' in launcher
    assert '1) container_action=(create)' in launcher
    assert 'container_command=(docker "${container_action[@]}"' in launcher
    assert 'container_id="$("${container_command[@]}")"' in launcher
    assert '"${rank}" == 0 && "${DFLASH_WARMUP}" == 1' in launcher
    assert "--entrypoint /opt/sparkring/bin/serve-with-warmup.py" in launcher
    assert "rank-0 engine readiness timed out" in launcher
    wrapper = (HERE / "serve_with_warmup.py").read_text(encoding="utf-8")
    main_source = wrapper.split("def main() -> int:", 1)[1]
    assert main_source.index("READY_PATH.unlink(missing_ok=True)") < main_source.index(
        'subprocess.Popen(["vllm", "serve"'
    )


def test_pins_bind_effective_sources_and_operator_defaults() -> None:
    pins = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
    assert pins["compatibility_locator"] == {
        "directory": "runtime/glm53-flash-jj-r8-gb10",
        "schema_prefix": "sparkring-glm53-jj-r8-gb10",
        "meaning": (
            "Stable filesystem and JSON interface locator. The r8 text identifies "
            "the interface family, not the embedded vLLM source composition."
        ),
    }
    assert pins["operator_image"] == {
        "reference": (
            "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@"
            "sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f"
        ),
        "image_id": (
            "sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075"
        ),
        "platform": "linux/arm64",
        "status": "qualified",
        "performance_status": "research-only",
        "qualification_scope": (
            "registry-pull verified on four ranks; GLM-5.3 Flash TP4/DCP4 "
            "passed embedded dual-rail SIRCL startup, concurrent SparkCache "
            "ownership drain, and persistent restart restore"
        ),
        "receipt": "glm53-dcp4-sircl-public-image-receipt.json",
    }
    assert pins["vllm"]["commit"] == (
        "e02b174693e13859de61811b5e8cd13d5308e259"
    )
    assert pins["vllm"]["public_tag"] == "sparkring-glm53-flash-gb10-e02b1746"
    assert pins["vllm"]["public_tag_object"] == (
        "2ac6883ac5156db713493fe9683bea99ecf928a4"
    )
    assert pins["vllm"]["public_tag_commit"] == pins["vllm"]["commit"]
    assert pins["vllm"]["tree"] == "6caadd392ddea2dc90441d0a078da67f38d2fd3a"
    assert pins["vllm"]["package_tree"] == (
        "c91299c2303dc05abc85aa2133224a749657a583"
    )
    assert pins["vllm"]["delta_patch_id"] == (
        "f3978963e81ccf3f932696e48079434223c365e3"
    )
    assert pins["vllm"]["proven_base_commit"] == (
        "22ffe1401ca9bd3e4503e62de7b414deca7661a1"
    )
    assert pins["vllm"]["b12x_kda_prefill_upstream_commit"] == (
        "54371894ecaa77f2725a1c99e018f3fe93d358dd"
    )
    assert pins["vllm"]["b12x_kda_workspace_isolation_upstream_commit"] == (
        "57a6169a5c229a5ca8c24791762b1fc51c89e58d"
    )
    assert pins["vllm"]["b12x_sparse_mla_dsa_upstream_commit"] == (
        "d662a1b0890271915c25439f22247ee22234739a"
    )
    assert pins["vllm"]["b12x_c4_indexer_binding_upstream_commit"] == (
        "d6687475a3f2dbe7848663fd3e5174d90921a3da"
    )
    assert pins["vllm"]["b12x_sparse_mla_cache_lengths_upstream_commit"] == (
        "83cb22a0e3f7ec4d2fb43f6ead34ba4d4a87a634"
    )
    assert pins["vllm"]["community_release"] == "Jovian Judgement Community R10"
    assert pins["vllm"]["community_parent_commit"] == (
        "55969c16d4da57da76ee5729f3102d4b2003833c"
    )
    assert pins["vllm"]["sparse_pooled_index_commit"] == (
        "adb69eac865a1a37081fa4edb9f7599a351f7aac"
    )
    assert pins["vllm"]["fwht_scaling_commit"] == (
        "ae37fd6ed8df72d4bd8cdc067c7c241c93408235"
    )
    assert pins["vllm"]["scheduler_prefill_cadence_component_commit"] == (
        "f1191b9090cd02ac49238c8e4f371050759703b6"
    )
    assert pins["vllm"]["scheduler_prefill_cadence_pull_request_head"] == (
        "2412a6f34ab1412f86ed3e4cdd355271a082d93d"
    )
    assert pins["b12x"] == {
        "repository": "https://github.com/voipmonitor/b12x.git",
        "upstream_repository": "https://github.com/local-inference-lab/b12x.git",
        "commit": "9ae41c5cb9935d740456479954b0089f80bd2ef2",
        "tree": "6e77441fe99f6ead7ff2cc2b6a8a37fa4e93e30b",
        "package_tree": "12029e19da6543c5d225395f6da199d946b0972e",
    }
    assert pins["sparkcache"] == {
        "repository": "https://github.com/FujitsuPolycom/sparkcache.git",
        "commit": "66057174301a4759ca3a45207ea41016689449cb",
        "tree": "2448cf08d155ba90c95699c02c46863bbf9ce301",
        "package_tree": "1fe4c76203be99c6a32640d8e2916889330d429b",
        "source_tree_sha256": (
            "80b049c647bc28fdc039021d08a7eb3276846c1616b77b9ba18ba2bc38da8d99"
        ),
        "cuda_placement_sha256": (
            "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c"
        ),
        "cuda_snapshot_sha256": (
            "4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5"
        ),
        "manager_page_lease_contract_sha256": (
            "480bdc463e3722cf28aa460021da689d12d9049ae9fd8238252fcf1db5544b53"
        ),
    }
    defaults = pins["defaults"]
    assert defaults["max_model_len"] == 1048576
    assert defaults["max_num_batched_tokens"] == 8192
    assert defaults["prefill_schedule_interval"] == 2
    assert defaults["shared_prefix_lease_ttl_seconds"] == 300
    assert defaults["kv_cache_bytes_per_rank"] == {
        "dcp1": 25769803776,
        "dcp2": 25769803776,
        "dcp4": 25769803776,
    }
    assert defaults["full_ckv_gather_max_tokens"] == 524288
    assert defaults["async_page_capture"] == {
        "preferred_dcp4_enabled": True,
        "launcher_fallback_enabled": False,
        "slot_bytes_by_dcp": {
            "dcp1": 8589934592,
            "dcp2": 5368709120,
            "dcp4": 3221225472,
        },
        "slot_count": 2,
    }
    assert defaults["dcp"] == {
        "1": {"cp_kv_cache_interleave_size": 1, "full_ckv_gather": False},
        "2": {"cp_kv_cache_interleave_size": 4, "full_ckv_gather": True},
        "4": {"cp_kv_cache_interleave_size": 4, "full_ckv_gather": True},
    }


def test_dockerfile_preserves_native_components_and_binds_overlays() -> None:
    recipe = (HERE / "Dockerfile").read_text(encoding="utf-8")
    for identity in (
        "f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5",
        "e02b174693e13859de61811b5e8cd13d5308e259",
        "6caadd392ddea2dc90441d0a078da67f38d2fd3a",
        "c91299c2303dc05abc85aa2133224a749657a583",
        "54371894ecaa77f2725a1c99e018f3fe93d358dd",
        "57a6169a5c229a5ca8c24791762b1fc51c89e58d",
        "d662a1b0890271915c25439f22247ee22234739a",
        "d6687475a3f2dbe7848663fd3e5174d90921a3da",
        "83cb22a0e3f7ec4d2fb43f6ead34ba4d4a87a634",
        "9ae41c5cb9935d740456479954b0089f80bd2ef2",
        "6e77441fe99f6ead7ff2cc2b6a8a37fa4e93e30b",
        "12029e19da6543c5d225395f6da199d946b0972e",
        "5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3",
        "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
        "4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5",
        "2aac02232a9115037723aa1dd40483a5693a3e1e",
        "85a231e6d2a290f7d6cccbc2cc6b1ccad7a6adbefc7ce4dde05b158f249aadd4",
        "61aa0ec56a1b438439bed8611dab0353d2c72c10af02bbd917fb77c87b33e5fc",
    ):
        assert identity in recipe
    assert (
        'org.sparkcache.commit="66057174301a4759ca3a45207ea41016689449cb"'
        in recipe
    )
    assert (
        'org.sparkcache.tree="2448cf08d155ba90c95699c02c46863bbf9ce301"'
        in recipe
    )
    assert (
        "org.sparkcache.source-sha256="
        '"80b049c647bc28fdc039021d08a7eb3276846c1616b77b9ba18ba2bc38da8d99"'
        in recipe
    )
    assert "COPY bundle/sircl/ /opt/spark-sircl/" in recipe
    assert "org.sparkcache.dcp-layouts=\"1,2,4\"" in recipe
    assert "org.sparkcache.cache-geometry=\"manager-pages-v2\"" in recipe
    assert (
        "org.sparkcache.publication-schema="
        '\"snapshot-v1,page-tail-cow-v1,page-tail-cow-v2\"'
        in recipe
    )
    assert "compact-startup-no-deep-ep" in recipe
    assert "fastsafetensors" in recipe


def test_parent_contract_binds_proven_arm64_image() -> None:
    pins = build.load_pins()
    document = {
        "Architecture": "arm64",
        "Os": "linux",
        "Id": pins["parent"]["image_id"],
        "Config": {"Labels": dict(pins["parent"]["required_labels"])},
    }
    build.validate_parent(document, pins)
    document["Config"]["Labels"]["org.sparkring.nccl.sha256"] = "0" * 64
    try:
        build.validate_parent(document, pins)
    except build.BuildError as error:
        assert "nccl.sha256" in str(error)
    else:
        raise AssertionError("transport drift was accepted")


def test_runtime_hashes_are_enforced_inside_image() -> None:
    pins = build.load_pins()
    verifier = (HERE / "verify_image.py").read_text(encoding="utf-8")
    assert 'pins["vllm"]["runtime_file_sha256"]' in verifier
    assert set(pins["vllm"]["runtime_file_sha256"]) == {
        "vllm/config/scheduler.py",
        "vllm/v1/core/sched/interface.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/engine/core.py",
    }
    labels = verify.expected_labels(pins)
    assert labels["org.sparkring.vllm.package-tree"] == (
        "c91299c2303dc05abc85aa2133224a749657a583"
    )
    assert labels["org.sparkring.vllm.delta-patch-id"] == (
        "f3978963e81ccf3f932696e48079434223c365e3"
    )
    assert labels["org.sparkring.vllm.community-parent"] == (
        "55969c16d4da57da76ee5729f3102d4b2003833c"
    )
    assert labels["org.sparkring.b12x.package-tree"] == (
        "12029e19da6543c5d225395f6da199d946b0972e"
    )
    assert labels["org.sparkring.vllm.b12x-kda-prefill-upstream"] == (
        "54371894ecaa77f2725a1c99e018f3fe93d358dd"
    )
    assert labels[
        "org.sparkring.vllm.b12x-kda-workspace-isolation-upstream"
    ] == (
        "57a6169a5c229a5ca8c24791762b1fc51c89e58d"
    )
    assert labels["org.sparkring.vllm.b12x-sparse-mla-dsa-upstream"] == (
        "d662a1b0890271915c25439f22247ee22234739a"
    )
    assert labels["org.sparkring.vllm.b12x-c4-indexer-binding-upstream"] == (
        "d6687475a3f2dbe7848663fd3e5174d90921a3da"
    )
    assert labels[
        "org.sparkring.vllm.b12x-sparse-mla-cache-lengths-upstream"
    ] == (
        "83cb22a0e3f7ec4d2fb43f6ead34ba4d4a87a634"
    )
    assert labels["org.sparkring.sircl.source-tree"] == (
        pins["sircl"]["spark_transport_tree"]
    )
    assert labels["org.sparkring.sircl.manifest-sha256"] == (
        pins["sircl"]["overlay_manifest_sha256"]
    )
    assert labels["org.sparkring.sircl.native-sha256"] == (
        pins["sircl"]["native_sha256"]
    )
    assert "verify_public_overlay(SIRCL_ROOT, sircl_manifest)" in verifier


def test_builder_requires_source_pinned_b12x_and_records_public_sources() -> None:
    builder = (HERE / "build_image.py").read_text(encoding="utf-8")
    assert '"--b12x-source"' in builder
    assert 'pins["b12x"], "b12x"' in builder
    assert '"b12x-source-manifest.json"' in builder
    assert '"public_tag": pins["vllm"]["public_tag"]' in builder
    assert '"public_tag_object": pins["vllm"]["public_tag_object"]' in builder
    assert '"upstream_repository": pins["b12x"]["upstream_repository"]' in builder


def test_builder_requires_the_receipt_bound_prebuilt_sircl_library() -> None:
    builder = (HERE / "build_image.py").read_text(encoding="utf-8")
    assert '"--sircl-library"' in builder
    assert "SIRCL native library digest mismatch" in builder
    assert '"HEAD:spark_transport"' in builder
    assert 'context / "bundle/sircl"' in builder
    assert '"runtime/public-overlay-files.json"' in builder


def test_b12x_source_overlay_replaces_the_complete_inherited_package(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "b12x"
    source_module = source / "sequence" / "kda_prefill.py"
    source_module.parent.mkdir(parents=True)
    source_module.write_text("SOURCE_IDENTITY = '9ae41c5c'\n", encoding="utf-8")
    (source / "__init__.py").write_text("", encoding="utf-8")

    installed = tmp_path / "site-packages" / "b12x"
    (installed / "sequence").mkdir(parents=True)
    (installed / "stale_parent_module.py").write_text("stale\n", encoding="utf-8")
    (installed / "sequence" / "kda_prefill.py").write_text(
        "SOURCE_IDENTITY = 'parent'\n", encoding="utf-8"
    )
    (installed / "__pycache__").mkdir()
    (installed / "__pycache__" / "stale.pyc").write_bytes(b"stale")

    install.replace_python_package(source, installed)

    installed_files = {
        path.relative_to(installed).as_posix()
        for path in installed.rglob("*")
        if path.is_file()
    }
    assert installed_files == {"__init__.py", "sequence/kda_prefill.py"}
    manifest = tmp_path / "b12x-source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "commit": "9ae41c5cb9935d740456479954b0089f80bd2ef2",
                "files": {
                    "b12x/__init__.py": install.file_sha256(
                        installed / "__init__.py"
                    ),
                    "b12x/sequence/kda_prefill.py": install.file_sha256(
                        installed / "sequence" / "kda_prefill.py"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    assert verify.verify_manifest(
        tmp_path / "site-packages",
        manifest,
        "9ae41c5cb9935d740456479954b0089f80bd2ef2",
    ) == 2
    install.verify_package_file_set(
        installed,
        install.load_manifest(
            manifest, "9ae41c5cb9935d740456479954b0089f80bd2ef2"
        ),
        "b12x",
    )

    source_module = installed / "sequence" / "kda_prefill.py"
    source_module.write_text("changed\n", encoding="utf-8")
    try:
        verify.verify_manifest(
            tmp_path / "site-packages",
            manifest,
            "9ae41c5cb9935d740456479954b0089f80bd2ef2",
        )
    except verify.VerificationError as error:
        assert "source identity mismatch: b12x/sequence/kda_prefill.py" in str(error)
    else:
        raise AssertionError("changed B12X source was accepted")


def test_launcher_keeps_gather_workspace_below_native_context_limit() -> None:
    launcher = (HERE / "launch-rank.sh").read_text(encoding="utf-8")
    environment = (HERE / "runtime.env.example").read_text(encoding="utf-8")
    assert 'MAX_MODEL_LEN:=1048576' in launcher
    assert 'MAX_NUM_BATCHED_TOKENS:=8192' in launcher
    assert 'PREFILL_SCHEDULE_INTERVAL:=2' in launcher
    assert 'KV_CACHE_MEMORY_BYTES:=auto' in launcher
    assert 'B12X_MLA_CKV_GATHER_MAX_TOKENS:=524288' in launcher
    assert 'SPARKCACHE_MAX_SPAN_TOKENS:=1048576' in launcher
    assert 'SPARKCACHE_ASYNC_PAGE_CAPTURE:=0' in launcher
    assert 'SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES:=auto' in launcher
    assert 'SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT:=2' in launcher
    assert 'SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS:=300' in launcher
    assert "spark_cache_async_page_capture_library" in launcher
    assert "vllm-manager-page-async-contract-55969c16.json" in launcher
    for value in (
        "MAX_MODEL_LEN=1048576",
        "MAX_NUM_BATCHED_TOKENS=8192",
        "PREFILL_SCHEDULE_INTERVAL=2",
        "MAX_IMAGES_PER_PROMPT=4",
        "MAX_VIDEOS_PER_PROMPT=1",
        "KV_CACHE_MEMORY_BYTES='auto'",
        "B12X_MLA_CKV_GATHER_MAX_TOKENS=524288",
        "SPARKCACHE_MAX_SPAN_TOKENS=1048576",
        "SPARKCACHE_ASYNC_PAGE_CAPTURE=1",
        "SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES='auto'",
        "SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT=2",
        "SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS=300",
    ):
        assert value in environment
    for text in (launcher, environment):
        assert (
            "glm53-flash-vllm-e02b1746-b12x-9ae41c5c-"
            "dcp4-page-tail-cow-v2"
        ) in text
    assert "IMAGE_REF" in launcher
    assert "IMAGE_ID" in launcher


def test_operator_docs_distinguish_page_tails_from_published_rollback() -> None:
    runtime_readme = (HERE / "README.md").read_text(encoding="utf-8")
    quickstart = (
        ROOT / "docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md"
    ).read_text(encoding="utf-8")
    runtime_index = (ROOT / "runtime/README.md").read_text(encoding="utf-8")

    for document in (runtime_readme, quickstart):
        for schema in ("snapshot-v1", "tail-cow-v1", "tail-cow-v2"):
            assert schema in document
        assert "sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f" in document
        assert "sparkcache: capacity" in document
        assert "sparkcache: publications" in document
        assert "sparkcache: writes" in document

    rollback_digest = (
        "3c377f1e4136285ebf66c32c36c3d01f"
        "d929f8aba0836cd0a16ed63cfd7e1762"
    )
    assert rollback_digest in runtime_readme
    assert rollback_digest in quickstart
    assert rollback_digest in runtime_index
    assert "380283a506aeb8f9" not in runtime_index


def test_operator_docs_name_public_sources_and_explain_stable_locators() -> None:
    runtime_readme = (HERE / "README.md").read_text(encoding="utf-8")
    quickstart = (
        ROOT / "docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md"
    ).read_text(encoding="utf-8")
    sircl_readme = (ROOT / "docs/SIRCL.md").read_text(encoding="utf-8")
    runtime_index = (ROOT / "runtime/README.md").read_text(encoding="utf-8")
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for document in (runtime_readme, quickstart):
        assert "sparkring-glm53-flash-gb10-e02b1746" in document
        assert "e02b174693e13859de61811b5e8cd13d5308e259" in document
        assert "local-inference-lab/b12x" in document
        assert "voipmonitor/b12x" in document
        assert "`tail-cow-v2` to the cache-identity\nwire value `page-tail-cow-v2`" in document

    for document in (runtime_readme, runtime_index):
        assert "stable compatibility locator" in document or (
            "stable filesystem and interface\nlocators" in document
        )
        assert "does not identify the embedded vLLM source" in " ".join(
            document.split()
        )

    assert "four-rank TP4/DCP4\nfunctional checks" in sircl_readme
    assert "**research-only**" in sircl_readme
    assert "b12x-kda-dcp4-20260903.md" in root_readme
    assert "C4: 90.36" in root_readme

    active_docs = "\n".join((runtime_readme, quickstart, sircl_readme))
    for ambiguous in (
        "Current evidence",
        "current composition",
        "fail-stop",
        "concurrent prompt gate",
        "earlier SparkCache source composition",
        "older runtime's default directory",
    ):
        assert ambiguous not in active_docs


def test_multimodal_lease_image_receipt_binds_public_artifact_and_smoke() -> None:
    receipt = json.loads(
        (HERE / "multimodal-lease300-image-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "live-smoke-verified"
    assert receipt["artifact"] == {
        "registry": (
            "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@"
            "sha256:3c377f1e4136285ebf66c32c36c3d01fd929f8aba0836cd0a16ed63cfd7e1762"
        ),
        "published_tag": (
            "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:"
            "20260902-r10-multimodal-lease300"
        ),
        "image_id": (
            "sha256:d1a07147c9e25f3d3e0af6b1499c4988b1ae61138e327aa05c9ad9dc568e39a9"
        ),
        "platform": "linux/arm64",
        "archive_sha256": (
            "a88cd040bb38ed0092d8de8fc00aa2ac7e15a4352258a73394fb876fea3756a4"
        ),
        "archive_bytes": 8469866878,
    }
    assert receipt["sources"]["sparkcache_commit"] == (
        "b7d1c188a3f9e78595e6e7b649f3751131e269ea"
    )
    assert receipt["launch"]["shared_prefix_lease_ttl_seconds"] == 300
    assert receipt["launch"]["max_images_per_prompt"] == 4
    assert receipt["launch"]["max_videos_per_prompt"] == 1
    assert receipt["verification"]["physical_ranks"] == 4
    assert receipt["verification"]["running_image_ids_equal"] is True
    assert receipt["verification"]["sparkcache_source_bind_mounts"] == 0
    assert receipt["verification"]["image_probe"] == {
        "input": "448x448 solid-red PNG",
        "multimodal_image_tokens": 256,
        "dominant_color_identified": "red",
    }


def test_page_tail_v2_receipt_binds_the_local_image() -> None:
    receipt = json.loads(
        (HERE / "page-tail-v2-local-image-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["schema"] == "sparkring-glm53-jj-r8-gb10-image-receipt/v1"
    assert receipt["status"] == "implemented"
    assert receipt["platform"] == "linux/arm64"
    assert receipt["image_id"] == (
        "sha256:c3f85b2350609b6ff1201b8c5998f881ff4cef8b671d6783b543f841040915c0"
    )
    assert receipt["inside_image"]["sparkcache_commit"] == (
        "737ed1399f559ba036fb0e358541744011afd47d"
    )
    assert receipt["labels"]["org.sparkcache.publication-schema"] == (
        "snapshot-v1,page-tail-cow-v1,page-tail-cow-v2"
    )


def test_dflash_readiness_receipt_records_engine_level_recovery() -> None:
    receipt = json.loads(
        (HERE / "dflash-jit-readiness-validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "qualified"
    assert receipt["failure_observation"]["api_health_status"] == 200
    assert receipt["failure_observation"]["requests_running_after_timeout"] == 2
    assert receipt["readiness"]["triton_block_sizes_covered"] == [
        16,
        32,
        64,
        128,
        256,
    ]
    assert receipt["validation"]["concurrent_restored_prefix_tokens"] == 921600
    assert receipt["validation"]["jit_events_after_readiness"] == 0
    assert receipt["validation"]["cuda_or_cublas_errors_after_readiness"] == 0
    assert receipt["validation"]["requests_running_after_validation"] == 0
    assert receipt["validation"]["image_transfer_pressure_replay_passed"] is True


def test_page_tail_v2_public_receipt_binds_registry_and_runtime() -> None:
    receipt = json.loads(
        (HERE / "page-tail-v2-public-image-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "published-live-verified"
    assert receipt["artifact"] == {
        "registry": (
            "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@"
            "sha256:4ce98659c30d9e9c313b1018a2675e5f135a0404e7cc00951b4ade161c0a711f"
        ),
        "published_tag": (
            "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:"
            "20260902-r10-page-tail-v2"
        ),
        "image_id": (
            "sha256:c3f85b2350609b6ff1201b8c5998f881ff4cef8b671d6783b543f841040915c0"
        ),
        "platform": "linux/arm64",
        "distribution_archive_sha256": (
            "841c6da413dbb5983c1b1051598a1015bdafb33f096463b3b276aae85c976578"
        ),
        "distribution_archive_bytes": 9224799706,
    }
    assert receipt["sources"]["sparkcache_commit"] == (
        "737ed1399f559ba036fb0e358541744011afd47d"
    )
    assert receipt["validation"]["post_readiness_jit_events"] == 0
    assert receipt["validation"]["requests_running_after_validation"] == 0


def test_async_store_completion_receipt_keeps_published_source_identity() -> None:
    receipt = json.loads(
        (HERE / "async-store-completion-public-image-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    pins = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "implemented"
    assert receipt["artifact"]["registry"] != pins["operator_image"]["reference"]
    assert receipt["artifact"]["image_id"] != pins["operator_image"]["image_id"]
    assert receipt["sources"]["sparkcache_commit"] == (
        "9c6218c96f1db233c0d17691dbc32a7d9fb2c0e4"
    )
    assert receipt["sources"]["sparkcache_source_sha256"] == (
        "f8adb4ecdadd524e79cf1ef14e7f3d83d1f20ff07c79333b2c7c0d9ea12919d5"
    )
    assert receipt["sources"]["sparkcache_commit"] != pins["sparkcache"]["commit"]
    assert receipt["construction_verification"] == {
        "inside_image_verification": "passed",
        "retained_native_extensions": 15,
        "registry_pull_image_id_matched": True,
        "registry_pull_sparkcache_commit_matched": True,
        "registry_pull_sparkring_commit_matched": True,
    }
    assert receipt["source_composition_validation"]["busy_saver_skips_per_rank"] == [
        12,
        12,
        12,
        12,
    ]
    assert receipt["source_composition_validation"]["idle_kv_cache_usage_percent"] == 0.0


def test_dcp4_public_image_receipt_records_b12x_kda() -> None:
    receipt = json.loads(
        (HERE / "glm53-dcp4-sircl-public-image-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "qualified"
    assert receipt["configuration"]["kda_prefill_backend"] == "b12x"
    assert receipt["sources"]["vllm_public_tag"] == (
        "sparkring-glm53-flash-gb10-e02b1746"
    )
    assert receipt["sources"]["vllm_composition"] == (
        "e02b174693e13859de61811b5e8cd13d5308e259"
    )
    assert receipt["sources"]["vllm_public_tag_commit"] == receipt["sources"][
        "vllm_composition"
    ]
    assert receipt["sources"]["b12x_upstream_repository"] == (
        "https://github.com/local-inference-lab/b12x.git"
    )
    assert receipt["sources"]["b12x_source_repository"] == (
        "https://github.com/voipmonitor/b12x.git"
    )
    assert receipt["sources"]["b12x"] == (
        "9ae41c5cb9935d740456479954b0089f80bd2ef2"
    )
    health_receipt = json.loads(
        (HERE / "sircl-capability-health-live-validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert health_receipt["system"]["sircl"]["post_output_health_check"] is True
    assert "post_output_health_gate" not in health_receipt["system"]["sircl"]
    assert receipt["validation"]["sircl_scope"] == {
        "functional_status": "qualified",
        "performance_status": "research-only",
        "details": (
            "The exact image passed the recorded embedded dual-rail SIRCL "
            "TP4/DCP4 functional checks. The receipt does not establish a broad "
            "SIRCL-versus-NCCL performance result."
        ),
    }


def test_sircl_public_build_receipt_binds_overlay_and_native_test(
    tmp_path: Path,
) -> None:
    receipt_path = HERE / "sircl-public-build-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    pins = build.load_pins()
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == (
        pins["sircl"]["build_receipt_sha256"]
    )
    assert receipt["schema"] == "sparkring-glm53-sircl-public-build/v1"
    assert receipt["status"] == "implemented"
    assert receipt["validation_scope"] == "native-built-tested"
    assert receipt["source"] == {
        "repository": "https://github.com/FujitsuPolycom/sparkring.git",
        "spark_transport_tree": "2aac02232a9115037723aa1dd40483a5693a3e1e",
        "public_overlay_spec_sha256": (
            "bc93d5069f2b3faaa7e87d2f24aadd9f7878bff67abd93eeec3a0975da46f6fd"
        ),
    }
    overlay_spec = ROOT / "runtime" / "public-overlay-files.json"
    assert hashlib.sha256(overlay_spec.read_bytes()).hexdigest() == (
        receipt["source"]["public_overlay_spec_sha256"]
    )
    overlay_builder = load_module(
        "sircl_public_overlay", ROOT / "runtime" / "build-public-overlay.py"
    )
    output = tmp_path / "sircl-overlay"
    manifest = overlay_builder.build(ROOT, overlay_spec, output)
    assert len(manifest["files"]) == receipt["overlay"]["files"]
    manifest_path = output / receipt["overlay"]["manifest"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        receipt["overlay"]["manifest_sha256"]
    )
    assert verify.verify_public_overlay(output, manifest_path) == 14
    (output / "sitecustomize.py").write_text("changed", encoding="utf-8")
    try:
        verify.verify_public_overlay(output, manifest_path)
    except verify.VerificationError as error:
        assert "SIRCL source identity mismatch" in str(error)
    else:
        raise AssertionError("changed SIRCL Python source was accepted")
    assert len(receipt["native"]["sha256"]) == 64
    int(receipt["native"]["sha256"], 16)
    assert receipt["validation"] == {
        "cmake_build_exit_code": 0,
        "ctest_total": 28,
        "ctest_passed": 28,
        "ctest_failed": 0,
        "alternating_stream_cuda_smoke": "passed",
    }


def test_async_capture_image_receipt_binds_public_artifact_and_live_results() -> None:
    receipt = json.loads(
        (HERE / "async-capture-image-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "qualified"
    assert receipt["artifact"]["registry"].endswith(
        "@sha256:bc7d079f16ff4a418669c58c5250f2da52e989a0c5805569ba9429d41b765f65"
    )
    assert receipt["artifact"]["image_id"] == (
        "sha256:35f397668c01075d0bdd28bbdb3398afd3744df6086646c6f68bcf7ebe7f918f"
    )
    assert receipt["artifact"]["published_tag"].endswith(
        ":20260901-r10-async-telemetry"
    )
    assert receipt["sources"]["sparkring_commit"] == (
        "d2f8911427d64bbb89c275814777fc3f8112fd21"
    )
    assert receipt["sources"]["sparkcache_commit"] == (
        "c5dda75ec46bf235f6ece6e0d0174c1e41bd805a"
    )
    assert receipt["conditions"]["capture_slot_bytes"] == 3 * 1024**3
    assert receipt["conditions"]["operator_template_storage_root"] == (
        "glm53-flash-dcp4-snapshot-v1"
    )
    assert receipt["conditions"]["live_validation_storage_root"] == (
        "jj-r10-async-ab-v1"
    )
    assert receipt["conditions"]["operator_template_storage_root_measured"] is False
    assert receipt["deep_entry_writer"] == {
        "image_id": (
            "sha256:8e586e6ad9b4f30a8ccef1bfd8b76194524e156089c958907872d0f8735a09b2"
        ),
        "archive_sha256": (
            "47c800fd73130c1fe26b707caa2c64f81ed43c951fe2019d8836cd0b883dbe48"
        ),
        "sparkcache_commit": "6d83c7d8cb6ace96e657b3d0150116d0fe4e011c",
        "sparkcache_tree": "0bb871bd1e8d3893a11686f0ba404bd4b6240e4d",
        "sparkcache_source_sha256": (
            "67edb651835b978cbaf2519f92e68251145c1368a22cc0339f706d5c2144f862"
        ),
        "vllm_commit": "22ffe1401ca9bd3e4503e62de7b414deca7661a1",
        "vllm_tree": "1bb7f10a5838d348ca2fcb0134b05ad768d3340b",
        "cuda_snapshot_sha256": (
            "4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5"
        ),
        "topology": "TP4/DCP4",
        "publication_schema": "snapshot-v1",
        "storage_root": "jj-r10-async-ab-v1",
        "checkpoint_identity": (
            "same target and draft repositories and revisions recorded in conditions"
        ),
    }
    assert (
        receipt["validation"]["live"]["identical_prompt_prime_before_restore"]
        is False
    )
    assert receipt["validation"]["live"]["startup_inventory"] == {
        "checked": 29,
        "offered": 29,
        "rejected": 0,
    }
    assert receipt["validation"]["live"]["cold_126k_publication"][
        "capture_completion_observed_ms_by_rank"
    ] == [408.5, 404.3, 403.7, 404.3]
    assert receipt["validation"]["live"]["900k_restore"]["needle"] == "passed"
    assert receipt["validation"]["live"]["1m_restore"]["needle"] == "passed"
