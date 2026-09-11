"""Contracts for the source-pinned R33 ARM64 image and entrypoint."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest import mock


HERE = Path(__file__).resolve().parent
CANONICAL_PROFILES = HERE.parent / "profiles"


class PublicationDocumentationTests(unittest.TestCase):
    def test_qualification_documents_match_publication_hashes(self):
        root = HERE.parents[3]
        publication = json.loads((HERE.parent / "publication.json").read_text())
        for entry in publication["qualification"].values():
            path = root / entry["record"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
            document = path.read_text()
            for heading in ("Conditions", "Measurement", "Result", "Conclusion", "Limitations"):
                self.assertIn(f"## {heading}\n", document)
            self.assertIn("**research-only**", document)

    def test_preserved_image_documentation_names_its_dcp1_cache_profiles(self):
        document = (HERE / "README.md").read_text()
        for name in ("tp2-dcp1-sparkcache", "tp4-dcp1-sparkcache"):
            self.assertIn(f"`{name}`", document)

    def test_canonical_tp4_guide_documents_dcp4_cache_alternative(self):
        guide = HERE.parents[3] / "profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md"
        document = guide.read_text(encoding="utf-8")
        contract = json.loads((CANONICAL_PROFILES / "profile-contract.json").read_text())
        profile = contract["profiles"]["tp4-dcp4-sparkcache"]
        self.assertTrue(profile["sparkcache"])
        self.assertEqual(profile["decode_context_parallel_size"], 4)
        self.assertIn("`tp4-dcp4-sparkcache`", document)


def load_entrypoint():
    spec = importlib.util.spec_from_file_location(
        "r33_candidate_entrypoint", HERE / "entrypoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.path.insert(0, str(HERE))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def tp4_environment(contract: dict) -> dict[str, str]:
    return {
        **contract["common_environment"],
        "SOURCE_IMAGE_PROFILE": "tp4-dcp1",
        "SPARKRING_PROFILE_MODE": "custom",
        "SPARKRING_MANAGED_MESH_RENDERED": "1",
        "VLLM_B12X_KDA_PREFILL_COALESCING": "1",
        "VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT": "4",
        "NODE_RANK": "2",
        "MASTER_ADDR": "rank-zero",
        "NCCL_IB_HCA": "=hca0:1,hca1:1,hca2:1,hca3:1",
        "SIRCL_ENABLED": "1",
        "VLLM_SPARK_TP4_MODE": "custom",
        "VLLM_SPARK_TP4_VOCAB_MODE": "custom",
        "SPARK_TP4_LIBRARY": "/opt/sparkring/sircl/libspark_transport_capi.so",
        "NCCL_SWITCHLESS_RING_ONLY": "1",
        "SPARK_TP4_PEER0": "peer-zero",
        "SPARK_TP4_PEER1": "peer-one",
        "SPARK_TP4_DEVICE0": "hca0",
        "SPARK_TP4_DEVICE1": "hca1",
        "SPARK_TP4_GID0": "3",
        "SPARK_TP4_GID1": "3",
        "SPARK_TP4_CONTROL_PORT0": "12000",
        "SPARK_TP4_CONTROL_PORT1": "12001",
        "SPARKCACHE_ENABLED": "0",
    }


class CandidateImageContractTests(unittest.TestCase):
    def test_wheel_metadata_ignores_vendored_dist_info(self):
        module = load_module("r33_finalize_lock", "finalize_lock.py")
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "package.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    "package-1.2.3.dist-info/METADATA",
                    "Metadata-Version: 2.4\nName: package\nVersion: 1.2.3\n",
                )
                archive.writestr(
                    "package/_vendor/helper-9.0.dist-info/METADATA",
                    "Metadata-Version: 2.4\nName: helper\nVersion: 9.0\n",
                )
            self.assertEqual(module.wheel_metadata(wheel), ("package", "1.2.3"))

    def test_payload_probe_uses_the_same_wheel_path_as_docker(self):
        resolver = (HERE / "resolve_closure.sh").read_text()
        dockerfile = (HERE / "Dockerfile.candidate").read_text()
        self.assertIn('-v "$context/wheelhouse:/wheelhouse:ro"', resolver)
        self.assertIn('wheels[$index]="/wheelhouse/${wheels[$index]}"', resolver)
        self.assertIn('wheels[$index]="/wheelhouse/${wheels[$index]}"', dockerfile)

    def test_artifact_lock_pins_critical_inputs_and_excludes_bad_audio(self):
        lock = json.loads((HERE / "artifact-lock.json").read_text())
        self.assertEqual(lock["foundation"]["cuda"], "13.3")
        self.assertEqual(
            lock["foundation"]["image_id"],
            "sha256:6704db5df61d1110afaba538554abb024c85edb8f37a98e4e631166a10af3217",
        )
        self.assertEqual(
            lock["media_runtime"]["image_id"],
            "sha256:a1a72e18ad49d99f6194a2585bfdc5f32d79180cdf2cd015c0d0d451479d6a42",
        )
        artifacts = {item["name"]: item for item in lock["artifacts"]}
        self.assertNotIn("vllm", artifacts)
        self.assertEqual(
            lock["source_identities"]["vllm_integrated_tree"],
            "547f7091841728f21ab419012a766fd1df70a569",
        )
        self.assertEqual(
            lock["source_identities"]["vllm_continuation_port_commit"],
            "b611611a643502542c2d900057eb47e407b8379e",
        )
        self.assertEqual(
            lock["source_identities"]["vllm_prefix_hit_metadata_compatibility_commit"],
            "4405a965e54f14df00d28e5e23f9793f866aae39",
        )
        self.assertEqual(
            lock["source_identities"]["vllm_tp2_continuation_port_commit"],
            "8fe550fd876ddea18a23b597611baec15dec048e",
        )
        self.assertEqual(
            lock["source_identities"]["vllm_flash_attn_commit"],
            "f3e1a4f74c99145c0717709860bf765de1703779",
        )
        self.assertEqual(
            lock["source_identities"]["b12x_commit"],
            "68acfc14893c087aa9b3120bb984fde4c4e7a21f",
        )
        self.assertEqual(
            lock["source_identities"]["b12x_tree"],
            "284e7df8caff930477a314fea20d826256844de4",
        )
        self.assertEqual(
            artifacts["nccl-2.31.2-sparkring-routing"]["sha256"],
            "84a4b8d83fb5fa1f0d640d311ad38b45140672dae9889775fe1e4a3990479e47",
        )
        self.assertEqual(
            artifacts["sircl"]["sha256"],
            "bea00f2ba6051c2c0bcd2853aae894672aa7f1fe5a1d905edaa9120aabf74246",
        )
        self.assertIn("+cu133-", artifacts["torchaudio"]["source"])
        selected = {
            Path(item["source"]).name for item in lock["artifacts"] if item["install"]
        }
        for forbidden in lock["forbidden_inputs"]:
            self.assertNotIn(Path(forbidden).name, selected)
        pending = {item["name"]: item for item in lock["pending_artifacts"]}
        self.assertEqual(
            set(pending),
            {"vllm", "b12x", "flashinfer-python", "flashinfer-jit-cache"},
        )
        self.assertTrue(all(item["required"] for item in pending.values()))
        self.assertEqual(
            pending["vllm"]["receipt_sums"], "artifacts/vllm-package/SHA256SUMS"
        )
        self.assertEqual(pending["b12x"]["receipt_sums"], "artifacts/b12x/SHA256SUMS")
        contract = json.loads(
            (CANONICAL_PROFILES / "profile-contract.json").read_text()
        )
        self.assertEqual(
            contract["image"]["required_sources"], lock["source_identities"]
        )
        self.assertEqual(
            contract["image"]["artifact_lock_sha256"],
            __import__("hashlib")
            .sha256((HERE / "artifact-lock.json").read_bytes())
            .hexdigest(),
        )

    def test_b12x_builder_and_context_bind_the_checkpoint_export_port(self):
        build = (HERE.parent / "build_b12x.sh").read_text()
        self.assertIn("expected_commit=68acfc14893c087aa9b3120bb984fde4c4e7a21f", build)
        self.assertIn("expected_tree=284e7df8caff930477a314fea20d826256844de4", build)
        prepare = (HERE / "prepare_context.py").read_text()
        self.assertIn('sums_path = args.build_root / pending["receipt_sums"]', prepare)
        self.assertNotIn("completed FlashInfer receipt", prepare)

    def test_pending_artifact_receipts_accept_one_exact_basename(self):
        module = load_module("r33_prepare_context", "prepare_context.py")
        with tempfile.TemporaryDirectory() as directory:
            sums = Path(directory) / "SHA256SUMS"
            digest = "a" * 64
            sums.write_text(f"{digest}  b12x-1.3.0-py3-none-any.whl\n")
            self.assertEqual(
                module.load_sums(sums), {"b12x-1.3.0-py3-none-any.whl": digest}
            )
            sums.write_text(
                f"{digest}  first/b12x-1.3.0-py3-none-any.whl\n"
                f"{digest}  second/b12x-1.3.0-py3-none-any.whl\n"
            )
            with self.assertRaisesRegex(ValueError, "ambiguous SHA256SUMS"):
                module.load_sums(sums)

    def test_sparkcache_lease_contract_matches_composed_vllm_sources(self):
        contract_path = HERE.parent / "contracts/vllm-connector-jobs-r33-547f7091.json"
        contract = json.loads(contract_path.read_text())
        manifest = json.loads(
            (HERE.parent / "patches/vllm-r33-sparkring.manifest.json").read_text()
        )
        self.assertEqual(contract["vllm_tree"], manifest["result"]["tree"])
        records = {item["path"]: item for item in contract["files"]}
        reviewed = set(contract["semantic_review"]["affected_files"])
        self.assertEqual(
            reviewed,
            {
                "vllm/v1/core/sched/output.py",
                "vllm/v1/core/sched/scheduler.py",
                "vllm/v1/core/kv_cache_manager.py",
                "vllm/v1/core/single_type_kv_cache_manager.py",
                "vllm/v1/worker/gpu/model_runner.py",
            },
        )
        for name in reviewed:
            self.assertEqual(records[name]["sha256"], manifest["files"][name]["sha256"])
        profile = json.loads(
            (HERE.parent / "profiles/profile-contract.json").read_text()
        )
        self.assertEqual(
            profile["sparkcache_native"]["lease_contract"],
            "/opt/sparkring/contracts/vllm-connector-jobs-r33-547f7091.json",
        )
        prepare = (HERE / "prepare_context.py").read_text()
        self.assertIn(contract_path.name, prepare)

    def test_dockerfile_verifies_context_and_uses_locked_media_runtime(self):
        source = (HERE / "Dockerfile.candidate").read_text()
        self.assertIn("ARG MEDIA_RUNTIME=local/sparkring:r33-media-probe", source)
        self.assertIn(
            "python3 -S /context/verify_context.py --context /context", source
        )
        self.assertIn("COPY --from=verified-context /context/wheelhouse/", source)
        self.assertIn("python3 -m venv /opt/venv", source)
        self.assertNotIn("--system-site-packages", source)
        self.assertNotIn("apt-get", source)
        self.assertIn("native/libnccl.so.2.31.2", source)
        self.assertIn(
            "VLLM_NCCL_SO_PATH=/opt/local-inference/nccl/lib/libnccl.so.2", source
        )
        self.assertNotIn("docker push", source)
        self.assertNotIn("--gpus", source)
        build = (HERE / "build_candidate.sh").read_text()
        self.assertIn("expected_media_id=sha256:a1a72e18", build)
        self.assertIn('docker image inspect "$media_runtime"', build)
        self.assertIn('--build-arg MEDIA_RUNTIME="$media_runtime"', build)

    def test_only_canonical_external_profile_contract_is_packaged(self):
        source = (HERE / "prepare_context.py").read_text()
        self.assertIn('runtime / "sparkring/jovian-r33/profiles"', source)
        self.assertIn('context / "profile-contract"', source)
        self.assertFalse(any((HERE / "profiles").glob("*.json")))
        contract = json.loads(
            (CANONICAL_PROFILES / "profile-contract.json").read_text()
        )
        self.assertEqual(
            set(contract["profiles"]),
            {
                "tp2-dcp1",
                "tp2-dcp1-sparkcache",
                "tp4-dcp1",
                "tp4-dcp1-sparkcache",
                "tp4-dcp4",
                "tp4-dcp4-sparkcache",
            },
        )
        for name in (
            "tp4-dcp1",
            "tp4-dcp1-sparkcache",
            "tp4-dcp4",
            "tp4-dcp4-sparkcache",
        ):
            selected = contract["profiles"][name]
            values = (CANONICAL_PROFILES / selected["template"]).read_text()
            if selected.get("inherits"):
                parent = contract["profiles"][selected["inherits"]]
                values = (CANONICAL_PROFILES / parent["template"]).read_text() + values
            self.assertIn("VLLM_SPARK_TP4_MODE=custom", values)
            self.assertIn("VLLM_SPARK_TP4_VOCAB_MODE=custom", values)

    def test_actual_profile_verifier_passes_in_staged_installed_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "opt/sparkring"
            shutil.copytree(CANONICAL_PROFILES, root / "profile-contract")
            (root / "image").mkdir(parents=True)
            shutil.copy2(HERE / "artifact-lock.json", root / "image/artifact-lock.json")
            transport_source = (
                HERE.parents[2] / "transport_profiles/tp2-rocenante-adaptive"
            )
            shutil.copytree(
                transport_source,
                root / "runtime/transport_profiles/tp2-rocenante-adaptive",
            )
            (root / "runtime/glm53-spark-mtp3-mesh").mkdir(parents=True)
            shutil.copy2(
                HERE.parents[2] / "glm53-spark-mtp3-mesh/pins.json",
                root / "runtime/glm53-spark-mtp3-mesh/pins.json",
            )
            verifier = root / "profile-contract/verify_profile.py"
            for profile in (
                "tp2-dcp1",
                "tp2-dcp1-sparkcache",
                "tp4-dcp1",
                "tp4-dcp1-sparkcache",
                "tp4-dcp4",
                "tp4-dcp4-sparkcache",
            ):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(verifier),
                        "template",
                        "--profile",
                        profile,
                        "--asset-root",
                        str(root / "runtime"),
                    ],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_entrypoint_consumes_custom_external_tp4_profile(self):
        module = load_entrypoint()
        contract = json.loads(
            (CANONICAL_PROFILES / "profile-contract.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile-contract.json").write_text(json.dumps(contract))
            (root / "verify_profile.py").write_text("# fixture")
            with (
                mock.patch.dict(os.environ, tp4_environment(contract), clear=True),
                mock.patch.object(module.subprocess, "run") as run,
            ):
                selected = module.validate_external_profile(root)
            self.assertEqual(selected["transport"], "sparkring-rocenante-mesh")
            run.assert_called_once()

    def test_copied_tp4_adapters_import_in_canonical_custom_mode(self):
        source = (
            HERE.parents[2]
            / "glm53-spark-mtp3-mesh/performance/transport/bundle-source"
        )
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "sircl-python"
            shutil.copytree(source, copied)
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "VLLM_SPARK_TP4_MODE": "custom",
                        "VLLM_SPARK_TP4_VOCAB_MODE": "custom",
                    },
                    clear=True,
                ),
                mock.patch.object(sys, "path", [str(copied), *sys.path]),
            ):
                for name in ("spark_tp4_backend", "spark_tp4_vocab_allgather_backend"):
                    sys.modules.pop(name, None)
                    module = __import__(name)
                    self.assertEqual(module._mode(), "custom")

    def test_embedded_sitecustomize_loads_its_adjacent_sircl_hook(self):
        bundle_source = (
            HERE.parents[2]
            / "glm53-spark-mtp3-mesh/performance/transport/bundle-source"
        )
        self.assertTrue((bundle_source / "sircl_sitecustomize.py").is_file())
        prepare = (HERE / "prepare_context.py").read_text()
        self.assertIn(
            'shutil.copy2(HERE / "sitecustomize.py", '
            'context / "sircl-python/sitecustomize.py")',
            prepare,
        )
        with tempfile.TemporaryDirectory() as directory:
            embedded = Path(directory) / "opt/sparkring/sircl/python"
            embedded.mkdir(parents=True)
            shutil.copy2(HERE / "sitecustomize.py", embedded / "sitecustomize.py")
            (embedded / "sircl_sitecustomize.py").write_text(
                "from pathlib import Path\n"
                "import os\n"
                "Path(os.environ['SIRCL_HOOK_MARKER']).write_text('loaded')\n"
            )
            for name in ("rocenante_vllm_overlay", "rocenante_health_gate"):
                (embedded / f"{name}.py").write_text(
                    "def install():\n    return None\n"
                )
            marker = Path(directory) / "hook-loaded"
            environment = {
                **os.environ,
                "PYTHONPATH": str(embedded),
                "SPARK_TP4_HEALTH_GATE": "1",
                "SIRCL_HOOK_MARKER": str(marker),
            }
            result = subprocess.run(
                [sys.executable, "-c", "print('candidate-started')"],
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(), "loaded")

        dockerfile = (HERE / "Dockerfile.candidate").read_text()
        self.assertIn(
            "ln -sfn /opt/sparkring/sircl/python /opt/spark-sircl",
            dockerfile,
        )

    def test_embedded_sitecustomize_fails_closed_without_sircl_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            embedded = Path(directory) / "opt/sparkring/sircl/python"
            embedded.mkdir(parents=True)
            shutil.copy2(HERE / "sitecustomize.py", embedded / "sitecustomize.py")
            result = subprocess.run(
                [sys.executable, "-c", "print('must-not-start')"],
                env={
                    **os.environ,
                    "PYTHONPATH": str(embedded),
                    "SPARK_TP4_HEALTH_GATE": "1",
                },
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 78)
            self.assertNotIn("must-not-start", result.stdout)
            self.assertIn(
                f"preserved SIRCL sitecustomize is missing: "
                f"{embedded / 'sircl_sitecustomize.py'}",
                result.stderr,
            )

    def test_entrypoint_rejects_tp4_without_managed_renderer(self):
        module = load_entrypoint()
        contract = json.loads(
            (CANONICAL_PROFILES / "profile-contract.json").read_text()
        )
        environment = tp4_environment(contract)
        environment.pop("SPARKRING_MANAGED_MESH_RENDERED")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile-contract.json").write_text(json.dumps(contract))
            (root / "verify_profile.py").write_text("# fixture")
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(module.subprocess, "run"),
            ):
                with self.assertRaisesRegex(RuntimeError, "MANAGED_MESH_RENDERED"):
                    module.validate_external_profile(root)

    def test_entrypoint_rejects_non_instanttensor_r33_loader(self):
        module = load_entrypoint()
        contract = json.loads(
            (CANONICAL_PROFILES / "profile-contract.json").read_text()
        )
        environment = tp4_environment(contract)
        environment["LOAD_FORMAT"] = "fastsafetensors"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile-contract.json").write_text(json.dumps(contract))
            (root / "verify_profile.py").write_text("# fixture")
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(module.subprocess, "run"),
            ):
                with self.assertRaisesRegex(RuntimeError, "LOAD_FORMAT=instanttensor"):
                    module.validate_external_profile(root)

    def test_verifier_environment_cannot_activate_a_transport(self):
        module = load_entrypoint()
        with mock.patch.dict(
            os.environ,
            {
                "PYTHONPATH": "/opt/sparkring/sircl/python",
                "SPARKRING_TRANSPORT_PROFILE": "tp2-rocenante-adaptive",
                "SPARKRING_TRANSPORT_MANIFEST_SHA256": "digest",
                "KEEP_ME": "yes",
            },
            clear=True,
        ):
            environment = module.verification_environment()
        self.assertEqual(environment, {"KEEP_ME": "yes"})

    def test_installed_payload_manifest_is_enforced(self):
        finalize = (HERE / "finalize_lock.py").read_text()
        verify = (HERE / "verify_candidate.py").read_text()
        self.assertIn("installed_python_manifest_sha256", finalize)
        self.assertIn('payload_manifest["files"]', verify)
        self.assertIn("installed Python payload differs", verify)

    def test_media_ancestry_rejects_tag_swap_and_non_descendant(self):
        qualify = load_module("r33_qualify_image", "qualify_image.py")
        expected = "sha256:" + "a" * 64
        media = {"Id": expected, "RootFS": {"Layers": ["layer-1", "layer-2"]}}
        candidate = {"RootFS": {"Layers": ["layer-1", "layer-2", "candidate"]}}
        qualify.validate_media_ancestry(candidate, media, expected)
        with self.assertRaisesRegex(RuntimeError, "tag"):
            qualify.validate_media_ancestry(
                candidate, {**media, "Id": "sha256:" + "b" * 64}, expected
            )
        with self.assertRaisesRegex(RuntimeError, "descend"):
            qualify.validate_media_ancestry(
                {"RootFS": {"Layers": ["other"]}}, media, expected
            )

    def test_receipt_parsers_reject_duplicate_and_ambiguous_identities(self):
        receipts = load_module("r33_validate_receipts", "validate_receipts.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate_json = root / "duplicate.json"
            duplicate_json.write_text('{"sha":"a","sha":"b"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                receipts.load_json(duplicate_json)
            duplicate_kv = root / "duplicate.txt"
            duplicate_kv.write_text("source.commit=abc\nsource.commit=def\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                receipts.load_kv(duplicate_kv)
            traversal = root / "SHA256SUMS"
            traversal.write_text("a" * 64 + "  ../../wheel.whl\n")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                receipts.load_sums(traversal)

    def test_terminal_receipt_finalizer_tracks_all_required_producers(self):
        finalizer = load_module(
            "r33_receipt_finalizer", "finalize_component_receipts.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            build_root = Path(directory) / "build-root"
            staging = Path(directory) / "staging"
            staging.mkdir()
            commands = finalizer.producer_commands(build_root, HERE, staging)
            self.assertEqual(
                [Path(command[1]).name for command in commands],
                [
                    "capture_post_build_sources.py",
                    "validate_flashinfer_resume.py",
                    "capture_flashkda_identity.py",
                ],
            )
            self.assertEqual(
                {
                    Path(command[command.index("--output") + 1]).name
                    for command in commands
                },
                set(finalizer.RECEIPTS),
            )
            for script in (
                "capture_post_build_sources.py",
                "validate_flashinfer_resume.py",
                "capture_flashkda_identity.py",
            ):
                self.assertTrue((HERE / script).is_file())
            documentation = (HERE / "README.md").read_text()
            self.assertIn("finalize_component_receipts.py", documentation)
            self.assertIn("--verify-existing", documentation)

    def test_terminal_receipt_finalizer_reproduces_existing_bytes(self):
        finalizer = load_module(
            "r33_receipt_reproducer", "finalize_component_receipts.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            build_root = Path(directory) / "build-root"
            expected = b'{"fixture":"deterministic"}\n'
            for relative in finalizer.RECEIPTS.values():
                destination = build_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(expected)

            def fake_run(command, check):
                self.assertTrue(check)
                output = Path(command[command.index("--output") + 1])
                output.write_bytes(expected)

            finalizer.finalize(build_root, HERE, True, run=fake_run)

    def test_sparkcache_native_slots_are_exact_and_required(self):
        lock = json.loads((HERE / "artifact-lock.json").read_text())
        pending = {item["name"]: item for item in lock["pending_native_artifacts"]}
        self.assertEqual(set(pending), {"sparkcache-placement", "sparkcache-snapshot"})
        self.assertEqual(
            pending["sparkcache-placement"]["sha256"],
            "d89c9fdae8dc99ae3f7a151cc3dd9e92fdc8fd0b994069fc263027fd4d056c93",
        )
        self.assertEqual(
            pending["sparkcache-snapshot"]["sha256"],
            "7da9e72f096ae679906ba71336c16e7894a247eb5b0d217aaccd115b85058953",
        )
        self.assertTrue(all(item["required"] for item in pending.values()))
        dockerfile = (HERE / "Dockerfile.candidate").read_text()
        self.assertIn("native/libspark_cache_placement.so", dockerfile)
        self.assertIn("native/libspark_cache_snapshot.so", dockerfile)
        prepare = (HERE / "prepare_context.py").read_text()
        self.assertIn("native artifact hash is pending", prepare)
        semantic = (HERE / "validate_receipts.py").read_text()
        self.assertIn("sparkring-r33-sparkcache-native-build/v1", semantic)


if __name__ == "__main__":
    unittest.main()
