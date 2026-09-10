import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
CANONICAL_PROFILES = HERE.parent / "profiles"


def load_entrypoint():
    spec = importlib.util.spec_from_file_location("r33_candidate_entrypoint", HERE / "entrypoint.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def tp4_environment(contract: dict) -> dict[str, str]:
    return {
        **contract["common_environment"],
        "SOURCE_IMAGE_PROFILE": "tp4-dcp1",
        "SPARKRING_PROFILE_MODE": "custom",
        "SPARKRING_MANAGED_MESH_RENDERED": "1",
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
    def test_artifact_lock_pins_critical_inputs_and_excludes_bad_audio(self):
        lock = json.loads((HERE / "artifact-lock.json").read_text())
        self.assertEqual(lock["foundation"]["cuda"], "13.3")
        self.assertEqual(lock["foundation"]["image_id"], "sha256:6704db5df61d1110afaba538554abb024c85edb8f37a98e4e631166a10af3217")
        self.assertEqual(lock["media_runtime"]["image_id"], "sha256:a1a72e18ad49d99f6194a2585bfdc5f32d79180cdf2cd015c0d0d451479d6a42")
        artifacts = {item["name"]: item for item in lock["artifacts"]}
        self.assertEqual(artifacts["vllm"]["sha256"], "e8661bc7890cda16762d79102814bf4ff25842d51ad902d672467674d5c02fa4")
        self.assertEqual(artifacts["nccl-2.31.2-sparkring-routing"]["sha256"], "84a4b8d83fb5fa1f0d640d311ad38b45140672dae9889775fe1e4a3990479e47")
        self.assertEqual(artifacts["sircl"]["sha256"], "bea00f2ba6051c2c0bcd2853aae894672aa7f1fe5a1d905edaa9120aabf74246")
        self.assertIn("+cu133-", artifacts["torchaudio"]["source"])
        selected = {Path(item["source"]).name for item in lock["artifacts"] if item["install"]}
        for forbidden in lock["forbidden_inputs"]:
            self.assertNotIn(Path(forbidden).name, selected)
        pending = {item["name"]: item for item in lock["pending_artifacts"]}
        self.assertEqual(set(pending), {"flashinfer-python", "flashinfer-jit-cache"})
        self.assertTrue(all(item["required"] for item in pending.values()))
        contract = json.loads((CANONICAL_PROFILES / "profile-contract.json").read_text())
        self.assertEqual(contract["image"]["required_sources"], lock["source_identities"])
        self.assertEqual(
            contract["image"]["artifact_lock_sha256"],
            __import__("hashlib").sha256((HERE / "artifact-lock.json").read_bytes()).hexdigest(),
        )

    def test_dockerfile_verifies_context_and_uses_locked_media_runtime(self):
        source = (HERE / "Dockerfile.candidate").read_text()
        self.assertIn("ARG MEDIA_RUNTIME=local/sparkring:r33-media-probe", source)
        self.assertIn("python3 -S /context/verify_context.py --context /context", source)
        self.assertIn("COPY --from=verified-context /context/wheelhouse/", source)
        self.assertIn("python3 -m venv /opt/venv", source)
        self.assertNotIn("--system-site-packages", source)
        self.assertNotIn("apt-get", source)
        self.assertIn("native/libnccl.so.2.31.2", source)
        self.assertIn("VLLM_NCCL_SO_PATH=/opt/local-inference/nccl/lib/libnccl.so.2", source)
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
        contract = json.loads((CANONICAL_PROFILES / "profile-contract.json").read_text())
        self.assertEqual(set(contract["profiles"]), {"tp2-dcp1", "tp4-dcp1", "tp4-dcp1-sparkcache"})
        for name in ("tp4-dcp1", "tp4-dcp1-sparkcache"):
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
            transport_source = HERE.parents[2] / "transport_profiles/tp2-rocenante-adaptive"
            shutil.copytree(transport_source, root / "runtime/transport_profiles/tp2-rocenante-adaptive")
            (root / "runtime/glm53-spark-mtp3-mesh").mkdir(parents=True)
            shutil.copy2(HERE.parents[2] / "glm53-spark-mtp3-mesh/pins.json", root / "runtime/glm53-spark-mtp3-mesh/pins.json")
            verifier = root / "profile-contract/verify_profile.py"
            for profile in ("tp2-dcp1", "tp4-dcp1", "tp4-dcp1-sparkcache"):
                result = subprocess.run(
                    [sys.executable, str(verifier), "template", "--profile", profile,
                     "--asset-root", str(root / "runtime")],
                    text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_entrypoint_consumes_custom_external_tp4_profile(self):
        module = load_entrypoint()
        contract = json.loads((CANONICAL_PROFILES / "profile-contract.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile-contract.json").write_text(json.dumps(contract))
            (root / "verify_profile.py").write_text("# fixture")
            with mock.patch.dict(os.environ, tp4_environment(contract), clear=True), mock.patch.object(module.subprocess, "run") as run:
                selected = module.validate_external_profile(root)
            self.assertEqual(selected["transport"], "sparkring-rocenante-mesh")
            run.assert_called_once()

    def test_copied_tp4_adapters_import_in_canonical_custom_mode(self):
        source = HERE.parents[2] / "glm53-spark-mtp3-mesh/performance/transport/bundle-source"
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "sircl-python"
            shutil.copytree(source, copied)
            with mock.patch.dict(os.environ, {
                "VLLM_SPARK_TP4_MODE": "custom",
                "VLLM_SPARK_TP4_VOCAB_MODE": "custom",
            }, clear=True), mock.patch.object(sys, "path", [str(copied), *sys.path]):
                for name in ("spark_tp4_backend", "spark_tp4_vocab_allgather_backend"):
                    sys.modules.pop(name, None)
                    module = __import__(name)
                    self.assertEqual(module._mode(), "custom")

    def test_entrypoint_rejects_tp4_without_managed_renderer(self):
        module = load_entrypoint()
        contract = json.loads((CANONICAL_PROFILES / "profile-contract.json").read_text())
        environment = tp4_environment(contract)
        environment.pop("SPARKRING_MANAGED_MESH_RENDERED")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "profile-contract.json").write_text(json.dumps(contract))
            (root / "verify_profile.py").write_text("# fixture")
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(module.subprocess, "run"):
                with self.assertRaisesRegex(RuntimeError, "MANAGED_MESH_RENDERED"):
                    module.validate_external_profile(root)

    def test_verifier_environment_cannot_activate_a_transport(self):
        module = load_entrypoint()
        with mock.patch.dict(os.environ, {
            "PYTHONPATH": "/opt/sparkring/sircl/python",
            "SPARKRING_TRANSPORT_PROFILE": "tp2-rocenante-adaptive",
            "SPARKRING_TRANSPORT_MANIFEST_SHA256": "digest",
            "KEEP_ME": "yes",
        }, clear=True):
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
            qualify.validate_media_ancestry(candidate, {**media, "Id": "sha256:" + "b" * 64}, expected)
        with self.assertRaisesRegex(RuntimeError, "descend"):
            qualify.validate_media_ancestry({"RootFS": {"Layers": ["other"]}}, media, expected)

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
        finalizer = load_module("r33_receipt_finalizer", "finalize_component_receipts.py")
        with tempfile.TemporaryDirectory() as directory:
            build_root = Path(directory) / "build-root"
            staging = Path(directory) / "staging"
            staging.mkdir()
            commands = finalizer.producer_commands(build_root, HERE, staging)
            self.assertEqual(
                [Path(command[1]).name for command in commands],
                ["capture_post_build_sources.py", "validate_flashinfer_resume.py", "capture_flashkda_identity.py"],
            )
            self.assertEqual(
                {Path(command[command.index("--output") + 1]).name for command in commands},
                set(finalizer.RECEIPTS),
            )
            for script in ("capture_post_build_sources.py", "validate_flashinfer_resume.py", "capture_flashkda_identity.py"):
                self.assertTrue((HERE / script).is_file())
            documentation = (HERE / "README.md").read_text()
            self.assertIn("finalize_component_receipts.py", documentation)
            self.assertIn("--verify-existing", documentation)

    def test_terminal_receipt_finalizer_reproduces_existing_bytes(self):
        finalizer = load_module("r33_receipt_reproducer", "finalize_component_receipts.py")
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
        self.assertEqual(pending["sparkcache-placement"]["sha256"], "d89c9fdae8dc99ae3f7a151cc3dd9e92fdc8fd0b994069fc263027fd4d056c93")
        self.assertEqual(pending["sparkcache-snapshot"]["sha256"], "7da9e72f096ae679906ba71336c16e7894a247eb5b0d217aaccd115b85058953")
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
