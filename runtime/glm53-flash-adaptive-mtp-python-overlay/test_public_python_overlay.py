from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
PINS = HERE / "pins.json"
MANIFEST = HERE / "vllm-python-overlay.json"
DFLASH_PATCH = HERE / "patches/010-dflash-draft-load-config.patch"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = _module("glm53_public_python_overlay_contract", HERE / "overlay_contract.py")
verify = _module("glm53_public_python_overlay_verify", HERE / "verify_image.py")
prepare = _module("glm53_public_python_overlay_prepare", HERE / "prepare_context.py")


def test_overlay_pins_public_base_and_mixed_vllm_provenance() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert pins["status"] == "implemented"
    assert pins["public_base"]["reference"] == (
        "ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:"
        "864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd"
    )
    assert pins["vllm"]["native_commit"] == (
        "da4d7be6c97434f6942292ed8abbf4b32dc44355"
    )
    assert pins["vllm"]["python_commit"] == (
        "0b67266a0f37d6146a8403fb8482403c62f412d5"
    )
    assert pins["b12x"]["commit"] == (
        "b1d541f9e71a35f030d45fae437630fff7507c2a"
    )
    assert pins["b12x"]["tree"] == "c69cdec1c59a08e8e0e549f930fa8abcfb5134ae"
    assert pins["b12x"]["package_version"] == "1.3.0"
    assert pins["b12x"]["commit"] != pins["b12x"]["base_commit"]
    sparkcache = pins["sparkcache"]
    assert sparkcache["commit"] == (
        "bd3eec1c10b259a24c5f335161d9f8be51c887cd"
    )
    assert sparkcache["tree"] == "ab6e25fd1126405a94ce8735a6261f9dd08c0b5f"
    assert sparkcache["source_tree_sha256"] == (
        "490d2c069c2eb755ecb93727aa47c41df38665427228895af0638b8588a049f3"
    )
    assert sparkcache["contract"]["sha256"] == (
        "f36ed14eaf1f97a5dffa94bda8151b1e0fa182afc0d121b757b70bebc6a43811"
    )
    assert sparkcache["cuda_config_schema"] == "canonical-v1"
    assert pins["dependencies"]["torch"] == "2.13.0+cu130"


def test_overlay_manifest_is_the_exact_31_file_python_delta() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = contract.validate_overlay_manifest(manifest)
    assert len(records) == 31
    assert sum(record["target_bytes"] for record in records) == 1_567_308
    assert [record["operation"] for record in records].count("add") == 1
    assert next(record for record in records if record["operation"] == "add")[
        "path"
    ] == "vllm/v1/spec_decode/dynamic/acceptance_length.py"
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == pins["vllm"][
        "overlay_manifest_sha256"
    ]


def test_dflash_loader_patch_binds_exact_0b_preimage_and_postimage() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert pins["vllm"]["runtime_patches"] == [
        {
            "status": "implemented",
            "path": (
                "runtime/glm53-flash-adaptive-mtp-python-overlay/patches/"
                "010-dflash-draft-load-config.patch"
            ),
            "target": "vllm/v1/worker/gpu/spec_decode/dflash/utils.py",
            "sha256": "39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279",
            "preimage_sha256": (
                "2301c8199b73ed893dfbd3ae14ad125816f100b2d2ed034215b1f2d9aa2c23c5"
            ),
            "postimage_sha256": (
                "98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4"
            ),
            "contract": (
                "DFlash passes SpeculativeConfig.draft_load_config to get_model; "
                "None retains the target LoadConfig fallback."
            ),
        }
    ]
    assert prepare.sha256_file(DFLASH_PATCH) == pins["vllm"]["runtime_patches"][
        0
    ]["sha256"]


def test_recurrent_boundary_patch_binds_composed_source_and_exact_tests() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    (record,) = pins["vllm"]["composed_runtime_patches"]
    assert prepare.sha256_file(HERE / "patches" / Path(record["path"]).name) == (
        record["sha256"]
    )
    test_record = record["test_patch"]
    assert prepare.sha256_file(
        HERE / "patches" / Path(test_record["path"]).name
    ) == test_record["sha256"]
    assert {target["path"] for target in record["targets"]} == {
        "vllm/v1/core/kv_cache_manager.py",
        "vllm/v1/core/sched/output.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/core/single_type_kv_cache_manager.py",
    }


def test_recurrent_boundary_source_validator_is_fail_closed(tmp_path: Path) -> None:
    sources = {
        "vllm/v1/core/sched/output.py": (
            "recurrent_boundary_blocks: "
            "dict[str, list[tuple[int, int, int]]] | None = None\n"
        ),
        "vllm/v1/core/sched/scheduler.py": (
            "recurrent_boundary_blocks=pending_recurrent_boundary_blocks\n"
            'getattr(self.connector, "supports_recurrent_boundary_blocks", False)\n'
            "kv_transfer_config.is_kv_consumer\n"
            "multiple_inflight_batches and (\n"
        ),
        "vllm/v1/core/single_type_kv_cache_manager.py": (
            "(request.num_prompt_tokens - 1) // self.block_pool.hash_block_size\n"
            "block.block_hash_num_tokens != replay_boundary\n"
            "get_group_id(block.block_hash) != self.kv_cache_group_id\n"
        ),
        "vllm/v1/core/kv_cache_manager.py": "self._pin_recurrent_boundary\n",
    }
    for relative, source in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    contract.validate_recurrent_boundary_sources(tmp_path)

    manager = tmp_path / "vllm/v1/core/single_type_kv_cache_manager.py"
    manager.write_text(
        sources["vllm/v1/core/single_type_kv_cache_manager.py"].replace(
            "request.num_prompt_tokens - 1", "request.num_prompt_tokens"
        ),
        encoding="utf-8",
    )
    with pytest.raises(contract.ContractError, match="prompt-minus-one"):
        contract.validate_recurrent_boundary_sources(tmp_path)


def test_dflash_loader_contract_honors_explicit_draft_load_config() -> None:
    contract.validate_dflash_loader_source(
        """
def load_dflash_model(target_model, vllm_config):
    speculative_config = vllm_config.speculative_config
    return get_model(
        vllm_config=vllm_config,
        model_config=speculative_config.draft_model_config,
        load_config=speculative_config.draft_load_config,
    )
"""
    )


def test_dflash_loader_contract_rejects_an_ignored_draft_load_config() -> None:
    with pytest.raises(contract.ContractError, match="draft_load_config"):
        contract.validate_dflash_loader_source(
            """
def load_dflash_model(target_model, vllm_config):
    speculative_config = vllm_config.speculative_config
    return get_model(
        vllm_config=vllm_config,
        model_config=speculative_config.draft_model_config,
    )
"""
        )


def test_missing_draft_load_config_uses_the_target_load_config_fallback() -> None:
    contract.validate_optional_load_config_fallback(
        """
def get_model(*, vllm_config, load_config=None):
    loader = get_model_loader(load_config or vllm_config.load_config)
    return loader.load_model()
"""
    )

    with pytest.raises(contract.ContractError, match="fall back"):
        contract.validate_optional_load_config_fallback(
            """
def get_model(*, vllm_config, load_config=None):
    loader = get_model_loader(vllm_config.load_config)
    return loader.load_model()
"""
        )


def test_documented_dflash_config_separates_target_and_draft_loaders() -> None:
    guide = (HERE / "README.md").read_text(encoding="utf-8")
    match = re.search(r"--speculative-config '([^']+)'", guide)
    assert match is not None
    speculative = json.loads(match.group(1))
    assert "--load-format fastsafetensors" in guide
    assert speculative["method"] == "dflash"
    assert speculative["model"] == "/mtp-draft"
    assert speculative["num_speculative_tokens"] == 7
    assert speculative["draft_load_config"] == {"load_format": "safetensors"}


def test_native_build_inputs_are_identical_git_objects() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert pins["vllm"]["native_source_objects"] == {
        "csrc": "9ada29088768f1bc08dadd2eed3c9738eb9ac8a1",
        "cmake": "5e5bbdbe1c1b3a479656d8d6a41cc32a1982c43d",
        "rust": "85c3cd52db223217d45377d3f7f884e756641de3",
        "CMakeLists.txt": "bb0f51b43ef4e1c57918b551b8cf213f9059b601",
        "setup.py": "ae64a13daa0f1facc255afcdb4ffcad264776b98",
        "pyproject.toml": "0766645fc7481da0ec439208128b838b3348d94c",
        "requirements": "d6e1c8e13cd4c4358ab422e3ef006d3f9f23e18b",
        "docker/Dockerfile": "6e20f6eab482782ed90a05d91b669b59641eaa46",
    }


def test_stale_b12x_caps_are_rejected_before_a_gpu_launch() -> None:
    @dataclass
    class StaleCaps:
        device: str

    def stale_bind(plan, *, mixed_qkv):
        return plan, mixed_qkv

    with pytest.raises(contract.ContractError, match="omits required field"):
        contract.validate_b12x_surface(
            StaleCaps,
            stale_bind,
            SimpleNamespace(plan=object()),
            required_field="kda_metadata_validation",
        )


def test_live_tensor_b12x_surface_requires_every_binding_input() -> None:
    @dataclass
    class Caps:
        kda_metadata_validation: str = "transactional"

    def incomplete_bind(plan, *, mixed_qkv, output):
        return plan, mixed_qkv, output

    with pytest.raises(contract.ContractError, match="omits live tensor parameters"):
        contract.validate_b12x_surface(
            Caps,
            incomplete_bind,
            SimpleNamespace(plan=object()),
            required_field="kda_metadata_validation",
        )


def test_elf_manifest_detects_retained_native_changes(tmp_path: Path) -> None:
    package = tmp_path / "vllm"
    package.mkdir()
    first = package / "_C_stable_libtorch.abi3.so"
    second = package / "vllm-rs"
    first.write_bytes(b"\x7fELFfirst")
    second.write_bytes(b"\x7fELFsecond")
    original = contract.elf_manifest(package)
    assert [record["path"] for record in original] == [
        "_C_stable_libtorch.abi3.so",
        "vllm-rs",
    ]
    second.write_bytes(b"\x7fELFchanged")
    assert contract.canonical_sha256(contract.elf_manifest(package)) != (
        contract.canonical_sha256(original)
    )


def test_containerfile_reuses_vllm_and_nccl_native_artifacts() -> None:
    recipe = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert "FROM ${PUBLIC_BASE}" in recipe
    assert "COPY bundle/vllm-overlay/ /usr/local/lib/python3.12/dist-packages/" in recipe
    assert "--reinstall --no-deps" in recipe
    assert "record-base" in recipe and "verify-composed" in recipe
    assert "native-elf-manifest-sha256" in recipe
    assert "native-dispatch-manifest-sha256" in recipe
    assert "setup.py bdist_wheel" not in recipe
    assert "make -C /build/nccl" not in recipe

    source_copy = recipe.index(
        "COPY bundle/sources/sparkcache/sparkcache /opt/sparkcache-src/sparkcache"
    )
    source_verification = recipe.index('printf \'sparkcache_source_sha256=%s')
    source_receipt = recipe.index("sparkcache-source-tree.sha256")
    cuda_placement_copy = recipe.index(
        "COPY --from=sparkcache-cuda-placement", source_verification
    )
    assert source_copy < source_verification < source_receipt < cuda_placement_copy


def test_image_verifier_reads_the_clean_sparkcache_source_receipt() -> None:
    verifier = (HERE / "verify_image.py").read_text(encoding="utf-8")
    assert "sparkcache-source-tree.sha256" in verifier
    assert 'artifacts["sparkcache_source_tree_sha256"]' in verifier


def test_image_verifier_accepts_dflash_member_and_four_recurrent_postimages() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    report = {
        "vllm_runtime_patches": [
            {
                "path": record["target"],
                "sha256": record["postimage_sha256"],
            }
            for record in pins["vllm"]["runtime_patches"]
        ]
        + [
            {
                "path": target["path"],
                "sha256": target["postimage_sha256"],
            }
            for record in pins["vllm"]["composed_runtime_patches"]
            for target in record["targets"]
        ]
    }
    assert len(report["vllm_runtime_patches"]) == 5
    verify.verify_runtime_patch_report(report, pins)

    missing_dflash = {
        "vllm_runtime_patches": report["vllm_runtime_patches"][1:]
    }
    with pytest.raises(verify.VerifyError, match="exact DFlash"):
        verify.verify_runtime_patch_report(missing_dflash, pins)

    wrong_recurrent = json.loads(json.dumps(report))
    wrong_recurrent["vllm_runtime_patches"][-1]["sha256"] = "0" * 64
    with pytest.raises(verify.VerifyError, match="exact recurrent"):
        verify.verify_runtime_patch_report(wrong_recurrent, pins)


def test_build_prepares_context_below_the_temporary_workspace() -> None:
    script = (HERE / "build-image.sh").read_text(encoding="utf-8")
    preparer = (HERE / "prepare_context.py").read_text(encoding="utf-8")
    assert 'workspace="$(mktemp -d)"' in script
    assert 'context="${workspace}/context"' in script
    assert 'rm -rf -- "${workspace}"' in script
    assert '"core.longpaths", "true"' in preparer


def test_sparkcache_patches_and_contract_run_after_the_python_overlay() -> None:
    recipe = (HERE / "Containerfile").read_text(encoding="utf-8")
    overlay = recipe.index("COPY bundle/vllm-overlay/")
    patch_020 = recipe.index("020-sparkcache-vmm-exemption.patch")
    dflash_patch = recipe.index("010-dflash-draft-load-config.patch")
    patch_030 = recipe.index("030-sparkcache-hma-load-failure.patch")
    patch_040 = recipe.index("040-sparkcache-shared-prefix-lease.patch")
    patch_041 = recipe.index("041-sparkcache-shared-prefix-attach.patch")
    recurrent = recipe.index("011-recurrent-boundary-contract.patch", patch_041)
    final_lease = recipe.index("verify_lease_contract.py", recurrent)
    assert recipe.count("verify_lease_contract.py") == 1
    assert (
        overlay
        < dflash_patch
        < patch_020
        < patch_030
        < patch_040
        < patch_041
        < recurrent
        < final_lease
    )


def test_pr35_lease_verifier_runs_only_after_recurrent_symbols_exist() -> None:
    recipe = (HERE / "Containerfile").read_text(encoding="utf-8")
    recurrent_patch = (HERE / "patches/011-recurrent-boundary-contract.patch").read_text(
        encoding="utf-8"
    )
    recurrent_apply = recipe.index(
        'patch --batch --forward -p1 -d "${root}" < "${recurrent_patch}"'
    )
    lease_verify = recipe.index("verify_lease_contract.py")
    assert recurrent_apply < lease_verify
    contract = json.loads(PINS.read_text(encoding="utf-8"))["sparkcache"]["contract"]
    assert contract["sha256"] == (
        "f36ed14eaf1f97a5dffa94bda8151b1e0fa182afc0d121b757b70bebc6a43811"
    )
    for definition in (
        "def take_recurrent_boundary_blocks(",
        "recurrent_boundary_blocks:",
        "def take_pending_aligned_recurrent_boundaries(",
    ):
        assert definition in recurrent_patch


def test_context_preparer_runs_the_real_lease_verifier_on_composed_source() -> None:
    source = inspect.getsource(prepare.verify_composed_runtime_patches)
    recurrent_apply = source.index('run(("git", "-C", str(source), "apply", str(patch)))')
    exact_postimage = source.index('target_record["postimage_sha256"]')
    verifier = source.index("verify_lease_contract.py")
    recurrent_reverse = source.index("for patch in reversed(applied_composed)")
    assert recurrent_apply < exact_postimage < verifier < recurrent_reverse
    assert '"--vllm-root"' in source
    assert "str(contract)" in source


def test_output_labels_do_not_claim_a_source_built_0b_wheel() -> None:
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    labels = verify.expected_output_labels(pins)
    assert labels["org.jovian.vllm.commit"] == pins["vllm"]["native_commit"]
    assert labels["org.sparkring.vllm.python.commit"] == pins["vllm"][
        "python_commit"
    ]
    assert labels["org.jovian.vllm.commit"] != labels[
        "org.sparkring.vllm.python.commit"
    ]
    assert labels["org.sparkring.vllm.dflash-draft-loader-patch-sha256"] == (
        "39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279"
    )
    assert labels["org.sparkring.vllm.dflash-draft-loader-postimage-sha256"] == (
        "98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4"
    )
    assert labels["org.sparkring.vllm.recurrent-boundary-patch-sha256"] == (
        "5a6561a5bbab990dcd03bfd6a485ea26c3b5a578c2fd61b76305767b16dbfba0"
    )
