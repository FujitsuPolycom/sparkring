from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts.json"
ENV_TEMPLATE = HERE / "runtime.env.example"
LAUNCHER = HERE / "launch-rank.sh"
QUICKSTART = ROOT / "docs/GLM53_JJ_R7_GB10_TP4_QUICKSTART.md"
RECEIPT = (
    ROOT
    / "performance/receipts/glm53-flash/jj-r7-gb10-tp4-smoke-20260830"
    / "validation.json"
)
RECORD = (
    ROOT
    / "performance/records/glm53-flash/jj-r7-gb10-tp4-smoke-20260830.md"
)

BASE_DIGEST = "sha256:11922064b342de1fc98f0ef85e6648843c8fa7eb3e4f4353c6ad82d6e457dde0"
BASE_ID = "sha256:8cff7a250f16bfb89df23d29f9233dbb1c700a780dcec86a64c535a71aee88be"
SPARKCACHE_DIGEST = "sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5"
SPARKCACHE_ID = "sha256:6af83baabb239db6b05e379401daf93c8f51694f81483c2781f6014c30e31db4"
VLLM_COMMIT = "331573d20bd47e78327ed8d8b4d2e6d350bbb1ab"
B12X_COMMIT = "6255090a03b12c3f7d552102a02fac0b542fb8c9"
SPARKCACHE_COMMIT = "dcbe040d339f243621163b0c6ed4ce96462403d8"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _bash_path(path: Path) -> str:
    if path.drive:
        return (
            f"/mnt/{path.drive[0].lower()}/"
            + path.as_posix().split(":/", maxsplit=1)[1]
        )
    return path.as_posix()


def test_artifacts_bind_public_images_sources_models_and_labels() -> None:
    artifacts = _json(ARTIFACTS)
    assert artifacts["schema"] == "sparkring-glm53-jj-r7-gb10-artifacts/v1"
    assert artifacts["release_id"] == "glm53-jj-r7-gb10-public-r2"
    assert artifacts["status"] == "implemented-and-tp4-smoke-verified"
    base = artifacts["images"]["base"]
    cache = artifacts["images"]["sparkcache"]
    assert base["registry_reference"].endswith("@" + BASE_DIGEST)
    assert base["manifest_digest"] == BASE_DIGEST
    assert base["config_digest"] == BASE_ID
    assert base["local_image_id"] == BASE_ID
    assert cache["registry_reference"].endswith("@" + SPARKCACHE_DIGEST)
    assert cache["manifest_digest"] == SPARKCACHE_DIGEST
    assert cache["config_digest"] == SPARKCACHE_ID
    assert cache["local_image_id"] == SPARKCACHE_ID
    assert cache["parent_local_image_id"] == BASE_ID
    assert base["entrypoint"] == cache["entrypoint"] == ["vllm", "serve"]
    assert base["required_labels"]["org.sparkring.vllm.sparkcache-composition"] == VLLM_COMMIT
    assert cache["required_labels"]["org.sparkcache.commit"] == SPARKCACHE_COMMIT
    assert cache["required_labels"]["org.sparkcache.publication-schema"] == "page-tail-cow-v1"
    assert base["required_labels"]["org.opencontainers.image.revision"] == (
        "ca91fa72a4cf7e1edaad9875a1a99ab4f71c49af"
    )
    assert cache["required_labels"]["org.opencontainers.image.revision"] == (
        SPARKCACHE_COMMIT
    )
    sources = artifacts["sources"]
    assert sources["sparkring"] == {
        "repository": "https://github.com/FujitsuPolycom/sparkring.git",
        "commit": "ca91fa72a4cf7e1edaad9875a1a99ab4f71c49af",
        "tree": "0fef36259b57402343a2cff94f46479dc758341b",
        "scope": "Base image composition and the OCI revision label.",
    }
    assert sources["vllm"]["upstream_repository"] == (
        "https://github.com/local-inference-lab/vllm.git"
    )
    assert sources["vllm"]["composition_repository"] == (
        "https://github.com/FujitsuPolycom/vllm.git"
    )
    assert sources["vllm"]["composition_commit"] == VLLM_COMMIT
    assert sources["vllm"]["composition_tree"] == "927f52a0085bcecfd2ba679e5abebe1a62623daf"
    assert sources["b12x"]["commit"] == B12X_COMMIT
    assert sources["b12x"]["tree"] == "0bb58d0dcc10e29e00ff9850c0d719fca1aba5ad"
    assert sources["sparkcache"]["commit"] == SPARKCACHE_COMMIT
    assert sources["sparkcache"]["source_digest_sha256"] == (
        "9cf50afd04e385975a487a0129645bd294e0395012424995569a9b50a7c447f1"
    )
    assert artifacts["models"]["draft"]["dtype"] == "bfloat16"
    inherited = artifacts["inherited_lower_layer_labels"]
    assert inherited["org.glm53.dflash2.checkpoint-revision"][
        "active_mounted_draft_identity"
    ] is False
    native = artifacts["native_extension_provenance"]
    assert native["compiled_build_environment_commit"] == (
        "3633d61c3c7b04bb4d598cadbdc342f3be40482d"
    )
    assert native["intermediate_parent_source_label"] == (
        "da4d7be6c97434f6942292ed8abbf4b32dc44355"
    )
    assert native["active_python_composition_commit"] == VLLM_COMMIT
    assert native["verification"]["status"] == "verified"
    assert inherited["org.glm53.dflash2.mxfp8-quant-plumbing"][
        "active_mounted_draft_identity"
    ] is False
    assert artifacts["image_delta"]["focused_tests"] == 15


def test_receipt_records_exact_c4_smoke_boundaries() -> None:
    receipt = _json(RECEIPT)
    status = receipt["status"]
    assert status["base_c4_semantics"] == "tp4-smoke-verified"
    assert status["sparkcache_restart_restore"] == "tp4-smoke-verified"
    assert status["general_qualification"] == "unsupported-by-this-receipt"
    measurement = receipt["measurement"]
    assert measurement["base"]["expected_and_observed_outputs"] == [
        "red",
        "blue",
        "green",
        "black",
    ]
    assert measurement["base"]["client_result_sha256"] == (
        "48361bf399a7b85d10c5ab8f768f81d4d91fb56a17d8192c09cd4e49b5775c0f"
    )
    publication = measurement["sparkcache_publication"]
    assert publication["client_result_sha256"] == (
        "edb9c082fc6fe1b99004fa4c04d9e4b53d0525fe5410313ba13f18f2dc09ffbc"
    )
    assert publication["manifest_count_per_rank"] == 4
    assert publication["root_bytes_per_rank"] == 605690671
    restore = measurement["sparkcache_restart_restore"]
    assert restore["client_result_sha256"] == (
        "02a0c0fafa95294008cd1b9a8a6269dabc0d161c10c307bc1922f1b9aa20c100"
    )
    assert restore["external_restore_count"] == 4
    assert restore["external_hit_ratio"] == 1.0
    assert restore["client_elapsed_seconds"] == {
        "minimum": 0.561595,
        "maximum": 1.582937,
    }
    assert measurement["shared_base"]["status"] == "not-measured"


def test_environment_exposes_common_settings_and_matches_artifact_defaults() -> None:
    template = ENV_TEMPLATE.read_text(encoding="utf-8")
    values = dict(
        re.findall(r"^([A-Z][A-Z0-9_]*)=['\"]?([^'\"\n]*)['\"]?$", template, re.MULTILINE)
    )
    required = {
        "HOST_IP",
        "MASTER_ADDR",
        "TARGET_MODEL_HOST_PATH",
        "DFLASH_MODEL_HOST_PATH",
        "CACHE_HOST_ROOT",
        "IMAGE_VARIANT",
        "BASE_IMAGE_REF",
        "SPARKCACHE_IMAGE_REF",
        "CONTAINER_PREFIX",
        "PORT",
        "MAX_MODEL_LEN",
        "MAX_NUM_SEQS",
        "MAX_NUM_BATCHED_TOKENS",
        "KV_CACHE_MEMORY_BYTES",
        "GPU_MEMORY_UTILIZATION",
        "KV_CACHE_DTYPE",
        "NUM_SPECULATIVE_TOKENS",
        "ATTENTION_BACKEND",
        "MOE_BACKEND",
        "LINEAR_BACKEND",
        "BASE_CACHE_NAMESPACE",
        "SPARKCACHE_CACHE_NAMESPACE",
        "SPARKCACHE_MAX_BYTES",
        "SPARKCACHE_TTL_SECONDS",
        "SPARKCACHE_LOAD_THREADS",
        "SPARKCACHE_MAX_PENDING_RESTORES",
        "SPARKCACHE_CUDA_RESTORE_IO_WORKERS",
        "SPARKCACHE_CUDA_ARENA_BYTES",
        "SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "OMP_NUM_THREADS",
    }
    assert required <= values.keys()
    assert values["BASE_IMAGE_REF"].endswith("@" + BASE_DIGEST)
    assert values["SPARKCACHE_IMAGE_REF"].endswith("@" + SPARKCACHE_DIGEST)
    assert values["BASE_CACHE_NAMESPACE"] != values["SPARKCACHE_CACHE_NAMESPACE"]
    artifacts = _json(ARTIFACTS)["serving_defaults"]
    assert int(values["MAX_MODEL_LEN"]) == artifacts["max_model_len"]
    assert int(values["MAX_NUM_SEQS"]) == artifacts["max_num_seqs"]
    assert int(values["MAX_NUM_BATCHED_TOKENS"]) == artifacts["max_num_batched_tokens"]
    assert values["SERVED_MODEL_NAME"] == artifacts["served_model_name"]


def test_launcher_selects_both_images_without_duplicate_serve(tmp_path: Path) -> None:
    subprocess.run(
        ["bash", "-n", LAUNCHER.relative_to(ROOT).as_posix()],
        check=True,
        cwd=ROOT,
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_bytes(
        f"""#!/bin/sh
if [ "$1" = image ] && [ "$2" = inspect ]; then
  case "$*" in
    *sparkring-glm53-runtime*) printf '%s\\n' '{BASE_ID}' ;;
    *sparkring-glm53-sparkcache*) printf '%s\\n' '{SPARKCACHE_ID}' ;;
    *) exit 96 ;;
  esac
elif [ "$1" = container ] && [ "$2" = inspect ]; then
  exit 1
elif [ "$1" = run ]; then
  printf '%s\\n' "$@" > "$CAPTURE_PATH"
else
  exit 97
fi
""".encode("utf-8")
    )
    os.chmod(docker, 0o755)
    sha256sum = fake_bin / "sha256sum"
    sha256sum.write_bytes(
        b"""#!/bin/sh
case "$2" in
  */target/config.json) hash=676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996 ;;
  */target/model.safetensors.index.json) hash=0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb ;;
  */draft/config.json) hash=c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573 ;;
  */draft/model.safetensors) hash=b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b ;;
  *) exit 95 ;;
esac
printf '%s  %s\n' "$hash" "$2"
"""
    )
    os.chmod(sha256sum, 0o755)
    directories = {name: tmp_path / name for name in ("target", "draft", "cache")}
    for directory in directories.values():
        directory.mkdir()
    for path in (
        directories["target"] / "config.json",
        directories["target"] / "model.safetensors.index.json",
        directories["draft"] / "config.json",
        directories["draft"] / "model.safetensors",
    ):
        path.write_bytes(b"test fixture")

    for variant in ("base", "sparkcache"):
        capture = tmp_path / f"{variant}-arguments.txt"
        config = tmp_path / f"{variant}.env"
        config.write_bytes(
            "\n".join(
                (
                    "HOST_IP=rank0.example.net",
                    "MASTER_ADDR=rank0.example.net",
                    f"TARGET_MODEL_HOST_PATH={_bash_path(directories['target'])}",
                    f"DFLASH_MODEL_HOST_PATH={_bash_path(directories['draft'])}",
                    f"CACHE_HOST_ROOT={_bash_path(directories['cache'])}",
                    f"PATH={_bash_path(fake_bin)}:$PATH",
                    f"export CAPTURE_PATH={_bash_path(capture)}",
                    f"IMAGE_VARIANT={variant}",
                )
            ).encode("utf-8")
        )
        result = subprocess.run(
            ["bash", LAUNCHER.relative_to(ROOT).as_posix(), "0", _bash_path(config)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        arguments = capture.read_text(encoding="utf-8").splitlines()
        image = next(value for value in arguments if value.startswith("ghcr.io/"))
        assert arguments[arguments.index(image) + 1] == "/models/target"
        assert "serve" not in arguments
        assert "org.sparkring.launch.status=implemented-tp4-smoke-verified" in arguments
        if variant == "base":
            assert "--kv-transfer-config" not in arguments
        else:
            encoded = arguments[arguments.index("--kv-transfer-config") + 1]
            connector = json.loads(encoded)
            extra = connector["kv_connector_extra_config"]
            assert extra["spark_cache_publication_schema"] == "page-tail-cow-v1"
            assert extra["spark_cache_load_threads"] == 8


def test_launcher_verifies_identity_files_before_starting_docker() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    expected = {
        "676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996",
        "0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb",
        "c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573",
        "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b",
    }
    assert expected <= set(re.findall(r"[0-9a-f]{64}", launcher))
    assert "verify_file_sha256" in launcher
    assert "identity mismatch" in launcher
    assert launcher.index("verify_file_sha256") < launcher.index("exec docker run")


def test_cudagraph_capture_list_uses_the_exposed_maximum() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert 'maximum = int(os.environ["MAX_CUDAGRAPH_CAPTURE_SIZE"])' in launcher
    assert "capture_sizes.append(maximum)" in launcher
    assert '"cudagraph_capture_sizes": capture_sizes' in launcher


def test_canonical_glm_indexes_route_to_published_images_without_stale_identities() -> None:
    canonical = (
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "runtime/README.md",
        ROOT / "docs/GLM53_FLASH_QUICKSTARTS.md",
        ROOT / "docs/profiles/README.md",
        ROOT / "docs/RESULTS.md",
        ROOT / "recipes/sparkcache/README.md",
    )
    stale = (
        "cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943",
        "864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd",
        "becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818",
        "35b58a7bf414059c65b8f74e4e4b17ee6a81b7008e1bffbc9bd298b5e08c739e",
        "0b67266a0f37d6146a8403fb8482403c62f412d5",
        "b1d541f9e71a35f030d45fae437630fff7507c2a",
    )
    for path in canonical:
        text = path.read_text(encoding="utf-8")
        assert "GLM53_JJ_R7_GB10_TP4_QUICKSTART.md" in text or (
            "glm53-flash-jj-r7-gb10" in text
        )
        assert not any(identity in text for identity in stale), path

    config_index = (ROOT / "scripts/config/README.md").read_text(encoding="utf-8")
    assert "runtime/glm53-flash-jj-r7-gb10/runtime.env.example" in config_index
    assert "Historical GLM-5.3 profile templates" in config_index


def test_prior_glm_quickstarts_identify_their_historical_scope() -> None:
    names = (
        "GLM53_FLASH_DFLASH2_BF16_TP4_QUICKSTART.md",
        "GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md",
        "GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md",
        "GLM53_SPLIT_PAGE_SPARKCACHE_TP4_QUICKSTART.md",
        "GLM53_B12X_KDA_ADAPTIVE_MTP_SPARKCACHE_TP4_QUICKSTART.md",
        "GLM53_E10536A_SPARKCACHE_TP4_QUICKSTART.md",
    )
    for name in names:
        opening = "\n".join(
            (ROOT / "docs" / name).read_text(encoding="utf-8").splitlines()[:9]
        )
        assert "Historical" in opening
        assert "GLM53_JJ_R7_GB10_TP4_QUICKSTART.md" in opening


def test_public_docs_state_evidence_and_unqualified_boundaries() -> None:
    quickstart = QUICKSTART.read_text(encoding="utf-8")
    record = RECORD.read_text(encoding="utf-8")
    runtime = (HERE / "README.md").read_text(encoding="utf-8")
    for text in (quickstart, record, runtime):
        assert "implemented and TP4 smoke-verified" in text
        assert "not generally qualified" in text
    assert "makes no shared-base" in quickstart
    assert "Download once and fan out over the local fabric" in quickstart
    assert quickstart.count("hf download ") == 2
    assert 'rsync -aH --partial --info=progress2 "${target_model}/"' in quickstart
    assert 'rsync -aH --partial --info=progress2 "${draft_model}/"' in quickstart
    assert 'docker pull "${image}"' in quickstart
    assert 'docker image save "${image}" | zstd' in quickstart
    assert "'zstd -d | docker image load'" in quickstart
    assert "codex/glm53-readme-quickstart-consolidation" in quickstart
    for term in (
        "Other models and topologies",
        "embedded MTP with SparkCache",
        "C16",
        "soak",
        "fault injection",
    ):
        assert term in quickstart
    assert re.search(r"(?i)\b[A-Z]:\\", quickstart) is None
    assert re.search(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)", quickstart) is None


def test_published_image_relative_document_links_resolve() -> None:
    documents = (QUICKSTART, RECORD, HERE / "README.md")
    link = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for document in documents:
        for target in link.findall(document.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://")):
                continue
            relative = target.split("#", maxsplit=1)[0]
            assert (document.parent / relative).resolve().exists(), (
                document,
                target,
            )
