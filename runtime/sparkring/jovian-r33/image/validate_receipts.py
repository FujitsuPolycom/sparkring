#!/usr/bin/env python3
"""Strictly bind every R33 component receipt to locked sources and artifact bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)


def load_kv(path: Path) -> dict[str, str]:
    result = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw != raw.strip() or "=" not in raw:
            raise ValueError(f"malformed key/value receipt line: {path.name}:{number}")
        key, value = raw.split("=", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_.-]*", key) or not value or key in result:
            raise ValueError(f"invalid or duplicate key/value receipt: {path.name}:{number}")
        result[key] = value
    return result


def load_sums(path: Path) -> dict[str, str]:
    result = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split(maxsplit=1)
        if len(parts) != 2 or not HEX64.fullmatch(parts[0]):
            raise ValueError(f"malformed SHA256SUMS line: {path.name}:{number}")
        name = parts[1].lstrip("*")
        posix = PurePosixPath(name)
        if ".." in posix.parts or not posix.name or posix.name in result:
            raise ValueError(f"ambiguous SHA256SUMS path: {path.name}:{number}")
        result[posix.name] = parts[0]
    return result


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"receipt identity mismatch for {label}: {actual!r} != {expected!r}")


def artifact_map(lock: dict, inputs: dict) -> dict[str, dict]:
    records = {item["name"]: item for item in lock["artifacts"]}
    for name, item in inputs.items():
        if name not in records:
            records[name] = item
    return records


def validate(lock: dict, inputs: dict, receipts: Path) -> dict:
    sources = lock["source_identities"]
    for name, value in sources.items():
        if name.endswith("sha256") and not HEX64.fullmatch(str(value)):
            raise ValueError(f"source lock contains a malformed SHA-256: {name}")
        if (name.endswith("commit") or name.endswith("_tree") or name.endswith("head")) and not HEX40.fullmatch(str(value)):
            raise ValueError(f"source lock contains a malformed Git identity: {name}")
    artifacts = artifact_map(lock, inputs)
    checks = []

    def check(actual, expected, label):
        require_equal(actual, expected, label)
        checks.append(label)

    post = load_json(receipts / "post-build-source-identities.json")["sources"]
    post_document = load_json(receipts / "post-build-source-identities.json")
    check(post_document["schema"], "sparkring-r33-post-build-source-identities/v1", "post-build schema")
    check(post_document["status"], "captured-after-component-builds-before-image-assembly", "post-build status")
    check(post["vllm"]["head"], sources["vllm_head"], "vllm post-build head")
    check(post["vllm"]["index_tree"], sources["vllm_integrated_tree"], "vllm integrated tree")
    check(post["vllm"]["unstaged_diff_sha256"], EMPTY_SHA256, "vllm unstaged diff")
    check(post["vllm"]["cached_diff_sha256"], sources["vllm_cached_diff_sha256"], "vllm cached diff")
    check(post["vllm"]["status_sha256"], sources["vllm_status_sha256"], "vllm status")
    check(post["vllm"]["status_lines"], sources["vllm_status_lines"], "vllm status lines")

    native = load_kv(receipts / "vllm-native-source-receipt.txt")
    all_native = load_kv(receipts / "vllm-all-native-source-receipt.txt")
    rust = load_kv(receipts / "rust-source-receipt.txt")
    package = load_json(receipts / "vllm-package-verification.json")
    check(native["status"], "compiled-cpu-import-and-gpu-qualification-pending", "focused native status")
    check(all_native["status"], "compiled-all-cmake-targets-cpu-import-and-gpu-qualification-pending", "all native status")
    check(rust["status"], "compiled-and-cpu-verified", "Rust status")
    check(rust["architecture"], "aarch64", "Rust architecture")
    check(package["status"], "package-structure-and-integrity-qualified-runtime-pending", "vllm package status")
    check(package["runtime_qualification"], "pending", "vllm runtime qualification")
    for actual, label in ((native["vllm.result.tree"], "focused native tree"),
                          (all_native["result.tree"], "all native tree"),
                          (rust["vllm.result.tree"], "rust vllm tree"),
                          (package["native_source_tree"], "packaged native tree")):
        check(actual, sources["vllm_native_tree"], label)
    check(package["source_tree"], sources["vllm_integrated_tree"], "packaged integrated tree")
    check(
        package["flash_attn_source_commit"],
        sources["vllm_flash_attn_commit"],
        "packaged vLLM FlashAttention source commit",
    )
    check(
        package["flash_attn_python_files_byte_checked"],
        5,
        "packaged vLLM FlashAttention Python file count",
    )
    check(package["wheel_sha256"], artifacts["vllm"]["sha256"], "vllm wheel")
    check(package["wheel"], Path(artifacts["vllm"]["source"]).name, "vllm wheel basename")
    check(package["record_valid"], True, "vllm RECORD")
    check(package["native_to_package_diff"], ["requirements/cuda.txt"], "vllm native/package diff")
    if any(item.get("machine") != "AArch64" for item in package["native_and_rust_elf"].values()):
        raise ValueError("vLLM package contains a non-AArch64 native payload")
    checks.append("vllm native AArch64 inventory")
    check(native["flashkda.base.commit"], sources["flashkda_commit"], "FlashKDA commit")
    check(native["flashkda.patch.sha256"], sources["flashkda_patch_sha256"], "FlashKDA patch")
    check(native["cutlass.commit"], sources["cutlass_commit"], "CUTLASS commit")
    flashkda_source = load_json(receipts / "flashkda-source-identity.json")
    for actual, expected, label in (
        (flashkda_source["schema"], "sparkring-r33-flashkda-source-identity/v1", "FlashKDA source schema"),
        (flashkda_source["status"], "patched-source-materialization-verified", "FlashKDA source status"),
        (flashkda_source["head"], sources["flashkda_commit"], "FlashKDA source head"),
        (flashkda_source["base_tree"], sources["flashkda_base_tree"], "FlashKDA base tree"),
        (flashkda_source["result_tree"], sources["flashkda_result_tree"], "FlashKDA result tree"),
        (flashkda_source["diff_sha256"], sources["flashkda_result_diff_sha256"], "FlashKDA result diff"),
    ):
        check(actual, expected, label)
    focused_sums = load_sums(receipts / "vllm-native-SHA256SUMS")
    if set(focused_sums) != {"_C_stable_libtorch.abi3.so", "_flashkda_C.abi3.so"}:
        raise ValueError("focused vLLM SHA256SUMS contains unexpected entries")
    flashkda_sha = package["native_and_rust_elf"]["vllm/_flashkda_C.abi3.so"]["sha256"]
    check(focused_sums["_flashkda_C.abi3.so"], flashkda_sha, "FlashKDA packaged payload")
    package_sums = load_sums(receipts / "vllm-package-SHA256SUMS")
    if set(package_sums) != {package["wheel"]}:
        raise ValueError("vLLM package SHA256SUMS contains unexpected entries")
    check(package_sums[package["wheel"]], artifacts["vllm"]["sha256"], "vllm wheel sums")
    rust_parser = package["native_and_rust_elf"]["vllm/_rust_tool_parser.abi3.so"]["sha256"]
    rust_sums = load_sums(receipts / "rust-SHA256SUMS")
    if set(rust_sums) != {"_rust_tool_parser.abi3.so", "vllm-rs"}:
        raise ValueError("Rust SHA256SUMS contains unexpected entries")
    check(rust_sums["_rust_tool_parser.abi3.so"], rust_parser, "Rust parser payload")
    check(rust["vllm.commit"], sources["vllm_head"], "Rust vllm head")
    check(rust["vllm.rust.tree"], sources["rust_tree"], "Rust source tree")
    check(rust["vllm.rust_toolchain.blob"], sources["rust_toolchain_blob"], "Rust toolchain blob")
    check(rust["rust.manifest.sha256"], sources["rust_manifest_sha256"], "Rust manifest")
    check(rust["rust.toolchain.sha256"], sources["rust_toolchain_sha256"], "Rust toolchain archive")

    flash = load_kv(receipts / "flashinfer-source-receipt.txt")
    resume = load_json(receipts / "flashinfer-resume-source-receipt.json")
    check(flash["status"], "compiled-cpu-import-and-gpu-qualification-pending", "FlashInfer build status")
    check(resume["schema"], "sparkring-r33-flashinfer-resume-receipt/v1", "FlashInfer resume schema")
    check(resume["status"], "tracked-and-recursive-submodule-inputs-pristine-generated-only-output-validated", "FlashInfer resume status")
    check(flash["source.commit"], sources["flashinfer_commit"], "FlashInfer source head")
    check(resume["source_head"], sources["flashinfer_commit"], "FlashInfer resume head")
    check(resume["source_tree"], sources["flashinfer_tree"], "FlashInfer source tree")
    check(resume["generated_file_map_sha256"], sources["flashinfer_generated_file_map_sha256"], "FlashInfer generated map")
    check(resume["tracked_input_pristine"], True, "FlashInfer tracked input")
    check(resume["generated_only"], True, "FlashInfer generated-only output")
    check(resume["untracked_source_files"], ["LICENSE.cutlass.txt", "LICENSE.flashattention3.txt", "LICENSE.fmt.txt", "LICENSE.spdlog.txt"], "FlashInfer allowed untracked files")
    check(resume["generated_roots"], ["__pycache__", "build", "flashinfer", "flashinfer-jit-cache", "flashinfer_python.egg-info"], "FlashInfer generated roots")
    if any(not line.startswith(" ") for line in resume["recursive_submodules"]):
        raise ValueError("FlashInfer resume receipt contains a dirty recursive submodule")
    checks.append("FlashInfer recursive submodule cleanliness")
    flash_sums = load_sums(receipts / "flashinfer-SHA256SUMS")
    expected_flash = {
        Path(inputs[name]["destination"]).name: inputs[name]["sha256"]
        for name in ("flashinfer-python", "flashinfer-jit-cache")
    }
    check(flash_sums, expected_flash, "FlashInfer terminal wheel set")
    check(resume["wheel_sha256"], expected_flash, "FlashInfer resume wheel set")
    check(post["flashinfer"]["head"], sources["flashinfer_commit"], "FlashInfer post-build head")
    check(post["flashinfer"]["index_tree"], sources["flashinfer_tree"], "FlashInfer post-build tree")
    check(post["flashinfer"]["recursive_submodules_sha256"], sources["flashinfer_recursive_submodules_sha256"], "FlashInfer post-build submodules")
    check(post["flashinfer"]["status_sha256"], sources["flashinfer_status_sha256"], "FlashInfer post-build status")

    component_specs = {
        "b12x": ("b12x-source-receipt.txt", "source.commit", "b12x_commit", "source.tree", "b12x_tree", "b12x-SHA256SUMS"),
        "lmcache": ("lmcache-source-receipt.txt", "source.commit", "lmcache_commit", None, "lmcache_tree", "lmcache-SHA256SUMS"),
        "instanttensor": ("instanttensor-source-receipt.txt", "source.commit", "instanttensor_commit", None, "instanttensor_tree", "instanttensor-SHA256SUMS"),
        "sparkcache": ("sparkcache-source-receipt.txt", "source.commit", "sparkcache_commit", "source.tree", "sparkcache_tree", "sparkcache-SHA256SUMS"),
        "torchaudio": ("torchaudio-source-receipt.txt", "source.commit", "torchaudio_commit", None, "torchaudio_tree", "torchaudio-SHA256SUMS"),
        "torchvision": ("torchvision-source-receipt.txt", "source.commit", "torchvision_commit", "source.tree", "torchvision_tree", "torchvision-SHA256SUMS"),
    }
    for name, (receipt_name, commit_field, commit_lock, tree_field, tree_lock, sums_name) in component_specs.items():
        item = load_kv(receipts / receipt_name)
        expected_status = {
            "b12x": "packaged-cpu-import-and-gpu-qualification-pending",
            "lmcache": "compiled-cpu-import-and-gpu-qualification-pending",
            "instanttensor": "compiled-cpu-import-and-gpu-qualification-pending",
            "sparkcache": "packaged-runtime-qualification-pending",
            "torchaudio": "compiled-import-and-functional-qualification-pending",
            "torchvision": "compiled-cpu-import-and-gpu-qualification-pending",
        }[name]
        check(item["status"], expected_status, f"{name} receipt status")
        check(item[commit_field], sources[commit_lock], f"{name} source head")
        check(post[name]["head"], sources[commit_lock], f"{name} post-build head")
        if tree_field:
            check(item[tree_field], sources[tree_lock], f"{name} source tree")
        if tree_lock:
            check(post[name]["index_tree"], sources[tree_lock], f"{name} post-build tree")
        sums = load_sums(receipts / sums_name)
        basename = Path(artifacts[name]["source"]).name
        if name != "lmcache" and set(sums) != {basename}:
            raise ValueError(f"{name} SHA256SUMS contains extra entries")
        check(sums[basename], artifacts[name]["sha256"], f"{name} artifact")
    instant = load_kv(receipts / "instanttensor-source-receipt.txt")
    check(instant["libaio.commit"], sources["libaio_commit"], "libaio commit")
    if not any(sources["libaio_commit"] in line and "csrc/third_party/libaio" in line for line in post["instanttensor"]["recursive_submodules"]):
        raise ValueError("InstantTensor post-build receipt lacks exact libaio submodule")
    checks.append("libaio recursive submodule")
    lm_sums = load_sums(receipts / "lmcache-SHA256SUMS")
    if set(lm_sums) != {Path(artifacts["lmcache"]["source"]).name, Path(artifacts["lmcache-cumem-interposer"]["source"]).name}:
        raise ValueError("LMCache SHA256SUMS does not exactly bind wheel and interposer")
    check(lm_sums[Path(artifacts["lmcache-cumem-interposer"]["source"]).name], artifacts["lmcache-cumem-interposer"]["sha256"], "LMCache interposer")
    check(load_kv(receipts / "torchvision-source-receipt.txt")["source.post-build-identical"], "true", "Torchvision post-build identity")

    sparkcache_native = load_json(receipts / "sparkcache-native-build-receipt.json")
    check(sparkcache_native["schema"], "sparkring-r33-sparkcache-native-build/v1", "SparkCache native schema")
    check(sparkcache_native["status"], "qualified-native-build", "SparkCache native status")
    check(sparkcache_native["source"]["repository"], "https://github.com/FujitsuPolycom/sparkcache.git", "SparkCache native repository")
    check(sparkcache_native["source"]["commit"], sources["sparkcache_native_commit"], "SparkCache native head")
    check(sparkcache_native["source"]["tree"], sources["sparkcache_native_tree"], "SparkCache native tree")
    check(sparkcache_native["source"]["status_porcelain"], "", "SparkCache native source status")
    check(sparkcache_native["source"]["archive"]["sha256"], sources["sparkcache_native_source_archive_sha256"], "SparkCache native source archive receipt")
    source_archive = receipts / "sources" / sparkcache_native["source"]["archive"]["filename"]
    check(hashlib.sha256(source_archive.read_bytes()).hexdigest(), sources["sparkcache_native_source_archive_sha256"], "SparkCache native source archive bytes")
    for component, artifact_name in (("placement", "sparkcache-placement"), ("snapshot", "sparkcache-snapshot")):
        record = sparkcache_native["artifacts"][component]
        check(record["filename"], Path(artifacts[artifact_name]["source"]).name, f"SparkCache {component} filename")
        check(record["sha256"], artifacts[artifact_name]["sha256"], f"SparkCache {component} artifact")
        check(record["elf_machine"], "AArch64", f"SparkCache {component} machine")
        if record["sm_targets"] != ["sm_121"]:
            raise ValueError(f"SparkCache {component} lacks SM121 code")
        checks.append(f"SparkCache {component} SM121")
    build = sparkcache_native["build"]
    for actual, expected, label in (
        (build["foundation_reference"], lock["foundation"]["reference"], "SparkCache foundation reference"),
        (build["foundation_image_id"], lock["foundation"]["image_id"], "SparkCache foundation image"),
        (build["cuda_architectures"], ["121"], "SparkCache CUDA architecture"),
        (build["host_arch"], "aarch64", "SparkCache host architecture"),
    ):
        check(actual, expected, label)
    if build["build_type"] != "Release":
        raise ValueError("SparkCache native build is not Release")
    checks.append("SparkCache Release build")
    if "cuda_13.3" not in build["cuda_version"]:
        raise ValueError("SparkCache native build did not use CUDA 13.3")
    expected_native_tests = {
        "ctest": "pass",
        "hybrid_page_c_api_byte_correctness": "pass",
        "hybrid_page_gpu_probe": "pass",
        "native_python": "pass",
        "placement_ctypes": "pass",
        "placement_gpu_probe": "pass",
        "snapshot_compact_matrix": "pass",
        "snapshot_ctypes_attested": "pass",
        "snapshot_gpu_probe": "pass",
    }
    for key, expected in expected_native_tests.items():
        check(sparkcache_native["tests"].get(key), expected, f"SparkCache native test {key}")
    page_copy = sparkcache_native["tests"].get("page_copy_gpu_modes_1_2_3", "")
    if not (page_copy == "pass" or page_copy.startswith("not_exercised: exact f220230a benchmark self-rejects")):
        raise ValueError("SparkCache page-copy test has an unrecognized result")
    checks.append("SparkCache page-copy test disposition")
    check(sparkcache_native["destinations"]["placement"], "/opt/sparkring/sparkcache/lib/libspark_cache_placement.so", "SparkCache placement destination")
    check(sparkcache_native["destinations"]["snapshot"], "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so", "SparkCache snapshot destination")
    checks.append("SparkCache native tests")

    xgrammar = load_kv(receipts / "xgrammar-source-receipt.txt")
    check(post["xgrammar"]["head"], sources["xgrammar_commit"], "XGrammar post-build head")
    check(post["xgrammar"]["index_tree"], sources["xgrammar_base_tree"], "XGrammar post-build base tree")
    for field, lock_name in (("source.commit", "xgrammar_commit"), ("source.base.tree", "xgrammar_base_tree"),
                             ("source.result.tree", "xgrammar_result_tree"), ("metadata.patch.sha256", "xgrammar_transformers_metadata_patch_sha256")):
        check(xgrammar[field], sources[lock_name], f"XGrammar {field}")
    check(xgrammar["source.post-build-input-identical"], "true", "XGrammar post-build input")
    patch_sha = hashlib.sha256((receipts / "xgrammar-transformers5.patch").read_bytes()).hexdigest()
    check(patch_sha, sources["xgrammar_transformers_metadata_patch_sha256"], "XGrammar patch bytes")
    xverify = load_json(receipts / "xgrammar-wheel-verification.json")
    check(xverify["schema"], "sparkring-r33-xgrammar-wheel-verification/v1", "XGrammar verifier schema")
    check(xverify["record_valid"], True, "XGrammar RECORD")
    check(xverify["metadata_adjustment"], "transformers>=4.38.0", "XGrammar metadata adjustment")
    check(xverify["wheel_sha256"], artifacts["xgrammar"]["sha256"], "XGrammar wheel verification")
    check(load_sums(receipts / "xgrammar-SHA256SUMS")[Path(artifacts["xgrammar"]["source"]).name], artifacts["xgrammar"]["sha256"], "XGrammar sums")

    foundation = load_json(receipts / "foundation-image.json")
    for field, expected in (("torch_source", sources["torch_commit"]), ("torch_version", lock["foundation"]["torch"]),
                            ("cuda_toolkit", lock["foundation"]["cuda"]), ("platform", "aarch64"),
                            ("image_id", lock["foundation"]["image_id"])):
        check(foundation[field], expected, f"foundation {field}")
    check(post["torch"]["head"], sources["torch_commit"], "Torch post-build head")
    check(post["torch"]["index_tree"], sources["torch_tree"], "Torch post-build tree")
    check(post["cutlass"]["head"], sources["cutlass_commit"], "CUTLASS post-build head")
    check(post["cutlass"]["index_tree"], sources["cutlass_tree"], "CUTLASS post-build tree")
    check(load_sums(receipts / "torch-SHA256SUMS")[Path(artifacts["torch-resolver-input"]["source"]).name], artifacts["torch-resolver-input"]["sha256"], "Torch wheel")

    nccl = load_json(receipts / "nccl-arm-port-receipt.json")
    for actual, expected, label in (
        (nccl["schema"], "sparkring-r33-arm64-nccl-port-build/v1", "NCCL schema"),
        (nccl["status"], "passed", "NCCL status"), (nccl["base_commit"], sources["nccl_base_commit"], "NCCL base"),
        (nccl["patched_tree"], sources["nccl_patched_tree"], "NCCL patched tree"),
        (nccl["patch_sha256"], sources["nccl_patch_sha256"], "NCCL patch"),
        (nccl["candidate"]["sha256"], artifacts["nccl-2.31.2-sparkring-routing"]["sha256"], "NCCL artifact"),
        (nccl["candidate"]["nccl_version"], 23102, "NCCL version"),
        (nccl["candidate"]["elf_machine"], "AArch64", "NCCL machine"),
    ):
        check(actual, expected, label)
    expected_nccl_checks = {"elf_machine", "soname", "needed", "version_requirements", "exported_symbols", "nccl_version", "public_header"}
    if set(nccl.get("checks", {})) != expected_nccl_checks or any(value is not True for value in nccl["checks"].values()):
        raise ValueError("NCCL receipt contains a failed check")
    checks.append("NCCL checks")

    sircl = load_json(receipts / "sircl-build-receipt.json")
    for actual, expected, label in (
        (sircl["schema"], "sparkring-r33-sircl-native-build/v1", "SIRCL schema"),
        (sircl["status"], "native-built-tested", "SIRCL status"),
        (sircl["source"]["source_commit"], sources["sircl_commit"], "SIRCL head"),
        (sircl["source"]["spark_transport_tree"], sources["sircl_tree"], "SIRCL tree"),
        (sircl["source"]["source_archive_sha256"], sources["sircl_source_archive_sha256"], "SIRCL source archive"),
        (sircl["native"]["sha256"], artifacts["sircl"]["sha256"], "SIRCL artifact"),
        (sircl["native"]["host_architecture"], "aarch64", "SIRCL architecture"),
        (sircl["native"]["cuda_architectures"], [121], "SIRCL CUDA architecture"),
        (sircl["validation"]["ctest_total"], 29, "SIRCL test total"),
        (sircl["validation"]["ctest_passed"], 29, "SIRCL tests passed"),
        (sircl["validation"]["ctest_failed"], 0, "SIRCL tests failed"),
        (sircl["validation"]["sm121_image_present"], True, "SIRCL SM121 image"),
    ):
        check(actual, expected, label)

    clean = ("b12x", "lmcache", "instanttensor", "xgrammar", "sparkcache", "torchvision", "torchaudio", "torch", "cutlass")
    for name in clean:
        record = post[name]
        if (record["status_lines"] != 0 or record["status_sha256"] != EMPTY_SHA256
                or record["cached_diff_sha256"] != EMPTY_SHA256 or record["unstaged_diff_sha256"] != EMPTY_SHA256):
            raise ValueError(f"post-build source is not clean: {name}")
        checks.append(f"{name} post-build cleanliness")

    return {
        "schema": "sparkring-r33-semantic-receipt-validation/v1",
        "checks_passed": True,
        "check_count": len(checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-lock", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(load_json(args.artifact_lock), load_json(args.inputs)["inputs"], args.receipts)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
