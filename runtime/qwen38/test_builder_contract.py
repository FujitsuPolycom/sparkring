from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).resolve().parent


def test_container_builds_nccl_before_cuda_132_runtime_stage() -> None:
    text = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert text.count("FROM ${BASE_IMAGE}") == 2
    nccl_stage, runtime_stage = text.split("FROM ${BASE_IMAGE} AS runtime", 1)
    assert "cuda-toolkit-13-2" not in nccl_stage
    assert 'NVCC_GENCODE="-gencode=arch=compute_${NCCL_CUDA_ARCH}' in nccl_stage
    assert "cuda-toolkit-13-2=${CUDA_TOOLKIT_PACKAGE_VERSION}" in runtime_stage
    assert "e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54" in text


def test_container_preserves_architecture_split_and_workspace_layout() -> None:
    text = (HERE / "Containerfile").read_text(encoding="utf-8")
    assert 'ARG VLLM_TORCH_CUDA_ARCH_LIST=12.0f' in text
    assert 'ARG EXLLAMAV3_TORCH_CUDA_ARCH_LIST=12.1' in text
    assert "CMAKE_CUDA_ARCHITECTURES=121" not in text
    assert "COPY bundle/sources/vllm /ws/src/vllm-gg" in text
    assert "COPY bundle/sources/exllamav3 /ws/src/exllamav3" in text
    assert "COPY bundle/runtime/qwen38_dgx4_serve.sh /ws/qwen38_dgx4_serve.sh" in text
    assert "huggingface" not in text.lower()
    assert "Qwen3.8-27B-EXL3-K5K6-hydrated" not in text
    assert "git iproute2 libibverbs-dev" in text


def test_build_wrapper_verifies_parent_and_clean_inputs() -> None:
    text = (HERE / "build-image.sh").read_text(encoding="utf-8")
    assert "parent reference drift" in text
    assert "parent image identity drift" in text
    assert "git -C \"${repo_root}\" diff --quiet HEAD" in text
    assert "prepare_context.py" in text
    assert "sparkring-qwen38:arm64-sm121" in text


def test_readme_uses_local_image_distribution_without_claiming_live_status() -> None:
    text = (HERE / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())
    assert "no published image is required" in text.lower()
    assert "docker save" in text
    assert "docker load" in text
    assert "live validation pending" in text
    assert "contains no model weights" in normalized
    assert "create one persistent container" not in normalized
    assert "do not create a separate idle set" in normalized
    assert "source-receipt.json" in text


def test_image_build_gates_vllm_and_exllamav3_imports() -> None:
    container = (HERE / "Containerfile").read_text(encoding="utf-8")
    verifier = (HERE / "verify_runtime.py").read_text(encoding="utf-8")
    assert "verify_runtime.py --imports" in container
    assert '"vllm", "exllamav3_ext"' in verifier
    assert "exl3_gemm" in verifier
    assert 'CMD ["sleep", "infinity"]' in container
