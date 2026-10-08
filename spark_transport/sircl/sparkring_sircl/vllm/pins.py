"""vLLM builds the SIRCL adapter was checked against, pinned by file hash.

Each :class:`VllmBuild` records the SHA-256 (line endings normalized to LF) of
every vLLM file whose code the adapter relies on. The official extension
points (platform plugin, device communicator) need only the interfaces they
use and are checked by capability at load; their pins record where those
interfaces were verified. A version-pinned shim (:mod:`.shims`) installs only
when every file it touches matches one pinned build, and refuses otherwise.

Adding a vLLM version:

1. print its hashes: ``python -m sparkring_sircl.vllm.pins /path/to/vllm``;
2. re-read the hook table (:mod:`.hooks` and the adapter's ``README.md``) against that version's sources
   (the listed files and line ranges) and confirm or update each hook;
3. add a :class:`VllmBuild` entry with the printed hashes, run the adapter's
   CPU tests and the ring checks in ``STATUS.md``.

The checks read files only; they do not import vLLM.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PLATFORM_FILES = (
    "plugins/__init__.py",
    "platforms/__init__.py",
    "platforms/interface.py",
    "platforms/cuda.py",
    "distributed/parallel_state.py",
    "distributed/communication_op.py",
    "distributed/device_communicators/base_device_communicator.py",
    "distributed/device_communicators/cuda_communicator.py",
    "distributed/device_communicators/pynccl.py",
    "v1/worker/gpu_worker.py",
    "v1/worker/worker_base.py",
)
DCP_FILES = ("v1/attention/ops/dcp.py",)
SLOT_FILES = ("distributed/device_communicators/b12x_roce_all_reduce.py",
              "distributed/device_communicators/cuda_communicator.py")
MHC_FILES = ("models/glm5next/nvidia/mhc_prefill_sharding.py", "models/glm5next/nvidia/model.py")
WORKER_FILES = ("v1/worker/gpu_worker.py",)
DCP_TRANSPORT_FILES = ("distributed/device_communicators/b12x_dcp.py",)
QWEN_HC_FILES = ("models/qwen4_exp/nvidia/hc_prefill.py", "models/qwen4_exp/nvidia/model.py")
FUSED_FILES = (
    "compilation/passes/fusion/allreduce_rms_fusion.py",
    "compilation/passes/pass_manager.py",
    "compilation/backends.py",
    "distributed/device_communicators/b12x_pcie_all_reduce.py",
    "models/common/ops/fused_allreduce_rms_norm.py",
)
NORM_FILES = (
    "models/common/ops/fused_allreduce_rms_norm.py",
    "model_executor/layers/layernorm.py",
    "ir/op.py",
    "ir/ops/layernorm.py",
    "kernels/vllm_c.py",
)
HOOK_FILES = tuple(dict.fromkeys(PLATFORM_FILES + DCP_FILES + FUSED_FILES + SLOT_FILES + MHC_FILES
                                  + DCP_TRANSPORT_FILES + NORM_FILES + QWEN_HC_FILES))


@dataclasses.dataclass(frozen=True)
class VllmBuild:
    name: str
    version: str
    source: str
    files: Mapping[str, str | None]       # None: the build does not have the file


_COMMON = {
    "plugins/__init__.py": "fb6e6ee432c5a4ae207aaffe4b546aca9381f56ee8498b0a3e0d9af2d289d023",
    "platforms/__init__.py": "12c5a4df6a59b078d12240c91eced342a7bc8fc216a5e20438f6f5acc7e5564b",
    "platforms/interface.py": "51eed6d2a949a9a86ffa7e35ff75b46aa2cf1ca8a9edc86be8ee90cc4be9ef73",
    "platforms/cuda.py": "8cdeadcddcc7e9ffa32554cdb5d1158db60e28fcd1dbb6c8ec88036391863fde",
    "distributed/parallel_state.py":
        "3aba1b4376c828a343d781d23fd1dd0c4bdb40b8dafa06a927006e7596359212",
    "distributed/communication_op.py":
        "6692824485475647139a00481913c8599c303e9e953eca97968ddd2f30c0d65c",
    "distributed/device_communicators/pynccl.py":
        "882fb83c5a3c565a3f8fd664b79f5ceae566dca180019ecafd74bc8c11d9f8cf",
    "distributed/device_communicators/b12x_roce_all_reduce.py":
        "cca86b8c97d96b3647e4ef3f90f9638ebc4d8b5dfc1673abc2e5a7750bed9018",
    "distributed/device_communicators/base_device_communicator.py":
        "4a7670f12ed7c7621b25d476e2687c24ca01f3be183fff4a6a25337596fe506b",
    "distributed/device_communicators/cuda_communicator.py":
        "33d27cad52def83159c7f63a9b7bd47e61d3e1d50c453ea9927576fc808f3ad8",
    "v1/attention/ops/dcp.py": "17586fe1dbf702e05757b1acba1926b2ad7341501c1d10d233e9d57720b919a2",
    "distributed/device_communicators/b12x_dcp.py":
        "a9a3c370cee1e2be46506612b863905f0fe4920f9dba3b5ea6287b418089e206",
    "compilation/passes/fusion/allreduce_rms_fusion.py":
        "306c3c1774402a6eb459179889841874eb30fdda5f0952ddcbd1e80ba61edbe0",
    "compilation/passes/pass_manager.py":
        "486979a058aa2e20ebedcbb6e039f7a59f38d2efd2cda11219fae4067be0968d",
    "compilation/backends.py": "fa4bff8920b363efbbde9456f135fa9d2c004a5bde1898877f177418b2d8b008",
    "model_executor/layers/layernorm.py":
        "4126258bf85aa3af54c78cfbf2a0f32000491e42be37661f2d1ac9f0beea00f3",
    "ir/op.py": "fd0355b398c0d96235bb6aba0c9c08f73c199429bd8ec2aed227178c297c1eb2",
    "ir/ops/layernorm.py": "65d33dcb96404ddde273acf84ef901151a8155a2cffc144bdd0c49fe1d576a22",
    "kernels/vllm_c.py": "ef37e71736807026f0b8ad187155b63f147984c03d287a78790ef8f8ed642ec1",
}

SUPPORTED: tuple[VllmBuild, ...] = (
    VllmBuild(
        name="lil-image-aba309e4610c",
        version="0.1.dev21553+gab86b7073.d20261001",
        source="Local Inference Lab vLLM fork as installed in serving image aba309e4610c",
        files={
            **_COMMON,
            "v1/worker/gpu_worker.py": "989330be6dca8f5068032b6fe4f0f6276b31fef8bea491443a9f6c7cafd038bd",
            "v1/worker/worker_base.py": "379879577dcc606ccfa9bc5bf631d40d7ba9cd05493faff319ccfe07d8793ed5",
            "distributed/device_communicators/b12x_pcie_all_reduce.py":
                "e431bfec41dc1b04547ab945670e633d3af70458d90019c93c624626cd561ec2",
            "models/common/ops/fused_allreduce_rms_norm.py":
                "036e07b212f3154fb58a26da97c18015315cc736f89097eac1b733456d4c3e4c",
            "models/glm5next/nvidia/mhc_prefill_sharding.py":
                "b3616d081087eec0a35475e202f37e08c5e88753cfd280e64fb09e0e2ed1a66c",
            "models/glm5next/nvidia/model.py":
                "23b4af130ebd443274a79695dd5a94871df6243ff9d6b66aa1fc254b298f89d1",
            "models/qwen4_exp/nvidia/hc_prefill.py":
                "7c7ed3319d99a7d2d7e608f06a0950b33b3d268713a3ea58ffee4720c205e1ab",
            "models/qwen4_exp/nvidia/model.py":
                "1808f802faf97a7ec5add2ff3db261786f92e5ef01b7b9aeaad26c8f07c692c3",
        },
    ),
    VllmBuild(
        name="lil-karmic-kraken-beta-4a87c588",
        version="source tree (no generated _version.py)",
        source="local-inference-lab/vllm integration/karmic-kraken-beta at "
               "4a87c5884c2d640ae4514f90df455de584472cb9",
        files={
            **_COMMON,
            "v1/worker/gpu_worker.py": "d0dd42c8ec547d2efb0135b455ef5372ef4bf93231c2c69c1904a5212821c368",
            "v1/worker/worker_base.py": "3a0ca44daf0b00fe626ca21a5b71d1a0e0e506759449a7fb8967292edccbcf35",
            "distributed/device_communicators/b12x_pcie_all_reduce.py":
                "06034b5133ee03c472eee9329d55b459d1f1df20c424dca583dec2a4d944a685",
            "models/common/ops/fused_allreduce_rms_norm.py":
                "664180169082cb65241bd9b82497d669e552ebb1f063c2e0b23657b3707dd080",
            "models/glm5next/nvidia/mhc_prefill_sharding.py": None,
            "models/glm5next/nvidia/model.py":
                "590196762c25ddfd8d4db3b88e6c6b6038f49f7af5bbb5cfe9a3882bf18b4fc8",
            "models/qwen4_exp/nvidia/hc_prefill.py": None,
            "models/qwen4_exp/nvidia/model.py":
                "75d071a584dc9a0be96afe1707db36cfc3ed840b899e14892de413679c7d0b69",
        },
    ),
    # Differs from the image's build in five hook files, none in code a shim wraps or relies on:
    # v1/attention/ops/dcp.py (MLADCPManager.__init__ places its buffers on the GPU when the weights are built
    # on the meta device; dcp_a2a_lse_reduce and its helpers are identical); v1/worker/gpu_worker.py
    # (_compile_or_warm_up_model_after_preparation sets the serving thread count before warm-up, and
    # _scoped_allocator_max_split does nothing under expandable segments; the seven methods worker_regimes
    # wraps and the post-step check are identical); models/glm5next/nvidia/model.py (the vision tower takes a
    # checkpoint's own quantization recipes; maybe_create's binding and every row-ownership call site are
    # identical, and mhc_prefill_sharding.py is the image's); models/common/ops/fused_allreduce_rms_norm.py
    # (tries the B12X PCIe communicator's fused path first; the file is the one the karmic-kraken-beta build
    # pins for the fused_allreduce_rms_norm shim); b12x_pcie_all_reduce.py (a row limit of its fused path).
    VllmBuild(
        name="sparkring-kraken-beta-20261007-bc9ea774",
        version="the image's vLLM with the CSF overlay payload over it",
        source="FujitsuPolycom/vllm sparkring/kraken-beta-20261007 at "
               "bc9ea7744bc73d1f45cf3232d5de1b8f823919ae: local-inference-lab/vllm integration/karmic-kraken-beta "
               "89f1ceec merged into the image's build, sparkring/kraken-beta-20261004 at d51b4181. Overlay "
               "payload csf_payload.tar, SHA-256 05ba9d1c6e5ac005c0279901b911afc78b7d74a792c06a359ee2aed854c53c84",
        files={
            **_COMMON,
            "v1/attention/ops/dcp.py": "5cc592b1baa5defd47336d19be4002fe0f98827134ebbd755ef9364350456a53",
            "v1/worker/gpu_worker.py": "8ce5974c30bfff30fb69cd080d8e68dfaf98ce431127687d2e2128c05baa5948",
            "v1/worker/worker_base.py": "379879577dcc606ccfa9bc5bf631d40d7ba9cd05493faff319ccfe07d8793ed5",
            "distributed/device_communicators/b12x_pcie_all_reduce.py":
                "06034b5133ee03c472eee9329d55b459d1f1df20c424dca583dec2a4d944a685",
            "models/common/ops/fused_allreduce_rms_norm.py":
                "664180169082cb65241bd9b82497d669e552ebb1f063c2e0b23657b3707dd080",
            "models/glm5next/nvidia/mhc_prefill_sharding.py":
                "b3616d081087eec0a35475e202f37e08c5e88753cfd280e64fb09e0e2ed1a66c",
            "models/glm5next/nvidia/model.py":
                "1576615576632007e80fbb7138996b0f4fe4cfdbeb1695841ee0e3cef2210791",
            "models/qwen4_exp/nvidia/hc_prefill.py":
                "7c7ed3319d99a7d2d7e608f06a0950b33b3d268713a3ea58ffee4720c205e1ab",
            "models/qwen4_exp/nvidia/model.py":
                "1808f802faf97a7ec5add2ff3db261786f92e5ef01b7b9aeaad26c8f07c692c3",
        },
    ),
)


def file_hash(path: Path) -> str:
    """SHA-256 of a file with CRLF line endings normalized to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def installed_root() -> Path | None:
    """Directory of the installed ``vllm`` package, found without importing it."""
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def hashes(root: Path, files: Sequence[str] = HOOK_FILES) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for relative in files:
        path = root / relative
        result[relative] = file_hash(path) if path.is_file() else None
    return result


def matching_build(root: Path, files: Sequence[str]) -> VllmBuild | None:
    """The pinned build whose hashes match every one of ``files`` under ``root``."""
    actual = hashes(root, files)
    for build in SUPPORTED:
        if all(build.files.get(name) is not None and build.files.get(name) == actual[name]
               for name in files):
            return build
    return None


@dataclasses.dataclass(frozen=True)
class Report:
    root: Path | None
    matches: tuple[str, ...]
    mismatches: Mapping[str, Mapping[str, str | None]]

    @property
    def supported(self) -> bool:
        return bool(self.matches)

    def describe(self) -> str:
        if self.root is None:
            return "vLLM is not installed"
        if self.matches:
            return f"vLLM at {self.root} matches pinned build(s): {', '.join(self.matches)}"
        lines = [f"vLLM at {self.root} matches no pinned build:"]
        for build, files in self.mismatches.items():
            lines.append(f"  {build}: differs in {', '.join(sorted(files))}")
        return "\n".join(lines)


def check(root: Path | None) -> Report:
    if root is None:
        return Report(None, (), {})
    actual = hashes(root)
    matches, mismatches = [], {}
    for build in SUPPORTED:
        # A file the build does not have (None) must be absent from the tree too.
        differing = {name: actual.get(name) for name, expected in build.files.items()
                     if actual.get(name) != expected}
        if differing:
            mismatches[build.name] = differing
        else:
            matches.append(build.name)
    return Report(root, tuple(matches), mismatches)


def check_installed() -> Report:
    return check(installed_root())


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else installed_root()
    if root is None:
        print("vLLM is not installed; pass the vllm package directory", file=sys.stderr)
        return 1
    for name, digest in hashes(root).items():
        print(f"{name} {digest or 'missing'}")
    print(check(root).describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
