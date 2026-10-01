"""Research layer: native B12X MXFP8 MoE kernels for the Qwen3.8-Flash-Next MTP draft experts.

Status: research-only. The layer is a hand-merged research patch of
local-inference-lab/vllm#955 (head 9b4f167af5904da3c329b03b54fcdbad548a34ee)
and local-inference-lab/b12x#449 (head b8e242af895156b85f8e3032189ffac45a8517a5)
onto the vLLM (source commit 03c4af34) and B12X (source commit c8e46128) that
installer image dev-20260930-spinwait-cuda1342-nccl2323-status033
(sha256:fcb20b0ce839) installs. With it, a speculative config with
``"moe_backend":"b12x"`` runs the draft's MXFP8 experts (E4M3 weights, UE8M0
K/32 scales) on B12X's W8A8 grouped MoE (backend ``B12X_MXFP8``); the target's
NVFP4 experts keep B12X, and the default ``humming`` draft backend is
unchanged.

The two patches beside this file are diffs against the parent's installed
files under site-packages. ``replace`` copies those parent files into an empty
directory and applies the patches with ``git apply`` (exact context, no fuzz),
so any difference in a parent file fails the build. Every replaced and added
file is pinned below to its inherited and resulting SHA-256. No file is native
code, and none is a B12X source that the prepared RoCE transport verifies.

Build on a host that has the parent image, from the repository root:

    python3 performance/records/qwen38-flash-next/b12x-mxfp8-draft-20260930/derive_b12x_mxfp8_draft.py \\
      prepare --parent-lock runtime/releases/dev-20260930-spinwait-cuda1342-nccl2323-status033/installer-image.json \\
      --output CONTEXT
    python3 performance/records/qwen38-flash-next/b12x-mxfp8-draft-20260930/derive_b12x_mxfp8_draft.py \\
      build --context CONTEXT --tag TAG --name NAME --output LOCK --profiles qwen38-flash-next-tp2
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from runtime.images.derived_layer import SITE, Layer, main  # noqa: E402

HERE = Path(__file__).resolve().parent
PATCHES = (HERE / "vllm955-on-image-03c4af34.patch", HERE / "b12x449-on-image-c8e46128.patch")

# Site-packages path: (inherited SHA-256 or None for an added file, resulting SHA-256).
FILES = {
    "vllm/config/kernel.py": (
        "c06bac5fb988d9578a9cbcd7319e55fdf1773811338c971a5508a71d1cc00175",
        "e3ffc467338888d490b2bf7fc20bf32c749fbffe7da0f61e72f415c8de7ab09a"),
    "vllm/model_executor/layers/fused_moe/b12x.py": (
        "30a4df7ba2a76693a09005a858f2a9a030e71361435584eaed4d9045b1487583",
        "93ab35203d5d87db8eeb1cb75700fc9cc2cc3d3add3e391d9b9091c9d5cc4353"),
    "vllm/model_executor/layers/fused_moe/oracle/fp8.py": (
        "245bd36c285df1e6f7c51e52dcda31ad247894577a9f8d765613c2c2feae9aee",
        "6c1b3d0ce00ea609d30bb60f1beb8efdea223a12a1f3e86813e67ed478ecdb85"),
    "vllm/model_executor/layers/fused_moe/oracle/mxfp8.py": (
        "9cfdc8102a04bb844dcd7ed88043bba96ec751794bc093da76c69aa8cf88cf48",
        "449b41269467acc6c08c519cd79ff2aa4596f580edaa6e9ea081053b766aad79"),
    "vllm/model_executor/layers/quantization/modelopt.py": (
        "9caa60e97c30412b24fa1c27266513d8c031e26a9e30b5d8e5e6e2810b20d1f9",
        "33ebc23537e1fe20fcba529080b7a815ef6bb5c52b9180002f978a625e4b05a2"),
    "b12x/_lib/intrinsics.py": (
        "4af8c25349e4e8eb7a42fff7580261969e2728f53f8fb99172b9ab4a3a2b5c2d",
        "e5888269b0073359c4d16a34758b4a92f0b394030bb01ae159ef87f4843b3041"),
    "b12x/moe/__init__.py": (
        "b04f0836d24bda5c2889bedcd01171880888d666842aeee8605cc0b1ec6b4902",
        "67390f84ab5bff5ec2dc046e1efd13ccaea06efec9f8c93d17f9a0c38de6f744"),
    "b12x/moe/_shared/execution.py": (
        "38f7a1bfbed499597645b93c56b7471394c66999f62ca171449583cfc0679ead",
        "654d0f2ba21e0a597f854bd821b2a1d8c5cc79eb9eb93f9cb41d06b6604ab8fc"),
    "b12x/moe/_shared/kernels/dynamic.py": (
        "335bce6271c66989a08ea3a00b023e5a07a6746a459e4bd715c4a068f5ac638a",
        "db21546119ca80d746f4142ad14f9abb3341e9bfc083a74816372ca18a78d707"),
    "b12x/moe/_shared/kernels/mxfp6_moe.py": (
        "4525eefb70abd2625605b14f90cc6f8036347e7d276c75790d3ae00dd703a811",
        "c26fa60083589bd4788a4ae2390598098ebce1c8435d2ed56a1ac26c9e4a1d2d"),
    "b12x/moe/_shared/kernels/reference.py": (
        "44a2bb6aa4d797d1631a803f7c5214d674be27a18af3b44524ee1ccff035f9b1",
        "352c4d2482213783afde0f6b623af79958dd93c04ca2076d42d29c9902513ce9"),
    "b12x/moe/_shared/kernels/w6a8/weights.py": (
        "e1883a2625d2e7cf59d31f357676d598858086b14f63e6719e98546823eb0756",
        "481ff5ac802a949b4250d08381a5432951ac223732ffcd1ea262a23573bc9eb3"),
    "b12x/moe/_shared/kernels/w8a8/__init__.py": (
        None, "824928e196057143c0e624072253fc34bacb54c56e5629fbfd348a76358f9aa4"),
    "b12x/moe/_shared/kernels/w8a8/weights.py": (
        None, "99914f3def552fd60de823cd9a108c34669e2459f09e65b0894c2e9c0a5f3f19"),
    "b12x/moe/fused_moe/__init__.py": (
        "93520aa721c0241c7993fef38a6a56399f6af5b48d6f1f956f99cbb7a7512048",
        "2c3234bb68a1f309d7091d9b9c3be22a1fcd59b1afe66dc845251bc37e93ccfd"),
    "b12x/moe/fused_moe/_impl.py": (
        "aa7bdcd6e7392d1a7820a25c671a0eae2cfcb29264b8d127c356795c1a473aff",
        "8ea0b472d3c719a2a312c1a1b5121da8e1c6ae0d66911de1cd7068f463974913"),
    "b12x/moe/fused_moe/_tuning.py": (
        "eca0da6f9e7524629830ff214d366d614f5b554143ce871710894fd2bf955f0b",
        "894d061c8f355addbb40814b8dc395a44c5f26ac9b588b1fa37f116ea01d0e8c"),
    "b12x/moe/fused_moe/planning.py": (
        "9962d761a333f5f66f0a027ffe3af0ff6d2ae0ef10cfe893c8d86aff2693c490",
        "1072833334e55fcc3b98563393b4ed0087876555159c11e2665bcb3495b113c4"),
    "b12x/moe/fused_moe/source.py": (
        "63f9bc2afb4bb48d1a9e78d7689e7cf76a6d388f3e1f34d49af04251d10f9285",
        "ece77f963e5a25311ba325021295e8824ee338b368a4a02069a2484820d68168"),
    "b12x/moe/fused_moe/weights.py": (
        "77e8164f53b5b4e376c03a855e24b4db1e0f43970858a73b268d41b414bdb89f",
        "e81bc25453488046ef4c469ac016a0b0ada32dc157ac4cf0c11cf99e66767035"),
}


def _targets(patch):
    """Site-packages relative paths that a patch writes, in order."""
    return [line.split(" b/", 1)[1] for line in patch.read_text(encoding="utf-8").splitlines()
            if line.startswith("diff --git a/")]


def replace(read, receipt, patches=PATCHES):
    """Apply both patches to copies of the parent's files; a hunk that does not apply fails."""
    targets = [rel for patch in patches for rel in _targets(patch)]
    if sorted(targets) != sorted(FILES):
        raise ValueError("The patches and the pinned file list name different paths")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel in targets:
            if FILES[rel][0] is not None:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(read(SITE + rel))
        for patch in patches:
            result = subprocess.run(["git", "apply", "--verbose", str(patch)], cwd=root,
                                    capture_output=True, text=True)
            if result.returncode != 0:
                raise ValueError(f"{patch.name} does not apply to the parent's files:\n{result.stderr}")
        return {SITE + rel: (root / rel).read_bytes() for rel in targets}


LAYER = Layer(
    name="research-b12x-mxfp8-draft",
    purpose=("Research-only hand-merged patch of local-inference-lab/vllm#955 (9b4f167a) and "
             "local-inference-lab/b12x#449 (b8e242af) onto the image's vLLM 03c4af34 and B12X c8e46128: "
             "moe_backend b12x runs ModelOpt MXFP8 MoE experts, such as the Qwen3.8-Flash-Next MTP draft's, "
             "on B12X W8A8 MXFP8 kernels (B12X_MXFP8)"),
    replace=replace,
    provenance="/opt/sparkring/receipts/research-b12x-mxfp8-draft.json",
    pins={SITE + rel: pin for rel, pin in FILES.items()},
)

if __name__ == "__main__":
    main(LAYER)
