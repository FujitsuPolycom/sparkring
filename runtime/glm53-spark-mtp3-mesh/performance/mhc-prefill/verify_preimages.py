"""Compare runtime preimages with maintained packages and pinned vLLM objects."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent


def verify(vllm_checkout=None):
    manifest = json.loads((HERE / "manifest.json").read_bytes())
    compute = HERE.parent.parent / "compute"
    source_lock = json.loads((compute / "source-lock.json").read_bytes())["vllm"]
    model = "vllm/models/glm5next/nvidia/model.py"
    checkpoint = "vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py"
    archive_path = compute / source_lock["replacement_archive"]
    if (
        hashlib.sha256(archive_path.read_bytes()).hexdigest()
        != source_lock["replacement_archive_sha256"]
    ):
        raise ValueError("Compute archive identity differs")
    with tarfile.open(archive_path) as archive:
        model_source = archive.extractfile(model).read()
    expected_checkpoint = manifest["files"][checkpoint]["before_sha256"]
    checkpoint_source = (
        HERE.parent
        / "checkpoints/payload-by-sha"
        / expected_checkpoint
        / "kimi_gdn_linear_attn.py"
    ).read_bytes()
    sources = {model: model_source, checkpoint: checkpoint_source}
    if vllm_checkout is not None:
        for name, row in manifest["files"].items():
            if row["before_sha256"] is not None and name not in sources:
                sources[name] = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(vllm_checkout),
                        "show",
                        source_lock["base_revision"] + ":" + name,
                    ]
                )
    for name, source in sources.items():
        if (
            hashlib.sha256(source).hexdigest()
            != manifest["files"][name]["before_sha256"]
        ):
            raise ValueError(f"Maintained preimage differs: {name}")
    return {
        "verified_files": sorted(sources),
        "vllm_revision": source_lock["base_revision"],
        "all_eight_preimages_checked": len(sources) == 8,
        "docker_build_exercised": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-checkout", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.vllm_checkout), indent=2))
