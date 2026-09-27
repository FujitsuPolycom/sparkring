"""Bind prebuilt CUDA 13.4.2 and NCCL 2.32.3 artifacts to a software-layer parent image.

`toolchain_context.py` builds the CUDA toolkit stage and the patched NCCL
library from source. This preparer takes those two artifacts after they have
been built and recorded (`prebuilt_artifacts` in the lock) and writes a Docker
context that copies them over a parent image, installs the toolchain verifier
and seals the toolchain receipt. It does not invoke Docker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

if __package__:
    from .toolchain_runtime import CUDA_LIBRARY_NAMES, CUDA_ROOT, CUDA_SEARCH_ROOT, NCCL_PATH
else:
    from toolchain_runtime import CUDA_LIBRARY_NAMES, CUDA_ROOT, CUDA_SEARCH_ROOT, NCCL_PATH

HERE = Path(__file__).resolve().parent
PARENT_RECEIPT = "/opt/sparkring/receipts/external-base-installed.json"
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
LOCAL_TAG = re.compile(r"[a-z0-9][a-z0-9._/-]*:[A-Za-z0-9_][A-Za-z0-9_.-]*")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _extract_artifacts(archive, destination):
    """Extract the NCCL build output, refusing paths or links outside it."""
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("NCCL artifact path escapes the archive: " + member.name)
            if not (member.isfile() or member.isdir() or member.issym()):
                raise ValueError("NCCL artifact archive holds an unsupported entry: " + member.name)
            if member.issym():
                link = PurePosixPath(member.linkname)
                if link.is_absolute() or ".." in link.parts:
                    raise ValueError("NCCL artifact link escapes its directory: " + member.name)
        source.extractall(destination, filter="data")
    library = destination / "lib" / "libnccl.so.2.32.3"
    if not library.is_file():
        raise ValueError("NCCL artifact archive lacks lib/libnccl.so.2.32.3")
    return digest(library)


def dockerfile(lock, parent_tag, toolkit_tag, version_label):
    """Return the assembly Dockerfile; the order of layers is part of the image identity."""
    nccl = NCCL_PATH.as_posix()
    nccl_lib = NCCL_PATH.parent.as_posix()
    preloads = ":".join([nccl, *(f"{CUDA_ROOT}/lib64/{name}" for name in CUDA_LIBRARY_NAMES)])
    return "\n".join([
        f"FROM {toolkit_tag} AS toolkit",
        f"FROM {parent_tag}",
        f"COPY --from=toolkit {CUDA_ROOT}/ {CUDA_ROOT}/",
        "COPY nccl-artifacts/lib/ /opt/sparkring/toolchain/nccl/lib/",
        "COPY nccl-artifacts/include/ /opt/sparkring/toolchain/nccl/include/",
        "COPY nccl-artifacts/licenses/ /opt/sparkring/toolchain/nccl/licenses/",
        "COPY toolchain_runtime.py /opt/sparkring/toolchain/toolchain.py",
        "COPY toolchain.json /opt/sparkring/toolchain/toolchain.json",
        "COPY gpu_smoke.py /opt/sparkring/toolchain/gpu_smoke.py",
        f"RUN test -L /usr/local/cuda && ln -sfn {CUDA_ROOT} /usr/local/cuda",
        f"ENV CUDA_VERSION={lock['cuda']['version']} CUDA_HOME={CUDA_ROOT} CUDA_PATH={CUDA_ROOT}",
        f"ENV PATH={CUDA_ROOT}/bin:${{PATH}}",
        f"ENV LD_LIBRARY_PATH={CUDA_ROOT}/compat:{nccl_lib}:{CUDA_ROOT}/lib64:${{LD_LIBRARY_PATH}}",
        f"ENV LD_PRELOAD={nccl} VLLM_NCCL_SO_PATH={nccl}",
        f"ENV TRITON_PTXAS_PATH={CUDA_ROOT}/bin/ptxas",
        f"RUN mkdir -p {CUDA_SEARCH_ROOT}/nvidia/cu13 && ln -s {CUDA_ROOT}/lib64 {CUDA_SEARCH_ROOT}/nvidia/cu13/lib",
        # The addon path carries the image's startup identity and media packages.
        f"ENV PYTHONPATH={CUDA_SEARCH_ROOT}:/opt/sparkring/addons/python",
        f"ENV LD_PRELOAD={preloads}",
        f"RUN mkdir -p {CUDA_SEARCH_ROOT}/nvidia/nccl && ln -s {nccl_lib} {CUDA_SEARCH_ROOT}/nvidia/nccl/lib",
        f'RUN echo "{lock["parent"]["receipt_sha256"]}  {PARENT_RECEIPT}" | sha256sum -c -',
        f'RUN echo "{lock["prebuilt_artifacts"]["nccl_library_sha256"]}  {nccl_lib}/libnccl.so.2.32.3" | sha256sum -c -',
        "RUN python3 /opt/sparkring/toolchain/toolchain.py seal",
        'ENTRYPOINT ["python3", "/opt/sparkring/toolchain/toolchain.py"]',
        'CMD ["verify"]',
        f'LABEL org.sparkring.version="{version_label}" org.sparkring.toolchain.status="{lock["status"]}"',
        f"ENV NCCL_VERSION={lock['nccl']['version']}",
    ]) + "\n"


def prepare(lock_path, parent_release, parent_image, parent_receipt, parent_tag, nccl_artifacts, output,
            version_label, toolkit_tag=None):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    artifacts = lock["prebuilt_artifacts"]
    if lock.get("variant") != "combined":
        raise ValueError("Prebuilt assembly supports only the combined CUDA and NCCL variant")
    if not IMAGE_ID.fullmatch(parent_image):
        raise ValueError("Parent image must be a local image ID (sha256:...)")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", parent_release):
        raise ValueError("Parent release label must be a single token")
    toolkit_tag = toolkit_tag or "sparkring:installer-toolkit-" + artifacts["toolkit_image_config"][7:15]
    for tag in (parent_tag, toolkit_tag):
        if not LOCAL_TAG.fullmatch(tag) or tag.startswith("sha256:"):
            raise ValueError("Docker stages are referenced through local tags: " + tag)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", version_label):
        raise ValueError("Version label must be a single token")
    output = Path(output)
    if output.exists():
        raise ValueError("Build context already exists")
    if digest(nccl_artifacts) != artifacts["nccl_archive_sha256"]:
        raise ValueError("NCCL artifact archive hash differs")
    lock["parent"] = {"release": parent_release, "reference": parent_image,
                      "receipt_sha256": digest(parent_receipt), "publication_kind": "local_candidate"}
    output.mkdir(parents=True)
    try:
        if _extract_artifacts(nccl_artifacts, output / "nccl-artifacts") != artifacts["nccl_library_sha256"]:
            raise ValueError("NCCL library hash differs from the recorded artifact")
    except Exception:
        shutil.rmtree(output)
        raise
    shutil.copyfile(HERE / "toolchain_runtime.py", output / "toolchain_runtime.py")
    shutil.copyfile(HERE / "toolchain_gpu_smoke.py", output / "gpu_smoke.py")
    (output / "toolchain.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "Dockerfile").write_text(dockerfile(lock, parent_tag, toolkit_tag, version_label),
                                       encoding="utf-8", newline="\n")
    return {"context": str(output), "parent": lock["parent"], "toolkit_tag": toolkit_tag,
            "toolkit_image_config": artifacts["toolkit_image_config"],
            "dockerfile_sha256": digest(output / "Dockerfile")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lock", type=Path, default=HERE / "cuda134-nccl232-installer.json")
    parser.add_argument("--parent-release", required=True, help="label recorded as the parent's release")
    parser.add_argument("--parent-image", required=True, help="local image ID of the software-layer parent")
    parser.add_argument("--parent-receipt", required=True, type=Path,
                        help="the parent's " + PARENT_RECEIPT + ", copied out of the image")
    parser.add_argument("--parent-tag", required=True, help="local tag that names the parent image")
    parser.add_argument("--toolkit-tag", help="local tag of the recorded CUDA toolkit stage image")
    parser.add_argument("--nccl-artifacts", required=True, type=Path,
                        help="archive of the NCCL build output written by build-nccl.sh")
    parser.add_argument("--version-label", required=True, help="org.sparkring.version label value")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.lock, args.parent_release, args.parent_image, args.parent_receipt,
                             args.parent_tag, args.nccl_artifacts, args.output, args.version_label,
                             args.toolkit_tag), indent=2))


if __name__ == "__main__":
    main()
