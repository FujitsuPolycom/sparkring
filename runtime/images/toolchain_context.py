"""Prepare an opt-in CUDA/NCCL layer over the immutable RC4 serving image."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

if __package__:
    from .toolchain_runtime import CUDA_LIBRARY_NAMES, CUDA_SEARCH_ROOT, vendor_search_aliases
else:
    from toolchain_runtime import CUDA_LIBRARY_NAMES, CUDA_SEARCH_ROOT, vendor_search_aliases

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(lock_path, source_archive, output, variant="combined"):
    lock = json.loads(Path(lock_path).read_text())
    if variant not in ("combined", "cuda", "nccl"):
        raise ValueError("variant must be combined, cuda or nccl")
    if output.exists():
        raise ValueError("Build context already exists")
    patch = HERE.parents[1] / lock["nccl"]["patch"]
    if digest(patch) != lock["nccl"]["patch_sha256"]:
        raise ValueError("NCCL routing patch hash differs")
    if digest(source_archive) != lock["nccl"]["archive_sha256"]:
        raise ValueError("NCCL source archive hash differs")
    with tarfile.open(source_archive) as source:
        for name in ("src/transport/generic.cc", "src/transport/net_ib/connect.cc"):
            if b"\r\n" in source.extractfile(name).read():
                raise ValueError("NCCL source archive requires LF; export with git -c core.autocrlf=false archive")
    cuda_enabled = variant in ("combined", "cuda")
    nccl_enabled = variant in ("combined", "nccl")
    lock["variant"] = variant
    output.mkdir(parents=True)
    shutil.copyfile(source_archive, output / "nccl.tar")
    shutil.copyfile(patch, output / "nccl.patch")
    shutil.copyfile(HERE / "toolchain_runtime.py", output / "toolchain.py")
    (output / "toolchain.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "build-nccl.sh").write_text("""#!/bin/bash
set -euo pipefail
mkdir /tmp/nccl
tar -xf /inputs/nccl.tar -C /tmp/nccl
cd /tmp/nccl
git apply --check /inputs/nccl.patch
git apply /inputs/nccl.patch
g++ -std=c++11 -O2 -Wall -Wextra -Werror tests/routing_handle/compat.cc -o /tmp/routing-test
/tmp/routing-test
make -j${BUILD_JOBS:-2} src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
mkdir -p /output/lib /output/include /output/licenses
cp -a build/lib/libnccl.so* /output/lib/
cp -a build/include/. /output/include/
cp LICENSE.txt ThirdPartyNotices.txt /output/licenses/
cp /inputs/toolchain.json /inputs/nccl.patch /output/
/usr/local/cuda/bin/nvcc --version > /output/nvcc.txt
dpkg-query -W > /output/packages.tsv
sha256sum /output/lib/libnccl.so.2.32.3 > /output/NCCL-SHA256SUMS
""", encoding="utf-8", newline="\n")
    lines = []
    if cuda_enabled:
        lines += [f'FROM {lock["cuda"]["base"]} AS cuda134',
                  'RUN apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages '
                  f'cuda-toolkit-13-4={lock["cuda"]["toolkit_package"]} '
                  f'cuda-compat-13-4={lock["cuda"]["compat_package"]} '
                  f'libcublas-13-4={lock["cuda"]["cublas_package"]} '
                  f'libcublas-dev-13-4={lock["cuda"]["cublas_package"]}',
                  'ENV CUDA_VERSION=13.4.2',
                  'FROM cuda134 AS toolchain_builder',
                  'RUN apt-get update && apt-get install -y --no-install-recommends git g++ make python3 libibverbs-dev libnuma-dev']
    lines += [f'FROM {lock["parent"]["reference"]} AS runtime']
    if cuda_enabled:
        lines += ['COPY --from=cuda134 /usr/local/cuda-13.4/ /usr/local/cuda-13.4/',
                  'RUN test -L /usr/local/cuda && ln -sfn /usr/local/cuda-13.4 /usr/local/cuda',
                  'ENV CUDA_HOME=/usr/local/cuda-13.4 CUDA_PATH=/usr/local/cuda-13.4 CUDA_VERSION=13.4.2',
                  'ENV PATH=/usr/local/cuda-13.4/bin:${PATH}',
                  'ENV LD_LIBRARY_PATH=/usr/local/cuda-13.4/compat:/usr/local/cuda-13.4/lib64:${LD_LIBRARY_PATH}',
                  'ENV TRITON_PTXAS_PATH=/usr/local/cuda-13.4/bin/ptxas']
    if nccl_enabled:
        lines += ['FROM runtime AS nccl_build',
                  'COPY nccl.tar nccl.patch build-nccl.sh toolchain.json /inputs/',
                  'ARG BUILD_JOBS=2',
                  'RUN BUILD_JOBS=${BUILD_JOBS} bash /inputs/build-nccl.sh',
                  'FROM runtime AS candidate',
                  'COPY --from=nccl_build /output/lib/ /opt/sparkring/toolchain/nccl/lib/',
                  'COPY --from=nccl_build /output/include/ /opt/sparkring/toolchain/nccl/include/',
                  'COPY --from=nccl_build /output/licenses/ /opt/sparkring/toolchain/nccl/licenses/',
                  'ENV LD_LIBRARY_PATH=/opt/sparkring/toolchain/nccl/lib:${LD_LIBRARY_PATH}',
                  'ENV LD_PRELOAD=/opt/sparkring/toolchain/nccl/lib/libnccl.so.2 '
                  'VLLM_NCCL_SO_PATH=/opt/sparkring/toolchain/nccl/lib/libnccl.so.2 '
                  'NCCL_LOCAL_INFERENCE_PATH=/opt/sparkring/toolchain/nccl/lib/libnccl.so.2 '
                  'NCCL_LIB_DIR=/opt/sparkring/toolchain/nccl/lib']
    else:
        lines += ['FROM runtime AS candidate']
    if cuda_enabled:
        preloads = ":".join("/usr/local/cuda-13.4/lib64/" + name for name in CUDA_LIBRARY_NAMES)
        lines += [f'ENV LD_PRELOAD=${{LD_PRELOAD}}:{preloads}']
    for relative, target in vendor_search_aliases(lock).items():
        alias = f'{CUDA_SEARCH_ROOT}/{relative}'
        lines += [f'RUN mkdir -p {alias.rsplit("/", 1)[0]} && ln -s {target} {alias}']
    lines += [f'ENV PYTHONPATH={CUDA_SEARCH_ROOT}:${{PYTHONPATH}}']
    lines += ['COPY toolchain.py toolchain.json /opt/sparkring/toolchain/',
              'RUN python3 /opt/sparkring/toolchain/toolchain.py seal',
              'LABEL org.sparkring.toolchain.status="research-only" '
              f'org.sparkring.toolchain.variant="{variant}"',
              'ENTRYPOINT ["python3", "/opt/sparkring/toolchain/toolchain.py"]',
              'CMD ["verify"]']
    (output / "Dockerfile").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return {"context": str(output), "variant": variant, "dockerfile_sha256": digest(output / "Dockerfile")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=HERE / "cuda134-nccl232.json")
    parser.add_argument("--nccl-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", choices=("combined", "cuda", "nccl"), default="combined")
    args = parser.parse_args()
    print(json.dumps(prepare(args.lock, args.nccl_source, args.output, args.variant)))


if __name__ == "__main__":
    main()
