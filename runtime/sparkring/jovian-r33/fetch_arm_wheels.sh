#!/usr/bin/env bash
# Fetch R33 runtime wheels whose published ARM64 artifacts were independently audited.
set -euo pipefail

out=${1:?usage: fetch_arm_wheels.sh OUTPUT_DIRECTORY}
test ! -e "$out"
mkdir -p "$out"
fetch() {
  local sha=$1 name=$2 url=$3
  curl --fail --location --retry 3 --output "$out/$name" "$url"
  printf '%s  %s\n' "$sha" "$out/$name" | sha256sum -c -
}
fetch 7171f810887e7cd1a4763974d5a1f2e1466692404315bb70705e0f49fb3a28e0 \
  torchaudio-2.11.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl \
  'https://download-r2.pytorch.org/whl/cu130/torchaudio-2.11.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl'
fetch 011f6ed28d60bae9fdc2147be5b6a67e6cce9a71cbc4ad697ecf27d22561f045 \
  torchcodec-0.16.0+cu130-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl \
  'https://download.pytorch.org/whl/cu130/torchcodec-0.16.0%2Bcu130-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl'
fetch bad9e25f494abdcfa8f9dffa33a840509eda3ffcdf6e7cf6465d73be307c0c82 \
  pynvvideocodec-2.0.4-cp312-cp312-manylinux_2_28_aarch64.whl \
  'https://files.pythonhosted.org/packages/38/1c/78f6fdf85133157a6a3405eab5ef4c2bc8048194dbda1c91bb9b8645bb36/pynvvideocodec-2.0.4-cp312-cp312-manylinux_2_28_aarch64.whl'
fetch 952209cf4b29a54e6b6e7e088be9d4a40f24792b06fdfa330de8077b9926a7d9 \
  tokenspeed_mla-0.1.8-py3-none-manylinux_2_28_aarch64.whl \
  'https://files.pythonhosted.org/packages/1e/65/81d7e9f14472bc4c6abb576c9b1edd8e40ab01832027c4e647bfe2890749/tokenspeed_mla-0.1.8-py3-none-manylinux_2_28_aarch64.whl'
fetch 988e8b9da41e3679ae2816bdfb51acab53c05cc56254b5ca98e847f4efcf24fd \
  humming_kernels-0.1.12-py3-none-manylinux_2_28_aarch64.whl \
  'https://files.pythonhosted.org/packages/58/0f/01f871dc26f8d7e1683df9464890d48cfd5c4e46c4a264d7ee0868749197/humming_kernels-0.1.12-py3-none-manylinux_2_28_aarch64.whl'
sha256sum "$out"/*.whl > "$out/SHA256SUMS"
