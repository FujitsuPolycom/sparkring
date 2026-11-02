#!/usr/bin/env bash
# Run from the SparkRing repository root on an owner-selected build host.
set -euo pipefail
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/sparkring-sparkcache.XXXXXXXX")
parent='sha256:3cc9a81fd344fe52fe9cc261b46dc051e3c3f494e51b8142846a5f854835c502'
docker image inspect "$parent" >/dev/null
docker run --rm --pull never --network none --entrypoint python3 \
  --mount "type=bind,source=$PWD/integrations/sparkcache/8abbe9da67b8/sparkcache/runtime_patches/verify_lease_contract.py,target=/verify.py,readonly" \
  --mount "type=bind,source=$PWD/integrations/sparkcache/gab86b70/runtime_patches/vllm-connector-jobs-gab86b70.json,target=/contract.json,readonly" \
  "$parent" /verify.py --vllm-root /usr/local/lib/python3.12/dist-packages --contract /contract.json
docker run --rm -i --pull never --network none --entrypoint sha256sum "$parent" --check - <<'HASHES'
d89c9fdae8dc99ae3f7a151cc3dd9e92fdc8fd0b994069fc263027fd4d056c93  /opt/sparkring/sparkcache/lib/libspark_cache_placement.so
b50db45123c4faf0fdfb021da0a0f1b7f329869ae5dc13dcc7432dd4ce765b68  /opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so
HASHES
docker run --rm --pull never --network none --entrypoint cat "$parent" \
  /opt/sparkring/receipts/external-base-installed.json > "$build_dir/base.json"
docker run --rm --pull never --network none --entrypoint cat "$parent" \
  /opt/sparkring/toolchain/installed.json > "$build_dir/toolchain.json"
docker run --rm --pull never --network none --entrypoint cat "$parent" \
  /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/metrics.py > "$build_dir/metrics.py"
python3 runtime/images/prepare_sparkcache.py \
  --base-receipt "$build_dir/base.json" --toolchain-receipt "$build_dir/toolchain.json" \
  --output "$build_dir/context" --metrics-source "$build_dir/metrics.py" \
  --sources runtime/images/compositions/installer-sparkcache-4a17f1f5198e/sources.json
python3 runtime/images/derived_layer.py build --context "$build_dir/context" \
  --tag 'sparkring:sparkcache-4a17f1f5198e-status038' \
  --name 'dev-20261016-sparkcache-4a17f1f5198e-status038' \
  --output 'runtime/images/compositions/installer-sparkcache-4a17f1f5198e/built-image.json' \
  --profiles 'deepseek-v41-flash-sparkcache-tp4,deepseek-v41-flash-tp4,deepseek-v41-flash-tp8,glm53-flash-csf-tp8,glm53-flash-nvfp4-spark-tp2,glm53-flash-nvfp4-spark-tp2-dcp2,glm53-flash-nvfp4-spark-tp4,glm53-flash-nvfp4-sparkcache-crc32-tp2,glm53-flash-nvfp4-sparkcache-crc32-tp4,glm53-nvfp4-csf-tp6,glm53-nvfp4-tp8,glm53-nvfp4-tp8-dcp1,mimo-v26-flash-mopd-tp2,mimo-v26-flash-mopd-tp4,qwen38-flash-next-csf-sparkcache-tp2,qwen38-flash-next-qad-tp4,qwen38-flash-next-step5500-sparkcache-tp4,qwen38-flash-next-tp2,swift15-qwen38-flash-next-tp2,swift15-qwen38-flash-next-tp4'
docker run --rm --pull never --network none --entrypoint python3 'sparkring:sparkcache-4a17f1f5198e-status038' \
  -m sparkcache.runtime_patches.verify_lease_contract \
  --vllm-root /usr/local/lib/python3.12/dist-packages \
  --contract /usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/vllm-connector-jobs-gab86b70.json
printf 'Build context and receipts: %s\n' "$build_dir"
