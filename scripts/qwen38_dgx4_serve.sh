#!/usr/bin/env bash
# Compatibility name of qwen38_ring4_serve.sh: start one rank of the Qwen3.8-27B
# EXL3 K5/K6 four-Spark (ring of four) profile ("dgx4" abbreviated four DGX Sparks).
# The image pins (runtime/qwen38/pins.json, a preserved input), the recipe's
# container_launcher and the guide's --entrypoint name /ws/qwen38_dgx4_serve.sh, and
# images built before qwen38_ring4_serve.sh existed carry only that name.
# TODO: remove this wrapper once the image pins name /ws/qwen38_ring4_serve.sh, no
# recipe or guide names this file and a release has shipped both names.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/qwen38_ring4_serve.sh" "$@"
