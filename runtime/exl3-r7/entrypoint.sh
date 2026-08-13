#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "FATAL: $*" >&2
  exit 2
}

if [[ -n "${SPARKRING_EXPLICITLY_UNSET:-}" ]]; then
  IFS=',' read -r -a unset_names <<<"${SPARKRING_EXPLICITLY_UNSET}"
  for name in "${unset_names[@]}"; do
    if [[ ! "${name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      fail "invalid explicitly-unset environment name: ${name}"
    fi
    unset "${name}"
  done
  unset SPARKRING_EXPLICITLY_UNSET
fi

/opt/venv/bin/python /opt/sparkring-r7/verify_runtime.py

if [[ "$#" -eq 0 ]]; then
  fail "pass a launcher or vLLM command to the R7 image"
fi

if [[ "$1" == "serve" ]]; then
  set -- /opt/venv/bin/vllm "$@"
fi

exec "$@"
