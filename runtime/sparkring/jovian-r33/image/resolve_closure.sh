#!/usr/bin/env bash
# Resolve and download a complete binary-only Python closure for the ARM64 candidate.
set -euo pipefail

context=$(realpath "$1")
foundation=${SPARKRING_R33_MEDIA_RUNTIME_IMAGE:-local/sparkring:r33-media-probe}
expected_id=sha256:a1a72e18ad49d99f6194a2585bfdc5f32d79180cdf2cd015c0d0d451479d6a42
test -f "$context/source-lock.partial.json"
test ! -e "$context/source-lock.json"
test "$(docker image inspect "$foundation" --format '{{.Id}}')" = "$expected_id"

docker run --rm --network host \
  -v "$context:/context" \
  "$foundation" \
  bash --noprofile --norc -ceu '
    mapfile -t wheels < /context/install-wheels.txt
    for index in "${!wheels[@]}"; do wheels[$index]="/context/wheelhouse/${wheels[$index]}"; done
    python3 -m pip install --dry-run --ignore-installed --only-binary=:all: \
      --find-links /context/wheelhouse --report /context/closure-report.json "${wheels[@]}"
    python3 /context/download_report.py \
      --report /context/closure-report.json \
      --wheelhouse /context/wheelhouse \
      --output /context/closure-downloads.json
  '
probe="$(dirname "$context")/venv-payload-probe-$(basename "$context")"
test ! -e "$probe"
mkdir -p "$probe"
docker run --rm --network none \
  -v "$context:/context:ro" \
  -v "$context/wheelhouse:/wheelhouse:ro" \
  -v "$probe:/opt/venv" \
  "$foundation" \
  bash --noprofile --norc -ceu '
    python3 -m venv /opt/venv
    mapfile -t wheels < /context/closure-install-wheels.txt
    for index in "${!wheels[@]}"; do wheels[$index]="/wheelhouse/${wheels[$index]}"; done
    /opt/venv/bin/python -m pip install --no-index --no-cache-dir --no-deps --ignore-installed --no-compile "${wheels[@]}"
    /opt/venv/bin/python -m pip check
    /opt/venv/bin/python /context/capture_installed.py \
      --closure /context/closure-downloads.json \
      --output /opt/venv/installed-python-files.json
  '
cp "$probe/installed-python-files.json" "$context/installed-python-files.json"
python3 "$context/finalize_lock.py" --context "$context"
python3 "$context/verify_context.py" --context "$context"
printf '%s\n' "$context"
