#!/usr/bin/env bash
# Build only the local candidate after its complete context has been reviewed.
set -euo pipefail

context=$(realpath "$1")
image=${SPARKRING_R33_CANDIDATE_IMAGE:-local/sparkring:r33-arm64-candidate}
media_runtime=${SPARKRING_R33_MEDIA_RUNTIME_IMAGE:-local/sparkring:r33-media-probe}
expected_media_id=sha256:a1a72e18ad49d99f6194a2585bfdc5f32d79180cdf2cd015c0d0d451479d6a42
test -f "$context/source-lock.json"
python3 "$context/verify_context.py" --context "$context"
test "$(docker image inspect "$media_runtime" --format '{{.Id}}')" = "$expected_media_id"
printf 'candidate_context_bytes=%s\n' "$(du -sb "$context" | cut -f1)"
printf 'media_runtime_image_id=%s\n' "$expected_media_id"
printf 'media_runtime_image_bytes=%s\n' "$(docker image inspect "$media_runtime" --format '{{.Size}}')"
docker build --pull=false --build-arg MEDIA_RUNTIME="$media_runtime" \
  --file "$context/Dockerfile.candidate" --tag "$image" "$context"
printf 'candidate_image=%s\n' "$image"
printf 'candidate_image_id=%s\n' "$(docker image inspect "$image" --format '{{.Id}}')"
