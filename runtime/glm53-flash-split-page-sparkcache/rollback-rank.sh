#!/usr/bin/env bash
set -euo pipefail

rank="${1:?usage: rollback-rank.sh RANK}"
case "${rank}" in
  0|1|2|3) ;;
  *) printf 'rank must be 0, 1, 2, or 3\n' >&2; exit 2 ;;
esac

qualified="glm53-pr535-sc59ac-c8-01-r${rank}"
rollback="glm53-pr535-sc78-hotpatch-c8-qualified-r${rank}"

docker container inspect "${qualified}" >/dev/null
docker container inspect "${rollback}" >/dev/null

docker stop "${qualified}" >/dev/null
docker start "${rollback}" >/dev/null
printf 'stopped=%s\nstarted=%s\n' "${qualified}" "${rollback}"
