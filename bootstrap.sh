#!/usr/bin/env bash
# Install the SparkRing operator command on a blank DGX Spark.
set -euo pipefail

REPOSITORY="https://github.com/FujitsuPolycom/sparkring.git"
REF="main"
INSTALL_DIR="${HOME}/.local/share/sparkring"
BIN_DIR="${HOME}/.local/bin"
ASSUME_YES=0

usage() {
  cat <<'EOF'
usage: bash bootstrap.sh [--ref BRANCH_OR_TAG] [--install-dir DIR] [--yes]

Downloads SparkRing, verifies local prerequisites, and installs the
`sparkring` command under ~/.local/bin. It does not configure networking or
contact another Spark.
EOF
}

while (($#)); do
  case "$1" in
    --ref)
      REF=${2:?--ref requires a value}
      shift 2
      ;;
    --install-dir)
      INSTALL_DIR=${2:?--install-dir requires a value}
      shift 2
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for command_name in git python3 ssh ssh-keygen ssh-keyscan ssh-copy-id; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing required command: $command_name" >&2
    exit 1
  fi
done

if ! python3 -c 'import yaml' >/dev/null 2>&1; then
  echo "PyYAML is required (Ubuntu package: python3-yaml)."
  if ((ASSUME_YES == 0)); then
    read -r -p "Install python3-yaml with apt? Type yes: " answer
    if [[ ${answer,,} != yes ]]; then
      echo "installation cancelled" >&2
      exit 1
    fi
  fi
  sudo apt-get update
  sudo apt-get install -y python3-yaml
fi

if [[ -e "$INSTALL_DIR" ]]; then
  if [[ ! -d "$INSTALL_DIR/.git" ]]; then
    echo "refusing to replace non-git path: $INSTALL_DIR" >&2
    exit 1
  fi
  if [[ -n $(git -C "$INSTALL_DIR" status --porcelain) ]]; then
    echo "refusing to update dirty managed checkout: $INSTALL_DIR" >&2
    exit 1
  fi
  # A single-branch clone's fetch configuration excludes other branches.
  # Fetch the requested branch or tag explicitly before selecting it.
  if git -C "$INSTALL_DIR" ls-remote --exit-code --heads origin "refs/heads/$REF" >/dev/null; then
    git -C "$INSTALL_DIR" fetch origin "refs/heads/$REF:refs/remotes/origin/$REF"
    git -C "$INSTALL_DIR" checkout -B "$REF" "origin/$REF"
  else
    git -C "$INSTALL_DIR" fetch origin "refs/tags/$REF:refs/tags/$REF"
    git -C "$INSTALL_DIR" checkout --detach "refs/tags/$REF"
  fi
else
  mkdir -p "$(dirname "$INSTALL_DIR")"
  git clone --branch "$REF" --single-branch "$REPOSITORY" "$INSTALL_DIR"
fi

mkdir -p "$BIN_DIR"
launcher="$BIN_DIR/sparkring"
if [[ -L "$launcher" ]]; then
  if [[ $(readlink "$launcher") == "$INSTALL_DIR/scripts/sparkring.py" ]]; then
    rm "$launcher"
  else
    backup="$launcher.before-sparkring-bootstrap-$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$launcher" "$backup"
    echo "backed up existing launcher symlink to $backup"
  fi
elif [[ -e "$launcher" ]]; then
  backup="$launcher.before-sparkring-bootstrap-$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$launcher" "$backup"
  echo "backed up existing launcher to $backup"
fi
# The launcher runs the tracked script through python3, so the checkout keeps
# its committed file modes and stays clean for later --ref updates.
printf '#!/usr/bin/env bash\nexec python3 %q "$@"\n' \
  "$INSTALL_DIR/scripts/sparkring.py" > "$launcher"
chmod 0755 "$launcher"

echo
echo "SparkRing installed at $INSTALL_DIR"
echo "Command installed at $launcher"
case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *) echo "Add this to your shell profile: export PATH=\"${BIN_DIR}:\$PATH\"" ;;
esac
echo
echo "Check this Spark first:"
echo "  sparkring host check"
echo
echo "Then start a new ring on the head Spark:"
echo "  sparkring cluster init --size 4"
