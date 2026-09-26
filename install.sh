#!/usr/bin/env bash
# Build the SparkRing Debian package from a Git ref, install it on this Spark
# (Node A), then run `sparkring install` to set up the cabled Sparks and start
# a model.
set -euo pipefail

REPOSITORY="https://github.com/FujitsuPolycom/sparkring.git"
# The branch that publishes this script. A copy fetched from another branch,
# tag or commit must be run with a matching --ref.
REF="one-command-installer"
INSTALL_ARGS=()

usage() {
  cat <<'EOF'
usage: install.sh [--ref BRANCH_TAG_OR_COMMIT] [--repository URL] [SPARKRING_INSTALL_OPTION ...]

Run on Node A, the Spark connected to your network, as a user with sudo.
Clones the SparkRing repository at --ref, builds its Debian package, installs
the package with apt, then runs `sudo sparkring install` with every other
option, for example --profile qwen38-flash-next-tp2. The final command asks
for approval before it changes any Spark.

  curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh \
    | bash -s -- --profile qwen38-flash-next-tp2
EOF
}

while (($#)); do
  case "$1" in
    --ref)
      REF=${2:?--ref requires a value}
      shift 2
      ;;
    --repository)
      REPOSITORY=${2:?--repository requires a value}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      INSTALL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ $(uname -m) != aarch64 ]]; then
  echo "SparkRing installs on DGX Spark (ARM64) hosts; this host is $(uname -m)." >&2
  exit 1
fi
for command_name in git python3 dpkg-deb apt-get; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "missing required command: $command_name" >&2
    exit 1
  fi
done
SUDO=()
if ((EUID != 0)); then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "Run as root or as a user with sudo." >&2
    exit 1
  fi
  SUDO=(sudo)
fi

WORK=$(mktemp -d /var/tmp/sparkring-install.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
# apt reads the local package as its unprivileged _apt user.
chmod 0755 "$WORK"

echo "Fetching SparkRing $REF from $REPOSITORY"
# The package embeds the commit's history as a Git bundle, so the clone must
# be complete, not shallow.
if [[ $REF =~ ^[0-9a-f]{40}$ ]]; then
  git init --quiet "$WORK/source"
  git -C "$WORK/source" remote add origin "$REPOSITORY"
  git -C "$WORK/source" fetch --quiet origin "$REF"
  git -C "$WORK/source" checkout --quiet --detach "$REF"
else
  git clone --quiet --branch "$REF" --single-branch "$REPOSITORY" "$WORK/source"
fi
echo "Source revision: $(git -C "$WORK/source" rev-parse HEAD)"

echo "Building the SparkRing package"
python3 "$WORK/source/scripts/build_deb.py" --output "$WORK/dist" >/dev/null
package=$(echo "$WORK"/dist/sparkring_*_arm64.deb)
chmod 0644 "$package"

echo "Installing $(basename "$package")"
# A package built from an earlier commit carries a lower version; installing
# it replaces the installed package.
"${SUDO[@]}" apt-get install --yes --allow-downgrades "$package"

echo
echo "Starting: sudo sparkring install ${INSTALL_ARGS[*]}"
# When this script arrives through a pipe, its standard input is the script
# itself; the installer's questions are read from the terminal instead.
if [[ ! -t 0 ]] && { : </dev/tty; } 2>/dev/null; then
  "${SUDO[@]}" /usr/bin/sparkring install "${INSTALL_ARGS[@]}" </dev/tty
else
  "${SUDO[@]}" /usr/bin/sparkring install "${INSTALL_ARGS[@]}"
fi
