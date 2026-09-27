#!/usr/bin/env bash
# Build the SparkRing Debian package from a Git ref, install it on this Spark
# (Node A), then run `sparkring install` to set up the cabled Sparks and start
# a model.
#
# Standard output carries only results: with --json it holds exactly one
# sparkring-install-result/v1 document, from `sparkring install` or, when this
# script stops before running it, from this script. Progress, questions and
# apt's output go to standard error.
set -euo pipefail

REPOSITORY="https://github.com/FujitsuPolycom/sparkring.git"
# The branch that publishes this script. A copy fetched from another branch,
# tag or commit must be run with a matching --ref.
REF="one-command-installer"
YES=0
PLAN=0
JSON=0
PACKAGE_ONLY=0

say() {
  printf '%s\n' "$*" >&2
}

json_string() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\n'/\\n}
  value=${value//$'\r'/\\r}
  value=${value//$'\t'/\\t}
  printf '"%s"' "$value"
}

# Stop before `sparkring install` runs, with the states and exit statuses of
# its own results: failed (2) names the stage that failed, needs_input (3) the
# missing input. DETAILS is an optional JSON object.
stop() {
  local state=$1 key=$2 message=$3 details=${4:-} name=stage
  say "$message"
  if ((JSON)); then
    [[ $state == needs_input ]] && name=field
    printf '{"schema": "sparkring-install-result/v1", "state": %s, "%s": %s, "message": %s%s}\n' \
      "$(json_string "$state")" "$name" "$(json_string "$key")" "$(json_string "$message")" \
      "${details:+, \"details\": $details}"
  fi
  [[ $state == needs_input ]] && exit 3
  exit 2
}

INSTALL_ARGS=()

usage() {
  cat <<'EOF'
usage: install.sh [--ref BRANCH_TAG_OR_COMMIT] [--repository URL] [--package-only] [SPARKRING_INSTALL_OPTION ...]

Run on Node A, the Spark connected to your network, as a user with sudo.
Clones the SparkRing repository at --ref and builds its Debian package, asks
before installing the package on this Spark, then runs `sudo sparkring install`
with every other option, for example --profile qwen38-flash-next-tp2. That
command asks for approval before it changes any Spark. --yes approves both.

--package-only stops after installing the package, before any other Spark or
model changes; `sudo sparkring install --plan` can then review the rest.
--plan installs nothing: when this Spark already has the package version just
built, `sparkring install --plan` prints the plan; otherwise the script stops
and names both versions. --json prints one result document on standard output.

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
    --package-only)
      PACKAGE_ONLY=1
      shift
      ;;
    *)
      case "$1" in
        --yes) YES=1 ;;
        --plan) PLAN=1 ;;
        --json) JSON=1 ;;
      esac
      INSTALL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ $(uname -m) != aarch64 ]]; then
  stop failed requirements "SparkRing installs on DGX Spark (ARM64) hosts; this host is $(uname -m)."
fi
if ((PACKAGE_ONLY && PLAN)); then
  stop failed arguments "--package-only installs the package and --plan installs nothing; use one of them."
fi
for command_name in git python3 dpkg dpkg-deb dpkg-query apt-get; do
  command -v "$command_name" >/dev/null 2>&1 || stop failed requirements "missing required command: $command_name"
done
SUDO=()
if ((EUID != 0)); then
  command -v sudo >/dev/null 2>&1 || stop failed requirements "Run as root or as a user with sudo."
  SUDO=(sudo)
fi

WORK=$(mktemp -d /var/tmp/sparkring-install.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
# apt reads the local package as its unprivileged _apt user.
chmod 0755 "$WORK"

say "Fetching SparkRing $REF from $REPOSITORY"
# The package embeds the commit's history as a Git bundle, so the clone must
# be complete, not shallow.
if [[ $REF =~ ^[0-9a-f]{40}$ ]]; then
  { git init --quiet "$WORK/source" \
      && git -C "$WORK/source" remote add origin "$REPOSITORY" \
      && git -C "$WORK/source" fetch --quiet origin "$REF" \
      && git -C "$WORK/source" checkout --quiet --detach "$REF"; } >&2 \
    || stop failed fetch "Could not fetch SparkRing $REF from $REPOSITORY."
else
  git clone --quiet --branch "$REF" --single-branch "$REPOSITORY" "$WORK/source" >&2 \
    || stop failed fetch "Could not fetch SparkRing $REF from $REPOSITORY."
fi
revision=$(git -C "$WORK/source" rev-parse HEAD)
say "Source revision: $revision"

say "Building the SparkRing package"
python3 "$WORK/source/scripts/build_deb.py" --output "$WORK/dist" >/dev/null \
  || stop failed build "Could not build the SparkRing package from $revision."
package=$(echo "$WORK"/dist/sparkring_*_arm64.deb)
chmod 0644 "$package"
version=$(dpkg-deb --field "$package" Version)
installed=$(dpkg-query --show --showformat='${db:Status-Abbrev}${Version}' sparkring 2>/dev/null || true)
if [[ $installed == "ii "* ]]; then
  installed=${installed#ii }
else
  installed=""
fi

if ((PLAN)); then
  if [[ $installed != "$version" ]]; then
    stop needs_input package "--plan changes nothing on this Spark, and the plan needs SparkRing $version installed; this Spark has ${installed:-no SparkRing package}. Run the command without --plan to install it, then plan again." \
      "{\"built\": $(json_string "$version"), \"installed\": $(if [[ -n $installed ]]; then json_string "$installed"; else printf null; fi)}"
  fi
  say "SparkRing $version is installed; planning with it."
elif [[ $installed == "$version" ]]; then
  say "SparkRing $version is already installed."
else
  if [[ -z $installed ]]; then
    say "Install SparkRing $version on this Spark."
  elif dpkg --compare-versions "$version" lt "$installed"; then
    say "Replace SparkRing $installed on this Spark with the earlier version $version."
  else
    say "Replace SparkRing $installed on this Spark with $version."
  fi
  say "The package installs /usr/bin/sparkring, initializes this Spark's SparkRing node, enables and starts avahi-daemon and lldpd for discovery, and restarts sparkring-agent. It changes no other Spark."
  if ((!YES)); then
    if ! { : </dev/tty; } 2>/dev/null; then
      stop needs_input approval "Installing the SparkRing package needs approval: run the command in a terminal, or pass --yes."
    fi
    answer=""
    read -r -p "Install the package? [Y/n] " answer </dev/tty || true
    [[ $answer =~ ^([Yy]([Ee][Ss])?)?$ ]] || stop needs_input approval "The SparkRing package was not installed."
  fi
  say "Installing $(basename "$package")"
  "${SUDO[@]}" apt-get install --yes --allow-downgrades "$package" >&2 \
    || stop failed package "apt could not install $(basename "$package")."
fi

if ((PACKAGE_ONLY)); then
  say "SparkRing $version is installed on this Spark; no other Spark and no model changed."
  say "Review the rest with: sudo sparkring install --profile PROFILE --plan"
  if ((JSON)); then
    printf '{"schema": "sparkring-install-result/v1", "state": "package-installed", "version": %s, "previous": %s, "changed": %s, "source_revision": %s}\n' \
      "$(json_string "$version")" "$(if [[ -n $installed ]]; then json_string "$installed"; else printf null; fi)" \
      "$([[ $installed == "$version" ]] && printf false || printf true)" "$(json_string "$revision")"
  fi
  exit 0
fi

say ""
say "Starting: sudo sparkring install ${INSTALL_ARGS[*]}"
rm -rf "$WORK"
trap - EXIT
# When this script arrives through a pipe, its standard input is the script
# itself; the installer's questions are read from the terminal instead.
if [[ ! -t 0 ]] && { : </dev/tty; } 2>/dev/null; then
  exec "${SUDO[@]}" /usr/bin/sparkring install "${INSTALL_ARGS[@]}" </dev/tty
else
  exec "${SUDO[@]}" /usr/bin/sparkring install "${INSTALL_ARGS[@]}"
fi
