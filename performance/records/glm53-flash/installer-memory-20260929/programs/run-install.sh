#!/bin/bash
# Measures one configuration's startup: installs the GLM TP4 profile from a local branch while 1-second MemAvailable samplers run on all four Sparks.
# Bundles BRANCH of the checkout REPO, copies the bundle to Node A, starts the samplers (start-samplers.sh TAG 3600),
# runs install.sh from the bundle, and saves the install log, the rank-0 container log and phase epochs in OUT_ROOT/TAG.
# Environment: REPO = SparkRing checkout holding BRANCH; BRANCH (default test/glm-tp4-memory);
#              NODE_A = SSH target of Node A; WORKERS = worker addresses reachable from Node A as root; REMOTE_DIR = directory on Node A holding the sampler scripts;
#              OUT_ROOT = directory for per-configuration results.
# Usage: run-install.sh TAG [extra installer arguments...]
set -u
tag=$1
shift
repo=${REPO:?set REPO to the SparkRing checkout}
branch=${BRANCH:-test/glm-tp4-memory}
node_a=${NODE_A:?set NODE_A to the SSH target of Node A}
workers=${WORKERS:?set WORKERS to the worker addresses reachable from Node A}
remote_dir=${REMOTE_DIR:-\$HOME/tp4mem}
out=${OUT_ROOT:?set OUT_ROOT}/$tag
mkdir -p "$out"
git -C "$repo" log --oneline -1 > "$out/commit.txt"
git -C "$repo" show HEAD:profiles/glm53-flash-nvfp4-spark-tp4/config.json > "$out/config.json"
bundle=$(mktemp -u).bundle
git -C "$repo" bundle create "$bundle" "$branch" 2>/dev/null
scp -q "$bundle" "$node_a:sparkring-tp4glm.bundle"
rm -f "$bundle"
ssh "$node_a" "WORKERS='$workers' REMOTE_DIR=$remote_dir sh $remote_dir/start-samplers.sh $tag 3600" > "$out/samplers.txt" 2>&1
echo "install_start $(date +%s)" > "$out/phases.txt"
ssh "$node_a" "rm -rf ~/srtp4glm && git clone -q --branch $branch ~/sparkring-tp4glm.bundle ~/srtp4glm && cd ~/srtp4glm && git log --oneline -1 && bash install.sh --repository \"\$HOME/sparkring-tp4glm.bundle\" --ref $branch --profile glm53-flash-nvfp4-spark-tp4 --yes $* > /tmp/tp4glm-install.log 2>&1; echo installer_exit=\$?"
echo "install_end $(date +%s)" >> "$out/phases.txt"
ssh "$node_a" 'cat /tmp/tp4glm-install.log' > "$out/install.log"
ssh "$node_a" 'docker logs -t $(docker ps -qf label=io.sparkring.rank=0) 2>&1' > "$out/rank0.log"
grep -E "GPU KV cache size|compiler workers|Model loading took|init engine|Application startup complete|reserved .* KV" "$out/rank0.log" | cut -c1-260
tail -3 "$out/install.log"
