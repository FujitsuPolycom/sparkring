# DeepSeek V4.1 SparkCache: H0-H11 hardware handoff

Status: **Experimental**. Worker topology was recorded on 2026-10-15;
serving qualification of the role-independent alignment fix is pending.
These are acceptance gates for `deepseek-v41-flash-sparkcache-tp4`, SparkCache `4a17f1f5198e`, TP4/DCP1
on positions 0-3. Serving stages **STOP SERVING** there and require an authorized
maintenance window. Keep positions 4-7 and their Qwen deployment unchanged.
Never assume a Spark is idle. Use `sudo -n` for Spark-side operations.

## Inputs and rollback

Transfer this committed branch with a Git bundle, then clone the bundle into
an isolated checkout on the authorized Linux controller. No push is needed.
Keep the prior deployment directory (including its lock and source bundle),
image, package and serving selection. Export its `deployment.lock.json` as
`/var/tmp/dsv41-prior.json`. Record node identities, cabling, GPU clocks and
the four containers before stopping them. The copied lock must describe the
existing DeepSeek TP4 deployment on positions 0-3, with SIRCL and NCCL disabled.
Retain its original directory for rollback with its original source version.

The source-rendered path below requires Python 3.12, Git, PyYAML, passwordless
authorized SSH, Docker/Compose, NVIDIA container tooling and the already
prepared SparkRing fabric and node identities. It distributes committed source
to only the selected four hosts. It does not update the ring-wide package,
controller selection, relay configuration or network services. Run no managed
recovery process that would restart the prior model during these tests; record
and restore any recovery setting changed for the window.

## H0: build and probe without GPU access

On an authorized ARM64 build host with the pinned parent image local, run:

```bash
sudo -n bash runtime/images/compositions/installer-sparkcache-4a17f1f5198e/build.sh
```

Keep the printed context directory, resolved descriptor and build logs.
The [recipe](../../runtime/images/compositions/installer-sparkcache-4a17f1f5198e/build.sh)
checks the eleven lease files and both native hashes before building. Copy its
`built-image.json` to `/var/tmp/dsv41-image.json` on the controller. Distribute
the image to each other selected host over its approved fabric SSH route:

```bash
set -o pipefail
sudo -n docker save sparkring:sparkcache-4a17f1f5198e-status038 | ssh FABRIC_SSH_TARGET sudo -n docker load
```

On **each** selected host, replace `EVIDENCE` with a private absolute directory
for this host's evidence. Save its actual image ID and probe artifacts:

```bash
mkdir -p "$EVIDENCE"
IMAGE=sparkring:sparkcache-4a17f1f5198e-status038
sudo -n docker image inspect "$IMAGE" > "$EVIDENCE/image.json"
sudo -n docker run --rm --pull never --network none --entrypoint cat "$IMAGE" /opt/sparkring/receipts/external-base-installed.json > "$EVIDENCE/base.json"
sudo -n docker run --rm --pull never --network none --entrypoint cat "$IMAGE" /usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/sparkring-source-4a17f1f5198e.json > "$EVIDENCE/inventory.json"
sudo -n docker run --rm --pull never --network none --entrypoint python3 "$IMAGE" -m sparkcache.runtime_patches.verify_lease_contract --vllm-root /usr/local/lib/python3.12/dist-packages --contract /usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/vllm-connector-jobs-gab86b70.json > "$EVIDENCE/lease.log"
python3 - "$EVIDENCE/inventory.json" <<'PY' > "$EVIDENCE/SHA256SUMS"
import json, sys
record = json.load(open(sys.argv[1]))
for path, row in record['files'].items():
    print(row['sha256'] + '  ' + path)
for path, digest in record['native_libraries'].items():
    print(digest + '  ' + path)
PY
sudo -n docker run --rm -i --pull never --network none --entrypoint sha256sum "$IMAGE" --check - < "$EVIDENCE/SHA256SUMS" > "$EVIDENCE/installed-hashes.log"
```

Compare each `image.json` ID to the build lock, and later authenticate each
inventory with the startup checker. Require every hash and all 11 lease files
to pass. Native mismatches stop the run; they require a reviewed rebuild.

With the committed source on each host, verify its actual checkpoint. Set
`MODEL_ROOT` to that host's checkpoint directory from the saved lock:

```bash
sudo -n docker run --rm --pull never --network none --entrypoint python3 \
  -e PYTHONPATH=/opt/sc-deploy \
  --mount "type=bind,source=$PWD/integrations/sparkcache/4a17f1f5198e,target=/opt/sc-deploy,readonly" \
  --mount "type=bind,source=$MODEL_ROOT,target=/model,readonly" \
  "$IMAGE" /opt/sc-deploy/deploy/deepseek_v41/checkpoint_identity.py verify-root /model
```

## Render both arms from this branch

On the controller, from the clean committed checkout, render private
deployments using the shared installer container adapter. This makes the
same serving containers as the installer without selecting a package or
changing a controller slot. It reuses the recorded transport section because
the derived image retains the parent's SIRCL/libsircl bytes and tuning key;
validation refuses incompatible transport pins.

```bash
python3 - <<'PY'
import copy, json
from pathlib import Path
from runtime.common import image_lock, installer
prior = json.load(open('/var/tmp/dsv41-prior.json'))
image = json.load(open('/var/tmp/dsv41-image.json'))
assert prior['selection']['profile'] == 'deepseek-v41-flash-tp4'
assert prior['transport']['group']['positions'] == [0, 1, 2, 3]
assert prior['transport']['nccl'] == 'never'
assert len(prior['site']['ranks']) == 4
assert prior['transport']['sircl'] == image['sircl']
section = copy.deepcopy(prior['transport'])
section['image'] = image['name']
for arm, profile in [('on', 'deepseek-v41-flash-sparkcache-tp4'), ('off', 'deepseek-v41-flash-tp4')]:
    raw = copy.deepcopy(prior['site_input'])
    raw.pop('native_mesh', None)
    raw['name'] = 'dsv41-cache-' + arm
    raw['workspace'] = '/srv/sparkring/' + raw['name']
    for host, row in zip(raw['hosts'], prior['site']['ranks'], strict=True):
        host.update(model=row['model'], cache=row['cache'], reuse_verified_model=True)
    directory = Path('/var/tmp/dsv41-cache-' + arm)
    lock = installer.init(directory, profile, raw, image_runtime=image_lock.v2_view(image),
                          transport=section)
    print(directory, installer.connection(lock), installer.containers(lock))
PY
```

Review all four `rankN/compose.yaml` and `container.json` files under each
output directory. They must name the same built image, actual model mounts,
the 108 GiB memory limit and only positions 0-3. The on arm alone enables the
connector. Both use the base served name and port 8015 and cannot run together.
No public Compose exports are added because the default lock refuses the variant.

## H1-H3: start the rendered containers and inspect topology

Stop the original four containers through their original deployment's lifecycle
after authorization. Preserve them for rollback. Before H1, inspect the exact
SparkCache root under each rank's host cache mount. If present, rename it to a
unique archived path while stopped; never clear another root. Check the
resolved path against the configured cache directory before moving anything.

The following source lifecycle uses the adapter's admission, ownership,
checkpoint, runtime-binding and read-only fabric checks. Its selected stages
create/start the rendered containers without restarting relay services or
installing packages. Run it from this branch's controller checkout, with
`ARM=off` first for the memory baseline, then `ARM=on` after stopping the off arm:

```bash
ARM=on python3 - <<'PY'
import os
from runtime.common import installer
from scripts.installer_runner import Runner
directory = '/var/tmp/dsv41-cache-' + os.environ['ARM']
runner = Runner(directory)
lock = installer.load(directory)
rows = lock['site']['ranks']
for operation in ('prerequisites', 'source', 'model', 'image', 'preflight', 'create', 'start', 'ready', 'smoke'):
    selected = [rows[0]] if operation == 'smoke' else rows[1:] + rows[:1]
    for row in selected:
        result = runner(row['host'], ['installer', operation, str(row['rank'])], 7200)
        print(operation, row['rank'], result, flush=True)
        if result['returncode']:
            raise SystemExit('Stop and inspect the failed stage; do not bypass admission')
PY
```

H1: record API readiness, KV tokens and bytes, `MemAvailable`, container memory,
page cache, SM clocks and clock event reasons on every rank for both arms. The
pool must stay within 3% of the cache-off arm on the same image; investigate
memory pressure before changing utilization. The [profile budget](README.md#storage-and-memory-budget)
keeps utilization 0.83 and explains inactive buffer settings.

Save `sudo -n docker logs --timestamps CONTAINER > rank-N.log 2>&1` on each
host using the exact container names printed by the render. Collect all four
logs on the controller and run, once per host's H0 probe directory:

```bash
python3 scripts/check_deepseek_sparkcache_start.py \
  --image-lock /var/tmp/dsv41-image.json --base-receipt "$EVIDENCE/base.json" \
  --inventory "$EVIDENCE/inventory.json" --lease-log "$EVIDENCE/lease.log" \
  --log rank-0.log --log rank-1.log --log rank-2.log --log rank-3.log
```

The config line carries role and storage settings; the topology line carries
`deepseek-v41-b12x-hma` and `full,replay,sliding,transient`. Worker config is
required everywhere, with 17 groups; scheduler config and
`kv lcm block sizes 256` are required only on rank 0. Contradictory records fail even alongside
a matching record. This is an offline evidence check, not proof of placement.

H2: collect `sparkcache: group_topology` DEBUG records in a separate inventory
start. Add `VLLM_LOGGING_LEVEL=DEBUG` to a **private** Compose override for that
start, archive the override, and remove it for performance measurements.
Record each role's digest, spec class, layers, dtype, page size, block size,
`tokens_per_state`, `reuse_policy` and `replay_tokens`. The recorded worker
inventory has 54 layers in 17 groups:

| Groups | Policy | Block tokens | Representative page bytes | Members |
|---|---|---:|---:|---|
| 0-6 | `sliding` | 128 | 67,584 | Encoder SWA layers 0-19, distributed by layer modulo 7 |
| 7 | `full` | 256 | 36,864 | Latent and indexer caches of layers 2, 8, 14 and 20 |
| 8 | `transient` | 8 | 32,768 | Compressor rings of layers 2, 8 and 14 |
| 9-16 | `replay` | 128 | 67,584 | Decoder/draft SWA layers 20-42, distributed by `(layer-20) % 8` |

The [CPU fixture](../../integrations/sparkcache/4a17f1f5198e/sparkcache/test_deepseek_v41_topology.py)
and [recorded topology](../../integrations/sparkcache/4a17f1f5198e/sparkcache/deepseek_v41_topology.json)
bind this packing. Per-layer latent/indexer page bytes differ within group 7;
the table reports its scheduler representative, not the sum of worker pages.
Confirm equal scheduler/worker topology digests and context identities. Both
decoder and draft replay reserves are 128, giving `floor((N-128)/256)*256`
within the configured span bounds. vLLM's scheduler hash block is 128 and
its worker backend block is 256; SparkCache must use the shared group LCM,
256, in both roles.


H3: use the same functional checker as `sudo sparkring check`, directly on the
source-rendered lock (the installed command still selects the prior controller
slot). Record the transport receipts with the branch's retained source:

```bash
python3 - <<'PY'
import json
from runtime.common import installer
from runtime.host import check, retained_source
directory = '/var/tmp/dsv41-cache-on'
lock = installer.load(directory)
print(json.dumps(check.functional(lock, log=print), indent=2))
print(json.dumps(check.connector_state(lock), indent=2))
print(json.dumps(retained_source.apply(directory, 'transport', cache='/var/tmp/dsv41-retained'), indent=2))
PY
```

Require 7/7 functional checks and transport verdict `as-expected`, with every
collective on SIRCL, NCCL absent and no GPU clock fault. Archive actual check
names and results; an absent receipt is not a pass.

## H4-H10: persistence, correctness and performance

Use temperature 0 and `chat_template_kwargs: {"thinking": false}` except the
explicit H10 workload. Both profiles enable prompt-token details. Query
`/v1/models` for the served name. Record exact messages, token IDs, templated
prompt lengths N, responses, `usage.prompt_tokens_details.cached_tokens`,
per-rank committed spans/counters and timing. Do not equate a GPU prefix hit
with a disk restore. Every disk-reader measurement starts after a restart.

For a preliminary semantic miss/hit, run the pinned gate in the built image
on the API host. Set `EVIDENCE` to an absolute writable evidence directory;
use phase `miss` before and `hit` after the planned restart:

```bash
sudo -n docker run --rm --pull never --network host --entrypoint python3 \
  -e PYTHONPATH=/opt/sc-deploy \
  --mount "type=bind,source=$PWD/integrations/sparkcache/4a17f1f5198e,target=/opt/sc-deploy,readonly" \
  --mount "type=bind,source=$EVIDENCE,target=/evidence" \
  "$IMAGE" -m deploy.deepseek_v41.semantic_gate miss \
  --endpoint http://127.0.0.1:8015 --model DeepSeek-V4.1-Flash-TP4 \
  --reference /evidence/semantic.json --records 384
```

This gate's archive-record count is not a token-length control. It does not
replace the exact-length corpus or numerical equality measurements below.

| Gate | Action and proposed acceptance |
|---|---|
| H4 | Cold writers at 12K, 32K, 64K and 128K actual prompt tokens. Exact answers; each rank commits `floor((N-128)/256)*256`. Confirm manifests persist no replay/transient pages. These lengths fit the configured span bounds. |
| H5 | Stop every on-arm container, start all four again preserving NVMe, then read each writer prefix with suffix lengths 0, 1, 127, 128, 129 and 1,000. Verify tokenized prefix identity across the chat-template boundary. After a fresh restart for each reader, cached-token credit and rank restore logs must equal that writer's committed span, with exact answers. Archive each writer root separately if reader publication would change subsequent offers. |
| H6 | `docker kill --signal KILL` all four on-arm containers, start them again, and repeat H5. Preserve persistent roots and require the same results. |
| H7 | 64 prompts of 8K-128K, three ways: cache-off with cold GPU prefix state, native GPU hit, and SparkCache restore after restart. Greedy, 256 output tokens, top-20 logprobs. First token identical for 64/64; complete sequence identical for at least 60/64; every first divergence has cache-off top-two logprob margin below 0.1. Compare native-hit and restore by the same rule. |
| H8 | While all four are stopped, archive one rank's synthetic test root and flip one bit in every payload chunk of that root, keeping manifests unchanged. Never corrupt weights or another rank. Restart; require CRC32 rejection/invalidation, whole-request recomputation and the cache-off answer. Other ranks' files remain unchanged. The vendored corruption gate checks a disposable copy only; it is an additional storage check, not proof of the live failure path. |
| H9 | At 32K, 64K and 128K, restored TTFT at most 10% of cold TTFT. Record streaming first-token timestamps and per-rank restore time. The expected subsecond restore at 128K is an estimate, not evidence. |
| H10 | Same-image A/B on identical four hosts, two starts per arm, medians: decode at 1/8/16 streams and 64K prefill, temperature 1.0. Decode within 3%, prefill within 5% of cache-off. Record Engram disk-read latency per step, memory and clocks in both arms. Keep debug logging off and verify exact profile arguments. |

For H5 use `sudo -n docker stop CONTAINER`, then `sudo -n docker start CONTAINER`
on the four selected hosts, workers before rank 0. For H6 replace stop with
`sudo -n docker kill --signal KILL CONTAINER`. Keep start logs separate and
check readiness before readers. Do not use `sparkring up --on` on the source
path: it would select the controller's registered deployment, not this one.
Switch A/B arms only after stopping all four containers of the other arm.

## H11: receipts and rollback

Keep per-rank image IDs, authenticated H0 receipts, source revision, deployment
locks, exact profile JSON, topology/config logs, corruption file list and hashes,
counters, KV pool sizes, memory/clock observations, requests, token IDs,
logprobs and timings. Record every failed or inconclusive gate. Draft a
sanitized record under `performance/records/images/` using the
[evidence format](../../performance/README.md#evidence-record-format). H7/H9/H10
thresholds are proposals, not measured baselines; CPU passes qualify none of them.

Stop the research containers and restore the prior four containers with the
saved deployment and its source. Run the original deployment's `sparkring check`
and verify serving on 0-3 and unchanged Qwen on 4-7. Retain research roots and
evidence for review. Do not promote this profile before H0-H11 support it.

## Separate installer path: ring-wide package prerequisite

`sudo sparkring install` uses the installed package's catalogs and registers
controller state. Before using it, authorize and install a package from this
branch **throughout the ring**, including the controller and positions 4-7;
this is a separate scope from the four-host source-rendered run. Do not infer
package authorization from permission to interrupt only positions 0-3.

On the authorized build host:

```bash
python3 scripts/build_deb.py --relay-marker require --output /var/tmp/dsv41-package
```

Retain the previous package, distribute the exact `.deb`, record its hash and
install it with `sudo -n apt-get install /absolute/path/to/package.deb` on all
required nodes. After that separately authorized package rollout, the managed
path is:

```bash
sudo -n sparkring down --on 0-3 --execute --yes
sudo -n sparkring install --profile deepseek-v41-flash-sparkcache-tp4 --on 0-3 --image-lock /var/tmp/dsv41-image.json --yes
sudo -n sparkring check --on 0-3
```

Managed H5/H6 use the recorded deployment's down/up operations. Managed A/B
selects `deepseek-v41-flash-tp4` with the same image lock; rollback restores
the saved original image/profile and package. This path still needs all gates.
