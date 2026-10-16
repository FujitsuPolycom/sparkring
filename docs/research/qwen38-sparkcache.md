# Qwen SparkCache on release 2026.10.4: hardware acceptance

Status: **research-only**. CPU-tested profiles and a build recipe are implemented;
the corrected composition requires hardware qualification. The 2026-10-15
research inventory on vLLM 2026.10.4 records five manager groups at TP2 and TP4:
three GDN groups of 12 layers, one TP-replicated PLE group, and one attention
group containing 12 QSA layers and the BF16 MTP layer. Incremental saves on
SparkCache a136fc5 failed because worker and scheduler topology identities
differed. CPU tests cover the representative-spec identity correction; they
do not qualify serving correctness or performance. The [TP2 profile](../../profiles/qwen38-flash-next-csf-sparkcache-tp2/README.md)
and [TP4 profile](../../profiles/qwen38-flash-next-step5500-sparkcache-tp4/README.md)
fix separate checkpoints and namespaces.

## Scope and prerequisites

Every serving stage below **STOPS SERVING** on positions 4-7. Obtain owner
authorization for those hosts and a window. No Spark is assumed idle. Keep
positions 0-3 serving unchanged. Record the actual cabling first: positions
4-5 and 6-7 must each be directly cabled. All Spark-side inspection uses
`sudo -n`. The controller's package must contain this branch's committed
profiles; installing that package on Node A requires authorization for that
controller action even when Node A is outside 4-7. Do not restart its model.

Transfer the committed source using a local Git bundle or archive; no push is
required. On the authorized ARM64 build Spark, build the package from the
committed source (Python 3.12, PyYAML, gcc and libibverbs-dev installed):

```bash
python3 scripts/build_deb.py --relay-marker require --output /tmp/qwen-sparkcache-package > /tmp/qwen-package.json
```

Copy the resulting `.deb` to Node A, retain its previous package for rollback,
and install that exact file with `sudo -n apt-get install /absolute/path/to/package.deb`.
Record the source revision and package SHA-256. Package installation must not
change the serving selection on 0-3. The installer distributes its package to
the selected workers. Retain the original 4-7 deployment lock and image before
stopping it.

## Build and image probe (Q0)

First stop the 4-7 deployment from Node A:

```bash
sudo -n sparkring down --on 4-7 --execute --yes
```

Then on one authorized Spark of 4-7, with parent image
`sha256:3cc9a81fd344fe52fe9cc261b46dc051e3c3f494e51b8142846a5f854835c502`
already local, run from the committed SparkRing source root:

```bash
sudo -n bash runtime/images/compositions/installer-sparkcache-3daf882e358b/build.sh
```

The [composition](../../runtime/images/compositions/installer-sparkcache-3daf882e358b/README.md)
pins SparkCache `3daf882e358b0c0846f14b44122204a8774f2f66`. It serves GLM CRC32
and both Qwen variants. It is separate from the recorded `8abbe9da67b8` GLM
build so that build remains reproducible. The recipe probes the parent without
GPU access, verifies both native hashes and eleven lease files, authenticates
receipts, builds and admits the layer, and verifies the resulting lease contract.
It writes `runtime/images/compositions/installer-sparkcache-3daf882e358b/built-image.json`,
never a release selection. Keep the printed build-context directory.

Copy the image Spark-to-Spark over the approved fabric route, using the real
SSH targets from the recorded site. On the build Spark, repeat for each other
host in 4-7, replacing `FABRIC_SSH_TARGET` with that host's approved route:

```bash
set -o pipefail
sudo -n docker save sparkring:sparkcache-3daf882e358b-status038 | ssh FABRIC_SSH_TARGET sudo -n docker load
```

Compare `sudo -n docker image inspect --format '{{.Id}}' sparkring:sparkcache-3daf882e358b-status038`
on every host with `image_id` in the build lock. Copy the exact lock to Node A
as `/var/tmp/qwen-sparkcache-image.json`. An unpublished lock stays outside
`runtime/releases/`.

On each selected host, save the image probe (the files below describe that
host's local image, not a synthetic CPU fixture):

```bash
mkdir -p /var/tmp/qwen-sparkcache-probe
IMAGE=sparkring:sparkcache-3daf882e358b-status038
sudo -n docker run --rm --pull never --network none --entrypoint cat "$IMAGE" /opt/sparkring/receipts/external-base-installed.json > /var/tmp/qwen-sparkcache-probe/base.json
sudo -n docker run --rm --pull never --network none --entrypoint cat "$IMAGE" /usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/sparkring-source-3daf882e358b.json > /var/tmp/qwen-sparkcache-probe/inventory.json
sudo -n docker run --rm --pull never --network none --entrypoint python3 "$IMAGE" -m sparkcache.runtime_patches.verify_lease_contract --vllm-root /usr/local/lib/python3.12/dist-packages --contract /usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/vllm-connector-jobs-gab86b70.json > /var/tmp/qwen-sparkcache-probe/lease.log
```

The build admission verifies installed bytes. Re-run the native `sha256sum`
probe from `build.sh` on every copied image. Expected placement hash:
`d89c9fdae8dc99ae3f7a151cc3dd9e92fdc8fd0b994069fc263027fd4d056c93`;
64-group snapshot hash:
`b50db45123c4faf0fdfb021da0a0f1b7f329869ae5dc13dcc7432dd4ce765b68`.

## Stage A: two pairs

From Node A, after the 4-7 stop and image distribution:

```bash
sudo -n sparkring install --profile qwen38-flash-next-csf-sparkcache-tp2 --on 4-5 --image-lock /var/tmp/qwen-sparkcache-image.json --yes
sudo -n sparkring install --profile qwen38-flash-next-tp2 --checkpoint csf --on 6-7 --image-lock /var/tmp/qwen-sparkcache-image.json --yes
sudo -n sparkring check --on 4-5
sudo -n sparkring check --on 6-7
```

Record GPU clocks, API endpoints, model names, container IDs and per-rank logs.
Use `sudo -n docker logs --timestamps CONTAINER > rank-N.log 2>&1` on each
host, replacing `CONTAINER` with the container in that deployment's saved lock.
Keep separate logs for each start. Assert the model profile and P=2848 on
every rank, LCM=P on node rank 0, and <=64 groups with only `full,recurrent_align`:

```bash
python3 scripts/check_qwen_sparkcache_start.py --profile qwen38-flash-next-csf-sparkcache-tp2 --image-lock /var/tmp/qwen-sparkcache-image.json --base-receipt /var/tmp/qwen-sparkcache-probe/base.json --inventory /var/tmp/qwen-sparkcache-probe/inventory.json --lease-log /var/tmp/qwen-sparkcache-probe/lease.log --log rank-0.log --log rank-1.log
```

Run the receipt/probe check for each host's probe files with the same rank-log
set. The checker is an offline evidence gate, not proof of host membership or
serving correctness; retain the SSH collection and image-ID record.
For Q2, capture the `sparkcache: group_topology` debug record and the vLLM spec
inventory in an instrumented start, including dtypes and layer membership.
If debug logging is disabled, enable it only for that inventory run, record
the override, and return to the exact profile for performance. Update the
SparkCache conformance fixture if the grouping differs from the five-group
2026-10-15 inventory. The MTP layer belongs to the target attention group:
12 target layers at 576 B/token plus MTP at 1,088 B/token total 8,000 B/token
per rank. No separate draft group was identified. GDN and PLE each use
`tokens_per_state=-1`, 3 speculative blocks, 1 prefill checkpoint block and
alignment 16. The scheduler retains one member spec while workers retain
`UniformTypeKVCacheSpecs`; their topology and context digests must agree.

Planned restart for Q5 (keeps the persistent root):

```bash
sudo -n sparkring restart --on 4-5 --execute --yes
```

For Q6, use `sudo -n docker kill --signal KILL CONTAINER` on every selected
rank, then `sudo -n sparkring up --on 4-5 --execute --yes`. Disable automatic
recovery for the test only if needed, recording and restoring its prior state.
No command may select containers on positions 0-3.

## Stage B: four Sparks and rollback

Stop both pairs before installing TP4:

```bash
sudo -n sparkring down --on 4-5 --execute --yes
sudo -n sparkring down --on 6-7 --execute --yes
sudo -n sparkring install --profile qwen38-flash-next-step5500-sparkcache-tp4 --on 4-7 --image-lock /var/tmp/qwen-sparkcache-image.json --yes
sudo -n sparkring check --on 4-7
```

Use the startup checker with the TP4 ID and four `--log` arguments. P=1440.
For Q5/Q6 use `--on 4-7`. For the same-image cache-off arm, stop TP4 and run:

```bash
sudo -n sparkring down --on 4-7 --execute --yes
sudo -n sparkring install --profile qwen38-flash-next-qad-tp4 --checkpoint qad-step5500-ple1000 --on 4-7 --image-lock /var/tmp/qwen-sparkcache-image.json --yes
```

Repeat the install commands to alternate arms, two starts each. Preserve the
isolated cache roots for restore tests; verify a root is empty only at Q1.
Never clear another profile's root. For Q10 stop all selected ranks, record the
synthetic entry's manifest and payload paths, flip one bit in each selected
chunk on one rank only, then restart and retain the CRC32 rejection evidence.

Final rollback selects the published release, not the development image:

```bash
sudo -n sparkring down --on 4-7 --execute --yes
sudo -n sparkring install --profile qwen38-flash-next-qad-tp4 --checkpoint qad-step5500-ple1000 --on 4-7 --image 2026.10.4 --yes
sudo -n sparkring check --on 4-7
```

## Acceptance gates

Use exact-answer requests at temperature 0 with
`chat_template_kwargs: {"enable_thinking": false}`. Both profiles enable
prompt-token details. Record tokenized lengths including chat-template tokens,
not approximate text lengths; query `/v1/models` for the exact served name.
Target writer lengths N=12,000, 32,000 and effective-limit+1 (48,417 at TP2,
61,921 at TP4). The last token is recomputed, so publication is
`floor((N-1)/P)*P` within these limits. Oversized writers are skipped rather
than guaranteed to clamp to the configured limit; do not use a longer writer
to claim full-limit publication. Reader cached-token credit may depend on
native prefix hits; each disk-restore measurement starts with empty GPU state.

| Step | Action | Pass criterion |
|---|---|---|
| Q0 | Image probe without GPU (shared with DeepSeek H0 and GLM step 1) | Every pinned hash matches; the contract verifies 11 of 11; the snapshot library is the 64-group build |
| Q1 | Cold start, empty root | API ready; every rank logs `qwen38-flash-next-hybrid` and block size P; node rank 0 logs LCM P; KV pool within 3 % of the cache-off profile on the same image (TP2 CSF reference: 2,534,058 tokens) |
| Q2 | Topology dump | Group list recorded; at most 64 groups; only `full` and `recurrent_align`; GDN groups show 3 speculative blocks and 1 prefill checkpoint block; PLE group `tp_replicated`; MTP group dtype and membership recorded; CPU fixture updated to match |
| Q3 | Functional | `sudo sparkring check` passes 7 of 7; SIRCL on every collective, NCCL absent |
| Q4 | Cold writers at 12K, 32K and the span limit | Exact answers; every rank commits `floor((N-1)/P)*P` for the bounded writer lengths above; capture-slot fallback counters 0 |
| Q5 | Planned restart, then readers sharing each writer's prefix with suffixes of 0, 1, 3, 4, 5, P-1, P and P+1 tokens | `cached_tokens` equals the committed span on every rank; exact answers. Suffixes that are not multiples of 4 exercise the QSA reset |
| Q6 | SIGKILL of every rank, `up`, same readers | As Q5 |
| Q7 | Media: three images and a six-second video, restart, replay; then one image changed | Restored on every rank with correct answers; changed media gives 0 cached tokens |
| Q8 | Output equality: 64 prompts of 8K-48K tokens, each served cache-off, as a native GPU prefix hit, and as a SparkCache restore after restart; greedy decoding, 256 tokens, top-20 logprobs | First token identical in 64 of 64; full sequences identical in at least 60 of 64; every divergence starts where the cache-off top-2 margin is below 0.1 |
| Q9 | Multi-turn session: 12 turns of about 3K appended tokens, each turn after a restart or a GPU prefix-cache reset | Each turn restores the previous committed span; at most 4 stages before compaction; bytes written per stage within 10 % of 8,000?d + 61 MB (TP2) or +31 MB (TP4), before headers |
| Q10 | Corruption: flip one bit in each chunk of one rank's root while stopped | CRC32 rejection on that rank; recomputation with the cache-off answer; other ranks unaffected |
| Q11 | TTFT at 16K, 32K and the span limit | Restored TTFT at most 25 % of cold at 32K and 15 % at the limit; per-rank restore milliseconds recorded |
| Q12 | A/B against the cache-off profile, same Sparks and image, two starts per arm: decode at 1/8/16 streams and 64K prefill at temperature 1.0, plus `phase1-16k` at temperature 0 | Decode within 3 % and prefill within 5 % of cache-off, medians of two starts; MemAvailable recorded per rank |
| Q13 | Receipts, then restore the Qwen TP4 deployment | Per-rank logs, counters, image IDs; a record under `sr:performance/records/images/`; `qwen38-flash-next-qad-tp4` serving on 4-7 |

The thresholds in Q8, Q11 and Q12 are proposed acceptance values, not
measured baselines. Bit-exact sequences are not required, because the
restored run computes the suffix in a different batch shape. The Q11
thresholds allow for the suffix of up to P-1 tokens that a restore still
prefills: up to 0.8 s at TP2.


Capture receipts, logs, counters, exact requests and responses, token IDs,
logprobs, timings and MemAvailable per rank in a record under
`performance/records/images/`. Q8/Q11/Q12 thresholds are proposed gates, not
baselines. CPU checks do not establish CUDA placement, QSA reset correctness,
media isolation, serving or performance. Do not mark either profile qualified
until the hardware record supports it.
