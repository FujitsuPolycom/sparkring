# DeepSeek-V4-Flash-0731 Spark quickstart

Deploy DeepSeek-V4-Flash across directly cabled DGX Sparks in either of two
topologies: **two tensor-parallel ranks on a cabled pair**, or **four on a
cycle**.

**Status: implemented.** The env-driven launchers at
[`scripts/deepseek_v4_pair_serve.sh`](../scripts/deepseek_v4_pair_serve.sh) and
[`scripts/deepseek_v4_cycle_serve.sh`](../scripts/deepseek_v4_cycle_serve.sh)
have offline contract coverage. Equivalent two-rank and four-rank serving
arguments have served requests with benchmark image
`ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028`
under the checkpoint conditions below. The published image selected by the
`deepseek_v4_flash_0731_hardened_serving_image` key in
[`runtime/faststart-lock.json`](../runtime/faststart-lock.json) has not completed an exact
post-publication pull-and-replay on either topology, so the profiles are not
qualified for general use or for 1,048,576-token output quality. The recorded
pair throughput used
`deepseek-ai/DeepSeek-V4-Flash-DSpark@913f0657a874f76844e2e91cbe706dbcaceeb6d7`;
the recorded cycle throughput used
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
The pair package has the same configuration and tensor index as plain 0731,
but different weight payloads. The measurements therefore establish launch
behavior and conditional throughput, not exact checkpoint scaling or output
quality.

The machine-readable serving contracts are
[`recipes/deepseek-v4-flash-0731-pair.json`](../recipes/deepseek-v4-flash-0731-pair.json)
for the pair and
[`recipes/deepseek-v4-flash-0731.json`](../recipes/deepseek-v4-flash-0731.json)
for the cycle.

Both public recipes use the plain 0731 package and configure a
1,048,576-token request limit.

| | Two-Spark pair | Four-Spark cycle |
|---|---|---|
| Ranks | 0-1 | 0-3 |
| Cabling | one DAC, cage 0 to cage 0 | four DACs as `0-1-2-3-0` |
| Weights resident per rank | 80.97 GiB | 40.82 GiB |
| `--tensor-parallel-size` / `--nnodes` | 2 | 4 |
| `--kv-cache-memory-bytes` | 17179869184 (16 GiB) | 17179869184 (16 GiB) |
| Request limit / sequences / scheduler tokens | 1M / 32 / 4096 | 1M / 32 / 4096 |
| Environment template | `deepseek-v4-flash-0731-pair.env.example` | `deepseek-v4-flash-0731.env.example` |

`--max-model-len 1048576`, `--max-num-seqs 32`, `--max-num-batched-tokens
4096`, `--block-size 256`, and the 16 GiB reservation are the settings used by
both base topologies.

Both commands explicitly enable asynchronous scheduling and reserve each
request's complete input length before admission:

```text
--async-scheduling --scheduler-reserve-full-isl
```

The flags are explicit so a vLLM default cannot silently change the profile.
Disabling full-input reservation is a separate research experiment because it
can improve admission latency while increasing later queueing, preemption, or
KV pressure under chunked prefill.

## 1. Prepare the ranks

Complete [the prerequisites](PREREQUISITES.md) for the topology you are
deploying, including the fabric routing and forwarding checks. On a cycle, a
rendezvous reaches ranks that are not directly cabled to each other only if
every node relays for its neighbours, and Docker's default `FORWARD` policy
blocks that relay.

Download the official FP8 checkpoint once, distribute identical bytes to every
rank, and choose the same container model mount on each host. The model
contains 48 safetensors shards totaling about 167 GB.

Pull the immutable ARM64 runtime image on every rank before launching any rank:

```bash
docker pull ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:827a8e8c5749b78529cc0015dd174e1b19a0accc116bc142282f8b75428f98bd
```

This is the `deepseek_v4_flash_0731_hardened_serving_image.manifest_digest`
pinned by
[`runtime/faststart-lock.json`](../runtime/faststart-lock.json). The image
registers `DeepseekV4ForCausalLM` and repairs malformed speculative-model
metadata plus five-token sparse-row/native top-k execution. The launch commands
override its GLM-specific entrypoint. The generic image used to roll back GLM
profiles is pinned by the `serving_image` key in
[`runtime/faststart-lock.json`](../runtime/faststart-lock.json).

Copy the environment template for your topology to one local file per rank.

For a pair, fill the rank, rank-0 rendezvous address, model/cache host paths,
fabric interface, and rank fabric address. Both ranks normally name the same
interface when the pair is cabled cage 0 to cage 0. Keep the serving defaults
unchanged for the first launch:

```bash
# Run on the rank 0 / API host.
cp scripts/config/deepseek-v4-flash-0731-pair.env.example /path/to/rank-0.env

# Run on the rank 1 / worker host.
cp scripts/config/deepseek-v4-flash-0731-pair.env.example /path/to/rank-1.env
```

Create the cache directory named by `CACHE_HOST_PATH` on each host before
validation. It must be writable by the account running Docker:

```bash
mkdir -p /absolute/path/to/deepseek-cache
test -w /absolute/path/to/deepseek-cache
```

For a cycle, resolve the rank/rendezvous values, model/cache host paths,
serving values, `<NCCL_SOCKET_IFNAME>`, and `<RANK_FABRIC_IP>`. Even ranks use
cage 0 (`enp1s0f0np0`) and odd ranks use cage 1 (`enp1s0f1np1`):

```bash
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-0.env
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-1.env
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-2.env
cp scripts/config/deepseek-v4-flash-0731.env.example /path/to/rank-3.env
```

Create the `CACHE_HOST_PATH` directory recorded in each cycle rank env before
running `deepseek_v4_cycle_serve.sh --check`.

The templates' `LD_PRELOAD`, `VLLM_NCCL_SO_PATH`, and `NCCL_*` values are
required by the image selected by the
`deepseek_v4_flash_0731_hardened_serving_image` lock key. Do not use a GLM
profile environment file, and
do not use the cycle template on a pair: its transport settings name two host
channel adapters, enable subnet-aware routing, and skip tree connect, none of
which applies when the two ranks are directly adjacent.

The pair template is the single operator input for the pair launcher. It
contains the host model/cache mounts, API and rendezvous ports, speculative
depth, request limit, sequence ceiling, and scheduler token budget in addition
to the container environment. Rank 0 and rank 1 may use different local model
and cache paths, but the mounted model bytes and all serving values must agree.

Use these rank-specific values; keep the serving values below them identical:

| Variable | Rank 0 / API host | Rank 1 / worker host |
|---|---|---|
| `NODE_RANK` | `0` | `1` |
| `MASTER_ADDR` | rank 0 fabric address | same rank 0 fabric address |
| `VLLM_HOST_IP` | rank 0 fabric address | rank 1 fabric address |
| `MODEL_HOST_PATH` | local model directory | local directory with identical model bytes |
| `CACHE_HOST_PATH` | local writable cache directory | local writable cache directory |

Both templates default to automatic GID policy:

```text
NCCL_IB_GID_AUTO=1
NCCL_IB_GID_INDEX=
```

During `--check` and before `--run`, the launcher reads the local
`/sys/class/infiniband` GID table. It applies `NCCL_IB_HCA` with NCCL's
comma-list, `^` exclusion, `=` exact-name, prefix-name, and
`name[:port[:rail[:plane]]]` rules. Only ports with a readable active state are
candidates, and only the first 32 non-empty selector entries are considered.
The pair selector must resolve to one HCA/port; the cycle selector must resolve
to two. A selector that resolves to more than NCCL's 32-device cap fails
preflight because sysfs traversal order cannot prove the order returned by
`ibv_get_device_list()`.

Every selected member must have a RoCE v2, IPv4-mapped GID that matches an IPv4
address reported for the member's netdev at preflight time or `VLLM_HOST_IP`.
The check prints the usable index set for every member and fails before Docker
if any set is empty. The sets do not need a common index. After validation,
the launch command removes
`NCCL_IB_GID_INDEX` from the container so the pinned NCCL 2.30 runtime selects
an appropriate RoCEv2/IPv4 index independently for each HCA.

Use a pin only as an intentional escape hatch:

```text
NCCL_IB_GID_AUTO=0
NCCL_IB_GID_INDEX=<locally verified decimal index>
```

Pinned mode preserves the configured value and does not validate sysfs. On a
cycle, that one rank-global index must be valid for both selected members.
Re-run `show_gids` on the affected host before pinning. A pin from another host
or from before a link or firmware change does not prove the host state read by
this preflight.

## 2. Launch one rank per host

Both launches mount a writable `/cache` directory. The environment sets
`XDG_CACHE_HOME=/cache/jit`, so without that mount every replaced container
recompiles its just-in-time kernels from scratch.

### Two-Spark pair

The pair launcher checks the resolved environment, host paths, API and
rendezvous ports, direct-pair NCCL settings, the local GID table read during
the check, and positive serving limits before constructing Docker arguments.
`--check` reads local host state but does not change it, and prints the exact
command without creating a container. Run it on both hosts:

```bash
scripts/deepseek_v4_pair_serve.sh --check /path/to/rank-0.env
scripts/deepseek_v4_pair_serve.sh --check /path/to/rank-1.env
```

Each check covers one rank. Compare the printed `MAX_MODEL_LEN`,
`MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, and `NUM_SPECULATIVE_TOKENS` values
on both hosts before launching; matching values do not verify that model bytes
are identical.

Start rank 1 first, then rank 0. `--run` refuses to replace an existing
container; remove an existing container intentionally before a relaunch.

```bash
# Rank 1 / worker host
scripts/deepseek_v4_pair_serve.sh --run /path/to/rank-1.env
docker logs -f deepseek-v4-flash-r1

# Rank 0 / API host
scripts/deepseek_v4_pair_serve.sh --run /path/to/rank-0.env
docker logs -f deepseek-v4-flash-r0
```

The pair environment exposes these operator-facing settings and uses the
defaults recorded in
[`recipes/deepseek-v4-flash-0731-pair.json`](../recipes/deepseek-v4-flash-0731-pair.json):

| Variable | Default | Purpose |
|---|---:|---|
| `MODEL_HOST_PATH` | required | Read-only host model directory mounted at the fixed container model path |
| `CACHE_HOST_PATH` | required | Writable parent for persistent JIT/compiler caches mounted at `/cache` |
| `API_PORT` | 8000 | Rank-0 OpenAI-compatible API port |
| `NCCL_IB_GID_AUTO` | 1 | Validate each selected HCA/port and leave GID selection to NCCL |
| `NCCL_IB_GID_INDEX` | empty | Rank-global index used only when automatic policy is disabled |
| `NUM_SPECULATIVE_TOKENS` | 5 | DSpark proposal depth |
| `MAX_MODEL_LEN` | 1048576 | Per-request token limit |
| `MAX_NUM_SEQS` | 32 | Scheduler admission ceiling |
| `MAX_NUM_BATCHED_TOKENS` | 4096 | Scheduler budget and chunked-prefill size |

The launcher uses `--ipc host --shm-size 16g`. Host IPC makes the host's
`/dev/shm` allocation authoritative, so changing the declaration from 16 GiB
to 64 GiB does not enlarge shared memory. Compare `df -h /dev/shm` on the host
and inside the running container when diagnosing shared-memory pressure.

### Four-Spark cycle

The cycle launcher consumes one resolved
`deepseek-v4-flash-0731.env.example` copy per rank and exposes the same model,
cache, port, speculation, context, sequence, and scheduler settings as the pair
launcher. Validate all four ranks and compare their printed serving values:

```bash
scripts/deepseek_v4_cycle_serve.sh --check /path/to/rank-0.env
scripts/deepseek_v4_cycle_serve.sh --check /path/to/rank-1.env
scripts/deepseek_v4_cycle_serve.sh --check /path/to/rank-2.env
scripts/deepseek_v4_cycle_serve.sh --check /path/to/rank-3.env
```

Start worker ranks 1-3 first, then rank 0:

```bash
# Run on the corresponding worker hosts.
scripts/deepseek_v4_cycle_serve.sh --run /path/to/rank-1.env
scripts/deepseek_v4_cycle_serve.sh --run /path/to/rank-2.env
scripts/deepseek_v4_cycle_serve.sh --run /path/to/rank-3.env

# Run on the API host after all workers are waiting for rendezvous.
scripts/deepseek_v4_cycle_serve.sh --run /path/to/rank-0.env
```

Follow the container for the rank on each host, for example:

```bash
docker logs -f deepseek-v4-flash-r1
docker logs -f deepseek-v4-flash-r0
```

`--kv-cache-dtype fp8_ds_mla` is required in both topologies: it declares the
model's MLA key-value geometry and matches the allocated layout. Do not
substitute generic `fp8`.

## 3. Sizing: four coupled parameters, and one inactive guard

`--max-model-len`, `--kv-cache-memory-bytes`, `--max-num-seqs` and
`--max-num-batched-tokens` are not independent. Changing one alone will
usually fail or waste memory.

For the pair, change `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, and
`MAX_NUM_BATCHED_TOKENS` in its environment file and rerun the launcher's
`--check` mode. For the four-Spark cycle, change the same values in every
rank's `deepseek-v4-flash-0731.env.example` copy, confirm that they match, and
rerun `deepseek_v4_cycle_serve.sh --check` on every rank.

**Set `--max-num-batched-tokens` explicitly.** The normalized comparison target
uses 4096 in every DeepSeek base and SparkCache profile. A recorded pair run
with 8192 measured higher throughput at concurrency 32, but the base and cache
comparison uses 4096 so the scheduler budget remains controlled. The
measurement records cover concurrency levels 1, 2, 4, 8, 16, and 32.

**A longer request limit is close to free; a larger pool is not.** This model
has bounded cache groups whose cost per sequence is fixed, so per-token pool
cost falls as the request limit rises — about 18.5 KB per token at a
131,072-token limit and about 6 KB at 1,048,576. Raising `--max-model-len` from
131,072 to 1,048,576 on a pair, changing nothing else, cost no additional
memory.

**Sequence count and speculation depth inflate the per-request floor.** The
engine refuses to start unless the pool can hold one full-length request. The
32-sequence pair profile uses a 16 GiB reservation. The recorded pair reported
a 2,198,756-token pool and 2.10x full-length capacity. A four-Spark concurrency
measurement must record its pool and free memory before load generation.

**`--kv-cache-memory-bytes` disables the memory-utilization guard.** The engine
states this explicitly: it "skipped memory profiling. This does not respect the
`gpu_memory_utilization` config." `--gpu-memory-utilization 0.70` is therefore
**not** a safety ceiling in either launch above — it is inert whenever an
explicit byte count is given. Nothing will stop an oversized reservation from
exhausting the node.

Check free memory after a launch rather than trusting the flag. The normalized
pair had only 4.9–6.0 GiB available per node and used swap. The roughly 50 GB
cycle observation used settings other than the block-256, batch-4096
configuration recorded here.
A node driven to 2-3 GB free with swap in use is one long prefill away from
having its engine core killed, and on this platform severe memory exhaustion
can leave a host answering ICMP while refusing to complete any new connection,
recoverable only by a power cycle.

## 4. Verify rank 0

Wait for the API health endpoint, then issue a deterministic chat request.
Read the configured port from the rank-0 environment so a non-default
`API_PORT` is verified correctly.

```bash
api_port=$(sed -n 's/^API_PORT=//p' /path/to/rank-0.env)
curl --fail "http://localhost:$api_port/health"
curl "http://localhost:$api_port/v1/models"
curl -s "http://localhost:$api_port/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-0731",
       "messages":[{"role":"user","content":"What is 17 * 23?"}],
       "max_tokens":16,"temperature":1.0,"top_p":1.0}'
```

Check rank logs for a successful rendezvous across every rank and for DSpark
metrics. `Mean acceptance length` above one confirms speculative decoding is
active, and `GPU KV cache size` reports the token capacity the reservation
bought.

## Measured

Both cache-disabled base setups were measured with the sampling settings in
their benchmark records. Concurrency levels 1 and 2 use at least five accepted
observations per context; every other applicable decode cell uses at least
three. Table headings `C1` through `C32` denote concurrent request counts.
Decode values are mean aggregate generated tokens per second.

The DSpark package used by TP2 has the same model configuration, tokenizer
configuration, and tensor index as the plain 0731 package used by TP4. Its 48
weight payloads differ, so the tables are useful serving results but not an
exact same-weight TP2-versus-TP4 scaling test.

### Two-Spark pair, TP2/DCP1

| Context | Prefill tok/s | C1 | C2 | C4 | C8 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 1,793 | 62.53 | 75.79 | 106.07 | 144.05 | 201.26 | 275.26 |
| 8K | 1,800 | 48.42 | 89.67 | 110.90 | 156.68 | 217.37 | 299.66 |
| 16K | 1,926 | 58.36 | 77.65 | 104.16 | 162.69 | 202.74 | 307.13 |
| 32K | 1,922 | 51.59 | 85.05 | 107.13 | 147.40 | 223.25 | 301.00 |
| 64K | 1,856 | 50.06 | 76.57 | 108.41 | 154.57 | 205.27 | — |
| 128K | 1,691 | 53.05 | 73.82 | 86.43 | 135.90 | — | — |

[Full TP2 record](../performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.md)
· [green console matrix](../performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.png)

### Four-Spark cycle, TP4/DCP1

| Context | Prefill tok/s | C1 | C2 | C4 | C8 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 2,343 | 105.88 | 131.89 | 187.88 | 245.96 | 373.16 | 463.06 |
| 8K | 2,409 | 103.41 | 122.96 | 184.75 | 253.33 | 367.58 | 463.98 |
| 16K | 2,488 | 68.84 | 139.01 | 210.48 | 265.16 | 428.48 | 508.11 |
| 32K | 2,464 | 92.48 | 118.21 | 176.80 | 233.96 | 399.50 | 476.95 |
| 64K | 2,389 | 92.91 | 141.49 | 186.10 | 277.81 | 364.56 | — |
| 128K | 2,223 | 89.98 | 136.07 | 184.66 | 251.52 | — | — |

[Full TP4 record](../performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.md)
· [green console matrix](../performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.png)

The full records report mean, standard deviation, N, Coding Peak, exclusions,
and source-receipt hashes. Synthetic sustained text changes DSpark acceptance,
so five-observation cells have meaningful variance.

The profile remains collective-latency-sensitive: the model issues two
all-reduces per layer across 43 layers, and each token waits on synchronous
round-trips that cannot fully overlap, because
decode is autoregressive: the input to step `N+1` includes the token produced
at step `N`.

Speculation amortises those round-trips. At a mean acceptance length of 3.06 on
a cycle, one forward pass yields about three tokens, so the collectives cost
roughly a third as much per output token. Raising acceptance is equivalent to
making collectives cheaper.

**Do not expect a faster all-reduce backend on this hardware.** The engine
reports selecting `PYNCCL` from the potential set `NCCL_SYMM_MEM`,
`QUICK_REDUCE`, `FLASHINFER`, `AITER_CUSTOM`, `CUSTOM`, `SYMM_MEM`, `PYNCCL`.
Every faster entry requires peer-accessible GPU memory on one host — NVLink or
PCIe peer-to-peer — or is ROCm-only. With one GPU per node communicating over
RoCE, `PYNCCL` is the correct selection rather than a fallback. Setting
`VLLM_ENABLE_PCIE_ALLREDUCE`, `VLLM_PCIE_ALLREDUCE_BACKEND=cpp` and
`VLLM_CPP_AR_1STAGE_NCCL_CUTOFF` on a cycle changed nothing: 56.0 against 56.5
tok/s at one request, 174.2 against 177.5 at eight. Those variables belong to
single-host multi-GPU profiles and are not worth carrying here.

## What these results cover

Each topology has separate results recorded in its serving contract. The
reported throughput values are client-observed for the prompt and generation
lengths named by each record. The two-rank matrix uses isolated-server
sustained decode and standalone prefill. Neither dataset measures output
quality or qualifies an untested image, checkpoint, topology, or request
shape.

The documented launch uses patched NCCL from the selected topology-specific
environment template. SIRCL width-4096 graph collectives are research-only and
are not part of this quickstart. Diagnostic tool-call evidence, benchmark
image identities, acceptance lengths, dates, and published-digest replay
requirements are retained in
[the profile record](profiles/DEEPSEEK_V4_FLASH_0731.md) and
[results](RESULTS.md).

The offline-only [four-Spark SIRCL A/B plan](DEEPSEEK_V4_FLASH_SIRCL_AB.md)
validates both transport arms against this quickstart's machine-readable
recipe, applies a 4,096-token batch budget to both arms, and uses the same
model, memory, scheduler mode, and five-token DSpark
contract. The plan has no execution mode and does not promote SIRCL into this
quickstart.
