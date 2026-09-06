# Run DeepSeek V4 Flash Vision-Exp on four GB10 systems

Serve `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at TP4 on a four-Spark direct-cable cycle
(`0-1-2-3-0`, two cables per node, no switch) using the MiaAI-Lab DSpark recipe image as the
serving stack and SparkRing's patched NCCL as the transport.

**Status: implemented, live-benchmarked on one site (2026-09-06), not qualified.** One operator
site brought the profile up from two production TP2 pairs, gated it (auth, text, image, per-rank
parity, Ring Doctor), measured it against the pairs, held a 1,048,576-token needle probe exactly, and
ran a 20-minute c=8 soak with zero preemptions. It has not been reproduced on a second site and has
no long soak yet. The machine-readable contract is
[`recipes/deepseek-v4-flash-vision-exp-tp4.json`](../recipes/deepseek-v4-flash-vision-exp-tp4.json).

## Why this profile exists

[`DEEPSEEK_V4_FLASH_QUICKSTART.md`](DEEPSEEK_V4_FLASH_QUICKSTART.md) serves the text-only 0731
checkpoint on `gb10-vllm-serving`. Vision-Exp adds a 32-block ViT, an aligner and image tokens
under the same `DeepseekV4ForCausalLM` architecture string, moves `num_nextn_predict_layers`
from 1 to 3, and sets `rms_norm_eps=1e-20`. The community stack that already handles all three on
GB10 is the MiaAI-Lab recipe on the Anemll `dspark-vllm-gx10` image (native vision hotfix, DSpark
k=5 with the block-k unlock, FlashInfer 0.6.18.post1). Reusing it for a cycle changes exactly two
things versus a pair: the rank count, and the NCCL library plus its environment. Nothing in the
model stack is touched, so pair-validated correctness fixes carry over unchanged.

## Prerequisites

Complete [PREREQUISITES.md](PREREQUISITES.md) for the four-Spark cycle: forwarding, static routes,
and the `DOCKER-USER` ACCEPT rule between the two fabric interfaces on every node. Then, on every
node:

- Admit the other three ranks' **management** addresses through the host firewall. The cycle
  bootstraps torch's TCPStore (`MASTER_PORT`) and NCCL/Gloo ephemeral sockets on the management
  interface; a default-deny inbound policy makes every rank hang silently at
  `distributed_init_method=tcp://<rank0>:<port>` with no error anywhere.
- Admit the fabric subnets on the host firewall as well.
- Reboot a node that has been serving for days if it shows few contiguous 32 MiB blocks
  (`/proc/buddyinfo`); the memory gate in the GLM quickstart applies here too.

Checkpoint: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@86f746b36186f0e567729a5c06a8c918caba82a9`
(82 files, 161.5 GB) in a standard Hugging Face hub cache on every rank. Weights resident per rank
at TP4: 41.6 GiB.

Serving stack: MiaAI-Lab `DeepSeek-v4-Flash-DSpark-2x-DGX-Spark` at `7440c53` checked out (or
copied) at the same path on every rank, and its image (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`,
optionally with the FlashInfer 0.6.18.post1 overlay used by the reference site) present on every
rank with an identical image id.

Transport: SparkRing's patched NCCL `libnccl.so.2.30.7` (`switchless-cycle`, `skip-tree-pat`,
`advertise-all-listener-gids`) at a host path on every rank. The library links only
`GLIBC_2.17/2.18`, so it loads in the Anemll image (glibc 2.35); it carries the same SONAME as
torch's bundled `libnccl.so.2`, so an `LD_PRELOAD` satisfies torch's `DT_NEEDED` with the patched
copy, and `VLLM_NCCL_SO_PATH` points pynccl at the same file. The image's own pip NCCL is already
2.30.7, so the ABI torch 2.11+cu130 expects is the one preloaded.

## Cabling and ranks

Existing pair cables stay on cage 0. The two new cables go cage 1 to cage 1 on the cross edges.
Which node is at the far end of each new cable must be discovered (temporary addresses, then
ping), not assumed: both pairings form a valid cycle, but vLLM node ranks must follow the physical
cycle so NCCL's ring `0-1-2-3-0` is the cable ring. The reference site's pairing was
a-head(0) — a-worker(1) — b-head(2) — b-worker(3) — a-head, subnets 100/110/102/111 (/24).

## Configure each rank

One environment file per rank. Start from the pair recipe's `.env.dspark` (keys, hotfix switches,
k, thinking default) and override:

```text
NODE_RANK=<0..3>                 HEADLESS=<empty on rank 0, 1 otherwise>
NNODES=4                         TP_SIZE=4
MASTER_ADDR=<rank-0 management IP>   MASTER_PORT=25000
VLLM_HOST_IP=<this rank's management IP>
NCCL_SOCKET_IFNAME=<management interface>   TP_SOCKET_IFNAME=<same>   GLOO_SOCKET_IFNAME=<same>
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1   NCCL_IB_GID_INDEX=3   NCCL_IB_MERGE_NICS=0
NCCL_IB_SUBNET_AWARE_ROUTING=1      NCCL_IB_SUBNET_PREFIX_LEN=24   NCCL_CROSS_NIC=1
GPU_MEMORY_UTILIZATION=0.80  MAX_NUM_SEQS=48  MAX_NUM_BATCHED_TOKENS=12288  MTP_NUM_TOKENS=5
DSPARK_ENABLE_SP_INDEXER=1       # recipe's sequence-parallel Lightning indexer: deep prefill 1.41x on TP4
DSPARK_RESTART_POLICY=no
```

Compose overlay (a second `-f` file on top of the recipe's `docker-compose.dspark.yml`):

```yaml
services:
  vllm-dspark:
    volumes:
      - /path/to/libnccl.so.2.30.7:/opt/sparkring/nccl/libnccl.so.2:ro
    environment:
      LD_PRELOAD: /opt/sparkring/nccl/libnccl.so.2
      VLLM_NCCL_SO_PATH: /opt/sparkring/nccl/libnccl.so.2
      NCCL_ALGO: Ring
      NCCL_PROTO: LL,LL128,Simple
      NCCL_P2P_LEVEL: SYS
      NCCL_MIN_NCHANNELS: "4"
      NCCL_MAX_NCHANNELS: "4"
      NCCL_SKIP_TREE_CONNECT: "1"
      NCCL_SWITCHLESS_RING_ONLY: "1"
```

Verify GID index 3 is the RoCEv2 entry for the fabric IPv4 on **both** devices of every rank from
sysfs before launching; a netplan apply on a rail can shift the table, and a reboot restores it.

## Start TP4

Workers first (ranks 3, 2, 1), then rank 0, each with

```text
docker compose -p deepseek-ring --env-file <rank env> \
  -f docker-compose.dspark.yml -f <overlay> up -d
```

run from the recipe checkout directory (its hotfix mounts are relative). Expect in rank 0's log:
`NCCL version 2.30.7+cuda13.0`, `Tree transport setup disabled by NCCL_SWITCHLESS_RING_ONLY`,
`PAT transport setup disabled`, `NET/IB: Subnet-aware routing: overriding dev …` for both devices,
`Using ['PYNCCL'] all-reduce backends`. Weights load in ~125 s from local NVMe; the first boot pays
JIT compiles for the TP4 shapes (persisted on the recipe's cache volume); healthy in ~260 s on later
boots.

## Gates

Keyless `/v1/models` → 401; keyed list carries the served name; a text canary returns exactly and
`finish_reason=stop`; a generated solid-colour PNG returns the colour; image id identical on all
ranks; the patched NCCL mapped in every rank's engine processes; `/metrics` open; Ring Doctor
`--verify` PASS on the site file; 0 preemptions after the ladder.

## Measured (reference site, 2026-09-06)

Fresh-prefix prompts, thinking off, temperature 0, 256 pinned decode tokens; prefill single request
with `max_tokens=1`. Decode is aggregate generated tok/s.

| Profile | C1 | C4 | C8 | C16 | C32 | C48 | C64 | Prefill 16K / 64K / 128K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **TP4 cycle**, 48-64 seqs | 49 | 112 | 160-175 | 235 | 327 | 391 | 435 | 2,300 / 2,255 / 2,100 |
| TP2 pair, same stack, 16 seqs | 31 | 63 | 97 | 141 | queue-bound | — | — | 1,600-1,800 / 1,600-1,685 / 1,410-1,550 |

Step rate 22/s single-stream (pair 15/s), 149/s at C32, 200/s at C64; DSpark acceptance 0.23-0.25
at every rung on both topologies, so the gain is per-step collective cost. KV pool 50-52 GiB per
rank (`nvfp4_ds_mla`, util 0.80) = 6.2 M tokens; five to six 1M requests resident. Per-request
speed crosses 10 tok/s at C48; the reference site caps admission at 48 for agent traffic.

Two pairs behind a router can at best sum to ~283 tok/s at fleet C32 (two direct ladders run
concurrently); the cycle measured 327. Prefill past ~180K context is bound by the DSV4 Lightning
indexer, which every TP rank runs over the full context (560-660 tok/s on TP4 and TP2 alike); a
cold 1M prompt takes ~16 min on either topology with the stock indexer. NIAH at 958,182 real tokens
returned the exact needle in 963.9 s stock and in **683.3 s with the recipe's sequence-parallel
indexer** (`DSPARK_ENABLE_SP_INDEXER=1`: each TP rank scores a quarter of the compressed keys for
chunks with ≥8,192 keys and the ranks merge candidates exactly; decode and ≤128K prefill unchanged).
On TP2 the same patch measured only −6 % at 128K; on TP4 the replicated term is quartered, which
is why it is part of this profile.

Knobs measured neutral or negative on the cycle (one variable per boot): replicated DSpark Markov
head, 8 NCCL channels, in-flight prefill cap 3, greedy draft sampling, 8,192 batched tokens. The NVIDIA text-only
`DeepSeek-V4-Flash-0731-NVFP4` checkpoint boots on the same stack only with
`--moe-backend flashinfer_cutlass` and the recipe's vision hotfix disabled; it measured prefill +6 %
and decode −5..−14 % (draft acceptance 0.20 with one MTP layer).

## Limits

One failure domain (a wedged rank takes the lane down; pairs gave a replica). No SparkCache or KV
connector in this stack. Rank 0's API port should be firewall-scoped to the router host. Boot
ownership needs a rank-0 unit that waits for peers and drives the launcher; the compose policy is
`restart: no` by design. Not qualified: no second-site reproduction, no multi-hour soak, NIAH at 1M
measured at the 10 % position only.
