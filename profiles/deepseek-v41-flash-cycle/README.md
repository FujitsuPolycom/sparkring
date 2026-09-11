# DeepSeek-V4.1-Flash four-Spark cycle quickstart

Profile: `deepseek-v41-flash-cycle`. Status: **implemented**. The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image.

Inspect its selected defaults with `python scripts/profiles.py resolve deepseek-v41-flash-cycle`.

Serve `deepseek-ai/DeepSeek-V4.1-Flash` (the stock checkpoint) as four tensor-parallel
ranks on a directly cabled four-Spark cycle, with the model's two Engram lookup tables
left on each rank's NVMe.

**Status: implemented; live-benchmarked on one private cycle; not qualified.** The
profile runs a stock upstream vLLM image that you build yourself from pinned sources
([`runtime/deepseek-v41-gb10`](../../runtime/deepseek-v41-gb10/README.md)); no public
image digest exists to replay. The machine-readable contract is
[`recipes/deepseek-v41-flash-cycle.json`](../../recipes/deepseek-v41-flash-cycle.json); the
evidence is in the [profile record](../../docs/profiles/DEEPSEEK_V41_FLASH.md) and the
[benchmark record](../../performance/records/deepseek-v41-flash/cycle-tp4-dspark5-graphs-20260910.md).

## Why this profile is shaped the way it is

DeepSeek-V4.1-Flash is 475 GiB on disk: a 552B MoE backbone whose routed experts are already
MXFP4, plus two FP8 Engram n-gram tables of 94.6 GiB each. Four GB10s hold 121.7 GiB each.
With the tables row-sharded in memory a rank needs about 118.8 GiB before KV, activations and
the CUDA context, so it does not fit; vLLM's `cpu_offload` does not help because pinned host
memory on GB10 is the same pool the GPU allocates from. Keeping the tables in the safetensors
shards and reading the 48 rows a token needs on demand brings a rank to **78.8 GiB**
(text-only) or **81.6 GiB** (with the DSpark draft layers and the vision encoder). That
method, and the SM12x fixes the vLLM `dsv41-feat` branch still needs on GB10, come from
[tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
(MIT; SM12x page fixes by Kai) and are bind-mounted over the image unchanged. What this
profile adds is the switchless-cycle transport: SparkRing's patched NCCL and the four-rank
cycle environment, so no Ethernet switch is needed.

| | value |
|---|---|
| Ranks / cabling | 0–3, four DACs as `0-1-2-3-0`, two RoCE devices per rank |
| Weights resident per rank | 78.79 GiB text-only; 81.6 GiB with DSpark draft + vision (measured) |
| Engram tables | on each rank's NVMe, read on demand (23.6 GiB per rank per table not allocated); balanced hash-column split + packed single-read shards (`ENGRAM_BALANCED=1`, `ENGRAM_PACKED_DIR`) |
| Request limit / sequences / scheduler tokens | 430,080 / 8 / 8,192 |
| `--gpu-memory-utilization` | 0.83 with `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` → KV 10.94 GiB = 2,182,642 tokens (5.07× 430K) measured; 13–15 GiB MemAvailable per rank |
| Speculation | DSpark k=5, greedy draft, block rejection, adaptive verification off |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes = every multiple of 5 and 6 up to 48 |
| Block size | 128 (required; an auto-picked 64 fails KV initialization) |
| Vision / tools | up to 4 images per request; `deepseek_v41` tool and reasoning parsers |
| Environment template / launcher | `scripts/config/deepseek-v41-flash-cycle.env.example` / `scripts/deepseek_v41_cycle_serve.sh` |

## 1. Prepare the ranks

Complete [the prerequisites](../../docs/operations/prerequisites.md) and the [bootstrap guide](../../docs/operations/bootstrap.md) for a
four-Spark cycle, including forwarding and the relay routes: a torch rendezvous reaches ranks
that are not directly cabled only if every node relays for its neighbours. Run
`scripts/ring_doctor.py --site <site> --verify` and require a clean reachability matrix.

Check the GPUs before trusting any number: GB10s can latch below 1 GHz with nothing visible
in `nvidia-smi` except speed (clear it by unplugging the adapter for 30–60 s), and can sit in a
slower hidden state under load. A 15 s fp16 matmul burn on a healthy unit reads about
2.2–2.4 GHz at 80 W or more.

### Weights

Download the stock checkpoint once and place a complete, byte-identical copy on **every rank's
local NVMe** (about 510 GB each). Do not serve the checkpoint over NFS: the Engram rows are
read from shards 47 and 48 at serve time, and the read path is the first thing that bounds
throughput under concurrency. Pin the revision; this profile records
`dba1be0a40aa45a94ad051997016db3960a90277` (revision `df42c109…` has identical weights and
`config.json`; only the reference encoder scripts differ). Between two Sparks a direct link
copies at ~600 MB/s, so copy to one rank and replicate over the fabric.

### Image

Build the image on an idle Spark with
[`runtime/deepseek-v41-gb10/build-image.sh`](../../runtime/deepseek-v41-gb10/build-image.sh),
then `docker save | ssh <rank> docker load` to the other three and confirm identical image
IDs. Record the ID in `IMAGE_ID` in every env file; the launcher refuses a mismatch. Read the
[builder README](../../runtime/deepseek-v41-gb10/README.md) first: the stable-extension compile
needs an idle node (one CUTLASS translation unit takes a compiler process past 7 GiB), while
the FlashInfer layers can be built inside a 7 GiB cgroup on a node that is still serving.

### Patches and NCCL

Copy `runtime/deepseek-v41-gb10/patches/` (seven files, `mounts.txt`, `MD5SUMS`) to the same
absolute path on every rank; the launcher verifies the md5s. Extract SparkRing's patched
NCCL from any published SparkRing image and put it at the same path on every rank:

```bash
docker create --name nccl-tmp <sparkring image> true
docker cp -L nccl-tmp:/opt/sparkring/nccl/libnccl.so.2 /path/to/libnccl.so.2
docker rm nccl-tmp
```

The image's own pip NCCL is also 2.30.7; vLLM logs a `Duplicate NCCL runtime` warning
because the two paths differ. The preloaded library is the one mapped in every process and
the one PyNccl loads by path.

### Engram packed shards (once per rank)

The stock loader reads two 4 KiB pages per Engram row (weight and scale sit ~24 GB apart in the
shard) and splits the 24 hash columns contiguously, which hands rank 3 the six four-gram columns
(nearly every row unique) and rank 0 the six bigram columns (heavily repeated): per-rank traces of
one 16K prefill showed rank 3 issuing 320K row reads to rank 0's 60K and the other ranks waiting
for it at the next all-reduce. The recipe therefore sets `ENGRAM_BALANCED=1` (strided columns,
two heads of each order per rank) and reads from packed shards built once per rank:

```bash
docker run --rm --entrypoint python3 --memory 6g \
  -v "$MODEL_HOST_PATH:/models/DeepSeek-V4.1-Flash:ro" -v "$CACHE_HOST_PATH:/cache" \
  -v "$PWD/runtime/deepseek-v41-gb10/tools:/tools:ro" "$IMAGE" \
  /tools/pack_engram_rows.py --model-dir /models/DeepSeek-V4.1-Flash --out-dir /cache/engram-packed \
  --tp 4 --rank "$NODE_RANK" --balanced
```

About nine minutes per rank; the two sparse files show 101 GB logical / ~48 GB allocated. The
manifest records the covered ranges and the loader refuses a shard that does not cover the rank's
columns (it then logs a warning and reads the checkpoint shards directly). Measured on the cycle:
prefill 1,590 → 1,873 tok/s at 16K and 1,745 → 2,058 at 64K, burst TTFT p50 11.1 → 9.8 s, decode
and acceptance unchanged, 131K/262K needle pass.

### Environment

Copy the template once per rank and resolve every placeholder. `NODE_RANK` and
`VLLM_HOST_IP` differ between ranks; everything else must be byte-identical on all four —
a configuration mismatch hangs the rendezvous with no error.

```bash
cp scripts/config/deepseek-v41-flash-cycle.env.example /path/to/rank-0.env   # and 1, 2, 3
scripts/deepseek_v41_cycle_serve.sh --check /path/to/rank-0.env
```

`--check` is offline: it validates the file, the model directory (`config.json` must name
`DeepseekV41ForCausalLM`, shard 48 must be present), the patch md5s, the NCCL library and
the cycle transport values, then prints the exact `docker run` command. Compare the printed
serving values across the four ranks before launching.

## 2. Launch one rank per host

Reboot the ranks before a first launch or a measurement (memory fragmentation on GB10 costs
real throughput), then start workers 3, 2 and 1 before rank 0:

```bash
scripts/deepseek_v41_cycle_serve.sh --run /path/to/rank-3.env   # on rank 3
scripts/deepseek_v41_cycle_serve.sh --run /path/to/rank-2.env   # on rank 2
scripts/deepseek_v41_cycle_serve.sh --run /path/to/rank-1.env   # on rank 1
scripts/deepseek_v41_cycle_serve.sh --run /path/to/rank-0.env   # on rank 0, the API host
docker logs -f deepseek-v41-flash-r0
```

`--run` refuses to start if the container exists, if `MemAvailable` is under 100 GiB, or if
the image identity differs from `IMAGE_ID`. Stop every rank (rank 0 first) before relaunching:
a worker that starts while an old head still listens on the rendezvous port joins the old
head and hangs.

Expect about eight minutes to readiness from local NVMe: ~4 min of weights, ~1 min for the
DSpark draft layers, then graph capture and FlashInfer autotune. Lines to look for:

```text
Engram DISK mode: layer 1 rows [<start>, <end>) read from model-00047-of-00048.safetensors
Model loading took 78.79 GiB memory            (text-only) / consumed 85.71 GiB (serving shape)
GPU KV cache size: 2,182,642 tokens, Maximum concurrency for 430,080 tokens per request: 5.07x
Application startup complete.
```

With `ENGRAM_BALANCED=1` each rank logs `BALANCED column assignment, rank r owns hash columns [...]`
(six distinct columns per rank) and, when the packed shard is used, `PACKED single-read shard`.
Without it the Engram row ranges must differ per rank and together cover the table; identical `off=`
values on every rank mean the rank-offset fix is not mounted.

## 3. Verify rank 0

```bash
curl --fail http://localhost:8000/health
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"deepseek-v4.1-flash",
  "messages":[{"role":"user","content":"Count from 1 to 30, comma separated, then say done."}],
  "max_tokens":120,"temperature":0}'
```

Thinking is off by default; a request enables it with `"chat_template_kwargs": {"thinking": true}`.
With `API_KEY_FILE` set, add `-H 'Authorization: Bearer <key>'` to the chat request; `/health` stays keyless
so router health probes keep working. Rank 0 without `API_KEY_FILE` is an open server — set it before
the port sits behind any route that does not authenticate on its own.
Check `SpecDecoding metrics` in the log for a mean acceptance length above one, and
`/metrics` for zero preemptions under load.

## 4. Sizing

`--kv-cache-memory-bytes` is not used here; `--gpu-memory-utilization 0.83` sizes the pool
against ~113 GiB free at startup, and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` stops the
profiler reserving an estimated 1.5 GiB for graphs that measure 0.54 GiB on this profile. On the
recorded boot the pool came out at 10.94 GiB (2,182,642 tokens, 5.07× the 430,080-token limit)
with 13–15 GiB MemAvailable per rank while serving. 0.85 was measured too (3,085,606 tokens,
400K needle pass, same speed) but left only 7–9 GiB MemAvailable, so the recorded profile keeps
the headroom; 0.80 at a 300,000-token limit gave 1,171,588 tokens. Raising `--max-num-batched-tokens`
to 16,384 did not boot at 0.80 (the profiler run needs 1.9 GiB of KV for one full-length request
and 1.84 GiB was left). `--max-num-seqs 8` is the soaked value; 16 booted at 0.80 with 13–15 GiB
free, was neutral up to eight streams and reached 285 tok/s aggregate on the prompt set at 16
streams, so it is a valid admission option when per-stream speed matters less than throughput.
With DSpark k=5 every decode batch is a multiple of 5 or 6 tokens and the graph capture list
follows from the sequence cap. 1M context has not been run on this profile.
