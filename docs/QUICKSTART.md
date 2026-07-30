# Four-Spark NF3 quickstart

New deployment? Complete
[the exhaustive prerequisites checklist](PREREQUISITES.md) first. It is the
source of truth for hardware, Docker/NVIDIA runtime, management SSH, storage,
200 GbE/RoCE setup, discovery commands, and the operator-versus-bot boundary.

This is the sole current model deployment, with two KV-storage profiles:

```text
madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid
TP4 / DCP4 / adaptive MTP2-4 / C8 / Q40
four directly cabled DGX Sparks
SparkCache disabled
```

| Profile | Status | KV format | Reported capacity |
|---|---|---|---:|
| `fp8` | conservative default | FP8 | 511,488 tokens |
| `nvfp4-rope8` | equivalent live profile is API-healthy; public bootstrap live gate pending | NVFP4 latent + FP8 RoPE, per-token scale | 875,520 tokens |

Both profiles use the same target checkpoint, MTP draft, topology, 7 GB/rank
KV allocation, and launch policy. Selecting the second profile does not
download the model again. It builds only a thin compatibility layer that keeps
the pinned NF3 expert kernels while restoring the packed-MLA reader. The
875,520-token result is from the equivalent running profile; do not describe
the new public build path as live-validated until its own four-rank gate passes.

The former Aiden MXFP4/GPTQ lane is historical and lives in
[history/AIDEN_MXFP4_GPTQ.md](history/AIDEN_MXFP4_GPTQ.md).

## 1. Cable and manage the four Sparks

Use a separate management network for SSH and four 200 Gb/s DACs in one
closed inference cycle:

```text
                 operator or bot
                       |
          Wi-Fi / LAN / USB / Tailscale
             |      |      |      |
             S0     S1     S2     S3

                   200 Gb/s
              S0 ========== S1
              ||             ||
     200 Gb/s ||             || 200 Gb/s
              ||             ||
              S3 ========== S2
                   200 Gb/s

       Management: SSH, downloads, launch, API
       200 GbE ring: RDMA inference traffic only
```

Use Wi-Fi, USB Ethernet, LAN, or Tailscale for management SSH. Keep the four
direct 200 Gb/s subnets dedicated to the inference fabric. Every Spark needs:

- Docker plus NVIDIA Container Toolkit;
- key-based SSH;
- ARM64/SM121 DGX Spark hardware;
- about 450 GB of usable capacity per rank for the checkpoint, MTP draft,
  image transfer, and safety headroom;
- additional long-term space for the JIT cache and retained image layers.

Use [SETUP.md](SETUP.md#stage-1--hardware-cabling) for netplan, MTU 9000,
RoCEv2 GIDs, and cable qualification.

## 2. Give the bot the facts it cannot discover

The operator must provide:

- one reachable management SSH target per Spark, including usernames;
- working credentials or public-key authorization;
- which physical machine should be rank 0;
- permission before changing networking, installing packages, or stopping a
  serving container.

The bot can discover the remaining machine-local facts through SSH:

| Fact | Read-only discovery |
|---|---|
| management address/interface | `ip -o -4 addr show` and `ip route` |
| 200 GbE interfaces and state | `ip -br link`; `cat /sys/class/net/*/speed` |
| netdev-to-RDMA mapping | `rdma link show`; `ls -l /sys/class/infiniband/*/device/net/` |
| RDMA port | `ls /sys/class/infiniband/<device>/ports/` |
| RoCEv2 GID index | inspect `gids/<i>` and `gid_attrs/types/<i>` |
| actual cable neighbors | assign temporary unique test IPs and use interface-bound pings |
| storage paths/free space | `df -h`; `docker info --format '{{.DockerRootDir}}'` |

The exact commands and IPv4-to-GID example are embedded beside every field in
[`site.example.yaml`](../scripts/config/site.example.yaml) and summarized in
[`scripts/config/README.md`](../scripts/config/README.md#deriving-the-non-obvious-values).
Do not infer cable neighbors from cage labels; probe them.

## 3. Establish SSH from a trusted operator checkout

First clone on a trusted operator/controller that already has authenticated
management SSH to all four ranks:

```bash
git clone https://github.com/FujitsuPolycom/sparkring.git
cd sparkring
cp scripts/config/site.example.yaml scripts/config/site.yaml
$EDITOR scripts/config/site.yaml
```

Change the management SSH targets, management NICs/addresses, two 200 Gb/s
interfaces per rank, ring addresses, RDMA devices, GID indices, and storage
paths. Do not put passwords or tokens in the file.

Check it without contacting a Spark:

```bash
python scripts/sparkring_site.py scripts/config/site.yaml
```

Check/fix key-based SSH before a long download:

```bash
# Read-only: check the exact controller/rank-0 management paths used by
# bootstrap_nf3.py.
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml \
  --scope bootstrap

# Mutating: install only missing public-key trust/authorization after approval.
python scripts/verify_ssh_mesh.py \
  --site scripts/config/site.yaml \
  --scope bootstrap \
  --fix
```

The bootstrap scope requires the controller to reach all four management SSH
targets and rank 0 to reach ranks 1-3 through those same management targets.
It deliberately does not substitute direct-ring addresses. Use
`--scope all-adjacent` only when validating the optional direct-ring SSH relay
paths as a separate diagnostic.

`--fix` must run from that trusted controller. Running it on rank 0 is also
valid, but only when rank 0 already has authenticated management SSH to all
four configured targets. The tool cannot bootstrap its own controller access.

After the bootstrap scope passes, put the **same commit** and the completed
untracked `site.yaml` on rank 0. For example, clone/checkout that commit on
rank 0 and copy `site.yaml` over the management network. If this trusted
checkout is already on rank 0, no copy is needed. Run every command below from
that exact rank-0 checkout; `bootstrap_nf3.py execute` refuses to run elsewhere.

Inspect the complete read-only machine/RDMA probe:

```bash
python scripts/preflight.py \
  --site scripts/config/site.yaml \
  --print-plan
```

The unresolved input site does not yet have the final image ID. The bootstrap
runs the full preflight after it builds the image and writes the resolved site.
Do not require an executing preflight against the unresolved input file.

## 4. Inspect the bootstrap plan

```bash
python scripts/bootstrap_nf3.py plan \
  --site scripts/config/site.yaml
```

Plan mode performs no remote mutation. It names the exact commit, model paths,
four SSH targets, image tag, and ordered operations.

To inspect the larger-capacity alternative instead:

```bash
python scripts/bootstrap_nf3.py plan \
  --site scripts/config/site.yaml \
  --profile nvfp4-rope8
```

## 5. Build, verify, distribute, and launch

```bash
python scripts/bootstrap_nf3.py execute \
  --site scripts/config/site.yaml \
  --confirmation BOOTSTRAP-NF3-ALL-FOUR
```

For the live NVFP4-latent/FP8-RoPE alternative:

```bash
python scripts/bootstrap_nf3.py execute \
  --site scripts/config/site.yaml \
  --profile nvfp4-rope8 \
  --confirmation BOOTSTRAP-NF3-ALL-FOUR
```

For a first deployment, add `--no-launch` to prepare, verify, distribute, and
run the resolved full preflight without starting containers. Review:

```bash
cat .sparkring/bootstrap/site.yaml
cat .sparkring/bootstrap/launch.nvfp4-rope8.json

python scripts/sparkring_launcher.py \
  --site .sparkring/bootstrap/site.yaml \
  --launch-config .sparkring/bootstrap/launch.nvfp4-rope8.json \
  plan
```

Then start that exact generated profile:

```bash
python scripts/sparkring_launcher.py \
  --site .sparkring/bootstrap/site.yaml \
  --launch-config .sparkring/bootstrap/launch.nvfp4-rope8.json \
  --execute start
```

The script:

1. validates the site and rank-0 identity;
2. skips already complete model/draft directories, otherwise resumes the
   immutable Hugging Face downloads and verifies their hashes;
3. fetches exact B12X and DGX-Spark port commits;
4. pulls/checks the pinned public ARM64 base;
5. builds SparkRing faststart without rebuilding Torch/vLLM/FlashInfer;
6. builds the small NF3 adapter layer and verifies its full input receipt; for
   `nvfp4-rope8`, it adds and ABI-checks one thin packed-MLA compatibility
   layer;
7. saves that image once and loads the identical image ID on ranks 1-3;
8. writes `.sparkring/bootstrap/site.yaml`;
9. runs the read-only hardware/image/model preflight;
10. launches all four ranks with the pinned C8/Q40 graph profile.

To prepare everything but leave the model down, add `--no-launch`.

The build is local because no SparkRing registry credential is required. It is
still deterministic at the input boundary: the ARM64 base, model, draft,
B12X, Spark port, SparkRing source, overlay files, NCCL, and transport are
recorded or hashed. A mismatch fails before serving.

## 6. Tail and smoke-test

The default container name is `glm52-sparkring-nf3-r0`:

```bash
docker logs --follow --tail 200 glm52-sparkring-nf3-r0
```

From another machine, use your rank-0 SSH target:

```powershell
ssh operator@RANK0-MANAGEMENT-IP `
  "docker logs --follow --tail 200 glm52-sparkring-nf3-r0 2>&1"
```

Health and model identity:

```bash
curl -fsS http://RANK0-MANAGEMENT-IP:8000/health
curl -fsS http://RANK0-MANAGEMENT-IP:8000/v1/models
```

The current configuration must report:

| Setting | Value |
|---|---:|
| target | `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid` |
| TP / DCP | 4 / 4 |
| adaptive MTP | depths 2,4; window 32 |
| maximum sequences | 8 |
| maximum query rows | 40 |
| maximum batch tokens | 4096 |
| model length | 458,752 |
| KV | selected profile, 7,000,000,000 bytes/rank |
| NF3 workspace reserve | 805,306,368 bytes/rank |
| SparkCache | disabled |
| execution | CUDA graphs through Q40 |

Run the gate only on a disposable/idle deployment:

```bash
python scripts/acceptance_gate.py \
  --site .sparkring/bootstrap/site.yaml \
  --launch-config .sparkring/bootstrap/launch.nvfp4-rope8.json \
  --gate-config scripts/config/gate.example.json
```

For the default FP8 profile, substitute
`.sparkring/bootstrap/launch.fp8.json`. Never gate the NVFP4 image with the
FP8 example launch file.

## Troubleshooting

- `timed out connecting/accepting control peer`: verify peer order is
  `[rank^1, rank^3]`, then rerun site/cable preflight.
- `index_topk_pattern`: do not remove the exact `--hf-overrides` from
  `launch.example.json`.
- prefix-retention failure: the launcher intentionally emits a bare
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` unset.
- wrong output after graph capture: do not weaken the receipt/profile guards;
  collect the full log and open an issue with the generated site redacted.
- interrupted model download: rerun the same bootstrap command; Hugging Face
  resumes partial files and SparkRing re-hashes completion.
- image already present: exact receipt/image-ID matches are reused; conflicting
  identities fail rather than being silently overwritten.
- NVFP4 profile rejected: do not add only `--kv-cache-dtype`; select
  `--profile nvfp4-rope8` so the image ABI and all per-token scale controls
  change together.

Resolved experiments and earlier bring-up failures are kept in
[TESTING_HISTORY.md](TESTING_HISTORY.md), not in this deployment path.
