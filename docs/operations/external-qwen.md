# Qwen on the external ARM64 runtime

The opt-in [shared-2026.09.4-rc.4 release](../../runtime/releases/shared-2026.09.4-rc.4/README.md)
adds SparkRing integrations to the pinned September 23 eugr foundation. Use the
same SparkRing checkout and immutable image on every rank. Stable profile
defaults remain unchanged.

| Profile ID | Configuration | Container prefix | API port |
| --- | --- | --- | --- |
| `qwen38-flash-next-qad-tp2-eugr` | `profiles/qwen38-flash-next-qad-tp2-eugr/config.json` | `qwen-flash-next-sparkcache-tp2` | 8000 |
| `qwen38-flash-next-qad-tp4-eugr` | `profiles/qwen38-flash-next-qad-tp4-eugr/config.json` | `qwen-flash-next-qad-sparkcache-tp4` | 8015 |

Complete the [host prerequisites](prerequisites.md). Fabric preparation is the
same as the [TP2 guide](../../profiles/qwen38-flash-next-tp2/README.md) or
[TP4 guide](../../profiles/qwen38-flash-next-qad-tp4/README.md). These launchers
consume existing network settings; they do not configure cables, addresses or
interfaces.

## Prepare each node

Use the QAD checkpoint at revision
`629bc3218833a38b475b719f34aa571666f4a03e` from
`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`. Reuse verified local weights and
check the existing [checkpoint inventory](../../profiles/qwen38-flash-next-tp2/SHA256SUMS).
Keep the writable cache directory outside the read-only model directory.

```bash
IMAGE_REF=ghcr.io/fujitsupolycom/sparkring@sha256:3d3411b66dd2a4f78ea062ec0a923308f302df14f0e0f9a6ae8961946ae603c8
docker pull --platform linux/arm64 "$IMAGE_REF"
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE_REF")
```

Select `PROFILE` from the table and set `RANK`, `MASTER`, `HOST_IP`, `INTERFACE`,
`MODEL_DIR` and `CACHE_DIR` to the prepared local site values. Ranks are zero
based; all nodes use the same master address and distinct local host addresses.
The cache directory must exist and have space for the selected profile's disk
budget: 16 GiB per TP2 rank or 4 GiB per TP4 rank, in addition to compiled kernels.

From the checkout root, inspect a plan:

```bash
python runtime/common/qwen_flash_next.py plan \
  --profile "$PROFILE" --rank "$RANK" --master "$MASTER" \
  --host-ip "$HOST_IP" --interface "$INTERFACE" \
  --image "$IMAGE_ID" --model "$MODEL_DIR" --cache "$CACHE_DIR"
```

Run the same command with `check` to verify the installed image and CLI, then
with `create` to create a stopped container. `check` is not inference or hardware
qualification. The adapter verifies publication, source, transport, cache-contract
and status-plugin identities; it supplies the loader's IPC-lock/seccomp envelope.

If an earlier deployment uses the same container names, stop it and rename its
containers to explicit rollback names before creating replacements. Preserve its
image and compatible cache directory. After every new rank has been created,
start each node's container using the table's prefix and `-r<RANK>` suffix. Rank
zero exposes the API. First startup compiles and tunes kernels before readiness.

## Inspect the running configuration

On rank zero, use the selected port:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/sparkring/status
```

For TP4, use port 8015. Windows Command Prompt can use `curl.exe`; PowerShell
can use `Invoke-RestMethod`. If inference authentication is enabled, supply the
same bearer token. The [status plugin guide](../../integrations/vllm/runtime_status/README.md)
describes configured/effective values, worker scope, stale data and observations
that remain unknown. Prepared kernel choices do not prove per-request execution.

TP2 selects HC projection sharding with row ownership off. TP4 selects replicated
HC projections with prefill row ownership, direct NCCL prefill collectives and
the prepared RoCEnante policy. Combining projection sharding with row ownership
is rejected. Persistent namespaces bind the composition and complete serving
profile; an incompatible configuration does not reuse the prior namespace.

The optional [Compose adapter](compose.md) consumes these same registered profiles
and container settings. No cluster daemon is installed by this release.
