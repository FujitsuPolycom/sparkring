# Qwen3.8-Flash-Next — TP2/DCP1

Experimental Compose translation of SparkRing’s R37 launch specification. These files passed local Compose rendering and argument/environment parity checks; they have not been launched or qualified as a Compose deployment.

262K context (262144 tokens), 16 sequences, 8192 batched tokens, 24 GiB GPU KV per rank, MTP3. Image limit 3, video limit 1, sampled video frames 16. Native prefix caching is enabled. SparkCache and GLM-specific mHC/coalescing are disabled, matching the canonical profile.

Image: `ghcr.io/fujitsupolycom/sparkring@sha256:f5a7e01c6112c8ef85a51b24bfacfd3934ee9cfff06b7e8c72abcf5d90b50270` (`linux/arm64`). Model: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4 at ada4da32a583a78aa47299f45a70603c950490b8`. Model downloads are not included. Verify the complete model snapshot using the [profile quickstart](../README.md) before use. Complete the full SHA256SUMS check, including the weight shards.

## Host preparation

Use 2 GB10 Linux hosts with Docker Compose, NVIDIA Container Toolkit, the pinned image, the verified model snapshot and working RDMA connectivity. Each host runs only its own rank file. Compose does not deploy to other hosts or provision the fabric. Do not start these containers alongside another model using the same GPUs or ports.

Qwen requires the direct-pair fabric expected by its transport profile: `rocep1s0f0,roceP2p1s0f0`, GID 3, two paths and peer-HCA mapping. Check those against the physical wiring before use. The memory limits are 108 GiB RAM / 112 GiB RAM-plus-swap; cgroups do not fully bound GB10 GPU allocations.

## Configure and render

Run the commands below from this `compose/` directory on each host. Copy its `.env.rankN.example` to `.env.rankN`. Replace the documentation IPs, interface and absolute host paths. `MASTER_ADDR` must identify rank 0 on every host; `HOST_IP` must identify that host.

Use an existing model directory and create a separate writable cache directory for this example. Model mounts are read-only; missing directories are rejected. Keep caches separate from any running deployment. Shell environment variables override the env file, so inspect the resolved configuration.

For rank 0 (substitute the local rank number on each other host):

```bash
docker compose --env-file .env.rank0 -f compose.rank0.yaml config
```

Review the rendered image, paths, rank, addresses and transport environment. With the hosts free and fabric checks passed, explicitly fetch the pinned image on each host:

```bash
docker pull ghcr.io/fujitsupolycom/sparkring@sha256:f5a7e01c6112c8ef85a51b24bfacfd3934ee9cfff06b7e8c72abcf5d90b50270
```

Then start each rank within the same startup window using its local file:

```bash
docker compose --env-file .env.rank0 -f compose.rank0.yaml up -d --no-build
docker compose --env-file .env.rank0 -f compose.rank0.yaml logs -f model
```

Workers are headless. Only rank 0 serves the API, on port 8000. Confirm every rank joined, then check rank 0:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

An API health response does not prove cache reuse or transport stability. Both ranks disable container healthchecks; use the explicit API check and worker logs. Automatic restart is disabled because a single restarted rank cannot safely rejoin all distributed executions.

Stop all ranks using their respective files before restarting the deployment:

```bash
docker compose --env-file .env.rank0 -f compose.rank0.yaml down
```

## Scope of translation

Container names and project names are example-specific. Site paths and addresses are required inputs. The published registry digest replaces its corresponding local image ID. Pull policy is `never` after the explicit image fetch; automatic restart is off. Otherwise the source serving arguments, environment values, resource limits and mounts are preserved. The configuration was checked with Docker Compose 5.1.3. Rendering checks do not establish hardware qualification.

The YAML is a snapshot of the R37 launch settings. The [profile quickstart](../README.md) owns image selection and preparation. Update and compare the examples when that procedure or its runtime configuration changes; there is no automatic Compose exporter yet. Do not add model enhancements from an unrelated profile without verifying support and testing the resulting configuration.
