# DeepSeek-V4.1-Flash on four Sparks: KV pool on release 2026.10.4's image

Status: **research-only**; one start. This record gives the KV pool that
`deepseek-v41-flash-tp4` reported when installed without options on the
default image of release 2026.10.4. The README's and the website's profile
tables show it ([profile-capacity.json](../../profile-capacity.json)).

## Conditions

- Command: `sudo sparkring install --profile deepseek-v41-flash-tp4 --on 0-3`,
  as the deployment's install result records it, with approval; no
  `--image-lock` and no research options. Deployment instance
  `i373895f6ea75`, planned 2026-10-14T10:57:42Z; the rank-0 container
  started at 11:09Z.
- Package source revision `4026296ae905` (release 2026.10.4's tag commit), as
  the deployment lock records it.
- Installer image `dev-20261013-dcp2prefill-status038-sircl-tp6`
  (configuration ID `3cc9a81fd344`), the default image of release 2026.10.4.
- Checkpoint `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa`.
- Positions 0-3 of a ring of eight DGX Sparks (GB10), SIRCL ring sessions
  with NCCL off. Other models served on positions 4-7.
- The profile sets no KV cache size. vLLM sizes the pool from the memory
  free at start under `--gpu-memory-utilization 0.83`, so the pool differs
  between starts.

## Measurement

The `GPU KV cache size` line of rank 0's startup log, from the container's
log and the deployment's receipt `20261014T111400Z`.

## Result

```text
(EngineCore pid=212) INFO 10-14 11:12:20 [kv_cache_utils.py:2850] GPU KV cache size: 12,751,634 tokens, Maximum concurrency for 1,048,576 tokens per request: 12.16x
```

Two earlier starts of the profile on the same image and Sparks, on
2026-10-14, logged 12,654,354 and 12,675,509 tokens.

## Conclusion

Installed without options on release 2026.10.4's image, the profile reports
a KV pool of about 12.65-12.75 million tokens; this start reported
12,751,634.

## Limitations

- The pool depends on the memory free at start; the figure is one start's.
- The figure is the engine's capacity. It does not show that concurrent
  full-context requests fit.
- The deployment's records are on the ring's Node A, not in this repository.
