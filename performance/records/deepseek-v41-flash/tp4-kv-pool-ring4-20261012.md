# DeepSeek-V4.1-Flash on four Sparks: KV pool of an installation without options

Status: **research-only**; one start. This record gives the KV pool that the
profile `deepseek-v41-flash-tp4` reported when installed without options on
the image of releases 2026.10.2 and 2026.10.3. The profile tables show the
pool on release 2026.10.4's image
([record](../images/dev-20261013-dcp2prefill-status038-sircl-tp6-deepseek-v41-flash-tp4-kv-ring8-20261014.md)).

## Conditions

- Profile `deepseek-v41-flash-tp4`, installed with `sudo sparkring install`
  and no research options, on SIRCL ring sessions with NCCL off (the
  profile's default transport), 2026-10-12.
- Package: release 2026.10.3's candidate, git `48e4ffb1`. Installer image
  `dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`
  (configuration ID `d52737a109e0`), the image of releases 2026.10.2 and
  2026.10.3.
- Checkpoint `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa`.
- Four DGX Sparks (GB10) cabled as a ring of four, GPU clocks locked at
  2,418 MHz.
- The profile sets no KV cache size. vLLM sizes the pool from the memory
  free at start under `--gpu-memory-utilization 0.83`, so the pool differs
  between starts.

## Measurement

The `GPU KV cache size` line of rank 0's startup log, as the start's run
receipt recorded it.

## Result

- SIRCL ring sessions: `GPU KV cache size: 11,308,216 tokens`.
- In the same transport comparison, `--transport nccl` started with
  12,545,976 tokens, and NCCL with the net-sircl network plugin with
  12,596,264 ([SIRCL ring sessions with NCCL off](../../enhancements.md#sircl-transport)).

## Conclusion

An installation of the profile without options reported a KV pool of about
11.3 million tokens on this image and ring.

## Limitations

- One start. The pool depends on the memory free at start, so another start
  can differ.
- Not measured on release 2026.10.4's image or on a path of four in a larger
  ring.
- The figure is the engine's capacity. It does not show that concurrent
  full-context requests fit.
- The run receipts are in the private record of 2026-10-12 that the
  enhancement catalog cites for this transport comparison, not in this
  repository.
