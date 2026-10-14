# KV pools of three installer profiles installed without research options

Status: **research-only**; one start per profile. This record gives the KV
pool that each profile below reported at start. The README's and the
website's profile tables show these figures
([profile-capacity.json](../../profile-capacity.json)).

## Conditions

- Package: release 2026.10.3's candidate, git `48e4ffb1`. Installer image
  `dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036`
  (configuration ID `d52737a109e0`), the image of releases 2026.10.2 and
  2026.10.3.
- Eight DGX Sparks (GB10) cabled as one ring of eight, SIRCL ring sessions
  with NCCL off, GPU clocks locked at 2,418 MHz, 2026-10-12.
- Each profile was installed with `sudo sparkring install` and the options
  below, and no research options. Each run receipt states the profile's own
  KV settings.

| Profile | Install options | Checkpoint | KV setting |
|---|---|---|---|
| `qwen38-flash-next-tp2` | `--checkpoint csf --on 6,7` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` @ `dc5b8ab2fa14` | 24 GiB per rank (`--kv-cache-memory-bytes 25769803776`), FP8 target KV, MTP draft KV `auto` |
| `glm53-nvfp4-tp8` | `--no-auto-recover` | `local-inference-lab/GLM-5.3-NVFP4` @ `b472e4ee53f6` | 43 GiB per rank (`--kv-cache-memory-bytes 46170898432`), `fp8_ds_mla`, DCP4 |
| `deepseek-v41-flash-tp8` | `--no-auto-recover` | `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a40aa` | no fixed size: `--gpu-memory-utilization 0.83` |

`--checkpoint csf` selects the checkpoint that `qwen38-flash-next-tp2`
installs without options on an image whose vLLM reads it, such as this image
and release 2026.10.4's.

## Measurement

The `GPU KV cache size` line of rank 0's startup log, as each start's run
receipt (`serving-ab-resolved/v1`) recorded it before the benchmark.

## Result

```text
qwen38-flash-next-tp2   (EngineCore pid=215) INFO 10-12 09:09:55 [kv_cache_utils.py:2850] GPU KV cache size: 2,534,058 tokens, Maximum concurrency for 262,144 tokens per request: 9.67x
glm53-nvfp4-tp8         (EngineCore pid=197) INFO 10-12 08:26:57 [kv_cache_utils.py:2850] GPU KV cache size: 3,374,336 tokens, Maximum concurrency for 1,048,576 tokens per request: 3.22x
deepseek-v41-flash-tp8  (EngineCore pid=213) INFO 10-12 08:02:51 [kv_cache_utils.py:2850] GPU KV cache size: 32,677,006 tokens, Maximum concurrency for 1,048,576 tokens per request: 31.16x
```

Eleven further run receipts of `qwen38-flash-next-tp2 --checkpoint csf`
without research options, from several starts on 2026-10-11 and 2026-10-12,
record the same 2,534,058 tokens.

## Conclusion

Installed without options on this image, `qwen38-flash-next-tp2` holds
2,534,058 KV tokens, `glm53-nvfp4-tp8` 3,374,336, and
`deepseek-v41-flash-tp8` held 32,677,006 at this start.

## Limitations

- DeepSeek's pool depends on the memory free at start, so another start can
  differ.
- Not measured on release 2026.10.4's image.
- Each figure is the engine's capacity. It does not show that concurrent
  full-context requests fit.
- The run receipts and logs are in the operator's qualification record of
  2026-10-12, which is not in this repository.
