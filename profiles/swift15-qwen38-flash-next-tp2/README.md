# Swift 1.5 Qwen3.8-Flash-Next NVFP4 on two Sparks

[Swift 1.5 Qwen3.8-Flash-Next NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4/tree/3ff0520224f264a2d0ac4ab56ece8f2f13aadb38)
([UkisAI](https://huggingface.co/ukisai)'s NVFP4 quantization of Swift 1.5, its
reasoning-efficient fine-tune of Qwen's
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); revision
`3ff0520224f2`) on two DGX Sparks, with MTP speculative decoding and 262K
context. Node A serves the API on port 8000 as
`Swift-1.5-Qwen3.8-Flash-Next-NVFP4-TP2`, with no API key. Status:
Experimental; no GB10 serving run covers this profile.

On the Spark connected to your network:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/one-command-installer/install.sh | bash -s -- --profile swift15-qwen38-flash-next-tp2
```

[Install SparkRing](../../docs/operations/install.md) covers requirements,
logs and recovery. [Compose files](../../docs/operations/compose-files.md)
lists this profile's per-rank Compose files.

## Settings

Swift keeps the architecture of Qwen3.8-Flash-Next, so this profile uses the
[qwen38-flash-next-tp2](../qwen38-flash-next-tp2/README.md) settings: HC
token-row prefill ownership with the image's `qwen-collectives` and
`qwen4-prefill` features, RoCEnante decode all-reduces, MXFP8 HC projections
and the NVFP4 MTP LM head quantized at load, and MTP3 with probabilistic
drafts. It differs where this checkpoint's format differs:

| Setting | This profile | `qwen38-flash-next-tp2` |
|---|---|---|
| Quantization | ModelOpt NVFP4 routed experts, `--quantization modelopt_fp4`; every other weight BF16 | ModelOpt mixed precision, `--quantization modelopt_mixed` |
| PLE n-gram table | 95.4 GiB BF16, read from the checkpoint files with io_uring each step (`VLLM_PLE_TABLE_MEMORY=disk`) | NVFP4, in GPU memory (`VLLM_PLE_CPU_OFFLOAD=0`) |
| MTP draft experts | BF16, on vLLM's unquantized MoE kernel | MXFP8, on the `humming` MoE backend |
| Checkpoint | 100 files, 173.7 GiB | 56 files, 102.6 GiB |
| Rendezvous port | 29639 | 29638 |

Both profiles serve 16 sequences of up to 262,144 tokens with 8,192-token
batches, 24 GiB of FP8 KV cache per rank, vLLM's prefix cache and no
SparkCache.

The PLE table stays on disk because two Sparks cannot hold it in GPU memory.
Resident, each rank would hold about 48 GiB of the table beside about 40 GiB
of other weights; with the 24 GiB KV cache that exceeds the profile's GPU
memory budget (`--gpu-memory-utilization 0.85`) and the container's 108 GiB
memory limit. The [four-Spark profile](../swift15-qwen38-flash-next-tp4/README.md)
keeps the table in GPU memory.

## Limitations

- Performance, MTP draft acceptance and the cost of reading PLE rows from
  NVMe are unmeasured. The model card's evaluations ran with MTP disabled.
- The model card's scores are for the BF16 checkpoint; the publisher's
  `QUANTIZATION_MANIFEST.json` records no quality benchmark of this NVFP4
  checkpoint.
- UkisAI validated this checkpoint on H100 with the Marlin weight-only
  fallback; native NVFP4 W4A4 on Blackwell is unverified by the publisher.
- The first start compiles and tunes kernels for every CUDA graph size.

## License

UkisAI's contribution is licensed under the
[Swift Open License v1.0](https://huggingface.co/ukisai/Swift-Qwen3.8-Flash-Next/blob/main/LICENSE):
free for individuals and organizations with gross annual revenue up to
US$1,000,000, above which commercial use needs a Swift Enterprise License. The
base model stays under Qwen's
[Qwen Community License 1.0](https://huggingface.co/ukisai/Swift-Qwen3.8-Flash-Next/blob/main/LICENSE-QWEN).
Review both before deployment. The installer downloads anonymously, which
requires the repository to remain ungated on the Hub.
