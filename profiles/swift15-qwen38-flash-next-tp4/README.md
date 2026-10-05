# Swift 1.5 Qwen3.8-Flash-Next NVFP4 on four Sparks

[Swift 1.5 Qwen3.8-Flash-Next NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4/tree/3ff0520224f264a2d0ac4ab56ece8f2f13aadb38)
([UkisAI](https://huggingface.co/ukisai)'s NVFP4 quantization of Swift 1.5, its
reasoning-efficient fine-tune of Qwen's
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); revision
`3ff0520224f2`) on a four-Spark ring, with MTP speculative decoding and 262K
context. Node A serves the API on port 8015 as
`Swift-1.5-Qwen3.8-Flash-Next-NVFP4-TP4`, with no API key. Status:
Experimental. On installer image `dev-20261004-kraken-cuda1342-nccl2323-status034`, an installation on one four-Spark ring
passed the functional checks and the correctness screen
([record](../../performance/records/images/dev-20261004-kraken-swift15-qwen38-flash-next-tp4-20261004.md));
decode by context and the coding peak are in the
[matrix record](../../performance/records/images/dev-20261004-kraken-matrix-20261004.md#swift15-qwen38-flash-next-tp4).

On the Spark connected to your network:

```bash
curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/install.sh | bash -s -- --profile swift15-qwen38-flash-next-tp4
```

[Install SparkRing](../../docs/operations/install.md) covers requirements,
the ring's ConnectX setting, logs and recovery.
[Compose files](../../docs/operations/compose-files.md) lists this profile's
per-rank Compose files.

## Settings

Swift keeps the architecture of Qwen3.8-Flash-Next, so this profile uses the
[qwen38-flash-next-qad-tp4](../qwen38-flash-next-qad-tp4/README.md) settings,
including the ring's NCCL and RoCEnante transport. It differs where this
checkpoint's format differs:

| Setting | This profile | `qwen38-flash-next-qad-tp4` |
|---|---|---|
| Quantization | ModelOpt NVFP4 routed experts, `--quantization modelopt_fp4`; every other weight BF16 | ModelOpt mixed precision, `--quantization modelopt_mixed` |
| PLE n-gram table | BF16 in GPU memory (`VLLM_PLE_CPU_OFFLOAD=0`), 23.8 GiB per rank | NVFP4 in GPU memory |
| MTP draft experts | BF16, on vLLM's unquantized MoE kernel | MXFP8, on the `humming` MoE backend |
| Checkpoint | 100 files, 173.7 GiB | 56 files, 102.6 GiB |
| Rendezvous port | 29780 | 29779 |

Each rank holds about 45 GiB of weights and 24 GiB of FP8 KV cache. The
[two-Spark profile](../swift15-qwen38-flash-next-tp2/README.md) explains why
two Sparks read the PLE table from disk instead, and lists the limitations
and license terms that apply to both profiles.
