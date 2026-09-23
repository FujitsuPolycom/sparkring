# Qwen Flash-Next QAD TP4 on the eugr ARM64 foundation

Status: **research-only opt-in prerelease**. Follow the
[external Qwen deployment guide](../../docs/operations/external-qwen.md) with
`PROFILE=profiles/qwen38-flash-next-qad-tp4-eugr/config.json`.

The [configuration](config.json) retains TP4/DCP1, context 262144, 16 sequences,
batch 8192, MTP3, capture ceiling 64 and 24 GiB FP8 KV per rank. It selects
`VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0` and
`VLLM_QWEN3_8_HC_PREFILL_MODE=shard`, with `QWEN_DISPATCH_MODE=both`.
HC prefill keeps token-row ownership and direct NCCL collectives; eligible
generic collectives use the prepared RoCEnante transport.

SparkCache retains a 4 GiB disk budget, two 512 MiB capture slots and a 256 MiB
restore budget per rank. The status plugin is enabled on all four ranks.

The [release qualification record](../../runtime/releases/shared-2026.09.4-rc.4/qualification.json)
identifies completed checks and limitations. Earlier prototype performance
measurements do not establish throughput for this exact source composition.
Stable recipes remain available for rollback.
