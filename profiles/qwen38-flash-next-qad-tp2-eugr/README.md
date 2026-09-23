# Qwen Flash-Next QAD TP2 on the eugr ARM64 foundation

Status: **research-only opt-in prerelease**. Follow the
[external Qwen deployment guide](../../docs/operations/external-qwen.md) with
`PROFILE=profiles/qwen38-flash-next-qad-tp2-eugr/config.json`.

The [configuration](config.json) retains TP2/DCP1, context 262144, 16 sequences,
batch 8192, MTP3, capture ceiling 64 and 24 GiB FP8 KV per rank. HC projection
sharding is enabled; HC token-row ownership is off. SparkCache uses a 16 GiB
disk budget, two 1.5 GiB capture slots and a 256 MiB restore budget per rank.

The [release qualification record](../../runtime/releases/shared-2026.09.4-rc.4/qualification.json)
distinguishes bounded text and restart checks from unqualified full-context,
sustained-load, arbitrary-media and performance behavior. The status endpoint
is enabled on both ranks. Stable recipes remain available for rollback.
