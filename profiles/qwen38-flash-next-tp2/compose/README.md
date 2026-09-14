# Generated Qwen TP2 Compose

Status: **implemented; Compose serving is unqualified**.

[Rank 0](compose.rank0.yaml) and [rank 1](compose.rank1.yaml) are generated from
[config.json](../config.json) and the [public site example](site.example.yaml).
They show effective settings using documentation addresses and example host
paths. Serving defaults remain in the profile.

Use the [Compose deployment guide](../../../docs/operations/compose.md) to render
a private deployment and coordinate both hosts. Complete the
[Qwen quickstart](../README.md) prerequisites first.

For persistent caching, select `qwen38-flash-next-tp2-sparkcache`; its
[generated examples](../../qwen38-flash-next-tp2-sparkcache/compose/README.md)
select the registered extension image. Private site files own host settings;
`.env` is not used. Edit the profile/site input and regenerate instead of editing
these exports.

## Host preparation

Follow the [host prerequisites](../../../docs/operations/compose.md#prepare-the-hosts).

## Configure and render

Use the [rendering procedure](../../../docs/operations/compose.md#render-and-inspect).

## Scope of translation

The [configuration ownership](../../../docs/operations/compose.md#configuration-ownership)
and [coordinator behavior](../../../docs/operations/compose.md#check-and-coordinate-hosts)
define the supported deployment contract.
