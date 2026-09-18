# Sparse-attention serving candidate: 2026.09.0-rc.2

Availability: **local build; not a published image selection**. Public profile
image defaults are unchanged.

The [source manifest](sources/manifest.json) reconstructs the vLLM and B12X
inputs for the ARM64/GB10 serving candidate. It retains the Kraken-based vLLM
composition and incorporates the sparse-attention selection implementation from
[B12X pull request 394](https://github.com/local-inference-lab/b12x/pull/394),
revision `7530ea27d7d923dbc71a76a356922bd8b6b1611a`.

| Artifact | Docker configuration ID |
|---|---|
| Runtime build | `sha256:556f6c882e14afdc51158ab20837ab704ed2a6bfcc5c9e9ef478c4773916c133` |
| Labels-only serving image | `sha256:5ea26fe19e7cc17f68c9adac0501c1639e87882614fc9e102bdd35c959b94867` |

These configuration IDs are not public registry manifest digests and cannot be
substituted into a GHCR pull command. The source manifest is not a publication
receipt. The [published RC1 candidate](../shared-2026.09.0-rc.1/README.md) retains
its own immutable inputs and narrower qualification record.

Hardware measurements are retained locally and are not published here.
This source description does not promote any profile or establish serving
qualification. A separately developed request-salt isolation fix is not included
in this image.
