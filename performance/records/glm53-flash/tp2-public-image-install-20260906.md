# Public-image installation and persistence checks

Status: research-only. Installation, bounded chat, still-image and persistent
restore checks passed; the blue-video recognition check failed.

## Conditions

Two DGX Spark GB10 nodes installed the public profile from repository commit
d215e4a47aeb3bd1b387a0ee3d3f663858eee6b7 and the image
`ghcr.io/fujitsupolycom/sparkring@sha256:d431841cd9adb75ec40777eeb435f7f72af5995018219fa3556d7b5c7f56c2d5`.
Image ID: 7c029aa6f74e46467e9903cad527ff584b3d53682f301cbd2151f046d55c316a.
The GLM-5.3 Flash NVFP4-Spark checkpoint was reused, not downloaded again;
its config and index hashes matched the previously recorded identity. The
RoCE fabric and 4 GiB guards were existing host prerequisites, not provisioned
from a bare OS by this test.

Both nodes used fresh repository checkouts and empty per-node JIT/SparkCache
directories. Container mounts were limited to read-only model weights and
writable cache data. No host attention module, native cache library, or
private serving script was mounted or used to launch the model. Settings:
TP2/DCP1, native adaptive MTP3, 5 GiB KV per rank, eight maximum sequences,
batch budget 8,192, prefill interval 2, context limit 524,288, image/video
prompt limits 1/1. The public launcher rendered and executed the container
commands; it did not replace existing containers automatically.

## Measurement

The public smoke client at revision `6095189` supplied eight chat codeword
requests, a generated blue PNG, and a 7,727-token long-prompt request. The
codeword scorer requires an exact final JSON object, allowing Markdown fences.
The long-prompt request was repeated unchanged after both model processes were
stopped and started with the same disk-cache mounts. Native restore logs and
cached-token usage were checked independently of answer correctness.

Video fixtures were eight-frame, one-second, 224x224 H.264 MP4 files generated
with FFmpeg's solid-color source. Blue used the documented smoke prompt;
additional red and green probes asked for the dominant frame color. The
fixture prompts therefore do not constitute a controlled color-only A/B test.
FFmpeg decoded the blue frame to mean RGB [0, 0, 253]; the runtime's own
Glm5NextVideoBackend decoded two selected frames to mean RGB [0, 0, 255]. Those
decoder observations were captured separately from model inference.

Raw records: [installation/cache/video](tp2-public-image-install-20260906.json),
[eight chat responses](tp2-public-image-text-20260906.json), and
[still-image response](tp2-public-image-image-20260906.json).

## Result

- Public-image startup completed with no private overlays and no guard trip.
- Eight exact chat codeword answers passed before and after engine restart.
- The blue still image was recognized correctly, with image-token usage.
- Cold long-prompt response: correct codeword, zero cached tokens, 8.87 seconds.
- Both nodes committed a 7,680-token, approximately 125.4 MiB cache entry.
- After restart, the identical request returned the correct codeword with
  7,680 cached tokens in 2.48 seconds. Worker restore logs recorded 322.8 ms
  and 225.0 ms. Each worker rediscovered one valid manifest.
- Blue-video recognition failed twice: the final answer was Black. Red and
  green video fixtures were recognized. Video tokens appeared in API usage.

The raw-completion codeword fixture at revision `e8a536a` failed three of eight
full codeword checks and produced repetitive continuations. It was not a
valid chat instruction-following gate: raw completions do not apply the chat
template, and substring scoring could accept prompt echoes. Both the failed
results and the corrected chat-fixture results are retained. No inference
runtime change was made to obtain the chat-fixture pass.

## Conclusion

The public quickstart can instantiate the image with reused model weights,
fresh cache data, and the documented host prerequisites. Disk-backed native
restoration is established for the specified prompt across a process restart.
This is not a full clean-OS/network provisioning test. Video input decoding
works, but blue-video recognition is not qualified; its cause downstream of
initial frame decoding remains unresolved.

## Limitations

No throughput sweep, full-context request, tool-use suite, or general video
accuracy evaluation is established. Synthetic filler can elicit repetitive
continuations after a correct codeword. The standalone long-prompt client
reports cache_restore_verified=false because it cannot itself prove worker
restoration; the combined cache conclusion uses the separate worker logs and
restart boundary. The operational production designation is separate from
this bounded research-only qualification and requires acceptance or resolution
of the video failure.
