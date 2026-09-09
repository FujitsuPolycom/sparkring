These are complete, unmodified source files from
[`local-inference-lab/vllm` at `e2666d9a65f41fc376607531453cbd57c4c71016`](https://github.com/local-inference-lab/vllm/tree/e2666d9a65f41fc376607531453cbd57c4c71016).
[sources.json](sources.json) identifies each upstream path and records SHA-256
for its compressed and decompressed bytes. Gzip timestamps are zero. The
upstream [Apache-2.0 license](LICENSE) and source notices are retained.

The Responses probe verifies these hashes, applies the content-addressed
runtime patch to the complete Responses protocol, and checks its result hash.
The complete `OpenAIBaseModel` class is extracted from the engine protocol.
The request, response, SDK event subclasses, and serializers execute unchanged.
Only rendering, sampling, and Harmony interfaces are replaced for this CPU
probe; it does not test those interfaces or serve an HTTP request.
