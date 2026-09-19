# Qwen sparse-attention graph padding

Status: implemented. The source transformation in
[`qwen_qsa_padding.py`](qwen_qsa_padding.py) clears inactive request-index storage
in the Qwen sparse-attention metadata builder. It requires an image build;
existing images and running processes are unchanged.

CUDA graphs retain views into persistent metadata buffers. A graph captured with
28 token rows can read that view when serving a 26-token request. Refreshing only
the 26 live rows leaves two request IDs from an earlier batch. Those IDs can make
padding appear to belong to a real request; sparse-attention validation then
rejects the request's metadata and produces nonfinite scores.

The correction writes the inactive sentinel, `-1`, into the builder's unused
buffer capacity. It preserves live mappings, mappings shared by other builders,
and the existing handling of padding within the live metadata view. CUDA graphs,
speculation, communication routing and attention calculations remain enabled.
The transformation rejects unfamiliar source structure and duplicate application.

The CPU regression checks retain a larger tensor view, refresh a shorter batch,
and verify both inactive sentinels and unchanged live/shared mappings. Image-build
checks execute the selected vLLM builder method rather than a replacement
implementation. GPU serving qualification additionally requires short requests
before and after longer text and multimodal requests, with finite token scores.

The correction does not change cache geometry or checkpoint identities. Images
with changed vLLM sources still require their own source-bound SparkCache
compatibility contract; release profiles must select that contract explicitly.
