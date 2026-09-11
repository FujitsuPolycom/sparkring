# Preserved DFlash2 image transport overlay

This archive contains the 14 Python files and content manifest bound by the
[public SIRCL build receipt](../../glm53-flash-jj-r8-gb10/sircl-public-build-receipt.json).
The source tree and manifest digests match that receipt. The archive preserves
the original source bytes; it is not generated from changing maintained adapters.

The retained DFlash2 image builder extracts and verifies these inputs alongside
its receipt-bound native library. Maintained adapters live in `integrations/vllm/`;
changes there do not silently replace this historical build input or inherit its
qualification. Source code in the archive is covered by the repository license.
