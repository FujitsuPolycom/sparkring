# Prototype and maintained transport boundary

A directory named `experiments` does not establish whether deployment depends
on its contents. The migration audited imports, source manifests and build inputs.

| Existing material | Disposition and evidence |
|---|---|
| CX7 hardware-forwarded fabric planner and marker | Maintained source is `spark_transport/fabric/cx7_hairpin_diagonal`; managed deployment and LIL import it |
| GLM RoCEnante bundle and framework hooks | Maintained source is `integrations/vllm/rocenante`; compatibility exports preserve source-bound paths |
| `spark_transport/experiments/tiled_prefill` | Retained native substrate; CMake/native consumers and content-addressed release snapshots depend on its paths |
| Model/cache research beneath version-specific runtime trees | Retained with its local tests and evidence; no deployment recommendation is implied |

Do not route additional maintained Python imports through experimental paths.
Prototype work should name its hypothesis, runnable check and adoption/removal
criterion. Graduate a runtime dependency into its component before advertising
it as maintained. Do not combine distinct transport protocols based on matching
filenames, or relocate immutable snapshots to make the tree look smaller.

The native tiled-prefill path is an explicit compatibility exception, not an
invitation to add unrelated prototypes there. Its extraction requires a native
ABI/build-context migration and hardware validation separate from this layout.
