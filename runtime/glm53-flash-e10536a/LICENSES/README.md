# GLM-5.3 runtime license boundary

The public runtime is a FujitsuPolycom community derivative. It is not an
official NVIDIA, vLLM, local-inference-lab, B12X, Inco AI, or Z.AI release.

The distributed image combines components governed by different terms:

| Component | License or terms | Included in image |
|---|---|---:|
| NVIDIA CUDA runtime parent | NVIDIA Deep Learning Container License | yes |
| local-inference-lab/vLLM | Apache-2.0 | yes |
| local-inference-lab/B12X | Apache-2.0 | yes |
| NVIDIA NCCL | Apache-2.0 and BSD-3-Clause portions identified by NCCL | yes |
| InstantTensor 0.1.9 | Apache-2.0; bundled third-party notices remain in its source archive | yes |
| SparkRing source and NCCL changes | Apache-2.0 | yes |
| SparkCache | Apache-2.0 | only in the SparkCache overlay |
| GLM-5.3 Flash NVFP4 target checkpoint | MIT at the pinned repository revision | no; mounted by operator |
| Inco AI GLM-5.3 Flash DFlash2 checkpoint | CC BY-NC-ND 4.0 for research and evaluation | no; mounted by operator |

The image retains the NVIDIA parent and adds the primary inference-runtime
functionality described by the runtime contract. Distribution must preserve
the NVIDIA license file already present in the parent and use terms at least as
protective as that license. The complete NVIDIA terms are published at
<https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf>.

The image copies the vLLM, B12X, and NCCL license files into
`/usr/share/licenses`. It also retains the exact InstantTensor source archive,
including its license and bundled third-party notices, under
`/usr/share/licenses/source-archives`.

The source-built NCCL binary uses the original SparkRing patch
`spark_transport/nccl/nccl-2.30.7-switchless-cycle.patch`. That patch does not
copy the `NCCL_SKIP_TREE_CONNECT` check from the unlicensed
`josephdrose/nccl-spark-switchless` repository. The compatibility patches
`spark_transport/nccl/nccl-2.29.7-skip-tree-pat.patch` and
`spark_transport/nccl/nccl-2.30.7-skip-tree-pat.patch` remain credited in the
repository notices but are not inputs to this public image builder.
