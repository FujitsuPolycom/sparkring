# NVIDIA GLM-5.3-Flash NVFP4 serving evidence

Status: **contributor-reported hardware results**. These observations describe
the image and configuration below, not the R37 host adaptation.

[sethforprivacy's report](https://github.com/FujitsuPolycom/sparkring/issues/257#issuecomment-5620063031)
and [PR #258](https://github.com/FujitsuPolycom/sparkring/pull/258) identify:

- Four GB10 Sparks in a ring, TP4/DCP4 and MTP3.
- NVIDIA checkpoint revision `423acf37583782c51c142d145aef733d72943d93`.
- Image config ID `sha256:6921a6c163ea40b603e19a0332330efe3dbccbf4dce9f6cbbf6b756c9231835a`.
- `modelopt` quantization, `safetensors` loading, BF16 MTP exclusions and a
  1500-second startup budget. Full-shard staging with `fastsafetensors`
  exhausted memory in the reported configuration.

| Conditions | Reported result |
|---|---|
| 24 GiB KV per rank | 50.11 GiB weights per rank; 3,761,495 KV tokens at DCP4 |
| Correctness and API checks | Eight corruption probes and six exact-signature checks passed; image/video requests and authentication checks passed |
| Matched BF16-teacher comparison | 24 windows, 49,128 positions; KLD 0.0508 nats and top-1 agreement 0.930; differences from NVFP4-Spark were within the reported repeat band |
| One-hour concurrent workload | Three workers, 100K–300K prompts, SparkCache capture; zero reported failures or preemptions and idle KV returned to zero |

The evidence supports the contributor's checkpoint/loader combination under
those conditions. It does not establish R37 Docker/Compose serving, transfer
benchmark results to another image, or replace exact-image cache-restore checks.
The [target guide](../../../profiles/glm53-nvidia-nvfp4.md) defines the maintained
selection and its remaining qualification boundary.
