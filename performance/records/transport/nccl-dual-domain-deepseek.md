# Dual-domain NCCL measurements on four-Spark DeepSeek runtimes

Status: **contributor-reported measurements**, separate from SparkRing's image
qualification. The reports identify source, library hashes and configurations;
their raw cluster receipts and complete image digests are not included here.

## Transport measurements

[sethforprivacy's transport report](https://github.com/FujitsuPolycom/sparkring/pull/248#issuecomment-5654125132)
uses four GB10 Sparks in a direct cycle and the NCCL revision and cumulative
patch in the [build manifest](../../../spark_transport/nccl/dual-pci-domain.json).
The contributor rebuilt with CUDA 13.0; the manifest's measured binary used
CUDA 13.3 and retains its original identity.

Seven sequential arms varied the library, HCA selection and two routing flags
together. Changing BF16 payloads passed across three communicator generations.
With both domains and flags enabled, 84/168 MiB reductions took 6.564/12.270 ms,
versus 13.354/25.569 ms for the repeated primary-domain control. Counters showed
traffic on both domains; merely enumerating four HCAs did not enable that use.
Small-message results and baseline drift do not establish a decode benefit.

## Serving measurements and library compatibility

The [serving report](https://github.com/FujitsuPolycom/sparkring/pull/248#issuecomment-5656543042)
describes separate ordered A/B/B/A comparisons on two four-Spark systems:

| Runtime and model | Fixed configuration | Reported prefill change |
|---|---|---|
| SGLang, DeepSeek-V4.1-Flash | TP4, DSpark five, 655360 context, 1499904 KV, 4096 prefill chunk, eight requests | 5.43–6.82% |
| vLLM, DeepSeek-V4-Flash-Vision-Exp | TP4, DSpark five, 1M context, 6291943 KV, 12288 batch, 48 sequences | 4.98–6.78% |

Decode differences did not establish a benefit. These are workload-specific,
ordered sessions; the library, HCA selection and routing flags form one treatment.
The reported library SHA256 values are:

- SGLang: `0553d4c74b9488224b3a9cb109c7fba5e4a4d663aed2dcdd4d2a42dbad8885d7`.
- vLLM: `8f34140608ccc72c8029db95f80b463876368075156cad03e61278ad6ffb4886`.

The SGLang-built library required `GLIBC_2.38` and could not load in the vLLM
image's 2.35 userspace. Rebuilding inside that target image resolved the loader
mismatch. The report also found automatic IPv6 link-local configuration changing
secondary-port GID slots; persistent static settings restored the intended mapping.

## Adoption boundary

Use the [runtime-specific library selection](../../../spark_transport/nccl/DUAL_PCI_DOMAIN.md#runtime-library-selection)
and the chosen profile's source/image checks. These observations do not promote
the SGLang adapter to a dual-domain configuration or replace any release hash.

A separate [four-versus-eight-channel report](https://github.com/FujitsuPolycom/sparkring/issues/193#issuecomment-5670099603)
found no clear DeepSeek serving benefit and increased head-node shared memory
from 0.48 to 0.62 GiB. Existing four-channel defaults remain unchanged. Per-size
tuning and Engram staging proposals require their own implementation and evidence.
