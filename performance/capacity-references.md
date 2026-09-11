# Recorded KV capacity references

These approximate figures were already published in the
[repository profile table](https://github.com/FujitsuPolycom/sparkring/blob/c65a9981e2a69f821ac716f6f13484d99d23f4ea/README.md#profiles).
They are retained sizing references, not additional startup measurements.
The linked profile specifies its KV allocation and other runtime settings.

| Profile | Layout | Recorded KV tokens |
|---|---|---:|
| GLM-5.3 Flash NVFP4-Spark · native MTP3 + SparkCache | TP4/DCP1 | ~2.28M |
| DeepSeek-V4-Flash-0731 | TP4/DCP1 | ~1M |
| DeepSeek-V4-Flash-0731 | TP2/DCP1 | ~1M |

The capacity index marks these entries as approximate. Exact startup counts
can replace them when the matching logs are available. Do not substitute a
completed prompt length, client workload budget or GiB allocation for a token
pool count.
