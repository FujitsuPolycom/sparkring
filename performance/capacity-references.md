# Recorded KV capacity references

These sizing estimates come from the
[repository profile table](https://github.com/FujitsuPolycom/sparkring/blob/c65a9981e2a69f821ac716f6f13484d99d23f4ea/README.md#profiles).
They are sizing references, not additional startup measurements.
The linked profile specifies its KV allocation and other runtime settings.

| Profile | Layout | Recorded KV tokens |
|---|---|---:|
| GLM-5.3-Flash NVFP4-Spark · MTP3 + SparkCache | TP4/DCP1 | 2.28M |
| DeepSeek-V4-Flash-0731 | TP4/DCP1 | 1M |
| DeepSeek-V4-Flash-0731 | TP2/DCP1 | 1M |

The capacity index marks these entries as approximate. Exact startup counts
can replace them when the matching logs are available. Do not substitute a
completed prompt length, client workload budget or GiB allocation for a token
pool count.

The [shared-image profile table](https://github.com/FujitsuPolycom/sparkring/blob/98f5787964013c3ad77c202842b23779fdae247a/README.md#profiles)
also recorded the cache-disabled MTP3 TP4/DCP1 profile at 2.3M tokens.
This is a sizing reference for its 24 GiB-per-rank layout, not an additional
measurement of the R33 cache-disabled image configuration.

The switched TP4/DCP1 profile uses the same NVFP4-Spark checkpoint, FP8 KV
format and 24 GiB KV allocation per rank. Its table uses the ring's 2.3M
sizing reference; this is an estimate, not a switched-startup measurement.
