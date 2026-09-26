# GLM-5.3-Flash DCP1 without SparkCache

Checkpoint: [`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark), Local Inference Lab's NVFP4 Spark checkpoint of Z.ai's GLM-5.3-Flash.

Status: **Experimental**. Use the [retained R37 four-Spark procedure](https://github.com/FujitsuPolycom/sparkring/blob/5b28d768b37b21f5c97d910887e07144fcf251ef/profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md#3-discover-and-plan-dcp1)
and select `tp4-dcp1`. Use that document's repository revision, R37 image receipt
and checkpoint selection. DCP1 is the default; DCP4 is an alternative.

Keep the R37 image receipt when choosing the cache-disabled selection. This is
Experimental; the R37 hardware record covers cache-on DCP1, not this selection.

The [catalog profile](profile.json) retains its original release identity and
evidence scope. The retained quickstart selects R37 explicitly.

For the bounded cache-enabled deployment, use the
[2026.09.3 quickstart](../glm53-flash-spark-tp4-dcp1-sparkcache/README.md).
Its native-image receipt is not interchangeable with the R37 procedure's receipt.
