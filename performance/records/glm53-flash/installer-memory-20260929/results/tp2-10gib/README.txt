GLM-5.3-Flash NVFP4-Spark on two Sparks (installer profile glm53-flash-nvfp4-spark-tp2): Node A memory with
10 GiB of KV cache per Spark, 2,048-token pages, a 1,048,576-token context window and up to 8 images per request,
the profile in this record's commit. The deployment was installed with install.sh and then ran the acceptance
harness (functional checks, correctness screen, throughput matrix) before these steps.

Program: pair-memory-steps.sh, with image_load.py, video_load.py, mixed_load.py and guarded.py from programs/.
Every image and video step used media the deployment had not seen. Values are GiB unless a file says kB.

Files:
- phases.txt: start and end epoch (seconds) of each step.
- memavailable-node-a.txt, memavailable-node-1.txt: "epoch MemAvailable_kB" once a second on each Spark, from
  memsample.sh. The logs start during the acceptance harness's correctness screen and continue past the steps.
- text128k.txt, text128k.json: llm_decode_bench.py with 4, then 8 streams at 128K tokens of context.
- img8c1.txt, img8c2.txt, img8c4.txt: 1, 2 and 4 concurrent requests with 8 images each. The guard stopped the
  2- and 4-request steps when its one-second reading of Node A fell to 1.16 and 1.01 GiB, after 3 s and 6 s.
- mixed3.txt: 4 text streams at 64K tokens, two 3-image requests and one 16-frame video request at once.
- mixed8.txt: the same with 8-image requests. It ended after 15 s, when an installation of another
  configuration took this deployment down (the engine log shows a shutdown, not a failure); its lows are not
  used in the record.
