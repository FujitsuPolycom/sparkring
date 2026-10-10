# vLLM shared-memory reader window on two Sparks, 2026-09-30

Status: **research-only**. One pair, one profile.

## Question

vLLM's shared-memory message queue readers (`SpinCondition.wait()` in
`vllm/distributed/device_communicators/shm_broadcast.py`) poll for one second
after each read before they sleep until notified. How much CPU does that
window cost while a model decodes, and what does a 2 ms window
(`sparkring install --save-cpu`, FujitsuPolycom/sparkring#189) cost in decode
speed?

## Conditions

- Two directly cabled DGX Sparks (spark-b rank 0, spark-a rank 1),
  profile `qwen38-flash-next-tp2`, checkpoint `qad-step5500-ple1000`.
- Image `dev-20260930-spinwait-cuda1342-nccl2323-status033` (image
  `sha256:fcb20b0ce839`), derived from
  `dev-20260928-plainstatus-cuda1342-nccl2323-status033` by
  `runtime/images/derive_spin_wait.py`. Two deployments installed by
  `sparkring install` from SparkRing revision `8c8423d9`: one with
  `--save-cpu` (`SPARKRING_SHM_BUSY_LOOP_S=0.002` in both ranks' containers,
  so a 2 ms window), one without (the variable unset, so vLLM's one-second
  window). Nothing else differs between them.
- CPU: the busy time of every process in rank 0's model container, from
  `/proc/<pid>/stat` over 5 s, as a percentage of one core, sampled 8 s into
  one request generating 1,500 tokens with end of sequence ignored. Idle: no
  requests.
- Decode speed: llm-inference-bench 0.6.2 `llm_decode_bench.py`,
  `--concurrency 1,8 --contexts 0 --duration 20 --decode-warmup-seconds 5
  --max-tokens 2048 --temperature 1.0 --token-targeting exact`, three runs
  per deployment. Steps per second is tokens per second divided by the MTP
  accept length: engine forward passes per second, which does not depend on
  how many draft tokens a run happens to accept.

## Results

| Window | Rank 0 CPU, decoding | Rank 0 CPU, idle |
|---|---|---|
| 1 s (no `--save-cpu`) | 306% (EngineCore 101%, Worker_TP0 203%) | 5% |
| 2 ms (`--save-cpu`) | 136% (Worker_TP0 125%; EngineCore under 10%) | 6% |

| Window | 1 request: steps/s | 1 request: tok/s (accept length) | 8 requests: steps/s | 8 requests: tok/s (accept length) |
|---|---|---|---|---|
| 1 s | 24.0 – 24.2 | 49.2 – 57.2 (2.04 – 2.36) | 86.0 – 88.5 | 203.1 – 205.5 (2.32 – 2.37) |
| 2 ms | 23.8 – 23.9 | 50.1 – 57.6 (2.10 – 2.42) | 83.8 – 86.6 | 203.9 – 208.8 (2.41 – 2.46) |

Rank 1's worker receives its inputs over a network socket; its container
used 113–122% of one core while decoding with either window.

## Conclusion

On rank 0, the 2 ms window freed about 1.7 cores while the model decoded:
the engine process and the local worker sleep between decode steps instead
of polling. Each decode step then waits for a sleeping reader to wake, which
lowered steps per second by about 1% with one request and about 2% with
eight. Tokens per second at temperature 1.0 varied by up to 16% between runs
of the same deployment, with the accept length, so single requests timed
without the accept length cannot resolve a difference of this size. With no
requests, both windows idle at the same CPU use, because the readers sleep
one second after their last read in either case. Other profiles and
four-Spark rings were not measured.
