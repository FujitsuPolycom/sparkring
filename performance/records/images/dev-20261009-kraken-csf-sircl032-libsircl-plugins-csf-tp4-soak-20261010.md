# GLM-5.3-Flash CSF TP4 soak on a ring of four

Status: **research-only; one 20-minute soak (66 rounds) of one installation; passed the soak rule below**.

## Conditions

- `sparkring install --profile glm53-flash-nvfp4-spark-tp4` (its default `csf` checkpoint) on ring A of the two rings of four (fabric `097062777e17…`), image lock `dev-20261009-kraken-csf-sircl032-libsircl-plugins`, source revision `6555b89b`, SIRCL ring sessions with NCCL off on the measured `cycle-4` tuning row; GPU SM clocks locked to 2,418 MHz. Known-answer requests (3 of 3) and `sparkring check` passed before the load and `sparkring check` again after it.
- Load: back-to-back rounds for 20 minutes (66 complete rounds; the load was then ended with SIGINT during the next round, whose requests already sent completed and whose unsent requests were cancelled; that round is not scored) of the acceptance screen's 32 requests (`performance/harnesses/acceptance/checks.round_items`: 24 short questions and 8 needle questions of about 5K prompt tokens, greedy, the profile's thinking-off fields, at most 300 tokens), alternately through 4 and 8 client threads; every reply scored as an error, a degenerate (a word repeated 8 times in a row) or a wrong answer. vLLM's metrics sampled every minute; each Spark's SM clock, temperature, power, clock-event reasons, MemAvailable and serving-container memory sampled every minute ([directory](dev-20261009-kraken-csf-sircl032-libsircl-plugins-csf-tp4-soak-20261010/)).
- Rule: no error, degenerate or wrong reply; no active clock-event reason; MemAvailable falls by less than 1 GiB on every Spark; every 5-minute window's engine steps per second within 5 % of their median.

## Result

| Measure | Result |
|---|---|
| Rounds / requests | 66 / 2112 (33 rounds at 4 streams, median 18.4 s; 33 at 8 streams, median 17.4 s) |
| Errors / degenerate / wrong | 0 / 0 / 0 |
| Engine steps per second, 5-minute windows | median 2.83, 2.74 to 2.90 (3.4 % largest deviation, 4 windows) |
| Prompt / output tokens per second, same windows | 2103-2179 / 22.0-23.1 |
| Round time, first 8 / last 8 rounds of each stream count | 4 streams 19.1 / 17.6 s; 8 streams 16.9 / 17.8 s |
| SM clock under load | 2385-2411 MHz; active clock-event reasons in 0 of 76 samples |
| Highest GPU temperature | 77 °C |
| MemAvailable, first to last sample per position | 0: 24.8 → 24.8 GiB; 1: 26.5 → 26.7 GiB; 2: 27.0 → 27.1 GiB; 3: 24.5 → 24.8 GiB |
| Serving-container memory, first to last sample | 0: 5.15 → 5.32 GiB; 1: 3.576 → 3.729 GiB; 2: 3.156 → 3.311 GiB; 3: 24.77 → 24.93 GiB |

## Conclusion

The installation served 2112 requests over 20 minutes of mostly-prefill load (about 2,156 prompt and 23 output tokens per second) with no error, degenerate or wrong reply and no clock event. MemAvailable changed by +0.00 to +0.30 GiB across the Sparks and serving-container memory grew 0.15-0.17 GiB; the engine step rate stayed within 3.4 % of its median. One 20-minute soak of one installation; it does not by itself qualify the profile for serving.
